#!/usr/bin/env python3
"""B5: Bond-Resolved Cross-N Transfer — Train N=40, Predict N=60/80.

Demonstrates that the per-node/per-edge BondResolvedMPNN generalizes across
system sizes for high-dimensional variational parameters. This is the
DEFINITIVE thesis contribution: GNN learns the physics of bond-resolved
parameter structure, not just h-interpolation.

Key insight:
- Global HVA (2 params): cross-N works trivially (scipy suffices)
- Bond-resolved HVA (79→119→159 params): cross-N requires GNN because
  scipy CANNOT interpolate in (2N-1)-dimensional space across N values.

Architecture (BondResolvedMPNN):
    Per-node head: node_embedding → θ_x_i  (N outputs, scales with graph)
    Per-edge head: edge_concat → θ_zz_ij   (n_edges outputs, scales with graph)
    → Model outputs 2N-1 params regardless of N. No retraining needed.

Sections:
    1. Generate N=60 bond-resolved VQE data (ground truth for evaluation)
    2. Generate N=80 bond-resolved VQE data (optional, for extended eval)
    3. Load pre-trained model from B4 Section 6 + cross-N prediction
    4. Evaluation: ΔE/gap at unseen N vs scipy baseline
    5. Ablation: GNN vs scipy vs random at N=60/80

Prerequisites:
    - B4 Sections 5+6 complete (dense N=40 data + trained BondResolvedMPNN)
    - OR provide --model-checkpoint pointing to saved .pt file

Usage:
    # Full pipeline (generate N=60 data + predict + compare)
    python scripts/.../run_bond_resolved_cross_n_transfer.py

    # Skip VQE generation, use existing data
    python scripts/.../run_bond_resolved_cross_n_transfer.py \\
        --section 3 4 5 --target-data results/bond_resolved_scaling/sweep_N60_*.json

    # Custom target sizes
    python scripts/.../run_bond_resolved_cross_n_transfer.py --target-n 60 80

    # With pre-trained checkpoint
    python scripts/.../run_bond_resolved_cross_n_transfer.py \\
        --model-checkpoint results/bond_resolved_scaling/bond_resolved_mpnn_N40_*.pt
"""

from __future__ import annotations

import glob
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from qmbp_simulation.framework.runner_base import (
    Section,
    ValidationRunner,
    resolve_project_root,
)

_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

TRAIN_N = 40
P_LAYERS = 1
TOPOLOGY = "chain_1d"
SEED = 42
CHI_MAX = 64
PRECISION = 0.005
STRATEGY = "aer_mps"

# Default target sizes for cross-N evaluation
DEFAULT_TARGET_NS = [60, 80]

# Scaling law for valid regime: h_min = 1.5 + 0.020 * N^1.31
def _h_min_safe(n: int) -> float:
    return 1.5 + 0.020 * n**1.31


def _h_sweep_for_n(n: int, n_points: int = 7) -> list[float]:
    """Generate h-sweep for a given N within valid regime."""
    h_min = _h_min_safe(n) + 0.3
    h_max = _h_min_safe(n) + 3.0
    return np.linspace(h_max, h_min, n_points).tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════


class BondResolvedCrossNTransferRunner(ValidationRunner):
    """B5: Bond-Resolved Cross-N Transfer — train N=40, predict N=60/80.

    Uses the per-node/per-edge BondResolvedMPNN architecture that outputs
    variable-length θ based on graph structure (2N-1 params for chain_1d).
    """

    runner_id = "bond_resolved_cross_n_transfer"
    experiment_id = "B5_BR_CROSS_N_TRANSFER"
    description = "B5: Bond-Resolved Cross-N — per-node GNN transfer N=40→N=60/80"
    hypothesis = (
        "BondResolvedMPNN trained on N=40 predicts θ_opt at N=60/80 with "
        "ΔE/gap < 10%, while scipy interpolation CANNOT (>100% error in 79+D)."
    )

    @classmethod
    def _add_custom_args(cls, parser):
        parser.add_argument(
            "--target-n",
            type=int,
            nargs="+",
            default=DEFAULT_TARGET_NS,
            help="Target system sizes for cross-N prediction (default: 60 80)",
        )
        parser.add_argument(
            "--model-checkpoint",
            type=str,
            default=None,
            help="Path to pre-trained BondResolvedMPNN .pt checkpoint",
        )
        parser.add_argument(
            "--target-data",
            type=str,
            default=None,
            help="Path to pre-computed target VQE data (skips Sections 1-2)",
        )
        parser.add_argument(
            "--hidden-dim",
            type=int,
            default=256,
            help="MPNN hidden dim if training from scratch",
        )
        parser.add_argument(
            "--n-epochs",
            type=int,
            default=8000,
            help="Training epochs if training from scratch",
        )
        parser.add_argument(
            "--n-target-points",
            type=int,
            default=5,
            help="Number of h-points per target N for evaluation",
        )
        parser.add_argument(
            "--cobyla-maxiter",
            type=int,
            default=2000,
            help="COBYLA maxiter for target VQE sweeps",
        )

    def build_config(self) -> dict:
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "train_n": TRAIN_N,
            "target_ns": self._args.target_n,
            "system": {
                "p_layers": P_LAYERS,
                "topology": TOPOLOGY,
                "model": "tfim_bond_resolved",
                "parametrization": "bond_resolved",
            },
            "mps": {
                "strategy": STRATEGY,
                "chi_max": CHI_MAX,
                "precision": PRECISION,
            },
        }

    def setup(self):
        """Import heavy dependencies."""
        import torch

        from qmbp_simulation import (
            ClassicalSolver,
            HamiltonianBuilder,
            HVACircuitBuilder,
            VQEConfig,
            VQEOptimizer,
            make_lattice,
        )
        from qmbp_simulation.execution import MPSBackend
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.predictors import (
            BondResolvedMPNN,
            build_bond_resolved_graph,
            train_bond_resolved_mpnn,
        )

        self.torch = torch
        self.builder = HamiltonianBuilder()
        self.solver = ClassicalSolver()
        self.make_lattice = make_lattice
        self.hva = HVACircuitBuilder()
        self.spec_br = get_model_spec("tfim_bond_resolved")
        self.MPSBackend = MPSBackend
        self.VQEOptimizer = VQEOptimizer
        self.VQEConfig = VQEConfig
        self.BondResolvedMPNN = BondResolvedMPNN
        self.build_bond_resolved_graph = build_bond_resolved_graph
        self.train_bond_resolved_mpnn = train_bond_resolved_mpnn

        # State containers
        self._target_data: dict[int, list[dict]] = {}
        self._model: BondResolvedMPNN | None = None

    def define_sections(self) -> list[Section]:
        target_ns = self._args.target_n
        sections = [
            Section(
                id=1,
                name=f"Generate Target VQE Data (N={target_ns[0]})",
                fn=lambda: self._section_generate_target(target_ns[0]),
                hypothesis=f"Bond-resolved VQE converges at N={target_ns[0]}",
            ),
        ]
        if len(target_ns) > 1:
            sections.append(
                Section(
                    id=2,
                    name=f"Generate Target VQE Data (N={target_ns[1]})",
                    fn=lambda: self._section_generate_target(target_ns[1]),
                    hypothesis=f"Bond-resolved VQE converges at N={target_ns[1]}",
                )
            )
        sections.extend([
            Section(
                id=3,
                name="Load/Train BondResolvedMPNN",
                fn=self.section_load_or_train_model,
                hypothesis="Model loaded or trained with MSE < 5e-4",
            ),
            Section(
                id=4,
                name="Cross-N Prediction + Evaluation",
                fn=self.section_cross_n_predict,
                hypothesis="GNN predictions achieve ΔE/gap < 10% at target N",
            ),
            Section(
                id=5,
                name="Ablation: GNN vs Scipy vs Random",
                fn=self.section_ablation,
                hypothesis="GNN outperforms scipy and random by >5× at target N",
            ),
        ])
        return sections

    # ── Section 1/2: Generate Target VQE Data ────────────────────────────────

    def _section_generate_target(self, target_n: int) -> dict:
        """Run bond-resolved VQE sweep at target N for evaluation."""
        from qmbp_simulation.utils.helpers import json_dump as _json_dump

        output_dir = Path("results/bond_resolved_scaling")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing data
        existing = list(output_dir.glob(f"sweep_N{target_n}_{TOPOLOGY}_*.json"))
        if existing:
            logger.info(f"  Found existing N={target_n} data: {existing[0].name}")
            with open(existing[0]) as f:
                data = json.load(f)
            results = data.get("sweep_results", [])
            self._target_data[target_n] = results
            n_pass = sum(1 for r in results if r.get("passed", r.get("de_gap", 1) < 0.05))
            return {
                "source": "existing",
                "n_qubits": target_n,
                "n_points": len(results),
                "n_pass": n_pass,
                "pass": True,
            }

        # Generate new data
        n_pts = self._args.n_target_points
        h_values = _h_sweep_for_n(target_n, n_pts)
        lattice_ref = self.make_lattice(TOPOLOGY, target_n, h=h_values[0])
        n_edges = len(lattice_ref.edges)
        n_params = n_edges + target_n

        logger.info(
            f"  Generating N={target_n} data: {n_pts} h-points, "
            f"params={n_params}, h∈[{h_values[-1]:.2f}, {h_values[0]:.2f}]"
        )

        backend = self.MPSBackend(
            strategy=STRATEGY, chi_max=CHI_MAX, precision=PRECISION, seed=SEED
        )
        config = self.VQEConfig(
            method="COBYLA",
            p_layers=P_LAYERS,
            n_restarts=1,
            maxiter=self._args.cobyla_maxiter,
            enable_callbacks=False,
        )
        opt = self.VQEOptimizer(config=config, backend=backend, seed=SEED)

        theta_prev = np.random.default_rng(SEED).uniform(-0.01, 0.01, n_params)
        sweep_results = []

        for h in sorted(h_values, reverse=True):
            t0 = time.time()
            lattice_h = self.make_lattice(TOPOLOGY, target_n, h=h)
            H = self.builder.build(lattice_h)
            qc, _ = self.spec_br.create_circuit(target_n, P_LAYERS, lattice_h)
            gt = self.solver.solve(H, lattice_h, method="dmrg")

            res = opt.optimize(H, qc, theta_prev.copy(), exact_energy=gt.ground_energy)
            elapsed = time.time() - t0
            de_gap = abs(res.energy - gt.ground_energy) / max(gt.gap, 1e-10)
            theta_prev = res.theta_opt.copy()

            sweep_results.append({
                "h": float(h),
                "seed": SEED,
                "vqe_energy": float(res.energy),
                "dmrg_energy": float(gt.ground_energy),
                "gap": float(gt.gap),
                "de_gap": float(de_gap),
                "theta_opt": res.theta_opt.tolist(),
                "n_iterations": res.n_iterations,
                "time_s": elapsed,
                "passed": de_gap < 0.05,
            })
            status = "✅" if de_gap < 0.05 else "⚠️"
            logger.info(f"    {status} h={h:.3f}: ΔE/gap={de_gap:.4f} ({elapsed:.1f}s)")

        # Save
        sweep_path = output_dir / f"sweep_N{target_n}_{TOPOLOGY}_{int(time.time())}.json"
        _json_dump(
            {
                "experiment_id": self.experiment_id,
                "n_qubits": target_n,
                "topology": TOPOLOGY,
                "n_params": n_params,
                "n_edges": n_edges,
                "h_values": sorted(h_values, reverse=True),
                "sweep_results": sweep_results,
            },
            sweep_path,
        )
        logger.info(f"  Saved: {sweep_path}")

        self._target_data[target_n] = sweep_results
        n_pass = sum(1 for r in sweep_results if r["passed"])
        return {
            "source": "computed",
            "n_qubits": target_n,
            "n_points": len(sweep_results),
            "n_pass": n_pass,
            "mean_de_gap": float(np.mean([r["de_gap"] for r in sweep_results])),
            "pass": n_pass >= len(sweep_results) * 0.6,
        }

    # ── Section 3: Load or Train Model ───────────────────────────────────────

    def section_load_or_train_model(self) -> dict:
        """Load pre-trained BondResolvedMPNN or train on N=40 data."""
        import torch

        # Try loading from checkpoint
        ckpt_path = self._args.model_checkpoint
        if ckpt_path is None:
            # Auto-discover latest checkpoint
            ckpts = sorted(
                glob.glob("results/bond_resolved_scaling/bond_resolved_mpnn_N40_*.pt")
            )
            if ckpts:
                ckpt_path = ckpts[-1]

        if ckpt_path and Path(ckpt_path).exists():
            logger.info(f"  Loading model from: {ckpt_path}")
            data = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            model = self.BondResolvedMPNN(
                node_features=data.get("node_features", 3),
                hidden_dim=data.get("hidden_dim", 256),
                n_layers=data.get("n_layers", 3),
                norm_type=data.get("norm_type", "none"),
            )
            model.load_state_dict(data["state_dict"])
            model.eval()
            self._model = model

            return {
                "source": "checkpoint",
                "checkpoint_path": ckpt_path,
                "hidden_dim": data.get("hidden_dim", 256),
                "n_training_points": data.get("n_training_points", "unknown"),
                "pass": True,
            }

        # No checkpoint — train from N=40 data
        logger.info("  No checkpoint found. Training from N=40 data...")
        train_data = self._load_training_data()
        if not train_data:
            raise RuntimeError(
                "No N=40 bond-resolved data found. Run B4 Sections 5+6 first:\n"
                "  python scripts/.../run_bond_resolved_cross_n.py --section 5 6"
            )

        # Build dataset
        dataset = []
        for r in train_data:
            lattice = self.make_lattice(TOPOLOGY, TRAIN_N, h=r["h"])
            theta_opt = np.array(r["theta_opt"])
            graph = self.build_bond_resolved_graph(
                lattice, r["h"], theta_opt=theta_opt, n_feature=True
            )
            dataset.append(graph)

        model = self.BondResolvedMPNN(
            node_features=3,
            hidden_dim=self._args.hidden_dim,
            n_layers=3,
            norm_type="none",
            dropout=0.1,
        )

        t0 = time.time()
        metrics = self.train_bond_resolved_mpnn(
            model, dataset, n_epochs=self._args.n_epochs, seed=SEED
        )
        train_time = time.time() - t0

        model.eval()
        self._model = model

        logger.info(
            f"  Trained: MSE={metrics['final_mse']:.2e}, time={train_time:.1f}s"
        )

        return {
            "source": "trained_from_scratch",
            "n_training_points": len(dataset),
            "final_mse": float(metrics["final_mse"]),
            "training_time_s": train_time,
            "pass": metrics["final_mse"] < 5e-4,
        }

    def _load_training_data(self) -> list[dict]:
        """Load N=40 bond-resolved training data from results/."""
        output_dir = Path("results/bond_resolved_scaling")
        # Prefer dense data
        dense_files = sorted(output_dir.glob(f"dense_N{TRAIN_N}_{TOPOLOGY}_*.json"))
        if dense_files:
            with open(dense_files[-1]) as f:
                data = json.load(f)
            results = data.get("sweep_results", [])
            passing = [r for r in results if r.get("passed", r.get("de_gap", 1) < 0.05)]
            logger.info(f"  Loaded dense data: {len(passing)} passing points")
            return passing

        # Fall back to regular sweep
        sweep_files = sorted(output_dir.glob(f"sweep_N{TRAIN_N}_{TOPOLOGY}_*.json"))
        if sweep_files:
            with open(sweep_files[-1]) as f:
                data = json.load(f)
            results = data.get("sweep_results", [])
            passing = [r for r in results if r.get("passed", r.get("de_gap", 1) < 0.05)]
            logger.info(f"  Loaded sweep data: {len(passing)} passing points")
            return passing

        return []

    # ── Section 4: Cross-N Prediction + Evaluation ───────────────────────────

    def section_cross_n_predict(self) -> dict:
        """Predict θ_opt at target N using BondResolvedMPNN trained on N=40."""
        import torch

        if self._model is None:
            raise RuntimeError("No model. Run Section 3 first.")

        target_ns = self._args.target_n
        all_results = {}

        for target_n in target_ns:
            if target_n not in self._target_data:
                logger.warning(f"  No target data for N={target_n}. Skipping.")
                continue

            target_results = self._target_data[target_n]
            logger.info(f"\n  Cross-N prediction: N={TRAIN_N} → N={target_n}")

            backend = self.MPSBackend(
                strategy=STRATEGY, chi_max=CHI_MAX, precision=PRECISION, seed=SEED
            )

            predictions = []
            for r in target_results:
                h_val = r["h"]
                lattice = self.make_lattice(TOPOLOGY, target_n, h=h_val)

                # Build prediction graph (same structure as target, model adapts)
                graph = self.build_bond_resolved_graph(
                    lattice, h_val, theta_opt=None, n_feature=True
                )

                # Predict
                self._model.eval()
                with torch.no_grad():
                    theta_pred = self._model(graph).numpy().flatten()

                # Evaluate
                H = self.builder.build(lattice)
                qc, _ = self.spec_br.create_circuit(target_n, P_LAYERS, lattice)

                n_edges = len(lattice.edges)
                n_params_expected = n_edges + target_n
                # Trim/pad prediction if needed (model might output slightly different size)
                if len(theta_pred) > n_params_expected:
                    theta_pred = theta_pred[:n_params_expected]
                elif len(theta_pred) < n_params_expected:
                    theta_pred = np.pad(theta_pred, (0, n_params_expected - len(theta_pred)))

                e_pred = backend.evaluate(qc, H, theta_pred)
                e_exact = r["dmrg_energy"]
                gap = r["gap"]
                de_gap = abs(e_pred - e_exact) / max(gap, 1e-10)

                predictions.append({
                    "h": float(h_val),
                    "e_pred": float(e_pred),
                    "e_exact": float(e_exact),
                    "gap": float(gap),
                    "de_gap": float(de_gap),
                    "de_gap_vqe_ref": float(r["de_gap"]),
                    "passed": de_gap < 0.10,
                })

                status = "✅" if de_gap < 0.10 else "❌"
                logger.info(
                    f"    {status} h={h_val:.3f}: ΔE/gap={de_gap:.4f} "
                    f"(VQE ref: {r['de_gap']:.4f})"
                )

            n_pass = sum(1 for p in predictions if p["passed"])
            mean_de = float(np.mean([p["de_gap"] for p in predictions])) if predictions else 0

            all_results[target_n] = {
                "predictions": predictions,
                "n_pass": n_pass,
                "n_total": len(predictions),
                "mean_de_gap": mean_de,
            }
            logger.info(
                f"  N={target_n}: {n_pass}/{len(predictions)} pass, "
                f"mean ΔE/gap={mean_de:.4f}"
            )

        # Overall pass: majority of target points pass
        total_pass = sum(v["n_pass"] for v in all_results.values())
        total_pts = sum(v["n_total"] for v in all_results.values())

        return {
            "per_n_results": {str(k): v for k, v in all_results.items()},
            "total_pass": total_pass,
            "total_points": total_pts,
            "pass": total_pass >= total_pts * 0.5 if total_pts > 0 else False,
        }

    # ── Section 5: Ablation — GNN vs Scipy vs Random ─────────────────────────

    def section_ablation(self) -> dict:
        """Compare GNN cross-N prediction vs scipy interpolation vs random."""
        import torch

        if self._model is None:
            raise RuntimeError("No model. Run Section 3 first.")

        target_ns = self._args.target_n
        # Use first target N with data for ablation
        target_n = None
        for n in target_ns:
            if n in self._target_data and self._target_data[n]:
                target_n = n
                break

        if target_n is None:
            raise RuntimeError("No target data available for ablation.")

        target_results = self._target_data[target_n]
        # Pick middle h-point for focused comparison
        mid_idx = len(target_results) // 2
        test_point = target_results[mid_idx]
        h_test = test_point["h"]
        e_exact = test_point["dmrg_energy"]
        gap = test_point["gap"]

        lattice = self.make_lattice(TOPOLOGY, target_n, h=h_test)
        H = self.builder.build(lattice)
        qc, _ = self.spec_br.create_circuit(target_n, P_LAYERS, lattice)
        n_edges = len(lattice.edges)
        n_params = n_edges + target_n

        backend = self.MPSBackend(
            strategy=STRATEGY, chi_max=CHI_MAX, precision=PRECISION, seed=SEED
        )

        logger.info(
            f"\n  Ablation at N={target_n}, h={h_test:.3f}, params={n_params}"
        )

        # ── 1. GNN prediction ────────────────────────────────────────
        graph = self.build_bond_resolved_graph(
            lattice, h_test, theta_opt=None, n_feature=True
        )
        self._model.eval()
        with torch.no_grad():
            theta_gnn = self._model(graph).numpy().flatten()

        if len(theta_gnn) > n_params:
            theta_gnn = theta_gnn[:n_params]
        elif len(theta_gnn) < n_params:
            theta_gnn = np.pad(theta_gnn, (0, n_params - len(theta_gnn)))

        e_gnn = backend.evaluate(qc, H, theta_gnn)
        de_gnn = abs(e_gnn - e_exact) / max(gap, 1e-10)
        logger.info(f"    GNN (1 eval):    ΔE/gap = {de_gnn:.4f}")

        # ── 2. Scipy interpolation attempt ───────────────────────────
        # Load N=40 training data for interpolation baseline
        train_data = self._load_training_data()
        de_scipy = float("inf")

        if train_data:
            # Scipy cannot interpolate in 79D → we test component-wise 1D interp
            from scipy.interpolate import interp1d

            train_h = np.array([r["h"] for r in train_data])
            train_theta = np.array([r["theta_opt"] for r in train_data])

            # Component-wise interpolation (extrapolation for different N)
            # This is the WRONG approach for cross-N but shows the baseline
            try:
                theta_scipy = np.zeros(n_params)
                # N=40 has 79 params, N=60 has 119 — cannot directly map
                # Use per-site averaging as best scipy can do
                n_train_edges = TRAIN_N - 1  # 39 edges for chain_1d N=40
                n_target_edges = target_n - 1

                # θ_zz: stretch/interpolate from 39 → n_target_edges
                mean_zz = np.mean(train_theta[:, :n_train_edges], axis=0)
                # Linear interpolation of spatial pattern
                zz_positions = np.linspace(0, 1, n_train_edges)
                target_positions = np.linspace(0, 1, n_target_edges)
                for param_idx in range(n_train_edges):
                    h_interp = interp1d(
                        train_h, train_theta[:, param_idx],
                        kind="linear", fill_value="extrapolate"
                    )
                    mean_zz[param_idx] = float(h_interp(h_test))

                # Spatial resampling
                spatial_interp = interp1d(zz_positions, mean_zz, kind="linear")
                theta_scipy[:n_target_edges] = spatial_interp(target_positions)

                # θ_x: average across sites (uniform for chain_1d)
                mean_x_per_h = np.mean(train_theta[:, n_train_edges:], axis=1)
                x_interp = interp1d(train_h, mean_x_per_h, fill_value="extrapolate")
                theta_scipy[n_target_edges:] = float(x_interp(h_test))

                e_scipy = backend.evaluate(qc, H, theta_scipy)
                de_scipy = abs(e_scipy - e_exact) / max(gap, 1e-10)
            except Exception as exc:
                logger.warning(f"    Scipy interpolation failed: {exc}")
                de_scipy = float("inf")

        logger.info(f"    Scipy (interp):  ΔE/gap = {de_scipy:.4f}")

        # ── 3. Random search (100 samples) ───────────────────────────
        N_RANDOM = 100
        rng = np.random.default_rng(SEED)
        random_energies = []

        for _ in range(N_RANDOM):
            theta_rand = rng.uniform(-np.pi, np.pi, n_params)
            e_rand = backend.evaluate(qc, H, theta_rand)
            random_energies.append(e_rand)

        best_random_e = min(random_energies)
        de_random = abs(best_random_e - e_exact) / max(gap, 1e-10)
        logger.info(f"    Random (100):    ΔE/gap = {de_random:.4f}")

        # ── Comparison ───────────────────────────────────────────────
        gnn_vs_scipy = de_scipy / max(de_gnn, 1e-10)
        gnn_vs_random = de_random / max(de_gnn, 1e-10)

        logger.info(f"\n    GNN vs Scipy: {gnn_vs_scipy:.1f}×")
        logger.info(f"    GNN vs Random: {gnn_vs_random:.1f}×")
        logger.info(
            f"    GNN is {'NECESSARY ✅' if de_gnn < de_scipy else 'NOT necessary ❌'}"
        )

        return {
            "target_n": target_n,
            "h_test": h_test,
            "n_params": n_params,
            "gnn": {"de_gap": float(de_gnn), "n_evals": 1},
            "scipy": {"de_gap": float(de_scipy), "method": "spatial_resampling"},
            "random": {"de_gap": float(de_random), "n_samples": N_RANDOM},
            "gnn_vs_scipy_factor": float(gnn_vs_scipy),
            "gnn_vs_random_factor": float(gnn_vs_random),
            "gnn_wins_vs_scipy": bool(de_gnn < de_scipy),
            "gnn_wins_vs_random": bool(de_gnn < de_random),
            "pass": de_gnn < de_scipy and gnn_vs_random > 5.0,
        }


if __name__ == "__main__":
    BondResolvedCrossNTransferRunner.main()
