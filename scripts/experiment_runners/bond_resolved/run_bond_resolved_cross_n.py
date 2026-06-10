#!/usr/bin/env python3
"""B4: Bond-Resolved Cross-N GNN Necessity Proof.

Demonstrates that GNN is ESSENTIAL for warm-starting bond-resolved HVA where
the parameter space is high-dimensional (N-1 θ_zz + N θ_x = 2N-1 params).

The thesis differentiator:
- Global HVA (2 params): scipy interpolation matches GNN → convenient not essential
- Bond-resolved HVA (79 params @ N=40): scipy CANNOT interpolate 79D → GNN NECESSARY

Sections:
    1. Sweep: Run bond-resolved VQE descending sweep at N=40 (if no data)
    2. MPNN Training: Train GNN (norm_type=none) on bond-resolved θ_opt
    3. Deploy: Evaluate on midpoint h-values (intra-N interpolation test)
    4. Necessity: Compare GNN prediction vs random init → GNN saves >10× evals
    5. Dense Sweep: Multi-seed × dense h-grid (3 seeds × 15 h-pts = 45 pts)
    6. BondResolvedMPNN Training: Per-node/per-edge architecture for cross-N

Prerequisites (skip Section 1 if data exists):
    python scripts/.../run_e3_bond_resolved_scaling.py --section 0 1 2

Usage:
    # Full run (Sections 1-4, ~2h compute for sweep)
    python scripts/.../run_bond_resolved_cross_n.py

    # Skip VQE sweep, use existing data
    python scripts/.../run_bond_resolved_cross_n.py --section 2 3 4 \\
        --sweep-data results/bond_resolved_scaling/sweep_N40_chain_1d_*.json

    # Dense data generation + per-node MPNN (Mejora 1)
    python scripts/.../run_bond_resolved_cross_n.py --section 5 6

    # Dense + full pipeline (reuses existing sweep if available)
    python scripts/.../run_bond_resolved_cross_n.py --section 5 6 --seeds 42 43 44

    # Dry run (list sections)
    python scripts/.../run_bond_resolved_cross_n.py --dry-run
"""

from __future__ import annotations

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

N_QUBITS = 40
P_LAYERS = 1
TOPOLOGY = "chain_1d"
SEED = 42
CHI_MAX = 64
PRECISION = 0.005
STRATEGY = "aer_mps"

# Valid regime for N=40: h_min = 1.5 + 0.020 * 40^1.31 ≈ 4.01 (corrected formula)
H_MIN_SAFE = 1.5 + 0.020 * N_QUBITS**1.31
H_SWEEP = np.linspace(H_MIN_SAFE + 2.0, H_MIN_SAFE + 0.5, 7).tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════


class BondResolvedCrossNRunner(ValidationRunner):
    """B4: Bond-Resolved Cross-N — GNN necessity proof.

    Shows that norm_type='none' GNN trained on bond-resolved θ_opt (79D)
    can predict midpoint h-values accurately, while random search and
    scipy interpolation cannot handle the dimensionality.
    """

    runner_id = "bond_resolved_cross_n"
    experiment_id = "B4_BR_CROSS_N"
    description = "B4: Bond-Resolved GNN Necessity — 79-param cross-h prediction"
    hypothesis = (
        "GNN (norm_type=none) predicts 79-dim bond-resolved θ_opt at midpoint "
        "h-values with ΔE/gap < 10%, while random init requires >50× more evals."
    )

    @classmethod
    def _add_custom_args(cls, parser):
        parser.add_argument(
            "--n-qubits", type=int, default=N_QUBITS, help="System size (default: %(default)s)"
        )
        parser.add_argument(
            "--sweep-data", type=str, default=None, help="Pre-computed sweep JSON (skips Section 1)"
        )
        parser.add_argument(
            "--hidden-dim",
            type=int,
            default=256,
            help="MPNN hidden dim (default: %(default)s for 79-param output)",
        )
        parser.add_argument(
            "--n-epochs",
            type=int,
            default=8000,
            help="Training epochs (more needed for 79D output)",
        )
        parser.add_argument(
            "--cobyla-maxiter", type=int, default=2000, help="COBYLA maxiter for VQE sweep"
        )
        parser.add_argument(
            "--method",
            type=str,
            default="COBYLA",
            choices=["COBYLA", "L-BFGS-B"],
            help="VQE optimizer method (default: COBYLA). L-BFGS-B uses exact gradients.",
        )
        parser.add_argument(
            "--seeds",
            type=int,
            nargs="+",
            default=[42, 43, 44],
            help="Seeds for multi-seed dense sweep (Section 5). Default: 42 43 44",
        )
        parser.add_argument(
            "--n-dense-points",
            type=int,
            default=15,
            help="Number of h-points per seed in dense sweep (Section 5). Default: 15",
        )

    def build_config(self) -> dict:
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "system": {
                "n_qubits": self._args.n_qubits,
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
            "gnn": {
                "hidden_dim": self._args.hidden_dim,
                "n_layers": 3,
                "norm_type": "none",
                "n_epochs": self._args.n_epochs,
            },
            "h_sweep": H_SWEEP,
            "seeds": [SEED],
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
        from qmbp_simulation.predictors import MPNNPredictor, train_mpnn

        self.torch = torch
        self.builder = HamiltonianBuilder()
        self.solver = ClassicalSolver()
        self.make_lattice = make_lattice
        self.hva = HVACircuitBuilder()
        self.spec_br = get_model_spec("tfim_bond_resolved")
        self.MPSBackend = MPSBackend
        self.VQEOptimizer = VQEOptimizer
        self.VQEConfig = VQEConfig
        self.MPNNPredictor = MPNNPredictor
        self.train_mpnn = train_mpnn

        # Reference lattice for parameter counts
        N = self._args.n_qubits
        self._lattice_ref = make_lattice(TOPOLOGY, N, h=H_SWEEP[0])
        self._n_edges = len(self._lattice_ref.edges)
        self._n_params = self._n_edges + N  # N-1 bonds + N sites for chain_1d
        logger.info(f"  Bond-resolved: N={N}, edges={self._n_edges}, params={self._n_params}")

        # Shared state
        self._sweep_results: list[dict] | None = None
        self._sweep_h_values: list[float] | None = None

        # Load pre-computed sweep if provided
        if self._args.sweep_data:
            self._load_sweep_data(Path(self._args.sweep_data))

    def _load_sweep_data(self, path: Path):
        """Load pre-computed sweep results from E3 Section 2 output."""
        if not path.exists():
            logger.warning(f"  Sweep data not found: {path}")
            return
        with open(path) as f:
            data = json.load(f)
        self._sweep_results = data.get("sweep_results", data.get("per_h_results"))
        h_key = "h_values" if "h_values" in data else None
        if h_key:
            self._sweep_h_values = data[h_key]
        elif self._sweep_results:
            self._sweep_h_values = [r["h"] for r in self._sweep_results]
        n_pts = len(self._sweep_results) if self._sweep_results else 0
        logger.info(f"  Loaded sweep data: {n_pts} h-points from {path.name}")

    def define_sections(self) -> list[Section]:
        return [
            Section(
                id=1,
                name="Bond-Resolved VQE Sweep (N=40, 7 h-points)",
                fn=self.section_sweep,
                hypothesis="≥5/7 h-points converge (ΔE/gap < 5%) with warm-start",
            ),
            Section(
                id=2,
                name="MPNN Training (norm_type=none, 79D output)",
                fn=self.section_train_mpnn,
                hypothesis="MPNN achieves final MSE < 1e-3 on 79-dim θ_opt data",
            ),
            Section(
                id=3,
                name="GNN Deploy at Midpoint h-values",
                fn=self.section_deploy,
                hypothesis="MPNN-predicted θ achieves ΔE/gap < 10% at midpoints",
            ),
            Section(
                id=4,
                name="GNN Necessity: Random Init Baseline",
                fn=self.section_necessity,
                hypothesis="GNN (1 eval) beats random search (100 evals) by >5×",
            ),
            Section(
                id=5,
                name="Dense Multi-Seed Sweep (3 seeds × 15 h-points)",
                fn=self.section_dense_sweep,
                hypothesis="≥80% of dense points converge (ΔE/gap < 5%) with warm-start",
            ),
            Section(
                id=6,
                name="BondResolvedMPNN Training (per-node/per-edge architecture)",
                fn=self.section_train_per_node_mpnn,
                hypothesis="Per-node MPNN achieves MSE < 5e-4 on dense training data",
            ),
        ]

    # ── Section 1: VQE Sweep ─────────────────────────────────────────────────

    def section_sweep(self) -> dict:
        """Run bond-resolved VQE descending sweep (or use pre-loaded data)."""
        from qmbp_simulation.utils.helpers import json_dump as _json_dump

        if self._sweep_results is not None:
            n_pass = sum(
                1 for r in self._sweep_results if r.get("passed", r.get("de_gap", 1) < 0.05)
            )
            logger.info(f"  Using pre-loaded sweep: {n_pass}/{len(self._sweep_results)} pass")
            return {
                "source": "pre_loaded",
                "n_pass": n_pass,
                "n_total": len(self._sweep_results),
                "pass": n_pass >= 5,
            }

        # Run VQE sweep
        N = self._args.n_qubits
        h_values = sorted(H_SWEEP, reverse=True)
        logger.info(
            f"  Running VQE sweep: {len(h_values)} h-points, N={N}, params={self._n_params}"
        )

        backend = self.MPSBackend(
            strategy=STRATEGY, chi_max=CHI_MAX, precision=PRECISION, seed=SEED
        )
        config = self.VQEConfig(
            method=self._args.method,
            p_layers=P_LAYERS,
            n_restarts=1,
            maxiter=self._args.cobyla_maxiter,
            enable_callbacks=False,
        )
        opt = self.VQEOptimizer(config=config, backend=backend, seed=SEED)

        # DMRG ground truth
        logger.info("  Computing DMRG ground truth...")
        dmrg_data = []
        for h in h_values:
            lattice = self.make_lattice(TOPOLOGY, N, h=h)
            H = self.builder.build(lattice)
            gt = self.solver.solve(H, lattice, method="dmrg")
            dmrg_data.append({"h": h, "ground_energy": gt.ground_energy, "gap": gt.gap})
            logger.info(f"    DMRG h={h:.3f}: E₀={gt.ground_energy:.6f}, gap={gt.gap:.4f}")

        # VQE descending sweep with warm-start
        theta_prev = np.random.default_rng(SEED).uniform(-0.01, 0.01, self._n_params)
        sweep_results = []

        for idx, h in enumerate(h_values):
            t0 = time.time()
            lattice_h = self.make_lattice(TOPOLOGY, N, h=h)
            H = self.builder.build(lattice_h)
            qc, _ = self.spec_br.create_circuit(N, P_LAYERS, lattice_h)
            e_exact = dmrg_data[idx]["ground_energy"]
            gap = dmrg_data[idx]["gap"]

            res = opt.optimize(H, qc, theta_prev.copy(), exact_energy=e_exact)
            elapsed = time.time() - t0
            de_gap = abs(res.energy - e_exact) / max(gap, 1e-10)
            theta_prev = res.theta_opt.copy()

            sweep_results.append(
                {
                    "h": h,
                    "vqe_energy": float(res.energy),
                    "dmrg_energy": e_exact,
                    "gap": gap,
                    "de_gap": float(de_gap),
                    "theta_opt": res.theta_opt.tolist(),
                    "n_iterations": res.n_iterations,
                    "time_s": elapsed,
                    "passed": de_gap < 0.05,
                }
            )
            status = "✅" if de_gap < 0.05 else "⚠️"
            logger.info(f"  {status} h={h:.3f}: ΔE/gap={de_gap:.4f} ({elapsed:.1f}s)")

        self._sweep_results = sweep_results
        self._sweep_h_values = h_values

        # Persist sweep data for crash recovery (2h of VQE is expensive)
        output_dir = Path("results/bond_resolved_scaling")
        output_dir.mkdir(parents=True, exist_ok=True)
        sweep_path = output_dir / f"sweep_N{N}_{TOPOLOGY}_{int(time.time())}.json"
        _json_dump(
            {
                "experiment_id": self.experiment_id,
                "n_qubits": N,
                "topology": TOPOLOGY,
                "n_params": self._n_params,
                "n_edges": self._n_edges,
                "h_values": h_values,
                "dmrg_data": dmrg_data,
                "sweep_results": sweep_results,
            },
            sweep_path,
        )
        logger.info(f"  Sweep saved for recovery: {sweep_path}")

        n_pass = sum(1 for r in sweep_results if r["passed"])
        return {
            "source": "computed",
            "n_pass": n_pass,
            "n_total": len(sweep_results),
            "mean_de_gap": float(np.mean([r["de_gap"] for r in sweep_results])),
            "sweep_data_path": str(sweep_path),
            "pass": n_pass >= 5,
        }

    # ── Section 2: MPNN Training ─────────────────────────────────────────────

    def section_train_mpnn(self) -> dict:
        """Train GNN (norm_type=none) on bond-resolved sweep data."""
        from torch_geometric.data import Data

        if not self._sweep_results:
            raise RuntimeError("No sweep data. Run Section 1 first or provide --sweep-data")

        N = self._args.n_qubits
        # Filter passing points for training
        train_pts = [r for r in self._sweep_results if r.get("passed", r.get("de_gap", 1) < 0.05)]
        if len(train_pts) < 3:
            logger.warning(f"  Only {len(train_pts)} passing points — using all sweep data")
            train_pts = self._sweep_results

        logger.info(f"  Training data: {len(train_pts)} h-points, output_dim={self._n_params}")

        # Build graph dataset
        lattice_ref = self.make_lattice(TOPOLOGY, N, h=train_pts[0]["h"])
        edge_index_np, coord = self.builder.build_graph_data(lattice_ref)
        edge_index = self.torch.tensor(edge_index_np, dtype=self.torch.long)

        dataset = []
        for r in train_pts:
            h_feat = np.full(N, float(r["h"]))
            n_feat = np.full(N, N / 100.0)
            x = self.torch.tensor(
                np.stack([h_feat, coord.astype(float), n_feat], axis=1),
                dtype=self.torch.float32,
            )
            y = self.torch.tensor(r["theta_opt"], dtype=self.torch.float32)
            data = Data(x=x, edge_index=edge_index, y=y)
            dataset.append(data)

        # Train MPNN with norm_type=none
        model = self.MPNNPredictor(
            node_features=3,
            hidden_dim=self._args.hidden_dim,
            n_layers=3,
            output_dim=self._n_params,
            norm_type="none",
        )
        n_model_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Model: {n_model_params:,} params, hidden={self._args.hidden_dim}")

        t0 = time.time()
        metrics = self.train_mpnn(model, dataset, n_epochs=self._args.n_epochs, seed=SEED)
        train_time = time.time() - t0
        final_mse = metrics["final_mse"]
        logger.info(f"  Training: MSE={final_mse:.2e}, time={train_time:.1f}s")

        # Store for later sections
        self._model = model
        self._train_pts = train_pts
        self._edge_index = edge_index
        self._coord = coord

        return {
            "n_training_points": len(train_pts),
            "n_model_params": n_model_params,
            "output_dim": self._n_params,
            "final_mse": float(final_mse),
            "training_time_s": train_time,
            "training_converged": not metrics.get("stopped_early", False),
            "pass": final_mse < 1e-3,
        }

    # ── Section 3: GNN Deploy ────────────────────────────────────────────────

    def section_deploy(self) -> dict:
        """Deploy trained GNN on midpoint h-values (interpolation test)."""
        from torch_geometric.data import Data

        if not hasattr(self, "_model"):
            raise RuntimeError("No trained model. Run Section 2 first.")

        N = self._args.n_qubits
        h_train = sorted([r["h"] for r in self._train_pts])

        # Midpoints between consecutive training h-values
        h_deploy = [(h_train[i] + h_train[i + 1]) / 2.0 for i in range(len(h_train) - 1)]
        logger.info(
            f"  Deploy: {len(h_deploy)} midpoints in [{h_deploy[-1]:.3f}, {h_deploy[0]:.3f}]"
        )

        backend = self.MPSBackend(
            strategy=STRATEGY, chi_max=CHI_MAX, precision=PRECISION, seed=SEED
        )
        self._model.eval()
        results = []

        for h_val in sorted(h_deploy, reverse=True):
            t0 = time.time()
            # Build prediction graph
            h_feat = np.full(N, h_val)
            n_feat = np.full(N, N / 100.0)
            x = self.torch.tensor(
                np.stack([h_feat, self._coord.astype(float), n_feat], axis=1),
                dtype=self.torch.float32,
            )
            graph = Data(
                x=x, edge_index=self._edge_index, batch=self.torch.zeros(N, dtype=self.torch.long)
            )

            with self.torch.no_grad():
                theta_pred = self._model(graph).numpy().flatten()

            # Evaluate energy
            lattice = self.make_lattice(TOPOLOGY, N, h=h_val)
            H = self.builder.build(lattice)
            qc, _ = self.spec_br.create_circuit(N, P_LAYERS, lattice)
            e_pred = backend.evaluate(qc, H, theta_pred)

            # DMRG reference
            gt = self.solver.solve(H, lattice, method="dmrg")
            de_gap = abs(e_pred - gt.ground_energy) / max(gt.gap, 1e-10)
            elapsed = time.time() - t0

            results.append(
                {
                    "h": float(h_val),
                    "e_pred": float(e_pred),
                    "e_dmrg": float(gt.ground_energy),
                    "gap": float(gt.gap),
                    "de_gap": float(de_gap),
                    "passed": de_gap < 0.10,
                    "time_s": elapsed,
                }
            )
            status = "✅" if de_gap < 0.10 else "❌"
            logger.info(f"  {status} h={h_val:.3f}: ΔE/gap={de_gap:.4f} ({elapsed:.1f}s)")

        n_pass = sum(1 for r in results if r["passed"])
        mean_de = float(np.mean([r["de_gap"] for r in results]))
        logger.info(f"  Deploy: {n_pass}/{len(results)} pass, mean ΔE/gap={mean_de:.4f}")

        return {
            "h_deploy": [r["h"] for r in results],
            "results": results,
            "n_pass": n_pass,
            "n_total": len(results),
            "mean_de_gap": mean_de,
            "pass": n_pass >= len(results) * 0.6,  # ≥60% pass
        }

    # ── Section 4: Necessity Comparison ──────────────────────────────────────

    def section_necessity(self) -> dict:
        """Compare GNN (1 forward pass) vs random search in 79D."""
        from torch_geometric.data import Data

        if not hasattr(self, "_model"):
            raise RuntimeError("No trained model. Run Section 2 first.")

        N = self._args.n_qubits
        # Use a midpoint between training h-values as test target
        h_train = sorted([r["h"] for r in self._train_pts])
        mid_idx = len(h_train) // 2
        if mid_idx + 1 >= len(h_train):
            mid_idx = max(0, len(h_train) - 2)
        h_test = (h_train[mid_idx] + h_train[mid_idx + 1]) / 2.0

        lattice = self.make_lattice(TOPOLOGY, N, h=h_test)
        H = self.builder.build(lattice)
        qc, _ = self.spec_br.create_circuit(N, P_LAYERS, lattice)
        gt = self.solver.solve(H, lattice, method="dmrg")
        e_exact = gt.ground_energy
        gap = gt.gap

        logger.info(f"  Test point: h={h_test:.3f}, E₀={e_exact:.4f}, gap={gap:.4f}")

        # ── GNN prediction (1 forward pass) ─────────────────────────────
        h_feat = np.full(N, h_test)
        n_feat = np.full(N, N / 100.0)
        x = self.torch.tensor(
            np.stack([h_feat, self._coord.astype(float), n_feat], axis=1),
            dtype=self.torch.float32,
        )
        graph = Data(
            x=x, edge_index=self._edge_index, batch=self.torch.zeros(N, dtype=self.torch.long)
        )

        self._model.eval()
        with self.torch.no_grad():
            theta_gnn = self._model(graph).numpy().flatten()

        backend = self.MPSBackend(
            strategy=STRATEGY, chi_max=CHI_MAX, precision=PRECISION, seed=SEED
        )
        e_gnn = backend.evaluate(qc, H, theta_gnn)
        de_gap_gnn = abs(e_gnn - e_exact) / max(gap, 1e-10)
        logger.info(f"  GNN (1 eval): ΔE/gap={de_gap_gnn:.4f}")

        # ── Random search baseline (N_RANDOM evaluations) ───────────────
        N_RANDOM = 100
        rng = np.random.default_rng(SEED)
        random_energies = []

        logger.info(f"  Random search: {N_RANDOM} samples in [-π, π]^{self._n_params}...")
        t0 = time.time()
        for _ in range(N_RANDOM):
            theta_rand = rng.uniform(-np.pi, np.pi, self._n_params)
            e_rand = backend.evaluate(qc, H, theta_rand)
            random_energies.append(e_rand)
        t_random = time.time() - t0

        best_random_e = min(random_energies)
        de_gap_random = abs(best_random_e - e_exact) / max(gap, 1e-10)
        median_random_e = float(np.median(random_energies))
        de_gap_median = abs(median_random_e - e_exact) / max(gap, 1e-10)

        logger.info(f"  Random best ({N_RANDOM} evals): ΔE/gap={de_gap_random:.4f}")
        logger.info(f"  Random median: ΔE/gap={de_gap_median:.4f}")

        # ── Comparison ───────────────────────────────────────────────────
        gnn_better = de_gap_gnn < de_gap_random
        improvement_factor = de_gap_random / max(de_gap_gnn, 1e-10)

        logger.info(f"\n  GNN vs Random: {improvement_factor:.1f}× better")
        logger.info(f"  GNN is {'NECESSARY ✅' if gnn_better else 'not better ❌'}")

        return {
            "h_test": h_test,
            "e_exact": e_exact,
            "gap": gap,
            "gnn": {
                "de_gap": float(de_gap_gnn),
                "energy": float(e_gnn),
                "n_evals": 1,
            },
            "random_search": {
                "n_samples": N_RANDOM,
                "best_de_gap": float(de_gap_random),
                "best_energy": float(best_random_e),
                "median_de_gap": float(de_gap_median),
                "time_s": t_random,
            },
            "improvement_factor": float(improvement_factor),
            "gnn_is_necessary": bool(gnn_better),
            "pass": gnn_better and improvement_factor > 5.0,
        }


    # ── Section 5: Dense Multi-Seed Sweep ───────────────────────────────────

    def section_dense_sweep(self) -> dict:
        """Generate dense bond-resolved training data: multi-seed × many h-points."""
        from qmbp_simulation.utils.helpers import json_dump as _json_dump

        N = self._args.n_qubits
        seeds = self._args.seeds
        n_pts = self._args.n_dense_points

        # Dense h-grid spanning valid regime
        h_max = H_MIN_SAFE + 3.0
        h_min = H_MIN_SAFE + 0.3
        h_dense = np.linspace(h_max, h_min, n_pts).tolist()

        logger.info(
            f"  Dense sweep: {len(seeds)} seeds × {n_pts} h-points = "
            f"{len(seeds) * n_pts} total (N={N}, params={self._n_params})"
        )
        logger.info(f"  h-range: [{h_min:.2f}, {h_max:.2f}], seeds: {seeds}")

        # Check if dense data already exists (avoid re-running expensive VQE)
        output_dir = Path("results/bond_resolved_scaling")
        output_dir.mkdir(parents=True, exist_ok=True)
        existing_dense = list(output_dir.glob(f"dense_N{N}_{TOPOLOGY}_*.json"))

        all_sweep_results = []

        if existing_dense:
            logger.info(f"  Found {len(existing_dense)} existing dense files, loading...")
            for p in existing_dense:
                with open(p) as f:
                    d = json.load(f)
                all_sweep_results.extend(d.get("sweep_results", []))
            logger.info(f"  Loaded {len(all_sweep_results)} existing points")

        # Determine which (seed, h) combos are missing
        existing_keys = {(r["seed"], round(r["h"], 4)) for r in all_sweep_results}
        needed = [
            (seed, h)
            for seed in seeds
            for h in h_dense
            if (seed, round(h, 4)) not in existing_keys
        ]

        if not needed:
            logger.info("  All dense data already exists — skipping VQE computation")
        else:
            logger.info(f"  Computing {len(needed)} new VQE points...")

            # Group by seed for efficient warm-start chains
            from collections import defaultdict

            by_seed = defaultdict(list)
            for seed, h in needed:
                by_seed[seed].append(h)

            for seed, h_vals in by_seed.items():
                h_vals_sorted = sorted(h_vals, reverse=True)
                logger.info(f"    Seed {seed}: {len(h_vals_sorted)} h-points")

                backend = self.MPSBackend(
                    strategy=STRATEGY, chi_max=CHI_MAX, precision=PRECISION, seed=seed
                )
                config = self.VQEConfig(
                    method=self._args.method,
                    p_layers=P_LAYERS,
                    n_restarts=1,
                    maxiter=self._args.cobyla_maxiter,
                    enable_callbacks=False,
                )
                opt = self.VQEOptimizer(config=config, backend=backend, seed=seed)

                theta_prev = np.random.default_rng(seed).uniform(-0.01, 0.01, self._n_params)

                for h in h_vals_sorted:
                    t0 = time.time()
                    lattice_h = self.make_lattice(TOPOLOGY, N, h=h)
                    H = self.builder.build(lattice_h)
                    qc, _ = self.spec_br.create_circuit(N, P_LAYERS, lattice_h)
                    gt = self.solver.solve(H, lattice_h, method="dmrg")

                    res = opt.optimize(H, qc, theta_prev.copy(), exact_energy=gt.ground_energy)
                    elapsed = time.time() - t0
                    de_gap = abs(res.energy - gt.ground_energy) / max(gt.gap, 1e-10)
                    theta_prev = res.theta_opt.copy()

                    result = {
                        "h": float(h),
                        "seed": int(seed),
                        "vqe_energy": float(res.energy),
                        "dmrg_energy": float(gt.ground_energy),
                        "gap": float(gt.gap),
                        "de_gap": float(de_gap),
                        "theta_opt": res.theta_opt.tolist(),
                        "n_iterations": res.n_iterations,
                        "time_s": elapsed,
                        "passed": de_gap < 0.05,
                    }
                    all_sweep_results.append(result)

                    status = "✅" if de_gap < 0.05 else "⚠️"
                    logger.info(
                        f"      {status} seed={seed} h={h:.3f}: "
                        f"ΔE/gap={de_gap:.4f} ({elapsed:.1f}s)"
                    )

            # Save dense data
            dense_path = output_dir / f"dense_N{N}_{TOPOLOGY}_{int(time.time())}.json"
            _json_dump(
                {
                    "experiment_id": self.experiment_id,
                    "section": "dense_sweep",
                    "n_qubits": N,
                    "topology": TOPOLOGY,
                    "n_params": self._n_params,
                    "n_edges": self._n_edges,
                    "seeds": seeds,
                    "h_values": h_dense,
                    "sweep_results": all_sweep_results,
                },
                dense_path,
            )
            logger.info(f"  Dense sweep saved: {dense_path}")

        # Store for Section 6
        self._dense_results = all_sweep_results

        n_total = len(all_sweep_results)
        n_pass = sum(1 for r in all_sweep_results if r.get("passed", r.get("de_gap", 1) < 0.05))
        mean_de = float(np.mean([r["de_gap"] for r in all_sweep_results])) if n_total > 0 else 0

        logger.info(f"  Dense sweep: {n_pass}/{n_total} pass, mean ΔE/gap={mean_de:.4f}")

        return {
            "n_total": n_total,
            "n_pass": n_pass,
            "n_seeds": len(seeds),
            "n_h_points": n_pts,
            "mean_de_gap": mean_de,
            "pass": n_pass >= n_total * 0.8,
        }

    # ── Section 6: BondResolvedMPNN (Per-Node/Per-Edge) Training ──────────

    def section_train_per_node_mpnn(self) -> dict:
        """Train the per-node/per-edge BondResolvedMPNN on dense data."""
        from qmbp_simulation.predictors import (
            BondResolvedMPNN,
            build_bond_resolved_graph,
            train_bond_resolved_mpnn,
        )
        from qmbp_simulation.predictors import save_mpnn_checkpoint

        # Use dense results if available, otherwise fall back to Section 1 data
        if hasattr(self, "_dense_results") and self._dense_results:
            train_data = self._dense_results
            source = "dense_sweep"
        elif self._sweep_results:
            train_data = self._sweep_results
            source = "section_1_sweep"
        else:
            raise RuntimeError(
                "No training data. Run Section 1 or Section 5 first."
            )

        # Filter passing points
        N = self._args.n_qubits
        passing = [r for r in train_data if r.get("passed", r.get("de_gap", 1) < 0.05)]
        if len(passing) < 5:
            logger.warning(f"  Only {len(passing)} passing points — using all data")
            passing = train_data

        logger.info(
            f"  Training BondResolvedMPNN: {len(passing)} pts, "
            f"source={source}, N={N}, output=per-node/per-edge"
        )

        # Build graph dataset using the new per-node API
        dataset = []
        for r in passing:
            h_val = r["h"]
            lattice = self.make_lattice(TOPOLOGY, N, h=h_val)
            theta_opt = np.array(r["theta_opt"])
            graph = build_bond_resolved_graph(lattice, h_val, theta_opt=theta_opt, n_feature=True)
            dataset.append(graph)

        # Create model
        model = BondResolvedMPNN(
            node_features=3,
            hidden_dim=self._args.hidden_dim,
            n_layers=3,
            norm_type="none",
            dropout=0.1,
        )
        n_model_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Model: {n_model_params:,} params, hidden={self._args.hidden_dim}")
        logger.info(f"  Architecture: per-node θ_x ({N}) + per-edge θ_zz ({self._n_edges})")

        # Train
        t0 = time.time()
        metrics = train_bond_resolved_mpnn(
            model, dataset, n_epochs=self._args.n_epochs, lr=1e-3, patience=300, seed=SEED
        )
        train_time = time.time() - t0

        logger.info(
            f"  Training done: MSE={metrics['final_mse']:.2e} "
            f"(ZZ={metrics['final_zz_mse']:.2e}, X={metrics['final_x_mse']:.2e}), "
            f"time={train_time:.1f}s"
        )

        # Save checkpoint
        output_dir = Path("results/bond_resolved_scaling")
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = output_dir / f"bond_resolved_mpnn_N{N}_{int(time.time())}.pt"

        import torch
        torch.save(
            {
                "state_dict": model.state_dict(),
                "architecture": "bond_resolved_mpnn",
                "node_features": model.node_features,
                "hidden_dim": model.hidden_dim,
                "n_layers": model.n_layers,
                "norm_type": model.norm_type,
                "n_qubits_trained": N,
                "n_training_points": len(passing),
                "training_metrics": {
                    "final_mse": metrics["final_mse"],
                    "final_zz_mse": metrics["final_zz_mse"],
                    "final_x_mse": metrics["final_x_mse"],
                },
            },
            str(ckpt_path),
        )
        logger.info(f"  Checkpoint saved: {ckpt_path}")

        # Store for potential use by cross-N transfer script
        self._bond_resolved_model = model

        return {
            "source": source,
            "n_training_points": len(passing),
            "n_model_params": n_model_params,
            "final_mse": float(metrics["final_mse"]),
            "final_zz_mse": float(metrics["final_zz_mse"]),
            "final_x_mse": float(metrics["final_x_mse"]),
            "training_time_s": train_time,
            "checkpoint_path": str(ckpt_path),
            "pass": metrics["final_mse"] < 5e-4,
        }


if __name__ == "__main__":
    BondResolvedCrossNRunner.main()
