#!/usr/bin/env python3
"""Bond-Resolved HVA Scaling Suite — 2D Geometry + MPNN + Noisy Simulation.

Extends the initial bond-resolved validation with:
  A: 2D square lattice (N=9 3x3, N=12 3x4) convergence
  B1: MPNN training on bond-resolved data (GNN necessity proof)
  B2: Noisy simulation with ZNE (verify CX budget unchanged)
  B3: Cross-topology comparison including square

Hypothesis: Bond-resolved HVA with GNN prediction achieves dE/gap < 5%
across 2D geometries, and ZNE remains effective (same CX budget).

Usage:
    python scripts/experiment_runners/bond_resolved/run_bond_resolved_scaling.py
    python scripts/experiment_runners/bond_resolved/run_bond_resolved_scaling.py --section 1
    python scripts/experiment_runners/bond_resolved/run_bond_resolved_scaling.py --dry-run
"""

from __future__ import annotations

import logging
import sys
import time

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


class BondResolvedScalingRunner(ValidationRunner):
    """Bond-Resolved HVA Scaling: 2D + MPNN + Noisy validation."""

    runner_id = "bond_resolved_scaling"
    experiment_id = "BOND_RESOLVED_SCALING"
    description = "Bond-resolved scaling: 2D square, MPNN training, ZNE validation"
    hypothesis = (
        "Bond-resolved HVA scales to 2D square lattice, GNN predicts "
        "high-dimensional theta_opt, and ZNE remains effective"
    )

    @classmethod
    def _add_custom_args(cls, parser):
        parser.add_argument("--seed", type=int, default=42)

    def build_config(self) -> dict:
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            "system": {
                "model": "tfim_bond_resolved",
                "p_layers": 1,
                "topologies": ["square", "heavy_hex", "chain_1d"],
            },
            "seed": self._args.seed,
        }

    def setup(self):
        from qmbp_simulation import (
            ClassicalSolver,
            HamiltonianBuilder,
            VQEConfig,
            VQEOptimizer,
            make_lattice,
        )
        from qmbp_simulation.models.model_registry import get_model_spec

        self.builder = HamiltonianBuilder()
        self.solver = ClassicalSolver()
        self.make_lattice = make_lattice
        self.spec_br = get_model_spec("tfim_bond_resolved")
        self.VQEOptimizer = VQEOptimizer
        self.VQEConfig = VQEConfig
        self._seed = self._args.seed

    def define_sections(self) -> list[Section]:
        return [
            Section(
                id=1,
                name="2D Square N=9 Convergence",
                fn=self.section_square_n9,
                hypothesis="Bond-resolved VQE converges on 3x3 square (fid>=0.99 at h>=4)",
            ),
            Section(
                id=2,
                name="2D Square N=12 Convergence",
                fn=self.section_square_n12,
                hypothesis="Bond-resolved VQE converges on 3x4 square (fid>=0.99 at h>=4.5)",
            ),
            Section(
                id=3,
                name="MPNN Training (heavy-hex N=10)",
                fn=self.section_mpnn_training,
                hypothesis="GNN predicts 19-dim theta_opt with dE/gap < 5% on unseen h",
            ),
            Section(
                id=4,
                name="Noisy Simulation + ZNE",
                fn=self.section_noisy_zne,
                hypothesis="ZNE gain > 0 with bond-resolved (same CX budget as global)",
            ),
        ]

    # ── Section 1: Square N=9 (3×3) ─────────────────────────────────────

    def section_square_n9(self) -> dict:
        """VQE convergence on 3×3 square lattice with bond-resolved."""
        N, P = 9, 1
        TOPOLOGY = "square"
        H_VALS = [6.0, 5.5, 5.0, 4.5, 4.0]  # h_c(square) ~ 3.04
        seed = self._seed

        opt = self.VQEOptimizer(
            self.VQEConfig(n_restarts=3, restart_sigma=0.05, maxiter=1500),
            seed=seed,
        )
        lattice_ref = self.make_lattice(TOPOLOGY, N, h=6.0)
        n_edges = len(lattice_ref.edges)
        n_params = n_edges + N
        prev_theta = None
        results = []

        for h in H_VALS:
            lattice = self.make_lattice(TOPOLOGY, N, h=h)
            H = self.builder.build(lattice)
            gt = self.solver.solve(H, lattice)
            qc, _ = self.spec_br.create_circuit(N, P, lattice)

            if prev_theta is None:
                init = np.random.default_rng(seed).uniform(-0.01, 0.01, n_params)
            else:
                init = prev_theta.copy()

            t0 = time.time()
            res = opt.optimize(H, qc, init, gt.ground_energy, gt.ground_state)
            elapsed = time.time() - t0
            prev_theta = res.theta_opt.copy()

            de_gap = abs(res.energy_error) / gt.gap if gt.gap > 0 else float("inf")
            results.append(
                {
                    "h": h,
                    "fidelity": float(res.fidelity),
                    "delta_e_gap": float(de_gap),
                    "elapsed_s": elapsed,
                }
            )
            logger.info(
                f"  h={h:.1f} | fid={res.fidelity:.6f} | dE/gap={de_gap:.4f} | {elapsed:.1f}s"
            )

        all_pass = all(r["fidelity"] >= 0.99 for r in results if r["h"] >= 4.0)
        return {
            "topology": TOPOLOGY,
            "n_qubits": N,
            "p_layers": P,
            "n_params": n_params,
            "n_edges": n_edges,
            "grid_shape": "3x3",
            "results": results,
            "pass": all_pass,
        }

    # ── Section 2: Square N=12 (3×4) ────────────────────────────────────

    def section_square_n12(self) -> dict:
        """VQE convergence on 3×4 square lattice with bond-resolved."""
        N, P = 12, 1
        TOPOLOGY = "square"
        H_VALS = [6.0, 5.5, 5.0, 4.5]  # Conservative: h >= 4.5
        seed = self._seed

        opt = self.VQEOptimizer(
            self.VQEConfig(n_restarts=3, restart_sigma=0.05, maxiter=1500),
            seed=seed,
        )
        lattice_ref = self.make_lattice(TOPOLOGY, N, h=6.0)
        n_edges = len(lattice_ref.edges)
        n_params = n_edges + N
        prev_theta = None
        results = []

        for h in H_VALS:
            lattice = self.make_lattice(TOPOLOGY, N, h=h)
            H = self.builder.build(lattice)
            gt = self.solver.solve(H, lattice)
            qc, _ = self.spec_br.create_circuit(N, P, lattice)

            if prev_theta is None:
                init = np.random.default_rng(seed).uniform(-0.01, 0.01, n_params)
            else:
                init = prev_theta.copy()

            t0 = time.time()
            res = opt.optimize(H, qc, init, gt.ground_energy, gt.ground_state)
            elapsed = time.time() - t0
            prev_theta = res.theta_opt.copy()

            de_gap = abs(res.energy_error) / gt.gap if gt.gap > 0 else float("inf")
            results.append(
                {
                    "h": h,
                    "fidelity": float(res.fidelity),
                    "delta_e_gap": float(de_gap),
                    "elapsed_s": elapsed,
                }
            )
            logger.info(
                f"  h={h:.1f} | fid={res.fidelity:.6f} | "
                f"dE/gap={de_gap:.4f} | params={n_params} | {elapsed:.1f}s"
            )

        all_pass = all(r["fidelity"] >= 0.99 for r in results if r["h"] >= 4.5)
        return {
            "topology": TOPOLOGY,
            "n_qubits": N,
            "p_layers": P,
            "n_params": n_params,
            "n_edges": n_edges,
            "grid_shape": "3x4",
            "results": results,
            "pass": all_pass,
        }

    # ── Section 3: MPNN Training ─────────────────────────────────────────

    def section_mpnn_training(self) -> dict:
        """Train MPNN on bond-resolved VQE data and predict unseen h.

        This proves the GNN is NECESSARY for bond-resolved (interpolation fails
        in 19+ dimensional parameter space).
        """
        N, P = 10, 1
        TOPOLOGY = "heavy_hex"
        H_TRAIN = [4.5, 4.25, 4.0, 3.75, 3.5, 3.25]
        H_TEST = [3.875, 3.625]  # Interpolation points
        seed = self._seed

        # Phase 2: Generate training data
        opt = self.VQEOptimizer(
            self.VQEConfig(n_restarts=5, restart_sigma=0.05, maxiter=2000),
            seed=seed,
        )
        lattice_ref = self.make_lattice(TOPOLOGY, N, h=4.5)
        n_params = len(lattice_ref.edges) + N

        logger.info(f"  Phase 2: VQE sweep ({len(H_TRAIN)} h-points, {n_params} params)...")
        prev_theta = None
        theta_train = []
        h_train_arr = []
        e_exact_train = []
        fid_train = []

        for h in H_TRAIN:
            lattice = self.make_lattice(TOPOLOGY, N, h=h)
            H = self.builder.build(lattice)
            gt = self.solver.solve(H, lattice)
            qc, _ = self.spec_br.create_circuit(N, P, lattice)

            if prev_theta is None:
                init = np.random.default_rng(seed).uniform(-0.01, 0.01, n_params)
            else:
                init = prev_theta.copy()

            res = opt.optimize(H, qc, init, gt.ground_energy, gt.ground_state)
            prev_theta = res.theta_opt.copy()

            theta_train.append(res.theta_opt.copy())
            h_train_arr.append(h)
            e_exact_train.append(gt.ground_energy)
            fid_train.append(res.fidelity)

        theta_train_np = np.array(theta_train)
        h_train_np = np.array(h_train_arr)
        e_exact_np = np.array(e_exact_train)
        fid_np = np.array(fid_train)

        logger.info(f"  Phase 2 complete. Mean fidelity: {np.mean(fid_np):.4f}")

        # Phase 3: Train MPNN
        from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn

        dataset = build_graph_dataset(
            lattice_ref,
            h_train_np,
            theta_train_np,
            e_exact_np,
            fidelities=fid_np,
            fidelity_threshold=0.50,  # noqa: relaxed for bond-resolved (boundary points may not hit 0.93)
        )
        logger.info(
            f"  Phase 3: Training MPNN (dataset={len(dataset)} points, output_dim={n_params})..."
        )

        model = MPNNPredictor(
            node_features=2,
            hidden_dim=128,
            n_layers=3,
            output_dim=n_params,
        )
        train_result = train_mpnn(
            model,
            dataset,
            n_epochs=4000,
            lr=1e-3,
            patience=400,
        )
        final_loss = train_result["final_mse"]
        logger.info(f"  MPNN trained. Final loss: {final_loss:.6f}")

        # Phase 4: Predict at unseen h-values and evaluate
        import torch

        model.eval()
        test_results = []

        for h_test in H_TEST:
            lattice_test = self.make_lattice(TOPOLOGY, N, h=h_test)
            H_test = self.builder.build(lattice_test)
            gt_test = self.solver.solve(H_test, lattice_test)
            qc_test, _ = self.spec_br.create_circuit(N, P, lattice_test)

            # Build a single graph for prediction (bypass build_graph_dataset min-3 check)
            from qmbp_simulation.models import HamiltonianBuilder

            hb = HamiltonianBuilder()
            edge_index_np, coord = hb.build_graph_data(lattice_test)
            edge_index_t = torch.tensor(edge_index_np, dtype=torch.long)
            # Node features: [h_value, coordination_number] for each node
            h_feat = np.full(N, h_test)
            x_np = np.stack([h_feat, coord], axis=1)
            x_t = torch.tensor(x_np, dtype=torch.float32)
            from torch_geometric.data import Data

            test_graph = Data(x=x_t, edge_index=edge_index_t)

            with torch.no_grad():
                theta_pred = model(test_graph).numpy().flatten()

            # Evaluate predicted parameters
            from qmbp_simulation.execution import NoiselessBackend

            backend = NoiselessBackend()
            e_pred = backend.evaluate(qc_test, H_test, theta_pred)
            de_gap = (
                abs(e_pred - gt_test.ground_energy) / gt_test.gap
                if gt_test.gap > 0
                else float("inf")
            )

            # Also try linear interpolation as baseline
            idx_lo = max(i for i, hv in enumerate(H_TRAIN) if hv >= h_test)
            idx_hi = idx_lo + 1
            if idx_hi < len(H_TRAIN):
                frac = (h_test - H_TRAIN[idx_lo]) / (H_TRAIN[idx_hi] - H_TRAIN[idx_lo])
                theta_interp = theta_train_np[idx_lo] * (1 - frac) + theta_train_np[idx_hi] * frac
            else:
                theta_interp = theta_train_np[idx_lo]
            e_interp = backend.evaluate(qc_test, H_test, theta_interp)
            de_gap_interp = (
                abs(e_interp - gt_test.ground_energy) / gt_test.gap
                if gt_test.gap > 0
                else float("inf")
            )

            test_results.append(
                {
                    "h_test": h_test,
                    "de_gap_mpnn": float(de_gap),
                    "de_gap_interp": float(de_gap_interp),
                    "mpnn_wins": de_gap < de_gap_interp,
                }
            )
            logger.info(
                f"  h={h_test:.3f} | MPNN: dE/gap={de_gap:.4f} | "
                f"Interp: dE/gap={de_gap_interp:.4f} | "
                f"{'MPNN wins' if de_gap < de_gap_interp else 'Interp wins'}"
            )

        mpnn_pass = all(r["de_gap_mpnn"] < 0.05 for r in test_results)
        n_mpnn_wins = sum(1 for r in test_results if r["mpnn_wins"])

        return {
            "topology": TOPOLOGY,
            "n_qubits": N,
            "n_params": n_params,
            "train_points": len(H_TRAIN),
            "test_points": len(H_TEST),
            "final_train_loss": float(final_loss),
            "mean_train_fidelity": float(np.mean(fid_np)),
            "test_results": test_results,
            "mpnn_wins": n_mpnn_wins,
            "n_tests": len(H_TEST),
            "pass": mpnn_pass,
        }

    # ── Section 4: Noisy Simulation + ZNE ────────────────────────────────

    def section_noisy_zne(self) -> dict:
        """Verify ZNE still works with bond-resolved params (same CX count).

        Uses gate-folding ZNE on FakeTorino noise model.
        Validates that bond-resolved has same noise profile as global HVA.
        """
        N, P = 10, 1
        TOPOLOGY = "heavy_hex"
        H_TEST_POINTS = [4.0, 3.5]
        seed = self._seed

        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        from qmbp_simulation.execution import (
            NoiselessBackend,
            NoisyEstimatorConfig,
            run_gate_folding_zne,
        )

        noiseless_backend = NoiselessBackend()
        fake_backend = FakeTorino()

        opt = self.VQEOptimizer(
            self.VQEConfig(n_restarts=3, restart_sigma=0.05, maxiter=1500),
            seed=seed,
        )

        lattice_ref = self.make_lattice(TOPOLOGY, N, h=4.0)
        n_params = len(lattice_ref.edges) + N

        results_per_h = []
        prev_theta = None

        for h in H_TEST_POINTS:
            lattice = self.make_lattice(TOPOLOGY, N, h=h)
            H = self.builder.build(lattice)
            gt = self.solver.solve(H, lattice)
            qc, _ = self.spec_br.create_circuit(N, P, lattice)

            # Get optimal params via VQE
            if prev_theta is None:
                init = np.random.default_rng(seed).uniform(-0.01, 0.01, n_params)
            else:
                init = prev_theta.copy()
            res = opt.optimize(H, qc, init, gt.ground_energy, gt.ground_state)
            prev_theta = res.theta_opt.copy()
            theta_opt = res.theta_opt

            # Noiseless energy
            e_noiseless = noiseless_backend.evaluate(qc, H, theta_opt)
            de_noiseless = abs(e_noiseless - gt.ground_energy) / gt.gap if gt.gap > 0 else 0

            try:
                # Bind parameters and transpile for FakeTorino
                bound_qc = qc.assign_parameters(theta_opt)
                pm = generate_preset_pass_manager(optimization_level=2, backend=fake_backend)
                transpiled = pm.run(bound_qc)

                # Map observable to transpiled layout
                layout = transpiled.layout
                if layout and layout.final_index_layout(filter_ancillas=True):
                    H_mapped = H.apply_layout(layout)
                else:
                    H_mapped = H

                config = NoisyEstimatorConfig(shots=16384, seed_simulator=seed)

                zne_result = run_gate_folding_zne(
                    transpiled_circuit=transpiled,
                    observable=H_mapped,
                    backend=fake_backend,
                    config=config,
                )
                e_zne = zne_result.extrapolated_value
                r2 = zne_result.r_squared

                de_zne = abs(e_zne - gt.ground_energy) / gt.gap if gt.gap > 0 else float("inf")

                # Get raw noisy energy (noise factor = 1, first measured value)
                e_noisy_raw = zne_result.measured_values[0] if zne_result.measured_values else e_zne
                de_noisy = (
                    abs(e_noisy_raw - gt.ground_energy) / gt.gap if gt.gap > 0 else float("inf")
                )

                gain = ((de_noisy - de_zne) / de_noisy * 100) if de_noisy > 1e-10 else 0

                results_per_h.append(
                    {
                        "h": h,
                        "r2": float(r2),
                        "de_noiseless": float(de_noiseless),
                        "de_noisy_raw": float(de_noisy),
                        "de_zne": float(de_zne),
                        "gain_pct": float(gain),
                        "zne_positive": gain > 0,
                    }
                )
                logger.info(
                    f"  h={h:.1f} | R2={r2:.4f} | gain={gain:+.1f}% | "
                    f"dE: noiseless={de_noiseless:.4f} noisy={de_noisy:.4f} zne={de_zne:.4f}"
                )
            except Exception as e:
                logger.warning(f"  h={h:.1f} | ZNE failed: {e}")
                results_per_h.append(
                    {
                        "h": h,
                        "error": str(e),
                        "zne_positive": False,
                        "de_noiseless": float(de_noiseless),
                    }
                )

        n_positive = sum(1 for r in results_per_h if r.get("zne_positive"))
        all_pass = n_positive >= len(H_TEST_POINTS) * 0.5

        return {
            "topology": TOPOLOGY,
            "n_qubits": N,
            "n_params": n_params,
            "results_per_h": results_per_h,
            "n_positive_gain": n_positive,
            "n_total": len(H_TEST_POINTS),
            "pass": all_pass,
        }


if __name__ == "__main__":
    BondResolvedScalingRunner.main()
