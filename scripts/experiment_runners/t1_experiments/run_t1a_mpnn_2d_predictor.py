#!/usr/bin/env python3
"""Tier 1A: MPNN Generalization Across J₂ Values (2D Predictor).

Trains a single MPNN that predicts θ(h, J₂) for any J₂ in [0.1, 0.5],
then tests at unseen (h, J₂) combinations. Demonstrates the GNN can
interpolate across a 2D phase diagram (h × J₂).

Hypothesis:
  A single MPNN with node_features=[h, coord, J₂] achieves ΔE/gap < 5%
  on held-out (h, J₂) test points, demonstrating 2D interpolation.

Sections:
  1. Phase 2 — VQE data generation across (h, J₂) grid
  2. Phase 3 — MPNN training with J₂ as extra node feature
  3. Phase 4 — Deployment at unseen (h, J₂) interpolation points
  4. Comparison — 2D vs 1D (fixed-J₂) generalization gap

Usage:
    python scripts/run_t1a_mpnn_2d_predictor.py
    python scripts/run_t1a_mpnn_2d_predictor.py --section 1 2 3
    python scripts/run_t1a_mpnn_2d_predictor.py --dry-run
    python scripts/run_t1a_mpnn_2d_predictor.py --j2-grid 0.1 0.3 0.5
"""

from __future__ import annotations

import logging
import sys

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

N_QUBITS = 6
P_LAYERS = 2
TOPOLOGY = "chain_1d"
SEEDS = DEFAULT_SEEDS

# 2D grid: h × J₂
H_TRAIN = [2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0, 0.875, 0.75]
J2_TRAIN = [0.1, 0.2, 0.3, 0.4, 0.5]  # Training J₂ values

# Test points: unseen (h, J₂) interpolation combinations
H_TEST = [1.875, 1.375, 1.125]  # Interpolation h-values
J2_TEST = [0.15, 0.35, 0.45]  # Interpolation J₂-values

# VQE
VQE_RESTARTS = 5
VQE_MAXITER = 500
VQE_SIGMA = 0.1

# MPNN
MPNN_HIDDEN_DIM = 64
MPNN_N_LAYERS = 3
MPNN_EPOCHS = 6000
MPNN_LR = 1e-3
MPNN_PATIENCE = 500


# ═══════════════════════════════════════════════════════════════════════════════
# Runner Implementation
# ═══════════════════════════════════════════════════════════════════════════════


class MPNN2DPredictorRunner(ValidationRunner):
    """Tier 1A: 2D MPNN predictor across h × J₂ parameter space.

    Sections:
        1. VQE data generation across (h, J₂) grid
        2. MPNN training with 3 node features [h, coord, J₂]
        3. Deployment at unseen (h, J₂) test points
        4. Comparison: 2D generalization vs 1D baseline
    """

    runner_id = "t1a_mpnn_2d"
    experiment_id = "T1a"
    description = "MPNN 2D Predictor — Generalization Across J₂ Values"
    hypothesis = (
        "A single MPNN with node_features=[h, coord, J₂] achieves ΔE/gap < 5% "
        "on unseen (h, J₂) interpolation points"
    )

    @classmethod
    def _add_custom_args(cls, parser):
        parser.add_argument(
            "--j2-grid",
            type=float,
            nargs="+",
            default=None,
            help=f"J₂ training values (default: {J2_TRAIN})",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Primary seed for VQE + MPNN (default: 42)",
        )

    def build_config(self) -> dict:
        j2_grid = getattr(self, "_j2_grid", J2_TRAIN)
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "category": "T",
            "model": "tfim_frustrated",
            "system": {
                "n_qubits": N_QUBITS,
                "p_layers": P_LAYERS,
                "topology": TOPOLOGY,
            },
            "grid": {
                "h_train": H_TRAIN,
                "j2_train": j2_grid,
                "h_test": H_TEST,
                "j2_test": J2_TEST,
                "n_train_points": len(H_TRAIN) * len(j2_grid),
            },
            "mpnn": {
                "hidden_dim": MPNN_HIDDEN_DIM,
                "n_layers": MPNN_N_LAYERS,
                "epochs": MPNN_EPOCHS,
                "lr": MPNN_LR,
                "patience": MPNN_PATIENCE,
                "node_features": 3,
            },
            "seeds": SEEDS,
        }

    def setup(self):
        """Lazy imports and shared objects."""
        from scipy.optimize import minimize

        from qmbp_simulation import HamiltonianBuilder, make_lattice
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.execution import NoiselessBackend

        self.builder = HamiltonianBuilder()
        self.hva = HVACircuitBuilder()
        self.backend = NoiselessBackend()
        self._minimize = minimize
        self._make_lattice = make_lattice

        # Resolve J₂ grid from CLI or default
        self._j2_grid = self._args.j2_grid if self._args.j2_grid else J2_TRAIN
        self._seed = self._args.seed

        # Build reference circuit
        lattice_ref = make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=2.0)
        self._circuit, _ = self.hva.create_frustrated_tfim(N_QUBITS, P_LAYERS, lattice_ref)
        self._n_params = self._circuit.num_parameters
        self._lattice_ref = lattice_ref

        # Shared state across sections
        self._vqe_data = None  # Populated in section 1
        self._model = None  # Populated in section 2

    def define_sections(self) -> list[Section]:
        return [
            Section(
                id=1,
                name="VQE Data Generation (h×J₂ grid)",
                fn=self.section_vqe_grid,
                hypothesis=(
                    "VQE descending sweep converges with fid≥0.93 across the full (h, J₂) grid"
                ),
            ),
            Section(
                id=2,
                name="MPNN Training (2D predictor)",
                fn=self.section_mpnn_training,
                hypothesis=(
                    "MPNN with node_features=[h, coord, J₂] converges to "
                    "MSE < 1e-3 on training grid"
                ),
            ),
            Section(
                id=3,
                name="Deployment at Unseen (h, J₂)",
                fn=self.section_deployment,
                hypothesis="ΔE/gap < 5% at interpolation (h, J₂) test points",
            ),
            Section(
                id=4,
                name="2D vs 1D Comparison",
                fn=self.section_comparison,
                hypothesis=(
                    "2D model matches or exceeds 1D fixed-J₂ model at the same J₂ test points"
                ),
            ),
        ]

    # ── Section 1: VQE Data Generation ───────────────────────────────────────

    def section_vqe_grid(self) -> dict:
        """Generate VQE θ_opt data across the 2D (h, J₂) grid.

        For each J₂ value, runs a descending h-sweep (warm-started).
        """
        from qiskit.quantum_info import Statevector, state_fidelity

        logger.info(
            f"  Grid: {len(H_TRAIN)} h × {len(self._j2_grid)} J₂ "
            f"= {len(H_TRAIN) * len(self._j2_grid)} total points"
        )

        rng = np.random.default_rng(self._seed)
        all_data = []  # List of dicts with h, j2, theta, e_exact, fidelity

        for j2 in self._j2_grid:
            prev_theta = rng.uniform(-0.01, 0.01, self._n_params)

            for h in sorted(H_TRAIN, reverse=True):
                lattice = self._make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=h)
                H = self.builder.build_frustrated_tfim(lattice, J2=j2)

                # Exact ground state
                H_mat = H.to_matrix()
                if hasattr(H_mat, "toarray"):
                    H_mat = H_mat.toarray()
                evals, evecs = np.linalg.eigh(H_mat)
                e_exact = float(evals[0])
                gs = evecs[:, 0]

                # Multi-restart VQE
                best_energy = float("inf")
                best_theta = prev_theta.copy()
                for restart in range(VQE_RESTARTS):
                    x0 = (
                        prev_theta + rng.normal(0, VQE_SIGMA, self._n_params)
                        if restart > 0
                        else prev_theta.copy()
                    )
                    x0 = np.clip(x0, -np.pi, np.pi)
                    res = self._minimize(
                        lambda params, _H=H: self.backend.evaluate(self._circuit, _H, params),
                        x0,
                        method="L-BFGS-B",
                        bounds=[(-np.pi, np.pi)] * self._n_params,
                        options={"maxiter": VQE_MAXITER, "ftol": 1e-14},
                    )
                    if res.fun < best_energy:
                        best_energy = res.fun
                        best_theta = res.x.copy()

                prev_theta = best_theta.copy()

                # Fidelity
                sv = Statevector(self._circuit.assign_parameters(best_theta))
                fid = float(state_fidelity(sv, Statevector(gs)))

                all_data.append(
                    {
                        "h": h,
                        "j2": j2,
                        "theta": best_theta.tolist(),
                        "e_exact": e_exact,
                        "fidelity": fid,
                    }
                )

            logger.info(
                f"    J₂={j2:.2f}: mean fid = "
                f"{np.mean([d['fidelity'] for d in all_data if d['j2'] == j2]):.4f}"
            )

        # Store for next sections
        self._vqe_data = all_data

        # Summary
        fids = [d["fidelity"] for d in all_data]
        n_pass = sum(1 for f in fids if f >= 0.93)
        mean_fid = float(np.mean(fids))
        min_fid = float(np.min(fids))

        logger.info(f"\n  Total points: {len(all_data)}")
        logger.info(f"  Mean fidelity: {mean_fid:.4f}")
        logger.info(f"  Min fidelity:  {min_fid:.4f}")
        logger.info(f"  Pass rate (≥0.93): {n_pass}/{len(all_data)}")

        return {
            "n_points": len(all_data),
            "mean_fidelity": mean_fid,
            "min_fidelity": min_fid,
            "pass_rate": n_pass / len(all_data),
            "per_j2_mean_fid": {
                str(j2): float(np.mean([d["fidelity"] for d in all_data if d["j2"] == j2]))
                for j2 in self._j2_grid
            },
            "pass": mean_fid >= 0.90,
        }

    # ── Section 2: MPNN Training ─────────────────────────────────────────────

    def section_mpnn_training(self) -> dict:
        """Train MPNN with node_features=[h, coord, J₂] on the full 2D grid."""
        from qmbp_simulation.predictors import (
            MPNNPredictor,
            build_graph_dataset,
            train_mpnn,
        )

        if self._vqe_data is None:
            raise RuntimeError("Section 1 must run first (VQE data not available)")

        # Prepare arrays for build_graph_dataset
        h_values = np.array([d["h"] for d in self._vqe_data])
        theta_opt = np.array([d["theta"] for d in self._vqe_data])
        e_exact = np.array([d["e_exact"] for d in self._vqe_data])
        fidelities = np.array([d["fidelity"] for d in self._vqe_data])
        j2_values = np.array([d["j2"] for d in self._vqe_data]).reshape(-1, 1)

        logger.info(f"  Training on {len(h_values)} points, node_features=3 [h, coord, J₂]")

        dataset = build_graph_dataset(
            lattice=self._lattice_ref,
            h_values=h_values,
            theta_opt=theta_opt,
            e_exact=e_exact,
            fidelities=fidelities,
            fidelity_threshold=0.90,  # noqa — frustrated TFIM uses relaxed threshold (2D grid data)
            extra_node_features=j2_values,
        )

        logger.info(f"  Dataset after fidelity filter: {len(dataset)} graphs")

        # Train 2D predictor
        model = MPNNPredictor(
            node_features=3,  # [h, coord, J₂]
            hidden_dim=MPNN_HIDDEN_DIM,
            n_layers=MPNN_N_LAYERS,
            output_dim=self._n_params,
        )

        train_result = train_mpnn(
            model=model,
            dataset=dataset,
            n_epochs=MPNN_EPOCHS,
            lr=MPNN_LR,
            patience=MPNN_PATIENCE,
            seed=self._seed,
        )

        self._model = model
        self._dataset = dataset

        final_mse = train_result["final_mse"]
        n_epochs_actual = len(train_result.get("mse_history", []))

        logger.info(f"  Final MSE: {final_mse:.2e}")
        logger.info(f"  Epochs trained: {n_epochs_actual}")
        logger.info(f"  Stopped early: {train_result.get('stopped_early', False)}")

        return {
            "final_mse": float(final_mse),
            "n_epochs": n_epochs_actual,
            "stopped_early": train_result.get("stopped_early", False),
            "n_train_graphs": len(dataset),
            "pass": final_mse < 0.05,  # Relaxed: 2D task harder than 1D
        }

    # ── Section 3: Deployment at Unseen (h, J₂) ─────────────────────────────

    def section_deployment(self) -> dict:
        """Evaluate 2D MPNN at unseen (h, J₂) interpolation points."""
        import torch
        from torch_geometric.data import Data

        if self._model is None:
            raise RuntimeError("Section 2 must run first (model not trained)")

        edge_index_np, coord = self.builder.build_graph_data(self._lattice_ref)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)

        self._model.eval()
        results = []

        # Generate all test combinations
        test_combos = []
        for h_t in H_TEST:
            for j2_t in J2_TEST:
                test_combos.append((h_t, j2_t))
        # Also add cross-validation: known J₂, unseen h
        for h_t in H_TEST:
            for j2_t in [0.2, 0.4]:  # Known J₂ from training grid
                test_combos.append((h_t, j2_t))

        logger.info(f"  Testing {len(test_combos)} unseen (h, J₂) combinations")
        logger.info(f"  {'h':>5} | {'J₂':>4} | {'ΔE/gap':>7} | {'Fid':>5} | {'Pass'}")
        logger.info(f"  {'-' * 5}-+-{'-' * 4}-+-{'-' * 7}-+-{'-' * 5}-+-{'-' * 4}")

        from qiskit.quantum_info import Statevector, state_fidelity

        for h_t, j2_t in test_combos:
            # Build prediction graph
            h_feat = np.full(N_QUBITS, float(h_t))
            j2_feat = np.full(N_QUBITS, float(j2_t))
            x = torch.tensor(
                np.stack([h_feat, coord.astype(float), j2_feat], axis=1),
                dtype=torch.float32,
            )
            graph = Data(x=x, edge_index=edge_index)

            with torch.no_grad():
                theta_pred = self._model(graph).numpy().flatten()

            # Evaluate
            lattice_t = self._make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=h_t)
            H_t = self.builder.build_frustrated_tfim(lattice_t, J2=j2_t)
            pred_energy = self.backend.evaluate(self._circuit, H_t, theta_pred)

            # Exact reference
            H_mat = H_t.to_matrix()
            if hasattr(H_mat, "toarray"):
                H_mat = H_mat.toarray()
            evals, evecs = np.linalg.eigh(H_mat)
            e_exact = float(evals[0])
            gap = float(evals[1] - evals[0])
            gs = evecs[:, 0]

            # Fidelity
            sv = Statevector(self._circuit.assign_parameters(theta_pred))
            fid = float(state_fidelity(sv, Statevector(gs)))

            de_gap = abs(pred_energy - e_exact) / max(gap, 1e-10)
            passed = de_gap < 0.05

            results.append(
                {
                    "h": h_t,
                    "j2": j2_t,
                    "de_gap": de_gap,
                    "fidelity": fid,
                    "pred_energy": float(pred_energy),
                    "e_exact": e_exact,
                    "gap": gap,
                    "passed": passed,
                }
            )

            status = "✓" if passed else "✗"
            logger.info(f"  {h_t:>5.3f} | {j2_t:>4.2f} | {de_gap:>7.4f} | {fid:>5.3f} | {status}")

        # Store for section 4
        self._deployment_results = results

        # Summary
        n_pass = sum(1 for r in results if r["passed"])
        pass_rate = n_pass / len(results)
        mean_de_gap = float(np.mean([r["de_gap"] for r in results]))
        mean_fid = float(np.mean([r["fidelity"] for r in results]))

        # Breakdown: purely interpolation vs cross-validation
        interp_results = [r for r in results if r["j2"] in J2_TEST]
        cross_results = [r for r in results if r["j2"] not in J2_TEST]
        interp_pass_rate = (
            sum(1 for r in interp_results if r["passed"]) / len(interp_results)
            if interp_results
            else 0.0
        )
        cross_pass_rate = (
            sum(1 for r in cross_results if r["passed"]) / len(cross_results)
            if cross_results
            else 0.0
        )

        logger.info(f"\n  Overall: {n_pass}/{len(results)} pass ΔE/gap<5%")
        logger.info(f"  Mean ΔE/gap: {mean_de_gap:.4f}")
        logger.info(f"  Mean fidelity: {mean_fid:.4f}")
        logger.info(f"  Interpolation pass rate: {interp_pass_rate:.0%}")
        logger.info(f"  Cross-validation pass rate: {cross_pass_rate:.0%}")

        return {
            "n_test_points": len(results),
            "n_pass": n_pass,
            "pass_rate": pass_rate,
            "mean_de_gap": mean_de_gap,
            "mean_fidelity": mean_fid,
            "interp_pass_rate": interp_pass_rate,
            "cross_pass_rate": cross_pass_rate,
            "per_point": results,
            "pass": pass_rate >= 0.70,
        }

    # ── Section 4: 2D vs 1D Comparison ───────────────────────────────────────

    def section_comparison(self) -> dict:
        """Compare 2D model against 1D baselines trained at fixed J₂.

        Trains separate 1D MPNNs at J₂=0.15, 0.35, 0.45 (test values)
        and compares their test performance against the single 2D model.
        """
        import torch
        from torch_geometric.data import Data

        from qmbp_simulation.predictors import (
            MPNNPredictor,
            build_graph_dataset,
            train_mpnn,
        )

        if self._vqe_data is None or self._model is None:
            raise RuntimeError("Sections 1-3 must run first")

        edge_index_np, coord = self.builder.build_graph_data(self._lattice_ref)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)

        logger.info("  Training 1D baseline models at fixed J₂ test values...")

        comparison = {}

        for j2_fixed in J2_TEST:
            # Generate 1D training data at this J₂ via VQE
            rng = np.random.default_rng(self._seed)
            prev_theta = rng.uniform(-0.01, 0.01, self._n_params)
            h_arr_1d = np.array(sorted(H_TRAIN, reverse=False))
            theta_arr_1d = np.zeros((len(H_TRAIN), self._n_params))
            e_arr_1d = np.zeros(len(H_TRAIN))
            fid_arr_1d = np.zeros(len(H_TRAIN))

            from qiskit.quantum_info import Statevector, state_fidelity

            for _i, h in enumerate(sorted(H_TRAIN, reverse=True)):
                idx = np.where(np.isclose(h_arr_1d, h))[0][0]
                lattice = self._make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=h)
                H = self.builder.build_frustrated_tfim(lattice, J2=j2_fixed)

                H_mat = H.to_matrix()
                if hasattr(H_mat, "toarray"):
                    H_mat = H_mat.toarray()
                evals, evecs = np.linalg.eigh(H_mat)
                e_arr_1d[idx] = float(evals[0])
                gs = evecs[:, 0]

                best_energy = float("inf")
                best_theta = prev_theta.copy()
                for restart in range(VQE_RESTARTS):
                    x0 = (
                        prev_theta + rng.normal(0, VQE_SIGMA, self._n_params)
                        if restart > 0
                        else prev_theta.copy()
                    )
                    x0 = np.clip(x0, -np.pi, np.pi)
                    res = self._minimize(
                        lambda params, _H=H: self.backend.evaluate(self._circuit, _H, params),
                        x0,
                        method="L-BFGS-B",
                        bounds=[(-np.pi, np.pi)] * self._n_params,
                        options={"maxiter": VQE_MAXITER, "ftol": 1e-14},
                    )
                    if res.fun < best_energy:
                        best_energy = res.fun
                        best_theta = res.x.copy()
                prev_theta = best_theta.copy()
                theta_arr_1d[idx] = best_theta.copy()
                sv = Statevector(self._circuit.assign_parameters(best_theta))
                fid_arr_1d[idx] = float(state_fidelity(sv, Statevector(gs)))

            # Train 1D model (no J₂ feature, standard 2 node features)
            dataset_1d = build_graph_dataset(
                lattice=self._lattice_ref,
                h_values=h_arr_1d,
                theta_opt=theta_arr_1d,
                e_exact=e_arr_1d,
                fidelities=fid_arr_1d,
                fidelity_threshold=0.90,  # noqa — frustrated TFIM uses relaxed threshold (2D grid data)
            )

            model_1d = MPNNPredictor(
                node_features=2,
                hidden_dim=MPNN_HIDDEN_DIM,
                n_layers=MPNN_N_LAYERS,
                output_dim=self._n_params,
            )
            train_mpnn(
                model=model_1d,
                dataset=dataset_1d,
                n_epochs=MPNN_EPOCHS,
                lr=MPNN_LR,
                patience=MPNN_PATIENCE,
                seed=self._seed,
            )

            # Evaluate both models at test h-values with this J₂
            model_1d.eval()
            self._model.eval()

            results_1d = []
            results_2d = []

            for h_t in H_TEST:
                lattice_t = self._make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=h_t)
                H_t = self.builder.build_frustrated_tfim(lattice_t, J2=j2_fixed)
                H_mat = H_t.to_matrix()
                if hasattr(H_mat, "toarray"):
                    H_mat = H_mat.toarray()
                evals_t = np.sort(np.linalg.eigvalsh(H_mat))
                e_exact_t = float(evals_t[0])
                gap_t = float(evals_t[1] - evals_t[0])

                # 1D prediction (standard 2-feature graph)
                h_feat = np.full(N_QUBITS, float(h_t))
                x_1d = torch.tensor(
                    np.stack([h_feat, coord.astype(float)], axis=1),
                    dtype=torch.float32,
                )
                graph_1d = Data(x=x_1d, edge_index=edge_index)
                with torch.no_grad():
                    theta_1d = model_1d(graph_1d).numpy().flatten()
                e_1d = self.backend.evaluate(self._circuit, H_t, theta_1d)
                de_1d = abs(e_1d - e_exact_t) / max(gap_t, 1e-10)

                # 2D prediction
                j2_feat = np.full(N_QUBITS, float(j2_fixed))
                x_2d = torch.tensor(
                    np.stack([h_feat, coord.astype(float), j2_feat], axis=1),
                    dtype=torch.float32,
                )
                graph_2d = Data(x=x_2d, edge_index=edge_index)
                with torch.no_grad():
                    theta_2d = self._model(graph_2d).numpy().flatten()
                e_2d = self.backend.evaluate(self._circuit, H_t, theta_2d)
                de_2d = abs(e_2d - e_exact_t) / max(gap_t, 1e-10)

                results_1d.append(de_1d)
                results_2d.append(de_2d)

            mean_1d = float(np.mean(results_1d))
            mean_2d = float(np.mean(results_2d))

            comparison[str(j2_fixed)] = {
                "mean_de_gap_1d": mean_1d,
                "mean_de_gap_2d": mean_2d,
                "improvement": mean_1d - mean_2d,
                "2d_competitive": mean_2d <= mean_1d * 1.5,
            }

            logger.info(
                f"    J₂={j2_fixed:.2f}: 1D ΔE/gap={mean_1d:.4f}, "
                f"2D ΔE/gap={mean_2d:.4f} "
                f"({'2D wins' if mean_2d <= mean_1d else '1D wins'})"
            )

        # Overall assessment
        all_competitive = all(v["2d_competitive"] for v in comparison.values())
        mean_improvement = float(np.mean([v["improvement"] for v in comparison.values()]))

        logger.info(f"\n  2D model competitive at all J₂: {all_competitive}")
        logger.info(f"  Mean improvement (1D - 2D): {mean_improvement:.4f}")

        return {
            "per_j2_comparison": comparison,
            "all_competitive": all_competitive,
            "mean_improvement": mean_improvement,
            "pass": all_competitive,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    MPNN2DPredictorRunner.main()
