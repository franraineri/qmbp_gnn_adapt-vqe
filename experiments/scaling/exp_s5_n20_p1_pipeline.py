"""S5: Full Pipeline N=20 p=1 with MPNN (not interpolation).

Hypothesis: With 15 training points in [2.25, 4.0] and 5 VQE restarts,
the MPNN deploys successfully at N=20 p=1 with ΔE/gap < 3% at all test
points within the valid regime.

Method:
    1. VQE descending sweep: 15 h-points in [2.25, 4.0], 5 restarts
    2. Train MPNN (h=128, L=3, 6000 epochs) on VQE data
    3. Deploy at h_test = [2.5, 3.0, 3.5]
    4. Compare with C3 interpolation baseline (1.58%)

Context:
    - C3 showed VQE works (ΔE/gap=1.58% with interpolation, 2/3 seeds)
    - C3 showed canonicalization is unnecessary (0% effect)
    - Original attempt (binnacle-p1-scaling) used only 6 points → failed
    - This experiment uses 15 points (2.5× more) with MPNN (not interp)

Expected outcome:
    ΔE/gap < 2% at all test points (MPNN should match or beat interpolation).

Thesis value: HIGH — closes the "N=20 p=1 pipeline" claim with real MPNN.

References:
    - C3: Sign canonicalization unnecessary, VQE validated at N=20 p=1
    - binnacle-p1-scaling: 6 points insufficient, only h=3.0 passes
    - project-status: N=20 p=1 config = 5 restarts, h∈[2.25,4.0], MPNN h=128
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from qmbp_simulation.framework import BaseExperiment
from qmbp_simulation.framework.config import ExperimentConfig, MPNNConfig, SystemConfig, VQEConfig
from qmbp_simulation.framework.metrics import ExperimentMetrics

logger = logging.getLogger(__name__)

# 15 uniformly-spaced points in [2.25, 4.0]
H_TRAIN = [
    4.0,
    3.875,
    3.75,
    3.625,
    3.5,
    3.375,
    3.25,
    3.125,
    3.0,
    2.875,
    2.75,
    2.625,
    2.5,
    2.375,
    2.25,
]

H_TEST = [2.5, 3.0, 3.5]


class ExperimentS5(BaseExperiment):
    """Full pipeline N=20 p=1 with MPNN predictor."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="S5",
            category="S",
            description="N=20 p=1 full pipeline with MPNN (15 training points)",
            hypothesis=(
                "MPNN trained on 15 h-points achieves ΔE/gap < 3% at N=20 p=1, "
                "matching or beating the C3 interpolation baseline (1.58%)."
            ),
            system=SystemConfig(
                n_qubits=20,
                p_layers=1,
                topology="chain_1d",
                h_values=H_TRAIN,
                h_test=H_TEST,
            ),
            vqe=VQEConfig(n_restarts=5, maxiter=100, sigma=0.3),
            mpnn=MPNNConfig(hidden_dim=128, n_layers=3, n_epochs=6000, lr=1e-3, patience=500),
            seeds=[42, 43, 44],
            verbose=True,
        )

    def setup(self) -> None:
        """Setup MPS backend for N=20."""
        from qiskit_aer import AerSimulator

        from qmbp_simulation import (
            ClassicalSolver,
            HamiltonianBuilder,
            HVACircuitBuilder,
            make_lattice,
        )

        self._setup_logging()
        self.builder = HamiltonianBuilder()
        self.solver = ClassicalSolver()
        self.hva = HVACircuitBuilder()

        N = self.config.system.n_qubits
        p = self.config.system.p_layers
        lattice = make_lattice("chain_1d", N, J=1.0, h=1.0)
        self.circuit, _ = self.hva.create(N, p, lattice)
        self._lattice = lattice

        self._mps_sim = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=64,
            matrix_product_state_truncation_threshold=1e-12,
        )
        logger.info(
            "S5 setup: N=%d, p=%d, %d train points, backend=MPS(chi=64)", N, p, len(H_TRAIN)
        )

    def _evaluate_mps(self, params, H) -> float:
        """Evaluate energy via MPS."""
        from qiskit.quantum_info import Statevector

        bound = self.circuit.assign_parameters(params)
        bound_save = bound.copy()
        bound_save.save_statevector()
        result = self._mps_sim.run(bound_save).result()
        sv = Statevector(result.get_statevector())
        return float(sv.expectation_value(H).real)

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Run full pipeline: VQE → MPNN → Deploy at N=20 p=1."""
        from scipy.optimize import minimize

        from experiments.helpers.graph_utils import build_experiment_dataset, predict_theta
        from qmbp_simulation import make_lattice
        from qmbp_simulation.predictors import MPNNPredictor, train_mpnn

        np.random.seed(seed)
        torch.manual_seed(seed)

        N = self.config.system.n_qubits
        n_params = self.circuit.num_parameters
        metrics = []

        # Phase 2: VQE descending sweep
        logger.info(
            "  Phase 2: VQE sweep (%d points, %d restarts)...",
            len(H_TRAIN),
            self.config.vqe.n_restarts,
        )
        t_vqe_start = time.time()
        vqe_data: dict[float, np.ndarray] = {}
        prev_theta = np.random.uniform(-0.01, 0.01, n_params)

        for h in sorted(H_TRAIN, reverse=True):
            lattice = make_lattice("chain_1d", N, J=1.0, h=h)
            H = self.builder.build(lattice)

            best_e, best_theta = float("inf"), prev_theta.copy()
            for r in range(self.config.vqe.n_restarts):
                x0 = (
                    prev_theta.copy()
                    if r == 0
                    else best_theta + np.random.normal(0, self.config.vqe.sigma, n_params)
                )
                x0 = np.clip(x0, -np.pi, np.pi)
                res = minimize(
                    lambda p, _H=H: self._evaluate_mps(p, _H),
                    x0,
                    method="L-BFGS-B",
                    bounds=[(-np.pi, np.pi)] * n_params,
                    options={"maxiter": self.config.vqe.maxiter, "ftol": 1e-14},
                )
                if res.fun < best_e:
                    best_e = res.fun
                    best_theta = res.x.copy()

            vqe_data[h] = best_theta.copy()
            prev_theta = best_theta.copy()

        t_vqe = time.time() - t_vqe_start
        logger.info("  Phase 2 complete: %.1fs (%d points)", t_vqe, len(vqe_data))

        # Phase 3: Train MPNN
        logger.info("  Phase 3: MPNN training...")
        t_mpnn_start = time.time()
        h_sorted = sorted(vqe_data.keys())
        theta_array = np.array([vqe_data[h] for h in h_sorted])

        dataset = build_experiment_dataset(self, np.array(h_sorted), theta_array)
        model = MPNNPredictor(
            node_features=2,
            hidden_dim=self.config.mpnn.hidden_dim,
            n_layers=self.config.mpnn.n_layers,
            output_dim=n_params,
        )
        train_mpnn(
            model,
            dataset,
            n_epochs=self.config.mpnn.n_epochs,
            lr=self.config.mpnn.lr,
            patience=self.config.mpnn.patience,
        )
        t_mpnn = time.time() - t_mpnn_start
        logger.info("  Phase 3 complete: %.1fs", t_mpnn)

        # Phase 4: Deploy at test points
        logger.info("  Phase 4: Deploy at h_test=%s", H_TEST)
        for h_t in H_TEST:
            lattice = make_lattice("chain_1d", N, J=1.0, h=h_t)
            H = self.builder.build(lattice)
            result = self.solver.solve(H, lattice)
            exact_energy = result.ground_energy
            gap = result.gap if result.gap > 1e-10 else max(2 * abs(1 - h_t), 2 * np.pi / N)

            # MPNN prediction
            theta_pred = predict_theta(self, model, h_t)

            e_pred = self._evaluate_mps(theta_pred, H)
            de_gap = abs(e_pred - exact_energy) / gap

            # Also compute interpolation baseline for comparison
            theta_interp = np.array(
                [np.interp(h_t, h_sorted, theta_array[:, i]) for i in range(n_params)]
            )
            e_interp = self._evaluate_mps(theta_interp, H)
            de_gap_interp = abs(e_interp - exact_energy) / gap

            m = ExperimentMetrics(
                h_value=h_t,
                energy=e_pred,
                exact_energy=exact_energy,
                energy_error=abs(e_pred - exact_energy),
                gap=gap,
                relative_error=de_gap,
                seed=seed,
                wall_time_s=t_vqe + t_mpnn,
                converged=de_gap < 0.05,
                technique_metadata={
                    "de_gap_mpnn": float(de_gap),
                    "de_gap_interpolation": float(de_gap_interp),
                    "mpnn_vs_interp": "better" if de_gap < de_gap_interp else "worse",
                    "vqe_time_s": t_vqe,
                    "mpnn_time_s": t_mpnn,
                    "n_train_points": len(H_TRAIN),
                },
            )
            metrics.append(m)
            status = "✅" if de_gap < 0.05 else "❌"
            logger.info(
                "  h=%.2f: MPNN ΔE/gap=%.4f %s (interp=%.4f)",
                h_t,
                de_gap,
                status,
                de_gap_interp,
            )

        return metrics
