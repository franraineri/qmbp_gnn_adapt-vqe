"""S4: Data Efficiency at N=10 (Extension of G1).

Hypothesis: The minimum number of training points k_min scales with N.
At N=10, k_min is expected to be 11-13 (vs 9 at N=6), reflecting the
more complex θ(h) landscape at larger system sizes.

Method:
    1. Use existing VQE data at N=10 (17-point grid)
    2. Subsample to k points (k = 5, 7, 9, 11, 13, 15, 17)
    3. Train MPNN on each subset
    4. Deploy at h_test and measure ΔE/gap
    5. Find k_min where all seeds pass 5%

Expected outcome:
    k_min(N=10) ≈ 11-13 (vs k_min(N=6) = 9).
    Seeds 43/44 may pass with fewer points (same pattern as G1).

Thesis value: MEDIUM — quantifies minimum pipeline cost at N=10.

References:
    - G1: k_min(N=6) = 9 (47% reduction from 17)
    - poc-results.md: N=10 optimal config (h=128, patience=500)
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

# Standard 17-point h-grid for N=10 (descending, within valid regime)
FULL_H_GRID = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

# Subsampling sizes to test
K_VALUES = [5, 7, 9, 11, 13, 15, 17]


class ExperimentS4(BaseExperiment):
    """Data efficiency curve at N=10."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="S4",
            category="S",
            description="Data efficiency at N=10: find k_min training points",
            hypothesis=(
                "k_min(N=10) ≈ 11-13 points for ΔE/gap < 5% "
                "(vs k_min(N=6) = 9). Scaling of k_min with N."
            ),
            system=SystemConfig(
                n_qubits=10,
                p_layers=2,
                topology="chain_1d",
                h_values=FULL_H_GRID,
                h_test=[1.55],
            ),
            vqe=VQEConfig(n_restarts=5, maxiter=1000, sigma=0.1),
            mpnn=MPNNConfig(hidden_dim=128, n_layers=3, n_epochs=6000, lr=1e-3, patience=500),
            seeds=[42, 43, 44],
            verbose=True,
        )

    def setup(self) -> None:
        """Setup for N=10 VQE + MPNN."""
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
        logger.info("S4 setup: N=%d, p=%d, k_values=%s", N, p, K_VALUES)

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Run VQE sweep, then test MPNN with varying training set sizes."""
        from scipy.optimize import minimize

        from experiments.helpers.graph_utils import build_experiment_dataset, predict_theta
        from qmbp_simulation import make_lattice
        from qmbp_simulation.predictors import MPNNPredictor, train_mpnn

        np.random.seed(seed)
        torch.manual_seed(seed)

        N = self.config.system.n_qubits
        n_params = self.circuit.num_parameters
        h_test = self.config.system.h_test[0]
        metrics = []

        # Phase 2: Full VQE sweep (17 points)
        logger.info("  Phase 2: VQE sweep (%d points)...", len(FULL_H_GRID))
        t_vqe_start = time.time()
        vqe_data: dict[float, np.ndarray] = {}
        prev_theta = np.random.uniform(-0.01, 0.01, n_params)

        for h in sorted(FULL_H_GRID, reverse=True):
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
                    lambda params, _H=H: self._evaluate_energy(params, _H),
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
        logger.info("  VQE complete: %.1fs", t_vqe)

        # Get exact solution at test point
        lattice_test = make_lattice("chain_1d", N, J=1.0, h=h_test)
        H_test = self.builder.build(lattice_test)
        result_test = self.solver.solve(H_test, lattice_test)
        exact_energy = result_test.ground_energy
        gap = result_test.gap

        # Phase 3+4: For each k, subsample → train → deploy
        for k in K_VALUES:
            # Uniform subsampling from the full grid
            h_subset = self._subsample_uniform(FULL_H_GRID, k)
            h_sorted = sorted(h_subset)
            theta_array = np.array([vqe_data[h] for h in h_sorted])

            # Train MPNN
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

            # Deploy at h_test
            theta_pred = predict_theta(self, model, h_test)

            e_pred = self._evaluate_energy(theta_pred, H_test)
            de_gap = abs(e_pred - exact_energy) / max(gap, 1e-10)

            m = ExperimentMetrics(
                h_value=h_test,
                energy=e_pred,
                exact_energy=exact_energy,
                energy_error=abs(e_pred - exact_energy),
                gap=gap,
                relative_error=de_gap,
                seed=seed,
                wall_time_s=t_vqe,
                converged=de_gap < 0.05,
                technique_metadata={
                    "k": k,
                    "h_subset": h_sorted,
                    "de_gap": float(de_gap),
                    "pass_5pct": de_gap < 0.05,
                },
            )
            metrics.append(m)
            status = "✅" if de_gap < 0.05 else "❌"
            logger.info("  k=%d: ΔE/gap=%.4f %s", k, de_gap, status)

        return metrics

    def _evaluate_energy(self, params, H) -> float:
        """Evaluate energy using StatevectorEstimator."""
        from qiskit.primitives import StatevectorEstimator

        estimator = StatevectorEstimator()
        bound = self.circuit.assign_parameters(params)
        job = estimator.run([(bound, H)])
        return float(job.result()[0].data.evs)

    @staticmethod
    def _subsample_uniform(h_grid: list[float], k: int) -> list[float]:
        """Uniformly subsample k points from h_grid (always include endpoints)."""
        if k >= len(h_grid):
            return h_grid[:]
        indices = np.linspace(0, len(h_grid) - 1, k, dtype=int)
        return [h_grid[i] for i in indices]
