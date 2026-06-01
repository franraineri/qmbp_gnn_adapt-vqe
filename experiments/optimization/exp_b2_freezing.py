"""B2: TITAN-Style Parameter Freezing for HVA p=2.

Hypothesis: In HVA p=2, the second-layer parameters (θ_zz2, θ_x2) have
|dθ/dh| < threshold for h >= 1.5, meaning they can be frozen after initial
convergence, reducing VQE cost by ~40% with < 1% accuracy loss.

Method:
    1. Analyze θ trajectories to identify frozen parameters
    2. Run VQE with freezing strategy vs full optimization
    3. Compare accuracy and evaluation count

Expected outcome: < 1% accuracy loss with >= 30% fewer evaluations.

Reference: Peng et al. (2025) TITAN, NeurIPS, arXiv:2509.15193.
"""

from __future__ import annotations

import time

import numpy as np

from experiments.helpers.parameter_freezing import (
    analyze_parameter_activity,
    frozen_vqe,
)
from qmbp_simulation.framework import BaseExperiment
from qmbp_simulation.framework.config import ExperimentConfig, SystemConfig, VQEConfig
from qmbp_simulation.framework.metrics import ExperimentMetrics


class ExperimentB2(BaseExperiment):
    """TITAN-style parameter freezing analysis and validation."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="B2",
            category="B",
            description="TITAN parameter freezing: identify and freeze inactive HVA params",
            hypothesis=(
                "Second-layer HVA params (θ_zz2, θ_x2) are frozen for h>=1.5, "
                "enabling 40% VQE cost reduction with <1% accuracy loss."
            ),
            system=SystemConfig(
                n_qubits=6,
                p_layers=2,
                h_values=[2.0, 1.75, 1.5, 1.25, 1.0, 0.8],
                h_test=[],  # B2 evaluates at all h_values (no MPNN deployment)
            ),
            vqe=VQEConfig(n_restarts=5, sigma=0.1),
            seeds=[42, 43, 44],
            verbose=True,
        )

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Phase 1: Generate trajectory. Phase 2: Test freezing."""
        from scipy.optimize import minimize

        np.random.seed(seed)
        n_params = self.circuit.num_parameters
        h_values = sorted(self.config.system.h_values, reverse=True)
        metrics = []

        # Phase 1: Full VQE sweep to get trajectory
        theta_trajectory = []
        prev_theta = np.random.uniform(-0.01, 0.01, n_params)

        for h in h_values:
            sol = self.get_exact_solution(h)
            H = sol["hamiltonian"]

            def cost_fn(params, _H=H):
                return self.evaluate_energy(params, _H)

            best_energy = float("inf")
            best_theta = prev_theta.copy()
            for restart in range(self.config.vqe.n_restarts):
                x0 = (
                    prev_theta + np.random.normal(0, self.config.vqe.sigma, n_params)
                    if restart > 0
                    else prev_theta.copy()
                )
                x0 = np.clip(x0, -np.pi, np.pi)
                result = minimize(
                    cost_fn,
                    x0,
                    method="L-BFGS-B",
                    bounds=[(-np.pi, np.pi)] * n_params,
                    options={"maxiter": 500, "ftol": 1e-14},
                )
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_theta = result.x.copy()

            theta_trajectory.append(best_theta.copy())
            prev_theta = best_theta.copy()

        # Analyze activity
        h_arr = np.array(h_values)
        theta_arr = np.array(theta_trajectory)
        activity = analyze_parameter_activity(h_arr, theta_arr, threshold=0.05)
        frozen_indices = activity["frozen_indices"]

        # Phase 2: Test freezing at each h
        for i, h in enumerate(h_values):
            t0 = time.time()
            sol = self.get_exact_solution(h)
            H = sol["hamiltonian"]
            exact_energy = sol["exact"].ground_energy
            N = self.config.system.n_qubits
            gap = (
                sol["exact"].gap if sol["exact"].gap > 1e-10 else max(2 * abs(1 - h), 2 * np.pi / N)
            )

            def cost_fn_full(params, _H=H):
                return self.evaluate_energy(params, _H)

            # Full VQE (baseline)
            e_full = cost_fn_full(theta_trajectory[i])
            de_gap_full = abs(e_full - exact_energy) / gap

            # Frozen VQE (only if we have frozen params)
            if frozen_indices and h <= 1.75:
                frozen_values = theta_trajectory[i][frozen_indices]
                result_frozen = frozen_vqe(
                    cost_fn_factory=cost_fn_full,
                    initial_guess=theta_trajectory[i],
                    frozen_indices=frozen_indices,
                    frozen_values=frozen_values,
                    n_restarts=self.config.vqe.n_restarts,
                    sigma=self.config.vqe.sigma,
                )
                e_frozen = result_frozen["energy"]
                de_gap_frozen = abs(e_frozen - exact_energy) / gap
            else:
                e_frozen = e_full
                de_gap_frozen = de_gap_full

            elapsed = time.time() - t0

            m = ExperimentMetrics(
                h_value=h,
                energy=e_frozen,
                exact_energy=exact_energy,
                energy_error=abs(e_frozen - exact_energy),
                gap=gap,
                relative_error=de_gap_frozen,
                seed=seed,
                wall_time_s=elapsed,
                technique_metadata={
                    "de_gap_full": float(de_gap_full),
                    "de_gap_frozen": float(de_gap_frozen),
                    "accuracy_loss_pct": float(
                        (de_gap_frozen - de_gap_full) / max(de_gap_full, 1e-10) * 100
                    ),
                    "frozen_indices": frozen_indices,
                    "n_frozen": len(frozen_indices),
                    "n_active": n_params - len(frozen_indices),
                },
            )
            metrics.append(m)

        return metrics
