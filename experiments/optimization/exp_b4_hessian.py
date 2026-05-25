"""B4: Hessian-Guided Adaptive Restarts.

Hypothesis: Computing the 4x4 Hessian at VQE convergence identifies saddle
points and provides escape directions, reducing restarts from 5 to 2-3
while maintaining accuracy.

Method:
    Compare standard 5-restart VQE vs Hessian-guided adaptive restart
    at N=6 and N=10 across multiple h-values.

Expected outcome: Same accuracy with 2-3 restarts (40-60% fewer evaluations).

References:
    - Cerezo et al. (2021) Nature Comms 12, 1791 — landscape structure
    - Wiersema et al. (2020) PRX Quantum 1, 020319 — HVA landscape
"""

from __future__ import annotations

import time

import numpy as np

from experiments.helpers.hessian_restart import (
    hessian_guided_vqe,
    standard_multistart_vqe,
)
from qmbp_simulation.framework import BaseExperiment
from qmbp_simulation.framework.config import ExperimentConfig, SystemConfig, VQEConfig
from qmbp_simulation.framework.metrics import ExperimentMetrics


class ExperimentB4(BaseExperiment):
    """Hessian-guided adaptive restarts vs standard multi-start."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="B4",
            category="B",
            description="Hessian-guided adaptive restarts reduce VQE cost",
            hypothesis=(
                "Hessian analysis at convergence identifies saddle points, "
                "enabling escape with 2-3 restarts instead of 5 blind restarts."
            ),
            system=SystemConfig(
                n_qubits=6,
                p_layers=2,
                h_values=[1.0, 1.25, 1.5, 2.0],
            ),
            vqe=VQEConfig(n_restarts=5, use_hessian_check=True),
            seeds=[42, 43, 44],
            verbose=True,
        )

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Compare Hessian-guided vs standard multi-start at each h."""
        from qiskit.primitives import StatevectorEstimator

        np.random.seed(seed)
        estimator = StatevectorEstimator()
        n_params = self.circuit.num_parameters
        h_values = self.config.system.h_values
        metrics = []

        prev_theta = np.random.uniform(-0.01, 0.01, n_params)

        for h in sorted(h_values, reverse=True):  # Descending
            t0 = time.time()
            sol = self.get_exact_solution(h)
            H = sol["hamiltonian"]
            exact_energy = sol["exact"].ground_energy
            N = self.config.system.n_qubits
            gap = (
                sol["exact"].gap if sol["exact"].gap > 1e-10 else max(2 * abs(1 - h), 2 * np.pi / N)
            )

            def cost_fn(params, _H=H):
                bound = self.circuit.assign_parameters(params)
                job = estimator.run([(bound, _H)])
                return float(job.result()[0].data.evs)

            # --- Standard multi-start (baseline) ---
            result_standard = standard_multistart_vqe(
                cost_fn,
                prev_theta,
                n_restarts=self.config.vqe.n_restarts,
                sigma=self.config.vqe.sigma,
            )

            # --- Hessian-guided ---
            result_hessian = hessian_guided_vqe(
                cost_fn,
                prev_theta,
                max_restarts=self.config.vqe.n_restarts,
                escape_scale=0.3,
            )

            # --- Cold-start (random, 5 restarts) ---
            cold_init = np.random.uniform(-0.5, 0.5, n_params)
            result_cold = standard_multistart_vqe(
                cost_fn,
                cold_init,
                n_restarts=self.config.vqe.n_restarts,
                sigma=0.3,
            )

            # Use best overall as reference
            best_theta = (
                result_hessian["theta_opt"]
                if result_hessian["energy"] <= result_standard["energy"]
                else result_standard["theta_opt"]
            )
            prev_theta = best_theta.copy()

            de_gap_standard = abs(result_standard["energy"] - exact_energy) / gap
            de_gap_hessian = abs(result_hessian["energy"] - exact_energy) / gap
            de_gap_cold = abs(result_cold["energy"] - exact_energy) / gap

            elapsed = time.time() - t0

            m = ExperimentMetrics(
                h_value=h,
                energy=result_hessian["energy"],
                exact_energy=exact_energy,
                energy_error=abs(result_hessian["energy"] - exact_energy),
                gap=gap,
                relative_error=de_gap_hessian,
                seed=seed,
                wall_time_s=elapsed,
                n_restarts_used=result_hessian["n_restarts_used"],
                n_evaluations=result_hessian["total_evaluations"],
                hessian_eigenvalues=(
                    result_hessian["hessian_eigenvalues"]
                    if isinstance(result_hessian["hessian_eigenvalues"], list)
                    else None
                ),
                technique_metadata={
                    "standard_de_gap": float(de_gap_standard),
                    "hessian_de_gap": float(de_gap_hessian),
                    "cold_de_gap": float(de_gap_cold),
                    "standard_restarts": self.config.vqe.n_restarts,
                    "hessian_restarts_used": result_hessian["n_restarts_used"],
                    "standard_evals": result_standard["total_evaluations"],
                    "hessian_evals": result_hessian["total_evaluations"],
                    "eval_savings_pct": float(
                        (result_standard["total_evaluations"] - result_hessian["total_evaluations"])
                        / max(result_standard["total_evaluations"], 1)
                        * 100
                    ),
                    "is_true_minimum": result_hessian["is_true_minimum"],
                    "convergence_history": result_hessian["convergence_history"],
                    "warm_vs_cold_gain_pct": float(
                        (de_gap_cold - de_gap_hessian) / max(de_gap_cold, 1e-10) * 100
                    ),
                },
            )
            metrics.append(m)

        return metrics
