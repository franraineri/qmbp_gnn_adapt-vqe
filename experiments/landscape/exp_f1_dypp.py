"""F1: Dynamic Parameter Prediction (DyPP) for VQE Acceleration.

Hypothesis: Linear/quadratic extrapolation from the last 2-3 converged θ(h)
points predicts θ(h_{i+1}) more accurately than standard warm-start (previous h
only), reducing VQE iterations by 30-50% in the smooth regime (h > 1.5).

Method:
    Run descending h-sweep with three initialization strategies:
    1. Standard warm-start: θ_init = θ_opt(h_{i-1})
    2. DyPP linear: extrapolate from last 2 points
    3. DyPP quadratic: extrapolate from last 3 points
    Compare iterations to converge and final accuracy.

Expected outcome: 30-50% fewer iterations for h > 1.5; DyPP fails near h_c.

Reference: arXiv:2307.12449 — Dynamic Parameter Prediction for VQA (2023).
"""

from __future__ import annotations

import time

import numpy as np

from experiments.helpers.dypp import (
    dypp_predict,
    evaluate_dypp_quality,
)
from qmbp_simulation.framework import BaseExperiment
from qmbp_simulation.framework.config import ExperimentConfig, SystemConfig, VQEConfig
from qmbp_simulation.framework.metrics import ExperimentMetrics


class ExperimentF1(BaseExperiment):
    """DyPP extrapolation vs standard warm-start."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="F1",
            category="F",
            description="DyPP extrapolation vs standard warm-start in descending sweep",
            hypothesis=(
                "DyPP linear/quadratic extrapolation reduces VQE iterations by "
                "30-50% in the smooth regime (h>1.5) compared to standard warm-start."
            ),
            system=SystemConfig(
                n_qubits=6,
                p_layers=2,
                h_values=[2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8],
                h_test=[],  # F1 evaluates at all h_values (no MPNN deployment)
            ),
            vqe=VQEConfig(use_dypp=True, dypp_order=2),
            seeds=[42, 43, 44],
            verbose=True,
        )

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Run descending sweep with 3 strategies: standard, DyPP-1, DyPP-2."""
        from scipy.optimize import minimize

        np.random.seed(seed)  # For scipy L-BFGS-B reproducibility
        rng = np.random.default_rng(seed)
        n_params = self.circuit.num_parameters
        h_values = self.config.system.h_values
        metrics = []

        # History for DyPP
        h_history: list[float] = []
        theta_history: list[np.ndarray] = []
        prev_theta: np.ndarray | None = None

        for h in h_values:
            t0 = time.time()
            sol = self.get_exact_solution(h)
            H = sol["hamiltonian"]
            exact_energy = sol["exact"].ground_energy
            gap = (
                sol["exact"].gap
                if sol["exact"].gap > 1e-10
                else max(2 * abs(1 - h), 2 * np.pi / self.config.system.n_qubits)
            )

            def cost_fn(params, _H=H):
                return self.evaluate_energy(params, _H)

            # Strategy 1: Standard warm-start
            if prev_theta is not None:
                init_standard = prev_theta.copy()
            else:
                init_standard = rng.uniform(-0.01, 0.01, n_params)

            res_standard = minimize(
                cost_fn,
                init_standard,
                method="L-BFGS-B",
                bounds=[(-np.pi, np.pi)] * n_params,
                options={"maxiter": 500, "ftol": 1e-14},
            )

            # Strategy 2: DyPP linear
            if len(h_history) >= 2:
                init_dypp1 = dypp_predict(h_history, theta_history, h, order=1)
            else:
                init_dypp1 = init_standard.copy()

            res_dypp1 = minimize(
                cost_fn,
                init_dypp1,
                method="L-BFGS-B",
                bounds=[(-np.pi, np.pi)] * n_params,
                options={"maxiter": 500, "ftol": 1e-14},
            )

            # Strategy 3: DyPP quadratic
            if len(h_history) >= 3:
                init_dypp2 = dypp_predict(h_history, theta_history, h, order=2)
            else:
                init_dypp2 = init_dypp1.copy()

            res_dypp2 = minimize(
                cost_fn,
                init_dypp2,
                method="L-BFGS-B",
                bounds=[(-np.pi, np.pi)] * n_params,
                options={"maxiter": 500, "ftol": 1e-14},
            )

            # Use best result
            best_energy = min(res_standard.fun, res_dypp1.fun, res_dypp2.fun)
            best_theta = [res_standard, res_dypp1, res_dypp2][
                np.argmin([res_standard.fun, res_dypp1.fun, res_dypp2.fun])
            ].x.copy()

            h_history.append(h)
            theta_history.append(best_theta)
            prev_theta = best_theta.copy()

            elapsed = time.time() - t0

            dypp_quality = {}
            if len(h_history) >= 3:
                dypp_quality = evaluate_dypp_quality(init_dypp1, best_theta, init_standard)

            m = ExperimentMetrics(
                h_value=h,
                energy=best_energy,
                exact_energy=exact_energy,
                energy_error=abs(best_energy - exact_energy),
                gap=gap,
                relative_error=abs(best_energy - exact_energy) / gap,
                seed=seed,
                wall_time_s=elapsed,
                technique_metadata={
                    "standard_nit": int(res_standard.nit),
                    "dypp1_nit": int(res_dypp1.nit),
                    "dypp2_nit": int(res_dypp2.nit),
                    "dypp_quality": dypp_quality,
                    "iter_savings_dypp1_vs_standard": float(
                        (res_standard.nit - res_dypp1.nit) / max(res_standard.nit, 1) * 100
                    ),
                },
            )
            metrics.append(m)

        return metrics
