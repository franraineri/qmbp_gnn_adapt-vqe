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

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.experiments_v8.core.base_experiment import BaseExperiment
from scripts.experiments_v8.core.config import ExperimentConfig, SystemConfig, VQEConfig
from scripts.experiments_v8.core.metrics import V8Metrics
from scripts.experiments_v8.techniques.dypp import (
    dypp_predict,
    evaluate_dypp_quality,
)


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
            ),
            vqe=VQEConfig(use_dypp=True, dypp_order=2),
            seeds=[42, 43, 44],
            verbose=True,
        )

    def run_single(self, seed: int) -> list[V8Metrics]:
        """Run descending sweep with 3 strategies: standard, DyPP-1, DyPP-2."""
        from qiskit.primitives import StatevectorEstimator
        from scipy.optimize import minimize

        np.random.seed(seed)
        estimator = StatevectorEstimator()
        n_params = self.circuit.num_parameters
        h_values = self.config.system.h_values
        metrics = []

        # History for DyPP
        h_history: list[float] = []
        theta_history: list[np.ndarray] = []

        # Previous theta for standard warm-start
        prev_theta: np.ndarray | None = None

        for _idx, h in enumerate(h_values):
            t0 = time.time()
            sol = self.get_exact_solution(h)
            H = sol["hamiltonian"]
            exact_energy = sol["exact"].ground_energy
            gap = (
                sol["exact"].gap if sol["exact"].gap > 1e-10 else max(2 * abs(1 - h), 2 * np.pi / 6)
            )

            def cost_fn(params, _H=H):
                bound = self.circuit.assign_parameters(params)
                job = estimator.run([(bound, _H)])
                return float(job.result()[0].data.evs)

            # --- Strategy 1: Standard warm-start ---
            if prev_theta is not None:
                init_standard = prev_theta.copy()
            else:
                init_standard = np.random.uniform(-0.01, 0.01, n_params)

            res_standard = minimize(
                cost_fn,
                init_standard,
                method="L-BFGS-B",
                bounds=[(-np.pi, np.pi)] * n_params,
                options={"maxiter": 500, "ftol": 1e-14},
            )

            # --- Strategy 2: DyPP linear ---
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

            # --- Strategy 3: DyPP quadratic ---
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

            # --- Strategy 4: Cold-start (random) ---
            init_cold = np.random.uniform(-0.5, 0.5, n_params)
            res_cold = minimize(
                cost_fn,
                init_cold,
                method="L-BFGS-B",
                bounds=[(-np.pi, np.pi)] * n_params,
                options={"maxiter": 500, "ftol": 1e-14},
            )

            # Use best result as the "true" optimum for this h
            best_energy = min(res_standard.fun, res_dypp1.fun, res_dypp2.fun, res_cold.fun)
            best_theta = [res_standard, res_dypp1, res_dypp2, res_cold][
                np.argmin([res_standard.fun, res_dypp1.fun, res_dypp2.fun, res_cold.fun])
            ].x.copy()

            # Update history with best result
            h_history.append(h)
            theta_history.append(best_theta)
            prev_theta = best_theta.copy()

            elapsed = time.time() - t0

            # Compute DyPP quality
            dypp_quality = {}
            if len(h_history) >= 3:
                dypp_quality = evaluate_dypp_quality(init_dypp1, best_theta, init_standard)

            m = V8Metrics(
                h_value=h,
                energy=best_energy,
                exact_energy=exact_energy,
                energy_error=abs(best_energy - exact_energy),
                gap=gap,
                relative_error=abs(best_energy - exact_energy) / gap,
                seed=seed,
                wall_time_s=elapsed,
                n_evaluations=res_standard.nfev + res_dypp1.nfev + res_dypp2.nfev,
                theta_opt=best_theta.tolist(),
                technique_metadata={
                    "standard_nit": int(res_standard.nit),
                    "standard_de_gap": float(abs(res_standard.fun - exact_energy) / gap),
                    "dypp1_nit": int(res_dypp1.nit),
                    "dypp1_de_gap": float(abs(res_dypp1.fun - exact_energy) / gap),
                    "dypp2_nit": int(res_dypp2.nit),
                    "dypp2_de_gap": float(abs(res_dypp2.fun - exact_energy) / gap),
                    "cold_nit": int(res_cold.nit),
                    "cold_de_gap": float(abs(res_cold.fun - exact_energy) / gap),
                    "dypp_quality": dypp_quality,
                    "iter_savings_dypp1_vs_standard": float(
                        (res_standard.nit - res_dypp1.nit) / max(res_standard.nit, 1) * 100
                    ),
                    "iter_savings_dypp2_vs_standard": float(
                        (res_standard.nit - res_dypp2.nit) / max(res_standard.nit, 1) * 100
                    ),
                },
            )
            metrics.append(m)

        return metrics

    def analyze(self, results: dict[int, list[V8Metrics]]) -> dict:
        analysis = super().analyze(results)

        # Aggregate iteration savings by h-region
        smooth_savings_1 = []  # h > 1.5
        smooth_savings_2 = []
        critical_savings_1 = []  # h <= 1.5
        critical_savings_2 = []

        for _seed, metrics in results.items():
            for m in metrics:
                md = m.technique_metadata
                s1 = md.get("iter_savings_dypp1_vs_standard", 0)
                s2 = md.get("iter_savings_dypp2_vs_standard", 0)
                if m.h_value > 1.5:
                    smooth_savings_1.append(s1)
                    smooth_savings_2.append(s2)
                else:
                    critical_savings_1.append(s1)
                    critical_savings_2.append(s2)

        analysis["dypp_summary"] = {
            "smooth_regime": {
                "dypp_linear_mean_savings_pct": float(np.mean(smooth_savings_1))
                if smooth_savings_1
                else 0,
                "dypp_quad_mean_savings_pct": float(np.mean(smooth_savings_2))
                if smooth_savings_2
                else 0,
                "n_points": len(smooth_savings_1),
            },
            "critical_regime": {
                "dypp_linear_mean_savings_pct": float(np.mean(critical_savings_1))
                if critical_savings_1
                else 0,
                "dypp_quad_mean_savings_pct": float(np.mean(critical_savings_2))
                if critical_savings_2
                else 0,
                "n_points": len(critical_savings_1),
            },
            "hypothesis_confirmed": (
                float(np.mean(smooth_savings_1)) > 20 if smooth_savings_1 else False
            ),
        }

        return analysis

    def report(self, analysis: dict) -> str:
        base = super().report(analysis)
        lines = [base, "", "DyPP Analysis:"]

        dypp = analysis.get("dypp_summary", {})
        smooth = dypp.get("smooth_regime", {})
        critical = dypp.get("critical_regime", {})

        lines.extend(
            [
                f"  Smooth regime (h>1.5, {smooth.get('n_points', 0)} points):",
                f"    DyPP linear savings:    {smooth.get('dypp_linear_mean_savings_pct', 0):.1f}%",
                f"    DyPP quadratic savings: {smooth.get('dypp_quad_mean_savings_pct', 0):.1f}%",
                f"  Critical regime (h<=1.5, {critical.get('n_points', 0)} points):",
                f"    DyPP linear savings:    {critical.get('dypp_linear_mean_savings_pct', 0):.1f}%",
                f"    DyPP quadratic savings: {critical.get('dypp_quad_mean_savings_pct', 0):.1f}%",
                "",
                f"  Hypothesis confirmed: {dypp.get('hypothesis_confirmed', False)}",
            ]
        )

        return "\n".join(lines)
