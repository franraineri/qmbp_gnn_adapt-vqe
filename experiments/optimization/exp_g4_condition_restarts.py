"""G4: Condition Number vs Required Restarts.

Hypothesis: The Hessian condition number κ predicts VQE restart needs.
κ < 100 → 1 restart sufficient; κ > 500 → 3+ restarts needed.

Method:
    For N=6, h ∈ {1.0, 1.25, 1.5, 1.75, 2.0}:
        1. Compute κ at the converged minimum (from B4 data)
        2. Run VQE with 1, 3, 5 restarts (10 trials each)
        3. Measure success rate (ΔE/gap < 5%) per restart count
        4. Correlate κ with minimum restarts needed

Thesis value: MEDIUM — adaptive VQE strategy.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from qmbp_simulation.framework import BaseExperiment, ExperimentConfig
from qmbp_simulation.framework.config import AnalysisConfig, SystemConfig, VQEConfig
from qmbp_simulation.framework.metrics import ExperimentMetrics

logger = logging.getLogger(__name__)


class ExperimentG4(BaseExperiment):
    """Condition number vs restart requirements."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="G4",
            category="G",
            description="Condition number κ predicts VQE restart needs",
            hypothesis=(
                "κ < 100 → 1 restart sufficient; κ > 500 → 3+ restarts. "
                "Enables adaptive restart allocation."
            ),
            system=SystemConfig(
                n_qubits=6,
                p_layers=2,
                h_values=[2.0, 1.75, 1.5, 1.25, 1.0],
                h_test=[],  # G4 evaluates at all h_values (no MPNN deployment)
            ),
            vqe=VQEConfig(maxiter=500),
            analysis=AnalysisConfig(compute_hessian=True),
            seeds=DEFAULT_SEEDS,
            verbose=True,
            auto_warm_cold_comparison=False,
        )

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Test restart success rate at each h-value."""
        from scipy.optimize import minimize

        np.random.seed(seed)
        N = self.config.system.n_qubits
        n_params = self.circuit.num_parameters
        h_values = self.config.system.h_values
        n_trials = 10  # Random inits per restart-count test

        metrics = []
        for h in h_values:
            t0 = time.time()
            sol = self.get_exact_solution(h)
            H = sol["hamiltonian"]
            e_exact = sol["exact"].ground_energy
            gap = (
                sol["exact"].gap if sol["exact"].gap > 1e-10 else max(2 * abs(1 - h), 2 * np.pi / N)
            )

            # Compute Hessian condition number at the minimum
            # First find the minimum with many restarts
            best_e, best_theta = float("inf"), np.zeros(n_params)
            for _ in range(10):
                x0 = np.random.uniform(-np.pi, np.pi, n_params)
                res = minimize(
                    lambda p, _H=H: self.backend.evaluate(self.circuit, _H, p),
                    x0,
                    method="L-BFGS-B",
                    bounds=[(-np.pi, np.pi)] * n_params,
                    options={"maxiter": 1000, "ftol": 1e-14},
                )
                if res.fun < best_e:
                    best_e = res.fun
                    best_theta = res.x.copy()

            # Hessian at minimum
            from experiments.helpers.hessian_restart import compute_hessian

            hess = compute_hessian(
                lambda p, _H=H: self.backend.evaluate(self.circuit, _H, p),
                best_theta,
            )
            eigvals = np.linalg.eigvalsh(hess)
            pos_eigvals = eigvals[eigvals > 1e-8]
            kappa = float(pos_eigvals[-1] / pos_eigvals[0]) if len(pos_eigvals) >= 2 else 1.0

            # Test success rate for different restart counts
            restart_counts = [1, 3, 5]
            success_rates = {}

            for n_restarts in restart_counts:
                successes = 0
                for _trial in range(n_trials):
                    trial_best_e = float("inf")
                    for _r in range(n_restarts):
                        x0 = np.random.uniform(-0.5, 0.5, n_params)
                        res = minimize(
                            lambda p, _H=H: self.backend.evaluate(self.circuit, _H, p),
                            x0,
                            method="L-BFGS-B",
                            bounds=[(-np.pi, np.pi)] * n_params,
                            options={"maxiter": self.config.vqe.maxiter, "ftol": 1e-14},
                        )
                        if res.fun < trial_best_e:
                            trial_best_e = res.fun
                    de_gap = abs(trial_best_e - e_exact) / gap
                    if de_gap < 0.05:
                        successes += 1
                success_rates[n_restarts] = successes / n_trials

            elapsed = time.time() - t0
            # Minimum restarts for >80% success
            min_restarts = 5
            for nr in restart_counts:
                if success_rates[nr] >= 0.8:
                    min_restarts = nr
                    break

            m = ExperimentMetrics(
                h_value=h,
                energy=best_e,
                exact_energy=e_exact,
                energy_error=abs(best_e - e_exact),
                gap=gap,
                relative_error=abs(best_e - e_exact) / gap,
                seed=seed,
                wall_time_s=elapsed,
                technique_metadata={
                    "condition_number": kappa,
                    "success_rate_1r": success_rates[1],
                    "success_rate_3r": success_rates[3],
                    "success_rate_5r": success_rates[5],
                    "min_restarts_80pct": min_restarts,
                    "hessian_eigenvalues": eigvals.tolist(),
                },
            )
            metrics.append(m)
            logger.info(
                f"    h={h:.2f}: κ={kappa:.0f}, "
                f"1r={success_rates[1] * 100:.0f}%, "
                f"3r={success_rates[3] * 100:.0f}%, "
                f"5r={success_rates[5] * 100:.0f}%"
            )

        return metrics

    def analyze(self, results: dict[int, list[ExperimentMetrics]]) -> dict:
        from scipy.stats import pearsonr

        analysis = super().analyze(results)
        kappas, min_restarts = [], []
        for _seed, metrics in results.items():
            for m in metrics:
                kappas.append(m.technique_metadata["condition_number"])
                min_restarts.append(m.technique_metadata["min_restarts_80pct"])

        r_val, p_val = pearsonr(kappas, min_restarts) if len(kappas) > 3 else (0, 1)
        analysis["condition_analysis"] = {
            "pearson_r": float(r_val),
            "p_value": float(p_val),
            "kappa_threshold_for_1_restart": max(
                [k for k, mr in zip(kappas, min_restarts, strict=False) if mr == 1], default=0
            ),
        }
        return analysis

    def report(self, analysis: dict) -> str:
        base = super().report(analysis)
        ca = analysis.get("condition_analysis", {})
        lines = [
            base,
            "",
            "Condition Number Analysis:",
            f"  r(κ, min_restarts): {ca.get('pearson_r', 0):.3f}",
            f"  Max κ for 1 restart: {ca.get('kappa_threshold_for_1_restart', 0):.0f}",
        ]
        return "\n".join(lines)
