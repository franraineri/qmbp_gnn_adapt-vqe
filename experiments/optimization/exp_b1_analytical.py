"""B1: Analytical Initial Guess from Perturbation Theory.

Hypothesis: For h >> h_c, the optimal HVA parameters can be derived analytically,
providing a deterministic initialization that eliminates seed sensitivity.

Method:
    Derive theta_opt(h) in the h>>1 limit, validate against VQE-optimized theta.

Expected outcome: Analytical guess within 5% of optimal at h>=2.0.
"""

from __future__ import annotations

import time

import numpy as np

from experiments.helpers.analytical_init import (
    analytical_init_p1,
    analytical_init_p2,
)
from qmbp_simulation.framework import BaseExperiment
from qmbp_simulation.framework.config import ExperimentConfig, SystemConfig
from qmbp_simulation.framework.metrics import ExperimentMetrics


class ExperimentB1(BaseExperiment):
    """Validate analytical initialization against VQE-optimized parameters."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="B1",
            category="B",
            description="Analytical initial guess validation (perturbation theory)",
            hypothesis=(
                "Analytical theta from perturbation theory is within 5% of VQE-optimal "
                "at h>=2.0, eliminating seed sensitivity for the first sweep point."
            ),
            system=SystemConfig(
                n_qubits=6,
                p_layers=2,
                h_values=[1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0],
            ),
            seeds=[42, 43, 44],
            verbose=True,
        )

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Compare analytical init vs VQE-optimized at each h."""
        from scipy.optimize import minimize

        np.random.seed(seed)
        p = self.config.system.p_layers
        N = self.config.system.n_qubits
        n_params = self.circuit.num_parameters

        metrics = []

        for h in self.config.system.h_values:
            t0 = time.time()
            sol = self.get_exact_solution(h)
            H = sol["hamiltonian"]
            exact_energy = sol["exact"].ground_energy
            gap = (
                sol["exact"].gap if sol["exact"].gap > 1e-10 else max(2 * abs(1 - h), 2 * np.pi / N)
            )

            # 1. Analytical init
            theta_analytical = analytical_init_p1(h) if p == 1 else analytical_init_p2(h)
            e_analytical = self.evaluate_energy(theta_analytical, H)
            de_analytical = abs(e_analytical - exact_energy) / gap

            # 2. VQE from analytical init
            def cost_fn(params, _H=H):
                return self.evaluate_energy(params, _H)

            result_from_analytical = minimize(
                cost_fn,
                theta_analytical,
                method="L-BFGS-B",
                bounds=[(-np.pi, np.pi)] * n_params,
                options={"maxiter": 500, "ftol": 1e-14},
            )
            e_from_analytical = result_from_analytical.fun
            nit_from_analytical = result_from_analytical.nit

            # 3. VQE from random init (baseline)
            theta_random = np.random.uniform(-0.01, 0.01, n_params)
            result_from_random = minimize(
                cost_fn,
                theta_random,
                method="L-BFGS-B",
                bounds=[(-np.pi, np.pi)] * n_params,
                options={"maxiter": 500, "ftol": 1e-14},
            )
            nit_from_random = result_from_random.nit

            # 4. Best VQE (5 restarts from random — gold standard)
            best_energy = result_from_random.fun
            for _ in range(4):
                x0 = np.random.uniform(-0.5, 0.5, n_params)
                trial = minimize(
                    cost_fn,
                    x0,
                    method="L-BFGS-B",
                    bounds=[(-np.pi, np.pi)] * n_params,
                    options={"maxiter": 500, "ftol": 1e-14},
                )
                if trial.fun < best_energy:
                    best_energy = trial.fun

            de_gap_from_analytical = abs(e_from_analytical - exact_energy) / gap
            de_gap_from_random = abs(result_from_random.fun - exact_energy) / gap

            elapsed = time.time() - t0

            m = ExperimentMetrics(
                h_value=h,
                energy=e_from_analytical,
                exact_energy=exact_energy,
                energy_error=abs(e_from_analytical - exact_energy),
                gap=gap,
                relative_error=de_gap_from_analytical,
                seed=seed,
                wall_time_s=elapsed,
                n_evaluations=nit_from_analytical,
                theta_init=theta_analytical.tolist(),
                technique_metadata={
                    "de_gap_analytical_raw": float(de_analytical),
                    "de_gap_from_analytical_vqe": float(de_gap_from_analytical),
                    "de_gap_from_random_vqe": float(de_gap_from_random),
                    "nit_from_analytical": int(nit_from_analytical),
                    "nit_from_random": int(nit_from_random),
                    "iteration_savings_pct": float(
                        (nit_from_random - nit_from_analytical) / max(nit_from_random, 1) * 100
                    ),
                },
            )
            metrics.append(m)

        return metrics
