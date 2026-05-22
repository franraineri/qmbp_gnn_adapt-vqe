"""B1: Analytical Initial Guess from Perturbation Theory.

Hypothesis: For h >> h_c, the optimal HVA parameters can be derived analytically,
providing a deterministic initialization that eliminates seed sensitivity.

Method:
    Derive theta_opt(h) in the h>>1 limit, validate against VQE-optimized theta.

Expected outcome: Analytical guess within 5% of optimal at h>=2.0.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.experiments_v8.core.base_experiment import BaseExperiment
from scripts.experiments_v8.core.config import ExperimentConfig, SystemConfig
from scripts.experiments_v8.core.metrics import V8Metrics
from scripts.experiments_v8.techniques.analytical_init import (
    analytical_init_p1,
    analytical_init_p2,
)


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

    def run_single(self, seed: int) -> list[V8Metrics]:
        """Compare analytical init vs VQE-optimized at each h."""
        from qiskit.primitives import StatevectorEstimator
        from scipy.optimize import minimize

        np.random.seed(seed)
        estimator = StatevectorEstimator()
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

            # 2. VQE from analytical init (does it converge faster?)
            def cost_fn(params, _H=H):
                bound = self.circuit.assign_parameters(params)
                job = estimator.run([(bound, _H)])
                return float(job.result()[0].data.evs)

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
            e_from_random = result_from_random.fun
            nit_from_random = result_from_random.nit

            # 4. Best VQE (5 restarts from random — gold standard)
            best_energy = e_from_random
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

            de_gap_best = abs(best_energy - exact_energy) / gap
            de_gap_from_analytical = abs(e_from_analytical - exact_energy) / gap
            de_gap_from_random = abs(e_from_random - exact_energy) / gap

            elapsed = time.time() - t0

            m = V8Metrics(
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
                    "de_gap_best_5restart": float(de_gap_best),
                    "nit_from_analytical": int(nit_from_analytical),
                    "nit_from_random": int(nit_from_random),
                    "iteration_savings_pct": float(
                        (nit_from_random - nit_from_analytical) / max(nit_from_random, 1) * 100
                    ),
                    "analytical_quality": (
                        "excellent"
                        if de_analytical < 0.01
                        else "good"
                        if de_analytical < 0.05
                        else "fair"
                        if de_analytical < 0.2
                        else "poor"
                    ),
                },
            )
            metrics.append(m)

        return metrics

    def analyze(self, results: dict[int, list[V8Metrics]]) -> dict:
        analysis = super().analyze(results)

        # Aggregate by h-value
        h_summary = {}
        for _seed, metrics in results.items():
            for m in metrics:
                h = m.h_value
                if h not in h_summary:
                    h_summary[h] = {
                        "analytical_raw": [],
                        "from_analytical": [],
                        "from_random": [],
                        "nit_savings": [],
                    }
                md = m.technique_metadata
                h_summary[h]["analytical_raw"].append(md["de_gap_analytical_raw"])
                h_summary[h]["from_analytical"].append(md["de_gap_from_analytical_vqe"])
                h_summary[h]["from_random"].append(md["de_gap_from_random_vqe"])
                h_summary[h]["nit_savings"].append(md["iteration_savings_pct"])

        analysis["per_h_comparison"] = []
        for h in sorted(h_summary.keys()):
            d = h_summary[h]
            analysis["per_h_comparison"].append(
                {
                    "h": h,
                    "analytical_raw_de_gap": float(np.mean(d["analytical_raw"])),
                    "vqe_from_analytical_de_gap": float(np.mean(d["from_analytical"])),
                    "vqe_from_random_de_gap": float(np.mean(d["from_random"])),
                    "mean_iteration_savings_pct": float(np.mean(d["nit_savings"])),
                }
            )

        # Find h threshold where analytical is "good" (< 5% raw)
        good_h = [p["h"] for p in analysis["per_h_comparison"] if p["analytical_raw_de_gap"] < 0.05]
        analysis["analytical_valid_from"] = min(good_h) if good_h else None

        return analysis

    def report(self, analysis: dict) -> str:
        base = super().report(analysis)
        lines = [base, "", "Per-h Comparison (analytical vs random init):"]
        lines.append("| h | Analytical raw | VQE(analytical) | VQE(random) | Iter savings |")
        lines.append("|---|---------------|-----------------|-------------|--------------|")
        for p in analysis.get("per_h_comparison", []):
            lines.append(
                f"| {p['h']:.2f} | {p['analytical_raw_de_gap']:.4f} | "
                f"{p['vqe_from_analytical_de_gap']:.4f} | "
                f"{p['vqe_from_random_de_gap']:.4f} | "
                f"{p['mean_iteration_savings_pct']:.0f}% |"
            )

        valid = analysis.get("analytical_valid_from")
        if valid:
            lines.append(f"\nAnalytical init valid (DE/gap<5% raw) from h >= {valid:.2f}")
        else:
            lines.append("\nAnalytical init does NOT reach DE/gap<5% at any tested h")

        return "\n".join(lines)
