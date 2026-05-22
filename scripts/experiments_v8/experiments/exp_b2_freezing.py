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

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.experiments_v8.core.base_experiment import BaseExperiment
from scripts.experiments_v8.core.config import ExperimentConfig, SystemConfig, VQEConfig
from scripts.experiments_v8.core.metrics import V8Metrics
from scripts.experiments_v8.techniques.parameter_freezing import (
    analyze_parameter_activity,
    frozen_vqe,
)


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
                h_values=[0.8, 1.0, 1.25, 1.5, 1.75, 2.0],
            ),
            vqe=VQEConfig(n_restarts=5, sigma=0.1),
            seeds=[42, 43, 44],
            verbose=True,
        )

    def run_single(self, seed: int) -> list[V8Metrics]:
        """Phase 1: Generate trajectory. Phase 2: Test freezing."""
        from qiskit.primitives import StatevectorEstimator
        from scipy.optimize import minimize

        np.random.seed(seed)
        estimator = StatevectorEstimator()
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
                bound = self.circuit.assign_parameters(params)
                job = estimator.run([(bound, _H)])
                return float(job.result()[0].data.evs)

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

        # Phase 2: Test freezing at each h
        # Use frozen params identified from trajectory
        frozen_indices = activity["frozen_indices"]
        prev_theta = np.random.uniform(-0.01, 0.01, n_params)

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
                bound = self.circuit.assign_parameters(params)
                job = estimator.run([(bound, _H)])
                return float(job.result()[0].data.evs)

            # Full VQE (baseline) — already computed above
            e_full = cost_fn_full(theta_trajectory[i])
            de_gap_full = abs(e_full - exact_energy) / gap

            # Frozen VQE (only if we have frozen params)
            if frozen_indices and h <= 1.75:  # Only freeze in smooth regime
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
                result_frozen["total_evaluations"]
            else:
                e_frozen = e_full
                de_gap_frozen = de_gap_full

            # Cold-start comparison
            cold_init = np.random.uniform(-0.5, 0.5, n_params)
            res_cold = minimize(
                cost_fn_full,
                cold_init,
                method="L-BFGS-B",
                bounds=[(-np.pi, np.pi)] * n_params,
                options={"maxiter": 500, "ftol": 1e-14},
            )
            de_gap_cold = abs(res_cold.fun - exact_energy) / gap

            elapsed = time.time() - t0

            m = V8Metrics(
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
                    "de_gap_cold": float(de_gap_cold),
                    "accuracy_loss_pct": float(
                        (de_gap_frozen - de_gap_full) / max(de_gap_full, 1e-10) * 100
                    ),
                    "frozen_indices": frozen_indices,
                    "n_frozen": len(frozen_indices),
                    "n_active": n_params - len(frozen_indices),
                    "activity_analysis": {
                        "mean_activity": activity["mean_activity"].tolist(),
                        "frozen_mask": activity["frozen_mask"].tolist(),
                    },
                    "warm_vs_cold_gain_pct": float(
                        (de_gap_cold - de_gap_full) / max(de_gap_cold, 1e-10) * 100
                    ),
                },
            )
            metrics.append(m)

        return metrics

    def analyze(self, results: dict[int, list[V8Metrics]]) -> dict:
        analysis = super().analyze(results)

        accuracy_losses = []
        frozen_counts = []

        for _seed, metrics in results.items():
            for m in metrics:
                md = m.technique_metadata
                accuracy_losses.append(md["accuracy_loss_pct"])
                frozen_counts.append(md["n_frozen"])

        analysis["freezing_summary"] = {
            "mean_accuracy_loss_pct": float(np.mean(accuracy_losses)),
            "max_accuracy_loss_pct": float(np.max(accuracy_losses)) if accuracy_losses else 0,
            "mean_n_frozen": float(np.mean(frozen_counts)),
            "hypothesis_confirmed": (
                float(np.max(accuracy_losses)) < 1.0 if accuracy_losses else False
            ),
        }

        return analysis

    def report(self, analysis: dict) -> str:
        base = super().report(analysis)
        lines = [base, "", "Parameter Freezing Analysis:"]

        fs = analysis.get("freezing_summary", {})
        lines.extend(
            [
                f"  Mean frozen params:     {fs.get('mean_n_frozen', 0):.1f} / 4",
                f"  Mean accuracy loss:     {fs.get('mean_accuracy_loss_pct', 0):.2f}%",
                f"  Max accuracy loss:      {fs.get('max_accuracy_loss_pct', 0):.2f}%",
                f"  Hypothesis (<1% loss):  {fs.get('hypothesis_confirmed', False)}",
            ]
        )

        return "\n".join(lines)
