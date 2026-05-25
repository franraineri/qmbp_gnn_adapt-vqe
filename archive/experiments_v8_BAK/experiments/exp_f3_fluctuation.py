"""F3: Landscape Fluctuation Analysis for Valid Regime Prediction.

Hypothesis: The landscape fluctuation metric (variance of energy over random
parameter samples) predicts HVA circuit quality WITHOUT running VQE, enabling
training-free identification of the valid regime boundary.

Method:
    For each (N, p, h): sample K random theta, compute E(theta) for each.
    Fluctuation = Var(E) / |E_mean|^2.
    High fluctuation = trainable landscape. Low fluctuation = flat/barren.

Expected outcome:
    Fluctuation drops sharply at h < h_min (landscape becomes flat in the
    ferromagnetic phase where HVA can't express the GS).

Reference: arXiv:2505.05380 — Scalable QAS via Landscape Analysis (2025).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.experiments_v8.core.base_experiment import BaseExperiment
from scripts.experiments_v8.core.config import AnalysisConfig, ExperimentConfig, SystemConfig
from scripts.experiments_v8.core.metrics import V8Metrics


class ExperimentF3(BaseExperiment):
    """Landscape fluctuation analysis for valid regime prediction."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        """Default configuration for F3."""
        return ExperimentConfig(
            experiment_id="F3",
            category="F",
            description="Landscape fluctuation analysis for valid regime prediction",
            hypothesis=(
                "Landscape fluctuation (Var(E)/E_mean^2) drops sharply at h < h_min, "
                "providing a training-free predictor of the valid regime boundary."
            ),
            system=SystemConfig(
                n_qubits=6,
                p_layers=2,
                h_values=[0.5, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0],
                h_test=[],  # No deployment test — this is analysis only
            ),
            analysis=AnalysisConfig(fluctuation_n_samples=100),
            seeds=[42, 43, 44],
            verbose=True,
        )

    def run_single(self, seed: int) -> list[V8Metrics]:
        """Compute landscape fluctuation at each h-value.

        For each h:
        1. Sample K random theta uniformly in [-pi, pi]^n_params
        2. Evaluate E(theta) for each sample
        3. Compute fluctuation = Var(E) / E_mean^2
        4. Also compute: min(E), max(E), std(E)
        """
        np.random.seed(seed)
        n_samples = self.config.analysis.fluctuation_n_samples
        n_params = self.circuit.num_parameters
        h_values = self.config.system.h_values

        metrics = []

        for h in h_values:
            # Get exact solution for reference
            sol = self.get_exact_solution(h)
            exact_energy = sol["exact"].ground_energy
            gap = sol["exact"].gap if sol["exact"].gap > 0 else 1e-10
            hamiltonian = sol["hamiltonian"]

            # Sample random parameters
            energies = np.zeros(n_samples)
            for i in range(n_samples):
                theta_random = np.random.uniform(-np.pi, np.pi, n_params)
                energies[i] = self.evaluate_energy(theta_random, hamiltonian)

            # Compute fluctuation metrics
            e_mean = np.mean(energies)
            e_var = np.var(energies)
            e_std = np.std(energies)
            e_min = np.min(energies)
            e_max = np.max(energies)

            # Normalized fluctuation (the key metric)
            fluctuation = e_var / (e_mean**2) if abs(e_mean) > 1e-10 else 0.0

            # Also compute: how close can random get to exact?
            best_random_error = abs(e_min - exact_energy)
            best_random_de_gap = best_random_error / gap

            # Build metrics
            m = V8Metrics(
                h_value=h,
                energy=e_min,  # Best random energy
                exact_energy=exact_energy,
                energy_error=best_random_error,
                gap=gap,
                relative_error=best_random_de_gap,
                seed=seed,
                landscape_fluctuation=float(fluctuation),
                technique_metadata={
                    "e_mean": float(e_mean),
                    "e_var": float(e_var),
                    "e_std": float(e_std),
                    "e_min": float(e_min),
                    "e_max": float(e_max),
                    "fluctuation": float(fluctuation),
                    "n_samples": n_samples,
                    "best_random_de_gap": float(best_random_de_gap),
                    # Landscape "width" — range of energies accessible
                    "energy_range": float(e_max - e_min),
                    # Fraction of samples below exact + gap (in the "useful" region)
                    "fraction_near_gs": float(np.mean(energies < exact_energy + gap)),
                },
            )
            metrics.append(m)

        return metrics

    def analyze(self, results: dict[int, list[V8Metrics]]) -> dict:
        """Extended analysis: find the fluctuation drop-off point."""
        analysis = super().analyze(results)

        # Aggregate fluctuation vs h across seeds
        h_fluctuation = {}
        for _seed, metrics in results.items():
            for m in metrics:
                h = m.h_value
                if h not in h_fluctuation:
                    h_fluctuation[h] = []
                if m.landscape_fluctuation is not None:
                    h_fluctuation[h].append(m.landscape_fluctuation)

        # Compute mean fluctuation per h
        fluctuation_curve = []
        for h in sorted(h_fluctuation.keys()):
            values = h_fluctuation[h]
            fluctuation_curve.append(
                {
                    "h": h,
                    "mean_fluctuation": float(np.mean(values)),
                    "std_fluctuation": float(np.std(values)),
                }
            )

        analysis["fluctuation_curve"] = fluctuation_curve

        # Find the boundary: where fluctuation drops below 50% of max
        if fluctuation_curve:
            max_fluct = max(p["mean_fluctuation"] for p in fluctuation_curve)
            threshold = max_fluct * 0.5
            boundary_h = None
            for point in reversed(fluctuation_curve):
                if point["mean_fluctuation"] < threshold:
                    boundary_h = point["h"]
                    break

            analysis["predicted_boundary"] = {
                "h_min_predicted": boundary_h,
                "h_min_known": 1.25,  # Known V6.1 boundary for N=6
                "max_fluctuation": float(max_fluct),
                "threshold_used": float(threshold),
            }

        return analysis

    def report(self, analysis: dict) -> str:
        """Generate report with fluctuation curve."""
        base_report = super().report(analysis)

        lines = [base_report, "", "Fluctuation Curve:"]
        lines.append("| h | Fluctuation | Interpretation |")
        lines.append("|---|-------------|----------------|")

        for point in analysis.get("fluctuation_curve", []):
            h = point["h"]
            f = point["mean_fluctuation"]
            if f > 0.1:
                interp = "✅ Trainable"
            elif f > 0.01:
                interp = "⚠️ Borderline"
            else:
                interp = "❌ Flat/barren"
            lines.append(f"| {h:.2f} | {f:.4f} | {interp} |")

        boundary = analysis.get("predicted_boundary", {})
        if boundary:
            lines.extend(
                [
                    "",
                    f"Predicted boundary: h >= {boundary.get('h_min_predicted', '?')}",
                    f"Known boundary:     h >= {boundary.get('h_min_known', '?')}",
                ]
            )

        return "\n".join(lines)
