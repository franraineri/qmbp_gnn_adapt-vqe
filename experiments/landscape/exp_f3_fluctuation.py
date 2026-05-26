"""F3: Landscape Fluctuation Analysis for Valid Regime Prediction.

Hypothesis: The landscape fluctuation metric (variance of energy over random
parameter samples) predicts HVA circuit quality WITHOUT running VQE, enabling
training-free identification of the valid regime boundary.

Method:
    For each (N, p, h): sample K random theta, compute E(theta) for each.
    Fluctuation = Var(E) / |E_mean|^2.
    High fluctuation = trainable landscape. Low fluctuation = flat/barren.

Expected outcome:
    Fluctuation drops sharply at h < h_min.

Reference: arXiv:2505.05380 — Scalable QAS via Landscape Analysis (2025).
"""

from __future__ import annotations

import numpy as np

from qmbp_simulation.analysis import landscape_fluctuation
from qmbp_simulation.framework import BaseExperiment
from qmbp_simulation.framework.config import AnalysisConfig, ExperimentConfig, SystemConfig
from qmbp_simulation.framework.metrics import ExperimentMetrics


class ExperimentF3(BaseExperiment):
    """Landscape fluctuation analysis for valid regime prediction."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
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
                h_test=[],
            ),
            analysis=AnalysisConfig(fluctuation_n_samples=100),
            seeds=[42, 43, 44],
            verbose=True,
        )

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Compute landscape fluctuation at each h-value."""
        np.random.seed(seed)
        n_samples = self.config.analysis.fluctuation_n_samples
        n_params = self.circuit.num_parameters
        h_values = self.config.system.h_values

        metrics = []

        for h in h_values:
            sol = self.get_exact_solution(h)
            exact_energy = sol["exact"].ground_energy
            gap = sol["exact"].gap if sol["exact"].gap > 0 else 1e-10
            hamiltonian = sol["hamiltonian"]

            # Use shared landscape_fluctuation from qmbp_simulation.analysis
            def cost_fn(params, _H=hamiltonian):
                return self.evaluate_energy(params, _H)

            result = landscape_fluctuation(cost_fn, n_params, n_samples=n_samples, seed=seed)

            fluctuation = result["fluctuation"]
            e_min = result["min"]
            best_random_error = abs(e_min - exact_energy)
            best_random_de_gap = best_random_error / gap

            # Compute fraction_near_gs (additional metric not in shared function)
            # Re-sample to get the raw energies for this calculation
            rng = np.random.default_rng(seed)
            energies = np.array(
                [
                    self.evaluate_energy(rng.uniform(-np.pi, np.pi, n_params), hamiltonian)
                    for _ in range(n_samples)
                ]
            )
            fraction_near_gs = float(np.mean(energies < exact_energy + gap))

            m = ExperimentMetrics(
                h_value=h,
                energy=e_min,
                exact_energy=exact_energy,
                energy_error=best_random_error,
                gap=gap,
                relative_error=best_random_de_gap,
                seed=seed,
                landscape_fluctuation=float(fluctuation),
                technique_metadata={
                    "e_mean": result["mean"],
                    "e_var": result["variance"],
                    "fluctuation": float(fluctuation),
                    "n_samples": n_samples,
                    "best_random_de_gap": float(best_random_de_gap),
                    "fraction_near_gs": fraction_near_gs,
                },
            )
            metrics.append(m)

        return metrics
