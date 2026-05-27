"""G1: Data Efficiency Curve — Minimum VQE points for MPNN deployment.

Hypothesis: The MPNN achieves ΔE/gap < 5% with 9-11 training points
(vs 17 current) if uniformly spaced in the valid regime [1.0, 2.0].

Method:
    For k = 5, 7, 9, 11, 13, 15, 17:
        1. Select k uniformly-spaced h-points in [1.0, 2.0]
        2. Run VQE at those points (descending warm-start)
        3. Train MPNN on k points
        4. Deploy at h_test = 1.5
        5. Measure ΔE/gap

Expected outcome: k_min ≈ 9-11 points for ΔE/gap < 5%.

Thesis value: HIGH — defines minimum pipeline cost.
Reference: Miao et al. (2024) PRApplied 21, 014053.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from qmbp_simulation.framework import BaseExperiment, ExperimentConfig
from qmbp_simulation.framework.config import MPNNConfig, SystemConfig, VQEConfig
from qmbp_simulation.framework.metrics import ExperimentMetrics

logger = logging.getLogger(__name__)


class ExperimentG1(BaseExperiment):
    """Data efficiency curve for MPNN training."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="G1",
            category="G",
            description="Data efficiency: minimum VQE points for MPNN deployment",
            hypothesis=(
                "MPNN achieves ΔE/gap < 5% with 9-11 training points "
                "(vs 17 current) if uniformly spaced in [1.0, 2.0]."
            ),
            system=SystemConfig(
                n_qubits=6,
                p_layers=2,
                h_values=[],  # Set dynamically per k
                h_test=[1.5],
            ),
            vqe=VQEConfig(n_restarts=5, maxiter=500),
            mpnn=MPNNConfig(hidden_dim=64, n_layers=3, n_epochs=6000, lr=1e-3),
            seeds=[42, 43, 44],
            verbose=True,
            auto_warm_cold_comparison=False,
        )

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Test MPNN deployment quality for each training set size k."""
        from qmbp_simulation.predictors import MPNNPredictor, train_mpnn

        np.random.seed(seed)
        torch.manual_seed(seed)
        N = self.config.system.n_qubits
        n_params = self.circuit.num_parameters
        h_test = self.config.system.h_test[0]

        # Get exact solution at test point
        sol_test = self.get_exact_solution(h_test)
        H_test = sol_test["hamiltonian"]
        e_exact = sol_test["exact"].ground_energy
        gap = sol_test["exact"].gap if sol_test["exact"].gap > 1e-10 else 0.5

        # Generate VQE data for full grid (17 points)
        h_full = np.linspace(1.0, 2.0, 17)
        vqe_data = self._run_vqe_sweep(h_full, seed)

        # Test subsets of size k
        k_values = [5, 7, 9, 11, 13, 15, 17]
        metrics = []

        for k in k_values:
            t0 = time.time()
            # Select k uniformly-spaced points from the full grid
            indices = np.linspace(0, len(h_full) - 1, k, dtype=int)
            h_subset = h_full[indices]
            theta_subset = np.array([vqe_data[h] for h in h_subset])

            # Build dataset and train MPNN
            dataset = self._build_dataset(h_subset, theta_subset, N)
            torch.manual_seed(seed + k)  # Different init per k
            model = MPNNPredictor(
                node_features=2,
                hidden_dim=self.config.mpnn.hidden_dim,
                n_layers=self.config.mpnn.n_layers,
                output_dim=n_params,
            )
            history = train_mpnn(
                model,
                dataset,
                n_epochs=self.config.mpnn.n_epochs,
                lr=self.config.mpnn.lr,
                patience=self.config.mpnn.patience,
            )

            # Deploy at test point
            theta_pred = self._predict(model, h_test, N)
            e_pred = self.backend.evaluate(self.circuit, H_test, theta_pred)
            de_gap = abs(e_pred - e_exact) / gap
            elapsed = time.time() - t0

            m = ExperimentMetrics(
                h_value=h_test,
                energy=e_pred,
                exact_energy=e_exact,
                energy_error=abs(e_pred - e_exact),
                gap=gap,
                relative_error=de_gap,
                seed=seed,
                wall_time_s=elapsed,
                n_evaluations=0,
                technique_metadata={
                    "k_points": k,
                    "de_gap": float(de_gap),
                    "final_mse": history["final_mse"],
                    "passes_threshold": de_gap < 0.05,
                },
            )
            metrics.append(m)
            logger.info(f"    k={k}: ΔE/gap={de_gap:.4f} {'✅' if de_gap < 0.05 else '❌'}")

        return metrics

    def _run_vqe_sweep(self, h_values: np.ndarray, seed: int) -> dict:
        """Run VQE descending sweep, return h -> theta_opt mapping."""
        from scipy.optimize import minimize

        n_params = self.circuit.num_parameters
        prev_theta = np.random.uniform(-0.01, 0.01, n_params)
        vqe_data = {}

        for h in sorted(h_values, reverse=True):
            sol = self.get_exact_solution(float(h))
            H = sol["hamiltonian"]

            best_e = float("inf")
            best_theta = prev_theta.copy()
            for r in range(self.config.vqe.n_restarts):
                x0 = (
                    prev_theta.copy()
                    if r == 0
                    else (best_theta + np.random.normal(0, 0.1, n_params))
                )
                x0 = np.clip(x0, -np.pi, np.pi)
                res = minimize(
                    lambda p, _H=H: self.backend.evaluate(self.circuit, _H, p),
                    x0,
                    method="L-BFGS-B",
                    bounds=[(-np.pi, np.pi)] * n_params,
                    options={"maxiter": self.config.vqe.maxiter, "ftol": 1e-14},
                )
                if res.fun < best_e:
                    best_e = res.fun
                    best_theta = res.x.copy()

            vqe_data[float(h)] = best_theta.copy()
            prev_theta = best_theta.copy()

        return vqe_data

    def _build_dataset(self, h_values, theta_array, N):
        """Build PyG dataset using the experiment's lattice topology."""
        from experiments.helpers.graph_utils import build_experiment_dataset

        return build_experiment_dataset(self, h_values, theta_array)

    def _predict(self, model, h, N):
        """Predict theta at a single h-value."""
        from experiments.helpers.graph_utils import predict_theta

        return predict_theta(self, model, h)

    def analyze(self, results: dict[int, list[ExperimentMetrics]]) -> dict:
        analysis = super().analyze(results)

        # Find k_min per seed
        k_min_per_seed = []
        for _seed, metrics in results.items():
            for m in metrics:
                if m.technique_metadata.get("passes_threshold"):
                    k_min_per_seed.append(m.technique_metadata["k_points"])
                    break
            else:
                k_min_per_seed.append(None)

        analysis["data_efficiency"] = {
            "k_min_per_seed": k_min_per_seed,
            "k_min_mean": float(np.mean([k for k in k_min_per_seed if k]))
            if any(k_min_per_seed)
            else None,
            "baseline_k": 17,
            "reduction_pct": float((17 - np.mean([k for k in k_min_per_seed if k])) / 17 * 100)
            if any(k_min_per_seed)
            else 0,
        }
        return analysis

    def report(self, analysis: dict) -> str:
        base = super().report(analysis)
        de = analysis.get("data_efficiency", {})
        lines = [
            base,
            "",
            "Data Efficiency:",
            f"  k_min (mean): {de.get('k_min_mean', '?')}",
            "  Baseline:     17 points",
            f"  Reduction:    {de.get('reduction_pct', 0):.1f}%",
        ]
        return "\n".join(lines)
