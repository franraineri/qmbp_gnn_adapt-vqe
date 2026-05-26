"""G2: MPNN Ensemble Uncertainty Calibration.

Hypothesis: The prediction variance of a 5-MPNN ensemble correlates with
actual ΔE/gap (r > 0.7), enabling detection of unreliable predictions
without running VQE.

Method:
    1. Generate VQE data at N=6 (17 h-points)
    2. Train 5 MPNNs with different seeds
    3. Predict θ at 20 h-points (including out-of-regime)
    4. Compute ensemble variance at each h
    5. Compute actual ΔE/gap at each h
    6. Measure correlation(variance, ΔE/gap)

Expected outcome: r > 0.7 — high variance = high error.

Thesis value: HIGH — uncertainty quantification for deployment.
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


class ExperimentG2(BaseExperiment):
    """Ensemble uncertainty calibration."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="G2",
            category="G",
            description="Ensemble uncertainty calibration: variance vs ΔE/gap",
            hypothesis=(
                "5-MPNN ensemble variance correlates with ΔE/gap (r>0.7), "
                "enabling unreliable prediction detection without VQE."
            ),
            system=SystemConfig(
                n_qubits=6,
                p_layers=2,
                h_values=list(np.linspace(1.0, 2.0, 17)),
                h_test=list(np.linspace(0.5, 2.5, 20)),
            ),
            vqe=VQEConfig(n_restarts=5, maxiter=500),
            mpnn=MPNNConfig(hidden_dim=64, n_layers=3, n_epochs=4000, lr=1e-3),
            seeds=[42, 43, 44],
            verbose=True,
            auto_warm_cold_comparison=False,
        )

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Train ensemble, measure variance vs actual error."""

        from qmbp_simulation.predictors import MPNNPredictor, train_mpnn

        np.random.seed(seed)
        N = self.config.system.n_qubits
        n_params = self.circuit.num_parameters
        h_train = np.array(self.config.system.h_values)
        h_test = np.array(self.config.system.h_test)

        # Phase 2: Generate VQE training data
        logger.info(f"    Generating VQE data ({len(h_train)} points)...")
        vqe_data = self._run_vqe_sweep(h_train, seed)

        # Build dataset
        dataset = self._build_dataset(h_train, vqe_data, N)

        # Train 5 MPNNs with different seeds
        n_ensemble = 5
        models = []
        logger.info(f"    Training {n_ensemble} MPNNs...")
        for i in range(n_ensemble):
            torch.manual_seed(seed * 100 + i)
            model = MPNNPredictor(
                node_features=2,
                hidden_dim=self.config.mpnn.hidden_dim,
                n_layers=self.config.mpnn.n_layers,
                output_dim=n_params,
            )
            train_mpnn(
                model,
                dataset,
                n_epochs=self.config.mpnn.n_epochs,
                lr=self.config.mpnn.lr,
                patience=self.config.mpnn.patience,
            )
            models.append(model)

        # Predict at all test points with ensemble
        metrics = []
        for h_t in h_test:
            t0 = time.time()
            sol = self.get_exact_solution(float(h_t))
            H = sol["hamiltonian"]
            e_exact = sol["exact"].ground_energy
            gap = (
                sol["exact"].gap
                if sol["exact"].gap > 1e-10
                else max(2 * abs(1 - h_t), 2 * np.pi / N)
            )

            # Ensemble predictions
            predictions = []
            for model in models:
                theta_pred = self._predict(model, float(h_t), N)
                predictions.append(theta_pred)

            predictions = np.array(predictions)
            ensemble_mean = predictions.mean(axis=0)
            ensemble_var = float(predictions.var(axis=0).mean())

            # Actual energy from ensemble mean
            e_pred = self.backend.evaluate(self.circuit, H, ensemble_mean)
            de_gap = abs(e_pred - e_exact) / gap
            elapsed = time.time() - t0

            m = ExperimentMetrics(
                h_value=float(h_t),
                energy=e_pred,
                exact_energy=e_exact,
                energy_error=abs(e_pred - e_exact),
                gap=gap,
                relative_error=de_gap,
                seed=seed,
                wall_time_s=elapsed,
                technique_metadata={
                    "ensemble_variance": ensemble_var,
                    "de_gap": float(de_gap),
                    "in_training_range": bool(1.0 <= h_t <= 2.0),
                },
            )
            metrics.append(m)

        return metrics

    def _run_vqe_sweep(self, h_values, seed):
        """Run VQE descending sweep."""
        from scipy.optimize import minimize

        n_params = self.circuit.num_parameters
        prev_theta = np.random.uniform(-0.01, 0.01, n_params)
        vqe_data = {}
        for h in sorted(h_values, reverse=True):
            sol = self.get_exact_solution(float(h))
            H = sol["hamiltonian"]
            best_e, best_theta = float("inf"), prev_theta.copy()
            for r in range(self.config.vqe.n_restarts):
                x0 = (
                    prev_theta.copy() if r == 0 else best_theta + np.random.normal(0, 0.1, n_params)
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

    def _build_dataset(self, h_values, vqe_data, N):
        """Build PyG dataset."""
        from torch_geometric.data import Data as PyGData

        dataset = []
        coord = 2 if N > 2 else 1
        edges = [[i, i + 1] for i in range(N - 1)] + [[i + 1, i] for i in range(N - 1)]
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        for h in sorted(h_values):
            x = torch.tensor([[float(h), coord]] * N, dtype=torch.float32)
            y = torch.tensor(vqe_data[float(h)], dtype=torch.float32)
            dataset.append(PyGData(x=x, edge_index=edge_index, y=y))
        return dataset

    def _predict(self, model, h, N):
        """Predict theta at single h."""
        from torch_geometric.data import Batch
        from torch_geometric.data import Data as PyGData

        coord = 2 if N > 2 else 1
        edges = [[i, i + 1] for i in range(N - 1)] + [[i + 1, i] for i in range(N - 1)]
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        x = torch.tensor([[float(h), coord]] * N, dtype=torch.float32)
        batch = Batch.from_data_list([PyGData(x=x, edge_index=edge_index)])
        model.eval()
        with torch.no_grad():
            return model(batch).numpy().flatten()

    def analyze(self, results: dict[int, list[ExperimentMetrics]]) -> dict:
        from scipy.stats import pearsonr

        analysis = super().analyze(results)
        variances, errors = [], []
        for _seed, metrics in results.items():
            for m in metrics:
                variances.append(m.technique_metadata["ensemble_variance"])
                errors.append(m.technique_metadata["de_gap"])

        r_val, p_val = pearsonr(variances, errors) if len(variances) > 3 else (0, 1)
        analysis["calibration"] = {
            "pearson_r": float(r_val),
            "p_value": float(p_val),
            "hypothesis_confirmed": abs(r_val) > 0.7,
            "n_points": len(variances),
        }
        return analysis

    def report(self, analysis: dict) -> str:
        base = super().report(analysis)
        cal = analysis.get("calibration", {})
        confirmed = "✅" if cal.get("hypothesis_confirmed") else "❌"
        lines = [
            base,
            "",
            "Ensemble Calibration:",
            f"  Pearson r(variance, ΔE/gap): {cal.get('pearson_r', 0):.3f} {confirmed}",
            f"  p-value: {cal.get('p_value', 1):.4f}",
            f"  Hypothesis (r>0.7): {cal.get('hypothesis_confirmed', False)}",
        ]
        return "\n".join(lines)
