"""S6: MC-Dropout Uncertainty Quantification (Fix for G2).

Hypothesis: MC-Dropout (dropout active at inference, 50 forward passes)
produces calibrated uncertainty estimates with Pearson r > 0.7 between
predicted variance and actual ΔE/gap.

Method:
    1. Train MPNN with dropout=0.1 (standard config)
    2. At inference, keep model in train mode (dropout active)
    3. Run 50 forward passes per h-point → get mean and variance
    4. Compute Pearson r between variance and actual ΔE/gap
    5. Compare with G2 ensemble result (r=0.195, not calibrated)

Expected outcome:
    r > 0.5 (better than G2's 0.195). If r > 0.7, publishable UQ method.

Thesis value: MEDIUM-HIGH — UQ without additional VQE cost.

References:
    - G2: Ensemble variance not calibrated (r=0.195, same data different init)
    - Miao et al. (2024): Dropout regularization for VQE parameter prediction
    - Gal & Ghahramani (2016): Dropout as approximate Bayesian inference
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from qmbp_simulation.framework import BaseExperiment
from qmbp_simulation.framework.config import ExperimentConfig, MPNNConfig, SystemConfig, VQEConfig
from qmbp_simulation.framework.metrics import ExperimentMetrics

logger = logging.getLogger(__name__)

N_MC_SAMPLES = 50  # Number of forward passes for MC-Dropout


class ExperimentS6(BaseExperiment):
    """MC-Dropout uncertainty quantification."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="S6",
            category="S",
            description="MC-Dropout UQ: calibrated uncertainty without extra VQE cost",
            hypothesis=(
                "MC-Dropout (50 forward passes) achieves Pearson r > 0.7 "
                "between predicted variance and actual ΔE/gap, vs G2's r=0.195."
            ),
            system=SystemConfig(
                n_qubits=6,
                p_layers=2,
                topology="chain_1d",
                h_values=[2.0, 1.75, 1.5, 1.25, 1.0, 0.8, 0.6],
                h_test=[1.75, 1.5, 1.25, 1.0, 0.8],
            ),
            vqe=VQEConfig(n_restarts=5, maxiter=1000, sigma=0.1),
            mpnn=MPNNConfig(hidden_dim=64, n_layers=3, n_epochs=6000, lr=1e-3, dropout=0.1),
            seeds=[42, 43, 44],
            verbose=True,
        )

    def setup(self) -> None:
        """Setup for N=6 VQE + MPNN."""
        from qmbp_simulation import (
            ClassicalSolver,
            HamiltonianBuilder,
            HVACircuitBuilder,
            make_lattice,
        )

        self._setup_logging()
        self.builder = HamiltonianBuilder()
        self.solver = ClassicalSolver()
        self.hva = HVACircuitBuilder()

        N = self.config.system.n_qubits
        p = self.config.system.p_layers
        lattice = make_lattice("chain_1d", N, J=1.0, h=1.0)
        self.circuit, _ = self.hva.create(N, p, lattice)
        self._lattice = lattice
        logger.info("S6 setup: N=%d, p=%d, MC samples=%d", N, p, N_MC_SAMPLES)

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Train MPNN, then evaluate MC-Dropout UQ at test points."""
        from scipy.optimize import minimize

        from experiments.helpers.graph_utils import build_experiment_dataset
        from qmbp_simulation import make_lattice
        from qmbp_simulation.predictors import MPNNPredictor, train_mpnn

        np.random.seed(seed)
        torch.manual_seed(seed)

        N = self.config.system.n_qubits
        n_params = self.circuit.num_parameters
        h_train = self.config.system.h_values
        h_test = self.config.system.h_test
        metrics = []

        # Phase 2: VQE sweep
        vqe_data: dict[float, np.ndarray] = {}
        prev_theta = np.random.uniform(-0.01, 0.01, n_params)

        for h in sorted(h_train, reverse=True):
            lattice = make_lattice("chain_1d", N, J=1.0, h=h)
            H = self.builder.build(lattice)

            best_e, best_theta = float("inf"), prev_theta.copy()
            for r in range(self.config.vqe.n_restarts):
                x0 = (
                    prev_theta.copy()
                    if r == 0
                    else best_theta + np.random.normal(0, self.config.vqe.sigma, n_params)
                )
                x0 = np.clip(x0, -np.pi, np.pi)
                res = minimize(
                    lambda p, _H=H: self._evaluate_energy(p, _H),
                    x0,
                    method="L-BFGS-B",
                    bounds=[(-np.pi, np.pi)] * n_params,
                    options={"maxiter": self.config.vqe.maxiter, "ftol": 1e-14},
                )
                if res.fun < best_e:
                    best_e = res.fun
                    best_theta = res.x.copy()

            vqe_data[h] = best_theta.copy()
            prev_theta = best_theta.copy()

        # Phase 3: Train MPNN with dropout
        h_sorted = sorted(vqe_data.keys())
        theta_array = np.array([vqe_data[h] for h in h_sorted])

        dataset = build_experiment_dataset(self, np.array(h_sorted), theta_array)
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

        # Phase 4: MC-Dropout inference at test points
        actual_errors = []
        predicted_variances = []

        for h_t in h_test:
            lattice = make_lattice("chain_1d", N, J=1.0, h=h_t)
            H = self.builder.build(lattice)
            result = self.solver.solve(H, lattice)
            exact_energy = result.ground_energy
            gap = max(result.gap, 1e-10)

            # MC-Dropout: N_MC_SAMPLES forward passes with dropout active
            # Use predict_theta but with model in train mode
            from experiments.helpers.graph_utils import _get_graph_structure

            edge_index, coord, n_sites = _get_graph_structure(self)

            h_feat = np.full(n_sites, float(h_t))
            x = torch.tensor(
                np.stack([h_feat, coord.astype(float)], axis=1),
                dtype=torch.float32,
            )
            from torch_geometric.data import Batch
            from torch_geometric.data import Data as PyGData

            graph = PyGData(x=x, edge_index=edge_index)
            batch = Batch.from_data_list([graph])

            predictions = []
            model.train()  # Keep dropout active
            with torch.no_grad():
                for _ in range(N_MC_SAMPLES):
                    pred = model(batch).numpy().flatten()
                    predictions.append(pred)

            predictions = np.array(predictions)  # (N_MC_SAMPLES, n_params)
            theta_mean = predictions.mean(axis=0)
            theta_var = predictions.var(axis=0).mean()  # Mean variance across params

            # Evaluate mean prediction
            e_pred = self._evaluate_energy(theta_mean, H)
            de_gap = abs(e_pred - exact_energy) / gap

            actual_errors.append(de_gap)
            predicted_variances.append(theta_var)

            m = ExperimentMetrics(
                h_value=h_t,
                energy=e_pred,
                exact_energy=exact_energy,
                energy_error=abs(e_pred - exact_energy),
                gap=gap,
                relative_error=de_gap,
                seed=seed,
                wall_time_s=0.0,
                converged=de_gap < 0.05,
                technique_metadata={
                    "mc_variance": float(theta_var),
                    "de_gap": float(de_gap),
                    "n_mc_samples": N_MC_SAMPLES,
                    "theta_std_per_param": predictions.std(axis=0).tolist(),
                },
            )
            metrics.append(m)

        # Compute correlation
        if len(actual_errors) >= 3:
            from scipy.stats import pearsonr

            r, p_value = pearsonr(predicted_variances, actual_errors)
            logger.info(
                "  MC-Dropout UQ: Pearson r=%.3f (p=%.4f) — %s",
                r,
                p_value,
                "CALIBRATED ✅" if r > 0.7 else ("PARTIAL ⚠️" if r > 0.5 else "NOT CALIBRATED ❌"),
            )
            # Add correlation as metadata to last metric
            metrics[-1].technique_metadata["pearson_r"] = float(r)
            metrics[-1].technique_metadata["pearson_p_value"] = float(p_value)
            metrics[-1].technique_metadata["g2_baseline_r"] = 0.195

        return metrics

    def _evaluate_energy(self, params, H) -> float:
        """Evaluate energy using StatevectorEstimator."""
        from qiskit.primitives import StatevectorEstimator

        estimator = StatevectorEstimator()
        bound = self.circuit.assign_parameters(params)
        job = estimator.run([(bound, H)])
        return float(job.result()[0].data.evs)
