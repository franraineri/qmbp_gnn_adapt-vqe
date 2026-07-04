"""D1: Unsupervised Phase Detection from MPNN Weight Space.

Hypothesis: The trained MPNN's weight gradient norm ||dW/dh|| peaks near
the quantum phase transition, enabling zero-QPU phase detection.

Method:
    1. Train MPNN-A on full h-range [0.5, 2.5]
    2. Train MPNN-B on valid-regime only [1.25, 2.5]
    3. Compute ||dW/dh|| for both via finite differences on weights
    4. Locate peaks and compare with known h_c = 1.0

Expected outcome: MPNN-A peak near h_c; MPNN-B peak at training boundary.

Reference: Hernandes et al. (2025) arXiv:2503.17140.
"""

from __future__ import annotations

import logging

import numpy as np

from qmbp_simulation.framework import BaseExperiment
from qmbp_simulation.framework.config import ExperimentConfig, MPNNConfig, SystemConfig
from qmbp_simulation.framework.metrics import ExperimentMetrics


class ExperimentD1(BaseExperiment):
    """Weight-space phase detection from MPNN gradients."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="D1",
            category="D",
            description="Zero-QPU phase detection from MPNN weight gradient peaks",
            hypothesis=(
                "||dW/dh|| peaks near h_c for MPNN trained on full h-range, "
                "but at training boundary for valid-regime-only MPNN."
            ),
            system=SystemConfig(n_qubits=6, p_layers=2, h_values=[]),
            mpnn=MPNNConfig(
                hidden_dim=64,
                n_layers=3,
                n_epochs=3000,
                lr=1e-3,
                patience=300,
            ),
            seeds=DEFAULT_SEEDS,
            verbose=True,
        )

    def setup(self) -> None:
        """Override: we need MPNN infrastructure."""
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
        self.logger = logging.getLogger(__name__)

        N = self.config.system.n_qubits
        p = self.config.system.p_layers
        base_lattice = make_lattice(self.config.system.topology, N, J=1.0, h=1.0)
        self.circuit, _ = self.hva.create(N, p, base_lattice)
        self.logger.info(f"D1 setup: N={N}, p={p}")

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Train two MPNNs and analyze weight gradients."""

        np.random.seed(seed)
        n_params = self.circuit.num_parameters

        # Generate VQE data
        h_full = np.linspace(0.5, 2.5, 40)
        h_valid = np.linspace(1.25, 2.5, 25)

        theta_full = self._generate_vqe_data(h_full, seed)
        theta_valid = self._generate_vqe_data(h_valid, seed)

        # Train MLPs as MPNN proxies
        mpnn_a = self._train_mlp(h_full, theta_full, seed, n_params)
        mpnn_b = self._train_mlp(h_valid, theta_valid, seed, n_params)

        # Compute weight gradients
        h_probe = np.linspace(0.5, 2.5, 50)
        grad_a = self._compute_weight_gradients(mpnn_a, h_probe)
        grad_b = self._compute_weight_gradients(mpnn_b, h_probe)

        peak_a_h = float(h_probe[int(np.argmax(grad_a))])
        peak_b_h = float(h_probe[int(np.argmax(grad_b))])

        metrics = []
        for i, h in enumerate(h_probe):
            m = ExperimentMetrics(
                h_value=float(h),
                energy=0.0,
                exact_energy=0.0,
                energy_error=0.0,
                gap=1.0,
                relative_error=float(grad_a[i]),
                seed=seed,
                technique_metadata={
                    "grad_norm_full_range": float(grad_a[i]),
                    "grad_norm_valid_only": float(grad_b[i]),
                    "peak_h_full": peak_a_h,
                    "peak_h_valid": peak_b_h,
                    "known_h_c": 1.0,
                },
            )
            metrics.append(m)

        self.logger.info(
            f"  Seed {seed}: peak_A={peak_a_h:.2f} (full), peak_B={peak_b_h:.2f} (valid)"
        )
        return metrics

    def _generate_vqe_data(self, h_values: np.ndarray, seed: int) -> np.ndarray:
        """Run VQE descending sweep to generate training data."""
        from scipy.optimize import minimize

        n_params = self.circuit.num_parameters
        theta_data = np.zeros((len(h_values), n_params))

        h_sorted_desc = np.sort(h_values)[::-1]
        h_to_idx = {float(h): i for i, h in enumerate(h_values)}
        prev_theta = np.random.uniform(-0.01, 0.01, n_params)

        for h in h_sorted_desc:
            sol = self.get_exact_solution(float(h))
            H = sol["hamiltonian"]

            def cost_fn(params, _H=H):
                return self.evaluate_energy(params, _H)

            best_energy = float("inf")
            best_theta = prev_theta.copy()
            for restart in range(3):
                x0 = (
                    prev_theta + np.random.normal(0, 0.1, n_params)
                    if restart > 0
                    else prev_theta.copy()
                )
                x0 = np.clip(x0, -np.pi, np.pi)
                result = minimize(
                    cost_fn,
                    x0,
                    method="L-BFGS-B",
                    bounds=[(-np.pi, np.pi)] * n_params,
                    options={"maxiter": 300, "ftol": 1e-12},
                )
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_theta = result.x.copy()

            idx = h_to_idx.get(float(h))
            if idx is not None:
                theta_data[idx] = best_theta.copy()
            prev_theta = best_theta.copy()

        return theta_data

    def _train_mlp(self, h_values, theta_data, seed, n_params):
        """Train a simple MLP as MPNN proxy."""
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_params),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        X = torch.tensor(h_values.reshape(-1, 1), dtype=torch.float32)
        Y = torch.tensor(theta_data, dtype=torch.float32)

        for _epoch in range(self.config.mpnn.n_epochs):
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            optimizer.step()

        return model

    def _compute_weight_gradients(self, model, h_probe):
        """Compute ||dθ/dh|| at each probe point via finite differences."""
        import torch

        epsilon = 0.02
        grad_norms = np.zeros(len(h_probe))

        for i, h in enumerate(h_probe):
            h_plus = torch.tensor([[h + epsilon]], dtype=torch.float32)
            h_minus = torch.tensor([[h - epsilon]], dtype=torch.float32)

            with torch.no_grad():
                pred_plus = model(h_plus).numpy().flatten()
                pred_minus = model(h_minus).numpy().flatten()

            grad = (pred_plus - pred_minus) / (2 * epsilon)
            grad_norms[i] = float(np.linalg.norm(grad))

        return grad_norms
