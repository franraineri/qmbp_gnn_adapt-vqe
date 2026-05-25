"""C1: Physics-Informed MPNN Loss.

Adds an energy-validation term to the MPNN training loss that evaluates
E(θ_pred) on the actual Hamiltonian every K epochs. This prevents the MPNN
from learning parameters with low MSE but high energy error.

IMPORTANT: This does NOT change the VQE cost function (V5.x lesson).
The θ targets remain pure-energy VQE optima. The energy term is a
MPNN training regularizer only.

References:
    - Miao et al. (2024) PRApplied 21, 014053 — energy-aware NN training
    - Zhang et al. (2025) arXiv:2505.01236 (Qracle) — GNN with energy feedback
    - Lee et al. (2026) arXiv:2602.19752 — energy-based loss for VQE params
"""

from __future__ import annotations

import numpy as np
import torch


class PhysicsInformedLoss(torch.nn.Module):
    """Combined MSE + energy validation loss for MPNN training.

    loss = MSE(θ_pred, θ_target) + λ · mean(|E(θ_pred) - E_exact|)

    The energy term is only evaluated every `eval_every` epochs and on a
    random subset of training points (for efficiency).
    """

    def __init__(
        self,
        weight: float = 0.1,
        start_epoch: int = 1000,
        eval_every: int = 100,
        n_eval_points: int = 5,
    ):
        """Initialize physics-informed loss.

        Parameters
        ----------
        weight : float
            Weight λ for the energy term.
        start_epoch : int
            Epoch at which to start adding energy term.
        eval_every : int
            Evaluate energy every N epochs.
        n_eval_points : int
            Number of random training points to evaluate energy on.
        """
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.weight = weight
        self.start_epoch = start_epoch
        self.eval_every = eval_every
        self.n_eval_points = n_eval_points
        self._current_epoch = 0
        self._energy_history: list[float] = []

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch (called by training loop)."""
        self._current_epoch = epoch

    def should_eval_energy(self) -> bool:
        """Check if energy should be evaluated this epoch."""
        if self._current_epoch < self.start_epoch:
            return False
        return (self._current_epoch - self.start_epoch) % self.eval_every == 0

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        energy_errors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute combined loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted θ, shape (batch, n_params).
        target : torch.Tensor
            Target θ, shape (batch, n_params).
        energy_errors : torch.Tensor | None
            |E(θ_pred) - E_exact| for selected points.

        Returns
        -------
        torch.Tensor
            Combined loss value.
        """
        mse_loss = self.mse(pred, target)

        if energy_errors is not None and self._current_epoch >= self.start_epoch:
            energy_loss = torch.mean(energy_errors)
            self._energy_history.append(float(energy_loss.item()))
            return mse_loss + self.weight * energy_loss

        return mse_loss

    @property
    def energy_history(self) -> list[float]:
        """History of energy loss values."""
        return self._energy_history


def evaluate_energy_batch(
    theta_batch: np.ndarray,
    hamiltonians: list,
    circuit,
    exact_energies: np.ndarray,
) -> np.ndarray:
    """Evaluate energy errors for a batch of predicted parameters.

    Parameters
    ----------
    theta_batch : np.ndarray
        Predicted parameters, shape (batch, n_params).
    hamiltonians : list
        List of Hamiltonians (one per batch element).
    circuit : QuantumCircuit
        Parameterized circuit.
    exact_energies : np.ndarray
        Exact ground state energies, shape (batch,).

    Returns
    -------
    np.ndarray
        |E(θ_pred) - E_exact| for each point, shape (batch,).
    """
    from qiskit.primitives import StatevectorEstimator

    estimator = StatevectorEstimator()
    errors = np.zeros(len(theta_batch))

    for i, (theta, H, e_exact) in enumerate(
        zip(theta_batch, hamiltonians, exact_energies, strict=False)
    ):
        bound = circuit.assign_parameters(theta)
        job = estimator.run([(bound, H)])
        e_pred = float(job.result()[0].data.evs)
        errors[i] = abs(e_pred - e_exact)

    return errors


def select_eval_subset(
    n_total: int,
    n_eval: int,
    seed: int | None = None,
) -> list[int]:
    """Select random subset of training points for energy evaluation.

    Parameters
    ----------
    n_total : int
        Total number of training points.
    n_eval : int
        Number to select.
    seed : int | None
        Random seed.

    Returns
    -------
    list[int]
        Indices of selected points.
    """
    rng = np.random.default_rng(seed)
    n_eval = min(n_eval, n_total)
    return rng.choice(n_total, size=n_eval, replace=False).tolist()
