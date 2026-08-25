"""C3: Sign-Equivariant MPNN for Z2 Symmetry.

The HVA has a Z2 symmetry: (theta_zz, theta_x) and (-theta_zz, -theta_x)
give the same energy. Different VQE seeds find different sign conventions,
causing inconsistent MPNN training targets.

This module provides three strategies:
    A) enforce_positive: Canonicalize theta_x > 0 before training
    B) min_loss: Use loss = min(MSE(pred, target), MSE(pred, -target))
    C) predict_magnitude: Predict |theta| and sign separately

Strategy A is simplest and recommended for p=1.
Strategy B is more general and works for any p.
"""

from __future__ import annotations

import numpy as np
import torch


def canonicalize_sign(theta: np.ndarray, reference_index: int = -1) -> np.ndarray:
    """Strategy A: Enforce canonical sign convention.


    Ensures the parameter at reference_index is positive.
    If negative, flips ALL parameters (exploiting Z2 symmetry).
    """
    if theta[reference_index] < 0:
        return -theta
    return theta.copy()


def canonicalize_dataset(
    theta_array: np.ndarray,
    reference_index: int = -1,
) -> np.ndarray:
    """Canonicalize an entire dataset of theta vectors."""
    result = theta_array.copy()
    for i in range(len(result)):
        result[i] = canonicalize_sign(result[i], reference_index)
    return result


class SignInvariantLoss(torch.nn.Module):
    """Strategy B: Loss that respects Z2 symmetry.

    loss = min(MSE(pred, target), MSE(pred, -target))

    This allows the MPNN to learn either sign convention —
    whichever is closer to its current prediction.
    """

    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute sign-invariant loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted parameters, shape (batch, n_params).
        target : torch.Tensor
            Target parameters, shape (batch, n_params).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        # MSE with original target
        loss_pos = self.mse(pred, target).mean(dim=-1)  # (batch,)

        # MSE with negated target (Z2 partner)
        loss_neg = self.mse(pred, -target).mean(dim=-1)  # (batch,)

        # Take minimum per sample
        loss = torch.min(loss_pos, loss_neg)

        return loss.mean()


def detect_sign_inconsistency(
    theta_array: np.ndarray,
    h_values: np.ndarray,
    threshold: float = 1.0,
) -> dict:
    """Detect sign inconsistencies in a theta dataset.

    Checks if adjacent h-points have sign flips (indicating Z2 ambiguity).

    Parameters
    ----------
    theta_array : np.ndarray
        Shape (n_points, n_params), ordered by h.
    h_values : np.ndarray
        Corresponding h-values (same length as theta_array).
    threshold : float
        Maximum allowed jump between adjacent points.

    Returns
    -------
    dict with keys:
        n_flips: number of detected sign flips
        flip_indices: list of indices where flips occur
        max_jump: maximum parameter jump between adjacent points
        needs_canonicalization: bool
    """
    n_points = len(theta_array)
    flips = []
    max_jump = 0.0

    for i in range(1, n_points):
        jump = np.max(np.abs(theta_array[i] - theta_array[i - 1]))
        neg_jump = np.max(np.abs(theta_array[i] + theta_array[i - 1]))

        max_jump = max(max_jump, min(jump, neg_jump))

        # A flip is detected if negating makes the trajectory smoother
        if neg_jump < jump and jump > threshold:
            flips.append(i)

    return {
        "n_flips": len(flips),
        "flip_indices": flips,
        "max_jump": float(max_jump),
        "needs_canonicalization": len(flips) > 0,
    }
