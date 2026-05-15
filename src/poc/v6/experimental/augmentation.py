"""
Data augmentation via θ interpolation (DEPRECATED).

Status: Tested and REJECTED during V6.0 development.
Reason: Linear interpolation of θ between adjacent h-points is inaccurate
        near the critical region where the θ landscape is non-linear.
        Hurts accuracy at both N=6 and N=10.

Kept for reproducibility of V6.0 benchmark results.
"""

from __future__ import annotations

import logging

from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def augment_graph_dataset(
    dataset: list[Data],
    alphas: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> list[Data]:
    """Augment training data by interpolating θ between adjacent h-points.

    DEPRECATED: Tested and rejected — hurts accuracy at N=6 and N=10.

    For each consecutive pair (h_i, h_{i+1}) in the dataset, creates
    synthetic points at h_interp = (1-α)*h_i + α*h_{i+1} with
    θ_interp = (1-α)*θ_i + α*θ_{i+1}.

    This exploits the smooth θ landscape from the descending sweep to
    multiply training data without extra VQE runs.

    Parameters
    ----------
    dataset : list[Data] — original fidelity-filtered dataset (sorted by h)
    alphas : tuple[float, ...] — interpolation fractions

    Returns
    -------
    list[Data] — augmented dataset (original + interpolated points)
    """
    if len(dataset) < 2:
        return dataset

    # Sort by h_value
    sorted_ds = sorted(dataset, key=lambda d: d.h_value)
    augmented = list(sorted_ds)

    for i in range(len(sorted_ds) - 1):
        d1, d2 = sorted_ds[i], sorted_ds[i + 1]
        h1, h2 = d1.h_value, d2.h_value
        theta1, theta2 = d1.y, d2.y
        e1, e2 = d1.e_exact, d2.e_exact

        for alpha in alphas:
            h_interp = (1 - alpha) * h1 + alpha * h2
            theta_interp = (1 - alpha) * theta1 + alpha * theta2
            e_interp = (1 - alpha) * e1 + alpha * e2

            # Build node features with interpolated h
            x_interp = d1.x.clone()
            x_interp[:, 0] = h_interp  # update h feature

            data = Data(
                x=x_interp,
                edge_index=d1.edge_index,
                y=theta_interp,
            )
            data.e_exact = float(e_interp)
            data.h_value = float(h_interp)
            augmented.append(data)

    logger.info(f"Augmented dataset: {len(sorted_ds)} → {len(augmented)} points (alphas={alphas})")
    return augmented
