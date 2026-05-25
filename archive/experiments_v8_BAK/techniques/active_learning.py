"""E3: Active Learning for Optimal h-Grid Selection.

Uses an ensemble of MPNNs to estimate prediction uncertainty, then selects
the next h-point to maximize information gain (reduce uncertainty).

References:
    - Miao et al. (2024) PRApplied 21, 014053 — active learning for NN-VQE
    - Zhang et al. (2025) arXiv:2505.01236 (Qracle) — data-efficient GNN init
"""

from __future__ import annotations

import numpy as np


def compute_ensemble_uncertainty(
    predictions: list[np.ndarray],
) -> dict:
    """Compute uncertainty from ensemble predictions.

    Parameters
    ----------
    predictions : list[np.ndarray]
        List of predictions from each ensemble member.
        Each element has shape (n_params,).

    Returns
    -------
    dict with keys:
        mean: np.ndarray — ensemble mean prediction
        std: np.ndarray — per-parameter standard deviation
        variance: float — total variance (sum of per-param variances)
        max_std: float — maximum per-parameter std
    """
    preds = np.array(predictions)  # (n_ensemble, n_params)
    mean = np.mean(preds, axis=0)
    std = np.std(preds, axis=0)
    variance = float(np.sum(std**2))
    max_std = float(np.max(std))

    return {
        "mean": mean,
        "std": std,
        "variance": variance,
        "max_std": max_std,
    }


def max_variance_acquisition(
    candidate_h: np.ndarray,
    uncertainties: list[float],
) -> int:
    """Select h-point with maximum prediction variance.

    Parameters
    ----------
    candidate_h : np.ndarray
        Array of candidate h-values.
    uncertainties : list[float]
        Uncertainty (variance) at each candidate.

    Returns
    -------
    int
        Index of selected h-point.
    """
    return int(np.argmax(uncertainties))


def expected_improvement_acquisition(
    candidate_h: np.ndarray,
    uncertainties: list[float],
    current_best_error: float,
    predictions_mean: list[float],
) -> int:
    """Select h-point with maximum expected improvement.

    Balances exploration (high uncertainty) with exploitation
    (predicted to be near current best).

    Parameters
    ----------
    candidate_h : np.ndarray
        Candidate h-values.
    uncertainties : list[float]
        Uncertainty at each candidate.
    current_best_error : float
        Current best deployment error.
    predictions_mean : list[float]
        Mean predicted error at each candidate.

    Returns
    -------
    int
        Index of selected h-point.
    """
    from scipy.stats import norm

    ei_values = []
    for i in range(len(candidate_h)):
        sigma = uncertainties[i]
        mu = predictions_mean[i] if predictions_mean else 0.0

        if sigma < 1e-10:
            ei_values.append(0.0)
            continue

        z = (current_best_error - mu) / sigma
        ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
        ei_values.append(ei)

    return int(np.argmax(ei_values))


def select_next_point(
    candidate_h: np.ndarray,
    uncertainties: list[float],
    acquisition: str = "max_variance",
    current_best_error: float = 0.0,
    predictions_mean: list[float] | None = None,
) -> tuple[int, float]:
    """Select next h-point using specified acquisition function.

    Parameters
    ----------
    candidate_h : np.ndarray
        Available h-values to choose from.
    uncertainties : list[float]
        Uncertainty at each candidate.
    acquisition : str
        "max_variance" or "expected_improvement".
    current_best_error : float
        Current best error (for EI).
    predictions_mean : list[float] | None
        Mean predictions (for EI).

    Returns
    -------
    tuple[int, float]
        (index, h_value) of selected point.
    """
    if acquisition == "max_variance":
        idx = max_variance_acquisition(candidate_h, uncertainties)
    elif acquisition == "expected_improvement":
        idx = expected_improvement_acquisition(
            candidate_h,
            uncertainties,
            current_best_error,
            predictions_mean or [0.0] * len(candidate_h),
        )
    else:
        raise ValueError(f"Unknown acquisition: {acquisition}")

    return idx, float(candidate_h[idx])


def should_stop(
    uncertainties: list[float],
    threshold: float = 0.01,
    test_indices: list[int] | None = None,
) -> bool:
    """Check if active learning should stop.

    Stops when uncertainty at test points drops below threshold.

    Parameters
    ----------
    uncertainties : list[float]
        Current uncertainties at all candidate points.
    threshold : float
        Variance threshold for stopping.
    test_indices : list[int] | None
        Indices of test points to check (if None, check all).

    Returns
    -------
    bool
        True if should stop.
    """
    if test_indices:
        test_uncert = [uncertainties[i] for i in test_indices if i < len(uncertainties)]
    else:
        test_uncert = uncertainties

    if not test_uncert:
        return True

    return max(test_uncert) < threshold
