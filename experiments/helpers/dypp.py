"""F1: Dynamic Parameter Prediction (DyPP) for VQE Acceleration.

Exploits smooth θ(h) trajectories to predict the next optimal parameters
via linear or quadratic extrapolation from previously converged points.

References:
    - arXiv:2307.12449 — Dynamic Parameter Prediction for VQA (2023)
    - Mele et al. (2022) PRA 106, L060401 — parameter transferability in HVA
    - Skogh et al. (2023) Electronic Structure 5, 035002 — parameter transfer
"""

from __future__ import annotations

import numpy as np


def dypp_linear(
    h_prev: list[float],
    theta_prev: list[np.ndarray],
    h_next: float,
) -> np.ndarray:
    """Linear extrapolation from last 2 converged points.

    θ(h_next) ≈ θ(h_1) + (θ(h_1) - θ(h_0)) * (h_next - h_1) / (h_1 - h_0)

    Parameters
    ----------
    h_prev : list[float]
        Last 2 h-values (ordered, most recent last).
    theta_prev : list[np.ndarray]
        Corresponding converged θ vectors.
    h_next : float
        Next h-value to predict.

    Returns
    -------
    np.ndarray
        Predicted θ for h_next.
    """
    if len(h_prev) < 2:
        return theta_prev[-1].copy()

    h0, h1 = h_prev[-2], h_prev[-1]
    t0, t1 = theta_prev[-2], theta_prev[-1]

    dh = h1 - h0
    if abs(dh) < 1e-10:
        return t1.copy()

    slope = (t1 - t0) / dh
    return t1 + slope * (h_next - h1)


def dypp_quadratic(
    h_prev: list[float],
    theta_prev: list[np.ndarray],
    h_next: float,
) -> np.ndarray:
    """Quadratic extrapolation from last 3 converged points.

    Fits a parabola through the last 3 (h, θ) points per parameter
    and extrapolates to h_next.

    Parameters
    ----------
    h_prev : list[float]
        Last 3 h-values (ordered, most recent last).
    theta_prev : list[np.ndarray]
        Corresponding converged θ vectors.
    h_next : float
        Next h-value to predict.

    Returns
    -------
    np.ndarray
        Predicted θ for h_next.
    """
    if len(h_prev) < 3:
        return dypp_linear(h_prev, theta_prev, h_next)

    h_arr = np.array(h_prev[-3:])
    n_params = len(theta_prev[-1])
    theta_pred = np.zeros(n_params)

    for p in range(n_params):
        y = np.array([theta_prev[-3][p], theta_prev[-2][p], theta_prev[-1][p]])
        # Fit quadratic: θ(h) = a*h² + b*h + c
        coeffs = np.polyfit(h_arr, y, 2)
        theta_pred[p] = np.polyval(coeffs, h_next)

    return theta_pred


def dypp_predict(
    h_history: list[float],
    theta_history: list[np.ndarray],
    h_next: float,
    order: int = 2,
    bounds: tuple[float, float] = (-np.pi, np.pi),
) -> np.ndarray:
    """Unified DyPP prediction with fallback and bounds clipping.

    Parameters
    ----------
    h_history : list[float]
        History of converged h-values.
    theta_history : list[np.ndarray]
        History of converged θ vectors.
    h_next : float
        Next h-value to predict.
    order : int
        1 = linear, 2 = quadratic.
    bounds : tuple
        Parameter bounds for clipping.

    Returns
    -------
    np.ndarray
        Predicted θ, clipped to bounds.
    """
    if len(h_history) == 0:
        raise ValueError("Need at least 1 history point for DyPP")

    if len(h_history) == 1:
        # No extrapolation possible — return last known
        return theta_history[-1].copy()

    if order >= 2 and len(h_history) >= 3:
        pred = dypp_quadratic(h_history, theta_history, h_next)
    else:
        pred = dypp_linear(h_history, theta_history, h_next)

    # Clip to bounds
    return np.clip(pred, bounds[0], bounds[1])


def evaluate_dypp_quality(
    theta_pred: np.ndarray,
    theta_opt: np.ndarray,
    theta_warmstart: np.ndarray,
) -> dict:
    """Evaluate DyPP prediction quality vs standard warm-start.

    Returns
    -------
    dict with keys:
        dypp_error: L2 distance from optimal
        warmstart_error: L2 distance from optimal (baseline)
        improvement_pct: (warmstart_error - dypp_error) / warmstart_error * 100
        dypp_better: bool
    """
    dypp_err = float(np.linalg.norm(theta_pred - theta_opt))
    warm_err = float(np.linalg.norm(theta_warmstart - theta_opt))

    improvement = (warm_err - dypp_err) / warm_err * 100 if warm_err > 1e-10 else 0.0

    return {
        "dypp_error": dypp_err,
        "warmstart_error": warm_err,
        "improvement_pct": improvement,
        "dypp_better": dypp_err < warm_err,
    }
