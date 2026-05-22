"""Landscape analysis utilities for V8 experiments.

Provides tools for:
- Hessian computation (B4)
- Landscape fluctuation (F3)
- Basin counting and characterization (A2, D3)
- Parameter trajectory analysis (B2, F1)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def compute_hessian(
    cost_fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """Compute the Hessian matrix via central finite differences.

    Parameters
    ----------
    cost_fn : callable
        Energy function E(theta) -> float.
    theta : np.ndarray
        Point at which to compute the Hessian.
    epsilon : float
        Step size for finite differences.

    Returns
    -------
    np.ndarray
        Hessian matrix of shape (n_params, n_params).
        Symmetric by construction.
    """
    n = len(theta)
    H = np.zeros((n, n))

    f0 = cost_fn(theta)

    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Diagonal: (f(x+e) - 2f(x) + f(x-e)) / e^2
                theta_p = theta.copy()
                theta_m = theta.copy()
                theta_p[i] += epsilon
                theta_m[i] -= epsilon
                H[i, i] = (cost_fn(theta_p) - 2 * f0 + cost_fn(theta_m)) / epsilon**2
            else:
                # Off-diagonal: (f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej)) / (4e^2)
                theta_pp = theta.copy()
                theta_pm = theta.copy()
                theta_mp = theta.copy()
                theta_mm = theta.copy()
                theta_pp[i] += epsilon
                theta_pp[j] += epsilon
                theta_pm[i] += epsilon
                theta_pm[j] -= epsilon
                theta_mp[i] -= epsilon
                theta_mp[j] += epsilon
                theta_mm[i] -= epsilon
                theta_mm[j] -= epsilon
                H[i, j] = (
                    cost_fn(theta_pp) - cost_fn(theta_pm) - cost_fn(theta_mp) + cost_fn(theta_mm)
                ) / (4 * epsilon**2)
                H[j, i] = H[i, j]  # Symmetric

    return H


def analyze_critical_point(
    cost_fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    epsilon: float = 1e-4,
) -> dict:
    """Analyze a critical point: is it a minimum, saddle, or maximum?

    Parameters
    ----------
    cost_fn : callable
        Energy function.
    theta : np.ndarray
        Critical point to analyze.
    epsilon : float
        Finite difference step.

    Returns
    -------
    dict with keys:
        hessian: np.ndarray
        eigenvalues: np.ndarray (sorted ascending)
        eigenvectors: np.ndarray
        type: "minimum" | "saddle" | "maximum"
        n_negative: int (number of negative eigenvalues)
        condition_number: float
        escape_direction: np.ndarray | None (eigenvector of most negative eigenvalue)
    """
    H = compute_hessian(cost_fn, theta, epsilon)
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Sort by eigenvalue (ascending)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    n_negative = int(np.sum(eigenvalues < -1e-8))

    if n_negative == 0:
        point_type = "minimum"
    elif n_negative == len(eigenvalues):
        point_type = "maximum"
    else:
        point_type = "saddle"

    # Escape direction: eigenvector of most negative eigenvalue
    escape_direction = None
    if n_negative > 0:
        escape_direction = eigenvectors[:, 0]  # Most negative eigenvalue

    # Condition number (ratio of largest to smallest positive eigenvalue)
    pos_eigs = eigenvalues[eigenvalues > 1e-8]
    condition_number = float(pos_eigs[-1] / pos_eigs[0]) if len(pos_eigs) >= 2 else 1.0

    return {
        "hessian": H,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "type": point_type,
        "n_negative": n_negative,
        "condition_number": condition_number,
        "escape_direction": escape_direction,
    }


def compute_landscape_fluctuation(
    cost_fn: Callable[[np.ndarray], float],
    n_params: int,
    n_samples: int = 100,
    bounds: tuple[float, float] = (-np.pi, np.pi),
    seed: int | None = None,
) -> dict:
    """Compute landscape fluctuation metric.

    Fluctuation = Var(E) / E_mean^2

    High fluctuation indicates a trainable landscape with meaningful gradients.
    Low fluctuation indicates a flat/barren landscape.

    Parameters
    ----------
    cost_fn : callable
        Energy function E(theta) -> float.
    n_params : int
        Number of parameters.
    n_samples : int
        Number of random samples.
    bounds : tuple
        Parameter bounds (default [-pi, pi]).
    seed : int | None
        Random seed.

    Returns
    -------
    dict with keys:
        fluctuation, mean, variance, std, min, max, n_samples
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    energies = np.zeros(n_samples)
    for i in range(n_samples):
        theta = rng.uniform(bounds[0], bounds[1], n_params)
        energies[i] = cost_fn(theta)

    e_mean = np.mean(energies)
    e_var = np.var(energies)
    fluctuation = e_var / (e_mean**2) if abs(e_mean) > 1e-10 else 0.0

    return {
        "fluctuation": float(fluctuation),
        "mean": float(e_mean),
        "variance": float(e_var),
        "std": float(np.std(energies)),
        "min": float(np.min(energies)),
        "max": float(np.max(energies)),
        "n_samples": n_samples,
    }


def compute_theta_smoothness(theta_array: np.ndarray) -> dict:
    """Analyze smoothness of theta trajectory across h-sweep.

    Parameters
    ----------
    theta_array : np.ndarray
        Shape (n_points, n_params), ordered by h.

    Returns
    -------
    dict with keys:
        max_jump: maximum L-inf jump between adjacent points
        mean_jump: mean L-inf jump
        smoothness_score: 1 / (1 + max_jump) — higher is smoother
        jumps: list of per-point jumps
    """
    n_points = len(theta_array)
    if n_points < 2:
        return {"max_jump": 0.0, "mean_jump": 0.0, "smoothness_score": 1.0, "jumps": []}

    jumps = []
    for i in range(1, n_points):
        jump = float(np.max(np.abs(theta_array[i] - theta_array[i - 1])))
        jumps.append(jump)

    max_jump = max(jumps)
    mean_jump = float(np.mean(jumps))

    return {
        "max_jump": max_jump,
        "mean_jump": mean_jump,
        "smoothness_score": 1.0 / (1.0 + max_jump),
        "jumps": jumps,
    }
