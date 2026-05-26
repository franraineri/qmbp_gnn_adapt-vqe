"""Landscape Analysis — Hessian computation and trainability metrics.

Provides tools for:
- Hessian computation via central finite differences (B4)
- Landscape fluctuation metric for barren plateau detection (F3)
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


def landscape_fluctuation(
    cost_fn: Callable[[np.ndarray], float],
    n_params: int,
    n_samples: int = 100,
    bounds: tuple[float, float] = (-np.pi, np.pi),
    seed: int | None = None,
) -> dict[str, float]:
    """Compute landscape fluctuation metric.

    Fluctuation = Var(E) / E_mean^2

    High fluctuation indicates a trainable landscape with meaningful gradients.
    Low fluctuation indicates a flat/barren landscape.

    Note: When |E_mean| < 1e-10 (possible near criticality where the energy
    landscape is symmetric around zero), the fluctuation is reported as
    float('inf') to indicate that the normalized metric is undefined rather
    than misleadingly returning 0.0. The raw variance is still meaningful
    in this case.

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

    if abs(e_mean) > 1e-10:
        fluctuation = e_var / (e_mean**2)
    else:
        # Mean near zero — normalized fluctuation is undefined.
        # Use inf to signal this rather than a misleading 0.0.
        fluctuation = float("inf") if e_var > 1e-15 else 0.0

    return {
        "fluctuation": float(fluctuation),
        "mean": float(e_mean),
        "variance": float(e_var),
        "std": float(np.std(energies)),
        "min": float(np.min(energies)),
        "max": float(np.max(energies)),
        "n_samples": n_samples,
    }
