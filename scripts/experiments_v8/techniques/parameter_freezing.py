"""B2: TITAN-Style Parameter Freezing for HVA.

Analyzes VQE parameter trajectories across h-sweep to identify parameters
that are effectively frozen (|dθ/dh| < threshold). These can be fixed at
their converged values, reducing optimization dimensionality.

References:
    - Peng et al. (2025) TITAN, NeurIPS, arXiv:2509.15193
    - Wiersema et al. (2020) PRX Quantum 1, 020319 — HVA structure
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def analyze_parameter_activity(
    h_values: np.ndarray,
    theta_trajectory: np.ndarray,
    threshold: float = 0.05,
) -> dict:
    """Analyze which parameters are active vs frozen across h-sweep.

    Parameters
    ----------
    h_values : np.ndarray
        Array of h-values (shape: n_points).
    theta_trajectory : np.ndarray
        Array of optimal θ at each h (shape: n_points × n_params).
    threshold : float
        Maximum |dθ/dh| to consider a parameter "frozen".

    Returns
    -------
    dict with keys:
        derivatives: np.ndarray (n_points-1 × n_params) — |dθ/dh| at each interval
        mean_activity: np.ndarray (n_params,) — mean |dθ/dh| per parameter
        frozen_mask: np.ndarray (n_params,) — True if parameter is frozen
        n_frozen: int
        frozen_indices: list[int]
        active_indices: list[int]
        activity_by_h: list[dict] — per-interval activity breakdown
    """
    n_points, n_params = theta_trajectory.shape
    derivatives = np.zeros((n_points - 1, n_params))

    for i in range(n_points - 1):
        dh = h_values[i + 1] - h_values[i]
        if abs(dh) > 1e-10:
            derivatives[i] = np.abs(theta_trajectory[i + 1] - theta_trajectory[i]) / abs(dh)

    mean_activity = np.mean(derivatives, axis=0)
    frozen_mask = mean_activity < threshold
    frozen_indices = list(np.where(frozen_mask)[0])
    active_indices = list(np.where(~frozen_mask)[0])

    activity_by_h = []
    for i in range(n_points - 1):
        h_mid = (h_values[i] + h_values[i + 1]) / 2
        activity_by_h.append(
            {
                "h_mid": float(h_mid),
                "derivatives": derivatives[i].tolist(),
                "n_frozen": int(np.sum(derivatives[i] < threshold)),
            }
        )

    return {
        "derivatives": derivatives,
        "mean_activity": mean_activity,
        "frozen_mask": frozen_mask,
        "n_frozen": int(np.sum(frozen_mask)),
        "frozen_indices": frozen_indices,
        "active_indices": active_indices,
        "activity_by_h": activity_by_h,
    }


def frozen_vqe(
    cost_fn_factory: Callable[[np.ndarray, list[int]], Callable],
    initial_guess: np.ndarray,
    frozen_indices: list[int],
    frozen_values: np.ndarray | None = None,
    n_restarts: int = 5,
    sigma: float = 0.1,
    maxiter: int = 500,
    ftol: float = 1e-14,
) -> dict:
    """Run VQE with specified parameters frozen.

    Parameters
    ----------
    cost_fn_factory : callable
        Factory that takes (frozen_values, frozen_indices) and returns
        a cost function over the ACTIVE parameters only.
    initial_guess : np.ndarray
        Full parameter vector (including frozen params).
    frozen_indices : list[int]
        Indices of parameters to freeze.
    frozen_values : np.ndarray | None
        Values to freeze at (default: use initial_guess values).
    n_restarts : int
        Number of restarts for active parameters.
    sigma : float
        Perturbation scale.
    maxiter : int
        Max iterations.
    ftol : float
        Function tolerance.

    Returns
    -------
    dict with keys:
        theta_opt: full parameter vector (frozen + optimized)
        energy: best energy
        n_active_params: number of optimized parameters
        n_frozen_params: number of frozen parameters
        total_evaluations: total cost function calls
    """
    from scipy.optimize import minimize

    n_params = len(initial_guess)
    active_indices = [i for i in range(n_params) if i not in frozen_indices]
    n_active = len(active_indices)

    if frozen_values is None:
        frozen_values = initial_guess[frozen_indices]

    # Build cost function over active params only
    def active_cost_fn(active_params):
        full_params = np.zeros(n_params)
        full_params[frozen_indices] = frozen_values
        full_params[active_indices] = active_params
        # Use the full cost function
        return cost_fn_factory(full_params)

    # Multi-start over active params
    active_init = initial_guess[active_indices]
    bounds = [(-np.pi, np.pi)] * n_active

    best_energy = float("inf")
    best_active = active_init.copy()
    total_evals = 0

    for restart in range(n_restarts):
        if restart == 0:
            x0 = active_init.copy()
        else:
            x0 = best_active + np.random.normal(0, sigma, n_active)
            x0 = np.clip(x0, -np.pi, np.pi)

        result = minimize(
            active_cost_fn,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": ftol},
        )
        total_evals += result.nfev

        if result.fun < best_energy:
            best_energy = result.fun
            best_active = result.x.copy()

    # Reconstruct full parameter vector
    theta_opt = np.zeros(n_params)
    theta_opt[frozen_indices] = frozen_values
    theta_opt[active_indices] = best_active

    return {
        "theta_opt": theta_opt,
        "energy": best_energy,
        "n_active_params": n_active,
        "n_frozen_params": len(frozen_indices),
        "total_evaluations": total_evals,
        "frozen_indices": frozen_indices,
        "frozen_values": frozen_values.tolist(),
    }
