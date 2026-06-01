"""SPSA refinement for hardware execution.

Activates only when ΔE/gap > threshold (0.05), using validated parameters
from V7-4A grid search. Uses local RNG for reproducibility and respects
the 10M shot cost ceiling.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qmbp_simulation.execution.hardware.config import HardwareConfig, SPSAConfig
    from qmbp_simulation.framework.logging import StructuredLogger


def spsa_refinement(
    evaluate_fn: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    initial_energy: float,
    e_exact: float,
    gap: float,
    config: HardwareConfig,
    spsa_config: SPSAConfig,
    logger: StructuredLogger,
    rng: np.random.Generator,
    current_total_shots: int,
) -> tuple[np.ndarray, float, bool]:
    """SPSA refinement — activates only if ΔE/gap > threshold.

    Uses local RNG for Bernoulli perturbations, checks cost ceiling before
    each iteration, and never worsens the result (returns better of initial
    vs refined).

    Parameters
    ----------
    evaluate_fn : Energy evaluation callable (params → float).
    initial_params : Parameters from MPNN warm-start.
    initial_energy : Energy evaluated with initial_params.
    e_exact : Exact ground-state energy.
    gap : Spectral gap of the system.
    config : HardwareConfig (shots, n_layouts, max_total_shots, spsa_threshold).
    spsa_config : SPSA hyperparameters (a, c, A, alpha, gamma, n_iterations).
    logger : StructuredLogger for event recording.
    rng : Local np.random.Generator (never np.random global).
    current_total_shots : Shots already consumed for cost ceiling check.

    Returns
    -------
    (best_params, best_energy, spsa_was_applied)
    """
    delta_e_gap = abs(initial_energy - e_exact) / gap if gap > 0 else float("inf")

    if delta_e_gap <= config.spsa_threshold:
        logger.log(
            "spsa_skipped",
            data={
                "delta_e_gap": delta_e_gap,
                "threshold": config.spsa_threshold,
            },
        )
        return initial_params, initial_energy, False

    logger.log(
        "spsa_start",
        data={
            "delta_e_gap": delta_e_gap,
            "n_iterations": spsa_config.n_iterations,
        },
    )

    params = initial_params.copy()
    best_energy = initial_energy
    best_params = initial_params.copy()
    shots_consumed = 0

    for k in range(spsa_config.n_iterations):
        ak = spsa_config.a / (k + 1 + spsa_config.A) ** spsa_config.alpha
        ck = spsa_config.c / (k + 1) ** spsa_config.gamma

        delta = rng.choice([-1, 1], size=len(params)).astype(float)
        params_plus = params + ck * delta
        params_minus = params - ck * delta

        # Check cost ceiling BEFORE evaluating
        shots_this_iter = 2 * config.shots * config.n_layouts
        if current_total_shots + shots_consumed + shots_this_iter > config.max_total_shots:
            logger.log(
                "spsa_abort_cost_ceiling",
                data={
                    "iteration": k,
                    "shots_consumed": shots_consumed,
                },
            )
            break

        e_plus = evaluate_fn(params_plus)
        e_minus = evaluate_fn(params_minus)
        shots_consumed += shots_this_iter

        grad = (e_plus - e_minus) / (2 * ck * delta)
        params = params - ak * grad

        # Track best from evaluated points
        energy = min(e_plus, e_minus)
        if abs(energy - e_exact) < abs(best_energy - e_exact):
            best_energy = energy
            best_params = params.copy()

    logger.log(
        "spsa_complete",
        data={
            "iterations": k + 1 if spsa_config.n_iterations > 0 else 0,
            "shots_consumed": shots_consumed,
            "initial_delta_e_gap": delta_e_gap,
            "final_delta_e_gap": abs(best_energy - e_exact) / gap if gap > 0 else float("inf"),
        },
    )

    # Always return the better of initial vs refined
    if abs(best_energy - e_exact) < abs(initial_energy - e_exact):
        return best_params, best_energy, True

    logger.log(
        "spsa_no_improvement",
        data={
            "initial_error": abs(initial_energy - e_exact),
            "refined_error": abs(best_energy - e_exact),
        },
    )
    return initial_params, initial_energy, True
