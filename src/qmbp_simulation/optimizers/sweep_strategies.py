"""VQE Sweep Strategies — Adaptive restarts and selective ascending pass.

Provides reusable strategies for optimizing VQE parameter sweeps:

1. **Adaptive restarts**: Adjusts n_restarts per h-point based on the
   convergence quality of neighboring points. Points near h_critical or
   with high ΔE/gap get more restarts; trivial points get fewer.

2. **Selective ascending pass**: Instead of re-optimizing ALL points in the
   ascending direction, only targets "suspicious" points where ΔE/gap
   exceeds a threshold. Saves ~50-80% of ascending pass cost.

Usage:
    from qmbp_simulation.optimizers.sweep_strategies import (
        compute_adaptive_restarts,
        select_suspicious_points,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# A. Adaptive Restarts
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AdaptiveRestartConfig:
    """Configuration for adaptive restart allocation.

    Parameters
    ----------
    base_restarts : int
        Default restarts for well-converged points (far from h_c). Default: 2.
    max_restarts : int
        Maximum restarts for difficult points (near h_c or high ΔE/gap). Default: 7.
    critical_restarts : int
        Restarts for points within the critical region. Default: 5.
    de_gap_threshold : float
        If neighbor's ΔE/gap exceeds this, increase restarts. Default: 0.02.
    h_critical : float | None
        Critical field value. Points within critical_radius get more restarts.
        If None, no region-based boost is applied.
    critical_radius : float
        Half-width of the critical region around h_critical. Default: 0.3.
    """

    base_restarts: int = 2
    max_restarts: int = 7
    critical_restarts: int = 5
    de_gap_threshold: float = 0.02
    h_critical: float | None = None
    critical_radius: float = 0.3


def compute_adaptive_restarts(
    h_value: float,
    *,
    prev_de_gap: float | None = None,
    config: AdaptiveRestartConfig | None = None,
) -> int:
    """Compute the number of restarts for a specific h-point.

    Uses the convergence quality of the previous point (neighbor) and
    the proximity to h_critical to determine how many restarts are needed.

    Parameters
    ----------
    h_value : float
        The current h-value being optimized.
    prev_de_gap : float | None
        ΔE/gap from the previous h-point (neighbor signal).
        If None (first point), uses critical_restarts.
    config : AdaptiveRestartConfig | None
        Configuration. If None, uses defaults.

    Returns
    -------
    int
        Number of restarts to use for this h-point.

    Examples
    --------
    >>> compute_adaptive_restarts(1.5, prev_de_gap=0.001)  # easy neighbor
    2
    >>> compute_adaptive_restarts(1.0, prev_de_gap=0.05)  # hard neighbor
    7
    >>> compute_adaptive_restarts(1.0, prev_de_gap=None)  # first point
    5
    """
    from qmbp_simulation.models.constants import MIN_N_RESTARTS

    if config is None:
        config = AdaptiveRestartConfig()

    # First point in sweep (no neighbor info): use critical restarts
    if prev_de_gap is None:
        return max(config.critical_restarts, MIN_N_RESTARTS)

    n_restarts = config.base_restarts

    # Boost 1: previous point had difficulty converging
    if prev_de_gap > config.de_gap_threshold:
        n_restarts = max(n_restarts, config.critical_restarts)
    if prev_de_gap > config.de_gap_threshold * 5:
        n_restarts = config.max_restarts

    # Boost 2: proximity to h_critical
    if config.h_critical is not None:
        dist_to_critical = abs(h_value - config.h_critical)
        if dist_to_critical < config.critical_radius:
            n_restarts = max(n_restarts, config.critical_restarts)

    # Hard floor: even trivial points get at least MIN_N_RESTARTS.
    n_restarts = max(n_restarts, MIN_N_RESTARTS)
    return min(n_restarts, max(config.max_restarts, MIN_N_RESTARTS))


def compute_restarts_for_sweep(
    h_values: list[float] | np.ndarray,
    *,
    config: AdaptiveRestartConfig | None = None,
) -> list[int]:
    """Pre-compute adaptive restarts for an entire h-sweep (no neighbor data yet).

    This gives a static estimate based on proximity to h_critical only.
    For dynamic adaptation during sweep, call compute_adaptive_restarts()
    per-point with the actual prev_de_gap.

    Parameters
    ----------
    h_values : array-like
        The h-grid (descending order expected).
    config : AdaptiveRestartConfig | None
        Configuration.

    Returns
    -------
    list[int]
        Restart count per h-point.
    """
    from qmbp_simulation.models.constants import MIN_N_RESTARTS

    if config is None:
        config = AdaptiveRestartConfig()

    restarts = []
    for h in h_values:
        # Static: only use h_critical proximity (no neighbor signal)
        n = config.base_restarts
        if config.h_critical is not None:
            dist = abs(h - config.h_critical)
            if dist < config.critical_radius:
                n = config.critical_restarts
        # Hard floor MIN_N_RESTARTS applies even to trivial points.
        n = max(n, MIN_N_RESTARTS)
        restarts.append(min(n, max(config.max_restarts, MIN_N_RESTARTS)))
    return restarts


# ═══════════════════════════════════════════════════════════════════════════════
# C. Selective Ascending Pass
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SelectiveAscendingConfig:
    """Configuration for the selective ascending pass.

    Parameters
    ----------
    de_gap_threshold : float
        Points with ΔE/gap above this are targeted for re-optimization.
        Default: 0.02 (2% of gap — tighter than the 5% pass criterion).
    include_neighbors : bool
        Also re-optimize the immediate neighbors of suspicious points
        (they may benefit from the improved θ propagation). Default: True.
    max_fraction : float
        Maximum fraction of points to re-optimize. If more than this
        fraction is suspicious, fall back to full ascending pass.
        Default: 0.5 (if >50% are suspicious, just do full pass).
    min_points : int
        Always re-optimize at least this many points (even if none are
        suspicious, re-optimize the worst N). Default: 0.
    """

    de_gap_threshold: float = 0.02
    include_neighbors: bool = True
    max_fraction: float = 0.5
    min_points: int = 0


@dataclass
class SelectiveAscendingReport:
    """Report from the selective ascending pass."""

    n_total_points: int = 0
    n_suspicious: int = 0
    n_targeted: int = 0
    n_improved: int = 0
    fell_back_to_full: bool = False
    targeted_indices: list[int] = field(default_factory=list)
    improvements: list[dict[str, Any]] = field(default_factory=list)


def select_suspicious_points(
    results: list[dict[str, Any]],
    *,
    config: SelectiveAscendingConfig | None = None,
) -> tuple[list[int], SelectiveAscendingReport]:
    """Identify points that should be re-optimized in the ascending pass.

    Instead of re-optimizing all points, only targets those where:
    - ΔE/gap > threshold (the primary signal)
    - Neighbors of suspicious points (optional, for θ propagation benefit)

    Parameters
    ----------
    results : list[dict]
        VQE results per h-point. Each dict must have at least:
        - "de_gap": float (ΔE/gap for that point)
        - "h": float

    config : SelectiveAscendingConfig | None
        Configuration. If None, uses defaults.

    Returns
    -------
    tuple[list[int], SelectiveAscendingReport]
        (indices_to_reoptimize, report)
        Indices are in ascending-h order (for the ascending pass traversal).

    Examples
    --------
    >>> results = [{"h": 2.0, "de_gap": 0.001}, {"h": 1.5, "de_gap": 0.08}, {"h": 1.0, "de_gap": 0.003}]
    >>> indices, report = select_suspicious_points(results)
    >>> indices  # Only point at index 1 (h=1.5) is suspicious
    [1]
    """
    if config is None:
        config = SelectiveAscendingConfig()

    n_total = len(results)
    report = SelectiveAscendingReport(n_total_points=n_total)

    if n_total == 0:
        return [], report

    # Identify suspicious points
    suspicious = set()
    for i, r in enumerate(results):
        de_gap = r.get("de_gap", 0)
        if de_gap is not None and de_gap > config.de_gap_threshold:
            suspicious.add(i)

    report.n_suspicious = len(suspicious)

    # Check if we should fall back to full pass
    if len(suspicious) > n_total * config.max_fraction:
        # Too many suspicious → full ascending pass is more efficient
        report.fell_back_to_full = True
        report.n_targeted = n_total
        all_indices = list(range(n_total - 2, -1, -1))  # ascending order
        report.targeted_indices = all_indices
        return all_indices, report

    # Add neighbors of suspicious points (for θ propagation benefit)
    targeted = set(suspicious)
    if config.include_neighbors:
        for i in list(suspicious):
            if i > 0:
                targeted.add(i - 1)
            if i < n_total - 1:
                targeted.add(i + 1)

    # If min_points requested, add the worst points
    if config.min_points > 0 and len(targeted) < config.min_points:
        sorted_by_de_gap = sorted(
            range(n_total),
            key=lambda i: results[i].get("de_gap", 0) or 0,
            reverse=True,
        )
        for i in sorted_by_de_gap:
            if len(targeted) >= config.min_points:
                break
            targeted.add(i)

    # Return in ascending-h order (last index first for ascending pass)
    targeted_list = sorted(targeted, reverse=True)
    report.n_targeted = len(targeted_list)
    report.targeted_indices = targeted_list
    return targeted_list, report
