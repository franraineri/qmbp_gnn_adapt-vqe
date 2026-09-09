"""h-grid generation utilities — non-uniform and frontier-dense sampling.

Pure numpy helpers (no pipeline/torch/qiskit deps) so ANY layer can build an
h-grid without importing from ``pipeline`` (which is orchestration). This is the
single home for h-grid density logic; ``pipeline.dataset_io`` re-exports these
for backward compatibility.

Both generators return a DESCENDING grid (h_max → h_min), the order the
warm-start VQE sweep expects.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def generate_nonuniform_h_grid(
    h_min: float,
    h_max: float,
    n_points: int,
    h_critical: float | None = None,
    dense_fraction: float = 0.4,
    dense_radius: float = 0.5,
) -> np.ndarray:
    """Generate a non-uniform h-grid with denser sampling near h_critical.

    Produces a descending grid (h_max → h_min) with more points concentrated
    around the critical region where the phase transition occurs and the VQE
    landscape is most complex.

    Parameters
    ----------
    h_min : float
        Minimum field value.
    h_max : float
        Maximum field value.
    n_points : int
        Total number of grid points.
    h_critical : float | None
        Critical field value for dense sampling. If None, uses midpoint.
    dense_fraction : float
        Fraction of points allocated to the dense region (default 0.4 = 40%).
    dense_radius : float
        Half-width of the dense region around h_critical (default 0.5).

    Returns
    -------
    np.ndarray
        Sorted descending h-values with denser sampling near h_critical.

    Examples
    --------
    >>> grid = generate_nonuniform_h_grid(1.0, 5.0, 30, h_critical=1.5)
    >>> len(grid)
    30
    >>> grid[0] > grid[-1]  # descending
    True
    """
    if h_min >= h_max:
        raise ValueError(f"h_min ({h_min}) must be < h_max ({h_max}).")
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}.")

    if h_critical is None:
        h_critical = (h_min + h_max) / 2.0

    logger.debug(
        "generate_nonuniform_h_grid: [%.2f, %.2f], n=%d, h_c=%.2f, dense_frac=%.1f, radius=%.2f",
        h_min,
        h_max,
        n_points,
        h_critical,
        dense_fraction,
        dense_radius,
    )

    # Clamp h_critical to valid range
    h_critical = max(h_min + dense_radius * 0.5, min(h_max - dense_radius * 0.5, h_critical))

    # Split points between dense and sparse regions
    n_dense = max(3, int(n_points * dense_fraction))
    n_sparse = n_points - n_dense

    # Dense region: ±dense_radius around h_critical
    dense_lo = max(h_min, h_critical - dense_radius)
    dense_hi = min(h_max, h_critical + dense_radius)
    dense_points = np.linspace(dense_lo, dense_hi, n_dense)

    # Sparse region: remainder of [h_min, h_max] excluding dense zone
    sparse_points = []
    n_below = n_sparse // 2
    n_above = n_sparse - n_below

    if dense_lo > h_min and n_below > 0:
        sparse_points.extend(np.linspace(h_min, dense_lo, n_below + 1)[:-1].tolist())
    if dense_hi < h_max and n_above > 0:
        sparse_points.extend(np.linspace(dense_hi, h_max, n_above + 1)[1:].tolist())

    # Combine, deduplicate, sort descending
    all_points = np.concatenate([dense_points, np.array(sparse_points)])
    all_points = np.unique(all_points)

    # Ensure exactly n_points by resampling from the combined range.
    # Interpolating preserves the density pattern while guaranteeing exact count.
    if len(all_points) != n_points:
        target_indices = np.linspace(0, len(all_points) - 1, n_points)
        all_points = np.interp(target_indices, np.arange(len(all_points)), all_points)

    # Sort descending (for warm-start sweep)
    return np.sort(all_points)[::-1]


def generate_frontier_dense_h_grid(
    h_min: float,
    h_max: float,
    n_points: int,
    h_frontier: float,
    *,
    dense_fraction: float = 0.5,
    dense_radius: float | None = None,
    include_below_frontier: bool = True,
) -> np.ndarray:
    """Generate h-grid with dense sampling around the empirical frontier.

    Unlike ``generate_nonuniform_h_grid`` which densifies around the
    *theoretical* critical point h_c, this function densifies around the
    *empirical* h_frontier — the boundary where ΔE/gap crosses 5% for
    a specific (topology, N, p) configuration.

    The frontier is where the pipeline transitions from "pass" to "fail",
    and denser sampling there produces:
    1. Better MPNN training data in the hardest region
    2. More accurate h_frontier estimates
    3. Higher pass rates (more points in the "easy" zone above frontier)

    Parameters
    ----------
    h_min : float
        Minimum field value.
    h_max : float
        Maximum field value.
    n_points : int
        Total number of grid points.
    h_frontier : float
        Empirical h_frontier from ``compute_h_frontier_from_npz`` or
        ``get_empirical_h_frontier()``. Points will be concentrated here.
    dense_fraction : float
        Fraction of points allocated to the frontier region (default 0.5).
        Higher → more resolution at the boundary, fewer in safe zone.
    dense_radius : float | None
        Half-width of the dense region around h_frontier.
        Default: ``0.15 * (h_max - h_min)`` (adaptive to range).
    include_below_frontier : bool
        If True (default), include points below h_frontier (the "hard"
        zone). If False, only sample h >= h_frontier (conservative: only
        the pass region + boundary). Useful for deployment grids.

    Returns
    -------
    np.ndarray
        Sorted descending h-values with denser sampling near h_frontier.

    Examples
    --------
    >>> grid = generate_frontier_dense_h_grid(1.5, 5.5, 25, h_frontier=2.3)
    >>> len(grid)
    25
    >>> # Dense region around 2.3 should have smaller spacing
    >>> near_frontier = grid[(grid > 1.8) & (grid < 2.8)]
    >>> far_from_frontier = grid[grid > 4.0]
    >>> np.mean(np.diff(np.sort(near_frontier))) < np.mean(np.diff(np.sort(far_from_frontier)))
    True
    """
    if h_min >= h_max:
        raise ValueError(f"h_min ({h_min}) must be < h_max ({h_max}).")
    if n_points < 3:
        return np.linspace(h_max, h_min, max(n_points, 1))

    # Adaptive dense_radius: 15% of total range
    if dense_radius is None:
        dense_radius = 0.15 * (h_max - h_min)

    # Clamp frontier to valid range
    effective_frontier = max(h_min, min(h_max, h_frontier))

    # Effective h_min for grid (respects include_below_frontier)
    effective_h_min = (
        h_min if include_below_frontier else max(h_min, effective_frontier - dense_radius)
    )

    # Split points: dense zone around frontier + sparse elsewhere
    n_dense = max(5, int(n_points * dense_fraction))
    n_sparse = n_points - n_dense

    # Dense region boundaries
    dense_lo = max(effective_h_min, effective_frontier - dense_radius)
    dense_hi = min(h_max, effective_frontier + dense_radius)

    # Asymmetric densification: slightly more points BELOW frontier
    # (that's where refinement helps most)
    n_below_frontier = int(n_dense * 0.6)
    n_above_frontier = n_dense - n_below_frontier

    dense_below = np.linspace(dense_lo, effective_frontier, n_below_frontier, endpoint=False)
    dense_above = np.linspace(effective_frontier, dense_hi, n_above_frontier)
    dense_points = np.concatenate([dense_below, dense_above])

    # Sparse points in the remaining range
    sparse_points = []
    n_below_dense = n_sparse // 3  # fewer below (harder region)
    n_above_dense = n_sparse - n_below_dense  # more above (easy region for training signal)

    if dense_lo > effective_h_min and n_below_dense > 0:
        sparse_points.extend(
            np.linspace(effective_h_min, dense_lo, n_below_dense + 1)[:-1].tolist()
        )
    if dense_hi < h_max and n_above_dense > 0:
        sparse_points.extend(np.linspace(dense_hi, h_max, n_above_dense + 1)[1:].tolist())

    # Combine, deduplicate, sort descending
    all_points = np.concatenate([dense_points, np.array(sparse_points)])
    all_points = np.unique(all_points)

    # Ensure exactly n_points
    if len(all_points) != n_points:
        target_indices = np.linspace(0, len(all_points) - 1, n_points)
        all_points = np.interp(target_indices, np.arange(len(all_points)), all_points)

    return np.sort(all_points)[::-1]
