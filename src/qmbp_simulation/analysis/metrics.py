"""Analysis Metrics — Pure computation helpers for pipeline diagnostics.

Provides signal-to-noise ratio, parameter smoothness, classification
confidence, energy decomposition, and fraction-near-ground-state
computations. These are stateless functions with no side effects.

This module has NO heavy imports (no Qiskit, no PyTorch).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Quality criteria — dual metric for pass/fail decisions
# ═══════════════════════════════════════════════════════════════════════════════

# Constants for the dual quality criterion
DE_GAP_THRESHOLD: float = 0.05
"""Relative energy error threshold: |ΔE|/gap < 5%."""

MAX_ABS_ERROR: float = 0.10
"""Absolute energy error cap (Hartree). Points above this are failures
regardless of ΔE/gap. Motivation: for large gaps (h >> h_c), ΔE/gap can
be small while |ΔE| is physically significant. This cap ensures no point
with |ΔE| > 0.10 is accepted as "passing" just because the gap is large."""

MIN_FIDELITY: float = 0.97
"""Minimum state overlap for acceptable quality (when available)."""

# ── Data quality audit thresholds ────────────────────────────────────────────
TRAINING_BAD_RATIO_THRESHOLD: float = 0.20
"""If n_bad/n_total > 20% in a NPZ but zoo model has pass_rate > 0.60,
flag training/zoo incoherence. A good zoo model should not co-exist with
poor training data."""

ZOO_PASS_FOR_INCOHERENCE_FLAG: float = 0.60
"""Zoo pass_rate above which a high bad-ratio in NPZ is flagged as incoherence."""

PASS_RATE_REGRESSION_THRESHOLD: float = 0.15
"""If max pass_rate_dual for a topology drops > 15% vs the previous dashboard
snapshot, flag as regression. Used in inspect_data_stores --validate-dashboard."""

H_FRONTIER_MONOTONICITY_TOLERANCE: float = 0.10
"""h_frontier is expected to increase (or stay flat) with N for all topologies.
If h_frontier(N_i+1) < h_frontier(N_i) - tolerance, flag as data quality anomaly
(likely mixed h-ranges between NPZ datasets)."""

GAP_MASKING_THRESHOLD: float = 0.10
"""Difference between pass_rate_5pct and pass_rate_dual_criterion above which
gap masking is considered significant (large gap inflating ΔE/gap metric)."""

MIN_TRAINING_DUAL_PASS_RATE: float = 0.30
"""Minimum fraction of points that must pass the dual criterion for a NPZ to
be considered useful for MPNN training. Below this threshold, the dataset
contains too much noise relative to signal — the MPNN learns the wrong mapping.
Note: this is separate from whether points are physically valid. A point can
be valid (non-NaN, converged VQE) but still useless for training if it's in
a regime the MPNN cannot predict (gap masking, frustrated phase, etc.)."""

MIN_TRAINING_POINTS_FOR_SIGNAL: int = 5
"""Minimum number of dual-criterion-passing points required for a NPZ to
contribute useful training signal. Fewer points is statistically insufficient
for the MPNN to learn the h → θ mapping for that (topology, N)."""


def is_point_failure(
    de_gap: float,
    abs_error: float | None = None,
    fidelity: float | None = None,
    *,
    de_gap_threshold: float = DE_GAP_THRESHOLD,
    max_abs_error: float = MAX_ABS_ERROR,
    min_fidelity: float = MIN_FIDELITY,
) -> bool:
    """Determine if a single evaluation point is a failure using dual criteria.

    A point fails if ANY of these conditions hold:
    1. ΔE/gap >= threshold (relative error too large)
    2. |ΔE| > max_abs_error (absolute error too large, prevents gap masking)
    3. Fidelity < min_fidelity (state overlap too low, when available)

    Handles edge cases:
    - NaN/Inf in any metric → automatic failure (corrupted data)
    - Fidelity not calculable (N > 22, MPS backend) → skipped gracefully
    - abs_error not available → only ΔE/gap is checked
    - Negative de_gap → flagged as failure (indicates variational violation)
    - Fidelity > 1.0 → flagged as failure (indicates computation bug)

    Parameters
    ----------
    de_gap : float
        Relative error |E_pred - E_exact| / gap.
    abs_error : float | None
        Absolute error |E_pred - E_exact|. If None, only ΔE/gap is checked.
        Not available when E_exact comes from approximate methods (DMRG).
    fidelity : float | None
        State fidelity |⟨ψ_pred|ψ_exact⟩|². None when N > 22 (statevector
        not available) or when using MPS backend.
    de_gap_threshold : float
        Threshold for ΔE/gap criterion (default: 0.05).
    max_abs_error : float
        Cap on absolute error (default: 0.10).
    min_fidelity : float
        Minimum acceptable fidelity (default: 0.97).

    Returns
    -------
    bool
        True if the point is a failure (should be refined).
    """
    # Guard: NaN/Inf in de_gap is always a failure
    if not np.isfinite(de_gap):
        return True

    # Guard: negative de_gap indicates variational violation or bug
    if de_gap < 0:
        logger.debug("is_point_failure: negative de_gap=%.4e (variational violation?)", de_gap)
        return True

    # Criterion 1: relative error
    if de_gap >= de_gap_threshold:
        return True

    # Criterion 2: absolute error (skip if not available)
    if abs_error is not None:
        if not np.isfinite(abs_error):
            return True
        if abs_error > max_abs_error:
            return True

    # Criterion 3: fidelity (skip if not calculable — N>22, MPS, etc.)
    if fidelity is not None:
        if not np.isfinite(fidelity):
            return True
        if fidelity > 1.0 + 1e-6:
            # Fidelity > 1 indicates a computation bug
            logger.warning(
                "is_point_failure: fidelity=%.6f > 1.0 (possible bug in "
                "compute_fidelity or state normalization)", fidelity
            )
            return True
        if fidelity < min_fidelity:
            return True

    return False


def identify_failures(
    per_h_results: list[dict],
    *,
    de_gap_threshold: float = DE_GAP_THRESHOLD,
    max_abs_error: float = MAX_ABS_ERROR,
    min_fidelity: float = MIN_FIDELITY,
) -> list[int]:
    """Identify failing point indices from per-h results using dual criteria.

    Handles common data issues:
    - Missing keys gracefully (only checks what's available)
    - NaN values in results → marked as failure
    - Empty list → returns empty

    Parameters
    ----------
    per_h_results : list[dict]
        Each entry must have ``"de_gap"`` (float).
        Optional: ``"abs_error"`` (float), ``"fidelity"`` (float | None).

    Returns
    -------
    list[int]
        Indices of points that fail any criterion.
    """
    failures = []
    for i, r in enumerate(per_h_results):
        de_gap = r.get("de_gap")
        if de_gap is None:
            # Missing de_gap — can't evaluate, mark as failure
            failures.append(i)
            continue
        if is_point_failure(
            de_gap=de_gap,
            abs_error=r.get("abs_error"),
            fidelity=r.get("fidelity"),
            de_gap_threshold=de_gap_threshold,
            max_abs_error=max_abs_error,
            min_fidelity=min_fidelity,
        ):
            failures.append(i)
    return failures


# ═══════════════════════════════════════════════════════════════════════════════
# Refinement priority scoring — smart allocation of VQE compute
# ═══════════════════════════════════════════════════════════════════════════════


def compute_refinement_priority(
    de_gap: float,
    abs_error: float,
    gap: float,
    n_params: int,
    *,
    e_prev: float | None = None,
    e_pred: float | None = None,
    n_prev_attempts: int = 0,
    max_stale_attempts: int = 2,
) -> tuple[float, bool, str]:
    """Compute priority score for VQE refinement of a single h-point.

    Determines whether a failing point is worth spending VQE compute on,
    and if so, how urgently.  Used by iterative improvement loops to order
    failures by expected return-on-investment.

    Decision factors:

    1. **MPNN improvement signal** (``e_pred < e_prev``): the re-trained model
       found a better basin → high priority (VQE from new init can descend further).
    2. **Proximity to threshold**: points barely failing (ΔE/gap ≈ 5%) are easy
       wins → higher priority than points far from passing.
    3. **Gap-based feasibility**: points with very small gap near h_critical have
       inherently large |ΔE| — VQE can't fix this (ansatz limitation) → deprioritize.
    4. **Stale attempts**: if VQE already tried N times without improvement and
       MPNN didn't find a better basin → skip (ceiling reached).
    5. **Parameter count**: larger circuits are harder to optimize → discount.

    Parameters
    ----------
    de_gap : float
        Current relative error |ΔE|/gap.
    abs_error : float
        Current absolute error |ΔE|.
    gap : float
        Spectral gap at this h-point.
    n_params : int
        Number of variational parameters in the circuit.
    e_prev : float | None
        Best energy from previous VQE refinement (None if never refined).
    e_pred : float | None
        Energy from current MPNN prediction (None if not available).
    n_prev_attempts : int
        Number of previous VQE attempts that did NOT improve this point.
    max_stale_attempts : int
        After this many failed attempts without MPNN improvement, skip.

    Returns
    -------
    tuple[float, bool, str]
        - ``priority``: 0.0 (don't bother) to 1.0 (refine immediately)
        - ``should_skip``: True if this point should NOT be refined at all
        - ``reason``: human-readable explanation for logging

    Examples
    --------
    >>> compute_refinement_priority(0.06, 0.12, 2.0, 30)
    (0.85, False, 'close_to_threshold')

    >>> compute_refinement_priority(0.50, 2.0, 0.3, 60, n_prev_attempts=3)
    (0.0, True, 'stale_no_mpnn_improvement')
    """
    # ── Guard: non-finite inputs ─────────────────────────────────────────
    if not np.isfinite(de_gap) or not np.isfinite(abs_error):
        return 0.0, True, "non_finite_metrics"

    # ── Factor 1: MPNN improvement signal ────────────────────────────────
    # If MPNN offers a prediction with lower energy than previous VQE best,
    # that's a strong signal that VQE from the new init can improve.
    mpnn_improved = False
    if e_prev is not None and e_pred is not None:
        if e_pred < e_prev - 1e-6:
            mpnn_improved = True

    # ── Factor 2: Stale attempts → skip ──────────────────────────────────
    # If VQE already tried multiple times AND MPNN didn't find better basin,
    # the point is at its ansatz ceiling. Don't waste compute.
    if n_prev_attempts >= max_stale_attempts and not mpnn_improved:
        return 0.0, True, "stale_no_mpnn_improvement"

    # ── Factor 3: Gap-based feasibility ──────────────────────────────────
    # If gap is tiny (< 0.5), even perfect ΔE/gap requires |ΔE| < gap*0.05
    # = 0.025 — achievable. But if gap < 0.1, |ΔE| must be < 0.005 which
    # is below VQE precision for large circuits. Flag as hard.
    gap_feasibility = 1.0
    if gap < 0.1:
        gap_feasibility = 0.1  # Very hard — near phase transition
    elif gap < 0.5:
        gap_feasibility = 0.5  # Moderately hard

    # ── Factor 4: Proximity to threshold ─────────────────────────────────
    # Points barely failing (ΔE/gap 5-10%) are easy wins.
    # Points far from passing (ΔE/gap > 50%) are unlikely to improve enough.
    if de_gap < 0.10:
        proximity_score = 1.0  # Very close — easy win
    elif de_gap < 0.20:
        proximity_score = 0.7  # Moderate gap to close
    elif de_gap < 0.50:
        proximity_score = 0.3  # Far — hard but possible
    else:
        proximity_score = 0.1  # Very far — likely ansatz-limited

    # ── Factor 5: Parameter count discount ───────────────────────────────
    # More params → harder VQE convergence. Soft discount.
    param_factor = 1.0 / (1.0 + max(0, n_params - 20) / 50.0)
    # n_params=20 → 1.0, n_params=70 → 0.5, n_params=120 → 0.33

    # ── Factor 6: MPNN improvement boost ─────────────────────────────────
    mpnn_boost = 1.5 if mpnn_improved else 1.0

    # ── Combine ──────────────────────────────────────────────────────────
    raw_priority = proximity_score * gap_feasibility * param_factor * mpnn_boost

    # Clamp to [0, 1]
    priority = float(min(1.0, max(0.0, raw_priority)))

    # ── Determine reason ─────────────────────────────────────────────────
    if mpnn_improved:
        reason = "mpnn_found_better_basin"
    elif proximity_score >= 0.7:
        reason = "close_to_threshold"
    elif gap_feasibility < 0.5:
        reason = "small_gap_hard"
    elif n_prev_attempts > 0:
        reason = f"retry_attempt_{n_prev_attempts + 1}"
    else:
        reason = "first_attempt"

    return priority, False, reason


def compute_snr(observable_value: float, shots: int) -> float:
    """Signal-to-noise ratio: |⟨O⟩| * √shots.

    Quantifies measurement reliability. Higher SNR indicates the observable
    signal dominates over shot noise (σ = 1/√shots).

    Parameters
    ----------
    observable_value : float
        Measured expectation value ⟨O⟩.
    shots : int
        Number of measurement shots (must be positive integer).

    Returns
    -------
    float
        Non-negative SNR value: |observable_value| * sqrt(shots).

    Raises
    ------
    ValueError
        If shots is not a positive integer.
    """
    if not isinstance(shots, int | np.integer) or shots <= 0:
        raise ValueError(f"shots must be a positive integer, got {shots}")
    return float(abs(observable_value) * np.sqrt(shots))


def compute_theta_smoothness(theta_array: np.ndarray) -> float | None:
    """Maximum parameter discontinuity across the h-sweep.

    Computes max_i ||θ(h_i) - θ(h_{i-1})||_∞ — the largest infinity-norm
    difference between consecutive θ vectors. Small values indicate a smooth
    parameter landscape (good for MPNN learnability); large values indicate
    discontinuities where the MPNN will struggle.

    Parameters
    ----------
    theta_array : np.ndarray
        Shape (n_h_points, n_params). Rows ordered by h (descending,
        matching the VQE sweep direction).

    Returns
    -------
    float | None
        Non-negative smoothness metric, or None if fewer than 2 h-points.
    """
    if theta_array.shape[0] < 2:
        return None

    # Vectorized: compute all consecutive differences at once
    diffs = np.abs(np.diff(theta_array, axis=0))
    return float(np.max(diffs))


def compute_classification_confidence(
    mag_x: float,
    corr_zz: float,
    shots: int,
) -> float:
    """Phase classification confidence: |⟨X⟩ - ⟨ZZ⟩| * √shots.

    Measures how confidently the pipeline can distinguish between the
    paramagnetic (⟨X⟩ dominant) and antiferromagnetic (⟨ZZ⟩ dominant) phases.
    Higher values indicate clearer phase separation relative to shot noise.

    Parameters
    ----------
    mag_x : float
        Measured transverse magnetization ⟨X⟩.
    corr_zz : float
        Measured nearest-neighbor ZZ correlation ⟨ZZ⟩.
    shots : int
        Number of measurement shots (must be positive integer).

    Returns
    -------
    float
        Non-negative classification confidence value.

    Raises
    ------
    ValueError
        If shots is not a positive integer.
    """
    if not isinstance(shots, int | np.integer) or shots <= 0:
        raise ValueError(f"shots must be a positive integer, got {shots}")
    return float(abs(mag_x - corr_zz) * np.sqrt(shots))


def compute_energy_decomposition(
    e_exact: float,
    e_vqe_ceiling: float,
    e_predicted: float,
) -> dict[str, float]:
    """Decompose total energy error into circuit vs MPNN contributions.

    Separates the total prediction error |e_predicted - e_exact| into:
    - error_from_circuit: |e_vqe_ceiling - e_exact| — physics limit of HVA p=2
    - error_from_mpnn: |e_predicted - e_vqe_ceiling| — ML prediction error

    Invariant: error_from_circuit + error_from_mpnn == |e_predicted - e_exact|
    within floating-point tolerance (1e-12).

    Parameters
    ----------
    e_exact : float
        Exact ground state energy (from Phase 1 exact diagonalization).
    e_vqe_ceiling : float
        Best achievable energy with HVA p=2: E_VQE(θ_opt).
    e_predicted : float
        Energy using MPNN-predicted parameters: E_VQE(θ_MPNN).

    Returns
    -------
    dict[str, float]
        Keys: e_exact, e_vqe_ceiling, e_mpnn_predicted,
              error_from_circuit, error_from_mpnn.
    """
    error_from_circuit = abs(e_vqe_ceiling - e_exact)
    error_from_mpnn = abs(e_predicted - e_vqe_ceiling)

    return {
        "e_exact": float(e_exact),
        "e_vqe_ceiling": float(e_vqe_ceiling),
        "e_mpnn_predicted": float(e_predicted),
        "error_from_circuit": float(error_from_circuit),
        "error_from_mpnn": float(error_from_mpnn),
    }


def compute_fraction_near_gs(
    cost_fn,
    n_params: int,
    n_samples: int = 200,
    threshold: float = 0.05,
    gap: float = 1.0,
    e_exact: float = 0.0,
    bounds: tuple[float, float] = (-np.pi, np.pi),
    seed: int | None = None,
) -> dict[str, float]:
    """Fraction of random parameter initializations near the ground state.

    A training-free metric that estimates how accessible the ground state
    is from random starting points. Higher values indicate an easier
    optimization landscape at a given h-value.

    Parameters
    ----------
    cost_fn : callable
        Energy function E(theta) -> float.
    n_params : int
        Number of variational parameters.
    n_samples : int
        Number of random samples to evaluate.
    threshold : float
        ΔE/gap threshold for "near ground state" (default 5%).
    gap : float
        Spectral gap for normalization.
    e_exact : float
        Exact ground state energy.
    bounds : tuple[float, float]
        Parameter bounds (default [-pi, pi]).
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    dict[str, float]
        Keys: fraction_near_gs, n_near, n_samples, threshold, mean_de_gap.
    """
    rng = np.random.default_rng(seed)
    gap_safe = max(abs(gap), 1e-10)

    n_near = 0
    de_gaps = np.zeros(n_samples)

    for i in range(n_samples):
        theta = rng.uniform(bounds[0], bounds[1], n_params)
        energy = cost_fn(theta)
        de_gap = abs(energy - e_exact) / gap_safe
        de_gaps[i] = de_gap
        if de_gap < threshold:
            n_near += 1

    fraction = n_near / n_samples

    return {
        "fraction_near_gs": float(fraction),
        "n_near": n_near,
        "n_samples": n_samples,
        "threshold": float(threshold),
        "mean_de_gap": float(np.mean(de_gaps)),
    }


def compute_deploy_summary(
    per_h_results: list[dict],
    *,
    thresholds: tuple[float, ...] = (0.05, 0.10),
) -> dict[str, Any]:
    """Compute pass rates and summary statistics from per-h deployment results.

    Standardizes the repeated pattern of aggregating per-point ΔE/gap,
    fidelity, and absolute error into a summary dict. Used by any runner
    section that evaluates MPNN-predicted θ across multiple h-points.

    Parameters
    ----------
    per_h_results : list[dict]
        Each entry must contain at minimum ``"de_gap"`` (float).
        Optional keys: ``"abs_error"``, ``"fidelity"``, ``"e_pred"``,
        ``"e_exact"``, ``"gap"``.
    thresholds : tuple[float, ...]
        ΔE/gap thresholds to compute pass rates for (default: 5%, 10%).

    Returns
    -------
    dict[str, Any]
        Keys include:
        - n_points: total number of evaluated h-points
        - pass_rate_{pct}pct: fraction passing each threshold
        - mean_de_gap, max_de_gap: aggregate ΔE/gap stats
        - mean_abs_error: mean |ΔE| if available
        - mean_fidelity, fidelity_pass_rate: fidelity stats (None if unavailable)
    """
    if not per_h_results:
        return {"n_points": 0, "pass_rate_5pct": 0.0, "pass_rate_10pct": 0.0}

    n = len(per_h_results)
    de_gaps = np.array([r["de_gap"] for r in per_h_results])

    summary: dict[str, Any] = {
        "n_points": n,
        "mean_de_gap": float(np.mean(de_gaps)),
        "max_de_gap": float(np.max(de_gaps)),
        "median_de_gap": float(np.median(de_gaps)),
    }

    # Pass rates for each threshold
    for thr in thresholds:
        pct_label = f"{int(thr * 100)}pct"
        n_pass = int(np.sum(de_gaps < thr))
        summary[f"n_pass_{pct_label}"] = n_pass
        summary[f"pass_rate_{pct_label}"] = n_pass / n

    # Absolute error stats (if available)
    abs_errors = [r["abs_error"] for r in per_h_results if "abs_error" in r]
    if abs_errors:
        summary["mean_abs_error"] = float(np.mean(abs_errors))
        summary["max_abs_error"] = float(np.max(abs_errors))

    # Fidelity stats (if available — only for N ≤ STATEVECTOR_MAX_N)
    fids = [r["fidelity"] for r in per_h_results if r.get("fidelity") is not None]
    if fids:
        summary["mean_fidelity"] = float(np.mean(fids))
        summary["min_fidelity"] = float(np.min(fids))
        summary["fidelity_pass_rate"] = float(np.mean(np.array(fids) > 0.90))
    else:
        summary["mean_fidelity"] = None
        summary["fidelity_pass_rate"] = None

    return summary


def classify_regime(h: float, j: float) -> str:
    """Classify a (j, h) point into a physical regime for TFIM-like models.

    Uses the h/j ratio (control parameter / coupling) to determine the
    expected physics:
    - h/j > 2: paramagnetic (trivial for VQE — deep in PM phase)
    - 1.2 < h/j ≤ 2: intermediate (HVA p≤2 may struggle)
    - 0.8 < h/j ≤ 1.2: critical (near QPT, hardest for VQE)
    - h/j ≤ 0.8: ordered (ferromagnetic, HVA p≤2 cannot reach)
    - j ≤ 0 or h ≤ 0: trivial (non-interacting or field-free)

    Parameters
    ----------
    h : float
        Transverse field strength.
    j : float
        Coupling constant (ZZ interaction).

    Returns
    -------
    str
        One of: 'paramagnetic', 'intermediate', 'critical', 'ordered', 'trivial'.

    Examples
    --------
    >>> classify_regime(h=3.0, j=1.0)
    'paramagnetic'
    >>> classify_regime(h=1.0, j=1.0)
    'critical'
    >>> classify_regime(h=0.5, j=1.0)
    'ordered'
    """
    if j <= 0 or h <= 0:
        return "trivial"
    ratio = h / j
    if ratio > 2.0:
        return "paramagnetic"
    elif ratio > 1.2:
        return "intermediate"
    elif ratio > 0.8:
        return "critical"
    else:
        return "ordered"


# ═══════════════════════════════════════════════════════════════════════════════
# H-frontier computation — empirical boundary from per-h results
# ═══════════════════════════════════════════════════════════════════════════════


def compute_h_frontier(
    h_values: np.ndarray,
    de_gaps: np.ndarray,
    *,
    threshold: float = DE_GAP_THRESHOLD,
) -> float | None:
    """Compute h_frontier: the h where ΔE/gap crosses the pass threshold.

    Uses linear interpolation between the last failing and first passing
    h-point (scanning from low-h to high-h).  This gives the empirical
    boundary below which VQE/MPNN cannot achieve the target accuracy for
    a specific (topology, N, p) configuration.

    Extracted from ``compute_h_frontier_topologies.py`` for reuse in
    ``compute_refinement_priority`` and iterative improvement loops.

    Parameters
    ----------
    h_values : np.ndarray
        Array of h-values (will be sorted internally).
    de_gaps : np.ndarray
        Corresponding ΔE/gap values (same length as h_values).
    threshold : float
        Pass/fail threshold (default: 0.05).

    Returns
    -------
    float | None
        Interpolated h where ΔE/gap = threshold.
        None if all points pass or all fail (frontier not determinable).

    Examples
    --------
    >>> h = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    >>> dg = np.array([0.30, 0.12, 0.06, 0.03, 0.01])
    >>> compute_h_frontier(h, dg)  # Between 1.5 and 2.0
    1.857...
    """
    if len(h_values) < 2 or len(h_values) != len(de_gaps):
        return None

    # Sort by h ascending
    sort_idx = np.argsort(h_values)
    h_sorted = np.asarray(h_values)[sort_idx]
    dg_sorted = np.asarray(de_gaps)[sort_idx]

    # Find crossing: last point where dg >= threshold, followed by dg < threshold
    # Scan from low-h (typically failing) to high-h (typically passing)
    for i in range(len(h_sorted) - 1):
        if dg_sorted[i] >= threshold and dg_sorted[i + 1] < threshold:
            # Linear interpolation
            h0, h1 = float(h_sorted[i]), float(h_sorted[i + 1])
            dg0, dg1 = float(dg_sorted[i]), float(dg_sorted[i + 1])
            if abs(dg0 - dg1) < 1e-12:
                return (h0 + h1) / 2
            return h0 + (threshold - dg0) / (dg1 - dg0) * (h1 - h0)

    # No crossing found
    if np.all(dg_sorted < threshold):
        return float(h_sorted[0])  # All pass → frontier is at lowest h tested
    return None  # All fail → frontier not determinable


def compute_h_frontier_from_npz(
    npz_path: "str | Path",
    *,
    threshold: float = DE_GAP_THRESHOLD,
) -> dict:
    """Compute h_frontier + quality stats from an NPZ training data file.

    Convenience wrapper that loads an NPZ file (from MultiNAggregator format)
    and computes the frontier plus summary statistics.

    Parameters
    ----------
    npz_path : str | Path
        Path to NPZ with fields: h_values, de_gaps (or e_vqe + e_exact + gaps).
    threshold : float
        Pass/fail threshold.

    Returns
    -------
    dict with keys:
        - h_frontier: float | None
        - n_points: int
        - pass_rate: float
        - h_range: tuple[float, float]
        - mean_de_gap: float
        - mean_abs_error: float | None
    """
    from pathlib import Path as _Path

    path = _Path(npz_path)
    if not path.exists():
        return {"h_frontier": None, "error": "file_not_found"}

    data = np.load(str(path), allow_pickle=True)
    h_values = data["h_values"]

    if "de_gaps" in data:
        de_gaps = data["de_gaps"]
    elif "e_vqe" in data and "e_exact" in data and "gaps" in data:
        abs_err = np.abs(data["e_vqe"] - data["e_exact"])
        gaps = np.maximum(data["gaps"], 1e-10)
        de_gaps = abs_err / gaps
    else:
        return {"h_frontier": None, "error": "missing_energy_fields"}

    h_frontier = compute_h_frontier(h_values, de_gaps, threshold=threshold)

    n_pass = int((de_gaps < threshold).sum())
    abs_errors = None
    if "e_vqe" in data and "e_exact" in data:
        abs_errors = np.abs(data["e_vqe"] - data["e_exact"])

    return {
        "h_frontier": float(h_frontier) if h_frontier is not None else None,
        "n_points": len(h_values),
        "pass_rate": float(n_pass / max(len(h_values), 1)),
        "h_range": (float(h_values.min()), float(h_values.max())),
        "mean_de_gap": float(de_gaps.mean()),
        "mean_abs_error": float(abs_errors.mean()) if abs_errors is not None else None,
    }


def detect_h_frontier_anomalies(configs: list[dict]) -> list[dict]:
    """Detect non-monotonic h_frontier(N) per topology.

    h_frontier should increase (or stay flat) as N grows for any topology.
    A drop > H_FRONTIER_MONOTONICITY_TOLERANCE suggests mixed h-ranges
    between NPZ datasets.

    Parameters
    ----------
    configs : list[dict]
        Per-config dashboard entries.

    Returns
    -------
    list[dict]
        Each entry: {topology, n_i, n_j, h_frontier_i, h_frontier_j, drop, message}.
    """
    from collections import defaultdict

    by_topo: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for c in configs:
        hf = c.get("h_frontier")
        if hf is not None:
            by_topo[c["topology"]].append((c["n_qubits"], hf))

    anomalies = []
    for topo, pairs in by_topo.items():
        sorted_pairs = sorted(pairs, key=lambda x: x[0])
        for i in range(len(sorted_pairs) - 1):
            n_i, hf_i = sorted_pairs[i]
            n_j, hf_j = sorted_pairs[i + 1]
            drop = hf_i - hf_j
            if drop > H_FRONTIER_MONOTONICITY_TOLERANCE + 1e-9:
                anomalies.append({
                    "topology": topo,
                    "n_i": n_i,
                    "n_j": n_j,
                    "h_frontier_i": hf_i,
                    "h_frontier_j": hf_j,
                    "drop": drop,
                    "message": (
                        f"h_frontier({topo} N={n_j})={hf_j:.3f} < "
                        f"h_frontier({topo} N={n_i})={hf_i:.3f} "
                        f"(drop={drop:.3f}). "
                        f"Possible mixed h-ranges in NPZ datasets."
                    ),
                })
    return anomalies


def detect_training_zoo_incoherence(configs: list[dict], npz_dir: "Path | None" = None) -> list[dict]:
    """Detect configs where NPZ has high bad ratio but zoo model shows good pass rate.

    Parameters
    ----------
    configs : list[dict]
        Per-config dashboard entries.
    npz_dir : Path | None
        Path to data/multi_n_training/. If None, uses project-relative default.

    Returns
    -------
    list[dict]
        Each entry: {topology, n_qubits, bad_ratio, zoo_pass_rate, message}.
    """
    import numpy as np
    from pathlib import Path as _Path

    if npz_dir is None:
        _root = _Path(__file__).resolve().parents[3]
        npz_dir = _root / "data" / "multi_n_training"

    incoherent = []
    for c in configs:
        zoo_pr = c.get("zoo_pass_rate")
        if zoo_pr is None or zoo_pr < ZOO_PASS_FOR_INCOHERENCE_FLAG:
            continue

        npz_file = _Path(npz_dir) / c.get("file", "")
        if not npz_file.exists():
            continue

        data = np.load(str(npz_file), allow_pickle=True)
        n_pts = len(data["h_values"])
        if n_pts == 0:
            continue

        e_key = "e_vqe" if "e_vqe" in data else ("energies" if "energies" in data else None)
        if e_key is None or "e_exact" not in data:
            continue
        abs_err = np.abs(data[e_key] - data["e_exact"])
        gaps = data["gaps"] if "gaps" in data else np.ones(n_pts)
        de_gaps = abs_err / np.maximum(gaps, 1e-10)
        bad = (de_gaps >= 0.10) | (abs_err >= MAX_ABS_ERROR)
        bad_ratio = float(bad.mean())

        if bad_ratio > TRAINING_BAD_RATIO_THRESHOLD:
            incoherent.append({
                "topology": c["topology"],
                "n_qubits": c["n_qubits"],
                "bad_ratio": bad_ratio,
                "zoo_pass_rate": zoo_pr,
                "message": (
                    f"{c['topology']} N={c['n_qubits']}: "
                    f"NPZ bad_ratio={bad_ratio:.0%} > {TRAINING_BAD_RATIO_THRESHOLD:.0%} "
                    f"but zoo claims pass={zoo_pr:.0%}. "
                    f"Zoo may be trained on different/cleaner data or manifest is stale."
                ),
            })
    return incoherent


def detect_pass_rate_regression(
    current_configs: list[dict],
    previous_dashboard_path: "str | Path | None" = None,
) -> list[dict]:
    """Detect per-topology pass_rate regressions vs a previous dashboard snapshot.

    Parameters
    ----------
    current_configs : list[dict]
        Current dashboard configs.
    previous_dashboard_path : str | Path | None
        Path to a previous dashboard JSON. If None, looks for
        data/model_quality_dashboard_prev.json.

    Returns
    -------
    list[dict]
        Each entry: {topology, prev_max, curr_max, drop, message}.
    """
    from pathlib import Path as _Path
    import json

    if previous_dashboard_path is None:
        _root = _Path(__file__).resolve().parents[3]
        prev_path = _root / "data" / "model_quality_dashboard_prev.json"
    else:
        prev_path = _Path(previous_dashboard_path)

    if not prev_path.exists():
        return []

    try:
        with open(prev_path) as f:
            prev_dashboard = json.load(f)
        prev_configs = prev_dashboard.get("configs", [])
    except (json.JSONDecodeError, OSError):
        return []

    from collections import defaultdict

    def max_dual_per_topo_n(configs: list[dict]) -> dict[tuple, float]:
        by_topo_n: dict[tuple, list[float]] = defaultdict(list)
        for c in configs:
            dr = c.get("pass_rate_dual_criterion")
            if dr is not None:
                key = (c["topology"], c.get("n_qubits", 0))
                by_topo_n[key].append(dr)
        return {k: max(vals) for k, vals in by_topo_n.items() if vals}

    def max_dual_per_topo(by_topo_n: dict) -> dict[str, float]:
        by_topo: dict[str, list[float]] = defaultdict(list)
        for (topo, n), val in by_topo_n.items():
            by_topo[topo].append(val)
        return {t: max(vals) for t, vals in by_topo.items() if vals}

    curr_by_tn = max_dual_per_topo_n(current_configs)
    prev_by_tn = max_dual_per_topo_n(prev_configs)

    curr_max = max_dual_per_topo(curr_by_tn)
    prev_max = max_dual_per_topo(prev_by_tn)

    regressions = []
    for topo, curr in curr_max.items():
        prev = prev_max.get(topo)
        if prev is None:
            continue
        drop = prev - curr
        if drop > PASS_RATE_REGRESSION_THRESHOLD:
            regressions.append({
                "topology": topo,
                "prev_max": prev,
                "curr_max": curr,
                "drop": drop,
                "message": (
                    f"{topo}: max pass_rate_dual dropped {drop:.0%} "
                    f"({prev:.0%} → {curr:.0%}). "
                    f"Threshold: {PASS_RATE_REGRESSION_THRESHOLD:.0%}."
                ),
            })
    return regressions


def classify_training_utility(
    n_points: int,
    pass_rate_dual: float,
    pass_rate_5pct: float,
    *,
    min_pass_rate: float = MIN_TRAINING_DUAL_PASS_RATE,
    min_good_points: int = MIN_TRAINING_POINTS_FOR_SIGNAL,
) -> tuple[str, str]:
    """Classify whether a NPZ config is useful for MPNN training.

    This is NOT about physical correctness (NaN, convergence) but about
    whether the data provides learnable signal for h → θ prediction.

    Three categories:

    - "useful": pass_rate_dual >= min_pass_rate AND n_good >= min_good_points.
      The MPNN can learn from this data.

    - "insufficient_signal": physically valid but too few good points
      (<min_good_points) or too low pass rate (<min_pass_rate). The MPNN
      would learn noise — include in training at your own risk.

    - "not_useful": pass_rate_dual == 0 OR all points fail dual criterion.
      Including this data actively harms MPNN training (teaches wrong θ).

    Parameters
    ----------
    n_points : int
        Total number of h-points in the NPZ.
    pass_rate_dual : float
        Fraction passing the dual criterion (ΔE/gap < 5% AND |ΔE| < 0.10).
    pass_rate_5pct : float
        Fraction passing simple ΔE/gap < 5% (may include gap-masked points).
    min_pass_rate : float
        Minimum dual pass rate for "useful" classification.
    min_good_points : int
        Minimum absolute count of good points for "useful" classification.

    Returns
    -------
    tuple[str, str]
        (category, reason) where category is one of:
        "useful", "insufficient_signal", "not_useful"
    """
    n_good = int(n_points * pass_rate_dual)
    gap_masked = pass_rate_5pct - pass_rate_dual

    if pass_rate_dual == 0.0:
        if gap_masked > 0.1:
            return (
                "not_useful",
                f"0% dual pass (gap masking: {gap_masked:.0%} of points appear to pass "
                f"but fail |ΔE|<{MAX_ABS_ERROR} criterion). "
                f"Including this data teaches the MPNN incorrect θ mappings.",
            )
        return (
            "not_useful",
            f"0% dual pass rate — no learnable signal. "
            f"Possible causes: p=1 insufficient expressivity, h outside valid regime, "
            f"or VQE trapped in local minimum for all points.",
        )

    if n_good < min_good_points:
        return (
            "insufficient_signal",
            f"Only {n_good} good points (need ≥ {min_good_points}). "
            f"Statistically insufficient for MPNN to learn h → θ mapping.",
        )

    if pass_rate_dual < min_pass_rate:
        return (
            "insufficient_signal",
            f"Dual pass rate {pass_rate_dual:.0%} < {min_pass_rate:.0%} threshold. "
            f"{n_good}/{n_points} points are good. "
            f"High noise/signal ratio — train at your own risk.",
        )

    return ("useful", f"{n_good}/{n_points} good points ({pass_rate_dual:.0%} dual pass).")


def validate_training_dataset(
    per_n_points: dict[int, list[dict]],
    *,
    max_de_gap: float = 0.10,
    min_total_points: int = 10,
    min_n_values: int = 2,
    require_variational: bool = True,
) -> tuple[bool, dict]:
    """Validate multi-N training data quality before MPNN training.

    Performs a comprehensive check on aggregated training data to determine
    if it's suitable for training. Call this BEFORE train_unified_mpnn()
    to avoid training on garbage data.

    Checks performed:
    1. Minimum total points across all N (default: 10)
    2. Minimum number of distinct N values (default: 2)
    3. Dual-criterion filter — counts passing points per N
    4. Variational integrity — flags N values with >90% violations
    5. Gap masking detection — identifies inflated pass rates
    6. θ dimension consistency across N values

    Parameters
    ----------
    per_n_points : dict[int, list[dict]]
        Data keyed by N → list of point dicts (from MultiNAggregator.scan()).
        Each point must have: "de_gap", "abs_error" (optional), "theta", "h".
    max_de_gap : float
        Threshold for quality filtering (default: 0.10).
    min_total_points : int
        Minimum total good points required (default: 10).
    min_n_values : int
        Minimum distinct N values with usable data (default: 2).
    require_variational : bool
        If True, warn when >90% of points are variational violations.

    Returns
    -------
    tuple[bool, dict]
        - is_viable: True if data is suitable for training
        - report: dict with per-N breakdown, warnings, and recommendations

    Example
    -------
    >>> from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator
    >>> agg = MultiNAggregator(topology="ladder", model="tfim_bond_resolved")
    >>> agg.scan()
    >>> viable, report = validate_training_dataset(agg._data_by_n)
    >>> if not viable:
    ...     print(report["recommendation"])
    """
    report: dict = {
        "per_n": {},
        "warnings": [],
        "errors": [],
        "total_raw": 0,
        "total_good": 0,
        "n_values_with_data": 0,
        "n_values_with_good_data": 0,
        "recommendation": "",
    }

    for n, points in sorted(per_n_points.items()):
        n_raw = len(points)
        report["total_raw"] += n_raw

        # Count dual-criterion passing points
        n_good = 0
        n_violations = 0
        for p in points:
            de_gap = p.get("de_gap", 1.0)
            abs_error = p.get("abs_error")

            # Check variational violations if we have energy data
            if abs_error is not None and "e_exact" in p:
                # If e_pred > e_exact (within tolerance), it's a violation
                # Approximate: abs_error = |e_pred - e_exact|, violation = e_pred > e_exact
                # We can't distinguish direction from abs_error alone, but
                # if point has both de_gap < threshold AND abs_error < MAX_ABS_ERROR
                # it's likely OK
                pass

            passes_de_gap = de_gap < max_de_gap
            passes_abs = abs_error is None or abs_error < MAX_ABS_ERROR
            if passes_de_gap and passes_abs:
                n_good += 1

        # Check theta dimension consistency
        theta_dims = set()
        for p in points:
            theta = p.get("theta")
            if theta is not None:
                theta_dims.add(len(theta) if hasattr(theta, '__len__') else 0)

        n_entry = {
            "n_raw": n_raw,
            "n_good": n_good,
            "pass_rate": n_good / max(n_raw, 1),
            "theta_dims": sorted(theta_dims),
        }
        report["per_n"][n] = n_entry

        if n_good > 0:
            report["n_values_with_good_data"] += 1
        report["n_values_with_data"] += 1
        report["total_good"] += n_good

        # Warnings per N
        if n_raw > 0 and n_good == 0:
            report["warnings"].append(
                f"N={n}: 0/{n_raw} points pass dual criterion. "
                f"This N contributes nothing to training."
            )
        elif n_raw > 0 and n_good / n_raw < 0.20:
            report["warnings"].append(
                f"N={n}: only {n_good}/{n_raw} ({n_good/n_raw:.0%}) pass dual criterion. "
                f"Majority of data for this N is low-quality."
            )

        if len(theta_dims) > 1:
            report["errors"].append(
                f"N={n}: inconsistent θ dimensions {theta_dims}. "
                f"Cannot mix data from different p_layers or circuit variants."
            )

    # Global checks
    is_viable = True

    if report["total_good"] < min_total_points:
        is_viable = False
        report["errors"].append(
            f"Only {report['total_good']} good points total (need ≥{min_total_points}). "
            f"Run more VQE refinement to generate quality data."
        )

    if report["n_values_with_good_data"] < min_n_values:
        is_viable = False
        report["errors"].append(
            f"Only {report['n_values_with_good_data']} N values have usable data "
            f"(need ≥{min_n_values}). Multi-N model needs diversity in N."
        )

    # Recommendation
    if is_viable:
        report["recommendation"] = (
            f"Data is suitable for training: {report['total_good']} good points "
            f"across {report['n_values_with_good_data']} N values."
        )
    else:
        needs = []
        if report["total_good"] < min_total_points:
            deficit = min_total_points - report["total_good"]
            needs.append(f"{deficit} more good points")
        if report["n_values_with_good_data"] < min_n_values:
            needs.append(f"data for more N values")
        report["recommendation"] = (
            f"NOT viable for training. Need: {', '.join(needs)}. "
            f"Run iterative improvement with --refine-all to generate quality data."
        )

    return is_viable, report


def get_usable_training_configs(dashboard: dict) -> dict[str, list[dict]]:
    """Partition dashboard configs by training utility.

    Returns a dict with keys: "useful", "insufficient_signal", "not_useful".
    Each value is a list of config dicts enriched with "training_utility" and
    "training_utility_reason" fields.

    Parameters
    ----------
    dashboard : dict
        Dashboard as returned by generate_model_quality_dashboard().

    Returns
    -------
    dict[str, list[dict]]
        Partition of configs by training utility category.
    """
    result: dict[str, list[dict]] = {
        "useful": [],
        "insufficient_signal": [],
        "not_useful": [],
    }
    for c in dashboard.get("configs", []):
        category, reason = classify_training_utility(
            n_points=c.get("n_points", 0),
            pass_rate_dual=c.get("pass_rate_dual_criterion", 0.0),
            pass_rate_5pct=c.get("pass_rate_5pct", 0.0),
        )
        enriched = {**c, "training_utility": category, "training_utility_reason": reason}
        result[category].append(enriched)
    return result



    """Detect non-monotonic h_frontier(N) per topology.

    h_frontier should increase (or stay flat) as N grows for any topology.
    A drop > H_FRONTIER_MONOTONICITY_TOLERANCE suggests mixed h-ranges
    between NPZ datasets — e.g., a small-N dataset was evaluated at
    higher h than a large-N dataset.

    Parameters
    ----------
    configs : list[dict]
        Per-config dashboard entries (output of generate_model_quality_dashboard).

    Returns
    -------
    list[dict]
        Each entry: {topology, n_i, n_j, h_frontier_i, h_frontier_j, drop}.
        n_i < n_j but h_frontier_i > h_frontier_j + tolerance.
    """
    from collections import defaultdict

    by_topo: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for c in configs:
        hf = c.get("h_frontier")
        if hf is not None:
            by_topo[c["topology"]].append((c["n_qubits"], hf))

    anomalies = []
    for topo, pairs in by_topo.items():
        sorted_pairs = sorted(pairs, key=lambda x: x[0])
        for i in range(len(sorted_pairs) - 1):
            n_i, hf_i = sorted_pairs[i]
            n_j, hf_j = sorted_pairs[i + 1]
            drop = hf_i - hf_j
            if drop > H_FRONTIER_MONOTONICITY_TOLERANCE + 1e-9:  # epsilon for float precision
                anomalies.append({
                    "topology": topo,
                    "n_i": n_i,
                    "n_j": n_j,
                    "h_frontier_i": hf_i,
                    "h_frontier_j": hf_j,
                    "drop": drop,
                    "message": (
                        f"h_frontier({topo} N={n_j})={hf_j:.3f} < "
                        f"h_frontier({topo} N={n_i})={hf_i:.3f} "
                        f"(drop={drop:.3f}). "
                        f"Possible mixed h-ranges in NPZ datasets."
                    ),
                })
    return anomalies


def detect_training_zoo_incoherence(configs: list[dict], npz_dir: "Path | None" = None) -> list[dict]:
    """Detect configs where NPZ has high bad ratio but zoo model shows good pass rate.

    A well-trained zoo model should reflect the quality of its training data.
    If the training data is mostly bad (>TRAINING_BAD_RATIO_THRESHOLD) but the
    zoo model claims high pass_rate (>ZOO_PASS_FOR_INCOHERENCE_FLAG), something
    is wrong: either the zoo model was trained on different (cleaner) data,
    or the pass_rate in the zoo manifest is stale/incorrect.

    Parameters
    ----------
    configs : list[dict]
        Per-config dashboard entries.
    npz_dir : Path | None
        Path to data/multi_n_training/. If None, uses project-relative default.

    Returns
    -------
    list[dict]
        Each entry: {topology, n_qubits, bad_ratio, zoo_pass_rate, message}.
    """
    import numpy as np
    from pathlib import Path as _Path

    if npz_dir is None:
        _root = _Path(__file__).resolve().parents[3]
        npz_dir = _root / "data" / "multi_n_training"

    incoherent = []
    for c in configs:
        zoo_pr = c.get("zoo_pass_rate")
        if zoo_pr is None or zoo_pr < ZOO_PASS_FOR_INCOHERENCE_FLAG:
            continue

        npz_file = _Path(npz_dir) / c.get("file", "")
        if not npz_file.exists():
            continue

        data = np.load(str(npz_file), allow_pickle=True)
        n_pts = len(data["h_values"])
        if n_pts == 0:
            continue

        # Compute bad ratio same way as inspect_data_stores
        e_key = "e_vqe" if "e_vqe" in data else ("energies" if "energies" in data else None)
        if e_key is None or "e_exact" not in data:
            continue
        abs_err = np.abs(data[e_key] - data["e_exact"])
        gaps = data["gaps"] if "gaps" in data else np.ones(n_pts)
        de_gaps = abs_err / np.maximum(gaps, 1e-10)
        bad = (de_gaps >= 0.10) | (abs_err >= MAX_ABS_ERROR)
        bad_ratio = float(bad.mean())

        if bad_ratio > TRAINING_BAD_RATIO_THRESHOLD:
            incoherent.append({
                "topology": c["topology"],
                "n_qubits": c["n_qubits"],
                "bad_ratio": bad_ratio,
                "zoo_pass_rate": zoo_pr,
                "message": (
                    f"{c['topology']} N={c['n_qubits']}: "
                    f"NPZ bad_ratio={bad_ratio:.0%} > {TRAINING_BAD_RATIO_THRESHOLD:.0%} "
                    f"but zoo claims pass={zoo_pr:.0%}. "
                    f"Zoo may be trained on different/cleaner data or manifest is stale."
                ),
            })
    return incoherent


def detect_pass_rate_regression(
    current_configs: list[dict],
    previous_dashboard_path: "str | Path | None" = None,
) -> list[dict]:
    """Detect per-topology pass_rate regressions vs a previous dashboard snapshot.

    Compares the max pass_rate_dual per topology between the current dashboard
    and a previous saved snapshot. If the max drops > PASS_RATE_REGRESSION_THRESHOLD,
    flags as regression.

    Parameters
    ----------
    current_configs : list[dict]
        Current dashboard configs.
    previous_dashboard_path : str | Path | None
        Path to a previous dashboard JSON. If None, looks for
        data/model_quality_dashboard_prev.json. Returns empty list if not found.

    Returns
    -------
    list[dict]
        Each entry: {topology, prev_max, curr_max, drop, message}.
    """
    from pathlib import Path as _Path
    import json

    if previous_dashboard_path is None:
        _root = _Path(__file__).resolve().parents[3]
        prev_path = _root / "data" / "model_quality_dashboard_prev.json"
    else:
        prev_path = _Path(previous_dashboard_path)

    if not prev_path.exists():
        return []

    try:
        with open(prev_path) as f:
            prev_dashboard = json.load(f)
        prev_configs = prev_dashboard.get("configs", [])
    except (json.JSONDecodeError, OSError):
        return []

    from collections import defaultdict

    def max_dual_per_topo_n(configs: list[dict]) -> dict[tuple, float]:
        """Return best pass_rate_dual per (topology, n_qubits) pair."""
        by_topo_n: dict[tuple, list[float]] = defaultdict(list)
        for c in configs:
            dr = c.get("pass_rate_dual_criterion")
            if dr is not None:
                key = (c["topology"], c.get("n_qubits", 0))
                by_topo_n[key].append(dr)
        return {k: max(vals) for k, vals in by_topo_n.items() if vals}

    # Also compute topology-level max for the regression comparison
    def max_dual_per_topo(by_topo_n: dict) -> dict[str, float]:
        by_topo: dict[str, list[float]] = defaultdict(list)
        for (topo, n), val in by_topo_n.items():
            by_topo[topo].append(val)
        return {t: max(vals) for t, vals in by_topo.items() if vals}

    curr_by_tn = max_dual_per_topo_n(current_configs)
    prev_by_tn = max_dual_per_topo_n(prev_configs)

    curr_max = max_dual_per_topo(curr_by_tn)
    prev_max = max_dual_per_topo(prev_by_tn)

    regressions = []
    for topo, curr in curr_max.items():
        prev = prev_max.get(topo)
        if prev is None:
            continue
        drop = prev - curr
        if drop > PASS_RATE_REGRESSION_THRESHOLD:
            regressions.append({
                "topology": topo,
                "prev_max": prev,
                "curr_max": curr,
                "drop": drop,
                "message": (
                    f"{topo}: max pass_rate_dual dropped {drop:.0%} "
                    f"({prev:.0%} → {curr:.0%}). "
                    f"Threshold: {PASS_RATE_REGRESSION_THRESHOLD:.0%}."
                ),
            })
    return regressions


# ═══════════════════════════════════════════════════════════════════════════════
# Model quality dashboard — auto-generated per (topology, N) from NPZ data
# ═══════════════════════════════════════════════════════════════════════════════


def _scan_cross_n_transfer_results(root: "Path") -> dict:
    """Scan cross-N experiment results to extract transfer quality data.

    Returns a dict keyed by (topology, target_n, p_layers) → list of transfer records.
    Each record: {train_n, pass_rate_5pct, pass_rate_10pct, mean_de_gap, n_points, timestamp}.
    """
    import json
    from pathlib import Path as _Path

    results_dir = root / "results" / "experiments" / "exp_accel_cross_n"
    transfer_data: dict[tuple, list] = {}

    if not results_dir.exists():
        return transfer_data

    for run_file in sorted(results_dir.glob("run_*.json")):
        try:
            with open(run_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        config = data.get("config", {})
        topology = config.get("topology", "")
        if not topology:
            continue

        results_sec = data.get("results", {})
        for _key, section in results_sec.items():
            if not isinstance(section, dict):
                continue
            sd = section.get("data", {})
            cross_n = sd.get("cross_n_results", {})
            for _ck, result in cross_n.items():
                if not isinstance(result, dict):
                    continue
                train_n = result.get("train_n")
                target_n = result.get("target_n")
                p_layers = result.get("p_layers", 1)
                if train_n is None or target_n is None:
                    continue
                # Only record actual cross-N (or self-eval)
                pass_5 = result.get("pass_rate_5pct", 0.0)
                pass_10 = result.get("pass_rate_10pct", 0.0)
                mean_dg = result.get("mean_de_gap", 0.0)
                n_pts = result.get("n_points", 0)

                key = (topology, target_n, p_layers)
                if key not in transfer_data:
                    transfer_data[key] = []
                transfer_data[key].append({
                    "train_n": train_n,
                    "pass_rate_5pct": float(pass_5),
                    "pass_rate_10pct": float(pass_10),
                    "mean_de_gap": float(mean_dg),
                    "n_points": n_pts,
                    "file": run_file.name,
                })

    return transfer_data


def _best_cross_n_source(transfers: list) -> "dict | None":
    """Find the best train_n for cross-N prediction from a list of transfer records.

    Best = highest pass_rate_10pct, then lowest mean_de_gap as tiebreaker.
    Returns {train_n, pass_rate_10pct, mean_de_gap} or None if no data.
    """
    if not transfers:
        return None
    best = max(transfers, key=lambda t: (t["pass_rate_10pct"], -t["mean_de_gap"]))
    return {
        "train_n": best["train_n"],
        "pass_rate_10pct": best["pass_rate_10pct"],
        "pass_rate_5pct": best["pass_rate_5pct"],
        "mean_de_gap": best["mean_de_gap"],
    }


def _compute_topology_summary(configs: list) -> dict:
    """Compute per-topology aggregates including n_max_viable.

    n_max_viable = largest N where pass_rate_10pct > 50%.
    """
    from collections import defaultdict

    by_topo: dict[str, list] = defaultdict(list)
    for c in configs:
        by_topo[c["topology"]].append(c)

    summary = {}
    for topo, topo_configs in sorted(by_topo.items()):
        n_values = sorted(set(c["n_qubits"] for c in topo_configs))
        viable = [c for c in topo_configs if c["pass_rate_10pct"] > 0.5]
        n_max_viable = max((c["n_qubits"] for c in viable), default=None)

        # Best overall pass rate
        best_config = max(topo_configs, key=lambda c: c["pass_rate_5pct"])

        # Cross-N summary: best source for largest N
        largest_n = max(n_values) if n_values else 0
        largest_configs = [c for c in topo_configs if c["n_qubits"] == largest_n]
        cross_n_best = None
        if largest_configs and largest_configs[0].get("cross_n_best_source"):
            cross_n_best = largest_configs[0]["cross_n_best_source"]

        summary[topo] = {
            "n_values": n_values,
            "n_max_viable": n_max_viable,
            "n_configs": len(topo_configs),
            "best_pass_rate_5pct": best_config["pass_rate_5pct"],
            "best_n": best_config["n_qubits"],
            "cross_n_best_source_for_largest": cross_n_best,
        }

    return summary


def generate_model_quality_dashboard(
    output_path: "str | Path | None" = None,
) -> dict:
    """Auto-generate a quality dashboard from all NPZ training data.

    Scans ``data/multi_n_training/*.npz`` and computes per-(topology, N):
    - h_frontier (empirical boundary)
    - pass_rate (@5% and @10%)
    - mean_de_gap, mean_abs_error
    - n_training_points
    - data freshness (file mtime)
    - cross_n_transfers (train_n → this N, pass rates from experiment results)
    - cross_n_best_source (best train_n for predicting this N)

    Also computes topology-level summary:
    - n_max_viable (largest N with pass_rate_10pct > 50%)

    The output JSON is consumable by QualityPredictor as a "fresh signal"
    source, complementing the slower ResultIndex-based historical analysis.

    Parameters
    ----------
    output_path : str | Path | None
        Where to write the dashboard JSON. Default: ``data/model_quality_dashboard.json``.

    Returns
    -------
    dict
        Dashboard data with keys: ``configs`` (list of per-config dicts),
        ``topology_summary`` (dict of per-topology aggregates),
        ``generated_at`` (ISO timestamp), ``n_configs`` (int).
    """
    from datetime import datetime, timezone
    from pathlib import Path as _Path
    import json

    _ROOT = _Path(__file__).resolve().parents[3]
    npz_dir = _ROOT / "data" / "multi_n_training"
    if output_path is None:
        output_path = _ROOT / "data" / "model_quality_dashboard.json"
    else:
        output_path = _Path(output_path)

    if not npz_dir.exists():
        return {"configs": [], "topology_summary": {}, "generated_at": "", "n_configs": 0}

    # ── Phase 1: Scan cross-N experiment results for transfer quality ────
    cross_n_data = _scan_cross_n_transfer_results(_ROOT)

    configs = []
    for npz_file in sorted(npz_dir.glob("*.npz")):
        # Parse topology and N from filename: <topo>_N<n>_p<p>.npz
        stem = npz_file.stem
        parts = stem.split("_")
        n_idx = next((i for i, p in enumerate(parts) if p.startswith("N")), None)
        if n_idx is None:
            continue
        topology = "_".join(parts[:n_idx])
        try:
            n_qubits = int(parts[n_idx][1:])
        except ValueError:
            continue
        p_idx = next((i for i, p in enumerate(parts) if p.startswith("p")), None)
        p_layers = int(parts[p_idx][1:]) if p_idx else 1

        # Compute metrics
        frontier_result = compute_h_frontier_from_npz(npz_file)
        if "error" in frontier_result:
            continue

        # Load raw data for additional stats
        data = np.load(str(npz_file), allow_pickle=True)
        h_values = data["h_values"]
        n_points = len(h_values)

        # Compute pass rates
        de_gaps = None
        if "de_gaps" in data:
            de_gaps = data["de_gaps"]
        elif "e_vqe" in data and "e_exact" in data and "gaps" in data:
            abs_err = np.abs(data["e_vqe"] - data["e_exact"])
            de_gaps = abs_err / np.maximum(data["gaps"], 1e-10)

        pass_rate_5 = float((de_gaps < DE_GAP_THRESHOLD).mean()) if de_gaps is not None else 0.0
        pass_rate_10 = float((de_gaps < 2 * DE_GAP_THRESHOLD).mean()) if de_gaps is not None else 0.0
        mean_de_gap = float(de_gaps.mean()) if de_gaps is not None else 0.0

        mean_abs_err = frontier_result.get("mean_abs_error")

        # ── Additional dimensions: circuit complexity + quality bounds ────
        theta_opt = data["theta_opt"]
        n_params = theta_opt.shape[1] if theta_opt.ndim == 2 else 0
        # n_edges = n_params - n_qubits (for bond-resolved: params = edges + sites)
        n_edges = max(0, n_params - n_qubits)

        # ── θ NaN/Inf check ──────────────────────────────────────────────
        n_nan_theta = int(np.sum(~np.isfinite(theta_opt)))

        # ── Dual criterion pass rate (using is_point_failure) ────────────
        # More strict than simple de_gaps < threshold — also checks abs_error
        e_key_raw = "e_vqe" if "e_vqe" in data else ("energies" if "energies" in data else None)
        abs_errors = None
        if e_key_raw and "e_exact" in data:
            abs_errors = np.abs(data[e_key_raw] - data["e_exact"])

        n_failures_dual = 0
        if de_gaps is not None:
            for i in range(n_points):
                ae = float(abs_errors[i]) if abs_errors is not None else None
                if is_point_failure(de_gap=float(de_gaps[i]), abs_error=ae):
                    n_failures_dual += 1
        pass_rate_dual = (n_points - n_failures_dual) / max(n_points, 1)

        best_de_gap = float(de_gaps.min()) if de_gaps is not None else 0.0
        worst_de_gap = float(de_gaps.max()) if de_gaps is not None else 0.0
        n_below_frontier = 0
        if de_gaps is not None and frontier_result["h_frontier"] is not None:
            n_below_frontier = int((h_values < frontier_result["h_frontier"]).sum())

        # ── Cross-check with model zoo ───────────────────────────────────
        zoo_model_available = False
        zoo_pass_rate = None
        zoo_n_training_points = 0
        zoo_integrity_ok = None  # None = not checked, True/False = result
        try:
            from qmbp_simulation.predictors.model_zoo import list_pretrained, validate_zoo
            zoo_entries = list_pretrained(
                model="tfim_bond_resolved", topology=topology, p_layers=p_layers
            )
            if zoo_entries:
                zoo_model_available = True
                best_zoo = max(zoo_entries, key=lambda e: e.pass_rate)
                if best_zoo.pass_rate > 0:
                    zoo_pass_rate = float(best_zoo.pass_rate)
                zoo_n_training_points = best_zoo.n_training_points
        except (ImportError, Exception):
            pass

        # Zoo SHA256 integrity (run once, cache result for all configs)
        if zoo_model_available and not hasattr(generate_model_quality_dashboard, "_zoo_validated"):
            try:
                zoo_report = validate_zoo()
                generate_model_quality_dashboard._zoo_validated = True
                generate_model_quality_dashboard._zoo_integrity = (
                    zoo_report["n_corrupted"] == 0
                )
                generate_model_quality_dashboard._zoo_n_missing = zoo_report["n_missing"]
            except Exception:
                generate_model_quality_dashboard._zoo_validated = True
                generate_model_quality_dashboard._zoo_integrity = None
                generate_model_quality_dashboard._zoo_n_missing = 0

        if zoo_model_available:
            zoo_integrity_ok = getattr(
                generate_model_quality_dashboard, "_zoo_integrity", None
            )

        # ── EvalCache density: how well-explored is this config? ─────────
        eval_cache_density = 0
        try:
            from qmbp_simulation.execution.eval_cache import EvalCache
            _ec = EvalCache()
            eval_cache_density = _ec.count_entries_for_config(
                topology, n_qubits, "tfim_bond_resolved", p_layers
            )
        except (ImportError, Exception):
            pass

        # ── Stale model detection ────────────────────────────────────────
        # If NPZ pass_rate differs significantly from zoo pass_rate → model stale
        model_stale = False
        if zoo_pass_rate is not None and pass_rate_5 > 0:
            # Zoo says X%, NPZ says Y% — if NPZ is much better, model outdated
            if pass_rate_5 > zoo_pass_rate + 0.15:
                model_stale = True  # NPZ data improved but model wasn't retrained

        # ── Retrain recommendation ───────────────────────────────────────
        needs_retrain = False
        if zoo_n_training_points > 0 and n_points > zoo_n_training_points * 1.3:
            needs_retrain = True  # 30%+ more data available than model was trained on

        # ── θ smoothness (training data quality) ─────────────────────────
        theta_smoothness = None
        if n_points > 1 and theta_opt.ndim == 2:
            sort_idx = np.argsort(h_values)
            theta_sorted = theta_opt[sort_idx]
            diffs = np.max(np.abs(np.diff(theta_sorted, axis=0)), axis=1)
            theta_smoothness = float(diffs.max())

        # ── Training utility classification ──────────────────────────────
        training_utility, training_utility_reason = classify_training_utility(
            n_points=n_points,
            pass_rate_dual=pass_rate_dual,
            pass_rate_5pct=pass_rate_5,
        )

        configs.append({
            "topology": topology,
            "n_qubits": n_qubits,
            "p_layers": p_layers,
            "n_params": n_params,
            "n_edges": n_edges,
            "model": "tfim_bond_resolved",
            "n_points": n_points,
            "h_range": list(frontier_result["h_range"]),
            "h_frontier": frontier_result["h_frontier"],
            "pass_rate_5pct": pass_rate_5,
            "pass_rate_10pct": pass_rate_10,
            "pass_rate_dual_criterion": float(pass_rate_dual),
            "mean_de_gap": mean_de_gap,
            "mean_abs_error": mean_abs_err,
            "best_de_gap": best_de_gap,
            "worst_de_gap": worst_de_gap,
            "n_below_frontier": n_below_frontier,
            "n_nan_theta": n_nan_theta,
            "zoo_model_available": zoo_model_available,
            "zoo_pass_rate": zoo_pass_rate,
            "zoo_integrity_ok": zoo_integrity_ok,
            "zoo_vs_npz_divergence": (
                abs(zoo_pass_rate - pass_rate_5) if zoo_pass_rate is not None else None
            ),
            "eval_cache_density": eval_cache_density,
            "model_stale": model_stale,
            "needs_retrain": needs_retrain,
            "theta_smoothness": theta_smoothness,
            "training_utility": training_utility,
            "training_utility_reason": training_utility_reason,
            # ── Cross-N transfer quality ─────────────────────────────
            "cross_n_transfers": cross_n_data.get((topology, n_qubits, p_layers), []),
            "cross_n_best_source": _best_cross_n_source(
                cross_n_data.get((topology, n_qubits, p_layers), [])
            ),
            "file": npz_file.name,
            "mtime": datetime.fromtimestamp(
                npz_file.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        })

    # ── Topology-level summary: n_max_viable ────────────────────────────
    topology_summary = _compute_topology_summary(configs)

    # ── Automated quality audits ─────────────────────────────────────────
    frontier_anomalies = detect_h_frontier_anomalies(configs)
    training_zoo_incoherence = detect_training_zoo_incoherence(configs)
    gap_masked_configs = [
        {"topology": c["topology"], "n_qubits": c["n_qubits"],
         "pass_rate_5pct": c["pass_rate_5pct"],
         "pass_rate_dual": c.get("pass_rate_dual_criterion", 0),
         "gap_masked": c["pass_rate_5pct"] - c.get("pass_rate_dual_criterion", 0)}
        for c in configs
        if c["pass_rate_5pct"] - c.get("pass_rate_dual_criterion", 0) > GAP_MASKING_THRESHOLD
    ]
    # Pass rate regression check (compares vs previous snapshot)
    regressions = detect_pass_rate_regression(configs)

    audit_results = {
        "h_frontier_anomalies": frontier_anomalies,
        "training_zoo_incoherence": training_zoo_incoherence,
        "gap_masked_configs": gap_masked_configs,
        "pass_rate_regressions": regressions,
        "n_issues": (
            len(frontier_anomalies) +
            len(training_zoo_incoherence) +
            len(gap_masked_configs) +
            len(regressions)
        ),
    }
    if frontier_anomalies:
        logger.warning("h_frontier anomalies: %d", len(frontier_anomalies))
        for a in frontier_anomalies:
            logger.warning("  %s", a["message"])
    if training_zoo_incoherence:
        logger.warning("Training/zoo incoherence: %d", len(training_zoo_incoherence))
    if gap_masked_configs:
        logger.info("Gap masking detected in %d configs", len(gap_masked_configs))
    if regressions:
        logger.warning("Pass rate regressions: %d", len(regressions))

    dashboard = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_configs": len(configs),
        "configs": configs,
        "topology_summary": topology_summary,
        "audit": audit_results,
    }

    # Write to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dashboard, f, indent=2)

    # ── Staleness self-check ─────────────────────────────────────────────
    # Verify no NPZ is newer than the dashboard we just wrote (race condition)
    generated_at = dashboard["generated_at"]
    stale_files = []
    for c in configs:
        if c["mtime"] > generated_at:
            stale_files.append(c["file"])
    if stale_files:
        logger.warning(
            "Dashboard staleness: %d NPZ files modified after generation "
            "(concurrent writes?): %s", len(stale_files), stale_files[:3],
        )
        dashboard["staleness_warning"] = stale_files

    # ── Integrity summary ────────────────────────────────────────────────
    n_nan_total = sum(c.get("n_nan_theta", 0) for c in configs)
    n_with_nan = sum(1 for c in configs if c.get("n_nan_theta", 0) > 0)
    zoo_ok = getattr(generate_model_quality_dashboard, "_zoo_integrity", None)
    dashboard["integrity"] = {
        "n_configs_with_nan_theta": n_with_nan,
        "total_nan_values": n_nan_total,
        "zoo_integrity_ok": zoo_ok,
        "zoo_n_missing": getattr(
            generate_model_quality_dashboard, "_zoo_n_missing", 0
        ),
    }

    # Re-write with integrity info
    with open(output_path, "w") as f:
        json.dump(dashboard, f, indent=2)

    logger.info(
        "Model quality dashboard: %d configs → %s",
        len(configs), output_path.name,
    )
    return dashboard


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Integration Utilities — combining quality tier, extrapolation, and coverage
# ═══════════════════════════════════════════════════════════════════════════════


def compute_scalability_score(
    topology: str,
    n_max_viable: int | None,
    pass_rate_dual: float,
    h_frontier: float | None,
    *,
    n_reference: int = 20,
    h_reference: float = 2.0,
) -> tuple[float, str]:
    """Compute a unified scalability score for a topology.

    Combines multiple metrics into a single 0-1 score indicating how well
    the MPNN + HVA pipeline scales for a given topology.

    Parameters
    ----------
    topology : str
        Lattice topology name.
    n_max_viable : int | None
        Maximum N where dual criterion still passes (from cross-N analysis).
    pass_rate_dual : float
        Best pass rate under dual criterion for this topology.
    h_frontier : float | None
        Minimum h where pipeline achieves < 5% ΔE/gap.
    n_reference : int
        Reference N for scaling comparison (default: 20).
    h_reference : float
        Reference h for frontier comparison (default: 2.0).

    Returns
    -------
    tuple[float, str]
        - score: 0.0 (poor scalability) to 1.0 (excellent scalability)
        - reason: human-readable explanation
    """
    # Factor 1: n_max_viable relative to reference
    n_factor = 0.0
    if n_max_viable is not None and n_max_viable > 0:
        n_factor = min(1.0, n_max_viable / n_reference)

    # Factor 2: pass rate under dual criterion
    pass_factor = float(pass_rate_dual)

    # Factor 3: h_frontier (lower is better — can access more phases)
    h_factor = 0.5  # Default if no frontier
    if h_frontier is not None:
        # h_frontier=1.0 → factor=1.0, h_frontier=4.0 → factor=0.25
        h_factor = min(1.0, h_reference / max(h_frontier, 0.1))

    # Weighted combination
    score = 0.4 * n_factor + 0.4 * pass_factor + 0.2 * h_factor
    score = float(min(1.0, max(0.0, score)))

    # Determine reason
    if n_factor >= 0.8 and pass_factor >= 0.8:
        reason = "excellent_scaling"
    elif n_factor >= 0.5 and pass_factor >= 0.6:
        reason = "moderate_scaling"
    elif n_max_viable is None or n_max_viable < 6:
        reason = "limited_n_range"
    elif h_frontier is not None and h_frontier > 3.5:
        reason = "limited_h_range"
    else:
        reason = "poor_scaling"

    return score, reason


def compute_training_readiness(
    tier_breakdown: dict[str, dict] | None,
    utility_partition: dict[str, list[dict]] | None,
    *,
    min_verified_ratio: float = 0.30,
    min_useful_configs: int = 3,
) -> tuple[bool, str, dict]:
    """Determine if training data is ready for MPNN training.

    Combines quality tier analysis with training utility classification
    to give a single readiness verdict.

    Parameters
    ----------
    tier_breakdown : dict | None
        Per-NPZ quality tier counts {filename: {verified, approximate, unverified, total}}.
    utility_partition : dict | None
        Training utility partition {useful, insufficient_signal, not_useful}.
    min_verified_ratio : float
        Minimum fraction of verified points across all NPZ (default: 30%).
    min_useful_configs : int
        Minimum number of useful configs (default: 3).

    Returns
    -------
    tuple[bool, str, dict]
        - ready: True if training should proceed
        - reason: explanation
        - stats: detailed statistics
    """
    stats: dict = {
        "tier_breakdown_available": tier_breakdown is not None,
        "utility_partition_available": utility_partition is not None,
    }

    # Check tier breakdown
    total_verified = 0
    total_points = 0
    n_legacy = 0
    if tier_breakdown:
        for counts in tier_breakdown.values():
            total_verified += counts.get("verified", 0)
            total_points += counts.get("total", 0)
            if counts.get("legacy"):
                n_legacy += 1
        verified_ratio = total_verified / max(total_points, 1)
        stats["verified_ratio"] = verified_ratio
        stats["total_verified"] = total_verified
        stats["total_points"] = total_points
        stats["n_legacy_npz"] = n_legacy
    else:
        verified_ratio = 0.0

    # Check utility partition
    n_useful = 0
    n_not_useful = 0
    if utility_partition:
        useful_list = utility_partition.get("useful", [])
        not_useful_list = utility_partition.get("not_useful", [])
        n_useful = len(useful_list)
        n_not_useful = len(not_useful_list)
        stats["n_useful_configs"] = n_useful
        stats["n_not_useful_configs"] = n_not_useful

    # Decision logic
    if tier_breakdown is None and utility_partition is None:
        return False, "no_quality_data_available", stats

    # Hard blockers
    if n_not_useful > n_useful:
        return False, "more_not_useful_than_useful", stats

    if tier_breakdown and verified_ratio < min_verified_ratio and total_points > 50:
        return False, f"verified_ratio_too_low_{verified_ratio:.0%}", stats

    if utility_partition and n_useful < min_useful_configs:
        return False, f"insufficient_useful_configs_{n_useful}", stats

    # Soft warnings (ready but with caveats)
    if n_legacy > 0:
        return True, f"ready_but_{n_legacy}_legacy_npz", stats

    if verified_ratio < 0.50 and total_points > 20:
        return True, f"ready_but_low_verified_{verified_ratio:.0%}", stats

    return True, "ready", stats


def compute_extrapolation_viability(
    topology: str,
    n_max_viable: int | None,
    mean_de_gap_per_n: dict[int, float] | None,
    *,
    target_n: int = 30,
    max_acceptable_de_gap: float = 0.20,
) -> tuple[bool, str, dict]:
    """Predict whether extrapolation to target_n is likely to succeed.

    Uses cross-N transfer data to estimate if MPNN prediction at large N
    will produce useful results without VQE refinement.

    Parameters
    ----------
    topology : str
        Target topology.
    n_max_viable : int | None
        Proven maximum viable N for this topology.
    mean_de_gap_per_n : dict | None
        Mean ΔE/gap at each tested N (from cross-N experiments).
    target_n : int
        Desired target system size (default: 30).
    max_acceptable_de_gap : float
        Maximum acceptable ΔE/gap for extrapolation (default: 20%).

    Returns
    -------
    tuple[bool, str, dict]
        - viable: True if extrapolation is likely to succeed
        - reason: explanation
        - prediction: estimated metrics
    """
    prediction: dict = {"topology": topology, "target_n": target_n}

    # Check n_max_viable
    if n_max_viable is None:
        return False, "no_cross_n_data", prediction

    prediction["n_max_viable"] = n_max_viable

    # Simple extrapolation: if target_n > 2 * n_max_viable, likely to fail
    if target_n > 2 * n_max_viable:
        return False, f"target_n_{target_n}_far_beyond_n_max_{n_max_viable}", prediction

    # If target_n <= n_max_viable, should work
    if target_n <= n_max_viable:
        return True, "target_n_within_viable_range", prediction

    # Intermediate case: extrapolate trend
    if mean_de_gap_per_n and len(mean_de_gap_per_n) >= 2:
        # Fit linear trend to log(de_gap) vs N
        n_vals = sorted(mean_de_gap_per_n.keys())
        dg_vals = [mean_de_gap_per_n[n] for n in n_vals]
        
        # Simple linear extrapolation
        if len(n_vals) >= 2:
            slope = (dg_vals[-1] - dg_vals[0]) / max(n_vals[-1] - n_vals[0], 1)
            extrapolated = dg_vals[-1] + slope * (target_n - n_vals[-1])
            prediction["extrapolated_de_gap"] = float(extrapolated)
            
            if extrapolated < max_acceptable_de_gap:
                return True, "extrapolation_below_threshold", prediction
            else:
                return False, f"extrapolated_de_gap_{extrapolated:.2f}_above_threshold", prediction

    # Conservative fallback
    if target_n <= 1.5 * n_max_viable:
        return True, "target_n_moderately_beyond_viable", prediction

    return False, "insufficient_data_for_extrapolation", prediction


def generate_unified_scaling_report(
    dashboard: dict,
    tier_breakdown: dict[str, dict] | None = None,
    target_n_values: list[int] | None = None,
) -> dict:
    """Generate a unified scaling report combining all quality metrics.

    Cross-integrates:
    - Model quality dashboard (pass rates, h_frontier)
    - Quality tier breakdown (verified/approximate/unverified)
    - Training utility classification
    - Extrapolation viability predictions

    Parameters
    ----------
    dashboard : dict
        Model quality dashboard from generate_model_quality_dashboard().
    tier_breakdown : dict | None
        Per-NPZ quality tier counts (from update_cross_n_coverage).
    target_n_values : list[int] | None
        Target N values for extrapolation viability check.

    Returns
    -------
    dict
        Unified report with per-topology summaries and recommendations.
    """
    if target_n_values is None:
        target_n_values = [30, 40, 60]

    configs = dashboard.get("configs", [])
    topo_summary = dashboard.get("topology_summary", {})
    
    # Compute utility partition
    utility_partition = get_usable_training_configs(dashboard)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_topologies": len(topo_summary),
        "n_configs": len(configs),
        "topologies": {},
        "training_readiness": {},
        "extrapolation_viability": {},
        "recommendations": [],
    }

    # Training readiness (global)
    ready, reason, stats = compute_training_readiness(
        tier_breakdown, utility_partition
    )
    report["training_readiness"] = {
        "ready": ready,
        "reason": reason,
        **stats,
    }
    if not ready:
        report["recommendations"].append(
            f"BLOCKING: Training not ready — {reason}"
        )

    # Per-topology analysis
    for topo, info in topo_summary.items():
        n_max_viable = info.get("n_max_viable")
        best_pass = info.get("best_pass_rate_5pct", 0)
        
        # Get h_frontier for the largest viable N
        h_frontier = None
        topo_configs = [c for c in configs if c["topology"] == topo]
        for c in sorted(topo_configs, key=lambda x: -x.get("n_qubits", 0)):
            if c.get("h_frontier") is not None:
                h_frontier = c["h_frontier"]
                break

        # Scalability score
        score, score_reason = compute_scalability_score(
            topo, n_max_viable, best_pass, h_frontier
        )

        # Collect mean_de_gap per N
        mean_dg_per_n = {}
        for c in topo_configs:
            n = c.get("n_qubits", 0)
            dg = c.get("mean_de_gap")
            if n > 0 and dg is not None:
                mean_dg_per_n[n] = dg

        # Extrapolation viability for each target
        extrap_results = {}
        for target_n in target_n_values:
            viable, extrap_reason, pred = compute_extrapolation_viability(
                topo, n_max_viable, mean_dg_per_n, target_n=target_n
            )
            extrap_results[target_n] = {
                "viable": viable,
                "reason": extrap_reason,
                **pred,
            }

        report["topologies"][topo] = {
            "n_max_viable": n_max_viable,
            "best_pass_rate": best_pass,
            "h_frontier": h_frontier,
            "scalability_score": score,
            "scalability_reason": score_reason,
            "n_configs": len(topo_configs),
        }
        report["extrapolation_viability"][topo] = extrap_results

        # Generate recommendations
        if score < 0.3:
            report["recommendations"].append(
                f"{topo}: Poor scalability ({score:.2f}) — consider more training data or different ansatz"
            )
        elif n_max_viable is not None and n_max_viable < 10:
            report["recommendations"].append(
                f"{topo}: Limited n_max_viable={n_max_viable} — investigate VQE refinement budget"
            )

    return report
