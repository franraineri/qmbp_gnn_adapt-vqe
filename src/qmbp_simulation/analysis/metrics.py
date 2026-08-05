"""Analysis Metrics — Pure computation helpers for pipeline diagnostics.

Provides signal-to-noise ratio, parameter smoothness, classification
confidence, energy decomposition, and fraction-near-ground-state
computations. These are stateless functions with no side effects.

This module has NO heavy imports (no Qiskit, no PyTorch).
"""

from __future__ import annotations

import logging
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
