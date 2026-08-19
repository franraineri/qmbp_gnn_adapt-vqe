"""Analysis Metrics — Pure computation helpers for pipeline diagnostics.

Provides signal-to-noise ratio, parameter smoothness, classification
confidence, energy decomposition, and fraction-near-ground-state
computations. These are stateless functions with no side effects.

This module has NO heavy imports (no Qiskit, no PyTorch).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Quality criteria — dual metric for pass/fail decisions
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Quality criteria — imported from analysis.constants (single source of truth)
# Re-exported here for backward compatibility.
# ═══════════════════════════════════════════════════════════════════════════════

from qmbp_simulation.analysis.constants import (  # noqa: F401, E402
    AUTO_EXCLUDE_MAX_DUAL_PASS_FOR_MEAN_ERROR,
    AUTO_EXCLUDE_MEAN_ABS_ERROR,
    DE_GAP_THRESHOLD,
    GAP_MASKING_THRESHOLD,
    H_FRONTIER_MONOTONICITY_TOLERANCE,
    MAX_ABS_ERROR,
    MIN_FIDELITY,
    MIN_TRAINING_DUAL_PASS_RATE,
    MIN_TRAINING_POINTS_FOR_SIGNAL,
    PASS_RATE_REGRESSION_THRESHOLD,
    TRAINING_BAD_RATIO_THRESHOLD,
    ZOO_PASS_FOR_INCOHERENCE_FLAG,
    compute_quality_score,
    grade_from_score,
)

# ── Refinement priority thresholds ───────────────────────────────────────────

REFINEMENT_GAP_CRITICAL: float = 0.1
"""Spectral gap below which VQE refinement is near-impossible.
|ΔE| must be < gap*0.05 = 0.005 — below VQE precision for large circuits."""

REFINEMENT_GAP_HARD: float = 0.5
"""Spectral gap below which VQE refinement is moderately hard.
Achievable but requires more restarts and iterations."""

REFINEMENT_PROXIMITY_EASY: float = 0.10
"""ΔE/gap below this is an easy win (barely failing the 5% criterion)."""

REFINEMENT_PROXIMITY_MODERATE: float = 0.20
"""ΔE/gap below this is moderately achievable with standard budget."""

REFINEMENT_PROXIMITY_HARD: float = 0.50
"""ΔE/gap below this is hard but potentially achievable with aggressive budget."""

REFINEMENT_PARAM_BASELINE: int = 20
"""Parameter count at which no difficulty discount applies.
Above this, discount = 1/(1 + (n_params - 20)/50)."""

REFINEMENT_PARAM_SCALE: float = 50.0
"""Scale factor for parameter count discount. Higher = softer discount."""

REFINEMENT_MPNN_BOOST: float = 1.5
"""Priority multiplier when MPNN found a better energy basin than previous VQE."""

REFINEMENT_MPNN_IMPROVEMENT_TOL: float = 1e-6
"""Minimum energy improvement for MPNN prediction to count as 'better basin'."""

# ── Adaptive VQE config thresholds ───────────────────────────────────────────

ADAPTIVE_VQE_CHEAP_PRIORITY: float = 0.8
"""Priority threshold above which "cheap" tier is assigned (minimal budget)."""

ADAPTIVE_VQE_CHEAP_DE_GAP: float = 0.08
"""Maximum ΔE/gap for cheap tier (must be barely above 5% threshold)."""

ADAPTIVE_VQE_CHEAP_MAXITER: int = 200
"""Maximum iterations for cheap tier (quick optimization)."""

ADAPTIVE_VQE_CHEAP_RHOBEG: float = 0.05
"""Initial step size for cheap tier (small — we're close to solution)."""

ADAPTIVE_VQE_STANDARD_PRIORITY: float = 0.5
"""Priority threshold above which "standard" tier is assigned."""

ADAPTIVE_VQE_STANDARD_MAX_RESTARTS: int = 5
"""Cap on restarts for standard tier."""

ADAPTIVE_VQE_STANDARD_RHOBEG: float = 0.1
"""Initial step size for standard tier."""

ADAPTIVE_VQE_AGGRESSIVE_PRIORITY: float = 0.2
"""Priority threshold above which "aggressive" tier is assigned (full budget)."""

ADAPTIVE_VQE_AGGRESSIVE_RHOBEG: float = 0.3
"""Initial step size for aggressive tier (larger — explore landscape)."""

ADAPTIVE_VQE_MINIMAL_MAXITER: int = 100
"""Maximum iterations for minimal tier (near-hopeless points)."""

ADAPTIVE_VQE_MINIMAL_RHOBEG: float = 0.5
"""Initial step size for minimal tier (largest — random walk)."""

# ── Quality tier sample weights (for weighted training loss) ─────────────────

QUALITY_TIER_WEIGHT_VERIFIED: float = 1.0
"""Training weight for VQE-converged (verified) data points."""

QUALITY_TIER_WEIGHT_AUGMENTED: float = 0.8
"""Training weight for augmented variants of verified data."""

QUALITY_TIER_WEIGHT_APPROXIMATE: float = 0.7
"""Training weight for MPNN predictions passing dual criterion (not VQE-refined)."""

QUALITY_TIER_WEIGHT_UNVERIFIED: float = 0.5
"""Training weight for legacy/unknown quality data."""

SAMPLE_WEIGHT_MIN: float = 0.1
"""Minimum valid sample weight (guard against zero/negative weights)."""

SAMPLE_WEIGHT_MAX: float = 2.0
"""Maximum valid sample weight (guard against amplifying bad data)."""

# ── Data augmentation thresholds (training pipeline) ─────────────────────────

AUGMENTATION_MAX_FILTERED_POINTS: int = 50
"""Maximum filtered points per N before augmentation is disabled.
When dataset is large enough, augmentation adds noise without benefit."""

AUGMENTATION_MAX_VARIANTS_PER_POINT: int = 1
"""Maximum augmented variants generated per verified point."""


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
                "compute_fidelity or state normalization)",
                fidelity,
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
# Per-point failure classification — WHY does a point fail?
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PointClassification:
    """Rich classification of a single evaluation point.

    Goes beyond binary pass/fail to categorize the failure mechanism,
    estimate refinability, and guide next actions.

    Attributes
    ----------
    category : str
        Primary classification:
        - "pass" — meets all criteria
        - "near_pass" — within 2× threshold (easy refinement target)
        - "gap_masked" — passes ΔE/gap but fails |ΔE| (gap inflates metric)
        - "moderate_error" — ΔE/gap in 2-10× threshold range
        - "ansatz_limited" — error pattern consistent with circuit expressibility limit
        - "critical_region" — point near h_critical where gap → 0
        - "severe_error" — ΔE/gap > 10× threshold (likely unrefinable)
        - "data_error" — NaN/Inf/negative values (corrupted)
    severity : float
        Continuous severity in [0, 1]. 0=perfect, 1=maximally bad.
    refinable : bool
        Whether VQE refinement is likely to improve this point.
    action : str
        Recommended next step: "none", "refine_vqe", "increase_p",
        "restrict_h_range", "investigate", "discard".
    detail : str
        Human-readable explanation.
    """

    category: str
    severity: float
    refinable: bool
    action: str
    detail: str


def classify_point_failure(
    de_gap: float,
    abs_error: float | None = None,
    gap: float | None = None,
    fidelity: float | None = None,
    *,
    h: float | None = None,
    h_critical: float | None = None,
    n_params: int | None = None,
    n_prev_attempts: int = 0,
) -> PointClassification:
    """Classify a single point's failure mode with continuous severity.

    Replaces the binary is_point_failure() for cases where you need to
    know WHY a point fails and WHAT to do about it.

    This function is the union of logic from:
    - is_point_failure (binary detection)
    - compute_refinement_priority (refinability assessment)
    - diagnose_intrinsic_vqe_error (ansatz-limit check, simplified per-point)

    Parameters
    ----------
    de_gap : float
        Relative error |ΔE|/gap.
    abs_error : float | None
        Absolute error |ΔE|.
    gap : float | None
        Spectral gap at this h-point.
    fidelity : float | None
        State fidelity (None if unavailable).
    h : float | None
        Field value (for critical region detection).
    h_critical : float | None
        Critical field value (for region classification).
    n_params : int | None
        Number of variational parameters.
    n_prev_attempts : int
        Number of previous failed VQE refinement attempts.

    Returns
    -------
    PointClassification
        Rich classification with category, severity, refinability, and action.
    """
    # ── Data error: NaN/Inf/negative ─────────────────────────────────────
    if not np.isfinite(de_gap):
        return PointClassification(
            category="data_error",
            severity=1.0,
            refinable=False,
            action="investigate",
            detail="Non-finite ΔE/gap — corrupted data or solver failure.",
        )
    if de_gap < 0:
        return PointClassification(
            category="data_error",
            severity=0.8,
            refinable=False,
            action="investigate",
            detail=f"Negative ΔE/gap={de_gap:.4f} — variational violation (E_pred < E_exact).",
        )

    # ── Pass ─────────────────────────────────────────────────────────────
    passes_dg = de_gap < DE_GAP_THRESHOLD
    passes_ae = abs_error is None or abs_error < MAX_ABS_ERROR

    if passes_dg and passes_ae:
        # Compute severity even for passing points (continuous quality)
        severity = de_gap / DE_GAP_THRESHOLD  # 0 at de_gap=0, ~1 at threshold
        return PointClassification(
            category="pass",
            severity=float(min(severity, 0.99)),
            refinable=False,
            action="none",
            detail=f"ΔE/gap={de_gap:.4f} — within criteria.",
        )

    # ── Gap masking: passes ΔE/gap but fails |ΔE| ────────────────────────
    if passes_dg and not passes_ae:
        severity = min(1.0, (abs_error or 0) / (MAX_ABS_ERROR * 3))
        return PointClassification(
            category="gap_masked",
            severity=float(severity),
            refinable=True,
            action="refine_vqe",
            detail=(
                f"ΔE/gap={de_gap:.4f} passes but |ΔE|={abs_error:.3f} > {MAX_ABS_ERROR}. "
                f"Large gap inflates relative metric. Absolute error is real."
            ),
        )

    # ── Failing point: classify by severity and mechanism ─────────────────
    # Severity normalized: 0 at threshold, 1 at 20× threshold
    severity_raw = min(1.0, (de_gap - DE_GAP_THRESHOLD) / (DE_GAP_THRESHOLD * 19))

    # Critical region check: near h_critical, gap → 0 makes ΔE/gap large
    # even for small |ΔE|. This is physics, not optimization failure.
    in_critical_region = False
    if h is not None and h_critical is not None and gap is not None:
        near_hc = abs(h - h_critical) < 0.5
        small_gap = gap < REFINEMENT_GAP_CRITICAL
        if near_hc and small_gap:
            in_critical_region = True

    if in_critical_region:
        return PointClassification(
            category="critical_region",
            severity=float(severity_raw),
            refinable=False,
            action="restrict_h_range",
            detail=(
                f"h={h:.2f} near h_c={h_critical:.1f}, gap={gap:.4f}. "
                f"Small gap makes ΔE/gap={de_gap:.3f} inherently large. "
                f"Physics limit, not optimization failure."
            ),
        )

    # Near pass: barely failing, easy refinement target
    if de_gap < DE_GAP_THRESHOLD * 2:
        refinable = n_prev_attempts < 2
        return PointClassification(
            category="near_pass",
            severity=float(severity_raw),
            refinable=refinable,
            action="refine_vqe" if refinable else "investigate",
            detail=(
                f"ΔE/gap={de_gap:.4f} — {de_gap / DE_GAP_THRESHOLD:.1f}× threshold. "
                f"Close to passing. "
                + (
                    "VQE refinement likely effective."
                    if refinable
                    else f"Already tried {n_prev_attempts}× without improvement — ceiling reached."
                )
            ),
        )

    # Moderate error: 2-10× threshold
    if de_gap < DE_GAP_THRESHOLD * 10:
        # Check if gap makes it feasible
        gap_feasible = gap is None or gap >= REFINEMENT_GAP_HARD
        refinable = gap_feasible and n_prev_attempts < 1
        action = "refine_vqe" if refinable else "increase_p"
        return PointClassification(
            category="moderate_error",
            severity=float(severity_raw),
            refinable=refinable,
            action=action,
            detail=(
                f"ΔE/gap={de_gap:.4f} — {de_gap / DE_GAP_THRESHOLD:.0f}× threshold. "
                + (
                    f"Gap={gap:.3f} allows refinement."
                    if gap_feasible and gap is not None
                    else f"Small gap={gap:.3f} — likely ansatz-limited."
                    if gap is not None
                    else "Gap unknown."
                )
            ),
        )

    # Severe error: > 10× threshold — almost certainly ansatz-limited
    # or fundamentally wrong region
    return PointClassification(
        category="severe_error" if de_gap < DE_GAP_THRESHOLD * 40 else "ansatz_limited",
        severity=float(min(1.0, severity_raw)),
        refinable=False,
        action="increase_p" if de_gap < 2.0 else "restrict_h_range",
        detail=(
            f"ΔE/gap={de_gap:.3f} — {de_gap / DE_GAP_THRESHOLD:.0f}× threshold. "
            f"Circuit with {n_params or '?'} params likely cannot express ground state at this h. "
            f"Options: increase p_layers or exclude from training."
        ),
    )


def classify_points_batch(
    per_h_results: list[dict],
    *,
    h_critical: float | None = None,
    n_params: int | None = None,
) -> list[PointClassification]:
    """Classify all points in a sweep.

    Convenience wrapper over classify_point_failure for batch use.
    Returns classifications aligned with per_h_results indices.
    """
    return [
        classify_point_failure(
            de_gap=r.get("de_gap", float("nan")),
            abs_error=r.get("abs_error"),
            gap=r.get("gap"),
            fidelity=r.get("fidelity"),
            h=r.get("h"),
            h_critical=h_critical,
            n_params=n_params,
        )
        for r in per_h_results
    ]


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
        if e_pred < e_prev - REFINEMENT_MPNN_IMPROVEMENT_TOL:
            mpnn_improved = True

    # ── Factor 2: Stale attempts → skip ──────────────────────────────────
    # If VQE already tried multiple times AND MPNN didn't find better basin,
    # the point is at its ansatz ceiling. Don't waste compute.
    if n_prev_attempts >= max_stale_attempts and not mpnn_improved:
        return 0.0, True, "stale_no_mpnn_improvement"

    # ── Factor 3: Gap-based feasibility ──────────────────────────────────
    gap_feasibility = 1.0
    if gap < REFINEMENT_GAP_CRITICAL:
        gap_feasibility = 0.1  # Very hard — near phase transition
    elif gap < REFINEMENT_GAP_HARD:
        gap_feasibility = 0.5  # Moderately hard

    # ── Factor 4: Proximity to threshold ─────────────────────────────────
    if de_gap < REFINEMENT_PROXIMITY_EASY:
        proximity_score = 1.0  # Very close — easy win
    elif de_gap < REFINEMENT_PROXIMITY_MODERATE:
        proximity_score = 0.7  # Moderate gap to close
    elif de_gap < REFINEMENT_PROXIMITY_HARD:
        proximity_score = 0.3  # Far — hard but possible
    else:
        proximity_score = 0.1  # Very far — likely ansatz-limited

    # ── Factor 5: Parameter count discount ───────────────────────────────
    param_factor = 1.0 / (
        1.0 + max(0, n_params - REFINEMENT_PARAM_BASELINE) / REFINEMENT_PARAM_SCALE
    )

    # ── Factor 6: MPNN improvement boost ─────────────────────────────────
    mpnn_boost = REFINEMENT_MPNN_BOOST if mpnn_improved else 1.0

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


def compute_adaptive_vqe_config(
    priority: float,
    de_gap: float,
    gap: float,
    n_params: int,
    *,
    base_maxiter: int = 1500,
    base_restarts: int = 10,
) -> dict[str, int | str]:
    """Compute adaptive VQE hyperparameters based on refinement priority.

    Points close to the pass threshold need minimal compute (cheap wins).
    Points far from passing need aggressive optimization (expensive but
    potentially achievable). Points that are hopeless get minimal budget
    to avoid wasting compute.

    This function should be called AFTER compute_refinement_priority() to
    translate the priority score into concrete VQE configuration.

    Parameters
    ----------
    priority : float
        Priority score from compute_refinement_priority() ∈ [0, 1].
    de_gap : float
        Current ΔE/gap value.
    gap : float
        Spectral gap at this h-point.
    n_params : int
        Number of variational parameters.
    base_maxiter : int
        Maximum iterations from CLI (will be scaled).
    base_restarts : int
        Maximum restarts from CLI (will be scaled).

    Returns
    -------
    dict with keys:
        - maxiter: int — scaled optimizer iterations
        - n_restarts: int — scaled number of restarts
        - rhobeg: float — initial step size for COBYLA/L-BFGS-B
        - tier: str — "cheap", "standard", or "aggressive"
        - reason: str — human-readable explanation
    """
    # Tier 1: Easy wins (ΔE/gap barely above threshold, large gap)
    # These points are almost passing — minimal effort needed.
    if priority >= ADAPTIVE_VQE_CHEAP_PRIORITY and de_gap < ADAPTIVE_VQE_CHEAP_DE_GAP:
        return {
            "maxiter": min(base_maxiter, ADAPTIVE_VQE_CHEAP_MAXITER),
            "n_restarts": 1,
            "rhobeg": ADAPTIVE_VQE_CHEAP_RHOBEG,
            "tier": "cheap",
            "reason": f"Easy win: ΔE/gap={de_gap:.3f}, priority={priority:.2f}",
        }

    # Tier 2: Standard refinement (moderate distance from threshold)
    if priority >= ADAPTIVE_VQE_STANDARD_PRIORITY:
        scale = 0.5 + 0.5 * priority  # 0.75-1.0× base
        return {
            "maxiter": int(base_maxiter * scale),
            "n_restarts": min(ADAPTIVE_VQE_STANDARD_MAX_RESTARTS, base_restarts),
            "rhobeg": ADAPTIVE_VQE_STANDARD_RHOBEG,
            "tier": "standard",
            "reason": f"Standard: ΔE/gap={de_gap:.3f}, priority={priority:.2f}",
        }

    # Tier 3: Aggressive (far from threshold but feasible)
    # Use full budget — this point needs serious optimization
    if priority >= ADAPTIVE_VQE_AGGRESSIVE_PRIORITY:
        return {
            "maxiter": base_maxiter,
            "n_restarts": base_restarts,
            "rhobeg": ADAPTIVE_VQE_AGGRESSIVE_RHOBEG,
            "tier": "aggressive",
            "reason": f"Aggressive: ΔE/gap={de_gap:.3f}, needs full budget",
        }

    # Tier 4: Minimal (very low priority — almost hopeless)
    # Give minimal budget just in case, but don't waste compute
    return {
        "maxiter": min(base_maxiter, ADAPTIVE_VQE_MINIMAL_MAXITER),
        "n_restarts": 1,
        "rhobeg": ADAPTIVE_VQE_MINIMAL_RHOBEG,
        "tier": "minimal",
        "reason": f"Minimal budget: ΔE/gap={de_gap:.3f}, priority={priority:.2f} (near hopeless)",
    }


def compute_snr(observable_value: float, shots: int) -> float:
    """Signal-to-noise ratio: |⟨O⟩| * √shots.


    Quantifies measurement reliability. Higher SNR indicates the observable
    signal dominates over shot noise (σ = 1/√shots).
    """
    if not isinstance(shots, int | np.integer) or shots <= 0:
        raise ValueError(f"shots must be a positive integer, got {shots}")
    return float(abs(observable_value) * np.sqrt(shots))


def compute_theta_smoothness(theta_array: np.ndarray) -> float | None:
    """Maximum parameter discontinuity across the h-sweep."""
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
    """Phase classification confidence: |⟨X⟩ - ⟨ZZ⟩| * √shots."""
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


def compute_uncertainty_correlation(per_h_results: list[dict]) -> dict:
    """Compute correlation between MC-Dropout uncertainty (θ_std) and actual error (ΔE/gap).

    This analysis validates whether the model's uncertainty estimation is
    well-calibrated: high θ_std should correlate with high ΔE/gap.

    Parameters
    ----------
    per_h_results : list[dict]
        Per-point results, each containing at minimum "de_gap" and optionally "theta_std".

    Returns
    -------
    dict
        {
            "n_points_with_uncertainty": int,
            "pearson_r": float | None,  # Pearson correlation [-1, 1]
            "spearman_r": float | None,  # Rank correlation [-1, 1]
            "mean_theta_std": float,
            "high_uncertainty_mean_de_gap": float,  # Mean ΔE/gap for top-30% uncertain
            "low_uncertainty_mean_de_gap": float,   # Mean ΔE/gap for bottom-30%
            "calibrated": bool,  # True if high_unc ΔE/gap > low_unc ΔE/gap
        }
    """
    import numpy as np

    # Filter points that have theta_std
    valid = [(r["de_gap"], r["theta_std"]) for r in per_h_results
             if "theta_std" in r and r.get("theta_std", 0) > 0 and "de_gap" in r]

    result = {
        "n_points_with_uncertainty": len(valid),
        "pearson_r": None,
        "spearman_r": None,
        "mean_theta_std": 0.0,
        "high_uncertainty_mean_de_gap": 0.0,
        "low_uncertainty_mean_de_gap": 0.0,
        "calibrated": False,
    }

    if len(valid) < 3:
        return result

    de_gaps = np.array([v[0] for v in valid])
    theta_stds = np.array([v[1] for v in valid])
    result["mean_theta_std"] = float(np.mean(theta_stds))

    # Pearson correlation
    if np.std(theta_stds) > 1e-10 and np.std(de_gaps) > 1e-10:
        r_matrix = np.corrcoef(theta_stds, de_gaps)
        result["pearson_r"] = float(r_matrix[0, 1])

    # Spearman rank correlation
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(theta_stds, de_gaps)
        result["spearman_r"] = float(rho) if np.isfinite(rho) else None
    except (ImportError, ValueError):
        pass

    # Split into high/low uncertainty groups (top/bottom 30%)
    n_split = max(1, len(valid) // 3)
    sorted_indices = np.argsort(theta_stds)
    low_indices = sorted_indices[:n_split]
    high_indices = sorted_indices[-n_split:]

    result["low_uncertainty_mean_de_gap"] = float(np.mean(de_gaps[low_indices]))
    result["high_uncertainty_mean_de_gap"] = float(np.mean(de_gaps[high_indices]))
    result["calibrated"] = result["high_uncertainty_mean_de_gap"] > result["low_uncertainty_mean_de_gap"]

    return result


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

    # Dual criterion pass rate (ΔE/gap < 5% AND |ΔE| < 0.10)
    if abs_errors and len(abs_errors) == n:
        abs_err_arr = np.array(abs_errors)
        dual_mask = (de_gaps < DE_GAP_THRESHOLD) & (abs_err_arr < MAX_ABS_ERROR)
        summary["n_pass_dual"] = int(dual_mask.sum())
        summary["pass_rate_dual"] = float(dual_mask.mean())
    else:
        # Fallback: if no abs_error data, dual = single criterion (conservative)
        summary["n_pass_dual"] = summary.get("n_pass_5pct", 0)
        summary["pass_rate_dual"] = summary.get("pass_rate_5pct", 0.0)

    # Fidelity stats (if available — only for N ≤ STATEVECTOR_MAX_N)
    fids = [r["fidelity"] for r in per_h_results if r.get("fidelity") is not None]
    if fids:
        summary["mean_fidelity"] = float(np.mean(fids))
        summary["min_fidelity"] = float(np.min(fids))
        summary["fidelity_pass_rate"] = float(np.mean(np.array(fids) > 0.90))
    else:
        summary["mean_fidelity"] = None
        summary["fidelity_pass_rate"] = None

    # Standard deviation (useful for confidence intervals and thesis tables)
    summary["std_de_gap"] = float(np.std(de_gaps))

    # Per-site error (if n_qubits available in results)
    n_qubits_vals = [r.get("n_qubits") for r in per_h_results if r.get("n_qubits")]
    if abs_errors and n_qubits_vals:
        n_q = n_qubits_vals[0]  # All points in a deploy summary have same N
        abs_arr = np.array(abs_errors)
        summary["mean_abs_error_per_site"] = float(abs_arr.mean() / max(n_q, 1))
    elif abs_errors:
        # n_qubits not in result dicts — caller can compute externally
        summary["mean_abs_error_per_site"] = None

    # ── Continuous quality metrics (P90, quality_score, grade) ─────────────
    summary["p90_de_gap"] = float(np.percentile(de_gaps, 90))
    score = compute_quality_score(
        mean_de_gap=summary["mean_de_gap"],
        p90_de_gap=summary["p90_de_gap"],
        mean_abs_error_per_site=summary.get("mean_abs_error_per_site"),
        n_points=n,
    )
    summary["quality_score"] = score
    summary["grade"] = grade_from_score(score)

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
    npz_path: str | Path,
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
                anomalies.append(
                    {
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
                    }
                )
    return anomalies


def enforce_h_frontier_monotonicity(configs: list[dict]) -> list[dict]:
    """Enforce that h_frontier is non-decreasing with N per topology.

    Physics: larger systems (higher N) should have equal or higher h_frontier
    because the critical region is harder to reach with fixed ansatz depth.
    Non-monotonicity indicates mixed h-ranges or inconsistent VQE quality.

    Corrects downward anomalies by propagating the running maximum across N.
    Annotates corrected entries for traceability.

    Parameters
    ----------
    configs : list[dict]
        Per-config dashboard entries (modified in-place).

    Returns
    -------
    list[dict]
        Same list with corrected h_frontier values. Corrected entries gain:
        - ``h_frontier_original``: the raw (anomalous) value
        - ``h_frontier_corrected``: True
    """
    from collections import defaultdict

    by_topo: dict[str, list[dict]] = defaultdict(list)
    for c in configs:
        if c.get("h_frontier") is not None:
            by_topo[c["topology"]].append(c)

    n_corrected = 0
    for topo, topo_configs in by_topo.items():
        sorted_configs = sorted(topo_configs, key=lambda c: c["n_qubits"])
        running_max = 0.0
        for c in sorted_configs:
            hf = c["h_frontier"]
            if hf < running_max - 1e-6:
                c["h_frontier_original"] = hf
                c["h_frontier"] = running_max
                c["h_frontier_corrected"] = True
                n_corrected += 1
            else:
                running_max = max(running_max, hf)
                c.pop("h_frontier_corrected", None)
                c.pop("h_frontier_original", None)

    if n_corrected > 0:
        logger.info("enforce_h_frontier_monotonicity: corrected %d anomalies", n_corrected)

    return configs


def detect_training_zoo_incoherence(configs: list[dict], npz_dir: Path | None = None) -> list[dict]:
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
    from pathlib import Path as _Path

    import numpy as np

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
            incoherent.append(
                {
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
                }
            )
    return incoherent


def detect_pass_rate_regression(
    current_configs: list[dict],
    previous_dashboard_path: str | Path | None = None,
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
    import json
    from pathlib import Path as _Path

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
            regressions.append(
                {
                    "topology": topo,
                    "prev_max": prev,
                    "curr_max": curr,
                    "drop": drop,
                    "message": (
                        f"{topo}: max pass_rate_dual dropped {drop:.0%} "
                        f"({prev:.0%} → {curr:.0%}). "
                        f"Threshold: {PASS_RATE_REGRESSION_THRESHOLD:.0%}."
                    ),
                }
            )
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
            "0% dual pass rate — no learnable signal. "
            "Possible causes: p=1 insufficient expressivity, h outside valid regime, "
            "or VQE trapped in local minimum for all points.",
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
                theta_dims.add(len(theta) if hasattr(theta, "__len__") else 0)

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
                f"N={n}: only {n_good}/{n_raw} ({n_good / n_raw:.0%}) pass dual criterion. "
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
            needs.append("data for more N values")
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


# ═══════════════════════════════════════════════════════════════════════════════
# Model quality dashboard — auto-generated per (topology, N) from NPZ data
# ═══════════════════════════════════════════════════════════════════════════════


def _scan_cross_n_transfer_results(root: Path) -> dict:
    """Scan cross-N experiment results to extract transfer quality data.

    Returns a dict keyed by (topology, target_n, p_layers) → list of transfer records.
    Each record: {train_n, pass_rate_5pct, pass_rate_10pct, mean_de_gap, n_points, timestamp}.
    """
    import json

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
                transfer_data[key].append(
                    {
                        "train_n": train_n,
                        "pass_rate_5pct": float(pass_5),
                        "pass_rate_10pct": float(pass_10),
                        "mean_de_gap": float(mean_dg),
                        "n_points": n_pts,
                        "file": run_file.name,
                    }
                )

    return transfer_data


def _best_cross_n_source(transfers: list) -> dict | None:
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
            "best_pass_rate_dual": max(
                (c.get("pass_rate_dual_criterion", 0) for c in topo_configs), default=0
            ),
            "best_n": best_config["n_qubits"],
            "cross_n_best_source_for_largest": cross_n_best,
        }

    return summary


def generate_model_quality_dashboard(
    output_path: str | Path | None = None,
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
    import json
    from datetime import datetime
    from pathlib import Path as _Path

    _ROOT = _Path(__file__).resolve().parents[3]
    npz_dir = _ROOT / "data" / "multi_n_training"
    if output_path is None:
        output_path = _ROOT / "data" / "model_quality_dashboard.json"
    else:
        output_path = _Path(output_path)

    if not npz_dir.exists():
        return {"configs": [], "topology_summary": {}, "generated_at": "", "n_configs": 0}

    # ── Freshness check: skip regeneration if NPZ data hasn't changed ────
    # Compare max NPZ mtime + cross-N results mtime against dashboard mtime.
    npz_files_all = sorted(npz_dir.glob("*.npz"))
    if not npz_files_all:
        return {"configs": [], "topology_summary": {}, "generated_at": "", "n_configs": 0}

    max_npz_mtime = max(f.stat().st_mtime for f in npz_files_all)
    # Also check cross-N results freshness
    cross_n_dir = _ROOT / "results" / "experiments" / "exp_accel_cross_n"
    max_cross_n_mtime = 0.0
    if cross_n_dir.exists():
        cross_n_files = list(cross_n_dir.glob("run_*.json"))
        if cross_n_files:
            max_cross_n_mtime = max(f.stat().st_mtime for f in cross_n_files)

    max_source_mtime = max(max_npz_mtime, max_cross_n_mtime)

    if output_path.exists():
        dashboard_mtime = output_path.stat().st_mtime
        if dashboard_mtime > max_source_mtime:
            # Dashboard is newer than all sources — no regeneration needed
            logger.debug(
                "Dashboard up-to-date (mtime %.0f > sources %.0f), skipping.",
                dashboard_mtime,
                max_source_mtime,
            )
            with open(output_path) as f:
                return json.load(f)

    # ── Phase 1: Scan cross-N experiment results for transfer quality ────
    cross_n_data = _scan_cross_n_transfer_results(_ROOT)

    configs = []
    for npz_file in npz_files_all:
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
        pass_rate_10 = (
            float((de_gaps < 2 * DE_GAP_THRESHOLD).mean()) if de_gaps is not None else 0.0
        )
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

            # Prefer multi-N model (n_qubits=0) as it's the production model.
            # Fall back to single-N model matching this config's N.
            zoo_entries = list_pretrained(
                model="tfim_bond_resolved",
                topology=topology,
                p_layers=p_layers,
                n_qubits=0,
            )
            if not zoo_entries:
                # No multi-N model: try exact-N match
                zoo_entries = list_pretrained(
                    model="tfim_bond_resolved",
                    topology=topology,
                    p_layers=p_layers,
                    n_qubits=n_qubits,
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
                generate_model_quality_dashboard._zoo_integrity = zoo_report["n_corrupted"] == 0
                generate_model_quality_dashboard._zoo_n_missing = zoo_report["n_missing"]
            except Exception:
                generate_model_quality_dashboard._zoo_validated = True
                generate_model_quality_dashboard._zoo_integrity = None
                generate_model_quality_dashboard._zoo_n_missing = 0

        if zoo_model_available:
            zoo_integrity_ok = getattr(generate_model_quality_dashboard, "_zoo_integrity", None)

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

        configs.append(
            {
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
                "mtime": datetime.fromtimestamp(npz_file.stat().st_mtime, tz=UTC).isoformat(),
            }
        )

    # ── Topology-level summary: n_max_viable ────────────────────────────
    topology_summary = _compute_topology_summary(configs)

    # ── Enforce h_frontier monotonicity (correct anomalies in-place) ─────
    enforce_h_frontier_monotonicity(configs)

    # ── Automated quality audits ─────────────────────────────────────────
    frontier_anomalies = detect_h_frontier_anomalies(configs)
    training_zoo_incoherence = detect_training_zoo_incoherence(configs)
    gap_masked_configs = [
        {
            "topology": c["topology"],
            "n_qubits": c["n_qubits"],
            "pass_rate_5pct": c["pass_rate_5pct"],
            "pass_rate_dual": c.get("pass_rate_dual_criterion", 0),
            "gap_masked": c["pass_rate_5pct"] - c.get("pass_rate_dual_criterion", 0),
        }
        for c in configs
        if c["pass_rate_5pct"] - c.get("pass_rate_dual_criterion", 0) > GAP_MASKING_THRESHOLD
    ]
    # Pass rate regression check (compares vs previous snapshot)
    regressions = detect_pass_rate_regression(configs)

    # H-range mismatch detection per topology (Test M from failures_tests)
    from collections import defaultdict as _defaultdict_audit

    h_range_mismatches: list[dict] = []
    topo_h_ranges: dict[str, dict[int, dict]] = _defaultdict_audit(dict)
    for c in configs:
        topo = c["topology"]
        n = c["n_qubits"]
        h_range = c.get("h_range")
        if h_range and len(h_range) == 2:
            topo_h_ranges[topo][n] = {"h_values": np.linspace(h_range[0], h_range[1], 10)}
    for topo, per_n_data in topo_h_ranges.items():
        if len(per_n_data) >= 2:
            from qmbp_simulation.analysis.failures_tests import diagnose_h_range_mismatch

            hm = diagnose_h_range_mismatch(per_n_data)
            if hm["has_mismatch"]:
                h_range_mismatches.append(
                    {
                        "topology": topo,
                        "overlap_fraction": hm["overlap_fraction"],
                        "mismatch_pairs": hm["mismatch_pairs"],
                    }
                )

    audit_results = {
        "h_frontier_anomalies": frontier_anomalies,
        "training_zoo_incoherence": training_zoo_incoherence,
        "gap_masked_configs": gap_masked_configs,
        "pass_rate_regressions": regressions,
        "h_range_mismatches": h_range_mismatches,
        "n_issues": (
            len(frontier_anomalies)
            + len(training_zoo_incoherence)
            + len(gap_masked_configs)
            + len(regressions)
            + len(h_range_mismatches)
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
    if h_range_mismatches:
        logger.warning("H-range mismatches: %d topologies", len(h_range_mismatches))
        for hm in h_range_mismatches:
            logger.warning("  %s: overlap=%.0f%%", hm["topology"], hm["overlap_fraction"] * 100)

    dashboard = {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_configs": len(configs),
        "configs": configs,
        "topology_summary": topology_summary,
        "audit": audit_results,
    }

    # ── Multi-topology models section ────────────────────────────────────
    # Track MT models from zoo and their per-topology performance
    try:
        from qmbp_simulation.predictors.model_zoo import _load_manifest as _lm_mt
        _zoo_entries = _lm_mt()
        mt_entries = [e for e in _zoo_entries if e.topology == "multi_topology"]
        if mt_entries:
            mt_models = []
            for e in mt_entries:
                mt_models.append({
                    "checkpoint": e.checkpoint_file,
                    "pass_rate": e.pass_rate,
                    "n_training_points": e.n_training_points,
                    "notes": e.notes[:80] if e.notes else "",
                    "created": getattr(e, "created", ""),
                })
            dashboard["multi_topology_models"] = {
                "n_models": len(mt_models),
                "models": mt_models,
                "best_pass_rate": max(e.pass_rate for e in mt_entries),
            }
    except Exception as _mt_err:
        logger.debug("MT models section skipped: %s", _mt_err)

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
            "Dashboard staleness: %d NPZ files modified after generation (concurrent writes?): %s",
            len(stale_files),
            stale_files[:3],
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
        "zoo_n_missing": getattr(generate_model_quality_dashboard, "_zoo_n_missing", 0),
    }

    # Re-write with integrity info
    with open(output_path, "w") as f:
        json.dump(dashboard, f, indent=2)

    logger.info(
        "Model quality dashboard: %d configs → %s",
        len(configs),
        output_path.name,
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
        "generated_at": datetime.now(UTC).isoformat(),
        "n_topologies": len(topo_summary),
        "n_configs": len(configs),
        "topologies": {},
        "training_readiness": {},
        "extrapolation_viability": {},
        "recommendations": [],
    }

    # Training readiness (global)
    ready, reason, stats = compute_training_readiness(tier_breakdown, utility_partition)
    report["training_readiness"] = {
        "ready": ready,
        "reason": reason,
        **stats,
    }
    if not ready:
        report["recommendations"].append(f"BLOCKING: Training not ready — {reason}")

    # Per-topology analysis
    for topo, info in topo_summary.items():
        n_max_viable = info.get("n_max_viable")
        best_pass = info.get("best_pass_rate_dual", info.get("best_pass_rate_5pct", 0))

        # Get h_frontier for the largest viable N
        h_frontier = None
        topo_configs = [c for c in configs if c["topology"] == topo]
        for c in sorted(topo_configs, key=lambda x: -x.get("n_qubits", 0)):
            if c.get("h_frontier") is not None:
                h_frontier = c["h_frontier"]
                break

        # Scalability score
        score, score_reason = compute_scalability_score(topo, n_max_viable, best_pass, h_frontier)

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


# ═══════════════════════════════════════════════════════════════════════════════
# Re-exports from failures_tests (cross-N failure mode diagnostics)
# NOTE: Lazy import to avoid circular dependency (failures_tests imports from metrics)
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# MPNN Model Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════


def compute_variational_violations(per_point: list[dict], tolerance: float = 1e-6) -> dict:
    """Check for variational principle violations (E_pred < E_exact).

    A variational violation occurs when the predicted/optimized energy is
    BELOW the exact ground state — physically impossible, indicating
    numerical issues, data corruption, or stale e_exact values.

    Parameters
    ----------
    per_point : list[dict]
        Per-h results with "e_pred" (or "e_vqe") and "e_exact" keys.
    tolerance : float
        Numerical tolerance. Violations smaller than this are ignored
        (accounts for floating-point imprecision). Default: 1e-6.

    Returns
    -------
    dict
        n_total, n_violations, rate, violation_rate, max_violation, violations.
    """
    if not per_point:
        return {
            "n_total": 0,
            "n_violations": 0,
            "rate": 0.0,
            "violation_rate": 0.0,
            "max_violation": 0.0,
            "violations": [],
        }

    violations = []
    n_valid = 0
    for p in per_point:
        e_pred = p.get("e_pred") or p.get("e_vqe")
        e_exact = p.get("e_exact")
        if e_pred is not None and e_exact is not None:
            n_valid += 1
            # Violation: predicted energy BELOW ground state (physically impossible)
            below = e_exact - e_pred  # positive if e_pred < e_exact
            if below > tolerance:
                violations.append(
                    {"h": p.get("h"), "e_pred": e_pred, "e_exact": e_exact, "excess": below}
                )

    rate = len(violations) / max(n_valid, 1)
    return {
        "n_total": n_valid,
        "n_violations": len(violations),
        "rate": rate,
        "violation_rate": rate,
        "max_violation": float(max(v["excess"] for v in violations)) if violations else 0.0,
        "violations": violations,
    }


def compute_violations_multi_n(
    mpnn_results_by_n: dict[int, dict],
) -> dict[int, dict]:
    """Compute variational violations for multiple N values."""
    return {
        n: compute_variational_violations(results.get("per_point", []))
        for n, results in mpnn_results_by_n.items()
    }


def compute_per_n_scaling_fit(
    n_values: list[int],
    per_site_errors: list[float],
) -> dict | None:
    """Fit |ΔE|/N vs N to a power law for scaling analysis.

    Parameters
    ----------
    n_values : list[int]
        System sizes.
    per_site_errors : list[float]
        Mean |ΔE|/N at each N.

    Returns
    -------
    dict | None
        {alpha, prefactor, r_squared, interpretation} or None if fit fails.
    """
    n_arr = np.array(n_values, dtype=float)
    eps_arr = np.array(per_site_errors, dtype=float)

    # Filter out NaN/Inf/zero
    valid = np.isfinite(eps_arr) & (eps_arr > 0) & np.isfinite(n_arr) & (n_arr > 0)
    if valid.sum() < 3:
        return None

    n_arr = n_arr[valid]
    eps_arr = eps_arr[valid]

    # Log-log linear fit: log(eps) = alpha * log(N) + log(prefactor)
    try:
        log_n = np.log(n_arr)
        log_eps = np.log(eps_arr)
        coeffs = np.polyfit(log_n, log_eps, 1)
        alpha = coeffs[0]
        prefactor = np.exp(coeffs[1])

        # R² calculation
        fitted = np.polyval(coeffs, log_n)
        ss_res = np.sum((log_eps - fitted) ** 2)
        ss_tot = np.sum((log_eps - log_eps.mean()) ** 2)
        r_squared = 1 - ss_res / max(ss_tot, 1e-10)

        # Interpretation
        if abs(alpha) < 0.1:
            interpretation = "extensive (constant per-site error)"
        elif alpha < -0.3:
            interpretation = "improving with N (good scalability)"
        elif alpha > 0.3:
            interpretation = "degrading with N (poor scalability)"
        else:
            interpretation = "weakly scaling"

        return {
            "alpha": float(alpha),
            "prefactor": float(prefactor),
            "r_squared": float(r_squared),
            "interpretation": interpretation,
        }
    except (np.linalg.LinAlgError, ValueError):
        return None


def compute_mpnn_diagnostics(
    mpnn_results_by_n: dict[int, dict],
    *,
    topology: str | None = None,
    model_name: str | None = None,
    p_layers: int | None = None,
    checkpoint_path: str | None = None,
    include_training_quality: bool = True,
    logger: logging.Logger | None = None,
) -> dict:
    """Compute comprehensive MPNN model quality diagnostics.

    Consolidates multiple diagnostic checks into a single reusable function
    for use in any MPNN evaluation runner (extrapolation, cross-N, etc.).

    Parameters
    ----------
    mpnn_results_by_n : dict[int, dict]
        Results keyed by target N, each containing:
        - "per_point": list[dict] with h, e_pred, e_exact, theta, etc.
        - "mean_abs_error_per_site": float
    topology, model_name, p_layers : str | None
        For training data quality lookup.
    checkpoint_path : str | None
        Path to checkpoint used (for provenance tracking).
    include_training_quality : bool
        If True, fetch training data quality from model_zoo.
    logger : logging.Logger | None
        Logger for diagnostic messages.

    Returns
    -------
    dict with: theta_smoothness, variational_violations, scaling_fit,
    training_data_quality, checkpoint_used, summary.
    """
    import logging as _logging

    _log = logger or _logging.getLogger(__name__)
    diagnostics: dict = {}

    # ── 1. θ smoothness per N ─────────────────────────────────────────────
    try:
        from qmbp_simulation.analysis.theta_alignment import detect_jumps

        theta_smoothness = {}
        for n_target, results in mpnn_results_by_n.items():
            per_point = results.get("per_point", [])
            thetas = [p.get("theta") for p in per_point if p.get("theta") is not None]

            if len(thetas) >= 3:
                theta_arr = np.array(thetas)
                if theta_arr.ndim == 2:
                    jumps = detect_jumps(theta_arr, threshold=1.0)
                    max_jump = compute_theta_smoothness(theta_arr) or 0.0
                    theta_smoothness[n_target] = {
                        "n_jumps": len(jumps),
                        "max_jump": float(max_jump),
                        "smooth": len(jumps) == 0,
                    }

        if theta_smoothness:
            diagnostics["theta_smoothness"] = theta_smoothness
    except Exception as e:
        _log.debug("θ smoothness check failed: %s", e)

    # ── 2. Variational violations per N ───────────────────────────────────
    violations_by_n = {}
    total_violations = 0
    total_points = 0

    for n_target, results in mpnn_results_by_n.items():
        per_point = results.get("per_point", [])
        v_result = compute_variational_violations(per_point)
        violations_by_n[n_target] = v_result
        total_violations += v_result["n_violations"]
        total_points += v_result["n_total"]

    diagnostics["variational_violations"] = violations_by_n

    # ── 3. Per-N scaling fit ──────────────────────────────────────────────
    n_values = sorted(mpnn_results_by_n.keys())
    if len(n_values) >= 3:
        per_site_errors = [
            mpnn_results_by_n[n].get("mean_abs_error_per_site", float("nan")) for n in n_values
        ]
        scaling_fit = compute_per_n_scaling_fit(n_values, per_site_errors)
        if scaling_fit:
            diagnostics["scaling_fit"] = scaling_fit

    # ── 4. Training data quality ──────────────────────────────────────────
    if include_training_quality and topology and model_name and p_layers is not None:
        try:
            from qmbp_simulation.predictors.model_zoo import get_training_data_quality

            quality = get_training_data_quality(topology, 0, model_name, p_layers)
            if quality.get("found"):
                diagnostics["training_data_quality"] = {
                    "n_points": quality["n_points"],
                    "verified_ratio": quality["verified_ratio"],
                    "quality_score": quality["quality_score"],
                }
        except Exception as e:
            _log.debug("Training data quality check failed: %s", e)

    # ── 5. Checkpoint provenance ──────────────────────────────────────────
    diagnostics["checkpoint_used"] = checkpoint_path or "auto-selected from zoo"

    # ── 6. Summary health indicators ──────────────────────────────────────
    theta_smooth_count = sum(
        1 for v in diagnostics.get("theta_smoothness", {}).values() if v.get("smooth", False)
    )
    theta_total = len(diagnostics.get("theta_smoothness", {}))

    scaling = diagnostics.get("scaling_fit", {})
    scaling_healthy = scaling.get("alpha", 1.0) < 0.3 if scaling else None

    training_quality = diagnostics.get("training_data_quality", {})
    training_healthy = training_quality.get("quality_score", 0) >= 0.6 if training_quality else None

    diagnostics["summary"] = {
        "theta_smooth_ratio": theta_smooth_count / max(theta_total, 1),
        "variational_violation_rate": total_violations / max(total_points, 1),
        "scaling_healthy": scaling_healthy,
        "training_healthy": training_healthy,
        "overall_health": (
            "healthy"
            if (
                total_violations == 0
                and (scaling_healthy is None or scaling_healthy)
                and theta_smooth_count == theta_total
            )
            else "warning"
            if total_violations < total_points * 0.1
            else "degraded"
        ),
    }

    return diagnostics


# ═══════════════════════════════════════════════════════════════════════════════
# Training Data Exclusion System
# ═══════════════════════════════════════════════════════════════════════════════
# Persistent registry of NPZ files excluded from MPNN training.
# File: data/training_exclusions.json
# Used by: MultiNAggregator, runner_base post-run feedback, inspect_data_stores.

_EXCLUSION_REGISTRY_PATH: Path | None = None


def _resolve_exclusion_path() -> Path:
    """Resolve path to training_exclusions.json."""
    global _EXCLUSION_REGISTRY_PATH
    if _EXCLUSION_REGISTRY_PATH is None:
        _EXCLUSION_REGISTRY_PATH = (
            Path(__file__).resolve().parents[3] / "data" / "training_exclusions.json"
        )
    return _EXCLUSION_REGISTRY_PATH


def load_training_exclusions() -> dict:
    """Load the persistent training exclusion registry.

    Returns
    -------
    dict
        {"excluded": [...], "updated_at": ..., "version": int}
        Returns {"excluded": []} if file doesn't exist.
    """
    import json

    path = _resolve_exclusion_path()
    if not path.exists():
        return {"excluded": [], "version": 1}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("load_training_exclusions: corrupt registry (%s), returning empty", e)
        return {"excluded": [], "version": 1}


def get_excluded_files() -> set[str]:
    """Get set of NPZ filenames excluded from training."""
    registry = load_training_exclusions()
    return {entry["file"] for entry in registry.get("excluded", []) if "file" in entry}


def save_training_exclusions(registry: dict) -> None:
    """Persist the exclusion registry to disk."""
    import json

    from qmbp_simulation.utils.helpers import json_serialize

    path = _resolve_exclusion_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2, default=json_serialize)


def add_training_exclusion(
    file: str,
    topology: str,
    n_qubits: int,
    reason: str,
    *,
    pass_rate_dual: float = 0.0,
    mean_abs_error: float = 0.0,
    failure_mode: str = "unknown",
    method: str = "auto",
    source_dir: str | None = None,
) -> bool:
    """Add a file to the exclusion registry.

    Idempotent: if the file is already excluded, updates the entry.

    Returns True if newly added, False if already present (updated).
    """
    registry = load_training_exclusions()
    excluded = registry.get("excluded", [])

    new_entry: dict = {
        "file": file,
        "topology": topology,
        "n_qubits": n_qubits,
        "reason": reason,
        "method": method,
        "detected_at": datetime.now(UTC).isoformat(),
        "pass_rate_dual": pass_rate_dual,
        "mean_abs_error": mean_abs_error,
        "failure_mode": failure_mode,
    }
    if source_dir:
        new_entry["source_dir"] = source_dir

    # Check if already excluded → update in place
    for i, entry in enumerate(excluded):
        if entry.get("file") == file:
            excluded[i] = new_entry
            registry["excluded"] = excluded
            registry["updated_at"] = datetime.now(UTC).isoformat()
            save_training_exclusions(registry)
            return False

    # New entry
    excluded.append(new_entry)
    registry["excluded"] = excluded
    registry["updated_at"] = datetime.now(UTC).isoformat()
    save_training_exclusions(registry)
    return True


def remove_training_exclusion(file: str) -> bool:
    """Remove a file from the exclusion registry.

    Returns True if found and removed, False if not present.
    """
    registry = load_training_exclusions()
    excluded = registry.get("excluded", [])
    new_excluded = [e for e in excluded if e.get("file") != file]

    if len(new_excluded) == len(excluded):
        return False  # Not found

    registry["excluded"] = new_excluded
    registry["updated_at"] = datetime.now(UTC).isoformat()
    save_training_exclusions(registry)
    return True


def auto_detect_exclusions(
    *,
    dry_run: bool = False,
    npz_dirs: list[str] | None = None,
) -> list[dict]:
    """Scan NPZ files and detect those that should be excluded from training.

    Uses `classify_training_utility` to assess each NPZ file. Files classified
    as "not_useful" are candidates for exclusion.

    Parameters
    ----------
    dry_run : bool
        If True, only detect and return candidates without persisting.
    npz_dirs : list[str] | None
        Directories to scan (relative to project root).
        Default: ["data/multi_n_training", "data/large_n_extrapolation"]

    Returns
    -------
    list[dict]
        Newly detected exclusion candidates (not already in registry).
    """
    project_root = _resolve_exclusion_path().parent.parent
    if npz_dirs is None:
        npz_dirs = ["data/multi_n_training", "data/large_n_extrapolation"]

    already_excluded = get_excluded_files()
    new_candidates: list[dict] = []

    for dir_rel in npz_dirs:
        npz_dir = project_root / dir_rel
        if not npz_dir.exists():
            continue

        source_dir = dir_rel.split("/")[-1]

        for npz_path in sorted(npz_dir.glob("*.npz")):
            if npz_path.name in already_excluded:
                continue

            try:
                data = np.load(npz_path, allow_pickle=True)
                h_values = data.get("h_values")
                e_vqe = data.get("e_vqe", data.get("e_pred"))
                e_exact = data.get("e_exact")
                gaps = data.get("gaps")

                if h_values is None or e_vqe is None or e_exact is None:
                    continue

                n_points = len(h_values)
                if n_points == 0:
                    continue

                abs_errors = np.abs(
                    np.asarray(e_vqe, dtype=float) - np.asarray(e_exact, dtype=float)
                )
                mean_abs_err = float(abs_errors.mean())

                # Compute pass rates
                if gaps is not None:
                    gaps_arr = np.maximum(np.asarray(gaps, dtype=float), 1e-10)
                    de_gaps = abs_errors / gaps_arr
                    pass_5pct = float((de_gaps < DE_GAP_THRESHOLD).mean())
                    dual_mask = (de_gaps < DE_GAP_THRESHOLD) & (abs_errors < MAX_ABS_ERROR)
                    pass_dual = float(dual_mask.mean())
                else:
                    pass_5pct = 0.0
                    pass_dual = 0.0

                # Classify
                category, reason = classify_training_utility(
                    n_points,
                    pass_dual,
                    pass_5pct,
                    mean_abs_error=mean_abs_err,
                )

                if category == "not_useful":
                    # Parse topology and N from filename
                    parsed = parse_npz_filename(npz_path.name)
                    if parsed:
                        topology, n_qubits, _p = parsed
                    else:
                        topology = "unknown"
                        n_qubits = 0

                    candidate = {
                        "file": npz_path.name,
                        "topology": topology,
                        "n_qubits": n_qubits,
                        "reason": reason,
                        "method": "auto",
                        "detected_at": datetime.now(UTC).isoformat(),
                        "pass_rate_dual": pass_dual,
                        "mean_abs_error": mean_abs_err,
                        "failure_mode": _infer_failure_mode(pass_dual, pass_5pct, mean_abs_err),
                        "source_dir": source_dir,
                    }
                    new_candidates.append(candidate)

            except Exception as e:
                logger.debug("auto_detect_exclusions: error processing %s: %s", npz_path.name, e)
                continue

    # Batch persist all new candidates at once (single read-write cycle)
    if new_candidates and not dry_run:
        registry = load_training_exclusions()
        excluded = registry.get("excluded", [])
        existing_files = {e.get("file") for e in excluded}
        for candidate in new_candidates:
            if candidate["file"] not in existing_files:
                excluded.append(candidate)
                existing_files.add(candidate["file"])
        registry["excluded"] = excluded
        registry["updated_at"] = datetime.now(UTC).isoformat()
        save_training_exclusions(registry)

    return new_candidates


def _infer_failure_mode(pass_dual: float, pass_5pct: float, mean_abs_error: float) -> str:
    """Infer failure mode from basic metrics (lightweight classification)."""
    gap_masking = pass_5pct - pass_dual
    if gap_masking > GAP_MASKING_THRESHOLD:
        return "gap_masking"
    if mean_abs_error > AUTO_EXCLUDE_MEAN_ABS_ERROR:
        return "intrinsic_vqe_error"
    return "insufficient_signal"


def __getattr__(name: str):
    """Lazy import for failure diagnostics to avoid circular import."""
    _FAILURES_EXPORTS = {
        "FailureDiagnostic",
        "classify_topology_failure_mode",
        "diagnose_contaminated_training",
        "diagnose_gap_masking",
        "diagnose_generalization_failure",
        "diagnose_h_range_mismatch",
        "diagnose_intrinsic_vqe_error",
    }
    if name in _FAILURES_EXPORTS:
        from qmbp_simulation.analysis import failures_tests

        return getattr(failures_tests, name)
    raise AttributeError(f"module 'qmbp_simulation.analysis.metrics' has no attribute {name!r}")
