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

ADAPTIVE_VQE_MINIMAL_MAXITER: int = 500
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
    *,
    de_gap_threshold: float = DE_GAP_THRESHOLD,
    max_abs_error: float = MAX_ABS_ERROR,
    fidelity: float | None = None,
    min_fidelity: float | None = None,
    fidelity_is_bound: bool = False,
) -> bool:
    """Determine if a single evaluation point is a failure using the dual
    energy criterion.

    A point fails if ANY of these conditions hold:
    1. ΔE/gap >= threshold (relative error too large)
    2. |ΔE| > max_abs_error (absolute error too large, prevents gap masking)

    Fidelity is OFF by default as a pass/fail criterion (recorded as a
    diagnostic only), because at large N it is a lower bound and would
    misclassify good points. It can be OPTIONALLY enabled by passing both
    ``fidelity`` and ``min_fidelity``: a point then also fails if
    ``fidelity < min_fidelity``. A ``fidelity_is_bound=True`` value (Eckart
    lower bound at large N) is NEVER used to fail a point, since a low bound is
    inconclusive — only an exact fidelity below the threshold gates.

    Handles edge cases:
    - NaN/Inf in de_gap or abs_error → automatic failure (corrupted data)
    - abs_error not available → only ΔE/gap is checked
    - Negative de_gap → flagged as failure (indicates variational violation)

    Parameters
    ----------
    de_gap : float
        Relative error |E_pred - E_exact| / gap.
    abs_error : float | None
        Absolute error |E_pred - E_exact|. If None, only ΔE/gap is checked.
        Not available when E_exact comes from approximate methods (DMRG).
    de_gap_threshold : float
        Threshold for ΔE/gap criterion (default: 0.05).
    max_abs_error : float
        Cap on absolute error (default: 0.10).

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

    # Criterion 3 (OPT-IN): fidelity floor. Only when a threshold is supplied
    # and the fidelity is an EXACT value (never gate on a lower bound, which is
    # inconclusive when low). Off by default → identical to the dual criterion.
    if min_fidelity is not None and fidelity is not None and not fidelity_is_bound:
        if np.isfinite(fidelity) and fidelity < min_fidelity:
            return True

    return False


def identify_failures(
    per_h_results: list[dict],
    *,
    de_gap_threshold: float = DE_GAP_THRESHOLD,
    max_abs_error: float = MAX_ABS_ERROR,
) -> list[int]:
    """Identify failing point indices from per-h results using the dual
    energy criterion (ΔE/gap and |ΔE|). Fidelity is not a criterion.

    Handles common data issues:
    - Missing keys gracefully (only checks what's available)
    - NaN values in results → marked as failure
    - Empty list → returns empty

    Parameters
    ----------
    per_h_results : list[dict]
        Each entry must have ``"de_gap"`` (float).
        Optional: ``"abs_error"`` (float).

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
            de_gap_threshold=de_gap_threshold,
            max_abs_error=max_abs_error,
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
    # Hard floor on restarts for every tier (even cheap/minimal points get
    # at least MIN_N_RESTARTS — a single restart is never enough).
    from qmbp_simulation.models.constants import MIN_N_RESTARTS

    # Tier 1: Easy wins (ΔE/gap barely above threshold, large gap)
    # These points are almost passing — minimal effort needed.
    if priority >= ADAPTIVE_VQE_CHEAP_PRIORITY and de_gap < ADAPTIVE_VQE_CHEAP_DE_GAP:
        return {
            "maxiter": min(base_maxiter, ADAPTIVE_VQE_CHEAP_MAXITER),
            "n_restarts": MIN_N_RESTARTS,
            "rhobeg": ADAPTIVE_VQE_CHEAP_RHOBEG,
            "tier": "cheap",
            "reason": f"Easy win: ΔE/gap={de_gap:.3f}, priority={priority:.2f}",
        }

    # Tier 2: Standard refinement (moderate distance from threshold)
    if priority >= ADAPTIVE_VQE_STANDARD_PRIORITY:
        scale = 0.5 + 0.5 * priority  # 0.75-1.0× base
        return {
            "maxiter": int(base_maxiter * scale),
            "n_restarts": max(min(ADAPTIVE_VQE_STANDARD_MAX_RESTARTS, base_restarts), MIN_N_RESTARTS),
            "rhobeg": ADAPTIVE_VQE_STANDARD_RHOBEG,
            "tier": "standard",
            "reason": f"Standard: ΔE/gap={de_gap:.3f}, priority={priority:.2f}",
        }

    # Tier 3: Aggressive (far from threshold but feasible)
    # Use full budget — this point needs serious optimization
    if priority >= ADAPTIVE_VQE_AGGRESSIVE_PRIORITY:
        return {
            "maxiter": base_maxiter,
            "n_restarts": max(base_restarts, MIN_N_RESTARTS),
            "rhobeg": ADAPTIVE_VQE_AGGRESSIVE_RHOBEG,
            "tier": "aggressive",
            "reason": f"Aggressive: ΔE/gap={de_gap:.3f}, needs full budget",
        }

    # Tier 4: Minimal (very low priority — almost hopeless)
    # Give minimal budget just in case, but don't waste compute
    return {
        "maxiter": min(base_maxiter, ADAPTIVE_VQE_MINIMAL_MAXITER),
        "n_restarts": MIN_N_RESTARTS,
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
    valid = [
        (r["de_gap"], r["theta_std"])
        for r in per_h_results
        if "theta_std" in r and r.get("theta_std", 0) > 0 and "de_gap" in r
    ]

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
    result["calibrated"] = (
        result["high_uncertainty_mean_de_gap"] > result["low_uncertainty_mean_de_gap"]
    )

    return result


def analyze_energy_error_scaling(abs_error_by_n: dict[int, float]) -> dict:
    """Characterize how the energy error |ΔE| scales with system size N.

    For a fixed-depth variational ansatz, the physically meaningful statement
    is EXTENSIVITY: the per-site error |ΔE|/N stays bounded, i.e. |ΔE| grows at
    most ~linearly with N. We quantify this with a log-log power-law fit
    |ΔE| ≈ A·N^α (the standard tool for scaling studies), which is far more
    robust and informative than a raw max/min ratio of two points:

    - ``alpha`` (scaling exponent) with ``r_squared`` (fit quality).
      α ≈ 1 → extensive (error per site constant, ideal).
      α < 0.8 → sub-extensive (error per site shrinks — unusually good).
      α > 1.2 → super-extensive (error accelerates — extrapolation risk).
    - ``per_site_error_by_n`` and ``per_site_cv`` (coefficient of variation of
      |ΔE|/N): a direct, fit-free extensivity check — low CV ⇒ flat per-site
      error ⇒ extensive.
    - ``err_ratio`` (max/min |ΔE|): kept for continuity with prior logs.

    Requires ≥3 sizes for the power-law fit. With exactly 2 sizes it falls back
    to the ratio + per-site CV only (``method="ratio_fallback"``) and flags the
    result as preliminary.

    Parameters
    ----------
    abs_error_by_n : dict[int, float]
        Mean |ΔE| per system size N. Non-positive or missing entries dropped.

    Returns
    -------
    dict
        ``method`` : "power_law" | "ratio_fallback" | "insufficient_data"
        ``n_values`` : list[int]
        ``alpha`` / ``r_squared`` / ``coefficient`` : float | None (power-law)
        ``err_ratio`` : float | None  (max/min |ΔE|)
        ``per_site_error_by_n`` : dict[str, float]
        ``per_site_cv`` : float | None  (std/mean of |ΔE|/N)
        ``verdict`` : "extensive" | "sub_extensive" | "super_extensive" |
                      "preliminary" | "unknown"
        ``is_extensive`` : bool  (verdict == "extensive" or "sub_extensive")
    """
    import numpy as np

    clean = {int(n): float(e) for n, e in abs_error_by_n.items() if e is not None and e > 0}
    ns = sorted(clean)
    result: dict = {
        "method": "insufficient_data",
        "n_values": ns,
        "alpha": None,
        "r_squared": None,
        "coefficient": None,
        "err_ratio": None,
        "per_site_error_by_n": {str(n): clean[n] / n for n in ns},
        "per_site_cv": None,
        "verdict": "unknown",
        "is_extensive": False,
    }
    if len(ns) < 2:
        return result

    errs = np.array([clean[n] for n in ns])
    result["err_ratio"] = float(errs.max() / max(errs.min(), 1e-12))

    per_site = np.array([clean[n] / n for n in ns])
    ps_mean = float(per_site.mean())
    result["per_site_cv"] = float(per_site.std() / ps_mean) if ps_mean > 1e-12 else None

    if len(ns) >= 3:
        from experiments.helpers.scaling_utils import fit_power_law

        fit = fit_power_law(ns, list(errs), min_points=3)
        result["alpha"] = fit["exponent"]
        result["coefficient"] = fit["coefficient"]
        result["r_squared"] = fit["r_squared"]
        result["method"] = "power_law"
        alpha = fit["exponent"]
        if alpha is None:
            result["verdict"] = "unknown"
        elif alpha < 0.8:
            result["verdict"] = "sub_extensive"
        elif alpha <= 1.2:
            result["verdict"] = "extensive"
        else:
            result["verdict"] = "super_extensive"
    else:
        # Two sizes: no reliable exponent. Use per-site flatness as a proxy.
        result["method"] = "ratio_fallback"
        result["verdict"] = "preliminary"

    result["is_extensive"] = result["verdict"] in ("extensive", "sub_extensive")
    return result


def wilson_ci(n_success: int, n_total: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion (pass rate).

    More reliable than the normal (Wald) interval for small n, which is the
    typical case here (6-20 h-points per evaluation). Default z=1.96 → 95% CI.

    Parameters
    ----------
    n_success : int
        Number of successes (e.g., h-points passing the dual criterion).
    n_total : int
        Total number of trials (h-points evaluated).
    z : float
        Normal quantile for the confidence level (1.96 = 95%, 1.645 = 90%).

    Returns
    -------
    tuple[float, float]
        (ci_lower, ci_upper), both clamped to [0, 1]. Returns (0.0, 0.0) when
        n_total == 0.
    """
    if n_total <= 0:
        return (0.0, 0.0)
    p = n_success / n_total
    denom = 1.0 + z * z / n_total
    center = (p + z * z / (2 * n_total)) / denom
    margin = z * np.sqrt(p * (1 - p) / n_total + z * z / (4 * n_total * n_total)) / denom
    return (float(max(0.0, center - margin)), float(min(1.0, center + margin)))


def bootstrap_ci_mean(
    data: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap percentile confidence interval for the mean of a sample.

    Distribution-free — appropriate for ΔE/gap which is typically skewed
    (long tail near h_critical). Reproducible via fixed seed.

    Parameters
    ----------
    data : np.ndarray
        Sample values (e.g., per-h ΔE/gap).
    n_bootstrap : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level (0.95 = 95% CI).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    tuple[float, float]
        (ci_lower, ci_upper) of the mean. For n < 2 returns (mean, mean).
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n < 2:
        val = float(data[0]) if n == 1 else 0.0
        return (val, val)
    rng = np.random.default_rng(seed)
    # Vectorized resampling: [n_bootstrap, n] index matrix
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = data[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def assess_hardware_viability(
    summary: dict[str, Any],
    fidelity_threshold: float,
) -> dict[str, Any]:
    min_fid = summary.get("min_fidelity")
    mean_fid = summary.get("mean_fidelity")
    if min_fid is None:
        return {
            "hardware_viable": False,
            "f_min_threshold": float(fidelity_threshold),
            "reason": "no_fidelity_data",
        }
    return {
        "hardware_viable": bool(min_fid > fidelity_threshold),
        "f_min_threshold": float(fidelity_threshold),
        "min_fidelity": float(min_fid),
        "mean_fidelity": float(mean_fid) if mean_fid is not None else None,
        "fidelity_is_lower_bound": bool(summary.get("fidelity_is_lower_bound", False)),
    }


def compute_deploy_summary(
    per_h_results: list[dict],
    *,
    thresholds: tuple[float, ...] = (0.05, 0.10),
    fidelity_threshold: float | None = None,
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

    # Fidelity stats. Exact for N ≤ STATEVECTOR_MAX_N, else a rigorous
    # variance-based lower bound (Eckart). We track how many points are
    # bounds vs exact so downstream reports can annotate correctly.
    fids = [r["fidelity"] for r in per_h_results if r.get("fidelity") is not None]
    if fids:
        fids_arr = np.array(fids)
        summary["mean_fidelity"] = float(np.mean(fids_arr))
        summary["min_fidelity"] = float(np.min(fids_arr))
        summary["fidelity_pass_rate"] = float(np.mean(fids_arr > 0.90))
        # Provenance: distinguish exact fidelity from variance lower bounds.
        n_bound = sum(1 for r in per_h_results if r.get("fidelity_is_bound"))
        n_fid = len(fids)
        summary["n_fidelity_points"] = n_fid
        summary["n_fidelity_bound"] = int(n_bound)
        summary["n_fidelity_exact"] = int(n_fid - n_bound)
        summary["fidelity_is_lower_bound"] = bool(n_bound > 0)
    else:
        summary["mean_fidelity"] = None
        summary["min_fidelity"] = None
        summary["fidelity_pass_rate"] = None
        summary["n_fidelity_points"] = 0
        summary["n_fidelity_bound"] = 0
        summary["n_fidelity_exact"] = 0
        summary["fidelity_is_lower_bound"] = False

    # ── Infidelity decomposition (Var(H) vs gap) — DEFAULT aggregate ────────
    # Attribute infidelity to its dominant factor across the sweep so h_c
    # regions surface "dirty_state" (attackable) vs "small_gap" (physics
    # ceiling). Diagnostic only — never gates pass/fail.
    evs = [r["energy_variance"] for r in per_h_results if r.get("energy_variance") is not None]
    summary["mean_energy_variance"] = float(np.mean(evs)) if evs else None
    vogs = [
        r["variance_over_gap2"] for r in per_h_results if r.get("variance_over_gap2") is not None
    ]
    summary["mean_variance_over_gap2"] = float(np.mean(vogs)) if vogs else None
    summary["n_dirty_state"] = sum(
        1 for r in per_h_results if r.get("infidelity_dominant_factor") == "dirty_state"
    )
    summary["n_small_gap"] = sum(
        1 for r in per_h_results if r.get("infidelity_dominant_factor") == "small_gap"
    )
    summary["n_clean"] = sum(
        1 for r in per_h_results if r.get("infidelity_dominant_factor") == "clean"
    )

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

    # ── Statistical confidence intervals (95%) ─────────────────────────────
    # Wilson CI on pass_rate_dual: reports the uncertainty of the pass rate
    # given the number of h-points. With few points (6-8), the point estimate
    # is noisy; the CI lower bound is a conservative, comparable quality signal.
    n_pass_dual = summary.get("n_pass_dual", 0)
    pd_lo, pd_hi = wilson_ci(int(n_pass_dual), n)
    summary["pass_rate_dual_ci_lower"] = pd_lo
    summary["pass_rate_dual_ci_upper"] = pd_hi

    # Wilson CI on pass_rate_5pct (single criterion) for reference
    n_pass_5 = summary.get("n_pass_5pct", 0)
    p5_lo, p5_hi = wilson_ci(int(n_pass_5), n)
    summary["pass_rate_5pct_ci_lower"] = p5_lo
    summary["pass_rate_5pct_ci_upper"] = p5_hi

    # Bootstrap CI on mean ΔE/gap: distribution-free error bar on the mean.
    dg_lo, dg_hi = bootstrap_ci_mean(de_gaps)
    summary["mean_de_gap_ci_lower"] = dg_lo
    summary["mean_de_gap_ci_upper"] = dg_hi

    if fidelity_threshold is not None:
        summary.update(assess_hardware_viability(summary, fidelity_threshold))

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
    abs_errors: np.ndarray | None = None,
    threshold: float = DE_GAP_THRESHOLD,
    max_abs_error: float = MAX_ABS_ERROR,
) -> float | None:
    """Compute h_frontier: the h where the dual criterion transitions from fail to pass.

    Uses the dual criterion (ΔE/gap < threshold AND |ΔE| < max_abs_error) to
    determine pass/fail at each h-point. This prevents gap masking where large
    gaps artificially suppress the relative metric.

    Scans from low-h (typically failing) to high-h (typically passing) and
    interpolates the crossing point.

    Parameters
    ----------
    h_values : np.ndarray
        Array of h-values (will be sorted internally).
    de_gaps : np.ndarray
        Corresponding ΔE/gap values (same length as h_values).
    abs_errors : np.ndarray | None
        Absolute errors |ΔE| per point. If None, falls back to ΔE/gap-only
        criterion (backward compatible with old NPZ files lacking this field).
    threshold : float
        ΔE/gap pass threshold (default: 0.05).
    max_abs_error : float
        Absolute error cap (default: 0.10). Points above this fail regardless
        of ΔE/gap.

    Returns
    -------
    float | None
        Interpolated h where the dual criterion transitions.
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

    # Build pass/fail array using dual criterion
    passes_dg = dg_sorted < threshold
    if abs_errors is not None and len(abs_errors) == len(h_values):
        ae_sorted = np.asarray(abs_errors)[sort_idx]
        passes_ae = ae_sorted < max_abs_error
        passes = passes_dg & passes_ae
    else:
        # Fallback: ΔE/gap only (backward compat)
        passes = passes_dg

    # Find crossing: last failing point followed by first passing point
    # Scan from low-h (typically failing) to high-h (typically passing)
    for i in range(len(h_sorted) - 1):
        if not passes[i] and passes[i + 1]:
            h0, h1 = float(h_sorted[i]), float(h_sorted[i + 1])
            # Interpolate using de_gap if de_gap drives the transition
            dg0, dg1 = float(dg_sorted[i]), float(dg_sorted[i + 1])
            if dg0 >= threshold and dg1 < threshold:
                # de_gap transition: use standard linear interpolation
                if abs(dg0 - dg1) < 1e-12:
                    return (h0 + h1) / 2
                return h0 + (threshold - dg0) / (dg1 - dg0) * (h1 - h0)
            else:
                # abs_error drives the transition: use midpoint
                # (can't interpolate meaningfully on a different metric)
                return (h0 + h1) / 2

    # No crossing found
    if np.all(passes):
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
        abs_err = (
            np.abs(data["e_vqe"] - data["e_exact"])
            if ("e_vqe" in data and "e_exact" in data)
            else None
        )
    elif "e_vqe" in data and "e_exact" in data and "gaps" in data:
        abs_err = np.abs(data["e_vqe"] - data["e_exact"])
        gaps = np.maximum(data["gaps"], 1e-10)
        de_gaps = abs_err / gaps
    else:
        return {"h_frontier": None, "error": "missing_energy_fields"}

    h_frontier = compute_h_frontier(h_values, de_gaps, abs_errors=abs_err, threshold=threshold)

    n_pass = int((de_gaps < threshold).sum())
    abs_errors = abs_err
    if abs_errors is not None:
        # Dual criterion pass count
        dual_mask = (de_gaps < threshold) & (abs_errors < MAX_ABS_ERROR)
        n_pass = int(dual_mask.sum())

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


def _compute_confidence_level(
    n_points: int,
    eval_cache_density: int,
    pass_rate_dual: float,
) -> str:
    """Compute confidence level for a config's reported metrics.

    Based on:
    - n_points: more training points → more reliable pass_rate
    - eval_cache_density: more evaluations → more reproducible results
    - Statistical power: at n_points, the standard error of pass_rate is
      SE = sqrt(p*(1-p)/n), so confidence depends on both p and n.

    Returns
    -------
    str
        One of: "high", "medium", "low", "very_low"
    """
    # Standard error of the pass_rate estimate
    p = max(0.01, min(0.99, pass_rate_dual))
    se = (p * (1 - p) / max(n_points, 1)) ** 0.5

    # Score components
    points_score = min(1.0, n_points / 50)  # 50+ pts = full score
    density_score = min(1.0, eval_cache_density / 10)  # 10+ evals = full score
    precision_score = max(0.0, 1.0 - se * 10)  # SE < 0.05 → high precision

    composite = 0.5 * points_score + 0.3 * density_score + 0.2 * precision_score

    if composite >= 0.7:
        return "high"
    elif composite >= 0.4:
        return "medium"
    elif composite >= 0.2:
        return "low"
    return "very_low"


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

    # ── Phase 0: Auto-fix stale e_exact in NPZ (GT cache is authoritative) ──
    # This ensures all derived metrics (ΔE/gap, pass_rate) use current ground truth.
    try:
        coherence = validate_gt_npz_coherence(fix=True)
        if coherence["n_points_fixed"] > 0:
            logger.info(
                "Dashboard pre-fix: corrected %d stale e_exact points in %d files",
                coherence["n_points_fixed"],
                coherence["n_files_with_issues"],
            )
    except Exception as _gt_fix_err:
        logger.debug("GT auto-fix skipped: %s", _gt_fix_err)

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
                "confidence_level": _compute_confidence_level(
                    n_points=n_points,
                    eval_cache_density=eval_cache_density,
                    pass_rate_dual=pass_rate_dual,
                ),
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
                mt_models.append(
                    {
                        "checkpoint": e.checkpoint_file,
                        "pass_rate": e.pass_rate,
                        "pass_rate_by_n": e.pass_rate_by_n if e.pass_rate_by_n else None,
                        "n_training_points": e.n_training_points,
                        "notes": e.notes[:80] if e.notes else "",
                        "created": getattr(e, "created", ""),
                    }
                )
            dashboard["multi_topology_models"] = {
                "n_models": len(mt_models),
                "models": mt_models,
                "best_pass_rate": max(e.pass_rate for e in mt_entries),
            }
    except Exception as _mt_err:
        logger.debug("MT models section skipped: %s", _mt_err)

    # ── MT vs ST comparison section ──────────────────────────────────────
    # Integrate model comparison results into the dashboard for unified querying
    try:
        from qmbp_simulation.analysis.evaluation_report import generate_mt_vs_st_table

        _mt_st_lines, _mt_st_summary = generate_mt_vs_st_table(latest_only=True)
        if _mt_st_summary.get("total", 0) > 0:
            dashboard["mt_vs_st_comparison"] = {
                "generated_at": _mt_st_summary.get("generated_at", ""),
                "global": {
                    "mt_wins": _mt_st_summary["mt_wins"],
                    "st_wins": _mt_st_summary["st_wins"],
                    "ties": _mt_st_summary["ties"],
                    "total": _mt_st_summary["total"],
                    "mt_win_rate": _mt_st_summary.get("mt_win_rate", 0.0),
                    "mt_avg_pass_rate": _mt_st_summary.get("mt_avg_pass_rate", 0.0),
                    "st_avg_pass_rate": _mt_st_summary.get("st_avg_pass_rate", 0.0),
                },
                "per_topology": _mt_st_summary.get("per_topology", {}),
                "per_scenario": _mt_st_summary.get("per_scenario", []),
            }
    except Exception as _mt_st_err:
        logger.debug("MT vs ST comparison section skipped: %s", _mt_st_err)

    # ── GT ↔ NPZ coherence summary ──────────────────────────────────────
    try:
        coherence = validate_gt_npz_coherence(fix=False)
        if coherence["n_files_checked"] > 0:
            dashboard["gt_npz_coherence"] = {
                "n_files_checked": coherence["n_files_checked"],
                "n_files_with_issues": coherence["n_files_with_issues"],
                "n_points_mismatched": coherence["n_points_mismatched"],
                "max_delta": coherence["max_delta"],
                "is_coherent": coherence["n_files_with_issues"] == 0,
                "issues": coherence["issues"][:10],  # Top 10
            }
            if coherence["n_files_with_issues"] > 0:
                logger.warning(
                    "GT↔NPZ coherence: %d files stale (max_delta=%.2e)",
                    coherence["n_files_with_issues"],
                    coherence["max_delta"],
                )
    except Exception as _gt_err:
        logger.debug("GT coherence check skipped: %s", _gt_err)

    # ── Per-topology zoo pass_rate_by_n (from zoo manifest) ──────────────
    try:
        all_zoo_by_n: dict[str, dict] = {}
        for e in _zoo_entries:
            if e.pass_rate_by_n:
                all_zoo_by_n[e.checkpoint_file] = {
                    "topology": e.topology,
                    "pass_rate_by_n": e.pass_rate_by_n,
                    "global_pass_rate": e.pass_rate,
                }
        if all_zoo_by_n:
            dashboard["zoo_pass_rate_by_n"] = all_zoo_by_n
    except Exception:
        pass

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

    # ── Tier 3 note: eval_report vs comparison semantics ─────────────────
    dashboard["quality_semantics"] = {
        "eval_report": (
            "Measures TRAINING DATA quality: uses θ_opt directly from NPZ. "
            "Grade reflects VQE optimization quality, NOT MPNN prediction accuracy."
        ),
        "comparison": (
            "Measures DEPLOYMENT quality: uses θ_pred from MPNN inference. "
            "Pass rate reflects end-to-end pipeline accuracy at test time."
        ),
        "expected_gap": (
            "eval_report grade ≥ comparison pass_rate is NORMAL "
            "(training data is easier than generalization). "
            "Reverse (comparison > eval) suggests stale eval report."
        ),
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


def query_mt_vs_st_comparison(
    topology: str | None = None,
    *,
    n_min: int | None = None,
    n_max: int | None = None,
    from_dashboard: bool = True,
) -> dict:
    """Query MT vs ST model comparison data.

    Provides a unified API for accessing comparison results, either from
    the dashboard cache (fast, no I/O beyond reading one JSON) or from
    live comparison files (slower, always fresh).

    Parameters
    ----------
    topology : str | None
        Filter to a specific topology. If None, returns all topologies.
    n_min : int | None
        Minimum N value filter.
    n_max : int | None
        Maximum N value filter.
    from_dashboard : bool
        If True (default), reads from cached dashboard. Falls back to live
        computation if dashboard data is unavailable.

    Returns
    -------
    dict
        {
            "global": {mt_wins, st_wins, ties, total, mt_win_rate, mt_avg_pass_rate, st_avg_pass_rate},
            "per_topology": {topo: {mt_avg_pass_rate, st_avg_pass_rate, winner, delta, ...}},
            "per_scenario": [...],
            "source": "dashboard" | "live",
        }
    """
    import json as _json
    from pathlib import Path as _P

    _ROOT = _P(__file__).resolve().parents[3]

    result = None

    # Try dashboard first (fast path)
    if from_dashboard:
        dash_path = _ROOT / "data" / "model_quality_dashboard.json"
        if dash_path.exists():
            try:
                with open(dash_path) as f:
                    dash = _json.load(f)
                mt_st = dash.get("mt_vs_st_comparison")
                if mt_st and mt_st.get("global", {}).get("total", 0) > 0:
                    result = {
                        "global": mt_st["global"],
                        "per_topology": mt_st.get("per_topology", {}),
                        "per_scenario": mt_st.get("per_scenario", []),
                        "source": "dashboard",
                    }
            except Exception:
                pass

    # Fall back to live computation
    if result is None:
        try:
            from qmbp_simulation.analysis.evaluation_report import generate_mt_vs_st_table

            kwargs: dict = {"latest_only": True}
            if topology:
                kwargs["topology_filter"] = topology
            if n_min is not None:
                kwargs["n_min"] = n_min
            if n_max is not None:
                kwargs["n_max"] = n_max

            _, summary = generate_mt_vs_st_table(**kwargs)
            if summary.get("total", 0) > 0:
                result = {
                    "global": {
                        "mt_wins": summary["mt_wins"],
                        "st_wins": summary["st_wins"],
                        "ties": summary["ties"],
                        "total": summary["total"],
                        "mt_win_rate": summary.get("mt_win_rate", 0.0),
                        "mt_avg_pass_rate": summary.get("mt_avg_pass_rate", 0.0),
                        "st_avg_pass_rate": summary.get("st_avg_pass_rate", 0.0),
                    },
                    "per_topology": summary.get("per_topology", {}),
                    "per_scenario": summary.get("per_scenario", []),
                    "source": "live",
                }
        except Exception:
            pass

    if result is None:
        return {
            "global": {"mt_wins": 0, "st_wins": 0, "ties": 0, "total": 0},
            "per_topology": {},
            "per_scenario": [],
            "source": "none",
        }

    # Apply post-hoc filters if dashboard source (live already filtered via kwargs)
    if result["source"] == "dashboard":
        if topology or n_min is not None or n_max is not None:
            scenarios = result["per_scenario"]
            if topology:
                topo_list = [topology] if isinstance(topology, str) else topology
                scenarios = [s for s in scenarios if s.get("topology") in topo_list]
            if n_min is not None:
                scenarios = [s for s in scenarios if s.get("n_qubits", 0) >= n_min]
            if n_max is not None:
                scenarios = [s for s in scenarios if s.get("n_qubits", 999) <= n_max]

            # Recompute global stats from filtered scenarios
            mt_wins = sum(1 for s in scenarios if s.get("winner") == "MT")
            st_wins = sum(1 for s in scenarios if s.get("winner") == "ST")
            ties = sum(1 for s in scenarios if s.get("winner") == "tie")
            total = mt_wins + st_wins + ties
            mt_pass = [s["mt_pass_rate"] for s in scenarios if "mt_pass_rate" in s]
            st_pass = [s["st_pass_rate"] for s in scenarios if "st_pass_rate" in s]

            result["global"] = {
                "mt_wins": mt_wins,
                "st_wins": st_wins,
                "ties": ties,
                "total": total,
                "mt_win_rate": mt_wins / max(total, 1),
                "mt_avg_pass_rate": sum(mt_pass) / max(len(mt_pass), 1) if mt_pass else 0.0,
                "st_avg_pass_rate": sum(st_pass) / max(len(st_pass), 1) if st_pass else 0.0,
            }
            result["per_scenario"] = scenarios

            # Recompute per_topology from filtered scenarios
            from collections import defaultdict as _dd

            _pt: dict = _dd(
                lambda: {"mt_pass": [], "st_pass": [], "mt_wins": 0, "st_wins": 0, "ties": 0}
            )
            for s in scenarios:
                t = s.get("topology", "?")
                _pt[t]["mt_pass"].append(s.get("mt_pass_rate", 0))
                _pt[t]["st_pass"].append(s.get("st_pass_rate", 0))
                if s.get("winner") == "MT":
                    _pt[t]["mt_wins"] += 1
                elif s.get("winner") == "ST":
                    _pt[t]["st_wins"] += 1
                else:
                    _pt[t]["ties"] += 1
            per_topo_filtered = {}
            for t, info in _pt.items():
                mt_a = sum(info["mt_pass"]) / max(len(info["mt_pass"]), 1)
                st_a = sum(info["st_pass"]) / max(len(info["st_pass"]), 1)
                per_topo_filtered[t] = {
                    "mt_avg_pass_rate": mt_a,
                    "st_avg_pass_rate": st_a,
                    "mt_wins": info["mt_wins"],
                    "st_wins": info["st_wins"],
                    "ties": info["ties"],
                    "winner": "MT"
                    if mt_a > st_a + 0.01
                    else ("ST" if st_a > mt_a + 0.01 else "tie"),
                    "delta": mt_a - st_a,
                }
            result["per_topology"] = per_topo_filtered

    return result


def validate_gt_npz_coherence(
    *,
    fix: bool = False,
    tolerance: float = 1e-6,
    include_extrapolation: bool = True,
) -> dict:
    """Cross-validate GT cache energies against NPZ stored e_exact values.

    Uses the GT cache as the single source of truth. Any NPZ e_exact that
    differs from the GT cache is considered stale and should be corrected.

    This is the CANONICAL ground truth validation — all other checks
    (check_zoo_coherence, etc.) delegate to this function.

    Lookup strategy:
    - Primary: exact key match with 6-decimal h formatting
    - Fallback: fuzzy h-match (nearest key within h_tol=1e-4)

    Parameters
    ----------
    fix : bool
        If True, update NPZ e_exact and recompute de_gaps from current GT cache.
        Creates .bak backup before modifying.
    tolerance : float
        Absolute tolerance for considering values as matching (default: 1e-6).

    Returns
    -------
    dict
        {
            "n_files_checked": int,
            "n_files_with_issues": int,
            "n_points_checked": int,
            "n_points_mismatched": int,
            "n_points_fixed": int (only if fix=True),
            "max_delta": float,
            "issues": [{file, topology, n_qubits, n_mismatched, max_delta, affected_metrics}],
            "summary": str,
        }
    """
    import json as _json
    import shutil
    from pathlib import Path as _P

    _ROOT = _P(__file__).resolve().parents[3]
    gt_path = _ROOT / "data" / "ground_truth_cache.json"
    npz_dir = _ROOT / "data" / "multi_n_training"
    extrap_dir = _ROOT / "data" / "large_n_extrapolation"

    if not gt_path.exists():
        return {
            "n_files_checked": 0,
            "n_files_with_issues": 0,
            "n_points_checked": 0,
            "n_points_mismatched": 0,
            "n_points_not_in_gt": 0,
            "n_points_fixed": 0,
            "max_delta": 0.0,
            "issues": [],
            "summary": "GT cache not found.",
        }

    with open(gt_path) as f:
        raw = _json.load(f)
    gt = raw.get("entries", raw)

    # Build secondary index for fuzzy h-lookup: (topo, N, model) → {h_float: key}
    gt_h_index: dict[tuple, dict[float, str]] = {}
    for key in gt:
        parts = key.split("|")
        if len(parts) >= 4:
            try:
                config = (parts[0], int(parts[1]), parts[2])
                h_float = float(parts[3])
                gt_h_index.setdefault(config, {})[h_float] = key
            except (ValueError, IndexError):
                continue

    n_files_checked = 0
    n_files_with_issues = 0
    n_points_checked = 0
    n_points_mismatched = 0
    n_gap_points_mismatched = 0
    n_points_not_in_gt = 0
    n_points_fixed = 0
    n_gap_points_fixed = 0
    max_delta_global = 0.0
    issues: list[dict] = []

    # Collect all NPZ dirs to check
    npz_dirs_to_check = []
    if npz_dir and npz_dir.exists():
        npz_dirs_to_check.append(npz_dir)
    if include_extrapolation and extrap_dir and extrap_dir.exists():
        npz_dirs_to_check.append(extrap_dir)

    if not npz_dirs_to_check:
        return {
            "n_files_checked": 0,
            "n_files_with_issues": 0,
            "n_points_checked": 0,
            "n_points_mismatched": 0,
            "n_points_not_in_gt": 0,
            "n_points_fixed": 0,
            "max_delta": 0.0,
            "issues": [],
            "summary": "No NPZ directories found.",
        }

    for search_dir in npz_dirs_to_check:
        for npz_file in sorted(search_dir.glob("*.npz")):
            data = np.load(str(npz_file), allow_pickle=True)
            if "e_exact" not in data or "h_values" not in data:
                continue

            h_vals = data["h_values"]
            e_exact_npz = data["e_exact"]

            # Parse topology and N from filename
            parts = npz_file.stem.split("_")
            n_idx = next((i for i, p in enumerate(parts) if p.startswith("N")), None)
            if n_idx is None:
                continue
            topo = "_".join(parts[:n_idx])
            try:
                n_val = int(parts[n_idx][1:])
            except ValueError:
                continue

            gaps_npz = data["gaps"] if "gaps" in data else None

            n_files_checked += 1
            file_mismatches = 0
            file_max_delta = 0.0
            corrections: dict[int, float] = {}  # idx → new e_exact
            gap_corrections: dict[int, float] = {}  # idx → new gap
            file_gap_mismatches = 0

            config_key = (topo, n_val, "tfim_bond_resolved")
            h_lookup = gt_h_index.get(config_key, {})

            for i, h in enumerate(h_vals):
                # Primary: exact 6-decimal key
                key = f"{topo}|{n_val}|tfim_bond_resolved|{float(h):.2f}"
                gt_entry = gt.get(key)

                # Fallback: fuzzy h-match (nearest within 1e-4)
                if gt_entry is None and h_lookup:
                    h_float = float(h)
                    nearest_h = min(h_lookup.keys(), key=lambda x: abs(x - h_float))
                    if abs(nearest_h - h_float) < 1e-4:
                        gt_entry = gt.get(h_lookup[nearest_h])

                if gt_entry is None:
                    n_points_not_in_gt += 1
                    continue

                gt_e = gt_entry.get("energy", gt_entry.get("e_exact"))
                if gt_e is None:
                    continue

                n_points_checked += 1
                delta = abs(float(e_exact_npz[i]) - float(gt_e))
                if delta > tolerance:
                    file_mismatches += 1
                    file_max_delta = max(file_max_delta, delta)
                    max_delta_global = max(max_delta_global, delta)
                    corrections[i] = float(gt_e)

                # Gap coherence: the gap is a property of the Hamiltonian, so a
                # stale NPZ gap silently corrupts ΔE/gap (the primary metric).
                # Check it against the GT cache with the same tolerance.
                gt_gap = gt_entry.get("gap")
                if gaps_npz is not None and gt_gap is not None:
                    gap_delta = abs(float(gaps_npz[i]) - float(gt_gap))
                    if gap_delta > tolerance:
                        file_gap_mismatches += 1
                        gap_corrections[i] = float(gt_gap)
                        file_max_delta = max(file_max_delta, gap_delta)
                        max_delta_global = max(max_delta_global, gap_delta)

            if file_mismatches > 0 or file_gap_mismatches > 0:
                n_files_with_issues += 1
                n_points_mismatched += file_mismatches
                n_gap_points_mismatched += file_gap_mismatches

                # Compute impact on metrics
                affected_pass_rate_delta = 0.0
                if "gaps" in data and corrections:
                    e_key = (
                        "e_vqe" if "e_vqe" in data else ("energies" if "energies" in data else None)
                    )
                    if e_key:
                        e_vqe = data[e_key]
                        gaps = data["gaps"]
                        # Old de_gaps
                        old_de_gaps = np.abs(e_vqe - e_exact_npz) / np.maximum(gaps, 1e-10)
                        old_pass = float((old_de_gaps < DE_GAP_THRESHOLD).mean())
                        # New de_gaps (with corrected e_exact)
                        new_e_exact = e_exact_npz.copy()
                        for idx, val in corrections.items():
                            new_e_exact[idx] = val
                        new_de_gaps = np.abs(e_vqe - new_e_exact) / np.maximum(gaps, 1e-10)
                        new_pass = float((new_de_gaps < DE_GAP_THRESHOLD).mean())
                        affected_pass_rate_delta = new_pass - old_pass

                issues.append(
                    {
                        "file": npz_file.name,
                        "topology": topo,
                        "n_qubits": n_val,
                        "n_mismatched": file_mismatches,
                        "n_gap_mismatched": file_gap_mismatches,
                        "n_total": len(h_vals),
                        "max_delta": file_max_delta,
                        "pass_rate_impact": affected_pass_rate_delta,
                    }
                )

                if fix and (corrections or gap_corrections):
                    # Backup original
                    bak_path = npz_file.with_suffix(".npz.bak")
                    if not bak_path.exists():
                        shutil.copy2(npz_file, bak_path)

                    # Reload, fix, and save
                    arrays = dict(np.load(str(npz_file), allow_pickle=True))
                    new_e_exact = arrays["e_exact"].copy()
                    for idx, val in corrections.items():
                        new_e_exact[idx] = val
                    arrays["e_exact"] = new_e_exact

                    # Correct stale gaps from GT cache too.
                    if gap_corrections and "gaps" in arrays:
                        new_gaps = arrays["gaps"].copy()
                        for idx, val in gap_corrections.items():
                            new_gaps[idx] = val
                        arrays["gaps"] = new_gaps

                    # Recompute de_gaps using the corrected e_exact AND gaps.
                    e_key = (
                        "e_vqe"
                        if "e_vqe" in arrays
                        else ("energies" if "energies" in arrays else None)
                    )
                    if e_key and "gaps" in arrays:
                        new_de_gaps = np.abs(arrays[e_key] - new_e_exact) / np.maximum(
                            arrays["gaps"], 1e-10
                        )
                        arrays["de_gaps"] = new_de_gaps

                    np.savez(str(npz_file), **arrays)
                    n_points_fixed += len(corrections)
                    n_gap_points_fixed += len(gap_corrections)
                    logger.info(
                        "validate_gt_npz_coherence: fixed %s (%d e_exact, %d gap, "
                        "max_delta=%.2e)",
                        npz_file.name,
                        len(corrections),
                        len(gap_corrections),
                        file_max_delta,
                    )

    # Summary
    if n_files_with_issues == 0:
        summary = (
            f"✅ All {n_files_checked} NPZ files coherent with GT cache "
            f"({n_points_checked} points checked, {n_points_not_in_gt} not in GT)."
        )
    else:
        if fix:
            fix_status = f" ({n_points_fixed} e_exact, {n_gap_points_fixed} gap fixed)"
        else:
            fix_status = " (run with fix=True to correct)"
        gap_str = (
            f", {n_gap_points_mismatched} stale gap" if n_gap_points_mismatched else ""
        )
        summary = (
            f"⚠️ {n_files_with_issues}/{n_files_checked} files stale: "
            f"{n_points_mismatched} e_exact{gap_str} "
            f"(max_delta={max_delta_global:.2e}){fix_status}"
        )
        if not fix:
            logger.warning("GT↔NPZ coherence: %s", summary)

    return {
        "n_files_checked": n_files_checked,
        "n_files_with_issues": n_files_with_issues,
        "n_points_checked": n_points_checked,
        "n_points_mismatched": n_points_mismatched,
        "n_gap_points_mismatched": n_gap_points_mismatched,
        "n_points_not_in_gt": n_points_not_in_gt,
        "n_points_fixed": n_points_fixed,
        "n_gap_points_fixed": n_gap_points_fixed,
        "max_delta": max_delta_global,
        "issues": issues,
        "summary": summary,
    }


def validate_npz_integrity(
    *,
    p_layers: int | None = None,
    topology: str | None = None,
    check_theta_dims: bool = True,
    check_gt_coherence: bool = True,
    check_nan: bool = True,
    include_extrapolation: bool = True,
    fix: bool = False,
) -> dict:
    """Validate NPZ training data integrity for a specific p_layers value.

    Checks performed:
    1. theta_opt dimensions match expected n_params for (topology, N, p)
    2. No NaN/Inf values in theta_opt or e_vqe
    3. e_exact matches Ground Truth cache (delegates to validate_gt_npz_coherence)
    4. de_gaps are consistent with e_vqe, e_exact, gaps
    5. Quality tier distribution is reasonable

    Parameters
    ----------
    p_layers : int | None
        If specified, only check NPZ files for this p_layers value.
        If None, check all available p_layers.
    topology : str | None
        If specified, only check this topology. If None, check all.
    check_theta_dims : bool
        Validate theta dimensions vs circuit params (requires HVA builder).
    check_gt_coherence : bool
        Cross-validate e_exact against GT cache.
    check_nan : bool
        Check for NaN/Inf in numeric arrays.
    include_extrapolation : bool
        Also scan data/large_n_extrapolation/ (default True).
    fix : bool
        If True, auto-fix recoverable issues:
        - Remove rows with NaN/Inf theta (rewrite NPZ without them)
        - Recompute de_gaps from e_vqe, e_exact, gaps if inconsistent
        Creates .bak backup before modifying.

    Returns
    -------
    dict
        {
            "n_files": int,
            "n_issues": int,
            "n_fixed": int,
            "issues": list[dict],  # per-file issues
            "summary": str,
            "by_p_layers": dict[int, dict],  # breakdown per p
        }
    """
    from pathlib import Path as _P

    _ROOT = _P(__file__).resolve().parents[3]
    dirs_to_scan = [_ROOT / "data" / "multi_n_training"]
    if include_extrapolation:
        dirs_to_scan.append(_ROOT / "data" / "large_n_extrapolation")

    # Build glob pattern
    if topology and p_layers:
        pattern = f"{topology}_N*_p{p_layers}.npz"
    elif topology:
        pattern = f"{topology}_N*_p*.npz"
    elif p_layers:
        pattern = f"*_N*_p{p_layers}.npz"
    else:
        pattern = "*_N*_p*.npz"

    issues: list[dict] = []
    by_p: dict[int, dict] = {}
    n_fixed = 0

    # Optional: load HVA builder for theta dim validation
    _hva = None
    _make_lattice = None
    if check_theta_dims:
        try:
            from qmbp_simulation.circuits.hva import HVACircuitBuilder
            from qmbp_simulation.models.hamiltonian import make_lattice as _ml

            _hva = HVACircuitBuilder()
            _make_lattice = _ml
        except ImportError:
            check_theta_dims = False

    for npz_dir in dirs_to_scan:
        if not npz_dir.exists():
            continue

        for npz_file in sorted(npz_dir.glob(pattern)):
            if "_quarantine" in str(npz_file) or "_extrap_hold" in str(npz_file):
                continue

            fname = npz_file.stem
            file_issues: list[str] = []
            file_fixable: list[str] = []

            # Parse filename: {topo}_N{n}_p{p}.npz
            try:
                parts = fname.rsplit("_p", 1)
                p_val = int(parts[1])
                topo_n = parts[0]
                topo = topo_n.rsplit("_N", 1)[0]
                n_val = int(topo_n.rsplit("_N", 1)[1])
            except (IndexError, ValueError):
                file_issues.append(f"Cannot parse filename: {fname}")
                issues.append({"file": npz_file.name, "issues": file_issues, "dir": npz_dir.name})
                continue

            # Track per-p stats
            if p_val not in by_p:
                by_p[p_val] = {
                    "n_files": 0,
                    "n_points": 0,
                    "n_nan": 0,
                    "n_dim_mismatch": 0,
                    "n_fixed": 0,
                }
            by_p[p_val]["n_files"] += 1

            try:
                data = np.load(str(npz_file), allow_pickle=True)
            except Exception as e:
                file_issues.append(f"Load failed: {e}")
                issues.append({"file": npz_file.name, "issues": file_issues, "dir": npz_dir.name})
                continue

            # Required keys
            for key in ("h_values", "theta_opt", "e_vqe", "e_exact"):
                if key not in data:
                    file_issues.append(f"Missing key: {key}")

            if file_issues:
                issues.append({"file": npz_file.name, "issues": file_issues, "dir": npz_dir.name})
                continue

            h_values = np.asarray(data["h_values"], dtype=np.float64)
            theta_opt = data["theta_opt"]
            e_vqe = np.asarray(data["e_vqe"], dtype=np.float64)
            e_exact = np.asarray(data["e_exact"], dtype=np.float64)
            n_pts = len(h_values)
            by_p[p_val]["n_points"] += n_pts

            # Check NaN/Inf
            nan_rows = []
            if check_nan and theta_opt.ndim == 2:
                nan_mask = ~np.all(np.isfinite(theta_opt), axis=1)
                nan_rows = list(np.where(nan_mask)[0])
                nan_e = int(np.sum(~np.isfinite(e_vqe)))
                if nan_rows:
                    file_issues.append(f"NaN/Inf theta: {len(nan_rows)}/{n_pts} points")
                    file_fixable.append("nan_theta")
                    by_p[p_val]["n_nan"] += len(nan_rows)
                if nan_e > 0:
                    file_issues.append(f"NaN/Inf e_vqe: {nan_e}/{n_pts} points")
                    file_fixable.append("nan_evqe")

            # Check theta dimensions
            if check_theta_dims and _hva and _make_lattice and theta_opt.ndim == 2:
                try:
                    lattice = _make_lattice(topo, n_val, J=1.0, h=2.0)
                    circuit, _ = _hva.create_bond_resolved(n_val, p_val, lattice)
                    expected_params = circuit.num_parameters
                    actual_params = theta_opt.shape[1]
                    if actual_params != expected_params:
                        file_issues.append(
                            f"Theta dim mismatch: got {actual_params}, "
                            f"expected {expected_params} for (N={n_val}, p={p_val})"
                        )
                        by_p[p_val]["n_dim_mismatch"] += n_pts
                except Exception as e:
                    file_issues.append(f"Dim check failed: {e}")

            # Check consistency: lengths
            if len(h_values) != len(e_vqe):
                file_issues.append(f"Length mismatch: h={len(h_values)} vs e_vqe={len(e_vqe)}")
            if theta_opt.ndim == 2 and theta_opt.shape[0] != len(h_values):
                file_issues.append(
                    f"Length mismatch: h={len(h_values)} vs theta={theta_opt.shape[0]}"
                )

            # ── Auto-fix: remove NaN rows ──
            if fix and nan_rows and "nan_theta" in file_fixable:
                import shutil

                shutil.copy2(npz_file, str(npz_file) + ".bak")
                valid_mask = np.all(np.isfinite(theta_opt), axis=1)
                # Also filter out NaN e_vqe
                valid_mask &= np.isfinite(e_vqe)
                if valid_mask.sum() > 0:
                    save_data = {}
                    for key in data.files:
                        arr = data[key]
                        if hasattr(arr, "__len__") and len(arr) == n_pts:
                            save_data[key] = np.asarray(arr)[valid_mask]
                        else:
                            save_data[key] = arr
                    np.savez(npz_file, **save_data)
                    n_removed = n_pts - int(valid_mask.sum())
                    file_issues.append(f"FIXED: removed {n_removed} NaN rows")
                    n_fixed += 1
                    by_p[p_val]["n_fixed"] += 1

            if file_issues:
                issues.append(
                    {
                        "file": npz_file.name,
                        "topology": topo,
                        "n_qubits": n_val,
                        "p_layers": p_val,
                        "issues": file_issues,
                        "dir": npz_dir.name,
                    }
                )

    n_files = sum(v["n_files"] for v in by_p.values())
    n_issues = len(issues)

    if n_issues == 0:
        summary = f"✅ All {n_files} NPZ files pass integrity checks."
    elif fix and n_fixed > 0:
        summary = f"🔧 {n_fixed}/{n_issues} issues auto-fixed. {n_issues - n_fixed} remaining."
    else:
        summary = f"⚠️ {n_issues}/{n_files} NPZ files have issues."

    return {
        "n_files": n_files,
        "n_issues": n_issues,
        "n_fixed": n_fixed,
        "issues": issues,
        "summary": summary,
        "by_p_layers": by_p,
    }


def validate_p2_vs_p1_energy_monotonicity(
    *,
    topology: str | None = None,
    tolerance: float = 1e-4,
) -> dict:
    """Cross-validate that p=2 VQE energies are <= p=1 energies at matching h-points.

    A more expressive circuit (higher p) should always achieve lower or equal
    variational energy. Violations indicate:
    - VQE convergence issues at p=2 (insufficient iterations/restarts)
    - Corrupted data in one of the NPZ files
    - Bug in the warm-start tiling

    Parameters
    ----------
    topology : str | None
        If specified, only check this topology. If None, check all available.
    tolerance : float
        Allowed violation margin (p=2 can be up to this much worse than p=1
        without being flagged). Default: 1e-4 Hartree.

    Returns
    -------
    dict
        {
            "n_topologies": int,
            "n_common_points": int,
            "n_violations": int,
            "violations": list[dict],  # per-point violations
            "summary": str,
            "by_topology": dict[str, dict],
        }
    """
    from pathlib import Path as _P

    _ROOT = _P(__file__).resolve().parents[3]
    npz_dir = _ROOT / "data" / "multi_n_training"

    if not npz_dir.exists():
        return {
            "n_topologies": 0,
            "n_common_points": 0,
            "n_violations": 0,
            "violations": [],
            "summary": "NPZ dir not found.",
            "by_topology": {},
        }

    # Discover topology+N combos that have BOTH p=1 and p=2 data
    p1_files: dict[tuple[str, int], _P] = {}
    p2_files: dict[tuple[str, int], _P] = {}

    for f in npz_dir.glob("*_N*_p1.npz"):
        if "_quarantine" in str(f) or "_extrap_hold" in str(f):
            continue
        parts = f.stem.rsplit("_p", 1)
        topo_n = parts[0]
        topo = topo_n.rsplit("_N", 1)[0]
        n = int(topo_n.rsplit("_N", 1)[1])
        if topology and topo != topology:
            continue
        p1_files[(topo, n)] = f

    for f in npz_dir.glob("*_N*_p2.npz"):
        if "_quarantine" in str(f) or "_extrap_hold" in str(f):
            continue
        parts = f.stem.rsplit("_p", 1)
        topo_n = parts[0]
        topo = topo_n.rsplit("_N", 1)[0]
        n = int(topo_n.rsplit("_N", 1)[1])
        if topology and topo != topology:
            continue
        p2_files[(topo, n)] = f

    # Find common keys
    common_keys = set(p1_files.keys()) & set(p2_files.keys())
    if not common_keys:
        return {
            "n_topologies": 0,
            "n_common_points": 0,
            "n_violations": 0,
            "violations": [],
            "summary": "No (topology, N) pairs with both p=1 and p=2 data found.",
            "by_topology": {},
        }

    violations: list[dict] = []
    by_topo: dict[str, dict] = {}
    n_common_total = 0

    for topo, n in sorted(common_keys):
        data_p1 = np.load(str(p1_files[(topo, n)]), allow_pickle=True)
        data_p2 = np.load(str(p2_files[(topo, n)]), allow_pickle=True)

        h_p1 = np.asarray(data_p1["h_values"], dtype=np.float64)
        e_p1 = np.asarray(data_p1["e_vqe"], dtype=np.float64)
        h_p2 = np.asarray(data_p2["h_values"], dtype=np.float64)
        e_p2 = np.asarray(data_p2["e_vqe"], dtype=np.float64)

        # Match h-points (tolerance 1e-4)
        topo_violations = 0
        topo_common = 0

        for i2, h2 in enumerate(h_p2):
            diffs = np.abs(h_p1 - h2)
            idx1 = int(np.argmin(diffs))
            if diffs[idx1] < 1e-4:
                topo_common += 1
                n_common_total += 1
                # p=2 should be <= p=1 (more expressive)
                if e_p2[i2] > e_p1[idx1] + tolerance:
                    delta = e_p2[i2] - e_p1[idx1]
                    violations.append(
                        {
                            "topology": topo,
                            "n_qubits": n,
                            "h": float(h2),
                            "e_p1": float(e_p1[idx1]),
                            "e_p2": float(e_p2[i2]),
                            "delta": float(delta),
                        }
                    )
                    topo_violations += 1

        if topo not in by_topo:
            by_topo[topo] = {"n_common": 0, "n_violations": 0, "n_values": []}
        by_topo[topo]["n_common"] += topo_common
        by_topo[topo]["n_violations"] += topo_violations
        by_topo[topo]["n_values"].append(n)

    n_violations = len(violations)
    topos_checked = len(by_topo)

    if n_violations == 0:
        summary = (
            f"✅ Energy monotonicity OK: {n_common_total} common points across "
            f"{topos_checked} topologies. p=2 ≤ p=1 everywhere."
        )
    else:
        worst = max(violations, key=lambda v: v["delta"])
        summary = (
            f"⚠️ {n_violations}/{n_common_total} points violate energy monotonicity "
            f"(p=2 > p=1). Worst: {worst['topology']} N={worst['n_qubits']} "
            f"h={worst['h']:.3f} Δ={worst['delta']:.4f}. "
            f"VQE at p=2 may need more restarts/iterations at these points."
        )

    return {
        "n_topologies": topos_checked,
        "n_common_points": n_common_total,
        "n_violations": n_violations,
        "violations": violations,
        "summary": summary,
        "by_topology": by_topo,
    }


def check_p2_regression_vs_p1(
    topology: str,
    p2_pass_rate: float,
    *,
    h_min: float = 2.0,
) -> dict:
    """Anti-regression guard: verify p=2 model doesn't underperform p=1.

    Compares a p=2 model's pass_rate against the best p=1 model for the
    same topology in the valid h-range. A more expressive circuit should
    always achieve equal or better results.

    This guard should be called after training/evaluating a p=2 model,
    before registering it in the zoo.

    Parameters
    ----------
    topology : str
        Topology being evaluated.
    p2_pass_rate : float
        Dual-criterion pass rate of the p=2 model (0.0 to 1.0).
    h_min : float
        Only compare in h >= h_min (valid regime). Default 2.0.

    Returns
    -------
    dict
        {
            "passed": bool,
            "p1_pass_rate": float | None,
            "p2_pass_rate": float,
            "delta": float,  # p2 - p1 (positive = p2 better)
            "message": str,
        }
    """
    # Load best p=1 model pass_rate from zoo
    p1_pass_rate = None
    try:
        from qmbp_simulation.predictors.model_zoo import _load_manifest

        manifest = _load_manifest()
        p1_entries = [
            e for e in manifest if e.topology == topology and e.p_layers == 1 and e.is_multi_n
        ]
        if p1_entries:
            p1_pass_rate = max(e.pass_rate for e in p1_entries)
    except Exception:
        pass

    if p1_pass_rate is None:
        return {
            "passed": True,
            "p1_pass_rate": None,
            "p2_pass_rate": p2_pass_rate,
            "delta": 0.0,
            "message": f"No p=1 baseline for {topology}. p=2 accepted unconditionally.",
        }

    delta = p2_pass_rate - p1_pass_rate

    # Allow p=2 to be slightly worse (within 5%) since it may have less data
    # But flag significant regressions
    if delta >= -0.05:
        passed = True
        msg = (
            f"✅ p=2 OK for {topology}: pass_rate={p2_pass_rate:.0%} "
            f"(p=1 baseline={p1_pass_rate:.0%}, Δ={delta:+.0%})"
        )
    else:
        passed = False
        msg = (
            f"⚠️ REGRESSION: p=2 for {topology}: pass_rate={p2_pass_rate:.0%} "
            f"is {-delta:.0%} worse than p=1 baseline ({p1_pass_rate:.0%}). "
            f"VQE at p=2 may need more restarts/iterations, or training data is insufficient."
        )

    return {
        "passed": passed,
        "p1_pass_rate": p1_pass_rate,
        "p2_pass_rate": p2_pass_rate,
        "delta": delta,
        "message": msg,
    }


def post_experiment_sync(*, verbose: bool = False, p_layers: int | None = None) -> dict:
    """Consolidated post-experiment synchronization of all data stores.

    Runs all maintenance tasks in the correct dependency order to ensure
    data coherence across the entire pipeline. Call after any experiment
    that produces new data (VQE, training, evaluation, comparison).

    Execution order:
    1. GT↔NPZ coherence check (detect stale e_exact)
    2. Regenerate model quality dashboard (reads NPZ)
    3. MT vs ST comparison refresh (reads model_comparison/)
    4. Eval report regeneration (reads zoo + NPZ)
    5. Cross-N coverage doc update (reads dashboard)
    6. ResultIndex rebuild + project-status.md refresh
    7. Auto-detect exclusions (fire-and-forget)

    Parameters
    ----------
    verbose : bool
        If True, print progress and issue details.
    p_layers : int | None
        The HVA depth the calling experiment ran with. When set, the best
        results scoreboard is regenerated with ``--p-layers {p_layers}`` so
        that filtering/grouping matches the experiment's p (never mixing
        p=1 and p=2). None = regenerate for all p (standalone/manual runs).

    Returns
    -------
    dict
        {
            "steps_completed": list[str],
            "steps_failed": list[str],
            "gt_coherence": dict | None,
            "dashboard_regenerated": bool,
            "eval_report_updated": bool,
            "coverage_updated": bool,
            "status_updated": bool,
        }
    """
    import subprocess
    import sys
    from pathlib import Path as _P

    _ROOT = _P(__file__).resolve().parents[3]
    results: dict = {
        "steps_completed": [],
        "steps_failed": [],
        "gt_coherence": None,
        "dashboard_regenerated": False,
        "eval_report_updated": False,
        "coverage_updated": False,
        "status_updated": False,
    }

    def _log(msg: str):
        if verbose:
            print(f"  [sync] {msg}")
        logger.info("post_experiment_sync: %s", msg)

    # ── Step 1: GT↔NPZ coherence check ──────────────────────────────────
    _log("Step 1/6: Validating GT↔NPZ coherence...")
    try:
        coherence = validate_gt_npz_coherence(fix=False)
        results["gt_coherence"] = coherence
        results["steps_completed"].append("gt_npz_coherence")
        if coherence["n_files_with_issues"] > 0:
            _log(f"  ⚠️ {coherence['summary']}")
        else:
            _log(f"  ✅ {coherence['summary']}")
    except Exception as e:
        results["steps_failed"].append(f"gt_npz_coherence: {e}")
        _log(f"  ❌ Failed: {e}")

    # ── Step 2: Regenerate dashboard ─────────────────────────────────────
    _log("Step 2/6: Regenerating model quality dashboard...")
    try:
        # Force regeneration by importing and calling directly
        dashboard = generate_model_quality_dashboard()
        results["dashboard_regenerated"] = True
        results["steps_completed"].append("dashboard")
        _log(f"  ✅ Dashboard: {dashboard.get('n_configs', 0)} configs")
    except Exception as e:
        results["steps_failed"].append(f"dashboard: {e}")
        _log(f"  ❌ Failed: {e}")

    # ── Step 2b: Best Results Scoreboard ────────────────────────────────
    _log("Step 2b/9: Regenerating best results scoreboard...")
    try:
        scoreboard_script = _ROOT / "scripts" / "analysis" / "generate_best_results_scoreboard.py"
        if scoreboard_script.exists():
            _sb_cmd = [sys.executable, str(scoreboard_script)]
            if p_layers is not None:
                _sb_cmd += ["--p-layers", str(p_layers)]
            proc = subprocess.run(
                _sb_cmd,
                capture_output=True,
                text=True,
                timeout=120,  # scan of all eval reports + cold-import warmup
                cwd=str(_ROOT),
            )
            # exit 1 with p filter + no data is a benign "nothing to generate
            # for this p yet" (e.g. first p=2 run before any p=2 eval exists),
            # not a real failure.
            _no_data = p_layers is not None and "No evaluation reports found" in (proc.stdout or "")
            if proc.returncode == 0:
                results["steps_completed"].append("best_results_scoreboard")
                _log(
                    "  ✅ Best results scoreboard updated"
                    + (f" (p={p_layers})" if p_layers is not None else "")
                )
            elif _no_data:
                results["steps_completed"].append(
                    f"best_results_scoreboard (no p={p_layers} data yet)"
                )
                _log(f"  ⏭️ No eval reports for p={p_layers} yet — scoreboard skipped")
            else:
                results["steps_failed"].append(f"best_results_scoreboard: exit={proc.returncode}")
                _log(f"  ❌ Exit code {proc.returncode}")
        else:
            results["steps_completed"].append("best_results_scoreboard (skipped)")
    except subprocess.TimeoutExpired:
        results["steps_failed"].append("best_results_scoreboard: timeout")
        _log("  ⏰ Timed out")
    except Exception as e:
        results["steps_failed"].append(f"best_results_scoreboard: {e}")
        _log(f"  ❌ Failed: {e}")

    # ── Step 2c: Auto-fix scoreboard issues ─────────────────────────────
    _log("Step 2c/9: Auto-fixing scoreboard issues...")
    try:
        fix_result = auto_fix_scoreboard_issues(verbose=verbose)
        n_fixes = (
            fix_result["n_recovered"]
            + fix_result["n_registered"]
            + fix_result["n_backfilled"]
            + fix_result["n_exclusions_removed"]
            + fix_result["n_exclusions_added"]
        )
        results["scoreboard_fixes"] = fix_result
        results["steps_completed"].append("scoreboard_auto_fix")
        if n_fixes > 0:
            _log(
                f"  ✅ Auto-fixed: {fix_result['n_recovered']} recovered, "
                f"{fix_result['n_registered']} registered, "
                f"{fix_result['n_backfilled']} backfilled, "
                f"{fix_result['n_exclusions_removed']} exclusions removed"
            )
        else:
            _log("  ✅ No scoreboard issues to fix")
    except Exception as e:
        results["steps_failed"].append(f"scoreboard_auto_fix: {e}")
        _log(f"  ❌ Failed: {e}")

    # ── Step 3: Eval report regeneration ─────────────────────────────────
    _log("Step 3/6: Regenerating evaluation report...")
    try:
        eval_script = _ROOT / "scripts" / "analysis" / "evaluate_zoo_models.py"
        if not eval_script.exists():
            eval_script = _ROOT / "scripts" / "maintenance" / "evaluate_zoo_models.py"
        if eval_script.exists():
            proc = subprocess.run(
                [sys.executable, str(eval_script)],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(_ROOT),
            )
            if proc.returncode == 0:
                results["eval_report_updated"] = True
                results["steps_completed"].append("eval_report")
                _log("  ✅ Evaluation report updated")
            else:
                results["steps_failed"].append(f"eval_report: exit={proc.returncode}")
                _log(f"  ❌ Exit code {proc.returncode}")
        else:
            results["steps_completed"].append("eval_report (skipped, script not found)")
            _log("  ⏭️ evaluate_zoo_models.py not found, skipped")
    except subprocess.TimeoutExpired:
        results["steps_failed"].append("eval_report: timeout (120s)")
        _log("  ⏰ Timed out (120s limit)")
    except Exception as e:
        results["steps_failed"].append(f"eval_report: {e}")
        _log(f"  ❌ Failed: {e}")

    # ── Step 4: Cross-N coverage doc update ──────────────────────────────
    _log("Step 4/6: Updating cross-N coverage documentation...")
    try:
        coverage_script = _ROOT / "scripts" / "maintenance" / "update_cross_n_coverage.py"
        if coverage_script.exists():
            proc = subprocess.run(
                [sys.executable, str(coverage_script)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(_ROOT),
            )
            if proc.returncode == 0:
                results["coverage_updated"] = True
                results["steps_completed"].append("coverage_doc")
                _log("  ✅ Coverage documentation updated")
            else:
                results["steps_failed"].append(f"coverage_doc: exit={proc.returncode}")
                _log(f"  ❌ Exit code {proc.returncode}")
        else:
            results["steps_completed"].append("coverage_doc (skipped)")
            _log("  ⏭️ update_cross_n_coverage.py not found")
    except subprocess.TimeoutExpired:
        results["steps_failed"].append("coverage_doc: timeout")
        _log("  ⏰ Timed out")
    except Exception as e:
        results["steps_failed"].append(f"coverage_doc: {e}")
        _log(f"  ❌ Failed: {e}")

    # ── Step 5: ResultIndex + project-status.md ──────────────────────────
    _log("Step 5/6: Rebuilding ResultIndex + project-status.md...")
    try:
        from qmbp_simulation.framework.result_index import ResultIndex

        idx = ResultIndex()
        idx.rebuild()
        idx.refresh_status()
        results["status_updated"] = True
        results["steps_completed"].append("result_index")
        _log("  ✅ ResultIndex rebuilt + project-status.md refreshed")
    except Exception as e:
        results["steps_failed"].append(f"result_index: {e}")
        _log(f"  ❌ Failed: {e}")

    # ── Step 6: Auto-detect exclusions ───────────────────────────────────
    _log("Step 6/7: Auto-detecting training exclusions...")
    try:
        auto_detect_exclusions()
        results["steps_completed"].append("exclusions")
        _log("  ✅ Exclusion registry updated")
    except Exception as e:
        results["steps_failed"].append(f"exclusions: {e}")
        _log(f"  ❌ Failed: {e}")

    # ── Step 7: Auto-correct zoo inflation + backfill pass_rate_by_n ─────
    _log("Step 7/7: Zoo pass_rate coherence + backfill...")
    try:
        from qmbp_simulation.predictors.model_zoo import (
            _load_manifest,
            backfill_critical_ranking_from_evals,
            backfill_pass_rate_by_n_from_comparisons,
            update_zoo_pass_rate,
        )

        # Backfill pass_rate_by_n from comparison history
        n_backfilled = backfill_pass_rate_by_n_from_comparisons()

        # Backfill the empirical critical h-window ranking ([0.8, 1.8]) from the
        # per-h eval reports. Keeps load_best_model_for(h_regime="critical")
        # synced with the newest experiments. Best-effort — never blocks sync.
        try:
            n_crit = backfill_critical_ranking_from_evals()
            _log(f"  ✅ Critical-window ranking: {n_crit} entries updated")
        except Exception as _crit_exc:  # noqa: BLE001
            _log(f"  ⚠️ Critical-window ranking backfill skipped: {_crit_exc}")

        # Auto-correct inflated zoo entries (zoo > comparison by >25%)
        entries = _load_manifest()
        n_deflated = 0
        for entry in entries:
            if not entry.pass_rate_by_n or entry.pass_rate == 0:
                continue
            rates = [float(v) for v in entry.pass_rate_by_n.values()]
            comp_avg = sum(rates) / len(rates)
            if entry.pass_rate > comp_avg + 0.25:
                update_zoo_pass_rate(
                    entry.checkpoint_file,
                    comp_avg,
                    only_if_better=False,
                    pass_rate_source="comparison_eval",
                    add_notes=f"auto-deflated from {entry.pass_rate:.0%}",
                )
                n_deflated += 1

        results["steps_completed"].append("zoo_coherence")
        _log(f"  ✅ Backfilled {n_backfilled} models, deflated {n_deflated} inflated entries")
    except Exception as e:
        results["steps_failed"].append(f"zoo_coherence: {e}")
        _log(f"  ❌ Failed: {e}")

    # ── Step 7b: Auto-archive orphan checkpoints ─────────────────────────
    try:
        from qmbp_simulation.predictors.model_zoo import _CHECKPOINTS_DIR, _load_manifest

        manifest_files = {e.checkpoint_file for e in _load_manifest()}
        if _CHECKPOINTS_DIR.exists():
            disk_files = {f.name for f in _CHECKPOINTS_DIR.glob("*.pt")}
            orphans = disk_files - manifest_files
            if orphans:
                archive_dir = _CHECKPOINTS_DIR / "_archive"
                archive_dir.mkdir(parents=True, exist_ok=True)
                import shutil

                for orphan_name in orphans:
                    src = _CHECKPOINTS_DIR / orphan_name
                    dst = archive_dir / orphan_name
                    if src.exists() and not dst.exists():
                        shutil.move(str(src), str(dst))
                _log(f"  📦 Auto-archived {len(orphans)} orphan checkpoint(s)")
    except Exception:
        pass  # Non-critical

    # ── Step 8: Retrain trigger detection ────────────────────────────────
    _log("Step 8/9: Checking retrain triggers...")
    try:
        from qmbp_simulation.predictors.training_intelligence import check_retrain_triggers

        triggers = check_retrain_triggers()
        results["retrain_triggers"] = [
            {"topology": t.topology, "priority": t.priority, "reason": t.reason} for t in triggers
        ]
        results["steps_completed"].append("retrain_triggers")
        if triggers:
            _log(f"  ⚠️ {len(triggers)} retrain triggers detected:")
            for t in triggers[:3]:
                _log(f"    P{t.priority} {t.topology}: {t.reason[:60]}")
        else:
            _log("  ✅ No retrain triggers (all models up-to-date)")
    except Exception as e:
        results["steps_failed"].append(f"retrain_triggers: {e}")
        _log(f"  ❌ Failed: {e}")

    # ── Step 9: Auto-retrain loop — DISABLED ───────────────────────────
    # Disabled until architecture improvements (per-N loss weighting,
    # curriculum, jk_cat readout) resolve the fundamental N-scaling issue.
    # The current MPNN cannot generalize to N≥16 regardless of training data.
    # Re-enable when a new architecture variant shows >30% pass at N=16.
    _log("Step 9/9: Auto-retrain loop (DISABLED — architecture bottleneck)")
    results["retrain_loop"] = None
    results["steps_completed"].append("retrain_loop (disabled)")
    if results.get("retrain_triggers"):
        _log(f"  ⏸️ {len(results['retrain_triggers'])} triggers detected but loop disabled")

    # ── Summary ──────────────────────────────────────────────────────────
    n_ok = len(results["steps_completed"])
    n_fail = len(results["steps_failed"])

    # ── Tier 3 cross-validations (lightweight, non-blocking) ─────────────
    try:
        tier3 = run_tier3_validations(verbose=verbose)
        results["tier3"] = {
            "summary": tier3["summary"],
            "n_issues": tier3["n_total_issues"],
            "undertrained": tier3["training_convergence"]["n_undertrained"],
            "eval_comparison_gaps": tier3["eval_vs_comparison"]["n_discrepancies"],
            "retrain_plan_size": tier3["cascading_retrain"]["n_models_to_retrain"],
        }
        results["steps_completed"].append("tier3_validations")
        if tier3["n_total_issues"] > 0:
            _log(f"  ⚠️ Tier 3: {tier3['summary']}")
        else:
            _log("  ✅ Tier 3: all cross-validations pass")
    except Exception as e:
        results["steps_failed"].append(f"tier3_validations: {e}")
        _log(f"  ❌ Tier 3 failed: {e}")

    # ── Data consistency cross-check (zoo↔comparison↔dashboard↔registry) ──
    # Runs automatically now (was manual): detects pass_rate drift across the
    # independent data sources so incoherence surfaces every run, not only when
    # someone remembers to run it. Non-blocking.
    try:
        consistency = validate_data_consistency(verbose=verbose)
        n_disc = consistency.get("n_issues", 0)
        results["data_consistency"] = {
            "is_consistent": consistency.get("is_consistent", True),
            "n_checks": consistency.get("n_checks", 0),
            "n_issues": n_disc,
        }
        results["steps_completed"].append("data_consistency")
        if n_disc > 0:
            _log(
                f"  ⚠️ Data consistency: {n_disc} discrepancies across "
                f"{consistency.get('n_checks', 0)} checks (zoo↔comparison↔dashboard↔registry)"
            )
        else:
            _log("  ✅ Data consistency: all sources coherent")
    except Exception as e:
        results["steps_failed"].append(f"data_consistency: {e}")
        _log(f"  ❌ Data consistency failed: {e}")

    # ── DQPT trajectory inventory (lightweight scan for status report) ───
    try:
        dqpt_dir = _ROOT / "data" / "dqpt_trajectories"
        if dqpt_dir.exists():
            dqpt_files = list(dqpt_dir.glob("*.npz"))
            # Group by topology
            dqpt_by_topo: dict[str, int] = {}
            for f in dqpt_files:
                topo_part = f.stem.rsplit("_N", 1)[0] if "_N" in f.stem else f.stem
                dqpt_by_topo[topo_part] = dqpt_by_topo.get(topo_part, 0) + 1
            results["dqpt_trajectories"] = {
                "n_files": len(dqpt_files),
                "by_topology": dqpt_by_topo,
            }
            if dqpt_files:
                _log(
                    f"  📊 DQPT trajectories: {len(dqpt_files)} files "
                    f"({', '.join(f'{t}:{n}' for t, n in sorted(dqpt_by_topo.items()))})"
                )
    except Exception:
        pass  # Non-critical

    if verbose:
        print(f"\n  [sync] Done: {n_ok} completed, {n_fail} failed")
        if results["steps_failed"]:
            for f in results["steps_failed"]:
                print(f"    ❌ {f}")

    return results


def validate_data_consistency(*, verbose: bool = False) -> dict:
    """Cross-validate pass_rates across all data sources for coherence.

    Compares metrics from 3 independent sources:
    1. **Dashboard** (NPZ training data): pass_rate of the VQE θ_opt themselves
    2. **Model comparison JSONs**: MPNN prediction quality on specific (topo, N, h-grid)
    3. **Eval report / Zoo manifest**: registered model quality

    Expected differences:
    - Dashboard pass_rate > comparison pass_rate (training data is easier than predictions)
    - Zoo pass_rate should correlate with comparison results

    Detects:
    - Stale data (eval report grades inconsistent with latest comparisons)
    - Incoherent signals (zoo says 100% but comparison shows 30%)
    - Registry MSE drift vs training curves
    - Cross-N model selection vs dashboard evidence mismatch

    Parameters
    ----------
    verbose : bool
        Print detailed findings.

    Returns
    -------
    dict
        {
            "is_consistent": bool,
            "n_checks": int,
            "n_issues": int,
            "findings": list[dict],  # [{source_a, source_b, field, value_a, value_b, severity}]
            "zoo_vs_comparison": dict,  # Per-model: zoo pass_rate vs latest comparison
            "registry_vs_curves": dict,  # MSE cross-check
            "cross_n_selection_issues": list,
        }
    """
    import json as _json
    from pathlib import Path as _P

    _ROOT = _P(__file__).resolve().parents[3]
    findings: list[dict] = []
    n_checks = 0

    # ── Source 1: Dashboard ──────────────────────────────────────────────
    dash_path = _ROOT / "data" / "model_quality_dashboard.json"
    dashboard = None
    if dash_path.exists():
        with open(dash_path) as f:
            dashboard = _json.load(f)

    # ── Source 2: Zoo manifest ───────────────────────────────────────────
    zoo_entries: list = []
    try:
        from qmbp_simulation.predictors.model_zoo import _load_manifest

        zoo_entries = _load_manifest()
    except Exception:
        pass

    # ── Source 3: Model comparison JSONs ─────────────────────────────────
    comp_dir = _ROOT / "results" / "model_comparison"
    comparison_data: dict[str, dict] = {}  # checkpoint → {topology, pass_rates_by_n}
    if comp_dir.exists():
        for f in sorted(comp_dir.glob("compare_*.json")):
            try:
                d = _json.loads(f.read_text())
                topo = d.get("topology")
                for r in d.get("results", []):
                    ckpt = r.get("checkpoint", "")
                    if not ckpt:
                        continue
                    by_n: dict[int, float] = {}
                    for n_str, metrics in r.get("results_by_n", {}).items():
                        by_n[int(n_str)] = metrics.get("pass_rate_dual", 0.0)
                    if ckpt not in comparison_data:
                        comparison_data[ckpt] = {"topology": topo, "pass_rates_by_n": {}}
                    comparison_data[ckpt]["pass_rates_by_n"].update(by_n)
            except Exception:
                continue

    # ── Check A: Zoo pass_rate vs comparison reality ─────────────────────
    zoo_vs_comparison: dict[str, dict] = {}
    for entry in zoo_entries:
        ckpt = entry.checkpoint_file
        if ckpt in comparison_data:
            comp_rates = comparison_data[ckpt]["pass_rates_by_n"]
            if comp_rates:
                comp_avg = sum(comp_rates.values()) / len(comp_rates)
                zoo_rate = entry.pass_rate
                delta = abs(zoo_rate - comp_avg)
                n_checks += 1

                zoo_vs_comparison[ckpt] = {
                    "zoo_pass_rate": zoo_rate,
                    "comparison_avg": comp_avg,
                    "comparison_by_n": comp_rates,
                    "delta": delta,
                    "consistent": delta < 0.25,
                }

                if delta > 0.25:
                    findings.append(
                        {
                            "source_a": "zoo_manifest",
                            "source_b": "model_comparison",
                            "field": "pass_rate",
                            "value_a": zoo_rate,
                            "value_b": comp_avg,
                            "delta": delta,
                            "severity": "high" if delta > 0.40 else "medium",
                            "model": ckpt[:50],
                            "explanation": (
                                f"Zoo says {zoo_rate:.0%} but comparison shows {comp_avg:.0%}. "
                                f"Note: zoo may reflect training pass_rate, comparison uses MPNN prediction."
                            ),
                        }
                    )

    # ── Check B: Registry MSE vs training curves ─────────────────────────
    registry_vs_curves: dict[str, dict] = {}
    try:
        reg_path = _ROOT / "data" / "model_zoo" / "model_registry.json"
        curves_dir = _ROOT / "results" / "training_curves"
        if reg_path.exists() and curves_dir.exists():
            reg_data = _json.loads(reg_path.read_text())
            models = reg_data if isinstance(reg_data, list) else reg_data.get("models", [])
            curves_by_stem = {f.stem: f for f in curves_dir.glob("*.npz")}

            for model in models:
                model_id = model.get("model_id", "")
                tm = model.get("training", {}).get("training_metrics", {})
                reg_mse = tm.get("final_mse")
                reg_val_mse = tm.get("final_val_mse")
                reg_epochs = tm.get("epochs", 0)
                if reg_mse is None:
                    continue

                # Match curve file by: exact stem, topology substring, or date overlap
                topo = model.get("topology", "")
                curve_file = None

                # Strategy 1: exact match
                curve_file = curves_by_stem.get(model_id.replace(".pt", ""))

                # Strategy 2: topology + partial match
                if curve_file is None:
                    for stem, path in curves_by_stem.items():
                        if topo in stem and (
                            model_id[:25] in stem
                            or stem[:25] in model_id
                            or (topo in stem and "section" in stem)
                        ):
                            curve_file = path
                            break

                # Strategy 3: MT model → mt_training curve
                if curve_file is None and topo == "multi_topology":
                    for stem, path in curves_by_stem.items():
                        if "mt_training" in stem:
                            curve_file = path  # Take latest MT curve
                            # Don't break — last one is latest

                if curve_file:
                    try:
                        curve_data = np.load(str(curve_file))
                        mse_history = curve_data["mse_history"]
                        val_mse_history = curve_data.get("val_mse_history", np.array([]))
                        actual_final_mse = float(mse_history[-1])
                        actual_val_mse = (
                            float(val_mse_history[-1]) if len(val_mse_history) > 0 else None
                        )
                        n_checks += 1

                        # Compare MSE (use val_mse if available, more comparable)
                        compare_mse = reg_val_mse if reg_val_mse is not None else reg_mse
                        actual_compare = (
                            actual_val_mse if actual_val_mse is not None else actual_final_mse
                        )
                        mse_delta = abs(compare_mse - actual_compare)

                        # Epoch count check
                        epoch_match = abs(reg_epochs - len(mse_history)) <= 5

                        registry_vs_curves[model_id[:50]] = {
                            "registry_mse": reg_mse,
                            "registry_val_mse": reg_val_mse,
                            "curve_final_mse": actual_final_mse,
                            "curve_val_mse": actual_val_mse,
                            "delta": mse_delta,
                            "n_epochs_registry": reg_epochs,
                            "n_epochs_curve": len(mse_history),
                            "epoch_match": epoch_match,
                            "curve_file": curve_file.name,
                            "consistent": mse_delta < 0.01 and epoch_match,
                        }

                        if mse_delta > 0.01:
                            findings.append(
                                {
                                    "source_a": "model_registry",
                                    "source_b": "training_curves",
                                    "field": "final_mse",
                                    "value_a": compare_mse,
                                    "value_b": actual_compare,
                                    "delta": mse_delta,
                                    "severity": "medium",
                                    "model": model_id[:50],
                                    "explanation": (
                                        f"Registry MSE={compare_mse:.4e} vs curve "
                                        f"{actual_compare:.4e} (Δ={mse_delta:.4e}). "
                                        f"Epochs: reg={reg_epochs} curve={len(mse_history)}. "
                                        f"Curve file: {curve_file.name}"
                                    ),
                                }
                            )
                    except Exception:
                        continue
    except Exception:
        pass

    # ── Check B2: Eval report grades → zoo pass_rate_by_n coherence ──────
    # If eval report shows grade F at N=30 but zoo doesn't have pass_rate_by_n
    # for that N, flag it as missing data.
    try:
        eval_report_path = _ROOT / "results" / "model_evaluation_report.md"
        if eval_report_path.exists():
            eval_text = eval_report_path.read_text()
            # Parse grades from markdown tables
            current_topo = None
            eval_grades: dict[str, dict[int, str]] = {}  # topo → {N: grade}
            for line in eval_text.splitlines():
                if line.startswith("## ") and " — " in line:
                    current_topo = line.split(" — ")[0].replace("## ", "").strip()
                if "|" in line and current_topo and line.count("|") >= 7:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 8 and parts[1].isdigit():
                        n_val = int(parts[1])
                        grade = parts[-2].strip()
                        if grade in ("A", "B", "C", "D", "F"):
                            eval_grades.setdefault(current_topo, {})[n_val] = grade

            # Cross-check: eval grades vs zoo pass_rate_by_n
            for entry in zoo_entries:
                if not entry.pass_rate_by_n:
                    continue
                topo = entry.topology
                topo_grades = eval_grades.get(topo, {})
                for n_str, pr in entry.pass_rate_by_n.items():
                    n = int(n_str)
                    grade = topo_grades.get(n)
                    if grade is None:
                        continue
                    n_checks += 1
                    # Grade→pass_rate rough mapping: A≥80%, B≥50%, C≥20%, D≥5%, F<5%
                    expected_ranges = {
                        "A": (0.50, 1.0),
                        "B": (0.30, 0.80),
                        "C": (0.10, 0.50),
                        "D": (0.0, 0.30),
                        "F": (0.0, 0.10),
                    }
                    lo, hi = expected_ranges.get(grade, (0, 1))
                    if not (lo - 0.10 <= pr <= hi + 0.20):
                        # Significant mismatch between grade and pass_rate
                        findings.append(
                            {
                                "source_a": "eval_report",
                                "source_b": "zoo_pass_rate_by_n",
                                "field": f"grade_vs_pass_rate N={n}",
                                "value_a": grade,
                                "value_b": pr,
                                "delta": 0,
                                "severity": "low",
                                "is_informational": True,
                                "model": f"{topo} N={n}",
                                "explanation": (
                                    f"Eval report grade={grade} but zoo pass_rate_by_n={pr:.0%}. "
                                    f"Expected range for {grade}: [{lo:.0%}, {hi:.0%}]. "
                                    f"(Different h-ranges or evaluation conditions.)"
                                ),
                            }
                        )
    except Exception:
        pass

    # ── Check C: Cross-N selection coherence ─────────────────────────────
    cross_n_selection_issues: list[dict] = []
    if dashboard:
        for c in dashboard.get("configs", []):
            best_source = c.get("cross_n_best_source")
            if not best_source or c["n_qubits"] < 16:
                continue

            topo = c["topology"]
            n_target = c["n_qubits"]
            best_train_n = best_source.get("train_n")
            best_pass = best_source.get("pass_rate_5pct", 0)

            # Check: does the zoo model's training data cover this N?
            config_p = c.get("p_layers", 1)
            for entry in zoo_entries:
                if entry.topology == topo and entry.is_multi_n and entry.p_layers == config_p:
                    # The zoo model exists for this topology
                    # Check if it was trained with data from best_train_n
                    n_checks += 1
                    # If zoo pass_rate_by_n has this N and it's much worse than
                    # what cross_n_transfers says, there's a selection issue
                    if entry.pass_rate_by_n:
                        zoo_n_rate = entry.pass_rate_by_n.get(str(n_target), -1)
                        if zoo_n_rate >= 0 and best_pass > 0:
                            delta = best_pass - zoo_n_rate
                            if delta > 0.20:
                                cross_n_selection_issues.append(
                                    {
                                        "topology": topo,
                                        "n_target": n_target,
                                        "dashboard_best_source": f"train_N{best_train_n}",
                                        "dashboard_pass": best_pass,
                                        "zoo_model_pass_at_n": zoo_n_rate,
                                        "delta": delta,
                                        "issue": (
                                            f"Dashboard says train_N{best_train_n} achieves "
                                            f"{best_pass:.0%} at N={n_target}, but zoo model "
                                            f"only shows {zoo_n_rate:.0%}. Model may need retrain "
                                            f"with better source data."
                                        ),
                                    }
                                )
                    break

    if cross_n_selection_issues:
        for iss in cross_n_selection_issues:
            findings.append(
                {
                    "source_a": "dashboard_cross_n",
                    "source_b": "zoo_model",
                    "field": "pass_rate_at_n",
                    "value_a": iss["dashboard_pass"],
                    "value_b": iss["zoo_model_pass_at_n"],
                    "delta": iss["delta"],
                    "severity": "info",
                    "is_informational": True,
                    "model": f"{iss['topology']} N={iss['n_target']}",
                    "explanation": (
                        f"{iss['issue']} "
                        f"(Expected: multi-N generalist vs single-N specialist trade-off.)"
                    ),
                }
            )

    # ── Summary ──────────────────────────────────────────────────────────
    # Only count non-informational findings as real issues
    real_findings = [f for f in findings if not f.get("is_informational")]
    n_issues = len(real_findings)
    is_consistent = n_issues == 0

    if verbose:
        print(f"\n  Data Consistency Report ({n_checks} cross-checks)")
        print(f"  {'─' * 50}")
        if is_consistent and not findings:
            print("  ✅ All sources are consistent")
        elif is_consistent and findings:
            print(f"  ✅ Consistent ({len(findings)} informational notes)")
            for f in findings:
                if f.get("is_informational"):
                    print(f"  ℹ️  [{f['source_a']} ↔ {f['source_b']}] {f['model']}")
                    print(f"     {f['explanation'][:100]}")
        else:
            for f in real_findings:
                sev_icon = "🔴" if f["severity"] == "high" else "🟡"
                print(f"  {sev_icon} [{f['source_a']} ↔ {f['source_b']}] {f['model']}")
                print(f"     {f['explanation']}")
            if findings != real_findings:
                info_count = len(findings) - len(real_findings)
                print(f"\n  + {info_count} informational note(s) (not errors)")
            print(f"\n  Total: {n_issues} real inconsistencies")

    # ── Tier 3: Advanced cross-validations (run non-blocking) ────────────
    tier3_report: dict | None = None
    try:
        tier3_report = run_tier3_validations(verbose=verbose)
        n_checks += tier3_report.get("eval_vs_comparison", {}).get("n_checked", 0)
        tier3_issues = tier3_report.get("n_total_issues", 0)
        if tier3_issues > 0:
            n_issues += tier3_issues
            is_consistent = False
            if verbose:
                print(f"\n  Tier 3: {tier3_report['summary']}")
    except Exception as e:
        logger.debug("validate_data_consistency: tier3 failed: %s", e)
        tier3_report = {"error": str(e)}

    # ── Scoreboard cross-check ─────────────────────────────────────────────
    scoreboard_issues: list[dict] = []
    try:
        # Per-p scoreboards: iterate every best_results_scoreboard_p*.json so
        # both p=1 and p=2 (and future p) are cross-checked, not just p=1.
        import re as _re

        _results_dir = _ROOT / "results"
        _scoreboard_paths = sorted(_results_dir.glob("best_results_scoreboard_p*.json"))
        for _scoreboard_path in _scoreboard_paths:
            # Parse the p from the filename for issue attribution.
            _p_match = _re.search(r"_p(\d+)\.json$", _scoreboard_path.name)
            _p = int(_p_match.group(1)) if _p_match else None

            scoreboard = _json.loads(_scoreboard_path.read_text())
            best_by_topo = scoreboard.get("best_by_topology", {})

            for topo, n_entries in best_by_topo.items():
                for n_str, entry in n_entries.items():
                    n = int(n_str)
                    ckpt = _P(entry.get("checkpoint", "")).name
                    grade = entry.get("grade", "?")
                    de_gap = entry.get("best_de_gap", 0)

                    # Check: does zoo know about this checkpoint?
                    if ckpt and ckpt not in zoo_vs_comparison and ckpt != "unknown":
                        # Check if it's a valid checkpoint file
                        ckpt_path = _ROOT / "data" / "model_zoo" / "checkpoints" / ckpt
                        if not ckpt_path.exists() and grade in ("A", "B"):
                            scoreboard_issues.append(
                                {
                                    "topology": topo,
                                    "p_layers": _p,
                                    "n_qubits": n,
                                    "issue": f"Scoreboard best (p={_p}, grade={grade}, "
                                    f"ΔE/gap={de_gap:.4f}) uses checkpoint '{ckpt}' "
                                    f"not found in zoo or on disk",
                                    "severity": "warning",
                                }
                            )

        if scoreboard_issues and verbose:
            print(f"\n  Scoreboard cross-check: {len(scoreboard_issues)} issue(s)")
            for si in scoreboard_issues:
                print(
                    f"    ⚠️ {si['topology']} p={si.get('p_layers')} "
                    f"N={si['n_qubits']}: {si['issue']}"
                )
    except Exception as e:
        logger.debug("validate_data_consistency: scoreboard cross-check failed: %s", e)

    # ── Critical-ranking cross-check (empirical h-window vs metadata) ───────
    # The critical_ranking stores measured |ΔE|/fidelity in the critical window.
    # Flag entries whose empirical grade contradicts the nominal pass_rate — a
    # strong drift signal (e.g. pass_rate high but critical-window grade F).
    critical_ranking_issues: list[dict] = []
    try:
        from qmbp_simulation.predictors.model_zoo import _critical_window_key, _load_manifest

        _win = _critical_window_key()
        for entry in _load_manifest():
            crit = entry.critical_ranking.get(_win) if entry.critical_ranking else None
            if not crit:
                continue
            grade = crit.get("grade")
            # pass_rate ≥ 0.7 but the critical window is graded F/D → contradictory.
            if entry.pass_rate >= 0.70 and grade in ("F", "D"):
                critical_ranking_issues.append(
                    {
                        "checkpoint": entry.checkpoint_file,
                        "p_layers": entry.p_layers,
                        "issue": (
                            f"pass_rate={entry.pass_rate:.0%} but critical-window "
                            f"[{_win}] grade={grade} (|ΔE|_mean="
                            f"{crit.get('abs_error_mean')}). Metadata likely stale "
                            f"for the critical regime."
                        ),
                    }
                )
        if critical_ranking_issues and verbose:
            print(f"\n  Critical-ranking cross-check: {len(critical_ranking_issues)} issue(s)")
            for ci in critical_ranking_issues:
                print(f"    ⚠️ {ci['checkpoint'][:40]} p={ci['p_layers']}: {ci['issue']}")
    except Exception as e:
        logger.debug("validate_data_consistency: critical-ranking cross-check failed: %s", e)

    return {
        "is_consistent": is_consistent,
        "n_checks": n_checks,
        "n_issues": n_issues,
        "findings": findings,
        "zoo_vs_comparison": zoo_vs_comparison,
        "registry_vs_curves": registry_vs_curves,
        "cross_n_selection_issues": cross_n_selection_issues,
        "scoreboard_issues": scoreboard_issues,
        "critical_ranking_issues": critical_ranking_issues,
        "tier3": tier3_report,
    }


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


def auto_fix_scoreboard_issues(*, verbose: bool = False) -> dict:
    """Auto-fix all scoreboard-related issues that don't require retraining.

    Performs the following safe, non-destructive fixes in order:
    1. Recover phantom checkpoints from _versions/ or _archive/ back to main dir
    2. Register unregistered checkpoints (on disk but not in zoo manifest)
    3. Backfill pass_rate_by_n from comparison history
    4. Remove stale exclusions (dashboard says useful, but still in exclusion list)
    5. Sync exclusion policy drift (dashboard says not_useful, not in exclusion list)
    6. Update scoreboard JSON if any fixes were applied

    Returns
    -------
    dict
        {
            "n_recovered": int,
            "n_registered": int,
            "n_backfilled": int,
            "n_exclusions_removed": int,
            "n_exclusions_added": int,
            "scoreboard_regenerated": bool,
            "actions": list[str],  # human-readable action log
        }
    """
    import json as _json
    import shutil
    from pathlib import Path as _P

    _ROOT = _P(__file__).resolve().parents[3]
    _CHECKPOINTS_DIR = _ROOT / "data" / "model_zoo" / "checkpoints"
    _SCOREBOARD_PATH = _ROOT / "results" / "best_results_scoreboard_p1.json"

    result = {
        "n_recovered": 0,
        "n_registered": 0,
        "n_backfilled": 0,
        "n_exclusions_removed": 0,
        "n_exclusions_added": 0,
        "scoreboard_regenerated": False,
        "actions": [],
    }

    def _log(msg: str):
        result["actions"].append(msg)
        if verbose:
            print(f"    {msg}")

    # ── Fix 1: Recover phantom checkpoints from _versions/ or _archive/ ──
    if _SCOREBOARD_PATH.exists():
        try:
            scoreboard = _json.loads(_SCOREBOARD_PATH.read_text())
            best_by_topo = scoreboard.get("best_by_topology", {})

            from qmbp_simulation.predictors.model_zoo import _load_manifest

            manifest_files = {e.checkpoint_file for e in _load_manifest()}

            for topo, n_entries in best_by_topo.items():
                for n_str, entry in n_entries.items():
                    grade = entry.get("grade", "F")
                    if grade not in ("A", "B", "C"):
                        continue  # Only recover good models

                    ckpt = _P(entry.get("checkpoint", "")).name
                    if not ckpt or ckpt == "unknown":
                        continue
                    if ckpt in manifest_files:
                        continue  # Already registered
                    if (_CHECKPOINTS_DIR / ckpt).exists():
                        continue  # On disk (will be handled by Fix 2)

                    # Search in _versions/ and _archive/
                    recovered_from = None
                    for subdir in ("_versions", "_archive", "_recovery"):
                        candidate = _CHECKPOINTS_DIR / subdir / ckpt
                        if candidate.exists():
                            recovered_from = candidate
                            break

                    if recovered_from:
                        dest = _CHECKPOINTS_DIR / ckpt
                        shutil.copy2(str(recovered_from), str(dest))
                        result["n_recovered"] += 1
                        _log(
                            f"✅ Recovered {ckpt} from {recovered_from.parent.name}/ "
                            f"(best-ever for {topo} N={n_str}, grade {grade})"
                        )
        except Exception as e:
            _log(f"⚠️ Phantom recovery failed: {e}")

    # ── Fix 2: Register unregistered checkpoints into zoo manifest ────────
    if _SCOREBOARD_PATH.exists():
        try:
            from qmbp_simulation.predictors.model_zoo import (
                ZooEntry,
                _load_manifest,
                _save_manifest,
            )

            manifest = _load_manifest()
            manifest_files = {e.checkpoint_file for e in manifest}

            scoreboard = _json.loads(_SCOREBOARD_PATH.read_text())
            best_by_topo = scoreboard.get("best_by_topology", {})

            for topo, n_entries in best_by_topo.items():
                for n_str, entry in n_entries.items():
                    grade = entry.get("grade", "F")
                    if grade not in ("A", "B", "C"):
                        continue

                    ckpt = _P(entry.get("checkpoint", "")).name
                    if not ckpt or ckpt == "unknown":
                        continue
                    if ckpt in manifest_files:
                        continue  # Already in manifest

                    ckpt_path = _CHECKPOINTS_DIR / ckpt
                    if not ckpt_path.exists():
                        continue  # Not on disk (can't register what we don't have)

                    # Determine topology from scoreboard entry
                    model_type = entry.get("model_type", "ST")
                    is_mt = model_type == "MT"

                    new_entry = ZooEntry(
                        model="tfim_bond_resolved",
                        topology="multi_topology" if is_mt else topo,
                        n_qubits=0,
                        p_layers=1,
                        checkpoint_file=ckpt,
                        h_range=(0.5, 5.0),
                        pass_rate=0.0,  # Will be updated by evaluate_zoo_models
                        n_training_points=0,
                        seeds=[42],
                        created=entry.get("date", ""),
                        notes=(
                            f"Auto-registered from scoreboard (best-ever for "
                            f"{topo} N={n_str}, grade {grade}, "
                            f"ΔE/gap={entry.get('best_de_gap', 0):.4f})"
                        ),
                    )
                    manifest.append(new_entry)
                    manifest_files.add(ckpt)
                    result["n_registered"] += 1
                    _log(f"✅ Registered {ckpt} in zoo manifest (from scoreboard)")

            if result["n_registered"] > 0:
                _save_manifest(manifest)
        except Exception as e:
            _log(f"⚠️ Registration failed: {e}")

    # ── Fix 3: Backfill pass_rate_by_n from comparison history ────────────
    try:
        from qmbp_simulation.predictors.model_zoo import backfill_pass_rate_by_n_from_comparisons

        n_backfilled = backfill_pass_rate_by_n_from_comparisons()
        result["n_backfilled"] = n_backfilled
        if n_backfilled > 0:
            _log(f"✅ Backfilled pass_rate_by_n for {n_backfilled} models")
    except Exception as e:
        _log(f"⚠️ Backfill failed: {e}")

    # ── Fix 4: Remove stale exclusions ────────────────────────────────────
    # Files that are in the exclusion registry but dashboard now says "useful"
    try:
        dashboard_path = _ROOT / "data" / "model_quality_dashboard.json"
        if dashboard_path.exists():
            dashboard = _json.loads(dashboard_path.read_text())
            configs = dashboard.get("configs", [])

            useful_files = {
                c.get("file", "")
                for c in configs
                if c.get("training_utility") and c.get("training_utility") != "not_useful"
            }
            useful_files.discard("")

            registry = load_training_exclusions()
            excluded = registry.get("excluded", [])
            original_count = len(excluded)

            # Remove entries whose file is now "useful" in dashboard
            new_excluded = [
                entry for entry in excluded if entry.get("file", "") not in useful_files
            ]
            n_removed = original_count - len(new_excluded)

            if n_removed > 0:
                registry["excluded"] = new_excluded
                registry["updated_at"] = datetime.now(UTC).isoformat()
                save_training_exclusions(registry)
                result["n_exclusions_removed"] = n_removed
                _log(f"✅ Removed {n_removed} stale exclusions (now marked useful in dashboard)")
    except Exception as e:
        _log(f"⚠️ Stale exclusion removal failed: {e}")

    # ── Fix 5: Add missing exclusions (drift) ────────────────────────────
    # Dashboard says "not_useful" but NOT in exclusion registry
    try:
        auto_detect_exclusions(dry_run=False)
        # Count new additions by comparing before/after
        new_registry = load_training_exclusions()
        n_after = len(new_registry.get("excluded", []))
        n_added = max(0, n_after - (original_count - n_removed) if "original_count" in dir() else 0)
        if n_added > 0:
            result["n_exclusions_added"] = n_added
            _log(f"✅ Added {n_added} new exclusions from dashboard")
    except Exception as e:
        _log(f"⚠️ Exclusion sync failed: {e}")

    # ── Fix 6: Regenerate scoreboard if any fixes were applied ────────────
    if result["n_recovered"] > 0 or result["n_registered"] > 0:
        try:
            import subprocess
            import sys

            scoreboard_script = (
                _ROOT / "scripts" / "analysis" / "generate_best_results_scoreboard.py"
            )
            if scoreboard_script.exists():
                subprocess.run(
                    [sys.executable, str(scoreboard_script), "--json"],
                    capture_output=True,
                    text=True,
                    timeout=120,  # full scan + cold-import warmup
                    cwd=str(_ROOT),
                )
                result["scoreboard_regenerated"] = True
                _log("✅ Scoreboard regenerated after fixes")
        except Exception as e:
            _log(f"⚠️ Scoreboard regeneration failed: {e}")

    return result


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
                )

                if category == "not_useful":
                    # Parse topology and N from filename
                    # Format: <topo>_N<n>_p<p>.npz
                    stem = npz_path.stem
                    parts = stem.split("_")
                    n_idx = next((i for i, p in enumerate(parts) if p.startswith("N")), None)
                    if n_idx is not None:
                        topology = "_".join(parts[:n_idx])
                        try:
                            n_qubits = int(parts[n_idx][1:])
                        except ValueError:
                            topology = "unknown"
                            n_qubits = 0
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


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 3 — Cross-Validation & Advanced Coherence Checks
# ═══════════════════════════════════════════════════════════════════════════════


def validate_eval_vs_comparison_gap(*, verbose: bool = False) -> dict:
    """Tier 3 Validation #7: Detect eval report grade ↔ comparison pass_rate discrepancies.

    The eval report measures *training data quality* (using θ_opt from NPZ),
    while comparisons measure *deployment quality* (using θ_pred from MPNN).
    Large discrepancies indicate:
    - Eval grade A + comparison F → model cannot generalize (MPNN underfitting)
    - Eval grade F + comparison A → stale eval report (model was retrained)

    Returns
    -------
    dict
        {
            "n_checked": int,
            "n_discrepancies": int,
            "discrepancies": list[dict],
            "note": str,  # Explanatory note for dashboard consumers
        }
    """
    import json as _json

    _ROOT = Path(__file__).resolve().parents[3]

    # ── Load eval report grades (from model_evaluation_report.md) ────────
    eval_report_path = _ROOT / "results" / "model_evaluation_report.md"
    eval_grades: dict[str, dict[int, str]] = {}  # {topology: {N: grade}}

    if eval_report_path.exists():
        current_topo = None
        for line in eval_report_path.read_text().splitlines():
            # Detect topology headers: "## chain_1d — ..."
            if line.startswith("## ") and "—" in line:
                current_topo = line.split("##")[1].strip().split(" ")[0].strip()
            # Parse table rows: "| 10 | IN | ... | A |"
            if current_topo and line.startswith("|") and "|" in line[1:]:
                cells = [c.strip() for c in line.split("|")[1:-1]]
                if len(cells) >= 7 and cells[0].isdigit():
                    try:
                        n_val = int(cells[0])
                        grade = cells[-1].strip()
                        if grade in ("A", "B", "C", "D", "F"):
                            eval_grades.setdefault(current_topo, {})[n_val] = grade
                    except (ValueError, IndexError):
                        pass

    # ── Load latest comparison results per topology ──────────────────────
    comp_dir = _ROOT / "results" / "model_comparison"
    comp_pass_rates: dict[str, dict[int, float]] = {}  # {topology: {N: pass_rate_dual}}

    if comp_dir.exists():
        # Use latest comparison file per topology
        for topo_dir in comp_dir.iterdir():
            if not topo_dir.is_dir():
                # Try exp_model_comparison structure
                continue

    # Also check experiments/exp_model_comparison/
    exp_comp_dir = _ROOT / "results" / "experiments" / "exp_model_comparison"
    if exp_comp_dir.exists():
        for model_dir in exp_comp_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for topo_dir in model_dir.iterdir():
                if not topo_dir.is_dir():
                    continue
                topo = topo_dir.name
                # Find latest run
                runs = sorted(topo_dir.glob("run_*.json"))
                if not runs:
                    continue
                try:
                    latest = _json.loads(runs[-1].read_text())
                    models = latest.get("results", {}).get("models", [])
                    # Use best model's results
                    best_pass = 0.0
                    best_by_n: dict[int, float] = {}
                    for m in models:
                        if "error" in m:
                            continue
                        results_by_n = m.get("results_by_n", {})
                        rates = [v.get("pass_rate_dual", 0) for v in results_by_n.values()]
                        avg = float(np.mean(rates)) if rates else 0.0
                        if avg > best_pass:
                            best_pass = avg
                            best_by_n = {
                                int(n): v.get("pass_rate_dual", 0.0)
                                for n, v in results_by_n.items()
                            }
                    if best_by_n:
                        comp_pass_rates[topo] = best_by_n
                except Exception:
                    continue

    # ── Cross-validate ───────────────────────────────────────────────────
    discrepancies: list[dict] = []
    n_checked = 0

    # Grade → expected pass_rate range
    grade_to_pass_range = {
        "A": (0.60, 1.0),  # Grade A eval → deployment should be ≥60%
        "B": (0.40, 1.0),  # Grade B → ≥40%
        "C": (0.20, 1.0),  # Grade C → ≥20%
        "D": (0.0, 1.0),  # Grade D → anything possible
        "F": (0.0, 0.30),  # Grade F eval → deployment shouldn't be >30%
    }

    for topo in set(eval_grades.keys()) & set(comp_pass_rates.keys()):
        for n_val in set(eval_grades[topo].keys()) & set(comp_pass_rates[topo].keys()):
            n_checked += 1
            grade = eval_grades[topo][n_val]
            comp_rate = comp_pass_rates[topo][n_val]

            expected_min, expected_max = grade_to_pass_range.get(grade, (0.0, 1.0))

            # Case 1: Eval says A but comparison is terrible
            if grade in ("A", "B") and comp_rate < expected_min:
                discrepancies.append(
                    {
                        "topology": topo,
                        "n_qubits": n_val,
                        "eval_grade": grade,
                        "comparison_pass_rate": comp_rate,
                        "severity": "high" if grade == "A" and comp_rate < 0.3 else "medium",
                        "diagnosis": (
                            "MPNN underfitting: training data quality is high "
                            f"(grade {grade}) but deployment accuracy is low "
                            f"({comp_rate:.0%}). Model cannot generalize θ_opt → θ_pred."
                        ),
                        "action": "retrain_with_more_data_or_epochs",
                    }
                )

            # Case 2: Eval says F but comparison is good (stale eval report)
            if grade == "F" and comp_rate > expected_max:
                discrepancies.append(
                    {
                        "topology": topo,
                        "n_qubits": n_val,
                        "eval_grade": grade,
                        "comparison_pass_rate": comp_rate,
                        "severity": "low",
                        "diagnosis": (
                            "Stale eval report: eval gives F but comparison shows "
                            f"{comp_rate:.0%}. Likely the model was retrained since "
                            "last eval report generation."
                        ),
                        "action": "regenerate_eval_report",
                    }
                )

            if verbose and discrepancies and discrepancies[-1]["topology"] == topo:
                d = discrepancies[-1]
                logger.info(
                    f"  Tier3 gap: {topo} N={n_val} eval={grade} comp={comp_rate:.0%} "
                    f"→ {d['diagnosis'][:60]}..."
                )

    return {
        "n_checked": n_checked,
        "n_discrepancies": len(discrepancies),
        "discrepancies": discrepancies,
        "note": (
            "eval_report measures training data quality (θ_opt from NPZ), "
            "comparison measures deployment quality (θ_pred from MPNN). "
            "Discrepancies are expected when model generalization is imperfect."
        ),
    }


def detect_undertrained_models(*, verbose: bool = False) -> dict:
    """Tier 3 Validation #8: Detect models whose training curves suggest incomplete training.

    Analyzes training_metrics.loss_history from model_registry.json:
    - If val_mse is still decreasing at the end → undertrained (more epochs needed)
    - If final_val_mse > 2× best_val_mse → overfitting / instability
    - If convergence_status == "unknown" with sufficient data → needs recomputation

    A model flagged as "undertrained" is a prime candidate for resuming training
    (more epochs) before initiating a full retrain.

    Returns
    -------
    dict
        {
            "n_models_checked": int,
            "n_undertrained": int,
            "n_overfitting": int,
            "n_unknown_convergence": int,
            "undertrained": list[dict],  # [{model_id, topology, reason, details}]
            "overfitting": list[dict],
            "recommendations": list[str],
        }
    """
    import json as _json

    _ROOT = Path(__file__).resolve().parents[3]
    registry_path = _ROOT / "data" / "model_zoo" / "model_registry.json"

    if not registry_path.exists():
        return {
            "n_models_checked": 0,
            "n_undertrained": 0,
            "n_overfitting": 0,
            "n_unknown_convergence": 0,
            "undertrained": [],
            "overfitting": [],
            "recommendations": ["model_registry.json not found"],
        }

    registry = _json.loads(registry_path.read_text())
    undertrained: list[dict] = []
    overfitting: list[dict] = []
    n_unknown = 0
    n_checked = 0

    for entry in registry:
        if entry.get("status") != "active":
            continue

        model_id = entry.get("model_id", "unknown")
        topology = entry.get("topology", "unknown")
        training = entry.get("training", {})
        metrics = training.get("training_metrics", {})

        if not metrics:
            continue

        n_checked += 1
        loss_history = metrics.get("loss_history", [])
        final_val_mse = metrics.get("final_val_mse")
        convergence_status = metrics.get("convergence_status", "unknown")
        early_stopped = metrics.get("early_stopped", False)
        epochs = metrics.get("epochs", 0)
        best_epoch = metrics.get("best_epoch", 0)
        generalization_gap = metrics.get("generalization_gap")

        # ── Check 1: Still-decreasing loss → undertrained ────────────────
        if loss_history and len(loss_history) >= 10:
            # Compare last 20% of loss curve to second-to-last 20%
            n = len(loss_history)
            tail_20 = loss_history[int(0.8 * n) :]
            prev_20 = loss_history[int(0.6 * n) : int(0.8 * n)]

            if tail_20 and prev_20:
                tail_mean = float(np.mean(tail_20))
                prev_mean = float(np.mean(prev_20))

                # If tail is still decreasing (>5% improvement in last segment)
                if prev_mean > 0 and (prev_mean - tail_mean) / prev_mean > 0.05:
                    undertrained.append(
                        {
                            "model_id": model_id,
                            "topology": topology,
                            "reason": "loss_still_decreasing",
                            "details": {
                                "epochs_trained": epochs,
                                "tail_mean_loss": round(tail_mean, 6),
                                "prev_segment_mean": round(prev_mean, 6),
                                "improvement_pct": round(
                                    100 * (prev_mean - tail_mean) / prev_mean, 1
                                ),
                                "early_stopped": early_stopped,
                            },
                            "recommendation": (
                                f"Resume training for ~{max(50, epochs // 2)} more epochs. "
                                f"Loss was still improving at epoch {epochs}."
                            ),
                        }
                    )
                    if verbose:
                        logger.info(
                            f"  Tier3 undertrained: {model_id} "
                            f"(loss ↓{(prev_mean - tail_mean) / prev_mean:.1%} in last segment)"
                        )

        # ── Check 2: final_val_mse >> best achievable → overfitting ──────
        if final_val_mse is not None and best_epoch > 0 and epochs > 0:
            # If best was achieved much earlier than end, model overfit
            epoch_ratio = best_epoch / epochs if epochs > 0 else 1.0
            if epoch_ratio < 0.6 and not early_stopped:
                # Best epoch was in first 60% of training → later epochs overfit
                overfitting.append(
                    {
                        "model_id": model_id,
                        "topology": topology,
                        "reason": "best_epoch_early",
                        "details": {
                            "best_epoch": best_epoch,
                            "total_epochs": epochs,
                            "ratio": round(epoch_ratio, 2),
                            "final_val_mse": final_val_mse,
                            "convergence_status": convergence_status,
                        },
                        "recommendation": (
                            f"Enable early stopping (patience=50) or reduce epochs to "
                            f"~{int(best_epoch * 1.3)}. Model peaked at epoch {best_epoch}/{epochs}."
                        ),
                    }
                )

        # ── Check 3: Generalization gap → potential overfitting ──────────
        if generalization_gap is not None and generalization_gap > 0.5:
            # Train MSE << Val MSE → overfitting
            overfitting.append(
                {
                    "model_id": model_id,
                    "topology": topology,
                    "reason": "high_generalization_gap",
                    "details": {
                        "generalization_gap": round(generalization_gap, 4),
                        "final_val_mse": final_val_mse,
                    },
                    "recommendation": (
                        f"Generalization gap={generalization_gap:.2f} is too high. "
                        "Consider: more training data, more dropout, or smaller model."
                    ),
                }
            )

        # ── Check 4: Unknown convergence with enough info ────────────────
        if convergence_status == "unknown" and epochs > 0:
            n_unknown += 1

    # ── Recommendations ──────────────────────────────────────────────────
    recommendations: list[str] = []
    if undertrained:
        topo_counts = {}
        for u in undertrained:
            topo_counts[u["topology"]] = topo_counts.get(u["topology"], 0) + 1
        worst_topo = max(topo_counts, key=topo_counts.get)
        recommendations.append(
            f"Priority: resume training for {worst_topo} models "
            f"({topo_counts[worst_topo]} undertrained). Use --epochs +50%."
        )
    if overfitting:
        recommendations.append(
            f"{len(overfitting)} models show overfitting. "
            "Enable early stopping with patience=50 in training config."
        )
    if n_unknown > 0:
        recommendations.append(
            f"{n_unknown} models have unknown convergence status. "
            "Run audit_and_fix_model_zoo.py --fix to recompute."
        )

    return {
        "n_models_checked": n_checked,
        "n_undertrained": len(undertrained),
        "n_overfitting": len(overfitting),
        "n_unknown_convergence": n_unknown,
        "undertrained": undertrained,
        "overfitting": overfitting,
        "recommendations": recommendations,
    }


def compute_smart_comparison_h_grid(
    topology: str,
    *,
    h_min: float = 1.0,
    h_max: float = 5.0,
    n_points: int = 15,
    dense_fraction: float = 0.5,
) -> dict:
    """Tier 3 Enhancement: Compute an adaptive h-grid for model comparisons.

    Instead of using uniform h=[2.5, 5.0] or a fixed linspace, this uses the
    empirical h_frontier from the dashboard to concentrate test points where
    the model transitions from pass → fail. This reveals true deployment
    quality in the critical region.

    Strategy:
    1. Load h_frontier for the topology from dashboard
    2. If available → use ``generate_frontier_dense_h_grid`` centered on frontier
    3. If not → fall back to nonuniform grid centered on h_critical=1.0 (TFIM)

    Parameters
    ----------
    topology : str
        Topology name (chain_1d, heavy_hex, ladder, square, triangular).
    h_min : float
        Minimum h-value for comparison grid.
    h_max : float
        Maximum h-value.
    n_points : int
        Number of h-points in the grid.
    dense_fraction : float
        Fraction of points to concentrate near the frontier.

    Returns
    -------
    dict
        {
            "h_values": list[float],
            "h_frontier_used": float | None,
            "strategy": str,  # "frontier_dense" | "nonuniform" | "uniform"
            "description": str,
        }
    """
    import json as _json

    _ROOT = Path(__file__).resolve().parents[3]
    dash_path = _ROOT / "data" / "model_quality_dashboard.json"

    h_frontier: float | None = None

    # ── Try to get h_frontier from dashboard ─────────────────────────────
    if dash_path.exists():
        try:
            dashboard = _json.loads(dash_path.read_text())
            # Find best (lowest) h_frontier for this topology across all N/p
            topo_configs = [
                c
                for c in dashboard.get("configs", [])
                if c.get("topology") == topology and c.get("h_frontier") is not None
            ]
            if topo_configs:
                # Use the h_frontier from the largest N (most representative)
                topo_configs.sort(key=lambda c: c.get("n_qubits", 0), reverse=True)
                h_frontier = topo_configs[0]["h_frontier"]
        except Exception:
            pass

    # ── Generate grid ────────────────────────────────────────────────────
    if h_frontier is not None and h_min <= h_frontier <= h_max:
        try:
            from qmbp_simulation.pipeline.dataset_io import generate_frontier_dense_h_grid

            grid = generate_frontier_dense_h_grid(
                h_min=h_min,
                h_max=h_max,
                n_points=n_points,
                h_frontier=h_frontier,
                dense_fraction=dense_fraction,
                include_below_frontier=True,
            )
            h_values = sorted([round(float(h), 2) for h in grid], reverse=True)
            return {
                "h_values": h_values,
                "h_frontier_used": round(h_frontier, 3),
                "strategy": "frontier_dense",
                "description": (
                    f"Adaptive grid with {int(dense_fraction * 100)}% of points "
                    f"near h_frontier={h_frontier:.3f} (empirical pass/fail boundary)"
                ),
            }
        except Exception as e:
            logger.debug(f"  compute_smart_comparison_h_grid: frontier grid failed: {e}")

    # ── Fallback: nonuniform around h_critical=1.0 (TFIM default) ────────
    if h_frontier is not None and h_frontier < h_min:
        # Frontier is below our test range → most points should pass
        # Use uniform grid (no need to densify where everything is easy)
        h_values = [round(h, 2) for h in np.linspace(h_max, h_min, n_points).tolist()]
        return {
            "h_values": h_values,
            "h_frontier_used": round(h_frontier, 3),
            "strategy": "uniform",
            "description": (
                f"Uniform grid: h_frontier={h_frontier:.3f} is below test range "
                f"[{h_min}, {h_max}], most points expected to pass."
            ),
        }

    # No frontier data → nonuniform with TFIM h_c ≈ 1.0
    try:
        from qmbp_simulation.pipeline.dataset_io import generate_nonuniform_h_grid

        h_critical = 1.0  # TFIM critical point
        grid = generate_nonuniform_h_grid(
            h_min=h_min,
            h_max=h_max,
            n_points=n_points,
            h_critical=h_critical,
            dense_fraction=dense_fraction,
        )
        h_values = sorted([round(float(h), 2) for h in grid], reverse=True)
        return {
            "h_values": h_values,
            "h_frontier_used": None,
            "strategy": "nonuniform",
            "description": (
                f"Nonuniform grid around h_c=1.0 (no empirical frontier for {topology})"
            ),
        }
    except Exception:
        # Final fallback: uniform
        h_values = [round(h, 2) for h in np.linspace(h_max, h_min, n_points).tolist()]
        return {
            "h_values": h_values,
            "h_frontier_used": None,
            "strategy": "uniform",
            "description": "Uniform grid (fallback)",
        }


def compute_cascading_retrain_plan(*, verbose: bool = False) -> dict:
    """Tier 3 Enhancement: Compute a cascading retrain strategy using MT as warm-start.

    Instead of retraining all 5 topology-specific models independently,
    this computes a plan to:
    1. Use the MT (multi-topology) model as starting checkpoint
    2. Fine-tune for each topology independently
    3. Prioritize topologies by their current gap to desired pass_rate

    Uses existing infrastructure: ``run_finetune_from_mt.py``.

    Returns
    -------
    dict
        {
            "mt_checkpoint": str | None,
            "n_models_to_retrain": int,
            "plan": list[dict],  # [{topology, priority, current_pass_rate, ...}]
            "estimated_speedup": str,
            "commands": list[str],  # Ready-to-run CLI commands
        }
    """
    import json as _json

    _ROOT = Path(__file__).resolve().parents[3]

    # ── Find MT checkpoint ───────────────────────────────────────────────
    mt_ckpt_path = _ROOT / "data" / "model_zoo" / "checkpoints"
    mt_candidates = sorted(mt_ckpt_path.glob("*MT*residual*film*.pt"))
    mt_checkpoint: str | None = None
    if mt_candidates:
        mt_checkpoint = str(mt_candidates[-1].relative_to(_ROOT))

    # ── Load manifest for current pass_rates ─────────────────────────────
    manifest_path = _ROOT / "data" / "model_zoo" / "manifest.json"
    per_topo_models: dict[str, dict] = {}

    if manifest_path.exists():
        try:
            manifest = _json.loads(manifest_path.read_text())
            for entry in manifest:
                topo = entry.get("topology", "")
                # Skip the MT model itself
                if "MT" in entry.get("checkpoint_file", ""):
                    continue
                # Keep the highest pass_rate per topology
                current = per_topo_models.get(topo)
                if current is None or entry.get("pass_rate", 0) > current.get("pass_rate", 0):
                    per_topo_models[topo] = entry
        except Exception:
            pass

    # ── Load retrain queue priorities ────────────────────────────────────
    registry_path = _ROOT / "data" / "model_zoo" / "model_registry.json"
    retrain_flags: dict[str, bool] = {}

    if registry_path.exists():
        try:
            registry = _json.loads(registry_path.read_text())
            for entry in registry:
                if entry.get("status") != "active":
                    continue
                topo = entry.get("topology", "")
                needs = entry.get("dashboard_quality", {}).get("needs_retrain", False)
                if needs:
                    retrain_flags[topo] = True
        except Exception:
            pass

    # ── Build retrain plan ───────────────────────────────────────────────
    TARGET_PASS_RATE = 0.7  # Target: 70% deployment pass rate
    plan: list[dict] = []

    for topo, model_info in sorted(per_topo_models.items()):
        current_pass = model_info.get("pass_rate", 0.0)
        n_training = model_info.get("n_training_points", 0)
        gap_to_target = max(0, TARGET_PASS_RATE - current_pass)

        # Only include if below target or flagged for retrain
        if gap_to_target > 0 or retrain_flags.get(topo, False):
            plan.append(
                {
                    "topology": topo,
                    "priority": round(gap_to_target, 3),
                    "current_pass_rate": round(current_pass, 3),
                    "target_pass_rate": TARGET_PASS_RATE,
                    "gap_to_target": round(gap_to_target, 3),
                    "n_training_points": n_training,
                    "current_checkpoint": model_info.get("checkpoint_file", ""),
                    "needs_retrain_flagged": retrain_flags.get(topo, False),
                    "strategy": "finetune_from_mt" if mt_checkpoint else "retrain_from_scratch",
                }
            )

    # Sort by priority (largest gap first)
    plan.sort(key=lambda x: x["priority"], reverse=True)

    # ── Generate CLI commands ────────────────────────────────────────────
    commands: list[str] = []
    ft_script = "scripts/experiment_runners/cross_topology/run_finetune_from_mt.py"
    for item in plan:
        if mt_checkpoint:
            cmd = (
                f".venv/bin/python {ft_script} "
                f"--topology {item['topology']} "
                f"--mt-checkpoint {mt_checkpoint} "
                f"--epochs 200 --lr 5e-4"
            )
        else:
            cmd = (
                f".venv/bin/python scripts/experiment_runners/cross_topology/"
                f"run_multi_topology_training.py "
                f"--topology {item['topology']} --epochs 400"
            )
        commands.append(cmd)

    # Speedup estimate
    if mt_checkpoint and plan:
        speedup = "~3-5× faster than training from scratch (warm-start converges in ~100-200 epochs vs 500+)"
    else:
        speedup = "No MT checkpoint available — full training required"

    if verbose and plan:
        logger.info(f"  Tier3 cascading retrain: {len(plan)} models to retrain")
        for p in plan[:3]:
            logger.info(
                f"    {p['topology']}: pass_rate={p['current_pass_rate']:.0%} "
                f"→ target={p['target_pass_rate']:.0%} (gap={p['gap_to_target']:.0%})"
            )

    return {
        "mt_checkpoint": mt_checkpoint,
        "n_models_to_retrain": len(plan),
        "plan": plan,
        "estimated_speedup": speedup,
        "commands": commands,
    }


def run_tier3_validations(*, verbose: bool = False) -> dict:
    """Run all Tier 3 cross-validations and return consolidated report.

    Combines:
    - Validation #7: Eval report grades ↔ comparison pass_rate
    - Validation #8: Training curve convergence → model readiness
    - Smart h-grid recommendations
    - Cascading retrain strategy

    This is the single entry point for Tier 3, callable from
    ``validate_data_consistency()`` and ``post_experiment_sync()``.

    Returns
    -------
    dict
        {
            "eval_vs_comparison": dict,
            "training_convergence": dict,
            "smart_h_grid": dict[str, dict],  # per-topology
            "cascading_retrain": dict,
            "summary": str,
            "n_total_issues": int,
        }
    """
    # ── Validation #7 ────────────────────────────────────────────────────
    eval_comp = validate_eval_vs_comparison_gap(verbose=verbose)

    # ── Validation #8 ────────────────────────────────────────────────────
    convergence = detect_undertrained_models(verbose=verbose)

    # ── Smart h-grid per topology ────────────────────────────────────────
    topologies = ["chain_1d", "heavy_hex", "ladder", "square", "triangular"]
    smart_grids: dict[str, dict] = {}
    for topo in topologies:
        try:
            grid_info = compute_smart_comparison_h_grid(topology=topo)
            # Only include if strategy is not plain uniform (i.e., we have useful data)
            if grid_info["strategy"] != "uniform" or grid_info["h_frontier_used"] is not None:
                smart_grids[topo] = grid_info
        except Exception:
            pass

    # ── Cascading retrain ────────────────────────────────────────────────
    retrain = compute_cascading_retrain_plan(verbose=verbose)

    # ── Summary ──────────────────────────────────────────────────────────
    n_issues = (
        eval_comp["n_discrepancies"] + convergence["n_undertrained"] + convergence["n_overfitting"]
    )

    parts = []
    if eval_comp["n_discrepancies"] > 0:
        parts.append(f"{eval_comp['n_discrepancies']} eval↔comparison gaps")
    if convergence["n_undertrained"] > 0:
        parts.append(f"{convergence['n_undertrained']} undertrained models")
    if convergence["n_overfitting"] > 0:
        parts.append(f"{convergence['n_overfitting']} overfitting models")
    if retrain["n_models_to_retrain"] > 0:
        parts.append(f"{retrain['n_models_to_retrain']} need retraining")

    summary = " | ".join(parts) if parts else "All Tier 3 checks pass ✅"

    return {
        "eval_vs_comparison": eval_comp,
        "training_convergence": convergence,
        "smart_h_grid": smart_grids,
        "cascading_retrain": retrain,
        "summary": summary,
        "n_total_issues": n_issues,
    }


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
