"""Analysis Constants — Single source of truth for quality criteria thresholds.

This module is the LOWEST-LEVEL module in the analysis subpackage.
It has NO imports from any qmbp_simulation module (only stdlib/builtins),
which makes it safe to import from anywhere without circular dependencies.

All quality thresholds, metric boundaries, and diagnostic constants live here.
Other modules (metrics.py, failures_tests.py, cross_n_validator.py) import
from this file rather than defining their own copies.

Usage:
    from qmbp_simulation.analysis.constants import DE_GAP_THRESHOLD, MAX_ABS_ERROR
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# Dual Quality Criterion (pass/fail decisions)
# ═══════════════════════════════════════════════════════════════════════════════

DE_GAP_THRESHOLD: float = 0.05
"""Relative energy error threshold: |ΔE|/gap < 5%.
A point passes if its relative error normalized by the spectral gap
is below this threshold. This is criterion 1 of the dual criterion."""

MAX_ABS_ERROR: float = 0.10
"""Absolute energy error cap (Hartree). Points above this are failures
regardless of ΔE/gap. Prevents gap masking: for large gaps (h >> h_c),
ΔE/gap can be small while |ΔE| is physically significant."""

MIN_FIDELITY: float = 0.97
"""Minimum state overlap for acceptable quality (noiseless only).
Fidelity check is skipped when N > 22 (statevector unavailable)."""

# ═══════════════════════════════════════════════════════════════════════════════
# Infidelity Decomposition (Var(H) vs gap) — diagnostic, not pass/fail
# ═══════════════════════════════════════════════════════════════════════════════

DIRTY_STATE_VARIANCE_THRESHOLD: float = 0.05
"""Var(H) above this flags a 'dirty state' — |ψ⟩ is far from an eigenstate,
so the infidelity is dominated by preparation/optimization error (ATTACKABLE
via more restarts, warm-start, or a variance-based objective).
Var(H) = ⟨H²⟩ − ⟨H⟩² is exactly zero for an eigenstate."""

SMALL_GAP_THRESHOLD: float = 0.5
"""Spectral gap below this flags 'small gap' — near a critical point the gap
Δ → 0 amplifies the Eckart term Var(H)/Δ² even for a clean state. When the
dominant factor is small_gap with low Var(H), the infidelity is a PHYSICS
ceiling (criticality), not an optimization failure."""

# ═══════════════════════════════════════════════════════════════════════════════
# Data Quality Audit Thresholds
# ═══════════════════════════════════════════════════════════════════════════════

TRAINING_BAD_RATIO_THRESHOLD: float = 0.20
"""If n_bad/n_total > 20% in a NPZ but zoo model has pass_rate > 0.60,
flag training/zoo incoherence."""

ZOO_PASS_FOR_INCOHERENCE_FLAG: float = 0.60
"""Zoo pass_rate above which a high bad-ratio in NPZ is flagged as incoherence."""

PASS_RATE_REGRESSION_THRESHOLD: float = 0.15
"""If max pass_rate_dual for a topology drops > 15% vs previous dashboard,
flag as regression."""

H_FRONTIER_MONOTONICITY_TOLERANCE: float = 0.10
"""h_frontier should increase (or stay flat) with N. If it drops more than
this tolerance, flag as data quality anomaly."""

GAP_MASKING_THRESHOLD: float = 0.10
"""Difference between pass_rate_5pct and pass_rate_dual above which
gap masking is considered significant."""

MIN_TRAINING_DUAL_PASS_RATE: float = 0.30
"""Minimum fraction of points passing dual criterion for a NPZ to be
considered useful for MPNN training."""

MIN_TRAINING_POINTS_FOR_SIGNAL: int = 5
"""Minimum absolute count of dual-passing points for a NPZ to contribute
useful training signal."""

AUTO_EXCLUDE_MEAN_ABS_ERROR: float = 0.20
"""If mean |ΔE| across all points exceeds this, AND pass_rate_dual < 20%,
auto-classify as 'not_useful' regardless of raw pass_rate_dual > 0.
This catches cases where most points have large errors but a few
pass the relative criterion due to large gaps (gap masking)."""

AUTO_EXCLUDE_MAX_DUAL_PASS_FOR_MEAN_ERROR: float = 0.20
"""Maximum dual pass rate to trigger the mean-error exclusion rule.
If pass_dual < this AND mean |ΔE| > AUTO_EXCLUDE_MEAN_ABS_ERROR,
the NPZ is auto-excluded."""


# ═══════════════════════════════════════════════════════════════════════════════
# Continuous Quality Scoring (QualityProfile system)
# ═══════════════════════════════════════════════════════════════════════════════
# These constants and pure functions define the continuous scoring system.
# They live here (lowest-level module, no deps) to break the circular
# dependency between analysis.metrics and framework.quality_profile.

SCORE_DE_GAP_SCALE: float = 0.05
"""Sigmoid scale for mean ΔE/gap. At this value, score_dg = 0.50.
Calibrated so that DE_GAP_THRESHOLD (5%) maps to the B/C boundary."""

SCORE_P90_SCALE: float = 0.10
"""Sigmoid scale for P90 ΔE/gap. At this value, score_p90 = 0.50."""

SCORE_PER_SITE_SCALE: float = 0.02
"""Sigmoid scale for |ΔE|/N. At this value, score_ps = 0.50."""

SCORE_WEIGHT_MEAN: float = 0.50
"""Weight of mean ΔE/gap in the composite quality score."""

SCORE_WEIGHT_P90: float = 0.30
"""Weight of P90 (worst-case) in the composite quality score."""

SCORE_WEIGHT_PER_SITE: float = 0.20
"""Weight of extensivity (|ΔE|/N) in the composite quality score."""

SCORE_MIN_POINTS_FULL_CONFIDENCE: int = 8
"""Minimum n_points for full confidence. Below this, score is scaled
by sqrt(n_points / MIN_POINTS) for a soft penalty."""

GRADE_A_THRESHOLD: float = 0.85
"""Quality score >= this -> Grade A (excellent, mean dE/gap < ~2%)."""

GRADE_B_THRESHOLD: float = 0.65
"""Quality score >= this -> Grade B (good, mean dE/gap < ~5%)."""

GRADE_C_THRESHOLD: float = 0.45
"""Quality score >= this -> Grade C (acceptable, mean dE/gap < ~10%)."""

GRADE_D_THRESHOLD: float = 0.25
"""Quality score >= this -> Grade D (poor, significant errors)."""


def compute_quality_score(
    mean_de_gap: float,
    p90_de_gap: float,
    mean_abs_error_per_site: float | None,
    n_points: int,
) -> float:
    """Compute a continuous quality score in [0, 1]. Higher = better.

    Uses smooth sigmoid mapping: 1/(1 + (x/scale)^2).
    Pure function, no heavy imports (only stdlib math).
    """
    import math

    if not math.isfinite(mean_de_gap) or not math.isfinite(p90_de_gap):
        return 0.0
    # Negative values are physically invalid (variational violation)
    if mean_de_gap < 0 or p90_de_gap < 0:
        return 0.0

    score_dg = 1.0 / (1.0 + (mean_de_gap / SCORE_DE_GAP_SCALE) ** 2)
    score_p90 = 1.0 / (1.0 + (p90_de_gap / SCORE_P90_SCALE) ** 2)

    score_ps = 1.0
    if mean_abs_error_per_site is not None and math.isfinite(mean_abs_error_per_site):
        score_ps = 1.0 / (1.0 + (mean_abs_error_per_site / SCORE_PER_SITE_SCALE) ** 2)

    raw_score = (
        SCORE_WEIGHT_MEAN * score_dg
        + SCORE_WEIGHT_P90 * score_p90
        + SCORE_WEIGHT_PER_SITE * score_ps
    )

    confidence = min(1.0, (n_points / SCORE_MIN_POINTS_FULL_CONFIDENCE) ** 0.5)
    return min(1.0, max(0.0, confidence * raw_score))


def grade_from_score(score: float) -> str:
    """Map quality score [0,1] to letter grade. Display only, never for logic."""
    if score >= GRADE_A_THRESHOLD:
        return "A"
    if score >= GRADE_B_THRESHOLD:
        return "B"
    if score >= GRADE_C_THRESHOLD:
        return "C"
    if score >= GRADE_D_THRESHOLD:
        return "D"
    return "F"
