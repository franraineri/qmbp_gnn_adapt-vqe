"""QualityProfile — Continuous quality representation for experiment evaluation.

Replaces binary pass/fail as the PRIMARY metric for:
- Summary tables and logs
- Project status tracking
- Temporal comparison across runs
- Model zoo ranking

Design principles:
- pass_rate is preserved internally (useful for stopping criteria, training triage)
- All presentation/reporting uses continuous scores and distributions
- Backward compatible: adds fields, never removes
- Reuses constants from analysis.constants (single source of truth)
- Reuses compute_deploy_summary (wraps it, never duplicates)

Usage:
    from qmbp_simulation.framework.quality_profile import (
        QualityProfile,
        compute_quality_profile,
        grade_from_score,
        format_quality_summary,
    )

    profile = compute_quality_profile(per_h_results)
    print(profile.grade, profile.quality_score)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Re-export score constants for consumers that import from here
from qmbp_simulation.analysis.constants import (  # noqa: F401
    DE_GAP_THRESHOLD,
    GRADE_A_THRESHOLD,
    GRADE_B_THRESHOLD,
    GRADE_C_THRESHOLD,
    GRADE_D_THRESHOLD,
    MAX_ABS_ERROR,
    SCORE_DE_GAP_SCALE,
    SCORE_MIN_POINTS_FULL_CONFIDENCE,
    SCORE_P90_SCALE,
    SCORE_PER_SITE_SCALE,
    SCORE_WEIGHT_MEAN,
    SCORE_WEIGHT_P90,
    SCORE_WEIGHT_PER_SITE,
    compute_quality_score,
    grade_from_score,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Note: Score constants (SCORE_DE_GAP_SCALE, GRADE_*_THRESHOLD, etc.) and pure
# functions (compute_quality_score, grade_from_score) live in
# analysis.constants — the lowest-level module with no deps. They are imported
# above and re-exported here for backward compatibility.
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# QualityProfile dataclass
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class QualityProfile:
    """Continuous quality representation for a set of per-h evaluation results.

    Captures the full distribution of errors (not just pass/fail counts),
    enabling meaningful temporal tracking, model comparison, and
    human-readable reporting.

    Attributes
    ----------
    n_points : int
        Number of evaluated h-points.
    mean_de_gap : float
        Mean relative error ΔE/gap across all points.
    median_de_gap : float
        Median ΔE/gap (robust to outliers).
    std_de_gap : float
        Standard deviation of ΔE/gap.
    p90_de_gap : float
        90th percentile of ΔE/gap (reasonable worst-case).
    max_de_gap : float
        Maximum ΔE/gap (absolute worst case, sensitive to outliers).
    mean_abs_error : float | None
        Mean |ΔE| (absolute energy error).
    mean_abs_error_per_site : float | None
        Mean |ΔE|/N (extensivity check).
    mean_fidelity : float | None
        Mean state fidelity (None if N > 22).
    quality_score : float
        Composite continuous score in [0, 1]. Higher is better.
    grade : str
        Letter grade (A/B/C/D/F) for quick display. NOT for logic decisions.
    pass_rate_5pct : float
        Fraction with ΔE/gap < 5%. Kept for stopping criteria only.
    pass_rate_dual : float
        Fraction passing dual criterion. Kept for training triage only.
    critical_region_mean_de_gap : float | None
        Mean ΔE/gap for points near h_c (if detectable).
    ordered_region_mean_de_gap : float | None
        Mean ΔE/gap for points far from h_c (h >> h_c).
    de_gap_distribution : dict[str, float]
        Percentile distribution {p10, p25, p50, p75, p90, p95}.
    """

    # ── Distribution metrics (primary for reporting) ──
    n_points: int
    mean_de_gap: float
    median_de_gap: float
    std_de_gap: float
    p90_de_gap: float
    max_de_gap: float

    # ── Absolute error ──
    mean_abs_error: float | None = None
    mean_abs_error_per_site: float | None = None

    # ── Fidelity ──
    mean_fidelity: float | None = None

    # ── Composite ──
    quality_score: float = 0.0
    grade: str = "F"

    # ── Binary (internal control flow only) ──
    pass_rate_5pct: float = 0.0
    pass_rate_dual: float = 0.0

    # ── Regional breakdown ──
    critical_region_mean_de_gap: float | None = None
    ordered_region_mean_de_gap: float | None = None

    # ── Full distribution ──
    de_gap_distribution: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict for result envelopes."""
        from qmbp_simulation.utils.helpers import json_serialize

        d = {
            "n_points": self.n_points,
            "mean_de_gap": json_serialize(self.mean_de_gap),
            "median_de_gap": json_serialize(self.median_de_gap),
            "std_de_gap": json_serialize(self.std_de_gap),
            "p90_de_gap": json_serialize(self.p90_de_gap),
            "max_de_gap": json_serialize(self.max_de_gap),
            "mean_abs_error": json_serialize(self.mean_abs_error),
            "mean_abs_error_per_site": json_serialize(self.mean_abs_error_per_site),
            "mean_fidelity": json_serialize(self.mean_fidelity),
            "quality_score": json_serialize(self.quality_score),
            "grade": self.grade,
            "pass_rate_5pct": json_serialize(self.pass_rate_5pct),
            "pass_rate_dual": json_serialize(self.pass_rate_dual),
            "critical_region_mean_de_gap": json_serialize(self.critical_region_mean_de_gap),
            "ordered_region_mean_de_gap": json_serialize(self.ordered_region_mean_de_gap),
            "de_gap_distribution": {
                k: json_serialize(v) for k, v in self.de_gap_distribution.items()
            },
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QualityProfile:
        """Reconstruct from a JSON-deserialized dict."""
        return cls(
            n_points=d.get("n_points", 0),
            mean_de_gap=d.get("mean_de_gap", 0.0),
            median_de_gap=d.get("median_de_gap", 0.0),
            std_de_gap=d.get("std_de_gap", 0.0),
            p90_de_gap=d.get("p90_de_gap", 0.0),
            max_de_gap=d.get("max_de_gap", 0.0),
            mean_abs_error=d.get("mean_abs_error"),
            mean_abs_error_per_site=d.get("mean_abs_error_per_site"),
            mean_fidelity=d.get("mean_fidelity"),
            quality_score=d.get("quality_score", 0.0),
            grade=d.get("grade", "F"),
            pass_rate_5pct=d.get("pass_rate_5pct", 0.0),
            pass_rate_dual=d.get("pass_rate_dual", 0.0),
            critical_region_mean_de_gap=d.get("critical_region_mean_de_gap"),
            ordered_region_mean_de_gap=d.get("ordered_region_mean_de_gap"),
            de_gap_distribution=d.get("de_gap_distribution", {}),
        )

    @property
    def summary_line(self) -> str:
        """One-line summary for compact logging.

        Example: "B(0.72) ΔE/gap=0.034±0.018 P90=0.065 |ΔE|/N=8.9e-03"
        """
        parts = [f"{self.grade}({self.quality_score:.2f})"]
        parts.append(f"ΔE/gap={self.mean_de_gap:.3f}±{self.std_de_gap:.3f}")
        parts.append(f"P90={self.p90_de_gap:.3f}")
        if self.mean_abs_error_per_site is not None:
            parts.append(f"|ΔE|/N={self.mean_abs_error_per_site:.2e}")
        if self.mean_fidelity is not None:
            parts.append(f"F={self.mean_fidelity:.4f}")
        return " ".join(parts)

    @property
    def compact(self) -> str:
        """Ultra-compact for table cells: "B 0.034" (grade + mean)."""
        return f"{self.grade} {self.mean_de_gap:.3f}"


def compute_quality_profile(
    per_h_results: list[dict],
    *,
    h_critical: float | None = None,
    n_qubits: int | None = None,
) -> QualityProfile:
    """Compute a full QualityProfile from per-h evaluation results.

    Wraps and extends compute_deploy_summary with continuous quality metrics.
    Does NOT duplicate any computation — delegates to existing utilities.

    """
    from qmbp_simulation.analysis.metrics import compute_deploy_summary

    if not per_h_results:
        return QualityProfile(
            n_points=0,
            mean_de_gap=0.0,
            median_de_gap=0.0,
            std_de_gap=0.0,
            p90_de_gap=0.0,
            max_de_gap=0.0,
        )

    # ── Delegate to existing compute_deploy_summary for base metrics ──────
    summary = compute_deploy_summary(per_h_results)

    # ── Extract raw ΔE/gap array for distribution analysis ────────────────
    de_gaps = np.array([r["de_gap"] for r in per_h_results])
    n_points = len(de_gaps)

    # ── Percentile distribution ───────────────────────────────────────────
    percentiles = {
        "p10": float(np.percentile(de_gaps, 10)),
        "p25": float(np.percentile(de_gaps, 25)),
        "p50": float(np.percentile(de_gaps, 50)),
        "p75": float(np.percentile(de_gaps, 75)),
        "p90": float(np.percentile(de_gaps, 90)),
        "p95": float(np.percentile(de_gaps, 95)),
    }

    p90_de_gap = percentiles["p90"]

    # ── Per-site error ────────────────────────────────────────────────────
    mean_abs_error_per_site = summary.get("mean_abs_error_per_site")
    if mean_abs_error_per_site is None and n_qubits is not None:
        # Compute from available abs_errors
        abs_errors = [r.get("abs_error") for r in per_h_results if "abs_error" in r]
        if abs_errors and n_qubits > 0:
            mean_abs_error_per_site = float(np.mean(abs_errors)) / n_qubits

    # ── Quality score ─────────────────────────────────────────────────────
    score = compute_quality_score(
        mean_de_gap=summary["mean_de_gap"],
        p90_de_gap=p90_de_gap,
        mean_abs_error_per_site=mean_abs_error_per_site,
        n_points=n_points,
    )
    grade = grade_from_score(score)

    # ── Regional breakdown (optional, requires h values) ──────────────────
    critical_region_dg: float | None = None
    ordered_region_dg: float | None = None

    if h_critical is not None:
        h_values = [r.get("h") for r in per_h_results]
        if all(h is not None for h in h_values):
            h_arr = np.array(h_values, dtype=float)
            # Critical region: |h - h_c| < 0.5
            critical_mask = np.abs(h_arr - h_critical) < 0.5
            # Ordered region: h > h_c + 1.0 (deep in paramagnetic phase)
            ordered_mask = h_arr > h_critical + 1.0

            if critical_mask.any():
                critical_region_dg = float(de_gaps[critical_mask].mean())
            if ordered_mask.any():
                ordered_region_dg = float(de_gaps[ordered_mask].mean())

    # ── Build profile ─────────────────────────────────────────────────────
    return QualityProfile(
        n_points=n_points,
        mean_de_gap=summary["mean_de_gap"],
        median_de_gap=summary["median_de_gap"],
        std_de_gap=summary["std_de_gap"],
        p90_de_gap=p90_de_gap,
        max_de_gap=summary["max_de_gap"],
        mean_abs_error=summary.get("mean_abs_error"),
        mean_abs_error_per_site=mean_abs_error_per_site,
        mean_fidelity=summary.get("mean_fidelity"),
        quality_score=score,
        grade=grade,
        pass_rate_5pct=summary.get("pass_rate_5pct", 0.0),
        pass_rate_dual=summary.get("pass_rate_dual", 0.0),
        critical_region_mean_de_gap=critical_region_dg,
        ordered_region_mean_de_gap=ordered_region_dg,
        de_gap_distribution=percentiles,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Formatting utilities (for runners and reports)
# ═══════════════════════════════════════════════════════════════════════════════


def format_quality_summary(profile: QualityProfile) -> str:
    """Multi-line formatted summary for logging.

    Example output:
        Quality: B (score=0.72)
        ΔE/gap: 0.034 ± 0.018 | P90=0.065 | max=0.112
        |ΔE|/N: 8.9e-03 | Fidelity: 0.9812
        Distribution: [P25=0.018 | P50=0.031 | P75=0.048 | P90=0.065]
    """
    lines = []
    lines.append(f"Quality: {profile.grade} (score={profile.quality_score:.2f})")
    lines.append(
        f"ΔE/gap: {profile.mean_de_gap:.4f} ± {profile.std_de_gap:.4f} | "
        f"P90={profile.p90_de_gap:.4f} | max={profile.max_de_gap:.4f}"
    )

    extras = []
    if profile.mean_abs_error_per_site is not None:
        extras.append(f"|ΔE|/N={profile.mean_abs_error_per_site:.2e}")
    if profile.mean_fidelity is not None:
        extras.append(f"Fidelity={profile.mean_fidelity:.4f}")
    if extras:
        lines.append(" | ".join(extras))

    dist = profile.de_gap_distribution
    if dist:
        lines.append(
            f"Distribution: [P25={dist.get('p25', 0):.3f} | "
            f"P50={dist.get('p50', 0):.3f} | "
            f"P75={dist.get('p75', 0):.3f} | "
            f"P90={dist.get('p90', 0):.3f}]"
        )

    if profile.critical_region_mean_de_gap is not None:
        lines.append(
            f"Regions: critical={profile.critical_region_mean_de_gap:.4f} | "
            f"ordered={profile.ordered_region_mean_de_gap:.4f}"
            if profile.ordered_region_mean_de_gap is not None
            else f"Regions: critical={profile.critical_region_mean_de_gap:.4f}"
        )

    return "\n".join(lines)


def format_per_h_status(de_gap: float, abs_error: float | None = None) -> str:
    """Format a per-h point status as continuous value with grade indicator.
    --------
    >>> format_per_h_status(0.012)
    'A(0.012)'
    >>> format_per_h_status(0.048, abs_error=0.09)
    'B(0.048)'
    >>> format_per_h_status(0.120)
    'D(0.120)'
    """
    # Per-point "score" using same sigmoid (just de_gap → grade)
    point_score = 1.0 / (1.0 + (de_gap / SCORE_DE_GAP_SCALE) ** 2)

    # Check gap masking (passes de_gap but fails abs_error)
    gap_masked = False
    if abs_error is not None and de_gap < DE_GAP_THRESHOLD and abs_error > MAX_ABS_ERROR:
        gap_masked = True

    grade = grade_from_score(point_score)
    if gap_masked:
        return f"{grade}*(gm:{de_gap:.3f})"
    return f"{grade}({de_gap:.3f})"


def format_coverage_cell(profile: QualityProfile, n_qubits: int) -> str:
    """Format a coverage matrix cell with grade + mean ΔE/gap."""
    return f"{profile.grade} {profile.mean_de_gap:.3f} (N={n_qubits})"


def compare_profiles(
    current: QualityProfile,
    previous: QualityProfile,
) -> dict[str, Any]:
    """Compare two QualityProfiles for temporal tracking.

    Returns improvement/regression deltas across all continuous metrics.
    This replaces the binary "pass_rate dropped by X%" regression detection
    with a richer signal.

    Parameters
    ----------
    current : QualityProfile
        The newer evaluation.
    previous : QualityProfile
        The older evaluation (baseline).

    Returns
    -------
    dict
        Comparison metrics including:
        - delta_score: quality_score change (positive = improvement)
        - delta_mean_de_gap: mean error change (negative = improvement)
        - delta_p90: P90 change (negative = improvement)
        - grade_change: e.g. "C→B" or "B→B" (stable)
        - regression: bool — True if quality_score dropped by > 0.10
        - improvement: bool — True if quality_score improved by > 0.05
    """
    delta_score = current.quality_score - previous.quality_score
    delta_mean = current.mean_de_gap - previous.mean_de_gap
    delta_p90 = current.p90_de_gap - previous.p90_de_gap

    return {
        "delta_score": float(delta_score),
        "delta_mean_de_gap": float(delta_mean),
        "delta_p90_de_gap": float(delta_p90),
        "grade_change": f"{previous.grade}→{current.grade}",
        "score_change": f"{previous.quality_score:.2f}→{current.quality_score:.2f}",
        "regression": delta_score < -0.10,
        "improvement": delta_score > 0.05,
        "stable": abs(delta_score) <= 0.05,
    }
