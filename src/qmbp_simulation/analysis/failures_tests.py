"""Cross-N Failure Mode Diagnostics (Tests A-M).

Provides structured diagnosis of WHY cross-N predictions fail for a given
topology. Each test isolates a specific failure mechanism:
  A-C: Gap masking (large gap inflates ΔE/gap metric)
  D-F: Generalization failure (non-extensive scaling)
  G-I: Intrinsic VQE / ansatz expressibility limit
  J-L: Contaminated training data (false positives in NPZ)
  M:   H-range mismatch between training datasets at different N

These are heavy-import-free diagnostics (only numpy + pathlib).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from qmbp_simulation.analysis.constants import (
    DE_GAP_THRESHOLD,
    GAP_MASKING_THRESHOLD,
    MAX_ABS_ERROR,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostic Thresholds (named constants for traceability)
# ═══════════════════════════════════════════════════════════════════════════════

# Tests G-I: per-site error thresholds (units: energy / qubit)
PER_SITE_ANSATZ_LIMIT: float = 0.015
"""Above this per-site error, verified VQE points indicate ansatz limit."""

PER_SITE_VQE_BUDGET_LIMIT: float = 0.012
"""Moderate per-site error suggesting VQE budget exhaustion."""

PER_SITE_BEST_N_THRESHOLD: float = 0.008
"""If best-converged N is below this, small-N VQE works fine."""

PER_SITE_MPNN_COPY_ERROR: float = 0.020
"""Approximate (MPNN) per-site error above this indicates MPNN copy error."""

VERIFIED_HIGH_ERROR_FRACTION_THRESHOLD: float = 0.30
"""Fraction of verified points with per-site > PER_SITE_ANSATZ_LIMIT."""

# Test E: variational violation thresholds
VIOLATION_TOLERANCE: float = 1e-6
"""Energy below E_exact minus this tolerance counts as violation."""

VIOLATION_RATE_CRITICAL: float = 0.30
"""Above this violation rate, the data is considered corrupted."""

VIOLATION_RATE_WARNING: float = 0.05
"""Above this, log a warning — some violations may be numerical noise."""

# Test F: scaling fit thresholds
SLOPE_CRITICAL: float = 5e-4
"""Per-site error slope (energy/qubit²) above which scaling is non-extensive."""

# Test L: theta discontinuity
THETA_DISCONTINUITY_THRESHOLD: float = 1.0
"""Smoothness value above which theta has a discontinuity (~ 30% of [-π,π] range)."""

# Test M: h-range mismatch
H_RANGE_OVERLAP_MINIMUM: float = 0.50
"""Minimum fraction of h-range overlap required between N values."""

# Extensive scaling classification thresholds
VARIATION_EXTENSIVE_MAX: float = 3.0
"""Max variation ratio for "extensive" classification."""

VARIATION_DEGRADING_MAX: float = 5.0
"""Max variation ratio for "degrading" classification."""


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FailureDiagnostic:
    """Result of cross-N failure mode analysis for a single topology."""

    topology: str
    primary_mode: str  # "gap_masking", "generalization_failure", "mixed", "healthy"
    confidence: float  # 0-1 confidence in classification
    secondary_modes: list[str] = field(default_factory=list)

    # Test A: per-site consistency between passing and masked points
    per_site_ratio_masked_vs_passing: float | None = None

    # Test B: correlation of h vs |ΔE|/N
    corr_h_vs_per_site: float | None = None

    # Test C: h-range of masked points
    h_min_masked: float | None = None
    h_critical_approx: float | None = None
    masked_near_critical: bool = False

    # Test D: per-site at overlapping h (cross-N)
    cross_n_per_site_ratios: dict | None = None  # {(N_small, N_large): ratio}

    # Test E: variational violations
    violation_rate: float | None = None
    violation_source: str | None = None  # "vqe" or "mpnn" or "unknown"

    # Test F: linear fit |ΔE|/N = a + b*N
    slope_b: float | None = None
    fit_r_squared: float | None = None

    # Test G: VQE-to-MPNN consistency (verified vs approximate per-site error)
    per_site_verified: float | None = None
    per_site_approximate: float | None = None
    verified_vs_approx_ratio: float | None = None

    # Test H: error at best-converged N (smallest N with most verified points)
    best_n_per_site: float | None = None
    best_n: int | None = None

    # Test I: fraction of verified points with high per-site error
    verified_high_error_fraction: float | None = None

    # Test J: fraction of gap-masked points in training data
    training_gap_masked_fraction: float | None = None

    # Test K: zoo model contamination (single-criterion inflated pass_rate)
    zoo_single_vs_dual_gap: float | None = None

    # Test L: theta discontinuity in training data
    max_theta_smoothness: float | None = None
    n_configs_discontinuous: int | None = None

    # Test M: h-range mismatch
    h_range_overlap_fraction: float | None = None
    h_range_mismatch_pairs: list[str] | None = None

    explanation: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Tests A-C: Gap Masking Diagnosis
# ═══════════════════════════════════════════════════════════════════════════════


def diagnose_gap_masking(
    h_values: np.ndarray,
    de_gaps: np.ndarray,
    abs_errors: np.ndarray,
    n_qubits: int,
    *,
    h_critical: float = 1.0,
) -> dict:
    """Test A+B+C: Diagnose whether failures are pure gap masking.

    Gap masking occurs when ΔE/gap < 5% (pass) but |ΔE| > 0.10 (fail dual).
    This means the large spectral gap at high h inflates the denominator,
    hiding genuinely large absolute errors.

    Parameters
    ----------
    h_values : array of h field values
    de_gaps : array of ΔE/gap per point
    abs_errors : array of |ΔE| per point
    n_qubits : system size
    h_critical : approximate critical h for this topology (used in Test C
        to flag masked points near criticality as potentially physics-limited)

    Returns
    -------
    dict with keys:
        is_gap_masking : bool — True if failures are predominantly gap masking
        per_site_ratio : float — ratio of per-site error (masked / passing)
        per_site_masked_std : float — std of per-site in masked group (dispersion)
        corr_h_per_site : float — Pearson correlation of h vs |ΔE|/N
        h_min_masked : float | None — lowest h of gap-masked points
        n_pass : int — number of dual-criterion passing points
        n_masked : int — number of gap-masked points
        n_real_fail : int — number of real failures (ΔE/gap >= 5%)
        masked_near_critical : bool — whether masked points are near h_critical
        masked_in_trivial_regime : bool — whether ALL masked are at h > 2.5

    Raises
    ------
    ValueError
        If input arrays have inconsistent lengths or n_qubits <= 0.
    """
    h_values = np.asarray(h_values, dtype=float)
    de_gaps = np.asarray(de_gaps, dtype=float)
    abs_errors = np.asarray(abs_errors, dtype=float)

    if not (len(h_values) == len(de_gaps) == len(abs_errors)):
        raise ValueError(
            f"Array length mismatch: h_values={len(h_values)}, "
            f"de_gaps={len(de_gaps)}, abs_errors={len(abs_errors)}"
        )
    if n_qubits <= 0:
        raise ValueError(f"n_qubits must be positive, got {n_qubits}")

    # Handle empty input
    if len(h_values) == 0:
        return {
            "is_gap_masking": False,
            "per_site_ratio": 0.0,
            "per_site_masked_std": 0.0,
            "corr_h_per_site": 0.0,
            "h_min_masked": None,
            "n_pass": 0,
            "n_masked": 0,
            "n_real_fail": 0,
            "per_site_passing": 0.0,
            "per_site_masked": 0.0,
            "masked_near_critical": False,
            "masked_in_trivial_regime": False,
        }

    per_site = abs_errors / n_qubits
    dual_mask = (de_gaps < DE_GAP_THRESHOLD) & (abs_errors < MAX_ABS_ERROR)
    gap_masked = (de_gaps < DE_GAP_THRESHOLD) & (abs_errors >= MAX_ABS_ERROR)
    real_fail = de_gaps >= DE_GAP_THRESHOLD

    n_pass = int(dual_mask.sum())
    n_masked = int(gap_masked.sum())
    n_real = int(real_fail.sum())

    # Test A: per-site consistency + dispersion within masked group
    per_site_passing = float(per_site[dual_mask].mean()) if n_pass > 0 else 0.0
    per_site_masked_val = float(per_site[gap_masked].mean()) if n_masked > 0 else 0.0
    per_site_masked_std = float(per_site[gap_masked].std()) if n_masked > 1 else 0.0
    ratio = per_site_masked_val / max(per_site_passing, 1e-10) if n_pass > 0 else float("inf")

    # Test B: correlation h vs |ΔE|/N
    corr_h_per_site = 0.0
    if len(h_values) >= 3 and np.std(per_site) > 1e-12:
        corr_h_per_site = float(np.corrcoef(h_values, per_site)[0, 1])
        if not np.isfinite(corr_h_per_site):
            corr_h_per_site = 0.0

    # Test C: h-range of masked points and proximity to criticality
    h_min_masked = float(h_values[gap_masked].min()) if n_masked > 0 else None
    masked_near_critical = h_min_masked is not None and abs(h_min_masked - h_critical) < 0.5
    # Check if ALL masked points are in trivial PM regime (h > 2.5)
    masked_in_trivial = bool(n_masked > 0 and np.all(h_values[gap_masked] > 2.5))

    # Classification: gap masking if most failures are masked (not real ΔE/gap fails),
    # per-site error is consistent between groups (extensive behavior), and
    # error doesn't depend strongly on h (uniform prediction quality).
    # Additional: high dispersion within masked group suggests mixed failure mode.
    is_gap_masking = (
        n_masked > n_real  # more gap-masked than real failures
        and ratio < 2.0  # per-site error similar between passing and masked
        and abs(corr_h_per_site) < 0.6  # per-site doesn't depend strongly on h
        and per_site_masked_std < per_site_masked_val  # low relative dispersion
    )

    return {
        "is_gap_masking": is_gap_masking,
        "per_site_ratio": float(ratio),
        "per_site_masked_std": per_site_masked_std,
        "corr_h_per_site": corr_h_per_site,
        "h_min_masked": h_min_masked,
        "n_pass": n_pass,
        "n_masked": n_masked,
        "n_real_fail": n_real,
        "per_site_passing": per_site_passing,
        "per_site_masked": per_site_masked_val,
        "masked_near_critical": masked_near_critical,
        "masked_in_trivial_regime": masked_in_trivial,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests D-F: Generalization Failure Diagnosis
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_violation_rate(
    per_n_data: dict[int, dict],
) -> tuple[float, str]:
    """Test E helper: compute variational violation rate with source detection.

    A variational violation is E_pred < E_exact - tolerance. For VQE outputs
    this should NEVER happen (variational principle); for MPNN predictions it
    can happen since the MPNN is not constrained to be variational.

    Returns
    -------
    (violation_rate, source) where source is "vqe", "mpnn", or "unknown".
    """
    total_pts = 0
    total_violations = 0
    has_e_vqe = False
    has_e_pred_only = False

    for n, data in per_n_data.items():
        # Detect source: if "e_vqe" key exists, violations are from VQE
        if "e_vqe" in data:
            e_vals = np.asarray(data["e_vqe"], dtype=float)
            has_e_vqe = True
        elif "e_pred" in data:
            e_vals = np.asarray(data["e_pred"], dtype=float)
            has_e_pred_only = True
        else:
            continue

        e_exact = np.asarray(data.get("e_exact", []), dtype=float)
        n_common = min(len(e_vals), len(e_exact))
        if n_common > 0:
            violations = int(np.sum(e_vals[:n_common] < e_exact[:n_common] - VIOLATION_TOLERANCE))
            total_violations += violations
            total_pts += n_common

    rate = total_violations / max(total_pts, 1)

    if has_e_vqe and not has_e_pred_only:
        source = "vqe"
    elif has_e_pred_only and not has_e_vqe:
        source = "mpnn"
    else:
        source = "unknown"

    if rate > VIOLATION_RATE_WARNING and source == "vqe":
        logger.warning(
            "Variational violations detected in VQE data: %.1f%% of points "
            "have E_VQE < E_exact. This indicates numerical instability or "
            "data corruption.",
            rate * 100,
        )

    return rate, source


def _compute_scaling_fit(
    per_n_data: dict[int, dict],
    n_values: list[int],
) -> tuple[list[int], list[float], float, float]:
    """Test F helper: compute linear fit |ΔE|/N = a + b*N.

    Returns (ns_arr, per_sites_arr, slope_b, fit_r_squared).
    """
    ns_arr: list[int] = []
    per_sites_arr: list[float] = []

    for n in n_values:
        data = per_n_data[n]
        if "abs_errors" in data:
            abs_err = np.asarray(data["abs_errors"], dtype=float)
        else:
            e_pred = np.asarray(data.get("e_pred", data.get("e_vqe", [])), dtype=float)
            e_exact = np.asarray(data.get("e_exact", []), dtype=float)
            n_common = min(len(e_pred), len(e_exact))
            if n_common == 0:
                continue
            abs_err = np.abs(e_pred[:n_common] - e_exact[:n_common])

        if len(abs_err) > 0 and n > 0:
            per_site_mean = float(abs_err.mean()) / n
            ns_arr.append(n)
            per_sites_arr.append(per_site_mean)

    slope_b = 0.0
    fit_r_squared = 0.0
    if len(ns_arr) >= 2:
        ns_np = np.array(ns_arr, dtype=float)
        ps_np = np.array(per_sites_arr, dtype=float)
        coeffs = np.polyfit(ns_np, ps_np, 1)
        slope_b = float(coeffs[0])
        predicted = np.polyval(coeffs, ns_np)
        ss_res = float(np.sum((ps_np - predicted) ** 2))
        ss_tot = float(np.sum((ps_np - ps_np.mean()) ** 2))
        raw_r2 = 1.0 - ss_res / max(ss_tot, 1e-15)
        fit_r_squared = float(np.clip(raw_r2, 0.0, 1.0))

    return ns_arr, per_sites_arr, slope_b, fit_r_squared


def _compute_cross_n_ratios(
    per_n_data: dict[int, dict],
    n_values: list[int],
    h_tolerance: float,
) -> dict[str, float]:
    """Test D helper: per-site ratio at overlapping h-values between N pairs."""
    cross_n_ratios: dict[str, float] = {}

    for i, n_small in enumerate(n_values[:-1]):
        for n_large in n_values[i + 1 :]:
            data_small = per_n_data[n_small]
            data_large = per_n_data[n_large]
            h_small = np.asarray(data_small.get("h_values", []), dtype=float)
            h_large = np.asarray(data_large.get("h_values", []), dtype=float)
            abs_small = np.asarray(data_small.get("abs_errors", []), dtype=float)
            abs_large = np.asarray(data_large.get("abs_errors", []), dtype=float)

            if len(h_small) == 0 or len(h_large) == 0:
                continue
            if len(abs_small) != len(h_small) or len(abs_large) != len(h_large):
                logger.debug(
                    "diagnose_generalization: length mismatch for N=%d→%d, skipping",
                    n_small,
                    n_large,
                )
                continue

            # Use adaptive tolerance: min of fixed tolerance and half the smaller grid spacing
            if len(h_small) > 1:
                spacing_small = float(np.min(np.abs(np.diff(np.sort(h_small)))))
                adaptive_tol = min(h_tolerance, spacing_small * 0.6)
            else:
                adaptive_tol = h_tolerance

            common_ps_small: list[float] = []
            common_ps_large: list[float] = []
            for j, h in enumerate(h_large):
                match_idx = int(np.argmin(np.abs(h_small - h)))
                if abs(float(h_small[match_idx]) - float(h)) <= adaptive_tol:
                    common_ps_small.append(float(abs_small[match_idx]) / n_small)
                    common_ps_large.append(float(abs_large[j]) / n_large)

            if len(common_ps_small) >= 2:
                mean_small = float(np.mean(common_ps_small))
                mean_large = float(np.mean(common_ps_large))
                ratio = mean_large / max(mean_small, 1e-10)
                cross_n_ratios[f"N{n_small}→N{n_large}"] = round(ratio, 2)

    return cross_n_ratios


def _diagnose_generalization_failure_dashboard(
    per_n_data: dict[int, dict],
    *,
    h_tolerance: float = 0.15,
) -> dict:
    """Test D+E+F: Diagnose whether cross-N shows generalization failure.

    Parameters
    ----------
    per_n_data : dict mapping N → {h_values, abs_errors, e_pred|e_vqe, e_exact, gaps}
        Raw per-h data for each system size.
    h_tolerance : float
        Max tolerance for h-matching in overlapping range comparison.
        Actual tolerance adapts to the smaller grid spacing.

    Returns
    -------
    dict with keys:
        is_generalization_failure : bool
        violation_rate : float — fraction of variational violations
        violation_source : str — "vqe", "mpnn", or "unknown"
        slope_b : float — slope of linear fit |ΔE|/N = a + b*N
        fit_r_squared : float — R² of the fit (clamped to [0, 1])
        cross_n_ratios : dict — per-site ratio at overlapping h for each N pair
        extensive_verdict : str — "extensive", "degrading", "non_extensive"
        per_site_values : dict — {N: mean_per_site_error} for each N
    """
    n_values = sorted(per_n_data.keys())
    if len(n_values) < 2:
        return {
            "is_generalization_failure": False,
            "violation_rate": 0.0,
            "violation_source": "unknown",
            "slope_b": 0.0,
            "fit_r_squared": 0.0,
            "cross_n_ratios": {},
            "extensive_verdict": "insufficient_data",
            "per_site_values": {},
        }

    # Test E: Variational violations with source detection
    violation_rate, violation_source = _compute_violation_rate(per_n_data)

    # Test F: Linear fit |ΔE|/N = a + b*N
    ns_arr, per_sites_arr, slope_b, fit_r_squared = _compute_scaling_fit(per_n_data, n_values)

    # Test D: Cross-N per-site at overlapping h-values
    cross_n_ratios = _compute_cross_n_ratios(per_n_data, n_values, h_tolerance)

    # Extensive scaling verdict
    if len(per_sites_arr) >= 2:
        ps_min = min(per_sites_arr)
        ps_max = max(per_sites_arr)
        variation = ps_max / max(ps_min, 1e-10)
        if variation < VARIATION_EXTENSIVE_MAX:
            extensive_verdict = "extensive"
        elif variation < VARIATION_DEGRADING_MAX:
            extensive_verdict = "degrading"
        else:
            extensive_verdict = "non_extensive"
    else:
        extensive_verdict = "insufficient_data"

    # Classification: use source-aware violation threshold
    violation_critical = (
        violation_rate > VIOLATION_RATE_CRITICAL
        if violation_source == "mpnn"
        else violation_rate > VIOLATION_RATE_WARNING
    )

    is_generalization_failure = (
        extensive_verdict == "non_extensive"
        or (slope_b > SLOPE_CRITICAL and fit_r_squared > 0.5)
        or violation_critical
    )

    return {
        "is_generalization_failure": is_generalization_failure,
        "violation_rate": violation_rate,
        "violation_source": violation_source,
        "slope_b": slope_b,
        "fit_r_squared": fit_r_squared,
        "cross_n_ratios": cross_n_ratios,
        "extensive_verdict": extensive_verdict,
        "per_site_values": dict(zip(ns_arr, per_sites_arr, strict=False)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Test D: Contaminated Training Data
# ═══════════════════════════════════════════════════════════════════════════════


def diagnose_contaminated_training(
    h_values: np.ndarray,
    de_gaps: np.ndarray,
    abs_errors: np.ndarray,
    theta_smoothness: float,
    n_qubits: int,
) -> dict:
    """Test D: Check if training data contamination causes failures.

    Contamination signs:
    - High theta discontinuity (smoothness > 0.5)
    - Isolated failure points surrounded by passing points
    - Energy worse than initial (variational violation)

    Returns dict with is_contaminated, evidence, recommendation.
    """
    n_points = len(h_values)
    if n_points < 3:
        return {"is_contaminated": False, "evidence": "insufficient_data"}

    # Check theta smoothness (proxy for inconsistent optimization)
    high_discontinuity = theta_smoothness > 0.5

    # Check for isolated failures (failing point between two passing points)
    pass_mask = de_gaps < DE_GAP_THRESHOLD
    n_isolated = 0
    for i in range(1, n_points - 1):
        if not pass_mask[i] and pass_mask[i - 1] and pass_mask[i + 1]:
            n_isolated += 1

    isolated_fraction = n_isolated / max(n_points - 2, 1)

    is_contaminated = high_discontinuity and isolated_fraction > 0.1

    return {
        "is_contaminated": is_contaminated,
        "theta_smoothness": theta_smoothness,
        "high_discontinuity": high_discontinuity,
        "n_isolated_failures": n_isolated,
        "isolated_fraction": isolated_fraction,
        "recommendation": (
            "Run canonicalize_theta + filter_consistent_theta on NPZ"
            if is_contaminated
            else "Training data appears consistent"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests G-I: Intrinsic VQE / Ansatz Error
# ═══════════════════════════════════════════════════════════════════════════════


def _diagnose_intrinsic_vqe_error_dashboard(
    dashboard_configs: list[dict],
    npz_dir: str | Path | None = None,
) -> dict:
    """Tests G+H+I: Diagnose whether failures stem from HVA ansatz expressibility.

    Distinguishes three sub-diagnoses:
    - ansatz_limit: Verified VQE points have high per-site error → HVA can't express GS.
    - vqe_budget_limit: Small N is fine but larger N has high verified error →
      optimizer iterations insufficient for the growing landscape.
    - mpnn_copy_error: Verified (VQE) is good but approximate (MPNN) is bad →
      the MPNN degrades the quality of the VQE-optimized parameters.

    Parameters
    ----------
    dashboard_configs : list of config dicts for ONE topology from dashboard.
    npz_dir : path to multi_n_training directory (for reading quality tiers).

    Returns
    -------
    dict with keys:
        is_intrinsic_vqe_error : bool
        per_site_verified : float — mean per-site error of verified points
        per_site_approximate : float — mean per-site error of approximate points
        verified_vs_approx_ratio : float — if > 0.8, verified are almost as bad
        best_n : int | None — smallest N with verified data
        best_n_per_site : float — per-site error at best-converged N
        verified_high_error_fraction : float — fraction of verified with |ΔE|/N > threshold
        n_verified_total : int — total number of verified points analyzed
        sub_diagnosis : str — "ansatz_limit", "vqe_budget_limit", "mpnn_copy_error", "no_intrinsic_issue"
    """
    from pathlib import Path as _Path

    if npz_dir is None:
        _ROOT = _Path(__file__).resolve().parents[3]
        npz_dir = _ROOT / "data" / "multi_n_training"
    else:
        npz_dir = _Path(npz_dir)

    per_site_by_tier: dict[str, list[float]] = {"verified": [], "approximate": [], "unverified": []}
    per_site_by_n: dict[int, float] = {}
    n_verified_high: int = 0
    n_verified_total: int = 0

    for c in dashboard_configs:
        topo = c["topology"]
        n = c["n_qubits"]
        npz_file = npz_dir / f"{topo}_N{n}_p{c.get('p_layers', 1)}.npz"
        if not npz_file.exists():
            continue

        try:
            data = np.load(str(npz_file), allow_pickle=True)
            e_vqe = data.get("e_vqe", np.array([]))
            e_exact = data.get("e_exact", np.array([]))
            tiers = data.get("quality_tier", np.array([]))

            if len(e_vqe) == 0 or len(e_exact) == 0:
                continue

            abs_err = np.abs(e_vqe - e_exact)
            per_site = abs_err / max(n, 1)

            # Aggregate per-tier
            for i, tier in enumerate(tiers):
                tier_str = str(tier)
                if tier_str in per_site_by_tier and i < len(per_site):
                    per_site_by_tier[tier_str].append(float(per_site[i]))

            # Track verified points with high error
            verified_mask = np.array([str(t) == "verified" for t in tiers])
            if verified_mask.any():
                verified_ps = per_site[verified_mask]
                n_verified_total += len(verified_ps)
                n_verified_high += int(np.sum(verified_ps > PER_SITE_ANSATZ_LIMIT))
                per_site_by_n[n] = float(verified_ps.mean())

        except (OSError, ValueError, KeyError) as exc:
            logger.debug("diagnose_intrinsic_vqe_error: failed to load %s: %s", npz_file, exc)
            continue

    # Compute metrics
    ps_verified = (
        float(np.mean(per_site_by_tier["verified"])) if per_site_by_tier["verified"] else 0.0
    )
    ps_approx = (
        float(np.mean(per_site_by_tier["approximate"])) if per_site_by_tier["approximate"] else 0.0
    )
    ratio = ps_verified / max(ps_approx, 1e-10) if ps_approx > 0 else 0.0

    best_n = min(per_site_by_n.keys()) if per_site_by_n else None
    best_n_ps = per_site_by_n.get(best_n, 0.0) if best_n else 0.0

    verified_high_frac = n_verified_high / max(n_verified_total, 1)

    # Sub-diagnosis logic using named thresholds
    if (
        ps_verified > PER_SITE_ANSATZ_LIMIT
        and verified_high_frac > VERIFIED_HIGH_ERROR_FRACTION_THRESHOLD
    ):
        sub_diagnosis = "ansatz_limit"
        is_intrinsic = True
    elif best_n_ps < PER_SITE_BEST_N_THRESHOLD and ps_verified > PER_SITE_VQE_BUDGET_LIMIT:
        sub_diagnosis = "vqe_budget_limit"
        is_intrinsic = False
    elif ps_verified < PER_SITE_BEST_N_THRESHOLD and ps_approx > PER_SITE_MPNN_COPY_ERROR:
        sub_diagnosis = "mpnn_copy_error"
        is_intrinsic = False
    elif ps_verified > PER_SITE_VQE_BUDGET_LIMIT:
        sub_diagnosis = "ansatz_limit"
        is_intrinsic = True
    else:
        sub_diagnosis = "no_intrinsic_issue"
        is_intrinsic = False

    return {
        "is_intrinsic_vqe_error": is_intrinsic,
        "per_site_verified": ps_verified,
        "per_site_approximate": ps_approx,
        "verified_vs_approx_ratio": float(ratio),
        "best_n": best_n,
        "best_n_per_site": float(best_n_ps),
        "verified_high_error_fraction": verified_high_frac,
        "n_verified_total": n_verified_total,
        "sub_diagnosis": sub_diagnosis,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Test E: Generalization Failure (cross-N)
# ═══════════════════════════════════════════════════════════════════════════════


def diagnose_generalization_failure(
    train_n_values: list[int],
    target_n: int,
    pass_rate_dual: float,
    mean_abs_error: float,
) -> dict:
    """Test E: Diagnose whether failure is due to poor generalization.

    Cross-N transfer fails when the target N is far from training data
    or the error is extensive (scales with N).

    Returns dict with is_generalization_failure, gap_factor, recommendation.
    """
    max_train_n = max(train_n_values) if train_n_values else 0
    min_train_n = min(train_n_values) if train_n_values else 0

    # Distance factor: how far is target from nearest training N
    if target_n > max_train_n:
        gap_factor = target_n / max(max_train_n, 1)
    elif target_n < min_train_n:
        gap_factor = min_train_n / max(target_n, 1)
    else:
        gap_factor = 1.0  # interpolation

    # Extensive error check: |ΔE|/N > 0.01 suggests per-site error accumulation
    per_site_error = mean_abs_error / max(target_n, 1)
    extensive_error = per_site_error > 0.01

    is_generalization_failure = pass_rate_dual < 0.5 and (gap_factor > 1.5 or extensive_error)

    recommendation = ""
    if is_generalization_failure:
        if gap_factor > 2.0:
            recommendation = f"Add training data at N={target_n} or intermediate N"
        elif extensive_error:
            recommendation = (
                f"Error is extensive (|ΔE|/N={per_site_error:.4f}). "
                "Increase VQE restarts or refine θ at target N."
            )
        else:
            recommendation = "Improve MPNN training or add more h-points"

    return {
        "is_generalization_failure": is_generalization_failure,
        "gap_factor": gap_factor,
        "per_site_error": per_site_error,
        "extensive_error": extensive_error,
        "recommendation": recommendation,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Test F: Intrinsic VQE Error (ansatz expressibility limit)
# ═══════════════════════════════════════════════════════════════════════════════


def diagnose_intrinsic_vqe_error(
    h_values: np.ndarray,
    de_gaps: np.ndarray,
    abs_errors: np.ndarray,
    n_qubits: int,
    p_layers: int,
    coordination: int = 2,
) -> dict:
    """Test F: Diagnose whether failures are due to HVA expressibility limits.

    Intrinsic VQE error occurs when the ansatz cannot represent the ground
    state regardless of optimization quality. Signs:
    - All methods (VQE + MPNN) fail at same h-values
    - Error grows monotonically as h → h_critical
    - Error scales with coordination number (z>2 harder)

    Returns dict with is_intrinsic, h_boundary, recommendation.
    """
    n_points = len(h_values)
    if n_points < 3:
        return {"is_intrinsic": False, "evidence": "insufficient_data"}

    # Sort by h ascending
    order = np.argsort(h_values)
    h_sorted = h_values[order]
    de_sorted = de_gaps[order]

    # Find the boundary: h where de_gap first exceeds 5% (scanning from high h)
    pass_mask = de_sorted < DE_GAP_THRESHOLD
    if pass_mask.all():
        return {"is_intrinsic": False, "h_boundary": None, "all_pass": True}
    if not pass_mask.any():
        h_boundary = float(h_sorted[-1])
    else:
        # Last passing index
        last_pass = np.where(pass_mask)[0][-1]
        h_boundary = float(h_sorted[last_pass])

    # Monotonicity: does error increase monotonically as h decreases?
    # (check from h_boundary downward)
    below_boundary = h_sorted <= h_boundary + 0.1
    if below_boundary.sum() > 2:
        de_below = de_sorted[below_boundary]
        monotonic = all(de_below[i] <= de_below[i + 1] * 1.2 for i in range(len(de_below) - 1))
    else:
        monotonic = False

    # Coordination factor: higher z → more parameters → harder optimization
    # z=2 (chain) is baseline, z=3 (ladder/heavy_hex), z=4 (square), z=6 (triangular)
    coord_penalty = coordination / 2.0

    is_intrinsic = monotonic and coord_penalty > 1.0 and not pass_mask.all()

    return {
        "is_intrinsic": is_intrinsic,
        "h_boundary": h_boundary,
        "monotonic_below_boundary": monotonic,
        "coordination": coordination,
        "coord_penalty": coord_penalty,
        "n_params_per_layer": n_qubits + n_qubits * (coordination // 2),
        "recommendation": (
            f"Physics limit at h<{h_boundary:.2f} for z={coordination} p={p_layers}. "
            "Options: increase p_layers (noise budget permitting) or restrict h-range."
            if is_intrinsic
            else "Not an expressibility limit — check optimization quality."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests J-L: Contaminated Training Data
# ═══════════════════════════════════════════════════════════════════════════════


def _diagnose_contaminated_training_dashboard(
    dashboard_configs: list[dict],
) -> dict:
    """Tests J+K+L: Diagnose whether training data is contaminated.

    Contamination = the NPZ data contains points that appear to be good
    (pass ΔE/gap < 5%) but are actually gap-masked false positives (|ΔE| > 0.10).
    The MPNN learns from these "good" points and reproduces the error.

    Parameters
    ----------
    dashboard_configs : list of config dicts for ONE topology from dashboard.

    Returns
    -------
    dict with keys:
        is_contaminated : bool
        gap_masked_fraction : float — fraction of training configs that are gap-masked
        masked_point_fraction : float — weighted fraction of gap-masked points
        zoo_inflation : float — zoo_pass_rate - actual_dual_pass (single vs dual gap)
        max_theta_smoothness : float — worst theta discontinuity
        n_discontinuous : int — configs with smoothness > threshold
        n_not_useful : int — configs classified as not_useful
        n_insufficient : int — configs classified as insufficient_signal
        contamination_severity : str — "severe", "moderate", "mild", "none"
    """
    n_configs = len(dashboard_configs)
    if n_configs == 0:
        return {
            "is_contaminated": False,
            "gap_masked_fraction": 0.0,
            "masked_point_fraction": 0.0,
            "zoo_inflation": 0.0,
            "max_theta_smoothness": 0.0,
            "n_discontinuous": 0,
            "n_not_useful": 0,
            "n_insufficient": 0,
            "contamination_severity": "none",
        }

    # Test J: Gap-masked fraction in training data
    n_gap_masked_configs = sum(
        1
        for c in dashboard_configs
        if c.get("pass_rate_5pct", 0) - c.get("pass_rate_dual_criterion", 0) > GAP_MASKING_THRESHOLD
    )
    gap_masked_fraction = n_gap_masked_configs / n_configs

    # Total points that are gap-masked (weighted by n_points)
    total_pts = sum(c.get("n_points", 0) for c in dashboard_configs)
    masked_pts = sum(
        int(
            c.get("n_points", 0)
            * (c.get("pass_rate_5pct", 0) - c.get("pass_rate_dual_criterion", 0))
        )
        for c in dashboard_configs
        if c.get("pass_rate_5pct", 0) - c.get("pass_rate_dual_criterion", 0) > 0
    )
    masked_point_fraction = masked_pts / max(total_pts, 1)

    # Test K: Zoo inflation (zoo pass_rate was single-criterion pre-2026-08-11)
    # Check if metric_version is available to determine if inflation is real
    zoo_pass = next(
        (c["zoo_pass_rate"] for c in dashboard_configs if c.get("zoo_pass_rate")),
        None,
    )
    best_dual = max((c.get("pass_rate_dual_criterion", 0) for c in dashboard_configs), default=0)
    zoo_inflation = (zoo_pass - best_dual) if zoo_pass is not None else 0.0
    # If zoo has metric_version == "dual_v1", inflation is real (model is wrong)
    # If zoo is legacy (no metric_version), inflation may be metric artifact
    has_dual_zoo = any(
        c.get("zoo_metric_version") == "dual_v1"
        for c in dashboard_configs
        if c.get("zoo_metric_version")
    )
    if not has_dual_zoo and zoo_inflation > 0:
        # Discount inflation from metric artifact
        zoo_inflation = max(0.0, zoo_inflation - 0.10)

    # Test L: Theta discontinuity (normalized check)
    smoothness_values = [
        c["theta_smoothness"] for c in dashboard_configs if c.get("theta_smoothness") is not None
    ]
    max_smoothness = max(smoothness_values) if smoothness_values else 0.0
    n_discontinuous = sum(1 for s in smoothness_values if s > THETA_DISCONTINUITY_THRESHOLD)

    # Training utility counts
    n_not_useful = sum(1 for c in dashboard_configs if c.get("training_utility") == "not_useful")
    n_insufficient = sum(
        1 for c in dashboard_configs if c.get("training_utility") == "insufficient_signal"
    )

    # Severity classification
    if n_not_useful >= 3 and masked_point_fraction > 0.40:
        severity = "severe"
        is_contaminated = True
    elif n_not_useful >= 2 or masked_point_fraction > 0.30:
        severity = "moderate"
        is_contaminated = True
    elif gap_masked_fraction > 0.30 or max_smoothness > THETA_DISCONTINUITY_THRESHOLD * 1.5:
        severity = "mild"
        is_contaminated = True
    else:
        severity = "none"
        is_contaminated = False

    return {
        "is_contaminated": is_contaminated,
        "gap_masked_fraction": gap_masked_fraction,
        "masked_point_fraction": masked_point_fraction,
        "zoo_inflation": zoo_inflation,
        "max_theta_smoothness": max_smoothness,
        "n_discontinuous": n_discontinuous,
        "n_not_useful": n_not_useful,
        "n_insufficient": n_insufficient,
        "contamination_severity": severity,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Master classifier: topology failure mode
# ═══════════════════════════════════════════════════════════════════════════════


def classify_topology_failure_mode(
    topology: str,
    n_qubits: int,
    p_layers: int,
    h_values: np.ndarray,
    de_gaps: np.ndarray,
    abs_errors: np.ndarray,
    *,
    theta_smoothness: float = 0.0,
    train_n_values: list[int] | None = None,
    coordination: int | None = None,
) -> FailureDiagnostic:
    """Classify the dominant failure mode for a topology/N/p configuration.

    Runs all diagnostic tests (A-F) and returns a FailureDiagnostic with
    the primary failure mode and actionable recommendations.

    Parameters
    ----------
    topology : str
        Lattice topology name.
    n_qubits : int
        System size.
    p_layers : int
        HVA depth.
    h_values, de_gaps, abs_errors : np.ndarray
        Per-h-point metrics.
    theta_smoothness : float
        Max consecutive theta L-inf difference.
    train_n_values : list[int] | None
        N values used for training (for cross-N diagnosis).
    coordination : int | None
        Lattice coordination number. Auto-detected from topology if None.

    Returns
    -------
    FailureDiagnostic
        Structured diagnosis with failure_mode, evidence, recommendations.
    """
    # Auto-detect coordination
    COORD_MAP = {
        "chain_1d": 2,
        "ladder": 3,
        "heavy_hex": 3,
        "square": 4,
        "triangular": 6,
        "kagome": 4,
    }
    z = coordination if coordination is not None else COORD_MAP.get(topology, 2)

    # Run all tests
    gap_result = diagnose_gap_masking(h_values, de_gaps, abs_errors, n_qubits)
    contam_result = diagnose_contaminated_training(
        h_values, de_gaps, abs_errors, theta_smoothness, n_qubits
    )
    intrinsic_result = diagnose_intrinsic_vqe_error(
        h_values, de_gaps, abs_errors, n_qubits, p_layers, coordination=z
    )

    pass_rate_dual = float(
        ((de_gaps < DE_GAP_THRESHOLD) & (abs_errors < MAX_ABS_ERROR)).sum() / max(len(h_values), 1)
    )

    gen_result = None
    if train_n_values:
        gen_result = diagnose_generalization_failure(
            train_n_values, n_qubits, pass_rate_dual, float(abs_errors.mean())
        )

    # Priority-based classification
    if gap_result["is_gap_masking"]:
        mode = "gap_masking"
        explanation = (
            f"Gap masking dominant: {gap_result['n_masked']} masked vs "
            f"{gap_result['n_real_fail']} real failures. "
            f"Per-site error ratio={gap_result['per_site_ratio']:.2f}."
        )
    elif intrinsic_result["is_intrinsic"]:
        mode = "physics_limit"
        explanation = (
            f"HVA p={p_layers} expressibility limit at h<{intrinsic_result['h_boundary']:.2f} "
            f"for z={z}. Error is monotonic below boundary."
        )
    elif contam_result["is_contaminated"]:
        mode = "contaminated_training"
        explanation = (
            f"Training data contamination: smoothness={theta_smoothness:.3f}, "
            f"{contam_result['n_isolated_failures']} isolated failures."
        )
    elif gen_result and gen_result["is_generalization_failure"]:
        mode = "generalization_failure"
        explanation = (
            f"Cross-N gap_factor={gen_result['gap_factor']:.2f}, "
            f"per_site_error={gen_result['per_site_error']:.4f}."
        )
    else:
        mode = "optimization_quality"
        explanation = (
            "No structural failure mode detected. "
            "Failures likely from insufficient VQE optimization (restarts/iterations)."
        )

    return FailureDiagnostic(
        topology=topology,
        primary_mode=mode,
        confidence=1.0 - pass_rate_dual,  # higher confidence when more failures
        max_theta_smoothness=theta_smoothness,
        explanation=explanation,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test M: H-Range Mismatch Detection
# ═══════════════════════════════════════════════════════════════════════════════


def diagnose_h_range_mismatch(
    per_n_data: dict[int, dict],
) -> dict:
    """Test M: Detect h-range mismatch between training datasets at different N.

    If N=6 has h ∈ [0.5, 5.0] but N=16 only has h ∈ [2.5, 5.0], the MPNN
    is trained on different physics at different N — cross-N predictions in
    the non-overlapping region (h < 2.5) are pure extrapolation.

    Parameters
    ----------
    per_n_data : dict mapping N → {h_values, ...}

    Returns
    -------
    dict with keys:
        has_mismatch : bool — True if overlap < H_RANGE_OVERLAP_MINIMUM
        overlap_fraction : float — fraction of h-range covered by all N values
        global_h_range : tuple[float, float] — union of all h-ranges
        per_n_ranges : dict[int, tuple[float, float]] — h-range per N
        mismatch_pairs : list[str] — pairs with insufficient overlap
    """
    n_values = sorted(per_n_data.keys())
    if len(n_values) < 2:
        return {
            "has_mismatch": False,
            "overlap_fraction": 1.0,
            "global_h_range": (0.0, 0.0),
            "per_n_ranges": {},
            "mismatch_pairs": [],
        }

    # Compute h-range per N
    per_n_ranges: dict[int, tuple[float, float]] = {}
    for n in n_values:
        h_vals = np.asarray(per_n_data[n].get("h_values", []), dtype=float)
        if len(h_vals) > 0:
            per_n_ranges[n] = (float(h_vals.min()), float(h_vals.max()))

    if len(per_n_ranges) < 2:
        return {
            "has_mismatch": False,
            "overlap_fraction": 1.0,
            "global_h_range": (0.0, 0.0),
            "per_n_ranges": per_n_ranges,
            "mismatch_pairs": [],
        }

    # Global range (union)
    all_mins = [r[0] for r in per_n_ranges.values()]
    all_maxs = [r[1] for r in per_n_ranges.values()]
    global_min = min(all_mins)
    global_max = max(all_maxs)
    global_span = global_max - global_min

    # Overlap = intersection of all ranges / union
    intersection_min = max(all_mins)
    intersection_max = min(all_maxs)
    overlap_span = max(0.0, intersection_max - intersection_min)
    overlap_fraction = overlap_span / max(global_span, 1e-10)

    # Pairwise mismatch detection
    mismatch_pairs: list[str] = []
    ns_with_range = sorted(per_n_ranges.keys())
    for i, n1 in enumerate(ns_with_range[:-1]):
        for n2 in ns_with_range[i + 1 :]:
            r1 = per_n_ranges[n1]
            r2 = per_n_ranges[n2]
            pair_overlap_min = max(r1[0], r2[0])
            pair_overlap_max = min(r1[1], r2[1])
            pair_overlap = max(0.0, pair_overlap_max - pair_overlap_min)
            pair_span = max(r1[1] - r1[0], r2[1] - r2[0])
            pair_frac = pair_overlap / max(pair_span, 1e-10)
            if pair_frac < H_RANGE_OVERLAP_MINIMUM:
                mismatch_pairs.append(
                    f"N{n1}[{r1[0]:.1f},{r1[1]:.1f}]↔N{n2}[{r2[0]:.1f},{r2[1]:.1f}] "
                    f"(overlap={pair_frac:.0%})"
                )

    has_mismatch = overlap_fraction < H_RANGE_OVERLAP_MINIMUM or len(mismatch_pairs) > 0

    return {
        "has_mismatch": has_mismatch,
        "overlap_fraction": overlap_fraction,
        "global_h_range": (global_min, global_max),
        "per_n_ranges": per_n_ranges,
        "mismatch_pairs": mismatch_pairs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Unified Classifier (dashboard-based, for project_health workflows)
# ═══════════════════════════════════════════════════════════════════════════════


def classify_topology_failure_mode_from_dashboard(
    topology: str,
    dashboard_configs: list[dict],
    extrapolation_data: dict[int, dict] | None = None,
) -> FailureDiagnostic:
    """Unified failure mode classifier for a topology.

    Combines Tests A-M to produce a single diagnosis with optional secondary
    modes. Priority is determined by evidence strength, not fixed ordering.

    Parameters
    ----------
    topology : str
    dashboard_configs : list of config dicts from dashboard for this topology
    extrapolation_data : dict mapping N → {h_values, abs_errors, e_pred, e_exact, gaps}
        from large-N NPZ files. Optional.

    Returns
    -------
    FailureDiagnostic with primary_mode, secondary_modes, and all test results.
    """
    diag = FailureDiagnostic(topology=topology, primary_mode="healthy", confidence=0.0)

    if not dashboard_configs:
        diag.explanation = "No dashboard configs available for this topology."
        return diag

    # Check if topology has any failures at all
    best_dual = max((c.get("pass_rate_dual_criterion", 0) for c in dashboard_configs), default=0)
    if best_dual >= 0.95:
        diag.primary_mode = "healthy"
        diag.confidence = 1.0
        diag.explanation = f"Best pass_rate_dual={best_dual:.0%} — no significant failures."
        return diag

    # ── Collect evidence from each test group ─────────────────────────────
    # Use a score-based approach: each test contributes a (mode, score) pair.
    # The mode with highest cumulative score wins as primary_mode.
    mode_scores: dict[str, float] = {
        "gap_masking": 0.0,
        "generalization_failure": 0.0,
        "intrinsic_vqe_error": 0.0,
        "contaminated_training": 0.0,
    }

    # ── Tests A-C: Gap masking from dashboard ─────────────────────────────
    total_masked = sum(
        c.get("pass_rate_5pct", 0) - c.get("pass_rate_dual_criterion", 0)
        for c in dashboard_configs
        if c.get("pass_rate_5pct", 0) - c.get("pass_rate_dual_criterion", 0) > GAP_MASKING_THRESHOLD
    )
    n_configs_masked = sum(
        1
        for c in dashboard_configs
        if c.get("pass_rate_5pct", 0) - c.get("pass_rate_dual_criterion", 0) > GAP_MASKING_THRESHOLD
    )
    avg_mask_severity = total_masked / max(n_configs_masked, 1)
    has_gap_masking = n_configs_masked >= 2 and avg_mask_severity > 0.20

    diag.per_site_ratio_masked_vs_passing = avg_mask_severity if has_gap_masking else None
    if has_gap_masking:
        mode_scores["gap_masking"] = 0.5 + min(avg_mask_severity, 0.45)

    # ── Tests D-F+M: Generalization failure from extrapolation ────────────
    gen_result = None
    h_range_result = None
    has_gen_failure = False
    if extrapolation_data and len(extrapolation_data) >= 2:
        gen_result = _diagnose_generalization_failure_dashboard(extrapolation_data)
        diag.violation_rate = gen_result["violation_rate"]
        diag.violation_source = gen_result["violation_source"]
        diag.slope_b = gen_result["slope_b"]
        diag.fit_r_squared = gen_result["fit_r_squared"]
        diag.cross_n_per_site_ratios = gen_result["cross_n_ratios"]
        has_gen_failure = gen_result["is_generalization_failure"]

        # Test M: h-range mismatch
        h_range_result = diagnose_h_range_mismatch(extrapolation_data)
        diag.h_range_overlap_fraction = h_range_result["overlap_fraction"]
        diag.h_range_mismatch_pairs = h_range_result["mismatch_pairs"]

        if has_gen_failure:
            gen_score = 0.4 + gen_result["fit_r_squared"] * 0.4
            mode_scores["generalization_failure"] = gen_score

        # h-range mismatch boosts generalization_failure score
        if h_range_result["has_mismatch"]:
            mode_scores["generalization_failure"] += 0.15

    # ── Tests G-I: Intrinsic VQE error ────────────────────────────────────
    vqe_result = _diagnose_intrinsic_vqe_error_dashboard(dashboard_configs)
    diag.per_site_verified = vqe_result["per_site_verified"]
    diag.per_site_approximate = vqe_result["per_site_approximate"]
    diag.verified_vs_approx_ratio = vqe_result["verified_vs_approx_ratio"]
    diag.best_n = vqe_result["best_n"]
    diag.best_n_per_site = vqe_result["best_n_per_site"]
    diag.verified_high_error_fraction = vqe_result["verified_high_error_fraction"]
    has_intrinsic_vqe = vqe_result["is_intrinsic_vqe_error"]

    if has_intrinsic_vqe:
        mode_scores["intrinsic_vqe_error"] = 0.75

    # ── Tests J-L: Contaminated training data ─────────────────────────────
    contam_result = _diagnose_contaminated_training_dashboard(dashboard_configs)
    diag.training_gap_masked_fraction = contam_result["masked_point_fraction"]
    diag.zoo_single_vs_dual_gap = contam_result["zoo_inflation"]
    diag.max_theta_smoothness = contam_result["max_theta_smoothness"]
    diag.n_configs_discontinuous = contam_result["n_discontinuous"]
    has_contamination = contam_result["is_contaminated"]

    if has_contamination:
        sev_score = {"severe": 0.85, "moderate": 0.65, "mild": 0.45, "none": 0.0}
        mode_scores["contaminated_training"] = sev_score.get(
            contam_result["contamination_severity"], 0.0
        )

    # ── Decision logic: score-based with secondary modes ──────────────────
    # Sort modes by score descending
    sorted_modes = sorted(mode_scores.items(), key=lambda x: -x[1])
    active_modes = [(mode, score) for mode, score in sorted_modes if score > 0.3]

    if not active_modes:
        diag.primary_mode = "unknown"
        diag.confidence = 0.3
        diag.explanation = "No dominant failure pattern detected."
        return diag

    # Primary = highest scoring mode
    primary_mode, primary_score = active_modes[0]

    # Check for mixed mode: if top 2 are both strong and close in score
    if len(active_modes) >= 2:
        secondary_mode, secondary_score = active_modes[1]
        if secondary_score > 0.5 and (primary_score - secondary_score) < 0.2:
            diag.primary_mode = "mixed"
            diag.confidence = 0.7
            diag.secondary_modes = [primary_mode, secondary_mode]
            diag.explanation = (
                f"Mixed failure: {primary_mode} (score={primary_score:.2f}) + "
                f"{secondary_mode} (score={secondary_score:.2f})"
            )
            return diag

        # Record secondary modes
        diag.secondary_modes = [m for m, s in active_modes[1:] if s > 0.3]

    diag.primary_mode = primary_mode
    diag.confidence = float(np.clip(primary_score, 0.1, 0.95))

    # Generate explanation based on primary mode
    if primary_mode == "gap_masking":
        diag.explanation = (
            f"Gap masking: {n_configs_masked} configs, severity={avg_mask_severity:.0%}. "
            f"Model predicts well (|ΔE|/N consistent), |ΔE|>0.10 from N×ε."
        )
    elif primary_mode == "contaminated_training":
        diag.explanation = (
            f"Contaminated training ({contam_result['contamination_severity']}): "
            f"{contam_result['n_not_useful']} not-useful configs, "
            f"{contam_result['masked_point_fraction']:.0%} points gap-masked, "
            f"θ_smoothness={contam_result['max_theta_smoothness']:.2f}."
        )
    elif primary_mode == "intrinsic_vqe_error":
        sub = vqe_result["sub_diagnosis"]
        diag.explanation = (
            f"Intrinsic VQE error ({sub}): verified per-site={vqe_result['per_site_verified']:.2e}, "
            f"{vqe_result['verified_high_error_fraction']:.0%} verified with |ΔE|/N>{PER_SITE_ANSATZ_LIMIT}. "
            f"Best N={vqe_result['best_n']} per-site={vqe_result['best_n_per_site']:.2e}."
        )
    elif primary_mode == "generalization_failure" and gen_result is not None:
        extra = ""
        if h_range_result and h_range_result["has_mismatch"]:
            extra = f" H-range mismatch detected: overlap={h_range_result['overlap_fraction']:.0%}."
        diag.explanation = (
            f"Non-extensive scaling: |ΔE|/N grows with N "
            f"(slope={gen_result['slope_b']:.2e}, R²={gen_result['fit_r_squared']:.2f}).{extra}"
        )

    return diag
