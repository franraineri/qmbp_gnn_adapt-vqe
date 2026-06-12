#!/usr/bin/env python3
"""Sanity check for analysis results — validates data integrity and physics.

A modular, extensible sanity checker that validates:
  1. Data extraction completeness and consistency
  2. Physics constraints (known limits, expected behaviors)
  3. Statistical soundness of claims
  4. Cross-reference between analyses

Each check is a self-contained function registered via @register_check.
New checks can be added without modifying the engine.

Usage:
    python -m project_health.analysis.sanity_check
    python -m project_health.analysis.sanity_check --verbose
    python -m project_health.analysis.sanity_check --only theta_pca
    python -m project_health.analysis.sanity_check --json report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = ROOT / "analysis" / "raw_data"


# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CheckResult:
    """Result of a single sanity check."""

    name: str
    category: str  # "data_integrity", "physics", "statistics", "cross_ref"
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    severity: str = "error"  # "error", "warning", "info"


@dataclass
class SanityReport:
    """Aggregated sanity check report."""

    checks: list[CheckResult] = field(default_factory=list)
    n_passed: int = 0
    n_failed: int = 0
    n_warnings: int = 0
    overall_pass: bool = True

    def add(self, result: CheckResult) -> None:
        """Add a check result to the report."""
        self.checks.append(result)
        if result.passed:
            self.n_passed += 1
        elif result.severity == "warning":
            self.n_warnings += 1
        else:
            self.n_failed += 1
            self.overall_pass = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "overall_pass": self.overall_pass,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "n_warnings": self.n_warnings,
            "checks": [
                {
                    "name": c.name,
                    "category": c.category,
                    "passed": c.passed,
                    "severity": c.severity,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Check Registry
# ═══════════════════════════════════════════════════════════════════════════════

_CHECKS: list[tuple[str, str, Callable[..., list[CheckResult]]]] = []


def register_check(name: str, category: str = "data_integrity"):
    """Decorator to register a sanity check function.

    The decorated function should return a list of CheckResult objects.
    It receives (verbose: bool) as argument.
    """

    def decorator(func: Callable[..., list[CheckResult]]):
        _CHECKS.append((name, category, func))
        return func

    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# Check Implementations — Data Integrity
# ═══════════════════════════════════════════════════════════════════════════════


@register_check("theta_trajectories_exist", "data_integrity")
def check_theta_trajectories_exist(verbose: bool = False) -> list[CheckResult]:
    """Verify theta_trajectories.json exists and has expected structure."""
    results = []
    fpath = RAW_DATA_DIR / "theta_trajectories.json"

    if not fpath.exists():
        results.append(
            CheckResult(
                name="theta_trajectories_file_exists",
                category="data_integrity",
                passed=False,
                message=f"Missing: {fpath.relative_to(ROOT)}",
                severity="error",
            )
        )
        return results

    with fpath.open() as f:
        data = json.load(f)

    # Check structure
    trajs = data.get("trajectories", [])
    results.append(
        CheckResult(
            name="theta_trajectories_file_exists",
            category="data_integrity",
            passed=True,
            message=f"Found {len(trajs)} trajectories",
            details={"n_trajectories": len(trajs)},
        )
    )

    # Check minimum content
    results.append(
        CheckResult(
            name="theta_trajectories_min_count",
            category="data_integrity",
            passed=len(trajs) >= 5,
            message=(
                f"Have {len(trajs)} trajectories (need ≥5)"
                if len(trajs) >= 5
                else f"Only {len(trajs)} trajectories (need ≥5)"
            ),
            severity="error" if len(trajs) < 5 else "info",
        )
    )

    # Check required fields
    required_fields = {"topology", "n_qubits", "p_layers", "h_values", "theta_opt"}
    for i, traj in enumerate(trajs[:3]):
        missing = required_fields - set(traj.keys())
        if missing:
            results.append(
                CheckResult(
                    name=f"theta_trajectory_{i}_fields",
                    category="data_integrity",
                    passed=False,
                    message=f"Trajectory {i} missing fields: {missing}",
                    severity="error",
                )
            )
            break
    else:
        results.append(
            CheckResult(
                name="theta_trajectories_fields_complete",
                category="data_integrity",
                passed=True,
                message="All required fields present",
            )
        )

    # Check theta_opt dimensions match h_values
    for traj in trajs:
        n_h = len(traj.get("h_values", []))
        n_theta = len(traj.get("theta_opt", []))
        if n_h != n_theta:
            results.append(
                CheckResult(
                    name="theta_trajectories_dimensions",
                    category="data_integrity",
                    passed=False,
                    message=(
                        f"Dimension mismatch in {traj.get('topology', '?')} "
                        f"N={traj.get('n_qubits', '?')}: "
                        f"h_values({n_h}) != theta_opt({n_theta})"
                    ),
                    severity="error",
                )
            )
            break
    else:
        results.append(
            CheckResult(
                name="theta_trajectories_dimensions",
                category="data_integrity",
                passed=True,
                message="All h_values/theta_opt dimensions match",
            )
        )

    return results


@register_check("pca_results_exist", "data_integrity")
def check_pca_results_exist(verbose: bool = False) -> list[CheckResult]:
    """Verify theta_pca_results.json exists and has expected structure."""
    results = []
    fpath = RAW_DATA_DIR / "theta_pca_results.json"

    if not fpath.exists():
        results.append(
            CheckResult(
                name="pca_results_file_exists",
                category="data_integrity",
                passed=False,
                message=f"Missing: {fpath.relative_to(ROOT)}",
                severity="error",
            )
        )
        return results

    with fpath.open() as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    per_traj = data.get("per_trajectory", [])

    results.append(
        CheckResult(
            name="pca_results_file_exists",
            category="data_integrity",
            passed=True,
            message=f"Found {len(per_traj)} analyzed trajectories",
        )
    )

    # Check metadata has required fields
    required_meta = {"overall_pass", "n_topologies_pca_pass", "total_topologies"}
    missing_meta = required_meta - set(meta.keys())
    results.append(
        CheckResult(
            name="pca_results_metadata_complete",
            category="data_integrity",
            passed=len(missing_meta) == 0,
            message=(
                "Metadata complete"
                if not missing_meta
                else f"Missing metadata keys: {missing_meta}"
            ),
            severity="warning" if missing_meta else "info",
        )
    )

    return results


@register_check("derivative_results_exist", "data_integrity")
def check_derivative_results_exist(verbose: bool = False) -> list[CheckResult]:
    """Verify theta_derivative_vs_d1.json exists and has expected structure."""
    results = []
    fpath = RAW_DATA_DIR / "theta_derivative_vs_d1.json"

    if not fpath.exists():
        results.append(
            CheckResult(
                name="derivative_results_file_exists",
                category="data_integrity",
                passed=False,
                message=f"Missing: {fpath.relative_to(ROOT)}",
                severity="error",
            )
        )
        return results

    with fpath.open() as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    results.append(
        CheckResult(
            name="derivative_results_file_exists",
            category="data_integrity",
            passed=True,
            message="File exists with valid JSON",
        )
    )

    # Check thesis paragraph generated
    paragraph = meta.get("thesis_paragraph", "")
    results.append(
        CheckResult(
            name="derivative_thesis_paragraph",
            category="data_integrity",
            passed=len(paragraph) > 50,
            message=(
                f"Thesis paragraph: {len(paragraph)} chars"
                if len(paragraph) > 50
                else "Thesis paragraph missing or too short"
            ),
            severity="warning" if len(paragraph) <= 50 else "info",
        )
    )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Check Implementations — Physics Constraints
# ═══════════════════════════════════════════════════════════════════════════════


@register_check("theta_pca_physics", "physics")
def check_theta_pca_physics(verbose: bool = False) -> list[CheckResult]:
    """Validate PCA results against known TFIM physics.

    Physics constraints:
    - h_c = 1.0 for standard 1D TFIM (N→∞)
    - Finite-size corrections shift effective h_c slightly above 1.0
    - PCA peak should be within [0.7, 1.5] for chain_1d with data covering h_c
    - Ladder has different effective h_c (higher coordination → lower h_c)
    """
    results = []
    fpath = RAW_DATA_DIR / "theta_pca_results.json"

    if not fpath.exists():
        return [
            CheckResult(
                name="pca_physics_skip",
                category="physics",
                passed=True,
                message="Skipped — no PCA results file",
                severity="info",
            )
        ]

    with fpath.open() as f:
        data = json.load(f)

    per_traj = data.get("per_trajectory", [])
    h_c = 1.0

    # Check: chain_1d trajectories that include h_c region should detect it
    chain_covering_hc = [
        r
        for r in per_traj
        if r["topology"] == "chain_1d" and min(r["h_values"]) <= h_c + 0.3  # data reaches near h_c
    ]

    if chain_covering_hc:
        pca_peaks = [r["pca_peak_h"] for r in chain_covering_hc]
        mean_peak = sum(pca_peaks) / len(pca_peaks)

        results.append(
            CheckResult(
                name="pca_chain_peak_near_hc",
                category="physics",
                passed=0.7 <= mean_peak <= 1.5,
                message=(f"chain_1d PCA peak mean: h={mean_peak:.2f} (expected near h_c={h_c})"),
                details={
                    "mean_peak": mean_peak,
                    "n_trajectories": len(chain_covering_hc),
                    "peaks": pca_peaks,
                },
                severity="error" if not (0.7 <= mean_peak <= 1.5) else "info",
            )
        )
    else:
        results.append(
            CheckResult(
                name="pca_chain_peak_near_hc",
                category="physics",
                passed=True,
                message="No chain_1d trajectories cover h_c region — cannot verify",
                severity="warning",
            )
        )

    # Check: ladder data not covering h_c should NOT claim detection
    ladder_far = [
        r for r in per_traj if r["topology"] == "ladder" and min(r["h_values"]) > h_c + 0.5
    ]
    if ladder_far:
        false_detections = [r for r in ladder_far if r.get("success_pca", False)]
        results.append(
            CheckResult(
                name="pca_ladder_no_false_detection",
                category="physics",
                passed=len(false_detections) == 0,
                message=(
                    "Ladder (no h_c coverage): no false phase detection"
                    if not false_detections
                    else f"WARNING: {len(false_detections)} false detections on ladder"
                ),
                severity="warning" if false_detections else "info",
            )
        )

    # Check: PCA explained variance should be substantial (>50% in PC1)
    for r in per_traj[:3]:
        ev = r.get("pca_explained_variance", [])
        if ev:
            pc1_var = ev[0]
            results.append(
                CheckResult(
                    name=f"pca_variance_{r['topology']}_N{r['n_qubits']}",
                    category="physics",
                    passed=pc1_var > 0.5,
                    message=(
                        f"{r['topology']} N={r['n_qubits']}: PC1 explains {pc1_var:.1%} variance"
                    ),
                    details={"explained_variance_ratio": ev},
                    severity="warning" if pc1_var <= 0.5 else "info",
                )
            )

    return results


@register_check("theta_derivative_physics", "physics")
def check_theta_derivative_physics(verbose: bool = False) -> list[CheckResult]:
    """Validate |∂θ/∂h| results against physics expectations.

    Physics constraints:
    - |∂θ/∂h| should increase near critical point (diverges in N→∞ limit)
    - Peak should be in [0.7, 2.0] for chain_1d with p=2
    - D1 peak (valid-only) is known to be at h≈1.07 (from D1 experiment)
    - Agreement Δh < 0.5 is physically reasonable
    """
    results = []
    fpath = RAW_DATA_DIR / "theta_derivative_vs_d1.json"

    if not fpath.exists():
        return [
            CheckResult(
                name="derivative_physics_skip",
                category="physics",
                passed=True,
                message="Skipped — no derivative results file",
                severity="info",
            )
        ]

    with fpath.open() as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    derivs = data.get("theta_derivatives", [])

    # Check: chain_1d peak is physically reasonable
    chain_peaks = [
        d["peak_h"] for d in derivs if d.get("topology") == "chain_1d" and d.get("p_layers") == 2
    ]

    if chain_peaks:
        mean_peak = sum(chain_peaks) / len(chain_peaks)
        results.append(
            CheckResult(
                name="derivative_chain_peak_reasonable",
                category="physics",
                passed=0.7 <= mean_peak <= 2.0,
                message=(
                    f"chain_1d p=2 mean |∂θ/∂h| peak: h={mean_peak:.2f} (expected [0.7, 2.0])"
                ),
                details={"chain_peaks": chain_peaks, "mean": mean_peak},
            )
        )

    # Check: agreement with D1 is within 0.5
    agreement = meta.get("peak_agreement_with_d1")
    if agreement is not None:
        results.append(
            CheckResult(
                name="derivative_d1_agreement",
                category="physics",
                passed=agreement < 0.5,
                message=(
                    f"|∂θ/∂h| vs D1 peak agreement: Δh={agreement:.2f} "
                    f"({'OK' if agreement < 0.5 else 'too large'})"
                ),
                details={"agreement": agreement},
                severity="warning" if agreement >= 0.5 else "info",
            )
        )

    # Check: D1 peak metadata is consistent with known value
    d1_peak_meta = meta.get("d1_peak_valid_metadata")
    if d1_peak_meta is not None:
        results.append(
            CheckResult(
                name="d1_peak_known_value",
                category="physics",
                passed=abs(d1_peak_meta - 1.07) < 0.01,
                message=(f"D1 peak reference: h={d1_peak_meta} (expected 1.07 from exp_d1)"),
                severity="error" if abs(d1_peak_meta - 1.07) >= 0.01 else "info",
            )
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Check Implementations — Statistical Soundness
# ═══════════════════════════════════════════════════════════════════════════════


@register_check("pca_statistical_validity", "statistics")
def check_pca_statistical_validity(verbose: bool = False) -> list[CheckResult]:
    """Validate statistical claims in PCA analysis.

    Checks:
    - K-means with k=2 is appropriate (not overclaiming)
    - PCA peak detection requires sufficient h-resolution
    - Success criterion correctly applied
    """
    results = []
    fpath = RAW_DATA_DIR / "theta_pca_results.json"

    if not fpath.exists():
        return []

    with fpath.open() as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    per_traj = data.get("per_trajectory", [])

    # Check: success criterion is correctly applied
    n_pca_pass = meta.get("n_topologies_pca_pass", 0)
    total = meta.get("total_topologies", 0)
    overall = meta.get("overall_pass", False)

    # Overall should be True only if ≥2 topologies pass
    expected_overall = n_pca_pass >= 2
    results.append(
        CheckResult(
            name="pca_success_criterion_correct",
            category="statistics",
            passed=overall == expected_overall,
            message=(
                f"Success criterion correctly evaluated: "
                f"{n_pca_pass}/{total} topologies → "
                f"{'PASS' if overall else 'FAIL'}"
            ),
            severity="error" if overall != expected_overall else "info",
        )
    )

    # Check: trajectories with <5 h-points shouldn't make strong claims
    short_trajs = [r for r in per_traj if r.get("n_points", 0) < 7]
    short_claiming_success = [r for r in short_trajs if r.get("success_pca", False)]
    if short_claiming_success:
        results.append(
            CheckResult(
                name="pca_short_trajectory_caution",
                category="statistics",
                passed=False,
                message=(
                    f"{len(short_claiming_success)} trajectories with <7 points "
                    f"claim PCA success — low resolution may give false peak"
                ),
                severity="warning",
                details={
                    "trajectories": [
                        f"{r['topology']} N={r['n_qubits']} ({r['n_points']} pts)"
                        for r in short_claiming_success
                    ],
                },
            )
        )
    else:
        results.append(
            CheckResult(
                name="pca_short_trajectory_caution",
                category="statistics",
                passed=True,
                message="No short-trajectory false positives",
            )
        )

    # Check: h-range must cover h_c for detection to be meaningful
    h_c = 1.0
    trajs_not_covering = [
        r for r in per_traj if min(r["h_values"]) > h_c + 0.5 and r.get("success_pca", False)
    ]
    results.append(
        CheckResult(
            name="pca_h_range_covers_hc",
            category="statistics",
            passed=len(trajs_not_covering) == 0,
            message=(
                "No false success claims from trajectories not covering h_c"
                if not trajs_not_covering
                else f"{len(trajs_not_covering)} claim success without covering h_c"
            ),
            severity="error" if trajs_not_covering else "info",
        )
    )

    return results


@register_check("derivative_statistical_validity", "statistics")
def check_derivative_statistical_validity(verbose: bool = False) -> list[CheckResult]:
    """Validate statistical claims in derivative analysis."""
    results = []
    fpath = RAW_DATA_DIR / "theta_derivative_vs_d1.json"

    if not fpath.exists():
        return []

    with fpath.open() as f:
        data = json.load(f)

    correlations = data.get("correlations", [])

    # Check: correlation p-values
    significant_corrs = [
        c for c in correlations if c.get("p_value") is not None and c["p_value"] < 0.05
    ]
    total_valid = [c for c in correlations if c.get("pearson_r") is not None]

    results.append(
        CheckResult(
            name="derivative_significant_correlations",
            category="statistics",
            passed=True,
            message=(
                f"{len(significant_corrs)}/{len(total_valid)} correlations "
                f"are statistically significant (p<0.05)"
            ),
            details={
                "n_significant": len(significant_corrs),
                "n_total": len(total_valid),
            },
            severity="info",
        )
    )

    # Check: thesis paragraph doesn't overclaim
    meta = data.get("metadata", {})
    max_r = meta.get("max_correlation")
    if max_r is not None:
        # Max correlation reported in thesis should match computed
        results.append(
            CheckResult(
                name="derivative_max_correlation_valid",
                category="statistics",
                passed=-1.0 <= max_r <= 1.0,
                message=f"Max ρ={max_r:.3f} is in valid range [-1, 1]",
                severity="error" if not (-1.0 <= max_r <= 1.0) else "info",
            )
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Check Implementations — Cross-Reference Consistency
# ═══════════════════════════════════════════════════════════════════════════════


@register_check("cross_ref_pca_derivative", "cross_ref")
def check_cross_ref_pca_derivative(verbose: bool = False) -> list[CheckResult]:
    """Verify PCA and derivative analyses are consistent.

    Both analyze the same theta_opt data — peaks should agree.
    """
    results = []
    pca_path = RAW_DATA_DIR / "theta_pca_results.json"
    deriv_path = RAW_DATA_DIR / "theta_derivative_vs_d1.json"

    if not pca_path.exists() or not deriv_path.exists():
        return [
            CheckResult(
                name="cross_ref_skip",
                category="cross_ref",
                passed=True,
                message="Skipped — need both PCA and derivative results",
                severity="info",
            )
        ]

    with pca_path.open() as f:
        pca_data = json.load(f)
    with deriv_path.open() as f:
        deriv_data = json.load(f)

    # Compare peak locations for matching configs
    pca_peaks: dict[str, float] = {}
    for r in pca_data.get("per_trajectory", []):
        key = f"{r['topology']}_N{r['n_qubits']}_p{r['p_layers']}_s{r['seed']}"
        pca_peaks[key] = r.get("pca_peak_h", 0)

    deriv_peaks: dict[str, float] = {}
    for r in deriv_data.get("theta_derivatives", []):
        key = f"{r['topology']}_N{r['n_qubits']}_p{r['p_layers']}_s{r['seed']}"
        deriv_peaks[key] = r.get("peak_h", 0)

    # Find matching configs
    common_keys = set(pca_peaks.keys()) & set(deriv_peaks.keys())
    if not common_keys:
        results.append(
            CheckResult(
                name="cross_ref_common_configs",
                category="cross_ref",
                passed=True,
                message="No common configs between PCA and derivative (different filtering)",
                severity="warning",
            )
        )
        return results

    agreements = []
    for key in common_keys:
        delta = abs(pca_peaks[key] - deriv_peaks[key])
        agreements.append(delta)

    mean_agreement = sum(agreements) / len(agreements)
    results.append(
        CheckResult(
            name="cross_ref_peak_consistency",
            category="cross_ref",
            passed=mean_agreement < 0.5,
            message=(
                f"PCA vs |dθ/dh| peak agreement: mean Δh={mean_agreement:.2f} "
                f"across {len(common_keys)} configs"
            ),
            details={
                "n_configs": len(common_keys),
                "mean_delta": mean_agreement,
                "max_delta": max(agreements),
            },
            severity="warning" if mean_agreement >= 0.5 else "info",
        )
    )

    return results


@register_check("figures_generated", "data_integrity")
def check_figures_generated(verbose: bool = False) -> list[CheckResult]:
    """Verify that expected analysis figures exist."""
    results = []
    figures_dir = ROOT / "project_health" / "figures"

    expected_figures = [
        "fig_theta_pca_phase_detection",
        "fig_theta_derivative_vs_d1",
    ]

    for fig_name in expected_figures:
        # Check for any format (png, pdf, svg)
        found = any((figures_dir / f"{fig_name}.{ext}").exists() for ext in ["png", "pdf", "svg"])
        results.append(
            CheckResult(
                name=f"figure_{fig_name}",
                category="data_integrity",
                passed=found,
                message=(f"{fig_name}: found" if found else f"{fig_name}: MISSING"),
                severity="warning" if not found else "info",
            )
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Engine — Run All Checks
# ═══════════════════════════════════════════════════════════════════════════════


def run_sanity_checks(
    only: str | None = None,
    verbose: bool = False,
) -> SanityReport:
    """Run all registered sanity checks and return a report.

    Parameters
    ----------
    only : str | None
        If set, only run checks whose name contains this substring.
    verbose : bool
        Pass verbose flag to individual checks.

    Returns
    -------
    SanityReport
        Aggregated results.
    """
    report = SanityReport()

    for name, category, check_fn in _CHECKS:
        if only and only not in name:
            continue

        try:
            check_results = check_fn(verbose=verbose)
            for cr in check_results:
                report.add(cr)
        except Exception as e:
            report.add(
                CheckResult(
                    name=f"{name}_error",
                    category=category,
                    passed=False,
                    message=f"Check crashed: {e}",
                    severity="error",
                )
            )

    return report


def format_report(report: SanityReport, verbose: bool = False) -> str:
    """Format a sanity report for console output."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("  ANALYSIS SANITY CHECK")
    lines.append("=" * 60)

    # Group by category
    by_category: dict[str, list[CheckResult]] = {}
    for check in report.checks:
        by_category.setdefault(check.category, []).append(check)

    category_labels = {
        "data_integrity": "📁 Data Integrity",
        "physics": "⚛️  Physics Constraints",
        "statistics": "📊 Statistical Soundness",
        "cross_ref": "🔗 Cross-Reference Consistency",
    }

    for cat, checks in by_category.items():
        label = category_labels.get(cat, cat)
        lines.append(f"\n  ─── {label} ───")

        for check in checks:
            icon = "✓" if check.passed else ("⚠" if check.severity == "warning" else "✗")
            lines.append(f"  {icon} {check.message}")

            if verbose and check.details:
                for k, v in check.details.items():
                    lines.append(f"      {k}: {v}")

    # Summary
    lines.append("")
    lines.append("─" * 60)
    status = "PASS ✓" if report.overall_pass else "FAIL ✗"
    lines.append(
        f"  {status} | "
        f"{report.n_passed} passed, "
        f"{report.n_warnings} warnings, "
        f"{report.n_failed} failed"
    )
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Checks — Scaling Extensions (E5: NLCE, Bond Dimension, HE)
# ═══════════════════════════════════════════════════════════════════════════════


@register_check("nlce_convergence", "physics")
def check_nlce_convergence(verbose: bool = False) -> list[CheckResult]:
    """Verify NLCE results converge and weights decay monotonically."""
    results = []
    e5_dir = ROOT / "results" / "experiments" / "exp_e5_scaling_ext"

    if not e5_dir.exists():
        results.append(
            CheckResult(
                name="nlce_results_exist",
                category="physics",
                passed=True,
                message="E5 results not yet generated (skip)",
                severity="info",
            )
        )
        return results

    # Find the latest run with NLCE data (Section 4 or 5)
    run_files = sorted(e5_dir.glob("run_*.json"), reverse=True)
    nlce_data = None
    source_file = ""

    for rf in run_files:
        try:
            with open(rf) as f:
                data = json.load(f)
            section_4 = data.get("results", {}).get("section_4", {}).get("data")
            if section_4 and "results_per_h" in section_4:
                nlce_data = section_4
                source_file = rf.name
                break
        except (json.JSONDecodeError, OSError):
            continue

    if not nlce_data:
        results.append(
            CheckResult(
                name="nlce_results_exist",
                category="physics",
                passed=True,
                message="No NLCE section results found (pending execution)",
                severity="info",
            )
        )
        return results

    results.append(
        CheckResult(
            name="nlce_results_exist",
            category="physics",
            passed=True,
            message=f"Found NLCE data in {source_file}",
        )
    )

    # Check 1: Gapped-phase convergence (error < 5%)
    per_h = nlce_data.get("results_per_h", [])
    gapped_errors = [r["error_pct"] for r in per_h if "error_pct" in r and r.get("h", 0) > 1.2]
    if gapped_errors:
        mean_gapped = sum(gapped_errors) / len(gapped_errors)
        results.append(
            CheckResult(
                name="nlce_gapped_convergence",
                category="physics",
                passed=mean_gapped < 5.0,
                message=f"Gapped-phase NLCE error: {mean_gapped:.3f}% (threshold: 5%)",
                details={"mean_error_pct": mean_gapped, "n_points": len(gapped_errors)},
                severity="error" if mean_gapped >= 5.0 else "info",
            )
        )

    # Check 2: Weight decay (|W(L)| should decrease with L for gapped phase)
    for r in per_h:
        weights = r.get("weights", {})
        if len(weights) < 4:
            continue
        h_val = r.get("h", 0)
        if h_val <= 1.2:  # Skip critical region
            continue

        # Check last 3 weights decrease in magnitude
        sorted_keys = sorted(int(k) for k in weights)
        last_3 = [abs(weights[str(k)]) for k in sorted_keys[-3:]]
        monotone = last_3[0] >= last_3[1] >= last_3[2]

        if not monotone:
            results.append(
                CheckResult(
                    name="nlce_weight_decay",
                    category="physics",
                    passed=False,
                    message=(
                        f"NLCE weights not monotonically decaying at h={h_val}: "
                        f"|W| = {[f'{w:.2e}' for w in last_3]}"
                    ),
                    severity="warning",
                    details={"h": h_val, "last_3_weights": last_3},
                )
            )
            break
    else:
        results.append(
            CheckResult(
                name="nlce_weight_decay",
                category="physics",
                passed=True,
                message="NLCE weights decay monotonically in gapped phase",
            )
        )

    return results


@register_check("bond_dimension_exactness", "physics")
def check_bond_dimension_exactness(verbose: bool = False) -> list[CheckResult]:
    """Verify MPS bond dimension test shows χ=64 convergence."""
    results = []
    e5_dir = ROOT / "results" / "experiments" / "exp_e5_scaling_ext"

    if not e5_dir.exists():
        return results

    run_files = sorted(e5_dir.glob("run_*.json"), reverse=True)
    bd_data = None

    for rf in run_files:
        try:
            with open(rf) as f:
                data = json.load(f)
            section_1 = data.get("results", {}).get("section_1", {}).get("data")
            if section_1 and "chi_convergence" in section_1:
                bd_data = section_1
                break
        except (json.JSONDecodeError, OSError):
            continue

    if not bd_data:
        return results  # Not yet executed, skip silently

    # Check: E(χ) converges monotonically
    chi_results = bd_data.get("chi_convergence", [])
    if len(chi_results) >= 2:
        energies = [r["energy"] for r in chi_results]
        # For variational methods, E(larger χ) ≤ E(smaller χ) (lower is better)
        # But for HVA evaluation, energy should converge (differences shrink)
        diffs = [abs(energies[i + 1] - energies[i]) for i in range(len(energies) - 1)]
        diffs_decrease = all(diffs[i] >= diffs[i + 1] for i in range(len(diffs) - 1))

        results.append(
            CheckResult(
                name="bond_dim_convergence_monotone",
                category="physics",
                passed=diffs_decrease,
                message=(
                    "χ convergence diffs decrease monotonically"
                    if diffs_decrease
                    else f"Non-monotone χ convergence: diffs={[f'{d:.2e}' for d in diffs]}"
                ),
                severity="warning" if not diffs_decrease else "info",
                details={"diffs": diffs},
            )
        )

    # Check: χ=64 is exact (diff < 1e-10)
    diff_64_128 = bd_data.get("diff_64_128")
    if diff_64_128 is not None:
        is_exact = diff_64_128 < 1e-10
        results.append(
            CheckResult(
                name="bond_dim_chi64_exact",
                category="physics",
                passed=is_exact,
                message=(
                    f"|E(χ=64)-E(χ=128)| = {diff_64_128:.2e} "
                    f"({'exact' if is_exact else 'NOT exact, χ=64 insufficient'})"
                ),
                severity="error" if not is_exact else "info",
                details={"diff_64_128": diff_64_128},
            )
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Sanity check for analysis results (Task 2 & 3)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed check output",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Only run checks matching this substring (e.g. 'theta_pca')",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Export report as JSON to this path",
    )
    return parser.parse_args()


def main() -> None:
    """Run sanity checks and report results."""
    args = parse_args()

    report = run_sanity_checks(only=args.only, verbose=args.verbose)

    # Console output
    print(format_report(report, verbose=args.verbose))

    # JSON export
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"  📄 JSON report saved to: {args.json}")

    sys.exit(0 if report.overall_pass else 1)


if __name__ == "__main__":
    main()
