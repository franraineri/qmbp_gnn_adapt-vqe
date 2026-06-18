"""Mitiq Integration Analyzer.

Analyzes results from Mitiq error mitigation comparison experiments:
- Section 21 rehearsal results (multi-method comparison)
- Standalone mitiq benchmark runs
- Hardware deployment with --mitiq-verify (CDR cross-check)

Scans:
    results/experiments/exp_hw_rehearsal_v3/run_*.json → Section 21 data
    results/mitiq/comparison_*.json → Standalone benchmark results
    results/hardware/run_*/execution_summary.json → Hardware Mitiq verifications

Usage:
    python -m project_health.analysis.mitiq_analyzer
    python -m project_health.analysis.mitiq_analyzer --verbose
    python -m project_health.analysis.mitiq_analyzer --json report.json
    python -m project_health.analysis.mitiq_analyzer --thesis-table
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MitiqMethodResult:
    """Result for one method at one h-point."""

    method: str = ""
    energy: float = 0.0
    delta_e_gap: float = 0.0


@dataclass
class MitiqComparisonSummary:
    """Summary of a multi-method mitigation comparison at one h-point."""

    h_value: float = 0.0
    e_exact: float = 0.0
    gap: float = 0.0
    methods: dict[str, float] = field(default_factory=dict)  # method → ΔE/gap
    best_method: str = ""
    best_delta_e_gap: float = 0.0
    rankings: list[str] = field(default_factory=list)


@dataclass
class MitiqAnalysisReport:
    """Complete analysis report across all Mitiq comparison data."""

    n_comparisons: int = 0
    n_h_points: int = 0
    method_win_counts: dict[str, int] = field(default_factory=dict)
    method_mean_de_gap: dict[str, float] = field(default_factory=dict)
    best_overall_method: str = ""
    best_overall_de_gap: float = 0.0
    per_h: list[MitiqComparisonSummary] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Scanner
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_ROOT = Path("results")
REHEARSAL_DIR = RESULTS_ROOT / "experiments" / "exp_hw_rehearsal_v3"
MITIQ_DIR = RESULTS_ROOT / "mitiq"
HARDWARE_DIR = RESULTS_ROOT / "hardware"


def scan_mitiq_results() -> MitiqAnalysisReport:
    """Scan all sources for Mitiq comparison data and produce a report."""
    report = MitiqAnalysisReport()

    # Source 1: Rehearsal V3 Section 21
    _scan_rehearsal_section_21(report)

    # Source 2: Standalone mitiq benchmark results
    _scan_standalone_benchmarks(report)

    # Source 3: Hardware deployment mitiq verification
    _scan_hardware_mitiq(report)

    # Aggregate
    _compute_aggregates(report)

    return report


def _scan_rehearsal_section_21(report: MitiqAnalysisReport) -> None:
    """Scan rehearsal V3 results for section_21 data."""
    if not REHEARSAL_DIR.exists():
        return

    for f in sorted(REHEARSAL_DIR.glob("run_*.json")):
        try:
            data = json.loads(f.read_text())
            results = data.get("results", {})
            s21 = results.get("section_21", {})
            if not s21 or s21.get("skipped"):
                continue

            per_h = s21.get("per_h", [])
            for entry in per_h:
                if "delta_e_gaps" not in entry:
                    continue
                summary = MitiqComparisonSummary(
                    h_value=entry.get("h", 0.0),
                    e_exact=entry.get("e_exact", 0.0),
                    gap=entry.get("gap", 0.0),
                    methods=entry.get("delta_e_gaps", {}),
                    best_method=entry.get("best_method", ""),
                    best_delta_e_gap=entry.get("best_delta_e_gap", 0.0),
                    rankings=entry.get("rankings", []),
                )
                report.per_h.append(summary)

            report.source_files.append(str(f))
            report.n_comparisons += 1
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"Skipping {f.name}: {e}")


def _scan_standalone_benchmarks(report: MitiqAnalysisReport) -> None:
    """Scan standalone mitiq comparison results."""
    if not MITIQ_DIR.exists():
        return

    for f in sorted(MITIQ_DIR.glob("comparison_*.json")):
        try:
            data = json.loads(f.read_text())
            summary = MitiqComparisonSummary(
                h_value=data.get("h_value", 0.0),
                e_exact=data.get("e_exact", 0.0),
                gap=data.get("gap", 0.0),
                methods=data.get("delta_e_gaps", {}),
                best_method=data.get("best_method", ""),
                best_delta_e_gap=data.get("best_delta_e_gap", 0.0),
                rankings=data.get("rankings", []),
            )
            report.per_h.append(summary)
            report.source_files.append(str(f))
            report.n_comparisons += 1
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"Skipping {f.name}: {e}")


def _scan_hardware_mitiq(report: MitiqAnalysisReport) -> None:
    """Scan hardware deployment results for mitiq verification data."""
    if not HARDWARE_DIR.exists():
        return

    for run_dir in sorted(HARDWARE_DIR.glob("run_*")):
        summary_file = run_dir / "execution_summary.json"
        if not summary_file.exists():
            continue
        try:
            data = json.loads(summary_file.read_text())
            mitiq_data = data.get("mitiq_verification", {})
            if not mitiq_data:
                continue

            for entry in mitiq_data.get("per_h", []):
                if "delta_e_gaps" not in entry:
                    continue
                summary = MitiqComparisonSummary(
                    h_value=entry.get("h", 0.0),
                    e_exact=entry.get("e_exact", 0.0),
                    gap=entry.get("gap", 0.0),
                    methods=entry.get("delta_e_gaps", {}),
                    best_method=entry.get("best_method", ""),
                    best_delta_e_gap=entry.get("best_delta_e_gap", 0.0),
                    rankings=entry.get("rankings", []),
                )
                report.per_h.append(summary)

            report.source_files.append(str(summary_file))
            report.n_comparisons += 1
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"Skipping {run_dir.name}: {e}")


def _compute_aggregates(report: MitiqAnalysisReport) -> None:
    """Compute aggregate statistics from all scanned data."""
    if not report.per_h:
        return

    report.n_h_points = len(report.per_h)

    # Count wins per method
    win_counts: dict[str, int] = {}
    method_de_gaps: dict[str, list[float]] = {}

    for entry in report.per_h:
        if entry.best_method:
            win_counts[entry.best_method] = win_counts.get(entry.best_method, 0) + 1
        for method, de_gap in entry.methods.items():
            if method not in method_de_gaps:
                method_de_gaps[method] = []
            method_de_gaps[method].append(de_gap)

    report.method_win_counts = win_counts
    report.method_mean_de_gap = {m: float(np.mean(gaps)) for m, gaps in method_de_gaps.items()}

    # Best overall
    if report.method_mean_de_gap:
        report.best_overall_method = min(
            report.method_mean_de_gap, key=lambda m: report.method_mean_de_gap[m]
        )
        report.best_overall_de_gap = report.method_mean_de_gap[report.best_overall_method]


# ═══════════════════════════════════════════════════════════════════════════════
# Formatters
# ═══════════════════════════════════════════════════════════════════════════════


def format_report(report: MitiqAnalysisReport, verbose: bool = False) -> str:
    """Format the analysis report as a human-readable string."""
    lines = []
    lines.append("=" * 70)
    lines.append("  Mitiq Integration Analysis Report")
    lines.append("=" * 70)

    if report.n_comparisons == 0:
        lines.append("\n  No Mitiq comparison data found.")
        lines.append("  Run rehearsal V3 section 21 or standalone benchmark first.")
        lines.append(f"  Scanned: {REHEARSAL_DIR}, {MITIQ_DIR}, {HARDWARE_DIR}")
        return "\n".join(lines)

    lines.append(f"\n  Sources: {report.n_comparisons} files, {report.n_h_points} h-points")
    lines.append("")

    # Method win counts
    lines.append("  ── Method Win Counts ──")
    for method, count in sorted(report.method_win_counts.items(), key=lambda x: -x[1]):
        pct = count / report.n_h_points * 100 if report.n_h_points > 0 else 0
        lines.append(f"    {method:25s}: {count}/{report.n_h_points} ({pct:.0f}%)")

    lines.append("")

    # Mean ΔE/gap per method
    lines.append("  ── Mean ΔE/gap per Method ──")
    for method, de_gap in sorted(report.method_mean_de_gap.items(), key=lambda x: x[1]):
        marker = " ◄ BEST" if method == report.best_overall_method else ""
        lines.append(f"    {method:25s}: {de_gap:.4f}{marker}")

    lines.append("")
    lines.append(
        f"  Best overall: {report.best_overall_method} "
        f"(mean ΔE/gap = {report.best_overall_de_gap:.4f})"
    )

    # Per-h detail (verbose)
    if verbose and report.per_h:
        lines.append("")
        lines.append("  ── Per h-point Detail ──")
        for entry in report.per_h:
            lines.append(
                f"    h={entry.h_value:.3f}: best={entry.best_method} "
                f"(ΔE/gap={entry.best_delta_e_gap:.4f})"
            )
            if entry.rankings:
                lines.append(f"      Rankings: {' > '.join(entry.rankings)}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def format_thesis_table(report: MitiqAnalysisReport) -> str:
    """Format as a thesis-ready Markdown table."""
    if not report.per_h:
        return "No Mitiq comparison data available for thesis table."

    lines = []
    lines.append("| Method | Mean ΔE/gap | Win Rate | Rank |")
    lines.append("|--------|:---:|:---:|:---:|")

    sorted_methods = sorted(report.method_mean_de_gap.items(), key=lambda x: x[1])
    for rank, (method, de_gap) in enumerate(sorted_methods, 1):
        wins = report.method_win_counts.get(method, 0)
        win_pct = wins / report.n_h_points * 100 if report.n_h_points > 0 else 0
        lines.append(f"| {method} | {de_gap:.4f} | {win_pct:.0f}% | {rank} |")

    lines.append("")
    lines.append(f"*{report.n_h_points} h-points from {report.n_comparisons} comparison runs.*")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Health Report Integration (called by engine.py)
# ═══════════════════════════════════════════════════════════════════════════════


def get_mitiq_health_summary() -> dict:
    """Get a summary dict suitable for inclusion in the main health report.

    Called by engine.py Step 6c to include Mitiq status in HealthReport.

    Returns
    -------
    dict
        Keys: status, best_method, mean_de_gap, n_comparisons, findings, warnings.
    """
    report = scan_mitiq_results()

    if report.n_comparisons == 0:
        return {
            "status": "not_run",
            "message": "No Mitiq comparison data available",
            "recommendation": (
                "Run: python scripts/experiment_runners/run_hardware_rehearsal_v3.py --sections 21"
            ),
        }

    # Determine status based on best method performance
    if report.best_overall_de_gap < 0.05:
        status = "validated"
    elif report.best_overall_de_gap < 0.10:
        status = "partial"
    else:
        status = "not_validated"

    # Check if Mitiq methods beat native (raw)
    raw_mean = report.method_mean_de_gap.get("raw", float("inf"))
    mitiq_methods = {k: v for k, v in report.method_mean_de_gap.items() if k != "raw"}
    mitiq_beats_raw = any(v < raw_mean for v in mitiq_methods.values())

    findings = []
    if mitiq_beats_raw:
        best_mitiq = min(mitiq_methods, key=lambda k: mitiq_methods[k])
        improvement = (1 - mitiq_methods[best_mitiq] / raw_mean) * 100 if raw_mean > 0 else 0
        findings.append(f"Best Mitiq method ({best_mitiq}) improves {improvement:.0f}% over raw")

    warnings = []
    if not mitiq_beats_raw:
        warnings.append("No Mitiq method beats raw — may indicate insufficient noise")

    return {
        "status": status,
        "best_method": report.best_overall_method,
        "best_mean_de_gap": report.best_overall_de_gap,
        "n_comparisons": report.n_comparisons,
        "n_h_points": report.n_h_points,
        "method_rankings": dict(sorted(report.method_mean_de_gap.items(), key=lambda x: x[1])),
        "mitiq_beats_raw": mitiq_beats_raw,
        "findings": findings,
        "warnings": warnings,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical Analysis (uses project_health.analysis.statistical_tests)
# ═══════════════════════════════════════════════════════════════════════════════


def run_statistical_analysis(report: MitiqAnalysisReport) -> dict:
    """Run rigorous statistical tests comparing Mitiq methods vs raw.

    Uses paired_ttest from project_health.analysis.statistical_tests
    to determine if each Mitiq method is statistically better than raw.

    Returns
    -------
    dict
        Statistical analysis with paired t-tests per method.
    """
    from project_health.analysis.statistical_tests import (
        effect_size_cohens_d,
        improvement_rate,
        paired_ttest,
    )

    if not report.per_h:
        return {"available": False, "reason": "No comparison data"}

    # Collect paired data: raw ΔE/gap vs each method ΔE/gap
    raw_de_gaps = [entry.methods.get("raw", float("nan")) for entry in report.per_h]
    if all(np.isnan(r) for r in raw_de_gaps):
        return {"available": False, "reason": "No raw baseline in data"}

    results = {"available": True, "methods": {}}

    for method in report.method_mean_de_gap:
        if method == "raw":
            continue

        method_de_gaps = [entry.methods.get(method, float("nan")) for entry in report.per_h]

        # Filter pairs where both exist
        pairs = [
            (r, m)
            for r, m in zip(raw_de_gaps, method_de_gaps, strict=False)
            if not np.isnan(r) and not np.isnan(m)
        ]
        if len(pairs) < 2:
            continue

        raw_vals = [p[0] for p in pairs]
        method_vals = [p[1] for p in pairs]

        try:
            ttest = paired_ttest(raw_vals, method_vals, alternative="greater")
            imp = improvement_rate(raw_vals, method_vals)
            d = effect_size_cohens_d(raw_vals, method_vals)

            results["methods"][method] = {
                "n_pairs": len(pairs),
                "paired_ttest": ttest,
                "improvement_rate": imp,
                "cohens_d": d,
                "effect": "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small",
                "significant": ttest.get("significant_005", False),
            }
        except (ValueError, ZeroDivisionError):
            continue

    return results


def _print_statistical_report(stats: dict) -> None:
    """Print formatted statistical analysis."""
    if not stats.get("available"):
        print(f"\n  Statistical analysis unavailable: {stats.get('reason', 'unknown')}")
        return

    print("\n  ── Statistical Analysis: Mitiq vs Raw ──")
    print("  " + "═" * 55)

    for method, data in stats.get("methods", {}).items():
        t = data["paired_ttest"]
        sig = "✅" if data["significant"] else "❌"
        print(f"\n  {method}:")
        print(
            f"    Paired t-test (H₁: Mitiq < raw): t={t['t_stat']:.3f}, p={t['p_value']:.4f} {sig}"
        )
        print(f"    Cohen's d: {data['cohens_d']:.2f} ({data['effect']})")
        imp = data["improvement_rate"]
        print(
            f"    Improvement: {imp['n_improved']}/{imp['n']} wins ({imp['improvement_rate_pct']:.0f}%)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Mitiq Integration Analyzer — multi-method comparison results"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Per-h detail")
    parser.add_argument("--json", type=str, metavar="PATH", help="Export JSON report")
    parser.add_argument("--thesis-table", action="store_true", help="Print thesis table")
    parser.add_argument(
        "--statistical",
        action="store_true",
        help="Run rigorous statistical tests (paired t-test for Mitiq vs raw)",
    )
    parser.add_argument(
        "--health-summary",
        action="store_true",
        help="Print health-check compatible summary dict",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    report = scan_mitiq_results()

    if args.thesis_table:
        print(format_thesis_table(report))
    else:
        print(format_report(report, verbose=args.verbose))

    if args.statistical:
        stats = run_statistical_analysis(report)
        _print_statistical_report(stats)

    if args.health_summary:
        import pprint

        summary = get_mitiq_health_summary()
        print("\n  Health Summary (for engine.py integration):")
        pprint.pprint(summary, indent=4)

    if args.json:
        from dataclasses import asdict

        from qmbp_simulation.utils.helpers import json_dump

        export = asdict(report)
        stats = run_statistical_analysis(report)
        if stats.get("available"):
            export["statistical_analysis"] = stats
        json_dump(export, Path(args.json))
        logger.info(f"  JSON exported to {args.json}")


if __name__ == "__main__":
    main()
