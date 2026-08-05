#!/usr/bin/env python3
"""Accelerated Cross-N Analyzer — Post-run analysis for accelerated experiments.

Scans results from ACCEL_CROSS_N experiments and produces:
- Per-h ΔE/gap breakdown (which h-regions work, which don't)
- Cross-N scaling analysis (does error grow with N_target?)
- Model reuse effectiveness (zoo hit rate, time savings)
- Training data utilization (anchor efficiency)
- Comparison: accelerated vs full VQE (if both exist)

Usage:
    python -m project_health.analysis.accelerated_cross_n_analyzer
    python -m project_health.analysis.accelerated_cross_n_analyzer --verbose
    python -m project_health.analysis.accelerated_cross_n_analyzer --compare-full
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "experiments"


@dataclass
class CrossNAnalysis:
    """Analysis summary for one cross-N prediction run."""

    train_n: int
    target_n: int
    p_layers: int
    topology: str
    n_points: int = 0
    pass_rate_5pct: float = 0.0
    pass_rate_10pct: float = 0.0
    mean_de_gap: float = 0.0
    mean_abs_error: float = 0.0
    time_s: float = 0.0
    # h-region breakdown
    h_easy_pass_rate: float = 0.0   # h > 2.5
    h_medium_pass_rate: float = 0.0  # 2.0 < h <= 2.5
    h_hard_pass_rate: float = 0.0    # h <= 2.0


@dataclass
class AcceleratedReport:
    """Complete analysis report."""

    analyses: list[CrossNAnalysis] = field(default_factory=list)
    zoo_entries_used: int = 0
    total_training_time_s: float = 0.0
    total_prediction_time_s: float = 0.0
    data_reuse_summary: dict[str, Any] = field(default_factory=dict)


def scan_results() -> list[dict[str, Any]]:
    """Find all ACCEL_CROSS_N results in the experiments directory."""
    results = []
    patterns = [
        RESULTS_DIR / "exp_accel_cross_n",
        RESULTS_DIR / "exp_accelerated_cross_n",
        RESULTS_DIR / "exp_mpnn_warmstart_accelerator",
    ]
    for base in patterns:
        if not base.exists():
            continue
        for f in sorted(base.rglob("run_*.json")):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                results.append(data)
            except (json.JSONDecodeError, OSError):
                continue
    return results


def analyze_cross_n_result(data: dict) -> list[CrossNAnalysis]:
    """Extract cross-N analysis from a single result file."""
    analyses = []
    results_section = data.get("results", {})

    # Look for section 3 (cross-N predict) data
    for key, section in results_section.items():
        if not isinstance(section, dict):
            continue
        section_data = section.get("data", {})
        cross_n = section_data.get("cross_n_results", {})

        for config_key, result in cross_n.items():
            if not isinstance(result, dict) or "per_point" not in result:
                continue

            per_point = result["per_point"]
            if not per_point:
                continue

            # h-region breakdown
            easy = [r for r in per_point if r["h"] > 2.5]
            medium = [r for r in per_point if 2.0 < r["h"] <= 2.5]
            hard = [r for r in per_point if r["h"] <= 2.0]

            analysis = CrossNAnalysis(
                train_n=result.get("train_n", 0),
                target_n=result.get("target_n", 0),
                p_layers=result.get("p_layers", 1),
                topology=data.get("config", {}).get("topology", "chain_1d"),
                n_points=len(per_point),
                pass_rate_5pct=result.get("pass_rate_5pct", 0.0),
                pass_rate_10pct=result.get("pass_rate_10pct", 0.0),
                mean_de_gap=result.get("mean_de_gap", 0.0),
                mean_abs_error=result.get("mean_abs_error", 0.0),
                time_s=result.get("elapsed_s", 0.0),
                h_easy_pass_rate=(
                    np.mean([1.0 if r["de_gap"] < 0.05 else 0.0 for r in easy])
                    if easy else 0.0
                ),
                h_medium_pass_rate=(
                    np.mean([1.0 if r["de_gap"] < 0.05 else 0.0 for r in medium])
                    if medium else 0.0
                ),
                h_hard_pass_rate=(
                    np.mean([1.0 if r["de_gap"] < 0.05 else 0.0 for r in hard])
                    if hard else 0.0
                ),
            )
            analyses.append(analysis)

    return analyses


def format_report(report: AcceleratedReport) -> str:
    """Format the analysis report as text."""
    lines = [
        "═" * 60,
        "  ACCELERATED CROSS-N ANALYSIS REPORT",
        "═" * 60,
        "",
        f"Total runs analyzed: {len(report.analyses)}",
        f"Training time total: {report.total_training_time_s:.0f}s",
        f"Prediction time total: {report.total_prediction_time_s:.0f}s",
        "",
        "─── Per-Config Results ───",
        "",
        f"{'Config':<30} {'@5%':<8} {'@10%':<8} {'mean ΔE/gap':<12} {'h>2.5':<8} {'2.0<h<2.5':<10}",
        "-" * 80,
    ]

    for a in sorted(report.analyses, key=lambda x: (x.topology, x.p_layers, x.target_n)):
        config = f"{a.topology[:6]} N={a.train_n}→{a.target_n} p={a.p_layers}"
        lines.append(
            f"{config:<30} {a.pass_rate_5pct:.0%}     {a.pass_rate_10pct:.0%}     "
            f"{a.mean_de_gap:.4f}       {a.h_easy_pass_rate:.0%}     "
            f"{a.h_medium_pass_rate:.0%}"
        )

    # Key findings
    lines.extend(["", "─── Key Findings ───", ""])
    if report.analyses:
        best = max(report.analyses, key=lambda a: a.pass_rate_10pct)
        lines.append(f"  Best config: N={best.train_n}→{best.target_n} p={best.p_layers}")
        lines.append(f"    Pass rate @10%: {best.pass_rate_10pct:.0%}")
        lines.append(f"    h>2.5 region: {best.h_easy_pass_rate:.0%} (easiest)")
        lines.append(f"    h<2.0 region: {best.h_hard_pass_rate:.0%} (hardest)")

        # Does error grow with N?
        n_targets = sorted(set(a.target_n for a in report.analyses))
        if len(n_targets) > 1:
            lines.append("")
            lines.append("  N-scaling trend:")
            for n in n_targets:
                subset = [a for a in report.analyses if a.target_n == n]
                avg_de_gap = np.mean([a.mean_de_gap for a in subset])
                lines.append(f"    N={n}: mean ΔE/gap = {avg_de_gap:.4f}")

    lines.extend(["", "═" * 60])
    return "\n".join(lines)


def main():
    """Run the analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Accelerated Cross-N Analyzer")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    # Scan results
    raw_results = scan_results()
    logger.info(f"Found {len(raw_results)} result files")

    # Analyze
    report = AcceleratedReport()
    for data in raw_results:
        analyses = analyze_cross_n_result(data)
        report.analyses.extend(analyses)

        # Timing
        elapsed = data.get("elapsed_s", 0)
        config = data.get("config", {})
        if config.get("from_zoo"):
            report.total_prediction_time_s += elapsed
            report.zoo_entries_used += 1
        else:
            report.total_training_time_s += elapsed

    if not report.analyses:
        print("No accelerated cross-N results found.")
        print(f"Run: python scripts/.../run_accelerated_cross_n.py")
        return

    # Print report
    print(format_report(report))


if __name__ == "__main__":
    main()
