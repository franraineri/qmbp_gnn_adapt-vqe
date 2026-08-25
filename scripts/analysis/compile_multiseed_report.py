#!/usr/bin/env python3
"""Compile multi-seed evaluation results into a markdown report.

Scans the most recent exp_accel_cross_n results, groups by topology,
and generates results/multiseed_evaluation.md with per-seed breakdown
and summary statistics.

Usage:
    .venv/bin/python scripts/analysis/compile_multiseed_report.py
    .venv/bin/python scripts/analysis/compile_multiseed_report.py --last 12
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "experiments" / "exp_accel_cross_n"
REPORT_PATH = ROOT / "results" / "multiseed_evaluation.md"

COORD_MAP = {"chain_1d": 2, "heavy_hex": 3, "square": 4, "triangular": 6, "ladder": 3}
TARGET_N = {"chain_1d": 10, "heavy_hex": 10, "square": 10, "triangular": 6}


def extract_metrics(result_path: Path) -> dict | None:
    """Extract cross-N metrics from a result JSON."""
    try:
        with open(result_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    config = data.get("config", {})
    topo = config.get("topology", "")
    if not topo:
        return None

    # Navigate: results → section_N → data → cross_n_results
    results_block = data.get("results", data.get("sections", {}))
    for key, sec in results_block.items():
        if not isinstance(sec, dict):
            continue
        # Check in sec directly or in sec["data"]
        search_in = [sec]
        if "data" in sec and isinstance(sec["data"], dict):
            search_in.append(sec["data"])

        for container in search_in:
            if "cross_n_results" in container:
                for k, v in container["cross_n_results"].items():
                    if "pass_rate_5pct" in v:
                        return {
                            "topology": topo,
                            "file": result_path.name,
                            "timestamp": data.get("timestamp", ""),
                            "pass_5pct": v["pass_rate_5pct"],
                            "pass_10pct": v.get("pass_rate_10pct", 0),
                            "pass_dual": v.get("pass_rate_dual", v["pass_rate_5pct"]),
                            "mean_de_gap": v.get("mean_de_gap", 0),
                            "mean_abs_error": v.get("mean_abs_error", 0),
                            "mean_fidelity": v.get("mean_fidelity"),
                            "n_points": v.get("n_points", 0),
                            "target_n": v.get("target_n", config.get("target_n", ["?"])),
                            "elapsed_s": v.get("elapsed_s", 0),
                        }
    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compile multiseed report")
    parser.add_argument("--last", type=int, default=12, help="Number of recent runs to scan")
    args = parser.parse_args()

    # Scan most recent results
    all_jsons = sorted(RESULTS_DIR.glob("run_*.json"), key=lambda p: p.stat().st_mtime)
    recent = all_jsons[-args.last :]

    print(f"Scanning {len(recent)} most recent results...")

    # Extract and group
    by_topo: dict[str, list[dict]] = defaultdict(list)
    for rpath in recent:
        metrics = extract_metrics(rpath)
        if metrics:
            by_topo[metrics["topology"]].append(metrics)

    if not by_topo:
        print("ERROR: No valid results found!")
        return 1

    # Generate report
    lines = [
        "# Multi-Seed Cross-Topology Evaluation",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "**Model**: tfim_bond_resolved, p=1",
        "**Method**: UnifiedMPNN retrained per seed (QMBP_GLOBAL_SEED)",
        f"**Runs scanned**: {len(recent)} (most recent from exp_accel_cross_n/)",
        "",
        "---",
        "",
    ]

    # Per-topology tables
    for topo in ["chain_1d", "heavy_hex", "square", "triangular"]:
        runs = by_topo.get(topo, [])
        if not runs:
            continue

        n_target = runs[0].get("target_n", "?")
        z = COORD_MAP.get(topo, "?")
        lines.append(f"## {topo} (z={z}, N={n_target})")
        lines.append("")
        lines.append(
            "| # | Pass@5% | Pass@dual | Mean ΔE/gap | Mean |ΔE| | "
            "Fidelity | Points | Time | File |"
        )
        lines.append(
            "|---|---------|-----------|-------------|-----------|"
            "----------|--------|------|------|"
        )

        for i, r in enumerate(runs, 1):
            fid = f"{r['mean_fidelity']:.4f}" if r.get("mean_fidelity") else "—"
            lines.append(
                f"| {i} | {r['pass_5pct']:.0%} | {r['pass_dual']:.0%} | "
                f"{r['mean_de_gap']:.4f} | {r['mean_abs_error']:.4f} | "
                f"{fid} | {r['n_points']} | {r['elapsed_s']:.0f}s | "
                f"`{r['file'][-25:]}` |"
            )

        lines.append("")

    # Summary table
    lines.append("---")
    lines.append("")
    lines.append("## Summary (mean ± std across seeds)")
    lines.append("")
    lines.append("| Topology | z | N | Seeds | Pass@5% | Pass@dual | Mean ΔE/gap | Verdict |")
    lines.append("|----------|---|---|-------|---------|-----------|-------------|---------|")

    for topo in ["chain_1d", "heavy_hex", "square", "triangular"]:
        runs = by_topo.get(topo, [])
        if not runs:
            continue

        z = COORD_MAP.get(topo, "?")
        n = TARGET_N.get(topo, "?")
        p5 = [r["pass_5pct"] for r in runs]
        pd = [r["pass_dual"] for r in runs]
        de = [r["mean_de_gap"] for r in runs]

        mean_p5 = np.mean(p5)
        std_p5 = np.std(p5)
        mean_pd = np.mean(pd)
        std_pd = np.std(pd)
        mean_de_val = np.mean(de)

        # Verdict
        if std_pd < 0.05:
            verdict = "✅ Robust"
        elif std_pd < 0.10:
            verdict = "⚠️ Moderate"
        else:
            verdict = "❌ Unstable"

        lines.append(
            f"| {topo} | {z} | {n} | {len(runs)} | "
            f"{mean_p5:.0%} ± {std_p5:.0%} | "
            f"{mean_pd:.0%} ± {std_pd:.0%} | "
            f"{mean_de_val:.4f} | {verdict} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by `scripts/analysis/compile_multiseed_report.py`*")

    # Write report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"Report written: {REPORT_PATH}")
    print(f"  Topologies: {list(by_topo.keys())}")
    print(f"  Total runs: {sum(len(v) for v in by_topo.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
