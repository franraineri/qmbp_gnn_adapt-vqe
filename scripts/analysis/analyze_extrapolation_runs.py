#!/usr/bin/env python3
"""Analyze specific large-N extrapolation runs by timestamp filter.

Reads result JSONs from exp_large_n_extrap/, filters by date/time range,
and produces a consolidated markdown report with per-topology, per-N metrics.

Usage:
    # Analyze all runs from today
    .venv/bin/python scripts/analysis/analyze_extrapolation_runs.py

    # Analyze runs from a specific time window
    .venv/bin/python scripts/analysis/analyze_extrapolation_runs.py --after 20260812_170000

    # Analyze only runs that used --force-recompute (pure evaluation)
    .venv/bin/python scripts/analysis/analyze_extrapolation_runs.py --after 20260812_170000 --force-only

    # Filter by topology
    .venv/bin/python scripts/analysis/analyze_extrapolation_runs.py --topology chain_1d

    # Output to specific file
    .venv/bin/python scripts/analysis/analyze_extrapolation_runs.py --output results/my_analysis.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "experiments" / "exp_large_n_extrap"
DEFAULT_OUTPUT = ROOT / "results" / "extrapolation_analysis.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze large-N extrapolation run results")
    parser.add_argument(
        "--after",
        type=str,
        default=None,
        help="Only include runs after this timestamp (format: YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--before",
        type=str,
        default=None,
        help="Only include runs before this timestamp",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default=None,
        help="Filter by topology",
    )
    parser.add_argument(
        "--force-only",
        action="store_true",
        help="Only include runs with force_recompute=True",
    )
    parser.add_argument(
        "--checkpoint-filter",
        type=str,
        default=None,
        help="Only include runs that used this checkpoint (substring match)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output markdown path (default: {DEFAULT_OUTPUT.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print to stdout only, don't save file",
    )
    return parser.parse_args()


def load_and_filter_runs(args: argparse.Namespace) -> list[dict]:
    """Load all matching run JSONs."""
    runs = []
    for f in sorted(RESULTS_DIR.glob("run_*.json")):
        # Timestamp filter from filename
        ts = f.stem.replace("run_", "")
        if args.after and ts < args.after:
            continue
        if args.before and ts > args.before:
            continue

        with open(f) as fp:
            data = json.load(fp)

        cfg = data.get("config", {})

        # Topology filter
        if args.topology and cfg.get("topology") != args.topology:
            continue

        # Force-recompute filter
        if args.force_only and not cfg.get("force_recompute", False):
            continue

        # Checkpoint filter
        if args.checkpoint_filter:
            ckpt = cfg.get("checkpoint") or ""
            if args.checkpoint_filter not in ckpt:
                continue

        # Only include successful runs
        if not data.get("summary", {}).get("all_passed", False):
            continue

        runs.append({"file": f.name, "timestamp": ts, "data": data})

    return runs


def extract_metrics(run: dict) -> dict:
    """Extract key metrics from a single run."""
    data = run["data"]
    cfg = data["config"]
    results = data["results"]

    topo = cfg.get("topology", "?")
    target_n = cfg.get("target_n", [])
    h_range = cfg.get("h_range", [0, 0])
    refine = cfg.get("refine_failing", False)

    # Get checkpoint used (canonical utility from result_io)
    from qmbp_simulation.framework.result_io import extract_checkpoint_used

    checkpoint = extract_checkpoint_used(data)

    # Extract MPNN results
    mpnn_sec = results.get("section_2", {}).get("data", {}).get("mpnn_results", {})
    per_n_metrics = []

    for n_str, mdata in mpnn_sec.items():
        if not isinstance(mdata, dict):
            continue
        n = mdata.get("n_qubits", int(n_str))
        per_point = mdata.get("per_point", [])

        # Compute metrics from per_point
        de_gaps = [p["de_gap"] for p in per_point if "de_gap" in p]
        abs_errors = [p.get("abs_error", 0) for p in per_point if "abs_error" in p]
        methods = [p.get("method", "?") for p in per_point]

        n_refined = methods.count("vqe_refined")

        entry = {
            "n": n,
            "n_pts": len(per_point),
            "mean_de_gap": float(np.mean(de_gaps)) if de_gaps else None,
            "max_de_gap": float(np.max(de_gaps)) if de_gaps else None,
            "mean_abs_error": float(np.mean(abs_errors)) if abs_errors else None,
            "per_site_error": float(np.mean(abs_errors)) / n if abs_errors else None,
            "pass_5pct": sum(1 for d in de_gaps if d < 0.05),
            "pass_dual": mdata.get("n_pass_dual", 0),
            "n_refined": n_refined,
        }
        per_n_metrics.append(entry)

    # Extract diagnostics from summary section
    diag = {}
    for sk in ["section_3", "section_4", "section_5"]:
        sec = results.get(sk, {}).get("data", {})
        if "comparison" in sec:
            comp = sec["comparison"]
            if "model_diagnostics" in comp:
                diag = comp["model_diagnostics"]
            break

    return {
        "file": run["file"],
        "timestamp": run["timestamp"],
        "topology": topo,
        "target_n": target_n,
        "h_range": h_range,
        "checkpoint": checkpoint,
        "refine": refine,
        "per_n": sorted(per_n_metrics, key=lambda x: x["n"]),
        "diagnostics": diag,
        "elapsed_s": data.get("elapsed_s", 0),
    }


def format_report(all_metrics: list[dict], args: argparse.Namespace) -> str:
    """Format all metrics as a markdown report."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    n_runs = len(all_metrics)

    lines = [
        "# Extrapolation Run Analysis",
        "",
        f"**Generated**: {now}",
        f"**Runs analyzed**: {n_runs}",
        f"**Filter**: after={args.after or 'all'}, topology={args.topology or 'all'}, "
        f"force_only={args.force_only}",
        "",
    ]

    if not all_metrics:
        lines.append("*No matching runs found.*")
        return "\n".join(lines)

    # Group by topology
    by_topo = defaultdict(list)
    for m in all_metrics:
        by_topo[m["topology"]].append(m)

    lines.append("---")
    lines.append("")

    # Per-topology detailed table
    for topo in sorted(by_topo.keys()):
        runs = by_topo[topo]
        lines.append(f"## {topo}")
        lines.append("")

        # Show which checkpoint was used
        checkpoints = set(r["checkpoint"] for r in runs)
        for ckpt in checkpoints:
            lines.append(f"**Checkpoint**: `{ckpt}`")
        lines.append("")

        # Consolidated per-N table (best result per N across runs)
        best_per_n = {}
        for run in runs:
            for entry in run["per_n"]:
                n = entry["n"]
                if n not in best_per_n or (entry["mean_de_gap"] or 999) < (
                    best_per_n[n]["mean_de_gap"] or 999
                ):
                    best_per_n[n] = {**entry, "from_run": run["file"], "refine": run["refine"]}

        lines.append(
            "| N | Pts | ΔE/gap | |ΔE| | |ΔE|/N | Pass@5% | Pass@dual | Refined | Source |"
        )
        lines.append(
            "|---|-----|--------|------|--------|---------|-----------|---------|--------|"
        )

        for n in sorted(best_per_n.keys()):
            e = best_per_n[n]
            dg = f"{e['mean_de_gap']:.4f}" if e["mean_de_gap"] is not None else "—"
            ae = f"{e['mean_abs_error']:.4f}" if e["mean_abs_error"] is not None else "—"
            ps = f"{e['per_site_error']:.2e}" if e["per_site_error"] is not None else "—"
            p5 = f"{e['pass_5pct']}/{e['n_pts']}"
            pd = f"{e['pass_dual']}/{e['n_pts']}"
            ref = f"{e['n_refined']}" if e["n_refined"] > 0 else "—"
            src = e["from_run"][:20]
            lines.append(
                f"| {n} | {e['n_pts']} | {dg} | {ae} | {ps} | {p5} | {pd} | {ref} | {src} |"
            )

        lines.append("")

        # Diagnostics if available
        for run in runs:
            diag = run.get("diagnostics", {})
            if diag.get("scaling_fit"):
                sf = diag["scaling_fit"]
                lines.append(
                    f"**Scaling fit**: |ΔE|/N ∝ N^{sf['alpha']:.2f} — {sf['interpretation']}"
                )
                lines.append("")
            if diag.get("variational_violations"):
                vv = diag["variational_violations"]
                total_v = sum(v["n_violations"] for v in vv.values())
                total_p = sum(v["n_total"] for v in vv.values())
                if total_v > 0:
                    lines.append(f"**Variational violations**: {total_v}/{total_p} points")
                lines.append("")
                break  # Only show once

    # Cross-topology comparison table
    lines.append("---")
    lines.append("")
    lines.append("## Cross-Topology Comparison")
    lines.append("")
    lines.append("| Topology | Checkpoint | N tested | Mean |ΔE|/N | Best ΔE/gap | Verdict |")
    lines.append("|----------|-----------|----------|:---:|:---:|:---:|")

    for topo in sorted(by_topo.keys()):
        runs = by_topo[topo]
        all_per_site = []
        all_de_gap = []
        all_n = set()
        ckpt = runs[0]["checkpoint"]

        for run in runs:
            for e in run["per_n"]:
                all_n.add(e["n"])
                if e["per_site_error"] is not None:
                    all_per_site.append(e["per_site_error"])
                if e["mean_de_gap"] is not None:
                    all_de_gap.append(e["mean_de_gap"])

        mean_ps = f"{np.mean(all_per_site):.2e}" if all_per_site else "—"
        best_dg = f"{min(all_de_gap):.4f}" if all_de_gap else "—"
        n_str = ",".join(str(n) for n in sorted(all_n))

        # Verdict
        if all_per_site and np.mean(all_per_site) < 0.010:
            verdict = "✅ Excellent"
        elif all_per_site and np.mean(all_per_site) < 0.020:
            verdict = "⚠️ Good"
        elif all_per_site and np.mean(all_per_site) < 0.035:
            verdict = "⚠️ Acceptable"
        else:
            verdict = "❌ Poor"

        ckpt_short = Path(ckpt).stem[:40] if ckpt != "auto (zoo)" else "auto"
        lines.append(f"| {topo} | {ckpt_short} | {n_str} | {mean_ps} | {best_dg} | {verdict} |")

    lines.append("")
    lines.append("---")
    lines.append("*Generated by `scripts/analysis/analyze_extrapolation_runs.py`*")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    print("Loading runs...", flush=True)
    runs = load_and_filter_runs(args)
    print(f"  Found {len(runs)} matching runs", flush=True)

    if not runs:
        print("  No matching runs. Check --after filter or results directory.")
        return 1

    print("Extracting metrics...", flush=True)
    all_metrics = [extract_metrics(r) for r in runs]

    # Print console summary
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    for m in all_metrics:
        print(f"\n  {m['topology']:12} N={m['target_n']} ({m['file']})")
        for e in m["per_n"]:
            dg = f"ΔE/gap={e['mean_de_gap']:.4f}" if e["mean_de_gap"] else ""
            ps = f"|ΔE|/N={e['per_site_error']:.2e}" if e["per_site_error"] else ""
            print(f"    N={e['n']:>3}: {dg} {ps} pass={e['pass_5pct']}/{e['n_pts']}")

    # Generate and save report
    report = format_report(all_metrics, args)
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT

    if not args.no_save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"\n  ✅ Report saved: {output_path}")
    else:
        print(f"\n{report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
