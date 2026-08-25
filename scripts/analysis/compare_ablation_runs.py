#!/usr/bin/env python3
"""Compare architecture ablation results across runs.

Reads all JSON files from results/arch_ablation/ and produces a summary
table showing how each variant evolved over time. Useful for tracking
whether new data/epochs improve model quality.

Usage:
    .venv/bin/python scripts/analysis/compare_ablation_runs.py
    .venv/bin/python scripts/analysis/compare_ablation_runs.py --topology chain_1d
    .venv/bin/python scripts/analysis/compare_ablation_runs.py --latest 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ABLATION_DIR = ROOT / "results" / "arch_ablation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ablation runs over time")
    parser.add_argument("--topology", default=None, help="Filter by topology")
    parser.add_argument("--latest", type=int, default=5, help="Show N latest runs (default: 5)")
    parser.add_argument("--variant", default=None, help="Filter by variant name")
    return parser.parse_args()


def load_runs(topology: str | None, latest: int) -> list[dict]:
    """Load ablation result JSONs, sorted by timestamp."""
    if not ABLATION_DIR.exists():
        return []

    runs = []
    for f in sorted(ABLATION_DIR.glob("ablation_*.json")):
        try:
            data = json.loads(f.read_text())
            if topology and data.get("topology") != topology:
                continue
            data["_file"] = f.name
            runs.append(data)
        except (json.JSONDecodeError, KeyError):
            continue

    return runs[-latest:]


def format_table(runs: list[dict], variant_filter: str | None) -> str:
    """Format a comparison table."""
    if not runs:
        return "No ablation results found."

    # Collect all variant names across runs
    all_variants = set()
    for run in runs:
        for r in run.get("results", []):
            if "name" in r:
                all_variants.add(r["name"])
    all_variants = sorted(all_variants)

    if variant_filter:
        all_variants = [v for v in all_variants if variant_filter.lower() in v.lower()]

    # Header
    lines = []
    lines.append(f"{'Run':<30} | {'Topo':<12} | {'Epochs':>6} | {'Graphs':>6} | "
                 + " | ".join(f"{v:<12}" for v in all_variants))
    lines.append("-" * len(lines[0]))

    for run in runs:
        date = run.get("timestamp", "")[:16]
        topo = run.get("topology", "?")[:12]
        epochs = run.get("max_epochs", "?")
        n_graphs = run.get("n_training_graphs", "?")

        # Build variant MSE map
        variant_mse = {}
        for r in run.get("results", []):
            name = r.get("name", "")
            val_mse = r.get("val_mse")
            if val_mse is not None:
                variant_mse[name] = val_mse

        cells = []
        for v in all_variants:
            if v in variant_mse:
                cells.append(f"{variant_mse[v]:.2e}")
            else:
                cells.append("—")

        lines.append(f"{date:<30} | {topo:<12} | {epochs:>6} | {n_graphs:>6} | "
                     + " | ".join(f"{c:<12}" for c in cells))

    # Add improvement summary
    if len(runs) >= 2:
        lines.append("")
        lines.append("Improvement (first → last):")
        first_run = runs[0]
        last_run = runs[-1]
        for v in all_variants:
            first_mse = next(
                (r.get("val_mse") for r in first_run.get("results", [])
                 if r.get("name") == v and r.get("val_mse")), None
            )
            last_mse = next(
                (r.get("val_mse") for r in last_run.get("results", [])
                 if r.get("name") == v and r.get("val_mse")), None
            )
            if first_mse and last_mse:
                pct = (first_mse - last_mse) / first_mse * 100
                arrow = "↓" if pct > 0 else "↑"
                lines.append(f"  {v:<15}: {first_mse:.2e} → {last_mse:.2e} ({arrow}{abs(pct):.1f}%)")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    runs = load_runs(args.topology, args.latest)

    print("=" * 70)
    print("  ABLATION RUN COMPARISON")
    if args.topology:
        print(f"  Filter: topology={args.topology}")
    print(f"  Runs found: {len(runs)}")
    print("=" * 70)
    print()
    print(format_table(runs, args.variant))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
