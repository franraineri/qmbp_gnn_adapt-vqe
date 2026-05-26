#!/usr/bin/env python3
"""Cross-experiment and pipeline result comparison CLI.

Each experiment is evaluated against its own success criteria — not a
blanket ΔE/gap baseline. Verdicts: confirmed (hypothesis holds),
rejected (hypothesis disproved = valid finding), failed (unexpected).

Usage:
    python scripts/compare.py --all
    python scripts/compare.py --exp G1 G5
    python scripts/compare.py --category G
    python scripts/compare.py --noisy
    python scripts/compare.py --noisy --group-by seed_layout
    python scripts/compare.py --all --json output.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --all                          Compare all experiments
    %(prog)s --exp G1 G5 B4                 Compare specific experiments
    %(prog)s --category G                   Compare by category letter
    %(prog)s --noisy                        Analyze ZNE robustness results
    %(prog)s --noisy --group-by n_layouts   Group noisy results by key
    %(prog)s --all --json results.json      Save JSON output
        """,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="Compare all experiments")
    mode.add_argument("--exp", nargs="+", dest="experiments", help="Experiment IDs")
    mode.add_argument("--category", type=str, help="Category letter or name")
    mode.add_argument("--noisy", action="store_true", help="Analyze noisy/ZNE results")

    parser.add_argument("--group-by", type=str, help="Group noisy results by key")
    parser.add_argument("--noisy-file", type=str, help="Specific noisy result file")
    parser.add_argument(
        "--format", choices=["table", "json"], default="table", help="Output format"
    )
    parser.add_argument("--json", type=str, metavar="FILE", dest="json_file", help="Save to file")
    parser.add_argument("--results-dir", type=str, default=None, help="Results directory")

    return parser.parse_args()


def _run_experiment_comparison(store, exp_ids: list[str], args) -> None:
    """Run experiment comparison mode."""
    comparisons = store.compare_experiments(exp_ids)
    if not comparisons:
        print("No comparable results found.")
        return

    if args.json_file:
        _write_json(comparisons, args.json_file)
        return
    if args.format == "json":
        print(json.dumps(comparisons, indent=2, default=str))
        return

    print("\nExperiment Results Summary")
    print("=" * 80)
    print(store.format_experiment_table(comparisons))
    print()

    confirmed = [c for c in comparisons if c["verdict"] == "confirmed"]
    rejected = [c for c in comparisons if c["verdict"] == "rejected"]
    failed = [c for c in comparisons if c["verdict"] == "failed"]
    print(f"  {len(confirmed)} confirmed ✅  {len(rejected)} rejected ⚠️  {len(failed)} failed ❌")

    if rejected:
        print("\n  Rejected hypotheses (valid findings):")
        for c in rejected:
            print(f"    {c['experiment_id']}: {c['hypothesis'][:65]}")
    if failed:
        print("\n  Failed experiments:")
        for c in failed:
            print(f"    {c['experiment_id']}: {c['hypothesis'][:65]}")


def _run_noisy_analysis(store, args) -> None:
    """Run noisy/ZNE analysis mode."""
    results = store.load_noisy_results(filename=args.noisy_file)
    if not results:
        print("No noisy experiment results found in exp_noisy_variants/")
        return

    if args.json_file:
        output = {
            "correlations": store.analyze_noisy_correlations(results),
            "by_group": (
                {args.group_by: store.analyze_noisy_by_group(results, args.group_by)}
                if args.group_by
                else {}
            ),
        }
        _write_json(output, args.json_file)
        return

    if args.format == "json":
        correlations = store.analyze_noisy_correlations(results)
        print(json.dumps(correlations, indent=2))
        return

    # Table output
    correlations = store.analyze_noisy_correlations(results)
    n = int(correlations.get("n_evaluations", len(results)))
    print(f"\nNoisy/ZNE Analysis ({n} evaluations)")
    print("=" * 60)
    print(f"  Mean R²:          {correlations.get('mean_r2', 0):.4f}")
    print(f"  R² > 0.8:         {correlations.get('pct_r2_gt_08', 0):.1f}%")
    print(f"  ZNE helps:        {correlations.get('pct_helps', 0):.1f}%")
    print(f"  Mean gain:        {correlations.get('mean_gain_pct', 0):+.1f}%")
    if "corr_r2_gain" in correlations:
        print(f"  Corr(R², gain):   {correlations['corr_r2_gain']:.4f}")
    if "corr_ces_ratio_r2" in correlations:
        print(f"  Corr(CES ratio, R²): {correlations['corr_ces_ratio_r2']:.4f}")

    # Group-by analysis
    keys = [args.group_by] if args.group_by else ["seed_layout", "n_layouts", "h_test"]
    for key in keys:
        if not any(key in r for r in results):
            continue
        grouped = store.analyze_noisy_by_group(results, key)
        if grouped:
            print(f"\n  By {key}:")
            print(store.format_noisy_table(grouped, key))


def _write_json(data, filepath: str) -> None:
    """Write data to JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    n = len(data) if isinstance(data, list) else "structured"
    print(f"Saved to {path} ({n} entries)")


def main() -> None:
    args = parse_args()

    from qmbp_simulation.framework import ResultStore

    results_dir = Path(args.results_dir) if args.results_dir else None
    store = ResultStore(results_root=results_dir)

    if args.noisy:
        _run_noisy_analysis(store, args)
        return

    # Determine experiment IDs
    available = store.list_experiments()

    if args.all:
        exp_ids = available
    elif args.category:
        exp_ids = store.resolve_category(args.category, available)
    elif args.experiments:
        exp_ids = [e.upper() for e in args.experiments]
    else:
        print("Specify --all, --exp, --category, or --noisy")
        sys.exit(1)

    exp_ids = [e for e in exp_ids if e in available]
    if not exp_ids:
        print("No results found.")
        if available:
            print(f"Available: {', '.join(available)}")
        return

    _run_experiment_comparison(store, exp_ids, args)


if __name__ == "__main__":
    main()
