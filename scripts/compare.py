#!/usr/bin/env python3
"""Cross-experiment result comparison CLI.

Usage:
    python scripts/compare.py --exp B1 B4 F3
    python scripts/compare.py --category optimization
    python scripts/compare.py --category B
    python scripts/compare.py --all
    python scripts/compare.py --all --format json
    python scripts/compare.py --all --json output.json
"""

from __future__ import annotations

import argparse
import json
import sys

# Mapping from directory category names to experiment ID prefixes
_CATEGORY_PREFIX_MAP: dict[str, list[str]] = {
    "optimization": ["B", "C3"],
    "scaling": ["A"],
    "landscape": ["F"],
    "predictor": ["C1", "D", "E3"],
    "hardware": [],
    "generalization": ["E4"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare experiment results against V6.1 baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --exp B1 B4 F3              # Compare specific experiments
    %(prog)s --category B                # Compare all starting with B
    %(prog)s --category optimization     # Compare all optimization experiments
    %(prog)s --all                       # Compare all available results
    %(prog)s --all --format json         # Output JSON to stdout
    %(prog)s --all --json output.json    # Save structured JSON to file
        """,
    )
    parser.add_argument(
        "--exp",
        "--experiments",
        nargs="+",
        dest="experiments",
        help="Experiment IDs to compare (e.g., B1 B4 F3)",
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Compare all in a category (A-F letter prefix, or name: optimization, scaling, landscape, predictor, hardware, generalization)",
    )
    parser.add_argument("--all", action="store_true", help="Compare all available results")
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--json",
        type=str,
        metavar="FILE",
        dest="json_file",
        help="Save structured JSON output to file",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results directory (default: results/experiments)",
    )
    return parser.parse_args()


def _resolve_category(category: str, available: list[str]) -> list[str]:
    """Resolve a category argument to a list of experiment IDs.

    Supports both single-letter prefixes (e.g., 'B') and full category
    names (e.g., 'optimization').
    """
    cat_lower = category.lower()

    # Check if it's a known category name
    if cat_lower in _CATEGORY_PREFIX_MAP:
        prefixes = _CATEGORY_PREFIX_MAP[cat_lower]
        if not prefixes:
            return []
        return [eid for eid in available if any(eid.startswith(p) for p in prefixes)]

    # Otherwise treat as a letter prefix (e.g., 'B', 'A')
    prefix = category.upper()
    return [eid for eid in available if eid.startswith(prefix)]


def main() -> None:
    args = parse_args()

    from pathlib import Path

    from qmbp_simulation.framework import ResultStore

    # Initialize result store
    results_dir = Path(args.results_dir) if args.results_dir else None
    store = ResultStore(results_root=results_dir)

    # Determine which experiments to compare
    available = store.list_experiments()

    if args.all:
        exp_ids = available
    elif args.category:
        exp_ids = _resolve_category(args.category, available)
    elif args.experiments:
        exp_ids = [e.upper() for e in args.experiments]
    else:
        print("Specify --exp, --category, or --all")
        sys.exit(1)

    # Filter to those with results (gracefully skip missing)
    exp_ids = [e for e in exp_ids if e in available]

    if not exp_ids:
        print("No results found for specified experiments.")
        if available:
            print(f"Available: {', '.join(available)}")
        else:
            print("No experiment results found in results directory.")
        return

    # Generate comparison (O(n) — one file load per experiment)
    comparisons = store.compare_experiments(exp_ids)

    if not comparisons:
        print("No comparable results found (experiments may lack summary metrics).")
        return

    # Handle --json file output
    if args.json_file:
        output_path = Path(args.json_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(comparisons, f, indent=2)
        print(f"Comparison saved to {output_path} ({len(comparisons)} experiments)")
        return

    # Handle --format output
    if args.format == "table":
        print("\nExperiment Comparison vs V6.1 Baseline")
        print("=" * 70)
        print(store.generate_comparison_table(comparisons))
        print()

        # Summary
        improvements = [c for c in comparisons if c["verdict"] == "improvement"]
        regressions = [c for c in comparisons if c["verdict"] == "regression"]
        neutral = len(comparisons) - len(improvements) - len(regressions)
        print(
            f"Summary: {len(improvements)} improvements, "
            f"{len(regressions)} regressions, {neutral} neutral"
        )
    else:
        print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
