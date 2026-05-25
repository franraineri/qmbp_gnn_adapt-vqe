#!/usr/bin/env python3
"""Cross-experiment comparison tool.

Usage:
    python scripts/experiments_v8/compare_results.py --experiments B1 B4 F3
    python scripts/experiments_v8/compare_results.py --category B
    python scripts/experiments_v8/compare_results.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Compare V8 experiment results")
    parser.add_argument("--experiments", nargs="+", help="Experiment IDs to compare")
    parser.add_argument("--category", type=str, help="Compare all in a category (A-F)")
    parser.add_argument("--all", action="store_true", help="Compare all available results")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    args = parser.parse_args()

    from scripts.experiments_v8.core.result_store import ResultStore
    from scripts.experiments_v8.experiments import EXPERIMENT_REGISTRY

    store = ResultStore()

    # Determine which experiments to compare
    if args.all:
        exp_ids = store.list_experiments()
    elif args.category:
        exp_ids = [eid for eid in EXPERIMENT_REGISTRY if eid.startswith(args.category.upper())]
    elif args.experiments:
        exp_ids = [e.upper() for e in args.experiments]
    else:
        print("Specify --experiments, --category, or --all")
        sys.exit(1)

    # Filter to those with results
    available = store.list_experiments()
    exp_ids = [e for e in exp_ids if e in available]

    if not exp_ids:
        print("No results found for specified experiments.")
        print(f"Available: {available}")
        return

    # Generate comparison
    comparisons = store.compare_experiments(exp_ids)

    if args.format == "table":
        print("\nV8 Experiment Comparison vs V6.1 Baseline")
        print("=" * 70)
        print(store.generate_comparison_table(comparisons))
        print()

        # Summary
        improvements = [c for c in comparisons if c.verdict == "improvement"]
        regressions = [c for c in comparisons if c.verdict == "regression"]
        print(
            f"Summary: {len(improvements)} improvements, {len(regressions)} regressions, "
            f"{len(comparisons) - len(improvements) - len(regressions)} neutral"
        )
    else:
        import json

        data = [c.to_dict() for c in comparisons]
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
