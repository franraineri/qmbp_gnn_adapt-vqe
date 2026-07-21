#!/usr/bin/env python3
"""Query the result index for fast experiment discovery.

Usage:
    # List all passing runs for a model/topology
    python project_health/cli/query_index.py --model tfim --topology heavy_hex --passed

    # Find best run for a config
    python project_health/cli/query_index.py --best --model tfim --n-qubits 20

    # Show coverage matrix
    python project_health/cli/query_index.py --coverage

    # Show stats summary
    python project_health/cli/query_index.py --stats

    # Detect regressions
    python project_health/cli/query_index.py --regressions

    # Estimate time for a new run
    python project_health/cli/query_index.py --estimate-time --model tfim --topology heavy_hex --n-qubits 20 --p-layers 4

    # Suggest next experiments
    python project_health/cli/query_index.py --suggest

    # Rebuild index from scratch
    python project_health/cli/query_index.py --rebuild
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qmbp_simulation.framework.result_index import ResultIndex


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Query the experiment result index.")
    # Filters
    p.add_argument("--model", type=str, help="Filter by model name")
    p.add_argument("--topology", type=str, help="Filter by topology")
    p.add_argument("--n-qubits", type=int, help="Filter by system size")
    p.add_argument("--p-layers", type=int, help="Filter by circuit depth")
    p.add_argument("--passed", action="store_true", help="Only passing runs")
    p.add_argument("--failed", action="store_true", help="Only failing runs")
    p.add_argument(
        "--backend", type=str, help="Filter by backend_type (e.g. 'mps', 'noiseless', 'hardware')"
    )
    # Actions
    p.add_argument("--best", action="store_true", help="Show only the best run per filter")
    p.add_argument("--coverage", action="store_true", help="Show coverage matrix")
    p.add_argument("--stats", action="store_true", help="Show aggregate statistics")
    p.add_argument("--regressions", action="store_true", help="Detect regressions")
    p.add_argument("--suggest", action="store_true", help="Suggest next experiments")
    p.add_argument(
        "--temporal-drift",
        action="store_true",
        help="Analyze temporal correlation of regressions (drift detection)",
    )
    p.add_argument(
        "--window-days",
        type=int,
        default=7,
        help="Window size in days for temporal analysis (default: 7)",
    )
    p.add_argument("--estimate-time", action="store_true", help="Estimate runtime")
    p.add_argument("--validate", action="store_true", help="Validate index integrity")
    p.add_argument("--rebuild", action="store_true", help="Force rebuild the index")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        index = ResultIndex()
    except Exception as e:
        print(f"ERROR: Could not initialize ResultIndex: {e}", file=sys.stderr)
        return 1

    if args.rebuild:
        n = index.rebuild()
        print(f"Index rebuilt: {n} entries")
        return 0

    # Ensure index is populated
    if len(index) == 0:
        index.rebuild()

    if args.stats:
        stats = index.stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("=== Index Statistics ===")
            print(f"  Total runs:    {stats['total_runs']}")
            print(f"  Passed:        {stats['n_passed']}")
            print(f"  Failed:        {stats['n_failed']}")
            print(f"  Pass rate:     {stats['pass_rate']:.0%}")
            print(f"  Compute time:  {stats['total_compute_hours']:.1f}h")
            print(f"  Models:        {', '.join(stats['models'])}")
            print(f"  Topologies:    {', '.join(stats['topologies'])}")
            print(f"  N values:      {stats['n_values']}")
            if stats.get("date_range"):
                print(
                    f"  Date range:    {stats['date_range'][0][:10]} → {stats['date_range'][1][:10]}"
                )
        return 0

    if args.coverage:
        if args.json:
            print(json.dumps(index.coverage_matrix(), indent=2))
        else:
            print("=== Coverage Matrix ===")
            index.print_coverage_matrix()
        return 0

    if args.regressions:
        regs = index.detect_regressions()
        if args.json:
            print(json.dumps(regs, indent=2))
        else:
            if not regs:
                print("No regressions detected.")
            else:
                print(f"=== {len(regs)} Regression(s) Detected ===\n")
                for r in regs:
                    print(f"  {r['config']}")
                    print(
                        f"     Latest: {r['latest_pass_rate']:.0%}  "
                        f"Median prev: {r.get('median_previous_pass_rate', r['best_previous_pass_rate']):.0%}  "
                        f"(delta={r['delta']:+.0%})"
                    )
                    print(f"     File: {r['latest_file']}")

                # Temporal drift summary
                drift = index.analyze_temporal_drift()
                if drift["has_drift"]:
                    print("\n  --- Temporal Drift Detected ---")
                    print(f"  Trend slope: {drift['trend_slope']:+.4f}/window")
                    if drift["breakpoint"]:
                        print(f"  Breakpoint:  {drift['breakpoint']}")
                    if drift["regression_cluster"]:
                        rc = drift["regression_cluster"]
                        print(
                            f"  Cluster:     {rc['n_regressions']} regressions on {rc['peak_date']}"
                        )
                    print("  Action: check git log around breakpoint date for introduced bugs")
                else:
                    print(f"\n  No temporal drift (slope={drift['trend_slope']:+.4f}).")

                # Stats at the end
                print(
                    f"\n  Summary: {len(regs)} regressions across "
                    f"{len(set(r['config'].split('|')[0] for r in regs))} models"
                )
        return 0

    if args.suggest:
        suggestions = index.suggest_next()
        if not suggestions:
            print("All viable configs have >80% pass rate. No suggestions.")
        else:
            print("=== Suggested Next Experiments ===")
            for s in suggestions:
                print(f"  → {s}")
        return 0

    if args.temporal_drift:
        drift = index.analyze_temporal_drift(
            model=args.model,
            topology=args.topology,
            window_days=args.window_days,
        )
        if args.json:
            print(json.dumps(drift, indent=2))
        else:
            print("=== Temporal Drift Analysis ===\n")
            print(f"  Entries analyzed: {drift['n_entries']}")
            if drift.get("reason"):
                print(f"  Status: {drift['reason']}")
            else:
                print(f"  Date range:      {drift['date_range'][0]} to {drift['date_range'][1]}")
                print(f"  Windows:         {len(drift['windows'])}")
                print(f"  Trend slope:     {drift['trend_slope']:+.4f} /window")
                print(f"  Correlation:     {drift['correlation']:+.3f}")
                print(f"  Max single drop: {drift.get('max_single_drop', 0):.0%}")
                if drift["breakpoint"]:
                    print(f"  Breakpoint:      {drift['breakpoint']}")
                if drift["regression_cluster"]:
                    rc = drift["regression_cluster"]
                    print(
                        f"  Cluster:         {rc['n_regressions']} regressions on {rc['peak_date']}"
                    )
                print()
                if drift["has_drift"]:
                    print("  RESULT: Temporal drift detected.")
                    print("  Action: check git log around breakpoint for introduced bugs.")
                else:
                    print("  RESULT: No significant temporal drift.")

                # Per-window table
                if drift["windows"]:
                    print(f"\n  {'Window':<25} {'Runs':>5} {'Pass Rate':>10}")
                    print(f"  {'-' * 25} {'-' * 5} {'-' * 10}")
                    for w in drift["windows"]:
                        print(
                            f"  {w['date_start']} - {w['date_end']}  "
                            f"{w['n_runs']:>4}  {w['pass_rate']:>8.0%}"
                        )
        return 0

    if args.estimate_time:
        if not args.model or not args.topology or not args.n_qubits or not args.p_layers:
            print("ERROR: --estimate-time requires --model, --topology, --n-qubits, --p-layers")
            return 1
        t = index.estimate_time(args.model, args.topology, args.n_qubits, args.p_layers)
        if t is None:
            print("No similar runs found — cannot estimate.")
        else:
            print(f"Estimated time: {t:.0f}s ({t / 3600:.1f}h)")
        return 0

    if args.validate:
        report = index.validate()
        print(f"Valid: {report['n_valid']} | Missing: {report['n_missing']}")
        if report["missing_files"]:
            for f in report["missing_files"]:
                print(f"  MISSING: {f}")
        return 0

    # Default: query with filters
    passed_filter = True if args.passed else (False if args.failed else None)
    results = index.query(
        model=args.model,
        topology=args.topology,
        n_qubits=args.n_qubits,
        p_layers=args.p_layers,
        passed=passed_filter,
    )

    # Post-filter by backend_type (substring match on simulation_diagnostics field)
    if args.backend:
        backend_filter = args.backend.lower()
        results = [
            r
            for r in results
            if r.get("backend_type") and backend_filter in r["backend_type"].lower()
        ]

    if args.best and results:
        results = [max(results, key=lambda r: (r.get("pass_rate", 0), r.get("timestamp", "")))]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"=== {len(results)} results ===")
        n_pass = sum(1 for r in results if r.get("passed"))
        for r in results[:30]:
            status = "PASS" if r.get("passed") else "FAIL"
            print(
                f"  [{status}] {r.get('model', '?'):<20} {r.get('topology', '?'):<12} "
                f"N={r.get('n_qubits', '?'):<3} p={r.get('p_layers', '?')} "
                f"rate={r.get('pass_rate', 0):.0%}  {r.get('_file', '')}"
            )
        if len(results) > 30:
            print(f"  ... and {len(results) - 30} more (use --json for full output)")
        # Stats at the end
        print(f"\n  Stats: {n_pass}/{len(results)} passed ({n_pass / max(len(results), 1):.0%})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
