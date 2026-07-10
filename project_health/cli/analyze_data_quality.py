#!/usr/bin/env python3
"""Data Quality Analyzer — Index health check and garbage detection.

Scans the ResultIndex for entries with missing/invalid metadata and
classifies them as valid, borderline, or garbage. Useful for periodic
health checks to ensure the index reflects real experiment data.

Usage:
    python project_health/cli/analyze_data_quality.py
    python project_health/cli/analyze_data_quality.py --verbose
    python project_health/cli/analyze_data_quality.py --json
    python project_health/cli/analyze_data_quality.py --purge-garbage  # removes garbage from index

Also importable for programmatic use:
    from project_health.cli.analyze_data_quality import analyze_index_quality
    report = analyze_index_quality()
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Ensure project root is in path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qmbp_simulation.framework.result_index import ResultIndex

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Quality Classification
# ═══════════════════════════════════════════════════════════════════════════════

# Known garbage experiment IDs from early development
GARBAGE_EXPERIMENT_IDS = {"TEST", "FAIL", "NONE", "XFAIL", "CNT", ""}


def classify_entry(entry: dict[str, Any]) -> tuple[str, list[str]]:
    """Classify an index entry as 'valid', 'borderline', or 'garbage'.

    Parameters
    ----------
    entry : dict
        Index entry metadata.

    Returns
    -------
    tuple[str, list[str]]
        (classification, list_of_issues)
    """
    issues: list[str] = []

    model = entry.get("model", "")
    topology = entry.get("topology", "")
    n_qubits = entry.get("n_qubits", 0)
    p_layers = entry.get("p_layers", 0)
    n_sections = entry.get("n_sections", 0)
    experiment_id = entry.get("experiment_id", "")
    timestamp = entry.get("timestamp", "")
    interrupted = entry.get("interrupted", False)
    pass_rate = entry.get("pass_rate") or 0

    # Hard garbage criteria
    if not model:
        issues.append("no_model")
    if not topology or topology == "[]":
        issues.append("empty_topology")
    if not n_qubits:
        issues.append("no_n_qubits")
    if not p_layers:
        issues.append("no_p_layers")
    if n_sections == 0:
        issues.append("zero_sections")
    if experiment_id.upper() in GARBAGE_EXPERIMENT_IDS:
        issues.append("garbage_experiment_id")

    # Borderline criteria (single issue, not necessarily garbage)
    if not timestamp:
        issues.append("no_timestamp")
    if interrupted:
        issues.append("interrupted")
    if n_sections == 1 and pass_rate == 0:
        issues.append("single_section_failed")

    # Classification
    hard_issues = {
        "no_model",
        "empty_topology",
        "no_n_qubits",
        "no_p_layers",
        "zero_sections",
        "garbage_experiment_id",
    }
    n_hard = sum(1 for i in issues if i in hard_issues)

    if n_hard >= 2:
        return "garbage", issues
    elif n_hard == 1 or issues:
        return "borderline", issues
    else:
        return "valid", []


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_index_quality(index: ResultIndex | None = None) -> dict[str, Any]:
    """Analyze data quality of all index entries.

    Parameters
    ----------
    index : ResultIndex | None
        Index to analyze. Creates one if not provided.

    Returns
    -------
    dict
        Quality report with counts, breakdowns, and recommendations.
    """
    if index is None:
        index = ResultIndex()

    entries = index.entries
    valid_list: list[dict] = []
    borderline_list: list[dict] = []
    garbage_list: list[dict] = []
    issue_counts: Counter = Counter()

    for entry in entries:
        classification, issues = classify_entry(entry)
        for issue in issues:
            issue_counts[issue] += 1

        if classification == "valid":
            valid_list.append(entry)
        elif classification == "borderline":
            borderline_list.append(entry)
        else:
            garbage_list.append(entry)

    # Topology breakdown
    topo_stats: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {"n": 0, "sum_rate": 0.0, "n_passed": 0}
    )
    for e in valid_list:
        topo = e.get("topology", "?")
        topo_stats[topo]["n"] += 1
        topo_stats[topo]["sum_rate"] += e.get("pass_rate", 0)
        if e.get("passed"):
            topo_stats[topo]["n_passed"] += 1

    # Model breakdown
    model_stats: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {"n": 0, "sum_rate": 0.0, "n_passed": 0}
    )
    for e in valid_list:
        model = e.get("model", "?")
        model_stats[model]["n"] += 1
        model_stats[model]["sum_rate"] += e.get("pass_rate", 0)
        if e.get("passed"):
            model_stats[model]["n_passed"] += 1

    # Experiment ID breakdown for garbage
    garbage_exp_ids: Counter = Counter()
    for e in garbage_list:
        garbage_exp_ids[e.get("experiment_id", "unknown")] += 1

    # Section count distribution for valid entries
    section_dist: Counter = Counter()
    for e in valid_list:
        section_dist[e.get("n_sections", 0)] += 1

    # Pass rate stats for valid entries
    valid_rates = [e.get("pass_rate") or 0 for e in valid_list]
    avg_rate = sum(valid_rates) / len(valid_rates) if valid_rates else 0

    # Stale checkpoints scan
    stale_checkpoints = _scan_stale_checkpoints()

    return {
        "total_entries": len(entries),
        "valid": len(valid_list),
        "borderline": len(borderline_list),
        "garbage": len(garbage_list),
        "garbage_pct": len(garbage_list) / max(len(entries), 1),
        "issue_counts": dict(issue_counts.most_common()),
        "topology_stats": dict(topo_stats),
        "model_stats": dict(model_stats),
        "garbage_experiment_ids": dict(garbage_exp_ids.most_common(10)),
        "section_distribution": dict(sorted(section_dist.items())),
        "valid_avg_pass_rate": avg_rate,
        "valid_n_passed": sum(1 for e in valid_list if e.get("passed")),
        "stale_checkpoints": stale_checkpoints,
        "recommendations": _generate_recommendations(
            len(entries),
            len(garbage_list),
            len(borderline_list),
            issue_counts,
            stale_checkpoints,
        ),
    }


def _scan_stale_checkpoints() -> list[dict[str, Any]]:
    """Find orphaned checkpoint files in the results directory.

    Stale checkpoints are left behind by interrupted runs that were
    never resumed. They're harmless (small, hidden) but indicate
    incomplete work.

    Returns
    -------
    list[dict]
        Each dict has: path, size_bytes, age_hours, label.
    """
    import time as _time

    results_dir = ROOT / "results" / "experiments"
    if not results_dir.exists():
        return []

    stale: list[dict[str, Any]] = []
    now = _time.time()

    for cp_file in results_dir.rglob(".checkpoint_*.json"):
        try:
            stat = cp_file.stat()
        except OSError:
            continue
        age_hours = (now - stat.st_mtime) / 3600

        # Try to extract label from filename
        label = cp_file.stem.replace(".checkpoint_", "")

        stale.append(
            {
                "path": str(cp_file.relative_to(results_dir)),
                "size_bytes": stat.st_size,
                "age_hours": round(age_hours, 1),
                "label": label,
            }
        )

    return sorted(stale, key=lambda x: -x["age_hours"])


def _generate_recommendations(
    total: int,
    n_garbage: int,
    n_borderline: int,
    issues: Counter,
    stale_checkpoints: list[dict[str, Any]],
) -> list[str]:
    """Generate actionable recommendations based on data quality."""
    recs: list[str] = []

    garbage_pct = n_garbage / max(total, 1) * 100
    if garbage_pct > 50:
        recs.append(
            f"⚠️  {garbage_pct:.0f}% of index entries are garbage — "
            f"consider running --purge-garbage to clean the index"
        )

    if issues.get("empty_topology", 0) > total * 0.3:
        recs.append(
            "Many entries have empty topology — legacy runs from before "
            "topology-aware system. These are auto-excluded from analysis."
        )

    if issues.get("zero_sections", 0) > 20:
        recs.append(
            f"{issues['zero_sections']} entries have zero sections "
            f"(crashed before any work). Check for infrastructure issues."
        )

    if issues.get("no_timestamp", 0) > 10:
        recs.append(
            f"{issues['no_timestamp']} entries lack timestamps — "
            f"chronological ordering may be unreliable for those."
        )

    if stale_checkpoints:
        old_ones = [cp for cp in stale_checkpoints if cp["age_hours"] > 24]
        if old_ones:
            total_kb = sum(cp["size_bytes"] for cp in old_ones) / 1024
            recs.append(
                f"{len(old_ones)} stale checkpoint(s) older than 24h "
                f"({total_kb:.1f} KB). These are from interrupted runs "
                f"that were never resumed. Safe to delete."
            )

    if not recs:
        recs.append("✅ Index quality is acceptable.")

    return recs


# ═══════════════════════════════════════════════════════════════════════════════
# Output Formatting
# ═══════════════════════════════════════════════════════════════════════════════


def print_report(report: dict[str, Any], verbose: bool = False) -> None:
    """Print a human-readable quality report."""
    total = report["total_entries"]
    valid = report["valid"]
    borderline = report["borderline"]
    garbage = report["garbage"]

    print(f"\n{'═' * 60}")
    print("  DATA QUALITY REPORT")
    print(f"{'═' * 60}")
    print(f"  Total index entries: {total}")
    if total == 0:
        print("  (empty index — nothing to analyze)")
        print(f"{'═' * 60}\n")
        return
    print(f"  ✅ Valid:      {valid:4d} ({valid / total * 100:.0f}%)")
    print(f"  ⚠️  Borderline: {borderline:4d} ({borderline / total * 100:.0f}%)")
    print(f"  ❌ Garbage:    {garbage:4d} ({garbage / total * 100:.0f}%)")
    print()

    # Valid stats
    print(f"  Valid entries avg pass rate: {report['valid_avg_pass_rate']:.0%}")
    print(f"  Valid entries passed: {report['valid_n_passed']}/{valid}")
    print()

    # Issue breakdown
    if report["issue_counts"]:
        print("  Issue Breakdown:")
        for issue, count in report["issue_counts"].items():
            pct = count / total * 100
            print(f"    {issue:25s}: {count:4d} ({pct:.0f}%)")
        print()

    # Topology stats (valid only)
    if report["topology_stats"]:
        print("  Topology Stats (valid entries only):")
        for topo, stats in sorted(report["topology_stats"].items(), key=lambda x: -x[1]["n"]):
            avg = stats["sum_rate"] / max(stats["n"], 1)
            print(
                f"    {topo:15s}: {stats['n']:3d} runs, "
                f"avg rate={avg:.0%}, passed={stats['n_passed']}"
            )
        print()

    # Model stats (valid only)
    if report.get("model_stats"):
        print("  Model Stats (valid entries only):")
        for model, stats in sorted(report["model_stats"].items(), key=lambda x: -x[1]["n"]):
            avg = stats["sum_rate"] / max(stats["n"], 1)
            print(
                f"    {model:25s}: {stats['n']:3d} runs, "
                f"avg rate={avg:.0%}, passed={stats['n_passed']}"
            )
        print()

    if verbose:
        # Garbage experiment IDs
        if report["garbage_experiment_ids"]:
            print("  Top Garbage Experiment IDs:")
            for eid, count in report["garbage_experiment_ids"].items():
                print(f"    {eid or '(empty)':30s}: {count}")
            print()

        # Section distribution
        if report["section_distribution"]:
            print("  Section Count Distribution (valid):")
            for n_sec, count in report["section_distribution"].items():
                print(f"    n_sections={n_sec}: {count} runs")
            print()

    # Recommendations
    print("  Recommendations:")
    for rec in report["recommendations"]:
        print(f"    {rec}")

    # Stale checkpoints
    if report.get("stale_checkpoints"):
        print()
        print(f"  Stale Checkpoints ({len(report['stale_checkpoints'])}):")
        for cp in report["stale_checkpoints"]:
            age_str = (
                f"{cp['age_hours']:.0f}h"
                if cp["age_hours"] < 48
                else f"{cp['age_hours'] / 24:.0f}d"
            )
            print(f"    {cp['path']:60s} {cp['size_bytes'] / 1024:.1f}KB  age={age_str}")

    print(f"{'═' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze data quality of the ResultIndex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python project_health/cli/analyze_data_quality.py
    python project_health/cli/analyze_data_quality.py --verbose
    python project_health/cli/analyze_data_quality.py --json
    python project_health/cli/analyze_data_quality.py --purge-garbage
""",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed breakdowns (garbage IDs, section distribution)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON (for programmatic consumption)",
    )
    parser.add_argument(
        "--purge-garbage",
        action="store_true",
        help="Remove garbage entries from the index (rebuilds without them)",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt for --purge-garbage (useful in CI/scripts)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    index = ResultIndex()
    report = analyze_index_quality(index)

    if args.json:
        # Make serializable
        output = {k: v for k, v in report.items()}
        print(json.dumps(output, indent=2, default=str))
    else:
        print_report(report, verbose=args.verbose)

    if args.purge_garbage:
        _purge_garbage(index, force=args.yes)

    return 0


def _purge_garbage(index: ResultIndex, force: bool = False) -> None:
    """Remove garbage entries from the index and rebuild.

    Keeps valid + borderline entries. Only removes entries classified
    as 'garbage' (2+ hard issues).
    """
    before = len(index.entries)

    # Count what will be removed
    keep_entries = []
    n_garbage = 0
    for entry in index.entries:
        classification, _ = classify_entry(entry)
        if classification == "garbage":
            n_garbage += 1
        else:
            keep_entries.append(entry)

    if n_garbage == 0:
        print("No garbage to purge.")
        return

    # Confirm before destructive operation
    print(f"\nWill remove {n_garbage} garbage entries from index ({before} → {len(keep_entries)}).")
    print("This keeps valid + borderline entries. Original result files are NOT deleted.")

    if not force:
        if not sys.stdin.isatty():
            print("Non-interactive mode — use --yes to confirm purge.")
            return
        response = input("Proceed? [y/N] ").strip().lower()
        if response not in ("y", "yes"):
            print("Aborted.")
            return

    # Write back
    index._entries = keep_entries
    index._save()
    print(f"  Done. Index: {before} → {len(keep_entries)} entries ({n_garbage} removed)")
    print("  Run `python project_health/cli/query_index.py --rebuild` to re-scan if needed.")


if __name__ == "__main__":
    sys.exit(main())
