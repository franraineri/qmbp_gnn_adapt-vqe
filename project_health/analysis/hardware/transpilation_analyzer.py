"""Analyze transpilation metrics from benchmark results.

Thin CLI wrapper over MitigationBenchmarkAnalyzer.compute_transpilation_summary().
Shows per-config circuit stats (depth, depth_2q, n_2q, fidelity) and
groups by optimization_level to reveal the transpilation quality impact.

Usage:
    python scripts/analyze_transpilation.py
    python scripts/analyze_transpilation.py --json results/transpilation_analysis.json

Integration:
    This analysis is also available programmatically via:
        from project_health.analysis.mitigation_benchmark_analyzer import MitigationBenchmarkAnalyzer
        a = MitigationBenchmarkAnalyzer()
        a.scan()
        summary = a.compute_transpilation_summary()
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results/mitigation_benchmark/fake_backend")


def scan_transpilation_stats() -> dict[str, dict]:
    """Scan benchmark results and extract circuit_stats per config.

    Returns dict[config_id → circuit_stats + delta_e_gap].
    Deduplicates by taking one representative result per config.
    """
    stats_by_config: dict[str, dict] = {}

    if not RESULTS_DIR.exists():
        return stats_by_config

    for config_dir in sorted(RESULTS_DIR.iterdir()):
        if not config_dir.is_dir():
            continue
        files = list(config_dir.glob("*.json"))
        if not files:
            continue
        # Take first file as representative (all same config = same transpilation)
        f = files[0]
        try:
            data = json.loads(f.read_text())
            cid = data["benchmark_metadata"]["config_id"]
            cs = data.get("circuit_stats", {})
            de = data.get("results", {}).get("delta_e_gap")
            r2 = data.get("results", {}).get("zne_r2")
            if cs:
                cs["delta_e_gap"] = de
                cs["zne_r2"] = r2
                stats_by_config[cid] = cs
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    return stats_by_config


def format_transpilation_table(stats: dict[str, dict]) -> str:
    """Format transpilation stats as a human-readable table."""
    lines = []
    lines.append(f"Configs with circuit_stats: {len(stats)}")
    lines.append("")
    header = f"{'Config':<27}|opt|depth|d_2q|n_2q|n_1q|route%| fidelity |dE/gap"
    lines.append(header)
    lines.append("-" * len(header))

    for cid in sorted(stats.keys()):
        s = stats[cid]
        opt = s.get("optimization_level", "?")
        depth = s.get("depth", "?")
        d2q = s.get("depth_2q", "?")
        n2q = s.get("n_2q_gates", "?")
        n1q = s.get("n_1q_gates", "?")
        routing = s.get("routing_overhead_pct", 0)
        fid = s.get("fidelity_estimate", 0)
        de = s.get("delta_e_gap")
        r_s = f"{routing:.0f}%" if isinstance(routing, (int, float)) else "?"
        f_s = f"{fid:.4f}" if isinstance(fid, (int, float)) else "?"
        d_s = f"{de:.3f}" if isinstance(de, (int, float)) and de else "N/A"
        lines.append(
            f"{cid:<27}| {opt!s:>1} |{depth!s:>5}|{d2q!s:>4}|"
            f"{n2q!s:>4}|{n1q!s:>4}|{r_s:>6}| {f_s:>8} |{d_s:>6}"
        )

    return "\n".join(lines)


def format_opt_level_summary(stats: dict[str, dict]) -> str:
    """Group stats by opt_level and compute aggregates."""
    by_opt: dict[int, list[tuple[str, dict]]] = defaultdict(list)
    for cid, s in stats.items():
        opt = s.get("optimization_level", 2)
        by_opt[opt].append((cid, s))

    lines = []
    lines.append("=" * 70)
    lines.append("TRANSPILATION SUMMARY BY OPTIMIZATION LEVEL")
    lines.append("=" * 70)

    for opt in sorted(by_opt.keys()):
        entries = by_opt[opt]
        depths = [s.get("depth", 0) for _, s in entries if s.get("depth")]
        d2qs = [s.get("depth_2q", 0) for _, s in entries if s.get("depth_2q")]
        n2qs = [s.get("n_2q_gates", 0) for _, s in entries if s.get("n_2q_gates")]
        fids = [s.get("fidelity_estimate", 0) for _, s in entries if s.get("fidelity_estimate")]
        des = [
            s.get("delta_e_gap")
            for _, s in entries
            if s.get("delta_e_gap") and s.get("delta_e_gap") > 0.0001
        ]

        lines.append(f"\n  opt_level={opt} ({len(entries)} configs):")
        if depths:
            lines.append(
                f"    depth:    mean={np.mean(depths):.1f}, range=[{min(depths)}, {max(depths)}]"
            )
        if d2qs:
            lines.append(
                f"    depth_2q: mean={np.mean(d2qs):.1f}, range=[{min(d2qs)}, {max(d2qs)}]"
            )
        if n2qs:
            lines.append(
                f"    n_2q:     mean={np.mean(n2qs):.1f}, range=[{min(n2qs)}, {max(n2qs)}]"
            )
        if fids:
            lines.append(
                f"    fidelity: mean={np.mean(fids):.4f}, range=[{min(fids):.4f}, {max(fids):.4f}]"
            )
        if des:
            lines.append(
                f"    dE/gap:   mean={np.mean(des):.4f}, range=[{min(des):.4f}, {max(des):.4f}]"
            )

    # Key conclusion
    if 0 in by_opt and 2 in by_opt:
        d2q_opt0 = np.mean([s.get("depth_2q", 0) for _, s in by_opt[0] if s.get("depth_2q")])
        d2q_opt2 = np.mean([s.get("depth_2q", 0) for _, s in by_opt[2] if s.get("depth_2q")])
        lines.append(
            f"\n  CONCLUSION: opt_level=0 produces {d2q_opt0 / d2q_opt2:.1f}× deeper "
            f"circuits (depth_2q) than opt_level=2"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze transpilation metrics from mitigation benchmark results"
    )
    parser.add_argument("--json", type=str, help="Export analysis to JSON file")
    parser.add_argument(
        "--use-analyzer",
        action="store_true",
        help="Use MitigationBenchmarkAnalyzer (programmatic path)",
    )
    args = parser.parse_args()

    if args.use_analyzer:
        # Programmatic path via the analyzer
        try:
            from project_health.analysis.mitigation_benchmark_analyzer import (
                MitigationBenchmarkAnalyzer,
            )

            a = MitigationBenchmarkAnalyzer()
            a.scan()
            summary = a.compute_transpilation_summary()
            print(f"Configs: {len(summary['per_config'])}")
            print(f"By opt_level: {json.dumps(summary['by_opt_level'], indent=2)}")
            print(f"Conclusion: {summary['conclusion']}")

            if args.json:
                Path(args.json).parent.mkdir(parents=True, exist_ok=True)
                Path(args.json).write_text(json.dumps(summary, indent=2, default=str))
                print(f"\nExported to {args.json}")
            return
        except ImportError as e:
            print(f"Analyzer not available ({e}), falling back to direct scan.")

    # Direct scan path (standalone, no analyzer dependency)
    stats = scan_transpilation_stats()

    if not stats:
        print("No benchmark results found. Run the mitigation benchmark first:")
        print(
            "  python scripts/experiment_runners/hardware/run_mitigation_benchmark.py "
            "--mode fake_backend --priority P0"
        )
        sys.exit(1)

    print(format_transpilation_table(stats))
    print()
    print(format_opt_level_summary(stats))

    if args.json:
        output = {"per_config": stats, "n_configs": len(stats)}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(output, indent=2, default=str))
        print(f"\nExported to {args.json}")


if __name__ == "__main__":
    main()
