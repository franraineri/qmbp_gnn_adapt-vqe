#!/usr/bin/env python3
"""Generate unified scaling report — cross-integration of quality metrics.

Combines:
- Model quality dashboard (pass rates, h_frontier, training utility)
- Quality tier breakdown (verified/approximate/unverified)
- Extrapolation viability predictions
- Training readiness assessment

Usage:
    .venv/bin/python scripts/maintenance/generate_scaling_report.py
    .venv/bin/python scripts/maintenance/generate_scaling_report.py --target-n 30 40 60 100
    .venv/bin/python scripts/maintenance/generate_scaling_report.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
NPZ_DIR = DATA / "multi_n_training"
DASHBOARD_PATH = DATA / "model_quality_dashboard.json"
OUTPUT_PATH = DATA / "unified_scaling_report.json"


def load_dashboard() -> dict:
    """Load model quality dashboard."""
    if not DASHBOARD_PATH.exists():
        print(f"  ⚠️  Dashboard not found at {DASHBOARD_PATH}")
        print("  Run: .venv/bin/python scripts/maintenance/inspect_data_stores.py --refresh")
        sys.exit(1)
    with open(DASHBOARD_PATH) as f:
        return json.load(f)


def compute_quality_tier_breakdown(npz_dir: Path) -> dict[str, dict]:
    """Compute quality tier distribution per NPZ file."""
    import numpy as np
    
    if not npz_dir.exists():
        return {}
    
    breakdown = {}
    for npz_file in sorted(npz_dir.glob("*.npz")):
        try:
            data = np.load(str(npz_file), allow_pickle=True)
            tiers = data.get("quality_tier")
            if tiers is None:
                n_pts = len(data["h_values"])
                breakdown[npz_file.name] = {
                    "verified": 0,
                    "approximate": 0,
                    "unverified": n_pts,
                    "total": n_pts,
                    "legacy": True,
                }
            else:
                tier_list = list(tiers)
                breakdown[npz_file.name] = {
                    "verified": tier_list.count("verified"),
                    "approximate": tier_list.count("approximate"),
                    "unverified": tier_list.count("unverified"),
                    "total": len(tier_list),
                    "legacy": False,
                }
        except Exception as e:
            print(f"  ⚠️ Error reading {npz_file.name}: {e}", file=sys.stderr)
    return breakdown


def format_report_text(report: dict) -> str:
    """Format report as human-readable text."""
    lines = [
        "=" * 70,
        "UNIFIED SCALING REPORT",
        f"Generated: {report['generated_at'][:19]}",
        "=" * 70,
        "",
    ]
    
    # Training readiness
    tr = report.get("training_readiness", {})
    ready_str = "✅ READY" if tr.get("ready") else "❌ NOT READY"
    lines.append(f"Training Readiness: {ready_str}")
    lines.append(f"  Reason: {tr.get('reason', 'unknown')}")
    if "verified_ratio" in tr:
        lines.append(f"  Verified ratio: {tr['verified_ratio']:.0%}")
    if "n_useful_configs" in tr:
        lines.append(f"  Useful configs: {tr['n_useful_configs']}")
    lines.append("")
    
    # Per-topology summary
    lines.append("-" * 70)
    lines.append("TOPOLOGY SUMMARY")
    lines.append("-" * 70)
    lines.append(
        f"{'Topology':<15} {'n_max':<6} {'Pass%':<8} {'h_front':<8} {'Score':<8} {'Reason'}"
    )
    lines.append("-" * 70)
    
    for topo, info in sorted(report.get("topologies", {}).items()):
        n_max = info.get("n_max_viable", "—")
        if isinstance(n_max, int):
            n_max_str = str(n_max)
        else:
            n_max_str = "—"
        pass_rate = info.get("best_pass_rate", 0)
        h_frontier = info.get("h_frontier")
        h_str = f"{h_frontier:.2f}" if h_frontier else "—"
        score = info.get("scalability_score", 0)
        reason = info.get("scalability_reason", "")
        lines.append(
            f"{topo:<15} {n_max_str:<6} {pass_rate:.0%}     {h_str:<8} {score:.2f}     {reason}"
        )
    lines.append("")
    
    # Extrapolation viability
    lines.append("-" * 70)
    lines.append("EXTRAPOLATION VIABILITY")
    lines.append("-" * 70)
    
    extrap = report.get("extrapolation_viability", {})
    target_ns = set()
    for topo_data in extrap.values():
        target_ns.update(topo_data.keys())
    target_ns = sorted(target_ns)
    
    if target_ns:
        header = f"{'Topology':<15}"
        for n in target_ns:
            header += f" N={n:<5}"
        lines.append(header)
        lines.append("-" * 70)
        
        for topo in sorted(extrap.keys()):
            row = f"{topo:<15}"
            for n in target_ns:
                v = extrap[topo].get(n, {})
                if v.get("viable"):
                    row += f" {'✅':<6}"
                else:
                    row += f" {'❌':<6}"
            lines.append(row)
    lines.append("")
    
    # Recommendations
    recs = report.get("recommendations", [])
    if recs:
        lines.append("-" * 70)
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 70)
        for rec in recs:
            lines.append(f"  • {rec}")
        lines.append("")
    
    lines.append("=" * 70)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate unified scaling report from quality metrics"
    )
    parser.add_argument(
        "--target-n", type=int, nargs="+", default=[30, 40, 60],
        help="Target N values for extrapolation viability (default: 30 40 60)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output JSON format instead of text"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=f"Output file path (default: {OUTPUT_PATH})"
    )
    args = parser.parse_args()
    
    print("Loading dashboard...")
    dashboard = load_dashboard()
    n_configs = dashboard.get("n_configs", 0)
    print(f"  {n_configs} configs loaded")
    
    print("Computing quality tier breakdown...")
    tier_breakdown = compute_quality_tier_breakdown(NPZ_DIR)
    if tier_breakdown:
        n_verified = sum(t.get("verified", 0) for t in tier_breakdown.values())
        total_pts = sum(t.get("total", 0) for t in tier_breakdown.values())
        print(f"  {len(tier_breakdown)} NPZ files, {n_verified}/{total_pts} verified")
    else:
        print("  No NPZ files found")
    
    print("Generating unified scaling report...")
    from qmbp_simulation.analysis.metrics import generate_unified_scaling_report
    
    report = generate_unified_scaling_report(
        dashboard,
        tier_breakdown=tier_breakdown,
        target_n_values=args.target_n,
    )
    
    # Output
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report_text(report))
    
    # Save to file
    output_path = Path(args.output) if args.output else OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
