#!/usr/bin/env python3
"""Full Pipeline Validation — Unified quality check and reporting.

Runs all validation checks and generates a comprehensive report:
1. Dashboard regeneration
2. Quality tier analysis  
3. Training readiness assessment
4. Scaling report generation
5. Cross-N coverage update
6. Discrepancy detection

Usage:
    .venv/bin/python scripts/maintenance/run_full_validation.py
    .venv/bin/python scripts/maintenance/run_full_validation.py --quick
    .venv/bin/python scripts/maintenance/run_full_validation.py --fix-orphans
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def step_1_regenerate_dashboard() -> dict:
    """Regenerate model quality dashboard from NPZ data."""
    print("\n" + "=" * 60)
    print("STEP 1: Regenerating Model Quality Dashboard")
    print("=" * 60)
    
    from qmbp_simulation.analysis.metrics import generate_model_quality_dashboard
    
    dashboard = generate_model_quality_dashboard()
    n_configs = dashboard.get("n_configs", 0)
    n_topos = len(dashboard.get("topology_summary", {}))
    
    print(f"  ✅ Dashboard regenerated: {n_configs} configs, {n_topos} topologies")
    return dashboard


def step_2_quality_tier_analysis(dashboard: dict) -> dict:
    """Analyze quality tier distribution across NPZ files."""
    print("\n" + "=" * 60)
    print("STEP 2: Quality Tier Analysis")
    print("=" * 60)
    
    import numpy as np
    
    npz_dir = DATA / "multi_n_training"
    if not npz_dir.exists():
        print("  ⚠️ No NPZ directory found")
        return {}
    
    tier_breakdown = {}
    total_verified = 0
    total_approx = 0
    total_unverified = 0
    n_legacy = 0
    
    for npz_file in sorted(npz_dir.glob("*.npz")):
        try:
            data = np.load(str(npz_file), allow_pickle=True)
            tiers = data.get("quality_tier")
            n_total = len(data["h_values"])
            
            if tiers is None:
                n_legacy += 1
                tier_breakdown[npz_file.name] = {
                    "verified": 0, "approximate": 0, "unverified": n_total,
                    "total": n_total, "legacy": True,
                }
                total_unverified += n_total
            else:
                tier_list = list(tiers)
                v = tier_list.count("verified")
                a = tier_list.count("approximate")
                u = tier_list.count("unverified")
                tier_breakdown[npz_file.name] = {
                    "verified": v, "approximate": a, "unverified": u,
                    "total": n_total, "legacy": False,
                }
                total_verified += v
                total_approx += a
                total_unverified += u
        except Exception as e:
            print(f"  ⚠️ Error reading {npz_file.name}: {e}")
    
    total = total_verified + total_approx + total_unverified
    print(f"  Total points: {total}")
    print(f"  ✅ Verified: {total_verified} ({total_verified*100//max(total,1)}%)")
    print(f"  ⚠️ Approximate: {total_approx} ({total_approx*100//max(total,1)}%)")
    print(f"  ❓ Unverified: {total_unverified} ({total_unverified*100//max(total,1)}%)")
    if n_legacy > 0:
        print(f"  📜 Legacy NPZ (no tier field): {n_legacy} files")
    
    return tier_breakdown


def step_3_training_readiness(dashboard: dict, tier_breakdown: dict) -> dict:
    """Check if training data is ready for MPNN training."""
    print("\n" + "=" * 60)
    print("STEP 3: Training Readiness Assessment")
    print("=" * 60)
    
    from qmbp_simulation.analysis.metrics import (
        compute_training_readiness,
        get_usable_training_configs,
    )
    
    utility_partition = get_usable_training_configs(dashboard)
    ready, reason, stats = compute_training_readiness(tier_breakdown, utility_partition)
    
    status = "✅ READY" if ready else "❌ NOT READY"
    print(f"  Status: {status}")
    print(f"  Reason: {reason}")
    
    if "n_useful_configs" in stats:
        print(f"  Useful configs: {stats['n_useful_configs']}")
    if "verified_ratio" in stats:
        print(f"  Verified ratio: {stats['verified_ratio']:.0%}")
    
    return {"ready": ready, "reason": reason, **stats}


def step_4_scaling_report(dashboard: dict, tier_breakdown: dict) -> dict:
    """Generate unified scaling report."""
    print("\n" + "=" * 60)
    print("STEP 4: Scaling Report Generation")
    print("=" * 60)
    
    from qmbp_simulation.analysis.metrics import generate_unified_scaling_report
    
    report = generate_unified_scaling_report(
        dashboard,
        tier_breakdown=tier_breakdown,
        target_n_values=[30, 40, 60],
    )
    
    # Print summary
    print("\n  Topology Scalability Scores:")
    for topo, info in sorted(report.get("topologies", {}).items()):
        score = info.get("scalability_score", 0)
        n_max = info.get("n_max_viable", "—")
        emoji = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
        print(f"    {emoji} {topo}: score={score:.2f}, n_max={n_max}")
    
    # Save report
    output_path = DATA / "unified_scaling_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved to: {output_path.relative_to(ROOT)}")
    
    return report


def step_5_update_coverage_doc() -> None:
    """Update the cross-N coverage documentation."""
    print("\n" + "=" * 60)
    print("STEP 5: Updating Cross-N Coverage Document")
    print("=" * 60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/maintenance/update_cross_n_coverage.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    
    # Extract key info from output
    for line in result.stdout.split("\n"):
        if "updated:" in line or "created:" in line or "✅" in line:
            print(f"  {line.strip()}")
    
    if result.returncode != 0:
        print(f"  ⚠️ Update returned code {result.returncode}")


def step_6_detect_discrepancies(dashboard: dict) -> list:
    """Detect discrepancies between dashboard and analyzer data."""
    print("\n" + "=" * 60)
    print("STEP 6: Discrepancy Detection")
    print("=" * 60)
    
    from qmbp_simulation.analysis.metrics import (
        detect_h_frontier_anomalies,
        detect_pass_rate_regression,
        detect_training_zoo_incoherence,
    )
    
    configs = dashboard.get("configs", [])
    issues = []
    
    # h_frontier anomalies
    anomalies = detect_h_frontier_anomalies(configs)
    if anomalies:
        print(f"  ⚠️ h_frontier anomalies: {len(anomalies)}")
        for a in anomalies[:3]:
            print(f"    - {a['topology']}: N={a['n_i']}→N={a['n_j']} (drop={a['drop']:.2f})")
        issues.extend(anomalies)
    
    # Training/zoo incoherence
    incoherent = detect_training_zoo_incoherence(configs)
    if incoherent:
        print(f"  ⚠️ Training/zoo incoherence: {len(incoherent)}")
        for i in incoherent[:3]:
            print(f"    - {i['topology']} N={i['n_qubits']}: "
                  f"bad_ratio={i['bad_ratio']:.0%}, zoo_pass={i['zoo_pass_rate']:.0%}")
        issues.extend(incoherent)
    
    # Pass rate regressions
    regressions = detect_pass_rate_regression(configs)
    if regressions:
        print(f"  ⚠️ Pass rate regressions: {len(regressions)}")
        for r in regressions[:3]:
            print(f"    - {r['topology']}: {r['prev_max']:.0%} → {r['curr_max']:.0%}")
        issues.extend(regressions)
    
    if not issues:
        print("  ✅ No significant discrepancies detected")
    
    return issues


def step_7_cleanup_orphans(dry_run: bool = True) -> int:
    """Identify and optionally clean up orphan checkpoints."""
    print("\n" + "=" * 60)
    print("STEP 7: Orphan Checkpoint Analysis")
    print("=" * 60)
    
    manifest_path = DATA / "model_zoo" / "manifest.json"
    ckpt_dir = DATA / "model_zoo" / "checkpoints"
    
    if not manifest_path.exists() or not ckpt_dir.exists():
        print("  No model zoo found")
        return 0
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    entries = manifest if isinstance(manifest, list) else manifest.get("entries", [])
    registered = {e.get("checkpoint_file") for e in entries}
    
    orphans = [f for f in ckpt_dir.glob("*.pt") if f.name not in registered]
    
    if not orphans:
        print("  ✅ No orphan checkpoints")
        return 0
    
    print(f"  Found {len(orphans)} orphan checkpoints:")
    for o in orphans[:5]:
        size_kb = o.stat().st_size / 1024
        print(f"    - {o.name} ({size_kb:.1f} KB)")
    if len(orphans) > 5:
        print(f"    ... and {len(orphans) - 5} more")
    
    if not dry_run:
        print("\n  Cleaning up orphans...")
        for o in orphans:
            o.unlink()
        print(f"  ✅ Removed {len(orphans)} orphan files")
    else:
        print("\n  Run with --fix-orphans to remove them")
    
    return len(orphans)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full pipeline validation and reporting"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip slow steps (orphan cleanup, detailed analysis)"
    )
    parser.add_argument(
        "--fix-orphans", action="store_true",
        help="Actually delete orphan checkpoint files"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("  FULL PIPELINE VALIDATION")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)
    
    # Run all steps
    dashboard = step_1_regenerate_dashboard()
    tier_breakdown = step_2_quality_tier_analysis(dashboard)
    readiness = step_3_training_readiness(dashboard, tier_breakdown)
    scaling_report = step_4_scaling_report(dashboard, tier_breakdown)
    
    if not args.quick:
        step_5_update_coverage_doc()
        step_6_detect_discrepancies(dashboard)
        step_7_cleanup_orphans(dry_run=not args.fix_orphans)
    
    # Final summary
    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)
    
    ready = readiness.get("ready", False)
    n_recs = len(scaling_report.get("recommendations", []))
    
    if ready and n_recs == 0:
        print("  🟢 Pipeline is healthy and ready for training")
    elif ready:
        print(f"  🟡 Pipeline is ready but has {n_recs} recommendations")
    else:
        print(f"  🔴 Pipeline NOT ready: {readiness.get('reason', 'unknown')}")
    
    # Print recommendations
    recs = scaling_report.get("recommendations", [])
    if recs:
        print("\n  Recommendations:")
        for rec in recs[:5]:
            print(f"    • {rec}")
    
    print("\n" + "=" * 60)
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
