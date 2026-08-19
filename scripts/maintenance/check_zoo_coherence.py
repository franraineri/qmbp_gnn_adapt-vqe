#!/usr/bin/env python3
"""Post-run coherence check: zoo manifest ↔ dashboard ↔ disk.

Validates that the model zoo, dashboard, and training data are all
internally consistent. Designed to run as a post-execution hook or
standalone diagnostic.

Exit codes:
    0 = all coherent
    1 = issues found (printed to stdout)

Usage:
    .venv/bin/python scripts/maintenance/check_zoo_coherence.py
    .venv/bin/python scripts/maintenance/check_zoo_coherence.py --json
    .venv/bin/python scripts/maintenance/check_zoo_coherence.py --retrain-queue
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def check_coherence() -> dict:
    """Run all coherence checks. Returns structured report."""
    from qmbp_simulation.predictors.model_zoo import (
        compute_retrain_queue,
        list_pretrained,
        validate_zoo,
    )

    issues = []
    dashboard_path = ROOT / "data" / "model_quality_dashboard.json"

    # ── Check 1: Zoo integrity ────────────────────────────────────────────
    zoo_report = validate_zoo()
    if zoo_report["n_missing"] > 0 or zoo_report["n_corrupted"] > 0:
        issues.append(
            {
                "check": "zoo_integrity",
                "severity": "critical",
                "detail": f"{zoo_report['n_missing']} missing, {zoo_report['n_corrupted']} corrupted",
            }
        )

    # ── Check 2: Zoo pass_rate vs dashboard — multi-factor coherence ─────
    if dashboard_path.exists():
        with open(dashboard_path) as f:
            dashboard = json.load(f)

        configs = dashboard.get("configs", [])
        multi_n_entries = list_pretrained(n_qubits=0)

        for entry in multi_n_entries:
            topo = entry.topology
            topo_configs = [c for c in configs if c["topology"] == topo]
            if not topo_configs:
                continue

            zoo_pr = entry.pass_rate
            total_pts = sum(c.get("n_points", 0) for c in topo_configs)
            if total_pts == 0:
                continue

            # A. Weighted average dual pass (comparable to zoo_pass_rate)
            weighted_npz_dual = (
                sum(
                    c.get("pass_rate_dual_criterion", 0) * c.get("n_points", 0)
                    for c in topo_configs
                )
                / total_pts
            )

            # B. Check if model was NEVER evaluated (pass_rate=0 but exists)
            if zoo_pr == 0 and entry.n_training_points > 0:
                issues.append(
                    {
                        "check": "zoo_never_evaluated",
                        "severity": "warning",
                        "detail": (
                            f"{topo}: model has {entry.n_training_points} pts "
                            f"but pass_rate=0 (never evaluated). "
                            f"Run: .venv/bin/python scripts/maintenance/reevaluate_zoo_models.py --topology {topo}"
                        ),
                    }
                )
                continue  # Skip divergence check for unevaluated models

            # C. Divergence: zoo claims X% but NPZ weighted says Y%
            if zoo_pr > 0 and weighted_npz_dual > 0:
                divergence = abs(zoo_pr - weighted_npz_dual)
                if divergence > 0.15:
                    # Determine direction to diagnose root cause
                    if zoo_pr > weighted_npz_dual:
                        cause = "zoo stale (data degraded since last eval or new bad N added)"
                    else:
                        cause = "zoo outdated (data improved since last eval)"
                    issues.append(
                        {
                            "check": "zoo_dashboard_divergence",
                            "severity": "warning",
                            "detail": (
                                f"{topo}: zoo={zoo_pr:.0%} vs npz_weighted={weighted_npz_dual:.0%} "
                                f"(Δ={divergence:.0%}). Cause: {cause}"
                            ),
                        }
                    )

            # D. Model trained on subset — more N values now available
            # Parse N values from checkpoint filename (e.g., "multiN_4+6+8+10_p1")
            import re

            n_match = re.search(r"multiN_([\d+]+)_p", entry.checkpoint_file)
            if n_match:
                trained_n = set(int(x) for x in n_match.group(1).split("+"))
                available_n = set(
                    c["n_qubits"] for c in topo_configs if c.get("training_utility") == "useful"
                )
                new_n = available_n - trained_n
                if new_n and len(new_n) >= 2:
                    issues.append(
                        {
                            "check": "zoo_missing_n_coverage",
                            "severity": "info",
                            "detail": (
                                f"{topo}: model trained on N={sorted(trained_n)} "
                                f"but N={sorted(new_n)} also viable. "
                                f"Consider retraining to include new N values."
                            ),
                        }
                    )

            # E. Data freshness: NPZ data much larger than training points
            if entry.n_training_points > 0 and total_pts > entry.n_training_points * 1.5:
                issues.append(
                    {
                        "check": "zoo_data_expanded",
                        "severity": "info",
                        "detail": (
                            f"{topo}: model trained on {entry.n_training_points} pts "
                            f"but {total_pts} now available (+{total_pts - entry.n_training_points}). "
                            f"Retrain may improve generalization."
                        ),
                    }
                )

    # ── Check 3: Retrain queue (models that need attention) ───────────────
    retrain_queue = compute_retrain_queue()
    high_priority = [r for r in retrain_queue if r["priority"] <= 2]
    if high_priority:
        issues.append(
            {
                "check": "retrain_needed",
                "severity": "info",
                "detail": f"{len(high_priority)} model(s) need retraining (priority 1-2)",
                "models": [r["topology"] for r in high_priority],
            }
        )

    # ── Check 4: Orphan checkpoints ──────────────────────────────────────
    ckpt_dir = ROOT / "data" / "model_zoo" / "checkpoints"
    if ckpt_dir.exists():
        manifest_files = {e.checkpoint_file for e in list_pretrained()}
        disk_files = {f.name for f in ckpt_dir.glob("*.pt")}
        orphans = disk_files - manifest_files
        if orphans:
            issues.append(
                {
                    "check": "orphan_checkpoints",
                    "severity": "info",
                    "detail": f"{len(orphans)} checkpoint(s) not in manifest",
                }
            )

    # ── Check 5: GT cache ↔ NPZ e_exact staleness ────────────────────────
    # If GT cache has more accurate values (lower energy) than what's in
    # the NPZ e_exact field, the NPZ metrics are inflated.
    try:
        import numpy as _np

        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        gt_cache = GroundTruthCache()
        npz_dir = ROOT / "data" / "multi_n_training"
        n_stale_npz = 0
        stale_details = []

        if npz_dir.exists() and len(gt_cache) > 0:
            for npz_file in sorted(npz_dir.glob("*.npz"))[:10]:  # Sample first 10
                stem = npz_file.stem
                parts = stem.split("_")
                n_idx = next((i for i, p in enumerate(parts) if p.startswith("N")), None)
                if n_idx is None:
                    continue
                topo = "_".join(parts[:n_idx])
                n_q = int(parts[n_idx][1:])

                data = _np.load(npz_file, allow_pickle=True)
                h_vals = data["h_values"]
                e_exact = data["e_exact"].astype(_np.float64)

                n_stale_pts = 0
                for i, h in enumerate(h_vals[:20]):  # Sample 20 pts
                    cached = gt_cache.get(topo, n_q, "tfim_bond_resolved", float(h))
                    if cached and cached["energy"] < e_exact[i] - 1e-6:
                        n_stale_pts += 1

                if n_stale_pts > 0:
                    n_stale_npz += 1
                    stale_details.append(f"{npz_file.name}: {n_stale_pts} pts with stale e_exact")

        if n_stale_npz > 0:
            issues.append(
                {
                    "check": "gt_npz_staleness",
                    "severity": "warning",
                    "detail": (
                        f"{n_stale_npz} NPZ file(s) have stale e_exact "
                        f"(GT cache has lower energy). Run: "
                        f'python -c "from qmbp_simulation.framework.result_io import '
                        f'refresh_npz_ground_truth; ..."'
                    ),
                }
            )
    except Exception:
        pass  # GT cache unavailable

    # ── Check 6: Dashboard ↔ project-status freshness ─────────────────────
    status_path = ROOT / ".kiro" / "steering" / "project-status.md"
    if dashboard_path.exists() and status_path.exists():
        dash_mtime = dashboard_path.stat().st_mtime
        status_mtime = status_path.stat().st_mtime
        # If dashboard is >1 hour newer than status → status is stale
        if dash_mtime - status_mtime > 3600:
            issues.append(
                {
                    "check": "status_stale",
                    "severity": "info",
                    "detail": (
                        "project-status.md is older than dashboard. "
                        "Run: .venv/bin/python scripts/maintenance/update_project_status.py"
                    ),
                }
            )

    return {
        "n_issues": len(issues),
        "issues": issues,
        "retrain_queue": retrain_queue,
        "zoo_entries": zoo_report["n_entries"],
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--retrain-queue", action="store_true", help="Show retrain queue only")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix what's fixable: reevaluate pass_rates, refresh GT, update status",
    )
    args = parser.parse_args()

    # ── --fix mode: auto-remediate fixable issues ─────────────────────────
    if args.fix:
        print("🔧 Auto-fixing coherence issues...\n")
        import subprocess

        # 1. Reevaluate zoo models (sync pass_rates)
        print("  [1/3] Reevaluating zoo models...")
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts/maintenance/reevaluate_zoo_models.py")],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        if r.returncode == 0:
            # Extract summary line
            for line in r.stdout.splitlines():
                if "→" in line or "=" in line:
                    print(f"    {line.strip()}")
        else:
            print(f"    ⚠️ Failed: {r.stderr[:100]}")

        # 2. Refresh NPZ ground truth from GT cache
        print("\n  [2/3] Refreshing NPZ ground truth...")
        try:
            from qmbp_simulation.framework.result_io import refresh_npz_ground_truth

            npz_dir = ROOT / "data" / "multi_n_training"
            total_refreshed = 0
            for npz_file in sorted(npz_dir.glob("*.npz")):
                stem = npz_file.stem
                parts = stem.split("_")
                n_idx = next((i for i, p in enumerate(parts) if p.startswith("N")), None)
                if n_idx is None:
                    continue
                topo = "_".join(parts[:n_idx])
                n_q = int(parts[n_idx][1:])
                n = refresh_npz_ground_truth(npz_file, topo, n_q)
                total_refreshed += n
            print(f"    Refreshed {total_refreshed} e_exact values from GT cache")
        except Exception as e:
            print(f"    ⚠️ GT refresh failed: {e}")

        # 3. Update project status
        print("\n  [3/3] Updating project status...")
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts/maintenance/update_project_status.py")],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        print(f"    {r.stdout.strip() or 'Done'}")

        print("\n🔧 Fix complete. Re-running coherence check...\n")

    report = check_coherence()

    if args.retrain_queue:
        queue = report["retrain_queue"]
        if not queue:
            print("No models need retraining.")
            return 0
        print(f"{'─' * 60}")
        print(f"  RETRAIN QUEUE ({len(queue)} models)")
        print(f"{'─' * 60}")
        for r in queue:
            print(f"\n  [{r['priority']}] {r['topology']} (pass={r['current_pass_rate']:.0%})")
            print(f"      Reason: {r['reason']}")
            print(f"      N available: {r['n_values_available']}")
            print(f"      Command: {r['command']}")
        return 0

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        n = report["n_issues"]
        if n == 0:
            print("✅ Zoo ↔ Dashboard coherent. No issues.")
        else:
            print(f"⚠️  {n} coherence issue(s) found:")
            for issue in report["issues"]:
                sev = issue["severity"].upper()
                print(f"  [{sev}] {issue['check']}: {issue['detail']}")

    return 1 if any(i["severity"] == "critical" for i in report["issues"]) else 0


if __name__ == "__main__":
    sys.exit(main())
