#!/usr/bin/env python3
"""Upgrade legacy NPZ files to include quality_tier field.

Computes quality tier for each h-point based on ΔE/gap:
- verified: ΔE/gap < 5% AND |ΔE| < 0.10 (dual criterion)
- approximate: ΔE/gap < 10% OR passes single criterion
- unverified: ΔE/gap >= 10%

Usage:
    .venv/bin/python scripts/maintenance/upgrade_npz_quality_tiers.py
    .venv/bin/python scripts/maintenance/upgrade_npz_quality_tiers.py --dry-run
    .venv/bin/python scripts/maintenance/upgrade_npz_quality_tiers.py --backup
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
NPZ_DIR = DATA / "multi_n_training"
BACKUP_DIR = DATA / "multi_n_training_backup"

# Thresholds from analysis/metrics.py
DE_GAP_THRESHOLD = 0.05
DE_GAP_LOOSE = 0.10
MAX_ABS_ERROR = 0.10


def compute_quality_tier(
    e_vqe: np.ndarray,
    e_exact: np.ndarray,
    gaps: np.ndarray,
) -> np.ndarray:
    """Compute quality tier for each point."""
    n_points = len(e_vqe)
    tiers = np.array(["unverified"] * n_points, dtype=object)
    
    for i in range(n_points):
        ev = e_vqe[i]
        ex = e_exact[i]
        gap = gaps[i] if i < len(gaps) else 0.1
        
        # Skip invalid data
        if np.isnan(ev) or np.isnan(ex) or gap <= 0:
            continue
            
        abs_err = abs(ev - ex)
        de_gap = abs_err / max(gap, 1e-6)
        
        # Dual criterion for verified
        if de_gap < DE_GAP_THRESHOLD and abs_err < MAX_ABS_ERROR:
            tiers[i] = "verified"
        # Loose criterion for approximate
        elif de_gap < DE_GAP_LOOSE or abs_err < MAX_ABS_ERROR:
            tiers[i] = "approximate"
        # else remains unverified
    
    return tiers


def upgrade_npz_file(
    npz_path: Path,
    backup: bool = False,
    dry_run: bool = False,
) -> dict:
    """Upgrade a single NPZ file with quality tier."""
    result = {
        "file": npz_path.name,
        "status": "skipped",
        "tiers": {"verified": 0, "approximate": 0, "unverified": 0},
    }
    
    try:
        data = dict(np.load(str(npz_path), allow_pickle=True))
    except Exception as e:
        result["status"] = f"error: {e}"
        return result
    
    # Check if already has quality_tier
    if "quality_tier" in data:
        tiers = list(data["quality_tier"])
        result["tiers"] = {
            "verified": tiers.count("verified"),
            "approximate": tiers.count("approximate"),
            "unverified": tiers.count("unverified"),
        }
        result["status"] = "already_upgraded"
        return result
    
    # Require minimum fields
    required = ["h_values", "e_vqe", "e_exact", "gaps"]
    missing = [k for k in required if k not in data]
    if missing:
        result["status"] = f"missing_fields: {missing}"
        return result
    
    # Compute quality tiers
    tiers = compute_quality_tier(
        data["e_vqe"],
        data["e_exact"],
        data["gaps"],
    )
    
    tier_counts = {
        "verified": int((tiers == "verified").sum()),
        "approximate": int((tiers == "approximate").sum()),
        "unverified": int((tiers == "unverified").sum()),
    }
    result["tiers"] = tier_counts
    
    if dry_run:
        result["status"] = "would_upgrade"
        return result
    
    # Backup if requested
    if backup:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_path = BACKUP_DIR / npz_path.name
        shutil.copy2(npz_path, backup_path)
    
    # Add quality_tier and save
    data["quality_tier"] = tiers
    data["quality_tier_version"] = "1.0"
    data["upgraded_at"] = datetime.now(timezone.utc).isoformat()
    
    np.savez(str(npz_path), **data)
    result["status"] = "upgraded"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upgrade legacy NPZ files with quality tier field"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without modifying files"
    )
    parser.add_argument(
        "--backup", action="store_true",
        help="Create backup of each file before upgrading"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Upgrade a single file (filename only, not path)"
    )
    args = parser.parse_args()
    
    if not NPZ_DIR.exists():
        print(f"NPZ directory not found: {NPZ_DIR}")
        return 1
    
    print("=" * 60)
    print("NPZ Quality Tier Upgrade")
    print(f"Directory: {NPZ_DIR}")
    if args.dry_run:
        print("MODE: Dry run (no changes)")
    elif args.backup:
        print(f"MODE: Backup enabled → {BACKUP_DIR}")
    print("=" * 60)
    
    # Get files to process
    if args.file:
        npz_files = [NPZ_DIR / args.file]
        if not npz_files[0].exists():
            print(f"File not found: {args.file}")
            return 1
    else:
        npz_files = sorted(NPZ_DIR.glob("*.npz"))
    
    print(f"\nProcessing {len(npz_files)} files...\n")
    
    total = {"verified": 0, "approximate": 0, "unverified": 0}
    n_upgraded = 0
    n_skipped = 0
    n_already = 0
    n_error = 0
    
    for npz_file in npz_files:
        result = upgrade_npz_file(npz_file, backup=args.backup, dry_run=args.dry_run)
        
        status = result["status"]
        tiers = result["tiers"]
        total["verified"] += tiers.get("verified", 0)
        total["approximate"] += tiers.get("approximate", 0)
        total["unverified"] += tiers.get("unverified", 0)
        
        # Status emoji
        if status == "upgraded":
            emoji = "✅"
            n_upgraded += 1
        elif status == "would_upgrade":
            emoji = "🔄"
            n_upgraded += 1
        elif status == "already_upgraded":
            emoji = "⏭️"
            n_already += 1
        elif status.startswith("error") or status.startswith("missing"):
            emoji = "❌"
            n_error += 1
        else:
            emoji = "⚠️"
            n_skipped += 1
        
        # Format tier counts
        v, a, u = tiers.get("verified", 0), tiers.get("approximate", 0), tiers.get("unverified", 0)
        print(f"  {emoji} {result['file']}: {status} (v={v}, a={a}, u={u})")
    
    # Summary
    total_pts = total["verified"] + total["approximate"] + total["unverified"]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files processed: {len(npz_files)}")
    print(f"  Upgraded: {n_upgraded}")
    print(f"  Already done: {n_already}")
    print(f"  Skipped/Error: {n_skipped + n_error}")
    print()
    print(f"Total points: {total_pts}")
    pct_v = total["verified"] * 100 // max(total_pts, 1)
    pct_a = total["approximate"] * 100 // max(total_pts, 1)
    pct_u = total["unverified"] * 100 // max(total_pts, 1)
    print(f"  ✅ Verified: {total['verified']} ({pct_v}%)")
    print(f"  ⚠️ Approximate: {total['approximate']} ({pct_a}%)")
    print(f"  ❓ Unverified: {total['unverified']} ({pct_u}%)")
    print("=" * 60)
    
    if args.dry_run:
        print("\nRun without --dry-run to apply changes.")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
