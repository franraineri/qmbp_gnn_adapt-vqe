#!/usr/bin/env python3
"""Upgrade NPZ files: refresh ground truth, recompute quality tiers, refresh zoo scores.

Four-phase pipeline:
1. Refresh e_exact from GroundTruthCache (prevents stale GT from inflating ΔE/gap)
2. Recompute quality_tier per point using classify_point_failure() from metrics
3. Refresh zoo training_quality_scores with updated NPZ stats
4. Auto-detect new training exclusions (data that became not_useful after refresh)

Quality tiers (assigned by classify_point_failure categories):
- verified: category="pass" (passes dual criterion)
- approximate: category="near_pass" or "gap_masked" (close, refinable)
- unverified: all other categories (moderate_error, severe, ansatz_limited, data_error)

Usage:
    .venv/bin/python scripts/maintenance/upgrade_npz_quality_tiers.py
    .venv/bin/python scripts/maintenance/upgrade_npz_quality_tiers.py --dry-run
    .venv/bin/python scripts/maintenance/upgrade_npz_quality_tiers.py --skip-gt-refresh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
NPZ_DIRS = [
    DATA / "multi_n_training",
    DATA / "large_n_extrapolation",
]


def compute_quality_tier_for_npz(
    e_vqe: np.ndarray,
    e_exact: np.ndarray,
    gaps: np.ndarray,
) -> np.ndarray:
    """Compute quality tier for each point using classify_point_failure.

    Uses the canonical per-point classification to assign tiers consistently
    with the rest of the pipeline (runners, reports, exclusion system).

    Tier assignment:
    - "verified": passes dual criterion (ΔE/gap < 5% AND |ΔE| < 0.10)
    - "approximate": near_pass or gap_masked (refinable, close to threshold)
    - "unverified": everything else (severe errors, ansatz-limited, data errors)
    """
    from qmbp_simulation.analysis.metrics import classify_point_failure

    n_points = len(e_vqe)
    tiers = np.array(["unverified"] * n_points, dtype=object)

    for i in range(n_points):
        ev, ex = float(e_vqe[i]), float(e_exact[i])
        gap = float(gaps[i]) if i < len(gaps) else 0.1

        if not np.isfinite(ev) or not np.isfinite(ex) or gap <= 0:
            continue

        abs_err = abs(ev - ex)
        de_gap = abs_err / max(gap, 1e-10)

        cls = classify_point_failure(de_gap=de_gap, abs_error=abs_err, gap=gap)

        if cls.category == "pass":
            tiers[i] = "verified"
        elif cls.category in ("near_pass", "gap_masked"):
            tiers[i] = "approximate"
        # else: moderate_error, severe_error, ansatz_limited, data_error → unverified

    return tiers


def _get_energy_key(data: dict) -> str | None:
    """Get the correct energy key from NPZ data (handles legacy 'energies')."""
    if "e_vqe" in data:
        return "e_vqe"
    if "energies" in data:
        return "energies"
    return None


def upgrade_single_npz(
    npz_path: Path,
    *,
    refresh_gt: bool = True,
    dry_run: bool = False,
) -> dict:
    """Upgrade a single NPZ: refresh GT → recompute tiers → atomic save."""
    result = {
        "file": npz_path.name,
        "source_dir": npz_path.parent.name,
        "status": "skipped",
        "n_points": 0,
        "tiers": {"verified": 0, "approximate": 0, "unverified": 0},
        "promotions": 0,  # approximate/unverified → verified
        "gt_refreshed": 0,
    }

    try:
        data = dict(np.load(str(npz_path), allow_pickle=True))
    except Exception as e:
        result["status"] = f"error_load: {e}"
        return result

    # Get energy key (handles legacy "energies" field)
    e_key = _get_energy_key(data)
    if e_key is None or "e_exact" not in data or "h_values" not in data:
        result["status"] = "missing_fields"
        return result

    n_points = len(data["h_values"])
    result["n_points"] = n_points
    if n_points == 0:
        result["status"] = "empty"
        return result

    # Ensure gaps exist
    if "gaps" not in data:
        data["gaps"] = np.zeros(n_points)

    e_vqe = np.asarray(data[e_key], dtype=np.float64)
    e_exact = np.asarray(data["e_exact"], dtype=np.float64)
    gaps = np.asarray(data["gaps"], dtype=np.float64)

    # Phase 1: Refresh ground truth from GroundTruthCache
    if refresh_gt and not dry_run:
        # Extract topology/N from filename: {topology}_N{n}_p{p}.npz
        fname = npz_path.stem
        try:
            parts = fname.rsplit("_", 2)  # ["chain_1d", "N10", "p1"]
            if len(parts) >= 3 and parts[-2].startswith("N") and parts[-1].startswith("p"):
                topology = parts[0] if len(parts) == 3 else "_".join(parts[:-2])
                n_qubits = int(parts[-2][1:])
                from qmbp_simulation.framework.result_io import refresh_npz_ground_truth

                n_gt = refresh_npz_ground_truth(
                    npz_path,
                    topology=topology,
                    n_qubits=n_qubits,
                )
                result["gt_refreshed"] = n_gt
                if n_gt > 0:
                    # Reload data after GT refresh
                    data = dict(np.load(str(npz_path), allow_pickle=True))
                    e_exact = np.asarray(data["e_exact"], dtype=np.float64)
                    gaps = np.asarray(data["gaps"], dtype=np.float64)
        except (ValueError, IndexError):
            pass  # Can't parse filename — skip GT refresh

    # Phase 2: Compute new quality tiers
    new_tiers = compute_quality_tier_for_npz(e_vqe, e_exact, gaps)

    # Count promotions (old tier worse than new tier)
    old_tiers = data.get("quality_tier")
    if old_tiers is not None:
        old_tiers = np.asarray(old_tiers)
        tier_rank = {"unverified": 0, "approximate": 1, "verified": 2}
        for i in range(n_points):
            old_rank = tier_rank.get(str(old_tiers[i]), 0)
            new_rank = tier_rank.get(str(new_tiers[i]), 0)
            if new_rank > old_rank:
                result["promotions"] += 1

    # Count final tiers
    result["tiers"] = {
        "verified": int((new_tiers == "verified").sum()),
        "approximate": int((new_tiers == "approximate").sum()),
        "unverified": int((new_tiers == "unverified").sum()),
    }

    if dry_run:
        result["status"] = "would_upgrade"
        return result

    # Phase 2b: Atomic save with updated tiers
    data["quality_tier"] = new_tiers

    # Recompute de_gaps with current e_exact (may have been refreshed)
    data["de_gaps"] = np.abs(e_vqe - e_exact) / np.maximum(gaps, 1e-10)

    # Normalize energy key to canonical "e_vqe"
    if e_key == "energies":
        data["e_vqe"] = data.pop("energies")

    tmp_path = npz_path.with_suffix(".tmp.npz")
    try:
        np.savez(tmp_path, **data)
        tmp_path.rename(npz_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    result["status"] = "upgraded" if old_tiers is None else "refreshed"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upgrade NPZ quality tiers + refresh GT + update zoo scores"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying")
    parser.add_argument(
        "--skip-gt-refresh", action="store_true", help="Skip GroundTruthCache refresh"
    )
    parser.add_argument("--file", type=str, default=None, help="Process single file")
    args = parser.parse_args()

    print("=" * 70)
    print("NPZ Quality Tier Upgrade + Ground Truth Refresh")
    if args.dry_run:
        print("MODE: Dry run")
    print("=" * 70)

    # Collect all NPZ files
    npz_files = []
    for d in NPZ_DIRS:
        if d.exists():
            npz_files.extend(sorted(d.glob("*.npz")))

    if args.file:
        npz_files = [f for f in npz_files if f.name == args.file]
        if not npz_files:
            print(f"File not found: {args.file}")
            return 1

    print(f"Files to process: {len(npz_files)}")
    print()

    # Process
    totals = {"verified": 0, "approximate": 0, "unverified": 0}
    n_upgraded = 0
    n_refreshed = 0
    n_gt_refreshed = 0
    total_promotions = 0
    n_error = 0

    for npz_file in npz_files:
        result = upgrade_single_npz(
            npz_file,
            refresh_gt=not args.skip_gt_refresh,
            dry_run=args.dry_run,
        )

        status = result["status"]
        tiers = result["tiers"]
        totals["verified"] += tiers["verified"]
        totals["approximate"] += tiers["approximate"]
        totals["unverified"] += tiers["unverified"]
        total_promotions += result["promotions"]
        n_gt_refreshed += result["gt_refreshed"]

        if status == "upgraded":
            n_upgraded += 1
        elif status == "refreshed":
            n_refreshed += 1
        elif status.startswith("error"):
            n_error += 1

        # Compact per-file output
        v, a, u = tiers["verified"], tiers["approximate"], tiers["unverified"]
        promo = f" (+{result['promotions']}↑)" if result["promotions"] > 0 else ""
        gt = f" GT:{result['gt_refreshed']}↻" if result["gt_refreshed"] > 0 else ""
        emoji = {"upgraded": "✅", "refreshed": "🔄", "would_upgrade": "🔍"}.get(status, "⚠️")
        print(f"  {emoji} {result['source_dir']}/{result['file']}: v={v} a={a} u={u}{promo}{gt}")

    # Summary
    total_pts = totals["verified"] + totals["approximate"] + totals["unverified"]
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"Files: {len(npz_files)} ({n_upgraded} new + {n_refreshed} refreshed + {n_error} errors)"
    )
    print(f"GT refreshed: {n_gt_refreshed} h-points updated from GroundTruthCache")
    print(f"Promotions: {total_promotions} points upgraded to higher tier")
    print()
    print(f"Total points: {total_pts}")
    if total_pts > 0:
        print(
            f"  ✅ Verified:    {totals['verified']:>5} ({totals['verified'] * 100 // total_pts}%)"
        )
        print(
            f"  ⚠️  Approximate: {totals['approximate']:>5} ({totals['approximate'] * 100 // total_pts}%)"
        )
        print(
            f"  ❓ Unverified:  {totals['unverified']:>5} ({totals['unverified'] * 100 // total_pts}%)"
        )

    # Phase 3: Refresh zoo quality scores
    if not args.dry_run and (n_upgraded > 0 or n_refreshed > 0 or total_promotions > 0):
        print()
        print("Refreshing zoo quality scores...")
        try:
            from qmbp_simulation.predictors.model_zoo import refresh_zoo_quality_scores

            updated = refresh_zoo_quality_scores()
            if updated:
                print(f"  Updated {len(updated)} model scores")
            else:
                print("  All scores already current")
        except Exception as e:
            print(f"  Score refresh failed (non-blocking): {e}")

    # Phase 4: Auto-detect new exclusions (data that became not_useful after GT refresh)
    if not args.dry_run:
        print()
        print("Checking for new exclusion candidates...")
        try:
            from qmbp_simulation.analysis.metrics import auto_detect_exclusions

            new_exclusions = auto_detect_exclusions(dry_run=False)
            if new_exclusions:
                print(f"  Auto-excluded {len(new_exclusions)} NPZ file(s):")
                for exc in new_exclusions[:5]:
                    print(f"    • {exc['file']} ({exc['topology']} N={exc['n_qubits']})")
                if len(new_exclusions) > 5:
                    print(f"    ... and {len(new_exclusions) - 5} more")
            else:
                print("  No new exclusions detected")
        except Exception as e:
            print(f"  Exclusion check failed (non-blocking): {e}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
