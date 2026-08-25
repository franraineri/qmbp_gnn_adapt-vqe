#!/usr/bin/env python3
"""Promote quality extrapolation data to training data.

Moves high-quality θ_opt from data/large_n_extrapolation/ into
data/multi_n_training/ so the MPNN can learn from large-N examples.

Quality filter:
- pass_rate@10% >= 50% (at least half the h-points have ΔE/gap < 10%)
- Must have theta_opt (not just theta_pred)
- Must have e_exact and gaps (for quality verification)

This is a ONE-WAY promotion: once in multi_n_training/, the data will be
picked up by all training scripts and dashboard regeneration.

Usage:
    # Dry-run: see what would be promoted
    .venv/bin/python scripts/maintenance/promote_extrapolation_data.py --dry-run

    # Promote with default quality threshold (pass@10% >= 50%)
    .venv/bin/python scripts/maintenance/promote_extrapolation_data.py

    # Stricter quality (pass@5% >= 50%)
    .venv/bin/python scripts/maintenance/promote_extrapolation_data.py --min-pass 0.50 --threshold 0.05

    # More permissive (pass@10% >= 30%)
    .venv/bin/python scripts/maintenance/promote_extrapolation_data.py --min-pass 0.30
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

EXTRAP_DIR = ROOT / "data" / "large_n_extrapolation"
TRAINING_DIR = ROOT / "data" / "multi_n_training"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote quality extrapolation data to training set"
    )
    parser.add_argument(
        "--min-pass", type=float, default=0.50,
        help="Minimum pass rate to promote (default: 0.50)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.10,
        help="ΔE/gap threshold for pass_rate calculation (default: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--max-n", type=int, default=60,
        help="Maximum N to promote (default: 60, skip N>=100 which are less reliable)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only show what would be promoted, don't copy",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing files in training dir",
    )
    return parser.parse_args()


def analyze_npz(npz_path: Path, threshold: float) -> dict:
    """Analyze an extrapolation NPZ file for promotion eligibility."""
    data = np.load(str(npz_path), allow_pickle=True)
    keys = list(data.keys())

    result = {
        "file": npz_path.name,
        "eligible": False,
        "reason": "",
    }

    # Must have theta_opt
    if "theta_opt" not in data:
        result["reason"] = "no theta_opt"
        return result

    # Must have e_exact and gaps
    if "e_exact" not in data:
        result["reason"] = "no e_exact"
        return result

    h_vals = data["h_values"]
    theta_opt = data["theta_opt"]
    n_points = len(h_vals)

    # Check for NaN in theta
    n_nan = int(np.sum(~np.isfinite(theta_opt)))
    if n_nan > 0:
        result["reason"] = f"{n_nan} NaN in theta_opt"
        return result

    # Compute pass rate
    if "de_gaps" in data:
        de_gaps = data["de_gaps"]
    elif "gaps" in data:
        e_key = "e_vqe" if "e_vqe" in data else ("energies" if "energies" in data else None)
        if e_key is None:
            result["reason"] = "no energy field"
            return result
        abs_err = np.abs(data[e_key] - data["e_exact"])
        de_gaps = abs_err / np.maximum(data["gaps"], 1e-10)
    else:
        result["reason"] = "no de_gaps or gaps field"
        return result

    pass_rate = float((de_gaps < threshold).mean())
    mean_de_gap = float(de_gaps.mean())
    n_params = theta_opt.shape[1] if theta_opt.ndim == 2 else 0

    # Parse N from filename
    stem = npz_path.stem
    try:
        n_str = stem.split("_N")[1].split("_")[0]
        n_val = int(n_str)
    except (IndexError, ValueError):
        result["reason"] = "can't parse N from filename"
        return result

    # Parse topology
    topo = stem.split("_N")[0]

    result.update({
        "topology": topo,
        "n_qubits": n_val,
        "n_points": n_points,
        "n_params": n_params,
        "pass_rate": pass_rate,
        "mean_de_gap": mean_de_gap,
        "h_range": [float(h_vals.min()), float(h_vals.max())],
        "has_quality_tier": "quality_tier" in data,
    })

    return result


def promote_file(
    src: Path,
    dst: Path,
    *,
    ensure_fields: bool = True,
) -> bool:
    """Copy extrapolation NPZ to training dir, ensuring required fields.

    Adds missing fields (de_gaps, quality_tier) if not present.
    """
    data = dict(np.load(str(src), allow_pickle=True))

    # Ensure de_gaps field exists
    if "de_gaps" not in data:
        e_key = "e_vqe" if "e_vqe" in data else ("energies" if "energies" in data else None)
        if e_key and "e_exact" in data and "gaps" in data:
            abs_err = np.abs(data[e_key] - data["e_exact"])
            data["de_gaps"] = abs_err / np.maximum(data["gaps"], 1e-10)

    # Ensure quality_tier field (mark as "approximate" since these are extrapolation)
    if "quality_tier" not in data:
        n_pts = len(data["h_values"])
        data["quality_tier"] = np.array(["approximate"] * n_pts)

    # Ensure e_vqe field (some files use "energies" key)
    if "e_vqe" not in data and "energies" in data:
        data["e_vqe"] = data["energies"]

    np.savez(str(dst), **data)
    return True


def main() -> int:
    args = parse_args()

    if not EXTRAP_DIR.exists():
        print("  ❌ Extrapolation data dir not found:", EXTRAP_DIR)
        return 1

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PROMOTE EXTRAPOLATION DATA TO TRAINING")
    print(f"  Threshold: pass@{args.threshold*100:.0f}% >= {args.min_pass:.0%}")
    print(f"  Max N: {args.max_n}")
    print(f"  Mode: {'DRY-RUN' if args.dry_run else 'LIVE'}")
    print("=" * 70)

    # Analyze all extrapolation files
    candidates = []
    rejected = []

    for npz_file in sorted(EXTRAP_DIR.glob("*.npz")):
        info = analyze_npz(npz_file, args.threshold)

        # N filter
        n_val = info.get("n_qubits", 999)
        if n_val > args.max_n:
            info["reason"] = f"N={n_val} > max_n={args.max_n}"
            info["eligible"] = False
            rejected.append(info)
            continue

        if info.get("pass_rate", 0) >= args.min_pass:
            info["eligible"] = True
            candidates.append(info)
        else:
            info["reason"] = f"pass@{args.threshold*100:.0f}%={info.get('pass_rate', 0):.0%} < {args.min_pass:.0%}"
            rejected.append(info)

    # Show results
    print(f"\n  Eligible for promotion: {len(candidates)}")
    print(f"  Rejected: {len(rejected)}")

    if candidates:
        print(f"\n  {'File':<40} {'Topo':12s} {'N':>3} {'Pts':>4} {'Pass%':>6} {'ΔE/gap':>8}")
        print(f"  {'─'*40} {'─'*12} {'─'*3} {'─'*4} {'─'*6} {'─'*8}")
        total_pts = 0
        for c in sorted(candidates, key=lambda x: (x["topology"], x["n_qubits"])):
            total_pts += c["n_points"]
            print(
                f"  {c['file']:<40} {c['topology']:12s} {c['n_qubits']:>3} "
                f"{c['n_points']:>4} {c['pass_rate']:>5.0%} {c['mean_de_gap']:>8.4f}"
            )
        print(f"\n  TOTAL: {total_pts} new training points from {len(candidates)} files")

    if rejected:
        print(f"\n  Rejected ({len(rejected)}):")
        for r in rejected[:10]:
            print(f"    ❌ {r['file']}: {r.get('reason', '?')}")
        if len(rejected) > 10:
            print(f"    ... and {len(rejected) - 10} more")

    # Perform promotion
    if args.dry_run:
        print("\n  [DRY-RUN] No files copied. Remove --dry-run to proceed.")
        return 0

    n_promoted = 0
    n_skipped = 0
    for c in candidates:
        src = EXTRAP_DIR / c["file"]
        dst = TRAINING_DIR / c["file"]

        if dst.exists() and not args.force:
            # Check if existing has more or equal points
            existing = np.load(str(dst), allow_pickle=True)
            if len(existing["h_values"]) >= c["n_points"]:
                n_skipped += 1
                continue

        if promote_file(src, dst):
            n_promoted += 1
            print(f"  ✅ Promoted: {c['file']} ({c['n_points']} pts)")

    print(f"\n  Done: {n_promoted} promoted, {n_skipped} skipped (already exist with more data)")

    # Trigger dashboard regeneration
    if n_promoted > 0:
        print("\n  Triggering dashboard regeneration...")
        try:
            from qmbp_simulation.analysis.metrics import generate_model_quality_dashboard
            d = generate_model_quality_dashboard()
            print(f"  ✅ Dashboard regenerated: {d['n_configs']} configs")
        except Exception as e:
            print(f"  ⚠️ Dashboard regen failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
