#!/usr/bin/env python3
"""Inspect data/ directory: cache sizes, NPZ quality, organization.

Reusable diagnostic for verifying data integrity before running experiments.
Usage:
    .venv/bin/python scripts/maintenance/inspect_data_stores.py
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def main():
    print("=" * 60)
    print("DATA STORAGE AUDIT")
    print("=" * 60)

    issues = []

    # 1. Ground Truth Cache
    gt_path = DATA / "ground_truth_cache.json"
    gt = {}
    if gt_path.exists():
        with open(gt_path) as f:
            raw = json.load(f)
        # Handle both v2 ({"version":..., "entries":{...}}) and legacy flat dict
        if isinstance(raw, dict) and "entries" in raw:
            gt = raw["entries"]
        else:
            gt = raw
        print(f"\n[GroundTruthCache] {gt_path.name}")
        print(f"  Entries: {len(gt)}")
        print(f"  File size: {gt_path.stat().st_size / 1024:.1f} KB")
        topos, ns, models = set(), set(), set()
        h_values_per_config = {}
        for key in gt:
            parts = key.split("|")
            if len(parts) >= 4:
                topos.add(parts[0])
                ns.add(int(parts[1]))
                models.add(parts[2])
                cfg = f"{parts[0]}|{parts[1]}|{parts[2]}"
                h_values_per_config.setdefault(cfg, []).append(float(parts[3]))
        print(f"  Topologies: {sorted(topos)}")
        print(f"  N values: {sorted(ns)}")
        print(f"  Models: {sorted(models)}")
        print(f"  Configs with data:")
        for k, hs in sorted(h_values_per_config.items()):
            print(f"    {k}: {len(hs)} h-points, h=[{min(hs):.2f}, {max(hs):.2f}]")
        n_bad = sum(1 for v in gt.values() if not v.get("gap") or v["gap"] == 0)
        if n_bad:
            issues.append(f"GT cache: {n_bad} entries with missing/zero gap")
    else:
        print(f"\n[GroundTruthCache] NOT FOUND")
        issues.append("GT cache file missing")

    # 2. Eval Cache
    ec_path = DATA / "eval_cache.json"
    if ec_path.exists():
        with open(ec_path) as f:
            ec = json.load(f)
        entries = ec.get("entries", ec) if isinstance(ec, dict) else ec
        print(f"\n[EvalCache] {ec_path.name}")
        print(f"  Entries: {len(entries)}")
        print(f"  File size: {ec_path.stat().st_size / 1024:.1f} KB")
        prefixes = {}
        for key in list(entries.keys())[:500]:
            prefix = "|".join(key.split("|")[:4])
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
        print(f"  Top configs (by entry count):")
        for p, count in sorted(prefixes.items(), key=lambda x: -x[1])[:8]:
            print(f"    {p}: {count} entries")
    else:
        print(f"\n[EvalCache] NOT FOUND")

    # 3. NPZ Training Data
    npz_dir = DATA / "multi_n_training"
    if npz_dir.exists():
        print(f"\n[NPZ Training Data] {npz_dir.relative_to(ROOT)}")
        for npz_file in sorted(npz_dir.glob("*.npz")):
            data = np.load(npz_file, allow_pickle=True)
            h_vals = data["h_values"]
            theta = data["theta_opt"]
            n_points = len(h_vals)
            n_params = theta.shape[1] if theta.ndim == 2 else 0
            n_nan = int(np.sum(~np.isfinite(theta)))
            e_key = "e_vqe" if "e_vqe" in data else ("energies" if "energies" in data else None)
            print(f"  {npz_file.name}:")
            print(f"    {n_points} pts, {n_params} params, h=[{h_vals.min():.2f},{h_vals.max():.2f}]")
            if n_nan:
                issues.append(f"NPZ {npz_file.name}: {n_nan} NaN/Inf in theta_opt!")
                print(f"    ⚠️ NaN count: {n_nan}")
            if e_key:
                e = data[e_key]
                print(f"    E range: [{e.min():.4f}, {e.max():.4f}]")
            else:
                issues.append(f"NPZ {npz_file.name}: no energy field (e_vqe/energies)")
            if "e_exact" not in data:
                issues.append(f"NPZ {npz_file.name}: missing e_exact")
            elif e_key:
                de = np.abs(data[e_key] - data["e_exact"])
                print(f"    |ΔE| range: [{de.min():.2e}, {de.max():.2e}]")
            if "de_gaps" not in data:
                issues.append(f"NPZ {npz_file.name}: no de_gaps field")
            # Check GT coverage
            parts = npz_file.stem.split("_")
            n_idx = next((i for i, p in enumerate(parts) if p.startswith("N")), None)
            if n_idx and gt:
                topo = "_".join(parts[:n_idx])
                n_val = int(parts[n_idx][1:])
                missing_gt = 0
                for h in h_vals:
                    key = f"{topo}|{n_val}|tfim_bond_resolved|{float(h):.6f}"
                    if key not in gt:
                        missing_gt += 1
                if missing_gt:
                    issues.append(
                        f"NPZ {npz_file.name}: {missing_gt}/{n_points} h-points "
                        f"have no GT cache entry"
                    )
    else:
        print(f"\n[NPZ Training Data] NOT FOUND")

    # 4. Model Zoo
    zoo_dir = DATA / "model_zoo"
    manifest_path = zoo_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        entries_list = manifest if isinstance(manifest, list) else manifest.get("entries", [])
        print(f"\n[Model Zoo] {len(entries_list)} models registered")
        for entry in entries_list:
            name = entry.get("checkpoint_file", "?")
            pr = entry.get("pass_rate", 0)
            ntp = entry.get("n_training_points", 0)
            topo = entry.get("topology", "?")
            n = entry.get("n_qubits", "?")
            print(f"  {name}: {topo} N={n}, pass={pr:.0%}, pts={ntp}")
        ckpt_dir = zoo_dir / "checkpoints"
        if ckpt_dir.exists():
            registered = {e.get("checkpoint_file") for e in entries_list}
            for ckpt in sorted(ckpt_dir.glob("*.pt")):
                status = "✓" if ckpt.name in registered else "⚠️ ORPHAN"
                print(f"    [{status}] {ckpt.name} ({ckpt.stat().st_size/1024:.0f}KB)")
                if ckpt.name not in registered:
                    issues.append(f"Zoo orphan: {ckpt.name} not in manifest")
    else:
        print(f"\n[Model Zoo] NOT FOUND")

    # Summary
    print("\n" + "=" * 60)
    if not issues:
        print("RESULT: All data stores consistent ✅")
    else:
        print(f"ISSUES FOUND: {len(issues)}")
        for issue in issues:
            print(f"  ⚠️ {issue}")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════════════
    # 5. Training Data Quality Analysis
    # ═══════════════════════════════════════════════════════════════
    if npz_dir.exists():
        from qmbp_simulation.analysis.metrics import MAX_ABS_ERROR, MIN_FIDELITY

        print(f"\n{'=' * 60}")
        print("TRAINING DATA QUALITY ANALYSIS")
        print(f"{'=' * 60}")

        total_pts, total_good, total_marginal, total_bad = 0, 0, 0, 0
        for npz_file in sorted(npz_dir.glob("*.npz")):
            data = np.load(npz_file, allow_pickle=True)
            h_vals = data["h_values"]
            theta = data["theta_opt"]
            e_key = "e_vqe" if "e_vqe" in data else ("energies" if "energies" in data else None)
            n_pts = len(h_vals)
            total_pts += n_pts

            if e_key is None or "e_exact" not in data:
                print(f"\n  {npz_file.name}: {n_pts} pts — ⚠️ missing energy fields")
                continue

            e_vqe = data[e_key]
            e_exact = data["e_exact"]
            abs_err = np.abs(e_vqe - e_exact)
            gaps = data["gaps"] if "gaps" in data else np.ones(n_pts)
            de_gaps = abs_err / np.maximum(gaps, 1e-10)

            # Classify each point
            good = (de_gaps < 0.05) & (abs_err < MAX_ABS_ERROR)
            marginal = (de_gaps < 0.10) & ~good
            bad = ~good & ~marginal

            n_good = int(good.sum())
            n_marginal = int(marginal.sum())
            n_bad = int(bad.sum())
            total_good += n_good
            total_marginal += n_marginal
            total_bad += n_bad

            # θ smoothness (max consecutive change)
            if n_pts > 1:
                sorted_idx = np.argsort(h_vals)
                theta_sorted = theta[sorted_idx]
                diffs = np.max(np.abs(np.diff(theta_sorted, axis=0)), axis=1)
                max_jump = float(diffs.max())
            else:
                max_jump = 0.0

            status = "✅" if n_bad == 0 else ("⚠️" if n_bad <= 2 else "❌")
            print(f"\n  {npz_file.name}: {n_pts} pts {status}")
            print(f"    Quality: {n_good} good, {n_marginal} marginal, {n_bad} bad")
            print(f"    |ΔE|:  mean={abs_err.mean():.4f} max={abs_err.max():.4f}")
            print(f"    ΔE/gap: mean={de_gaps.mean():.4f} max={de_gaps.max():.4f}")
            print(f"    θ max jump: {max_jump:.3f} {'(smooth)' if max_jump < 0.5 else '⚠️ discontinuous'}")

            # Flag specific problematic points
            if n_bad > 0:
                bad_idx = np.where(bad)[0]
                for bi in bad_idx[:3]:
                    print(f"    ❌ h={h_vals[bi]:.2f}: |ΔE|={abs_err[bi]:.3f} ΔE/gap={de_gaps[bi]:.3f}")

        print(f"\n  {'─' * 40}")
        print(f"  TOTAL: {total_pts} points")
        print(f"    Good (ΔE/gap<5% & |ΔE|<0.10): {total_good} ({total_good/max(total_pts,1):.0%})")
        print(f"    Marginal (5-10%): {total_marginal} ({total_marginal/max(total_pts,1):.0%})")
        print(f"    Bad (>10% or |ΔE|>0.10): {total_bad} ({total_bad/max(total_pts,1):.0%})")

        is_valid = total_bad <= total_pts * 0.1
        print(f"\n  Training dataset {'VALID ✅' if is_valid else 'NEEDS CLEANING ❌'}")
        if not is_valid:
            print(f"    > 10% bad points — consider re-running with higher maxiter")

    return len(issues)


if __name__ == "__main__":
    raise SystemExit(main())
