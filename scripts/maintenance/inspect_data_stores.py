#!/usr/bin/env python3
"""Inspect data/ directory: cache sizes, NPZ quality, organization.

Reusable diagnostic for verifying data integrity before running experiments.
Usage:
    .venv/bin/python scripts/maintenance/inspect_data_stores.py
    .venv/bin/python scripts/maintenance/inspect_data_stores.py --validate-dashboard
"""
import argparse
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def main():
    parser = argparse.ArgumentParser(description="Inspect data stores")
    parser.add_argument(
        "--validate-dashboard", action="store_true",
        help="Recompute metrics from raw NPZ and cross-validate against dashboard",
    )
    args = parser.parse_args()

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
            
            # ── Cross-integration: Quality Tier Distribution ─────────────
            if "quality_tier" in data:
                tiers = data["quality_tier"].tolist()
                n_verified = tiers.count("verified")
                n_approx = tiers.count("approximate")
                n_unverified = tiers.count("unverified")
                quality_score = (n_verified * 1.0 + n_approx * 0.7 + n_unverified * 0.5) / max(n_points, 1)
                status = "✅" if n_verified > n_points * 0.5 else ("⚠️" if n_verified > 0 else "❓")
                print(f"    Quality tiers: ✅{n_verified} ⚠️{n_approx} ❓{n_unverified} (score={quality_score:.2f}) {status}")
            else:
                print(f"    Quality tiers: 📜 legacy NPZ (no quality_tier field)")
            
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

    # 3b. Large-N Extrapolation Data (bootstrapping cycle)
    extrap_dir = DATA / "large_n_extrapolation"
    if extrap_dir.exists():
        extrap_files = sorted(extrap_dir.glob("*.npz"))
        if extrap_files:
            print(f"\n[Large-N Extrapolation Data] {len(extrap_files)} files")
            for npz_file in extrap_files:
                data = np.load(str(npz_file), allow_pickle=True)
                h_vals = data["h_values"]
                n_pts = len(h_vals)
                tier_str = ""
                if "quality_tier" in data:
                    tiers = data["quality_tier"].tolist()
                    n_approx = sum(1 for t in tiers if str(t) == "approximate")
                    n_verified = sum(1 for t in tiers if str(t) == "verified")
                    tier_str = f" (V={n_verified} A={n_approx})"
                print(f"  {npz_file.name}: {n_pts} pts, "
                      f"h=[{h_vals.min():.2f},{h_vals.max():.2f}]{tier_str}")

    # 4. Model Zoo — use validate_zoo() for integrity + orphan detection
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
            print(f"  {name}: {topo} N={n}, pass_dual={pr:.0%}, pts={ntp}")

        # Use validate_zoo() for SHA256 integrity + missing file detection
        try:
            from qmbp_simulation.predictors.model_zoo import validate_zoo
            zoo_report = validate_zoo()
            print(f"  Integrity: {zoo_report['n_valid']} valid, "
                  f"{zoo_report['n_missing']} missing, "
                  f"{zoo_report['n_corrupted']} corrupted")
            if zoo_report["n_corrupted"] > 0:
                for err in zoo_report["errors"]:
                    issues.append(f"Zoo integrity: {err[:80]}")
            if zoo_report["n_missing"] > 0:
                issues.append(f"Zoo: {zoo_report['n_missing']} checkpoints missing from disk")
        except (ImportError, Exception) as e:
            print(f"  ⚠️ validate_zoo() failed: {e}")

        # Orphan detection (files on disk not in manifest)
        ckpt_dir = zoo_dir / "checkpoints"
        if ckpt_dir.exists():
            registered = {e.get("checkpoint_file") for e in entries_list}
            for ckpt in sorted(ckpt_dir.glob("*.pt")):
                if ckpt.name not in registered:
                    print(f"    [⚠️ ORPHAN] {ckpt.name} ({ckpt.stat().st_size/1024:.0f}KB)")
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
    # 5. Training Data Quality Analysis + H-Frontier per NPZ
    # ═══════════════════════════════════════════════════════════════
    if npz_dir.exists():
        from qmbp_simulation.analysis.metrics import (
            MAX_ABS_ERROR, MIN_FIDELITY, compute_h_frontier_from_npz,
            identify_failures, DE_GAP_THRESHOLD,
        )

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

            # Classify using identify_failures (dual criterion)
            per_h_results = [
                {"de_gap": float(de_gaps[i]), "abs_error": float(abs_err[i])}
                for i in range(n_pts)
            ]
            failure_indices = set(identify_failures(per_h_results))
            good = np.array([i not in failure_indices for i in range(n_pts)])
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

            # H-frontier (empirical boundary where ΔE/gap crosses 5%)
            frontier_result = compute_h_frontier_from_npz(npz_file)
            h_front = frontier_result.get("h_frontier")
            if h_front is not None:
                print(f"    h_frontier: {h_front:.3f} (below this → pipeline fails)")
            else:
                print(f"    h_frontier: N/A (all pass or all fail)")

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

    # ═══════════════════════════════════════════════════════════════
    # 5b. Quality Tier Breakdown + Training Viability Assessment
    # ═══════════════════════════════════════════════════════════════
    if npz_dir.exists():
        from collections import defaultdict as _defaultdict

        print(f"\n{'=' * 60}")
        print("QUALITY TIERS & TRAINING VIABILITY")
        print(f"{'=' * 60}")

        tier_totals = {"verified": 0, "approximate": 0, "unverified": 0}
        topo_tier_breakdown = _defaultdict(lambda: {"verified": 0, "approximate": 0, "unverified": 0, "n_values": set()})

        for npz_file in sorted(npz_dir.glob("*.npz")):
            data = np.load(str(npz_file), allow_pickle=True)
            parts = npz_file.stem.split("_")
            n_idx = next((i for i, p in enumerate(parts) if p.startswith("N")), None)
            if n_idx is None:
                continue
            topo = "_".join(parts[:n_idx])
            n_val = int(parts[n_idx][1:])

            # Count tiers
            if "quality_tier" in data:
                tiers = data["quality_tier"].tolist()
                for t in tiers:
                    tier_totals[str(t)] = tier_totals.get(str(t), 0) + 1
                    topo_tier_breakdown[topo][str(t)] = topo_tier_breakdown[topo].get(str(t), 0) + 1
            else:
                n_pts = len(data["h_values"])
                tier_totals["unverified"] += n_pts
                topo_tier_breakdown[topo]["unverified"] += n_pts

            topo_tier_breakdown[topo]["n_values"].add(n_val)

        # Global summary
        total_tiered = sum(tier_totals.values())
        print(f"\n  Quality Tier Distribution ({total_tiered} total points):")
        print(f"    verified:     {tier_totals['verified']:>4} ({tier_totals['verified']/max(total_tiered,1):.0%})")
        print(f"    approximate:  {tier_totals['approximate']:>4} ({tier_totals['approximate']/max(total_tiered,1):.0%})")
        print(f"    unverified:   {tier_totals['unverified']:>4} ({tier_totals['unverified']/max(total_tiered,1):.0%})")

        # Per-topology breakdown
        print(f"\n  Per-Topology Tier Breakdown:")
        for topo in sorted(topo_tier_breakdown.keys()):
            tb = topo_tier_breakdown[topo]
            n_v = tb["verified"]
            n_a = tb["approximate"]
            n_u = tb["unverified"]
            n_total = n_v + n_a + n_u
            n_vals = sorted(tb["n_values"])
            print(f"    {topo:12s}: {n_total:>4} pts "
                  f"(V={n_v} A={n_a} U={n_u}) "
                  f"N={n_vals}")

        # Training viability per topology using validate_training_dataset
        try:
            from qmbp_simulation.analysis.metrics import validate_training_dataset
            from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

            print(f"\n  Training Viability (multi-N model training):")
            for topo in sorted(topo_tier_breakdown.keys()):
                agg = MultiNAggregator(topology=topo, model="tfim_bond_resolved")
                agg.scan()
                viable, report = validate_training_dataset(agg._data_by_n)
                status = "✅ VIABLE" if viable else "❌ NOT VIABLE"
                reason = report.get("reason", "")[:60]
                n_usable = report.get("n_usable_points", 0)
                n_values = report.get("n_values_with_data", 0)
                print(f"    {topo:12s}: {status} "
                      f"({n_usable} pts, {n_values} N values) "
                      f"{reason}")
        except (ImportError, Exception) as e:
            print(f"    (validate_training_dataset unavailable: {e})")

    # ═══════════════════════════════════════════════════════════════
    # 6. Dashboard Cross-Validation (only with --validate-dashboard)
    # ═══════════════════════════════════════════════════════════════
    dashboard_path = DATA / "model_quality_dashboard.json"

    if args.validate_dashboard and dashboard_path.exists() and npz_dir.exists():
        from qmbp_simulation.analysis.metrics import (
            compute_h_frontier_from_npz, compute_theta_smoothness,
            DE_GAP_THRESHOLD,
        )

        print(f"\n{'=' * 60}")
        print("DASHBOARD CROSS-VALIDATION (--validate-dashboard)")
        print(f"{'=' * 60}")

        with open(dashboard_path) as f:
            cached_dashboard = json.load(f)
        cached_configs = {c["file"]: c for c in cached_dashboard.get("configs", [])}
        n_checked, n_mismatch = 0, 0

        for npz_file in sorted(npz_dir.glob("*.npz")):
            cached = cached_configs.get(npz_file.name)
            if not cached:
                continue
            n_checked += 1

            data = np.load(str(npz_file), allow_pickle=True)
            h_vals = data["h_values"]
            theta = data["theta_opt"]
            e_key = "e_vqe" if "e_vqe" in data else ("energies" if "energies" in data else None)
            if e_key is None or "e_exact" not in data:
                continue

            if "de_gaps" in data:
                de_gaps = data["de_gaps"]
            elif "gaps" in data:
                abs_err = np.abs(data[e_key] - data["e_exact"])
                de_gaps = abs_err / np.maximum(data["gaps"], 1e-10)
            else:
                continue

            recomputed_pass_rate = float((de_gaps < DE_GAP_THRESHOLD).mean())
            frontier_result = compute_h_frontier_from_npz(npz_file)
            recomputed_h_frontier = frontier_result.get("h_frontier")

            recomputed_smoothness = None
            if len(h_vals) > 1 and theta.ndim == 2:
                sort_idx = np.argsort(h_vals)
                recomputed_smoothness = float(
                    np.max(np.abs(np.diff(theta[sort_idx], axis=0)))
                )

            recomputed_divergence = None
            if cached.get("zoo_pass_rate") is not None:
                recomputed_divergence = abs(cached["zoo_pass_rate"] - recomputed_pass_rate)

            mismatches = []
            tol = 1e-4

            cached_pr = cached.get("pass_rate_5pct")
            if cached_pr is not None and abs(cached_pr - recomputed_pass_rate) > tol:
                mismatches.append({
                    "field": "pass_rate_5pct",
                    "dashboard": cached_pr,
                    "recomputed": recomputed_pass_rate,
                    "delta": recomputed_pass_rate - cached_pr,
                })

            cached_hf = cached.get("h_frontier")
            if cached_hf is not None and recomputed_h_frontier is not None:
                if abs(cached_hf - recomputed_h_frontier) > 0.01:
                    mismatches.append({
                        "field": "h_frontier",
                        "dashboard": cached_hf,
                        "recomputed": recomputed_h_frontier,
                        "delta": recomputed_h_frontier - cached_hf,
                    })

            cached_sm = cached.get("theta_smoothness")
            if cached_sm is not None and recomputed_smoothness is not None:
                if abs(cached_sm - recomputed_smoothness) > tol:
                    mismatches.append({
                        "field": "theta_smoothness",
                        "dashboard": cached_sm,
                        "recomputed": recomputed_smoothness,
                        "delta": recomputed_smoothness - cached_sm,
                    })

            cached_div = cached.get("zoo_vs_npz_divergence")
            if cached_div is not None and recomputed_divergence is not None:
                if abs(cached_div - recomputed_divergence) > tol:
                    mismatches.append({
                        "field": "zoo_vs_npz_divergence",
                        "dashboard": cached_div,
                        "recomputed": recomputed_divergence,
                        "delta": recomputed_divergence - cached_div,
                    })

            if mismatches:
                n_mismatch += 1
                topo = cached.get("topology", "?")
                n_q = cached.get("n_qubits", "?")
                print(f"\n  ❌ {npz_file.name} ({topo} N={n_q}): DASHBOARD STALE")
                print(f"     NPZ mtime: {cached.get('mtime', '?')}")
                print(f"     Dashboard generated: {cached_dashboard.get('generated_at', '?')}")
                for m in mismatches:
                    direction = "↑" if m["delta"] > 0 else "↓"
                    print(f"     • {m['field']}: "
                          f"dashboard={m['dashboard']:.6f} → "
                          f"recomputed={m['recomputed']:.6f} "
                          f"(Δ={m['delta']:+.6f} {direction})")
                    if m["field"] == "pass_rate_5pct" and abs(m["delta"]) > 0.1:
                        print(f"       ⚠️  >10% shift — likely new VQE data was added to NPZ")
                    elif m["field"] == "h_frontier" and abs(m["delta"]) > 0.5:
                        print(f"       ⚠️  Large frontier shift — regime boundary moved significantly")
                issues.append(
                    f"Dashboard stale for {npz_file.name}: "
                    f"{len(mismatches)} field(s) differ"
                )

        if n_mismatch == 0:
            print(f"  ✅ All {n_checked} configs match between dashboard and raw NPZ")
        else:
            print(f"\n  SUMMARY: {n_mismatch}/{n_checked} configs have stale dashboard data")
            print(f"  FIX: regenerate with:")
            print(f"    .venv/bin/python -c \"from qmbp_simulation.analysis.metrics import "
                  f"generate_model_quality_dashboard; generate_model_quality_dashboard()\"")

        # ── Bidirectional: NPZ on disk but NOT in dashboard ──────────────
        all_npz_names = {f.name for f in npz_dir.glob("*.npz")}
        dashboard_npz_names = set(cached_configs.keys())

        missing_from_dashboard = sorted(all_npz_names - dashboard_npz_names)
        if missing_from_dashboard:
            print(f"\n  ⚠️ NPZ files on disk NOT in dashboard ({len(missing_from_dashboard)}):")
            for fname in missing_from_dashboard:
                fpath = npz_dir / fname
                n_pts = len(np.load(str(fpath), allow_pickle=True)["h_values"])
                print(f"    + {fname} ({n_pts} pts) — dashboard doesn't know about this data")
            issues.append(
                f"Dashboard missing {len(missing_from_dashboard)} NPZ file(s): "
                f"{missing_from_dashboard[:3]}"
            )

        # ── Bidirectional: dashboard references NPZ that no longer exists ─
        orphan_in_dashboard = sorted(dashboard_npz_names - all_npz_names)
        if orphan_in_dashboard:
            print(f"\n  ❌ Dashboard references NPZ files that DON'T EXIST ({len(orphan_in_dashboard)}):")
            for fname in orphan_in_dashboard:
                entry = cached_configs[fname]
                print(f"    - {fname} ({entry.get('topology','?')} N={entry.get('n_qubits','?')}) "
                      f"— file deleted but dashboard not regenerated")
            issues.append(
                f"Dashboard has {len(orphan_in_dashboard)} orphan entries: "
                f"{orphan_in_dashboard[:3]}"
            )

        if not missing_from_dashboard and not orphan_in_dashboard and n_mismatch == 0:
            print(f"\n  ✅ Bidirectional check passed: dashboard ↔ disk are in sync")

    elif args.validate_dashboard and not dashboard_path.exists():
        print(f"\n  ⚠️ Cannot validate: dashboard not found at {dashboard_path}")
        print(f"     Run any experiment to auto-generate it.")

    # ═══════════════════════════════════════════════════════════════
    # 7. Model Quality Dashboard (read from cached, show analysis)
    # ═══════════════════════════════════════════════════════════════
    if dashboard_path.exists():
        print(f"\n{'=' * 60}")
        print("MODEL QUALITY DASHBOARD")
        print(f"{'=' * 60}")

        with open(dashboard_path) as f:
            dashboard = json.load(f)
        print(f"  Generated: {dashboard.get('generated_at', '?')}")
        print(f"  Configs: {dashboard.get('n_configs', 0)}")

        # Integrity summary
        integrity = dashboard.get("integrity", {})
        if integrity:
            n_nan = integrity.get("n_configs_with_nan_theta", 0)
            zoo_ok = integrity.get("zoo_integrity_ok")
            zoo_miss = integrity.get("zoo_n_missing", 0)
            print(f"  Integrity: NaN configs={n_nan}, zoo_ok={zoo_ok}, zoo_missing={zoo_miss}")
            if n_nan > 0:
                issues.append(f"Dashboard: {n_nan} configs have NaN in theta")
            if zoo_ok is False:
                issues.append("Dashboard: zoo integrity check FAILED (corrupted checkpoints)")

        # Actionable items from dashboard
        configs = dashboard.get("configs", [])
        stale = [c for c in configs if c.get("model_stale")]
        retrain = [c for c in configs if c.get("needs_retrain")]
        high_smooth = [c for c in configs
                       if c.get("theta_smoothness") and c["theta_smoothness"] > 0.5]
        high_div = [c for c in configs
                    if c.get("zoo_vs_npz_divergence") is not None
                    and c["zoo_vs_npz_divergence"] > 0.20]

        if stale:
            print(f"\n  ⚠️ Stale models ({len(stale)}):")
            for c in stale[:5]:
                npz_pass = c.get('pass_rate_dual_criterion', c.get('pass_rate_5pct', 0))
                print(f"    {c['topology']} N={c['n_qubits']}: "
                      f"NPZ pass_dual={npz_pass:.0%} > zoo pass={c['zoo_pass_rate']:.0%}")

        if retrain:
            print(f"\n  🔄 Need retrain ({len(retrain)}):")
            for c in retrain[:5]:
                print(f"    {c['topology']} N={c['n_qubits']}: "
                      f"{c['n_points']} NPZ pts available")

        if high_smooth:
            print(f"\n  ⚠️ High θ discontinuity ({len(high_smooth)}):")
            for c in high_smooth[:3]:
                print(f"    {c['topology']} N={c['n_qubits']}: "
                      f"smoothness={c['theta_smoothness']:.3f}")

        if high_div:
            print(f"\n  ⚠️ Zoo/NPZ divergence > 20% ({len(high_div)}):")
            for c in high_div[:5]:
                print(f"    {c['topology']} N={c['n_qubits']}: "
                      f"divergence={c['zoo_vs_npz_divergence']:.3f}")

        if not stale and not retrain and not high_smooth and not high_div:
            print("  ✅ No actionable items")

        # ── Training utility partition ─────────────────────────────────
        from qmbp_simulation.analysis.metrics import get_usable_training_configs
        partition = get_usable_training_configs(dashboard)

        useful = partition["useful"]
        insuff = partition["insufficient_signal"]
        not_useful = partition["not_useful"]

        print(f"\n  Training utility:")
        print(f"    Useful:               {len(useful):>3} configs  (MPNN can learn from these)")
        print(f"    Insufficient signal:  {len(insuff):>3} configs  (risk — too few good points)")
        print(f"    Not useful:           {len(not_useful):>3} configs  (exclude from training)")

        if not_useful:
            print(f"\n  ❌ NOT USEFUL for training (exclude or delete NPZ):")
            for c in not_useful:
                print(f"    {c['topology']:12s} N={c['n_qubits']:>2}: {c['training_utility_reason'][:80]}")

        if insuff:
            print(f"\n  ⚠️ INSUFFICIENT SIGNAL (train at your own risk):")
            for c in insuff:
                print(f"    {c['topology']:12s} N={c['n_qubits']:>2}: "
                      f"{c['n_points']}pts, dual={c.get('pass_rate_dual_criterion',0):.0%}")
        audit = dashboard.get("audit", {})
        if audit:
            # h_frontier anomalies
            frontier_issues = audit.get("h_frontier_anomalies", [])
            if frontier_issues:
                print(f"\n  ⚠️ h_frontier non-monotonic ({len(frontier_issues)}):")
                for a in frontier_issues:
                    print(f"    {a['message']}")
                    issues.append(f"h_frontier anomaly: {a['message'][:80]}")

            # Training/zoo incoherence
            incoherent = audit.get("training_zoo_incoherence", [])
            if incoherent:
                print(f"\n  ⚠️ Training/zoo incoherence ({len(incoherent)}):")
                for a in incoherent:
                    print(f"    {a['topology']} N={a['n_qubits']}: "
                          f"bad_ratio={a['bad_ratio']:.0%}, zoo_pass={a['zoo_pass_rate']:.0%}")
                    issues.append(f"Incoherence: {a['message'][:80]}")

            # Gap masking
            masked = audit.get("gap_masked_configs", [])
            if masked:
                print(f"\n  📊 Gap masking in {len(masked)} configs (pass@5% >> pass@dual):")
                for m in sorted(masked, key=lambda x: -x['gap_masked'])[:5]:
                    print(f"    {m['topology']} N={m['n_qubits']}: "
                          f"{m['pass_rate_5pct']:.0%} → {m['pass_rate_dual']:.0%} "
                          f"(masked={m['gap_masked']:.0%})")

            # Regressions
            regs = audit.get("pass_rate_regressions", [])
            if regs:
                print(f"\n  ❌ Pass rate regressions ({len(regs)}):")
                for r in regs:
                    print(f"    {r['message']}")
                    issues.append(f"Regression: {r['message'][:80]}")

            # H-range mismatches (Test M)
            h_mismatches = audit.get("h_range_mismatches", [])
            if h_mismatches:
                print(f"\n  ⚠️ H-range mismatches ({len(h_mismatches)} topologies):")
                for hm in h_mismatches:
                    print(f"    {hm['topology']}: overlap={hm['overlap_fraction']:.0%}")
                    for pair in hm.get("mismatch_pairs", [])[:3]:
                        print(f"      {pair}")
                    issues.append(
                        f"H-range mismatch: {hm['topology']} overlap={hm['overlap_fraction']:.0%}"
                    )

            if audit.get("n_issues", 0) == 0 and not stale and not retrain:
                print("  ✅ All automated audits passed")

        # Topology summary
        topo_sum = dashboard.get("topology_summary", {})
        if topo_sum:
            print(f"\n  Topology viability:")
            for topo, info in sorted(topo_sum.items()):
                n_max = info.get("n_max_viable")
                best = info.get("best_pass_rate_5pct", 0)
                n_max_str = f"N≤{n_max}" if n_max else "none"
                print(f"    {topo:12s}: viable {n_max_str}, best={best:.0%}")
    else:
        print(f"\n  [Dashboard] NOT FOUND — run any experiment to auto-generate")

    # ── Save snapshot for next regression detection run ─────────────────
    if dashboard_path.exists():
        import shutil
        prev_path = DATA / "model_quality_dashboard_prev.json"
        shutil.copy2(dashboard_path, prev_path)

    return len(issues)


if __name__ == "__main__":
    raise SystemExit(main())
