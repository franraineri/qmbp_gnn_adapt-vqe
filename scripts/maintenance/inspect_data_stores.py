#!/usr/bin/env python3
"""Inspect data/ directory: cache sizes, NPZ quality, organization.

Reusable diagnostic for verifying data integrity before running experiments.
Usage:
    .venv/bin/python scripts/maintenance/inspect_data_stores.py
    .venv/bin/python scripts/maintenance/inspect_data_stores.py --validate-dashboard
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def main():
    parser = argparse.ArgumentParser(description="Inspect data stores")
    parser.add_argument(
        "--validate-dashboard",
        action="store_true",
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
        print("  Configs with data:")
        for k, hs in sorted(h_values_per_config.items()):
            print(f"    {k}: {len(hs)} h-points, h=[{min(hs):.2f}, {max(hs):.2f}]")
        n_bad = sum(1 for v in gt.values() if not v.get("gap") or v["gap"] == 0)
        if n_bad:
            issues.append(f"GT cache: {n_bad} entries with missing/zero gap")

        # Check for stale floor gaps (N>18 with gap≈2π/N)
        n_stale_floor = 0
        for key, val in gt.items():
            parts = key.split("|")
            if len(parts) < 4:
                continue
            try:
                n = int(parts[1])
            except ValueError:
                continue
            if n > 18:
                gap = val.get("gap", 0)
                floor = 2 * np.pi / n
                if abs(gap - floor) < 1e-4:
                    n_stale_floor += 1
        if n_stale_floor:
            print(f"  ⚠️ Stale floor gaps (N>18): {n_stale_floor} entries (will auto-recompute)")
            issues.append(f"GT cache: {n_stale_floor} entries with stale floor gap (N>18)")
    else:
        print("\n[GroundTruthCache] NOT FOUND")
        issues.append("GT cache file missing")

    # 1b. GT ↔ NPZ Coherence Check
    try:
        from qmbp_simulation.analysis.metrics import validate_gt_npz_coherence

        coherence = validate_gt_npz_coherence(fix=False)
        if coherence["n_files_checked"] > 0:
            if coherence["n_files_with_issues"] == 0:
                print(f"\n  [GT↔NPZ Coherence] ✅ {coherence['summary']}")
            else:
                print(f"\n  [GT↔NPZ Coherence] {coherence['summary']}")
                for iss in coherence["issues"][:5]:
                    impact = f" (pass_rate Δ={iss['pass_rate_impact']:+.0%})" if iss["pass_rate_impact"] else ""
                    print(
                        f"    ⚠️ {iss['file']}: {iss['n_mismatched']}/{iss['n_total']} points "
                        f"stale (max_delta={iss['max_delta']:.2e}){impact}"
                    )
                issues.append(
                    f"GT↔NPZ: {coherence['n_files_with_issues']} files with stale e_exact "
                    f"(max_delta={coherence['max_delta']:.2e})"
                )
    except Exception as e:
        print(f"\n  [GT↔NPZ Coherence] ⚠️ Check failed: {e}")

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
        print("  Top configs (by entry count):")
        for p, count in sorted(prefixes.items(), key=lambda x: -x[1])[:8]:
            print(f"    {p}: {count} entries")
    else:
        print("\n[EvalCache] NOT FOUND")

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
            print(
                f"    {n_points} pts, {n_params} params, h=[{h_vals.min():.2f},{h_vals.max():.2f}]"
            )
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
                quality_score = (n_verified * 1.0 + n_approx * 0.7 + n_unverified * 0.5) / max(
                    n_points, 1
                )
                status = "✅" if n_verified > n_points * 0.5 else ("⚠️" if n_verified > 0 else "❓")
                print(
                    f"    Quality tiers: ✅{n_verified} ⚠️{n_approx} ❓{n_unverified} (score={quality_score:.2f}) {status}"
                )
            else:
                print("    Quality tiers: 📜 legacy NPZ (no quality_tier field)")

            # Check GT coverage
            parts = npz_file.stem.split("_")
            n_idx = next((i for i, p in enumerate(parts) if p.startswith("N")), None)
            if n_idx and gt:
                topo = "_".join(parts[:n_idx])
                n_val = int(parts[n_idx][1:])
                missing_gt = 0
                for h in h_vals:
                    key = f"{topo}|{n_val}|tfim_bond_resolved|{float(h):.2f}"
                    if key not in gt:
                        missing_gt += 1
                if missing_gt:
                    issues.append(
                        f"NPZ {npz_file.name}: {missing_gt}/{n_points} h-points "
                        f"have no GT cache entry"
                    )
    else:
        print("\n[NPZ Training Data] NOT FOUND")

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

                # Check gap reliability (floor vs analytical/exact)
                gap_info = ""
                if "gaps" in data:
                    gaps = data["gaps"]
                    # Parse N from filename
                    n_str = npz_file.stem.split("_N")[1].split("_")[0]
                    n_val = int(n_str)
                    floor = 2 * np.pi / n_val
                    n_floor = int(np.sum(np.abs(gaps - floor) < 1e-3))
                    if n_floor > 0:
                        gap_info = f" ⚠️{n_floor}/{n_pts} floor-gaps"

                print(
                    f"  {npz_file.name}: {n_pts} pts, "
                    f"h=[{h_vals.min():.2f},{h_vals.max():.2f}]{tier_str}{gap_info}"
                )

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
            pr_by_n = entry.get("pass_rate_by_n", {})
            pr_by_n_str = ""
            if pr_by_n:
                pr_by_n_str = " | by_N: " + ", ".join(
                    f"N{k}={v:.0%}" for k, v in sorted(pr_by_n.items(), key=lambda x: int(x[0]))
                )
            print(f"  {name}: {topo} N={n}, pass_dual={pr:.0%}, pts={ntp}{pr_by_n_str}")

        # Use validate_zoo() for SHA256 integrity + missing file detection
        try:
            from qmbp_simulation.predictors.model_zoo import validate_zoo

            zoo_report = validate_zoo()
            print(
                f"  Integrity: {zoo_report['n_valid']} valid, "
                f"{zoo_report['n_missing']} missing, "
                f"{zoo_report['n_corrupted']} corrupted"
            )
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
                    print(f"    [⚠️ ORPHAN] {ckpt.name} ({ckpt.stat().st_size / 1024:.0f}KB)")
                    issues.append(f"Zoo orphan: {ckpt.name} not in manifest")

            # Report safety backups (informational, not an issue)
            best_dir = ckpt_dir / "_best"
            if best_dir.exists():
                n_best = len(list(best_dir.glob("*.pt")))
                if n_best > 0:
                    print(f"    [ℹ️ BACKUPS] {n_best} models in _best/ (anti-regression safety net)")
    else:
        print("\n[Model Zoo] NOT FOUND")

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
    # 4b. Recent Extrapolation Runs (last 10, with checkpoint info)
    # ═══════════════════════════════════════════════════════════════
    extrap_results_dir = ROOT / "results" / "experiments" / "exp_large_n_extrap"
    if extrap_results_dir.exists():
        recent_runs = sorted(extrap_results_dir.glob("run_*.json"))[-10:]
        if recent_runs:
            print(f"\n[Recent Extrapolation Runs] (last {len(recent_runs)})")
            try:
                from qmbp_simulation.framework.result_io import extract_checkpoint_used
            except ImportError:
                extract_checkpoint_used = None

            for rfile in recent_runs:
                try:
                    with open(rfile) as f:
                        rdata = json.load(f)
                    cfg = rdata.get("config", {})
                    topo = cfg.get("topology", "?")
                    target_n = cfg.get("target_n", [])
                    passed = rdata.get("summary", {}).get("all_passed", False)
                    elapsed = rdata.get("elapsed_s", 0)
                    ckpt = "?"
                    if extract_checkpoint_used:
                        ckpt = extract_checkpoint_used(rdata)
                        if len(ckpt) > 50:
                            ckpt = "..." + ckpt[-47:]
                    status = "✅" if passed else "❌"
                    print(f"  {status} {rfile.name}: {topo} N={target_n} ({elapsed:.0f}s) → {ckpt}")
                except Exception:
                    print(f"  ❌ {rfile.name}: PARSE ERROR")

    # ═══════════════════════════════════════════════════════════════
    # 5. Training Data Quality Analysis + H-Frontier per NPZ
    # ═══════════════════════════════════════════════════════════════
    if npz_dir.exists():
        from qmbp_simulation.analysis.metrics import (
            DE_GAP_THRESHOLD,
            compute_h_frontier_from_npz,
            identify_failures,
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
                {"de_gap": float(de_gaps[i]), "abs_error": float(abs_err[i])} for i in range(n_pts)
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
            print(
                f"    θ max jump: {max_jump:.3f} {'(smooth)' if max_jump < 0.5 else '⚠️ discontinuous'}"
            )

            # Metric reliability warnings (variational violations, outliers, gap issues)
            from qmbp_simulation.analysis.evaluation_report import validate_metrics

            per_h_rich = [
                {
                    "h": float(h_vals[i]),
                    "e_pred": float(e_vqe[i]),
                    "e_exact": float(e_exact[i]),
                    "gap": float(gaps[i]),
                    "de_gap": float(de_gaps[i]),
                    "abs_error": float(abs_err[i]),
                }
                for i in range(n_pts)
            ]
            metric_warnings = validate_metrics(per_h_rich)
            for w in metric_warnings:
                print(f"    {w}")
                issues.append(f"NPZ {npz_file.name}: {w}")

            # H-frontier (empirical boundary where ΔE/gap crosses 5%)
            frontier_result = compute_h_frontier_from_npz(npz_file)
            h_front = frontier_result.get("h_frontier")
            if h_front is not None:
                print(f"    h_frontier: {h_front:.3f} (below this → pipeline fails)")
            else:
                print("    h_frontier: N/A (all pass or all fail)")

            # Flag specific problematic points
            if n_bad > 0:
                bad_idx = np.where(bad)[0]
                for bi in bad_idx[:3]:
                    print(
                        f"    ❌ h={h_vals[bi]:.2f}: |ΔE|={abs_err[bi]:.3f} ΔE/gap={de_gaps[bi]:.3f}"
                    )

        print(f"\n  {'─' * 40}")
        print(f"  TOTAL: {total_pts} points")
        print(
            f"    Good (ΔE/gap<5% & |ΔE|<0.10): {total_good} ({total_good / max(total_pts, 1):.0%})"
        )
        print(f"    Marginal (5-10%): {total_marginal} ({total_marginal / max(total_pts, 1):.0%})")
        print(f"    Bad (>10% or |ΔE|>0.10): {total_bad} ({total_bad / max(total_pts, 1):.0%})")

        is_valid = total_bad <= total_pts * 0.1
        print(f"\n  Training dataset {'VALID ✅' if is_valid else 'NEEDS CLEANING ❌'}")
        if not is_valid:
            print("    > 10% bad points — consider re-running with higher maxiter")

    # ═══════════════════════════════════════════════════════════════
    # 5b. Quality Tier Breakdown + Training Viability Assessment
    # ═══════════════════════════════════════════════════════════════
    if npz_dir.exists():
        from collections import defaultdict as _defaultdict

        print(f"\n{'=' * 60}")
        print("QUALITY TIERS & TRAINING VIABILITY")
        print(f"{'=' * 60}")

        tier_totals = {"verified": 0, "approximate": 0, "unverified": 0}
        topo_tier_breakdown = _defaultdict(
            lambda: {"verified": 0, "approximate": 0, "unverified": 0, "n_values": set()}
        )

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
        print(
            f"    verified:     {tier_totals['verified']:>4} ({tier_totals['verified'] / max(total_tiered, 1):.0%})"
        )
        print(
            f"    approximate:  {tier_totals['approximate']:>4} ({tier_totals['approximate'] / max(total_tiered, 1):.0%})"
        )
        print(
            f"    unverified:   {tier_totals['unverified']:>4} ({tier_totals['unverified'] / max(total_tiered, 1):.0%})"
        )

        # Per-topology breakdown
        print("\n  Per-Topology Tier Breakdown:")
        for topo in sorted(topo_tier_breakdown.keys()):
            tb = topo_tier_breakdown[topo]
            n_v = tb["verified"]
            n_a = tb["approximate"]
            n_u = tb["unverified"]
            n_total = n_v + n_a + n_u
            n_vals = sorted(tb["n_values"])
            print(f"    {topo:12s}: {n_total:>4} pts (V={n_v} A={n_a} U={n_u}) N={n_vals}")

        # Training viability per topology using validate_training_dataset
        try:
            from qmbp_simulation.analysis.metrics import validate_training_dataset
            from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

            print("\n  Training Viability (multi-N model training):")
            for topo in sorted(topo_tier_breakdown.keys()):
                agg = MultiNAggregator(topology=topo, model="tfim_bond_resolved")
                agg.scan()
                viable, report = validate_training_dataset(agg._data_by_n)
                status = "✅ VIABLE" if viable else "❌ NOT VIABLE"
                reason = report.get("reason", "")[:60]
                n_usable = report.get("n_usable_points", 0)
                n_values = report.get("n_values_with_data", 0)
                print(f"    {topo:12s}: {status} ({n_usable} pts, {n_values} N values) {reason}")
        except (ImportError, Exception) as e:
            print(f"    (validate_training_dataset unavailable: {e})")

    # ═══════════════════════════════════════════════════════════════
    # 6. Dashboard Cross-Validation (only with --validate-dashboard)
    # ═══════════════════════════════════════════════════════════════
    dashboard_path = DATA / "model_quality_dashboard.json"

    if args.validate_dashboard and dashboard_path.exists() and npz_dir.exists():
        from qmbp_simulation.analysis.metrics import (
            DE_GAP_THRESHOLD,
            compute_h_frontier_from_npz,
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
                recomputed_smoothness = float(np.max(np.abs(np.diff(theta[sort_idx], axis=0))))

            recomputed_divergence = None
            if cached.get("zoo_pass_rate") is not None:
                recomputed_divergence = abs(cached["zoo_pass_rate"] - recomputed_pass_rate)

            mismatches = []
            tol = 1e-4

            cached_pr = cached.get("pass_rate_5pct")
            if cached_pr is not None and abs(cached_pr - recomputed_pass_rate) > tol:
                mismatches.append(
                    {
                        "field": "pass_rate_5pct",
                        "dashboard": cached_pr,
                        "recomputed": recomputed_pass_rate,
                        "delta": recomputed_pass_rate - cached_pr,
                    }
                )

            cached_hf = cached.get("h_frontier")
            if cached_hf is not None and recomputed_h_frontier is not None:
                if abs(cached_hf - recomputed_h_frontier) > 0.01:
                    mismatches.append(
                        {
                            "field": "h_frontier",
                            "dashboard": cached_hf,
                            "recomputed": recomputed_h_frontier,
                            "delta": recomputed_h_frontier - cached_hf,
                        }
                    )

            cached_sm = cached.get("theta_smoothness")
            if cached_sm is not None and recomputed_smoothness is not None:
                if abs(cached_sm - recomputed_smoothness) > tol:
                    mismatches.append(
                        {
                            "field": "theta_smoothness",
                            "dashboard": cached_sm,
                            "recomputed": recomputed_smoothness,
                            "delta": recomputed_smoothness - cached_sm,
                        }
                    )

            cached_div = cached.get("zoo_vs_npz_divergence")
            if cached_div is not None and recomputed_divergence is not None:
                if abs(cached_div - recomputed_divergence) > tol:
                    mismatches.append(
                        {
                            "field": "zoo_vs_npz_divergence",
                            "dashboard": cached_div,
                            "recomputed": recomputed_divergence,
                            "delta": recomputed_divergence - cached_div,
                        }
                    )

            if mismatches:
                n_mismatch += 1
                topo = cached.get("topology", "?")
                n_q = cached.get("n_qubits", "?")
                print(f"\n  ❌ {npz_file.name} ({topo} N={n_q}): DASHBOARD STALE")
                print(f"     NPZ mtime: {cached.get('mtime', '?')}")
                print(f"     Dashboard generated: {cached_dashboard.get('generated_at', '?')}")
                for m in mismatches:
                    direction = "↑" if m["delta"] > 0 else "↓"
                    print(
                        f"     • {m['field']}: "
                        f"dashboard={m['dashboard']:.6f} → "
                        f"recomputed={m['recomputed']:.6f} "
                        f"(Δ={m['delta']:+.6f} {direction})"
                    )
                    if m["field"] == "pass_rate_5pct" and abs(m["delta"]) > 0.1:
                        print("       ⚠️  >10% shift — likely new VQE data was added to NPZ")
                    elif m["field"] == "h_frontier" and abs(m["delta"]) > 0.5:
                        print(
                            "       ⚠️  Large frontier shift — regime boundary moved significantly"
                        )
                issues.append(
                    f"Dashboard stale for {npz_file.name}: {len(mismatches)} field(s) differ"
                )

        if n_mismatch == 0:
            print(f"  ✅ All {n_checked} configs match between dashboard and raw NPZ")
        else:
            print(f"\n  SUMMARY: {n_mismatch}/{n_checked} configs have stale dashboard data")
            print("  FIX: regenerate with:")
            print(
                '    .venv/bin/python -c "from qmbp_simulation.analysis.metrics import '
                'generate_model_quality_dashboard; generate_model_quality_dashboard()"'
            )

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
            print(
                f"\n  ❌ Dashboard references NPZ files that DON'T EXIST ({len(orphan_in_dashboard)}):"
            )
            for fname in orphan_in_dashboard:
                entry = cached_configs[fname]
                print(
                    f"    - {fname} ({entry.get('topology', '?')} N={entry.get('n_qubits', '?')}) "
                    f"— file deleted but dashboard not regenerated"
                )
            issues.append(
                f"Dashboard has {len(orphan_in_dashboard)} orphan entries: "
                f"{orphan_in_dashboard[:3]}"
            )

        if not missing_from_dashboard and not orphan_in_dashboard and n_mismatch == 0:
            print("\n  ✅ Bidirectional check passed: dashboard ↔ disk are in sync")

    elif args.validate_dashboard and not dashboard_path.exists():
        print(f"\n  ⚠️ Cannot validate: dashboard not found at {dashboard_path}")
        print("     Run any experiment to auto-generate it.")

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
        high_smooth = [
            c for c in configs if c.get("theta_smoothness") and c["theta_smoothness"] > 0.5
        ]
        high_div = [
            c
            for c in configs
            if c.get("zoo_vs_npz_divergence") is not None and c["zoo_vs_npz_divergence"] > 0.20
        ]

        if stale:
            print(f"\n  ⚠️ Stale models ({len(stale)}):")
            for c in stale[:5]:
                npz_pass = c.get("pass_rate_dual_criterion", c.get("pass_rate_5pct", 0))
                print(
                    f"    {c['topology']} N={c['n_qubits']}: "
                    f"NPZ pass_dual={npz_pass:.0%} > zoo pass={c['zoo_pass_rate']:.0%}"
                )

        if retrain:
            print(f"\n  🔄 Need retrain ({len(retrain)}):")
            for c in retrain[:5]:
                print(f"    {c['topology']} N={c['n_qubits']}: {c['n_points']} NPZ pts available")

        if high_smooth:
            print(f"\n  ⚠️ High θ discontinuity ({len(high_smooth)}):")
            for c in high_smooth[:3]:
                print(
                    f"    {c['topology']} N={c['n_qubits']}: smoothness={c['theta_smoothness']:.3f}"
                )

        if high_div:
            print(f"\n  ⚠️ Zoo/NPZ divergence > 20% ({len(high_div)}):")
            for c in high_div[:5]:
                print(
                    f"    {c['topology']} N={c['n_qubits']}: "
                    f"divergence={c['zoo_vs_npz_divergence']:.3f}"
                )

        if not stale and not retrain and not high_smooth and not high_div:
            print("  ✅ No actionable items")

        # ── Training utility partition ─────────────────────────────────
        from qmbp_simulation.analysis.metrics import get_usable_training_configs

        partition = get_usable_training_configs(dashboard)

        useful = partition["useful"]
        insuff = partition["insufficient_signal"]
        not_useful = partition["not_useful"]

        print("\n  Training utility:")
        print(f"    Useful:               {len(useful):>3} configs  (MPNN can learn from these)")
        print(f"    Insufficient signal:  {len(insuff):>3} configs  (risk — too few good points)")
        print(f"    Not useful:           {len(not_useful):>3} configs  (exclude from training)")

        if not_useful:
            print("\n  ❌ NOT USEFUL for training (excluded from MultiNAggregator):")
            for c in not_useful:
                print(
                    f"    {c['topology']:12s} N={c['n_qubits']:>2}: {c['training_utility_reason'][:80]}"
                )

        if insuff:
            print("\n  ⚠️ INSUFFICIENT SIGNAL (train at your own risk):")
            for c in insuff:
                print(
                    f"    {c['topology']:12s} N={c['n_qubits']:>2}: "
                    f"{c['n_points']}pts, dual={c.get('pass_rate_dual_criterion', 0):.0%}"
                )

        # ── Exclusion registry status ────────────────────────────────────
        try:
            from qmbp_simulation.analysis.metrics import load_training_exclusions

            registry = load_training_exclusions()
            excluded = registry.get("excluded", [])
            if excluded:
                print(f"\n  🚫 Exclusion Registry ({len(excluded)} files):")
                for e in excluded:
                    print(
                        f"    {e['topology']:12s} N={e['n_qubits']:>2}: "
                        f"{e['file']} ({e['method']}, dual={e['pass_rate_dual']:.0%})"
                    )
        except Exception:
            pass
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
                    print(
                        f"    {a['topology']} N={a['n_qubits']}: "
                        f"bad_ratio={a['bad_ratio']:.0%}, zoo_pass={a['zoo_pass_rate']:.0%}"
                    )
                    issues.append(f"Incoherence: {a['message'][:80]}")

            # Gap masking
            masked = audit.get("gap_masked_configs", [])
            if masked:
                print(f"\n  📊 Gap masking in {len(masked)} configs (pass@5% >> pass@dual):")
                for m in sorted(masked, key=lambda x: -x["gap_masked"])[:5]:
                    print(
                        f"    {m['topology']} N={m['n_qubits']}: "
                        f"{m['pass_rate_5pct']:.0%} → {m['pass_rate_dual']:.0%} "
                        f"(masked={m['gap_masked']:.0%})"
                    )

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
            print("\n  Topology viability:")
            for topo, info in sorted(topo_sum.items()):
                n_max = info.get("n_max_viable")
                best = info.get("best_pass_rate_5pct", 0)
                n_max_str = f"N≤{n_max}" if n_max else "none"
                print(f"    {topo:12s}: viable {n_max_str}, best={best:.0%}")

        # ── Zoo training quality scores ────────────────────────────────────
        try:
            from qmbp_simulation.predictors.model_zoo import compute_training_quality_score

            print("\n  Zoo training quality scores (multi-N, p=1):")
            for topo in sorted(topo_sum.keys()) if topo_sum else []:
                score = compute_training_quality_score(topo, n_qubits=0, p_layers=1)
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                print(f"    {topo:12s}: {bar} {score:.3f}")
        except Exception as e:
            print(f"\n  Zoo quality scores: skipped ({e})")

        # ── MT vs ST Comparison (from dashboard or live) ─────────────────
        mt_vs_st = dashboard.get("mt_vs_st_comparison")
        if mt_vs_st:
            print(f"\n  {'─' * 50}")
            print("  MT vs ST Model Comparison (embedded in dashboard)")
            print(f"  {'─' * 50}")
            g = mt_vs_st.get("global", {})
            mt_w = g.get("mt_wins", 0)
            st_w = g.get("st_wins", 0)
            ties = g.get("ties", 0)
            total = g.get("total", 0)
            mt_avg = g.get("mt_avg_pass_rate", 0)
            st_avg = g.get("st_avg_pass_rate", 0)
            winner = "MT" if mt_avg > st_avg + 0.01 else ("ST" if st_avg > mt_avg + 0.01 else "Tie")
            winner_icon = "🟢" if winner == "MT" else ("🔴" if winner == "ST" else "⚪")
            print(f"    Score: MT {mt_w} — ST {st_w} — Ties {ties} (total: {total})")
            print(f"    MT avg pass: {mt_avg:.0%} | ST avg pass: {st_avg:.0%}")
            print(f"    Overall: {winner_icon} {winner}")

            per_topology = mt_vs_st.get("per_topology", {})
            if per_topology:
                print(f"\n    {'Topology':<14} {'MT%':>5} {'ST%':>5} {'Win':>4} {'Δ':>6}")
                for topo_name in sorted(per_topology.keys()):
                    info = per_topology[topo_name]
                    icon = "🟢" if info["winner"] == "MT" else (
                        "🔴" if info["winner"] == "ST" else "⚪"
                    )
                    print(
                        f"    {topo_name:<14} "
                        f"{info['mt_avg_pass_rate']:>4.0%} "
                        f"{info['st_avg_pass_rate']:>5.0%} "
                        f"{icon}{info['winner']:<3} "
                        f"{info['delta']:>+5.0%}"
                    )
        else:
            # Fall back to live computation
            try:
                from qmbp_simulation.analysis.evaluation_report import generate_mt_vs_st_table

                _, summary = generate_mt_vs_st_table(latest_only=True)
                if summary.get("total", 0) > 0:
                    print(f"\n  {'─' * 50}")
                    print("  MT vs ST Model Comparison (live from comparison JSONs)")
                    print(f"  {'─' * 50}")
                    mt_w = summary["mt_wins"]
                    st_w = summary["st_wins"]
                    ties = summary["ties"]
                    mt_avg = summary.get("mt_avg_pass_rate", 0)
                    st_avg = summary.get("st_avg_pass_rate", 0)
                    winner = "MT" if mt_avg > st_avg + 0.01 else (
                        "ST" if st_avg > mt_avg + 0.01 else "Tie"
                    )
                    winner_icon = "🟢" if winner == "MT" else (
                        "🔴" if winner == "ST" else "⚪"
                    )
                    print(f"    Score: MT {mt_w} — ST {st_w} — Ties {ties}")
                    print(f"    MT avg: {mt_avg:.0%} | ST avg: {st_avg:.0%} → {winner_icon} {winner}")
                    per_topology = summary.get("per_topology", {})
                    for topo_name in sorted(per_topology.keys()):
                        info = per_topology[topo_name]
                        icon = "🟢" if info["winner"] == "MT" else (
                            "🔴" if info["winner"] == "ST" else "⚪"
                        )
                        print(
                            f"    {topo_name:<14} "
                            f"MT={info['mt_avg_pass_rate']:.0%} "
                            f"ST={info['st_avg_pass_rate']:.0%} "
                            f"{icon} {info['winner']}"
                        )
            except Exception as _mt_e:
                pass  # No comparison data available
    else:
        print("\n  [Dashboard] NOT FOUND — run any experiment to auto-generate")

    # ── Save snapshot for next regression detection run ─────────────────
    if dashboard_path.exists():
        import shutil

        prev_path = DATA / "model_quality_dashboard_prev.json"
        shutil.copy2(dashboard_path, prev_path)

    return len(issues)


if __name__ == "__main__":
    raise SystemExit(main())
