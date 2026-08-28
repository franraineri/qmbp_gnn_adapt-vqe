#!/usr/bin/env python3
"""Quick Health Check — Unified diagnostics for cross-N pipeline.

Combines multiple health checks into a single command:
1. Model zoo integrity (pass_rate, orphans, stale entries)
2. Training data quality (dual criterion, gap masking)
3. Ground truth cache coverage
4. ResultIndex traceability (runner_tag coverage)

Usage:
    .venv/bin/python scripts/maintenance/quick_health_check.py
    .venv/bin/python scripts/maintenance/quick_health_check.py --verbose
    .venv/bin/python scripts/maintenance/quick_health_check.py --topology ladder
    .venv/bin/python scripts/maintenance/quick_health_check.py --check-gap-masking
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Quick health check for cross-N pipeline")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--topology", type=str, default=None, help="Filter by topology")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--check-gap-masking",
        action="store_true",
        help="Include gap masking analysis (ΔE/gap<5%% but |ΔE|>0.10)",
    )
    return parser.parse_args()


def check_model_zoo() -> dict:
    """Check model zoo health."""
    from qmbp_simulation.predictors.model_zoo import _load_manifest, validate_zoo

    entries = _load_manifest()
    validation = validate_zoo()

    # Count by topology
    by_topo = defaultdict(list)
    for e in entries:
        by_topo[e.topology].append(e)

    # Find entries without runner_tag
    no_tag = [e for e in entries if not getattr(e, "runner_tag", None) or e.runner_tag == "XX"]

    # Find entries with pass_rate = 0
    unvalidated = [e for e in entries if e.pass_rate == 0]

    # Find multi-N models (n_qubits=0)
    multi_n = [e for e in entries if e.n_qubits == 0]

    return {
        "n_entries": len(entries),
        "n_valid": validation["n_valid"],
        "n_missing": validation["n_missing"],
        "n_corrupted": validation["n_corrupted"],
        "n_unvalidated": len(unvalidated),
        "n_no_tag": len(no_tag),
        "n_multi_n": len(multi_n),
        "topologies": list(by_topo.keys()),
        "unvalidated_files": [e.checkpoint_file for e in unvalidated[:5]],
        "no_tag_files": [e.checkpoint_file for e in no_tag[:5]],
    }


def check_training_data(topo_filter: str | None = None) -> dict:
    """Check training data quality using dual criterion."""
    from qmbp_simulation.analysis.metrics import (
        DE_GAP_THRESHOLD,
        classify_training_utility,
        identify_failures,
    )

    npz_dir = DATA / "multi_n_training"
    if not npz_dir.exists():
        return {"error": "NPZ directory not found"}

    useful = []
    insufficient = []
    not_useful = []
    total_points = 0
    total_good = 0

    for npz_file in sorted(npz_dir.glob("*.npz")):
        parts = npz_file.stem.split("_")
        n_idx = next((i for i, x in enumerate(parts) if x.startswith("N")), None)
        if n_idx is None:
            continue
        topo = "_".join(parts[:n_idx])
        if topo_filter and topo != topo_filter:
            continue

        try:
            data = np.load(str(npz_file), allow_pickle=True)
            h_vals = data["h_values"]
            n_pts = len(h_vals)
            total_points += n_pts

            # Compute dual criterion
            e_key = "e_vqe" if "e_vqe" in data else ("energies" if "energies" in data else None)
            if e_key is None or "e_exact" not in data:
                continue

            abs_err = np.abs(data[e_key] - data["e_exact"])
            gaps = data.get("gaps", data.get("de_gaps", np.ones_like(abs_err)))
            if "de_gaps" in data:
                de_gaps = data["de_gaps"]
            else:
                de_gaps = abs_err / np.maximum(gaps, 1e-10)

            # Use identify_failures() for canonical dual-criterion evaluation
            per_h_results = [
                {"de_gap": float(de_gaps[i]), "abs_error": float(abs_err[i])} for i in range(n_pts)
            ]
            n_failures = len(identify_failures(per_h_results))
            n_good = n_pts - n_failures
            total_good += n_good
            pass_dual = n_good / max(n_pts, 1)
            pass_5pct = float((de_gaps < DE_GAP_THRESHOLD).mean())

            category, reason = classify_training_utility(
                n_pts,
                pass_dual,
                pass_5pct,
            )

            entry = {"file": npz_file.name, "topo": topo, "n_pts": n_pts, "n_good": n_good}
            if category == "useful":
                useful.append(entry)
            elif category == "insufficient_signal":
                insufficient.append(entry)
            else:
                not_useful.append(entry)

        except Exception as e:
            not_useful.append({"file": npz_file.name, "error": str(e)})

    return {
        "total_files": len(useful) + len(insufficient) + len(not_useful),
        "total_points": total_points,
        "total_good": total_good,
        "n_useful": len(useful),
        "n_insufficient": len(insufficient),
        "n_not_useful": len(not_useful),
        "useful_files": [e["file"] for e in useful[:5]],
        "not_useful_files": [e["file"] for e in not_useful[:5]],
    }


def check_gt_cache() -> dict:
    """Check ground truth cache status."""
    gt_path = DATA / "ground_truth_cache.json"
    if not gt_path.exists():
        return {"n_entries": 0, "warning": "Cache file not found"}

    with open(gt_path) as f:
        raw = json.load(f)

    entries = raw.get("entries", raw) if isinstance(raw, dict) else {}

    # Parse keys to get topology distribution
    by_topo = defaultdict(int)
    by_n = defaultdict(int)
    for key in entries.keys():
        parts = key.split("|")
        if len(parts) >= 2:
            by_topo[parts[0]] += 1
            try:
                by_n[int(parts[1])] += 1
            except ValueError:
                pass

    return {
        "n_entries": len(entries),
        "by_topology": dict(by_topo),
        "by_n": dict(sorted(by_n.items())),
        "n_topologies": len(by_topo),
    }


def check_result_index() -> dict:
    """Check ResultIndex coverage and runner_tag distribution."""
    from qmbp_simulation.framework.result_index import ResultIndex

    try:
        index = ResultIndex()
        entries = index.valid_entries
    except Exception as e:
        return {"error": str(e)}

    # Count by runner_tag
    by_tag = defaultdict(int)
    by_date = defaultdict(int)
    no_tag = 0

    for e in entries:
        tag = e.get("runner_tag")
        if tag:
            by_tag[tag] += 1
        else:
            no_tag += 1

        date = e.get("date_tag")
        if date:
            by_date[date] += 1

    return {
        "n_valid_runs": len(entries),
        "n_without_tag": no_tag,
        "by_runner_tag": dict(by_tag),
        "by_date_tag": dict(sorted(by_date.items(), reverse=True)[:10]),
    }


def check_gap_masking(topo_filter: str | None = None) -> dict:
    """Analyze gap masking in NPZ training data.

    Gap masking: ΔE/gap < 5% (passes relative criterion) but |ΔE| > 0.10
    (fails absolute criterion). This inflates perceived quality at high h
    where the spectral gap is large.
    """
    npz_dir = DATA / "multi_n_training"
    if not npz_dir.exists():
        return {"error": "NPZ directory not found"}

    configs_with_masking = []
    total_masked = 0
    total_points = 0

    for npz_file in sorted(npz_dir.glob("*.npz")):
        if topo_filter and not npz_file.stem.startswith(topo_filter):
            continue

        try:
            data = np.load(str(npz_file), allow_pickle=True)
            h_vals = data["h_values"]
            e_vqe = data.get("e_vqe")
            e_exact = data.get("e_exact")
            gaps = data.get("gaps")

            if e_vqe is None or e_exact is None or gaps is None:
                continue

            abs_error = np.abs(e_vqe - e_exact)
            de_gap = abs_error / np.maximum(gaps, 1e-10)
            n_pts = len(h_vals)
            total_points += n_pts

            # Gap masking: passes ΔE/gap but fails |ΔE|
            gap_masked = (de_gap < 0.05) & (abs_error >= 0.10)
            n_masked = int(gap_masked.sum())
            total_masked += n_masked

            if n_masked > 0:
                masking_rate = n_masked / n_pts
                masked_h = h_vals[gap_masked].tolist()
                configs_with_masking.append(
                    {
                        "file": npz_file.name,
                        "n_masked": n_masked,
                        "n_points": n_pts,
                        "masking_rate": masking_rate,
                        "masked_h_range": (float(min(masked_h)), float(max(masked_h)))
                        if masked_h
                        else None,
                    }
                )
        except Exception:
            continue

    configs_with_masking.sort(key=lambda x: x["masking_rate"], reverse=True)

    return {
        "total_points": total_points,
        "total_masked": total_masked,
        "overall_masking_rate": total_masked / max(total_points, 1),
        "n_affected_configs": len(configs_with_masking),
        "top_offenders": configs_with_masking[:10],
    }


def print_report(
    zoo: dict,
    training: dict,
    gt: dict,
    index: dict,
    gap_masking: dict | None = None,
    verbose: bool = False,
):
    """Print formatted health report."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    print(f"╔{'═' * 68}╗")
    print(f"║  CROSS-N PIPELINE HEALTH CHECK — {now}  ║")
    print(f"╚{'═' * 68}╝")

    # Model Zoo
    print("\n📦 MODEL ZOO")
    print(
        f"   Entries: {zoo['n_entries']} | Valid: {zoo['n_valid']} | "
        f"Missing: {zoo['n_missing']} | Corrupted: {zoo['n_corrupted']}"
    )
    print(
        f"   Multi-N models: {zoo['n_multi_n']} | Unvalidated: {zoo['n_unvalidated']} | "
        f"No tag: {zoo['n_no_tag']}"
    )
    if zoo["unvalidated_files"] and verbose:
        print(f"   ⚠️  Unvalidated: {', '.join(zoo['unvalidated_files'][:3])}...")

    # Training Data
    print("\n📊 TRAINING DATA")
    print(
        f"   Files: {training['total_files']} | Points: {training['total_points']} | "
        f"Good: {training['total_good']}"
    )
    status = "✅" if training["n_not_useful"] == 0 else "⚠️"
    print(
        f"   {status} Useful: {training['n_useful']} | "
        f"Insufficient: {training['n_insufficient']} | "
        f"Not useful: {training['n_not_useful']}"
    )
    if training["not_useful_files"] and verbose:
        print(f"   ❌ Bad files: {', '.join(training['not_useful_files'][:3])}...")

    # Ground Truth Cache
    print("\n🎯 GROUND TRUTH CACHE")
    print(f"   Entries: {gt['n_entries']} | Topologies: {gt.get('n_topologies', 0)}")
    if gt.get("by_n") and verbose:
        n_dist = ", ".join(f"N={n}:{c}" for n, c in list(gt["by_n"].items())[:6])
        print(f"   Distribution: {n_dist}")

    # Result Index
    print("\n📋 RESULT INDEX")
    print(
        f"   Valid runs: {index.get('n_valid_runs', 0)} | Without tag: {index.get('n_without_tag', 0)}"
    )
    if index.get("by_runner_tag") and verbose:
        tags = ", ".join(f"{t}:{c}" for t, c in index["by_runner_tag"].items())
        print(f"   By runner: {tags}")

    # Gap Masking (optional)
    if gap_masking is not None and "error" not in gap_masking:
        print("\n🎭 GAP MASKING")
        rate = gap_masking["overall_masking_rate"]
        emoji = "✅" if rate < 0.02 else ("🟡" if rate < 0.10 else "🔴")
        print(
            f"   {emoji} Masked points: {gap_masking['total_masked']}/{gap_masking['total_points']} "
            f"({rate * 100:.1f}%)"
        )
        print(f"   Affected configs: {gap_masking['n_affected_configs']}")
        if gap_masking["top_offenders"] and verbose:
            print("   Top offenders:")
            for entry in gap_masking["top_offenders"][:5]:
                print(
                    f"     - {entry['file']}: {entry['masking_rate'] * 100:.0f}% "
                    f"({entry['n_masked']}/{entry['n_points']})"
                )

    # Summary
    print(f"\n{'─' * 70}")
    issues = []
    if zoo["n_corrupted"] > 0:
        issues.append(f"🔴 {zoo['n_corrupted']} corrupted checkpoints")
    if zoo["n_unvalidated"] > 0:
        issues.append(f"🟡 {zoo['n_unvalidated']} models never evaluated")
    if training["n_not_useful"] > 0:
        issues.append(f"🔴 {training['n_not_useful']} NPZ files not useful")
    if training["n_insufficient"] > 0:
        issues.append(f"🟡 {training['n_insufficient']} NPZ files need more data")
    if gap_masking is not None and gap_masking.get("overall_masking_rate", 0) >= 0.10:
        issues.append(
            f"🔴 Gap masking at {gap_masking['overall_masking_rate'] * 100:.0f}% "
            f"({gap_masking['n_affected_configs']} configs affected)"
        )
    elif gap_masking is not None and gap_masking.get("overall_masking_rate", 0) >= 0.02:
        issues.append(
            f"🟡 Gap masking at {gap_masking['overall_masking_rate'] * 100:.1f}% "
            f"({gap_masking['n_affected_configs']} configs)"
        )

    if issues:
        print("ISSUES:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("✅ All checks passed!")


def main():
    args = parse_args()

    zoo = check_model_zoo()
    training = check_training_data(args.topology)
    gt = check_gt_cache()
    index = check_result_index()
    gap_masking = check_gap_masking(args.topology) if args.check_gap_masking else None

    if args.json:
        import json as _json

        report = {
            "zoo": zoo,
            "training": training,
            "ground_truth": gt,
            "result_index": index,
        }
        if gap_masking is not None:
            report["gap_masking"] = gap_masking
        print(_json.dumps(report, indent=2))
    else:
        print_report(zoo, training, gt, index, gap_masking=gap_masking, verbose=args.verbose)


if __name__ == "__main__":
    main()
