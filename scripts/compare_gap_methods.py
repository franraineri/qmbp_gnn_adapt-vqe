#!/usr/bin/env python3
"""Compare h_boundary and pass rates between pre-fix (floor gap) and post-fix (eigsh gap) runs.

After running new experiments with the gap fix active, this script compares
the old vs new metrics for the same (model, topology, N, p) configs.

New runs will have gap_method='eigsh_fallback' in their section_1 data.
Old runs will NOT have gap_method (or it will be missing).

Usage:
    python scripts/compare_gap_methods.py
    python scripts/compare_gap_methods.py --model tfim --topology heavy_hex
    python scripts/compare_gap_methods.py --json results/h2_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qmbp_simulation.utils.helpers import json_dump

DE_GAP_THRESHOLD = 0.05


NOISELESS_DIRS = [
    "exp_noiseless",
    "exp_noiseless_tfim_v2",
    "exp_noiseless_tfim_v3",
    "exp_noiseless_tfim_4",
    "exp_noiseless_tfim_longitudinal",
    "exp_noiseless_tfim_longitudinal_v2",
    "exp_noiseless_tfim_longitudinal_v3",
    "exp_noiseless_tfim_longitudinal_4",
]


def scan_runs(base: Path, model_filter: str | None, topo_filter: str | None) -> dict:
    """Scan runs and classify as pre-fix or post-fix based on gap_method field.

    Uses ResultIndex for fast pre-filtering, then loads only matching files.
    """
    # key = (model, topology, n_qubits, p_layers)
    # value = {"pre_fix": [run_info], "post_fix": [run_info]}
    by_config: dict[str, dict[str, list]] = defaultdict(lambda: {"pre_fix": [], "post_fix": []})

    # Use ResultIndex to get candidate files (fast)
    try:
        from qmbp_simulation.framework.result_index import ResultIndex

        index = ResultIndex(base)
        entries = index.query(model=model_filter, topology=topo_filter)
        # Only keep runs with ≥4 sections (have deploy data) and non-chain
        candidate_files = []
        for entry in entries:
            if entry.get("n_sections", 0) < 4:
                continue
            topo = entry.get("topology", "")
            if topo == "chain_1d":
                continue
            fpath = base / entry["_file"]
            if fpath.exists():
                candidate_files.append(fpath)
    except Exception:
        # Fallback: glob scan
        candidate_files = []
        for d in NOISELESS_DIRS:
            exp_dir = base / d
            if not exp_dir.exists():
                continue
            for f in sorted(exp_dir.rglob("run_*.json")):
                candidate_files.append(f)

    for f in candidate_files:
        try:
            with open(f) as fh:
                data = json.load(fh)
        except Exception:
            continue

        cfg = data.get("config", {})
        system = cfg.get("system", {})
        model = system.get("model", "")
        topos = system.get("topologies", [])
        topology = topos[0] if topos else ""
        n_qubits = system.get("n_qubits", 0)
        p_layers = system.get("p_layers", 0)

        if not model or not topology or not n_qubits:
            continue
        if topology == "chain_1d":
            continue  # chain_1d was never affected by floor bug
        if model_filter and model_filter.lower() not in model.lower():
            continue
        if topo_filter and topo_filter.lower() not in topology.lower():
            continue

        # Check section_4 deploy data
        s4 = data.get("results", {}).get("section_4", {})
        if not s4 or not s4.get("data", {}).get("per_point"):
            continue

        s4_data = s4["data"]
        per_point = s4_data.get("per_point", [])

        # Detect gap_method from section_1
        s1_data = data.get("results", {}).get("section_1", {}).get("data", {})
        topo_s1 = s1_data.get("topologies", {}).get(topology, {})
        points_s1 = topo_s1.get("points", [])
        has_eigsh = any(p.get("gap_method") == "eigsh_fallback" for p in points_s1)

        # Compute h_boundary from deploy
        passing = [p for p in per_point if p.get("de_gap", 1) < DE_GAP_THRESHOLD]
        h_boundary = min(p.get("h_test", p.get("h", 99)) for p in passing) if passing else None
        n_pass = len(passing)
        n_total = len(per_point)

        run_info = {
            "file": str(f.relative_to(ROOT)),
            "pass_rate": n_pass / n_total if n_total > 0 else 0,
            "h_boundary": h_boundary,
            "n_points": n_total,
            "timestamp": data.get("timestamp", ""),
            "has_eigsh_gap": has_eigsh,
        }

        key = f"{model}|{topology}|N={n_qubits}|p={p_layers}"
        if has_eigsh:
            by_config[key]["post_fix"].append(run_info)
        else:
            by_config[key]["pre_fix"].append(run_info)

    return dict(by_config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare pre-fix vs post-fix gap runs")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--topology", type=str, default=None)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    base = ROOT / "results" / "experiments"
    by_config = scan_runs(base, args.model, args.topology)

    if not by_config:
        print("No non-chain runs found.")
        return 1

    print(f"\n{'=' * 80}")
    print("  H2 COMPARISON: Pre-Fix (floor gap) vs Post-Fix (eigsh gap)")
    print(f"{'=' * 80}")
    print(
        f"\n  {'Config':<35} | {'Pre-fix best':>12} | {'Post-fix best':>13} | {'Δ pass%':>8} | {'h_b shift':>9}"
    )
    print(f"  {'-' * 80}")

    comparisons = []
    for key in sorted(by_config.keys()):
        pre = by_config[key]["pre_fix"]
        post = by_config[key]["post_fix"]

        best_pre = max(pre, key=lambda r: r["pass_rate"]) if pre else None
        best_post = max(post, key=lambda r: r["pass_rate"]) if post else None

        pre_str = f"{best_pre['pass_rate'] * 100:.0f}%" if best_pre else "—"
        post_str = f"{best_post['pass_rate'] * 100:.0f}%" if best_post else "—"

        if best_pre and best_post:
            delta = (best_post["pass_rate"] - best_pre["pass_rate"]) * 100
            delta_str = f"{delta:+.0f}%"
            hb_pre = best_pre["h_boundary"]
            hb_post = best_post["h_boundary"]
            if hb_pre and hb_post:
                shift = f"{hb_post - hb_pre:+.2f}"
            else:
                shift = "—"
        else:
            delta_str = "—"
            shift = "—"

        print(f"  {key:<35} | {pre_str:>12} | {post_str:>13} | {delta_str:>8} | {shift:>9}")

        comparisons.append(
            {
                "config": key,
                "n_pre_runs": len(pre),
                "n_post_runs": len(post),
                "best_pre_pass_rate": best_pre["pass_rate"] if best_pre else None,
                "best_post_pass_rate": best_post["pass_rate"] if best_post else None,
                "h_boundary_pre": best_pre["h_boundary"] if best_pre else None,
                "h_boundary_post": best_post["h_boundary"] if best_post else None,
            }
        )

    # Summary
    has_both = [c for c in comparisons if c["n_pre_runs"] > 0 and c["n_post_runs"] > 0]
    only_pre = [c for c in comparisons if c["n_pre_runs"] > 0 and c["n_post_runs"] == 0]
    only_post = [c for c in comparisons if c["n_pre_runs"] == 0 and c["n_post_runs"] > 0]

    print(f"\n  {'─' * 80}")
    print(f"  Configs with both pre & post: {len(has_both)}")
    print(f"  Configs with only pre-fix:    {len(only_pre)} (need new runs)")
    print(f"  Configs with only post-fix:   {len(only_post)} (new configs)")

    if only_pre:
        print("\n  Awaiting post-fix runs for:")
        for c in only_pre[:10]:
            print(f"    {c['config']} (best pre: {c['best_pre_pass_rate'] * 100:.0f}%)")

    if args.json:
        report = {
            "comparisons": comparisons,
            "summary": {
                "n_both": len(has_both),
                "n_only_pre": len(only_pre),
                "n_only_post": len(only_post),
            },
        }
        json_dump(report, Path(args.json))
        print(f"\n  Saved to: {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
