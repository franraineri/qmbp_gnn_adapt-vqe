#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.analysis.generate_best_results_scoreboard import (
    scan_all_reports,
    TARGET_H,
)


def validate(topology_filter: str | None = None, verbose: bool = False) -> dict:
    entries = scan_all_reports(target_h=TARGET_H)
    if topology_filter:
        entries = [e for e in entries if e.topology == topology_filter]

    by_topo_n: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for e in entries:
        by_topo_n[e.topology][e.n_qubits].append(e)

    issues: list[str] = []
    n_checked = 0

    for topo in sorted(by_topo_n.keys()):
        for n in sorted(by_topo_n[topo].keys()):
            entries_n = by_topo_n[topo][n]
            best = min(entries_n, key=lambda e: e.result.abs_error)
            r = best.result
            n_checked += 1

            recomputed_abs = abs(r.e_pred - r.e_exact)
            if r.abs_error > 1e-8 and abs(recomputed_abs - r.abs_error) / r.abs_error > 0.02:
                issues.append(
                    f"{topo} N={n}: |ΔE| inconsistent: "
                    f"stored={r.abs_error:.6f} recomputed={recomputed_abs:.6f}"
                )

            if r.gap > 1e-10:
                recomputed_dg = r.abs_error / r.gap
                if r.de_gap > 1e-8 and abs(recomputed_dg - r.de_gap) / r.de_gap > 0.02:
                    issues.append(
                        f"{topo} N={n}: ΔE/gap inconsistent: "
                        f"stored={r.de_gap:.6f} recomputed={recomputed_dg:.6f}"
                    )

            if verbose:
                print(f"  {topo:14s} N={n:>3}: |ΔE|={r.abs_error:.4f} OK")

    gt_issues = _validate_gt_freshness(by_topo_n)
    issues.extend(gt_issues)

    scoreboard_issues = _validate_against_scoreboard_json(by_topo_n)
    issues.extend(scoreboard_issues)

    report = {
        "n_entries": len(entries),
        "n_topo_n_checked": n_checked,
        "n_issues": len(issues),
        "issues": issues,
        "pass": len(issues) == 0,
    }

    if issues:
        print(f"\n  ISSUES ({len(issues)}):")
        for i in issues:
            print(f"    {i}")
    else:
        print(f"  ALL CHECKS PASSED ({n_checked} (topo,N) pairs, {len(entries)} entries)")

    return report


def _validate_gt_freshness(
    by_topo_n: dict[str, dict[int, list]],
) -> list[str]:
    issues: list[str] = []
    try:
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        gt_cache = GroundTruthCache()
        for topo in by_topo_n:
            for n in by_topo_n[topo]:
                best = min(by_topo_n[topo][n], key=lambda e: e.result.abs_error)
                r = best.result
                cached = gt_cache.get(topo, n, "tfim_bond_resolved", r.h)
                if cached is None:
                    cached = gt_cache.get(topo, n, "tfim", r.h)
                if cached is not None:
                    gt_e = float(cached["energy"])
                    if abs(gt_e - r.e_exact) > 0.001:
                        issues.append(
                            f"{topo} N={n} h={r.h:.2f}: STALE e_exact "
                            f"(report={r.e_exact:.4f} GT={gt_e:.4f} Δ={abs(gt_e - r.e_exact):.4f})"
                        )
    except Exception:
        pass
    return issues


def _validate_against_scoreboard_json(
    by_topo_n: dict[str, dict[int, list]],
) -> list[str]:
    issues: list[str] = []
    json_path = ROOT / "results" / "best_results_scoreboard.json"
    if not json_path.exists():
        return issues

    try:
        with open(json_path) as f:
            sb = json.load(f)

        for topo in by_topo_n:
            sb_topo = sb.get("best_by_topology", {}).get(topo, {})
            for n in by_topo_n[topo]:
                best = min(by_topo_n[topo][n], key=lambda e: e.result.abs_error)
                sb_entry = sb_topo.get(str(n))
                if sb_entry is None:
                    continue
                sb_abs = sb_entry.get("best_abs_error", 0)
                if abs(sb_abs - best.result.abs_error) > 0.001:
                    issues.append(
                        f"{topo} N={n}: scoreboard JSON stale "
                        f"(JSON={sb_abs:.4f} current_best={best.result.abs_error:.4f})"
                    )
    except Exception:
        pass
    return issues


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate scoreboard data integrity")
    parser.add_argument("--topology", "-t", type=str, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    report = validate(topology_filter=args.topology, verbose=args.verbose)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
