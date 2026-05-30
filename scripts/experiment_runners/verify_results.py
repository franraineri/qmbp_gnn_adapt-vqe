#!/usr/bin/env python3
"""Post-verification analysis — checks results from run_verification_plan.py.

Reads the output of the verification plan and produces a structured report
indicating which claims are confirmed, which need correction, and what
the updated valid regime boundaries should be.

Usage:
    # After running the verification plan:
    python scripts/experiment_runners/verify_results.py

    # With JSON export:
    python scripts/experiment_runners/verify_results.py --json verification_report.json

    # Check specific tier only:
    python scripts/experiment_runners/verify_results.py --tier 1
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
VERIFICATION_DIR = ROOT / "results" / "thesis" / "verification_r1"

# Expected outcomes per variant group
VERIFICATION_SPECS = {
    "V1": {
        "claim": "p=1 ladder N=6 is reproducible",
        "source": "08_summary §1.1 (topology-agnostic)",
        "pass_criteria": "≥2/3 seeds pass (ΔE/gap < 5%)",
        "if_pass": "Ladder N=6 p=1 confirmed viable → add to thesis table",
        "if_fail": "Ladder N=6 p=1 is seed-dependent → note as limitation",
        "tier": 1,
    },
    "V2": {
        "claim": "p=1 triangular N=6 works deeper in valid regime",
        "source": "08_summary §1.1 (topology-agnostic)",
        "pass_criteria": "≥2/3 seeds pass (ΔE/gap < 5%)",
        "if_pass": "Previous failure was boundary effect → update h_test recommendation",
        "if_fail": "Triangular N=6 p=1 has genuine expressibility issue",
        "tier": 1,
    },
    "V3": {
        "claim": "p=1 ladder N=10 valid regime boundary is h≥3.0",
        "source": "10_key_findings §6 (p=1 pipeline funciona)",
        "pass_criteria": "≥2/3 seeds pass (ΔE/gap < 5%)",
        "if_pass": "Valid regime is h≥3.0 → update P1_VALID_REGIME",
        "if_fail": "Valid regime is h≥3.25 → keep current boundary",
        "tier": 1,
    },
    "V4": {
        "claim": "chain_1d p=1 N=10 valid regime is h≥1.9",
        "source": "binnacle-p1-scaling.md",
        "pass_criteria": "≥2/3 seeds pass (ΔE/gap < 5%)",
        "if_pass": "Boundary h≥1.9 confirmed → no change needed",
        "if_fail": "Boundary should be h≥2.25 → correct binnacle",
        "tier": 2,
    },
    "V5": {
        "claim": "Triangular N=10 p=1 failure at h=3.75 is seed-specific",
        "source": "comp5_tri_multi_htest (seed=42 only)",
        "pass_criteria": "≥1/2 seeds pass (ΔE/gap < 10%)",
        "if_pass": "Seed=42 failure was chain break → h=3.75 is viable",
        "if_fail": "h=3.75 is systematically outside valid regime → raise to h≥4.0",
        "tier": 2,
    },
    "V6": {
        "claim": "chain_1d p=1 N=6 is seed-independent",
        "source": "08_summary §1.5 (reproducibility)",
        "pass_criteria": "3/3 seeds pass (ΔE/gap < 5%)",
        "if_pass": "chain_1d p=1 N=6 confirmed seed-independent",
        "if_fail": "Unexpected — investigate (chain_1d p=2 has std=0.004)",
        "tier": 3,
    },
}


@dataclass
class VerificationResult:
    """Result of one verification variant."""

    variant_id: str
    group: str  # V1, V2, etc.
    seed: int | None
    h_test: float | None
    de_gap: float | None
    verdict: str  # PASS, MARGINAL, FAIL, NO_DATA
    file: str


def scan_verification_results() -> list[VerificationResult]:
    """Scan the verification output directory for results."""
    results: list[VerificationResult] = []

    if not VERIFICATION_DIR.exists():
        print(f"  ⚠️  Verification directory not found: {VERIFICATION_DIR}")
        print("  Run the verification plan first:")
        print("    python scripts/experiment_runners/run_verification_plan.py")
        return results

    for subdir in sorted(VERIFICATION_DIR.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue

        # Parse variant group and seed from folder name
        # Format: v1_ladder_N6_seed42, v4_chain_N10_boundary_seed43, etc.
        parts = subdir.name.split("_")
        group = parts[0].upper()  # v1 → V1
        seed = None
        for p in parts:
            if p.startswith("seed"):
                with contextlib.suppress(ValueError):
                    seed = int(p.replace("seed", ""))

        # Find pipeline_run file
        pipeline_files = sorted(subdir.glob("pipeline_run_*.json"))
        if not pipeline_files:
            results.append(
                VerificationResult(
                    variant_id=subdir.name,
                    group=group,
                    seed=seed,
                    h_test=None,
                    de_gap=None,
                    verdict="NO_DATA",
                    file=str(subdir),
                )
            )
            continue

        # Parse the result (use worst-case across all h_test points)
        for pf in pipeline_files:
            try:
                with open(pf) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            p4 = data.get("phase4_results", [])
            if not p4:
                results.append(
                    VerificationResult(
                        variant_id=subdir.name,
                        group=group,
                        seed=seed,
                        h_test=None,
                        de_gap=None,
                        verdict="NO_DATA",
                        file=str(pf.relative_to(ROOT)),
                    )
                )
                continue

            # Worst-case across all test points
            worst_de = None
            worst_h = None
            for entry in p4:
                de = entry.get("delta_e_over_gap")
                if de is not None and (worst_de is None or de > worst_de):
                    worst_de = de
                    worst_h = entry.get("h_test")

            if worst_de is None:
                verdict = "NO_DATA"
            elif worst_de < 0.05:
                verdict = "PASS"
            elif worst_de < 0.10:
                verdict = "MARGINAL"
            else:
                verdict = "FAIL"

            results.append(
                VerificationResult(
                    variant_id=subdir.name,
                    group=group,
                    seed=seed,
                    h_test=worst_h,
                    de_gap=worst_de,
                    verdict=verdict,
                    file=str(pf.relative_to(ROOT)),
                )
            )

    return results


def analyze_verification(results: list[VerificationResult], tier_filter: int | None) -> dict:
    """Analyze verification results and produce conclusions per group."""
    # Group by variant group (V1, V2, etc.)
    groups: dict[str, list[VerificationResult]] = defaultdict(list)
    for r in results:
        groups[r.group].append(r)

    conclusions: dict[str, dict] = {}

    print("=" * 80)
    print("  VERIFICATION RESULTS ANALYSIS")
    print("=" * 80)

    for group_id in sorted(groups.keys()):
        spec = VERIFICATION_SPECS.get(group_id)
        if spec is None:
            continue
        if tier_filter is not None and spec["tier"] != tier_filter:
            continue

        group_results = groups[group_id]
        n_pass = sum(1 for r in group_results if r.verdict == "PASS")
        n_marginal = sum(1 for r in group_results if r.verdict == "MARGINAL")
        n_fail = sum(1 for r in group_results if r.verdict == "FAIL")
        n_nodata = sum(1 for r in group_results if r.verdict == "NO_DATA")
        n_total = len(group_results)

        # Determine if criteria met
        criteria = spec["pass_criteria"]
        if "3/3" in criteria:
            criteria_met = n_pass == 3
        elif "2/3" in criteria:
            criteria_met = n_pass >= 2
        elif "1/2" in criteria:
            criteria_met = n_pass >= 1
        else:
            criteria_met = n_pass > n_fail

        conclusion = spec["if_pass"] if criteria_met else spec["if_fail"]
        status_emoji = "✅" if criteria_met else "❌"

        print(f"\n  ─── {group_id}: {spec['claim']} ───")
        print(f"  Source: {spec['source']}")
        print(f"  Criteria: {criteria}")
        print(f"  Results: {n_pass} PASS, {n_marginal} MARGINAL, {n_fail} FAIL, {n_nodata} NO_DATA")

        # Detail per seed
        for r in sorted(group_results, key=lambda x: x.seed or 0):
            de_str = f"{r.de_gap:.4f}" if r.de_gap is not None else "N/A"
            print(f"    seed={r.seed}: ΔE/gap={de_str} [{r.verdict}]")

        print(f"  {status_emoji} Conclusion: {conclusion}")

        conclusions[group_id] = {
            "claim": spec["claim"],
            "tier": spec["tier"],
            "criteria_met": criteria_met,
            "conclusion": conclusion,
            "n_pass": n_pass,
            "n_total": n_total,
            "results": [
                {"seed": r.seed, "de_gap": r.de_gap, "verdict": r.verdict} for r in group_results
            ],
        }

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    confirmed = sum(1 for c in conclusions.values() if c["criteria_met"])
    rejected = sum(1 for c in conclusions.values() if not c["criteria_met"])
    print(f"\n  Claims confirmed: {confirmed}")
    print(f"  Claims needing correction: {rejected}")

    if rejected > 0:
        print("\n  ⚠️  CORRECTIONS NEEDED:")
        for gid, c in conclusions.items():
            if not c["criteria_met"]:
                print(f"    {gid}: {c['conclusion']}")

    return conclusions


def main() -> None:
    """Run post-verification analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze results from the verification plan.",
        epilog="Run after: python scripts/experiment_runners/run_verification_plan.py",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Export analysis as JSON",
    )
    parser.add_argument(
        "--tier",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="Only analyze a specific tier",
    )
    args = parser.parse_args()

    results = scan_verification_results()

    if not results:
        print("\n  No verification results found.")
        print("  Run the verification plan first:")
        print("    python scripts/experiment_runners/run_verification_plan.py --noiseless-only")
        sys.exit(1)

    conclusions = analyze_verification(results, args.tier)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(conclusions, f, indent=2, default=str)
        print(f"\n  📄 Saved report to {args.json}")


if __name__ == "__main__":
    main()
