#!/usr/bin/env python3
"""Post-verification analysis — checks pipeline results against defined specs.

Reads experiment output directories and produces a structured report
indicating which claims are confirmed, which need correction, and what
the updated valid regime boundaries should be.

This module is generic and reusable: verification specs are passed as
configuration (not hardcoded), and the analysis engine is decoupled from
I/O concerns.

Usage:
    # Verify against default specs (thesis verification plan)
    python -m project_health.analysis.verify_results

    # With JSON export:
    python -m project_health.analysis.verify_results --json report.json

    # Check specific tier only:
    python -m project_health.analysis.verify_results --tier 1

    # Custom results directory:
    python -m project_health.analysis.verify_results --results-dir results/custom/

    # Custom specs file (JSON):
    python -m project_health.analysis.verify_results --specs my_specs.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS_DIR = ROOT / "results" / "thesis" / "verification_r1"


# ─── Default verification specs (thesis verification plan) ───────────────────
DEFAULT_SPECS: dict[str, dict[str, Any]] = {
    "V1": {
        "claim": "p=1 ladder N=6 is reproducible",
        "source": "08_summary §1.1 (topology-agnostic)",
        "pass_criteria": ">=2/3",
        "threshold": 0.05,
        "if_pass": "Ladder N=6 p=1 confirmed viable → add to thesis table",
        "if_fail": "Ladder N=6 p=1 is seed-dependent → note as limitation",
        "tier": 1,
    },
    "V2": {
        "claim": "p=1 triangular N=6 works deeper in valid regime",
        "source": "08_summary §1.1 (topology-agnostic)",
        "pass_criteria": ">=2/3",
        "threshold": 0.05,
        "if_pass": "Previous failure was boundary effect → update h_test recommendation",
        "if_fail": "Triangular N=6 p=1 has genuine expressibility issue",
        "tier": 1,
    },
    "V3": {
        "claim": "p=1 ladder N=10 valid regime boundary is h≥3.0",
        "source": "10_key_findings §6 (p=1 pipeline funciona)",
        "pass_criteria": ">=2/3",
        "threshold": 0.05,
        "if_pass": "Valid regime is h≥3.0 → update P1_VALID_REGIME",
        "if_fail": "Valid regime is h≥3.25 → keep current boundary",
        "tier": 1,
    },
    "V4": {
        "claim": "chain_1d p=1 N=10 valid regime is h≥1.9",
        "source": "binnacle-p1-scaling.md",
        "pass_criteria": ">=2/3",
        "threshold": 0.05,
        "if_pass": "Boundary h≥1.9 confirmed → no change needed",
        "if_fail": "Boundary should be h≥2.25 → correct binnacle",
        "tier": 2,
    },
    "V5": {
        "claim": "Triangular N=10 p=1 failure at h=3.75 is seed-specific",
        "source": "comp5_tri_multi_htest (seed=42 only)",
        "pass_criteria": ">=1/2",
        "threshold": 0.10,
        "if_pass": "Seed=42 failure was chain break → h=3.75 is viable",
        "if_fail": "h=3.75 is systematically outside valid regime → raise to h≥4.0",
        "tier": 2,
    },
    "V6": {
        "claim": "chain_1d p=1 N=6 is seed-independent",
        "source": "08_summary §1.5 (reproducibility)",
        "pass_criteria": "==3/3",
        "threshold": 0.05,
        "if_pass": "chain_1d p=1 N=6 confirmed seed-independent",
        "if_fail": "Unexpected — investigate (chain_1d p=2 has std=0.004)",
        "tier": 3,
    },
}


# ─── Data models ─────────────────────────────────────────────────────────────


@dataclass
class VerificationResult:
    """Result of one verification variant."""

    variant_id: str
    group: str
    seed: int | None
    h_test: float | None
    de_gap: float | None
    verdict: str  # PASS, MARGINAL, FAIL, NO_DATA
    file: str


@dataclass
class GroupConclusion:
    """Conclusion for a verification group."""

    group_id: str
    claim: str
    tier: int
    criteria_met: bool
    conclusion: str
    n_pass: int
    n_marginal: int
    n_fail: int
    n_nodata: int
    n_total: int
    results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class VerificationReport:
    """Full verification report."""

    conclusions: dict[str, GroupConclusion] = field(default_factory=dict)
    n_confirmed: int = 0
    n_rejected: int = 0
    corrections_needed: list[str] = field(default_factory=list)


# ─── Core logic (reusable, no I/O side effects) ─────────────────────────────


def parse_pass_criteria(criteria_str: str) -> tuple[str, int, int]:
    """Parse criteria like '>=2/3' into (operator, required, total).

    Supported formats: '>=2/3', '==3/3', '>=1/2'
    """
    criteria_str = criteria_str.strip()
    if criteria_str.startswith(">="):
        op = ">="
        rest = criteria_str[2:]
    elif criteria_str.startswith("=="):
        op = "=="
        rest = criteria_str[2:]
    else:
        msg = f"Unknown criteria format: {criteria_str}"
        raise ValueError(msg)

    parts = rest.split("/")
    if len(parts) != 2:
        msg = f"Expected 'N/M' format in criteria: {criteria_str}"
        raise ValueError(msg)

    return op, int(parts[0]), int(parts[1])


def evaluate_criteria(n_pass: int, criteria_str: str) -> bool:
    """Evaluate whether n_pass meets the criteria string."""
    op, required, _total = parse_pass_criteria(criteria_str)
    if op == ">=":
        return n_pass >= required
    if op == "==":
        return n_pass == required
    return False


def classify_de_gap(de_gap: float | None, threshold: float = 0.05) -> str:
    """Classify a delta_e/gap value into a verdict."""
    if de_gap is None:
        return "NO_DATA"
    if de_gap < threshold:
        return "PASS"
    if de_gap < threshold * 2:
        return "MARGINAL"
    return "FAIL"


def scan_results_directory(
    results_dir: Path,
    threshold: float = 0.05,
) -> list[VerificationResult]:
    """Scan a results directory for pipeline output files.

    Expects subdirectories with naming convention:
        {group}_{description}_seed{N}/pipeline_run_*.json

    Each pipeline result should have a 'phase4_results' key with
    entries containing 'delta_e_over_gap' and optionally 'h_test'.
    """
    results: list[VerificationResult] = []

    if not results_dir.exists():
        return results

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue

        parts = subdir.name.split("_")
        group = parts[0].upper()

        seed = None
        for p in parts:
            if p.startswith("seed"):
                with contextlib.suppress(ValueError):
                    seed = int(p.replace("seed", ""))

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
            worst_de: float | None = None
            worst_h: float | None = None
            for entry in p4:
                de = entry.get("delta_e_over_gap")
                if de is not None and (worst_de is None or de > worst_de):
                    worst_de = de
                    worst_h = entry.get("h_test")

            verdict = classify_de_gap(worst_de, threshold)
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


def analyze_verification(
    results: list[VerificationResult],
    specs: dict[str, dict[str, Any]],
    *,
    tier_filter: int | None = None,
) -> VerificationReport:
    """Analyze verification results against specs. Pure logic, no I/O."""
    groups: dict[str, list[VerificationResult]] = defaultdict(list)
    for r in results:
        groups[r.group].append(r)

    report = VerificationReport()

    for group_id in sorted(specs.keys()):
        spec = specs[group_id]
        if tier_filter is not None and spec.get("tier") != tier_filter:
            continue

        group_results = groups.get(group_id, [])
        n_pass = sum(1 for r in group_results if r.verdict == "PASS")
        n_marginal = sum(1 for r in group_results if r.verdict == "MARGINAL")
        n_fail = sum(1 for r in group_results if r.verdict == "FAIL")
        n_nodata = sum(1 for r in group_results if r.verdict == "NO_DATA")

        criteria_str = spec.get("pass_criteria", ">=2/3")
        criteria_met = evaluate_criteria(n_pass, criteria_str)
        conclusion = spec["if_pass"] if criteria_met else spec["if_fail"]

        group_conclusion = GroupConclusion(
            group_id=group_id,
            claim=spec["claim"],
            tier=spec.get("tier", 0),
            criteria_met=criteria_met,
            conclusion=conclusion,
            n_pass=n_pass,
            n_marginal=n_marginal,
            n_fail=n_fail,
            n_nodata=n_nodata,
            n_total=len(group_results),
            results=[
                {"seed": r.seed, "de_gap": r.de_gap, "verdict": r.verdict} for r in group_results
            ],
        )
        report.conclusions[group_id] = group_conclusion

    report.n_confirmed = sum(1 for c in report.conclusions.values() if c.criteria_met)
    report.n_rejected = sum(1 for c in report.conclusions.values() if not c.criteria_met)
    report.corrections_needed = [
        f"{gid}: {c.conclusion}" for gid, c in report.conclusions.items() if not c.criteria_met
    ]

    return report


# ─── Output formatting ───────────────────────────────────────────────────────


def format_report_text(report: VerificationReport, specs: dict[str, dict[str, Any]]) -> str:
    """Format verification report as human-readable text."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  VERIFICATION RESULTS ANALYSIS")
    lines.append("=" * 80)

    for group_id, conclusion in sorted(report.conclusions.items()):
        spec = specs.get(group_id, {})
        status = "✅" if conclusion.criteria_met else "❌"

        lines.append(f"\n  ─── {group_id}: {conclusion.claim} ───")
        lines.append(f"  Source: {spec.get('source', 'N/A')}")
        lines.append(f"  Criteria: {spec.get('pass_criteria', 'N/A')}")
        lines.append(
            f"  Results: {conclusion.n_pass} PASS, {conclusion.n_marginal} MARGINAL, "
            f"{conclusion.n_fail} FAIL, {conclusion.n_nodata} NO_DATA"
        )

        for r in conclusion.results:
            de_str = f"{r['de_gap']:.4f}" if r["de_gap"] is not None else "N/A"
            lines.append(f"    seed={r['seed']}: ΔE/gap={de_str} [{r['verdict']}]")

        lines.append(f"  {status} Conclusion: {conclusion.conclusion}")

    lines.append("\n" + "=" * 80)
    lines.append("  SUMMARY")
    lines.append("=" * 80)
    lines.append(f"\n  Claims confirmed: {report.n_confirmed}")
    lines.append(f"  Claims needing correction: {report.n_rejected}")

    if report.corrections_needed:
        lines.append("\n  ⚠️  CORRECTIONS NEEDED:")
        for correction in report.corrections_needed:
            lines.append(f"    {correction}")

    return "\n".join(lines)


def format_report_json(report: VerificationReport) -> str:
    """Format verification report as JSON."""
    data = {gid: asdict(c) for gid, c in report.conclusions.items()}
    data["_summary"] = {
        "n_confirmed": report.n_confirmed,
        "n_rejected": report.n_rejected,
        "corrections_needed": report.corrections_needed,
    }
    return json.dumps(data, indent=2, default=str)


# ─── CLI entry point ─────────────────────────────────────────────────────────


def main() -> None:
    """Run post-verification analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze pipeline results against verification specs.",
        epilog="Run after generating pipeline outputs in the results directory.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Export analysis as JSON to this path",
    )
    parser.add_argument(
        "--tier",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="Only analyze a specific tier",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing verification results (default: {DEFAULT_RESULTS_DIR.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--specs",
        type=Path,
        default=None,
        help="JSON file with custom verification specs (default: built-in thesis specs)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="ΔE/gap threshold for PASS verdict (default: 0.05)",
    )
    args = parser.parse_args()

    # Load specs
    if args.specs:
        with open(args.specs) as f:
            specs = json.load(f)
    else:
        specs = DEFAULT_SPECS

    # Scan results
    results = scan_results_directory(args.results_dir, threshold=args.threshold)

    if not results:
        print(f"\n  No results found in: {args.results_dir}")
        print("  Ensure the directory contains subdirectories with pipeline_run_*.json files.")
        sys.exit(1)

    # Analyze
    report = analyze_verification(results, specs, tier_filter=args.tier)

    # Output
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        output = format_report_json(report)
        args.json.write_text(output)
        print(f"  📄 Saved JSON report to {args.json}")
    else:
        print(format_report_text(report, specs))

    sys.exit(0 if report.n_rejected == 0 else 1)


if __name__ == "__main__":
    main()
