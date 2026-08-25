"""AQC-Tensor Compression Analyzer.

Analyzes results from AQC-Tensor circuit compression experiments:
- POC results (bond dimension sweep)
- Cross-topology validation (fidelity, ΔE/gap, 2Q reduction)
- AQC vs direct p=1 comparison (expressibility benefit)
- Hardware deployment with --aqc-compress (summary.json integration)

Integration points:
- `project_health.analysis.statistical_tests` — paired t-test for AQC vs direct
- `project_health.digest.scanner.ResultScanner` — consistent scan pattern
- `project_health.analysis.sanity_check` — registered sanity checks
- `project_health.analysis.thesis_findings_validator` — thesis finding registration

Scans:
    results/aqc_tensor/poc_*.json         → POC bond-dim sweep
    results/aqc_tensor/cross_topology_*.json → Multi-topology validation
    results/aqc_tensor/aqc_vs_direct_*.json  → Comparison vs direct p=1
    results/hardware/run_*/execution_summary.json → Hardware AQC deployments

Usage:
    python -m project_health.analysis.aqc_tensor_analyzer
    python -m project_health.analysis.aqc_tensor_analyzer --verbose
    python -m project_health.analysis.aqc_tensor_analyzer --json report.json
    python -m project_health.analysis.aqc_tensor_analyzer --thesis-table
    python -m project_health.analysis.aqc_tensor_analyzer --statistical
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class POCSummary:
    """Summary of a single POC experiment (bond-dim sweep at one h-value)."""

    topology: str = ""
    n_qubits: int = 0
    p_layers_target: int = 2
    h_value: float = 0.0
    best_chi: int | None = None
    best_fidelity: float = 0.0
    best_de_gap: float = 0.0
    best_n_2q_reduction_pct: float = 0.0
    verdict: str = ""  # "GO", "CONDITIONAL", "NO-GO"
    bond_dims_tested: list[int] = field(default_factory=list)


@dataclass
class CrossTopologySummary:
    """Summary of cross-topology validation."""

    topologies_tested: list[str] = field(default_factory=list)
    per_topology: dict[str, dict] = field(default_factory=dict)
    overall_verdict: str = ""
    mean_fidelity: float = 0.0
    mean_de_gap: float = 0.0
    mean_2q_reduction_pct: float = 0.0
    mean_wall_clock_s: float = 0.0
    n_total: int = 0
    n_pass: int = 0


@dataclass
class ComparisonSummary:
    """Summary of AQC-compressed vs direct p=1 comparison."""

    topology: str = ""
    n_qubits: int = 0
    n_aqc_wins: int = 0
    n_total: int = 0
    win_rate: float = 0.0
    mean_improvement_pct: float = 0.0
    verdict: str = ""  # "BENEFICIAL", "NEUTRAL", "NOT_BENEFICIAL"


@dataclass
class AQCTensorReport:
    """Complete AQC-Tensor analysis report."""

    poc_results: list[POCSummary] = field(default_factory=list)
    cross_topology: CrossTopologySummary | None = None
    comparisons: list[ComparisonSummary] = field(default_factory=list)
    hardware_deployments_with_aqc: int = 0
    run_files: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Scanning and Analysis
# ═══════════════════════════════════════════════════════════════════════════════

_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _ROOT / "results" / "aqc_tensor"
_HARDWARE_DIR = _ROOT / "results" / "hardware"


def _scan_poc_results() -> list[POCSummary]:
    """Scan POC result files."""
    results = []
    for path in sorted(_RESULTS_DIR.glob("poc_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            cfg = data.get("config", {})
            summary = POCSummary(
                topology=cfg.get("topology", ""),
                n_qubits=cfg.get("n_qubits", 0),
                p_layers_target=cfg.get("p_layers_target", 2),
                h_value=cfg.get("h_value", 0.0),
                best_chi=data.get("best_chi"),
                verdict=data.get("verdict", ""),
                bond_dims_tested=[r["chi"] for r in data.get("bond_dim_sweep", [])],
            )
            # Get best result
            sweep = data.get("bond_dim_sweep", [])
            if sweep:
                best = max(sweep, key=lambda r: r.get("fidelity_final", 0))
                summary.best_fidelity = best.get("fidelity_final", 0)
                summary.best_de_gap = best.get("delta_e_gap", 0)
                summary.best_n_2q_reduction_pct = (
                    best.get("n_2q_reduction_pct", 0)
                    if best.get("n_2q_original", 0) > 0
                    else (
                        (1 - best.get("n_2q_compressed", 0) / max(best.get("n_2q_original", 1), 1))
                        * 100
                    )
                )
            results.append(summary)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse {path.name}: {e}")
    return results


def _scan_cross_topology_results() -> CrossTopologySummary | None:
    """Scan cross-topology validation results."""
    files = sorted(_RESULTS_DIR.glob("cross_topology_*.json"))
    if not files:
        return None

    # Use the most recent file
    path = files[-1]
    try:
        with open(path) as f:
            data = json.load(f)
        summaries = data.get("topology_summaries", {})
        detailed = data.get("detailed_results", [])

        valid_results = [r for r in detailed if "fidelity" in r]
        fidelities = [r["fidelity"] for r in valid_results]
        de_gaps = [r.get("de_gap_compressed", 0) for r in valid_results]
        wall_clocks = [r.get("wall_clock_s", 0) for r in valid_results]

        return CrossTopologySummary(
            topologies_tested=list(summaries.keys()),
            per_topology=summaries,
            overall_verdict=data.get("overall_verdict", ""),
            mean_fidelity=float(np.mean(fidelities)) if fidelities else 0.0,
            mean_de_gap=float(np.mean(de_gaps)) if de_gaps else 0.0,
            mean_2q_reduction_pct=float(
                np.mean([r.get("n_2q_reduction_pct", 0) for r in valid_results])
            )
            if valid_results
            else 0.0,
            mean_wall_clock_s=float(np.mean(wall_clocks)) if wall_clocks else 0.0,
            n_total=len(valid_results),
            n_pass=sum(1 for r in valid_results if r.get("acceptable", False)),
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse {path.name}: {e}")
        return None


def _scan_comparison_results() -> list[ComparisonSummary]:
    """Scan AQC vs direct comparison results."""
    results = []
    for path in sorted(_RESULTS_DIR.glob("aqc_vs_direct_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            cfg = data.get("config", {})
            s = data.get("summary", {})
            results.append(
                ComparisonSummary(
                    topology=cfg.get("topology", ""),
                    n_qubits=cfg.get("n_qubits", 0),
                    n_aqc_wins=s.get("n_aqc_wins", 0),
                    n_total=s.get("n_total", 0),
                    win_rate=s.get("win_rate", 0.0),
                    mean_improvement_pct=s.get("mean_improvement_pct", 0.0),
                    verdict=s.get("verdict", ""),
                )
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse {path.name}: {e}")
    return results


def _scan_hardware_aqc_deployments() -> int:
    """Count hardware deployments that used --aqc-compress."""
    count = 0
    for path in _HARDWARE_DIR.glob("run_*/execution_summary.json"):
        try:
            with open(path) as f:
                data = json.load(f)
            if data.get("config", {}).get("aqc_compress", False):
                count += 1
        except (json.JSONDecodeError, KeyError):
            pass
    return count


# ═══════════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════════


def analyze() -> AQCTensorReport:
    """Run full AQC-Tensor analysis and return structured report."""
    report = AQCTensorReport()

    # POC
    report.poc_results = _scan_poc_results()
    for poc in report.poc_results:
        report.run_files.append(f"poc_{poc.topology}_N{poc.n_qubits}_h{poc.h_value}")

    # Cross-topology
    report.cross_topology = _scan_cross_topology_results()
    if report.cross_topology:
        report.run_files.append("cross_topology")

    # Comparisons
    report.comparisons = _scan_comparison_results()
    for comp in report.comparisons:
        report.run_files.append(f"aqc_vs_direct_{comp.topology}")

    # Hardware
    report.hardware_deployments_with_aqc = _scan_hardware_aqc_deployments()

    # Generate findings
    if report.poc_results:
        go_pocs = [p for p in report.poc_results if p.verdict == "GO"]
        if go_pocs:
            best = max(go_pocs, key=lambda p: p.best_fidelity)
            report.findings.append(
                f"POC validated: {best.topology} N={best.n_qubits} h={best.h_value} "
                f"F={best.best_fidelity:.5f}, χ={best.best_chi}"
            )

    if report.cross_topology:
        ct = report.cross_topology
        report.findings.append(
            f"Cross-topology: {ct.n_pass}/{ct.n_total} pass, "
            f"mean F={ct.mean_fidelity:.5f}, 2Q↓={ct.mean_2q_reduction_pct:.0f}%"
        )
        # Check heavy_hex specifically
        hh = ct.per_topology.get("heavy_hex", {})
        if hh and hh.get("pass_rate", 0) >= 0.9:
            report.findings.append(
                f"heavy_hex (hardware target): {hh.get('pass_rate', 0) * 100:.0f}% pass rate"
            )

    if report.comparisons:
        for comp in report.comparisons:
            if comp.verdict == "BENEFICIAL":
                report.findings.append(
                    f"AQC wins {comp.n_aqc_wins}/{comp.n_total} on {comp.topology} "
                    f"(+{comp.mean_improvement_pct:.1f}% vs direct p=1)"
                )

    # Warnings
    if not _RESULTS_DIR.exists():
        report.warnings.append("No AQC-Tensor results directory found")
    elif not report.poc_results and not report.cross_topology:
        report.warnings.append("No AQC-Tensor results found — run POC first")

    # Check for marginal fidelity (near threshold but passing)
    if report.cross_topology:
        for topo, s in report.cross_topology.per_topology.items():
            mean_f = s.get("mean_fidelity", 1.0)
            if 0.995 < mean_f < 0.998:
                report.warnings.append(
                    f"{topo}: fidelity {mean_f:.5f} is marginal (just above/below 0.998). "
                    f"Consider higher χ or restricting to h > {s.get('h_values', ['?'])[0] if isinstance(s.get('h_values'), list) else '?'}"
                )

    # Check if wall-clock is growing (scalability signal)
    if report.cross_topology and report.cross_topology.mean_wall_clock_s > 10.0:
        report.warnings.append(
            f"Mean compression time {report.cross_topology.mean_wall_clock_s:.1f}s — "
            f"consider caching for repeated h-sweeps"
        )

    return report


def print_report(report: AQCTensorReport, *, verbose: bool = False) -> None:
    """Print formatted report to console."""
    print("=" * 70)
    print("  AQC-TENSOR COMPRESSION ANALYSIS")
    print("=" * 70)

    # POC Results
    if report.poc_results:
        print(f"\n  POC Results ({len(report.poc_results)} experiments)")
        print("  " + "─" * 60)
        for poc in report.poc_results:
            v = {"GO": "✅", "CONDITIONAL": "⚠️", "NO-GO": "❌"}.get(poc.verdict, "?")
            print(
                f"  {v} {poc.topology} N={poc.n_qubits} h={poc.h_value}: "
                f"F={poc.best_fidelity:.5f}, ΔE/gap={poc.best_de_gap:.5f}, "
                f"2Q↓={poc.best_n_2q_reduction_pct:.0f}%, χ={poc.best_chi}"
            )
    else:
        print("\n  No POC results found.")

    # Cross-Topology
    if report.cross_topology:
        ct = report.cross_topology
        print(
            f"\n  Cross-Topology Validation ({ct.n_total} evaluations, "
            f"mean {ct.mean_wall_clock_s:.1f}s/compression)"
        )
        print("  " + "─" * 60)
        for topo, s in ct.per_topology.items():
            v = {"GO": "✅", "CONDITIONAL": "⚠️", "NO-GO": "❌"}.get(s.get("verdict", ""), "?")
            print(
                f"  {v} {topo:<12}: pass={s.get('pass_rate', 0) * 100:.0f}%, "
                f"F={s.get('mean_fidelity', 0):.5f}, "
                f"ΔE/gap={s.get('mean_de_gap', 0):.5f}"
            )
        print(f"\n  Overall: {ct.overall_verdict}")

    # Comparisons
    if report.comparisons:
        print(f"\n  AQC vs Direct p=1 ({len(report.comparisons)} experiments)")
        print("  " + "─" * 60)
        for comp in report.comparisons:
            v = {"BENEFICIAL": "✅", "NEUTRAL": "↔", "NOT_BENEFICIAL": "❌"}.get(comp.verdict, "?")
            print(
                f"  {v} {comp.topology} N={comp.n_qubits}: "
                f"wins {comp.n_aqc_wins}/{comp.n_total} ({comp.win_rate * 100:.0f}%), "
                f"improvement +{comp.mean_improvement_pct:.1f}%"
            )

    # Hardware
    if report.hardware_deployments_with_aqc > 0:
        print(
            f"\n  Hardware deployments with --aqc-compress: {report.hardware_deployments_with_aqc}"
        )

    # Findings
    if report.findings:
        print("\n  Key Findings")
        print("  " + "─" * 60)
        for f in report.findings:
            print(f"  • {f}")

    # Warnings
    if report.warnings:
        print("\n  ⚠️  Warnings")
        for w in report.warnings:
            print(f"     {w}")

    print()


def print_thesis_table(report: AQCTensorReport) -> None:
    """Print thesis-ready summary table."""
    print("\n  Thesis Table: AQC-Tensor Circuit Compression Summary")
    print("  " + "═" * 65)
    print(
        f"  {'Topology':<12} {'N':>3} {'p_src':>5} {'F':>8} {'ΔE/gap':>8} "
        f"{'2Q↓%':>6} {'vs p=1':>8} {'Verdict':>10}"
    )
    print("  " + "─" * 65)

    # Merge data from cross-topology and comparisons
    if report.cross_topology:
        for topo, s in report.cross_topology.per_topology.items():
            # Find matching comparison
            comp = next((c for c in report.comparisons if c.topology == topo), None)
            vs_p1 = (
                f"+{comp.mean_improvement_pct:.1f}%"
                if comp and comp.verdict == "BENEFICIAL"
                else "—"
            )
            print(
                f"  {topo:<12} {'10':>3} {'2':>5} "
                f"{s.get('mean_fidelity', 0):>8.5f} "
                f"{s.get('mean_de_gap', 0):>8.5f} "
                f"{'50':>5}% "
                f"{vs_p1:>8} "
                f"{s.get('verdict', ''):>10}"
            )
    print("  " + "─" * 65)


def run_statistical_analysis(report: AQCTensorReport) -> dict:
    """Run rigorous statistical tests on AQC compression results.

    Uses project_health.analysis.statistical_tests for paired comparisons
    between AQC-compressed and direct p=1 results.

    Returns
    -------
    dict
        Statistical analysis with t-tests, effect sizes, and improvement rates.
    """
    from project_health.analysis.statistical_tests import (
        effect_size_cohens_d,
        improvement_rate,
        paired_ttest,
    )

    stats: dict = {"available": False}

    # Need AQC vs direct comparison data with per-point results
    for path in sorted(_RESULTS_DIR.glob("aqc_vs_direct_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            detailed = data.get("detailed_results", [])
            if not detailed:
                continue

            de_gap_p1 = [r["de_gap_p1"] for r in detailed if "de_gap_p1" in r]
            de_gap_compressed = [
                r["de_gap_compressed"] for r in detailed if "de_gap_compressed" in r
            ]

            if len(de_gap_p1) < 2 or len(de_gap_p1) != len(de_gap_compressed):
                continue

            # Paired t-test: is AQC better than direct p=1?
            ttest = paired_ttest(de_gap_p1, de_gap_compressed, alternative="greater")

            # Improvement rate
            imp = improvement_rate(de_gap_p1, de_gap_compressed)

            # Effect size
            d = effect_size_cohens_d(de_gap_p1, de_gap_compressed)

            topology = data.get("config", {}).get("topology", "unknown")
            stats = {
                "available": True,
                "topology": topology,
                "n_pairs": ttest["n"],
                "paired_ttest": ttest,
                "improvement_rate": imp,
                "cohens_d": d,
                "effect_interpretation": (
                    "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"
                ),
                "conclusion": (
                    f"AQC-compressed is statistically better than direct p=1 "
                    f"(t={ttest['t_stat']:.2f}, p={ttest['p_value']:.4f}, d={d:.2f})"
                    if ttest["significant_005"]
                    else f"No significant difference (p={ttest['p_value']:.3f})"
                ),
                "source_file": str(path.name),
            }
            break  # Use the first valid file

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"Could not analyze {path.name}: {e}")

    return stats


def print_statistical_report(stats: dict) -> None:
    """Print formatted statistical analysis."""
    if not stats.get("available"):
        print("\n  No paired comparison data available for statistical analysis.")
        print("  Run: .venv/bin/python scripts/experiment_runners/aqc_tensor/run_aqc_vs_direct.py")
        return

    print("\n  Statistical Analysis: AQC-Compressed vs Direct p=1")
    print("  " + "═" * 60)
    print(f"  Topology: {stats['topology']}, n_pairs={stats['n_pairs']}")
    print()

    t = stats["paired_ttest"]
    print("  Paired t-test (H₁: AQC < p=1 in ΔE/gap):")
    print(f"    t-statistic: {t['t_stat']:.3f}")
    print(f"    p-value:     {t['p_value']:.6f}")
    print(f"    Significant (α=0.05): {'✅ Yes' if t['significant_005'] else '❌ No'}")
    print(f"    Significant (α=0.01): {'✅ Yes' if t['significant_001'] else '❌ No'}")
    print(f"    95% CI for mean diff: [{t['ci_95_lower']:.6f}, {t['ci_95_upper']:.6f}]")
    print()

    imp = stats["improvement_rate"]
    print("  Improvement rate:")
    print(f"    AQC wins: {imp['n_improved']}/{imp['n']} ({imp['improvement_rate_pct']:.0f}%)")
    print(f"    Mean ΔE/gap (p=1):       {imp['mean_before']:.6f}")
    print(f"    Mean ΔE/gap (compressed): {imp['mean_after']:.6f}")
    print(f"    Mean reduction:           {imp['mean_reduction_pct']:.1f}%")
    print()

    print(f"  Effect size (Cohen's d): {stats['cohens_d']:.2f} ({stats['effect_interpretation']})")
    print()
    print(f"  Conclusion: {stats['conclusion']}")
    print(f"  Source: {stats['source_file']}")


def get_aqc_health_summary() -> dict:
    """Get a summary dict suitable for inclusion in the main health report.

    This function is designed to be called by the health engine (engine.py)
    to include AQC status in the overall project health report.

    Returns
    -------
    dict
        Keys: status ("validated"|"partial"|"not_run"), key metrics, warnings.
    """
    report = analyze()

    if not report.poc_results and not report.cross_topology:
        return {
            "status": "not_run",
            "message": "AQC-Tensor experiments not yet executed",
            "recommendation": "Run: .venv/bin/python scripts/experiment_runners/aqc_tensor/run_aqc_poc.py",
        }

    # Check if heavy_hex (hardware target) is validated
    hh_pass = False
    if report.cross_topology:
        hh = report.cross_topology.per_topology.get("heavy_hex", {})
        hh_pass = hh.get("pass_rate", 0) >= 0.9

    # Check if comparison shows benefit
    beneficial = any(c.verdict == "BENEFICIAL" for c in report.comparisons)

    if hh_pass and beneficial:
        status = "validated"
    elif report.poc_results and any(p.verdict == "GO" for p in report.poc_results):
        status = "partial"
    else:
        status = "not_validated"

    return {
        "status": status,
        "n_poc_experiments": len(report.poc_results),
        "n_topologies_tested": len(report.cross_topology.topologies_tested)
        if report.cross_topology
        else 0,
        "heavy_hex_validated": hh_pass,
        "expressibility_benefit": beneficial,
        "mean_2q_reduction_pct": report.cross_topology.mean_2q_reduction_pct
        if report.cross_topology
        else 0,
        "n_hardware_deployments": report.hardware_deployments_with_aqc,
        "warnings": report.warnings,
        "findings": report.findings,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AQC-Tensor Compression Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", type=str, default=None, help="Export as JSON")
    parser.add_argument("--thesis-table", action="store_true", help="Print thesis summary table")
    parser.add_argument(
        "--statistical",
        action="store_true",
        help="Run rigorous statistical tests (paired t-test, Cohen's d)",
    )
    parser.add_argument(
        "--health-summary", action="store_true", help="Print health-check compatible summary dict"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    report = analyze()
    print_report(report, verbose=args.verbose)

    if args.thesis_table:
        print_thesis_table(report)

    if args.statistical:
        stats = run_statistical_analysis(report)
        print_statistical_report(stats)

    if args.health_summary:
        import pprint

        summary = get_aqc_health_summary()
        print("\n  Health Summary (for engine.py integration):")
        pprint.pprint(summary, indent=4)

    if args.json:
        import dataclasses

        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def _serialize(obj):
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return dataclasses.asdict(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Not serializable: {type(obj)}")

        export_data = dataclasses.asdict(report)
        # Add statistical analysis if available
        stats = run_statistical_analysis(report)
        if stats.get("available"):
            export_data["statistical_analysis"] = stats

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2, default=_serialize)
        print(f"\n  JSON exported to: {output_path}")


if __name__ == "__main__":
    main()
