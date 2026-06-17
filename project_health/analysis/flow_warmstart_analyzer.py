"""Flow Warmstart & σ_flow Extension Analyzer.

Analyzes results from the hardware-extension-integration feature:
- FlowWarmstartManager performance (mode d in §10)
- BondResolvedMPNN performance (mode e in §10)
- σ_flow boost impact on kappa_go_no_go() recommendations
- Ext1b p=1 revalidation results

Usage:
    python -m project_health.analysis.flow_warmstart_analyzer
    python -m project_health.analysis.flow_warmstart_analyzer --verbose
    python -m project_health.analysis.flow_warmstart_analyzer --json report.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════


from dataclasses import dataclass, field


@dataclass
class FlowWarmstartSummary:
    """Summary of flow warmstart mode (d) performance."""

    mean_de_gap: float = 0.0
    mean_iters: float = 0.0
    speedup_vs_random: float | None = None
    trainable_params: int = 0
    final_nll: float = 0.0
    converged: bool = False  # NLL < 2.0
    mean_sigma_flow: float = 0.0
    n_high_sigma: int = 0  # σ > 0.5
    sigma_flow_per_h: dict[float, float] = field(default_factory=dict)


@dataclass
class BondResolvedSummary:
    """Summary of bond-resolved mode (e) performance."""

    mean_de_gap: float = 0.0
    mean_iters: float = 0.0
    speedup_vs_random: float | None = None
    eligible: bool = False  # chain_1d N=6 p=2


@dataclass
class SigmaFlowBoostSummary:
    """Summary of σ_flow boost impact on kappa_go_no_go()."""

    n_total_h_points: int = 0
    n_boosted: int = 0  # h-points where sigma_flow_boost=True
    total_extra_shots: int = 0
    boost_rate: float = 0.0


@dataclass
class Ext1bSummary:
    """Summary of Ext1b p=1 revalidation."""

    n_cv_points: int = 0
    n_passed: int = 0
    pass_rate: float = 0.0
    skipped: bool = False
    per_h: dict[float, dict] = field(default_factory=dict)


@dataclass
class FlowExtensionReport:
    """Complete analysis report for hardware-extension-integration."""

    flow_warmstart: FlowWarmstartSummary | None = None
    bond_resolved: BondResolvedSummary | None = None
    sigma_flow_boost: SigmaFlowBoostSummary | None = None
    ext1b: Ext1bSummary | None = None
    run_files: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis Functions
# ═══════════════════════════════════════════════════════════════════════════════


def _analyze_flow_warmstart(section_data: dict) -> FlowWarmstartSummary:
    """Extract flow warmstart metrics from §10 section results."""
    fw = section_data.get("flow_warmstart", {})
    if not fw:
        return FlowWarmstartSummary()

    nll_history = fw.get("train_nll_history", [])
    final_nll = nll_history[-1] if nll_history else 0.0
    sigma_per_h = fw.get("sigma_flow_per_h", {})
    sigmas = list(sigma_per_h.values())

    return FlowWarmstartSummary(
        mean_de_gap=fw.get("de_gap", 0.0),
        mean_iters=fw.get("n_iterations", 0.0),
        speedup_vs_random=fw.get("speedup_vs_random"),
        trainable_params=fw.get("trainable_params", 0),
        final_nll=final_nll,
        converged=final_nll < 2.0,
        mean_sigma_flow=sum(sigmas) / len(sigmas) if sigmas else 0.0,
        n_high_sigma=sum(1 for s in sigmas if s > 0.5),
        sigma_flow_per_h={float(k): v for k, v in sigma_per_h.items()},
    )


def _analyze_bond_resolved(section_data: dict) -> BondResolvedSummary:
    """Extract bond-resolved mode (e) metrics."""
    br = section_data.get("bond_resolved_warmstart", {})
    if not br:
        return BondResolvedSummary(eligible=False)

    return BondResolvedSummary(
        mean_de_gap=br.get("de_gap", 0.0),
        mean_iters=br.get("n_iterations", 0.0),
        speedup_vs_random=br.get("speedup_vs_random"),
        eligible=True,
    )


def _analyze_sigma_boost(recommendations: dict[str, dict]) -> SigmaFlowBoostSummary:
    """Analyze σ_flow boost impact from kappa_go_no_go() output."""
    if not recommendations:
        return SigmaFlowBoostSummary()

    n_total = len(recommendations)
    n_boosted = sum(1 for r in recommendations.values() if r.get("sigma_flow_boost", False))
    # Estimate extra shots from boost (shots doubled for boosted h-points)
    extra_shots = sum(
        r.get("shots", 0) // 2 for r in recommendations.values() if r.get("sigma_flow_boost", False)
    )

    return SigmaFlowBoostSummary(
        n_total_h_points=n_total,
        n_boosted=n_boosted,
        total_extra_shots=extra_shots,
        boost_rate=n_boosted / n_total if n_total > 0 else 0.0,
    )


def _analyze_ext1b(result_data: dict) -> Ext1bSummary:
    """Extract Ext1b p=1 revalidation metrics."""
    if result_data.get("skipped", False):
        return Ext1bSummary(skipped=True)

    return Ext1bSummary(
        n_cv_points=result_data.get("n_cv_points", 0),
        n_passed=result_data.get("n_passed", 0),
        pass_rate=result_data.get("pass_rate", 0.0),
        skipped=False,
        per_h={float(k): v for k, v in result_data.get("per_h", {}).items()},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Scanner
# ═══════════════════════════════════════════════════════════════════════════════


def scan_flow_extension_results(
    results_dir: Path | None = None,
) -> FlowExtensionReport:
    """Scan result directories for hardware-extension-integration data.

    Searches:
    - results/experiments/exp_hw_rehearsal_v3/ → flow_warmstart, bond_resolved
    - results/experiments/exp_ext1b_p1/ → Ext1b revalidation
    - §10 results with sigma_flow_per_h → boost analysis

    Parameters
    ----------
    results_dir : Path | None
        Root results directory. Defaults to project_root/results.

    Returns
    -------
    FlowExtensionReport
        Aggregated analysis report.
    """
    if results_dir is None:
        results_dir = Path(__file__).parent.parent.parent / "results"

    report = FlowExtensionReport()

    # Scan V3 rehearsal results for flow/bond-resolved data
    v3_dir = results_dir / "experiments" / "exp_hw_rehearsal_v3"
    if v3_dir.exists():
        for run_file in sorted(v3_dir.glob("run_*.json"), reverse=True):
            try:
                with open(run_file) as f:
                    data = json.load(f)
                report.run_files.append(str(run_file.name))

                # Look for §10 results
                sections = data.get("sections", data.get("results", {}))
                section_10 = None
                if isinstance(sections, dict):
                    section_10 = sections.get("10", sections.get("section_10"))
                elif isinstance(sections, list):
                    for s in sections:
                        if isinstance(s, dict) and s.get("id") == 10:
                            section_10 = s.get("result", s)
                            break

                # Navigate to section_10.data (where flow_warmstart lives)
                if section_10 and isinstance(section_10, dict):
                    s10_data = section_10.get("data", section_10)
                else:
                    s10_data = {}

                if s10_data and "flow_warmstart" in s10_data:
                    report.flow_warmstart = _analyze_flow_warmstart(s10_data)
                    report.findings.append(
                        f"Flow warmstart: ΔE/gap={report.flow_warmstart.mean_de_gap:.4f}, "
                        f"σ_flow_mean={report.flow_warmstart.mean_sigma_flow:.3f}, "
                        f"NLL={report.flow_warmstart.final_nll:.3f}"
                    )
                    if not report.flow_warmstart.converged:
                        report.warnings.append(
                            f"Flow NLL={report.flow_warmstart.final_nll:.2f} > 2.0 — "
                            "may not have converged. Consider more epochs or "
                            "patience-based early stopping."
                        )
                    break  # Use latest run only

                if s10_data and "bond_resolved_warmstart" in s10_data:
                    report.bond_resolved = _analyze_bond_resolved(s10_data)

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"  Failed to parse {run_file.name}: {e}")

    # Scan Ext1b results
    ext1b_dir = results_dir / "experiments" / "exp_ext1b_p1"
    if ext1b_dir.exists():
        for run_file in sorted(ext1b_dir.glob("run_*.json"), reverse=True):
            try:
                with open(run_file) as f:
                    data = json.load(f)
                # Extract section 1 result
                sections = data.get("sections", [])
                if isinstance(sections, list) and sections:
                    sec1 = sections[0].get("result", sections[0])
                    report.ext1b = _analyze_ext1b(sec1)
                elif isinstance(data, dict) and "per_h" in data:
                    report.ext1b = _analyze_ext1b(data)
                break  # Latest only
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"  Failed to parse {run_file.name}: {e}")

    # Generate findings
    if report.flow_warmstart and report.flow_warmstart.speedup_vs_random:
        spd = report.flow_warmstart.speedup_vs_random
        report.findings.append(
            f"Flow speedup vs random: {spd:.2f}x"
            + (" ✅" if spd > 1.5 else " ⚠️ below 1.5x threshold")
        )

    if report.flow_warmstart and report.flow_warmstart.n_high_sigma > 0:
        report.findings.append(
            f"σ_flow > 0.5 at {report.flow_warmstart.n_high_sigma} h-points — "
            "these will receive 2× shots + 3 layouts in hardware deployment."
        )

    if report.ext1b and not report.ext1b.skipped:
        status = "✅ ALL PASS" if report.ext1b.pass_rate == 1.0 else "⚠️ PARTIAL"
        report.findings.append(
            f"Ext1b p=1 revalidation: {report.ext1b.n_passed}/{report.ext1b.n_cv_points} "
            f"CV points pass ({report.ext1b.pass_rate:.0%}) {status}"
        )

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Formatter
# ═══════════════════════════════════════════════════════════════════════════════


def format_report(report: FlowExtensionReport, verbose: bool = False) -> str:
    """Format the extension report for console output."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  Flow Warmstart & Hardware Extension Analysis")
    lines.append("=" * 72)

    if not report.run_files and report.ext1b is None:
        lines.append("\n  No hardware-extension-integration results found.")
        lines.append("  Run §10 with --use-flow-warmstart or --use-bond-resolved first.")
        return "\n".join(lines)

    # Flow warmstart section
    if report.flow_warmstart:
        fw = report.flow_warmstart
        lines.append("\n── Flow Warmstart (mode d) ──────────────────────────────")
        lines.append(f"  ΔE/gap (mean):      {fw.mean_de_gap:.4f}")
        lines.append(f"  Iterations (mean):  {fw.mean_iters:.0f}")
        if fw.speedup_vs_random:
            lines.append(f"  Speedup vs random:  {fw.speedup_vs_random:.2f}x")
        lines.append(f"  Trainable params:   {fw.trainable_params}")
        lines.append(
            f"  Final NLL:          {fw.final_nll:.3f} {'✅' if fw.converged else '⚠️ not converged'}"
        )
        lines.append(f"  Mean σ_flow:        {fw.mean_sigma_flow:.3f}")
        lines.append(f"  High-σ points:      {fw.n_high_sigma} (σ > 0.5 → boost)")

        if verbose and fw.sigma_flow_per_h:
            lines.append("\n  Per-h σ_flow:")
            for h, s in sorted(fw.sigma_flow_per_h.items(), reverse=True):
                flag = " ← BOOST" if s > 0.5 else ""
                lines.append(f"    h={h:.3f}: σ={s:.4f}{flag}")

    # Bond-resolved section
    if report.bond_resolved and report.bond_resolved.eligible:
        br = report.bond_resolved
        lines.append("\n── BondResolved (mode e) ────────────────────────────────")
        lines.append(f"  ΔE/gap (mean):      {br.mean_de_gap:.4f}")
        lines.append(f"  Iterations (mean):  {br.mean_iters:.0f}")
        if br.speedup_vs_random:
            lines.append(f"  Speedup vs random:  {br.speedup_vs_random:.2f}x")

    # σ_flow boost section
    if report.sigma_flow_boost:
        sb = report.sigma_flow_boost
        lines.append("\n── σ_flow Boost Impact ──────────────────────────────────")
        lines.append(f"  Total h-points:     {sb.n_total_h_points}")
        lines.append(f"  Boosted:            {sb.n_boosted} ({sb.boost_rate:.0%})")
        lines.append(f"  Extra shots used:   {sb.total_extra_shots:,}")

    # Ext1b section
    if report.ext1b:
        e = report.ext1b
        lines.append("\n── Ext1b p=1 Revalidation ──────────────────────────────")
        if e.skipped:
            lines.append("  SKIPPED: no CONDITIONALLY_VIABLE h-points found")
        else:
            status = "✅" if e.pass_rate == 1.0 else "❌"
            lines.append(f"  CV points:          {e.n_cv_points}")
            lines.append(
                f"  Passed:             {e.n_passed}/{e.n_cv_points} ({e.pass_rate:.0%}) {status}"
            )
            if verbose and e.per_h:
                for h, data in sorted(e.per_h.items(), reverse=True):
                    p = "PASS" if data.get("pass") else "FAIL"
                    lines.append(f"    h={h:.3f}: ΔE/gap={data['de_gap']:.4f} ({p})")

    # Findings
    if report.findings:
        lines.append("\n── Findings ────────────────────────────────────────────")
        for f in report.findings:
            lines.append(f"  • {f}")

    # Warnings
    if report.warnings:
        lines.append("\n── Warnings ────────────────────────────────────────────")
        for w in report.warnings:
            lines.append(f"  ⚠️ {w}")

    lines.append("\n" + "=" * 72)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entry point for the flow warmstart analyzer."""
    parser = argparse.ArgumentParser(description="Analyze hardware-extension-integration results")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", type=str, default=None, help="Save JSON report to this path")
    parser.add_argument("--results-dir", type=str, default=None, help="Override results directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    results_dir = Path(args.results_dir) if args.results_dir else None
    report = scan_flow_extension_results(results_dir)

    # Console output
    print(format_report(report, verbose=args.verbose))

    # Optional JSON export
    if args.json:
        from dataclasses import asdict

        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        print(f"\n  JSON report saved to: {json_path}")


if __name__ == "__main__":
    main()
