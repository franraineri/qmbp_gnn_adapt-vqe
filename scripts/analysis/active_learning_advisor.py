#!/usr/bin/env python
"""Active Learning Advisor for QPT/DQPT Data Generation.

Analyzes existing data coverage for a given topology and recommends
the highest-value (N, h) configurations to compute next. Prioritizes:

1. QPT gap-filling: N values with insufficient h-coverage near h_c
2. DQPT coverage: N values without trajectory data
3. Derivative undersampling: h-regions where d²E/dh² changes rapidly
   but data is sparse (information gain maximized)
4. Finite-size scaling: N values needed to improve FSS fit quality

Usage:
    # Get recommendations for heavy_hex
    python scripts/analysis/active_learning_advisor.py --topology heavy_hex

    # Generate commands ready to run
    python scripts/analysis/active_learning_advisor.py --topology heavy_hex --generate-commands

    # Focus on QPT data only
    python scripts/analysis/active_learning_advisor.py --topology heavy_hex --focus qpt

    # Focus on DQPT data only
    python scripts/analysis/active_learning_advisor.py --topology heavy_hex --focus dqpt
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Recommendation:
    """A single data generation recommendation."""

    priority: int  # 1 = highest
    category: str  # "qpt", "dqpt", "fss", "crossover"
    n_qubits: int
    h_values: list[float] | None = None  # For QPT: h-grid to generate
    h_pre: float | None = None  # For DQPT: quench start
    h_post: float | None = None  # For DQPT: quench end
    reason: str = ""
    estimated_time_min: float = 0.0
    command: str = ""


@dataclass
class CoverageReport:
    """Summary of data coverage for a topology."""

    topology: str
    # QPT
    qpt_n_values: list[int] = field(default_factory=list)
    qpt_reliable_n: list[int] = field(default_factory=list)
    qpt_gaps: list[int] = field(default_factory=list)  # N with insufficient QPT data
    qpt_h_coverage: dict[int, tuple[float, float, int]] = field(default_factory=dict)
    # DQPT
    dqpt_n_values: list[int] = field(default_factory=list)
    dqpt_missing_n: list[int] = field(default_factory=list)
    # FSS
    fss_quality: float = 0.0  # R² of current fit
    fss_n_points: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Coverage Analysis
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_coverage(topology: str, p_layers: int = 1) -> CoverageReport:
    """Analyze current data coverage for QPT and DQPT.

    Parameters
    ----------
    topology : str
        Target topology.
    p_layers : int
        HVA depth.

    Returns
    -------
    CoverageReport
        Detailed coverage information.
    """
    from scripts.analysis.qpt_detection import (
        load_energy_curves,
        run_qpt_analysis,
    )

    report = CoverageReport(topology=topology)

    # ── QPT coverage ─────────────────────────────────────────────────────────
    try:
        curves = load_energy_curves(topology, p_layers, use_predicted=False)
    except Exception as e:
        logger.warning(f"Failed to load energy curves for {topology}: {e}")
        curves = {}

    for n in sorted(curves.keys()):
        data = curves[n]
        h = data["h"]
        # Count points in QPT zone (h < 2.0)
        mask_qpt = h < 2.0
        n_qpt_pts = int(mask_qpt.sum())
        h_min_qpt = float(h[mask_qpt].min()) if n_qpt_pts > 0 else 999.0
        h_max_qpt = float(h[mask_qpt].max()) if n_qpt_pts > 0 else 0.0

        report.qpt_n_values.append(n)
        report.qpt_h_coverage[n] = (h_min_qpt, h_max_qpt, n_qpt_pts)

        # Sufficient for QPT: need >=8 points spanning [0.3, 1.5+]
        if n_qpt_pts >= 8 and h_min_qpt <= 0.5 and h_max_qpt >= 1.3:
            report.qpt_reliable_n.append(n)
        else:
            report.qpt_gaps.append(n)

    # Run FSS to get quality
    try:
        qpt_result = run_qpt_analysis(topology, p_layers, use_predicted=False)
        fss = qpt_result.get("finite_size_scaling")
    except Exception:
        qpt_result = {}
        fss = None
    if fss and "error" not in fss:
        report.fss_quality = fss.get("r_squared", 0.0)
        report.fss_n_points = len(fss.get("N_values", []))

    # ── DQPT coverage ────────────────────────────────────────────────────────
    dqpt_dir = _project_root / "data" / "dqpt_trajectories"
    if dqpt_dir.exists():
        for npz_file in dqpt_dir.glob(f"{topology}_N*.npz"):
            try:
                stem = npz_file.stem
                n_str = stem.split("_N")[1].split("_h")[0]
                n = int(n_str)
                if n not in report.dqpt_n_values:
                    report.dqpt_n_values.append(n)
            except (IndexError, ValueError):
                continue

    report.dqpt_n_values.sort()

    # Target N values for DQPT (ED-accessible: 8-22)
    target_dqpt_n = [8, 10, 12, 14, 16, 18, 20]
    report.dqpt_missing_n = [n for n in target_dqpt_n if n not in report.dqpt_n_values]

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Recommendation Engine
# ═══════════════════════════════════════════════════════════════════════════════


def generate_recommendations(
    topology: str,
    p_layers: int = 1,
    focus: str | None = None,
    max_recommendations: int = 10,
) -> list[Recommendation]:
    """Generate prioritized recommendations for next data to compute.

    The advisor uses an information-gain heuristic:
    - QPT: prioritize N values where adding data improves FSS R² most
    - DQPT: prioritize N values that complete the scaling ladder
    - Derivative undersampling: flag h-regions with large d²E spacing

    Parameters
    ----------
    topology : str
        Target topology.
    p_layers : int
        HVA depth.
    focus : str | None
        If "qpt" or "dqpt", only generate that category.
    max_recommendations : int
        Maximum number of recommendations.

    Returns
    -------
    list[Recommendation]
        Sorted by priority (1 = highest).
    """
    report = analyze_coverage(topology, p_layers)
    recs: list[Recommendation] = []
    priority = 1

    # ── QPT Recommendations ──────────────────────────────────────────────────
    if focus in (None, "qpt"):
        # Priority 1: N values with ZERO data near QPT that are in the FSS range
        # These are the highest-value computations (unlock FSS)
        fss_critical_n = []
        for n in report.qpt_gaps:
            if 8 <= n <= 22:  # ED-accessible
                coverage = report.qpt_h_coverage.get(n, (999, 0, 0))
                h_min, h_max, n_pts = coverage
                if n_pts < 5 or h_min > 1.0:
                    fss_critical_n.append(n)

        for n in sorted(fss_critical_n):
            # Generate dense grid around expected h_c
            h_grid = np.arange(0.30, 2.05, 0.05).tolist()
            est_time = 0.5 + n * 0.3  # Rough: ED scales ~exponentially
            recs.append(Recommendation(
                priority=priority,
                category="qpt",
                n_qubits=n,
                h_values=h_grid,
                reason=f"N={n} has {report.qpt_h_coverage.get(n, (0,0,0))[2]} pts in QPT zone — "
                       f"need ~35 pts in h=[0.3, 2.0] for reliable h_c detection",
                estimated_time_min=est_time,
                command=(
                    f".venv/bin/python scripts/experiment_runners/bond_resolved/"
                    f"run_accelerated_cross_n.py "
                    f"--topology {topology} --train-n {n} --p-layers {p_layers} "
                    f"--h-min 0.30 --h-max 2.00 --h-points 35"
                ),
            ))
            priority += 1

        # Priority 2: N values with SOME data but sparse (could improve FSS)
        for n in sorted(report.qpt_gaps):
            if n in fss_critical_n:
                continue  # Already recommended
            if n > 22:
                continue  # MPS-only, not ED
            coverage = report.qpt_h_coverage.get(n, (999, 0, 0))
            h_min, h_max, n_pts = coverage
            if 5 <= n_pts < 15:
                # Add more points to improve resolution
                recs.append(Recommendation(
                    priority=priority,
                    category="qpt",
                    n_qubits=n,
                    h_values=np.arange(0.30, 2.05, 0.05).tolist(),
                    reason=f"N={n} has {n_pts} pts (h=[{h_min:.2f},{h_max:.2f}]) — "
                           f"augmenting to 35 pts improves d²E/dh² resolution",
                    estimated_time_min=0.5 + n * 0.3,
                    command=(
                        f".venv/bin/python scripts/experiment_runners/bond_resolved/"
                        f"run_accelerated_cross_n.py "
                        f"--topology {topology} --train-n {n} --p-layers {p_layers} "
                        f"--h-min 0.30 --h-max 2.00 --h-points 35"
                    ),
                ))
                priority += 1

    # ── DQPT Recommendations ─────────────────────────────────────────────────
    if focus in (None, "dqpt"):
        for n in sorted(report.dqpt_missing_n):
            est_time = 1.0 + n * 0.5  # N=20 takes ~11 min
            recs.append(Recommendation(
                priority=priority,
                category="dqpt",
                n_qubits=n,
                h_pre=0.5,
                h_post=2.0,
                reason=f"N={n} has no DQPT trajectory — needed for L(t) scaling analysis",
                estimated_time_min=est_time,
                command=(
                    f".venv/bin/python scripts/experiment_runners/scaling/"
                    f"run_quench_dynamics_study.py "
                    f"--section 4 --n-qubits {n} --topology {topology} "
                    f"--dqpt-h-pre 0.5 --dqpt-h-post 2.0 --dqpt-dt 0.05 --dqpt-steps 80"
                ),
            ))
            priority += 1

        # Also recommend reverse-direction quench (h=3.0→0.5) for hardware prep validation
        if focus == "dqpt":
            for n in [10, 14, 20]:
                if n <= 22:
                    recs.append(Recommendation(
                        priority=priority,
                        category="dqpt",
                        n_qubits=n,
                        h_pre=3.0,
                        h_post=0.5,
                        reason=f"N={n} reverse quench (h=3.0→0.5) — validates hardware config "
                               f"(GNN works best at h=3.0)",
                        estimated_time_min=1.0 + n * 0.5,
                        command=(
                            f".venv/bin/python scripts/experiment_runners/scaling/"
                            f"run_quench_dynamics_study.py "
                            f"--section 4 --n-qubits {n} --topology {topology} "
                            f"--dqpt-h-pre 3.0 --dqpt-h-post 0.5 --dqpt-dt 0.05 --dqpt-steps 80"
                        ),
                    ))
                    priority += 1

    # ── Multi-h DQPT for QPT harvesting ──────────────────────────────────────
    # Running DQPTs at different h_pre values enriches E(h) for QPT for free
    if focus in (None, "qpt"):
        for n in report.qpt_gaps:
            if n > 22 or n in fss_critical_n:
                continue
            # Suggest DQPT sweep at multiple h_pre to harvest E(h_pre) data
            h_values_for_sweep = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
            recs.append(Recommendation(
                priority=priority + 5,  # Lower priority — bonus data
                category="qpt_via_dqpt",
                n_qubits=n,
                h_values=h_values_for_sweep,
                reason=f"N={n}: run DQPTs at h_pre={h_values_for_sweep} to harvest E(h) for QPT "
                       f"(free ground truth at each h_pre)",
                estimated_time_min=len(h_values_for_sweep) * (1.0 + n * 0.3),
                command=(
                    f"for H in {' '.join(f'{h:.1f}' for h in h_values_for_sweep)}; do\n"
                    f"    .venv/bin/python scripts/experiment_runners/scaling/"
                    f"run_quench_dynamics_study.py \\\n"
                    f"        --section 4 --n-qubits {n} --topology {topology} \\\n"
                    f"        --dqpt-h-pre $H --dqpt-h-post 2.0 --dqpt-dt 0.05 --dqpt-steps 60\n"
                    f"done"
                ),
            ))

    # Sort by priority and cap
    recs.sort(key=lambda r: r.priority)
    return recs[:max_recommendations]


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def print_coverage(report: CoverageReport) -> None:
    """Print coverage analysis."""
    print(f"\n{'='*70}")
    print(f"  Coverage Report: {report.topology}")
    print(f"{'='*70}")

    print(f"\n  ── QPT Data Coverage ──")
    print(f"  N values with any data: {report.qpt_n_values}")
    print(f"  N values RELIABLE (>8 pts, h in [0.3, 1.5+]): {report.qpt_reliable_n}")
    print(f"  N values with GAPS: {report.qpt_gaps}")
    print(f"  FSS quality: R²={report.fss_quality:.4f} ({report.fss_n_points} points)")

    if report.qpt_h_coverage:
        print(f"\n  {'N':>3} | {'h_min':>5} | {'h_max':>5} | {'n_pts':>5} | Status")
        print(f"  {'-'*45}")
        for n in sorted(report.qpt_h_coverage.keys()):
            h_min, h_max, n_pts = report.qpt_h_coverage[n]
            reliable = n in report.qpt_reliable_n
            status = "RELIABLE" if reliable else "GAP"
            if h_min > 2.0:
                status = "NO QPT DATA"
            print(f"  {n:>3} | {h_min:>5.2f} | {h_max:>5.2f} | {n_pts:>5} | {status}")

    print(f"\n  ── DQPT Coverage ──")
    print(f"  N with trajectories: {report.dqpt_n_values}")
    print(f"  N missing (needed): {report.dqpt_missing_n}")


def print_recommendations(recs: list[Recommendation], generate_commands: bool = False) -> None:
    """Print recommendations."""
    if not recs:
        print("\n  No recommendations — coverage is complete!")
        return

    print(f"\n{'─'*70}")
    print(f"  Recommendations (highest value first):")
    print(f"{'─'*70}")

    total_time = 0.0
    for rec in recs:
        sym = {"qpt": "📊", "dqpt": "⚡", "fss": "📈", "qpt_via_dqpt": "♻️"}.get(
            rec.category, "•"
        )
        print(f"\n  {sym} [{rec.category.upper()}] N={rec.n_qubits} (~{rec.estimated_time_min:.0f} min)")
        print(f"     {rec.reason}")
        if generate_commands:
            print(f"     $ {rec.command}")
        total_time += rec.estimated_time_min

    print(f"\n  Total estimated time: ~{total_time:.0f} min ({total_time/60:.1f} h)")

    if generate_commands:
        print(f"\n{'─'*70}")
        print(f"  All commands (copy-paste ready):")
        print(f"{'─'*70}")
        for rec in recs:
            print(f"\n# {rec.category.upper()} N={rec.n_qubits}: {rec.reason[:60]}")
            print(rec.command)


def generate_validation_report_md(topology: str, p_layers: int = 1) -> str:
    """Generate a markdown validation report with auto-populated tables.

    Reads live data from QPT detection, DQPT validation, and go/no-go evaluation
    to produce tables matching the format of heavy_hex_qpt_dqpt_validation_report.md.

    Parameters
    ----------
    topology : str
        Target topology.
    p_layers : int
        HVA depth.

    Returns
    -------
    str
        Complete markdown report.
    """
    from datetime import datetime

    lines = []
    lines.append(f"# {topology} QPT/DQPT Validation Report (Auto-Generated)")
    lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Topology**: {topology}")
    lines.append("")

    # ── QPT Table ────────────────────────────────────────────────────────────
    lines.append("## QPT Detection")
    lines.append("")
    try:
        from scripts.analysis.qpt_detection import run_qpt_analysis

        qpt = run_qpt_analysis(topology, p_layers, use_predicted=False)
        if "error" not in qpt:
            lines.append("| N | h_c detectado | n_puntos | Rango h | Status |")
            lines.append("|---|---|---|---|---|")

            per_n = qpt.get("per_n_results", {})
            reliable = set(str(n) for n in qpt.get("n_values_reliable", []))

            for n_str in sorted(per_n.keys(), key=lambda x: int(x)):
                info = per_n[n_str]
                h_c = info["h_c"]
                n_pts = info["n_points"]
                h_range = info["h_range"]
                is_edge = info.get("edge_artifact", False)

                if n_str in reliable:
                    status = "RELIABLE"
                elif is_edge:
                    status = "EDGE"
                else:
                    status = "GAP"

                lines.append(
                    f"| {n_str} | {h_c:.3f} | {n_pts} | "
                    f"[{h_range[0]:.2f}, {h_range[1]:.2f}] | {status} |"
                )

            # FSS summary
            fss = qpt.get("finite_size_scaling")
            if fss and "error" not in fss:
                lines.append(f"\n**FSS**: h_c(∞) = {fss['h_c_inf']:.4f} ± {fss['h_c_inf_err']:.4f}, "
                             f"R² = {fss['r_squared']:.4f}, N used: {fss['N_values']}")
            lines.append(f"\n**Reliable N**: {qpt.get('n_values_reliable', [])}")
        else:
            lines.append(f"*Error: {qpt['error']}*")
    except Exception as e:
        lines.append(f"*QPT analysis failed: {e}*")

    lines.append("")

    # ── DQPT Table ───────────────────────────────────────────────────────────
    lines.append("## DQPT Validation")
    lines.append("")
    try:
        from scripts.analysis.validate_dqpt_results import validate_dqpt_topology

        dqpt = validate_dqpt_topology(topology)
        if dqpt.n_trajectories > 0:
            lines.append("| N | DQPTs | t*_1 | L_min | r_max | S_max | Status |")
            lines.append("|---|---|---|---|---|---|---|")

            for n in sorted(dqpt.per_n_results.keys()):
                info = dqpt.per_n_results[n]
                t_star = info["critical_times"][0] if info["critical_times"] else None
                checks = info["checks"]
                n_pass = sum(checks.values())
                n_total = len(checks)
                status = f"PASS ({n_pass}/{n_total})" if n_pass == n_total else f"PARTIAL ({n_pass}/{n_total})"
                t_str = f"{t_star:.3f}" if t_star else "—"

                lines.append(
                    f"| {n} | {info['n_dqpts']} | {t_str} | "
                    f"{info['L_min']:.4f} | {info['r_max']:.3f} | "
                    f"{info['S_max']:.3f} | {status} |"
                )

            lines.append(f"\n**Overall**: {'PASS' if dqpt.overall_pass else 'FAIL'} "
                         f"({sum(1 for c in dqpt.checks if c.passed)}/{len(dqpt.checks)} global checks)")
        else:
            lines.append("*No DQPT trajectories found.*")
    except Exception as e:
        lines.append(f"*DQPT validation failed: {e}*")

    lines.append("")

    # ── Go/No-Go Table ───────────────────────────────────────────────────────
    lines.append("## Go/No-Go Evaluation")
    lines.append("")
    try:
        from scripts.analysis.validate_dqpt_results import compute_go_no_go

        gng = compute_go_no_go(topology, p_layers)

        lines.append(f"**Overall**: {'GO' if gng['overall_go'] else 'NO-GO'} "
                     f"({gng['n_passed']}/{gng['n_total']} criteria pass)")
        lines.append("")
        lines.append("| Criterion | Threshold | Current Value | Status |")
        lines.append("|---|---|---|---|")

        for c in gng["criteria"]:
            sym = "PASS" if c["passed"] else "FAIL"
            lines.append(f"| {c['criterion']} | {c['threshold']} | {c['current_value']} | {sym} |")

        if gng["blocking_issues"]:
            lines.append("\n**Blocking Issues**:")
            for issue in gng["blocking_issues"]:
                lines.append(f"- {issue}")
    except Exception as e:
        lines.append(f"*Go/No-Go evaluation failed: {e}*")

    lines.append("")

    # ── Coverage & Recommendations ───────────────────────────────────────────
    lines.append("## Data Coverage & Next Steps")
    lines.append("")
    try:
        report = analyze_coverage(topology, p_layers)
        lines.append(f"- **QPT reliable N**: {report.qpt_reliable_n}")
        lines.append(f"- **QPT gaps**: {report.qpt_gaps[:10]}")
        lines.append(f"- **DQPT trajectories**: N={report.dqpt_n_values}")
        lines.append(f"- **DQPT missing**: N={report.dqpt_missing_n}")
        lines.append(f"- **FSS quality**: R²={report.fss_quality:.4f} ({report.fss_n_points} points)")

        recs = generate_recommendations(topology, p_layers, max_recommendations=5)
        if recs:
            lines.append("\n### Top Recommendations")
            lines.append("")
            for rec in recs[:5]:
                lines.append(f"1. **[{rec.category.upper()}] N={rec.n_qubits}** (~{rec.estimated_time_min:.0f} min): {rec.reason}")
    except Exception as e:
        lines.append(f"*Coverage analysis failed: {e}*")

    lines.append("")

    # ── Fidelity Section ─────────────────────────────────────────────────────
    lines.append("## GNN Fidelity (hardware readiness)")
    lines.append("")
    try:
        from scripts.analysis.evaluate_gnn_fidelity import evaluate_direct_fidelity

        f_result = evaluate_direct_fidelity(topology, 10, 3.0, p_layers)
        if f_result is not None:
            lines.append(f"- **F(N=10, h=3.0)** = {f_result.fidelity:.4f} (direct statevector)")
            lines.append(f"- **ΔE/gap** = {f_result.de_gap:.4f}")
            lines.append(f"- **F_min** = 0.50 (from fidelity threshold analysis)")
            lines.append(f"- **Hardware viable**: {'YES' if f_result.fidelity > 0.50 else 'NO'}")
        else:
            lines.append("*No MPNN available for fidelity evaluation*")
    except Exception as e:
        lines.append(f"*Fidelity evaluation failed: {e}*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Active Learning Advisor — recommends next data to generate"
    )
    parser.add_argument(
        "--topology", type=str, default="heavy_hex",
        help="Target topology (default: heavy_hex)",
    )
    parser.add_argument(
        "--p-layers", type=int, default=1,
        help="HVA depth (default: 1)",
    )
    parser.add_argument(
        "--focus", type=str, choices=["qpt", "dqpt"], default=None,
        help="Focus on QPT or DQPT recommendations only",
    )
    parser.add_argument(
        "--generate-commands", action="store_true",
        help="Output ready-to-run commands",
    )
    parser.add_argument(
        "--auto-report", action="store_true",
        help="Generate validation report markdown (auto-populates QPT/DQPT tables + go/no-go)",
    )
    parser.add_argument(
        "--max-recs", type=int, default=10,
        help="Maximum recommendations (default: 10)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    # Auto-report mode: generate markdown and exit
    if args.auto_report:
        md = generate_validation_report_md(args.topology, args.p_layers)
        print(md)
        # Also save to file
        out_path = _project_root / "results" / "analysis" / f"validation_report_{args.topology}.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(md)
        print(f"\n  Saved: {out_path}")
        return

    # Analyze coverage
    report = analyze_coverage(args.topology, args.p_layers)
    print_coverage(report)

    # Generate recommendations
    recs = generate_recommendations(
        args.topology, args.p_layers,
        focus=args.focus,
        max_recommendations=args.max_recs,
    )
    print_recommendations(recs, generate_commands=args.generate_commands)


if __name__ == "__main__":
    main()
