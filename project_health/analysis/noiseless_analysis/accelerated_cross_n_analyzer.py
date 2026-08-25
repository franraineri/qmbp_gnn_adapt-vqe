#!/usr/bin/env python3
"""Accelerated Cross-N Analyzer — Post-run analysis for accelerated experiments.

Scans results from ACCEL_CROSS_N experiments and produces:
- Per-h ΔE/gap breakdown (which h-regions work, which don't)
- Cross-N scaling analysis (does error grow with N_target?)
- Model reuse effectiveness (zoo hit rate, time savings)
- Training data utilization (anchor efficiency)
- Comparison: accelerated vs full VQE (if both exist)

Usage:
    python -m project_health.analysis.accelerated_cross_n_analyzer
    python -m project_health.analysis.accelerated_cross_n_analyzer --verbose
    python -m project_health.analysis.accelerated_cross_n_analyzer --compare-full
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "experiments"


@dataclass
class CrossNAnalysis:
    """Analysis summary for one cross-N prediction run."""

    train_n: int
    target_n: int
    p_layers: int
    topology: str
    n_points: int = 0
    pass_rate_5pct: float = 0.0
    pass_rate_10pct: float = 0.0
    mean_de_gap: float = 0.0
    mean_abs_error: float = 0.0
    time_s: float = 0.0
    # h-region breakdown
    h_easy_pass_rate: float = 0.0   # h > 2.5
    h_medium_pass_rate: float = 0.0  # 2.0 < h <= 2.5
    h_hard_pass_rate: float = 0.0    # h <= 2.0


@dataclass
class AcceleratedReport:
    """Complete analysis report."""

    analyses: list[CrossNAnalysis] = field(default_factory=list)
    zoo_entries_used: int = 0
    total_training_time_s: float = 0.0
    total_prediction_time_s: float = 0.0
    data_reuse_summary: dict[str, Any] = field(default_factory=dict)


def scan_results() -> list[dict[str, Any]]:
    """Find all ACCEL_CROSS_N results in the experiments directory."""
    results = []
    patterns = [
        RESULTS_DIR / "exp_accel_cross_n",
        RESULTS_DIR / "exp_accelerated_cross_n",
        RESULTS_DIR / "exp_mpnn_warmstart_accelerator",
    ]
    for base in patterns:
        if not base.exists():
            continue
        for f in sorted(base.rglob("run_*.json")):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                results.append(data)
            except (json.JSONDecodeError, OSError):
                continue
    return results


def analyze_cross_n_result(data: dict) -> list[CrossNAnalysis]:
    """Extract cross-N analysis from a single result file."""
    analyses = []
    results_section = data.get("results", {})

    # Look for section 3 (cross-N predict) data
    for key, section in results_section.items():
        if not isinstance(section, dict):
            continue
        section_data = section.get("data", {})
        cross_n = section_data.get("cross_n_results", {})

        for config_key, result in cross_n.items():
            if not isinstance(result, dict) or "per_point" not in result:
                continue

            per_point = result["per_point"]
            if not per_point:
                continue

            # h-region breakdown
            easy = [r for r in per_point if r["h"] > 2.5]
            medium = [r for r in per_point if 2.0 < r["h"] <= 2.5]
            hard = [r for r in per_point if r["h"] <= 2.0]

            analysis = CrossNAnalysis(
                train_n=result.get("train_n", 0),
                target_n=result.get("target_n", 0),
                p_layers=result.get("p_layers", 1),
                topology=data.get("config", {}).get("topology", "chain_1d"),
                n_points=len(per_point),
                pass_rate_5pct=result.get("pass_rate_5pct", 0.0),
                pass_rate_10pct=result.get("pass_rate_10pct", 0.0),
                mean_de_gap=result.get("mean_de_gap", 0.0),
                mean_abs_error=result.get("mean_abs_error", 0.0),
                time_s=result.get("elapsed_s", 0.0),
                h_easy_pass_rate=(
                    np.mean([1.0 if r["de_gap"] < 0.05 else 0.0 for r in easy])
                    if easy else 0.0
                ),
                h_medium_pass_rate=(
                    np.mean([1.0 if r["de_gap"] < 0.05 else 0.0 for r in medium])
                    if medium else 0.0
                ),
                h_hard_pass_rate=(
                    np.mean([1.0 if r["de_gap"] < 0.05 else 0.0 for r in hard])
                    if hard else 0.0
                ),
            )
            analyses.append(analysis)

    return analyses


def format_report(report: AcceleratedReport) -> str:
    """Format the analysis report as text, grouped by topology."""
    from collections import defaultdict

    lines = [
        "═" * 60,
        "  ACCELERATED CROSS-N ANALYSIS REPORT",
        "═" * 60,
        "",
        f"Total runs analyzed: {len(report.analyses)}",
        f"Training time total: {report.total_training_time_s:.0f}s",
        f"Prediction time total: {report.total_prediction_time_s:.0f}s",
    ]

    # Group by topology
    by_topo: dict[str, list] = defaultdict(list)
    for a in report.analyses:
        by_topo[a.topology].append(a)

    for topo in sorted(by_topo.keys()):
        analyses = sorted(by_topo[topo], key=lambda x: (x.p_layers, x.target_n))
        lines.extend([
            "",
            f"{'─' * 60}",
            f"  [{topo.upper()}] ({len(analyses)} configs)",
            f"{'─' * 60}",
            "",
            f"  {'Config':<28} {'@5%':<7} {'@10%':<7} {'mean ΔE/gap':<12} {'h>2.5':<7} {'2.0<h<2.5':<10}",
            "  " + "-" * 75,
        ])

        for a in analyses:
            config = f"N={a.train_n}→{a.target_n} p={a.p_layers}"
            lines.append(
                f"  {config:<28} {a.pass_rate_5pct:.0%}     {a.pass_rate_10pct:.0%}     "
                f"{a.mean_de_gap:.4f}       {a.h_easy_pass_rate:.0%}     "
                f"{a.h_medium_pass_rate:.0%}"
            )

        # Per-topology key findings
        best = max(analyses, key=lambda a: (a.pass_rate_10pct, -a.mean_de_gap))
        lines.extend([
            "",
            f"  Best: N={best.train_n}→{best.target_n} "
            f"(@10%={best.pass_rate_10pct:.0%}, mean={best.mean_de_gap:.4f})",
        ])

        # N-scaling trend for this topology
        n_targets = sorted(set(a.target_n for a in analyses))
        if len(n_targets) > 1:
            lines.append("  N-scaling:")
            for n in n_targets:
                subset = [a for a in analyses if a.target_n == n]
                avg_de_gap = np.mean([a.mean_de_gap for a in subset])
                best_pr = max(a.pass_rate_10pct for a in subset)
                lines.append(f"    N={n:>2}: mean ΔE/gap={avg_de_gap:.4f}, best@10%={best_pr:.0%}")

        # n_max_viable for this topology
        viable = [a for a in analyses if a.pass_rate_10pct > 0.5]
        n_max = max((a.target_n for a in viable), default=None)
        if n_max:
            lines.append(f"  n_max_viable (pass@10%>50%): N={n_max}")
        else:
            lines.append("  n_max_viable: NONE (no config passes @10%>50%)")

    lines.extend(["", "═" * 60])
    return "\n".join(lines)


def analyze_from_dashboard() -> str:
    """Deep cross-topology analysis using the dashboard as primary source.

    This is the recommended entry point for thesis-ready analysis.
    Uses dashboard (NPZ-derived, dual criterion) for training data quality,
    and raw large-N NPZ for extrapolation per-h breakdown.

    Returns formatted text report.
    """
    dashboard_path = ROOT / "data" / "model_quality_dashboard.json"
    large_n_dir = ROOT / "data" / "large_n_extrapolation"
    exp_dir = RESULTS_DIR / "exp_large_n_extrap"

    if not dashboard_path.exists():
        return "Dashboard not found. Run any experiment to generate."

    dashboard = json.load(open(dashboard_path))
    configs = dashboard.get("configs", [])
    topo_sum = dashboard.get("topology_summary", {})
    audit = dashboard.get("audit", {})

    lines = [
        "",
        "═" * 70,
        "  UNIFIED PIPELINE ANALYSIS (from dashboard + large-N NPZ)",
        f"  Dashboard generated: {dashboard.get('generated_at', '?')[:19]}",
        "  Criterion: pass_rate_dual (ΔE/gap < 5% AND |ΔE| < 0.10)",
        "═" * 70,
    ]

    # ── Section A: Training Data Overview (from dashboard) ────────────────
    lines.extend(["", "  ┌── A. Training Data (dashboard, N≤20) ──────────────────────────┐", ""])
    lines.append(f"  {'Topology':<12} {'N values':<20} {'pts':>4} {'dual%':>6} {'5pct%':>6} "
                 f"{'gap_mask':>8} {'h_front':>8} {'quality':>8}")
    lines.append("  " + "─" * 68)

    for topo in sorted(topo_sum.keys()):
        topo_configs = sorted(
            [c for c in configs if c["topology"] == topo],
            key=lambda c: c["n_qubits"],
        )
        n_values = [c["n_qubits"] for c in topo_configs]
        total_pts = sum(c["n_points"] for c in topo_configs)
        best_dual = max((c["pass_rate_dual_criterion"] for c in topo_configs), default=0)
        best_5pct = max((c["pass_rate_5pct"] for c in topo_configs), default=0)
        gap_mask_pct = best_5pct - best_dual
        h_frontiers = [c["h_frontier"] for c in topo_configs if c.get("h_frontier")]
        h_front_str = f"{min(h_frontiers):.2f}" if h_frontiers else "—"
        utility = sum(1 for c in topo_configs if c.get("training_utility") == "useful")
        quality_str = f"{utility}/{len(topo_configs)}"

        n_str = ",".join(str(n) for n in n_values[:6])
        if len(n_values) > 6:
            n_str += "..."

        lines.append(
            f"  {topo:<12} {n_str:<20} {total_pts:>4} {best_dual:>5.0%} {best_5pct:>5.0%} "
            f"{gap_mask_pct:>7.0%} {h_front_str:>8} {quality_str:>8}"
        )

    # ── Section B: Per-N Scaling (from dashboard) ─────────────────────────
    lines.extend(["", "  ┌── B. Scaling: pass_rate_dual per (Topology, N) ────────────────┐", ""])
    all_n = sorted(set(c["n_qubits"] for c in configs))
    useful_n = [n for n in all_n if sum(1 for c in configs if c["n_qubits"] == n) >= 2][:8]

    header = f"  {'Topo':<12}" + "".join(f" N={n:>2}" for n in useful_n)
    lines.append(header)
    lines.append("  " + "─" * (12 + 5 * len(useful_n)))
    for topo in sorted(topo_sum.keys()):
        row = f"  {topo:<12}"
        for n in useful_n:
            match = next((c for c in configs if c["topology"] == topo and c["n_qubits"] == n), None)
            if match is None:
                row += "   — "
            else:
                val = match["pass_rate_dual_criterion"]
                if val >= 0.80:
                    row += f" {val:>3.0%}✓"
                elif val >= 0.50:
                    row += f" {val:>3.0%}~"
                else:
                    row += f" {val:>3.0%}✗"
        lines.append(row)

    # ── Section C: Large-N Extrapolation (from NPZ raw) ───────────────────
    if large_n_dir.exists() and list(large_n_dir.glob("*.npz")):
        lines.extend(["", "  ┌── C. Large-N Extrapolation (per-h from NPZ) ────────────────────┐", ""])
        lines.append(f"  {'Topo':<12} {'N':>4} {'h-range':<12} {'pts':>3} "
                     f"{'pass':>5} {'g.mask':>6} {'fail':>4} {'|ΔE|/N':>9} {'scaling':>10}")
        lines.append("  " + "─" * 68)

        topo_scaling = defaultdict(list)

        for npz_file in sorted(large_n_dir.glob("*.npz")):
            try:
                parts = npz_file.stem.rsplit("_", 2)
                if len(parts) < 3 or not parts[1].startswith("N"):
                    continue
                topo = parts[0]
                n_val = int(parts[1][1:])

                data = np.load(str(npz_file), allow_pickle=True)
                h_vals = data["h_values"]
                n_pts = len(h_vals)
                if n_pts == 0:
                    continue
                e_pred = data.get("e_pred", data.get("e_vqe"))
                e_exact = data["e_exact"]
                gaps = data.get("gaps", np.ones(n_pts))
                abs_err = np.abs(e_pred - e_exact)
                de_gaps_arr = abs_err / np.maximum(gaps, 1e-10)

                n_pass = int(np.sum((de_gaps_arr < 0.05) & (abs_err < 0.10)))
                n_gm = int(np.sum((de_gaps_arr < 0.05) & (abs_err >= 0.10)))
                n_fail = int(np.sum(de_gaps_arr >= 0.05))
                per_site = float(abs_err.mean() / max(n_val, 1))

                topo_scaling[topo].append((n_val, per_site))

                h_range = f"[{h_vals.min():.1f},{h_vals.max():.1f}]"
                lines.append(
                    f"  {topo:<12} {n_val:>4} {h_range:<12} {n_pts:>3} "
                    f"{n_pass:>5} {n_gm:>6} {n_fail:>4} {per_site:>9.2e}"
                )
            except Exception:
                continue

        # Scaling verdict per topology
        lines.append("")
        for topo in sorted(topo_scaling.keys()):
            points = sorted(topo_scaling[topo])
            if len(points) < 2:
                continue
            per_sites = [p[1] for p in points]
            variation = max(per_sites) / max(min(per_sites), 1e-10)
            if variation < 3.0:
                verdict = "✅ extensive"
            elif variation < 5.0:
                verdict = "⚠️ degrading"
            else:
                verdict = "❌ non-extensive"
            lines.append(f"  {topo}: |ΔE|/N variation={variation:.1f}× → {verdict}")

    # ── Section D: Audit Issues (from dashboard) ──────────────────────────
    if audit:
        lines.extend(["", "  ┌── D. Audit Issues ──────────────────────────────────────────────┐", ""])
        n_issues = audit.get("n_issues", 0)
        lines.append(f"  Total issues: {n_issues}")
        if audit.get("h_frontier_anomalies"):
            lines.append(f"    h_frontier anomalies: {audit['h_frontier_anomalies']}")
        if audit.get("training_zoo_incoherence"):
            lines.append(f"    Training/zoo incoherence: {audit['training_zoo_incoherence']}")
        gap_masked = audit.get("gap_masked_configs", 0)
        if gap_masked:
            lines.append(f"    Gap-masked configs: {gap_masked}")

    # ── Section E: Speedup Data (shared helper) ─────────────────────────
    speedups = _scan_speedup_data()

    if speedups:
        lines.extend(["", "  ┌── E. QPU Speedup (MPNN vs VQE random-init) ────────────────────┐", ""])
        for (topo, n), spd in sorted(speedups.items()):
            lines.append(f"    {topo:<12} N={n:>3}: {spd:>8,.0f}×")

    # ── Section F: Failure Mode Diagnostics (Tests A-F) ───────────────────
    from qmbp_simulation.analysis.metrics import (
        classify_topology_failure_mode, diagnose_gap_masking,
    )

    lines.extend(["", "  ┌── F. Failure Mode Diagnostics (Tests A-F) ──────────────────────┐", ""])

    for topo in sorted(topo_sum.keys()):
        topo_configs = [c for c in configs if c["topology"] == topo]

        # Build extrapolation_data dict for this topology from large-N NPZ
        extrap_per_n: dict[int, dict] = {}
        if large_n_dir.exists():
            for npz_file in sorted(large_n_dir.glob(f"{topo}_N*_p*.npz")):
                try:
                    parts = npz_file.stem.rsplit("_", 2)
                    if len(parts) < 3 or not parts[1].startswith("N"):
                        continue
                    n_val = int(parts[1][1:])
                    data = np.load(str(npz_file), allow_pickle=True)
                    h_vals = data["h_values"]
                    e_pred = data.get("e_pred", data.get("e_vqe"))
                    e_exact = data["e_exact"]
                    gaps = data.get("gaps", np.ones(len(h_vals)))
                    abs_err = np.abs(e_pred - e_exact)
                    extrap_per_n[n_val] = {
                        "h_values": h_vals,
                        "abs_errors": abs_err,
                        "e_pred": e_pred,
                        "e_exact": e_exact,
                        "gaps": gaps,
                    }
                except Exception:
                    continue

        # Run unified classifier
        diag = classify_topology_failure_mode(
            topo, topo_configs,
            extrapolation_data=extrap_per_n if extrap_per_n else None,
        )

        mode_icon = {
            "healthy": "✅",
            "gap_masking": "🔵",
            "generalization_failure": "🔴",
            "mixed": "🟡",
            "intrinsic_vqe_error": "⚠️",
            "unknown": "❓",
        }.get(diag.primary_mode, "❓")

        lines.append(
            f"  {mode_icon} {topo:<12} [{diag.primary_mode}] "
            f"(confidence={diag.confidence:.0%})"
        )
        lines.append(f"    {diag.explanation}")

        # Show test details if non-healthy
        if diag.primary_mode != "healthy":
            if diag.violation_rate is not None:
                lines.append(f"    Variational violations: {diag.violation_rate:.0%}")
            if diag.slope_b is not None and diag.fit_r_squared is not None:
                lines.append(f"    |ΔE|/N trend: slope={diag.slope_b:.2e}, R²={diag.fit_r_squared:.2f}")
            if diag.cross_n_per_site_ratios:
                ratios_str = ", ".join(f"{k}={v:.1f}×" for k, v in diag.cross_n_per_site_ratios.items())
                lines.append(f"    Cross-N per-site ratios: {ratios_str}")
            # Tests G-I
            if diag.per_site_verified is not None:
                lines.append(
                    f"    VQE quality: verified={diag.per_site_verified:.2e}/site, "
                    f"approx={diag.per_site_approximate:.2e}/site, "
                    f"ratio={diag.verified_vs_approx_ratio:.2f}"
                )
            if diag.verified_high_error_fraction is not None and diag.verified_high_error_fraction > 0:
                lines.append(
                    f"    Verified with high error: {diag.verified_high_error_fraction:.0%} "
                    f"(best N={diag.best_n}, per-site={diag.best_n_per_site:.2e})"
                )
            # Tests J-L
            if diag.training_gap_masked_fraction is not None and diag.training_gap_masked_fraction > 0.05:
                lines.append(
                    f"    Training contamination: {diag.training_gap_masked_fraction:.0%} "
                    f"points gap-masked, θ_smoothness={diag.max_theta_smoothness:.2f}, "
                    f"{diag.n_configs_discontinuous} discontinuous configs"
                )
            if diag.zoo_single_vs_dual_gap is not None and diag.zoo_single_vs_dual_gap > 0.10:
                lines.append(
                    f"    Zoo inflation: zoo_pass - best_dual = {diag.zoo_single_vs_dual_gap:+.0%}"
                )
        lines.append("")

    lines.extend(["═" * 70])
    return "\n".join(lines)


def main():
    """Run the analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Accelerated Cross-N Analyzer")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--deep", action="store_true",
        help="Run unified deep analysis from dashboard (thesis-ready)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    # ── Deep analysis mode: uses dashboard as primary source ──────────────
    if args.deep:
        print(analyze_from_dashboard())
        return

    # ── Standard mode: scan result JSONs ──────────────────────────────────
    # Scan results
    raw_results = scan_results()
    logger.info(f"Found {len(raw_results)} result files")

    # Analyze
    report = AcceleratedReport()
    for data in raw_results:
        analyses = analyze_cross_n_result(data)
        report.analyses.extend(analyses)

        # Timing
        elapsed = data.get("elapsed_s", 0)
        config = data.get("config", {})
        if config.get("from_zoo"):
            report.total_prediction_time_s += elapsed
            report.zoo_entries_used += 1
        else:
            report.total_training_time_s += elapsed

    if not report.analyses:
        print("No accelerated cross-N results found.")
        print(f"Run: .venv/bin/python scripts/.../run_accelerated_cross_n.py")
        return

    # Print report
    print(format_report(report))

    # ── Scan and integrate large-N extrapolation results ──
    large_n_results = scan_large_n_extrapolation_results()
    if large_n_results:
        print(format_large_n_report(large_n_results))
        print(analyze_large_n_scaling(large_n_results))

    # ── Dashboard cross-validation: compare our scan vs cached dashboard ──
    _cross_validate_with_dashboard(report)


# ═══════════════════════════════════════════════════════════════════════════════
# Large-N Extrapolation Integration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LargeNResult:
    """Summary of a large-N extrapolation run."""

    topology: str
    target_n: int
    p_layers: int
    n_points: int
    pass_rate_dual: float
    mean_de_gap: float
    mean_abs_error_per_site: float
    method: str  # 'mpnn' or 'random_vqe'


def scan_large_n_extrapolation_results() -> list[LargeNResult]:
    """Scan for large-N extrapolation results from run_large_n_extrapolation.py."""
    results = []
    
    # Check both results directory and NPZ data directory
    large_n_dir = ROOT / "data" / "large_n_extrapolation"
    exp_dir = RESULTS_DIR / "exp_large_n_extrap"
    
    # 1. Scan NPZ files for cached predictions
    if large_n_dir.exists():
        for npz_file in sorted(large_n_dir.glob("*.npz")):
            try:
                data = np.load(str(npz_file), allow_pickle=True)
                # Parse filename: {topology}_N{n}_p{p}.npz
                parts = npz_file.stem.split("_")
                n_idx = next((i for i, p in enumerate(parts) if p.startswith("N")), None)
                p_idx = next((i for i, p in enumerate(parts) if p.startswith("p")), None)
                if n_idx is None or p_idx is None:
                    continue
                topo = "_".join(parts[:n_idx])
                n_val = int(parts[n_idx][1:])
                p_val = int(parts[p_idx][1:])
                
                # Compute metrics
                h_vals = data["h_values"]
                e_pred = data.get("e_pred", data.get("e_vqe"))
                e_exact = data["e_exact"]
                gaps = data.get("gaps", np.ones(len(h_vals)))
                
                abs_err = np.abs(e_pred - e_exact)
                de_gaps = abs_err / np.maximum(gaps, 1e-10)
                
                # Dual criterion pass rate
                n_pass = int(np.sum((de_gaps < 0.05) & (abs_err < 0.10)))
                
                results.append(LargeNResult(
                    topology=topo,
                    target_n=n_val,
                    p_layers=p_val,
                    n_points=len(h_vals),
                    pass_rate_dual=n_pass / max(len(h_vals), 1),
                    mean_de_gap=float(de_gaps.mean()),
                    mean_abs_error_per_site=float(abs_err.mean() / max(n_val, 1)),
                    method="npz_cache",
                ))
            except Exception as e:
                logger.debug(f"Error reading {npz_file}: {e}")
    
    # 2. Scan experiment results for JSON reports
    if exp_dir.exists():
        for f in sorted(exp_dir.glob("run_*.json")):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                
                config = data.get("config", {})
                topo = config.get("topology", "chain_1d")
                p = config.get("p_layers", 1)
                
                # Extract mpnn_results
                results_sec = data.get("results", {})
                for sec_key, section in results_sec.items():
                    if not isinstance(section, dict):
                        continue
                    sec_data = section.get("data", {})
                    mpnn_res = sec_data.get("mpnn_results", {})
                    
                    for n_str, res in mpnn_res.items():
                        if not isinstance(res, dict):
                            continue
                        results.append(LargeNResult(
                            topology=topo,
                            target_n=res.get("n_qubits", int(n_str)),
                            p_layers=p,
                            n_points=res.get("n_points", 0),
                            pass_rate_dual=res.get("pass_rate_dual", 0.0),
                            mean_de_gap=res.get("mean_de_gap", 0.0),
                            mean_abs_error_per_site=res.get("mean_abs_error_per_site", 0.0),
                            method="json_result",
                        ))
            except (json.JSONDecodeError, OSError, KeyError):
                continue
    
    # Deduplicate by (topology, target_n, p_layers), keeping best pass_rate_dual
    dedup = {}
    for r in results:
        key = (r.topology, r.target_n, r.p_layers)
        if key not in dedup or r.pass_rate_dual > dedup[key].pass_rate_dual:
            dedup[key] = r
    
    return sorted(dedup.values(), key=lambda x: (x.topology, x.target_n))


def format_large_n_report(results: list[LargeNResult]) -> str:
    """Format large-N extrapolation results as text."""
    if not results:
        return ""
    
    lines = [
        "",
        "═" * 60,
        "  LARGE-N EXTRAPOLATION RESULTS",
        "═" * 60,
        "",
        f"  {'Topology':<12} {'N':<6} {'p':<3} {'pts':<5} {'dual%':<8} {'ΔE/gap':<10} {'|ΔE|/N':<10}",
        "  " + "-" * 58,
    ]
    
    for r in results:
        lines.append(
            f"  {r.topology:<12} {r.target_n:<6} {r.p_layers:<3} {r.n_points:<5} "
            f"{r.pass_rate_dual:.0%}     {r.mean_de_gap:<10.4f} {r.mean_abs_error_per_site:<10.2e}"
        )
    
    # Summary by topology
    from collections import defaultdict
    by_topo = defaultdict(list)
    for r in results:
        by_topo[r.topology].append(r)
    
    lines.extend(["", "  Key findings:"])
    for topo, topo_results in sorted(by_topo.items()):
        n_max = max((r.target_n for r in topo_results if r.pass_rate_dual > 0.5), default=None)
        if n_max:
            lines.append(f"    {topo}: extrapolates to N={n_max} (dual pass>50%)")
        else:
            best = max(topo_results, key=lambda r: r.pass_rate_dual, default=None)
            if best:
                lines.append(
                    f"    {topo}: best N={best.target_n} "
                    f"(dual={best.pass_rate_dual:.0%}, ΔE/gap={best.mean_de_gap:.4f})"
                )
    
    lines.append("═" * 60)
    return "\n".join(lines)


def _scan_speedup_data() -> dict[tuple[str, int], float]:
    """Scan experiment results for speedup data (MPNN vs random-init VQE).

    Returns dict mapping (topology, N) → best speedup factor.
    Shared helper to avoid duplicating this scan in multiple functions.
    """
    exp_dir = RESULTS_DIR / "exp_large_n_extrap"
    speedups: dict[tuple[str, int], float] = {}
    if not exp_dir.exists():
        return speedups
    for f in sorted(exp_dir.glob("run_*.json")):
        try:
            data = json.load(open(f))
            cfg = data.get("config", {})
            topo = cfg.get("topology", "")
            results_sec = data.get("results", {})
            for sec_key in ["section_3", "section_4", "section_5"]:
                sec = results_sec.get(sec_key, {})
                comp = sec.get("data", {}).get("comparison", {})
                for n_str, entry in comp.items():
                    spd = entry.get("speedup")
                    if spd and spd > 1:
                        key = (topo, int(n_str))
                        if key not in speedups or spd > speedups[key]:
                            speedups[key] = spd
        except Exception:
            continue
    return speedups


def analyze_large_n_scaling(results: list[LargeNResult]) -> str:
    """Extended analysis: failure modes, extensive scaling, speedup.

    Uses `failures_tests` module for structured diagnostics instead of
    reimplementing threshold logic inline.
    """
    if not results:
        return ""

    from qmbp_simulation.analysis.failures_tests import (
        diagnose_gap_masking,
        diagnose_generalization_failure,
        diagnose_h_range_mismatch,
        VARIATION_EXTENSIVE_MAX,
        VARIATION_DEGRADING_MAX,
    )

    large_n_dir = ROOT / "data" / "large_n_extrapolation"

    lines = [
        "",
        "═" * 60,
        "  LARGE-N SCALING ANALYSIS (thesis-ready)",
        "═" * 60,
    ]

    # ── Extensive scaling via diagnose_generalization_failure ──────────────
    by_topo: dict[str, list[LargeNResult]] = defaultdict(list)
    for r in results:
        by_topo[r.topology].append(r)

    lines.extend(["", "  Extensive Scaling (|ΔE|/N should be ~constant):", ""])
    for topo in sorted(by_topo.keys()):
        topo_r = sorted(by_topo[topo], key=lambda x: x.target_n)
        if len(topo_r) < 2:
            continue

        # Build per_n_data for diagnose_generalization_failure
        per_n_data: dict[int, dict] = {}
        for r in topo_r:
            per_n_data[r.target_n] = {
                "abs_errors": np.array([r.mean_abs_error_per_site * r.target_n]),
                "e_exact": np.zeros(1),
                "e_pred": np.array([r.mean_abs_error_per_site * r.target_n]),
            }

        # Use variation directly from per-site values for display
        per_site_errs = [r.mean_abs_error_per_site for r in topo_r]
        variation = max(per_site_errs) / max(min(per_site_errs), 1e-10)
        if variation < VARIATION_EXTENSIVE_MAX:
            trend = "✅ EXTENSIVE"
        elif variation < VARIATION_DEGRADING_MAX:
            trend = "⚠️ DEGRADING"
        else:
            trend = "❌ NON-EXTENSIVE"
        lines.append(f"  {topo}: {trend} (variation={variation:.1f}×)")
        for r in topo_r:
            bar = "█" * min(int(r.mean_abs_error_per_site * 500), 40)
            lines.append(f"    N={r.target_n:>3}: |ΔE|/N={r.mean_abs_error_per_site:.2e} {bar}")
        lines.append("")

    # ── Per-h failure mode analysis using diagnose_gap_masking ─────────────
    if large_n_dir.exists():
        lines.extend(["  Failure Mode Analysis (per-h from NPZ):", ""])
        for topo in sorted(by_topo.keys()):
            topo_failures = []
            for npz_file in sorted(large_n_dir.glob(f"{topo}_N*_p*.npz")):
                try:
                    data = np.load(str(npz_file), allow_pickle=True)
                    parts = npz_file.stem.split("_")
                    n_idx = next((i for i, p in enumerate(parts) if p.startswith("N")), None)
                    if n_idx is None:
                        continue
                    n_val = int(parts[n_idx][1:])

                    h_vals = data["h_values"]
                    e_pred = data.get("e_pred", data.get("e_vqe"))
                    e_exact = data["e_exact"]
                    gaps = data.get("gaps", np.ones(len(h_vals)))
                    abs_err = np.abs(e_pred - e_exact)
                    de_gaps_arr = abs_err / np.maximum(gaps, 1e-10)

                    # Use diagnose_gap_masking for structured breakdown
                    gm = diagnose_gap_masking(h_vals, de_gaps_arr, abs_err, n_val)

                    if gm["n_masked"] > 0 or gm["n_real_fail"] > 0:
                        topo_failures.append({
                            "n": n_val,
                            "total": len(h_vals),
                            "pass": gm["n_pass"],
                            "gap_masked": gm["n_masked"],
                            "real_fail": gm["n_real_fail"],
                            "is_gap_masking": gm["is_gap_masking"],
                        })
                except Exception:
                    continue

            if topo_failures:
                lines.append(f"  {topo}:")
                for f in sorted(topo_failures, key=lambda x: x["n"]):
                    gm_flag = " [gap-masking]" if f["is_gap_masking"] else ""
                    lines.append(
                        f"    N={f['n']:>3}: {f['pass']}/{f['total']} pass | "
                        f"{f['gap_masked']} gap-masked | {f['real_fail']} real-fail{gm_flag}"
                    )
                lines.append("")

        # H-range mismatch detection across N values
        for topo in sorted(by_topo.keys()):
            extrap_per_n: dict[int, dict] = {}
            for npz_file in sorted(large_n_dir.glob(f"{topo}_N*_p*.npz")):
                try:
                    parts = npz_file.stem.split("_")
                    n_idx = next((i for i, p in enumerate(parts) if p.startswith("N")), None)
                    if n_idx is None:
                        continue
                    n_val = int(parts[n_idx][1:])
                    data = np.load(str(npz_file), allow_pickle=True)
                    extrap_per_n[n_val] = {"h_values": data["h_values"]}
                except Exception:
                    continue
            if len(extrap_per_n) >= 2:
                hm = diagnose_h_range_mismatch(extrap_per_n)
                if hm["has_mismatch"]:
                    lines.append(f"  ⚠️ {topo}: h-range mismatch (overlap={hm['overlap_fraction']:.0%})")
                    for pair in hm["mismatch_pairs"]:
                        lines.append(f"      {pair}")
                    lines.append("")

    # ── Speedup data (shared helper) ──────────────────────────────────────
    speedups = _scan_speedup_data()
    if speedups:
        lines.extend(["  Speedup (MPNN vs random-init VQE):", ""])
        for (topo, n), spd in sorted(speedups.items()):
            lines.append(f"    {topo:<12} N={n:>3}: {spd:>7,.0f}×")
        lines.append("")

    lines.append("═" * 60)
    return "\n".join(lines)


def _cross_validate_with_dashboard(report: AcceleratedReport) -> None:
    """Cross-validate analyzer results against the model quality dashboard.

    Reads cross_n_transfers and topology_summary from the dashboard and
    compares against freshly-scanned results. Reports discrepancies.
    """
    import json
    from pathlib import Path

    dashboard_path = Path(ROOT) / "data" / "model_quality_dashboard.json"
    if not dashboard_path.exists():
        return

    try:
        with open(dashboard_path) as f:
            dashboard = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    topo_summary = dashboard.get("topology_summary", {})
    if not topo_summary:
        return

    print(f"\n{'─' * 60}")
    print("  DASHBOARD CROSS-VALIDATION")
    print(f"{'─' * 60}")

    # Compare n_max_viable from dashboard vs our fresh analysis
    # Our fresh analysis: per topology, find max target_n with pass_rate_10pct > 50%
    fresh_n_max = {}
    for a in report.analyses:
        topo = a.topology
        if a.pass_rate_10pct > 0.5:
            fresh_n_max[topo] = max(fresh_n_max.get(topo, 0), a.target_n)

    mismatches = []
    for topo, info in topo_summary.items():
        dashboard_n_max = info.get("n_max_viable")
        fresh_val = fresh_n_max.get(topo)
        # Only flag if both have data and they differ
        if dashboard_n_max is not None and fresh_val is not None:
            if dashboard_n_max != fresh_val:
                mismatches.append(
                    f"  {topo}: dashboard n_max_viable={dashboard_n_max} "
                    f"vs analyzer={fresh_val}"
                )

    # Check cross_n_best_source consistency
    configs = dashboard.get("configs", [])
    dashboard_transfers = {}
    for c in configs:
        transfers = c.get("cross_n_transfers", [])
        if transfers:
            key = (c["topology"], c["n_qubits"])
            dashboard_transfers[key] = len(transfers)

    # Our scan found these cross-N configs
    fresh_transfer_count = {}
    for a in report.analyses:
        if a.train_n != a.target_n:
            key = (a.topology, a.target_n)
            fresh_transfer_count[key] = fresh_transfer_count.get(key, 0) + 1

    for key, fresh_count in fresh_transfer_count.items():
        dash_count = dashboard_transfers.get(key, 0)
        if dash_count > 0 and fresh_count > dash_count:
            mismatches.append(
                f"  {key[0]} N={key[1]}: dashboard has {dash_count} transfers, "
                f"analyzer found {fresh_count} (dashboard may be stale)"
            )

    if not mismatches:
        print("  ✅ Dashboard consistent with fresh analysis")
        # Show topology summary from dashboard for convenience
        for topo, info in sorted(topo_summary.items()):
            n_max = info.get("n_max_viable", "—")
            best_src = info.get("cross_n_best_source_for_largest")
            src_str = f"best_source=N{best_src['train_n']}" if best_src else "no cross-N data"
            print(f"    {topo:12s}: n_max_viable={n_max}, {src_str}")
    else:
        print("  ⚠️ Discrepancies found (dashboard may need regeneration):")
        for m in mismatches:
            print(m)



if __name__ == "__main__":
    main()
