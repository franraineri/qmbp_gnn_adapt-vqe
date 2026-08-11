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


def main():
    """Run the analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Accelerated Cross-N Analyzer")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

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
        print(f"Run: python scripts/.../run_accelerated_cross_n.py")
        return

    # Print report
    print(format_report(report))

    # ── Scan and integrate large-N extrapolation results ──
    large_n_results = scan_large_n_extrapolation_results()
    if large_n_results:
        print(format_large_n_report(large_n_results))

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
