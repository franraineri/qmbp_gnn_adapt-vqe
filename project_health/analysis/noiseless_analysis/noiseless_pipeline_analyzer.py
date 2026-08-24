#!/usr/bin/env python3
"""Noiseless Pipeline Analyzer — processes exact statevector experiment results.

Scans `results/experiments/exp_noiseless/` for `run_*.json` files produced by
`scripts/experiment_runners/noiseless/run_noiseless_pipeline.py`, then:

1. Extracts per-section metrics (ExactDiag, VQE, MPNN, Deploy)
2. Groups by topology and system size
3. Computes cross-topology comparison tables
4. Identifies VQE convergence limits (where ΔE/gap > 5%)
5. Summarizes MPNN training quality and deploy accuracy

Usage:
    python -m project_health.analysis.noiseless_pipeline_analyzer_means
    python -m project_health.analysis.noiseless_pipeline_analyzer_means --verbose
    python -m project_health.analysis.noiseless_pipeline_analyzer_means --json report.json
    python -m project_health.analysis.noiseless_pipeline_analyzer_means --results-dir results/experiments/exp_noiseless
    python -m project_health.analysis.noiseless_pipeline_analyzer_means --topology chain_1d
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS_DIR = ROOT / "results" / "experiments"


# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class NoiselessRunSummary:
    """Summary of one noiseless pipeline run."""

    source_file: str
    n_qubits: int
    p_layers: int
    topology: str
    model: str
    h_min: float
    h_max: float
    h_points: int
    n_restarts: int
    maxiter: int
    elapsed_s: float
    # Section verdicts
    s1_pass: bool
    s2_pass: bool
    s3_pass: bool | None  # None if not run
    s4_pass: bool | None
    # Section 1 metrics
    gap_min: float | None = None
    gap_max: float | None = None
    mag_x_range: list[float] | None = None
    corr_zz_range: list[float] | None = None
    # Section 2 metrics
    vqe_n_pass: int = 0
    vqe_n_total: int = 0
    mean_fidelity: float | None = None
    min_fidelity: float | None = None
    mean_de_gap: float | None = None
    max_de_gap: float | None = None
    theta_smoothness_max: float | None = None
    n_converged: int | None = None
    vqe_time_s: float = 0.0
    # Section 3 metrics
    mpnn_final_mse: float | None = None
    mpnn_stopped_early: bool | None = None
    # Section 4 metrics
    deploy_mean_de_gap: float | None = None
    deploy_max_de_gap: float | None = None
    deploy_mean_fidelity: float | None = None
    deploy_n_pass: int | None = None
    deploy_n_total: int | None = None
    deploy_n_correct_label: int | None = None
    deploy_mag_x_error: float | None = None
    deploy_corr_zz_error: float | None = None


@dataclass
class TopologyComparison:
    """Cross-topology comparison for a given (N, p) pair."""

    n_qubits: int
    p_layers: int
    topologies: dict[str, NoiselessRunSummary] = field(default_factory=dict)


@dataclass
class NoiselessReport:
    """Complete noiseless pipeline analysis report."""

    runs: list[NoiselessRunSummary] = field(default_factory=list)
    comparisons: list[TopologyComparison] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    n_files_scanned: int = 0
    overall_verdict: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "overall_verdict": self.overall_verdict,
            "n_files_scanned": self.n_files_scanned,
            "n_runs": len(self.runs),
            "anomalies": self.anomalies,
            "runs": [asdict(r) for r in self.runs],
            "comparisons": [
                {
                    "n_qubits": c.n_qubits,
                    "p_layers": c.p_layers,
                    "topologies": {k: asdict(v) for k, v in c.topologies.items()},
                }
                for c in self.comparisons
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Scanning & Parsing
# ═══════════════════════════════════════════════════════════════════════════════


def scan_noiseless_results(results_dir: Path) -> list[dict[str, Any]]:
    """Discover and load all noiseless pipeline result JSON files.

    If results_dir points to a specific exp_noiseless_* folder, scans only that.
    If it points to a parent (e.g. results/experiments/), scans all exp_noiseless* subfolders.

    Supports both flat (exp_noiseless_tfim_4/run_*.json) and nested
    (exp_noiseless/tfim/heavy_hex/run_*.json) directory structures.
    """
    dirs_to_scan: list[Path] = []

    if results_dir.name.startswith("exp_noiseless"):
        # Specific folder
        dirs_to_scan = [results_dir]
    elif results_dir.exists():
        # Parent directory — find all exp_noiseless* subfolders
        dirs_to_scan = sorted(results_dir.glob("exp_noiseless*"))
        if not dirs_to_scan:
            # Maybe the dir itself contains run_*.json
            dirs_to_scan = [results_dir]
    else:
        logger.warning(f"Noiseless results directory not found: {results_dir}")
        return []

    results = []
    for d in dirs_to_scan:
        if not d.is_dir():
            continue
        # Use rglob to find run_*.json in nested subdirectories
        for f in sorted(d.rglob("run_*.json")):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                data["_source_file"] = str(f.name)
                data["_source_folder"] = str(f.parent.relative_to(d.parent))
                results.append(data)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load {f}: {e}")

    logger.info(f"Scanned {len(results)} noiseless result files from {len(dirs_to_scan)} folder(s)")
    return results


def parse_run(data: dict[str, Any]) -> NoiselessRunSummary | None:
    """Parse a single noiseless pipeline JSON into NoiselessRunSummary."""
    cfg = data.get("config", {})
    sys_cfg = cfg.get("system", {})
    h_grid = cfg.get("h_grid", {})
    vqe_cfg = cfg.get("vqe", {})
    results = data.get("results", {})

    # Must have at least section_1
    if "section_1" not in results:
        return None

    topos = sys_cfg.get("topologies", [])
    topo = topos[0] if topos else "unknown"
    n = sys_cfg.get("n_qubits", 0)
    p = sys_cfg.get("p_layers", 0)

    # Section verdicts
    s1_data = results.get("section_1", {}).get("data", {})
    s2_data = results.get("section_2", {}).get("data", {})
    s3_data = results.get("section_3", {}).get("data", {})
    s4_data = results.get("section_4", {}).get("data", {})

    s1_pass = s1_data.get("pass", False)
    s2_pass = s2_data.get("pass", False) if s2_data else False
    s3_pass = s3_data.get("pass") if s3_data else None
    s4_pass = s4_data.get("pass") if s4_data else None

    # Section 1 metrics
    topo_s1 = s1_data.get("topologies", {}).get(topo, {})
    gap_min = topo_s1.get("gap_min")
    gap_max = topo_s1.get("gap_max")
    mag_x_range = topo_s1.get("mag_x_range")
    corr_zz_range = topo_s1.get("corr_zz_range")

    # Section 2 metrics
    topo_s2 = s2_data.get("topologies", {}).get(topo, {})

    # Section 4 metrics
    deploy_mag_x_error = s4_data.get("mean_mag_x_error")
    deploy_corr_zz_error = s4_data.get("mean_corr_zz_error")

    return NoiselessRunSummary(
        source_file=data.get("_source_file", ""),
        n_qubits=n,
        p_layers=p,
        topology=topo,
        model=sys_cfg.get("model", "tfim"),
        h_min=h_grid.get("h_min", 0),
        h_max=h_grid.get("h_max", 0),
        h_points=h_grid.get("h_points", 0),
        n_restarts=vqe_cfg.get("n_restarts", 0),
        maxiter=vqe_cfg.get("maxiter", 0),
        elapsed_s=data.get("elapsed_s", 0),
        s1_pass=s1_pass,
        s2_pass=s2_pass,
        s3_pass=s3_pass,
        s4_pass=s4_pass,
        gap_min=gap_min,
        gap_max=gap_max,
        mag_x_range=mag_x_range,
        corr_zz_range=corr_zz_range,
        vqe_n_pass=topo_s2.get("n_pass_5pct", 0),
        vqe_n_total=topo_s2.get("n_points", 0),
        mean_fidelity=topo_s2.get("mean_fidelity"),
        min_fidelity=topo_s2.get("min_fidelity"),
        mean_de_gap=topo_s2.get("mean_de_gap"),
        max_de_gap=topo_s2.get("max_de_gap"),
        theta_smoothness_max=topo_s2.get("theta_smoothness_max"),
        n_converged=topo_s2.get("n_converged"),
        vqe_time_s=topo_s2.get("total_time_s", 0),
        mpnn_final_mse=s3_data.get("final_mse"),
        mpnn_stopped_early=s3_data.get("stopped_early"),
        deploy_mean_de_gap=s4_data.get("mean_de_gap"),
        deploy_max_de_gap=s4_data.get("max_de_gap"),
        deploy_mean_fidelity=s4_data.get("mean_fidelity"),
        deploy_n_pass=s4_data.get("n_pass_energy"),
        deploy_n_total=s4_data.get("n_test_points"),
        deploy_n_correct_label=s4_data.get("n_correct_label"),
        deploy_mag_x_error=deploy_mag_x_error,
        deploy_corr_zz_error=deploy_corr_zz_error,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════════════


def detect_anomalies(runs: list[NoiselessRunSummary]) -> list[str]:
    """Detect anomalies in noiseless results."""
    anomalies = []
    for r in runs:
        if r.max_de_gap is not None and r.max_de_gap > 10.0:
            anomalies.append(
                f"{r.source_file}: max ΔE/gap={r.max_de_gap:.1f} (VQE catastrophic failure "
                f"at {r.topology} N={r.n_qubits} p={r.p_layers})"
            )
        if r.mean_fidelity is not None and r.mean_fidelity < 0.8:
            anomalies.append(
                f"{r.source_file}: mean_F={r.mean_fidelity:.3f} < 0.8 "
                f"(ansatz cannot express ground state)"
            )
        if r.mpnn_final_mse is not None and r.mpnn_final_mse > 0.1:
            anomalies.append(
                f"{r.source_file}: MPNN MSE={r.mpnn_final_mse:.2e} (training diverged)"
            )
    return anomalies


def build_comparisons(runs: list[NoiselessRunSummary]) -> list[TopologyComparison]:
    """Build cross-topology comparisons grouped by (N, p)."""
    groups: dict[tuple[int, int], dict[str, NoiselessRunSummary]] = {}
    for r in runs:
        key = (r.n_qubits, r.p_layers)
        if key not in groups:
            groups[key] = {}
        # Keep the best (most sections passing) run per topology
        existing = groups[key].get(r.topology)
        if existing is None or _run_quality(r) > _run_quality(existing):
            groups[key][r.topology] = r

    comparisons = []
    for (n, p), topo_map in sorted(groups.items()):
        if len(topo_map) > 1:  # Only if multiple topologies
            comparisons.append(TopologyComparison(n_qubits=n, p_layers=p, topologies=topo_map))
    return comparisons


def _run_quality(r: NoiselessRunSummary) -> float:
    """Score a run for quality comparison (higher = better)."""
    score = 0.0
    if r.s1_pass:
        score += 1
    if r.s2_pass:
        score += 1
    if r.s3_pass:
        score += 1
    if r.s4_pass:
        score += 1
    if r.mean_fidelity:
        score += r.mean_fidelity
    return score


def generate_report(
    results_dir: Path,
    verbose: bool = False,
    topology_filter: str | None = None,
    n_qubits_filter: int | None = None,
    p_layers_filter: int | None = None,
    model_filter: str | None = None,
) -> NoiselessReport:
    """Generate a complete noiseless pipeline analysis report."""
    raw_data = scan_noiseless_results(results_dir)
    report = NoiselessReport(n_files_scanned=len(raw_data))

    for data in raw_data:
        run = parse_run(data)
        if run:
            if topology_filter and run.topology != topology_filter:
                continue
            if n_qubits_filter and run.n_qubits != n_qubits_filter:
                continue
            if p_layers_filter and run.p_layers != p_layers_filter:
                continue
            if model_filter and run.model != model_filter:
                continue
            report.runs.append(run)

    report.anomalies = detect_anomalies(report.runs)
    report.comparisons = build_comparisons(report.runs)

    # Overall verdict
    if not report.runs:
        report.overall_verdict = "NO_DATA"
    elif all(r.s2_pass for r in report.runs):
        report.overall_verdict = "PASS"
    elif any(r.s2_pass for r in report.runs):
        report.overall_verdict = "PARTIAL"
    else:
        report.overall_verdict = "FAIL"

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Text Formatting
# ═══════════════════════════════════════════════════════════════════════════════


def format_report_text(report: NoiselessReport) -> str:
    """Format the report as human-readable text."""
    lines = []
    lines.append("=" * 78)
    lines.append("  NOISELESS PIPELINE ANALYSIS")
    lines.append("=" * 78)
    lines.append(f"  Files scanned: {report.n_files_scanned}")
    lines.append(f"  Runs parsed:   {len(report.runs)}")
    lines.append(f"  Verdict:       {report.overall_verdict}")
    lines.append("")

    if not report.runs:
        lines.append("  No noiseless pipeline results found.")
        lines.append(f"  Expected location: {DEFAULT_RESULTS_DIR}")
        return "\n".join(lines)

    # ── Per-run table ────────────────────────────────────────────────────
    lines.append(
        f"  {'File':<28} {'N':>3} {'p':>2} {'Topology':<12} "
        f"{'h-range':<10} {'S1':>2} {'S2':>2} {'S3':>2} {'S4':>2} "
        f"{'VQE%':>5} {'mean_F':>7} {'max_ΔE':>8}"
    )
    lines.append("  " + "-" * 100)

    for r in report.runs:
        s1 = "Y" if r.s1_pass else "N"
        s2 = "Y" if r.s2_pass else "N"
        s3 = "Y" if r.s3_pass else ("N" if r.s3_pass is not None else "-")
        s4 = "Y" if r.s4_pass else ("N" if r.s4_pass is not None else "-")
        vqe_pct = f"{r.vqe_n_pass}/{r.vqe_n_total}" if r.vqe_n_total else "---"
        mean_f = f"{r.mean_fidelity:.4f}" if r.mean_fidelity else "---"
        max_de = f"{r.max_de_gap:.1e}" if r.max_de_gap else "---"
        h_range = f"{r.h_min}-{r.h_max}"

        lines.append(
            f"  {r.source_file:<28} {r.n_qubits:>3} {r.p_layers:>2} {r.topology:<12} "
            f"{h_range:<10} {s1:>2} {s2:>2} {s3:>2} {s4:>2} "
            f"{vqe_pct:>5} {mean_f:>7} {max_de:>8}"
        )

    # ── Cross-topology comparison ────────────────────────────────────────
    if report.comparisons:
        lines.append("")
        lines.append("  CROSS-TOPOLOGY COMPARISONS")
        lines.append("  " + "-" * 70)
        for comp in report.comparisons:
            lines.append(f"  N={comp.n_qubits}, p={comp.p_layers}:")
            for topo, r in sorted(comp.topologies.items()):
                vqe_str = f"{r.vqe_n_pass}/{r.vqe_n_total}"
                f_str = f"F={r.mean_fidelity:.4f}" if r.mean_fidelity else "F=---"
                de_str = f"max_ΔE={r.max_de_gap:.1e}" if r.max_de_gap else "max_ΔE=---"
                lines.append(f"    {topo:<14} VQE:{vqe_str:<6} {f_str}  {de_str}")

    # ── Best complete runs ───────────────────────────────────────────────
    complete = [r for r in report.runs if r.s4_pass]
    if complete:
        lines.append("")
        lines.append("  BEST COMPLETE RUNS (all 4 sections pass)")
        lines.append("  " + "-" * 70)
        for r in complete:
            lines.append(f"    {r.source_file}: N={r.n_qubits} p={r.p_layers} {r.topology}")
            lines.append(
                f"      VQE: mean_F={r.mean_fidelity:.5f}, θ_smooth={r.theta_smoothness_max or '?'}"
            )
            if r.mpnn_final_mse is not None:
                lines.append(f"      MPNN: MSE={r.mpnn_final_mse:.2e}")
            if r.deploy_mean_de_gap is not None:
                mag_err = f"{r.deploy_mag_x_error:.4f}" if r.deploy_mag_x_error else "N/A"
                zz_err = f"{r.deploy_corr_zz_error:.4f}" if r.deploy_corr_zz_error else "N/A"
                lines.append(
                    f"      Deploy: ΔE/gap={r.deploy_mean_de_gap:.2e}, "
                    f"F={r.deploy_mean_fidelity:.5f}, "
                    f"⟨X⟩_err={mag_err}, "
                    f"⟨ZZ⟩_err={zz_err}"
                )
                lines.append(f"      Labels: {r.deploy_n_correct_label}/{r.deploy_n_total}")

    # ── Anomalies ────────────────────────────────────────────────────────
    if report.anomalies:
        lines.append("")
        lines.append("  ANOMALIES")
        lines.append("  " + "-" * 70)
        for a in report.anomalies:
            lines.append(f"    ⚠ {a}")

    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)


def format_report_markdown(report: NoiselessReport) -> str:
    """Format as markdown table (thesis/figure integration compatible)."""
    lines = []
    lines.append("# Noiseless Pipeline Analysis Report\n")
    lines.append(f"- **Files scanned**: {report.n_files_scanned}")
    lines.append(f"- **Runs parsed**: {len(report.runs)}")
    lines.append(f"- **Verdict**: {report.overall_verdict}\n")

    if not report.runs:
        lines.append("No results found.\n")
        return "\n".join(lines)

    # Main results table
    lines.append("## Per-Run Results\n")
    lines.append(
        "| File | N | p | Topology | h-range | S1 | S2 | S3 | S4 | VQE pass | mean_F | max_ΔE/gap |"
    )
    lines.append(
        "|------|---|---|----------|---------|----|----|----|----|----------|--------|------------|"
    )

    for r in report.runs:
        s1 = "✓" if r.s1_pass else "✗"
        s2 = "✓" if r.s2_pass else "✗"
        s3 = "✓" if r.s3_pass else ("✗" if r.s3_pass is not None else "—")
        s4 = "✓" if r.s4_pass else ("✗" if r.s4_pass is not None else "—")
        vqe = f"{r.vqe_n_pass}/{r.vqe_n_total}" if r.vqe_n_total else "—"
        f_str = f"{r.mean_fidelity:.4f}" if r.mean_fidelity else "—"
        de_str = f"{r.max_de_gap:.2e}" if r.max_de_gap else "—"
        lines.append(
            f"| {r.source_file} | {r.n_qubits} | {r.p_layers} | {r.topology} | "
            f"{r.h_min}–{r.h_max} | {s1} | {s2} | {s3} | {s4} | "
            f"{vqe} | {f_str} | {de_str} |"
        )

    # Cross-topology comparison
    if report.comparisons:
        lines.append("\n## Cross-Topology Comparison\n")
        for comp in report.comparisons:
            lines.append(f"### N={comp.n_qubits}, p={comp.p_layers}\n")
            lines.append("| Topology | VQE pass | mean_F | max_ΔE/gap | θ_smooth |")
            lines.append("|----------|----------|--------|------------|----------|")
            for topo, r in sorted(comp.topologies.items()):
                vqe = f"{r.vqe_n_pass}/{r.vqe_n_total}" if r.vqe_n_total else "—"
                f_str = f"{r.mean_fidelity:.4f}" if r.mean_fidelity else "—"
                de_str = f"{r.max_de_gap:.2e}" if r.max_de_gap else "—"
                sm_str = f"{r.theta_smoothness_max:.2f}" if r.theta_smoothness_max else "—"
                lines.append(f"| {topo} | {vqe} | {f_str} | {de_str} | {sm_str} |")

    # Anomalies
    if report.anomalies:
        lines.append("\n## Anomalies\n")
        for a in report.anomalies:
            lines.append(f"- ⚠️ {a}")

    lines.append("")
    return "\n".join(lines)


def format_per_h_tables(
    results_dir: Path,
    h_targets: list[float],
    n_qubits_filter: int | None = None,
    model_filter: str | None = None,
    h_tolerance: float = 0.05,
) -> str:
    """Extract exact per-h-point data and format as tables per h-value.

    For each target h, finds the closest h-point in each run's VQE results
    and displays the exact ΔE/gap, fidelity, entanglement entropy, and
    convergence info — grouped by topology and p.

    Parameters
    ----------
    results_dir : Path
        Directory to scan for results.
    h_targets : list[float]
        Specific h values to extract (e.g., [1.0, 2.0, 3.0]).
    n_qubits_filter : int | None
        Filter by system size.
    model_filter : str | None
        Filter by model name.
    h_tolerance : float
        Maximum |h - h_target| to consider a match (default 0.05).
    """
    raw_data = scan_noiseless_results(results_dir)
    lines = []

    # Parse raw data and extract per-h VQE results
    # Structure: {h_target: {(p, topology): {metrics}}}
    for h_target in sorted(h_targets):
        lines.append(f"\n{'=' * 78}")
        lines.append(f"  PER-H TABLE: h = {h_target:.1f}")
        lines.append(f"{'=' * 78}")
        lines.append(
            f"  {'Topology':<14} {'p':>2} {'ΔE/gap':>10} {'Fidelity':>9} "
            f"{'Entropy':>8} {'E_vqe':>12} {'E_exact':>12} {'Gap':>8} {'Conv':>5} {'Iters':>6}"
        )
        lines.append("  " + "-" * 95)

        found_any = False
        for data in raw_data:
            cfg = data.get("config", {})
            sys_cfg = cfg.get("system", {})
            n = sys_cfg.get("n_qubits", 0)
            p = sys_cfg.get("p_layers", 0)
            topos = sys_cfg.get("topologies", [])
            topo = topos[0] if topos else ""
            model = sys_cfg.get("model", "tfim")

            if n_qubits_filter and n != n_qubits_filter:
                continue
            if model_filter and model != model_filter:
                continue

            # Get per-h VQE data from section_2 raw results
            results = data.get("results", {})
            s2 = results.get("section_2", {}).get("data", {})
            # The per-h data is not in the summary — need to look for it
            # in the internal _vqe_results which is NOT serialized to section_2 summary
            # BUT the per-point data IS available in the section envelope if we
            # check the full section data structure

            # Actually, the runner stores per-h data in self._vqe_results internally
            # but the JSON only has the topology-level summary.
            # We need to re-scan using the raw file and look for per-point arrays.
            # Let's check if there's a "per_h_results" key or similar...

            # The section_2 data only has the summary. But some runs may have
            # section_4.per_point data. Let's try section 4 first for deploy h-points,
            # and for VQE h-points we need a different approach.

            # WORKAROUND: Use section_1 ground truth + section_4 per_point for deploy,
            # but for VQE we don't have per-h in the JSON.
            # The actual per-h VQE data was added to topo_results[] in the runner
            # but NOT serialized to the JSON (only the summary is saved).

            # For now, use section_1 ground truth data which HAS per-h info
            s1 = results.get("section_1", {}).get("data", {})
            gt_data = s1.get("topologies", {}).get(topo, {})
            # gt_data is summary only (e_min, e_max, etc.), not per-h

            # BEST APPROACH: look at the full raw data — some recent runs
            # might have the analysis field with per-point data
            analysis = data.get("analysis", {})

            # Skip if no section_2 topology data
            if not s2:
                continue

            # Check if h_target is within the run's h-range
            h_grid = cfg.get("h_grid", {})
            h_min = h_grid.get("h_min", 999)
            h_max = h_grid.get("h_max", 0)
            if h_target < h_min - h_tolerance or h_target > h_max + h_tolerance:
                continue

            # We have the run but can't extract per-h VQE data from the JSON
            # Report the topology-level metrics as the best available
            topo_s2 = s2.get("topologies", {}).get(topo, {})
            if not topo_s2:
                continue

            # For per-h data, we need to read from section_4 per_point
            s4 = results.get("section_4", {}).get("data", {})
            per_point = s4.get("per_point", [])

            # Find closest h in deploy per_point
            best_match = None
            for pt in per_point:
                h_pt = pt.get("h_test", 999)
                if abs(h_pt - h_target) < h_tolerance:
                    if best_match is None or abs(h_pt - h_target) < abs(
                        best_match["h_test"] - h_target
                    ):
                        best_match = pt

            if best_match:
                de_gap = best_match.get("de_gap", None)
                fid = best_match.get("fidelity", None)
                ent = best_match.get("entanglement_entropy", None)
                e_pred = best_match.get("e_pred", None)
                e_exact = best_match.get("e_exact", None)

                de_str = f"{de_gap:.4e}" if de_gap is not None else "---"
                fid_str = f"{fid:.6f}" if fid is not None else "---"
                ent_str = f"{ent:.4f}" if ent is not None else "---"
                e_p_str = f"{e_pred:.6f}" if e_pred is not None else "---"
                e_e_str = f"{e_exact:.6f}" if e_exact is not None else "---"

                lines.append(
                    f"  {topo:<14} {p:>2} {de_str:>10} {fid_str:>9} "
                    f"{ent_str:>8} {e_p_str:>12} {e_e_str:>12} {'---':>8} {'---':>5} {'---':>6}"
                )
                found_any = True
            else:
                # Fallback: show summary-level data
                mean_de = topo_s2.get("mean_de_gap")
                mean_f = topo_s2.get("mean_fidelity")
                if mean_de is not None:
                    lines.append(
                        f"  {topo:<14} {p:>2} {'~' + f'{mean_de:.2e}':>10} "
                        f"{'~' + f'{mean_f:.4f}' if mean_f else '---':>9} "
                        f"{'---':>8} {'---':>12} {'---':>12} {'---':>8} {'---':>5} {'---':>6}"
                        f"  (avg, no per-h data)"
                    )
                    found_any = True

        if not found_any:
            lines.append("  No data found for this h-value.")

    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze noiseless pipeline (exact statevector) experiment results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing noiseless run_*.json files",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default=None,
        help="Filter by topology (chain_1d, ladder, triangular, heavy_hex, square)",
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=None,
        help="Filter by system size",
    )
    parser.add_argument(
        "--p-layers",
        type=int,
        default=None,
        help="Filter by HVA depth (1, 2, or 3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter by model (tfim, tfim_longitudinal, tfim_frustrated, etc.)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Export report as JSON to this file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Save text report to this file (auto-creates parent dirs)",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output in markdown format (for thesis integration)",
    )
    parser.add_argument(
        "--per-h",
        type=float,
        nargs="+",
        default=None,
        help="Show exact per-h-point table for specific h values "
        "(e.g. --per-h 1.0 2.0 3.0). Displays ΔE/gap, fidelity, entropy "
        "for each topology×p at those h-values.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    results_dir = Path(args.results_dir)
    report = generate_report(
        results_dir,
        verbose=args.verbose,
        topology_filter=args.topology,
        n_qubits_filter=args.n_qubits,
        p_layers_filter=args.p_layers,
        model_filter=args.model,
    )

    # Print text report
    if args.markdown:
        text = format_report_markdown(report)
    else:
        text = format_report_text(report)

    print(text)

    # Save text to file if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        print(f"Report saved: {out_path}")

    # Optionally export JSON
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        print(f"JSON report saved: {json_path}")

    # Per-h detailed tables
    if args.per_h:
        per_h_text = format_per_h_tables(
            results_dir, args.per_h, n_qubits_filter=args.n_qubits, model_filter=args.model
        )
        print(per_h_text)
        if args.output:
            # Append per-h tables to the output file
            out_path = Path(args.output)
            with open(out_path, "a") as f:
                f.write("\n" + per_h_text)

    return 0 if report.overall_verdict in ("PASS", "NO_DATA") else 1


if __name__ == "__main__":
    sys.exit(main())
