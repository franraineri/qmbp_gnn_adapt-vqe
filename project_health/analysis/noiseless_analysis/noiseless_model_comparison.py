#!/usr/bin/env python3
"""Noiseless Model Comparison — extracts structured metrics from per-h analysis reports.

Parses the markdown analysis files (*_noiseless_v2_per_h.md) that contain
per-run, per-h-point results for different Hamiltonians and topologies.

Produces structured data suitable for:
  1. Comparing models (TFIM transverse vs TFIM longitudinal vs Heisenberg)
  2. Comparing topologies within each model
  3. Identifying which (model, topology, p) combinations are viable

Usage:
    python -m project_health.analysis.noiseless_model_comparison
    python -m project_health.analysis.noiseless_model_comparison --output report.json
    python -m project_health.analysis.noiseless_model_comparison --summary
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent


# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PerHPoint:
    """Metrics for a single h-value in a deploy run."""

    h: float
    de_gap: float
    fidelity: float
    entropy: float
    e_pred: float
    e_exact: float
    x_err: float
    zz_err: float
    label_correct: bool


@dataclass
class SectionMetrics:
    """Metrics for a pipeline section (VQE, MPNN, Deploy)."""

    passed: bool
    # VQE-specific
    pass_rate: str | None = None  # e.g. "36/40"
    mean_fidelity: float | None = None
    min_fidelity: float | None = None
    mean_de_gap: float | None = None
    max_de_gap: float | None = None
    theta_smooth_max: float | None = None
    theta_smooth_mean: float | None = None
    mean_entropy: float | None = None
    # MPNN-specific
    final_mse: float | None = None
    final_de_gap: float | None = None
    n_params: int | None = None
    n_training: int | None = None
    per_h_mse_mean: float | None = None
    per_h_mse_max: float | None = None
    per_h_mse_min: float | None = None
    # Deploy-specific
    n_test: int | None = None
    pass_energy: int | None = None
    correct_labels: int | None = None
    deploy_mean_de_gap: float | None = None
    deploy_max_de_gap: float | None = None
    deploy_mean_fidelity: float | None = None
    speedup_factor: float | None = None
    mpnn_wins_vs_random: int | None = None


@dataclass
class RunResult:
    """Complete result of one pipeline run (one file, one topology)."""

    source_file: str
    n_qubits: int
    p_layers: int
    topology: str
    model: str
    h_min: float
    h_max: float
    h_points: int
    elapsed_s: float
    vqe: SectionMetrics | None = None
    mpnn: SectionMetrics | None = None
    deploy: SectionMetrics | None = None
    per_h_data: list[PerHPoint] = field(default_factory=list)
    deploy_pass_rate_pct: float | None = None  # convenience: pass_energy / n_test * 100


@dataclass
class ModelSummary:
    """Aggregated summary for one Hamiltonian model across all runs."""

    model: str
    n_runs: int
    topologies: list[str]
    p_layers_tested: list[int]
    best_deploy_pass_rate: float  # % pass across all runs
    worst_deploy_pass_rate: float
    mean_deploy_pass_rate: float
    best_topology: str  # topology with highest mean deploy pass rate
    best_p: int  # p_layers with highest mean deploy pass rate
    runs: list[RunResult] = field(default_factory=list)


@dataclass
class TopologySummary:
    """Aggregated summary for one topology across all models."""

    topology: str
    n_runs: int
    models: list[str]
    mean_deploy_pass_rate: float
    per_model: dict[str, float] = field(default_factory=dict)  # model -> mean pass rate


@dataclass
class ComparisonReport:
    """Full comparison report across models and topologies."""

    models: list[ModelSummary] = field(default_factory=list)
    topologies: list[TopologySummary] = field(default_factory=list)
    all_runs: list[RunResult] = field(default_factory=list)
    ranking_by_model: list[tuple[str, float]] = field(default_factory=list)
    ranking_by_topology: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON."""
        return {
            "ranking_by_model": self.ranking_by_model,
            "ranking_by_topology": self.ranking_by_topology,
            "models": [asdict(m) for m in self.models],
            "topologies": [asdict(t) for t in self.topologies],
            "n_total_runs": len(self.all_runs),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Parsing
# ═══════════════════════════════════════════════════════════════════════════════

# Regex patterns for the markdown analysis format
RE_HEADER = re.compile(
    r"N=(\d+)\s+p=(\d+)\s+topo=(\w+)\s+model=(\w+)\s+"
    r"h=\[([0-9.]+),\s*([0-9.]+)\]\s+pts=(\d+)"
)
RE_FILE = re.compile(r"File:\s+(run_\S+\.json)")
RE_ELAPSED = re.compile(r"Elapsed:\s+([0-9.]+)s")
RE_SECTION = re.compile(r"Section\s+(\d)\s+\((\w+)\):\s+(✅ PASS|❌ FAIL)")
RE_PASS_RATE = re.compile(r"pass_rate:\s+(\d+)/(\d+)")
RE_MEAN_F = re.compile(r"mean_F=([0-9.]+)")
RE_MIN_F = re.compile(r"min_F=([0-9.]+)")
RE_MEAN_DE = re.compile(r"mean_ΔE/gap=([0-9.e+\-]+)")
RE_MAX_DE = re.compile(r"max_ΔE/gap=([0-9.e+\-]+)")
RE_THETA_MAX = re.compile(r"θ_smooth_max=([0-9.]+)")
RE_THETA_MEAN = re.compile(r"θ_smooth_mean=([0-9.]+)")
RE_ENTROPY = re.compile(r"mean_entropy=([0-9.]+)")
RE_MSE = re.compile(r"final_mse=([0-9.e+\-]+)")
RE_MSE_DE = re.compile(r"final_de_gap=([0-9.e+\-]+)")
RE_NPARAMS = re.compile(r"n_params=(\d+)")
RE_NTRAINING = re.compile(r"n_training=(\d+)")
RE_PER_H_MSE = re.compile(
    r"per_h_mse:\s+mean=([0-9.e+\-]+)\s+max=([0-9.e+\-]+)\s+min=([0-9.e+\-]+)"
)
RE_NTEST = re.compile(r"n_test=(\d+)")
RE_PASS_ENERGY = re.compile(r"pass_energy=(\d+)")
RE_CORRECT_LABELS = re.compile(r"correct_labels=(\d+)")
RE_SPEEDUP = re.compile(r"speedup_factor=([0-9.]+)x")
RE_MPNN_WINS = re.compile(r"mpnn_wins_vs_random=(\d+)")
RE_DEPLOY_MEAN_DE = re.compile(r"mean_ΔE/gap=([0-9.e+\-]+)")
RE_DEPLOY_MAX_DE = re.compile(r"max_ΔE/gap=([0-9.e+\-]+)")
RE_DEPLOY_MEAN_F = re.compile(r"mean_F=([0-9.]+)")
RE_PER_H_ROW = re.compile(
    r"\s+([0-9.]+)\s+([0-9.e+\-]+)\s+([0-9.]+)\s+([0-9.]+)\s+"
    r"([0-9.\-]+)\s+([0-9.\-]+)\s+([0-9.e+\-]+)\s+([0-9.e+\-]+)\s+(✓|✗)"
)


def _safe_float(s: str) -> float:
    """Parse float from scientific/decimal notation."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def parse_markdown_report(filepath: Path) -> list[RunResult]:
    """Parse a *_noiseless_v2_per_h.md file into structured RunResult objects."""
    text = filepath.read_text(encoding="utf-8")
    runs: list[RunResult] = []

    # Split by file headers
    blocks = re.split(r"(?=\s*File:\s+run_)", text)

    for block in blocks:
        if not block.strip():
            continue

        file_m = RE_FILE.search(block)
        header_m = RE_HEADER.search(block)
        elapsed_m = RE_ELAPSED.search(block)

        if not header_m:
            continue

        source_file = file_m.group(1) if file_m else "unknown"
        n_qubits = int(header_m.group(1))
        p_layers = int(header_m.group(2))
        topology = header_m.group(3)
        model = header_m.group(4)
        h_min = float(header_m.group(5))
        h_max = float(header_m.group(6))
        h_points = int(header_m.group(7))
        elapsed_s = float(elapsed_m.group(1)) if elapsed_m else 0.0

        # Check for ERROR — skip these blocks
        if "ERROR processing" in block:
            continue

        run = RunResult(
            source_file=source_file,
            n_qubits=n_qubits,
            p_layers=p_layers,
            topology=topology,
            model=model,
            h_min=h_min,
            h_max=h_max,
            h_points=h_points,
            elapsed_s=elapsed_s,
        )

        # Parse sections
        run.vqe = _parse_vqe_section(block)
        run.mpnn = _parse_mpnn_section(block)
        run.deploy = _parse_deploy_section(block)

        # Parse per-h data table
        run.per_h_data = _parse_per_h_table(block)

        # Compute deploy pass rate convenience field
        if run.deploy and run.deploy.pass_energy is not None and run.deploy.n_test:
            run.deploy_pass_rate_pct = run.deploy.pass_energy / run.deploy.n_test * 100.0

        runs.append(run)

    return runs


def _parse_vqe_section(block: str) -> SectionMetrics | None:
    """Extract VQE (Section 2) metrics from a block."""
    m = re.search(r"Section 2 \(VQE\):\s+(✅ PASS|❌ FAIL)", block)
    if not m:
        return None

    passed = m.group(1) == "✅ PASS"
    # Extract the VQE-specific lines (between Section 2 and Section 3)
    section_text = _extract_between(block, "Section 2", "Section 3")

    metrics = SectionMetrics(passed=passed)
    pr = RE_PASS_RATE.search(section_text)
    if pr:
        metrics.pass_rate = f"{pr.group(1)}/{pr.group(2)}"

    mf = RE_MEAN_F.search(section_text)
    if mf:
        metrics.mean_fidelity = _safe_float(mf.group(1))

    minf = RE_MIN_F.search(section_text)
    if minf:
        metrics.min_fidelity = _safe_float(minf.group(1))

    mde = RE_MEAN_DE.search(section_text)
    if mde:
        metrics.mean_de_gap = _safe_float(mde.group(1))

    xde = RE_MAX_DE.search(section_text)
    if xde:
        metrics.max_de_gap = _safe_float(xde.group(1))

    tm = RE_THETA_MAX.search(section_text)
    if tm:
        metrics.theta_smooth_max = _safe_float(tm.group(1))

    tmn = RE_THETA_MEAN.search(section_text)
    if tmn:
        metrics.theta_smooth_mean = _safe_float(tmn.group(1))

    ent = RE_ENTROPY.search(section_text)
    if ent:
        metrics.mean_entropy = _safe_float(ent.group(1))

    return metrics


def _parse_mpnn_section(block: str) -> SectionMetrics | None:
    """Extract MPNN (Section 3) metrics from a block."""
    m = re.search(r"Section 3 \(MPNN\):\s+(✅ PASS|❌ FAIL)", block)
    if not m:
        return None

    passed = m.group(1) == "✅ PASS"
    section_text = _extract_between(block, "Section 3", "Section 4")

    metrics = SectionMetrics(passed=passed)
    mse = RE_MSE.search(section_text)
    if mse:
        metrics.final_mse = _safe_float(mse.group(1))

    mde = RE_MSE_DE.search(section_text)
    if mde:
        metrics.final_de_gap = _safe_float(mde.group(1))

    np_ = RE_NPARAMS.search(section_text)
    if np_:
        metrics.n_params = int(np_.group(1))

    nt = RE_NTRAINING.search(section_text)
    if nt:
        metrics.n_training = int(nt.group(1))

    phm = RE_PER_H_MSE.search(section_text)
    if phm:
        metrics.per_h_mse_mean = _safe_float(phm.group(1))
        metrics.per_h_mse_max = _safe_float(phm.group(2))
        metrics.per_h_mse_min = _safe_float(phm.group(3))

    return metrics


def _parse_deploy_section(block: str) -> SectionMetrics | None:
    """Extract Deploy (Section 4) metrics from a block."""
    m = re.search(r"Section 4 \(Deploy\):\s+(✅ PASS|❌ FAIL)", block)
    if not m:
        return None

    passed = m.group(1) == "✅ PASS"
    # Deploy section goes to end or next "File:" marker
    section_text = _extract_between(block, "Section 4", "Statistics")

    metrics = SectionMetrics(passed=passed)
    nt = RE_NTEST.search(section_text)
    if nt:
        metrics.n_test = int(nt.group(1))

    pe = RE_PASS_ENERGY.search(section_text)
    if pe:
        metrics.pass_energy = int(pe.group(1))

    cl = RE_CORRECT_LABELS.search(section_text)
    if cl:
        metrics.correct_labels = int(cl.group(1))

    mde = RE_DEPLOY_MEAN_DE.search(section_text)
    if mde:
        metrics.deploy_mean_de_gap = _safe_float(mde.group(1))

    xde = RE_DEPLOY_MAX_DE.search(section_text)
    if xde:
        metrics.deploy_max_de_gap = _safe_float(xde.group(1))

    mf = RE_DEPLOY_MEAN_F.search(section_text)
    if mf:
        metrics.deploy_mean_fidelity = _safe_float(mf.group(1))

    sp = RE_SPEEDUP.search(section_text)
    if sp:
        metrics.speedup_factor = _safe_float(sp.group(1))

    mw = RE_MPNN_WINS.search(section_text)
    if mw:
        metrics.mpnn_wins_vs_random = int(mw.group(1))

    return metrics


def _parse_per_h_table(block: str) -> list[PerHPoint]:
    """Extract per-h data rows from the table in a block."""
    points: list[PerHPoint] = []
    for m in RE_PER_H_ROW.finditer(block):
        # Only take first 6 rows (sample table in header area)
        # Actually, take all matches for the per-h table
        points.append(
            PerHPoint(
                h=_safe_float(m.group(1)),
                de_gap=_safe_float(m.group(2)),
                fidelity=_safe_float(m.group(3)),
                entropy=_safe_float(m.group(4)),
                e_pred=_safe_float(m.group(5)),
                e_exact=_safe_float(m.group(6)),
                x_err=_safe_float(m.group(7)),
                zz_err=_safe_float(m.group(8)),
                label_correct=(m.group(9) == "✓"),
            )
        )
    return points


def _extract_between(text: str, start_marker: str, end_marker: str) -> str:
    """Extract text between two markers."""
    start_idx = text.find(start_marker)
    if start_idx == -1:
        return ""
    end_idx = text.find(end_marker, start_idx + len(start_marker))
    if end_idx == -1:
        return text[start_idx:]
    return text[start_idx:end_idx]


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation & Comparison
# ═══════════════════════════════════════════════════════════════════════════════


def build_model_summaries(all_runs: list[RunResult]) -> list[ModelSummary]:
    """Group runs by model and compute aggregated metrics."""
    from collections import defaultdict

    by_model: dict[str, list[RunResult]] = defaultdict(list)
    for r in all_runs:
        by_model[r.model].append(r)

    summaries = []
    for model, runs in sorted(by_model.items()):
        pass_rates = [r.deploy_pass_rate_pct for r in runs if r.deploy_pass_rate_pct is not None]
        if not pass_rates:
            continue

        # Best topology: group by topology, compute mean pass rate per topology
        by_topo: dict[str, list[float]] = defaultdict(list)
        by_p: dict[int, list[float]] = defaultdict(list)
        for r in runs:
            if r.deploy_pass_rate_pct is not None:
                by_topo[r.topology].append(r.deploy_pass_rate_pct)
                by_p[r.p_layers].append(r.deploy_pass_rate_pct)

        best_topo = max(by_topo, key=lambda t: sum(by_topo[t]) / len(by_topo[t]))
        best_p = max(by_p, key=lambda p: sum(by_p[p]) / len(by_p[p]))

        summaries.append(
            ModelSummary(
                model=model,
                n_runs=len(runs),
                topologies=sorted(set(r.topology for r in runs)),
                p_layers_tested=sorted(set(r.p_layers for r in runs)),
                best_deploy_pass_rate=max(pass_rates),
                worst_deploy_pass_rate=min(pass_rates),
                mean_deploy_pass_rate=sum(pass_rates) / len(pass_rates),
                best_topology=best_topo,
                best_p=best_p,
                runs=runs,
            )
        )

    return summaries


def build_topology_summaries(all_runs: list[RunResult]) -> list[TopologySummary]:
    """Group runs by topology and compute aggregated metrics."""
    from collections import defaultdict

    by_topo: dict[str, list[RunResult]] = defaultdict(list)
    for r in all_runs:
        by_topo[r.topology].append(r)

    summaries = []
    for topo, runs in sorted(by_topo.items()):
        pass_rates = [r.deploy_pass_rate_pct for r in runs if r.deploy_pass_rate_pct is not None]
        if not pass_rates:
            continue

        per_model: dict[str, list[float]] = defaultdict(list)
        for r in runs:
            if r.deploy_pass_rate_pct is not None:
                per_model[r.model].append(r.deploy_pass_rate_pct)

        summaries.append(
            TopologySummary(
                topology=topo,
                n_runs=len(runs),
                models=sorted(set(r.model for r in runs)),
                mean_deploy_pass_rate=sum(pass_rates) / len(pass_rates),
                per_model={m: sum(v) / len(v) for m, v in per_model.items()},
            )
        )

    return summaries


def build_comparison_report(all_runs: list[RunResult]) -> ComparisonReport:
    """Build the full comparison report."""
    model_summaries = build_model_summaries(all_runs)
    topo_summaries = build_topology_summaries(all_runs)

    # Rankings
    ranking_model = sorted(
        [(m.model, m.mean_deploy_pass_rate) for m in model_summaries],
        key=lambda x: x[1],
        reverse=True,
    )
    ranking_topo = sorted(
        [(t.topology, t.mean_deploy_pass_rate) for t in topo_summaries],
        key=lambda x: x[1],
        reverse=True,
    )

    return ComparisonReport(
        models=model_summaries,
        topologies=topo_summaries,
        all_runs=all_runs,
        ranking_by_model=ranking_model,
        ranking_by_topology=ranking_topo,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Output Formatting
# ═══════════════════════════════════════════════════════════════════════════════


def format_summary(report: ComparisonReport) -> str:
    """Human-readable summary of the comparison."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  NOISELESS MODEL COMPARISON — SUMMARY")
    lines.append("=" * 80)
    lines.append(f"\n  Total runs analyzed: {len(report.all_runs)}")
    lines.append(f"  Models: {len(report.models)}")
    lines.append(f"  Topologies: {len(report.topologies)}")

    # Model ranking
    lines.append("\n" + "─" * 80)
    lines.append("  RANKING BY MODEL (mean deploy pass rate %)")
    lines.append("─" * 80)
    for i, (model, rate) in enumerate(report.ranking_by_model, 1):
        bar = "█" * int(rate / 2) + "░" * (50 - int(rate / 2))
        lines.append(f"  {i}. {model:<25} {rate:6.1f}%  {bar}")

    # Topology ranking
    lines.append("\n" + "─" * 80)
    lines.append("  RANKING BY TOPOLOGY (mean deploy pass rate %)")
    lines.append("─" * 80)
    for i, (topo, rate) in enumerate(report.ranking_by_topology, 1):
        bar = "█" * int(rate / 2) + "░" * (50 - int(rate / 2))
        lines.append(f"  {i}. {topo:<25} {rate:6.1f}%  {bar}")

    # Per-model detail
    lines.append("\n" + "─" * 80)
    lines.append("  DETAIL PER MODEL")
    lines.append("─" * 80)
    for ms in report.models:
        lines.append(f"\n  ▸ {ms.model}")
        lines.append(f"    Runs: {ms.n_runs}")
        lines.append(f"    Topologies tested: {', '.join(ms.topologies)}")
        lines.append(f"    p-layers tested: {ms.p_layers_tested}")
        lines.append(
            f"    Deploy pass rate: "
            f"best={ms.best_deploy_pass_rate:.1f}%, "
            f"mean={ms.mean_deploy_pass_rate:.1f}%, "
            f"worst={ms.worst_deploy_pass_rate:.1f}%"
        )
        lines.append(f"    Best topology: {ms.best_topology}")
        lines.append(f"    Best p: {ms.best_p}")

        # Table per (topology, p)
        lines.append(
            f"    {'Topology':<15} {'p':>3} {'Pass%':>7} "
            f"{'ΔE/gap':>12} {'F_mean':>8} {'Speedup':>8}"
        )
        lines.append(f"    {'─' * 15} {'─' * 3} {'─' * 7} {'─' * 12} {'─' * 8} {'─' * 8}")
        for r in sorted(ms.runs, key=lambda x: (-x.deploy_pass_rate_pct or 0)):
            pr = f"{r.deploy_pass_rate_pct:.0f}%" if r.deploy_pass_rate_pct is not None else "N/A"
            de = (
                f"{r.deploy.deploy_mean_de_gap:.2e}"
                if r.deploy and r.deploy.deploy_mean_de_gap
                else "N/A"
            )
            fm = (
                f"{r.deploy.deploy_mean_fidelity:.4f}"
                if r.deploy and r.deploy.deploy_mean_fidelity
                else "N/A"
            )
            sp = (
                f"{r.deploy.speedup_factor:.1f}x" if r.deploy and r.deploy.speedup_factor else "N/A"
            )
            lines.append(f"    {r.topology:<15} {r.p_layers:>3} {pr:>7} {de:>12} {fm:>8} {sp:>8}")

    # Cross-model × topology matrix
    lines.append("\n" + "─" * 80)
    lines.append("  CROSS-TABLE: MODEL × TOPOLOGY (mean deploy pass rate %)")
    lines.append("─" * 80)
    all_topos = sorted(set(r.topology for r in report.all_runs))
    all_models = sorted(set(r.model for r in report.all_runs))

    header = f"    {'Model':<25}" + "".join(f"{t:>12}" for t in all_topos)
    lines.append(header)
    lines.append("    " + "─" * (25 + 12 * len(all_topos)))

    from collections import defaultdict

    matrix: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in report.all_runs:
        if r.deploy_pass_rate_pct is not None:
            matrix[r.model][r.topology].append(r.deploy_pass_rate_pct)

    for model in all_models:
        row = f"    {model:<25}"
        for topo in all_topos:
            vals = matrix[model].get(topo, [])
            if vals:
                row += f"{sum(vals) / len(vals):>11.1f}%"
            else:
                row += f"{'—':>12}"
        lines.append(row)

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Discovery & Main
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_MD_FILES = [
    ROOT / "tfim_noiseless_v2_per_h.md",
    ROOT / "tfim_long_noiseless_v2_per_h.md",
    ROOT / "heisenberg_noiseless_v2_per_h.md",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Thesis Table Generation (TableSpec-compatible)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TableSpec:
    """Specification for one thesis table (matches thesis_tables_compiler format)."""

    table_id: str
    title: str
    caption: str
    columns: list[str]
    rows: list[list[str]] = field(default_factory=list)
    notes: str = ""


def format_table_markdown(table: TableSpec) -> str:
    """Format a TableSpec as Markdown."""
    lines = [
        f"## {table.table_id} — {table.title}",
        "",
        f"*{table.caption}*",
        "",
    ]
    lines.append("| " + " | ".join(table.columns) + " |")
    lines.append("|" + "|".join("---" for _ in table.columns) + "|")
    for row in table.rows:
        lines.append("| " + " | ".join(row) + " |")
    if table.notes:
        lines.extend(["", f"**Notes**: {table.notes}"])
    lines.append("")
    return "\n".join(lines)


def format_table_latex(table: TableSpec) -> str:
    """Format a TableSpec as LaTeX."""
    n_cols = len(table.columns)
    col_spec = "l" + "c" * (n_cols - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{table.caption}}}",
        f"\\label{{tab:{table.table_id.lower()}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(f"\\textbf{{{c}}}" for c in table.columns) + r" \\",
        r"\midrule",
    ]
    for row in table.rows:
        escaped = [c.replace("_", r"\_").replace("%", r"\%") for c in row]
        lines.append(" & ".join(escaped) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if table.notes:
        lines.append(f"\\tablefoot{{{table.notes}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_tables(all_runs: list[RunResult]) -> list[TableSpec]:
    """Generate all comparison tables from parsed runs."""
    tables: list[TableSpec] = []
    tables.append(_table_model_ranking(all_runs))
    tables.append(_table_topology_ranking(all_runs))
    tables.append(_table_cross_matrix(all_runs))
    tables.append(_table_best_configs(all_runs))
    tables.append(_table_vqe_quality(all_runs))
    tables.append(_table_mpnn_quality(all_runs))
    tables.append(_table_per_p_breakdown(all_runs))
    return tables


def _table_model_ranking(runs: list[RunResult]) -> TableSpec:
    """TN1: Model ranking by deploy pass rate."""
    from collections import defaultdict

    by_model: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        if r.deploy_pass_rate_pct is not None:
            by_model[r.model].append(r.deploy_pass_rate_pct)

    rows = []
    for model in sorted(by_model, key=lambda m: -sum(by_model[m]) / len(by_model[m])):
        vals = by_model[model]
        mean_pr = sum(vals) / len(vals)
        best_pr = max(vals)
        worst_pr = min(vals)
        n_runs = len(vals)
        rows.append(
            [
                model,
                str(n_runs),
                f"{mean_pr:.1f}%",
                f"{best_pr:.1f}%",
                f"{worst_pr:.1f}%",
                "✅" if mean_pr > 50 else "❌",
            ]
        )

    return TableSpec(
        table_id="TN1",
        title="Noiseless Model Ranking",
        caption="Deploy pass rate (ΔE/gap < 5%) by Hamiltonian model, aggregated across all topologies and p-layers.",
        columns=["Model", "N Runs", "Mean Pass%", "Best", "Worst", "Viable"],
        rows=rows,
        notes="Heisenberg p=4 uniformly fails — HVA expressibility insufficient for frustrated Heisenberg.",
    )


def _table_topology_ranking(runs: list[RunResult]) -> TableSpec:
    """TN2: Topology ranking by deploy pass rate."""
    from collections import defaultdict

    by_topo: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        if r.deploy_pass_rate_pct is not None:
            by_topo[r.topology].append(r.deploy_pass_rate_pct)

    rows = []
    for topo in sorted(by_topo, key=lambda t: -sum(by_topo[t]) / len(by_topo[t])):
        vals = by_topo[topo]
        mean_pr = sum(vals) / len(vals)
        best_pr = max(vals)
        n_runs = len(vals)
        rows.append(
            [
                topo,
                str(n_runs),
                f"{mean_pr:.1f}%",
                f"{best_pr:.1f}%",
                "✅" if mean_pr > 50 else "⚠️" if mean_pr > 30 else "❌",
            ]
        )

    return TableSpec(
        table_id="TN2",
        title="Noiseless Topology Ranking",
        caption="Deploy pass rate by topology, aggregated across all models and p-layers.",
        columns=["Topology", "N Runs", "Mean Pass%", "Best", "Status"],
        rows=rows,
        notes="heavy_hex leads due to low connectivity (9 edges for N=10) matching HVA expressibility.",
    )


def _table_cross_matrix(runs: list[RunResult]) -> TableSpec:
    """TN3: Model × Topology cross-table."""
    from collections import defaultdict

    all_topos = sorted(set(r.topology for r in runs))
    all_models = sorted(set(r.model for r in runs))

    matrix: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if r.deploy_pass_rate_pct is not None:
            matrix[r.model][r.topology].append(r.deploy_pass_rate_pct)

    columns = ["Model"] + all_topos
    rows = []
    for model in all_models:
        row = [model]
        for topo in all_topos:
            vals = matrix[model].get(topo, [])
            if vals:
                row.append(f"{sum(vals) / len(vals):.1f}%")
            else:
                row.append("—")
        rows.append(row)

    return TableSpec(
        table_id="TN3",
        title="Model × Topology Cross-Table",
        caption="Mean deploy pass rate for each (model, topology) combination.",
        columns=columns,
        rows=rows,
        notes="TFIM models perform best on heavy_hex; triangular topology penalizes all models.",
    )


def _table_best_configs(runs: list[RunResult]) -> TableSpec:
    """TN4: Best configuration per model (top-3 runs)."""
    from collections import defaultdict

    by_model: dict[str, list[RunResult]] = defaultdict(list)
    for r in runs:
        if r.deploy_pass_rate_pct is not None:
            by_model[r.model].append(r)

    rows = []
    for model in sorted(by_model):
        sorted_runs = sorted(by_model[model], key=lambda x: -(x.deploy_pass_rate_pct or 0))
        for r in sorted_runs[:3]:
            de = (
                f"{r.deploy.deploy_mean_de_gap:.2e}"
                if r.deploy and r.deploy.deploy_mean_de_gap
                else "—"
            )
            fm = (
                f"{r.deploy.deploy_mean_fidelity:.4f}"
                if r.deploy and r.deploy.deploy_mean_fidelity
                else "—"
            )
            sp = f"{r.deploy.speedup_factor:.1f}x" if r.deploy and r.deploy.speedup_factor else "—"
            rows.append(
                [
                    model,
                    r.topology,
                    str(r.p_layers),
                    f"{r.deploy_pass_rate_pct:.0f}%",
                    de,
                    fm,
                    sp,
                ]
            )

    return TableSpec(
        table_id="TN4",
        title="Best Configurations Per Model",
        caption="Top-3 performing (topology, p) configurations for each Hamiltonian.",
        columns=["Model", "Topology", "p", "Pass%", "Mean ΔE/gap", "Mean F", "Speedup"],
        rows=rows,
        notes="Speedup = MPNN inference time / VQE solve time (higher = more practical).",
    )


def _table_vqe_quality(runs: list[RunResult]) -> TableSpec:
    """TN5: VQE optimization quality per model."""
    from collections import defaultdict

    by_model: dict[str, list[RunResult]] = defaultdict(list)
    for r in runs:
        if r.vqe:
            by_model[r.model].append(r)

    rows = []
    for model in sorted(by_model):
        model_runs = by_model[model]
        n_pass = sum(1 for r in model_runs if r.vqe and r.vqe.passed)
        fids = [r.vqe.mean_fidelity for r in model_runs if r.vqe and r.vqe.mean_fidelity]
        des = [r.vqe.mean_de_gap for r in model_runs if r.vqe and r.vqe.mean_de_gap]
        ents = [r.vqe.mean_entropy for r in model_runs if r.vqe and r.vqe.mean_entropy]

        mean_f = sum(fids) / len(fids) if fids else 0
        mean_de = sum(des) / len(des) if des else 0
        mean_ent = sum(ents) / len(ents) if ents else 0

        rows.append(
            [
                model,
                str(len(model_runs)),
                f"{n_pass}/{len(model_runs)}",
                f"{mean_f:.4f}",
                f"{mean_de:.2e}",
                f"{mean_ent:.3f}",
                "✅" if mean_f > 0.95 else "⚠️" if mean_f > 0.80 else "❌",
            ]
        )

    return TableSpec(
        table_id="TN5",
        title="VQE Optimization Quality by Model",
        caption="VQE convergence quality — fidelity, energy gap error, and entanglement entropy.",
        columns=["Model", "Runs", "VQE Pass", "Mean F", "Mean ΔE/gap", "Mean S", "Quality"],
        rows=rows,
        notes="Heisenberg VQE fidelity ≈ 0 confirms HVA ansatz cannot express ground state.",
    )


def _table_mpnn_quality(runs: list[RunResult]) -> TableSpec:
    """TN6: MPNN training quality per model."""
    from collections import defaultdict

    by_model: dict[str, list[RunResult]] = defaultdict(list)
    for r in runs:
        if r.mpnn:
            by_model[r.model].append(r)

    rows = []
    for model in sorted(by_model):
        model_runs = by_model[model]
        n_pass = sum(1 for r in model_runs if r.mpnn and r.mpnn.passed)
        mses = [r.mpnn.final_mse for r in model_runs if r.mpnn and r.mpnn.final_mse]
        per_h_means = [
            r.mpnn.per_h_mse_mean for r in model_runs if r.mpnn and r.mpnn.per_h_mse_mean
        ]

        mean_mse = sum(mses) / len(mses) if mses else 0
        mean_ph = sum(per_h_means) / len(per_h_means) if per_h_means else 0

        rows.append(
            [
                model,
                str(len(model_runs)),
                f"{n_pass}/{len(model_runs)}",
                f"{mean_mse:.2e}",
                f"{mean_ph:.2e}",
                "✅" if mean_mse < 0.01 else "⚠️" if mean_mse < 0.02 else "❌",
            ]
        )

    return TableSpec(
        table_id="TN6",
        title="MPNN Training Quality by Model",
        caption="MPNN generalization quality — final MSE and per-h prediction error.",
        columns=["Model", "Runs", "MPNN Pass", "Mean MSE", "Mean per-h MSE", "Quality"],
        rows=rows,
        notes="MPNN can fit even bad VQE data (low MSE) — but garbage in → garbage out at deploy.",
    )


def _table_per_p_breakdown(runs: list[RunResult]) -> TableSpec:
    """TN7: Performance breakdown by p-layers (circuit depth)."""
    from collections import defaultdict

    by_mp: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in runs:
        if r.deploy_pass_rate_pct is not None:
            by_mp[(r.model, r.p_layers)].append(r.deploy_pass_rate_pct)

    rows = []
    for model, p in sorted(by_mp):
        vals = by_mp[(model, p)]
        mean_pr = sum(vals) / len(vals)
        rows.append(
            [
                model,
                str(p),
                str(len(vals)),
                f"{mean_pr:.1f}%",
                f"{max(vals):.1f}%",
                f"{min(vals):.1f}%",
            ]
        )

    return TableSpec(
        table_id="TN7",
        title="Performance by Circuit Depth (p-layers)",
        caption="Deploy pass rate grouped by model and HVA p-layers.",
        columns=["Model", "p", "N Runs", "Mean Pass%", "Best", "Worst"],
        rows=rows,
        notes="TFIM: p=2 optimal for chain_1d/heavy_hex. p=4 helps ladder/square. Heisenberg: p=4 insufficient.",
    )


def discover_analysis_files(search_dir: Path | None = None) -> list[Path]:
    """Find all *_noiseless_v2_per_h.md files."""
    if search_dir is None:
        search_dir = ROOT
    files = list(search_dir.glob("*_noiseless_v2_per_h.md"))
    return sorted(files)


def run_analysis(
    md_files: list[Path] | None = None,
    output_json: Path | None = None,
    summary: bool = True,
    tables: bool = False,
    latex_dir: Path | None = None,
    markdown_out: Path | None = None,
) -> ComparisonReport:
    """Main entry point: parse files, build report, output."""
    if md_files is None:
        md_files = discover_analysis_files()

    if not md_files:
        print("No analysis files found.", file=sys.stderr)
        sys.exit(1)

    all_runs: list[RunResult] = []
    for f in md_files:
        if not f.exists():
            print(f"  ⚠ File not found: {f}", file=sys.stderr)
            continue
        print(f"  Parsing: {f.name} ...", file=sys.stderr)
        runs = parse_markdown_report(f)
        print(f"    → {len(runs)} valid runs extracted", file=sys.stderr)
        all_runs.extend(runs)

    print(f"\n  Total runs: {len(all_runs)}", file=sys.stderr)

    report = build_comparison_report(all_runs)

    if summary:
        print(format_summary(report))

    # Generate thesis-quality tables
    if tables or latex_dir or markdown_out:
        thesis_tables = generate_tables(all_runs)
        md_content = "\n".join(format_table_markdown(t) for t in thesis_tables)

        if tables:
            print("\n" + md_content)

        if markdown_out:
            markdown_out.parent.mkdir(parents=True, exist_ok=True)
            markdown_out.write_text(md_content, encoding="utf-8")
            print(f"\n  Markdown tables saved to: {markdown_out}", file=sys.stderr)

        if latex_dir:
            latex_dir.mkdir(parents=True, exist_ok=True)
            for t in thesis_tables:
                latex_path = latex_dir / f"{t.table_id.lower()}_noiseless_comparison.tex"
                latex_path.write_text(format_table_latex(t), encoding="utf-8")
            print(f"\n  LaTeX tables saved to: {latex_dir}/", file=sys.stderr)

    if output_json:
        from dataclasses import asdict as _asdict

        full_data = {
            "report": report.to_dict(),
            "runs": [_asdict(r) for r in all_runs],
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as fp:
            json.dump(full_data, fp, indent=2, default=str)
        print(f"\n  JSON report saved to: {output_json}", file=sys.stderr)

    return report


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare noiseless pipeline results across models and topologies"
    )
    parser.add_argument(
        "--files",
        nargs="*",
        type=Path,
        help="Specific markdown files to analyze (defaults to auto-discovery)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output JSON file path for full structured data",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        default=True,
        help="Print human-readable summary (default: True)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Suppress human-readable summary (only write JSON)",
    )
    parser.add_argument(
        "--tables",
        action="store_true",
        help="Print thesis-quality tables to stdout",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Save markdown tables to file",
    )
    parser.add_argument(
        "--latex",
        type=Path,
        default=None,
        help="Save LaTeX tables to directory",
    )

    args = parser.parse_args()

    md_files = args.files if args.files else None
    show_summary = not args.no_summary

    run_analysis(
        md_files=md_files,
        output_json=args.output,
        summary=show_summary,
        tables=args.tables,
        latex_dir=args.latex,
        markdown_out=args.markdown,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
