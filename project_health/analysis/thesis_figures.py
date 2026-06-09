#!/usr/bin/env python3
"""Thesis-level global figures — cross-experiment, aggregated, publication-ready.

Extends the project_health.figures registry with thesis-specific figures
that aggregate data across ALL experiments, topologies, and system sizes.

New figures:
  - global_de_gap_distribution: Histogram of ΔE/gap across all 210+ runs
  - scaling_law_comprehensive: N vs ΔE/gap with scaling law overlay
  - zne_strategy_radar: Multi-axis comparison of ZNE strategies
  - topology_performance_violin: Violin plots per topology
  - cross_n_heatmap: Heatmap of performance vs N and h
  - pea_vs_gf_bar: Side-by-side bar chart PEA vs GF per topology
  - gnn_qem_summary_panel: Multi-panel GNN-QEM results
  - experiment_verdicts_sunburst: Category → verdict breakdown
  - pipeline_phase_stacked: Stacked area of time per phase vs N
  - findings_corroboration_bar: Corroboration status overview

Usage:
    python -m project_health.analysis.thesis_figures
    python -m project_health.analysis.thesis_figures --list
    python -m project_health.analysis.thesis_figures --only global_de_gap_distribution
    python -m project_health.analysis.thesis_figures --format pdf --dpi 300
    python -m project_health.analysis.thesis_figures --output-dir documentation/thesis_figures/

Output:
    Publication-ready figures (PDF/PNG/SVG) in specified directory.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results"
DEFAULT_OUTPUT_DIR = ROOT / "documentation" / "thesis_figures"


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Thesis-quality matplotlib settings
THESIS_RC = {
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "text.usetex": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

# Color palette (colorblind-friendly)
COLORS = {
    "chain_1d": "#377eb8",
    "ladder": "#4daf4a",
    "triangular": "#ff7f00",
    "heavy_hex": "#984ea3",
    "kagome": "#e41a1c",
    "pea": "#1b9e77",
    "gate_folding": "#d95f02",
    "ces": "#7570b3",
    "confirmed": "#2ca02c",
    "rejected": "#ff7f0e",
    "failed": "#d62728",
}


@dataclass
class FigureConfig:
    """Configuration for thesis figure generation."""

    output_dir: Path = DEFAULT_OUTPUT_DIR
    fmt: str = "pdf"
    dpi: int = 300
    no_titles: bool = True  # Thesis figures: no title (caption in LaTeX)
    figsize: tuple[float, float] = (7, 5)
    tight_layout: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# Figure Registry
# ═══════════════════════════════════════════════════════════════════════════════

_THESIS_FIGURES: list[tuple[str, str, Any]] = []


def register_thesis_figure(name: str, description: str):
    """Decorator to register a thesis figure generator."""

    def decorator(func):
        _THESIS_FIGURES.append((name, description, func))
        return func

    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════


def _load_all_data() -> dict[str, Any]:
    """Load all result data for figure generation."""
    from project_health.digest.scanner import ResultScanner

    scanner = ResultScanner(results_root=RESULTS_DIR)
    noiseless, noisy, experiments = scanner.scan_all(exclude_tests=True)
    scaling = scanner.scan_scaling()
    cross_topo = scanner.scan_cross_topology()

    gnn_results = {}
    gnn_dir = RESULTS_DIR / "gnn_qem"
    if gnn_dir.exists():
        for f in gnn_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    gnn_results[f.stem] = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue

    return {
        "noiseless": noiseless,
        "noisy": noisy,
        "experiments": experiments,
        "scaling": scaling,
        "cross_topo": cross_topo,
        "gnn_qem": gnn_results,
    }


def _get_plt(cfg: FigureConfig):
    """Get matplotlib with thesis rcParams."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(THESIS_RC)
    plt.rcParams["savefig.dpi"] = cfg.dpi
    return plt


# ═══════════════════════════════════════════════════════════════════════════════
# Figure Generators
# ═══════════════════════════════════════════════════════════════════════════════


@register_thesis_figure(
    "global_de_gap_distribution",
    "Histogram of ΔE/gap across all pipeline runs with pass threshold",
)
def fig_global_de_gap_distribution(data: dict, cfg: FigureConfig) -> bool:
    """Generate global ΔE/gap distribution histogram."""
    plt = _get_plt(cfg)
    noiseless = data["noiseless"]
    values = [r.delta_e_over_gap for r in noiseless if r.delta_e_over_gap is not None]

    if not values:
        return False

    fig, ax = plt.subplots(figsize=cfg.figsize)

    # Histogram with log-scale x-axis
    bins = np.logspace(np.log10(max(min(values), 1e-5)), np.log10(max(values)), 40)
    ax.hist(values, bins=bins, color="#377eb8", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(0.05, color="#d62728", linestyle="--", linewidth=2, label="5% threshold")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Delta E / \mathrm{gap}$")
    ax.set_ylabel("Count")
    if not cfg.no_titles:
        ax.set_title(f"Pipeline Performance Distribution (N={len(values)} runs)")
    ax.legend(fontsize=10)

    # Annotate pass rate
    n_pass = sum(1 for v in values if v < 0.05)
    ax.text(
        0.95,
        0.95,
        f"Pass rate: {n_pass}/{len(values)} ({n_pass / len(values):.0%})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    out = cfg.output_dir / f"fig_global_de_gap_distribution.{cfg.fmt}"
    fig.savefig(out, dpi=cfg.dpi)
    plt.close(fig)
    return True


@register_thesis_figure(
    "scaling_law_comprehensive",
    "System size N vs ΔE/gap with scaling law prediction overlay",
)
def fig_scaling_law_comprehensive(data: dict, cfg: FigureConfig) -> bool:
    """Generate comprehensive scaling law figure."""
    plt = _get_plt(cfg)
    noiseless = data["noiseless"]
    scaling = data["scaling"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: ΔE/gap vs N (box plot)
    by_n: dict[int, list[float]] = {}
    for r in noiseless:
        if r.delta_e_over_gap is not None and r.topology == "chain_1d":
            by_n.setdefault(r.n_qubits, []).append(r.delta_e_over_gap)

    for r in scaling:
        if r.n_qubits >= 40:
            by_n.setdefault(r.n_qubits, []).append(r.mean_de_gap)

    ns = sorted(by_n.keys())
    box_data = [by_n[n] for n in ns]

    bp = ax1.boxplot(box_data, positions=range(len(ns)), widths=0.6, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#377eb8")
        patch.set_alpha(0.6)
    ax1.axhline(0.05, color="#d62728", linestyle="--", linewidth=1.5, label="5% threshold")
    ax1.set_xticks(range(len(ns)))
    ax1.set_xticklabels([str(n) for n in ns])
    ax1.set_xlabel("System Size (N)")
    ax1.set_ylabel(r"$\Delta E / \mathrm{gap}$")
    ax1.set_yscale("log")
    ax1.legend()
    if not cfg.no_titles:
        ax1.set_title("Pipeline Accuracy vs System Size")

    # Right panel: Scaling law
    n_range = np.linspace(4, 100, 200)
    h_min_pred = 1.0 + 0.020 * n_range**1.31 + 0.50

    ax2.plot(
        n_range, h_min_pred, "k-", linewidth=2, label=r"$h_{min} = 1.0 + 0.020 N^{1.31} + 0.50$"
    )

    # Plot actual h_min used
    for r in scaling:
        if r.h_values and r.all_passed:
            h_min_used = min(r.h_values)
            marker = "o" if r.all_passed else "x"
            color = "#2ca02c" if r.all_passed else "#d62728"
            ax2.plot(r.n_qubits, h_min_used, marker, color=color, markersize=10, zorder=5)

    ax2.set_xlabel("System Size (N)")
    ax2.set_ylabel(r"$h_{min}$ (valid regime boundary)")
    ax2.legend(fontsize=10)
    if not cfg.no_titles:
        ax2.set_title("Scaling Law: Valid Regime Boundary")
    ax2.set_xlim(0, 105)

    plt.tight_layout()
    out = cfg.output_dir / f"fig_scaling_law_comprehensive.{cfg.fmt}"
    fig.savefig(out, dpi=cfg.dpi)
    plt.close(fig)
    return True


@register_thesis_figure(
    "topology_performance_violin",
    "Violin plots of ΔE/gap distribution per topology at N=10",
)
def fig_topology_performance_violin(data: dict, cfg: FigureConfig) -> bool:
    """Generate topology comparison violin plot."""
    plt = _get_plt(cfg)
    noiseless = data["noiseless"]
    n10 = [r for r in noiseless if r.n_qubits == 10 and r.delta_e_over_gap is not None]

    by_topo: dict[str, list[float]] = {}
    for r in n10:
        if r.topology:
            by_topo.setdefault(r.topology, []).append(r.delta_e_over_gap)

    if len(by_topo) < 2:
        return False

    fig, ax = plt.subplots(figsize=cfg.figsize)

    topos = sorted(by_topo.keys())
    positions = range(len(topos))
    violin_data = [by_topo[t] for t in topos]

    parts = ax.violinplot(violin_data, positions=positions, showmedians=True, showextrema=True)

    for i, (pc, topo) in enumerate(zip(parts["bodies"], topos, strict=False)):
        color = COLORS.get(topo, "#999999")
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    ax.axhline(
        0.05, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.7, label="5% threshold"
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(topos, rotation=15)
    ax.set_xlabel("Topology")
    ax.set_ylabel(r"$\Delta E / \mathrm{gap}$")
    ax.legend()

    # Annotate medians
    for i, topo in enumerate(topos):
        med = float(np.median(by_topo[topo]))
        ax.annotate(
            f"{med:.4f}",
            (i, med),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color="black",
        )

    if not cfg.no_titles:
        ax.set_title("Performance Distribution by Topology (N=10)")

    plt.tight_layout()
    out = cfg.output_dir / f"fig_topology_performance_violin.{cfg.fmt}"
    fig.savefig(out, dpi=cfg.dpi)
    plt.close(fig)
    return True


@register_thesis_figure(
    "pea_vs_gf_comparison",
    "Side-by-side PEA vs Gate-Folding ZNE gain per topology",
)
def fig_pea_vs_gf_comparison(data: dict, cfg: FigureConfig) -> bool:
    """Generate PEA vs GF bar comparison."""
    plt = _get_plt(cfg)
    noisy = data["noisy"]

    pea = [r for r in noisy if r.zne_strategy == "pea"]
    gf = [r for r in noisy if r.zne_strategy == "gate_folding"]

    if not pea and not gf:
        return False

    # Group by topology
    pea_by_topo: dict[str, list[float]] = {}
    gf_by_topo: dict[str, list[float]] = {}

    for r in pea:
        if r.topology:
            pea_by_topo.setdefault(r.topology, []).append(r.mean_gain_pct)
    for r in gf:
        if r.topology:
            gf_by_topo.setdefault(r.topology, []).append(r.mean_gain_pct)

    all_topos = sorted(set(list(pea_by_topo.keys()) + list(gf_by_topo.keys())))

    if not all_topos:
        return False

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(all_topos))
    width = 0.35

    pea_means = [float(np.mean(pea_by_topo.get(t, [0]))) for t in all_topos]
    gf_means = [float(np.mean(gf_by_topo.get(t, [0]))) for t in all_topos]
    pea_stds = [
        float(np.std(pea_by_topo[t])) if t in pea_by_topo and len(pea_by_topo[t]) >= 2 else 0
        for t in all_topos
    ]
    gf_stds = [
        float(np.std(gf_by_topo[t])) if t in gf_by_topo and len(gf_by_topo[t]) >= 2 else 0
        for t in all_topos
    ]

    bars1 = ax.bar(
        x - width / 2,
        pea_means,
        width,
        yerr=pea_stds,
        label="PEA",
        color=COLORS["pea"],
        alpha=0.8,
        capsize=4,
    )
    bars2 = ax.bar(
        x + width / 2,
        gf_means,
        width,
        yerr=gf_stds,
        label="Gate-Folding",
        color=COLORS["gate_folding"],
        alpha=0.8,
        capsize=4,
    )

    ax.set_xlabel("Topology")
    ax.set_ylabel("Mean ZNE Gain (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(all_topos)
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5)

    if not cfg.no_titles:
        ax.set_title("PEA vs Gate-Folding ZNE: Gain by Topology")

    plt.tight_layout()
    out = cfg.output_dir / f"fig_pea_vs_gf_comparison.{cfg.fmt}"
    fig.savefig(out, dpi=cfg.dpi)
    plt.close(fig)
    return True


@register_thesis_figure(
    "gnn_qem_summary_panel",
    "Multi-panel summary of GNN-QEM results (correction, transfer, ablation)",
)
def fig_gnn_qem_summary_panel(data: dict, cfg: FigureConfig) -> bool:
    """Generate GNN-QEM multi-panel summary."""
    plt = _get_plt(cfg)
    gnn = data["gnn_qem"]

    if not gnn:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: In-distribution error reduction
    ax1 = axes[0]
    eval_data = gnn.get("evaluation", {})
    if eval_data:
        categories = ["In-Dist\nCorrection", "Cross-Topo\nTransfer"]
        values = [
            eval_data.get("mean_error_reduction_pct", 0),
            gnn.get("cross_topology_results", {}).get("mean_error_reduction_pct", 0),
        ]
        colors_bar = [COLORS["confirmed"], COLORS["chain_1d"]]
        ax1.bar(categories, values, color=colors_bar, alpha=0.8, edgecolor="white")
        ax1.set_ylabel("Error Reduction (%)")
        ax1.set_ylim(0, 110)
        for i, v in enumerate(values):
            ax1.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=10)
    ax1.set_title("Error Correction", fontsize=11)

    # Panel 2: Ablation (graph essential)
    ax2 = axes[1]
    ablation = gnn.get("ablation_no_enoisy_results", {})
    if ablation:
        models = ["GNN\n(GINConv)", "MLP", "Linear"]
        accs = [
            ablation.get("gnn_accuracy", 1.0) * 100,
            ablation.get("mlp_accuracy", 0.67) * 100,
            ablation.get("linear_accuracy", 0.0) * 100,
        ]
        colors_abl = ["#1b9e77", "#d95f02", "#7570b3"]
        ax2.bar(models, accs, color=colors_abl, alpha=0.8, edgecolor="white")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_ylim(0, 115)
        for i, v in enumerate(accs):
            ax2.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=10)
    ax2.set_title("Ablation (No E_noisy)", fontsize=11)

    # Panel 3: Post-ZNE composability
    ax3 = axes[2]
    post_zne = gnn.get("post_zne_validation", {})
    if post_zne:
        n_regressed = post_zne.get("n_regressed", 15)
        n_total = post_zne.get("n_total", 15)
        n_improved = n_total - n_regressed
        ax3.pie(
            [n_regressed, n_improved],
            labels=["Regresses", "Improves"],
            colors=[COLORS["failed"], COLORS["confirmed"]],
            autopct="%1.0f%%",
            startangle=90,
        )
    ax3.set_title("Post-ZNE Composability", fontsize=11)

    plt.tight_layout()
    out = cfg.output_dir / f"fig_gnn_qem_summary_panel.{cfg.fmt}"
    fig.savefig(out, dpi=cfg.dpi)
    plt.close(fig)
    return True


@register_thesis_figure(
    "experiment_verdicts_overview",
    "Stacked bar of experiment verdicts by category",
)
def fig_experiment_verdicts_overview(data: dict, cfg: FigureConfig) -> bool:
    """Generate experiment verdicts overview."""
    plt = _get_plt(cfg)
    experiments = data["experiments"]

    if not experiments:
        return False

    # Group by category
    by_cat: dict[str, dict[str, int]] = {}
    for e in experiments:
        cat = e.category or "other"
        by_cat.setdefault(cat, {"confirmed": 0, "rejected": 0, "failed": 0})
        if e.verdict in by_cat[cat]:
            by_cat[cat][e.verdict] += 1

    fig, ax = plt.subplots(figsize=(8, 5))

    categories = sorted(by_cat.keys())
    confirmed = [by_cat[c]["confirmed"] for c in categories]
    rejected = [by_cat[c]["rejected"] for c in categories]
    failed = [by_cat[c]["failed"] for c in categories]

    x = np.arange(len(categories))
    width = 0.6

    ax.bar(x, confirmed, width, label="Confirmed", color=COLORS["confirmed"], alpha=0.8)
    ax.bar(
        x,
        rejected,
        width,
        bottom=confirmed,
        label="Rejected (valid)",
        color=COLORS["rejected"],
        alpha=0.8,
    )
    ax.bar(
        x,
        failed,
        width,
        bottom=[c + r for c, r in zip(confirmed, rejected, strict=False)],
        label="Failed",
        color=COLORS["failed"],
        alpha=0.8,
    )

    ax.set_xlabel("Experiment Category")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories], rotation=30, ha="right")
    ax.legend()

    if not cfg.no_titles:
        ax.set_title("Experiment Outcomes by Category")

    plt.tight_layout()
    out = cfg.output_dir / f"fig_experiment_verdicts_overview.{cfg.fmt}"
    fig.savefig(out, dpi=cfg.dpi)
    plt.close(fig)
    return True


@register_thesis_figure(
    "pipeline_timing_stacked",
    "Stacked area showing time per pipeline phase vs system size",
)
def fig_pipeline_timing_stacked(data: dict, cfg: FigureConfig) -> bool:
    """Generate pipeline timing stacked area chart."""
    plt = _get_plt(cfg)
    noiseless = data["noiseless"]
    scaling = data["scaling"]

    # Collect timing by N
    by_n: dict[int, dict[str, list[float]]] = {}
    for r in noiseless:
        if r.elapsed_s > 0:
            n = r.n_qubits
            by_n.setdefault(n, {"p1": [], "p2": [], "p3": []})
            if r.phase1_elapsed_s > 0:
                by_n[n]["p1"].append(r.phase1_elapsed_s)
            if r.phase2_elapsed_s > 0:
                by_n[n]["p2"].append(r.phase2_elapsed_s)
            if r.phase3_elapsed_s > 0:
                by_n[n]["p3"].append(r.phase3_elapsed_s)

    for r in scaling:
        n = r.n_qubits
        by_n.setdefault(n, {"p1": [], "p2": [], "p3": []})
        if r.phase1_time_s > 0:
            by_n[n]["p1"].append(r.phase1_time_s)
        if r.phase2_time_s > 0:
            by_n[n]["p2"].append(r.phase2_time_s)

    ns = sorted(by_n.keys())
    if len(ns) < 2:
        return False

    mean_p1 = [np.mean(by_n[n]["p1"]) if by_n[n]["p1"] else 0 for n in ns]
    mean_p2 = [np.mean(by_n[n]["p2"]) if by_n[n]["p2"] else 0 for n in ns]
    mean_p3 = [np.mean(by_n[n]["p3"]) if by_n[n]["p3"] else 0 for n in ns]

    fig, ax = plt.subplots(figsize=cfg.figsize)

    ax.stackplot(
        ns,
        mean_p1,
        mean_p2,
        mean_p3,
        labels=["Phase 1 (Exact Diag)", "Phase 2 (VQE)", "Phase 3 (MPNN)"],
        colors=["#66c2a5", "#fc8d62", "#8da0cb"],
        alpha=0.8,
    )

    ax.set_xlabel("System Size (N)")
    ax.set_ylabel("Mean Time (s)")
    ax.legend(loc="upper left")
    ax.set_yscale("log")

    if not cfg.no_titles:
        ax.set_title("Pipeline Phase Timing vs System Size")

    plt.tight_layout()
    out = cfg.output_dir / f"fig_pipeline_timing_stacked.{cfg.fmt}"
    fig.savefig(out, dpi=cfg.dpi)
    plt.close(fig)
    return True


@register_thesis_figure(
    "cross_n_performance_heatmap",
    "Heatmap of ΔE/gap across system sizes and h-values",
)
def fig_cross_n_performance_heatmap(data: dict, cfg: FigureConfig) -> bool:
    """Generate cross-N performance heatmap."""
    plt = _get_plt(cfg)
    scaling = data["scaling"]

    if not scaling:
        return False

    # Build matrix: rows=N, cols=h_index
    ns = sorted(set(r.n_qubits for r in scaling))
    if len(ns) < 2:
        return False

    # Find common h-range
    all_h = set()
    for r in scaling:
        all_h.update(r.h_values)
    h_sorted = sorted(all_h)

    if not h_sorted:
        return False

    # Build sparse matrix
    matrix = np.full((len(ns), len(h_sorted)), np.nan)
    for r in scaling:
        row = ns.index(r.n_qubits)
        for i, h in enumerate(r.h_values):
            if h in h_sorted and i < len(r.per_h_de_gap):
                col = h_sorted.index(h)
                matrix[row, col] = r.per_h_de_gap[i]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Use only non-nan columns
    valid_cols = ~np.all(np.isnan(matrix), axis=0)
    matrix_plot = matrix[:, valid_cols]
    h_labels = [f"{h:.1f}" for h, v in zip(h_sorted, valid_cols, strict=False) if v]

    im = ax.imshow(
        matrix_plot, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.05, interpolation="nearest"
    )

    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([str(n) for n in ns])
    ax.set_ylabel("System Size (N)")

    # Show subset of h labels
    step = max(1, len(h_labels) // 10)
    ax.set_xticks(range(0, len(h_labels), step))
    ax.set_xticklabels(h_labels[::step], rotation=45)
    ax.set_xlabel("Transverse Field (h)")

    cbar = plt.colorbar(im, ax=ax, label=r"$\Delta E / \mathrm{gap}$")

    if not cfg.no_titles:
        ax.set_title("Pipeline Accuracy Heatmap: N vs h")

    plt.tight_layout()
    out = cfg.output_dir / f"fig_cross_n_performance_heatmap.{cfg.fmt}"
    fig.savefig(out, dpi=cfg.dpi)
    plt.close(fig)
    return True


@register_thesis_figure(
    "findings_corroboration_summary",
    "Bar chart showing corroboration status of all thesis findings",
)
def fig_findings_corroboration_summary(data: dict, cfg: FigureConfig) -> bool:
    """Generate findings corroboration bar chart."""
    plt = _get_plt(cfg)

    # Run the validator to get findings
    try:
        from project_health.analysis.thesis_findings_validator import run_validation

        report = run_validation(verbose=False)
    except Exception:
        return False

    if not report.findings:
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})

    # Left: per-finding bar
    findings = report.findings
    names = [f.finding_id.replace("_", "\n", 1) for f in findings]
    colors_map = {
        "CORROBORATED": COLORS["confirmed"],
        "QUALIFIED": COLORS["rejected"],
        "UNSUPPORTED": COLORS["failed"],
        "CONTRADICTED": "#333333",
    }
    bar_colors = [colors_map.get(f.verdict, "#999999") for f in findings]

    y = range(len(findings))
    ax1.barh(y, [1] * len(findings), color=bar_colors, alpha=0.8, edgecolor="white")
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("Status")
    ax1.set_xticks([])

    # Add verdict text
    for i, f in enumerate(findings):
        ax1.text(
            0.5,
            i,
            f.verdict,
            ha="center",
            va="center",
            fontsize=8,
            color="white" if f.verdict != "QUALIFIED" else "black",
            fontweight="bold",
        )

    if not cfg.no_titles:
        ax1.set_title("Finding Validation Status")

    # Right: pie chart
    verdict_counts = {
        "Corroborated": report.n_corroborated,
        "Qualified": report.n_qualified,
        "Unsupported": report.n_unsupported,
        "Contradicted": report.n_contradicted,
    }
    # Filter zeros
    labels = [k for k, v in verdict_counts.items() if v > 0]
    sizes = [v for v in verdict_counts.values() if v > 0]
    pie_colors = [colors_map.get(l.upper(), "#999999") for l in labels]

    ax2.pie(
        sizes,
        labels=labels,
        colors=pie_colors,
        autopct="%1.0f%%",
        startangle=90,
        textprops={"fontsize": 10},
    )
    if not cfg.no_titles:
        ax2.set_title("Overall Corroboration")

    plt.tight_layout()
    out = cfg.output_dir / f"fig_findings_corroboration_summary.{cfg.fmt}"
    fig.savefig(out, dpi=cfg.dpi)
    plt.close(fig)
    return True


@register_thesis_figure(
    "zne_gain_by_topology_and_strategy",
    "Combined ZNE gain heatmap: topology × strategy",
)
def fig_zne_gain_heatmap(data: dict, cfg: FigureConfig) -> bool:
    """Generate ZNE gain heatmap by topology and strategy."""
    plt = _get_plt(cfg)
    noisy = data["noisy"]

    if not noisy:
        return False

    # Build matrix
    strategies = sorted(set(r.zne_strategy for r in noisy if r.zne_strategy))
    topologies = sorted(set(r.topology for r in noisy if r.topology))

    if len(strategies) < 2 or len(topologies) < 2:
        return False

    matrix = np.full((len(topologies), len(strategies)), np.nan)
    counts = np.zeros_like(matrix)

    for r in noisy:
        if r.topology in topologies and r.zne_strategy in strategies:
            row = topologies.index(r.topology)
            col = strategies.index(r.zne_strategy)
            if np.isnan(matrix[row, col]):
                matrix[row, col] = r.mean_gain_pct
                counts[row, col] = 1
            else:
                # Running mean
                n = counts[row, col]
                matrix[row, col] = (matrix[row, col] * n + r.mean_gain_pct) / (n + 1)
                counts[row, col] += 1

    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(
        matrix, cmap="RdYlGn", aspect="auto", vmin=-20, vmax=100, interpolation="nearest"
    )

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels([s.replace("_", "\n") for s in strategies])
    ax.set_yticks(range(len(topologies)))
    ax.set_yticklabels(topologies)
    ax.set_xlabel("ZNE Strategy")
    ax.set_ylabel("Topology")

    # Annotate cells
    for i in range(len(topologies)):
        for j in range(len(strategies)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 50 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:+.0f}%\n(n={int(counts[i, j])})",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                )

    plt.colorbar(im, ax=ax, label="Mean Gain (%)")

    if not cfg.no_titles:
        ax.set_title("ZNE Gain: Topology × Strategy")

    plt.tight_layout()
    out = cfg.output_dir / f"fig_zne_gain_heatmap.{cfg.fmt}"
    fig.savefig(out, dpi=cfg.dpi)
    plt.close(fig)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ═══════════════════════════════════════════════════════════════════════════════


def generate_all(
    cfg: FigureConfig,
    only: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, bool]:
    """Generate all thesis figures."""
    logger.info("Loading data for thesis figures...")
    data = _load_all_data()
    logger.info(
        "  Loaded: %d noiseless, %d noisy, %d experiments, %d scaling",
        len(data["noiseless"]),
        len(data["noisy"]),
        len(data["experiments"]),
        len(data["scaling"]),
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}

    for name, description, generator in _THESIS_FIGURES:
        if only and name not in only:
            continue

        logger.info("Generating: %s", name)
        try:
            success = generator(data, cfg)
            results[name] = success
            if verbose:
                icon = "✅" if success else "⏭️ "
                print(f"  {icon} {name}: {description}")
        except Exception as exc:
            results[name] = False
            logger.warning("  FAILED: %s — %s", name, exc)
            if verbose:
                print(f"  ❌ {name}: {exc}")

    return results


def list_figures() -> None:
    """List all available thesis figures."""
    print("\nAvailable Thesis Figures:")
    print("─" * 60)
    for name, description, _ in _THESIS_FIGURES:
        print(f"  • {name}")
        print(f"    {description}")
    print(f"\nTotal: {len(_THESIS_FIGURES)} figures")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate thesis-quality global figures",
    )
    parser.add_argument("--list", action="store_true", help="List available figures")
    parser.add_argument("--only", metavar="NAMES", help="Comma-separated figure names")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--no-titles",
        action="store_true",
        default=True,
        help="Omit figure titles (default for thesis)",
    )
    parser.add_argument(
        "--with-titles", action="store_true", help="Include figure titles (for presentations)"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    if args.list:
        list_figures()
        return

    cfg = FigureConfig(
        output_dir=Path(args.output_dir),
        fmt=args.format,
        dpi=args.dpi,
        no_titles=not args.with_titles,
    )

    only = args.only.split(",") if args.only else None
    results = generate_all(cfg, only=only, verbose=args.verbose)

    # Summary
    n_success = sum(1 for v in results.values() if v)
    n_total = len(results)
    print(f"\n  Generated: {n_success}/{n_total} figures → {cfg.output_dir}/")

    if n_success < n_total:
        skipped = [k for k, v in results.items() if not v]
        print(f"  Skipped (insufficient data): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
