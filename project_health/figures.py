#!/usr/bin/env python3
"""Generalized figure generation for the GNN-HVA project.

Produces thesis-quality and analysis figures from:
  1. Analysis diagnostics data (analysis/raw_data/all_diagnostics.json)
  2. Project health reports (health_report.json or HealthReport objects)
  3. Piped JSON from stdin

Architecture:
  - Each figure is a registered function decorated with @register_figure
  - Figures are grouped into categories: diagnostics, health, comparison
  - The FigureRegistry manages discovery, filtering, and execution

Usage:
    # From analysis data (original behavior)
    python -m project_health.figures --source analysis

    # From health report
    python -m project_health.figures --source health

    # Specific figures only
    python -m project_health.figures --only gen_gap,smoothness_hist

    # List available figures
    python -m project_health.figures --list

    # Custom output
    python -m project_health.figures --format svg --dpi 300 --output-dir ./figs

    # From piped JSON
    cat health_report.json | python -m project_health.figures --source stdin
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════════
# Figure Registry
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FigureSpec:
    """Metadata for a registered figure."""

    name: str
    description: str
    category: str  # "diagnostics", "health", "comparison", "validation"
    source: str  # "analysis", "health", "both"
    func: Callable[..., bool] = field(repr=False)


class FigureRegistry:
    """Registry of all available figure generators."""

    def __init__(self) -> None:
        self._figures: dict[str, FigureSpec] = {}

    def register(
        self,
        name: str,
        description: str,
        category: str = "diagnostics",
        source: str = "analysis",
    ) -> Callable:
        """Decorator to register a figure generator function."""

        def decorator(func: Callable[..., bool]) -> Callable[..., bool]:
            self._figures[name] = FigureSpec(
                name=name,
                description=description,
                category=category,
                source=source,
                func=func,
            )
            return func

        return decorator

    def list_figures(self, category: str | None = None) -> list[FigureSpec]:
        """List all registered figures, optionally filtered by category."""
        specs = list(self._figures.values())
        if category:
            specs = [s for s in specs if s.category == category]
        return sorted(specs, key=lambda s: (s.category, s.name))

    def get(self, name: str) -> FigureSpec | None:
        """Get a figure spec by name."""
        return self._figures.get(name)

    def names(self) -> list[str]:
        """Return all registered figure names."""
        return sorted(self._figures.keys())


# Global registry instance
registry = FigureRegistry()


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration & Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class FigureConfig:
    """Configuration for figure generation."""

    output_dir: Path = ROOT / "project_health" / "figures"
    format: str = "png"  # png, svg, pdf
    dpi: int = 150
    theme: str = "default"  # default, dark, minimal, thesis
    figsize_scale: float = 1.0
    show_title: bool = True
    tight_layout: bool = True


def _get_plt(theme: str = "default"):
    """Import and configure matplotlib with the specified theme."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figure generation", file=sys.stderr)
        return None

    # Apply theme
    if theme == "dark":
        plt.style.use("dark_background")
    elif theme == "minimal":
        plt.rcParams.update(
            {
                "axes.spines.top": False,
                "axes.spines.right": False,
                "font.size": 11,
            }
        )
    elif theme == "thesis":
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 11,
                "axes.labelsize": 12,
                "axes.titlesize": 13,
                "legend.fontsize": 9,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "axes.grid": True,
                "grid.alpha": 0.3,
            }
        )
    else:
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update({"figure.dpi": 100})

    return plt


def load_diagnostics_data(path: Path | None = None) -> list[dict[str, Any]]:
    """Load analysis diagnostics data."""
    if path is None:
        path = ROOT / "analysis" / "raw_data" / "all_diagnostics.json"
    if not path.exists():
        print(f"  Warning: diagnostics file not found: {path}", file=sys.stderr)
        return []
    with open(path) as f:
        return json.load(f)


def load_health_report(path: Path | None = None) -> dict[str, Any]:
    """Load a health report JSON file."""
    if path is None:
        path = ROOT / "health_report.json"
    if not path.exists():
        print(f"  Warning: health report not found: {path}", file=sys.stderr)
        return {}
    with open(path) as f:
        return json.load(f)


def load_from_stdin() -> dict[str, Any] | list[dict[str, Any]]:
    """Load JSON data from stdin (pipe support)."""
    if sys.stdin.isatty():
        return {}
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    return json.loads(raw)


# ═══════════════════════════════════════════════════════════════════════════════
# Color palettes (reused across figures)
# ═══════════════════════════════════════════════════════════════════════════════

TOPOLOGY_COLORS = {
    "chain_1d": "#2196F3",
    "ladder": "#4CAF50",
    "triangular": "#FF5722",
    "heavy_hex": "#9C27B0",
    "kagome": "#FF9800",
}

TOPOLOGY_MARKERS = {
    "chain_1d": "o",
    "ladder": "s",
    "triangular": "^",
    "heavy_hex": "D",
    "kagome": "p",
}

VERDICT_COLORS = {
    "confirmed": "#4CAF50",
    "rejected": "#FF9800",
    "failed": "#F44336",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Category: DIAGNOSTICS — figures from analysis/raw_data/all_diagnostics.json
# ═══════════════════════════════════════════════════════════════════════════════


@registry.register(
    name="gen_gap_vs_de_gap",
    description="Scatter: generalization_gap vs ΔE/gap with threshold lines",
    category="diagnostics",
    source="analysis",
)
def fig_gen_gap_vs_de_gap(data: list[dict], cfg: FigureConfig) -> bool:
    """Scatter: generalization_gap vs ΔE/gap with threshold lines."""
    plt = _get_plt(cfg.theme)
    if plt is None:
        return False

    valid = [
        d
        for d in data
        if d.get("generalization_gap") is not None and d.get("delta_e_over_gap") is not None
    ]
    if not valid:
        print("  No valid data for gen_gap_vs_de_gap", file=sys.stderr)
        return False

    fig, ax = plt.subplots(1, 1, figsize=(8 * cfg.figsize_scale, 6 * cfg.figsize_scale))

    for topo in TOPOLOGY_COLORS:
        subset = [d for d in valid if d.get("topology") == topo]
        if not subset:
            continue
        x = [d["generalization_gap"] for d in subset]
        y = [d["delta_e_over_gap"] for d in subset]
        ax.scatter(
            x,
            y,
            c=TOPOLOGY_COLORS[topo],
            marker=TOPOLOGY_MARKERS.get(topo, "o"),
            alpha=0.7,
            s=40,
            label=topo,
            edgecolors="white",
            linewidth=0.5,
        )

    ax.axhline(y=0.05, color="green", linestyle="--", alpha=0.7, label="ΔE/gap = 5% (PASS)")
    ax.axhline(y=0.10, color="orange", linestyle="--", alpha=0.7, label="ΔE/gap = 10% (FAIL)")
    ax.axvline(x=1e-2, color="red", linestyle=":", alpha=0.7, label="gen_gap = 1e-2 (abort)")
    ax.axvline(x=1e-3, color="orange", linestyle=":", alpha=0.5, label="gen_gap = 1e-3 (warn)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Generalization Gap (Phase 3)", fontsize=12)
    ax.set_ylabel("ΔE/gap (Phase 4 outcome)", fontsize=12)
    if cfg.show_title:
        ax.set_title(
            "MPNN Generalization Gap vs Pipeline Outcome\n(gen_gap > 1e-2 → 85% failure rate)",
            fontsize=13,
        )
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(1e-6, 1)
    ax.set_ylim(1e-4, 20)
    ax.grid(True, alpha=0.3)

    if cfg.tight_layout:
        plt.tight_layout()
    path = cfg.output_dir / f"fig_gen_gap_vs_de_gap.{cfg.format}"
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {path}", file=sys.stderr)
    return True


@registry.register(
    name="smoothness_histogram",
    description="Distribution of theta_smoothness by topology and system size",
    category="diagnostics",
    source="analysis",
)
def fig_smoothness_histogram(data: list[dict], cfg: FigureConfig) -> bool:
    """Histogram of theta_smoothness by topology."""
    plt = _get_plt(cfg.theme)
    if plt is None:
        return False

    valid = [d for d in data if d.get("theta_smoothness") is not None]
    if not valid:
        return False

    fig, axes = plt.subplots(2, 3, figsize=(14 * cfg.figsize_scale, 8 * cfg.figsize_scale))
    configs = [
        ("chain_1d", 6),
        ("ladder", 6),
        ("triangular", 6),
        ("chain_1d", 10),
        ("ladder", 10),
        ("triangular", 10),
    ]

    for ax, (topo, n) in zip(axes.flat, configs, strict=False):
        subset = [
            d["theta_smoothness"]
            for d in valid
            if d.get("topology") == topo and d.get("n_qubits") == n
        ]
        if not subset:
            ax.set_title(f"{topo} N={n}\n(no data)")
            continue

        clipped = [min(s, 5.0) for s in subset]
        n_breaks = sum(1 for s in subset if s >= 1.0)

        ax.hist(
            clipped,
            bins=20,
            color=TOPOLOGY_COLORS.get(topo, "#2196F3"),
            alpha=0.7,
            edgecolor="white",
        )
        ax.axvline(
            x=1.0, color="red", linestyle="--", linewidth=2, label=f"Chain break (n={n_breaks})"
        )
        ax.axvline(x=0.05, color="green", linestyle="--", alpha=0.7, label="Good threshold")
        ax.set_title(
            f"{topo} N={n}\n(breaks: {n_breaks}/{len(subset)} = {n_breaks / len(subset):.0%})",
            fontsize=11,
        )
        ax.set_xlabel("θ_smoothness")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    if cfg.show_title:
        plt.suptitle(
            "Distribution of θ_smoothness by Topology\n(>1.0 = warm-start chain break)",
            fontsize=14,
            y=1.02,
        )
    if cfg.tight_layout:
        plt.tight_layout()
    path = cfg.output_dir / f"fig_smoothness_histogram.{cfg.format}"
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {path}", file=sys.stderr)
    return True


@registry.register(
    name="cross_topology_bar",
    description="Bar chart: pass rate and median error by topology",
    category="diagnostics",
    source="analysis",
)
def fig_cross_topology_bar(data: list[dict], cfg: FigureConfig) -> bool:
    """Bar chart: pass rate by topology."""
    plt = _get_plt(cfg.theme)
    if plt is None:
        return False

    valid = [d for d in data if d.get("delta_e_over_gap") is not None]
    if not valid:
        return False

    configs = [
        ("chain_1d", 6),
        ("ladder", 6),
        ("ladder", 10),
        ("triangular", 6),
        ("triangular", 10),
    ]
    labels = []
    pass_rates = []
    medians = []
    counts = []

    for topo, n in configs:
        subset = [
            d["delta_e_over_gap"]
            for d in valid
            if d.get("topology") == topo and d.get("n_qubits") == n
        ]
        if subset:
            labels.append(f"{topo}\nN={n}")
            pass_rates.append(sum(1 for v in subset if v < 0.05) / len(subset))
            medians.append(sorted(subset)[len(subset) // 2])
            counts.append(len(subset))

    if not labels:
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12 * cfg.figsize_scale, 5 * cfg.figsize_scale))

    colors = [
        TOPOLOGY_COLORS.get(c[0], "#607D8B")
        for c in configs
        if any(d.get("topology") == c[0] and d.get("n_qubits") == c[1] for d in valid)
    ][: len(labels)]

    bars = ax1.bar(
        labels,
        [r * 100 for r in pass_rates],
        color=colors,
        alpha=0.8,
        edgecolor="white",
        linewidth=1.5,
    )
    ax1.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Pass Rate (%)", fontsize=12)
    ax1.set_title("Pipeline Pass Rate (ΔE/gap < 5%)", fontsize=13)
    ax1.set_ylim(0, 100)
    for bar, count in zip(bars, counts, strict=False):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"n={count}",
            ha="center",
            fontsize=9,
        )

    ax2.bar(
        labels,
        [m * 100 for m in medians],
        color=colors,
        alpha=0.8,
        edgecolor="white",
        linewidth=1.5,
    )
    ax2.axhline(y=5, color="green", linestyle="--", alpha=0.7, label="5% threshold")
    ax2.set_ylabel("Median ΔE/gap (%)", fontsize=12)
    ax2.set_title("Median Energy Error by Topology", fontsize=13)
    ax2.legend()

    if cfg.show_title:
        plt.suptitle("GNN-HVA Framework: Cross-Topology Performance", fontsize=14, y=1.02)
    if cfg.tight_layout:
        plt.tight_layout()
    path = cfg.output_dir / f"fig_cross_topology_bar.{cfg.format}"
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {path}", file=sys.stderr)
    return True


@registry.register(
    name="smoothness_vs_de_gap",
    description="Scatter: theta_smoothness vs ΔE/gap (chain break threshold effect)",
    category="diagnostics",
    source="analysis",
)
def fig_smoothness_vs_de_gap(data: list[dict], cfg: FigureConfig) -> bool:
    """Scatter: theta_smoothness vs ΔE/gap showing the threshold effect."""
    plt = _get_plt(cfg.theme)
    if plt is None:
        return False

    valid = [
        d
        for d in data
        if d.get("theta_smoothness") is not None and d.get("delta_e_over_gap") is not None
    ]
    if not valid:
        return False

    fig, ax = plt.subplots(1, 1, figsize=(8 * cfg.figsize_scale, 6 * cfg.figsize_scale))

    for topo in TOPOLOGY_COLORS:
        subset = [d for d in valid if d.get("topology") == topo]
        if not subset:
            continue
        x = [d["theta_smoothness"] for d in subset]
        y = [d["delta_e_over_gap"] for d in subset]
        ax.scatter(
            x,
            y,
            c=TOPOLOGY_COLORS[topo],
            alpha=0.6,
            s=40,
            label=topo,
            edgecolors="white",
            linewidth=0.5,
        )

    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=2, label="Chain break threshold")
    ax.axhline(y=0.05, color="green", linestyle="--", alpha=0.7, label="5% PASS")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("θ_smoothness (Phase 2)", fontsize=12)
    ax.set_ylabel("ΔE/gap (Phase 4 outcome)", fontsize=12)
    if cfg.show_title:
        ax.set_title(
            "Warm-Start Chain Smoothness vs Pipeline Outcome\n"
            "(θ > 1.0 = chain break → 76% failure/marginal)",
            fontsize=13,
        )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if cfg.tight_layout:
        plt.tight_layout()
    path = cfg.output_dir / f"fig_smoothness_vs_de_gap.{cfg.format}"
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {path}", file=sys.stderr)
    return True


@registry.register(
    name="error_decomposition",
    description="Scatter: circuit error vs MPNN error contribution",
    category="diagnostics",
    source="analysis",
)
def fig_error_decomposition(data: list[dict], cfg: FigureConfig) -> bool:
    """Scatter plot of circuit error vs MPNN error contribution."""
    plt = _get_plt(cfg.theme)
    if plt is None:
        return False

    valid = [
        d
        for d in data
        if d.get("error_from_circuit") is not None
        and d.get("error_from_mpnn") is not None
        and d.get("delta_e_over_gap") is not None
    ]
    if not valid:
        return False

    fig, ax = plt.subplots(1, 1, figsize=(8 * cfg.figsize_scale, 6 * cfg.figsize_scale))

    for topo in TOPOLOGY_COLORS:
        subset = [d for d in valid if d.get("topology") == topo]
        if not subset:
            continue
        x = [d["error_from_circuit"] for d in subset]
        y = [d["error_from_mpnn"] for d in subset]
        ax.scatter(
            x,
            y,
            c=TOPOLOGY_COLORS[topo],
            marker=TOPOLOGY_MARKERS.get(topo, "o"),
            alpha=0.6,
            s=40,
            label=topo,
            edgecolors="white",
            linewidth=0.5,
        )

    # Diagonal line (equal contribution)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Equal contribution")
    ax.set_xlabel("Circuit Error Fraction", fontsize=12)
    ax.set_ylabel("MPNN Error Fraction", fontsize=12)
    if cfg.show_title:
        ax.set_title("Error Decomposition: Circuit vs MPNN Contribution", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if cfg.tight_layout:
        plt.tight_layout()
    path = cfg.output_dir / f"fig_error_decomposition.{cfg.format}"
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {path}", file=sys.stderr)
    return True


@registry.register(
    name="h_test_vs_de_gap",
    description="Scatter: h_test value vs pipeline outcome (regime boundary check)",
    category="diagnostics",
    source="analysis",
)
def fig_h_test_vs_de_gap(data: list[dict], cfg: FigureConfig) -> bool:
    """Scatter: h_test vs ΔE/gap showing valid regime boundaries."""
    plt = _get_plt(cfg.theme)
    if plt is None:
        return False

    valid = [
        d for d in data if d.get("h_test") is not None and d.get("delta_e_over_gap") is not None
    ]
    if not valid:
        return False

    fig, ax = plt.subplots(1, 1, figsize=(9 * cfg.figsize_scale, 6 * cfg.figsize_scale))

    for topo in TOPOLOGY_COLORS:
        subset = [d for d in valid if d.get("topology") == topo]
        if not subset:
            continue
        x = [d["h_test"] for d in subset]
        y = [d["delta_e_over_gap"] for d in subset]
        ax.scatter(
            x,
            y,
            c=TOPOLOGY_COLORS[topo],
            marker=TOPOLOGY_MARKERS.get(topo, "o"),
            alpha=0.6,
            s=40,
            label=topo,
            edgecolors="white",
            linewidth=0.5,
        )

    ax.axhline(y=0.05, color="green", linestyle="--", alpha=0.7, label="5% PASS")
    ax.axhline(y=0.10, color="orange", linestyle="--", alpha=0.5, label="10% FAIL")
    # Regime boundaries (approximate, from steering)
    ax.axvline(x=1.25, color="#2196F3", linestyle=":", alpha=0.4, label="chain p=2 boundary")
    ax.axvline(x=1.6, color="#2196F3", linestyle="-.", alpha=0.4, label="chain p=1 boundary")

    ax.set_yscale("log")
    ax.set_xlabel("h_test (transverse field)", fontsize=12)
    ax.set_ylabel("ΔE/gap (Phase 4 outcome)", fontsize=12)
    if cfg.show_title:
        ax.set_title("Field Strength vs Pipeline Outcome (Regime Boundary Check)", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    if cfg.tight_layout:
        plt.tight_layout()
    path = cfg.output_dir / f"fig_h_test_vs_de_gap.{cfg.format}"
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {path}", file=sys.stderr)
    return True


@registry.register(
    name="restarts_vs_outcome",
    description="Box plot: number of VQE restarts vs pipeline outcome",
    category="diagnostics",
    source="analysis",
)
def fig_restarts_vs_outcome(data: list[dict], cfg: FigureConfig) -> bool:
    """Box plot: n_restarts grouped by pass/fail outcome."""
    plt = _get_plt(cfg.theme)
    if plt is None:
        return False

    valid = [
        d for d in data if d.get("n_restarts") is not None and d.get("delta_e_over_gap") is not None
    ]
    if not valid:
        return False

    fig, ax = plt.subplots(1, 1, figsize=(8 * cfg.figsize_scale, 5 * cfg.figsize_scale))

    # Group by n_restarts
    restart_values = sorted(set(d["n_restarts"] for d in valid))
    box_data = []
    box_labels = []
    for r in restart_values:
        subset = [d["delta_e_over_gap"] for d in valid if d["n_restarts"] == r]
        if len(subset) >= 3:
            box_data.append(subset)
            box_labels.append(str(r))

    if not box_data:
        plt.close()
        return False

    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#2196F3")
        patch.set_alpha(0.6)

    ax.axhline(y=0.05, color="green", linestyle="--", alpha=0.7, label="5% PASS")
    ax.set_yscale("log")
    ax.set_xlabel("Number of VQE Restarts", fontsize=12)
    ax.set_ylabel("ΔE/gap (Phase 4 outcome)", fontsize=12)
    if cfg.show_title:
        ax.set_title("VQE Restarts vs Pipeline Outcome", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    if cfg.tight_layout:
        plt.tight_layout()
    path = cfg.output_dir / f"fig_restarts_vs_outcome.{cfg.format}"
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {path}", file=sys.stderr)
    return True


@registry.register(
    name="hidden_dim_comparison",
    description="Violin plot: hidden_dim impact on ΔE/gap",
    category="diagnostics",
    source="analysis",
)
def fig_hidden_dim_comparison(data: list[dict], cfg: FigureConfig) -> bool:
    """Violin plot showing hidden_dim impact on outcome."""
    plt = _get_plt(cfg.theme)
    if plt is None:
        return False

    valid = [
        d for d in data if d.get("hidden_dim") is not None and d.get("delta_e_over_gap") is not None
    ]
    if not valid:
        return False

    fig, ax = plt.subplots(1, 1, figsize=(8 * cfg.figsize_scale, 5 * cfg.figsize_scale))

    dim_values = sorted(set(d["hidden_dim"] for d in valid))
    positions = []
    violin_data = []
    dim_labels = []

    for _i, dim in enumerate(dim_values):
        subset = [d["delta_e_over_gap"] for d in valid if d["hidden_dim"] == dim]
        if len(subset) >= 3:
            violin_data.append(subset)
            positions.append(len(positions))
            dim_labels.append(str(dim))

    if len(violin_data) < 2:
        plt.close()
        return False

    parts = ax.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#2196F3")
        pc.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(dim_labels)
    ax.axhline(y=0.05, color="green", linestyle="--", alpha=0.7, label="5% PASS")
    ax.set_yscale("log")
    ax.set_xlabel("MPNN Hidden Dimension", fontsize=12)
    ax.set_ylabel("ΔE/gap (Phase 4 outcome)", fontsize=12)
    if cfg.show_title:
        ax.set_title("MPNN Hidden Dimension vs Pipeline Outcome", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    if cfg.tight_layout:
        plt.tight_layout()
    path = cfg.output_dir / f"fig_hidden_dim_comparison.{cfg.format}"
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {path}", file=sys.stderr)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Category: HEALTH — figures from health_report.json
# ═══════════════════════════════════════════════════════════════════════════════


@registry.register(
    name="experiment_verdicts",
    description="Stacked bar: experiment verdicts overview (confirmed/rejected/failed)",
    category="health",
    source="health",
)
def fig_experiment_verdicts(report: dict[str, Any], cfg: FigureConfig) -> bool:
    """Bar chart of experiment verdicts with pass rates."""
    plt = _get_plt(cfg.theme)
    if plt is None:
        return False

    experiments = report.get("experiments", [])
    if not experiments:
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14 * cfg.figsize_scale, 6 * cfg.figsize_scale))

    # Left: verdict pie chart
    n_confirmed = sum(1 for e in experiments if e["verdict"] == "confirmed")
    n_rejected = sum(1 for e in experiments if e["verdict"] == "rejected")
    n_failed = sum(1 for e in experiments if e["verdict"] == "failed")

    sizes = [n_confirmed, n_rejected, n_failed]
    labels_pie = [f"Confirmed ({n_confirmed})", f"Rejected ({n_rejected})", f"Failed ({n_failed})"]
    colors_pie = [VERDICT_COLORS["confirmed"], VERDICT_COLORS["rejected"], VERDICT_COLORS["failed"]]
    explode = (0.02, 0.02, 0.05)

    ax1.pie(
        sizes,
        explode=explode,
        labels=labels_pie,
        colors=colors_pie,
        autopct="%1.0f%%",
        shadow=False,
        startangle=90,
        textprops={"fontsize": 10},
    )
    ax1.set_title("Experiment Verdict Distribution", fontsize=13)

    # Right: horizontal bar with pass rates
    sorted_exps = sorted(experiments, key=lambda e: e.get("pass_rate") or 0, reverse=True)
    exp_ids = [e["experiment_id"] for e in sorted_exps]
    pass_rates = [e.get("pass_rate") or 0 for e in sorted_exps]
    bar_colors = [VERDICT_COLORS.get(e["verdict"], "#607D8B") for e in sorted_exps]

    y_pos = range(len(exp_ids))
    ax2.barh(y_pos, [r * 100 for r in pass_rates], color=bar_colors, alpha=0.8, edgecolor="white")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(exp_ids, fontsize=8)
    ax2.axvline(x=50, color="gray", linestyle="--", alpha=0.4)
    ax2.set_xlabel("Pass Rate (%)", fontsize=11)
    ax2.set_title("Per-Experiment Pass Rate", fontsize=13)
    ax2.set_xlim(0, 105)
    ax2.invert_yaxis()

    if cfg.show_title:
        plt.suptitle("Project Health: Experiment Overview", fontsize=14, y=1.02)
    if cfg.tight_layout:
        plt.tight_layout()
    path = cfg.output_dir / f"fig_experiment_verdicts.{cfg.format}"
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {path}", file=sys.stderr)
    return True


@registry.register(
    name="topology_comparison",
    description="Grouped bar: pass rate, median ΔE, and run count by topology",
    category="health",
    source="health",
)
def fig_topology_comparison(report: dict[str, Any], cfg: FigureConfig) -> bool:
    """Grouped bar chart from health report topology stats."""
    plt = _get_plt(cfg.theme)
    if plt is None:
        return False

    by_topo = report.get("noiseless_by_topology", {})
    if not by_topo:
        return False

    topos = list(by_topo.keys())
    n_runs = [by_topo[t]["n_runs"] for t in topos]
    pass_rates = [by_topo[t]["pass_rate"] * 100 for t in topos]
    median_des = [by_topo[t]["median_de"] * 100 for t in topos]
    colors = [TOPOLOGY_COLORS.get(t, "#607D8B") for t in topos]

    fig, axes = plt.subplots(1, 3, figsize=(14 * cfg.figsize_scale, 5 * cfg.figsize_scale))

    # Pass rate
    axes[0].bar(topos, pass_rates, color=colors, alpha=0.8, edgecolor="white")
    axes[0].axhline(y=50, color="gray", linestyle="--", alpha=0.4)
    axes[0].set_ylabel("Pass Rate (%)")
    axes[0].set_title("Pass Rate (ΔE/gap < 5%)")
    axes[0].set_ylim(0, 100)
    axes[0].tick_params(axis="x", rotation=30)

    # Median ΔE
    axes[1].bar(topos, median_des, color=colors, alpha=0.8, edgecolor="white")
    axes[1].axhline(y=5, color="green", linestyle="--", alpha=0.7, label="5% threshold")
    axes[1].set_ylabel("Median ΔE/gap (%)")
    axes[1].set_title("Median Energy Error")
    axes[1].legend(fontsize=9)
    axes[1].tick_params(axis="x", rotation=30)

    # Run count
    axes[2].bar(topos, n_runs, color=colors, alpha=0.8, edgecolor="white")
    axes[2].set_ylabel("Number of Runs")
    axes[2].set_title("Data Coverage")
    axes[2].tick_params(axis="x", rotation=30)

    if cfg.show_title:
        plt.suptitle("Project Health: Topology Comparison", fontsize=14, y=1.02)
    if cfg.tight_layout:
        plt.tight_layout()
    path = cfg.output_dir / f"fig_topology_comparison.{cfg.format}"
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {path}", file=sys.stderr)
    return True


@registry.register(
    name="coverage_gaps_summary",
    description="Heatmap: coverage gaps by topology × priority",
    category="health",
    source="health",
)
def fig_coverage_gaps_summary(report: dict[str, Any], cfg: FigureConfig) -> bool:
    """Summary visualization of coverage gaps."""
    plt = _get_plt(cfg.theme)
    if plt is None:
        return False

    gaps = report.get("gaps", [])
    if not gaps:
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12 * cfg.figsize_scale, 5 * cfg.figsize_scale))

    # Left: gaps by type
    gap_types: dict[str, int] = {}
    for g in gaps:
        gtype = g.get("gap_type", "unknown")
        gap_types[gtype] = gap_types.get(gtype, 0) + 1

    types_sorted = sorted(gap_types.items(), key=lambda x: x[1], reverse=True)
    labels = [t[0].replace("_", " ") for t in types_sorted]
    values = [t[1] for t in types_sorted]

    ax1.barh(labels, values, color="#FF9800", alpha=0.8, edgecolor="white")
    ax1.set_xlabel("Count")
    ax1.set_title("Gap Types Distribution")

    # Right: gaps by priority
    prio_map = {1: "CRITICAL", 2: "HIGH", 3: "MEDIUM", 4: "LOW"}
    prio_counts: dict[str, int] = {}
    for g in gaps:
        p = g.get("priority", 4)
        pname = prio_map.get(p, f"P{p}")
        prio_counts[pname] = prio_counts.get(pname, 0) + 1

    prio_colors = {"CRITICAL": "#F44336", "HIGH": "#FF9800", "MEDIUM": "#FFC107", "LOW": "#9E9E9E"}
    plabels = list(prio_counts.keys())
    pvalues = list(prio_counts.values())
    pcolors = [prio_colors.get(p, "#607D8B") for p in plabels]

    ax2.bar(plabels, pvalues, color=pcolors, alpha=0.8, edgecolor="white")
    ax2.set_ylabel("Count")
    ax2.set_title("Gaps by Priority")

    if cfg.show_title:
        plt.suptitle(f"Coverage Gaps Summary ({len(gaps)} total)", fontsize=14, y=1.02)
    if cfg.tight_layout:
        plt.tight_layout()
    path = cfg.output_dir / f"fig_coverage_gaps_summary.{cfg.format}"
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {path}", file=sys.stderr)
    return True


@registry.register(
    name="noisy_quality_overview",
    description="ZNE quality overview: R², gain, success rate",
    category="health",
    source="health",
)
def fig_noisy_quality_overview(report: dict[str, Any], cfg: FigureConfig) -> bool:
    """Dashboard-style overview of noisy/ZNE quality metrics."""
    plt = _get_plt(cfg.theme)
    if plt is None:
        return False

    n_noisy = report.get("n_noisy", 0)
    if n_noisy == 0:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(12 * cfg.figsize_scale, 4 * cfg.figsize_scale))

    # Metric gauges
    metrics = [
        ("Success Rate", report.get("noisy_success_rate", 0), 1.0, "%"),
        ("Mean R²", report.get("noisy_mean_r2", 0), 1.0, ""),
        ("Mean ZNE Gain", report.get("noisy_mean_gain", 0), 100.0, "%"),
    ]

    for ax, (label, value, max_val, unit) in zip(axes, metrics, strict=False):
        # Simple gauge using a horizontal bar
        color = (
            "#4CAF50"
            if value > max_val * 0.7
            else "#FF9800"
            if value > max_val * 0.4
            else "#F44336"
        )
        if unit == "%":
            display_val = f"{value * 100:.1f}%" if max_val <= 1 else f"+{value:.1f}%"
        else:
            display_val = f"{value:.4f}"

        ax.barh([0], [value], color=color, alpha=0.8, height=0.4)
        if max_val <= 1:
            ax.set_xlim(0, max_val)
        ax.set_yticks([])
        ax.set_xlabel(label)
        ax.text(
            0.5,
            0.7,
            display_val,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
        )
        ax.set_title(label, fontsize=11)

    if cfg.show_title:
        plt.suptitle(f"Noisy/ZNE Quality Overview ({n_noisy} runs)", fontsize=13, y=1.05)
    if cfg.tight_layout:
        plt.tight_layout()
    path = cfg.output_dir / f"fig_noisy_quality_overview.{cfg.format}"
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {path}", file=sys.stderr)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Category: HEALTH — VQE/MPNN quality + distribution (from health report JSON)
# ═══════════════════════════════════════════════════════════════════════════════


@registry.register(
    name="vqe_quality_dashboard",
    description="VQE convergence quality: convergence rate dist + θ-smoothness + chain breaks",
    category="health",
    source="health",
)
def fig_vqe_quality_dashboard(report: dict[str, Any], cfg: FigureConfig) -> bool:
    """Dashboard of VQE convergence quality metrics from health report."""
    plt = _get_plt(cfg.theme)
    if plt is None:
        return False

    vqe = report.get("vqe_quality", {})
    if not vqe or vqe.get("n_results", 0) == 0:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(14 * cfg.figsize_scale, 5 * cfg.figsize_scale))

    n_results = vqe["n_results"]
    conv_mean = vqe.get("convergence_rate_mean", 0)
    conv_min = vqe.get("convergence_rate_min", 0)
    smooth_mean = vqe.get("theta_smoothness_mean", 0)
    smooth_max = vqe.get("theta_smoothness_max", 0)
    n_breaks = vqe.get("n_chain_break_warnings", 0)

    # Panel 1: Convergence rate gauge
    ax = axes[0]
    bars = ax.bar(
        ["Mean", "Min"],
        [conv_mean * 100, conv_min * 100],
        color=["#4CAF50", "#FF9800" if conv_min < 0.8 else "#4CAF50"],
        alpha=0.8,
        edgecolor="white",
    )
    ax.axhline(y=80, color="red", linestyle="--", alpha=0.6, label="80% threshold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Convergence Rate (%)")
    ax.set_title("VQE Convergence")
    ax.legend(fontsize=9)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", fontsize=10)

    # Panel 2: θ-smoothness
    ax = axes[1]
    ax.bar(
        ["Mean", "Max"],
        [smooth_mean, smooth_max],
        color=["#2196F3", "#F44336" if smooth_max > 1.0 else "#2196F3"],
        alpha=0.8,
        edgecolor="white",
    )
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.6, label="Chain break threshold")
    ax.set_ylabel("θ-smoothness")
    ax.set_title("Angle Smoothness")
    ax.legend(fontsize=9)

    # Panel 3: Chain break pie
    ax = axes[2]
    n_ok = n_results - n_breaks
    if n_breaks > 0:
        ax.pie(
            [n_ok, n_breaks],
            labels=[f"OK ({n_ok})", f"Break ({n_breaks})"],
            colors=["#4CAF50", "#F44336"],
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": 10},
        )
    else:
        ax.pie(
            [n_ok],
            labels=[f"All OK ({n_ok})"],
            colors=["#4CAF50"],
            autopct="%1.0f%%",
            startangle=90,
        )
    ax.set_title("Chain Breaks (θ > 1.0)")

    if cfg.show_title:
        plt.suptitle(f"VQE Quality Dashboard ({n_results} runs)", fontsize=14, y=1.02)
    if cfg.tight_layout:
        plt.tight_layout()
    path = cfg.output_dir / f"fig_vqe_quality_dashboard.{cfg.format}"
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {path}", file=sys.stderr)
    return True


@registry.register(
    name="mpnn_quality_dashboard",
    description="MPNN training quality: generalization gap dist + overfit warnings",
    category="health",
    source="health",
)
def fig_mpnn_quality_dashboard(report: dict[str, Any], cfg: FigureConfig) -> bool:
    """Dashboard of MPNN training quality metrics from health report."""
    plt = _get_plt(cfg.theme)
    if plt is None:
        return False

    mpnn = report.get("mpnn_quality", {})
    if not mpnn or mpnn.get("n_results", 0) == 0:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(14 * cfg.figsize_scale, 5 * cfg.figsize_scale))

    n_results = mpnn["n_results"]
    gap_mean = mpnn.get("gen_gap_mean", 0)
    gap_median = mpnn.get("gen_gap_median", 0)
    gap_max = mpnn.get("gen_gap_max", 0)
    n_overfit = mpnn.get("n_overfit_warnings", 0)
    mse_mean = mpnn.get("theta_mse_mean", 0)

    # Panel 1: Gen gap summary bars
    ax = axes[0]
    labels = ["Mean", "Median", "Max"]
    values = [gap_mean, gap_median, gap_max]
    colors = ["#F44336" if v > 0.01 else "#4CAF50" for v in values]
    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor="white")  # noqa: F841
    ax.axhline(y=0.01, color="red", linestyle="--", alpha=0.6, label="Overfit threshold")
    ax.set_ylabel("Generalization Gap")
    ax.set_title("Gen. Gap Summary")
    ax.legend(fontsize=9)
    ax.set_yscale("log")

    # Panel 2: Overfit pie
    ax = axes[1]
    n_ok = n_results - n_overfit
    if n_overfit > 0:
        ax.pie(
            [n_ok, n_overfit],
            labels=[f"OK ({n_ok})", f"Overfit ({n_overfit})"],
            colors=["#4CAF50", "#FF9800"],
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": 10},
        )
    else:
        ax.pie(
            [n_ok],
            labels=[f"All OK ({n_ok})"],
            colors=["#4CAF50"],
            autopct="%1.0f%%",
            startangle=90,
        )
    ax.set_title("Overfit Status (gap > 0.01)")

    # Panel 3: θ-MSE gauge
    ax = axes[2]
    color = "#F44336" if mse_mean > 0.01 else "#4CAF50"
    ax.barh([0], [mse_mean], color=color, alpha=0.8, height=0.4)
    ax.axvline(x=0.01, color="red", linestyle="--", alpha=0.6, label="Alert threshold")
    ax.set_yticks([])
    ax.set_xlabel("θ-MSE")
    ax.set_title("Mean θ Prediction MSE")
    ax.text(
        0.5,
        0.7,
        f"{mse_mean:.5f}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )
    ax.legend(fontsize=9)

    if cfg.show_title:
        plt.suptitle(f"MPNN Quality Dashboard ({n_results} runs)", fontsize=14, y=1.02)
    if cfg.tight_layout:
        plt.tight_layout()
    path = cfg.output_dir / f"fig_mpnn_quality_dashboard.{cfg.format}"
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {path}", file=sys.stderr)
    return True


@registry.register(
    name="result_distribution",
    description="Bar charts: result distribution by model, topology, N-qubits, p-layers",
    category="health",
    source="health",
)
def fig_result_distribution(report: dict[str, Any], cfg: FigureConfig) -> bool:
    """Multi-panel bar chart showing result distribution across dimensions."""
    plt = _get_plt(cfg.theme)
    if plt is None:
        return False

    dist = report.get("distribution", {})
    if not dist:
        return False

    by_model = dist.get("by_model", {})
    by_topology = dist.get("by_topology", {})
    by_nq = dist.get("by_n_qubits", {})
    by_pl = dist.get("by_p_layers", {})

    if not by_topology and not by_model:
        return False

    n_panels = sum(1 for d in [by_model, by_topology, by_nq, by_pl] if d)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(4 * n_panels * cfg.figsize_scale, 5 * cfg.figsize_scale),
    )
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    if by_model:
        ax = axes[panel_idx]
        labels = sorted(by_model.keys())
        values = [by_model[k] for k in labels]
        ax.bar(labels, values, color="#2196F3", alpha=0.8, edgecolor="white")
        ax.set_title("By Model")
        ax.set_ylabel("Run Count")
        ax.tick_params(axis="x", rotation=30)
        panel_idx += 1

    if by_topology:
        ax = axes[panel_idx]
        labels = sorted(by_topology.keys())
        values = [by_topology[k] for k in labels]
        colors = [TOPOLOGY_COLORS.get(t, "#607D8B") for t in labels]
        ax.bar(labels, values, color=colors, alpha=0.8, edgecolor="white")
        ax.set_title("By Topology")
        ax.set_ylabel("Run Count")
        ax.tick_params(axis="x", rotation=30)
        panel_idx += 1

    if by_nq:
        ax = axes[panel_idx]
        labels = [str(k) for k in sorted(int(k) for k in by_nq)]
        values = [by_nq[k] for k in sorted(by_nq, key=lambda x: int(x))]
        ax.bar(labels, values, color="#9C27B0", alpha=0.8, edgecolor="white")
        ax.set_title("By N-qubits")
        ax.set_ylabel("Run Count")
        ax.set_xlabel("N")
        panel_idx += 1

    if by_pl:
        ax = axes[panel_idx]
        labels = [f"p={k}" for k in sorted(int(k) for k in by_pl)]
        values = [by_pl[k] for k in sorted(by_pl, key=lambda x: int(x))]
        ax.bar(labels, values, color="#FF5722", alpha=0.8, edgecolor="white")
        ax.set_title("By p-layers")
        ax.set_ylabel("Run Count")
        panel_idx += 1

    if cfg.show_title:
        total = sum(by_topology.values()) if by_topology else sum(by_model.values())
        plt.suptitle(f"Result Distribution ({total} runs)", fontsize=14, y=1.02)
    if cfg.tight_layout:
        plt.tight_layout()
    path = cfg.output_dir / f"fig_result_distribution.{cfg.format}"
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {path}", file=sys.stderr)
    return True


@registry.register(
    name="energy_decomposition_pie",
    description="Pie chart: energy error attribution (circuit vs MPNN)",
    category="health",
    source="health",
)
def fig_energy_decomposition_pie(report: dict[str, Any], cfg: FigureConfig) -> bool:
    """Pie chart showing circuit vs MPNN error attribution."""
    plt = _get_plt(cfg.theme)
    if plt is None:
        return False

    ed = report.get("energy_decomposition", {})
    if not ed or ed.get("n_results", 0) == 0:
        return False

    circuit_frac = ed.get("circuit_error_fraction", 0)
    mpnn_frac = ed.get("mpnn_error_fraction", 0)

    if circuit_frac == 0 and mpnn_frac == 0:
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11 * cfg.figsize_scale, 5 * cfg.figsize_scale))

    # Left: pie chart
    sizes = [circuit_frac, mpnn_frac]
    labels = [
        f"Circuit\n({circuit_frac:.0%})",
        f"MPNN\n({mpnn_frac:.0%})",
    ]
    colors = ["#2196F3", "#FF9800"]
    explode = (0.03, 0.03)

    ax1.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=False,
        startangle=90,
        textprops={"fontsize": 11},
    )
    ax1.set_title("Error Source Attribution", fontsize=12)

    # Right: mean values comparison
    mean_circuit = ed.get("mean_circuit_error", 0)
    mean_mpnn = ed.get("mean_mpnn_error", 0)
    bars = ax2.bar(
        ["Circuit Error", "MPNN Error"],
        [mean_circuit, mean_mpnn],
        color=colors,
        alpha=0.8,
        edgecolor="white",
    )
    ax2.set_ylabel("Mean Error (ΔE/gap fraction)")
    ax2.set_title("Mean Error by Source")
    for bar in bars:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.002,
            f"{h:.4f}",
            ha="center",
            fontsize=10,
        )

    if cfg.show_title:
        plt.suptitle(
            f"Energy Error Decomposition ({ed['n_results']} runs analyzed)",
            fontsize=14,
            y=1.02,
        )
    if cfg.tight_layout:
        plt.tight_layout()
    path = cfg.output_dir / f"fig_energy_decomposition_pie.{cfg.format}"
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {path}", file=sys.stderr)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Category: COMPARISON — cross-cutting analytical figures
# ═══════════════════════════════════════════════════════════════════════════════


@registry.register(
    name="failure_mode_breakdown",
    description="Pie chart: failure mode distribution from diagnostics",
    category="comparison",
    source="analysis",
)
def fig_failure_mode_breakdown(data: list[dict], cfg: FigureConfig) -> bool:
    """Pie chart: root cause failure distribution."""
    plt = _get_plt(cfg.theme)
    if plt is None:
        return False

    # Classify failures using known thresholds
    fails = [
        d for d in data if d.get("delta_e_over_gap") is not None and d["delta_e_over_gap"] >= 0.05
    ]
    if len(fails) < 5:
        return False

    categories: dict[str, int] = {
        "CHAIN_BREAK (θ>1.0)": 0,
        "MPNN_OVERFIT (gap>0.01)": 0,
        "BOUNDARY_EFFECT": 0,
        "VQE_DIVERGENCE": 0,
        "OTHER": 0,
    }

    for d in fails:
        theta = d.get("theta_smoothness")
        gen_gap = d.get("generalization_gap")
        conv = d.get("convergence_rate")
        h_test = d.get("h_test")

        if theta is not None and theta > 1.0:
            categories["CHAIN_BREAK (θ>1.0)"] += 1
        elif gen_gap is not None and gen_gap > 0.01:
            categories["MPNN_OVERFIT (gap>0.01)"] += 1
        elif h_test is not None and h_test < 1.5:
            categories["BOUNDARY_EFFECT"] += 1
        elif conv is not None and conv < 0.5:
            categories["VQE_DIVERGENCE"] += 1
        else:
            categories["OTHER"] += 1

    # Remove zero categories
    categories = {k: v for k, v in categories.items() if v > 0}
    if not categories:
        return False

    fig, ax = plt.subplots(1, 1, figsize=(8 * cfg.figsize_scale, 6 * cfg.figsize_scale))

    colors_map = {
        "CHAIN_BREAK (θ>1.0)": "#F44336",
        "MPNN_OVERFIT (gap>0.01)": "#FF9800",
        "BOUNDARY_EFFECT": "#FFC107",
        "VQE_DIVERGENCE": "#9C27B0",
        "OTHER": "#607D8B",
    }
    labels = list(categories.keys())
    sizes = list(categories.values())
    colors = [colors_map.get(label, "#607D8B") for label in labels]

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.0f%%",
        shadow=False,
        startangle=90,
        textprops={"fontsize": 10},
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")

    if cfg.show_title:
        ax.set_title(
            f"Failure Mode Distribution ({len(fails)} failed runs)\n"
            "69% detectable pre-run or during Phase 2",
            fontsize=13,
        )

    if cfg.tight_layout:
        plt.tight_layout()
    path = cfg.output_dir / f"fig_failure_mode_breakdown.{cfg.format}"
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {path}", file=sys.stderr)
    return True


@registry.register(
    name="convergence_rate_dist",
    description="Histogram: VQE convergence rate distribution with topology breakdown",
    category="comparison",
    source="analysis",
)
def fig_convergence_rate_dist(data: list[dict], cfg: FigureConfig) -> bool:
    """Stacked histogram of convergence rates."""
    plt = _get_plt(cfg.theme)
    if plt is None:
        return False

    valid = [d for d in data if d.get("convergence_rate") is not None]
    if not valid:
        return False

    fig, ax = plt.subplots(1, 1, figsize=(8 * cfg.figsize_scale, 5 * cfg.figsize_scale))

    import numpy as np

    bins = np.linspace(0, 1, 21)

    for topo in ["chain_1d", "ladder", "triangular"]:
        subset = [d["convergence_rate"] for d in valid if d.get("topology") == topo]
        if subset:
            ax.hist(
                subset,
                bins=bins,
                alpha=0.6,
                color=TOPOLOGY_COLORS.get(topo, "#607D8B"),
                label=f"{topo} (n={len(subset)})",
                edgecolor="white",
            )

    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.7, label="Divergence threshold")
    ax.set_xlabel("Convergence Rate", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    if cfg.show_title:
        ax.set_title("VQE Convergence Rate Distribution", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    if cfg.tight_layout:
        plt.tight_layout()
    path = cfg.output_dir / f"fig_convergence_rate_dist.{cfg.format}"
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {path}", file=sys.stderr)
    return True


@registry.register(
    name="seed_stability",
    description="Scatter: per-seed variability of ΔE/gap (reproducibility check)",
    category="comparison",
    source="analysis",
)
def fig_seed_stability(data: list[dict], cfg: FigureConfig) -> bool:
    """Plot seed-to-seed variability for the same configuration."""
    plt = _get_plt(cfg.theme)
    if plt is None:
        return False

    # Group by (topology, n_qubits, n_restarts, hidden_dim) to find seed runs
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for d in data:
        if d.get("seed") is not None and d.get("delta_e_over_gap") is not None:
            key = (
                d.get("topology", ""),
                d.get("n_qubits", 0),
                d.get("n_restarts", 0),
                d.get("hidden_dim", 0),
            )
            groups[key].append(d)

    # Only keep groups with 2+ seeds
    multi_seed = {k: v for k, v in groups.items() if len(v) >= 2}
    if len(multi_seed) < 3:
        return False

    fig, ax = plt.subplots(1, 1, figsize=(9 * cfg.figsize_scale, 6 * cfg.figsize_scale))

    x_ticks = []
    x_labels = []

    for x_pos, (key, runs) in enumerate(sorted(multi_seed.items(), key=lambda x: x[0])):
        topo, n, restarts, hdim = key
        values = sorted(d["delta_e_over_gap"] for d in runs)
        color = TOPOLOGY_COLORS.get(topo, "#607D8B")

        for v in values:
            ax.scatter(x_pos, v, c=color, alpha=0.7, s=30, edgecolors="white", linewidth=0.3)
        # Connect with line
        ax.plot([x_pos] * len(values), values, color=color, alpha=0.4, linewidth=1)
        # Range bar
        ax.plot([x_pos, x_pos], [min(values), max(values)], color=color, linewidth=2, alpha=0.6)

        x_ticks.append(x_pos)
        x_labels.append(f"{topo[:3]}N{n}")

    ax.axhline(y=0.05, color="green", linestyle="--", alpha=0.7, label="5% PASS")
    ax.set_yscale("log")
    ax.set_xticks(x_ticks[:: max(1, len(x_ticks) // 15)])
    ax.set_xticklabels(x_labels[:: max(1, len(x_labels) // 15)], rotation=45, fontsize=8)
    ax.set_xlabel("Configuration", fontsize=11)
    ax.set_ylabel("ΔE/gap (per seed)", fontsize=12)
    if cfg.show_title:
        ax.set_title(
            f"Seed-to-Seed Variability ({len(multi_seed)} configs, {sum(len(v) for v in multi_seed.values())} runs)",
            fontsize=13,
        )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    if cfg.tight_layout:
        plt.tight_layout()
    path = cfg.output_dir / f"fig_seed_stability.{cfg.format}"
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {path}", file=sys.stderr)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Category: VALIDATION — data integrity and sanity checks
# ═══════════════════════════════════════════════════════════════════════════════


@registry.register(
    name="data_completeness",
    description="Heatmap: data completeness by topology × n_qubits × p_layers",
    category="validation",
    source="analysis",
)
def fig_data_completeness(data: list[dict], cfg: FigureConfig) -> bool:
    """Heatmap showing data coverage."""
    plt = _get_plt(cfg.theme)
    if plt is None:
        return False
    try:
        import numpy as np
    except ImportError:
        return False

    if not data:
        return False

    # Build count matrix: topology × n_qubits
    topos = sorted(set(d.get("topology", "") for d in data if d.get("topology")))
    n_qubits_vals = sorted(set(d.get("n_qubits", 0) for d in data if d.get("n_qubits")))

    if not topos or not n_qubits_vals:
        return False

    matrix = np.zeros((len(topos), len(n_qubits_vals)))
    for d in data:
        t = d.get("topology", "")
        n = d.get("n_qubits", 0)
        if t in topos and n in n_qubits_vals:
            matrix[topos.index(t)][n_qubits_vals.index(n)] += 1

    fig, ax = plt.subplots(
        1, 1, figsize=(max(8, len(n_qubits_vals) * 1.5) * cfg.figsize_scale, 5 * cfg.figsize_scale)
    )

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(n_qubits_vals)))
    ax.set_xticklabels([str(n) for n in n_qubits_vals])
    ax.set_yticks(range(len(topos)))
    ax.set_yticklabels(topos)
    ax.set_xlabel("N (qubits)")
    ax.set_ylabel("Topology")

    # Annotate cells
    for i in range(len(topos)):
        for j in range(len(n_qubits_vals)):
            count = int(matrix[i, j])
            if count > 0:
                ax.text(j, i, str(count), ha="center", va="center", fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Run count")
    if cfg.show_title:
        ax.set_title(f"Data Coverage Matrix ({len(data)} total runs)", fontsize=13)

    if cfg.tight_layout:
        plt.tight_layout()
    path = cfg.output_dir / f"fig_data_completeness.{cfg.format}"
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {path}", file=sys.stderr)
    return True


@registry.register(
    name="outlier_detection",
    description="Scatter: highlight outlier runs (ΔE/gap > 3σ from median)",
    category="validation",
    source="analysis",
)
def fig_outlier_detection(data: list[dict], cfg: FigureConfig) -> bool:
    """Identify and visualize statistical outliers."""
    plt = _get_plt(cfg.theme)
    if plt is None:
        return False
    try:
        import numpy as np
    except ImportError:
        return False

    valid = [d for d in data if d.get("delta_e_over_gap") is not None]
    if len(valid) < 10:
        return False

    de_values = np.array([d["delta_e_over_gap"] for d in valid])
    log_de = np.log10(de_values + 1e-10)
    median = np.median(log_de)
    mad = np.median(np.abs(log_de - median))
    # Modified Z-score
    modified_z = 0.6745 * (log_de - median) / max(mad, 1e-10)

    fig, ax = plt.subplots(1, 1, figsize=(9 * cfg.figsize_scale, 6 * cfg.figsize_scale))

    normal_mask = np.abs(modified_z) < 3.5
    outlier_mask = ~normal_mask

    # Normal points
    for topo in TOPOLOGY_COLORS:
        indices = [i for i, d in enumerate(valid) if d.get("topology") == topo and normal_mask[i]]
        if indices:
            ax.scatter(
                indices,
                de_values[indices],
                c=TOPOLOGY_COLORS[topo],
                alpha=0.5,
                s=20,
                label=topo,
            )

    # Outliers (highlighted)
    outlier_indices = np.where(outlier_mask)[0]
    if len(outlier_indices) > 0:
        ax.scatter(
            outlier_indices,
            de_values[outlier_indices],
            c="red",
            marker="x",
            s=80,
            linewidths=2,
            label=f"Outliers (n={len(outlier_indices)})",
            zorder=5,
        )

    ax.axhline(y=0.05, color="green", linestyle="--", alpha=0.7, label="5% PASS")
    ax.set_yscale("log")
    ax.set_xlabel("Run Index", fontsize=11)
    ax.set_ylabel("ΔE/gap", fontsize=12)
    if cfg.show_title:
        ax.set_title(
            f"Outlier Detection (Modified Z-score > 3.5)\n"
            f"{len(outlier_indices)}/{len(valid)} runs flagged",
            fontsize=13,
        )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    if cfg.tight_layout:
        plt.tight_layout()
    path = cfg.output_dir / f"fig_outlier_detection.{cfg.format}"
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {path}", file=sys.stderr)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Main execution engine
# ═══════════════════════════════════════════════════════════════════════════════


def generate_figures(
    source: str = "both",
    only: list[str] | None = None,
    exclude: list[str] | None = None,
    category: str | None = None,
    config: FigureConfig | None = None,
    diagnostics_path: Path | None = None,
    health_path: Path | None = None,
    stdin_data: dict | None = None,
) -> dict[str, bool]:
    """Generate figures based on source and filters.

    Parameters
    ----------
    source : str
        Data source: "analysis", "health", "both", "stdin"
    only : list[str] | None
        If set, only generate these figure names.
    exclude : list[str] | None
        If set, skip these figure names.
    category : str | None
        If set, only generate figures in this category.
    config : FigureConfig | None
        Configuration for figure output. Uses defaults if None.
    diagnostics_path : Path | None
        Custom path to diagnostics JSON.
    health_path : Path | None
        Custom path to health report JSON.
    stdin_data : dict | None
        Pre-loaded data from stdin.

    Returns
    -------
    dict[str, bool]
        Map of figure_name → success.
    """
    if config is None:
        config = FigureConfig()

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data based on source
    diag_data: list[dict] = []
    health_data: dict[str, Any] = {}

    if source in ("analysis", "both"):
        diag_data = load_diagnostics_data(diagnostics_path)
    if source in ("health", "both"):
        health_data = load_health_report(health_path)
    if source == "stdin" and stdin_data:
        # Detect data type: list → diagnostics, dict with "experiments" → health
        if isinstance(stdin_data, list):
            diag_data = stdin_data
        elif isinstance(stdin_data, dict):
            health_data = stdin_data

    # Determine which figures to generate
    all_specs = registry.list_figures(category=category)

    if only:
        all_specs = [s for s in all_specs if s.name in only]
    if exclude:
        all_specs = [s for s in all_specs if s.name not in exclude]

    # Filter by available data
    results: dict[str, bool] = {}
    for spec in all_specs:
        # Check if we have the right data
        if spec.source == "analysis" and not diag_data:
            results[spec.name] = False
            continue
        if spec.source == "health" and not health_data:
            results[spec.name] = False
            continue

        # Execute
        try:
            if spec.source == "analysis":
                success = spec.func(diag_data, config)
            elif spec.source == "health":
                success = spec.func(health_data, config)
            else:  # "both" - try analysis data first
                if diag_data:
                    success = spec.func(diag_data, config)
                elif health_data:
                    success = spec.func(health_data, config)
                else:
                    success = False
            results[spec.name] = success
        except Exception as e:
            print(f"  ✗ {spec.name}: {e}", file=sys.stderr)
            results[spec.name] = False

    return results


def list_available_figures() -> str:
    """Return a formatted list of all available figures."""
    lines: list[str] = []
    lines.append("Available figures:")
    lines.append("")

    current_cat = ""
    for spec in registry.list_figures():
        if spec.category != current_cat:
            current_cat = spec.category
            lines.append(f"  [{current_cat.upper()}]")
        lines.append(f"    {spec.name:<25} {spec.description}")
        lines.append(f"    {'':25} source: {spec.source}")
    lines.append("")
    lines.append(f"Total: {len(registry.names())} figures")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args():
    """Parse CLI arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate analysis and health figures for the GNN-HVA project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    python -m project_health.figures                         # Generate all from both sources
    python -m project_health.figures --source analysis       # Only from diagnostics data
    python -m project_health.figures --source health         # Only from health report
    python -m project_health.figures --only gen_gap,smoothness_hist
    python -m project_health.figures --category diagnostics
    python -m project_health.figures --exclude outlier_detection
    python -m project_health.figures --format svg --dpi 300
    python -m project_health.figures --theme thesis
    python -m project_health.figures --list
    python -m project_health.figures --output-dir ./custom_figs
    cat health_report.json | python -m project_health.figures --source stdin
""",
    )

    parser.add_argument(
        "--source",
        choices=["analysis", "health", "both", "stdin"],
        default="both",
        help="Data source for figure generation (default: both)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of figure names to generate",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Comma-separated list of figure names to skip",
    )
    parser.add_argument(
        "--category",
        choices=["diagnostics", "health", "comparison", "validation"],
        default=None,
        help="Only generate figures in this category",
    )
    parser.add_argument(
        "--format",
        choices=["png", "svg", "pdf"],
        default="png",
        help="Output image format (default: png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI (default: 150, thesis: 300)",
    )
    parser.add_argument(
        "--theme",
        choices=["default", "dark", "minimal", "thesis"],
        default="default",
        help="Visual theme (default: default)",
    )
    parser.add_argument(
        "--figsize-scale",
        type=float,
        default=1.0,
        help="Scale factor for figure dimensions (default: 1.0)",
    )
    parser.add_argument(
        "--no-titles",
        action="store_true",
        help="Suppress figure titles (useful for thesis embedding)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (default: project_health/figures/)",
    )
    parser.add_argument(
        "--diagnostics-file",
        type=str,
        default=None,
        help="Custom path to diagnostics JSON file",
    )
    parser.add_argument(
        "--health-file",
        type=str,
        default=None,
        help="Custom path to health report JSON file",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available figures and exit",
    )

    return parser.parse_args()


def main() -> None:
    """CLI main entry point."""
    args = parse_args()

    if args.list:
        print(list_available_figures())
        return

    # Build config
    config = FigureConfig(
        format=args.format,
        dpi=args.dpi,
        theme=args.theme,
        figsize_scale=args.figsize_scale,
        show_title=not args.no_titles,
    )

    if args.output_dir:
        config.output_dir = Path(args.output_dir)

    # Parse filter lists
    only = args.only.split(",") if args.only else None
    exclude = args.exclude.split(",") if args.exclude else None

    # Handle stdin
    stdin_data = None
    if args.source == "stdin":
        stdin_data = load_from_stdin()
        if not stdin_data:
            print("Error: --source stdin but no data on stdin", file=sys.stderr)
            sys.exit(1)

    # Custom file paths
    diag_path = Path(args.diagnostics_file) if args.diagnostics_file else None
    health_path = Path(args.health_file) if args.health_file else None

    print(
        f"Generating figures (source={args.source}, format={args.format}, "
        f"theme={args.theme}, dpi={args.dpi})...",
        file=sys.stderr,
    )

    results = generate_figures(
        source=args.source,
        only=only,
        exclude=exclude,
        category=args.category,
        config=config,
        diagnostics_path=diag_path,
        health_path=health_path,
        stdin_data=stdin_data,
    )

    # Summary
    n_success = sum(1 for v in results.values() if v)
    n_total = len(results)
    n_skipped = sum(1 for v in results.values() if not v)

    print(f"\n✅ Generated {n_success}/{n_total} figures in {config.output_dir}", file=sys.stderr)
    if n_skipped:
        skipped = [k for k, v in results.items() if not v]
        print(f"   Skipped: {', '.join(skipped)}", file=sys.stderr)


if __name__ == "__main__":
    main()
