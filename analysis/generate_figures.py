#!/usr/bin/env python3
"""Generate thesis-quality figures from diagnostics data.

Produces:
  - fig_01_gen_gap_vs_de_gap.png: Scatter plot with threshold lines
  - fig_02_smoothness_histogram.png: Distribution by topology
  - fig_03_restarts_vs_smoothness.png: Restart paradox mechanism
  - fig_04_cross_topology_bar.png: Pass rate comparison

Usage:
    python analysis/generate_figures.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIAG_PATH = ROOT / "analysis" / "raw_data" / "all_diagnostics.json"
FIG_DIR = ROOT / "analysis" / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_data():
    with open(DIAG_PATH) as f:
        return json.load(f)


def fig_01_gen_gap_vs_de_gap(data):
    """Scatter: generalization_gap vs ΔE/gap with threshold lines."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figure generation")
        return False

    valid = [
        d for d in data if d["generalization_gap"] is not None and d["delta_e_over_gap"] is not None
    ]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Color by topology
    colors = {"chain_1d": "#2196F3", "ladder": "#4CAF50", "triangular": "#FF5722"}
    markers = {"chain_1d": "o", "ladder": "s", "triangular": "^"}

    for topo in ["chain_1d", "ladder", "triangular"]:
        subset = [d for d in valid if d["topology"] == topo]
        x = [d["generalization_gap"] for d in subset]
        y = [d["delta_e_over_gap"] for d in subset]
        ax.scatter(
            x,
            y,
            c=colors[topo],
            marker=markers[topo],
            alpha=0.7,
            s=40,
            label=topo,
            edgecolors="white",
            linewidth=0.5,
        )

    # Threshold lines
    ax.axhline(y=0.05, color="green", linestyle="--", alpha=0.7, label="ΔE/gap = 5% (PASS)")
    ax.axhline(y=0.10, color="orange", linestyle="--", alpha=0.7, label="ΔE/gap = 10% (FAIL)")
    ax.axvline(x=1e-2, color="red", linestyle=":", alpha=0.7, label="gen_gap = 1e-2 (abort)")
    ax.axvline(x=1e-3, color="orange", linestyle=":", alpha=0.5, label="gen_gap = 1e-3 (warn)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Generalization Gap (Phase 3)", fontsize=12)
    ax.set_ylabel("ΔE/gap (Phase 4 outcome)", fontsize=12)
    ax.set_title(
        "MPNN Generalization Gap vs Pipeline Outcome\n(gen_gap > 1e-2 → 85% failure rate)",
        fontsize=13,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(1e-6, 1)
    ax.set_ylim(1e-4, 20)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIG_DIR / "fig_01_gen_gap_vs_de_gap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")
    return True


def fig_02_smoothness_histogram(data):
    """Histogram of theta_smoothness by topology."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    valid = [d for d in data if d["theta_smoothness"] is not None]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    configs = [
        ("chain_1d", 6),
        ("ladder", 6),
        ("triangular", 6),
        ("ladder", 10),
        ("triangular", 10),
        (None, None),
    ]

    for ax, (topo, n) in zip(axes.flat, configs, strict=False):
        if topo is None:
            ax.axis("off")
            continue

        subset = [
            d["theta_smoothness"] for d in valid if d["topology"] == topo and d["n_qubits"] == n
        ]
        if not subset:
            ax.set_title(f"{topo} N={n}\n(no data)")
            continue

        # Clip for visualization
        clipped = [min(s, 5.0) for s in subset]
        n_breaks = sum(1 for s in subset if s >= 1.0)

        ax.hist(clipped, bins=20, color="#2196F3", alpha=0.7, edgecolor="white")
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

    plt.suptitle(
        "Distribution of θ_smoothness by Topology\n(>1.0 = warm-start chain break)",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    path = FIG_DIR / "fig_02_smoothness_histogram.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")
    return True


def fig_03_cross_topology_bar(data):
    """Bar chart: pass rate by topology."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    valid = [d for d in data if d["delta_e_over_gap"] is not None]

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
            d["delta_e_over_gap"] for d in valid if d["topology"] == topo and d["n_qubits"] == n
        ]
        if subset:
            labels.append(f"{topo}\nN={n}")
            pass_rates.append(sum(1 for v in subset if v < 0.05) / len(subset))
            medians.append(sorted(subset)[len(subset) // 2])
            counts.append(len(subset))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pass rate
    colors = ["#2196F3", "#4CAF50", "#4CAF50", "#FF5722", "#FF5722"]
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

    # Median ΔE/gap
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

    plt.suptitle("GNN-HVA Framework: Cross-Topology Performance", fontsize=14, y=1.02)
    plt.tight_layout()
    path = FIG_DIR / "fig_03_cross_topology_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")
    return True


def fig_04_smoothness_vs_de_gap(data):
    """Scatter: theta_smoothness vs ΔE/gap showing the threshold effect."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    valid = [
        d for d in data if d["theta_smoothness"] is not None and d["delta_e_over_gap"] is not None
    ]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = {"chain_1d": "#2196F3", "ladder": "#4CAF50", "triangular": "#FF5722"}

    for topo in ["chain_1d", "ladder", "triangular"]:
        subset = [d for d in valid if d["topology"] == topo]
        x = [d["theta_smoothness"] for d in subset]
        y = [d["delta_e_over_gap"] for d in subset]
        ax.scatter(
            x, y, c=colors[topo], alpha=0.6, s=40, label=topo, edgecolors="white", linewidth=0.5
        )

    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=2, label="Chain break threshold")
    ax.axhline(y=0.05, color="green", linestyle="--", alpha=0.7, label="5% PASS")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("θ_smoothness (Phase 2)", fontsize=12)
    ax.set_ylabel("ΔE/gap (Phase 4 outcome)", fontsize=12)
    ax.set_title(
        "Warm-Start Chain Smoothness vs Pipeline Outcome\n"
        "(θ > 1.0 = chain break → 76% failure/marginal)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIG_DIR / "fig_04_smoothness_vs_de_gap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")
    return True


def main():
    import sys

    print("Generating thesis figures...", file=sys.stderr)

    data = load_data()
    print(f"  Loaded {len(data)} diagnostic records", file=sys.stderr)

    success = 0
    success += fig_01_gen_gap_vs_de_gap(data)
    success += fig_02_smoothness_histogram(data)
    success += fig_03_cross_topology_bar(data)
    success += fig_04_smoothness_vs_de_gap(data)

    print(f"\n✅ Generated {success}/4 figures in {FIG_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
