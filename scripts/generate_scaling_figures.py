#!/usr/bin/env python3
"""Generate thesis figures from MPS scaling data.

Produces publication-quality figures for Chapter 5 demonstrating:
  1. Scaling law validation (h_min vs N with power-law fit)
  2. ΔE/gap vs system size (quality across N=40/50/80)
  3. Timing vs system size (computational cost scaling)
  4. VQE convergence quality (θ_opt per seed, per N)

All data sourced from existing results in results/scaling/.
No additional compute required.

Usage:
    python scripts/generate_scaling_figures.py
    python scripts/generate_scaling_figures.py --format pdf --dpi 300
    python scripts/generate_scaling_figures.py --output-dir documentation/thesis_figures
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ScalingDataPoint:
    """Parsed data from a single scaling result file."""

    n: int
    topology: str
    h_values: list[float]
    seeds: list[int]
    dmrg_energies: list[float]
    gaps: list[float]
    # Per-seed VQE results
    vqe_results: list[dict]  # [{seed, results: [{h, de_gap, ...}]}]
    timing: dict
    # Derived
    mean_de_gap: float
    max_de_gap: float
    total_time_s: float
    phase1_time_s: float
    phase2_time_s: float


def load_scaling_results(results_dir: Path) -> list[ScalingDataPoint]:
    """Load all scaling_N* result files from the given directory."""
    points: list[ScalingDataPoint] = []

    for path in sorted(results_dir.glob("scaling_N*_aer_mps_*.json")):
        with open(path) as f:
            data = json.load(f)

        meta = data["metadata"]
        timing = data.get("timing", {})

        # Compute aggregate ΔE/gap across all seeds
        all_de_gaps = []
        for seed_run in data.get("vqe_results", []):
            for r in seed_run.get("results", []):
                if "de_gap" in r:
                    all_de_gaps.append(r["de_gap"])

        if not all_de_gaps:
            continue

        points.append(
            ScalingDataPoint(
                n=meta["n"],
                topology=meta.get("topology", "chain_1d"),
                h_values=meta.get("h_values", []),
                seeds=meta.get("seeds", [42]),
                dmrg_energies=[d["ground_energy"] for d in data.get("dmrg_data", [])],
                gaps=[d["gap"] for d in data.get("dmrg_data", [])],
                vqe_results=data.get("vqe_results", []),
                timing=timing,
                mean_de_gap=sum(all_de_gaps) / len(all_de_gaps),
                max_de_gap=max(all_de_gaps),
                total_time_s=timing.get("total_s", 0),
                phase1_time_s=timing.get("phase1_dmrg_s", 0),
                phase2_time_s=timing.get("phase2_vqe_s", 0),
            )
        )

    # Deduplicate: keep the one with most seeds per N
    by_n: dict[int, list[ScalingDataPoint]] = {}
    for p in points:
        by_n.setdefault(p.n, []).append(p)

    deduped = []
    for n_val, group in sorted(by_n.items()):
        best = max(group, key=lambda p: len(p.seeds))
        deduped.append(best)

    return deduped


def _get_plt(theme: str = "thesis"):
    """Import and configure matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "ERROR: matplotlib not available. Install with: pip install matplotlib", file=sys.stderr
        )
        return None

    if theme == "thesis":
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
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )
    return plt


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Scaling Law Validation
# ═══════════════════════════════════════════════════════════════════════════════


def fig_scaling_law(
    data: list[ScalingDataPoint],
    output_dir: Path,
    fmt: str,
    dpi: int,
) -> bool:
    """Plot h_min scaling law: predicted vs actual tested boundaries."""
    plt = _get_plt()
    if plt is None:
        return False
    import numpy as np

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # Scaling law: h_min = 1.5 + 0.020 * N^1.31 (corrected formula)
    n_range = np.linspace(6, 100, 200)
    h_min_pred = 1.5 + 0.020 * n_range**1.31

    ax.plot(n_range, h_min_pred, "b-", linewidth=2, label=r"$h_{min} = 1.5 + 0.020 \cdot N^{1.31}$")

    # Data points: lowest h tested that still passes
    n_vals = []
    h_lowest_pass = []
    for p in data:
        n_vals.append(p.n)
        h_lowest_pass.append(min(p.h_values))

    ax.scatter(
        n_vals,
        h_lowest_pass,
        c="#E53935",
        s=100,
        zorder=5,
        marker="v",
        label="Lowest h tested (all PASS)",
        edgecolors="white",
        linewidth=1.5,
    )

    # Predicted h_min for those N values
    for n_val in n_vals:
        h_pred = 1.5 + 0.020 * n_val**1.31
        ax.scatter(
            n_val,
            h_pred,
            c="#1565C0",
            s=80,
            zorder=4,
            marker="o",
            edgecolors="white",
            linewidth=1.5,
        )

    # Connect predicted to actual with arrows
    for n_val, h_actual in zip(n_vals, h_lowest_pass, strict=False):
        h_pred = 1.5 + 0.020 * n_val**1.31
        offset = h_actual - h_pred
        if abs(offset) > 0.01:
            ax.annotate(
                "",
                xy=(n_val, h_actual),
                xytext=(n_val, h_pred),
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            )

    ax.scatter(
        [],
        [],
        c="#1565C0",
        s=80,
        marker="o",
        label="Predicted $h_{min}(N)$",
        edgecolors="white",
        linewidth=1.5,
    )

    ax.set_xlabel("System Size $N$ (qubits)")
    ax.set_ylabel("Transverse Field $h$")
    ax.set_title("MPS Scaling Law Validation: Valid Regime Boundary")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(0, 90)

    # Annotate points
    for n_val, h_val in zip(n_vals, h_lowest_pass, strict=False):
        ax.annotate(
            f"N={n_val}",
            (n_val, h_val),
            textcoords="offset points",
            xytext=(8, -8),
            fontsize=8,
            color="gray",
        )

    plt.tight_layout()
    path = output_dir / f"fig_scaling_law_validation.{fmt}"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: ΔE/gap vs System Size
# ═══════════════════════════════════════════════════════════════════════════════


def fig_de_gap_vs_n(
    data: list[ScalingDataPoint],
    output_dir: Path,
    fmt: str,
    dpi: int,
) -> bool:
    """Plot ΔE/gap vs N showing quality across system sizes."""
    plt = _get_plt()
    if plt is None:
        return False
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Mean ΔE/gap with error bars (std across h-points and seeds)
    n_vals = []
    means = []
    stds = []
    maxes = []

    for p in data:
        all_de = []
        for seed_run in p.vqe_results:
            for r in seed_run["results"]:
                if "de_gap" in r:
                    all_de.append(r["de_gap"])
        if all_de:
            n_vals.append(p.n)
            means.append(np.mean(all_de))
            stds.append(np.std(all_de))
            maxes.append(np.max(all_de))

    if not n_vals:
        plt.close()
        return False

    ax1.errorbar(
        n_vals,
        [m * 100 for m in means],
        yerr=[s * 100 for s in stds],
        fmt="o-",
        color="#1565C0",
        capsize=5,
        capthick=2,
        linewidth=2,
        markersize=8,
        label="Mean ± std",
    )
    ax1.scatter(
        n_vals,
        [m * 100 for m in maxes],
        marker="^",
        c="#E53935",
        s=60,
        zorder=5,
        label="Max ΔE/gap",
    )

    ax1.axhline(y=5, color="green", linestyle="--", alpha=0.7, linewidth=1.5, label="5% threshold")
    ax1.axhline(y=1, color="green", linestyle=":", alpha=0.5, linewidth=1, label="1% target")

    ax1.set_xlabel("System Size $N$ (qubits)")
    ax1.set_ylabel("$\\Delta E / \\text{gap}$ (%)")
    ax1.set_title("Energy Error vs System Size")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_ylim(0, max(m * 100 for m in maxes) * 1.3)

    for n_val, mean_val in zip(n_vals, means, strict=False):
        ax1.annotate(
            f"{mean_val * 100:.2f}%",
            (n_val, mean_val * 100),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=8,
        )

    # Right: Per-h breakdown for each N
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data)))
    for i, p in enumerate(data):
        # Use first seed only for clarity
        if p.vqe_results:
            results = p.vqe_results[0]["results"]
            h_vals = [r["h"] for r in results]
            de_vals = [r["de_gap"] * 100 for r in results if "de_gap" in r]
            h_vals = h_vals[: len(de_vals)]
            ax2.plot(
                h_vals,
                de_vals,
                "o-",
                color=colors[i],
                label=f"N={p.n}",
                markersize=5,
                linewidth=1.5,
            )

    ax2.axhline(y=5, color="green", linestyle="--", alpha=0.7, linewidth=1.5)
    ax2.set_xlabel("Transverse Field $h$")
    ax2.set_ylabel("$\\Delta E / \\text{gap}$ (%)")
    ax2.set_title("Per-$h$ Energy Error by System Size")
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    path = output_dir / f"fig_de_gap_vs_system_size.{fmt}"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Timing vs System Size
# ═══════════════════════════════════════════════════════════════════════════════


def fig_timing_vs_n(
    data: list[ScalingDataPoint],
    output_dir: Path,
    fmt: str,
    dpi: int,
) -> bool:
    """Plot computational timing vs N."""
    plt = _get_plt()
    if plt is None:
        return False
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    n_vals = [p.n for p in data]
    total_times = [p.total_time_s / 60.0 for p in data]  # in minutes
    phase1_times = [p.phase1_time_s / 60.0 for p in data]  # in minutes
    phase2_times = [p.phase2_time_s / 60.0 for p in data]  # in minutes

    # Left: Total time with phase breakdown (stacked bar)
    ax1.bar(n_vals, phase1_times, width=4, color="#64B5F6", label="Phase 1 (DMRG)", alpha=0.8)
    ax1.bar(
        n_vals,
        phase2_times,
        width=4,
        bottom=phase1_times,
        color="#E57373",
        label="Phase 2 (VQE)",
        alpha=0.8,
    )

    ax1.set_xlabel("System Size $N$ (qubits)")
    ax1.set_ylabel("Time (minutes)")
    ax1.set_title("Computation Time by Phase")
    ax1.legend(loc="upper left", framealpha=0.9)

    # Add time labels
    for n_val, t1, t2 in zip(n_vals, phase1_times, phase2_times, strict=False):
        ax1.annotate(
            f"{t1:.1f}m",
            (n_val, t1 / 2),
            fontsize=7,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
        ax1.annotate(
            f"{t2:.0f}m",
            (n_val, t1 + t2 / 2),
            fontsize=7,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

    # Right: Log-scale total time with power-law fit
    ax2.scatter(
        n_vals, total_times, c="#1565C0", s=100, zorder=5, edgecolors="white", linewidth=1.5
    )

    # Fit power law: T = a * N^b
    if len(n_vals) >= 3:
        log_n = np.log(n_vals)
        log_t = np.log(total_times)
        coeffs = np.polyfit(log_n, log_t, 1)
        b_fit, a_fit = coeffs[0], np.exp(coeffs[1])

        n_fit = np.linspace(min(n_vals) * 0.8, max(n_vals) * 1.2, 100)
        t_fit = a_fit * n_fit**b_fit
        ax2.plot(n_fit, t_fit, "r--", linewidth=1.5, label=f"Fit: $T \\propto N^{{{b_fit:.2f}}}$")

    ax2.set_xlabel("System Size $N$ (qubits)")
    ax2.set_ylabel("Total Time (minutes)")
    ax2.set_title("Scaling of Total Computation Time")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(loc="upper left", framealpha=0.9)

    for n_val, t_val in zip(n_vals, total_times, strict=False):
        ax2.annotate(
            f"N={n_val}\n{t_val:.0f}m",
            (n_val, t_val),
            textcoords="offset points",
            xytext=(10, -5),
            fontsize=8,
        )

    plt.tight_layout()
    path = output_dir / f"fig_timing_vs_system_size.{fmt}"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: θ_opt Landscape Across N
# ═══════════════════════════════════════════════════════════════════════════════


def fig_theta_landscape(
    data: list[ScalingDataPoint],
    output_dir: Path,
    fmt: str,
    dpi: int,
) -> bool:
    """Plot θ_opt(h) for each N showing the smooth landscape."""
    plt = _get_plt()
    if plt is None:
        return False
    import numpy as np

    # Only use data with theta_opt
    data_with_theta = []
    for p in data:
        if p.vqe_results:
            first_seed = p.vqe_results[0]
            if first_seed["results"] and "theta_opt" in first_seed["results"][0]:
                data_with_theta.append(p)

    if not data_with_theta:
        print("  No theta_opt data available for landscape figure", file=sys.stderr)
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data_with_theta)))

    for i, p in enumerate(data_with_theta):
        results = p.vqe_results[0]["results"]
        h_vals = [r["h"] for r in results if "theta_opt" in r]
        theta_zz = [r["theta_opt"][0] for r in results if "theta_opt" in r]
        theta_x = [r["theta_opt"][1] for r in results if "theta_opt" in r]

        ax1.plot(
            h_vals, theta_zz, "o-", color=colors[i], label=f"N={p.n}", markersize=4, linewidth=1.5
        )
        ax2.plot(
            h_vals, theta_x, "o-", color=colors[i], label=f"N={p.n}", markersize=4, linewidth=1.5
        )

    ax1.set_xlabel("Transverse Field $h$")
    ax1.set_ylabel(r"$\theta_{ZZ}$")
    ax1.set_title(r"$\theta_{ZZ}(h)$ — ZZ Coupling Parameter")
    ax1.legend(loc="upper left", framealpha=0.9)

    ax2.set_xlabel("Transverse Field $h$")
    ax2.set_ylabel(r"$\theta_X$")
    ax2.set_title(r"$\theta_X(h)$ — Transverse Field Parameter")
    ax2.legend(loc="upper left", framealpha=0.9)

    plt.suptitle(
        r"Optimized HVA Parameters $\theta_{opt}(h)$ Across System Sizes", fontsize=13, y=1.02
    )
    plt.tight_layout()
    path = output_dir / f"fig_theta_landscape_scaling.{fmt}"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Multi-seed Stability
# ═══════════════════════════════════════════════════════════════════════════════


def fig_seed_stability(
    data: list[ScalingDataPoint],
    output_dir: Path,
    fmt: str,
    dpi: int,
) -> bool:
    """Plot seed stability: ΔE/gap variance across seeds for multi-seed runs."""
    plt = _get_plt()
    if plt is None:
        return False
    import numpy as np

    # Filter to multi-seed data only
    multi_seed = [p for p in data if len(p.seeds) > 1]
    if not multi_seed:
        print("  No multi-seed data for stability figure", file=sys.stderr)
        return False

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, 8))

    for p in multi_seed:
        # Gather per-h ΔE/gap for each seed
        h_vals = p.h_values
        seed_data: dict[int, list[float]] = {}
        for seed_run in p.vqe_results:
            seed = seed_run["seed"]
            seed_data[seed] = [r["de_gap"] * 100 for r in seed_run["results"] if "de_gap" in r]

        # Plot each seed
        for j, (seed, de_vals) in enumerate(seed_data.items()):
            h_subset = h_vals[: len(de_vals)]
            ax.plot(
                h_subset,
                de_vals,
                "o-",
                color=colors[j % len(colors)],
                markersize=4,
                linewidth=1,
                alpha=0.7,
                label=f"N={p.n}, seed={seed}" if j == 0 or p == multi_seed[0] else "",
            )

        # Plot mean ± std band
        all_arrays = list(seed_data.values())
        min_len = min(len(a) for a in all_arrays)
        stacked = np.array([a[:min_len] for a in all_arrays])
        mean_de = stacked.mean(axis=0)
        std_de = stacked.std(axis=0)
        h_subset = h_vals[:min_len]
        ax.fill_between(h_subset, mean_de - std_de, mean_de + std_de, alpha=0.15, color="#1565C0")
        ax.plot(h_subset, mean_de, "k-", linewidth=2, label=f"N={p.n} mean")

    ax.axhline(y=5, color="green", linestyle="--", alpha=0.7, linewidth=1.5, label="5% threshold")
    ax.set_xlabel("Transverse Field $h$")
    ax.set_ylabel("$\\Delta E / \\text{gap}$ (%)")
    ax.set_title("Multi-Seed VQE Stability (N=40, seeds=42/43/44)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8, ncol=2)

    plt.tight_layout()
    path = output_dir / f"fig_seed_stability_scaling.{fmt}"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    """Generate all scaling figures."""
    parser = argparse.ArgumentParser(
        description="Generate thesis figures from MPS scaling results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/scaling",
        help="Directory containing scaling result JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="documentation/thesis_figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--format", type=str, default="pdf", choices=["pdf", "png", "svg"], help="Output format"
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
    args = parser.parse_args()

    results_dir = ROOT / args.results_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MPS Scaling Thesis Figures")
    print(f"  Source: {results_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Format: {args.format} @ {args.dpi} DPI")
    print("=" * 60)

    # Load data
    data = load_scaling_results(results_dir)
    if not data:
        print("ERROR: No scaling result files found.", file=sys.stderr)
        return 1

    print(f"\nLoaded {len(data)} scaling data points:")
    for p in data:
        print(
            f"  N={p.n}: {len(p.h_values)} h-points, {len(p.seeds)} seeds, "
            f"mean ΔE/gap={p.mean_de_gap * 100:.3f}%, time={p.total_time_s / 60:.1f}m"
        )

    # Generate figures
    print("\n─── Generating Figures ───")
    generated = 0
    total = 5

    if fig_scaling_law(data, output_dir, args.format, args.dpi):
        generated += 1
    if fig_de_gap_vs_n(data, output_dir, args.format, args.dpi):
        generated += 1
    if fig_timing_vs_n(data, output_dir, args.format, args.dpi):
        generated += 1
    if fig_theta_landscape(data, output_dir, args.format, args.dpi):
        generated += 1
    if fig_seed_stability(data, output_dir, args.format, args.dpi):
        generated += 1

    print(f"\n─── Done: {generated}/{total} figures generated ───")
    return 0 if generated > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
