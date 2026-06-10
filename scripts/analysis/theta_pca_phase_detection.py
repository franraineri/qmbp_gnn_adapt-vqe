#!/usr/bin/env python3
"""Task 2.2: PCA and clustering analysis of θ_opt(h) for unsupervised phase detection.

Hypothesis: PCA/clustering of θ_opt(h) trajectories reveals the Z₂ phase
transition at h_c≈1.0 without supervision (labels or known h_c).

Reads: analysis/raw_data/theta_trajectories.json (from Task 2.1)
Outputs:
  - project_health/figures/fig_theta_pca_phase_detection.{format}
  - analysis/raw_data/theta_pca_results.json

Usage:
    python scripts/analysis/theta_pca_phase_detection.py
    python scripts/analysis/theta_pca_phase_detection.py --format pdf --theme thesis
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

INPUT_FILE = ROOT / "analysis" / "raw_data" / "theta_trajectories.json"
OUTPUT_RESULTS = ROOT / "analysis" / "raw_data" / "theta_pca_results.json"
FIGURES_DIR = ROOT / "project_health" / "figures"

# Known critical field for TFIM
H_C = 1.0


def load_trajectories() -> list[dict]:
    """Load theta trajectories from Task 2.1 output."""
    if not INPUT_FILE.exists():
        logger.error(f"Input not found: {INPUT_FILE}")
        logger.error("Run scripts/analysis/extract_theta_trajectories.py first.")
        sys.exit(1)

    with open(INPUT_FILE) as f:
        data = json.load(f)

    return data["trajectories"]


def analyze_trajectory(traj: dict) -> dict | None:
    """Run PCA + clustering + derivative analysis on a single trajectory.

    Returns analysis dict or None if insufficient data.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    h_values = np.array(traj["h_values"])
    theta_opt = np.array(traj["theta_opt"])  # shape: (n_h, n_params)

    if len(h_values) < 5:
        return None

    # Ensure descending h (our sweep convention)
    if h_values[0] < h_values[-1]:
        h_values = h_values[::-1]
        theta_opt = theta_opt[::-1]

    # Remove duplicate h-values (causes inf in np.gradient)
    _, unique_idx = np.unique(h_values, return_index=True)
    unique_idx = np.sort(unique_idx)  # Keep original order
    if len(unique_idx) < len(h_values):
        h_values = h_values[unique_idx]
        theta_opt = theta_opt[unique_idx]
    if len(h_values) < 5:
        return None

    n_params = theta_opt.shape[1]

    # --- PCA Analysis ---
    scaler = StandardScaler()
    theta_scaled = scaler.fit_transform(theta_opt)

    pca = PCA(n_components=min(2, n_params))
    pc_scores = pca.fit_transform(theta_scaled)
    pc1 = pc_scores[:, 0]

    # --- |dPC1/dh| derivative peak ---
    dpc1_dh = np.abs(np.gradient(pc1, h_values))
    # Guard against inf/nan (can happen with very close h-values)
    dpc1_dh = np.where(np.isfinite(dpc1_dh), dpc1_dh, 0.0)
    peak_idx = np.argmax(dpc1_dh)
    pca_peak_h = float(h_values[peak_idx])

    # --- |dθ/dh| total derivative (RMS across params, normalized by √n_params) ---
    dtheta_dh = np.zeros(len(h_values))
    for i in range(n_params):
        grad_i = np.gradient(theta_opt[:, i], h_values)
        grad_i = np.where(np.isfinite(grad_i), grad_i, 0.0)
        dtheta_dh += grad_i ** 2
    dtheta_dh = np.sqrt(dtheta_dh / n_params)  # RMS per param (fair cross-N)
    theta_deriv_peak_idx = np.argmax(dtheta_dh)
    theta_deriv_peak_h = float(h_values[theta_deriv_peak_idx])

    # --- K-means clustering (k=2) ---
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(theta_scaled)

    # Find cluster boundary: h-value where label changes
    boundary_h = None
    for i in range(len(labels) - 1):
        if labels[i] != labels[i + 1]:
            boundary_h = float((h_values[i] + h_values[i + 1]) / 2)
            break

    # --- Assess quality ---
    pca_agreement = abs(pca_peak_h - H_C)
    kmeans_agreement = abs(boundary_h - H_C) if boundary_h else float("inf")
    theta_deriv_agreement = abs(theta_deriv_peak_h - H_C)

    return {
        "topology": traj["topology"],
        "n_qubits": traj["n_qubits"],
        "p_layers": traj["p_layers"],
        "seed": traj["seed"],
        "n_points": len(h_values),
        "h_values": h_values.tolist(),
        "pc1": pc1.tolist(),
        "dpc1_dh": dpc1_dh.tolist(),
        "dtheta_dh": dtheta_dh.tolist(),
        "kmeans_labels": labels.tolist(),
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "pca_peak_h": pca_peak_h,
        "theta_deriv_peak_h": theta_deriv_peak_h,
        "kmeans_boundary_h": boundary_h,
        "agreement_pca": pca_agreement,
        "agreement_kmeans": kmeans_agreement,
        "agreement_theta_deriv": theta_deriv_agreement,
        "success_pca": pca_agreement <= 0.3,
        "success_kmeans": kmeans_agreement <= 0.3 if boundary_h else False,
        "success_theta_deriv": theta_deriv_agreement <= 0.3,
    }


def generate_figure(results: list[dict], fmt: str = "png", theme: str = "default") -> bool:
    """Generate the theta PCA phase detection figure.

    3-panel figure:
      1. PC1(h) colored by phase
      2. |dPC1/dh| peak location vs h_c
      3. K-means cluster boundary vs h_c
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not available")
        return False

    # Apply theme
    if theme == "thesis":
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 11,
                "axes.labelsize": 12,
                "axes.titlesize": 13,
                "legend.fontsize": 9,
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "axes.grid": True,
                "grid.alpha": 0.3,
            }
        )

    # Select best trajectory per topology (most h-points)
    best_per_topo: dict[str, dict] = {}
    for r in results:
        topo = r["topology"]
        if topo not in best_per_topo or r["n_points"] > best_per_topo[topo]["n_points"]:
            best_per_topo[topo] = r

    if not best_per_topo:
        logger.error("No valid results to plot")
        return False

    n_topos = len(best_per_topo)
    fig, axes = plt.subplots(3, n_topos, figsize=(5 * n_topos, 10), squeeze=False)

    colors = {"chain_1d": "#2196F3", "ladder": "#4CAF50", "triangular": "#FF9800"}

    for col, (topo, r) in enumerate(sorted(best_per_topo.items())):
        h = np.array(r["h_values"])
        pc1 = np.array(r["pc1"])
        dpc1 = np.array(r["dpc1_dh"])
        labels = np.array(r["kmeans_labels"])
        color = colors.get(topo, "#9C27B0")

        # Panel 1: PC1(h) colored by k-means label
        ax1 = axes[0, col]
        for label_val in [0, 1]:
            mask = labels == label_val
            marker = "o" if label_val == 0 else "s"
            ax1.scatter(
                h[mask],
                pc1[mask],
                c=color,
                marker=marker,
                alpha=0.8,
                s=40,
                label=f"Cluster {label_val}",
            )
        ax1.axvline(H_C, color="red", linestyle="--", alpha=0.7, label=f"$h_c$={H_C}")
        ax1.set_xlabel("$h$ (transverse field)")
        ax1.set_ylabel("PC1 score")
        ax1.set_title(f"{topo} (N={r['n_qubits']}, p={r['p_layers']})")
        ax1.legend(fontsize=8)

        # Panel 2: |dPC1/dh| with peak
        ax2 = axes[1, col]
        ax2.plot(h, dpc1, "-o", color=color, markersize=4, linewidth=1.5)
        ax2.axvline(H_C, color="red", linestyle="--", alpha=0.7)
        ax2.axvline(
            r["pca_peak_h"],
            color="purple",
            linestyle=":",
            alpha=0.8,
            label=f"Peak: h={r['pca_peak_h']:.2f}",
        )
        ax2.set_xlabel("$h$")
        ax2.set_ylabel("|dPC1/dh|")
        ax2.set_title(f"|dPC1/dh| (Δ from $h_c$: {r['agreement_pca']:.2f})")
        ax2.legend(fontsize=8)

        # Panel 3: |dθ/dh| total
        ax3 = axes[2, col]
        dtheta = np.array(r["dtheta_dh"])
        ax3.plot(h, dtheta, "-^", color=color, markersize=4, linewidth=1.5)
        ax3.axvline(H_C, color="red", linestyle="--", alpha=0.7)
        ax3.axvline(
            r["theta_deriv_peak_h"],
            color="orange",
            linestyle=":",
            alpha=0.8,
            label=f"Peak: h={r['theta_deriv_peak_h']:.2f}",
        )
        if r["kmeans_boundary_h"]:
            ax3.axvline(
                r["kmeans_boundary_h"],
                color="green",
                linestyle="-.",
                alpha=0.7,
                label=f"K-means: h={r['kmeans_boundary_h']:.2f}",
            )
        ax3.set_xlabel("$h$")
        ax3.set_ylabel("$|d\\theta/dh|$ (L2)")
        ax3.set_title(f"|dθ/dh| peak (Δ: {r['agreement_theta_deriv']:.2f})")
        ax3.legend(fontsize=8)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    outpath = FIGURES_DIR / f"fig_theta_pca_phase_detection.{fmt}"
    plt.savefig(outpath, dpi=300 if fmt == "pdf" else 150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Figure saved: {outpath.relative_to(ROOT)}")
    return True


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="PCA-based unsupervised phase detection from θ_opt(h)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output figure format (default: png)",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="default",
        choices=["default", "thesis"],
        help="Figure theme (default: default)",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=5,
        help="Minimum h-points for a trajectory to be analyzed (default: 5)",
    )
    parser.add_argument(
        "--scaling-analysis",
        action="store_true",
        help="Run PCA peak vs N analysis (requires scaling data in theta_trajectories.json).",
    )
    return parser.parse_args()


def run_scaling_analysis(
    results: list[dict], fmt: str = "png", theme: str = "default"
) -> dict:
    """Analyze how PCA peak position and |∂θ/∂h| amplitude scale with N.

    Key questions:
    1. Does PCA peak converge to h_c=1.0 as N→∞?
    2. Does |∂θ/∂h| amplitude decrease with N (smoother landscape)?
    3. At what N does PCA become unreliable (h-range doesn't cover h_c)?

    Produces:
    - 2-panel figure: PCA peak vs N (left), |∂θ/∂h| max amplitude vs N (right)
    - JSON with per-N statistics

    Returns analysis dict with findings.
    """
    logger.info("\n" + "=" * 60)
    logger.info("  SCALING ANALYSIS: PCA Peak Position vs System Size N")
    logger.info("=" * 60)

    # Group results by N (chain_1d only for consistency)
    from collections import defaultdict

    by_n: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        if r["topology"] == "chain_1d":
            by_n[r["n_qubits"]].append(r)

    if not by_n:
        logger.warning("  No chain_1d results for scaling analysis")
        return {"error": "no_chain_1d_data"}

    # Compute statistics per N
    n_values = sorted(by_n.keys())
    stats_per_n: list[dict] = []

    logger.info(f"\n  {'N':>4} | {'PCA peak h':>10} | {'|∂θ/∂h| max':>12} | "
                f"{'h_range':>12} | {'Covers h_c?':>11} | {'Seeds':>5}")
    logger.info(f"  {'─' * 65}")

    for n in n_values:
        n_results = by_n[n]
        pca_peaks = [r["pca_peak_h"] for r in n_results]
        # For derivative amplitude, use only trajectories with ≥8 points (reliable gradient)
        reliable = [r for r in n_results if r["n_points"] >= 8]
        if not reliable:
            reliable = n_results  # Fall back to all
        deriv_maxes = [
            max(d for d in r["dtheta_dh"] if np.isfinite(d)) if any(np.isfinite(d) for d in r["dtheta_dh"]) else 0.0
            for r in reliable
        ]
        h_ranges = [(min(r["h_values"]), max(r["h_values"])) for r in n_results]

        # Check if h-range covers h_c
        covers_hc = any(h_lo <= H_C <= h_hi for h_lo, h_hi in h_ranges)
        # Alternative: check if lowest h is within 0.5 of h_c
        closest_to_hc = min(abs(min(r["h_values"]) - H_C) for r in n_results)

        mean_peak = float(np.mean(pca_peaks))
        std_peak = float(np.std(pca_peaks)) if len(pca_peaks) > 1 else 0.0
        mean_deriv_max = float(np.mean(deriv_maxes))
        std_deriv_max = float(np.std(deriv_maxes)) if len(deriv_maxes) > 1 else 0.0
        h_lo_best = min(h_lo for h_lo, _ in h_ranges)
        h_hi_best = max(h_hi for _, h_hi in h_ranges)

        covers_str = "✅ YES" if covers_hc else f"❌ ({closest_to_hc:.1f} away)"
        logger.info(
            f"  {n:>4} | {mean_peak:>8.3f}±{std_peak:.3f} | "
            f"{mean_deriv_max:>10.4f}±{std_deriv_max:.4f} | "
            f"[{h_lo_best:.1f},{h_hi_best:.1f}] | {covers_str:>11} | {len(n_results):>5}"
        )

        stats_per_n.append({
            "n_qubits": n,
            "n_seeds": len(n_results),
            "pca_peak_mean": mean_peak,
            "pca_peak_std": std_peak,
            "pca_peaks": pca_peaks,
            "deriv_max_mean": mean_deriv_max,
            "deriv_max_std": std_deriv_max,
            "h_range": [h_lo_best, h_hi_best],
            "covers_hc": covers_hc,
            "closest_to_hc": closest_to_hc,
        })

    # ── Key findings ─────────────────────────────────────────────────
    # Classify results by regime
    regime_near_hc = [s for s in stats_per_n if s["covers_hc"]]
    regime_paramagnetic = [s for s in stats_per_n if not s["covers_hc"]]

    logger.info(f"\n  ── Key Findings ──")
    logger.info(f"  Trajectories covering h_c: {len(regime_near_hc)} system sizes")
    logger.info(f"  Trajectories in paramagnetic only: {len(regime_paramagnetic)} system sizes")

    if regime_near_hc:
        peaks_near_hc = [(s["n_qubits"], s["pca_peak_mean"]) for s in regime_near_hc]
        logger.info(f"\n  PCA peaks when h-range covers h_c (detectable regime):")
        for n, peak in peaks_near_hc:
            delta = abs(peak - H_C)
            status = "✅" if delta <= 0.3 else "⚠️"
            logger.info(f"    {status} N={n:>3}: peak at h={peak:.3f} (Δ from h_c = {delta:.3f})")

        # Convergence toward h_c
        if len(peaks_near_hc) >= 2:
            ns = [p[0] for p in peaks_near_hc]
            peaks = [p[1] for p in peaks_near_hc]
            # Linear trend
            if ns[-1] > ns[0]:
                slope = (peaks[-1] - peaks[0]) / (ns[-1] - ns[0])
                logger.info(
                    f"\n  Trend: PCA peak moves {'toward' if abs(peaks[-1] - H_C) < abs(peaks[0] - H_C) else 'away from'} h_c "
                    f"as N increases (slope={slope:.5f}/qubit)"
                )
                if abs(peaks[-1] - H_C) < abs(peaks[0] - H_C):
                    logger.info("  → Convergence toward thermodynamic limit h_c=1.0 ✅")

    if regime_paramagnetic:
        logger.info(f"\n  In paramagnetic regime (h >> h_c), PCA peak = lowest h tested:")
        for s in regime_paramagnetic[:5]:  # Show first 5
            logger.info(
                f"    N={s['n_qubits']:>3}: peak={s['pca_peak_mean']:.2f}, "
                f"h_range=[{s['h_range'][0]:.1f}, {s['h_range'][1]:.1f}]"
            )
        logger.info("  → Expected: no phase transition in valid regime (trivial landscape)")

    # Derivative amplitude scaling
    logger.info(f"\n  |∂θ/∂h| maximum amplitude vs N:")
    for s in stats_per_n:
        logger.info(f"    N={s['n_qubits']:>3}: max|∂θ/∂h| = {s['deriv_max_mean']:.4f}")

    # Check if amplitude decreases with N (smoother landscape at large N)
    if len(stats_per_n) >= 3:
        derivs = [(s["n_qubits"], s["deriv_max_mean"]) for s in stats_per_n if s["n_qubits"] >= 40]
        if len(derivs) >= 2:
            is_decreasing = all(derivs[i][1] >= derivs[i + 1][1] for i in range(len(derivs) - 1))
            logger.info(
                f"\n  Derivative amplitude {'decreases' if is_decreasing else 'does NOT monotonically decrease'} "
                f"with N (for N≥40)"
            )

    # ── Generate figure ──────────────────────────────────────────────
    _generate_scaling_figure(stats_per_n, fmt=fmt, theme=theme)

    # ── Save results ─────────────────────────────────────────────────
    output = {
        "analysis": "pca_peak_vs_N",
        "description": "How PCA peak position and θ-derivative amplitude scale with system size",
        "h_c_reference": H_C,
        "n_values_analyzed": n_values,
        "stats_per_n": stats_per_n,
        "findings": {
            "n_covering_hc": len(regime_near_hc),
            "n_paramagnetic_only": len(regime_paramagnetic),
            "peaks_converge_to_hc": (
                len(regime_near_hc) >= 2
                and abs(regime_near_hc[-1]["pca_peak_mean"] - H_C)
                < abs(regime_near_hc[0]["pca_peak_mean"] - H_C)
            )
            if len(regime_near_hc) >= 2
            else None,
        },
    }

    out_path = ROOT / "analysis" / "raw_data" / "pca_peak_vs_N.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\n  Results saved: {out_path.relative_to(ROOT)}")

    return output


def _generate_scaling_figure(
    stats_per_n: list[dict], fmt: str = "png", theme: str = "default"
) -> bool:
    """Generate 2-panel figure: PCA peak vs N + derivative amplitude vs N."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not available — skipping figure")
        return False

    if theme == "thesis":
        plt.rcParams.update({
            "font.family": "serif", "font.size": 11,
            "axes.labelsize": 12, "axes.titlesize": 13,
            "legend.fontsize": 9, "figure.dpi": 150,
            "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3,
        })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    n_vals = [s["n_qubits"] for s in stats_per_n]
    peaks = [s["pca_peak_mean"] for s in stats_per_n]
    peak_errs = [s["pca_peak_std"] for s in stats_per_n]
    derivs = [s["deriv_max_mean"] for s in stats_per_n]
    deriv_errs = [s["deriv_max_std"] for s in stats_per_n]
    covers = [s["covers_hc"] for s in stats_per_n]

    # ── Panel 1: PCA peak vs N ───────────────────────────────────────
    # Color by whether h-range covers h_c
    colors = ["#2196F3" if c else "#9E9E9E" for c in covers]
    markers = ["o" if c else "^" for c in covers]

    for i, (n, p, e, c, m) in enumerate(zip(n_vals, peaks, peak_errs, colors, markers)):
        ax1.errorbar(n, p, yerr=e, fmt=m, color=c, markersize=8, capsize=3, linewidth=1.5)

    # Reference line at h_c
    ax1.axhline(H_C, color="red", linestyle="--", alpha=0.7, linewidth=1.5, label=f"$h_c = {H_C}$")

    # Connect points covering h_c with a trend line
    hc_ns = [n_vals[i] for i in range(len(n_vals)) if covers[i]]
    hc_peaks = [peaks[i] for i in range(len(peaks)) if covers[i]]
    if len(hc_ns) >= 2:
        ax1.plot(hc_ns, hc_peaks, "b--", alpha=0.5, linewidth=1)

    ax1.set_xlabel("System size $N$ (qubits)")
    ax1.set_ylabel("PCA peak position $h_{\\mathrm{peak}}$")
    ax1.set_title("PCA Peak Position vs System Size")
    ax1.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="#2196F3", linestyle="", markersize=8,
                       label="h-range covers $h_c$"),
            plt.Line2D([0], [0], marker="^", color="#9E9E9E", linestyle="", markersize=8,
                       label="Paramagnetic only"),
            plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1.5, label=f"$h_c = {H_C}$"),
        ],
        loc="upper right",
    )
    ax1.set_xscale("log")

    # ── Panel 2: |∂θ/∂h| amplitude vs N ─────────────────────────────
    ax2.errorbar(n_vals, derivs, yerr=deriv_errs, fmt="s-", color="#E53935",
                 markersize=7, capsize=3, linewidth=1.5)
    ax2.set_xlabel("System size $N$ (qubits)")
    ax2.set_ylabel("max $|\\partial\\theta/\\partial h|$")
    ax2.set_title("θ-Derivative Amplitude vs System Size")
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    # Annotate key N values
    for i, n in enumerate(n_vals):
        if n in (6, 10, 40, 100, 200):
            ax2.annotate(f"N={n}", (n_vals[i], derivs[i]),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    outpath = FIGURES_DIR / f"fig_pca_peak_vs_N.{fmt}"
    plt.savefig(outpath, dpi=300 if fmt == "pdf" else 150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Figure saved: {outpath.relative_to(ROOT)}")
    return True


def main() -> None:
    """Run PCA phase detection analysis."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Task 2.2: PCA/Clustering Phase Detection from θ_opt(h)")
    logger.info("=" * 60)
    logger.info(f"  Hypothesis: PCA/clustering detects h_c≈{H_C} unsupervised")
    logger.info("  Success criterion: |detected - h_c| ≤ 0.3 for ≥2 topologies")
    logger.info("")

    trajectories = load_trajectories()
    logger.info(f"  Loaded {len(trajectories)} trajectories")

    # Filter by minimum points
    valid = [t for t in trajectories if t["n_points"] >= args.min_points]
    logger.info(f"  Valid (≥{args.min_points} h-points): {len(valid)}")

    # Analyze each trajectory
    results: list[dict] = []
    for traj in valid:
        result = analyze_trajectory(traj)
        if result:
            results.append(result)

    logger.info(f"\n  Analyzed: {len(results)} trajectories")

    # --- Cross-topology validation (Task 2.3) ---
    logger.info("\n" + "─" * 40)
    logger.info("  Cross-Topology Validation:")
    logger.info(
        f"  {'Topology':<12} | {'N':>3} | {'p':>2} | {'PCA peak':>8} | "
        f"{'K-means':>7} | {'|dθ/dh|':>7} | {'PCA ok':>6} | {'KM ok':>5}"
    )
    logger.info(
        f"  {'-' * 12}-+-{'-' * 3}-+-{'-' * 2}-+-{'-' * 8}-+-"
        f"{'-' * 7}-+-{'-' * 7}-+-{'-' * 6}-+-{'-' * 5}"
    )

    topo_success_pca: dict[str, bool] = {}
    topo_success_km: dict[str, bool] = {}

    for r in results:
        km_h = f"{r['kmeans_boundary_h']:.2f}" if r["kmeans_boundary_h"] else "N/A"
        logger.info(
            f"  {r['topology']:<12} | {r['n_qubits']:>3} | {r['p_layers']:>2} | "
            f"{r['pca_peak_h']:>8.2f} | {km_h:>7} | "
            f"{r['theta_deriv_peak_h']:>7.2f} | "
            f"{'✓' if r['success_pca'] else '✗':>6} | "
            f"{'✓' if r['success_kmeans'] else '✗':>5}"
        )
        topo = r["topology"]
        if topo not in topo_success_pca or r["success_pca"]:
            topo_success_pca[topo] = r["success_pca"]
        if topo not in topo_success_km or r["success_kmeans"]:
            topo_success_km[topo] = r["success_kmeans"]

    # Overall verdict
    n_topo_pca_pass = sum(1 for v in topo_success_pca.values() if v)
    n_topo_km_pass = sum(1 for v in topo_success_km.values() if v)
    total_topos = len(topo_success_pca)

    logger.info(f"\n  PCA detection: {n_topo_pca_pass}/{total_topos} topologies pass")
    logger.info(f"  K-means detection: {n_topo_km_pass}/{total_topos} topologies pass")

    overall_pass = n_topo_pca_pass >= 2 or n_topo_km_pass >= 2
    logger.info(
        f"\n  Overall: {'[PASS]' if overall_pass else '[FAIL]'} "
        f"(need ≥2 topologies within ±0.3 of h_c)"
    )

    # --- Generate figure ---
    logger.info("\n" + "─" * 40)
    generate_figure(results, fmt=args.format, theme=args.theme)

    # --- Save results ---
    output = {
        "metadata": {
            "hypothesis": (
                "PCA/clustering of θ_opt(h) reveals Z₂ transition at h_c≈1.0 without supervision"
            ),
            "success_criterion": "|detected - h_c| ≤ 0.3 for ≥2 topologies",
            "overall_pass": overall_pass,
            "n_topologies_pca_pass": n_topo_pca_pass,
            "n_topologies_kmeans_pass": n_topo_km_pass,
            "total_topologies": total_topos,
        },
        "per_trajectory": results,
    }

    OUTPUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_RESULTS, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"  Results saved: {OUTPUT_RESULTS.relative_to(ROOT)}")

    # --- Scaling analysis (PCA peak vs N) ---
    if args.scaling_analysis:
        run_scaling_analysis(results, fmt=args.format, theme=args.theme)

    logger.info("\n  Done.")


if __name__ == "__main__":
    main()
