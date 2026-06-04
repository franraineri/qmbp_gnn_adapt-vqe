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

    # --- PCA Analysis ---
    scaler = StandardScaler()
    theta_scaled = scaler.fit_transform(theta_opt)

    pca = PCA(n_components=min(2, theta_opt.shape[1]))
    pc_scores = pca.fit_transform(theta_scaled)
    pc1 = pc_scores[:, 0]

    # --- |dPC1/dh| derivative peak ---
    dpc1_dh = np.abs(np.gradient(pc1, h_values))
    peak_idx = np.argmax(dpc1_dh)
    pca_peak_h = float(h_values[peak_idx])

    # --- |dθ/dh| total derivative (L2 norm across all params) ---
    dtheta_dh = np.zeros(len(h_values))
    for i in range(theta_opt.shape[1]):
        dtheta_dh += np.gradient(theta_opt[:, i], h_values) ** 2
    dtheta_dh = np.sqrt(dtheta_dh)
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
    return parser.parse_args()


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
    logger.info("\n  Done.")


if __name__ == "__main__":
    main()
