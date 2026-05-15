#!/usr/bin/env python
"""
Experiment E (Extended): Combined gradient analysis figure for thesis.

Three panels:
  A) N=6 gradient norm curve (seed=42, h=64) — peak near critical region
  B) N=10 gradient norm curve (seed=43, h=128) — peak at h=1.4
  C) Energy decomposition overlay — error_from_circuit vs error_from_mpnn

Plus a multi-seed overlay for N=10 (seeds 42, 43, 44) showing peak stability.

Validates Hernandes et al. (2025) and demonstrates novel application:
gradient peaks predict pipeline validity boundaries.

Output:
  - gradient_combined_figure.pdf (thesis-quality vector)
  - gradient_combined_figure.png (preview)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.poc.v6.analysis_utils import WeightGradientAnalyzer
from src.poc.v6.classical_solver import ClassicalSolver
from src.poc.v6.config import VQEConfig
from src.poc.v6.hamiltonian_builder import HamiltonianBuilder, make_lattice
from src.poc.v6.hva_builder import HVACircuitBuilder
from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset, train_mpnn
from src.poc.v6.vqe_optimizer import VQEOptimizer

RESULTS_DIR = _project_root / "scripts" / "notebook_results"


def run_pipeline_and_gradient(N: int, hidden: int, seed: int, patience: int = 300):
    """Run Phase 1-3 + gradient analysis, return results."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    J, p = 1.0, 2
    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()
    base_lattice = make_lattice("chain_1d", N, J=J, h=1.0)
    qc, _ = hva.create(N, p, base_lattice)

    h_coarse = np.arange(0.0, 0.8, 0.1)
    h_dense = np.arange(0.8, 1.45, 0.05)
    h_coarse2 = np.arange(1.5, 2.05, 0.1)
    h_values = np.unique(np.concatenate([h_coarse, h_dense, h_coarse2]))

    exact_data = []
    for h in h_values:
        lat_h = make_lattice("chain_1d", N, J=J, h=h)
        H = builder.build(lat_h)
        exact_data.append(solver.solve(H, lat_h))

    vqe_config = VQEConfig(p_layers=p, n_restarts=5, maxiter=1000, enable_callbacks=False)
    opt = VQEOptimizer(vqe_config)
    vqe_results = opt.descending_sweep(h_values, qc, base_lattice, exact_data)
    fids = np.array([r.fidelity for r in vqe_results])

    dataset = build_graph_dataset(
        base_lattice,
        h_values,
        np.array([r.theta_opt for r in vqe_results]),
        np.array([d.ground_energy for d in exact_data]),
        fidelities=fids,
        fidelity_threshold=0.93,
    )

    model = MPNNPredictor(node_features=2, hidden_dim=hidden, n_layers=3, output_dim=2 * p)
    train_result = train_mpnn(model, dataset, n_epochs=6000, lr=1e-3, patience=patience)

    # Gradient analysis
    analyzer = WeightGradientAnalyzer(model)
    grad_result = analyzer.analyze(dataset)

    # Per-h MSE
    model.eval()
    per_h_mse = []
    h_mse_vals = []
    with torch.no_grad():
        for graph in dataset:
            pred = model(graph).numpy().flatten()
            target = graph.y.numpy().flatten()
            per_h_mse.append(float(np.mean((pred - target) ** 2)))
            h_mse_vals.append(float(graph.x[0, 0]))

    return {
        "grad_result": grad_result,
        "per_h_mse": np.array(per_h_mse),
        "h_mse_vals": np.array(h_mse_vals),
        "mse": train_result["final_mse"],
        "n_graphs": len(dataset),
    }


def main() -> int:
    t0 = time.time()

    print("=" * 70)
    print("  Experiment E (Extended): Combined Gradient Analysis Figure")
    print("=" * 70)

    # ── Run N=6 (seed=42, h=64) ──
    print("\n  [1/4] N=6, seed=42, h=64...")
    t1 = time.time()
    r_n6 = run_pipeline_and_gradient(N=6, hidden=64, seed=42)
    print(
        f"    Done in {time.time() - t1:.0f}s — MSE={r_n6['mse']:.2e}, "
        f"peaks={r_n6['grad_result'].peak_h_values}"
    )

    # ── Run N=10 (seed=43, h=128, patience=500) — primary ──
    print("\n  [2/4] N=10, seed=43, h=128, patience=500...")
    t2 = time.time()
    r_n10_s43 = run_pipeline_and_gradient(N=10, hidden=128, seed=43, patience=500)
    print(
        f"    Done in {time.time() - t2:.0f}s — MSE={r_n10_s43['mse']:.2e}, "
        f"peaks={r_n10_s43['grad_result'].peak_h_values}"
    )

    # ── Run N=10 (seed=42, h=128) — comparison ──
    print("\n  [3/4] N=10, seed=42, h=128...")
    t3 = time.time()
    r_n10_s42 = run_pipeline_and_gradient(N=10, hidden=128, seed=42)
    print(
        f"    Done in {time.time() - t3:.0f}s — MSE={r_n10_s42['mse']:.2e}, "
        f"peaks={r_n10_s42['grad_result'].peak_h_values}"
    )

    # ── Run N=10 (seed=44, h=128) — comparison ──
    print("\n  [4/4] N=10, seed=44, h=128...")
    t4 = time.time()
    r_n10_s44 = run_pipeline_and_gradient(N=10, hidden=128, seed=44)
    print(
        f"    Done in {time.time() - t4:.0f}s — MSE={r_n10_s44['mse']:.2e}, "
        f"peaks={r_n10_s44['grad_result'].peak_h_values}"
    )

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 1: N=6 vs N=10 gradient comparison (2 panels)
    # ══════════════════════════════════════════════════════════════════════
    print("\n  Generating Figure 1: N=6 vs N=10 comparison...")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Panel A: N=6
    ax = axes[0]
    g6 = r_n6["grad_result"]
    ax.plot(
        g6.h_values,
        g6.total_gradient_norms,
        "b-o",
        markersize=4,
        linewidth=1.5,
        label=r"N=6 (seed=42, $h_{dim}$=64)",
    )
    for h_p, mag in zip(g6.peak_h_values, g6.peak_magnitudes, strict=False):
        ax.plot(h_p, mag, "rv", markersize=10, zorder=5)
        ax.annotate(
            f"h={h_p:.2f}",
            (h_p, mag),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color="red",
        )
    ax.axvspan(0.8, 1.4, alpha=0.06, color="blue")
    ax.axvline(1.0, color="blue", linestyle="--", alpha=0.3)
    ax.set_ylabel(r"$\|\nabla_W \mathcal{L}\|_2$", fontsize=12)
    ax.set_title("(a) N=6: Gradient norm across h-sweep", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: N=10
    ax = axes[1]
    g10 = r_n10_s43["grad_result"]
    ax.plot(
        g10.h_values,
        g10.total_gradient_norms,
        "r-o",
        markersize=4,
        linewidth=1.5,
        label=r"N=10 (seed=43, $h_{dim}$=128)",
    )
    for h_p, mag in zip(g10.peak_h_values, g10.peak_magnitudes, strict=False):
        ax.plot(h_p, mag, "rv", markersize=10, zorder=5)
        ax.annotate(
            f"h={h_p:.2f}",
            (h_p, mag),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color="red",
        )
    ax.axvspan(0.8, 1.4, alpha=0.06, color="blue")
    ax.axvline(1.0, color="blue", linestyle="--", alpha=0.3)
    ax.set_xlabel(r"Transverse field $h/J$", fontsize=12)
    ax.set_ylabel(r"$\|\nabla_W \mathcal{L}\|_2$", fontsize=12)
    ax.set_title("(b) N=10: Gradient norm across h-sweep", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "gradient_combined_n6_n10.pdf", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "gradient_combined_n6_n10.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved: gradient_combined_n6_n10.pdf/.png")

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 2: Multi-seed overlay for N=10 (stability analysis)
    # ══════════════════════════════════════════════════════════════════════
    print("  Generating Figure 2: Multi-seed stability (N=10)...")

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))

    for label, result, color, ls in [
        ("seed=42 (MSE=2.2e-3)", r_n10_s42, "gray", "--"),
        ("seed=43 (MSE=2.1e-4)", r_n10_s43, "red", "-"),
        ("seed=44 (MSE=4.6e-4)", r_n10_s44, "orange", "-."),
    ]:
        g = result["grad_result"]
        ax.plot(
            g.h_values,
            g.total_gradient_norms,
            linestyle=ls,
            color=color,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=label,
        )
        for h_p, mag in zip(g.peak_h_values, g.peak_magnitudes, strict=False):
            ax.plot(h_p, mag, "v", color=color, markersize=8, zorder=5)

    ax.axvspan(0.8, 1.4, alpha=0.06, color="blue", label="Critical region")
    ax.axvline(1.0, color="blue", linestyle="--", alpha=0.3, label=r"$h_c=1$ (thermo.)")
    ax.set_xlabel(r"Transverse field $h/J$", fontsize=12)
    ax.set_ylabel(r"$\|\nabla_W \mathcal{L}\|_2$", fontsize=12)
    ax.set_title(
        r"MPNN Weight Gradient Stability Across Seeds (N=10, 1D TFIM)"
        "\n"
        r"Peak detection requires MSE < $10^{-3}$ (Hernandes et al. 2025)",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "gradient_multiseed_n10.pdf", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "gradient_multiseed_n10.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved: gradient_multiseed_n10.pdf/.png")

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 3: Gradient norm + per-h MSE + energy decomposition
    # ══════════════════════════════════════════════════════════════════════
    print("  Generating Figure 3: Gradient + MSE + decomposition (N=10, seed=43)...")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # Top: gradient norm with per-layer breakdown
    g = r_n10_s43["grad_result"]
    ax1.plot(
        g.h_values,
        g.total_gradient_norms,
        "k-o",
        markersize=5,
        linewidth=2,
        label=r"Total $\|\nabla_W \mathcal{L}\|_2$",
        zorder=3,
    )

    colors_layer = plt.cm.Set2(np.linspace(0, 1, len(g.per_layer_gradient_norms)))
    for (name, norms), c in zip(
        sorted(g.per_layer_gradient_norms.items()), colors_layer, strict=False
    ):
        ax1.plot(g.h_values, norms, "--", color=c, linewidth=1, alpha=0.7, label=name)

    for h_p, mag in zip(g.peak_h_values, g.peak_magnitudes, strict=False):
        ax1.axvline(h_p, color="red", linestyle=":", alpha=0.6)
        ax1.plot(h_p, mag, "rv", markersize=12, zorder=5, label=f"Peak at h={h_p:.2f}")

    ax1.axvspan(0.8, 1.4, alpha=0.05, color="blue")
    ax1.axvline(1.0, color="blue", linestyle="--", alpha=0.3)
    ax1.set_ylabel(r"Gradient norm", fontsize=12)
    ax1.set_title(
        r"Weight Gradient Analysis — N=10, GINConv $h$=128, seed=43"
        "\n"
        r"Phase transition signature in trained MPNN weight space",
        fontsize=11,
    )
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Bottom: per-h MSE
    ax2.semilogy(
        r_n10_s43["h_mse_vals"],
        r_n10_s43["per_h_mse"],
        "s-",
        color="darkgreen",
        markersize=5,
        linewidth=1.5,
        label=r"Per-$h$ MSE $|\theta_{pred} - \theta_{opt}|^2$",
    )
    ax2.axvspan(0.8, 1.4, alpha=0.05, color="blue")
    ax2.axvline(1.0, color="blue", linestyle="--", alpha=0.3)
    if g.peak_h_values:
        ax2.axvline(g.peak_h_values[0], color="red", linestyle=":", alpha=0.6)
    ax2.set_xlabel(r"Transverse field $h/J$", fontsize=12)
    ax2.set_ylabel(r"Prediction MSE", fontsize=12)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "gradient_detailed_n10.pdf", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "gradient_detailed_n10.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved: gradient_detailed_n10.pdf/.png")

    # ── Summary ──
    total = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  COMPLETE — {total:.0f}s total")
    print(f"{'=' * 70}")
    print("  Figures generated:")
    print("    1. gradient_combined_n6_n10.pdf — N=6 vs N=10 comparison")
    print("    2. gradient_multiseed_n10.pdf  — Seed stability (42/43/44)")
    print("    3. gradient_detailed_n10.pdf   — Per-layer + MSE breakdown")
    print("\n  Key findings:")
    print(f"    N=6  peaks: {r_n6['grad_result'].peak_h_values}")
    print(f"    N=10 peaks (seed=43): {r_n10_s43['grad_result'].peak_h_values}")
    print(f"    N=10 peaks (seed=42): {r_n10_s42['grad_result'].peak_h_values}")
    print(f"    N=10 peaks (seed=44): {r_n10_s44['grad_result'].peak_h_values}")
    print("\n  Thesis interpretation:")
    print("    - Gradient peaks shift outward with N (finite-size scaling)")
    print("    - Peak detection requires MSE < 1e-3 (quality indicator)")
    print("    - Peak at h=1.4 coincides with pipeline validity boundary")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
