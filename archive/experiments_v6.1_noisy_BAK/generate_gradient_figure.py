#!/usr/bin/env python
"""
Experiment E: Generate thesis Figure 4.x — Gradient norm vs h.

Uses the N=10 optimal config (seed=43, patience=500, h=128) to train the MPNN,
then computes the full weight gradient norm curve across the h-sweep.
Produces a publication-quality matplotlib figure showing the phase transition
signature in weight space (Hernandes et al. 2025).

Output: scripts/notebook_results/gradient_analysis_figure.png
"""

from __future__ import annotations

import sys
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


def main() -> int:
    N, J, p = 10, 1.0, 2
    seed = 43
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Generating gradient analysis figure (N=10, seed=43)...")

    # ── Pipeline (same as n10_patience500) ──
    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()
    base_lattice = make_lattice("chain_1d", N, J=J, h=1.0)
    qc, _ = hva.create(N, p, base_lattice)

    h_coarse = np.arange(0.0, 0.8, 0.1)
    h_dense = np.arange(0.8, 1.45, 0.05)
    h_coarse2 = np.arange(1.5, 2.05, 0.1)
    h_values = np.unique(np.concatenate([h_coarse, h_dense, h_coarse2]))

    print(f"  Phase 1: Exact diag ({len(h_values)} points)...")
    exact_data = []
    for h in h_values:
        lat_h = make_lattice("chain_1d", N, J=J, h=h)
        H = builder.build(lat_h)
        exact_data.append(solver.solve(H, lat_h))

    print("  Phase 2: VQE sweep...")
    vqe_config = VQEConfig(p_layers=p, n_restarts=5, maxiter=1000, enable_callbacks=False)
    opt = VQEOptimizer(vqe_config)
    vqe_results = opt.descending_sweep(h_values, qc, base_lattice, exact_data)
    fids = np.array([r.fidelity for r in vqe_results])

    print("  Phase 3: MPNN training...")
    dataset = build_graph_dataset(
        base_lattice,
        h_values,
        np.array([r.theta_opt for r in vqe_results]),
        np.array([d.ground_energy for d in exact_data]),
        fidelities=fids,
        fidelity_threshold=0.93,
    )
    model = MPNNPredictor(node_features=2, hidden_dim=128, n_layers=3, output_dim=2 * p)
    train_result = train_mpnn(model, dataset, n_epochs=6000, lr=1e-3, patience=500)
    print(f"  MSE={train_result['final_mse']:.2e}, graphs={len(dataset)}")

    # ── Gradient Analysis ──
    print("  Computing gradient norms...")
    analyzer = WeightGradientAnalyzer(model)
    grad_result = analyzer.analyze(dataset)

    h_grad = grad_result.h_values
    total_norms = grad_result.total_gradient_norms
    per_layer = grad_result.per_layer_gradient_norms
    peaks_h = grad_result.peak_h_values
    peaks_mag = grad_result.peak_magnitudes

    print(f"  Peaks: {len(peaks_h)} at h={peaks_h}")
    print(f"  Gradient norm range: [{total_norms.min():.6f}, {total_norms.max():.6f}]")

    # ── Generate Figure ──
    print("  Generating figure...")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Top panel: Gradient norms
    ax1.plot(
        h_grad,
        total_norms,
        "k-o",
        markersize=4,
        linewidth=1.5,
        label=r"$\|\nabla_W \mathcal{L}\|_2$ (total)",
    )

    # Per-layer breakdown
    colors = plt.cm.Set2(np.linspace(0, 1, len(per_layer)))
    for (layer_name, norms), color in zip(sorted(per_layer.items()), colors, strict=False):
        ax1.plot(h_grad, norms, "--", color=color, linewidth=1.0, alpha=0.7, label=f"{layer_name}")

    # Mark peaks
    for h_p, mag in zip(peaks_h, peaks_mag, strict=False):
        ax1.axvline(h_p, color="red", linestyle=":", alpha=0.5)
        ax1.plot(h_p, mag, "rv", markersize=10, zorder=5)

    # Mark critical region
    ax1.axvspan(0.8, 1.4, alpha=0.05, color="blue", label="Critical region")
    ax1.axvline(1.0, color="blue", linestyle="--", alpha=0.3, label=r"$h_c = 1$ (thermo. limit)")

    ax1.set_ylabel(r"Weight gradient norm $\|\nabla_W \mathcal{L}\|_2$", fontsize=12)
    ax1.set_title(
        r"Phase Transition Signature in MPNN Weight Space"
        "\n"
        r"(N=10, 1D TFIM, GINConv $h$=128, seed=43 — Hernandes et al. 2025)",
        fontsize=11,
    )
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Per-h MSE (prediction difficulty)
    model.eval()
    per_h_mse = []
    h_mse_vals = []
    with torch.no_grad():
        for graph in dataset:
            pred = model(graph).numpy().flatten()
            target = graph.y.numpy().flatten()
            mse = float(np.mean((pred - target) ** 2))
            per_h_mse.append(mse)
            h_mse_vals.append(float(graph.x[0, 0]))

    ax2.semilogy(h_mse_vals, per_h_mse, "s-", color="darkgreen", markersize=4, linewidth=1.5)
    ax2.set_xlabel(r"Transverse field $h/J$", fontsize=12)
    ax2.set_ylabel(r"Per-$h$ MSE $|\theta_{pred} - \theta_{opt}|^2$", fontsize=12)
    ax2.axvspan(0.8, 1.4, alpha=0.05, color="blue")
    ax2.axvline(1.0, color="blue", linestyle="--", alpha=0.3)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    fig_path = RESULTS_DIR / "gradient_analysis_figure.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved: {fig_path}")

    # Also save as PDF for thesis
    pdf_path = RESULTS_DIR / "gradient_analysis_figure.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  PDF saved: {pdf_path}")

    plt.close()

    # ── Print data for thesis table ──
    print("\n  Data for thesis (Table/Figure caption):")
    print("    N=10, 1D TFIM chain, J=1.0, HVA p=2")
    print("    MPNN: GINConv h=128, L=3, 6000 epochs, seed=43")
    print(f"    Training points: {len(dataset)}/27 (fidelity filter >= 0.93)")
    print(f"    Final MSE: {train_result['final_mse']:.2e}")
    print(f"    Gradient peak at h={peaks_h[0]:.2f}" if peaks_h else "    No peaks detected")
    print(
        f"    Critical region h in [0.8, 1.4]: {'peak detected' if grad_result.critical_region_detected else 'peak at boundary'}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
