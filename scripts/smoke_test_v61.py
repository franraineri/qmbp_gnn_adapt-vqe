#!/usr/bin/env python
"""
GNN-HVA v6.1 — Smoke Test for Hardware Deployer (Simulation Mode)

Runs an end-to-end pipeline exercising the V6.1 deployer with a denser
h-grid (12 points), more VQE restarts, longer MPNN training, multi-point
deployment validation, and weight gradient analysis.

Usage:
    python scripts/smoke_test_v61.py
"""

import logging
import sys
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
np.random.seed(42)
torch.manual_seed(42)


def main() -> int:
    t0 = time.time()

    from torch_geometric.data import Data

    from src.poc.v6.analysis_utils import WeightGradientAnalyzer
    from src.poc.v6.classical_solver import ClassicalSolver
    from src.poc.v6.config import VQEConfig
    from src.poc.v6.hamiltonian_builder import HamiltonianBuilder, make_lattice
    from src.poc.v6.hardware_deployer_v61 import HardwareDeployerV61
    from src.poc.v6.hva_builder import HVACircuitBuilder
    from src.poc.v6.mpnn_predictor import (
        MPNNPredictor,
        build_graph_dataset,
        load_mpnn_checkpoint,
        save_mpnn_checkpoint,
        train_mpnn,
    )
    from src.poc.v6.vqe_optimizer import VQEOptimizer

    N, J, p = 6, 1.0, 2
    # Denser h-grid: 12 points with extra density near critical region
    h_values = np.array([0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.0])

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()
    base_lattice = make_lattice("chain_1d", N, J=J, h=1.0)
    qc, _ = hva.create(N, p, base_lattice)

    # ── Phase 1 ──
    print(f"\n{'=' * 60}")
    print("PHASE 1: Exact diagonalization (12 h-points)")
    print(f"{'=' * 60}")
    t1 = time.time()
    exact_data = []
    for h in h_values:
        lat_h = make_lattice("chain_1d", N, J=J, h=h)
        H = builder.build(lat_h)
        exact_data.append(solver.solve(H, lat_h))
    print(f"  {len(exact_data)} points solved in {time.time() - t1:.1f}s")
    print(f"  E0 range: [{exact_data[0].ground_energy:.4f}, {exact_data[-1].ground_energy:.4f}]")
    print(
        f"  Gap range: [{min(d.gap for d in exact_data):.4f}, {max(d.gap for d in exact_data):.4f}]"
    )

    # ── Phase 2 ──
    print(f"\n{'=' * 60}")
    print("PHASE 2: VQE descending sweep (2 restarts, 300 iter)")
    print(f"{'=' * 60}")
    t2 = time.time()
    config = VQEConfig(n_restarts=2, maxiter=300, enable_callbacks=False)
    opt = VQEOptimizer(config)
    vqe_results = opt.descending_sweep(h_values, qc, base_lattice, exact_data)
    fids = [r.fidelity for r in vqe_results]
    n_good = sum(1 for f in fids if f >= 0.93)
    print(f"  Completed in {time.time() - t2:.1f}s")
    print(f"  Avg fidelity: {np.mean(fids) * 100:.1f}%")
    print(f"  Points fid ≥ 93%: {n_good}/{len(fids)}")
    for r in vqe_results:
        tag = "✅" if r.fidelity >= 0.93 else "⚠️"
        print(f"    {tag} h={r.h_value:.2f}: fid={r.fidelity:.4f}, ΔE={r.energy_error:.2e}")

    # ── Phase 3 ──
    print(f"\n{'=' * 60}")
    print("PHASE 3: MPNN training (1500 epochs, h=64, L=3)")
    print(f"{'=' * 60}")
    t3 = time.time()
    dataset = build_graph_dataset(
        base_lattice,
        h_values,
        np.array([r.theta_opt for r in vqe_results]),
        np.array([d.ground_energy for d in exact_data]),
        fidelities=np.array(fids),
        fidelity_threshold=0.93,
    )
    print(f"  Training graphs: {len(dataset)}/{len(h_values)} (fid threshold=0.93)")

    model = MPNNPredictor(node_features=2, hidden_dim=64, n_layers=3, output_dim=2 * p)
    train_result = train_mpnn(model, dataset, n_epochs=1500, lr=1e-3, patience=200)
    print(f"  Completed in {time.time() - t3:.1f}s")
    print(f"  Final MSE: {train_result['final_mse']:.6f}")
    print(f"  Early stop: {train_result['stopped_early']} ({train_result['stop_reason']})")

    # Save/load checkpoint round-trip
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    save_mpnn_checkpoint(model, ckpt_path, {"epochs": 1500, "mse": train_result["final_mse"]})
    load_mpnn_checkpoint(ckpt_path)  # verify round-trip works
    os.unlink(ckpt_path)
    print("  Checkpoint save/load: ✅")

    # ── Phase 4: V6.1 Deployer — Multi-point validation ──
    print(f"\n{'=' * 60}")
    print("PHASE 4: V6.1 HardwareDeployerV61 — multi-point deployment")
    print(f"{'=' * 60}")

    test_h_values = [0.5, 1.0, 1.25, 1.5, 2.0]
    deployer = HardwareDeployerV61(mode="simulation")
    model.eval()

    results = []
    for h_test in test_h_values:
        lat_test = make_lattice("chain_1d", N, J=J, h=h_test)
        H_test = builder.build(lat_test)
        exact_test = solver.solve(H_test, lat_test)

        # MPNN prediction
        edge_idx, coord = builder.build_graph_data(base_lattice)
        x_test = torch.tensor(
            np.stack([np.full(N, h_test), coord.astype(float)], axis=1),
            dtype=torch.float32,
        )
        test_graph = Data(x=x_test, edge_index=torch.tensor(edge_idx, dtype=torch.long))
        with torch.no_grad():
            theta_pred = model(test_graph).numpy().flatten()

        result = deployer.deploy_adapt_vqe(qc, H_test, theta_pred, lat_test, exact_test)
        results.append(result)

        de_tag = "✅" if result.delta_e_over_gap < 0.05 else "❌"
        print(
            f"  h={h_test:.2f}: E={result.predicted_energy:.4f}, "
            f"ΔE/gap={result.delta_e_over_gap:.4f} {de_tag}, "
            f"phase={result.phase_label}, "
            f"⟨X⟩={result.mag_x_pred:.3f}, ⟨ZZ⟩={result.corr_zz_pred:.3f}"
        )

    n_pass = sum(1 for r in results if r.delta_e_over_gap < 0.05)
    print(f"\n  Checklist: {n_pass}/{len(test_h_values)} pass ΔE/gap < 5%")

    # ── Weight Gradient Analysis ──
    print(f"\n{'=' * 60}")
    print("ANALYSIS: Weight gradient analyzer (Hernandes et al. 2025)")
    print(f"{'=' * 60}")
    analyzer = WeightGradientAnalyzer(model)
    grad_result = analyzer.analyze(dataset)
    print(f"  h-values analyzed: {len(grad_result.h_values)}")
    print(f"  Per-layer groups: {list(grad_result.per_layer_gradient_norms.keys())}")
    print(
        f"  Gradient norm range: [{grad_result.total_gradient_norms.min():.4f}, {grad_result.total_gradient_norms.max():.4f}]"
    )
    print(f"  Peaks detected: {len(grad_result.peak_h_values)}")
    print(f"  Critical region detected: {grad_result.critical_region_detected}")
    if grad_result.peak_h_values:
        for h_p, mag in zip(grad_result.peak_h_values, grad_result.peak_magnitudes, strict=False):
            print(f"    Peak at h={h_p:.2f}, magnitude={mag:.4f}")

    # ── Per-parameter heads test ──
    print(f"\n{'=' * 60}")
    print("BONUS: Per-parameter heads model")
    print(f"{'=' * 60}")
    model_pp = MPNNPredictor(
        node_features=2,
        hidden_dim=32,
        n_layers=2,
        output_dim=2 * p,
        per_parameter_heads=True,
    )
    train_pp = train_mpnn(model_pp, dataset, n_epochs=500, lr=1e-3, patience=100)
    print(f"  Final MSE: {train_pp['final_mse']:.6f}")
    if train_pp.get("zz_head_loss_history"):
        print(f"  ZZ-head loss (last): {train_pp['zz_head_loss_history'][-1]:.6f}")
        print(f"  X-head loss (last): {train_pp['x_head_loss_history'][-1]:.6f}")
    print("  Per-parameter heads: ✅")

    # ── Summary ──
    total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"V6.1 SMOKE TEST PASSED — {total:.1f}s total")
    print(f"  Deployment: {n_pass}/{len(test_h_values)} points pass ΔE/gap < 5%")
    print(
        f"  Gradient analysis: {'peaks found' if grad_result.peak_h_values else 'no peaks (expected with few points)'}"
    )
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\n❌ V6.1 SMOKE TEST FAILED: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
