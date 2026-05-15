#!/usr/bin/env python
"""
LEGACY: GNN-HVA v6.0 — Quick Smoke Test

╔══════════════════════════════════════════════════════════════════════════╗
║  SUPERSEDED by scripts/smoke_test_v61.py for V6.1 validation.          ║
║  Kept for backward compatibility with `make smoke-test` and to         ║
║  reproduce V6.0 baseline results.                                       ║
║  Uses deprecated HardwareDeployer and QRCPipeline.                      ║
╚══════════════════════════════════════════════════════════════════════════╝

Runs a reduced end-to-end pipeline (6 h-points, 500 MPNN epochs) to verify
all modules work correctly.  Expected runtime: ~7 seconds.

Usage:
    python scripts/smoke_test.py
    make smoke-test
"""

import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
)

np.random.seed(42)
torch.manual_seed(42)


def main() -> int:
    t0 = time.time()

    # ── Imports ───────────────────────────────────────────────────────
    print("Loading modules...")
    from torch_geometric.data import Data

    from src.poc.v6.classical_solver import ClassicalSolver
    from src.poc.v6.config import VQEConfig
    from src.poc.v6.hamiltonian_builder import HamiltonianBuilder, make_lattice
    from src.poc.v6.hardware_deployer import HardwareDeployer
    from src.poc.v6.hva_builder import HVACircuitBuilder
    from src.poc.v6.mpnn_predictor import (
        MPNNPredictor,
        build_graph_dataset,
        train_mpnn,
    )
    from src.poc.v6.pipeline_utils import (
        assert_observable_locality,
        load_phase12_dataset,
        save_phase12_dataset,
    )
    from src.poc.v6.qrc_pipeline import QRCPipeline
    from src.poc.v6.vqe_optimizer import VQEOptimizer

    N = 6
    J = 1.0
    p_layers = 2
    h_values = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 2.0])

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()
    base_lattice = make_lattice("chain_1d", N, J=J, h=1.0)
    qc, _ = hva.create(N, p_layers, base_lattice)

    # ── Phase 1: Ground Truth ─────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("PHASE 1: Exact diagonalization")
    print(f"{'=' * 60}")
    t1 = time.time()
    exact_data = []
    for h in h_values:
        lat_h = make_lattice("chain_1d", N, J=J, h=h)
        H = builder.build(lat_h)
        exact_data.append(solver.solve(H, lat_h))
    print(f"  {len(exact_data)} points solved in {time.time() - t1:.1f}s")
    print(f"  E0 range: [{exact_data[0].ground_energy:.4f}, {exact_data[-1].ground_energy:.4f}]")

    # ── Phase 2: VQE ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("PHASE 2: VQE descending sweep")
    print(f"{'=' * 60}")
    t2 = time.time()
    config = VQEConfig(n_restarts=1, maxiter=200, enable_callbacks=False)
    optimizer = VQEOptimizer(config)
    vqe_results = optimizer.descending_sweep(h_values, qc, base_lattice, exact_data)
    fids = [r.fidelity for r in vqe_results]
    n_good = sum(1 for f in fids if f >= 0.995)
    print(f"  Completed in {time.time() - t2:.1f}s")
    print(f"  Avg fidelity: {np.mean(fids) * 100:.1f}%")
    print(f"  Points fid ≥ 99.5%: {n_good}/{len(fids)}")
    for r in vqe_results:
        tag = "✅" if r.fidelity >= 0.995 else "⚠️"
        print(f"    {tag} h={r.h_value:.1f}: fid={r.fidelity:.4f}, ΔE={r.energy_error:.2e}")

    # ── Phase 3: MPNN ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("PHASE 3: MPNN training")
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

    model = MPNNPredictor(
        node_features=2,
        hidden_dim=32,
        n_layers=2,
        output_dim=2 * p_layers,
    )
    train_result = train_mpnn(model, dataset, n_epochs=500, lr=1e-3, patience=100)
    print(f"  Completed in {time.time() - t3:.1f}s")
    print(f"  Final MSE: {train_result['final_mse']:.4f}")
    print(f"  Early stop: {train_result['stopped_early']} ({train_result['stop_reason']})")

    # ── Phase 4: Deployment ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("PHASE 4: Dual-route deployment (h_test=1.5)")
    print(f"{'=' * 60}")
    h_test = 1.5

    # Exact reference for test point
    lat_test = make_lattice("chain_1d", N, J=J, h=h_test)
    H_test = builder.build(lat_test)
    exact_test = solver.solve(H_test, lat_test)

    # MPNN prediction
    model.eval()
    edge_idx, coord = builder.build_graph_data(base_lattice)
    x_test = torch.tensor(
        np.stack([np.full(N, h_test), coord.astype(float)], axis=1),
        dtype=torch.float32,
    )
    test_graph = Data(
        x=x_test,
        edge_index=torch.tensor(edge_idx, dtype=torch.long),
    )
    with torch.no_grad():
        theta_pred = model(test_graph).numpy().flatten()

    # Observable locality check
    ops_x, ops_zz = builder.build_local_observables(base_lattice)
    assert_observable_locality(ops_x + ops_zz, base_lattice.edges)

    # Route 1: Adapt-VQE
    deployer = HardwareDeployer()
    adapt_result = deployer.deploy_adapt_vqe(
        qc,
        H_test,
        theta_pred,
        lat_test,
        exact_test,
    )

    print("\n  --- Adapt-VQE ---")
    for key, passed in adapt_result.metrics_checklist.items():
        tag = "✅" if passed else "❌"
        print(f"    {tag} {key}")
    n_pass = sum(adapt_result.metrics_checklist.values())
    print(f"  Checklist: {n_pass}/6")
    print(f"  Phase: {adapt_result.phase_label}")

    # Route 2: QRC
    qrc = QRCPipeline(seed=42)
    qrc.build_reservoir(N, p_layers, base_lattice)
    qrc.train_readout(
        h_values,
        np.array([d.mag_x for d in exact_data]),
        np.array([d.corr_zz for d in exact_data]),
    )
    qrc_result = deployer.deploy_qrc(qrc, h_test, exact_test)

    print("\n  --- QRC Fallback ---")
    print(f"    ⟨X⟩ error:  {qrc_result.mag_x_error:.2e}")
    print(f"    ⟨ZZ⟩ error: {qrc_result.corr_zz_error:.2e}")
    print(f"    Phase: {qrc_result.phase_label}")

    # ── Pipeline integrity check ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("PIPELINE INTEGRITY")
    print(f"{'=' * 60}")
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = f.name
    save_phase12_dataset(
        tmp_path,
        h_values=h_values,
        J=J,
        n_qubits=N,
        p_layers=p_layers,
        ground_energies=np.array([d.ground_energy for d in exact_data]),
        gaps=np.array([d.gap for d in exact_data]),
        mag_x=np.array([d.mag_x for d in exact_data]),
        corr_zz=np.array([d.corr_zz for d in exact_data]),
        theta_opt=np.array([r.theta_opt for r in vqe_results]),
        vqe_energies=np.array([r.energy for r in vqe_results]),
        fidelities=np.array(fids),
    )
    loaded = load_phase12_dataset(tmp_path)
    os.unlink(tmp_path)
    print(f"  Save/load: ✅ (cost={loaded['cost_function']}, version={loaded['version']})")
    print("  Observable locality: ✅")

    # ── Summary ───────────────────────────────────────────────────────
    total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"V6.0 SMOKE TEST PASSED — {total:.0f}s total")
    print(f"  Adapt-VQE checklist: {n_pass}/6")
    print(f"  QRC phase classification: {qrc_result.phase_label}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"\n❌ SMOKE TEST FAILED: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
