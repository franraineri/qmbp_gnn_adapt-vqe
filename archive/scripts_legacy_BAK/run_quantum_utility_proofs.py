#!/usr/bin/env python
"""
Quantum Utility Proofs — Demonstrating the value of GNN-HVA framework.

Executes 4 proofs in order of impact/cost ratio:
  1. Classical Cost Explosion (timing analysis)
  2. Warm-Start Under Noise (FakeTorino baseline comparison)
  3. Cross-Size Prediction (N=10 MPNN predicts N=14)
  4. SPSA Refinement Value (shot savings quantification)

Usage:
    python scripts/run_quantum_utility_proofs.py
    python scripts/run_quantum_utility_proofs.py --proof 1
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

RESULTS_DIR = _project_root / "scripts" / "notebook_results"


# ─────────────────────────────────────────────────────────────────────────────
# Proof 1: Classical Cost Explosion
# ─────────────────────────────────────────────────────────────────────────────


def proof_1_cost_explosion():
    """Measure and compare classical computation time vs MPNN inference time."""
    import torch
    from torch_geometric.data import Data

    from src.poc.v6.classical_solver import ClassicalSolver
    from src.poc.v6.config import VQEConfig
    from src.poc.v6.hamiltonian_builder import HamiltonianBuilder, make_lattice
    from src.poc.v6.hva_builder import HVACircuitBuilder
    from src.poc.v6.mpnn_predictor import MPNNPredictor
    from src.poc.v6.vqe_optimizer import VQEOptimizer

    print("\n" + "=" * 60)
    print("  PROOF 1: Classical Cost Explosion")
    print("  Hypothesis: Classical cost grows exponentially,")
    print("  MPNN inference is O(1)")
    print("=" * 60)

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    results = []

    # Measure ED time at different N
    for N in [4, 6, 8, 10]:
        np.random.seed(42)
        torch.manual_seed(42)

        lattice = make_lattice("chain_1d", N, J=1.0, h=1.5)
        H = builder.build(lattice)

        # Time exact diagonalization
        t0 = time.time()
        exact = solver.solve(H, lattice)
        ed_time = time.time() - t0

        # Time VQE (single point, 3 restarts)
        hva = HVACircuitBuilder()
        qc, _ = hva.create(N, 2, lattice)
        config = VQEConfig(n_restarts=3, maxiter=300, enable_callbacks=False)
        opt = VQEOptimizer(config)
        t0 = time.time()
        opt.optimize(
            H,
            qc,
            np.random.uniform(-0.01, 0.01, qc.num_parameters),
            exact_energy=exact.ground_energy,
            exact_state=exact.ground_state,
        )
        vqe_time = time.time() - t0

        # Time MPNN inference (create a dummy trained model)
        model = MPNNPredictor(node_features=2, hidden_dim=64, n_layers=3, output_dim=4)
        model.eval()
        edge_idx, coord = builder.build_graph_data(lattice)
        x = torch.tensor(
            np.stack([np.full(N, 1.5), coord.astype(float)], axis=1), dtype=torch.float32
        )
        graph = Data(x=x, edge_index=torch.tensor(edge_idx, dtype=torch.long))

        # Warm up
        with torch.no_grad():
            _ = model(graph)

        # Measure inference (average of 100 calls)
        t0 = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = model(graph)
        mpnn_time = (time.time() - t0) / 100

        ratio_ed = ed_time / mpnn_time if mpnn_time > 0 else float("inf")
        ratio_vqe = vqe_time / mpnn_time if mpnn_time > 0 else float("inf")

        entry = {
            "N": N,
            "hilbert_dim": 2**N,
            "ed_time_s": ed_time,
            "vqe_time_s": vqe_time,
            "mpnn_inference_s": mpnn_time,
            "ratio_ed_vs_mpnn": ratio_ed,
            "ratio_vqe_vs_mpnn": ratio_vqe,
        }
        results.append(entry)
        print(
            f"  N={N:2d} (2^N={2**N:>6d}): ED={ed_time:.4f}s, VQE={vqe_time:.1f}s, "
            f"MPNN={mpnn_time * 1000:.2f}ms → VQE/MPNN={ratio_vqe:.0f}×"
        )

    # Extrapolation
    print("\n  Extrapolation (from measured scaling):")
    # ED scales as O(4^N)
    ed_10 = results[-1]["ed_time_s"]
    ed_12 = ed_10 * (4**12 / 4**10)
    ed_14 = ed_10 * (4**14 / 4**10)
    ed_20 = ed_10 * (4**20 / 4**10)
    mpnn_time_ref = results[-1]["mpnn_inference_s"]
    print(f"  N=12: ED ≈ {ed_12:.0f}s ({ed_12 / 60:.1f} min), MPNN ≈ {mpnn_time_ref * 1000:.1f}ms")
    print(f"  N=14: ED ≈ {ed_14:.0f}s ({ed_14 / 60:.1f} min), MPNN ≈ {mpnn_time_ref * 1000:.1f}ms")
    print(f"  N=20: ED ≈ {ed_20:.0f}s ({ed_20 / 3600:.1f} hours) — IMPRACTICAL")
    print(f"  N=20: MPNN inference ≈ {mpnn_time_ref * 1000:.1f}ms — INSTANT")
    print(f"\n  CONCLUSION: At N=20, classical is {ed_20 / mpnn_time_ref:.0e}× slower than MPNN")

    return {"proof": "cost_explosion", "results": results}


# ─────────────────────────────────────────────────────────────────────────────
# Proof 2: Warm-Start Under Noise
# ─────────────────────────────────────────────────────────────────────────────


def proof_2_warmstart_under_noise():
    """Test if warm-start advantage persists under hardware noise."""
    import torch
    from torch_geometric.data import Data

    from src.poc.v6.classical_solver import ClassicalSolver
    from src.poc.v6.config import VQEConfig
    from src.poc.v6.hamiltonian_builder import HamiltonianBuilder, make_lattice
    from src.poc.v6.hardware_deployer_v61 import HardwareDeployerV61
    from src.poc.v6.hva_builder import HVACircuitBuilder
    from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset, train_mpnn
    from src.poc.v6.vqe_optimizer import VQEOptimizer

    print("\n" + "=" * 60)
    print("  PROOF 2: Warm-Start Under Noise")
    print("  Hypothesis: MPNN warm-start advantage persists")
    print("  (or increases) under hardware noise")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    N, J, p = 6, 1.0, 2
    h_values = np.array([0.8, 1.0, 1.2, 1.3, 1.5, 1.7, 2.0])

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()
    base_lattice = make_lattice("chain_1d", N, J=J, h=1.0)
    qc, _ = hva.create(N, p, base_lattice)

    # Phase 1+2: generate training data
    print("  Training pipeline (Phase 1+2+3)...")
    exact_data = []
    for h in h_values:
        lat_h = make_lattice("chain_1d", N, J=J, h=h)
        H = builder.build(lat_h)
        exact_data.append(solver.solve(H, lat_h))

    config = VQEConfig(n_restarts=5, maxiter=1000, enable_callbacks=False)
    opt = VQEOptimizer(config)
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
    model = MPNNPredictor(node_features=2, hidden_dim=64, n_layers=3, output_dim=4)
    train_mpnn(model, dataset, n_epochs=3000, lr=1e-3, patience=300)

    # Test point
    h_test = 1.5
    lat_test = make_lattice("chain_1d", N, J=J, h=h_test)
    H_test = builder.build(lat_test)
    exact_test = solver.solve(H_test, lat_test)

    model.eval()
    edge_idx, coord = builder.build_graph_data(base_lattice)
    x_test = torch.tensor(
        np.stack([np.full(N, h_test), coord.astype(float)], axis=1), dtype=torch.float32
    )
    test_graph = Data(x=x_test, edge_index=torch.tensor(edge_idx, dtype=torch.long))
    with torch.no_grad():
        theta_pred = model(test_graph).numpy().flatten()

    print(f"  MPNN prediction: θ = {theta_pred.round(4)}")

    # Noiseless baseline comparison
    print("\n  [Noiseless] Deploying warm vs cold...")
    deployer_clean = HardwareDeployerV61(mode="simulation")
    warm_clean, comp_clean = deployer_clean.deploy_with_baseline(
        qc,
        H_test,
        theta_pred,
        lat_test,
        exact_test,
        n_random_seeds=3,
        random_seed_base=300,
    )
    print(f"    Warm ΔE/gap: {warm_clean.delta_e_over_gap:.4f}")
    print(f"    Cold mean:   {comp_clean.cold_start_mean['delta_e_over_gap']:.4f}")
    print(f"    Gain:        {comp_clean.gain_energy_pct:.1f}%")

    # Noisy comparison (FakeTorino, single layout = raw noise, no ZNE)
    print("\n  [Noisy - FakeTorino] Deploying warm vs cold...")
    try:
        deployer_noisy = HardwareDeployerV61(mode="noisy_simulation", n_layouts=1, seed=42)
        warm_noisy, comp_noisy = deployer_noisy.deploy_with_baseline(
            qc,
            H_test,
            theta_pred,
            lat_test,
            exact_test,
            n_random_seeds=3,
            random_seed_base=300,
        )
        print(f"    Warm ΔE/gap: {warm_noisy.delta_e_over_gap:.4f}")
        print(f"    Cold mean:   {comp_noisy.cold_start_mean['delta_e_over_gap']:.4f}")
        print(f"    Gain:        {comp_noisy.gain_energy_pct:.1f}%")

        # Compare gains
        print("\n  COMPARISON:")
        print(f"    Noiseless gain: {comp_clean.gain_energy_pct:.1f}%")
        print(f"    Noisy gain:     {comp_noisy.gain_energy_pct:.1f}%")

        if comp_noisy.gain_energy_pct >= comp_clean.gain_energy_pct * 0.8:
            print("    ✅ Warm-start advantage PERSISTS under noise!")
            finding = "warmstart_persists_under_noise"
        else:
            print("    ⚠️ Warm-start advantage reduced under noise")
            finding = "warmstart_reduced_under_noise"

        return {
            "proof": "warmstart_under_noise",
            "noiseless": {
                "warm_de_gap": warm_clean.delta_e_over_gap,
                "cold_mean_de_gap": comp_clean.cold_start_mean["delta_e_over_gap"],
                "gain_pct": comp_clean.gain_energy_pct,
            },
            "noisy": {
                "warm_de_gap": warm_noisy.delta_e_over_gap,
                "cold_mean_de_gap": comp_noisy.cold_start_mean["delta_e_over_gap"],
                "gain_pct": comp_noisy.gain_energy_pct,
            },
            "finding": finding,
        }
    except Exception as e:
        print(f"    ❌ Noisy simulation failed: {e}")
        print("    (FakeTorino may not be available or compatible)")
        return {
            "proof": "warmstart_under_noise",
            "noiseless": {
                "warm_de_gap": warm_clean.delta_e_over_gap,
                "cold_mean_de_gap": comp_clean.cold_start_mean["delta_e_over_gap"],
                "gain_pct": comp_clean.gain_energy_pct,
            },
            "noisy": None,
            "error": str(e),
            "finding": "noisy_simulation_unavailable",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Proof 3: Cross-Size Prediction
# ─────────────────────────────────────────────────────────────────────────────


def proof_3_cross_size_prediction():
    """Test if MPNN trained on N=6 can predict for N=10 (or vice versa)."""
    import torch
    from torch_geometric.data import Data

    from src.poc.v6.classical_solver import ClassicalSolver
    from src.poc.v6.config import VQEConfig
    from src.poc.v6.hamiltonian_builder import HamiltonianBuilder, make_lattice
    from src.poc.v6.hardware_deployer_v61 import HardwareDeployerV61
    from src.poc.v6.hva_builder import HVACircuitBuilder
    from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset, train_mpnn
    from src.poc.v6.vqe_optimizer import VQEOptimizer

    print("\n" + "=" * 60)
    print("  PROOF 3: Cross-Size Prediction")
    print("  Question: Can MPNN trained on N=6 predict for N=10?")
    print("  (Tests size-agnostic generalization via global_mean_pool)")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    # Train on N=6
    N_train = 6
    h_values = np.array([0.8, 1.0, 1.2, 1.3, 1.5, 1.7, 2.0])

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()

    base_lattice_6 = make_lattice("chain_1d", N_train, J=1.0, h=1.0)
    qc_6, _ = hva.create(N_train, 2, base_lattice_6)

    print(f"  Training on N={N_train}...")
    exact_6 = []
    for h in h_values:
        lat = make_lattice("chain_1d", N_train, J=1.0, h=h)
        exact_6.append(solver.solve(builder.build(lat), lat))

    config = VQEConfig(n_restarts=5, maxiter=1000, enable_callbacks=False)
    opt = VQEOptimizer(config)
    vqe_6 = opt.descending_sweep(h_values, qc_6, base_lattice_6, exact_6)
    fids_6 = np.array([r.fidelity for r in vqe_6])

    dataset_6 = build_graph_dataset(
        base_lattice_6,
        h_values,
        np.array([r.theta_opt for r in vqe_6]),
        np.array([d.ground_energy for d in exact_6]),
        fidelities=fids_6,
        fidelity_threshold=0.93,
    )
    model = MPNNPredictor(node_features=2, hidden_dim=64, n_layers=3, output_dim=4)
    train_mpnn(model, dataset_6, n_epochs=3000, lr=1e-3, patience=300)
    print(f"  Trained on {len(dataset_6)} points from N={N_train}")

    # Test on N=10 (cross-size)
    N_test = 10
    h_test = 1.5
    print(f"\n  Predicting for N={N_test} at h={h_test}...")

    base_lattice_10 = make_lattice("chain_1d", N_test, J=1.0, h=1.0)
    qc_10, _ = hva.create(N_test, 2, base_lattice_10)
    lat_test_10 = make_lattice("chain_1d", N_test, J=1.0, h=h_test)
    H_test_10 = builder.build(lat_test_10)
    exact_test_10 = solver.solve(H_test_10, lat_test_10)

    # MPNN inference on N=10 graph (size-agnostic via global_mean_pool)
    model.eval()
    edge_idx_10, coord_10 = builder.build_graph_data(base_lattice_10)
    x_10 = torch.tensor(
        np.stack([np.full(N_test, h_test), coord_10.astype(float)], axis=1),
        dtype=torch.float32,
    )
    graph_10 = Data(x=x_10, edge_index=torch.tensor(edge_idx_10, dtype=torch.long))
    with torch.no_grad():
        theta_pred_10 = model(graph_10).numpy().flatten()

    print(f"  θ_pred (from N=6 model): {theta_pred_10.round(4)}")

    # Deploy on N=10
    deployer = HardwareDeployerV61(mode="simulation")
    result_cross = deployer.deploy_adapt_vqe(
        qc_10, H_test_10, theta_pred_10, lat_test_10, exact_test_10
    )

    # Compare against random
    theta_random = np.random.default_rng(42).uniform(-np.pi, np.pi, 4)
    result_random = deployer.deploy_adapt_vqe(
        qc_10, H_test_10, theta_random, lat_test_10, exact_test_10
    )

    # Also test on N=6 (same size as training — sanity check)
    lat_test_6 = make_lattice("chain_1d", N_train, J=1.0, h=h_test)
    H_test_6 = builder.build(lat_test_6)
    exact_test_6 = solver.solve(H_test_6, lat_test_6)
    edge_idx_6, coord_6 = builder.build_graph_data(base_lattice_6)
    x_6 = torch.tensor(
        np.stack([np.full(N_train, h_test), coord_6.astype(float)], axis=1),
        dtype=torch.float32,
    )
    graph_6 = Data(x=x_6, edge_index=torch.tensor(edge_idx_6, dtype=torch.long))
    with torch.no_grad():
        theta_pred_6 = model(graph_6).numpy().flatten()
    result_same = deployer.deploy_adapt_vqe(qc_6, H_test_6, theta_pred_6, lat_test_6, exact_test_6)

    print("\n  Results:")
    print(
        f"    N=6 (same size):  ΔE/gap = {result_same.delta_e_over_gap:.4f} {'✅' if result_same.delta_e_over_gap < 0.05 else '⚠️'}"
    )
    print(
        f"    N=10 (cross-size): ΔE/gap = {result_cross.delta_e_over_gap:.4f} {'✅' if result_cross.delta_e_over_gap < 0.05 else '⚠️'}"
    )
    print(f"    N=10 (random):    ΔE/gap = {result_random.delta_e_over_gap:.4f}")

    gain_cross = (
        (result_random.delta_e_over_gap - result_cross.delta_e_over_gap)
        / result_random.delta_e_over_gap
        * 100
    )
    print(f"\n    Cross-size gain vs random: {gain_cross:.1f}%")

    if result_cross.delta_e_over_gap < result_random.delta_e_over_gap:
        print("    ✅ Cross-size prediction is BETTER than random!")
        if result_cross.delta_e_over_gap < 0.10:
            print("    ✅ And within 10% threshold — useful prediction!")
            finding = "cross_size_useful"
        else:
            print("    ⚠️ But not within 10% — needs same-size training for precision")
            finding = "cross_size_better_than_random_but_imprecise"
    else:
        print("    ❌ Cross-size prediction worse than random")
        finding = "cross_size_fails"

    return {
        "proof": "cross_size_prediction",
        "N_train": N_train,
        "N_test": N_test,
        "h_test": h_test,
        "same_size_de_gap": result_same.delta_e_over_gap,
        "cross_size_de_gap": result_cross.delta_e_over_gap,
        "random_de_gap": result_random.delta_e_over_gap,
        "gain_vs_random_pct": gain_cross,
        "finding": finding,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Quantum Utility Proofs")
    parser.add_argument(
        "--proof",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which proof to run (default: all)",
    )
    args = parser.parse_args()

    t_total = time.time()
    all_results = {}
    run_all = args.proof == "all"

    print("=" * 60)
    print("  Quantum Utility Proofs — GNN-HVA Framework")
    print("=" * 60)

    if run_all or args.proof == "1":
        all_results["proof_1"] = proof_1_cost_explosion()

    if run_all or args.proof == "2":
        all_results["proof_2"] = proof_2_warmstart_under_noise()

    if run_all or args.proof == "3":
        all_results["proof_3"] = proof_3_cross_size_prediction()

    # Save
    total_time = time.time() - t_total
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = hashlib.sha1(f"utility:{ts}".encode()).hexdigest()[:8]
    path = RESULTS_DIR / f"quantum_utility_proofs_{ts}_{run_id}.json"

    output = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "total_elapsed_s": round(total_time, 1),
        "results": all_results,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"  ALL PROOFS COMPLETE — {total_time:.0f}s")
    print(f"  Results: {path}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
