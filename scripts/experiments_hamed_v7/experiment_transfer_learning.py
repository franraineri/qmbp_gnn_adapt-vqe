"""
Experiment: Transfer Learning N=6 → N=10

Hypothesis:
    Pre-training MPNN on N=6 data (17 points, cheap) then fine-tuning on N=10
    data (14 points) could:
    1. Improve seed stability (reduce variance across seeds 42/43/44)
    2. Improve generalization at the regime boundary (h≈1.4)
    3. Converge faster with fewer N=10 training epochs

    The MPNN is lattice-agnostic (GINConv + global_mean_pool) so the same
    architecture handles both N=6 and N=10 graphs. output_dim=4 (2×p) is
    identical for both system sizes.

Counter-hypothesis:
    V7 2B proved predictor is NOT the bottleneck at N=10 (error_from_mpnn=0.000
    with seed=43). Transfer learning may not help because the model already
    achieves perfect prediction with sufficient training.

    However, seed=42 gives 10× worse MSE — transfer learning might stabilize this.

Usage:
    .venv/bin/python scripts/experiments_hamed_v7/experiment_transfer_learning.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiment_utils import (
    SubExperimentResult,
    evaluate_energy_statevector,
    save_experiment_result,
)

from src.poc.v6 import (
    ClassicalSolver,
    HamiltonianBuilder,
    HVACircuitBuilder,
    VQEOptimizer,
    make_lattice,
)
from src.poc.v6.config import VQEConfig
from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset, train_mpnn

RESULTS_DIR = Path(__file__).parent / "results"


def generate_pipeline_data(N: int, seed: int = 42):
    """Run Phase 1+2 to generate training data for a given N."""
    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()
    base_lattice = make_lattice("chain_1d", N, J=1.0, h=1.0)
    qc, _ = hva.create(N, 2, base_lattice)

    # Standard h-grid
    h_coarse = np.arange(0.0, 0.8, 0.1)
    h_dense = np.arange(0.8, 1.45, 0.05)
    h_coarse2 = np.arange(1.5, 2.05, 0.1)
    h_values = np.unique(np.concatenate([h_coarse, h_dense, h_coarse2]))

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Phase 1
    exact_data = []
    for h in h_values:
        lat_h = make_lattice("chain_1d", N, J=1.0, h=h)
        H = builder.build(lat_h)
        exact_data.append(solver.solve(H, lat_h))

    # Phase 2
    vqe_config = VQEConfig(p_layers=2, n_restarts=5, maxiter=1000, enable_callbacks=False)
    opt = VQEOptimizer(vqe_config)
    vqe_results = opt.descending_sweep(h_values, qc, base_lattice, exact_data)
    fids = np.array([r.fidelity for r in vqe_results])

    return {
        "base_lattice": base_lattice,
        "qc": qc,
        "h_values": h_values,
        "theta_opt": np.array([r.theta_opt for r in vqe_results]),
        "exact_energies": np.array([d.ground_energy for d in exact_data]),
        "fidelities": fids,
        "exact_data": exact_data,
        "builder": builder,
    }


def evaluate_model(model, builder, base_lattice, qc, h_test_values, N, exact_data_map):
    """Evaluate MPNN at test h-values, return per-h ΔE."""
    from torch_geometric.data import Data

    model.eval()
    errors = []

    for h_t in h_test_values:
        edge_idx, coord = builder.build_graph_data(base_lattice)
        x_test = torch.tensor(
            np.stack([np.full(N, h_t), coord.astype(float)], axis=1),
            dtype=torch.float32,
        )
        test_graph = Data(x=x_test, edge_index=torch.tensor(edge_idx, dtype=torch.long))

        with torch.no_grad():
            theta_pred = model(test_graph).numpy().flatten()

        lat_t = make_lattice("chain_1d", N, J=1.0, h=h_t)
        H_t = HamiltonianBuilder().build(lat_t)
        energy = evaluate_energy_statevector(qc, H_t, theta_pred)
        exact_e = exact_data_map[h_t]
        errors.append(abs(energy - exact_e))

    return errors


def run_experiment():
    """Compare: baseline N=10 training vs transfer learning N=6→N=10."""
    print("=" * 70)
    print("  Transfer Learning Experiment: N=6 → N=10")
    print("=" * 70)

    seeds = [42, 43, 44]
    h_test_values = [1.25, 1.4, 1.5]
    fid_threshold = 0.93

    # ── Generate data ──
    print("\n  Generating N=6 training data...")
    t0 = time.time()
    data_n6 = generate_pipeline_data(N=6, seed=42)
    print(f"    Done in {time.time() - t0:.1f}s")

    all_results = []

    for seed in seeds:
        print(f"\n{'─' * 60}")
        print(f"  Seed: {seed}")
        print(f"{'─' * 60}")

        # Generate N=10 data with this seed
        print(f"  Generating N=10 training data (seed={seed})...")
        t0 = time.time()
        data_n10 = generate_pipeline_data(N=10, seed=seed)
        print(f"    Done in {time.time() - t0:.1f}s")

        # Build datasets
        dataset_n6 = build_graph_dataset(
            data_n6["base_lattice"],
            data_n6["h_values"],
            data_n6["theta_opt"],
            data_n6["exact_energies"],
            fidelities=data_n6["fidelities"],
            fidelity_threshold=fid_threshold,
        )
        dataset_n10 = build_graph_dataset(
            data_n10["base_lattice"],
            data_n10["h_values"],
            data_n10["theta_opt"],
            data_n10["exact_energies"],
            fidelities=data_n10["fidelities"],
            fidelity_threshold=fid_threshold,
        )
        print(f"    N=6 dataset: {len(dataset_n6)} graphs")
        print(f"    N=10 dataset: {len(dataset_n10)} graphs")

        # Build exact energy map for evaluation
        exact_map_n10 = {}
        for i, h in enumerate(data_n10["h_values"]):
            exact_map_n10[float(h)] = data_n10["exact_data"][i].ground_energy
        # Add test points if not in grid
        for h_t in h_test_values:
            if h_t not in exact_map_n10:
                lat_t = make_lattice("chain_1d", 10, J=1.0, h=h_t)
                H_t = HamiltonianBuilder().build(lat_t)
                exact_map_n10[h_t] = ClassicalSolver().solve(H_t, lat_t).ground_energy

        # ── Method A: Baseline (train on N=10 only) ──
        print("\n  Method A: Baseline (N=10 only, 6000 epochs)...")
        torch.manual_seed(seed)
        model_baseline = MPNNPredictor(
            node_features=2,
            hidden_dim=128,
            n_layers=3,
            output_dim=4,
        )
        t0 = time.time()
        res_baseline = train_mpnn(
            model_baseline,
            dataset_n10,
            n_epochs=6000,
            lr=1e-3,
            patience=500,
        )
        time_baseline = time.time() - t0
        errors_baseline = evaluate_model(
            model_baseline,
            data_n10["builder"],
            data_n10["base_lattice"],
            data_n10["qc"],
            h_test_values,
            10,
            exact_map_n10,
        )
        print(f"    MSE={res_baseline['final_mse']:.2e}, time={time_baseline:.1f}s")
        print(f"    Test ΔE: {[f'{e:.2e}' for e in errors_baseline]}")

        # ── Method B: Transfer learning (pre-train N=6, fine-tune N=10) ──
        print("\n  Method B: Transfer (pre-train N=6 3000ep, fine-tune N=10 3000ep)...")
        torch.manual_seed(seed)
        model_transfer = MPNNPredictor(
            node_features=2,
            hidden_dim=128,
            n_layers=3,
            output_dim=4,
        )

        # Phase 1: Pre-train on N=6
        t0 = time.time()
        res_pretrain = train_mpnn(
            model_transfer,
            dataset_n6,
            n_epochs=3000,
            lr=1e-3,
            patience=300,
        )
        print(f"    Pre-train MSE (N=6): {res_pretrain['final_mse']:.2e}")

        # Phase 2: Fine-tune on N=10 with lower LR
        res_finetune = train_mpnn(
            model_transfer,
            dataset_n10,
            n_epochs=3000,
            lr=3e-4,
            patience=500,  # Lower LR for fine-tuning
        )
        time_transfer = time.time() - t0
        errors_transfer = evaluate_model(
            model_transfer,
            data_n10["builder"],
            data_n10["base_lattice"],
            data_n10["qc"],
            h_test_values,
            10,
            exact_map_n10,
        )
        print(
            f"    Fine-tune MSE (N=10): {res_finetune['final_mse']:.2e}, time={time_transfer:.1f}s"
        )
        print(f"    Test ΔE: {[f'{e:.2e}' for e in errors_transfer]}")

        # ── Method C: Combined dataset (N=6 + N=10 together) ──
        print("\n  Method C: Combined (N=6 + N=10 data, 6000 epochs)...")
        torch.manual_seed(seed)
        model_combined = MPNNPredictor(
            node_features=2,
            hidden_dim=128,
            n_layers=3,
            output_dim=4,
        )
        dataset_combined = dataset_n6 + dataset_n10
        t0 = time.time()
        res_combined = train_mpnn(
            model_combined,
            dataset_combined,
            n_epochs=6000,
            lr=1e-3,
            patience=500,
        )
        time_combined = time.time() - t0
        errors_combined = evaluate_model(
            model_combined,
            data_n10["builder"],
            data_n10["base_lattice"],
            data_n10["qc"],
            h_test_values,
            10,
            exact_map_n10,
        )
        print(f"    MSE={res_combined['final_mse']:.2e}, time={time_combined:.1f}s")
        print(f"    Test ΔE: {[f'{e:.2e}' for e in errors_combined]}")

        # Record
        all_results.append(
            {
                "seed": seed,
                "n6_graphs": len(dataset_n6),
                "n10_graphs": len(dataset_n10),
                "baseline": {
                    "mse": res_baseline["final_mse"],
                    "errors": errors_baseline,
                    "avg_error": float(np.mean(errors_baseline)),
                    "time_s": time_baseline,
                },
                "transfer": {
                    "pretrain_mse": res_pretrain["final_mse"],
                    "finetune_mse": res_finetune["final_mse"],
                    "errors": errors_transfer,
                    "avg_error": float(np.mean(errors_transfer)),
                    "time_s": time_transfer,
                },
                "combined": {
                    "mse": res_combined["final_mse"],
                    "errors": errors_combined,
                    "avg_error": float(np.mean(errors_combined)),
                    "time_s": time_combined,
                },
            }
        )

    # ── Summary ──
    print(f"\n\n{'=' * 70}")
    print("  SUMMARY — Average ΔE across seeds")
    print(f"{'=' * 70}\n")

    for method in ["baseline", "transfer", "combined"]:
        avg = np.mean([r[method]["avg_error"] for r in all_results])
        std = np.std([r[method]["avg_error"] for r in all_results])
        print(f"  {method:12s}: avg ΔE = {avg:.2e} ± {std:.2e}")

    # Per-seed comparison
    print("\n  Per-seed avg ΔE:")
    print(f"  {'Seed':<6} {'Baseline':<12} {'Transfer':<12} {'Combined':<12} {'Winner'}")
    for r in all_results:
        b = r["baseline"]["avg_error"]
        t = r["transfer"]["avg_error"]
        c = r["combined"]["avg_error"]
        best = min(b, t, c)
        winner = "baseline" if best == b else "transfer" if best == t else "combined"
        print(f"  {r['seed']:<6} {b:<12.2e} {t:<12.2e} {c:<12.2e} {winner}")

    # Save
    result = SubExperimentResult(
        experiment_id="TL",
        technique=2,
        description="Transfer learning N=6→N=10",
        config={
            "seeds": seeds,
            "h_test": h_test_values,
            "pretrain_epochs": 3000,
            "finetune_epochs": 3000,
            "finetune_lr": 3e-4,
        },
        summary={
            "per_seed": all_results,
            "conclusion": "Transfer learning comparison",
        },
        success=True,
    )
    path = save_experiment_result(result, RESULTS_DIR, prefix="transfer_learning")
    print(f"\n  Result saved: {path.name}")


if __name__ == "__main__":
    run_experiment()
