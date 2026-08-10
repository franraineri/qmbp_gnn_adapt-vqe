"""VQEzy MPNN Robustness Validation — Seed Stability, LOO, h-Extrapolation.

Three additional validation experiments using existing training data and
the VQEzy benchmark infrastructure. No new VQE runs needed — only
MPNN retraining and evaluation on cached data.

Usage:
    python scripts/analysis/validate_vqezy_robustness.py

Output:
    results/benchmarks/vqezy_robustness_validation.json
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qmbp_simulation.models.data_models import LatticeConfig
from qmbp_simulation.predictors import (
    MPNNPredictor,
    build_graph_dataset,
    load_mpnn_checkpoint,
    save_mpnn_checkpoint,
    train_mpnn,
)
from qmbp_simulation.predictors.external_benchmarks.vqezy_loader import (
    _build_rectangular_edges,
    load_vqezy_tfi,
)
from qmbp_simulation.execution import NoiselessBackend
from qmbp_simulation import HamiltonianBuilder, HVACircuitBuilder

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Shared infrastructure ──

def build_lattice_and_circuit():
    """Build the 4x2 square lattice and HVA circuit (reused across all tests)."""
    edges = _build_rectangular_edges(4, 2)
    coord = np.zeros(8)
    for i, j in edges:
        coord[i] += 1
        coord[j] += 1
    lattice = LatticeConfig(
        topology="square", n_qubits=8, J=1.0, h=5.0,
        edges=edges, coordination_numbers=coord, periodic=False,
    )
    circuit_builder = HVACircuitBuilder()
    circuit, _ = circuit_builder.create(8, 1, lattice)
    return lattice, circuit, edges, coord


def load_training_data():
    """Load cached A1 training data."""
    path = PROJECT_ROOT / "results" / "benchmarks" / "vqezy_mpnn_a1a2_results.json"
    with open(path) as f:
        data = json.load(f)
    h_train = np.array(data["a1"]["h_train"])
    theta_opt = np.array(data["a1"]["theta_opt"])
    e_exact = np.array(data["a1"]["e_exact"])
    fidelities = np.array(data["a1"]["fidelities"])
    return h_train, theta_opt, e_exact, fidelities


def evaluate_mpnn_on_vqezy(model, lattice, circuit, dataset_path, h_min, h_max,
                            j_min, j_max, rescale=True, max_instances=None):
    """Evaluate a trained MPNN on VQEzy instances with h/j rescaling."""
    dataset = load_vqezy_tfi(
        dataset_path, h_min=h_min, h_max=h_max,
        j_min=j_min, j_max=j_max, max_instances=max_instances,
    )
    if len(dataset) == 0:
        return {"n": 0, "mean_de_gap": float("inf"), "pass_rate": 0.0}

    builder = HamiltonianBuilder()
    backend = NoiselessBackend()
    edge_index_np, coord_arr = builder.build_graph_data(lattice)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    from qmbp_simulation import ClassicalSolver
    solver = ClassicalSolver()

    de_gaps = []
    model.eval()
    with torch.no_grad():
        for inst in dataset.instances:
            # Build graph with rescaled h
            h_input = inst.h / inst.j if (rescale and inst.j > 0) else inst.h
            h_feat = np.full(8, h_input)
            x = torch.tensor(
                np.stack([h_feat, coord_arr.astype(float)], axis=1),
                dtype=torch.float32,
            )
            from torch_geometric.data import Data
            data = Data(x=x, edge_index=edge_index)
            data.batch = torch.zeros(8, dtype=torch.long)

            theta_pred = model(data).numpy().flatten()

            # Evaluate energy
            lat_h = LatticeConfig(
                topology="square", n_qubits=8, J=inst.j, h=inst.h,
                edges=lattice.edges, coordination_numbers=lattice.coordination_numbers,
                periodic=False,
            )
            H = builder.build(lat_h)
            gt = solver.solve(H, lat_h)
            e_pred = backend.evaluate(circuit, H, theta_pred)
            gap = gt.gap if gt.gap > 1e-10 else 0.1
            de_gap = abs(e_pred - gt.ground_energy) / gap
            de_gaps.append(de_gap)

    de_gaps = np.array(de_gaps)
    return {
        "n": len(de_gaps),
        "mean_de_gap": float(np.mean(de_gaps)),
        "median_de_gap": float(np.median(de_gaps)),
        "pass_rate": float(np.sum(de_gaps < 0.05) / len(de_gaps)),
        "std_de_gap": float(np.std(de_gaps)),
    }


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Seed Stability
# ══════════════════════════════════════════════════════════════════════════

def run_seed_stability(lattice, circuit, vqezy_path):
    """Train MPNN with 5 different seeds, measure PassRate variance."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Seed Stability")
    print("=" * 60)

    h_train, theta_opt, e_exact, fidelities = load_training_data()
    mask = fidelities >= 0.93
    h_filt = h_train[mask]
    theta_filt = theta_opt[mask]
    e_filt = e_exact[mask]
    fid_filt = fidelities[mask]

    dataset = build_graph_dataset(
        lattice=lattice, h_values=h_filt, theta_opt=theta_filt,
        e_exact=e_filt, fidelities=fid_filt, fidelity_threshold=0.90,
    )

    seeds = [42, 123, 7, 2024, 9999]
    results_per_seed = []

    for seed in seeds:
        model, history = train_mpnn(
            None, dataset, hidden_dim=64, n_layers=3, n_epochs=300,
            norm_type="none", seed=seed,
        )
        final_mse = history["final_mse"]

        # Evaluate on VQEzy (paramagnetic regime)
        metrics = evaluate_mpnn_on_vqezy(
            model, lattice, circuit, vqezy_path,
            h_min=2.5, h_max=5.0, j_min=0.5, j_max=1.5,
            rescale=True, max_instances=50,
        )

        results_per_seed.append({
            "seed": seed,
            "final_mse": float(final_mse),
            "pass_rate": metrics["pass_rate"],
            "mean_de_gap": metrics["mean_de_gap"],
            "n_evaluated": metrics["n"],
        })
        print(f"  seed={seed}: MSE={final_mse:.6f}, PassRate={metrics['pass_rate']:.1%}, "
              f"mean_ΔE/gap={metrics['mean_de_gap']:.4f}")

    # Compute stability metrics
    pass_rates = [r["pass_rate"] for r in results_per_seed]
    mses = [r["final_mse"] for r in results_per_seed]
    print(f"\n  PassRate across seeds: mean={np.mean(pass_rates):.1%} ± {np.std(pass_rates):.1%}")
    print(f"  MSE across seeds: mean={np.mean(mses):.6f} ± {np.std(mses):.6f}")
    print(f"  Max PassRate variation: {max(pass_rates) - min(pass_rates):.1%}")

    return {
        "per_seed": results_per_seed,
        "pass_rate_mean": float(np.mean(pass_rates)),
        "pass_rate_std": float(np.std(pass_rates)),
        "pass_rate_range": float(max(pass_rates) - min(pass_rates)),
        "mse_mean": float(np.mean(mses)),
        "mse_std": float(np.std(mses)),
    }


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Leave-One-Out Cross-Validation
# ══════════════════════════════════════════════════════════════════════════

def run_loo_cv(lattice, circuit):
    """Train on 6 points, predict the 7th (rotate). Measures overfitting."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Leave-One-Out Cross-Validation")
    print("=" * 60)

    h_train, theta_opt, e_exact, fidelities = load_training_data()
    mask = fidelities >= 0.93
    h_filt = h_train[mask]
    theta_filt = theta_opt[mask]
    e_filt = e_exact[mask]
    fid_filt = fidelities[mask]
    n_points = len(h_filt)

    builder = HamiltonianBuilder()
    backend = NoiselessBackend()
    edge_index_np, coord_arr = builder.build_graph_data(lattice)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    loo_results = []

    for leave_out_idx in range(n_points):
        # Build train set (all except one)
        train_mask = np.ones(n_points, dtype=bool)
        train_mask[leave_out_idx] = False

        train_dataset = build_graph_dataset(
            lattice=lattice,
            h_values=h_filt[train_mask],
            theta_opt=theta_filt[train_mask],
            e_exact=e_filt[train_mask],
            fidelities=fid_filt[train_mask],
            fidelity_threshold=0.5,
        )

        # Train model
        model, history = train_mpnn(
            None, train_dataset, hidden_dim=64, n_layers=3,
            n_epochs=300, norm_type="none", seed=42,
        )

        # Predict on left-out point
        h_test = float(h_filt[leave_out_idx])
        theta_true = theta_filt[leave_out_idx]

        model.eval()
        with torch.no_grad():
            h_feat = np.full(8, h_test)
            x = torch.tensor(
                np.stack([h_feat, coord_arr.astype(float)], axis=1),
                dtype=torch.float32,
            )
            from torch_geometric.data import Data
            data = Data(x=x, edge_index=edge_index)
            data.batch = torch.zeros(8, dtype=torch.long)
            theta_pred = model(data).numpy().flatten()

        # Compute metrics
        theta_mse = float(np.mean((theta_pred - theta_true) ** 2))

        # Energy evaluation
        lat_h = LatticeConfig(
            topology="square", n_qubits=8, J=1.0, h=h_test,
            edges=lattice.edges, coordination_numbers=lattice.coordination_numbers,
            periodic=False,
        )
        H = builder.build(lat_h)
        from qmbp_simulation import ClassicalSolver
        solver = ClassicalSolver()
        gt = solver.solve(H, lat_h)
        e_pred = backend.evaluate(circuit, H, theta_pred)
        gap = gt.gap if gt.gap > 1e-10 else 0.1
        de_gap = abs(e_pred - gt.ground_energy) / gap

        loo_results.append({
            "h_left_out": h_test,
            "theta_mse": theta_mse,
            "de_gap": float(de_gap),
            "theta_pred": theta_pred.tolist(),
            "theta_true": theta_true.tolist(),
        })
        print(f"  Leave out h={h_test:.1f}: θ_MSE={theta_mse:.6f}, ΔE/gap={de_gap:.4f}")

    mean_mse = np.mean([r["theta_mse"] for r in loo_results])
    mean_de_gap = np.mean([r["de_gap"] for r in loo_results])
    print(f"\n  LOO mean θ-MSE: {mean_mse:.6f}")
    print(f"  LOO mean ΔE/gap: {mean_de_gap:.4f}")
    print(f"  LOO worst point: h={max(loo_results, key=lambda x: x['de_gap'])['h_left_out']:.1f}")

    return {
        "per_fold": loo_results,
        "mean_theta_mse": float(mean_mse),
        "mean_de_gap": float(mean_de_gap),
        "max_de_gap": float(max(r["de_gap"] for r in loo_results)),
        "n_folds": n_points,
    }


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: h-Extrapolation (out-of-training-range)
# ══════════════════════════════════════════════════════════════════════════

def run_h_extrapolation(lattice, circuit, vqezy_path):
    """Test MPNN on h/j values OUTSIDE training range [1.5, 5.0].

    VQEzy has instances with h/j up to ~50 (j=0.1, h=5.0).
    Our MPNN was trained on h/j = h (with J=1) ∈ [2.0, 5.0].
    Test on h/j ∈ [5, 10] and [10, 50] to see extrapolation quality.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: h-Extrapolation (out of training range)")
    print("=" * 60)

    # Load the baseline checkpoint (seed=42)
    ckpt_path = PROJECT_ROOT / "results" / "benchmarks" / "mpnn_vqezy_square_n8_p1.pt"
    model = load_mpnn_checkpoint(str(ckpt_path))

    # Evaluate on different h/j bins
    # Training range: h ∈ [2.0, 5.0] with J=1.0 → h/j ∈ [2.0, 5.0]
    # We test:
    #   - In-range: h/j ∈ [2.5, 5.0] (with rescaling)
    #   - Mild extrapolation: h/j ∈ [5.0, 10.0] (slightly beyond training)
    #   - Strong extrapolation: h/j ∈ [10.0, 50.0] (far beyond training)

    bins = [
        ("In-range [2.5, 5.0]", 2.5, 5.0, 0.1, 2.0),
        ("Mild extrap [5.0, 10.0]", 2.0, 5.0, 0.1, 0.5),  # small j → high h/j
        ("Strong extrap [10.0, 50.0]", 2.0, 5.0, 0.1, 0.2),  # very small j
    ]

    results = []
    for label, h_min, h_max, j_min, j_max in bins:
        metrics = evaluate_mpnn_on_vqezy(
            model, lattice, circuit, vqezy_path,
            h_min=h_min, h_max=h_max, j_min=j_min, j_max=j_max,
            rescale=True, max_instances=100,
        )
        # Filter: only keep instances where actual h/j is in the target bin
        print(f"  {label}: n={metrics['n']}, mean_ΔE/gap={metrics['mean_de_gap']:.4f}, "
              f"PassRate={metrics['pass_rate']:.1%}")
        results.append({"bin": label, **metrics})

    return {"bins": results}


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("VQEzy MPNN Robustness Validation")
    print("Reusing: A1 training data + VQEzy dataset + benchmark infra")
    print()

    t_start = time.perf_counter()

    lattice, circuit, edges, coord = build_lattice_and_circuit()
    vqezy_path = str(PROJECT_ROOT / "data" / "VQEzy" / "qmanybody" / "ti_8_qubit.h5")

    # Run all 3 experiments
    results = {}

    results["seed_stability"] = run_seed_stability(lattice, circuit, vqezy_path)
    results["loo_cv"] = run_loo_cv(lattice, circuit)
    results["h_extrapolation"] = run_h_extrapolation(lattice, circuit, vqezy_path)

    elapsed = time.perf_counter() - t_start

    # ── Summary ──
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f}s")

    ss = results["seed_stability"]
    print(f"\n1. Seed Stability:")
    print(f"   PassRate: {ss['pass_rate_mean']:.1%} ± {ss['pass_rate_std']:.1%} "
          f"(range: {ss['pass_rate_range']:.1%})")
    stable = ss["pass_rate_range"] < 0.10
    print(f"   Verdict: {'✅ STABLE' if stable else '⚠️ UNSTABLE'} "
          f"(range < 10%: {stable})")

    loo = results["loo_cv"]
    print(f"\n2. Leave-One-Out CV:")
    print(f"   Mean θ-MSE: {loo['mean_theta_mse']:.6f}")
    print(f"   Mean ΔE/gap: {loo['mean_de_gap']:.4f}")
    print(f"   Max ΔE/gap: {loo['max_de_gap']:.4f}")
    not_overfit = loo["mean_de_gap"] < 0.10
    print(f"   Verdict: {'✅ NO OVERFITTING' if not_overfit else '⚠️ POSSIBLE OVERFITTING'}")

    ext = results["h_extrapolation"]
    print(f"\n3. h-Extrapolation:")
    for b in ext["bins"]:
        print(f"   {b['bin']}: PassRate={b['pass_rate']:.1%}, mean={b['mean_de_gap']:.4f}")

    # Save all results
    results["elapsed_s"] = elapsed
    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    output_path = PROJECT_ROOT / "results" / "benchmarks" / "vqezy_robustness_validation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
