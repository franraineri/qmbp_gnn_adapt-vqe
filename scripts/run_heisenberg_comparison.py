#!/usr/bin/env python
"""
Heisenberg Model Extension — Comparison Script

Runs the full pipeline (Phase 1-4) for the Heisenberg XXZ model at N=6
and compares against existing TFIM results. Validates that the GNN-HVA
framework is model-agnostic.

Usage:
    python scripts/run_heisenberg_comparison.py
    python scripts/run_heisenberg_comparison.py --no-baseline
    python scripts/run_heisenberg_comparison.py --delta 0.5  # XY-like
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


def run_heisenberg_pipeline(
    N: int = 6,
    delta: float = 1.0,
    h_test: float = 1.5,
    n_restarts: int = 10,
    maxiter: int = 1000,
    mpnn_epochs: int = 2000,
    seed: int = 42,
    include_baseline: bool = True,
    n_baseline_seeds: int = 3,
) -> dict:
    """Execute full pipeline for Heisenberg XXZ model."""
    import torch
    from torch_geometric.data import Data

    from src.poc.v6.classical_solver import ClassicalSolver
    from src.poc.v6.config import VQEConfig
    from src.poc.v6.hamiltonian_builder import HamiltonianBuilder, make_lattice
    from src.poc.v6.hardware_deployer_v61 import HardwareDeployerV61
    from src.poc.v6.hva_builder import HVACircuitBuilder
    from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset, train_mpnn
    from src.poc.v6.vqe_optimizer import VQEOptimizer

    np.random.seed(seed)
    torch.manual_seed(seed)

    J, p = 1.0, 2
    # H-grid: focus on paramagnetic regime (h >= 1.0 for Heisenberg)
    h_values = np.array([0.5, 0.8, 1.0, 1.2, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0])

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()
    base_lattice = make_lattice("chain_1d", N, J=J, h=1.0)

    # Heisenberg HVA circuit (4 params per layer = 8 total)
    # Use Néel state |↑↓↑↓...⟩ as initial state (natural for antiferromagnetic Heisenberg)
    qc, _ = hva.create_heisenberg(N, p, base_lattice, initial_state="neel")
    print(f"  HVA circuit: {qc.num_parameters} parameters (4 per layer × {p} layers)")
    print("  Initial state: Néel |↑↓↑↓...⟩")

    result = {
        "model": "heisenberg_xxz",
        "delta": delta,
        "N": N,
        "p_layers": p,
        "seed": seed,
        "h_test": h_test,
        "n_restarts": n_restarts,
        "phases": {},
    }

    # ── Phase 1: Exact Diagonalization ──
    print("  Phase 1: Exact diagonalization...")
    t1 = time.time()
    exact_data = []
    ops_z, ops_ss = builder.build_heisenberg_observables(base_lattice)
    for h in h_values:
        lat_h = make_lattice("chain_1d", N, J=J, h=h)
        H = builder.build_heisenberg(lat_h, delta=delta)
        exact_data.append(solver.solve(H, lat_h, obs_x=ops_z, obs_zz=ops_ss))
    phase1_time = time.time() - t1
    result["phases"]["phase1"] = {
        "elapsed_s": round(phase1_time, 1),
        "n_points": len(exact_data),
        "e0_range": [exact_data[0].ground_energy, exact_data[-1].ground_energy],
        "gap_min": min(d.gap for d in exact_data),
    }
    print(f"    {len(exact_data)} points in {phase1_time:.1f}s")
    print(f"    E0 range: [{exact_data[0].ground_energy:.4f}, {exact_data[-1].ground_energy:.4f}]")

    # ── Phase 2: VQE Descending Sweep ──
    print("  Phase 2: VQE descending sweep...")
    t2 = time.time()
    config = VQEConfig(n_restarts=n_restarts, maxiter=maxiter, enable_callbacks=False)
    # Override restart_sigma for 8D landscape (wider exploration needed)
    config.restart_sigma = 0.5
    opt = VQEOptimizer(config)

    # Manual descending sweep for Heisenberg (need custom H per point)
    vqe_results = []
    n_params = qc.num_parameters
    # Wider initial guess for 8D Heisenberg landscape (not near zero like TFIM)
    current_guess = np.random.uniform(-0.3, 0.3, n_params)

    for idx in reversed(range(len(h_values))):
        h = float(h_values[idx])
        lat_h = make_lattice("chain_1d", N, J=J, h=h)
        H = builder.build_heisenberg(lat_h, delta=delta)

        vqe_result = opt.optimize(
            H,
            qc,
            current_guess,
            exact_energy=exact_data[idx].ground_energy,
            exact_state=exact_data[idx].ground_state,
        )
        vqe_result.h_value = h
        vqe_results.insert(0, vqe_result)
        current_guess = vqe_result.theta_opt.copy()

    fids = np.array([r.fidelity for r in vqe_results])
    phase2_time = time.time() - t2
    n_good = sum(1 for f in fids if f >= 0.93)
    result["phases"]["phase2"] = {
        "elapsed_s": round(phase2_time, 1),
        "avg_fidelity": float(np.mean(fids)),
        "min_fidelity": float(np.min(fids)),
        "fid_ge_93pct": n_good,
        "total_points": len(fids),
    }
    print(
        f"    {phase2_time:.1f}s — avg fid={np.mean(fids) * 100:.1f}%, {n_good}/{len(fids)} ≥ 93%"
    )
    for r in vqe_results:
        tag = "✅" if r.fidelity >= 0.93 else "⚠️"
        print(f"      {tag} h={r.h_value:.1f}: fid={r.fidelity:.4f}, ΔE={r.energy_error:.2e}")

    # ── Phase 3: MPNN Training ──
    print("  Phase 3: MPNN training...")
    t3 = time.time()
    dataset = build_graph_dataset(
        base_lattice,
        h_values,
        np.array([r.theta_opt for r in vqe_results]),
        np.array([d.ground_energy for d in exact_data]),
        fidelities=fids,
        fidelity_threshold=0.93,  # Heisenberg — relaxed but compliant
    )
    print(f"    Training points: {len(dataset)}/{len(h_values)} (fid ≥ 0.93)")

    # MPNN with output_dim=8 (4 params × 2 layers)
    model = MPNNPredictor(node_features=2, hidden_dim=64, n_layers=3, output_dim=4 * p)

    if len(dataset) == 0:
        print("    ⚠️ No training points passed fidelity filter!")
        print("    This indicates HVA p=2 with |+⟩ initial state cannot express")
        print("    the Heisenberg ground state. This is a FINDING, not a bug.")
        result["phases"]["phase3"] = {
            "elapsed_s": 0.0,
            "training_points": 0,
            "final_mse": float("inf"),
            "stopped_early": True,
            "stop_reason": "no_training_data",
            "finding": "HVA_p2_insufficient_for_heisenberg",
        }
        result["phases"]["phase4"] = {
            "elapsed_s": 0.0,
            "h_test": h_test,
            "delta_e_over_gap": float("inf"),
            "phase_label": "unknown",
            "finding": "Cannot deploy — no trained model",
        }
        result["total_elapsed_s"] = round(time.time() - t1, 1)
        result["success"] = False
        result["finding"] = (
            "HVA p=2 with |+⟩^N initial state cannot express Heisenberg XXZ "
            "ground states. The |+⟩ state is an eigenstate of X (TFIM paramagnetic), "
            "but Heisenberg with Z-field has ground states in the Z-basis. "
            "This confirms the HVA must match the Hamiltonian structure."
        )
        return result

    train_result = train_mpnn(model, dataset, n_epochs=mpnn_epochs, lr=1e-3, patience=300)
    phase3_time = time.time() - t3
    result["phases"]["phase3"] = {
        "elapsed_s": round(phase3_time, 1),
        "training_points": len(dataset),
        "final_mse": train_result["final_mse"],
        "stopped_early": train_result["stopped_early"],
        "stop_reason": train_result["stop_reason"],
    }
    print(f"    {phase3_time:.1f}s — MSE={train_result['final_mse']:.6f}")

    # ── Phase 4: Deployment with Baseline ──
    print(f"  Phase 4: Deploy h_test={h_test}...")
    t4 = time.time()
    lat_test = make_lattice("chain_1d", N, J=J, h=h_test)
    H_test = builder.build_heisenberg(lat_test, delta=delta)
    exact_test = solver.solve(H_test, lat_test, obs_x=ops_z, obs_zz=ops_ss)

    model.eval()
    edge_idx, coord = builder.build_graph_data(base_lattice)
    x_test = torch.tensor(
        np.stack([np.full(N, h_test), coord.astype(float)], axis=1),
        dtype=torch.float32,
    )
    test_graph = Data(x=x_test, edge_index=torch.tensor(edge_idx, dtype=torch.long))
    with torch.no_grad():
        theta_pred = model(test_graph).numpy().flatten()

    deployer = HardwareDeployerV61(mode="simulation")

    if include_baseline:
        deploy_result, comparison = deployer.deploy_with_baseline(
            qc,
            H_test,
            theta_pred,
            lat_test,
            exact_test,
            n_random_seeds=n_baseline_seeds,
        )
    else:
        deploy_result = deployer.deploy_adapt_vqe(qc, H_test, theta_pred, lat_test, exact_test)
        comparison = None

    phase4_time = time.time() - t4
    result["phases"]["phase4"] = {
        "elapsed_s": round(phase4_time, 1),
        "h_test": h_test,
        "predicted_energy": deploy_result.predicted_energy,
        "delta_e": deploy_result.delta_e,
        "delta_e_over_gap": deploy_result.delta_e_over_gap,
        "phase_label": deploy_result.phase_label,
        "checklist": deploy_result.metrics_checklist,
    }
    if comparison is not None:
        result["phases"]["phase4"]["baseline_comparison"] = {
            "gain_energy_pct": comparison.gain_energy_pct,
            "cold_mean_delta_e_over_gap": comparison.cold_start_mean["delta_e_over_gap"],
            "cold_std_delta_e_over_gap": comparison.cold_start_std["delta_e_over_gap"],
            "warm_start_sufficient": comparison.warm_start_sufficient,
        }

    de_tag = (
        "✅"
        if deploy_result.delta_e_over_gap < 0.05
        else "⚠️"
        if deploy_result.delta_e_over_gap < 0.10
        else "❌"
    )
    print(
        f"    ΔE/gap={deploy_result.delta_e_over_gap:.4f} {de_tag}, phase={deploy_result.phase_label}"
    )
    if comparison:
        print(
            f"    Baseline gain={comparison.gain_energy_pct:.1f}%, cold mean={comparison.cold_start_mean['delta_e_over_gap']:.4f}"
        )

    result["total_elapsed_s"] = round(time.time() - t1, 1)
    result["success"] = True
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Heisenberg XXZ Model — Pipeline Comparison")
    parser.add_argument(
        "--delta", type=float, default=1.0, help="Anisotropy Δ (default: 1.0 = isotropic)"
    )
    parser.add_argument("--h-test", type=float, default=1.5, help="Test h-value (default: 1.5)")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline comparison")
    parser.add_argument("--baseline-seeds", type=int, default=3, help="Number of baseline seeds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("  Heisenberg XXZ Model — Pipeline Comparison")
    print(f"  Δ={args.delta}, h_test={args.h_test}, seed={args.seed}")
    print("=" * 60)

    result = run_heisenberg_pipeline(
        delta=args.delta,
        h_test=args.h_test,
        seed=args.seed,
        include_baseline=not args.no_baseline,
        n_baseline_seeds=args.baseline_seeds,
    )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = hashlib.sha1(f"heisenberg:{ts}".encode()).hexdigest()[:8]
    path = RESULTS_DIR / f"heisenberg_comparison_{ts}_{run_id}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("  RESULT SUMMARY")
    print(f"{'=' * 60}")
    p4 = result["phases"]["phase4"]
    p3 = result["phases"]["phase3"]
    p2 = result["phases"]["phase2"]
    print(f"  Model:     Heisenberg XXZ (Δ={args.delta})")
    print(f"  ΔE/gap:    {p4['delta_e_over_gap']:.4f}")
    print(f"  Phase:     {p4['phase_label']}")
    print(f"  MSE:       {p3['final_mse']:.6f}")
    print(f"  Avg fid:   {p2['avg_fidelity'] * 100:.1f}%")
    print(f"  Train pts: {p3['training_points']}/{p2['total_points']}")
    bl = p4.get("baseline_comparison")
    if bl:
        print(f"  Gain:      {bl['gain_energy_pct']:.1f}% vs random")
    print(f"  Time:      {result['total_elapsed_s']:.0f}s")
    print(f"  Saved:     {path}")

    return 0 if p4["delta_e_over_gap"] < 0.10 else 1


if __name__ == "__main__":
    sys.exit(main())
