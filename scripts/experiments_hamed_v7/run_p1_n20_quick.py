#!/usr/bin/env python
"""
Quick p=1 N=20 scaling test using StatevectorEstimator (not MPS).

N=20 with p=1 has only 2 parameters, so the optimization landscape is trivial.
We use StatevectorEstimator which handles N=20 (2^20 = 1M states) in memory.
This is faster than MPS for VQE because it avoids repeated circuit simulation.

Output: JSON with VQE results + MPNN pipeline test.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize
from torch_geometric.data import Data

from src.poc.v6.classical_solver import ClassicalSolver
from src.poc.v6.hamiltonian_builder import HamiltonianBuilder, make_lattice
from src.poc.v6.hva_builder import HVACircuitBuilder
from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset, train_mpnn


def approx_gap_1d_tfim(h, N, J=1.0):
    """Approximate finite-size gap for 1D TFIM.

    In thermodynamic limit: Δ = 2|J - h|.
    At finite N near criticality: Δ ~ 2π/N.
    """
    bulk_gap = 2 * abs(J - h)
    finite_size = 2 * np.pi / N
    return max(bulk_gap, finite_size)


def run_vqe_statevector(qc, hamiltonian, initial_guess, n_restarts=5, maxiter=500):
    """VQE using StatevectorEstimator (exact, fast for N<=20)."""
    estimator = StatevectorEstimator()

    def cost_fn(params):
        bound = qc.assign_parameters(params)
        job = estimator.run([(bound, hamiltonian)])
        return float(job.result()[0].data.evs)

    bounds = [(-np.pi, np.pi)] * len(initial_guess)

    best_energy = float("inf")
    best_params = initial_guess.copy()

    for restart in range(n_restarts + 1):
        x0 = (
            initial_guess
            if restart == 0
            else best_params + np.random.normal(0, 0.1, len(initial_guess))
        )
        x0 = np.clip(x0, -np.pi, np.pi)
        result = minimize(
            cost_fn,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-14},
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_params = result.x

    return best_params, best_energy


def main():
    print("=" * 60, flush=True)
    print("  p=1 Scaling Test: N=20 Full Pipeline", flush=True)
    print("=" * 60, flush=True)

    N = 20
    p = 1
    seeds = [42, 43, 44]
    # Training: only valid regime (h >= 2.0 expected for p=1 at N=20)
    h_train = np.array([1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0])
    h_test_values = [2.0, 2.5, 3.0]

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()
    base_lattice = make_lattice("chain_1d", N, J=1.0, h=1.0)
    qc, _ = hva.create(N, p, base_lattice)

    print(f"  N={N}, p={p}, params={qc.num_parameters}", flush=True)
    print(f"  h_train: {list(h_train)}", flush=True)
    print(f"  h_test: {h_test_values}", flush=True)

    all_results = []

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"\n  --- Seed {seed} ---", flush=True)

        # Phase 1: Ground truth (DMRG)
        print("  Phase 1: DMRG ground truth...", flush=True)
        t1 = time.time()
        exact_data = []
        for h in h_train:
            lat_h = make_lattice("chain_1d", N, J=1.0, h=h)
            H = builder.build(lat_h)
            exact_data.append(solver.solve(H, lat_h))
        phase1_time = time.time() - t1
        print(f"    Done in {phase1_time:.1f}s", flush=True)

        # Phase 2: VQE descending sweep (StatevectorEstimator)
        print("  Phase 2: VQE descending sweep...", flush=True)
        t2 = time.time()

        h_desc = np.sort(h_train)[::-1]
        theta_results_desc = []
        energies_desc = []
        prev_theta = np.random.uniform(-0.01, 0.01, qc.num_parameters)

        for h in h_desc:
            lat_h = make_lattice("chain_1d", N, J=1.0, h=h)
            H = builder.build(lat_h)
            theta_opt, energy = run_vqe_statevector(qc, H, prev_theta, n_restarts=5, maxiter=500)
            theta_results_desc.append(theta_opt)
            energies_desc.append(energy)
            prev_theta = theta_opt.copy()

            # Quick status
            idx = np.where(h_train == h)[0][0]
            exact_e = exact_data[idx].ground_energy
            de = abs(energy - exact_e)
            gap = approx_gap_1d_tfim(h, N)
            de_gap = de / gap
            status = "pass" if de_gap < 0.05 else "FAIL"
            print(f"    h={h:.2f}: dE/gap={de_gap:.4f} [{status}], theta={theta_opt}", flush=True)

        # Reverse to ascending
        theta_results = list(reversed(theta_results_desc))
        energies_vqe = list(reversed(energies_desc))
        phase2_time = time.time() - t2
        print(f"    Phase 2 time: {phase2_time:.1f}s", flush=True)

        # Compute ΔE/gap for all points
        de_gap_all = []
        for i, h in enumerate(h_train):
            exact_e = exact_data[i].ground_energy
            de = abs(energies_vqe[i] - exact_e)
            gap = approx_gap_1d_tfim(h, N)
            de_gap_all.append(de / gap)

        # Filter valid regime (ΔE/gap < 5%)
        valid_mask = np.array(de_gap_all) < 0.05
        h_valid = h_train[valid_mask]
        theta_valid = np.array(theta_results)[valid_mask]
        n_valid = len(h_valid)

        print(
            f"    Valid points: {n_valid}/{len(h_train)} (h >= {h_valid[0]:.2f} if any)", flush=True
        )

        if n_valid < 3:
            print("    WARNING: <3 valid points, expanding to 10%", flush=True)
            valid_mask = np.array(de_gap_all) < 0.10
            h_valid = h_train[valid_mask]
            theta_valid = np.array(theta_results)[valid_mask]
            n_valid = len(h_valid)

        # Phase 3: MPNN
        print("  Phase 3: MPNN training...", flush=True)
        t3 = time.time()

        exact_energies_valid = np.array(
            [exact_data[i].ground_energy for i, m in enumerate(valid_mask) if m]
        )

        dataset = build_graph_dataset(
            base_lattice,
            h_valid,
            theta_valid,
            exact_energies_valid,
            fidelities=np.ones(n_valid),
            fidelity_threshold=0.0,
        )

        model = MPNNPredictor(
            node_features=2,
            hidden_dim=128,
            n_layers=3,
            output_dim=qc.num_parameters,
        )
        train_result = train_mpnn(model, dataset, n_epochs=6000, lr=1e-3, patience=500)
        phase3_time = time.time() - t3
        print(
            f"    Done in {phase3_time:.1f}s, MSE={train_result['final_mse']:.2e}, "
            f"points={len(dataset)}",
            flush=True,
        )

        # Phase 4: Deploy
        print("  Phase 4: Deploy...", flush=True)
        model.eval()
        seed_results = []

        for h_test in h_test_values:
            lat_test = make_lattice("chain_1d", N, J=1.0, h=h_test)
            H_test = builder.build(lat_test)
            exact_test = solver.solve(H_test, lat_test)
            gap = approx_gap_1d_tfim(h_test, N)

            # MPNN prediction
            edge_idx, coord = builder.build_graph_data(base_lattice)
            x_test = torch.tensor(
                np.stack([np.full(N, h_test), coord.astype(float)], axis=1),
                dtype=torch.float32,
            )
            test_graph = Data(x=x_test, edge_index=torch.tensor(edge_idx, dtype=torch.long))
            with torch.no_grad():
                theta_pred = model(test_graph).numpy().flatten()

            # Evaluate
            estimator = StatevectorEstimator()
            bound = qc.assign_parameters(theta_pred)
            job = estimator.run([(bound, H_test)])
            energy_pred = float(job.result()[0].data.evs)

            delta_e = abs(energy_pred - exact_test.ground_energy)
            de_gap = delta_e / gap
            status = "PASS" if de_gap < 0.05 else "FAIL"

            print(
                f"    h={h_test:.1f}: E_pred={energy_pred:.6f}, E_exact={exact_test.ground_energy:.6f}, "
                f"dE/gap={de_gap:.4f} [{status}]",
                flush=True,
            )

            seed_results.append(
                {
                    "seed": seed,
                    "h_test": float(h_test),
                    "energy_pred": float(energy_pred),
                    "exact_energy": float(exact_test.ground_energy),
                    "delta_e": float(delta_e),
                    "delta_e_over_gap": float(de_gap),
                    "gap_approx": float(gap),
                    "theta_pred": theta_pred.tolist(),
                    "passes_5pct": de_gap < 0.05,
                }
            )

        all_results.append(
            {
                "seed": seed,
                "phase1_time": phase1_time,
                "phase2_time": phase2_time,
                "phase3_time": phase3_time,
                "mpnn_mse": train_result["final_mse"],
                "n_valid_points": n_valid,
                "valid_regime_start": float(h_valid[0]) if n_valid > 0 else None,
                "vqe_de_gap": [float(x) for x in de_gap_all],
                "deployments": seed_results,
            }
        )

    # Save results
    output = {
        "experiment": "p1_scaling_N20",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "N": N,
            "p": p,
            "h_train": list(h_train),
            "h_test": h_test_values,
            "seeds": seeds,
        },
        "results": all_results,
    }

    out_path = (
        Path("scripts/notebook_results")
        / f"p1_scaling_N20_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Summary
    print(f"\n{'=' * 60}", flush=True)
    print("  SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)

    for h_test in h_test_values:
        vals = [d for r in all_results for d in r["deployments"] if d["h_test"] == h_test]
        if vals:
            mean_dg = np.mean([v["delta_e_over_gap"] for v in vals])
            std_dg = np.std([v["delta_e_over_gap"] for v in vals])
            n_pass = sum(1 for v in vals if v["passes_5pct"])
            print(
                f"  h={h_test:.1f}: ΔE/gap = {mean_dg:.4f} ± {std_dg:.4f}, pass={n_pass}/{len(vals)}",
                flush=True,
            )

    print(f"\n  Results saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
