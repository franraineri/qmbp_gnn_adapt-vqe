#!/usr/bin/env python3
"""Quick Noisy Comparison — DMRG(1D) vs NoisyBackend(θ_pred) vs Exact.

Lightweight comparison using NoisyBackend (Gaussian noise, instant) instead of
FakeTorino (full noise model, hours). Gives a fast quantitative estimate of
whether QPU can beat DMRG on 2D topologies.

This is NOT a full deployment rehearsal — it uses Gaussian noise approximation.
For real hardware validation, use run_parametric_deployment.py --mode hardware.

Usage:
    python scripts/analysis/quick_noisy_comparison.py \
        --n-qubits 10 --topology heavy_hex --h-values 1.5 2.0 3.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qmbp_simulation import ClassicalSolver, HVACircuitBuilder, make_lattice
from qmbp_simulation.execution import NoiselessBackend, NoisyBackend
from qmbp_simulation.execution.mps_backend import MPSBackend
from qmbp_simulation.models.data_models import VQEConfig
from qmbp_simulation.models.model_registry import get_model_spec
from qmbp_simulation.optimizers.vqe import VQEOptimizer
from qmbp_simulation.utils.helpers import json_dump


def run_comparison(
    n_qubits: int = 10,
    topology: str = "heavy_hex",
    h_values: list[float] | None = None,
    h_train: list[float] | None = None,
    p_layers: int = 1,
    model: str = "tfim",
    noise_sigma: float = 0.01,
    seed: int = 42,
) -> dict:
    """Run the quick 3-way comparison: exact vs DMRG(1D) vs noisy(θ_pred)."""

    if h_values is None:
        h_values = [1.5, 2.0, 3.0]
    if h_train is None:
        h_train = [4.0, 3.5, 2.5, 1.75, 1.25]

    spec = get_model_spec(model)
    solver = ClassicalSolver()
    noiseless = NoiselessBackend()

    print(f"\n{'='*65}")
    print(f"  Quick Noisy Comparison: {topology} N={n_qubits} p={p_layers}")
    print(f"  h_test: {h_values}")
    print(f"  Noise σ: {noise_sigma}")
    print(f"{'='*65}")

    # Step 1: Ground truth + DMRG
    print("\n  [1] Ground truth (exact diag + DMRG)...")
    gt_data = {}
    for h in sorted(set(h_values + h_train), reverse=True):
        lattice = make_lattice(topology, n_qubits, J=1.0, h=h)
        H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
        gt_exact = solver.solve(H, lattice, method="exact")
        gt_dmrg = solver.solve(H, lattice, method="dmrg")
        gt_data[h] = {
            "e_exact": gt_exact.ground_energy,
            "e_dmrg": gt_dmrg.ground_energy,
            "gap": gt_exact.gap,
            "dmrg_error": abs(gt_exact.ground_energy - gt_dmrg.ground_energy),
        }

    # Step 2: VQE training (noiseless, MPS chi=64)
    print("  [2] VQE training (noiseless, MPS chi=64)...")
    backend_train = MPSBackend(strategy="aer_mps", chi_max=64, seed=seed)
    lattice_ref = make_lattice(topology, n_qubits, J=1.0, h=h_train[0])
    circuit, _ = spec.create_circuit(n_qubits, p_layers, lattice_ref, **spec.circuit_kwargs)
    n_params = circuit.num_parameters

    vqe_config = VQEConfig(p_layers=p_layers, n_restarts=2, maxiter=300, method="L-BFGS-B")
    optimizer = VQEOptimizer(config=vqe_config, backend=backend_train, seed=seed)

    rng = np.random.default_rng(seed)
    prev_theta = rng.uniform(-0.01, 0.01, n_params)
    vqe_thetas = {}

    for h in sorted(h_train, reverse=True):
        lattice_h = make_lattice(topology, n_qubits, J=1.0, h=h)
        H = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)
        result = optimizer.optimize(H, circuit, prev_theta, exact_energy=gt_data[h]["e_exact"])
        prev_theta = result.theta_opt.copy()
        vqe_thetas[h] = result.theta_opt

    # Step 3: MPNN training
    print("  [3] MPNN training...")
    from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn

    dataset = build_graph_dataset(
        lattice=lattice_ref,
        h_values=np.array(sorted(h_train, reverse=True)),
        theta_opt=np.array([vqe_thetas[h] for h in sorted(h_train, reverse=True)]),
        e_exact=np.array([gt_data[h]["e_exact"] for h in sorted(h_train, reverse=True)]),
        fidelities=None, fidelity_threshold=0.0,
    )
    mpnn = MPNNPredictor(node_features=2, hidden_dim=64, n_layers=3, output_dim=n_params)
    train_mpnn(mpnn, dataset, n_epochs=1500, lr=1e-3, patience=300, seed=seed)

    # Step 4: Predict theta for test h-values
    print("  [4] MPNN prediction + evaluation...")
    import torch
    from torch_geometric.data import Data
    from qmbp_simulation import HamiltonianBuilder

    builder = HamiltonianBuilder()
    mpnn.eval()
    theta_pred = {}
    for h in h_values:
        lattice_h = make_lattice(topology, n_qubits, J=1.0, h=h)
        edge_index_np, coord = builder.build_graph_data(lattice_h)
        h_feat = np.full(n_qubits, float(h))
        x = torch.tensor(np.stack([h_feat, coord.astype(float)], axis=1), dtype=torch.float32)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index)
        with torch.no_grad():
            theta_pred[h] = mpnn(graph).numpy().flatten()

    # Step 5: Evaluate at each h — noiseless + noisy
    # NoisyBackend uses Gaussian noise: σ ≈ 1/√shots
    # shots=100 gives σ≈0.1, shots=10000 gives σ≈0.01
    effective_shots = max(1, int(1.0 / noise_sigma**2))
    noisy = NoisyBackend(shots=effective_shots, seed_simulator=seed)

    results_table = []
    print(f"\n  {'h':>6} {'E_exact':>12} {'E_DMRG':>12} {'E_noiseless':>12} "
          f"{'E_noisy':>12} {'ΔE/gap_DMRG':>12} {'ΔE/gap_QPU':>12} {'QPU>DMRG':>9}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*9}")

    for h in h_values:
        lattice_h = make_lattice(topology, n_qubits, J=1.0, h=h)
        H = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)
        theta = theta_pred[h]
        e_exact = gt_data[h]["e_exact"]
        e_dmrg = gt_data[h]["e_dmrg"]
        gap = gt_data[h]["gap"]

        # Noiseless evaluation (θ_pred quality)
        e_noiseless = noiseless.evaluate(circuit, H, theta)

        # Noisy evaluation (simulates QPU noise)
        e_noisy = noisy.evaluate(circuit, H, theta)

        de_gap_dmrg = abs(e_dmrg - e_exact) / max(gap, 1e-10)
        de_gap_noiseless = abs(e_noiseless - e_exact) / max(gap, 1e-10)
        de_gap_noisy = abs(e_noisy - e_exact) / max(gap, 1e-10)
        qpu_beats = abs(e_noisy - e_exact) < abs(e_dmrg - e_exact)

        row = {
            "h": h,
            "e_exact": e_exact,
            "e_dmrg": e_dmrg,
            "e_noiseless": e_noiseless,
            "e_noisy": e_noisy,
            "gap": gap,
            "de_gap_dmrg": de_gap_dmrg,
            "de_gap_noiseless": de_gap_noiseless,
            "de_gap_noisy": de_gap_noisy,
            "qpu_beats_dmrg": qpu_beats,
        }
        results_table.append(row)

        beats_str = "✅" if qpu_beats else "❌"
        print(f"  {h:>6.3f} {e_exact:>12.6f} {e_dmrg:>12.6f} {e_noiseless:>12.6f} "
              f"{e_noisy:>12.6f} {de_gap_dmrg:>12.5f} {de_gap_noisy:>12.5f} {beats_str:>9}")

    # Summary
    n_beats = sum(1 for r in results_table if r["qpu_beats_dmrg"])
    print(f"\n  QPU beats DMRG: {n_beats}/{len(results_table)} h-points")
    print(f"  (Using Gaussian noise σ={noise_sigma} as QPU proxy)")

    return {
        "config": {
            "n_qubits": n_qubits, "topology": topology,
            "h_values": h_values, "h_train": h_train,
            "noise_sigma": noise_sigma, "seed": seed,
        },
        "results": results_table,
        "n_qpu_beats_dmrg": n_beats,
        "n_total": len(results_table),
    }


def main():
    parser = argparse.ArgumentParser(description="Quick Noisy Comparison")
    parser.add_argument("--n-qubits", type=int, default=10)
    parser.add_argument("--topology", type=str, default="heavy_hex")
    parser.add_argument("--h-values", type=float, nargs="+", default=[1.5, 2.0, 3.0])
    parser.add_argument("--h-train", type=float, nargs="+", default=[4.0, 3.5, 2.5, 1.75, 1.25])
    parser.add_argument("--noise-sigma", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    data = run_comparison(
        n_qubits=args.n_qubits, topology=args.topology,
        h_values=args.h_values, h_train=args.h_train,
        noise_sigma=args.noise_sigma, seed=args.seed,
    )

    if args.output:
        json_dump(data, Path(args.output))
        print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
