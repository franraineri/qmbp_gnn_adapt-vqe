#!/usr/bin/env python3
"""Generate pre-computed sample data for demo notebooks.

Usage:
    python notebooks/data/generate_samples.py

Produces:
    - notebooks/data/sample_results.json (noisy samples for GNN-QEM demo)
    - notebooks/data/pretrained_mpnn_tfim_chain.pt (MPNN checkpoint)

Requires the full qmbp_simulation package to be installed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Ensure package is importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def generate_demo_samples(output_path: str = "notebooks/data/sample_results.json") -> None:
    """Generate synthetic noisy VQE samples for the GNN-QEM demo notebook."""
    from qmbp_simulation import HamiltonianBuilder, ClassicalSolver, make_lattice

    solver = ClassicalSolver()
    builder = HamiltonianBuilder()
    rng = np.random.default_rng(42)

    N = 10
    topology = "heavy_hex"
    h_values = np.linspace(3.5, 1.5, 12)

    samples = []
    for h in h_values:
        lattice = make_lattice(topology, N, J=1.0, h=float(h))
        H = builder.build(lattice)
        gt = solver.solve(H, lattice)

        # Realistic noise model: systematic upward bias + fluctuation
        noise_bias = 0.02 * N * (1 + 0.5 / max(float(h) - 1.0, 0.1))
        noise_fluct = float(rng.normal(0, 0.01 * N))
        e_noisy = gt.ground_energy + noise_bias + noise_fluct

        samples.append({
            "h": round(float(h), 2),
            "exact_energy": round(float(gt.ground_energy), 4),
            "noisy_energy": round(float(e_noisy), 4),
            "gap": round(float(gt.gap), 4),
            "n_qubits": N,
            "topology": topology,
        })

    data = {
        "description": "Pre-computed noisy VQE samples for GNN-QEM demo (Notebook 03)",
        "topology": topology,
        "N": N,
        "p": 1,
        "noise_model": "synthetic_gaussian_bias",
        "generated_by": "notebooks/data/generate_samples.py",
        "samples": samples,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Generated {len(samples)} samples → {output}")


def generate_mpnn_checkpoint(output_path: str = "notebooks/data/pretrained_mpnn_tfim_chain.pt") -> None:
    """Run minimal pipeline and save MPNN checkpoint for Notebook 01."""
    from qmbp_simulation import (
        ClassicalSolver, HamiltonianBuilder, VQEConfig, VQEOptimizer, make_lattice,
    )
    from qmbp_simulation.execution import NoiselessBackend
    from qmbp_simulation.models.model_registry import get_model_spec
    from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn, save_mpnn_checkpoint

    N, P = 6, 2
    TOPOLOGY = "chain_1d"
    MODEL = "tfim_longitudinal"
    spec = get_model_spec(MODEL)

    h_train = np.linspace(3.5, 1.0, 20)

    # Phase 1
    solver = ClassicalSolver()
    builder = HamiltonianBuilder()
    exact_results = []
    for h in h_train:
        lattice_h = make_lattice(TOPOLOGY, N, J=1.0, h=float(h))
        H = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)
        exact_results.append(solver.solve(H, lattice_h))

    e_exact = np.array([r.ground_energy for r in exact_results])

    # Phase 2
    lattice_ref = make_lattice(TOPOLOGY, N, J=1.0, h=float(h_train[0]))
    circuit, _ = spec.create_circuit(N, P, lattice_ref, **spec.circuit_kwargs)
    vqe_config = VQEConfig(method="L-BFGS-B", maxiter=500, n_restarts=3, restart_sigma=0.1)
    optimizer = VQEOptimizer(config=vqe_config, backend=NoiselessBackend(), seed=42)
    vqe_results = optimizer.descending_sweep(
        h_values=h_train, circuit=circuit, lattice=lattice_ref, exact_data=exact_results,
    )
    theta_opt = np.array([r.theta_opt for r in vqe_results])
    fidelities = np.array([r.fidelity for r in vqe_results])

    # Phase 3
    dataset = build_graph_dataset(
        lattice=lattice_ref, h_values=h_train, theta_opt=theta_opt,
        e_exact=e_exact, fidelities=fidelities, fidelity_threshold=0.0,
    )
    n_params = spec.total_params_for_p(P)
    model = MPNNPredictor(
        node_features=dataset[0].x.shape[1], hidden_dim=64, n_layers=3, output_dim=n_params,
    )
    train_mpnn(model=model, dataset=dataset, n_epochs=3000, lr=1e-3, patience=200, seed=42)

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_mpnn_checkpoint(model, output, metadata={
        "model": MODEL, "topology": TOPOLOGY, "N": N, "p": P,
        "n_train": len(h_train), "h_range": [float(h_train[-1]), float(h_train[0])],
    })
    print(f"✅ MPNN checkpoint saved → {output}")


if __name__ == "__main__":
    print("Generating notebook demo data...\n")
    generate_demo_samples()
    print()
    generate_mpnn_checkpoint()
    print("\n🎉 All demo data generated successfully!")
