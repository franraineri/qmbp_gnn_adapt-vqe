#!/usr/bin/env python
"""Rebuild the exact HVA circuits run on hardware, export to input_*.json format.

Reconstructs each hardware-run circuit LOCALLY (deterministic pipeline) and
writes it in the same JSON schema as input_tfim_heavy_hex_n4_p1.json:
``parameters``, ``circuit_qasm`` (OpenQASM 3.0), ``hamiltonian`` (list of
[pauli, {real, imaginary}]), ``metadata`` (str), ``metadata_extended`` (dict).

The circuits are the BOND-RESOLVED HVA (one theta per lattice edge + one per
site) that actually ran on ibm_pittsburgh — verified to match the stored circuit
(N=10: 19 params, depth 9). The theta bound into ``parameters`` is the GNN
prediction recovered from each Haiqu job (the theta the deploy actually used).

Usage
-----
    # Recommended: predict theta locally from the best zoo GNN (offline, reproducible)
    python scripts/experiment_runners/hardware/rebuild_circuit_input.py --from-gnn --n 10
    python .../rebuild_circuit_input.py --from-gnn --n 10 --h 2.5   # pick the field

    # Recover the theta that ran on hardware (needs Haiqu)
    export HAIQU_API_KEY=...
    python .../rebuild_circuit_input.py --n 10 20

    # Structure only (zero theta, no Haiqu)
    python .../rebuild_circuit_input.py --no-fetch-theta
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from qiskit import qasm3

from qmbp_simulation import HamiltonianBuilder, HVACircuitBuilder, make_lattice

TOPOLOGY = "heavy_hex"
P_LAYERS = 1
J = 1.0
H_VALUE = 1.0

# Each hardware experiment: N -> the Haiqu job whose GNN-predicted theta was run.
EXPERIMENTS = {
    10: "jb-4ccab22f-89b5-4f5c-924c-d4b073b1211b",
    20: "jb-87d6c383-a6c6-4947-bb7b-cd54cb1cb26a",
}


def fetch_gnn_theta(job_id: str) -> list[float]:
    """Recover the GNN-predicted theta the single-shot job ran (job.parameters[0])."""
    from haiqu.sdk import haiqu as hq

    hq.login()
    job = hq.get_job(job_id)
    params = getattr(job, "parameters", None)
    if not params:
        raise RuntimeError(f"job {job_id} has no parameters to recover")
    return [float(x) for x in params[0]]


def predict_gnn_theta(n_qubits: int, h: float = H_VALUE) -> list[float]:
    """Predict theta locally with the best zoo GNN for this topology/p (no Haiqu).

    Loads the best model for heavy_hex p=1 from the zoo and runs a zero-shot
    prediction for the given h. Deterministic and offline — the reproducible
    counterpart to fetch_gnn_theta.
    """
    import torch

    from qmbp_simulation.predictors.model_zoo import load_best_model_for
    from qmbp_simulation.predictors.unified_graph import build_graph_for_model

    model, _, _ = load_best_model_for(TOPOLOGY, p_layers=P_LAYERS)
    lattice = make_lattice(TOPOLOGY, n_qubits, J=J, h=h)
    graph = build_graph_for_model(model, lattice, h, P_LAYERS)
    model.eval()
    with torch.no_grad():
        theta = model(graph).cpu().numpy().flatten()
    return [float(x) for x in np.clip(theta, -np.pi, np.pi)]


def hamiltonian_to_json(hamiltonian) -> list:
    """Serialize a SparsePauliOp-like Hamiltonian to [[pauli, {real, imaginary}], ...]."""
    out = []
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs, strict=False):
        c = complex(coeff)
        out.append([str(pauli), {"real": c.real, "imaginary": c.imag}])
    return out


def build_input_json(
    n_qubits: int,
    theta: list[float],
    h: float = H_VALUE,
    theta_source: str = "hardware_job",
) -> dict:
    """Reconstruct the bond-resolved HVA circuit + Hamiltonian and pack the JSON.

    theta_source: "hardware_job" (theta recovered from a QPU job) or
    "gnn_local" (theta predicted locally from the zoo GNN). Controls the
    provenance strings in metadata so the JSON never misrepresents its origin.
    """
    lattice = make_lattice(TOPOLOGY, n_qubits, J=J, h=h)
    circuit, _ = HVACircuitBuilder().create_bond_resolved(n_qubits, P_LAYERS, lattice)
    if circuit.num_parameters != len(theta):
        raise ValueError(
            f"theta size {len(theta)} != circuit params {circuit.num_parameters} "
            f"(N={n_qubits}); the recovered theta does not match a bond-resolved p=1 circuit."
        )
    hamiltonian = HamiltonianBuilder().build(lattice)
    n_edges = len(lattice.edges)

    # coordination number = degree of each site in the lattice graph
    coordination = [0] * n_qubits
    for i, jj in lattice.edges:
        coordination[i] += 1
        coordination[jj] += 1

    circuit_qasm = qasm3.dumps(circuit)

    if theta_source == "gnn_local":
        origin = "GNN prediction (zero-shot, best zoo model, predicted locally)"
        warm_start = (
            "Trained GNN (UnifiedMPNN, cross-N) predicts theta(h). These parameters "
            "are a fresh zero-shot GNN prediction — NOT run on hardware."
        )
    elif theta_source == "structure_only":
        origin = "zero theta (ansatz structure only, no prediction)"
        warm_start = "Parameters are zeros — this JSON captures the ansatz structure only."
    else:  # hardware_job
        origin = "GNN prediction recovered from the hardware job"
        warm_start = (
            "Trained GNN (UnifiedMPNN, cross-N) predicts theta(h). These parameters ARE "
            "the GNN prediction that was deployed directly on hardware."
        )

    metadata = (
        f"GNN-HVA Framework | Model: tfim | Topology: {TOPOLOGY} | N={n_qubits} | "
        f"p={P_LAYERS} | h={h} | J={J} | Ansatz: HVA bond-resolved "
        f"(lattice-aware, {len(theta)} params) | Initial state: |+>^N | "
        f"theta_source: {theta_source} ({origin})"
    )

    metadata_extended = {
        "framework": "GNN-HVA (Hybrid GNN-HVA for Topological Phase Characterization)",
        "model": "tfim",
        "model_description": "Transverse-Field Ising Model: H = -J\u00b7ZZ - h\u00b7X",
        "topology": TOPOLOGY,
        "n_qubits": n_qubits,
        "p_layers": P_LAYERS,
        "h_value": h,
        "J": J,
        "n_parameters": len(theta),
        "params_per_layer": n_edges + n_qubits,
        "initial_state": "plus",
        "ansatz_type": "HVA bond-resolved (Hamiltonian Variational Ansatz) \u2014 lattice-aware",
        "ansatz_structure": (
            "Per layer: RZZ(2*theta_zz_k) on each lattice edge k (independent per edge) "
            "+ RX(2*theta_x_i) on each qubit i (independent per site). Parameter order: "
            "[theta_zz_0..theta_zz_{E-1}, theta_x_0..theta_x_{N-1}]."
        ),
        "theta_source": theta_source,
        "warm_start_strategy": warm_start,
        "model_kwargs": {},
        "n_lattice_edges": n_edges,
        "lattice_edges": [[int(i), int(jj)] for i, jj in lattice.edges],
        "coordination_numbers": coordination,
    }

    if theta_source == "hardware_job":
        metadata_extended["vqe_optimizer"] = (
            "NFT (Nakanishi-Fujii-Todo), server-side via haiqu.variational_optimization"
        )
        # Provenance of the hardware run (only meaningful for recovered theta).
        metadata_extended["hardware_execution"] = {
            "device": "ibm_pittsburgh",
            "shots": 16384,
            "use_mitigation": True,
            "error_shield": {
                "noise_tailoring": True,
                "advanced_mitigation": True,
                "dynamical_decoupling": False,
                "readout_mitigation": False,
            },
            "use_compression": False,
        }

    return {
        "parameters": [float(x) for x in theta],
        "circuit_qasm": circuit_qasm,
        "hamiltonian": hamiltonian_to_json(hamiltonian),
        "metadata": metadata,
        "metadata_extended": metadata_extended,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--out-dir", type=Path, default=Path.home() / "Downloads", help="output directory (default: ~/Downloads)"
    )
    ap.add_argument(
        "--n", type=int, nargs="+", default=sorted(EXPERIMENTS), help="system sizes to rebuild (default: 10 20)"
    )
    ap.add_argument(
        "--from-gnn",
        action="store_true",
        help="predict theta locally from the best zoo GNN (offline, reproducible; no Haiqu). Recommended default.",
    )
    ap.add_argument(
        "--h",
        type=float,
        default=H_VALUE,
        help=f"transverse field for GNN prediction (default: {H_VALUE}); only used with --from-gnn",
    )
    ap.add_argument(
        "--no-fetch-theta", action="store_true", help="do not query Haiqu; use zeros for parameters (structure only)"
    )
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n in args.n:
        theta_source = "hardware_job"
        h = H_VALUE
        if args.from_gnn:
            h = args.h
            theta = predict_gnn_theta(n, h)
            theta_source = "gnn_local"
            print(f"N={n}: predicted GNN theta ({len(theta)} params) locally at h={h}")
        elif args.no_fetch_theta:
            lattice = make_lattice(TOPOLOGY, n, J=J, h=H_VALUE)
            circ, _ = HVACircuitBuilder().create_bond_resolved(n, P_LAYERS, lattice)
            theta = [0.0] * circ.num_parameters
            theta_source = "structure_only"
            print(f"N={n}: using zero theta ({len(theta)} params, structure only)")
        else:
            job_id = EXPERIMENTS.get(n)
            if job_id is None:
                print(f"N={n}: no known hardware job; skipping (or use --from-gnn / --no-fetch-theta)")
                continue
            theta = fetch_gnn_theta(job_id)
            print(f"N={n}: recovered GNN theta ({len(theta)} params) from {job_id}")

        payload = build_input_json(n, theta, h=h, theta_source=theta_source)
        out_path = args.out_dir / f"input_tfim_{TOPOLOGY}_n{n}_p{P_LAYERS}.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(
            f"       wrote -> {out_path}  "
            f"(params={payload['metadata_extended']['n_parameters']}, "
            f"H_terms={len(payload['hamiltonian'])})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
