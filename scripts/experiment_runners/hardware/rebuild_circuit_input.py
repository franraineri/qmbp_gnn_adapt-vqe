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
    export HAIQU_API_KEY=...
    python scripts/experiment_runners/hardware/rebuild_circuit_input.py
    # writes ~/Downloads/input_tfim_heavy_hex_n10_p1.json and _n20_p1.json

    # offline (no Haiqu): pass explicit theta files or skip theta (zeros)
    python .../rebuild_circuit_input.py --no-fetch-theta
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from qiskit import qasm3

from qmbp_simulation import HVACircuitBuilder, HamiltonianBuilder, make_lattice

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


def hamiltonian_to_json(hamiltonian) -> list:
    """Serialize a SparsePauliOp-like Hamiltonian to [[pauli, {real, imaginary}], ...]."""
    out = []
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        c = complex(coeff)
        out.append([str(pauli), {"real": c.real, "imaginary": c.imag}])
    return out


def build_input_json(n_qubits: int, theta: list[float]) -> dict:
    """Reconstruct the bond-resolved HVA circuit + Hamiltonian and pack the JSON."""
    lattice = make_lattice(TOPOLOGY, n_qubits, J=J, h=H_VALUE)
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

    metadata = (
        f"GNN-HVA Framework | Model: tfim | Topology: {TOPOLOGY} | N={n_qubits} | "
        f"p={P_LAYERS} | h={H_VALUE} | J={J} | Ansatz: HVA bond-resolved "
        f"(lattice-aware, {len(theta)} params) | Initial state: |+>^N | "
        f"Optimizer: NFT (server-side, warm-started from GNN prediction) | "
        f"Hardware run: ibm_pittsburgh"
    )

    metadata_extended = {
        "framework": "GNN-HVA (Hybrid GNN-HVA for Topological Phase Characterization)",
        "model": "tfim",
        "model_description": "Transverse-Field Ising Model: H = -J\u00b7ZZ - h\u00b7X",
        "topology": TOPOLOGY,
        "n_qubits": n_qubits,
        "p_layers": P_LAYERS,
        "h_value": H_VALUE,
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
        "vqe_optimizer": "NFT (Nakanishi-Fujii-Todo), server-side via haiqu.variational_optimization",
        "warm_start_strategy": (
            "Trained GNN (UnifiedMPNN, cross-N) predicts theta(h). These parameters ARE "
            "the GNN prediction that was deployed directly on hardware."
        ),
        "model_kwargs": {},
        "n_lattice_edges": n_edges,
        "lattice_edges": [[int(i), int(jj)] for i, jj in lattice.edges],
        "coordination_numbers": coordination,
        # Provenance of this reconstruction (not in the n4 example; additive only).
        "hardware_execution": {
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
        },
    }

    return {
        "parameters": [float(x) for x in theta],
        "circuit_qasm": circuit_qasm,
        "hamiltonian": hamiltonian_to_json(hamiltonian),
        "metadata": metadata,
        "metadata_extended": metadata_extended,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "Downloads",
                    help="output directory (default: ~/Downloads)")
    ap.add_argument("--n", type=int, nargs="+", default=sorted(EXPERIMENTS),
                    help="system sizes to rebuild (default: 10 20)")
    ap.add_argument("--no-fetch-theta", action="store_true",
                    help="do not query Haiqu; use zeros for parameters (structure only)")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n in args.n:
        if args.no_fetch_theta:
            lattice = make_lattice(TOPOLOGY, n, J=J, h=H_VALUE)
            circ, _ = HVACircuitBuilder().create_bond_resolved(n, P_LAYERS, lattice)
            theta = [0.0] * circ.num_parameters
            print(f"N={n}: using zero theta ({len(theta)} params, structure only)")
        else:
            job_id = EXPERIMENTS.get(n)
            if job_id is None:
                print(f"N={n}: no known hardware job; skipping (or use --no-fetch-theta)")
                continue
            theta = fetch_gnn_theta(job_id)
            print(f"N={n}: recovered GNN theta ({len(theta)} params) from {job_id}")

        payload = build_input_json(n, theta)
        out_path = args.out_dir / f"input_tfim_{TOPOLOGY}_n{n}_p{P_LAYERS}.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"       wrote -> {out_path}  "
              f"(params={payload['metadata_extended']['n_parameters']}, "
              f"H_terms={len(payload['hamiltonian'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
