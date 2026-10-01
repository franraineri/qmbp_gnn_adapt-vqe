#!/usr/bin/env python
"""Standalone rebuilder for the bond-resolved HVA circuit (Qiskit-only).

No project dependency: this recreates the SAME circuit as the exporter using
only `qiskit` and `numpy`. Run it to get the Qiskit QuantumCircuit and (if
angles are provided) the bound circuit ready to simulate.

    python recreate.py            # build parametric + bound, print summary
    python recreate.py --draw     # also print an ASCII drawing

Requires: qiskit>=1.0, numpy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# ── Fixed topology (heavy_hex N=10) ─────────────────────────────────
N_QUBITS = 10
P_LAYERS = 1
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 7), (3, 8), (5, 9)]  # 9 bonds
N_EDGES = len(EDGES)
N_PARAMS = (N_EDGES + N_QUBITS) * P_LAYERS  # = 19


def build_parametric() -> tuple[QuantumCircuit, ParameterVector]:
    """Build the bond-resolved HVA circuit with symbolic parameters.

    Layout per layer: [theta_zz_0..theta_zz_{E-1}, theta_x_0..theta_x_{N-1}].
    Gates: RZZ(2*theta_zz_k) per edge, RX(2*theta_x_i) per qubit.
    """
    theta = ParameterVector("theta", N_PARAMS)
    qc = QuantumCircuit(N_QUBITS)
    qc.h(range(N_QUBITS))  # |+>^N initial state
    params_per_layer = N_EDGES + N_QUBITS
    for layer in range(P_LAYERS):
        off = layer * params_per_layer
        for k, (i, j) in enumerate(EDGES):
            qc.rzz(2 * theta[off + k], i, j)
        for i in range(N_QUBITS):
            qc.rx(2 * theta[off + N_EDGES + i], i)
    return qc, theta


def build_bound(angles: np.ndarray) -> QuantumCircuit:
    """Build the circuit with concrete angles assigned."""
    qc, theta = build_parametric()
    angles = np.asarray(angles, dtype=float).flatten()
    if angles.shape[0] != N_PARAMS:
        raise ValueError(f"expected {N_PARAMS} angles, got {angles.shape[0]}")
    return qc.assign_parameters({theta: angles})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", action="store_true", help="print ASCII drawing")
    args = ap.parse_args()

    qc, _ = build_parametric()
    print(f"Parametric circuit: {qc.num_qubits} qubits, "
          f"{qc.num_parameters} params, depth {qc.depth()}")
    print("Gate counts:", dict(qc.count_ops()))

    theta_path = Path(__file__).with_name("theta_gnn.json")
    if theta_path.exists():
        data = json.loads(theta_path.read_text())
        angles = np.array(data["theta"], dtype=float)
        bound = build_bound(angles)
        print(f"\nBound circuit built with {len(angles)} GNN angles.")
        if args.draw:
            print(bound.draw(output="text", fold=100))
        # Optional: evaluate energy if qiskit has the estimator + you build H
    else:
        print("\n(theta_gnn.json not found — parametric circuit only)")
        if args.draw:
            print(qc.draw(output="text", fold=100))


if __name__ == "__main__":
    main()
