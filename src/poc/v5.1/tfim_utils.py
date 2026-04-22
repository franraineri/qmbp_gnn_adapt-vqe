"""Shared utilities for PoC V5.1 — TFIM 1D HVA pipeline."""
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit, ParameterVector


def build_tfim_hamiltonian(N, J, h):
    """H = -J Σ ZᵢZᵢ₊₁ - h Σ Xᵢ"""
    terms = [("ZZ", [i, i + 1], -J) for i in range(N - 1)] + [("X", [i], -h) for i in range(N)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=N)


def create_hva_circuit(N, p):
    """HVA: e^{-iθ_x H_X} · e^{-iθ_zz H_ZZ} per layer on |+⟩^N. Factor 2θ for RZZ/RX."""
    qc = QuantumCircuit(N)
    qc.h(range(N))
    theta = ParameterVector('θ', 2 * p)
    for layer in range(p):
        for i in range(N - 1):
            qc.rzz(2 * theta[layer * 2], i, i + 1)
        for i in range(N):
            qc.rx(2 * theta[layer * 2 + 1], i)
    return qc, theta


def build_local_observables(N):
    """Local observables: ⟨Xᵢ⟩ per site, ⟨ZᵢZᵢ₊₁⟩ per bond."""
    ops_X = [SparsePauliOp.from_sparse_list([("X", [i], 1.0)], num_qubits=N) for i in range(N)]
    ops_ZZ = [SparsePauliOp.from_sparse_list([("ZZ", [i, i+1], 1.0)], num_qubits=N) for i in range(N-1)]
    return ops_X, ops_ZZ
