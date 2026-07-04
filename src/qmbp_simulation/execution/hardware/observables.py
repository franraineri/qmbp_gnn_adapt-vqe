"""Construction and extraction of per-site observables.

Pure functions for building SparsePauliOp observables and extracting
expectation values from EstimatorV2 array results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qiskit.quantum_info import SparsePauliOp

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def build_per_site_observables(
    n_qubits: int,
    edges: list[tuple[int, int]],
) -> tuple[list[SparsePauliOp], list[SparsePauliOp]]:
    """Build per-site X_i and per-bond Z_iZ_j observables.

    Pure function — no backend or circuit required.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the system.
    edges : list[tuple[int, int]]
        List of lattice bonds (i, j).

    Returns
    -------
    tuple[list[SparsePauliOp], list[SparsePauliOp]]
        (x_ops, zz_ops) — lists of single-term operators.

    Raises
    ------
    ValueError
        If n_qubits < 2 or edges reference qubits outside [0, n_qubits).
    """
    print(f"  [DEBUG] build_per_site_observables: n_qubits={n_qubits}, n_edges={len(edges)}")
    if n_qubits < 2:
        raise ValueError(f"n_qubits must be >= 2, got {n_qubits}")
    # Validate edges reference valid qubit indices
    for i, j in edges:
        if not (0 <= i < n_qubits and 0 <= j < n_qubits):
            raise ValueError(
                f"Edge ({i}, {j}) references qubit outside [0, {n_qubits}). Check lattice topology."
            )
        if i == j:
            raise ValueError(f"Self-loop edge ({i}, {j}) is invalid for ZZ observable.")
    x_ops = [
        SparsePauliOp.from_sparse_list([("X", [i], 1.0)], num_qubits=n_qubits)
        for i in range(n_qubits)
    ]
    zz_ops = [
        SparsePauliOp.from_sparse_list([("ZZ", [i, j], 1.0)], num_qubits=n_qubits) for i, j in edges
    ]
    return x_ops, zz_ops


def map_observables_to_layout(
    observables: list[SparsePauliOp],
    isa_circuit: QuantumCircuit,
) -> list[SparsePauliOp]:
    """Apply transpiled circuit layout to each observable.

    Parameters
    ----------
    observables : list[SparsePauliOp]
        Observables in logical qubit space.
    isa_circuit : QuantumCircuit
        Transpiled circuit (has .layout attribute).

    Returns
    -------
    list[SparsePauliOp]
        Observables mapped to physical qubit layout.
    """
    print(f"  [DEBUG] map_observables_to_layout: mapping {len(observables)} observables")
    if not hasattr(isa_circuit, "layout") or isa_circuit.layout is None:
        raise ValueError(
            "isa_circuit has no layout attribute. Ensure the circuit was transpiled "
            "with a pass manager that sets the layout (e.g., generate_preset_pass_manager)."
        )
    mapped = [op.apply_layout(isa_circuit.layout) for op in observables]
    # Post-mapping validation: all mapped observables must have same qubit count as circuit
    for i, m_op in enumerate(mapped):
        if m_op.num_qubits != isa_circuit.num_qubits:
            raise RuntimeError(
                f"Observable mapping error: mapped_obs[{i}] has "
                f"{m_op.num_qubits} qubits but ISA circuit has "
                f"{isa_circuit.num_qubits}. Layout mismatch."
            )
    return mapped


def extract_array_result(result: object, n_x: int, n_zz: int) -> tuple[list[float], list[float]]:
    """Extract per-site values from an EstimatorV2 array result.

    When submitting (circuit, [obs_list]), the result contains an array
    of expectation values. This function splits X_i and Z_iZ_j values.

    Parameters
    ----------
    result : PubResult
        Result from estimator.run([(circuit, [obs_list])]).result().
    n_x : int
        Number of X_i observables (= n_qubits).
    n_zz : int
        Number of Z_iZ_j observables (= n_edges).

    Returns
    -------
    tuple[list[float], list[float]]
        (x_values, zz_values)
    """
    print(f"  [DEBUG] extract_array_result: extracting {n_x} X + {n_zz} ZZ values")
    evs = result[0].data.evs  # type: ignore[index]
    # Handle scalar evs (0-d array from single observable)
    evs = np.atleast_1d(evs)
    # Validate dimension matches expected observable count
    expected_n = n_x + n_zz
    if len(evs) < expected_n:
        raise RuntimeError(
            f"EstimatorV2 returned {len(evs)} values but expected {expected_n} "
            f"({n_x} X + {n_zz} ZZ). Possible observable grouping mismatch."
        )
    x_values = [float(evs[i]) for i in range(n_x)]
    zz_values = [float(evs[n_x + i]) for i in range(n_zz)]
    return x_values, zz_values
