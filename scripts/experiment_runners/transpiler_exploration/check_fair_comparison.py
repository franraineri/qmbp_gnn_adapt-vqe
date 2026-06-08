"""Fair comparison: same CES-selected layout, compare circuit representations.

Uses select_layouts_low_ces to get the SAME layout as production,
then compares Original HVA vs PauliEvolutionGate representation.
"""

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeTorino

from qmbp_simulation import HVACircuitBuilder, make_lattice
from qmbp_simulation.execution.noisy_utils import (
    build_adjacency,
    compute_circuit_ces,
    find_layouts_bfs,
    select_layouts_low_ces,
)
from qmbp_simulation.models.model_registry import get_model_spec

# ─── Setup ───────────────────────────────────────────────────────────────
lattice = make_lattice("heavy_hex", 10, J=1.0, h=3.25)
spec = get_model_spec("tfim")
H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)

# ─── Build original HVA ──────────────────────────────────────────────────
hva_builder = HVACircuitBuilder()
qc_orig, theta_orig = hva_builder.create(10, 1, lattice)
rng = np.random.default_rng(42)
theta_vals = rng.uniform(-0.5, 0.5, size=len(theta_orig))
bound_orig = qc_orig.assign_parameters(dict(zip(theta_orig, theta_vals, strict=False)))

# ─── Build PauliEvolution HVA ────────────────────────────────────────────
zz_labels, zz_coeffs = [], []
x_labels, x_coeffs = [], []
for label, coeff in zip(H.paulis.to_labels(), H.coeffs, strict=False):
    n_z = sum(1 for c in label if c == "Z")
    n_x = sum(1 for c in label if c == "X")
    if n_z >= 2:
        zz_labels.append(label)
        zz_coeffs.append(float(np.real(coeff)))
    elif n_x >= 1:
        x_labels.append(label)
        x_coeffs.append(float(np.real(coeff)))

H_zz = SparsePauliOp.from_list(list(zip(zz_labels, zz_coeffs, strict=False)))
H_x = SparsePauliOp.from_list(list(zip(x_labels, x_coeffs, strict=False)))

theta_pe = ParameterVector("t", 2)
qc_pauli = QuantumCircuit(10)
qc_pauli.h(range(10))
qc_pauli.append(PauliEvolutionGate(H_zz, time=theta_pe[0]), range(10))
qc_pauli.append(PauliEvolutionGate(H_x, time=theta_pe[1]), range(10))
bound_pauli = qc_pauli.assign_parameters({theta_pe[0]: 0.3, theta_pe[1]: -0.5})

# ─── Get production layout (CES-selected) ───────────────────────────────
backend = FakeTorino()
adj = build_adjacency(backend)
candidates = find_layouts_bfs(adj, 10, n_candidates=40, seed=42)

# Use the SAME layout selection as production (optimization_level=2, max_ces=0.5)
layout_sel = select_layouts_low_ces(
    bound_orig,
    backend,
    candidates,
    n_select=1,
    optimization_level=2,
    max_ces=0.5,
)
layout = layout_sel.layouts[0]
ces_orig = layout_sel.ces_values[0]
n_2q_orig = sum(
    1 for inst in layout_sel.transpiled_circuits[0].data if inst.operation.num_qubits == 2
)
depth_2q_orig = layout_sel.transpiled_circuits[0].depth(lambda x: x.operation.num_qubits == 2)
total_depth_orig = layout_sel.transpiled_circuits[0].depth()

print(f"Production layout: {layout}")
print(f"Production CES: {ces_orig:.4f}")
print(
    f"Production (Orig HVA + level 2): total_depth={total_depth_orig}, "
    f"2Q_depth={depth_2q_orig}, n_2Q={n_2q_orig}"
)

# ─── Now transpile PauliEvolution HVA with SAME layout ───────────────────
print("\n--- Comparing with SAME layout ---")

pm_pauli_l2 = generate_preset_pass_manager(
    optimization_level=2, backend=backend, initial_layout=layout
)
tc_pauli = pm_pauli_l2.run(bound_pauli)
depth_2q_pauli = tc_pauli.depth(lambda x: x.operation.num_qubits == 2)
n_2q_pauli = sum(1 for inst in tc_pauli.data if inst.operation.num_qubits == 2)
ces_pauli, _ = compute_circuit_ces(tc_pauli, backend)

print(
    f"PauliEvol HVA + level 2:         total_depth={tc_pauli.depth()}, "
    f"2Q_depth={depth_2q_pauli}, n_2Q={n_2q_pauli}, CES={ces_pauli:.4f}"
)

# Also try level 3 with PauliEvolution on same layout
pm_pauli_l3 = generate_preset_pass_manager(
    optimization_level=3, backend=backend, initial_layout=layout
)
tc_pauli_l3 = pm_pauli_l3.run(bound_pauli)
depth_2q_pauli_l3 = tc_pauli_l3.depth(lambda x: x.operation.num_qubits == 2)
n_2q_pauli_l3 = sum(1 for inst in tc_pauli_l3.data if inst.operation.num_qubits == 2)

print(
    f"PauliEvol HVA + level 3:         total_depth={tc_pauli_l3.depth()}, "
    f"2Q_depth={depth_2q_pauli_l3}, n_2Q={n_2q_pauli_l3}"
)

# ─── Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY (same CES-optimized layout)")
print("=" * 65)
print(f"{'Method':<35} {'Depth':<7} {'2Q_D':<6} {'n_2Q':<6} {'CES'}")
print(f"{'-' * 65}")
print(
    f"{'Orig HVA + SABRE lvl2 (CURRENT)':<35} {total_depth_orig:<7} {depth_2q_orig:<6} {n_2q_orig:<6} {ces_orig:.4f}"
)
print(
    f"{'PauliEvol HVA + SABRE lvl2':<35} {tc_pauli.depth():<7} {depth_2q_pauli:<6} {n_2q_pauli:<6} {ces_pauli:.4f}"
)
print(
    f"{'PauliEvol HVA + SABRE lvl3':<35} {tc_pauli_l3.depth():<7} {depth_2q_pauli_l3:<6} {n_2q_pauli_l3:<6} —"
)

print("\n--- Improvement ---")
d2q_improve = (depth_2q_orig - depth_2q_pauli) / depth_2q_orig * 100
n2q_improve = (n_2q_orig - n_2q_pauli) / n_2q_orig * 100
print(f"2Q-depth: {depth_2q_orig} → {depth_2q_pauli} ({d2q_improve:+.1f}%)")
print(f"n_2Q:    {n_2q_orig} → {n_2q_pauli} ({n2q_improve:+.1f}%)")
print(
    f"CES:     {ces_orig:.4f} → {ces_pauli:.4f} ({'same' if abs(ces_orig - ces_pauli) < 0.001 else 'different'})"
)
