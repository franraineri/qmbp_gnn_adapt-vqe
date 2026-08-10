#!/usr/bin/env python3
"""Print + save transpiled circuit for ladder N=10 h=3.5 using MPNN prediction."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from qmbp_simulation import make_lattice, HVACircuitBuilder
from qmbp_simulation.predictors.model_zoo import load_pretrained
from qmbp_simulation.predictors.unified_graph import build_unified_bond_resolved_graph
from qmbp_simulation.analysis import (
    circuit_summary, print_circuit_comparison, save_circuit_diagram,
)

# Load model + predict
model, meta = load_pretrained(
    model="tfim_bond_resolved", topology="ladder", n_qubits=10, p_layers=1
)
model.eval()
print(f"Model: {meta.checkpoint_file} (pass={meta.pass_rate:.0%})")

N, h = 10, 3.5
lattice = make_lattice("ladder", N, J=1.0, h=h)
circuit, _ = HVACircuitBuilder().create_bond_resolved(N, 1, lattice)

g = build_unified_bond_resolved_graph(
    lattice, h_value=h, p_layers=1, include_circuit_nodes=True
)
with torch.no_grad():
    theta = model(g).numpy().flatten()
theta = np.clip(theta, -np.pi, np.pi)

print(f"Predicted theta: {len(theta)} params, range [{theta.min():.4f}, {theta.max():.4f}]")

# Bind parameters
bound = circuit.assign_parameters(theta)

# Transpile for IBM Heron (FakeTorino)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeTorino

backend = FakeTorino()
pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
transpiled = pm.run(bound)

# Print comparison
print_circuit_comparison(bound, transpiled, params=theta)

# Save PNG
out_path = ROOT / "results" / "ladder_N10_h3.5_transpiled.png"
save_circuit_diagram(transpiled, str(out_path))
print(f"\nPNG saved: {out_path}")

# Also save logical circuit
out_logical = ROOT / "results" / "ladder_N10_h3.5_logical.png"
save_circuit_diagram(bound, str(out_logical))
print(f"Logical PNG: {out_logical}")
