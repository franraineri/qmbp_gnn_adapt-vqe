#!/usr/bin/env python3
"""Print the EXACT circuit used in Tier 0 QPU execution (2026-06-14).

Loads the same MPNN checkpoint, generates the same theta_pred for h=4.0,
and displays the bound HVA circuit that was submitted to ibm_kingston.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qmbp_simulation import HamiltonianBuilder, HVACircuitBuilder, make_lattice
from qmbp_simulation.analysis import circuit_summary, print_circuit
from qmbp_simulation.predictors import load_mpnn_checkpoint

# ── Load the EXACT MPNN checkpoint used in Tier 0 ────────────────────────
ckpt_path = ROOT / "results" / "hardware" / "mpnn_checkpoints" / "mpnn_heavy_hex_n10_p1_seed42.pt"
model = load_mpnn_checkpoint(ckpt_path)
model.eval()

# ── Generate the EXACT theta_pred for h=4.0 (same code path as deployment) ─
builder = HamiltonianBuilder()
lattice_h = make_lattice("heavy_hex", 10, J=1.0, h=4.0)
edge_index_np, coord = builder.build_graph_data(lattice_h)
h_feat = np.full(10, 4.0)
x = torch.tensor(np.stack([h_feat, coord.astype(float)], axis=1), dtype=torch.float32)
edge_index = torch.tensor(edge_index_np, dtype=torch.long)
graph = Data(x=x, edge_index=edge_index)
with torch.no_grad():
    theta_pred = model(graph).numpy().flatten()

print("MPNN-predicted parameters for Tier 0 (h=4.0, heavy_hex, N=10, p=1):")
print(f"  theta_zz = {theta_pred[0]:.6f}")
print(f"  theta_x  = {theta_pred[1]:.6f}")
print(f"  ||theta|| = {np.linalg.norm(theta_pred):.6f}")
print()

# ── Build and display the EXACT circuit ──────────────────────────────────
lattice = make_lattice("heavy_hex", 10, J=1.0, h=4.0)
circuit, _ = HVACircuitBuilder().create(10, 1, lattice)

print_circuit(
    circuit,
    params=theta_pred,
    title="Tier 0 Circuit (ibm_kingston, 2026-06-14) - EXACT params from MPNN",
)

print()
info = circuit_summary(circuit, theta_pred)
print("Circuit properties:")
print(f"  Qubits:      {info['n_qubits']}")
print(f"  Depth:       {info['depth']}")
print(f"  2Q gates:    {info['n_2q_gates']} (all RZZ)")
print(f"  1Q gates:    {info['n_1q_gates']} (10 H + 10 RX)")
print(f"  Parameters:  {info['n_parameters']} (theta_zz, theta_x)")
print(f"  Gate types:  {info['gate_counts']}")
print()
print("Note: On ibm_kingston (Heron r2), RZZ is a NATIVE gate.")
print("      The transpiled circuit has 9 native RZZ pulses (no CX decomposition).")
