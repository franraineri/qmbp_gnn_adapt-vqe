#!/usr/bin/env python3
"""Quick end-to-end test of the new PEA configuration in fake_backend mode."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qmbp_simulation import ClassicalSolver, HamiltonianBuilder, HVACircuitBuilder, make_lattice
from qmbp_simulation.execution.backends import MitigationOptions
from qmbp_simulation.execution.hardware import HardwareBackend, HardwareConfig

# Build config with new PEA settings (fake_backend mode — no real QPU)
config = HardwareConfig(
    mode="fake_backend",
    n_qubits=6,
    shots=1024,
    n_layouts=2,
    n_candidates=10,
    mitigation=MitigationOptions(
        zne_enabled=True,
        zne_amplifier="pea",
        zne_noise_factors=[1, 1.5, 2, 3],
        num_randomizations=64,
        shots_per_randomization=256,
        dd_enabled=True,
        trex_enabled=True,
        twirling_enabled=True,
    ),
)
backend = HardwareBackend(config=config)

# Build circuit + Hamiltonian
lattice = make_lattice("chain_1d", 6, J=1.0, h=4.0)
H = HamiltonianBuilder().build(lattice)
circuit, _ = HVACircuitBuilder().create(6, 1, lattice)
solver = ClassicalSolver()
exact = solver.solve(H, lattice)

# Run deployment (fake_backend — local simulation, no QPU used)
params = np.array([0.1, 0.2])
result = backend.run_deployment(
    circuit,
    H,
    params,
    h_value=4.0,
    e_exact=exact.ground_energy,
    gap=exact.gap,
    expected_label="paramagnetic",
)

print(f"E_ZNE: {result.e_zne:.4f}")
print(f"E_exact: {result.e_exact:.4f}")
print(f"Delta_E/gap: {result.delta_e_gap:.4f}")
print(f"Verdict: {result.verdict}")
print(f"ZNE amplifier: {result.zne_amplifier_used}")
print(f"Mitigation: {result.mitigation_strategy}")
print()
print("✅ Full pipeline works with new PEA config in fake_backend mode")
