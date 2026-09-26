#!/usr/bin/env python3
"""Verify HVA circuit parameter periodicities numerically.

Determines which θ transforms produce the SAME quantum state/energy,
to define the correct canonicalization for multi-seed consistency.
"""

import numpy as np
from qiskit.quantum_info import Statevector

from qmbp_simulation import HamiltonianBuilder, HVACircuitBuilder, make_lattice

hva = HVACircuitBuilder()
builder = HamiltonianBuilder()
lattice = make_lattice("chain_1d", 6, J=1.0, h=3.0)
H = builder.build(lattice)
qc, _ = hva.create(6, 1, lattice)

# Reference θ
theta_ref = np.array([0.07, 0.39270])

# Equivalent θ values (test all symmetries)
equivalents = [
    ("Reference", theta_ref),
    ("Z2 flip: -θ", -theta_ref),
    ("θ_zz + π", np.array([theta_ref[0] + np.pi, theta_ref[1]])),
    ("θ_zz - π", np.array([theta_ref[0] - np.pi, theta_ref[1]])),
    ("θ_x + π", np.array([theta_ref[0], theta_ref[1] + np.pi])),
    ("θ_x - π", np.array([theta_ref[0], theta_ref[1] - np.pi])),
    ("both + π", theta_ref + np.pi),
    ("both - π", theta_ref - np.pi),
    ("θ_zz+π, θ_x-π", np.array([theta_ref[0] + np.pi, theta_ref[1] - np.pi])),
    ("θ_zz-π, θ_x+π", np.array([theta_ref[0] - np.pi, theta_ref[1] + np.pi])),
    ("θ_zz + 2π", np.array([theta_ref[0] + 2 * np.pi, theta_ref[1]])),
    ("θ_x + 2π", np.array([theta_ref[0], theta_ref[1] + 2 * np.pi])),
]

print("Verifying HVA gate periodicities (N=6, p=1, h=3.0):")
print("Reference theta = [%.5f, %.5f]" % (theta_ref[0], theta_ref[1]))
print()
print("%-25s %12s %12s %s" % ("Transform", "Energy", "Diff", "Same E?"))
print("-" * 65)

bound_ref = qc.assign_parameters(theta_ref)
sv_ref = Statevector(bound_ref)
e_ref = float(sv_ref.expectation_value(H).real)

for name, theta in equivalents:
    bound = qc.assign_parameters(theta)
    sv = Statevector(bound)
    e = float(sv.expectation_value(H).real)
    diff = abs(e - e_ref)
    eq = "YES" if diff < 1e-10 else "no (%.2e)" % diff
    print("%-25s %12.8f %12.2e %s" % (name, e, diff, eq))

print()
print("=" * 65)
print("Now check the ACTUAL anomalous values from the experiments:")
print("=" * 65)

# From the N=50 seed 43 data: θ=[-0.0547, -0.3927]
# From the N=50 seed 44 data: θ=[-3.0869, 1.9635] and θ=[3.0947, 2.7489]
anomalous = [
    ("Seed 43 Z2: [-0.0547, -0.3927]", np.array([-0.0547, -0.3927])),
    ("Seed 44 A: [-3.0869, -1.1781]", np.array([-3.0869, -1.1781])),
    ("Seed 44 B: [3.0947, 2.7489]", np.array([3.0947, 2.7489])),
    ("Seed 44 C: [-0.0413, -1.9635]", np.array([-0.0413, -1.9635])),
]

# Use N=50 compatible h but test with N=6 for speed
# The symmetry is structural (gate-level), independent of N
theta_norm = np.array([0.0547, 0.3927])
bound_norm = qc.assign_parameters(theta_norm)
e_norm = float(Statevector(bound_norm).expectation_value(H).real)

print()
print("%-35s %12s %12s %s" % ("Anomalous θ", "Energy", "vs normal", "Equivalent?"))
print("-" * 75)
for name, theta in anomalous:
    bound = qc.assign_parameters(theta)
    e = float(Statevector(bound).expectation_value(H).real)
    diff = abs(e - e_norm)
    eq = "YES" if diff < 1e-6 else "no (Δ=%.2e)" % diff
    print("%-35s %12.8f %12.2e %s" % (name, e, diff, eq))

print()
print("=" * 65)
print("SYMMETRY GROUP IDENTIFICATION:")
print("=" * 65)
print("""
If (θ_zz, θ_x) gives energy E, which transforms also give E?
This defines the equivalence classes that canonicalization must handle.
""")
