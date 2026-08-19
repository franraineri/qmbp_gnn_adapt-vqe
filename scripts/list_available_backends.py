#!/usr/bin/env python3
"""List available IBM Quantum backends for your account.

Usage:
    .venv/bin/python scripts/list_available_backends.py
"""

import os

from qiskit_ibm_runtime import QiskitRuntimeService

key = os.environ.get("IBM_KEY")
crn = os.environ.get("IBM_INSTANCE_CRN")

if not key:
    print("❌ IBM_KEY not set")
    raise SystemExit(1)
if not crn:
    print("❌ IBM_INSTANCE_CRN not set")
    raise SystemExit(1)

print(f"🔑 Key: ****{key[-4:]}")
print(f"🏢 Instance: ...{crn[-20:]}")
print()

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token=key,
    instance=crn,
)

print("Available backends (operational, ≥10 qubits):")
print("-" * 60)
backends = service.backends(min_num_qubits=10, operational=True)
for b in sorted(backends, key=lambda x: x.num_qubits, reverse=True):
    print(f"  {b.name:<25} {b.num_qubits:>4} qubits")

print(f"\nTotal: {len(backends)} backends")
print("\nTo use a specific backend:")
print("  make hw-deploy ARGS='--backend <name> --no-spsa'")
