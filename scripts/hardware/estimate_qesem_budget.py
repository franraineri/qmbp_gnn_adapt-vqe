#!/usr/bin/env python3
"""QESEM QPU Budget Estimator.

Estimates the QPU time required to run QESEM mitigation for our HVA p=1 N=10
deployment circuits. Uses QESEM's analytical time estimation (no QPU cost)
or empirical estimation (~2 min QPU for more accurate results).

This script helps decide whether to use QESEM or PEA-ZNE for a given run,
based on the QPU budget tradeoff.

Usage:
    # Analytical estimation (free, rough)
    .venv/bin/python scripts/estimate_qesem_budget.py

    # Empirical estimation (uses ~2 min QPU, more accurate)
    .venv/bin/python scripts/estimate_qesem_budget.py --empirical

    # Custom precision
    .venv/bin/python scripts/estimate_qesem_budget.py --precision 0.005

Requirements:
    - IBM credentials: export IBM_KEY="..." and IBM_INSTANCE_CRN="..."
    - qiskit-ibm-catalog >= 0.8.0
    - IBM Premium/Flex plan access
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from qmbp_simulation.framework.runner_base import resolve_project_root

_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qmbp_simulation import HamiltonianBuilder, HVACircuitBuilder, make_lattice
from qmbp_simulation.execution.hardware.config import HardwareConfig
from qmbp_simulation.execution.hardware.observables import build_per_site_observables

# Default deployment parameters
TOPOLOGY = "heavy_hex"
N_QUBITS = 10
P_LAYERS = 1
H_VALUES = [4.0, 3.5, 3.25, 3.0]  # TIER_1_H from deployment script


def build_test_circuit(h_value: float) -> tuple:
    """Build a bound HVA p=1 circuit for the given h-value.

    Returns (circuit, hamiltonian, x_ops, zz_ops).
    """
    lattice = make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=h_value)
    builder = HamiltonianBuilder()
    hamiltonian = builder.build(lattice)

    circuit_builder = HVACircuitBuilder()
    circuit, _ = circuit_builder.create_pauli_evolution(N_QUBITS, P_LAYERS, lattice)

    # Use a reasonable warm-start guess (θ_zz ≈ -0.5, θ_x ≈ -1.2)
    params = np.array([-0.5, -1.2])
    bound_circuit = circuit.assign_parameters(params)

    edges = [(i, i + 1) for i in range(N_QUBITS - 1)]
    x_ops, zz_ops = build_per_site_observables(N_QUBITS, edges)

    return bound_circuit, hamiltonian, x_ops, zz_ops


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate QESEM QPU budget for GNN-HVA deployment circuits",
    )
    parser.add_argument(
        "--empirical",
        action="store_true",
        help="Use empirical estimation (~2 min QPU, more accurate)",
    )
    parser.add_argument(
        "--precision",
        type=float,
        default=0.01,
        help="Target precision per observable (default: 0.01)",
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=300,
        help="Max execution time per PUB in seconds (default: 300)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="ibm_kingston",
        help="Backend name (default: ibm_kingston)",
    )
    args = parser.parse_args()

    from qmbp_simulation.execution.hardware.qesem import (
        check_qesem_available,
        estimate_qesem_time,
    )

    # Check dependencies
    available, err = check_qesem_available()
    if not available:
        print(f"❌ QESEM not available: {err}")
        print("   Install: pip install qiskit-ibm-catalog>=0.8.0")
        sys.exit(1)

    print("═" * 60)
    print("  QESEM Budget Estimation — GNN-HVA Deployment Circuits")
    print("═" * 60)
    print(f"  Mode: {'empirical (~2 min QPU)' if args.empirical else 'analytical (free)'}")
    print(f"  Precision: ε = {args.precision}")
    print(f"  Backend: {args.backend}")
    print(f"  Circuit: HVA p={P_LAYERS}, N={N_QUBITS}, topology={TOPOLOGY}")
    print(f"  h-values: {H_VALUES}")
    print("-" * 60)

    config = HardwareConfig(
        backend_name=args.backend,
        qesem_precision=args.precision,
        qesem_max_execution_time=args.max_time,
    )

    total_time = 0.0
    results = []

    for h in H_VALUES:
        circuit, hamiltonian, x_ops, zz_ops = build_test_circuit(h)
        all_obs = [hamiltonian] + x_ops + zz_ops

        mode = "empirical" if args.empirical else "analytical"

        try:
            estimate = estimate_qesem_time(circuit, all_obs, config, mode=mode)
            time_sec = estimate.get("time_estimation_sec", None)
            results.append({"h": h, "time_sec": time_sec, "job_id": estimate.get("job_id")})
            if time_sec:
                total_time += time_sec
            status = f"{time_sec:.0f}s" if time_sec else "N/A"
        except Exception as e:
            results.append({"h": h, "time_sec": None, "error": str(e)})
            status = f"ERROR: {e}"

        print(f"  h={h:.2f}: estimated QPU time = {status}")

    print("-" * 60)
    print(f"  Total estimated QPU time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"  Per h-point average: {total_time / len(H_VALUES):.0f}s")
    print()
    print("  Comparison (from simulation benchmarks):")
    print("    PEA-ZNE (local): ~30-60s per h-point (16K shots × 3 layouts)")
    print(f"    QESEM:           ~{total_time / len(H_VALUES):.0f}s per h-point (estimated)")
    print("═" * 60)


if __name__ == "__main__":
    main()
