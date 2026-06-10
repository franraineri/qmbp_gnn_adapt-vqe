"""Benchmark PEA-ZNE performance: measures per-component timing.

Use to verify optimization effectiveness and detect regressions.
Expected output (post-optimization, 2026-06-10):
  N=6:  ~0.2-0.4s total (10× faster than pre-optimization 2.0s)
  N=10: ~1.5-2.0s total (2× faster than pre-optimization 3.0s)

Usage:
    .venv/bin/python scripts/benchmarks/benchmark_pea_performance.py
"""

import sys
import time

import numpy as np

sys.path.insert(0, "src")

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qmbp_simulation import HamiltonianBuilder, HVACircuitBuilder, make_lattice
from qmbp_simulation.execution.noisy_utils import (
    NoisyEstimatorConfig,
    _filter_rates_to_circuit,
    _get_circuit_qubits,
    _learn_noise_rates,
    run_pea_zne,
)


def benchmark_pea(n_qubits: int, h: float, seed: int = 42) -> dict:
    """Run a single PEA benchmark and return timing breakdown."""
    np.random.seed(seed)

    lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=h)
    H = HamiltonianBuilder().build(lattice)
    qc, _ = HVACircuitBuilder().create(n_qubits, 1, lattice)
    theta = np.random.uniform(-0.5, 0.5, qc.num_parameters)
    bound = qc.assign_parameters(theta)

    from qiskit_ibm_runtime.fake_provider import FakeTorino

    fake_backend = FakeTorino()

    # Transpile
    t0 = time.time()
    pm = generate_preset_pass_manager(optimization_level=2, backend=fake_backend)
    transpiled = pm.run(bound)
    H_mapped = H.apply_layout(transpiled.layout)
    t_transpile = time.time() - t0

    # Noise filtering stats
    learned_rates = _learn_noise_rates(fake_backend)
    circuit_qubits = _get_circuit_qubits(transpiled)
    relevant = _filter_rates_to_circuit(learned_rates, circuit_qubits)

    # Full PEA
    config = NoisyEstimatorConfig(shots=4096, seed_simulator=seed)
    t0 = time.time()
    result = run_pea_zne(transpiled, H_mapped, fake_backend, config, noise_factors=(1, 3, 5))
    t_pea = time.time() - t0

    return {
        "n_qubits": n_qubits,
        "h": h,
        "transpile_s": round(t_transpile, 3),
        "pea_total_s": round(t_pea, 3),
        "circuit_qubits": len(circuit_qubits),
        "relevant_pairs": len(relevant),
        "total_pairs": len(learned_rates),
        "filter_ratio": f"{len(relevant)}/{len(learned_rates)}",
        "r_squared": round(result.r_squared, 4),
        "extrapolated_energy": round(result.extrapolated_value, 6),
    }


def main():
    """Run benchmarks for N=6 and N=10."""
    print("=" * 60)
    print("  PEA-ZNE Performance Benchmark")
    print("=" * 60)
    print()

    configs = [
        {"n_qubits": 6, "h": 1.5, "label": "N=6 chain_1d (typical)"},
        {"n_qubits": 10, "h": 2.0, "label": "N=10 chain_1d (hardware target)"},
    ]

    for cfg in configs:
        print(f"--- {cfg['label']} ---")
        result = benchmark_pea(cfg["n_qubits"], cfg["h"])
        for key, val in result.items():
            print(f"  {key}: {val}")
        print()

    # Performance regression check
    print("=" * 60)
    print("  REGRESSION CHECK")
    print("=" * 60)
    r6 = benchmark_pea(6, 1.5)
    r10 = benchmark_pea(10, 2.0)

    # Post-optimization thresholds (2026-06-10 baseline)
    pass_6 = r6["pea_total_s"] < 1.0  # Should be ~0.3s (was 2.0s)
    pass_10 = r10["pea_total_s"] < 3.0  # Should be ~1.7s (was 3.0s)

    print(f"  N=6:  {r6['pea_total_s']}s {'PASS' if pass_6 else 'REGRESSION'} (threshold: <1.0s)")
    print(f"  N=10: {r10['pea_total_s']}s {'PASS' if pass_10 else 'REGRESSION'} (threshold: <3.0s)")

    if not (pass_6 and pass_10):
        sys.exit(1)
    print("\n  All performance checks PASS ✅")


if __name__ == "__main__":
    main()
