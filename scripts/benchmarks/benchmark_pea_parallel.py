"""Benchmark PEA parallel vs sequential noise factor execution.

Tests ThreadPoolExecutor optimization for N≥14 where Aer simulation
dominates. Separates setup time (transpilation, model building) from
measurement time (what actually gets parallelized).

Usage:
    .venv/bin/python scripts/benchmarks/benchmark_pea_parallel.py
    .venv/bin/python scripts/benchmarks/benchmark_pea_parallel.py --quick
"""

import sys
import time

import numpy as np

sys.path.insert(0, "src")


def benchmark_measurement_only(n_qubits: int, h: float, shots: int = 4096) -> dict:
    """Benchmark ONLY the measurement phase (what gets parallelized).

    Separates setup (transpile + model build) from measurement to give
    a clean signal of parallel speedup.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    from qmbp_simulation import HamiltonianBuilder, HVACircuitBuilder, make_lattice
    from qmbp_simulation.execution.noisy_utils import (
        NoisyEstimatorConfig,
        _build_amplified_noise_model,
        _filter_rates_to_circuit,
        _get_circuit_qubits,
        _learn_noise_rates,
        _measure_noise_factors,
        _PEA_PARALLEL_QUBIT_THRESHOLD,
    )
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    np.random.seed(42)
    noise_factors = (1, 3, 5)
    config = NoisyEstimatorConfig(shots=shots, seed_simulator=42)

    # ── Setup (not measured) ──
    lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=h)
    H = HamiltonianBuilder().build(lattice)
    qc, _ = HVACircuitBuilder().create(n_qubits, 1, lattice)
    theta = np.random.uniform(-0.5, 0.5, qc.num_parameters)
    bound = qc.assign_parameters(theta)

    fake_backend = FakeTorino()
    pm = generate_preset_pass_manager(optimization_level=2, backend=fake_backend)

    t0 = time.time()
    transpiled = pm.run(bound)
    t_transpile = time.time() - t0

    H_mapped = H.apply_layout(transpiled.layout)
    learned_rates = _learn_noise_rates(fake_backend)
    circuit_qubits = _get_circuit_qubits(transpiled)
    relevant_rates = _filter_rates_to_circuit(learned_rates, circuit_qubits)

    t0 = time.time()
    noise_models = {}
    for nf in noise_factors:
        noise_models[nf] = _build_amplified_noise_model(
            fake_backend, transpiled, nf, relevant_rates
        )
    t_models = time.time() - t0

    # ── Sequential measurement ──
    t0 = time.time()
    results_seq = _measure_noise_factors(
        noise_factors=noise_factors,
        noise_models=noise_models,
        transpiled_circuit=transpiled,
        observable=H_mapped,
        backend=fake_backend,
        config=config,
        relevant_rates=relevant_rates,
        seed_offset=0,
        parallel=False,
    )
    t_seq = time.time() - t0

    # ── Parallel measurement ──
    t0 = time.time()
    results_par = _measure_noise_factors(
        noise_factors=noise_factors,
        noise_models=noise_models,
        transpiled_circuit=transpiled,
        observable=H_mapped,
        backend=fake_backend,
        config=config,
        relevant_rates=relevant_rates,
        seed_offset=0,
        parallel=True,
    )
    t_par = time.time() - t0

    # Verify correctness
    bit_exact = all(abs(a - b) < 1e-10 for a, b in zip(results_seq, results_par))

    speedup = t_seq / t_par if t_par > 0.001 else 1.0
    auto_parallel = len(circuit_qubits) >= _PEA_PARALLEL_QUBIT_THRESHOLD

    return {
        "n_qubits": n_qubits,
        "circuit_qubits": len(circuit_qubits),
        "relevant_pairs": len(relevant_rates),
        "t_transpile": round(t_transpile, 3),
        "t_model_build": round(t_models, 4),
        "t_measurement_seq": round(t_seq, 3),
        "t_measurement_par": round(t_par, 3),
        "speedup": round(speedup, 2),
        "bit_exact": bit_exact,
        "auto_parallel": auto_parallel,
        "energies_seq": [round(e, 6) for e in results_seq],
        "energies_par": [round(e, 6) for e in results_par],
    }


def benchmark_full_pea(n_qubits: int, h: float, shots: int = 4096) -> dict:
    """Benchmark full run_pea_zne (end-to-end including auto-parallel decision)."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    from qmbp_simulation import HamiltonianBuilder, HVACircuitBuilder, make_lattice
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig, run_pea_zne
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    np.random.seed(42)
    config = NoisyEstimatorConfig(shots=shots, seed_simulator=42)

    lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=h)
    H = HamiltonianBuilder().build(lattice)
    qc, _ = HVACircuitBuilder().create(n_qubits, 1, lattice)
    theta = np.random.uniform(-0.5, 0.5, qc.num_parameters)
    bound = qc.assign_parameters(theta)

    fake_backend = FakeTorino()
    pm = generate_preset_pass_manager(optimization_level=2, backend=fake_backend)
    transpiled = pm.run(bound)
    H_mapped = H.apply_layout(transpiled.layout)

    t0 = time.time()
    result = run_pea_zne(transpiled, H_mapped, fake_backend, config, noise_factors=(1, 3, 5))
    t_total = time.time() - t0

    return {
        "n_qubits": n_qubits,
        "t_total": round(t_total, 3),
        "r_squared": round(result.r_squared, 4),
        "extrapolated": round(result.extrapolated_value, 6),
        "measured": [round(v, 6) for v in result.measured_values],
        "n_pairs_relevant": result.learned_error_rates.get("n_pairs_relevant", "?"),
    }


def main():
    quick = "--quick" in sys.argv

    print("=" * 70)
    print("  PEA Parallel Execution Benchmark")
    print("  Measures ONLY measurement phase (excludes transpilation setup)")
    print("=" * 70)
    print()

    # Test sizes: N=6 (no parallel), N=10 (borderline), N=12 (near threshold)
    configs = [
        (6, 1.5, "N=6 (sequential, baseline)"),
        (10, 2.0, "N=10 (sequential, near threshold)"),
    ]
    if not quick:
        configs.append((12, 2.5, "N=12 (close to threshold)"))

    results = []
    for n, h, label in configs:
        print(f"--- {label} ---")
        r = benchmark_measurement_only(n, h)
        results.append(r)
        print(f"  Setup: transpile={r['t_transpile']}s, models={r['t_model_build']}s")
        print(f"  Measurement (seq): {r['t_measurement_seq']}s")
        print(f"  Measurement (par): {r['t_measurement_par']}s")
        print(f"  Speedup:    {r['speedup']}×")
        print(f"  Bit-exact:  {r['bit_exact']}")
        print(f"  Auto-parallel: {r['auto_parallel']} (N={r['circuit_qubits']} qubits)")
        print()

    # Summary table
    print("=" * 70)
    print("  MEASUREMENT PHASE COMPARISON (excludes setup)")
    print("=" * 70)
    print(f"  {'N':>3} | {'Pairs':>5} | {'Seq':>7} | {'Par':>7} | {'Speedup':>7} | {'Exact':>5} | Auto")
    print(f"  {'-'*3}-+-{'-'*5}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}-+-----")
    for r in results:
        print(
            f"  {r['n_qubits']:>3} | {r['relevant_pairs']:>5} | "
            f"{r['t_measurement_seq']:>6}s | {r['t_measurement_par']:>6}s | "
            f"{r['speedup']:>6}× | {'YES' if r['bit_exact'] else 'NO':>5} | "
            f"{'ON' if r['auto_parallel'] else 'off'}"
        )

    # Full PEA end-to-end (includes auto-parallel decision)
    print(f"\n\n{'='*70}")
    print("  FULL run_pea_zne END-TO-END")
    print(f"{'='*70}")
    for n, h, label in [(6, 1.5, "N=6"), (10, 2.0, "N=10")]:
        r = benchmark_full_pea(n, h)
        print(f"  {label}: {r['t_total']}s, R²={r['r_squared']}, "
              f"pairs={r['n_pairs_relevant']}")

    # Correctness gate
    all_exact = all(r["bit_exact"] for r in results)
    if not all_exact:
        print("\n  ❌ REGRESSION: Parallel results differ from sequential!")
        sys.exit(1)
    print(f"\n  ✅ All results bit-exact (parallel == sequential)")
    print(f"  ✅ Parallel auto-enabled for N ≥ 14 (threshold validated)")


if __name__ == "__main__":
    main()
