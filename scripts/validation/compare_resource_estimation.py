#!/usr/bin/env python3
"""Compare & validate Qiskit ResourceEstimation integration.

Tests the unified `transpiled_circuit_stats` function against multiple
circuit configurations and validates consistency across topologies.

Sections:
  1. Consistency check: transpiled_circuit_stats vs raw ResourceEstimation
  2. Cross-topology comparison (chain_1d, ladder, heavy_hex, triangular)
  3. Error budget estimation from count_ops + calibration data
  4. depth_2q validation across optimization levels
  5. Logical vs transpiled comparison (gate overhead analysis)

Usage:
    .venv/bin/python scripts/compare_resource_estimation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ResourceEstimation
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeKingston

from qmbp_simulation import HVACircuitBuilder, make_lattice
from qmbp_simulation.analysis.circuit_visualizer import (
    circuit_summary,
    transpiled_circuit_stats,
)


def section_1_consistency_check(backend, hva) -> dict:
    """Verify transpiled_circuit_stats matches raw ResourceEstimation."""
    print("\n" + "=" * 70)
    print("  Section 1: Consistency — transpiled_circuit_stats vs raw RE")
    print("=" * 70)

    lattice = make_lattice("heavy_hex", 10, J=1.0, h=4.0)
    circuit, _ = hva.create_pauli_evolution(10, 1, lattice)
    bound = circuit.assign_parameters(np.array([0.3, -0.2]))

    pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
    transpiled = pm.run(bound)

    # Our unified function
    stats = transpiled_circuit_stats(transpiled)

    # Raw ResourceEstimation for cross-check
    re_pm = PassManager([ResourceEstimation()])
    re_pm.run(transpiled)
    prop = re_pm.property_set

    checks = []
    checks.append(("depth", stats["depth"] == prop.get("depth")))
    checks.append(("total_gates == size", stats["total_gates"] == prop.get("size")))
    checks.append(("width", stats["width"] == prop.get("width")))
    checks.append(
        ("num_tensor_factors", stats["num_tensor_factors"] == prop.get("num_tensor_factors"))
    )
    re_ops = dict(prop.get("count_ops")) if prop.get("count_ops") else {}
    checks.append(("count_ops match", stats["count_ops"] == re_ops))

    # Verify 2Q count from count_ops matches our n_2q_gates
    n_2q_from_ops = sum(
        v
        for k, v in stats["count_ops"].items()
        if k in ("cz", "cx", "ecr", "rzz", "rxx", "ryy", "cp")
    )
    checks.append(("n_2q consistent with count_ops", stats["n_2q_gates"] == n_2q_from_ops))

    # Verify active_qubits = N (our logical qubits)
    checks.append(("active_qubits == 10", stats["active_qubits"] == 10))

    all_pass = all(ok for _, ok in checks)
    for name, ok in checks:
        print(f"  {'✅' if ok else '❌'} {name}")

    print(f"\n  Result: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    return {"section": 1, "all_pass": all_pass, "stats": stats, "checks": checks}


def section_2_cross_topology(backend, hva) -> dict:
    """Compare resource stats across all 4 topologies."""
    print("\n" + "=" * 70)
    print("  Section 2: Cross-topology resource comparison")
    print("=" * 70)

    topologies = [
        ("chain_1d", 10, 1),
        ("ladder", 10, 1),
        ("heavy_hex", 10, 1),
        ("triangular", 9, 1),  # triangular needs square N for some configs
    ]

    results = []
    for topo, n, p in topologies:
        lattice = make_lattice(topo, n, J=1.0, h=4.0)
        circuit, _ = hva.create_pauli_evolution(n, p, lattice)
        theta = np.random.uniform(-0.1, 0.1, circuit.num_parameters)
        bound = circuit.assign_parameters(theta)

        pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
        transpiled = pm.run(bound)
        stats = transpiled_circuit_stats(transpiled)
        stats["topology"] = topo
        stats["n_logical"] = n
        stats["p_layers"] = p
        results.append(stats)

        print(
            f"  {topo:<12} N={n} p={p}: depth={stats['depth']:>3}, "
            f"depth_2q={stats['depth_2q']:>3}, "
            f"n_2q={stats['n_2q_gates']:>3}, "
            f"active={stats.get('active_qubits', '?')}"
        )

    print(f"\n  Basis gates (should be uniform): {results[0].get('count_ops', {}).keys()}")
    return {"section": 2, "results": results}


def section_3_error_budget(backend, hva) -> dict:
    """Estimate error budget from count_ops + typical calibration rates."""
    print("\n" + "=" * 70)
    print("  Section 3: Error budget estimation from gate-type breakdown")
    print("=" * 70)

    # Typical IBM Heron R2 error rates (from public calibration data)
    # CZ/ECR: 5e-3 to 1.5e-2 (layout dependent)
    # SX: 1.5e-4 to 5e-4
    # RZ: virtual gate (0 error, frame change only)
    # X: same as SX (~2e-4)
    ERROR_RATES = {
        "cz": {"optimistic": 5e-3, "typical": 8e-3, "pessimistic": 1.5e-2},
        "ecr": {"optimistic": 5e-3, "typical": 8e-3, "pessimistic": 1.5e-2},
        "cx": {"optimistic": 5e-3, "typical": 8e-3, "pessimistic": 1.5e-2},
        "sx": {"optimistic": 1.5e-4, "typical": 2.5e-4, "pessimistic": 5e-4},
        "x": {"optimistic": 1.5e-4, "typical": 2.5e-4, "pessimistic": 5e-4},
        "rz": {"optimistic": 0.0, "typical": 0.0, "pessimistic": 0.0},
    }

    configs = [
        ("heavy_hex N=10 p=1 (deploy)", "heavy_hex", 10, 1),
        ("chain_1d N=6 p=2 (ref)", "chain_1d", 6, 2),
        ("chain_1d N=10 p=1 (scaling)", "chain_1d", 10, 1),
    ]

    results = []
    for label, topo, n, p in configs:
        lattice = make_lattice(topo, n, J=1.0, h=4.0)
        circuit, _ = hva.create_pauli_evolution(n, p, lattice)
        theta = np.random.uniform(-0.1, 0.1, circuit.num_parameters)
        bound = circuit.assign_parameters(theta)

        pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
        transpiled = pm.run(bound)
        stats = transpiled_circuit_stats(transpiled)

        # Compute error budgets
        budgets = {}
        for scenario in ("optimistic", "typical", "pessimistic"):
            total_error = 0.0
            for gate, count in stats["count_ops"].items():
                rate = ERROR_RATES.get(gate, {}).get(scenario, 0.0)
                total_error += count * rate
            budgets[scenario] = total_error

        # Fidelity estimate: F ≈ (1 - ε_avg)^n_gates ≈ exp(-total_error)
        fidelity_est = np.exp(-budgets["typical"])

        stats["error_budget"] = budgets
        stats["fidelity_estimate"] = fidelity_est
        stats["label"] = label
        results.append(stats)

        print(f"\n  {label}:")
        print(f"    count_ops: {stats['count_ops']}")
        print(
            f"    Error budget: opt={budgets['optimistic']:.4f}, "
            f"typ={budgets['typical']:.4f}, pess={budgets['pessimistic']:.4f}"
        )
        print(f"    Fidelity estimate (typical): {fidelity_est:.4f}")
        print(f"    depth_2q={stats['depth_2q']} → decoherence layers")

    # Interpretation
    print("\n  Interpretation:")
    print("    - Error budget < 0.30 → ZNE can recover (linear extrapolation valid)")
    print("    - Error budget > 0.50 → circuit too deep, ZNE unreliable")
    print("    - depth_2q × T_2q / T2 gives decoherence fraction")
    return {"section": 3, "results": results}


def section_4_depth_2q_vs_opt_level(backend, hva) -> dict:
    """Check if depth_2q varies across optimization levels."""
    print("\n" + "=" * 70)
    print("  Section 4: depth_2q across optimization levels (0, 1, 2, 3)")
    print("=" * 70)

    lattice = make_lattice("heavy_hex", 10, J=1.0, h=4.0)
    circuit, _ = hva.create_pauli_evolution(10, 1, lattice)
    bound = circuit.assign_parameters(np.array([0.3, -0.2]))

    results = []
    for level in [0, 1, 2, 3]:
        pm = generate_preset_pass_manager(
            optimization_level=level, backend=backend, seed_transpiler=42
        )
        transpiled = pm.run(bound)
        stats = transpiled_circuit_stats(transpiled)
        stats["opt_level"] = level
        results.append(stats)

        print(
            f"  Level {level}: depth={stats['depth']:>3}, "
            f"depth_2q={stats['depth_2q']:>3}, "
            f"n_2q={stats['n_2q_gates']:>3}, "
            f"total={stats['total_gates']:>4}"
        )

    # Key insight: does opt_level change depth_2q?
    d2q_vals = [r["depth_2q"] for r in results]
    print(f"\n  depth_2q range: [{min(d2q_vals)}, {max(d2q_vals)}]")
    print(
        f"  n_2q range: [{min(r['n_2q_gates'] for r in results)}, "
        f"{max(r['n_2q_gates'] for r in results)}]"
    )

    if min(d2q_vals) == max(d2q_vals):
        print("  → depth_2q invariant across opt levels (good: HVA structure preserved)")
    else:
        print("  → depth_2q VARIES — higher levels may parallelize 2Q gates better")

    return {"section": 4, "results": results}


def section_5_logical_vs_transpiled(backend, hva) -> dict:
    """Analyze transpilation overhead (logical → physical)."""
    print("\n" + "=" * 70)
    print("  Section 5: Logical vs Transpiled overhead")
    print("=" * 70)

    configs = [
        ("heavy_hex N=10 p=1", "heavy_hex", 10, 1),
        ("chain_1d N=6 p=2", "chain_1d", 6, 2),
        ("chain_1d N=20 p=1", "chain_1d", 20, 1),
    ]

    results = []
    for label, topo, n, p in configs:
        lattice = make_lattice(topo, n, J=1.0, h=4.0)
        circuit, _ = hva.create_pauli_evolution(n, p, lattice)
        theta = np.random.uniform(-0.1, 0.1, circuit.num_parameters)
        bound = circuit.assign_parameters(theta)

        logical = circuit_summary(bound)

        pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
        transpiled = pm.run(bound)
        physical = transpiled_circuit_stats(transpiled)

        overhead = {
            "label": label,
            "logical_depth": logical["depth"],
            "physical_depth": physical["depth"],
            "depth_overhead": physical["depth"] - logical["depth"],
            "logical_2q": logical["n_2q_gates"],
            "physical_2q": physical["n_2q_gates"],
            "swap_overhead_2q": physical["n_2q_gates"] - logical["n_2q_gates"],
            "depth_2q": physical["depth_2q"],
            "active_qubits": physical.get("active_qubits"),
        }
        results.append(overhead)

        swap_2q = overhead["swap_overhead_2q"]
        # Note: for PauliEvolutionGate, logical circuit has 0 explicit 2Q gates
        # (they are abstract evolution operators). The "overhead" is really the
        # decomposition into basis gates, not SWAP insertion.
        if logical["n_2q_gates"] == 0:
            swap_note = "from decomposition"
        elif swap_2q == 0:
            swap_note = "NO SWAPs"
        else:
            swap_note = f"+{swap_2q} from SWAPs"
        print(
            f"  {label}:\n"
            f"    depth: {logical['depth']} → {physical['depth']} "
            f"(+{overhead['depth_overhead']})\n"
            f"    2Q gates: {logical['n_2q_gates']} → {physical['n_2q_gates']} "
            f"({swap_note})\n"
            f"    depth_2q: {physical['depth_2q']}"
        )

    return {"section": 5, "results": results}


def main():
    """Run all comparison sections."""
    print("=" * 70)
    print("  ResourceEstimation Integration — Comparison & Validation Suite")
    print("=" * 70)

    backend = FakeKingston()
    hva = HVACircuitBuilder()

    t0 = time.time()
    results = {}
    results["s1"] = section_1_consistency_check(backend, hva)
    results["s2"] = section_2_cross_topology(backend, hva)
    results["s3"] = section_3_error_budget(backend, hva)
    results["s4"] = section_4_depth_2q_vs_opt_level(backend, hva)
    results["s5"] = section_5_logical_vs_transpiled(backend, hva)

    elapsed = time.time() - t0

    # ── Summary & Recommendations ──
    print("\n" + "=" * 70)
    print("  SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    print(f"""
  Execution time: {elapsed:.1f}s

  ┌─────────────────────────────────────────────────────────────────────┐
  │ What transpiled_circuit_stats provides (unified, canonical):         │
  │                                                                     │
  │  • depth        — total circuit depth (scheduling)                  │
  │  • depth_2q     — 2Q critical path (ERROR PREDICTOR)                │
  │  • n_2q_gates   — total 2Q count (ZNE viability threshold: ≤18)     │
  │  • count_ops    — per-gate-type breakdown (error budget calc)        │
  │  • active_qubits — confirms no unwanted routing expansion            │
  │  • num_tensor_factors — disconnected components (sanity check)       │
  │                                                                     │
  │ Improvement opportunities for ad-hoc measurements:                  │
  │                                                                     │
  │  1. REPLACE all raw `circ.depth()` + `circ.data` iterations         │
  │     with `transpiled_circuit_stats(circ)` in scripts that need      │
  │     full stats (rehearsals, transpiler comparison, deployment).      │
  │                                                                     │
  │  2. ADD depth_2q to result JSONs — better error predictor than      │
  │     total depth. Correlate with actual ΔE/gap on hardware.          │
  │                                                                     │
  │  3. ADD error_budget = Σ(n_gate_i × ε_i) from calibration data      │
  │     → predicted fidelity F ≈ exp(-error_budget).                    │
  │     Compare F_predicted vs F_measured post-hardware.                 │
  │                                                                     │
  │  4. USE count_ops to detect unexpected gates (e.g. SWAP=3CX):       │
  │     if n_2q_physical > n_2q_logical → routing introduced SWAPs.     │
  │                                                                     │
  │  5. MONITOR depth_2q across layouts — if it varies, some layouts     │
  │     have better parallelism. Pick lowest depth_2q for ZNE primary.  │
  └─────────────────────────────────────────────────────────────────────┘
    """)

    # Save results
    output_path = Path("results/resource_estimation_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _serialize(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(v) for v in obj]
        return obj

    output_path.write_text(json.dumps(_serialize(results), indent=2, default=str))
    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
