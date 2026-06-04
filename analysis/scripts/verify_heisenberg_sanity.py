#!/usr/bin/env python3
"""Sanity checks for Heisenberg HVA implementation.

Verifies that the negative result is genuine (not a bug) by checking:
1. Circuit structure: correct params, gates, CX count
2. VQE optimization: E_vqe < E_Néel (optimizer works, but can't reach GS)
3. Néel trap: VQE from zero params doesn't move (gradient = 0)
4. Exact energies: physically reasonable (strong-field limit, critical gap)

Results (2026-06-01):
- Circuit: 8 params, 30 two-qubit gates (RXX:10, RYY:10, RZZ:10) ✅
- VQE from random init: E=-8.55 (improves 3.55 over Néel) ✅
- VQE from Néel: E=-5.00 (doesn't move — Néel is a trap) ⚠️
- Ground state: E=-14.46 (unreachable, gap=5.92, fid=0.05%) ✅
- Exact energies: physically consistent with known limits ✅
- Conclusion: expressibility limit + initial state trap, not a bug

Usage:
    python analysis/verify_heisenberg_sanity.py
    python analysis/verify_heisenberg_sanity.py --verbose
    python analysis/verify_heisenberg_sanity.py --h 2.0
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from qmbp_simulation import make_lattice
from qmbp_simulation.models import get_model_spec
from qmbp_simulation.solvers import ClassicalSolver


def test_circuit_structure(n_qubits: int = 6, p: int = 2) -> bool:
    """Verify Heisenberg circuit has correct structure."""
    print("=" * 70)
    print("  TEST 1: Circuit Structure Verification")
    print("=" * 70)

    lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=3.0)
    spec = get_model_spec("heisenberg")
    qc, theta = spec.create_circuit(n_qubits, p, lattice, **spec.circuit_kwargs)

    n_params = len(theta)
    bound = qc.assign_parameters(np.zeros(n_params))
    ops = bound.count_ops()
    rxx = ops.get("rxx", 0)
    ryy = ops.get("ryy", 0)
    rzz = ops.get("rzz", 0)
    two_qubit = rxx + ryy + rzz + ops.get("cx", 0)

    n_edges = len(lattice.edges)
    expected_params = 4 * p
    expected_2q = 3 * n_edges * p

    print(f"\n  N={n_qubits}, p={p}, chain_1d ({n_edges} edges):")
    print(
        f"    Parameters:     {n_params} (expected: {expected_params}) "
        f"{'✅' if n_params == expected_params else '❌'}"
    )
    print(
        f"    2-qubit gates:  {two_qubit} (expected: {expected_2q}) "
        f"{'✅' if two_qubit == expected_2q else '❌'}"
    )
    print(f"    Breakdown: RXX={rxx}, RYY={ryy}, RZZ={rzz}")
    print(f"    Depth: {bound.depth()}")

    ok = n_params == expected_params and two_qubit == expected_2q
    if not ok:
        print("    ❌ CIRCUIT STRUCTURE BUG DETECTED")
    return ok


def test_vqe_optimizes(h: float = 3.0, verbose: bool = False) -> bool:
    """Verify VQE optimizes from random init but can't reach ground state."""
    print("\n" + "=" * 70)
    print("  TEST 2: VQE Optimization (random init → does it improve?)")
    print("=" * 70)

    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import Statevector
    from scipy.optimize import minimize

    lattice = make_lattice("chain_1d", 6, J=1.0, h=h)
    spec = get_model_spec("heisenberg")
    H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
    qc, theta = spec.create_circuit(6, 2, lattice, **spec.circuit_kwargs)
    estimator = StatevectorEstimator()

    n_params = len(theta)

    def cost_fn(params):
        bound = qc.assign_parameters(params)
        return float(estimator.run([(bound, H)]).result()[0].data.evs)

    # E_Néel (zero params = initial state)
    E_neel = cost_fn(np.zeros(n_params))

    # VQE from multiple random inits (σ=π for full exploration)
    n_restarts = 5
    best_energy = E_neel
    best_params = np.zeros(n_params)
    all_energies = []

    for i in range(n_restarts):
        rng = np.random.default_rng(42 + i)
        x0 = rng.uniform(-np.pi, np.pi, size=n_params)
        res = minimize(cost_fn, x0, method="L-BFGS-B", options={"maxiter": 500})
        all_energies.append(res.fun)
        if verbose:
            print(f"    restart {i}: E={res.fun:.4f} (converged={res.success}, nit={res.nit})")
        if res.fun < best_energy:
            best_energy = res.fun
            best_params = res.x

    # E_exact + fidelity
    solver = ClassicalSolver()
    exact = solver.solve(H, lattice)
    psi_vqe = Statevector(qc.assign_parameters(best_params))
    fidelity = abs(Statevector(exact.ground_state).inner(psi_vqe)) ** 2

    print(f"\n  h={h}, N=6, Δ=1.0 (isotropic Heisenberg):")
    print(f"    E_exact (ground state):  {exact.ground_energy:.6f}")
    print(f"    E_Néel (zero params):    {E_neel:.6f}")
    print(f"    E_vqe (best of {n_restarts}):      {best_energy:.6f}")
    print(f"    E_vqe (mean±std):        {np.mean(all_energies):.4f} ± {np.std(all_energies):.4f}")
    print(f"    Fidelity (best):         {fidelity:.6f}")
    print(f"    ΔE from Néel:            {E_neel - best_energy:.4f}")
    print(f"    Gap to ground state:     {best_energy - exact.ground_energy:.4f}")
    print(
        f"    Expressibility ratio:    {(E_neel - best_energy) / (E_neel - exact.ground_energy) * 100:.1f}%"
    )

    optimizes = best_energy < E_neel - 0.01
    print(f"\n    VQE optimizes:  {'✅' if optimizes else '❌'}")
    print(f"    Reaches GS:     {'✅' if fidelity > 0.5 else '❌ (expressibility limit)'}")
    return optimizes


def test_neel_trap(h: float = 3.0) -> bool:
    """Verify that VQE from Néel (zero params) doesn't move."""
    print("\n" + "=" * 70)
    print("  TEST 3: Néel Initial State Trap")
    print("=" * 70)

    from qiskit.primitives import StatevectorEstimator
    from scipy.optimize import minimize

    lattice = make_lattice("chain_1d", 6, J=1.0, h=h)
    spec = get_model_spec("heisenberg")
    H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
    qc, theta = spec.create_circuit(6, 2, lattice, **spec.circuit_kwargs)
    estimator = StatevectorEstimator()

    n_params = len(theta)

    def cost_fn(params):
        bound = qc.assign_parameters(params)
        return float(estimator.run([(bound, H)]).result()[0].data.evs)

    # VQE starting from zero params (Néel state)
    E_neel = cost_fn(np.zeros(n_params))
    res = minimize(cost_fn, np.zeros(n_params), method="L-BFGS-B", options={"maxiter": 500})

    delta = abs(E_neel - res.fun)
    is_trapped = delta < 0.01

    print(f"\n  h={h}, N=6, starting from θ=0 (Néel state):")
    print(f"    E_initial:  {E_neel:.6f}")
    print(f"    E_final:    {res.fun:.6f}")
    print(f"    ΔE:         {delta:.6f}")
    print(f"    Iterations: {res.nit}")
    print(f"    Converged:  {res.success}")
    print(f"    Gradient norm: {np.linalg.norm(res.jac):.2e}" if hasattr(res, "jac") else "")

    if is_trapped:
        print(f"\n    ⚠️  CONFIRMED: Néel is a trap (ΔE={delta:.6f} < 0.01)")
        print("    The optimizer finds zero gradient at the Néel state.")
        print("    This explains why warm-start from h=4.0 propagates E≈-3.")
    else:
        print(f"\n    ✅ VQE escapes Néel (ΔE={delta:.4f})")

    return is_trapped


def test_exact_energies() -> bool:
    """Verify exact energies are physically reasonable."""
    print("\n" + "=" * 70)
    print("  TEST 4: Exact Energy Verification")
    print("=" * 70)

    from qmbp_simulation import HamiltonianBuilder

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()

    checks_passed = 0
    total_checks = 3

    # Check 1: Strong field limit (h >> J → E ≈ -hN)
    lattice_h10 = make_lattice("chain_1d", 6, J=1.0, h=10.0)
    H_h10 = builder.build_heisenberg(lattice_h10, delta=1.0)
    result_h10 = solver.solve(H_h10, lattice_h10)
    ratio = result_h10.ground_energy / (-10.0 * 6)
    ok1 = 0.85 < ratio < 1.0  # Should be close to -hN but not exact (J corrections)
    print(
        f"\n  Strong field (h=10, N=6): E={result_h10.ground_energy:.3f}, "
        f"ratio to -hN: {ratio:.4f} {'✅' if ok1 else '❌'}"
    )
    if ok1:
        checks_passed += 1

    # Check 2: TFIM critical gap should be small
    lattice_crit = make_lattice("chain_1d", 6, J=1.0, h=1.0)
    H_crit = builder.build(lattice_crit)
    result_crit = solver.solve(H_crit, lattice_crit)
    ok2 = result_crit.gap < 0.6  # Gap closes at criticality
    print(f"  TFIM critical (h=1): gap={result_crit.gap:.4f} {'✅' if ok2 else '❌'}")
    if ok2:
        checks_passed += 1

    # Check 3: Heisenberg at h=0 should have E/N < 0 (antiferromagnetic)
    lattice_h0 = make_lattice("chain_1d", 6, J=1.0, h=0.001)  # Small h to avoid degeneracy
    H_h0 = builder.build_heisenberg(lattice_h0, delta=1.0)
    result_h0 = solver.solve(H_h0, lattice_h0)
    e_per_site = result_h0.ground_energy / 6
    ok3 = e_per_site < 0  # Must be negative (antiferromagnetic)
    print(f"  Heisenberg h≈0: E/N={e_per_site:.4f} {'✅' if ok3 else '❌'} (must be < 0)")
    if ok3:
        checks_passed += 1

    print(f"\n  Passed: {checks_passed}/{total_checks}")
    return checks_passed == total_checks


def main():
    parser = argparse.ArgumentParser(description="Heisenberg HVA sanity checks")
    parser.add_argument("--h", type=float, default=3.0, help="Field value for VQE test")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("\n" + "═" * 70)
    print("  HEISENBERG HVA SANITY CHECKS")
    print("═" * 70)

    ok1 = test_circuit_structure()
    ok2 = test_vqe_optimizes(h=args.h, verbose=args.verbose)
    ok3 = test_neel_trap(h=args.h)
    ok4 = test_exact_energies()

    print("\n" + "═" * 70)
    print("  SUMMARY")
    print("═" * 70)
    print(f"  1. Circuit structure:    {'✅ PASS' if ok1 else '❌ FAIL'}")
    print(f"  2. VQE optimizes:        {'✅ PASS' if ok2 else '❌ FAIL'}")
    print(f"  3. Néel trap confirmed:  {'✅ CONFIRMED' if ok3 else '⚠️ NOT TRAPPED'}")
    print(f"  4. Exact energies valid: {'✅ PASS' if ok4 else '❌ FAIL'}")
    print()

    all_ok = ok1 and ok2 and ok4  # ok3 is informational (trap is expected)
    if all_ok:
        print("  CONCLUSION: Negative result is GENUINE (not a bug).")
        print("  The HVA circuit works but cannot express the Heisenberg ground state.")
    else:
        print("  ⚠️  POTENTIAL ISSUE DETECTED — investigate before trusting results.")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
