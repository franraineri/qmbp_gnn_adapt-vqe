#!/usr/bin/env python3
"""Test 4: Verify that increasing HVA depth (p>2) improves fidelity.

If fidelity increases with p, the circuit implementation is correct and
the p=2 failure is genuinely an expressibility limit. If fidelity stays
at zero even at p=6, there's a deeper bug.

This script TEMPORARILY patches MAX_P_LAYERS to allow p>2 for testing.
No stable code is modified.

Tests:
  A) Heisenberg (Δ=1.0) at h=3.0, N=6: p=1,2,3,4,5,6
  B) XY (Δ=0.0) at h=3.0, N=6: p=1,2,3,4,5,6
  C) TFIM at h=1.5, N=6: p=1,2,3,4 (should be ~1.0 at p≥2)
  D) Heisenberg at h=0.5, N=6: p=2,4,6 (deep in correlated regime)

Expected:
  - Heisenberg: fidelity should increase monotonically with p
  - TFIM: fidelity should be ~1.0 already at p=2 (no improvement needed)
  - If Heisenberg stays at 0% for all p → BUG in create_heisenberg

Usage:
    python analysis/verify_depth_scaling.py
    python analysis/verify_depth_scaling.py --verbose
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

# Patch MAX_P_LAYERS BEFORE importing circuits
import qmbp_simulation.models.constants as constants_mod

ORIGINAL_MAX_P = constants_mod.MAX_P_LAYERS
constants_mod.MAX_P_LAYERS = 8  # Allow up to p=8 for testing

# Also patch the hva module's local copy
import qmbp_simulation.circuits.hva as _hva_mod  # noqa: E402

_hva_mod.MAX_P_LAYERS = 8

from qmbp_simulation import make_lattice  # noqa: E402
from qmbp_simulation.models import get_model_spec  # noqa: E402
from qmbp_simulation.solvers import ClassicalSolver  # noqa: E402


def run_vqe_at_depth(
    model: str,
    delta: float | None,
    h: float,
    p: int,
    n_qubits: int = 6,
    n_restarts: int = 5,
    maxiter: int = 500,
    seed: int = 42,
) -> dict:
    """Run VQE at a given depth and return energy + fidelity."""
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import Statevector
    from scipy.optimize import minimize

    lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=h)
    spec = get_model_spec(model)
    if delta is not None:
        spec = spec.with_delta(delta)

    H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
    qc, theta = spec.create_circuit(n_qubits, p, lattice, **spec.circuit_kwargs)
    estimator = StatevectorEstimator()
    n_params = len(theta)

    def cost_fn(params):
        bound = qc.assign_parameters(params)
        return float(estimator.run([(bound, H)]).result()[0].data.evs)

    # Multi-restart VQE
    rng = np.random.default_rng(seed)
    best_energy = float("inf")
    best_params = np.zeros(n_params)

    for _ in range(n_restarts):
        x0 = rng.uniform(-np.pi, np.pi, size=n_params)
        res = minimize(cost_fn, x0, method="L-BFGS-B", options={"maxiter": maxiter})
        if res.fun < best_energy:
            best_energy = res.fun
            best_params = res.x

    # Exact solution + fidelity
    solver = ClassicalSolver()
    exact = solver.solve(H, lattice)
    psi_vqe = Statevector(qc.assign_parameters(best_params))
    fidelity = abs(Statevector(exact.ground_state).inner(psi_vqe)) ** 2

    return {
        "model": model,
        "delta": delta,
        "h": h,
        "p": p,
        "n_params": n_params,
        "E_exact": exact.ground_energy,
        "E_vqe": best_energy,
        "fidelity": fidelity,
        "gap_to_gs": best_energy - exact.ground_energy,
    }


def run_test_suite(verbose: bool = False) -> list[dict]:
    """Run the full depth-scaling test suite."""
    results = []

    # ─── Test A: Heisenberg Δ=1.0 at h=3.0 ────────────────────────────
    print("\n" + "═" * 70)
    print("  TEST A: Heisenberg (Δ=1.0) at h=3.0, N=6")
    print("═" * 70)
    print(f"  {'p':<4} {'Params':<8} {'E_exact':<12} {'E_vqe':<12} {'Fidelity':<10} {'Gap':<10}")
    print(f"  {'-' * 60}")

    for p in [1, 2, 3, 4, 5, 6]:
        t0 = time.time()
        r = run_vqe_at_depth("heisenberg", 1.0, h=3.0, p=p, n_restarts=5)
        elapsed = time.time() - t0
        results.append(r)
        marker = "✅" if r["fidelity"] > 0.5 else "❌" if r["fidelity"] < 0.01 else "⚠️"
        print(
            f"  {p:<4} {r['n_params']:<8} {r['E_exact']:<12.4f} {r['E_vqe']:<12.4f} "
            f"{r['fidelity']:<10.4f} {r['gap_to_gs']:<10.4f} {marker} ({elapsed:.1f}s)"
        )

    # ─── Test B: XY (Δ=0.0) at h=3.0 ─────────────────────────────────
    print("\n" + "═" * 70)
    print("  TEST B: XY Model (Δ=0.0) at h=3.0, N=6")
    print("═" * 70)
    print(f"  {'p':<4} {'Params':<8} {'E_exact':<12} {'E_vqe':<12} {'Fidelity':<10} {'Gap':<10}")
    print(f"  {'-' * 60}")

    for p in [1, 2, 3, 4, 5, 6]:
        t0 = time.time()
        r = run_vqe_at_depth("heisenberg", 0.0, h=3.0, p=p, n_restarts=5)
        elapsed = time.time() - t0
        results.append(r)
        marker = "✅" if r["fidelity"] > 0.5 else "❌" if r["fidelity"] < 0.01 else "⚠️"
        print(
            f"  {p:<4} {r['n_params']:<8} {r['E_exact']:<12.4f} {r['E_vqe']:<12.4f} "
            f"{r['fidelity']:<10.4f} {r['gap_to_gs']:<10.4f} {marker} ({elapsed:.1f}s)"
        )

    # ─── Test C: TFIM at h=1.5 (control — should work at p≥2) ────────
    print("\n" + "═" * 70)
    print("  TEST C: TFIM at h=1.5, N=6 (control — should be ~1.0 at p≥2)")
    print("═" * 70)
    print(f"  {'p':<4} {'Params':<8} {'E_exact':<12} {'E_vqe':<12} {'Fidelity':<10} {'Gap':<10}")
    print(f"  {'-' * 60}")

    for p in [1, 2, 3, 4]:
        t0 = time.time()
        r = run_vqe_at_depth("tfim", None, h=1.5, p=p, n_restarts=5)
        elapsed = time.time() - t0
        results.append(r)
        marker = "✅" if r["fidelity"] > 0.99 else "⚠️" if r["fidelity"] > 0.5 else "❌"
        print(
            f"  {p:<4} {r['n_params']:<8} {r['E_exact']:<12.4f} {r['E_vqe']:<12.4f} "
            f"{r['fidelity']:<10.4f} {r['gap_to_gs']:<10.4f} {marker} ({elapsed:.1f}s)"
        )

    # ─── Test D: Heisenberg at h=0.5 (deep correlated regime) ────────
    print("\n" + "═" * 70)
    print("  TEST D: Heisenberg (Δ=1.0) at h=0.5, N=6 (deep correlated)")
    print("═" * 70)
    print(f"  {'p':<4} {'Params':<8} {'E_exact':<12} {'E_vqe':<12} {'Fidelity':<10} {'Gap':<10}")
    print(f"  {'-' * 60}")

    for p in [2, 4, 6]:
        t0 = time.time()
        r = run_vqe_at_depth("heisenberg", 1.0, h=0.5, p=p, n_restarts=8, maxiter=800)
        elapsed = time.time() - t0
        results.append(r)
        marker = "✅" if r["fidelity"] > 0.5 else "❌" if r["fidelity"] < 0.01 else "⚠️"
        print(
            f"  {p:<4} {r['n_params']:<8} {r['E_exact']:<12.4f} {r['E_vqe']:<12.4f} "
            f"{r['fidelity']:<10.4f} {r['gap_to_gs']:<10.4f} {marker} ({elapsed:.1f}s)"
        )

    return results


def print_conclusions(results: list[dict]) -> None:
    """Print scientific conclusions from the depth scaling test."""
    print("\n" + "═" * 70)
    print("  CONCLUSIONS")
    print("═" * 70)

    # Check if fidelity increases with p for Heisenberg
    heis_h3 = [
        r for r in results if r["model"] == "heisenberg" and r["delta"] == 1.0 and r["h"] == 3.0
    ]
    if heis_h3:
        fids = [(r["p"], r["fidelity"]) for r in heis_h3]
        fids.sort()
        print("\n  Heisenberg (Δ=1.0, h=3.0) fidelity vs depth:")
        for p, f in fids:
            bar = "█" * int(f * 40)
            print(f"    p={p}: {f:.4f} {bar}")

        max_fid = max(f for _, f in fids)
        if max_fid > 0.9:
            print(
                f"\n  ✅ Circuit DOES reach ground state at sufficient depth (max fid={max_fid:.4f})"
            )
            print("     → p=2 failure is CONFIRMED as expressibility limit, not a bug.")
        elif max_fid > 0.1:
            print(f"\n  ⚠️  Fidelity improves with depth (max={max_fid:.4f}) but doesn't reach 1.0")
            print("     → Expressibility limit confirmed, but even p=6 may be insufficient.")
        else:
            print(f"\n  ❌ Fidelity stays near zero even at p=6 (max={max_fid:.4f})")
            print("     → POSSIBLE BUG in create_heisenberg circuit!")

    # TFIM control
    tfim = [r for r in results if r["model"] == "tfim"]
    if tfim:
        tfim_p2 = next((r for r in tfim if r["p"] == 2), None)
        if tfim_p2 and tfim_p2["fidelity"] > 0.99:
            print(f"\n  ✅ TFIM control: fidelity={tfim_p2['fidelity']:.6f} at p=2 (as expected)")
        else:
            print("\n  ❌ TFIM control FAILED — this would indicate a framework bug")


def main():
    parser = argparse.ArgumentParser(description="Depth scaling validation for HVA circuits")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("═" * 70)
    print("  TEST 4: DEPTH SCALING VALIDATION")
    print("  Does increasing p improve fidelity for Heisenberg?")
    print("═" * 70)
    print("  (Temporarily patches MAX_P_LAYERS=8 for testing — no code modified)")

    t0 = time.time()
    results = run_test_suite(verbose=args.verbose)
    total = time.time() - t0

    print_conclusions(results)

    print(f"\n  Total time: {total:.1f}s ({total / 60:.1f} min)")

    # Restore original constant
    constants_mod.MAX_P_LAYERS = ORIGINAL_MAX_P
    print(f"  MAX_P_LAYERS restored to {ORIGINAL_MAX_P}")

    sys.exit(0)


if __name__ == "__main__":
    main()
