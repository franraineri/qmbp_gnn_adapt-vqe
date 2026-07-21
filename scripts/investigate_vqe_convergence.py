#!/usr/bin/env python3
"""Investigate VQE convergence bottleneck at criticality for tfim_longitudinal.

Hypothesis: HVA p=3 on chain_1d N=10 with tfim_longitudinal cannot reach
the ground state near h_c=1.0. This script determines whether the issue is:
  A) Expressibility limit (circuit cannot represent the state at ANY theta)
  B) Optimization landscape (local minima trap the optimizer)
  C) Warm-start failure (theta propagation breaks at the phase transition)

Tests:
  1. Expressibility: brute-force many restarts to find max achievable fidelity
  2. Landscape: compare warm-start vs cold-start vs fine-step propagation
  3. Depth comparison: p=3 vs p=4 vs p=5 at the hardest point (h=1.0)
  4. Optimizer comparison: L-BFGS-B vs COBYLA vs Nelder-Mead

Usage:
    python investigate_vqe_convergence.py
    python investigate_vqe_convergence.py --test 1      # Only expressibility
    python investigate_vqe_convergence.py --test 2      # Only landscape
    python investigate_vqe_convergence.py --test 3      # Only depth comparison
    python investigate_vqe_convergence.py --test 4      # Only optimizer comparison
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.optimize import minimize

from qmbp_simulation import ClassicalSolver, make_lattice
from qmbp_simulation.execution import NoiselessBackend
from qmbp_simulation.models.model_registry import get_model_spec

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

N = 10
TOPOLOGY = "chain_1d"
MODEL = "tfim_longitudinal"
H_CRITICAL = [1.0, 1.1, 1.2, 1.3]
H_EASY = [1.5, 2.0, 3.0]

spec = get_model_spec(MODEL)
solver = ClassicalSolver()
backend = NoiselessBackend()


def get_fidelity(circuit, theta, ground_state):
    """Compute |<psi_exact|psi_vqe>|^2."""
    from qiskit.quantum_info import Statevector, state_fidelity

    sv_vqe = Statevector(circuit.assign_parameters(theta))
    sv_exact = Statevector(ground_state)
    return float(state_fidelity(sv_vqe, sv_exact))


def vqe_single_point(circuit, H, x0, maxiter=1000, method="L-BFGS-B"):
    """Run a single VQE optimization. Returns (energy, theta, n_iters)."""
    n_params = circuit.num_parameters
    res = minimize(
        lambda params: backend.evaluate(circuit, H, params),
        x0,
        method=method,
        bounds=[(-np.pi, np.pi)] * n_params if method == "L-BFGS-B" else None,
        options={"maxiter": maxiter, "ftol": 1e-15}
        if method == "L-BFGS-B"
        else {"maxiter": maxiter},
    )
    return float(res.fun), res.x.copy(), getattr(res, "nit", 0)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: Expressibility (can the circuit reach ground state?)
# ═══════════════════════════════════════════════════════════════════════════════


def test_expressibility(p_layers=3, n_restarts=20):
    """Find max achievable fidelity at each h with many restarts."""
    print("\n" + "=" * 70)
    print(f"  TEST 1: EXPRESSIBILITY (p={p_layers}, {n_restarts} restarts)")
    print("=" * 70)
    print(
        f"  {'h':>5} | {'E_exact':>10} | {'E_best':>10} | {'dE/gap':>8} | {'F_best':>7} | {'gap':>6} | {'time':>5}"
    )
    print(f"  {'-' * 65}")

    lattice_ref = make_lattice(TOPOLOGY, N, J=1.0, h=3.0)
    circuit, _ = spec.create_circuit(N, p_layers, lattice_ref, **spec.circuit_kwargs)
    n_params = circuit.num_parameters
    rng = np.random.default_rng(42)

    results = []
    for h in H_CRITICAL + H_EASY:
        lattice = make_lattice(TOPOLOGY, N, J=1.0, h=h)
        H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
        gt = solver.solve(H, lattice)

        best_energy = float("inf")
        best_theta = None
        t0 = time.time()

        for r in range(n_restarts):
            if r == 0:
                x0 = np.zeros(n_params)
            elif r < 5:
                x0 = rng.uniform(-0.1, 0.1, n_params)
            elif r < 12:
                x0 = rng.uniform(-1.0, 1.0, n_params)
            else:
                x0 = rng.uniform(-np.pi, np.pi, n_params)

            e, theta, _ = vqe_single_point(circuit, H, x0, maxiter=1500)
            if e < best_energy:
                best_energy = e
                best_theta = theta

        elapsed = time.time() - t0
        de_gap = abs(best_energy - gt.ground_energy) / max(gt.gap, 1e-10)
        fid = (
            get_fidelity(circuit, best_theta, gt.ground_state) if gt.ground_state is not None else 0
        )
        results.append({"h": h, "de_gap": de_gap, "fidelity": fid, "gap": gt.gap})
        print(
            f"  {h:>5.2f} | {gt.ground_energy:>10.6f} | {best_energy:>10.6f} | {de_gap:>8.4f} | {fid:>7.4f} | {gt.gap:>6.3f} | {elapsed:>5.1f}s"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: Warm-start propagation (fine steps from easy to hard)
# ═══════════════════════════════════════════════════════════════════════════════


def test_warm_start_propagation(p_layers=3, n_restarts_per_point=5):
    """Propagate theta from h=2.0 down to h=1.0 in fine steps."""
    print("\n" + "=" * 70)
    print(f"  TEST 2: WARM-START PROPAGATION (p={p_layers}, {n_restarts_per_point} restarts/pt)")
    print("=" * 70)

    lattice_ref = make_lattice(TOPOLOGY, N, J=1.0, h=3.0)
    circuit, _ = spec.create_circuit(N, p_layers, lattice_ref, **spec.circuit_kwargs)
    n_params = circuit.num_parameters
    rng = np.random.default_rng(42)

    # Start from h=2.0 (easy)
    lattice = make_lattice(TOPOLOGY, N, J=1.0, h=2.0)
    H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
    x0 = rng.uniform(-0.01, 0.01, n_params)
    e_start, theta_start, _ = vqe_single_point(circuit, H, x0, maxiter=2000)
    current_theta = theta_start.copy()

    # Fine steps: h=2.0 -> 1.0 in 21 steps (Δh=0.05)
    h_path = np.linspace(2.0, 1.0, 21)
    print(f"  Propagating: h=2.0 -> 1.0 in {len(h_path)} steps (Δh=0.05)")
    print(f"  {'h':>5} | {'dE/gap':>8} | {'F':>7} | {'gap':>6} | {'theta_change':>12}")
    print(f"  {'-' * 50}")

    for h in h_path:
        lattice = make_lattice(TOPOLOGY, N, J=1.0, h=h)
        H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
        gt = solver.solve(H, lattice)

        best_e = float("inf")
        best_theta = current_theta.copy()
        for r in range(n_restarts_per_point):
            x0 = current_theta + rng.normal(0, 0.03, n_params) if r > 0 else current_theta.copy()
            x0 = np.clip(x0, -np.pi, np.pi)
            e, theta, _ = vqe_single_point(circuit, H, x0, maxiter=1500)
            if e < best_e:
                best_e = e
                best_theta = theta

        theta_change = np.max(np.abs(best_theta - current_theta))
        current_theta = best_theta.copy()
        de_gap = abs(best_e - gt.ground_energy) / max(gt.gap, 1e-10)
        fid = (
            get_fidelity(circuit, best_theta, gt.ground_state) if gt.ground_state is not None else 0
        )
        print(
            f"  {h:>5.2f} | {de_gap:>8.4f} | {fid:>7.4f} | {gt.gap:>6.3f} | {theta_change:>12.6f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Depth comparison (p=3 vs p=4 vs p=5 at h=1.0)
# ═══════════════════════════════════════════════════════════════════════════════


def test_depth_comparison(h_test=1.0, n_restarts=15):
    """Compare circuit depths at the hardest point."""
    print("\n" + "=" * 70)
    print(f"  TEST 3: DEPTH COMPARISON at h={h_test} ({n_restarts} restarts)")
    print("=" * 70)
    print(
        f"  {'p':>3} | {'n_params':>8} | {'E_best':>10} | {'dE/gap':>8} | {'F_best':>7} | {'time':>5}"
    )
    print(f"  {'-' * 55}")

    lattice = make_lattice(TOPOLOGY, N, J=1.0, h=h_test)
    H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
    gt = solver.solve(H, lattice)
    rng = np.random.default_rng(42)

    for p in [2, 3, 4, 5, 6]:
        lattice_ref = make_lattice(TOPOLOGY, N, J=1.0, h=3.0)
        circuit, _ = spec.create_circuit(N, p, lattice_ref, **spec.circuit_kwargs)
        n_params = circuit.num_parameters

        best_energy = float("inf")
        best_theta = None
        t0 = time.time()

        for r in range(n_restarts):
            x0 = rng.uniform(-np.pi, np.pi, n_params) if r > 2 else rng.uniform(-0.1, 0.1, n_params)
            e, theta, _ = vqe_single_point(circuit, H, x0, maxiter=2000)
            if e < best_energy:
                best_energy = e
                best_theta = theta

        elapsed = time.time() - t0
        de_gap = abs(best_energy - gt.ground_energy) / max(gt.gap, 1e-10)
        fid = (
            get_fidelity(circuit, best_theta, gt.ground_state) if gt.ground_state is not None else 0
        )
        print(
            f"  {p:>3} | {n_params:>8} | {best_energy:>10.6f} | {de_gap:>8.4f} | {fid:>7.4f} | {elapsed:>5.1f}s"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: Optimizer comparison at h=1.0
# ═══════════════════════════════════════════════════════════════════════════════


def test_optimizer_comparison(h_test=1.0, p_layers=3, n_restarts=10):
    """Compare optimizers at the hardest point."""
    print("\n" + "=" * 70)
    print(f"  TEST 4: OPTIMIZER COMPARISON at h={h_test}, p={p_layers}")
    print("=" * 70)

    lattice = make_lattice(TOPOLOGY, N, J=1.0, h=h_test)
    H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
    gt = solver.solve(H, lattice)
    lattice_ref = make_lattice(TOPOLOGY, N, J=1.0, h=3.0)
    circuit, _ = spec.create_circuit(N, p_layers, lattice_ref, **spec.circuit_kwargs)
    n_params = circuit.num_parameters
    rng = np.random.default_rng(42)

    print(f"  {'Method':<15} | {'E_best':>10} | {'dE/gap':>8} | {'F_best':>7} | {'time':>5}")
    print(f"  {'-' * 55}")

    for method in ["L-BFGS-B", "COBYLA", "Nelder-Mead"]:
        best_energy = float("inf")
        best_theta = None
        t0 = time.time()

        for r in range(n_restarts):
            x0 = rng.uniform(-0.5, 0.5, n_params) if r > 0 else rng.uniform(-0.01, 0.01, n_params)
            e, theta, _ = vqe_single_point(circuit, H, x0, maxiter=2000, method=method)
            if e < best_energy:
                best_energy = e
                best_theta = theta

        elapsed = time.time() - t0
        de_gap = abs(best_energy - gt.ground_energy) / max(gt.gap, 1e-10)
        fid = (
            get_fidelity(circuit, best_theta, gt.ground_state) if gt.ground_state is not None else 0
        )
        print(
            f"  {method:<15} | {best_energy:>10.6f} | {de_gap:>8.4f} | {fid:>7.4f} | {elapsed:>5.1f}s"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: Entanglement entropy at criticality
# ═══════════════════════════════════════════════════════════════════════════════


def test_entanglement_at_criticality():
    """Measure the exact ground state entanglement entropy vs h.
    High entanglement at h_c means the circuit needs more depth."""
    print("\n" + "=" * 70)
    print("  TEST 5: ENTANGLEMENT ENTROPY OF EXACT GROUND STATE vs h")
    print("=" * 70)
    print(f"  {'h':>5} | {'S_vN (half-cut)':>15} | {'gap':>6} | {'interpretation':>30}")
    print(f"  {'-' * 65}")

    for h in [0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]:
        lattice = make_lattice(TOPOLOGY, N, J=1.0, h=h)
        H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
        gt = solver.solve(H, lattice)

        if gt.ground_state is None:
            print(f"  {h:>5.2f} | {'N/A':>15} | {gt.gap:>6.3f} | DMRG (no statevector)")
            continue

        # Half-chain entanglement entropy
        psi = gt.ground_state
        dim = 2**N
        n_a = N // 2
        psi_matrix = psi.reshape(2**n_a, 2 ** (N - n_a))
        svd_vals = np.linalg.svd(psi_matrix, compute_uv=False)
        svd_vals = svd_vals[svd_vals > 1e-15]  # Remove numerical zeros
        probs = svd_vals**2
        entropy = -np.sum(probs * np.log2(probs + 1e-30))

        # Interpretation
        max_entropy = n_a  # log2(2^n_a) = n_a for maximally entangled
        if entropy > 0.8 * max_entropy:
            interp = "VOLUME LAW (circuit limited)"
        elif entropy > 2.0:
            interp = "HIGH (needs deep circuit)"
        elif entropy > 1.0:
            interp = "MODERATE (p=3 marginal)"
        else:
            interp = "LOW (p=3 sufficient)"

        print(f"  {h:>5.2f} | {entropy:>15.4f} | {gt.gap:>6.3f} | {interp}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Investigate VQE convergence bottleneck")
    parser.add_argument("--test", type=int, default=None, help="Run specific test (1-5)")
    args = parser.parse_args()

    print(f"System: {MODEL} | {TOPOLOGY} | N={N}")
    print("Question: Why does VQE fail at h=1.0-1.3 with p=3?")

    if args.test is None or args.test == 5:
        test_entanglement_at_criticality()

    if args.test is None or args.test == 1:
        test_expressibility(p_layers=3, n_restarts=20)

    if args.test is None or args.test == 3:
        test_depth_comparison(h_test=1.0, n_restarts=15)

    if args.test is None or args.test == 2:
        test_warm_start_propagation(p_layers=3, n_restarts_per_point=5)

    if args.test is None or args.test == 4:
        test_optimizer_comparison(h_test=1.0, p_layers=3, n_restarts=10)

    print("\n" + "=" * 70)
    print("  INVESTIGATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
