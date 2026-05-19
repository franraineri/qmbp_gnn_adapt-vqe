"""
Experiment: Qiskit Aer MPS Simulator for Scaling VQE Beyond N=10

Sub-experiments:
    3A: N=6 accuracy validation (chi sweep, compare vs statevector)
    3B: N=10 accuracy validation (same protocol)
    3C: N=20 MPS-only VQE (chi=[64,128,256,512], DMRG reference)
    3D: N=30 stretch goal (chi=256, h=[1.5,2.0] only)
    3E: Critical region stress test (N=20, chi up to 1024, h near 1.0)

References:
    - Qiskit Aer MPS method
    - Schollwock (2011) — DMRG/MPS review

Usage:
    python scripts/experiments_hamed_v7/experiment_mps_simulation.py --sub-experiment 3A
    python scripts/experiments_hamed_v7/experiment_mps_simulation.py --sub-experiment all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from experiment_utils import (
    SubExperimentResult,
    compute_metrics,
    evaluate_energy_statevector,
    save_experiment_result,
)
from shared_runners import get_exact_solution, setup_experiment

RESULTS_DIR = Path(__file__).parent / "results"


# ── MPS Helpers ──────────────────────────────────────────────────────────────


def create_mps_backend(max_bond_dimension: int = 256):
    """Create AerSimulator with MPS method."""
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        print("ERROR: qiskit-aer not installed.")
        sys.exit(1)

    return AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=max_bond_dimension,
        matrix_product_state_truncation_threshold=1e-12,
    )


def evaluate_energy_mps(circuit, hamiltonian, params, backend):
    """Evaluate energy using MPS simulator via statevector save + Pauli expectation.

    Uses AerSimulator in MPS mode with save_statevector to get the state,
    then computes ⟨H⟩ directly. Much faster than BackendEstimatorV2 for VQE loops.
    """
    from qiskit.quantum_info import Statevector

    # Use statevector-level computation via MPS backend
    # This avoids the shot-based overhead of BackendEstimatorV2
    from qiskit_aer import AerSimulator

    bound = circuit.assign_parameters(params)
    # Run with statevector output via MPS method
    sim = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=backend.options.get(
            "matrix_product_state_max_bond_dimension", 256
        ),
    )
    bound_with_save = bound.copy()
    bound_with_save.save_statevector()
    result = sim.run(bound_with_save).result()
    sv = Statevector(result.get_statevector())
    return float(sv.expectation_value(hamiltonian).real)


def run_vqe_mps(circuit, hamiltonian, initial_guess, backend, maxiter=200):
    """Run VQE using MPS simulator with COBYLA."""
    from scipy.optimize import minimize

    eval_count = [0]

    def cost_fn(params):
        eval_count[0] += 1
        return evaluate_energy_mps(circuit, hamiltonian, params, backend)

    t0 = time.time()
    result = minimize(
        cost_fn,
        initial_guess,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.5},
    )
    elapsed = time.time() - t0

    return result.x, result.fun, eval_count[0], elapsed


# ── Sub-experiment 3A ────────────────────────────────────────────────────────


def run_sub_experiment_3A(args) -> SubExperimentResult:
    """MPS accuracy validation at N=6 (compare vs statevector).

    Protocol: Run VQE on statevector to get θ_opt, then evaluate the SAME θ
    on both statevector and MPS backends. This tests MPS simulator accuracy,
    not optimizer convergence.
    """
    N = 6
    chi_values = args.chi or [32, 64, 128, 256]
    h_values = args.h_values or [0.5, 1.0, 1.5, 2.0]

    print(f"\n{'=' * 70}")
    print(f"  Sub-experiment 3A: MPS accuracy validation (N={N})")
    print("  Protocol: same θ evaluated on SV vs MPS")
    print(f"  chi={chi_values}, h={h_values}")
    print(f"{'=' * 70}\n")

    env = setup_experiment(N)
    qc = env["circuit"]
    all_metrics = []
    detailed = []

    for h in h_values:
        sol = get_exact_solution(env["builder"], env["solver"], N, h)
        H = sol["hamiltonian"]
        exact = sol["exact"]

        # Run VQE on statevector to get reference θ_opt
        np.random.seed(42)
        initial_guess = np.random.uniform(-0.01, 0.01, env["n_params"])

        from shared_runners import run_lbfgsb_with_restarts

        theta_opt, energy_sv, evals_sv, time_sv = run_lbfgsb_with_restarts(
            qc,
            H,
            initial_guess,
            n_restarts=3,
            maxiter=500,
        )
        # Also get exact SV energy at theta_opt (no optimizer noise)
        energy_sv_exact = evaluate_energy_statevector(qc, H, theta_opt)
        err_sv = abs(energy_sv_exact - exact.ground_energy)
        print(f"  h={h:.1f}: SV E(θ_opt)={energy_sv_exact:.8f}, ΔE={err_sv:.2e}")

        # Evaluate the SAME θ_opt on MPS with different chi
        for chi in chi_values:
            backend = create_mps_backend(chi)
            try:
                t0 = time.time()
                energy_mps = evaluate_energy_mps(qc, H, theta_opt, backend)
                time_mps = time.time() - t0

                diff_vs_sv = abs(energy_mps - energy_sv_exact)
                err_mps = abs(energy_mps - exact.ground_energy)

                m = compute_metrics(
                    energy=energy_mps,
                    exact_energy=exact.ground_energy,
                    gap=exact.gap,
                    wall_time_s=time_mps,
                    n_evaluations=1,
                    seed=42,
                    h_value=h,
                )
                all_metrics.append(m)
                detailed.append(
                    {
                        "h": h,
                        "chi": chi,
                        "energy_sv": energy_sv_exact,
                        "energy_mps": energy_mps,
                        "diff_vs_sv": diff_vs_sv,
                        "error_mps": err_mps,
                        "time_mps": time_mps,
                        "validated": diff_vs_sv < 1e-3,
                    }
                )
                status = "✓" if diff_vs_sv < 1e-3 else "✗"
                print(
                    f"    chi={chi:3d}: MPS E={energy_mps:.8f}, "
                    f"|MPS-SV|={diff_vs_sv:.2e} {status}, t={time_mps:.2f}s"
                )
            except Exception as e:
                print(f"    chi={chi:3d}: FAILED — {e}")
                detailed.append({"h": h, "chi": chi, "error": str(e), "validated": False})
                print(f"    chi={chi:3d}: FAILED — {e}")
                detailed.append(
                    {
                        "h": h,
                        "chi": chi,
                        "error": str(e),
                        "validated": False,
                    }
                )

    n_validated = sum(1 for d in detailed if d.get("validated", False))
    summary = {
        "n_validated": n_validated,
        "total_tests": len(detailed),
        "detailed": detailed,
        "conclusion": f"{n_validated}/{len(detailed)} MPS results within 1e-3 of statevector (same-θ protocol)",
    }

    result = SubExperimentResult(
        experiment_id="3A",
        technique=3,
        description="MPS accuracy validation (N=6, compare vs statevector)",
        config={"N": N, "chi_values": chi_values, "h_values": h_values},
        metrics=all_metrics,
        summary=summary,
        success=True,
    )
    path = save_experiment_result(result, RESULTS_DIR, prefix="mps")
    print(f"\n  Result saved: {path.name}")
    return result


# ── Sub-experiment 3B ────────────────────────────────────────────────────────


def run_sub_experiment_3B(args) -> SubExperimentResult:
    """MPS accuracy validation at N=10 (same protocol as 3A)."""
    print("\n  [3B: MPS accuracy at N=10, same-θ protocol]")

    N = 10
    chi_values = args.chi or [32, 64, 128, 256]
    h_values = args.h_values or [0.5, 1.0, 1.5, 2.0]

    print(f"\n{'=' * 70}")
    print(f"  Sub-experiment 3B: MPS accuracy validation (N={N})")
    print("  Protocol: same θ evaluated on SV vs MPS")
    print(f"  chi={chi_values}, h={h_values}")
    print(f"{'=' * 70}\n")

    from shared_runners import clear_exact_cache

    clear_exact_cache()

    env = setup_experiment(N)
    qc = env["circuit"]
    all_metrics = []
    detailed = []

    for h in h_values:
        sol = get_exact_solution(env["builder"], env["solver"], N, h)
        H = sol["hamiltonian"]
        exact = sol["exact"]

        np.random.seed(42)
        initial_guess = np.random.uniform(-0.01, 0.01, env["n_params"])

        # VQE on statevector to get reference θ_opt
        from shared_runners import run_lbfgsb_with_restarts

        theta_opt, energy_sv, _, _ = run_lbfgsb_with_restarts(
            qc,
            H,
            initial_guess,
            n_restarts=3,
            maxiter=500,
        )
        energy_sv_exact = evaluate_energy_statevector(qc, H, theta_opt)
        err_sv = abs(energy_sv_exact - exact.ground_energy)
        print(f"  h={h:.1f}: SV E(θ_opt)={energy_sv_exact:.8f}, ΔE={err_sv:.2e}")

        # Evaluate same θ_opt on MPS
        for chi in chi_values:
            backend = create_mps_backend(chi)
            try:
                t0 = time.time()
                energy_mps = evaluate_energy_mps(qc, H, theta_opt, backend)
                time_mps = time.time() - t0

                diff_vs_sv = abs(energy_mps - energy_sv_exact)

                m = compute_metrics(
                    energy=energy_mps,
                    exact_energy=exact.ground_energy,
                    gap=exact.gap,
                    wall_time_s=time_mps,
                    n_evaluations=1,
                    seed=42,
                    h_value=h,
                )
                all_metrics.append(m)
                detailed.append(
                    {
                        "h": h,
                        "chi": chi,
                        "energy_sv": energy_sv_exact,
                        "energy_mps": energy_mps,
                        "diff_vs_sv": diff_vs_sv,
                        "time_mps": time_mps,
                        "validated": diff_vs_sv < 1e-3,
                    }
                )
                status = "✓" if diff_vs_sv < 1e-3 else "✗"
                print(f"    chi={chi:3d}: |MPS-SV|={diff_vs_sv:.2e} {status}, t={time_mps:.2f}s")
            except Exception as e:
                print(f"    chi={chi:3d}: FAILED — {e}")
                detailed.append({"h": h, "chi": chi, "error": str(e), "validated": False})

    n_validated = sum(1 for d in detailed if d.get("validated", False))
    summary = {
        "n_validated": n_validated,
        "total_tests": len(detailed),
        "detailed": detailed,
        "conclusion": f"{n_validated}/{len(detailed)} validated at N=10",
    }

    result = SubExperimentResult(
        experiment_id="3B",
        technique=3,
        description="MPS accuracy validation (N=10)",
        config={"N": N, "chi_values": chi_values, "h_values": h_values},
        metrics=all_metrics,
        summary=summary,
        success=True,
    )
    path = save_experiment_result(result, RESULTS_DIR, prefix="mps")
    print(f"\n  Result saved: {path.name}")
    return result


# ── Sub-experiment 3C ────────────────────────────────────────────────────────


def run_sub_experiment_3C(args) -> SubExperimentResult:
    """MPS-only VQE at N=20 (DMRG reference, no statevector comparison).

    Only tests h≥1.0 (paramagnetic phase) because HVA p=2 + |+⟩^N cannot
    express the ferromagnetic ground state (h<1.0) — known physics limit.
    Uses warm-start descending sweep for better VQE convergence.
    """
    N = 20
    chi_values = args.chi or [64, 128, 256]
    # Only paramagnetic phase — HVA p=2 cannot reach ferromagnetic GS
    h_values = args.h_values or [1.0, 1.25, 1.5, 2.0]

    print(f"\n{'=' * 70}")
    print(f"  Sub-experiment 3C: MPS-only VQE (N={N})")
    print(f"  chi={chi_values}, h={h_values}")
    print("  Reference: DMRG ground truth")
    print("  Note: h≥1.0 only (HVA p=2 physics limit for h<1.0)")
    print(f"{'=' * 70}\n")

    env = setup_experiment(N)
    qc = env["circuit"]
    all_metrics = []
    detailed = []

    # Use descending sweep warm-start for better convergence
    h_desc = sorted(h_values, reverse=True)
    np.random.seed(42)
    prev_theta = np.random.uniform(-0.01, 0.01, env["n_params"])

    for h in h_desc:
        sol = get_exact_solution(env["builder"], env["solver"], N, h)
        H = sol["hamiltonian"]
        exact = sol["exact"]
        gap = exact.gap if exact.gap > 0 else 0.1  # fallback if DMRG gap fails
        print(f"  h={h:.2f}: E_dmrg={exact.ground_energy:.8f}, gap={gap:.4f}")

        for chi in chi_values:
            backend = create_mps_backend(chi)
            try:
                theta_mps, energy_mps, evals_mps, time_mps = run_vqe_mps(
                    qc,
                    H,
                    prev_theta.copy(),
                    backend,
                    maxiter=500,
                )
                err_mps = abs(energy_mps - exact.ground_energy)
                rel_err = err_mps / gap

                m = compute_metrics(
                    energy=energy_mps,
                    exact_energy=exact.ground_energy,
                    gap=gap,
                    wall_time_s=time_mps,
                    n_evaluations=evals_mps,
                    seed=42,
                    h_value=h,
                )
                all_metrics.append(m)
                detailed.append(
                    {
                        "h": h,
                        "chi": chi,
                        "energy_mps": energy_mps,
                        "error": err_mps,
                        "relative_error": rel_err,
                        "time_mps": time_mps,
                        "evals": evals_mps,
                    }
                )
                print(
                    f"    chi={chi:3d}: E={energy_mps:.8f}, ΔE/gap={rel_err:.4f}, t={time_mps:.1f}s"
                )

                # Use best chi result as warm-start for next h
                if chi == chi_values[0]:
                    prev_theta = theta_mps.copy()
            except Exception as e:
                print(f"    chi={chi:3d}: FAILED — {e}")
                detailed.append({"h": h, "chi": chi, "error": str(e)})

    summary = {"detailed": detailed, "conclusion": "MPS VQE scaling to N=20"}

    result = SubExperimentResult(
        experiment_id="3C",
        technique=3,
        description="MPS-only VQE (N=20, DMRG reference)",
        config={"N": N, "chi_values": chi_values, "h_values": h_values},
        metrics=all_metrics,
        summary=summary,
        success=True,
    )
    path = save_experiment_result(result, RESULTS_DIR, prefix="mps")
    print(f"\n  Result saved: {path.name}")
    return result


# ── Sub-experiment 3D ────────────────────────────────────────────────────────


def run_sub_experiment_3D(args) -> SubExperimentResult:
    """MPS VQE at N=30 (stretch goal, paramagnetic phase only).

    Uses warm-start descending sweep and higher maxiter for convergence.
    """
    N = 30
    chi_values = [256]
    h_values = [2.0, 1.5]  # Descending order for warm-start

    print(f"\n{'=' * 70}")
    print(f"  Sub-experiment 3D: MPS VQE at N={N} (stretch)")
    print(f"  chi={chi_values}, h={h_values} (paramagnetic only, descending)")
    print(f"{'=' * 70}\n")

    env = setup_experiment(N)
    qc = env["circuit"]
    all_metrics = []
    detailed = []

    np.random.seed(42)
    prev_theta = np.random.uniform(-0.01, 0.01, env["n_params"])

    for h in h_values:
        sol = get_exact_solution(env["builder"], env["solver"], N, h)
        H = sol["hamiltonian"]
        exact = sol["exact"]
        gap = exact.gap if exact.gap > 0 else 0.1
        print(f"  h={h:.1f}: E_dmrg={exact.ground_energy:.8f}, gap={gap:.4f}")

        for chi in chi_values:
            backend = create_mps_backend(chi)
            try:
                theta_mps, energy_mps, evals_mps, time_mps = run_vqe_mps(
                    qc,
                    H,
                    prev_theta.copy(),
                    backend,
                    maxiter=500,
                )
                err_mps = abs(energy_mps - exact.ground_energy)
                rel_err = err_mps / gap

                m = compute_metrics(
                    energy=energy_mps,
                    exact_energy=exact.ground_energy,
                    gap=gap,
                    wall_time_s=time_mps,
                    n_evaluations=evals_mps,
                    seed=42,
                    h_value=h,
                )
                all_metrics.append(m)
                detailed.append(
                    {
                        "h": h,
                        "chi": chi,
                        "energy_mps": energy_mps,
                        "error": err_mps,
                        "relative_error": rel_err,
                        "time_mps": time_mps,
                    }
                )
                print(f"    chi={chi}: E={energy_mps:.8f}, ΔE/gap={rel_err:.4f}, t={time_mps:.1f}s")
                prev_theta = theta_mps.copy()
            except Exception as e:
                print(f"    chi={chi}: FAILED — {e}")
                detailed.append({"h": h, "chi": chi, "error": str(e)})

    summary = {"detailed": detailed, "conclusion": "MPS VQE at N=30 (stretch)"}

    result = SubExperimentResult(
        experiment_id="3D",
        technique=3,
        description="MPS VQE at N=30 (paramagnetic phase only)",
        config={"N": N, "chi_values": chi_values, "h_values": h_values},
        metrics=all_metrics,
        summary=summary,
        success=True,
    )
    path = save_experiment_result(result, RESULTS_DIR, prefix="mps")
    print(f"\n  Result saved: {path.name}")
    return result


# ── Sub-experiment 3E ────────────────────────────────────────────────────────


def run_sub_experiment_3E(args) -> SubExperimentResult:
    """Critical region stress test (N=20, high chi, h near 1.0).

    Tests MPS VQE convergence near the phase transition where entanglement
    grows logarithmically. Uses warm-start from h=1.1 descending to h=0.9.
    """
    N = 20
    chi_values = args.chi or [128, 256, 512]
    h_values = [1.1, 1.05, 1.0, 0.95, 0.9]  # Descending for warm-start

    print(f"\n{'=' * 70}")
    print(f"  Sub-experiment 3E: Critical region stress test (N={N})")
    print(f"  chi={chi_values}, h={h_values} (descending)")
    print(f"{'=' * 70}\n")

    from shared_runners import clear_exact_cache

    clear_exact_cache()

    env = setup_experiment(N)
    qc = env["circuit"]
    all_metrics = []
    detailed = []

    np.random.seed(42)
    prev_theta = np.random.uniform(-0.01, 0.01, env["n_params"])

    for h in h_values:
        sol = get_exact_solution(env["builder"], env["solver"], N, h)
        H = sol["hamiltonian"]
        exact = sol["exact"]
        gap = exact.gap if exact.gap > 0 else 0.05  # small fallback near criticality
        print(f"  h={h:.2f}: E_dmrg={exact.ground_energy:.8f}, gap={gap:.6f}")

        for chi in chi_values:
            backend = create_mps_backend(chi)
            try:
                theta_mps, energy_mps, evals_mps, time_mps = run_vqe_mps(
                    qc,
                    H,
                    prev_theta.copy(),
                    backend,
                    maxiter=500,
                )
                err_mps = abs(energy_mps - exact.ground_energy)
                rel_err = err_mps / gap

                m = compute_metrics(
                    energy=energy_mps,
                    exact_energy=exact.ground_energy,
                    gap=gap,
                    wall_time_s=time_mps,
                    n_evaluations=evals_mps,
                    seed=42,
                    h_value=h,
                )
                all_metrics.append(m)
                detailed.append(
                    {
                        "h": h,
                        "chi": chi,
                        "energy_mps": energy_mps,
                        "error": err_mps,
                        "relative_error": rel_err,
                        "time_mps": time_mps,
                    }
                )
                print(f"    chi={chi:4d}: ΔE/gap={rel_err:.4f}, t={time_mps:.1f}s")

                # Use first chi result as warm-start
                if chi == chi_values[0]:
                    prev_theta = theta_mps.copy()
            except Exception as e:
                print(f"    chi={chi:4d}: FAILED — {e}")
                detailed.append({"h": h, "chi": chi, "error": str(e)})

    summary = {"detailed": detailed, "conclusion": "Critical region MPS stress test"}

    result = SubExperimentResult(
        experiment_id="3E",
        technique=3,
        description="Critical region stress test (N=20, chi up to 1024)",
        config={"N": N, "chi_values": chi_values, "h_values": h_values},
        metrics=all_metrics,
        summary=summary,
        success=True,
    )
    path = save_experiment_result(result, RESULTS_DIR, prefix="mps")
    print(f"\n  Result saved: {path.name}")
    return result


# ── CLI & Dispatch ───────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MPS simulator scaling experiments (Technique 3)",
    )
    parser.add_argument(
        "--sub-experiment", type=str, default="all", choices=["3A", "3B", "3C", "3D", "3E", "all"]
    )
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--chi", type=int, nargs="+", default=None)
    parser.add_argument("--h-values", type=float, nargs="+", default=None)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--compare-sv", action="store_true")
    return parser.parse_args()


DISPATCH = {
    "3A": run_sub_experiment_3A,
    "3B": run_sub_experiment_3B,
    "3C": run_sub_experiment_3C,
    "3D": run_sub_experiment_3D,
    "3E": run_sub_experiment_3E,
}


def main():
    args = parse_args()

    if args.sub_experiment == "all":
        results = []
        for sub_id, fn in DISPATCH.items():
            try:
                result = fn(args)
                results.append(result)
            except Exception as e:
                print(f"\n  ERROR in {sub_id}: {e}")
                results.append(
                    SubExperimentResult(
                        experiment_id=sub_id,
                        technique=3,
                        description=f"Failed: {e}",
                        success=False,
                        error=str(e),
                    )
                )
        if any(not r.success for r in results):
            sys.exit(1)
    else:
        fn = DISPATCH[args.sub_experiment]
        result = fn(args)
        if not result.success:
            sys.exit(1)


if __name__ == "__main__":
    main()
