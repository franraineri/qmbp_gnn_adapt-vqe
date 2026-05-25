"""
Experiment: Nevergrad Gradient-Free Optimizer vs L-BFGS-B for VQE

Hypothesis:
    For shallow HVA (p=2, 4 parameters) on 1D TFIM, gradient-based L-BFGS-B
    should outperform gradient-free methods because:
    1. Mele et al. (2026) proves no barren plateaus for shallow + local cost
    2. Only 4 parameters → gradient computation is cheap
    3. Energy landscape is smooth for HVA on TFIM

    Nevergrad might be competitive or better ONLY if:
    - The landscape has many local minima (unlikely for HVA)
    - Shot noise corrupts gradients (hardware scenario)

    This experiment validates our L-BFGS-B choice or identifies cases where
    gradient-free methods are preferable.

Sub-experiments:
    1A: Fair budget comparison at N=6 (budget=1000, 7 h-values, 5 seeds, cold-start)
    1B: Warm-start scenario at N=6 (budget=200, θ from adjacent h-point)
    1C: Scaling to N=10 (budget=2000, 4 h-values, 5 seeds)

References:
    - Nevergrad: https://facebookresearch.github.io/nevergrad/
    - Singh et al. (2025) arXiv:2510.08727 — optimizer benchmarking for VQE
    - arXiv:2402.05227 — evolutionary optimization to avoid barren plateaus

Usage:
    # Run specific sub-experiment
    python scripts/experiments_hamed_v7/experiment_nevergrad.py --sub-experiment 1A

    # Run all sub-experiments
    python scripts/experiments_hamed_v7/experiment_nevergrad.py --sub-experiment all

    # Override parameters
    python scripts/experiments_hamed_v7/experiment_nevergrad.py --sub-experiment 1A --seeds 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiment_utils import (
    ExperimentMetrics,
    SubExperimentResult,
    build_experiment_circuit,
    compute_metrics,
    save_experiment_result,
)
from shared_runners import (
    run_lbfgsb,
    run_lbfgsb_with_restarts,
    run_nevergrad,
)

from src.poc.v6 import (
    ClassicalSolver,
    HamiltonianBuilder,
    make_lattice,
)

# ── Constants ────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent / "results"

# NGOpt excluded — fails with array dimensionality issues on small param spaces
DEFAULT_NG_OPTIMIZERS = ["CMA", "OnePlusOne", "DE", "TwoPointsDE"]

# Sub-experiment default configs
CONFIG_1A = {
    "N": 6,
    "budget": 1000,
    "seeds": 5,
    "h_values": [0.5, 0.8, 1.0, 1.1, 1.25, 1.5, 2.0],
    "optimizers": DEFAULT_NG_OPTIMIZERS,
    "warm_start": False,
}

CONFIG_1B = {
    "N": 6,
    "budget": 200,
    "seeds": 5,
    "h_values": [2.0, 1.5, 1.25, 1.1, 1.0, 0.8, 0.5],  # Descending for warm-start
    "optimizers": DEFAULT_NG_OPTIMIZERS,
    "warm_start": True,
}

CONFIG_1C = {
    "N": 10,
    "budget": 2000,
    "seeds": 5,
    "h_values": [1.0, 1.25, 1.5, 2.0],
    "optimizers": DEFAULT_NG_OPTIMIZERS,
    "warm_start": False,
}


# ── Sub-Experiment Implementations ───────────────────────────────────────────


def run_sub_experiment_1A(args) -> SubExperimentResult:
    """Experiment 1A: Fair budget comparison at N=6 (cold-start).

    Compares L-BFGS-B (5 restarts) vs CMA, OnePlusOne, DE, TwoPointsDE
    with budget=1000 evaluations, 7 h-values, 5 seeds.
    """
    N = args.N
    budget = args.budget
    seeds = args.seeds
    h_values = args.h_values if args.h_values else CONFIG_1A["h_values"]
    optimizers = args.optimizers if args.optimizers else CONFIG_1A["optimizers"]

    print(f"\n{'=' * 70}")
    print("  Sub-experiment 1A: Fair budget comparison (cold-start)")
    print(f"  N={N}, budget={budget}, seeds={seeds}, h-values={h_values}")
    print(f"{'=' * 70}\n")

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    qc, n_params, _ = build_experiment_circuit(N)

    all_metrics: list[ExperimentMetrics] = []
    detailed_results: list[dict] = []

    for h in h_values:
        lattice = make_lattice("chain_1d", N, J=1.0, h=h)
        H = builder.build(lattice)
        exact = solver.solve(H, lattice)
        print(f"  h={h:.2f}: E_exact={exact.ground_energy:.8f}, gap={exact.gap:.6f}")

        for seed in range(seeds):
            np.random.seed(seed + 42)
            initial_guess = np.random.uniform(-0.01, 0.01, n_params)

            # L-BFGS-B (5 restarts, budget matched via restart count)
            params_lb, energy_lb, evals_lb, time_lb = run_lbfgsb_with_restarts(
                qc, H, initial_guess, n_restarts=5, maxiter=1000
            )
            metrics_lb = compute_metrics(
                energy=energy_lb,
                exact_energy=exact.ground_energy,
                gap=exact.gap,
                circuit=qc,
                params=params_lb,
                exact_state=exact.ground_state,
                wall_time_s=time_lb,
                n_evaluations=evals_lb,
                seed=seed + 42,
                h_value=h,
            )
            all_metrics.append(metrics_lb)
            detailed_results.append(
                {
                    "optimizer": "L-BFGS-B (5 restarts)",
                    "h": h,
                    "seed": seed + 42,
                    **_metrics_to_dict(metrics_lb),
                }
            )

            # Nevergrad optimizers
            for ng_name in optimizers:
                try:
                    params_ng, energy_ng, evals_ng, time_ng = run_nevergrad(
                        qc, H, initial_guess.copy(), ng_name, budget=budget
                    )
                    metrics_ng = compute_metrics(
                        energy=energy_ng,
                        exact_energy=exact.ground_energy,
                        gap=exact.gap,
                        circuit=qc,
                        params=params_ng,
                        exact_state=exact.ground_state,
                        wall_time_s=time_ng,
                        n_evaluations=evals_ng,
                        seed=seed + 42,
                        h_value=h,
                    )
                    all_metrics.append(metrics_ng)
                    detailed_results.append(
                        {
                            "optimizer": ng_name,
                            "h": h,
                            "seed": seed + 42,
                            **_metrics_to_dict(metrics_ng),
                        }
                    )
                except Exception as e:
                    print(f"    {ng_name} FAILED at h={h}, seed={seed}: {e}")

        # Print summary for this h-value
        _print_h_summary(h, detailed_results, optimizers)

    # Build result
    summary = _build_optimizer_summary(detailed_results, optimizers)
    result = SubExperimentResult(
        experiment_id="1A",
        technique=1,
        description="Fair budget comparison at N=6 (cold-start)",
        config={
            "N": N,
            "budget": budget,
            "seeds": seeds,
            "h_values": h_values,
            "optimizers": ["L-BFGS-B (5 restarts)"] + optimizers,
        },
        metrics=all_metrics,
        summary=summary,
        success=True,
    )

    path = save_experiment_result(result, RESULTS_DIR, prefix="nevergrad")
    print(f"\n  Result saved: {path.name}")
    return result


def run_sub_experiment_1B(args) -> SubExperimentResult:
    """Experiment 1B: Warm-start scenario at N=6.

    Uses θ_opt from adjacent h-point as initial guess (descending sweep).
    Budget=200 (refinement, not global search).
    Tests whether gradient-free methods help when already near the optimum.
    """
    N = args.N
    budget = args.budget if args.budget != 1000 else CONFIG_1B["budget"]
    seeds = args.seeds
    h_values = args.h_values if args.h_values else CONFIG_1B["h_values"]
    optimizers = args.optimizers if args.optimizers else CONFIG_1B["optimizers"]

    # Ensure descending order for warm-start
    h_values = sorted(h_values, reverse=True)

    print(f"\n{'=' * 70}")
    print("  Sub-experiment 1B: Warm-start scenario")
    print(f"  N={N}, budget={budget}, seeds={seeds}, h-values={h_values} (descending)")
    print(f"{'=' * 70}\n")

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    qc, n_params, _ = build_experiment_circuit(N)

    all_metrics: list[ExperimentMetrics] = []
    detailed_results: list[dict] = []

    for seed in range(seeds):
        np.random.seed(seed + 42)
        initial_theta = np.random.uniform(-0.01, 0.01, n_params)
        prev_theta_lb = initial_theta.copy()
        prev_theta_ng = {opt: initial_theta.copy() for opt in optimizers}

        for h in h_values:
            lattice = make_lattice("chain_1d", N, J=1.0, h=h)
            H = builder.build(lattice)
            exact = solver.solve(H, lattice)

            # L-BFGS-B warm-start (no restarts — just refine from prev θ)
            params_lb, energy_lb, evals_lb, time_lb, _ = run_lbfgsb(
                qc, H, prev_theta_lb, maxiter=budget
            )
            metrics_lb = compute_metrics(
                energy=energy_lb,
                exact_energy=exact.ground_energy,
                gap=exact.gap,
                circuit=qc,
                params=params_lb,
                exact_state=exact.ground_state,
                wall_time_s=time_lb,
                n_evaluations=evals_lb,
                seed=seed + 42,
                h_value=h,
            )
            all_metrics.append(metrics_lb)
            detailed_results.append(
                {
                    "optimizer": "L-BFGS-B (warm-start)",
                    "h": h,
                    "seed": seed + 42,
                    **_metrics_to_dict(metrics_lb),
                }
            )
            prev_theta_lb = params_lb

            # Nevergrad optimizers (warm-start from their own prev θ)
            for ng_name in optimizers:
                try:
                    params_ng, energy_ng, evals_ng, time_ng = run_nevergrad(
                        qc, H, prev_theta_ng[ng_name].copy(), ng_name, budget=budget
                    )
                    metrics_ng = compute_metrics(
                        energy=energy_ng,
                        exact_energy=exact.ground_energy,
                        gap=exact.gap,
                        circuit=qc,
                        params=params_ng,
                        exact_state=exact.ground_state,
                        wall_time_s=time_ng,
                        n_evaluations=evals_ng,
                        seed=seed + 42,
                        h_value=h,
                    )
                    all_metrics.append(metrics_ng)
                    detailed_results.append(
                        {
                            "optimizer": ng_name,
                            "h": h,
                            "seed": seed + 42,
                            **_metrics_to_dict(metrics_ng),
                        }
                    )
                    prev_theta_ng[ng_name] = params_ng
                except Exception as e:
                    print(f"    {ng_name} FAILED at h={h}, seed={seed}: {e}")

        if seed == 0:
            print(f"  Seed {seed + 42} complete (first sweep shown):")
            for h in h_values:
                h_results = [r for r in detailed_results if r["h"] == h and r["seed"] == seed + 42]
                if h_results:
                    lb_r = next((r for r in h_results if "L-BFGS-B" in r["optimizer"]), None)
                    if lb_r:
                        print(f"    h={h:.2f}: L-BFGS-B ΔE={lb_r['energy_error']:.2e}")

    # Build result
    summary = _build_optimizer_summary(
        detailed_results, optimizers, lb_label="L-BFGS-B (warm-start)"
    )
    result = SubExperimentResult(
        experiment_id="1B",
        technique=1,
        description="Warm-start scenario at N=6 (budget=200, θ from adjacent h-point)",
        config={
            "N": N,
            "budget": budget,
            "seeds": seeds,
            "h_values": h_values,
            "optimizers": ["L-BFGS-B (warm-start)"] + optimizers,
            "warm_start": True,
        },
        metrics=all_metrics,
        summary=summary,
        success=True,
    )

    path = save_experiment_result(result, RESULTS_DIR, prefix="nevergrad")
    print(f"\n  Result saved: {path.name}")
    return result


def run_sub_experiment_1C(args) -> SubExperimentResult:
    """Experiment 1C: Scaling to N=10 (cold-start, budget=2000).

    Tests whether higher dimensionality (8 params) changes the optimizer ranking.
    """
    N = 10  # Always N=10 for this sub-experiment
    budget = args.budget if args.budget != 1000 else CONFIG_1C["budget"]
    seeds = args.seeds
    h_values = args.h_values if args.h_values else CONFIG_1C["h_values"]
    optimizers = args.optimizers if args.optimizers else CONFIG_1C["optimizers"]

    print(f"\n{'=' * 70}")
    print("  Sub-experiment 1C: Scaling to N=10 (cold-start)")
    print(f"  N={N}, budget={budget}, seeds={seeds}, h-values={h_values}")
    print(f"{'=' * 70}\n")

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    qc, n_params, _ = build_experiment_circuit(N)
    print(f"  Circuit: {n_params} parameters")

    all_metrics: list[ExperimentMetrics] = []
    detailed_results: list[dict] = []

    for h in h_values:
        lattice = make_lattice("chain_1d", N, J=1.0, h=h)
        H = builder.build(lattice)
        exact = solver.solve(H, lattice)
        print(f"  h={h:.2f}: E_exact={exact.ground_energy:.8f}, gap={exact.gap:.6f}")

        for seed in range(seeds):
            np.random.seed(seed + 42)
            initial_guess = np.random.uniform(-0.01, 0.01, n_params)

            # L-BFGS-B (5 restarts)
            params_lb, energy_lb, evals_lb, time_lb = run_lbfgsb_with_restarts(
                qc, H, initial_guess, n_restarts=5, maxiter=1000
            )
            metrics_lb = compute_metrics(
                energy=energy_lb,
                exact_energy=exact.ground_energy,
                gap=exact.gap,
                circuit=qc,
                params=params_lb,
                exact_state=exact.ground_state,
                wall_time_s=time_lb,
                n_evaluations=evals_lb,
                seed=seed + 42,
                h_value=h,
            )
            all_metrics.append(metrics_lb)
            detailed_results.append(
                {
                    "optimizer": "L-BFGS-B (5 restarts)",
                    "h": h,
                    "seed": seed + 42,
                    **_metrics_to_dict(metrics_lb),
                }
            )

            # Nevergrad optimizers
            for ng_name in optimizers:
                try:
                    params_ng, energy_ng, evals_ng, time_ng = run_nevergrad(
                        qc, H, initial_guess.copy(), ng_name, budget=budget
                    )
                    metrics_ng = compute_metrics(
                        energy=energy_ng,
                        exact_energy=exact.ground_energy,
                        gap=exact.gap,
                        circuit=qc,
                        params=params_ng,
                        exact_state=exact.ground_state,
                        wall_time_s=time_ng,
                        n_evaluations=evals_ng,
                        seed=seed + 42,
                        h_value=h,
                    )
                    all_metrics.append(metrics_ng)
                    detailed_results.append(
                        {
                            "optimizer": ng_name,
                            "h": h,
                            "seed": seed + 42,
                            **_metrics_to_dict(metrics_ng),
                        }
                    )
                except Exception as e:
                    print(f"    {ng_name} FAILED at h={h}, seed={seed}: {e}")

        _print_h_summary(h, detailed_results, optimizers)

    # Build result
    summary = _build_optimizer_summary(detailed_results, optimizers)
    result = SubExperimentResult(
        experiment_id="1C",
        technique=1,
        description="Scaling to N=10 (cold-start, budget=2000)",
        config={
            "N": N,
            "budget": budget,
            "seeds": seeds,
            "h_values": h_values,
            "optimizers": ["L-BFGS-B (5 restarts)"] + optimizers,
        },
        metrics=all_metrics,
        summary=summary,
        success=True,
    )

    path = save_experiment_result(result, RESULTS_DIR, prefix="nevergrad")
    print(f"\n  Result saved: {path.name}")
    return result


# ── Helpers ──────────────────────────────────────────────────────────────────


def _metrics_to_dict(m: ExperimentMetrics) -> dict:
    """Convert ExperimentMetrics to a flat dict for detailed results."""
    return {
        "energy": m.energy,
        "energy_error": m.energy_error,
        "relative_error": m.relative_error,
        "fidelity": m.fidelity,
        "phase_label": m.phase_label,
        "phase_correct": m.phase_correct,
        "wall_time_s": m.wall_time_s,
        "n_evaluations": m.n_evaluations,
    }


def _print_h_summary(h: float, detailed_results: list[dict], optimizers: list[str]):
    """Print a compact summary for one h-value."""
    h_results = [r for r in detailed_results if r["h"] == h]
    if not h_results:
        return

    lb_results = [r for r in h_results if "L-BFGS-B" in r["optimizer"]]
    if lb_results:
        avg_err = np.mean([r["energy_error"] for r in lb_results])
        print(f"    L-BFGS-B avg ΔE={avg_err:.2e}")

    for opt in optimizers:
        opt_results = [r for r in h_results if r["optimizer"] == opt]
        if opt_results:
            avg_err = np.mean([r["energy_error"] for r in opt_results])
            print(f"    {opt:12s} avg ΔE={avg_err:.2e}")


def _build_optimizer_summary(
    detailed_results: list[dict],
    ng_optimizers: list[str],
    lb_label: str = "L-BFGS-B (5 restarts)",
) -> dict:
    """Build summary comparing optimizers across all h-values and seeds."""
    all_optimizers = [lb_label] + ng_optimizers
    summary = {"mean_error_by_optimizer": {}, "mean_fidelity_by_optimizer": {}}

    for opt in all_optimizers:
        opt_results = [r for r in detailed_results if r["optimizer"] == opt]
        if opt_results:
            summary["mean_error_by_optimizer"][opt] = float(
                np.mean([r["energy_error"] for r in opt_results])
            )
            fids = [r["fidelity"] for r in opt_results if r.get("fidelity") is not None]
            if fids:
                summary["mean_fidelity_by_optimizer"][opt] = float(np.mean(fids))

    # Determine winner
    if summary["mean_error_by_optimizer"]:
        best_opt = min(
            summary["mean_error_by_optimizer"], key=summary["mean_error_by_optimizer"].get
        )
        summary["best_optimizer"] = best_opt
        summary["conclusion"] = f"{best_opt} achieves lowest mean ΔE across all h-values and seeds"

    return summary


# ── CLI & Dispatch ───────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Nevergrad vs L-BFGS-B comparison (Technique 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sub-experiments:
  1A: Fair budget comparison at N=6 (budget=1000, 7 h-values, 5 seeds)
  1B: Warm-start scenario at N=6 (budget=200, θ from adjacent h-point)
  1C: Scaling to N=10 (budget=2000, 4 h-values, 5 seeds)
""",
    )
    parser.add_argument(
        "--sub-experiment",
        type=str,
        default="all",
        choices=["1A", "1B", "1C", "all"],
        help="Sub-experiment to run (default: all)",
    )
    parser.add_argument(
        "--N", type=int, default=6, help="Number of qubits (overrides sub-experiment default)"
    )
    parser.add_argument("--budget", type=int, default=1000, help="Nevergrad eval budget")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument(
        "--h-values",
        type=float,
        nargs="+",
        default=None,
        help="Override h-values (default: per sub-experiment)",
    )
    parser.add_argument(
        "--optimizers",
        type=str,
        nargs="+",
        default=None,
        help="Nevergrad optimizers to test (default: CMA OnePlusOne DE TwoPointsDE)",
    )
    return parser.parse_args()


DISPATCH = {
    "1A": run_sub_experiment_1A,
    "1B": run_sub_experiment_1B,
    "1C": run_sub_experiment_1C,
}


def main():
    """Main entry point with sub-experiment dispatch."""
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
                        technique=1,
                        description=f"Failed: {e}",
                        success=False,
                        error=str(e),
                    )
                )
        # Exit with failure if any sub-experiment failed
        if any(not r.success for r in results):
            sys.exit(1)
    else:
        fn = DISPATCH[args.sub_experiment]
        result = fn(args)
        if not result.success:
            sys.exit(1)


if __name__ == "__main__":
    main()
