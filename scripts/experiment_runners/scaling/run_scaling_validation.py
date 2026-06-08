#!/usr/bin/env python3
"""MPS Scaling Validation — DMRG ground truth + MPS-VQE at N>30.

Validates that MPS-based VQE converges to DMRG ground truth across
a range of N values and h-points for 1D TFIM.

Phases:
    1. DMRG ground truth: Compute E₀ and gap for each h-point via ClassicalSolver
    2. MPS-VQE: Descending sweep with MPSBackend, compute ΔE/gap per h-point

Success criterion: ΔE/gap < 5% for all h-points in valid regime.

Usage:
    python scripts/experiment_runners/scaling/run_scaling_validation.py \\
        --n 40 --topology chain_1d --strategy aer_mps --precision 0.005
    python scripts/experiment_runners/scaling/run_scaling_validation.py \\
        --n 50 --strategy tenpy_exact --h-values 3.0 2.5 2.0

Output:
    JSON file in --output-dir with metadata, timing, and per-h results.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running from project root
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np

from qmbp_simulation import (
    ClassicalSolver,
    HamiltonianBuilder,
    HVACircuitBuilder,
    VQEOptimizer,
    make_lattice,
)
from qmbp_simulation.execution import MPSBackend
from qmbp_simulation.models import VQEConfig
from qmbp_simulation.utils.helpers import json_dump

logger = logging.getLogger(__name__)


def _get_version(package: str) -> str:
    """Safely get package version, returning 'not installed' on failure."""
    try:
        import importlib.metadata

        return importlib.metadata.version(package)
    except Exception:
        return "not installed"


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="MPS Scaling Validation: DMRG ground truth + MPS-VQE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=40, help="System size (qubits)")
    parser.add_argument("--topology", type=str, default="chain_1d", help="Lattice topology")
    parser.add_argument("--p-layers", type=int, default=1, help="HVA circuit depth")
    parser.add_argument(
        "--h-values",
        type=float,
        nargs="+",
        default=None,
        help="Transverse field values (descending). Auto-computed if not given.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds")
    parser.add_argument(
        "--strategy",
        type=str,
        default="aer_mps",
        choices=["aer_mps", "tenpy_exact"],
        help="MPS backend strategy",
    )
    parser.add_argument("--chi-max", type=int, default=64, help="MPS bond dimension")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/scaling",
        help="Output directory for results JSON",
    )
    parser.add_argument(
        "--precision",
        type=float,
        default=0.005,
        help="Precision for aer_mps strategy (controls shot budget)",
    )
    return parser


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: DMRG Ground Truth
# ═══════════════════════════════════════════════════════════════════════════════


def phase1_dmrg_ground_truth(n: int, topology: str, h_values: list[float]) -> list[dict]:
    """Compute DMRG ground truth for each h-point.

    Returns list of dicts with h, ground_energy, gap, and local observables.
    """
    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    results = []

    for h in h_values:
        t0 = time.time()
        lattice = make_lattice(topology, n, J=1.0, h=h)
        H = builder.build(lattice)
        gt = solver.solve(H, lattice, method="dmrg")
        elapsed = time.time() - t0

        results.append(
            {
                "h": h,
                "ground_energy": gt.ground_energy,
                "gap": gt.gap,
                "mag_x": gt.mag_x,
                "corr_zz": gt.corr_zz,
                "per_site_mag_x": gt.per_site_mag_x.tolist()
                if gt.per_site_mag_x is not None
                else None,
                "per_bond_corr_zz": gt.per_bond_corr_zz.tolist()
                if gt.per_bond_corr_zz is not None
                else None,
                "time_s": elapsed,
            }
        )
        logger.info(
            f"  DMRG h={h:.3f}: E₀={gt.ground_energy:.8f}, "
            f"gap={gt.gap:.4f}, ⟨X⟩={gt.mag_x:.4f}, ⟨ZZ⟩={gt.corr_zz:.4f}, "
            f"time={elapsed:.1f}s"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: MPS-VQE Descending Sweep
# ═══════════════════════════════════════════════════════════════════════════════


def phase2_mps_vqe(
    n: int,
    topology: str,
    p_layers: int,
    h_values: list[float],
    strategy: str,
    chi_max: int,
    precision: float,
    seed: int,
    dmrg_data: list[dict],
) -> list[dict]:
    """Run MPS-VQE descending sweep and compute ΔE/gap vs DMRG.

    Returns list of per-h result dicts.
    """
    builder = HamiltonianBuilder()
    hva = HVACircuitBuilder()

    # Determine optimizer method based on strategy
    method = "COBYLA" if strategy == "aer_mps" else "L-BFGS-B"

    backend = MPSBackend(strategy=strategy, chi_max=chi_max, precision=precision, seed=seed)
    config = VQEConfig(
        method=method,
        p_layers=p_layers,
        n_restarts=3,
        maxiter=500,
        enable_callbacks=False,
    )
    optimizer = VQEOptimizer(config=config, backend=backend, seed=seed)

    # Build circuit once (topology-independent param count for TFIM)
    base_lattice = make_lattice(topology, n, J=1.0, h=h_values[0])
    circuit, _ = hva.create(n, p_layers, base_lattice)

    # Record circuit structure (invariant across h-values)
    circuit_info = {
        "n_qubits": circuit.num_qubits,
        "n_parameters": circuit.num_parameters,
        "depth": circuit.depth(),
        "n_gates": sum(circuit.count_ops().values()),
        "gate_counts": dict(circuit.count_ops()),
    }
    logger.info(
        f"  Circuit: {circuit_info['n_qubits']} qubits, "
        f"{circuit_info['n_parameters']} params, depth={circuit_info['depth']}, "
        f"gates={circuit_info['n_gates']}"
    )

    # Descending sweep with warm-start propagation
    theta_prev = np.zeros(circuit.num_parameters)
    theta_prev_for_smoothness = np.zeros(circuit.num_parameters)
    results = []

    for idx, h in enumerate(h_values):
        t0 = time.time()
        lattice_h = make_lattice(topology, n, J=1.0, h=h)
        H = builder.build(lattice_h)

        dmrg_entry = dmrg_data[idx]
        e_exact = dmrg_entry["ground_energy"]
        gap = dmrg_entry["gap"]

        vqe_result = optimizer.optimize(H, circuit, theta_prev, exact_energy=e_exact)
        elapsed = time.time() - t0

        de_gap = abs(vqe_result.energy - e_exact) / max(gap, 1e-10)
        theta_prev_for_smoothness = theta_prev.copy()
        theta_prev = vqe_result.theta_opt.copy()

        if de_gap > 0.05:
            logger.warning(f"  ⚠ h={h:.3f}: ΔE/gap={de_gap:.4f} > 5%")

        results.append(
            {
                "h": h,
                "vqe_energy": vqe_result.energy,
                "dmrg_energy": e_exact,
                "gap": gap,
                "de_gap": de_gap,
                "energy_error": abs(vqe_result.energy - e_exact),
                "n_iterations": vqe_result.n_iterations,
                "converged": vqe_result.n_iterations < config.maxiter,
                "theta_opt": vqe_result.theta_opt.tolist(),
                "theta_init": theta_prev_for_smoothness.tolist(),
                "theta_smoothness": float(
                    np.max(np.abs(vqe_result.theta_opt - theta_prev_for_smoothness))
                )
                if idx > 0
                else 0.0,
                "time_s": elapsed,
                "passed": de_gap < 0.05,
            }
        )
        logger.info(
            f"  VQE h={h:.3f}: E={vqe_result.energy:.8f}, ΔE/gap={de_gap:.4f}, time={elapsed:.1f}s"
        )

    return results, circuit_info


def main() -> int:
    """Entry point for scaling validation."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    n = args.n
    topology = args.topology
    p_layers = args.p_layers
    strategy = args.strategy
    chi_max = args.chi_max
    precision = args.precision
    seeds = args.seeds
    output_dir = Path(args.output_dir)

    # Auto-compute h-values if not provided (valid regime for this N)
    if args.h_values is not None:
        h_values = sorted(args.h_values, reverse=True)
    else:
        h_min = 1.0 + 0.020 * n**1.31
        h_max = h_min + 1.5
        h_values = np.linspace(h_max, h_min + 0.5, 5).tolist()

    logger.info("=" * 60)
    logger.info(f"MPS Scaling Validation: N={n}, topology={topology}")
    logger.info(f"Strategy={strategy}, chi_max={chi_max}, precision={precision}")
    logger.info(f"h-values={[f'{h:.3f}' for h in h_values]}")
    logger.info("=" * 60)

    # Phase 1: DMRG ground truth
    logger.info("\n─── Phase 1: DMRG Ground Truth ───")
    t_phase1 = time.time()
    dmrg_data = phase1_dmrg_ground_truth(n, topology, h_values)
    t_phase1 = time.time() - t_phase1
    logger.info(f"  Phase 1 total: {t_phase1:.1f}s")

    # Phase 2: MPS-VQE for each seed
    all_seed_results = []
    circuit_info = None
    t_phase2 = time.time()
    for seed in seeds:
        logger.info(f"\n─── Phase 2: MPS-VQE (seed={seed}) ───")
        vqe_results, c_info = phase2_mps_vqe(
            n,
            topology,
            p_layers,
            h_values,
            strategy,
            chi_max,
            precision,
            seed,
            dmrg_data,
        )
        if circuit_info is None:
            circuit_info = c_info
        all_seed_results.append({"seed": seed, "results": vqe_results})
    t_phase2 = time.time() - t_phase2

    # Summary
    all_passed = all(r["passed"] for seed_run in all_seed_results for r in seed_run["results"])
    n_total = sum(len(sr["results"]) for sr in all_seed_results)
    n_pass = sum(1 for sr in all_seed_results for r in sr["results"] if r["passed"])

    logger.info("\n─── Summary ───")
    logger.info(f"  Passed: {n_pass}/{n_total}")
    logger.info(f"  Phase 1 time: {t_phase1:.1f}s")
    logger.info(f"  Phase 2 time: {t_phase2:.1f}s")
    logger.info(f"  Overall: {'PASS' if all_passed else 'FAIL'}")

    # Persist results
    # Compute aggregate statistics for analysis
    all_de_gaps = [r["de_gap"] for seed_run in all_seed_results for r in seed_run["results"]]
    all_theta_smoothness = [
        r.get("theta_smoothness", 0)
        for seed_run in all_seed_results
        for r in seed_run["results"]
        if r.get("theta_smoothness", 0) > 0
    ]

    # Scaling law prediction
    h_min_predicted = 1.0 + 0.020 * n**1.31

    envelope = {
        "experiment": "mps_scaling_validation",
        "version": "2.0",
        "metadata": {
            "n": n,
            "topology": topology,
            "p_layers": p_layers,
            "strategy": strategy,
            "chi_max": chi_max,
            "precision": precision,
            "seeds": seeds,
            "h_values": h_values,
            "optimizer_method": "COBYLA" if strategy == "aer_mps" else "L-BFGS-B",
            "n_restarts": 3,
            "maxiter": 500,
            "n_params": 2 * p_layers,
            "model": "tfim",
            "J": 1.0,
        },
        "environment": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python_version": sys.version.split()[0],
            "qiskit_version": _get_version("qiskit"),
            "qiskit_aer_version": _get_version("qiskit_aer"),
            "tenpy_version": _get_version("tenpy"),
            "numpy_version": np.__version__,
        },
        "scaling_law": {
            "formula": "h_min = 1.0 + 0.020 * N^1.31",
            "predicted_h_min": h_min_predicted,
            "lowest_h_tested": min(h_values),
            "highest_h_tested": max(h_values),
        },
        "circuit": circuit_info,
        "timing": {
            "phase1_dmrg_s": t_phase1,
            "phase2_vqe_s": t_phase2,
            "total_s": t_phase1 + t_phase2,
            "avg_per_hpoint_s": t_phase2 / max(len(h_values) * len(seeds), 1),
        },
        "dmrg_data": dmrg_data,
        "vqe_results": all_seed_results,
        "summary": {
            "n_pass": n_pass,
            "n_total": n_total,
            "all_passed": all_passed,
            "mean_de_gap": float(np.mean(all_de_gaps)) if all_de_gaps else None,
            "max_de_gap": float(np.max(all_de_gaps)) if all_de_gaps else None,
            "min_de_gap": float(np.min(all_de_gaps)) if all_de_gaps else None,
            "std_de_gap": float(np.std(all_de_gaps)) if all_de_gaps else None,
            "mean_theta_smoothness": float(np.mean(all_theta_smoothness))
            if all_theta_smoothness
            else None,
            "max_theta_smoothness": float(np.max(all_theta_smoothness))
            if all_theta_smoothness
            else None,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"scaling_N{n}_{strategy}_{timestamp}.json"
    json_dump(envelope, output_path)
    logger.info(f"  Results saved: {output_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
