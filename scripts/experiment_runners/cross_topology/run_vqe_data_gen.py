#!/usr/bin/env python3
"""VQE data generation for cross-topology transfer experiments.

Generates VQE optimal parameters for heavy_hex at N=6 (statevector) and
N=16 (MPS chi=64, COBYLA). Produces at least 5 h-values in the valid
regime per size, saving results in the standard scaling JSON format.

Supports both p=1 (2 params, single start) and p=2 (4 params, 5 restarts).

Usage:
    python scripts/experiment_runners/cross_topology/run_vqe_data_gen.py \\
        --topology heavy_hex --n 6 16 --p 1 \\
        --seeds 42,43,44 --output-dir results/scaling/cross_topology

    # Skip existing results:
    python scripts/experiment_runners/cross_topology/run_vqe_data_gen.py \\
        --topology heavy_hex --n 6 16 --p 1 --skip-existing

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 7.3
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT / "scripts" / "experiment_runners") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts" / "experiment_runners"))

import numpy as np

from cross_topology.helpers import (
    save_validation_checkpoint,
    validate_vqe_sweep_quality,
)
from qmbp_simulation import (
    ClassicalSolver,
    HamiltonianBuilder,
    HVACircuitBuilder,
    VQEOptimizer,
    make_lattice,
)
from qmbp_simulation.execution import MPSBackend, NoiselessBackend
from qmbp_simulation.models.data_models import VQEConfig
from qmbp_simulation.utils.helpers import json_dump

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# h-value regime for heavy_hex
# ═══════════════════════════════════════════════════════════════════════════════

# Scaling law: h_min_safe = 1.5 + 0.020·N^1.31 (project-status.md)
# heavy_hex N=10: h >= 3.25 (validated)
# Scale for other sizes:
#   N=6:  1.5 + 0.020*6^1.31 ≈ 1.5 + 0.18 = 1.68 → use h >= 2.5 (conservative)
#   N=16: 1.5 + 0.020*16^1.31 ≈ 1.5 + 0.68 = 2.18 → use h >= 3.5 (conservative for MPS)

DEFAULT_H_VALUES: dict[int, list[float]] = {
    6: [5.0, 4.5, 4.0, 3.5, 3.0],
    10: [5.0, 4.5, 4.0, 3.5, 3.25],
    16: [6.0, 5.5, 5.0, 4.5, 4.0, 3.5],
}


def get_h_values(n: int) -> list[float]:
    """Get appropriate h-values for a given system size.

    Returns descending h-values in the valid regime for heavy_hex.
    """
    if n in DEFAULT_H_VALUES:
        return DEFAULT_H_VALUES[n]
    # Fallback: compute from scaling law
    h_min = 1.5 + 0.020 * n**1.31 + 0.5  # add safety margin
    h_max = h_min + 2.5
    return np.linspace(h_max, h_min, 6).tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# VQE sweep for a single (topology, N, seed) configuration
# ═══════════════════════════════════════════════════════════════════════════════


def run_vqe_sweep(
    topology: str,
    n: int,
    p: int,
    seed: int,
    h_values: list[float],
) -> dict:
    """Run a VQE descending sweep for a given topology, size, and seed.

    Parameters
    ----------
    topology : str
        Lattice topology (e.g. "heavy_hex").
    n : int
        Number of qubits.
    p : int
        HVA depth (1 or 2).
    seed : int
        Random seed for VQE optimizer.
    h_values : list[float]
        h-values in descending order.

    Returns
    -------
    dict
        Seed run result with "seed" and "results" keys.
    """
    # Backend dispatch: NoiselessBackend for N<=15, MPSBackend for N=16+
    if n > 15:
        backend = MPSBackend(
            strategy="aer_mps", chi_max=MPS_DEFAULT_CHI_MAX, precision=0.005, seed=seed
        )
        method = "COBYLA"
        logger.info(
            "  Backend: MPSBackend(chi_max=MPS_DEFAULT_CHI_MAX, strategy=aer_mps), method=COBYLA"
        )
    else:
        backend = NoiselessBackend()
        method = "L-BFGS-B"
        logger.info("  Backend: NoiselessBackend, method=L-BFGS-B")

    # Configure VQE: p=2 uses 5 restarts, p=1 uses 1 restart
    n_restarts = 5 if p == 2 else 1
    vqe_config = VQEConfig(
        p_layers=p,
        n_restarts=n_restarts,
        method=method,
        maxiter=1000,
        ftol=1e-14,
        enable_callbacks=False,
    )
    optimizer = VQEOptimizer(config=vqe_config, backend=backend, seed=seed)

    builder = HamiltonianBuilder()
    hva = HVACircuitBuilder()
    solver = ClassicalSolver()

    results_list: list[dict] = []
    h_values_desc = sorted(h_values, reverse=True)

    # Build circuit once (topology-dependent, p-layers)
    lattice_ref = make_lattice(topology, n, J=1.0, h=h_values_desc[0])
    circuit, _ = hva.create(n, p, lattice_ref)
    n_params = circuit.num_parameters
    logger.info(f"  Circuit: {n_params} parameters (p={p}), N={n}")

    previous_theta: np.ndarray | None = None

    for h_val in h_values_desc:
        t0 = time.perf_counter()

        # Build Hamiltonian and get exact solution
        lattice = make_lattice(topology, n, J=1.0, h=h_val)
        H = builder.build(lattice)
        gt = solver.solve(H, lattice, method="auto")

        # Initial guess: warm-start from previous h-point
        if previous_theta is not None:
            x0 = previous_theta.copy()
        else:
            np.random.seed(seed)
            x0 = np.random.uniform(-0.01, 0.01, n_params)

        # Run VQE optimization
        vqe_result = optimizer.optimize(
            hamiltonian=H,
            circuit=circuit,
            initial_guess=x0,
            exact_energy=gt.ground_energy,
        )

        elapsed = time.perf_counter() - t0
        de_gap = abs(vqe_result.energy - gt.ground_energy) / max(gt.gap, 1e-10)
        energy_error = float(vqe_result.energy - gt.ground_energy)

        # Validate VQE result
        if energy_error < -1e-6:
            logger.warning(
                f"    ⚠️ Variational principle violated at h={h_val:.2f}: "
                f"E_vqe={vqe_result.energy:.6f} < E_exact={gt.ground_energy:.6f}"
            )
        if de_gap > 0.05:
            logger.warning(
                f"    ⚠️ VQE did not converge well at h={h_val:.2f}: "
                f"ΔE/gap={de_gap * 100:.2f}% > 5% threshold"
            )
        if elapsed > 300:  # 5 min per point is concerning
            logger.warning(
                f"    ⚠️ Slow convergence at h={h_val:.2f}: {elapsed:.0f}s (budget risk for N={n})"
            )

        result_entry = {
            "h": float(h_val),
            "vqe_energy": float(vqe_result.energy),
            "dmrg_energy": float(gt.ground_energy),
            "gap": float(gt.gap),
            "de_gap": float(de_gap),
            "energy_error": energy_error,
            "variational_ok": energy_error >= -1e-6,
            "n_iterations": vqe_result.n_iterations,
            "theta_opt": vqe_result.theta_opt.tolist(),
            "time_s": elapsed,
            "passed": bool(de_gap < 0.05),
        }
        results_list.append(result_entry)

        status = "✅" if de_gap < 0.05 else "⚠️"
        logger.info(
            f"    {status} h={h_val:.2f}: ΔE/gap={de_gap * 100:.3f}%, "
            f"nit={vqe_result.n_iterations}, time={elapsed:.1f}s"
        )

        # Propagate warm-start
        previous_theta = vqe_result.theta_opt.copy()

    return {"seed": seed, "results": results_list}


# ═══════════════════════════════════════════════════════════════════════════════
# Output path generation
# ═══════════════════════════════════════════════════════════════════════════════


def get_output_path(output_dir: Path, topology: str, n: int, p: int) -> Path:
    """Generate the output file path for a given configuration."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"vqe_{topology}_N{n}_p{p}_{timestamp}.json"
    return output_dir / filename


def get_expected_path_pattern(output_dir: Path, topology: str, n: int, p: int) -> str:
    """Generate a glob pattern for finding existing results."""
    return f"vqe_{topology}_N{n}_p{p}_*.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VQE data generation for cross-topology transfer experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--topology",
        type=str,
        default="heavy_hex",
        help="Lattice topology to generate data for",
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[6, 16],
        help="System sizes to generate (list)",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=1,
        choices=[1, 2],
        help="HVA depth (1=2 params, 2=4 params with 5 restarts)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated list of random seeds",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/scaling/cross_topology",
        help="Output directory for result JSON files",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip generation if result file already exists",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log all parameters at INFO level
    logger.info("=" * 60)
    logger.info("VQE Data Generation for Cross-Topology Transfer")
    logger.info("=" * 60)
    logger.info(f"  Topology: {args.topology}")
    logger.info(f"  System sizes: {args.n}")
    logger.info(f"  HVA depth: p={args.p} ({2 * args.p} parameters)")
    logger.info(f"  Seeds: {seeds}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Skip existing: {args.skip_existing}")
    if args.p == 2:
        logger.info("  Restarts: 5 (p=2 mode)")
    else:
        logger.info("  Restarts: 1 (p=1 mode)")
    logger.info("=" * 60)

    t_total_start = time.perf_counter()
    generated_files: list[str] = []

    for n_size in args.n:
        # Check for existing results if --skip-existing
        if args.skip_existing:
            pattern = get_expected_path_pattern(output_dir, args.topology, n_size, args.p)
            existing = list(output_dir.glob(pattern))
            if existing:
                logger.info(
                    f"\n  SKIP: {args.topology} N={n_size} p={args.p} — "
                    f"found existing: {existing[0].name}"
                )
                continue

        logger.info(f"\n{'─' * 60}")
        logger.info(f"  Generating: {args.topology} N={n_size} p={args.p}")
        logger.info(f"{'─' * 60}")

        h_values = get_h_values(n_size)
        logger.info(f"  h-values: {h_values}")

        # Run VQE sweep for each seed
        t_size_start = time.perf_counter()
        all_seed_results: list[dict] = []

        for seed in seeds:
            logger.info(f"\n  ── Seed {seed} ──")
            seed_result = run_vqe_sweep(
                topology=args.topology,
                n=n_size,
                p=args.p,
                seed=seed,
                h_values=h_values,
            )
            all_seed_results.append(seed_result)

        t_size_elapsed = time.perf_counter() - t_size_start

        # ── Validate VQE sweep quality (Check 6) ─────────────────────
        for seed_result in all_seed_results:
            sweep_report = validate_vqe_sweep_quality(
                seed_result["results"],
                topology=args.topology,
                n=n_size,
                seed=seed_result["seed"],
            )
            if not sweep_report.passed:
                logger.error(
                    f"VQE sweep quality check FAILED for {args.topology} "
                    f"N={n_size} seed={seed_result['seed']}. "
                    f"Data may not be usable for GNN training."
                )
                save_validation_checkpoint(
                    sweep_report,
                    output_dir,
                    f"vqe_{args.topology}_N{n_size}_s{seed_result['seed']}",
                )

        # Build DMRG data (from first seed's results)
        dmrg_data = [
            {
                "h": r["h"],
                "ground_energy": r["dmrg_energy"],
                "gap": r["gap"],
            }
            for r in all_seed_results[0]["results"]
        ]

        # Save in scaling JSON format
        result_json = {
            "experiment": "cross_topology_vqe_data_gen",
            "metadata": {
                "n": n_size,
                "topology": args.topology,
                "p_layers": args.p,
                "strategy": "aer_mps" if n_size > 15 else "statevector",
                "chi_max": 64 if n_size > 15 else None,
                "precision": 0.005 if n_size > 15 else None,
                "method": "COBYLA" if n_size > 15 else "L-BFGS-B",
                "n_restarts": 5 if args.p == 2 else 1,
                "seeds": seeds,
                "h_values": h_values,
            },
            "timing": {
                "total_s": t_size_elapsed,
            },
            "environment": {
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "dmrg_data": dmrg_data,
            "vqe_results": all_seed_results,
        }

        out_path = get_output_path(output_dir, args.topology, n_size, args.p)
        json_dump(result_json, out_path)
        generated_files.append(str(out_path))

        logger.info(f"\n  ✅ Saved: {out_path}")
        logger.info(f"  Time for N={n_size}: {t_size_elapsed:.1f}s ({t_size_elapsed / 60:.1f}m)")

    t_total_elapsed = time.perf_counter() - t_total_start

    # Final summary
    logger.info(f"\n{'═' * 60}")
    logger.info("  VQE Data Generation Complete")
    logger.info(f"{'═' * 60}")
    logger.info(f"  Total time: {t_total_elapsed:.1f}s ({t_total_elapsed / 60:.1f}m)")
    logger.info(f"  Files generated: {len(generated_files)}")
    for f in generated_files:
        logger.info(f"    - {f}")

    if t_total_elapsed > 3600:
        logger.warning(f"  ⚠️ Exceeded 60-min budget: {t_total_elapsed / 60:.1f}m")

    return 0


if __name__ == "__main__":
    sys.exit(main())
