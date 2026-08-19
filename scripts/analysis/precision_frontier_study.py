#!/usr/bin/env python3
"""Precision Frontier Study — Map where each method breaks down near h_c.

Systematic comparison of:
  - DMRG(graph, chi) at chi = 32, 64, 128, 256, 512
  - VQE noiseless (HVA p=1,2,3,4) — best achievable energy

At N = 10, 16, 22 on heavy_hex, with h dense near h_c ≈ 1.0.

This produces the definitive table:
  "At what h does each method reach <5% ΔE/gap?"
  "What is the minimum p for HVA to express the ground state at each h?"
  "At what chi does DMRG become chi-limited (if ever)?"

Usage:
    # Full study (N=10,16,22, p=1-4, chi=32-512, h dense near h_c)
    .venv/bin/python scripts/analysis/precision_frontier_study.py

    # Quick (N=10 only, p=1-2)
    .venv/bin/python scripts/analysis/precision_frontier_study.py --quick

    # Save output
    .venv/bin/python scripts/analysis/precision_frontier_study.py --output results/analysis/precision_frontier.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qmbp_simulation import ClassicalSolver, make_lattice
from qmbp_simulation.execution import NoiselessBackend
from qmbp_simulation.models.data_models import VQEConfig
from qmbp_simulation.models.model_registry import get_model_spec
from qmbp_simulation.optimizers.vqe import VQEOptimizer
from qmbp_simulation.utils.helpers import json_dump


def run_study(
    n_values: list[int],
    h_values: list[float],
    p_values: list[int],
    chi_values: list[int],
    topology: str = "heavy_hex",
    model: str = "tfim",
    vqe_restarts: int = 5,
    vqe_maxiter: int = 500,
    seed: int = 42,
) -> dict:
    """Run the full precision frontier study."""
    spec = get_model_spec(model)
    solver = ClassicalSolver()
    noiseless = NoiselessBackend()

    all_results = []

    for N in n_values:
        print(f"\n{'='*70}")
        print(f"  N = {N}, topology = {topology}")
        print(f"{'='*70}")

        # Header
        chi_headers = " ".join(f"{'χ='+str(c):>10}" for c in chi_values)
        p_headers = " ".join(f"{'p='+str(p):>10}" for p in p_values)
        print(f"\n  {'h':>5} {'E_exact':>12} {'gap':>8} {chi_headers}  |  {p_headers}")
        print(f"  {'-'*5} {'-'*12} {'-'*8} " + "-"*(11*len(chi_values)) +
              "  |  " + "-"*(11*len(p_values)))

        # Pre-compute VQE with warm-start descending sweep per p
        # This gives each p its best shot at each h via warm-start
        vqe_best_per_p: dict[int, dict[float, float]] = {p: {} for p in p_values}

        for p in p_values:
            lattice_ref = make_lattice(topology, N, J=1.0, h=max(h_values))
            circuit, _ = spec.create_circuit(N, p, lattice_ref, **spec.circuit_kwargs)
            n_params = circuit.num_parameters

            config = VQEConfig(
                p_layers=p, n_restarts=vqe_restarts,
                maxiter=vqe_maxiter, method="L-BFGS-B",
                enable_callbacks=False,
            )
            opt = VQEOptimizer(config=config, backend=noiseless, seed=seed)
            rng = np.random.default_rng(seed)
            prev_theta = rng.uniform(-0.01, 0.01, n_params)

            # Descending sweep
            for h in sorted(h_values, reverse=True):
                lattice_h = make_lattice(topology, N, J=1.0, h=h)
                H_h = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)
                result = opt.optimize(H_h, circuit, prev_theta.copy())
                prev_theta = result.theta_opt.copy()
                vqe_best_per_p[p][h] = result.energy

        # Now compute exact + DMRG for each h and assemble the table
        for h in sorted(h_values, reverse=True):
            lattice = make_lattice(topology, N, J=1.0, h=h)
            H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)

            # Exact diag (reference)
            gt = solver.solve(H, lattice, method="exact")
            e_exact = gt.ground_energy
            gap = gt.gap

            # DMRG at each chi
            dmrg_de_gaps = {}
            for chi in chi_values:
                gt_chi = solver.solve(H, lattice, method="dmrg", chi_max=chi)
                de = abs(gt_chi.ground_energy - e_exact)
                dmrg_de_gaps[chi] = de / max(gap, 1e-10)

            # VQE ΔE/gap at each p
            vqe_de_gaps = {}
            for p in p_values:
                e_vqe = vqe_best_per_p[p].get(h, float("nan"))
                de = abs(e_vqe - e_exact)
                vqe_de_gaps[p] = de / max(gap, 1e-10)

            # Store
            row = {
                "N": N, "topology": topology, "h": h,
                "e_exact": e_exact, "gap": gap,
                "dmrg_de_gap": {str(c): dmrg_de_gaps[c] for c in chi_values},
                "vqe_de_gap": {str(p): vqe_de_gaps[p] for p in p_values},
            }
            all_results.append(row)

            # Print row
            chi_strs = " ".join(
                f"{dmrg_de_gaps[c]*100:>9.3f}%" for c in chi_values
            )
            p_strs = " ".join(
                f"{vqe_de_gaps[p]*100:>9.3f}%" for p in p_values
            )
            print(f"  {h:>5.2f} {e_exact:>12.6f} {gap:>8.4f} {chi_strs}  |  {p_strs}")

    # Summary: find frontier for each method
    print(f"\n{'='*70}")
    print("  PRECISION FRONTIER SUMMARY")
    print(f"{'='*70}")
    print(f"  (ΔE/gap threshold: 5% = method is 'accurate' below this)")
    print()

    for N in n_values:
        n_rows = [r for r in all_results if r["N"] == N]
        print(f"  N={N}:")

        # For each chi: lowest h where ΔE/gap < 5%
        for chi in chi_values:
            passing_h = [r["h"] for r in n_rows if r["dmrg_de_gap"][str(chi)] < 0.05]
            h_min = min(passing_h) if passing_h else ">max_h"
            print(f"    DMRG χ={chi:>3d}: accurate for h ≥ {h_min}")

        # For each p: lowest h where ΔE/gap < 5%
        for p in p_values:
            passing_h = [r["h"] for r in n_rows if r["vqe_de_gap"][str(p)] < 0.05]
            h_min = min(passing_h) if passing_h else ">max_h"
            print(f"    VQE  p={p}:    accurate for h ≥ {h_min}")
        print()

    return {
        "config": {
            "n_values": n_values, "h_values": h_values,
            "p_values": p_values, "chi_values": chi_values,
            "topology": topology, "model": model,
        },
        "results": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Precision Frontier Study")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: N=10 only, p=1-2, fewer h values")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--topology", type=str, default="heavy_hex")
    args = parser.parse_args()

    if args.quick:
        n_values = [10]
        h_values = [0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0]
        p_values = [1, 2, 3, 4]
        chi_values = [64, 128, 256]
    else:
        n_values = [10, 16, 22]
        # Dense near h_c, sparse far away
        h_values = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
        p_values = [1, 2, 3, 4]
        chi_values = [32, 64, 128, 256, 512]

    data = run_study(
        n_values=n_values,
        h_values=h_values,
        p_values=p_values,
        chi_values=chi_values,
        topology=args.topology,
    )

    if args.output:
        json_dump(data, Path(args.output))
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
