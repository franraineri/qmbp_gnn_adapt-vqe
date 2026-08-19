#!/usr/bin/env python3
"""DMRG vs Exact Diag — Direct comparison of classical ground truth precision.

This is the TRUE test of MPS limitation: compares the GROUND STATE ENERGY
computed by DMRG(χ=64) versus exact diagonalization, without any VQE or
ansatz in the loop. If DMRG(χ=64) ≠ exact, then MPS is genuinely limited
and QPU has a role. If DMRG(χ=64) = exact, then MPS is sufficient for
ground truth and QPU advantage must come from another source.

Key insight: The previous MPS precision study showed |trunc|=0 because it
tested the ANSATZ state (HVA p=1 output), which has low entanglement.
This script tests the GROUND STATE itself, which has high entanglement
near h_c where quantum correlations diverge.

Usage:
    # Default: N=10-22, heavy_hex + chain_1d, h near critical
    .venv/bin/python scripts/analysis/dmrg_vs_exact_comparison.py

    # Specific range
    .venv/bin/python scripts/analysis/dmrg_vs_exact_comparison.py \
        --n-values 10 14 18 20 22 \
        --topologies chain_1d heavy_hex triangular \
        --h-values 1.0 1.2 1.5 2.0 3.0 4.0

    # Focus on critical region
    .venv/bin/python scripts/analysis/dmrg_vs_exact_comparison.py \
        --h-values 0.8 0.9 1.0 1.1 1.2 1.3 1.5 2.0
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
from qmbp_simulation.models.model_registry import get_model_spec
from qmbp_simulation.utils.helpers import json_dump


def run_comparison(
    n_values: list[int],
    topologies: list[str],
    h_values: list[float],
    chi_values: list[int],
    model: str = "tfim",
) -> dict:
    """Compare DMRG(χ) vs exact diag for all configurations."""
    spec = get_model_spec(model)
    solver = ClassicalSolver()

    results = []
    print(f"\n{'Topology':<12} {'N':>3} {'h':>5} {'E_exact':>14} "
          + " ".join(f"{'E_dmrg_'+str(c):>14}" for c in chi_values)
          + f" {'|ΔE|_χ64':>12} {'ΔE/gap_χ64':>12}")
    print("-" * (60 + 15 * len(chi_values)))

    for topology in topologies:
        for N in n_values:
            if N > 22:
                print(f"  Skipping N={N} (exact diag limit is 22)")
                continue

            for h in h_values:
                lattice = make_lattice(topology, N, J=1.0, h=h)
                H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)

                # Exact diag (ground truth)
                t0 = time.perf_counter()
                gt_exact = solver.solve(H, lattice, method="exact")
                t_exact = time.perf_counter() - t0

                e_exact = gt_exact.ground_energy
                gap = gt_exact.gap

                # DMRG at each chi value
                dmrg_energies = {}
                for chi in chi_values:
                    t0 = time.perf_counter()
                    gt_dmrg = solver.solve(H, lattice, method="dmrg", chi_max=chi)
                    t_dmrg = time.perf_counter() - t0
                    dmrg_energies[chi] = {
                        "energy": gt_dmrg.ground_energy,
                        "time_s": round(t_dmrg, 3),
                    }

                # Metrics
                e_64 = dmrg_energies.get(64, {}).get("energy")
                abs_error_64 = abs(e_64 - e_exact) if e_64 is not None else None
                de_gap_64 = abs_error_64 / max(gap, 1e-10) if abs_error_64 is not None else None

                row = {
                    "topology": topology,
                    "n_qubits": N,
                    "h": h,
                    "e_exact": e_exact,
                    "gap": gap,
                    "t_exact_s": round(t_exact, 3),
                    "dmrg_per_chi": dmrg_energies,
                    "abs_error_chi64": abs_error_64,
                    "de_gap_chi64": de_gap_64,
                }
                results.append(row)

                # Print row
                dmrg_strs = []
                for c in chi_values:
                    e = dmrg_energies.get(c, {}).get("energy")
                    dmrg_strs.append(f"{e:>14.8f}" if e is not None else f"{'—':>14}")
                err_str = f"{abs_error_64:.2e}" if abs_error_64 else "—"
                dg_str = f"{de_gap_64:.6f}" if de_gap_64 else "—"
                print(f"{topology:<12} {N:>3} {h:>5.2f} {e_exact:>14.8f} "
                      + " ".join(dmrg_strs)
                      + f" {err_str:>12} {dg_str:>12}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — DMRG(χ=64) vs Exact Diag")
    print("=" * 60)

    for topology in topologies:
        topo_rows = [r for r in results if r["topology"] == topology]
        if not topo_rows:
            continue
        errors = [r["abs_error_chi64"] for r in topo_rows if r["abs_error_chi64"] is not None]
        de_gaps = [r["de_gap_chi64"] for r in topo_rows if r["de_gap_chi64"] is not None]
        if errors:
            print(f"\n  {topology}:")
            print(f"    mean |ΔE|(χ=64) = {np.mean(errors):.2e}")
            print(f"    max  |ΔE|(χ=64) = {np.max(errors):.2e}")
            print(f"    mean ΔE/gap(χ=64) = {np.mean(de_gaps):.6f}")
            print(f"    max  ΔE/gap(χ=64) = {np.max(de_gaps):.6f}")
            n_nonzero = sum(1 for e in errors if e > 1e-12)
            print(f"    Non-zero truncation: {n_nonzero}/{len(errors)} points")

    return {"results": results, "chi_values": chi_values}


def main():
    parser = argparse.ArgumentParser(
        description="DMRG vs Exact Diag — Direct ground state comparison"
    )
    parser.add_argument("--n-values", type=int, nargs="+", default=[10, 14, 18, 20, 22])
    parser.add_argument("--topologies", type=str, nargs="+",
                        default=["chain_1d", "heavy_hex"])
    parser.add_argument("--h-values", type=float, nargs="+",
                        default=[1.0, 1.2, 1.5, 2.0, 3.0, 4.0])
    parser.add_argument("--chi-values", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--model", type=str, default="tfim")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON (optional)")
    args = parser.parse_args()

    data = run_comparison(
        n_values=args.n_values,
        topologies=args.topologies,
        h_values=args.h_values,
        chi_values=args.chi_values,
        model=args.model,
    )

    if args.output:
        json_dump(data, Path(args.output))
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
