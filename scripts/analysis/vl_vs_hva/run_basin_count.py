#!/usr/bin/env python
"""Basin-counting: how many distinguishable minima does the HVA landscape have?

Rigorous open question (a): the fair-convergence experiment established that the
N=18 bottleneck is *basin selection* — different seeds converge to clearly
different, stable fidelities. This asks the structural question directly:

    For a given (N, h), how many distinguishable fidelity basins does the
    bond-resolved nnn HVA p=2 landscape have, and what is the true multi-seed
    ceiling (best fidelity found over many unbiased random starts)?

Method
------
- Launch K RANDOM starts (unbiased sampling of the landscape, NOT analytic seeds),
  each optimized to genuine convergence (nit < maxiter; adjoint gradient).
- Cluster the converged fidelities with a tolerance ``--tol`` (default 0.02):
  each cluster = one distinguishable basin (by ground-state overlap).
- Report: number of basins, the members per basin, the true multi-seed ceiling
  (max fidelity), and the fraction of random starts that reach the best basin
  (its "basin of attraction" size under random init).

Sweeping ``--n`` over small sizes shows how the basin count scales with N — the
mechanism behind "the problem gets harder with N" (more basins ⇒ lower chance a
single seed lands in the best one).

Cheap by design at small N. Incremental JSON persist (crash-safe, NaN-safe writer).

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/run_basin_count.py \
        --n 8 --h 0.5 --p 2 --k 20 --maxiter 2000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import save_json  # noqa: E402


def _cluster(values: list[float], tol: float) -> list[list[int]]:
    """Greedy 1D clustering of fidelities: sort, split where consecutive gap > tol.

    Returns a list of clusters, each a list of indices into ``values``. Two
    fidelities in the same cluster are within ``tol`` of a shared chain — a
    distinguishable basin is a run of fidelities with no gap larger than tol.
    """
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i])
    clusters = [[order[0]]]
    for idx in order[1:]:
        if values[idx] - values[clusters[-1][-1]] <= tol:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    return clusters


def main(argv=None) -> int:
    from qmbp_simulation.framework.study_core import (
        diagnose_starvation,
        ground_state,
        make_cost_fid,
        starvation_probe,
    )

    p = argparse.ArgumentParser(description="Basin-counting for the frustrated HVA landscape")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--p", type=int, default=2)
    p.add_argument("--k", type=int, default=20, help="Number of random starts.")
    p.add_argument("--maxiter", type=int, default=2000, help="Per-start budget; convergence is nit<maxiter.")
    p.add_argument("--tol", type=float, default=0.02, help="Fidelity clustering tolerance (basin resolution).")
    p.add_argument("--seed0", type=int, default=90000)
    args = p.parse_args(argv)

    subdir = "hva_nnn_sweep"
    out_file = f"basincount_{args.topology}_N{args.n}_p{args.p}_h{args.h:.2f}_k{args.k}.json"

    _lat, qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(args.topology, args.n, args.h, args.j2, args.p)
    npar = qc.num_parameters
    # Block layout persisted at top level so saved theta can be decomposed by
    # role (theta_nn/nnn/x) later via results_query.decompose_theta.
    layout = {"n_nn": n_nn, "n_nnn": n_nnn, "n_qubits": args.n}
    cost, fid, grad, _ = make_cost_fid(qc, H, psi)

    print(
        f"[basin-count] N={args.n} h={args.h} p={args.p} K={args.k} "
        f"maxiter={args.maxiter} tol={args.tol} e0={e0:.5f} gap={gap:.5f} "
        f"npar={npar}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed0)
    runs = []
    for k in range(args.k):
        t0 = time.time()
        x0 = rng.uniform(-np.pi, np.pi, npar)
        # Two-segment starvation probe (shared core): a capped start may be
        # starved (still descending), not in a worse basin. Splitting the budget
        # measures the energy still gained in the 2nd half and flags starvation so
        # the basin clustering isn't contaminated by unfinished runs.
        x_fin, e_fin, total_nit, e_half, late_gain, conv = starvation_probe(cost, grad, x0, maxiter=args.maxiter)
        f = fid(x_fin)
        diagnosis = diagnose_starvation(conv, late_gain)
        runs.append(
            {
                "k": k,
                "fidelity": f,
                "e_final": e_fin,
                "nit": total_nit,
                "converged": conv,
                "e_halfway": e_half,
                "late_gain": late_gain,
                "diagnosis": diagnosis,
                "theta_init": np.asarray(x0, float).tolist(),
                "theta_final": np.asarray(x_fin, float).tolist(),
                "seconds": round(time.time() - t0, 1),
            }
        )
        # Incremental persist after every start
        save_json(
            {
                "runs": runs,
                "N": args.n,
                "h": args.h,
                "p_layers": args.p,
                "J2": args.j2,
                "e0": e0,
                "gap": gap,
                "maxiter": args.maxiter,
                "tol": args.tol,
                **layout,
            },
            subdir,
            out_file,
            params={
                "experiment": "basin_count",
                "N": args.n,
                "h": args.h,
                "p_layers": args.p,
                "K": args.k,
                "tol": args.tol,
            },
            description="Basin-counting: K random converged starts, clustered "
            "by final fidelity to count distinguishable minima",
        )
        print(f"  start {k:>2}: fid={f:.4f} nit={total_nit:>4} conv={conv} ({runs[-1]['seconds']}s)", flush=True)

    # Analyze: cluster the CONVERGED runs by fidelity
    conv_runs = [r for r in runs if r["converged"]]
    n_starved = len(runs) - len(conv_runs)
    fids = [r["fidelity"] for r in conv_runs]
    clusters = _cluster(fids, args.tol)
    # Sort clusters by their max fidelity (best basin first)
    clusters.sort(key=lambda c: -max(fids[i] for i in c))

    print(f"\n=== BASIN STRUCTURE (N={args.n} h={args.h} p={args.p}) ===", flush=True)
    print(f"converged: {len(conv_runs)}/{args.k}  (starved: {n_starved})", flush=True)
    print(f"distinguishable basins (tol={args.tol}): {len(clusters)}", flush=True)
    best_fid = max(fids) if fids else None
    for ci, c in enumerate(clusters):
        cfids = [fids[i] for i in c]
        print(
            f"  basin {ci}: fid {min(cfids):.4f}-{max(cfids):.4f}  "
            f"({len(c)}/{len(conv_runs)} starts = "
            f"{len(c) / len(conv_runs):.0%} attraction)",
            flush=True,
        )
    if best_fid is not None:
        best_cluster = clusters[0]
        print(f"\nTRUE MULTI-SEED CEILING (random): {best_fid:.4f}", flush=True)
        print(f"best-basin attraction: {len(best_cluster) / len(conv_runs):.0%} of random starts reach it", flush=True)

    # Persist the analysis alongside the runs
    analysis = {
        "n_converged": len(conv_runs),
        "n_starved": n_starved,
        "n_basins": len(clusters),
        "true_ceiling": best_fid,
        "best_basin_attraction": (len(clusters[0]) / len(conv_runs)) if clusters else None,
        "basins": [
            {
                "fid_min": min(fids[i] for i in c),
                "fid_max": max(fids[i] for i in c),
                "size": len(c),
                "attraction": len(c) / len(conv_runs),
            }
            for c in clusters
        ],
    }
    save_json(
        {
            "runs": runs,
            "analysis": analysis,
            "N": args.n,
            "h": args.h,
            "p_layers": args.p,
            "J2": args.j2,
            "e0": e0,
            "gap": gap,
            "maxiter": args.maxiter,
            "tol": args.tol,
            **layout,
        },
        subdir,
        out_file,
        params={
            "experiment": "basin_count",
            "N": args.n,
            "h": args.h,
            "p_layers": args.p,
            "K": args.k,
            "tol": args.tol,
        },
        description="Basin-counting: K random converged starts, clustered "
        "by final fidelity to count distinguishable minima",
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
