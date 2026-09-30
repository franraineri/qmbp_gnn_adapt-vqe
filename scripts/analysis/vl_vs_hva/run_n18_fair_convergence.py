#!/usr/bin/env python
"""Fair-convergence comparison of warm-start seeds at N=18 h=0.5 (frustrated square).

Motivation
----------
The ``mixed`` strategy run at N=18 h=0.5 gave every alternative restart a fixed
maxiter that 4 of 6 restarts HIT (nit==maxiter) — i.e. they never converged. It
is therefore invalid to conclude those seeds reach "worse" basins: they were
still descending when the budget ran out (the budget-starvation failure mode).

This experiment removes that bias. Each seed type is optimized with a LARGE
per-seed iteration budget until it genuinely converges (nit < maxiter), and only
then are the converged fidelities compared on equal footing. For any seed that
still hits the cap, we record a starvation diagnostic (energy still dropping →
needs more iterations, vs energy flat → genuine local minimum) instead of
mislabeling it "worse".

Design
------
- Seeds compared (same set the ``mixed`` strategy used, plus the pure seeds):
  second_pure, first_pure, second_iso_small, second_iso_large, second_dir_x,
  second_dir_zz.
- Fair budget: a high ``--maxiter`` (default 3000) so far-from-optimum seeds can
  reach their basin floor. Convergence is ``nit < maxiter``.
- Starvation probe: L-BFGS-B is run in two segments (half budget, then full) to
  measure the energy still gained in the second half. Large late gain ⇒ the cap
  is the limit, not the basin.
- Reuses the production physics/optimizer stack: NoiselessBackend (cached H +
  adjoint gradient), the warm-start seeds, and the mixed block masks — no
  duplicated physics.
- Incremental JSON persist (crash-safe); NaN-safe writer.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/run_n18_fair_convergence.py \
        --n 18 --h 0.5 --p 2 --maxiter 3000
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


def _seeds(n_nn, n_nnn, n, p, h, j2):
    """Return the seed-type table: name -> (base_theta, block, sigma)."""
    from qmbp_simulation.analysis.warmstart import (
        first_order_warmstart_theta,
        second_order_warmstart_theta,
    )

    th2 = second_order_warmstart_theta(n_nn, n_nnn, n, p, h, J=1.0, J2=j2)
    th1 = first_order_warmstart_theta(n_nn, n_nnn, n, p, h, J=1.0, J2=j2)
    return {
        "second_pure": (th2, "all", 0.0),
        "first_pure": (th1, "all", 0.0),
        "second_iso_small": (th2, "all", 0.10),
        "second_iso_large": (th2, "all", 0.30),
        "second_dir_x": (th2, "x", 0.20),
        "second_dir_zz": (th2, "zz", 0.20),
    }


def main(argv=None) -> int:
    from qmbp_simulation.framework.study_core import (
        diagnose_starvation,
        ground_state,
        make_cost_fid,
        starvation_probe,
    )
    from qmbp_simulation.framework.study_runner import _block_mask

    p = argparse.ArgumentParser(description="Fair-convergence seed comparison (N=18 frustrated)")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=18)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--p", type=int, default=2)
    p.add_argument(
        "--maxiter", type=int, default=3000, help="High per-seed budget so far seeds can converge (nit<maxiter)."
    )
    p.add_argument("--seed0", type=int, default=50000)
    args = p.parse_args(argv)

    subdir = "hva_nnn_sweep"
    out_file = f"n{args.n}_{args.topology}_N{args.n}_p{args.p}_faircov_mi{args.maxiter}.json"

    lat, qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(args.topology, args.n, args.h, args.j2, args.p)
    npar = qc.num_parameters
    cost, fid, grad, _ = make_cost_fid(qc, H, psi)

    masks = {b: _block_mask(b, n_nn, n_nnn, args.n, args.p) for b in ("all", "x", "zz")}
    seeds = _seeds(n_nn, n_nnn, args.n, args.p, args.h, args.j2)

    print(
        f"[fair-cov] N={args.n} h={args.h} p={args.p} maxiter={args.maxiter} e0={e0:.5f} gap={gap:.5f} npar={npar}",
        flush=True,
    )

    results = []
    for name, (base, block, sigma) in seeds.items():
        t0 = time.time()
        if sigma == 0.0:
            x0 = np.asarray(base, float)
        else:
            rng = np.random.default_rng(args.seed0 + hash(name) % 1000)
            noise = np.zeros(npar)
            m = masks[block]
            noise[m] = rng.normal(0, sigma, int(m.sum()))
            x0 = np.clip(np.asarray(base, float) + noise, -np.pi, np.pi)

        # Two-segment starvation probe (shared core): half budget, then the rest.
        x_fin, e_fin, total_nit, e_half, late_gain, converged = starvation_probe(cost, grad, x0, maxiter=args.maxiter)
        f_fin = fid(x_fin)
        diagnosis = diagnose_starvation(
            converged,
            late_gain,
            starved_label="starved (energy still dropping — needs more iters)",
            local_min_label="genuine local minimum (flat, budget not the issue)",
        )

        row = {
            "seed_type": name,
            "block": block,
            "sigma": sigma,
            "fidelity": f_fin,
            "e_final": e_fin,
            "e0": e0,
            "gap": gap,
            "abs_error": abs(e_fin - e0),
            "de_gap": abs(e_fin - e0) / gap if gap > 0 else None,
            "total_nit": total_nit,
            "converged": converged,
            "e_halfway": e_half,
            "late_gain": late_gain,
            "diagnosis": diagnosis,
            # HVA angles at start (seed) and end (optimized) for reproducibility.
            "theta_init": np.asarray(x0, float).tolist(),
            "theta_final": np.asarray(x_fin, float).tolist(),
            "seconds": round(time.time() - t0, 1),
        }
        results.append(row)
        save_json(
            {
                "rows": results,
                "N": args.n,
                "h": args.h,
                "p_layers": args.p,
                "J2": args.j2,
                "maxiter": args.maxiter,
                "e0": e0,
                "gap": gap,
                "n_nn": n_nn,
                "n_nnn": n_nnn,
                "n_qubits": args.n,
            },
            subdir,
            out_file,
            params={
                "experiment": "fair_convergence",
                "N": args.n,
                "h": args.h,
                "p_layers": args.p,
                "maxiter": args.maxiter,
            },
            description="Fair-convergence seed comparison at N=18 h=0.5: each "
            "seed optimized to genuine convergence, capped seeds "
            "flagged starved vs local-min (no mislabeling as 'worse')",
        )
        print(
            f"  {name:18s} fid={f_fin:.4f} nit={total_nit:>4} "
            f"conv={converged} late_gain={late_gain:.4f} -> {diagnosis} "
            f"({row['seconds']}s)",
            flush=True,
        )

    # Summary: fair ranking (converged seeds only) + starved list
    conv = [r for r in results if r["converged"]]
    starved = [r for r in results if not r["converged"] and "starved" in r["diagnosis"]]
    print("\n=== FAIR RANKING (converged seeds) ===", flush=True)
    for r in sorted(conv, key=lambda x: -x["fidelity"]):
        print(f"  {r['seed_type']:18s} fid={r['fidelity']:.4f} (nit={r['total_nit']})", flush=True)
    if starved:
        print("\n=== STARVED (need more budget, NOT worse) ===", flush=True)
        for r in starved:
            print(f"  {r['seed_type']:18s} fid={r['fidelity']:.4f} late_gain={r['late_gain']:.4f}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
