#!/usr/bin/env python
"""Audit the ANALYTIC warm-start layer against random initialisation.

Fair A/B on the frustrated square HVA (same circuit, same L-BFGS-B budget):

  WARM   : prepare_warmstart cascade seed (no cross-N donors) -> 1 descent
  RANDOM : R uniform U(-pi, pi) starts -> best-of-R by energy (more budget than WARM)
  COLD1  : the first random start alone (crude cold-start baseline)

Lesson baked in: the verdict flips with the budget. At maxiter=400 near h_c the
analytic seed and random best-of tie (everything is budget-starved); at
maxiter>=800 the converged analytic seed reaches ~0.95 at N14. Always audit with
an adequate budget. For the donor-enabled cascade use audit_warmstart_donors.py.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/audit_warmstart_vs_random.py \
        --points 8:0.5 10:1.0 10:2.0 --maxiter 400 --random-starts 4
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from qmbp_simulation.framework.study_core import ground_state, make_cost_fid, prepare_warmstart
from qmbp_simulation.framework.study_runner import _lbfgsb


def _converge(cost, grad, fid, x0, maxiter):
    x, e, nit = _lbfgsb(cost, np.asarray(x0, float), maxiter=maxiter, grad=grad)
    return float(fid(x)), float(e), int(nit)


def audit_point(n, h, args):
    lat, qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(args.topology, n, h, args.j2, args.p)
    cost, fid, grad, backend = make_cost_fid(qc, H, psi)
    npar = qc.num_parameters

    t0 = time.time()
    ws = prepare_warmstart(
        qc,
        H,
        psi,
        n_nn=n_nn,
        n_nnn=n_nnn,
        n_qubits=n,
        p_layers=args.p,
        h=h,
        J2=args.j2,
        micro_descent=args.select_descent or args.maxiter,
        cost=cost,
        fid=fid,
        grad=grad,
        backend=backend,
    )
    warm_init = float(fid(ws["seed"]))
    warm_fid, _, warm_nit = _converge(cost, grad, fid, ws["seed"], args.maxiter)
    warm_t = time.time() - t0

    t0 = time.time()
    rng = np.random.default_rng(args.seed + n * 100 + int(round(h * 10)))
    best_e, best_fid, cold1, matched_at = None, -1.0, None, None
    for r in range(args.random_starts):
        f, e, _ = _converge(cost, grad, fid, rng.uniform(-np.pi, np.pi, npar), args.maxiter)
        if r == 0:
            cold1 = f
        if best_e is None or e < best_e:
            best_e, best_fid = e, f
        if matched_at is None and best_fid >= warm_fid - 0.005:
            matched_at = r + 1
    return {
        "n": n,
        "h": h,
        "gap": gap,
        "npar": npar,
        "warm_init": warm_init,
        "warm_fid": warm_fid,
        "warm_nit": warm_nit,
        "warm_prov": ws["provenance"],
        "warm_t": warm_t,
        "cold1": cold1,
        "rand_best": best_fid,
        "rand_t": time.time() - t0,
        "matched_at": matched_at,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Analytic warm-start vs random audit")
    p.add_argument("--topology", default="square")
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--p", type=int, default=1, help="HVA layers (standard bond-resolved layout).")
    p.add_argument(
        "--points", nargs="+", default=["8:0.5", "10:0.5", "10:1.0", "10:2.0"], help="N:h pairs, e.g. 10:0.5 14:0.5"
    )
    p.add_argument(
        "--maxiter", type=int, default=400, help="Descent budget. Use >=800 for N>=14 near h_c (400 is budget-starved)."
    )
    p.add_argument(
        "--select-descent",
        type=int,
        default=None,
        help="Micro-descent iters used to rank cascade candidates (default: maxiter).",
    )
    p.add_argument("--random-starts", type=int, default=4)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args(argv)

    points = [(int(a), float(b)) for a, b in (s.split(":") for s in args.points)]
    print(
        f"[audit] warm vs random | {args.topology} J2={args.j2} p={args.p} "
        f"maxiter={args.maxiter} random_starts={args.random_starts}",
        flush=True,
    )
    print(
        f"{'N':>3} {'h':>4} {'gap':>8} {'npar':>4} | {'warm_init':>9} {'warm_fid':>8} {'nit':>4} | "
        f"{'cold1':>7} {'rand_bo':>7} | {'d(w-rand)':>9} {'d(w-cold)':>9} | {'rand=warm@':>10} prov",
        flush=True,
    )
    rows = []
    for n, h in points:
        r = audit_point(n, h, args)
        rows.append(r)
        matched = f"start {r['matched_at']}" if r["matched_at"] else "never"
        print(
            f"{r['n']:>3} {r['h']:>4.2f} {r['gap']:>8.4f} {r['npar']:>4} | "
            f"{r['warm_init']:>9.4f} {r['warm_fid']:>8.4f} {r['warm_nit']:>4} | "
            f"{r['cold1']:>7.4f} {r['rand_best']:>7.4f} | "
            f"{r['warm_fid'] - r['rand_best']:>+9.4f} {r['warm_fid'] - r['cold1']:>+9.4f} | "
            f"{matched:>10} {r['warm_prov']}",
            flush=True,
        )

    d_rand = [r["warm_fid"] - r["rand_best"] for r in rows]
    d_cold = [r["warm_fid"] - r["cold1"] for r in rows]
    wins = sum(d > 0.005 for d in d_rand)
    ties = sum(abs(d) <= 0.005 for d in d_rand)
    print(
        f"\n[audit] vs random best-of-{args.random_starts}: warm wins {wins}, ties {ties}, "
        f"loses {len(d_rand) - wins - ties} (mean {np.mean(d_rand):+.4f})",
        flush=True,
    )
    print(f"[audit] vs single cold start: mean {np.mean(d_cold):+.4f} (max {max(d_cold):+.4f})", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
