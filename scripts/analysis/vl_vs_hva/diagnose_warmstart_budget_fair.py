#!/usr/bin/env python
"""Budget-FAIR check: is the LX lever better strategy, or just more compute?

The levers diagnostic used unequal budgets. Here every arm gets the SAME total
L-BFGS-B iteration budget B (counting θ_x-best-of restarts), so we isolate
strategy from raw compute:

  base_B  : single full L-BFGS-B of B iters from the best candidate
  xbest_B : θ_x-subspace best-of, R restarts × (B/R) iters (same total)
  split_B : B/2 full descent, then θ_x best-of with the remaining B/2

If xbest_B / split_B beat base_B at equal B → the θ_x multistart is genuinely a
better use of the budget (worth wiring into the default). If base_B ties → the
gain was only the extra iterations (just raise micro_descent).

Usage:
    .venv/bin/python -u scripts/analysis/vl_vs_hva/diagnose_warmstart_budget_fair.py \
        | tee /tmp/warmstart_budget_fair.log
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis" / "vl_vs_hva"))

from qmbp_simulation.analysis.warmstart import theta_x_indices  # noqa: E402
from qmbp_simulation.framework.study_core import (  # noqa: E402
    ground_state,
    make_cost_fid,
    optimize_xspace_bestof,
    prepare_warmstart,
)
from qmbp_simulation.framework.study_runner import _lbfgsb  # noqa: E402

J2, P, TOPO, MODEL = 0.5, 2, "square", "tfim_frustrated"
B = 48          # total iteration budget per arm
R = 4           # restarts for θ_x best-of (so maxiter = B // R)


def validate_point(n, h):
    lat, qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(TOPO, n, h, J2, P)
    cost, fid, grad, _ = make_cost_fid(qc, H, psi)
    # best candidate seed BEFORE descent: use prepare_warmstart with micro_descent=0
    ws0 = prepare_warmstart(qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=n,
                            p_layers=P, h=h, J=1.0, J2=J2, topology=TOPO,
                            model=MODEL, micro_descent=0, cost=cost, fid=fid, grad=grad)
    seed = ws0["seed"]
    x_idx = theta_x_indices(n_nn, n_nnn, n, P)

    # arm 1: single full descent of B iters
    xb, _e, _n = _lbfgsb(cost, np.asarray(seed, float), maxiter=B, grad=grad)
    f_base = fid(xb)

    # arm 2: θ_x best-of, R restarts × (B/R) iters (same total)
    f_x, _ex, _r = optimize_xspace_bestof(
        cost, fid, grad, seed, x_idx, restarts=R, maxiter=B // R, seed0=0, sigma_x=0.3)

    # arm 3: B/2 full, then θ_x best-of (R/2 restarts × (B/2)/(R/2))
    xh, _eh, _nh = _lbfgsb(cost, np.asarray(seed, float), maxiter=B // 2, grad=grad)
    f_split, _es, _rs = optimize_xspace_bestof(
        cost, fid, grad, xh, x_idx, restarts=max(2, R // 2),
        maxiter=(B // 2) // max(2, R // 2), seed0=1, sigma_x=0.3)

    return {"n": n, "h": h, "base": f_base, "xbest": float(f_x),
            "split": float(f_split), "prov": ws0["provenance"]}


def main():
    points = [(10, 0.3), (10, 0.5), (10, 0.7), (12, 0.5), (10, 1.3), (10, 1.8)]
    print(f"[fair] B={B} R={R} topo={TOPO} model={MODEL} p={P} J2={J2}", flush=True)
    print(f"{'N':>3} {'h':>4} | {'base_B':>7} {'xbest_B':>8} {'split_B':>8} | winner",
          flush=True)
    gx, gs = [], []
    for n, h in points:
        r = validate_point(n, h)
        cands = {"base": r["base"], "xbest": r["xbest"], "split": r["split"]}
        win = max(cands, key=cands.get)
        gx.append(r["xbest"] - r["base"])
        gs.append(r["split"] - r["base"])
        print(f"{r['n']:>3} {r['h']:>4.2f} | {r['base']:>7.4f} {r['xbest']:>8.4f} "
              f"{r['split']:>8.4f} | {win}", flush=True)
    gx, gs = np.array(gx), np.array(gs)
    print(f"\n[fair] vs base_B at EQUAL budget:", flush=True)
    print(f"  xbest: mean={gx.mean():+.4f} max={gx.max():+.4f} "
          f"win={int((gx > 0.005).sum())}/{len(gx)}", flush=True)
    print(f"  split: mean={gs.mean():+.4f} max={gs.max():+.4f} "
          f"win={int((gs > 0.005).sum())}/{len(gs)}", flush=True)
    best = "split" if gs.mean() >= gx.mean() else "xbest"
    if max(gx.mean(), gs.mean()) > 0.005:
        print(f"  → '{best}' is a genuinely better use of the budget → wire into default.",
              flush=True)
    else:
        print("  → ties at equal budget → the gain was just iterations; raise micro_descent.",
              flush=True)


if __name__ == "__main__":
    main()
