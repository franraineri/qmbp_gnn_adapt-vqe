#!/usr/bin/env python
"""Diagnose which refinement LEVER most improves the default warm-start.

The default (prepare_warmstart strategy=combined) picks the best coherent
candidate + a 12-iter micro-descent. This measures, per (N, h), where that
default leaves fidelity on the table and which cheap lever recovers it:

  base : current default — best candidate + 12-iter full L-BFGS-B
  L    : longer micro-descent (24 iters) from the same best candidate
  X    : θ_x-subspace best-of (3 restarts × 12 iters) from the same candidate
  LX   : longer (24) + θ_x best-of combined

All levers reuse existing helpers (optimize_xspace_bestof, theta_x_indices,
_lbfgsb) — nothing new is invented yet. The winner per regime tells us whether
the gap is "more iterations", "escape the θ_x basin", or "already optimal".

Usage:
    .venv/bin/python -u scripts/analysis/vl_vs_hva/diagnose_warmstart_levers.py \
        | tee /tmp/warmstart_levers.log
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

J2 = 0.5
P = 2
TOPO = "square"
MODEL = "tfim_frustrated"


def validate_point(n, h):
    lat, qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(TOPO, n, h, J2, P)
    cost, fid, grad, _ = make_cost_fid(qc, H, psi)

    # current default: get the chosen candidate (its refined θ) at micro_descent=12
    ws = prepare_warmstart(qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=n,
                           p_layers=P, h=h, J=1.0, J2=J2, topology=TOPO,
                           model=MODEL, micro_descent=12, cost=cost, fid=fid, grad=grad)
    base_seed = ws["seed"]
    f_base = fid(base_seed)

    # L: longer descent from the SAME pre-descent best — approximate by continuing
    # the base seed (already 12-refined) a further 12 iters (total 24-equivalent).
    xL, _e, _n = _lbfgsb(cost, np.asarray(base_seed, float), maxiter=12, grad=grad)
    f_L = fid(xL)

    # X: θ_x-subspace best-of from the base seed
    x_idx = theta_x_indices(n_nn, n_nnn, n, P)
    fX, _eX, _runs = optimize_xspace_bestof(
        cost, fid, grad, base_seed, x_idx, restarts=3, maxiter=12, seed0=0, sigma_x=0.3)

    # LX: longer base then θ_x best-of
    fLX, _eLX, _r2 = optimize_xspace_bestof(
        cost, fid, grad, xL, x_idx, restarts=3, maxiter=12, seed0=1, sigma_x=0.3)

    return {"n": n, "h": h, "prov": ws["provenance"],
            "base": f_base, "L": f_L, "X": float(fX), "LX": float(fLX)}


def main():
    points = [(10, 0.3), (10, 0.5), (10, 0.7), (12, 0.5), (10, 1.3), (10, 1.8)]
    print(f"[levers] topo={TOPO} model={MODEL} p={P} J2={J2}", flush=True)
    print(f"{'N':>3} {'h':>4} | {'base':>7} {'L(+12)':>7} {'X(xbest)':>8} "
          f"{'LX':>7} | best lever  provenance", flush=True)
    gains = {"L": [], "X": [], "LX": []}
    for n, h in points:
        r = validate_point(n, h)
        cands = {"base": r["base"], "L": r["L"], "X": r["X"], "LX": r["LX"]}
        best = max(cands, key=cands.get)
        for k in ("L", "X", "LX"):
            gains[k].append(r[k] - r["base"])
        print(f"{r['n']:>3} {r['h']:>4.2f} | {r['base']:>7.4f} {r['L']:>7.4f} "
              f"{r['X']:>8.4f} {r['LX']:>7.4f} | {best:>9}  {r['prov']}", flush=True)

    print("\n[levers] MEAN GAIN vs base (fidelity delta)", flush=True)
    for k in ("L", "X", "LX"):
        g = np.array(gains[k])
        print(f"  {k:>3}: mean={g.mean():+.4f}  max={g.max():+.4f}  "
              f"n_improved={int((g > 0.005).sum())}/{len(g)}", flush=True)
    print("\n[levers] READING", flush=True)
    print("  If L wins → gap is iteration budget (cheap: raise micro_descent).", flush=True)
    print("  If X wins → gap is the θ_x basin (add θ_x best-of to the default).", flush=True)
    print("  If base ties → default already optimal at this budget.", flush=True)


if __name__ == "__main__":
    main()
