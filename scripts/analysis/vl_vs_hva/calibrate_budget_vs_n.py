#!/usr/bin/env python
"""Calibrate how many micro-descent iters each N needs to cross f>0.9 (transition).

For the hard transition (h=0.5) at growing N, run the chosen cascade seed through
increasing L-BFGS-B budgets and record fidelity. The goal is the empirical law
iters_needed(N) so the adaptive budget scales with N (not just h). Only fidelity.

Reuses prepare_warmstart (micro_descent=0 → raw chosen seed) + _lbfgsb directly so
each budget is a single clean descent from the SAME seed (not cumulative restarts).

Usage:
    .venv/bin/python -u scripts/analysis/vl_vs_hva/calibrate_budget_vs_n.py \
        | tee /tmp/budget_vs_n.log
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from qmbp_simulation.framework.study_core import ground_state, make_cost_fid, prepare_warmstart  # noqa: E402
from qmbp_simulation.framework.study_runner import _lbfgsb  # noqa: E402

J2, P, TOPO, MODEL = 0.5, 2, "square", "tfim_frustrated"
BUDGETS = [24, 48, 96, 160, 260, 400]
TARGET = 0.90


def curve_for(n, h):
    lat, qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(TOPO, n, h, J2, P)
    cost, fid, grad, _ = make_cost_fid(qc, H, psi)
    ws = prepare_warmstart(qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=n,
                           p_layers=P, h=h, J=1.0, J2=J2, topology=TOPO,
                           model=MODEL, micro_descent=0, cost=cost, fid=fid, grad=grad)
    seed = np.asarray(ws["seed"], float)
    fids = {}
    for b in BUDGETS:
        x, _e, _n = _lbfgsb(cost, seed, maxiter=b, grad=grad)
        fids[b] = fid(x)
    # first budget crossing TARGET
    cross = next((b for b in BUDGETS if fids[b] >= TARGET), None)
    return n, h, ws["provenance"], fids, cross


def main():
    points = [(10, 0.5), (12, 0.5), (14, 0.5)]
    print(f"[budget-vs-N] transition h=0.5, target f>={TARGET}  budgets={BUDGETS}",
          flush=True)
    header = "  N |" + "".join(f" {b:>7}" for b in BUDGETS) + " | cross@ | prov"
    print(header, flush=True)
    rows = []
    for n, h in points:
        n, h, prov, fids, cross = curve_for(n, h)
        rows.append((n, cross))
        cells = "".join(f" {fids[b]:>7.4f}" for b in BUDGETS)
        print(f" {n:>2} |{cells} | {str(cross):>6} | {prov}", flush=True)

    print("\n[budget-vs-N] iters needed to cross f>=0.90:", flush=True)
    for n, cross in rows:
        print(f"  N={n:>2}: {cross if cross else '>'+str(BUDGETS[-1])}", flush=True)
    # crude linear/quadratic read on the crossing budget vs N
    known = [(n, c) for n, c in rows if c is not None]
    if len(known) >= 2:
        ns = np.array([k[0] for k in known], float)
        cs = np.array([k[1] for k in known], float)
        # fit c ~ a*N + b and c ~ a*N^2
        A1 = np.vstack([ns, np.ones_like(ns)]).T
        lin, *_ = np.linalg.lstsq(A1, cs, rcond=None)
        A2 = np.vstack([ns ** 2, np.ones_like(ns)]).T
        quad, *_ = np.linalg.lstsq(A2, cs, rcond=None)
        print(f"\n  linear fit  iters ≈ {lin[0]:.1f}·N + {lin[1]:.1f}", flush=True)
        print(f"  quad   fit  iters ≈ {quad[0]:.2f}·N² + {quad[1]:.1f}", flush=True)
        print(f"  → extrapolation N=18: linear≈{lin[0]*18+lin[1]:.0f}  "
              f"quad≈{quad[0]*18**2+quad[1]:.0f}", flush=True)


if __name__ == "__main__":
    main()
