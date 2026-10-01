#!/usr/bin/env python
"""Validate the two COUPLED hybrid warm-starts (respect fidelity non-separability).

Both start from a single coherent θ and refine it against the FULL state, so the
inter-block coupling is respected (unlike the spliced block_mix, which regressed).

Compared per (N, h) at an EQUALIZED optimization budget B:
  - baseline : best single candidate + full micro-descent (B iters)
  - opt1     : best single candidate + block_coordinate_descent (nnn→nn→x),
               sweeps×blocks×maxiter_per_block ≈ B
  - opt2     : block_mix seed + full micro-descent (B iters)

Metric: post-refinement fidelity (all three get the SAME budget B, so this is a
fair "which refinement path reaches the better basin" test). Also reports raw
(0-iter) fidelities for context.

Reuses ground_state / make_cost_fid, study_runner._lbfgsb, the warmstart helpers
and the runner's donor loader — nothing re-implemented.

Usage:
    .venv/bin/python -u scripts/analysis/vl_vs_hva/validate_coupled_hybrid.py \
        | tee /tmp/coupled_hybrid_validate.log
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis" / "vl_vs_hva"))

from qmbp_simulation.analysis.warmstart import (  # noqa: E402
    block_coordinate_descent,
    block_mix_policy_for,
    block_mix_warmstart,
    calibrated_warmstart_theta,
    select_regime_seed,
    structural_warmstart_theta,
    transfer_theta,
)
from qmbp_simulation.framework.study_core import ground_state, make_cost_fid  # noqa: E402
from qmbp_simulation.framework.study_runner import _lbfgsb  # noqa: E402
from qmbp_simulation.models import make_lattice  # noqa: E402
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402
from run_bond_topk_regime import _autodiscover_donors  # noqa: E402

TOL = 0.01
BUDGET = 12            # total iters for baseline/opt2 full descent
CD_SWEEPS = 2          # opt1: 2 sweeps × 3 blocks × 2 iters = 12 (budget-matched)
CD_MAXITER = 2
J2 = 0.5
P = 2
TOPO = "square"


def _full_descent(theta, cost, grad, maxiter=BUDGET):
    x, _f, _n = _lbfgsb(cost, np.asarray(theta, float), maxiter=maxiter, grad=grad)
    return x


def _best_donor_seed(donors, *, n_nn, n_nnn, n, h, fid, target_nnn_edges, fill):
    best = (-1.0, None)
    for d in donors:
        d_nq = d.get("n_qubits", n)
        blocks = ("nn", "nnn", "x") if (d_nq == n and d["n_nnn"] == n_nnn) else ("nn", "x")
        seed = transfer_theta(
            d["theta"], donor_n_nn=d["n_nn"], donor_n_nnn=d["n_nnn"], donor_p=d["p"],
            target_n_nn=n_nn, target_n_nnn=n_nnn, target_p=P, n_qubits=n,
            donor_nnn_edges=d.get("nnn_edges"), target_nnn_edges=target_nnn_edges,
            fill_theta=fill, canonicalize=True, donor_n_qubits=d.get("n_qubits"),
            donor_blocks=blocks)
        if seed is not None and seed.size == (n_nn + n_nnn + n) * P:
            f = fid(seed)
            if f > best[0]:
                best = (f, seed)
    return best[1]


def validate_point(n, h):
    lat, qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(TOPO, n, h, J2, P)
    cost, fid, grad, _ = make_cost_fid(qc, H, psi)
    target_nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)
    fill, _ = select_regime_seed(n_nn, n_nnn, n, P, h, J=1.0, J2=J2)

    donors = _autodiscover_donors(TOPO, n, h, P, J2, max_donors=6) or []
    donor_seed = _best_donor_seed(donors, n_nn=n_nn, n_nnn=n_nnn, n=n, h=h,
                                  fid=fid, target_nnn_edges=target_nnn_edges, fill=fill)

    src = {
        "calibrated": calibrated_warmstart_theta(n_nn, n_nnn, n, P, h, J2=J2),
        "structural": structural_warmstart_theta(n_nn, n_nnn, n, P, h, J2=J2),
        "regime": fill,
    }
    if donor_seed is not None:
        src["donor"] = donor_seed

    # best single candidate (by raw fid)
    raw = {lbl: fid(th) for lbl, th in src.items()}
    best_lbl = max(raw, key=raw.get)
    best_seed = src[best_lbl]

    # baseline: best single + full descent
    f_base = fid(_full_descent(best_seed, cost, grad))

    # opt1: best single + coordinate descent (budget-matched)
    cd = block_coordinate_descent(
        best_seed, cost, grad, n_nn=n_nn, n_nnn=n_nnn, n_qubits=n, p_layers=P,
        lbfgsb=_lbfgsb, order=("nnn", "nn", "x"), sweeps=CD_SWEEPS,
        maxiter_per_block=CD_MAXITER)
    f_opt1 = fid(cd)

    # opt2: block_mix seed + full descent
    policy = block_mix_policy_for(h)
    default_src = "calibrated" if "calibrated" in src else "regime"
    mix = block_mix_warmstart(src, policy, n_nn=n_nn, n_nnn=n_nnn, n_qubits=n,
                              p_layers=P, default_source=default_src)
    f_opt2 = fid(_full_descent(mix, cost, grad)) if mix is not None else None

    return {
        "n": n, "h": h, "best_lbl": best_lbl, "best_raw": raw[best_lbl],
        "base": f_base, "opt1": f_opt1, "opt2": f_opt2,
    }


def _f(x):
    return f"{x:.4f}" if isinstance(x, float) else "  -   "


def main():
    points = [(10, 0.3), (10, 0.5), (10, 0.7), (12, 0.5), (10, 1.3), (10, 1.8)]
    print(f"[coupled-hybrid] budget={BUDGET} cd_sweeps={CD_SWEEPS}x3x{CD_MAXITER} "
          f"TOL={TOL} topo={TOPO} p={P} J2={J2}", flush=True)
    print(f"{'N':>3} {'h':>4} | {'best':>11} {'raw':>7} | {'baseline':>8} "
          f"{'opt1_cd':>8} {'opt2_mix':>8} | verdict", flush=True)
    v1_wins = v2_wins = v1_reg = v2_reg = 0
    for n, h in points:
        r = validate_point(n, h)
        d1 = (r["opt1"] - r["base"])
        d2 = (r["opt2"] - r["base"]) if r["opt2"] is not None else None
        tags = []
        if d1 > TOL:
            tags.append("opt1+"); v1_wins += 1
        elif d1 < -TOL:
            tags.append("opt1-"); v1_reg += 1
        if d2 is not None and d2 > TOL:
            tags.append("opt2+"); v2_wins += 1
        elif d2 is not None and d2 < -TOL:
            tags.append("opt2-"); v2_reg += 1
        verdict = " ".join(tags) if tags else "parity"
        print(f"{r['n']:>3} {r['h']:>4.2f} | {r['best_lbl']:>11} {_f(r['best_raw']):>7} | "
              f"{_f(r['base']):>8} {_f(r['opt1']):>8} {_f(r['opt2']):>8} | {verdict}",
              flush=True)

    print(f"\n[coupled-hybrid] SUMMARY  opt1: wins={v1_wins} reg={v1_reg} | "
          f"opt2: wins={v2_wins} reg={v2_reg}", flush=True)
    print("[coupled-hybrid] RECOMMENDATION", flush=True)
    for name, w, rg in (("opt1 (coordinate descent)", v1_wins, v1_reg),
                        ("opt2 (block_mix + descent)", v2_wins, v2_reg)):
        if rg == 0 and w > 0:
            print(f"  {name}: KEEP — wins {w}, no regression.", flush=True)
        elif rg == 0:
            print(f"  {name}: neutral — no regression, no win (same basin at budget).",
                  flush=True)
        else:
            print(f"  {name}: regresses in {rg} — do not enable.", flush=True)


if __name__ == "__main__":
    main()
