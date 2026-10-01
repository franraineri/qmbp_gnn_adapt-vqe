#!/usr/bin/env python
"""Validate the hybrid block-mix warm-start against the best single candidate.

Per (N, h), measures raw (0-iter) and post-micro-descent fidelity for:
  - block_mix   : the per-phase hybrid (best source per block, BLOCK_MIX_POLICY)
  - best_single : the best of {calibrated, structural, regime, best donor}
  - calibrated  : analytic floor

Decision: keep include_block_mix=True only if the hybrid matches or beats the
best single candidate (no regression) AND wins somewhere by > TOL. Reuses
ground_state / make_cost_fid, the runner's donor loader, and the warmstart
helpers — nothing re-implemented.

Usage:
    .venv/bin/python -u scripts/analysis/vl_vs_hva/validate_block_mix.py \
        | tee /tmp/block_mix_validate.log
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis" / "vl_vs_hva"))

from qmbp_simulation.analysis.warmstart import (  # noqa: E402
    block_mix_policy_for,
    block_mix_warmstart,
    calibrated_warmstart_theta,
    select_regime_seed,
    structural_warmstart_theta,
    transfer_theta,
)
from qmbp_simulation.framework.study_core import ground_state, make_cost_fid  # noqa: E402
from qmbp_simulation.models import make_lattice  # noqa: E402
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402
from run_bond_topk_regime import _autodiscover_donors  # noqa: E402

TOL = 0.01
MICRO_ITERS = 12
J2 = 0.5
P = 2
TOPO = "square"


def _md(theta, cost, grad, maxiter=MICRO_ITERS):
    return minimize(cost, np.asarray(theta, float), jac=grad, method="L-BFGS-B",
                    options={"maxiter": maxiter}).x


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
                                  fid=fid, target_nnn_edges=target_nnn_edges,
                                  fill=fill)

    src = {
        "calibrated": calibrated_warmstart_theta(n_nn, n_nnn, n, P, h, J2=J2),
        "structural": structural_warmstart_theta(n_nn, n_nnn, n, P, h, J2=J2),
        "regime": fill,
    }
    if donor_seed is not None:
        src["donor"] = donor_seed

    # best single candidate (raw + md)
    single = {lbl: fid(th) for lbl, th in src.items()}
    best_single_lbl = max(single, key=single.get)
    f_single_raw = single[best_single_lbl]
    f_single_md = fid(_md(src[best_single_lbl], cost, grad))

    # hybrid block-mix
    policy = block_mix_policy_for(h)
    default_src = "calibrated" if "calibrated" in src else "regime"
    mix = block_mix_warmstart(src, policy, n_nn=n_nn, n_nnn=n_nnn, n_qubits=n,
                              p_layers=P, default_source=default_src)
    f_mix_raw = fid(mix) if mix is not None else None
    f_mix_md = fid(_md(mix, cost, grad)) if mix is not None else None

    used = {b: (policy[b] if policy[b] in src else default_src) for b in ("nn", "nnn", "x")}
    return {
        "n": n, "h": h, "best_single": best_single_lbl,
        "single_raw": f_single_raw, "single_md": f_single_md,
        "mix_raw": f_mix_raw, "mix_md": f_mix_md, "policy": used,
    }


def _f(x):
    return f"{x:.4f}" if isinstance(x, float) else "  -   "


def main():
    points = [(10, 0.3), (10, 0.5), (10, 0.7), (12, 0.5), (10, 1.3), (10, 1.8)]
    print(f"[block-mix-validate] TOL={TOL} micro_iters={MICRO_ITERS} "
          f"topo={TOPO} p={P} J2={J2}", flush=True)
    print(f"{'N':>3} {'h':>4} | {'best_single':>11} {'sgl_raw':>8} {'sgl_md':>7} | "
          f"{'mix_raw':>8} {'mix_md':>7} | verdict  policy", flush=True)
    verdicts = []
    for n, h in points:
        r = validate_point(n, h)
        win = ((r["mix_raw"] is not None and r["mix_raw"] > r["single_raw"] + TOL)
               or (r["mix_md"] is not None and r["single_md"] is not None
                   and r["mix_md"] > r["single_md"] + TOL))
        regress = ((r["mix_md"] is not None and r["single_md"] is not None
                    and r["mix_md"] < r["single_md"] - TOL))
        v = "MIX WINS" if win else ("mix regress" if regress else "parity")
        verdicts.append(v)
        pol = f"{r['policy']['nn']}/{r['policy']['nnn']}/{r['policy']['x']}"
        print(f"{r['n']:>3} {r['h']:>4.2f} | {r['best_single']:>11} "
              f"{_f(r['single_raw']):>8} {_f(r['single_md']):>7} | "
              f"{_f(r['mix_raw']):>8} {_f(r['mix_md']):>7} | {v:>11}  {pol}",
              flush=True)

    wins = sum(1 for v in verdicts if v == "MIX WINS")
    regr = sum(1 for v in verdicts if v == "mix regress")
    print(f"\n[block-mix-validate] SUMMARY wins={wins} regress={regr} "
          f"parity={len(verdicts) - wins - regr}", flush=True)
    if regr == 0 and wins > 0:
        print("  → RECOMMENDATION: keep include_block_mix=True (no regression, "
              f"wins in {wins} regime(s)).", flush=True)
    elif regr == 0:
        print("  → RECOMMENDATION: safe (no regression) but no raw win — keep on; "
              "it is one extra free candidate the micro-descent can use.", flush=True)
    else:
        print(f"  → RECOMMENDATION: regresses in {regr} regime(s); review policy "
              "or gate the hybrid to winning phases only.", flush=True)


if __name__ == "__main__":
    main()
