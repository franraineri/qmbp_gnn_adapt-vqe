#!/usr/bin/env python
"""Validate the ensemble-donor warm-start candidate against the best single donor.

Measures, per (N, h), both raw (0-iter) init-fidelity and post-micro-descent
fidelity for:
  - best single same-phase donor (transferred, canonicalized)
  - the ensemble (circular mean of same-phase donors)
  - the analytic calibrated seed (reference floor)

Decision rule: keep the ensemble only if it beats the best single donor on
either metric in a regime by a margin > TOL. Pure measurement; writes nothing
except the log. Reuses ground_state / make_cost_fid and the runner's donor
loaders — nothing re-implemented.

Usage:
    .venv/bin/python -u scripts/analysis/vl_vs_hva/validate_ensemble_donor.py \
        | tee /tmp/ensemble_validate.log
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
    best_combined_warmstart,
    calibrated_warmstart_theta,
    ensemble_donor_seed,
    select_regime_seed,
    transfer_theta,
)
from qmbp_simulation.framework.study_core import ground_state, make_cost_fid  # noqa: E402
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402
from qmbp_simulation.models import make_lattice  # noqa: E402

# runner loaders (donor corpus)
from run_bond_topk_regime import _autodiscover_donors  # noqa: E402

TOL = 0.01            # margin to call the ensemble a win
MICRO_ITERS = 12      # short relaxation budget
J2 = 0.5
P = 2
TOPO = "square"


def _micro_descent(theta, cost, grad, maxiter=MICRO_ITERS):
    """Short L-BFGS-B relaxation; returns refined θ."""
    res = minimize(cost, np.asarray(theta, float), jac=grad, method="L-BFGS-B",
                   options={"maxiter": maxiter})
    return res.x


def _best_single_donor(donors, *, n_nn, n_nnn, n_qubits, h, fid, target_nnn_edges):
    """Init-fid of the best single same-phase donor (0 iters)."""
    fill, _ = select_regime_seed(n_nn, n_nnn, n_qubits, P, h, J=1.0, J2=J2)
    best = (-1.0, None, None)
    for d in donors:
        d_nq = d.get("n_qubits", n_qubits)
        blocks = ("nn", "nnn", "x") if (d_nq == n_qubits and d["n_nnn"] == n_nnn) \
            else ("nn", "x")
        seed = transfer_theta(
            d["theta"], donor_n_nn=d["n_nn"], donor_n_nnn=d["n_nnn"], donor_p=d["p"],
            target_n_nn=n_nn, target_n_nnn=n_nnn, target_p=P, n_qubits=n_qubits,
            donor_nnn_edges=d.get("nnn_edges"), target_nnn_edges=target_nnn_edges,
            fill_theta=fill, canonicalize=True, donor_n_qubits=d.get("n_qubits"),
            donor_blocks=blocks)
        if seed is None or seed.size != (n_nn + n_nnn + n_qubits) * P:
            continue
        f = fid(seed)
        if f > best[0]:
            best = (f, seed, d.get("label", "donor"))
    return best


def validate_point(n, h):
    lat, qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(TOPO, n, h, J2, P)
    cost, fid, grad, _ = make_cost_fid(qc, H, psi)
    target_nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)

    donors = _autodiscover_donors(TOPO, n, h, P, J2, max_donors=6) or []
    n_same = len(donors)

    # analytic floor
    cal = calibrated_warmstart_theta(n_nn, n_nnn, n, P, h, J2=J2)
    f_cal_raw = fid(cal)
    f_cal_md = fid(_micro_descent(cal, cost, grad))

    # best single donor
    f_single_raw, single_seed, single_lbl = _best_single_donor(
        donors, n_nn=n_nn, n_nnn=n_nnn, n_qubits=n, h=h, fid=fid,
        target_nnn_edges=target_nnn_edges)
    f_single_md = fid(_micro_descent(single_seed, cost, grad)) \
        if single_seed is not None else None

    # ensemble
    ens = ensemble_donor_seed(
        donors, n_nn=n_nn, n_nnn=n_nnn, n_qubits=n, p_layers=P, h=h, J=1.0, J2=J2,
        target_nnn_edges=target_nnn_edges)
    if ens is not None:
        ens_seed, k = ens
        f_ens_raw = fid(ens_seed)
        f_ens_md = fid(_micro_descent(ens_seed, cost, grad))
    else:
        ens_seed, k, f_ens_raw, f_ens_md = None, 0, None, None

    return {
        "n": n, "h": h, "gap": gap, "n_donors": n_same, "k_pooled": k,
        "cal_raw": f_cal_raw, "cal_md": f_cal_md,
        "single_raw": f_single_raw, "single_md": f_single_md, "single_lbl": single_lbl,
        "ens_raw": f_ens_raw, "ens_md": f_ens_md,
    }


def _fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else ("  -   " if x is None else str(x))


def main():
    # transition (where donors matter) + one paramagnet sanity
    points = [(10, 0.5), (10, 0.7), (12, 0.5), (10, 1.3)]
    print(f"[ensemble-validate] TOL={TOL} micro_iters={MICRO_ITERS} "
          f"topo={TOPO} p={P} J2={J2}", flush=True)
    print(f"{'N':>3} {'h':>4} {'#don':>4} {'k':>2} | "
          f"{'cal_raw':>8} {'cal_md':>7} | {'sgl_raw':>8} {'sgl_md':>7} | "
          f"{'ens_raw':>8} {'ens_md':>7} | verdict", flush=True)
    verdicts = []
    for n, h in points:
        r = validate_point(n, h)
        # verdict: ensemble win if it beats best single by > TOL on raw OR md
        win_raw = (r["ens_raw"] is not None and r["single_raw"] is not None
                   and r["ens_raw"] > r["single_raw"] + TOL)
        win_md = (r["ens_md"] is not None and r["single_md"] is not None
                  and r["ens_md"] > r["single_md"] + TOL)
        lose_raw = (r["ens_raw"] is not None and r["single_raw"] is not None
                    and r["ens_raw"] < r["single_raw"] - TOL)
        if r["ens_raw"] is None:
            v = "n/a (ensemble absent)"
        elif win_raw or win_md:
            v = "ENSEMBLE WINS"
        elif lose_raw and not win_md:
            v = "ensemble worse"
        else:
            v = "parity"
        verdicts.append(v)
        print(f"{r['n']:>3} {r['h']:>4.2f} {r['n_donors']:>4} {r['k_pooled']:>2} | "
              f"{_fmt(r['cal_raw']):>8} {_fmt(r['cal_md']):>7} | "
              f"{_fmt(r['single_raw']):>8} {_fmt(r['single_md']):>7} | "
              f"{_fmt(r['ens_raw']):>8} {_fmt(r['ens_md']):>7} | {v}", flush=True)

    print("\n[ensemble-validate] SUMMARY", flush=True)
    wins = sum(1 for v in verdicts if v == "ENSEMBLE WINS")
    worse = sum(1 for v in verdicts if v == "ensemble worse")
    print(f"  wins={wins}  worse={worse}  parity/na={len(verdicts) - wins - worse}",
          flush=True)
    if wins == 0:
        print("  → RECOMMENDATION: ensemble adds no value; keep include_ensemble=False "
              "(documented, not forced).", flush=True)
    else:
        print(f"  → RECOMMENDATION: ensemble helps in {wins} regime(s); enable "
              "include_ensemble there.", flush=True)


if __name__ == "__main__":
    main()
