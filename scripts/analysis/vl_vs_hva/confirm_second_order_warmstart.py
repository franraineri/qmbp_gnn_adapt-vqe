#!/usr/bin/env python
"""Multi-seed confirmation of the second-order warm-start + regime-gate calibration.

Compares the first-order analytic warm-start against the second-order (frustration-
aware) warm-start across a dense h-sweep, with GENUINE seed variation (each method is
run as a perturbed best-of, so both have real distributions and the difference can be
tested in sigma units). Also records the correction magnitude (J/2h)^2 * c at each h
to calibrate a regime-gate for when the second-order term should be enabled.

Robustness / reusability (vs the earlier /tmp scripts):
- Per-(h, seed) checkpointing: results persist after EVERY seed, not per-h. A crash
  loses at most one seed.
- Resume: on restart, completed (h, seed) pairs are loaded from the checkpoint and
  skipped; only missing work runs.
- Ground-truth reuse: E0/gap come from GroundTruthCache (eigsh); psi is cached in
  memory per h within the run.
- Exception-safe: partial state is always on disk (written per seed).
- Parametrized (seeds, h-sweep, sigma, restarts, shrink/curv coefficients).

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/confirm_second_order_warmstart.py \
        --seeds 8 --sigma 0.15 --restarts 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import (  # noqa: E402
    analytic_warmstart_theta,
    analytic_warmstart_theta_2nd,
    eigsh_ground_state,
    exact_ground_state_vector,
    read_json,
    save_json,
    state_fidelity_exact,
    study_dir,
)

SUBDIR = "hva_nnn_sweep"
_OUT_FILE = "n9_confirm_second_order.json"

N, J2, J, P = 9, 0.5, 1.0, 2
MAXITER = 60
H_SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.2]


def _build(h, model):
    from qmbp_simulation import HVACircuitBuilder, make_lattice
    from qmbp_simulation.models.constants import STATEVECTOR_MAX_N
    from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

    if N <= STATEVECTOR_MAX_N:
        psi, e0, gap, H = exact_ground_state_vector(
            "square", N, h, model=model, j2=J2, return_hamiltonian=True
        )
    else:
        psi, e0, gap, H = eigsh_ground_state("square", N, h, model=model, j2=J2)
    lat = make_lattice("square", N, J=1.0, h=h)
    qc, _ = HVACircuitBuilder().create_bond_resolved_frustrated(N, P, lat)
    n_nn = len(lat.edges)
    n_nnn = len(HamiltonianBuilder._generate_nnn_edges(lat))
    return psi, e0, gap, H, qc, qc.num_parameters, n_nn, n_nnn


def _ws_2nd(n_nn, n_nnn, h, shrink_coef, curv_coef):
    """Second-order warm-start — now delegates to the promoted, unit-tested core.
    """
    return analytic_warmstart_theta_2nd(n_nn, n_nnn, N, P, h, J=J, J2=J2,
                                        shrink_coef=shrink_coef, curv_coef=curv_coef)


def _load_ckpt():
    """Resume the per-seed map from the checkpoint via the shared reader."""
    data = read_json(study_dir(SUBDIR) / _OUT_FILE)
    return data.get("per_seed", {}) if data else {}  # keyed "h|seed" -> {fid1, fid2}


def _summarize(per_seed):
    """Aggregate per-(h,seed) records into per-h mean/std/separation + gate."""
    by_h = {}
    for key, rec in per_seed.items():
        hk = key.split("|")[0]
        by_h.setdefault(hk, {"f1": [], "f2": [], "h": rec["h"],
                             "gap": rec["gap"], "correction_mag": rec["correction_mag"]})
        by_h[hk]["f1"].append(rec["fid1"])
        by_h[hk]["f2"].append(rec["fid2"])
    from qmbp_simulation.analysis.warmstart import aggregate_seed_stats

    data = {}
    for hk, v in sorted(by_h.items(), key=lambda kv: float(kv[0])):
        # mean/std/separation-in-sigma via the shared, unit-tested core.
        stats = aggregate_seed_stats(v["f1"], v["f2"])
        data[hk] = {
            "h": v["h"], "gap": v["gap"], "correction_mag": v["correction_mag"],
            **stats,
        }
    helps = [(v["h"], v["correction_mag"]) for v in data.values() if v["second_order_helps"]]
    if helps:
        gate = {"h_window": [min(h for h, _ in helps), max(h for h, _ in helps)],
                "correction_mag_range_where_helps":
                    [float(min(c for _, c in helps)), float(max(c for _, c in helps))],
                "rule": "enable 2nd-order when delta>0.01 AND separation>1sigma"}
    else:
        gate = {"h_window": None, "note": "no h where 2nd-order significantly helps"}
    return data, gate


def run(seeds, sigma, restarts, shrink_coef, curv_coef, model):
    from scipy.optimize import minimize

    from qmbp_simulation.execution import NoiselessBackend

    be = NoiselessBackend()
    per_seed = _load_ckpt()  # resume
    n_resumed = len(per_seed)
    if n_resumed:
        print(f"[resume] {n_resumed} (h,seed) records already done", flush=True)

    def opt(cost, x0, nP):
        r = minimize(cost, x0, method="L-BFGS-B", bounds=[(-np.pi, np.pi)] * nP,
                     options={"maxiter": MAXITER, "ftol": 1e-11})
        return r.x, float(r.fun)

    def perturbed_bestof(cost, th_seed, nP, seed):
        rng = np.random.default_rng(30000 + seed)
        x, e = opt(cost, th_seed, nP)
        bx, bE = x, e
        for _ in range(restarts - 1):
            x0 = np.clip(th_seed + rng.normal(0, sigma, nP), -np.pi, np.pi)
            x, e = opt(cost, x0, nP)
            if e < bE:
                bx, bE = x, e
        return bx

    def persist():
        data, gate = _summarize(per_seed)
        save_json({"schema": "confirm_second_order_v1",
                   "per_seed": per_seed, "data": data, "regime_gate": gate},
                  SUBDIR, _OUT_FILE,
                  params={"N": N, "J2": J2, "p_layers": P, "maxiter": MAXITER,
                          "n_seeds": seeds, "sigma": sigma, "restarts": restarts,
                          "shrink_coef": shrink_coef, "curv_coef": curv_coef,
                          "h_sweep": H_SWEEP, "model": model},
                  description="Multi-seed confirmation of the second-order warm-start "
                              "(genuine variation via perturbed best-of) + regime-gate")
        return data, gate

    for h in H_SWEEP:
        # skip build if all seeds for this h are already done
        pending = [s for s in range(seeds) if f"{h:.2f}|{s}" not in per_seed]
        if not pending:
            print(f"[h={h:.2f}] all {seeds} seeds cached — skip", flush=True)
            continue
        psi, e0, gap, H, qc, nP, n_nn, n_nnn = _build(h, model)

        def cost(x):
            return be.evaluate(qc, H, x)

        th1 = analytic_warmstart_theta(n_nn, n_nnn, N, P, h, J=J, J2=J2)
        th2 = _ws_2nd(n_nn, n_nnn, h, shrink_coef, curv_coef)
        corr_mag = (J / (2 * h)) ** 2 * shrink_coef

        for s in pending:
            x1 = perturbed_bestof(cost, th1, nP, s)
            x2 = perturbed_bestof(cost, th2, nP, s)
            per_seed[f"{h:.2f}|{s}"] = {
                "h": h, "seed": s, "gap": float(gap), "correction_mag": float(corr_mag),
                "fid1": float(state_fidelity_exact(qc, x1, psi)),
                "fid2": float(state_fidelity_exact(qc, x2, psi)),
            }
            persist()  # checkpoint after EVERY seed
            r = per_seed[f"{h:.2f}|{s}"]
            print(f"[h={h:.2f} seed={s}] 1st={r['fid1']:.4f} 2nd={r['fid2']:.4f} "
                  f"delta={r['fid2'] - r['fid1']:+.4f}", flush=True)

        data, _ = persist()
        q = data[f"{h:.2f}"]
        print(f"[h={h:.2f} SUMMARY corr={corr_mag:.3f}] 1st={q['fid1_mean']:.4f}+/-{q['fid1_std']:.3f} "
              f"2nd={q['fid2_mean']:.4f}+/-{q['fid2_std']:.3f} delta={q['delta_mean']:+.4f} "
              f"sep={q['separation_sigma']:+.2f}sigma help={q['second_order_helps']}", flush=True)

    data, gate = persist()
    print("\n[GATE]", json.dumps(gate), flush=True)
    print("ALL DONE", flush=True)
    return data, gate


def build_parser():
    p = argparse.ArgumentParser(description="Multi-seed confirmation of 2nd-order warm-start")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.15)
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--shrink-coef", type=float, default=1.0 / 3.0)
    p.add_argument("--curv-coef", type=float, default=1.0 / 6.0)
    p.add_argument("--model", default="tfim_frustrated")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(f"[confirm-2nd-order] seeds={args.seeds} sigma={args.sigma} restarts={args.restarts} "
          f"shrink={args.shrink_coef:.3f} curv={args.curv_coef:.3f}", flush=True)
    try:
        run(args.seeds, args.sigma, args.restarts, args.shrink_coef, args.curv_coef, args.model)
    except Exception:  # noqa: BLE001 — partial checkpoint preserved per seed
        import traceback
        print("[ERROR] run failed; per-seed checkpoint preserved.", flush=True)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
