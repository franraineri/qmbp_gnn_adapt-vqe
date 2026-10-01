#!/usr/bin/env python
"""Head-to-head: renormalized θ_nn ≈ -0.112/h warm-start vs naive -0.25/h.

Tests whether the empirically-renormalized ZZ prefactor (from the θ-pattern
study) seeds a *better* optimum than the textbook first-order value, under a
FAIR, seed-controlled protocol:

- pure L-BFGS-B descent from the seed (NO restarts / basin-hopping) so the
  measured quality reflects the SEED, not the optimizer escaping it;
- multiple seeds via a small Gaussian perturbation around each candidate seed
  (same perturbation draws for both coefficients at a given seed index → paired
  comparison);
- swept over h across the transition and over N (8, 10).

Reports, per (N, h): mean ± σ over seeds of final fidelity and |ΔE|/gap for each
coefficient, the paired mean fidelity gain (renorm − naive), and its separation
in σ units. Uses the exact adjoint gradient for speed. Saves a JSON to the
organized tree and prints a summary.

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/warmstart_coef_shootout.py \
        --n 8 10 --hmin 0.3 --hmax 1.5 --nh 7 --seeds 6 --maxiter 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import save_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Warm-start coefficient shootout: -0.112/h vs -0.25/h")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, nargs="+", default=[8, 10])
    p.add_argument("--p", type=int, default=1)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--model", default="tfim_frustrated")
    p.add_argument("--hmin", type=float, default=0.3)
    p.add_argument("--hmax", type=float, default=1.5)
    p.add_argument("--nh", type=int, default=7)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--maxiter", type=int, default=300)
    p.add_argument("--perturb", type=float, default=0.15,
                   help="Gaussian σ of the per-seed perturbation around each candidate seed")
    p.add_argument("--coef-naive", type=float, default=0.25)
    p.add_argument("--coef-renorm", type=float, default=0.112)
    return p


def _build_seeds(n_nn, n_nnn, n_qubits, p_layers, h, *, J, j2, coef_naive, coef_renorm):
    """Build the four candidate warm-start seeds at (N, h). Returns {name: θ}.

    - naive       : first-order, zz_coef = coef_naive (textbook -0.25/h)
    - renorm      : first-order, zz_coef = coef_renorm (empirical -0.11/h, flat)
    - second_order: (J/2h)² shrink on ZZ + curvature on θ_x (report Q3)
    - so_nn_shrink: second-order + extra 0.4× shrink on θ_nn only (E1 study)
    """
    from qmbp_simulation.analysis.warmstart import (
        first_order_warmstart_theta,
        second_order_nn_shrink_theta,
        second_order_warmstart_theta,
    )

    return {
        "naive": first_order_warmstart_theta(n_nn, n_nnn, n_qubits, p_layers, h,
                                             J=J, J2=j2, zz_coef=coef_naive),
        "renorm": first_order_warmstart_theta(n_nn, n_nnn, n_qubits, p_layers, h,
                                              J=J, J2=j2, zz_coef=coef_renorm),
        "second_order": second_order_warmstart_theta(n_nn, n_nnn, n_qubits, p_layers, h,
                                                     J=J, J2=j2),
        "so_nn_shrink": second_order_nn_shrink_theta(n_nn, n_nnn, n_qubits, p_layers, h,
                                                    J=J, J2=j2),
    }


def _point(runner, h, *, seeds, maxiter, perturb, j2, coef_naive, coef_renorm,
           baseline="naive"):
    """Paired seed-controlled comparison of ALL candidate seeds at one (N, h).

    Every seed sees the SAME per-seed perturbation (paired). Gains are reported
    relative to ``baseline`` (default naive -0.25/h). Also names the per-h winner
    by mean fidelity.
    """
    from scipy.optimize import minimize

    from qmbp_simulation.framework.study_runner import _state_fidelity_exact
    from qmbp_simulation.models import make_lattice

    psi, e0, gap, H, _ = runner.ground_state(h)
    lat = make_lattice(runner.topology, runner.n_qubits, J=runner.J, h=h)
    qc, n_nn, n_nnn = runner.build_circuit(lat)
    grad = runner._make_adjoint_grad(qc, H)
    nP = qc.num_parameters

    def cost(x):
        return runner.backend.evaluate(qc, H, x)

    def descend(x0):
        r = minimize(cost, x0, method="L-BFGS-B", jac=grad,
                     bounds=[(-np.pi, np.pi)] * nP,
                     options={"maxiter": maxiter, "ftol": 1e-12})
        return r.x, float(r.fun), int(r.nit)

    cand = _build_seeds(n_nn, n_nnn, runner.n_qubits, runner.p_layers, h,
                        J=runner.J, j2=j2, coef_naive=coef_naive, coef_renorm=coef_renorm)
    res = {name: {"fid": [], "de_gap": [], "nit": []} for name in cand}
    for s in range(seeds):
        rng = np.random.default_rng(200_000 + 1000 * s)
        noise = rng.normal(0, perturb, nP)  # SAME noise for every candidate (paired)
        for name, seed in cand.items():
            x0 = np.clip(seed + noise, -np.pi, np.pi)
            x, e, nit = descend(x0)
            res[name]["fid"].append(_state_fidelity_exact(qc, x, psi))
            res[name]["de_gap"].append(abs(e - e0) / gap if gap > 0 else float("nan"))
            res[name]["nit"].append(nit)

    def stats(d):
        f = np.array(d["fid"]); g = np.array(d["de_gap"])
        return {"fid_mean": float(f.mean()), "fid_std": float(f.std()),
                "de_gap_mean": float(np.nanmean(g)), "nit_mean": float(np.mean(d["nit"]))}

    per_seed_stats = {name: stats(res[name]) for name in cand}
    base_fid = np.array(res[baseline]["fid"])
    gains = {}
    for name in cand:
        if name == baseline:
            continue
        d = np.array(res[name]["fid"]) - base_fid
        gains[name] = {"mean": float(d.mean()), "std": float(d.std()),
                       "sep_sigma": float(d.mean() / (d.std() + 1e-12))}
    winner = max(per_seed_stats, key=lambda k: per_seed_stats[k]["fid_mean"])
    return {
        "h": h, "gap": float(gap), "n_qubits": runner.n_qubits, "n_params": nP,
        "stats": per_seed_stats, "gain_vs_baseline": gains, "baseline": baseline,
        "winner": winner, "winner_fid": per_seed_stats[winner]["fid_mean"],
        "seed_fids": {name: res[name]["fid"] for name in cand},
    }


def main(argv=None) -> int:
    from qmbp_simulation.framework.study_runner import StudyRunner

    args = build_parser().parse_args(argv)
    h_values = [round(h, 2) for h in np.linspace(args.hmin, args.hmax, args.nh)]
    all_rows = []
    print(f"[shootout] {args.topology} N={args.n} p={args.p} J2={args.j2} "
          f"coef {args.coef_renorm} vs {args.coef_naive} | seeds={args.seeds} "
          f"h={args.hmin}..{args.hmax} ({args.nh})", flush=True)

    contenders = ["naive", "renorm", "second_order", "so_nn_shrink"]
    for N in args.n:
        runner = StudyRunner(topology=args.topology, n_qubits=N, p_layers=args.p,
                             j2=args.j2, model=args.model, strategy="first_order",
                             maxiter=args.maxiter, ansatz="nnn")
        for h in h_values:
            row = _point(runner, h, seeds=args.seeds, maxiter=args.maxiter,
                         perturb=args.perturb, j2=args.j2,
                         coef_naive=args.coef_naive, coef_renorm=args.coef_renorm)
            all_rows.append(row)
            fids = " ".join(f"{c[:4]}={row['stats'][c]['fid_mean']:.3f}" for c in contenders)
            print(f"  N={N} h={h:.2f} | {fids} | WIN={row['winner']}", flush=True)

    # aggregate: per-h winner counts + mean gain of each contender vs naive
    win_counts: dict[str, int] = {}
    for r in all_rows:
        win_counts[r["winner"]] = win_counts.get(r["winner"], 0) + 1
    mean_gain = {}
    for c in contenders:
        if c == "naive":
            continue
        gs = [r["gain_vs_baseline"][c]["mean"] for r in all_rows if c in r["gain_vs_baseline"]]
        mean_gain[c] = float(np.mean(gs)) if gs else None
    # regime split
    def regime_gain(c, lo, hi):
        gs = [r["gain_vs_baseline"][c]["mean"] for r in all_rows
              if lo <= r["h"] <= hi and c in r["gain_vs_baseline"]]
        return float(np.mean(gs)) if gs else None
    verdict = {
        "n_points": len(all_rows),
        "win_counts": win_counts,
        "mean_gain_vs_naive": mean_gain,
        "ordered_gain_vs_naive": {c: regime_gain(c, 0.0, 0.45) for c in contenders if c != "naive"},
        "transition_gain_vs_naive": {c: regime_gain(c, 0.4, 0.65) for c in contenders if c != "naive"},
        "coef_renorm": args.coef_renorm, "coef_naive": args.coef_naive,
    }
    save_json({"schema": "warmstart_coef_shootout_v2", "contenders": contenders,
               "rows": all_rows, "verdict": verdict},
              "theta_patterns", f"warmstart_coef_shootout_{args.topology}_p{args.p}_J2{args.j2:.2f}.json",
              params={"n_qubits": args.n, "p_layers": args.p, "J2": args.j2,
                      "seeds": args.seeds, "maxiter": args.maxiter, "perturb": args.perturb,
                      "h_values": h_values},
              description="4-way paired seed-controlled shootout: naive vs renorm vs "
                          "second-order vs second-order-nn-shrink warm-start (pure descent)")
    print(f"\n[verdict] per-h winners: {win_counts}")
    print(f"  mean fid gain vs naive (all h): {mean_gain}")
    print(f"  ordered phase (h≤0.45): {verdict['ordered_gain_vs_naive']}")
    print(f"  transition (0.4≤h≤0.65): {verdict['transition_gain_vs_naive']}")
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
