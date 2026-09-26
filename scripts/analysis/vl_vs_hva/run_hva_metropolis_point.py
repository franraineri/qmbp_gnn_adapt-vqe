#!/usr/bin/env python
"""Single-point HVA warm-start + Metropolis basin-hopping runner (crash-safe, resumable).

Prepares the frustrated square-TFIM ground state at one (N, h) with the confirmed
best warm-start heuristic (analytic warm-start + Metropolis basin-hopping, T ~ gap)
and records exact fidelity + dE to the eigsh ground state. Built for large N (e.g.
N=18, 168 parameters) where a single optimization is expensive, so every hop is
checkpointed and the run can resume after an interruption.

Migrated to the shared study layer:
- Physics setup (ground state + circuit) via ``StudyRunner`` (single source of
  truth for the eigsh/exact ground state and the bond-resolved +NNN ansatz).
- Crash-safe per-hop persistence + resume via ``StudyCheckpoint`` /
  ``resume_ordered_list`` (framework core), written through the traceable
  ``save_json`` (result_path / source_script / generated_utc).
- Warm-start seed from the service ``analytic_warmstart_theta`` (delegates to the
  unit-tested core). The Metropolis loop stays here because it interleaves
  per-hop artifact writes; it uses the same RNG (``13000 + seed``) as before, so
  numbers are identical to the pre-migration script.

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/run_hva_metropolis_point.py \
        --topology square --n 18 --h 0.5 --j2 0.5 --p 2 --maxiter 150 --n-hops 8
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import (  # noqa: E402 (same dir)
    analytic_warmstart_theta,
    n_2q,
    read_json,
    resume_ordered_list,
    save_circuit_qpy,
    save_json,
    state_fidelity_exact,
    study_dir,
    transpile_hw,
)

SUBDIR = "hva_nnn_sweep"


def _paths(topology, n, p, h, maxiter):
    art = study_dir("resources") / f"n{n}_frustrated"
    art.mkdir(parents=True, exist_ok=True)
    tag = f"{topology}_N{n}_p{p}_h{h:.2f}_metropolis_mi{maxiter}"
    out_file = f"hva_nnn_{tag}.json"
    hops_dir = art / f"hops_{tag}"
    hops_dir.mkdir(parents=True, exist_ok=True)
    return art, out_file, hops_dir, tag


def run_point(topology, n, h, *, j2, p, maxiter, sigma, n_hops, model, seed):
    from scipy.optimize import minimize

    from qmbp_simulation.framework.study_runner import StudyRunner

    art, out_file, hops_dir, tag = _paths(topology, n, p, h, maxiter)
    t0 = time.time()

    # Ground state + circuit via the shared StudyRunner (eigsh for N>16, exact
    # vector otherwise; bond-resolved +NNN ansatz). Single source of truth.
    runner = StudyRunner(topology=topology, n_qubits=n, p_layers=p, j2=j2,
                         model=model, strategy="metropolis", maxiter=maxiter,
                         sigma=sigma, n_hops=n_hops, ansatz="nnn")
    from qmbp_simulation.models import make_lattice

    psi, e0, gap, H, gs_method = runner.ground_state(h)
    lat = make_lattice(topology, n, J=1.0, h=h)
    qc, n_nn, n_nnn = runner.build_circuit(lat)
    nP = qc.num_parameters
    be = runner.backend
    T = max(gap, 1e-6)

    def cost(x):
        return be.evaluate(qc, H, x)

    def opt(x0):
        r = minimize(cost, x0, method="L-BFGS-B", bounds=[(-np.pi, np.pi)] * nP,
                     options={"maxiter": maxiter, "ftol": 1e-12})
        return r.x, float(r.fun), int(r.nit)

    def save_hop_theta(hop_idx, theta):
        np.savez(hops_dir / f"hop{hop_idx:02d}.npz", theta=theta, hop=hop_idx)

    def persist(runs, x_best, e_best, done):
        """Crash-safe write after every hop: JSON + best-theta NPZ + (final) QPY."""
        fid = state_fidelity_exact(qc, x_best, psi)
        point = {
            "topology": topology, "h": h, "n_qubits": n, "ansatz": "nnn",
            "p_layers": p, "J2": j2, "model": model,
            "e_vqe": float(e_best), "e0_exact": float(e0), "gap": float(gap),
            "abs_error": float(abs(e_best - e0)),
            "de_over_gap": float(abs(e_best - e0) / gap) if gap > 0 else None,
            "fidelity": float(fid), "n_params": int(nP),
            "gs_method": gs_method,
            "optimizer": "warmstart+metropolis_basinhop", "maxiter": maxiter,
            "sigma": sigma, "n_hops": n_hops, "temperature": float(T),
            "runs": runs, "done": done, "wall_s": float(time.time() - t0),
            "hops_dir": str(hops_dir),
        }
        # best-theta NPZ always (so reuse works even mid-run)
        theta_npz = art / f"theta_{tag}.npz"
        np.savez(theta_npz, theta=x_best, h=h, e_vqe=e_best, e0=e0, gap=gap,
                 fidelity=fid, n_nn=n_nn, n_nnn=n_nnn, n_qubits=n, p_layers=p,
                 optimizer="warmstart+metropolis_basinhop", maxiter=maxiter, done=done)
        point["theta_npz"] = str(theta_npz)
        if done:
            bound = qc.assign_parameters(x_best)
            qc_t = transpile_hw(bound)
            point["cx"] = int(n_2q(qc_t))
            point["depth_2q"] = int(qc_t.depth())
            qpy_path = art / f"circuit_{tag}.qpy"
            save_circuit_qpy(bound, qpy_path)
            point["circuit_qpy"] = str(qpy_path)
        save_json({"schema": "hva_metropolis_point_v3", "point": point},
                  SUBDIR, out_file,
                  params={"N": n, "p_layers": p, "J2": j2, "model": model,
                          "maxiter": maxiter, "sigma": sigma, "n_hops": n_hops,
                          "optimizer": "warmstart+metropolis_basinhop",
                          "artifacts_dir": str(art)},
                  description=f"{topology} N={n} h={h} frustrated HVA nnn p={p}, Metropolis "
                              f"basin-hopping maxiter={maxiter}, crash-safe per-hop (v3)")
        return fid

    # --- Resume from checkpoint if present (shared helper) ---------------
    out_path = study_dir(SUBDIR) / out_file
    prior = read_json(out_path)
    runs = resume_ordered_list(out_path, "runs", inside="point")
    rng = np.random.default_rng(13000 + seed)
    if prior and runs and not prior.get("point", {}).get("done"):
        start_hop = len(runs) - 1  # runs[0] is hop0
        best_npz = np.load(art / f"theta_{tag}.npz")
        x_best = best_npz["theta"]; e_best = float(best_npz["e_vqe"])
        last_hop_npz = hops_dir / f"hop{start_hop:02d}.npz"
        x_cur = np.load(last_hop_npz)["theta"] if last_hop_npz.exists() else x_best.copy()
        e_cur = e_best
        print(f"[resume] continuing from hop {start_hop} (best_fid so far "
              f"{state_fidelity_exact(qc, x_best, psi):.4f})", flush=True)
    else:
        # Fresh start: warm-start, save it immediately (close the hop-0 blind spot)
        th_ws = analytic_warmstart_theta(n_nn, n_nnn, n, p, h, J=1.0, J2=j2)
        save_hop_theta(-1, th_ws)  # the raw warm-start seed, before optimization
        x_cur, e_cur, nit0 = opt(th_ws)
        x_best, e_best = x_cur.copy(), e_cur
        save_hop_theta(0, x_cur)
        runs = [{"hop": 0, "energy": float(e_cur),
                 "fidelity": float(state_fidelity_exact(qc, x_cur, psi)),
                 "accepted": True, "nit": nit0}]
        persist(runs, x_best, e_best, done=(n_hops == 0))
        print(f"[hop0] e={e_cur:.4f} fid={runs[0]['fidelity']:.4f} nit={nit0}", flush=True)
        start_hop = 0

    # --- Metropolis hops --------------------------------------------------
    for k in range(start_hop, n_hops):
        x0 = np.clip(x_cur + rng.normal(0, sigma, nP), -np.pi, np.pi)
        x_new, e_new, nit = opt(x0)
        de = e_new - e_cur
        accepted = de < 0 or rng.random() < np.exp(-de / T)
        if accepted:
            x_cur, e_cur = x_new, e_new
        if e_new < e_best:
            x_best, e_best = x_new.copy(), e_new
        save_hop_theta(k + 1, x_new)
        runs.append({"hop": k + 1, "energy": float(e_new),
                     "fidelity": float(state_fidelity_exact(qc, x_new, psi)),
                     "accepted": bool(accepted), "nit": nit})
        fid = persist(runs, x_best, e_best, done=(k == n_hops - 1))
        print(f"[hop{k+1}] e={e_new:.4f} acc={accepted} best_fid={fid:.4f} nit={nit}", flush=True)

    print("ALL DONE", flush=True)
    return e_best, x_best


def build_parser():
    p = argparse.ArgumentParser(description="HVA warm-start + Metropolis single-point runner")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=18)
    p.add_argument("--h", type=float, required=True)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--p", type=int, default=2)
    p.add_argument("--maxiter", type=int, default=150)
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--n-hops", type=int, default=8)
    p.add_argument("--model", default="tfim_frustrated")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(f"[metropolis-point] {args.topology} N={args.n} h={args.h} J2={args.j2} "
          f"p={args.p} maxiter={args.maxiter} sigma={args.sigma} n_hops={args.n_hops}",
          flush=True)
    try:
        run_point(args.topology, args.n, args.h, j2=args.j2, p=args.p,
                  maxiter=args.maxiter, sigma=args.sigma, n_hops=args.n_hops,
                  model=args.model, seed=args.seed)
    except Exception:  # noqa: BLE001 — persist partial state then re-raise
        print("[ERROR] run failed; partial checkpoint (if any) is preserved.", flush=True)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
