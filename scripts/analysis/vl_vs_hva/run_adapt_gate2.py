#!/usr/bin/env python
"""ADAPT Gate 2 — production bond-growth at N18 h0.5 (hard regime).

Run ONLY after Gate 0 (gradient informative) and Gate 1 (ADAPT ≥ top-k at small
N) pass. This is the expensive production run: build the cheapest high-fidelity
masked ansatz at N18 in the hard frustrated regime, using every lever the gates
justified:

- ADAPT bond growth (``study_core.grow_bonds_adapt``) with per-step warm-start —
  each growth step continues from the previous step's θ, never cold.
- An ADAPTIVE grow-step: coarse early (add several nnn cheaply to lay down gross
  structure), fine near the target (add one at a time where resolution matters),
  bounding the N18 cost without losing ADAPT's fine selectivity.
- SEEDED FROM TOP-K: instead of growing from an empty nnn set, start from the
  known good top-k selection (strongest ``--seed-k`` nnn by ``|θ|`` of the full
  ansatz) so ADAPT refines a near-optimal structure rather than discovering it
  from scratch — far cheaper and safer at N18.
- An analytic/donor warm-start (``prepare_warmstart``: calibrated + regime +
  cross-N donors) for the FIRST step's θ.
- The efficiency (fid/CX) stop so the result is the Pareto point.

The full-θ ranking that seeds top-k is read from a provided ``--seed-npz`` (e.g.
the gate0/gate1 FULL-θ NPZ, or a zoo donor) WITHOUT reconverging the full ansatz
at N18; only if no NPZ is given (or it lacks this h) is the full ansatz converged
once as a fallback.

Persistence is MANDATORY and per-step (``on_step`` → crash-safe partial write
after every growth step, since each N18 step costs minutes), plus a final write.
The growth curve θ is upserted into a reusable NPZ; e0/gap go through the shared
``GroundTruthCache``.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/run_adapt_gate2.py \
        --n 18 --h 0.5 --seed-k 20 --seed-npz results/hva_vl_study/gate1_theta/gate1_full_theta_square_N18.npz \
        --target-fid 0.95 --restarts 2 --maxiter 2000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import STUDY_ROOT, StudyPersister, sync_scoreboard  # noqa: E402

from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.bond_mask import (  # noqa: E402
    bond_weights_from_theta,
    top_k_by_weight,
)
from qmbp_simulation.circuits.hva_variants import (  # noqa: E402
    AnsatzVariant,
    build_variant,
    make_masked_variant,
)
from qmbp_simulation.framework.study_core import (  # noqa: E402
    converge_circuit,
    cx_and_params,
    grow_bonds_adapt,
    ground_state,
    prepare_warmstart,
)
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402
from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache  # noqa: E402

SUBDIR = "bond_ablation"
MODEL = "tfim_frustrated"


def _theta_full_from_npz(npz_path, h, expected_len):
    """Read the FULL-ansatz θ for this h from a θ-NPZ, or None if unavailable."""
    if not npz_path or not Path(npz_path).exists():
        return None
    try:
        z = np.load(npz_path, allow_pickle=True)
    except Exception:
        return None
    hs = np.asarray(z["h_values"], float)
    thetas = np.asarray(z["theta_opt"], float)
    idx = int(np.argmin(np.abs(hs - h)))
    if abs(float(hs[idx]) - h) > 0.005:  # 2-decimal h convention
        return None
    theta = thetas[idx]
    if theta.shape[0] != expected_len:
        return None
    return theta


def _converge_full(builder, n, lat, H, psi, n_nn, n_nnn, h, j2, *,
                   restarts, maxiter, seed0, backend):
    """Fallback: converge the full frustrated p=1 ansatz once; return θ."""
    ref = AnsatzVariant(name="p1_full_ref", description="p=1 full frustrated (all bonds)",
                        blocks=["nn", "nnn", "x"], tags=("reference",))
    qc, _ = build_variant(builder, n, lat, ref)
    ws = prepare_warmstart(qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=n,
                           p_layers=1, h=h, J2=j2, backend=backend,
                           target_len=qc.num_parameters)
    _fid, _e, runs = converge_circuit(qc, H, psi, restarts=restarts, maxiter=maxiter,
                                      seed0=seed0, warm_theta=ws["seed"], backend=backend)
    return np.asarray(min(runs, key=lambda r: r["energy"])["theta_final"], float)


def _adaptive_grow_step(coarse, fine, switch_frac, n_nnn):
    """grow_step callable: ``coarse`` nnn/step until ``switch_frac`` of nnn chosen,
    then ``fine`` (default 1) for fine resolution near the target."""
    switch_at = switch_frac * n_nnn

    def _step(step_index, n_sel, n_tot):
        return coarse if n_sel < switch_at else fine

    return _step


def _seed_warmstart(builder, n, lat, H, psi, sel, n_nn, h, j2, backend):
    """Analytic/donor warm-start θ for the SEEDED selection's circuit layout."""
    v = make_masked_variant("adapt_seed", f"adapt seed ({sel.n_bonds} bonds)",
                            ["nn", "nnn", "x"], sel)
    qc, _ = build_variant(builder, n, lat, v)
    _n2q, npar = cx_and_params(qc)
    ws = prepare_warmstart(qc, H, psi, n_nn=n_nn, n_nnn=len(sel.nnn_edges),
                           n_qubits=n, p_layers=1, h=h, J2=j2,
                           target_nnn_edges=sel.nnn_edges, backend=backend,
                           target_len=npar)
    return np.asarray(ws["seed"], float), ws["provenance"]


def run(args) -> int:
    from qmbp_simulation.execution import NoiselessBackend

    builder = HVACircuitBuilder()
    gt_cache = GroundTruthCache()
    backend = NoiselessBackend()

    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(
        args.topology, args.n, args.h, args.j2, 1)
    nn_edges = list(lat.edges)
    nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)
    gt_cache.put(args.topology, args.n, MODEL, args.h, energy=e0, gap=gap,
                 method="eigsh_exact")
    gt_cache.flush()
    full_len = n_nn + n_nnn + args.n  # [nn | nnn | x] for the full ansatz

    print(f"[adapt_gate2] N={args.n} h={args.h} topology={args.topology} J2={args.j2} "
          f"e0={e0:.5f} gap={gap:.6f} n_nn={n_nn} n_nnn={n_nnn} "
          f"seed_k={args.seed_k} target_fid={args.target_fid}", flush=True)

    # Seed selection from top-k (|θ| of the full ansatz).
    theta_full = _theta_full_from_npz(args.seed_npz, args.h, full_len)
    if theta_full is not None:
        print(f"  seed θ_full from NPZ {Path(args.seed_npz).name}", flush=True)
    else:
        print("  no usable seed NPZ — converging the full ansatz once (fallback)",
              flush=True)
        theta_full = _converge_full(
            builder, args.n, lat, H, psi, n_nn, n_nnn, args.h, args.j2,
            restarts=args.full_restarts, maxiter=args.maxiter, seed0=args.seed0,
            backend=backend)

    w_nn, w_nnn = bond_weights_from_theta(theta_full, n_nn, n_nnn, args.n, 1)
    seed_k = min(args.seed_k, n_nnn)
    seed_sel = top_k_by_weight(nn_edges, nnn_edges, w_nn, w_nnn,
                               k_nn=None, k_nnn=seed_k)
    print(f"  seeded from top-{seed_k} nnn (|θ|) → start with "
          f"{seed_sel.n_bonds} bonds; ADAPT refines upward", flush=True)

    warm0, warm0_prov = _seed_warmstart(
        builder, args.n, lat, H, psi, seed_sel, n_nn, args.h, args.j2, backend)
    print(f"  step-0 warm-start: {warm0_prov}", flush=True)

    # Persistence (crash-safe per step).
    steps_ref: list[dict] = []
    out_file = f"adapt_gate2_{args.topology}_N{args.n}_h{args.h:.2f}.json"
    persister = StudyPersister(
        subdir=SUBDIR, out_file=out_file,
        fingerprint={"n_qubits": args.n, "p_layers": 1, "model": MODEL,
                     "topology": args.topology, "h": args.h},
        extra={"topology": args.topology, "N": args.n, "h": args.h, "p_layers": 1,
               "J2": args.j2, "e0": e0, "gap": gap, "seed_k": seed_k,
               "n_nnn_total": n_nnn, "target_fid": args.target_fid,
               "grow_coarse": args.grow_coarse, "grow_fine": args.grow_fine,
               "warm0_provenance": warm0_prov, "schema": "adapt_gate2_v1"},
        params={"experiment": "adapt_gate2", "N": args.n, "h": args.h},
        description="ADAPT Gate 2 — production bond growth at N18 (seeded from "
                    "top-k, adaptive grow-step, per-step warm-start)",
        rows_ref=steps_ref)

    def _on_step(step):
        steps_ref.append(step)
        persister.persist(steps_ref)  # crash-safe: minutes per N18 step
        print(f"  step{step['step']:>2} bonds={step['n_bonds']:>3} "
              f"(nnn={step['n_nnn_bonds']}) fid={step['best_fidelity']:.4f} "
              f"2q={step['n_2q_transpiled']} fid/cx={step['fidelity_per_cx']:.2e} "
              f"({step['seconds']}s)", flush=True)

    grow_step = _adaptive_grow_step(args.grow_coarse, args.grow_fine,
                                    args.grow_switch_frac, n_nnn)

    t0 = time.time()
    steps, reached = grow_bonds_adapt(
        builder, args.n, lat, H, psi, e0, gap, nn_all=nn_edges, nnn_all=nnn_edges,
        n_nn=n_nn, n_nnn=n_nnn, restarts=args.restarts, maxiter=args.maxiter,
        seed0=args.seed0, target_fid=args.target_fid, grow_step=grow_step,
        grad_tol=args.grad_tol, warm_per_step=True, efficiency_stop=True,
        efficiency_patience=args.efficiency_patience, seed_selection=seed_sel,
        warm_theta0=warm0, backend=backend, on_step=_on_step)

    # Final persistence + θ NPZ of the best (last) step.
    persister.persist(steps, status="final")
    _save_theta_npz(steps, args, e0, gap)

    best = max(steps, key=lambda s: s["best_fidelity"]) if steps else None
    print("\n=== GATE 2 RESULT (N18 production) ===", flush=True)
    for s in steps:
        print(f"  step{s['step']:>2} bonds={s['n_bonds']:>3} fid={s['best_fidelity']:.4f} "
              f"2q={s['n_2q_transpiled']:>4} fid/cx={s['fidelity_per_cx']:.2e}", flush=True)
    if best is not None:
        print(f"  BEST: fid={best['best_fidelity']:.4f} at {best['n_bonds']} bonds "
              f"({best['n_2q_transpiled']} CX), stop={steps[-1].get('stop_reason')}, "
              f"target_reached={reached}", flush=True)
    print(f"  total {time.time() - t0:.0f}s", flush=True)
    if not args.no_sync:
        sync_scoreboard()
    print("DONE", flush=True)
    return 0


def _save_theta_npz(steps, args, e0, gap) -> Path | None:
    """Upsert the best-step θ into a reusable per-(N,h) NPZ (anti-regression)."""
    from qmbp_simulation.framework.result_io import upsert_theta_npz

    pts = [s for s in steps if s.get("best_theta_final")]
    if not pts:
        return None
    best = max(pts, key=lambda s: s["best_fidelity"])
    theta = np.asarray(best["best_theta_final"], float)
    h_arr = np.array([args.h], float)
    theta_arr = theta.reshape(1, -1)
    e_vqe = np.array([best["e_best"]], float)
    e_exact = np.array([e0], float)
    gaps = np.array([gap], float)
    npz_dir = STUDY_ROOT / "gate2_theta"
    npz_dir.mkdir(parents=True, exist_ok=True)
    npz = npz_dir / f"gate2_best_theta_{args.topology}_N{args.n}_h{args.h:.2f}.npz"
    upsert_theta_npz(npz, h_arr, theta_arr, e_vqe, e_exact, gaps,
                     model=MODEL, j2=args.j2)
    return npz


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ADAPT Gate 2 — production N18 bond growth")
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=18)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--seed-k", type=int, default=20,
                   help="Seed ADAPT from the strongest k nnn (|θ| of full).")
    p.add_argument("--seed-npz", default=None,
                   help="θ-NPZ with the full-ansatz θ to derive the top-k seed "
                        "(avoids reconverging the full ansatz at N18).")
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--full-restarts", type=int, default=3,
                   help="Restarts for the fallback full-ansatz converge.")
    p.add_argument("--maxiter", type=int, default=2000)
    p.add_argument("--seed0", type=int, default=70000)
    p.add_argument("--target-fid", type=float, default=0.95)
    p.add_argument("--grad-tol", type=float, default=1e-6)
    p.add_argument("--grow-coarse", type=int, default=4,
                   help="nnn added per step BEFORE the switch fraction (coarse).")
    p.add_argument("--grow-fine", type=int, default=1,
                   help="nnn added per step AFTER the switch fraction (fine).")
    p.add_argument("--grow-switch-frac", type=float, default=0.6,
                   help="Switch from coarse to fine once this fraction of nnn chosen.")
    p.add_argument("--efficiency-patience", type=int, default=2)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
