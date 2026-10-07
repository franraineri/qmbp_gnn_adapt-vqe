#!/usr/bin/env python
"""ADAPT Option B — canonical operator-list growth (no layers), Δfid / Δ2q.

Where Option A grows by {add nnn bond, repeat whole layer}, Option B is the
literal ADAPT-VQE: a pool of INDIVIDUAL operators (every RZZ edge + single-qubit
rotations) with NO block/layer structure. Each step appends the single operator
whose energy gradient is largest at the current optimum — "add the gate that
helps most, wherever it helps" — re-converging warm. The target is the same:
highest fidelity for the fewest two-qubit gates.

Per h-point (reusing only existing modules):
  1. ``study_core.ground_state`` → lattice, H, exact ψ, E0, gap.
  2. ``study_core.grow_adapt_operators`` → the operator-by-operator growth curve,
     warm per step, crash-safe via ``on_step``.
  3. Converge the p=2 FULL ansatz as the EXPRESSIVITY ANCHOR, to see whether the
     canonical ADAPT reaches p=2 fidelity at a lower 2q cost (better fid/CX).

Persistence: crash-safe partial+final via ``StudyPersister`` (one row per h, plus
per-operator partials inside each point). e0/gap via the shared ``GroundTruthCache``;
the best ADAPT-B θ per h upserted into a reusable NPZ (only when θ length is
uniform across h — operator-grown circuits usually differ, so it is skipped then).

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/run_adapt_poolB.py \
        --n 10 --h-list 0.3 0.5 0.7 0.9 --restarts 3 --maxiter 1500 \
        --grow-step 1 --max-ops 120 --target-fid 0.95
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
from qmbp_simulation.circuits.hva_variants import AnsatzVariant, build_variant  # noqa: E402
from qmbp_simulation.framework.study_core import (  # noqa: E402
    converge_circuit,
    cx_and_params,
    ground_state,
    grow_adapt_operators,
    prepare_warmstart,
)
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402
from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache  # noqa: E402

SUBDIR = "bond_ablation"
MODEL = "tfim_frustrated"


def _converge_full_pN(builder, n, lat, H, psi, n_nn, n_nnn, p, h, j2, *,
                      restarts, maxiter, seed0, backend):
    """Converge the full frustrated p-layer ansatz anchor; return (fid, n_2q)."""
    ref = AnsatzVariant(name=f"p{p}_full", description=f"p={p} full frustrated",
                        blocks=["nn", "nnn", "x"] * p, tags=("anchor",))
    qc, _ = build_variant(builder, n, lat, ref)
    n_2q, _npar = cx_and_params(qc)
    ws = prepare_warmstart(qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=n,
                           p_layers=p, h=h, J2=j2, backend=backend,
                           target_len=qc.num_parameters)
    fid, _e, runs = converge_circuit(qc, H, psi, restarts=restarts, maxiter=maxiter,
                                     seed0=seed0, warm_theta=ws["seed"], backend=backend)
    return float(fid), n_2q


def _point(args, builder, backend, gt_cache, persister, rows, h) -> dict:
    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(
        args.topology, args.n, h, args.j2, 1)
    nn_edges = list(lat.edges)
    nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)
    gt_cache.put(args.topology, args.n, MODEL, h, energy=e0, gap=gap,
                 method="eigsh_exact")

    op_steps_ref: list[dict] = []

    def _on_op_step(step):
        op_steps_ref.append(step)
        persister.persist(rows + [{"h": round(float(h), 2), "status": "in_progress",
                                   "op_steps": op_steps_ref}])

    t0 = time.time()
    steps, reached = grow_adapt_operators(
        builder, args.n, lat, H, psi, e0, gap, nn_all=nn_edges, nnn_all=nnn_edges,
        restarts=args.restarts, maxiter=args.maxiter, seed0=args.seed0,
        target_fid=args.target_fid, include_rotations=not args.no_rotations,
        grow_step=args.grow_step, max_ops=args.max_ops,
        efficiency_patience=args.efficiency_patience,
        min_dfid_keep_growing=args.min_dfid, backend=backend, on_step=_on_op_step)

    best = max(steps, key=lambda s: s["best_fidelity"])
    b_fid = best["best_fidelity"]
    b_2q = best["n_2q_transpiled"]
    b_fpc = best["fidelity_per_cx"]

    p2_fid, p2_2q = _converge_full_pN(
        builder, args.n, lat, H, psi, n_nn, n_nnn, 2, h, args.j2,
        restarts=args.restarts, maxiter=args.maxiter, seed0=args.seed0, backend=backend)
    p2_fpc = (p2_fid / p2_2q) if p2_2q > 0 else None

    beats_p2_fid = b_fid >= p2_fid - args.fid_tol
    better_fid_per_cx = (b_fpc is not None and p2_fpc is not None and b_fpc > p2_fpc)

    row = {
        "h": round(float(h), 2), "n_qubits": args.n, "topology": args.topology,
        "e0": e0, "gap": gap, "n_nn": n_nn, "n_nnn": n_nnn,
        "poolB_best_fidelity": b_fid, "poolB_2q": b_2q, "poolB_fid_per_cx": b_fpc,
        "poolB_n_ops": best["n_ops"], "poolB_reached_target": reached,
        "poolB_stop_reason": steps[-1].get("stop_reason"), "poolB_n_steps": len(steps),
        "poolB_last_ops": [s["last_op"] for s in steps],
        "p2_full_fidelity": p2_fid, "p2_full_2q": p2_2q, "p2_full_fid_per_cx": p2_fpc,
        "poolB_matches_p2_fid": beats_p2_fid,
        "poolB_better_fid_per_cx_than_p2": better_fid_per_cx,
        "best_theta_final": best["best_theta_final"],
        "seconds": round(time.time() - t0, 1),
    }
    verdict = "B≥p2 fid" if beats_p2_fid else "B<p2 fid"
    fpc_tag = "fid/cx>p2" if better_fid_per_cx else "fid/cx≤p2"
    print(f"  h={h:.2f} gap={gap:.4f} | B fid={b_fid:.4f} 2q={b_2q} ops={best['n_ops']} "
          f"fpc={b_fpc:.2e} | p2 fid={p2_fid:.4f} 2q={p2_2q} | {verdict} {fpc_tag} "
          f"({row['seconds']}s)", flush=True)
    return row


def _save_theta_npz(rows, args) -> Path | None:
    from qmbp_simulation.framework.result_io import upsert_theta_npz

    pts = [r for r in rows if r.get("best_theta_final") and "poolB_best_fidelity" in r]
    if not pts or len({len(r["best_theta_final"]) for r in pts}) != 1:
        return None  # operator-grown circuits usually have different θ lengths per h
    h_arr = np.array([r["h"] for r in pts], float)
    theta_arr = np.array([r["best_theta_final"] for r in pts], float)
    e_vqe = np.array([r["e0"] + abs(r["e0"]) * (1.0 - r["poolB_best_fidelity"]) for r in pts], float)
    e_exact = np.array([r["e0"] for r in pts], float)
    gaps = np.array([r["gap"] for r in pts], float)
    npz_dir = STUDY_ROOT / "poolB_theta"
    npz_dir.mkdir(parents=True, exist_ok=True)
    npz = npz_dir / f"poolB_best_theta_{args.topology}_N{args.n}.npz"
    upsert_theta_npz(npz, h_arr, theta_arr, e_vqe, e_exact, gaps,
                     model=MODEL, j2=args.j2)
    return npz


def run(args) -> int:
    from qmbp_simulation.execution import NoiselessBackend

    builder = HVACircuitBuilder()
    gt_cache = GroundTruthCache()
    out_file = f"adapt_poolB_{args.topology}_N{args.n}.json"

    print(f"[adapt_poolB] N={args.n} topology={args.topology} J2={args.j2} "
          f"h_list={args.h_list} grow_step={args.grow_step} max_ops={args.max_ops} "
          f"rotations={not args.no_rotations} restarts={args.restarts} "
          f"maxiter={args.maxiter} target_fid={args.target_fid}", flush=True)

    rows: list[dict] = []
    persister = StudyPersister(
        subdir=SUBDIR, out_file=out_file,
        fingerprint={"n_qubits": args.n, "p_layers": None, "model": MODEL,
                     "topology": args.topology, "h": None},
        extra={"topology": args.topology, "N": args.n, "J2": args.j2,
               "grow_step": args.grow_step, "max_ops": args.max_ops,
               "include_rotations": not args.no_rotations,
               "target_fid": args.target_fid, "schema": "adapt_poolB_v1"},
        params={"experiment": "adapt_poolB", "N": args.n},
        description="ADAPT Option B — canonical operator-list growth (no layers) "
                    "by Δfid/Δ2q, vs p=2 full expressivity anchor",
        rows_ref=rows)

    for h in args.h_list:
        backend = NoiselessBackend()
        row = _point(args, builder, backend, gt_cache, persister, rows, h)
        rows.append(row)
        persister.persist(rows)

    gt_cache.flush()
    npz = _save_theta_npz(rows, args)
    persister.persist(rows, status="final")

    n_match = sum(1 for r in rows if r["poolB_matches_p2_fid"])
    n_better_fpc = sum(1 for r in rows if r["poolB_better_fid_per_cx_than_p2"])
    print("\n=== POOL-B VERDICT (vs p=2 full anchor) ===", flush=True)
    print(f"  ADAPT-B reaches p2 fidelity: {n_match}/{len(rows)} | "
          f"better fid-per-CX than p2: {n_better_fpc}/{len(rows)}", flush=True)
    if rows:
        mean_b = float(np.mean([r["poolB_best_fidelity"] for r in rows]))
        mean_p2 = float(np.mean([r["p2_full_fidelity"] for r in rows]))
        print(f"  mean fidelity — ADAPT-B: {mean_b:.4f}  p2-full: {mean_p2:.4f}", flush=True)
    verdict = ("CANONICAL ADAPT reaches p2 expressivity, cheaper per-CX where it wins"
               if n_match >= max(1, len(rows) // 2)
               else "ADAPT-B below p2 — needs more ops / restarts, or deeper ceiling")
    print(f"  → {verdict}", flush=True)
    if npz is not None:
        print(f"  ADAPT-B θ saved: {npz.relative_to(STUDY_ROOT.parent.parent)}", flush=True)
    if not args.no_sync:
        sync_scoreboard()
    print("DONE", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ADAPT Option B — canonical operator-list growth")
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--h-list", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    p.add_argument("--grow-step", type=int, default=1,
                   help="Operators appended per step (1 = finest, canonical ADAPT).")
    p.add_argument("--max-ops", type=int, default=200,
                   help="Cap on total operators in the grown circuit.")
    p.add_argument("--no-rotations", action="store_true",
                   help="Pool of RZZ edges only (no single-qubit rotations).")
    p.add_argument("--restarts", type=int, default=3,
                   help="Must be >=2 so a freshly appended operator can activate.")
    p.add_argument("--maxiter", type=int, default=1500)
    p.add_argument("--seed0", type=int, default=70000)
    p.add_argument("--target-fid", type=float, default=0.95)
    p.add_argument("--efficiency-patience", type=int, default=3)
    p.add_argument("--min-dfid", type=float, default=0.01,
                   help="Keep growing while a step's absolute Δfidelity ≥ this.")
    p.add_argument("--fid-tol", type=float, default=0.005,
                   help="ADAPT-B counts as matching p2 if within this of p2 fidelity.")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
