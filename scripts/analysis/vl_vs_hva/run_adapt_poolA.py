#!/usr/bin/env python
"""ADAPT Option A — unrestricted-p bond growth by Δfidelity / Δ(2q gates).

The N14 gates showed the hard-regime ceiling is p=1 EXPRESSIVITY, not bond
selection: even the full p=1 ansatz saturates well below useful fidelity near
h_c. Option A removes the fixed-p restriction: ADAPT grows the circuit by the
action — add an nnn bond OR repeat an entangling layer — that buys the most
fidelity per two-qubit gate, applied only where the gradient / cost trade-off
justifies it (a local, data-driven stand-in for raising p).

Per h-point (reusing only existing modules):
  1. ``study_core.ground_state`` → lattice, H, exact ψ, E0, gap.
  2. ``study_core.grow_adapt_pool`` → the Δfid/Δ2q-greedy growth curve, warm
     per step, crash-safe via ``on_step``.
  3. Converge the p=2 FULL ansatz as the EXPRESSIVITY ANCHOR: the question is
     whether Pool-A reaches the p=2 fidelity at a LOWER 2q cost (better fid/CX),
     or beats p=2 outright.

Reports, per h: Pool-A best fidelity / 2q / fid-per-CX, the p=2 anchor's, and
whether Pool-A broke the p=1 ceiling and how it compares to p=2 on fid-per-CX.

Persistence: crash-safe partial+final via ``StudyPersister`` (one row per h, plus
per-step partials inside each point). e0/gap via the shared ``GroundTruthCache``;
the best Pool-A θ per h upserted into a reusable NPZ.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/run_adapt_poolA.py \
        --n 10 --h-list 0.3 0.5 0.7 0.9 --restarts 3 --maxiter 1500 --target-fid 0.95
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
    grow_adapt_pool,
    prepare_warmstart,
)
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402
from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache  # noqa: E402

SUBDIR = "bond_ablation"
MODEL = "tfim_frustrated"


def _converge_full_pN(builder, n, lat, H, psi, n_nn, n_nnn, p, h, j2, *,
                      restarts, maxiter, seed0, backend):
    """Converge the full frustrated p-layer ansatz; return (fid, n_2q, npar)."""
    ref = AnsatzVariant(name=f"p{p}_full", description=f"p={p} full frustrated",
                        blocks=["nn", "nnn", "x"] * p, tags=("anchor",))
    qc, _ = build_variant(builder, n, lat, ref)
    n_2q, npar = cx_and_params(qc)
    ws = prepare_warmstart(qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=n,
                           p_layers=p, h=h, J2=j2, backend=backend,
                           target_len=qc.num_parameters)
    fid, _e, runs = converge_circuit(qc, H, psi, restarts=restarts, maxiter=maxiter,
                                     seed0=seed0, warm_theta=ws["seed"], backend=backend)
    return float(fid), n_2q, npar


def _point(args, builder, backend, gt_cache, persister, rows, h) -> dict:
    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(
        args.topology, args.n, h, args.j2, 1)
    nn_edges = list(lat.edges)
    nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)
    gt_cache.put(args.topology, args.n, MODEL, h, energy=e0, gap=gap,
                 method="eigsh_exact")

    pool_steps_ref: list[dict] = []

    def _on_pool_step(step):
        pool_steps_ref.append(step)
        persister.persist(rows + [{"h": round(float(h), 2), "status": "in_progress",
                                   "pool_steps": pool_steps_ref}])

    repeat_block = ("nn", "nnn") if args.repeat_entangle_only else ("nn", "nnn", "x")

    # Palanca B: seed step 0 from the analytic/donor warm-start cascade instead of
    # the raw backbone, so the (cold, most expensive) first optimization starts
    # near the optimum. Built on the base_blocks + nn-only backbone circuit.
    warm0 = None
    if args.warm_seed:
        from qmbp_simulation.circuits.bond_mask import BondSelection
        from qmbp_simulation.circuits.hva_variants import build_variant, make_masked_variant
        seed_sel = BondSelection(nn_edges=nn_edges, nnn_edges=[])
        v0 = make_masked_variant("seed0", "warm seed backbone", ["nn", "nnn", "x"],
                                 seed_sel)
        qc0, _ = build_variant(builder, args.n, lat, v0)
        ws = prepare_warmstart(qc0, H, psi, n_nn=n_nn, n_nnn=0, n_qubits=args.n,
                               p_layers=1, h=h, J2=args.j2, backend=backend,
                               target_len=qc0.num_parameters)
        warm0 = np.asarray(ws["seed"], float)

    t0 = time.time()
    steps, reached = grow_adapt_pool(
        builder, args.n, lat, H, psi, e0, gap, nn_all=nn_edges, nnn_all=nnn_edges,
        n_nn=n_nn, n_nnn=n_nnn, restarts=args.restarts, maxiter=args.maxiter,
        seed0=args.seed0, target_fid=args.target_fid, grow_step=args.grow_step,
        repeat_block=repeat_block, allow_repeat_layer=not args.no_repeat_layer,
        max_layers=args.max_layers, efficiency_patience=args.efficiency_patience,
        min_dfid_keep_growing=args.min_dfid,
        layer_growth_penalty=args.layer_growth_penalty, fast_rank=args.fast_rank,
        analytic_seed_new=args.analytic_seed_new,
        select_by_fidelity=args.select_by_fidelity,
        block_precondition=args.block_precondition,
        warm_theta0=warm0, backend=backend, on_step=_on_pool_step)

    best = max(steps, key=lambda s: s["best_fidelity"])
    pool_fid = best["best_fidelity"]
    pool_2q = best["n_2q_transpiled"]
    pool_fpc = best["fidelity_per_cx"]

    # p=2 full expressivity anchor.
    p2_fid, p2_2q, _p2_npar = _converge_full_pN(
        builder, args.n, lat, H, psi, n_nn, n_nnn, 2, h, args.j2,
        restarts=args.restarts, maxiter=args.maxiter, seed0=args.seed0, backend=backend)
    p2_fpc = (p2_fid / p2_2q) if p2_2q > 0 else None

    beats_p2_fid = pool_fid >= p2_fid - args.fid_tol
    better_fid_per_cx = (pool_fpc is not None and p2_fpc is not None
                         and pool_fpc > p2_fpc)

    row = {
        "h": round(float(h), 2), "n_qubits": args.n, "topology": args.topology,
        "e0": e0, "gap": gap, "n_nn": n_nn, "n_nnn": n_nnn,
        "pool_best_fidelity": pool_fid, "pool_2q": pool_2q, "pool_fid_per_cx": pool_fpc,
        "pool_n_layers": best["n_layers"], "pool_n_bonds": best["n_bonds"],
        "pool_blocks": best["blocks"], "pool_reached_target": reached,
        "pool_stop_reason": steps[-1].get("stop_reason"),
        "pool_n_steps": len(steps),
        "pool_action_sequence": [s["action_kind"] for s in steps],
        "p2_full_fidelity": p2_fid, "p2_full_2q": p2_2q, "p2_full_fid_per_cx": p2_fpc,
        "pool_matches_p2_fid": beats_p2_fid,
        "pool_better_fid_per_cx_than_p2": better_fid_per_cx,
        "best_theta_final": best["best_theta_final"],
        "seconds": round(time.time() - t0, 1),
    }
    verdict = ("Pool≥p2 fid" if beats_p2_fid else "Pool<p2 fid")
    fpc_tag = "fid/cx>p2" if better_fid_per_cx else "fid/cx≤p2"
    print(f"  h={h:.2f} gap={gap:.4f} | Pool fid={pool_fid:.4f} 2q={pool_2q} "
          f"layers={best['n_layers']} fpc={pool_fpc:.2e} | p2 fid={p2_fid:.4f} "
          f"2q={p2_2q} | {verdict} {fpc_tag} ({row['seconds']}s)", flush=True)
    return row


def _save_theta_npz(rows, args) -> Path | None:
    from qmbp_simulation.framework.result_io import upsert_theta_npz

    pts = [r for r in rows if r.get("best_theta_final") and "pool_best_fidelity" in r]
    if not pts or len({len(r["best_theta_final"]) for r in pts}) != 1:
        return None  # variable θ length across h (different grown structures)
    h_arr = np.array([r["h"] for r in pts], float)
    theta_arr = np.array([r["best_theta_final"] for r in pts], float)
    e_vqe = np.array([r["e0"] + abs(r["e0"]) * (1.0 - r["pool_best_fidelity"]) for r in pts], float)
    e_exact = np.array([r["e0"] for r in pts], float)
    gaps = np.array([r["gap"] for r in pts], float)
    npz_dir = STUDY_ROOT / "poolA_theta"
    npz_dir.mkdir(parents=True, exist_ok=True)
    npz = npz_dir / f"poolA_best_theta_{args.topology}_N{args.n}.npz"
    upsert_theta_npz(npz, h_arr, theta_arr, e_vqe, e_exact, gaps,
                     model=MODEL, j2=args.j2)
    return npz


def run(args) -> int:
    from qmbp_simulation.execution import NoiselessBackend

    builder = HVACircuitBuilder()
    gt_cache = GroundTruthCache()
    out_file = f"adapt_poolA_{args.topology}_N{args.n}.json"

    print(f"[adapt_poolA] N={args.n} topology={args.topology} J2={args.j2} "
          f"h_list={args.h_list} grow_step={args.grow_step} max_layers={args.max_layers} "
          f"repeat_layer={not args.no_repeat_layer} restarts={args.restarts} "
          f"maxiter={args.maxiter} target_fid={args.target_fid}", flush=True)

    rows: list[dict] = []
    persister = StudyPersister(
        subdir=SUBDIR, out_file=out_file,
        fingerprint={"n_qubits": args.n, "p_layers": None, "model": MODEL,
                     "topology": args.topology, "h": None},
        extra={"topology": args.topology, "N": args.n, "J2": args.j2,
               "grow_step": args.grow_step, "max_layers": args.max_layers,
               "allow_repeat_layer": not args.no_repeat_layer,
               "target_fid": args.target_fid, "schema": "adapt_poolA_v1"},
        params={"experiment": "adapt_poolA", "N": args.n},
        description="ADAPT Option A — unrestricted-p growth by Δfid/Δ2q "
                    "(add-bond vs repeat-layer), vs p=2 full expressivity anchor",
        rows_ref=rows)

    for h in args.h_list:
        backend = NoiselessBackend()  # cached dense H is h-specific
        row = _point(args, builder, backend, gt_cache, persister, rows, h)
        rows.append(row)
        persister.persist(rows)

    gt_cache.flush()
    npz = _save_theta_npz(rows, args)
    persister.persist(rows, status="final")

    n_match = sum(1 for r in rows if r["pool_matches_p2_fid"])
    n_better_fpc = sum(1 for r in rows if r["pool_better_fid_per_cx_than_p2"])
    print("\n=== POOL-A VERDICT (vs p=2 full anchor) ===", flush=True)
    print(f"  Pool reaches p2 fidelity: {n_match}/{len(rows)} | "
          f"Pool better fid-per-CX than p2: {n_better_fpc}/{len(rows)}", flush=True)
    if rows:
        mean_pool = float(np.mean([r["pool_best_fidelity"] for r in rows]))
        mean_p2 = float(np.mean([r["p2_full_fidelity"] for r in rows]))
        print(f"  mean fidelity — Pool-A: {mean_pool:.4f}  p2-full: {mean_p2:.4f}", flush=True)
    verdict = ("UNRESTRICTED-p HELPS: Pool-A reaches p2 expressivity, "
               "cheaper per-CX where it wins") if n_match >= max(1, len(rows) // 2) \
        else "Pool-A still below p2 — ceiling is deeper than layer count"
    print(f"  → {verdict}", flush=True)
    if npz is not None:
        print(f"  Pool-A θ saved: {npz.relative_to(STUDY_ROOT.parent.parent)}", flush=True)
    if not args.no_sync:
        sync_scoreboard()
    print("DONE", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ADAPT Option A — unrestricted-p growth")
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--h-list", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    p.add_argument("--grow-step", type=int, default=2)
    p.add_argument("--max-layers", type=int, default=6)
    p.add_argument("--no-repeat-layer", action="store_true",
                   help="Disable the repeat-layer action (bonds-only ADAPT).")
    p.add_argument("--repeat-entangle-only", action="store_true",
                   help="Repeat only the entangling blocks [nn,nnn] (no x) — "
                        "cheaper in params per repeated layer.")
    p.add_argument("--min-dfid", type=float, default=0.01,
                   help="Keep growing while a step's absolute Δfidelity ≥ this, "
                        "even if fid-per-CX falls (0 = efficiency-only stop).")
    p.add_argument("--fast-rank", action="store_true",
                   help="NOT recommended for this pool: the gradient proxy mis-ranks "
                        "add_bonds vs repeat_layer and collapses fidelity (measured). "
                        "Use --warm-seed + --layer-growth-penalty instead. The lever "
                        "belongs to the operator-list loop (Option B).")
    p.add_argument("--layer-growth-penalty", type=float, default=0.0,
                   help="Raise the Δfid needed to keep growing as layers accumulate "
                        "(stops over-growth at the critical point). 0 = flat.")
    p.add_argument("--warm-seed", action="store_true",
                   help="Seed step 0 with the analytic/donor warm-start cascade "
                        "instead of the raw backbone (faster first convergence).")
    p.add_argument("--analytic-seed-new", action="store_true",
                   help="Seed each newly added bond/layer by a 1D Newton step "
                        "instead of 0 (measured: faster AND higher fidelity — "
                        "the recommended guidance lever).")
    p.add_argument("--select-by-fidelity", action="store_true",
                   help="Rank candidate bonds by |∂fidelity/∂θ| instead of energy "
                        "(more fidelity, somewhat slower). Opt-in.")
    p.add_argument("--block-precondition", action="store_true",
                   help="Per-block rescaling θ=D·φ before optimizing. NOT "
                        "recommended by default (measured: slower, worse fid/CX "
                        "at N10 h=0.7). Kept for experimentation.")
    p.add_argument("--restarts", type=int, default=3,
                   help="Must be >=2 so repeat-layer's new layer can activate.")
    p.add_argument("--maxiter", type=int, default=1500)
    p.add_argument("--seed0", type=int, default=70000)
    p.add_argument("--target-fid", type=float, default=0.95)
    p.add_argument("--efficiency-patience", type=int, default=2)
    p.add_argument("--fid-tol", type=float, default=0.005,
                   help="Pool counts as matching p2 if within this of p2 fidelity.")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
