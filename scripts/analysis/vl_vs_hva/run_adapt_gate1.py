#!/usr/bin/env python
"""ADAPT Gate 1 — ADAPT bond-growth vs top-k at EQUAL 2q budget (N10/N14).

Gate 0 asks whether the growth gradient is informative; Gate 1 asks the decisive
question: does ADAPT, growing nnn bonds greedily by gradient from an empty nnn
set, actually BEAT the top-k selector (strongest nnn by |θ| of the converged
full ansatz) when both are given the SAME number of nnn bonds — i.e. the same 2q
cost? If ADAPT does not match or beat top-k on fidelity / fidelity-per-CX at N10
or N14, it will not do so magically at N18, and the bond-growth axis is not worth
the N18 compute.

Fair comparison per h-point (reusing only existing modules):
  1. ``study_core.ground_state`` → lattice, H, exact ψ, E0, gap.
  2. ADAPT: ``study_core.grow_bonds_adapt`` from the nn-only backbone, with
     per-step warm-start and the efficiency (fid/CX) stop. Its final step fixes
     the budget ``k = n_nnn`` bonds it chose and the fidelity it reached.
  3. top-k AT THE SAME k: converge the FULL reference (mandatory warm-start),
     read per-bond ``|θ_nnn|`` via ``bond_weights_from_theta``, keep the top ``k``
     nnn via ``bond_mask.top_k_by_weight``, build the masked variant, converge it.
  4. Compare fidelity and fidelity-per-CX of ADAPT vs top-k at that shared k.

The gate PASSES at an h-point when ADAPT's fidelity is within --fid-tol of top-k
(ADAPT competitive) OR strictly higher. The overall verdict is the win/tie rate
across the hard regime.

Persistence: crash-safe partial+final via ``StudyPersister`` (one row per h). The
converged FULL θ and both selections' θ go into a reusable θ-NPZ; e0/gap through
the shared ``GroundTruthCache``.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/run_adapt_gate1.py \
        --n 10 --h-list 0.5 0.8 1.0 1.2 --grow-step 2 --restarts 3 --maxiter 1500
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


def _converge_full(builder, n, lat, H, psi, n_nn, n_nnn, h, j2, *,
                   restarts, maxiter, seed0, backend):
    """Converge the full frustrated p=1 ansatz; return (fid, theta, provenance)."""
    ref = AnsatzVariant(name="p1_full_ref", description="p=1 full frustrated (all bonds)",
                        blocks=["nn", "nnn", "x"], tags=("reference",))
    qc, _ = build_variant(builder, n, lat, ref)
    ws = prepare_warmstart(qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=n,
                           p_layers=1, h=h, J2=j2, backend=backend,
                           target_len=qc.num_parameters)
    fid, _e, runs = converge_circuit(qc, H, psi, restarts=restarts, maxiter=maxiter,
                                     seed0=seed0, warm_theta=ws["seed"], backend=backend)
    theta = np.asarray(min(runs, key=lambda r: r["energy"])["theta_final"], float)
    return float(fid), theta, ws["provenance"]


def _converge_topk(builder, n, lat, H, psi, nn_edges, nnn_edges, theta_full, *,
                   n_nn, n_nnn, k_nnn, h, j2, restarts, maxiter, seed0, backend):
    """Build + converge the top-k masked variant (strongest k nnn by |θ_full|)."""
    w_nn, w_nnn = bond_weights_from_theta(theta_full, n_nn, n_nnn, n, 1)
    sel = top_k_by_weight(nn_edges, nnn_edges, w_nn, w_nnn,
                          k_nn=None, k_nnn=k_nnn)
    v = make_masked_variant(f"topk_nnn{k_nnn}", f"top-{k_nnn} nnn by |θ|",
                            ["nn", "nnn", "x"], sel, tags=("T2", "subset"))
    qc, _ = build_variant(builder, n, lat, v)
    n_2q, npar = cx_and_params(qc)
    ws = prepare_warmstart(qc, H, psi, n_nn=n_nn, n_nnn=len(sel.nnn_edges),
                           n_qubits=n, p_layers=1, h=h, J2=j2,
                           target_nnn_edges=sel.nnn_edges, backend=backend,
                           target_len=npar)
    fid, e_best, runs = converge_circuit(qc, H, psi, restarts=restarts,
                                         maxiter=maxiter, seed0=seed0,
                                         warm_theta=ws["seed"], backend=backend)
    theta = min(runs, key=lambda r: r["energy"])["theta_final"]
    return float(fid), n_2q, npar, sel, theta


def _point(args, builder, backend, gt_cache, h) -> dict:
    """Run the Gate-1 ADAPT-vs-top-k comparison at one h-point → a result row."""
    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(
        args.topology, args.n, h, args.j2, 1)
    nn_edges = list(lat.edges)
    nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)
    gt_cache.put(args.topology, args.n, MODEL, h, energy=e0, gap=gap,
                 method="eigsh_exact")

    t0 = time.time()
    # 1) ADAPT growth from the nn-only backbone (warm per step + efficiency stop).
    adapt_steps, adapt_reached = grow_bonds_adapt(
        builder, args.n, lat, H, psi, e0, gap, nn_all=nn_edges, nnn_all=nnn_edges,
        n_nn=n_nn, n_nnn=n_nnn, restarts=args.restarts, maxiter=args.maxiter,
        seed0=args.seed0, target_fid=args.target_fid, grow_step=args.grow_step,
        grad_tol=args.grad_tol, warm_per_step=True,
        efficiency_stop=args.efficiency_stop, efficiency_patience=args.efficiency_patience,
        backend=backend)
    adapt_last = adapt_steps[-1]
    k = adapt_last["n_nnn_bonds"]  # the nnn budget ADAPT chose
    adapt_fid = adapt_last["best_fidelity"]
    adapt_2q = adapt_last["n_2q_transpiled"]
    adapt_fpc = adapt_last["fidelity_per_cx"]

    # 2) FULL reference → |θ| ranking for top-k.
    full_fid, theta_full, full_prov = _converge_full(
        builder, args.n, lat, H, psi, n_nn, n_nnn, h, args.j2,
        restarts=args.restarts, maxiter=args.maxiter, seed0=args.seed0, backend=backend)

    # 3) top-k at the SAME nnn budget k.
    if k > 0:
        topk_fid, topk_2q, _npar, topk_sel, _topk_theta = _converge_topk(
            builder, args.n, lat, H, psi, nn_edges, nnn_edges, theta_full,
            n_nn=n_nn, n_nnn=n_nnn, k_nnn=k, h=h, j2=args.j2,
            restarts=args.restarts, maxiter=args.maxiter, seed0=args.seed0, backend=backend)
        topk_fpc = (topk_fid / topk_2q) if topk_2q > 0 else None
    else:
        # ADAPT added no nnn (backbone already stopped): top-k with 0 nnn == backbone.
        topk_fid, topk_2q, topk_fpc = adapt_fid, adapt_2q, adapt_fpc

    fid_delta = adapt_fid - topk_fid
    adapt_wins = fid_delta > args.fid_tol
    tie = abs(fid_delta) <= args.fid_tol
    competitive = adapt_wins or tie  # ADAPT matches or beats top-k
    passed = bool(competitive)

    row = {
        "h": round(float(h), 2),
        "n_qubits": args.n,
        "topology": args.topology,
        "e0": e0, "gap": gap, "n_nn": n_nn, "n_nnn": n_nnn,
        "k_nnn_budget": k,
        "adapt_fidelity": adapt_fid,
        "adapt_2q": adapt_2q,
        "adapt_fid_per_cx": adapt_fpc,
        "adapt_reached_target": adapt_reached,
        "adapt_stop_reason": adapt_last.get("stop_reason"),
        "adapt_n_steps": len(adapt_steps),
        "topk_fidelity": topk_fid,
        "topk_2q": topk_2q,
        "topk_fid_per_cx": topk_fpc,
        "full_fidelity": full_fid,
        "full_seed_kind": full_prov,
        "fid_delta_adapt_minus_topk": fid_delta,
        "adapt_wins": adapt_wins,
        "tie": tie,
        "pass": passed,
        "best_theta_final": theta_full.tolist(),  # FULL θ for NPZ reuse
        "seconds": round(time.time() - t0, 1),
    }
    verdict = "ADAPT>topk" if adapt_wins else ("tie" if tie else "topk>ADAPT")
    print(f"  h={h:.2f} gap={gap:.4f} k={k} | ADAPT fid={adapt_fid:.4f} 2q={adapt_2q} "
          f"fpc={adapt_fpc:.2e} | topk fid={topk_fid:.4f} 2q={topk_2q} "
          f"| Δfid={fid_delta:+.4f} → {verdict} ({row['seconds']}s)", flush=True)
    return row


def _save_theta_npz(rows, args) -> Path | None:
    """Upsert the converged FULL θ per h into a reusable θ-NPZ (anti-regression)."""
    from qmbp_simulation.framework.result_io import upsert_theta_npz

    pts = [r for r in rows if r.get("best_theta_final")]
    if not pts:
        return None
    if len({len(r["best_theta_final"]) for r in pts}) != 1:
        return None
    h_arr = np.array([r["h"] for r in pts], float)
    theta_arr = np.array([r["best_theta_final"] for r in pts], float)
    e_vqe = np.array([r["e0"] + abs(r["e0"]) * (1.0 - r["full_fidelity"]) for r in pts], float)
    e_exact = np.array([r["e0"] for r in pts], float)
    gaps = np.array([r["gap"] for r in pts], float)
    npz_dir = STUDY_ROOT / "gate1_theta"
    npz_dir.mkdir(parents=True, exist_ok=True)
    npz = npz_dir / f"gate1_full_theta_{args.topology}_N{args.n}.npz"
    upsert_theta_npz(npz, h_arr, theta_arr, e_vqe, e_exact, gaps,
                     model=MODEL, j2=args.j2)
    return npz


def run(args) -> int:
    from qmbp_simulation.execution import NoiselessBackend

    builder = HVACircuitBuilder()
    gt_cache = GroundTruthCache()
    out_file = f"adapt_gate1_{args.topology}_N{args.n}.json"

    print(f"[adapt_gate1] N={args.n} topology={args.topology} J2={args.j2} "
          f"h_list={args.h_list} grow_step={args.grow_step} restarts={args.restarts} "
          f"maxiter={args.maxiter} fid_tol={args.fid_tol}", flush=True)

    rows: list[dict] = []
    persister = StudyPersister(
        subdir=SUBDIR, out_file=out_file,
        fingerprint={"n_qubits": args.n, "p_layers": 1, "model": MODEL,
                     "topology": args.topology, "h": None},
        extra={"topology": args.topology, "N": args.n, "p_layers": 1,
               "J2": args.j2, "grow_step": args.grow_step, "fid_tol": args.fid_tol,
               "efficiency_stop": args.efficiency_stop, "schema": "adapt_gate1_v1"},
        params={"experiment": "adapt_gate1", "N": args.n},
        description="ADAPT Gate 1 — bond-growth vs top-k at equal 2q budget "
                    "(fidelity / fid-per-CX on the Pareto frontier)",
        rows_ref=rows)

    for h in args.h_list:
        backend = NoiselessBackend()  # cached dense H is h-specific
        row = _point(args, builder, backend, gt_cache, h)
        rows.append(row)
        persister.persist(rows)  # crash-safe after every h

    gt_cache.flush()
    npz = _save_theta_npz(rows, args)
    persister.persist(rows, status="final")

    # Overall verdict.
    n_pass = sum(1 for r in rows if r["pass"])
    n_win = sum(1 for r in rows if r["adapt_wins"])
    hard = [r for r in rows if r["h"] <= args.hard_h]
    hard_pass = sum(1 for r in hard if r["pass"])
    print("\n=== GATE 1 VERDICT ===", flush=True)
    print(f"  ADAPT competitive (win or tie): {n_pass}/{len(rows)} "
          f"(strict wins: {n_win}) | hard regime h≤{args.hard_h}: {hard_pass}/{len(hard)}",
          flush=True)
    if rows:
        mean_delta = float(np.mean([r["fid_delta_adapt_minus_topk"] for r in rows]))
        print(f"  mean Δfid (ADAPT − top-k) = {mean_delta:+.4f}", flush=True)
    verdict = "ADAPT MATCHES/BEATS top-k → bond-growth axis viable, proceed to Gate 2" \
        if (hard and hard_pass >= max(1, len(hard) // 2)) \
        else "ADAPT LOSES to top-k → keep top-k, ADAPT axis not worth N18 compute"
    print(f"  → {verdict}", flush=True)
    if npz is not None:
        print(f"  FULL θ saved: {npz.relative_to(STUDY_ROOT.parent.parent)}", flush=True)
    if not args.no_sync:
        sync_scoreboard()
    print("DONE", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ADAPT Gate 1 — ADAPT vs top-k at equal 2q")
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--h-list", type=float, nargs="+", default=[0.5, 0.8, 1.0, 1.2])
    p.add_argument("--grow-step", type=int, default=2)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--maxiter", type=int, default=1500)
    p.add_argument("--seed0", type=int, default=70000)
    p.add_argument("--target-fid", type=float, default=0.999,
                   help="ADAPT fidelity target (kept high so the efficiency stop "
                        "fixes the budget, not an early fidelity crossing).")
    p.add_argument("--grad-tol", type=float, default=1e-6)
    p.add_argument("--efficiency-stop", action="store_true", default=True,
                   help="Stop ADAPT at the Pareto (fid/CX) point (default on).")
    p.add_argument("--no-efficiency-stop", dest="efficiency_stop",
                   action="store_false")
    p.add_argument("--efficiency-patience", type=int, default=2)
    p.add_argument("--fid-tol", type=float, default=0.005,
                   help="|Δfid| within this → ADAPT counts as tied with top-k.")
    p.add_argument("--hard-h", type=float, default=1.0)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
