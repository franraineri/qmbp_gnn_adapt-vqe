#!/usr/bin/env python
"""ADAPT Gate 0 — is the bond-growth GRADIENT ranking informative near h_c?

The cheap, refutable experiment that decides whether ADAPT-style bond growth is
worth pursuing in the hard (small-gap) regime BEFORE spending any N18 compute.

The question: when ADAPT picks the next nnn bond by ``|∂E/∂θ|`` evaluated at the
nn-only optimized backbone, does that ranking agree with the ``|θ_nnn|`` ranking
of the FULLY converged ansatz (the information the top-k selector uses)? If the
two rankings agree, ADAPT's greedy local signal reproduces the global optimum's
structure and growth is well-founded. If they DISAGREE near h_c — where the gap
is tiny and ``|∂E/∂θ|`` is numerically small — the gradient ranking is noise and
ADAPT would grow bonds almost at random: the whole bond-growth axis is dead in
the hard regime and we keep top-k instead.

Per h-point (reusing only existing modules — no new physics/optimizer):
  1. ``study_core.ground_state`` → lattice, H, exact ψ, E0, gap.
  2. Converge the FULL frustrated p=1 ansatz via the mandatory warm-start
     cascade (``prepare_warmstart`` + ``converge_circuit``). Its θ → per-bond
     ``|θ_nnn|`` via ``bond_weights_from_theta`` (the top-k ranking signal).
  3. Converge the nn-ONLY backbone (BondSelection nn=all, nnn=[]). From its
     optimized θ, score every nnn candidate by ``|∂E/∂θ|`` via the shared
     ``study_core.bond_energy_gradients`` (ADAPT's step-0 growth signal).
  4. Compare the two nnn rankings with ``bond_mask.rank_correlation`` (Spearman)
     and ``rank_agreement_topk`` (top-k overlap at a few k).

Decision: the gate PASSES at an h-point when Spearman ρ ≥ --rho-pass AND the
top-k overlap ≥ --overlap-pass. The run reports pass/fail per h and an overall
verdict (fraction of hard-regime points passing).

Persistence: crash-safe partial+final via ``StudyPersister`` (one row per h,
written after every point). The converged FULL θ is also upserted into a θ-NPZ
(reused by later gates / warm-start donors) and e0/gap go through the shared
``GroundTruthCache``.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/run_adapt_gate0.py \
        --n 10 --h-list 0.5 0.8 1.0 1.2 1.5 --topk 5 10 --restarts 3 --maxiter 1500
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
    BondSelection,
    bond_weights_from_theta,
    rank_agreement_topk,
    rank_correlation,
)
from qmbp_simulation.circuits.hva_variants import (  # noqa: E402
    AnsatzVariant,
    build_variant,
    make_masked_variant,
)
from qmbp_simulation.framework.study_core import (  # noqa: E402
    bond_energy_gradients,
    converge_circuit,
    cx_and_params,
    ground_state,
    prepare_warmstart,
)
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402
from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache  # noqa: E402

SUBDIR = "bond_ablation"
MODEL = "tfim_frustrated"


def _converge_full(builder, n, lat, H, psi, e0, gap, n_nn, n_nnn, h, j2, *,
                   restarts, maxiter, seed0, backend):
    """Converge the full frustrated p=1 ansatz; return (fid, best_theta)."""
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


def _converge_backbone(builder, n, lat, H, psi, n_nn, h, j2, nn_edges, *,
                       restarts, maxiter, seed0, backend):
    """Converge the nn-ONLY backbone (no nnn); return its optimized θ.

    This is the state from which ADAPT evaluates the step-0 nnn gradients, so the
    gradient ranking we test is exactly the one the real algorithm would use.
    """
    sel = BondSelection(nn_edges=list(nn_edges), nnn_edges=[], provenance="nn_only")
    v = make_masked_variant("nn_backbone", "nn-only backbone (ADAPT seed)",
                            ["nn", "nnn", "x"], sel)
    qc, _ = build_variant(builder, n, lat, v)
    ws = prepare_warmstart(qc, H, psi, n_nn=n_nn, n_nnn=0, n_qubits=n, p_layers=1,
                           h=h, J2=j2, backend=backend, target_len=qc.num_parameters)
    fid, _e, runs = converge_circuit(qc, H, psi, restarts=restarts, maxiter=maxiter,
                                     seed0=seed0, warm_theta=ws["seed"], backend=backend)
    theta = np.asarray(min(runs, key=lambda r: r["energy"])["theta_final"], float)
    return float(fid), theta, sel


def _point(args, builder, backend, gt_cache, h) -> dict:
    """Run the Gate-0 comparison at one h-point → a result row."""
    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(
        args.topology, args.n, h, args.j2, 1)
    nn_edges = list(lat.edges)
    nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)
    gt_cache.put(args.topology, args.n, MODEL, h, energy=e0, gap=gap,
                 method="eigsh_exact")

    t0 = time.time()
    # 1) FULL reference → |θ_nnn| ranking (the top-k signal).
    full_fid, theta_full, full_prov = _converge_full(
        builder, args.n, lat, H, psi, e0, gap, n_nn, n_nnn, h, args.j2,
        restarts=args.restarts, maxiter=args.maxiter, seed0=args.seed0, backend=backend)
    _w_nn, w_nnn_full = bond_weights_from_theta(theta_full, n_nn, n_nnn, args.n, 1)

    # 2) nn-only backbone → |∂E/∂θ_nnn| ranking (ADAPT's step-0 growth signal).
    bb_fid, theta_bb, bb_sel = _converge_backbone(
        builder, args.n, lat, H, psi, n_nn, h, args.j2, nn_edges,
        restarts=args.restarts, maxiter=args.maxiter, seed0=args.seed0, backend=backend)
    _g_nn, grad_nnn = bond_energy_gradients(
        builder, args.n, lat, bb_sel, [], nnn_edges, H, psi, theta_bb)

    # 3) Compare the two nnn rankings.
    rho = rank_correlation(grad_nnn, w_nnn_full)
    overlaps = {}
    for k in args.topk:
        if k <= n_nnn:
            overlaps[k] = rank_agreement_topk(grad_nnn, w_nnn_full, k)
    # Primary overlap used for the pass gate: the smallest requested k that fits.
    k_primary = min((k for k in args.topk if k <= n_nnn), default=n_nnn)
    overlap_primary = overlaps.get(k_primary)

    grad_informative = bool(np.max(grad_nnn) > args.grad_floor) if grad_nnn.size else False
    rho_ok = rho is not None and rho >= args.rho_pass
    ov_ok = overlap_primary is not None and overlap_primary >= args.overlap_pass
    passed = bool(grad_informative and rho_ok and ov_ok)

    row = {
        "h": round(float(h), 2),
        "n_qubits": args.n,
        "topology": args.topology,
        "e0": e0,
        "gap": gap,
        "n_nn": n_nn,
        "n_nnn": n_nnn,
        "full_fidelity": full_fid,
        "full_seed_kind": full_prov,
        "backbone_fidelity": bb_fid,
        "spearman_grad_vs_theta": rho,
        "topk_overlap": {str(k): v for k, v in overlaps.items()},
        "topk_overlap_primary_k": k_primary,
        "topk_overlap_primary": overlap_primary,
        "grad_nnn_max": float(np.max(grad_nnn)) if grad_nnn.size else 0.0,
        "grad_nnn_min": float(np.min(grad_nnn)) if grad_nnn.size else 0.0,
        "grad_informative": grad_informative,
        "pass": passed,
        "best_theta_final": theta_full.tolist(),  # FULL θ (for NPZ / donors)
        "seconds": round(time.time() - t0, 1),
    }
    rho_s = f"{rho:+.3f}" if rho is not None else "n/a"
    ov_s = f"{overlap_primary:.2f}" if overlap_primary is not None else "n/a"
    print(f"  h={h:.2f} gap={gap:.4f} full_fid={full_fid:.4f} bb_fid={bb_fid:.4f} "
          f"ρ(grad,|θ|)={rho_s} overlap@{k_primary}={ov_s} "
          f"grad_max={row['grad_nnn_max']:.2e} → {'PASS' if passed else 'fail'} "
          f"({row['seconds']}s)", flush=True)
    return row


def _save_theta_npz(rows, args) -> Path | None:
    """Upsert the converged FULL θ per h into a reusable θ-NPZ (anti-regression)."""
    from qmbp_simulation.framework.result_io import upsert_theta_npz

    pts = [r for r in rows if r.get("best_theta_final")]
    if not pts:
        return None
    # Only points with a uniform θ length can share one NPZ (same N → same length).
    lengths = {len(r["best_theta_final"]) for r in pts}
    if len(lengths) != 1:
        return None
    h_arr = np.array([r["h"] for r in pts], float)
    theta_arr = np.array([r["best_theta_final"] for r in pts], float)
    e_vqe = np.array([r["e0"] + abs(r["e0"]) * (1.0 - r["full_fidelity"]) for r in pts], float)
    e_exact = np.array([r["e0"] for r in pts], float)
    gaps = np.array([r["gap"] for r in pts], float)
    npz_dir = STUDY_ROOT / "gate0_theta"
    npz_dir.mkdir(parents=True, exist_ok=True)
    npz = npz_dir / f"gate0_full_theta_{args.topology}_N{args.n}.npz"
    upsert_theta_npz(npz, h_arr, theta_arr, e_vqe, e_exact, gaps,
                     model=MODEL, j2=args.j2)
    return npz


def run(args) -> int:
    from qmbp_simulation.execution import NoiselessBackend

    builder = HVACircuitBuilder()
    gt_cache = GroundTruthCache()
    out_file = f"adapt_gate0_{args.topology}_N{args.n}.json"

    print(f"[adapt_gate0] N={args.n} topology={args.topology} J2={args.j2} "
          f"h_list={args.h_list} topk={args.topk} restarts={args.restarts} "
          f"maxiter={args.maxiter} | pass if ρ≥{args.rho_pass} & overlap≥{args.overlap_pass}",
          flush=True)

    rows: list[dict] = []
    persister = StudyPersister(
        subdir=SUBDIR, out_file=out_file,
        fingerprint={"n_qubits": args.n, "p_layers": 1, "model": MODEL,
                     "topology": args.topology, "h": None},
        extra={"topology": args.topology, "N": args.n, "p_layers": 1,
               "J2": args.j2, "topk": args.topk, "rho_pass": args.rho_pass,
               "overlap_pass": args.overlap_pass, "grad_floor": args.grad_floor,
               "schema": "adapt_gate0_v1"},
        params={"experiment": "adapt_gate0", "N": args.n},
        description="ADAPT Gate 0 — gradient ranking vs |θ|-of-full ranking "
                    "agreement for nnn bonds, per h near h_c",
        rows_ref=rows)

    for h in args.h_list:
        # Fresh backend per h: the cached dense H is h-specific.
        backend = NoiselessBackend()
        row = _point(args, builder, backend, gt_cache, h)
        rows.append(row)
        persister.persist(rows)  # crash-safe: after every h-point

    gt_cache.flush()
    npz = _save_theta_npz(rows, args)
    persister.persist(rows, status="final")

    # Overall verdict.
    tested = [r for r in rows if r["spearman_grad_vs_theta"] is not None]
    n_pass = sum(1 for r in rows if r["pass"])
    hard = [r for r in rows if r["h"] <= args.hard_h]
    hard_pass = sum(1 for r in hard if r["pass"])
    print("\n=== GATE 0 VERDICT ===", flush=True)
    print(f"  points passing: {n_pass}/{len(rows)} "
          f"(hard regime h≤{args.hard_h}: {hard_pass}/{len(hard)})", flush=True)
    if tested:
        mean_rho = float(np.mean([r["spearman_grad_vs_theta"] for r in tested]))
        print(f"  mean Spearman ρ(grad,|θ|) = {mean_rho:+.3f}", flush=True)
    verdict = "GRADIENT RANKING INFORMATIVE → ADAPT axis worth pursuing" \
        if (hard and hard_pass >= max(1, len(hard) // 2)) \
        else "GRADIENT RANKING NOISY in hard regime → prefer top-k, ADAPT axis weak"
    print(f"  → {verdict}", flush=True)
    if npz is not None:
        print(f"  FULL θ saved: {npz.relative_to(STUDY_ROOT.parent.parent)}", flush=True)
    if not args.no_sync:
        sync_scoreboard()
    print("DONE", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ADAPT Gate 0 — gradient-ranking validator")
    p.add_argument("--no-sync", action="store_true",
                   help="Skip the post-run scoreboard refresh.")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--h-list", type=float, nargs="+",
                   default=[0.5, 0.8, 1.0, 1.2, 1.5],
                   help="h values to test (include points near h_c).")
    p.add_argument("--topk", type=int, nargs="+", default=[5, 10],
                   help="top-k overlap sizes to report (first that fits gates).")
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--maxiter", type=int, default=1500)
    p.add_argument("--seed0", type=int, default=70000)
    p.add_argument("--rho-pass", type=float, default=0.5,
                   help="Min Spearman ρ for a point to pass.")
    p.add_argument("--overlap-pass", type=float, default=0.5,
                   help="Min top-k overlap for a point to pass.")
    p.add_argument("--grad-floor", type=float, default=1e-5,
                   help="Max |∂E/∂θ_nnn| below this → gradient uninformative. "
                        "Same 'gradient too small = noise' notion as grow_bonds_adapt's "
                        "grad_tol (set higher here: a validator threshold, not a growth cut).")
    p.add_argument("--hard-h", type=float, default=1.0,
                   help="h at/below which a point counts as the hard (near-h_c) regime.")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
