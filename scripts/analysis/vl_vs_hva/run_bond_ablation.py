#!/usr/bin/env python
"""Bond-selection ablation: fewer 2q gates via masked bond-resolved HVA.

Studies whether entangling only a SUBSET of RZZ bonds keeps the fidelity of the
full frustrated layer at a lower 2-qubit-gate cost. Two selection techniques,
both derived from a converged p=1 reference θ:

- **T1 pruning** (:func:`bond_mask.prune_by_theta`): drop bonds whose optimized
  ``|θ_zz|`` is below a tolerance (near-identity RZZ).
- **T2 subset** (:func:`bond_mask.top_k_by_weight`): keep every nn bond plus the
  strongest fraction of nnn bonds by ``|θ_zz|``.

Flow (all reused from existing modules — no new physics/optimizer code):
  1. ``study_core.ground_state`` → lattice, H, exact ψ, E0, gap.
  2. Converge the p=1 reference (blocks ``[nn,nnn,x]``) via
     ``study_core.optimize_bestof`` (same robust best-of as run_ansatz_variants).
  3. Slice its θ into per-bond θ_nn / θ_nnn; derive T1/T2 selections with
     ``bond_mask``; wrap them as masked ``AnsatzVariant`` via ``make_masked_variant``.
  4. Optimize each masked variant (same optimizer), record fidelity, real 2q
     count (``study_core.cx_and_params``), and fidelity-per-CX.
  5. Persist to the DEDICATED ``bond_ablation`` study subfolder (crash-safe,
     traceable) so this analysis stays separate from the main variant sweep.

Compatible with a future ADAPT-VQE runner: it consumes the same
``bond_mask`` + ``build_variant`` + masked builder, growing a selection instead
of deriving it once.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/run_bond_ablation.py \
        --n 10 --h 0.5 --p 1 --restarts 2 --maxiter 1500 \
        --prune-tol 0.05 --nnn-keep-frac 0.5
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

from hva_vl_study_common import StudyPersister, save_ansatz_spec, sync_scoreboard  # noqa: E402

from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.bond_mask import (  # noqa: E402
    bond_weights_from_theta,
    prune_by_theta,
    top_k_by_weight,
)
from qmbp_simulation.circuits.hva_variants import (  # noqa: E402
    AnsatzVariant,
    build_variant,
    make_masked_variant,
)
from qmbp_simulation.framework.study_core import (  # noqa: E402
    build_variant_row,
    converge_circuit,
    cx_and_params,
    ground_state,
    prepare_warmstart,
)
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402

SUBDIR = "bond_ablation"


def _eval_variant(builder, n, lat, variant, H, psi, *, restarts, maxiter, seed0,
                  e0, gap, n_nn, n_nnn, p_ref, h, j2, backend=None,
                  on_restart=None, micro_descent=60, topology=None, save_spec=False):
    """Build + optimize one (masked or full) variant → a canonical result row.

    The warm-start is OBLIGATORY via the canonical ``prepare_warmstart`` cascade
    (calibrated + regime + micro-descent), not a θ=0 cold start. For a masked
    variant the ``bond_selection`` gives the real nnn edges / counts so the
    cascade's candidates match the circuit width. Reuses ``converge_circuit`` +
    ``build_variant_row``; a shared ``backend`` reuses the cached dense H.
    """
    qc, _ = build_variant(builder, n, lat, variant)
    n_2q, npar = cx_and_params(qc)
    sel = variant.bond_selection
    v_nn = len(sel.nn_edges) if sel is not None else n_nn
    v_nnn = len(sel.nnn_edges) if sel is not None else n_nnn
    tgt_nnn = sel.nnn_edges if sel is not None else None
    ws = prepare_warmstart(
        qc, H, psi, n_nn=v_nn, n_nnn=v_nnn, n_qubits=n, p_layers=p_ref, h=h,
        J2=j2, target_nnn_edges=tgt_nnn, micro_descent=micro_descent,
        backend=backend, target_len=npar)
    t0 = time.time()
    fid, e_best, runs = converge_circuit(
        qc, H, psi, restarts=restarts, maxiter=maxiter, seed0=seed0,
        warm_theta=ws["seed"], backend=backend, on_restart=on_restart)
    n_conv = sum(1 for r in runs if r["converged"])
    best_theta = min(runs, key=lambda r: r["energy"])["theta_final"]

    spec_path = None
    if save_spec and topology is not None:
        spec_path = str(save_ansatz_spec(
            variant, topology=topology, n_qubits=n, h=h, j2=j2, theta=best_theta,
            nn_edges=list(lat.edges),
            nnn_edges=HamiltonianBuilder._generate_nnn_edges(lat),
            fidelity=fid, n_2q=n_2q, n_params=npar, seed_kind=ws["provenance"]))

    row = build_variant_row(
        variant.name, fid, e_best, runs, n_2q=n_2q, n_params=npar, e0=e0, gap=gap,
        blocks=variant.blocks, rx_final=variant.rx_final,
        bond_selection=variant.bond_selection, seed_kind=ws["provenance"],
        seconds=time.time() - t0,
        description=variant.description, converged=f"{n_conv}/{restarts}")
    if spec_path is not None:
        row["ansatz_spec_path"] = spec_path
    return row


def run(args) -> int:
    from qmbp_simulation.execution import NoiselessBackend

    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(
        args.topology, args.n, args.h, args.j2, args.p)
    nn_edges = lat.edges
    nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)
    builder = HVACircuitBuilder()
    backend = NoiselessBackend()  # shared across variants (reuses cached dense H)
    _dtag = f"_from{args.derive_from}" if args.derive_from else ""
    out_file = f"bond_ablation_{args.topology}_N{args.n}_p{args.p}{_dtag}_h{args.h:.2f}.json"

    print(f"[bond_ablation] N={args.n} h={args.h} p={args.p} "
          f"e0={e0:.5f} gap={gap:.5f} n_nn={n_nn} n_nnn={n_nnn} "
          f"restarts={args.restarts} maxiter={args.maxiter}", flush=True)

    rows: list[dict] = []
    persister = StudyPersister(
        subdir=SUBDIR, out_file=out_file,
        fingerprint={"n_qubits": args.n, "p_layers": args.p,
                     "model": "tfim_frustrated", "topology": args.topology,
                     "h": args.h},
        extra={"topology": args.topology, "N": args.n, "h": args.h,
               "p_layers": args.p, "J2": args.j2, "e0": e0, "gap": gap,
               "prune_tol": args.prune_tol, "nnn_keep_frac": args.nnn_keep_frac,
               "derive_from": args.derive_from, "schema": "bond_ablation_v1"},
        params={"experiment": "bond_ablation", "N": args.n, "h": args.h,
                "p_layers": args.p},
        description="Bond-selection ablation (T1 prune / T2 subset) — "
                    "fewer 2q via masked bond-resolved HVA",
        rows_ref=rows, total_restarts=args.restarts)

    def _persist(status="partial"):
        persister.persist(rows, status=status)

    # ── Reference for bond ranking ──────────────────────────────────────────
    # Default: a p=args.p full reference. With --derive-from P (e.g. 2), converge
    # a DEEPER p=P full reference and rank bonds from its well-converged θ —
    # fixing the "p1 unsaturated at small gap → uninformative ranking" issue. The
    # masked variants are still built at args.p (the depth we want to cheapen).
    # The full reference (all edges) is built via the configurable engine, which
    # handles any p natively; no bond_selection needed.
    p_ref = args.derive_from if args.derive_from else args.p
    ref_variant = AnsatzVariant(
        name=f"p{p_ref}_full_ref",
        description=f"p={p_ref} full frustrated reference (all bonds)",
        blocks=["nn", "nnn", "x"] * p_ref,
        tags=("reference",),
    )
    ref_row = _eval_variant(builder, args.n, lat, ref_variant, H, psi,
                            restarts=args.restarts, maxiter=args.maxiter,
                            seed0=args.seed0, e0=e0, gap=gap, backend=backend,
                            n_nn=n_nn, n_nnn=n_nnn, p_ref=p_ref, h=args.h, j2=args.j2)
    rows.append(ref_row)
    _persist()
    print(f"  {ref_row['variant']:16s} fid={ref_row['best_fidelity']:.4f} "
          f"2q={ref_row['n_2q_transpiled']} conv={ref_row['converged']} "
          f"({ref_row['seconds']}s)", flush=True)

    # Reference θ → per-bond weights (aggregated across p_ref layers).
    theta_ref = np.asarray(ref_row["best_theta_final"], float)
    theta_nn, theta_nnn = bond_weights_from_theta(
        theta_ref, n_nn, n_nnn, args.n, p_ref)

    tgt_blocks = ["nn", "nnn", "x"] * args.p  # masked variants at the TARGET depth

    # ── T1: prune near-zero bonds by |θ| ─────────────────────────────────────
    sel_prune = prune_by_theta(nn_edges, nnn_edges, theta_nn, theta_nnn,
                               tol=args.prune_tol)
    if not sel_prune.is_empty():
        v_prune = make_masked_variant(
            f"p{args.p}_pruned", f"p={args.p} pruned (drop near-zero RZZ)",
            tgt_blocks, sel_prune, tags=("T1", "pruned"))
        row = _eval_variant(builder, args.n, lat, v_prune, H, psi,
                            restarts=args.restarts, maxiter=args.maxiter,
                            seed0=args.seed0, e0=e0, gap=gap, backend=backend,
                            n_nn=n_nn, n_nnn=n_nnn, p_ref=p_ref, h=args.h, j2=args.j2,
                            topology=args.topology, save_spec=True)
        rows.append(row)
        _persist()
        print(f"  {row['variant']:16s} fid={row['best_fidelity']:.4f} "
              f"2q={row['n_2q_transpiled']} (nn={row['n_nn_bonds']} "
              f"nnn={row['n_nnn_bonds']}) conv={row['converged']} "
              f"({row['seconds']}s)", flush=True)
    else:
        print(f"  p{args.p}_pruned SKIPPED — pruning removed all bonds", flush=True)

    # ── T2: keep all nn + strongest fraction of nnn by |θ| ──────────────────
    k_nnn = max(1, int(round(args.nnn_keep_frac * n_nnn)))
    sel_top = top_k_by_weight(nn_edges, nnn_edges, theta_nn, theta_nnn,
                              k_nn=None, k_nnn=k_nnn)
    v_top = make_masked_variant(
        f"p{args.p}_topk_nnn", f"p={args.p} with strongest {k_nnn}/{n_nnn} nnn bonds",
        tgt_blocks, sel_top, tags=("T2", "subset"))
    row = _eval_variant(builder, args.n, lat, v_top, H, psi,
                        restarts=args.restarts, maxiter=args.maxiter,
                        seed0=args.seed0, e0=e0, gap=gap, backend=backend,
                        n_nn=n_nn, n_nnn=n_nnn, p_ref=p_ref, h=args.h, j2=args.j2,
                        topology=args.topology, save_spec=True)
    rows.append(row)
    _persist(status="final")
    print(f"  {row['variant']:16s} fid={row['best_fidelity']:.4f} "
          f"2q={row['n_2q_transpiled']} (nn={row['n_nn_bonds']} "
          f"nnn={row['n_nnn_bonds']}) conv={row['converged']} "
          f"({row['seconds']}s)", flush=True)

    # ── Ranking (fidelity, then fidelity-per-CX) ─────────────────────────────
    print("\n=== RANKING (fidelity | 2q | fid/cx) ===", flush=True)
    for r in sorted(rows, key=lambda x: -x["best_fidelity"]):
        fpc = r["fidelity_per_cx"]
        print(f"  {r['variant']:16s} fid={r['best_fidelity']:.4f} "
              f"2q={r['n_2q_transpiled']:>4} fid/cx={fpc:.2e}", flush=True)
    if not getattr(args, "no_sync", False):
        sync_scoreboard()  # fire-and-forget refresh of the shared scoreboard
    print("DONE", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Bond-selection ablation (masked HVA)")
    p.add_argument("--no-sync", action="store_true",
                   help="Skip the post-run scoreboard refresh.")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--p", type=int, default=1)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--maxiter", type=int, default=1500)
    p.add_argument("--seed0", type=int, default=70000)
    p.add_argument("--prune-tol", type=float, default=0.05,
                   help="Drop RZZ bonds with |θ| below this (T1).")
    p.add_argument("--nnn-keep-frac", type=float, default=0.5,
                   help="Fraction of nnn bonds to keep by |θ| (T2).")
    p.add_argument("--derive-from", type=int, default=None, metavar="P_REF",
                   help="Rank bonds from a converged p=P_REF full reference "
                        "instead of the target p (fixes unsaturated-p1 ranking). "
                        "E.g. --p 1 --derive-from 2.")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
