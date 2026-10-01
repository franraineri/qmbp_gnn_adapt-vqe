#!/usr/bin/env python
"""Apply nnn bond top-k selection ON TOP of a winning ansatz STRUCTURE variant.

The N=18 h=0.5 scoreboard shows the best state is a p=2 STRUCTURE trick, not raw
depth: ``p2_half_nn_rx`` (half extra nn layer + free RX) reaches F=0.9268 @ 318
2q, beating ``p3_base`` (0.9132 @ 396 2q). Separately, the bond-selection study
found keeping only the strongest ~half of the nnn bonds (k=0.5) is Pareto-optimal
within p=2. This runner COMBINES both levers: it ranks the nnn bonds of a chosen
registry variant from its converged θ, then rebuilds the SAME block structure
(``blocks`` + ``rx_final``) with only the top-k nnn bonds entangled — the direct
path to "more faithful state at fewer 2q gates".

Flow (all reused — no new physics/optimizer):
  1. ``study_core.ground_state`` → lattice, H, exact ψ, E0, gap.
  2. Converge the FULL structure variant (e.g. p2_half_nn_rx) with its analytic
     warm-start (``warmstart.variant_warmstart_theta``), via ``optimize_bestof``.
  3. Rank nnn bonds by ``|θ|`` across the variant's real block offsets
     (``bond_mask.bond_weights_for_blocks`` — structure-aware, not uniform-layer).
  4. For each keep-fraction, build the masked variant (same blocks + rx_final,
     nnn restricted to the top-k) via ``make_masked_variant`` and converge it.
  5. Persist to the dedicated ``bond_ablation`` subfolder (crash-safe partials).

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/run_variant_topk.py \
        --n 18 --h 0.5 --variant p2_half_nn_rx \
        --keep-fracs 0.75 0.5 0.33 --restarts 3 --maxiter 2000
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

from hva_vl_study_common import (  # noqa: E402
    StudyPersister,
    save_ansatz_spec,
    sync_scoreboard,
)

from qmbp_simulation.analysis.warmstart import (  # noqa: E402
    transfer_theta_for_blocks,
    variant_warmstart_theta,
)
from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.bond_mask import (  # noqa: E402
    bond_weights_for_blocks,
    selection_from_variant_theta,
    top_k_by_weight,
)
from qmbp_simulation.circuits.hva_variants import (  # noqa: E402
    VARIANTS,
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


def _best_topk_seed(qc, H, psi, variant, sel, theta_ref, base, nnn_edges, args,
                    backend):
    """Best warm-start for a masked top-k variant via the canonical cascade.

    Uses ``study_core.prepare_warmstart`` (the general one-call combined warm-start
    from README_warmstart.md): it ranks calibrated + regime + donor candidates by
    a short micro-descent. For this NON-standard structure (half-layer + rx_final)
    the built-in standard-layout candidates are length-filtered out, so we inject
    the two structure-aware seeds as ``extra_candidates``:
      1. the converged FULL θ transferred onto this masked layout
         (:func:`transfer_theta_for_blocks`) — starts in the full's good basin;
      2. the analytic second-order seed (:func:`variant_warmstart_theta`).
    ``target_len`` is the real circuit width so the extra candidates survive the
    filter. Returns ``(seed, provenance)``.
    """
    npar = qc.num_parameters
    extra: list[tuple] = []
    transferred = transfer_theta_for_blocks(
        theta_ref, list(base.blocks), donor_nnn_edges=nnn_edges,
        target_nnn_edges=sel.nnn_edges, n_nn=len(sel.nn_edges), n_qubits=args.n,
        rx_final=base.rx_final, rz_final=base.rz_final)
    if transferred is not None and transferred.size == npar:
        extra.append((transferred, "transfer<full>"))
    analytic = variant_warmstart_theta(
        variant.blocks, len(sel.nn_edges), len(sel.nnn_edges), args.n, args.h,
        J=1.0, J2=args.j2, rx_final=variant.rx_final, rz_final=variant.rz_final)
    if len(analytic) == npar:
        extra.append((analytic, "analytic"))
    if not extra:
        return None, "zero"

    # p_layers=1 here only shapes the built-in standard-layout candidates, which
    # are length-filtered out for this structure anyway; the structure-aware
    # extras (gated by target_len=npar) are what actually compete.
    ws = prepare_warmstart(
        qc, H, psi, n_nn=len(sel.nn_edges), n_nnn=len(sel.nnn_edges),
        n_qubits=args.n, p_layers=1, h=args.h, J2=args.j2,
        target_nnn_edges=sel.nnn_edges, extra_candidates=extra,
        micro_descent=args.micro_descent, backend=backend, target_len=npar)
    return ws["seed"], ws["provenance"]


def _grid_coord_maps(n_qubits):
    """idx→(row,col) and (row,col)→idx for the square-grid qubit numbering.

    Mirrors ``generate_square``'s mapping (cols = ceil(sqrt(N))). The grid WIDTH
    changes with N (N10→4 cols, N18→5 cols), so a bond's raw qubit indices mean
    DIFFERENT physical positions at different N — which is why index-based donor
    transfer barely overlapped across N. These maps let us align by position.
    """
    import math

    cols = math.ceil(math.sqrt(n_qubits))
    idx2rc = {s: (s // cols, s % cols) for s in range(n_qubits)}
    rc2idx = {rc: s for s, rc in idx2rc.items()}
    return idx2rc, rc2idx


def _remap_edges_by_geometry(donor_edges, donor_n, target_n):
    """Translate donor edges to the target grid by matching (row,col) positions.

    Returns a list the SAME length/order as ``donor_edges`` (to stay aligned with
    the donor θ), each entry the target-index edge at the same two grid cells, or
    ``None`` when a cell has no counterpart in the target. This fixes the cross-N
    overlap: on the square lattice the geometric match covers ~74% of the N18
    edges from an N14 donor vs ~17% with raw indices.
    """
    d_idx2rc, _ = _grid_coord_maps(donor_n)
    _, t_rc2idx = _grid_coord_maps(target_n)
    out = []
    for a, b in donor_edges:
        rc_a, rc_b = d_idx2rc[int(a)], d_idx2rc[int(b)]
        if rc_a in t_rc2idx and rc_b in t_rc2idx:
            out.append((t_rc2idx[rc_a], t_rc2idx[rc_b]))
        else:
            # Sentinel out-of-range edge: keeps positional alignment with the
            # donor θ and is a valid int-tuple (won't break the per-edge match),
            # but can never equal a real target edge.
            out.append((-1, -1))
    return out


def _donor_coverage(donor_edges_geo, target_edges):
    """Fraction of TARGET edges covered by the geometry-remapped donor edges."""
    tgt = {tuple(sorted((int(a), int(b)))) for a, b in target_edges}
    hit = {tuple(sorted((int(e[0]), int(e[1])))) for e in donor_edges_geo
           if e is not None and int(e[0]) >= 0
           and tuple(sorted((int(e[0]), int(e[1])))) in tgt}
    return (len(hit) / len(tgt)) if tgt else 0.0


def _load_cross_n_full_donor(args, base, nn_edges, nnn_edges, npar_ref):
    """Find the best cross-N FULL donor and transfer it onto this N.

    Looks in the ``ansatz_specs`` library for a converged full of the SAME
    structure/topology/h at a SMALLER N with fidelity above ``--cross-n-min-fid``.
    Donor SELECTION maximizes the geometry-aware target COVERAGE (fraction of
    target edges the donor can actually fill after geometric remapping) with
    fidelity as a tie-breaker — not raw fidelity, because a higher-fidelity donor
    at a far N transfers almost nothing when the grid width differs. The donor's
    edges are remapped by grid position (:func:`_remap_edges_by_geometry`) so the
    per-edge transfer aligns physically. Returns ``(theta, label)`` for
    :func:`prepare_warmstart` ``extra_candidates``, or ``None``. Warns if the best
    coverage is still poor (<50%).
    """
    import glob

    from hva_vl_study_common import STUDY_ROOT

    from qmbp_simulation.circuits.ansatz_spec import AnsatzSpec
    from qmbp_simulation.models import make_lattice

    specs_dir = STUDY_ROOT / "ansatz_specs"
    pattern = f"{args.variant}_{args.topology}_N*_h{args.h:.2f}.spec.json"
    best = None  # (coverage, fid, spec, geo_nn, geo_nnn)
    for f in glob.glob(str(specs_dir / pattern)):
        try:
            s = AnsatzSpec.load(f)
        except Exception:
            continue
        if s.n_qubits >= args.n or s.theta is None:
            continue  # only SMALLER N donors with a stored θ
        if args.cross_n_donor_n and s.n_qubits != args.cross_n_donor_n:
            continue
        if (s.fidelity or 0.0) < args.cross_n_min_fid:
            continue
        lat_d = make_lattice(args.topology, s.n_qubits, J=1.0, h=args.h)
        nn_d = list(lat_d.edges)
        nnn_d = HamiltonianBuilder._generate_nnn_edges(lat_d)
        geo_nn = _remap_edges_by_geometry(nn_d, s.n_qubits, args.n)
        geo_nnn = _remap_edges_by_geometry(nnn_d, s.n_qubits, args.n)
        cov = 0.5 * (_donor_coverage(geo_nn, nn_edges)
                     + _donor_coverage(geo_nnn, nnn_edges))
        key = (cov, s.fidelity or 0.0)
        if best is None or key > (best[0], best[1]):
            best = (cov, s.fidelity or 0.0, s, geo_nn, geo_nnn)
    if best is None:
        return None

    cov, fid_d, s, geo_nn, geo_nnn = best
    # Pass geometry-remapped donor edges so transfer_theta_for_blocks matches by
    # physical position. ``None`` entries (donor cells absent in the target) are
    # kept so each donor-θ slot stays aligned; they simply never match a target.
    seed = transfer_theta_for_blocks(
        s.theta, list(base.blocks), donor_nnn_edges=geo_nnn,
        target_nnn_edges=nnn_edges, n_nn=len(nn_edges), n_qubits=args.n,
        rx_final=base.rx_final, rz_final=base.rz_final,
        donor_n_nn=len(geo_nn), donor_n_qubits=s.n_qubits,
        donor_nn_edges=geo_nn, target_nn_edges=nn_edges)
    if seed is None or seed.size != npar_ref:
        return None
    if cov < 0.5:
        print(f"[variant_topk] WARNING: best cross-N donor coverage only "
              f"{cov:.0%} (N{s.n_qubits}→N{args.n}); seed is mostly regime-fill. "
              f"Consider a closer-N donor (ladder N→N).", flush=True)
    return (seed, f"crossN<N{s.n_qubits}@{fid_d:.3f},cov{cov:.0%}>")


def run(args) -> int:
    from qmbp_simulation.execution import NoiselessBackend
    if args.variant not in VARIANTS:
        raise SystemExit(f"unknown variant {args.variant!r}; available: {list(VARIANTS)}")
    base = VARIANTS[args.variant]
    if "nnn" not in base.blocks:
        raise SystemExit(f"variant {args.variant!r} has no nnn block to prune")
    for kf in args.keep_fracs:
        if not 0.0 < kf <= 1.0:
            raise SystemExit(f"--keep-fracs must be in (0, 1]; got {kf}")
    if args.restarts < 1:
        raise SystemExit(f"--restarts must be >= 1, got {args.restarts}")

    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(
        args.topology, args.n, args.h, args.j2, 1)
    nn_edges = list(lat.edges)
    nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)
    builder = HVACircuitBuilder()
    backend = NoiselessBackend()  # shared across variants (reuses cached dense H)
    out_file = f"variant_topk_{args.variant}_{args.topology}_N{args.n}_h{args.h:.2f}.json"

    print(f"[variant_topk] N={args.n} h={args.h} variant={args.variant} "
          f"e0={e0:.5f} gap={gap:.5f} n_nn={n_nn} n_nnn={n_nnn} "
          f"keep_fracs={args.keep_fracs} restarts={args.restarts}", flush=True)

    rows: list[dict] = []
    persister = StudyPersister(
        subdir=SUBDIR, out_file=out_file,
        fingerprint={"n_qubits": args.n, "p_layers": None,
                     "model": "tfim_frustrated", "topology": args.topology,
                     "h": args.h},
        extra={"topology": args.topology, "N": args.n, "h": args.h, "J2": args.j2,
               "e0": e0, "gap": gap, "base_variant": args.variant,
               "schema": "variant_topk_v1"},
        params={"experiment": "variant_topk", "N": args.n, "h": args.h,
                "variant": args.variant},
        description="nnn top-k bond selection on a winning structure "
                    "variant — faithful state at fewer 2q gates",
        rows_ref=rows, total_restarts=args.restarts)

    def _persist(status="partial"):
        persister.persist(rows, status=status)

    # ── 1) Full structure reference (converge once) → nnn bond ranking ──────
    # The reference goes through the SAME canonical warm-start cascade as the
    # masked variants (prepare_warmstart): its analytic structure seed is one
    # extra_candidate ranked by micro-descent against the built-in calibrated/
    # regime candidates. A homogeneous warm-start across all rows + a stronger
    # donor for every masked transfer.
    ref_qc, _ = build_variant(builder, args.n, lat, base)
    n2q_ref, npar_ref = cx_and_params(ref_qc)
    analytic_ref = variant_warmstart_theta(
        base.blocks, n_nn, n_nnn, args.n, args.h, J=1.0, J2=args.j2,
        rx_final=base.rx_final, rz_final=base.rz_final)
    extra_ref = ([(analytic_ref, "analytic")]
                 if len(analytic_ref) == npar_ref else None)
    # Cross-N continuation: a converged full θ from a SMALLER N (same structure,
    # topology, h) transferred onto this N as an extra warm-start candidate — the
    # cascade's micro-descent decides if it beats the analytic/calibrated seeds.
    # Auto-discovered from the ansatz_specs library (best high-fidelity donor) or
    # forced with --cross-n-donor-n.
    cross = _load_cross_n_full_donor(args, base, nn_edges, nnn_edges, npar_ref)
    if cross is not None:
        extra_ref = (extra_ref or []) + [cross]
        print(f"[variant_topk] cross-N donor: {cross[1]}", flush=True)
    ws_ref = prepare_warmstart(
        ref_qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=args.n, p_layers=1,
        h=args.h, J2=args.j2, extra_candidates=extra_ref,
        micro_descent=args.micro_descent, backend=backend, target_len=npar_ref)
    seed_ref, seed_ref_kind = ws_ref["seed"], ws_ref["provenance"]
    t0 = time.time()
    fid_ref, e_ref, runs_ref = converge_circuit(
        ref_qc, H, psi, restarts=args.restarts, maxiter=args.maxiter,
        seed0=args.seed0, warm_theta=seed_ref, backend=backend,
        on_restart=persister.restart_callback(f"{args.variant}_full", n2q_ref, npar_ref))
    theta_ref_best = min(runs_ref, key=lambda r: r["energy"])["theta_final"]
    rows.append(build_variant_row(
        f"{args.variant}_full", fid_ref, e_ref, runs_ref, n_2q=n2q_ref,
        n_params=npar_ref, e0=e0, gap=gap, blocks=base.blocks,
        rx_final=base.rx_final, bond_selection=base.bond_selection,
        seed_kind=seed_ref_kind, seconds=time.time() - t0,
        description=base.description))
    # Portable spec for the full ansatz (records its full nn/nnn edge sets).
    save_ansatz_spec(base, topology=args.topology, n_qubits=args.n, h=args.h,
                     j2=args.j2, theta=theta_ref_best, nn_edges=nn_edges,
                     nnn_edges=nnn_edges, fidelity=fid_ref, n_2q=n2q_ref,
                     n_params=npar_ref, seed_kind=seed_ref_kind,
                     base_variant=args.variant)
    _persist()
    print(f"  {args.variant}_full fid={fid_ref:.4f} 2q={n2q_ref} "
          f"({rows[-1]['seconds']}s)", flush=True)

    theta_ref = np.asarray(rows[-1]["best_theta_final"], float)
    w_nn, w_nnn = bond_weights_for_blocks(
        theta_ref, base.blocks, n_nn, n_nnn, args.n,
        rx_final=base.rx_final, rz_final=base.rz_final)

    # ── 2) Masked selections over the SAME structure ────────────────────────
    # Two modes, both ranked from the converged full θ:
    #   T2 top-k (default): keep the strongest --keep-fracs of nnn (and, with
    #     --nn-keep-frac<1, the strongest nn subset — probes the backbone limit).
    #   T1 prune (--prune-tols): drop every nn/nnn bond with |θ| below a
    #     tolerance (selection_from_variant_theta method='prune'), the
    #     tolerance-based alternative to a fixed count.
    k_nn = (None if args.nn_keep_frac >= 1.0
            else max(1, int(round(args.nn_keep_frac * n_nn))))

    selections = []  # (label, BondSelection)
    if args.prune_tols:
        for tol in args.prune_tols:
            s = selection_from_variant_theta(
                theta_ref, list(base.blocks), nn_edges, nnn_edges, args.n,
                rx_final=base.rx_final, rz_final=base.rz_final,
                method="prune", tol=tol)
            if s.is_empty():
                print(f"  prune tol={tol} removed ALL bonds — skipped", flush=True)
                continue
            selections.append(
                (f"{args.variant}_prune{tol}_nn{len(s.nn_edges)}_nnn{len(s.nnn_edges)}", s))
    else:
        for kf in args.keep_fracs:
            k = max(1, int(round(kf * n_nnn)))
            s = top_k_by_weight(nn_edges, nnn_edges, w_nn, w_nnn, k_nn=k_nn, k_nnn=k)
            nn_tag = "" if k_nn is None else f"_nn{k_nn}of{n_nn}"
            selections.append((f"{args.variant}_topk{k}of{n_nnn}{nn_tag}", s))

    for label, sel in selections:
        v = make_masked_variant(
            label, f"{args.variant} masked: {len(sel.nn_edges)} nn + "
            f"{len(sel.nnn_edges)} nnn bonds",
            list(base.blocks), sel, rx_final=base.rx_final, rz_final=base.rz_final,
            tags=("masked", "structure"))
        qc, _ = build_variant(builder, args.n, lat, v)
        n2q, npar = cx_and_params(qc)
        seed_m, seed_kind = _best_topk_seed(
            qc, H, psi, v, sel, theta_ref, base, nnn_edges, args, backend)
        if seed_m is not None and len(seed_m) != npar:
            seed_m, seed_kind = None, "zero"
        t1 = time.time()
        fid_m, e_m, runs_m = converge_circuit(
            qc, H, psi, restarts=args.restarts, maxiter=args.maxiter,
            seed0=args.seed0, warm_theta=seed_m, backend=backend,
            on_restart=persister.restart_callback(label, n2q, npar))
        theta_m = min(runs_m, key=lambda r: r["energy"])["theta_final"]
        rows.append(build_variant_row(
            label, fid_m, e_m, runs_m, n_2q=n2q, n_params=npar, e0=e0, gap=gap,
            blocks=v.blocks, rx_final=v.rx_final, bond_selection=v.bond_selection,
            seed_kind=seed_kind, seconds=time.time() - t1, description=v.description))
        # Portable complete definition (exact edges + θ) for reuse in any runner.
        save_ansatz_spec(v, topology=args.topology, n_qubits=args.n, h=args.h,
                         j2=args.j2, theta=theta_m, fidelity=fid_m, n_2q=n2q,
                         n_params=npar, seed_kind=seed_kind,
                         base_variant=args.variant)
        _persist(status="final")
        print(f"  {label} fid={fid_m:.4f} 2q={n2q} (nn={len(sel.nn_edges)}/{n_nn} "
              f"nnn={len(sel.nnn_edges)}/{n_nnn}) Δ2q={n2q - n2q_ref:+d} "
              f"({rows[-1]['seconds']}s)", flush=True)

    print(f"\n=== RESULT (N={args.n} {args.variant}) — fidelity | 2q | fid/cx ===",
          flush=True)
    for r in sorted(rows, key=lambda x: -x["n_2q_transpiled"]):
        fpc = r["fidelity_per_cx"]
        fpc_s = f"{fpc:.2e}" if fpc is not None else "n/a"
        print(f"  {r['variant']:28s} fid={r['best_fidelity']:.4f} "
              f"2q={r['n_2q_transpiled']:>4} fid/cx={fpc_s}", flush=True)
    if not getattr(args, "no_sync", False):
        sync_scoreboard()  # fire-and-forget refresh of the shared scoreboard
    print("DONE", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="nnn top-k bond selection on a structure variant")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=18)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--variant", default="p2_half_nn_rx",
                   help="Registry variant to prune (must contain an nnn block).")
    p.add_argument("--keep-fracs", type=float, nargs="+",
                   default=[0.75, 0.5, 0.33],
                   help="nnn keep-fractions to sweep (each in (0, 1]).")
    p.add_argument("--nn-keep-frac", type=float, default=1.0,
                   help="Fraction of NN backbone bonds to keep by |θ| (default "
                        "1.0 = all nn; <1 also prunes the strongest nn subset).")
    p.add_argument("--prune-tols", type=float, nargs="+", default=None,
                   help="T1 mode: drop every nn/nnn bond with |θ| below each "
                        "tolerance (overrides --keep-fracs top-k selection).")
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--maxiter", type=int, default=800,
                   help="L-BFGS-B iters per restart (800 converges this regime; "
                        "2000 was oversized and ~3x slower at N=18).")
    p.add_argument("--micro-descent", type=int, default=60,
                   help="L-BFGS-B iters to score each top-k warm-start candidate "
                        "(transfer-from-full vs analytic); 0 = prefer transfer.")
    p.add_argument("--seed0", type=int, default=70000)
    p.add_argument("--cross-n-donor-n", type=int, default=None,
                   help="Force a specific smaller donor N for the full cross-N "
                        "warm-start (default: auto-pick highest-fidelity match).")
    p.add_argument("--cross-n-min-fid", type=float, default=0.94,
                   help="Minimum donor fidelity to reuse a smaller-N spec as a "
                        "cross-N warm-start (default 0.94 = auto-learn threshold).")
    p.add_argument("--no-sync", action="store_true",
                   help="Skip the post-run scoreboard refresh.")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
