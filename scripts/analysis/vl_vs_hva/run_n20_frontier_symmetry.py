#!/usr/bin/env python
"""N=20 frontier: orbit-complete + symmetry-tying on a HIGH-fidelity structure.

The shallow-structure tied study (run_n20_symmetry_tied) was inconclusive: its
p1 ansätze collapse to ~0.02 at N=20 (a physical depth limit, mem), so tied vs
independent had no signal. This runner instead compares on ``p3_half_nn_rx`` —
the structure that actually reaches 0.9939 at N=20 — where the differences are
measurable, and tests BOTH symmetry levers at once:

  A) bond-loose independent  — top-k bonds by |∂F/∂θ|, one param per bond (today)
  B) bond-loose tied         — same bonds, one param per symmetry orbit
  C) orbit-complete tied     — whole-orbit selection, one param per orbit

A and B share the EXACT same bonds (same 2q); C keeps whole orbits at the SAME
nnn bond-count (so all three sit at ~equal 2q). The question: does respecting the
D2 symmetry — in the SELECTION (C) and/or the PARAMETERS (B,C) — beat the
bond-resolved independent baseline at equal 2q in the degenerate manifold?

Reuses (no new physics): the converged full reference θ from the ansatz_specs
library (no re-convergence of the 524-2q full), fidelity_bond_weights_for_blocks
(the fid-grad ranking), select_orbit_complete + top_k_by_weight (the two
selectors), the tied/masked builders, compress_seed_to_orbits, prepare_warmstart
(structure-aware seed), converge_circuit with early-stop, subspace_fidelity.

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/run_n20_frontier_symmetry.py \
        --h 0.5 --keep-orbits 4 6 8 --restarts 3 --maxiter 300
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

from hva_vl_study_common import StudyPersister, save_ansatz_spec, study_dir  # noqa: E402

from qmbp_simulation.analysis.fidelity import (  # noqa: E402
    cached_degenerate_ground_manifold,
    subspace_fidelity,
)
from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.ansatz_spec import AnsatzSpec  # noqa: E402
from qmbp_simulation.circuits.bond_mask import (  # noqa: E402
    BondSelection,
    bond_weights_for_blocks,
    fidelity_bond_weights_for_blocks,
    select_orbit_complete,
    top_k_by_weight,
)
from qmbp_simulation.circuits.symmetry import (  # noqa: E402
    analyze_symmetry,
    compress_seed_to_orbits,
    detect_symmetry_group,
)
from qmbp_simulation.framework.study_core import (  # noqa: E402
    converge_circuit,
    cx_and_params,
    ground_state,
    make_cost_fid,
    prepare_warmstart,
)
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402

SUBDIR = "bond_ablation"
MODEL = "tfim_frustrated"


def _resolve_structure(name):
    """(blocks, rx_final) for a registry structure name — reuses hva_variants.VARIANTS.

    The runner is structure-parametric: p3_half_nn_rx (frontier, high-fidelity) or
    the shallow p2/p1 families where <170 2q is reachable. Blocks come from the
    single-source registry so there is no hardcoded layout to drift.
    """
    from qmbp_simulation.circuits.hva_variants import VARIANTS

    if name not in VARIANTS:
        raise SystemExit(f"unknown structure {name!r}; available: {sorted(VARIANTS)}")
    v = VARIANTS[name]
    return list(v.blocks), bool(v.rx_final)


def _load_reference_weights(args, lat, H, psi, nn_edges, nnn_edges, backend, blocks, rx_final):
    """Per-bond importance from the converged full reference of ``args.structure``.

    Loads the stored full θ from the ansatz_specs library (no re-convergence) when
    present, else converges the full once. ``--rank-by theta`` (default) uses |θ|
    (instant); ``--rank-by fidelity-grad`` uses |∂F/∂θ| (better in the degenerate
    manifold, ~500 statevector evals — opt-in).
    """
    spec_path = study_dir("ansatz_specs") / f"{args.structure}_{args.topology}_N{args.n}_h{args.h:.2f}.spec.json"
    builder = HVACircuitBuilder()
    full_sel = BondSelection(nn_edges=nn_edges, nnn_edges=nnn_edges, provenance="full")
    qc_full, _ = builder.create_bond_resolved_masked(
        args.n, lat, blocks=blocks, bond_selection=full_sel, rx_final=rx_final
    )
    _c, fid_fn, _g, _ = make_cost_fid(qc_full, H, psi, backend=backend)

    theta_full = None
    if spec_path.exists():
        spec = AnsatzSpec.load(spec_path)
        if spec.theta is not None and len(spec.theta) == qc_full.num_parameters:
            theta_full = np.asarray(spec.theta, float)
            print(f"[frontier] reference θ from spec (fid={fid_fn(theta_full):.4f})", flush=True)
    if theta_full is None:
        print("[frontier] no usable full spec — converging full reference once", flush=True)
        ws = prepare_warmstart(
            qc_full,
            H,
            psi,
            n_nn=len(nn_edges),
            n_nnn=len(nnn_edges),
            n_qubits=args.n,
            p_layers=1,
            h=args.h,
            J2=args.j2,
            backend=backend,
            topology=args.topology,
            model=MODEL,
            gap=None,
            target_len=qc_full.num_parameters,
            auto_donors=False,
            micro_descent=args.micro_descent,
            blocks=blocks,
            rx_final=rx_final,
        )
        _f, _e, runs = converge_circuit(
            qc_full,
            H,
            psi,
            restarts=args.restarts,
            maxiter=args.maxiter,
            seed0=args.seed0,
            warm_theta=np.asarray(ws["seed"], float),
            backend=backend,
            target_fid=args.target_fid,
            ftol=args.ftol,
        )
        theta_full = np.asarray(min(runs, key=lambda r: r["energy"])["theta_final"], float)

    if args.rank_by == "fidelity-grad":
        print(f"[frontier] ranking bonds by |∂F/∂θ| (~500 statevector evals, slow at large N={args.n})", flush=True)
        return fidelity_bond_weights_for_blocks(
            fid_fn, theta_full, blocks, len(nn_edges), len(nnn_edges), args.n, rx_final=rx_final
        )
    return bond_weights_for_blocks(theta_full, blocks, len(nn_edges), len(nnn_edges), args.n, rx_final=rx_final)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="N=20 frontier orbit-complete + tying (3-way)")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument(
        "--structure",
        default="p3_half_nn_rx",
        help="Registry structure to prune (e.g. p3_half_nn_rx frontier, "
        "or shallow p2_half_nn_rx / p1_half_nn_rx where <170 2q is reachable).",
    )
    p.add_argument(
        "--keep-orbits",
        type=int,
        nargs="+",
        default=[4, 6, 8],
        help="nnn-orbit budgets to sweep (orbit-complete picks this many whole orbits).",
    )
    p.add_argument(
        "--keep-nn-orbits",
        type=int,
        nargs="+",
        default=None,
        help="nn-orbit budgets to prune the BACKBONE (symmetry-faithful). "
        "None (default) keeps the full nn backbone. Pass e.g. 5 7 to also "
        "shrink nn by whole orbits — the lever for <170 2q at fid>0.8.",
    )
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--maxiter", type=int, default=300)
    p.add_argument("--seed0", type=int, default=91000)
    p.add_argument("--target-fid", type=float, default=0.98)
    p.add_argument("--ftol", type=float, default=1e-9)
    p.add_argument("--micro-descent", type=int, default=40)
    p.add_argument(
        "--rank-by",
        choices=["theta", "fidelity-grad"],
        default="theta",
        help="Bond/orbit ranking signal from the reference θ. 'theta' "
        "(default) = |θ|, instant. 'fidelity-grad' = |∂F/∂θ|, better in "
        "the degenerate manifold but ~500 statevector evals at N=20.",
    )
    args = p.parse_args(argv)
    blocks, rx_final = _resolve_structure(args.structure)

    from qmbp_simulation.execution import NoiselessBackend
    from qmbp_simulation.execution.eval_cache import CachedBackend, EvalCache

    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(args.topology, args.n, args.h, args.j2, 1)
    nn_edges = [(int(i), int(j)) for i, j in lat.edges]
    nnn_edges = [(int(i), int(j)) for i, j in HamiltonianBuilder._generate_nnn_edges(lat)]
    manifold, _e0m, phys_gap, m_dim = cached_degenerate_ground_manifold(
        H, topology=args.topology, n_qubits=args.n, model=MODEL, h=args.h, j2=args.j2
    )
    degenerate = m_dim > 1

    group = detect_symmetry_group(nn_edges, nnn_edges, args.n)
    _g, sym_report = analyze_symmetry(nn_edges, nnn_edges, args.n, topology=args.topology, h=args.h, j2=args.j2)
    print(
        f"[frontier] N={args.n} h={args.h} gap={gap:.5f} manifold_dim={m_dim} "
        f"group_order={group.order} reduction=×{sym_report.reduction_factor:.2f} "
        f"keep_orbits={args.keep_orbits}",
        flush=True,
    )

    builder = HVACircuitBuilder()
    backend = CachedBackend(
        NoiselessBackend(), topology=args.topology, n_qubits=args.n, model=MODEL, p_layers=1, cache=EvalCache()
    )
    backend.set_h(float(args.h))

    # Reference bond importance (fid-grad at the converged full θ) — drives both
    # the orbit ranking (C) and the bond ranking (A/B).
    w_nn, w_nnn = _load_reference_weights(args, lat, H, psi, nn_edges, nnn_edges, backend, blocks, rx_final)

    rows: list[dict] = []
    persister = StudyPersister(
        subdir=SUBDIR,
        out_file=f"frontier_symmetry_{args.structure}_{args.topology}_N{args.n}_h{args.h:.2f}.json",
        fingerprint={"n_qubits": args.n, "p_layers": None, "model": MODEL, "topology": args.topology, "h": args.h},
        extra={
            "topology": args.topology,
            "N": args.n,
            "h": args.h,
            "J2": args.j2,
            "schema": "frontier_symmetry_v1",
            "structure": args.structure,
            "symmetry": sym_report.to_dict(),
        },
        params={"experiment": "frontier_symmetry", "N": args.n, "h": args.h},
        description="N=20 orbit-complete + tying vs bond-loose independent (equal 2q)",
        rows_ref=rows,
    )

    def _subfid(qc, theta):
        return subspace_fidelity(qc, np.asarray(theta, float), manifold) if degenerate else None

    def _run_band(name, qc, seed):
        """Converge one band and return a result row (early-stop, subspace fid)."""
        n2q, npar = cx_and_params(qc)
        # Guard: a seed of the wrong length OR carrying NaN/Inf (a degenerate
        # transfer/projection can produce non-finite entries) falls back to a
        # cold start instead of crashing L-BFGS-B with "Binding to NaN".
        if seed is not None and (np.asarray(seed).size != npar or not np.all(np.isfinite(seed))):
            seed = None
        t0 = time.time()
        c0 = time.process_time()
        try:
            fid, e_best, runs = converge_circuit(
                qc,
                H,
                psi,
                restarts=args.restarts,
                maxiter=args.maxiter,
                seed0=args.seed0,
                warm_theta=seed,
                backend=backend,
                target_fid=args.target_fid,
                ftol=args.ftol,
            )
        except (IndexError, ValueError, RuntimeError) as exc:
            # A band may fail to converge (e.g. a tied adjoint-gradient param-index
            # edge case on some orbit-complete selections). Skip it so the sweep
            # completes and the other bands/budgets are still reported.
            print(f"  {name:38s} SKIPPED ({type(exc).__name__}: {exc})", flush=True)
            return None
        theta_best = np.asarray(min(runs, key=lambda r: r["energy"])["theta_final"], float)
        sub = _subfid(qc, theta_best)
        report_fid = sub if sub is not None else fid
        row = {
            "label": name,
            "band": name.split("|")[-1],
            "h": round(float(args.h), 2),
            "n_qubits": args.n,
            "topology": args.topology,
            "blocks": blocks,
            "e0": e0,
            "gap": gap,
            "phys_gap": phys_gap,
            "manifold_dim": m_dim,
            "degenerate": degenerate,
            "e_best": float(e_best),
            "best_fidelity": float(fid),
            "subspace_fidelity": sub,
            "report_fidelity": float(report_fid),
            "n_2q_transpiled": n2q,
            "n_params": npar,
            "fidelity_per_cx": (report_fid / n2q) if n2q else None,
            "cpu_seconds": round(time.process_time() - c0, 1),
            "seconds": round(time.time() - t0, 1),
        }
        rows.append(row)
        persister.persist(rows)
        print(f"  {name:38s} fid={report_fid:.4f} 2q={n2q:3d} params={npar:3d} ({row['cpu_seconds']}s)", flush=True)
        return row

    def _seed_for(qc, sel, blk, tied):
        """Structure-aware analytic seed for a band (orbit-compressed when tied).

        The warm-start is symmetry-projected (symmetrize_group) so a selection
        that prunes the nn backbone still starts on the symmetric manifold the
        GS lives on — the lever that lets aggressive pruning keep fidelity.
        """
        qc_ind, _ = builder.create_bond_resolved_masked(args.n, lat, blocks=blk, bond_selection=sel, rx_final=rx_final)
        ws = prepare_warmstart(
            qc_ind,
            H,
            psi,
            n_nn=len(sel.nn_edges),
            n_nnn=len(sel.nnn_edges),
            n_qubits=args.n,
            p_layers=1,
            h=args.h,
            J2=args.j2,
            backend=backend,
            topology=args.topology,
            model=MODEL,
            gap=gap,
            target_len=qc_ind.num_parameters,
            auto_donors=False,
            micro_descent=args.micro_descent,
            blocks=blk,
            rx_final=rx_final,
            symmetrize_group=group,
            symmetrize_nn_edges=sel.nn_edges,
            symmetrize_nnn_edges=sel.nnn_edges,
        )
        seed = np.asarray(ws["seed"], float)
        if not tied:
            return seed
        return compress_seed_to_orbits(seed, blk, sel.nn_edges, sel.nnn_edges, args.n, group, rx_final=rx_final)

    # nn-orbit budgets: None → keep the full backbone (the original 3-way frontier
    # mode). A list → ALSO prune the nn backbone by whole orbits (the <170 2q lever
    # for fid>0.8). The full group has len(nn_orbits) nn orbits; keep_nn=that = full.
    nn_budgets = args.keep_nn_orbits if args.keep_nn_orbits is not None else [None]

    for k_nn_orb in nn_budgets:
        for ko in args.keep_orbits:
            # orbit-complete selection: whole nnn orbits (+ optionally a pruned nn
            # backbone by whole orbits). Symmetry-faithful at every budget.
            sel_oc = select_orbit_complete(
                nn_edges, nnn_edges, w_nn, w_nnn, group, keep_nnn_orbits=ko, keep_nn_orbits=k_nn_orb
            )
            k_nnn = len(sel_oc.nnn_edges)
            k_nn = len(sel_oc.nn_edges)
            nn_tag = "" if k_nn_orb is None else f"_nnorb{k_nn_orb}"
            tag = f"ko{ko}{nn_tag}_nn{k_nn}_nnn{k_nnn}"

            # A/B comparison uses the SAME bond set (same 2q): bond-loose top-k on
            # both nn and nnn matched to the orbit-complete counts.
            k_nn_loose = None if k_nn_orb is None else k_nn
            sel_tk = top_k_by_weight(nn_edges, nnn_edges, w_nn, w_nnn, k_nn=k_nn_loose, k_nnn=k_nnn)

            qc_a, _ = builder.create_bond_resolved_masked(
                args.n, lat, blocks=blocks, bond_selection=sel_tk, rx_final=rx_final
            )
            _run_band(f"{tag}|loose_independent", qc_a, _seed_for(qc_a, sel_tk, blocks, tied=False))

            qc_b, _ = builder.create_bond_resolved_symmetry_tied(
                args.n, lat, blocks=blocks, bond_selection=sel_tk, symmetry_group=group, rx_final=rx_final
            )
            _run_band(f"{tag}|loose_tied", qc_b, _seed_for(qc_b, sel_tk, blocks, tied=True))

            qc_c, _ = builder.create_bond_resolved_symmetry_tied(
                args.n, lat, blocks=blocks, bond_selection=sel_oc, symmetry_group=group, rx_final=rx_final
            )
            row_c = _run_band(f"{tag}|orbit_complete_tied", qc_c, _seed_for(qc_c, sel_oc, blocks, tied=True))

            # Persist the orbit-complete (symmetry-faithful) band as a portable spec
            # (skip if that band failed to converge).
            if row_c is not None:
                save_ansatz_spec(
                    _as_variant(sel_oc, group, args.structure, blocks, rx_final),
                    topology=args.topology,
                    n_qubits=args.n,
                    h=args.h,
                    j2=args.j2,
                    theta=None,
                    fidelity=row_c["report_fidelity"],
                    n_2q=row_c["n_2q_transpiled"],
                    n_params=row_c["n_params"],
                    base_variant=args.structure,
                    nn_edges=nn_edges,
                    nnn_edges=nnn_edges,
                )

    persister.persist(rows, status="final")
    backend.flush()

    # 3-way summary per budget, flagging the ones under the 2q target with fid>0.8.
    print("\n=== 3-WAY (equal 2q): loose-indep vs loose-tied vs orbit-complete-tied ===", flush=True)
    by_budget: dict[str, dict] = {}
    for r in rows:
        by_budget.setdefault(r["label"].split("|")[0], {})[r["band"]] = r
    for budget, bands in by_budget.items():
        li = bands.get("loose_independent")
        lt = bands.get("loose_tied")
        oc = bands.get("orbit_complete_tied")
        if li and lt and oc:
            best = max(li["report_fidelity"], lt["report_fidelity"], oc["report_fidelity"])
            flag = "  <-- <170 2q, fid>0.8" if (li["n_2q_transpiled"] < 170 and best > 0.8) else ""
            print(
                f"  {budget:22s} 2q={li['n_2q_transpiled']:3d}  "
                f"loose_indep={li['report_fidelity']:.4f}({li['n_params']}p)  "
                f"loose_tied={lt['report_fidelity']:.4f}({lt['n_params']}p)  "
                f"orbit_tied={oc['report_fidelity']:.4f}({oc['n_params']}p){flag}",
                flush=True,
            )
    print("DONE", flush=True)
    return 0


def _as_variant(sel, group, structure, blocks, rx_final):
    """A tied AnsatzVariant carrying the orbit-complete selection (for spec save)."""
    from dataclasses import replace

    from qmbp_simulation.circuits.hva_variants import make_masked_variant

    v = make_masked_variant(
        f"{structure}_orbit_complete",
        "orbit-complete tied",
        list(blocks),
        sel,
        rx_final=rx_final,
        tags=("orbit_complete", "tied"),
    )
    return replace(v, symmetry_group=group)


if __name__ == "__main__":
    raise SystemExit(main())
