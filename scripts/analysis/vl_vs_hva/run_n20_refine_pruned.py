#!/usr/bin/env python
"""Refine already-pruned ansätze from their stored θ — the efficient path.

The frontier runner (run_n20_frontier_symmetry) was inviable at N=20: it builds
the FULL 524-2q circuit and runs an expensive micro-descent warm-start PER BAND,
so a single band took >20 min. This runner takes the opposite, efficient route —
it never touches the full circuit. It loads the already-converged PRUNED specs
the study saved (which carry their exact bond edges AND their converged θ),
rebuilds each one directly via ``AnsatzSpec.build``, and re-optimizes it from a
warm-start that is ALREADY near the known optimum.

Two phases, both seeded from solved states (no cold start, no full-circuit rank):

PHASE 1 — refine the known pruned optima, 3 arms per spec at IDENTICAL 2q:
  stored      : re-converge from the spec's own stored θ (its known optimum)
  symmetrized : project that θ onto the D2-symmetric subspace (symmetrize_seed,
                per-bond width kept — no tying) then re-converge; cheap, can
                rescue fidelity in the degenerate manifold
  orbit_tied  : tie the SAME pruned selection by symmetry orbit (fewer params →
                easier landscape, same 2q), seeded by orbit-compressing the
                stored θ (compress_seed_to_orbits)

PHASE 2 — push BELOW the smallest pruned optimum with p1/p2 orbit-complete
structures, seeded by TRANSFERRING the converged full-p3 θ onto the shallow
layout (transfer_theta_for_blocks). Fewer layers + whole-orbit nnn selection =
fewer 2q; the transferred seed starts the shallow circuit from the solved
large-circuit angles.

Reuses (no new physics, no re-implementation):
  AnsatzSpec.load/build (exact pruned rebuild + stored θ),
  cached_degenerate_ground_manifold (the manifold NPZ cache),
  detect_symmetry_group / symmetrize_seed / compress_seed_to_orbits / bond_orbits,
  select_orbit_complete + bond_weights_for_blocks (orbit-complete selection),
  create_bond_resolved_symmetry_tied (tied builder),
  transfer_theta_for_blocks (p3→p1/p2 seed),
  converge_circuit / cx_and_params / ground_state / make_cost_fid,
  subspace_fidelity (degenerate-manifold metric),
  StudyPersister + save_ansatz_spec + sync_scoreboard (persistence).

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/run_n20_refine_pruned.py \
        --n 20 --h 0.5 --restarts 4 --maxiter 600
    .venv/bin/python scripts/analysis/vl_vs_hva/run_n20_refine_pruned.py \
        --n 20 --h 0.5 --phase2 --p2-keep-orbits 2 3 --restarts 4 --maxiter 600
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
    study_dir,
    sync_scoreboard,
)

from qmbp_simulation.analysis.fidelity import (  # noqa: E402
    cached_degenerate_ground_manifold,
    subspace_fidelity,
)
from qmbp_simulation.analysis.warmstart import transfer_theta_for_blocks  # noqa: E402
from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.ansatz_spec import AnsatzSpec  # noqa: E402
from qmbp_simulation.circuits.bond_mask import (  # noqa: E402
    bond_weights_for_blocks,
    select_orbit_complete,
)
from qmbp_simulation.circuits.symmetry import (  # noqa: E402
    compress_seed_to_orbits,
    detect_symmetry_group,
    symmetrize_seed,
)
from qmbp_simulation.framework.study_core import (  # noqa: E402
    converge_circuit,
    cx_and_params,
    ground_state,
    prepare_warmstart,
)
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402

SUBDIR = "bond_ablation"
MODEL = "tfim_frustrated"

# The pruned optima to refine (Phase 1). These are the study's best <180-2q
# specs at N=20 h=0.5; each is loaded by name and refined from its stored θ.
# Any that don't exist for the requested (N, h) are skipped with a warning.
DEFAULT_REFINE_SPECS = (
    "p3_half_nn_rx_prune0.3_th_nn11_nnn10",
    "p3_half_nn_rx_prune0.25_th_nn13_nnn12",
    "p3_half_nn_rx_prune0.2_th_nn13_nnn13",
)
# The full donor for Phase 2 transfer seeds (the converged high-fidelity ansatz).
FULL_DONOR_SPEC = "p3_half_nn_rx"


def _spec_path(name, topology, n, h):
    return study_dir("ansatz_specs") / f"{name}_{topology}_N{n}_h{h:.2f}.spec.json"


def _load_spec(name, topology, n, h):
    p = _spec_path(name, topology, n, h)
    if not p.exists():
        return None
    try:
        return AnsatzSpec.load(p)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] could not load {name}: {exc}", flush=True)
        return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Refine pruned ansätze from stored θ (efficient)")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument(
        "--specs",
        nargs="+",
        default=list(DEFAULT_REFINE_SPECS),
        help="Pruned spec names to refine (Phase 1). Defaults to the study's best <180-2q specs at N=20 h=0.5.",
    )
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--maxiter", type=int, default=600)
    p.add_argument("--seed0", type=int, default=93000)
    p.add_argument(
        "--target-fid", type=float, default=0.995, help="Early-stop once a restart clears this (set 1.0 to disable)."
    )
    p.add_argument("--ftol", type=float, default=1e-10)
    p.add_argument(
        "--arms",
        nargs="+",
        default=["orbit_tied"],
        choices=["stored", "symmetrized", "orbit_tied"],
        help="Phase-1 warm-start arms per spec. DEFAULT = orbit_tied only "
        "(the reduced symmetry space, ~2.2× fewer params at the SAME 2q, so "
        "it converges far faster than the per-bond arms). Add 'stored'/"
        "'symmetrized' only for a per-bond comparison (much slower at N≥20).",
    )
    p.add_argument("--phase2", action="store_true", help="Also run Phase 2 (p1/p2 orbit-complete, transfer-seeded).")
    p.add_argument(
        "--p2-structures",
        nargs="+",
        default=["p1_half_nn_rx", "p2_half_nn_rx"],
        help="Shallow registry structures for Phase 2.",
    )
    p.add_argument(
        "--p2-keep-orbits",
        type=int,
        nargs="+",
        default=[2, 3],
        help="nnn-orbit budgets for the Phase-2 orbit-complete selection.",
    )
    p.add_argument(
        "--save-specs", action="store_true", help="Persist improved ansätze as specs + refresh the scoreboard."
    )
    args = p.parse_args(argv)

    from qmbp_simulation.execution import NoiselessBackend
    from qmbp_simulation.execution.eval_cache import CachedBackend, EvalCache

    t_start = time.time()
    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(args.topology, args.n, args.h, args.j2, 1)
    nn_edges = [(int(i), int(j)) for i, j in lat.edges]
    nnn_edges = [(int(i), int(j)) for i, j in HamiltonianBuilder._generate_nnn_edges(lat)]
    # Manifold via the persistent NPZ cache (eigsh paid once across all runs).
    manifold, _e0m, phys_gap, m_dim = cached_degenerate_ground_manifold(
        H, topology=args.topology, n_qubits=args.n, model=MODEL, h=args.h, j2=args.j2
    )
    degenerate = m_dim > 1
    group = detect_symmetry_group(nn_edges, nnn_edges, args.n)
    print(
        f"[refine] N={args.n} h={args.h} gap={gap:.5f} manifold_dim={m_dim} "
        f"group_order={group.order} (setup {time.time() - t_start:.1f}s)",
        flush=True,
    )

    builder = HVACircuitBuilder()
    backend = CachedBackend(
        NoiselessBackend(), topology=args.topology, n_qubits=args.n, model=MODEL, p_layers=1, cache=EvalCache()
    )
    backend.set_h(float(args.h))

    rows: list[dict] = []
    persister = StudyPersister(
        subdir=SUBDIR,
        out_file=f"refine_pruned_{args.topology}_N{args.n}_h{args.h:.2f}.json",
        fingerprint={"n_qubits": args.n, "p_layers": None, "model": MODEL, "topology": args.topology, "h": args.h},
        extra={"topology": args.topology, "N": args.n, "h": args.h, "J2": args.j2, "schema": "refine_pruned_v1"},
        params={"experiment": "refine_pruned", "N": args.n, "h": args.h},
        description="Refine pruned ansätze from stored θ (stored/symmetrized/orbit_tied) + p1p2 transfer",
        rows_ref=rows,
    )

    def _subfid(qc, theta):
        return subspace_fidelity(qc, np.asarray(theta, float), manifold) if degenerate else None

    # Side table label → (variant, nn_edges, nnn_edges) for spec persistence.
    # Kept OUT of the row dicts because an AnsatzVariant is not JSON-serializable
    # and the rows are persisted incrementally to JSON by StudyPersister.
    spec_sources: dict = {}

    def _converge_and_row(label, qc, seed, *, base_fid=None, kind="phase1", variant=None, sp_nn=None, sp_nnn=None):
        """Converge one circuit from ``seed`` and record a result row.

        Guards a seed of wrong length / non-finite entries → cold start (never
        crashes L-BFGS-B). Reports subspace fidelity in the degenerate manifold.
        ``variant``/``sp_nn``/``sp_nnn`` feed the (non-serializable) spec_sources
        side table so an improved result can be re-saved as a portable spec.
        """
        if variant is not None:
            spec_sources[label] = (variant, sp_nn, sp_nnn)
        n2q, npar = cx_and_params(qc)
        s = None if seed is None else np.asarray(seed, float)
        if s is not None and (s.size != npar or not np.all(np.isfinite(s))):
            s = None
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
                warm_theta=s,
                backend=backend,
                target_fid=args.target_fid,
                ftol=args.ftol,
            )
        except (IndexError, ValueError, RuntimeError) as exc:
            print(f"  {label:42s} SKIPPED ({type(exc).__name__}: {exc})", flush=True)
            return None
        theta_best = np.asarray(min(runs, key=lambda r: r["energy"])["theta_final"], float)
        sub = _subfid(qc, theta_best)
        report_fid = sub if sub is not None else fid
        row = {
            "label": label,
            "kind": kind,
            "h": round(float(args.h), 2),
            "n_qubits": args.n,
            "topology": args.topology,
            "e0": e0,
            "gap": gap,
            "phys_gap": phys_gap,
            "manifold_dim": m_dim,
            "degenerate": degenerate,
            "e_best": float(e_best),
            "best_fidelity": float(fid),
            "subspace_fidelity": sub,
            "report_fidelity": float(report_fid),
            "baseline_fidelity": base_fid,
            "delta_vs_baseline": (float(report_fid) - base_fid) if base_fid is not None else None,
            "n_2q_transpiled": n2q,
            "n_params": npar,
            "fidelity_per_cx": (report_fid / n2q) if n2q else None,
            "theta_best": theta_best.tolist(),
            "cpu_seconds": round(time.process_time() - c0, 1),
            "seconds": round(time.time() - t0, 1),
        }
        rows.append(row)
        persister.persist(rows)
        d = f" Δ={row['delta_vs_baseline']:+.4f}" if row["delta_vs_baseline"] is not None else ""
        print(f"  {label:42s} fid={report_fid:.4f} 2q={n2q:3d} p={npar:3d}{d} ({row['cpu_seconds']}s)", flush=True)
        return row

    # ── PHASE 1: refine known pruned optima from their stored θ ──────────────
    print("\n=== PHASE 1: refine pruned optima (stored / symmetrized / orbit_tied) ===", flush=True)
    for name in args.specs:
        spec = _load_spec(name, args.topology, args.n, args.h)
        if spec is None:
            print(f"  [skip] {name}: no spec at N={args.n} h={args.h}", flush=True)
            continue
        if spec.theta is None or len(spec.theta) == 0:
            print(f"  [skip] {name}: spec has no stored θ to refine", flush=True)
            continue
        blocks = list(spec.blocks)
        rx_final = bool(spec.rx_final)
        sel_nn = [(int(i), int(j)) for i, j in spec.nn_edges]
        sel_nnn = [(int(i), int(j)) for i, j in spec.nnn_edges]
        stored_theta = np.asarray(spec.theta, float)
        base_fid = float(spec.fidelity) if spec.fidelity is not None else None
        print(f"\n [{name}] blocks={blocks} nn={len(sel_nn)} nnn={len(sel_nnn)} baseline_fid={base_fid}", flush=True)

        # Rebuild the EXACT pruned (independent) circuit — never the full.
        from qmbp_simulation.circuits.hva_variants import build_variant

        # ORBIT-TIED FIRST — the reduced symmetry space (the default path). At the
        # SAME 2q it optimizes ~2.2× fewer params than the per-bond circuit, so it
        # converges far faster; this is the "start already reduced" principle.
        if "orbit_tied" in args.arms:
            try:
                tied_variant = spec.to_variant(symmetry_group=group)
                qc_tied, _ = build_variant(builder, args.n, lat, tied_variant)
                _n2q_t, npar_t = cx_and_params(qc_tied)

                # Seed selection — the KEY lesson (discovery mem 387): compressing
                # the STORED per-bond θ destroys the state (init_fid≈0) because the
                # per-bond optimum is NOT orbit-constant. The ANALYTIC seed IS
                # orbit-constant by construction, so its orbit-compression is exact.
                # Build the analytic seed on the pruned independent circuit (cheap,
                # micro_descent=0) and compress it — the receta validated in
                # run_symmetry_sweep. Keep the stored-compressed as a second
                # candidate and start from whichever has the higher init fidelity.
                qc_ind_seed, _ = spec.build(builder=builder, lattice=lat)
                candidates = []
                try:
                    ws = prepare_warmstart(
                        qc_ind_seed,
                        H,
                        psi,
                        n_nn=len(sel_nn),
                        n_nnn=len(sel_nnn),
                        n_qubits=args.n,
                        p_layers=1,
                        h=args.h,
                        J2=args.j2,
                        backend=backend,
                        topology=args.topology,
                        model=MODEL,
                        gap=gap,
                        target_len=qc_ind_seed.num_parameters,
                        auto_donors=False,
                        micro_descent=0,
                        blocks=blocks,
                        rx_final=rx_final,
                    )
                    analytic_tied = compress_seed_to_orbits(
                        np.asarray(ws["seed"], float), blocks, sel_nn, sel_nnn, args.n, group, rx_final=rx_final
                    )
                    if analytic_tied.size == npar_t:
                        candidates.append(("analytic", analytic_tied))
                except (ValueError, IndexError):
                    pass
                stored_tied = compress_seed_to_orbits(
                    stored_theta, blocks, sel_nn, sel_nnn, args.n, group, rx_final=rx_final
                )
                if stored_tied.size == npar_t:
                    candidates.append(("stored", stored_tied))

                # Pick the seed with the higher initial subspace fidelity.
                best_seed, best_src, best_init = None, "cold", -1.0
                for src, cand in candidates:
                    f0 = _subfid(qc_tied, cand)
                    f0 = f0 if f0 is not None else 0.0
                    if f0 > best_init:
                        best_seed, best_src, best_init = cand, src, f0
                print(f"  [{name}|orbit_tied] seed={best_src} init_fid={best_init:.4f} (params {npar_t})", flush=True)
                _converge_and_row(
                    f"{name}|orbit_tied",
                    qc_tied,
                    best_seed,
                    base_fid=base_fid,
                    variant=tied_variant,
                    sp_nn=sel_nn,
                    sp_nnn=sel_nnn,
                )
            except (ValueError, IndexError) as exc:
                print(f"  {name}|orbit_tied SKIPPED ({type(exc).__name__}: {exc})", flush=True)

        # Per-bond arms are OPT-IN comparisons only (174 vs 78 params → much slower
        # at N≥20). The pruned independent circuit is built only when requested.
        if "stored" in args.arms or "symmetrized" in args.arms:
            qc_ind, _ = spec.build(builder=builder, lattice=lat)
            ind_variant = spec.to_variant()

            if "stored" in args.arms:
                _converge_and_row(
                    f"{name}|stored",
                    qc_ind,
                    stored_theta,
                    base_fid=base_fid,
                    variant=ind_variant,
                    sp_nn=sel_nn,
                    sp_nnn=sel_nnn,
                )

            if "symmetrized" in args.arms:
                try:
                    sym_seed = symmetrize_seed(stored_theta, blocks, sel_nn, sel_nnn, args.n, group, rx_final=rx_final)
                    _converge_and_row(
                        f"{name}|symmetrized",
                        qc_ind,
                        sym_seed,
                        base_fid=base_fid,
                        variant=ind_variant,
                        sp_nn=sel_nn,
                        sp_nnn=sel_nnn,
                    )
                except (ValueError, IndexError) as exc:
                    print(f"  {name}|symmetrized SKIPPED ({type(exc).__name__}: {exc})", flush=True)

    # ── PHASE 2: p1/p2 orbit-complete, seeded by transferring the full-p3 θ ──
    if args.phase2:
        print("\n=== PHASE 2: p1/p2 orbit-complete (transfer-seeded from full p3) ===", flush=True)
        donor = _load_spec(FULL_DONOR_SPEC, args.topology, args.n, args.h)
        if donor is None or donor.theta is None:
            print(f"  [skip] Phase 2: no usable full donor '{FULL_DONOR_SPEC}'", flush=True)
        else:
            donor_theta = np.asarray(donor.theta, float)
            donor_nn = [(int(i), int(j)) for i, j in donor.nn_edges]
            donor_nnn = [(int(i), int(j)) for i, j in donor.nnn_edges]
            # Rank nnn orbits by |θ| of the donor so the orbit-complete selection
            # keeps the most-important whole orbits first (reuse bond_weights).
            w_nn, w_nnn = bond_weights_for_blocks(
                donor_theta, list(donor.blocks), len(donor_nn), len(donor_nnn), args.n, rx_final=bool(donor.rx_final)
            )
            from qmbp_simulation.circuits.hva_variants import VARIANTS, build_variant

            for struct in args.p2_structures:
                if struct not in VARIANTS:
                    print(f"  [skip] unknown structure {struct}", flush=True)
                    continue
                v0 = VARIANTS[struct]
                blk = list(v0.blocks)
                rxf = bool(v0.rx_final)
                for ko in args.p2_keep_orbits:
                    sel = select_orbit_complete(nn_edges, nnn_edges, w_nn, w_nnn, group, keep_nnn_orbits=ko)
                    tag = f"{struct}_oc_nnn{len(sel.nnn_edges)}_ko{ko}"
                    # orbit-complete tied circuit at this shallow structure
                    from dataclasses import replace

                    from qmbp_simulation.circuits.hva_variants import make_masked_variant

                    mv = make_masked_variant(
                        tag, "p1/p2 orbit-complete tied", blk, sel, rx_final=rxf, tags=("orbit_complete", "tied")
                    )
                    tied_v = replace(mv, symmetry_group=group)
                    qc, _ = build_variant(builder, args.n, lat, tied_v)
                    # Seed: transfer the donor p3 θ onto this shallow nn/nnn layout,
                    # then orbit-compress to the tied layout.
                    transferred = transfer_theta_for_blocks(
                        donor_theta,
                        blk,
                        donor_nnn_edges=donor_nnn,
                        target_nnn_edges=sel.nnn_edges,
                        n_nn=len(sel.nn_edges),
                        n_qubits=args.n,
                        rx_final=rxf,
                        donor_n_nn=len(donor_nn),
                        donor_n_qubits=args.n,
                        donor_nn_edges=donor_nn,
                        target_nn_edges=sel.nn_edges,
                    )
                    seed = None
                    if transferred is not None:
                        try:
                            seed = compress_seed_to_orbits(
                                np.asarray(transferred, float),
                                blk,
                                sel.nn_edges,
                                sel.nnn_edges,
                                args.n,
                                group,
                                rx_final=rxf,
                            )
                        except (ValueError, IndexError):
                            seed = None
                    _converge_and_row(f"{tag}|orbit_tied", qc, seed, kind="phase2")

    persister.persist(rows, status="final")
    backend.flush()

    # ── Summary + best-per-2q ────────────────────────────────────────────────
    print("\n=== REFINE SUMMARY (sorted by 2q) ===", flush=True)
    improved = []
    for r in sorted(rows, key=lambda x: (x["n_2q_transpiled"], -x["report_fidelity"])):
        flag = ""
        if r.get("delta_vs_baseline") is not None and r["delta_vs_baseline"] > 1e-4:
            flag = f"  IMPROVED (+{r['delta_vs_baseline']:.4f})"
            improved.append(r)
        print(
            f"  {r['label']:42s} fid={r['report_fidelity']:.4f} 2q={r['n_2q_transpiled']:3d} "
            f"p={r['n_params']:3d}{flag}",
            flush=True,
        )

    # Persist improved ansätze as specs so sync_scoreboard surfaces them.
    # Each row carries its refined θ and its source variant (recorded at
    # convergence time), so we rebuild the exact portable spec — no inline
    # circuit rebuild, reusing save_ansatz_spec like every other runner.
    if args.save_specs and improved:
        n_saved = 0
        for r in improved:
            src = spec_sources.get(r["label"])
            if src is None:
                continue  # no persistable variant recorded for this row
            variant, sp_nn, sp_nnn = src
            try:
                save_ansatz_spec(
                    variant,
                    topology=args.topology,
                    n_qubits=args.n,
                    h=args.h,
                    j2=args.j2,
                    theta=r["theta_best"],
                    nn_edges=sp_nn,
                    nnn_edges=sp_nnn,
                    fidelity=r["report_fidelity"],
                    n_2q=r["n_2q_transpiled"],
                    n_params=r["n_params"],
                    p_layers=1,
                )
                n_saved += 1
            except Exception as exc:  # noqa: BLE001 — persistence is best-effort
                print(f"  [spec save skipped: {r['label']}: {exc}]", flush=True)
        print(f"\n[refine] persisted {n_saved} improved spec(s); refreshing scoreboard", flush=True)
        sync_scoreboard()  # idempotent, fire-and-forget scoreboard refresh

    print(f"\nDONE ({time.time() - t_start:.1f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
