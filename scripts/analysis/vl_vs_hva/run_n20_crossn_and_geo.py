#!/usr/bin/env python
"""N=20 h=0.5 — two per-bond (NO tying) paths to short, high-fidelity ansätze.

Parameter-tying by symmetry is ruled out for this degenerate frustrated phase
(mem 390: the GS doublet breaks the symmetry spontaneously, so the optimal θ is
NOT orbit-constant — averaging within an orbit destroys the state). Both paths
here stay in the per-bond space and differ only in the WARM-START:

EXP1 — CROSS-N TRANSFER. Take a converged same-structure θ from a SMALLER N
(N=10 p3_half_nn_rx fid 0.9985, N=18 fid 0.9515) and transfer it onto the N=20
layout by GEOMETRIC grid-cell matching (transfer_theta_for_blocks with
lattice_coords). A seed that inherits a solved neighbour's basin starts far from
the fid≈0 cold-start that killed every analytic/compressed attempt. Then converge
on the full N=20 p3 layout (and a few pruned selections) and report the fidelity.

EXP2 — GEOMETRIC/SYMMETRIC p1/p2. Build SHORT p1/p2 ansätze (full bond selection,
per-bond) seeded by the structure-aware analytic warm-start (prepare_warmstart:
second-order angles informed by geometry + the lattice regime — symmetric by
construction, but applied per-bond, not tied). Measures what a cheap short ansatz
reaches at N=20 h=0.5 from an informed seed.

Reuses (no new physics): ground_state, converge_circuit, cx_and_params,
prepare_warmstart, make_cost_fid (study_core); transfer_theta_for_blocks +
lattice_coords (warmstart); AnsatzSpec.load/build (donor θ + pruned rebuild);
cached_degenerate_ground_manifold + subspace_fidelity (metric); HVACircuitBuilder
+ VARIANTS/build_variant/make_masked_variant + BondSelection (circuits);
StudyPersister + save_ansatz_spec + sync_scoreboard (persistence).

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/run_n20_crossn_and_geo.py \
        --exp crossn --donors N10 N18 --restarts 3 --maxiter 400
    .venv/bin/python scripts/analysis/vl_vs_hva/run_n20_crossn_and_geo.py \
        --exp geo --geo-structures p1_half_nn_rx p2_half_nn_rx --restarts 3 --maxiter 400
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

from qmbp_simulation.analysis.fidelity import (  # noqa: E402
    cached_degenerate_ground_manifold,
    subspace_fidelity,
)
from qmbp_simulation.analysis.warmstart import (  # noqa: E402
    lattice_coords,
    transfer_theta_for_blocks,
)
from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.ansatz_spec import AnsatzSpec  # noqa: E402
from qmbp_simulation.circuits.bond_mask import BondSelection  # noqa: E402
from qmbp_simulation.circuits.hva_variants import (  # noqa: E402
    VARIANTS,
    make_masked_variant,
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
STRUCTURE = "p3_half_nn_rx"  # the frontier structure shared across N for EXP1

# Cross-N donor specs (same structure, smaller N, converged θ).
DONOR_SPECS = {
    "N10": "p3_half_nn_rx_square_N10_h0.50",
    "N18": "p3_half_nn_rx_square_N18_h0.50",
}


def _load_donor(key, topology):
    name = DONOR_SPECS.get(key)
    if name is None:
        return None
    p = Path("results/hva_vl_study/ansatz_specs") / f"{name}.spec.json"
    if not p.exists():
        return None
    try:
        s = AnsatzSpec.load(p)
        if s.theta is None or len(s.theta) == 0:
            return None
        return s
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] donor {key} load failed: {exc}", flush=True)
        return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="N=20 cross-N transfer + geometric p1/p2 (per-bond)")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--exp", choices=["crossn", "geo", "both"], default="both")
    p.add_argument("--donors", nargs="+", default=["N10", "N18"], help="Cross-N donor keys for EXP1 (N10, N18).")
    p.add_argument(
        "--geo-structures",
        nargs="+",
        default=["p1_half_nn_rx", "p2_half_nn_rx"],
        help="Short structures for EXP2 (full per-bond selection).",
    )
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--maxiter", type=int, default=400)
    p.add_argument("--seed0", type=int, default=94000)
    p.add_argument("--target-fid", type=float, default=0.97)
    p.add_argument("--ftol", type=float, default=1e-10)
    p.add_argument(
        "--micro-descent",
        type=int,
        default=0,
        help="micro-descent for the analytic seed ranking (EXP2). 0 = "
        "no ranking pass (the transferred/analytic seed is used as-is).",
    )
    p.add_argument("--save-specs", action="store_true")
    args = p.parse_args(argv)

    from qmbp_simulation.execution import NoiselessBackend
    from qmbp_simulation.execution.eval_cache import CachedBackend, EvalCache

    t_start = time.time()
    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(args.topology, args.n, args.h, args.j2, 1)
    nn_edges = [(int(i), int(j)) for i, j in lat.edges]
    nnn_edges = [(int(i), int(j)) for i, j in HamiltonianBuilder._generate_nnn_edges(lat)]
    manifold, _e0m, phys_gap, m_dim = cached_degenerate_ground_manifold(
        H, topology=args.topology, n_qubits=args.n, model=MODEL, h=args.h, j2=args.j2
    )
    degenerate = m_dim > 1
    print(
        f"[crossn/geo] N={args.n} h={args.h} gap={gap:.5f} manifold_dim={m_dim} (setup {time.time() - t_start:.1f}s)",
        flush=True,
    )

    builder = HVACircuitBuilder()
    backend = CachedBackend(
        NoiselessBackend(), topology=args.topology, n_qubits=args.n, model=MODEL, p_layers=1, cache=EvalCache()
    )
    backend.set_h(float(args.h))
    target_coords = lattice_coords(args.topology, args.n)

    rows: list[dict] = []
    spec_sources: dict = {}
    persister = StudyPersister(
        subdir=SUBDIR,
        out_file=f"crossn_geo_{args.topology}_N{args.n}_h{args.h:.2f}.json",
        fingerprint={"n_qubits": args.n, "p_layers": None, "model": MODEL, "topology": args.topology, "h": args.h},
        extra={"topology": args.topology, "N": args.n, "h": args.h, "J2": args.j2, "schema": "crossn_geo_v1"},
        params={"experiment": "crossn_geo", "N": args.n, "h": args.h},
        description="N=20 cross-N transfer + geometric p1/p2 (per-bond, no tying)",
        rows_ref=rows,
    )

    def _subfid(qc, theta):
        return subspace_fidelity(qc, np.asarray(theta, float), manifold) if degenerate else None

    def _run(label, qc, seed, *, kind, base_fid=None, variant=None, sp_nn=None, sp_nnn=None):
        n2q, npar = cx_and_params(qc)
        s = None if seed is None else np.asarray(seed, float)
        if s is not None and (s.size != npar or not np.all(np.isfinite(s))):
            s = None
        init = _subfid(qc, s) if s is not None else None
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
            print(f"  {label:40s} SKIPPED ({type(exc).__name__}: {exc})", flush=True)
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
            "init_fidelity": init,
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
        if variant is not None:
            spec_sources[label] = (variant, sp_nn, sp_nnn)
        persister.persist(rows)
        init_s = f" init={init:.4f}" if init is not None else ""
        print(f"  {label:40s} fid={report_fid:.4f}{init_s} 2q={n2q:3d} p={npar:3d} ({row['cpu_seconds']}s)", flush=True)
        return row

    # ── EXP1: cross-N transfer onto the N=20 full p3 layout ──────────────────
    if args.exp in ("crossn", "both"):
        print("\n=== EXP1: cross-N transfer (per-bond, geometric grid matching) ===", flush=True)
        full_sel = BondSelection(nn_edges=nn_edges, nnn_edges=nnn_edges, provenance="full")
        qc_target, _ = builder.create_bond_resolved_masked(
            args.n,
            lat,
            blocks=list(VARIANTS[STRUCTURE].blocks),
            bond_selection=full_sel,
            rx_final=bool(VARIANTS[STRUCTURE].rx_final),
        )
        blk = list(VARIANTS[STRUCTURE].blocks)
        rxf = bool(VARIANTS[STRUCTURE].rx_final)
        tgt_variant = make_masked_variant(f"{STRUCTURE}_crossN", "cross-N transfer full", blk, full_sel, rx_final=rxf)
        for key in args.donors:
            donor = _load_donor(key, args.topology)
            if donor is None:
                print(f"  [skip] donor {key}: no usable spec", flush=True)
                continue
            d_nn = [(int(i), int(j)) for i, j in donor.nn_edges]
            d_nnn = [(int(i), int(j)) for i, j in donor.nnn_edges]
            d_coords = lattice_coords(args.topology, donor.n_qubits)
            transferred = transfer_theta_for_blocks(
                np.asarray(donor.theta, float),
                blk,
                donor_nnn_edges=d_nnn,
                target_nnn_edges=nnn_edges,
                n_nn=len(nn_edges),
                n_qubits=args.n,
                rx_final=rxf,
                donor_n_nn=len(d_nn),
                donor_n_qubits=donor.n_qubits,
                donor_nn_edges=d_nn,
                target_nn_edges=nn_edges,
                donor_coords=d_coords,
                target_coords=target_coords,
            )
            if transferred is None:
                print(f"  [skip] donor {key}: transfer returned None (length mismatch)", flush=True)
                continue
            _run(
                f"crossN<{key} fid{donor.fidelity:.3f}>|full",
                qc_target,
                transferred,
                kind="crossn",
                variant=tgt_variant,
                sp_nn=nn_edges,
                sp_nnn=nnn_edges,
            )

    # ── EXP2: geometric/symmetric short p1/p2, analytic structural seed ──────
    if args.exp in ("geo", "both"):
        print("\n=== EXP2: geometric/symmetric p1/p2 (per-bond, analytic seed) ===", flush=True)
        full_sel = BondSelection(nn_edges=nn_edges, nnn_edges=nnn_edges, provenance="full")
        for struct in args.geo_structures:
            if struct not in VARIANTS:
                print(f"  [skip] unknown structure {struct}", flush=True)
                continue
            v0 = VARIANTS[struct]
            blk = list(v0.blocks)
            rxf = bool(v0.rx_final)
            qc, _ = builder.create_bond_resolved_masked(args.n, lat, blocks=blk, bond_selection=full_sel, rx_final=rxf)
            ws = prepare_warmstart(
                qc,
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
                gap=gap,
                target_len=qc.num_parameters,
                auto_donors=False,
                micro_descent=args.micro_descent,
                blocks=blk,
                rx_final=rxf,
            )
            seed = np.asarray(ws["seed"], float)
            gv = make_masked_variant(f"{struct}_geo", "geometric/symmetric seed", blk, full_sel, rx_final=rxf)
            _run(f"geo<{struct}>|full", qc, seed, kind="geo", variant=gv, sp_nn=nn_edges, sp_nnn=nnn_edges)

    persister.persist(rows, status="final")
    backend.flush()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n=== CROSS-N / GEO SUMMARY (sorted by fid/2q) ===", flush=True)
    for r in sorted(rows, key=lambda x: -(x["fidelity_per_cx"] or 0)):
        init_s = f" init={r['init_fidelity']:.3f}" if r["init_fidelity"] is not None else ""
        print(
            f"  {r['label']:40s} fid={r['report_fidelity']:.4f}{init_s} "
            f"2q={r['n_2q_transpiled']:3d} p={r['n_params']:3d}",
            flush=True,
        )

    if args.save_specs and rows:
        best = max(rows, key=lambda x: x["report_fidelity"])
        print(
            f"\n[crossn/geo] best: {best['label']} fid={best['report_fidelity']:.4f} 2q={best['n_2q_transpiled']}",
            flush=True,
        )
        n_saved = 0
        for r in rows:
            src = spec_sources.get(r["label"])
            if src is None:
                continue
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
            except Exception as exc:  # noqa: BLE001
                print(f"  [spec save skipped: {r['label']}: {exc}]", flush=True)
        if n_saved:
            print(f"[crossn/geo] persisted {n_saved} spec(s); refreshing scoreboard", flush=True)
            sync_scoreboard()

    print(f"\nDONE ({time.time() - t_start:.1f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
