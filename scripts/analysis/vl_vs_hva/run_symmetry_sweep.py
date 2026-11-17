#!/usr/bin/env python
"""Symmetry-LEVEL sweep: how much does each level of the point group buy?

The 3-way frontier run answered "does tying help fidelity?" (N=9: no, it trades
~0.5% fidelity for ×4 fewer params). This runner answers the complementary,
finer question the full D4 of the 3x3 square enables: as we tie by LARGER and
larger subgroups — trivial → one reflection (order 2) → a D2 (order 4) → the
full D4 (order 8) — how do the parameter count AND the achievable fidelity move?

For a FIXED bond selection (so 2q is identical across every level), it builds the
symmetry-tied circuit under each subgroup order, converges it from the same
structure-aware warm-start (orbit-compressed to that subgroup), and reports the
params/fidelity curve. One representative subgroup per order (the one with the
strongest orbit reduction) keeps the sweep small.

Reuses (no new physics): generate_subgroups (the subgroup lattice),
create_bond_resolved_symmetry_tied, compress_seed_to_orbits, prepare_warmstart,
converge_circuit, subspace_fidelity, select_orbit_complete / full_selection.

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/run_symmetry_sweep.py \
        --n 9 --h 0.5 --keep-orbits 2 --restarts 4 --maxiter 400
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

from hva_vl_study_common import StudyPersister  # noqa: E402

from qmbp_simulation.analysis.fidelity import (  # noqa: E402
    cached_degenerate_ground_manifold,
    subspace_fidelity,
)
from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.bond_mask import (  # noqa: E402
    bond_weights_for_blocks,
    select_orbit_complete,
)
from qmbp_simulation.circuits.symmetry import (  # noqa: E402
    analyze_symmetry,
    bond_orbits,
    compress_seed_to_orbits,
    detect_symmetry_group,
    generate_subgroups,
    site_orbits,
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
BLOCKS = ["nn", "nnn", "x", "nn", "nnn", "x", "nn", "nnn", "x", "nn", "x"]
RX_FINAL = True


def _representatives(full_group, nn_edges, nnn_edges):
    """One subgroup per distinct order — the strongest-reduction representative.

    generate_subgroups returns every closed subgroup; for the sweep we want a
    single clean point per order (trivial, 2, 4, 8). Among equal-order subgroups
    we pick the one that reduces the TOTAL orbit count the most (most aggressive
    tying), so the curve is the best-case parameter reduction at each level.
    """
    subs = generate_subgroups(full_group)
    by_order: dict[int, object] = {}
    for s in subs:
        tot = len(bond_orbits(nn_edges, s)) + len(bond_orbits(nnn_edges, s)) + len(site_orbits(s.n_qubits, s))
        prev = by_order.get(s.order)
        if prev is None:
            by_order[s.order] = s
        else:
            prev_tot = (
                len(bond_orbits(nn_edges, prev))
                + len(bond_orbits(nnn_edges, prev))
                + len(site_orbits(prev.n_qubits, prev))
            )
            if tot < prev_tot:
                by_order[s.order] = s
    return [by_order[o] for o in sorted(by_order)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Symmetry-level sweep (subgroup order vs params/fidelity)")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=9)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument(
        "--keep-orbits",
        type=int,
        default=2,
        help="nnn-orbit budget (full-group orbit-complete) that fixes the bond selection / 2q.",
    )
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--maxiter", type=int, default=400)
    p.add_argument("--seed0", type=int, default=92000)
    p.add_argument("--target-fid", type=float, default=0.995)
    p.add_argument("--ftol", type=float, default=1e-10)
    p.add_argument("--micro-descent", type=int, default=60)
    args = p.parse_args(argv)

    from qmbp_simulation.execution import NoiselessBackend
    from qmbp_simulation.execution.eval_cache import CachedBackend, EvalCache

    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(args.topology, args.n, args.h, args.j2, 1)
    nn_edges = [(int(i), int(j)) for i, j in lat.edges]
    nnn_edges = [(int(i), int(j)) for i, j in HamiltonianBuilder._generate_nnn_edges(lat)]
    manifold, _e0m, phys_gap, m_dim = cached_degenerate_ground_manifold(
        H, topology=args.topology, n_qubits=args.n, model=MODEL, h=args.h, j2=args.j2
    )
    degenerate = m_dim > 1

    full_group = detect_symmetry_group(nn_edges, nnn_edges, args.n)
    _g, sym_report = analyze_symmetry(nn_edges, nnn_edges, args.n, topology=args.topology, h=args.h, j2=args.j2)
    print(
        f"[symsweep] N={args.n} h={args.h} manifold_dim={m_dim} full_group_order={full_group.order} "
        f"gs_invariant={'yes' if all(abs(v) > 0.999 for v in (sym_report.gs_invariance or {}).values()) else '?'}",
        flush=True,
    )

    builder = HVACircuitBuilder()
    backend = CachedBackend(
        NoiselessBackend(), topology=args.topology, n_qubits=args.n, model=MODEL, p_layers=1, cache=EvalCache()
    )
    backend.set_h(float(args.h))

    # Fixed bond selection (orbit-complete at the FULL group) → identical 2q for
    # every symmetry level. |θ| ranking from a quick full-circuit reference seed.
    full_sel_all = select_orbit_complete(
        nn_edges, nnn_edges, np.ones(len(nn_edges)), np.ones(len(nnn_edges)), full_group, keep_nnn_orbits=None
    )  # start from all; budget applied next
    # Rank nnn orbits by |θ| of a cheap analytic reference, then keep the budget.
    qc_ref, _ = builder.create_bond_resolved_masked(
        args.n, lat, blocks=BLOCKS, bond_selection=full_sel_all, rx_final=RX_FINAL
    )
    ws_ref = prepare_warmstart(
        qc_ref,
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
        target_len=qc_ref.num_parameters,
        auto_donors=False,
        micro_descent=args.micro_descent,
        blocks=BLOCKS,
        rx_final=RX_FINAL,
    )
    w_nn, w_nnn = bond_weights_for_blocks(
        np.asarray(ws_ref["seed"], float), BLOCKS, len(nn_edges), len(nnn_edges), args.n, rx_final=RX_FINAL
    )
    sel = select_orbit_complete(nn_edges, nnn_edges, w_nn, w_nnn, full_group, keep_nnn_orbits=args.keep_orbits)
    n2q_fixed = sel.n_2q
    print(
        f"[symsweep] fixed selection: {len(sel.nn_edges)} nn + {len(sel.nnn_edges)} nnn bonds "
        f"= {n2q_fixed} 2q (keep_orbits={args.keep_orbits})",
        flush=True,
    )

    reps = _representatives(full_group, nn_edges, nnn_edges)
    print(f"[symsweep] subgroup representatives by order: {[s.order for s in reps]}", flush=True)

    rows: list[dict] = []
    persister = StudyPersister(
        subdir=SUBDIR,
        out_file=f"symmetry_sweep_{args.topology}_N{args.n}_h{args.h:.2f}.json",
        fingerprint={"n_qubits": args.n, "p_layers": None, "model": MODEL, "topology": args.topology, "h": args.h},
        extra={
            "topology": args.topology,
            "N": args.n,
            "h": args.h,
            "J2": args.j2,
            "schema": "symmetry_sweep_v1",
            "structure": "p3_half_nn_rx",
            "n_2q_fixed": n2q_fixed,
            "keep_orbits": args.keep_orbits,
            "symmetry": sym_report.to_dict(),
        },
        params={"experiment": "symmetry_sweep", "N": args.n, "h": args.h},
        description="Symmetry-level sweep: subgroup order vs params/fidelity at fixed 2q",
        rows_ref=rows,
    )

    def _subfid(qc, theta):
        return subspace_fidelity(qc, np.asarray(theta, float), manifold) if degenerate else None

    for sub in reps:
        qc, _ = builder.create_bond_resolved_symmetry_tied(
            args.n, lat, blocks=BLOCKS, bond_selection=sel, symmetry_group=sub, rx_final=RX_FINAL
        )
        n2q, npar = cx_and_params(qc)
        # orbit-compress the full-layout analytic seed to THIS subgroup's layout
        qc_ind, _ = builder.create_bond_resolved_masked(
            args.n, lat, blocks=BLOCKS, bond_selection=sel, rx_final=RX_FINAL
        )
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
            blocks=BLOCKS,
            rx_final=RX_FINAL,
        )
        seed = compress_seed_to_orbits(
            np.asarray(ws["seed"], float), BLOCKS, sel.nn_edges, sel.nnn_edges, args.n, sub, rx_final=RX_FINAL
        )
        if seed.size != npar:
            seed = None
        t0 = time.time()
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
        theta_best = np.asarray(min(runs, key=lambda r: r["energy"])["theta_final"], float)
        sub_fid = _subfid(qc, theta_best)
        report_fid = sub_fid if sub_fid is not None else fid
        row = {
            "label": f"order{sub.order}",
            "group_order": sub.order,
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
            "subspace_fidelity": sub_fid,
            "report_fidelity": float(report_fid),
            "n_2q_transpiled": n2q,
            "n_params": npar,
            "nn_orbits": len(bond_orbits(sel.nn_edges, sub)),
            "nnn_orbits": len(bond_orbits(sel.nnn_edges, sub)),
            "site_orbits": len(site_orbits(args.n, sub)),
            "seconds": round(time.time() - t0, 1),
        }
        rows.append(row)
        persister.persist(rows)
        print(
            f"  order={sub.order:>1}  fid={report_fid:.4f}  2q={n2q:3d}  params={npar:3d}  "
            f"(nn_orb={row['nn_orbits']} nnn_orb={row['nnn_orbits']} site_orb={row['site_orbits']}) "
            f"{row['seconds']}s",
            flush=True,
        )

    persister.persist(rows, status="final")
    backend.flush()

    print(f"\n=== SYMMETRY-LEVEL SWEEP (fixed {n2q_fixed} 2q) ===", flush=True)
    print(f"{'order':>5} {'params':>7} {'fidelity':>9}  (more symmetry → fewer params)", flush=True)
    for r in sorted(rows, key=lambda x: x["group_order"]):
        print(f"{r['group_order']:>5} {r['n_params']:>7} {r['report_fidelity']:>9.4f}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
