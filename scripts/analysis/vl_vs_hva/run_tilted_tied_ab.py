#!/usr/bin/env python
"""Phase-1 A/B on a tilted PBC cluster: symmetry-TIED vs per-bond ansatz.

Phase 0 (probe_tilted_symmetry.py) showed that the tilted PBC clusters N=16
(4,0) and N=18 (3,3) break the symmetry-vs-degeneracy exclusion that blocked
every open-boundary route: both carry the full point group (translations × C4v,
order 128 / 144) AND a non-degenerate ground state. That makes parameter TYING
viable at scale — the orbit structure collapses each block to a SINGLE orbit, so
a tied ansatz has only 3 params/layer (1 nn + 1 nnn + 1 site) independent of N.

This runner converges, on the SAME tilted cluster and at the SAME 2q count, two
parametrizations of one [nn, nnn, x] (+rx_final) layer:
  tied    : create_bond_resolved_symmetry_tied — one param per symmetry orbit
  perbond : create_bond_resolved_masked (full selection) — one param per bond

The circuits are gate-identical (same RZZ/RX placement, same 2q); only the
parameter sharing differs. If tying wins (or ties) at a fraction of the
parameters, the symmetry route finally pays off at scale — the opposite of the
open-boundary degenerate regime where tying destroyed fidelity (mem 390).

IMPORTANT: PBC is a DIFFERENT physical model than the open-boundary (OBC)
square studied elsewhere. Fidelities here are NOT comparable to the OBC
scoreboard baselines (0.8988 / 0.9491 / 0.9939 at N=20 h=0.5 were OBC). The
honest comparison is tied-vs-perbond WITHIN this tilted cluster. OBC N=16 h=0.5
has no scoreboard entry at all, so there is no same-N OBC number to beat.

Reuses (no reimplementation): generate_square_tilted, degenerate_ground_manifold,
subspace_fidelity, detect_symmetry_group, bond_orbits/orbit_index_map (via the
tied builder), make_cost_fid, converge_circuit, full_selection, BondSelection,
the two bond-resolved builders, StudyPersister.

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/run_tilted_tied_ab.py \
        --n 16 --h 0.5 --j2 0.5 --restarts 6 --maxiter 600
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SparsePauliOp

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import StudyPersister  # noqa: E402

from qmbp_simulation.analysis.fidelity import (  # noqa: E402
    degenerate_ground_manifold,
    subspace_fidelity,
)
from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.bond_mask import BondSelection  # noqa: E402
from qmbp_simulation.circuits.symmetry import (  # noqa: E402
    detect_symmetry_group,
    orbit_index_map,
    site_index_map,
)
from qmbp_simulation.framework.study_core import converge_circuit, cx_and_params  # noqa: E402
from qmbp_simulation.models import LatticeConfig  # noqa: E402
from qmbp_simulation.models.hamiltonian import compute_coordination_numbers  # noqa: E402
from qmbp_simulation.models.tilted_square import MAGIC_TILTS, generate_square_tilted  # noqa: E402

SUBDIR = "bond_ablation"
MODEL = "tfim_frustrated"
_FULL_SYMS = 100000  # uncapped group enumeration (default cap of 16 is too small for PBC)


def build_frustrated_h(nn_edges, nnn_edges, n, h, j2, j1=1.0):
    """H = -J1 Σ_nn ZZ + J2 Σ_nnn ZZ - h Σ X from explicit geometric bonds.

    Same sign convention as HamiltonianBuilder.build_frustrated_tfim but fed the
    tilted-cluster diagonals directly (NOT the 2-hop _generate_nnn_edges).
    """
    terms: list[tuple[str, list[int], complex]] = []
    for i, j in nn_edges:
        terms.append(("ZZ", [i, j], -j1))
    for i, j in nnn_edges:
        terms.append(("ZZ", [i, j], j2))
    for site in range(n):
        terms.append(("X", [site], -h))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n)


def _tilted_lattice(cl, h, j2):
    """A LatticeConfig whose edges ARE the tilted nn bonds (periodic).

    The bond-resolved builders use ``lattice`` only for do_checks and read the
    actual RZZ edges from the BondSelection, so a config carrying the tilted nn
    edges is the correct, minimal adapter. nnn is supplied separately via the
    selection (the builders never re-derive it here).
    """
    coord = compute_coordination_numbers(cl.n_qubits, cl.nn_edges)
    return LatticeConfig(
        topology="square",
        n_qubits=cl.n_qubits,
        J=1.0,
        h=h,
        edges=list(cl.nn_edges),
        coordination_numbers=coord,
        periodic=True,
    )


def _expand_tied_to_perbond(theta_tied, blocks, nn_edges, nnn_edges, n_qubits, group, rx_final):
    """Map an orbit-tied theta to the per-bond layout (inverse of the tying).

    Each per-bond slot takes the value of its symmetry orbit, reproducing the
    SAME state the tied circuit prepares but in the per-bond parametrization. The
    orbit ids come from the shared orbit_index_map / site_index_map (the exact
    maps the tied builder uses), so the expansion is the builder's tying undone.
    This is a symmetric seed the per-bond ansatz can then refine by breaking it.
    """
    nn_orbit = orbit_index_map(nn_edges, group) if nn_edges else []
    nnn_orbit = orbit_index_map(nnn_edges, group) if nnn_edges else []
    site_orbit = site_index_map(n_qubits, group)
    n_nn_p = (max(nn_orbit) + 1) if nn_orbit else 0
    n_nnn_p = (max(nnn_orbit) + 1) if nnn_orbit else 0
    n_site_p = (max(site_orbit) + 1) if site_orbit else 0
    tied_block = {"nn": n_nn_p, "nnn": n_nnn_p, "x": n_site_p, "y": n_site_p, "z": n_site_p}
    full_block = {"nn": len(nn_edges), "nnn": len(nnn_edges), "x": n_qubits, "y": n_qubits, "z": n_qubits}
    orbit_of = {"nn": nn_orbit, "nnn": nnn_orbit, "x": site_orbit, "y": site_orbit, "z": site_orbit}

    theta_tied = np.asarray(theta_tied, float)
    out: list[float] = []
    t_off = 0
    seq = list(blocks) + (["x"] if rx_final else [])
    for b in seq:
        tied_vals = theta_tied[t_off : t_off + tied_block[b]]
        out.extend(float(tied_vals[orbit_of[b][k]]) for k in range(full_block[b]))
        t_off += tied_block[b]
    return np.asarray(out, float)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Tilted PBC A/B: symmetry-tied vs per-bond, p-sweep")
    p.add_argument("--n", type=int, default=16, help="magic N (must be in MAGIC_TILTS)")
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument(
        "--p-layers",
        type=int,
        nargs="+",
        default=[1, 2],
        help="HVA depths to sweep; each p repeats the [nn,nnn,x] block p times",
    )
    p.add_argument("--rx-final", action="store_true", default=True)
    p.add_argument("--no-rx-final", dest="rx_final", action="store_false")
    p.add_argument(
        "--tied-restarts", type=int, default=40, help="many restarts for the tiny tied landscape (3-4 params/layer)"
    )
    p.add_argument("--perbond-restarts", type=int, default=6)
    p.add_argument("--maxiter", type=int, default=800)
    p.add_argument("--seed0", type=int, default=97000)
    p.add_argument("--target-fid", type=float, default=0.999)
    p.add_argument("--ftol", type=float, default=1e-12)
    args = p.parse_args(argv)

    if args.n not in MAGIC_TILTS:
        print(f"N={args.n} not a magic number; available {sorted(MAGIC_TILTS)}")
        return 1

    from qmbp_simulation.execution import NoiselessBackend

    t_start = time.time()
    l1, l2 = MAGIC_TILTS[args.n]
    cl = generate_square_tilted(l1, l2)
    nn_edges, nnn_edges = cl.nn_edges, cl.nnn_edges
    H = build_frustrated_h(nn_edges, nnn_edges, cl.n_qubits, args.h, args.j2)
    manifold, e0, phys_gap, m_dim = degenerate_ground_manifold(H, k=6, doublet_tol=1e-3)
    psi = manifold[:, 0]
    degenerate = m_dim > 1
    group = detect_symmetry_group(nn_edges, nnn_edges, cl.n_qubits, max_syms=_FULL_SYMS)
    print(
        f"[tilted_ab] N={args.n} tilt=({l1},{l2}) h={args.h} J2={args.j2} "
        f"group_order={group.order} manifold_dim={m_dim} phys_gap={phys_gap:.5f} e0={e0:.5f} "
        f"degenerate={degenerate} nn={len(nn_edges)} nnn={len(nnn_edges)} "
        f"(setup {time.time() - t_start:.1f}s)",
        flush=True,
    )

    lat = _tilted_lattice(cl, args.h, args.j2)
    sel = BondSelection(nn_edges=nn_edges, nnn_edges=nnn_edges, provenance=f"tilted_full({l1},{l2})")
    builder = HVACircuitBuilder()
    backend = NoiselessBackend()

    def report_fid(qc, theta):
        """Subspace fidelity when degenerate, exact overlap (1-col subspace) otherwise."""
        th = np.asarray(theta, float)
        return subspace_fidelity(qc, th, manifold if degenerate else manifold[:, :1])

    rows: list[dict] = []
    persister = StudyPersister(
        subdir=SUBDIR,
        out_file=f"tilted_tied_ab_N{args.n}_h{args.h:.2f}.json",
        fingerprint={"n_qubits": args.n, "p_layers": None, "model": MODEL, "topology": "square_tilted", "h": args.h},
        extra={
            "topology": "square_tilted",
            "tilt": [l1, l2],
            "N": args.n,
            "h": args.h,
            "J2": args.j2,
            "schema": "tilted_tied_ab_v2",
            "manifold_dim": m_dim,
            "group_order": group.order,
            "phys_gap": phys_gap,
            "e0": e0,
            "p_sweep": list(args.p_layers),
            "rx_final": args.rx_final,
        },
        params={"experiment": "tilted_tied_ab", "N": args.n, "h": args.h},
        description="Tilted PBC A/B: symmetry-tied vs per-bond (perbond seeded from tied), p-sweep",
        rows_ref=rows,
    )

    def _converge(qc, restarts, warm_theta=None):
        return converge_circuit(
            qc,
            H,
            psi,
            restarts=restarts,
            maxiter=args.maxiter,
            seed0=args.seed0,
            warm_theta=warm_theta,
            backend=backend,
            target_fid=args.target_fid,
            ftol=args.ftol,
        )

    def _record(arm, p_layers, qc, fid, e_best, runs, seed_prov):
        theta_best = np.asarray(min(runs, key=lambda r: r["energy"])["theta_final"], float)
        n2q, npar = cx_and_params(qc)
        rfid = report_fid(qc, theta_best)
        row = {
            "arm": arm,
            "p_layers": p_layers,
            "n_qubits": args.n,
            "tilt": [l1, l2],
            "h": round(float(args.h), 2),
            "topology": "square_tilted",
            "e0": e0,
            "phys_gap": phys_gap,
            "manifold_dim": m_dim,
            "degenerate": degenerate,
            "e_best": float(e_best),
            "best_fidelity": float(fid),
            "report_fidelity": float(rfid),
            "delta_e": float(e_best) - e0,
            "n_2q_transpiled": n2q,
            "n_params": npar,
            "fidelity_per_cx": (rfid / n2q) if n2q else None,
            "seed_provenance": seed_prov,
            "theta_best": theta_best.tolist(),
        }
        rows.append(row)
        persister.persist(rows)
        print(
            f"  p={p_layers} {arm:8s} fid={rfid:.4f} 2q={n2q:3d} p={npar:3d} ΔE={row['delta_e']:+.4f} ({seed_prov})",
            flush=True,
        )
        return row, theta_best

    blocks_for = lambda p: ["nn", "nnn", "x"] * p  # noqa: E731

    print("\n=== A/B p-sweep: tied (many restarts) vs perbond (seeded from tied) ===", flush=True)
    summary: list[tuple] = []
    for p_layers in args.p_layers:
        blocks = blocks_for(p_layers)
        print(f"\n [p={p_layers}] blocks={blocks} rx_final={args.rx_final}", flush=True)

        # TIED arm: tiny landscape (few params) → many random restarts sweep it.
        qc_tied, _ = builder.create_bond_resolved_symmetry_tied(
            cl.n_qubits, lat, blocks=blocks, bond_selection=sel, symmetry_group=group, rx_final=args.rx_final
        )
        t0 = time.time()
        fid_t, e_t, runs_t = _converge(qc_tied, args.tied_restarts)
        r_tied, theta_tied = _record("tied", p_layers, qc_tied, fid_t, e_t, runs_t, f"random×{args.tied_restarts}")
        print(f"    (tied {time.time() - t0:.1f}s)", flush=True)

        # PERBOND arm: seed from the tied optimum expanded to per-bond, then refine.
        qc_perbond, _ = builder.create_bond_resolved_masked(
            cl.n_qubits, lat, blocks=blocks, bond_selection=sel, rx_final=args.rx_final
        )
        seed_pb = _expand_tied_to_perbond(theta_tied, blocks, nn_edges, nnn_edges, cl.n_qubits, group, args.rx_final)
        assert seed_pb.size == qc_perbond.num_parameters, (
            f"seed size {seed_pb.size} != perbond params {qc_perbond.num_parameters}"
        )
        init_pb = report_fid(qc_perbond, seed_pb)
        print(f"    perbond seed init_fid={init_pb:.4f} (expanded from tied)", flush=True)
        t0 = time.time()
        fid_p, e_p, runs_p = _converge(qc_perbond, args.perbond_restarts, warm_theta=seed_pb)
        r_perbond, _ = _record("perbond", p_layers, qc_perbond, fid_p, e_p, runs_p, "tied_expanded+refine")
        print(f"    (perbond {time.time() - t0:.1f}s)", flush=True)

        summary.append((p_layers, r_tied, r_perbond, init_pb))

    persister.persist(rows, status="final")

    print("\n=== SUMMARY (per depth) ===", flush=True)
    for p_layers, r_tied, r_perbond, init_pb in summary:
        d = r_tied["report_fidelity"] - r_perbond["report_fidelity"]
        winner = "TIED" if d > 1e-4 else ("perbond" if d < -1e-4 else "tie")
        red = r_perbond["n_params"] / max(r_tied["n_params"], 1)
        print(
            f"  p={p_layers} 2q={r_tied['n_2q_transpiled']:3d}  "
            f"tied={r_tied['report_fidelity']:.4f}({r_tied['n_params']}p)  "
            f"perbond={r_perbond['report_fidelity']:.4f}({r_perbond['n_params']}p) "
            f"[seed {init_pb:.3f}]  Δ={d:+.4f} → {winner} ({red:.0f}x fewer params)",
            flush=True,
        )

    print(f"\nDONE ({time.time() - t_start:.1f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
