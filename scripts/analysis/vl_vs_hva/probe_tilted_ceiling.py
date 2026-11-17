#!/usr/bin/env python
"""Diagnostic: what fidelity ceiling can the [nn,nnn,x]*p ansatz reach on the
tilted N=16 cluster, and does depth lift it? Cheap, decisive — before spending
more compute on the full A/B.

The A/B showed tied p=1→p=2 lifts fid 0.16→0.30 but absolute fidelity stays low.
This isolates the question: is that a tying limitation, a depth limitation, or a
structural limitation of [nn,nnn,x] on this cluster? It converges the TIED ansatz
(tiny landscape, many restarts sweep it) at p=1..4 and reports the fidelity ceiling
per depth. Since the GS is fully symmetric (verified), the tied ceiling IS the
structural ceiling — if it climbs toward 1 with p, the route works and only needs
depth; if it plateaus low, [nn,nnn,x] is the wrong block grammar for this cluster.

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/probe_tilted_ceiling.py --n 16 --p-max 4
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

from run_tilted_tied_ab import _tilted_lattice, build_frustrated_h  # noqa: E402

from qmbp_simulation.analysis.fidelity import degenerate_ground_manifold, subspace_fidelity  # noqa: E402
from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.bond_mask import BondSelection  # noqa: E402
from qmbp_simulation.circuits.symmetry import detect_symmetry_group  # noqa: E402
from qmbp_simulation.framework.study_core import converge_circuit, cx_and_params  # noqa: E402
from qmbp_simulation.models.tilted_square import MAGIC_TILTS, generate_square_tilted  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tilted ansatz fidelity-ceiling by depth")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--j2", type=float, default=0.5)
    ap.add_argument("--p-max", type=int, default=4)
    ap.add_argument("--restarts", type=int, default=30)
    ap.add_argument("--maxiter", type=int, default=1000)
    args = ap.parse_args(argv)

    from qmbp_simulation.execution import NoiselessBackend

    l1, l2 = MAGIC_TILTS[args.n]
    cl = generate_square_tilted(l1, l2)
    H = build_frustrated_h(cl.nn_edges, cl.nnn_edges, cl.n_qubits, args.h, args.j2)
    manifold, e0, phys_gap, m_dim = degenerate_ground_manifold(H, k=6, doublet_tol=1e-3)
    psi = manifold[:, 0]
    group = detect_symmetry_group(cl.nn_edges, cl.nnn_edges, cl.n_qubits, max_syms=100000)
    lat = _tilted_lattice(cl, args.h, args.j2)
    sel = BondSelection(nn_edges=cl.nn_edges, nnn_edges=cl.nnn_edges)
    builder = HVACircuitBuilder()
    backend = NoiselessBackend()

    print(f"[ceiling] N={args.n} tilt=({l1},{l2}) h={args.h} e0={e0:.5f} md={m_dim} group={group.order}", flush=True)
    print(f"{'p':>3} {'params':>7} {'2q':>5} {'fid_tied':>9} {'ΔE':>8} {'sec':>6}", flush=True)

    for p in range(1, args.p_max + 1):
        blocks = ["nn", "nnn", "x"] * p
        qc, _ = builder.create_bond_resolved_symmetry_tied(
            cl.n_qubits, lat, blocks=blocks, bond_selection=sel, symmetry_group=group, rx_final=True
        )
        n2q, npar = cx_and_params(qc)
        t0 = time.time()
        fid, e_best, runs = converge_circuit(
            qc,
            H,
            psi,
            restarts=args.restarts,
            maxiter=args.maxiter,
            seed0=98000,
            backend=backend,
            target_fid=0.9999,
            ftol=1e-12,
        )
        theta = np.asarray(min(runs, key=lambda r: r["energy"])["theta_final"], float)
        rfid = subspace_fidelity(qc, theta, manifold if m_dim > 1 else manifold[:, :1])
        print(f"{p:>3} {npar:>7} {n2q:>5} {rfid:>9.4f} {e_best - e0:>8.4f} {time.time() - t0:>6.0f}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
