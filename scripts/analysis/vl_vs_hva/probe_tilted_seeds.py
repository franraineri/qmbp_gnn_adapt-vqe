#!/usr/bin/env python
"""Verify analytic warm-start seeds on the TIED tilted ansatz, small N first.

The tied ansatz has one parameter per symmetry orbit per block (1 nn + 1 nnn +
1 site per layer), and the analytic seeds (first-order, calibrated, second-order)
are ALREADY orbit-constant — a single scalar per block. So a tied seed is just
those scalars, no OBC nnn derivation needed. This probe measures the INIT fidelity
(no optimization) of each analytic seed on the tied tilted circuit, for the small
non-degenerate magic Ns first (9, 13) then 16, to see which seed launches closest.

It also reports the degeneracy + symmetry of each tilted N (control: the small
ones should be non-degenerate + symmetric like N=16, so the seed transfers).

Analytic tied seed per block (J1-J2 TFIM):
    theta_nn  = -J /(4h)              (× shrink for second-order)
    theta_nnn = -J2/(4h)              (× shrink; × suppress for structural)
    theta_x   = arctan(J/h) · x_scale  (calibrated) or arctan(J/h) (first-order)

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/probe_tilted_seeds.py --ns 9 13 16 --h 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from run_tilted_tied_ab import _tilted_lattice, build_frustrated_h  # noqa: E402

from qmbp_simulation.analysis.fidelity import degenerate_ground_manifold, subspace_fidelity  # noqa: E402
from qmbp_simulation.analysis.warmstart import calibrated_x_scale, calibrated_zz_coef  # noqa: E402
from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.bond_mask import BondSelection  # noqa: E402
from qmbp_simulation.circuits.symmetry import detect_symmetry_group  # noqa: E402
from qmbp_simulation.models.tilted_square import MAGIC_TILTS, generate_square_tilted  # noqa: E402


def tied_analytic_seed(kind, p_layers, h, j, j2, rx_final):
    """Orbit-constant analytic seed for the tied ansatz (3 scalars/layer + rx_final).

    Layout matches create_bond_resolved_symmetry_tied with 1 orbit per block:
    per layer [theta_nn, theta_nnn, theta_x]; then one rx_final site-orbit value.
    """
    a_nn = j / (4 * h)
    a_nnn = j2 / (4 * h)
    arctan_x = float(np.arctan(j / h))
    if kind == "first_order":
        nn, nnn, x = -a_nn, -a_nnn, arctan_x
    elif kind == "calibrated":
        zz = calibrated_zz_coef(h)
        xs = calibrated_x_scale(h)
        # calibrated uses zz_coef as the ZZ prefactor magnitude; same sign as first-order
        nn, nnn, x = -zz, -(zz * (j2 / j if j else 1.0)), arctan_x * xs
    elif kind == "second_order":
        shrink = 1.0 - (j / (2 * h)) ** 2 * (1.0 / 3.0)
        curv = 1.0 - (j / (2 * h)) ** 2 * (1.0 / 3.0)
        nn, nnn, x = -a_nn * shrink, -a_nnn * shrink, arctan_x * curv
    elif kind == "second_order_nn_shrink":
        shrink = 1.0 - (j / (2 * h)) ** 2 * (1.0 / 3.0)
        curv = 1.0 - (j / (2 * h)) ** 2 * (1.0 / 3.0)
        nn, nnn, x = -a_nn * shrink * 0.4, -a_nnn * shrink, arctan_x * curv
    else:
        raise ValueError(f"unknown seed kind {kind!r}")
    layer = [nn, nnn, x]
    seed = layer * p_layers
    if rx_final:
        seed = seed + [x]
    return np.clip(np.asarray(seed, float), -np.pi, np.pi)


SEEDS = ["first_order", "calibrated", "second_order", "second_order_nn_shrink"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Probe analytic tied seeds on tilted clusters")
    ap.add_argument("--ns", type=int, nargs="+", default=[9, 13, 16])
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--j2", type=float, default=0.5)
    ap.add_argument("--p-layers", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args(argv)

    builder = HVACircuitBuilder()
    for n in args.ns:
        if n not in MAGIC_TILTS:
            print(f"N={n} not magic; skip")
            continue
        l1, l2 = MAGIC_TILTS[n]
        cl = generate_square_tilted(l1, l2)
        H = build_frustrated_h(cl.nn_edges, cl.nnn_edges, cl.n_qubits, args.h, args.j2)
        manifold, e0, phys_gap, m_dim = degenerate_ground_manifold(H, k=6, doublet_tol=1e-3)
        group = detect_symmetry_group(cl.nn_edges, cl.nnn_edges, cl.n_qubits, max_syms=100000)
        lat = _tilted_lattice(cl, args.h, args.j2)
        sel = BondSelection(nn_edges=cl.nn_edges, nnn_edges=cl.nnn_edges)
        mref = manifold if m_dim > 1 else manifold[:, :1]
        tag = "NONDEGEN" if m_dim == 1 else f"DEGEN(md={m_dim})"
        print(f"\nN={n} tilt=({l1},{l2}) group={group.order} {tag} gap={phys_gap:.4f} e0={e0:.4f}", flush=True)
        for p in args.p_layers:
            blocks = ["nn", "nnn", "x"] * p
            qc, _ = builder.create_bond_resolved_symmetry_tied(
                cl.n_qubits, lat, blocks=blocks, bond_selection=sel, symmetry_group=group, rx_final=True
            )
            cells = []
            for kind in SEEDS:
                seed = tied_analytic_seed(kind, p, args.h, 1.0, args.j2, rx_final=True)
                if seed.size != qc.num_parameters:
                    cells.append(f"{kind}=len!{seed.size}")
                    continue
                fid = subspace_fidelity(qc, seed, mref)
                cells.append(f"{kind}={fid:.4f}")
            print(f"  p={p} ({qc.num_parameters}p): " + "  ".join(cells), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
