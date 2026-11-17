#!/usr/bin/env python
"""Phase-0 probe: do tilted PBC clusters give C4v symmetry AND a non-degenerate
ground state for the frustrated square TFIM (J2=0.5, h=0.5)?

Background. On open rectangular grids symmetry and non-degeneracy are mutually
exclusive at N>=16 (clean grids degenerate; truncated grids break the point
group). The literature avoids both with tilted clusters N=L1^2+L2^2 and periodic
boundaries. This probe measures, on OUR model, whether that holds — the cheap,
decisive gate before any convergence work.

For each magic N it builds the tilted cluster, assembles the frustrated TFIM
    H = -J1 Σ_nn Z_iZ_j + J2 Σ_nnn Z_iZ_j - h Σ_i X_i
directly from the GEOMETRIC nn/nnn bonds (NOT the 2-hop _generate_nnn_edges,
which only matches the diagonals by accident on open grids), then reports:
  - group_order         (C4v -> 8; trivial -> 1)
  - manifold_dim        (non-degenerate -> 1; doublet -> 2)
  - phys_gap            (distance to the first state above the manifold)
  - tying reduction     (independent params / orbit-tied params)

A magic N with group_order=8 AND manifold_dim=1 reopens the symmetry route at
scale. If every magic N still degenerates, the route closes here.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/probe_tilted_symmetry.py \
        --h 0.5 --j2 0.5 --ns 10 16 18 20
"""

from __future__ import annotations

import argparse

from qiskit.quantum_info import SparsePauliOp

from qmbp_simulation.analysis.fidelity import degenerate_ground_manifold
from qmbp_simulation.circuits.symmetry import (
    bond_orbits,
    detect_symmetry_group,
    site_orbits,
)
from qmbp_simulation.models.tilted_square import MAGIC_TILTS, generate_square_tilted

# Tilted PBC groups are large (translations × C4v ≈ 128 at N=16); the default
# analyze_symmetry cap (max_syms=16) truncates the reported order to a spurious
# 17. Enumerate the full group here so group_order and the tying reduction are
# measured correctly. Orbit counts are unaffected by the cap, but the reported
# order is, so we recompute the group explicitly.
_FULL_SYMS = 100000


def build_frustrated_h(nn_edges, nnn_edges, n, h, j2, j1=1.0):
    """H = -J1 Σ_nn ZZ + J2 Σ_nnn ZZ - h Σ X, built from explicit geometric bonds.

    Same sign convention as HamiltonianBuilder.build_frustrated_tfim (NN
    ferromagnetic -J1, NNN antiferromagnetic +J2) but fed the tilted-cluster
    diagonals directly instead of the 2-hop derivation.
    """
    terms: list[tuple[str, list[int], complex]] = []
    for i, j in nn_edges:
        terms.append(("ZZ", [i, j], -j1))
    for i, j in nnn_edges:
        terms.append(("ZZ", [i, j], j2))
    for site in range(n):
        terms.append(("X", [site], -h))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n)


def probe_one(n, h, j2):
    l1, l2 = MAGIC_TILTS[n]
    cl = generate_square_tilted(l1, l2)
    H = build_frustrated_h(cl.nn_edges, cl.nnn_edges, cl.n_qubits, h, j2)

    manifold, e0, phys_gap, m_dim = degenerate_ground_manifold(H, k=6, doublet_tol=1e-3)

    # Full symmetry group (uncapped) for the true order and tying reduction.
    group = detect_symmetry_group(cl.nn_edges, cl.nnn_edges, cl.n_qubits, max_syms=_FULL_SYMS)
    nn_orb = bond_orbits(cl.nn_edges, group)
    nnn_orb = bond_orbits(cl.nnn_edges, group)
    s_orb = site_orbits(cl.n_qubits, group)
    params_indep = len(cl.nn_edges) + len(cl.nnn_edges) + cl.n_qubits
    params_tied = len(nn_orb) + len(nnn_orb) + len(s_orb)
    reduction = params_indep / params_tied if params_tied else 1.0

    # C4v is a subgroup of order 8; a tilted PBC cluster that preserves the full
    # point group has order (translations × 8). Require at least the 8 point-group
    # elements to call it "C4v-preserving".
    is_c4v = group.order >= 8
    return {
        "n": n,
        "tilt": (l1, l2),
        "group_order": group.order,
        "manifold_dim": m_dim,
        "phys_gap": phys_gap,
        "e0": e0,
        "nn": len(cl.nn_edges),
        "nnn": len(cl.nnn_edges),
        "params_indep": params_indep,
        "params_tied": params_tied,
        "reduction": reduction,
        "pass": is_c4v and m_dim == 1,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Tilted PBC symmetry+degeneracy probe")
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument(
        "--ns", type=int, nargs="+", default=[10, 16, 18, 20], help="magic Ns to probe (must be in MAGIC_TILTS)"
    )
    args = p.parse_args(argv)

    for n in args.ns:
        if n not in MAGIC_TILTS:
            print(f"  N={n} not a magic number — skip (available {sorted(MAGIC_TILTS)})")

    rows = [probe_one(n, args.h, args.j2) for n in args.ns if n in MAGIC_TILTS]

    print(f"\nTilted PBC frustrated square — h={args.h} J2={args.j2}")
    print(
        f"{'N':>4} {'tilt':>7} {'nn':>4} {'nnn':>4} {'group':>6} {'md':>3} "
        f"{'phys_gap':>9} {'reduction':>10} {'verdict':>18}"
    )
    for r in rows:
        if r["group_order"] >= 8 and r["manifold_dim"] == 1:
            verdict = "C4v + NONDEGEN ✓"
        elif r["manifold_dim"] > 1:
            verdict = f"degenerate (md={r['manifold_dim']})"
        else:
            verdict = f"low sym (g={r['group_order']})"
        print(
            f"{r['n']:>4} {str(r['tilt']):>7} {r['nn']:>4} {r['nnn']:>4} "
            f"{r['group_order']:>6} {r['manifold_dim']:>3} {r['phys_gap']:>9.5f} "
            f"{r['reduction']:>9.2f}x {verdict:>18}"
        )

    winners = [r for r in rows if r["pass"]]
    print()
    if winners:
        print(f"GATE PASS: {len(winners)} magic N give C4v + non-degenerate:")
        for r in winners:
            print(
                f"  N={r['n']} tilt={r['tilt']}: tying {r['params_indep']}→"
                f"{r['params_tied']} params ({r['reduction']:.2f}x reduction)"
            )
    else:
        print("GATE FAIL: no magic N gives C4v + non-degenerate simultaneously.")
        print("The tilted-PBC symmetry route closes — frustration induces its own")
        print("degeneracy independent of the boundary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
