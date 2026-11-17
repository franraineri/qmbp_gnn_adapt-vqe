#!/usr/bin/env python
"""N=20 geometric-CLUSTER nnn selection — connected bond clusters, not greedy.

The topk3of46 probe found a "magic trio" of nnn bonds — [5,7],[5,11],[11,13] —
that reached 0.9373 while topk2 and topk4 (by |∂Fid/∂θ|) both collapsed to ~0.45.
The trio works because it is GEOMETRICALLY CONNECTED (shares sites 5 and 11): a
local cluster, not three isolated high-gradient bonds. The greedy gradient ranking
missed this — fidelity is non-additive over bonds, so the 4th best-individual bond
breaks the cluster.

This runner selects nnn bonds by CONNECTIVITY instead of per-bond weight or type:
it grows connected clusters of nnn bonds (sharing sites) seeded from the lattice
centre, and converges the masked ansatz for each cluster size with the validated
calibrated warm-start. The question: does a connected cluster of k nnn bonds beat
the greedy topk-k (and the diag_axial geometric split) at the same count?

Reuses run_n20_geometric's _converge_selection (masked variant + prepare_warmstart
cascade + converge + spec persistence), BLOCKS, _grid_cols, _classify_nnn, the
shared manifold/subspace-fidelity helpers, StudyPersister.

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/run_n20_geo_cluster.py \
        --n 20 --h 0.5 --cluster-sizes 3 4 6 8 --restarts 2 --maxiter 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import StudyPersister  # noqa: E402
from run_n20_geometric import (  # noqa: E402
    MODEL,
    SUBDIR,
    _converge_selection,
    _degenerate_ground_space,
    _grid_cols,
)

from qmbp_simulation.analysis.warmstart import lattice_coords  # noqa: E402
from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.bond_mask import BondSelection  # noqa: E402
from qmbp_simulation.framework.study_core import ground_state  # noqa: E402
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402


def _bond_centrality(nnn_edges, coord):
    """Rank nnn bonds by closeness of their midpoint to the lattice centroid.

    A bulk bond (near the centre) carries more of the frustration than a boundary
    one; the centroid distance is the geometric proxy used to pick the cluster
    seed. ``coord`` is the canonical lattice_coords map (qubit → (row, col)).
    Returns nnn_edges sorted by ascending midpoint-to-centre distance.
    """
    rowvals = [rc[0] for rc in coord.values()]
    colvals = [rc[1] for rc in coord.values()]
    centre = ((min(rowvals) + max(rowvals)) / 2.0, (min(colvals) + max(colvals)) / 2.0)
    scored = []
    for a, b in nnn_edges:
        (ra, ca), (rb, cb) = coord[a], coord[b]
        mr, mc = (ra + rb) / 2.0, (ca + cb) / 2.0
        d = (mr - centre[0]) ** 2 + (mc - centre[1]) ** 2
        scored.append((d, (a, b)))
    scored.sort(key=lambda t: t[0])
    return [e for _d, e in scored]


def _grow_connected_cluster(nnn_edges, k, coord):
    """Grow a CONNECTED cluster of k nnn bonds by shared sites, seeded at centre.

    Start from the most central nnn bond; repeatedly add the bond that (a) shares
    a site with the current cluster AND (b) is itself most central, until k bonds
    are selected. If the connected frontier is exhausted before k, fall back to
    the next most-central bond (keeps growing even across a disconnection). This
    encodes the trio insight: a connected local cluster, not isolated bonds.
    """
    by_centre = _bond_centrality(nnn_edges, coord)
    if not by_centre:
        return []
    cluster = [by_centre[0]]
    sites = set(by_centre[0])
    remaining = list(by_centre[1:])
    while len(cluster) < k and remaining:
        # prefer a bond connected to the current cluster (shares a site)
        connected = [e for e in remaining if e[0] in sites or e[1] in sites]
        pick = connected[0] if connected else remaining[0]  # both centre-sorted
        cluster.append(pick)
        sites.update(pick)
        remaining.remove(pick)
    return cluster


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="N=20 geometric connected-cluster nnn selection")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--j2", type=float, default=0.5)
    ap.add_argument("--topology", default="square")
    ap.add_argument("--cluster-sizes", type=int, nargs="+", default=[3, 4, 6, 8])
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--maxiter", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=71000)
    args = ap.parse_args(argv)

    from qmbp_simulation.execution import NoiselessBackend
    from qmbp_simulation.execution.eval_cache import CachedBackend, EvalCache

    t_start = time.time()
    builder = HVACircuitBuilder()
    backend = CachedBackend(
        NoiselessBackend(), topology=args.topology, n_qubits=args.n, model=MODEL, p_layers=1, cache=EvalCache()
    )
    backend.set_h(float(args.h))

    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(args.topology, args.n, args.h, args.j2, 1)
    nn_edges = list(lat.edges)
    nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)
    cols = _grid_cols(args.n)
    coord = lattice_coords(args.topology, args.n)  # canonical qubit→(row,col) map
    if coord is None:
        print(f"[geo_cluster] no grid coords for topology {args.topology!r} — abort")
        return 1
    manifold, _e0m, phys_gap, m_dim = _degenerate_ground_space(H)
    degenerate = m_dim > 1
    print(
        f"[geo_cluster] N={args.n} h={args.h} grid {args.n // cols}x{cols} e0={e0:.5f} "
        f"phys_gap={phys_gap:.5f} manifold={m_dim} nnn={len(nnn_edges)} "
        f"(setup {time.time() - t_start:.1f}s)",
        flush=True,
    )

    rows: list[dict] = []
    persister = StudyPersister(
        subdir=SUBDIR,
        out_file=f"n20_geo_cluster_{args.topology}_N{args.n}_h{args.h:.2f}.json",
        fingerprint={"n_qubits": args.n, "p_layers": None, "model": MODEL, "topology": args.topology, "h": args.h},
        extra={
            "topology": args.topology,
            "N": args.n,
            "h": args.h,
            "J2": args.j2,
            "grid_cols": cols,
            "schema": "n20_geo_cluster_v1",
            "manifold_dim": m_dim,
            "phys_gap": phys_gap,
            "e0": e0,
            "cluster_sizes": list(args.cluster_sizes),
        },
        params={"experiment": "n20_geo_cluster", "N": args.n, "h": args.h},
        description="N=20 connected-cluster nnn selection (geometric) + calibrated warm-start",
        rows_ref=rows,
    )

    print("\n=== connected-cluster nnn (centrality-seeded, shared-site growth) ===", flush=True)
    for k in args.cluster_sizes:
        cluster = _grow_connected_cluster(nnn_edges, k, coord)
        if len(cluster) < k:
            print(f"  cluster_k{k}: only {len(cluster)} nnn reachable — skip", flush=True)
            continue
        sel = BondSelection(nn_edges=nn_edges, nnn_edges=cluster, provenance=f"geo_cluster_k{k}")
        label = f"geo_cluster_k{k}"
        row, _theta = _converge_selection(
            args, builder, backend, lat, H, psi, manifold, degenerate, e0, phys_gap, label, sel
        )
        row["cluster_bonds"] = [list(e) for e in cluster]
        rows.append(row)
        persister.persist(rows)
        print(
            f"  {label:14s} fid={row['report_fidelity']:.4f} 2q={row['n_2q_transpiled']} "
            f"nnn={len(cluster)} fid/cx={row['fidelity_per_cx']:.3e} bonds={row['cluster_bonds']}",
            flush=True,
        )

    persister.persist(rows, status="final")
    backend.flush()

    print("\n=== GEO-CLUSTER vs known topk (report_fid | 2q | nnn) ===", flush=True)
    print("  reference greedy topk (already run): topk2=0.4299 topk3=0.9373 topk4=0.4723", flush=True)
    for r in sorted(rows, key=lambda x: x["n_2q_transpiled"]):
        print(
            f"  {r['label']:14s} fid={r['report_fidelity']:.4f} 2q={r['n_2q_transpiled']:>4} "
            f"nnn={r['n_nnn']} fid/cx={r['fidelity_per_cx']:.3e}",
            flush=True,
        )

    print(f"\nDONE ({time.time() - t_start:.1f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
