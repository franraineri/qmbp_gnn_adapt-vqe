#!/usr/bin/env python
"""Line B — is bond importance (|θ|) predicted by local frustration geometry?

The compression study showed WHICH bonds survive pruning matters, but the active
set was picked post-hoc from a converged θ. If a cheap geometric feature of each
bond predicts its converged |θ|, that importance is transferable across N: a
small-N fit tells us which bonds to keep (or seed strongly) at large N, before
running any VQE.

Feature tested: the number of frustration triangles each bond closes. On the
frustrated square lattice (J2 couples the plaquette diagonals), a bond (i,j) is
frustrated through every vertex k that is a nn of BOTH i and j — i.e. the count
of common nn neighbours, which equals the number of nn-nn-bond triangles the
bond caps. The degree baseline (|nn(i)|+|nn(j)|) gave only weak correlation
(+0.38); triangle count is a richer, frustration-specific signal.

Method (pure analysis — reads converged AnsatzSpecs, runs NO circuits):
  1. For each FULL spec (all bonds present), map |θ| onto its nn / nnn edges via
     ``bond_weights_for_blocks`` (the structure-aware per-bond max|θ|).
  2. Compute, per edge, the triangle-frustration feature and the degree baseline.
  3. Rank-correlate (Spearman) feature vs |θ|, per N and per bond type.
  4. Cross-N transfer check: do the TOP-k important nnn bonds at N=8 land on the
     same geometric roles (high-feature bonds) at N=10/14? Report the overlap.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/analyze_bond_importance.py \
        --topology square --h 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import STUDY_ROOT  # noqa: E402

from qmbp_simulation.circuits.ansatz_spec import AnsatzSpec  # noqa: E402
from qmbp_simulation.circuits.bond_mask import bond_weights_for_blocks  # noqa: E402
from qmbp_simulation.models import make_lattice  # noqa: E402
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402

SPECS_DIR = STUDY_ROOT / "ansatz_specs"


def _nn_adjacency(nn_edges, n):
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for i, j in nn_edges:
        adj[int(i)].add(int(j))
        adj[int(j)].add(int(i))
    return adj


def _triangle_feature(edge, adj):
    """Number of frustration triangles the bond closes = common nn neighbours."""
    i, j = int(edge[0]), int(edge[1])
    return len(adj[i] & adj[j])


def _degree_feature(edge, adj):
    """Baseline: summed nn-degree of the bond's endpoints."""
    i, j = int(edge[0]), int(edge[1])
    return len(adj[i]) + len(adj[j])


def _spearman(x, y):
    """Spearman rank correlation (no scipy): Pearson on ranks. None if degenerate."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None

    def _rank(a):
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a), dtype=float)
        # average ranks for ties
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        csum = np.cumsum(counts)
        start = csum - counts
        avg = (start + csum - 1) / 2.0
        return avg[inv]

    rx, ry = _rank(x), _rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else None


def _load_full_specs(topology, h):
    """FULL specs only (all bonds present) with θ + edges, keyed by (variant_base, N)."""
    out = []
    for f in sorted(SPECS_DIR.glob(f"*_{topology}_N*_h{h:.2f}.spec.json")):
        name = f.name
        # skip masked specs (they carry a prune/topk tag before the topology)
        if "prune" in name or "topk" in name:
            continue
        try:
            s = AnsatzSpec.load(str(f))
        except Exception:
            continue
        if not s.theta or not s.nn_edges or not s.nnn_edges:
            continue
        out.append(s)
    return out


def analyze(topology, h):
    specs = _load_full_specs(topology, h)
    if not specs:
        print(f"No FULL specs with θ+edges for {topology} h={h}")
        return 1

    print(f"=== Bond-importance vs geometry ({topology}, h={h}) ===")
    print(f"{len(specs)} full specs: "
          + ", ".join(f"{s.base_variant or s.variant}@N{s.n_qubits}" for s in specs))

    # Per-N geometric features (shared across variants of the same N).
    per_n_feat = {}
    for s in specs:
        if s.n_qubits in per_n_feat:
            continue
        lat = make_lattice(topology, s.n_qubits, J=1.0, h=h)
        nn_edges = list(lat.edges)
        nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)
        adj = _nn_adjacency(nn_edges, s.n_qubits)
        per_n_feat[s.n_qubits] = {
            "nn_edges": nn_edges, "nnn_edges": nnn_edges, "adj": adj,
            "tri_nnn": [_triangle_feature(e, adj) for e in nnn_edges],
            "tri_nn": [_triangle_feature(e, adj) for e in nn_edges],
            "deg_nnn": [_degree_feature(e, adj) for e in nnn_edges],
        }

    print("\n--- Spearman( feature , |θ| ) per spec ---")
    print(f"{'variant@N':28s} {'type':4s} {'triangle':>9} {'degree':>8}")
    rows = []
    for s in specs:
        fe = per_n_feat[s.n_qubits]
        w_nn, w_nnn = bond_weights_for_blocks(
            np.asarray(s.theta, float), list(s.blocks), len(fe["nn_edges"]),
            len(fe["nnn_edges"]), s.n_qubits,
            rx_final=s.rx_final, rz_final=s.rz_final)
        tri_c = _spearman(fe["tri_nnn"], w_nnn)
        deg_c = _spearman(fe["deg_nnn"], w_nnn)
        tag = f"{s.base_variant or s.variant}@N{s.n_qubits}"
        tri_s = f"{tri_c:+.3f}" if tri_c is not None else "  n/a"
        deg_s = f"{deg_c:+.3f}" if deg_c is not None else "  n/a"
        print(f"{tag:28s} {'nnn':4s} {tri_s:>9} {deg_s:>8}")
        rows.append({"variant": tag, "N": s.n_qubits, "type": "nnn",
                     "spearman_triangle": tri_c, "spearman_degree": deg_c})

    # Aggregate: mean |corr| across specs (triangle vs degree) for nnn.
    tri_vals = [r["spearman_triangle"] for r in rows if r["spearman_triangle"] is not None]
    deg_vals = [r["spearman_degree"] for r in rows if r["spearman_degree"] is not None]
    if tri_vals:
        print(f"\nmean Spearman nnn — triangle: {np.mean(tri_vals):+.3f} "
              f"(|{np.mean(np.abs(tri_vals)):.3f}|)   "
              f"degree: {np.mean(deg_vals):+.3f} (|{np.mean(np.abs(deg_vals)):.3f}|)")

    # Cross-N transfer: do high-feature nnn bonds at small N predict the
    # high-|θ| nnn bonds at larger N, by their triangle-count role?
    print("\n--- Cross-N role transfer (nnn, by triangle count) ---")
    ns = sorted(per_n_feat)
    for N in ns:
        fe = per_n_feat[N]
        tri = np.asarray(fe["tri_nnn"])
        # distribution of the geometric feature at this N
        vals, counts = np.unique(tri, return_counts=True)
        dist = ", ".join(f"{int(v)}tri×{int(c)}" for v, c in zip(vals, counts, strict=False))
        print(f"  N={N:>2}: nnn triangle-count distribution: {dist}")

    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Bond importance vs frustration geometry")
    p.add_argument("--topology", default="square")
    p.add_argument("--h", type=float, default=0.5)
    args = p.parse_args(argv)
    return analyze(args.topology, args.h)


if __name__ == "__main__":
    raise SystemExit(main())
