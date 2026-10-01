#!/usr/bin/env python
"""Line B (deep) — richer geometry (#1) + importance persistence (#4).

The simple triangle feature failed to predict bond |θ| (only 2 feature levels,
Spearman ~ -0.11). This goes deeper on two fronts, still pure analysis over the
converged AnsatzSpecs — NO circuits:

#1 Richer geometry. For each nnn bond, several continuous geometric features
   from the exact 2D grid coordinates (site s → (s//cols, s%cols)):
     - euclid    : Euclidean length of the bond (plaquette diagonal ≈ √2 vs
                   longer 2-hops) — many levels, unlike the triangle count.
     - mid2center: distance of the bond midpoint to the lattice centroid
                   (edge vs interior — finite-size boundary effect).
     - min_endpoint_deg / common_nn : connectivity baselines.
   Each is rank-correlated (Spearman) with |θ|, plus a multivariate OLS R² of
   all features jointly (does any linear combination explain |θ|?).

#4 Importance persistence. Is the per-bond |θ| ranking an intrinsic, stable
   property of the lattice (hence transferable by edge index), even if not
   geometrically simple?
     - depth persistence : Spearman of nnn |θ| ranking between p1/p2/p3 at the
                           SAME N (does importance survive adding layers?).
     - cross-N persistence: Spearman restricted to the nnn edges SHARED between
                           a smaller and a larger N (does a bond important at
                           N=8 stay important at N=10/14?).
     - top-k overlap     : Jaccard of the top-⌈n/2⌉ important nnn across p's.

A strong #4 (even with weak #1) is the real justification for the cross-N donor:
it transfers the actual θ per shared edge, which only helps if importance is
stable across depth and N.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/analyze_bond_importance_deep.py \
        --topology square --h 0.5
"""
from __future__ import annotations

import argparse
import math
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


def _coords(n):
    """Exact 2D grid coordinates per site (same mapping as generate_square)."""
    cols = int(math.ceil(math.sqrt(n)))
    return {s: (s // cols, s % cols) for s in range(n)}, cols


def _spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None

    def _rank(a):
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


def _ols_r2(X, y):
    """Multivariate OLS R² of features X (cols) explaining y. None if degenerate."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if X.shape[0] < X.shape[1] + 2 or np.allclose(y, y[0]):
        return None
    A = np.column_stack([np.ones(X.shape[0]), X])
    # standardize feature columns to compare coefficients fairly
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else None


def _nn_adj(nn_edges, n):
    adj = {i: set() for i in range(n)}
    for i, j in nn_edges:
        adj[int(i)].add(int(j))
        adj[int(j)].add(int(i))
    return adj


def _geo_features(nnn_edges, n):
    """Dict of per-nnn-bond geometric feature arrays, aligned to nnn_edges."""
    coord, cols = _coords(n)
    rows = int(math.ceil(n / cols))
    center = ((rows - 1) / 2.0, (cols - 1) / 2.0)
    euclid, mid2c, orient = [], [], []
    for i, j in nnn_edges:
        (ri, ci), (rj, cj) = coord[int(i)], coord[int(j)]
        euclid.append(math.hypot(ri - rj, ci - cj))
        mr, mc = (ri + rj) / 2.0, (ci + cj) / 2.0
        mid2c.append(math.hypot(mr - center[0], mc - center[1]))
        # +1 for ↘ diagonal, -1 for ↗ (sign of slope); 0 if axis-aligned 2-hop
        dr, dc = rj - ri, cj - ci
        orient.append(0 if dr == 0 or dc == 0 else (1 if dr * dc > 0 else -1))
    return {"euclid": euclid, "mid2center": mid2c, "orient": orient}


def _load_full_specs(topology, h):
    out = []
    for f in sorted(SPECS_DIR.glob(f"*_{topology}_N*_h{h:.2f}.spec.json")):
        if "prune" in f.name or "topk" in f.name:
            continue
        try:
            s = AnsatzSpec.load(str(f))
        except Exception:
            continue
        if s.theta and s.nn_edges and s.nnn_edges:
            out.append(s)
    return out


def _w_nnn_for_spec(s, nn_edges, nnn_edges):
    _w_nn, w_nnn = bond_weights_for_blocks(
        np.asarray(s.theta, float), list(s.blocks), len(nn_edges), len(nnn_edges),
        s.n_qubits, rx_final=s.rx_final, rz_final=s.rz_final)
    return w_nnn


def analyze(topology, h):
    specs = _load_full_specs(topology, h)
    if not specs:
        print(f"No FULL specs for {topology} h={h}")
        return 1

    # Per-N lattice geometry (shared across variants of the same N).
    geo = {}
    for s in specs:
        if s.n_qubits in geo:
            continue
        lat = make_lattice(topology, s.n_qubits, J=1.0, h=h)
        nn_edges = list(lat.edges)
        nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)
        adj = _nn_adj(nn_edges, s.n_qubits)
        feats = _geo_features(nnn_edges, s.n_qubits)
        feats["common_nn"] = [len(adj[int(i)] & adj[int(j)]) for i, j in nnn_edges]
        feats["min_deg"] = [min(len(adj[int(i)]), len(adj[int(j)])) for i, j in nnn_edges]
        geo[s.n_qubits] = {"nn_edges": nn_edges, "nnn_edges": nnn_edges, "feats": feats}

    # ── #1 Richer geometry ──────────────────────────────────────────────────
    print(f"=== #1 Richer geometry → Spearman(feature, |θ_nnn|) ({topology}, h={h}) ===")
    feat_names = ["euclid", "mid2center", "orient", "common_nn", "min_deg"]
    hdr = "variant@N".ljust(20) + "".join(f"{fn:>12}" for fn in feat_names) + f"{'OLS_R²':>9}"
    print(hdr)
    for s in specs:
        g = geo[s.n_qubits]
        w = _w_nnn_for_spec(s, g["nn_edges"], g["nnn_edges"])
        cells, Xcols = "", []
        for fn in feat_names:
            c = _spearman(g["feats"][fn], w)
            cells += f"{(f'{c:+.3f}' if c is not None else 'n/a'):>12}"
            if c is not None and not np.allclose(g["feats"][fn], g["feats"][fn][0]):
                Xcols.append(g["feats"][fn])
        r2 = _ols_r2(np.column_stack(Xcols), w) if Xcols else None
        tag = f"{s.base_variant or s.variant}@N{s.n_qubits}"
        print(tag.ljust(20) + cells + f"{(f'{r2:.3f}' if r2 is not None else 'n/a'):>9}")

    # Aggregate mean |Spearman| per feature.
    print("\nmean |Spearman| per feature across specs:")
    for fn in feat_names:
        vals = []
        for s in specs:
            g = geo[s.n_qubits]
            w = _w_nnn_for_spec(s, g["nn_edges"], g["nnn_edges"])
            c = _spearman(g["feats"][fn], w)
            if c is not None:
                vals.append(abs(c))
        if vals:
            print(f"  {fn:12s}: {np.mean(vals):.3f}")

    # ── #4 Importance persistence ───────────────────────────────────────────
    print("\n=== #4a Depth persistence — Spearman(|θ_nnn|) between p's at same N ===")
    by_n: dict[int, dict[str, object]] = {}
    for s in specs:
        g = geo[s.n_qubits]
        w = _w_nnn_for_spec(s, g["nn_edges"], g["nnn_edges"])
        by_n.setdefault(s.n_qubits, {})[(s.base_variant or s.variant)] = w
    for N in sorted(by_n):
        variants = by_n[N]
        keys = sorted(variants)
        pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
        for a, b in pairs:
            c = _spearman(variants[a], variants[b])
            jac = _topk_jaccard(variants[a], variants[b])
            an = a.replace("_half_nn_rx", "")
            bn = b.replace("_half_nn_rx", "")
            print(f"  N={N:>2}: {an:>3} vs {bn:<3}  spearman={_fmt(c)}  "
                  f"top-k Jaccard={jac:.2f}")

    print("\n=== #4b Cross-N persistence — Spearman on SHARED nnn edges ===")
    # Use the p2 full at each N as the canonical importance vector.
    canon = {}
    for s in specs:
        if (s.base_variant or s.variant) == "p2_half_nn_rx":
            g = geo[s.n_qubits]
            w = _w_nnn_for_spec(s, g["nn_edges"], g["nnn_edges"])
            canon[s.n_qubits] = {tuple(sorted(map(int, e))): wi
                                 for e, wi in zip(g["nnn_edges"], w, strict=True)}
    ns = sorted(canon)
    for i, Na in enumerate(ns):
        for Nb in ns[i + 1:]:
            shared = sorted(set(canon[Na]) & set(canon[Nb]))
            if len(shared) < 3:
                print(f"  N{Na}↔N{Nb}: only {len(shared)} shared nnn edges — skip")
                continue
            wa = [canon[Na][e] for e in shared]
            wb = [canon[Nb][e] for e in shared]
            c = _spearman(wa, wb)
            print(f"  N{Na}↔N{Nb}: {len(shared)} shared nnn edges  spearman={_fmt(c)}")

    return 0


def _topk_jaccard(wa, wb, frac=0.5):
    wa = np.asarray(wa, float)
    wb = np.asarray(wb, float)
    if wa.size != wb.size or wa.size == 0:
        return float("nan")
    k = max(1, int(round(frac * wa.size)))
    ta = set(np.argsort(-np.abs(wa))[:k])
    tb = set(np.argsort(-np.abs(wb))[:k])
    return len(ta & tb) / len(ta | tb)


def _fmt(c):
    return f"{c:+.3f}" if c is not None else "  n/a"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Deep bond-importance: geometry + persistence")
    p.add_argument("--topology", default="square")
    p.add_argument("--h", type=float, default=0.5)
    args = p.parse_args(argv)
    return analyze(args.topology, args.h)


if __name__ == "__main__":
    raise SystemExit(main())
