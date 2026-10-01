#!/usr/bin/env python
"""Dissect a converged θ to learn how to compress it without losing fidelity.

Per-bond |θ| pruning collapses fidelity at N>=14 (the N18 p3 full F=0.935 breaks
below 0.90 for every tolerance tried). This analyzes the WHOLE θ vector of a
converged full ansatz along several axes the raw |θ|-prune ignores, to find
compression levers that preserve the state:

  A. Block/layer energy map — mean & max |θ| per block instance (each nn, nnn, x
     layer separately), to spot near-zero LAYERS that could be dropped whole
     (structural compression) vs layers that carry the state.
  B. Per-type magnitude distribution — nn vs nnn vs x vs rx_final, with the
     fraction of angles below small thresholds (how much is "almost identity").
  C. Angle clustering — do many angles collapse onto a few recurrent values?
     Recurrent values → tie them (shared-parameter compression) instead of
     deleting them (which is what prune does and what collapses the state).
  D. Cross-N layer pattern — compare the per-layer |θ| profile of the SAME
     variant across N (p3 at N8/N10/N18). An N-invariant layer profile means the
     "which layer to compress" decision transfers, even though individual bonds
     do not (per the bond-importance finding).
  E. θ_x invariance — mean θ_x per x-layer and overall (H3 check at this N).

Pure analysis — reads AnsatzSpecs, runs NO circuits.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/analyze_theta_patterns.py \
        --spec results/hva_vl_study/ansatz_specs/p3_half_nn_rx_square_N18_h0.50.spec.json \
        --compare-variant p3_half_nn_rx
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

SPECS_DIR = STUDY_ROOT / "ansatz_specs"


def _block_slices(blocks, n_nn, n_nnn, n_qubits, rx_final, rz_final):
    """Yield (block_label_with_index, start, stop) for every block + trailing."""
    size = {"nn": n_nn, "nnn": n_nnn, "x": n_qubits, "z": n_qubits}
    counts: dict[str, int] = {}
    off = 0
    for b in blocks:
        counts[b] = counts.get(b, 0) + 1
        yield f"{b}#{counts[b]}", off, off + size[b]
        off += size[b]
    if rx_final:
        yield "rx_final", off, off + n_qubits
        off += n_qubits
    if rz_final:
        yield "rz_final", off, off + n_qubits


def _load(spec_path):
    return AnsatzSpec.load(spec_path)


def analyze_one(spec, label=""):
    theta = np.asarray(spec.theta, float)
    n_nn = len(spec.nn_edges)
    n_nnn = len(spec.nnn_edges)
    nq = spec.n_qubits
    slices = list(_block_slices(spec.blocks, n_nn, n_nnn, nq,
                                spec.rx_final, spec.rz_final))

    print(f"\n=== A. Block/layer energy map — {label or spec.variant} "
          f"(N={nq}, {theta.size} params, F={spec.fidelity:.4f}) ===")
    print(f"{'block':10s} {'npar':>5} {'mean|θ|':>8} {'max|θ|':>8} {'<0.05':>6} {'<0.1':>6}")
    profile = {}
    for lab, a, b in slices:
        seg = np.abs(theta[a:b])
        frac_tiny = float(np.mean(seg < 0.05)) if seg.size else 0.0
        frac_small = float(np.mean(seg < 0.1)) if seg.size else 0.0
        print(f"{lab:10s} {seg.size:>5} {seg.mean():>8.4f} {seg.max():>8.4f} "
              f"{frac_tiny:>6.2f} {frac_small:>6.2f}")
        profile[lab] = {"mean": float(seg.mean()), "max": float(seg.max()),
                        "frac_lt_005": frac_tiny, "frac_lt_01": frac_small}

    # B. Per-type magnitude distribution.
    print("\n=== B. Per-type magnitude ===")
    by_type = {"nn": [], "nnn": [], "x": [], "rx_final": [], "rz_final": []}
    for lab, a, b in slices:
        key = lab.split("#")[0] if "#" in lab else lab
        by_type.setdefault(key, []).extend(np.abs(theta[a:b]).tolist())
    for k, v in by_type.items():
        if not v:
            continue
        v = np.asarray(v)
        print(f"  {k:9s}: n={v.size:>3} mean|θ|={v.mean():.4f} median={np.median(v):.4f} "
              f"frac<0.05={np.mean(v<0.05):.2f} frac<0.1={np.mean(v<0.1):.2f}")

    # C. Angle clustering — recurrent values (candidates for parameter tying).
    print("\n=== C. Angle clustering (round to 0.1 rad) ===")
    rounded = np.round(theta, 1)
    vals, counts = np.unique(rounded, return_counts=True)
    order = np.argsort(-counts)
    top = [(float(vals[i]), int(counts[i])) for i in order[:8]]
    n_recurrent = int(sum(c for _, c in top if c >= 3))
    print(f"  distinct(0.1 bins)={vals.size}/{theta.size}  "
          f"most common: " + ", ".join(f"{v:+.1f}×{c}" for v, c in top))
    print(f"  angles in a bin shared by >=3 others: {n_recurrent}/{theta.size} "
          f"({n_recurrent/theta.size:.0%}) → parameter-tying candidates")

    return profile, by_type


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Analyze converged θ patterns for compression")
    p.add_argument("--spec", required=True, help="Path to the AnsatzSpec JSON to dissect.")
    p.add_argument("--compare-variant", default=None,
                   help="Variant base name to compare the per-layer profile across N "
                        "(e.g. p3_half_nn_rx).")
    p.add_argument("--topology", default="square")
    p.add_argument("--h", type=float, default=0.5)
    args = p.parse_args(argv)

    main_spec = _load(args.spec)
    profile, _ = analyze_one(main_spec, label=Path(args.spec).stem)

    # E. θ_x invariance (per x-layer mean).
    theta = np.asarray(main_spec.theta, float)
    n_nn = len(main_spec.nn_edges)
    n_nnn = len(main_spec.nnn_edges)
    nq = main_spec.n_qubits
    print("\n=== E. θ_x invariance (per x-layer mean) ===")
    xmeans = []
    for lab, a, b in _block_slices(main_spec.blocks, n_nn, n_nnn, nq,
                                   main_spec.rx_final, main_spec.rz_final):
        if lab.startswith("x") or lab == "rx_final":
            seg = theta[a:b]
            xmeans.append(seg.mean())
            print(f"  {lab:10s}: mean θ_x = {seg.mean():+.4f}  std = {seg.std():.4f}")
    if xmeans:
        xm = np.asarray(xmeans)
        cv = xm.std() / abs(xm.mean()) if xm.mean() != 0 else float("nan")
        print(f"  across x-layers: mean={xm.mean():+.4f} CV={cv:.3f}")

    # D. Cross-N per-layer profile comparison.
    if args.compare_variant:
        print(f"\n=== D. Cross-N layer profile — {args.compare_variant} "
              f"(mean|θ| per block) ===")
        specs = []
        for f in sorted(SPECS_DIR.glob(
                f"{args.compare_variant}_{args.topology}_N*_h{args.h:.2f}.spec.json")):
            if "prune" in f.name or "topk" in f.name:
                continue
            try:
                specs.append(AnsatzSpec.load(str(f)))
            except Exception:
                pass
        # Collect per-block mean|θ| keyed by block label, columns = N.
        cols = {}
        labels = None
        for s in sorted(specs, key=lambda x: x.n_qubits):
            th = np.asarray(s.theta, float)
            prof = {}
            for lab, a, b in _block_slices(s.blocks, len(s.nn_edges), len(s.nnn_edges),
                                           s.n_qubits, s.rx_final, s.rz_final):
                prof[lab] = float(np.abs(th[a:b]).mean())
            cols[s.n_qubits] = prof
            labels = list(prof.keys()) if labels is None else labels
        ns = sorted(cols)
        if labels and len(ns) >= 2:
            hdr = f"{'block':10s}" + "".join(f"{('N'+str(n)):>9}" for n in ns)
            print(hdr)
            for lab in labels:
                row = f"{lab:10s}" + "".join(f"{cols[n].get(lab, float('nan')):>9.4f}" for n in ns)
                print(row)

    print("\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
