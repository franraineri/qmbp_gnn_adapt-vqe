#!/usr/bin/env python
"""Fit a per-layer seed profile from converged full specs (h=0.5 corpus).

Goal: a warm-start that starts at a HIGHER init-fidelity than the current
calibrated/regime seeds, by matching the actual converged θ structure instead of
a single global x_scale. The θ-pattern analysis showed the state lives in the
θ_x blocks and that θ_x GROWS per layer (x#1<x#2<...<x#4), while the current
calibrated seed applies ONE uniform x_scale to every x-layer. This extracts the
real per-layer profile from every converged full spec and reports whether it is
consistent enough across N / p to serve as a seed template.

Only h=0.5 full specs exist, so this characterizes the per-LAYER profile at the
hard regime (it does NOT fit θ_x(h) — that needs specs at other h). Outputs, per
block instance (nn#k, nnn#k, x#k, rx_final):
  - mean & std of the SIGNED angle (x-layers: the per-qubit rotation value;
    zz-layers: typically near 0, so we also report mean|θ|)
  - consistency across N (is the profile N-stable → transferable template?)

Pure analysis — reads AnsatzSpecs, runs NO circuits.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/fit_seed_profile.py \
        --variant p2_half_nn_rx --topology square --h 0.5
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


def _layout(blocks, n_nn, n_nnn, nq, rx_final, rz_final):
    size = {"nn": n_nn, "nnn": n_nnn, "x": nq, "z": nq}
    counts: dict[str, int] = {}
    out = []
    off = 0
    for b in blocks:
        counts[b] = counts.get(b, 0) + 1
        out.append((f"{b}#{counts[b]}", b, off, off + size[b]))
        off += size[b]
    if rx_final:
        out.append(("rx_final", "x", off, off + nq))
        off += nq
    if rz_final:
        out.append(("rz_final", "z", off, off + nq))
    return out


def _full_specs(variant, topology, h):
    out = []
    for f in sorted(SPECS_DIR.glob(f"{variant}_{topology}_N*_h{h:.2f}.spec.json")):
        if "prune" in f.name or "topk" in f.name:
            continue
        try:
            out.append(AnsatzSpec.load(str(f)))
        except Exception:
            pass
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fit per-layer seed profile")
    p.add_argument("--variant", default="p2_half_nn_rx")
    p.add_argument("--topology", default="square")
    p.add_argument("--h", type=float, default=0.5)
    args = p.parse_args(argv)

    specs = _full_specs(args.variant, args.topology, args.h)
    if not specs:
        print(f"No full specs for {args.variant} {args.topology} h={args.h}")
        return 1
    specs.sort(key=lambda s: s.n_qubits)
    ns = [s.n_qubits for s in specs]
    print(f"=== Per-layer seed profile — {args.variant} ({args.topology}, h={args.h}) ===")
    print("specs: " + ", ".join(f"N{n}" for n in ns))

    # Build per-block-label profile across N. For x-layers report SIGNED mean
    # (the rotation value to seed); for zz report mean|θ| (sign is gauge).
    labels = None
    per_label_meanx = {}   # label -> {N: signed mean (x) or mean|θ| (zz)}
    for s in specs:
        th = np.asarray(s.theta, float)
        lay = _layout(s.blocks, len(s.nn_edges), len(s.nnn_edges), s.n_qubits,
                      s.rx_final, s.rz_final)
        if labels is None:
            labels = [lab for lab, *_ in lay]
        for lab, btype, a, b in lay:
            seg = th[a:b]
            val = float(np.mean(seg)) if btype == "x" else float(np.mean(np.abs(seg)))
            per_label_meanx.setdefault(lab, {})[s.n_qubits] = val

    print(f"\n{'block':10s} " + "".join(f"{('N'+str(n)):>9}" for n in ns)
          + f"{'mean':>9}{'CV':>7}")
    template = {}
    for lab in labels:
        vals = [per_label_meanx[lab].get(n, float('nan')) for n in ns]
        arr = np.array([v for v in vals if not np.isnan(v)])
        m = float(arr.mean()) if arr.size else float('nan')
        cv = float(arr.std() / abs(arr.mean())) if arr.size and arr.mean() != 0 else float('nan')
        template[lab] = m
        row = f"{lab:10s} " + "".join(f"{v:>9.4f}" for v in vals)
        print(row + f"{m:>9.4f}{cv:>7.2f}")

    # Summary: is the x-layer profile N-stable (low CV) and increasing per layer?
    x_labels = [lab for lab in labels if lab.startswith("x") or lab == "rx_final"]
    print("\n--- θ_x per-layer template (the proposed seed values) ---")
    for lab in x_labels:
        print(f"  {lab:10s}: θ_x ≈ {template[lab]:+.4f}")
    xs_vals = [template[lab] for lab in x_labels if lab != "rx_final"]
    if len(xs_vals) >= 2:
        incr = all(xs_vals[i] <= xs_vals[i + 1] + 0.15 for i in range(len(xs_vals) - 1))
        print(f"  monotonic-ish increase across x-layers: {incr}  "
              f"(values {[round(v, 2) for v in xs_vals]})")
    print("\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
