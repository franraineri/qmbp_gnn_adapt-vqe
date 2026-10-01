#!/usr/bin/env python
"""Compress a converged full by zeroing tiny angles and dropping the RZZ(0) gates.

The layer-ablation study showed per-|θ| PRUNING collapses fidelity (reordering
non-commuting blocks), but ZEROING tiny angles in place keeps it (F=0.891 at N18
when 130 params → 0). An RZZ(θ=0) is the identity and commutes trivially, so —
unlike dropping a whole block — it can be REMOVED without reordering the rest.
This turns the parameter-level zeroing into a REAL 2q-gate reduction.

Method, per threshold ε (NO optimization — direct evaluation):
  1. Take the converged full θ. Flag RZZ (nn/nnn) params with |θ| < ε.
  2. Build the masked circuit WITHOUT those bonds (BondSelection drops the
     zero RZZ edges per layer), binding the surviving angles verbatim.
  3. Report F of the compressed circuit and its REAL transpiled 2q count.
Because RZZ(0)=I, F of the compressed circuit must equal F of the full θ with
those angles set to 0 (verified as a cross-check).

Pure evaluation — builds circuits + exact fidelity + transpile count; NO optimizer.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/run_zero_tiny_compress.py \
        --spec results/hva_vl_study/ansatz_specs/p3_half_nn_rx_square_N18_h0.50.spec.json \
        --eps 0.02 0.05 0.08 0.1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.ansatz_spec import AnsatzSpec  # noqa: E402
from qmbp_simulation.circuits.bond_mask import BondSelection  # noqa: E402
from qmbp_simulation.circuits.hva_variants import build_variant, make_masked_variant  # noqa: E402
from qmbp_simulation.framework.study_core import cx_and_params, ground_state, make_cost_fid  # noqa: E402


def _per_layer_zero_mask(theta, blocks, n_nn, n_nnn, nq, rx_final, rz_final, eps):
    """Which nn / nnn edges have |θ|<ε in EVERY layer they appear (safe to drop).

    An edge's RZZ can be removed only if it is ~0 in all its layer instances
    (dropping it removes the gate from every layer). Returns (keep_nn_idx,
    keep_nnn_idx): booleans over the nn / nnn edge lists marking edges to KEEP.
    """
    size = {"nn": n_nn, "nnn": n_nnn, "x": nq, "z": nq}
    # max |θ| per edge across its layer instances
    max_nn = np.zeros(n_nn)
    max_nnn = np.zeros(n_nnn)
    off = 0
    for b in blocks:
        seg = np.abs(theta[off:off + size[b]])
        if b == "nn":
            max_nn = np.maximum(max_nn, seg)
        elif b == "nnn":
            max_nnn = np.maximum(max_nnn, seg)
        off += size[b]
    return max_nn >= eps, max_nnn >= eps


def main(argv=None) -> int:
    from qmbp_simulation.execution import NoiselessBackend

    p = argparse.ArgumentParser(description="zero-tiny + drop RZZ(0) compression")
    p.add_argument("--spec", required=True)
    p.add_argument("--topology", default="square")
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--eps", type=float, nargs="+", default=[0.02, 0.05, 0.08, 0.1])
    args = p.parse_args(argv)

    spec = AnsatzSpec.load(args.spec)
    n = spec.n_qubits
    lat, _qc, H, psi, e0, gap, _nn, _nnn = ground_state(args.topology, n, args.h, args.j2, 1)
    builder = HVACircuitBuilder()
    backend = NoiselessBackend()
    theta = np.asarray(spec.theta, float)
    nn_edges = list(spec.nn_edges)
    nnn_edges = list(spec.nnn_edges)
    n_nn, n_nnn = len(nn_edges), len(nnn_edges)

    # Full reference.
    qc_full, _ = build_variant(builder, n, lat, spec.to_variant())
    n2q_full, npar_full = cx_and_params(qc_full)
    _c, fid_full, _g, _ = make_cost_fid(qc_full, H, psi, backend=backend)
    F_full = float(fid_full(theta))

    print(f"=== zero-tiny + drop RZZ(0) — {Path(args.spec).stem} ===")
    print(f"N={n} F_full={F_full:.4f} 2q={n2q_full} npar={npar_full} "
          f"(nn={n_nn} nnn={n_nnn})\n")
    print(f"{'eps':>5} {'F_zeroed':>9} {'F_dropped':>10} {'2q':>5} {'Δ2q':>7} "
          f"{'nn_kept':>8} {'nnn_kept':>9} {'match':>6}")

    for eps in args.eps:
        # 1) in-place zeroing cross-check (RZZ with max|θ|<eps across layers → 0)
        keep_nn, keep_nnn = _per_layer_zero_mask(
            theta, spec.blocks, n_nn, n_nnn, n, spec.rx_final, spec.rz_final, eps)
        tiny = theta.copy()
        size = {"nn": n_nn, "nnn": n_nnn, "x": n, "z": n}
        off = 0
        for b in spec.blocks:
            if b == "nn":
                blk = tiny[off:off + n_nn]
                blk[~keep_nn] = 0.0
                tiny[off:off + n_nn] = blk
            elif b == "nnn":
                blk = tiny[off:off + n_nnn]
                blk[~keep_nnn] = 0.0
                tiny[off:off + n_nnn] = blk
            off += size[b]
        F_zeroed = float(fid_full(tiny))

        # 2) actually DROP the zero RZZ edges → masked circuit, bind surviving θ
        sel = BondSelection(
            nn_edges=[e for e, k in zip(nn_edges, keep_nn, strict=True) if k],
            nnn_edges=[e for e, k in zip(nnn_edges, keep_nnn, strict=True) if k],
            provenance=f"zero_tiny_drop(eps={eps})")
        v = make_masked_variant(f"zt{eps}", f"zero-tiny drop eps={eps}",
                                list(spec.blocks), sel,
                                rx_final=spec.rx_final, rz_final=spec.rz_final)
        qc_c, _ = build_variant(builder, n, lat, v)
        n2q_c, npar_c = cx_and_params(qc_c)
        # Build the surviving-θ vector in the masked layout order.
        theta_c = _project_theta(theta, spec.blocks, keep_nn, keep_nnn,
                                 n_nn, n_nnn, n, spec.rx_final, spec.rz_final)
        _cc, fid_c, _gc, _ = make_cost_fid(qc_c, H, psi, backend=backend)
        F_dropped = float(fid_c(theta_c)) if theta_c.size == npar_c else float("nan")
        match = "OK" if abs(F_dropped - F_zeroed) < 1e-6 else "DIFF"
        print(f"{eps:>5.2f} {F_zeroed:>9.4f} {F_dropped:>10.4f} {n2q_c:>5} "
              f"{n2q_c - n2q_full:>+7} {int(keep_nn.sum()):>8} "
              f"{int(keep_nnn.sum()):>9} {match:>6}")

    print("\nDONE", flush=True)
    return 0


def _project_theta(theta, blocks, keep_nn, keep_nnn, n_nn, n_nnn, nq, rx_final, rz_final):
    """Theta for the masked circuit: keep only surviving nn/nnn entries, x verbatim."""
    size = {"nn": n_nn, "nnn": n_nnn, "x": nq, "z": nq}
    out, off = [], 0
    for b in blocks:
        seg = theta[off:off + size[b]]
        if b == "nn":
            out.append(seg[keep_nn])
        elif b == "nnn":
            out.append(seg[keep_nnn])
        else:
            out.append(seg)
        off += size[b]
    if rx_final:
        out.append(theta[off:off + nq])
        off += nq
    if rz_final:
        out.append(theta[off:off + nq])
    return np.concatenate(out)


if __name__ == "__main__":
    raise SystemExit(main())
