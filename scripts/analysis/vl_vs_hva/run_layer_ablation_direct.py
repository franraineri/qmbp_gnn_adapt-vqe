#!/usr/bin/env python
"""Direct layer-ablation test: drop a layer, reuse the converged θ, NO re-opt.

The θ-pattern analysis of the N18 p3 full (F=0.935) showed the state lives in the
x-layers while whole nnn layers (nnn#1, nnn#3) sit near identity, and that
per-|θ| pruning collapses fidelity. This probes the complementary structural
lever: remove a WHOLE block (layer) and keep the converged angles on every
surviving block, WITHOUT any optimization. Fidelity of that transplanted θ
measures how much the dropped layer actually contributed — isolating the
structural effect from re-optimization masking it.

Mapping: the full θ is a concatenation of block slices in ``blocks`` order
(+ rx_final). Dropping block instance k removes exactly its slice; every other
block keeps its angles verbatim (same edges, same order, full edge set). For the
``zero_tiny`` strategy the structure is unchanged and only |θ|<ε angles are set
to 0 (the trivial parameter-tying baseline).

Strategies (per the pattern analysis):
  drop_nnn1, drop_nnn3, drop_nnn1_nnn3, drop_nn1, zero_tiny(ε=0.05)

Pure evaluation — builds circuits and computes exact fidelity; runs NO optimizer.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/run_layer_ablation_direct.py \
        --spec results/hva_vl_study/ansatz_specs/p3_half_nn_rx_square_N18_h0.50.spec.json
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
from qmbp_simulation.circuits.hva_variants import AnsatzVariant, build_variant  # noqa: E402
from qmbp_simulation.framework.study_core import cx_and_params, ground_state, make_cost_fid  # noqa: E402


def _block_layout(blocks, n_nn, n_nnn, n_qubits, rx_final, rz_final):
    """List of (label, size) for every block instance + trailing, in θ order."""
    size = {"nn": n_nn, "nnn": n_nnn, "x": n_qubits, "z": n_qubits}
    counts: dict[str, int] = {}
    layout = []
    for b in blocks:
        counts[b] = counts.get(b, 0) + 1
        layout.append((f"{b}#{counts[b]}", b, size[b]))
    if rx_final:
        layout.append(("rx_final", "x", n_qubits))
    if rz_final:
        layout.append(("rz_final", "z", n_qubits))
    return layout


def _theta_slices(theta, layout):
    """Map label → θ slice (verbatim), consuming the vector in order."""
    out = {}
    off = 0
    for lab, _btype, sz in layout:
        out[lab] = np.asarray(theta[off:off + sz], float)
        off += sz
    return out


def _drop_blocks(spec, drop_labels):
    """Return (new_blocks, new_rx_final, new_theta) with the named layers removed.

    ``drop_labels`` are block-instance labels like 'nnn#1' or 'nn#1'. Trailing
    rx_final cannot be dropped here (kept). The surviving blocks keep their
    converged angles verbatim; the result is the concatenation in original order.
    """
    theta = np.asarray(spec.theta, float)
    layout = _block_layout(spec.blocks, len(spec.nn_edges), len(spec.nnn_edges),
                           spec.n_qubits, spec.rx_final, spec.rz_final)
    sl = _theta_slices(theta, layout)

    new_blocks = []
    kept_theta = []
    # Walk only the real blocks (not trailing) to rebuild the block list.
    counts: dict[str, int] = {}
    for b in spec.blocks:
        counts[b] = counts.get(b, 0) + 1
        lab = f"{b}#{counts[b]}"
        if lab in drop_labels:
            continue
        new_blocks.append(b)
        kept_theta.append(sl[lab])
    if spec.rx_final and "rx_final" not in drop_labels:
        kept_theta.append(sl["rx_final"])
    return new_blocks, spec.rx_final, np.concatenate(kept_theta)


def _eval_fid(builder, n, lat, blocks, rx_final, theta, H, psi, backend):
    """Build the configurable circuit for ``blocks`` and return (fid, n_2q, npar)."""
    v = AnsatzVariant("ablation", "direct layer ablation", blocks=list(blocks),
                      rx_final=rx_final)
    qc, _ = build_variant(builder, n, lat, v)
    n_2q, npar = cx_and_params(qc)
    if theta.size != npar:
        return None, n_2q, npar
    _cost, fid, _grad, _ = make_cost_fid(qc, H, psi, backend=backend)
    return float(fid(theta)), n_2q, npar


def main(argv=None) -> int:
    from qmbp_simulation.execution import NoiselessBackend

    p = argparse.ArgumentParser(description="Direct layer ablation (no re-optimization)")
    p.add_argument("--spec", required=True)
    p.add_argument("--topology", default="square")
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--tiny-eps", type=float, default=0.05)
    args = p.parse_args(argv)

    spec = AnsatzSpec.load(args.spec)
    n = spec.n_qubits
    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(args.topology, n, args.h, args.j2, 1)
    builder = HVACircuitBuilder()
    backend = NoiselessBackend()
    theta_full = np.asarray(spec.theta, float)

    # Reference: the full θ on the full circuit (sanity — should match spec.fidelity).
    v_full = AnsatzVariant("full", "full", blocks=list(spec.blocks), rx_final=spec.rx_final)
    qc_full, _ = build_variant(builder, n, lat, v_full)
    n2q_full, npar_full = cx_and_params(qc_full)
    _c, fid_full, _g, _ = make_cost_fid(qc_full, H, psi, backend=backend)
    F_full = float(fid_full(theta_full))

    print(f"=== Direct layer ablation (NO re-opt) — {Path(args.spec).stem} ===")
    print(f"N={n} blocks={spec.blocks} rx_final={spec.rx_final}")
    print(f"reference FULL: F={F_full:.4f} (spec recorded {spec.fidelity:.4f}) "
          f"2q={n2q_full} npar={npar_full}\n")
    print(f"{'strategy':16s} {'F_direct':>9} {'ΔF':>8} {'2q':>5} {'Δ2q':>6} {'npar':>5}")

    strategies = {
        "drop_nnn1": {"nnn#1"},
        "drop_nnn3": {"nnn#3"},
        "drop_nnn1_nnn3": {"nnn#1", "nnn#3"},
        "drop_nn1": {"nn#1"},
    }
    for name, drop in strategies.items():
        blocks, rxf, theta = _drop_blocks(spec, drop)
        F, n2q, npar = _eval_fid(builder, n, lat, blocks, rxf, theta, H, psi, backend)
        if F is None:
            print(f"{name:16s} {'len-mismatch':>9} npar={npar}")
            continue
        print(f"{name:16s} {F:>9.4f} {F - F_full:>+8.4f} {n2q:>5} {n2q - n2q_full:>+6} {npar:>5}")

    # zero_tiny: same structure, set |θ|<eps to 0 (trivial tying baseline).
    tiny = theta_full.copy()
    n_zeroed = int(np.sum(np.abs(tiny) < args.tiny_eps))
    tiny[np.abs(tiny) < args.tiny_eps] = 0.0
    F_tiny = float(fid_full(tiny))
    tiny_label = f"zero_tiny(<{args.tiny_eps:.2f})"
    print(f"{tiny_label:16s} {F_tiny:>9.4f} "
          f"{F_tiny - F_full:>+8.4f} {n2q_full:>5} {0:>+6} {npar_full - n_zeroed:>5}  "
          f"({n_zeroed} angles zeroed)")

    print("\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
