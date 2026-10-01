#!/usr/bin/env python
"""Per-block scaling sweep on the calibrated seed at h=1.0 (0-iter, no VQE).

Isolates which block (nn / nnn / x) the analytic seed can still improve at
h=1.0 for large N, and whether the best scale factor is stable in N (a scalable
rule). Scales ONE block at a time by a factor, keeps the others at calibrated,
measures raw seed fidelity. Pure statevector.

Usage:
    .venv/bin/python -u scripts/analysis/vl_vs_hva/sweep_block_scale_h1.py \
        | tee /tmp/block_scale_h1.log
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from qmbp_simulation.analysis.warmstart import (  # noqa: E402
    _block_slices,
    calibrated_warmstart_theta,
)
from qmbp_simulation.framework.study_core import ground_state, make_cost_fid  # noqa: E402

J2, P, TOPO, H = 0.5, 2, "square", 1.0
FACTORS = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
BLK = {"nn": 0, "nnn": 1, "x": 2}


def scale_block(theta, n_nn, n_nnn, n_q, which, factor):
    out = np.array(theta, float, copy=True)
    for layer in range(P):
        sl = _block_slices(n_nn, n_nnn, n_q, layer)[BLK[which]]
        out[sl] *= factor
    return out


def main():
    print(f"[block-scale h={H}] 0-iter seed fidelity scaling ONE block, calibrated base", flush=True)
    print("block N  | " + " ".join(f"×{f:<4}" for f in FACTORS) + " | best_f best_fid base_fid", flush=True)
    for which in ("nn", "nnn", "x"):
        for N in (10, 12, 14, 18):
            lat, qc, H_op, psi, e0, gap, n_nn, n_nnn = ground_state(TOPO, N, H, J2, P)
            cost, fid, grad, be = make_cost_fid(qc, H_op, psi)
            base = calibrated_warmstart_theta(n_nn, n_nnn, N, P, H, J=1.0, J2=J2)
            base_fid = fid(base)
            fids = [fid(scale_block(base, n_nn, n_nnn, N, which, f)) for f in FACTORS]
            bi = int(np.argmax(fids))
            cells = " ".join(f"{f:5.3f}" for f in fids)
            print(f"{which:<4} {N:>2} | {cells} | ×{FACTORS[bi]:<4} {fids[bi]:.3f} ({base_fid:.3f})",
                  flush=True)
    print("\n[block-scale] Look for a block whose best factor (a) is far from 1.0", flush=True)
    print("(leverage) AND (b) stable across N (scalable). That is the analytic win.", flush=True)


if __name__ == "__main__":
    main()
