#!/usr/bin/env python
"""Analytic seed sweep at h=1.0: find the best nn-shrink per base, scalable in N.

Pure 0-iter seed fidelity (statevector, NO VQE). The angle-trends report found
the optimum uses smaller θ_nn than the analytic seed at h=1.0 (|opt|/|seed|≈0.3-0.5)
and that an nn-shrink lifts the SEED fidelity 2-3× — but only measured on the
second-order base. Here we sweep the nn-shrink factor on BOTH analytic bases
(second_order and calibrated/structural) across N, to see if a single scalable
factor improves the raw seed the cascade would start from.

Reuses second_order_warmstart_theta, calibrated_warmstart_theta, _block_slices.

Usage:
    .venv/bin/python -u scripts/analysis/vl_vs_hva/sweep_nn_shrink_h1.py \
        | tee /tmp/nn_shrink_h1.log
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
    second_order_warmstart_theta,
)
from qmbp_simulation.framework.study_core import ground_state, make_cost_fid  # noqa: E402

J2, P, TOPO = 0.5, 2, "square"
H = 1.0
SHRINKS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]


def shrink_nn(theta, n_nn, n_nnn, n_q, factor):
    out = np.array(theta, float, copy=True)
    for layer in range(P):
        s_nn, _s_nnn, _s_x = _block_slices(n_nn, n_nnn, n_q, layer)
        out[s_nn] *= factor
    return out


def main():
    print(f"[nn-shrink h={H}] 0-iter seed fidelity vs nn-shrink factor, per N", flush=True)
    print("base       N  | " + " ".join(f"s={s:<4}" for s in SHRINKS) + " | best_s best_fid", flush=True)
    for base_name in ("second_order", "calibrated"):
        for N in (10, 12, 14, 18):
            lat, qc, H_op, psi, e0, gap, n_nn, n_nnn = ground_state(TOPO, N, H, J2, P)
            cost, fid, grad, be = make_cost_fid(qc, H_op, psi)
            if base_name == "second_order":
                base = second_order_warmstart_theta(n_nn, n_nnn, N, P, H, J=1.0, J2=J2)
            else:
                base = calibrated_warmstart_theta(n_nn, n_nnn, N, P, H, J=1.0, J2=J2)
            fids = []
            for s in SHRINKS:
                th = shrink_nn(base, n_nn, n_nnn, N, s)
                fids.append(fid(th))
            best_i = int(np.argmax(fids))
            cells = " ".join(f"{f:5.3f}" for f in fids)
            print(f"{base_name:<10} {N:>2} | {cells} | s={SHRINKS[best_i]:<4} {fids[best_i]:.3f}",
                  flush=True)

    print("\n[nn-shrink] READING: if a single best_s is consistent across N for a", flush=True)
    print("base, that is a scalable analytic rule — raise the raw seed for free,", flush=True)
    print("so the micro-descent starts closer and needs fewer iters at large N.", flush=True)


if __name__ == "__main__":
    main()
