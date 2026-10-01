#!/usr/bin/env python
"""Zero-iteration verification across N: raw cascade seed fidelity + per-block error.

Answers two things, measuring ONLY fidelity (no VQE/descent iterations at all):

  1. How does the default warm-start (prepare_warmstart, micro_descent=0 → the RAW
     chosen seed) behave as N grows? → raw fidelity vs the corpus optimum fidelity.
  2. Which θ block (nn / nnn / x) is well predicted and which is not? → sign/wrap-
     invariant per-block distance of the raw seed to the corpus θ_opt
     (block_source_distances with the seed as the single source).

Pure fidelity, no ΔE. Reuses ground_state (for qc/psi), prepare_warmstart (the
default gate), the corpus θ_opt loader, and block_source_distances — nothing new.

Usage:
    .venv/bin/python -u scripts/analysis/vl_vs_hva/verify_zero_iter_blocks.py \
        | tee /tmp/zero_iter_blocks.log
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from qmbp_simulation.analysis.warmstart import block_source_distances  # noqa: E402
from qmbp_simulation.framework.study_core import ground_state, make_cost_fid, prepare_warmstart  # noqa: E402

J2, P, TOPO, MODEL = 0.5, 2, "square", "tfim_frustrated"
PAT = f"results/hva_vl_study/bond_ablation/bond_topk_regime_{TOPO}_N*_p{P}_h*.json"


def _load_opt(n, h):
    """Best full-ref θ_opt (+its fidelity, n_nn, n_nnn) for (n, h) from the corpus."""
    for f in glob.glob(PAT):
        if "COLLAPSE" in f:
            continue
        m = re.search(r"_N(\d+)_p\d+_h([\d.]+)\.json", os.path.basename(f))
        if not m or int(m.group(1)) != n or abs(float(m.group(2)) - h) > 1e-6:
            continue
        d = json.load(open(f))
        best = None
        for r in d.get("rows", []):
            th, fid = r.get("best_theta_final"), r.get("best_fidelity")
            nn, nnn = r.get("n_nn_bonds"), r.get("n_nnn_bonds")
            if th and fid and nn is not None and nnn is not None and (nn + nnn + n) * P == len(th):
                if best is None or fid > best[0]:
                    best = (fid, np.asarray(th, float), nn, nnn)
        return best
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro-descent", type=int, default=0,
                    help="micro-descent iters applied to the chosen seed "
                         "(0 = raw seed, no optimization; e.g. 48 = short VQE)")
    args = ap.parse_args()
    md = int(args.micro_descent)
    tag = f"micro_descent={md}" if md > 0 else "0 VQE iters (raw seed)"

    # grandes con θ_opt disponible, uno por fase donde se pueda
    points = [(10, 0.5), (12, 0.5), (14, 0.5), (18, 0.5),
              (10, 0.3), (12, 0.3), (10, 1.3), (12, 1.3)]
    print(f"[zero-iter] topo={TOPO} model={MODEL} p={P} J2={J2}  ({tag})", flush=True)
    print(f"{'N':>3} {'h':>4} | {'seed_fid':>8} {'opt_fid':>7} {'gap':>7} | "
          f"per-block |seed-opt| sign-inv (nn / nnn / x) | prov", flush=True)
    for n, h in points:
        opt = _load_opt(n, h)
        lat, qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(TOPO, n, h, J2, P)
        cost, fid, grad, _ = make_cost_fid(qc, H, psi)
        # seed after `md` micro-descent iters (md=0 → raw seed, no optimization)
        ws = prepare_warmstart(qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=n,
                               p_layers=P, h=h, J=1.0, J2=J2, topology=TOPO,
                               model=MODEL, micro_descent=md, cost=cost, fid=fid, grad=grad)
        raw_fid = fid(ws["seed"])
        if opt is not None:
            opt_fid, theta_opt, _onn, _onnn = opt
            rep = block_source_distances(theta_opt, {"seed": ws["seed"]},
                                         n_nn=n_nn, n_nnn=n_nnn, n_qubits=n, p_layers=P)
            # mean per-block distance across layers (lower = better predicted)
            def _blk_mean(b):
                ds = [rep["per_block"][(l, b)]["dist"]["seed"]
                      for l in range(P) if (l, b) in rep["per_block"]]
                return float(np.mean(ds)) if ds else float("nan")
            dnn, dnnn, dx = _blk_mean("nn"), _blk_mean("nnn"), _blk_mean("x")
            print(f"{n:>3} {h:>4.2f} | {raw_fid:>7.4f} {opt_fid:>7.4f} "
                  f"{opt_fid - raw_fid:>7.4f} | {dnn:>8.4f} / {dnnn:>8.4f} / {dx:>8.4f} | "
                  f"{ws['provenance']}", flush=True)
        else:
            print(f"{n:>3} {h:>4.2f} | {raw_fid:>7.4f} {'--':>7} {'--':>7} | "
                  f"(no corpus θ_opt) | {ws['provenance']}", flush=True)

    print("\n[zero-iter] READING", flush=True)
    print(f"  seed_fid = fidelity of the chosen seed after {md} micro-descent iters.",
          flush=True)
    print("  gap      = opt_fid - seed_fid (corpus optimum minus this seed);", flush=True)
    print("             <=0 means the micro-descent matched/beat the corpus θ_opt.", flush=True)
    print("  per-block distance: lower = that block's angles sit closer to θ_opt.", flush=True)
    print("  Expect θ_x to stay the worst block even after descent if the gap is", flush=True)
    print("  in the soft transverse-field direction.", flush=True)


if __name__ == "__main__":
    main()
