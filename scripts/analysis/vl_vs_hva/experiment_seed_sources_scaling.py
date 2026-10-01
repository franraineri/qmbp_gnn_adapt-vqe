#!/usr/bin/env python
"""Which raw-seed SOURCE scales best to large N? (fidelity @ 0 and @ 10 iters).

Goal: improve the warm-start approximation at large N WITHOUT spending a big VQE
budget. We compare several seed sources for the frustrated square TFIM transition
(h=0.5), each measured at 0 micro-descent iters (raw seed) AND after 10 iters.

Candidates (all respect fidelity non-separability — each is a single coherent θ):
  default : the current cascade choice (uniform analytic seed)
  A       : a converged donor from a SMALLER N, transferred raw (cross-N, full)
  B       : default with θ_x replaced by the donor's N-invariant θ_x mean
  C       : default with θ_nn shrunk ×0.2 (angle-trends E2: scales with N, gapped)
  D       : default with θ_nn/θ_nnn copied from the donor ONLY on the valuable
            (top-|θ|) bonds — the bond-selection report's "select from a converged
            θ, on the valuable bonds, not all" recommendation.

Reuses: ground_state, make_cost_fid, _lbfgsb, transfer_theta, select_regime_seed,
second_order_nn_shrink_theta, bond_weights_from_theta, block_slices. No new module
code yet — if a candidate wins it gets promoted to warmstart.py.

Usage:
    .venv/bin/python -u scripts/analysis/vl_vs_hva/experiment_seed_sources_scaling.py \
        | tee /tmp/seed_sources_scaling.log
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from qmbp_simulation.analysis.warmstart import (  # noqa: E402
    _block_slices,
    second_order_nn_shrink_theta,
    select_regime_seed,
    transfer_theta,
)
from qmbp_simulation.circuits.bond_mask import bond_weights_from_theta  # noqa: E402
from qmbp_simulation.framework.study_core import ground_state, make_cost_fid, prepare_warmstart  # noqa: E402
from qmbp_simulation.framework.study_runner import _lbfgsb  # noqa: E402
from qmbp_simulation.models import make_lattice  # noqa: E402
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402

J2, P, TOPO, MODEL = 0.5, 2, "square", "tfim_frustrated"
PAT = f"results/hva_vl_study/bond_ablation/bond_topk_regime_{TOPO}_N*_p{P}_h*.json"
TOPK_FRAC = 0.4   # candidate D: fraction of bonds considered "valuable"


def _load_donor(n, h):
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


def _candidate_D(base, donor_full, n_nn, n_nnn, n_qubits, topk_frac):
    """base seed with nn/nnn copied from the (same-layout) donor ONLY on the
    valuable top-|θ| bonds; the rest keeps base. donor_full must be target-layout."""
    out = np.array(base, float, copy=True)
    w_nn, w_nnn = bond_weights_from_theta(donor_full, n_nn, n_nnn, n_qubits, P)
    k_nn = max(1, int(round(topk_frac * n_nn)))
    k_nnn = max(1, int(round(topk_frac * n_nnn)))
    top_nn = set(np.argsort(w_nn)[::-1][:k_nn].tolist())
    top_nnn = set(np.argsort(w_nnn)[::-1][:k_nnn].tolist())
    for layer in range(P):
        s_nn, s_nnn, _s_x = _block_slices(n_nn, n_nnn, n_qubits, layer)
        for j in top_nn:
            out[s_nn.start + j] = donor_full[s_nn.start + j]
        for j in top_nnn:
            out[s_nnn.start + j] = donor_full[s_nnn.start + j]
    return out


def run_point(n_target, h, donor_n):
    lat, qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(TOPO, n_target, h, J2, P)
    cost, fid, grad, _ = make_cost_fid(qc, H, psi)
    tgt_nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)
    fill, _ = select_regime_seed(n_nn, n_nnn, n_target, P, h, J=1.0, J2=J2)

    # default (cascade raw seed)
    ws = prepare_warmstart(qc, H, psi, n_nn=n_nn, n_nnn=n_nnn, n_qubits=n_target,
                           p_layers=P, h=h, J=1.0, J2=J2, topology=TOPO, model=MODEL,
                           micro_descent=0, cost=cost, fid=fid, grad=grad)
    default_seed = np.asarray(ws["seed"], float)

    # donor transferred to target layout (cross-N full; θ_x = donor mean)
    dinfo = _load_donor(donor_n, h)
    donor_full = None
    if dinfo is not None:
        _dfid, dtheta, dnn, dnnn = dinfo
        lat_d = make_lattice(TOPO, donor_n, J=1.0, h=h)
        d_nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat_d)
        donor_full = transfer_theta(
            dtheta, donor_n_nn=dnn, donor_n_nnn=dnnn, donor_p=P,
            target_n_nn=n_nn, target_n_nnn=n_nnn, target_p=P, n_qubits=n_target,
            donor_nnn_edges=d_nnn_edges, target_nnn_edges=tgt_nnn_edges,
            fill_theta=fill, canonicalize=True, donor_n_qubits=donor_n,
            donor_blocks=("nn", "nnn", "x"))

    seeds = {"default": default_seed}
    if donor_full is not None and donor_full.size == default_seed.size:
        seeds["A_donor_raw"] = donor_full
        # B: default with θ_x from donor
        b = np.array(default_seed, float, copy=True)
        for layer in range(P):
            _s_nn, _s_nnn, s_x = _block_slices(n_nn, n_nnn, n_target, layer)
            b[s_x] = donor_full[s_x]
        seeds["B_donor_x"] = b
        # D: default with valuable-bond nn/nnn from donor
        seeds["D_valuable_bonds"] = _candidate_D(
            default_seed, donor_full, n_nn, n_nnn, n_target, TOPK_FRAC)
    # C: nn-shrink 0.2 (donor-free)
    seeds["C_nn_shrink"] = second_order_nn_shrink_theta(
        n_nn, n_nnn, n_target, P, h, J=1.0, J2=J2, nn_extra_shrink=0.2)

    out = {}
    for name, s in seeds.items():
        f0 = fid(s)
        x10, _e, _nit = _lbfgsb(cost, np.asarray(s, float), maxiter=10, grad=grad)
        out[name] = (f0, fid(x10))
    return n_target, donor_n, out


def main():
    # target large-N, donor from a smaller high-fidelity N
    cases = [(12, 10), (14, 10), (18, 10), (18, 12)]
    h = 0.5
    names = ["default", "A_donor_raw", "B_donor_x", "C_nn_shrink", "D_valuable_bonds"]
    print(f"[seed-sources] transition h={h} topk_frac={TOPK_FRAC} "
          f"(fid@0 / fid@10 micro-descent)", flush=True)
    hdr = f"{'tgtN':>5} {'don':>4} | " + " | ".join(f"{nm:>16}" for nm in names)
    print(hdr, flush=True)
    for n_target, donor_n in cases:
        _n, _d, out = run_point(n_target, h, donor_n)
        cells = []
        for nm in names:
            if nm in out:
                f0, f10 = out[nm]
                cells.append(f"{f0:.3f}/{f10:.3f}".rjust(16))
            else:
                cells.append("--".rjust(16))
        print(f"{n_target:>5} {donor_n:>4} | " + " | ".join(cells), flush=True)

    print("\n[seed-sources] READING", flush=True)
    print("  Each cell = fid@0 / fid@10. Looking for a source whose fid@0 (and", flush=True)
    print("  fid@10) DEGRADES LESS with N than 'default' → that is the scalable", flush=True)
    print("  warm-start. D tests the report's advice: valuable-bond θ from a", flush=True)
    print("  converged donor, not all bonds, not geometry.", flush=True)


if __name__ == "__main__":
    main()
