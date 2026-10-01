#!/usr/bin/env python
"""First bounded MPNN warm-start experiment: p=2 frustrated square, learnable regimes.

Trains a UnifiedMPNN to predict θ_opt for the bond-resolved frustrated square HVA
(p=2, full nn+nnn layout), using ALL the converged θ with F>0.90 we already have,
with single-basin + Z₂ canonicalization and symmetry augmentation (the learnability
research said: canonicalize to one basin, treat θ_x as soft, learn where signal is
clean). Validates HONESTLY via leave-one-N-out: the model never sees the test N.

Reuses the full existing MPNN stack:
  - build_unified_bond_resolved_graph(include_nnn=True)  (dataset)
  - UnifiedMPNN + train_unified_mpnn (sign_invariant loss, canonicalize_targets)
  - build_graph_for_model + forward                      (prediction)
  - augment_theta_symmetries, canonicalize_theta         (data expansion)
  - ground_state + make_cost_fid                         (fidelity scoring)

Metric: raw (0-iter) fidelity of the MPNN seed vs the analytic cascade seed, at the
HELD-OUT N, in the clean regimes. If the learned seed matches/beats analytic there,
the GNN learned something generalizable.

Usage:
    .venv/bin/python -u scripts/analysis/vl_vs_hva/train_mpnn_warmstart_p2.py \
        | tee /tmp/mpnn_warmstart_p2.log
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import torch  # noqa: E402

from qmbp_simulation.analysis.warmstart import calibrated_warmstart_theta  # noqa: E402
from qmbp_simulation.framework.study_core import ground_state, make_cost_fid  # noqa: E402
from qmbp_simulation.models import make_lattice  # noqa: E402
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402
from qmbp_simulation.predictors.unified_graph import (  # noqa: E402
    build_graph_for_model,
    build_unified_bond_resolved_graph,
)
from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN, train_unified_mpnn  # noqa: E402
from qmbp_simulation.utils.helpers import augment_theta_symmetries, canonicalize_theta  # noqa: E402

J2, P, TOPO, MODEL = 0.5, 2, "square", "tfim_frustrated"
PAT = f"results/hva_vl_study/bond_ablation/bond_topk_regime_{TOPO}_N*_p{P}_h*.json"
MIN_FID = 0.90


def load_full_nnn_points():
    """All F>0.90 full-nnn p=2 points: list of (N, h, theta, nn, nnn)."""
    by_key = {}
    for f in glob.glob(PAT):
        if "COLLAPSE" in f:
            continue
        m = re.search(r"_N(\d+)_p2_h([\d.]+)\.json", os.path.basename(f))
        if not m:
            continue
        N, h = int(m.group(1)), float(m.group(2))
        d = json.load(open(f))
        for r in d.get("rows", []):
            th, fid = r.get("best_theta_final"), r.get("best_fidelity")
            nn, nnn = r.get("n_nn_bonds"), r.get("n_nnn_bonds")
            if th and fid and fid >= MIN_FID and nn and nnn and (nn + nnn + N) * P == len(th):
                k = (N, round(h, 2))
                # keep the FULL-nnn variant (max nnn) per (N,h)
                if k not in by_key or nnn > by_key[k][4]:
                    by_key[k] = (N, h, np.asarray(th, float), nn, nnn)
    return list(by_key.values())


def to_single_basin(theta, nn, nnn, N):
    """Canonicalize to Z₂ AND the ORDERED basin (sum θ_nn <= 0) so the GNN sees
    one consistent target, not the mean of two basins."""
    th = canonicalize_theta(np.asarray(theta, float))  # Z₂ + period
    # ordered-basin gauge: first-layer nn sum <= 0 (matches analytic seeds)
    if float(np.sum(th[:nn])) > 0:
        per = nn + nnn + N
        for layer in range(P):
            o = layer * per
            th[o:o + nn + nnn] *= -1.0  # flip ZZ blocks (Z₂), θ_x untouched
    return th


def build_dataset(points, *, augment=True, noise_std=0.03, n_noise=2):
    """Graphs for training. Each point → canonical target + symmetry variants."""
    graphs = []
    for N, h, theta, nn, nnn in points:
        lat = make_lattice(TOPO, N, J=1.0, h=h)
        base = to_single_basin(theta, nn, nnn, N)
        thetas = [base]
        if augment:
            thetas += augment_theta_symmetries(
                base, include_z2=True, noise_std=noise_std,
                n_noise_variants=n_noise, seed=hash((N, round(h, 2))) % (2**31))
        for th in thetas:
            g = build_unified_bond_resolved_graph(
                lat, h_value=h, p_layers=P, theta_opt=th,
                include_circuit_nodes=True, include_nnn=True)
            graphs.append(g)
    return graphs


def predict_seed(model, N, h):
    lat = make_lattice(TOPO, N, J=1.0, h=h)
    g = build_graph_for_model(model, lat, h_value=h, p_layers=P, include_nnn=True)
    model.eval()
    with torch.no_grad():
        th = model(g).cpu().numpy().flatten()
    return np.clip(th, -np.pi, np.pi)


def eval_point(model, N, h):
    """raw fidelity of MPNN seed vs analytic calibrated seed at (N,h)."""
    lat, qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(TOPO, N, h, J2, P)
    cost, fid, grad, be = make_cost_fid(qc, H, psi)
    cal = calibrated_warmstart_theta(n_nn, n_nnn, N, P, h, J2=J2)
    th = predict_seed(model, N, h)
    f_mpnn = fid(th) if th.size == qc.num_parameters else float("nan")
    return f_mpnn, fid(cal), th.size, qc.num_parameters


def main():
    points = load_full_nnn_points()
    Ns = sorted(set(p[0] for p in points))
    print(f"[mpnn-p2] {len(points)} full-nnn F>{MIN_FID} points | N={Ns}", flush=True)
    for N, h, th, nn, nnn in sorted(points):
        print(f"   N={N} h={h:.2f} nn={nn} nnn={nnn} len={th.size}", flush=True)

    # Leave-one-N-out: for each test N, train on the others, predict test N.
    print(f"\n[mpnn-p2] LEAVE-ONE-N-OUT validation (raw 0-iter fidelity)", flush=True)
    print(f"{'testN':>5} {'h':>4} | {'mpnn_raw':>8} {'analytic':>8} | winner", flush=True)
    wins = ties = losses = 0
    for testN in Ns:
        train_pts = [p for p in points if p[0] != testN]
        test_pts = [p for p in points if p[0] == testN]
        if len(train_pts) < 3 or not test_pts:
            continue
        ds = build_dataset(train_pts, augment=True)
        model = UnifiedMPNN(node_features=5, hidden_dim=128, n_layers=3,
                            edge_dim=2, gate_readout=True, dropout=0.1)
        train_unified_mpnn(model, ds, n_epochs=1500, lr=1e-3, patience=200,
                           val_fraction=0.15, loss_type="sign_invariant",
                           canonicalize_targets=True, fidelity_loss_weight=0.0)
        for N, h, th, nn, nnn in sorted(test_pts):
            f_mpnn, f_cal, lp, lq = eval_point(model, N, h)
            if not np.isfinite(f_mpnn):
                verdict = f"n/a (len {lp}!={lq})"
            elif f_mpnn > f_cal + 0.02:
                verdict = "MPNN"; wins += 1
            elif f_mpnn < f_cal - 0.02:
                verdict = "analytic"; losses += 1
            else:
                verdict = "tie"; ties += 1
            print(f"{N:>5} {h:>4.2f} | {f_mpnn:>8.4f} {f_cal:>8.4f} | {verdict}", flush=True)

    print(f"\n[mpnn-p2] SUMMARY wins={wins} ties={ties} losses={losses}", flush=True)
    if wins > losses:
        print("  → MPNN seed shows signal: beats analytic on held-out N. Worth integrating.", flush=True)
    elif wins + ties >= losses:
        print("  → MPNN roughly matches analytic (learned the clean regimes). Marginal.", flush=True)
    else:
        print("  → MPNN loses to analytic on held-out N: not enough data to generalize yet.", flush=True)


if __name__ == "__main__":
    main()
