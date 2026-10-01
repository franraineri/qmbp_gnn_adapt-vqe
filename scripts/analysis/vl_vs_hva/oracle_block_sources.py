#!/usr/bin/env python
"""Per-block oracle: which warm-start SOURCE best predicts each θ block?

For every converged artifact (N, h) with a high-fidelity full-ref θ, builds the
candidate sources (calibrated, structural, regime, best transferred same-phase
donor) and measures — per block (nn / nnn / x) — which source lands closest to
the OPTIMIZED θ (sign/wrap-invariant). Aggregates winners by phase so we can see
whether a hybrid "best-of-each-tool" seed is worth building, and with what policy.

Pure analysis: no backend, no optimization. Reuses block_source_distances and
the existing source builders — nothing re-implemented.

Usage:
    .venv/bin/python -u scripts/analysis/vl_vs_hva/oracle_block_sources.py \
        | tee /tmp/oracle_block_sources.log
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from qmbp_simulation.analysis.warmstart import (  # noqa: E402
    ORDERED_H_MAX,
    PARAMAGNETIC_H_MIN,
    block_source_distances,
    calibrated_warmstart_theta,
    select_regime_seed,
    structural_warmstart_theta,
    transfer_theta,
)
from qmbp_simulation.models import make_lattice  # noqa: E402
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402

J2 = 0.5
P = 2
TOPO = "square"
MIN_FID = 0.85
PAT = f"results/hva_vl_study/bond_ablation/bond_topk_regime_{TOPO}_N*_p{P}_h*.json"


def _phase(h):
    return "ordered" if h < ORDERED_H_MAX else (
        "near_hc" if h < PARAMAGNETIC_H_MIN else "paramag")


def _load_opt(path):
    """Return (N, h, theta_opt, fid, n_nn, n_nnn) for the best full-ref row, or None."""
    try:
        d = json.load(open(path))
    except Exception:
        return None
    m = re.search(r"_N(\d+)_p\d+_h([\d.]+)\.json", os.path.basename(path))
    if not m:
        return None
    n, h = int(m.group(1)), float(m.group(2))
    best = None  # (fid, theta, n_nn, n_nnn)
    for row in d.get("rows", []):
        th = row.get("best_theta_final")
        fid = row.get("best_fidelity")
        if not th or fid is None or fid < MIN_FID:
            continue
        # full-ref row: n_nnn_bonds equals the full lattice nnn count
        n_nn = row.get("n_nn_bonds")
        n_nnn = row.get("n_nnn_bonds")
        if n_nn is None or n_nnn is None:
            continue
        if (n_nn + n_nnn + n) * P != len(th):
            continue  # not the standard full layout (masked variant) — skip
        if best is None or fid > best[0]:
            best = (fid, np.asarray(th, float), n_nn, n_nnn)
    if best is None:
        return None
    return n, h, best[1], best[0], best[2], best[3]


def _best_donor_seed(n, h, n_nn, n_nnn, target_nnn_edges, fill):
    """Best same-phase transferred donor (lowest total block-distance proxy):
    we just pick the highest-fidelity same-phase donor from the corpus."""
    tgt_phase = _phase(h)
    best = None  # (fid, seed)
    for path in glob.glob(PAT):
        if "COLLAPSE" in path:
            continue
        info = _load_opt(path)
        if info is None:
            continue
        dn, dh, dtheta, dfid, dn_nn, dn_nnn = info
        if (dn, round(dh, 2)) == (n, round(h, 2)):
            continue  # self
        if _phase(dh) != tgt_phase:
            continue
        blocks = ("nn", "nnn", "x") if (dn == n and dn_nnn == n_nnn) else ("nn", "x")
        lat_d = make_lattice(TOPO, dn, J=1.0, h=dh)
        d_nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat_d)
        seed = transfer_theta(
            dtheta, donor_n_nn=dn_nn, donor_n_nnn=dn_nnn, donor_p=P,
            target_n_nn=n_nn, target_n_nnn=n_nnn, target_p=P, n_qubits=n,
            donor_nnn_edges=d_nnn_edges, target_nnn_edges=target_nnn_edges,
            fill_theta=fill, canonicalize=True, donor_n_qubits=dn,
            donor_blocks=blocks)
        if seed is not None and seed.size == (n_nn + n_nnn + n) * P:
            if best is None or dfid > best[0]:
                best = (dfid, seed)
    return best[1] if best else None


def main():
    print(f"[oracle] topo={TOPO} p={P} J2={J2} min_fid={MIN_FID}", flush=True)
    # phase -> block -> source -> win count
    phase_winners = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    print(f"\n{'N':>3} {'h':>4} {'phase':>8} {'fid':>6} | per-layer winners "
          f"(nn / nnn / x)", flush=True)
    for path in sorted(glob.glob(PAT)):
        if "COLLAPSE" in path:
            continue
        info = _load_opt(path)
        if info is None:
            continue
        n, h, theta_opt, fid, n_nn, n_nnn = info
        lat = make_lattice(TOPO, n, J=1.0, h=h)
        target_nnn_edges = HamiltonianBuilder._generate_nnn_edges(lat)
        fill, _ = select_regime_seed(n_nn, n_nnn, n, P, h, J=1.0, J2=J2)

        sources = {
            "calibrated": calibrated_warmstart_theta(n_nn, n_nnn, n, P, h, J2=J2),
            "structural": structural_warmstart_theta(n_nn, n_nnn, n, P, h, J2=J2),
            "regime": fill,
        }
        donor = _best_donor_seed(n, h, n_nn, n_nnn, target_nnn_edges, fill)
        if donor is not None:
            sources["donor"] = donor

        rep = block_source_distances(theta_opt, sources, n_nn=n_nn, n_nnn=n_nnn,
                                     n_qubits=n, p_layers=P)
        ph = _phase(h)
        # aggregate
        for b in ("nn", "nnn", "x"):
            for lbl, c in rep["winners"][b].items():
                phase_winners[ph][b][lbl] += c
        # per-point line (winner per block, layer 0 for brevity)
        w = {b: rep["per_block"].get((0, b), {}).get("best", "-")
             for b in ("nn", "nnn", "x")}
        print(f"{n:>3} {h:>4.2f} {ph:>8} {fid:>6.3f} | "
              f"{w['nn']:>10} / {w['nnn']:>10} / {w['x']:>10}", flush=True)

    print("\n[oracle] AGGREGATED WINNERS BY PHASE (block -> source: count)", flush=True)
    policy_by_phase = {}
    for ph in ("ordered", "near_hc", "paramag"):
        if ph not in phase_winners:
            continue
        print(f"\n  == {ph} ==", flush=True)
        policy = {}
        for b in ("nn", "nnn", "x"):
            counts = phase_winners[ph][b]
            if not counts:
                continue
            ranked = sorted(counts.items(), key=lambda kv: -kv[1])
            policy[b] = ranked[0][0]
            tallies = "  ".join(f"{k}={v}" for k, v in ranked)
            print(f"    {b:>4}: {tallies}", flush=True)
        policy_by_phase[ph] = policy

    print("\n[oracle] SUGGESTED PER-BLOCK POLICY (per phase)", flush=True)
    for ph, pol in policy_by_phase.items():
        print(f"    {ph:>8}: {pol}", flush=True)

    # Is the hybrid non-trivial? (i.e. different sources win different blocks)
    print("\n[oracle] VERDICT", flush=True)
    nontrivial = any(len(set(pol.values())) > 1 for pol in policy_by_phase.values())
    if nontrivial:
        print("  → Different sources win different blocks in at least one phase: "
              "a per-block HYBRID seed is worth building+validating.", flush=True)
    else:
        print("  → One source dominates all blocks in every phase: a hybrid adds "
              "nothing; the single best candidate is already optimal.", flush=True)


if __name__ == "__main__":
    main()
