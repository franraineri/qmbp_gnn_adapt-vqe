#!/usr/bin/env python
"""Isolate structure-vs-warmstart: optimize each pruned ansatz from 4 seeds.

Open question after the N14/18 queue: when a pruned/top-k ansatz only reached
F≈0.6–0.85, was the STRUCTURE incapable, or did the warm-start just fail to find
its good basin? This resolves it by optimizing every pruned spec from FOUR
independent seeds and reporting the best F reached and WHICH seed produced it:

  - analytic  : variant_warmstart_theta for the masked layout (second-order seed)
  - transfer  : the converged FULL θ at this N transferred onto the pruned
                layout (transfer_theta_for_blocks — the same-N bond-mask path)
  - random×2  : two in-box random restarts (basin coverage)

Interpretation:
  * If best-of-4 >> the queue's F → it was the WARM-START (structure is capable).
  * If best-of-4 ≈ the queue's low F across ALL seeds → the STRUCTURE is the
    limit (the pruned ansatz genuinely cannot represent the state at this N).

Each pruned spec is rebuilt EXACTLY via AnsatzSpec.build() (masked by its stored
edges). The full θ comes from the matching full spec at the same N/variant.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/run_seed_ablation_study.py \
        --n 14 --maxiter 800 --seed0 90000
    .venv/bin/python scripts/analysis/vl_vs_hva/run_seed_ablation_study.py \
        --n 18 --maxiter 800
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import STUDY_ROOT, save_json  # noqa: E402

from qmbp_simulation.analysis.warmstart import (  # noqa: E402
    transfer_theta_for_blocks,
    variant_warmstart_theta,
)
from qmbp_simulation.circuits.ansatz_spec import AnsatzSpec  # noqa: E402
from qmbp_simulation.framework.study_core import (  # noqa: E402
    cx_and_params,
    ground_state,
    make_cost_fid,
)
from qmbp_simulation.framework.study_runner import _lbfgsb  # noqa: E402

SPECS_DIR = STUDY_ROOT / "ansatz_specs"
# Full variant per p-family, used to source the transfer seed.
FULL_VARIANT = {14: "p2_half_nn_rx", 18: "p3_half_nn_rx"}


def _pruned_specs(n, topology, h):
    """All masked specs (prune/topk) for this N — rebuilt exactly from edges."""
    out = []
    for f in sorted(SPECS_DIR.glob(f"*_{topology}_N{n}_h{h:.2f}.spec.json")):
        if "prune" not in f.name and "topk" not in f.name:
            continue
        try:
            out.append(AnsatzSpec.load(str(f)))
        except Exception:
            pass
    return out


def _full_theta(n, topology, h):
    """Converged FULL θ + its edges for this N (source of the transfer seed)."""
    fv = FULL_VARIANT.get(n)
    f = SPECS_DIR / f"{fv}_{topology}_N{n}_h{h:.2f}.spec.json"
    if not f.exists():
        return None
    s = AnsatzSpec.load(str(f))
    return s


def _seeds_for(spec, full_spec, n, h, j2, rng):
    """Build the 4 named seeds for a pruned spec. Returns list of (theta, name)."""
    npar = len(spec.theta) if spec.theta else None
    seeds = []

    # analytic — second-order seed for THIS masked layout
    analytic = variant_warmstart_theta(
        spec.blocks, len(spec.nn_edges), len(spec.nnn_edges), n, h,
        J=1.0, J2=j2, rx_final=spec.rx_final, rz_final=spec.rz_final)
    if npar and len(analytic) == npar:
        seeds.append((np.asarray(analytic, float), "analytic"))

    # transfer — full θ at this N transferred onto the pruned nn/nnn selection.
    # The prune drops nn bonds too (nn_full > nn_pruned at the same N), so pass
    # both edge lists to activate per-bond nn/nnn matching (not verbatim copy).
    if full_spec is not None and full_spec.theta:
        transferred = transfer_theta_for_blocks(
            np.asarray(full_spec.theta, float), list(spec.blocks),
            donor_nnn_edges=full_spec.nnn_edges, target_nnn_edges=spec.nnn_edges,
            n_nn=len(spec.nn_edges), n_qubits=n,
            rx_final=spec.rx_final, rz_final=spec.rz_final,
            donor_n_nn=len(full_spec.nn_edges), donor_n_qubits=n,
            donor_nn_edges=full_spec.nn_edges, target_nn_edges=spec.nn_edges)
        if transferred is not None and (not npar or transferred.size == npar):
            seeds.append((transferred, "transfer<full>"))

    # 2 random in-box
    for k in range(2):
        seeds.append((rng.uniform(-np.pi, np.pi, npar), f"random{k + 1}"))
    return seeds


def main(argv=None) -> int:
    from qmbp_simulation.execution import NoiselessBackend

    p = argparse.ArgumentParser(description="Structure-vs-warmstart seed ablation")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--topology", default="square")
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--maxiter", type=int, default=800)
    p.add_argument("--seed0", type=int, default=90000)
    args = p.parse_args(argv)

    specs = _pruned_specs(args.n, args.topology, args.h)
    if not specs:
        print(f"No pruned specs for N={args.n}")
        return 1
    full_spec = _full_theta(args.n, args.topology, args.h)
    lat, _qc, H, psi, e0, gap, _n_nn, _n_nnn = ground_state(
        args.topology, args.n, args.h, args.j2, 1)
    backend = NoiselessBackend()
    rng = np.random.default_rng(args.seed0)

    print(f"=== Seed ablation (structure vs warm-start) N={args.n} h={args.h} ===")
    print(f"full donor: {FULL_VARIANT.get(args.n)} "
          f"(F={full_spec.fidelity:.4f})" if full_spec else "full donor: NONE")
    print(f"{'pruned spec':40s} {'queue_F':>7} {'best_F':>7} {'best_seed':>16} "
          f"{'per-seed F (analytic/transfer/rnd1/rnd2)':>42}")

    rows = []
    for spec in specs:
        qc, _ = spec.build(lattice=lat)
        n_2q, npar = cx_and_params(qc)
        cost, fid, grad, _ = make_cost_fid(qc, H, psi, backend=backend)
        seeds = _seeds_for(spec, full_spec, args.n, args.h, args.j2, rng)

        per_seed = {}
        best_f, best_name = -1.0, None
        t0 = time.time()
        for theta0, name in seeds:
            if theta0 is None or len(theta0) != npar:
                per_seed[name] = None
                continue
            x, _e, _nit = _lbfgsb(cost, np.clip(theta0, -np.pi, np.pi),
                                  maxiter=args.maxiter, grad=grad)
            f = float(fid(x))
            per_seed[name] = f
            if f > best_f:
                best_f, best_name = f, name

        tag = spec.name[:40]
        queue_F = spec.fidelity or 0.0
        ps = per_seed
        detail = (f"a={_fmt(ps.get('analytic'))} t={_fmt(ps.get('transfer<full>'))} "
                  f"r1={_fmt(ps.get('random1'))} r2={_fmt(ps.get('random2'))}")
        print(f"{tag:40s} {queue_F:>7.4f} {best_f:>7.4f} {str(best_name):>16} {detail:>42}")
        rows.append({
            "variant": spec.name, "n_qubits": args.n, "n_2q": n_2q, "n_params": npar,
            "queue_fidelity": queue_F, "best_fidelity": best_f, "best_seed": best_name,
            "per_seed_fidelity": per_seed, "seconds": round(time.time() - t0, 1),
            "verdict": ("warmstart-limited" if best_f - queue_F > 0.02
                        else "structure-limited"),
        })

    save_json(
        {"rows": rows, "topology": args.topology, "N": args.n, "h": args.h,
         "e0": e0, "gap": gap, "maxiter": args.maxiter,
         "schema": "seed_ablation_v1"},
        "bond_ablation", f"seed_ablation_square_N{args.n}_h{args.h:.2f}.json",
        params={"experiment": "seed_ablation", "N": args.n, "h": args.h},
        description="Structure-vs-warmstart: pruned ansätze optimized from 4 seeds")

    # Verdict summary.
    print("\n=== Verdict (best-of-4 vs queue) ===")
    for r in rows:
        delta = r["best_fidelity"] - r["queue_fidelity"]
        print(f"  {r['variant'][:40]:40s} Δ={delta:+.4f}  → {r['verdict']}")
    print("\nDONE", flush=True)
    return 0


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
