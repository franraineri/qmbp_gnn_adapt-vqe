#!/usr/bin/env python
"""Reduce 2q by dropping near-zero nnn blocks from the tied N=16 p4 ansatz.

The converged N=16 p4 tied θ (fidelity 0.7951) revealed that the nnn block of the
FIRST layer is essentially inactive (|θ_L0_nnn| = 0.025, vs ~0.13-0.17 for the
other nnn layers and ~1.0-1.5 for the x blocks). A near-zero tied angle means that
block contributes almost nothing to the prepared state — so dropping it should
keep the fidelity while removing its 2q cost. Each dropped nnn block removes one
full symmetry orbit (32 bonds × 2 CX = 64 CX), so the ansatz stays symmetric.

This builds structural variants that omit the nnn block from the earliest layer(s),
seeds each from the N=16 p4 θ (with the dropped slot removed), optimizes FIDELITY
(the objective that works at scale — energy degrades it), and compares 2q vs fid
against the full p4 baseline (512 CX, 0.7951).

Variants (all tied, symmetric):
  p4_full          [nn,nnn,x]×4                                512 2q (baseline)
  p4_drop_L0nnn    [nn,x][nn,nnn,x]×3                          448 2q  (-64)
  p4_drop_L0L1nnn  [nn,x][nn,x][nn,nnn,x]×2                    384 2q  (-128)

Reuses: optimize_fidelity + build_frustrated_h + _tilted_lattice +
_load_src_tied_theta from run_tilted_transfer, cached_degenerate_ground_manifold,
tied builder, cx_and_params, subspace_fidelity, StudyPersister.

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/run_tilted_drop_nnn.py \
        --n 16 --h 0.5 --restarts 4 --maxiter 2000
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

from hva_vl_study_common import StudyPersister  # noqa: E402
from run_tilted_tied_ab import _tilted_lattice  # noqa: E402
from run_tilted_transfer import (  # noqa: E402
    build_frustrated_h,
    optimize_fidelity,
)

from qmbp_simulation.analysis.fidelity import (  # noqa: E402
    cached_degenerate_ground_manifold,
    subspace_fidelity,
)
from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.bond_mask import BondSelection  # noqa: E402
from qmbp_simulation.circuits.symmetry import detect_symmetry_group  # noqa: E402
from qmbp_simulation.framework.study_core import cx_and_params  # noqa: E402
from qmbp_simulation.models.tilted_square import MAGIC_TILTS, generate_square_tilted  # noqa: E402

SUBDIR = "bond_ablation"
MODEL = "tfim_frustrated"
TOPO_KEY = "square_tilted"
_FULL_SYMS = 100000

# Structural variants: which layers keep the nnn block (p=4, 4 layers).
# Each layer unit is [nn, (nnn?), x]; a dropped nnn removes one orbit → -64 2q.
VARIANTS = {
    "p4_full": [True, True, True, True],  # 512 2q
    "p4_drop_L0nnn": [False, True, True, True],  # 448 2q
    "p4_drop_L0L1nnn": [False, False, True, True],  # 384 2q
    "p4_drop_L0L1L2nnn": [False, False, False, True],  # 320 2q (aggressive)
}


def _blocks_for(keep_nnn):
    blocks = []
    for keep in keep_nnn:
        blocks.append("nn")
        if keep:
            blocks.append("nnn")
        blocks.append("x")
    return blocks


def _seed_from_p4(full_theta, keep_nnn, rx_final=True):
    """Build a seed for a drop-nnn variant from the full p4 tied θ.

    Full p4 layout (tied, 1 val/block): per layer [nn, nnn, x]; then rx_final.
    For a variant that drops some nnn, we take the full θ and SKIP the nnn slot of
    any layer whose keep flag is False. The remaining slots line up with the
    variant's block order, so the kept blocks inherit their converged angles.
    """
    out = []
    idx = 0
    for keep in keep_nnn:
        nn_v, nnn_v, x_v = full_theta[idx], full_theta[idx + 1], full_theta[idx + 2]
        idx += 3
        out.append(nn_v)
        if keep:
            out.append(nnn_v)
        out.append(x_v)
    if rx_final:
        out.append(full_theta[idx])  # rx_final slot (last)
    return np.asarray(out, float)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Drop near-zero nnn blocks from tied N=16 p4 to cut 2q")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--j2", type=float, default=0.5)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--maxiter", type=int, default=2000)
    ap.add_argument("--seed0", type=int, default=98800)
    ap.add_argument("--ftol", type=float, default=1e-12)
    args = ap.parse_args(argv)

    if args.n not in MAGIC_TILTS:
        print(f"N={args.n} not magic; available {sorted(MAGIC_TILTS)}")
        return 1

    t_start = time.time()
    l1, l2 = MAGIC_TILTS[args.n]
    cl = generate_square_tilted(l1, l2)
    nn_edges, nnn_edges = cl.nn_edges, cl.nnn_edges
    H = build_frustrated_h(nn_edges, nnn_edges, cl.n_qubits, args.h, args.j2)
    manifold, e0, phys_gap, m_dim = cached_degenerate_ground_manifold(
        H, topology=TOPO_KEY, n_qubits=args.n, model=MODEL, h=args.h, j2=args.j2
    )
    psi = manifold[:, 0]
    mref = manifold if m_dim > 1 else manifold[:, :1]
    group = detect_symmetry_group(nn_edges, nnn_edges, cl.n_qubits, max_syms=_FULL_SYMS)
    lat = _tilted_lattice(cl, args.h, args.j2)
    sel = BondSelection(nn_edges=nn_edges, nnn_edges=nnn_edges)
    builder = HVACircuitBuilder()

    # The full p4 converged θ (from the transfer+fidelity run) is the seed source.
    # The seed MUST be the transfer+fidelity-optimized p4 θ (fid ~0.7951). Do NOT use
    # _load_src_tied_theta(args.n, ...): it reads the N=16 pareto JSON, whose p4 row is
    # the failed calibrated/energy run (fid 0.10, also 13 params) — a silent wrong seed.
    import json

    full_theta = None
    tp = Path("results/hva_vl_study/bond_ablation") / f"tilted_transfer_N9to{args.n}_h{args.h:.2f}.json"
    if tp.exists():
        data = json.loads(tp.read_text())
        for r in data.get("rows", []):
            if r.get("objective") == "fidelity" and r.get("p_layers") == 4 and r.get("theta_best"):
                full_theta = np.asarray(r["theta_best"], float)
                break
    if full_theta is None or full_theta.size != 13:
        print(f"[drop_nnn] could not load N={args.n} p4 full θ (len 13) — abort")
        return 1

    def report_fid(qc, theta):
        return subspace_fidelity(qc, np.asarray(theta, float), mref)

    rows: list[dict] = []
    persister = StudyPersister(
        subdir=SUBDIR,
        out_file=f"tilted_drop_nnn_N{args.n}_h{args.h:.2f}.json",
        fingerprint={"n_qubits": args.n, "p_layers": 4, "model": MODEL, "topology": TOPO_KEY, "h": args.h},
        extra={
            "topology": TOPO_KEY,
            "tilt": [l1, l2],
            "N": args.n,
            "h": args.h,
            "J2": args.j2,
            "schema": "tilted_drop_nnn_v1",
            "manifold_dim": m_dim,
            "group_order": group.order,
            "phys_gap": phys_gap,
            "e0": e0,
            "baseline_p4_full_fid": 0.7951,
        },
        params={"experiment": "tilted_drop_nnn", "N": args.n, "h": args.h},
        description="Drop near-zero nnn blocks from tied N p4 to cut 2q at equal fidelity",
        rows_ref=rows,
    )

    print(
        f"[drop_nnn] N={args.n} tilt=({l1},{l2}) h={args.h} group={group.order} md={m_dim} "
        f"gap={phys_gap:.5f} | seed=N{args.n} p4 full θ (fid 0.7951) (setup {time.time() - t_start:.1f}s)",
        flush=True,
    )
    print(f"\n{'variant':20s} {'2q':>4} {'params':>7} {'seed_init':>10} {'converged':>10} {'Δ2q':>6}", flush=True)

    for name in args.variants:
        keep = VARIANTS[name]
        blocks = _blocks_for(keep)
        qc, _ = builder.create_bond_resolved_symmetry_tied(
            cl.n_qubits, lat, blocks=blocks, bond_selection=sel, symmetry_group=group, rx_final=True
        )
        n2q, npar = cx_and_params(qc)
        seed = _seed_from_p4(full_theta, keep, rx_final=True)
        if seed.size != npar:
            print(f"  {name}: seed len {seed.size} != params {npar} — skip", flush=True)
            continue
        init = report_fid(qc, seed)
        if name == "p4_full" and init < 0.75:
            print(
                f"[drop_nnn] ABORT: p4_full seed init={init:.4f} but the stored θ should "
                f"give ~0.795 — wrong seed loaded",
                flush=True,
            )
            return 1
        t0 = time.time()
        bf, theta_best, runs = optimize_fidelity(
            qc, H, psi, seed, restarts=args.restarts, maxiter=args.maxiter, seed0=args.seed0, ftol=args.ftol
        )
        rfid = report_fid(qc, theta_best)
        row = {
            "variant": name,
            "blocks": blocks,
            "n_qubits": args.n,
            "tilt": [l1, l2],
            "h": round(float(args.h), 2),
            "topology": TOPO_KEY,
            "n_nnn_blocks": sum(keep),
            "seed_init_fidelity": float(init),
            "report_fidelity": float(rfid),
            "n_2q_transpiled": n2q,
            "n_params": npar,
            "delta_2q_vs_full": n2q - 512,
            "fidelity_per_cx": (rfid / n2q) if n2q else None,
            "theta_best": theta_best.tolist(),
        }
        rows.append(row)
        persister.persist(rows)
        print(
            f"{name:20s} {n2q:>4} {npar:>7} {init:>10.4f} {rfid:>10.4f} {n2q - 512:>+6} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    persister.persist(rows, status="final")

    print("\n=== 2q↔fidelity (dropping near-zero nnn blocks) ===", flush=True)
    print("  baseline p4_full: 0.7951 @ 512 2q", flush=True)
    for r in sorted(rows, key=lambda x: x["n_2q_transpiled"]):
        verdict = (
            "✓ kept fid"
            if r["report_fidelity"] >= 0.7951 - 0.01
            else ("~ close" if r["report_fidelity"] >= 0.75 else "✗ dropped")
        )
        print(
            f"  {r['variant']:20s} fid={r['report_fidelity']:.4f} 2q={r['n_2q_transpiled']:>4} "
            f"({r['delta_2q_vs_full']:+d})  fpc={r['fidelity_per_cx']:.3e}  {verdict}",
            flush=True,
        )
    print(f"\nDONE ({time.time() - t_start:.1f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
