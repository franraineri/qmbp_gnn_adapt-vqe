#!/usr/bin/env python
"""Recurring angle-study analysis over the HVA result corpus (2D frustrated).

Turns the raw saved θ vectors into the per-block, physically-interpretable
metrics an angle study needs — so the analysis is reproducible and extendable
rather than ad-hoc JSON parsing. Consumes the reusable query layer
(``results_query``) and the block decomposition (``decompose_theta``).

Analyses (each a subcommand):
- ``blocks``   : per-block angle stats (mean/std/absmean of θ_nn, θ_nnn, θ_x) for
                 the best converged run of a config — the angle signature.
- ``basins``   : for basin-count data, cluster runs by fidelity and report the
                 per-block angle stats + pairwise L2 distances between basin
                 representatives (good vs bad basin angle signature).
- ``seed-gap`` : per-block L2 distance between the analytic warm-start seed and
                 the reached optimum, vs fidelity (why the seed misfires).
- ``scan-n``   : how the best-run per-block angles drift with N (fixed p, h).

Only CONVERGED runs are used (starved ones flagged, never pooled — see the
warm-start guardrail). Outputs a markdown table to stdout and, with ``--save``,
a NaN-safe JSON under the study tree for reuse.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/analyze_angles.py blocks \
        --topology square --n 10 --h 0.5
    .venv/bin/python scripts/analysis/vl_vs_hva/analyze_angles.py scan-n \
        --topology square --p 2 --h 0.5 --n-list 8 10 12 18
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import results_query as rq  # noqa: E402


def _layout_for(topology, n, h):
    """Recover (n_nn, n_nnn, n_qubits) for a config from the lattice."""
    from qmbp_simulation.models import make_lattice
    from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

    lat = make_lattice(topology, n, J=1.0, h=h)
    return len(lat.edges), len(HamiltonianBuilder._generate_nnn_edges(lat)), n


def _decompose_record(rec, topology, h):
    """Decompose a record's theta_final into blocks, inferring p from length."""
    if rec.theta_final is None or rec.n_qubits is None:
        return None
    n_nn, n_nnn, n_q = _layout_for(topology, rec.n_qubits, h)
    per = n_nn + n_nnn + n_q
    total = len(rec.theta_final)
    if total % per != 0:
        return None  # a variant/half-layer — needs its own blocks (handled elsewhere)
    p = total // per
    return rq.decompose_theta(rec.theta_final, n_nn, n_nnn, n_q, p)


def cmd_blocks(args):
    b = rq.best_result(topology=args.topology, n=args.n, p_layers=args.p, h=args.h)
    if b is None or not b.extra.get("converged", True):
        print("No converged best record found.")
        return {}
    dec = _decompose_record(b, args.topology, args.h)
    if dec is None:
        print("Could not decompose (variant layout?).")
        return {}
    stats = rq.block_stats(dec)
    print(
        f"\nAngle signature — {args.topology} N={args.n} p={args.p} h={args.h} "
        f"(best fid={b.fidelity:.4f}, src={b.source_file})"
    )
    print(f"  {'block':10s} {'mean':>8s} {'std':>8s} {'absmean':>8s} {'n':>4s}")
    for blk, s in stats.items():
        print(f"  {blk:10s} {s['mean']:>8.4f} {s['std']:>8.4f} {s['absmean']:>8.4f} {s['n']:>4d}")
    return {"config": vars(args), "fidelity": b.fidelity, "block_stats": stats}


def cmd_basins(args):
    recs = [
        r
        for r in rq.query_results(topology=args.topology, n=args.n, p_layers=args.p, h=args.h)
        if r.theta_final is not None and r.extra.get("converged", True)
    ]
    if not recs:
        print("No converged records.")
        return {}
    fids = np.array([r.fidelity for r in recs])
    order = np.argsort(fids)
    clusters = [[order[0]]]
    for i in order[1:]:
        if fids[i] - fids[clusters[-1][-1]] <= args.tol:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    clusters.sort(key=lambda c: -max(fids[i] for i in c))
    print(
        f"\nBasins — {args.topology} N={args.n} p={args.p} h={args.h} ({len(recs)} converged, {len(clusters)} basins)"
    )
    reps = []
    for ci, c in enumerate(clusters):
        rep = max(c, key=lambda i: fids[i])
        dec = _decompose_record(recs[rep], args.topology, args.h)
        reps.append((ci, recs[rep], dec))
        s = rq.block_stats(dec) if dec else {}
        line = " ".join(f"{b}:abs={s[b]['absmean']:.3f}" for b in s)
        print(f"  basin {ci}: fid {min(fids[i] for i in c):.4f}-{max(fids[i] for i in c):.4f} ({len(c)} runs)  {line}")
    # pairwise L2 between basin representatives (same length only)
    print("  pairwise L2(theta) between basin reps:")
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            ti, tj = reps[i][1].theta_final, reps[j][1].theta_final
            if ti is not None and tj is not None and len(ti) == len(tj):
                d = float(np.linalg.norm(np.array(ti) - np.array(tj)))
                print(f"    basin{reps[i][0]}–basin{reps[j][0]}: {d:.3f}")
    return {"n_basins": len(clusters)}


def cmd_seed_gap(args):
    from qmbp_simulation.analysis.warmstart import second_order_warmstart_theta

    n_nn, n_nnn, n_q = _layout_for(args.topology, args.n, args.h)
    recs = [
        r
        for r in rq.query_results(topology=args.topology, n=args.n, p_layers=args.p, h=args.h)
        if r.theta_final is not None and r.extra.get("converged", True)
    ]
    if not recs:
        print("No converged records.")
        return {}
    seed = second_order_warmstart_theta(n_nn, n_nnn, n_q, args.p, args.h, J=1.0, J2=args.j2)
    print(f"\nSeed→optimum gap — {args.topology} N={args.n} p={args.p} h={args.h}")
    print(f"  {'fid':>6s} {'L2(seed,opt)':>13s}  per-block absΔ")
    rows = []
    for r in sorted(recs, key=lambda x: -x.fidelity)[: args.top]:
        opt = np.array(r.theta_final)
        if len(opt) != len(seed):
            continue
        d = float(np.linalg.norm(opt - seed))
        ds = rq.decompose_theta(opt - seed, n_nn, n_nnn, n_q, args.p)
        blk = " ".join(f"{b.split('_')[1]}={np.mean(np.abs(v)):.3f}" for b, v in ds.items())
        print(f"  {r.fidelity:>6.4f} {d:>13.3f}  {blk}")
        rows.append({"fidelity": r.fidelity, "l2": d})
    return {"rows": rows}


def cmd_scan_n(args):
    print(f"\nAngle drift with N — {args.topology} p={args.p} h={args.h}")
    print(f"  {'N':>3s} {'fid':>6s} {'nn_abs':>7s} {'nnn_abs':>7s} {'x_abs':>7s}")
    rows = []
    for n in args.n_list:
        b = rq.best_result(topology=args.topology, n=n, p_layers=args.p, h=args.h)
        if b is None:
            print(f"  {n:>3d}  (no data)")
            continue
        dec = _decompose_record(b, args.topology, args.h)
        if dec is None:
            print(f"  {n:>3d}  (no theta)")
            continue
        s = rq.block_stats(dec)
        print(
            f"  {n:>3d} {b.fidelity:>6.4f} {s['theta_nn']['absmean']:>7.3f} "
            f"{s['theta_nnn']['absmean']:>7.3f} {s['theta_x']['absmean']:>7.3f}"
        )
        rows.append(
            {
                "N": n,
                "fidelity": b.fidelity,
                "nn_abs": s["theta_nn"]["absmean"],
                "nnn_abs": s["theta_nnn"]["absmean"],
                "x_abs": s["theta_x"]["absmean"],
            }
        )
    return {"rows": rows}


def _block_table(dec):
    """Per-block mean/std/absmean dict for one decomposed theta (NaN-safe)."""
    return rq.block_stats(dec)


def cmd_basin_signature(args):
    """Good- vs bad-basin per-block angle signature for a multi-run config.

    Splits converged runs at a fidelity threshold, reports per-block
    mean/std/absmean for each group and the per-block L2 between the best and
    worst converged representative — answering which block distinguishes basins.
    Starved/unknown-convergence runs are listed but never pooled.
    """
    all_recs = rq.query_results(topology=args.topology, n=args.n, p_layers=args.p, h=args.h)
    with_theta = [r for r in all_recs if r.theta_final is not None]
    conv = [r for r in with_theta if r.converged is True]
    non_conv = [r for r in with_theta if r.converged is not True]
    print(f"\nBasin signature — {args.topology} N={args.n} p={args.p} h={args.h}")
    print(f"  {len(with_theta)} angle-bearing runs: {len(conv)} converged, {len(non_conv)} starved/unknown (excluded)")
    if non_conv:
        print("  excluded (not converged):")
        for r in sorted(non_conv, key=lambda x: -(x.fidelity or -1)):
            print(f"    fid={r.fidelity:.4f} nit={r.extra.get('nit')} conv={r.converged} src={r.source_file}")
    if len(conv) < 2:
        print("  <2 converged runs — cannot compare basins.")
        return {"n_converged": len(conv), "note": "insufficient converged runs"}
    fids = np.array([r.fidelity for r in conv])
    thr = args.fid_thr if args.fid_thr is not None else float(np.median(fids))
    hi = [r for r in conv if r.fidelity >= thr]
    lo = [r for r in conv if r.fidelity < thr]
    print(f"  threshold fid={thr:.4f}: {len(hi)} high, {len(lo)} low")
    out = {"n_converged": len(conv), "threshold": thr, "groups": {}}
    for label, grp in (("high", hi), ("low", lo)):
        if not grp:
            continue
        # aggregate per-block over the group
        blocks = {"theta_nn": [], "theta_nnn": [], "theta_x": []}
        for r in grp:
            dec = _decompose_record(r, args.topology, args.h)
            if dec is None:
                continue
            for b in blocks:
                blocks[b].append(dec[b])
        print(f"  [{label}] fids={[round(r.fidelity, 4) for r in grp]}")
        gstats = {}
        for b, arrs in blocks.items():
            if not arrs:
                continue
            allv = np.concatenate(arrs)
            gstats[b] = {
                "mean": float(np.mean(allv)),
                "std": float(np.std(allv)),
                "absmean": float(np.mean(np.abs(allv))),
                "n": int(allv.size),
            }
            print(
                f"    {b:10s} mean={gstats[b]['mean']:+.4f} "
                f"std={gstats[b]['std']:.4f} absmean={gstats[b]['absmean']:.4f}"
            )
        out["groups"][label] = gstats
    # per-block L2 between best and worst converged representative
    best = max(conv, key=lambda r: r.fidelity)
    worst = min(conv, key=lambda r: r.fidelity)
    db = _decompose_record(best, args.topology, args.h)
    dw = _decompose_record(worst, args.topology, args.h)
    if db and dw:
        print(f"  per-block L2(best fid={best.fidelity:.4f} vs worst fid={worst.fidelity:.4f}):")
        l2 = {}
        for b in ("theta_nn", "theta_nnn", "theta_x"):
            d = float(np.linalg.norm(db[b] - dw[b]))
            l2[b] = d
            print(f"    {b:10s} L2={d:.4f}")
        disc = max(l2, key=l2.get)
        print(f"  -> largest per-block separation: {disc} (L2={l2[disc]:.4f})")
        out["l2_best_vs_worst"] = l2
        out["discriminating_block"] = disc
    return out


def cmd_npz_blocks(args):
    """Per-block angle signature of the resources/*.npz optima (best-known points).

    Reads the NPZ layout metadata (n_nn/n_nnn/n_qubits/p_layers) directly so no
    lattice recompute is needed, and flags convergence via the NPZ ``done`` /
    ``maxiter`` fields when present (best-known != converged).
    """
    import glob

    npz_dir = Path(rq.study_dir("resources")) / args.subdir
    files = sorted(glob.glob(str(npz_dir / "*.npz")))
    print(f"\nNPZ block signatures — {npz_dir}")
    rows = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "theta" not in d.files:
            continue
        theta = d["theta"]
        meta = {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files if k != "theta"}
        n_nn = int(meta.get("n_nn", 0))
        n_nnn = int(meta.get("n_nnn", 0))
        n_q = int(meta.get("n_qubits", 0))
        p = int(meta.get("p_layers", 0))
        if not (n_nn and n_q and p):
            print(f"  {Path(f).name}: missing layout metadata, skipped")
            continue
        try:
            dec = rq.decompose_theta(theta, n_nn, n_nnn, n_q, p)
        except ValueError as e:
            print(f"  {Path(f).name}: {e}")
            continue
        s = rq.block_stats(dec)
        done = meta.get("done")
        mi = meta.get("maxiter")
        conv_note = (
            "done=True"
            if done is True
            else (
                "done=False (interrupted)"
                if done is False
                else (f"maxiter={mi} (starved)" if mi is not None else "convergence unknown")
            )
        )
        print(f"  {Path(f).name}  fid={meta.get('fidelity'):.4f}  gap={meta.get('gap'):.4f}  [{conv_note}]")
        for b in ("theta_nn", "theta_nnn", "theta_x"):
            print(f"    {b:10s} mean={s[b]['mean']:+.4f} std={s[b]['std']:.4f} absmean={s[b]['absmean']:.4f}")
        rows.append(
            {
                "file": Path(f).name,
                "fidelity": float(meta.get("fidelity")),
                "h": float(meta.get("h")),
                "gap": float(meta.get("gap")),
                "converged_note": conv_note,
                "block_stats": s,
            }
        )
    return {"rows": rows}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Recurring HVA angle-study analysis")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("blocks", "basins", "seed-gap", "scan-n", "basin-signature", "npz-blocks"):
        sp = sub.add_parser(name)
        sp.add_argument("--topology", default="square")
        sp.add_argument("--n", type=int, default=10)
        sp.add_argument("--p", type=int, default=2)
        sp.add_argument("--h", type=float, default=0.5)
        sp.add_argument("--j2", type=float, default=0.5)
        sp.add_argument("--tol", type=float, default=0.02)
        sp.add_argument("--top", type=int, default=8)
        sp.add_argument("--n-list", type=int, nargs="+", default=[8, 10, 12, 18])
        sp.add_argument("--fid-thr", type=float, default=None, help="basin-signature high/low split (default: median)")
        sp.add_argument("--subdir", default="n18_frustrated", help="npz-blocks: subdir under resources/")
        sp.add_argument("--save", action="store_true")
    args = p.parse_args(argv)
    fn = {
        "blocks": cmd_blocks,
        "basins": cmd_basins,
        "seed-gap": cmd_seed_gap,
        "scan-n": cmd_scan_n,
        "basin-signature": cmd_basin_signature,
        "npz-blocks": cmd_npz_blocks,
    }[args.cmd]
    out = fn(args)
    if getattr(args, "save", False) and out:
        from hva_vl_study_common import save_json

        save_json(
            out,
            "hva_nnn_sweep",
            f"angle_analysis_{args.cmd}_{args.topology}_N{args.n}_h{args.h:.2f}.json",
            params={"analysis": args.cmd},
            description=f"Angle analysis: {args.cmd}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
