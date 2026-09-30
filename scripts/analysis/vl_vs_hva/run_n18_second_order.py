#!/usr/bin/env python
"""N=18 frustrated square, second-order warm-start via the refactored StudyRunner.

Runs the confirmed best method for the frustrated transition (second-order /
frustration-aware warm-start, regime-gated) at N=18. h=0.5 sits inside the gate
[0.4, 0.6] so the second-order seed is used; other h fall back to Metropolis per
the gate. Uses the shared StudyRunner + StudyCheckpoint for crash-safe per-h
persistence and resume, and the study service for traceable JSON + reusable
artifacts (theta NPZ, bound QPY).

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/run_n18_second_order.py \
        --h 0.5 --maxiter 150
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import save_circuit_qpy, save_json, study_dir  # noqa: E402


def main(argv=None) -> int:
    from qmbp_simulation.framework.study_checkpoint import StudyCheckpoint
    from qmbp_simulation.framework.study_runner import StudyRunner

    p = argparse.ArgumentParser(description="N=18 second-order warm-start (StudyRunner)")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=18)
    p.add_argument("--h", type=float, nargs="+", default=[0.5])
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--p", type=int, default=2)
    p.add_argument("--maxiter", type=int, default=150)
    p.add_argument("--sigma", type=float, default=0.15)
    p.add_argument("--model", default="tfim_frustrated")
    p.add_argument(
        "--strategy",
        default="second_order",
        choices=["second_order", "mixed", "metropolis", "bestof", "first_order"],
        help="Warm-start strategy. 'mixed' runs varied-basin restarts "
        "(see MIXED_RESTART_TYPES) with per-restart convergence flags.",
    )
    p.add_argument(
        "--n-hops", type=int, default=8, help="Metropolis basin-hopping hops (only used by --strategy metropolis)."
    )
    args = p.parse_args(argv)

    strat_tag = "secondorder" if args.strategy == "second_order" else args.strategy
    subdir = "hva_nnn_sweep"
    tag = f"{args.topology}_N{args.n}_p{args.p}_{strat_tag}_mi{args.maxiter}"
    out_file = f"n18_{tag}.json"
    art_dir = study_dir("resources") / f"n{args.n}_frustrated"

    runner = StudyRunner(
        topology=args.topology,
        n_qubits=args.n,
        p_layers=args.p,
        j2=args.j2,
        model=args.model,
        strategy=args.strategy,
        maxiter=args.maxiter,
        sigma=args.sigma,
        n_hops=args.n_hops,
        ansatz="nnn",
    )

    out_path = study_dir(subdir) / out_file

    def writer(payload):
        return save_json(
            payload,
            subdir,
            out_file,
            params={
                "N": args.n,
                "p_layers": args.p,
                "J2": args.j2,
                "model": args.model,
                "maxiter": args.maxiter,
                "sigma": args.sigma,
                "strategy": args.strategy,
                "regime_gate": "enabled",
            },
            description=f"{args.topology} N={args.n} frustrated HVA nnn "
            f"p={args.p}, {args.strategy} warm-start (regime-gated), "
            "crash-safe per-h via StudyRunner",
        )

    ckpt = StudyCheckpoint("per_h", writer=writer, path=out_path)
    n_resumed = ckpt.resume()
    if n_resumed:
        print(f"[resume] {n_resumed} h already done", flush=True)

    print(
        f"[n18-warmstart] topo={args.topology} N={args.n} h={args.h} "
        f"maxiter={args.maxiter} sigma={args.sigma} strategy={args.strategy} (gated)",
        flush=True,
    )

    def on_point(pt):
        runs = pt.get("runs", [])
        n_conv = sum(1 for r in runs if r.get("converged"))
        print(
            f"[h={pt['h']:.2f}] strategy={pt['strategy_resolved']} "
            f"fid={pt['fidelity']:.4f} dE/gap="
            f"{pt['de_gap'] if pt['de_gap'] is None else round(pt['de_gap'], 4)} "
            f"e_vqe={pt['e_vqe']:.4f} gap={pt['gap']:.5f} "
            f"converged={n_conv}/{len(runs)}",
            flush=True,
        )
        for r in runs:
            conv = "conv" if r.get("converged") else "CAP "
            print(
                f"    · {r.get('type', 'r' + str(r.get('restart', '?'))):18s} "
                f"fid={r['fidelity']:.4f} nit={r['nit']:>4} [{conv}]",
                flush=True,
            )

    runner.run_sweep(
        args.h,
        ckpt,
        artifact_dir=art_dir,
        extra={"schema": "n18_second_order_v1"},
        on_point=on_point,
        qpy_saver=save_circuit_qpy,
    )
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
