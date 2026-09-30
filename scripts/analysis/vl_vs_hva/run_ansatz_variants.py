#!/usr/bin/env python
"""Ansatz-structure variant study (p=1-based) — expressivity vs 2q-gate cost.

Compares p=1 ansatz structure modifications (Options A–D) at a FIXED protocol
(same N, h=0.5, same maxiter, same restarts, best-of by energy) to see which
modification recovers expressivity — and at what 2-qubit-gate cost — relative to
adding a full extra layer (p=2, p=3 anchors).

Two modes
---------
1. ``--find-saturation``: sweep N for the p1_base ceiling to locate the smallest
   N where plain p=1 stops saturating (ceiling drops below --sat-threshold), so
   the variant comparison runs where there is expressivity headroom to recover.
   Persists a saturation-scan artifact.

2. default (variant comparison at fixed --n): for each variant in the registry,
   run a multi-seed best-of to genuine convergence, record best fidelity, the
   REAL transpiled 2q-gate count, params, ΔE/gap, and derived efficiency
   (fidelity-per-CX). Anchors p1/p2/p3_base frame the variants.

Reuse
-----
- Circuit engine: HVACircuitBuilder.create_bond_resolved_frustrated_configurable
  + the declarative VARIANTS registry (circuits/hva_variants.py).
- Physics/optimizer: NoiselessBackend (cached H + adjoint gradient), warm-start
  seeds, exact fidelity — same robust stack as run_n18_fair_convergence.py.
- 2q counting: analysis.circuit_visualizer.circuit_summary / transpiled_circuit_stats.
- Persistence: hva_vl_study_common.save_json (NaN-safe), incremental after each item.

Usage
-----
    # Phase 0: find where p=1 stops saturating
    .venv/bin/python scripts/analysis/vl_vs_hva/run_ansatz_variants.py \
        --find-saturation --n-list 6 8 10 12 --h 0.5 --restarts 3 --maxiter 2000

    # Phase 2: compare all variants at the chosen N
    .venv/bin/python scripts/analysis/vl_vs_hva/run_ansatz_variants.py \
        --n 8 --h 0.5 --restarts 3 --maxiter 2000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import save_json  # noqa: E402

from qmbp_simulation.framework.study_checkpoint import build_resumable_payload  # noqa: E402


def _ground_state(topology, n, h, j2):
    """Frustrated ground state for the variant study: ``(lat, H, psi, e0, gap)``.

    Thin adapter over :func:`study_core.ground_state`; the default bond-resolved
    circuit is discarded because variants build their own circuits via
    ``build_variant``.
    """
    from qmbp_simulation.framework.study_core import ground_state

    lat, _qc, H, psi, e0, gap, _n_nn, _n_nnn = ground_state(topology, n, h, j2, 1)
    return lat, H, psi, e0, gap


def _optimize_bestof(
    qc, H, psi, *, restarts, maxiter, sigma, seed0, warm_theta=None, analytic_theta=None, on_restart=None
):
    """Robust multi-seed best-of. Returns (best_fid, best_e, runs).

    Seeds tried (best-of by energy across all): the analytic variant warm-start
    (``analytic_theta`` — the informed starting point every variant deserves), an
    optional checkpoint seed (``warm_theta``), and ``restarts`` random in-box
    starts for basin coverage. Uses the cached-H NoiselessBackend + adjoint
    gradient. ``sigma`` kept for signature compatibility. ``on_restart`` is the
    base per-restart persistence hook (fires after each restart; crash-safe).
    """
    import numpy as np

    from qmbp_simulation.framework.study_core import make_cost_fid, optimize_bestof

    cost, fid, grad, _ = make_cost_fid(qc, H, psi)
    npar = qc.num_parameters

    # Random best-of (restart 0 = analytic seed when provided, else θ=0).
    best_fid, best_e, runs = optimize_bestof(
        cost,
        fid,
        grad,
        npar,
        restarts=restarts,
        maxiter=maxiter,
        seed0=seed0,
        warm_theta=analytic_theta if analytic_theta is not None else warm_theta,
        on_restart=on_restart,
    )

    # If BOTH an analytic seed and a checkpoint exist, also try the checkpoint as
    # an extra seed-0 run and keep the better (best-of by energy).
    if analytic_theta is not None and warm_theta is not None and len(warm_theta) == npar:
        alt_fid, alt_e, alt_runs = optimize_bestof(
            cost,
            fid,
            grad,
            npar,
            restarts=1,
            maxiter=maxiter,
            seed0=seed0,
            warm_theta=np.asarray(warm_theta, float),
            on_restart=on_restart,
        )
        runs = runs + alt_runs
        if alt_e < best_e:
            best_fid, best_e = alt_fid, alt_e
    return best_fid, best_e, runs


def _cx_and_params(qc):
    """Real transpiled 2q count + logical params (shared core helper)."""
    from qmbp_simulation.framework.study_core import cx_and_params

    return cx_and_params(qc)


def run_saturation(args) -> int:
    from qmbp_simulation.circuits import HVACircuitBuilder
    from qmbp_simulation.circuits.hva_variants import VARIANTS, build_variant

    builder = HVACircuitBuilder()
    base = VARIANTS["p1_base"]
    rows = []
    out_file = f"ansatz_saturation_{args.topology}_p1_h{args.h:.2f}.json"
    print(
        f"[saturation] p1_base ceiling vs N (h={args.h}, restarts={args.restarts}, maxiter={args.maxiter})", flush=True
    )
    for n in args.n_list:
        t0 = time.time()
        lat, H, psi, e0, gap = _ground_state(args.topology, n, args.h, args.j2)
        qc, _ = build_variant(builder, n, lat, base)
        n_2q, npar = _cx_and_params(qc)
        fid, e_best, runs = _optimize_bestof(
            qc, H, psi, restarts=args.restarts, maxiter=args.maxiter, sigma=args.sigma, seed0=args.seed0
        )
        saturates = fid >= args.sat_threshold
        row = {
            "N": n,
            "gap": gap,
            "ceiling_fid": fid,
            "e_best": e_best,
            "e0": e0,
            "n_2q": n_2q,
            "n_params": npar,
            "de_gap": abs(e_best - e0) / gap if gap > 0 else None,
            "saturates": saturates,
            "seconds": round(time.time() - t0, 1),
        }
        rows.append(row)
        save_json(
            {
                "rows": rows,
                "topology": args.topology,
                "h": args.h,
                "restarts": args.restarts,
                "maxiter": args.maxiter,
                "sat_threshold": args.sat_threshold,
                "schema": "ansatz_saturation_v1",
            },
            "hva_nnn_sweep",
            out_file,
            params={"experiment": "ansatz_saturation", "h": args.h},
            description="p1_base ceiling vs N to locate expressivity saturation",
        )
        print(
            f"  N={n:>2} gap={gap:.4f} ceiling={fid:.4f} n_2q={n_2q} "
            f"saturates(>{args.sat_threshold})={saturates} ({row['seconds']}s)",
            flush=True,
        )
    # Recommend the smallest N with headroom (not saturating)
    not_sat = [r for r in rows if not r["saturates"]]
    if not_sat:
        rec = min(not_sat, key=lambda r: r["N"])
        print(
            f"\nRECOMMENDED N for variant study: {rec['N']} (p1 ceiling={rec['ceiling_fid']:.4f}, has headroom)",
            flush=True,
        )
    else:
        print("\nAll tested N saturate p=1 — use a larger N or lower --sat-threshold.", flush=True)
    print("DONE", flush=True)
    return 0


def run_variants(args) -> int:
    from qmbp_simulation.circuits import HVACircuitBuilder
    from qmbp_simulation.circuits.hva_variants import VARIANTS, build_variant
    from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

    builder = HVACircuitBuilder()
    lat, H, psi, e0, gap = _ground_state(args.topology, args.n, args.h, args.j2)
    # Bond counts for the analytic variant warm-start seed (robust starting point).
    n_nn = len(lat.edges)
    n_nnn = len(HamiltonianBuilder._generate_nnn_edges(lat))
    out_file = f"ansatz_variants_{args.topology}_N{args.n}_h{args.h:.2f}.json"

    # Optional subset (e.g. only p1_half_nn/p2_base/p3_base at expensive N).
    selected = args.variants if args.variants else list(VARIANTS.keys())
    unknown = [v for v in selected if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s): {unknown}; available: {list(VARIANTS)}")

    # Resolve warm-start mode (backward-compatible: --warm-from-best == best-effort).
    # off         : no warm-start — homogeneous, unbiased structure comparison.
    # all-or-none : warm-start ONLY if EVERY selected variant has a checkpoint of
    #               its own param length — guarantees homogeneous treatment.
    # best-effort : warm each variant that has a checkpoint, skip those that don't
    #               (may be a MIXED, non-homogeneous comparison — flagged loudly).
    warm_mode = args.warm_mode
    if args.warm_from_best and warm_mode == "off":
        warm_mode = "best-effort"  # legacy flag maps to best-effort

    from results_query import load_best_theta

    # Precompute per-variant checkpoint availability (by exact param length).
    ckpt = {}
    if warm_mode != "off":
        for name in selected:
            qcv, _ = build_variant(builder, args.n, lat, VARIANTS[name])
            hit = load_best_theta(
                topology=args.topology, n=args.n, p_layers=None, h=args.h, expected_len=qcv.num_parameters
            )
            ckpt[name] = hit  # (fid, theta, src) or None
        n_have = sum(1 for v in ckpt.values() if v is not None)
        if warm_mode == "all-or-none" and n_have != len(selected):
            print(
                f"[warm] all-or-none: only {n_have}/{len(selected)} variants have a "
                f"checkpoint → warm-start DISABLED for all (homogeneous comparison).",
                flush=True,
            )
            warm_mode = "off"
            ckpt = {}
        elif warm_mode == "best-effort" and 0 < n_have < len(selected):
            print(
                f"[warm] ⚠ MIXED treatment: {n_have}/{len(selected)} variants "
                f"warm-started, the rest from θ=0. Comparison is NOT homogeneous — "
                f"see warm_applied per row. Use --warm-mode all-or-none for a fair "
                f"structure comparison.",
                flush=True,
            )

    print(
        f"[variants] N={args.n} h={args.h} restarts={args.restarts} "
        f"maxiter={args.maxiter} e0={e0:.5f} gap={gap:.5f} "
        f"variants={selected} warm_mode={warm_mode}",
        flush=True,
    )

    rows = []
    for name in selected:
        v = VARIANTS[name]
        t0 = time.time()
        qc, _ = build_variant(builder, args.n, lat, v)
        n_2q, npar = _cx_and_params(qc)
        # Robust warm-start for THIS variant's exact block layout.
        from qmbp_simulation.analysis.warmstart import (
            compose_extend_theta,
            variant_warmstart_theta,
        )

        seed_note = "analytic"
        analytic_theta = None
        # Prefer extend-from-base: if the variant extends a base (e.g. p2 + partial
        # layer), seed the shared prefix from the base's BEST KNOWN θ (near the
        # ground state) and the extra blocks analytically — the strongest seed.
        if v.extends and v.extends in VARIANTS:
            base_qc, _ = build_variant(builder, args.n, lat, VARIANTS[v.extends])
            base_hit = load_best_theta(
                topology=args.topology, n=args.n, p_layers=None, h=args.h, expected_len=base_qc.num_parameters
            )
            if base_hit is not None:
                _, base_theta, base_src = base_hit
                composed = compose_extend_theta(
                    base_theta,
                    list(v.extra_blocks),
                    n_nn,
                    n_nnn,
                    args.n,
                    args.h,
                    J=1.0,
                    J2=args.j2,
                    extra_rx_final=v.rx_final,
                    extra_rz_final=v.rz_final,
                )
                if len(composed) == npar:
                    analytic_theta = composed
                    seed_note = f"extend-from-{v.extends}({base_src})"
        if analytic_theta is None:
            analytic_theta = variant_warmstart_theta(
                v.blocks, n_nn, n_nnn, args.n, args.h, J=1.0, J2=args.j2, rx_final=v.rx_final, rz_final=v.rz_final
            )
            if len(analytic_theta) != npar:
                analytic_theta = None  # safety: layout mismatch → fall back to θ=0
                seed_note = "none (layout mismatch)"
        warm_theta, warm_src = None, None
        if warm_mode != "off" and ckpt.get(name) is not None:
            _, warm_theta, warm_src = ckpt[name]

        # Per-restart crash-safety: persist a PARTIAL artifact after every restart
        # of the in-progress variant, so an interrupt at large N never loses the
        # expensive completed restarts (this is what cost 4h once). Delegates to
        # the base resumable-partial contract (build_resumable_payload) for the
        # θ-sanitized, canonically-shaped snapshot, then wraps it in the study's
        # traceable artifact so the consolidated report can read it.
        def _persist_partial(partial_runs, best, _name=name, _v=v, _n2q=n_2q, _npar=npar):
            best_theta = None
            if partial_runs:
                best_theta = min(partial_runs, key=lambda rr: rr.get("energy", float("inf"))).get("theta_final")
            partial_row = {
                "variant": _name,
                "blocks": _v.blocks,
                "status": "in_progress",
                "n_restarts_done": len(partial_runs),
                "best_fidelity": (best or {}).get("fidelity"),
                "best_energy": (best or {}).get("energy"),
                "n_2q_transpiled": _n2q,
                "n_params": _npar,
                "runs": partial_runs,
            }
            payload = build_resumable_payload(
                rows=rows + [partial_row],
                theta=best_theta,
                fingerprint={
                    "n_qubits": args.n,
                    "variant": _name,
                    "model": "tfim_frustrated",
                    "topology": args.topology,
                    "h": args.h,
                },
                extra={
                    "topology": args.topology,
                    "N": args.n,
                    "h": args.h,
                    "restarts": args.restarts,
                    "maxiter": args.maxiter,
                    "e0": e0,
                    "gap": gap,
                    "schema": "ansatz_variants_v1",
                    "status": "partial",
                },
            )
            save_json(
                payload,
                "hva_nnn_sweep",
                out_file,
                params={
                    "experiment": "ansatz_variants",
                    "N": args.n,
                    "h": args.h,
                    "restarts": args.restarts,
                    "maxiter": args.maxiter,
                },
                description="Ansatz structure variants (partial — in progress)",
            )

        fid, e_best, runs = _optimize_bestof(
            qc,
            H,
            psi,
            restarts=args.restarts,
            maxiter=args.maxiter,
            sigma=args.sigma,
            seed0=args.seed0,
            warm_theta=warm_theta,
            analytic_theta=analytic_theta,
            on_restart=_persist_partial,
        )
        n_conv = sum(1 for r in runs if r["converged"])
        # Best restart's angles surfaced at row level for direct reuse; every
        # restart's init/final angles are in ``runs`` (see _optimize_bestof).
        best_run = min(runs, key=lambda rr: rr["energy"])
        row = {
            "variant": name,
            "description": v.description,
            "tags": list(v.tags),
            "blocks": v.blocks,
            "rx_final": v.rx_final,
            "rz_final": v.rz_final,
            "best_fidelity": fid,
            "e_best": e_best,
            "e0": e0,
            "gap": gap,
            "abs_error": abs(e_best - e0),
            "de_gap": abs(e_best - e0) / gap if gap > 0 else None,
            "n_2q_transpiled": n_2q,
            "n_params": npar,
            "fidelity_per_cx": (fid / n_2q) if n_2q > 0 else None,
            "converged": f"{n_conv}/{args.restarts}",
            "warm_mode": warm_mode,
            "analytic_warmstart": analytic_theta is not None,
            "seed_kind": seed_note,
            "warm_applied": warm_src is not None,
            "warm_source": warm_src,
            "best_theta_init": best_run["theta_init"],
            "best_theta_final": best_run["theta_final"],
            "runs": runs,
            "seconds": round(time.time() - t0, 1),
        }
        rows.append(row)
        save_json(
            {
                "rows": rows,
                "topology": args.topology,
                "N": args.n,
                "h": args.h,
                "restarts": args.restarts,
                "maxiter": args.maxiter,
                "e0": e0,
                "gap": gap,
                "schema": "ansatz_variants_v1",
            },
            "hva_nnn_sweep",
            out_file,
            params={
                "experiment": "ansatz_variants",
                "N": args.n,
                "h": args.h,
                "restarts": args.restarts,
                "maxiter": args.maxiter,
            },
            description="Ansatz structure variants (p=1-based): best fidelity "
            "vs 2q cost at a fixed protocol, anchored by p1/p2/p3",
        )
        fpc = row["fidelity_per_cx"]
        fpc_s = f"{fpc:.2e}" if fpc is not None else "n/a"
        print(
            f"  {name:16s} fid={fid:.4f} n_2q={n_2q:>4} npar={npar:>3} "
            f"fid/cx={fpc_s} conv={row['converged']} "
            f"({row['seconds']}s)",
            flush=True,
        )

    print("\n=== RANKING by best fidelity ===", flush=True)
    for r in sorted(rows, key=lambda x: -x["best_fidelity"]):
        fpc = r["fidelity_per_cx"]
        fpc_s = f"{fpc:.2e}" if fpc is not None else "n/a"
        print(
            f"  {r['variant']:16s} fid={r['best_fidelity']:.4f} n_2q={r['n_2q_transpiled']:>4} fid/cx={fpc_s}",
            flush=True,
        )
    print("DONE", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Ansatz-structure variant study")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--n-list", type=int, nargs="+", default=[6, 8, 10, 12])
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--maxiter", type=int, default=2000)
    p.add_argument("--sigma", type=float, default=0.0)  # reserved (best-of uses random)
    p.add_argument("--seed0", type=int, default=70000)
    p.add_argument(
        "--sat-threshold", type=float, default=0.93, help="Ceiling below this = p=1 no longer saturates (headroom)."
    )
    p.add_argument("--find-saturation", action="store_true")
    p.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Subset of variants to run (default: all). E.g. --variants p1_half_nn p2_base p3_base",
    )
    p.add_argument(
        "--warm-from-best",
        action="store_true",
        help="Legacy alias for --warm-mode best-effort. Warm-start "
        "restart 0 from the best saved theta (same param length).",
    )
    p.add_argument(
        "--warm-mode",
        choices=["off", "all-or-none", "best-effort"],
        default="off",
        help="Warm-start policy for variant comparison. off (default): "
        "no warm-start, homogeneous unbiased comparison. all-or-none: "
        "warm only if EVERY variant has a checkpoint (guarantees "
        "homogeneous treatment). best-effort: warm where available "
        "(may be MIXED — flagged). Use off/all-or-none for a fair "
        "structure comparison; best-effort to maximize each variant's "
        "reached fidelity.",
    )
    args = p.parse_args(argv)
    return run_saturation(args) if args.find_saturation else run_variants(args)


if __name__ == "__main__":
    raise SystemExit(main())
