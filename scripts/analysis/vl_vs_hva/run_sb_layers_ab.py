#!/usr/bin/env python
"""A/B: HVA puro vs HVA + symmetry-breaking layer (bloque RY por-sitio).

Motivado por Park, *Efficient ground state preparation in VQE with symmetry-
breaking layers* (arXiv:2106.02509): el HVA puro comparte la simetría del
Hamiltoniano (RX + RZZ preservan la Z2 de spin-flip), así que NO alcanza en
profundidad constante los estados fundamentales con simetría ROTA que prevalecen
en la fase frustrada. Agregar una capa que rompe la simetría — aquí un bloque
``"y"`` (RY por sitio, 0 CX extra) — debería permitir mayor fidelidad al MISMO
número de compuertas de 2 qubits.

Este runner compara, para cada estructura base, dos circuitos al 2q IDÉNTICO:
  pure : la variante del registro (p.ej. p1_half_nn_rx)
  sb   : la misma + bloque RY final (p1_half_nn_rx_y) — mismo 2q, +N params

Ambos arms son bond-resolved (per-bond, SIN tying — atar contradiría el objetivo
de romper la simetría). El seed del bloque y es 0.0 (identidad): arranca en el
punto simétrico y el optimizador descubre la dirección de ruptura.

Reutiliza (sin reimplementar): ground_state, cached_degenerate_ground_manifold,
subspace_fidelity, prepare_warmstart, converge_circuit, cx_and_params,
build_variant + VARIANTS, StudyPersister, save_ansatz_spec, sync_scoreboard.

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/run_sb_layers_ab.py \
        --n 9 --h 0.5 --pairs p1_half_nn_rx p2_half_nn_rx --restarts 4 --maxiter 400
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

from hva_vl_study_common import StudyPersister, save_ansatz_spec, sync_scoreboard  # noqa: E402

from qmbp_simulation.analysis.fidelity import (  # noqa: E402
    cached_degenerate_ground_manifold,
    subspace_fidelity,
)
from qmbp_simulation.circuits import HVACircuitBuilder  # noqa: E402
from qmbp_simulation.circuits.hva_variants import VARIANTS, build_variant  # noqa: E402
from qmbp_simulation.circuits.symmetry import detect_symmetry_group  # noqa: E402
from qmbp_simulation.framework.study_core import (  # noqa: E402
    converge_circuit,
    cx_and_params,
    ground_state,
    prepare_warmstart,
)
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder  # noqa: E402

SUBDIR = "bond_ablation"
MODEL = "tfim_frustrated"


def _sb_name(base: str) -> str:
    """Registry name of the symmetry-breaking counterpart of ``base``."""
    return f"{base}_y"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="A/B HVA puro vs HVA + symmetry-breaking layer")
    p.add_argument("--topology", default="square")
    p.add_argument("--n", type=int, default=9)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument(
        "--pairs",
        nargs="+",
        default=["p1_half_nn_rx", "p2_half_nn_rx"],
        help="Base structures; for each X, runs X (pure) and X_y (sb) at the same 2q.",
    )
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--maxiter", type=int, default=400)
    p.add_argument("--seed0", type=int, default=96000)
    p.add_argument("--target-fid", type=float, default=0.995)
    p.add_argument("--ftol", type=float, default=1e-10)
    p.add_argument("--save-specs", action="store_true")
    args = p.parse_args(argv)

    from qmbp_simulation.execution import NoiselessBackend
    from qmbp_simulation.execution.eval_cache import CachedBackend, EvalCache

    t_start = time.time()
    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(args.topology, args.n, args.h, args.j2, 1)
    nn_edges = [(int(i), int(j)) for i, j in lat.edges]
    nnn_edges = [(int(i), int(j)) for i, j in HamiltonianBuilder._generate_nnn_edges(lat)]
    manifold, _e0m, phys_gap, m_dim = cached_degenerate_ground_manifold(
        H, topology=args.topology, n_qubits=args.n, model=MODEL, h=args.h, j2=args.j2
    )
    degenerate = m_dim > 1
    group = detect_symmetry_group(nn_edges, nnn_edges, args.n)
    print(
        f"[sb_ab] N={args.n} h={args.h} gap={gap:.5f} manifold_dim={m_dim} "
        f"group_order={group.order} degenerate={degenerate} (setup {time.time() - t_start:.1f}s)",
        flush=True,
    )

    builder = HVACircuitBuilder()
    backend = CachedBackend(
        NoiselessBackend(), topology=args.topology, n_qubits=args.n, model=MODEL, p_layers=1, cache=EvalCache()
    )
    backend.set_h(float(args.h))

    rows: list[dict] = []
    spec_sources: dict = {}
    persister = StudyPersister(
        subdir=SUBDIR,
        out_file=f"sb_layers_ab_{args.topology}_N{args.n}_h{args.h:.2f}.json",
        fingerprint={"n_qubits": args.n, "p_layers": None, "model": MODEL, "topology": args.topology, "h": args.h},
        extra={
            "topology": args.topology,
            "N": args.n,
            "h": args.h,
            "J2": args.j2,
            "schema": "sb_layers_ab_v1",
            "manifold_dim": m_dim,
            "group_order": group.order,
        },
        params={"experiment": "sb_layers_ab", "N": args.n, "h": args.h},
        description="A/B HVA pure vs HVA + symmetry-breaking RY layer (same 2q)",
        rows_ref=rows,
    )

    def _subfid(qc, theta):
        return subspace_fidelity(qc, np.asarray(theta, float), manifold) if degenerate else None

    def _run(name, base, arm):
        if name not in VARIANTS:
            print(f"  [skip] {name}: not in VARIANTS registry", flush=True)
            return None
        variant = VARIANTS[name]
        qc, _ = build_variant(builder, args.n, lat, variant)
        n2q, npar = cx_and_params(qc)
        ws = prepare_warmstart(
            qc,
            H,
            psi,
            n_nn=len(nn_edges),
            n_nnn=len(nnn_edges),
            n_qubits=args.n,
            p_layers=1,
            h=args.h,
            J2=args.j2,
            backend=backend,
            topology=args.topology,
            model=MODEL,
            gap=gap,
            target_len=qc.num_parameters,
            auto_donors=False,
            micro_descent=0,
            blocks=list(variant.blocks),
            rx_final=variant.rx_final,
            rz_final=variant.rz_final,
        )
        seed = np.asarray(ws["seed"], float)
        if seed.size != npar or not np.all(np.isfinite(seed)):
            seed = None
        init = _subfid(qc, seed) if seed is not None else None
        t0 = time.time()
        c0 = time.process_time()
        try:
            fid, e_best, runs = converge_circuit(
                qc,
                H,
                psi,
                restarts=args.restarts,
                maxiter=args.maxiter,
                seed0=args.seed0,
                warm_theta=seed,
                backend=backend,
                target_fid=args.target_fid,
                ftol=args.ftol,
            )
        except (IndexError, ValueError, RuntimeError) as exc:
            print(f"  {name:22s} SKIPPED ({type(exc).__name__}: {exc})", flush=True)
            return None
        theta_best = np.asarray(min(runs, key=lambda r: r["energy"])["theta_final"], float)
        sub = _subfid(qc, theta_best)
        report_fid = sub if sub is not None else fid
        row = {
            "label": name,
            "base": base,
            "arm": arm,
            "h": round(float(args.h), 2),
            "n_qubits": args.n,
            "topology": args.topology,
            "e0": e0,
            "gap": gap,
            "phys_gap": phys_gap,
            "manifold_dim": m_dim,
            "degenerate": degenerate,
            "e_best": float(e_best),
            "init_fidelity": init,
            "best_fidelity": float(fid),
            "subspace_fidelity": sub,
            "report_fidelity": float(report_fid),
            "n_2q_transpiled": n2q,
            "n_params": npar,
            "fidelity_per_cx": (report_fid / n2q) if n2q else None,
            "seed_provenance": ws.get("provenance"),
            "theta_best": theta_best.tolist(),
            "cpu_seconds": round(time.process_time() - c0, 1),
            "seconds": round(time.time() - t0, 1),
        }
        rows.append(row)
        spec_sources[name] = (variant, nn_edges, nnn_edges)
        persister.persist(rows)
        init_s = f" init={init:.4f}" if init is not None else ""
        print(
            f"  {name:22s} [{arm:4s}] fid={report_fid:.4f}{init_s} 2q={n2q:3d} p={npar:3d} ({row['cpu_seconds']}s)",
            flush=True,
        )
        return row

    # ── A/B per base structure: pure vs sb at identical 2q ───────────────────
    print("\n=== A/B: HVA pure vs HVA + symmetry-breaking RY layer ===", flush=True)
    for base in args.pairs:
        print(f"\n [{base}]", flush=True)
        _run(base, base, "pure")
        _run(_sb_name(base), base, "sb")

    persister.persist(rows, status="final")
    backend.flush()

    # ── Summary: pure vs sb at the same 2q ───────────────────────────────────
    print("\n=== SUMMARY (pure vs sb @ same 2q) ===", flush=True)
    by_base: dict[str, dict] = {}
    for r in rows:
        by_base.setdefault(r["base"], {})[r["arm"]] = r
    for base, arms in by_base.items():
        pu, sb = arms.get("pure"), arms.get("sb")
        if pu and sb:
            d = sb["report_fidelity"] - pu["report_fidelity"]
            winner = "SB" if d > 1e-4 else ("pure" if d < -1e-4 else "tie")
            print(
                f"  {base:18s} 2q={pu['n_2q_transpiled']:3d}  "
                f"pure={pu['report_fidelity']:.4f}({pu['n_params']}p)  "
                f"sb={sb['report_fidelity']:.4f}({sb['n_params']}p)  "
                f"Δ={d:+.4f} → {winner}",
                flush=True,
            )
        elif pu:
            print(f"  {base:18s} pure={pu['report_fidelity']:.4f} (sb arm missing)", flush=True)

    if args.save_specs and rows:
        n_saved = 0
        for r in rows:
            src = spec_sources.get(r["label"])
            if src is None:
                continue
            variant, sp_nn, sp_nnn = src
            try:
                save_ansatz_spec(
                    variant,
                    topology=args.topology,
                    n_qubits=args.n,
                    h=args.h,
                    j2=args.j2,
                    theta=r["theta_best"],
                    nn_edges=sp_nn,
                    nnn_edges=sp_nnn,
                    fidelity=r["report_fidelity"],
                    n_2q=r["n_2q_transpiled"],
                    n_params=r["n_params"],
                    p_layers=1,
                )
                n_saved += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  [spec save skipped: {r['label']}: {exc}]", flush=True)
        if n_saved:
            print(f"\n[sb_ab] persisted {n_saved} spec(s); refreshing scoreboard", flush=True)
            sync_scoreboard()

    print(f"\nDONE ({time.time() - t_start:.1f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
