#!/usr/bin/env python
"""h-sweep of the HVA +NNN (frustrated bond-resolved) ansatz, p=2, frustrated TFIM.

For each (topology, h) with fixed J2, builds the +NNN bond-resolved HVA
(create_bond_resolved_frustrated — nearest + next-nearest RZZ + RX), finds its
best fidelity to the exact ground state via robust multi-restart VQE, and records
energy accuracy (ΔE/gap) plus circuit resources (2q, depth_2q) and an estimated
NISQ fidelity. Consolidates everything into one JSON, persisted incrementally
(crash-safe), and prints a summary table.

Reuses (from hva_vl_study_common):
- `exact_ground_state_vector` (cached E0/gap via GroundTruthCache + exact vector)
- `robust_vqe_fidelity` (multi-restart expressivity-ceiling fidelity)
- `circuit_stats` / `n_2q` / `transpile_hw` (shared hardware-basis stats)
- `save_json` incremental-persist pattern
Plus `compute_error_budget` from analysis.circuit_visualizer (NISQ fidelity estimate).

Usage:
    .venv/bin/python scripts/analysis/hva_nnn_h_sweep_frustrated.py \
        --topologies square triangular --n 9 --p 2 --j2 0.5 --h 0.5 1.0 2.0 3.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import (  # noqa: E402 (same dir)
    circuit_stats,
    n_2q,
    robust_vqe_fidelity,
    save_json,
    study_dir,
    transpile_hw,
    warmstart_metropolis_vqe,
)

# Ansatz variants under test. "nn" = nearest-neighbor bond-resolved (cheapest in
# 2q); "nnn" = + next-nearest-neighbor RZZ (maps the J2 frustration, more 2q).
ANSATZE = {
    "nn": ("create_bond_resolved", "hva_nn (create_bond_resolved)"),
    "nnn": ("create_bond_resolved_frustrated", "hva_nnn (create_bond_resolved_frustrated)"),
}


def _point(topology, n, p, j2, h, model, n_random, maxiter, ansatz="nnn") -> dict:
    from qmbp_simulation import make_lattice
    from qmbp_simulation.analysis.circuit_visualizer import compute_error_budget
    from qmbp_simulation.framework.study_runner import StudyRunner
    from qmbp_simulation.models.constants import STATEVECTOR_MAX_N

    # Ground state + circuit via the shared StudyRunner: single source of truth
    # for the cached/exact (N<=16) vs eigsh (16<N<=22) ground state and the
    # bond-resolved (+NNN) ansatz build. Same numbers as the pre-migration path.
    runner = StudyRunner(
        topology=topology,
        n_qubits=n,
        p_layers=p,
        j2=j2,
        model=model,
        strategy="metropolis",
        maxiter=maxiter,
        ansatz=ansatz,
    )
    lat = make_lattice(topology, n, J=1.0, h=h)
    psi, e0, gap, H, gs_method = runner.ground_state(h)
    qc, n_nn, n_nnn = runner.build_circuit(lat)
    ansatz_label = ANSATZE[ansatz][1]

    # Optimizer choice by size. For N<=16 pure random restarts are viable and
    # measure the expressivity ceiling. For N>16 (168 params at N=18) random
    # collapses ~40% of the time near the transition, so seed from the analytic
    # warm-start and perturb (best-of) — see REPORT_HEURISTIC_warmstart_restarts.md.
    # zeros is a trivial stationary point for the bond-resolved ansätze.
    warmstart_runs = None
    if n <= STATEVECTOR_MAX_N:
        e_best, bound, fid, best_theta = robust_vqe_fidelity(
            qc,
            H,
            psi,
            n_random=n_random,
            maxiter=maxiter,
            include_zeros=False,
            return_theta=True,
        )
        optimizer = f"robust_random_x{n_random}"
    else:
        e_best, bound, fid, warmstart_runs, best_theta = warmstart_metropolis_vqe(
            qc,
            H,
            psi,
            n_nn=n_nn,
            n_nnn=(n_nnn if ansatz == "nnn" else 0),
            n_qubits=n,
            p_layers=p,
            h=h,
            J=1.0,
            J2=(j2 if ansatz == "nnn" else 0.0),
            maxiter=maxiter,
            temperature=gap,
        )
        optimizer = "warmstart+metropolis_basinhop"

    t = transpile_hw(bound)
    stats = circuit_stats(t)
    n2q = n_2q(t)
    budget = compute_error_budget(t, backend=None)

    return {
        "topology": topology,
        "N": n,
        "p_layers": p,
        "model": model,
        "J2": j2,
        "h": h,
        "ansatz": ansatz,
        "ansatz_desc": ansatz_label,
        "gs_method": gs_method,
        "E0": e0,
        "gap": gap,
        "n_nn_edges": len(lat.edges),
        "energy": e_best,
        "abs_error": abs(e_best - e0),
        "de_gap": abs(e_best - e0) / gap if gap > 0 else None,
        "fidelity": fid,
        "n_2q_transpiled": n2q,
        "depth_transpiled": stats["depth"],
        "depth_2q": stats["depth_2q"],
        "n_params": qc.num_parameters,
        "nisq_fidelity_est": budget.get("fidelity_estimate"),
        "optimizer": optimizer,
        "warmstart_runs": warmstart_runs,
        # Optimal angles persisted so every sweep row is reproducible/reusable
        # (reload θ → assign_parameters → exact state, no re-optimization).
        "theta_final": np.asarray(best_theta, float).tolist(),
    }


_OUT_FILE = "hva_ansatze_h_sweep_frustrated.json"


def run(topologies, n, p_values, j2, h_values, model, n_random, maxiter, ansatze) -> list[dict]:
    rows: list[dict] = []
    for topo in topologies:
        for h in h_values:
            for ansatz in ansatze:
                for p in p_values:
                    try:
                        rows.append(_point(topo, n, p, j2, h, model, n_random, maxiter, ansatz=ansatz))
                        print(f"  done: {topo} h={h} {ansatz} p={p}", flush=True)
                    except Exception as exc:  # noqa: BLE001 — never abort the batch
                        rows.append(
                            {
                                "topology": topo,
                                "N": n,
                                "h": h,
                                "J2": j2,
                                "ansatz": ansatz,
                                "p_layers": p,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        print(f"  FAILED: {topo} h={h} {ansatz} p={p}: {exc}", flush=True)
                    save_json(
                        {"schema": "hva_ansatze_h_sweep_frustrated_v1", "rows": rows},
                        "hva_nnn_sweep",
                        _OUT_FILE,
                        params={
                            "N": n,
                            "p_layers": p_values,
                            "J2": j2,
                            "model": model,
                            "ansatze": ansatze,
                            "n_random_restarts": n_random,
                            "maxiter": maxiter,
                        },
                        description="h-sweep of NN and NN+NNN bond-resolved HVA ansätze, "
                        "frustrated 2D TFIM (analog to the VL report)",
                    )
    return rows


def format_rows(rows) -> str:
    out = [
        "=" * 96,
        f"{'topo':<10}{'h':<5}{'ansatz':<6}{'p':<3}{'gap':<8}{'fidelity':<10}"
        f"{'dE/gap':<9}{'2q':<5}{'depth2q':<8}{'NISQ':<7}{'nP':<4}",
        "-" * 96,
    ]
    for r in rows:
        if "error" in r:
            out.append(
                f"{r['topology']:<10}{r['h']:<5}{r.get('ansatz', ''):<6}"
                f"{r.get('p_layers', ''):<3}ERROR: {r['error'][:40]}"
            )
            continue

        def f(x, w=".4f"):
            return format(x, w) if isinstance(x, (int, float)) else str(x)

        out.append(
            f"{r['topology']:<10}{r['h']:<5}{r['ansatz']:<6}{r['p_layers']:<3}"
            f"{f(r['gap']):<8}{f(r['fidelity']):<10}{f(r['de_gap']):<9}"
            f"{str(r['n_2q_transpiled']):<5}{str(r['depth_2q']):<8}"
            f"{f(r['nisq_fidelity_est']):<7}{str(r['n_params']):<4}"
        )
    out.append("=" * 96)
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="h-sweep of NN and NN+NNN HVA ansätze, frustrated TFIM (VL-report analog)")
    p.add_argument("--topologies", nargs="+", default=["square"])
    p.add_argument("--n", type=int, default=9)
    p.add_argument("--p", type=int, nargs="+", default=[1, 2])
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--h", type=float, nargs="+", default=[0.5, 1.0])
    p.add_argument("--ansatze", nargs="+", default=["nn", "nnn"], choices=["nn", "nnn"])
    p.add_argument("--model", default="tfim_frustrated")
    p.add_argument("--n-random", type=int, default=4, help="random VQE restarts")
    p.add_argument("--maxiter", type=int, default=400)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        f"[hva_sweep] topos={args.topologies} h={args.h} ansatze={args.ansatze} p={args.p} J2={args.j2} N={args.n}",
        flush=True,
    )
    rows = run(args.topologies, args.n, args.p, args.j2, args.h, args.model, args.n_random, args.maxiter, args.ansatze)
    print("\n" + format_rows(rows))
    print(f"\n→ Resultados: {study_dir('hva_nnn_sweep') / _OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
