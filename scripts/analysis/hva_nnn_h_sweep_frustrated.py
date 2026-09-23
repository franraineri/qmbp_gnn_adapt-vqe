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

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import (  # noqa: E402 (same dir)
    circuit_stats,
    exact_ground_state_vector,
    n_2q,
    robust_vqe_fidelity,
    save_json,
    study_dir,
    transpile_hw,
)


def _point(topology, n, p, j2, h, model, n_random, maxiter) -> dict:
    from qmbp_simulation import HVACircuitBuilder, make_lattice
    from qmbp_simulation.analysis.circuit_visualizer import compute_error_budget
    from qmbp_simulation.models.model_registry import get_model_spec

    spec = get_model_spec(model)
    ham_kwargs = dict(getattr(spec, "hamiltonian_kwargs", {}))
    if "J2" in ham_kwargs:
        ham_kwargs["J2"] = j2

    lat = make_lattice(topology, n, J=1.0, h=h)
    H = spec.build_hamiltonian(lat, **ham_kwargs)
    # Shared solve: cached (E0, gap) via GroundTruthCache + exact vector
    # (frustrated forwards J2 so the cache key stays distinct from J2=0).
    psi, e0, gap = exact_ground_state_vector(topology, n, h, model=model, j2=j2)

    qc, _ = HVACircuitBuilder().create_bond_resolved_frustrated(n, p, lat)
    # zeros is a trivial stationary point for the +NNN ansatz — random restarts only.
    e_best, bound, fid = robust_vqe_fidelity(
        qc, H, psi, n_random=n_random, maxiter=maxiter, include_zeros=False
    )

    t = transpile_hw(bound)
    stats = circuit_stats(t)
    n2q = n_2q(t)
    budget = compute_error_budget(t, backend=None)

    return {
        "topology": topology, "N": n, "p_layers": p, "model": model, "J2": j2, "h": h,
        "ansatz": "hva_nnn (create_bond_resolved_frustrated)",
        "E0": e0, "gap": gap, "n_nn_edges": len(lat.edges),
        "energy": e_best, "abs_error": abs(e_best - e0),
        "de_gap": abs(e_best - e0) / gap if gap > 0 else None,
        "fidelity": fid,
        "n_2q_transpiled": n2q, "depth_transpiled": stats["depth"],
        "depth_2q": stats["depth_2q"], "n_params": qc.num_parameters,
        "nisq_fidelity_est": budget.get("fidelity_estimate"),
    }


def run(topologies, n, p, j2, h_values, model, n_random, maxiter) -> list[dict]:
    rows: list[dict] = []
    for topo in topologies:
        for h in h_values:
            try:
                rows.append(_point(topo, n, p, j2, h, model, n_random, maxiter))
                print(f"  done: {topo} h={h}", flush=True)
            except Exception as exc:  # noqa: BLE001 — never abort the batch
                rows.append({"topology": topo, "N": n, "h": h, "J2": j2,
                             "error": f"{type(exc).__name__}: {exc}"})
                print(f"  FAILED: {topo} h={h}: {exc}", flush=True)
            save_json(
                {"schema": "hva_nnn_h_sweep_frustrated_v1", "rows": rows},
                "hva_nnn_sweep", "hva_nnn_h_sweep_frustrated.json",
                params={"N": n, "p_layers": p, "J2": j2, "model": model,
                        "n_random_restarts": n_random, "maxiter": maxiter},
                description="h-sweep of the +NNN bond-resolved HVA ansatz, frustrated 2D TFIM",
            )
    return rows


def format_rows(rows) -> str:
    out = ["=" * 88,
           f"{'topo':<11}{'h':<6}{'gap':<9}{'fidelity':<10}{'dE/gap':<9}"
           f"{'2q':<5}{'depth2q':<8}{'NISQ':<7}",
           "-" * 88]
    for r in rows:
        if "error" in r:
            out.append(f"{r['topology']:<11}{r['h']:<6}ERROR: {r['error'][:45]}")
            continue
        def f(x, w=".4f"):
            return format(x, w) if isinstance(x, (int, float)) else str(x)
        out.append(
            f"{r['topology']:<11}{r['h']:<6}{f(r['gap']):<9}{f(r['fidelity']):<10}"
            f"{f(r['de_gap']):<9}{str(r['n_2q_transpiled']):<5}"
            f"{str(r['depth_2q']):<8}{f(r['nisq_fidelity_est']):<7}"
        )
    out.append("=" * 88)
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="h-sweep HVA+NNN ansatz, frustrated TFIM")
    p.add_argument("--topologies", nargs="+", default=["square", "triangular"])
    p.add_argument("--n", type=int, default=9)
    p.add_argument("--p", type=int, default=2)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--h", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0])
    p.add_argument("--model", default="tfim_frustrated")
    p.add_argument("--n-random", type=int, default=6, help="random VQE restarts (+ zeros)")
    p.add_argument("--maxiter", type=int, default=600)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[hva_nnn_sweep] topos={args.topologies} h={args.h} J2={args.j2} "
          f"N={args.n} p={args.p}", flush=True)
    rows = run(args.topologies, args.n, args.p, args.j2, args.h, args.model,
               args.n_random, args.maxiter)
    print("\n" + format_rows(rows))
    print(f"\n→ Resultados: {study_dir('hva_nnn_sweep') / 'hva_nnn_h_sweep_frustrated.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
