#!/usr/bin/env python
"""Ansatz expressivity ceiling on 2D lattices (VQE-optimal fidelity).

For each (topology, N, h, p) this optimizes the bond-resolved HVA with VQE and
measures how close the BEST ansatz state gets to the exact ground state:
exact fidelity F(|ψ_HVA(θ_opt)⟩, |ψ_exact⟩) and ΔE/gap. This is the ansatz
expressivity ceiling — the intrinsic limit of the ansatz, independent of any
predictor. Comparing p=1 vs p=2 shows whether depth recovers expressivity on
the more-connected 2D geometries.

Statevector-exact (N ≤ STATEVECTOR_MAX_N).

Usage:
    .venv/bin/python scripts/analysis/ansatz_expressivity_2d.py \
        --topologies triangular square --n 10 --p 1 2 --h 1.0 2.5 \
        --out results/analysis/ansatz_expressivity_2d.json
"""

from __future__ import annotations

import argparse

import numpy as np
from hva_vl_study_common import exact_ground_state_vector, save_json, state_fidelity_exact, study_dir


def expressivity_ceiling(
    topology: str,
    n: int,
    h: float,
    p_layers: int,
    vqe_restarts: int = 5,
    vqe_maxiter: int = 400,
) -> dict:
    from qmbp_simulation import (
        HamiltonianBuilder,
        HVACircuitBuilder,
        VQEConfig,
        VQEOptimizer,
        make_lattice,
    )

    lat = make_lattice(topology, n, J=1.0, h=h)
    # Shared solve: cached (E0, gap) via GroundTruthCache + exact vector. tfim
    # spec build_hamiltonian IS HamiltonianBuilder().build, so VQE optimizes the
    # same operator that defines the ground state used for fidelity.
    psi_exact, e0, gap = exact_ground_state_vector(topology, n, h, model="tfim")
    H = HamiltonianBuilder().build(lat)
    circuit, _ = HVACircuitBuilder().create_bond_resolved(n, p_layers, lat)

    rng = np.random.default_rng(42)
    init = rng.uniform(-0.1, 0.1, circuit.num_parameters)
    cfg = VQEConfig(n_restarts=vqe_restarts, maxiter=vqe_maxiter, p_layers=p_layers)
    res = VQEOptimizer(config=cfg, seed=42).optimize(H, circuit, init)

    fid = state_fidelity_exact(circuit, res.theta_opt, psi_exact)
    return {
        "topology": topology, "N": n, "h": h, "p_layers": p_layers,
        "n_params": int(circuit.num_parameters),
        "E0": e0, "gap": gap,
        "ceil_E": float(res.energy),
        "ceil_dE": abs(float(res.energy) - e0),
        "ceil_de_gap": abs(float(res.energy) - e0) / max(gap, 1e-10),
        "ceil_fidelity": fid,
    }


def run(topologies, n_values, h_values, p_values, filename="ansatz_expressivity_2d.json", **vqe_kw) -> list[dict]:
    rows: list[dict] = []
    for topo in topologies:
        for n in n_values:
            for h in h_values:
                for p in p_values:
                    try:
                        rows.append(expressivity_ceiling(topo, n, h, p, **vqe_kw))
                    except Exception as exc:  # noqa: BLE001 — robustness
                        rows.append(
                            {"topology": topo, "N": n, "h": h, "p_layers": p,
                             "error": f"{type(exc).__name__}: {exc}"}
                        )
                    save_json(
                        {"rows": rows}, "expressivity", filename,
                        description="Ansatz expressivity ceiling (VQE-optimal fidelity) on 2D lattices",
                    )
    return rows


def format_rows(rows: list[dict]) -> str:
    out = ["=" * 74]
    for r in rows:
        if "error" in r:
            out.append(f"{r['topology']:<11} N={r['N']} h={r['h']} p={r['p_layers']}: ERR {r['error'][:40]}")
            continue
        out.append(
            f"{r['topology']:<11} N={r['N']} h={r['h']:<4} p={r['p_layers']}  "
            f"ceil_fid={r['ceil_fidelity']:.4f}  ΔE/gap={r['ceil_de_gap']:.3f}  "
            f"params={r['n_params']}"
        )
    out.append("=" * 74)
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ansatz expressivity ceiling on 2D lattices")
    p.add_argument("--topologies", nargs="+", default=["triangular", "square"])
    p.add_argument("--n", type=int, nargs="+", default=[10])
    p.add_argument("--p", type=int, nargs="+", default=[1, 2])
    p.add_argument("--h", type=float, nargs="+", default=[1.0, 2.5])
    p.add_argument("--vqe-restarts", type=int, default=5)
    p.add_argument("--vqe-maxiter", type=int, default=400)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = run(
        args.topologies, args.n, args.h, args.p,
        vqe_restarts=args.vqe_restarts, vqe_maxiter=args.vqe_maxiter,
    )
    print(format_rows(rows))
    print(f"\n→ Resultados: {study_dir('expressivity') / 'ansatz_expressivity_2d.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
