#!/usr/bin/env python
"""Native-routing 2q budget for square-frustrated dynamics on Nighthawk.

Measures how much of the previously-quoted ×6 transpiled-2q overhead is a generic
transpilation artefact versus the real cost on Nighthawk's native square-lattice
coupling map. Decomposes the overhead into:

  - nn-only layer  → native square edges, no SWAP; overhead ≈ 2.0× (RZZ → 2·CZ).
  - nn+nnn layer   → the diagonal nnn RZZ need SWAP routing; overhead > 2.0×.

The routing excess (full − nn) is the part a native-nnn hardware or an nnn-free
model would avoid. Uses the REAL FakeNighthawk coupling map (its connectivity is
correct; only its noise model is not) with the native CZ basis — no noise model.

Also projects the quantum-advantage 2q budget with the measured overhead against
the Nighthawk gate limits (noise ~333 2q at F=1/e for median CZ; ~5000 arch.).

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/run_native_routing_budget.py \
        --n-values 10 12 14 16 18 --h2 2.5 --j2 0.5
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from qiskit.circuit import QuantumCircuit

from qmbp_simulation import make_lattice
from qmbp_simulation.analysis.dynamics import (
    NIGHTHAWK_CZ_ERROR,
    gates_to_floor,
    nighthawk_coupling_map,
    routing_overhead_breakdown,
    square_grid_coupling_map,
)
from qmbp_simulation.circuits.trotter import _apply_zz_layer
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder
from qmbp_simulation.utils.helpers import write_json_atomic

_REPO = Path(__file__).resolve().parents[3]
builder = HamiltonianBuilder()


def _layers(n, nn_edges, nnn_edges, j2, dt):
    """One nn-only ZZ layer and one nn+nnn ZZ layer (frustrated couplings)."""
    nn_qc = QuantumCircuit(n)
    _apply_zz_layer(nn_qc, nn_edges, 1.0, dt)
    full_qc = QuantumCircuit(n)
    _apply_zz_layer(full_qc, nn_edges, 1.0, dt)
    _apply_zz_layer(full_qc, nnn_edges, -j2, dt)
    return nn_qc, full_qc


def run(args) -> int:
    cmap = nighthawk_coupling_map()
    cmap_source = "FakeNighthawk (real square coupling map)"
    if cmap is None:
        cmap_source = "square_grid fallback (per-N)"

    print(f"[native_routing] coupling map: {cmap_source}", flush=True)
    rows = []
    for n in args.n_values:
        lat = make_lattice("square", n, J=1.0, h=args.h2)
        nn_edges = list(lat.edges)
        nnn_edges = builder._generate_nnn_edges(lat)
        nn_qc, full_qc = _layers(n, nn_edges, nnn_edges, args.j2, args.dt)
        cm = cmap if cmap is not None else square_grid_coupling_map(n)
        bd = routing_overhead_breakdown(nn_qc, full_qc, cm)
        bd["n_qubits"] = n
        bd["n_nn"] = len(nn_edges)
        bd["n_nnn"] = len(nnn_edges)
        rows.append(bd)
        print(f"  N={n:>2}: nn {bd['logical_nn']}→{bd['transpiled_nn']} "
              f"({bd['overhead_nn']:.2f}x) | nn+nnn {bd['logical_full']}→"
              f"{bd['transpiled_full']} ({bd['overhead_full']:.2f}x) | "
              f"routing excess {bd['routing_excess']:.2f}", flush=True)

    # Advantage-budget projection with the measured full overhead at the largest N.
    noise_floor_2q = gates_to_floor(args.eps_2q)
    big = rows[-1]
    overhead = big["overhead_full"]
    budget = {
        "overhead_full_measured": overhead,
        "noise_floor_2q_gates": noise_floor_2q,
        "eps_2q": args.eps_2q,
        "note": (
            f"nn-only overhead ≈2.0 is RZZ→2·CZ (native, no SWAP); "
            f"routing excess ≈{big['routing_excess']:.2f} is the diagonal-nnn SWAP cost. "
            f"Noise floor ~{noise_floor_2q} 2q at F=1/e (median CZ, no mitigation)."
        ),
    }
    doc = {
        "schema": "native_routing_budget_v1",
        "status": "complete",
        "specs": {
            "model": "tfim_frustrated", "topology": "square", "j2": args.j2,
            "h2": args.h2, "dt": args.dt, "coupling_map": cmap_source,
            "basis_gates": ["rz", "sx", "x", "cz"],
        },
        "rows": rows,
        "advantage_budget": budget,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    subdir = _REPO / "results/experiments/exp_frustrated/native_routing_budget"
    subdir.mkdir(parents=True, exist_ok=True)
    out = subdir / f"native_routing_budget_{doc['timestamp']}.json"
    write_json_atomic(out, doc)
    print(f"\n→ {out.relative_to(_REPO)}", flush=True)
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-values", type=int, nargs="+", default=[10, 12, 14, 16, 18])
    p.add_argument("--h2", type=float, default=2.5)
    p.add_argument("--j2", type=float, default=0.5)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--eps-2q", type=float, default=NIGHTHAWK_CZ_ERROR)
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
