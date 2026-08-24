#!/usr/bin/env python
"""Circuit Cost Check for Hardware Quench Dynamics.

Builds the full HVA(θ) + Trotter(H(h_post), dt, n_steps) circuit for the
hardware experiment configuration and computes:
- Total 2-qubit gate count (ECR/CX equivalent)
- Circuit depth (2q layers)
- Estimated execution time vs T2 coherence
- Whether QESEM error mitigation is viable at this depth

For heavy_hex topology, the circuit maps natively (zero SWAP overhead).

Usage:
    python scripts/analysis/circuit_cost_check.py \
        --topology heavy_hex --n-qubits 51 --p-layers 1 \
        --h-prep 3.0 --h-quench 0.5 --dt 0.2 --trotter-steps 15 --save

    # Sweep trotter steps to find optimal depth
    python scripts/analysis/circuit_cost_check.py \
        --topology heavy_hex --n-qubits 51 \
        --trotter-steps 10 12 15 18 20 --save
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger(__name__)

# IBM Heron r2 hardware parameters (2026)
ECR_TIME_NS = 84  # ECR gate duration in nanoseconds
T2_MEDIAN_US = 200  # Median T2 in microseconds (Heron r2)
MEAN_2Q_ERROR = 0.004  # Mean ECR error rate (~0.3-0.5%)
QESEM_MAX_T2_RATIO = 4.0  # QESEM viable up to 4x T2 (IBM benchmark)


@dataclass
class CircuitCostResult:
    """Cost analysis for a single circuit configuration."""

    n_qubits: int
    topology: str
    p_layers: int
    h_prep: float
    h_quench: float
    dt: float
    trotter_steps: int
    # Gate counts
    n_rzz_hva: int  # RZZ gates in HVA preparation
    n_rzz_trotter_step: int  # RZZ per Trotter step
    n_rzz_total: int  # Total RZZ (HVA + all Trotter steps)
    n_ecr_estimated: int  # ECR equivalent (2 ECR per RZZ for native heavy-hex)
    # Depth and timing
    circuit_depth_2q: int  # Approximate 2q depth
    estimated_time_us: float  # Estimated circuit time
    t2_budget_ratio: float  # time / T2
    # Viability
    fits_qesem: bool  # t2_ratio < QESEM_MAX (4.0)
    fits_strict: bool  # t2_ratio < 2.0 (unmitigated)
    error_budget: float  # n_ecr * mean_2q_error (total expected error)


def compute_circuit_cost(
    topology: str,
    n_qubits: int,
    p_layers: int = 1,
    h_prep: float = 3.0,
    h_quench: float = 0.5,
    dt: float = 0.2,
    trotter_steps: int = 15,
) -> CircuitCostResult:
    """Compute circuit cost for HVA + Trotter experiment.

    Parameters
    ----------
    topology : str
        Lattice topology (heavy_hex is native → zero SWAP overhead).
    n_qubits : int
        System size.
    p_layers : int
        HVA depth.
    h_prep : float
        Preparation field (HVA ground state).
    h_quench : float
        Post-quench field (Trotter evolution).
    dt : float
        Trotter time step.
    trotter_steps : int
        Number of Trotter steps.

    Returns
    -------
    CircuitCostResult
    """
    from qmbp_simulation.circuits import HVACircuitBuilder
    from qmbp_simulation.circuits.trotter import build_trotter_step
    from qmbp_simulation.models.hamiltonian import make_lattice

    # Build HVA circuit to count gates
    lattice = make_lattice(topology, n_qubits, J=1.0, h=h_prep)
    hva = HVACircuitBuilder()
    hva_circuit, _ = hva.create_bond_resolved(n_qubits, p_layers, lattice)

    # Count RZZ in HVA
    hva_ops = hva_circuit.count_ops()
    n_rzz_hva = hva_ops.get("rzz", 0)

    # Build one Trotter step to count gates
    lattice_quench = make_lattice(topology, n_qubits, J=1.0, h=h_quench)
    trotter_qc = build_trotter_step(lattice_quench, dt, order=2, model="tfim")
    trotter_ops = trotter_qc.count_ops()
    n_rzz_trotter = trotter_ops.get("rzz", 0)

    # Total RZZ
    n_rzz_total = n_rzz_hva + n_rzz_trotter * trotter_steps

    # ECR conversion: for native heavy-hex topology, each RZZ maps to
    # exactly 2 ECR gates (decomposition: RZZ(θ) = ECR · Rz(θ) · ECR)
    # No SWAP overhead because edges match hardware connectivity.
    n_ecr = n_rzz_total * 2

    # Depth estimation: heavy-hex has max degree 3, so edges can be
    # parallelized into ~3 coloring layers per RZZ layer.
    # HVA p=1: 1 RZZ layer, Trotter 2nd order: 2 RZZ layers per step
    n_edges = len(lattice.edges)
    # Graph coloring gives ~3 parallel layers for heavy-hex
    layers_per_rzz_layer = 3  # heavy-hex chromatic index
    hva_depth_2q = p_layers * layers_per_rzz_layer
    trotter_depth_per_step = 2 * layers_per_rzz_layer  # 2nd order: 2 ZZ layers
    total_depth_2q = hva_depth_2q + trotter_depth_per_step * trotter_steps

    # Time estimation
    estimated_time_us = total_depth_2q * ECR_TIME_NS / 1000.0  # ns → μs

    # T2 budget
    t2_ratio = estimated_time_us / T2_MEDIAN_US

    # Error budget
    error_budget = n_ecr * MEAN_2Q_ERROR

    return CircuitCostResult(
        n_qubits=n_qubits,
        topology=topology,
        p_layers=p_layers,
        h_prep=h_prep,
        h_quench=h_quench,
        dt=dt,
        trotter_steps=trotter_steps,
        n_rzz_hva=n_rzz_hva,
        n_rzz_trotter_step=n_rzz_trotter,
        n_rzz_total=n_rzz_total,
        n_ecr_estimated=n_ecr,
        circuit_depth_2q=total_depth_2q,
        estimated_time_us=estimated_time_us,
        t2_budget_ratio=t2_ratio,
        fits_qesem=t2_ratio < QESEM_MAX_T2_RATIO,
        fits_strict=t2_ratio < 2.0,
        error_budget=error_budget,
    )


def print_cost_report(results: list[CircuitCostResult]) -> None:
    """Pretty-print circuit cost analysis."""
    if not results:
        return

    r0 = results[0]
    print(f"\n{'='*70}")
    print(f"  Circuit Cost Check: {r0.topology} N={r0.n_qubits} p={r0.p_layers}")
    print(f"  HVA(h={r0.h_prep}) + Trotter(h={r0.h_quench}, dt={r0.dt})")
    print(f"{'='*70}")

    print(f"\n  {'Steps':>5} | {'RZZ_tot':>7} | {'ECR_est':>7} | {'Depth2q':>7} | {'Time(μs)':>8} | {'T2 ratio':>8} | {'QESEM?':>6} | {'Err budget':>10}")
    print(f"  {'-'*75}")
    for r in results:
        qesem = "✓" if r.fits_qesem else "✗"
        print(
            f"  {r.trotter_steps:>5} | {r.n_rzz_total:>7} | {r.n_ecr_estimated:>7} | "
            f"{r.circuit_depth_2q:>7} | {r.estimated_time_us:>8.1f} | "
            f"{r.t2_budget_ratio:>8.2f} | {qesem:>6} | {r.error_budget:>10.2f}"
        )

    # Summary
    print(f"\n  HVA contribution: {r0.n_rzz_hva} RZZ ({r0.n_rzz_hva*2} ECR)")
    print(f"  Trotter per step: {r0.n_rzz_trotter_step} RZZ ({r0.n_rzz_trotter_step*2} ECR)")
    print(f"  T2 median: {T2_MEDIAN_US} μs, ECR time: {ECR_TIME_NS} ns")
    print(f"  QESEM viable: T2 ratio < {QESEM_MAX_T2_RATIO}")

    # Recommendation
    viable = [r for r in results if r.fits_qesem]
    if viable:
        best = min(viable, key=lambda r: r.trotter_steps)
        max_steps = max(r.trotter_steps for r in viable)
        print(f"\n  RECOMMENDATION: Up to {max_steps} Trotter steps viable with QESEM")
        print(f"  Conservative: {best.trotter_steps} steps (T2 ratio={best.t2_budget_ratio:.2f})")
    else:
        print(f"\n  ⚠️ NO configuration fits within QESEM budget at N={r0.n_qubits}")


def save_report(results: list[CircuitCostResult], out_path: Path) -> None:
    """Save to JSON."""
    from qmbp_simulation.utils.helpers import json_serialize

    output = {
        "topology": results[0].topology,
        "n_qubits": results[0].n_qubits,
        "hardware_params": {
            "ecr_time_ns": ECR_TIME_NS,
            "t2_median_us": T2_MEDIAN_US,
            "mean_2q_error": MEAN_2Q_ERROR,
            "qesem_max_t2_ratio": QESEM_MAX_T2_RATIO,
        },
        "results": [
            {
                "trotter_steps": r.trotter_steps,
                "n_rzz_total": r.n_rzz_total,
                "n_ecr_estimated": r.n_ecr_estimated,
                "circuit_depth_2q": r.circuit_depth_2q,
                "estimated_time_us": r.estimated_time_us,
                "t2_budget_ratio": r.t2_budget_ratio,
                "fits_qesem": r.fits_qesem,
                "fits_strict": r.fits_strict,
                "error_budget": r.error_budget,
            }
            for r in results
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=json_serialize)
    print(f"\n  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Circuit cost check for hardware quench")
    parser.add_argument("--topology", type=str, default="heavy_hex")
    parser.add_argument("--n-qubits", type=int, default=51)
    parser.add_argument("--p-layers", type=int, default=1)
    parser.add_argument("--h-prep", type=float, default=3.0)
    parser.add_argument("--h-quench", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--trotter-steps", type=int, nargs="+", default=[10, 12, 15, 18, 20])
    parser.add_argument("--save", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    results = []
    for steps in args.trotter_steps:
        r = compute_circuit_cost(
            topology=args.topology,
            n_qubits=args.n_qubits,
            p_layers=args.p_layers,
            h_prep=args.h_prep,
            h_quench=args.h_quench,
            dt=args.dt,
            trotter_steps=steps,
        )
        results.append(r)

    print_cost_report(results)

    if args.save:
        out_path = _project_root / "results" / "analysis" / f"circuit_cost_{args.topology}_N{args.n_qubits}.json"
        save_report(results, out_path)


if __name__ == "__main__":
    main()
