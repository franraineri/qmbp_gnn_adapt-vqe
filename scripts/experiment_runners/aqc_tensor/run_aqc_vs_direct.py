#!/usr/bin/env python3
"""AQC-Tensor vs Direct p=1 Comparison — Quantify expressibility benefit.

Compares the energy accuracy of:
1. Direct p=1 HVA (standard hardware deployment)
2. AQC-compressed p=2 → shallow circuit (same 2Q gate count as p=1)

This answers the key question: does AQC-compressed p=2 give better ΔE/gap
than direct p=1 at the same hardware depth?

For heavy_hex N=10 (our hardware target), the POC showed:
- p=2 has better θ_opt (lower ΔE/gap than p=1)
- AQC compression retains most of that advantage (F>0.999)
- 2Q gates: 18→9 (50% reduction), same as p=1 gate count

Usage:
    python scripts/experiment_runners/aqc_tensor/run_aqc_vs_direct.py
    python scripts/experiment_runners/aqc_tensor/run_aqc_vs_direct.py --topology heavy_hex
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from qmbp_simulation import (
    ClassicalSolver,
    HamiltonianBuilder,
    HVACircuitBuilder,
    VQEConfig,
    VQEOptimizer,
    make_lattice,
)
from qmbp_simulation.circuits.aqc_compression import (
    AQCCircuitCompressor,
    AQCCompressionConfig,
)
from qmbp_simulation.utils.helpers import json_dump


def run_comparison(
    topology: str = "heavy_hex",
    n_qubits: int = 10,
    h_values: list[float] | None = None,
    bond_dim: int = 64,
    seeds: list[int] | None = None,
) -> dict:
    """Run AQC-compressed p=2 vs direct p=1 comparison."""
    if h_values is None:
        h_values = [3.25, 3.5, 3.75, 4.0]
    if seeds is None:
        seeds = DEFAULT_SEEDS

    print("=" * 70)
    print("  AQC-COMPRESSED p=2 vs DIRECT p=1 COMPARISON")
    print(f"  Topology: {topology}, N={n_qubits}")
    print(f"  h-values: {h_values}, seeds: {seeds}")
    print("=" * 70)

    circuit_builder = HVACircuitBuilder()
    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    config = AQCCompressionConfig(max_bond_dim=bond_dim, max_iterations=200)
    compressor = AQCCircuitCompressor(config)

    all_results = []

    for h in h_values:
        for seed in seeds:
            lattice = make_lattice(topology, n_qubits, J=1.0, h=h)
            H = builder.build(lattice)
            exact = solver.solve(H, lattice)
            rng = np.random.default_rng(seed)

            # ── Direct p=1 ────────────────────────────────────────────────
            circuit_p1, _ = circuit_builder.create(n_qubits, 1, lattice)
            vqe_p1 = VQEOptimizer(config=VQEConfig(n_restarts=5, maxiter=500), seed=seed).optimize(
                H, circuit_p1, rng.uniform(-0.01, 0.01, circuit_p1.num_parameters)
            )

            de_gap_p1 = abs(vqe_p1.energy - exact.ground_energy) / exact.gap
            n_2q_p1 = sum(1 for inst in circuit_p1.data if inst.operation.num_qubits == 2)

            # ── p=2 VQE + AQC compression ─────────────────────────────────
            circuit_p2, _ = circuit_builder.create(n_qubits, 2, lattice)
            vqe_p2 = VQEOptimizer(config=VQEConfig(n_restarts=5, maxiter=500), seed=seed).optimize(
                H, circuit_p2, rng.uniform(-0.01, 0.01, circuit_p2.num_parameters)
            )

            de_gap_p2 = abs(vqe_p2.energy - exact.ground_energy) / exact.gap
            target_circuit = circuit_p2.assign_parameters(vqe_p2.theta_opt)

            t0 = time.time()
            result = compressor.compress_circuit(target_circuit, lattice)
            t_compress = time.time() - t0

            # Energy of compressed circuit
            validation = compressor.validate_compression(
                result,
                hamiltonian=H,
                energy_exact=exact.ground_energy,
                gap=exact.gap,
                energy_original=vqe_p2.energy,
            )
            de_gap_compressed = validation.delta_e_gap

            # ── Comparison ────────────────────────────────────────────────
            aqc_wins = de_gap_compressed < de_gap_p1
            improvement_pct = (
                (de_gap_p1 - de_gap_compressed) / de_gap_p1 * 100 if de_gap_p1 > 0 else 0.0
            )

            entry = {
                "topology": topology,
                "h": h,
                "seed": seed,
                "e_exact": exact.ground_energy,
                "gap": exact.gap,
                # Direct p=1
                "e_p1": vqe_p1.energy,
                "de_gap_p1": de_gap_p1,
                "n_2q_p1": n_2q_p1,
                # Uncompressed p=2 (ceiling)
                "e_p2": vqe_p2.energy,
                "de_gap_p2": de_gap_p2,
                # AQC compressed
                "e_compressed": validation.energy_compressed,
                "de_gap_compressed": de_gap_compressed,
                "fidelity": result.fidelity,
                "n_2q_compressed": result.n_2q_compressed,
                "compression_time_s": t_compress,
                # Comparison
                "aqc_wins": aqc_wins,
                "improvement_vs_p1_pct": improvement_pct,
            }
            all_results.append(entry)

            status = "✅ AQC wins" if aqc_wins else "↔ p=1 wins"
            print(
                f"  h={h}, seed={seed}: "
                f"p1={de_gap_p1:.5f}, compressed={de_gap_compressed:.5f}, "
                f"p2={de_gap_p2:.5f} | {status} ({improvement_pct:+.1f}%)"
            )

    # ── Summary ───────────────────────────────────────────────────────────
    n_aqc_wins = sum(1 for r in all_results if r["aqc_wins"])
    n_total = len(all_results)
    win_rate = n_aqc_wins / n_total if n_total > 0 else 0.0
    mean_improvement = np.mean([r["improvement_vs_p1_pct"] for r in all_results])

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  AQC-compressed wins: {n_aqc_wins}/{n_total} ({win_rate * 100:.0f}%)")
    print(f"  Mean improvement vs p=1: {mean_improvement:+.1f}%")
    print(f"  Mean fidelity: {np.mean([r['fidelity'] for r in all_results]):.5f}")

    verdict = (
        "BENEFICIAL" if win_rate > 0.6 else ("NEUTRAL" if win_rate > 0.4 else "NOT_BENEFICIAL")
    )
    print(f"  Verdict: {verdict}")

    output_data = {
        "experiment": "aqc_vs_direct",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": {
            "topology": topology,
            "n_qubits": n_qubits,
            "h_values": h_values,
            "seeds": seeds,
            "bond_dim": bond_dim,
        },
        "summary": {
            "n_aqc_wins": n_aqc_wins,
            "n_total": n_total,
            "win_rate": win_rate,
            "mean_improvement_pct": mean_improvement,
            "verdict": verdict,
        },
        "detailed_results": all_results,
    }

    output_dir = ROOT / "results" / "aqc_tensor"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir / f"aqc_vs_direct_{topology}_N{n_qubits}_{output_data['timestamp']}.json"
    )
    json_dump(output_data, output_path)
    print(f"\n  Results saved: {output_path}")

    return output_data


def main():
    parser = argparse.ArgumentParser(description="AQC vs Direct p=1 Comparison")
    parser.add_argument("--topology", default="heavy_hex")
    parser.add_argument("--n-qubits", type=int, default=10)
    parser.add_argument("--h-values", type=float, nargs="+", default=[3.25, 3.5, 3.75, 4.0])
    parser.add_argument("--bond-dim", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--single-seed", action="store_true")
    args = parser.parse_args()

    seeds = [42] if args.single_seed else args.seeds
    run_comparison(
        topology=args.topology,
        n_qubits=args.n_qubits,
        h_values=args.h_values,
        bond_dim=args.bond_dim,
        seeds=seeds,
    )


if __name__ == "__main__":
    main()
