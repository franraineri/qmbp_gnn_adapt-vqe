#!/usr/bin/env python3
"""AQC-Tensor Cross-Topology Validation — Phase 3.

Validates AQC-Tensor compression across all supported topologies:
- chain_1d (baseline, already validated in POC)
- heavy_hex (primary hardware target)
- ladder
- triangular

For each topology, compresses HVA p=2 at multiple h-values and measures:
- Fidelity of compressed vs target state
- ΔE/gap of compressed circuit vs exact
- 2Q gate reduction (the key metric for ZNE viability)
- Wall-clock time

Also runs a comparison: compressed p=2 vs direct p=1 to quantify the
expressibility benefit retained after compression.

Usage:
    python scripts/experiment_runners/aqc_tensor/run_aqc_cross_topology.py
    python scripts/experiment_runners/aqc_tensor/run_aqc_cross_topology.py --topologies chain_1d heavy_hex
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

# ─── Configuration ─────────────────────────────────────────────────────────────

N_QUBITS = 10
P_LAYERS_TARGET = 2
BOND_DIM = 64  # Validated sufficient in POC (χ=32 identical to χ=128)
SEEDS = [42, 43, 44]

# h-values per topology (within validated valid regime + safety margin)
H_VALUES_PER_TOPOLOGY = {
    "chain_1d": [3.0, 3.5, 4.0],
    "heavy_hex": [3.25, 3.5, 4.0],
    "ladder": [3.5, 4.0, 4.5],
    "triangular": [4.0, 4.5, 5.0],
}


def run_single_compression(
    topology: str,
    h_value: float,
    seed: int,
    config: AQCCompressionConfig,
) -> dict:
    """Run a single compression experiment and return metrics."""
    lattice = make_lattice(topology, N_QUBITS, J=1.0, h=h_value)
    builder = HamiltonianBuilder()
    H = builder.build(lattice)
    solver = ClassicalSolver()
    exact = solver.solve(H, lattice)

    # VQE for p=2
    circuit_builder = HVACircuitBuilder()
    circuit_p2, _ = circuit_builder.create(N_QUBITS, P_LAYERS_TARGET, lattice)
    vqe_config = VQEConfig(n_restarts=5, maxiter=500)
    optimizer = VQEOptimizer(config=vqe_config, seed=seed)
    rng = np.random.default_rng(seed)
    init = rng.uniform(-0.01, 0.01, circuit_p2.num_parameters)
    vqe_p2 = optimizer.optimize(H, circuit_p2, init)

    # VQE for p=1 (reference)
    circuit_p1, _ = circuit_builder.create(N_QUBITS, 1, lattice)
    vqe_config_p1 = VQEConfig(n_restarts=3, maxiter=300)
    optimizer_p1 = VQEOptimizer(config=vqe_config_p1, seed=seed)
    init_p1 = rng.uniform(-0.01, 0.01, circuit_p1.num_parameters)
    vqe_p1 = optimizer_p1.optimize(H, circuit_p1, init_p1)

    # Compress p=2
    target_circuit = circuit_p2.assign_parameters(vqe_p2.theta_opt)
    compressor = AQCCircuitCompressor(config)

    t_start = time.time()
    result = compressor.compress_circuit(target_circuit, lattice)
    t_compress = time.time() - t_start

    # Validate
    validation = compressor.validate_compression(
        result,
        hamiltonian=H,
        energy_exact=exact.ground_energy,
        gap=exact.gap,
        energy_original=vqe_p2.energy,
    )

    return {
        "topology": topology,
        "h": h_value,
        "seed": seed,
        "n_qubits": N_QUBITS,
        "p_layers_target": P_LAYERS_TARGET,
        # Exact reference
        "e_exact": exact.ground_energy,
        "gap": exact.gap,
        # p=2 VQE result
        "e_vqe_p2": vqe_p2.energy,
        "de_gap_vqe_p2": abs(vqe_p2.energy - exact.ground_energy) / exact.gap,
        # p=1 VQE result (comparison baseline)
        "e_vqe_p1": vqe_p1.energy,
        "de_gap_vqe_p1": abs(vqe_p1.energy - exact.ground_energy) / exact.gap,
        # Compression result
        "fidelity": result.fidelity,
        "e_compressed": validation.energy_compressed,
        "de_gap_compressed": validation.delta_e_gap,
        "depth_original": result.depth_original,
        "depth_compressed": result.depth_compressed,
        "depth_reduction_pct": result.depth_reduction_pct,
        "n_2q_original": result.n_2q_original,
        "n_2q_compressed": result.n_2q_compressed,
        "n_2q_reduction_pct": result.n_2q_reduction_pct,
        "n_params_ansatz": result.n_params,
        "n_iterations": result.n_iterations,
        "wall_clock_s": t_compress,
        "converged": result.converged,
        # Validation
        "acceptable": validation.acceptable,
        "recommendation": validation.recommendation,
        # Comparison: compressed vs direct p=1
        "compressed_better_than_p1": validation.delta_e_gap
        < abs(vqe_p1.energy - exact.ground_energy) / exact.gap,
    }


def main():
    parser = argparse.ArgumentParser(description="AQC-Tensor Cross-Topology Validation")
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=["chain_1d", "heavy_hex", "ladder", "triangular"],
        help="Topologies to test",
    )
    parser.add_argument("--bond-dim", type=int, default=BOND_DIM)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--single-seed", action="store_true", help="Use only seed=42 for faster execution"
    )
    args = parser.parse_args()

    seeds = [42] if args.single_seed else args.seeds
    config = AQCCompressionConfig(max_bond_dim=args.bond_dim, max_iterations=200)

    print("=" * 70)
    print("  AQC-TENSOR CROSS-TOPOLOGY VALIDATION")
    print(f"  Topologies: {args.topologies}")
    print(f"  N={N_QUBITS}, p={P_LAYERS_TARGET}→compressed, χ={args.bond_dim}")
    print(f"  Seeds: {seeds}")
    print("=" * 70)

    all_results = []
    topology_summaries = {}

    for topology in args.topologies:
        h_values = H_VALUES_PER_TOPOLOGY.get(topology, [3.5, 4.0])
        print(f"\n{'─' * 70}")
        print(f"  Topology: {topology} | h-values: {h_values}")
        print(f"{'─' * 70}")

        topo_results = []
        for h in h_values:
            for seed in seeds:
                print(f"    h={h}, seed={seed}...", end=" ", flush=True)
                try:
                    r = run_single_compression(topology, h, seed, config)
                    status = "✅" if r["acceptable"] else "⚠️"
                    better = "↑" if r["compressed_better_than_p1"] else "↓"
                    print(
                        f"{status} F={r['fidelity']:.5f}, "
                        f"ΔE/gap={r['de_gap_compressed']:.5f}, "
                        f"2Q: {r['n_2q_original']}→{r['n_2q_compressed']} "
                        f"({r['n_2q_reduction_pct']:.0f}%), "
                        f"vs p=1: {better}, "
                        f"{r['wall_clock_s']:.1f}s"
                    )
                    topo_results.append(r)
                    all_results.append(r)
                except Exception as e:
                    print(f"❌ ERROR: {e}")
                    all_results.append(
                        {
                            "topology": topology,
                            "h": h,
                            "seed": seed,
                            "error": str(e),
                            "acceptable": False,
                        }
                    )

        # Topology summary
        if topo_results:
            n_pass = sum(1 for r in topo_results if r["acceptable"])
            n_total = len(topo_results)
            mean_fidelity = np.mean([r["fidelity"] for r in topo_results])
            mean_de_gap = np.mean([r["de_gap_compressed"] for r in topo_results])
            mean_2q_reduction = np.mean([r["n_2q_reduction_pct"] for r in topo_results])
            n_better = sum(1 for r in topo_results if r["compressed_better_than_p1"])
            topology_summaries[topology] = {
                "pass_rate": n_pass / n_total,
                "n_pass": n_pass,
                "n_total": n_total,
                "mean_fidelity": mean_fidelity,
                "mean_de_gap": mean_de_gap,
                "mean_2q_reduction_pct": mean_2q_reduction,
                "n_better_than_p1": n_better,
                "verdict": "GO"
                if n_pass / n_total >= 0.75
                else ("CONDITIONAL" if n_pass / n_total >= 0.5 else "NO-GO"),
            }

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CROSS-TOPOLOGY SUMMARY")
    print("=" * 70)
    print(
        f"\n  {'Topology':<12} {'Pass%':>6} {'F̄':>8} {'ΔE/gap':>8} "
        f"{'2Q↓%':>6} {'vs p=1':>7} {'Verdict':>10}"
    )
    print("  " + "─" * 65)

    n_go = 0
    for topo, s in topology_summaries.items():
        print(
            f"  {topo:<12} {s['pass_rate'] * 100:>5.0f}% "
            f"{s['mean_fidelity']:>8.5f} {s['mean_de_gap']:>8.5f} "
            f"{s['mean_2q_reduction_pct']:>5.0f}% "
            f"{s['n_better_than_p1']}/{s['n_total']:>3} "
            f"{s['verdict']:>10}"
        )
        if s["verdict"] == "GO":
            n_go += 1

    # Overall verdict
    print("\n  " + "─" * 65)
    n_topologies_tested = len(topology_summaries)
    if n_go >= 3:
        overall = "GO"
        print(f"  ✅ OVERALL GO — {n_go}/{n_topologies_tested} topologies viable")
    elif n_go >= 2:
        overall = "CONDITIONAL"
        print(f"  ⚠️  CONDITIONAL — {n_go}/{n_topologies_tested} topologies viable")
    else:
        overall = "NO-GO"
        print(f"  ❌ NO-GO — only {n_go}/{n_topologies_tested} topologies viable")

    # Save results
    output_dir = ROOT / "results" / "aqc_tensor"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"cross_topology_N{N_QUBITS}_p{P_LAYERS_TARGET}_{timestamp}.json"

    json_dump(
        {
            "experiment": "aqc_tensor_cross_topology",
            "timestamp": timestamp,
            "config": {
                "n_qubits": N_QUBITS,
                "p_layers_target": P_LAYERS_TARGET,
                "bond_dim": args.bond_dim,
                "seeds": seeds,
                "topologies": args.topologies,
            },
            "topology_summaries": topology_summaries,
            "overall_verdict": overall,
            "detailed_results": all_results,
        },
        output_path,
    )
    print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
