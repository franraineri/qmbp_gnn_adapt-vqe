#!/usr/bin/env python3
"""AQC-Tensor Proof of Concept — Phase 1 Validation.

Hypothesis H1: AQC-Tensor compresses HVA p=2 N=10 chain_1d (h=3.5, paramagnetic)
to ~50% depth with fidelity > 0.999 and ΔE/gap < 1%.

This script:
1. Builds an HVA p=2 circuit for TFIM chain_1d N=10
2. Runs VQE to get θ_opt (or uses warm-start from MPNN)
3. Binds θ_opt to get the target circuit
4. Compresses via AQC-Tensor (MPS target → shallow ansatz)
5. Validates: fidelity, ΔE/gap, depth reduction, wall-clock
6. Sweeps bond dimension χ = [32, 64, 128, 256]

GO criteria: fidelity ≥ 0.999 AND ΔE/gap < 1% AND wall_clock < 5 min
NO-GO criteria: fidelity < 0.99 OR ΔE/gap > 3%

Usage:
    python scripts/experiment_runners/aqc_tensor/run_aqc_poc.py
    python scripts/experiment_runners/aqc_tensor/run_aqc_poc.py --h-value 4.0
    python scripts/experiment_runners/aqc_tensor/run_aqc_poc.py --bond-dims 64 128 256
"""

from __future__ import annotations

import argparse
import sys
import time
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


def run_poc(
    h_value: float = 3.5,
    n_qubits: int = 10,
    p_layers_target: int = 2,
    bond_dims: list[int] | None = None,
    seed: int = 42,
) -> dict:
    """Run the AQC-Tensor POC compression experiment."""
    if bond_dims is None:
        bond_dims = [32, 64, 128, 256]

    topology = "chain_1d"
    lattice = make_lattice(topology, n_qubits, J=1.0, h=h_value)

    print("=" * 70)
    print("  AQC-TENSOR PROOF OF CONCEPT")
    print(f"  TFIM {topology} N={n_qubits}, p={p_layers_target}, h={h_value}")
    print("=" * 70)

    # ── Phase 1: Exact ground state energy ─────────────────────────────────
    builder = HamiltonianBuilder()
    H = builder.build(lattice)
    solver = ClassicalSolver()
    exact = solver.solve(H, lattice)
    e_exact = exact.ground_energy
    gap = exact.gap
    print(f"\n  Exact: E₀ = {e_exact:.8f}, gap = {gap:.6f}")

    # ── Phase 2: VQE to get θ_opt for p=2 ─────────────────────────────────
    print(f"\n  Running VQE (p={p_layers_target}, 5 restarts)...")
    circuit_builder = HVACircuitBuilder()
    circuit_p2, _ = circuit_builder.create(n_qubits, p_layers_target, lattice)

    vqe_config = VQEConfig(n_restarts=5, maxiter=500)
    optimizer = VQEOptimizer(config=vqe_config, seed=seed)

    rng = np.random.default_rng(seed)
    init_theta = rng.uniform(-0.01, 0.01, circuit_p2.num_parameters)

    t0 = time.time()
    vqe_result = optimizer.optimize(H, circuit_p2, init_theta)
    t_vqe = time.time() - t0

    theta_opt = vqe_result.theta_opt
    e_vqe = vqe_result.energy
    de_gap_vqe = abs(e_vqe - e_exact) / gap

    print(f"  VQE done in {t_vqe:.1f}s: E = {e_vqe:.8f}, ΔE/gap = {de_gap_vqe:.6f}")
    print(f"  θ_opt = {theta_opt}")

    # ── Phase 3: Bind circuit → target state ───────────────────────────────
    target_circuit = circuit_p2.assign_parameters(theta_opt)
    depth_original = target_circuit.depth()
    n_2q_original = sum(1 for inst in target_circuit.data if inst.operation.num_qubits == 2)
    print(f"\n  Target circuit: depth={depth_original}, 2Q gates={n_2q_original}")

    # ── Phase 4: AQC-Tensor compression sweep over bond dims ──────────────
    from functools import partial

    import quimb.tensor
    from qiskit_addon_aqc_tensor import generate_ansatz_from_circuit
    from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
    from qiskit_addon_aqc_tensor.simulation import (
        compute_overlap,
        tensornetwork_from_circuit,
    )
    from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator
    from scipy.optimize import minimize

    # Build a p=1 "good" circuit as ansatz template (same time, fewer layers)
    circuit_p1, _ = circuit_builder.create(n_qubits, 1, lattice)
    # Use the VQE-optimized p=1 parameters as starting point for ansatz generation
    # But generate_ansatz_from_circuit needs a bound circuit, so we bind with
    # a simple proxy: optimize p=1 quickly
    vqe_config_p1 = VQEConfig(n_restarts=3, maxiter=300)
    optimizer_p1 = VQEOptimizer(config=vqe_config_p1, seed=seed)
    init_p1 = rng.uniform(-0.01, 0.01, circuit_p1.num_parameters)
    vqe_p1 = optimizer_p1.optimize(H, circuit_p1, init_p1)
    good_circuit = circuit_p1.assign_parameters(vqe_p1.theta_opt)

    print(
        f"  p=1 reference: E = {vqe_p1.energy:.8f}, ΔE/gap = {abs(vqe_p1.energy - e_exact) / gap:.6f}"
    )
    print(
        f"  p=1 circuit: depth={good_circuit.depth()}, 2Q gates="
        f"{sum(1 for inst in good_circuit.data if inst.operation.num_qubits == 2)}"
    )

    # Generate ansatz from the p=1 circuit structure
    ansatz, initial_params = generate_ansatz_from_circuit(good_circuit, qubits_initially_zero=False)
    depth_ansatz = ansatz.depth()
    n_params_ansatz = len(initial_params)
    print(f"\n  AQC Ansatz: depth={depth_ansatz}, params={n_params_ansatz}")

    results = []
    for chi in bond_dims:
        print(f"\n  ─── Bond dimension χ={chi} ─────────────────────────────────")
        t_start = time.time()

        # Build simulator settings for this chi
        simulator_settings = QuimbSimulator(
            partial(quimb.tensor.CircuitMPS, max_bond=chi, cutoff=1e-8),
            autodiff_backend="jax",
        )

        # Build target MPS from the p=2 optimized circuit
        target_mps = tensornetwork_from_circuit(target_circuit, simulator_settings)

        # Compute initial fidelity (before optimization)
        ansatz_mps_init = tensornetwork_from_circuit(
            ansatz.assign_parameters(initial_params), simulator_settings
        )
        fidelity_init = abs(compute_overlap(ansatz_mps_init, target_mps)) ** 2

        # Optimize parameters to maximize fidelity
        objective = MaximizeStateFidelity(target_mps, ansatz, simulator_settings)

        result = minimize(
            objective.loss_function,
            initial_params,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8},
        )

        t_elapsed = time.time() - t_start
        fidelity_final = 1.0 - result.fun
        optimal_params = result.x
        converged = result.status in (0, 1, 99)

        # Build compressed circuit and measure energy
        compressed_circuit = ansatz.assign_parameters(optimal_params)
        depth_compressed = compressed_circuit.depth()
        n_2q_compressed = sum(
            1 for inst in compressed_circuit.data if inst.operation.num_qubits == 2
        )

        # Compute energy of compressed circuit via statevector
        from qiskit.primitives import StatevectorEstimator

        estimator = StatevectorEstimator()
        pub = estimator.run([(compressed_circuit, H)]).result()
        e_compressed = pub[0].data.evs.item()
        de_gap_compressed = abs(e_compressed - e_exact) / gap

        depth_reduction = (1.0 - depth_compressed / depth_original) * 100

        entry = {
            "chi": chi,
            "fidelity_init": fidelity_init,
            "fidelity_final": fidelity_final,
            "e_compressed": e_compressed,
            "delta_e_gap": de_gap_compressed,
            "depth_original": depth_original,
            "depth_compressed": depth_compressed,
            "depth_reduction_pct": depth_reduction,
            "n_2q_original": n_2q_original,
            "n_2q_compressed": n_2q_compressed,
            "n_iterations": result.nit,
            "wall_clock_s": t_elapsed,
            "converged": converged,
            "n_params": n_params_ansatz,
        }
        results.append(entry)

        # Print result
        status = "✅" if fidelity_final >= 0.999 and de_gap_compressed < 0.01 else "⚠️"
        print(
            f"  {status} χ={chi}: F={fidelity_final:.6f}, ΔE/gap={de_gap_compressed:.6f}, "
            f"depth {depth_original}→{depth_compressed} ({depth_reduction:.1f}% reduction), "
            f"{result.nit} iters, {t_elapsed:.1f}s"
        )

    # ── Phase 5: GO/NO-GO Decision ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(
        f"\n  {'χ':>4} {'Fidelity':>10} {'ΔE/gap':>10} {'Depth↓%':>8} "
        f"{'2Q orig':>7} {'2Q comp':>7} {'Time(s)':>8} {'Status':>8}"
    )
    print("  " + "─" * 70)

    best = None
    for r in results:
        status_str = (
            "GO"
            if r["fidelity_final"] >= 0.999 and r["delta_e_gap"] < 0.01
            else ("COND" if r["fidelity_final"] >= 0.99 else "NO-GO")
        )
        print(
            f"  {r['chi']:>4} {r['fidelity_final']:>10.6f} {r['delta_e_gap']:>10.6f} "
            f"{r['depth_reduction_pct']:>7.1f}% {r['n_2q_original']:>7} "
            f"{r['n_2q_compressed']:>7} {r['wall_clock_s']:>7.1f}s {status_str:>8}"
        )
        if status_str == "GO" and (best is None or r["chi"] < best["chi"]):
            best = r

    # Final verdict
    print("\n  " + "─" * 70)
    if best is not None:
        print(f"  ✅ GO — AQC-Tensor viable at χ={best['chi']}")
        print(
            f"     Optimal: F={best['fidelity_final']:.6f}, ΔE/gap={best['delta_e_gap']:.6f}, "
            f"depth reduction={best['depth_reduction_pct']:.1f}%, "
            f"2Q: {best['n_2q_original']}→{best['n_2q_compressed']}"
        )
        print(f"     Wall-clock: {best['wall_clock_s']:.1f}s (negligible vs QPU time)")
    elif any(r["fidelity_final"] >= 0.99 for r in results):
        print("  ⚠️  CONDITIONAL GO — fidelity ∈ [0.99, 0.999)")
        print("     Consider higher χ or accept marginal compression quality")
    else:
        print("  ❌ NO-GO — AQC-Tensor not viable for this configuration")
        print("     Fidelity too low or ΔE/gap too high")

    # Save results
    output_dir = ROOT / "results" / "aqc_tensor"
    output_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        output_dir / f"poc_{topology}_N{n_qubits}_p{p_layers_target}_h{h_value}_{timestamp}.json"
    )

    from qmbp_simulation.utils.helpers import json_dump

    output_data = {
        "experiment": "aqc_tensor_poc",
        "timestamp": timestamp,
        "config": {
            "topology": topology,
            "n_qubits": n_qubits,
            "p_layers_target": p_layers_target,
            "h_value": h_value,
            "seed": seed,
        },
        "reference": {
            "e_exact": e_exact,
            "gap": gap,
            "e_vqe_p2": e_vqe,
            "de_gap_vqe_p2": de_gap_vqe,
            "e_vqe_p1": vqe_p1.energy,
            "de_gap_vqe_p1": abs(vqe_p1.energy - e_exact) / gap,
            "depth_p2_original": depth_original,
            "n_2q_p2_original": n_2q_original,
            "depth_ansatz": depth_ansatz,
            "n_params_ansatz": n_params_ansatz,
        },
        "bond_dim_sweep": results,
        "verdict": "GO"
        if best
        else ("CONDITIONAL" if any(r["fidelity_final"] >= 0.99 for r in results) else "NO-GO"),
        "best_chi": best["chi"] if best else None,
    }
    json_dump(output_data, output_path)
    print(f"\n  Results saved: {output_path}")

    return output_data


def main():
    parser = argparse.ArgumentParser(description="AQC-Tensor POC Validation")
    parser.add_argument(
        "--h-value", type=float, default=3.5, help="Transverse field value (default: 3.5)"
    )
    parser.add_argument("--n-qubits", type=int, default=10, help="Number of qubits (default: 10)")
    parser.add_argument(
        "--bond-dims",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="Bond dimensions to sweep (default: 32 64 128 256)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    run_poc(
        h_value=args.h_value,
        n_qubits=args.n_qubits,
        bond_dims=args.bond_dims,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
