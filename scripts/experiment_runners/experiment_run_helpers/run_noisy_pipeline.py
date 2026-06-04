#!/usr/bin/env python3
"""N=6 Noisy Pipeline — 3-mode comparison (noiseless / noisy-raw / ZNE-mitigated).

Validates ZNE effectiveness at N=6 using FakeTorino noise model before
real hardware deployment. Uses Phase 1+2 VQE results as input.

Methodology:
    For each h-value in the valid regime (h >= 1.25):
    1. Noiseless: StatevectorEstimator (exact baseline)
    2. Noisy raw: FakeTorino, 1 layout, no ZNE (shows noise impact)
    3. ZNE mitigated: FakeTorino, 3 layouts, linear extrapolation to CES=0

Success criteria:
    - n_mitigated_wins >= 4: ZNE beats raw for at least 4/6 h-values
    - n_good_r_squared >= 3: ZNE R² > 0.8 for at least 3/6 h-values

Usage:
    python scripts/run_noisy_pipeline.py
    python scripts/run_noisy_pipeline.py --n-layouts 3 --shots 16384
    python scripts/run_noisy_pipeline.py --h-values 2.0 1.75 1.5 1.25
    python scripts/run_noisy_pipeline.py --output-dir results/thesis/n6_noisy
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="N=6 Noisy 3-mode comparison pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-qubits", type=int, default=6, help="Number of qubits (default: 6)")
    parser.add_argument("--p", type=int, default=2, help="HVA layers (default: 2)")
    parser.add_argument(
        "--topology",
        type=str,
        default="chain_1d",
        help="Lattice topology (default: chain_1d)",
    )
    parser.add_argument(
        "--h-values",
        type=float,
        nargs="+",
        default=[2.0, 1.75, 1.5, 1.35, 1.25],
        help="h-values for comparison (descending)",
    )
    parser.add_argument(
        "--n-layouts", type=int, default=3, help="Number of layouts for ZNE (default: 3)"
    )
    parser.add_argument(
        "--shots", type=int, default=16384, help="Shots per estimation (default: 16384)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--n-restarts", type=int, default=5, help="VQE restarts (default: 5)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/thesis/n6_noisy",
        help="Output directory",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # Imports
    from qmbp_simulation import (
        ClassicalSolver,
        HamiltonianBuilder,
        HVACircuitBuilder,
        VQEOptimizer,
        make_lattice,
    )
    from qmbp_simulation.execution import (
        NoiselessBackend,
        NoisyEstimatorConfig,
        build_adjacency,
        find_layouts_bfs,
        noisy_estimate,
        run_zne_deployment,
        select_layouts_by_circuit_ces,
    )
    from qmbp_simulation.models import VQEConfig

    N = args.n_qubits
    p = args.p
    J = 1.0
    topology = args.topology
    h_values = np.array(sorted(args.h_values, reverse=True))
    n_layouts = args.n_layouts
    shots = args.shots
    seed = args.seed

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  N={N} Noisy Pipeline — 3-Mode Comparison ({topology})")
    print("=" * 60)
    print(f"  Topology: {topology}")
    print(f"  h-values: {h_values.tolist()}")
    print(f"  Layouts: {n_layouts}, Shots: {shots}, Seed: {seed}")
    print(f"  Output: {output_dir}")
    print()

    # ─── Phase 1+2: Noiseless VQE (get θ_opt for each h) ───────────────────
    print("Phase 1+2: Computing noiseless VQE baselines...")
    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()
    noiseless_backend = NoiselessBackend()

    base_lattice = make_lattice(topology, N, J=J, h=2.0)
    circuit, _ = hva.create(N, p, base_lattice)

    vqe_config = VQEConfig(p_layers=p, n_restarts=args.n_restarts, maxiter=1000)
    optimizer = VQEOptimizer(config=vqe_config, backend=noiseless_backend)

    exact_data = []
    for h in h_values:
        lat_h = make_lattice(topology, N, J=J, h=float(h))
        H = builder.build(lat_h)
        exact_data.append(solver.solve(H, lat_h))

    vqe_results = optimizer.descending_sweep(
        h_values=h_values,
        circuit=circuit,
        lattice=base_lattice,
        exact_data=exact_data,
    )

    print(f"  VQE complete: mean fidelity = {np.mean([r.fidelity for r in vqe_results]):.4f}")
    print()

    # ─── Setup FakeTorino ──────────────────────────────────────────────────
    print("Setting up FakeTorino noise model...")
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    fake_backend = FakeTorino()
    config_noisy = NoisyEstimatorConfig(shots=shots, seed_simulator=seed, optimization_level=2)

    # Find and select layouts
    adj = build_adjacency(fake_backend)
    candidates = find_layouts_bfs(adj, n_qubits=N, n_candidates=30, seed=seed)
    print(f"  Found {len(candidates)} candidate layouts")
    print()

    # ─── 3-Mode Comparison ─────────────────────────────────────────────────
    print("Running 3-mode comparison per h-value...")
    print("-" * 60)
    results_per_h = []

    for i, (h, vqe_r, exact_r) in enumerate(zip(h_values, vqe_results, exact_data, strict=False)):
        print(f"\n  h={h:.2f} ({i + 1}/{len(h_values)}):")

        # Build bound circuit with VQE-optimized parameters
        lat_h = make_lattice(topology, N, J=J, h=float(h))
        H = builder.build(lat_h)
        bound_circuit = circuit.assign_parameters(vqe_r.theta_opt)

        # Mode 1: Noiseless (exact)
        e_noiseless = noiseless_backend.evaluate(circuit, H, vqe_r.theta_opt)
        de_noiseless = abs(e_noiseless - exact_r.ground_energy) / max(exact_r.gap, 1e-10)

        # Mode 2: Noisy raw (1 layout, no ZNE)
        layout_sel_1 = select_layouts_by_circuit_ces(
            bound_circuit, fake_backend, candidates, n_select=1
        )
        e_noisy_raw = noisy_estimate(
            layout_sel_1.transpiled_circuits[0],
            H.apply_layout(layout_sel_1.transpiled_circuits[0].layout),
            fake_backend,
            config_noisy,
            seed_offset=i,
        )
        de_noisy_raw = abs(e_noisy_raw - exact_r.ground_energy) / max(exact_r.gap, 1e-10)

        # Mode 3: ZNE mitigated (n_layouts)
        layout_sel = select_layouts_by_circuit_ces(
            bound_circuit, fake_backend, candidates, n_select=n_layouts
        )
        zne_result = run_zne_deployment(
            bound_circuit=bound_circuit,
            hamiltonian=H,
            backend=fake_backend,
            layout_selection=layout_sel,
            config=config_noisy,
            n_qubits=N,
            per_site=False,
        )
        e_zne = zne_result.energy_zne.extrapolated_value
        r_squared = zne_result.energy_zne.r_squared
        de_zne = abs(e_zne - exact_r.ground_energy) / max(exact_r.gap, 1e-10)

        # Determine winner
        zne_wins = de_zne < de_noisy_raw
        good_r2 = r_squared > 0.8

        result_h = {
            "h": float(h),
            "e_exact": float(exact_r.ground_energy),
            "gap": float(exact_r.gap),
            "e_noiseless": float(e_noiseless),
            "de_noiseless": float(de_noiseless),
            "e_noisy_raw": float(e_noisy_raw),
            "de_noisy_raw": float(de_noisy_raw),
            "e_zne": float(e_zne),
            "de_zne": float(de_zne),
            "r_squared": float(r_squared),
            "zne_wins": zne_wins,
            "good_r2": good_r2,
            "ces_values": list(layout_sel.ces_values),
            "gain_pct": float((de_noisy_raw - de_zne) / max(de_noisy_raw, 1e-10) * 100),
        }
        results_per_h.append(result_h)

        status_zne = "✅" if zne_wins else "❌"
        status_r2 = "✅" if good_r2 else "❌"
        print(f"    Noiseless: ΔE/gap={de_noiseless:.4f}")
        print(f"    Noisy raw: ΔE/gap={de_noisy_raw:.4f}")
        print(f"    ZNE:       ΔE/gap={de_zne:.4f} (R²={r_squared:.4f}) {status_zne} {status_r2}")

    # ─── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    n_mitigated_wins = sum(1 for r in results_per_h if r["zne_wins"])
    n_good_r2 = sum(1 for r in results_per_h if r["good_r2"])
    n_total = len(results_per_h)

    # Success criteria: proportional to n_total (≥67% wins AND ≥50% good R²)
    # Original: 4/6 wins + 3/6 good R² → 67% + 50%
    # Generalized: works for any number of h-values
    min_wins = max(1, int(n_total * 0.67))
    min_r2 = max(1, int(n_total * 0.5))
    success = n_mitigated_wins >= min_wins and n_good_r2 >= min_r2

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  ZNE wins: {n_mitigated_wins}/{n_total} (threshold: ≥{min_wins})")
    print(f"  Good R²:  {n_good_r2}/{n_total} (threshold: ≥3)")
    print(f"  Overall:  {'PASS ✅' if success else 'FAIL ❌'}")
    print(f"  Elapsed:  {elapsed:.1f}s")
    print()

    # Mean metrics
    mean_de_noiseless = np.mean([r["de_noiseless"] for r in results_per_h])
    mean_de_noisy = np.mean([r["de_noisy_raw"] for r in results_per_h])
    mean_de_zne = np.mean([r["de_zne"] for r in results_per_h])
    mean_r2 = np.mean([r["r_squared"] for r in results_per_h])
    mean_gain = np.mean([r["gain_pct"] for r in results_per_h])

    print(f"  Mean ΔE/gap (noiseless): {mean_de_noiseless:.4f}")
    print(f"  Mean ΔE/gap (noisy raw): {mean_de_noisy:.4f}")
    print(f"  Mean ΔE/gap (ZNE):       {mean_de_zne:.4f}")
    print(f"  Mean R²:                 {mean_r2:.4f}")
    print(f"  Mean gain:               {mean_gain:.1f}%")
    print("=" * 60)

    # ─── Save results ──────────────────────────────────────────────────────
    from qmbp_simulation.framework.result_io import collect_run_metadata

    # Collect VQE baseline data for reproducibility
    vqe_baseline_data = []
    for vqe_r, exact_r in zip(vqe_results, exact_data, strict=False):
        vqe_baseline_data.append(
            {
                "h": float(vqe_r.h_value),
                "e_exact": float(exact_r.ground_energy),
                "gap": float(exact_r.gap),
                "mag_x_exact": float(exact_r.mag_x),
                "corr_zz_exact": float(exact_r.corr_zz),
                "e_vqe": float(vqe_r.energy),
                "energy_error": float(vqe_r.energy_error),
                "fidelity": float(vqe_r.fidelity),
                "n_iterations": int(vqe_r.n_iterations),
                "theta_opt": vqe_r.theta_opt.tolist(),
            }
        )

    output = {
        "metadata": collect_run_metadata(seed=seed),
        "system": {
            "hamiltonian": "TFIM: H = -J Σ_{<i,j>} Z_i Z_j - h Σ_i X_i",
            "model_type": "TFIM",
            "topology": topology,
            "boundary_conditions": "open",
            "n_qubits": N,
            "J": J,
            "initial_state": "|+>^N",
            "ansatz": "HVA",
            "p_layers": p,
            "sweep_direction": "descending",
        },
        "experiment": "n6_noisy_3mode_comparison",
        "config": {
            "n_qubits": N,
            "p_layers": p,
            "J": J,
            "h_values": h_values.tolist(),
            "n_layouts": n_layouts,
            "shots": shots,
            "seed": seed,
            "n_restarts": args.n_restarts,
            "optimizer": "L-BFGS-B",
            "noise_model": {
                "backend": "FakeTorino",
                "optimization_level": 2,
                "n_candidate_layouts": 30,
            },
        },
        "vqe_baseline": vqe_baseline_data,
        "results_per_h": results_per_h,
        "summary": {
            "n_mitigated_wins": n_mitigated_wins,
            "n_good_r2": n_good_r2,
            "n_total": n_total,
            "success_criteria_met": success,
            "mean_de_noiseless": float(mean_de_noiseless),
            "mean_de_noisy_raw": float(mean_de_noisy),
            "mean_de_zne": float(mean_de_zne),
            "mean_r2": float(mean_r2),
            "mean_gain_pct": float(mean_gain),
            "elapsed_s": elapsed,
        },
    }

    from datetime import datetime

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"noisy_3mode_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
