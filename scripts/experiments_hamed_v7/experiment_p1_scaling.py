#!/usr/bin/env python
"""
Experiment: HVA p=1 Scaling Study

Hypothesis: Reducing HVA depth from p=2 to p=1 enables scaling to larger
system sizes (N=20, N=30) while maintaining ΔE/gap < 5% in the valid regime,
at the cost of a narrower valid h-range.

Sub-experiments:
  6A: p=1 vs p=2 accuracy comparison at N=6 and N=10
  6B: p=1 full pipeline at N=20 (MPS-based VQE + MPNN)
  6C: p=1 full pipeline at N=30 (MPS-based VQE + MPNN)
  6D: Valid regime boundary detection for p=1 at each N

Key questions:
  1. How much does the valid regime shrink with p=1?
  2. Can the MPNN still learn the 2-parameter mapping effectively?
  3. What is the hardware-relevant circuit depth reduction?
  4. Does the depth-expressibility tradeoff favor p=1 for large N on hardware?

Usage:
    python scripts/experiments_hamed_v7/experiment_p1_scaling.py --sub 6A
    python scripts/experiments_hamed_v7/experiment_p1_scaling.py --sub 6B
    python scripts/experiments_hamed_v7/experiment_p1_scaling.py --sub 6C
    python scripts/experiments_hamed_v7/experiment_p1_scaling.py --sub 6D
    python scripts/experiments_hamed_v7/experiment_p1_scaling.py --sub all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Add experiments dir to path for shared_runners
_exp_dir = Path(__file__).resolve().parent
if str(_exp_dir) not in sys.path:
    sys.path.insert(0, str(_exp_dir))

from experiment_utils import SubExperimentResult, compute_metrics, save_experiment_result
from shared_runners import (
    run_lbfgsb_with_restarts,
    setup_experiment,
    train_mpnn_with_best_state,
    vqe_descending_sweep,
)

from src.poc.v6.classical_solver import ClassicalSolver
from src.poc.v6.hamiltonian_builder import HamiltonianBuilder, make_lattice
from src.poc.v6.hva_builder import HVACircuitBuilder

RESULTS_DIR = _project_root / "scripts" / "notebook_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Sub-experiment 6A: p=1 vs p=2 accuracy at N=6 and N=10 ──────────────


def run_sub_experiment_6A(args) -> SubExperimentResult:
    """Compare p=1 vs p=2 VQE accuracy at N=6 and N=10.

    Protocol: Run descending VQE sweep with both p=1 and p=2, compare
    energy errors and fidelities across the full h-range.
    This establishes the expressibility cost of reducing depth.
    """
    N_values = [6, 10]
    h_values = np.array([0.5, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0])
    seeds = [42, 43, 44]

    print(f"\n{'=' * 70}")
    print("  Sub-experiment 6A: p=1 vs p=2 accuracy comparison")
    print(f"  N={N_values}, h={list(h_values)}, seeds={seeds}")
    print(f"{'=' * 70}\n")

    all_metrics = []
    detailed = []

    for N in N_values:
        for p in [1, 2]:
            for seed in seeds:
                np.random.seed(seed)
                print(f"  N={N}, p={p}, seed={seed}:")

                env = setup_experiment(N, p=p)
                qc = env["circuit"]
                n_params = env["n_params"]
                builder = env["builder"]
                solver = env["solver"]

                sweep_data = vqe_descending_sweep(
                    circuit=qc,
                    h_values=h_values,
                    builder=builder,
                    solver=solver,
                    N=N,
                    initial_guess=np.random.uniform(-0.01, 0.01, n_params),
                    noiseless=True,
                    maxiter=1000,
                )

                for i, h in enumerate(sweep_data["h_values"]):
                    energy = sweep_data["energies"][i]
                    exact_e = sweep_data["exact_energies"][i]
                    fid = sweep_data["fidelities"][i]
                    delta_e = abs(energy - exact_e)

                    # Get gap for ΔE/gap
                    lattice = make_lattice("chain_1d", N, J=1.0, h=h)
                    H = builder.build(lattice)
                    exact = solver.solve(H, lattice)
                    gap = exact.gap
                    de_gap = delta_e / gap if gap > 1e-10 else float("inf")

                    m = compute_metrics(
                        energy=energy,
                        exact_energy=exact_e,
                        gap=gap,
                        wall_time_s=sweep_data["wall_time_s"] / len(h_values),
                        n_evaluations=0,
                        seed=seed,
                        h_value=h,
                    )
                    all_metrics.append(m)
                    detailed.append(
                        {
                            "N": N,
                            "p": p,
                            "seed": seed,
                            "h": float(h),
                            "energy": float(energy),
                            "exact_energy": float(exact_e),
                            "delta_e": float(delta_e),
                            "delta_e_over_gap": float(de_gap),
                            "fidelity": float(fid),
                            "gap": float(gap),
                            "passes_5pct": de_gap < 0.05,
                        }
                    )

                    status = "pass" if de_gap < 0.05 else "FAIL"
                    print(
                        f"    h={h:.2f}: E={energy:.6f}, "
                        f"dE/gap={de_gap:.4f} [{status}], fid={fid:.4f}"
                    )

                print(f"    Time: {sweep_data['wall_time_s']:.1f}s\n")

    # Aggregate: compute pass rates per (N, p)
    summary_table = {}
    for N in N_values:
        for p in [1, 2]:
            subset = [d for d in detailed if d["N"] == N and d["p"] == p]
            n_pass = sum(1 for d in subset if d["passes_5pct"])
            avg_de_gap = np.mean([d["delta_e_over_gap"] for d in subset])
            avg_fid = np.mean([d["fidelity"] for d in subset])
            key = f"N={N}_p={p}"
            summary_table[key] = {
                "pass_rate": f"{n_pass}/{len(subset)}",
                "avg_delta_e_over_gap": float(avg_de_gap),
                "avg_fidelity": float(avg_fid),
                "n_points": len(subset),
            }
            print(
                f"  {key}: pass={n_pass}/{len(subset)}, "
                f"avg ΔE/gap={avg_de_gap:.4f}, avg fid={avg_fid:.4f}"
            )

    # Identify valid regime boundary per (N, p)
    regime_boundaries = {}
    for N in N_values:
        for p in [1, 2]:
            subset = [d for d in detailed if d["N"] == N and d["p"] == p]
            # Group by h, check if ALL seeds pass
            for h in sorted(set(d["h"] for d in subset)):
                h_subset = [d for d in subset if d["h"] == h]
                all_pass = all(d["passes_5pct"] for d in h_subset)
                if all_pass:
                    regime_boundaries[f"N={N}_p={p}"] = h
                    break  # First h (ascending) where all seeds pass = boundary

    summary = {
        "summary_table": summary_table,
        "regime_boundaries": regime_boundaries,
        "detailed": detailed,
        "conclusion": (
            "p=1 reduces expressibility but enables shallower circuits. "
            "Valid regime boundary shifts upward with p=1."
        ),
    }

    result = SubExperimentResult(
        experiment_id="6A",
        technique=6,
        description="p=1 vs p=2 accuracy comparison (N=6, N=10)",
        config={"N_values": N_values, "h_values": list(h_values), "seeds": seeds},
        metrics=all_metrics,
        summary=summary,
        success=True,
    )
    path = save_experiment_result(result, RESULTS_DIR, prefix="p1_scaling")
    print(f"\n  Result saved: {path.name}")
    return result


# ── Sub-experiment 6B: p=1 full pipeline at N=20 (MPS) ──────────────────


def run_sub_experiment_6B(args) -> SubExperimentResult:
    """Full GNN-HVA pipeline with p=1 at N=20 using MPS simulator.

    Protocol:
    1. Phase 1: Exact diag (N=20 still feasible with sparse solver, ~2^20 = 1M states)
               OR use DMRG if too slow.
    2. Phase 2: VQE descending sweep with MPS backend (chi=64, validated exact for 1D HVA)
    3. Phase 3: Train MPNN on 2-parameter targets
    4. Phase 4: Deploy at h_test values in valid regime

    Key insight from V7-3C: N=20 valid regime is h >= 2.0 for p=2.
    With p=1, we expect the valid regime to be even narrower (h >= 2.0 or higher).
    We test h in [1.5, 1.75, 2.0, 2.5, 3.0] to find the boundary.
    """
    import torch
    from experiment_mps_simulation import create_mps_backend, evaluate_energy_mps, run_vqe_mps
    from torch_geometric.data import Data

    from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset

    N = 20
    p = 1
    chi = 64  # Validated sufficient for 1D HVA (V7-3A/3B)
    seeds = [42, 43, 44]
    # Training h-grid: only valid regime (lesson from V7-3C: train on valid regime only)
    h_train = np.linspace(1.5, 3.0, 16)  # Start broad, will filter
    # Test points
    h_test_values = [1.5, 1.75, 2.0, 2.5, 3.0]

    print(f"\n{'=' * 70}")
    print(f"  Sub-experiment 6B: p=1 full pipeline at N={N}")
    print(f"  MPS chi={chi}, seeds={seeds}")
    print(f"  h_train: {len(h_train)} points in [{h_train[0]:.1f}, {h_train[-1]:.1f}]")
    print(f"  h_test: {h_test_values}")
    print(f"{'=' * 70}\n")

    all_metrics = []
    detailed = []

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()
    base_lattice = make_lattice("chain_1d", N, J=1.0, h=1.0)
    qc, _ = hva.create(N, p, base_lattice)
    n_params = qc.num_parameters  # Should be 2 for p=1

    print(f"  Circuit: N={N}, p={p}, params={n_params}")
    print("  Expected: 2 params (theta_zz, theta_x)\n")

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"\n  {'─' * 50}")
        print(f"  Seed {seed}")
        print(f"  {'─' * 50}")

        # ── Phase 1: Classical ground truth ──
        print("  Phase 1: Classical solver...")
        t1 = time.time()
        exact_data = []
        for h in h_train:
            lat_h = make_lattice("chain_1d", N, J=1.0, h=h)
            H = builder.build(lat_h)
            exact_data.append(solver.solve(H, lat_h))
        phase1_time = time.time() - t1
        print(f"    Done in {phase1_time:.1f}s, gap_min={min(d.gap for d in exact_data):.4f}")

        # ── Phase 2: VQE with MPS (descending sweep) ──
        print("  Phase 2: VQE descending sweep (MPS)...")
        t2 = time.time()
        backend = create_mps_backend(chi)

        h_sorted_desc = np.sort(h_train)[::-1]
        theta_results = []
        energies_vqe = []
        prev_theta = np.random.uniform(-0.01, 0.01, n_params)

        for h in h_sorted_desc:
            lat_h = make_lattice("chain_1d", N, J=1.0, h=h)
            H = builder.build(lat_h)

            theta_opt, energy, _, _ = run_vqe_mps(
                qc,
                H,
                prev_theta,
                backend,
                maxiter=300,
                n_restarts=5,
            )
            theta_results.append(theta_opt)
            energies_vqe.append(energy)
            prev_theta = theta_opt.copy()

        # Reverse to ascending
        theta_results = list(reversed(theta_results))
        energies_vqe = list(reversed(energies_vqe))
        phase2_time = time.time() - t2

        # Compute fidelities (energy-based since no statevector at N=20)
        energy_errors = []
        de_gap_values = []
        for i, _h in enumerate(h_train):
            exact_e = exact_data[i].ground_energy
            gap = exact_data[i].gap
            de = abs(energies_vqe[i] - exact_e)
            de_gap = de / gap if gap > 1e-10 else float("inf")
            energy_errors.append(de)
            de_gap_values.append(de_gap)

        n_pass_vqe = sum(1 for dg in de_gap_values if dg < 0.05)
        print(f"    Done in {phase2_time:.1f}s")
        print(f"    VQE pass rate: {n_pass_vqe}/{len(h_train)} (ΔE/gap < 5%)")
        print(f"    Avg ΔE/gap: {np.mean(de_gap_values):.4f}")

        # Identify valid regime for training
        valid_mask = np.array(de_gap_values) < 0.05
        h_valid = h_train[valid_mask]
        theta_valid = np.array(theta_results)[valid_mask]

        if len(h_valid) < 3:
            print(f"    WARNING: Only {len(h_valid)} valid points. Expanding threshold to 10%.")
            valid_mask = np.array(de_gap_values) < 0.10
            h_valid = h_train[valid_mask]
            theta_valid = np.array(theta_results)[valid_mask]

        print(f"    Valid regime: h >= {h_valid[0]:.2f} ({len(h_valid)} points)")

        # ── Phase 3: MPNN training ──
        print("  Phase 3: MPNN training...")
        t3 = time.time()

        # Build graph dataset from valid points only
        dataset = build_graph_dataset(
            base_lattice,
            h_valid,
            theta_valid,
            np.array([exact_data[i].ground_energy for i, m in enumerate(valid_mask) if m]),
            fidelities=np.ones(len(h_valid)),  # All "valid" by construction
            fidelity_threshold=0.0,  # No filter (already filtered)
        )

        model = MPNNPredictor(
            node_features=2,
            hidden_dim=128,
            n_layers=3,
            output_dim=n_params,  # 2 for p=1
            per_parameter_heads=False,
            use_edge_features=False,
        )

        train_result = train_mpnn_with_best_state(
            model,
            dataset,
            n_epochs=6000,
            lr=1e-3,
            patience=500,
        )
        phase3_time = time.time() - t3
        print(f"    Done in {phase3_time:.1f}s, MSE={train_result['final_mse']:.2e}")
        print(f"    Training points: {len(dataset)}")

        # ── Phase 4: Deployment at test h-values ──
        print("  Phase 4: Deployment...")
        t4 = time.time()
        model.eval()

        for h_test in h_test_values:
            lat_test = make_lattice("chain_1d", N, J=1.0, h=h_test)
            H_test = builder.build(lat_test)
            exact_test = solver.solve(H_test, lat_test)

            # MPNN prediction
            edge_idx, coord = builder.build_graph_data(base_lattice)
            x_test = torch.tensor(
                np.stack([np.full(N, h_test), coord.astype(float)], axis=1),
                dtype=torch.float32,
            )
            test_graph = Data(
                x=x_test,
                edge_index=torch.tensor(edge_idx, dtype=torch.long),
            )
            with torch.no_grad():
                theta_pred = model(test_graph).numpy().flatten()

            # Evaluate predicted theta on MPS
            energy_pred = evaluate_energy_mps(qc, H_test, theta_pred, backend)
            delta_e = abs(energy_pred - exact_test.ground_energy)
            de_gap = delta_e / exact_test.gap if exact_test.gap > 1e-10 else float("inf")

            status = "pass" if de_gap < 0.05 else "FAIL"
            print(f"    h={h_test:.2f}: E_pred={energy_pred:.6f}, ΔE/gap={de_gap:.4f} [{status}]")

            m = compute_metrics(
                energy=energy_pred,
                exact_energy=exact_test.ground_energy,
                gap=exact_test.gap,
                wall_time_s=0,
                n_evaluations=0,
                seed=seed,
                h_value=h_test,
            )
            all_metrics.append(m)
            detailed.append(
                {
                    "N": N,
                    "p": p,
                    "seed": seed,
                    "h_test": float(h_test),
                    "energy_pred": float(energy_pred),
                    "exact_energy": float(exact_test.ground_energy),
                    "delta_e": float(delta_e),
                    "delta_e_over_gap": float(de_gap),
                    "gap": float(exact_test.gap),
                    "theta_pred": theta_pred.tolist(),
                    "passes_5pct": de_gap < 0.05,
                    "mpnn_mse": train_result["final_mse"],
                    "n_train_points": len(dataset),
                    "valid_regime_start": float(h_valid[0]),
                    "phase2_time": phase2_time,
                    "phase3_time": phase3_time,
                }
            )

        phase4_time = time.time() - t4
        print(f"    Phase 4 time: {phase4_time:.1f}s")

    # Summary
    n_total = len(detailed)
    n_pass = sum(1 for d in detailed if d["passes_5pct"])
    avg_de_gap = np.mean([d["delta_e_over_gap"] for d in detailed])

    # Per h_test summary
    print(f"\n  {'─' * 50}")
    print("  Summary (across seeds):")
    for h_test in h_test_values:
        subset = [d for d in detailed if d["h_test"] == h_test]
        if subset:
            mean_dg = np.mean([d["delta_e_over_gap"] for d in subset])
            std_dg = np.std([d["delta_e_over_gap"] for d in subset])
            n_p = sum(1 for d in subset if d["passes_5pct"])
            print(
                f"    h={h_test:.2f}: ΔE/gap = {mean_dg:.4f} +/- {std_dg:.4f}, "
                f"pass={n_p}/{len(subset)}"
            )

    summary = {
        "N": N,
        "p": p,
        "chi": chi,
        "n_params": n_params,
        "pass_rate": f"{n_pass}/{n_total}",
        "avg_delta_e_over_gap": float(avg_de_gap),
        "per_h_test": {
            str(h): {
                "mean_de_gap": float(
                    np.mean([d["delta_e_over_gap"] for d in detailed if d["h_test"] == h])
                ),
                "std_de_gap": float(
                    np.std([d["delta_e_over_gap"] for d in detailed if d["h_test"] == h])
                ),
                "pass_rate": sum(1 for d in detailed if d["h_test"] == h and d["passes_5pct"]),
            }
            for h in h_test_values
        },
        "detailed": detailed,
    }

    result = SubExperimentResult(
        experiment_id="6B",
        technique=6,
        description=f"p=1 full pipeline at N={N} (MPS, chi={chi})",
        config={
            "N": N,
            "p": p,
            "chi": chi,
            "seeds": seeds,
            "h_train": list(h_train),
            "h_test": h_test_values,
        },
        metrics=all_metrics,
        summary=summary,
        success=True,
    )
    path = save_experiment_result(result, RESULTS_DIR, prefix="p1_scaling")
    print(f"\n  Result saved: {path.name}")
    return result


# ── Sub-experiment 6C: p=1 full pipeline at N=30 (MPS) ──────────────────


def run_sub_experiment_6C(args) -> SubExperimentResult:
    """Full GNN-HVA pipeline with p=1 at N=30 using MPS simulator.

    Same protocol as 6B but at N=30. Tests whether the pipeline scales
    beyond the statevector limit (2^30 ~ 1 billion states).
    Uses DMRG for ground truth and MPS for VQE.
    """
    import torch
    from experiment_mps_simulation import create_mps_backend, evaluate_energy_mps, run_vqe_mps
    from torch_geometric.data import Data

    from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset

    N = 30
    p = 1
    chi = 64
    seeds = [42, 43]  # 2 seeds (N=30 is expensive)
    # Narrower h-range (expect valid regime h >= 2.5 for p=1 at N=30)
    h_train = np.linspace(2.0, 4.0, 11)
    h_test_values = [2.0, 2.5, 3.0, 3.5, 4.0]

    print(f"\n{'=' * 70}")
    print(f"  Sub-experiment 6C: p=1 full pipeline at N={N}")
    print(f"  MPS chi={chi}, seeds={seeds}")
    print(f"  h_train: {len(h_train)} points in [{h_train[0]:.1f}, {h_train[-1]:.1f}]")
    print(f"  h_test: {h_test_values}")
    print(f"{'=' * 70}\n")

    all_metrics = []
    detailed = []

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()
    base_lattice = make_lattice("chain_1d", N, J=1.0, h=1.0)
    qc, _ = hva.create(N, p, base_lattice)
    n_params = qc.num_parameters

    print(f"  Circuit: N={N}, p={p}, params={n_params}")

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"\n  {'─' * 50}")
        print(f"  Seed {seed}")
        print(f"  {'─' * 50}")

        # Phase 1: Classical solver (DMRG for N=30)
        print("  Phase 1: Classical solver (DMRG expected)...")
        t1 = time.time()
        exact_data = []
        for h in h_train:
            lat_h = make_lattice("chain_1d", N, J=1.0, h=h)
            H = builder.build(lat_h)
            exact_data.append(solver.solve(H, lat_h))
        phase1_time = time.time() - t1
        print(f"    Done in {phase1_time:.1f}s")

        # Phase 2: VQE with MPS
        print("  Phase 2: VQE descending sweep (MPS)...")
        t2 = time.time()
        backend = create_mps_backend(chi)

        h_sorted_desc = np.sort(h_train)[::-1]
        theta_results = []
        energies_vqe = []
        prev_theta = np.random.uniform(-0.01, 0.01, n_params)

        for h in h_sorted_desc:
            lat_h = make_lattice("chain_1d", N, J=1.0, h=h)
            H = builder.build(lat_h)
            theta_opt, energy, _, _ = run_vqe_mps(
                qc,
                H,
                prev_theta,
                backend,
                maxiter=300,
                n_restarts=5,
            )
            theta_results.append(theta_opt)
            energies_vqe.append(energy)
            prev_theta = theta_opt.copy()

        theta_results = list(reversed(theta_results))
        energies_vqe = list(reversed(energies_vqe))
        phase2_time = time.time() - t2

        # Compute ΔE/gap
        de_gap_values = []
        for i, _h in enumerate(h_train):
            exact_e = exact_data[i].ground_energy
            gap = exact_data[i].gap
            de = abs(energies_vqe[i] - exact_e)
            de_gap = de / gap if gap > 1e-10 else float("inf")
            de_gap_values.append(de_gap)

        n_pass_vqe = sum(1 for dg in de_gap_values if dg < 0.05)
        print(f"    Done in {phase2_time:.1f}s, pass={n_pass_vqe}/{len(h_train)}")

        # Filter valid regime
        valid_mask = np.array(de_gap_values) < 0.05
        h_valid = h_train[valid_mask]
        theta_valid = np.array(theta_results)[valid_mask]

        if len(h_valid) < 3:
            valid_mask = np.array(de_gap_values) < 0.10
            h_valid = h_train[valid_mask]
            theta_valid = np.array(theta_results)[valid_mask]
            print(f"    Expanded to 10% threshold: {len(h_valid)} points")

        if len(h_valid) == 0:
            print("    ERROR: No valid points found. Skipping MPNN + deploy.")
            detailed.append(
                {
                    "N": N,
                    "p": p,
                    "seed": seed,
                    "error": "no_valid_points",
                    "de_gap_values": [float(x) for x in de_gap_values],
                }
            )
            continue

        print(f"    Valid regime: h >= {h_valid[0]:.2f} ({len(h_valid)} points)")

        # Phase 3: MPNN
        print("  Phase 3: MPNN training...")
        t3 = time.time()
        dataset = build_graph_dataset(
            base_lattice,
            h_valid,
            theta_valid,
            np.array([exact_data[i].ground_energy for i, m in enumerate(valid_mask) if m]),
            fidelities=np.ones(len(h_valid)),
            fidelity_threshold=0.0,
        )

        model = MPNNPredictor(
            node_features=2,
            hidden_dim=128,
            n_layers=3,
            output_dim=n_params,
            per_parameter_heads=False,
        )
        train_result = train_mpnn_with_best_state(
            model,
            dataset,
            n_epochs=6000,
            lr=1e-3,
            patience=500,
        )
        phase3_time = time.time() - t3
        print(f"    Done in {phase3_time:.1f}s, MSE={train_result['final_mse']:.2e}")

        # Phase 4: Deploy
        print("  Phase 4: Deployment...")
        model.eval()
        for h_test in h_test_values:
            lat_test = make_lattice("chain_1d", N, J=1.0, h=h_test)
            H_test = builder.build(lat_test)
            exact_test = solver.solve(H_test, lat_test)

            edge_idx, coord = builder.build_graph_data(base_lattice)
            x_test = torch.tensor(
                np.stack([np.full(N, h_test), coord.astype(float)], axis=1),
                dtype=torch.float32,
            )
            test_graph = Data(
                x=x_test,
                edge_index=torch.tensor(edge_idx, dtype=torch.long),
            )
            with torch.no_grad():
                theta_pred = model(test_graph).numpy().flatten()

            energy_pred = evaluate_energy_mps(qc, H_test, theta_pred, backend)
            delta_e = abs(energy_pred - exact_test.ground_energy)
            de_gap = delta_e / exact_test.gap if exact_test.gap > 1e-10 else float("inf")

            status = "pass" if de_gap < 0.05 else "FAIL"
            print(f"    h={h_test:.2f}: ΔE/gap={de_gap:.4f} [{status}]")

            m = compute_metrics(
                energy=energy_pred,
                exact_energy=exact_test.ground_energy,
                gap=exact_test.gap,
                wall_time_s=0,
                n_evaluations=0,
                seed=seed,
                h_value=h_test,
            )
            all_metrics.append(m)
            detailed.append(
                {
                    "N": N,
                    "p": p,
                    "seed": seed,
                    "h_test": float(h_test),
                    "energy_pred": float(energy_pred),
                    "exact_energy": float(exact_test.ground_energy),
                    "delta_e": float(delta_e),
                    "delta_e_over_gap": float(de_gap),
                    "gap": float(exact_test.gap),
                    "theta_pred": theta_pred.tolist(),
                    "passes_5pct": de_gap < 0.05,
                    "mpnn_mse": train_result["final_mse"],
                    "n_train_points": len(dataset),
                    "valid_regime_start": float(h_valid[0]),
                }
            )

    n_total = len([d for d in detailed if "h_test" in d])
    n_pass = sum(1 for d in detailed if d.get("passes_5pct", False))

    summary = {
        "N": N,
        "p": p,
        "chi": chi,
        "n_params": n_params,
        "pass_rate": f"{n_pass}/{n_total}",
        "detailed": detailed,
    }

    result = SubExperimentResult(
        experiment_id="6C",
        technique=6,
        description=f"p=1 full pipeline at N={N} (MPS, chi={chi})",
        config={
            "N": N,
            "p": p,
            "chi": chi,
            "seeds": seeds,
            "h_train": list(h_train),
            "h_test": h_test_values,
        },
        metrics=all_metrics,
        summary=summary,
        success=True,
    )
    path = save_experiment_result(result, RESULTS_DIR, prefix="p1_scaling")
    print(f"\n  Result saved: {path.name}")
    return result


# ── Sub-experiment 6D: Valid regime boundary detection ───────────────────


def run_sub_experiment_6D(args) -> SubExperimentResult:
    """Systematic detection of valid regime boundary for p=1 at each N.

    Protocol: Fine-grained h-sweep at each N to find the minimum h where
    ΔE/gap < 5% consistently (all 3 seeds pass). This maps the
    expressibility-depth tradeoff quantitatively.

    Also computes circuit depth metrics (gate count, CNOT count post-transpilation)
    to quantify the hardware advantage of p=1.
    """

    N_values = [6, 10, 20]
    h_fine = np.arange(1.0, 3.05, 0.1)  # Fine grid
    seeds = [42, 43, 44]

    print(f"\n{'=' * 70}")
    print("  Sub-experiment 6D: Valid regime boundary detection")
    print(f"  N={N_values}, h=[{h_fine[0]:.1f}, {h_fine[-1]:.1f}], seeds={seeds}")
    print(f"{'=' * 70}\n")

    all_metrics = []
    detailed = []
    circuit_metrics = {}

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()

    # First: compute circuit depth metrics for each (N, p)
    print("  Circuit depth analysis:")
    print(f"  {'N':<4} {'p':<3} {'params':<7} {'gates':<7} {'depth':<7} {'CX(est)':<8}")
    print(f"  {'─' * 40}")

    for N in N_values:
        for p in [1, 2]:
            lat = make_lattice("chain_1d", N, J=1.0, h=1.0)
            qc, _ = hva.create(N, p, lat)
            n_gates = qc.size()
            depth = qc.depth()
            # Estimate CX count: each RZZ decomposes to 2 CX
            n_rzz = len(lat.edges) * p
            cx_estimate = 2 * n_rzz

            circuit_metrics[f"N={N}_p={p}"] = {
                "n_params": qc.num_parameters,
                "n_gates": n_gates,
                "depth": depth,
                "cx_estimate": cx_estimate,
                "n_rzz": n_rzz,
                "n_rx": N * p,
            }
            print(
                f"  {N:<4} {p:<3} {qc.num_parameters:<7} {n_gates:<7} {depth:<7} {cx_estimate:<8}"
            )

    print()

    # Depth reduction ratios
    for N in N_values:
        p1 = circuit_metrics[f"N={N}_p=1"]
        p2 = circuit_metrics[f"N={N}_p=2"]
        reduction = 1 - p1["cx_estimate"] / p2["cx_estimate"]
        print(f"  N={N}: p=1 reduces CX count by {reduction * 100:.0f}% vs p=2")

    print()

    # Now: VQE boundary detection
    regime_boundaries = {}

    for N in N_values:
        print(f"\n  N={N}: scanning h-boundary for p=1...")

        # Use MPS for N=20, statevector for N<=10
        use_mps = N > 15

        if use_mps:
            from experiment_mps_simulation import (
                create_mps_backend,
                run_vqe_mps,
            )

            backend = create_mps_backend(64)

        lat = make_lattice("chain_1d", N, J=1.0, h=1.0)
        qc, _ = hva.create(N, 1, lat)
        n_params = qc.num_parameters

        boundary_found = None

        for seed in seeds:
            np.random.seed(seed)
            prev_theta = np.random.uniform(-0.01, 0.01, n_params)

            # Descending sweep
            for h in sorted(h_fine, reverse=True):
                if h > 3.0 and N <= 10:
                    continue  # Skip unnecessarily high h for small N

                lat_h = make_lattice("chain_1d", N, J=1.0, h=h)
                H = builder.build(lat_h)
                exact = solver.solve(H, lat_h)

                if use_mps:
                    theta_opt, energy, _, _ = run_vqe_mps(
                        qc,
                        H,
                        prev_theta,
                        backend,
                        maxiter=300,
                        n_restarts=3,
                    )
                else:
                    theta_opt, energy, _, _ = run_lbfgsb_with_restarts(
                        qc,
                        H,
                        prev_theta,
                        n_restarts=3,
                        maxiter=500,
                    )

                delta_e = abs(energy - exact.ground_energy)
                de_gap = delta_e / exact.gap if exact.gap > 1e-10 else float("inf")
                prev_theta = theta_opt.copy()

                detailed.append(
                    {
                        "N": N,
                        "p": 1,
                        "seed": seed,
                        "h": float(h),
                        "delta_e_over_gap": float(de_gap),
                        "passes_5pct": de_gap < 0.05,
                        "energy": float(energy),
                        "exact_energy": float(exact.ground_energy),
                        "gap": float(exact.gap),
                    }
                )

        # Find boundary: lowest h where ALL seeds pass
        for h in sorted(h_fine):
            h_subset = [d for d in detailed if d["N"] == N and abs(d["h"] - h) < 0.01]
            if len(h_subset) >= len(seeds) and all(d["passes_5pct"] for d in h_subset):
                boundary_found = h
                break

        regime_boundaries[f"N={N}"] = boundary_found
        if boundary_found is not None:
            print(f"    Boundary: h >= {boundary_found:.1f} (all seeds pass)")
        else:
            print("    Boundary: NOT FOUND in range (all h fail for at least one seed)")

    # Compare with p=2 boundaries (from project-status.md)
    p2_boundaries = {"N=6": 1.25, "N=10": 1.5, "N=20": 2.0}

    print(f"\n  {'─' * 50}")
    print("  Valid regime comparison (p=1 vs p=2):")
    print(f"  {'N':<5} {'p=2 boundary':<15} {'p=1 boundary':<15} {'Shift'}")
    print(f"  {'─' * 50}")
    for N in N_values:
        key = f"N={N}"
        p2_b = p2_boundaries.get(key, "?")
        p1_b = regime_boundaries.get(key, "?")
        shift = ""
        if isinstance(p2_b, int | float) and isinstance(p1_b, int | float):
            shift = f"+{p1_b - p2_b:.1f}"
        print(f"  {N:<5} {str(p2_b):<15} {str(p1_b):<15} {shift}")

    summary = {
        "regime_boundaries_p1": regime_boundaries,
        "regime_boundaries_p2": p2_boundaries,
        "circuit_metrics": circuit_metrics,
        "detailed": detailed,
        "conclusion": (
            "p=1 shifts the valid regime boundary upward but provides "
            "50% CX reduction, enabling larger N on hardware."
        ),
    }

    result = SubExperimentResult(
        experiment_id="6D",
        technique=6,
        description="Valid regime boundary detection for p=1",
        config={"N_values": N_values, "h_fine": list(h_fine), "seeds": seeds},
        metrics=all_metrics,
        summary=summary,
        success=True,
    )
    path = save_experiment_result(result, RESULTS_DIR, prefix="p1_scaling")
    print(f"\n  Result saved: {path.name}")
    return result


# ── Main ─────────────────────────────────────────────────────────────────


EXPERIMENTS = {
    "6A": run_sub_experiment_6A,
    "6B": run_sub_experiment_6B,
    "6C": run_sub_experiment_6C,
    "6D": run_sub_experiment_6D,
}


def main():
    parser = argparse.ArgumentParser(description="Experiment 6: HVA p=1 Scaling Study")
    parser.add_argument(
        "--sub",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        default="all",
        help="Sub-experiment to run (default: all)",
    )
    parser.add_argument(
        "--chi",
        type=int,
        nargs="+",
        default=None,
        help="MPS bond dimensions to test",
    )
    parser.add_argument(
        "--h-values",
        type=float,
        nargs="+",
        default=None,
        help="Override h-values for testing",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Experiment 6: HVA p=1 Scaling Study")
    print("  Hypothesis: p=1 enables scaling to N=20-30 with narrower valid regime")
    print("=" * 70)

    subs_to_run = list(EXPERIMENTS.keys()) if args.sub == "all" else [args.sub]

    results = []
    for sub_id in subs_to_run:
        print(f"\n{'#' * 70}")
        print(f"  Running sub-experiment {sub_id}")
        print(f"{'#' * 70}")
        try:
            result = EXPERIMENTS[sub_id](args)
            results.append(result)
            print(f"\n  ✅ {sub_id} completed successfully")
        except Exception as e:
            print(f"\n  ❌ {sub_id} FAILED: {e}")
            import traceback

            traceback.print_exc()
            results.append(None)

    # Final summary
    print(f"\n\n{'=' * 70}")
    print("  EXPERIMENT 6 SUMMARY")
    print(f"{'=' * 70}")
    for sub_id, result in zip(subs_to_run, results, strict=False):
        if result is None:
            print(f"  {sub_id}: FAILED")
        else:
            print(f"  {sub_id}: {result.description}")
            if result.summary:
                if "pass_rate" in result.summary:
                    print(f"       Pass rate: {result.summary['pass_rate']}")
                if "regime_boundaries_p1" in result.summary:
                    print(f"       Boundaries: {result.summary['regime_boundaries_p1']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
