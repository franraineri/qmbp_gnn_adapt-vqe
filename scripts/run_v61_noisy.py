#!/usr/bin/env python
"""
GNN-HVA v6.1 — Noisy Simulation Sweep

Deploys at 6 h-values across three modes (noiseless, noisy-raw, ZNE-mitigated)
to quantify ZNE effectiveness before real QPU deployment. Produces thesis-ready
data for Section 4.5.

Modes:
  - noiseless:      StatevectorEstimator (exact baseline)
  - noisy raw:      FakeTorino + BackendEstimatorV2, n_layouts=1 (no ZNE)
  - ZNE mitigated:  FakeTorino + BackendEstimatorV2, n_layouts=3 (inhomogeneous ZNE)

Usage:
    python scripts/run_v61_noisy.py            # N=6 default (faster)
    python scripts/run_v61_noisy.py --n10      # N=10 optimal config
    python scripts/run_v61_noisy.py --quick    # 2 h-values, fewer epochs (CI)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

import numpy as np
import torch
from torch_geometric.data import Data

from src.poc.v6.classical_solver import ClassicalSolver
from src.poc.v6.config import VQEConfig
from src.poc.v6.config_v61 import DeployResultV61, NoisySweepResult, SweepSummary
from src.poc.v6.diagnostics import DiagnosticCollector
from src.poc.v6.hamiltonian_builder import HamiltonianBuilder, make_lattice
from src.poc.v6.hardware_deployer_v61 import HardwareDeployerV61
from src.poc.v6.hva_builder import HVACircuitBuilder
from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset, train_mpnn
from src.poc.v6.vqe_optimizer import VQEOptimizer

RESULTS_DIR = _project_root / "scripts" / "notebook_results"

# ── H-values for sweep ───────────────────────────────────────────────────

H_TEST_VALUES = [1.0, 1.25, 1.4, 1.5, 1.7, 2.0]
H_TEST_QUICK = [1.25, 1.5]

SHOTS = 16384


# ── JSON serialization helpers ───────────────────────────────────────────


def deploy_result_to_dict(r: DeployResultV61) -> dict:
    """Convert a DeployResultV61 to a JSON-serializable dict."""
    d = {}
    for f in dataclasses.fields(r):
        val = getattr(r, f.name)
        if isinstance(val, np.ndarray):
            d[f.name] = val.tolist()
        elif isinstance(val, np.floating):
            d[f.name] = float(val)
        elif isinstance(val, np.integer):
            d[f.name] = int(val)
        elif isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            d[f.name] = None  # JSON-safe: NaN/Inf → null
        else:
            d[f.name] = val
    return d


def sweep_result_to_dict(sr: NoisySweepResult) -> dict:
    """Convert a NoisySweepResult to a JSON-serializable dict."""
    return {
        "h_test": sr.h_test,
        "noiseless": deploy_result_to_dict(sr.noiseless),
        "noisy_raw": deploy_result_to_dict(sr.noisy_raw),
        "mitigated": deploy_result_to_dict(sr.mitigated),
        "zne_gain_energy": sr.zne_gain_energy,
        "zne_gain_mag_x": sr.zne_gain_mag_x,
        "mitigated_better": sr.mitigated_better,
    }


# ── Main pipeline ────────────────────────────────────────────────────────


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="V6.1 Noisy Simulation Sweep")
    parser.add_argument(
        "--n10",
        action="store_true",
        help="Use N=10 optimal config (slower, thesis-grade)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 2 h-values, fewer epochs (for CI/testing)",
    )
    args = parser.parse_args()

    # ── Configuration ──
    if args.n10:
        N = 10
        mpnn_hidden = 128
        mpnn_epochs = 6000
        mpnn_patience = 500
    else:
        N = 6
        mpnn_hidden = 64
        mpnn_epochs = 6000
        mpnn_patience = 500

    if args.quick:
        mpnn_epochs = 1500
        mpnn_patience = 200

    J, p = 1.0, 2
    seed_mpnn = 43  # Optimal for N=10 (validated)
    seed_layout = 42  # Reproducible layout selection

    h_test_values = H_TEST_QUICK if args.quick else H_TEST_VALUES

    np.random.seed(seed_mpnn)
    torch.manual_seed(seed_mpnn)
    import random as _random

    _random.seed(seed_mpnn)

    print("=" * 60)
    print("  GNN-HVA v6.1 — Noisy Simulation Sweep")
    print(f"  N={N}, h_test={h_test_values}, shots={SHOTS}")
    print(f"  MPNN: hidden={mpnn_hidden}, epochs={mpnn_epochs}, patience={mpnn_patience}")
    print(f"  Seeds: MPNN/VQE={seed_mpnn}, Layout={seed_layout}")
    print("=" * 60)

    # ── DiagnosticCollector (always active) ──
    collector = DiagnosticCollector(verbose=False, save_dir=RESULTS_DIR)

    # ══════════════════════════════════════════════════════════════════════
    # Task 6.1: MPNN training and deployer setup
    # ══════════════════════════════════════════════════════════════════════

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()
    base_lattice = make_lattice("chain_1d", N, J=J, h=1.0)
    qc, _ = hva.create(N, p, base_lattice)

    # ── Phase 1: Exact diagonalization (standard 27-point grid) ──
    h_coarse = np.arange(0.0, 0.8, 0.1)
    h_dense = np.arange(0.8, 1.45, 0.05)
    h_coarse2 = np.arange(1.5, 2.05, 0.1)
    h_values = np.unique(np.concatenate([h_coarse, h_dense, h_coarse2]))

    print(f"\n  Phase 1: Exact diag ({len(h_values)} h-points)...")
    t1 = time.time()
    exact_data = []
    for h in h_values:
        lat_h = make_lattice("chain_1d", N, J=J, h=h)
        H = builder.build(lat_h)
        exact_data.append(solver.solve(H, lat_h))
    print(f"    Done in {time.time() - t1:.1f}s")

    # ── Diagnostics: record Phase 1 ──
    collector.record_phase1(
        n_points=len(exact_data),
        elapsed_s=time.time() - t1,
        gap_min=min(d.gap for d in exact_data),
    )
    collector.save_checkpoint("phase1")

    # ── Phase 2: VQE descending sweep ──
    print("  Phase 2: VQE descending sweep (5 restarts, 1000 iter)...")
    t2 = time.time()
    vqe_config = VQEConfig(p_layers=p, n_restarts=5, maxiter=1000, enable_callbacks=False)
    opt = VQEOptimizer(vqe_config)
    vqe_results = opt.descending_sweep(h_values, qc, base_lattice, exact_data)
    fids = np.array([r.fidelity for r in vqe_results])
    print(f"    Done in {time.time() - t2:.1f}s — avg fid={np.mean(fids) * 100:.1f}%")

    # ── Diagnostics: record Phase 2 per-h VQE data ──
    per_h_time = (time.time() - t2) / len(vqe_results) if vqe_results else 0.0
    for i, vqe_r in enumerate(vqe_results):
        collector.record_vqe_point(
            h=float(h_values[i]),
            n_iters=vqe_r.n_iterations,
            restart_energies=[],
            theta_opt=vqe_r.theta_opt,
            elapsed_s=per_h_time,
        )
    collector.save_checkpoint("phase2")

    # ── Phase 3: MPNN training ──
    print(
        f"  Phase 3: MPNN (h={mpnn_hidden}, L=3, epochs={mpnn_epochs}, patience={mpnn_patience})..."
    )
    t3 = time.time()
    dataset = build_graph_dataset(
        base_lattice,
        h_values,
        np.array([r.theta_opt for r in vqe_results]),
        np.array([d.ground_energy for d in exact_data]),
        fidelities=fids,
        fidelity_threshold=0.93,
    )

    model = MPNNPredictor(
        node_features=2,
        hidden_dim=mpnn_hidden,
        n_layers=3,
        output_dim=2 * p,
    )
    train_result = train_mpnn(
        model,
        dataset,
        n_epochs=mpnn_epochs,
        lr=1e-3,
        patience=mpnn_patience,
    )
    print(
        f"    Done in {time.time() - t3:.1f}s — MSE={train_result['final_mse']:.2e}, "
        f"graphs={len(dataset)}/{len(h_values)}"
    )

    # ── Diagnostics: record Phase 3 per-h error ──
    model.eval()
    per_h_mse_values = []
    with torch.no_grad():
        for graph in dataset:
            pred = model(graph).numpy().flatten()
            target = graph.y.numpy().flatten()
            mse_val = float(np.mean((pred - target) ** 2))
            per_h_mse_values.append(mse_val)
    per_h_mse_arr = np.array(per_h_mse_values)
    h_train = np.array([float(graph.x[0, 0]) for graph in dataset])
    collector.record_mpnn_per_h_error(h_train, per_h_mse_arr)
    collector.save_checkpoint("phase3")

    # ── Deployer setup ──
    print("\n  Setting up deployers...")
    deployer_noiseless = HardwareDeployerV61(mode="simulation")
    deployer_noisy_raw = HardwareDeployerV61(mode="noisy_simulation", n_layouts=1, seed=seed_layout)
    deployer_mitigated = HardwareDeployerV61(mode="noisy_simulation", n_layouts=3, seed=seed_layout)
    print("    ✅ noiseless (simulation)")
    print("    ✅ noisy_raw (noisy_simulation, n_layouts=1)")
    print("    ✅ mitigated (noisy_simulation, n_layouts=3)")

    # ══════════════════════════════════════════════════════════════════════
    # Task 6.2: Multi-point deployment loop
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'─' * 60}")
    print(f"  Deploying at {len(h_test_values)} h-values × 3 modes...")
    print(f"{'─' * 60}")

    model.eval()
    edge_idx, coord = builder.build_graph_data(base_lattice)
    sweep_results: list[NoisySweepResult] = []

    for h_test in h_test_values:
        print(f"\n  h={h_test:.2f}:")

        # Build test lattice and exact solution
        lat_test = make_lattice("chain_1d", N, J=J, h=h_test)
        H_test = builder.build(lat_test)
        exact_test = solver.solve(H_test, lat_test)

        # MPNN prediction
        x_test = torch.tensor(
            np.stack([np.full(N, h_test), coord.astype(float)], axis=1),
            dtype=torch.float32,
        )
        test_graph = Data(x=x_test, edge_index=torch.tensor(edge_idx, dtype=torch.long))
        with torch.no_grad():
            theta_pred = model(test_graph).numpy().flatten()

        # Deploy noiseless
        t_dep = time.time()
        res_noiseless = deployer_noiseless.deploy_adapt_vqe(
            qc, H_test, theta_pred, lat_test, exact_test
        )
        print(
            f"    noiseless:  ΔE/gap={res_noiseless.delta_e_over_gap:.4f} "
            f"({time.time() - t_dep:.1f}s)"
        )

        # Deploy noisy raw (1 layout, no ZNE)
        t_dep = time.time()
        res_noisy_raw = deployer_noisy_raw.deploy_adapt_vqe(
            qc, H_test, theta_pred, lat_test, exact_test
        )
        print(
            f"    noisy_raw:  ΔE/gap={res_noisy_raw.delta_e_over_gap:.4f} "
            f"({time.time() - t_dep:.1f}s)"
        )

        # Deploy ZNE mitigated (3 layouts)
        t_dep = time.time()
        res_mitigated = deployer_mitigated.deploy_adapt_vqe(
            qc, H_test, theta_pred, lat_test, exact_test
        )
        print(
            f"    mitigated:  ΔE/gap={res_mitigated.delta_e_over_gap:.4f}, "
            f"R²={res_mitigated.zne_r_squared} ({time.time() - t_dep:.1f}s)"
        )

        # Compute ZNE gains
        noisy_de_gap = res_noisy_raw.delta_e_over_gap
        if noisy_de_gap not in (0, float("inf")) and not np.isnan(noisy_de_gap):
            zne_gain_energy = (noisy_de_gap - res_mitigated.delta_e_over_gap) / noisy_de_gap
        else:
            zne_gain_energy = 0.0

        noisy_mag_err = res_noisy_raw.mag_x_error
        if noisy_mag_err not in (0, float("inf")) and not np.isnan(noisy_mag_err):
            zne_gain_mag_x = (noisy_mag_err - res_mitigated.mag_x_error) / noisy_mag_err
        else:
            zne_gain_mag_x = 0.0

        mitigated_better = res_mitigated.delta_e_over_gap < res_noisy_raw.delta_e_over_gap

        sweep_results.append(
            NoisySweepResult(
                h_test=h_test,
                noiseless=res_noiseless,
                noisy_raw=res_noisy_raw,
                mitigated=res_mitigated,
                zne_gain_energy=zne_gain_energy,
                zne_gain_mag_x=zne_gain_mag_x,
                mitigated_better=mitigated_better,
            )
        )

        # ── Diagnostics: record deployment for mitigated result ──
        per_layout_data = None
        if res_mitigated.energies_per_layout and res_mitigated.ces_values:
            per_layout_data = {
                "energies": res_mitigated.energies_per_layout,
                "ces_values": res_mitigated.ces_values,
            }
        collector.record_deployment(
            h_test=h_test,
            result=res_mitigated,
            per_layout_data=per_layout_data,
        )

    # ── Diagnostics: save Phase 4 checkpoint after all deployments ──
    collector.save_checkpoint("phase4")

    # ══════════════════════════════════════════════════════════════════════
    # Task 6.3: Success criteria evaluation and reporting
    # ══════════════════════════════════════════════════════════════════════

    n_mitigated_wins = sum(1 for sr in sweep_results if sr.mitigated_better)
    n_good_r_squared = sum(
        1
        for sr in sweep_results
        if sr.mitigated.zne_r_squared is not None and sr.mitigated.zne_r_squared > 0.8
    )
    success_criteria_met = n_mitigated_wins >= 4 and n_good_r_squared >= 3

    # Print comparison table
    print(f"\n\n{'=' * 80}")
    print("  ZNE EFFECTIVENESS — COMPARISON TABLE")
    print(f"{'=' * 80}")
    print(
        f"  {'h_test':<8} {'Noiseless':<12} {'Noisy Raw':<12} "
        f"{'Mitigated':<12} {'ZNE Gain':<10} {'R²':<8} {'Win?'}"
    )
    print(f"  {'─' * 74}")
    for sr in sweep_results:
        r2_str = (
            f"{sr.mitigated.zne_r_squared:.3f}" if sr.mitigated.zne_r_squared is not None else "N/A"
        )
        win_str = "✅" if sr.mitigated_better else "❌"
        print(
            f"  {sr.h_test:<8.2f} {sr.noiseless.delta_e_over_gap:<12.4f} "
            f"{sr.noisy_raw.delta_e_over_gap:<12.4f} "
            f"{sr.mitigated.delta_e_over_gap:<12.4f} "
            f"{sr.zne_gain_energy:<10.2%} {r2_str:<8} {win_str}"
        )

    print(f"\n  Mitigated wins: {n_mitigated_wins}/{len(sweep_results)} (need ≥4)")
    print(f"  Good R² (>0.8): {n_good_r_squared}/{len(sweep_results)} (need ≥3)")
    status_str = "✅ PASS" if success_criteria_met else "❌ FAIL"
    print(f"  Success criteria: {status_str}")

    # ══════════════════════════════════════════════════════════════════════
    # Task 6.4: JSON persistence
    # ══════════════════════════════════════════════════════════════════════

    timestamp = datetime.now().isoformat()
    summary = SweepSummary(
        timestamp=timestamp,
        n_qubits=N,
        h_values=h_test_values,
        shots=SHOTS,
        n_layouts_mitigated=3,
        results=sweep_results,
        n_mitigated_wins=n_mitigated_wins,
        n_good_r_squared=n_good_r_squared,
        success_criteria_met=success_criteria_met,
    )

    # Serialize to JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = hashlib.sha1(f"noisy_sweep:{ts_file}".encode()).hexdigest()[:8]
    json_path = RESULTS_DIR / f"noisy_sweep_{ts_file}_{run_id}.json"

    summary_dict = {
        "timestamp": summary.timestamp,
        "n_qubits": summary.n_qubits,
        "h_values": summary.h_values,
        "shots": summary.shots,
        "n_layouts_mitigated": summary.n_layouts_mitigated,
        "results": [sweep_result_to_dict(sr) for sr in summary.results],
        "n_mitigated_wins": summary.n_mitigated_wins,
        "n_good_r_squared": summary.n_good_r_squared,
        "success_criteria_met": summary.success_criteria_met,
        "diagnostics": collector.to_dict(),
    }

    with open(json_path, "w") as f:
        json.dump(summary_dict, f, indent=2, default=str)

    # Cleanup checkpoints on success
    collector.cleanup_checkpoints()

    print(f"\n  Results saved: {json_path}")
    print(f"{'=' * 60}")

    return 0 if success_criteria_met else 1


if __name__ == "__main__":
    sys.exit(main())
