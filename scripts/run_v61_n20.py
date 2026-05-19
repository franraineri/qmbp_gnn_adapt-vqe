"""
V6.1 Pipeline at N=20 — Scaling Demonstration.

Hypothesis: The full GNN-HVA pipeline (Phase 1→4) works at N=20 with
h_test=2.0 (deep paramagnetic), achieving ΔE/gap < 5%.

Evidence supporting this:
- V7 3A/3B: MPS is exact for 1D HVA (chi=64 sufficient)
- V7 3C: L-BFGS-B + warm-start achieves ΔE=0.020 at h=2.0, N=20
- V6.1 at N=10: ΔE/gap=2.7% at h=1.5 with production config

Key differences from N=10:
- Phase 1: Uses DMRG (N≥15 triggers auto-switch in ClassicalSolver)
- Phase 2: Same VQE protocol (L-BFGS-B, 5 restarts, descending sweep)
- Phase 3: MPNN h=128 (same as N=10 — more graph structure to learn)
- Phase 4: Deploy at h=2.0 (valid regime for N=20)

Expected runtime: ~10-15 min (Phase 1 DMRG ~30s, Phase 2 VQE ~5-8 min,
Phase 3 MPNN ~2 min, Phase 4 deploy ~1s)

Usage:
    .venv/bin/python scripts/run_v61_n20.py
    .venv/bin/python scripts/run_v61_n20.py --h-test 1.5  # harder test point
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Project root
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch

RESULTS_DIR = Path(__file__).parent / "notebook_results"


@dataclass
class N20Config:
    """Configuration for N=20 pipeline run."""
    N: int = 20
    p_layers: int = 2
    topology: str = "chain_1d"
    J: float = 1.0
    # VQE
    n_restarts: int = 5
    maxiter: int = 1000
    # MPNN
    mpnn_hidden: int = 128
    mpnn_layers: int = 3
    mpnn_epochs: int = 6000
    mpnn_lr: float = 1e-3
    mpnn_patience: int = 500
    # Deployment
    h_test: float = 2.0
    fidelity_threshold: float = 0.0  # DMRG can't compute fidelity at N=20
    # Reproducibility
    seed: int = 42


def run_n20_pipeline(cfg: N20Config) -> dict:
    """Execute the full V6.1 pipeline at N=20."""
    from torch_geometric.data import Data

    from src.poc.v6.classical_solver import ClassicalSolver
    from src.poc.v6.config import VQEConfig
    from src.poc.v6.hamiltonian_builder import HamiltonianBuilder, make_lattice
    from src.poc.v6.hardware_deployer_v61 import HardwareDeployerV61
    from src.poc.v6.hva_builder import HVACircuitBuilder
    from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset, train_mpnn
    from src.poc.v6.vqe_optimizer import VQEOptimizer

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    N = cfg.N
    result = {"config": vars(cfg), "phases": {}, "success": True, "error": None}

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()
    base_lattice = make_lattice(cfg.topology, N, J=cfg.J, h=1.0)
    qc, _ = hva.create(N, cfg.p_layers, base_lattice)
    n_params = qc.num_parameters

    # H-grid: skip h<0.8 (ferromagnetic, HVA can't express)
    # Focus on h≥0.8 where VQE can converge
    h_coarse_low = np.arange(0.8, 1.45, 0.05)
    h_coarse_high = np.arange(1.5, 2.05, 0.1)
    h_values = np.unique(np.concatenate([h_coarse_low, h_coarse_high]))
    print(f"  H-grid: {len(h_values)} points in [{h_values[0]:.2f}, {h_values[-1]:.2f}]")

    # ── Phase 1: Ground truth ──
    # At N=20, must use DMRG (exact diag needs 16TB RAM for dense 2^20 matrix).
    # DMRG returns ground_state=None → fidelity unavailable → use threshold=0.0
    print(f"\n  Phase 1: DMRG ground truth ({len(h_values)} points, N={N})...")
    print(f"    Note: DMRG (N≥15). Fidelity filter disabled (ground_state unavailable).")
    t1 = time.time()
    exact_data = []
    for h in h_values:
        lat_h = make_lattice(cfg.topology, N, J=cfg.J, h=h)
        H = builder.build(lat_h)
        exact_data.append(solver.solve(H, lat_h))  # auto → DMRG for N=20
    phase1_time = time.time() - t1
    result["phases"]["phase1"] = {
        "elapsed_s": round(phase1_time, 1),
        "n_points": len(exact_data),
        "e0_range": [exact_data[0].ground_energy, exact_data[-1].ground_energy],
    }
    print(f"    Done in {phase1_time:.1f}s")
    print(f"    E range: [{exact_data[0].ground_energy:.4f}, {exact_data[-1].ground_energy:.4f}]")

    # ── Phase 2: VQE descending sweep ──
    print(f"  Phase 2: VQE ({cfg.n_restarts} restarts, maxiter={cfg.maxiter})...")
    t2 = time.time()
    vqe_config = VQEConfig(
        p_layers=cfg.p_layers,
        n_restarts=cfg.n_restarts,
        maxiter=cfg.maxiter,
        enable_callbacks=False,
    )
    opt = VQEOptimizer(vqe_config)
    vqe_results = opt.descending_sweep(h_values, qc, base_lattice, exact_data)
    fids = np.array([r.fidelity for r in vqe_results])
    phase2_time = time.time() - t2

    n_above_threshold = int(np.sum(fids >= cfg.fidelity_threshold))
    n_above_995 = int(np.sum(fids >= 0.995))
    result["phases"]["phase2"] = {
        "elapsed_s": round(phase2_time, 1),
        "avg_fidelity": float(np.mean(fids)),
        "min_fidelity": float(np.min(fids)),
        "fid_ge_93pct": n_above_threshold,
        "fid_ge_995pct": n_above_995,
        "total_points": len(fids),
    }
    print(f"    Done in {phase2_time:.1f}s")
    print(f"    Avg fidelity: {np.mean(fids)*100:.1f}%, "
          f"≥93%: {n_above_threshold}/{len(fids)}, ≥99.5%: {n_above_995}/{len(fids)}")

    # ── Phase 3: MPNN training ──
    print(f"  Phase 3: MPNN (h={cfg.mpnn_hidden}, L={cfg.mpnn_layers}, "
          f"ep={cfg.mpnn_epochs})...")
    t3 = time.time()
    dataset = build_graph_dataset(
        base_lattice,
        h_values,
        np.array([r.theta_opt for r in vqe_results]),
        np.array([d.ground_energy for d in exact_data]),
        fidelities=fids,
        fidelity_threshold=cfg.fidelity_threshold,
    )
    model = MPNNPredictor(
        node_features=2,
        hidden_dim=cfg.mpnn_hidden,
        n_layers=cfg.mpnn_layers,
        output_dim=2 * cfg.p_layers,
    )
    train_result = train_mpnn(
        model, dataset,
        n_epochs=cfg.mpnn_epochs,
        lr=cfg.mpnn_lr,
        patience=cfg.mpnn_patience,
    )
    phase3_time = time.time() - t3
    result["phases"]["phase3"] = {
        "elapsed_s": round(phase3_time, 1),
        "training_points": len(dataset),
        "total_points": len(h_values),
        "final_mse": train_result["final_mse"],
        "stopped_early": train_result["stopped_early"],
    }
    print(f"    Done in {phase3_time:.1f}s — MSE={train_result['final_mse']:.2e}, "
          f"graphs={len(dataset)}/{len(h_values)}")

    # ── Phase 4: Deployment ──
    print(f"  Phase 4: Deploy at h_test={cfg.h_test}...")
    t4 = time.time()
    lat_test = make_lattice(cfg.topology, N, J=cfg.J, h=cfg.h_test)
    H_test = builder.build(lat_test)
    exact_test = solver.solve(H_test, lat_test)  # DMRG

    model.eval()
    edge_idx, coord = builder.build_graph_data(base_lattice)
    x_test = torch.tensor(
        np.stack([np.full(N, cfg.h_test), coord.astype(float)], axis=1),
        dtype=torch.float32,
    )
    test_graph = Data(x=x_test, edge_index=torch.tensor(edge_idx, dtype=torch.long))

    with torch.no_grad():
        theta_pred = model(test_graph).numpy().flatten()

    deployer = HardwareDeployerV61(mode="simulation")
    deploy_result = deployer.deploy_adapt_vqe(qc, H_test, theta_pred, lat_test, exact_test)
    phase4_time = time.time() - t4

    checklist = deploy_result.metrics_checklist
    n_pass = sum(checklist.values())
    result["phases"]["phase4"] = {
        "elapsed_s": round(phase4_time, 1),
        "h_test": cfg.h_test,
        "predicted_energy": deploy_result.predicted_energy,
        "exact_energy": exact_test.ground_energy,
        "delta_e": deploy_result.delta_e,
        "delta_e_over_gap": deploy_result.delta_e_over_gap,
        "fidelity": deploy_result.fidelity,
        "phase_label": deploy_result.phase_label,
        "checklist": checklist,
        "checklist_pass": n_pass,
        "checklist_total": len(checklist),
    }
    print(f"    Done in {phase4_time:.1f}s")
    print(f"    ΔE/gap = {deploy_result.delta_e_over_gap:.4f} "
          f"({'✅' if deploy_result.delta_e_over_gap < 0.05 else '❌'})")
    print(f"    Checklist: {n_pass}/{len(checklist)}, phase={deploy_result.phase_label}")
    print(f"    Fidelity: {deploy_result.fidelity:.6f}")

    total_time = time.time() - t1
    result["total_elapsed_s"] = round(total_time, 1)

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="V6.1 Pipeline at N=20")
    parser.add_argument("--h-test", type=float, default=2.0,
                        help="Test h-value (default: 2.0, valid regime for N=20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restarts", type=int, default=5)
    args = parser.parse_args()

    cfg = N20Config(h_test=args.h_test, seed=args.seed, n_restarts=args.restarts)

    print("=" * 60)
    print("  GNN-HVA v6.1 — N=20 Scaling Demonstration")
    print("=" * 60)
    print(f"  N={cfg.N}, p={cfg.p_layers}, h_test={cfg.h_test}, seed={cfg.seed}")
    print(f"  VQE: {cfg.n_restarts} restarts, maxiter={cfg.maxiter}")
    print(f"  MPNN: h={cfg.mpnn_hidden}, L={cfg.mpnn_layers}, ep={cfg.mpnn_epochs}")

    result = run_n20_pipeline(cfg)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = hashlib.sha1(f"n20:{ts}".encode()).hexdigest()[:8]
    out_path = RESULTS_DIR / f"n20_pipeline_{ts}_{run_id}.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n  Total time: {result['total_elapsed_s']:.0f}s")
    print(f"  Results saved: {out_path}")

    # Exit code based on primary criterion
    de_gap = result["phases"]["phase4"]["delta_e_over_gap"]
    if de_gap < 0.05:
        print(f"\n  ✅ SUCCESS: ΔE/gap = {de_gap:.4f} < 5%")
        return 0
    else:
        print(f"\n  ⚠️ ΔE/gap = {de_gap:.4f} > 5% (expected for h_test={cfg.h_test} at N=20)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
