#!/usr/bin/env python
"""
GNN-HVA v6.1 — Parametric Pipeline Runner

Runs the full V6.1 pipeline (Phases 1–4 + gradient analysis) with
configurable parameters for systematic testing. Produces JSON summaries
compatible with the notebook_results auto-registry.

Usage:
    python scripts/run_v61_parametric.py --config <config_name>
    python scripts/run_v61_parametric.py --config all  # run all configs sequentially

Configurations test the validation targets from the project objectives:
  - optimal:       5 restarts, 6000 epochs, h_test=1.25 (optimal config)
  - h_test_1.4:    5 restarts, 6000 epochs, h_test=1.4 (expected 4-5/6)
  - h_test_1.5:    5 restarts, 6000 epochs, h_test=1.5 (expected 5/6)
  - per_param:     5 restarts, 6000 epochs, h_test=1.25, per-parameter heads
  - mpnn_128:      5 restarts, 6000 epochs, h_test=1.25, hidden=128 (N=10 prep)
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

RESULTS_DIR = _project_root / "scripts" / "notebook_results"
BINNACLE_DIR = _project_root / "documentation" / "binnacles"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from src.poc.v6.diagnostics import DiagnosticCollector, configure_pipeline_logging  # noqa: I001, E402


# ── Configuration presets ────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """Full pipeline configuration for a single parametric run."""

    name: str
    N: int = 6
    J: float | str = 1.0  # float for uniform, "ladder_nonuniform" for special
    p_layers: int = 2
    topology: str = "chain_1d"
    # VQE
    n_restarts: int = 5
    maxiter: int = 1000
    # MPNN
    mpnn_hidden: int = 64
    mpnn_layers: int = 3
    mpnn_epochs: int = 6000
    mpnn_lr: float = 1e-3
    mpnn_patience: int = 300
    per_parameter_heads: bool = False
    use_edge_features: bool = False
    # Deployment
    h_test: float = 1.25
    fidelity_threshold: float = 0.93
    # Reproducibility
    seed: int = 42
    # H-grid
    h_grid: str = "standard"  # "standard" (27pts) or "dense" (45pts near critical)


CONFIGS_N6 = {
    "optimal": PipelineConfig(
        name="optimal",
        n_restarts=5,
        mpnn_epochs=6000,
        h_test=1.25,
    ),
    "h_test_1.4": PipelineConfig(
        name="h_test_1.4",
        n_restarts=5,
        mpnn_epochs=6000,
        h_test=1.4,
    ),
    "h_test_1.5": PipelineConfig(
        name="h_test_1.5",
        n_restarts=5,
        mpnn_epochs=6000,
        h_test=1.5,
    ),
    "per_param": PipelineConfig(
        name="per_param",
        n_restarts=5,
        mpnn_epochs=6000,
        h_test=1.25,
        per_parameter_heads=True,
    ),
    "mpnn_128": PipelineConfig(
        name="mpnn_128",
        n_restarts=5,
        mpnn_epochs=6000,
        h_test=1.25,
        mpnn_hidden=128,
    ),
}

# N=10 configs — based on binnacle-N10 findings + V6.1 techniques
CONFIGS_N10 = {
    "n10_baseline": PipelineConfig(
        name="n10_baseline",
        N=10,
        n_restarts=5,
        mpnn_hidden=128,
        mpnn_epochs=6000,
        h_test=1.5,
    ),
    "n10_h1.4": PipelineConfig(
        name="n10_h1.4",
        N=10,
        n_restarts=5,
        mpnn_hidden=128,
        mpnn_epochs=6000,
        h_test=1.4,
    ),
    "n10_per_param": PipelineConfig(
        name="n10_per_param",
        N=10,
        n_restarts=5,
        mpnn_hidden=128,
        mpnn_epochs=6000,
        h_test=1.5,
        per_parameter_heads=True,
    ),
    "n10_seed43": PipelineConfig(
        name="n10_seed43",
        N=10,
        n_restarts=5,
        mpnn_hidden=128,
        mpnn_epochs=6000,
        h_test=1.5,
        seed=43,
    ),
    "n10_seed44": PipelineConfig(
        name="n10_seed44",
        N=10,
        n_restarts=5,
        mpnn_hidden=128,
        mpnn_epochs=6000,
        h_test=1.5,
        seed=44,
    ),
    # ── Exploration configs (V6.1 features + N=10 optimization) ──
    "n10_ladder": PipelineConfig(
        name="n10_ladder",
        N=10,
        topology="ladder",
        n_restarts=5,
        mpnn_hidden=128,
        mpnn_epochs=6000,
        h_test=1.5,
    ),
    "n10_best_h1.4": PipelineConfig(
        name="n10_best_h1.4",
        N=10,
        n_restarts=5,
        mpnn_hidden=128,
        mpnn_epochs=6000,
        mpnn_patience=500,
        h_test=1.4,
        seed=43,
    ),
    "n10_patience500": PipelineConfig(
        name="n10_patience500",
        N=10,
        n_restarts=5,
        mpnn_hidden=128,
        mpnn_epochs=6000,
        mpnn_patience=500,
        h_test=1.5,
        seed=43,
    ),
    # ── V6.1 advanced exploration ──
    "n10_ladder_weak": PipelineConfig(
        name="n10_ladder_weak",
        N=10,
        J="ladder_nonuniform",
        topology="ladder",
        n_restarts=5,
        mpnn_hidden=128,
        mpnn_epochs=6000,
        mpnn_patience=500,
        h_test=2.0,
        seed=43,
        use_edge_features=True,
    ),
    "n10_dense_grid": PipelineConfig(
        name="n10_dense_grid",
        N=10,
        n_restarts=5,
        mpnn_hidden=128,
        mpnn_epochs=6000,
        mpnn_patience=500,
        h_test=1.5,
        seed=43,
        h_grid="dense",
    ),
}

# N=12 configs — scaling from N=10 findings (h=128→160, patience=600, seed=43 optimal)
CONFIGS_N12 = {
    "n12_baseline": PipelineConfig(
        name="n12_baseline",
        N=12,
        n_restarts=5,
        mpnn_hidden=128,
        mpnn_epochs=6000,
        mpnn_patience=500,
        h_test=1.5,
        seed=43,
    ),
    "n12_h160": PipelineConfig(
        name="n12_h160",
        N=12,
        n_restarts=5,
        mpnn_hidden=160,
        mpnn_epochs=6000,
        mpnn_patience=600,
        h_test=1.5,
        seed=43,
    ),
    "n12_h1.4": PipelineConfig(
        name="n12_h1.4",
        N=12,
        n_restarts=5,
        mpnn_hidden=128,
        mpnn_epochs=6000,
        mpnn_patience=500,
        h_test=1.4,
        seed=43,
    ),
    "n12_h1.7": PipelineConfig(
        name="n12_h1.7",
        N=12,
        n_restarts=5,
        mpnn_hidden=128,
        mpnn_epochs=6000,
        mpnn_patience=500,
        h_test=1.7,
        seed=43,
    ),
    "n12_seed42": PipelineConfig(
        name="n12_seed42",
        N=12,
        n_restarts=5,
        mpnn_hidden=128,
        mpnn_epochs=6000,
        mpnn_patience=500,
        h_test=1.5,
        seed=42,
    ),
    "n12_seed44": PipelineConfig(
        name="n12_seed44",
        N=12,
        n_restarts=5,
        mpnn_hidden=128,
        mpnn_epochs=6000,
        mpnn_patience=500,
        h_test=1.5,
        seed=44,
    ),
    "n12_patience700": PipelineConfig(
        name="n12_patience700",
        N=12,
        n_restarts=5,
        mpnn_hidden=160,
        mpnn_epochs=8000,
        mpnn_patience=700,
        h_test=1.5,
        seed=43,
    ),
}

CONFIGS = {**CONFIGS_N6, **CONFIGS_N10, **CONFIGS_N12}


# ── Environment capture ──────────────────────────────────────────────────


def capture_environment() -> dict:
    env = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(_project_root),
        ).stdout.strip()
        git_branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            cwd=str(_project_root),
        ).stdout.strip()
        env["git_commit"] = git_hash
        env["git_branch"] = git_branch
    except Exception:
        env["git_commit"] = "unknown"
        env["git_branch"] = "unknown"

    for pkg in ["qiskit", "torch", "torch_geometric", "numpy"]:
        try:
            mod = __import__(pkg)
            env[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env[f"{pkg}_version"] = "not installed"
    return env


# ── Pipeline execution ───────────────────────────────────────────────────


def run_pipeline(cfg: PipelineConfig, verbose: bool = False, debug: bool = False) -> dict:
    """Execute the full V6.1 pipeline with the given configuration."""
    import numpy as np
    import torch
    from torch_geometric.data import Data

    from src.poc.v6.analysis_utils import WeightGradientAnalyzer
    from src.poc.v6.classical_solver import ClassicalSolver
    from src.poc.v6.config import VQEConfig
    from src.poc.v6.hamiltonian_builder import HamiltonianBuilder, make_lattice
    from src.poc.v6.hardware_deployer_v61 import HardwareDeployerV61
    from src.poc.v6.hva_builder import HVACircuitBuilder
    from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset, train_mpnn
    from src.poc.v6.vqe_optimizer import VQEOptimizer

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ── Configure logging (verbose/debug control level only) ──
    if verbose or debug:
        configure_pipeline_logging(verbose=verbose, debug=debug)

    # ── DiagnosticCollector is ALWAYS active (metrics always recorded) ──
    collector = DiagnosticCollector(verbose=verbose or debug, save_dir=RESULTS_DIR)

    result = {
        "config_name": cfg.name,
        "config": {
            "N": cfg.N,
            "J": str(cfg.J),
            "p_layers": cfg.p_layers,
            "topology": cfg.topology,
            "n_restarts": cfg.n_restarts,
            "maxiter": cfg.maxiter,
            "mpnn_hidden": cfg.mpnn_hidden,
            "mpnn_layers": cfg.mpnn_layers,
            "mpnn_epochs": cfg.mpnn_epochs,
            "mpnn_lr": cfg.mpnn_lr,
            "per_parameter_heads": cfg.per_parameter_heads,
            "use_edge_features": cfg.use_edge_features,
            "h_test": cfg.h_test,
            "fidelity_threshold": cfg.fidelity_threshold,
            "seed": cfg.seed,
            "h_grid": cfg.h_grid,
        },
        "phases": {},
        "success": True,
        "error": None,
    }

    # Attach config to collector for checkpoint metadata
    collector._config_dict = result["config"]

    N, _J, p = cfg.N, cfg.J, cfg.p_layers

    # Non-uniform h-grid selection
    if cfg.h_grid == "dense":
        # Dense grid: 45 points with Δh=0.025 near critical region
        h_coarse = np.arange(0.0, 0.8, 0.1)
        h_dense = np.arange(0.8, 1.45, 0.025)  # 2x denser near critical
        h_coarse2 = np.arange(1.5, 2.05, 0.1)
        h_values = np.unique(np.concatenate([h_coarse, h_dense, h_coarse2]))
    else:
        # Standard V6 pattern (27 points)
        h_coarse = np.arange(0.0, 0.8, 0.1)
        h_dense = np.arange(0.8, 1.45, 0.05)
        h_coarse2 = np.arange(1.5, 2.05, 0.1)
        h_values = np.unique(np.concatenate([h_coarse, h_dense, h_coarse2]))

    # Handle non-uniform J for ladder topology
    if cfg.J == "ladder_nonuniform":
        from src.poc.v6.hamiltonian_builder import generate_ladder

        _edges = generate_ladder(N)
        leg = N // 2
        n_leg_edges = 2 * (leg - 1)  # intra-leg edges (both legs)
        n_rung_edges = leg  # rung edges
        J_array = np.concatenate(
            [
                np.ones(n_leg_edges) * 1.0,  # J_leg = 1.0
                np.ones(n_rung_edges) * 0.5,  # J_rung = 0.5
            ]
        )
        J_param = J_array
    else:
        J_param = float(cfg.J)

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()
    base_lattice = make_lattice(cfg.topology, N, J=J_param, h=1.0)
    qc, _ = hva.create(N, p, base_lattice)

    # ── Phase 1: Exact diagonalization ──
    print(f"\n  Phase 1: Exact diag ({len(h_values)} h-points)...")
    t1 = time.time()
    exact_data = []
    for h in h_values:
        lat_h = make_lattice(cfg.topology, N, J=J_param, h=h)
        H = builder.build(lat_h)
        exact_data.append(solver.solve(H, lat_h))
    phase1_time = time.time() - t1
    result["phases"]["phase1"] = {
        "elapsed_s": round(phase1_time, 1),
        "n_points": len(exact_data),
        "e0_range": [exact_data[0].ground_energy, exact_data[-1].ground_energy],
        "gap_min": min(d.gap for d in exact_data),
    }
    print(f"    Done in {phase1_time:.1f}s")

    # ── Diagnostics: record Phase 1 ──
    collector.record_phase1(
        n_points=len(exact_data),
        elapsed_s=phase1_time,
        gap_min=min(d.gap for d in exact_data),
    )
    collector.save_checkpoint("phase1")

    # ── Phase 2: VQE descending sweep ──
    print(f"  Phase 2: VQE ({cfg.n_restarts} restarts, maxiter={cfg.maxiter})...")
    t2 = time.time()
    vqe_config = VQEConfig(
        p_layers=p,
        n_restarts=cfg.n_restarts,
        maxiter=cfg.maxiter,
        enable_callbacks=(verbose or debug),
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
    print(
        f"    Done in {phase2_time:.1f}s — avg fid={np.mean(fids) * 100:.1f}%, "
        f"≥93%: {n_above_threshold}/{len(fids)}, ≥99.5%: {n_above_995}/{len(fids)}"
    )

    # ── Diagnostics: record Phase 2 per-h VQE data ──
    per_h_time = phase2_time / len(vqe_results) if vqe_results else 0.0
    for i, vqe_r in enumerate(vqe_results):
        # Use n_iterations directly (always available, no callback dependency)
        n_iters = vqe_r.n_iterations
        # Restart energies not directly available from sweep; use empty list
        restart_energies: list[float] = []
        collector.record_vqe_point(
            h=float(h_values[i]),
            n_iters=n_iters,
            restart_energies=restart_energies,
            theta_opt=vqe_r.theta_opt,
            elapsed_s=per_h_time,
        )
    collector.save_checkpoint("phase2")

    # ── Phase 3: MPNN training ──
    print(
        f"  Phase 3: MPNN (h={cfg.mpnn_hidden}, L={cfg.mpnn_layers}, "
        f"epochs={cfg.mpnn_epochs}, per_param={cfg.per_parameter_heads}, "
        f"edge_feat={cfg.use_edge_features})..."
    )
    t3 = time.time()
    dataset = build_graph_dataset(
        base_lattice,
        h_values,
        np.array([r.theta_opt for r in vqe_results]),
        np.array([d.ground_energy for d in exact_data]),
        fidelities=fids,
        fidelity_threshold=cfg.fidelity_threshold,
        include_edge_features=cfg.use_edge_features,
    )

    model = MPNNPredictor(
        node_features=2,
        hidden_dim=cfg.mpnn_hidden,
        n_layers=cfg.mpnn_layers,
        output_dim=2 * p,
        per_parameter_heads=cfg.per_parameter_heads,
        use_edge_features=cfg.use_edge_features,
    )
    train_result = train_mpnn(
        model,
        dataset,
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
        "stop_reason": train_result.get("stop_reason", ""),
        "best_epoch": train_result.get("best_epoch", cfg.mpnn_epochs),
    }
    print(
        f"    Done in {phase3_time:.1f}s — MSE={train_result['final_mse']:.2e}, "
        f"graphs={len(dataset)}/{len(h_values)}"
    )

    # ── Diagnostics: record Phase 3 per-h error ──
    # Compute per-h MSE from model predictions vs actual theta values
    model.eval()
    per_h_mse_values = []
    with torch.no_grad():
        for graph in dataset:
            pred = model(graph).numpy().flatten()
            target = graph.y.numpy().flatten()
            mse_val = float(np.mean((pred - target) ** 2))
            per_h_mse_values.append(mse_val)
    per_h_mse_arr = np.array(per_h_mse_values)
    # Use h_values for the training points (filtered by fidelity)
    # dataset may have fewer points than h_values due to fidelity filter
    h_train = np.array([float(graph.x[0, 0]) for graph in dataset])
    collector.record_mpnn_per_h_error(h_train, per_h_mse_arr)
    collector.save_checkpoint("phase3")

    # ── Phase 4: V6.1 Deployment ──
    print(f"  Phase 4: Deploy h_test={cfg.h_test}...")
    t4 = time.time()
    lat_test = make_lattice(cfg.topology, N, J=J_param, h=cfg.h_test)
    H_test = builder.build(lat_test)
    exact_test = solver.solve(H_test, lat_test)

    model.eval()
    edge_idx, coord = builder.build_graph_data(base_lattice)
    x_test = torch.tensor(
        np.stack([np.full(N, cfg.h_test), coord.astype(float)], axis=1),
        dtype=torch.float32,
    )
    test_graph = Data(x=x_test, edge_index=torch.tensor(edge_idx, dtype=torch.long))

    # Add edge features for NNConv models
    if cfg.use_edge_features and isinstance(J_param, np.ndarray):
        j_values = np.concatenate([J_param, J_param])  # both directions
        test_graph.edge_attr = torch.tensor(j_values.reshape(-1, 1), dtype=torch.float32)

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
        "delta_e": deploy_result.delta_e,
        "delta_e_over_gap": deploy_result.delta_e_over_gap,
        "mag_x_pred": deploy_result.mag_x_pred,
        "corr_zz_pred": deploy_result.corr_zz_pred,
        "mag_x_error": deploy_result.mag_x_error,
        "corr_zz_error": deploy_result.corr_zz_error,
        "fidelity": deploy_result.fidelity,
        "adapt_iterations": deploy_result.adapt_iterations,
        "phase_label": deploy_result.phase_label,
        "checklist": checklist,
        "checklist_pass": n_pass,
        "checklist_total": len(checklist),
        "zne_r_squared": getattr(deploy_result, "zne_r_squared", None),
    }
    print(
        f"    Done in {phase4_time:.1f}s — ΔE/gap={deploy_result.delta_e_over_gap:.4f}, "
        f"checklist={n_pass}/{len(checklist)}, phase={deploy_result.phase_label}"
    )

    # ── Diagnostics: record Phase 4 deployment ──
    # Build per_layout_data from deploy_result if available
    per_layout_data = None
    if (
        hasattr(deploy_result, "energies_per_layout")
        and deploy_result.energies_per_layout
        and hasattr(deploy_result, "ces_values")
        and deploy_result.ces_values
    ):
        per_layout_data = {
            "energies": deploy_result.energies_per_layout,
            "ces_values": deploy_result.ces_values,
        }
    collector.record_deployment(
        h_test=cfg.h_test,
        result=deploy_result,
        per_layout_data=per_layout_data,
    )
    collector.save_checkpoint("phase4")

    # ── Gradient Analysis ──
    print("  Analysis: Weight gradients...")
    t5 = time.time()
    analyzer = WeightGradientAnalyzer(model)
    grad_result = analyzer.analyze(dataset)
    analysis_time = time.time() - t5
    result["phases"]["gradient_analysis"] = {
        "elapsed_s": round(analysis_time, 1),
        "peaks_detected": len(grad_result.peak_h_values),
        "peak_h_values": grad_result.peak_h_values,
        "peak_magnitudes": grad_result.peak_magnitudes,
        "critical_region_detected": grad_result.critical_region_detected,
        "gradient_norm_range": [
            float(grad_result.total_gradient_norms.min()),
            float(grad_result.total_gradient_norms.max()),
        ],
    }
    print(
        f"    Done in {analysis_time:.1f}s — peaks={len(grad_result.peak_h_values)}, "
        f"critical_region={grad_result.critical_region_detected}"
    )

    result["total_elapsed_s"] = round(time.time() - t1 + phase1_time, 1)

    # ── Diagnostics: always append to result and cleanup checkpoints ──
    result["diagnostics"] = collector.to_dict()
    collector.cleanup_checkpoints()

    return result


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="V6.1 Parametric Pipeline Runner")
    parser.add_argument(
        "--config",
        choices=list(CONFIGS.keys()) + ["all", "n6", "n10", "n12"],
        default="all",
        help="Configuration preset to run (default: all). Use 'n6', 'n10', or 'n12' for size-specific sets.",
    )
    parser.add_argument("--binnacle", action="store_true", help="Append binnacle entry")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable INFO logging, DiagnosticCollector, and VQE callbacks",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable DEBUG logging and all verbose features",
    )
    args = parser.parse_args()

    if args.config == "all":
        configs_to_run = list(CONFIGS.values())
    elif args.config == "n6":
        configs_to_run = list(CONFIGS_N6.values())
    elif args.config == "n10":
        configs_to_run = list(CONFIGS_N10.values())
    elif args.config == "n12":
        configs_to_run = list(CONFIGS_N12.values())
    else:
        configs_to_run = [CONFIGS[args.config]]

    print("=" * 60)
    print("  GNN-HVA v6.1 — Parametric Pipeline Runner")
    print(f"  Configs: {[c.name for c in configs_to_run]}")
    print("=" * 60)

    env = capture_environment()
    print(f"  Git: {env.get('git_branch', '?')}@{env.get('git_commit', '?')}")
    print(f"  Python: {env['python_version'].split()[0]}")

    all_results = []
    t_total = time.time()

    for cfg in configs_to_run:
        print(f"\n{'─' * 60}")
        print(f"  Config: {cfg.name}")
        print(
            f"    N={cfg.N}, p={cfg.p_layers}, restarts={cfg.n_restarts}, "
            f"MPNN h={cfg.mpnn_hidden} L={cfg.mpnn_layers} ep={cfg.mpnn_epochs}, "
            f"h_test={cfg.h_test}, per_param={cfg.per_parameter_heads}"
        )
        print(f"{'─' * 60}")

        try:
            result = run_pipeline(cfg, verbose=args.verbose, debug=args.debug)
            all_results.append(result)
        except Exception as e:
            print(f"\n  ❌ Config '{cfg.name}' FAILED: {e}")
            import traceback

            traceback.print_exc()
            all_results.append(
                {
                    "config_name": cfg.name,
                    "success": False,
                    "error": str(e),
                }
            )

    # ── Save results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = hashlib.sha1(f"parametric:{ts}".encode()).hexdigest()[:8]
    summary_path = RESULTS_DIR / f"parametric_run_{ts}_{run_id}.json"

    summary = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "runner": "run_v61_parametric.py",
        "environment": env,
        "configs_run": [c.name for c in configs_to_run],
        "results": all_results,
        "total_elapsed_s": round(time.time() - t_total, 1),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Summary table ──
    print(f"\n\n{'=' * 60}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Config':<12} {'ΔE/gap':<10} {'Checklist':<12} {'Phase':<14} {'MSE':<12} {'Time'}")
    print(f"  {'─' * 72}")
    for r in all_results:
        if not r.get("success", False):
            print(f"  {r['config_name']:<12} ❌ FAILED: {r.get('error', '?')[:40]}")
            continue
        p4 = r["phases"]["phase4"]
        p3 = r["phases"]["phase3"]
        de_gap = p4["delta_e_over_gap"]
        de_tag = "✅" if de_gap < 0.05 else "⚠️" if de_gap < 0.10 else "❌"
        total_t = sum(ph.get("elapsed_s", 0) for ph in r["phases"].values())
        print(
            f"  {r['config_name']:<12} {de_gap:.4f} {de_tag}  "
            f"{p4['checklist_pass']}/{p4['checklist_total']:<8} "
            f"{p4['phase_label']:<14} {p3['final_mse']:.2e}   {total_t:.0f}s"
        )

    print(f"\n  Total time: {time.time() - t_total:.0f}s")
    print(f"  Results saved: {summary_path}")

    # ── Binnacle ──
    if args.binnacle:
        BINNACLE_DIR.mkdir(parents=True, exist_ok=True)
        # Determine binnacle file from N value of first config
        n_val = configs_to_run[0].N
        binnacle_path = BINNACLE_DIR / f"binnacle-N{n_val}.md"
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"\n---\n\n## {now} — Parametric V6.1 Run ({len(configs_to_run)} configs, N={n_val})\n\n"
        ]
        lines.append(f"- Git: `{env.get('git_branch', '?')}` @ `{env.get('git_commit', '?')}`\n")
        lines.append("- Runner: `run_v61_parametric.py`\n")
        lines.append("- V6.1 features: HardwareDeployerV61, WeightGradientAnalyzer\n\n")
        lines.append("| Config | Seed | ΔE/gap | Checklist | Phase | MSE | h_test | Grad Peaks |\n")
        lines.append("|--------|------|--------|-----------|-------|-----|--------|------------|\n")
        for r in all_results:
            if not r.get("success"):
                lines.append(f"| {r['config_name']} | — | FAILED | — | — | — | — | — |\n")
                continue
            p4 = r["phases"]["phase4"]
            p3 = r["phases"]["phase3"]
            ga = r["phases"].get("gradient_analysis", {})
            seed = r["config"].get("seed", 42)
            lines.append(
                f"| {r['config_name']} | {seed} | {p4['delta_e_over_gap']:.4f} | "
                f"{p4['checklist_pass']}/{p4['checklist_total']} | "
                f"{p4['phase_label']} | {p3['final_mse']:.2e} | {p4['h_test']} | "
                f"{ga.get('peaks_detected', '?')} |\n"
            )
        lines.append("\n")
        with open(binnacle_path, "a") as f:
            f.writelines(lines)
        print(f"  Binnacle appended: {binnacle_path}")

    n_success = sum(1 for r in all_results if r.get("success"))
    return 0 if n_success == len(all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
