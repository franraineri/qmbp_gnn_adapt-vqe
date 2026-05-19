#!/usr/bin/env python
"""
LEGACY: GNN-HVA v6.0 — Multi-Run Benchmark

╔══════════════════════════════════════════════════════════════════════════╗
║  SUPERSEDED by scripts/run_v61_parametric.py for V6.1 benchmarking.    ║
║  Kept for reproducibility of the 40+ V6.0 benchmark runs.              ║
║  Uses deprecated HardwareDeployer, GATPredictor, augment_graph_dataset,║
║  and QRCPipeline.                                                       ║
╚══════════════════════════════════════════════════════════════════════════╝

Executes the full 4-phase pipeline N times with different random seeds,
collects per-run metrics, and appends a summary to the binnacle.

Usage:
    python scripts/benchmark_v6.py              # 3 runs (default)
    python scripts/benchmark_v6.py --runs 5     # 5 runs
    python scripts/benchmark_v6.py --runs 3 --h-test 1.5

Each run uses seed = 42 + run_index so results are reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is importable
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import torch
from torch_geometric.data import Data

from src.poc.v6 import (
    ClassicalSolver,
    HamiltonianBuilder,
    HVACircuitBuilder,
    VQEConfig,
    VQEOptimizer,
    make_lattice,
)
from src.poc.v6.hardware_deployer import HardwareDeployer
from src.poc.v6.mpnn_predictor import (
    MPNNPredictor,
    build_graph_dataset,
    train_mpnn,
)

# DEPRECATED: GATPredictor and augment_graph_dataset were in experimental/ (now deleted).
# benchmark_v6.py is kept for reference only; GAT model type will raise if selected.
try:
    from src.poc.v6.experimental.augmentation import augment_graph_dataset
    from src.poc.v6.experimental.gat_predictor import GATPredictor
except ImportError:
    GATPredictor = None  # type: ignore[assignment,misc]
    augment_graph_dataset = None  # type: ignore[assignment]
from src.poc.v6.pipeline_utils import assert_observable_locality
from src.poc.v6.qrc_pipeline import QRCPipeline

# ── Single run ───────────────────────────────────────────────────────────


def run_pipeline(
    seed: int,
    h_test: float = 1.25,
    n_qubits: int = 6,
    n_restarts: int = 3,
    restart_sigma: float = 0.1,
    vqe_maxiter: int = 1000,
    mpnn_epochs: int = 4000,
    mpnn_hidden: int = 64,
    mpnn_layers: int = 3,
    mpnn_lr: float = 1e-3,
    mpnn_patience: int = 150,
    fid_threshold: float = 0.93,
    h_points: int = 27,
    augment: bool = False,
    model_type: str = "gin",
    deployer_version: str = "v6.0",
    per_parameter_heads: bool = False,
) -> dict:
    """Execute the full V6 pipeline once and return metrics.

    Parameters
    ----------
    deployer_version : str
        "v6.0" for original deployer, "v6.1" for HardwareDeployerV61 with
        full error mitigation stack (simulation mode).
    per_parameter_heads : bool
        When True and deployer_version="v6.1", use separate ZZ/X output heads.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    N, J, p = n_qubits, 1.0, 2

    # Build h-grid with configurable density
    # Denser near critical region h∈[0.8,1.4], coarser elsewhere
    if h_points <= 27:
        # Original grid: 27 points
        h_coarse = np.arange(0.0, 0.8, 0.1)
        h_dense = np.arange(0.8, 1.45, 0.05)
        h_coarse2 = np.arange(1.5, 2.05, 0.1)
    elif h_points <= 40:
        # Denser grid: ~40 points (Δh=0.05 everywhere, Δh=0.025 near critical)
        h_coarse = np.arange(0.0, 0.8, 0.05)
        h_dense = np.arange(0.8, 1.45, 0.025)
        h_coarse2 = np.arange(1.5, 2.05, 0.05)
    else:
        # Very dense: ~55 points (Δh=0.05 coarse, Δh=0.02 critical)
        h_coarse = np.arange(0.0, 0.8, 0.05)
        h_dense = np.arange(0.8, 1.45, 0.02)
        h_coarse2 = np.arange(1.5, 2.05, 0.05)
    h_values = np.unique(np.concatenate([h_coarse, h_dense, h_coarse2]))

    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()
    base_lattice = make_lattice("chain_1d", N, J=J, h=1.0)
    qc, _ = hva.create(N, p, base_lattice)

    # Phase 1
    t1 = time.time()
    exact_data = []
    for h in h_values:
        lat_h = make_lattice("chain_1d", N, J=J, h=h)
        exact_data.append(solver.solve(builder.build(lat_h), lat_h))
    phase1_time = time.time() - t1

    # Phase 2
    t2 = time.time()
    config = VQEConfig(
        n_restarts=n_restarts,
        restart_sigma=restart_sigma,
        maxiter=vqe_maxiter,
        ftol=1e-14,
        enable_callbacks=False,
    )
    vqe_results = VQEOptimizer(config).descending_sweep(
        h_values,
        qc,
        base_lattice,
        exact_data,
    )
    fids = [r.fidelity for r in vqe_results]
    phase2_time = time.time() - t2

    # Phase 3
    t3 = time.time()
    dataset = build_graph_dataset(
        base_lattice,
        h_values,
        np.array([r.theta_opt for r in vqe_results]),
        np.array([d.ground_energy for d in exact_data]),
        fidelities=np.array(fids),
        fidelity_threshold=fid_threshold,
    )

    # Data augmentation (interpolated θ between adjacent h-points)
    if augment:
        dataset = augment_graph_dataset(dataset)

    # Model selection
    if model_type == "gat":
        model = GATPredictor(
            node_features=2,
            hidden_dim=mpnn_hidden,
            n_layers=mpnn_layers,
            output_dim=2 * p,
        )
    else:
        model = MPNNPredictor(
            node_features=2,
            hidden_dim=mpnn_hidden,
            n_layers=mpnn_layers,
            output_dim=2 * p,
            per_parameter_heads=per_parameter_heads,
        )
    train_result = train_mpnn(
        model, dataset, n_epochs=mpnn_epochs, lr=mpnn_lr, patience=mpnn_patience
    )
    phase3_time = time.time() - t3

    # Phase 4 — Adapt-VQE
    t4 = time.time()
    lat_test = make_lattice("chain_1d", N, J=J, h=h_test)
    H_test = builder.build(lat_test)
    exact_test = solver.solve(H_test, lat_test)

    model.eval()
    edge_idx, coord = builder.build_graph_data(base_lattice)
    x_test = torch.tensor(
        np.stack([np.full(N, h_test), coord.astype(float)], axis=1),
        dtype=torch.float32,
    )
    test_graph = Data(x=x_test, edge_index=torch.tensor(edge_idx, dtype=torch.long))
    with torch.no_grad():
        theta_pred = model(test_graph).numpy().flatten()

    ops_x, ops_zz = builder.build_local_observables(base_lattice)
    assert_observable_locality(ops_x + ops_zz, base_lattice.edges)

    deployer = HardwareDeployer()
    adapt = deployer.deploy_adapt_vqe(qc, H_test, theta_pred, lat_test, exact_test)

    # Phase 4 — QRC
    qrc = QRCPipeline(seed=seed)
    qrc.build_reservoir(N, p, base_lattice)
    qrc.train_readout(
        h_values,
        np.array([d.mag_x for d in exact_data]),
        np.array([d.corr_zz for d in exact_data]),
    )
    qrc_result = deployer.deploy_qrc(qrc, h_test, exact_test)
    phase4_time = time.time() - t4

    n_pass = sum(adapt.metrics_checklist.values())

    # ── V6.1 deployer metrics (optional) ──
    v61_metrics = {}
    if deployer_version == "v6.1":
        from src.poc.v6.analysis_utils import WeightGradientAnalyzer
        from src.poc.v6.hardware_deployer_v61 import HardwareDeployerV61

        deployer_v61 = HardwareDeployerV61(mode="simulation")
        result_v61 = deployer_v61.deploy_adapt_vqe(qc, H_test, theta_pred, lat_test, exact_test)
        v61_metrics = {
            "v61_energy": result_v61.predicted_energy,
            "v61_delta_e": result_v61.delta_e,
            "v61_delta_e_over_gap": result_v61.delta_e_over_gap,
            "v61_mag_x": result_v61.mag_x_pred,
            "v61_corr_zz": result_v61.corr_zz_pred,
            "v61_phase_label": result_v61.phase_label,
            "v61_sigma": result_v61.sigma,
            "v61_extrapolation_method": result_v61.extrapolation_method,
        }

        # Weight gradient analysis
        analyzer = WeightGradientAnalyzer(model)
        grad_result = analyzer.analyze(dataset)
        v61_metrics["gradient_norm_min"] = float(grad_result.total_gradient_norms.min())
        v61_metrics["gradient_norm_max"] = float(grad_result.total_gradient_norms.max())
        v61_metrics["gradient_peaks"] = len(grad_result.peak_h_values)
        v61_metrics["gradient_peak_h_values"] = grad_result.peak_h_values
        v61_metrics["critical_region_detected"] = grad_result.critical_region_detected

        # Per-parameter head losses (if enabled)
        if per_parameter_heads and train_result.get("zz_head_loss_history"):
            v61_metrics["zz_head_loss_final"] = train_result["zz_head_loss_history"][-1]
            v61_metrics["x_head_loss_final"] = train_result["x_head_loss_history"][-1]

    return {
        "seed": seed,
        "h_test": h_test,
        "config": {
            "n_qubits": n_qubits,
            "n_restarts": n_restarts,
            "restart_sigma": restart_sigma,
            "vqe_maxiter": vqe_maxiter,
            "mpnn_epochs": mpnn_epochs,
            "mpnn_hidden": mpnn_hidden,
            "mpnn_layers": mpnn_layers,
            "mpnn_lr": mpnn_lr,
            "mpnn_patience": mpnn_patience,
            "fid_threshold": fid_threshold,
            "h_points": h_points,
            "augment": augment,
            "model_type": model_type,
        },
        "n_h_points": len(h_values),
        "n_training_graphs": len(dataset),
        "phase2_fid_avg": float(np.mean(fids)),
        "phase2_fid_good": sum(1 for f in fids if f >= 0.995),
        "phase3_final_mse": train_result["final_mse"],
        "phase3_stopped_early": train_result["stopped_early"],
        "adapt_delta_e_over_gap": adapt.delta_e_over_gap,
        "adapt_mag_x_error": adapt.mag_x_error,
        "adapt_corr_zz_error": adapt.corr_zz_error,
        "adapt_delta_e": adapt.delta_e,
        "adapt_fidelity": adapt.fidelity,
        "adapt_iterations": adapt.adapt_iterations,
        "adapt_checklist": n_pass,
        "adapt_phase": adapt.phase_label,
        "qrc_mag_x_error": qrc_result.mag_x_error,
        "qrc_corr_zz_error": qrc_result.corr_zz_error,
        "qrc_phase": qrc_result.phase_label,
        "time_phase1": phase1_time,
        "time_phase2": phase2_time,
        "time_phase3": phase3_time,
        "time_phase4": phase4_time,
        "time_total": phase1_time + phase2_time + phase3_time + phase4_time,
        "deployer_version": deployer_version,
        "per_parameter_heads": per_parameter_heads,
        **v61_metrics,
    }


# ── Binnacle formatting ──────────────────────────────────────────────────


def format_binnacle_entry(
    all_results: list[dict], h_test: float, n_runs: int, label: str = ""
) -> str:
    """Format results as a binnacle markdown entry."""
    now = datetime.now().strftime("%Y-%m-%d")

    cfg = all_results[0].get("config", {})
    config_desc = (
        f"restarts={cfg.get('n_restarts', '?')}, maxiter={cfg.get('vqe_maxiter', '?')}, "
        f"MPNN(h={cfg.get('mpnn_hidden', '?')}, L={cfg.get('mpnn_layers', '?')}, "
        f"ep={cfg.get('mpnn_epochs', '?')}, lr={cfg.get('mpnn_lr', '?')}, "
        f"pat={cfg.get('mpnn_patience', '?')}), fid≥{cfg.get('fid_threshold', '?')}"
    )

    # Aggregate stats
    checklists = [r["adapt_checklist"] for r in all_results]
    de_gaps = [r["adapt_delta_e_over_gap"] * 100 for r in all_results]
    mx_errs = [r["adapt_mag_x_error"] for r in all_results]
    zz_errs = [r["adapt_corr_zz_error"] for r in all_results]
    fids = [r["adapt_fidelity"] for r in all_results]
    times = [r["time_total"] for r in all_results]

    title = f"V6.0 Benchmark — {label}" if label else f"V6.0 Benchmark ({n_runs} runs)"

    lines = [
        "\n---\n",
        f"## {now} — {title}\n\n",
        "### Configuration\n",
        f"- System: 1D TFIM, N={cfg.get('n_qubits', '?')}, p=2, {all_results[0].get('n_h_points', '?')} h-points, h_test={h_test}\n",
        f"- {config_desc}\n",
        f"- Seeds: {[r['seed'] for r in all_results]}\n",
        f"\n### Per-Run Results (Adapt-VQE at h={h_test})\n\n",
        "| Run | Seed | ΔE/gap | ⟨X⟩ err | ⟨ZZ⟩ err | ΔE | Fidelity | ADAPT | Checklist | Time |\n",
        "|-----|------|--------|---------|----------|-----|----------|-------|-----------|------|\n",
    ]

    for i, r in enumerate(all_results):
        de_gap_s = f"{r['adapt_delta_e_over_gap'] * 100:.2f}%"
        de_gap_ok = "✅" if r["adapt_delta_e_over_gap"] < 0.05 else "❌"
        mx_ok = "✅" if r["adapt_mag_x_error"] < 1e-2 else "❌"
        zz_ok = "✅" if r["adapt_corr_zz_error"] < 1e-2 else "❌"
        de_ok = "✅" if r["adapt_delta_e"] < 1e-2 else "❌"
        fid_ok = "✅" if r["adapt_fidelity"] is not None and r["adapt_fidelity"] >= 0.995 else "❌"
        lines.append(
            f"| {i + 1} | {r['seed']} | {de_gap_s} {de_gap_ok} | "
            f"{r['adapt_mag_x_error']:.2e} {mx_ok} | "
            f"{r['adapt_corr_zz_error']:.2e} {zz_ok} | "
            f"{r['adapt_delta_e']:.2e} {de_ok} | "
            f"{r['adapt_fidelity']:.4f} {fid_ok} | "
            f"{r['adapt_iterations']} | "
            f"**{r['adapt_checklist']}/6** | "
            f"{r['time_total']:.0f}s |\n"
        )

    lines.append("\n### Aggregate Statistics\n\n")
    lines.append("| Metric | Mean | Std | Min | Max |\n")
    lines.append("|--------|------|-----|-----|-----|\n")
    lines.append(
        f"| ΔE/gap | {np.mean(de_gaps):.2f}% | {np.std(de_gaps):.2f}% | {np.min(de_gaps):.2f}% | {np.max(de_gaps):.2f}% |\n"
    )
    lines.append(
        f"| ⟨X⟩ error | {np.mean(mx_errs):.2e} | {np.std(mx_errs):.2e} | {np.min(mx_errs):.2e} | {np.max(mx_errs):.2e} |\n"
    )
    lines.append(
        f"| ⟨ZZ⟩ error | {np.mean(zz_errs):.2e} | {np.std(zz_errs):.2e} | {np.min(zz_errs):.2e} | {np.max(zz_errs):.2e} |\n"
    )
    lines.append(
        f"| Fidelity | {np.mean(fids):.4f} | {np.std(fids):.4f} | {np.min(fids):.4f} | {np.max(fids):.4f} |\n"
    )
    lines.append(
        f"| Checklist | {np.mean(checklists):.1f}/6 | {np.std(checklists):.1f} | {min(checklists)}/6 | {max(checklists)}/6 |\n"
    )
    lines.append(
        f"| Runtime | {np.mean(times):.0f}s | {np.std(times):.0f}s | {np.min(times):.0f}s | {np.max(times):.0f}s |\n"
    )

    lines.append("\n### Key Observations\n\n")
    lines.append(
        f"1. ΔE/gap (primary metric): {'all pass' if all(d < 5 for d in de_gaps) else 'some fail'} across {n_runs} runs.\n"
    )
    lines.append(f"2. Checklist range: {min(checklists)}/6 – {max(checklists)}/6.\n")
    lines.append(
        f"3. Results are {'stable' if np.std(checklists) < 1 else 'variable'} across seeds (std={np.std(checklists):.1f}).\n"
    )

    return "".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="V6 multi-run benchmark")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--h-test", type=float, default=1.25, help="Test h value")
    parser.add_argument("--n-qubits", type=int, default=6, help="Number of qubits")
    parser.add_argument("--n-restarts", type=int, default=5, help="VQE restarts")
    parser.add_argument(
        "--restart-sigma", type=float, default=0.1, help="VQE restart perturbation std"
    )
    parser.add_argument("--vqe-maxiter", type=int, default=1000, help="VQE max iterations")
    parser.add_argument("--mpnn-epochs", type=int, default=4000, help="MPNN training epochs")
    parser.add_argument("--mpnn-hidden", type=int, default=64, help="MPNN hidden dim")
    parser.add_argument("--mpnn-layers", type=int, default=3, help="MPNN GINConv layers")
    parser.add_argument("--mpnn-lr", type=float, default=1e-3, help="MPNN learning rate")
    parser.add_argument("--mpnn-patience", type=int, default=150, help="MPNN scheduler patience")
    parser.add_argument("--h-points", type=int, default=27, help="H-grid density (27, 40, or 55)")
    parser.add_argument(
        "--augment", action="store_true", help="Augment training data via θ interpolation"
    )
    parser.add_argument(
        "--model", choices=["gin", "gat"], default="gin", help="MPNN architecture (gin or gat)"
    )
    parser.add_argument(
        "--fid-threshold", type=float, default=0.93, help="Fidelity filter threshold"
    )
    parser.add_argument("--no-binnacle", action="store_true", help="Skip binnacle append")
    parser.add_argument("--label", type=str, default="", help="Custom label for this experiment")
    parser.add_argument(
        "--deployer",
        choices=["v6.0", "v6.1"],
        default="v6.0",
        help="Deployer version: v6.0 (original) or v6.1 (full mitigation stack, simulation mode)",
    )
    parser.add_argument(
        "--per-param-heads",
        action="store_true",
        help="Use per-parameter output heads (V6.1 MPNN enhancement)",
    )
    args = parser.parse_args()

    config_str = (
        f"N={args.n_qubits}, restarts={args.n_restarts}, σ={args.restart_sigma}, "
        f"maxiter={args.vqe_maxiter}, "
        f"mpnn=[h={args.mpnn_hidden}, L={args.mpnn_layers}, ep={args.mpnn_epochs}, "
        f"lr={args.mpnn_lr}, pat={args.mpnn_patience}], "
        f"fid≥{args.fid_threshold}, h_pts={args.h_points}"
    )
    label = args.label or config_str

    print(f"{'=' * 60}")
    print(f"  GNN-HVA v6.0 Benchmark — {args.runs} runs, h_test={args.h_test}")
    print(f"  Config: {config_str}")
    print(f"{'=' * 60}\n")

    all_results = []
    for i in range(args.runs):
        seed = 42 + i
        print(f"--- Run {i + 1}/{args.runs} (seed={seed}) ---")
        t0 = time.time()
        result = run_pipeline(
            seed=seed,
            h_test=args.h_test,
            n_qubits=args.n_qubits,
            n_restarts=args.n_restarts,
            restart_sigma=args.restart_sigma,
            vqe_maxiter=args.vqe_maxiter,
            mpnn_epochs=args.mpnn_epochs,
            mpnn_hidden=args.mpnn_hidden,
            mpnn_layers=args.mpnn_layers,
            mpnn_lr=args.mpnn_lr,
            mpnn_patience=args.mpnn_patience,
            fid_threshold=args.fid_threshold,
            h_points=args.h_points,
            augment=args.augment,
            model_type=args.model,
            deployer_version=args.deployer,
            per_parameter_heads=args.per_param_heads,
        )
        elapsed = time.time() - t0
        all_results.append(result)
        print(
            f"  Checklist: {result['adapt_checklist']}/6, "
            f"ΔE/gap={result['adapt_delta_e_over_gap'] * 100:.2f}%, "
            f"fid={result['adapt_fidelity']:.4f}, "
            f"time={elapsed:.0f}s\n"
        )

    # Save raw JSON
    output_dir = _root / "scripts" / "benchmark_results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"benchmark_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Raw results saved: {json_path}")

    # Format and append to binnacle
    entry = format_binnacle_entry(all_results, args.h_test, args.runs, label)
    print(f"\n{entry}")

    if not args.no_binnacle:
        # Append to the correct binnacle based on N
        n_qubits = all_results[0].get("config", {}).get("n_qubits", 6)
        binnacle_name = "binnacle-N6.md" if n_qubits <= 6 else "binnacle-N10.md"
        binnacle_path = _root / "documentation" / binnacle_name
        with open(binnacle_path, "a") as f:
            f.write(entry)
        print(f"Appended to: {binnacle_path}")

    # Summary
    checklists = [r["adapt_checklist"] for r in all_results]
    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK COMPLETE — {args.runs} runs")
    print(
        f"  Checklist: {min(checklists)}/6 – {max(checklists)}/6 (mean {np.mean(checklists):.1f})"
    )
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
