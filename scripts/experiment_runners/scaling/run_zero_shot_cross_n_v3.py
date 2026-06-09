#!/usr/bin/env python3
"""A1-v3: Zero-shot Cross-N — comparison of GNN vs direct interpolation.

This script diagnoses the cross-N generalization failure and tests
multiple prediction strategies:

1. GNN with BatchNorm disabled (train with BN off)
2. GNN with InstanceNorm (size-invariant normalization)
3. Direct scipy interpolation baseline (no GNN)

The key insight from v2 runs: theta_x ≈ 0.39 is near-constant across
N=40 and N=80, yet the GNN predicts 0.24-0.31. This is a ~30% undershoot
caused by BatchNorm statistics collapsing on chain_1d (all nodes identical
→ zero variance within a graph).

Fix strategy: Replace BatchNorm1d with InstanceNorm1d or remove it entirely.
For chain_1d where nodes are uniform, BN is actively harmful.

Usage:
    python scripts/experiment_runners/scaling/run_zero_shot_cross_n_v3.py \\
        --source-file results/scaling/scaling_N40_aer_mps_20260608_001053.json \\
                      results/scaling/scaling_N80_aer_mps_20260607_211634.json \\
        --target-n 60
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import torch
from torch_geometric.data import Data

from qmbp_simulation import (
    ClassicalSolver,
    HamiltonianBuilder,
    HVACircuitBuilder,
    make_lattice,
)
from qmbp_simulation.execution import MPSBackend
from qmbp_simulation.predictors import MPNNPredictor, train_mpnn
from qmbp_simulation.utils.helpers import json_dump

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Custom MPNN without BatchNorm (fix for chain_1d zero-variance problem)
# ═══════════════════════════════════════════════════════════════════════════════
# NOTE: This script previously defined a custom MPNNNoBN class.
# It is now replaced by `MPNNPredictor(norm_type="none")` from the package.
# The package fix (norm_type param) was applied 2026-06-08.
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers (shared with v2)
# ═══════════════════════════════════════════════════════════════════════════════


def canonicalize_theta(theta: np.ndarray) -> np.ndarray:
    """Enforce theta_x > 0 sign convention."""
    if len(theta) == 0:
        return theta
    if theta[-1] < 0:
        return -theta
    return theta


def _load_source_data(source_path: Path, source_seed: int):
    """Load and validate a single source file. Returns (n, topology, h_values, theta_opt, e_dmrg)."""
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    with open(source_path) as f:
        data = json.load(f)
    meta = data["metadata"]
    n = meta["n"]
    topology = meta["topology"]
    seed_runs = [r for r in data["vqe_results"] if r["seed"] == source_seed]
    if not seed_runs:
        raise ValueError(f"Seed {source_seed} not found in {source_path}")
    results = seed_runs[0]["results"]
    if "theta_opt" not in results[0]:
        raise ValueError(f"No theta_opt in {source_path}")
    h_values = np.array([r["h"] for r in results])
    theta_opt = np.array([canonicalize_theta(np.array(r["theta_opt"])) for r in results])
    e_dmrg = np.array([r["dmrg_energy"] for r in results])
    return n, topology, h_values, theta_opt, e_dmrg


def _build_dataset_no_bn(sources, use_n_feature: bool) -> list[Data]:
    """Build dataset with optional N/100 feature."""
    builder = HamiltonianBuilder()
    dataset: list[Data] = []
    for n, topology, h_values, theta_opt, e_dmrg in sources:
        lattice = make_lattice(topology, n, J=1.0, h=float(h_values[0]))
        edge_index_np, coord = builder.build_graph_data(lattice)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        for i, h in enumerate(h_values):
            cols = [np.full(n, float(h)), coord.astype(float)]
            if use_n_feature:
                cols.append(np.full(n, n / 100.0))
            x = torch.tensor(np.stack(cols, axis=1), dtype=torch.float32)
            y = torch.tensor(theta_opt[i], dtype=torch.float32)
            data = Data(x=x, edge_index=edge_index, y=y)
            data.e_exact = float(e_dmrg[i])
            dataset.append(data)
    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 3: Direct interpolation baseline
# ═══════════════════════════════════════════════════════════════════════════════


def interpolation_predict(sources, target_n: int, h_val: float) -> np.ndarray:
    """Predict theta for (target_n, h_val) via 2D linear interpolation.

    Uses the relationship: theta is a smooth function of (h, N).
    For TFIM chain_1d: theta_x ~ constant (~0.39), theta_zz ~ f(h/N).
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    # Collect all (h, N) -> theta pairs
    points = []
    values_zz = []
    values_x = []
    for n, _, h_values, theta_opt, _ in sources:
        for i, h in enumerate(h_values):
            points.append([h, n])
            values_zz.append(theta_opt[i, 0])
            values_x.append(theta_opt[i, 1])

    points = np.array(points)
    values_zz = np.array(values_zz)
    values_x = np.array(values_x)

    # Try linear interpolation, fall back to nearest
    query = np.array([[h_val, target_n]])

    interp_zz = LinearNDInterpolator(points, values_zz)
    interp_x = LinearNDInterpolator(points, values_x)

    theta_zz = interp_zz(query).flatten()[0]
    theta_x = interp_x(query).flatten()[0]

    # If NaN (outside convex hull), use nearest neighbor
    if np.isnan(theta_zz):
        nn_zz = NearestNDInterpolator(points, values_zz)
        theta_zz = nn_zz(query).flatten()[0]
    if np.isnan(theta_x):
        nn_x = NearestNDInterpolator(points, values_x)
        theta_x = nn_x(query).flatten()[0]

    return np.array([theta_zz, theta_x])


# ═══════════════════════════════════════════════════════════════════════════════
# Deploy and evaluate
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_theta(
    theta_pred: np.ndarray,
    n_target: int,
    h_val: float,
    topology: str,
    strategy: str,
    precision: float,
) -> dict:
    """Evaluate a theta prediction at target N, h.

    Returns a dict with comprehensive metrics for analysis.
    """
    t0 = time.time()
    builder = HamiltonianBuilder()
    hva = HVACircuitBuilder()
    solver = ClassicalSolver()
    backend = MPSBackend(strategy=strategy, chi_max=64, precision=precision, seed=42)

    lattice = make_lattice(topology, n_target, J=1.0, h=h_val)
    H = builder.build(lattice)
    circuit, _ = hva.create(n_target, 1, lattice)
    e_pred = backend.evaluate(circuit, H, theta_pred)
    gt = solver.solve(H, lattice, method="dmrg")
    de_gap = abs(e_pred - gt.ground_energy) / max(gt.gap, 1e-10)
    energy_error = float(e_pred - gt.ground_energy)
    elapsed = time.time() - t0

    return {
        "h": float(h_val),
        "e_pred": float(e_pred),
        "e_dmrg": float(gt.ground_energy),
        "gap": float(gt.gap),
        "de_gap": float(de_gap),
        "energy_error": energy_error,
        "energy_error_abs": abs(energy_error),
        "variational_ok": energy_error >= -1e-6,  # E_pred ≥ E_exact (within numerical noise)
        "theta_pred": theta_pred.tolist(),
        "passed": bool(de_gap < 0.05),
        "time_s": elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A1-v3: Zero-shot cross-N with BN-free GNN + interpolation baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-file", type=str, required=True, nargs="+")
    parser.add_argument("--source-seed", type=int, default=42)
    parser.add_argument("--target-n", type=int, default=60)
    parser.add_argument("--target-h-values", type=float, nargs="+", default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=6000)
    parser.add_argument("--strategy", type=str, default="aer_mps")
    parser.add_argument("--precision", type=float, default=0.005)
    parser.add_argument("--output-dir", type=str, default="results/scaling/zero_shot")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Load sources
    source_paths = [Path(p) for p in args.source_file]
    sources = []
    for sp in source_paths:
        try:
            sources.append(_load_source_data(sp, args.source_seed))
        except (FileNotFoundError, ValueError) as e:
            logger.error(str(e))
            return 1

    topology = sources[0][1]
    theta_dim = sources[0][3].shape[1]
    n_source_sizes = sorted(set(s[0] for s in sources))
    n_target = args.target_n

    # Auto h-values
    if args.target_h_values:
        h_test = sorted(args.target_h_values, reverse=True)
    else:
        h_min = 1.0 + 0.020 * n_target**1.31
        h_test = np.linspace(h_min + 1.5, h_min + 0.5, 5).tolist()

    logger.info("=" * 60)
    logger.info("A1-v3: Cross-N Comparison (BN-free GNN vs Interpolation)")
    logger.info(f"  Sources: N={n_source_sizes}, Target: N={n_target}")
    logger.info(f"  h_test = {[f'{h:.3f}' for h in h_test]}")
    logger.info("=" * 60)

    t_experiment_start = time.time()

    # ── Strategy A: GNN without BatchNorm (3 features) ───────────────
    logger.info("\n─── Strategy A: GNN without BatchNorm (3feat) ───")
    dataset = _build_dataset_no_bn(sources, use_n_feature=True)
    model = MPNNPredictor(
        node_features=3,
        hidden_dim=args.hidden_dim,
        n_layers=3,
        output_dim=theta_dim,
        norm_type="none",
    )
    n_model_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model: {n_model_params:,} parameters")

    t0 = time.time()
    metrics_a = train_mpnn(model, dataset, n_epochs=args.n_epochs, seed=42)
    train_time_a = time.time() - t0
    logger.info(f"  Training: MSE={metrics_a['final_mse']:.2e}, time={train_time_a:.1f}s")
    if metrics_a.get("stopped_early"):
        logger.warning(f"  Early stop: {metrics_a.get('stop_reason')}")

    model.eval()
    builder = HamiltonianBuilder()
    lattice_ref = make_lattice(topology, n_target, J=1.0, h=h_test[0])
    edge_index_np, coord = builder.build_graph_data(lattice_ref)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    results_a = []
    for h_val in h_test:
        cols = [np.full(n_target, h_val), coord.astype(float), np.full(n_target, n_target / 100.0)]
        x = torch.tensor(np.stack(cols, axis=1), dtype=torch.float32)
        graph = Data(x=x, edge_index=edge_index, batch=torch.zeros(n_target, dtype=torch.long))
        with torch.no_grad():
            theta_pred = model(graph).numpy().flatten()
        theta_pred = canonicalize_theta(theta_pred)
        r = evaluate_theta(theta_pred, n_target, h_val, topology, args.strategy, args.precision)
        status = "✅" if r["passed"] else "❌"
        var_flag = "" if r["variational_ok"] else " ⚠VARIATIONAL"
        logger.info(
            f"  h={h_val:.3f}: theta=[{theta_pred[0]:.5f}, {theta_pred[1]:.5f}]  "
            f"ΔE/gap={r['de_gap'] * 100:.2f}% {status}{var_flag} ({r['time_s']:.1f}s)"
        )
        results_a.append(r)

    # ── Strategy B: Direct interpolation (no ML) ─────────────────────
    logger.info("\n─── Strategy B: Direct Interpolation (scipy) ───")
    results_b = []
    for h_val in h_test:
        theta_pred = interpolation_predict(sources, n_target, h_val)
        theta_pred = canonicalize_theta(theta_pred)
        r = evaluate_theta(theta_pred, n_target, h_val, topology, args.strategy, args.precision)
        status = "✅" if r["passed"] else "❌"
        var_flag = "" if r["variational_ok"] else " ⚠VARIATIONAL"
        logger.info(
            f"  h={h_val:.3f}: theta=[{theta_pred[0]:.5f}, {theta_pred[1]:.5f}]  "
            f"ΔE/gap={r['de_gap'] * 100:.2f}% {status}{var_flag} ({r['time_s']:.1f}s)"
        )
        results_b.append(r)

    t_experiment_total = time.time() - t_experiment_start

    # ── Summary ──────────────────────────────────────────────────────
    de_a = [r["de_gap"] for r in results_a]
    de_b = [r["de_gap"] for r in results_b]
    pass_a = sum(1 for r in results_a if r["passed"])
    pass_b = sum(1 for r in results_b if r["passed"])
    var_ok_a = all(r["variational_ok"] for r in results_a)
    var_ok_b = all(r["variational_ok"] for r in results_b)

    logger.info("\n─── Comparison ───")
    logger.info(
        f"  GNN (no BN, 3feat):    mean={np.mean(de_a) * 100:.3f}%, pass={pass_a}/{len(h_test)}, variational={'OK' if var_ok_a else 'VIOLATION'}"
    )
    logger.info(
        f"  Interpolation:         mean={np.mean(de_b) * 100:.3f}%, pass={pass_b}/{len(h_test)}, variational={'OK' if var_ok_b else 'VIOLATION'}"
    )
    logger.info("  v2 3feat (with BN):    mean=18.5%, pass=0/5 (reference)")
    logger.info("  v2 2feat (with BN):    mean=9.5%, pass=1/5 (reference)")
    logger.info(f"  Total time: {t_experiment_total:.1f}s ({t_experiment_total / 60:.1f}m)")

    # ── Save results ─────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    n_tag = "_".join(str(n) for n in n_source_sizes)
    out_path = output_dir / f"zero_shot_v3_N{n_tag}_to_N{n_target}_{timestamp}.json"

    # Extract last 10 MSE values for convergence verification
    mse_history = metrics_a.get("mse_history", [])
    mse_last_10 = mse_history[-10:] if len(mse_history) >= 10 else mse_history

    json_dump(
        {
            "experiment": "zero_shot_cross_n_v3",
            "version": "3.1",
            "metadata": {
                "n_source_sizes": n_source_sizes,
                "n_target": n_target,
                "topology": topology,
                "source_seed": args.source_seed,
                "source_files": [str(p) for p in source_paths],
                "strategy": args.strategy,
                "precision": args.precision,
                "hidden_dim": args.hidden_dim,
                "n_epochs": args.n_epochs,
                "n_model_params": n_model_params,
                "n_training_points": len(dataset),
                "de_gap_threshold": 0.05,
            },
            "environment": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
                "torch_version": torch.__version__,
            },
            "timing": {
                "training_s": train_time_a,
                "deploy_gnn_s": sum(r["time_s"] for r in results_a),
                "deploy_interp_s": sum(r["time_s"] for r in results_b),
                "total_s": t_experiment_total,
            },
            "strategy_a_gnn_no_bn": {
                "description": "GINConv without BatchNorm, 3 features (h, coord, N/100)",
                "norm_type": "none",
                "training_mse": float(metrics_a["final_mse"]),
                "mse_last_10": [float(v) for v in mse_last_10],
                "training_converged": not metrics_a.get("stopped_early", False),
                "results": results_a,
                "mean_de_gap": float(np.mean(de_a)),
                "max_de_gap": float(np.max(de_a)),
                "std_de_gap": float(np.std(de_a)),
                "n_pass": pass_a,
                "n_total": len(h_test),
                "variational_principle_ok": var_ok_a,
            },
            "strategy_b_interpolation": {
                "description": "Direct scipy LinearNDInterpolator on (h, N) -> theta",
                "results": results_b,
                "mean_de_gap": float(np.mean(de_b)),
                "max_de_gap": float(np.max(de_b)),
                "std_de_gap": float(np.std(de_b)),
                "n_pass": pass_b,
                "n_total": len(h_test),
                "variational_principle_ok": var_ok_b,
            },
            "comparison": {
                "winner": "gnn_no_bn" if np.mean(de_a) < np.mean(de_b) else "interpolation",
                "gnn_improvement_over_bn": {
                    "vs_v2_3feat_bn": f"{(0.185 - np.mean(de_a)) / 0.185 * 100:.1f}% reduction",
                    "vs_v2_2feat_bn": f"{(0.095 - np.mean(de_a)) / 0.095 * 100:.1f}% reduction",
                },
                "v2_3feat_bn_mean": 0.185,
                "v2_2feat_bn_mean": 0.095,
            },
            "summary": {
                "generalization_success": pass_a == len(h_test) or pass_b == len(h_test),
                "best_strategy": "gnn_no_bn" if np.mean(de_a) < np.mean(de_b) else "interpolation",
                "best_mean_de_gap": float(min(np.mean(de_a), np.mean(de_b))),
                "verdict": "PASS" if max(pass_a, pass_b) == len(h_test) else "FAIL",
            },
        },
        out_path,
    )
    logger.info(f"\n  Results saved: {out_path}")

    best_pass = max(pass_a, pass_b)
    return 0 if best_pass == len(h_test) else 1


if __name__ == "__main__":
    sys.exit(main())
