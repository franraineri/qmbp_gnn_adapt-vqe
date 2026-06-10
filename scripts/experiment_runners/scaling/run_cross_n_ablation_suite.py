#!/usr/bin/env python3
"""Cross-N Ablation Suite — Comprehensive validation of zero-shot generalization.

Runs multiple ablation variants in a single script for rigorous claim validation:
  - Ablation E: norm_type=none WITHOUT N/100 feature (isolates BN vs N-feature)
  - Multi-seed: Train with seeds 42/43/44, report variance
  - Multi-target: Deploy to N=50,55,65,70 (interpolation range)
  - Extrapolation: Deploy to N=100 (beyond training range)

Each ablation produces a separate result file for independent analysis.

Usage:
    # Run all ablations
    python scripts/experiment_runners/scaling/run_cross_n_ablation_suite.py \
        --source-file results/scaling/scaling_N40_aer_mps_20260608_001053.json \
                      results/scaling/scaling_N80_aer_mps_20260607_211634.json

    # Run specific ablation only
    python scripts/.../run_cross_n_ablation_suite.py --ablation no-n-feature \
        --source-file results/scaling/scaling_N40_aer_mps_20260608_001053.json \
                      results/scaling/scaling_N80_aer_mps_20260607_211634.json

    # Multi-seed only
    python scripts/.../run_cross_n_ablation_suite.py --ablation multi-seed \
        --source-file ...
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
# Helpers (shared with v3)
# ═══════════════════════════════════════════════════════════════════════════════


def canonicalize_theta(theta: np.ndarray) -> np.ndarray:
    """Enforce theta_x > 0 sign convention."""
    if len(theta) == 0:
        return theta
    if theta[-1] < 0:
        return -theta
    return theta


def _load_source_data(source_path: Path, source_seed: int):
    """Load source file. Returns (n, topology, h_values, theta_opt, e_dmrg)."""
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    with open(source_path) as f:
        data = json.load(f)
    meta = data["metadata"]
    n = meta["n"]
    topology = meta["topology"]
    seed_runs = [r for r in data["vqe_results"] if r["seed"] == source_seed]
    if not seed_runs:
        available = [r["seed"] for r in data["vqe_results"]]
        raise ValueError(f"Seed {source_seed} not in {source_path}. Available: {available}")
    results = seed_runs[0]["results"]
    if "theta_opt" not in results[0]:
        raise ValueError(f"No theta_opt in {source_path}")
    h_values = np.array([r["h"] for r in results])
    theta_opt = np.array([canonicalize_theta(np.array(r["theta_opt"])) for r in results])
    e_dmrg = np.array([r["dmrg_energy"] for r in results])
    return n, topology, h_values, theta_opt, e_dmrg


def _build_dataset(sources, use_n_feature: bool) -> list[Data]:
    """Build combined training dataset."""
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


def _deploy_and_evaluate(
    model: nn.Module,
    n_target: int,
    h_test: list[float],
    topology: str,
    use_n_feature: bool,
    strategy: str,
    precision: float,
) -> list[dict]:
    """Deploy model to target N, evaluate all h-points."""
    builder = HamiltonianBuilder()
    hva = HVACircuitBuilder()
    solver = ClassicalSolver()
    backend = MPSBackend(strategy=strategy, chi_max=64, precision=precision, seed=42)

    # Pre-compute target graph structure
    lattice_ref = make_lattice(topology, n_target, J=1.0, h=h_test[0])
    edge_index_np, coord = builder.build_graph_data(lattice_ref)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    model.eval()
    results = []

    for h_val in h_test:
        t0 = time.time()
        cols = [np.full(n_target, h_val), coord.astype(float)]
        if use_n_feature:
            cols.append(np.full(n_target, n_target / 100.0))
        x = torch.tensor(np.stack(cols, axis=1), dtype=torch.float32)
        graph = Data(x=x, edge_index=edge_index, batch=torch.zeros(n_target, dtype=torch.long))

        with torch.no_grad():
            theta_pred = model(graph).numpy().flatten()
        theta_pred = canonicalize_theta(theta_pred)

        lattice_t = make_lattice(topology, n_target, J=1.0, h=h_val)
        H = builder.build(lattice_t)
        circuit, _ = hva.create(n_target, 1, lattice_t)
        e_pred = backend.evaluate(circuit, H, theta_pred)
        gt = solver.solve(H, lattice_t, method="dmrg")
        de_gap = abs(e_pred - gt.ground_energy) / max(gt.gap, 1e-10)
        elapsed = time.time() - t0

        results.append(
            {
                "h": float(h_val),
                "e_pred": float(e_pred),
                "e_dmrg": float(gt.ground_energy),
                "gap": float(gt.gap),
                "de_gap": float(de_gap),
                "theta_pred": theta_pred.tolist(),
                "passed": bool(de_gap < 0.05),
                "time_s": elapsed,
            }
        )

    return results


def _compute_h_test(n_target: int) -> list[float]:
    """Auto-compute h-test values in valid regime for target N."""
    h_min = 1.5 + 0.020 * n_target**1.31  # Corrected formula
    return np.linspace(h_min + 1.5, h_min + 0.5, 5).tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation E: norm_type=none WITHOUT N-feature
# ═══════════════════════════════════════════════════════════════════════════════


def run_ablation_no_n_feature(sources, args) -> dict:
    """Test norm_type=none WITHOUT N/100 feature (isolates BN contribution)."""
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION E: norm_type=none, NO N-feature (2 features: h, coord)")
    logger.info("=" * 60)

    topology = sources[0][1]
    theta_dim = sources[0][3].shape[1]

    dataset = _build_dataset(sources, use_n_feature=False)
    model = MPNNPredictor(
        node_features=2,
        hidden_dim=args.hidden_dim,
        n_layers=3,
        output_dim=theta_dim,
        norm_type="none",
    )

    t0 = time.time()
    metrics = train_mpnn(model, dataset, n_epochs=args.n_epochs, seed=42)
    train_time = time.time() - t0
    logger.info(f"  Training: MSE={metrics['final_mse']:.2e}, time={train_time:.1f}s")

    h_test = _compute_h_test(args.target_n)
    results = _deploy_and_evaluate(
        model,
        args.target_n,
        h_test,
        topology,
        use_n_feature=False,
        strategy=args.strategy,
        precision=args.precision,
    )

    n_pass = sum(1 for r in results if r["passed"])
    de_gaps = [r["de_gap"] for r in results]
    mean_de = float(np.mean(de_gaps))

    for r in results:
        s = "✅" if r["passed"] else "❌"
        logger.info(f"  h={r['h']:.3f}: ΔE/gap={r['de_gap'] * 100:.2f}% {s}")
    logger.info(f"  RESULT: mean={mean_de * 100:.2f}%, pass={n_pass}/{len(results)}")

    return {
        "ablation": "no_n_feature_no_bn",
        "norm_type": "none",
        "use_n_feature": False,
        "n_features": 2,
        "training_mse": float(metrics["final_mse"]),
        "training_time_s": train_time,
        "results": results,
        "mean_de_gap": mean_de,
        "n_pass": n_pass,
        "n_total": len(results),
        "verdict": "PASS" if n_pass == len(results) else "FAIL",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-seed validation
# ═══════════════════════════════════════════════════════════════════════════════


def run_multi_seed(sources, args) -> dict:
    """Train GNN with multiple seeds, report variance in predictions."""
    logger.info("\n" + "=" * 60)
    logger.info("MULTI-SEED: Training seeds 42, 43, 44 (norm_type=none, 3feat)")
    logger.info("=" * 60)

    topology = sources[0][1]
    theta_dim = sources[0][3].shape[1]
    seeds = [42, 43, 44]

    dataset = _build_dataset(sources, use_n_feature=True)
    h_test = _compute_h_test(args.target_n)

    all_seed_results = []
    all_de_gaps_per_seed = []

    for seed in seeds:
        logger.info(f"\n  ─── Seed {seed} ───")
        model = MPNNPredictor(
            node_features=3,
            hidden_dim=args.hidden_dim,
            n_layers=3,
            output_dim=theta_dim,
            norm_type="none",
        )
        metrics = train_mpnn(model, dataset, n_epochs=args.n_epochs, seed=seed)
        logger.info(f"  MSE={metrics['final_mse']:.2e}")

        results = _deploy_and_evaluate(
            model,
            args.target_n,
            h_test,
            topology,
            use_n_feature=True,
            strategy=args.strategy,
            precision=args.precision,
        )

        n_pass = sum(1 for r in results if r["passed"])
        de_gaps = [r["de_gap"] for r in results]
        mean_de = float(np.mean(de_gaps))
        all_de_gaps_per_seed.append(de_gaps)

        for r in results:
            s = "✅" if r["passed"] else "❌"
            logger.info(f"    h={r['h']:.3f}: ΔE/gap={r['de_gap'] * 100:.3f}% {s}")
        logger.info(f"  Seed {seed}: mean={mean_de * 100:.3f}%, pass={n_pass}/{len(results)}")

        all_seed_results.append(
            {
                "seed": seed,
                "training_mse": float(metrics["final_mse"]),
                "results": results,
                "mean_de_gap": mean_de,
                "n_pass": n_pass,
            }
        )

    # Cross-seed statistics
    all_means = [s["mean_de_gap"] for s in all_seed_results]
    all_passes = [s["n_pass"] for s in all_seed_results]

    # Per-h-point variance across seeds
    de_matrix = np.array(all_de_gaps_per_seed)  # [3 seeds × 5 h-points]
    per_h_std = de_matrix.std(axis=0).tolist()
    per_h_mean = de_matrix.mean(axis=0).tolist()

    logger.info("\n  ─── Multi-Seed Summary ───")
    logger.info(
        f"  Mean ΔE/gap across seeds: {np.mean(all_means) * 100:.3f}% ± {np.std(all_means) * 100:.3f}%"
    )
    logger.info(f"  Pass rates: {all_passes}")
    logger.info(f"  Per-h std: {[f'{s * 100:.3f}%' for s in per_h_std]}")
    logger.info(
        f"  Consistent: {'YES ✅' if all(p == len(h_test) for p in all_passes) else 'NO ⚠️'}"
    )

    return {
        "ablation": "multi_seed",
        "seeds": seeds,
        "norm_type": "none",
        "use_n_feature": True,
        "n_features": 3,
        "per_seed": all_seed_results,
        "cross_seed_mean_de_gap": float(np.mean(all_means)),
        "cross_seed_std_de_gap": float(np.std(all_means)),
        "per_h_point_std": per_h_std,
        "per_h_point_mean": per_h_mean,
        "all_seeds_pass": all(p == len(h_test) for p in all_passes),
        "verdict": "PASS" if all(p == len(h_test) for p in all_passes) else "PARTIAL",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-target N sweep
# ═══════════════════════════════════════════════════════════════════════════════


def run_multi_target(sources, args) -> dict:
    """Deploy to multiple target N values (interpolation + extrapolation)."""
    logger.info("\n" + "=" * 60)
    logger.info("MULTI-TARGET: Deploy to N=50,55,65,70,100 (norm_type=none, 3feat)")
    logger.info("=" * 60)

    topology = sources[0][1]
    theta_dim = sources[0][3].shape[1]
    target_ns = [50, 55, 65, 70, 100]

    dataset = _build_dataset(sources, use_n_feature=True)
    model = MPNNPredictor(
        node_features=3,
        hidden_dim=args.hidden_dim,
        n_layers=3,
        output_dim=theta_dim,
        norm_type="none",
    )
    metrics = train_mpnn(model, dataset, n_epochs=args.n_epochs, seed=42)
    logger.info(f"  Training: MSE={metrics['final_mse']:.2e}")

    per_target = []
    for n_target in target_ns:
        h_test = _compute_h_test(n_target)
        logger.info(f"\n  ─── Target N={n_target} (h=[{h_test[0]:.2f}..{h_test[-1]:.2f}]) ───")

        results = _deploy_and_evaluate(
            model,
            n_target,
            h_test,
            topology,
            use_n_feature=True,
            strategy=args.strategy,
            precision=args.precision,
        )

        n_pass = sum(1 for r in results if r["passed"])
        de_gaps = [r["de_gap"] for r in results]
        mean_de = float(np.mean(de_gaps))

        n_source_sizes = sorted(set(s[0] for s in sources))
        in_range = min(n_source_sizes) <= n_target <= max(n_source_sizes)
        mode = "interpolation" if in_range else "extrapolation"

        for r in results:
            s = "✅" if r["passed"] else "❌"
            logger.info(f"    h={r['h']:.3f}: ΔE/gap={r['de_gap'] * 100:.3f}% {s}")
        logger.info(
            f"  N={n_target} ({mode}): mean={mean_de * 100:.3f}%, pass={n_pass}/{len(results)}"
        )

        per_target.append(
            {
                "n_target": n_target,
                "mode": mode,
                "h_test": h_test,
                "results": results,
                "mean_de_gap": mean_de,
                "max_de_gap": float(np.max(de_gaps)),
                "n_pass": n_pass,
                "n_total": len(results),
            }
        )

    # Summary
    interp_targets = [t for t in per_target if t["mode"] == "interpolation"]
    extrap_targets = [t for t in per_target if t["mode"] == "extrapolation"]

    interp_all_pass = all(t["n_pass"] == t["n_total"] for t in interp_targets)
    extrap_pass_rate = sum(t["n_pass"] for t in extrap_targets) / max(
        sum(t["n_total"] for t in extrap_targets), 1
    )

    logger.info("\n  ─── Multi-Target Summary ───")
    logger.info(f"  Interpolation (N=50,55,65,70): all pass = {interp_all_pass}")
    logger.info(f"  Extrapolation (N=100): pass rate = {extrap_pass_rate:.0%}")

    return {
        "ablation": "multi_target",
        "norm_type": "none",
        "use_n_feature": True,
        "training_mse": float(metrics["final_mse"]),
        "per_target": per_target,
        "interpolation_all_pass": interp_all_pass,
        "extrapolation_pass_rate": float(extrap_pass_rate),
        "verdict": "PASS" if interp_all_pass else "FAIL",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


ABLATIONS = {
    "no-n-feature": run_ablation_no_n_feature,
    "multi-seed": run_multi_seed,
    "multi-target": run_multi_target,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-N Ablation Suite — validates zero-shot generalization claims",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-file",
        type=str,
        required=True,
        nargs="+",
        help="Phase 2 result JSON(s) with theta_opt",
    )
    parser.add_argument("--source-seed", type=int, default=42)
    parser.add_argument(
        "--target-n",
        type=int,
        default=60,
        help="Default target N (used by no-n-feature and multi-seed)",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=6000)
    parser.add_argument("--strategy", type=str, default="aer_mps")
    parser.add_argument("--precision", type=float, default=0.005)
    parser.add_argument("--output-dir", type=str, default="results/scaling/zero_shot")
    parser.add_argument(
        "--ablation",
        type=str,
        default="all",
        choices=["all", "no-n-feature", "multi-seed", "multi-target"],
        help="Which ablation to run (default: all)",
    )
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

    n_source_sizes = sorted(set(s[0] for s in sources))
    logger.info("=" * 60)
    logger.info("Cross-N Ablation Suite")
    logger.info(f"  Sources: N={n_source_sizes}")
    logger.info(f"  Ablation: {args.ablation}")
    logger.info("=" * 60)

    t_total = time.time()
    all_results = {}

    # Run selected ablations
    if args.ablation in ("all", "no-n-feature"):
        all_results["no_n_feature"] = run_ablation_no_n_feature(sources, args)

    if args.ablation in ("all", "multi-seed"):
        all_results["multi_seed"] = run_multi_seed(sources, args)

    if args.ablation in ("all", "multi-target"):
        all_results["multi_target"] = run_multi_target(sources, args)

    t_total = time.time() - t_total

    # Save combined results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    n_tag = "_".join(str(n) for n in n_source_sizes)
    abl_tag = args.ablation.replace("-", "_")
    out_path = output_dir / f"ablation_{abl_tag}_N{n_tag}_{timestamp}.json"

    envelope = {
        "experiment": "cross_n_ablation_suite",
        "version": "1.0",
        "metadata": {
            "n_source_sizes": n_source_sizes,
            "target_n_default": args.target_n,
            "topology": sources[0][1],
            "source_seed": args.source_seed,
            "source_files": [str(p) for p in source_paths],
            "hidden_dim": args.hidden_dim,
            "n_epochs": args.n_epochs,
            "strategy": args.strategy,
            "ablation_mode": args.ablation,
        },
        "environment": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
        },
        "timing": {"total_s": t_total},
        "results": all_results,
    }

    # Overall verdict
    verdicts = [r.get("verdict", "UNKNOWN") for r in all_results.values()]
    envelope["summary"] = {
        "ablations_run": list(all_results.keys()),
        "verdicts": {k: v.get("verdict") for k, v in all_results.items()},
        "all_pass": all(v == "PASS" for v in verdicts),
    }

    json_dump(envelope, out_path)
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Results saved: {out_path}")
    logger.info(f"Total time: {t_total:.1f}s ({t_total / 60:.1f}m)")
    logger.info(f"Verdicts: {envelope['summary']['verdicts']}")
    logger.info("=" * 60)

    return 0 if envelope["summary"]["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
