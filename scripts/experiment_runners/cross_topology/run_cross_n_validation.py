#!/usr/bin/env python3
"""Within-topology cross-N validation for triangular and heavy_hex.

Validates that GNN cross-N generalization works on heterogeneous topologies
(not just chain_1d), establishing a baseline before cross-topology transfer.

For each topology (triangular, heavy_hex):
    - Train MPNNPredictor on N=6 + N=16 data
    - Predict theta_opt for N=10 at 5 test h-values
    - Success: ΔE/gap < 10% on at least 3 of 5 h-values

Seeds 42, 43, 44 for statistical robustness with mean/std reporting.

Usage:
    python scripts/experiment_runners/cross_topology/run_cross_n_validation.py \\
        --topology triangular heavy_hex \\
        --train-sizes 6 16 --target-n 10 \\
        --norm-type none --seeds 42,43,44 \\
        --output-dir results/scaling/cross_topology

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 8.1, 8.5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT / "scripts" / "experiment_runners") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts" / "experiment_runners"))

import numpy as np
import torch

from cross_topology.helpers import (
    SourceData,
    build_cross_topology_dataset,
    build_experiment_envelope,
    build_target_graph,
    canonicalize_theta,
    evaluate_theta,
    load_source_data_filtered,
    save_validation_checkpoint,
    validate_training_data,
)
from qmbp_simulation.predictors import MPNNPredictor, train_mpnn
from qmbp_simulation.utils.helpers import json_dump

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# h-value regimes per topology
# ═══════════════════════════════════════════════════════════════════════════════

# Valid regime h-values for N=10 targets (from project-status.md scaling law)
# h_min_safe = 1.5 + 0.020·N^1.31 ≈ 1.5 + 0.41 = 1.91 for N=10
# triangular: h >= 4.0 for p=1 (from project-status.md)
# heavy_hex: h >= 3.25 for N=10 (from project-status.md)
DEFAULT_TEST_H: dict[str, list[float]] = {
    "triangular": [6.0, 5.5, 5.0, 4.5, 4.0],
    "heavy_hex": [5.0, 4.5, 4.0, 3.5, 3.25],
}


def get_test_h_values(topology: str) -> list[float]:
    """Get appropriate test h-values for N=10 target prediction."""
    if topology in DEFAULT_TEST_H:
        return DEFAULT_TEST_H[topology]
    # Fallback: generic safe regime
    h_min = 1.5 + 0.020 * 10**1.31 + 1.0
    return np.linspace(h_min + 2.0, h_min, 5).tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# Source data discovery
# ═══════════════════════════════════════════════════════════════════════════════


def find_source_file(
    data_dir: Path,
    topology: str,
    n: int,
    p: int = 1,
) -> Path | None:
    """Find existing VQE result file for given topology/size.

    Searches for files matching the pattern:
        vqe_{topology}_N{n}_p{p}_*.json  (cross_topology format)
        scaling_{topology}_N{n}_*.json   (older scaling format)
    """
    # Try cross_topology format first
    patterns = [
        f"vqe_{topology}_N{n}_p{p}_*.json",
        f"scaling_{topology}_N{n}_*.json",
        f"*_{topology}_N{n}_*.json",
    ]
    for pattern in patterns:
        matches = sorted(data_dir.glob(pattern))
        if matches:
            return matches[-1]  # Most recent
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Single-seed training + evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def run_single_seed(
    topology: str,
    train_sources: list[SourceData],
    target_n: int,
    h_test: list[float],
    hidden_dim: int,
    n_epochs: int,
    lr: float,
    norm_type: str,
    seed: int,
    threshold: float = 0.10,
) -> dict:
    """Train GNN on source data and evaluate predictions at target N.

    Parameters
    ----------
    topology : str
        Topology for training and prediction (same for within-topology).
    train_sources : list[SourceData]
        Training data (N=6 + N=16 from the same topology).
    target_n : int
        Target system size for prediction (typically 10).
    h_test : list[float]
        h-values at which to evaluate predictions.
    hidden_dim : int
        Hidden dimension for MPNNPredictor.
    n_epochs : int
        Number of training epochs.
    lr : float
        Learning rate for Adam optimizer.
    norm_type : str
        Normalization type for GNN ("none", "batch", "layer").
    seed : int
        Random seed for training.
    threshold : float
        Pass threshold for ΔE/gap (default 0.10 = 10%).

    Returns
    -------
    dict
        Per-seed results including training metrics and per-h evaluations.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build training dataset from sources
    dataset = build_cross_topology_dataset(train_sources, use_n_feature=True)
    theta_dim = train_sources[0].param_dim

    # Create and train model
    model = MPNNPredictor(
        node_features=3,
        hidden_dim=hidden_dim,
        output_dim=theta_dim,
        n_layers=3,
        norm_type=norm_type,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"    Model: {n_params:,} params, norm_type={norm_type}")

    t_train_start = time.perf_counter()
    train_metrics = train_mpnn(model, dataset, n_epochs=n_epochs, lr=lr, seed=seed)
    t_train = time.perf_counter() - t_train_start

    logger.info(f"    Training: MSE={train_metrics['final_mse']:.2e}, time={t_train:.1f}s")

    # Evaluate on target N
    model.eval()
    eval_results: list[dict] = []

    for h_val in h_test:
        graph = build_target_graph(topology, target_n, h_val, use_n_feature=True)
        with torch.no_grad():
            theta_pred = model(graph).numpy().flatten()
        theta_pred = canonicalize_theta(theta_pred)

        result = evaluate_theta(
            theta_pred=theta_pred,
            n_target=target_n,
            h_val=h_val,
            topology=topology,
            threshold=threshold,
            seed=seed,
        )
        eval_results.append(result)

        status = "✅" if result["passed"] else "❌"
        var_flag = "" if result["variational_ok"] else " ⚠VAR"
        logger.info(f"      h={h_val:.3f}: ΔE/gap={result['de_gap'] * 100:.2f}% {status}{var_flag}")

    # Compute summary stats for this seed
    de_gaps = [r["de_gap"] for r in eval_results]
    n_pass = sum(1 for r in eval_results if r["passed"])
    n_variational_ok = sum(1 for r in eval_results if r["variational_ok"])

    if n_pass < 3:
        logger.warning(
            f"    ⚠️ Seed {seed} FAILED: only {n_pass}/{len(h_test)} passed "
            f"(need ≥3). Mean ΔE/gap={np.mean(de_gaps) * 100:.2f}%"
        )
    if n_variational_ok < len(h_test):
        logger.warning(
            f"    ⚠️ Variational principle violated for "
            f"{len(h_test) - n_variational_ok}/{len(h_test)} points (seed={seed})"
        )

    return {
        "seed": seed,
        "training": {
            "final_mse": float(train_metrics["final_mse"]),
            "training_time_s": t_train,
            "n_training_points": len(dataset),
            "n_model_params": n_params,
            "converged": not train_metrics.get("stopped_early", False),
        },
        "results": eval_results,
        "summary": {
            "mean_de_gap": float(np.mean(de_gaps)),
            "std_de_gap": float(np.std(de_gaps)),
            "max_de_gap": float(np.max(de_gaps)),
            "min_de_gap": float(np.min(de_gaps)),
            "n_pass": n_pass,
            "n_total": len(h_test),
            "pass_rate": n_pass / len(h_test),
            "n_variational_ok": n_variational_ok,
            "variational_ok_all": n_variational_ok == len(h_test),
            "passed": n_pass >= 3,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-seed topology validation
# ═══════════════════════════════════════════════════════════════════════════════


def run_topology_validation(
    topology: str,
    train_sources: list[SourceData],
    target_n: int,
    h_test: list[float],
    hidden_dim: int,
    n_epochs: int,
    lr: float,
    norm_type: str,
    seeds: list[int],
    threshold: float = 0.10,
) -> dict:
    """Run cross-N validation for a single topology across multiple seeds.

    Parameters
    ----------
    topology : str
        Topology to validate (e.g. "triangular", "heavy_hex").
    train_sources : list[SourceData]
        Training data for this topology (N=6 + N=16).
    target_n : int
        Target system size (N=10).
    h_test : list[float]
        Test h-values.
    hidden_dim, n_epochs, lr, norm_type :
        GNN training hyperparameters.
    seeds : list[int]
        Seeds for multi-seed robustness (42, 43, 44).
    threshold : float
        ΔE/gap pass threshold.

    Returns
    -------
    dict
        Aggregate results across all seeds with mean/std statistics.
    """
    logger.info(f"\n{'─' * 60}")
    logger.info(
        f"  Topology: {topology} | Train: N={[s.n for s in train_sources]} → Target: N={target_n}"
    )
    logger.info(f"  h_test: {[f'{h:.2f}' for h in h_test]}")
    logger.info(f"  Seeds: {seeds}")
    logger.info(f"{'─' * 60}")

    t_start = time.perf_counter()
    per_seed_results: list[dict] = []

    for seed in seeds:
        logger.info(f"\n  ── Seed {seed} ──")
        seed_result = run_single_seed(
            topology=topology,
            train_sources=train_sources,
            target_n=target_n,
            h_test=h_test,
            hidden_dim=hidden_dim,
            n_epochs=n_epochs,
            lr=lr,
            norm_type=norm_type,
            seed=seed,
            threshold=threshold,
        )
        per_seed_results.append(seed_result)

    elapsed = time.perf_counter() - t_start

    # Aggregate statistics across seeds (Requirement 8.5)
    mean_de_gaps = [r["summary"]["mean_de_gap"] for r in per_seed_results]
    max_de_gaps = [r["summary"]["max_de_gap"] for r in per_seed_results]
    pass_rates = [r["summary"]["pass_rate"] for r in per_seed_results]
    n_passes = [r["summary"]["n_pass"] for r in per_seed_results]
    var_ok_counts = [r["summary"]["n_variational_ok"] for r in per_seed_results]

    aggregate = {
        "mean_de_gap": float(np.mean(mean_de_gaps)),
        "std_de_gap": float(np.std(mean_de_gaps)),
        "max_de_gap": float(np.max(max_de_gaps)),
        "mean_pass_rate": float(np.mean(pass_rates)),
        "std_pass_rate": float(np.std(pass_rates)),
        "per_seed_mean_de_gap": [float(v) for v in mean_de_gaps],
        "per_seed_max_de_gap": [float(v) for v in max_de_gaps],
        "per_seed_n_pass": [int(v) for v in n_passes],
        "n_variational_ok_total": int(sum(var_ok_counts)),
        "n_variational_ok_expected": int(len(seeds) * len(h_test)),
        "all_seeds_pass": all(r["summary"]["passed"] for r in per_seed_results),
        "any_seed_pass": any(r["summary"]["passed"] for r in per_seed_results),
    }

    # Warnings for failed validations
    if not aggregate["all_seeds_pass"]:
        failed_seeds = [r["seed"] for r in per_seed_results if not r["summary"]["passed"]]
        logger.warning(
            f"  ⚠️ {topology} cross-N validation FAILED for seeds {failed_seeds}. "
            f"Mean ΔE/gap={aggregate['mean_de_gap'] * 100:.3f}%"
        )
    if aggregate["n_variational_ok_total"] < aggregate["n_variational_ok_expected"]:
        n_violations = aggregate["n_variational_ok_expected"] - aggregate["n_variational_ok_total"]
        logger.warning(f"  ⚠️ {n_violations} variational principle violations detected across seeds")

    # Log summary
    logger.info(f"\n  ── {topology} Summary ──")
    logger.info(
        f"    Mean ΔE/gap: {aggregate['mean_de_gap'] * 100:.3f}% "
        f"± {aggregate['std_de_gap'] * 100:.3f}%"
    )
    logger.info(
        f"    Pass rate: {aggregate['mean_pass_rate'] * 100:.1f}% "
        f"± {aggregate['std_pass_rate'] * 100:.1f}%"
    )
    logger.info(f"    Per-seed passes: {n_passes}")
    verdict = "✅ PASS" if aggregate["all_seeds_pass"] else "❌ FAIL"
    logger.info(f"    Verdict: {verdict}")

    return {
        "topology": topology,
        "target_n": target_n,
        "h_test": h_test,
        "norm_type": norm_type,
        "seeds": seeds,
        "per_seed": per_seed_results,
        "aggregate": aggregate,
        "timing_s": elapsed,
        "verdict": "PASS" if aggregate["all_seeds_pass"] else "FAIL",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Within-topology cross-N validation (triangular, heavy_hex)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--topology",
        type=str,
        nargs="+",
        default=["triangular", "heavy_hex"],
        help="Topologies to validate (runs both by default)",
    )
    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs="+",
        default=[6, 16],
        help="System sizes used for training",
    )
    parser.add_argument(
        "--target-n",
        type=int,
        default=10,
        help="Target system size for prediction",
    )
    parser.add_argument(
        "--norm-type",
        type=str,
        default="none",
        choices=["none", "batch", "layer"],
        help="Normalization type for MPNNPredictor",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated random seeds for multi-seed validation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/scaling/cross_topology",
        help="Output directory for result JSON files",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="results/scaling/cross_topology",
        help="Directory to search for source VQE data files",
    )
    parser.add_argument(
        "--source-files",
        type=str,
        nargs="*",
        default=None,
        help="Explicit source file paths (overrides data-dir search)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=6000,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for MPNNPredictor",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="ΔE/gap pass threshold (default 10%%)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # Log all parameters at INFO level (Requirement 8.4)
    logger.info("=" * 60)
    logger.info("Within-Topology Cross-N Validation")
    logger.info("=" * 60)
    logger.info(f"  Topologies: {args.topology}")
    logger.info(f"  Train sizes: {args.train_sizes}")
    logger.info(f"  Target N: {args.target_n}")
    logger.info(f"  Norm type: {args.norm_type}")
    logger.info(f"  Seeds: {seeds}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Hidden dim: {args.hidden_dim}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Threshold: {args.threshold * 100:.1f}%")
    logger.info(f"  Data dir: {data_dir}")
    logger.info(f"  Output dir: {output_dir}")
    if args.source_files:
        logger.info(f"  Source files (explicit): {args.source_files}")
    logger.info("=" * 60)

    t_experiment_start = time.perf_counter()
    topology_results: dict[str, dict] = {}
    source_files_used: list[str] = []

    for topology in args.topology:
        # Load training data for this topology
        train_sources: list[SourceData] = []

        if args.source_files:
            # Use explicitly provided source files
            topo_files = [f for f in args.source_files if topology in f]
            for fp in topo_files:
                path = Path(fp)
                src = load_source_data_filtered(path, seed=seeds[0])
                if src.n in args.train_sizes:
                    train_sources.append(src)
                    source_files_used.append(str(path))
        else:
            # Auto-discover source files from data_dir
            for n_size in args.train_sizes:
                src_path = find_source_file(data_dir, topology, n_size)
                if src_path is None:
                    logger.error(
                        f"No source file found for {topology} N={n_size} "
                        f"in {data_dir}. Run run_vqe_data_gen.py first."
                    )
                    return 1
                logger.info(f"  Found: {src_path.name} ({topology} N={n_size})")
                src = load_source_data_filtered(src_path, seed=seeds[0])
                train_sources.append(src)
                source_files_used.append(str(src_path))

        if len(train_sources) < 2:
            logger.error(
                f"Need at least 2 training sizes for {topology}, found {len(train_sources)}."
            )
            return 1

        # ── Validate training data before proceeding ──────────────────
        validation = validate_training_data(
            train_sources,
            min_points=6,
            min_sizes=2,
            experiment_label=f"cross_n_{topology}",
        )
        if not validation.passed:
            logger.error(
                f"Training data validation FAILED for {topology}. "
                f"Cannot proceed — results would be meaningless."
            )
            save_validation_checkpoint(validation, output_dir, f"cross_n_{topology}")
            return 1

        # Get test h-values for target
        h_test = get_test_h_values(topology)

        # Run multi-seed validation
        result = run_topology_validation(
            topology=topology,
            train_sources=train_sources,
            target_n=args.target_n,
            h_test=h_test,
            hidden_dim=args.hidden_dim,
            n_epochs=args.epochs,
            lr=args.lr,
            norm_type=args.norm_type,
            seeds=seeds,
            threshold=args.threshold,
        )
        topology_results[topology] = result

    t_total = time.perf_counter() - t_experiment_start

    # ── Comparison summary (Requirement 3.4) ─────────────────────────
    logger.info(f"\n{'═' * 60}")
    logger.info("  Cross-N Validation Comparison Summary")
    logger.info(f"{'═' * 60}")
    logger.info(
        f"  {'Topology':<15} {'Mean ΔE/gap':>12} {'Std':>8} {'Pass Rate':>10} {'Verdict':>8}"
    )
    logger.info(f"  {'─' * 55}")

    for topo, res in topology_results.items():
        agg = res["aggregate"]
        logger.info(
            f"  {topo:<15} {agg['mean_de_gap'] * 100:>10.3f}% "
            f"{agg['std_de_gap'] * 100:>6.3f}% "
            f"{agg['mean_pass_rate'] * 100:>8.1f}% "
            f"  {res['verdict']:>6}"
        )

    logger.info(f"\n  Total time: {t_total:.1f}s ({t_total / 60:.1f}m)")

    # ── Build and save result JSON (Requirement 3.5) ─────────────────
    envelope = build_experiment_envelope(
        experiment_name="cross_n_validation",
        source_files=source_files_used,
        seeds=seeds,
        total_time_s=t_total,
        train_sizes=args.train_sizes,
        target_n=args.target_n,
        norm_type=args.norm_type,
        hidden_dim=args.hidden_dim,
        n_epochs=args.epochs,
        lr=args.lr,
        threshold=args.threshold,
    )

    # Add per-topology strategy sections (zero_shot schema compatible)
    for topo, res in topology_results.items():
        key = f"strategy_{topo}"
        agg = res["aggregate"]
        envelope[key] = {
            "description": (
                f"Within-topology cross-N: train {topo} "
                f"N={args.train_sizes} → predict N={args.target_n}"
            ),
            "topology": topo,
            "norm_type": args.norm_type,
            "target_n": args.target_n,
            "h_test": res["h_test"],
            "seeds": seeds,
            "per_seed": res["per_seed"],
            "mean_de_gap": agg["mean_de_gap"],
            "std_de_gap": agg["std_de_gap"],
            "mean_pass_rate": agg["mean_pass_rate"],
            "std_pass_rate": agg["std_pass_rate"],
            "per_seed_mean_de_gap": agg["per_seed_mean_de_gap"],
            "per_seed_n_pass": agg["per_seed_n_pass"],
            "timing_s": res["timing_s"],
            "verdict": res["verdict"],
        }

    # Per-h breakdown table
    envelope["comparison"] = {}
    for topo, res in topology_results.items():
        per_h_summary: list[dict] = []
        h_test = res["h_test"]
        for h_idx, h_val in enumerate(h_test):
            h_de_gaps = []
            for seed_res in res["per_seed"]:
                h_de_gaps.append(seed_res["results"][h_idx]["de_gap"])
            per_h_summary.append(
                {
                    "h": h_val,
                    "mean_de_gap": float(np.mean(h_de_gaps)),
                    "std_de_gap": float(np.std(h_de_gaps)),
                    "pass_count": sum(1 for v in h_de_gaps if v < args.threshold),
                }
            )
        envelope["comparison"][topo] = {
            "per_h_results": per_h_summary,
            "overall_verdict": res["verdict"],
        }

    # Final summary
    all_pass = all(r["verdict"] == "PASS" for r in topology_results.values())
    envelope["summary"] = {
        "all_topologies_pass": all_pass,
        "verdict": "PASS" if all_pass else "FAIL",
        "topologies_tested": list(topology_results.keys()),
        "total_time_s": t_total,
    }

    # Save with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    topos_tag = "_".join(args.topology)
    out_path = output_dir / f"cross_n_validation_{topos_tag}_{timestamp}.json"
    json_dump(envelope, out_path)
    logger.info(f"\n  Results saved: {out_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
