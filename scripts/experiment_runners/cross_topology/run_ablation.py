#!/usr/bin/env python3
"""Ablation study: GNN vs MLP vs Scipy + BatchNorm comparison.

Compares three predictor architectures (GNN, MLP, Scipy interpolation) on
identical train/test splits, and ablates norm_type ("batch", "layer", "none")
on both within-topology and cross-topology experiments.

Key outputs:
- Per-predictor metrics: mean ΔE/gap, max ΔE/gap, pass rate, Spearman ρ
- Graph-structure-essential flag (GNN > 2× better than MLP)
- Norm ablation table: rows=norm_type, columns=experiment
- Training convergence: final MSE, epochs to convergence per norm_type

Usage:
    python scripts/experiment_runners/cross_topology/run_ablation.py \\
        --topologies triangular heavy_hex \\
        --target-n 10 \\
        --seeds 42,43,44 \\
        --output-dir results/scaling/cross_topology

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3, 6.4, 6.5
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
from scipy.stats import spearmanr

from cross_topology.helpers import (
    MLPBaseline,
    SourceData,
    build_cross_topology_dataset,
    build_experiment_envelope,
    build_target_graph,
    canonicalize_theta,
    evaluate_theta,
    extract_mlp_features,
    load_source_data_filtered,
    save_validation_checkpoint,
    scipy_interpolation_predict,
    train_mlp_baseline,
    validate_training_data,
)
from qmbp_simulation.predictors import MPNNPredictor, train_mpnn
from qmbp_simulation.utils.helpers import json_dump

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# h-value regimes for target prediction
# ═══════════════════════════════════════════════════════════════════════════════

TARGET_H_VALUES: dict[str, list[float]] = {
    "triangular": [5.5, 5.0, 4.5, 4.25, 4.0],
    "heavy_hex": [5.0, 4.5, 4.0, 3.5, 3.25],
}


def get_target_h_values(topology: str, n: int) -> list[float]:
    """Get appropriate h-values for target prediction."""
    if topology in TARGET_H_VALUES:
        return TARGET_H_VALUES[topology]
    h_min = 1.5 + 0.020 * n**1.31 + 0.5
    h_max = h_min + 2.0
    return np.linspace(h_max, h_min, 5).tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# Source file discovery (shared pattern with other runners)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_SOURCE_DIRS = [
    Path("results/scaling/cross_topology"),
    Path("results/thesis/p1_variants_N10"),
    Path("results/thesis/verification_r1"),
]


def _discover_source_files(topology: str, cross_topo_dir: Path) -> list[Path]:
    """Auto-discover source VQE data files for a given topology."""
    found: list[Path] = []

    if cross_topo_dir.exists():
        for f in sorted(cross_topo_dir.glob(f"vqe_{topology}_*.json")):
            found.append(f)

    thesis_dirs = [
        Path("results/thesis/p1_variants_N10"),
        Path("results/thesis/p1_variants_N16_r2"),
        Path("results/thesis/verification_r1"),
    ]
    for td in thesis_dirs:
        if td.exists():
            for subdir in sorted(td.iterdir()):
                if not subdir.is_dir():
                    continue
                if topology not in subdir.name:
                    continue
                for f in sorted(subdir.glob("pipeline_run_*.json")):
                    found.append(f)

    return found


def _get_target_topology(source_topo: str, available_topos: list[str]) -> str | None:
    """Get the target topology (the other one in the pair)."""
    for t in available_topos:
        if t != source_topo:
            return t
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Spearman correlation computation
# ═══════════════════════════════════════════════════════════════════════════════


def compute_spearman_correlation(
    predicted_thetas: list[np.ndarray],
    actual_thetas: list[np.ndarray],
) -> float:
    """Compute Spearman rank correlation between predicted and actual theta.

    Flattens all theta vectors and computes a single Spearman ρ.
    Returns NaN if fewer than 3 data points.

    Parameters
    ----------
    predicted_thetas : list[np.ndarray]
        List of predicted theta vectors.
    actual_thetas : list[np.ndarray]
        List of ground-truth theta vectors (same order as predicted).

    Returns
    -------
    float
        Spearman rank correlation coefficient (ρ).
    """
    pred_flat = np.concatenate(predicted_thetas)
    actual_flat = np.concatenate(actual_thetas)

    if len(pred_flat) < 3:
        return float("nan")

    rho, _ = spearmanr(pred_flat, actual_flat)
    return float(rho)


# ═══════════════════════════════════════════════════════════════════════════════
# Predictor evaluation functions
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_gnn_predictor(
    sources: list[SourceData],
    target_topology: str,
    target_n: int,
    h_test: list[float],
    hidden_dim: int,
    n_epochs: int,
    norm_type: str,
    seed: int,
    threshold: float,
) -> dict:
    """Train and evaluate GNN (MPNNPredictor) on the given data.

    Returns dict with predictions, metrics, training info, and predicted thetas.
    """
    output_dim = sources[0].param_dim
    dataset = build_cross_topology_dataset(sources, use_n_feature=True)

    model = MPNNPredictor(
        node_features=3,
        hidden_dim=hidden_dim,
        n_layers=3,
        output_dim=output_dim,
        norm_type=norm_type,
    )

    torch.manual_seed(seed)
    t_train_start = time.perf_counter()
    train_metrics = train_mpnn(model, dataset, n_epochs=n_epochs, seed=seed)
    t_train = time.perf_counter() - t_train_start

    model.eval()
    predictions = []
    predicted_thetas: list[np.ndarray] = []

    for h_val in h_test:
        target_graph = build_target_graph(target_topology, target_n, h_val, use_n_feature=True)
        with torch.no_grad():
            theta_pred = model(target_graph).numpy().flatten()
        theta_pred = canonicalize_theta(theta_pred)
        predicted_thetas.append(theta_pred)

        result = evaluate_theta(
            theta_pred=theta_pred,
            n_target=target_n,
            h_val=h_val,
            topology=target_topology,
            threshold=threshold,
            seed=seed,
        )
        predictions.append(result)

    de_gaps = [r["de_gap"] for r in predictions]
    n_pass = sum(1 for r in predictions if r["passed"])

    # Compute epochs_to_convergence from MSE history (first epoch where loss < 1% of initial)
    mse_history = train_metrics.get("mse_history", [])
    epochs_to_convergence = n_epochs
    if mse_history and mse_history[0] > 0:
        initial_loss = mse_history[0]
        for epoch_idx, loss_val in enumerate(mse_history):
            if loss_val < 0.01 * initial_loss:
                epochs_to_convergence = epoch_idx
                break

    return {
        "predictor": "GNN",
        "norm_type": norm_type,
        "predictions": predictions,
        "predicted_thetas": predicted_thetas,
        "metrics": {
            "mean_de_gap": float(np.mean(de_gaps)),
            "max_de_gap": float(np.max(de_gaps)),
            "pass_rate": n_pass / len(h_test),
            "n_pass": n_pass,
            "n_total": len(h_test),
        },
        "training": {
            "final_mse": train_metrics["final_mse"],
            "epochs_to_convergence": epochs_to_convergence,
            "mse_history_len": len(mse_history),
            "time_s": t_train,
        },
    }


def evaluate_mlp_predictor(
    sources: list[SourceData],
    target_topology: str,
    target_n: int,
    h_test: list[float],
    hidden_dim: int,
    n_epochs: int,
    seed: int,
    threshold: float,
) -> dict:
    """Train and evaluate MLP baseline on the given data.

    MLP receives flattened features (mean_h, mean_coord, N/100) — no graph.
    """
    output_dim = sources[0].param_dim
    features, targets = extract_mlp_features(sources)

    model = MLPBaseline(hidden_dim=hidden_dim, output_dim=output_dim)
    train_result = train_mlp_baseline(
        model,
        features,
        targets,
        n_epochs=n_epochs,
        seed=seed,
    )

    # Build target features for prediction
    from qmbp_simulation import HamiltonianBuilder, make_lattice

    builder = HamiltonianBuilder()
    lattice = make_lattice(target_topology, target_n, J=1.0, h=float(h_test[0]))
    _, coord = builder.build_graph_data(lattice)
    mean_coord = float(coord.mean())

    model.eval()
    predictions = []
    predicted_thetas: list[np.ndarray] = []

    for h_val in h_test:
        target_features = torch.tensor(
            [[float(h_val), mean_coord, target_n / 100.0]], dtype=torch.float32
        )
        with torch.no_grad():
            theta_pred = model(target_features).numpy().flatten()
        theta_pred = canonicalize_theta(theta_pred)
        predicted_thetas.append(theta_pred)

        result = evaluate_theta(
            theta_pred=theta_pred,
            n_target=target_n,
            h_val=h_val,
            topology=target_topology,
            threshold=threshold,
            seed=seed,
        )
        predictions.append(result)

    de_gaps = [r["de_gap"] for r in predictions]
    n_pass = sum(1 for r in predictions if r["passed"])

    return {
        "predictor": "MLP",
        "predictions": predictions,
        "predicted_thetas": predicted_thetas,
        "metrics": {
            "mean_de_gap": float(np.mean(de_gaps)),
            "max_de_gap": float(np.max(de_gaps)),
            "pass_rate": n_pass / len(h_test),
            "n_pass": n_pass,
            "n_total": len(h_test),
        },
        "training": {
            "final_mse": train_result["final_mse"],
            "epochs_to_convergence": train_result["epochs_to_convergence"],
            "time_s": None,  # Not separately timed
        },
    }


def evaluate_scipy_predictor(
    sources: list[SourceData],
    target_topology: str,
    target_n: int,
    h_test: list[float],
    seed: int,
    threshold: float,
) -> dict:
    """Evaluate scipy interpolation baseline on the given data.

    Uses LinearNDInterpolator on (h, N) → θ, no graph awareness.
    """
    predictions = []
    predicted_thetas: list[np.ndarray] = []

    for h_val in h_test:
        theta_pred = scipy_interpolation_predict(sources, target_h=h_val, target_n=target_n)
        theta_pred = canonicalize_theta(theta_pred)
        predicted_thetas.append(theta_pred)

        result = evaluate_theta(
            theta_pred=theta_pred,
            n_target=target_n,
            h_val=h_val,
            topology=target_topology,
            threshold=threshold,
            seed=seed,
        )
        predictions.append(result)

    de_gaps = [r["de_gap"] for r in predictions]
    n_pass = sum(1 for r in predictions if r["passed"])

    return {
        "predictor": "Scipy",
        "predictions": predictions,
        "predicted_thetas": predicted_thetas,
        "metrics": {
            "mean_de_gap": float(np.mean(de_gaps)),
            "max_de_gap": float(np.max(de_gaps)),
            "pass_rate": n_pass / len(h_test),
            "n_pass": n_pass,
            "n_total": len(h_test),
        },
        "training": {
            "final_mse": None,
            "epochs_to_convergence": None,
            "time_s": None,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Predictor comparison (GNN vs MLP vs Scipy)
# ═══════════════════════════════════════════════════════════════════════════════


def run_predictor_comparison(
    sources: list[SourceData],
    target_topology: str,
    target_n: int,
    h_test: list[float],
    actual_thetas: list[np.ndarray] | None,
    hidden_dim: int,
    n_epochs: int,
    seed: int,
    threshold: float,
    experiment_label: str,
) -> dict:
    """Run all three predictors on identical data and compare.

    Parameters
    ----------
    sources : list[SourceData]
        Training data (same for all predictors).
    target_topology : str
        Topology for target predictions.
    target_n : int
        Target system size.
    h_test : list[float]
        Test h-values.
    actual_thetas : list[np.ndarray] | None
        Ground-truth theta values for Spearman correlation (None if unavailable).
    hidden_dim, n_epochs : int
        Shared hyperparameters for GNN and MLP.
    seed : int
        Random seed.
    threshold : float
        ΔE/gap pass threshold.
    experiment_label : str
        Label for this comparison (e.g. "within_tri", "tri_to_hex").

    Returns
    -------
    dict
        Comparison results with per-predictor metrics, Spearman ρ,
        and graph_structure_essential flag.
    """
    logger.info(f"\n  ── Predictor Comparison: {experiment_label} (seed={seed}) ──")

    # GNN (default norm_type="none")
    logger.info("    Training GNN...")
    gnn_result = evaluate_gnn_predictor(
        sources=sources,
        target_topology=target_topology,
        target_n=target_n,
        h_test=h_test,
        hidden_dim=hidden_dim,
        n_epochs=n_epochs,
        norm_type="none",
        seed=seed,
        threshold=threshold,
    )
    logger.info(
        f"    GNN: mean ΔE/gap={gnn_result['metrics']['mean_de_gap'] * 100:.3f}%, "
        f"pass={gnn_result['metrics']['n_pass']}/{gnn_result['metrics']['n_total']}"
    )

    # MLP
    logger.info("    Training MLP...")
    mlp_result = evaluate_mlp_predictor(
        sources=sources,
        target_topology=target_topology,
        target_n=target_n,
        h_test=h_test,
        hidden_dim=hidden_dim,
        n_epochs=n_epochs,
        seed=seed,
        threshold=threshold,
    )
    logger.info(
        f"    MLP: mean ΔE/gap={mlp_result['metrics']['mean_de_gap'] * 100:.3f}%, "
        f"pass={mlp_result['metrics']['n_pass']}/{mlp_result['metrics']['n_total']}"
    )

    # Scipy
    logger.info("    Evaluating Scipy...")
    scipy_result = evaluate_scipy_predictor(
        sources=sources,
        target_topology=target_topology,
        target_n=target_n,
        h_test=h_test,
        seed=seed,
        threshold=threshold,
    )
    logger.info(
        f"    Scipy: mean ΔE/gap={scipy_result['metrics']['mean_de_gap'] * 100:.3f}%, "
        f"pass={scipy_result['metrics']['n_pass']}/{scipy_result['metrics']['n_total']}"
    )

    # Compute Spearman correlations (Requirement 5.4)
    spearman_results = {}
    for label, res in [("GNN", gnn_result), ("MLP", mlp_result), ("Scipy", scipy_result)]:
        if actual_thetas is not None and len(actual_thetas) >= 3:
            rho = compute_spearman_correlation(res["predicted_thetas"], actual_thetas)
        else:
            # Use predicted vs actual from evaluate_theta as proxy
            # (Spearman on theta parameter ordering across h-values)
            pred_thetas = res["predicted_thetas"]
            rho = float("nan")
            if len(pred_thetas) >= 3:
                # Compare rank ordering: do predicted theta magnitudes track actual?
                pred_norms = np.array([np.linalg.norm(t) for t in pred_thetas])
                actual_energies = np.array([r["e_exact"] for r in res["predictions"]])
                if len(pred_norms) >= 3:
                    rho_val, _ = spearmanr(pred_norms, actual_energies)
                    rho = float(rho_val)
        spearman_results[label] = rho

    # Graph-structure-essential flag (Requirement 5.5)
    gnn_mean = gnn_result["metrics"]["mean_de_gap"]
    mlp_mean = mlp_result["metrics"]["mean_de_gap"]
    graph_structure_essential = bool(mlp_mean > 2.0 * gnn_mean)

    # Warnings for concerning results
    if gnn_result["metrics"]["n_pass"] == 0:
        logger.warning(
            f"    ⚠️ GNN achieved 0/{gnn_result['metrics']['n_total']} pass — "
            f"check training data quality for {experiment_label}"
        )
    if gnn_mean > 0.5:
        logger.warning(
            f"    ⚠️ GNN mean ΔE/gap={gnn_mean * 100:.1f}% is very high — "
            f"GNN may not generalize for this experiment"
        )

    logger.info(
        f"    Spearman ρ: GNN={spearman_results['GNN']:.3f}, "
        f"MLP={spearman_results['MLP']:.3f}, "
        f"Scipy={spearman_results['Scipy']:.3f}"
    )
    flag_str = "✅ YES" if graph_structure_essential else "❌ NO"
    logger.info(
        f"    Graph essential: {flag_str} (MLP/GNN ratio = {mlp_mean / max(gnn_mean, 1e-10):.2f}×)"
    )

    return {
        "experiment_label": experiment_label,
        "seed": seed,
        "target_topology": target_topology,
        "target_n": target_n,
        "predictors": {
            "GNN": {
                "metrics": gnn_result["metrics"],
                "training": gnn_result["training"],
                "spearman_rho": spearman_results["GNN"],
            },
            "MLP": {
                "metrics": mlp_result["metrics"],
                "training": mlp_result["training"],
                "spearman_rho": spearman_results["MLP"],
            },
            "Scipy": {
                "metrics": scipy_result["metrics"],
                "training": scipy_result["training"],
                "spearman_rho": spearman_results["Scipy"],
            },
        },
        "graph_structure_essential": graph_structure_essential,
        "mlp_gnn_ratio": float(mlp_mean / max(gnn_mean, 1e-10)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Norm ablation (batch vs layer vs none)
# ═══════════════════════════════════════════════════════════════════════════════

NORM_TYPES = ["batch", "layer", "none"]


def run_norm_ablation(
    sources: list[SourceData],
    target_topology: str,
    target_n: int,
    h_test: list[float],
    hidden_dim: int,
    n_epochs: int,
    seed: int,
    threshold: float,
    experiment_label: str,
) -> dict:
    """Compare norm_type variants on a single experiment configuration.

    Trains GNN with each norm_type and reports metrics + convergence.

    Parameters
    ----------
    sources : list[SourceData]
        Training data.
    target_topology : str
        Target topology for prediction.
    target_n : int
        Target system size.
    h_test : list[float]
        Test h-values.
    hidden_dim, n_epochs : int
        GNN hyperparameters.
    seed : int
        Random seed.
    threshold : float
        ΔE/gap pass threshold.
    experiment_label : str
        Label (e.g. "within_tri", "tri_to_hex").

    Returns
    -------
    dict
        Per-norm_type metrics and convergence info.
    """
    logger.info(f"\n  ── Norm Ablation: {experiment_label} (seed={seed}) ──")
    norm_results: dict[str, dict] = {}

    for norm_type in NORM_TYPES:
        logger.info(f"    norm_type={norm_type}...")
        gnn_result = evaluate_gnn_predictor(
            sources=sources,
            target_topology=target_topology,
            target_n=target_n,
            h_test=h_test,
            hidden_dim=hidden_dim,
            n_epochs=n_epochs,
            norm_type=norm_type,
            seed=seed,
            threshold=threshold,
        )
        norm_results[norm_type] = {
            "metrics": gnn_result["metrics"],
            "training": gnn_result["training"],
        }
        logger.info(
            f"      mean ΔE/gap={gnn_result['metrics']['mean_de_gap'] * 100:.3f}%, "
            f"MSE={gnn_result['training']['final_mse']:.2e}, "
            f"conv_epoch={gnn_result['training']['epochs_to_convergence']}"
        )

    # Determine best norm_type
    best_norm = min(norm_results, key=lambda k: norm_results[k]["metrics"]["mean_de_gap"])

    return {
        "experiment_label": experiment_label,
        "seed": seed,
        "norm_results": norm_results,
        "best_norm_type": best_norm,
        "best_mean_de_gap": norm_results[best_norm]["metrics"]["mean_de_gap"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-seed aggregation
# ═══════════════════════════════════════════════════════════════════════════════


def aggregate_predictor_comparisons(seed_results: list[dict]) -> dict:
    """Aggregate predictor comparison results across seeds."""
    if not seed_results:
        return {"error": "No results"}

    aggregated: dict[str, dict] = {}
    for predictor in ["GNN", "MLP", "Scipy"]:
        mean_de_gaps = [r["predictors"][predictor]["metrics"]["mean_de_gap"] for r in seed_results]
        pass_rates = [r["predictors"][predictor]["metrics"]["pass_rate"] for r in seed_results]
        spearman_rhos = [
            r["predictors"][predictor]["spearman_rho"]
            for r in seed_results
            if not np.isnan(r["predictors"][predictor]["spearman_rho"])
        ]
        aggregated[predictor] = {
            "mean_de_gap": {
                "mean": float(np.mean(mean_de_gaps)),
                "std": float(np.std(mean_de_gaps)),
            },
            "pass_rate": {"mean": float(np.mean(pass_rates)), "std": float(np.std(pass_rates))},
            "spearman_rho": {
                "mean": float(np.mean(spearman_rhos)) if spearman_rhos else float("nan"),
                "std": float(np.std(spearman_rhos)) if len(spearman_rhos) > 1 else 0.0,
            },
        }

    # Aggregate graph_structure_essential flag
    essential_flags = [r["graph_structure_essential"] for r in seed_results]
    ratios = [r["mlp_gnn_ratio"] for r in seed_results]

    return {
        "predictors": aggregated,
        "graph_structure_essential": {
            "all_seeds": all(essential_flags),
            "any_seed": any(essential_flags),
            "per_seed": essential_flags,
            "mean_ratio": float(np.mean(ratios)),
        },
        "per_seed": seed_results,
    }


def aggregate_norm_ablations(seed_results: list[dict]) -> dict:
    """Aggregate norm ablation results across seeds."""
    if not seed_results:
        return {"error": "No results"}

    aggregated: dict[str, dict] = {}
    for norm_type in NORM_TYPES:
        mean_de_gaps = [
            r["norm_results"][norm_type]["metrics"]["mean_de_gap"] for r in seed_results
        ]
        final_mses = [r["norm_results"][norm_type]["training"]["final_mse"] for r in seed_results]
        conv_epochs = [
            r["norm_results"][norm_type]["training"]["epochs_to_convergence"] for r in seed_results
        ]
        aggregated[norm_type] = {
            "mean_de_gap": {
                "mean": float(np.mean(mean_de_gaps)),
                "std": float(np.std(mean_de_gaps)),
            },
            "final_mse": {"mean": float(np.mean(final_mses)), "std": float(np.std(final_mses))},
            "epochs_to_convergence": {
                "mean": float(np.mean(conv_epochs)),
                "std": float(np.std(conv_epochs)),
            },
        }

    best_norms = [r["best_norm_type"] for r in seed_results]
    # Majority vote for best norm
    from collections import Counter

    best_norm_majority = Counter(best_norms).most_common(1)[0][0]

    return {
        "norm_types": aggregated,
        "best_norm_type": best_norm_majority,
        "per_seed_best": best_norms,
        "per_seed": seed_results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ablation: GNN vs MLP vs Scipy + norm_type comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--topologies",
        type=str,
        nargs="+",
        default=["triangular", "heavy_hex"],
        help="Topologies to include in ablation experiments",
    )
    parser.add_argument(
        "--target-n",
        type=int,
        default=10,
        help="Target system size for prediction",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated random seeds",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/scaling/cross_topology",
        help="Output directory for result JSON",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=6000,
        help="Number of training epochs for GNN and MLP",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for GNN and MLP",
    )
    parser.add_argument(
        "--source-files-tri",
        type=str,
        nargs="*",
        default=None,
        help="Explicit source files for triangular topology",
    )
    parser.add_argument(
        "--source-files-hex",
        type=str,
        nargs="*",
        default=None,
        help="Explicit source files for heavy_hex topology",
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

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log experiment parameters (Requirement 8.4)
    logger.info("=" * 60)
    logger.info("Ablation Study: GNN vs MLP vs Scipy + Norm Comparison")
    logger.info("=" * 60)
    logger.info(f"  Topologies: {args.topologies}")
    logger.info(f"  Target N: {args.target_n}")
    logger.info(f"  Seeds: {seeds}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Hidden dim: {args.hidden_dim}")
    logger.info(f"  Threshold: {args.threshold * 100:.0f}%")
    logger.info(f"  Output dir: {output_dir}")
    logger.info("=" * 60)

    t_experiment_start = time.perf_counter()

    # Resolve source files
    source_files_map: dict[str, list[Path]] = {}
    if args.source_files_tri:
        source_files_map["triangular"] = [Path(f) for f in args.source_files_tri]
    if args.source_files_hex:
        source_files_map["heavy_hex"] = [Path(f) for f in args.source_files_hex]

    for topo in args.topologies:
        if topo not in source_files_map:
            discovered = _discover_source_files(topo, output_dir)
            if discovered:
                source_files_map[topo] = discovered
                logger.info(f"  Auto-discovered {len(discovered)} files for {topo}")
            else:
                logger.error(
                    f"  No source files found for {topo}. "
                    f"Use --source-files-tri or --source-files-hex."
                )
                return 1

    # Load source data per topology
    sources_map: dict[str, list[SourceData]] = {}
    for topo, files in source_files_map.items():
        sources_map[topo] = []
        for sf in files:
            try:
                src = load_source_data_filtered(sf, seed=seeds[0])
                sources_map[topo].append(src)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"  Skip {sf}: {e}")

    # ── Validate training data before proceeding ──────────────────────
    for topo, sources in sources_map.items():
        validation = validate_training_data(
            sources,
            min_points=4,
            min_sizes=1,
            experiment_label=f"ablation_{topo}",
        )
        if not validation.passed:
            logger.error(f"Training data validation FAILED for {topo}. Cannot run ablation.")
            save_validation_checkpoint(validation, output_dir, f"ablation_{topo}")
            return 1

    # ══════════════════════════════════════════════════════════════════════
    # Part 1: Predictor comparison (Requirement 5.1–5.5)
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'═' * 60}")
    logger.info("  PART 1: Predictor Comparison (GNN vs MLP vs Scipy)")
    logger.info(f"{'═' * 60}")

    predictor_comparisons: dict[str, dict] = {}

    # Within-topology experiments
    for topo in args.topologies:
        if topo not in sources_map or not sources_map[topo]:
            continue
        exp_label = f"within_{topo[:3]}"
        h_test = get_target_h_values(topo, args.target_n)
        seed_results = []

        for seed in seeds:
            result = run_predictor_comparison(
                sources=sources_map[topo],
                target_topology=topo,
                target_n=args.target_n,
                h_test=h_test,
                actual_thetas=None,
                hidden_dim=args.hidden_dim,
                n_epochs=args.epochs,
                seed=seed,
                threshold=args.threshold,
                experiment_label=exp_label,
            )
            seed_results.append(result)

        predictor_comparisons[exp_label] = aggregate_predictor_comparisons(seed_results)

    # Cross-topology experiments
    for source_topo in args.topologies:
        target_topo = _get_target_topology(source_topo, args.topologies)
        if target_topo is None:
            continue
        if source_topo not in sources_map or not sources_map[source_topo]:
            continue

        exp_label = f"{source_topo[:3]}_to_{target_topo[:3]}"
        h_test = get_target_h_values(target_topo, args.target_n)
        seed_results = []

        for seed in seeds:
            result = run_predictor_comparison(
                sources=sources_map[source_topo],
                target_topology=target_topo,
                target_n=args.target_n,
                h_test=h_test,
                actual_thetas=None,
                hidden_dim=args.hidden_dim,
                n_epochs=args.epochs,
                seed=seed,
                threshold=args.threshold,
                experiment_label=exp_label,
            )
            seed_results.append(result)

        predictor_comparisons[exp_label] = aggregate_predictor_comparisons(seed_results)

    # ══════════════════════════════════════════════════════════════════════
    # Part 2: Norm ablation (Requirement 6.1–6.5)
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\n{'═' * 60}")
    logger.info("  PART 2: Norm Ablation (batch vs layer vs none)")
    logger.info(f"{'═' * 60}")

    norm_ablations: dict[str, dict] = {}

    # Within-topology norm ablation
    for topo in args.topologies:
        if topo not in sources_map or not sources_map[topo]:
            continue
        exp_label = f"within_{topo[:3]}"
        h_test = get_target_h_values(topo, args.target_n)
        seed_results = []

        for seed in seeds:
            result = run_norm_ablation(
                sources=sources_map[topo],
                target_topology=topo,
                target_n=args.target_n,
                h_test=h_test,
                hidden_dim=args.hidden_dim,
                n_epochs=args.epochs,
                seed=seed,
                threshold=args.threshold,
                experiment_label=exp_label,
            )
            seed_results.append(result)

        norm_ablations[exp_label] = aggregate_norm_ablations(seed_results)

    # Cross-topology norm ablation
    for source_topo in args.topologies:
        target_topo = _get_target_topology(source_topo, args.topologies)
        if target_topo is None:
            continue
        if source_topo not in sources_map or not sources_map[source_topo]:
            continue

        exp_label = f"{source_topo[:3]}_to_{target_topo[:3]}"
        h_test = get_target_h_values(target_topo, args.target_n)
        seed_results = []

        for seed in seeds:
            result = run_norm_ablation(
                sources=sources_map[source_topo],
                target_topology=target_topo,
                target_n=args.target_n,
                h_test=h_test,
                hidden_dim=args.hidden_dim,
                n_epochs=args.epochs,
                seed=seed,
                threshold=args.threshold,
                experiment_label=exp_label,
            )
            seed_results.append(result)

        norm_ablations[exp_label] = aggregate_norm_ablations(seed_results)

    t_experiment_total = time.perf_counter() - t_experiment_start

    # ══════════════════════════════════════════════════════════════════════
    # Print summary tables
    # ══════════════════════════════════════════════════════════════════════
    _print_predictor_table(predictor_comparisons)
    _print_norm_table(norm_ablations)

    logger.info(f"\n  Total time: {t_experiment_total:.1f}s ({t_experiment_total / 60:.1f}m)")

    # ══════════════════════════════════════════════════════════════════════
    # Build and save result JSON
    # ══════════════════════════════════════════════════════════════════════
    all_source_files = []
    for files in source_files_map.values():
        all_source_files.extend(str(f) for f in files)

    envelope = build_experiment_envelope(
        experiment_name="ablation_study",
        source_files=all_source_files,
        seeds=seeds,
        target_n=args.target_n,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        threshold=args.threshold,
        total_time_s=t_experiment_total,
    )

    envelope["predictor_comparison"] = predictor_comparisons
    envelope["norm_ablation"] = norm_ablations
    envelope["verdict"] = _build_verdict(predictor_comparisons, norm_ablations)
    envelope["timing"] = {"total_s": t_experiment_total}

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"ablation_study_{timestamp}.json"
    json_dump(envelope, out_path)
    logger.info(f"\n  Results saved: {out_path}")

    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════════════════════


def _print_predictor_table(comparisons: dict[str, dict]) -> None:
    """Print predictor comparison table to logger."""
    logger.info(f"\n{'═' * 70}")
    logger.info("  PREDICTOR COMPARISON TABLE")
    logger.info(f"{'═' * 70}")
    logger.info(
        f"  {'Experiment':<15} {'Predictor':<10} {'Mean ΔE/gap':>12} "
        f"{'Pass Rate':>10} {'Spearman ρ':>11} {'Essential':>10}"
    )
    logger.info(f"  {'─' * 68}")

    for exp_label, agg in comparisons.items():
        if "error" in agg:
            logger.info(f"  {exp_label:<15} ERROR")
            continue

        essential = agg["graph_structure_essential"]
        essential_str = (
            f"{'YES' if essential['all_seeds'] else 'NO'} ({essential['mean_ratio']:.1f}×)"
        )

        for pred_name in ["GNN", "MLP", "Scipy"]:
            pred = agg["predictors"][pred_name]
            mean_de = pred["mean_de_gap"]["mean"]
            pass_rate = pred["pass_rate"]["mean"]
            rho = pred["spearman_rho"]["mean"]
            rho_str = f"{rho:.3f}" if not np.isnan(rho) else "N/A"

            ess_col = essential_str if pred_name == "GNN" else ""
            logger.info(
                f"  {exp_label if pred_name == 'GNN' else '':<15} "
                f"{pred_name:<10} {mean_de * 100:>10.3f}% "
                f"{pass_rate * 100:>8.0f}% "
                f"{rho_str:>11} {ess_col:>10}"
            )
        logger.info(f"  {'─' * 68}")


def _print_norm_table(ablations: dict[str, dict]) -> None:
    """Print norm ablation table (rows=norm_type, columns=experiment)."""
    logger.info(f"\n{'═' * 70}")
    logger.info("  NORM ABLATION TABLE (Mean ΔE/gap %)")
    logger.info(f"{'═' * 70}")

    # Header
    experiments = list(ablations.keys())
    header = f"  {'norm_type':<10}"
    for exp in experiments:
        header += f" {exp:>15}"
    logger.info(header)
    logger.info(f"  {'─' * (10 + 16 * len(experiments))}")

    # Rows
    for norm_type in NORM_TYPES:
        row = f"  {norm_type:<10}"
        for exp in experiments:
            agg = ablations[exp]
            if "error" in agg:
                row += f" {'ERROR':>15}"
            else:
                mean_de = agg["norm_types"][norm_type]["mean_de_gap"]["mean"]
                std_de = agg["norm_types"][norm_type]["mean_de_gap"]["std"]
                row += f" {mean_de * 100:>6.3f}±{std_de * 100:.3f}%"
        logger.info(row)

    # Best row
    row = f"  {'BEST':<10}"
    for exp in experiments:
        agg = ablations[exp]
        if "error" not in agg:
            row += f" {agg['best_norm_type']:>15}"
        else:
            row += f" {'—':>15}"
    logger.info(row)

    # Convergence table
    logger.info("\n  TRAINING CONVERGENCE (Final MSE / Epochs to convergence)")
    logger.info(f"  {'─' * (10 + 16 * len(experiments))}")
    for norm_type in NORM_TYPES:
        row = f"  {norm_type:<10}"
        for exp in experiments:
            agg = ablations[exp]
            if "error" in agg:
                row += f" {'ERROR':>15}"
            else:
                mse = agg["norm_types"][norm_type]["final_mse"]["mean"]
                conv = agg["norm_types"][norm_type]["epochs_to_convergence"]["mean"]
                row += f" {mse:.1e}/{int(conv):>4}"
        logger.info(row)


def _build_verdict(
    predictor_comparisons: dict[str, dict],
    norm_ablations: dict[str, dict],
) -> dict:
    """Build verdict summary from all ablation results."""
    # Collect graph_structure_essential across experiments
    essential_results: dict[str, bool] = {}
    for exp_label, agg in predictor_comparisons.items():
        if "error" not in agg:
            essential_results[exp_label] = agg["graph_structure_essential"]["all_seeds"]

    # Collect best norm per experiment
    best_norms: dict[str, str] = {}
    for exp_label, agg in norm_ablations.items():
        if "error" not in agg:
            best_norms[exp_label] = agg["best_norm_type"]

    return {
        "graph_structure_essential": essential_results,
        "best_norm_per_experiment": best_norms,
        "graph_essential_any": any(essential_results.values()) if essential_results else False,
    }


if __name__ == "__main__":
    sys.exit(main())
