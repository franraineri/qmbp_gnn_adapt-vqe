#!/usr/bin/env python
"""GNN-QEM V2 Training Pipeline — Scalable architecture with per-edge features.

Generates V2 training data using θ_opt from Zoo/NPZ (not random),
trains the GATv2Conv-based GNNQEMCorrectorV2, evaluates, and compares vs V1.

Usage:
    python scripts/experiment_runners/gnn_experiments/run_gnn_qem_v2_training.py
    python scripts/experiment_runners/gnn_experiments/run_gnn_qem_v2_training.py --quick
    python scripts/experiment_runners/gnn_experiments/run_gnn_qem_v2_training.py --skip-gen

Flags:
    --quick:     Minimal config (chain_1d N=6, 4 h-values, 2048 shots) ~3 min
    --skip-gen:  Load existing V2 training data from disk (skip FakeTorino)
    --compare:   Also train V1 model on same data for A/B comparison

Output:
    results/gnn_qem_v2/training_data_v2.json  — Persisted V2 samples
    results/gnn_qem_v2/model_v2.pt            — Trained V2 checkpoint
    results/gnn_qem_v2/evaluation_v2.json     — Per-sample results + summary
    results/gnn_qem_v2/comparison_v1_v2.json  — A/B comparison (if --compare)
"""

import json
import logging
import time
from pathlib import Path

import numpy as np

from qmbp_simulation.predictors.gnn_qem import (
    # V1 (for comparison)
    GNNQEMConfig,
    GNNQEMCorrector,
    build_qem_dataset,
    correct_energy,
    train_gnn_qem,
    # V2
    GNNQEMConfigV2,
    GNNQEMCorrectorV2,
    QEMSampleV2,
    build_qem_graph_v2,
    correct_energy_v2,
    generate_qem_training_data_v2,
    load_qem_samples_v2,
    save_qem_samples_v2,
    save_qem_v2_checkpoint,
    train_gnn_qem_v2,
)
from qmbp_simulation.utils.helpers import json_dump

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("qiskit.passmanager").setLevel(logging.WARNING)


OUTPUT_DIR = Path("results/gnn_qem_v2")


def parse_args():
    """Parse CLI arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="GNN-QEM V2 Training Pipeline")
    parser.add_argument("--quick", action="store_true", help="Minimal config for fast validation")
    parser.add_argument("--skip-gen", action="store_true", help="Load existing training data")
    parser.add_argument("--compare", action="store_true", help="Train V1 for A/B comparison")
    parser.add_argument("--shots", type=int, default=8192, help="Shots per noisy estimation")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--theta-source", choices=["zoo", "npz", "random"], default="zoo")
    return parser.parse_args()


def generate_or_load_data(args) -> list[QEMSampleV2]:
    """Step 1: Generate V2 training data or load from disk."""
    data_path = OUTPUT_DIR / ("training_data_v2_quick.json" if args.quick else "training_data_v2.json")

    if args.skip_gen and data_path.exists():
        logger.info(f"Loading existing V2 data from {data_path}")
        return load_qem_samples_v2(data_path)

    logger.info(f"Generating V2 training data (theta_source={args.theta_source})...")

    if args.quick:
        samples = generate_qem_training_data_v2(
            topologies=["chain_1d"],
            n_qubits_list=[6],
            h_values=[2.0, 2.5, 3.0, 3.5],
            shots=min(args.shots, 2048),
            theta_source=args.theta_source,
        )
    else:
        samples = generate_qem_training_data_v2(
            topologies=["chain_1d", "ladder"],
            n_qubits_list=[6, 10],
            h_values=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            shots=args.shots,
            theta_source=args.theta_source,
        )

    # Persist for reuse
    save_qem_samples_v2(samples, data_path)
    return samples


def augment_samples(samples: list[QEMSampleV2], n_augment: int = 4) -> list[QEMSampleV2]:
    """Step 2: Augment with shot-noise perturbations (same strategy as V1)."""
    from dataclasses import replace
    rng = np.random.default_rng(42)
    augmented = []
    for s in samples:
        augmented.append(s)
        delta = abs(s.noisy_energy - s.exact_energy)
        for _ in range(n_augment):
            noise = rng.normal(0, max(delta * 0.2, 0.001))
            augmented.append(replace(s, noisy_energy=s.noisy_energy + noise))
    return augmented


def build_v2_dataset(samples: list[QEMSampleV2], augment_cal: bool = True) -> list:
    """Step 3: Convert samples to graph Data objects."""
    dataset = []
    for s in samples:
        data = build_qem_graph_v2(s, augment=augment_cal, augment_scale=0.2)
        dataset.append(data)
    return dataset


def train_v2_model(train_data, val_data, args) -> tuple:
    """Step 4: Train GNNQEMCorrectorV2."""
    config = GNNQEMConfigV2(
        hidden_dim=64 if args.quick else 128,
        n_heads=2 if args.quick else 4,
        n_layers=2 if args.quick else 4,
        epochs=args.epochs or (500 if args.quick else 3000),
        patience=50 if args.quick else 300,
    )
    model = GNNQEMCorrectorV2(config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"V2 Model: {n_params:,} params, h={config.hidden_dim}, "
                f"heads={config.n_heads}, layers={config.n_layers}")

    train_result = train_gnn_qem_v2(model, train_data, val_data, config)
    return model, train_result, config


def train_v1_baseline(samples: list[QEMSampleV2], args) -> tuple:
    """Optional: Train V1 model on same data for comparison."""
    # Convert V2 samples to V1 graphs (uses V1 build_qem_graph via build_qem_dataset)
    from qmbp_simulation.predictors.gnn_qem import QEMSample
    v1_samples = []
    for s in samples:
        # Downcast to V1 sample (loses V2 fields)
        v1_samples.append(QEMSample(
            noisy_energy=s.noisy_energy, exact_energy=s.exact_energy,
            h_value=s.h_value, n_2q_gates=s.n_2q_gates, ces=s.ces,
            topology=s.topology, n_qubits=s.n_qubits,
            qubit_t1=s.qubit_t1, qubit_t2=s.qubit_t2,
            readout_errors=s.readout_errors, gate_errors_2q=s.gate_errors_2q,
            edge_index=s.edge_index,
        ))

    dataset = build_qem_dataset(v1_samples)
    rng = np.random.default_rng(99)
    indices = rng.permutation(len(dataset))
    n_train = int(0.8 * len(dataset))
    train_data = [dataset[i] for i in indices[:n_train]]
    val_data = [dataset[i] for i in indices[n_train:]]

    config = GNNQEMConfig(hidden_dim=64, n_layers=3, epochs=500 if args.quick else 1000, patience=100)
    model = GNNQEMCorrector(config)
    train_result = train_gnn_qem(model, train_data, val_data, config)
    return model, train_result


def evaluate_model(model, samples: list[QEMSampleV2], version: int = 2) -> list[dict]:
    """Step 5: Evaluate model on all original (non-augmented) samples."""
    results = []
    for s in samples:
        if version == 2:
            correction = correct_energy_v2(model, s, confidence_threshold=0.0)
        else:
            from qmbp_simulation.predictors.gnn_qem import QEMSample
            v1_s = QEMSample(
                noisy_energy=s.noisy_energy, exact_energy=s.exact_energy,
                h_value=s.h_value, n_2q_gates=s.n_2q_gates, ces=s.ces,
                topology=s.topology, n_qubits=s.n_qubits,
                qubit_t1=s.qubit_t1, qubit_t2=s.qubit_t2,
                readout_errors=s.readout_errors, gate_errors_2q=s.gate_errors_2q,
                edge_index=s.edge_index,
            )
            correction = correct_energy(model, v1_s, confidence_threshold=0.0)

        err_before = abs(s.noisy_energy - s.exact_energy)
        err_after = abs(correction.corrected_energy - s.exact_energy)
        results.append({
            "h": s.h_value, "topology": s.topology, "n_qubits": s.n_qubits,
            "e_noisy": s.noisy_energy, "e_exact": s.exact_energy,
            "e_corrected": correction.corrected_energy,
            "delta_e_predicted": correction.delta_e_predicted,
            "confidence": correction.confidence,
            "err_before": err_before, "err_after": err_after,
            "improved": err_after < err_before,
            "gap": getattr(s, "gap", 0.0),
        })
    return results


def compute_summary(results: list[dict]) -> dict:
    """Compute aggregate metrics from per-sample results."""
    n = len(results)
    n_improved = sum(1 for r in results if r["improved"])
    err_before = [r["err_before"] for r in results]
    err_after = [r["err_after"] for r in results]
    return {
        "n_samples": n,
        "improvement_rate_pct": n_improved / max(n, 1) * 100,
        "n_improved": n_improved,
        "mean_err_before": float(np.mean(err_before)),
        "mean_err_after": float(np.mean(err_after)),
        "mean_reduction_pct": (1 - np.mean(err_after) / max(np.mean(err_before), 1e-10)) * 100,
        "median_err_before": float(np.median(err_before)),
        "median_err_after": float(np.median(err_after)),
    }


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── Step 1: Data ──────────────────────────────────────────────────────
    samples = generate_or_load_data(args)
    logger.info(f"Raw samples: {len(samples)}")
    t_data = time.time() - t0

    # ── Step 2: Augment ───────────────────────────────────────────────────
    augmented = augment_samples(samples, n_augment=4)
    logger.info(f"Augmented: {len(augmented)} (5× original)")

    # ── Step 3: Build V2 graphs + split ───────────────────────────────────
    dataset = build_v2_dataset(augmented, augment_cal=True)
    rng = np.random.default_rng(99)
    indices = rng.permutation(len(dataset))
    n_train = int(0.8 * len(dataset))
    train_data = [dataset[i] for i in indices[:n_train]]
    val_data = [dataset[i] for i in indices[n_train:]]
    logger.info(f"Split: {len(train_data)} train, {len(val_data)} val")

    # ── Step 4: Train V2 ─────────────────────────────────────────────────
    model_v2, result_v2, config_v2 = train_v2_model(train_data, val_data, args)
    t_train = time.time() - t0 - t_data
    logger.info(f"V2 trained: epoch={result_v2.best_epoch}, "
                f"val_MAE={result_v2.val_mae:.4f}, imp={result_v2.val_improvement_pct:.1f}%")

    # Save V2 checkpoint
    ckpt_path = OUTPUT_DIR / ("model_v2_quick.pt" if args.quick else "model_v2.pt")
    save_qem_v2_checkpoint(model_v2, ckpt_path, result_v2, metadata={
        "mode": "quick" if args.quick else "full",
        "theta_source": args.theta_source,
        "n_samples": len(samples),
    })

    # ── Step 5: Evaluate V2 ──────────────────────────────────────────────
    v2_results = evaluate_model(model_v2, samples, version=2)
    v2_summary = compute_summary(v2_results)
    v2_summary.update({
        "model_params": sum(p.numel() for p in model_v2.parameters()),
        "best_epoch": result_v2.best_epoch,
        "val_mae": result_v2.val_mae,
        "time_data_s": t_data,
        "time_train_s": t_train,
    })

    # ── Step 6 (optional): Compare with V1 ───────────────────────────────
    comparison = None
    if args.compare:
        logger.info("Training V1 baseline for comparison...")
        model_v1, result_v1 = train_v1_baseline(augmented, args)
        v1_results = evaluate_model(model_v1, samples, version=1)
        v1_summary = compute_summary(v1_results)

        comparison = {
            "v1": v1_summary,
            "v2": v2_summary,
            "v2_vs_v1_improvement_pct": v2_summary["mean_reduction_pct"] - v1_summary["mean_reduction_pct"],
            "v2_wins": v2_summary["improvement_rate_pct"] > v1_summary["improvement_rate_pct"],
        }
        json_dump(comparison, OUTPUT_DIR / "comparison_v1_v2.json")
        logger.info(f"V1 imp_rate={v1_summary['improvement_rate_pct']:.1f}%, "
                    f"V2 imp_rate={v2_summary['improvement_rate_pct']:.1f}%")

    # ── Step 7: Save results ─────────────────────────────────────────────
    eval_path = OUTPUT_DIR / ("evaluation_v2_quick.json" if args.quick else "evaluation_v2.json")
    json_dump({"summary": v2_summary, "per_sample": v2_results}, eval_path)

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("GNN-QEM V2 RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Samples: {len(samples)} raw, {len(augmented)} augmented")
    logger.info(f"  Training: {result_v2.best_epoch} epochs, val_MAE={result_v2.val_mae:.4f}")
    logger.info(f"  Improvement: {v2_summary['improvement_rate_pct']:.1f}% "
                f"({v2_summary['n_improved']}/{len(samples)})")
    logger.info(f"  Error: {v2_summary['mean_err_before']:.4f} → {v2_summary['mean_err_after']:.4f} "
                f"({v2_summary['mean_reduction_pct']:+.1f}%)")
    logger.info(f"  Time: data={t_data:.1f}s, train={t_train:.1f}s, total={time.time()-t0:.1f}s")
    if comparison:
        logger.info(f"  V2 vs V1: {'V2 WINS' if comparison['v2_wins'] else 'V1 WINS'} "
                    f"(Δ={comparison['v2_vs_v1_improvement_pct']:+.1f}%)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
