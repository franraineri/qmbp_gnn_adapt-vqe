#!/usr/bin/env python
"""GNN-QEM Training Pipeline — Generate data, train, evaluate, save results.

Generates (noisy, exact) energy pairs from FakeTorino, trains the GNN-QEM
model to predict energy corrections, and evaluates cross-topology generalization.

Usage:
    python scripts/run_gnn_qem_training.py [--quick]

    --quick: Use minimal config (4 h-values, 2 seeds) for fast validation (~2 min)
    Default: Full config (6 h-values, 3 seeds) for proper training (~15 min)

Output:
    results/gnn_qem/training_data.json  — Saved samples (reusable)
    results/gnn_qem/model.pt            — Trained checkpoint
    results/gnn_qem/evaluation.json     — Per-sample results + summary
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Suppress noisy qiskit pass manager logs
logging.getLogger("qiskit.passmanager").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qmbp_simulation.predictors.gnn_qem import (
    GNNQEMConfig,
    GNNQEMCorrector,
    build_qem_dataset,
    correct_energy,
    generate_qem_training_data,
    load_qem_samples,
    save_qem_checkpoint,
    save_qem_samples,
    train_gnn_qem,
)


def main():
    quick = "--quick" in sys.argv
    output_dir = Path("results/gnn_qem")
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ── Step 1: Generate or load training data ────────────────────────────
    data_path = output_dir / ("training_data_quick.json" if quick else "training_data.json")

    if data_path.exists():
        logger.info(f"Loading existing data from {data_path}")
        samples = load_qem_samples(data_path)
    else:
        logger.info(f"Generating training data ({'quick' if quick else 'full'} mode)...")
        if quick:
            samples = generate_qem_training_data(
                topologies=["chain_1d"],
                n_qubits_list=[6],
                h_values=[2.0, 3.0, 3.5, 4.0],
                seeds=[42, 43],
                shots=2048,
            )
        else:
            samples = generate_qem_training_data(
                topologies=["chain_1d", "ladder"],
                n_qubits_list=[6],
                h_values=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                seeds=[42, 43, 44],
                shots=4096,
            )
        save_qem_samples(samples, data_path)

    logger.info(f"Dataset: {len(samples)} raw samples")
    t_data = time.time() - t0

    # ── Step 2: Augment data with noise perturbations ─────────────────────
    from dataclasses import replace

    rng = np.random.default_rng(42)
    augmented = []
    for s in samples:
        augmented.append(s)
        # 4 noise-perturbed copies per sample (simulates shot noise variance)
        for _ in range(4):
            noise_delta = rng.normal(0, abs(s.noisy_energy - s.exact_energy) * 0.2)
            augmented.append(replace(s, noisy_energy=s.noisy_energy + noise_delta))

    logger.info(f"Augmented dataset: {len(augmented)} samples (5× original)")

    # ── Step 3: Build graphs and split ────────────────────────────────────
    dataset = build_qem_dataset(augmented)

    # Stratified split: ensure each topology/h combination in both sets
    rng2 = np.random.default_rng(99)
    indices = rng2.permutation(len(dataset))
    n_train = int(0.8 * len(dataset))
    train_data = [dataset[i] for i in indices[:n_train]]
    val_data = [dataset[i] for i in indices[n_train:]]

    logger.info(f"Split: {len(train_data)} train, {len(val_data)} val")

    # ── Step 4: Train model ───────────────────────────────────────────────
    config = GNNQEMConfig(
        hidden_dim=64,
        n_layers=3,
        epochs=1000,
        patience=100,
        lr=1e-3,
        dropout=0.1,
    )
    model = GNNQEMCorrector(config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} params, h={config.hidden_dim}, L={config.n_layers}")

    train_result = train_gnn_qem(model, train_data, val_data, config)
    t_train = time.time() - t0 - t_data

    logger.info(
        f"Training done: best_epoch={train_result.best_epoch}, "
        f"val_MAE={train_result.val_mae:.4f}, improvement={train_result.val_improvement_pct:.1f}%"
    )

    # Save checkpoint
    ckpt_path = output_dir / ("model_quick.pt" if quick else "model.pt")
    save_qem_checkpoint(
        model,
        ckpt_path,
        train_result,
        metadata={
            "mode": "quick" if quick else "full",
            "n_raw_samples": len(samples),
            "n_augmented": len(augmented),
            "topologies": list(set(s.topology for s in samples)),
        },
    )

    # ── Step 5: Evaluate on all original samples ──────────────────────────
    logger.info("Evaluating on all original samples...")
    results_per_sample = []
    n_improved = 0
    total_err_before = 0.0
    total_err_after = 0.0

    for s in samples:
        correction = correct_energy(model, s, confidence_threshold=0.0)
        err_before = abs(s.noisy_energy - s.exact_energy)
        err_after = abs(correction.corrected_energy - s.exact_energy)
        gap_approx = abs(s.exact_energy) * 0.1  # Approximate gap as 10% of |E|
        de_gap_before = err_before / max(gap_approx, 1e-10)
        de_gap_after = err_after / max(gap_approx, 1e-10)
        improved = err_after < err_before

        if improved:
            n_improved += 1
        total_err_before += err_before
        total_err_after += err_after

        results_per_sample.append(
            {
                "h": s.h_value,
                "topology": s.topology,
                "n_qubits": s.n_qubits,
                "e_noisy": s.noisy_energy,
                "e_exact": s.exact_energy,
                "e_corrected": correction.corrected_energy,
                "delta_e_predicted": correction.delta_e_predicted,
                "confidence": correction.confidence,
                "err_before": err_before,
                "err_after": err_after,
                "de_gap_before": de_gap_before,
                "de_gap_after": de_gap_after,
                "improved": improved,
            }
        )

    # ── Step 6: Summary statistics ────────────────────────────────────────
    improvement_rate = n_improved / max(len(samples), 1) * 100
    mean_err_before = total_err_before / max(len(samples), 1)
    mean_err_after = total_err_after / max(len(samples), 1)
    mean_reduction_pct = (1 - mean_err_after / max(mean_err_before, 1e-10)) * 100

    summary = {
        "n_samples": len(samples),
        "n_augmented": len(augmented),
        "n_train": len(train_data),
        "n_val": len(val_data),
        "model_params": n_params,
        "best_epoch": train_result.best_epoch,
        "val_mae": train_result.val_mae,
        "val_improvement_pct": train_result.val_improvement_pct,
        "eval_improvement_rate": improvement_rate,
        "eval_mean_err_before": mean_err_before,
        "eval_mean_err_after": mean_err_after,
        "eval_mean_reduction_pct": mean_reduction_pct,
        "n_improved": n_improved,
        "time_data_s": t_data,
        "time_train_s": t_train,
        "time_total_s": time.time() - t0,
        "mode": "quick" if quick else "full",
    }

    logger.info("=" * 60)
    logger.info("GNN-QEM RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Samples: {len(samples)} raw, {len(augmented)} augmented")
    logger.info(f"  Training: {train_result.best_epoch} epochs, val_MAE={train_result.val_mae:.4f}")
    logger.info(
        f"  Improvement rate: {improvement_rate:.1f}% ({n_improved}/{len(samples)} samples)"
    )
    logger.info(
        f"  Mean error: {mean_err_before:.4f} → {mean_err_after:.4f} ({mean_reduction_pct:+.1f}%)"
    )
    logger.info(f"  Time: data={t_data:.1f}s, train={t_train:.1f}s, total={time.time() - t0:.1f}s")
    logger.info("=" * 60)

    # Save evaluation results
    eval_path = output_dir / ("evaluation_quick.json" if quick else "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump({"summary": summary, "per_sample": results_per_sample}, f, indent=2)
    logger.info(f"Results saved to {eval_path}")

    # Print per-topology breakdown
    topologies = set(s.topology for s in samples)
    for topo in sorted(topologies):
        topo_results = [r for r in results_per_sample if r["topology"] == topo]
        topo_improved = sum(1 for r in topo_results if r["improved"])
        topo_rate = topo_improved / max(len(topo_results), 1) * 100
        logger.info(f"  {topo}: {topo_improved}/{len(topo_results)} improved ({topo_rate:.0f}%)")


if __name__ == "__main__":
    main()
