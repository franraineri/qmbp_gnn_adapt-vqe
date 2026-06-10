#!/usr/bin/env python
"""GNN-QEM Cross-Topology Validation — Test generalization to heavy_hex N=10.

Hypothesis: GNN-QEM trained on chain_1d+ladder (N=6) generalizes to heavy_hex
(N=10) because it learns noise propagation patterns from the hardware graph,
not topology-specific energy mappings.

Protocol:
  1. Load model trained on chain_1d + ladder (from run_gnn_qem_training.py)
  2. Generate held-out test data: heavy_hex N=10 (unseen topology + size)
  3. Zero-shot evaluate on heavy_hex WITHOUT retraining
  4. Fine-tune with heavy_hex included, evaluate on held-out subset
  5. Report generalization gap

Success criteria:
  - Zero-shot improvement rate >= 70% on heavy_hex
  - Fine-tuned improvement rate >= 70% on held-out heavy_hex

Usage:
    .venv/bin/python scripts/run_gnn_qem_cross_topology.py

Prerequisite:
    .venv/bin/python scripts/run_gnn_qem_training.py  (creates model.pt)
"""

import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("qiskit.passmanager").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qmbp_simulation.predictors.gnn_qem import (
    GNNQEMConfig,
    GNNQEMCorrector,
    build_qem_dataset,
    correct_energy,
    generate_qem_training_data,
    load_qem_checkpoint,
    load_qem_samples,
    save_qem_checkpoint,
    save_qem_samples,
    train_gnn_qem,
)


def main():
    output_dir = Path("results/gnn_qem")
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── Step 1: Load trained model ────────────────────────────────────
    model_path = output_dir / "model.pt"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run run_gnn_qem_training.py first.")
        sys.exit(1)

    model, train_result, metadata = load_qem_checkpoint(model_path)
    logger.info(f"Loaded model: {sum(p.numel() for p in model.parameters()):,} params")

    # ── Step 2: Generate heavy_hex N=10 test data ─────────────────────
    hh_data_path = output_dir / "test_data_heavy_hex.json"
    if hh_data_path.exists():
        hh_samples = load_qem_samples(hh_data_path)
    else:
        logger.info("Generating heavy_hex N=10 test data...")
        hh_samples = generate_qem_training_data(
            topologies=["heavy_hex"],
            n_qubits_list=[10],
            h_values=[2.5, 3.0, 3.25, 3.5, 4.0],
            seeds=[42, 43, 44],
            shots=4096,
            p_layers=1,
        )
        if not hh_samples:
            logger.error("No heavy_hex samples generated")
            sys.exit(1)
        save_qem_samples(hh_samples, hh_data_path)

    logger.info(f"Heavy_hex test set: {len(hh_samples)} samples")

    # ── Step 3: Zero-shot evaluation ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("ZERO-SHOT: chain_1d+ladder model -> heavy_hex N=10")
    logger.info("=" * 60)

    results_zs = []
    n_improved_zs = 0
    for s in hh_samples:
        c = correct_energy(model, s, confidence_threshold=0.0)
        err_before = abs(s.noisy_energy - s.exact_energy)
        err_after = abs(c.corrected_energy - s.exact_energy)
        improved = err_after < err_before
        if improved:
            n_improved_zs += 1
        results_zs.append(
            {
                "h": s.h_value,
                "n_qubits": s.n_qubits,
                "n_2q_gates": s.n_2q_gates,
                "ces": s.ces,
                "e_noisy": s.noisy_energy,
                "e_exact": s.exact_energy,
                "e_corrected": c.corrected_energy,
                "delta_e_predicted": c.delta_e_predicted,
                "err_before": err_before,
                "err_after": err_after,
                "confidence": c.confidence,
                "improved": improved,
            }
        )

    zs_rate = n_improved_zs / max(len(hh_samples), 1) * 100
    errs_before = [r["err_before"] for r in results_zs]
    errs_after = [r["err_after"] for r in results_zs]
    mean_before_zs = float(np.mean(errs_before))
    mean_after_zs = float(np.mean(errs_after))
    median_before_zs = float(np.median(errs_before))
    median_after_zs = float(np.median(errs_after))
    max_residual_zs = float(np.max(errs_after))
    reduction_zs = (1 - mean_after_zs / max(mean_before_zs, 1e-10)) * 100
    mean_conf_zs = float(np.mean([r["confidence"] for r in results_zs]))

    logger.info(f"  Rate: {zs_rate:.1f}% ({n_improved_zs}/{len(hh_samples)})")
    logger.info(f"  Mean error: {mean_before_zs:.3f} -> {mean_after_zs:.4f} ({reduction_zs:+.1f}%)")
    logger.info(f"  Median error: {median_before_zs:.3f} -> {median_after_zs:.4f}")
    logger.info(f"  Max residual: {max_residual_zs:.4f}")
    logger.info(f"  Confidence: {mean_conf_zs:.3f}")

    # ── Step 4: Fine-tuned evaluation ─────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINE-TUNED: retrain with chain_1d + ladder + heavy_hex")
    logger.info("=" * 60)

    # Load original training data
    train_data_path = output_dir / "training_data.json"
    train_samples = load_qem_samples(train_data_path) if train_data_path.exists() else []

    # Split heavy_hex: 60% for fine-tuning, 40% for held-out test
    # Shuffle to avoid h-value bias (data is ordered by h then seed)
    hh_indices = np.random.default_rng(77).permutation(len(hh_samples))
    n_ft = int(0.6 * len(hh_samples))
    hh_finetune = [hh_samples[i] for i in hh_indices[:n_ft]]
    hh_test = [hh_samples[i] for i in hh_indices[n_ft:]]

    # Combine and augment
    all_samples = train_samples + hh_finetune
    rng = np.random.default_rng(42)
    augmented = []
    for s in all_samples:
        augmented.append(s)
        for _ in range(4):
            delta = rng.normal(0, abs(s.noisy_energy - s.exact_energy) * 0.2)
            augmented.append(replace(s, noisy_energy=s.noisy_energy + delta))

    dataset = build_qem_dataset(augmented)
    indices = rng.permutation(len(dataset))
    n_train = int(0.85 * len(dataset))
    train_data = [dataset[i] for i in indices[:n_train]]
    val_data = [dataset[i] for i in indices[n_train:]]
    logger.info(
        f"  Dataset: {len(augmented)} augmented, {len(train_data)} train, {len(val_data)} val"
    )

    # Train
    config_ft = GNNQEMConfig(hidden_dim=64, n_layers=3, epochs=1000, patience=100, lr=1e-3)
    model_ft = GNNQEMCorrector(config_ft)
    result_ft = train_gnn_qem(model_ft, train_data, val_data, config_ft)
    logger.info(f"  Training: epoch={result_ft.best_epoch}, val_MAE={result_ft.val_mae:.4f}")

    # Evaluate on held-out heavy_hex
    results_ft = []
    n_improved_ft = 0
    for s in hh_test:
        c = correct_energy(model_ft, s, confidence_threshold=0.0)
        err_before = abs(s.noisy_energy - s.exact_energy)
        err_after = abs(c.corrected_energy - s.exact_energy)
        improved = err_after < err_before
        if improved:
            n_improved_ft += 1
        results_ft.append(
            {
                "h": s.h_value,
                "err_before": err_before,
                "err_after": err_after,
                "confidence": c.confidence,
                "improved": improved,
            }
        )

    ft_rate = n_improved_ft / max(len(hh_test), 1) * 100
    mean_before_ft = float(np.mean([r["err_before"] for r in results_ft])) if results_ft else 0
    mean_after_ft = float(np.mean([r["err_after"] for r in results_ft])) if results_ft else 0
    reduction_ft = (1 - mean_after_ft / max(mean_before_ft, 1e-10)) * 100 if results_ft else 0

    logger.info(f"  Held-out heavy_hex: {ft_rate:.1f}% ({n_improved_ft}/{len(hh_test)})")
    logger.info(f"  Error: {mean_before_ft:.3f} -> {mean_after_ft:.4f} ({reduction_ft:+.1f}%)")

    # Save fine-tuned model
    save_qem_checkpoint(
        model_ft,
        output_dir / "model_cross_topo.pt",
        result_ft,
        metadata={
            "topologies": ["chain_1d", "ladder", "heavy_hex"],
            "n_qubits_trained": [6, 10],
        },
    )

    # ── Step 5: Summary ───────────────────────────────────────────────
    zs_pass = zs_rate >= 70.0
    ft_pass = ft_rate >= 70.0

    logger.info("")
    logger.info("=" * 60)
    logger.info("VERDICT")
    logger.info("=" * 60)
    logger.info(f"  Zero-shot: {zs_rate:.1f}% {'PASS' if zs_pass else 'FAIL'}")
    logger.info(f"  Fine-tuned: {ft_rate:.1f}% {'PASS' if ft_pass else 'FAIL'}")
    logger.info("=" * 60)

    output = {
        "zero_shot": {
            "n_samples": len(hh_samples),
            "improvement_rate": zs_rate,
            "mean_err_before": mean_before_zs,
            "mean_err_after": mean_after_zs,
            "median_err_before": median_before_zs,
            "median_err_after": median_after_zs,
            "max_residual": max_residual_zs,
            "reduction_pct": reduction_zs,
            "mean_confidence": mean_conf_zs,
            "pass": zs_pass,
            "per_sample": results_zs,
        },
        "fine_tuned": {
            "n_finetune": n_ft,
            "n_test": len(hh_test),
            "improvement_rate": ft_rate,
            "mean_err_before": mean_before_ft,
            "mean_err_after": mean_after_ft,
            "reduction_pct": reduction_ft,
            "best_epoch": result_ft.best_epoch,
            "val_mae": result_ft.val_mae,
            "pass": ft_pass,
            "per_sample": results_ft,
        },
        "verdict": "PASS" if (zs_pass or ft_pass) else "FAIL",
        "time_s": time.time() - t0,
    }
    out_path = output_dir / "cross_topology_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
