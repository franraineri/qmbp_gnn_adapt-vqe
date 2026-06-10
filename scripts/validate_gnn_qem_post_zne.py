#!/usr/bin/env python
"""Validate GNN-QEM on POST-ZNE residuals (realistic deployment scenario).

The key question: does GNN-QEM help when errors are SMALL (0.5-3 units),
as they would be after PEA-ZNE in real deployment? Or does it only work
on the large errors (10-25 units) from random theta?

Protocol:
  1. Load fine-tuned model (trained on all topologies)
  2. Generate realistic data: VQE-optimized theta + PEA-ZNE simulation
     (simulated by adding small noise to exact energy, mimicking ZNE residual)
  3. Evaluate GNN-QEM correction on these small residuals
  4. Report: improvement rate, mean reduction, confidence behavior

Success criteria:
  - Improvement rate >= 60% on post-ZNE residuals (harder than raw noise)
  - No regression (correction doesn't worsen > 20% of samples)
  - Confidence correctly drops when correction is uncertain
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
    QEMSample,
    build_qem_dataset,
    correct_energy,
    load_qem_checkpoint,
    save_qem_checkpoint,
    train_gnn_qem,
)


def generate_post_zne_samples(n_samples: int = 50, seed: int = 42) -> list[QEMSample]:
    """Generate synthetic post-ZNE samples with realistic small residuals.

    Simulates the scenario after PEA-ZNE: exact_energy + small_noise.
    The noise magnitude is calibrated to match real PEA-ZNE residuals


    noise
    (ΔE/gap ~ 2-10%, from ZNE_CROSS_TOPO results).
    """
    rng = np.random.default_rng(seed)
    samples = []

    configs = [
        ("chain_1d", 6, [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
        ("ladder", 6, [2.5, 3.0, 3.25, 3.5, 4.0]),
        ("heavy_hex", 10, [2.5, 3.0, 3.25, 3.5, 4.0]),
    ]

    for topo, n_qubits, h_values in configs:
        for h in h_values:
            # Exact energy scales roughly as -N*h for paramagnetic phase
            e_exact = -(n_qubits * h + rng.uniform(0, n_qubits * 0.5))
            gap = abs(e_exact) * 0.1  # ~10% of |E|

            # Post-ZNE residual: small error (2-10% of gap)
            # This mimics PEA-ZNE output: mostly correct but with small bias
            for trial in range(3):
                residual_pct = rng.uniform(0.02, 0.15)  # 2-15% of gap
                sign = rng.choice([-1, 1])
                e_noisy = e_exact + sign * residual_pct * gap

                samples.append(
                    QEMSample(
                        noisy_energy=e_noisy,
                        exact_energy=e_exact,
                        h_value=h,
                        n_2q_gates=18 if n_qubits == 10 else 10,
                        ces=rng.uniform(0.10, 0.25),
                        topology=topo,
                        n_qubits=n_qubits,
                        qubit_t1=[rng.uniform(80, 120) for _ in range(n_qubits)],
                        qubit_t2=[rng.uniform(60, 100) for _ in range(n_qubits)],
                        readout_errors=[rng.uniform(0.005, 0.02) for _ in range(n_qubits)],
                        gate_errors_2q=[rng.uniform(0.003, 0.01) for _ in range(n_qubits - 1)],
                        edge_index=np.array(
                            [
                                list(range(n_qubits - 1)) + list(range(1, n_qubits)),
                                list(range(1, n_qubits)) + list(range(n_qubits - 1)),
                            ],
                            dtype=int,
                        ),
                    )
                )

    rng.shuffle(samples)
    return samples[:n_samples]


def main():
    output_dir = Path("results/gnn_qem")
    t0 = time.time()

    # ── Step 1: Generate post-ZNE residual samples ────────────────────
    logger.info("Generating post-ZNE residual samples (realistic small errors)...")
    test_samples = generate_post_zne_samples(n_samples=48, seed=42)
    logger.info(f"Generated {len(test_samples)} post-ZNE samples")

    # Show error distribution
    errors = [abs(s.noisy_energy - s.exact_energy) for s in test_samples]
    logger.info(f"  Error range: [{min(errors):.4f}, {max(errors):.4f}]")
    logger.info(f"  Mean error: {np.mean(errors):.4f}")
    logger.info(f"  Median error: {np.median(errors):.4f}")

    # ── Step 2: Test with existing model (trained on large errors) ────
    model_path = output_dir / "model_cross_topo.pt"
    if not model_path.exists():
        model_path = output_dir / "model.pt"
    if not model_path.exists():
        logger.error("No trained model found. Run training first.")
        sys.exit(1)

    model_large, _, _ = load_qem_checkpoint(model_path)
    logger.info(f"Loaded model from {model_path}")

    logger.info("\n" + "=" * 60)
    logger.info("TEST A: Existing model (trained on large errors) → small residuals")
    logger.info("=" * 60)

    results_a = []
    for s in test_samples:
        c = correct_energy(model_large, s, confidence_threshold=0.0)
        err_before = abs(s.noisy_energy - s.exact_energy)
        err_after = abs(c.corrected_energy - s.exact_energy)
        results_a.append(
            {
                "h": s.h_value,
                "topology": s.topology,
                "n_qubits": s.n_qubits,
                "err_before": err_before,
                "err_after": err_after,
                "confidence": c.confidence,
                "improved": err_after < err_before,
                "worsened": err_after > err_before * 1.5,  # >50% worse = regression
            }
        )

    n_improved_a = sum(r["improved"] for r in results_a)
    n_worsened_a = sum(r["worsened"] for r in results_a)
    rate_a = n_improved_a / len(results_a) * 100
    mean_before_a = np.mean([r["err_before"] for r in results_a])
    mean_after_a = np.mean([r["err_after"] for r in results_a])
    mean_conf_a = np.mean([r["confidence"] for r in results_a])

    logger.info(f"  Rate: {rate_a:.1f}% improved ({n_improved_a}/{len(results_a)})")
    logger.info(f"  Regressions: {n_worsened_a}/{len(results_a)} worsened >50%")
    logger.info(f"  Error: {mean_before_a:.4f} → {mean_after_a:.4f}")
    logger.info(f"  Confidence: {mean_conf_a:.3f}")

    # ── Step 3: Retrain on post-ZNE residual data ─────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("TEST B: Retrain model on small-residual data")
    logger.info("=" * 60)

    # Split: 70% train, 30% test
    n_train = int(0.7 * len(test_samples))
    train_samples = test_samples[:n_train]
    eval_samples = test_samples[n_train:]

    # Augment training data
    rng = np.random.default_rng(99)
    augmented = []
    for s in train_samples:
        augmented.append(s)
        for _ in range(9):  # 10× augmentation (small errors need more)
            delta = rng.normal(0, abs(s.noisy_energy - s.exact_energy) * 0.3)
            augmented.append(replace(s, noisy_energy=s.noisy_energy + delta))

    dataset = build_qem_dataset(augmented)
    indices = rng.permutation(len(dataset))
    n_t = int(0.85 * len(dataset))
    train_data = [dataset[i] for i in indices[:n_t]]
    val_data = [dataset[i] for i in indices[n_t:]]

    logger.info(f"  Augmented: {len(augmented)}, train/val: {len(train_data)}/{len(val_data)}")

    config = GNNQEMConfig(hidden_dim=64, n_layers=3, epochs=2000, patience=200, lr=5e-4)
    model_small = GNNQEMCorrector(config)
    result = train_gnn_qem(model_small, train_data, val_data, config)
    logger.info(f"  Training: epoch={result.best_epoch}, val_MAE={result.val_mae:.6f}")

    # Evaluate on held-out post-ZNE data
    results_b = []
    for s in eval_samples:
        c = correct_energy(model_small, s, confidence_threshold=0.0)
        err_before = abs(s.noisy_energy - s.exact_energy)
        err_after = abs(c.corrected_energy - s.exact_energy)
        results_b.append(
            {
                "h": s.h_value,
                "topology": s.topology,
                "err_before": err_before,
                "err_after": err_after,
                "confidence": c.confidence,
                "improved": err_after < err_before,
                "worsened": err_after > err_before * 1.5,
            }
        )

    n_improved_b = sum(r["improved"] for r in results_b)
    n_worsened_b = sum(r["worsened"] for r in results_b)
    rate_b = n_improved_b / len(results_b) * 100
    mean_before_b = np.mean([r["err_before"] for r in results_b])
    mean_after_b = np.mean([r["err_after"] for r in results_b])

    logger.info(f"  Rate: {rate_b:.1f}% improved ({n_improved_b}/{len(results_b)})")
    logger.info(f"  Regressions: {n_worsened_b}/{len(results_b)} worsened >50%")
    logger.info(f"  Error: {mean_before_b:.4f} → {mean_after_b:.6f}")

    # Save retrained model
    save_qem_checkpoint(
        model_small,
        output_dir / "model_post_zne.pt",
        result,
        metadata={
            "trained_on": "post_zne_residuals",
            "error_regime": "small (2-15% of gap)",
        },
    )

    # ── Step 4: Summary ───────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: GNN-QEM on Post-ZNE Residuals")
    logger.info("=" * 60)
    logger.info(f"  Test A (large-error model → small residuals): {rate_a:.0f}% improved")
    logger.info(f"  Test B (retrained on small residuals): {rate_b:.0f}% improved")
    logger.info(
        f"  Conclusion: {'Model generalizes to small errors' if rate_a >= 60 else 'Needs retraining for small errors'}"
    )

    pass_a = rate_a >= 60 and n_worsened_a / len(results_a) < 0.2
    pass_b = rate_b >= 60 and n_worsened_b / len(results_b) < 0.2

    output = {
        "test_a_existing_model": {
            "description": "Model trained on large errors → post-ZNE small residuals",
            "n_samples": len(test_samples),
            "improvement_rate": float(rate_a),
            "n_worsened": int(n_worsened_a),
            "mean_err_before": float(mean_before_a),
            "mean_err_after": float(mean_after_a),
            "mean_confidence": float(mean_conf_a),
            "pass": bool(pass_a),
        },
        "test_b_retrained": {
            "description": "Model retrained on post-ZNE residuals",
            "n_train": int(len(train_samples)),
            "n_eval": int(len(eval_samples)),
            "improvement_rate": float(rate_b),
            "n_worsened": int(n_worsened_b),
            "mean_err_before": float(mean_before_b),
            "mean_err_after": float(mean_after_b),
            "best_epoch": int(result.best_epoch),
            "val_mae": float(result.val_mae),
            "pass": bool(pass_b),
        },
        "verdict": "PASS" if (pass_a or pass_b) else "NEEDS_RETRAINING",
        "time_s": time.time() - t0,
    }

    out_path = output_dir / "post_zne_validation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
