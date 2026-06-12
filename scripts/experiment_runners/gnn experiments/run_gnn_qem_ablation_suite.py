#!/usr/bin/env python
"""GNN-QEM Ablation Suite — Validate that the GNN learns from graph structure.

Tests the claim: "GNN learns noise propagation from hardware topology graph
and generalizes zero-shot to unseen topologies."

Ablations:
  V1. MLP context-only (no graph) — proves graph structure is essential
  V2. Shuffled graph edges — proves topology structure matters
  V3. Multi-seed reproducibility — proves result is not split-dependent
  V5. Linear regression baseline — proves the task is non-trivial

Each ablation trains a model variant, evaluates on the SAME heavy_hex test
data, and compares improvement rate vs the full GNN (100% baseline).

Success criteria for claim:
  - V1: GNN >> MLP (at least 20% advantage in improvement rate)
  - V2: GNN >> Shuffled (shuffled degrades significantly)
  - V3: GNN improvement rate std < 15% across 3 seeds
  - V5: GNN >> Linear (linear R² < 0.8 OR improvement rate < 70%)

Usage:
    .venv/bin/python "scripts/experiment_runners/gnn experiments/run_gnn_qem_ablation_suite.py"

Prerequisite:
    results/gnn_qem/training_data.json (from run_gnn_qem_training.py)
    results/gnn_qem/test_data_heavy_hex.json (from run_gnn_qem_cross_topology.py)
"""

import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("qiskit.passmanager").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qmbp_simulation.predictors.gnn_qem import (
    GNNQEMConfig,
    GNNQEMCorrector,
    QEMSample,
    build_qem_dataset,
    correct_energy,
    load_qem_checkpoint,
    load_qem_samples,
    train_gnn_qem,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def augment(samples: list[QEMSample], factor: int = 5, seed: int = 42) -> list[QEMSample]:
    """Standard augmentation: noise perturbation copies."""
    rng = np.random.default_rng(seed)
    out = list(samples)
    for s in samples:
        err = abs(s.noisy_energy - s.exact_energy)
        for _ in range(factor - 1):
            delta = rng.normal(0, max(err * 0.2, 0.01))
            out.append(replace(s, noisy_energy=s.noisy_energy + delta))
    return out


def evaluate_on_test(model, test_samples: list[QEMSample], threshold: float = 0.0) -> dict:
    """Evaluate model on test samples, return metrics."""
    n_improved = 0
    errs_before, errs_after = [], []
    for s in test_samples:
        c = correct_energy(model, s, confidence_threshold=threshold)
        eb = abs(s.noisy_energy - s.exact_energy)
        ea = abs(c.corrected_energy - s.exact_energy)
        errs_before.append(eb)
        errs_after.append(ea)
        if ea < eb:
            n_improved += 1
    rate = n_improved / max(len(test_samples), 1) * 100
    mean_before = float(np.mean(errs_before))
    mean_after = float(np.mean(errs_after))
    reduction = (1 - mean_after / max(mean_before, 1e-10)) * 100
    return {
        "improvement_rate": rate,
        "n_improved": n_improved,
        "n_total": len(test_samples),
        "mean_err_before": mean_before,
        "mean_err_after": mean_after,
        "reduction_pct": reduction,
    }


def train_and_eval(
    train_samples: list[QEMSample],
    test_samples: list[QEMSample],
    config: GNNQEMConfig,
    split_seed: int = 99,
    augment_factor: int = 5,
    shuffle_edges: bool = False,
) -> dict:
    """Standard train→eval pipeline for ablation variants."""
    aug = augment(train_samples, factor=augment_factor, seed=42)
    dataset = build_qem_dataset(aug)

    # Optionally shuffle edges (V2)
    if shuffle_edges:
        rng_e = np.random.default_rng(123)
        for g in dataset:
            g.edge_index.shape[1]
            perm = rng_e.permutation(g.edge_index.shape[1])
            # Shuffle target nodes (breaks topology structure)
            g.edge_index[1] = g.edge_index[1, perm]

    rng = np.random.default_rng(split_seed)
    indices = rng.permutation(len(dataset))
    n_train = int(0.8 * len(dataset))
    train_data = [dataset[i] for i in indices[:n_train]]
    val_data = [dataset[i] for i in indices[n_train:]]

    model = GNNQEMCorrector(config)
    train_gnn_qem(model, train_data, val_data, config)
    return evaluate_on_test(model, test_samples)


# ═══════════════════════════════════════════════════════════════════════════════
# V1: MLP Context-Only (no graph)
# ═══════════════════════════════════════════════════════════════════════════════


class MLPContextOnly(nn.Module):
    """MLP that uses only the context vector (h, n_2q, CES, E_noisy/N).
    No graph structure, no node features, no message passing."""

    def __init__(self, context_dim: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.conf = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, data):
        context = data.context
        delta_e = self.net(context)
        confidence = self.conf(context)
        return delta_e, confidence


def run_v1_mlp_ablation(train_samples, test_samples, config):
    """V1: Train MLP with context only, no graph."""
    logger.info("\n" + "=" * 60)
    logger.info("V1: MLP Context-Only Ablation")
    logger.info("=" * 60)

    aug = augment(train_samples, factor=5)
    dataset = build_qem_dataset(aug)

    rng = np.random.default_rng(99)
    indices = rng.permutation(len(dataset))
    n_train = int(0.8 * len(dataset))
    train_data = [dataset[i] for i in indices[:n_train]]
    val_data = [dataset[i] for i in indices[n_train:]]

    model = MLPContextOnly(context_dim=config.context_dim, hidden=config.hidden_dim)
    train_gnn_qem(model, train_data, val_data, config)
    result = evaluate_on_test(model, test_samples)
    logger.info(
        f"  MLP result: {result['improvement_rate']:.1f}% ({result['n_improved']}/{result['n_total']})"
    )
    logger.info(
        f"  Error: {result['mean_err_before']:.3f} → {result['mean_err_after']:.3f} ({result['reduction_pct']:+.1f}%)"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# V2: Shuffled Graph Edges
# ═══════════════════════════════════════════════════════════════════════════════


def run_v2_shuffled_edges(train_samples, test_samples, config):
    """V2: Train GNN with randomized edge structure."""
    logger.info("\n" + "=" * 60)
    logger.info("V2: Shuffled Graph Edges Ablation")
    logger.info("=" * 60)
    result = train_and_eval(train_samples, test_samples, config, shuffle_edges=True)
    logger.info(
        f"  Shuffled result: {result['improvement_rate']:.1f}% ({result['n_improved']}/{result['n_total']})"
    )
    logger.info(
        f"  Error: {result['mean_err_before']:.3f} → {result['mean_err_after']:.3f} ({result['reduction_pct']:+.1f}%)"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# V3: Multi-Seed Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════


def run_v3_multi_seed(train_samples, test_samples, config):
    """V3: Train GNN with 3 different random seeds."""
    logger.info("\n" + "=" * 60)
    logger.info("V3: Multi-Seed Reproducibility")
    logger.info("=" * 60)
    seeds = [77, 99, 123]
    rates = []
    for seed in seeds:
        result = train_and_eval(train_samples, test_samples, config, split_seed=seed)
        rates.append(result["improvement_rate"])
        logger.info(f"  Seed {seed}: {result['improvement_rate']:.1f}%")
    mean_rate = float(np.mean(rates))
    std_rate = float(np.std(rates))
    logger.info(f"  Mean: {mean_rate:.1f}% ± {std_rate:.1f}%")
    return {"rates": rates, "mean": mean_rate, "std": std_rate}


# ═══════════════════════════════════════════════════════════════════════════════
# V5: Linear Regression Baseline
# ═══════════════════════════════════════════════════════════════════════════════


def run_v5_linear_baseline(train_samples, test_samples):
    """V5: Linear regression on context features."""
    logger.info("\n" + "=" * 60)
    logger.info("V5: Linear Regression Baseline")
    logger.info("=" * 60)

    # Build feature matrix: [E_noisy, h, n_2q, CES, N]
    def build_X_y(samples):
        X = np.array(
            [[s.noisy_energy, s.h_value, s.n_2q_gates, s.ces, s.n_qubits] for s in samples]
        )
        y = np.array([s.exact_energy - s.noisy_energy for s in samples])  # Target: ΔE
        return X, y

    X_train, y_train = build_X_y(train_samples)
    X_test, y_test = build_X_y(test_samples)

    # Fit linear regression (with intercept)
    X_train_aug = np.column_stack([X_train, np.ones(len(X_train))])
    X_test_aug = np.column_stack([X_test, np.ones(len(X_test))])

    # Least squares: w = (X^T X)^-1 X^T y
    w, residuals, rank, sv = np.linalg.lstsq(X_train_aug, y_train, rcond=None)

    y_pred_train = X_train_aug @ w
    y_pred_test = X_test_aug @ w

    # R² on training data
    ss_res = np.sum((y_train - y_pred_train) ** 2)
    ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
    r2_train = 1 - ss_res / max(ss_tot, 1e-10)

    # Evaluate on test: E_corrected = E_noisy + ΔE_predicted
    n_improved = 0
    errs_before, errs_after = [], []
    for i, s in enumerate(test_samples):
        e_corrected = s.noisy_energy + y_pred_test[i]
        eb = abs(s.noisy_energy - s.exact_energy)
        ea = abs(e_corrected - s.exact_energy)
        errs_before.append(eb)
        errs_after.append(ea)
        if ea < eb:
            n_improved += 1

    rate = n_improved / max(len(test_samples), 1) * 100
    mean_before = float(np.mean(errs_before))
    mean_after = float(np.mean(errs_after))
    reduction = (1 - mean_after / max(mean_before, 1e-10)) * 100

    logger.info(f"  Linear R² (train): {r2_train:.4f}")
    logger.info(f"  Linear test: {rate:.1f}% ({n_improved}/{len(test_samples)})")
    logger.info(f"  Error: {mean_before:.3f} → {mean_after:.3f} ({reduction:+.1f}%)")
    return {
        "r2_train": r2_train,
        "improvement_rate": rate,
        "n_improved": n_improved,
        "n_total": len(test_samples),
        "mean_err_before": mean_before,
        "mean_err_after": mean_after,
        "reduction_pct": reduction,
        "weights": w.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    output_dir = Path("results/gnn_qem")
    t0 = time.time()

    # Load data
    train_path = output_dir / "training_data.json"
    test_path = output_dir / "test_data_heavy_hex.json"

    if not train_path.exists() or not test_path.exists():
        logger.error(
            "Missing data. Run run_gnn_qem_training.py and run_gnn_qem_cross_topology.py first."
        )
        sys.exit(1)

    train_samples = load_qem_samples(train_path)
    test_samples = load_qem_samples(test_path)
    logger.info(
        f"Data: {len(train_samples)} train (chain_1d+ladder N=6), {len(test_samples)} test (heavy_hex N=10)"
    )

    config = GNNQEMConfig(hidden_dim=64, n_layers=3, epochs=800, patience=80, lr=1e-3, dropout=0.1)

    # Load GNN baseline for reference
    model_path = output_dir / "model.pt"
    gnn_model, _, _ = load_qem_checkpoint(model_path)
    gnn_baseline = evaluate_on_test(gnn_model, test_samples)
    logger.info(
        f"\nGNN Baseline (full model): {gnn_baseline['improvement_rate']:.1f}% improvement, {gnn_baseline['reduction_pct']:+.1f}% reduction"
    )

    # Run ablations
    v1_result = run_v1_mlp_ablation(train_samples, test_samples, config)
    v2_result = run_v2_shuffled_edges(train_samples, test_samples, config)
    v3_result = run_v3_multi_seed(train_samples, test_samples, config)
    v5_result = run_v5_linear_baseline(train_samples, test_samples)

    # ── Verdict ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION SUITE VERDICT")
    logger.info("=" * 60)

    gnn_rate = gnn_baseline["improvement_rate"]
    mlp_rate = v1_result["improvement_rate"]
    shuf_rate = v2_result["improvement_rate"]
    linear_rate = v5_result["improvement_rate"]
    multi_std = v3_result["std"]

    v1_pass = (gnn_rate - mlp_rate) >= 20.0
    v2_pass = (gnn_rate - shuf_rate) >= 15.0
    v3_pass = multi_std < 15.0
    v5_pass = (gnn_rate - linear_rate) >= 20.0 or v5_result["r2_train"] < 0.8

    logger.info(f"  GNN (baseline):       {gnn_rate:.1f}%")
    logger.info(
        f"  V1 MLP context-only:  {mlp_rate:.1f}% | Δ={gnn_rate - mlp_rate:+.1f}% | {'✅ PASS' if v1_pass else '❌ FAIL'} (need Δ≥20%)"
    )
    logger.info(
        f"  V2 Shuffled edges:    {shuf_rate:.1f}% | Δ={gnn_rate - shuf_rate:+.1f}% | {'✅ PASS' if v2_pass else '❌ FAIL'} (need Δ≥15%)"
    )
    logger.info(
        f"  V3 Multi-seed std:    ±{multi_std:.1f}% | {'✅ PASS' if v3_pass else '❌ FAIL'} (need <15%)"
    )
    logger.info(
        f"  V5 Linear baseline:   {linear_rate:.1f}% (R²={v5_result['r2_train']:.3f}) | {'✅ PASS' if v5_pass else '❌ FAIL'} (need Δ≥20% OR R²<0.8)"
    )

    all_pass = v1_pass and v2_pass and v3_pass and v5_pass
    logger.info(f"\n  OVERALL: {'✅ CLAIM VALIDATED' if all_pass else '⚠️ CLAIM NEEDS REVISION'}")
    logger.info(f"  Time: {time.time() - t0:.1f}s")
    logger.info("=" * 60)

    # Save
    output = {
        "gnn_baseline": gnn_baseline,
        "v1_mlp_context_only": v1_result,
        "v2_shuffled_edges": v2_result,
        "v3_multi_seed": v3_result,
        "v5_linear_baseline": v5_result,
        "verdict": {
            "v1_pass": v1_pass,
            "v2_pass": v2_pass,
            "v3_pass": v3_pass,
            "v5_pass": v5_pass,
            "all_pass": all_pass,
            "gnn_advantage_over_mlp": gnn_rate - mlp_rate,
            "gnn_advantage_over_shuffled": gnn_rate - shuf_rate,
            "gnn_advantage_over_linear": gnn_rate - linear_rate,
            "reproducibility_std": multi_std,
        },
        "time_s": time.time() - t0,
    }
    out_path = output_dir / "ablation_suite_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
