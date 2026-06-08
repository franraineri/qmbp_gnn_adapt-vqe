#!/usr/bin/env python
"""GNN-QEM Ablation T1 — Remove E_noisy from context vector.

Tests: When the model cannot "cheat" by seeing E_noisy (which is ~linearly
related to the target ΔE), does the graph structure become essential?

Context vector variants:
  A. Full context: [h, n_2q/50, CES, E_noisy/N] — original (4 features)
  B. No-E_noisy:  [h, n_2q/50, CES, 0.0]       — E_noisy zeroed out (3 useful features)

Models compared:
  1. GNN + full context (baseline from ablation suite)
  2. GNN + no-E_noisy context
  3. MLP + no-E_noisy context
  4. Linear + no-E_noisy features

If GNN(no-E) >> MLP(no-E) → graph IS essential when E_noisy is unavailable
If GNN(no-E) ≈ MLP(no-E) → graph provides no advantage even without E_noisy

Usage:
    .venv/bin/python "scripts/experiment_runners/gnn experiments/run_gnn_qem_ablation_no_enoisy.py"
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
    build_qem_dataset,
    correct_energy,
    load_qem_samples,
    train_gnn_qem,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def augment(samples, factor=5, seed=42):
    rng = np.random.default_rng(seed)
    out = list(samples)
    for s in samples:
        err = abs(s.noisy_energy - s.exact_energy)
        for _ in range(factor - 1):
            delta = rng.normal(0, max(err * 0.2, 0.01))
            out.append(replace(s, noisy_energy=s.noisy_energy + delta))
    return out


def zero_out_enoisy_in_dataset(dataset):
    """Zero out the E_noisy/N component (index 3) from context vectors."""
    for g in dataset:
        if hasattr(g, "context") and g.context is not None:
            g.context[:, 3] = 0.0  # E_noisy/N is the 4th context feature
    return dataset


def evaluate_model(model, test_samples, threshold=0.0, zero_enoisy=False):
    """Evaluate model, optionally zeroing E_noisy in context during inference."""
    n_improved = 0
    errs_before, errs_after = [], []

    for s in test_samples:
        if zero_enoisy:
            # Create a modified sample with E_noisy info hidden
            s_mod = replace(s, noisy_energy=0.0)
            # But we still need the actual noisy_energy for error calc
            c = correct_energy(model, s_mod, confidence_threshold=threshold)
            # The correction is relative to s_mod.noisy_energy (0.0)
            # So corrected = 0.0 + delta_e_predicted
            e_corrected = c.corrected_energy
        else:
            c = correct_energy(model, s, confidence_threshold=threshold)
            e_corrected = c.corrected_energy

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
    return {
        "improvement_rate": rate,
        "n_improved": n_improved,
        "n_total": len(test_samples),
        "mean_err_before": mean_before,
        "mean_err_after": mean_after,
        "reduction_pct": reduction,
    }


class MLPContextOnly(nn.Module):
    """MLP baseline using only context features (no graph)."""

    def __init__(self, context_dim=4, hidden=64):
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
        return self.net(context), self.conf(context)


def train_variant(train_samples, config, split_seed=99, zero_enoisy=False, model_cls=None):
    """Train a model variant and return it."""
    aug = augment(train_samples, factor=5)
    dataset = build_qem_dataset(aug)
    if zero_enoisy:
        dataset = zero_out_enoisy_in_dataset(dataset)

    rng = np.random.default_rng(split_seed)
    indices = rng.permutation(len(dataset))
    n_train = int(0.8 * len(dataset))
    train_data = [dataset[i] for i in indices[:n_train]]
    val_data = [dataset[i] for i in indices[n_train:]]

    if model_cls is not None:
        model = model_cls(context_dim=config.context_dim, hidden=config.hidden_dim)
    else:
        model = GNNQEMCorrector(config)

    train_gnn_qem(model, train_data, val_data, config)
    return model


def linear_no_enoisy(train_samples, test_samples):
    """Linear regression WITHOUT E_noisy feature."""

    def build_X_y(samples):
        # Features: [h, n_2q, CES, N] — NO E_noisy
        X = np.array([[s.h_value, s.n_2q_gates, s.ces, s.n_qubits] for s in samples])
        y = np.array([s.exact_energy - s.noisy_energy for s in samples])
        return X, y

    X_train, y_train = build_X_y(train_samples)
    X_test, y_test = build_X_y(test_samples)

    X_train_aug = np.column_stack([X_train, np.ones(len(X_train))])
    X_test_aug = np.column_stack([X_test, np.ones(len(X_test))])

    w, _, _, _ = np.linalg.lstsq(X_train_aug, y_train, rcond=None)
    y_pred_test = X_test_aug @ w

    # R² on train
    y_pred_train = X_train_aug @ w
    ss_res = np.sum((y_train - y_pred_train) ** 2)
    ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)

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
    return {
        "r2_train": float(r2),
        "improvement_rate": rate,
        "n_improved": n_improved,
        "mean_err_after": float(np.mean(errs_after)),
        "reduction_pct": (1 - np.mean(errs_after) / max(np.mean(errs_before), 1e-10)) * 100,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    output_dir = Path("results/gnn_qem")
    t0 = time.time()

    train_samples = load_qem_samples(output_dir / "training_data.json")
    test_samples = load_qem_samples(output_dir / "test_data_heavy_hex.json")
    logger.info(f"Data: {len(train_samples)} train, {len(test_samples)} test")

    config = GNNQEMConfig(hidden_dim=64, n_layers=3, epochs=800, patience=80, lr=1e-3, dropout=0.1)

    # ── 1. GNN + full context (reference) ─────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("1. GNN + FULL context (baseline)")
    logger.info("=" * 60)
    gnn_full = train_variant(train_samples, config, zero_enoisy=False)
    r_gnn_full = evaluate_model(gnn_full, test_samples, zero_enoisy=False)
    logger.info(
        f"   Rate: {r_gnn_full['improvement_rate']:.1f}%, Reduction: {r_gnn_full['reduction_pct']:+.1f}%, MAE: {r_gnn_full['mean_err_after']:.3f}"
    )

    # ── 2. GNN + NO E_noisy context ──────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("2. GNN + NO E_noisy context")
    logger.info("=" * 60)
    gnn_no_e = train_variant(train_samples, config, zero_enoisy=True)
    r_gnn_no_e = evaluate_model(gnn_no_e, test_samples, zero_enoisy=True)
    logger.info(
        f"   Rate: {r_gnn_no_e['improvement_rate']:.1f}%, Reduction: {r_gnn_no_e['reduction_pct']:+.1f}%, MAE: {r_gnn_no_e['mean_err_after']:.3f}"
    )

    # ── 3. MLP + NO E_noisy context ──────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("3. MLP + NO E_noisy context")
    logger.info("=" * 60)
    mlp_no_e = train_variant(train_samples, config, zero_enoisy=True, model_cls=MLPContextOnly)
    r_mlp_no_e = evaluate_model(mlp_no_e, test_samples, zero_enoisy=True)
    logger.info(
        f"   Rate: {r_mlp_no_e['improvement_rate']:.1f}%, Reduction: {r_mlp_no_e['reduction_pct']:+.1f}%, MAE: {r_mlp_no_e['mean_err_after']:.3f}"
    )

    # ── 4. Linear + NO E_noisy ────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("4. Linear Regression (no E_noisy)")
    logger.info("=" * 60)
    r_linear = linear_no_enoisy(train_samples, test_samples)
    logger.info(
        f"   R²: {r_linear['r2_train']:.4f}, Rate: {r_linear['improvement_rate']:.1f}%, MAE: {r_linear['mean_err_after']:.3f}"
    )

    # ── Verdict ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("T1 ABLATION VERDICT")
    logger.info("=" * 60)

    gnn_no_e_rate = r_gnn_no_e["improvement_rate"]
    mlp_no_e_rate = r_mlp_no_e["improvement_rate"]
    linear_rate = r_linear["improvement_rate"]
    gnn_no_e_mae = r_gnn_no_e["mean_err_after"]
    mlp_no_e_mae = r_mlp_no_e["mean_err_after"]

    graph_advantage_rate = gnn_no_e_rate - mlp_no_e_rate
    graph_advantage_mae = mlp_no_e_mae - gnn_no_e_mae  # positive = GNN better

    logger.info(
        f"  GNN (full context):   {r_gnn_full['improvement_rate']:.1f}% | MAE={r_gnn_full['mean_err_after']:.3f}"
    )
    logger.info(f"  GNN (no E_noisy):     {gnn_no_e_rate:.1f}% | MAE={gnn_no_e_mae:.3f}")
    logger.info(f"  MLP (no E_noisy):     {mlp_no_e_rate:.1f}% | MAE={mlp_no_e_mae:.3f}")
    logger.info(f"  Linear (no E_noisy):  {linear_rate:.1f}% | R²={r_linear['r2_train']:.4f}")
    logger.info("")
    logger.info(f"  Graph advantage (rate): {graph_advantage_rate:+.1f}%")
    logger.info(f"  Graph advantage (MAE):  {graph_advantage_mae:+.3f} units")
    logger.info("")

    # Interpretation
    graph_essential = graph_advantage_rate >= 20.0 or (graph_advantage_mae > 2.0)
    if graph_essential:
        logger.info("  ✅ GRAPH IS ESSENTIAL when E_noisy removed")
        logger.info("     → Claim supported: GNN learns from topology structure")
    elif graph_advantage_rate > 0 or graph_advantage_mae > 0:
        logger.info("  🟡 GRAPH HELPS MARGINALLY (not essential)")
        logger.info("     → Claim needs nuance: graph provides regularization, not primary signal")
    else:
        logger.info("  ❌ GRAPH PROVIDES NO ADVANTAGE even without E_noisy")
        logger.info("     → Claim invalid: context features alone suffice")

    logger.info("=" * 60)

    # Save
    output = {
        "gnn_full_context": r_gnn_full,
        "gnn_no_enoisy": r_gnn_no_e,
        "mlp_no_enoisy": r_mlp_no_e,
        "linear_no_enoisy": r_linear,
        "verdict": {
            "graph_advantage_rate": graph_advantage_rate,
            "graph_advantage_mae": graph_advantage_mae,
            "graph_essential": graph_essential,
            "interpretation": (
                "essential"
                if graph_essential
                else "marginal"
                if (graph_advantage_rate > 0 or graph_advantage_mae > 0)
                else "none"
            ),
        },
        "time_s": time.time() - t0,
    }
    out_path = output_dir / "ablation_no_enoisy_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
