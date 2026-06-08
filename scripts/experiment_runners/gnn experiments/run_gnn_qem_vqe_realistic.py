#!/usr/bin/env python
"""GNN-QEM with VQE-Optimized θ — Realistic Error Regime + Circuit Selection Mode.

Two experiments in one script:

EXP 1: Train GNN-QEM on VQE-optimized data (errors ~0.5-3 units instead of 10-25).
  Tests whether the model still works when errors are realistic (post-VQE, pre-ZNE).

EXP 2: "Circuit Selection" predictive mode (no E_noisy).
  Given ONLY (h, topology_graph, calibration_data, n_2q_gates, CES), can the GNN
  predict HOW MUCH error a circuit will have? This enables:
  - Pre-execution feasibility checks ("will this config pass ΔE/gap<5%?")
  - Layout ranking ("which layout will have lowest error?")
  - Hardware resource allocation ("is ZNE needed for this h-point?")

Usage:
    .venv/bin/python "scripts/experiment_runners/gnn experiments/run_gnn_qem_vqe_realistic.py"
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qmbp_simulation import ClassicalSolver, HamiltonianBuilder, make_lattice
from qmbp_simulation.execution.noisy_utils import (
    NoisyEstimatorConfig,
    build_adjacency,
    find_layouts_bfs,
    noisy_estimate,
    select_layouts_low_ces,
)
from qmbp_simulation.models.model_registry import get_model_spec
from qmbp_simulation.predictors.gnn_qem import (
    GNNQEMConfig,
    GNNQEMCorrector,
    QEMSample,
    build_qem_dataset,
    correct_energy,
    load_qem_samples,
    save_qem_checkpoint,
    save_qem_samples,
    train_gnn_qem,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

TRAIN_TOPOLOGIES = ["chain_1d", "ladder"]
TEST_TOPOLOGY = "heavy_hex"
N_QUBITS_TRAIN = 6
N_QUBITS_TEST = 10
P_LAYERS = 1
TRAIN_H = [2.0, 2.5, 3.0, 3.5, 4.0]
TEST_H = [3.0, 3.25, 3.5, 4.0]
SEEDS = [42, 43, 44]
SHOTS = 4096


# ═══════════════════════════════════════════════════════════════════════════════
# Data Generation with VQE-optimized θ
# ═══════════════════════════════════════════════════════════════════════════════


def generate_vqe_data(topologies, n_qubits, h_values, seeds, shots):
    """Generate QEMSamples using VQE-optimized parameters (realistic errors)."""
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    from qmbp_simulation import VQEOptimizer

    fake_backend = FakeTorino()
    noisy_config = NoisyEstimatorConfig(shots=shots, seed_simulator=42)
    adj = build_adjacency(fake_backend)
    candidates = find_layouts_bfs(adj, n_qubits, n_candidates=20)

    spec = get_model_spec("tfim")
    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    optimizer = VQEOptimizer()

    samples = []
    for topo in topologies:
        for seed in seeds:
            prev_theta = None
            for h in sorted(h_values, reverse=True):
                lattice = make_lattice(topo, n_qubits, J=1.0, h=h)
                H = builder.build(lattice)
                circuit, _ = spec.create_circuit(n_qubits, P_LAYERS, lattice)
                gt = solver.solve(H, lattice)
                e_exact = gt.ground_energy
                gap = gt.gap

                # VQE warm-start
                if prev_theta is not None:
                    init = prev_theta
                else:
                    rng = np.random.default_rng(seed)
                    init = rng.uniform(-0.01, 0.01, size=circuit.num_parameters)

                vqe_result = optimizer.optimize(H, circuit, init)
                theta_opt = vqe_result.theta_opt
                prev_theta = theta_opt

                # Transpile and measure noisy energy
                bound = circuit.assign_parameters(theta_opt)
                layout_sel = select_layouts_low_ces(
                    bound,
                    fake_backend,
                    candidates,
                    n_select=1,
                    optimization_level=2,
                    max_ces=0.5,
                )
                transpiled = layout_sel.transpiled_circuits[0]
                H_mapped = H.apply_layout(transpiled.layout)
                ces = layout_sel.ces_values[0] if layout_sel.ces_values else 0.15

                e_noisy = noisy_estimate(
                    transpiled,
                    H_mapped,
                    fake_backend,
                    noisy_config,
                    seed_offset=seed,
                )
                n_2q = sum(1 for inst in transpiled.data if inst.operation.num_qubits == 2)

                sample = QEMSample(
                    noisy_energy=e_noisy,
                    exact_energy=e_exact,
                    h_value=h,
                    n_2q_gates=n_2q,
                    ces=ces,
                    topology=topo,
                    n_qubits=n_qubits,
                )
                samples.append(sample)

                err = abs(e_noisy - e_exact)
                logger.info(
                    f"  {topo} seed={seed} h={h:.1f}: err={err:.3f} (ΔE/gap={err / max(gap, 1e-10):.3f})"
                )

    return samples


def augment(samples, factor=5, seed=42):
    rng = np.random.default_rng(seed)
    out = list(samples)
    for s in samples:
        err = abs(s.noisy_energy - s.exact_energy)
        for _ in range(factor - 1):
            delta = rng.normal(0, max(err * 0.15, 0.01))
            out.append(replace(s, noisy_energy=s.noisy_energy + delta))
    return out


def zero_enoisy_in_dataset(dataset):
    """Zero E_noisy/N in context (index 3) for predictive mode."""
    for g in dataset:
        if hasattr(g, "context") and g.context is not None:
            g.context[:, 3] = 0.0
    return dataset


def evaluate(model, test_samples, zero_enoisy=False, threshold=0.0):
    """Evaluate model on test samples."""
    n_improved = 0
    errs_before, errs_after, confs = [], [], []
    for s in test_samples:
        if zero_enoisy:
            s_in = replace(s, noisy_energy=0.0)
        else:
            s_in = s
        c = correct_energy(model, s_in, confidence_threshold=threshold)
        eb = abs(s.noisy_energy - s.exact_energy)
        ea = abs(c.corrected_energy - s.exact_energy)
        errs_before.append(eb)
        errs_after.append(ea)
        confs.append(c.confidence)
        if ea < eb:
            n_improved += 1
    rate = n_improved / max(len(test_samples), 1) * 100
    return {
        "rate": rate,
        "n_improved": n_improved,
        "n_total": len(test_samples),
        "mae_before": float(np.mean(errs_before)),
        "mae_after": float(np.mean(errs_after)),
        "reduction_pct": (1 - np.mean(errs_after) / max(np.mean(errs_before), 1e-10)) * 100,
        "mean_confidence": float(np.mean(confs)),
        "median_err_before": float(np.median(errs_before)),
        "median_err_after": float(np.median(errs_after)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    output_dir = Path("results/gnn_qem")
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── Step 1: Generate VQE-optimized training data ──────────────────
    train_path = output_dir / "vqe_train_data.json"
    if train_path.exists():
        logger.info(f"Loading VQE training data from {train_path}")
        train_samples = load_qem_samples(train_path)
    else:
        logger.info("Generating VQE-optimized TRAINING data...")
        train_samples = generate_vqe_data(
            TRAIN_TOPOLOGIES,
            N_QUBITS_TRAIN,
            TRAIN_H,
            SEEDS,
            SHOTS,
        )
        save_qem_samples(train_samples, train_path)

    train_errs = [abs(s.noisy_energy - s.exact_energy) for s in train_samples]
    logger.info(
        f"Train: {len(train_samples)} samples, err range [{min(train_errs):.3f}, {max(train_errs):.3f}], mean={np.mean(train_errs):.3f}"
    )

    # ── Step 2: Generate VQE-optimized TEST data (heavy_hex N=10) ─────
    test_path = output_dir / "vqe_test_data_heavy_hex.json"
    if test_path.exists():
        logger.info(f"Loading VQE test data from {test_path}")
        test_samples = load_qem_samples(test_path)
    else:
        logger.info("Generating VQE-optimized TEST data (heavy_hex N=10)...")
        test_samples = generate_vqe_data(
            [TEST_TOPOLOGY],
            N_QUBITS_TEST,
            TEST_H,
            SEEDS,
            SHOTS,
        )
        save_qem_samples(test_samples, test_path)

    test_errs = [abs(s.noisy_energy - s.exact_energy) for s in test_samples]
    logger.info(
        f"Test: {len(test_samples)} samples, err range [{min(test_errs):.3f}, {max(test_errs):.3f}], mean={np.mean(test_errs):.3f}"
    )

    t_data = time.time() - t0

    # ── Step 3: EXP 1 — Train with full context (correction mode) ─────
    logger.info("\n" + "=" * 60)
    logger.info("EXP 1: GNN-QEM with VQE-realistic errors (full context)")
    logger.info("=" * 60)

    aug_train = augment(train_samples, factor=5)
    dataset_full = build_qem_dataset(aug_train)

    rng = np.random.default_rng(99)
    idx = rng.permutation(len(dataset_full))
    n_tr = int(0.8 * len(dataset_full))
    train_data = [dataset_full[i] for i in idx[:n_tr]]
    val_data = [dataset_full[i] for i in idx[n_tr:]]

    config = GNNQEMConfig(
        hidden_dim=64, n_layers=3, epochs=1000, patience=100, lr=1e-3, dropout=0.1
    )
    model_full = GNNQEMCorrector(config)
    result_full = train_gnn_qem(model_full, train_data, val_data, config)
    logger.info(f"  Training: epoch={result_full.best_epoch}, val_MAE={result_full.val_mae:.4f}")

    eval_full = evaluate(model_full, test_samples, zero_enoisy=False)
    logger.info(
        f"  Cross-topo (full ctx): rate={eval_full['rate']:.1f}%, MAE={eval_full['mae_before']:.3f}→{eval_full['mae_after']:.3f} ({eval_full['reduction_pct']:+.1f}%)"
    )

    save_qem_checkpoint(
        model_full,
        output_dir / "model_vqe_realistic.pt",
        result_full,
        metadata={
            "mode": "vqe_realistic",
            "train_topologies": TRAIN_TOPOLOGIES,
        },
    )

    # ── Step 4: EXP 2 — Train WITHOUT E_noisy (circuit selection mode) ─
    logger.info("\n" + "=" * 60)
    logger.info("EXP 2: Circuit Selection Mode (no E_noisy)")
    logger.info("=" * 60)

    dataset_no_e = build_qem_dataset(aug_train)
    dataset_no_e = zero_enoisy_in_dataset(dataset_no_e)

    idx2 = rng.permutation(len(dataset_no_e))
    train_no_e = [dataset_no_e[i] for i in idx2[:n_tr]]
    val_no_e = [dataset_no_e[i] for i in idx2[n_tr:]]

    model_pred = GNNQEMCorrector(config)
    result_pred = train_gnn_qem(model_pred, train_no_e, val_no_e, config)
    logger.info(f"  Training: epoch={result_pred.best_epoch}, val_MAE={result_pred.val_mae:.4f}")

    eval_pred = evaluate(model_pred, test_samples, zero_enoisy=True)
    logger.info(
        f"  Cross-topo (no E_noisy): rate={eval_pred['rate']:.1f}%, MAE={eval_pred['mae_before']:.3f}→{eval_pred['mae_after']:.3f} ({eval_pred['reduction_pct']:+.1f}%)"
    )

    save_qem_checkpoint(
        model_pred,
        output_dir / "model_circuit_selection.pt",
        result_pred,
        metadata={
            "mode": "circuit_selection_predictive",
            "train_topologies": TRAIN_TOPOLOGIES,
        },
    )

    # ── Step 5: Circuit Selection Application ─────────────────────────
    # Use the predictive model to rank h-points by predicted error
    logger.info("\n" + "=" * 60)
    logger.info("CIRCUIT SELECTION APPLICATION")
    logger.info("=" * 60)
    logger.info("  Can the model predict which h-points will have low error?")

    predicted_errors = []
    actual_errors = []
    for s in test_samples:
        s_in = replace(s, noisy_energy=0.0)
        c = correct_energy(model_pred, s_in, confidence_threshold=0.0)
        # The model predicts ΔE (correction). Without E_noisy, this IS the predicted error magnitude
        predicted_err = abs(c.delta_e_predicted)
        actual_err = abs(s.noisy_energy - s.exact_energy)
        predicted_errors.append(predicted_err)
        actual_errors.append(actual_err)

    # Rank correlation: does the model correctly rank h-points by error?
    from scipy import stats

    rho, p_val = stats.spearmanr(predicted_errors, actual_errors)

    # Binary classification: predict "will error be > median?"
    median_err = np.median(actual_errors)
    pred_high = np.array(predicted_errors) > np.median(predicted_errors)
    actual_high = np.array(actual_errors) > median_err
    accuracy = np.mean(pred_high == actual_high) * 100

    logger.info(f"  Spearman rank correlation: ρ={rho:.3f} (p={p_val:.4f})")
    logger.info(f"  Binary classification (high/low error): {accuracy:.1f}% accuracy")
    logger.info(
        f"  Practical: model can {'✅ rank' if rho > 0.5 else '❌ NOT rank'} circuits by expected error"
    )

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(
        f"  Error regime: {np.mean(train_errs):.2f} units (VQE-realistic) vs 23 units (random θ)"
    )
    logger.info(
        f"  EXP 1 (correction): {eval_full['rate']:.1f}% improvement, {eval_full['reduction_pct']:+.1f}% reduction"
    )
    logger.info(
        f"  EXP 2 (predictive): {eval_pred['rate']:.1f}% improvement, {eval_pred['reduction_pct']:+.1f}% reduction"
    )
    logger.info(f"  Circuit selection: ρ={rho:.3f}, accuracy={accuracy:.1f}%")
    logger.info(f"  Time: data={t_data:.1f}s, total={time.time() - t0:.1f}s")
    logger.info("=" * 60)

    # Save results
    output = {
        "data_stats": {
            "n_train": len(train_samples),
            "n_test": len(test_samples),
            "train_err_mean": float(np.mean(train_errs)),
            "train_err_range": [float(min(train_errs)), float(max(train_errs))],
            "test_err_mean": float(np.mean(test_errs)),
            "test_err_range": [float(min(test_errs)), float(max(test_errs))],
        },
        "exp1_correction_mode": eval_full,
        "exp2_predictive_mode": eval_pred,
        "circuit_selection": {
            "spearman_rho": float(rho),
            "spearman_p": float(p_val),
            "binary_accuracy_pct": float(accuracy),
            "median_actual_error": float(median_err),
        },
        "time_s": time.time() - t0,
    }
    out_path = output_dir / "vqe_realistic_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
