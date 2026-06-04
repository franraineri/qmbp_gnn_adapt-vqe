#!/usr/bin/env python3
"""E4c Full Pipeline: Phases 1→4 for Frustrated TFIM with MPNN.

Demonstrates the MPNN can learn θ(h, J₂) using the extra_node_features
parameter, then predict circuit parameters at unseen (h, J₂) points.

Pipeline:
  Phase 1: Exact diag across h-sweep at fixed J₂=0.3
  Phase 2: VQE descending sweep → θ_opt dataset
  Phase 3: MPNN training with node_features=[h, coord, J₂]
  Phase 4: Deploy at unseen h_test, verify ΔE/gap < 5%
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("e4c_pipeline")


def main():
    from qmbp_simulation import HamiltonianBuilder, make_lattice
    from qmbp_simulation.circuits import HVACircuitBuilder
    from qmbp_simulation.execution import NoiselessBackend
    from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn

    t_start = time.time()

    # ── Configuration ────────────────────────────────────────────────
    N, p, J2 = 6, 2, 0.3
    topology = "chain_1d"
    h_train = np.arange(0.75, 2.55, 0.125)  # 15 training points (dense grid for better MPNN)
    h_test = [1.6, 1.35]  # Unseen interpolation points
    seed = 42
    n_restarts = 5

    logger.info("=" * 65)
    logger.info("E4c FULL PIPELINE: Frustrated TFIM (J₂=%.1f)", J2)
    logger.info("=" * 65)
    logger.info(f"  N={N}, p={p}, topology={topology}")
    logger.info(f"  h_train: {h_train.tolist()}")
    logger.info(f"  h_test: {h_test}")
    logger.info("  Node features: [h, coord, J₂] (3 features)")
    logger.info("")

    builder = HamiltonianBuilder()
    hva = HVACircuitBuilder()
    backend = NoiselessBackend()

    lattice_ref = make_lattice(topology, N, J=1.0, h=2.0)
    circuit, _ = hva.create_frustrated_tfim(N, p, lattice_ref)
    n_params = circuit.num_parameters

    # ── Phase 1: Exact Diag ──────────────────────────────────────────
    logger.info("--- Phase 1: Exact Diagonalization ---")
    e_exact = np.zeros(len(h_train))
    gaps = np.zeros(len(h_train))

    for i, h in enumerate(h_train):
        lat_h = make_lattice(topology, N, J=1.0, h=h)
        H = builder.build_frustrated_tfim(lat_h, J2=J2)
        mat = H.to_matrix()
        if hasattr(mat, "toarray"):
            mat = mat.toarray()
        evals = np.sort(np.linalg.eigvalsh(mat))
        e_exact[i] = evals[0]
        gaps[i] = evals[1] - evals[0]

    logger.info(
        f"  {len(h_train)} points computed. Gap range: [{gaps.min():.3f}, {gaps.max():.3f}]"
    )

    # ── Phase 2: VQE Sweep ───────────────────────────────────────────
    logger.info("\n--- Phase 2: VQE Descending Sweep ---")
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    theta_opt = np.zeros((len(h_train), n_params))
    fidelities = np.zeros(len(h_train))
    prev_theta = rng.uniform(-0.01, 0.01, n_params)

    for _i, h in enumerate(sorted(h_train, reverse=True)):
        idx = np.where(np.isclose(h_train, h))[0][0]
        lat_h = make_lattice(topology, N, J=1.0, h=h)
        H = builder.build_frustrated_tfim(lat_h, J2=J2)

        best_energy = float("inf")
        best_theta = prev_theta.copy()
        for r in range(n_restarts):
            x0 = prev_theta + rng.normal(0, 0.1, n_params) if r > 0 else prev_theta.copy()
            x0 = np.clip(x0, -np.pi, np.pi)
            res = minimize(
                lambda params, _H=H: backend.evaluate(circuit, _H, params),
                x0,
                method="L-BFGS-B",
                bounds=[(-np.pi, np.pi)] * n_params,
                options={"maxiter": 500, "ftol": 1e-14},
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_theta = res.x.copy()

        theta_opt[idx] = best_theta.copy()
        prev_theta = best_theta.copy()

        # Compute fidelity
        from qiskit.quantum_info import Statevector, state_fidelity

        mat = H.to_matrix()
        if hasattr(mat, "toarray"):
            mat = mat.toarray()
        _, evecs = np.linalg.eigh(mat)
        gs = evecs[:, 0]
        sv = Statevector(circuit.assign_parameters(best_theta))
        fidelities[idx] = float(state_fidelity(sv, Statevector(gs)))

    mean_fid = np.mean(fidelities)
    logger.info(f"  Mean fidelity: {mean_fid:.4f} (min: {np.min(fidelities):.4f})")
    logger.info(f"  All pass fid≥0.93: {np.all(fidelities >= 0.93)}")

    # ── Phase 3: MPNN Training with J₂ feature ──────────────────────
    logger.info("\n--- Phase 3: MPNN Training (node_features=3: h, coord, J₂) ---")

    # Build dataset with J₂ as extra node feature
    j2_array = np.full((len(h_train), 1), J2)

    dataset = build_graph_dataset(
        lattice=lattice_ref,
        h_values=h_train,
        theta_opt=theta_opt,
        e_exact=e_exact,
        fidelities=fidelities,
        fidelity_threshold=0.90,  # noqa — frustrated TFIM uses relaxed threshold (validated in E4c)
        extra_node_features=j2_array,
    )

    logger.info(f"  Dataset: {len(dataset)} graphs, node_features=3")

    model = MPNNPredictor(
        node_features=3,  # [h, coord, J₂]
        hidden_dim=64,
        n_layers=3,
        output_dim=n_params,
    )

    train_result = train_mpnn(
        model=model,
        dataset=dataset,
        n_epochs=6000,
        lr=1e-3,
        patience=500,
        seed=seed,
    )

    logger.info(f"  Final MSE: {train_result['final_mse']:.2e}")
    logger.info(f"  Stopped early: {train_result.get('stopped_early', False)}")
    logger.info(f"  Epochs trained: {len(train_result.get('mse_history', []))}")

    # ── Phase 4: Deployment at unseen h_test ─────────────────────────
    logger.info("\n--- Phase 4: Deployment at unseen h_test ---")

    import torch
    from torch_geometric.data import Data

    edge_index_np, coord = builder.build_graph_data(lattice_ref)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    model.eval()
    results_phase4 = []

    logger.info(
        f"  {'h_test':>6} | {'Pred Energy':>11} | {'E_exact':>8} | {'ΔE/gap':>7} | {'Pass':>4}"
    )
    logger.info(f"  {'-' * 6}-+-{'-' * 11}-+-{'-' * 8}-+-{'-' * 7}-+-{'-' * 4}")

    for h_t in h_test:
        # Build graph for prediction
        h_feat = np.full(N, float(h_t))
        j2_feat = np.full(N, float(J2))
        x = torch.tensor(
            np.stack([h_feat, coord.astype(float), j2_feat], axis=1),
            dtype=torch.float32,
        )
        graph = Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            theta_pred = model(graph).numpy().flatten()

        # Evaluate predicted circuit
        lat_test = make_lattice(topology, N, J=1.0, h=h_t)
        H_test = builder.build_frustrated_tfim(lat_test, J2=J2)
        pred_energy = backend.evaluate(circuit, H_test, theta_pred)

        # Exact reference
        mat_test = H_test.to_matrix()
        if hasattr(mat_test, "toarray"):
            mat_test = mat_test.toarray()
        evals_test = np.sort(np.linalg.eigvalsh(mat_test))
        e_test_exact = evals_test[0]
        gap_test = evals_test[1] - evals_test[0]

        de_gap = abs(pred_energy - e_test_exact) / max(gap_test, 1e-10)
        passed = "✓" if de_gap < 0.05 else "✗"

        logger.info(
            f"  {h_t:>6.2f} | {pred_energy:>11.4f} | {e_test_exact:>8.4f} | {de_gap:>7.4f} | {passed}"
        )
        results_phase4.append(
            {
                "h_test": h_t,
                "predicted_energy": float(pred_energy),
                "exact_energy": float(e_test_exact),
                "delta_e_over_gap": float(de_gap),
                "gap": float(gap_test),
                "passed": bool(de_gap < 0.05),
            }
        )

    # ── Save Results ─────────────────────────────────────────────────
    elapsed = time.time() - t_start

    output_dir = Path("results/experiments/exp_e4c_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"run_{timestamp}.json"

    payload = {
        "config": {
            "experiment_id": "E4c_pipeline",
            "category": "E",
            "model": "tfim_frustrated",
            "system": {"n_qubits": N, "p_layers": p, "topology": topology},
            "J2": J2,
            "h_train": h_train.tolist(),
            "h_test": h_test,
            "seeds": [seed],
            "description": "Full 4-phase pipeline for frustrated TFIM with MPNN (J₂ as node feature)",
            "hypothesis": "MPNN with node_features=[h, coord, J₂] predicts θ at unseen h with ΔE/gap<5%",
        },
        "analysis": {
            "summary": {
                "mean_de_gap": float(np.mean([r["delta_e_over_gap"] for r in results_phase4])),
                "pass_rate": float(np.mean([r["passed"] for r in results_phase4])),
                "total_time_s": elapsed,
            },
            "phase2_mean_fidelity": float(mean_fid),
            "phase3_final_mse": float(train_result["final_mse"]),
            "phase4_results": results_phase4,
        },
        "results": {
            str(seed): [
                {"h_test": r["h_test"], "de_gap": r["delta_e_over_gap"]} for r in results_phase4
            ]
        },
        "elapsed_s": elapsed,
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"\n  Results saved: {output_path}")
    logger.info(f"  Total time: {elapsed:.1f}s")

    # Summary
    n_pass = sum(1 for r in results_phase4 if r["passed"])
    logger.info(f"\n  PIPELINE RESULT: {n_pass}/{len(results_phase4)} test points pass ΔE/gap<5%")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
