#!/usr/bin/env python3
"""MPS Scaling Phase 3 — MPNN Training at N=40/50.

Trains GINConv MPNN on θ_opt data from Phase 2 scaling runs,
then evaluates deployment accuracy (Phase 4 simulation).

Prerequisites:
    Run run_scaling_validation.py first to produce θ_opt data.

Usage:
    python scripts/experiment_runners/scaling/run_scaling_phase3_mpnn.py \\
        --result-file results/scaling/scaling_N40_aer_mps_*.json \\
        --output-dir results/scaling/phase3

Strategy:
    1. Load θ_opt from Phase 2 result JSON
    2. Canonicalize θ signs (enforce θ_x > 0) to handle Z₂ symmetry
    3. Build graph dataset from lattice + θ_opt
    4. Train MPNN (GINConv h=128, L=3, 6000 epochs)
    5. Deploy: predict θ at held-out h-values → evaluate energy → ΔE/gap
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import torch

from qmbp_simulation import (
    ClassicalSolver,
    HamiltonianBuilder,
    HVACircuitBuilder,
    make_lattice,
)
from qmbp_simulation.execution import MPSBackend
from qmbp_simulation.predictors import (
    MPNNPredictor,
    build_graph_dataset,
    train_mpnn,
)
from qmbp_simulation.utils.helpers import json_dump

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading & Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════


def canonicalize_theta(theta: np.ndarray) -> np.ndarray:
    """Enforce Z₂ sign convention: θ_x > 0.

    HVA p=1 has symmetry (θ_zz, θ_x) ↔ (-θ_zz, -θ_x) giving same energy.
    Different seeds may converge to opposite sign conventions.
    Canonicalize by ensuring the last parameter (θ_x) is positive.

    This is critical for MPNN training — inconsistent signs produce
    high MSE even when all θ are physically equivalent.

    Reference: binnacle-p1-scaling.md §Key Observations point 1.
    """
    if theta[-1] < 0:
        return -theta
    return theta


def load_theta_from_result(result_file: Path, seed: int | None = None) -> dict:
    """Load θ_opt, h_values, energies from a scaling result JSON.

    Parameters
    ----------
    result_file : Path
        Path to scaling_N*_*.json from run_scaling_validation.py
    seed : int | None
        Which seed's results to use. If None, uses first available seed.

    Returns
    -------
    dict with keys: h_values, theta_opt, e_dmrg, n_qubits, topology, p_layers
    """
    with open(result_file) as f:
        data = json.load(f)

    meta = data["metadata"]
    vqe_results = data["vqe_results"]

    # Select seed
    if seed is not None:
        seed_run = next((r for r in vqe_results if r["seed"] == seed), None)
        if seed_run is None:
            available = [r["seed"] for r in vqe_results]
            raise ValueError(f"Seed {seed} not found. Available: {available}")
    else:
        seed_run = vqe_results[0]

    results = seed_run["results"]

    # Check theta_opt presence
    if "theta_opt" not in results[0]:
        raise ValueError(
            f"theta_opt not found in {result_file}. "
            f"Re-run scaling validation with the updated runner that saves theta_opt."
        )

    h_values = np.array([r["h"] for r in results])
    theta_opt = np.array([r["theta_opt"] for r in results])
    e_dmrg = np.array([r["dmrg_energy"] for r in results])

    # Canonicalize signs
    for i in range(len(theta_opt)):
        theta_opt[i] = canonicalize_theta(theta_opt[i])

    return {
        "h_values": h_values,
        "theta_opt": theta_opt,
        "e_dmrg": e_dmrg,
        "n_qubits": meta["n"],
        "topology": meta["topology"],
        "p_layers": meta["p_layers"],
        "seed": seed_run["seed"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: MPNN Training
# ═══════════════════════════════════════════════════════════════════════════════


def train_scaling_mpnn(
    data: dict,
    hidden_dim: int = 128,
    n_layers: int = 3,
    n_epochs: int = 6000,
    patience: int = 500,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[MPNNPredictor, dict]:
    """Train MPNN on Phase 2 θ_opt data.

    Returns (trained_model, training_metrics).
    """
    n_qubits = data["n_qubits"]
    topology = data["topology"]
    p_layers = data["p_layers"]
    h_values = data["h_values"]
    theta_opt = data["theta_opt"]
    e_dmrg = data["e_dmrg"]

    # Build lattice (using first h-value for structure — only topology matters)
    lattice = make_lattice(topology, n_qubits, J=1.0, h=float(h_values[0]))

    # Build graph dataset (no fidelity filter — all points validated by ΔE/gap<5%)
    output_dim = theta_opt.shape[1]
    dataset = build_graph_dataset(
        lattice=lattice,
        h_values=h_values,
        theta_opt=theta_opt,
        e_exact=e_dmrg,
        fidelities=None,  # No fidelity available from MPS (not needed — all <1% ΔE/gap)
        fidelity_threshold=0.0,  # noqa: disabled — MPS data validated by ΔE/gap<1%, no fidelity available
    )

    logger.info(
        f"Training MPNN: {len(dataset)} points, output_dim={output_dim}, "
        f"hidden={hidden_dim}, layers={n_layers}"
    )

    # Create model
    model = MPNNPredictor(
        node_features=2,  # h_i + coordination_number
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        output_dim=output_dim,
    )

    # Train
    t0 = time.time()
    metrics = train_mpnn(
        model=model,
        dataset=dataset,
        n_epochs=n_epochs,
        lr=lr,
        patience=patience,
        seed=seed,
    )
    train_time = time.time() - t0

    metrics["train_time_s"] = train_time
    metrics["n_train_points"] = len(dataset)
    metrics["output_dim"] = output_dim

    logger.info(
        f"Training complete: final_mse={metrics['final_mse']:.2e}, "
        f"time={train_time:.1f}s, stopped_early={metrics['stopped_early']}"
    )

    return model, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 (Deployment Simulation): Predict θ → Evaluate energy → ΔE/gap
# ═══════════════════════════════════════════════════════════════════════════════


def deploy_mpnn(
    model: MPNNPredictor,
    data: dict,
    h_test: list[float],
    strategy: str = "aer_mps",
    chi_max: int = 64,
    precision: float = 0.005,
    seed: int = 42,
) -> list[dict]:
    """Deploy trained MPNN at held-out h-values.

    Predicts θ from the graph, then evaluates energy via MPSBackend.
    Compares with DMRG ground truth for ΔE/gap.
    """
    from torch_geometric.data import Data

    n_qubits = data["n_qubits"]
    topology = data["topology"]
    p_layers = data["p_layers"]

    builder = HamiltonianBuilder()
    hva = HVACircuitBuilder()
    solver = ClassicalSolver()

    backend = MPSBackend(strategy=strategy, chi_max=chi_max, precision=precision, seed=seed)

    results = []
    model.eval()

    for h_val in h_test:
        t0 = time.time()

        # Build graph for this h-value
        lattice = make_lattice(topology, n_qubits, J=1.0, h=h_val)
        H = builder.build(lattice)
        edge_index_np, coord = builder.build_graph_data(lattice)

        # Create Data object
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        h_feat = np.full(n_qubits, h_val)
        x = torch.tensor(
            np.stack([h_feat, coord.astype(float)], axis=1),
            dtype=torch.float32,
        )
        graph = Data(x=x, edge_index=edge_index, batch=torch.zeros(n_qubits, dtype=torch.long))

        # Predict θ
        with torch.no_grad():
            theta_pred = model(graph).numpy().flatten()

        # Canonicalize
        theta_pred = canonicalize_theta(theta_pred)

        # Evaluate energy
        circuit, _ = hva.create(n_qubits, p_layers, lattice)
        e_pred = backend.evaluate(circuit, H, theta_pred)

        # DMRG reference
        gt = solver.solve(H, lattice, method="dmrg")
        de_gap = abs(e_pred - gt.ground_energy) / max(gt.gap, 1e-10)
        elapsed = time.time() - t0

        results.append(
            {
                "h": h_val,
                "theta_pred": theta_pred.tolist(),
                "e_pred": e_pred,
                "e_dmrg": gt.ground_energy,
                "gap": gt.gap,
                "de_gap": de_gap,
                "passed": de_gap < 0.05,
                "time_s": elapsed,
            }
        )

        status = "✅" if de_gap < 0.05 else "❌"
        logger.info(
            f"  Deploy h={h_val:.3f}: ΔE/gap={de_gap:.4f} {status} "
            f"(θ=[{theta_pred[0]:.4f}, {theta_pred[1]:.4f}])"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI & Main
# ═══════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 3+4: MPNN Training + Deployment at N>30",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--result-file",
        type=str,
        required=True,
        help="Path to scaling result JSON (from run_scaling_validation.py)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for data selection")
    parser.add_argument(
        "--h-test",
        type=float,
        nargs="+",
        default=None,
        help="Test h-values for deployment. Auto-computed if not given.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128, help="MPNN hidden dim")
    parser.add_argument("--n-epochs", type=int, default=6000, help="Training epochs")
    parser.add_argument(
        "--strategy",
        type=str,
        default="aer_mps",
        choices=["aer_mps", "tenpy_exact"],
    )
    parser.add_argument("--output-dir", type=str, default="results/scaling/phase3")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    result_file = Path(args.result_file)
    if not result_file.exists():
        logger.error(f"Result file not found: {result_file}")
        return 1

    # Load data
    logger.info(f"Loading θ_opt from {result_file}")
    data = load_theta_from_result(result_file, seed=args.seed)
    logger.info(
        f"  N={data['n_qubits']}, {len(data['h_values'])} h-points, "
        f"θ shape={data['theta_opt'].shape}, seed={data['seed']}"
    )

    # Phase 3: Train MPNN
    logger.info("\n─── Phase 3: MPNN Training ───")
    model, train_metrics = train_scaling_mpnn(
        data,
        hidden_dim=args.hidden_dim,
        n_epochs=args.n_epochs,
        seed=args.seed,
    )

    # Phase 4: Deploy at test h-values
    if args.h_test is not None:
        h_test = sorted(args.h_test, reverse=True)
    else:
        # Use midpoints between training h-values (interpolation test)
        h_vals = data["h_values"]
        h_test = [(h_vals[i] + h_vals[i + 1]) / 2 for i in range(len(h_vals) - 1)]

    logger.info(f"\n─── Phase 4: Deployment (h_test={[f'{h:.2f}' for h in h_test]}) ───")
    deploy_results = deploy_mpnn(
        model,
        data,
        h_test,
        strategy=args.strategy,
        seed=args.seed,
    )

    # Summary
    n_pass = sum(1 for r in deploy_results if r["passed"])
    n_total = len(deploy_results)
    mean_de = np.mean([r["de_gap"] for r in deploy_results])

    logger.info("\n─── Summary ───")
    logger.info(
        f"  Training: MSE={train_metrics['final_mse']:.2e}, {train_metrics['n_train_points']} points"
    )
    logger.info(f"  Deploy: {n_pass}/{n_total} pass, mean ΔE/gap={mean_de:.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"phase3_N{data['n_qubits']}_{timestamp}.json"

    envelope = {
        "experiment": "mps_scaling_phase3",
        "metadata": {
            "n_qubits": data["n_qubits"],
            "topology": data["topology"],
            "p_layers": data["p_layers"],
            "source_file": str(result_file),
            "seed": data["seed"],
            "hidden_dim": args.hidden_dim,
            "n_epochs": args.n_epochs,
            "strategy": args.strategy,
        },
        "training": {
            "final_mse": train_metrics["final_mse"],
            "n_train_points": train_metrics["n_train_points"],
            "train_time_s": train_metrics["train_time_s"],
            "stopped_early": train_metrics["stopped_early"],
        },
        "deployment": {
            "h_test": h_test,
            "results": deploy_results,
            "n_pass": n_pass,
            "n_total": n_total,
            "mean_de_gap": float(mean_de),
            "all_passed": n_pass == n_total,
        },
    }

    json_dump(envelope, output_path)
    logger.info(f"  Results saved: {output_path}")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
