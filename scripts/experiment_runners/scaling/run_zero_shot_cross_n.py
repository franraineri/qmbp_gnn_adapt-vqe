#!/usr/bin/env python3
"""A1: Zero-shot Cross-N GNN Generalization (with system-size encoding).

Tests whether a GNN trained on N_source data can predict θ_opt at N_target
WITHOUT retraining. This is the key thesis claim: the GNN learns PHYSICS
(the h→θ mapping) not just data interpolation.

The test works because:
- GINConv + global_mean_pool is SIZE-AGNOSTIC (accepts any N)
- Node features include (h, coordination_number, N/100) — the N/100 feature
  encodes system size so the GNN "knows" how large the system is and can
  generalize across sizes via the learned size→θ relationship.
- For chain_1d, all nodes are identical → the pooled embedding captures h and N.

A1 fix (2026-06-08): Added N/100 as third node feature. Without this, the GNN
has no way to distinguish N=10 from N=80 (same node features, just different
graph sizes — but global_mean_pool averages out the size information). With
N/100, the model learns the scaling law h_min(N) implicitly.

Multi-source training: When multiple --source-file paths are provided (different
N values), all are combined into a single training set. This gives the GNN
exposure to the N-dependent structure and enables proper cross-N generalization.

Strategy:
1. Load Phase 2 results from one or more source files (potentially different N)
2. Build graph dataset with 3 features: (h, coord, N/100) per node
3. Train MPNN on combined multi-N dataset
4. Build graph for N_target (unseen size, same feature semantics)
5. Predict θ at N_target h-values
6. Evaluate predicted θ against DMRG ground truth at N_target
7. Report ΔE/gap — if <5%, zero-shot generalization works

Usage:
    # Single source (same as before, now with N/100 feature)
    python scripts/experiment_runners/scaling/run_zero_shot_cross_n.py \\
        --source-file results/scaling/scaling_N40_aer_mps_20260608_001053.json \\
        --target-n 60 --target-h-values 6.0 5.5 5.0 4.5

    # Multi-source (recommended: combines N=40 + N=50 + N=80 data)
    python scripts/experiment_runners/scaling/run_zero_shot_cross_n.py \\
        --source-file results/scaling/scaling_N40_aer_mps_20260608_001053.json \\
                      results/scaling/scaling_N50_aer_mps_20260607_172041.json \\
                      results/scaling/scaling_N80_aer_mps_20260607_211634.json \\
        --target-n 60

    # Ablation mode (disable N/100 to confirm it matters)
    python scripts/experiment_runners/scaling/run_zero_shot_cross_n.py \\
        --source-file results/scaling/scaling_N40_aer_mps_20260608_001053.json \\
        --target-n 60 --no-n-feature
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
from torch_geometric.data import Data

from qmbp_simulation import (
    ClassicalSolver,
    HamiltonianBuilder,
    HVACircuitBuilder,
    make_lattice,
)
from qmbp_simulation.execution import MPSBackend
from qmbp_simulation.predictors import MPNNPredictor, train_mpnn
from qmbp_simulation.utils.helpers import json_dump

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def canonicalize_theta(theta: np.ndarray) -> np.ndarray:
    """Enforce θ_x > 0 sign convention (breaks Z2 symmetry consistently)."""
    if len(theta) == 0:
        return theta
    if theta[-1] < 0:
        return -theta
    return theta


def _build_graph(
    n: int,
    h_val: float,
    topology: str,
    use_n_feature: bool,
    *,
    edge_index: torch.Tensor | None = None,
    coord: np.ndarray | None = None,
) -> Data:
    """Build a single torch_geometric Data object with node features.

    Features per node:
    - h (transverse field, broadcast to all sites)
    - coordination_number (from lattice structure)
    - N/100 (system size encoding, if use_n_feature=True)

    Parameters
    ----------
    edge_index, coord : optional
        Pre-computed graph structure. If not provided, builds from scratch.
        Pass these when calling in a loop to avoid redundant computation
        (topology is h-independent).
    """
    if edge_index is None or coord is None:
        builder = HamiltonianBuilder()
        lattice = make_lattice(topology, n, J=1.0, h=h_val)
        edge_index_np, coord = builder.build_graph_data(lattice)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    h_feat = np.full(n, h_val)
    feature_cols = [h_feat, coord.astype(float)]

    if use_n_feature:
        n_feat = np.full(n, n / 100.0)
        feature_cols.append(n_feat)

    x = torch.tensor(np.stack(feature_cols, axis=1), dtype=torch.float32)
    graph = Data(x=x, edge_index=edge_index, batch=torch.zeros(n, dtype=torch.long))
    return graph


def _load_source_data(
    source_path: Path,
    source_seed: int,
) -> tuple[int, str, np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate a single source file.

    Returns (n, topology, h_values, theta_opt, e_dmrg).

    Raises
    ------
    FileNotFoundError
        If source file does not exist.
    ValueError
        If source file lacks theta_opt or has incompatible format.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    with open(source_path) as f:
        data = json.load(f)

    meta = data["metadata"]
    n = meta["n"]
    topology = meta["topology"]

    # Find seed run
    seed_runs = [r for r in data["vqe_results"] if r["seed"] == source_seed]
    if not seed_runs:
        available_seeds = [r["seed"] for r in data["vqe_results"]]
        raise ValueError(
            f"Seed {source_seed} not found in {source_path}. Available seeds: {available_seeds}"
        )

    results = seed_runs[0]["results"]
    if "theta_opt" not in results[0]:
        raise ValueError(
            f"Source file {source_path} has no theta_opt in results. "
            f"Re-run scaling validation with updated runner."
        )

    h_values = np.array([r["h"] for r in results])
    theta_opt = np.array([canonicalize_theta(np.array(r["theta_opt"])) for r in results])
    e_dmrg = np.array([r["dmrg_energy"] for r in results])

    return n, topology, h_values, theta_opt, e_dmrg


def _build_training_dataset(
    sources: list[tuple[int, str, np.ndarray, np.ndarray, np.ndarray]],
    use_n_feature: bool,
) -> list[Data]:
    """Build combined training dataset from multiple source files.

    Each source is (n, topology, h_values, theta_opt, e_dmrg).
    All sources must have the same topology and theta dimensionality.
    """
    builder = HamiltonianBuilder()
    dataset: list[Data] = []

    # Validate consistency across sources
    topologies = set(s[1] for s in sources)
    if len(topologies) > 1:
        raise ValueError(f"All source files must have same topology. Got: {topologies}")
    theta_dims = set(s[3].shape[1] for s in sources)
    if len(theta_dims) > 1:
        raise ValueError(f"All source files must have same theta dimensionality. Got: {theta_dims}")

    for n, topology, h_values, theta_opt, e_dmrg in sources:
        lattice_ref = make_lattice(topology, n, J=1.0, h=float(h_values[0]))
        edge_index_np, coord = builder.build_graph_data(lattice_ref)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)

        for i, h in enumerate(h_values):
            h_feat = np.full(n, float(h))
            feature_cols = [h_feat, coord.astype(float)]
            if use_n_feature:
                n_feat = np.full(n, n / 100.0)
                feature_cols.append(n_feat)

            x = torch.tensor(np.stack(feature_cols, axis=1), dtype=torch.float32)
            y = torch.tensor(theta_opt[i], dtype=torch.float32)

            data = Data(x=x, edge_index=edge_index, y=y)
            data.e_exact = float(e_dmrg[i])
            data.h_value = float(h)
            data.n_qubits = n
            dataset.append(data)

    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="A1: Zero-shot GNN cross-N generalization (with N/100 encoding)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-file",
        type=str,
        required=True,
        nargs="+",
        help="Phase 2 result JSON(s) for training. Multiple files with "
        "different N enable multi-N training (recommended).",
    )
    parser.add_argument(
        "--source-seed", type=int, default=42, help="Seed to extract from source files"
    )
    parser.add_argument(
        "--target-n", type=int, default=60, help="Target system size for zero-shot prediction"
    )
    parser.add_argument(
        "--target-h-values",
        type=float,
        nargs="+",
        default=None,
        help="h-values to test at N_target (auto if not given)",
    )
    parser.add_argument("--hidden-dim", type=int, default=128, help="MPNN hidden dimension")
    parser.add_argument("--n-epochs", type=int, default=6000, help="Training epochs")
    parser.add_argument(
        "--strategy",
        type=str,
        default="aer_mps",
        choices=["aer_mps", "tenpy_exact"],
        help="MPS backend strategy for evaluation",
    )
    parser.add_argument(
        "--precision", type=float, default=0.005, help="MPS precision (controls shot budget)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/scaling/zero_shot",
        help="Output directory for results JSON",
    )
    parser.add_argument(
        "--no-n-feature",
        action="store_true",
        help="Disable N/100 system-size encoding (ablation mode)",
    )
    parser.add_argument(
        "--norm-type",
        type=str,
        default="none",
        choices=["batch", "layer", "none"],
        help="MPNN normalization type. Use 'none' for cross-N (default).",
    )
    parser.add_argument(
        "--de-gap-threshold", type=float, default=0.05, help="ΔE/gap success threshold"
    )
    return parser


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    """Entry point for zero-shot cross-N generalization experiment."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    use_n_feature = not args.no_n_feature
    n_features = 3 if use_n_feature else 2
    threshold = args.de_gap_threshold

    # ── Step 1: Load all source files ────────────────────────────────
    source_paths = [Path(p) for p in args.source_file]
    sources: list[tuple[int, str, np.ndarray, np.ndarray, np.ndarray]] = []

    for sp in source_paths:
        try:
            source_data = _load_source_data(sp, args.source_seed)
            sources.append(source_data)
            n_src, topo, h_vals, theta, _ = source_data
            logger.info(f"  Loaded: N={n_src}, {len(h_vals)} h-points from {sp.name}")
        except (FileNotFoundError, ValueError) as e:
            logger.error(str(e))
            return 1

    # Verify all have same topology and theta dim
    topology = sources[0][1]
    theta_dim = sources[0][3].shape[1]
    n_source_sizes = sorted(set(s[0] for s in sources))

    # Sanity checks
    if args.target_n in n_source_sizes:
        logger.warning(
            f"  ⚠ target_n={args.target_n} is IN source sizes {n_source_sizes}. "
            f"This is NOT zero-shot — the model sees this N during training. "
            f"Results valid but not a generalization test."
        )

    total_points = sum(len(s[2]) for s in sources)
    if total_points < 10:
        logger.warning(
            f"  ⚠ Small training set ({total_points} points). "
            f"Consider adding more source files for robust generalization."
        )

    logger.info("=" * 60)
    logger.info("A1: Zero-shot Cross-N GNN Generalization")
    logger.info(
        f"  N-feature encoding: {'ENABLED (N/100)' if use_n_feature else 'DISABLED (ablation)'}"
    )
    logger.info(f"  Source sizes: {n_source_sizes}")
    logger.info(f"  Total training points: {sum(len(s[2]) for s in sources)}")
    logger.info(f"  Target: N={args.target_n}")
    logger.info(
        f"  Node features: {n_features} ({'h, coord, N/100' if use_n_feature else 'h, coord'})"
    )
    logger.info(f"  Topology: {topology}")
    logger.info("=" * 60)

    # ── Step 2: Build combined training dataset ──────────────────────
    dataset = _build_training_dataset(sources, use_n_feature)
    logger.info(f"\nTraining dataset: {len(dataset)} graphs (multi-N combined)")

    # ── Step 3: Train MPNN ───────────────────────────────────────────
    model = MPNNPredictor(
        node_features=n_features,
        hidden_dim=args.hidden_dim,
        n_layers=3,
        output_dim=theta_dim,
        norm_type=args.norm_type,
    )

    logger.info(f"Training MPNN (hidden={args.hidden_dim}, layers=3, epochs={args.n_epochs})...")
    t0 = time.time()
    metrics = train_mpnn(model, dataset, n_epochs=args.n_epochs, seed=42)
    train_time = time.time() - t0
    logger.info(f"  Final MSE={metrics['final_mse']:.2e}, time={train_time:.1f}s")

    if metrics.get("stopped_early"):
        logger.warning(f"  Training stopped early: {metrics.get('stop_reason')}")

    # ── Step 4: Deploy at N_target (DIFFERENT graph size) ────────────
    n_target = args.target_n
    if args.target_h_values:
        h_test = sorted(args.target_h_values, reverse=True)
    else:
        # Auto-compute using scaling law: h_min = 1.5 + 0.020 * N^1.31 (corrected)
        h_min_target = 1.5 + 0.020 * n_target**1.31
        h_test = np.linspace(h_min_target + 1.5, h_min_target + 0.5, 5).tolist()

    logger.info(f"\n─── Zero-shot Deploy: N_target={n_target} ───")
    logger.info(f"  h_test = {[f'{h:.3f}' for h in h_test]}")

    builder = HamiltonianBuilder()
    hva = HVACircuitBuilder()
    solver = ClassicalSolver()
    backend = MPSBackend(
        strategy=args.strategy,
        chi_max=64,
        precision=args.precision,
        seed=42,
    )

    # Pre-compute target graph structure (topology is h-independent)
    lattice_ref = make_lattice(topology, n_target, J=1.0, h=h_test[0])
    edge_index_np, coord_target = builder.build_graph_data(lattice_ref)
    edge_index_target = torch.tensor(edge_index_np, dtype=torch.long)

    model.eval()
    deploy_results = []

    for h_val in h_test:
        t0 = time.time()

        # Build target graph with N_target nodes (reuse pre-computed structure)
        graph = _build_graph(
            n_target,
            h_val,
            topology,
            use_n_feature,
            edge_index=edge_index_target,
            coord=coord_target,
        )

        # Predict θ using model trained on source N(s)
        with torch.no_grad():
            theta_pred = model(graph).numpy().flatten()
        theta_pred = canonicalize_theta(theta_pred)

        # Evaluate energy at N_target via MPS backend
        lattice_target = make_lattice(topology, n_target, J=1.0, h=h_val)
        H = builder.build(lattice_target)
        circuit, _ = hva.create(n_target, 1, lattice_target)
        e_pred = backend.evaluate(circuit, H, theta_pred)

        # DMRG reference
        gt = solver.solve(H, lattice_target, method="dmrg")
        de_gap = abs(e_pred - gt.ground_energy) / max(gt.gap, 1e-10)
        elapsed = time.time() - t0

        status = "✅" if de_gap < threshold else "❌"
        theta_str = ", ".join(f"{t:.4f}" for t in theta_pred)
        logger.info(
            f"  h={h_val:.3f}: ΔE/gap={de_gap:.4f} ({de_gap * 100:.2f}%) "
            f"θ=[{theta_str}] {status} ({elapsed:.1f}s)"
        )

        deploy_results.append(
            {
                "h": float(h_val),
                "e_pred": float(e_pred),
                "e_dmrg": float(gt.ground_energy),
                "gap": float(gt.gap),
                "de_gap": float(de_gap),
                "theta_pred": theta_pred.tolist(),
                "passed": bool(de_gap < threshold),
                "time_s": elapsed,
            }
        )

    # ── Step 5: Summary ──────────────────────────────────────────────
    n_pass = sum(1 for r in deploy_results if r["passed"])
    n_total = len(deploy_results)
    de_gaps = [r["de_gap"] for r in deploy_results]
    mean_de = float(np.mean(de_gaps))
    max_de = float(np.max(de_gaps))
    std_de = float(np.std(de_gaps)) if len(de_gaps) > 1 else 0.0

    logger.info("\n─── Zero-Shot Summary ───")
    logger.info(f"  Source: N={n_source_sizes} ({sum(len(s[2]) for s in sources)} points)")
    logger.info(f"  Target: N={n_target} ({n_total} h-points)")
    logger.info(f"  N-feature: {'ENABLED' if use_n_feature else 'DISABLED'}")
    logger.info(f"  Pass: {n_pass}/{n_total}")
    logger.info(f"  Mean ΔE/gap: {mean_de:.4f} ({mean_de * 100:.2f}%)")
    logger.info(f"  Max ΔE/gap: {max_de:.4f} ({max_de * 100:.2f}%)")
    logger.info(f"  Std ΔE/gap: {std_de:.4f}")

    if n_pass == n_total:
        logger.info("  🎯 FULL GENERALIZATION — GNN cross-N zero-shot works!")
    elif n_pass > 0:
        logger.info(f"  ⚠️ PARTIAL — {n_pass}/{n_total} passed threshold")
    else:
        logger.info("  ❌ FAILED — no h-points below threshold")

    # ── Step 6: Save results ─────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    n_src_tag = "_".join(str(n) for n in n_source_sizes)
    feat_tag = "3feat" if use_n_feature else "2feat_ablation"
    out_path = output_dir / f"zero_shot_N{n_src_tag}_to_N{n_target}_{feat_tag}_{timestamp}.json"

    result_envelope = {
        "experiment": "zero_shot_cross_n_v2",
        "version": "2.0",
        "metadata": {
            "n_source_sizes": n_source_sizes,
            "n_target": n_target,
            "topology": topology,
            "source_seed": args.source_seed,
            "source_files": [str(p) for p in source_paths],
            "strategy": args.strategy,
            "use_n_feature": use_n_feature,
            "n_features": n_features,
            "norm_type": args.norm_type,
            "de_gap_threshold": threshold,
            "hidden_dim": args.hidden_dim,
            "n_epochs": args.n_epochs,
        },
        "training": {
            "n_points": len(dataset),
            "n_points_per_source": {str(s[0]): len(s[2]) for s in sources},
            "final_mse": float(metrics["final_mse"]),
            "stopped_early": metrics.get("stopped_early", False),
            "stop_reason": metrics.get("stop_reason", "completed"),
            "time_s": train_time,
        },
        "deployment": {
            "h_test": [float(h) for h in h_test],
            "results": deploy_results,
            "n_pass": n_pass,
            "n_total": n_total,
            "mean_de_gap": mean_de,
            "max_de_gap": max_de,
            "std_de_gap": std_de,
        },
        "summary": {
            "generalization_success": n_pass == n_total,
            "pass_rate": n_pass / max(n_total, 1),
            "verdict": "FULL" if n_pass == n_total else ("PARTIAL" if n_pass > 0 else "FAILED"),
        },
    }

    json_dump(result_envelope, out_path)
    logger.info(f"  Results saved: {out_path}")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
