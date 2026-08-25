"""VQEzy External Benchmark — Validate MPNN generalization on external dataset.

Evaluates our GNN-HVA pipeline against VQEzy (Zhang et al., 2025) to
demonstrate cross-dataset generalization. Key claims validated:

1. Our HVA + warm-start VQE achieves comparable/better energy than
   VQEzy's HEA (CZRXRY) + Adam with significantly fewer iterations.
2. Our trained MPNN generalizes zero-shot to VQEzy Hamiltonians.

Prerequisites:
    git clone https://github.com/chizhang24/VQEzy.git data/VQEzy

Usage:
    # Quick smoke test (10 instances)
    .venv/bin/python scripts/analysis/benchmark_vqezy.py --quick

    # Full TFI benchmark (all 1000 instances)
    .venv/bin/python scripts/analysis/benchmark_vqezy.py --dataset data/VQEzy/qmanybody/ti_8_qubit.h5

    # With MPNN zero-shot evaluation
    .venv/bin/python scripts/analysis/benchmark_vqezy.py \\
        --dataset data/VQEzy/qmanybody/ti_8_qubit.h5 \\
        --mpnn-checkpoint results/checkpoints/mpnn_tfim_square_n8_p1.pt

    # Filter to our valid regime (h >= 1.0)
    .venv/bin/python scripts/analysis/benchmark_vqezy.py \\
        --dataset data/VQEzy/qmanybody/ti_8_qubit.h5 \\
        --h-min 1.0 --h-max 5.0

    # Per-instance mode (no warm-start, fair 1:1 comparison)
    .venv/bin/python scripts/analysis/benchmark_vqezy.py \\
        --dataset data/VQEzy/qmanybody/ti_8_qubit.h5 \\
        --mode per_instance
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qmbp_simulation.predictors.external_benchmarks import (
    VQEzyBenchmarkEvaluator,
    load_vqezy_tfi,
    load_vqezy_xyz,
)
from qmbp_simulation.utils.helpers import json_dump

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="VQEzy External Benchmark — validate MPNN on external VQE dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to VQEzy HDF5 file (e.g., data/VQEzy/qmanybody/ti_8_qubit.h5)",
    )
    parser.add_argument(
        "--model-type",
        choices=["tfi", "xyz"],
        default="tfi",
        help="VQEzy model type to load (default: tfi)",
    )
    parser.add_argument(
        "--mode",
        choices=["sweep", "per_instance"],
        default="sweep",
        help="Evaluation mode: 'sweep' (warm-start) or 'per_instance' (independent)",
    )
    parser.add_argument("--n", type=int, default=8, help="Number of qubits (default: 8)")
    parser.add_argument("--p", type=int, default=1, help="HVA layers (default: 1)")
    parser.add_argument(
        "--topology", default="square", help="Lattice topology (default: square)"
    )
    parser.add_argument(
        "--h-min", type=float, default=None, help="Minimum h-value filter"
    )
    parser.add_argument(
        "--h-max", type=float, default=None, help="Maximum h-value filter"
    )
    parser.add_argument(
        "--j-min", type=float, default=None, help="Minimum j-value filter"
    )
    parser.add_argument(
        "--j-max", type=float, default=None, help="Maximum j-value filter"
    )
    parser.add_argument(
        "--max-instances", type=int, default=None, help="Limit number of instances"
    )
    parser.add_argument(
        "--n-restarts", type=int, default=3, help="VQE restarts (default: 3)"
    )
    parser.add_argument(
        "--maxiter", type=int, default=200, help="VQE max iterations (default: 200)"
    )
    parser.add_argument(
        "--mpnn-checkpoint",
        type=str,
        default=None,
        help="Path to MPNN checkpoint for zero-shot eval",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/benchmarks/vqezy_benchmark.json)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 10 instances, h∈[1.0, 3.0], for smoke testing",
    )
    parser.add_argument(
        "--rescale-h-by-j",
        action="store_true",
        help="Rescale h→h/j before MPNN prediction (tests ratio generalization)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def find_vqezy_dataset(args) -> Path:
    """Find VQEzy dataset file, with helpful error messages."""
    if args.dataset:
        path = Path(args.dataset)
        if path.exists():
            return path
        # Try relative to project root
        path = PROJECT_ROOT / args.dataset
        if path.exists():
            return path
        raise FileNotFoundError(
            f"VQEzy dataset not found at: {args.dataset}\n"
            f"Clone it with: git clone https://github.com/chizhang24/VQEzy.git "
            f"{PROJECT_ROOT / 'data' / 'VQEzy'}"
        )

    # Auto-discover in standard locations
    search_paths = [
        PROJECT_ROOT / "data" / "VQEzy" / "qmanybody",
        PROJECT_ROOT / "VQEzy" / "qmanybody",
        Path.home() / "VQEzy" / "qmanybody",
    ]

    filename = "ti_8_qubit.h5" if args.model_type == "tfi" else "xyz_4_qubit.h5"

    for base in search_paths:
        candidate = base / filename
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"VQEzy dataset not found in standard locations.\n"
        f"Clone it with:\n"
        f"  git clone https://github.com/chizhang24/VQEzy.git {PROJECT_ROOT / 'data' / 'VQEzy'}\n"
        f"Or specify path with --dataset"
    )


def main() -> None:
    """Main entry point."""
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Quick mode defaults
    if args.quick:
        args.max_instances = args.max_instances or 10
        args.h_min = args.h_min or 1.0
        args.h_max = args.h_max or 3.0
        logger.info("🚀 Quick mode: 10 instances, h∈[1.0, 3.0]")

    # Find and load dataset
    dataset_path = find_vqezy_dataset(args)
    logger.info(f"Loading VQEzy dataset: {dataset_path}")

    if args.model_type == "tfi":
        dataset = load_vqezy_tfi(
            dataset_path,
            h_min=args.h_min,
            h_max=args.h_max,
            j_min=args.j_min,
            j_max=args.j_max,
            max_instances=args.max_instances,
        )
    else:
        dataset = load_vqezy_xyz(dataset_path, max_instances=args.max_instances)

    logger.info(f"Dataset: {dataset.summary()}")

    if len(dataset) == 0:
        logger.error("No instances match the specified filters. Exiting.")
        sys.exit(1)

    # Create evaluator
    evaluator = VQEzyBenchmarkEvaluator(
        n_qubits=args.n,
        p_layers=args.p,
        topology=args.topology,
        n_restarts=args.n_restarts,
        maxiter=args.maxiter,
        seed=args.seed,
        mpnn_checkpoint=args.mpnn_checkpoint,
    )

    # Run benchmark
    logger.info(f"Running benchmark in '{args.mode}' mode...")
    results = evaluator.evaluate(
        dataset, mode=args.mode, rescale_h_by_j=getattr(args, "rescale_h_by_j", False)
    )

    # Print summary
    print("\n" + results.summary())

    # Save results
    output_path = Path(args.output) if args.output else (
        PROJECT_ROOT / "results" / "benchmarks" / "vqezy_benchmark.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_dump(results.to_dict(), output_path)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
