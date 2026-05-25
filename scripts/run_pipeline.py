#!/usr/bin/env python3
"""CLI for executing the full 4-phase quantum simulation pipeline.

Usage:
    python scripts/run_pipeline.py --n-qubits 6 --p 2
    python scripts/run_pipeline.py --n-qubits 10 --h-values 2.0 1.75 1.5 1.25
    python scripts/run_pipeline.py --n-qubits 6 --output-dir results/my_run
    python scripts/run_pipeline.py --n-qubits 6 --skip-phase3 --skip-phase4

Phases:
    1. Classical ground truth (exact diag / DMRG)
    2. VQE optimization (descending warm-start sweep)
    3. MPNN training (GINConv predictor)
    4. Deployment (predict unseen h-points)
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full 4-phase pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --n-qubits 6 --p 2
    %(prog)s --n-qubits 10 --h-values 2.0 1.75 1.5 1.25 1.0
    %(prog)s --n-qubits 6 --output-dir results/custom_run
    %(prog)s --n-qubits 6 --skip-phase4
        """,
    )

    # Lattice configuration
    parser.add_argument("--n-qubits", type=int, default=6, help="Number of qubits (default: 6)")
    parser.add_argument(
        "--topology", type=str, default="chain_1d", help="Lattice topology (default: chain_1d)"
    )
    parser.add_argument("--J", type=float, default=1.0, help="Coupling constant (default: 1.0)")
    parser.add_argument("--periodic", action="store_true", help="Use periodic boundary conditions")

    # Sweep configuration
    parser.add_argument(
        "--h-values",
        nargs="+",
        type=float,
        help="Transverse field values (descending). Default: linspace(2.0, 0.5, 31)",
    )
    parser.add_argument(
        "--h-test",
        nargs="+",
        type=float,
        default=[1.5],
        help="Unseen h-value(s) for Phase 4 deployment (default: 1.5)",
    )

    # VQE configuration
    parser.add_argument("--p", type=int, default=2, help="HVA layers (default: 2, max: 2)")
    parser.add_argument("--n-restarts", type=int, default=5, help="VQE restarts (default: 5)")
    parser.add_argument(
        "--maxiter", type=int, default=1000, help="VQE max iterations (default: 1000)"
    )

    # MPNN configuration
    parser.add_argument("--hidden-dim", type=int, default=64, help="MPNN hidden dim (default: 64)")
    parser.add_argument(
        "--n-epochs", type=int, default=4000, help="MPNN training epochs (default: 4000)"
    )

    # Phase control
    parser.add_argument(
        "--skip-phase1", action="store_true", help="Skip Phase 1 (load from checkpoint)"
    )
    parser.add_argument(
        "--skip-phase2", action="store_true", help="Skip Phase 2 (load from checkpoint)"
    )
    parser.add_argument("--skip-phase3", action="store_true", help="Skip Phase 3")
    parser.add_argument("--skip-phase4", action="store_true", help="Skip Phase 4")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint file for skip/resume")

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/pipeline",
        help="Output directory (default: results/pipeline)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable INFO logging")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configure logging
    level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING)
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    from qmbp_simulation import PipelineRunner, make_lattice
    from qmbp_simulation.models import VQEConfig

    # Build lattice
    lattice = make_lattice(
        topology=args.topology,
        n_qubits=args.n_qubits,
        J=args.J,
        h=2.0,  # Base h (will be varied in sweep)
        periodic=args.periodic,
    )

    # Build VQE config
    vqe_config = VQEConfig(
        p_layers=args.p,
        n_restarts=args.n_restarts,
        maxiter=args.maxiter,
    )

    # Determine h-values (must be descending)
    if args.h_values:
        h_values = np.array(sorted(args.h_values, reverse=True))
    else:
        h_values = np.linspace(2.0, 0.5, 31)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # MPNN config
    mpnn_config = {
        "hidden_dim": args.hidden_dim,
        "n_epochs": args.n_epochs,
    }

    print("Pipeline Configuration:")
    print(f"  Lattice: {args.topology}, N={args.n_qubits}, J={args.J}")
    print(f"  VQE: p={args.p}, restarts={args.n_restarts}, maxiter={args.maxiter}")
    print(f"  Sweep: {len(h_values)} h-values [{h_values[0]:.2f} → {h_values[-1]:.2f}]")
    print(f"  Test points: {args.h_test}")
    print(f"  Output: {output_dir}")
    print()

    # Run pipeline
    runner = PipelineRunner(
        lattice=lattice,
        config=vqe_config,
        checkpoint_dir=output_dir / "checkpoints",
    )

    t0 = time.time()
    results = runner.run_full(
        h_values=h_values,
        h_test=args.h_test,
        mpnn_config=mpnn_config,
        skip_phase1=args.skip_phase1,
        skip_phase2=args.skip_phase2,
        skip_phase3=args.skip_phase3,
        skip_phase4=args.skip_phase4,
        checkpoint_path=args.checkpoint,
    )
    elapsed = time.time() - t0

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"{'=' * 60}")

    if results.get("phase4"):
        for deploy in results["phase4"]:
            print(
                f"  h_test={deploy.h_test:.2f}: "
                f"ΔE/gap={deploy.delta_e_over_gap:.4f} "
                f"({'PASS' if deploy.delta_e_over_gap < 0.05 else 'FAIL'})"
            )


if __name__ == "__main__":
    main()
