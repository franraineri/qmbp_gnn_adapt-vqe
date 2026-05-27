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
import time


def parse_args() -> argparse.Namespace:
    from qmbp_simulation.framework.cli import (
        add_mpnn_args,
        add_output_args,
        add_sweep_args,
        add_system_args,
        add_vqe_args,
        create_base_parser,
    )

    parser = create_base_parser(
        description="Full 4-phase pipeline runner",
        epilog="""
Examples:
    %(prog)s --n-qubits 6 --p 2
    %(prog)s --n-qubits 10 --h-values 2.0 1.75 1.5 1.25 1.0
    %(prog)s --n-qubits 6 --output-dir results/custom_run
    %(prog)s --n-qubits 6 --skip-phase4
        """,
    )

    add_system_args(parser)
    add_sweep_args(parser)
    add_vqe_args(parser)
    add_mpnn_args(parser)
    add_output_args(parser)

    # Phase control (specific to this script)
    phase_group = parser.add_argument_group("Phase control")
    phase_group.add_argument(
        "--skip-phase1", action="store_true", help="Skip Phase 1 (load from checkpoint)"
    )
    phase_group.add_argument(
        "--skip-phase2", action="store_true", help="Skip Phase 2 (load from checkpoint)"
    )
    phase_group.add_argument("--skip-phase3", action="store_true", help="Skip Phase 3")
    phase_group.add_argument("--skip-phase4", action="store_true", help="Skip Phase 4")
    phase_group.add_argument("--checkpoint", type=str, help="Checkpoint file for skip/resume")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from qmbp_simulation.framework.cli import (
        build_mpnn_config_dict,
        configure_logging,
        resolve_output_dir,
        validate_descending_sweep,
    )
    from qmbp_simulation.framework.result_io import save_pipeline_result

    configure_logging(verbose=args.verbose, debug=args.debug)

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

    # Validate h-values (must be descending)
    h_values = validate_descending_sweep(args.h_values)

    # Output directory
    output_dir = resolve_output_dir(args.output_dir)

    # MPNN config
    mpnn_config = build_mpnn_config_dict(args)

    print("Pipeline Configuration:")
    print(f"  Lattice: {args.topology}, N={args.n_qubits}, J={args.J}")
    print(f"  VQE: p={args.p}, restarts={args.n_restarts}, maxiter={args.maxiter}")
    print(f"  Sweep: {len(h_values)} h-values [{h_values[0]:.2f} → {h_values[-1]:.2f}]")
    print(f"  Test points: {args.h_test}")
    print(
        f"  MPNN: hidden={mpnn_config.get('hidden_dim', 128)}, "
        f"epochs={mpnn_config.get('n_epochs', 6000)}"
    )
    print(f"  Output: {output_dir}")
    print()

    # Run pipeline
    runner = PipelineRunner(
        lattice=lattice,
        config=vqe_config,
        checkpoint_dir=output_dir / "checkpoints",
        verbose=args.verbose or args.debug,
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
    elif not args.skip_phase4:
        print("  ⚠️  Phase 4 skipped (Phase 3 MPNN training failed or was skipped)")
        print("  Possible causes: insufficient training data, fidelity filter too strict,")
        print("  or h-values outside the valid regime for this topology/N.")

    # Save diagnostics
    if results.get("diagnostics"):
        from qmbp_simulation.utils.helpers import json_dump

        diag_path = output_dir / "diagnostics.json"
        json_dump(results["diagnostics"], diag_path)
        print(f"\n  Diagnostics saved to: {diag_path}")

    # Save full pipeline results
    run_output = {
        "config": {
            "n_qubits": args.n_qubits,
            "topology": args.topology,
            "J": args.J,
            "p_layers": args.p,
            "n_restarts": args.n_restarts,
            "maxiter": args.maxiter,
            "mpnn": mpnn_config,
            "h_values": h_values.tolist(),
            "h_test": args.h_test,
        },
        "elapsed_s": elapsed,
        "phase4_results": [
            {
                "h_test": d.h_test,
                "predicted_energy": d.predicted_energy,
                "delta_e": d.delta_e,
                "delta_e_over_gap": d.delta_e_over_gap,
                "mag_x_pred": d.mag_x_pred,
                "corr_zz_pred": d.corr_zz_pred,
                "mag_x_error": d.mag_x_error,
                "corr_zz_error": d.corr_zz_error,
                "phase_label": d.phase_label,
                "metrics_checklist": d.metrics_checklist,
            }
            for d in (results.get("phase4") or [])
        ],
        "diagnostics": results.get("diagnostics"),
    }

    path = save_pipeline_result(run_output, output_dir=output_dir)
    print(f"  Results saved to: {path}")


if __name__ == "__main__":
    main()
