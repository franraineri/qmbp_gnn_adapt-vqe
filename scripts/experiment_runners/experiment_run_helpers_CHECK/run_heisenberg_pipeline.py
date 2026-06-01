#!/usr/bin/env python3
"""CLI for executing the full 4-phase pipeline with Heisenberg XXZ model.

Extends run_pipeline.py with --model and --delta support, dispatching to
the appropriate ModelSpec via the ModelRegistry.

Usage:
    python scripts/experiment_runners/experiment_run_helpers_CHECK/run_heisenberg_pipeline.py \
        --n-qubits 6 --model heisenberg --delta 1.0 --h-values 4.0 3.5 3.0 2.5 2.0
    python scripts/experiment_runners/experiment_run_helpers_CHECK/run_heisenberg_pipeline.py \
        --n-qubits 6 --model xy --h-values 4.0 3.5 3.0 2.5 2.0

Phases:
    1. Classical ground truth (exact diag) — uses ModelSpec.build_hamiltonian
    2. VQE optimization (descending warm-start) — uses ModelSpec.create_circuit
    3. MPNN training (GINConv predictor) — output_dim = params_per_layer × p
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
        description="Full 4-phase pipeline runner (model-agnostic)",
        epilog="""
Examples:
    %(prog)s --n-qubits 6 --model heisenberg --delta 1.0
    %(prog)s --n-qubits 6 --model xy --h-values 4.0 3.5 3.0 2.5
    %(prog)s --n-qubits 6 --model heisenberg --delta 0.5 --topology ladder
        """,
    )

    add_system_args(parser)
    add_sweep_args(parser)
    add_vqe_args(parser)
    add_mpnn_args(parser)
    add_output_args(parser)

    # Model selection (specific to this script)
    model_group = parser.add_argument_group("Model configuration")
    model_group.add_argument(
        "--model",
        type=str,
        default="heisenberg",
        choices=["heisenberg", "xy", "tfim"],
        help="Spin model type (default: heisenberg)",
    )
    model_group.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Anisotropy parameter Δ for XXZ model (default: model-specific)",
    )

    # Phase control
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


def _classify_scientific_result(phase2_summary: dict | None, spec) -> dict:
    """Classify the scientific outcome for thesis documentation.

    Returns a structured classification of the result:
    - "full_success": ΔE/gap < 5% achieved
    - "partial_success": some fidelity above threshold but pipeline incomplete
    - "negative_expressibility": max fidelity < threshold (HVA limitation)
    - "negative_fundamental": max fidelity < 0.30 (fundamental limitation)
    """
    if phase2_summary is None:
        return {"classification": "incomplete", "reason": "Phase 2 data unavailable"}

    max_fid = phase2_summary["max_fidelity"]
    n_above = phase2_summary["n_above_threshold"]
    threshold = phase2_summary["fidelity_threshold"]

    if max_fid < 0.30:
        return {
            "classification": "negative_fundamental",
            "reason": (
                f"Max fidelity {max_fid:.4f} << threshold {threshold}. "
                f"HVA p≤2 fundamentally cannot express the ground state. "
                f"Entanglement exceeds ansatz capacity."
            ),
            "max_fidelity": max_fid,
            "thesis_value": "Documents expressibility limit of shallow HVA for this model.",
        }
    elif max_fid < threshold:
        return {
            "classification": "negative_expressibility",
            "reason": (
                f"Max fidelity {max_fid:.4f} < threshold {threshold}. "
                f"HVA p=2 has insufficient expressibility at these h-values."
            ),
            "max_fidelity": max_fid,
            "n_above_threshold": n_above,
            "thesis_value": "Quantifies the gap between HVA capacity and model requirements.",
        }
    elif n_above < 5:
        return {
            "classification": "partial_success",
            "reason": (
                f"Only {n_above} points above threshold (need ≥5 for MPNN). "
                f"Valid regime too narrow for full pipeline."
            ),
            "max_fidelity": max_fid,
            "n_above_threshold": n_above,
        }
    else:
        return {
            "classification": "full_success",
            "reason": f"{n_above} points above threshold. Full pipeline viable.",
            "max_fidelity": max_fid,
            "n_above_threshold": n_above,
        }


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
    from qmbp_simulation.models import VQEConfig, get_model_spec

    # Get model spec with delta override if applicable
    spec = get_model_spec(args.model)
    if args.delta is not None and args.model in ("heisenberg", "xy"):
        # User explicitly provided --delta, override the model's built-in value
        spec = spec.with_delta(args.delta)

    # Build lattice
    lattice = make_lattice(
        topology=args.topology,
        n_qubits=args.n_qubits,
        J=args.J,
        h=2.0,  # Base h (will be varied in sweep)
        periodic=args.periodic,
    )

    # Build VQE config — CLI args take precedence, then model defaults
    # Note: CLI defaults are n_restarts=5, maxiter=1000 (TFIM-oriented).
    # For Heisenberg, model defaults are n_restarts=10, maxiter=1500.
    # The variant runner always passes explicit values, so this fallback
    # only matters for manual CLI usage without --n-restarts/--maxiter.
    model_vqe_defaults = spec.get_vqe_config_overrides()
    # Use model defaults when CLI has its built-in default (5/1000)
    effective_restarts = args.n_restarts
    effective_maxiter = args.maxiter
    if effective_restarts == 5 and "n_restarts" in model_vqe_defaults:
        effective_restarts = model_vqe_defaults["n_restarts"]
    if effective_maxiter == 1000 and "maxiter" in model_vqe_defaults:
        effective_maxiter = model_vqe_defaults["maxiter"]

    vqe_config = VQEConfig(
        p_layers=args.p,
        n_restarts=effective_restarts,
        maxiter=effective_maxiter,
        restart_sigma=model_vqe_defaults.get("restart_sigma", 0.5),
    )

    # Validate h-values (must be descending)
    h_values = validate_descending_sweep(args.h_values)

    # Output directory
    output_dir = resolve_output_dir(args.output_dir)

    # MPNN config — use model-specific hidden_dim as default
    mpnn_config = build_mpnn_config_dict(args)
    if "hidden_dim" not in mpnn_config or mpnn_config["hidden_dim"] is None:
        mpnn_config["hidden_dim"] = spec.mpnn_hidden_dim

    print("Pipeline Configuration (Model-Agnostic):")
    print(f"  Model: {spec.name} | Δ={spec.hamiltonian_kwargs.get('delta', 'N/A')}")
    print(f"  Params/layer: {spec.params_per_layer} | Total: {spec.total_params_for_p(args.p)}")
    print(f"  Initial state: {spec.initial_state}")
    print(f"  Fidelity threshold: {spec.fidelity_threshold}")
    print(f"  Lattice: {args.topology}, N={args.n_qubits}, J={args.J}")
    print(f"  VQE: p={args.p}, restarts={vqe_config.n_restarts}, maxiter={vqe_config.maxiter}")
    print(f"  Sweep: {len(h_values)} h-values [{h_values[0]:.2f} → {h_values[-1]:.2f}]")
    print(f"  Test points: {args.h_test}")
    print(
        f"  MPNN: hidden={mpnn_config.get('hidden_dim', 128)}, "
        f"epochs={mpnn_config.get('n_epochs', 6000)}"
    )
    print(f"  Seed: {args.seed if args.seed is not None else 'None (non-deterministic)'}")
    print(f"  Output: {output_dir}")
    print()

    # Run pipeline with model_spec
    runner = PipelineRunner(
        lattice=lattice,
        config=vqe_config,
        checkpoint_dir=output_dir / "checkpoints",
        verbose=args.verbose or args.debug,
        seed=args.seed,
        model_spec=spec,
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

    # ── Compute entanglement analysis (key for Heisenberg negative results) ──
    entanglement_data = None
    if results.get("phase1") and spec.name in ("heisenberg", "xy"):
        try:
            from qmbp_simulation.analysis.entanglement import EntanglementAnalyzer

            analyzer = EntanglementAnalyzer()
            ground_states = [r.ground_state for r in results["phase1"]]
            ent_results = analyzer.analyze_sweep(h_values, ground_states, args.n_qubits)
            entanglement_data = [
                {"h": e.h, "entropy": e.entropy, "normalized_entropy": e.normalized_entropy}
                for e in ent_results
            ]

            # Find HVA capacity if Phase 2 data available
            if results.get("phase2"):
                fidelities = [r.fidelity for r in results["phase2"]]
                capacity = analyzer.find_hva_capacity_threshold(
                    ent_results, fidelities, fidelity_threshold=spec.fidelity_threshold
                )
                if capacity is not None:
                    print(f"  HVA capacity (fid≥{spec.fidelity_threshold}): S_max = {capacity:.3f}")
                else:
                    print(
                        f"  HVA capacity: None (no point reaches fidelity ≥ "
                        f"{spec.fidelity_threshold})"
                    )
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Entanglement analysis failed: {e}")

    # ── Extract Phase 2 fidelity summary (critical for negative results) ──
    phase2_summary = None
    if results.get("phase2"):
        vqe_results_list = results["phase2"]
        fidelities = [r.fidelity for r in vqe_results_list]
        energies = [r.energy for r in vqe_results_list]
        phase2_summary = {
            "per_h_fidelity": {
                f"{float(h):.2f}": round(f, 6) for h, f in zip(h_values, fidelities, strict=False)
            },
            "per_h_energy": {
                f"{float(h):.2f}": round(e, 6) for h, e in zip(h_values, energies, strict=False)
            },
            "max_fidelity": round(max(fidelities), 6),
            "min_fidelity": round(min(fidelities), 6),
            "mean_fidelity": round(sum(fidelities) / len(fidelities), 6),
            "n_above_threshold": sum(1 for f in fidelities if f >= spec.fidelity_threshold),
            "fidelity_threshold": spec.fidelity_threshold,
        }

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"  Model: {spec.name} (Δ={spec.hamiltonian_kwargs.get('delta', 'N/A')})")
    print(f"{'=' * 60}")

    # Phase 2 fidelity summary (always print — critical for negative results)
    if phase2_summary:
        print("\n  Phase 2 VQE Fidelity Summary:")
        print(f"    Max fidelity:  {phase2_summary['max_fidelity']:.4f}")
        print(f"    Mean fidelity: {phase2_summary['mean_fidelity']:.4f}")
        print(
            f"    Points above threshold ({spec.fidelity_threshold}): "
            f"{phase2_summary['n_above_threshold']}/{len(h_values)}"
        )
        for h_str, fid in phase2_summary["per_h_fidelity"].items():
            marker = "✓" if fid >= spec.fidelity_threshold else "✗"
            print(f"      h={h_str}: fidelity={fid:.4f} {marker}")

    # Entanglement summary
    if entanglement_data:
        max_ent = max(e["entropy"] for e in entanglement_data)
        min_ent = min(e["entropy"] for e in entanglement_data)
        print("\n  Entanglement Entropy (half-chain):")
        print(f"    Range: [{min_ent:.3f}, {max_ent:.3f}] bits")
        for e in entanglement_data:
            print(
                f"      h={e['h']:.2f}: S={e['entropy']:.3f} "
                f"(normalized={e['normalized_entropy']:.3f})"
            )

    if results.get("phase4"):
        print("\n  Phase 4 Deployment:")
        for deploy in results["phase4"]:
            print(
                f"    h_test={deploy.h_test:.2f}: "
                f"ΔE/gap={deploy.delta_e_over_gap:.4f} "
                f"({'PASS' if deploy.delta_e_over_gap < 0.05 else 'FAIL'})"
            )
    elif not args.skip_phase4:
        print("\n  ⚠️  Phase 4 skipped (Phase 3 MPNN training failed or was skipped)")
        print("  Likely cause: fidelity too low for Heisenberg model at this p/N.")
        print("  This is an expected negative result — HVA p≤2 has limited expressibility.")

    # Save diagnostics
    if results.get("diagnostics"):
        from qmbp_simulation.utils.helpers import json_dump

        diag_path = output_dir / "diagnostics.json"
        json_dump(results["diagnostics"], diag_path)
        print(f"\n  Diagnostics saved to: {diag_path}")

    # Save full pipeline results
    run_output = {
        "config": {
            "model": spec.name,
            "delta": spec.hamiltonian_kwargs.get("delta"),
            "params_per_layer": spec.params_per_layer,
            "initial_state": spec.initial_state,
            "fidelity_threshold": spec.fidelity_threshold,
            "n_qubits": args.n_qubits,
            "topology": args.topology,
            "J": args.J,
            "p_layers": args.p,
            "n_restarts": vqe_config.n_restarts,
            "maxiter": vqe_config.maxiter,
            "restart_sigma": vqe_config.restart_sigma,
            "seed": args.seed,
            "mpnn": mpnn_config,
            "h_values": h_values.tolist(),
            "h_test": args.h_test,
        },
        "elapsed_s": elapsed,
        "phase2_summary": phase2_summary,
        "entanglement": entanglement_data,
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
        "scientific_conclusion": _classify_scientific_result(phase2_summary, spec),
    }

    path = save_pipeline_result(run_output, output_dir=output_dir)
    print(f"  Results saved to: {path}")


if __name__ == "__main__":
    main()
