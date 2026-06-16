#!/usr/bin/env python3
"""Unified hardware deployment CLI — single entry point for all QPU operations.

Subcommands:
    cost        Estimate QPU cost for a deployment configuration
    preflight   Run preflight checks against a backend (fake or real)
    rehearsal   Execute the full hardware rehearsal (9 sections + optional preflight)
    analyze     Analyze rehearsal results and produce GO/NO-GO verdict
    deploy      (future) Submit to real IBM Torino hardware

Usage:
    python scripts/hardware.py cost --n-qubits 10 --h-points 3
    python scripts/hardware.py cost --n-qubits 10 --profile nighthawk --spsa disabled
    python scripts/hardware.py preflight --n-qubits 10
    python scripts/hardware.py rehearsal --topology heavy_hex --zne-amplifier pea
    python scripts/hardware.py rehearsal --run-preflight --section 1 2 3
    python scripts/hardware.py analyze --all
    python scripts/hardware.py analyze --json --threshold 0.03

Equivalent make targets:
    make hw-cost N=10 H=3
    make hw-preflight N=10
    make hw-rehearsal
    make hw-analyze
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


# ═══════════════════════════════════════════════════════════════════════════════
# Subcommand: cost
# ═══════════════════════════════════════════════════════════════════════════════


def cmd_cost(args: argparse.Namespace) -> int:
    """Estimate QPU cost for a given deployment configuration."""
    from qmbp_simulation.execution.backends import MitigationOptions
    from qmbp_simulation.execution.hardware import (
        HardwareConfig,
        QPUThroughputProfile,
        SPSACostModel,
        estimate_qpu_cost,
    )

    # Resolve hardware profile
    profiles = {
        "kingston": QPUThroughputProfile.ibm_kingston,
        "torino": QPUThroughputProfile.ibm_torino,
        "heron_r2": QPUThroughputProfile.ibm_heron_r2,
        "nighthawk": QPUThroughputProfile.ibm_nighthawk,
    }
    profile = profiles[args.profile]()

    # Resolve SPSA model
    spsa_models = {
        "default": SPSACostModel,
        "disabled": SPSACostModel.disabled,
        "conservative": SPSACostModel.conservative,
        "aggressive": SPSACostModel.aggressive,
    }
    spsa = spsa_models[args.spsa]()

    # Build config
    mitigation = MitigationOptions(
        zne_amplifier=args.amplifier,
        num_randomizations=args.num_randomizations,
        shots_per_randomization=args.shots_per_randomization,
    )
    config = HardwareConfig(
        n_qubits=args.n_qubits,
        shots=args.shots,
        n_layouts=args.n_layouts,
        mitigation=mitigation,
    )

    est = estimate_qpu_cost(
        config,
        n_h_points=args.h_points,
        profile=profile,
        spsa_model=spsa,
        circuit_depth=args.circuit_depth,
        cx_count=args.cx_count,
    )

    if args.json:
        import dataclasses

        print(json.dumps(dataclasses.asdict(est), indent=2))
        return 0

    print()
    print(f"  QPU Cost Estimate — N={args.n_qubits}, {args.h_points} h-points, {est.amplifier}")
    print(f"  Backend: {profile.name} | SPSA: {args.spsa}")
    print(f"  {'─' * 58}")
    print(f"    Effective CLOPS:         {est.effective_clops} (ref: {est.estimated_clops})")
    print(f"    Time per circuit:        {est.time_per_circuit_s:.2f}s")
    print(f"    Circuits per h-point:    {est.circuits_per_h}")
    print(f"    Shots per h-point:       {est.shots_per_h:,}")
    print(f"    Total shots:             {est.total_shots:,}")
    print(f"    PEA noise learning:      {est.pea_noise_learning_s:.1f}s (one-time)")
    print(f"    Classical latency:       {est.classical_latency_s:.1f}s")
    print(f"    SPSA per h (if triggered): {est.spsa_per_h_if_triggered_s:.1f}s")
    print()
    print("    ┌─────────────────────────────────────────────────────┐")
    print(
        f"    │  Optimistic (no SPSA):  {est.est_total_optimistic_s:7.1f}s  ({est.est_total_optimistic_s / 60:5.1f} min)  │"
    )
    print(
        f"    │  Expected (P={spsa.trigger_probability:.2f}):   {est.est_total_s:7.1f}s  ({est.est_total_s / 60:5.1f} min)  │"
    )
    print(
        f"    │  Pessimistic (always):  {est.est_total_pessimistic_s:7.1f}s  ({est.est_total_pessimistic_s / 60:5.1f} min)  │"
    )
    print("    └─────────────────────────────────────────────────────┘")
    print()
    fits_job_s = "✅" if est.fits_per_job else "❌"
    fits_sweep_s = "✅" if est.fits_full_sweep_10min else "❌"
    print(f"    Fits per job ({est.max_execution_time_s}s):  {fits_job_s}")
    print(f"    Fits full sweep (10 min): {fits_sweep_s}")
    print()

    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# Subcommand: preflight
# ═══════════════════════════════════════════════════════════════════════════════


def cmd_preflight(args: argparse.Namespace) -> int:
    """Run preflight checks against a FakeTorino or real backend."""
    from qmbp_simulation.execution.backends import MitigationOptions
    from qmbp_simulation.execution.hardware import HardwareConfig
    from qmbp_simulation.execution.hardware.preflight import run_preflight_checks
    from qmbp_simulation.framework.logging import StructuredLogger

    mitigation = MitigationOptions(
        zne_amplifier=args.amplifier,
        num_randomizations=32,
        shots_per_randomization=128,
    )
    config = HardwareConfig(
        mode="fake_backend" if not args.real else "hardware",
        n_qubits=args.n_qubits,
        shots=args.shots,
        n_layouts=args.n_layouts,
        mitigation=mitigation,
    )

    slog = StructuredLogger(experiment_id="HW_PREFLIGHT")

    # Resolve the backend object
    if args.real:
        # Real hardware requires IBM credentials
        try:
            import os

            from qiskit_ibm_runtime import QiskitRuntimeService

            service = QiskitRuntimeService(
                channel="ibm_quantum_platform",
                token=os.environ.get("IBM_KEY"),
                instance=os.environ.get("IBM_INSTANCE_CRN"),
            )
            backend = service.backend(config.backend_name)
        except Exception as exc:
            print(f"\n  ❌ Cannot connect to real hardware: {exc}")
            print("  Set IBM_KEY and IBM_INSTANCE_CRN environment variables.")
            return 1
    else:
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        backend = FakeTorino()

    print(f"\n  Preflight — N={args.n_qubits}, mode={config.mode}, amplifier={args.amplifier}")
    print(f"  {'─' * 50}")

    checks = run_preflight_checks(backend, config, slog)

    if args.json:
        print(json.dumps(checks, indent=2, default=str))
        return 1 if checks.get("abort") else 0

    # Print results
    for key, val in sorted(checks.items()):
        if key in ("abort", "abort_reason"):
            continue
        icon = "  "
        if "warning" in key.lower():
            icon = "⚠️"
        elif "error" in key.lower():
            icon = "❌"
        elif "sufficient" in key.lower() or "operational" in key.lower():
            icon = "✅" if val else "❌"
        print(f"    {icon} {key}: {val}")

    print()
    if checks.get("abort"):
        print(f"  ❌ ABORT: {checks['abort_reason']}")
        return 1
    else:
        print("  ✅ Preflight PASSED — backend viable for deployment")
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# Subcommand: rehearsal
# ═══════════════════════════════════════════════════════════════════════════════


def cmd_rehearsal(args: argparse.Namespace) -> int:
    """Run the hardware rehearsal (delegates to HardwareRehearsalV2.main)."""
    # Build the argv list that the rehearsal runner expects
    argv = []
    if args.topology:
        argv.extend(["--topology", args.topology])
    if args.n_qubits:
        argv.extend(["--n-qubits", str(args.n_qubits)])
    if args.amplifier:
        argv.extend(["--zne-amplifier", args.amplifier])
    if args.shots:
        argv.extend(["--shots", str(args.shots)])
    if args.section:
        argv.extend(["--section"] + [str(s) for s in args.section])
    if args.run_preflight:
        argv.append("--run-preflight")
    if args.dry_run:
        argv.append("--dry-run")
    if args.verbose:
        argv.append("--verbose")
    if args.skip_preflight:
        argv.append("--skip-preflight")
    if args.p_layers:
        argv.extend(["--p-layers", str(args.p_layers)])
    if args.h_test:
        argv.extend(["--h-test"] + [str(h) for h in args.h_test])
    if args.h_train:
        argv.extend(["--h-train"] + [str(h) for h in args.h_train])
    if args.vqe_restarts:
        argv.extend(["--vqe-restarts", str(args.vqe_restarts)])
    if args.mpnn_epochs:
        argv.extend(["--mpnn-epochs", str(args.mpnn_epochs)])

    # Patch sys.argv and run
    sys.argv = ["run_hardware_rehearsal_v2.py"] + argv

    from scripts.experiment_runners.run_hardware_rehearsal_v2 import HardwareRehearsalV2

    HardwareRehearsalV2.main()
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# Subcommand: analyze
# ═══════════════════════════════════════════════════════════════════════════════


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze rehearsal results (delegates to analyze_hw_rehearsal_v2)."""
    argv = []
    if args.all:
        argv.append("--all")
    if args.json:
        argv.append("--json")
    if args.threshold:
        argv.extend(["--threshold", str(args.threshold)])
    if args.section_filter:
        argv.extend(["--section-filter"] + [str(s) for s in args.section_filter])
    if args.exp_dir:
        argv.extend(["--exp-dir", args.exp_dir])

    sys.argv = ["analyze_hw_rehearsal_v2.py"] + argv

    # Import and run main directly
    sys.path.insert(0, str(_ROOT / "scripts"))
    from analyze_hw_rehearsal_v2 import main

    main()
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# Argument Parser
# ═══════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    """Build the unified CLI parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="hardware",
        description="Unified hardware deployment CLI — QPU cost, preflight, rehearsal, and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s cost --n-qubits 10 --h-points 3
  %(prog)s cost --n-qubits 6 --profile nighthawk --spsa disabled --json
  %(prog)s preflight --n-qubits 10
  %(prog)s rehearsal --topology heavy_hex --run-preflight
  %(prog)s rehearsal --section 8 9 --dry-run
  %(prog)s analyze --all --json
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── cost ─────────────────────────────────────────────────────────────────
    p_cost = subparsers.add_parser(
        "cost",
        help="Estimate QPU time and shot budget",
        description="Compute QPU cost estimate with depth-aware CLOPS, amortized PEA, and SPSA scenarios.",
    )
    p_cost.add_argument("--n-qubits", "-N", type=int, default=10, help="System size (default: 10)")
    p_cost.add_argument(
        "--h-points", type=int, default=3, help="Number of h-values to evaluate (default: 3)"
    )
    p_cost.add_argument(
        "--shots", type=int, default=16384, help="Shots per circuit (default: 16384)"
    )
    p_cost.add_argument("--n-layouts", type=int, default=3, help="Number of layouts (default: 3)")
    p_cost.add_argument(
        "--amplifier",
        type=str,
        default="pea",
        choices=["pea", "gate_folding", "adaptive"],
        help="ZNE amplifier (default: pea)",
    )
    p_cost.add_argument(
        "--profile",
        type=str,
        default="kingston",
        choices=["kingston", "torino", "heron_r2", "nighthawk"],
        help="QPU throughput profile (default: kingston)",
    )
    p_cost.add_argument(
        "--spsa",
        type=str,
        default="default",
        choices=["default", "disabled", "conservative", "aggressive"],
        help="SPSA cost model (default: P=0.30)",
    )
    p_cost.add_argument(
        "--circuit-depth",
        type=int,
        default=None,
        help="Known circuit depth (overrides auto-estimate)",
    )
    p_cost.add_argument(
        "--cx-count", type=int, default=None, help="Known 2Q gate count (overrides auto-estimate)"
    )
    p_cost.add_argument(
        "--num-randomizations",
        type=int,
        default=32,
        help="PEA noise learning randomizations (default: 32)",
    )
    p_cost.add_argument(
        "--shots-per-randomization",
        type=int,
        default=128,
        help="PEA shots per randomization (default: 128)",
    )
    p_cost.add_argument("--json", action="store_true", help="Output as JSON")
    p_cost.set_defaults(func=cmd_cost)

    # ── preflight ────────────────────────────────────────────────────────────
    p_pre = subparsers.add_parser(
        "preflight",
        help="Run preflight checks against a backend",
        description="Verify topology, calibration, cost ceiling, T1/T2, readout error, and gate support.",
    )
    p_pre.add_argument("--n-qubits", "-N", type=int, default=10, help="System size (default: 10)")
    p_pre.add_argument("--shots", type=int, default=16384, help="Shots (default: 16384)")
    p_pre.add_argument("--n-layouts", type=int, default=3, help="Layouts (default: 3)")
    p_pre.add_argument(
        "--amplifier",
        type=str,
        default="pea",
        choices=["pea", "gate_folding", "adaptive"],
        help="Amplifier (default: pea)",
    )
    p_pre.add_argument(
        "--real", action="store_true", help="Use real hardware (requires IBM credentials)"
    )
    p_pre.add_argument("--json", action="store_true", help="Output as JSON")
    p_pre.set_defaults(func=cmd_preflight)

    # ── rehearsal ────────────────────────────────────────────────────────────
    p_reh = subparsers.add_parser(
        "rehearsal",
        help="Run the full hardware rehearsal (9 sections)",
        description="Execute HardwareRehearsalV2 — validates the complete deployment pipeline on FakeTorino.",
    )
    p_reh.add_argument("--n-qubits", "-N", type=int, default=None, help="System size (default: 10)")
    p_reh.add_argument(
        "--topology",
        type=str,
        default=None,
        choices=["chain_1d", "ladder", "heavy_hex"],
        help="Topology (default: heavy_hex)",
    )
    p_reh.add_argument(
        "--amplifier",
        type=str,
        default=None,
        choices=["pea", "gate_folding", "adaptive"],
        help="ZNE amplifier (default: pea)",
    )
    p_reh.add_argument("--shots", type=int, default=None, help="Shots (default: 16384)")
    p_reh.add_argument(
        "--section", type=int, nargs="+", default=None, help="Run only these sections"
    )
    p_reh.add_argument(
        "--run-preflight", action="store_true", help="Include Section 0 (backend preflight)"
    )
    p_reh.add_argument("--dry-run", action="store_true", help="List sections without executing")
    p_reh.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    p_reh.add_argument("--skip-preflight", action="store_true", help="Skip structural preflight")
    p_reh.add_argument("--p-layers", type=int, default=None, help="HVA layers (default: 1)")
    p_reh.add_argument(
        "--h-test", type=float, nargs="+", default=None, help="Override test h-values"
    )
    p_reh.add_argument(
        "--h-train", type=float, nargs="+", default=None, help="Override training h-values"
    )
    p_reh.add_argument("--vqe-restarts", type=int, default=None, help="VQE restarts")
    p_reh.add_argument("--mpnn-epochs", type=int, default=None, help="MPNN epochs")
    p_reh.set_defaults(func=cmd_rehearsal)

    # ── analyze ──────────────────────────────────────────────────────────────
    p_ana = subparsers.add_parser(
        "analyze",
        help="Analyze rehearsal results (GO/NO-GO verdict)",
        description="Parse rehearsal JSON results and report per-section metrics with hardware readiness.",
    )
    p_ana.add_argument("--all", action="store_true", help="Analyze all runs (not just latest)")
    p_ana.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    p_ana.add_argument(
        "--threshold", type=float, default=None, help="ΔE/gap threshold (default: 0.05)"
    )
    p_ana.add_argument(
        "--section-filter", type=int, nargs="+", default=None, help="Only show these sections"
    )
    p_ana.add_argument("--exp-dir", type=str, default=None, help="Experiment directory override")
    p_ana.set_defaults(func=cmd_analyze)

    return parser


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """Unified hardware CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        exit_code = args.func(args)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        exit_code = 130
    except Exception as exc:
        print(f"\n  ERROR: {exc}", file=sys.stderr)
        if "--verbose" in sys.argv or "-v" in sys.argv:
            import traceback

            traceback.print_exc()
        exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
