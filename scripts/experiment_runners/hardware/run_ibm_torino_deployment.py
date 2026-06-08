#!/usr/bin/env python3
"""IBM Torino QPU Deployment — Tiered Execution Plan.

Executes the GNN-HVA framework on real IBM Torino quantum hardware.
Implements a tiered strategy where each tier only proceeds if the
previous one passes:

    Tier 0: Smoke test (h=4.0 single point, must-pass)
    Tier 1: Core validation (h=[4.0, 3.25, 3.0, 2.5], primary thesis data)
    Tier 2: Statistical validation (Tier 1 × 3 seeds, robustness evidence)
    Tier 3: Cross-model extension (tfim_longitudinal at g=0.3)

Prerequisites:
    - IBM credentials: export IBM_KEY="..." and IBM_INSTANCE_CRN="..."
    - Trained MPNN checkpoint (auto-generated if not present)
    - run_hardware_rehearsal_v2.py must have passed (green light)

Usage:
    # Full deployment (Tier 0 → 1 → 2 → 3, auto-advancing)
    python scripts/experiment_runners/hardware/run_ibm_torino_deployment.py

    # Single tier only
    python scripts/experiment_runners/hardware/run_ibm_torino_deployment.py --tier 0
    python scripts/experiment_runners/hardware/run_ibm_torino_deployment.py --tier 1

    # Dry run (preflight + cost estimate only, no QPU usage)
    python scripts/experiment_runners/hardware/run_ibm_torino_deployment.py --dry-run

    # Custom configuration
    python scripts/experiment_runners/hardware/run_ibm_torino_deployment.py \
        --shots 32768 --zne-amplifier adaptive --tier 1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from qmbp_simulation.framework.runner_base import resolve_project_root

_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qmbp_simulation import (
    ClassicalSolver,
    HamiltonianBuilder,
    HVACircuitBuilder,
    make_lattice,
)
from qmbp_simulation.execution.hardware import HardwareBackend, HardwareConfig
from qmbp_simulation.execution.hardware.preflight import estimate_qpu_cost
from qmbp_simulation.framework.logging import StructuredLogger
from qmbp_simulation.predictors import load_mpnn_checkpoint
from qmbp_simulation.utils.helpers import json_dump

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration — aligned with project-status.md validated parameters
# ═══════════════════════════════════════════════════════════════════════════════

TOPOLOGY = "heavy_hex"
N_QUBITS = 10
P_LAYERS = 1
MODEL = "tfim"
BACKEND_NAME = "ibm_torino"

# h-values for each tier
TIER_0_H = [4.0]
TIER_1_H = [4.0, 3.25, 3.0, 2.5]
TIER_2_SEEDS = [42, 43, 44]
TIER_3_MODEL = "tfim_longitudinal"
TIER_3_G = 0.3
TIER_3_H = [3.25]

# VQE/MPNN training config (for generating predictions if no checkpoint)
H_TRAIN_GRID = [4.5, 4.25, 4.0, 3.75, 3.5, 3.25, 3.0]
VQE_RESTARTS = 1
VQE_MAXITER = 500
MPNN_HIDDEN = 128
MPNN_EPOCHS = 6000
MPNN_LR = 1e-3
MPNN_PATIENCE = 500

# Hardware execution config
DEFAULT_SHOTS = 16384
DEFAULT_N_LAYOUTS = 3
DEFAULT_AMPLIFIER = "pea"

# Success thresholds
DE_GAP_THRESHOLD = 0.05
SMOKE_ABORT_THRESHOLD = 0.10
ZNE_R2_THRESHOLD = 0.80


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════


def check_credentials() -> tuple[str, str]:
    """Verify IBM credentials are available. Raises if missing."""
    key = os.environ.get("IBM_KEY")
    crn = os.environ.get("IBM_INSTANCE_CRN")
    if not key:
        raise OSError(
            "IBM_KEY environment variable not set.\n"
            "Export your IBM Quantum API token:\n"
            "  export IBM_KEY='your_token_here'"
        )
    if not crn:
        raise OSError(
            "IBM_INSTANCE_CRN environment variable not set.\n"
            "Export your IBM instance CRN:\n"
            "  export IBM_INSTANCE_CRN='crn:v1:bluemix:public:...'"
        )
    return key, crn


def build_hardware_config(
    shots: int = DEFAULT_SHOTS,
    n_layouts: int = DEFAULT_N_LAYOUTS,
    amplifier: str = DEFAULT_AMPLIFIER,
    output_dir: str = "results/hardware",
) -> HardwareConfig:
    """Build HardwareConfig for real QPU execution."""
    from qmbp_simulation.execution.backends import MitigationOptions

    return HardwareConfig(
        backend_name=BACKEND_NAME,
        mode="hardware",
        n_qubits=N_QUBITS,
        shots=shots,
        n_layouts=n_layouts,
        n_candidates=40,
        max_ces=0.5,
        optimization_level=2,
        layout_seed=42,
        job_timeout_s=900,  # 15 min per job (generous for PEA noise learning)
        max_retries=3,
        retry_delay_s=60,
        max_total_shots=10_000_000,
        spsa_enabled=True,
        spsa_threshold=DE_GAP_THRESHOLD,
        output_dir=output_dir,
        mitigation=MitigationOptions(
            dd_enabled=True,
            trex_enabled=True,
            twirling_enabled=True,
            zne_enabled=True,
            zne_amplifier=amplifier,
            num_randomizations=32,
            shots_per_randomization=128,
        ),
    )


def prepare_mpnn_predictions(
    h_values: list[float],
    lattice,
    seed: int = 42,
) -> tuple[dict[float, np.ndarray], dict[float, float], dict[float, float]]:
    """Generate MPNN predictions for each h-value.

    If a trained MPNN checkpoint exists, loads it. Otherwise trains one
    using the standard training grid on heavy_hex N=10 p=1.

    Returns
    -------
    params_per_h : dict mapping h → θ_pred
    e_exact_per_h : dict mapping h → exact ground state energy
    gap_per_h : dict mapping h → spectral gap
    """
    from qmbp_simulation import VQEConfig, VQEOptimizer
    from qmbp_simulation.predictors import build_graph_dataset, train_mpnn

    solver = ClassicalSolver()
    builder = HamiltonianBuilder()
    lattice_cfg = lattice

    # Get exact energies and gaps for all test h-values
    e_exact_per_h: dict[float, float] = {}
    gap_per_h: dict[float, float] = {}
    for h in set(h_values + H_TRAIN_GRID):
        H = builder.build_tfim(lattice_cfg, h=h)
        exact = solver.solve(H)
        e_exact_per_h[h] = exact.energy
        gap_per_h[h] = exact.gap

    # Check for existing checkpoint
    ckpt_dir = _ROOT / "results" / "hardware" / "mpnn_checkpoints"
    ckpt_path = ckpt_dir / f"mpnn_heavy_hex_n{N_QUBITS}_p{P_LAYERS}_seed{seed}.pt"

    if ckpt_path.exists():
        logger.info(f"Loading MPNN checkpoint: {ckpt_path}")
        model = load_mpnn_checkpoint(ckpt_path)
    else:
        logger.info("No checkpoint found. Training MPNN from scratch...")
        # Phase 2: VQE to get θ_opt for training grid
        circuit_builder = HVACircuitBuilder()
        circuit, _ = circuit_builder.create(N_QUBITS, P_LAYERS, lattice_cfg)

        vqe_config = VQEConfig(
            n_restarts=VQE_RESTARTS,
            maxiter=VQE_MAXITER,
            seed=seed,
        )
        optimizer = VQEOptimizer(config=vqe_config)

        vqe_results = []
        for h in H_TRAIN_GRID:
            H = builder.build_tfim(lattice_cfg, h=h)
            result = optimizer.optimize(circuit, H)
            vqe_results.append({"h": h, "theta_opt": result.params, "energy": result.energy})

        # Phase 3: Train MPNN
        dataset = build_graph_dataset(vqe_results, lattice_cfg, model_name=MODEL)
        model = train_mpnn(
            dataset,
            hidden_dim=MPNN_HIDDEN,
            n_epochs=MPNN_EPOCHS,
            lr=MPNN_LR,
            patience=MPNN_PATIENCE,
            seed=seed,
        )
        # Save checkpoint
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        from qmbp_simulation.predictors import save_mpnn_checkpoint

        save_mpnn_checkpoint(model, ckpt_path)
        logger.info(f"MPNN checkpoint saved: {ckpt_path}")

    # Generate predictions for test h-values
    params_per_h: dict[float, np.ndarray] = {}
    for h in h_values:
        params_per_h[h] = model.predict(h, lattice_cfg)

    return params_per_h, e_exact_per_h, gap_per_h


# ═══════════════════════════════════════════════════════════════════════════════
# Tier execution functions
# ═══════════════════════════════════════════════════════════════════════════════


def run_tier_0(config: HardwareConfig, lattice, slogger: StructuredLogger) -> bool:
    """Tier 0: Smoke test at h=4.0.

    Returns True if PASS (proceed to Tier 1), False if FAIL (abort).
    """
    slogger.log("tier_0_start", data={"h_values": TIER_0_H})
    print("\n" + "=" * 70)
    print("TIER 0: SMOKE TEST (h=4.0, deep paramagnetic)")
    print("=" * 70)

    circuit_builder = HVACircuitBuilder()
    circuit, _ = circuit_builder.create(N_QUBITS, P_LAYERS, lattice)

    params_per_h, e_exact_per_h, gap_per_h = prepare_mpnn_predictions(TIER_0_H, lattice, seed=42)

    backend = HardwareBackend(config=config)

    # Single point execution
    result = backend.run_deployment(
        circuit,
        HamiltonianBuilder().build_tfim(lattice, h=4.0),
        params_per_h[4.0],
        h_value=4.0,
        e_exact=e_exact_per_h[4.0],
        gap=gap_per_h[4.0],
        expected_label="paramagnetic",
    )

    print(
        f"\n  h=4.0: ΔE/gap={result.delta_e_gap:.4f} | "
        f"phase={result.phase_label} | R²={result.zne_r2:.3f} | "
        f"verdict={result.verdict}"
    )
    print(f"  Strategy: {result.mitigation_strategy}")
    if result.layout_std is not None:
        print(f"  Layout std: {result.layout_std:.6f}")

    passed = result.verdict == "PASS"
    slogger.log(
        "tier_0_result",
        data={
            "delta_e_gap": result.delta_e_gap,
            "verdict": result.verdict,
            "phase_label": result.phase_label,
            "zne_r2": result.zne_r2,
            "passed": passed,
        },
    )

    if passed:
        print("\n  ✅ TIER 0 PASSED — safe to proceed to Tier 1")
    else:
        print(f"\n  ❌ TIER 0 FAILED — {result.verdict_reason}")
        print("  ACTION: Check calibration, retry during off-peak, or debug.")
    return passed


def run_tier_1(config: HardwareConfig, lattice, slogger: StructuredLogger) -> dict:
    """Tier 1: Core validation (4 h-points).

    Returns summary dict with pass count and results.
    """
    slogger.log("tier_1_start", data={"h_values": TIER_1_H})
    print("\n" + "=" * 70)
    print("TIER 1: CORE VALIDATION (4 h-points)")
    print("=" * 70)

    circuit_builder = HVACircuitBuilder()
    circuit, _ = circuit_builder.create(N_QUBITS, P_LAYERS, lattice)

    params_per_h, e_exact_per_h, gap_per_h = prepare_mpnn_predictions(TIER_1_H, lattice, seed=42)

    backend = HardwareBackend(config=config)
    results = backend.run_h_sweep(
        circuit,
        hamiltonian_builder=lambda h: HamiltonianBuilder().build_tfim(lattice, h=h),
        h_values=TIER_1_H,
        params_per_h=params_per_h,
        e_exact_per_h=e_exact_per_h,
        gap_per_h=gap_per_h,
    )

    print("\n  Results:")
    print(f"  {'h':>5} | {'ΔE/gap':>8} | {'Phase':>12} | {'R²':>5} | {'Verdict':>12}")
    print(f"  {'-' * 5}-+-{'-' * 8}-+-{'-' * 12}-+-{'-' * 5}-+-{'-' * 12}")
    n_pass = 0
    for r in results:
        v_mark = "✅" if r.verdict == "PASS" else "❌"
        print(
            f"  {r.h_value:5.2f} | {r.delta_e_gap:8.4f} | {r.phase_label:>12} | "
            f"{r.zne_r2:5.3f} | {v_mark} {r.verdict}"
        )
        if r.verdict == "PASS":
            n_pass += 1

    summary = {
        "n_pass": n_pass,
        "n_total": len(results),
        "pass_rate": n_pass / len(results) if results else 0,
        "results": [
            {
                "h": r.h_value,
                "delta_e_gap": r.delta_e_gap,
                "verdict": r.verdict,
                "phase": r.phase_label,
                "r2": r.zne_r2,
            }
            for r in results
        ],
    }

    slogger.log("tier_1_result", data=summary)
    print(f"\n  Tier 1: {n_pass}/{len(results)} PASS ({summary['pass_rate']:.0%})")

    if n_pass >= 3:
        print("  ✅ TIER 1 SUCCESS — thesis Table 5.23 data acquired")
    elif n_pass >= 2:
        print("  ⚠️  TIER 1 PARTIAL — 2/4 pass, investigate boundary points")
    else:
        print("  ❌ TIER 1 INSUFFICIENT — check calibration or methodology")

    return summary


def run_tier_2(config: HardwareConfig, lattice, slogger: StructuredLogger) -> dict:
    """Tier 2: Statistical validation (3 seeds × 4 h-points).

    Returns summary with per-seed and aggregated statistics.
    """
    slogger.log("tier_2_start", data={"seeds": TIER_2_SEEDS, "h_values": TIER_1_H})
    print("\n" + "=" * 70)
    print("TIER 2: STATISTICAL VALIDATION (3 seeds × 4 h-points)")
    print("=" * 70)

    circuit_builder = HVACircuitBuilder()
    circuit, _ = circuit_builder.create(N_QUBITS, P_LAYERS, lattice)

    all_results = {}
    for seed in TIER_2_SEEDS:
        print(f"\n  --- Seed {seed} ---")
        params_per_h, e_exact_per_h, gap_per_h = prepare_mpnn_predictions(
            TIER_1_H, lattice, seed=seed
        )

        # Use different layout_seed per MPNN seed for diversity
        seed_config = HardwareConfig(
            backend_name=config.backend_name,
            mode=config.mode,
            n_qubits=config.n_qubits,
            shots=config.shots,
            n_layouts=config.n_layouts,
            n_candidates=config.n_candidates,
            max_ces=config.max_ces,
            optimization_level=config.optimization_level,
            layout_seed=seed,
            job_timeout_s=config.job_timeout_s,
            max_retries=config.max_retries,
            retry_delay_s=config.retry_delay_s,
            max_total_shots=config.max_total_shots,
            spsa_enabled=config.spsa_enabled,
            spsa_threshold=config.spsa_threshold,
            output_dir=config.output_dir,
            mitigation=config.mitigation,
        )
        backend = HardwareBackend(config=seed_config)
        results = backend.run_h_sweep(
            circuit,
            hamiltonian_builder=lambda h: HamiltonianBuilder().build_tfim(lattice, h=h),
            h_values=TIER_1_H,
            params_per_h=params_per_h,
            e_exact_per_h=e_exact_per_h,
            gap_per_h=gap_per_h,
        )
        all_results[seed] = results
        n_pass = sum(1 for r in results if r.verdict == "PASS")
        print(f"    Seed {seed}: {n_pass}/{len(results)} PASS")

    # Aggregate statistics
    all_de_gaps = []
    all_verdicts = []
    for seed, results in all_results.items():
        for r in results:
            all_de_gaps.append(r.delta_e_gap)
            all_verdicts.append(r.verdict == "PASS")

    summary = {
        "total_evaluations": len(all_de_gaps),
        "total_pass": sum(all_verdicts),
        "pass_rate": sum(all_verdicts) / len(all_verdicts) if all_verdicts else 0,
        "mean_de_gap": float(np.mean(all_de_gaps)),
        "std_de_gap": float(np.std(all_de_gaps, ddof=1)),
        "max_de_gap": float(np.max(all_de_gaps)),
        "per_seed": {
            seed: {
                "n_pass": sum(1 for r in res if r.verdict == "PASS"),
                "mean_de_gap": float(np.mean([r.delta_e_gap for r in res])),
            }
            for seed, res in all_results.items()
        },
    }

    slogger.log("tier_2_result", data=summary)
    print(
        f"\n  Tier 2 Aggregate: {summary['total_pass']}/{summary['total_evaluations']} "
        f"PASS ({summary['pass_rate']:.0%})"
    )
    print(f"  Mean ΔE/gap: {summary['mean_de_gap']:.4f} ± {summary['std_de_gap']:.4f}")

    if summary["pass_rate"] >= 0.75:
        print("  ✅ TIER 2 SUCCESS — robust hardware validation confirmed")
    else:
        print("  ⚠️  TIER 2 PARTIAL — seed-dependent behavior observed")

    return summary


def run_tier_3(config: HardwareConfig, lattice, slogger: StructuredLogger) -> dict:
    """Tier 3: Cross-model extension (tfim_longitudinal at g=0.3).

    Demonstrates framework generalization on real hardware.
    """
    slogger.log(
        "tier_3_start",
        data={
            "model": TIER_3_MODEL,
            "g": TIER_3_G,
            "h_values": TIER_3_H,
        },
    )
    print("\n" + "=" * 70)
    print(f"TIER 3: CROSS-MODEL ({TIER_3_MODEL}, g={TIER_3_G})")
    print("=" * 70)

    from qmbp_simulation.models.model_registry import get_model_spec

    spec = get_model_spec(TIER_3_MODEL).with_params(g=TIER_3_G)
    circuit, _ = spec.create_circuit(N_QUBITS, P_LAYERS, lattice, **spec.circuit_kwargs)

    solver = ClassicalSolver()
    builder = HamiltonianBuilder()
    h = TIER_3_H[0]
    H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs, h=h)
    exact = solver.solve(H)

    # Use MPNN from standard TFIM as warm-start (validated in E4b)
    params_tfim, _, _ = prepare_mpnn_predictions([h], lattice, seed=42)
    # Extend params for longitudinal model (add g-dependent parameter)
    params = np.append(params_tfim[h], [0.1])  # Initial RZ rotation for g-term

    backend = HardwareBackend(config=config)
    result = backend.run_deployment(
        circuit,
        H,
        params,
        h_value=h,
        e_exact=exact.energy,
        gap=exact.gap,
        expected_label="paramagnetic",
    )

    print(f"\n  {TIER_3_MODEL} (g={TIER_3_G}) at h={h}:")
    print(
        f"  ΔE/gap={result.delta_e_gap:.4f} | phase={result.phase_label} | "
        f"R²={result.zne_r2:.3f} | verdict={result.verdict}"
    )

    summary = {
        "model": TIER_3_MODEL,
        "g": TIER_3_G,
        "h": h,
        "delta_e_gap": result.delta_e_gap,
        "verdict": result.verdict,
        "phase_label": result.phase_label,
        "zne_r2": result.zne_r2,
    }
    slogger.log("tier_3_result", data=summary)

    if result.verdict == "PASS":
        print("  ✅ TIER 3 SUCCESS — model extensibility validated on hardware")
    else:
        print(f"  ❌ TIER 3 FAILED — {result.verdict_reason}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IBM Torino QPU Deployment — GNN-HVA Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[0, 1, 2, 3],
        default=None,
        help="Execute only a specific tier (default: all, auto-advancing)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only run preflight + cost estimate, no QPU usage"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
        help=f"Shots per circuit (default: {DEFAULT_SHOTS})",
    )
    parser.add_argument(
        "--n-layouts",
        type=int,
        default=DEFAULT_N_LAYOUTS,
        help=f"Number of layouts (default: {DEFAULT_N_LAYOUTS})",
    )
    parser.add_argument(
        "--zne-amplifier",
        choices=["pea", "gate_folding", "adaptive"],
        default=DEFAULT_AMPLIFIER,
        help=f"ZNE amplifier strategy (default: {DEFAULT_AMPLIFIER})",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/hardware", help="Output directory for results"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     IBM Torino QPU Deployment — GNN-HVA Framework              ║")
    print("║     Thesis: Hybrid GNN-HVA for Topological Phase Detection     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"\n  Timestamp: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Config: N={N_QUBITS}, p={P_LAYERS}, topology={TOPOLOGY}")
    print(f"  Amplifier: {args.zne_amplifier}, Shots: {args.shots}, Layouts: {args.n_layouts}")

    # ─── Credential check ──────────────────────────────────────────────────
    try:
        key, crn = check_credentials()
        print(f"  IBM Key: {'*' * 8}...{key[-4:]}")
        print(f"  Instance: ...{crn[-20:]}")
    except OSError as e:
        print(f"\n  ❌ {e}")
        sys.exit(1)

    # ─── Build config ──────────────────────────────────────────────────────
    output_dir = Path(args.output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = build_hardware_config(
        shots=args.shots,
        n_layouts=args.n_layouts,
        amplifier=args.zne_amplifier,
        output_dir=str(output_dir),
    )

    # ─── QPU cost estimate ─────────────────────────────────────────────────
    n_h_total = len(TIER_1_H) * (1 + len(TIER_2_SEEDS)) + len(TIER_3_H) + 1
    cost = estimate_qpu_cost(config, n_h_points=n_h_total)
    print("\n  Estimated QPU budget:")
    print(f"    Total h-points: {n_h_total}")
    print(f"    Circuits/h-point: {cost.circuits_per_h}")
    print(f"    Total shots: {cost.total_shots:,}")
    print(f"    Est. QPU time: {cost.est_total_s / 60:.1f} min")
    print(f"    Fits per job: {'✅' if cost.fits_per_job else '❌'}")

    if args.dry_run:
        print("\n  [DRY RUN] No QPU jobs submitted. Exiting.")
        # Still run preflight on fake_backend for validation
        fake_config = HardwareConfig(
            mode="fake_backend",
            n_qubits=N_QUBITS,
            shots=args.shots,
            n_layouts=args.n_layouts,
            output_dir=str(output_dir),
        )
        lattice = make_lattice(TOPOLOGY, N_QUBITS)
        backend = HardwareBackend(config=fake_config)
        preflight = backend.run_preflight()
        print(f"  Preflight (fake): abort={preflight.get('abort', False)}")
        if preflight.get("abort"):
            print(f"  Reason: {preflight.get('abort_reason')}")
        sys.exit(0)

    # ─── Initialize structured logger ─────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    slogger = StructuredLogger("ibm_torino_deployment")
    lattice = make_lattice(TOPOLOGY, N_QUBITS)

    # ─── Tier execution ────────────────────────────────────────────────────
    execution_summary = {
        "start_time": datetime.now(UTC).isoformat(),
        "config": {
            "topology": TOPOLOGY,
            "n_qubits": N_QUBITS,
            "p_layers": P_LAYERS,
            "shots": args.shots,
            "amplifier": args.zne_amplifier,
        },
        "tiers": {},
    }

    tiers_to_run = [args.tier] if args.tier is not None else [0, 1, 2, 3]

    for tier in tiers_to_run:
        if tier == 0:
            passed = run_tier_0(config, lattice, slogger)
            execution_summary["tiers"]["tier_0"] = {"passed": passed}
            if not passed and args.tier is None:
                print("\n  ⛔ Smoke test failed. Aborting remaining tiers.")
                break

        elif tier == 1:
            summary = run_tier_1(config, lattice, slogger)
            execution_summary["tiers"]["tier_1"] = summary
            if summary["pass_rate"] < 0.5 and args.tier is None:
                print("\n  ⛔ Tier 1 pass rate < 50%. Skipping Tier 2/3.")
                break

        elif tier == 2:
            summary = run_tier_2(config, lattice, slogger)
            execution_summary["tiers"]["tier_2"] = summary

        elif tier == 3:
            summary = run_tier_3(config, lattice, slogger)
            execution_summary["tiers"]["tier_3"] = summary

    # ─── Save execution summary ───────────────────────────────────────────
    execution_summary["end_time"] = datetime.now(UTC).isoformat()
    summary_path = output_dir / "execution_summary.json"
    json_dump(execution_summary, summary_path)
    print(f"\n  📄 Summary saved: {summary_path}")
    print("\n  Done.")


if __name__ == "__main__":
    main()
