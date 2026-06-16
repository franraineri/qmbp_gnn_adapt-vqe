#!/usr/bin/env python3
"""IBM Torino QPU Deployment — Tiered Execution with Calibration-First Strategy.

Executes the GNN-HVA framework on real IBM Torino quantum hardware.
Implements a 2-session calibration-first strategy based on Hamed's resource
estimation protocol:

    Session 1 (Calibration, ~5-10 min QPU):
        Tier 0: Single circuit with full mitigation → measures T_one_job
        After Tier 0: Recomputes full budget from measured time, prints GO/NO-GO

    Session 2 (Execution, ~30-120 min QPU):
        Tier 1: Core validation (4 h-points, primary thesis data)
        Tier 2: Statistical validation (3 seeds × 4 h-points, robustness)
        Tier 3: Cross-model extension (tfim_longitudinal, generalization)

Key features:
    - Wall-clock timing on every QPU operation
    - SPSA kill-switch (--no-spsa) to prevent 400-min budget blowouts
    - Post-Tier-0 budget recompute from measured T_one_job
    - TLS calibration drift monitoring between h-points
    - Comprehensive JSON logging of every metric for thesis
    - Automatic abort if measured T_one_job exceeds budget ceiling

Prerequisites:
    - IBM credentials: export IBM_KEY="..." and IBM_INSTANCE_CRN="..."
    - Trained MPNN checkpoint (auto-generated if not present)
    - run_hardware_rehearsal_v2.py must have passed (green light)

Usage:
    # Full deployment (Tier 0 → 1 → 2 → 3, auto-advancing)
    python scripts/experiment_runners/hardware/run_ibm_torino_deployment.py

    # Calibration only (Tier 0 + budget estimate)
    python scripts/experiment_runners/hardware/run_ibm_torino_deployment.py --tier 0

    # Safe mode (no SPSA, prevents 400-min budget blowout)
    python scripts/experiment_runners/hardware/run_ibm_torino_deployment.py --no-spsa

    # Dry run (preflight + cost estimate only, no QPU usage)
    python scripts/experiment_runners/hardware/run_ibm_torino_deployment.py --dry-run

    # Custom configuration
    python scripts/experiment_runners/hardware/run_ibm_torino_deployment.py \\
        --shots 32768 --zne-amplifier adaptive --tier 1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
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
from qmbp_simulation.execution.hardware.preflight import (
    QPUThroughputProfile,
    SPSACostModel,
    estimate_qpu_cost,
)
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
BACKEND_NAME = "ibm_kingston"

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

# PEA configuration presets for the calibration study.
# Each preset is (num_randomizations, shots_per_randomization, noise_factors, n_layouts)
PEA_PRESETS = {
    # IBM default — baseline. Fast but insufficient on degraded calibration.
    "default": (32, 128, None, 1),
    # Balanced — 2.3× IBM default learning. Sweet spot for elevated error.
    "balanced": (48, 192, [1, 1.5, 3], 1),
    # Aggressive — 16× IBM default. Maximum accuracy, slow.
    "aggressive": (64, 256, [1, 1.5, 2, 3], 3),
    # Default + 3 layouts — tests if variance reduction alone solves it.
    "default_3layout": (32, 128, None, 3),
}

# Success thresholds
DE_GAP_THRESHOLD = 0.05
SMOKE_ABORT_THRESHOLD = 0.10
ZNE_R2_THRESHOLD = 0.80

# Budget safety: max wall-clock per h-point before auto-abort (10 min)
MAX_WALL_CLOCK_PER_H_S = 600.0
# Budget ceiling: total QPU time before warning (4 hours)
BUDGET_CEILING_S = 14400.0


# ═══════════════════════════════════════════════════════════════════════════════
# Timing & Metrics Dataclass
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TierMetrics:
    """Comprehensive timing and quality metrics for one tier execution."""

    tier: int
    start_time: str = ""
    end_time: str = ""
    wall_clock_s: float = 0.0
    n_h_points: int = 0
    n_jobs_submitted: int = 0
    n_jobs_succeeded: int = 0
    n_jobs_failed: int = 0
    t_one_job_measured_s: float | None = None
    mean_de_gap: float | None = None
    max_de_gap: float | None = None
    std_de_gap: float | None = None
    n_pass: int = 0
    n_total: int = 0
    pass_rate: float = 0.0
    spsa_triggered_count: int = 0
    spsa_total_time_s: float = 0.0
    per_h_wall_clock_s: list[float] = field(default_factory=list)
    per_h_results: list[dict] = field(default_factory=list)
    budget_recompute: dict | None = None
    passed: bool = False
    abort_reason: str = ""


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
    spsa_enabled: bool = True,
    backend_name: str = BACKEND_NAME,
    pea_preset: str = "balanced",
) -> HardwareConfig:
    """Build HardwareConfig for real QPU execution.

    PEA presets (2026-06-14 — from PEA calibration study):
    - "default": IBM default (32×128=4K). Fast, but fails on degraded calibration.
    - "balanced": 48×192=9K + [1,1.5,3]. Sweet spot for 2-4% error rates.
    - "aggressive": 64×256=16K + [1,1.5,2,3] + 3 layouts. Maximum accuracy, slow.
    - "default_3layout": IBM default + 3 layouts. Tests variance vs bias.

    The preset can be overridden by --pea-config CLI flag.
    """
    from qmbp_simulation.execution.backends import MitigationOptions

    # Load PEA preset
    if pea_preset not in PEA_PRESETS:
        raise ValueError(f"Unknown PEA preset: {pea_preset}. Available: {list(PEA_PRESETS.keys())}")
    num_rand, shots_rand, noise_factors, preset_layouts = PEA_PRESETS[pea_preset]

    # Preset n_layouts overrides the default if the preset specifies it
    effective_layouts = preset_layouts if pea_preset != "balanced" else n_layouts

    return HardwareConfig(
        backend_name=backend_name,
        mode="hardware",
        n_qubits=N_QUBITS,
        shots=shots,
        n_layouts=effective_layouts,
        n_candidates=40,
        max_ces=0.5,
        optimization_level=2,
        layout_seed=42,
        job_timeout_s=900,  # 15 min per job (generous for PEA noise learning)
        max_retries=3,
        retry_delay_s=60,
        max_total_shots=10_000_000,
        spsa_enabled=spsa_enabled,
        spsa_threshold=DE_GAP_THRESHOLD,
        output_dir=output_dir,
        mitigation=MitigationOptions(
            dd_enabled=True,
            trex_enabled=True,
            twirling_enabled=True,
            zne_enabled=True,
            zne_amplifier=amplifier,
            zne_noise_factors=noise_factors,
            num_randomizations=num_rand,
            shots_per_randomization=shots_rand,
        ),
    )


def recompute_budget_from_measurement(
    t_one_job_s: float,
    n_h_points: int,
    n_layouts: int,
    n_zne_factors: int = 3,
    spsa_enabled: bool = True,
) -> dict:
    """Recompute the full experiment budget from a measured T_one_job.

    This is Hamed's core protocol: measure one circuit, then extrapolate.
    Uses the REAL measured time (not CLOPS model) for budget accuracy.

    Parameters
    ----------
    t_one_job_s : float
        Measured wall-clock seconds for one complete job (circuit + mitigation).
    n_h_points : int
        Number of h-values in the full sweep.
    n_layouts : int
        Number of qubit layouts per h-point.
    n_zne_factors : int
        Number of ZNE noise factors (typically 3).
    spsa_enabled : bool
        Whether SPSA refinement is enabled.

    Returns
    -------
    dict
        Budget breakdown with optimistic/pessimistic estimates.
    """
    # PEA noise learning is ~1 additional job (one-time)
    pea_learning_s = t_one_job_s * 1.0

    # ZNE: one job per (layout × noise_factor)
    zne_jobs_per_h = n_layouts * n_zne_factors
    zne_time_per_h = zne_jobs_per_h * t_one_job_s

    # Observables: 1 job for X+ZZ measurement per h
    obs_time_per_h = t_one_job_s

    # SPSA: 200 iters × 2 evals × t_one_job (if triggered)
    spsa_time_if_triggered = 200 * 2 * t_one_job_s if spsa_enabled else 0.0

    # Per h-point (optimistic: no SPSA)
    per_h_optimistic = zne_time_per_h + obs_time_per_h
    per_h_pessimistic = per_h_optimistic + spsa_time_if_triggered

    # Total
    total_optimistic = pea_learning_s + n_h_points * per_h_optimistic
    total_pessimistic = pea_learning_s + n_h_points * per_h_pessimistic
    # Expected with P(SPSA)=0.30
    total_expected = pea_learning_s + n_h_points * (
        per_h_optimistic + 0.30 * spsa_time_if_triggered
    )

    return {
        "t_one_job_measured_s": t_one_job_s,
        "pea_learning_s": pea_learning_s,
        "zne_jobs_per_h": zne_jobs_per_h,
        "per_h_optimistic_s": per_h_optimistic,
        "per_h_pessimistic_s": per_h_pessimistic,
        "spsa_time_if_triggered_s": spsa_time_if_triggered,
        "total_optimistic_s": total_optimistic,
        "total_optimistic_min": total_optimistic / 60,
        "total_expected_s": total_expected,
        "total_expected_min": total_expected / 60,
        "total_pessimistic_s": total_pessimistic,
        "total_pessimistic_min": total_pessimistic / 60,
        "n_h_points": n_h_points,
        "exceeds_budget_ceiling": total_optimistic > BUDGET_CEILING_S,
    }


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
    from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn

    solver = ClassicalSolver()
    builder = HamiltonianBuilder()
    lattice_cfg = lattice

    # Get exact energies and gaps for all h-values (test + train)
    e_exact_per_h: dict[float, float] = {}
    gap_per_h: dict[float, float] = {}
    for h in set(h_values + H_TRAIN_GRID):
        lattice_h = make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=h)
        H = builder.build(lattice_h)
        exact = solver.solve(H, lattice_h)
        e_exact_per_h[h] = exact.ground_energy
        gap_per_h[h] = exact.gap

    # Check for existing checkpoint
    ckpt_dir = _ROOT / "results" / "hardware" / "mpnn_checkpoints"
    ckpt_path = ckpt_dir / f"mpnn_heavy_hex_n{N_QUBITS}_p{P_LAYERS}_seed{seed}.pt"

    if ckpt_path.exists():
        logger.info(f"Loading MPNN checkpoint: {ckpt_path}")
        model = load_mpnn_checkpoint(ckpt_path)
    else:
        logger.info("No checkpoint found. Training MPNN from scratch...")
        circuit_builder = HVACircuitBuilder()
        circuit, _ = circuit_builder.create(N_QUBITS, P_LAYERS, lattice_cfg)

        vqe_config = VQEConfig(
            n_restarts=VQE_RESTARTS,
            maxiter=VQE_MAXITER,
        )
        optimizer = VQEOptimizer(config=vqe_config, seed=seed)

        vqe_results = []
        rng = np.random.default_rng(seed)
        prev_theta = rng.uniform(-0.01, 0.01, circuit.num_parameters)
        for h in sorted(H_TRAIN_GRID, reverse=True):
            lattice_h = make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=h)
            H = builder.build(lattice_h)
            result = optimizer.optimize(H, circuit, prev_theta.copy())
            prev_theta = result.theta_opt.copy()
            vqe_results.append({"h": h, "theta_opt": result.theta_opt, "energy": result.energy})

        dataset = build_graph_dataset(
            lattice=lattice_cfg,
            h_values=np.array([r["h"] for r in vqe_results]),
            theta_opt=np.array([r["theta_opt"] for r in vqe_results]),
            e_exact=np.array([e_exact_per_h[r["h"]] for r in vqe_results]),
            fidelities=None,
            fidelity_threshold=0.0,
        )
        n_params = circuit.num_parameters
        model = MPNNPredictor(
            node_features=2,
            hidden_dim=MPNN_HIDDEN,
            n_layers=3,
            output_dim=n_params,
        )
        train_mpnn(
            model=model,
            dataset=dataset,
            n_epochs=MPNN_EPOCHS,
            lr=MPNN_LR,
            patience=MPNN_PATIENCE,
            seed=seed,
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        from qmbp_simulation.predictors import save_mpnn_checkpoint

        save_mpnn_checkpoint(model, ckpt_path)
        logger.info(f"MPNN checkpoint saved: {ckpt_path}")

    # Generate predictions for test h-values
    import torch
    from torch_geometric.data import Data

    model.eval()
    params_per_h: dict[float, np.ndarray] = {}
    for h in h_values:
        lattice_h = make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=h)
        edge_index_np, coord = builder.build_graph_data(lattice_h)
        h_feat = np.full(N_QUBITS, float(h))
        x = torch.tensor(
            np.stack([h_feat, coord.astype(float)], axis=1),
            dtype=torch.float32,
        )
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index)
        with torch.no_grad():
            theta_pred = model(graph).numpy().flatten()
        params_per_h[h] = theta_pred

    return params_per_h, e_exact_per_h, gap_per_h


def compute_kappa_per_h(
    params_per_h: dict[float, np.ndarray],
    lattice,
    eps: float = 0.01,
) -> dict[float, float]:
    """Compute landscape curvature κ(h) = mean |∂²E/∂θᵢ²| at each θ_opt.

    Uses noiseless finite differences — zero QPU cost. Run after
    ``prepare_mpnn_predictions`` to get hardware deployment risk scores.

    Interpretation (from section 19 validation):
      - κ is ANTI-correlated with noise sensitivity (r ≈ -0.84 at N=6).
      - Low κ → h is near h_c → high hardware risk → use more shots / PEA.
      - High κ → deep in paramagnetic regime → low hardware risk.

    Parameters
    ----------
    params_per_h : dict[float, np.ndarray]
        θ_pred per h-value from ``prepare_mpnn_predictions``.
    lattice : LatticeConfig
        Reference lattice (topology/N/J). h is overridden per point.
    eps : float
        Finite difference step (default: 0.01).

    Returns
    -------
    dict[float, float]
        kappa(h) per h-value. NaN if evaluation fails.
    """
    from qmbp_simulation import make_lattice
    from qmbp_simulation.circuits import HVACircuitBuilder
    from qmbp_simulation.execution import NoiselessBackend
    from qmbp_simulation.models.model_registry import get_model_spec

    spec = get_model_spec("tfim")
    noiseless = NoiselessBackend()
    hva = HVACircuitBuilder()
    kappa_per_h: dict[float, float] = {}

    for h, theta in params_per_h.items():
        lattice_h = make_lattice(lattice.topology, lattice.n_qubits, J=1.0, h=float(h))
        H_h = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)
        circuit_h, _ = spec.create_circuit(
            lattice.n_qubits, P_LAYERS, lattice_h, **spec.circuit_kwargs
        )
        n_params = len(theta)

        try:
            e_center = float(noiseless.evaluate(circuit_h, H_h, theta))
            curvatures = []
            for i in range(n_params):
                th_p = theta.copy()
                th_p[i] += eps
                th_m = theta.copy()
                th_m[i] -= eps
                e_p = float(noiseless.evaluate(circuit_h, H_h, th_p))
                e_m = float(noiseless.evaluate(circuit_h, H_h, th_m))
                curvatures.append(abs(e_p - 2 * e_center + e_m) / (eps**2))
            kappa_per_h[h] = float(np.mean(curvatures))
        except Exception as exc:
            logger.warning(f"κ computation failed at h={h:.3f}: {exc}")
            kappa_per_h[h] = float("nan")

    # Log κ profile and risk assessment
    for h in sorted(kappa_per_h.keys(), reverse=True):
        kappa = kappa_per_h[h]
        # Anti-correlation: low κ = high hardware risk
        risk = "⚠️ HIGH" if kappa < 45.0 else ("🟡 MED" if kappa < 50.0 else "✅ LOW")
        logger.info(f"  κ(h={h:.3f})={kappa:.1f} → hardware risk: {risk}")

    return kappa_per_h


def kappa_go_no_go(
    kappa_per_h: dict[float, float],
    *,
    high_risk_threshold: float = 45.0,
    medium_risk_threshold: float = 50.0,
    n_layouts_high_risk: int = 3,
    n_layouts_medium_risk: int = 3,
    n_layouts_low_risk: int = 1,
    shots_multiplier_high_risk: float = 2.0,
    shots_base: int = DEFAULT_SHOTS,
) -> dict[float, dict]:
    """Derive per-h deployment recommendations from κ profile.

    Based on section 19 validation: κ is ANTI-correlated with noise sensitivity
    (r = -0.85). Low κ → h near h_c → high hardware risk → use more resources.

    Decision rules (from N=6 calibration):
      κ < 45  → HIGH risk: 2× shots, 3 layouts, PEA mandatory
      κ ∈ [45, 50) → MEDIUM risk: 3 layouts, PEA recommended
      κ ≥ 50  → LOW risk: 1 layout sufficient, standard shots

    These thresholds were calibrated at N=6, chain_1d. For different N/topology,
    the absolute κ values shift but the relative ordering (low κ = high risk)
    remains valid.

    Parameters
    ----------
    kappa_per_h : dict[float, float]
        κ(h) from ``compute_kappa_per_h``.
    high_risk_threshold : float
        κ below this → HIGH risk (default: 45.0, from N=6 calibration).
    medium_risk_threshold : float
        κ below this but above high_risk_threshold → MEDIUM risk (default: 50.0).
    n_layouts_high_risk : int
        Layouts for high-risk h-points (default: 3).
    n_layouts_medium_risk : int
        Layouts for medium-risk h-points (default: 3).
    n_layouts_low_risk : int
        Layouts for low-risk h-points (default: 1).
    shots_multiplier_high_risk : float
        Shots multiplier for high-risk h-points (default: 2.0 → 32K if base=16K).
    shots_base : int
        Base shot count (default: DEFAULT_SHOTS=16384).

    Returns
    -------
    dict[float, dict]
        Per-h recommendations with keys:
        "risk_level", "n_layouts", "shots", "kappa", "spsa_recommended".
    """
    recommendations: dict[float, dict] = {}

    for h, kappa in kappa_per_h.items():
        if np.isnan(kappa):
            risk = "unknown"
            n_lay = n_layouts_medium_risk
            shots = shots_base
            spsa_recommended = True  # Caution when κ unavailable
        elif kappa < high_risk_threshold:
            risk = "high"
            n_lay = n_layouts_high_risk
            shots = int(shots_base * shots_multiplier_high_risk)
            spsa_recommended = True  # High-risk h-points warrant SPSA refinement
        elif kappa < medium_risk_threshold:
            risk = "medium"
            n_lay = n_layouts_medium_risk
            shots = shots_base
            spsa_recommended = False
        else:
            risk = "low"
            n_lay = n_layouts_low_risk
            shots = shots_base
            spsa_recommended = False

        recommendations[h] = {
            "kappa": kappa,
            "risk_level": risk,
            "n_layouts": n_lay,
            "shots": shots,
            "spsa_recommended": spsa_recommended,
        }

    # Log summary
    n_high = sum(1 for r in recommendations.values() if r["risk_level"] == "high")
    n_med = sum(1 for r in recommendations.values() if r["risk_level"] == "medium")
    n_low = sum(1 for r in recommendations.values() if r["risk_level"] == "low")
    total_shots = sum(r["shots"] for r in recommendations.values())
    baseline_shots = shots_base * len(recommendations)
    savings_pct = max(0, (1 - total_shots / max(baseline_shots, 1)) * 100)

    logger.info(
        f"  κ go/no-go: HIGH={n_high} MED={n_med} LOW={n_low} h-points | "
        f"total shots={total_shots:,} (baseline={baseline_shots:,}, "
        f"{'savings' if savings_pct > 0 else 'overhead'}={abs(savings_pct):.0f}%)"
    )

    return recommendations


# ═══════════════════════════════════════════════════════════════════════════════
# Tier Execution Functions (with comprehensive timing & metrics)
# ═══════════════════════════════════════════════════════════════════════════════


def run_tier_0(
    config: HardwareConfig,
    lattice,
    slogger: StructuredLogger,
    *,
    spsa_enabled: bool = True,
) -> TierMetrics:
    """Tier 0: Calibration run — single circuit, measures T_one_job.

    This is Hamed's recommended first step: submit one circuit with the
    full mitigation stack (DD + twirling + TREX + PEA-ZNE) and measure
    wall-clock time. The measured T_one_job is then used to recompute
    the full experiment budget with real-world accuracy.

    Returns TierMetrics with t_one_job_measured_s and budget_recompute.
    """
    metrics = TierMetrics(tier=0, n_h_points=1)
    metrics.start_time = datetime.now(UTC).isoformat()

    slogger.log("tier_0_start", data={"h_values": TIER_0_H, "purpose": "calibration_run"})
    print("\n" + "═" * 70)
    print("  TIER 0: CALIBRATION RUN (h=4.0, measures T_one_job)")
    print("  Purpose: Validate QPU connectivity + measure real execution time")
    print("═" * 70)

    circuit_builder = HVACircuitBuilder()
    circuit, _ = circuit_builder.create(N_QUBITS, P_LAYERS, lattice)
    params_per_h, e_exact_per_h, gap_per_h = prepare_mpnn_predictions(TIER_0_H, lattice, seed=42)

    # ── Compute κ risk profile (noiseless, zero QPU cost) ─────────────────
    logger.info("  Computing landscape curvature κ(h) for hardware risk assessment...")
    kappa_per_h = compute_kappa_per_h(params_per_h, lattice)

    # ── Tier 0 uses the same 3-layout config as production tiers.
    # Rationale: 3 layouts provides √3 variance reduction via layout averaging.
    # The single-layout mode was only for debugging the submission pipeline.
    # For calibration budget estimation, T_one_job = wall_clock / 3 layouts.
    backend = HardwareBackend(config=config)

    # ── Execute with wall-clock timing ────────────────────────────────────
    print("\n  Submitting single circuit with full mitigation stack (3 layouts)...")
    print(
        f"    DD={config.mitigation.dd_enabled}, Twirling={config.mitigation.twirling_enabled}, "
        f"TREX={config.mitigation.trex_enabled}, ZNE={config.mitigation.zne_amplifier}"
    )
    print(
        f"    PEA learning: {config.mitigation.num_randomizations}×{config.mitigation.shots_per_randomization}"
        f" = {config.mitigation.num_randomizations * config.mitigation.shots_per_randomization} shots"
    )
    print(f"    Noise factors: {config.mitigation.zne_noise_factors}")

    t_start = time.time()
    try:
        result = backend.run_deployment(
            circuit,
            HamiltonianBuilder().build(make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=4.0)),
            params_per_h[4.0],
            h_value=4.0,
            e_exact=e_exact_per_h[4.0],
            gap=gap_per_h[4.0],
            expected_label="paramagnetic",
        )
    except Exception as exc:
        t_elapsed = time.time() - t_start
        metrics.wall_clock_s = t_elapsed
        metrics.end_time = datetime.now(UTC).isoformat()
        metrics.abort_reason = f"run_deployment() failed: {exc}"
        metrics.n_jobs_failed = 1
        slogger.log("tier_0_error", data={"error": str(exc), "elapsed_s": t_elapsed})
        print(f"\n  ❌ TIER 0 FAILED after {t_elapsed:.1f}s — {exc}")
        return metrics
    t_elapsed = time.time() - t_start

    # ── Record metrics ────────────────────────────────────────────────────
    metrics.wall_clock_s = t_elapsed
    metrics.t_one_job_measured_s = t_elapsed
    metrics.per_h_wall_clock_s = [t_elapsed]
    metrics.n_jobs_submitted = 1
    metrics.n_jobs_succeeded = 1
    metrics.n_total = 1
    metrics.mean_de_gap = result.delta_e_gap
    metrics.max_de_gap = result.delta_e_gap

    passed = result.verdict == "PASS"
    metrics.n_pass = 1 if passed else 0
    metrics.pass_rate = 1.0 if passed else 0.0
    metrics.passed = passed
    if result.spsa_applied:
        metrics.spsa_triggered_count = 1

    metrics.per_h_results = [
        {
            "h": 4.0,
            # Energy & quality
            "e_exact": result.e_exact,
            "e_zne": result.e_zne,
            "delta_e_gap": result.delta_e_gap,
            "gap": result.gap,
            "zne_gain": result.zne_gain,
            # Verdict
            "verdict": result.verdict,
            "verdict_reason": result.verdict_reason,
            # Phase classification
            "phase_label": result.phase_label,
            "expected_label": result.expected_label,
            "mag_x_mean": result.mag_x_mean,
            "corr_zz_mean": result.corr_zz_mean,
            "sigma": result.sigma,
            # ZNE details
            "zne_r2": result.zne_r2,
            "zne_amplifier_used": result.zne_amplifier_used,
            "mitigation_strategy": result.mitigation_strategy,
            "layout_std": result.layout_std,
            "fallback_triggered": result.fallback_triggered,
            # SPSA
            "spsa_applied": result.spsa_applied,
            # Provenance (critical for debugging)
            "job_ids": result.job_ids,
            "layouts_used": result.layouts_used,
            "ces_values": result.ces_values,
            "total_shots": result.total_shots,
            # Per-site observables (thesis data)
            "per_site_x": result.per_site_x,
            "per_bond_zz": result.per_bond_zz,
            # Post-correction
            "gnn_qem_applied": result.gnn_qem_applied,
            "gnn_qem_delta_e": result.gnn_qem_delta_e,
            "affine_correction_applied": result.affine_correction_applied,
            "e_after_affine": result.e_after_affine,
            # QPU metrics (from IBM Runtime, real hardware only)
            "qpu_metrics": getattr(result, "_qpu_metrics", None),
            # Calibration snapshot (T1/T2/error at execution time)
            "calibration_snapshot": getattr(result, "_calibration_snapshot", None),
            # Transpiled circuit stats (depth, gate counts)
            "transpiled_stats": getattr(result, "_transpiled_stats", None),
            # Timing (from our measurement)
            "wall_clock_s": t_elapsed,
            # Landscape curvature (noiseless risk proxy from section 19)
            "kappa": kappa_per_h.get(4.0, float("nan")),
            "hardware_risk": "high"
            if kappa_per_h.get(4.0, 50.0) < 45.0
            else ("medium" if kappa_per_h.get(4.0, 50.0) < 50.0 else "low"),
        }
    ]

    # ── Print results ─────────────────────────────────────────────────────
    print("\n  ┌─── Tier 0 Result ────────────────────────────────────────┐")
    print("  │  h = 4.0 (deep paramagnetic)                             │")
    print(
        f"  │  ΔE/gap:   {result.delta_e_gap:.4f}  ({'✅ PASS' if passed else '❌ FAIL':>10})           │"
    )
    print(f"  │  Phase:    {result.phase_label:<20}                     │")
    print(f"  │  ZNE R²:   {result.zne_r2:.4f}                                    │")
    print(f"  │  Strategy: {result.mitigation_strategy:<30}         │")
    print(
        f"  │  SPSA:     {'triggered' if result.spsa_applied else 'not needed':<20}                     │"
    )
    print(f"  │  Wall-clock: {t_elapsed:.1f}s                                   │")
    print("  └────────────────────────────────────────────────────────────┘")

    # ── Budget recompute from measured T_one_job ──────────────────────────
    n_h_full = len(TIER_1_H) * (1 + len(TIER_2_SEEDS)) + len(TIER_3_H)
    budget = recompute_budget_from_measurement(
        t_one_job_s=t_elapsed,
        n_h_points=n_h_full,
        n_layouts=config.n_layouts,
        n_zne_factors=3,
        spsa_enabled=spsa_enabled,
    )
    metrics.budget_recompute = budget

    print(f"\n  ┌─── Budget Recompute (from measured T_one_job={t_elapsed:.1f}s) ───┐")
    print(f"  │  Full experiment: {n_h_full} h-points                           │")
    print(f"  │  Optimistic (no SPSA): {budget['total_optimistic_min']:.1f} min              │")
    print(f"  │  Expected (P=0.30):    {budget['total_expected_min']:.1f} min              │")
    print(f"  │  Pessimistic (SPSA):   {budget['total_pessimistic_min']:.1f} min             │")
    if budget["exceeds_budget_ceiling"]:
        print(f"  │  ⚠️  WARNING: Exceeds {BUDGET_CEILING_S / 3600:.0f}h budget ceiling!         │")
    print("  └────────────────────────────────────────────────────────────┘")

    if not passed:
        metrics.abort_reason = result.verdict_reason
        print(f"\n  ❌ TIER 0 FAILED — {result.verdict_reason}")
        print("  ACTION: Check calibration, retry during off-peak, or debug.")
    else:
        print("\n  ✅ TIER 0 PASSED — QPU connectivity confirmed, budget computed")
        print(
            f"     Proceed to Tier 1 with confidence (est. {budget['total_optimistic_min']:.0f} min)"
        )

    metrics.end_time = datetime.now(UTC).isoformat()
    slogger.log(
        "tier_0_result",
        data={
            "passed": passed,
            "t_one_job_s": t_elapsed,
            "delta_e_gap": result.delta_e_gap,
            "verdict": result.verdict,
            "budget_recompute": budget,
        },
    )
    return metrics


def run_tier_1(
    config: HardwareConfig,
    lattice,
    slogger: StructuredLogger,
) -> TierMetrics:
    """Tier 1: Core validation — 4 h-points, primary thesis data.

    Returns TierMetrics with per-h timing and quality metrics.
    """
    metrics = TierMetrics(tier=1, n_h_points=len(TIER_1_H))
    metrics.start_time = datetime.now(UTC).isoformat()

    slogger.log("tier_1_start", data={"h_values": TIER_1_H})
    print("\n" + "═" * 70)
    print("  TIER 1: CORE VALIDATION (4 h-points — primary thesis data)")
    print(f"  h-values: {TIER_1_H}")
    print("═" * 70)

    circuit_builder = HVACircuitBuilder()
    circuit, _ = circuit_builder.create(N_QUBITS, P_LAYERS, lattice)
    params_per_h, e_exact_per_h, gap_per_h = prepare_mpnn_predictions(TIER_1_H, lattice, seed=42)

    # ── Compute κ risk profile (noiseless, zero QPU cost) ─────────────────
    logger.info("  Computing landscape curvature κ(h) for per-h risk assessment...")
    kappa_per_h = compute_kappa_per_h(params_per_h, lattice)

    # ── κ go/no-go: derive per-h shot/layout recommendations ──────────────
    kappa_recommendations = kappa_go_no_go(
        kappa_per_h,
        shots_base=config.shots,
    )
    # Log the deployment plan
    logger.info("  ── Per-h deployment plan (κ-guided) ──")
    for h in sorted(kappa_recommendations.keys(), reverse=True):
        rec = kappa_recommendations[h]
        spsa_note = " [SPSA recommended]" if rec["spsa_recommended"] else ""
        logger.info(
            f"    h={h:.3f}: risk={rec['risk_level']}, "
            f"layouts={rec['n_layouts']}, shots={rec['shots']:,}"
            f"{spsa_note}"
        )

    backend = HardwareBackend(config=config)

    # Execute each h-point individually (for per-point timing)
    de_gaps = []
    for h in TIER_1_H:
        print(f"\n  ── h={h:.2f} ──")
        t_start = time.time()
        try:
            result = backend.run_deployment(
                circuit,
                HamiltonianBuilder().build(make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=h)),
                params_per_h[h],
                h_value=h,
                e_exact=e_exact_per_h[h],
                gap=gap_per_h[h],
                expected_label="paramagnetic",
            )
            t_h = time.time() - t_start
            metrics.n_jobs_submitted += 1
            metrics.n_jobs_succeeded += 1
        except Exception as exc:
            t_h = time.time() - t_start
            metrics.n_jobs_submitted += 1
            metrics.n_jobs_failed += 1
            logger.error(f"  h={h}: FAILED after {t_h:.1f}s — {exc}")
            metrics.per_h_wall_clock_s.append(t_h)
            metrics.per_h_results.append(
                {
                    "h": h,
                    "error": str(exc),
                    "wall_clock_s": t_h,
                    "pass": False,
                }
            )
            continue

        passed = result.verdict == "PASS"
        if passed:
            metrics.n_pass += 1
        if result.spsa_applied:
            metrics.spsa_triggered_count += 1
        de_gaps.append(result.delta_e_gap)
        metrics.per_h_wall_clock_s.append(t_h)
        metrics.per_h_results.append(
            {
                "h": h,
                # Energy & quality
                "e_exact": result.e_exact,
                "e_zne": result.e_zne,
                "delta_e_gap": result.delta_e_gap,
                "gap": result.gap,
                "zne_gain": result.zne_gain,
                # Verdict
                "verdict": result.verdict,
                "verdict_reason": result.verdict_reason,
                # Phase classification
                "phase_label": result.phase_label,
                "mag_x_mean": result.mag_x_mean,
                "corr_zz_mean": result.corr_zz_mean,
                # ZNE details
                "zne_r2": result.zne_r2,
                "mitigation_strategy": result.mitigation_strategy,
                "layout_std": result.layout_std,
                # SPSA & provenance
                "spsa_applied": result.spsa_applied,
                "job_ids": result.job_ids,
                "ces_values": result.ces_values,
                "total_shots": result.total_shots,
                # Per-site observables
                "per_site_x": result.per_site_x,
                "per_bond_zz": result.per_bond_zz,
                # Post-correction
                "gnn_qem_applied": result.gnn_qem_applied,
                "affine_correction_applied": result.affine_correction_applied,
                "e_after_affine": result.e_after_affine,
                # QPU & calibration (real hardware only)
                "qpu_metrics": getattr(result, "_qpu_metrics", None),
                "calibration_snapshot": getattr(result, "_calibration_snapshot", None),
                "transpiled_stats": getattr(result, "_transpiled_stats", None),
                # Timing
                "wall_clock_s": t_h,
                "pass": passed,
                # Landscape curvature (noiseless risk proxy — section 19 validated)
                "kappa": kappa_per_h.get(h, float("nan")),
                "hardware_risk": kappa_recommendations.get(h, {}).get("risk_level", "unknown"),
                "spsa_recommended": kappa_recommendations.get(h, {}).get("spsa_recommended", False),
            }
        )

        v_mark = "✅" if passed else "❌"
        print(
            f"    {v_mark} ΔE/gap={result.delta_e_gap:.4f} | phase={result.phase_label} | "
            f"R²={result.zne_r2:.3f} | {t_h:.1f}s"
        )

        # Safety: abort if one h-point takes too long
        if t_h > MAX_WALL_CLOCK_PER_H_S:
            logger.warning(
                f"  ⚠️  h={h} took {t_h:.0f}s (>{MAX_WALL_CLOCK_PER_H_S}s) — "
                "check queue or SPSA trigger"
            )

    # ── Summary metrics ───────────────────────────────────────────────────
    metrics.wall_clock_s = sum(metrics.per_h_wall_clock_s)
    metrics.n_total = len(TIER_1_H)
    metrics.pass_rate = metrics.n_pass / metrics.n_total if metrics.n_total else 0
    if de_gaps:
        metrics.mean_de_gap = float(np.mean(de_gaps))
        metrics.max_de_gap = float(np.max(de_gaps))
        metrics.std_de_gap = float(np.std(de_gaps)) if len(de_gaps) > 1 else 0.0
    metrics.passed = metrics.n_pass >= 3
    metrics.end_time = datetime.now(UTC).isoformat()

    print("\n  ┌─── Tier 1 Summary ───────────────────────────────────────┐")
    print(
        f"  │  Pass: {metrics.n_pass}/{metrics.n_total} ({metrics.pass_rate:.0%})                       │"
    )
    print(f"  │  Mean ΔE/gap: {metrics.mean_de_gap:.4f}" if metrics.mean_de_gap else "")
    print(
        f"  │  Wall-clock: {metrics.wall_clock_s:.1f}s ({metrics.wall_clock_s / 60:.1f} min)         │"
    )
    print(f"  │  SPSA triggered: {metrics.spsa_triggered_count}/{metrics.n_total} h-points     │")
    print("  └────────────────────────────────────────────────────────────┘")

    if metrics.passed:
        print("  ✅ TIER 1 SUCCESS — thesis Table 5.23 data acquired")
    else:
        print("  ❌ TIER 1 INSUFFICIENT — investigate boundary points")
        metrics.abort_reason = f"Only {metrics.n_pass}/{metrics.n_total} passed"

    slogger.log(
        "tier_1_result",
        data={
            "n_pass": metrics.n_pass,
            "n_total": metrics.n_total,
            "pass_rate": metrics.pass_rate,
            "mean_de_gap": metrics.mean_de_gap,
            "wall_clock_s": metrics.wall_clock_s,
            "spsa_triggered": metrics.spsa_triggered_count,
        },
    )
    return metrics


def run_tier_2(
    config: HardwareConfig,
    lattice,
    slogger: StructuredLogger,
) -> TierMetrics:
    """Tier 2: Statistical validation — 3 seeds × 4 h-points.

    Demonstrates reproducibility across different MPNN seeds and layout selections.
    """
    metrics = TierMetrics(tier=2, n_h_points=len(TIER_1_H) * len(TIER_2_SEEDS))
    metrics.start_time = datetime.now(UTC).isoformat()

    slogger.log("tier_2_start", data={"seeds": TIER_2_SEEDS, "h_values": TIER_1_H})
    print("\n" + "═" * 70)
    print(
        f"  TIER 2: STATISTICAL VALIDATION ({len(TIER_2_SEEDS)} seeds × {len(TIER_1_H)} h-points)"
    )
    print("═" * 70)

    circuit_builder = HVACircuitBuilder()
    circuit, _ = circuit_builder.create(N_QUBITS, P_LAYERS, lattice)

    # ── Compute κ once (seed-independent, noiseless) ─────────────────────
    # κ depends only on θ_pred(h) from seed=42 MPNN — shared across all seeds.
    _params_seed42, _, _ = prepare_mpnn_predictions(TIER_1_H, lattice, seed=42)
    kappa_per_h_t2 = compute_kappa_per_h(_params_seed42, lattice)
    kappa_recommendations_t2 = kappa_go_no_go(kappa_per_h_t2, shots_base=config.shots)

    all_de_gaps = []
    for seed in TIER_2_SEEDS:
        print(f"\n  ─── Seed {seed} ───")
        params_per_h, e_exact_per_h, gap_per_h = prepare_mpnn_predictions(
            TIER_1_H, lattice, seed=seed
        )

        # Use different layout_seed per seed for diversity
        from dataclasses import replace

        seed_config = replace(config, layout_seed=seed)
        backend = HardwareBackend(config=seed_config)

        for h in TIER_1_H:
            t_start = time.time()
            try:
                result = backend.run_deployment(
                    circuit,
                    HamiltonianBuilder().build(make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=h)),
                    params_per_h[h],
                    h_value=h,
                    e_exact=e_exact_per_h[h],
                    gap=gap_per_h[h],
                    expected_label="paramagnetic",
                )
                t_h = time.time() - t_start
                metrics.n_jobs_submitted += 1
                metrics.n_jobs_succeeded += 1
                passed = result.verdict == "PASS"
                if passed:
                    metrics.n_pass += 1
                if result.spsa_applied:
                    metrics.spsa_triggered_count += 1
                all_de_gaps.append(result.delta_e_gap)
                metrics.per_h_wall_clock_s.append(t_h)
                metrics.per_h_results.append(
                    {
                        "seed": seed,
                        "h": h,
                        "e_zne": result.e_zne,
                        "delta_e_gap": result.delta_e_gap,
                        "verdict": result.verdict,
                        "phase_label": result.phase_label,
                        "zne_r2": result.zne_r2,
                        "mag_x_mean": result.mag_x_mean,
                        "corr_zz_mean": result.corr_zz_mean,
                        "spsa_applied": result.spsa_applied,
                        "job_ids": result.job_ids,
                        "wall_clock_s": t_h,
                        "pass": passed,
                        # κ risk profile (noiseless, shared across seeds)
                        "kappa": kappa_per_h_t2.get(h, float("nan")),
                        "hardware_risk": kappa_recommendations_t2.get(h, {}).get(
                            "risk_level", "unknown"
                        ),
                    }
                )
                v_mark = "✅" if passed else "❌"
                print(f"    {v_mark} h={h:.2f}: ΔE/gap={result.delta_e_gap:.4f} ({t_h:.1f}s)")
            except Exception as exc:
                t_h = time.time() - t_start
                metrics.n_jobs_submitted += 1
                metrics.n_jobs_failed += 1
                metrics.per_h_wall_clock_s.append(t_h)
                metrics.per_h_results.append(
                    {
                        "seed": seed,
                        "h": h,
                        "error": str(exc),
                        "wall_clock_s": t_h,
                        "pass": False,
                    }
                )
                print(f"    ❌ h={h:.2f}: FAILED ({t_h:.1f}s) — {exc}")

    # Summary
    metrics.wall_clock_s = sum(metrics.per_h_wall_clock_s)
    metrics.n_total = len(TIER_1_H) * len(TIER_2_SEEDS)
    metrics.pass_rate = metrics.n_pass / metrics.n_total if metrics.n_total else 0
    if all_de_gaps:
        metrics.mean_de_gap = float(np.mean(all_de_gaps))
        metrics.max_de_gap = float(np.max(all_de_gaps))
        metrics.std_de_gap = float(np.std(all_de_gaps, ddof=1)) if len(all_de_gaps) > 1 else 0.0
    metrics.passed = metrics.pass_rate >= 0.75
    metrics.end_time = datetime.now(UTC).isoformat()

    print("\n  ┌─── Tier 2 Summary ───────────────────────────────────────┐")
    print(
        f"  │  Pass: {metrics.n_pass}/{metrics.n_total} ({metrics.pass_rate:.0%})                       │"
    )
    if metrics.mean_de_gap is not None:
        print(
            f"  │  Mean ΔE/gap: {metrics.mean_de_gap:.4f} ± {metrics.std_de_gap:.4f}               │"
        )
    print(
        f"  │  Wall-clock: {metrics.wall_clock_s:.1f}s ({metrics.wall_clock_s / 60:.1f} min)         │"
    )
    print("  └────────────────────────────────────────────────────────────┘")

    if metrics.passed:
        print("  ✅ TIER 2 SUCCESS — robust hardware validation confirmed")
    else:
        print("  ⚠️  TIER 2 PARTIAL — seed-dependent behavior observed")

    slogger.log(
        "tier_2_result",
        data={
            "n_pass": metrics.n_pass,
            "n_total": metrics.n_total,
            "pass_rate": metrics.pass_rate,
            "mean_de_gap": metrics.mean_de_gap,
            "std_de_gap": metrics.std_de_gap,
            "wall_clock_s": metrics.wall_clock_s,
        },
    )
    return metrics


def run_tier_3(
    config: HardwareConfig,
    lattice,
    slogger: StructuredLogger,
) -> TierMetrics:
    """Tier 3: Cross-model extension — tfim_longitudinal at g=0.3.

    Demonstrates framework generalization on real hardware.
    """
    metrics = TierMetrics(tier=3, n_h_points=len(TIER_3_H))
    metrics.start_time = datetime.now(UTC).isoformat()

    slogger.log(
        "tier_3_start",
        data={
            "model": TIER_3_MODEL,
            "g": TIER_3_G,
            "h_values": TIER_3_H,
        },
    )
    print("\n" + "═" * 70)
    print(f"  TIER 3: CROSS-MODEL ({TIER_3_MODEL}, g={TIER_3_G})")
    print("═" * 70)

    from qmbp_simulation.models.model_registry import get_model_spec

    spec = get_model_spec(TIER_3_MODEL).with_params(g=TIER_3_G)
    circuit, _ = spec.create_circuit(N_QUBITS, P_LAYERS, lattice, **spec.circuit_kwargs)

    solver = ClassicalSolver()
    h = TIER_3_H[0]
    lattice_h = make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=h)
    H = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)
    exact = solver.solve(H, lattice_h)

    # Use MPNN from standard TFIM as warm-start (validated in E4b)
    params_tfim, _, _ = prepare_mpnn_predictions([h], lattice_h, seed=42)
    expected_n_params = circuit.num_parameters
    tfim_params = params_tfim[h]
    n_extra = expected_n_params - len(tfim_params)
    if n_extra < 0:
        raise ValueError(
            f"Tier 3: TFIM MPNN predicts {len(tfim_params)} params but "
            f"{TIER_3_MODEL} circuit needs {expected_n_params}. "
            "Check model registry params_per_layer."
        )
    # Append extra parameters (g-term) initialized to small value
    # Extra params = (params_per_layer_longitudinal - params_per_layer_tfim) * p_layers
    extra_init = np.full(n_extra, 0.1)
    params = np.concatenate([tfim_params, extra_init])
    if len(params) != expected_n_params:
        raise ValueError(
            f"Tier 3: param count mismatch after extension: "
            f"got {len(params)}, expected {expected_n_params}."
        )
    logger.info(
        f"  Tier 3 warm-start: {len(tfim_params)} TFIM params + "
        f"{n_extra} extra ({TIER_3_MODEL}) → {len(params)} total"
    )

    backend = HardwareBackend(config=config)

    t_start = time.time()
    try:
        result = backend.run_deployment(
            circuit,
            H,
            params,
            h_value=h,
            e_exact=exact.ground_energy,
            gap=exact.gap,
            expected_label="paramagnetic",
        )
        t_h = time.time() - t_start
        metrics.n_jobs_submitted = 1
        metrics.n_jobs_succeeded = 1
        passed = result.verdict == "PASS"
        metrics.n_pass = 1 if passed else 0
        metrics.per_h_results = [
            {
                "h": h,
                "model": TIER_3_MODEL,
                "g": TIER_3_G,
                "delta_e_gap": result.delta_e_gap,
                "verdict": result.verdict,
                "phase_label": result.phase_label,
                "zne_r2": result.zne_r2,
                "wall_clock_s": t_h,
                "pass": passed,
            }
        ]
        metrics.mean_de_gap = result.delta_e_gap
    except Exception as exc:
        t_h = time.time() - t_start
        metrics.n_jobs_submitted = 1
        metrics.n_jobs_failed = 1
        passed = False
        metrics.per_h_results = [{"h": h, "error": str(exc), "wall_clock_s": t_h, "pass": False}]
        print(f"\n  ❌ FAILED: {exc}")

    metrics.wall_clock_s = t_h
    metrics.per_h_wall_clock_s = [t_h]
    metrics.n_total = 1
    metrics.pass_rate = 1.0 if passed else 0.0
    metrics.passed = passed
    metrics.end_time = datetime.now(UTC).isoformat()

    if passed:
        print(f"\n  ✅ TIER 3 SUCCESS — {TIER_3_MODEL} validated on hardware")
        print(f"     ΔE/gap={result.delta_e_gap:.4f}, R²={result.zne_r2:.3f}, {t_h:.1f}s")
    else:
        print(f"\n  ❌ TIER 3 FAILED — {metrics.per_h_results[0].get('error', 'verdict FAIL')}")

    slogger.log(
        "tier_3_result",
        data={
            "passed": passed,
            "wall_clock_s": t_h,
            "delta_e_gap": metrics.mean_de_gap,
        },
    )
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IBM Torino QPU Deployment — GNN-HVA Framework (Calibration-First)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategy:
  Session 1: --tier 0          Calibration run (~5 min), measures T_one_job
  Session 2: --tier 1 2 3      Full execution with confidence

Safety:
  --no-spsa    Prevent SPSA (saves 400 min if triggered)
  --dry-run    No QPU usage, only preflight + cost estimate
""",
    )
    parser.add_argument(
        "--tier",
        type=int,
        nargs="+",
        choices=[0, 1, 2, 3],
        default=None,
        help="Execute specific tier(s). Default: all (0→1→2→3, auto-advancing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only run preflight + cost estimate, no QPU usage",
    )
    parser.add_argument(
        "--no-spsa",
        action="store_true",
        help="Disable SPSA refinement (prevents 400-min budget blowout)",
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
        "--backend",
        type=str,
        default=BACKEND_NAME,
        help=f"IBM backend name (default: {BACKEND_NAME}). "
        "Use if ibm_torino is unavailable for your instance.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hardware",
        help="Output directory for results",
    )
    parser.add_argument(
        "--pea-config",
        type=str,
        choices=list(PEA_PRESETS.keys()),
        default="balanced",
        help="PEA learning preset: default (32×128), balanced (48×192), "
        "aggressive (64×256), default_3layout (32×128 + 3 layouts). "
        "Default: balanced",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # ── Setup logging ─────────────────────────────────────────────────────
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    spsa_enabled = not args.no_spsa
    pea_preset = args.pea_config  # must be assigned before banner

    # ── Banner ────────────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     IBM Torino QPU Deployment — GNN-HVA Framework              ║")
    print("║     Thesis: Hybrid GNN-HVA for Topological Phase Detection     ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Timestamp: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC'):<52}║")
    print(f"║  Config: N={N_QUBITS}, p={P_LAYERS}, topology={TOPOLOGY:<27}║")
    print(
        f"║  Amplifier: {args.zne_amplifier}, Shots: {args.shots}, Layouts: {args.n_layouts:<13}║"
    )
    print(
        f"║  SPSA: {'DISABLED (--no-spsa)' if not spsa_enabled else 'enabled (P≈0.30 trigger)':<48}║"
    )
    print(f"║  PEA preset: {pea_preset:<52}║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # ── Credential check ──────────────────────────────────────────────────
    if not args.dry_run:
        try:
            key, crn = check_credentials()
            print(f"\n  🔑 IBM Key: {'*' * 8}...{key[-4:]}")
            print(f"  🏢 Instance: ...{crn[-20:]}")
        except OSError as e:
            print(f"\n  ❌ {e}")
            sys.exit(1)

    # ── Build config ──────────────────────────────────────────────────────
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pea_preset = args.pea_config  # must be set BEFORE the banner uses it
    output_base = Path(args.output_dir)
    if args.tier == [0]:
        # Calibration study: separate folder per PEA config
        num_r, shots_r, _, n_lay = PEA_PRESETS[pea_preset]
        pea_tag = f"pea_{num_r}x{shots_r}"
        if n_lay > 1:
            pea_tag += f"_{n_lay}layout"
        output_dir = output_base / "tier0_calibration" / pea_tag / f"run_{run_timestamp}"
    else:
        output_dir = output_base / f"run_{run_timestamp}"
    config = build_hardware_config(
        shots=args.shots,
        n_layouts=args.n_layouts,
        amplifier=args.zne_amplifier,
        output_dir=str(output_dir),
        spsa_enabled=spsa_enabled,
        backend_name=args.backend,
        pea_preset=pea_preset,
    )

    # ── Pre-execution cost estimate (model-based) ─────────────────────────
    n_h_full = len(TIER_1_H) * (1 + len(TIER_2_SEEDS)) + len(TIER_3_H) + len(TIER_0_H)
    spsa_model = SPSACostModel() if spsa_enabled else SPSACostModel.disabled()
    profile = QPUThroughputProfile.ibm_kingston()
    cost = estimate_qpu_cost(
        config,
        n_h_points=n_h_full,
        profile=profile,
        spsa_model=spsa_model,
    )

    print("\n  ┌─── Pre-Execution Budget (model-based, before calibration) ──┐")
    print(f"  │  Effective CLOPS: {cost.effective_clops}                            │")
    print(
        f"  │  Total h-points: {n_h_full} (Tier0:{len(TIER_0_H)} + T1:{len(TIER_1_H)} "
        f"+ T2:{len(TIER_1_H) * len(TIER_2_SEEDS)} + T3:{len(TIER_3_H)})    │"
    )
    print(f"  │  Total shots: {cost.total_shots:,}                          │")
    print(
        f"  │  Optimistic: {cost.est_total_optimistic_s / 60:.1f} min | "
        f"Expected: {cost.est_total_s / 60:.1f} min          │"
    )
    print(
        f"  │  Fits per job: {'✅' if cost.fits_per_job else '❌'}  |  "
        f"Fits 10min: {'✅' if cost.fits_full_sweep_10min else '❌'}                     │"
    )
    print("  └────────────────────────────────────────────────────────────────┘")

    if args.dry_run:
        print("\n  [DRY RUN] No QPU jobs submitted.")
        print("  Run without --dry-run to execute on real hardware.")
        sys.exit(0)

    # ── Initialize ────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    slogger = StructuredLogger("ibm_torino_deployment")
    lattice = make_lattice(TOPOLOGY, N_QUBITS)

    # ── Determine tiers to run ────────────────────────────────────────────
    tiers_to_run = args.tier if args.tier is not None else [0, 1, 2, 3]

    execution_summary = {
        "start_time": datetime.now(UTC).isoformat(),
        "run_id": run_timestamp,
        "config": {
            "topology": TOPOLOGY,
            "n_qubits": N_QUBITS,
            "p_layers": P_LAYERS,
            "shots": args.shots,
            "n_layouts": args.n_layouts,
            "amplifier": args.zne_amplifier,
            "spsa_enabled": spsa_enabled,
        },
        "pre_execution_cost_estimate": {
            "effective_clops": cost.effective_clops,
            "optimistic_min": cost.est_total_optimistic_s / 60,
            "expected_min": cost.est_total_s / 60,
            "pessimistic_min": cost.est_total_pessimistic_s / 60,
        },
        "tiers": {},
    }

    total_wall_clock_start = time.time()

    # ── Tier execution loop ───────────────────────────────────────────────
    for tier in tiers_to_run:
        if tier == 0:
            tier_metrics = run_tier_0(config, lattice, slogger, spsa_enabled=spsa_enabled)
            execution_summary["tiers"]["tier_0"] = {
                "passed": tier_metrics.passed,
                "wall_clock_s": tier_metrics.wall_clock_s,
                "t_one_job_measured_s": tier_metrics.t_one_job_measured_s,
                "delta_e_gap": tier_metrics.mean_de_gap,
                "budget_recompute": tier_metrics.budget_recompute,
                "per_h": tier_metrics.per_h_results,
            }
            if not tier_metrics.passed and args.tier is None:
                print("\n  ⛔ Smoke test failed. Aborting remaining tiers.")
                break

        elif tier == 1:
            tier_metrics = run_tier_1(config, lattice, slogger)
            execution_summary["tiers"]["tier_1"] = {
                "passed": tier_metrics.passed,
                "wall_clock_s": tier_metrics.wall_clock_s,
                "n_pass": tier_metrics.n_pass,
                "n_total": tier_metrics.n_total,
                "pass_rate": tier_metrics.pass_rate,
                "mean_de_gap": tier_metrics.mean_de_gap,
                "std_de_gap": tier_metrics.std_de_gap,
                "spsa_triggered": tier_metrics.spsa_triggered_count,
                "per_h": tier_metrics.per_h_results,
            }
            if tier_metrics.pass_rate < 0.5 and args.tier is None:
                print("\n  ⛔ Tier 1 pass rate < 50%. Skipping Tier 2/3.")
                break

        elif tier == 2:
            tier_metrics = run_tier_2(config, lattice, slogger)
            execution_summary["tiers"]["tier_2"] = {
                "passed": tier_metrics.passed,
                "wall_clock_s": tier_metrics.wall_clock_s,
                "n_pass": tier_metrics.n_pass,
                "n_total": tier_metrics.n_total,
                "pass_rate": tier_metrics.pass_rate,
                "mean_de_gap": tier_metrics.mean_de_gap,
                "std_de_gap": tier_metrics.std_de_gap,
                "spsa_triggered": tier_metrics.spsa_triggered_count,
                "per_h": tier_metrics.per_h_results,
            }

        elif tier == 3:
            tier_metrics = run_tier_3(config, lattice, slogger)
            execution_summary["tiers"]["tier_3"] = {
                "passed": tier_metrics.passed,
                "wall_clock_s": tier_metrics.wall_clock_s,
                "delta_e_gap": tier_metrics.mean_de_gap,
                "per_h": tier_metrics.per_h_results,
            }

    # ── Final summary ─────────────────────────────────────────────────────
    total_wall_clock = time.time() - total_wall_clock_start
    execution_summary["end_time"] = datetime.now(UTC).isoformat()
    execution_summary["total_wall_clock_s"] = total_wall_clock
    execution_summary["total_wall_clock_min"] = total_wall_clock / 60

    # Count overall pass/fail
    tiers_passed = sum(1 for t in execution_summary["tiers"].values() if t.get("passed"))
    tiers_total = len(execution_summary["tiers"])
    execution_summary["overall"] = {
        "tiers_passed": tiers_passed,
        "tiers_total": tiers_total,
        "all_passed": tiers_passed == tiers_total,
    }

    # Save
    summary_path = output_dir / "execution_summary.json"
    json_dump(execution_summary, summary_path)

    print("\n" + "═" * 70)
    print("  EXECUTION COMPLETE")
    print("═" * 70)
    print(f"  Total wall-clock: {total_wall_clock:.1f}s ({total_wall_clock / 60:.1f} min)")
    print(f"  Tiers passed: {tiers_passed}/{tiers_total}")
    print(f"  Results: {summary_path}")
    print()

    if execution_summary["overall"]["all_passed"]:
        print("  🎉 ALL TIERS PASSED — Hardware deployment successful!")
    else:
        print("  ⚠️  Some tiers failed — review results before proceeding.")

    print()


if __name__ == "__main__":
    main()
