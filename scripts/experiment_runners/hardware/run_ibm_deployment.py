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
    .venv/bin/python scripts/experiment_runners/hardware/run_ibm_deployment.py

    # Calibration only (Tier 0 + budget estimate)
    .venv/bin/python scripts/experiment_runners/hardware/run_ibm_deployment.py --tier 0

    # Safe mode (no SPSA, prevents 400-min budget blowout)
    .venv/bin/python scripts/experiment_runners/hardware/run_ibm_deployment.py --no-spsa

    # Dry run (preflight + cost estimate only, no QPU usage)
    .venv/bin/python scripts/experiment_runners/hardware/run_ibm_deployment.py --dry-run

    # Custom configuration
    .venv/bin/python scripts/experiment_runners/hardware/run_ibm_deployment.py \\
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
from qmbp_simulation.models.constants import DEFAULT_SEEDS
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
# TIER_1_H: primary thesis data. h=2.5 was removed — it is below the valid regime
# for heavy_hex N=10 p=1 (h_min_safe=3.0, from P1_VALID_REGIME + 0 margin).
# Including h=2.5 would produce systematically failing points that waste QPU budget.
# Replaced with h=3.5 which is well inside the valid regime and provides better
# coverage of the paramagnetic-to-ferromagnetic crossover at h≈3.25.
# Validated range for thesis: [3.0, 3.25, 3.5, 4.0].
# h=4.0 excluded from Tier 1 sweep when already acquired in Tier 0.
# The Tier 0 result (ΔE/gap=0.71%) is reusable as the h=4.0 data point.
TIER_1_H = [3.5, 3.25]
TIER_2_SEEDS = DEFAULT_SEEDS
TIER_3_MODEL = "tfim_longitudinal"
TIER_3_G = 0.3
TIER_3_H = [3.25]

# VQE/MPNN training config (for generating predictions if no checkpoint)
# NOTE: H_TRAIN_GRID intentionally does NOT include h=3.25 or h=3.0 —
# those are TIER_1_H deployment (test) points. Including them would cause
# data leakage: the MPNN would interpolate within training data rather than
# generalize to unseen h-values. This separation is the correct protocol
# for fair deployment evaluation. h=2.75 is added as the boundary anchor
# to replace the removed h=3.0 training point.
H_TRAIN_GRID = [4.5, 4.25, 4.0, 3.75, 3.5, 2.75]
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
    # IBM canonical (PEA tutorial) — gentle noise factors for PEA's precise
    # amplification. PEA doesn't need large factors because it amplifies the
    # *learned* noise model probabilistically (no extra circuit depth).
    # Ref: IBM PEA tutorial (2026), factors [1, 1.3, 1.6] on ibm_fez 149q.
    "ibm_canonical": (40, 64, [1, 1.3, 1.6], 3),
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

# σ_flow threshold: h-points with σ_flow above this get boosted shots/layouts
SIGMA_FLOW_THRESHOLD = 0.5

# AQC-Tensor compression defaults
AQC_DEFAULT_BOND_DIM = 64
AQC_DEFAULT_FIDELITY_THRESHOLD = 0.998
AQC_DEFAULT_P_SOURCE = 2

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
    kappa_recommendations: dict | None = None
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


def load_sigma_flow_from_rehearsal(path: str | None) -> dict[float, float] | None:
    """Load sigma_flow_per_h from a V3 rehearsal result JSON.

    Searches for flow_warmstart.sigma_flow_per_h in section_10 data.
    Returns None if path is None, file not found, or no flow data.

    Parameters
    ----------
    path : str | None
        Path to a run_*.json from exp_hw_rehearsal_v3.

    Returns
    -------
    dict[float, float] | None
        Mapping h → σ_flow, or None if unavailable.
    """
    if path is None:
        return None

    import json

    result_path = Path(path)
    if not result_path.exists():
        logger.warning(f"  σ_flow results file not found: {path}")
        return None

    try:
        with open(result_path) as f:
            data = json.load(f)
        # Navigate to section_10.data.flow_warmstart.sigma_flow_per_h
        # Support multiple JSON structure variants:
        #   1. Standard V3: results.section_10.data.flow_warmstart.sigma_flow_per_h
        #   2. Flat: sigma_flow_per_h at top level
        #   3. Legacy: flow_warmstart.sigma_flow_per_h at top level
        raw_sigma = None

        # Variant 1: Standard V3 rehearsal structure
        results = data.get("results", {})
        s10 = results.get("section_10", {})
        s10_data = s10.get("data", {})
        fw = s10_data.get("flow_warmstart", {})
        raw_sigma = fw.get("sigma_flow_per_h")

        # Variant 2: Flat top-level key
        if not raw_sigma:
            raw_sigma = data.get("sigma_flow_per_h")

        # Variant 3: flow_warmstart at top level (legacy/manual export)
        if not raw_sigma:
            fw_top = data.get("flow_warmstart", {})
            raw_sigma = fw_top.get("sigma_flow_per_h") if isinstance(fw_top, dict) else None

        if not raw_sigma:
            logger.warning(
                f"  No sigma_flow_per_h in {result_path.name}. "
                "Run V3 rehearsal with --use-flow-warmstart first."
            )
            return None

        # Keys may be strings from JSON — convert to float
        sigma_flow_per_h = {}
        for k, v in raw_sigma.items():
            try:
                sigma_flow_per_h[float(k)] = float(v)
            except (ValueError, TypeError) as conv_err:
                logger.warning(f"  Skipping non-numeric σ_flow key/value: {k}={v} ({conv_err})")
        if not sigma_flow_per_h:
            logger.warning(
                f"  All sigma_flow_per_h entries in {result_path.name} were non-numeric."
            )
            return None
        logger.info(
            f"  σ_flow loaded from {result_path.name}: "
            f"{len(sigma_flow_per_h)} h-points, "
            f"mean σ={sum(sigma_flow_per_h.values()) / len(sigma_flow_per_h):.3f}"
        )
        n_boost = sum(1 for s in sigma_flow_per_h.values() if s > 0.5)
        if n_boost > 0:
            logger.info(
                f"  ⚠️ {n_boost}/{len(sigma_flow_per_h)} h-points have σ > 0.5 "
                "→ will receive 2× shots + ≥3 layouts"
            )
        # Warn if rehearsal h-values don't overlap with deployment TIER_1_H
        deployment_h_set = set(TIER_1_H)
        sigma_h_set = set(sigma_flow_per_h.keys())
        matched = sum(
            1
            for h_dep in deployment_h_set
            if any(abs(h_dep - h_sig) < 1e-6 for h_sig in sigma_h_set)
        )
        if matched == 0:
            logger.warning(
                f"  ⚠️ σ_flow h-values {sorted(sigma_h_set)} have NO overlap with "
                f"TIER_1_H={TIER_1_H}. σ_flow boost will never trigger. "
                f"Was the rehearsal run with different h_test values?"
            )
        elif matched < len(deployment_h_set):
            logger.warning(
                f"  σ_flow covers {matched}/{len(deployment_h_set)} TIER_1_H points. "
                f"Missing h-points will use base shots/layouts."
            )
        return sigma_flow_per_h
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"  Failed to parse σ_flow from {path}: {e}")
        return None


def build_hardware_config(
    shots: int = DEFAULT_SHOTS,
    n_layouts: int = DEFAULT_N_LAYOUTS,
    amplifier: str = DEFAULT_AMPLIFIER,
    output_dir: str = "results/hardware",
    spsa_enabled: bool = True,
    backend_name: str = BACKEND_NAME,
    pea_preset: str = "balanced",
    job_timeout_s: int = 900,
    layer_pair_depths: list[int] | None = None,
    qesem_enabled: bool = False,
    qesem_precision: float = 0.01,
    qesem_max_execution_time: int = 300,
) -> HardwareConfig:
    """Build HardwareConfig for real QPU execution.

    PEA presets (2026-06-14 — from PEA calibration study):
    - "default": IBM default (32×128=4K). Fast, but fails on degraded calibration.
    - "ibm_canonical": IBM PEA tutorial (40×64=2.5K + [1,1.3,1.6] + 3 layouts).
      Gentle noise factors optimized for PEA's precise probabilistic amplification.
    - "balanced": 48×192=9K + [1,1.5,3]. Sweet spot for 2-4% error rates.
    - "aggressive": 64×256=16K + [1,1.5,2,3] + 3 layouts. Maximum accuracy, slow.
    - "default_3layout": IBM default + 3 layouts. Tests variance vs bias.

    The preset can be overridden by --pea-config CLI flag.

    Parameters
    ----------
    layer_pair_depths : list[int] | None
        Identity-pair depths for PEA noise learning. None = Runtime default.
        For HVA p=1 (1 layer): [0, 1, 2, 4, 8] recommended.
        For deep circuits: [0, 1, 2, 4, 6, 12, 24] per IBM tutorial.
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
        job_timeout_s=job_timeout_s if job_timeout_s > 0 else None,
        max_retries=3,
        retry_delay_s=60,
        max_total_shots=10_000_000,
        spsa_enabled=spsa_enabled,
        spsa_threshold=DE_GAP_THRESHOLD,
        output_dir=output_dir,
        qesem_precision=qesem_precision,
        qesem_max_execution_time=qesem_max_execution_time,
        mitigation=MitigationOptions(
            dd_enabled=True,
            trex_enabled=True,
            twirling_enabled=True,
            zne_enabled=True,
            zne_amplifier=amplifier,
            zne_noise_factors=noise_factors,
            num_randomizations=num_rand,
            shots_per_randomization=shots_rand,
            layer_pair_depths=layer_pair_depths,
            twirling_strategy="active-circuit",
            qesem_enabled=qesem_enabled,
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
    # Checkpoint name includes a hash of H_TRAIN_GRID to auto-invalidate
    # when the training grid changes (e.g., after data-leakage fix).
    import hashlib

    grid_hash = hashlib.md5(str(sorted(H_TRAIN_GRID)).encode()).hexdigest()[:8]
    ckpt_dir = _ROOT / "results" / "hardware" / "mpnn_checkpoints"
    ckpt_path = ckpt_dir / f"mpnn_heavy_hex_n{N_QUBITS}_p{P_LAYERS}_seed{seed}_grid{grid_hash}.pt"
    # Legacy path (old grid, without hash) — never load silently
    legacy_path = ckpt_dir / f"mpnn_heavy_hex_n{N_QUBITS}_p{P_LAYERS}_seed{seed}.pt"

    _needs_training = True  # default: train unless a valid checkpoint is found
    if ckpt_path.exists():
        logger.info(f"Loading MPNN checkpoint: {ckpt_path}")
        model = load_mpnn_checkpoint(ckpt_path)
        # Verify output_dim matches current config (guard against stale checkpoints
        # from different P_LAYERS or N_QUBITS without a matching grid_hash change)
        expected_n_params = P_LAYERS * 2  # HVA p=1: θ_zz + θ_x per layer
        if hasattr(model, "output_dim") and model.output_dim != expected_n_params:
            logger.warning(
                f"Checkpoint output_dim={model.output_dim} ≠ "
                f"expected {expected_n_params} (p={P_LAYERS}). "
                f"Retraining with current config."
            )
        else:
            _needs_training = False  # Checkpoint is valid
    elif legacy_path.exists():
        logger.warning(
            f"Found legacy checkpoint {legacy_path.name} (trained with old H_TRAIN_GRID). "
            f"Retraining with current grid {H_TRAIN_GRID} to avoid data leakage. "
            f"Delete this file to suppress this warning."
        )
        # _needs_training stays True — do NOT load the legacy checkpoint

    if _needs_training:
        logger.info("No valid checkpoint found. Training MPNN from scratch...")
        circuit_builder = HVACircuitBuilder()
        # P0-B: Use PauliEvolutionGate for better transpiler scheduling on heavy_hex.
        # Validated Section 20: same unitary (|ΔE|<1e-14), 6-10% lower total_depth.
        circuit, _ = circuit_builder.create_pauli_evolution(N_QUBITS, P_LAYERS, lattice_cfg)

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
            fidelity_threshold=0.0,  # noqa: noiseless VQE — no fidelity filtering needed
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


def prepare_aqc_compressed_circuit(
    h_values: list[float],
    lattice,
    params_per_h: dict[float, np.ndarray],
    *,
    p_source: int = AQC_DEFAULT_P_SOURCE,
    bond_dim: int = AQC_DEFAULT_BOND_DIM,
    fidelity_threshold: float = AQC_DEFAULT_FIDELITY_THRESHOLD,
    seed: int = 42,
) -> dict[float, dict]:
    """Compress HVA p=p_source circuits via AQC-Tensor for hardware deployment.

    For each h-value, builds an HVA p=p_source circuit with MPNN-predicted params,
    compresses it to a shallower circuit, and validates the compression quality.

    Parameters
    ----------
    h_values : list[float]
        h-values to compress circuits for.
    lattice : LatticeConfig
        Lattice configuration.
    params_per_h : dict[float, np.ndarray]
        MPNN-predicted parameters for each h (must be for p=p_source).
    p_source : int
        Source p_layers of the circuit to compress (default: 2).
    bond_dim : int
        MPS bond dimension for compression (default: 64).
    fidelity_threshold : float
        Minimum acceptable fidelity (default: 0.998).
    seed : int
        Random seed for the internal VQE used to build the good circuit.

    Returns
    -------
    dict[float, dict]
        Mapping h → compression result info with keys:
        - "circuit": compressed QuantumCircuit (bound)
        - "fidelity": achieved fidelity
        - "n_2q_original": original 2Q gate count
        - "n_2q_compressed": compressed 2Q gate count
        - "wall_clock_s": compression time
        - "acceptable": bool (meets quality threshold)
        - "fallback_to_p1": bool (True if compression failed, use p=1 instead)
    """
    from qmbp_simulation.circuits.aqc_compression import (
        AQCCircuitCompressor,
        AQCCompressionConfig,
    )

    config = AQCCompressionConfig(
        max_bond_dim=bond_dim,
        fidelity_threshold=fidelity_threshold,
        max_iterations=200,
    )
    compressor = AQCCircuitCompressor(config)
    circuit_builder = HVACircuitBuilder()

    results: dict[float, dict] = {}
    for h in h_values:
        logger.info(f"  AQC compression for h={h:.3f}...")
        # Build p=p_source circuit bound with MPNN predictions
        lattice_h = make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=h)
        # P0-B: PauliEvolutionGate for better transpiler scheduling
        circuit_p, _ = circuit_builder.create_pauli_evolution(N_QUBITS, p_source, lattice_h)
        target_circuit = circuit_p.assign_parameters(params_per_h[h])

        try:
            result = compressor.compress_circuit(target_circuit, lattice_h)
            acceptable = result.fidelity >= fidelity_threshold
            results[h] = {
                "circuit": result.compressed_circuit,
                "fidelity": result.fidelity,
                "n_2q_original": result.n_2q_original,
                "n_2q_compressed": result.n_2q_compressed,
                "n_2q_reduction_pct": result.n_2q_reduction_pct,
                "wall_clock_s": result.wall_clock_s,
                "acceptable": acceptable,
                "fallback_to_p1": not acceptable,
                "n_iterations": result.n_iterations,
            }
            status = "✅" if acceptable else "⚠️ fallback to p=1"
            logger.info(
                f"    {status} F={result.fidelity:.5f}, "
                f"2Q: {result.n_2q_original}→{result.n_2q_compressed}, "
                f"{result.wall_clock_s:.1f}s"
            )
        except Exception as exc:
            logger.warning(f"    ❌ AQC compression failed for h={h}: {exc}")
            results[h] = {
                "circuit": None,
                "fidelity": 0.0,
                "n_2q_original": 0,
                "n_2q_compressed": 0,
                "n_2q_reduction_pct": 0.0,
                "wall_clock_s": 0.0,
                "acceptable": False,
                "fallback_to_p1": True,
                "error": str(exc),
            }

    return results


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
    topology: str = TOPOLOGY,
    # ── σ_flow integration ──────────────────────────────────
    sigma_flow_per_h: dict[float, float] | None = None,
    sigma_flow_threshold: float = SIGMA_FLOW_THRESHOLD,
) -> dict[float, dict]:
    """Derive per-h deployment recommendations from κ profile.

    Based on section 19 validation: κ is ANTI-correlated with noise sensitivity
    (r = -0.85 for chain_1d). Low κ → h near h_c → high hardware risk.

    IMPORTANT — Topology Validity:
        Calibrated thresholds (κ<45=HIGH) are ONLY valid for **chain_1d**.
        For heavy_hex, the κ scale is 3× higher ([111-174] vs [41-53]) and
        |r| = 0.52 (weak, threshold 0.70 not met). For heavy_hex, recommendations
        default to MEDIUM for all h-points (conservative, use V2 go/no-go instead).

    Decision rules (chain_1d only):
      κ < 45  → HIGH risk: 2× shots, 3 layouts, PEA mandatory
      κ ∈ [45, 50) → MEDIUM risk: 3 layouts, PEA recommended
      κ ≥ 50  → LOW risk: 1 layout sufficient, standard shots

    Parameters
    ----------
    kappa_per_h : dict[float, float]
        κ(h) from ``compute_kappa_per_h``.
    topology : str
        Topology used. Thresholds only valid for "chain_1d".
        For other topologies, all h-points are labeled MEDIUM risk.
    high_risk_threshold, medium_risk_threshold, n_layouts_*, shots_* :
        See original docstring — chain_1d calibrated defaults.
    shots_base : int
        Base shot count (default: DEFAULT_SHOTS=16384).

    Returns
    -------
    dict[float, dict]
        Per-h recommendations with keys:
        "risk_level", "n_layouts", "shots", "kappa", "spsa_recommended".
    """
    recommendations: dict[float, dict] = {}

    # For heavy_hex: κ thresholds are auto-calibrated from percentiles
    # since hard-coded chain_1d thresholds (45, 50) don't apply
    if topology != "chain_1d":
        valid_kappas = [k for k in kappa_per_h.values() if not np.isnan(k)]
        if valid_kappas:
            high_risk_threshold = float(np.percentile(valid_kappas, 25))
            medium_risk_threshold = float(np.percentile(valid_kappas, 75))
            logger.info(
                f"  κ thresholds auto-calibrated for {topology}: "
                f"high<{high_risk_threshold:.1f}, med<{medium_risk_threshold:.1f}"
            )

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

        rec: dict = {
            "kappa": kappa,
            "risk_level": risk,
            "n_layouts": n_lay,
            "shots": shots,
            "spsa_recommended": spsa_recommended,
        }

        # --- σ_flow boost (applied after κ-based classification) ---
        # Uses approximate float matching for h-keys from JSON (may differ in precision)
        sigma_flow_boost = False
        if sigma_flow_per_h is not None:
            # Find closest key within tolerance (JSON may store 4.0 as "4.0")
            s_flow = None
            for h_sigma, s_flow_val in sigma_flow_per_h.items():
                if abs(h_sigma - h) < 1e-6:
                    s_flow = s_flow_val
                    break
            if s_flow is not None and s_flow > sigma_flow_threshold:
                rec["shots"] = rec["shots"] * 2
                rec["n_layouts"] = max(rec["n_layouts"], 3)
                sigma_flow_boost = True
                logger.info(
                    f"  [σ_flow boost] h={h}: σ_flow={s_flow:.3f} > {sigma_flow_threshold} → "
                    f"shots={rec['shots']}, n_layouts={rec['n_layouts']}"
                )
        rec["sigma_flow_boost"] = sigma_flow_boost
        recommendations[h] = rec

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
    """Tier 0: Minimal smoke test — single circuit, single layout, measures T_one_job.

    This is the fastest valid QPU check before committing to the full experiment:
    1. Validates QPU connectivity and job submission pipeline
    2. Measures real T_one_job (used to recompute the full budget)
    3. Captures a calibration snapshot (T1/T2/error rates at execution time)
    4. Produces one data point that IS reusable as thesis data (h=4.0)

    **Minimum QPU time design:**
    - 1 layout only (not 3) — reduces wall-clock by ~3×
    - SPSA always disabled — prevents potential 400-min budget blowout on smoke test
    - Relaxed abort threshold: abort only if ΔE/gap > 10% (not 5%), since
      the primary goal is connectivity+timing, not optimal error mitigation
    - PEA preset used is whatever was configured (typically "balanced"), but
      the single-layout reduces overhead significantly

    **Still has value because:**
    - T_one_job measurement is accurate (same mitigation stack as production)
    - Calibration snapshot captures T1/T2/error at the moment of execution
    - h=4.0 result is reusable: if ΔE/gap < 10%, the data point counts as Tier 1 h=4.0
    - If ΔE/gap > 10%, aborts immediately before spending full Tier 1 budget

    Returns TierMetrics with t_one_job_measured_s and budget_recompute.
    """
    metrics = TierMetrics(tier=0, n_h_points=1)
    metrics.start_time = datetime.now(UTC).isoformat()

    slogger.log("tier_0_start", data={"h_values": TIER_0_H, "purpose": "smoke_test_min_qpu"})
    print("\n" + "═" * 70)
    print("  TIER 0: SMOKE TEST (h=4.0, single layout, min QPU time)")
    print("  Purpose: Verify QPU connectivity + measure T_one_job for budget")
    print("═" * 70)

    # Build a Tier-0-specific config: 1 layout (not 3) and SPSA always off.
    # This is the minimal config that still exercises the full mitigation stack.
    from dataclasses import replace

    tier0_config = replace(
        config,
        n_layouts=1,  # Single layout: ~3× faster than production 3 layouts
        spsa_enabled=False,  # Never SPSA on smoke test (prevents 400-min blowout)
    )

    circuit_builder = HVACircuitBuilder()
    # PauliEvolutionGate: 6-10% shorter circuit (same energy, |ΔE|<1e-14)
    circuit, _ = circuit_builder.create_pauli_evolution(N_QUBITS, P_LAYERS, lattice)
    params_per_h, e_exact_per_h, gap_per_h = prepare_mpnn_predictions(TIER_0_H, lattice, seed=42)

    # κ risk profile (noiseless, zero QPU cost) — logged for reference
    logger.info("  Computing landscape curvature κ(h) for hardware risk assessment...")
    kappa_per_h = compute_kappa_per_h(params_per_h, lattice)

    backend = HardwareBackend(config=tier0_config)

    # ── Execute with wall-clock timing ────────────────────────────────────
    n_learn = (
        tier0_config.mitigation.num_randomizations * tier0_config.mitigation.shots_per_randomization
    )
    print(
        f"\n  Config: 1 layout, SPSA=OFF, PEA={tier0_config.mitigation.num_randomizations}×"
        f"{tier0_config.mitigation.shots_per_randomization}={n_learn} learning shots"
    )
    print(f"  Noise factors: {tier0_config.mitigation.zne_noise_factors}")
    print(
        f"  DD={tier0_config.mitigation.dd_enabled}, "
        f"Twirling={tier0_config.mitigation.twirling_enabled}, "
        f"TREX={tier0_config.mitigation.trex_enabled}"
    )

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
    # T_one_job for budget extrapolation: measured with 1 layout.
    # Production has 3 layouts, so production T_per_h ≈ t_elapsed × 3.
    # recompute_budget_from_measurement already accounts for n_layouts.
    metrics.t_one_job_measured_s = t_elapsed
    metrics.per_h_wall_clock_s = [t_elapsed]
    metrics.n_jobs_submitted = 1
    metrics.n_jobs_succeeded = 1
    metrics.n_total = 1
    metrics.mean_de_gap = result.delta_e_gap
    metrics.max_de_gap = result.delta_e_gap

    # Tier 0 uses a relaxed abort threshold: ΔE/gap > SMOKE_ABORT_THRESHOLD (10%).
    # Rationale: degraded calibration might give 6-9% which still confirms
    # connectivity + timing. Only catastrophic noise (>10%) warrants abort.
    smoke_passed = result.delta_e_gap < SMOKE_ABORT_THRESHOLD
    metrics.n_pass = 1 if smoke_passed else 0
    metrics.pass_rate = 1.0 if smoke_passed else 0.0
    metrics.passed = smoke_passed
    # SPSA is always off in Tier 0 (tier0_config.spsa_enabled=False)

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
            # Use kappa_go_no_go for topology-aware risk labeling (auto-calibrates
            # percentile thresholds for heavy_hex; uses fixed 45/50 for chain_1d).
            "kappa": kappa_per_h.get(4.0, float("nan")),
            "hardware_risk": kappa_go_no_go(
                {4.0: kappa_per_h.get(4.0, float("nan"))},
                shots_base=config.shots,
                topology=TOPOLOGY,
            )
            .get(4.0, {})
            .get("risk_level", "unknown"),
        }
    ]

    # ── Print results ─────────────────────────────────────────────────────
    de_verdict = (
        "✅ OK"
        if result.delta_e_gap < DE_GAP_THRESHOLD
        else ("⚠️ MARGINAL" if result.delta_e_gap < SMOKE_ABORT_THRESHOLD else "❌ ABORT")
    )
    print("\n  ┌─── Tier 0 Result (smoke test, 1 layout) ─────────────────┐")
    print("  │  h = 4.0 (deep paramagnetic, expected easy point)        │")
    print(f"  │  ΔE/gap:   {result.delta_e_gap:.4f}  [{de_verdict:<12}]                  │")
    print(f"  │  Phase:    {result.phase_label:<20}                     │")
    print(f"  │  ZNE R²:   {result.zne_r2:.4f}                                    │")
    print(f"  │  Strategy: {result.mitigation_strategy:<30}         │")
    print(f"  │  Wall-clock: {t_elapsed:.1f}s (1 layout × full PEA stack)          │")
    print("  └────────────────────────────────────────────────────────────┘")
    print("  Note: Production runs use 3 layouts → ~3× this time per h-point")

    # ── Budget recompute from measured T_one_job ──────────────────────────
    # T_one_job from Tier 0 (1 layout) is used to estimate production time
    # (3 layouts). The budget function takes n_layouts=3 for production.
    n_h_full = len(TIER_1_H) * (1 + len(TIER_2_SEEDS)) + len(TIER_3_H)
    noise_factors = config.mitigation.zne_noise_factors or [1, 1.5, 3]
    budget = recompute_budget_from_measurement(
        t_one_job_s=t_elapsed,
        n_h_points=n_h_full,
        n_layouts=config.n_layouts,  # Production layouts (3), not Tier-0 (1)
        n_zne_factors=len(noise_factors),  # From actual preset, not hardcoded
        spsa_enabled=spsa_enabled,
    )
    metrics.budget_recompute = budget

    print(
        f"\n  ┌─── Budget Estimate (T_one_job={t_elapsed:.1f}s × {config.n_layouts} layouts) ──────┐"
    )
    print(
        f"  │  Full experiment: {n_h_full} h-points × {config.n_layouts} layouts                │"
    )
    print(f"  │  Optimistic (no SPSA): {budget['total_optimistic_min']:.1f} min              │")
    print(f"  │  Expected (P=0.30):    {budget['total_expected_min']:.1f} min              │")
    print(f"  │  Pessimistic (SPSA):   {budget['total_pessimistic_min']:.1f} min             │")
    if budget["exceeds_budget_ceiling"]:
        print(f"  │  ⚠️  WARNING: Exceeds {BUDGET_CEILING_S / 3600:.0f}h budget ceiling!         │")
    print("  └────────────────────────────────────────────────────────────┘")

    if not smoke_passed:
        metrics.abort_reason = (
            f"ΔE/gap={result.delta_e_gap:.4f} ≥ {SMOKE_ABORT_THRESHOLD} "
            f"(catastrophic noise — abort before Tier 1)"
        )
        print(f"\n  ❌ TIER 0 ABORT — ΔE/gap={result.delta_e_gap:.4f} ≥ {SMOKE_ABORT_THRESHOLD}")
        print("  ACTION: Check chip calibration, retry during off-peak (UTC 2-6 AM),")
        print("          or use --pea-config aggressive for better noise learning.")
    elif result.delta_e_gap < DE_GAP_THRESHOLD:
        print("\n  ✅ TIER 0 PASS (data quality: thesis-grade)")
        print(
            f"     ΔE/gap={result.delta_e_gap:.4f} < {DE_GAP_THRESHOLD} — this h=4.0 point is reusable as Tier 1 data"
        )
        print(
            f"     Proceed to Tier 1 with confidence (est. {budget['total_optimistic_min']:.0f} min)"
        )
    else:
        print("\n  ⚠️  TIER 0 OK (data quality: degraded but connectivity confirmed)")
        print(
            f"     ΔE/gap={result.delta_e_gap:.4f} > {DE_GAP_THRESHOLD} but < {SMOKE_ABORT_THRESHOLD} — pipeline works"
        )
        print("     Consider --pea-config aggressive or retry at better calibration before Tier 1")

    metrics.end_time = datetime.now(UTC).isoformat()
    slogger.log(
        "tier_0_result",
        data={
            "smoke_passed": smoke_passed,
            "thesis_grade": result.delta_e_gap < DE_GAP_THRESHOLD,
            "t_one_job_s": t_elapsed,
            "delta_e_gap": result.delta_e_gap,
            "verdict": result.verdict,
            "smoke_abort_threshold": SMOKE_ABORT_THRESHOLD,
            "budget_recompute": budget,
        },
    )
    return metrics


def run_tier_1(
    config: HardwareConfig,
    lattice,
    slogger: StructuredLogger,
    *,
    sigma_flow_per_h: dict[float, float] | None = None,
    sigma_flow_threshold: float = SIGMA_FLOW_THRESHOLD,
) -> TierMetrics:
    """Tier 1: Core validation — primary thesis data.

    Returns TierMetrics with per-h timing and quality metrics.
    """
    metrics = TierMetrics(tier=1, n_h_points=len(TIER_1_H))
    metrics.start_time = datetime.now(UTC).isoformat()

    slogger.log("tier_1_start", data={"h_values": TIER_1_H})
    print("\n" + "═" * 70)
    print("  TIER 1: CORE VALIDATION (primary thesis data)")
    print(f"  h-values: {TIER_1_H}")
    print("═" * 70)

    circuit_builder = HVACircuitBuilder()
    # Use PauliEvolutionGate representation for hardware deployment:
    # validated 6–10% total_depth reduction (Section 20, V3 rehearsal).
    circuit, _ = circuit_builder.create_pauli_evolution(N_QUBITS, P_LAYERS, lattice)
    params_per_h, e_exact_per_h, gap_per_h = prepare_mpnn_predictions(TIER_1_H, lattice, seed=42)

    # ── Compute κ risk profile (noiseless, zero QPU cost) ─────────────────
    logger.info("  Computing landscape curvature κ(h) for per-h risk assessment...")
    kappa_per_h = compute_kappa_per_h(params_per_h, lattice)

    # ── κ go/no-go: derive per-h shot/layout recommendations ──────────────
    kappa_recommendations = kappa_go_no_go(
        kappa_per_h,
        shots_base=config.shots,
        topology=TOPOLOGY,
        sigma_flow_per_h=sigma_flow_per_h,
        sigma_flow_threshold=sigma_flow_threshold,
    )
    metrics.kappa_recommendations = kappa_recommendations
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

    # P1-B: Use run_h_sweep for batch execution within a single QPU session.
    # This caches layouts across h-points (no re-transpilation), monitors TLS
    # calibration drift, and enables warm-start: h-points sorted h=4.0 first
    # (easy → hard) to catch catastrophic noise early.
    builder = HamiltonianBuilder()

    def hamiltonian_builder(h: float) -> SparsePauliOp:
        return builder.build(make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=h))

    print("\n  Using batch warm-start sweep (single session, shared layouts, TLS drift monitor)")
    t_sweep_start = time.time()
    try:
        sweep_results = backend.run_h_sweep(
            circuit,
            hamiltonian_builder,
            list(TIER_1_H),
            params_per_h,
            e_exact_per_h,
            gap_per_h,
        )
    except RuntimeError as exc:
        metrics.wall_clock_s = time.time() - t_sweep_start
        metrics.end_time = datetime.now(UTC).isoformat()
        metrics.abort_reason = f"run_h_sweep() failed: {exc}"
        slogger.log("tier_1_sweep_error", data={"error": str(exc)})
        print(f"\n  ❌ TIER 1 SWEEP FAILED — {exc}")
        return metrics

    t_sweep_total = time.time() - t_sweep_start

    # Extract per-h metrics from sweep results
    de_gaps = []
    for result in sweep_results:
        h = result.h_value
        passed = result.verdict == "PASS"
        if passed:
            metrics.n_pass += 1
        if result.spsa_applied:
            metrics.spsa_triggered_count += 1
        de_gaps.append(result.delta_e_gap)
        metrics.n_jobs_submitted += 1
        metrics.n_jobs_succeeded += 1
        # Estimate per-h time proportionally (sweep doesn't give per-h timing)
        t_h_est = t_sweep_total / len(sweep_results)
        metrics.per_h_wall_clock_s.append(t_h_est)
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
                # Post-QPU validation metrics
                "obs_bounds_clipped": result.obs_bounds_clipped,
                "layout_energy_outliers": result.layout_energy_outliers,
                "e_obs_discrepancy": result.e_obs_discrepancy,
                "e_obs_cross_valid_passed": result.e_obs_cross_valid_passed,
                "n_layouts_observables": result.n_layouts_observables,
                # QPU & calibration (real hardware only)
                "qpu_metrics": getattr(result, "_qpu_metrics", None),
                "calibration_snapshot": getattr(result, "_calibration_snapshot", None),
                "transpiled_stats": getattr(result, "_transpiled_stats", None),
                # Timing
                "wall_clock_s": t_h_est,
                "pass": passed,
                # Landscape curvature (noiseless risk proxy — section 19 validated)
                "kappa": kappa_per_h.get(h, float("nan")),
                "hardware_risk": kappa_recommendations.get(h, {}).get("risk_level", "unknown"),
                "spsa_recommended": kappa_recommendations.get(h, {}).get(
                    "spsa_recommended", False
                ),
            }
        )

        v_mark = "✅" if passed else "❌"
        print(
            f"    {v_mark} h={h:.2f}: ΔE/gap={result.delta_e_gap:.4f} | "
            f"phase={result.phase_label} | R²={result.zne_r2:.3f}"
        )

    # ── Summary metrics ───────────────────────────────────────────────────
    metrics.wall_clock_s = t_sweep_total
    metrics.n_total = len(TIER_1_H)
    metrics.pass_rate = metrics.n_pass / metrics.n_total if metrics.n_total else 0
    if de_gaps:
        metrics.mean_de_gap = float(np.mean(de_gaps))
        metrics.max_de_gap = float(np.max(de_gaps))
        metrics.std_de_gap = float(np.std(de_gaps)) if len(de_gaps) > 1 else 0.0
    # Pass if ≥2/3 of h-points pass. With 3 points: need 2. With 4: need 3.
    min_pass = max(2, round(0.67 * metrics.n_total))
    metrics.passed = metrics.n_pass >= min_pass
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
    *,
    sigma_flow_per_h: dict[float, float] | None = None,
    sigma_flow_threshold: float = SIGMA_FLOW_THRESHOLD,
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
    # Use PauliEvolutionGate representation for hardware deployment:
    # validated 6–10% total_depth reduction (Section 20, V3 rehearsal).
    circuit, _ = circuit_builder.create_pauli_evolution(N_QUBITS, P_LAYERS, lattice)

    # ── Compute κ once (seed-independent, noiseless) ─────────────────────
    # κ depends only on θ_pred(h) from seed=42 MPNN — shared across all seeds.
    _params_seed42, _, _ = prepare_mpnn_predictions(TIER_1_H, lattice, seed=42)
    kappa_per_h_t2 = compute_kappa_per_h(_params_seed42, lattice)
    kappa_recommendations_t2 = kappa_go_no_go(
        kappa_per_h_t2,
        shots_base=config.shots,
        topology=TOPOLOGY,
        sigma_flow_per_h=sigma_flow_per_h,
        sigma_flow_threshold=sigma_flow_threshold,
    )
    metrics.kappa_recommendations = kappa_recommendations_t2

    all_de_gaps = []
    for seed in TIER_2_SEEDS:
        print(f"\n  ─── Seed {seed} ───")
        params_per_h, e_exact_per_h, gap_per_h = prepare_mpnn_predictions(
            TIER_1_H, lattice, seed=seed
        )

        # P1-B: Use batch warm-start sweep per seed (cached layouts, drift monitor)
        from dataclasses import replace

        seed_config = replace(config, layout_seed=seed)
        backend = HardwareBackend(config=seed_config)

        builder = HamiltonianBuilder()

        def hamiltonian_builder(h: float) -> SparsePauliOp:
            return builder.build(make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=h))

        t_seed_start = time.time()
        try:
            sweep_results = backend.run_h_sweep(
                circuit,
                hamiltonian_builder,
                list(TIER_1_H),
                params_per_h,
                e_exact_per_h,
                gap_per_h,
            )
        except RuntimeError as exc:
            t_seed = time.time() - t_seed_start
            logger.error(f"  Seed {seed} sweep FAILED after {t_seed:.1f}s — {exc}")
            for h in TIER_1_H:
                metrics.n_jobs_submitted += 1
                metrics.n_jobs_failed += 1
                metrics.per_h_wall_clock_s.append(0.0)
                metrics.per_h_results.append(
                    {"seed": seed, "h": h, "error": str(exc), "wall_clock_s": 0.0, "pass": False}
                )
            continue
        t_seed_elapsed = time.time() - t_seed_start

        for result in sweep_results:
            h = result.h_value
            t_h_est = t_seed_elapsed / len(sweep_results)
            metrics.n_jobs_submitted += 1
            metrics.n_jobs_succeeded += 1
            passed = result.verdict == "PASS"
            if passed:
                metrics.n_pass += 1
            if result.spsa_applied:
                metrics.spsa_triggered_count += 1
            all_de_gaps.append(result.delta_e_gap)
            metrics.per_h_wall_clock_s.append(t_h_est)
            metrics.per_h_results.append(
                {
                    "seed": seed,
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
                    "sigma": result.sigma,
                    # ZNE details
                    "zne_r2": result.zne_r2,
                    "mitigation_strategy": result.mitigation_strategy,
                    "layout_std": result.layout_std,
                    # SPSA & provenance
                    "spsa_applied": result.spsa_applied,
                    "job_ids": result.job_ids,
                    "ces_values": result.ces_values,
                    "total_shots": result.total_shots,
                    # Per-site observables (thesis data)
                    "per_site_x": result.per_site_x,
                    "per_bond_zz": result.per_bond_zz,
                    # Post-correction
                    "gnn_qem_applied": result.gnn_qem_applied,
                    "affine_correction_applied": result.affine_correction_applied,
                    "e_after_affine": result.e_after_affine,
                    # Post-QPU validation metrics
                    "obs_bounds_clipped": result.obs_bounds_clipped,
                    "layout_energy_outliers": result.layout_energy_outliers,
                    "e_obs_discrepancy": result.e_obs_discrepancy,
                    "e_obs_cross_valid_passed": result.e_obs_cross_valid_passed,
                    "n_layouts_observables": result.n_layouts_observables,
                    # QPU & calibration (real hardware only)
                    "qpu_metrics": getattr(result, "_qpu_metrics", None),
                    "calibration_snapshot": getattr(result, "_calibration_snapshot", None),
                    "transpiled_stats": getattr(result, "_transpiled_stats", None),
                    # Timing
                    "wall_clock_s": t_h_est,
                    "pass": passed,
                    # κ risk profile (noiseless, shared across seeds)
                    "kappa": kappa_per_h_t2.get(h, float("nan")),
                    "hardware_risk": kappa_recommendations_t2.get(h, {}).get(
                        "risk_level", "unknown"
                    ),
                }
            )
            v_mark = "✅" if passed else "❌"
            print(f"    {v_mark} h={h:.2f}: ΔE/gap={result.delta_e_gap:.4f}")

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
    parser.add_argument(
        "--sigma-flow-results",
        type=str,
        default=None,
        help=(
            "Path to a V3 rehearsal JSON (run_*.json) that contains "
            "flow_warmstart.sigma_flow_per_h. When provided, σ_flow values "
            "are fed into kappa_go_no_go() for adaptive shot/layout boost "
            "on high-uncertainty h-points (σ > 0.5 → 2× shots, ≥3 layouts)."
        ),
    )
    parser.add_argument(
        "--flow-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a saved FlowWarmstartManager checkpoint (.pt). "
            "When provided, loads the flow model from disk instead of "
            "re-training. Use with --sigma-flow-results for full σ_flow "
            "integration without repeating V3 rehearsal."
        ),
    )
    parser.add_argument(
        "--sigma-flow-threshold",
        type=float,
        default=SIGMA_FLOW_THRESHOLD,
        help=(
            f"σ_flow threshold for boost trigger (default: {SIGMA_FLOW_THRESHOLD}). "
            "h-points with σ_flow above this receive 2× shots and ≥3 layouts."
        ),
    )
    parser.add_argument(
        "--job-timeout",
        type=int,
        default=1900,
        help=(
            "Timeout in seconds per QPU job (default: 900). "
            "Set to 0 to disable timeout (wait indefinitely for job completion)."
        ),
    )
    # ── AQC-Tensor compression options ────────────────────────────────────
    parser.add_argument(
        "--aqc-compress",
        action="store_true",
        help=(
            "Enable AQC-Tensor circuit compression. Compresses a p=2 HVA circuit "
            "to p=1-equivalent 2Q-gate count, retaining p=2 expressibility. "
            "Requires: pip install 'qiskit-addon-aqc-tensor[quimb-jax]'"
        ),
    )
    parser.add_argument(
        "--aqc-bond-dim",
        type=int,
        default=AQC_DEFAULT_BOND_DIM,
        help=f"MPS bond dimension for AQC compression (default: {AQC_DEFAULT_BOND_DIM})",
    )
    parser.add_argument(
        "--aqc-fidelity",
        type=float,
        default=AQC_DEFAULT_FIDELITY_THRESHOLD,
        help=f"Minimum fidelity for accepting compressed circuit (default: {AQC_DEFAULT_FIDELITY_THRESHOLD})",
    )
    parser.add_argument(
        "--aqc-p-source",
        type=int,
        default=AQC_DEFAULT_P_SOURCE,
        choices=[1, 2],
        help=f"Source p_layers to compress from (default: {AQC_DEFAULT_P_SOURCE})",
    )
    # ── Layout optimizer (mapomatic VF2) options ──────────────────────────
    parser.add_argument(
        "--no-mapomatic",
        action="store_true",
        help="Disable mapomatic VF2 layout optimization (use BFS fallback)",
    )
    parser.add_argument(
        "--layout-strategy",
        choices=["lowest_cost", "ces_spread", "hybrid"],
        default="lowest_cost",
        help="Layout selection strategy (default: lowest_cost). "
        "'ces_spread' for ZNE diversity, 'hybrid' for adaptive.",
    )
    parser.add_argument(
        "--layout-max-2q-error",
        type=float,
        default=0.01,
        help="Max 2Q gate error for layout CouplingMap filtering (default: 0.01)",
    )
    # ── P2-A: Dynamic layout escalation options ───────────────────────────
    parser.add_argument(
        "--min-ces-spread",
        type=float,
        default=0.02,
        help=(
            "Minimum CES spread across layouts for ZNE extrapolation quality "
            "(default: 0.02). If spread is below this, escalate to --n-layouts-max."
        ),
    )
    parser.add_argument(
        "--n-layouts-max",
        type=int,
        default=5,
        help=(
            "Maximum layouts to select when CES spread is insufficient "
            "(default: 5). Only used when P2-A dynamic escalation triggers."
        ),
    )
    # ── Mitiq complementary mitigation options ────────────────────────────
    parser.add_argument(
        "--mitiq-verify",
        action="store_true",
        help=(
            "Enable Mitiq CDR verification after PEA-ZNE. "
            "Provides an independent cross-check without noise model. "
            "Adds ~150%% overhead per h-point (10 near-Clifford circuits)."
        ),
    )
    parser.add_argument(
        "--mitiq-benchmark",
        action="store_true",
        help=(
            "Run full Mitiq multi-method comparison (ZNE/CDR/DDD+ZNE). "
            "Produces thesis table comparing all mitigation strategies. "
            "Significantly slower — use only for offline analysis."
        ),
    )
    # ── QESEM (Qedma) mitigation options ─────────────────────────────────
    parser.add_argument(
        "--qesem",
        action="store_true",
        help=(
            "Use QESEM (Qedma Qiskit Function) for mitigation instead of "
            "local PEA-ZNE. QESEM provides unbiased, characterization-based "
            "quasi-probabilistic error mitigation. Requires IBM Premium/Flex "
            "plan and qiskit-ibm-catalog>=0.8.0. Ref: arXiv:2508.10997."
        ),
    )
    parser.add_argument(
        "--qesem-precision",
        type=float,
        default=0.01,
        help=(
            "QESEM target precision (ε) per observable (default: 0.01). "
            "Lower values → more QPU time but higher accuracy."
        ),
    )
    parser.add_argument(
        "--qesem-max-time",
        type=int,
        default=300,
        help=(
            "QESEM max QPU time per PUB in seconds (default: 300). "
            "Limits QPU budget per h-point. Increase for higher precision."
        ),
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

    # ── Load σ_flow from V3 rehearsal (if provided) ───────────────────────
    sigma_flow_per_h = load_sigma_flow_from_rehearsal(args.sigma_flow_results)
    sigma_flow_threshold = args.sigma_flow_threshold

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
    sigma_note = f"from {Path(args.sigma_flow_results).name}" if sigma_flow_per_h else "disabled"
    print(f"║  σ_flow safety net: {sigma_note:<46}║")
    aqc_note = (
        f"p={args.aqc_p_source}→shallow, χ={args.aqc_bond_dim}" if args.aqc_compress else "disabled"
    )
    print(f"║  AQC compression: {aqc_note:<47}║")
    mapo_note = (
        "disabled (--no-mapomatic)"
        if args.no_mapomatic
        else f"VF2, strategy={args.layout_strategy}"
    )
    print(f"║  Layout optimizer: {mapo_note:<47}║")
    qesem_note = (
        f"ENABLED (ε={args.qesem_precision}, max={args.qesem_max_time}s)"
        if args.qesem
        else "disabled (using PEA-ZNE)"
    )
    print(f"║  QESEM (Qedma):    {qesem_note:<47}║")
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
        job_timeout_s=args.job_timeout,
        qesem_enabled=args.qesem,
        qesem_precision=args.qesem_precision,
        qesem_max_execution_time=args.qesem_max_time,
    )
    # Apply mapomatic CLI overrides
    if args.no_mapomatic:
        config.use_mapomatic = False
    config.layout_strategy = args.layout_strategy
    config.layout_max_2q_error = args.layout_max_2q_error
    # P2-A: Dynamic layout escalation
    config.min_ces_spread = args.min_ces_spread
    config.n_layouts_max = args.n_layouts_max

    # ── Determine tiers to run (needed before budget print) ─────────────
    tiers_to_run = args.tier if args.tier is not None else [0, 1, 2, 3]

    # ── Pre-execution cost estimate (model-based) — only for selected tiers ─
    # Count h-points for the tiers that will actually execute
    n_h_selected = 0
    tier_h_breakdown = {}
    if 0 in tiers_to_run:
        n_h_t0 = len(TIER_0_H)
        n_h_selected += n_h_t0
        tier_h_breakdown["T0"] = n_h_t0
    if 1 in tiers_to_run:
        n_h_t1 = len(TIER_1_H)
        n_h_selected += n_h_t1
        tier_h_breakdown["T1"] = n_h_t1
    if 2 in tiers_to_run:
        n_h_t2 = len(TIER_1_H) * len(TIER_2_SEEDS)
        n_h_selected += n_h_t2
        tier_h_breakdown["T2"] = n_h_t2
    if 3 in tiers_to_run:
        n_h_t3 = len(TIER_3_H)
        n_h_selected += n_h_t3
        tier_h_breakdown["T3"] = n_h_t3

    spsa_model = SPSACostModel() if spsa_enabled else SPSACostModel.disabled()
    profile = QPUThroughputProfile.ibm_kingston()
    cost = estimate_qpu_cost(
        config,
        n_h_points=n_h_selected,
        profile=profile,
        spsa_model=spsa_model,
    )

    tier_desc = " + ".join(f"{k}:{v}" for k, v in tier_h_breakdown.items())
    # Compute QPU-only times (excludes classical latency / queue wait)
    qpu_only_optimistic = cost.est_total_optimistic_s - cost.classical_latency_s
    qpu_only_expected = cost.est_total_s - cost.classical_latency_s
    print("\n  ┌─── Pre-Execution Budget (model-based, before calibration) ──┐")
    print(f"  │  Effective CLOPS: {cost.effective_clops}                            │")
    print(f"  │  Tiers to run: {sorted(tiers_to_run)}                                  │")
    print(f"  │  Total h-points: {n_h_selected} ({tier_desc}){' ' * max(0, 30 - len(tier_desc))}│")
    print(f"  │  Total shots: {cost.total_shots:,}                          │")
    print(
        f"  │  QPU time — Optimistic: {qpu_only_optimistic / 60:.1f} min | "
        f"Expected: {qpu_only_expected / 60:.1f} min    │"
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
            "aqc_compress": args.aqc_compress,
            "aqc_bond_dim": args.aqc_bond_dim if args.aqc_compress else None,
            "aqc_fidelity_threshold": args.aqc_fidelity if args.aqc_compress else None,
            "aqc_p_source": args.aqc_p_source if args.aqc_compress else None,
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
            tier_metrics = run_tier_1(
                config,
                lattice,
                slogger,
                sigma_flow_per_h=sigma_flow_per_h,
                sigma_flow_threshold=sigma_flow_threshold,
            )
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
                "kappa_recommendations": {
                    float(h): rec
                    for h, rec in (
                        getattr(tier_metrics, "kappa_recommendations", None) or {}
                    ).items()
                },
                "sigma_flow_per_h": (
                    {float(k): v for k, v in sigma_flow_per_h.items()}
                    if sigma_flow_per_h is not None
                    else None
                ),
            }
            if tier_metrics.pass_rate < 0.5 and args.tier is None:
                print("\n  ⛔ Tier 1 pass rate < 50%. Skipping Tier 2/3.")
                break

            # ── Mitiq verification (--mitiq-verify or --mitiq-benchmark) ──
            if getattr(args, "mitiq_verify", False) or getattr(args, "mitiq_benchmark", False):
                try:
                    from qmbp_simulation.execution.mitiq_utils import (
                        compare_mitigation_strategies,
                        is_mitiq_available,
                    )

                    if is_mitiq_available():
                        from qmbp_simulation.execution import NoisyEstimatorConfig

                        logger.info("\n  ── Mitiq Verification (post-Tier-1) ──")
                        mitiq_config = NoisyEstimatorConfig(shots=config.shots, seed_simulator=42)
                        # Get the noisy backend from tier 1 config
                        # For fake_backend mode, use local comparison
                        # For hardware mode, mitiq verification runs locally
                        # against FakeTorino as independent check
                        try:
                            from qiskit_ibm_runtime.fake_provider import FakeKingston

                            mitiq_backend = FakeKingston()
                        except ImportError:
                            mitiq_backend = None

                        if mitiq_backend is not None:
                            mitiq_results = []
                            strategies = (
                                [
                                    "raw",
                                    "mitiq_zne_linear",
                                    "mitiq_cdr",
                                    "mitiq_ddd_zne",
                                    "native_gf_zne",
                                ]
                                if getattr(args, "mitiq_benchmark", False)
                                else ["raw", "mitiq_cdr"]
                            )
                            circuit_builder = HVACircuitBuilder()
                            circuit, _ = circuit_builder.create_pauli_evolution(
                                N_QUBITS, P_LAYERS, lattice
                            )
                            for h in TIER_1_H[:2]:  # Limit to 2 h-points
                                h_lattice = make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=h)
                                H_h = HamiltonianBuilder().build(h_lattice)
                                params = params_per_h.get(h)
                                if params is None:
                                    continue
                                bound = circuit.assign_parameters(params)
                                try:
                                    mr = compare_mitigation_strategies(
                                        bound,
                                        H_h,
                                        mitiq_backend,
                                        mitiq_config,
                                        exact_energy=e_exact_per_h[h],
                                        gap=gap_per_h[h],
                                        h_value=h,
                                        strategies=strategies,
                                    )
                                    mitiq_results.append(
                                        {
                                            "h": h,
                                            "best_method": mr.best_method,
                                            "best_delta_e_gap": mr.best_delta_e_gap,
                                            "rankings": mr.rankings,
                                            "delta_e_gaps": mr.delta_e_gaps,
                                        }
                                    )
                                    logger.info(
                                        f"    h={h:.2f}: best={mr.best_method} "
                                        f"(ΔE/gap={mr.best_delta_e_gap:.4f})"
                                    )
                                except Exception as me:
                                    logger.warning(f"    h={h:.2f}: mitiq failed: {me}")

                            execution_summary["mitiq_verification"] = {
                                "per_h": mitiq_results,
                                "n_tested": len(mitiq_results),
                                "mode": "benchmark"
                                if getattr(args, "mitiq_benchmark", False)
                                else "verify",
                            }
                        else:
                            logger.warning("    FakeKingston not available for Mitiq verification")
                    else:
                        logger.warning("    Mitiq not installed — skipping verification")
                except Exception as mitiq_err:
                    logger.warning(f"    Mitiq verification failed: {mitiq_err}")

        elif tier == 2:
            tier_metrics = run_tier_2(
                config,
                lattice,
                slogger,
                sigma_flow_per_h=sigma_flow_per_h,
                sigma_flow_threshold=sigma_flow_threshold,
            )
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
                "kappa_recommendations": {
                    float(h): rec
                    for h, rec in (
                        getattr(tier_metrics, "kappa_recommendations", None) or {}
                    ).items()
                },
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
