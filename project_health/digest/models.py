"""Data models for the result digest system.

Each dataclass captures the metrics that matter for one result kind:
- NoiselessResult: 4-phase pipeline runs (ΔE/gap, convergence, MPNN quality)
- NoisyResult: ZNE/noise mitigation (R², gain%, mitigation success)
- ExperimentResult: Hypothesis tests (verdict, pass_rate, criteria)

Experiment success criteria and verdict logic are imported from the canonical
source at ``qmbp_simulation.framework.criteria`` and re-exported here for
backward compatibility with existing consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qmbp_simulation.framework.criteria import (
    EXPERIMENT_CRITERIA,
    REJECTION_IS_FINDING,
    compute_verdict,
)

__all__ = [
    "EXPERIMENT_CRITERIA",
    "REJECTION_IS_FINDING",
    "ExperimentResult",
    "NoiselessResult",
    "NoisyResult",
    "ScalingResult",
    "compute_verdict",
]


@dataclass
class NoiselessResult:
    """Key metrics from a noiseless pipeline run (4-phase)."""

    source_file: str
    folder: str
    # System
    n_qubits: int = 0
    p_layers: int = 2
    topology: str = ""
    n_restarts: int = 5
    seed: int | None = None
    # Model identification (for cross-Hamiltonian comparison)
    model: str = "tfim"  # tfim, tfim_longitudinal, heisenberg, xy
    model_params: dict[str, Any] = field(default_factory=dict)  # e.g., {"g": 0.3, "delta": 1.0}
    # Config
    h_values: list[float] = field(default_factory=list)
    h_test: list[float] = field(default_factory=list)
    hidden_dim: int = 128
    n_epochs: int = 6000
    patience: int = 500
    # Phase 1 — ground truth quality
    gap_min: float | None = None  # Minimum spectral gap (criticality indicator)
    phase1_elapsed_s: float = 0.0
    # Phase 4 — primary metrics
    delta_e_over_gap: float | None = None
    phase_label: str = ""
    phase_correct: bool | None = None
    mag_x_error: float | None = None
    corr_zz_error: float | None = None
    # Phase 2 — VQE quality
    convergence_rate: float | None = None
    theta_smoothness: float | None = None
    worst_convergence_h: float | None = None
    mean_iterations: float | None = None  # Average VQE iterations per h-point
    max_restart_spread: float | None = None  # Worst restart energy spread
    phase2_elapsed_s: float = 0.0
    # Phase 3 — MPNN quality
    generalization_gap: float | None = None
    theta_zz_mse: float | None = None
    theta_x_mse: float | None = None  # Complementary observable MSE
    phase3_elapsed_s: float = 0.0
    # Phase 4 — hardware readiness indicators
    error_from_circuit: float | None = None  # VQE ceiling contribution
    error_from_mpnn: float | None = None  # MPNN prediction contribution
    ces_energy_r: float | None = None  # CES-energy Pearson correlation
    classification_confidence: float | None = None  # Phase label reliability
    # Timing
    elapsed_s: float = 0.0
    # Provenance
    run_timestamp: str = ""  # ISO timestamp extracted from filename
    # Identity
    variant_id: str = ""
    description: str = ""


@dataclass
class NoisyResult:
    """Key metrics from a noisy/ZNE run."""

    source_file: str
    folder: str
    # System
    n_qubits: int = 0
    p_layers: int = 2
    topology: str = ""
    seed: int = 42
    # Config
    n_layouts: int = 3
    shots: int = 16384
    h_values: list[float] = field(default_factory=list)
    zne_strategy: str = ""  # "gate_folding", "pea", "ces", "" (legacy/unknown)
    # ZNE — primary metrics
    mean_r2: float = 0.0
    mean_gain_pct: float = 0.0
    n_mitigated_wins: int = 0
    n_total: int = 0
    success_criteria_met: bool = False
    # Error comparison
    mean_de_noiseless: float = 0.0
    mean_de_noisy_raw: float = 0.0
    mean_de_zne: float = 0.0
    # Per-h breakdown
    per_h_r2: list[float] = field(default_factory=list)
    per_h_gain: list[float] = field(default_factory=list)
    # Timing
    elapsed_s: float = 0.0
    # Identity
    variant_id: str = ""


@dataclass
class ExperimentResult:
    """Key metrics from a BaseExperiment hypothesis test."""

    source_file: str
    folder: str
    # Identity
    experiment_id: str = ""
    category: str = ""
    hypothesis: str = ""
    description: str = ""
    # System
    n_qubits: int = 0
    p_layers: int = 2
    topology: str = ""
    model: str = "tfim"  # Model type for cross-Hamiltonian analysis
    h_values: list[float] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    # Verdict
    verdict: str = ""  # confirmed, rejected, failed
    criteria: str = ""
    # Key metrics
    mean_de_gap: float | None = None
    std_de_gap: float | None = None
    pass_rate: float | None = None
    n_seeds: int = 0
    # Timing
    total_time_s: float = 0.0
    # Experiment-specific extras
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingResult:
    """Key metrics from an MPS scaling validation run."""

    source_file: str
    folder: str
    # System
    n_qubits: int = 0
    p_layers: int = 1
    topology: str = "chain_1d"
    strategy: str = "aer_mps"  # aer_mps or tenpy_exact
    chi_max: int = 64
    precision: float = 0.005
    seed: int = 42
    # Results
    h_values: list[float] = field(default_factory=list)
    n_pass: int = 0
    n_total: int = 0
    all_passed: bool = False
    mean_de_gap: float = 0.0
    max_de_gap: float = 0.0
    # Timing
    phase1_time_s: float = 0.0
    phase2_time_s: float = 0.0
    total_time_s: float = 0.0
    # Per-h breakdown
    per_h_de_gap: list[float] = field(default_factory=list)
    per_h_passed: list[bool] = field(default_factory=list)
