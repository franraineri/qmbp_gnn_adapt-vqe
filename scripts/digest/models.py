"""Data models for the result digest system.

Each dataclass captures the metrics that matter for one result kind:
- NoiselessResult: 4-phase pipeline runs (ΔE/gap, convergence, MPNN quality)
- NoisyResult: ZNE/noise mitigation (R², gain%, mitigation success)
- ExperimentResult: Hypothesis tests (verdict, pass_rate, criteria)

No external dependencies — stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Experiment criteria (mirrored from result_store.py to avoid heavy imports)
# Thresholds calibrated against actual project findings (see documentation/analysis/04_verdict_reconciliation.md)
EXPERIMENT_CRITERIA: dict[str, dict[str, Any]] = {
    "A3": {"metric": "pass_rate", "threshold": 1.0, "desc": "Scaling law R²>0.99"},
    "A3_N20": {"metric": "pass_rate", "threshold": 1.0, "desc": "Scaling at N=20"},
    "B1": {"metric": "mean_de_gap", "threshold": 0.05, "desc": "ΔE/gap < 5%"},
    "B2": {"metric": "pass_rate", "threshold": 0.60, "desc": "Freeze works at h≥1.5"},
    "B4": {
        "metric": "pass_rate",
        "threshold": 0.70,
        "desc": "No saddle points (physics-limited pts excluded)",
    },
    "C1": {"metric": "mean_de_gap", "threshold": 0.05, "desc": "Physics loss < 5%"},
    "C3": {"metric": "mean_de_gap", "threshold": 0.05, "desc": "N=20 VQE < 5%"},
    "D1": {"metric": "pass_rate", "threshold": 0.0, "desc": "Gradient peak detected near h_c"},
    "E4": {"metric": "pass_rate", "threshold": 0.5, "desc": "HVA fails at g>0"},
    "F1": {"metric": "pass_rate", "threshold": 0.8, "desc": "DyPP > 30%"},
    "F3": {"metric": "pass_rate", "threshold": 0.0, "desc": "Fluctuation > 1.0 everywhere"},
    "G1": {"metric": "pass_rate", "threshold": 0.80, "desc": "≤9 pts sufficient"},
    "G2": {"metric": "pass_rate", "threshold": 0.8, "desc": "Ensemble r > 0.7"},
    "G3": {"metric": "mean_de_gap", "threshold": 0.05, "desc": "N=20 < 5%"},
    "G4": {"metric": "pass_rate", "threshold": 0.8, "desc": "κ predicts restarts"},
    "G5": {"metric": "pass_rate", "threshold": 0.85, "desc": "Seed-independent (std<0.01)"},
}
REJECTION_IS_FINDING = {"E4", "F1", "G2", "G3", "G4"}


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
    # Config
    h_values: list[float] = field(default_factory=list)
    h_test: list[float] = field(default_factory=list)
    hidden_dim: int = 128
    n_epochs: int = 6000
    patience: int = 500
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
    # Phase 3 — MPNN quality
    generalization_gap: float | None = None
    theta_zz_mse: float | None = None
    # Timing
    elapsed_s: float = 0.0
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
