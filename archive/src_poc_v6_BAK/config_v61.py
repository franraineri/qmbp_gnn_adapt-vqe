"""
V6.1 configuration constants and data models for hardware deployment.

This module defines all configuration constants and dataclasses introduced
in V6.1 for the hardware deployment pipeline (inhomogeneous ZNE, shot budget
scaling, Pauli twirling, NN extrapolation) and MPNN enhancements (weight
gradient analysis, per-parameter heads, edge features).

The existing ``config.py`` remains untouched — this module extends it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Shot budget thresholds
# ---------------------------------------------------------------------------

SHOT_BUDGET_SMALL = 8192  # N ≤ 6
SHOT_BUDGET_MEDIUM = 16384  # 7 ≤ N ≤ 10
SHOT_BUDGET_LARGE = 32768  # N > 10
MIN_SHOT_OVERRIDE = 4096  # Minimum allowed user override

# ---------------------------------------------------------------------------
# Twirling defaults
# ---------------------------------------------------------------------------

DEFAULT_NUM_RANDOMIZATIONS = 32
DEFAULT_SHOTS_PER_RANDOMIZATION = 256

# ---------------------------------------------------------------------------
# ZNE configuration
# ---------------------------------------------------------------------------

MIN_LAYOUTS = 3
MAX_LAYOUTS = 10
MIN_CES_RATIO = 1.5  # Minimum CES spread for meaningful extrapolation (lowered from 2.0 for small circuits on large backends)
MAX_CES_RATIO = 10.0  # Maximum allowed CES ratio (max/min) to prevent outlier layouts
ZNE_R_SQUARED_WARNING_THRESHOLD = 0.8

# ---------------------------------------------------------------------------
# AdaptVQE hardware limits
# ---------------------------------------------------------------------------

MAX_ADAPT_ITERATIONS_HARDWARE = 2
MAX_TWO_QUBIT_GATES_P2 = 24  # Approximate for N=6

# ---------------------------------------------------------------------------
# NN Extrapolator
# ---------------------------------------------------------------------------

NN_HIDDEN_LAYERS = (16, 8)
NN_MAX_ITER = 1000
NN_MIN_DATA_POINTS = 5  # MLP has ~169 params; need ≥5 points to avoid overfitting

# ---------------------------------------------------------------------------
# Calibration freshness
# ---------------------------------------------------------------------------

CALIBRATION_MAX_AGE_HOURS = 24

# ---------------------------------------------------------------------------
# MPNN Enhancement Constants — Weight Gradient Analyzer
# ---------------------------------------------------------------------------

GRADIENT_CRITICAL_REGION = (0.8, 1.4)  # h range for peak detection
GRADIENT_PEAK_PROMINENCE = 0.1  # Minimum prominence for peak detection

# ---------------------------------------------------------------------------
# MPNN Enhancement Constants — Per-Parameter Heads
# ---------------------------------------------------------------------------

DEFAULT_PER_PARAMETER_HEADS = False  # Backward compatible default

# ---------------------------------------------------------------------------
# MPNN Enhancement Constants — Edge Features
# ---------------------------------------------------------------------------

DEFAULT_EDGE_FEATURE_DIM = 1  # Single J_ij value per edge
DEFAULT_USE_EDGE_FEATURES = False  # GINConv default
NNCONV_EDGE_MLP_HIDDEN = 32  # Hidden dim for NNConv edge network


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class DeployResultV61:
    """Extended deployment result with hardware provenance and mitigation metadata.

    Backward-compatible with V6.0 DeployResult fields, plus hardware-specific
    extensions for ZNE data, provenance, and uncertainty quantification.

    Serialization note
    ------------------
    The fields ``per_site_mag_x`` and ``per_bond_corr_zz`` are
    ``np.ndarray | None``.  For JSON serialization, callers should convert
    them to ``list[float]`` via ``.tolist()`` before encoding (numpy arrays
    are not JSON-serializable by default).
    """

    # ── V6.0 compatible fields ──
    route: str
    h_test: float
    predicted_energy: float
    delta_e: float
    delta_e_over_gap: float
    mag_x_pred: float
    corr_zz_pred: float
    mag_x_error: float
    corr_zz_error: float
    fidelity: float | None
    adapt_iterations: int
    phase_label: str
    metrics_checklist: dict[str, bool]

    # ── V6.1 hardware extensions ──
    mode: str  # "hardware" or "simulation"
    backend_name: str | None
    job_id: str | None
    calibration_date: str | None
    execution_timestamp: str | None
    total_shots: int

    # ── ZNE data ──
    ces_values: list[float]
    energies_per_layout: list[float]
    zne_r_squared: float | None
    nn_fit_loss: float | None
    extrapolation_method: str  # "linear" | "nn" | "none"

    # ── Raw vs mitigated ──
    raw_energy: float | None
    raw_mag_x: float | None
    raw_corr_zz: float | None

    # ── Uncertainty ──
    sigma: float  # Statistical uncertainty 1/√shots
    per_site_mag_x: np.ndarray | None
    per_bond_corr_zz: np.ndarray | None


@dataclass
class LayoutResult:
    """Result of layout selection for a single qubit mapping.

    Attributes
    ----------
    initial_layout : list[int]
        Physical qubit indices for this layout.
    ces : float
        Circuit Error Sum for this layout.
    two_qubit_gate_count : int
        Number of two-qubit gates after transpilation.
    """

    initial_layout: list[int]
    ces: float
    two_qubit_gate_count: int


@dataclass
class GradientAnalysisResult:
    """Output of the weight gradient analysis across the h-sweep.

    Contains per-layer and total gradient norms for each h-value,
    enabling visualization of phase transition signatures in weight space.

    Attributes
    ----------
    h_values : np.ndarray
        Array of h-values analyzed.
    total_gradient_norms : np.ndarray
        Total gradient norm ‖∂L/∂W‖₂ at each h-value.
    per_layer_gradient_norms : dict[str, np.ndarray]
        Gradient norms per layer (e.g. ``{"ginconv_0": [...], "head": [...]}``)
    peak_h_values : list[float]
        h-values where gradient norm peaks are detected.
    peak_magnitudes : list[float]
        Gradient norm magnitude at each detected peak.
    critical_region_detected : bool
        Whether peaks were found in h ∈ [0.8, 1.4].
    """

    h_values: np.ndarray
    total_gradient_norms: np.ndarray
    per_layer_gradient_norms: dict[str, np.ndarray]
    peak_h_values: list[float]
    peak_magnitudes: list[float]
    critical_region_detected: bool


@dataclass
class MPNNCheckpoint:
    """Model checkpoint with architecture metadata for save/load compatibility.

    Stores both the model state dict and the configuration needed to
    reconstruct the correct architecture (single-head vs per-parameter-heads,
    GINConv vs NNConv).

    Attributes
    ----------
    state_dict : dict
        Model weights.
    architecture : str
        Architecture type (``"ginconv"`` or ``"nnconv"``).
    per_parameter_heads : bool
        Whether dual output heads are used.
    node_features : int
        Input node feature dimension.
    hidden_dim : int
        Hidden layer dimension.
    n_layers : int
        Number of message passing layers.
    output_dim : int
        Total output dimension (2p).
    use_edge_features : bool
        Whether edge features are expected.
    edge_feature_dim : int
        Edge feature dimension (1 for J_ij).
    training_metadata : dict
        Training info (epoch, loss, dataset details).
    """

    state_dict: dict
    architecture: str  # "ginconv" | "nnconv"
    per_parameter_heads: bool
    node_features: int
    hidden_dim: int
    n_layers: int
    output_dim: int
    use_edge_features: bool
    edge_feature_dim: int
    training_metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Noisy Simulation V6.1 — Data Models
# ---------------------------------------------------------------------------


@dataclass
class NoisySweepResult:
    """Result of a single h-point deployment across three noise modes.

    Compares noiseless (StatevectorEstimator), noisy-raw (FakeTorino, 1 layout),
    and ZNE-mitigated (FakeTorino, 3+ layouts) to quantify ZNE effectiveness.

    Attributes
    ----------
    h_test : float
        Transverse field value tested.
    noiseless : DeployResultV61
        Noiseless simulation result (baseline truth).
    noisy_raw : DeployResultV61
        Noisy simulation without ZNE (single layout, raw).
    mitigated : DeployResultV61
        Noisy simulation with inhomogeneous ZNE (multi-layout extrapolation).
    zne_gain_energy : float
        Relative ZNE improvement in ΔE/gap:
        ``(noisy_raw.delta_e_over_gap - mitigated.delta_e_over_gap) / noisy_raw.delta_e_over_gap``
    zne_gain_mag_x : float
        Relative ZNE improvement in ⟨X⟩ error:
        ``(noisy_raw.mag_x_error - mitigated.mag_x_error) / noisy_raw.mag_x_error``
        Set to 0.0 if noisy_raw.mag_x_error == 0.
    mitigated_better : bool
        True iff ``mitigated.delta_e_over_gap < noisy_raw.delta_e_over_gap``.
    """

    h_test: float
    noiseless: DeployResultV61
    noisy_raw: DeployResultV61
    mitigated: DeployResultV61
    zne_gain_energy: float
    zne_gain_mag_x: float
    mitigated_better: bool


@dataclass
class SweepSummary:
    """Aggregated summary of a multi-point noisy simulation sweep.

    Collects results across all h-values and evaluates success criteria
    for ZNE effectiveness validation.

    Attributes
    ----------
    timestamp : str
        ISO-8601 timestamp of the sweep execution.
    n_qubits : int
        Number of qubits (N) used in the sweep.
    h_values : list[float]
        List of h-values tested.
    shots : int
        Shot budget per noisy deployment.
    n_layouts_mitigated : int
        Number of layouts used for ZNE extrapolation.
    results : list[NoisySweepResult]
        Per-h-point results.
    n_mitigated_wins : int
        Count of h-points where ZNE mitigated ΔE/gap < noisy raw ΔE/gap.
    n_good_r_squared : int
        Count of h-points where ZNE R² > 0.8.
    success_criteria_met : bool
        True iff ``n_mitigated_wins >= 4 AND n_good_r_squared >= 3``.
    """

    timestamp: str
    n_qubits: int
    h_values: list[float]
    shots: int
    n_layouts_mitigated: int
    results: list[NoisySweepResult]
    n_mitigated_wins: int
    n_good_r_squared: int
    success_criteria_met: bool


# ---------------------------------------------------------------------------
# Random Baseline Comparison — Data Models
# ---------------------------------------------------------------------------


@dataclass
class BaselineMetrics:
    """Metrics for a single deployment attempt (warm-start or cold-start).

    Attributes
    ----------
    theta_init : list[float]
        Initial parameters used (serializable).
    predicted_energy : float
        Final energy after deployment.
    delta_e : float
        |E_pred - E_exact|.
    delta_e_over_gap : float
        ΔE / gap (primary metric).
    fidelity : float | None
        State fidelity (simulation only).
    adapt_iterations : int
        Number of AdaptVQE iterations used.
    phase_label : str
        Classified phase.
    phase_correct : bool
        Whether phase classification matches exact.
    """

    theta_init: list[float]
    predicted_energy: float
    delta_e: float
    delta_e_over_gap: float
    fidelity: float | None
    adapt_iterations: int
    phase_label: str
    phase_correct: bool


@dataclass
class BaselineComparison:
    """Comparison between MPNN warm-start and random cold-start deployment.

    Quantifies the value of the MPNN warm-start by comparing against
    random initialization. This is the core thesis metric: how much
    does the GNN prediction improve over naive random search?

    Attributes
    ----------
    n_random_seeds : int
        Number of random initializations tested.
    random_seeds : list[int]
        Seeds used for reproducibility.
    warm_start : BaselineMetrics
        Metrics from MPNN-predicted θ.
    cold_start_mean : dict
        Mean metrics across random initializations.
    cold_start_std : dict
        Std of metrics across random initializations.
    cold_start_per_seed : list[BaselineMetrics]
        Per-seed breakdown (for debugging).
    gain_energy_pct : float
        (ΔE_cold - ΔE_warm) / ΔE_cold × 100. Positive = warm-start better.
    gain_fidelity_abs : float | None
        fid_warm - fid_cold (only in simulation mode). Positive = warm-start better.
    warm_start_sufficient : bool
        True if warm-start achieves ΔE/gap < 5% without refinement.
    cold_start_any_success : bool
        True if any cold-start seed achieves ΔE/gap < 5%.
    """

    n_random_seeds: int
    random_seeds: list[int]
    warm_start: BaselineMetrics
    cold_start_mean: dict
    cold_start_std: dict
    cold_start_per_seed: list[BaselineMetrics]
    gain_energy_pct: float
    gain_fidelity_abs: float | None
    warm_start_sufficient: bool
    cold_start_any_success: bool
