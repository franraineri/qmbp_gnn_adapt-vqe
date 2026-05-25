"""Analysis Data Models — Dataclasses for analysis results.

Contains structured output types for gradient analysis, cross-experiment
comparison, and baseline metrics. These are pure data containers with
no computation logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
class ComparisonResult:
    """Result of cross-experiment comparison.

    Stores metrics from comparing two or more experiment runs,
    including statistical summaries and pass/fail criteria.

    Attributes
    ----------
    experiment_id : str
        Identifier for the experiment being compared.
    metric_name : str
        Name of the metric being compared (e.g. "delta_e_over_gap").
    values : list[float]
        Metric values across seeds/runs.
    mean : float
        Mean of values.
    std : float
        Standard deviation of values.
    pass_rate : float
        Fraction of values meeting the threshold criterion.
    threshold : float
        Pass/fail threshold for the metric.
    """

    experiment_id: str
    metric_name: str
    values: list[float]
    mean: float
    std: float
    pass_rate: float
    threshold: float = 0.05


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
