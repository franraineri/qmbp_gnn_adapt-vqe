"""Extended metrics for V8 experiments.

Builds on V7's ExperimentMetrics with additional fields for landscape analysis,
scaling studies, and cross-experiment comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class V8Metrics:
    """Extended metrics for a single experiment run point.

    Superset of V7 ExperimentMetrics — adds landscape, timing breakdown,
    and technique-specific fields.
    """

    # Core (same as V7)
    h_value: float
    energy: float
    exact_energy: float
    energy_error: float
    gap: float
    relative_error: float  # DE/gap
    fidelity: float | None = None
    phase_label: str = "unknown"
    phase_correct: bool = True
    wall_time_s: float = 0.0
    n_evaluations: int = 0
    seed: int = 42

    # Extended: VQE details
    n_restarts_used: int = 0  # How many restarts actually ran (B4: may be < configured)
    converged: bool = True
    theta_opt: list[float] = field(default_factory=list)
    theta_init: list[float] = field(default_factory=list)

    # Extended: Landscape (A2, D3, F3)
    hessian_eigenvalues: list[float] | None = None  # B4
    landscape_fluctuation: float | None = None  # F3
    basin_index: int | None = None  # Which basin the optimizer found

    # Extended: MPNN (C1, C3, E3)
    mpnn_mse: float | None = None
    mpnn_energy_error: float | None = None  # Physics-informed validation
    prediction_uncertainty: float | None = None  # E3: ensemble variance

    # Extended: Technique-specific
    technique_metadata: dict[str, Any] = field(default_factory=dict)

    def passes_threshold(self, threshold: float = 0.05) -> bool:
        """Check if DE/gap passes the given threshold."""
        return self.relative_error < threshold

    def validate(self) -> list[str]:
        """Run sanity checks on metrics. Returns list of issues found."""
        issues = []
        if self.relative_error < 0:
            issues.append(f"Negative ΔE/gap: {self.relative_error:.4e}")
        if self.energy_error < 0:
            issues.append(f"Negative energy_error: {self.energy_error:.4e}")
        if self.fidelity is not None and (self.fidelity < -0.001 or self.fidelity > 1.001):
            issues.append(f"Invalid fidelity: {self.fidelity:.6f} (must be in [0,1])")
        if self.gap <= 0:
            issues.append(f"Non-positive gap: {self.gap:.4e}")
        if self.energy > 0 and self.exact_energy < -1.0:
            issues.append(
                f"Energy sign mismatch: E={self.energy:.4f} vs exact={self.exact_energy:.4f}"
            )
        if self.n_evaluations < 0:
            issues.append(f"Negative n_evaluations: {self.n_evaluations}")
        return issues

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        d = asdict(self)
        # Convert numpy types
        for k, v in d.items():
            if isinstance(v, np.floating | np.integer):
                d[k] = float(v) if isinstance(v, np.floating) else int(v)
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d


@dataclass
class WarmColdComparison:
    """Result of comparing warm-start vs cold-start VQE at a single h-point.

    Logged automatically by experiments that use VQE.
    """

    h_value: float
    seed: int

    # Warm-start (from previous h or MPNN prediction)
    warm_init_theta: list[float]
    warm_final_energy: float
    warm_de_gap: float
    warm_n_iterations: int

    # Cold-start (random initialization)
    cold_init_theta: list[float]
    cold_final_energy: float
    cold_de_gap: float
    cold_n_iterations: int

    # Flags
    warm_converged: bool = True
    cold_converged: bool = True

    # Comparison
    gain_pct: float = 0.0  # (cold - warm) / cold * 100
    iteration_savings_pct: float = 0.0  # (cold_nit - warm_nit) / cold_nit * 100

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.floating | np.integer):
                d[k] = float(v) if isinstance(v, np.floating) else int(v)
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d

    @classmethod
    def compute(
        cls,
        h_value: float,
        seed: int,
        warm_init: np.ndarray,
        warm_energy: float,
        warm_de_gap: float,
        warm_nit: int,
        cold_init: np.ndarray,
        cold_energy: float,
        cold_de_gap: float,
        cold_nit: int,
    ) -> WarmColdComparison:
        gain = (cold_de_gap - warm_de_gap) / cold_de_gap * 100 if cold_de_gap > 1e-10 else 0.0
        iter_savings = (cold_nit - warm_nit) / cold_nit * 100 if cold_nit > 0 else 0.0
        return cls(
            h_value=h_value,
            seed=seed,
            warm_init_theta=warm_init.tolist()
            if isinstance(warm_init, np.ndarray)
            else list(warm_init),
            warm_final_energy=float(warm_energy),
            warm_de_gap=float(warm_de_gap),
            warm_n_iterations=int(warm_nit),
            cold_init_theta=cold_init.tolist()
            if isinstance(cold_init, np.ndarray)
            else list(cold_init),
            cold_final_energy=float(cold_energy),
            cold_de_gap=float(cold_de_gap),
            cold_n_iterations=int(cold_nit),
            gain_pct=float(gain),
            iteration_savings_pct=float(iter_savings),
        )


@dataclass
class ComparisonResult:
    """Result of comparing an experiment against a baseline.

    Provides structured improvement/regression metrics.
    """

    experiment_id: str
    baseline_id: str
    system_desc: str  # e.g., "N=6, h=1.5, chain_1d"

    # Core comparison
    exp_de_gap: float
    baseline_de_gap: float
    improvement_pct: float  # (baseline - exp) / baseline * 100

    # Time comparison
    exp_time_s: float
    baseline_time_s: float
    speedup: float  # baseline_time / exp_time

    # Statistical (across seeds)
    exp_de_gap_std: float = 0.0
    baseline_de_gap_std: float = 0.0
    n_seeds: int = 1

    # Verdict
    statistically_significant: bool = False  # t-test p < 0.05
    verdict: str = "neutral"  # "improvement", "regression", "neutral"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def compute(
        cls,
        experiment_id: str,
        baseline_id: str,
        system_desc: str,
        exp_metrics: list[V8Metrics],
        baseline_metrics: list[V8Metrics],
    ) -> ComparisonResult:
        """Compute comparison from lists of metrics (one per seed)."""
        exp_errors = [m.relative_error for m in exp_metrics]
        base_errors = [m.relative_error for m in baseline_metrics]
        exp_times = [m.wall_time_s for m in exp_metrics]
        base_times = [m.wall_time_s for m in baseline_metrics]

        exp_mean = float(np.mean(exp_errors))
        base_mean = float(np.mean(base_errors))
        exp_std = float(np.std(exp_errors)) if len(exp_errors) > 1 else 0.0
        base_std = float(np.std(base_errors)) if len(base_errors) > 1 else 0.0

        improvement = (base_mean - exp_mean) / base_mean * 100 if abs(base_mean) > 1e-10 else 0.0
        speedup = np.mean(base_times) / np.mean(exp_times) if np.mean(exp_times) > 1e-6 else 1.0

        # Simple significance test (Welch's t-test)
        significant = False
        if len(exp_errors) >= 3 and len(base_errors) >= 3:
            from scipy.stats import ttest_ind

            _, p_value = ttest_ind(exp_errors, base_errors, equal_var=False)
            significant = p_value < 0.05

        # Verdict
        if improvement > 10 and significant:
            verdict = "improvement"
        elif improvement < -10 and significant:
            verdict = "regression"
        else:
            verdict = "neutral"

        return cls(
            experiment_id=experiment_id,
            baseline_id=baseline_id,
            system_desc=system_desc,
            exp_de_gap=exp_mean,
            baseline_de_gap=base_mean,
            improvement_pct=improvement,
            exp_time_s=float(np.mean(exp_times)),
            baseline_time_s=float(np.mean(base_times)),
            speedup=float(speedup),
            exp_de_gap_std=exp_std,
            baseline_de_gap_std=base_std,
            n_seeds=len(exp_errors),
            statistically_significant=significant,
            verdict=verdict,
        )
