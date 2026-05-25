"""Experiment metrics for framework runs.

Provides structured metrics for single experiment points and
warm-start vs cold-start comparisons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class ExperimentMetrics:
    """Metrics for a single experiment run point.

    Captures energy accuracy, fidelity, timing, and technique-specific
    metadata for post-hoc analysis.
    """

    # Core metrics
    h_value: float
    energy: float
    exact_energy: float
    energy_error: float
    gap: float
    relative_error: float  # ΔE/gap
    fidelity: float | None = None
    phase_label: str = "unknown"
    phase_correct: bool = True
    wall_time_s: float = 0.0
    n_evaluations: int = 0
    seed: int = 42

    # VQE details
    n_restarts_used: int = 0
    converged: bool = True
    theta_opt: list[float] = field(default_factory=list)
    theta_init: list[float] = field(default_factory=list)

    # Landscape
    hessian_eigenvalues: list[float] | None = None
    landscape_fluctuation: float | None = None
    basin_index: int | None = None

    # MPNN
    mpnn_mse: float | None = None
    mpnn_energy_error: float | None = None
    prediction_uncertainty: float | None = None

    # Technique-specific
    technique_metadata: dict[str, Any] = field(default_factory=dict)

    def passes_threshold(self, threshold: float = 0.05) -> bool:
        """Check if ΔE/gap passes the given threshold."""
        return self.relative_error < threshold

    def validate(self) -> list[str]:
        """Run sanity checks on metrics. Returns list of issues found."""
        issues: list[str] = []
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
        for k, v in d.items():
            if isinstance(v, np.floating | np.integer):
                d[k] = float(v) if isinstance(v, np.floating) else int(v)
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d


@dataclass
class WarmColdComparison:
    """Result of comparing warm-start vs cold-start VQE at a single h-point."""

    h_value: float
    seed: int

    # Warm-start
    warm_init_theta: list[float]
    warm_final_energy: float
    warm_de_gap: float
    warm_n_iterations: int

    # Cold-start
    cold_init_theta: list[float]
    cold_final_energy: float
    cold_de_gap: float
    cold_n_iterations: int

    # Flags
    warm_converged: bool = True
    cold_converged: bool = True

    # Comparison
    gain_pct: float = 0.0
    iteration_savings_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
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
        """Compute comparison from raw VQE results."""
        gain = (cold_de_gap - warm_de_gap) / cold_de_gap * 100 if cold_de_gap > 1e-10 else 0.0
        iter_savings = (cold_nit - warm_nit) / cold_nit * 100 if cold_nit > 0 else 0.0
        return cls(
            h_value=h_value,
            seed=seed,
            warm_init_theta=(
                warm_init.tolist() if isinstance(warm_init, np.ndarray) else list(warm_init)
            ),
            warm_final_energy=float(warm_energy),
            warm_de_gap=float(warm_de_gap),
            warm_n_iterations=int(warm_nit),
            cold_init_theta=(
                cold_init.tolist() if isinstance(cold_init, np.ndarray) else list(cold_init)
            ),
            cold_final_energy=float(cold_energy),
            cold_de_gap=float(cold_de_gap),
            cold_n_iterations=int(cold_nit),
            gain_pct=float(gain),
            iteration_savings_pct=float(iter_savings),
        )
