"""Experiment configuration dataclasses.

Provides typed, validated configuration for all experiments.
Serializable to/from JSON for reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SystemConfig:
    """Physical system configuration."""

    n_qubits: int = 6
    p_layers: int = 2
    topology: str = "chain_1d"
    J: float = 1.0
    h_values: list[float] = field(default_factory=lambda: [1.5])
    h_test: list[float] = field(default_factory=lambda: [1.5])
    g_longitudinal: float = 0.0
    boundary: str = "open"
    model: str = "tfim"

    @property
    def n_params(self) -> int:
        """Number of variational parameters (2 per HVA layer)."""
        return 2 * self.p_layers


@dataclass
class VQEConfig:
    """VQE optimizer configuration."""

    optimizer: str = "L-BFGS-B"
    n_restarts: int = 5
    maxiter: int = 1000
    sigma: float = 0.1
    ftol: float = 1e-14
    use_analytical_init: bool = False
    freeze_params: list[int] | None = None
    freeze_after_h: float | None = None
    use_hessian_check: bool = False
    hessian_escape_threshold: float = -1e-6
    use_dypp: bool = False
    dypp_order: int = 2


@dataclass
class MPNNConfig:
    """MPNN predictor configuration."""

    hidden_dim: int = 64
    n_layers: int = 3
    n_epochs: int = 6000
    lr: float = 1e-3
    patience: int = 300
    dropout: float = 0.1
    use_physics_loss: bool = False
    physics_loss_weight: float = 0.1
    physics_loss_start_epoch: int = 1000
    physics_loss_eval_every: int = 100
    sign_canonicalization: str = "none"
    use_active_learning: bool = False
    n_ensemble: int = 5
    acquisition: str = "max_variance"


@dataclass
class AnalysisConfig:
    """Analysis and landscape configuration."""

    landscape_resolution: int = 20
    tci_max_rank: int = 10
    tci_tolerance: float = 1e-4
    scaling_n_values: list[int] = field(default_factory=lambda: [4, 6, 8, 10, 14, 20])
    weight_gradient_n_points: int = 40
    fluctuation_n_samples: int = 100
    threshold: float = 0.05
    compute_hessian: bool = False


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    Combines all sub-configs and adds experiment metadata.
    Fully serializable for reproducibility.
    """

    experiment_id: str = "unnamed"
    category: str = "X"
    description: str = ""
    hypothesis: str = ""

    system: SystemConfig = field(default_factory=SystemConfig)
    vqe: VQEConfig = field(default_factory=VQEConfig)
    mpnn: MPNNConfig = field(default_factory=MPNNConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    verbose: bool = False
    debug: bool = False

    compare_with_baseline: bool = True
    baseline_config: str = "v61_optimal"

    auto_warm_cold_comparison: bool = True
    """When True, experiments that use VQE should call run_warm_cold_comparison()
    and include results in saved output. Experiments that don't use VQE (e.g.,
    landscape analysis) can set this to False to skip the comparison.
    The framework provides the method but does not force invocation — subclasses
    decide when warm-start parameters are available."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return asdict(self)

    def to_json(self, path: Path) -> None:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> ExperimentConfig:
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            experiment_id=data.get("experiment_id", "unnamed"),
            category=data.get("category", "X"),
            description=data.get("description", ""),
            hypothesis=data.get("hypothesis", ""),
            system=SystemConfig(**data.get("system", {})),
            vqe=VQEConfig(**data.get("vqe", {})),
            mpnn=MPNNConfig(**data.get("mpnn", {})),
            analysis=AnalysisConfig(**data.get("analysis", {})),
            seeds=data.get("seeds", [42, 43, 44]),
            verbose=data.get("verbose", False),
            debug=data.get("debug", False),
            compare_with_baseline=data.get("compare_with_baseline", True),
            baseline_config=data.get("baseline_config", "v61_optimal"),
            auto_warm_cold_comparison=data.get("auto_warm_cold_comparison", True),
        )

    def validate(self) -> list[str]:
        """Validate config against project constraints.

        Returns list of warnings. Raises ValueError for hard constraint violations.
        """
        warnings: list[str] = []

        # Hard constraints — raise immediately
        if self.system.p_layers > 2:
            raise ValueError(
                "CONSTRAINT VIOLATION: p_layers > 2 is forbidden (Mele et al. 2022). "
                "HVA depth is limited to p ≤ 2 for this framework."
            )
        if not self.seeds:
            raise ValueError("No seeds specified — at least one seed is required.")

        # Soft warnings
        if self.system.n_qubits > 20 and self.system.topology != "chain_1d":
            warnings.append("N>20 with non-chain topology may be infeasible")
        if self.system.n_qubits == 12:
            warnings.append("N=12 is very slow (>30 min per run). Consider N=10 or N=14.")
        if self.vqe.n_restarts > 10:
            warnings.append("n_restarts > 10 is likely wasteful (diminishing returns)")
        return warnings
