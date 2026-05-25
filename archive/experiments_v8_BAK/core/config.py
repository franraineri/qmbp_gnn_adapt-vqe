"""Experiment configuration dataclasses for V8 suite.

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
    # Extended fields for new models (E4)
    g_longitudinal: float = 0.0  # Longitudinal field strength
    boundary: str = "open"  # "open" or "periodic"
    model: str = "tfim"  # "tfim", "tfim_longitudinal", "heisenberg"

    @property
    def n_params(self) -> int:
        return 2 * self.p_layers


@dataclass
class VQEConfig:
    """VQE optimizer configuration."""

    optimizer: str = "L-BFGS-B"
    n_restarts: int = 5
    maxiter: int = 1000
    sigma: float = 0.1  # Restart perturbation scale
    ftol: float = 1e-14
    # Analytical init (B1)
    use_analytical_init: bool = False
    # Parameter freezing (B2)
    freeze_params: list[int] | None = None
    freeze_after_h: float | None = None
    # Hessian-guided (B4)
    use_hessian_check: bool = False
    hessian_escape_threshold: float = -1e-6
    # DyPP (F1)
    use_dypp: bool = False
    dypp_order: int = 2  # Linear=1, Quadratic=2


@dataclass
class MPNNConfig:
    """MPNN predictor configuration."""

    hidden_dim: int = 64
    n_layers: int = 3
    n_epochs: int = 6000
    lr: float = 1e-3
    patience: int = 300
    dropout: float = 0.1
    # Physics-informed loss (C1)
    use_physics_loss: bool = False
    physics_loss_weight: float = 0.1
    physics_loss_start_epoch: int = 1000
    physics_loss_eval_every: int = 100
    # Sign equivariance (C3)
    sign_canonicalization: str = "none"  # "none", "enforce_positive", "min_loss"
    # Active learning (E3)
    use_active_learning: bool = False
    n_ensemble: int = 5
    acquisition: str = "max_variance"  # "max_variance", "expected_improvement"


@dataclass
class AnalysisConfig:
    """Analysis and landscape configuration."""

    # Landscape (A2, D3)
    landscape_resolution: int = 20  # Grid points per dimension
    tci_max_rank: int = 10
    tci_tolerance: float = 1e-4
    # Finite-size scaling (A3)
    scaling_n_values: list[int] = field(default_factory=lambda: [4, 6, 8, 10, 14, 20])
    # Weight space (D1)
    weight_gradient_n_points: int = 40
    # Fluctuation (F3)
    fluctuation_n_samples: int = 100


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    Combines all sub-configs and adds experiment metadata.
    Fully serializable for reproducibility.
    """

    experiment_id: str = "unnamed"
    category: str = "X"  # A, B, C, D, E, F
    description: str = ""
    hypothesis: str = ""

    system: SystemConfig = field(default_factory=SystemConfig)
    vqe: VQEConfig = field(default_factory=VQEConfig)
    mpnn: MPNNConfig = field(default_factory=MPNNConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    verbose: bool = False
    debug: bool = False

    # Comparison
    compare_with_baseline: bool = True
    baseline_config: str = "v61_optimal"  # Which baseline to compare against

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
        )

    def validate(self) -> list[str]:
        """Validate config against project constraints.

        Returns list of warnings. Raises ValueError for hard constraint violations.
        """
        warnings = []
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
