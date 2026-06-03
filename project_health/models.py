"""Data models for the project health report.

All structured output lives here — no logic, just typed containers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Priority(Enum):
    """Action item priority levels."""

    CRITICAL = 1  # Blocks hardware deployment
    HIGH = 2  # Missing data for thesis tables
    MEDIUM = 3  # Reproducibility / completeness
    LOW = 4  # Nice-to-have coverage


class GapType(Enum):
    """Types of coverage gaps detected."""

    MISSING_P1_NOISELESS = "p1_noiseless_missing"
    INVALID_REGIME = "invalid_regime"
    INSUFFICIENT_SEEDS = "insufficient_seeds"
    MISSING_ZNE = "missing_zne"
    MISSING_EXPERIMENT = "missing_experiment"


@dataclass
class CoverageGap:
    """A single coverage gap identified by the scanner."""

    gap_type: GapType
    topology: str = ""
    n_qubits: int = 0
    p_layers: int = 1
    detail: str = ""
    recommendation: str = ""
    priority: Priority = Priority.MEDIUM


@dataclass
class ActionItem:
    """A concrete action the user should take."""

    priority: Priority
    title: str
    detail: str
    category: str = ""  # "hardware", "coverage", "reproducibility", "thesis"


@dataclass
class ExperimentSummary:
    """Aggregated verdict summary for one experiment."""

    experiment_id: str
    verdict: str  # confirmed, rejected, failed
    criteria: str
    pass_rate: float | None = None
    hypotheses: dict[str, bool] = field(default_factory=dict)
    n_hypotheses: int = 0
    n_confirmed: int = 0


@dataclass
class HealthReport:
    """Complete project health report — the main output."""

    timestamp: str = ""

    # Scanner totals
    n_noiseless: int = 0
    n_noisy: int = 0
    n_experiments: int = 0

    # Verdict aggregation
    n_confirmed: int = 0
    n_rejected: int = 0
    n_failed: int = 0
    experiments: list[ExperimentSummary] = field(default_factory=list)

    # Noiseless quality summary
    noiseless_pass_rate: float = 0.0  # fraction with ΔE/gap < 5%
    noiseless_median_de: float | None = None
    noiseless_by_topology: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Noisy/ZNE summary
    noisy_success_rate: float = 0.0  # fraction meeting criteria
    noisy_mean_r2: float = 0.0
    noisy_mean_gain: float = 0.0

    # Coverage analysis
    gaps: list[CoverageGap] = field(default_factory=list)

    # Delta since last run
    new_results: list[str] = field(default_factory=list)
    removed_results: list[str] = field(default_factory=list)
    n_new: int = 0
    n_removed: int = 0

    # Actionable items (sorted by priority)
    actions: list[ActionItem] = field(default_factory=list)

    # Metadata
    results_dir: str = "results"
    state_file: str = ""

    # Timing
    elapsed_s: float = 0.0

    # Timing
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict with enum→string conversion."""
        return _serialize_dataclass(self)


def _serialize_dataclass(obj: Any) -> Any:
    """Recursively serialize a dataclass, converting Enums to their value."""
    from dataclasses import fields, is_dataclass

    if is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = _serialize_value(value)
        return result
    return obj


def _serialize_value(value: Any) -> Any:
    """Serialize a single value, handling enums, lists, dicts, dataclasses."""
    from dataclasses import is_dataclass
    from enum import Enum

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _serialize_dataclass(value)
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    return value
