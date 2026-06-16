"""Data models and enumerations for thesis extension analysis.

Provides ExtensionClassification enum, RejectionReport, ExtensionResult,
ExtensionAnalysisResult dataclasses, and custom exception types.

Req: 4.4, 4.5
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ExtensionClassification(Enum):
    """All valid classification outcomes for extension evaluations."""

    VIABLE = "VIABLE"
    CONDITIONALLY_VIABLE = "CONDITIONALLY_VIABLE"
    REJECTED_INSUFFICIENT_DATA = "REJECTED_INSUFFICIENT_DATA"
    HARDWARE_INCOMPATIBLE = "HARDWARE_INCOMPATIBLE"
    EXPRESSIBILITY_INSUFFICIENT = "EXPRESSIBILITY_INSUFFICIENT"
    OVERPARAMETERIZED_FOR_DATASET = "OVERPARAMETERIZED_FOR_DATASET"
    DEGRADED_VS_BASELINE = "DEGRADED_VS_BASELINE"
    HARD_PHYSICS_LIMIT = "HARD_PHYSICS_LIMIT"
    PREREQUISITE_FAILED = "PREREQUISITE_FAILED"


#: Set of classifications that constitute a rejection (non-viable) outcome.
REJECTION_CLASSIFICATIONS: frozenset[ExtensionClassification] = frozenset(
    {
        ExtensionClassification.REJECTED_INSUFFICIENT_DATA,
        ExtensionClassification.HARDWARE_INCOMPATIBLE,
        ExtensionClassification.EXPRESSIBILITY_INSUFFICIENT,
        ExtensionClassification.OVERPARAMETERIZED_FOR_DATASET,
        ExtensionClassification.DEGRADED_VS_BASELINE,
        ExtensionClassification.HARD_PHYSICS_LIMIT,
        ExtensionClassification.PREREQUISITE_FAILED,
    }
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PrerequisiteFailedError(Exception):
    """Raised when a hard prerequisite (import, data) is unavailable.

    Caught by the orchestrator; the affected extension is marked
    PREREQUISITE_FAILED and the others continue (Req 4.4).
    """

    def __init__(self, extension_id: str, reason: str) -> None:
        self.extension_id = extension_id
        self.reason = reason
        super().__init__(f"[{extension_id}] PREREQUISITE_FAILED: {reason}")


class HardPhysicsLimitError(Exception):
    """Raised when an established physics constraint blocks evaluation.

    Immediately generates a RejectionReport with HARD_PHYSICS_LIMIT
    without running new experiments (Req 2.9).
    """

    def __init__(self, constraint: str, reference: str) -> None:
        self.constraint = constraint
        self.reference = reference
        super().__init__(f"HARD_PHYSICS_LIMIT: {constraint} (ref: {reference})")


# ---------------------------------------------------------------------------
# RejectionReport
# ---------------------------------------------------------------------------


@dataclass
class RejectionReport:
    """Structured rejection report satisfying Req 4.5.

    Every rejection event generates one of these with all required fields
    non-empty; the narrative is usable directly in the thesis.
    """

    extension_id: str  # "ext1", "ext2", "ext3"
    criterion_id: str  # e.g. "1.3", "2.9"
    criterion_description: str
    classification: ExtensionClassification
    measured_value: float | str
    threshold: float | str
    narrative: str  # One paragraph for thesis
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "extension_id": self.extension_id,
            "criterion_id": self.criterion_id,
            "criterion_description": self.criterion_description,
            "classification": self.classification.value,
            "measured_value": self.measured_value,
            "threshold": self.threshold,
            "narrative": self.narrative,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# ExtensionResult
# ---------------------------------------------------------------------------


@dataclass
class ExtensionResult:
    """Unified result for a single extension evaluation."""

    extension_id: str
    classification: ExtensionClassification
    key_metric: float | str  # The deciding quantitative value
    thesis_narrative: str
    thesis_chapter_section: str  # e.g. "5.3.1"
    rejection_report: RejectionReport | None
    hardware_viable: bool
    estimated_time_to_result_hours: float
    implementation_risk: str  # "low" | "medium" | "high"
    raw_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "extension_id": self.extension_id,
            "classification": self.classification.value,
            "key_metric": self.key_metric,
            "thesis_narrative": self.thesis_narrative,
            "thesis_chapter_section": self.thesis_chapter_section,
            "hardware_viable": self.hardware_viable,
            "estimated_time_to_result_hours": self.estimated_time_to_result_hours,
            "implementation_risk": self.implementation_risk,
            "raw_metrics": self.raw_metrics,
            "rejection_report": (
                self.rejection_report.to_dict() if self.rejection_report is not None else None
            ),
        }
        return d


# ---------------------------------------------------------------------------
# ExtensionAnalysisResult  (top-level output)
# ---------------------------------------------------------------------------


@dataclass
class ExtensionAnalysisResult:
    """Top-level output of ThesisExtensionAnalyzer.run()."""

    run_timestamp: str
    ext1_bond_resolved: ExtensionResult
    ext2_kagome: ExtensionResult
    ext3_normalizing_flows: ExtensionResult
    priority_ranking: list[str] = field(default_factory=list)
    ranking_rationale: dict[str, str] = field(default_factory=dict)
    prerequisite_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_timestamp": self.run_timestamp,
            "ext1_bond_resolved": self.ext1_bond_resolved.to_dict(),
            "ext2_kagome": self.ext2_kagome.to_dict(),
            "ext3_normalizing_flows": self.ext3_normalizing_flows.to_dict(),
            "priority_ranking": self.priority_ranking,
            "ranking_rationale": self.ranking_rationale,
            "prerequisite_failures": self.prerequisite_failures,
        }

    def to_json(self, path: str) -> None:
        """Serialize to a JSON file, creating parent directories if needed."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    """Fallback serializer for json.dump."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "value"):  # Enum
        return obj.value
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
