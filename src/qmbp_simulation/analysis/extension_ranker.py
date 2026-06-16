"""Extension priority ranker for thesis extension analysis.

Produces a valid total order (transitive, antisymmetric) over extensions
by composite score: (narrative_impact, implementation_risk, -time_to_result).

Req: 5.8, 5.9
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExtensionScore:
    """Scoring components for a single extension.

    Attributes:
        extension_id: Identifier string, e.g. "ext1", "ext2", "ext3".
        narrative_impact: 0-3 where 3=HIGH, 2=MEDIUM, 1=LOW, 0=REJECTED.
        implementation_risk: 0-3 where 3=low_risk, 2=medium, 1=high.
        time_to_result: Estimated hours (lower is better).
    """

    extension_id: str
    narrative_impact: float  # 0-3: 3=HIGH, 2=MEDIUM, 1=LOW, 0=REJECTED
    implementation_risk: float  # 0-3: 3=low_risk, 2=medium, 1=high
    time_to_result: float  # hours (lower → better)

    def to_dict(self) -> dict[str, Any]:
        return {
            "extension_id": self.extension_id,
            "narrative_impact": self.narrative_impact,
            "implementation_risk": self.implementation_risk,
            "time_to_result": self.time_to_result,
        }


class ExtensionPriorityRanker:
    """Rank extensions by composite score.

    Primary sort key:  narrative_impact    (descending, higher = more important)
    Secondary key:     implementation_risk (descending, 3=low_risk preferred)
    Tertiary key:      -time_to_result     (ascending hours, faster preferred)

    The composite tuple (narrative_impact, implementation_risk, -time_to_result)
    ensures a valid total order: transitive and antisymmetric (Req 5.8, Property 9).
    """

    def rank(self, scores: list[ExtensionScore]) -> list[str]:
        """Return extension IDs sorted from highest to lowest priority.

        Args:
            scores: List of ExtensionScore objects (one per extension).

        Returns:
            Sorted list of extension_id strings, highest priority first.
        """
        return sorted(
            [s.extension_id for s in scores],
            key=lambda eid: self._composite(eid, scores),
            reverse=True,
        )

    def _composite(
        self,
        eid: str,
        scores: list[ExtensionScore],
    ) -> tuple[float, float, float]:
        """Build the sort key tuple for a given extension ID."""
        s = next(x for x in scores if x.extension_id == eid)
        return (s.narrative_impact, s.implementation_risk, -s.time_to_result)

    def ranking_rationale(self, scores: list[ExtensionScore]) -> dict[str, str]:
        """Produce a human-readable rationale string for each extension."""
        rationale: dict[str, str] = {}
        ranked = self.rank(scores)
        for position, eid in enumerate(ranked, start=1):
            s = next(x for x in scores if x.extension_id == eid)
            rationale[eid] = (
                f"Priority #{position}: "
                f"narrative_impact={s.narrative_impact:.1f}/3, "
                f"implementation_risk={s.implementation_risk:.1f}/3 "
                f"(3=low risk), "
                f"time_to_result={s.time_to_result:.1f}h"
            )
        return rationale
