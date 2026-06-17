"""Extension priority ranking for thesis extension analysis.

Provides ExtensionScore dataclass and ExtensionPriorityRanker class for
ordering thesis extensions by composite score (narrative impact, risk, time).

Req: 5.8
"""

from __future__ import annotations

from dataclasses import dataclass

from qmbp_simulation.analysis.extension_models import ExtensionResult  # noqa: F401 (re-export)

# ---------------------------------------------------------------------------
# ExtensionScore
# ---------------------------------------------------------------------------


@dataclass
class ExtensionScore:
    """Composite score for a single extension candidate.

    Fields
    ------
    extension_id : str
        Identifier matching ExtensionResult.extension_id (e.g. "ext1").
    narrative_impact : float
        Contribution to thesis narrative: 3=HIGH, 2=MEDIUM, 1=LOW, 0=REJECTED.
    implementation_risk : float
        Inverted risk scale: 3=low_risk, 2=medium, 1=high (higher = better rank).
    time_to_result : float
        Estimated wall-clock hours to obtain a publishable result (lower = better).
    """

    extension_id: str
    narrative_impact: float
    implementation_risk: float
    time_to_result: float


# ---------------------------------------------------------------------------
# ExtensionPriorityRanker
# ---------------------------------------------------------------------------


class ExtensionPriorityRanker:
    """Rank extensions by composite score (Req 5.8).

    Sorting criteria (applied in order):
      Primary sort key   : narrative_impact  (higher = better)
      Tiebreak 1         : implementation_risk (higher = better, i.e. lower actual risk)
      Tiebreak 2         : -time_to_result  (lower hours = better)

    The resulting order is a valid total order: transitive and antisymmetric.
    Equal scores produce a stable sort (Python's ``sorted`` is stable).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(self, scores: list[ExtensionScore]) -> list[str]:
        """Return extension IDs sorted highest priority first.

        Parameters
        ----------
        scores:
            One ``ExtensionScore`` per candidate extension.

        Returns
        -------
        list[str]
            Extension IDs ordered from best to worst composite score.
        """
        return sorted(
            [s.extension_id for s in scores],
            key=lambda eid: self._composite_key(eid, scores),
            reverse=True,
        )

    def rank_with_rationale(self, scores: list[ExtensionScore]) -> tuple[list[str], dict[str, str]]:
        """Return ranked IDs together with a human-readable rationale per entry.

        Parameters
        ----------
        scores:
            One ``ExtensionScore`` per candidate extension.

        Returns
        -------
        ranked_ids : list[str]
            Extension IDs ordered from best to worst composite score.
        rationale : dict[str, str]
            Maps each extension_id to a one-line explanation of its rank position,
            suitable for inclusion in ``ExtensionAnalysisResult.ranking_rationale``.
        """
        ranked = self.rank(scores)
        rationale: dict[str, str] = {}
        for i, eid in enumerate(ranked):
            s = next(x for x in scores if x.extension_id == eid)
            rationale[eid] = (
                f"Rank {i + 1}: narrative_impact={s.narrative_impact:.1f}, "
                f"implementation_risk={s.implementation_risk:.1f}, "
                f"time_to_result={s.time_to_result:.1f}h"
            )
        return ranked, rationale

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _composite_key(self, eid: str, scores: list[ExtensionScore]) -> tuple[float, float, float]:
        """Compute the (impact, risk, -time) sort key for a given extension ID."""
        s = next(x for x in scores if x.extension_id == eid)
        return (s.narrative_impact, s.implementation_risk, -s.time_to_result)
