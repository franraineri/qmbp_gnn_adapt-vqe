"""Project health checker — unified orchestration of Phase 4 analysis tools.

Produces a single report covering:
- Experiment verdicts (confirmed/rejected/failed)
- Coverage gaps (missing configs, regimes, seeds, ZNE)
- New results since last run
- Actionable items prioritized by impact

Usage:
    python -m project_health              # Full report (text)
    python -m project_health --json       # JSON output
    python -m project_health --compact    # Summary only
"""

from project_health.models import ActionItem, CoverageGap, HealthReport

__all__ = ["ActionItem", "CoverageGap", "HealthReport"]
