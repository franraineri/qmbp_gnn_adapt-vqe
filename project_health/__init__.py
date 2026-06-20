"""Project health checker — unified orchestration of Phase 4 analysis tools.

Produces a single report covering:
- Experiment verdicts (confirmed/rejected/failed)
- Coverage gaps (missing configs, regimes, seeds, ZNE)
- VQE convergence quality diagnostics
- MPNN training quality diagnostics
- Pipeline timing and distribution analytics
- Energy error decomposition (circuit vs MPNN)
- New results since last run
- Actionable items prioritized by impact

Usage:
    python -m project_health              # Full report (text)
    python -m project_health --json       # JSON output
    python -m project_health --markdown   # Markdown output
    python -m project_health --compact    # Summary only
    python -m project_health -o reports/  # Timestamped file in directory
"""

from project_health.core.models import (
    ActionItem,
    CoverageGap,
    EnergyDecompositionStats,
    HealthReport,
    ModelDistribution,
    MPNNQualityStats,
    TimingStats,
    VQEQualityStats,
)

__all__ = [
    "ActionItem",
    "CoverageGap",
    "EnergyDecompositionStats",
    "HealthReport",
    "ModelDistribution",
    "MPNNQualityStats",
    "TimingStats",
    "VQEQualityStats",
]
