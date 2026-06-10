"""Result digest package — lightweight, no heavy dependencies.

Public API for programmatic use:

    from project_health.digest import ResultScanner, NoiselessResult, NoisyResult, ExperimentResult

    scanner = ResultScanner(Path("results"))
    noiseless, noisy, experiments = scanner.scan_all()
    scaling = scanner.scan_scaling()
    mode_comparison = scanner.scan_mode_comparison()
    n120_sweep = scanner.scan_n120_sweep()
"""

from project_health.digest.models import (
    CrossTopologyResult,
    ExperimentResult,
    ModeComparisonResult,
    N120SweepResult,
    NoiselessResult,
    NoisyResult,
    ScalingResult,
)
from project_health.digest.scanner import ResultScanner

__all__ = [
    "CrossTopologyResult",
    "ExperimentResult",
    "ModeComparisonResult",
    "N120SweepResult",
    "NoiselessResult",
    "NoisyResult",
    "ResultScanner",
    "ScalingResult",
]
