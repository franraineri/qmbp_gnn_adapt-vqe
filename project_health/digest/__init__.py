"""Result digest package — lightweight, no heavy dependencies.

Public API for programmatic use:

    from project_health.digest import ResultScanner, NoiselessResult, NoisyResult, ExperimentResult

    scanner = ResultScanner(Path("results"))
    noiseless, noisy, experiments = scanner.scan_all()
"""

from project_health.digest.models import ExperimentResult, NoiselessResult, NoisyResult
from project_health.digest.scanner import ResultScanner

__all__ = [
    "ExperimentResult",
    "NoiselessResult",
    "NoisyResult",
    "ResultScanner",
]
