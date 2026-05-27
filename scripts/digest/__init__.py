"""Result digest package — lightweight, no heavy dependencies.

Public API for programmatic use:

    from scripts.digest import ResultScanner, NoiselessResult, NoisyResult, ExperimentResult

    scanner = ResultScanner(Path("results"))
    noiseless, noisy, experiments = scanner.scan_all()
"""

from scripts.digest.models import ExperimentResult, NoiselessResult, NoisyResult
from scripts.digest.scanner import ResultScanner

__all__ = [
    "ExperimentResult",
    "NoiselessResult",
    "NoisyResult",
    "ResultScanner",
]
