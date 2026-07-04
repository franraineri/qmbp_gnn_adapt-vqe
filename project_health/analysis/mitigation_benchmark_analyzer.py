"""Backward-compatible re-export.

The module was moved to project_health.analysis.hardware.mitigation_benchmark_analyzer
during the reorganization. This shim keeps old imports working.
"""

from project_health.analysis.hardware.mitigation_benchmark_analyzer import *  # noqa: F401,F403
from project_health.analysis.hardware.mitigation_benchmark_analyzer import (  # noqa: F401
    HYPOTHESIS_MAP,
    MitigationBenchmarkAnalyzer,
)
