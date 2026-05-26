"""Analysis submodule — gradient analysis, diagnostics, and landscape."""

from qmbp_simulation.analysis.data_models import (
    BaselineComparison,
    BaselineMetrics,
    ComparisonResult,
    GradientAnalysisResult,
)
from qmbp_simulation.analysis.diagnostics import (
    DiagnosticCollector,
    configure_pipeline_logging,
)
from qmbp_simulation.analysis.gradient import WeightGradientAnalyzer
from qmbp_simulation.analysis.landscape import (
    compute_hessian,
    landscape_fluctuation,
)
from qmbp_simulation.analysis.metrics import (
    compute_classification_confidence,
    compute_energy_decomposition,
    compute_fraction_near_gs,
    compute_snr,
    compute_theta_smoothness,
)

__all__ = [
    "BaselineComparison",
    "BaselineMetrics",
    "ComparisonResult",
    "DiagnosticCollector",
    "GradientAnalysisResult",
    "WeightGradientAnalyzer",
    "compute_classification_confidence",
    "compute_energy_decomposition",
    "compute_fraction_near_gs",
    "compute_hessian",
    "compute_snr",
    "compute_theta_smoothness",
    "configure_pipeline_logging",
    "landscape_fluctuation",
]
