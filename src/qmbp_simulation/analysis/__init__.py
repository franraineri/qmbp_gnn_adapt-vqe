"""Analysis submodule — gradient analysis, diagnostics, landscape, and entanglement."""

from qmbp_simulation.analysis.comparative import (
    ComparativeMetrics,
    RegimeDiscoveryResult,
    classify_outcome,
    classify_result,
    compute_cx_budget,
    compute_staggered_magnetization,
    filter_by_threshold,
    find_h_min,
    find_minimum_viable_threshold,
    generate_comparison_table,
)
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
from qmbp_simulation.analysis.entanglement import (
    EntanglementAnalyzer,
    EntanglementResult,
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
    "ComparativeMetrics",
    "ComparisonResult",
    "DiagnosticCollector",
    "EntanglementAnalyzer",
    "EntanglementResult",
    "GradientAnalysisResult",
    "RegimeDiscoveryResult",
    "WeightGradientAnalyzer",
    "classify_outcome",
    "classify_result",
    "compute_classification_confidence",
    "compute_cx_budget",
    "compute_energy_decomposition",
    "compute_fraction_near_gs",
    "compute_hessian",
    "compute_snr",
    "compute_staggered_magnetization",
    "compute_theta_smoothness",
    "configure_pipeline_logging",
    "filter_by_threshold",
    "find_h_min",
    "find_minimum_viable_threshold",
    "generate_comparison_table",
    "landscape_fluctuation",
]
