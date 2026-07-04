"""Analysis submodule — gradient analysis, diagnostics, landscape, and entanglement.

Also includes thesis extension analysis modules:
  extension_models, extension_classifiers, extension_ranker,
  extension_analyzer, normalizing_flow
"""

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
from qmbp_simulation.analysis.theta_alignment import (
    AlignmentReport,
    EnergyGuardReport,
    OutlierReport,
    align_theta_array,
    align_theta_sweep,
    cross_h_energy_guard,
    detect_jumps,
    detect_theta_outliers,
    filter_theta_outliers,
)
from qmbp_simulation.analysis.theta_validator import (
    ThetaValidationReport,
    ThetaValidator,
)
from qmbp_simulation.analysis.vqe_validator import (
    VQEValidationReport,
    VQEValidator,
    ValidationIssue,
    Severity,
)
from qmbp_simulation.analysis.ground_truth_validator import (
    GroundTruthValidationReport,
    GroundTruthValidator,
)
from qmbp_simulation.analysis.cross_n_validator import (
    CrossNValidationReport,
    CrossNValidator,
    preflight_cross_n,
)
from qmbp_simulation.analysis.nlce import (
    ClusterResult,
    ClusterSolver,
    NLCEConfig,
    NLCEResult,
    NLCERunner,
    VQEClusterSolver,
    nlce_convergence_analysis,
    tfim_analytical_energy_per_site,
)
from qmbp_simulation.analysis.circuit_visualizer import (
    circuit_summary,
    compute_circuit_feasibility,
    compute_decoherence_penalty,
    compute_error_budget,
    compute_parallelism_efficiency,
    compute_shot_noise_floor,
    build_error_prediction,
    validate_prediction_vs_result,
    print_circuit,
    print_circuit_comparison,
    rank_layouts_by_depth_2q,
    save_circuit_diagram,
    select_best_layout_for_zne,
    transpiled_circuit_stats,
)

__all__ = [
    "AlignmentReport",
    "BaselineComparison",
    "BaselineMetrics",
    "ClusterResult",
    "ClusterSolver",
    "ComparativeMetrics",
    "ComparisonResult",
    "DiagnosticCollector",
    "EntanglementAnalyzer",
    "EntanglementResult",
    "GradientAnalysisResult",
    "GroundTruthValidationReport",
    "GroundTruthValidator",
    "NLCEConfig",
    "NLCEResult",
    "NLCERunner",
    "RegimeDiscoveryResult",
    "Severity",
    "ThetaValidationReport",
    "ThetaValidator",
    "VQEClusterSolver",
    "VQEValidationReport",
    "VQEValidator",
    "ValidationIssue",
    "WeightGradientAnalyzer",
    "align_theta_array",
    "align_theta_sweep",
    "circuit_summary",
    "compute_circuit_feasibility",
    "compute_decoherence_penalty",
    "compute_error_budget",
    "compute_parallelism_efficiency",
    "compute_shot_noise_floor",
    "build_error_prediction",
    "validate_prediction_vs_result",
    "rank_layouts_by_depth_2q",
    "select_best_layout_for_zne",
    "transpiled_circuit_stats",
    "classify_outcome",
    "classify_result",
    "compute_classification_confidence",
    "compute_cx_budget",
    "detect_jumps",
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
    "nlce_convergence_analysis",
    "print_circuit",
    "print_circuit_comparison",
    "save_circuit_diagram",
    "tfim_analytical_energy_per_site",
]
