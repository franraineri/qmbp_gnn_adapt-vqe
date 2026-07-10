"""Framework submodule — experiment engine, config, metrics, CLI, and benchmarking."""

from qmbp_simulation.framework.base import BaseExperiment
from qmbp_simulation.framework.benchmarking import BenchmarkResult, BenchmarkSuite
from qmbp_simulation.framework.cli import (
    add_format_args,
    add_mpnn_args,
    add_noisy_args,
    add_output_args,
    add_result_filter_args,
    add_sweep_args,
    add_system_args,
    add_validation_args,
    add_variant_runner_args,
    add_vqe_args,
    build_mpnn_config_dict,
    configure_logging,
    create_base_parser,
    resolve_output_dir,
    validate_descending_sweep,
    validate_system_size,
)
from qmbp_simulation.framework.config import (
    AnalysisConfig,
    ExperimentConfig,
    MPNNConfig,
    SystemConfig,
    VQEConfig,
)
from qmbp_simulation.framework.criteria import (
    EXPERIMENT_CRITERIA,
    REJECTION_IS_FINDING,
    Verdict,
    compute_verdict,
)
from qmbp_simulation.framework.logging import ProgressReporter, StructuredLogger
from qmbp_simulation.framework.metrics import ExperimentMetrics, WarmColdComparison
from qmbp_simulation.framework.result_io import (
    build_experiment_id,
    build_result_envelope,
    generate_timestamp,
    load_result,
    load_results_from_dir,
    save_benchmark_result,
    save_experiment_result,
    save_pipeline_result,
)
from qmbp_simulation.framework.result_store import CATEGORY_MAP, ResultStore
from qmbp_simulation.framework.preflight import (
    ExperimentChecker,
    ExperimentSpec,
    P1_VALID_REGIME,
    P2_VALID_REGIME,
    PreflightChecker,
    PreflightReport,
    VariantSpec,
    get_regime_threshold,
    get_valid_regime,
    validate_regime_tables,
    specs_from_json,
    specs_from_pipeline_variants,
    specs_from_variant_runner,
)
from qmbp_simulation.framework.variant_runner import (
    PipelineVariant,
    RunResult,
    VariantRunner,
    run_variant_script,
)
from qmbp_simulation.framework.runner_base import (
    ExperimentRunner,
    HardwareValidationRunner,
    Section,
    SectionResult,
    ValidationRunner,
    VariantPipelineRunner,
)
from qmbp_simulation.framework.presets import (
    list_presets,
    load_preset,
    preset_to_args,
)
from qmbp_simulation.framework.artifact_store import (
    ArtifactCollector,
    find_artifacts_for_run,
    load_artifact,
    load_manifest,
)

__all__ = [
    # Base
    "BaseExperiment",
    # Config
    "ExperimentConfig",
    "SystemConfig",
    "VQEConfig",
    "MPNNConfig",
    "AnalysisConfig",
    # Metrics
    "ExperimentMetrics",
    "WarmColdComparison",
    # Logging
    "StructuredLogger",
    "ProgressReporter",
    # Result store
    "ResultStore",
    "CATEGORY_MAP",
    # Criteria
    "EXPERIMENT_CRITERIA",
    "REJECTION_IS_FINDING",
    "Verdict",
    "compute_verdict",
    # Result I/O
    "save_experiment_result",
    "save_pipeline_result",
    "save_benchmark_result",
    "build_experiment_id",
    "load_result",
    "load_results_from_dir",
    "build_result_envelope",
    "generate_timestamp",
    # CLI
    "create_base_parser",
    "add_system_args",
    "add_sweep_args",
    "add_vqe_args",
    "add_mpnn_args",
    "add_output_args",
    "add_noisy_args",
    "add_result_filter_args",
    "add_format_args",
    "add_validation_args",
    "add_variant_runner_args",
    "configure_logging",
    "validate_descending_sweep",
    "validate_system_size",
    "build_mpnn_config_dict",
    "resolve_output_dir",
    # Benchmarking
    "BenchmarkSuite",
    "BenchmarkResult",
    # Variant runner
    "PipelineVariant",
    "RunResult",
    "VariantRunner",
    "run_variant_script",
    # Runner bases
    "ExperimentRunner",
    "ValidationRunner",
    "VariantPipelineRunner",
    "HardwareValidationRunner",
    "Section",
    "SectionResult",
    # Presets
    "load_preset",
    "list_presets",
    "preset_to_args",
    # Preflight
    "PreflightChecker",
    "PreflightReport",
    "VariantSpec",
    "ExperimentChecker",
    "ExperimentSpec",
    "P1_VALID_REGIME",
    "P2_VALID_REGIME",
    "get_valid_regime",
    "get_regime_threshold",
    "validate_regime_tables",
    "specs_from_pipeline_variants",
    "specs_from_json",
    "specs_from_variant_runner",
    # Artifact store
    "ArtifactCollector",
    "load_artifact",
    "load_manifest",
    "find_artifacts_for_run",
]
