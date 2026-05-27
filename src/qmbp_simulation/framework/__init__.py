"""Framework submodule — experiment engine, config, metrics, CLI, and benchmarking."""

from qmbp_simulation.framework.base import BaseExperiment
from qmbp_simulation.framework.benchmarking import BenchmarkResult, BenchmarkSuite
from qmbp_simulation.framework.cli import (
    add_mpnn_args,
    add_output_args,
    add_sweep_args,
    add_system_args,
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
from qmbp_simulation.framework.logging import ProgressReporter, StructuredLogger
from qmbp_simulation.framework.metrics import ExperimentMetrics, WarmColdComparison
from qmbp_simulation.framework.result_io import (
    build_result_envelope,
    generate_timestamp,
    load_result,
    save_benchmark_result,
    save_experiment_result,
    save_pipeline_result,
)
from qmbp_simulation.framework.result_store import CATEGORY_MAP, ResultStore
from qmbp_simulation.framework.variant_runner import (
    PipelineVariant,
    RunResult,
    VariantRunner,
    run_variant_script,
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
    # Result I/O
    "save_experiment_result",
    "save_pipeline_result",
    "save_benchmark_result",
    "load_result",
    "build_result_envelope",
    "generate_timestamp",
    # CLI
    "create_base_parser",
    "add_system_args",
    "add_sweep_args",
    "add_vqe_args",
    "add_mpnn_args",
    "add_output_args",
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
]
