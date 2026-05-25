"""Core infrastructure for V8 experiments."""

from scripts.experiments_v8.core.auto_registry import (
    RunSummary,
    StructuredLogger,
    register_experiment,
)
from scripts.experiments_v8.core.base_experiment import BaseExperiment
from scripts.experiments_v8.core.config import ExperimentConfig, SystemConfig
from scripts.experiments_v8.core.config import VQEConfig as V8VQEConfig
from scripts.experiments_v8.core.metrics import ComparisonResult, V8Metrics, WarmColdComparison
from scripts.experiments_v8.core.result_store import ResultStore

__all__ = [
    "BaseExperiment",
    "ExperimentConfig",
    "SystemConfig",
    "V8VQEConfig",
    "V8Metrics",
    "ComparisonResult",
    "WarmColdComparison",
    "ResultStore",
    "register_experiment",
    "StructuredLogger",
    "RunSummary",
]
