"""Execution submodule — quantum backend abstraction layer."""

from qmbp_simulation.execution.backends import (
    ExecutionBackend,
    HardwareBackend,
    MitigationOptions,
    NoiselessBackend,
    NoisyBackend,
)
from qmbp_simulation.execution.noisy_utils import (
    LayoutSelection,
    NoisyEstimatorConfig,
    ZNEDeploymentResult,
    ZNEResult,
    build_adjacency,
    compute_circuit_ces,
    find_layouts_bfs,
    linear_zne,
    noisy_estimate,
    noisy_estimate_batch,
    run_zne_deployment,
    select_layouts_by_circuit_ces,
    select_layouts_low_ces,
)

__all__ = [
    # Backends
    "ExecutionBackend",
    "HardwareBackend",
    "MitigationOptions",
    "NoiselessBackend",
    "NoisyBackend",
    # Noisy simulation utilities
    "NoisyEstimatorConfig",
    "LayoutSelection",
    "ZNEResult",
    "ZNEDeploymentResult",
    "build_adjacency",
    "find_layouts_bfs",
    "compute_circuit_ces",
    "select_layouts_by_circuit_ces",
    "select_layouts_low_ces",
    "noisy_estimate",
    "noisy_estimate_batch",
    "run_zne_deployment",
    "linear_zne",
]
