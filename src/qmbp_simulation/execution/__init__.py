"""Execution submodule — quantum backend abstraction layer."""

from qmbp_simulation.execution.backends import (
    ExecutionBackend,
    MitigationOptions,
    NoiselessBackend,
    NoisyBackend,
)
from qmbp_simulation.execution.hardware import (
    HardwareBackend,
    HardwareConfig,
    HardwareRunResult,
    SPSAConfig,
)
from qmbp_simulation.execution.noisy_utils import (
    GateFoldingDeploymentResult,
    GateFoldingZNEResult,
    LayoutSelection,
    NoisyEstimatorConfig,
    PEADeploymentResult,
    PEAResult,
    ZNEDeploymentResult,
    ZNEResult,
    build_adjacency,
    compute_circuit_ces,
    find_layouts_bfs,
    fold_gates,
    linear_zne,
    noisy_estimate,
    noisy_estimate_batch,
    run_gate_folding_zne,
    run_gate_folding_zne_deployment,
    run_pea_zne,
    run_pea_zne_deployment,
    run_zne_deployment,
    select_layouts_by_circuit_ces,
    select_layouts_low_ces,
)

__all__ = [
    # Backends
    "ExecutionBackend",
    "HardwareBackend",
    "HardwareConfig",
    "HardwareRunResult",
    "SPSAConfig",
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
    # Gate-folding ZNE
    "fold_gates",
    "run_gate_folding_zne",
    "run_gate_folding_zne_deployment",
    "GateFoldingZNEResult",
    "GateFoldingDeploymentResult",
    # PEA (Probabilistic Error Amplification)
    "run_pea_zne",
    "run_pea_zne_deployment",
    "PEAResult",
    "PEADeploymentResult",
]
