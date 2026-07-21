"""Execution submodule — quantum backend abstraction layer."""

from qmbp_simulation.execution.backends import (
    ExecutionBackend,
    MitigationOptions,
    NoiselessBackend,
    NoisyBackend,
    select_backend,
    select_backend_with_topology_warning,
)
from qmbp_simulation.execution.mps_backend import MPSBackend
from qmbp_simulation.execution.hardware import (
    HardwareBackend,
    HardwareConfig,
    HardwareRunResult,
    SPSAConfig,
)
from qmbp_simulation.execution.noisy_utils import (
    AdaptiveZNEResult,
    AffineCorrectedResult,
    BlockZNEResult,
    CalibrationSnapshot,
    DriftReport,
    GateFoldingDeploymentResult,
    GateFoldingZNEResult,
    LayoutSelection,
    NoisyEstimatorConfig,
    PEADeploymentResult,
    PEAResult,
    ZNEDeploymentResult,
    ZNEResult,
    affine_correct_energy,
    build_adjacency,
    check_calibration_drift,
    compute_circuit_ces,
    find_layouts_bfs,
    fold_gates,
    fold_single_layer,
    linear_zne,
    noisy_estimate,
    noisy_estimate_batch,
    run_adaptive_zne,
    run_block_zne,
    run_gate_folding_zne,
    run_gate_folding_zne_deployment,
    run_pea_zne,
    run_pea_zne_deployment,
    run_zne_deployment,
    select_layouts_by_circuit_ces,
    select_layouts_low_ces,
    take_calibration_snapshot,
)

__all__ = [
    # Backends
    "ExecutionBackend",
    "HardwareBackend",
    "HardwareConfig",
    "HardwareRunResult",
    "SPSAConfig",
    "MitigationOptions",
    "MPSBackend",
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
    # Adaptive ZNE (GF→PEA fallback)
    "run_adaptive_zne",
    "AdaptiveZNEResult",
    # Block-level ZNE (layer-wise folding)
    "fold_single_layer",
    "run_block_zne",
    "BlockZNEResult",
    # Dual-branch affine correction
    "affine_correct_energy",
    "AffineCorrectedResult",
    # TLS-aware calibration monitoring
    "take_calibration_snapshot",
    "check_calibration_drift",
    "CalibrationSnapshot",
    "DriftReport",
]

# ── Mitiq integration (optional — pip install 'qmbp-simulation[mitiq]') ──
try:
    from qmbp_simulation.execution.mitiq_utils import (
        MitiqCDRResult,
        MitiqComparisonResult,
        MitiqDDDZNEResult,
        MitiqPECResult,
        MitiqZNEResult,
        compare_mitigation_strategies,
        is_mitiq_available,
        make_mitiq_executor,
        make_noiseless_executor,
        run_mitiq_cdr,
        run_mitiq_ddd_zne,
        run_mitiq_pec,
        run_mitiq_zne,
    )

    __all__ += [
        # Mitiq integration
        "is_mitiq_available",
        "make_mitiq_executor",
        "make_noiseless_executor",
        "run_mitiq_zne",
        "run_mitiq_cdr",
        "run_mitiq_ddd_zne",
        "run_mitiq_pec",
        "compare_mitigation_strategies",
        "MitiqZNEResult",
        "MitiqCDRResult",
        "MitiqDDDZNEResult",
        "MitiqPECResult",
        "MitiqComparisonResult",
    ]
except ImportError:
    pass
