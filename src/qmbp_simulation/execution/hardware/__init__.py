"""Hardware execution backend for IBM Quantum processors.

Provides HardwareBackend (ExecutionBackend subclass) that executes HVA circuits
on IBM Torino via qiskit_ibm_runtime.EstimatorV2 with full error mitigation stack.
"""

from __future__ import annotations

from .backend import HardwareBackend
from .config import HardwareConfig, HardwareRunResult, SPSAConfig
from .preflight import (
    QPUCostEstimate,
    QPUThroughputProfile,
    SPSACostModel,
    estimate_effective_clops,
    estimate_qpu_cost,
)

__all__ = [
    "HardwareBackend",
    "HardwareConfig",
    "HardwareRunResult",
    "SPSAConfig",
    # Cost estimation
    "QPUCostEstimate",
    "QPUThroughputProfile",
    "SPSACostModel",
    "estimate_effective_clops",
    "estimate_qpu_cost",
]
