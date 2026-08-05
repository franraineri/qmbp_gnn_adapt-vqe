"""External benchmarks for MPNN predictor evaluation.

Provides loaders and evaluators for external VQE datasets (VQEzy, etc.)
to validate cross-dataset generalization of our trained MPNN.
"""

from qmbp_simulation.predictors.external_benchmarks.vqezy_loader import (
    VQEzyInstance,
    VQEzyDataset,
    load_vqezy_tfi,
    load_vqezy_xyz,
    reconstruct_tfi_hamiltonian,
)
from qmbp_simulation.predictors.external_benchmarks.benchmark_evaluator import (
    BenchmarkResult,
    InstanceResult,
    VQEzyBenchmarkEvaluator,
)

__all__ = [
    "VQEzyInstance",
    "VQEzyDataset",
    "load_vqezy_tfi",
    "load_vqezy_xyz",
    "reconstruct_tfi_hamiltonian",
    "BenchmarkResult",
    "InstanceResult",
    "VQEzyBenchmarkEvaluator",
]
