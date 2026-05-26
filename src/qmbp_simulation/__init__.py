"""Hybrid GNN-HVA Framework for Topological Phase Characterization.

This package provides a complete pipeline for quantum phase characterization
using Hamiltonian Variational Ansatz (HVA) circuits and Graph Neural Network
(GNN) parameter prediction.

Submodules:
    models      — Data models, Hamiltonians, lattice construction, constants
    solvers     — Classical exact diagonalization and DMRG
    circuits    — HVA circuit construction
    execution   — Quantum backend abstraction layer
    optimizers  — VQE and SPSA optimization
    predictors  — MPNN parameter prediction
    pipeline    — Orchestration, dataset I/O, QRC fallback
    framework   — Experiment engine, config, metrics
    analysis    — Gradient analysis, diagnostics, landscape
    utils       — Shared helpers (seeding, serialization, timing)
"""

__version__ = "1.0.0"

# --- models ---
from qmbp_simulation.models import (
    GroundTruthResult,
    HamiltonianBuilder,
    LatticeConfig,
    MAX_P_LAYERS,
    SUPPORTED_TOPOLOGIES,
    VQEConfig,
    VQEResult,
    make_lattice,
)

# --- solvers ---
from qmbp_simulation.solvers import ClassicalSolver

# --- circuits ---
from qmbp_simulation.circuits import HVACircuitBuilder

# --- execution ---
from qmbp_simulation.execution import (
    ExecutionBackend,
    HardwareBackend,
    NoiselessBackend,
    NoisyBackend,
)

# --- optimizers ---
from qmbp_simulation.optimizers import SPSAOptimizer, VQEOptimizer

# --- predictors ---
from qmbp_simulation.predictors import (
    MPNNPredictor,
    build_graph_dataset,
    train_mpnn,
)

# --- pipeline ---
from qmbp_simulation.pipeline import (
    PipelineRunner,
    load_phase12_dataset,
    save_phase12_dataset,
)

# --- framework ---
from qmbp_simulation.framework import (
    BaseExperiment,
    ExperimentConfig,
    ExperimentMetrics,
)

# --- analysis ---
from qmbp_simulation.analysis import (
    DiagnosticCollector,
    WeightGradientAnalyzer,
    compute_hessian,
)

__all__ = [
    # models
    "HamiltonianBuilder",
    "make_lattice",
    "LatticeConfig",
    "GroundTruthResult",
    "VQEConfig",
    "VQEResult",
    "SUPPORTED_TOPOLOGIES",
    "MAX_P_LAYERS",
    # solvers
    "ClassicalSolver",
    # circuits
    "HVACircuitBuilder",
    # execution
    "ExecutionBackend",
    "NoiselessBackend",
    "NoisyBackend",
    "HardwareBackend",
    # optimizers
    "VQEOptimizer",
    "SPSAOptimizer",
    # predictors
    "MPNNPredictor",
    "build_graph_dataset",
    "train_mpnn",
    # pipeline
    "save_phase12_dataset",
    "load_phase12_dataset",
    "PipelineRunner",
    # framework
    "BaseExperiment",
    "ExperimentConfig",
    "ExperimentMetrics",
    # analysis
    "WeightGradientAnalyzer",
    "DiagnosticCollector",
    "compute_hessian",
]
