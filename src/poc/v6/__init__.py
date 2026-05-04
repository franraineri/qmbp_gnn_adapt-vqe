"""
GNN-HVA v6.0 — Hybrid GNN-HVA Framework for Topological Phase Characterization.

Modular architecture integrating:
  - Lattice-generalized Hamiltonian construction (Phase 1)
  - VQE with diagnostic callbacks and expanded bounds (Phase 2)
  - MPNN predictor via PyTorch Geometric (Phase 3)
  - Dual-route deployment: Adapt-VQE + QRC fallback (Phase 4)
"""

from .classical_solver import ClassicalSolver
from .config import (
    DeployResult,
    GroundTruthResult,
    LatticeConfig,
    OptimizationTrajectory,
    VQEConfig,
    VQEResult,
)
from .hamiltonian_builder import HamiltonianBuilder, make_lattice
from .hva_builder import HVACircuitBuilder
from .pipeline_utils import load_phase12_dataset, save_phase12_dataset
from .vqe_optimizer import VQEOptimizer

# HardwareDeployer imported lazily to avoid pulling in qiskit_algorithms
# at package import time (not needed for Phase 1-3 work).
# Use: from src.poc.v6.hardware_deployer import HardwareDeployer

__all__ = [
    # Config / data models
    "LatticeConfig",
    "GroundTruthResult",
    "VQEConfig",
    "VQEResult",
    "OptimizationTrajectory",
    "DeployResult",
    # Phase 1
    "HamiltonianBuilder",
    "make_lattice",
    "ClassicalSolver",
    # Phase 2
    "HVACircuitBuilder",
    "VQEOptimizer",
    # Utilities
    "save_phase12_dataset",
    "load_phase12_dataset",
]
