"""Models submodule — data models, Hamiltonians, constants, and model registry."""

from qmbp_simulation.models.constants import (
    DMRG_QUBIT_LIMIT,
    EXACT_DIAG_QUBIT_LIMIT,
    MAX_P_LAYERS,
    SUPPORTED_TOPOLOGIES,
)
from qmbp_simulation.models.data_models import (
    DeployResult,
    GroundTruthResult,
    LatticeConfig,
    OptimizationTrajectory,
    VQEConfig,
    VQEResult,
)
from qmbp_simulation.models.hamiltonian import HamiltonianBuilder, make_lattice
from qmbp_simulation.models.model_registry import (
    get_model_spec,
    list_models,
    register_model,
)
from qmbp_simulation.models.model_spec import ModelSpec

__all__ = [
    "DMRG_QUBIT_LIMIT",
    "EXACT_DIAG_QUBIT_LIMIT",
    "MAX_P_LAYERS",
    "ModelSpec",
    "SUPPORTED_TOPOLOGIES",
    "DeployResult",
    "GroundTruthResult",
    "HamiltonianBuilder",
    "LatticeConfig",
    "OptimizationTrajectory",
    "VQEConfig",
    "VQEResult",
    "get_model_spec",
    "list_models",
    "make_lattice",
    "register_model",
]
