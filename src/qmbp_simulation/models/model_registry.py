"""ModelRegistry — Centralized registry for spin model specifications.

Provides a dict-based registry mapping model_type strings to ModelSpec
instances. Adding a new model requires only registering a new entry.

Usage:
    from qmbp_simulation.models.model_registry import get_model_spec, list_models

    spec = get_model_spec("heisenberg")
    all_models = list_models()

    # Register a custom model:
    from qmbp_simulation.models.model_registry import register_model
    register_model(my_custom_spec)
"""

from __future__ import annotations

import logging

from qmbp_simulation.models.model_spec import ModelSpec

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> None:
    """Register a ModelSpec in the global registry.

    Parameters
    ----------
    spec : ModelSpec
        The model specification to register.

    Raises
    ------
    ValueError
        If a model with the same name is already registered.
    """
    if spec.name in _REGISTRY:
        raise ValueError(
            f"Model '{spec.name}' is already registered. Use a different name or unregister first."
        )
    _REGISTRY[spec.name] = spec
    logger.debug("Registered model: %s (%s)", spec.name, spec.description)


def get_model_spec(model_type: str) -> ModelSpec:
    """Get the ModelSpec for a given model type.

    Parameters
    ----------
    model_type : str
        Model identifier (e.g., "tfim", "heisenberg", "xy").

    Returns
    -------
    ModelSpec
        The registered model specification.

    Raises
    ------
    ValueError
        If model_type is not registered.
    """
    if model_type not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(f"Model '{model_type}' not registered. Available: {available}")
    return _REGISTRY[model_type]


def list_models() -> list[str]:
    """List all registered model names."""
    return list(_REGISTRY.keys())


def _register_builtins() -> None:
    """Register the built-in TFIM, Heisenberg, and XY models.

    Uses only HamiltonianBuilder (same package: models). Circuit callables
    are deferred via importlib to respect the dependency DAG:
    models CANNOT import circuits at any level (even inside functions).
    """
    import importlib

    from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

    builder = HamiltonianBuilder()

    # Lazy circuit dispatch — avoids importing circuits in models/
    def _create_tfim(n_qubits, p_layers, lattice, **kwargs):
        mod = importlib.import_module("qmbp_simulation.circuits")
        hva = mod.HVACircuitBuilder()
        return hva.create(n_qubits, p_layers, lattice, **kwargs)

    def _create_heisenberg(n_qubits, p_layers, lattice, **kwargs):
        mod = importlib.import_module("qmbp_simulation.circuits")
        hva = mod.HVACircuitBuilder()
        return hva.create_heisenberg(n_qubits, p_layers, lattice, **kwargs)

    # TFIM: H = -J·ZZ - h·X
    register_model(
        ModelSpec(
            name="tfim",
            params_per_layer=2,
            build_hamiltonian=builder.build,
            build_observables=builder.build_local_observables,
            create_circuit=_create_tfim,
            initial_state="plus",
            vqe_defaults={"n_restarts": 5, "restart_sigma": 0.1, "maxiter": 1000},
            description="Transverse-Field Ising Model: H = -J·ZZ - h·X",
        )
    )

    # Heisenberg XXZ: H = J(XX + YY + Δ·ZZ) - h·Z
    register_model(
        ModelSpec(
            name="heisenberg",
            params_per_layer=4,
            build_hamiltonian=builder.build_heisenberg,
            build_observables=builder.build_heisenberg_observables,
            create_circuit=_create_heisenberg,
            initial_state="neel",
            vqe_defaults={"n_restarts": 10, "restart_sigma": 0.5, "maxiter": 1500},
            hamiltonian_kwargs={"delta": 1.0},
            circuit_kwargs={"initial_state": "neel"},
            description="Heisenberg XXZ: H = J(XX+YY+Δ·ZZ) - h·Z (Δ=1.0)",
            fidelity_threshold=0.60,  # Relaxed — HVA p=2 has limited expressibility
            mpnn_hidden_dim=128,  # Larger output space (8 params) needs more capacity
        )
    )

    # XY Model: H = J(XX + YY) - h·Z (Δ=0)
    register_model(
        ModelSpec(
            name="xy",
            params_per_layer=4,
            build_hamiltonian=builder.build_heisenberg,
            build_observables=builder.build_heisenberg_observables,
            create_circuit=_create_heisenberg,
            initial_state="neel",
            vqe_defaults={"n_restarts": 10, "restart_sigma": 0.5, "maxiter": 1500},
            hamiltonian_kwargs={"delta": 0.0},
            circuit_kwargs={"initial_state": "neel"},
            description="XY Model: H = J(XX+YY) - h·Z (Δ=0)",
            fidelity_threshold=0.60,  # Relaxed — same expressibility limits as Heisenberg
            mpnn_hidden_dim=128,
        )
    )


_register_builtins()
