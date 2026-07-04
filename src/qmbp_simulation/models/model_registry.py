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
    logger.debug(
        "get_model_spec: model=%s, params_per_layer=%d",
        model_type,
        _REGISTRY[model_type].params_per_layer,
    )
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
            initial_state="plus",
            vqe_defaults={"n_restarts": 10, "restart_sigma": 0.5, "maxiter": 1500},
            hamiltonian_kwargs={"delta": 1.0},
            circuit_kwargs={"initial_state": "plus"},
            description="Heisenberg XXZ: H = J(XX+YY+Δ·ZZ) - h·Z (Δ=1.0)",
            fidelity_threshold=0.60,  # noqa — Relaxed: HVA p=2 has limited expressibility for non-TFIM
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
            initial_state="plus",
            vqe_defaults={"n_restarts": 10, "restart_sigma": 0.5, "maxiter": 1500},
            hamiltonian_kwargs={"delta": 0.0},
            circuit_kwargs={"initial_state": "plus"},
            description="XY Model: H = J(XX+YY) - h·Z (Δ=0)",
            fidelity_threshold=0.60,  # noqa — Relaxed: same expressibility limits as Heisenberg
            mpnn_hidden_dim=128,
        )
    )

    # Heisenberg XXZ with TRANSVERSE field: H = J(XX+YY+Δ·ZZ) - h·X
    # Key difference from "heisenberg": field in X direction (transverse) instead of Z.
    # This creates a QPT (AF→paramagnet) accessible to HVA with |+⟩^N initial state.
    # Δ=0.5 reduces frustration vs isotropic (Δ=1), making shallow circuits viable.
    register_model(
        ModelSpec(
            name="heisenberg_transverse",
            params_per_layer=4,
            build_hamiltonian=builder.build_heisenberg_transverse,
            build_observables=builder.build_heisenberg_observables,
            create_circuit=_create_heisenberg,
            initial_state="plus",
            vqe_defaults={"n_restarts": 10, "restart_sigma": 0.3, "maxiter": 1500},
            hamiltonian_kwargs={"delta": 0.5},
            circuit_kwargs={"initial_state": "plus"},
            description="Heisenberg XXZ transverse: H = J(XX+YY+0.5·ZZ) - h·X",
            fidelity_threshold=0.60,
            mpnn_hidden_dim=128,
        )
    )

    # TFIM + Longitudinal: H = -J·ZZ - h·X - g·Z
    def _create_tfim_longitudinal(n_qubits, p_layers, lattice, **kwargs):
        mod = importlib.import_module("qmbp_simulation.circuits")
        hva = mod.HVACircuitBuilder()
        return hva.create_tfim_longitudinal(n_qubits, p_layers, lattice, **kwargs)

    register_model(
        ModelSpec(
            name="tfim_longitudinal",
            params_per_layer=3,
            build_hamiltonian=builder.build_tfim_longitudinal,
            build_observables=builder.build_local_observables,
            create_circuit=_create_tfim_longitudinal,
            initial_state="plus",
            vqe_defaults={"n_restarts": 5, "restart_sigma": 0.1, "maxiter": 1000},
            hamiltonian_kwargs={"g": 0.0},
            description=(
                "TFIM + Longitudinal Field: H = -J·ZZ - h·X - g·Z. "
                "Extends standard TFIM with Z₂-symmetry-breaking longitudinal field."
            ),
            fidelity_threshold=0.90,
            mpnn_hidden_dim=64,
        )
    )

    # Frustrated TFIM (J1-J2): H = -J₁·ZZ_nn + J₂·ZZ_nnn - h·X
    def _create_frustrated_tfim(n_qubits, p_layers, lattice, **kwargs):
        mod = importlib.import_module("qmbp_simulation.circuits")
        hva = mod.HVACircuitBuilder()
        return hva.create_frustrated_tfim(n_qubits, p_layers, lattice, **kwargs)

    register_model(
        ModelSpec(
            name="tfim_frustrated",
            params_per_layer=3,
            build_hamiltonian=builder.build_frustrated_tfim,
            build_observables=builder.build_local_observables,
            create_circuit=_create_frustrated_tfim,
            initial_state="plus",
            vqe_defaults={"n_restarts": 5, "restart_sigma": 0.1, "maxiter": 1000},
            hamiltonian_kwargs={"J2": 0.0},
            description=(
                "Frustrated TFIM (J1-J2): H = -J₁·ZZ_nn + J₂·ZZ_nnn - h·X. "
                "NNN antiferromagnetic coupling introduces frustration. "
                "Hardware-viable only at N=4 (27 CZ at N=6 exceeds ZNE budget)."
            ),
            fidelity_threshold=0.90,
            mpnn_hidden_dim=64,
        )
    )

    # Bond-Resolved TFIM: H = -J·ZZ - h·X (same Hamiltonian, local parameters)
    def _create_bond_resolved(n_qubits, p_layers, lattice, **kwargs):
        mod = importlib.import_module("qmbp_simulation.circuits")
        hva = mod.HVACircuitBuilder()
        return hva.create_bond_resolved(n_qubits, p_layers, lattice, **kwargs)

    register_model(
        ModelSpec(
            name="tfim_bond_resolved",
            params_per_layer=-1,  # Variable: n_edges + n_qubits (topology-dependent)
            build_hamiltonian=builder.build,
            build_observables=builder.build_local_observables,
            create_circuit=_create_bond_resolved,
            initial_state="plus",
            vqe_defaults={
                "n_restarts": 5,
                "restart_sigma": 0.05,
                "maxiter": 1500,
            },
            description=(
                "Bond-Resolved TFIM: Same H = -J·ZZ - h·X but with per-bond θ_zz_k "
                "and per-site θ_x_i parameters. Increases expressibility without "
                "increasing depth or gate count. Toward quantum advantage via "
                "high-dimensional parameter space (Fusco et al., 2026)."
            ),
            fidelity_threshold=0.93,
            mpnn_hidden_dim=128,
        )
    )


_register_builtins()
