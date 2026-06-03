"""ModelSpec — Strategy pattern for model-agnostic pipeline execution.

Encapsulates all model-specific behavior (Hamiltonian construction, circuit
creation, observables, VQE defaults) in a frozen dataclass. The pipeline
dispatches to the appropriate methods via the ModelSpec without if/elif chains.

Usage:
    from qmbp_simulation.models import get_model_spec

    spec = get_model_spec("heisenberg")
    H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
    qc, theta = spec.create_circuit(n, p, lattice, **spec.circuit_kwargs)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """Specification of a spin model for the GNN-HVA pipeline.

    Encapsulates model-specific behavior following the Strategy pattern,
    enabling extensibility without modifying pipeline code.

    Parameters
    ----------
    name : str
        Model identifier (e.g., "tfim", "heisenberg", "xy").
    params_per_layer : int
        Number of variational parameters per HVA layer.
    build_hamiltonian : Callable
        Function to construct the Hamiltonian SparsePauliOp.
        Signature: (lattice, **hamiltonian_kwargs) -> SparsePauliOp
    build_observables : Callable
        Function to construct measurement observables.
        Signature: (lattice) -> tuple[list, list]
    create_circuit : Callable
        Function to construct the parameterized HVA circuit.
        Signature: (n_qubits, p_layers, lattice, **circuit_kwargs) -> (QC, PV)
    initial_state : str
        Default initial state for the HVA circuit ("plus", "neel", "zero").
    vqe_defaults : dict
        Model-specific VQE configuration overrides.
    circuit_kwargs : dict
        Additional kwargs passed to create_circuit.
    hamiltonian_kwargs : dict
        Additional kwargs passed to build_hamiltonian.
    description : str
        Human-readable description of the model.
    """

    name: str
    params_per_layer: int
    build_hamiltonian: Callable[..., Any]
    build_observables: Callable[..., Any]
    create_circuit: Callable[..., Any]
    initial_state: str = "plus"
    vqe_defaults: dict[str, Any] = field(default_factory=dict)
    circuit_kwargs: dict[str, Any] = field(default_factory=dict)
    hamiltonian_kwargs: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    fidelity_threshold: float = 0.93  # Model-specific fidelity floor for Phase 3
    mpnn_hidden_dim: int = 64  # Recommended MPNN hidden dim for this model

    @property
    def total_params(self) -> int:
        """Total variational parameters for p=2 (maximum allowed)."""
        return self.params_per_layer * 2

    def get_vqe_config_overrides(self) -> dict[str, Any]:
        """Return model-specific VQE configuration overrides.

        These override the default VQEConfig values when the model
        requires different optimization settings (e.g., more restarts
        for higher-dimensional landscapes).
        """
        return self.vqe_defaults.copy()

    def total_params_for_p(self, p_layers: int) -> int:
        """Total variational parameters for a given number of layers."""
        return self.params_per_layer * p_layers

    def with_delta(self, delta: float) -> ModelSpec:
        """Return a new ModelSpec with overridden delta in hamiltonian_kwargs.

        Convenience method for varying anisotropy without manual replace().

        Parameters
        ----------
        delta : float
            New anisotropy parameter value.

        Returns
        -------
        ModelSpec
            New frozen instance with updated hamiltonian_kwargs.
        """
        from dataclasses import replace

        new_kwargs = {**self.hamiltonian_kwargs, "delta": delta}
        return replace(self, hamiltonian_kwargs=new_kwargs)

    def with_g(self, g: float) -> ModelSpec:
        """Return a new ModelSpec with overridden g in hamiltonian_kwargs.

        Convenience method for varying the longitudinal field strength
        in the TFIM + longitudinal model.

        Parameters
        ----------
        g : float
            New longitudinal field strength.

        Returns
        -------
        ModelSpec
            New frozen instance with updated hamiltonian_kwargs.
        """
        from dataclasses import replace

        new_kwargs = {**self.hamiltonian_kwargs, "g": g}
        return replace(self, hamiltonian_kwargs=new_kwargs)

    def with_params(self, **kwargs: Any) -> ModelSpec:
        """Return a new ModelSpec with arbitrary hamiltonian_kwargs overrides.

        Generic method for varying any Hamiltonian parameter. Preferred over
        model-specific methods (with_delta, with_g) when writing code that
        must work across multiple models.

        Parameters
        ----------
        **kwargs
            Key-value pairs to merge into hamiltonian_kwargs.

        Returns
        -------
        ModelSpec
            New frozen instance with updated hamiltonian_kwargs.

        Examples
        --------
        >>> spec = get_model_spec("heisenberg").with_params(delta=0.5)
        >>> spec = get_model_spec("tfim_longitudinal").with_params(g=0.3)
        >>> spec = get_model_spec("kitaev").with_params(mu=1.0, delta=0.5)
        """
        from dataclasses import replace

        new_kwargs = {**self.hamiltonian_kwargs, **kwargs}
        return replace(self, hamiltonian_kwargs=new_kwargs)
