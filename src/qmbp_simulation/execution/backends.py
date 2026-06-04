"""Execution backends for quantum circuit evaluation.

Provides an abstract ExecutionBackend interface and concrete implementations
for noiseless simulation, noisy simulation, and hardware execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


@dataclass
class MitigationOptions:
    """Error mitigation configuration for noisy/hardware backends."""

    zne_enabled: bool = False
    zne_noise_factors: list[float] | None = None  # e.g. [1, 3, 5]
    zne_amplifier: str = "gate_folding"  # "gate_folding" | "pea"
    dd_enabled: bool = False  # Dynamical decoupling
    trex_enabled: bool = False  # Twirled readout error extinction
    twirling_enabled: bool = False
    num_randomizations: int = 32
    shots_per_randomization: int = 128


class ExecutionBackend(ABC):
    """Abstract base class for quantum circuit evaluation.

    All optimizers accept an ExecutionBackend instance, enabling the same
    optimization code to run against noiseless simulation, noisy simulation,
    or real hardware without modification.
    """

    @abstractmethod
    def evaluate(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        params: np.ndarray,
    ) -> float:
        """Evaluate ⟨H⟩ for the given circuit parameters.

        Parameters
        ----------
        circuit : QuantumCircuit
            Parameterized circuit (not yet bound).
        hamiltonian : SparsePauliOp
            Observable to measure.
        params : np.ndarray
            Parameter values to bind.

        Returns
        -------
        float
            Expectation value ⟨ψ(params)|H|ψ(params)⟩.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier."""
        ...


class NoiselessBackend(ExecutionBackend):
    """Exact statevector simulation via StatevectorEstimator.

    This is the default backend for all noiseless experiments.
    """

    def __init__(self) -> None:
        from qiskit.primitives import StatevectorEstimator

        self._estimator = StatevectorEstimator()

    def evaluate(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        params: np.ndarray,
    ) -> float:
        """Evaluate expectation value using exact statevector simulation."""
        if len(params) != circuit.num_parameters:
            raise ValueError(
                f"Parameter count mismatch: got {len(params)}, expected {circuit.num_parameters}."
            )
        bound = circuit.assign_parameters(params)
        job = self._estimator.run([(bound, hamiltonian)])
        energy = float(job.result()[0].data.evs)
        if not np.isfinite(energy):
            raise RuntimeError(
                f"Non-finite energy returned from StatevectorEstimator: {energy}. "
                f"Check circuit parameters for NaN/Inf."
            )
        return energy

    @property
    def name(self) -> str:
        return "noiseless_statevector"


class NoisyBackend(ExecutionBackend):
    """Shot-noise simulation with configurable noise model and mitigation.

    Two evaluation modes:
    - If noise_model is None: Gaussian shot noise approximation
      (exact energy + N(0, 1/√shots)).
    - If noise_model is provided: Full simulation via AerSimulator
      (requires qiskit-aer installed).
    """

    def __init__(
        self,
        shots: int = 8192,
        noise_model=None,
        mitigation: MitigationOptions | None = None,
        seed_simulator: int | None = None,
    ) -> None:
        self._shots = shots
        self._noise_model = noise_model
        self._mitigation = mitigation or MitigationOptions()
        self._seed_simulator = seed_simulator
        self._noiseless = NoiselessBackend()
        # Persistent RNG for Gaussian shot noise approximation — advances
        # on each evaluate() call to produce realistic stochastic noise.
        self._rng = np.random.default_rng(seed_simulator)

    def evaluate(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        params: np.ndarray,
    ) -> float:
        """Evaluate expectation value with shot noise or full noise model."""
        if len(params) != circuit.num_parameters:
            raise ValueError(
                f"Parameter count mismatch: got {len(params)}, expected {circuit.num_parameters}."
            )
        if self._noise_model is None:
            # Gaussian shot noise approximation — RNG advances each call
            exact_energy = self._noiseless.evaluate(circuit, hamiltonian, params)
            noise = self._rng.normal(0.0, 1.0 / np.sqrt(self._shots))
            return exact_energy + noise

        # Full noise model simulation via AerSimulator
        try:
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "qiskit-aer is required for noise model simulation. "
                "Install with: pip install qiskit-aer"
            ) from e

        backend = AerSimulator(noise_model=self._noise_model)
        from qiskit.primitives import BackendEstimatorV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        bound = circuit.assign_parameters(params)
        isa_circuit = pm.run(bound)

        precision = 1.0 / np.sqrt(self._shots)
        options = {"default_precision": precision}
        if self._seed_simulator is not None:
            options["seed_simulator"] = self._seed_simulator

        estimator = BackendEstimatorV2(backend=backend, options=options)
        job = estimator.run([(isa_circuit, hamiltonian)])
        energy = float(job.result()[0].data.evs)
        if not np.isfinite(energy):
            raise RuntimeError(
                f"Non-finite energy returned from noisy backend: {energy}. "
                f"Check circuit depth and noise model compatibility."
            )
        return energy

    @property
    def name(self) -> str:
        return f"noisy_shots={self._shots}"


class HardwareBackend(ExecutionBackend):
    """IBM Runtime hardware backend (stub — pending IBM Quantum integration).

    Raises NotImplementedError until IBM Runtime credentials and
    session management are configured.
    """

    def __init__(
        self,
        backend_name: str = "ibm_torino",
        mitigation: MitigationOptions | None = None,
    ) -> None:
        self._backend_name = backend_name
        self._mitigation = mitigation or MitigationOptions()

    def evaluate(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        params: np.ndarray,
    ) -> float:
        """Raise NotImplementedError — hardware integration pending."""
        raise NotImplementedError(
            f"HardwareBackend('{self._backend_name}') is not yet implemented. "
            "IBM Runtime integration is pending. Use NoiselessBackend or "
            "NoisyBackend for local development."
        )

    @property
    def name(self) -> str:
        return f"hardware_{self._backend_name}"
