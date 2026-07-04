"""Execution backends for quantum circuit evaluation.

Provides an abstract ExecutionBackend interface and concrete implementations
for noiseless simulation, noisy simulation, and hardware execution.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

logger = logging.getLogger(__name__)


@dataclass
class MitigationOptions:
    """Error mitigation configuration for noisy/hardware backends.

    PEA learning budget (2026-06-14 update):
    - num_randomizations × shots_per_randomization = total learning shots
    - Increased from 32×128=4K to 64×256=16K for better noise model accuracy
      on processors with elevated 2Q error (>2%). Adds ~1 min QPU/h-point
      but dramatically improves ZNE extrapolation quality.

    noise_factors (2026-06-14 update):
    - For short circuits (≤18 CZ), use (1, 1.5, 2, 3) to capture the linear
      regime better. Default IBM (1, 2, 3) can miss curvature at low factors.
    - For longer circuits, stick with default.

    layer_pair_depths (2026-06-17 update):
    - Controls the identity-pair insertion depths used by IBM Runtime to learn
      the per-layer noise model for PEA. IBM tutorial uses [0,1,2,4,6,12,24]
      for deep Trotter circuits (18 layers). For shallow circuits (HVA p=1,
      1 layer of 2Q gates), fewer depths suffice: [0, 1, 2, 4, 8].
    - None → let Runtime use its default (recommended for most cases).
    - Explicit list → fine-grained control for calibration studies.
    - Ref: IBM PEA tutorial (2026), Kim et al. Nature 618 (2023).

    twirling_strategy (2026-06-17 update):
    - "active-circuit": twirl only gates in the active circuit (IBM default
      for utility-scale). Avoids inserting Pauli twirls on idle qubits that
      could add unnecessary noise. Recommended for dense circuits.
    - None → let Runtime choose (defaults to "active-circuit" on Heron r2+).
    """

    zne_enabled: bool = False
    zne_noise_factors: list[float] | None = None  # e.g. [1, 1.5, 2, 3]
    zne_amplifier: str = "gate_folding"  # "gate_folding" | "pea" | "adaptive"
    zne_r2_fallback_threshold: float = 0.90  # R² threshold for adaptive GF→PEA fallback
    dd_enabled: bool = False  # Dynamical decoupling
    dd_sequence: str = "XpXm"  # "XX" | "XpXm" | "XY4"
    trex_enabled: bool = False  # Twirled readout error extinction
    twirling_enabled: bool = False
    # PEA noise learning budget: higher = better noise model, more QPU cost
    # 64 randomizations × 256 shots = 16K learning shots (~4× IBM default)
    num_randomizations: int = 64
    shots_per_randomization: int = 256
    # PEA layer noise learning: identity-pair depths for exponential decay fit.
    # None = Runtime default. For HVA p=1 (1 layer): [0, 1, 2, 4, 8] is sufficient.
    # For deep circuits (Trotter 6+ steps): [0, 1, 2, 4, 6, 12, 24] per IBM tutorial.
    layer_pair_depths: list[int] | None = None
    # Twirling strategy: "active-circuit" avoids twirling idle qubits.
    # None = let Runtime decide (Heron r2+ defaults to "active-circuit").
    twirling_strategy: str | None = None
    # ── QESEM integration (Qedma Qiskit Function, arXiv:2508.10997) ──────
    # When enabled, bypasses local ZNE pipeline and delegates mitigation
    # entirely to Qedma's QESEM function (characterization-based, unbiased,
    # quasi-probabilistic mitigation). Requires IBM Premium/Flex plan access
    # and qiskit-ibm-catalog package.
    qesem_enabled: bool = False

    def __post_init__(self) -> None:
        if self.dd_enabled and self.dd_sequence not in ("XX", "XpXm", "XY4"):
            raise ValueError(
                f"Invalid dd_sequence '{self.dd_sequence}'. Valid values: 'XX', 'XpXm', 'XY4'"
            )
        if self.zne_noise_factors is not None:
            if len(self.zne_noise_factors) < 2:
                raise ValueError(
                    f"zne_noise_factors must have at least 2 elements for extrapolation, "
                    f"got {len(self.zne_noise_factors)}."
                )
            if self.zne_noise_factors != sorted(self.zne_noise_factors):
                raise ValueError(
                    f"zne_noise_factors must be in ascending order, got {self.zne_noise_factors}."
                )
            if self.zne_noise_factors[0] < 1.0:
                raise ValueError(
                    f"zne_noise_factors[0] must be >= 1.0 (base noise level), "
                    f"got {self.zne_noise_factors[0]}."
                )


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

    def compute_fidelity(
        self,
        circuit: QuantumCircuit,
        params: np.ndarray,
        exact_state: np.ndarray,
    ) -> float:
        """Compute state fidelity |⟨ψ_exact|ψ(params)⟩|²."""
        logger.debug(
            "[%s] compute_fidelity: N=%d, n_params=%d",
            self.name,
            circuit.num_qubits,
            len(params),
        )
        from qiskit.quantum_info import Statevector, state_fidelity

        sv_ansatz = Statevector(circuit.assign_parameters(params))
        fid = float(state_fidelity(sv_ansatz, Statevector(exact_state)))

        # Physics guard: fidelity must be in [0, 1]. Numerical noise can push
        # it slightly outside due to floating-point arithmetic in inner products.
        if fid > 1.0 + 1e-6 or fid < -1e-6:
            logger.error(
                "[%s] compute_fidelity: value %.8f far outside [0, 1] — "
                "possible bug in circuit or exact_state (not a valid quantum state).",
                self.name,
                fid,
            )
        elif fid > 1.0 + 1e-10 or fid < -1e-10:
            logger.warning(
                "[%s] compute_fidelity: value %.8f slightly outside [0, 1] — "
                "numerical noise, clipping.",
                self.name,
                fid,
            )
        fid = float(np.clip(fid, 0.0, 1.0))

        logger.debug("[%s] compute_fidelity: result=%.6f", self.name, fid)
        return fid

    def get_statevector(
        self,
        circuit: QuantumCircuit,
        params: np.ndarray,
    ) -> np.ndarray:
        """Extract the full statevector for the given circuit and parameters."""
        logger.debug(
            "[%s] get_statevector: N=%d, n_params=%d",
            self.name,
            circuit.num_qubits,
            len(params),
        )
        if len(params) != circuit.num_parameters:
            raise ValueError(
                f"Parameter count mismatch in get_statevector: "
                f"got {len(params)}, circuit expects {circuit.num_parameters}."
            )
        from qiskit.quantum_info import Statevector

        sv = Statevector(circuit.assign_parameters(params))
        return np.asarray(sv.data)


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
        logger.debug(
            "NoiselessBackend.evaluate: n_qubits=%d, n_params=%d, H_terms=%d",
            circuit.num_qubits,
            len(params),
            len(hamiltonian),
        )
        if len(params) != circuit.num_parameters:
            raise ValueError(
                f"Parameter count mismatch: got {len(params)}, expected {circuit.num_parameters}."
            )
        if not np.all(np.isfinite(params)):
            raise ValueError(
                f"NoiselessBackend.evaluate: params contain NaN/Inf. "
                f"Non-finite indices: {np.where(~np.isfinite(params))[0].tolist()}"
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
    """Shot-noise simulation — RAW noise only, no mitigation applied.

    Two evaluation modes:
    - If noise_model is None: Gaussian shot noise approximation
      (exact energy + N(0, 1/√shots)).
    - If noise_model is provided: Full simulation via AerSimulator
      (requires qiskit-aer installed).

    This backend is intended for:
    - Generating "noisy raw" baselines (factor=1, no mitigation).
    - VQE training with shot noise approximation.
    - Quick noise-level estimation without full mitigation stack.

    For mitigated noisy simulation, use the utility functions directly:
    - Gate-folding ZNE: ``run_gate_folding_zne()`` from ``noisy_utils``.
    - PEA-ZNE: ``run_pea_zne()`` from ``noisy_utils``.
    - CES-ZNE: ``run_zne_deployment()`` from ``noisy_utils``.

    .. deprecated::
        The ``mitigation`` parameter is accepted for backward compatibility
        but is NOT applied. Pass ``MitigationOptions(zne_enabled=True)``
        and you will receive a ``DeprecationWarning``. Use the utility
        functions above for mitigated estimation.
    """

    def __init__(
        self,
        shots: int = 8192,
        noise_model=None,
        mitigation: MitigationOptions | None = None,  # DEPRECATED — ignored
        seed_simulator: int | None = None,
    ) -> None:
        self._shots = shots
        self._noise_model = noise_model
        self._seed_simulator = seed_simulator
        self._noiseless = NoiselessBackend()
        # Persistent RNG for Gaussian shot noise approximation — advances
        # on each evaluate() call to produce realistic stochastic noise.
        self._rng = np.random.default_rng(seed_simulator)

        # Emit DeprecationWarning if mitigation flags are active — they are
        # NOT applied by this backend (raw-only).
        if mitigation is not None and (
            mitigation.zne_enabled
            or mitigation.dd_enabled
            or mitigation.trex_enabled
            or mitigation.twirling_enabled
        ):
            import warnings

            warnings.warn(
                "NoisyBackend does not apply mitigation options. "
                "Use run_gate_folding_zne() or run_pea_zne() from "
                "qmbp_simulation.execution.noisy_utils for mitigated estimation.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Store for reference only (never consumed in evaluate)
        self._mitigation = mitigation or MitigationOptions()

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
            return float(exact_energy + noise)

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
        backend_name: str = "ibm_kingston",
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


# ═══════════════════════════════════════════════════════════════════════════════
# Backend Factory
# ═══════════════════════════════════════════════════════════════════════════════

# Threshold: N ≤ EXACT_DIAG_QUBIT_LIMIT → StatevectorEstimator (exact, fastest).
#            N > EXACT_DIAG_QUBIT_LIMIT → MPSBackend (Aer MPS, χ=64, O(N·χ³) per eval).
# Rationale: StatevectorEstimator is O(2^N) per gate application. At N=20 with
# 57 RZZ gates, a single eval takes >60s. MPSBackend at χ=64 does ~127ms/eval
# and is exact for HVA p≤2 on 1D-like topologies (validated |MPS-SV|≈1e-14).


def select_backend(
    n_qubits: int,
    *,
    chi_max: int = 64,
    deterministic: bool = True,
    seed: int | None = None,
    for_vqe_loop: bool = False,
) -> ExecutionBackend:
    """Auto-select the optimal noiseless backend based on system size.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    chi_max : int
        MPS bond dimension (only used when N > EXACT_DIAG_QUBIT_LIMIT).
        Default 64 is exact for HVA p≤2 on 1D TFIM at any N.
    deterministic : bool
        If True (default), MPS uses exact expectation value computation.
        If False, uses shot-based sampling (for noise-tolerance testing).
    seed : int | None
        Random seed for reproducibility.
    for_vqe_loop : bool
        If True, optimizes for iterative evaluation (VQE optimization loops).
        Uses MPS for N>10 instead of N>15, because StatevectorEstimator's
        O(2^N) scaling makes VQE prohibitively slow at N≥12 (~60s/point vs
        ~0.1s/point with MPS at N=12). Default False preserves original
        behavior for single evaluations.

    Returns
    -------
    ExecutionBackend
        NoiselessBackend for small N, MPSBackend for larger N.

    Examples
    --------
    >>> from qmbp_simulation.execution import select_backend
    >>> backend = select_backend(n_qubits=20)  # → MPSBackend
    >>> backend = select_backend(n_qubits=6)   # → NoiselessBackend
    >>> backend = select_backend(n_qubits=12, for_vqe_loop=True)  # → MPSBackend
    """
    from qmbp_simulation.models.constants import EXACT_DIAG_QUBIT_LIMIT, MPS_DEFAULT_CHI_MAX

    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}.")

    if chi_max == 64:
        chi_max = MPS_DEFAULT_CHI_MAX  # Use canonical constant

    # For VQE loops, use a lower threshold (N>10) because StatevectorEstimator
    # is O(2^N) per eval and VQE does thousands of evals per h-point.
    threshold = 10 if for_vqe_loop else EXACT_DIAG_QUBIT_LIMIT

    if n_qubits <= threshold:
        logger.debug("select_backend: N=%d ≤ %d → NoiselessBackend", n_qubits, threshold)
        return NoiselessBackend()
    else:
        from qmbp_simulation.execution.mps_backend import MPSBackend

        logger.debug("select_backend: N=%d > %d → MPSBackend(χ=%d)", n_qubits, threshold, chi_max)
        return MPSBackend(
            strategy="aer_mps",
            chi_max=chi_max,
            deterministic=deterministic,
            seed=seed,
        )
