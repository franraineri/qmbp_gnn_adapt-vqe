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
            raise ValueError(f"Invalid dd_sequence '{self.dd_sequence}'. Valid values: 'XX', 'XpXm', 'XY4'")
        if self.zne_noise_factors is not None:
            if len(self.zne_noise_factors) < 2:
                raise ValueError(
                    f"zne_noise_factors must have at least 2 elements for extrapolation, "
                    f"got {len(self.zne_noise_factors)}."
                )
            if self.zne_noise_factors != sorted(self.zne_noise_factors):
                raise ValueError(f"zne_noise_factors must be in ascending order, got {self.zne_noise_factors}.")
            if self.zne_noise_factors[0] < 1.0:
                raise ValueError(
                    f"zne_noise_factors[0] must be >= 1.0 (base noise level), got {self.zne_noise_factors[0]}."
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

    def gradient(self, circuit: QuantumCircuit, hamiltonian: SparsePauliOp):
        """Return an exact analytic gradient callable ``grad(params)->np.ndarray``
        for ``d<H>/dparams``, or ``None`` if this backend cannot provide one.
        """
        return None

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
                "[%s] compute_fidelity: value %.8f slightly outside [0, 1] — numerical noise, clipping.",
                self.name,
                fid,
            )
        fid = float(np.clip(fid, 0.0, 1.0))

        logger.debug("[%s] compute_fidelity: result=%.6f", self.name, fid)
        return fid

    def compute_energy_variance(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        params: np.ndarray,
    ) -> float:
        """Compute energy variance Var(H) = ⟨H²⟩ - ⟨H⟩².

        The energy variance measures how close the state |ψ(θ)⟩ is to an
        eigenstate of H. For an exact eigenstate, Var(H) = 0. A large
        variance indicates the state is a superposition of eigenstates with
        different energies.

        This metric is independent of knowing E_exact and quantifies the
        "eigenstate quality" of the VQE solution:
        - Var(H) = 0: perfect eigenstate
        - Var(H) < 10⁻⁶: essentially an eigenstate (numerical noise)
        - Var(H) ~ 10⁻³: close to eigenstate (typical for good VQE)
        - Var(H) ~ 10⁻¹: poor convergence or high entanglement mismatch

        Parameters
        ----------
        circuit : QuantumCircuit
            Parameterized circuit (not yet bound).
        hamiltonian : SparsePauliOp
            The Hamiltonian H.
        params : np.ndarray
            Parameter values to bind.

        Returns
        -------
        float
            Var(H) = ⟨H²⟩ - ⟨H⟩². Always ≥ 0 (clipped for numerical safety).

        Notes
        -----
        Default implementation uses statevector extraction (O(2^N) memory).
        MPSBackend overrides this for N>22 with an efficient MPS-based approach.
        For backends where statevector extraction is infeasible and no override
        exists, returns NaN.
        """
        from qmbp_simulation.models.constants import STATEVECTOR_MAX_N

        n_qubits = circuit.num_qubits

        if n_qubits > STATEVECTOR_MAX_N:
            logger.debug(
                "[%s] compute_energy_variance: N=%d > %d, returning NaN "
                "(statevector extraction infeasible, no MPS override).",
                self.name,
                n_qubits,
                STATEVECTOR_MAX_N,
            )
            return float("nan")

        try:
            sv_data = self.get_statevector(circuit, params)
        except Exception as e:
            logger.warning("[%s] compute_energy_variance: get_statevector failed: %s", self.name, e)
            return float("nan")

        # ⟨H⟩
        from qiskit.quantum_info import Statevector

        sv = Statevector(sv_data)
        e_mean = float(np.real(sv.expectation_value(hamiltonian)))

        # ⟨H²⟩ via H² operator
        h_squared = hamiltonian @ hamiltonian
        # Simplify H² to reduce term count (combine duplicate Paulis)
        # Critical for large Hamiltonians where |H²| ∝ |H|² terms
        h_squared = h_squared.simplify(atol=1e-12)
        e2_mean = float(np.real(sv.expectation_value(h_squared)))

        variance = e2_mean - e_mean**2

        # Sub-noise negatives → clamp to 0 (genuine eigenstate). A beyond-noise
        # negative on the exact statevector path signals a real numerical problem
        # (e.g. a malformed H²); return NaN so the fidelity bound reports N/A
        # rather than a spurious Var=0 → F≈1.
        if variance < 0:
            if variance < -1e-8:
                logger.warning(
                    "[%s] compute_energy_variance: negative variance %.2e (numerical issue) — returning NaN.",
                    self.name,
                    variance,
                )
                return float("nan")
            variance = 0.0

        # Warning for unexpectedly large variance (possible poor VQE convergence)
        if variance > 10.0:
            logger.warning(
                "[%s] compute_energy_variance: very large variance %.2f "
                "(state is far from any eigenstate — check VQE convergence).",
                self.name,
                variance,
            )

        return float(variance)

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

    def __init__(self, cache_hamiltonian: bool = True) -> None:
        from qiskit.primitives import StatevectorEstimator

        self._estimator = StatevectorEstimator()
        # When enabled, ⟨H⟩ is computed as ⟨ψ|H_sparse|ψ⟩ with H materialized
        # to a CSR matrix ONCE per (hamiltonian object, N) and reused across
        # calls. This avoids the StatevectorEstimator's per-call symbolic
        # observable rebuild (SparseObservable.from_sparse_pauli_op), which
        # dominates runtime when a gradient calls evaluate() many times.
        # Numerically identical to the estimator path (same exact inner product).
        self._cache_hamiltonian = cache_hamiltonian
        self._hmat_cache: dict[int, Any] = {}

    def _hmat(self, hamiltonian: SparsePauliOp):
        """CSR matrix for ``hamiltonian``, cached by object id (materialized once)."""
        key = id(hamiltonian)
        cached = self._hmat_cache.get(key)
        if cached is None:
            cached = hamiltonian.to_matrix(sparse=True).tocsr()
            self._hmat_cache[key] = cached
        return cached

    def evaluate(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        params: np.ndarray,
    ) -> float:
        """Evaluate ⟨H⟩ using exact statevector simulation.

        With ``cache_hamiltonian`` (default) this binds the circuit, extracts the
        statevector, and returns ⟨ψ|H|ψ⟩ against a cached sparse H — identical to
        the estimator result but far cheaper under repeated (gradient) calls.
        """
        if len(params) != circuit.num_parameters:
            raise ValueError(f"Parameter count mismatch: got {len(params)}, expected {circuit.num_parameters}.")
        if not np.all(np.isfinite(params)):
            raise ValueError(
                f"NoiselessBackend.evaluate: params contain NaN/Inf. "
                f"Non-finite indices: {np.where(~np.isfinite(params))[0].tolist()}"
            )

        if self._cache_hamiltonian:
            from qiskit.quantum_info import Statevector

            psi = np.asarray(Statevector(circuit.assign_parameters(params)).data)
            hmat = self._hmat(hamiltonian)
            energy = float(np.real(np.vdot(psi, hmat @ psi)))
        else:
            bound = circuit.assign_parameters(params)
            job = self._estimator.run([(bound, hamiltonian)])
            energy = float(job.result()[0].data.evs)

        if not np.isfinite(energy):
            raise RuntimeError(
                f"Non-finite energy returned from NoiselessBackend.evaluate: {energy}. "
                f"Check circuit parameters for NaN/Inf."
            )
        return energy

    def energy_and_fidelity(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        params: np.ndarray,
        target_state: np.ndarray,
    ) -> tuple[float, float]:
        """Build the statevector ONCE and return (⟨H⟩, |⟨target|ψ⟩|²) together.

        A fused evaluate+fidelity: the two quantities share the single
        ``Statevector`` construction (the expensive part), avoiding the duplicate
        state build that ``cost`` and ``fid`` incur when both are needed at the
        same θ (candidate ranking, convergence probes). Numerically identical to
        calling them separately. Returns ``(energy, fidelity)``.
        """
        from qiskit.quantum_info import Statevector

        psi = np.asarray(Statevector(circuit.assign_parameters(params)).data)
        hmat = self._hmat(hamiltonian)
        energy = float(np.real(np.vdot(psi, hmat @ psi)))
        fid = float(abs(np.vdot(np.asarray(target_state), psi)) ** 2)
        return energy, float(np.clip(fid, 0.0, 1.0))

    @property
    def name(self) -> str:
        return "noiseless_statevector"

    def gradient(self, circuit: QuantumCircuit, hamiltonian: SparsePauliOp):
        """Exact adjoint (reverse-mode) gradient of ⟨H⟩ for statevector sim.

        Fast path: when every parametrized gate is a single-parameter ``rx`` /
        ``rz`` / ``rzz`` (the HVA/bond-resolved family), a hand-rolled adjoint
        computes ∇⟨H⟩ in ONE forward + ONE backward sweep using vectorized Pauli
        applications and the cached H-CSR — no Qiskit estimator pipeline. This is
        numerically identical to ``ReverseEstimatorGradient`` (validated ≤1e-9)
        but ~8-20× faster because it skips the per-call symbolic observable
        rebuild and primitive plumbing that dominate the reverse estimator.

        Fallback: any unsupported parametrized gate (or a missing dependency)
        routes to ``qiskit_algorithms.gradients.ReverseEstimatorGradient`` — so
        the result is never worse than before, only faster where it applies.
        Returns ``grad(params)->np.ndarray`` or ``None``.
        """
        plan = self._adjoint_plan(circuit)
        if plan is not None:
            hmat = self._hmat(hamiltonian)

            def grad(params: np.ndarray) -> np.ndarray:
                return self._adjoint_gradient(plan, hmat, np.asarray(params, float))

            return grad

        # Fallback: Qiskit reverse estimator (slower but fully general).
        try:
            from qiskit_algorithms.gradients import ReverseEstimatorGradient
        except Exception:
            return None
        rev = ReverseEstimatorGradient()

        def grad_fallback(params: np.ndarray) -> np.ndarray:
            job = rev.run([circuit], [hamiltonian], [list(params)])
            return np.asarray(job.result().gradients[0], dtype=float)

        return grad_fallback

    # ── fast analytic adjoint gradient (rx / rz / rzz circuits) ──────────────
    def _adjoint_plan(self, circuit: QuantumCircuit):
        """Compile ``circuit`` into an ordered gate plan for the adjoint sweep.

        Returns a dict with the fixed (non-parametric) and parametric gate lists
        aligned to the flat parameter vector, or ``None`` if any parametrized
        gate is not a single-parameter rx/rz/rzz (then the caller falls back).
        Cached by ``id(circuit)`` so the symbolic walk happens once.
        """
        cache = getattr(self, "_adjoint_plan_cache", None)
        if cache is None:
            cache = {}
            self._adjoint_plan_cache = cache
        key = id(circuit)
        if key in cache:
            return cache[key]

        try:
            params_order = list(circuit.parameters)
            pidx = {p: i for i, p in enumerate(params_order)}
            gates = []  # (kind, qubits, coeff, param_index_or_None, fixed_angle)
            for inst in circuit.data:
                op = inst.operation
                name = op.name
                qubits = tuple(circuit.find_bit(q).index for q in inst.qubits)
                if name == "h":
                    gates.append(("h", qubits, None, None, None))
                    continue
                if name in ("rx", "rz", "rzz"):
                    expr = op.params[0]
                    free = getattr(expr, "parameters", None)
                    if free:
                        if len(free) != 1:
                            cache[key] = None
                            return None
                        p = next(iter(free))
                        # linear coeff of the single parameter (rzz uses 2θ etc.)
                        coeff = float(expr.gradient(p)) if hasattr(expr, "gradient") \
                            else 1.0
                        gates.append((name, qubits, coeff, pidx[p], None))
                    else:
                        gates.append((name, qubits, None, None, float(expr)))
                    continue
                # any other gate type → bail to the general fallback
                cache[key] = None
                return None
            plan = {"n": circuit.num_qubits, "nparam": len(params_order),
                    "gates": gates}
            cache[key] = plan
            return plan
        except Exception:
            cache[key] = None
            return None

    @staticmethod
    def _apply_x(state, n, q):
        """X_q |state| (vectorized bit-flip on qubit q, little-endian)."""
        st = state.reshape((2,) * n)
        st = np.flip(st, axis=n - 1 - q)
        return st.reshape(-1)

    @staticmethod
    def _apply_z(state, n, q):
        """Z_q |state| (vectorized sign flip on qubit q)."""
        st = state.reshape((2,) * n).copy()
        sl = [slice(None)] * n
        sl[n - 1 - q] = 1
        st[tuple(sl)] *= -1.0
        return st.reshape(-1)

    @classmethod
    def _apply_generator(cls, kind, state, n, qubits):
        """Apply the (unit-norm) Pauli generator of an rx/rz/rzz gate."""
        if kind == "rx":
            return cls._apply_x(state, n, qubits[0])
        if kind == "rz":
            return cls._apply_z(state, n, qubits[0])
        # rzz: Z⊗Z
        return cls._apply_z(cls._apply_z(state, n, qubits[1]), n, qubits[0])

    @classmethod
    def _apply_gate(cls, kind, state, n, qubits, angle, dagger=False):
        """Apply exp(-i angle/2 · G) (rx/rz) or exp(-i angle/2 · ZZ) (rzz), or H."""
        if kind == "h":
            inv = 1.0 / np.sqrt(2.0)
            st = state.reshape((2,) * n)
            ax = n - 1 - qubits[0]
            st = np.moveaxis(st, ax, 0)
            a, b = st[0].copy(), st[1].copy()
            out = np.empty_like(st)
            out[0] = inv * (a + b)
            out[1] = inv * (a - b)
            out = np.moveaxis(out, 0, ax)
            return out.reshape(-1)
        theta = -angle if dagger else angle
        c = np.cos(theta / 2.0)
        s = -1j * np.sin(theta / 2.0)  # exp(-i θ/2 G) = c·I + s·G
        return c * state + s * cls._apply_generator(kind, state, n, qubits)

    def _adjoint_gradient(self, plan, hmat, params):
        """Exact ∇⟨H⟩ via one forward + one backward sweep (adjoint method)."""
        n = plan["n"]
        dim = 1 << n
        gates = plan["gates"]
        grad = np.zeros(plan["nparam"], dtype=float)

        # forward: |ψ⟩ = U_m … U_1 |0⟩
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        for kind, qubits, coeff, pj, fixed in gates:
            angle = fixed if pj is None else coeff * params[pj]
            if kind == "h":
                angle = None
            state = self._apply_gate(kind, state, n, qubits, angle)

        # backward: |λ⟩ = H|ψ⟩; sweep gates in reverse. At a parametric gate,
        # with |φ⟩,|λ⟩ at the POST-gate point, the generator G (which commutes
        # with its own U) gives  ∂⟨H⟩/∂θ = 2·Re⟨λ|(-i·coeff/2·G)|φ⟩
        #                                = coeff · Im⟨λ|G|φ⟩.
        # Then un-apply U† from both to move to the pre-gate point.
        lam = hmat @ state
        for kind, qubits, coeff, pj, fixed in reversed(gates):
            angle = fixed if (pj is None) else coeff * params[pj]
            if kind == "h":
                state = self._apply_gate("h", state, n, qubits, None)
                lam = self._apply_gate("h", lam, n, qubits, None)
                continue
            if pj is not None:
                g_phi = self._apply_generator(kind, state, n, qubits)
                grad[pj] += coeff * float(np.imag(np.vdot(lam, g_phi)))
            # move both states back to BEFORE this gate (apply U†)
            state = self._apply_gate(kind, state, n, qubits, angle, dagger=True)
            lam = self._apply_gate(kind, lam, n, qubits, angle, dagger=True)
        return grad


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
            mitigation.zne_enabled or mitigation.dd_enabled or mitigation.trex_enabled or mitigation.twirling_enabled
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
            raise ValueError(f"Parameter count mismatch: got {len(params)}, expected {circuit.num_parameters}.")
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
                "qiskit-aer is required for noise model simulation. Install with: pip install qiskit-aer"
            ) from e

        # Cache AerSimulator + PassManager to avoid re-creation per evaluate()
        # (transpilation overhead is ~30-50ms per call otherwise)
        if not hasattr(self, "_aer_backend") or self._aer_backend is None:
            self._aer_backend = AerSimulator(noise_model=self._noise_model)
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

            self._aer_pm = generate_preset_pass_manager(backend=self._aer_backend, optimization_level=1)

        from qiskit.primitives import BackendEstimatorV2

        bound = circuit.assign_parameters(params)
        isa_circuit = self._aer_pm.run(bound)

        precision = 1.0 / np.sqrt(self._shots)
        estimator = BackendEstimatorV2(backend=self._aer_backend)
        job = estimator.run([(isa_circuit, hamiltonian)], precision=precision)
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

    def compute_energy_variance(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        params: np.ndarray,
    ) -> float:
        """Return NaN — energy variance cannot be computed from hardware QPU."""
        return float("nan")

    def get_statevector(
        self,
        circuit: QuantumCircuit,
        params: np.ndarray,
    ) -> np.ndarray:
        """Raise error — statevector extraction not possible on real hardware."""
        raise RuntimeError(
            f"get_statevector() is not supported on HardwareBackend('{self._backend_name}'). "
            "Statevector extraction is only possible in simulation."
        )

    @property
    def name(self) -> str:
        return f"hardware_{self._backend_name}"


# ═══════════════════════════════════════════════════════════════════════════════
# FakeBackend — Coherent noise simulation via IBM fake providers
# ═══════════════════════════════════════════════════════════════════════════════


class FakeBackend(ExecutionBackend):
    """Coherent noise simulation using IBM Qiskit fake provider backends.

    Unlike NoisyBackend (shot noise only), FakeBackend uses real calibration
    data from IBM processors (FakeTorino, FakeKingston, etc.) to simulate
    coherent errors: T1/T2 decay, gate over-rotation, crosstalk, readout.

    This produces *systematic, structured* noise that is potentially
    learnable by an MPNN (coherent shift hypothesis, plan 06).

    Usage:
        from qmbp_simulation.execution import FakeBackend

        backend = FakeBackend("torino", shots=8192)
        energy = backend.evaluate(circuit, hamiltonian, params)

    Parameters
    ----------
    provider : str
        Fake backend name: "torino", "kingston", "brisbane".
        Maps to FakeTorino, FakeKingston, FakeBrisbane from
        qiskit_ibm_runtime.fake_provider.
    shots : int
        Number of measurement shots (default 8192).
    optimization_level : int
        Transpiler optimization level (default 1). Higher levels give
        shorter circuits but take longer to transpile.
    seed_simulator : int | None
        Random seed for reproducibility.

    Raises
    ------
    ImportError
        If qiskit-ibm-runtime is not installed.
    ValueError
        If provider name is not recognized.
    """

    _PROVIDERS = {
        "torino": "FakeTorino",
        "kingston": "FakeKingston",
        "brisbane": "FakeBrisbane",
    }

    def __init__(
        self,
        provider: str = "torino",
        shots: int = 8192,
        optimization_level: int = 1,
        seed_simulator: int | None = None,
    ) -> None:
        self._provider_name = provider.lower()
        self._shots = shots
        self._optimization_level = optimization_level
        self._seed_simulator = seed_simulator

        if self._provider_name not in self._PROVIDERS:
            raise ValueError(f"Unknown fake provider '{provider}'. Available: {list(self._PROVIDERS.keys())}")

        # Lazy initialization — only import when first evaluate() is called
        self._backend = None
        self._estimator = None

    def _ensure_backend(self) -> None:
        """Lazy-initialize the fake backend on first use."""
        if self._backend is not None:
            return

        try:
            from qiskit_ibm_runtime.fake_provider import (
                FakeBrisbane,
                FakeKingston,
                FakeTorino,
            )
        except ImportError as e:
            raise ImportError(
                "qiskit-ibm-runtime is required for FakeBackend. Install with: pip install qiskit-ibm-runtime"
            ) from e

        provider_map = {
            "torino": FakeTorino,
            "kingston": FakeKingston,
            "brisbane": FakeBrisbane,
        }
        self._backend = provider_map[self._provider_name]()
        logger.info(
            "FakeBackend: initialized %s (%d qubits)",
            self._PROVIDERS[self._provider_name],
            self._backend.num_qubits,
        )

    def evaluate(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        params: np.ndarray,
    ) -> float:
        """Evaluate energy with coherent noise from fake calibration data.

        Transpiles the circuit to the fake backend's native gate set and
        coupling map, then runs with BackendEstimatorV2.
        """
        if len(params) != circuit.num_parameters:
            raise ValueError(f"Parameter count mismatch: got {len(params)}, expected {circuit.num_parameters}.")

        self._ensure_backend()

        from qiskit.primitives import BackendEstimatorV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        # Transpile to native gates + coupling map
        pm = generate_preset_pass_manager(
            backend=self._backend,
            optimization_level=self._optimization_level,
        )
        bound = circuit.assign_parameters(params)
        isa_circuit = pm.run(bound)

        # Evaluate with BackendEstimatorV2
        precision = 1.0 / np.sqrt(self._shots)
        estimator = BackendEstimatorV2(backend=self._backend)
        job = estimator.run([(isa_circuit, hamiltonian)], precision=precision)
        energy = float(job.result()[0].data.evs)

        if not np.isfinite(energy):
            logger.warning(
                "FakeBackend: non-finite energy at %s. Circuit may be too deep for noise level.",
                self._provider_name,
            )
            return float("inf")

        return energy

    def compute_fidelity(
        self,
        circuit: QuantumCircuit,
        params: np.ndarray,
        exact_state: np.ndarray,
    ) -> float:
        """Not supported for FakeBackend (no statevector access)."""
        raise RuntimeError(
            "compute_fidelity() not supported on FakeBackend. Use noiseless backend for fidelity computation."
        )

    @property
    def name(self) -> str:
        return f"fake_{self._provider_name}_shots={self._shots}"


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


def select_backend_with_topology_warning(
    n_qubits: int,
    *,
    topology: str = "chain_1d",
    chi_max: int = 64,
    deterministic: bool = True,
    seed: int | None = None,
    for_vqe_loop: bool = False,
) -> ExecutionBackend:
    """Auto-select backend with 2D topology chi-sufficiency warning.

    Wraps select_backend() and emits a warning when MPS is selected for
    2D topologies with N>16, where chi=64 may be insufficient.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    topology : str
        Topology name (used for 2D chi warning).
    chi_max : int
        MPS bond dimension.
    deterministic : bool
        MPS deterministic mode.
    seed : int | None
        Random seed.
    for_vqe_loop : bool
        VQE loop optimization flag.

    Returns
    -------
    ExecutionBackend
        Selected backend (same as select_backend).
    """
    backend = select_backend(
        n_qubits,
        chi_max=chi_max,
        deterministic=deterministic,
        seed=seed,
        for_vqe_loop=for_vqe_loop,
    )

    # Warn for 2D topologies where MPS chi may be insufficient
    _2D_TOPOLOGIES = ("square", "triangular", "heavy_hex", "kagome")
    if topology in _2D_TOPOLOGIES and n_qubits > 16:
        if hasattr(backend, "_chi_max"):
            logger.warning(
                "2D topology '%s' with N=%d: MPS chi=%d may be insufficient. "
                "Results should include chi-convergence verification (--verify-chi).",
                topology,
                n_qubits,
                backend._chi_max,
            )

    return backend
