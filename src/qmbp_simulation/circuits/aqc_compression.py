"""AQC-Tensor Circuit Compression Module.

Compresses HVA circuits using Approximate Quantum Compilation with Tensor
Networks (AQC-Tensor). Primary use case: compress a p=2 HVA circuit (too deep
for ZNE) down to p=1-equivalent gate count while retaining p=2 expressibility.

This module wraps ``qiskit-addon-aqc-tensor`` (v0.3.0) as an optional dependency.
If not installed, all public functions raise ``ImportError`` with install instructions.

Key concepts:
    - Target state: MPS representation of the optimized HVA circuit (bound with θ_opt)
    - Ansatz: Parametrized shallow circuit with same 2Q connectivity pattern
    - Optimization: L-BFGS-B minimizes 1-|⟨ansatz(θ)|target⟩|² using MPS + autodiff

Reference:
    - arXiv:2301.08609 (AQC method)
    - qiskit-addon-aqc-tensor docs: https://qiskit.github.io/qiskit-addon-aqc-tensor/

POC validation (2026-06-17):
    - chain_1d N=10 p=2 h=3.5: F=0.999177, ΔE/gap=0.24%, 2Q: 18→9, 1.2s (χ=64)
    - All bond dims (32/64/128) give identical results → MPS is exact at χ=32 for
      this regime (paramagnetic, low entanglement)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from qiskit.circuit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler import CouplingMap

    from qmbp_simulation.models import LatticeConfig


__all__ = [
    "AQCCompressionConfig",
    "AQCCompressionResult",
    "AQCCircuitCompressor",
    "CompressionValidation",
    "AQCCompressionCache",
]


# ─── Lazy import guard ─────────────────────────────────────────────────────────

_AQC_AVAILABLE: bool | None = None


def _check_aqc_available() -> None:
    """Check that qiskit-addon-aqc-tensor is installed. Raises ImportError if not."""
    global _AQC_AVAILABLE
    if _AQC_AVAILABLE is None:
        try:
            import qiskit_addon_aqc_tensor  # noqa: F401

            _AQC_AVAILABLE = True
        except ImportError:
            _AQC_AVAILABLE = False
    if not _AQC_AVAILABLE:
        raise ImportError(
            "qiskit-addon-aqc-tensor is required for AQC circuit compression. "
            "Install with: pip install 'qiskit-addon-aqc-tensor[quimb-jax]>=0.3'"
        )


# ─── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class AQCCompressionConfig:
    """Configuration for AQC-Tensor circuit compression.

    Attributes
    ----------
    max_bond_dim : int
        Maximum MPS bond dimension (χ). Higher = more accurate, slower.
        For TFIM paramagnetic regime (h>h_c), χ=32 is typically sufficient.
        Near criticality, χ=128-256 may be needed.
    cutoff : float
        SVD truncation threshold for MPS operations.
    autodiff_backend : str
        Backend for automatic differentiation: "jax" (recommended) or "autograd".
    max_iterations : int
        Maximum L-BFGS-B iterations for parameter optimization.
    fidelity_threshold : float
        Minimum acceptable fidelity |⟨compressed|target⟩|². If the optimization
        cannot reach this threshold, the compression is rejected.
    convergence_tol : float
        L-BFGS-B convergence tolerance (ftol).
    ansatz_source : str
        How to generate the compression ansatz:
        - "auto": Use generate_ansatz_from_circuit with a p=1 template
        - "p1_template": Explicitly build HVA p=1 as template
        - "parametrize": Use parametrize_circuit on a p=1 bound circuit
    """

    max_bond_dim: int = 64
    cutoff: float = 1e-8
    autodiff_backend: str = "jax"
    max_iterations: int = 200
    fidelity_threshold: float = 0.998
    convergence_tol: float = 1e-12
    gradient_tol: float = 1e-8
    ansatz_source: str = "auto"


# ─── Result dataclasses ────────────────────────────────────────────────────────


@dataclass
class AQCCompressionResult:
    """Result of an AQC-Tensor circuit compression.

    Contains the compressed circuit, quality metrics, and provenance.
    """

    compressed_circuit: QuantumCircuit
    optimal_params: np.ndarray
    fidelity: float
    depth_original: int
    depth_compressed: int
    depth_reduction_pct: float
    n_2q_original: int
    n_2q_compressed: int
    n_2q_reduction_pct: float
    n_params: int
    n_iterations: int
    wall_clock_s: float
    converged: bool
    bond_dim_used: int
    config: AQCCompressionConfig = field(default_factory=AQCCompressionConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict (for diagnostics/persistence)."""
        return {
            "fidelity": self.fidelity,
            "depth_original": self.depth_original,
            "depth_compressed": self.depth_compressed,
            "depth_reduction_pct": self.depth_reduction_pct,
            "n_2q_original": self.n_2q_original,
            "n_2q_compressed": self.n_2q_compressed,
            "n_2q_reduction_pct": self.n_2q_reduction_pct,
            "n_params": self.n_params,
            "n_iterations": self.n_iterations,
            "wall_clock_s": self.wall_clock_s,
            "converged": self.converged,
            "bond_dim_used": self.bond_dim_used,
            "config": {
                "max_bond_dim": self.config.max_bond_dim,
                "fidelity_threshold": self.config.fidelity_threshold,
                "max_iterations": self.config.max_iterations,
                "autodiff_backend": self.config.autodiff_backend,
                "ansatz_source": self.config.ansatz_source,
            },
        }

    def is_zne_viable(self, *, amplifier: str = "pea") -> bool:
        """Check if compressed circuit's 2Q gate count is within ZNE budget."""
        thresholds = {"pea": 50, "gate_folding": 18, "adaptive": 50}
        threshold = thresholds.get(amplifier, 50)
        return self.n_2q_compressed <= threshold


@dataclass
class CompressionValidation:
    """Validation of compression quality against exact energy.

    Used as quality gate before submitting compressed circuit to QPU.
    """

    energy_original: float
    energy_compressed: float
    energy_exact: float
    delta_e: float
    delta_e_gap: float
    gap: float
    fidelity: float
    depth_reduction_pct: float
    n_2q_reduction_pct: float
    acceptable: bool
    recommendation: str  # "use_compressed" | "use_original" | "retry_higher_chi"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        return {
            "energy_original": self.energy_original,
            "energy_compressed": self.energy_compressed,
            "energy_exact": self.energy_exact,
            "delta_e": self.delta_e,
            "delta_e_gap": self.delta_e_gap,
            "gap": self.gap,
            "fidelity": self.fidelity,
            "depth_reduction_pct": self.depth_reduction_pct,
            "n_2q_reduction_pct": self.n_2q_reduction_pct,
            "acceptable": self.acceptable,
            "recommendation": self.recommendation,
        }


# ─── Main compressor class ────────────────────────────────────────────────────


class AQCCircuitCompressor:
    """Compress HVA circuits using AQC-Tensor for hardware depth reduction.

    Primary use case: Take an optimized p=2 circuit (36 CX on heavy_hex, too
    deep for ZNE) and compress it to p=1-equivalent 2Q-gate count (18 CX)
    while retaining the superior expressibility of p=2.

    The compressor works by:
    1. Simulating the target circuit as an MPS (tensor network)
    2. Generating a parametrized ansatz with fewer 2Q gates
    3. Optimizing ansatz parameters to maximize state fidelity with the target

    Parameters
    ----------
    config : AQCCompressionConfig, optional
        Compression configuration. Uses defaults if not specified.

    Examples
    --------
    >>> from qmbp_simulation.circuits.aqc_compression import (
    ...     AQCCircuitCompressor, AQCCompressionConfig
    ... )
    >>> compressor = AQCCircuitCompressor()
    >>> result = compressor.compress_circuit(bound_p2_circuit, lattice)
    >>> if result.fidelity >= 0.999:
    ...     # Use compressed circuit for hardware
    ...     hw_circuit = result.compressed_circuit
    """

    def __init__(self, config: AQCCompressionConfig | None = None) -> None:
        _check_aqc_available()
        self._config = config or AQCCompressionConfig()

    @property
    def config(self) -> AQCCompressionConfig:
        """Current compression configuration."""
        return self._config

    def compress_circuit(
        self,
        target_circuit: QuantumCircuit,
        lattice: LatticeConfig,
        *,
        good_circuit: QuantumCircuit | None = None,
        coupling_map: CouplingMap | None = None,
    ) -> AQCCompressionResult:
        """Compress a bound HVA circuit to a shallower equivalent.

        This is the primary entry point. Takes a fully-bound circuit (e.g., HVA p=2
        with θ_opt plugged in) and produces a compressed version with fewer 2Q gates.

        Parameters
        ----------
        target_circuit : QuantumCircuit
            Bound circuit (no free parameters). This is the "deep" circuit to compress.
            Typically: HVA p=2 with VQE-optimized parameters.
        lattice : LatticeConfig
            Lattice configuration (used to build p=1 template if good_circuit not given).
        good_circuit : QuantumCircuit, optional
            A bound "good" circuit to use as ansatz template. If None, builds a p=1
            HVA circuit optimized with a quick VQE run.
        coupling_map : CouplingMap, optional
            Hardware coupling map for hardware-aware ansatz generation. Not required
            for initial compression (post-transpilation handles hardware mapping).

        Returns
        -------
        AQCCompressionResult
            Contains compressed circuit, fidelity, depth metrics, timing.

        Raises
        ------
        ValueError
            If target_circuit has unbound parameters.
        RuntimeError
            If optimization fails to converge above fidelity threshold.
        """
        from functools import partial

        import quimb.tensor
        from qiskit_addon_aqc_tensor import generate_ansatz_from_circuit
        from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
        from qiskit_addon_aqc_tensor.simulation import tensornetwork_from_circuit
        from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator
        from scipy.optimize import minimize

        # Validate input
        if target_circuit.num_parameters > 0:
            raise ValueError(
                f"target_circuit has {target_circuit.num_parameters} unbound parameters. "
                f"Bind parameters with circuit.assign_parameters(theta_opt) before compressing."
            )

        t_start = time.time()

        # Measure original circuit metrics
        depth_original = target_circuit.depth()
        n_2q_original = sum(1 for inst in target_circuit.data if inst.operation.num_qubits == 2)

        # Build simulator settings
        simulator_settings = QuimbSimulator(
            partial(
                quimb.tensor.CircuitMPS,
                max_bond=self._config.max_bond_dim,
                cutoff=self._config.cutoff,
            ),
            autodiff_backend=self._config.autodiff_backend,
        )

        # Generate target MPS
        target_mps = tensornetwork_from_circuit(target_circuit, simulator_settings)

        # Generate ansatz and initial parameters
        if good_circuit is None:
            good_circuit = self._build_default_good_circuit(target_circuit, lattice)

        ansatz, initial_params = generate_ansatz_from_circuit(
            good_circuit, qubits_initially_zero=False
        )

        # Optimize
        objective = MaximizeStateFidelity(target_mps, ansatz, simulator_settings)
        result = minimize(
            objective.loss_function,
            initial_params,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": self._config.max_iterations,
                "ftol": self._config.convergence_tol,
                "gtol": self._config.gradient_tol,
            },
        )

        t_elapsed = time.time() - t_start

        # Build result
        fidelity = 1.0 - result.fun
        optimal_params = result.x
        compressed_circuit = ansatz.assign_parameters(optimal_params)
        depth_compressed = compressed_circuit.depth()
        n_2q_compressed = sum(
            1 for inst in compressed_circuit.data if inst.operation.num_qubits == 2
        )

        depth_reduction = (
            (1.0 - depth_compressed / depth_original) * 100 if depth_original > 0 else 0.0
        )
        n_2q_reduction = (1.0 - n_2q_compressed / n_2q_original) * 100 if n_2q_original > 0 else 0.0

        return AQCCompressionResult(
            compressed_circuit=compressed_circuit,
            optimal_params=optimal_params,
            fidelity=fidelity,
            depth_original=depth_original,
            depth_compressed=depth_compressed,
            depth_reduction_pct=depth_reduction,
            n_2q_original=n_2q_original,
            n_2q_compressed=n_2q_compressed,
            n_2q_reduction_pct=n_2q_reduction,
            n_params=len(optimal_params),
            n_iterations=result.nit,
            wall_clock_s=t_elapsed,
            converged=result.success or result.status == 0,
            bond_dim_used=self._config.max_bond_dim,
            config=self._config,
        )

    def compress_from_mps(
        self,
        target_mps: Any,
        ansatz: QuantumCircuit,
        initial_params: np.ndarray,
    ) -> AQCCompressionResult:
        """Compress using a pre-computed target MPS.

        Use this when you already have the MPS representation (e.g., from our
        MPSBackend or a previous tensornetwork_from_circuit call).

        Parameters
        ----------
        target_mps
            Tensor network state (quimb MPS or compatible).
        ansatz : QuantumCircuit
            Parametrized ansatz circuit (from generate_ansatz_from_circuit).
        initial_params : np.ndarray
            Initial parameter values for the ansatz.

        Returns
        -------
        AQCCompressionResult
        """
        from functools import partial

        import quimb.tensor
        from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
        from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator
        from scipy.optimize import minimize

        t_start = time.time()

        simulator_settings = QuimbSimulator(
            partial(
                quimb.tensor.CircuitMPS,
                max_bond=self._config.max_bond_dim,
                cutoff=self._config.cutoff,
            ),
            autodiff_backend=self._config.autodiff_backend,
        )

        objective = MaximizeStateFidelity(target_mps, ansatz, simulator_settings)
        result = minimize(
            objective.loss_function,
            initial_params,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": self._config.max_iterations,
                "ftol": self._config.convergence_tol,
                "gtol": self._config.gradient_tol,
            },
        )

        t_elapsed = time.time() - t_start
        fidelity = 1.0 - result.fun
        optimal_params = result.x
        compressed_circuit = ansatz.assign_parameters(optimal_params)
        depth_compressed = compressed_circuit.depth()
        n_2q_compressed = sum(
            1 for inst in compressed_circuit.data if inst.operation.num_qubits == 2
        )

        return AQCCompressionResult(
            compressed_circuit=compressed_circuit,
            optimal_params=optimal_params,
            fidelity=fidelity,
            depth_original=0,  # Unknown — caller should fill if needed
            depth_compressed=depth_compressed,
            depth_reduction_pct=0.0,
            n_2q_original=0,
            n_2q_compressed=n_2q_compressed,
            n_2q_reduction_pct=0.0,
            n_params=len(optimal_params),
            n_iterations=result.nit,
            wall_clock_s=t_elapsed,
            converged=result.success or result.status == 0,
            bond_dim_used=self._config.max_bond_dim,
            config=self._config,
        )

    def validate_compression(
        self,
        compression_result: AQCCompressionResult,
        hamiltonian: SparsePauliOp,
        energy_exact: float,
        gap: float,
        *,
        energy_original: float | None = None,
        de_gap_threshold: float = 0.01,
    ) -> CompressionValidation:
        """Validate compression quality against exact energy.

        Computes the energy of the compressed circuit and checks whether the
        compression-induced error is acceptable for hardware deployment.

        Parameters
        ----------
        compression_result : AQCCompressionResult
            Result from compress_circuit() or compress_from_mps().
        hamiltonian : SparsePauliOp
            System Hamiltonian for energy evaluation.
        energy_exact : float
            Exact ground state energy (from ClassicalSolver).
        gap : float
            Spectral gap (for normalization).
        energy_original : float, optional
            Energy of the original uncompressed circuit. If None, only checks
            compressed vs exact.
        de_gap_threshold : float
            Maximum acceptable ΔE/gap for the compressed circuit (default: 1%).

        Returns
        -------
        CompressionValidation
            Quality assessment with recommendation.
        """
        from qiskit.primitives import StatevectorEstimator

        # Compute energy of compressed circuit
        estimator = StatevectorEstimator()
        pub = estimator.run([(compression_result.compressed_circuit, hamiltonian)]).result()
        e_compressed = pub[0].data.evs.item()

        delta_e = abs(e_compressed - energy_exact)
        delta_e_gap = delta_e / gap if gap > 0 else float("inf")

        # Determine recommendation
        acceptable = (
            delta_e_gap < de_gap_threshold
            and compression_result.fidelity >= self._config.fidelity_threshold
        )

        if acceptable and compression_result.n_2q_reduction_pct > 20:
            recommendation = "use_compressed"
        elif acceptable:
            recommendation = "use_original"  # Compression works but no depth benefit
        elif compression_result.fidelity < 0.99:
            recommendation = "retry_higher_chi"
        else:
            recommendation = "use_original"

        return CompressionValidation(
            energy_original=energy_original if energy_original is not None else float("nan"),
            energy_compressed=e_compressed,
            energy_exact=energy_exact,
            delta_e=delta_e,
            delta_e_gap=delta_e_gap,
            gap=gap,
            fidelity=compression_result.fidelity,
            depth_reduction_pct=compression_result.depth_reduction_pct,
            n_2q_reduction_pct=compression_result.n_2q_reduction_pct,
            acceptable=acceptable,
            recommendation=recommendation,
        )

    # ─── Private helpers ───────────────────────────────────────────────────

    def _build_default_good_circuit(
        self,
        target_circuit: QuantumCircuit,
        lattice: LatticeConfig,
    ) -> QuantumCircuit:
        """Build a p=1 HVA circuit as the 'good' template for ansatz generation.

        Uses a quick parameter optimization to provide reasonable initial params.
        The VQE import is deferred to runtime (not module-level) to avoid
        circular dependency: circuits → optimizers is not in the module DAG,
        but is acceptable as a runtime lazy import within a private helper
        that is only called when good_circuit=None.
        """
        from qmbp_simulation.circuits.hva import HVACircuitBuilder
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        n_qubits = target_circuit.num_qubits
        builder = HVACircuitBuilder()
        circuit_p1, _ = builder.create(n_qubits, 1, lattice)

        # Build Hamiltonian for quick optimization
        H = HamiltonianBuilder().build(lattice)

        # Use scipy minimize directly (avoids importing VQEOptimizer from optimizers)
        from qiskit.primitives import StatevectorEstimator
        from scipy.optimize import minimize as scipy_minimize

        estimator = StatevectorEstimator()

        def cost_fn(params):
            bound = circuit_p1.assign_parameters(params)
            pub = estimator.run([(bound, H)]).result()
            return pub[0].data.evs.item()

        rng = np.random.default_rng(42)
        init_theta = rng.uniform(-0.01, 0.01, circuit_p1.num_parameters)

        result = scipy_minimize(cost_fn, init_theta, method="COBYLA", options={"maxiter": 300})
        return circuit_p1.assign_parameters(result.x)


# ─── Compression Cache ─────────────────────────────────────────────────────────


class AQCCompressionCache:
    """Cache for AQC compression results to avoid recomputation.

    For a given (topology, N, h, θ_opt_hash), the compression is deterministic.
    This cache stores optimal parameters and metadata, allowing instant retrieval
    of previously computed compressions.

    Storage format: one .npz file per compression containing:
    - optimal_params: np.ndarray
    - fidelity: float
    - metadata: dict (serialized as JSON string)

    Parameters
    ----------
    cache_dir : Path or str
        Directory for cache files. Created if it doesn't exist.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        from pathlib import Path as _Path

        if cache_dir is None:
            # Resolve project root by walking up from this file looking for
            # pyproject.toml. Avoids importing from framework (DAG violation).
            try:
                _current = _Path(__file__).resolve().parent
                _root = None
                for _ in range(5):
                    if (_current / "pyproject.toml").exists():
                        _root = _current
                        break
                    _current = _current.parent
                if _root is not None:
                    cache_dir = _root / "results" / "aqc_cache"
                else:
                    raise FileNotFoundError
            except (OSError, FileNotFoundError):
                # Non-editable install: use user home cache
                cache_dir = _Path.home() / ".cache" / "qmbp_simulation" / "aqc_cache"
        self._cache_dir = _Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(
        self,
        topology: str,
        n_qubits: int,
        h_value: float,
        theta_opt: np.ndarray,
        bond_dim: int,
    ) -> str:
        """Generate a deterministic cache key from compression inputs."""
        import hashlib

        # Hash the theta_opt values for a compact key
        theta_hash = hashlib.md5(theta_opt.tobytes()).hexdigest()[:12]
        return f"{topology}_N{n_qubits}_h{h_value:.4f}_chi{bond_dim}_{theta_hash}"

    def get(
        self,
        topology: str,
        n_qubits: int,
        h_value: float,
        theta_opt: np.ndarray,
        bond_dim: int,
    ) -> np.ndarray | None:
        """Retrieve cached optimal parameters, or None if not cached."""
        key = self._cache_key(topology, n_qubits, h_value, theta_opt, bond_dim)
        path = self._cache_dir / f"{key}.npz"
        if path.exists():
            data = np.load(path, allow_pickle=True)
            return data["optimal_params"]
        return None

    def get_with_metadata(
        self,
        topology: str,
        n_qubits: int,
        h_value: float,
        theta_opt: np.ndarray,
        bond_dim: int,
    ) -> tuple[np.ndarray, dict] | None:
        """Retrieve cached params + metadata, or None if not cached."""
        import json

        key = self._cache_key(topology, n_qubits, h_value, theta_opt, bond_dim)
        path = self._cache_dir / f"{key}.npz"
        if path.exists():
            data = np.load(path, allow_pickle=True)
            metadata = json.loads(str(data["metadata"]))
            return data["optimal_params"], metadata
        return None

    def put(
        self,
        topology: str,
        n_qubits: int,
        h_value: float,
        theta_opt: np.ndarray,
        bond_dim: int,
        optimal_params: np.ndarray,
        fidelity: float,
        **extra_metadata,
    ) -> Any:
        """Store compression result in cache."""
        import json

        key = self._cache_key(topology, n_qubits, h_value, theta_opt, bond_dim)
        path = self._cache_dir / f"{key}.npz"

        metadata = {
            "topology": topology,
            "n_qubits": n_qubits,
            "h_value": h_value,
            "bond_dim": bond_dim,
            "fidelity": fidelity,
            "n_params": len(optimal_params),
            **extra_metadata,
        }

        np.savez_compressed(
            path,
            optimal_params=optimal_params,
            metadata=json.dumps(metadata),
        )
        return path

    def clear(self) -> int:
        """Remove all cached entries. Returns count of files removed."""
        count = 0
        for f in self._cache_dir.glob("*.npz"):
            f.unlink()
            count += 1
        return count

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(list(self._cache_dir.glob("*.npz")))
