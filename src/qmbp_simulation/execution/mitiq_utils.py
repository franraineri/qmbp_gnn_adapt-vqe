"""Mitiq integration module — complementary error mitigation via Mitiq toolkit.

Wraps Mitiq techniques (CDR, DDD+ZNE composition, random-folding ZNE,
PEC benchmark) into the project's NoisyEstimatorConfig-based pattern.

This module is COMPLEMENTARY to the existing PEA/GF/adaptive ZNE stack:
- PEA-ZNE remains primary strategy for hardware (validated +94.4% gain)
- Mitiq CDR is the main addition: learning-based, no noise model required
- Mitiq DDD+ZNE stacking adds a second mitigation layer
- Mitiq PEC is only for offline benchmarking (exponential overhead)

Requires: pip install mitiq (optional dependency)

References:
- Mitiq docs: https://mitiq.readthedocs.io/en/stable/
- CDR: Czarnik et al., Quantum 5, 592 (2021)
- vnCDR: Lowe et al., arXiv:2011.01157
- PEA: Kim et al., Nature 618 (2023)
- GNN-QEM: Wang et al., arXiv:2604.16815

Usage:
    from qmbp_simulation.execution.mitiq_utils import (
        make_mitiq_executor,
        run_mitiq_zne,
        run_mitiq_cdr,
        run_mitiq_ddd_zne,
        run_mitiq_pec,
        compare_mitigation_strategies,
    )
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

_logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Mitiq availability guard
# ═══════════════════════════════════════════════════════════════════════════


def _check_mitiq() -> Any:
    """Lazy-import mitiq and return the module. Raises ImportError if unavailable."""
    try:
        import mitiq

        return mitiq
    except ImportError as e:
        raise ImportError(
            "mitiq is required for this function. "
            "Install with: pip install 'qmbp-simulation[mitiq]' or pip install mitiq"
        ) from e


def is_mitiq_available() -> bool:
    """Check if mitiq is installed without raising."""
    try:
        import mitiq  # noqa: F401

        return True
    except ImportError:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Result dataclasses
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class MitiqZNEResult:
    """Result from Mitiq ZNE with rich folding/extrapolation options."""

    extrapolated_value: float
    r_squared: float
    factory_name: str  # "linear"|"richardson"|"poly"|"exp"
    folding_method: str  # "random"|"global"|"all"
    scale_factors: list[float] = field(default_factory=list)
    measured_values: list[float] = field(default_factory=list)
    execution_time_s: float = 0.0


@dataclass
class MitiqCDRResult:
    """Result from Mitiq Clifford Data Regression."""

    mitigated_value: float
    raw_value: float
    improvement_pct: float  # |raw - mitigated| / |raw| * 100 (correction magnitude)
    n_training_circuits: int
    execution_time_s: float = 0.0


@dataclass
class MitiqDDDZNEResult:
    """Result from Mitiq DDD+ZNE composition."""

    extrapolated_value: float
    r_squared: float
    ddd_rule: str  # "xx"|"yy"|"xyxy"
    zne_factory: str
    scale_factors: list[float] = field(default_factory=list)
    measured_values: list[float] = field(default_factory=list)
    execution_time_s: float = 0.0


@dataclass
class MitiqPECResult:
    """Result from Mitiq Probabilistic Error Cancellation (benchmark only)."""

    mitigated_value: float
    raw_value: float
    improvement_pct: float
    n_samples: int
    overhead_factor: float
    execution_time_s: float = 0.0


@dataclass
class MitiqComparisonResult:
    """Result of multi-method mitigation comparison for one h-point."""

    h_value: float
    e_exact: float
    gap: float
    raw_energy: float
    results: dict[str, float] = field(default_factory=dict)
    delta_e_gaps: dict[str, float] = field(default_factory=dict)
    rankings: list[str] = field(default_factory=list)
    best_method: str = ""
    best_delta_e_gap: float = 0.0
    execution_time_s: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Executor Factory — adapts our backend+config to Mitiq's pattern
# ═══════════════════════════════════════════════════════════════════════════


def make_mitiq_executor(
    observable: SparsePauliOp,
    backend: Any,
    config: NoisyEstimatorConfig,
    seed_offset: int = 0,
    transpile: bool = True,
) -> Callable[[QuantumCircuit], float]:
    """Build a Mitiq-compatible executor from our backend + observable + config.

    Mitiq's core abstraction: `execute_with_<technique>(circuit, executor)`
    where `executor(circuit) → float`. This factory constructs that executor
    using our BackendEstimatorV2 pattern with proper seeds and precision.

    The executor handles:
    - Transpilation of folded circuits (Mitiq folds BEFORE transpilation)
    - BackendEstimatorV2 with correct seed_simulator and precision
    - Observable measurement and float extraction

    CRITICAL: Transpilation MUST use optimization_level=0 for Mitiq executors.
    Qiskit 2.x transpiler at level ≥ 1 cancels inverse gate pairs (U·U†),
    which destroys the gate-folding that Mitiq inserts for noise amplification.
    Verified: Qiskit 2.4.1 with opt_level=2 reduces 3 CX (folded) → 1 CX.

    Parameters
    ----------
    observable : SparsePauliOp
        Observable to measure. NOT layout-mapped (executor handles it).
    backend : BackendV2
        Noisy or noiseless backend (e.g., FakeTorino, AerSimulator).
    config : NoisyEstimatorConfig
        Shots, seed, optimization_level (IGNORED — forced to 0 for Mitiq).
    seed_offset : int
        Base offset for seed (executor adds per-call counter).
    transpile : bool
        If True (default), transpile each circuit Mitiq passes.
        Set False only if Mitiq operates on pre-transpiled circuits.

    Returns
    -------
    Callable[[QuantumCircuit], float]
        Mitiq-compatible executor.
    """
    from qiskit.primitives import BackendEstimatorV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    # Counter for per-call seed independence
    call_count = [0]

    # Pre-build pass manager if transpiling
    # CRITICAL: optimization_level=0 is MANDATORY for Mitiq executors.
    # Qiskit 2.x at opt_level>=1 cancels U·U† pairs, destroying gate-folding.
    # config.optimization_level is deliberately ignored here.
    pm = None
    if transpile:
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=0,
        )

    def executor(circuit: QuantumCircuit) -> float:
        """Execute circuit and return ⟨observable⟩."""
        nonlocal call_count

        # Transpile the (possibly folded) circuit
        if pm is not None:
            transpiled = pm.run(circuit)
        else:
            transpiled = circuit

        # Map observable to transpiled layout
        if pm is not None and hasattr(transpiled, "layout") and transpiled.layout is not None:
            mapped_obs = observable.apply_layout(transpiled.layout)
        else:
            mapped_obs = observable

        # Execute via BackendEstimatorV2
        current_seed = config.seed_simulator + seed_offset + call_count[0]
        call_count[0] += 1

        estimator = BackendEstimatorV2(
            backend=backend,
            options={
                "seed_simulator": current_seed,
                "default_precision": config.precision,
            },
        )
        job = estimator.run([(transpiled, mapped_obs)])
        return float(job.result()[0].data.evs)

    # Mitiq 1.0 Executor class uses typing.get_type_hints() on the callable
    # to determine return type. Closures from factory functions may lose the
    # correct `float` identity reference during annotation resolution.
    # Explicitly setting __annotations__ ensures Mitiq detects FloatLike.
    executor.__annotations__ = {"circuit": QuantumCircuit, "return": float}

    return executor


def make_noiseless_executor(
    observable: SparsePauliOp,
) -> Callable[[QuantumCircuit], float]:
    """Build a noiseless executor for CDR training data.

    Uses StatevectorEstimator (exact) — no noise, no shots, no seed needed.
    Required by CDR which needs both noisy and noiseless executor.

    Parameters
    ----------
    observable : SparsePauliOp
        Observable to measure.

    Returns
    -------
    Callable[[QuantumCircuit], float]
        Noiseless executor for CDR simulator_executor argument.
    """
    from qiskit.primitives import StatevectorEstimator

    def executor(circuit: QuantumCircuit) -> float:
        """Execute circuit noiseless and return exact ⟨observable⟩."""
        estimator = StatevectorEstimator()
        job = estimator.run([(circuit, observable)])
        return float(job.result()[0].data.evs)

    # Ensure Mitiq Executor detects float return type (closure annotation fix)
    executor.__annotations__ = {"circuit": QuantumCircuit, "return": float}

    return executor


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Mitiq ZNE — Random folding + rich extrapolation factories
# ═══════════════════════════════════════════════════════════════════════════


def _build_zne_factory(
    factory_name: str,
    scale_factors: tuple[float, ...],
    asymptote: float | None = None,
) -> Any:
    """Build a Mitiq extrapolation factory by name.

    Parameters
    ----------
    factory_name : str
        One of "linear", "richardson", "poly", "exp".
    scale_factors : tuple[float, ...]
        Noise scale factors (can be non-integer for Mitiq).
    asymptote : float | None
        Asymptote for exponential factory (required for "exp").

    Returns
    -------
    mitiq.zne.inference.Factory
        Configured factory instance.
    """
    _check_mitiq()
    from mitiq.zne.inference import (
        ExpFactory,
        LinearFactory,
        PolyFactory,
        RichardsonFactory,
    )

    if factory_name == "linear":
        return LinearFactory(scale_factors=list(scale_factors))
    elif factory_name == "richardson":
        return RichardsonFactory(scale_factors=list(scale_factors))
    elif factory_name == "poly":
        # Degree = len(scale_factors) - 1 (full polynomial)
        order = len(scale_factors) - 1
        return PolyFactory(scale_factors=list(scale_factors), order=order)
    elif factory_name == "exp":
        asymp = asymptote if asymptote is not None else 0.0
        return ExpFactory(scale_factors=list(scale_factors), asymptote=asymp)
    else:
        raise ValueError(
            f"Unknown factory: {factory_name!r}. Use 'linear', 'richardson', 'poly', or 'exp'."
        )


def _get_folding_function(folding_method: str) -> Callable:
    """Get Mitiq's folding function by name.

    Parameters
    ----------
    folding_method : str
        One of "random", "global", "all".

    Returns
    -------
    Callable
        Mitiq folding function (circuit, scale_factor) → folded_circuit.
    """
    _check_mitiq()
    from mitiq.zne.scaling import (
        fold_all,
        fold_gates_at_random,
        fold_global,
    )

    methods = {
        "random": fold_gates_at_random,
        "global": fold_global,
        "all": fold_all,
    }
    if folding_method not in methods:
        raise ValueError(
            f"Unknown folding method: {folding_method!r}. Use one of: {list(methods.keys())}"
        )
    return methods[folding_method]


def run_mitiq_zne(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    backend: Any,
    config: NoisyEstimatorConfig,
    scale_factors: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0),
    factory_name: str = "linear",
    folding_method: str = "random",
    asymptote: float | None = None,
    seed_offset: int = 0,
) -> MitiqZNEResult:
    """Run ZNE via Mitiq with configurable folding and extrapolation.

    Advantages over our native gate-folding ZNE:
    - Fractional scale factors (1.5, 2.5) for denser sampling
    - Random folding adds statistical robustness (ensemble averaging)
    - Richardson/Poly factories for non-linear E(λ) relationships
    - Exponential factory with asymptote for deep circuits

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to mitigate (NOT transpiled — executor handles transpilation).
        Must have measurements removed (Mitiq adds them internally for ZNE,
        but our executor uses Estimator, so circuit should be measurement-free).
    observable : SparsePauliOp
        Observable to measure.
    backend : BackendV2
        Noisy backend (e.g., FakeTorino).
    config : NoisyEstimatorConfig
        Shots, seed, optimization_level.
    scale_factors : tuple[float, ...]
        Noise amplification factors. Default (1.0, 1.5, 2.0, 2.5, 3.0).
    factory_name : str
        Extrapolation factory: "linear", "richardson", "poly", "exp".
    folding_method : str
        Gate folding strategy: "random", "global", "all".
    asymptote : float | None
        Asymptote for exponential factory.
    seed_offset : int
        Added to config.seed_simulator for independence.

    Returns
    -------
    MitiqZNEResult
        Extrapolation result with metadata.
    """
    _check_mitiq()
    from mitiq import zne

    t0 = time.time()
    _logger.info(
        f"[mitiq_zne] Starting: factory={factory_name}, folding={folding_method}, "
        f"scale_factors={scale_factors}, shots={config.shots}"
    )

    # Build executor and factory
    executor = make_mitiq_executor(observable, backend, config, seed_offset=seed_offset)
    factory = _build_zne_factory(factory_name, scale_factors, asymptote)
    fold_fn = _get_folding_function(folding_method)

    # Remove measurements if present (Mitiq ZNE operates on unitary circuits)
    clean_circuit = circuit.remove_final_measurements(inplace=False)

    # Execute ZNE
    mitigated_value = zne.execute_with_zne(
        circuit=clean_circuit,
        executor=executor,
        factory=factory,
        scale_noise=fold_fn,
        num_to_average=3 if folding_method == "random" else 1,
    )

    # Extract measured values and compute R² from factory
    measured_values = list(factory.get_expectation_values())
    used_scale_factors = list(factory.get_scale_factors())

    # Compute R² (coefficient of determination)
    r_squared = _compute_r_squared(np.array(used_scale_factors), np.array(measured_values))

    elapsed = time.time() - t0
    _logger.info(
        f"[mitiq_zne] Done: mitigated={mitigated_value:.6f}, R²={r_squared:.4f}, "
        f"time={elapsed:.1f}s"
    )

    return MitiqZNEResult(
        extrapolated_value=float(mitigated_value),
        r_squared=r_squared,
        factory_name=factory_name,
        folding_method=folding_method,
        scale_factors=used_scale_factors,
        measured_values=measured_values,
        execution_time_s=elapsed,
    )


def _compute_r_squared(x: np.ndarray, y: np.ndarray) -> float:
    """Compute R² for linear fit of y vs x."""
    if len(x) < 2:
        return 0.0
    # Linear fit
    coeffs = np.polyfit(x, y, 1)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot < 1e-15:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Mitiq CDR — Clifford Data Regression
# ═══════════════════════════════════════════════════════════════════════════


def run_mitiq_cdr(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    backend: Any,
    config: NoisyEstimatorConfig,
    n_training_circuits: int = 10,
    seed_offset: int = 0,
) -> MitiqCDRResult:
    """Run Clifford Data Regression via Mitiq.

    CDR is a learning-based error mitigation method that:
    1. Generates near-Clifford variants of the input circuit
    2. Executes them on BOTH noisy backend and noiseless simulator
    3. Fits a linear model: E_ideal ≈ a * E_noisy + b
    4. Applies the learned correction to the original circuit's noisy result

    Advantages:
    - No noise model characterization required (unlike PEA)
    - Works with ANY noise type (not just depolarizing)
    - Particularly effective when E_noisy/E_ideal is quasi-linear
      (true for TFIM in paramagnetic regime h > h_c)

    CDR overhead: ~2N+1 circuit executions (N training circuits × 2 + original)
    For n_training_circuits=10: ~21 executions = ~150% overhead.

    Reference: Czarnik et al., Quantum 5, 592 (2021)

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to mitigate (measurement-free).
    observable : SparsePauliOp
        Observable to measure.
    backend : BackendV2
        Noisy backend.
    config : NoisyEstimatorConfig
        Shots, seed, optimization_level.
    n_training_circuits : int
        Number of near-Clifford training circuits (default 10).
    seed_offset : int
        Added to config.seed_simulator for independence.

    Returns
    -------
    MitiqCDRResult
        Mitigated value with improvement metrics.
    """
    _check_mitiq()
    from mitiq import cdr

    t0 = time.time()
    _logger.info(f"[mitiq_cdr] Starting: n_training={n_training_circuits}, shots={config.shots}")

    # Build noisy executor
    noisy_executor = make_mitiq_executor(observable, backend, config, seed_offset=seed_offset)

    # Build noiseless executor (for CDR training data)
    noiseless_executor = make_noiseless_executor(observable)

    # Remove measurements if present
    clean_circuit = circuit.remove_final_measurements(inplace=False)

    # Get raw (unmitigated) value first
    raw_value = noisy_executor(clean_circuit)

    # Execute CDR
    mitigated_value = cdr.execute_with_cdr(
        circuit=clean_circuit,
        executor=noisy_executor,
        simulator=noiseless_executor,
        num_training_circuits=n_training_circuits,
    )

    # Compute improvement
    # Note: improvement is relative to raw, towards exact (unknown here)
    # We report the absolute difference as improvement percentage of raw
    if abs(raw_value) > 1e-10:
        improvement_pct = abs(mitigated_value - raw_value) / abs(raw_value) * 100
    else:
        improvement_pct = 0.0

    elapsed = time.time() - t0
    _logger.info(
        f"[mitiq_cdr] Done: raw={raw_value:.6f}, mitigated={mitigated_value:.6f}, "
        f"improvement={improvement_pct:.1f}%, time={elapsed:.1f}s"
    )

    return MitiqCDRResult(
        mitigated_value=float(mitigated_value),
        raw_value=float(raw_value),
        improvement_pct=improvement_pct,
        n_training_circuits=n_training_circuits,
        execution_time_s=elapsed,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: DDD+ZNE Composition — Digital Dynamical Decoupling + ZNE
# ═══════════════════════════════════════════════════════════════════════════


def _get_ddd_rule(rule_name: str) -> Any:
    """Get Mitiq DDD rule by name.

    Parameters
    ----------
    rule_name : str
        One of "xx", "yy", "xyxy".

    Returns
    -------
    mitiq.ddd rule object
    """
    _check_mitiq()
    from mitiq.ddd import rules

    rule_map = {
        "xx": rules.xx,
        "yy": rules.yy,
        "xyxy": rules.xyxy,
    }
    if rule_name not in rule_map:
        raise ValueError(f"Unknown DDD rule: {rule_name!r}. Use one of: {list(rule_map.keys())}")
    return rule_map[rule_name]


def run_mitiq_ddd_zne(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    backend: Any,
    config: NoisyEstimatorConfig,
    ddd_rule: str = "xx",
    scale_factors: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0),
    factory_name: str = "linear",
    folding_method: str = "global",
    seed_offset: int = 0,
) -> MitiqDDDZNEResult:
    """Run DDD+ZNE composition via Mitiq.

    Applies Digital Dynamical Decoupling FIRST (mitigates coherent errors
    during idle periods), THEN applies ZNE (mitigates incoherent errors).

    The composition works because:
    - DDD inserts identity-equivalent gate sequences in idle slots
    - These sequences refocus qubit evolution against T2 decay
    - ZNE then extrapolates residual incoherent noise to zero

    For our HVA circuits on heavy_hex: routing creates idle periods where
    DDD can be effective. For chain_1d (minimal routing), DDD adds less value.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to mitigate (measurement-free).
    observable : SparsePauliOp
        Observable to measure.
    backend : BackendV2
        Noisy backend.
    config : NoisyEstimatorConfig
        Shots, seed, optimization_level.
    ddd_rule : str
        DDD sequence rule: "xx", "yy", or "xyxy".
    scale_factors : tuple[float, ...]
        ZNE noise amplification factors.
    factory_name : str
        ZNE extrapolation factory.
    folding_method : str
        ZNE gate folding method.
    seed_offset : int
        Added to config.seed_simulator.

    Returns
    -------
    MitiqDDDZNEResult
        Combined DDD+ZNE result.
    """
    _check_mitiq()
    from mitiq import ddd, zne

    t0 = time.time()
    _logger.info(
        f"[mitiq_ddd_zne] Starting: ddd_rule={ddd_rule}, factory={factory_name}, "
        f"folding={folding_method}, scale_factors={scale_factors}"
    )

    # Build base executor
    base_executor = make_mitiq_executor(observable, backend, config, seed_offset=seed_offset)

    # Get DDD rule
    rule = _get_ddd_rule(ddd_rule)

    # Build DDD-wrapped executor: applies DDD before execution
    def ddd_executor(circuit: QuantumCircuit) -> float:
        """Apply DDD then execute."""
        ddd_circuit = ddd.insert_ddd_sequences(circuit, rule=rule)
        return base_executor(ddd_circuit)

    # Ensure Mitiq Executor detects float return type (closure annotation fix)
    ddd_executor.__annotations__ = {"circuit": QuantumCircuit, "return": float}

    # Build ZNE factory and folding function
    factory = _build_zne_factory(factory_name, scale_factors)
    fold_fn = _get_folding_function(folding_method)

    # Remove measurements
    clean_circuit = circuit.remove_final_measurements(inplace=False)

    # Execute ZNE with DDD-wrapped executor
    mitigated_value = zne.execute_with_zne(
        circuit=clean_circuit,
        executor=ddd_executor,
        factory=factory,
        scale_noise=fold_fn,
    )

    # Extract factory data
    measured_values = list(factory.get_expectation_values())
    used_scale_factors = list(factory.get_scale_factors())
    r_squared = _compute_r_squared(np.array(used_scale_factors), np.array(measured_values))

    elapsed = time.time() - t0
    _logger.info(
        f"[mitiq_ddd_zne] Done: mitigated={mitigated_value:.6f}, R²={r_squared:.4f}, "
        f"time={elapsed:.1f}s"
    )

    return MitiqDDDZNEResult(
        extrapolated_value=float(mitigated_value),
        r_squared=r_squared,
        ddd_rule=ddd_rule,
        zne_factory=factory_name,
        scale_factors=used_scale_factors,
        measured_values=measured_values,
        execution_time_s=elapsed,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: Mitiq PEC — Probabilistic Error Cancellation (benchmark only)
# ═══════════════════════════════════════════════════════════════════════════


def run_mitiq_pec(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    backend: Any,
    config: NoisyEstimatorConfig,
    noise_level: float = 0.003,
    n_samples: int = 1000,
    seed_offset: int = 0,
) -> MitiqPECResult:
    """Run Probabilistic Error Cancellation via Mitiq (benchmark only).

    PEC provides theoretically optimal error mitigation at the cost of
    exponential sampling overhead. For our circuits (N=10, p=1, 17 CZ gates,
    ε≈0.003 per CZ on Heron R2), the overhead is:
        exp(2 * n_2q * ε) = exp(2 * 17 * 0.003) ≈ 1.11 → ~11% overhead

    This makes PEC surprisingly viable for our short circuits.

    WARNING: PEC requires a noise representation (quasi-probability
    decomposition) of each noisy gate. For simplicity, we use a depolarizing
    model. For hardware, you'd need the actual noise characterization.

    Reference: Temme et al., PRL 119, 180509 (2017)

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to mitigate (measurement-free).
    observable : SparsePauliOp
        Observable to measure.
    backend : BackendV2
        Noisy backend.
    config : NoisyEstimatorConfig
        Shots, seed, optimization_level.
    noise_level : float
        Depolarizing noise probability per gate (for PEC representation).
    n_samples : int
        Number of PEC samples (higher = lower variance).
    seed_offset : int
        Added to config.seed_simulator.

    Returns
    -------
    MitiqPECResult
        PEC mitigated value with overhead metrics.
    """
    _check_mitiq()
    from mitiq import pec

    t0 = time.time()
    _logger.info(f"[mitiq_pec] Starting: noise_level={noise_level}, n_samples={n_samples}")

    # Build executor
    executor = make_mitiq_executor(observable, backend, config, seed_offset=seed_offset)

    # Remove measurements
    clean_circuit = circuit.remove_final_measurements(inplace=False)

    # Get raw value
    raw_value = executor(clean_circuit)

    # For PEC we need OperationRepresentation objects.
    # Use Mitiq's depolarizing representations as approximation.
    # This generates quasi-probability representations assuming depolarizing noise.
    try:
        from mitiq.pec import (
            represent_operations_in_circuit_with_local_depolarizing_noise,
        )

        representations = represent_operations_in_circuit_with_local_depolarizing_noise(
            ideal_circuit=clean_circuit,
            noise_level=noise_level,
        )

        # Compute one-norm (sampling overhead)
        one_norms = [rep.norm for rep in representations]
        overhead_factor = float(np.prod(one_norms) ** 2) if one_norms else 1.0

        # Execute PEC
        mitigated_value = pec.execute_with_pec(
            circuit=clean_circuit,
            executor=executor,
            representations=representations,
            num_samples=n_samples,
            random_state=config.seed_simulator + seed_offset,
        )
    except Exception as e:
        _logger.warning(
            f"[mitiq_pec] PEC failed ({type(e).__name__}: {e}). Returning raw value as fallback."
        )
        mitigated_value = raw_value
        overhead_factor = 1.0

    if abs(raw_value) > 1e-10:
        improvement_pct = abs(mitigated_value - raw_value) / abs(raw_value) * 100
    else:
        improvement_pct = 0.0

    elapsed = time.time() - t0
    _logger.info(
        f"[mitiq_pec] Done: raw={raw_value:.6f}, mitigated={mitigated_value:.6f}, "
        f"overhead={overhead_factor:.2f}×, time={elapsed:.1f}s"
    )

    return MitiqPECResult(
        mitigated_value=float(mitigated_value),
        raw_value=float(raw_value),
        improvement_pct=improvement_pct,
        n_samples=n_samples,
        overhead_factor=overhead_factor,
        execution_time_s=elapsed,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6/7: Strategy Comparator — run all methods and produce ranking
# ═══════════════════════════════════════════════════════════════════════════


def compare_mitigation_strategies(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    backend: Any,
    config: NoisyEstimatorConfig,
    exact_energy: float,
    gap: float,
    h_value: float = 0.0,
    strategies: list[str] | None = None,
    seed_offset: int = 0,
) -> MitiqComparisonResult:
    """Run multiple mitigation strategies and produce a ranked comparison.

    Executes each requested strategy on the same circuit and produces
    a ΔE/gap ranking for thesis table material. Includes both our native
    methods and Mitiq methods.

    Available strategies:
    - "raw": No mitigation (baseline)
    - "mitiq_zne_linear": Mitiq ZNE with linear factory + random folding
    - "mitiq_zne_richardson": Mitiq ZNE with Richardson factory
    - "mitiq_cdr": Mitiq CDR (10 near-Clifford circuits)
    - "mitiq_ddd_zne": Mitiq DDD(XX) + ZNE(linear)
    - "mitiq_pec": Mitiq PEC (benchmark, 1000 samples)
    - "native_gf_zne": Our gate-folding ZNE (for comparison)
    - "native_pea_zne": Our PEA-ZNE (for comparison, requires qiskit-aer)

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to mitigate (bound parameters, no measurements).
    observable : SparsePauliOp
        Observable (NOT layout-mapped).
    backend : BackendV2
        Noisy backend.
    config : NoisyEstimatorConfig
        Execution configuration.
    exact_energy : float
        Exact ground state energy (from Phase 1).
    gap : float
        Energy gap (for ΔE/gap computation).
    h_value : float
        Transverse field value (for metadata).
    strategies : list[str] | None
        Which strategies to run. None = all available.
    seed_offset : int
        Base seed offset.

    Returns
    -------
    MitiqComparisonResult
        Ranked comparison of all methods.
    """
    t0 = time.time()

    if strategies is None:
        strategies = [
            "raw",
            "mitiq_zne_linear",
            "mitiq_zne_richardson",
            "mitiq_cdr",
            "mitiq_ddd_zne",
        ]

    _logger.info(f"[compare] Starting comparison: h={h_value}, strategies={strategies}")

    results: dict[str, float] = {}
    delta_e_gaps: dict[str, float] = {}

    # Raw baseline
    if "raw" in strategies:
        executor = make_mitiq_executor(observable, backend, config, seed_offset=seed_offset)
        clean = circuit.remove_final_measurements(inplace=False)
        raw_energy = executor(clean)
        results["raw"] = raw_energy
        delta_e_gaps["raw"] = abs(raw_energy - exact_energy) / abs(gap) if abs(gap) > 1e-15 else 0.0
    else:
        # Still need raw for comparison
        executor = make_mitiq_executor(observable, backend, config, seed_offset=seed_offset)
        clean = circuit.remove_final_measurements(inplace=False)
        raw_energy = executor(clean)

    # Mitiq ZNE (linear)
    if "mitiq_zne_linear" in strategies:
        try:
            res = run_mitiq_zne(
                circuit,
                observable,
                backend,
                config,
                factory_name="linear",
                folding_method="random",
                seed_offset=seed_offset + 100,
            )
            results["mitiq_zne_linear"] = res.extrapolated_value
            delta_e_gaps["mitiq_zne_linear"] = (
                abs(res.extrapolated_value - exact_energy) / abs(gap) if abs(gap) > 1e-15 else 0.0
            )
        except Exception as e:
            _logger.warning(f"[compare] mitiq_zne_linear failed: {e}")

    # Mitiq ZNE (Richardson)
    if "mitiq_zne_richardson" in strategies:
        try:
            res = run_mitiq_zne(
                circuit,
                observable,
                backend,
                config,
                factory_name="richardson",
                folding_method="random",
                seed_offset=seed_offset + 200,
            )
            results["mitiq_zne_richardson"] = res.extrapolated_value
            delta_e_gaps["mitiq_zne_richardson"] = (
                abs(res.extrapolated_value - exact_energy) / abs(gap) if abs(gap) > 1e-15 else 0.0
            )
        except Exception as e:
            _logger.warning(f"[compare] mitiq_zne_richardson failed: {e}")

    # Mitiq CDR
    if "mitiq_cdr" in strategies:
        try:
            res = run_mitiq_cdr(
                circuit,
                observable,
                backend,
                config,
                n_training_circuits=10,
                seed_offset=seed_offset + 300,
            )
            results["mitiq_cdr"] = res.mitigated_value
            delta_e_gaps["mitiq_cdr"] = (
                abs(res.mitigated_value - exact_energy) / abs(gap) if abs(gap) > 1e-15 else 0.0
            )
        except Exception as e:
            _logger.warning(f"[compare] mitiq_cdr failed: {e}")

    # Mitiq DDD+ZNE
    if "mitiq_ddd_zne" in strategies:
        try:
            res = run_mitiq_ddd_zne(
                circuit,
                observable,
                backend,
                config,
                ddd_rule="xx",
                factory_name="linear",
                seed_offset=seed_offset + 400,
            )
            results["mitiq_ddd_zne"] = res.extrapolated_value
            delta_e_gaps["mitiq_ddd_zne"] = (
                abs(res.extrapolated_value - exact_energy) / abs(gap) if abs(gap) > 1e-15 else 0.0
            )
        except Exception as e:
            _logger.warning(f"[compare] mitiq_ddd_zne failed: {e}")

    # Mitiq PEC (expensive — only if requested)
    if "mitiq_pec" in strategies:
        try:
            res = run_mitiq_pec(
                circuit,
                observable,
                backend,
                config,
                n_samples=1000,
                seed_offset=seed_offset + 500,
            )
            results["mitiq_pec"] = res.mitigated_value
            delta_e_gaps["mitiq_pec"] = (
                abs(res.mitigated_value - exact_energy) / abs(gap) if abs(gap) > 1e-15 else 0.0
            )
        except Exception as e:
            _logger.warning(f"[compare] mitiq_pec failed: {e}")

    # Native GF-ZNE (our implementation, for comparison)
    if "native_gf_zne" in strategies:
        try:
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

            from qmbp_simulation.execution.noisy_utils import (
                run_gate_folding_zne,
            )

            # Need to transpile circuit for our native ZNE
            pm = generate_preset_pass_manager(
                backend=backend,
                optimization_level=config.optimization_level,
            )
            clean = circuit.remove_final_measurements(inplace=False)
            transpiled = pm.run(clean)
            mapped_obs = observable.apply_layout(transpiled.layout)

            gf_res = run_gate_folding_zne(
                transpiled,
                mapped_obs,
                backend,
                config,
                seed_offset=seed_offset + 600,
            )
            results["native_gf_zne"] = gf_res.extrapolated_value
            delta_e_gaps["native_gf_zne"] = (
                abs(gf_res.extrapolated_value - exact_energy) / abs(gap)
                if abs(gap) > 1e-15
                else 0.0
            )
        except Exception as e:
            _logger.warning(f"[compare] native_gf_zne failed: {e}")

    # Native PEA-ZNE (our implementation, requires qiskit-aer)
    if "native_pea_zne" in strategies:
        try:
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

            from qmbp_simulation.execution.noisy_utils import run_pea_zne

            pm = generate_preset_pass_manager(
                backend=backend,
                optimization_level=config.optimization_level,
            )
            clean = circuit.remove_final_measurements(inplace=False)
            transpiled = pm.run(clean)
            mapped_obs = observable.apply_layout(transpiled.layout)

            pea_res = run_pea_zne(
                transpiled,
                mapped_obs,
                backend,
                config,
                seed_offset=seed_offset + 700,
            )
            results["native_pea_zne"] = pea_res.extrapolated_value
            delta_e_gaps["native_pea_zne"] = (
                abs(pea_res.extrapolated_value - exact_energy) / abs(gap)
                if abs(gap) > 1e-15
                else 0.0
            )
        except Exception as e:
            _logger.warning(f"[compare] native_pea_zne failed: {e}")

    # Rank by ΔE/gap (lower is better)
    rankings = sorted(delta_e_gaps.keys(), key=lambda k: delta_e_gaps[k])
    best_method = rankings[0] if rankings else ""
    best_delta = delta_e_gaps.get(best_method, 0.0)

    elapsed = time.time() - t0
    _logger.info(
        f"[compare] Done: best={best_method} (ΔE/gap={best_delta:.4f}), "
        f"rankings={rankings}, time={elapsed:.1f}s"
    )

    return MitiqComparisonResult(
        h_value=h_value,
        e_exact=exact_energy,
        gap=gap,
        raw_energy=raw_energy,
        results=results,
        delta_e_gaps=delta_e_gaps,
        rankings=rankings,
        best_method=best_method,
        best_delta_e_gap=best_delta,
        execution_time_s=elapsed,
    )
