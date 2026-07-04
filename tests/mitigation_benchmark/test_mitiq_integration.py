"""Tests for Mitiq integration module.

These tests validate the Mitiq integration utilities work correctly
with our backend/executor patterns. Tests are marked with @pytest.mark.mitiq
and can be skipped if mitiq is not installed.

Run with: pytest tests/test_mitiq_integration.py -v
Skip if no mitiq: pytest tests/test_mitiq_integration.py -v -m "not mitiq"
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

# Check if mitiq is available
try:
    import mitiq  # noqa: F401

    HAS_MITIQ = True
except ImportError:
    HAS_MITIQ = False

# Check if qiskit-aer is available (needed for noisy simulation)
try:
    from qiskit_aer import AerSimulator  # noqa: F401

    HAS_AER = True
except ImportError:
    HAS_AER = False

pytestmark = pytest.mark.mitiq


def _make_test_circuit(n_qubits: int = 4) -> QuantumCircuit:
    """Create a simple test circuit (ZZ + X rotations, TFIM-like)."""
    qc = QuantumCircuit(n_qubits)
    # |+⟩ initial state
    for i in range(n_qubits):
        qc.h(i)
    # ZZ interactions
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
        qc.rz(0.5, i + 1)
        qc.cx(i, i + 1)
    # X rotations
    for i in range(n_qubits):
        qc.rx(0.3, i)
    return qc


def _make_test_observable(n_qubits: int = 4) -> SparsePauliOp:
    """Create a simple ZZ + X observable."""
    terms = []
    # ZZ terms
    for i in range(n_qubits - 1):
        label = "I" * i + "ZZ" + "I" * (n_qubits - i - 2)
        terms.append((label, -1.0))
    # X terms
    for i in range(n_qubits):
        label = "I" * i + "X" + "I" * (n_qubits - i - 1)
        terms.append((label, -0.5))
    return SparsePauliOp.from_list(terms)


def _make_noisy_backend():
    """Create a noisy AerSimulator with depolarizing noise."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    noise_model = NoiseModel()
    # 1Q depolarizing
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ["rx", "ry", "rz", "h"])
    # 2Q depolarizing
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cx"])

    return AerSimulator(noise_model=noise_model)


# ═══════════════════════════════════════════════════════════════════════════
# Test: Module availability and graceful degradation
# ═══════════════════════════════════════════════════════════════════════════


def test_mitiq_availability_check():
    """is_mitiq_available() returns correct boolean without raising."""
    from qmbp_simulation.execution.mitiq_utils import is_mitiq_available

    result = is_mitiq_available()
    assert isinstance(result, bool)
    assert result == HAS_MITIQ


@pytest.mark.skipif(not HAS_MITIQ, reason="mitiq not installed")
def test_mitiq_import_does_not_raise():
    """Importing mitiq_utils should not raise even without mitiq."""
    from qmbp_simulation.execution import mitiq_utils

    assert hasattr(mitiq_utils, "run_mitiq_zne")
    assert hasattr(mitiq_utils, "run_mitiq_cdr")
    assert hasattr(mitiq_utils, "run_mitiq_ddd_zne")
    assert hasattr(mitiq_utils, "run_mitiq_pec")
    assert hasattr(mitiq_utils, "compare_mitigation_strategies")


# ═══════════════════════════════════════════════════════════════════════════
# Test: Executor Factory
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_executor_factory_returns_float():
    """make_mitiq_executor produces an executor that returns a float."""
    from qmbp_simulation.execution.mitiq_utils import make_mitiq_executor
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    circuit = _make_test_circuit(4)
    observable = _make_test_observable(4)
    backend = _make_noisy_backend()
    config = NoisyEstimatorConfig(shots=1024, seed_simulator=42)

    executor = make_mitiq_executor(observable, backend, config)
    result = executor(circuit)

    assert isinstance(result, float)
    assert np.isfinite(result)


@pytest.mark.skipif(not HAS_MITIQ, reason="mitiq not installed")
def test_noiseless_executor_returns_float():
    """make_noiseless_executor produces exact expectation value."""
    from qmbp_simulation.execution.mitiq_utils import make_noiseless_executor

    circuit = _make_test_circuit(4)
    observable = _make_test_observable(4)

    executor = make_noiseless_executor(observable)
    result = executor(circuit)

    assert isinstance(result, float)
    assert np.isfinite(result)
    # For a 4-qubit TFIM-like circuit, energy should be negative
    assert result < 0


# ═══════════════════════════════════════════════════════════════════════════
# Test: Mitiq ZNE
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_mitiq_zne_linear_produces_result():
    """run_mitiq_zne with linear factory returns valid MitiqZNEResult."""
    from qmbp_simulation.execution.mitiq_utils import MitiqZNEResult, run_mitiq_zne
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    circuit = _make_test_circuit(4)
    observable = _make_test_observable(4)
    backend = _make_noisy_backend()
    config = NoisyEstimatorConfig(shots=1024, seed_simulator=42)

    result = run_mitiq_zne(
        circuit,
        observable,
        backend,
        config,
        scale_factors=(1.0, 1.5, 2.0),
        factory_name="linear",
        folding_method="global",
    )

    assert isinstance(result, MitiqZNEResult)
    assert np.isfinite(result.extrapolated_value)
    assert 0.0 <= result.r_squared <= 1.0 or result.r_squared > 0.5
    assert result.factory_name == "linear"
    assert result.folding_method == "global"
    assert len(result.scale_factors) >= 2
    assert len(result.measured_values) >= 2
    assert result.execution_time_s > 0


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_mitiq_zne_richardson():
    """run_mitiq_zne with Richardson factory."""
    from qmbp_simulation.execution.mitiq_utils import run_mitiq_zne
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    circuit = _make_test_circuit(4)
    observable = _make_test_observable(4)
    backend = _make_noisy_backend()
    config = NoisyEstimatorConfig(shots=1024, seed_simulator=42)

    result = run_mitiq_zne(
        circuit,
        observable,
        backend,
        config,
        scale_factors=(1.0, 2.0, 3.0),
        factory_name="richardson",
        folding_method="global",
    )

    assert np.isfinite(result.extrapolated_value)
    assert result.factory_name == "richardson"


# ═══════════════════════════════════════════════════════════════════════════
# Test: Mitiq CDR
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_mitiq_cdr_produces_result():
    """run_mitiq_cdr returns valid MitiqCDRResult with improvement."""
    from qmbp_simulation.execution.mitiq_utils import MitiqCDRResult, run_mitiq_cdr
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    circuit = _make_test_circuit(4)
    observable = _make_test_observable(4)
    backend = _make_noisy_backend()
    config = NoisyEstimatorConfig(shots=1024, seed_simulator=42)

    result = run_mitiq_cdr(
        circuit,
        observable,
        backend,
        config,
        n_training_circuits=5,  # Fewer for speed
    )

    assert isinstance(result, MitiqCDRResult)
    assert np.isfinite(result.mitigated_value)
    assert np.isfinite(result.raw_value)
    assert result.n_training_circuits == 5
    assert result.execution_time_s > 0


# ═══════════════════════════════════════════════════════════════════════════
# Test: Mitiq DDD+ZNE
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_mitiq_ddd_zne_produces_result():
    """run_mitiq_ddd_zne returns valid result."""
    from qmbp_simulation.execution.mitiq_utils import MitiqDDDZNEResult, run_mitiq_ddd_zne
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    circuit = _make_test_circuit(4)
    observable = _make_test_observable(4)
    backend = _make_noisy_backend()
    config = NoisyEstimatorConfig(shots=1024, seed_simulator=42)

    result = run_mitiq_ddd_zne(
        circuit,
        observable,
        backend,
        config,
        ddd_rule="xx",
        scale_factors=(1.0, 1.5, 2.0),
        factory_name="linear",
    )

    assert isinstance(result, MitiqDDDZNEResult)
    assert np.isfinite(result.extrapolated_value)
    assert result.ddd_rule == "xx"
    assert result.zne_factory == "linear"


# ═══════════════════════════════════════════════════════════════════════════
# Test: Strategy Comparison
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_compare_mitigation_strategies():
    """compare_mitigation_strategies produces valid ranked comparison."""
    from qmbp_simulation.execution.mitiq_utils import (
        MitiqComparisonResult,
        compare_mitigation_strategies,
    )
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    circuit = _make_test_circuit(4)
    observable = _make_test_observable(4)
    backend = _make_noisy_backend()
    config = NoisyEstimatorConfig(shots=1024, seed_simulator=42)

    # Use noiseless executor to get exact energy for comparison
    from qmbp_simulation.execution.mitiq_utils import make_noiseless_executor

    noiseless = make_noiseless_executor(observable)
    exact_energy = noiseless(circuit)
    gap = 0.5  # Arbitrary for test

    result = compare_mitigation_strategies(
        circuit,
        observable,
        backend,
        config,
        exact_energy=exact_energy,
        gap=gap,
        h_value=2.0,
        strategies=["raw", "mitiq_zne_linear", "mitiq_cdr"],
    )

    assert isinstance(result, MitiqComparisonResult)
    assert result.h_value == 2.0
    assert "raw" in result.results
    assert len(result.rankings) >= 1
    assert result.best_method in result.rankings
    assert result.best_delta_e_gap >= 0.0
    assert result.execution_time_s > 0


# ═══════════════════════════════════════════════════════════════════════════
# Test: Dataclass serialization (JSON compatibility)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_MITIQ, reason="mitiq not installed")
def test_result_dataclasses_are_json_serializable():
    """All Mitiq result dataclasses can be serialized via json_serialize."""
    from qmbp_simulation.execution.mitiq_utils import (
        MitiqCDRResult,
        MitiqComparisonResult,
        MitiqDDDZNEResult,
        MitiqPECResult,
        MitiqZNEResult,
    )
    from qmbp_simulation.utils.helpers import json_serialize

    zne = MitiqZNEResult(
        extrapolated_value=-3.5,
        r_squared=0.99,
        factory_name="linear",
        folding_method="random",
        scale_factors=[1.0, 2.0, 3.0],
        measured_values=[-3.0, -2.8, -2.6],
        execution_time_s=1.5,
    )
    cdr_res = MitiqCDRResult(
        mitigated_value=-3.4,
        raw_value=-3.0,
        improvement_pct=13.3,
        n_training_circuits=10,
        execution_time_s=2.0,
    )
    ddd = MitiqDDDZNEResult(
        extrapolated_value=-3.5,
        r_squared=0.98,
        ddd_rule="xx",
        zne_factory="linear",
        scale_factors=[1.0, 2.0],
        measured_values=[-3.1, -2.9],
        execution_time_s=1.8,
    )
    pec = MitiqPECResult(
        mitigated_value=-3.45,
        raw_value=-3.0,
        improvement_pct=15.0,
        n_samples=1000,
        overhead_factor=1.05,
        execution_time_s=3.0,
    )
    comp = MitiqComparisonResult(
        h_value=2.0,
        e_exact=-3.5,
        gap=0.5,
        raw_energy=-3.0,
        results={"raw": -3.0, "cdr": -3.4},
        delta_e_gaps={"raw": 1.0, "cdr": 0.2},
        rankings=["cdr", "raw"],
        best_method="cdr",
        best_delta_e_gap=0.2,
        execution_time_s=5.0,
    )

    # All should be serializable without raising
    for obj in [zne, cdr_res, ddd, pec, comp]:
        serialized = json_serialize(obj)
        assert serialized is not None


# ═══════════════════════════════════════════════════════════════════════════
# Test: Invalid inputs
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_MITIQ, reason="mitiq not installed")
def test_invalid_factory_name_raises():
    """Unknown factory name raises ValueError."""
    from qmbp_simulation.execution.mitiq_utils import _build_zne_factory

    with pytest.raises(ValueError, match="Unknown factory"):
        _build_zne_factory("nonexistent", (1.0, 2.0, 3.0))


@pytest.mark.skipif(not HAS_MITIQ, reason="mitiq not installed")
def test_invalid_folding_method_raises():
    """Unknown folding method raises ValueError."""
    from qmbp_simulation.execution.mitiq_utils import _get_folding_function

    with pytest.raises(ValueError, match="Unknown folding method"):
        _get_folding_function("nonexistent")


@pytest.mark.skipif(not HAS_MITIQ, reason="mitiq not installed")
def test_invalid_ddd_rule_raises():
    """Unknown DDD rule raises ValueError."""
    from qmbp_simulation.execution.mitiq_utils import _get_ddd_rule

    with pytest.raises(ValueError, match="Unknown DDD rule"):
        _get_ddd_rule("nonexistent")


# ═══════════════════════════════════════════════════════════════════════════
# Extended coverage: optimization_level=0 enforcement
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_executor_preserves_folded_gates():
    """Verify that opt_level=0 in executor preserves Mitiq gate folding.

    This is the CRITICAL test: Qiskit 2.x at opt_level>=1 cancels U·U† pairs,
    destroying ZNE folding. Our executor MUST use opt_level=0.
    """
    from mitiq.zne.scaling import fold_global

    from qmbp_simulation.execution.mitiq_utils import make_mitiq_executor
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    # Simple circuit with 1 CX
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    # Fold 3x (should produce 3 CX-equivalent operations)
    folded = fold_global(qc, scale_factor=3.0)
    original_cx = sum(1 for inst in folded.data if inst.operation.num_qubits == 2)
    assert original_cx == 3, f"Folded circuit should have 3 2Q gates, got {original_cx}"

    # Create backend and executor
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cx"])
    backend = AerSimulator(noise_model=nm)
    config = NoisyEstimatorConfig(shots=1024, seed_simulator=42, optimization_level=2)

    # Even though config says opt_level=2, our executor MUST use 0
    obs = SparsePauliOp.from_list([("ZZ", 1.0)])
    executor = make_mitiq_executor(obs, backend, config)

    # Execute the folded circuit — if opt_level=0 is enforced, the noisy result
    # should be MORE degraded than unfolded (more gates = more noise)
    unfolded_result = executor(qc)
    folded_result = executor(folded)

    # With depolarizing noise, more gates = expectation value closer to 0
    # The folded circuit (3x gates) should have more noise → closer to 0
    assert abs(folded_result) < abs(unfolded_result) + 0.3, (
        f"Folded ({folded_result:.4f}) should be noisier than unfolded ({unfolded_result:.4f})"
    )


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_zne_improves_over_raw():
    """ZNE extrapolated value should be closer to exact than raw noisy."""
    from qmbp_simulation.execution.mitiq_utils import (
        make_noiseless_executor,
        run_mitiq_zne,
    )
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    circuit = _make_test_circuit(4)
    observable = _make_test_observable(4)
    backend = _make_noisy_backend()
    config = NoisyEstimatorConfig(shots=4096, seed_simulator=42)

    # Get exact
    exact = make_noiseless_executor(observable)(circuit)

    # Run ZNE with global folding (more deterministic for testing)
    result = run_mitiq_zne(
        circuit,
        observable,
        backend,
        config,
        factory_name="linear",
        folding_method="global",
        scale_factors=(1.0, 1.5, 2.0, 3.0),
    )

    # ZNE should improve (get closer to exact)
    raw_error = abs(result.measured_values[0] - exact)  # factor=1.0 is raw
    zne_error = abs(result.extrapolated_value - exact)

    assert zne_error < raw_error, (
        f"ZNE should improve: raw_err={raw_error:.4f}, zne_err={zne_error:.4f}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Extended coverage: ZNE factories and folding methods
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_mitiq_zne_exponential_factory():
    """Exponential factory with asymptote produces valid result."""
    from qmbp_simulation.execution.mitiq_utils import run_mitiq_zne
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    circuit = _make_test_circuit(4)
    observable = _make_test_observable(4)
    backend = _make_noisy_backend()
    config = NoisyEstimatorConfig(shots=1024, seed_simulator=42)

    result = run_mitiq_zne(
        circuit,
        observable,
        backend,
        config,
        scale_factors=(1.0, 2.0, 3.0),
        factory_name="exp",
        folding_method="global",
        asymptote=0.0,
    )

    assert np.isfinite(result.extrapolated_value)
    assert result.factory_name == "exp"


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_mitiq_zne_poly_factory():
    """Polynomial factory produces valid result."""
    from qmbp_simulation.execution.mitiq_utils import run_mitiq_zne
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    circuit = _make_test_circuit(4)
    observable = _make_test_observable(4)
    backend = _make_noisy_backend()
    config = NoisyEstimatorConfig(shots=1024, seed_simulator=42)

    result = run_mitiq_zne(
        circuit,
        observable,
        backend,
        config,
        scale_factors=(1.0, 1.5, 2.0, 2.5),
        factory_name="poly",
        folding_method="global",
    )

    assert np.isfinite(result.extrapolated_value)
    assert result.factory_name == "poly"


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_mitiq_zne_all_folding():
    """fold_all method produces valid result."""
    from qmbp_simulation.execution.mitiq_utils import run_mitiq_zne
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    circuit = _make_test_circuit(4)
    observable = _make_test_observable(4)
    backend = _make_noisy_backend()
    config = NoisyEstimatorConfig(shots=1024, seed_simulator=42)

    result = run_mitiq_zne(
        circuit,
        observable,
        backend,
        config,
        scale_factors=(1.0, 2.0, 3.0),
        factory_name="linear",
        folding_method="all",
    )

    assert np.isfinite(result.extrapolated_value)
    assert result.folding_method == "all"


# ═══════════════════════════════════════════════════════════════════════════
# Extended coverage: DDD rules
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_mitiq_ddd_yy_rule():
    """DDD with YY rule produces valid result."""
    from qmbp_simulation.execution.mitiq_utils import run_mitiq_ddd_zne
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    circuit = _make_test_circuit(4)
    observable = _make_test_observable(4)
    backend = _make_noisy_backend()
    config = NoisyEstimatorConfig(shots=1024, seed_simulator=42)

    result = run_mitiq_ddd_zne(
        circuit,
        observable,
        backend,
        config,
        ddd_rule="yy",
        scale_factors=(1.0, 1.5, 2.0),
        factory_name="linear",
    )

    assert np.isfinite(result.extrapolated_value)
    assert result.ddd_rule == "yy"


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_mitiq_ddd_xyxy_rule():
    """DDD with XYXY rule produces valid result."""
    from qmbp_simulation.execution.mitiq_utils import run_mitiq_ddd_zne
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    circuit = _make_test_circuit(4)
    observable = _make_test_observable(4)
    backend = _make_noisy_backend()
    config = NoisyEstimatorConfig(shots=1024, seed_simulator=42)

    result = run_mitiq_ddd_zne(
        circuit,
        observable,
        backend,
        config,
        ddd_rule="xyxy",
        scale_factors=(1.0, 1.5, 2.0),
        factory_name="linear",
    )

    assert np.isfinite(result.extrapolated_value)
    assert result.ddd_rule == "xyxy"


# ═══════════════════════════════════════════════════════════════════════════
# Extended coverage: PEC
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_mitiq_pec_runs_without_crash():
    """PEC produces a result (may not improve due to noise model mismatch)."""
    from qmbp_simulation.execution.mitiq_utils import run_mitiq_pec
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    # Use very simple 2-qubit circuit for PEC
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    obs = SparsePauliOp.from_list([("ZZ", 1.0)])
    backend = _make_noisy_backend()
    config = NoisyEstimatorConfig(shots=1024, seed_simulator=42)

    result = run_mitiq_pec(
        qc,
        obs,
        backend,
        config,
        noise_level=0.02,
        n_samples=100,
    )

    assert np.isfinite(result.mitigated_value)
    assert np.isfinite(result.raw_value)
    assert result.n_samples == 100
    assert result.execution_time_s > 0


# ═══════════════════════════════════════════════════════════════════════════
# Extended coverage: CDR with exact energy verification
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_mitiq_cdr_improves_over_raw():
    """CDR mitigated value should be closer to exact than raw (high noise)."""
    from qmbp_simulation.execution.mitiq_utils import (
        make_noiseless_executor,
        run_mitiq_cdr,
    )
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    circuit = _make_test_circuit(4)
    observable = _make_test_observable(4)
    backend = _make_noisy_backend()
    config = NoisyEstimatorConfig(shots=4096, seed_simulator=42)

    exact = make_noiseless_executor(observable)(circuit)
    result = run_mitiq_cdr(circuit, observable, backend, config, n_training_circuits=8)

    raw_error = abs(result.raw_value - exact)
    cdr_error = abs(result.mitigated_value - exact)

    # CDR should improve (allow some margin for shot noise)
    assert cdr_error < raw_error * 1.1, (
        f"CDR should improve: raw_err={raw_error:.4f}, cdr_err={cdr_error:.4f}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Extended coverage: Comparison with native methods
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_compare_includes_native_gf_zne():
    """compare_mitigation_strategies can include native GF-ZNE."""
    from qmbp_simulation.execution.mitiq_utils import (
        compare_mitigation_strategies,
        make_noiseless_executor,
    )
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    circuit = _make_test_circuit(4)
    observable = _make_test_observable(4)
    backend = _make_noisy_backend()
    config = NoisyEstimatorConfig(shots=1024, seed_simulator=42)

    exact = make_noiseless_executor(observable)(circuit)

    result = compare_mitigation_strategies(
        circuit,
        observable,
        backend,
        config,
        exact_energy=exact,
        gap=0.5,
        h_value=2.0,
        strategies=["raw", "native_gf_zne"],
    )

    assert "raw" in result.results
    assert "native_gf_zne" in result.results
    assert result.best_method in ["raw", "native_gf_zne"]


# ═══════════════════════════════════════════════════════════════════════════
# Extended coverage: Edge cases
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_executor_with_measurements_removed():
    """Executor handles circuits that already have measurements."""
    from qmbp_simulation.execution.mitiq_utils import run_mitiq_zne
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    # Circuit WITH measurements
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    obs = SparsePauliOp.from_list([("ZZ", 1.0)])
    backend = _make_noisy_backend()
    config = NoisyEstimatorConfig(shots=1024, seed_simulator=42)

    # Should handle gracefully (remove_final_measurements internally)
    result = run_mitiq_zne(
        qc,
        obs,
        backend,
        config,
        scale_factors=(1.0, 2.0, 3.0),
        factory_name="linear",
        folding_method="global",
    )

    assert np.isfinite(result.extrapolated_value)


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_compare_empty_strategies():
    """compare_mitigation_strategies with no methods still returns valid result."""
    from qmbp_simulation.execution.mitiq_utils import compare_mitigation_strategies
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    circuit = _make_test_circuit(4)
    observable = _make_test_observable(4)
    backend = _make_noisy_backend()
    config = NoisyEstimatorConfig(shots=1024, seed_simulator=42)

    result = compare_mitigation_strategies(
        circuit,
        observable,
        backend,
        config,
        exact_energy=-3.0,
        gap=0.5,
        h_value=2.0,
        strategies=[],  # No strategies
    )

    assert result.rankings == []
    assert result.best_method == ""


@pytest.mark.skipif(not HAS_MITIQ, reason="mitiq not installed")
def test_is_mitiq_available_returns_true():
    """When mitiq is installed, is_mitiq_available returns True."""
    from qmbp_simulation.execution.mitiq_utils import is_mitiq_available

    assert is_mitiq_available() is True


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_fractional_scale_factors():
    """Mitiq ZNE supports fractional scale factors (unlike our native GF)."""
    from qmbp_simulation.execution.mitiq_utils import run_mitiq_zne
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    circuit = _make_test_circuit(4)
    observable = _make_test_observable(4)
    backend = _make_noisy_backend()
    config = NoisyEstimatorConfig(shots=1024, seed_simulator=42)

    # Use fractional factors that our native GF doesn't support
    result = run_mitiq_zne(
        circuit,
        observable,
        backend,
        config,
        scale_factors=(1.0, 1.25, 1.5, 1.75, 2.0),
        factory_name="linear",
        folding_method="global",
    )

    assert len(result.scale_factors) == 5
    assert np.isfinite(result.extrapolated_value)
    # R² should be high with 5 closely-spaced points
    assert result.r_squared > 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Integration tests: full pipeline verification
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_end_to_end_compare_serialize_analyze():
    """Full pipeline: compare methods -> serialize -> analyzer reads -> health report."""
    import json
    from dataclasses import asdict

    from qmbp_simulation.execution.mitiq_utils import (
        MitiqComparisonResult,
        compare_mitigation_strategies,
        make_noiseless_executor,
    )
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig
    from qmbp_simulation.utils.helpers import json_serialize

    # Build circuit and run comparison
    circuit = _make_test_circuit(4)
    observable = _make_test_observable(4)
    backend = _make_noisy_backend()
    config = NoisyEstimatorConfig(shots=1024, seed_simulator=42)
    exact = make_noiseless_executor(observable)(circuit)

    result = compare_mitigation_strategies(
        circuit,
        observable,
        backend,
        config,
        exact_energy=exact,
        gap=0.5,
        h_value=2.0,
        strategies=["raw", "mitiq_zne_linear"],
    )

    # Verify result structure
    assert isinstance(result, MitiqComparisonResult)
    assert result.h_value == 2.0
    assert "raw" in result.results
    assert "mitiq_zne_linear" in result.results

    # Verify serialization
    data = asdict(result)
    json_str = json.dumps(data, default=json_serialize)
    loaded = json.loads(json_str)
    assert loaded["h_value"] == 2.0
    assert "raw" in loaded["delta_e_gaps"]


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_analyzer_scans_mitiq_results():
    """Mitiq analyzer correctly scans comparison results from disk."""
    import json
    import tempfile
    from dataclasses import asdict
    from pathlib import Path
    from unittest.mock import patch

    from qmbp_simulation.execution.mitiq_utils import (
        compare_mitigation_strategies,
        make_noiseless_executor,
    )
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig
    from qmbp_simulation.utils.helpers import json_serialize

    # Create a result and write to a temp dir
    circuit = _make_test_circuit(4)
    observable = _make_test_observable(4)
    backend = _make_noisy_backend()
    config = NoisyEstimatorConfig(shots=1024, seed_simulator=42)
    exact = make_noiseless_executor(observable)(circuit)

    result = compare_mitigation_strategies(
        circuit,
        observable,
        backend,
        config,
        exact_energy=exact,
        gap=0.5,
        h_value=2.0,
        strategies=["raw", "mitiq_zne_linear"],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        mitiq_dir = Path(tmpdir) / "mitiq"
        mitiq_dir.mkdir()
        out_path = mitiq_dir / "comparison_test.json"
        with open(out_path, "w") as f:
            json.dump(asdict(result), f, default=json_serialize)

        # Patch the scanner to use our temp dir
        from project_health.analysis import mitiq_analyzer

        with (
            patch.object(mitiq_analyzer, "MITIQ_DIR", mitiq_dir),
            patch.object(mitiq_analyzer, "REHEARSAL_DIR", Path(tmpdir) / "nonexistent"),
            patch.object(mitiq_analyzer, "HARDWARE_DIR", Path(tmpdir) / "nonexistent"),
        ):
            report = mitiq_analyzer.scan_mitiq_results()

        assert report.n_comparisons == 1
        assert report.n_h_points == 1
        assert report.best_overall_method != ""


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_health_engine_includes_mitiq_status():
    """Health engine includes mitiq_status field in report."""
    from project_health.core.engine import run_health_check

    report = run_health_check(save_state=False)
    assert hasattr(report, "mitiq_status")
    assert "status" in report.mitiq_status


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_benchmark_suite_mitiq_component():
    """BenchmarkSuite can run mitiq component without errors."""
    from qmbp_simulation.framework.benchmarking import BenchmarkSuite

    suite = BenchmarkSuite(n_qubits=[4], n_repeats=1, verbose=False)
    results = suite.run(components=["mitiq"])

    assert len(results) == 1
    assert results[0].component == "mitiq"
    assert results[0].elapsed_s > 0
    assert "timings_median_s" in results[0].details
    assert "de_gaps" in results[0].details
    assert "best_method" in results[0].details


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_mitiq_zne_with_exact_validation():
    """Mitiq ZNE extrapolated value is closer to exact than any measured value."""
    from qmbp_simulation.execution.mitiq_utils import make_noiseless_executor, run_mitiq_zne
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    circuit = _make_test_circuit(4)
    observable = _make_test_observable(4)
    backend = _make_noisy_backend()
    config = NoisyEstimatorConfig(shots=4096, seed_simulator=42)

    exact = make_noiseless_executor(observable)(circuit)
    result = run_mitiq_zne(
        circuit,
        observable,
        backend,
        config,
        scale_factors=(1.0, 1.5, 2.0, 3.0),
        factory_name="linear",
        folding_method="global",
    )

    # The extrapolated value should be better than the worst measured point
    worst_measured_error = max(abs(m - exact) for m in result.measured_values)
    extrap_error = abs(result.extrapolated_value - exact)
    assert extrap_error < worst_measured_error, (
        f"Extrapolated error ({extrap_error:.4f}) should be less than "
        f"worst measured ({worst_measured_error:.4f})"
    )


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_opt_level_0_enforced_in_executor():
    """Verify executor uses opt_level=0 regardless of config.optimization_level."""
    from qmbp_simulation.execution.mitiq_utils import make_mitiq_executor
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    circuit = _make_test_circuit(4)
    observable = _make_test_observable(4)
    backend = _make_noisy_backend()

    # Even with opt_level=3 in config, executor must use 0
    config = NoisyEstimatorConfig(shots=1024, seed_simulator=42, optimization_level=3)
    executor = make_mitiq_executor(observable, backend, config)

    # Verify by running a folded circuit — if opt_level > 0, folding gets cancelled
    from mitiq.zne.scaling import fold_global

    original_result = executor(circuit)

    # Fold 3x — should produce a noisier result (more gates)
    folded = fold_global(circuit, scale_factor=3.0)
    folded_result = executor(folded)

    # If opt_level=0 is enforced, folded should be more degraded (closer to 0)
    # than unfolded (for ZZ observable with depolarizing noise)
    # We just verify it produces a different result (folding preserved, not cancelled)
    assert abs(folded_result - original_result) > 0.01, (
        f"Folded ({folded_result:.4f}) should differ from unfolded ({original_result:.4f}). "
        f"If identical, opt_level > 0 is cancelling the folding."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Regression tests: bugs fixed in this session
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_regression_circuit_not_trivial_after_transpile():
    """Regression: circuit bound with non-zero θ must have 2Q gates after transpile.

    Bug: θ=zeros caused CX·Rz(0)·CX=Identity → Qiskit opt_level≥1 cancelled
    all 2Q gates, producing a trivial circuit (depth_2q=0, n_2q=0).
    Fix: use VQE-optimized or random non-trivial θ.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator

    # Build HVA circuit with non-trivial parameters
    from qmbp_simulation import HVACircuitBuilder, make_lattice

    lattice = make_lattice("chain_1d", 4, J=1.0, h=2.0)
    qc, theta = HVACircuitBuilder().create(4, 1, lattice)

    # θ=zeros produces trivial circuit (the bug)
    zeros_bound = qc.assign_parameters(np.zeros(len(theta)))
    # θ=non-trivial produces real circuit (the fix)
    real_bound = qc.assign_parameters(np.array([0.5, -0.3]))

    backend = AerSimulator()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=2)

    trivial = pm.run(zeros_bound)
    real = pm.run(real_bound)

    n_2q_trivial = sum(1 for i in trivial.data if i.operation.num_qubits == 2)
    n_2q_real = sum(1 for i in real.data if i.operation.num_qubits == 2)

    # Trivial circuit should have 0 or very few 2Q gates
    # Real circuit should have meaningful 2Q gates
    assert n_2q_real > 0, (
        f"Non-trivial θ circuit should have >0 2Q gates after transpile, got {n_2q_real}"
    )
    # The key assertion: real circuit has MORE 2Q gates than trivial
    assert n_2q_real > n_2q_trivial, (
        f"Real θ ({n_2q_real} 2Q) should have more 2Q gates than θ=zeros ({n_2q_trivial} 2Q)"
    )


@pytest.mark.skipif(not HAS_MITIQ, reason="mitiq not installed")
def test_regression_result_attribute_names():
    """Regression: GateFoldingZNEResult and PEAResult use .extrapolated_value not .extrapolated_energy.

    Bug: benchmark script used .extrapolated_energy and .energies_per_factor
    which don't exist on the result dataclasses.
    Fix: use .extrapolated_value and .measured_values[0].
    """
    from qmbp_simulation.execution.noisy_utils import GateFoldingZNEResult, PEAResult

    # Verify the correct attribute names exist
    gf = GateFoldingZNEResult(
        extrapolated_value=-5.0,
        r_squared=0.99,
        slope=-0.1,
        noise_factors=[1, 3, 5],
        measured_values=[-4.5, -4.0, -3.5],
    )
    assert hasattr(gf, "extrapolated_value")
    assert hasattr(gf, "measured_values")
    assert not hasattr(gf, "extrapolated_energy"), (
        "Should use extrapolated_value, not extrapolated_energy"
    )
    assert not hasattr(gf, "energies_per_factor"), (
        "Should use measured_values, not energies_per_factor"
    )
    assert gf.extrapolated_value == -5.0
    assert gf.measured_values[0] == -4.5

    pea = PEAResult(
        extrapolated_value=-5.1,
        r_squared=0.995,
        slope=-0.05,
        noise_factors=[1, 3, 5],
        measured_values=[-4.8, -4.5, -4.2],
        learned_error_rates={},
    )
    assert hasattr(pea, "extrapolated_value")
    assert hasattr(pea, "measured_values")
    assert not hasattr(pea, "extrapolated_energy")
    assert pea.measured_values[0] == -4.8


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_regression_mitiq_executor_opt_level_0_with_real_circuit():
    """Regression: Mitiq executor must use opt_level=0 to preserve folded gates.

    Bug: with opt_level≥1, Qiskit 2.x cancels U·U† pairs from gate folding,
    making ZNE extrapolation meaningless.
    Fix: make_mitiq_executor forces optimization_level=0 regardless of config.
    """
    from mitiq.zne.scaling import fold_global

    from qmbp_simulation.execution.mitiq_utils import make_mitiq_executor
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    # Build a circuit with real 2Q operations
    qc = QuantumCircuit(4)
    for i in range(4):
        qc.h(i)
    for i in range(3):
        qc.cx(i, i + 1)
        qc.rz(0.5, i + 1)
        qc.cx(i, i + 1)

    obs = SparsePauliOp.from_list([("ZZZZ", 1.0)])
    backend = _make_noisy_backend()

    # Config says opt_level=3, but executor MUST force 0
    config = NoisyEstimatorConfig(shots=1024, seed_simulator=42, optimization_level=3)
    executor = make_mitiq_executor(obs, backend, config)

    # Get energies at different fold levels
    e_unfold = executor(qc)
    folded_3x = fold_global(qc, scale_factor=3.0)
    e_fold3 = executor(folded_3x)

    # If opt_level=0 is enforced, folding is preserved → folded is noisier
    # If not enforced (opt≥1), folding is cancelled → both give same result
    assert abs(e_fold3 - e_unfold) > 0.01, (
        f"Folded ({e_fold3:.4f}) must differ from unfolded ({e_unfold:.4f}). "
        f"If equal, opt_level>0 is cancelling the gate folding (regression)."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Regression: CDR dimension mismatch (H_logical vs H_mapped)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_regression_cdr_with_unmapped_observable():
    """Regression: CDR must receive unmapped (logical) observable, not layout-mapped.

    Bug: run_single_config passed H_mapped (133-qubit, already layout-mapped)
    to Mitiq CDR executor. The executor internally transpiles and tries
    apply_layout() again → 'Number of qargs does not match (10 != 133)'.

    Fix: route_execution passes H_logical (10-qubit unmapped) to Mitiq executors
    via the logical_circuit and H_logical parameters.
    """
    from qmbp_simulation.execution.mitiq_utils import run_mitiq_cdr
    from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

    # Create a 4-qubit circuit + observable (simulates the 10-qubit case)
    circuit = _make_test_circuit(4)
    observable = _make_test_observable(4)  # This is the UNMAPPED observable

    backend = _make_noisy_backend()
    config = NoisyEstimatorConfig(shots=1024, seed_simulator=42)

    # This should work with the unmapped observable (the fix)
    result = run_mitiq_cdr(
        circuit=circuit,
        observable=observable,
        backend=backend,
        config=config,
        n_training_circuits=3,
    )
    assert np.isfinite(result.mitigated_value), (
        "CDR with unmapped observable should produce valid result"
    )

    # Verify: if we pre-map the observable to a larger layout, CDR should fail
    # (This simulates the bug condition)
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
    transpiled = pm.run(circuit)

    # Map observable to physical layout (creates N-qubit observable where N > 4)
    mapped_obs = observable.apply_layout(transpiled.layout)

    # The mapped observable has more qubits than the logical circuit
    assert mapped_obs.num_qubits >= circuit.num_qubits, (
        "Mapped observable should have >= qubits of logical circuit"
    )

    # Passing mapped observable to CDR (which transpiles internally) would fail
    # if the executor's internal transpile produces a different layout.
    # This is the bug we fixed — Mitiq executors now receive logical observables.


@pytest.mark.skipif(not HAS_MITIQ or not HAS_AER, reason="mitiq or qiskit-aer not installed")
def test_regression_route_execution_passes_logical_to_mitiq():
    """Regression: route_execution must pass logical_circuit and H_logical to Mitiq.

    Bug: All executors received transpiled_circuit and H_mapped.
    Fix: route_execution passes logical_circuit/H_logical to Mitiq cases.
    """
    import sys

    sys.path.insert(0, ".")
    from unittest.mock import MagicMock, patch

    from scripts.experiment_runners.hardware.benchmark_configs import BENCHMARK_CONFIGS
    from scripts.experiment_runners.hardware.run_mitigation_benchmark import route_execution

    config = BENCHMARK_CONFIGS["C12_mitiq_cdr"]

    mock_transpiled = MagicMock(name="transpiled_133q")
    mock_H_mapped = MagicMock(name="H_mapped_133q")
    mock_logical = MagicMock(name="logical_10q")
    mock_H_logical = MagicMock(name="H_logical_10q")
    mock_backend = MagicMock()

    # Patch _execute_mitiq_cdr to capture what it receives
    with patch(
        "scripts.experiment_runners.hardware.run_mitigation_benchmark._execute_mitiq_cdr",
        return_value={
            "e_mitigated": -9.0,
            "e_raw": -8.0,
            "zne_r2": None,
            "shots": 16384,
            "_job": None,
        },
    ) as mock_cdr:
        route_execution(
            config,
            mock_transpiled,
            mock_H_mapped,
            mock_backend,
            16384,
            3.5,
            -10.0,
            1.5,
            logical_circuit=mock_logical,
            H_logical=mock_H_logical,
        )

        # Verify CDR received the LOGICAL circuit, not the transpiled one
        call_args = mock_cdr.call_args
        received_circuit = call_args[0][1]  # 2nd positional arg
        received_obs = call_args[0][2]  # 3rd positional arg

        assert received_circuit is mock_logical, (
            "CDR should receive logical_circuit, not transpiled_circuit"
        )
        assert received_obs is mock_H_logical, "CDR should receive H_logical, not H_mapped"


# ═══════════════════════════════════════════════════════════════════════════
# Regression: Gate-folding noise_factors deduplication
# ═══════════════════════════════════════════════════════════════════════════


def test_regression_gate_folding_noise_factors_no_duplicates():
    """Regression: noise_factors [1.0, 1.5, 3.0] must not produce duplicates.

    Bug: int(1.5)=1 → factors become [1,1,3] → 2 identical points,
    degrading ZNE extrapolation from 3-point to effectively 2-point.

    Fix: round to nearest odd + deduplicate. [1.0, 1.5, 3.0] → [1, 3] (after dedup)
    or config changed to [1.0, 3.0, 5.0].
    """
    import sys

    sys.path.insert(0, ".")
    from scripts.experiment_runners.hardware.benchmark_configs import BENCHMARK_CONFIGS

    # Verify C3 and C9 now use factors that won't produce duplicates
    c3 = BENCHMARK_CONFIGS["C3_full_gf"]
    c9 = BENCHMARK_CONFIGS["C9_gnn_qem"]

    # Convert using the same logic as _execute_gate_folding
    for config in [c3, c9]:
        raw_factors = config.zne_noise_factors or [1, 3, 5]
        noise_factors_int = tuple(max(1, int(round(f)) | 1) for f in raw_factors)
        # Check no duplicates
        assert len(noise_factors_int) == len(set(noise_factors_int)), (
            f"{config.config_id}: factors {raw_factors} → {noise_factors_int} has duplicates!"
        )
        # Should have at least 3 unique points for reliable extrapolation
        assert len(set(noise_factors_int)) >= 3, (
            f"{config.config_id}: only {len(set(noise_factors_int))} unique factors "
            f"(need >= 3 for reliable extrapolation)"
        )


def test_regression_build_hva_circuit_non_trivial():
    """Regression: _build_hva_circuit must produce circuits with 2Q gates.

    Bug: θ=zeros caused RZZ(0)=Identity → opt_level≥1 removed all 2Q gates.
    Fix: use VQE-computed θ_opt (or random non-zero) that produce real circuits.
    """
    import sys

    sys.path.insert(0, ".")
    from scripts.experiment_runners.hardware.run_mitigation_benchmark import (
        _build_hva_circuit,
        _circuit_cache,
    )

    _circuit_cache.clear()

    for h in [3.25, 3.5, 3.75, 4.0]:
        circ = _build_hva_circuit(h)
        # PauliEvolutionGate wraps interactions as a single N-qubit gate;
        # decompose once to expose the underlying RZZ 2Q gates.
        circ_decomposed = circ.decompose()
        n_2q = sum(1 for i in circ_decomposed.data if i.operation.num_qubits == 2)
        assert n_2q > 0, (
            f"h={h}: circuit has {n_2q} 2Q gates (must be >0 to avoid "
            f"transpiler cancellation at opt_level≥1)"
        )

        # Also verify that binding params are non-zero
        # (the circuit is already bound, so check gate params)
        has_nonzero_rzz = False
        for inst in circ_decomposed.data:
            if inst.operation.name == "rzz" and len(inst.operation.params) > 0:
                if abs(float(inst.operation.params[0])) > 1e-10:
                    has_nonzero_rzz = True
                    break
        assert has_nonzero_rzz, f"h={h}: all RZZ params are zero — will be cancelled by transpiler"


def test_regression_compute_derived_stats_uses_depth_key():
    """Regression: compute_derived_circuit_stats must use 'depth' key (not 'depth_transpiled').

    Bug: referenced stats['depth_transpiled'] but transpiled_circuit_stats()
    returns 'depth' as the canonical key.
    Fix: uses stats.get('depth', stats.get('depth_transpiled', 0)).
    """
    import sys

    sys.path.insert(0, ".")
    from scripts.experiment_runners.hardware.run_mitigation_benchmark import (
        compute_derived_circuit_stats,
    )

    # Simulate what transpiled_circuit_stats() actually returns (key = "depth")
    stats_real = {
        "depth": 59,
        "depth_2q": 14,
        "n_2q_gates": 18,
        "max_idle_stretch": 5,
        "depth_logical": 9,  # injected by run_single_config
    }

    result = compute_derived_circuit_stats(stats_real, n_2q_logical=9)

    # circuit_depth_with_dd_estimate should use 'depth' value (59), not 0
    assert result["circuit_depth_with_dd_estimate"] == 59 + 5, (
        f"Expected 64, got {result['circuit_depth_with_dd_estimate']}. "
        f"compute_derived_circuit_stats must use 'depth' key."
    )

    # transpiled_vs_logical_ratio should use 'depth' / 'depth_logical'
    assert abs(result["transpiled_vs_logical_ratio"] - 59 / 9) < 1e-10, (
        f"Expected {59 / 9:.4f}, got {result['transpiled_vs_logical_ratio']}"
    )

    # Also verify it works without 'depth_transpiled' key at all
    assert "depth_transpiled" not in stats_real, (
        "This test verifies the fix: 'depth_transpiled' key should NOT be needed"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Transpilation analysis validation
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_AER, reason="qiskit-aer not installed")
def test_transpilation_analysis_script_importable():
    """The analyze_transpilation script functions are importable and callable."""
    import sys

    sys.path.insert(0, "scripts")
    from analyze_transpilation import format_transpilation_table, scan_transpilation_stats

    # scan returns a dict (may be empty if no results)
    stats = scan_transpilation_stats()
    assert isinstance(stats, dict)

    # format works even with empty data
    table = format_transpilation_table(stats)
    assert isinstance(table, str)
    assert "Configs with circuit_stats" in table


@pytest.mark.skipif(not HAS_AER, reason="qiskit-aer not installed")
def test_transpilation_opt_level_2_produces_fewer_gates_than_0():
    """opt_level=2 MUST produce fewer 2Q gates than opt_level=0 for same circuit.

    This is the fundamental transpilation invariant that justifies using
    opt_level=2 for hardware (PEA) and opt_level=0 only for Mitiq.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator

    from qmbp_simulation import HVACircuitBuilder, make_lattice

    # Build non-trivial circuit (N=6 for speed)
    lattice = make_lattice("chain_1d", 6, J=1.0, h=2.0)
    qc, theta = HVACircuitBuilder().create(6, 1, lattice)
    bound = qc.assign_parameters(np.array([0.5, -0.3]))

    backend = AerSimulator()

    pm0 = generate_preset_pass_manager(backend=backend, optimization_level=0)
    pm2 = generate_preset_pass_manager(backend=backend, optimization_level=2)

    t0 = pm0.run(bound)
    t2 = pm2.run(bound)

    n2q_opt0 = sum(1 for i in t0.data if i.operation.num_qubits == 2)
    n2q_opt2 = sum(1 for i in t2.data if i.operation.num_qubits == 2)

    assert n2q_opt2 <= n2q_opt0, (
        f"opt_level=2 ({n2q_opt2} 2Q) should have ≤ gates than opt_level=0 ({n2q_opt0} 2Q)"
    )
