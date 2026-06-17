"""Tests for ResourceEstimation integration and circuit resource utilities.

Tests the unified `transpiled_circuit_stats`, error budget calculation,
prediction tracking, and layout ranking functions.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit

from qmbp_simulation.analysis.circuit_visualizer import (
    build_error_prediction,
    circuit_summary,
    compute_error_budget,
    rank_layouts_by_depth_2q,
    select_best_layout_for_zne,
    transpiled_circuit_stats,
    validate_prediction_vs_result,
)

# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def simple_circuit():
    """A simple 4-qubit circuit with known structure."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.rz(0.5, 3)
    qc.cx(2, 3)
    return qc


@pytest.fixture
def parallel_2q_circuit():
    """Circuit with parallelizable 2Q gates (depth_2q < n_2q)."""
    qc = QuantumCircuit(4)
    qc.cx(0, 1)
    qc.cx(2, 3)  # Parallel with above
    qc.cx(0, 1)
    qc.cx(2, 3)  # Parallel with above
    return qc


@pytest.fixture
def serial_2q_circuit():
    """Circuit with serial 2Q gates (depth_2q == n_2q)."""
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    return qc


@pytest.fixture
def multi_layout_circuits():
    """Three circuits with different depth_2q for ranking tests."""
    # Circuit A: depth_2q = 1 (all parallel)
    a = QuantumCircuit(4)
    a.cx(0, 1)
    a.cx(2, 3)

    # Circuit B: depth_2q = 2 (serial pair)
    b = QuantumCircuit(4)
    b.cx(0, 1)
    b.cx(1, 2)

    # Circuit C: depth_2q = 3 (fully serial)
    c = QuantumCircuit(4)
    c.cx(0, 1)
    c.cx(1, 2)
    c.cx(2, 3)

    return [a, b, c]


# ═══════════════════════════════════════════════════════════════════════
# Test: transpiled_circuit_stats
# ═══════════════════════════════════════════════════════════════════════


class TestTranspiledCircuitStats:
    """Tests for the unified circuit resource stats function."""

    def test_returns_all_required_keys(self, simple_circuit):
        stats = transpiled_circuit_stats(simple_circuit)
        required = {
            "depth",
            "depth_2q",
            "n_2q_gates",
            "n_1q_gates",
            "total_gates",
            "count_ops",
            "num_tensor_factors",
            "width",
        }
        assert required.issubset(stats.keys())

    def test_depth_matches_circuit_depth(self, simple_circuit):
        stats = transpiled_circuit_stats(simple_circuit)
        assert stats["depth"] == simple_circuit.depth()

    def test_total_gates_equals_data_length(self, simple_circuit):
        stats = transpiled_circuit_stats(simple_circuit)
        assert stats["total_gates"] == len(simple_circuit.data)

    def test_gate_count_consistency(self, simple_circuit):
        """n_2q + n_1q should equal total_gates (no 3Q+ gates)."""
        stats = transpiled_circuit_stats(simple_circuit)
        assert stats["n_2q_gates"] + stats["n_1q_gates"] == stats["total_gates"]

    def test_count_ops_sums_to_total(self, simple_circuit):
        stats = transpiled_circuit_stats(simple_circuit)
        assert sum(stats["count_ops"].values()) == stats["total_gates"]

    def test_2q_from_count_ops_matches_n_2q(self, simple_circuit):
        """2Q gates counted from count_ops should match n_2q_gates."""
        stats = transpiled_circuit_stats(simple_circuit)
        two_q_names = {"cx", "cz", "ecr", "rzz", "rxx", "ryy", "cp"}
        n_2q_from_ops = sum(v for k, v in stats["count_ops"].items() if k in two_q_names)
        assert n_2q_from_ops == stats["n_2q_gates"]

    def test_depth_2q_less_or_equal_depth(self, simple_circuit):
        stats = transpiled_circuit_stats(simple_circuit)
        assert stats["depth_2q"] <= stats["depth"]

    def test_parallel_2q_depth_less_than_count(self, parallel_2q_circuit):
        """Parallel CX gates: depth_2q < n_2q_gates."""
        stats = transpiled_circuit_stats(parallel_2q_circuit)
        assert stats["n_2q_gates"] == 4
        assert stats["depth_2q"] == 2  # Two parallel pairs

    def test_serial_2q_depth_equals_count(self, serial_2q_circuit):
        """Serial CX gates: depth_2q == n_2q_gates."""
        stats = transpiled_circuit_stats(serial_2q_circuit)
        assert stats["n_2q_gates"] == 3
        assert stats["depth_2q"] == 3

    def test_connected_circuit_has_one_tensor_factor(self, simple_circuit):
        stats = transpiled_circuit_stats(simple_circuit)
        assert stats["num_tensor_factors"] == 1

    def test_disconnected_circuit_has_multiple_factors(self):
        """Disconnected qubits → multiple tensor factors."""
        qc = QuantumCircuit(6)
        qc.cx(0, 1)  # Component A
        qc.cx(3, 4)  # Component B (qubits 2, 5 idle)
        stats = transpiled_circuit_stats(qc)
        # Components: {0,1}, {3,4}, {2}, {5} = 4 tensor factors
        assert stats["num_tensor_factors"] == 4

    def test_active_qubits_computed_correctly(self, simple_circuit):
        stats = transpiled_circuit_stats(simple_circuit)
        expected = stats["width"] - stats["num_tensor_factors"] + 1
        assert stats["active_qubits"] == expected


# ═══════════════════════════════════════════════════════════════════════
# Test: compute_error_budget
# ═══════════════════════════════════════════════════════════════════════


class TestComputeErrorBudget:
    """Tests for error budget estimation."""

    def test_typical_fallback_returns_positive_budget(self, simple_circuit):
        budget = compute_error_budget(simple_circuit, backend=None)
        assert budget["error_budget"] > 0
        assert budget["source"] == "typical_fallback"

    def test_fidelity_estimate_in_valid_range(self, simple_circuit):
        budget = compute_error_budget(simple_circuit, backend=None)
        assert 0 < budget["fidelity_estimate"] <= 1.0

    def test_fidelity_is_exp_neg_error_budget(self, simple_circuit):
        budget = compute_error_budget(simple_circuit, backend=None)
        expected = np.exp(-budget["error_budget"])
        assert abs(budget["fidelity_estimate"] - expected) < 1e-10

    def test_rz_contributes_zero_error(self):
        """RZ is virtual on IBM → zero error contribution."""
        qc = QuantumCircuit(1)
        for _ in range(100):
            qc.rz(0.5, 0)
        budget = compute_error_budget(qc, backend=None)
        assert budget["error_budget"] == 0.0
        assert budget["fidelity_estimate"] == 1.0

    def test_more_gates_means_higher_error(self):
        """More 2Q gates → higher error budget."""
        qc_small = QuantumCircuit(2)
        qc_small.cx(0, 1)

        qc_large = QuantumCircuit(2)
        for _ in range(10):
            qc_large.cx(0, 1)

        budget_s = compute_error_budget(qc_small, backend=None)
        budget_l = compute_error_budget(qc_large, backend=None)
        assert budget_l["error_budget"] > budget_s["error_budget"]

    def test_per_gate_contribution_sums_to_total(self, simple_circuit):
        budget = compute_error_budget(simple_circuit, backend=None)
        total_from_parts = sum(budget["per_gate_contribution"].values())
        assert abs(total_from_parts - budget["error_budget"]) < 1e-10

    def test_depth_2q_included(self, simple_circuit):
        budget = compute_error_budget(simple_circuit, backend=None)
        assert "depth_2q" in budget
        assert budget["depth_2q"] >= 0


# ═══════════════════════════════════════════════════════════════════════
# Test: build_error_prediction / validate_prediction_vs_result
# ═══════════════════════════════════════════════════════════════════════


class TestErrorPrediction:
    """Tests for the prediction/validation tracking workflow."""

    def test_prediction_has_required_keys(self, simple_circuit):
        pred = build_error_prediction(simple_circuit, h_value=4.0)
        required = {
            "h",
            "depth_2q",
            "error_budget",
            "fidelity_estimate",
            "predicted_risk",
            "explanation",
            "source",
        }
        assert required.issubset(pred.keys())

    def test_low_error_circuit_predicts_low_risk(self):
        """A trivial circuit should predict low risk."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        pred = build_error_prediction(qc, h_value=4.0)
        assert pred["predicted_risk"] == "low"

    def test_high_error_circuit_predicts_high_risk(self):
        """Many 2Q gates should predict high risk."""
        qc = QuantumCircuit(2)
        for _ in range(80):
            qc.cx(0, 1)
        pred = build_error_prediction(qc, h_value=4.0)
        assert pred["predicted_risk"] == "high"

    def test_kappa_escalates_medium_to_high(self):
        """Low kappa + medium budget → high risk."""
        qc = QuantumCircuit(2)
        # ~35 CX gates → error_budget ≈ 35 × 0.008 = 0.28 (medium)
        for _ in range(35):
            qc.cx(0, 1)
        pred = build_error_prediction(qc, h_value=3.25, kappa=38.0)
        assert pred["predicted_risk"] == "high"

    def test_validation_correct_when_pass(self, simple_circuit):
        """Low-risk prediction + actual pass → prediction_correct=True."""
        pred = build_error_prediction(simple_circuit, h_value=4.0)
        val = validate_prediction_vs_result(pred, actual_de_gap=0.01)
        assert val["actual_outcome"] == "pass"
        if pred["predicted_risk"] == "low":
            assert val["prediction_correct"] is True

    def test_validation_records_all_metrics(self, simple_circuit):
        pred = build_error_prediction(simple_circuit, h_value=4.0)
        val = validate_prediction_vs_result(pred, actual_de_gap=0.03, actual_zne_r2=0.99)
        assert val["actual_de_gap"] == 0.03
        assert val["actual_zne_r2"] == 0.99
        assert "depth_2q" in val
        assert "error_budget" in val


# ═══════════════════════════════════════════════════════════════════════
# Test: rank_layouts_by_depth_2q / select_best_layout_for_zne
# ═══════════════════════════════════════════════════════════════════════


class TestLayoutRanking:
    """Tests for depth_2q-based layout ranking."""

    def test_ranking_sorted_by_depth_2q(self, multi_layout_circuits):
        ranked = rank_layouts_by_depth_2q(multi_layout_circuits)
        depths = [r["depth_2q"] for r in ranked]
        assert depths == sorted(depths)

    def test_best_layout_is_first_in_ranking(self, multi_layout_circuits):
        ranked = rank_layouts_by_depth_2q(multi_layout_circuits)
        best_idx, best_info = select_best_layout_for_zne(multi_layout_circuits)
        assert best_idx == ranked[0]["layout_idx"]
        assert best_info["depth_2q"] == ranked[0]["depth_2q"]

    def test_best_layout_has_minimum_depth_2q(self, multi_layout_circuits):
        _, best = select_best_layout_for_zne(multi_layout_circuits)
        all_d2q = [
            c.depth(filter_function=lambda x: x.operation.num_qubits == 2)
            for c in multi_layout_circuits
        ]
        assert best["depth_2q"] == min(all_d2q)

    def test_ranking_includes_layout_info_when_provided(self, multi_layout_circuits):
        layouts = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]]
        ranked = rank_layouts_by_depth_2q(multi_layout_circuits, layouts=layouts)
        for r in ranked:
            assert "layout" in r

    def test_single_circuit_returns_index_zero(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        best_idx, _ = select_best_layout_for_zne([qc])
        assert best_idx == 0

    def test_equal_depth_2q_preserves_order(self):
        """When all depth_2q are equal, original order is preserved."""
        qc1 = QuantumCircuit(4)
        qc1.cx(0, 1)
        qc1.cx(2, 3)

        qc2 = QuantumCircuit(4)
        qc2.cx(0, 1)
        qc2.cx(2, 3)

        ranked = rank_layouts_by_depth_2q([qc1, qc2])
        # Both have depth_2q=1 — original order should be kept
        assert ranked[0]["layout_idx"] == 0


# ═══════════════════════════════════════════════════════════════════════
# Test: circuit_summary (existing function, verify not broken)
# ═══════════════════════════════════════════════════════════════════════


class TestCircuitSummary:
    """Regression tests for the existing circuit_summary function."""

    def test_returns_expected_keys(self, simple_circuit):
        summary = circuit_summary(simple_circuit)
        expected = {
            "n_qubits",
            "depth",
            "n_parameters",
            "n_gates_total",
            "n_2q_gates",
            "n_1q_gates",
            "gate_counts",
        }
        assert expected.issubset(summary.keys())

    def test_with_params_includes_param_info(self):
        from qiskit.circuit import Parameter

        qc = QuantumCircuit(1)
        theta = Parameter("θ")
        qc.rx(theta, 0)
        summary = circuit_summary(qc, params=np.array([0.5]))
        assert "params" in summary
        assert summary["params"] == [0.5]


# ═══════════════════════════════════════════════════════════════════════
# Test: validate_transpiled_circuit_quality (preflight)
# ═══════════════════════════════════════════════════════════════════════


class TestTranspiledCircuitQualityPreflight:
    """Tests for the new preflight quality validation function."""

    @pytest.fixture
    def mock_logger(self):
        """Minimal logger mock for preflight tests."""

        class MockLogger:
            def __init__(self):
                self.logs = []

            def log(self, event, data=None):
                self.logs.append((event, data))

        return MockLogger()

    def test_simple_circuit_passes(self, simple_circuit, mock_logger):
        """A simple circuit with no backend should not abort."""
        from qmbp_simulation.execution.hardware.preflight import (
            validate_transpiled_circuit_quality,
        )

        # Use a fake backend for calibration data
        try:
            from qiskit_ibm_runtime.fake_provider import FakeKingston

            backend = FakeKingston()
        except ImportError:
            pytest.skip("FakeKingston not available")

        checks = validate_transpiled_circuit_quality(
            simple_circuit, backend, layout=None, logger=mock_logger
        )
        assert checks["abort"] is False
        assert "depth_2q" in checks
        assert checks["depth_2q"] >= 0

    def test_returns_fidelity_estimate(self, simple_circuit, mock_logger):
        """Should return a fidelity estimate in [0, 1]."""
        from qmbp_simulation.execution.hardware.preflight import (
            validate_transpiled_circuit_quality,
        )

        try:
            from qiskit_ibm_runtime.fake_provider import FakeKingston

            backend = FakeKingston()
        except ImportError:
            pytest.skip("FakeKingston not available")

        checks = validate_transpiled_circuit_quality(
            simple_circuit, backend, layout=None, logger=mock_logger
        )
        assert 0 < checks.get("fidelity_estimate", 0) <= 1.0

    def test_active_qubits_computed(self, simple_circuit, mock_logger):
        """Should report active_qubits."""
        from qmbp_simulation.execution.hardware.preflight import (
            validate_transpiled_circuit_quality,
        )

        try:
            from qiskit_ibm_runtime.fake_provider import FakeKingston

            backend = FakeKingston()
        except ImportError:
            pytest.skip("FakeKingston not available")

        checks = validate_transpiled_circuit_quality(
            simple_circuit, backend, layout=None, logger=mock_logger
        )
        assert checks.get("active_qubits") == 4

    def test_depth_2q_warning_for_deep_circuit(self, mock_logger):
        """A very deep serial circuit should trigger depth_2q warning."""
        from qmbp_simulation.execution.hardware.preflight import (
            validate_transpiled_circuit_quality,
        )

        try:
            from qiskit_ibm_runtime.fake_provider import FakeKingston

            backend = FakeKingston()
        except ImportError:
            pytest.skip("FakeKingston not available")

        # Build a very deep serial 2Q circuit (depth_2q = 40)
        qc = QuantumCircuit(2)
        for _ in range(40):
            qc.cx(0, 1)

        checks = validate_transpiled_circuit_quality(
            qc,
            backend,
            layout=None,
            logger=mock_logger,
            depth_2q_warn_threshold=30,
        )
        assert "depth_2q_warning" in checks
        assert checks["depth_2q"] == 40
