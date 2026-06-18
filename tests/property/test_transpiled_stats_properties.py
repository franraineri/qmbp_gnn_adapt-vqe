"""Property-based tests for transpiled_circuit_stats backward compatibility.

# Feature: mitigation-benchmark
# **Validates: Requirements 6.1, 6.4, 6.5**
#
# Property 9: Backward compatibility with new metrics — all original + new
#     fields present in the returned dict for any valid transpiled circuit.
#
# Property 10: Division-safe parallelism metrics — when depth_2q=0 (only 1Q
#     gates), parallelism_ratio and gate_density_2q return 0.0 safely.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
from qiskit.circuit import QuantumCircuit

from qmbp_simulation.analysis.circuit_visualizer import transpiled_circuit_stats

# ═══════════════════════════════════════════════════════════════════════════════
# Expected keys
# ═══════════════════════════════════════════════════════════════════════════════

ORIGINAL_KEYS = {
    "depth",
    "depth_2q",
    "n_2q_gates",
    "n_1q_gates",
    "total_gates",
    "count_ops",
    "num_tensor_factors",
    "width",
    "active_qubits",
}

NEW_KEYS = {
    "idle_cycles_per_qubit",
    "max_idle_stretch",
    "parallelism_ratio",
    "gate_density_2q",
}

ALL_KEYS = ORIGINAL_KEYS | NEW_KEYS


# ═══════════════════════════════════════════════════════════════════════════════
# Strategies
# ═══════════════════════════════════════════════════════════════════════════════


@st.composite
def random_circuits(draw: st.DrawFn) -> QuantumCircuit:
    """Generate random quantum circuits with a mix of 1Q and 2Q gates.

    Produces circuits with 2-8 qubits and 1-20 gates drawn from {h, cx, rz}.
    """
    n = draw(st.integers(min_value=2, max_value=8))
    qc = QuantumCircuit(n)
    n_gates = draw(st.integers(min_value=1, max_value=20))
    for _ in range(n_gates):
        gate_type = draw(st.sampled_from(["h", "cx", "rz"]))
        if gate_type == "h":
            q = draw(st.integers(0, n - 1))
            qc.h(q)
        elif gate_type == "cx":
            q1 = draw(st.integers(0, n - 1))
            q2 = draw(st.integers(0, n - 1).filter(lambda x, q1=q1: x != q1))
            qc.cx(q1, q2)
        elif gate_type == "rz":
            q = draw(st.integers(0, n - 1))
            qc.rz(0.5, q)
    return qc


@st.composite
def single_qubit_only_circuits(draw: st.DrawFn) -> QuantumCircuit:
    """Generate circuits containing ONLY 1-qubit gates (depth_2q guaranteed 0).

    This ensures parallelism_ratio and gate_density_2q must return 0.0.
    """
    n = draw(st.integers(min_value=2, max_value=6))
    qc = QuantumCircuit(n)
    n_gates = draw(st.integers(min_value=1, max_value=15))
    for _ in range(n_gates):
        gate_type = draw(st.sampled_from(["h", "rz", "x", "s"]))
        q = draw(st.integers(0, n - 1))
        if gate_type == "h":
            qc.h(q)
        elif gate_type == "rz":
            qc.rz(0.5, q)
        elif gate_type == "x":
            qc.x(q)
        elif gate_type == "s":
            qc.s(q)
    return qc


# ═══════════════════════════════════════════════════════════════════════════════
# Property 9: Backward compatibility with new metrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestTranspiledCircuitStatsBackwardCompat:
    """Property 9: all original + new fields present for any valid circuit.

    **Validates: Requirements 6.1, 6.5**
    """

    @given(circuit=random_circuits())
    @settings(max_examples=50, deadline=None)
    def test_all_keys_present_random_circuits(self, circuit: QuantumCircuit) -> None:
        """For any random circuit, stats dict contains ALL expected keys."""
        stats = transpiled_circuit_stats(circuit)

        missing = ALL_KEYS - set(stats.keys())
        assert not missing, f"Missing keys in stats: {missing}"

    @given(circuit=single_qubit_only_circuits())
    @settings(max_examples=30, deadline=None)
    def test_all_keys_present_single_qubit_circuits(self, circuit: QuantumCircuit) -> None:
        """For circuits with only 1Q gates, stats dict still has ALL keys."""
        stats = transpiled_circuit_stats(circuit)

        missing = ALL_KEYS - set(stats.keys())
        assert not missing, f"Missing keys in stats: {missing}"

    @given(circuit=random_circuits())
    @settings(max_examples=30, deadline=None)
    def test_original_fields_types(self, circuit: QuantumCircuit) -> None:
        """Original fields maintain expected types (int/dict)."""
        stats = transpiled_circuit_stats(circuit)

        assert isinstance(stats["depth"], int)
        assert isinstance(stats["depth_2q"], int)
        assert isinstance(stats["n_2q_gates"], int)
        assert isinstance(stats["n_1q_gates"], int)
        assert isinstance(stats["total_gates"], int)
        assert isinstance(stats["count_ops"], dict)

    @given(circuit=random_circuits())
    @settings(max_examples=30, deadline=None)
    def test_new_fields_types(self, circuit: QuantumCircuit) -> None:
        """New fields have correct types (float/int)."""
        stats = transpiled_circuit_stats(circuit)

        assert isinstance(stats["idle_cycles_per_qubit"], (int, float))
        assert isinstance(stats["max_idle_stretch"], int)
        assert isinstance(stats["parallelism_ratio"], (int, float))
        assert isinstance(stats["gate_density_2q"], (int, float))


# ═══════════════════════════════════════════════════════════════════════════════
# Property 10: Division-safe parallelism metrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestDivisionSafeParallelismMetrics:
    """Property 10: depth_2q=0 returns 0.0 safely without exceptions.

    **Validates: Requirements 6.4**
    """

    @given(circuit=single_qubit_only_circuits())
    @settings(max_examples=50, deadline=None)
    def test_parallelism_ratio_zero_when_no_2q_gates(self, circuit: QuantumCircuit) -> None:
        """When depth_2q=0, parallelism_ratio == 0.0 (no ZeroDivisionError)."""
        stats = transpiled_circuit_stats(circuit)

        assert stats["depth_2q"] == 0
        assert stats["parallelism_ratio"] == 0.0

    @given(circuit=single_qubit_only_circuits())
    @settings(max_examples=50, deadline=None)
    def test_gate_density_2q_zero_when_no_2q_gates(self, circuit: QuantumCircuit) -> None:
        """When depth_2q=0, gate_density_2q == 0.0 (no ZeroDivisionError)."""
        stats = transpiled_circuit_stats(circuit)

        assert stats["depth_2q"] == 0
        assert stats["gate_density_2q"] == 0.0

    @given(circuit=random_circuits())
    @settings(max_examples=30, deadline=None)
    def test_parallelism_ratio_non_negative(self, circuit: QuantumCircuit) -> None:
        """parallelism_ratio is always >= 0.0 for any circuit."""
        stats = transpiled_circuit_stats(circuit)
        assert stats["parallelism_ratio"] >= 0.0

    @given(circuit=random_circuits())
    @settings(max_examples=30, deadline=None)
    def test_gate_density_2q_non_negative(self, circuit: QuantumCircuit) -> None:
        """gate_density_2q is always >= 0.0 for any circuit."""
        stats = transpiled_circuit_stats(circuit)
        assert stats["gate_density_2q"] >= 0.0
