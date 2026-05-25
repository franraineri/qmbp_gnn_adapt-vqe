"""Property-based tests for qmbp_simulation.circuits submodule.

Uses Hypothesis to verify universal properties of HVA circuit construction
across many random inputs.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qmbp_simulation.circuits import HVACircuitBuilder
from qmbp_simulation.models import make_lattice

# ─────────────────────────────────────────────────────────────────────────────
# Property 4: HVA circuit parameter count invariant
# **Validates: Requirements 4.1, 4.2, 4.5, 8.3, 19.2**
# ─────────────────────────────────────────────────────────────────────────────


class TestProperty4HVACircuitParameterCountInvariant:
    """Property 4: For any valid n_qubits (2-8) and p in {1, 2}, the circuit
    returned by HVACircuitBuilder().create() has exactly 2*p parameters.
    """

    @given(
        n_qubits=st.integers(min_value=2, max_value=8),
        p_layers=st.sampled_from([1, 2]),
    )
    @settings(max_examples=50, deadline=None)
    def test_parameter_vector_length_equals_2p(self, n_qubits, p_layers):
        """ParameterVector length is exactly 2*p for any valid configuration."""
        lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=1.0)
        builder = HVACircuitBuilder()
        qc, theta = builder.create(n_qubits, p_layers, lattice)

        assert len(theta) == 2 * p_layers

    @given(
        n_qubits=st.integers(min_value=2, max_value=8),
        p_layers=st.sampled_from([1, 2]),
    )
    @settings(max_examples=50, deadline=None)
    def test_circuit_num_parameters_equals_2p(self, n_qubits, p_layers):
        """QuantumCircuit.num_parameters is exactly 2*p for any valid configuration."""
        lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=1.0)
        builder = HVACircuitBuilder()
        qc, theta = builder.create(n_qubits, p_layers, lattice)

        assert qc.num_parameters == 2 * p_layers

    @given(
        n_qubits=st.integers(min_value=2, max_value=8),
        p_layers=st.sampled_from([1, 2]),
    )
    @settings(max_examples=50, deadline=None)
    def test_circuit_qubit_count_matches_input(self, n_qubits, p_layers):
        """QuantumCircuit.num_qubits matches the requested n_qubits."""
        lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=1.0)
        builder = HVACircuitBuilder()
        qc, theta = builder.create(n_qubits, p_layers, lattice)

        assert qc.num_qubits == n_qubits


# ─────────────────────────────────────────────────────────────────────────────
# Property 5: Depth constraint enforcement
# **Validates: Requirements 4.1, 4.2, 4.5, 8.3, 19.2**
# ─────────────────────────────────────────────────────────────────────────────


class TestProperty5DepthConstraintEnforcement:
    """Property 5: For any p > 2, HVACircuitBuilder().create() always raises
    ValueError regardless of n_qubits or topology.
    """

    @given(
        n_qubits=st.integers(min_value=2, max_value=8),
        p_layers=st.integers(min_value=3, max_value=10),
    )
    @settings(max_examples=50, deadline=None)
    def test_p_greater_than_2_raises_valueerror(self, n_qubits, p_layers):
        """create() raises ValueError for any p > 2, regardless of n_qubits."""
        lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=1.0)
        builder = HVACircuitBuilder()

        with pytest.raises(ValueError, match="exceeds the maximum"):
            builder.create(n_qubits, p_layers, lattice)

    @given(
        p_layers=st.integers(min_value=3, max_value=10),
    )
    @settings(max_examples=50, deadline=None)
    def test_heisenberg_p_greater_than_2_raises_valueerror(self, p_layers):
        """create_heisenberg() also raises ValueError for p > 2."""
        n_qubits = 4
        lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=1.0)
        builder = HVACircuitBuilder()

        with pytest.raises(ValueError, match="exceeds the maximum"):
            builder.create_heisenberg(n_qubits, p_layers, lattice)
