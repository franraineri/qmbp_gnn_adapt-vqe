"""Property-based tests for qmbp_simulation.models submodule.

Uses Hypothesis to verify universal properties of lattice construction
and Hamiltonian building across many random inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from qiskit.quantum_info import SparsePauliOp

from qmbp_simulation.models import (
    SUPPORTED_TOPOLOGIES,
    HamiltonianBuilder,
    make_lattice,
)

# ─────────────────────────────────────────────────────────────────────────────
# Strategies
# ─────────────────────────────────────────────────────────────────────────────


def valid_topology_and_qubits() -> st.SearchStrategy[tuple[str, int]]:
    """Generate valid (topology, n_qubits) pairs respecting topology constraints.

    - chain_1d: n_qubits in [2, 8]
    - ladder: even n_qubits in [4, 8]
    - kagome: multiples of 3 in [3, 6]
    - triangular: n_qubits in [3, 8]
    """
    return st.one_of(
        st.tuples(st.just("chain_1d"), st.integers(min_value=2, max_value=8)),
        st.tuples(st.just("ladder"), st.sampled_from([4, 6, 8])),
        st.tuples(st.just("kagome"), st.sampled_from([3, 6])),
        st.tuples(st.just("triangular"), st.integers(min_value=3, max_value=8)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Property 1: Lattice construction produces valid Hamiltonians
# **Validates: Requirements 2.3, 2.4, 19.5**
# ─────────────────────────────────────────────────────────────────────────────


class TestProperty1LatticeConstructionProducesValidHamiltonians:
    """Property 1: For any valid topology and n_qubits, make_lattice followed
    by HamiltonianBuilder().build() produces a Hermitian operator with the
    correct number of qubits as a valid SparsePauliOp.
    """

    @given(
        topo_qubits=valid_topology_and_qubits(),
        j_val=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        h_val=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_hamiltonian_is_hermitian(self, topo_qubits, j_val, h_val):
        """H == H† for any valid lattice configuration."""
        topology, n_qubits = topo_qubits
        lattice = make_lattice(topology, n_qubits, J=j_val, h=h_val)
        builder = HamiltonianBuilder()
        H = builder.build(lattice)

        # Verify it's a SparsePauliOp
        assert isinstance(H, SparsePauliOp)

        # Verify correct qubit count
        assert H.num_qubits == n_qubits

        # Verify Hermiticity via matrix representation
        mat = H.to_matrix()
        np.testing.assert_allclose(
            mat,
            mat.conj().T,
            atol=1e-12,
            err_msg=f"Hamiltonian not Hermitian for {topology}, n={n_qubits}, J={j_val}, h={h_val}",
        )

    @given(
        topo_qubits=valid_topology_and_qubits(),
        j_val=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        h_val=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_hamiltonian_correct_qubit_count(self, topo_qubits, j_val, h_val):
        """H.num_qubits == n_qubits for any valid configuration."""
        topology, n_qubits = topo_qubits
        lattice = make_lattice(topology, n_qubits, J=j_val, h=h_val)
        builder = HamiltonianBuilder()
        H = builder.build(lattice)

        assert H.num_qubits == n_qubits

    @given(
        topo_qubits=valid_topology_and_qubits(),
        j_val=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        h_val=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_hamiltonian_has_real_coefficients(self, topo_qubits, j_val, h_val):
        """All Pauli coefficients are real (required for Hermiticity)."""
        topology, n_qubits = topo_qubits
        lattice = make_lattice(topology, n_qubits, J=j_val, h=h_val)
        builder = HamiltonianBuilder()
        H = builder.build(lattice)

        # SparsePauliOp coefficients must all be real for a Hermitian operator
        assert np.allclose(H.coeffs.imag, 0, atol=1e-12), (
            f"Complex coefficients found for {topology}, n={n_qubits}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Property 2: Input validation rejects invalid lattice parameters
# **Validates: Requirements 2.3, 2.4, 19.5**
# ─────────────────────────────────────────────────────────────────────────────


class TestProperty2InputValidationRejectsInvalidParameters:
    """Property 2: For invalid inputs, make_lattice raises ValueError:
    - n_qubits < 2
    - Invalid topology strings (not in SUPPORTED_TOPOLOGIES)
    And HamiltonianBuilder.validate() raises ValueError for invalid Hamiltonians.
    """

    @given(
        n_qubits=st.integers(min_value=-100, max_value=1),
        topology=st.sampled_from(list(SUPPORTED_TOPOLOGIES)),
    )
    @settings(max_examples=50, deadline=None)
    def test_rejects_n_qubits_less_than_2(self, n_qubits, topology):
        """make_lattice raises ValueError for n_qubits < 2."""
        with pytest.raises(ValueError):
            make_lattice(topology, n_qubits, J=1.0, h=1.0)

    @given(
        topology=st.text(min_size=1, max_size=20).filter(lambda t: t not in SUPPORTED_TOPOLOGIES),
    )
    @settings(max_examples=50, deadline=None)
    def test_rejects_invalid_topology(self, topology):
        """make_lattice raises ValueError for unsupported topology strings."""
        with pytest.raises(ValueError):
            make_lattice(topology, 4, J=1.0, h=1.0)

    @given(
        wrong_n=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=30, deadline=None)
    def test_validate_rejects_wrong_qubit_count(self, wrong_n):
        """HamiltonianBuilder.validate() raises ValueError when qubit count mismatches."""
        # Build a valid Hamiltonian for a 4-qubit chain
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.0)
        builder = HamiltonianBuilder()
        H = builder.build(lattice)

        # Validate with a different qubit count (must differ from 4)
        assume(wrong_n != 4)
        with pytest.raises(ValueError):
            builder.validate(H, n_qubits=wrong_n)
