"""Property-based tests for qmbp_simulation.predictors submodule.

Uses Hypothesis to verify universal properties of graph dataset construction,
fidelity filtering, and MPNN checkpoint round-trip across many random inputs.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from qmbp_simulation.models import make_lattice
from qmbp_simulation.predictors import (
    MPNNPredictor,
    build_graph_dataset,
    load_mpnn_checkpoint,
    save_mpnn_checkpoint,
)

# ─────────────────────────────────────────────────────────────────────────────
# Strategies
# ─────────────────────────────────────────────────────────────────────────────


@st.composite
def vqe_results_strategy(draw):
    """Generate synthetic VQE results for a 4-qubit chain with p=1.

    Returns (n_points, h_values, theta_opt, e_exact, fidelities) where
    all fidelities are above 0.93 to ensure they pass the filter.
    """
    n_points = draw(st.integers(min_value=3, max_value=10))
    # h_values: distinct values in [0.5, 3.0]
    h_values = np.sort(
        draw(
            st.lists(
                st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False),
                min_size=n_points,
                max_size=n_points,
            ).map(np.array)
        )
    )
    # theta_opt: random angles in [-pi, pi], shape [n_points, 2] for p=1
    theta_opt = draw(
        st.lists(
            st.lists(
                st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False),
                min_size=2,
                max_size=2,
            ),
            min_size=n_points,
            max_size=n_points,
        ).map(lambda x: np.array(x))
    )
    # e_exact: negative energies
    e_exact = draw(
        st.lists(
            st.floats(min_value=-10.0, max_value=-0.1, allow_nan=False, allow_infinity=False),
            min_size=n_points,
            max_size=n_points,
        ).map(np.array)
    )
    # fidelities: all above 0.93 to pass filter
    fidelities = draw(
        st.lists(
            st.floats(min_value=0.93, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n_points,
            max_size=n_points,
        ).map(np.array)
    )
    return n_points, h_values, theta_opt, e_exact, fidelities


@st.composite
def low_fidelity_strategy(draw):
    """Generate fidelity arrays where fewer than 3 points pass the 0.93 threshold.

    Returns (n_points, fidelities) where at most 2 values are >= 0.93.
    """
    n_points = draw(st.integers(min_value=3, max_value=10))
    # Number of passing points: 0, 1, or 2
    n_passing = draw(st.integers(min_value=0, max_value=2))
    n_failing = n_points - n_passing

    # Generate failing fidelities (all < 0.93)
    failing = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=0.929, allow_nan=False, allow_infinity=False),
            min_size=n_failing,
            max_size=n_failing,
        )
    )
    # Generate passing fidelities
    passing = draw(
        st.lists(
            st.floats(min_value=0.93, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n_passing,
            max_size=n_passing,
        )
    )
    fidelities = np.array(failing + passing)
    # Shuffle so passing points aren't always at the end
    indices = draw(st.permutations(list(range(n_points))))
    fidelities = fidelities[indices]
    return n_points, fidelities


@st.composite
def mpnn_config_strategy(draw):
    """Generate valid MPNNPredictor configurations (small for speed)."""
    hidden_dim = draw(st.sampled_from([8, 16, 32]))
    n_layers = draw(st.integers(min_value=1, max_value=2))
    output_dim = draw(st.sampled_from([2, 4]))
    per_parameter_heads = draw(st.booleans())
    norm_type = draw(st.sampled_from(["batch", "layer", "none"]))
    # per_parameter_heads requires even output_dim (always true for 2, 4)
    return {
        "node_features": 2,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "output_dim": output_dim,
        "per_parameter_heads": per_parameter_heads,
        "norm_type": norm_type,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Property 10: Graph dataset construction preserves structure
# **Validates: Requirements 7.2, 7.4, 7.5, 10.5, 19.6**
# ─────────────────────────────────────────────────────────────────────────────


class TestProperty10GraphDatasetConstructionPreservesStructure:
    """Property 10: For any valid set of VQE results (n_points in 3-10,
    n_qubits=4, p=1 → 2 params), each Data object in the returned list
    has correct tensor shapes and the count matches fidelity-filtered points.
    """

    @given(data=vqe_results_strategy())
    @settings(max_examples=30, deadline=None)
    def test_node_features_shape(self, data):
        """data.x has shape (n_qubits, node_features) where node_features=2."""
        n_points, h_values, theta_opt, e_exact, fidelities = data
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.0)

        dataset = build_graph_dataset(lattice, h_values, theta_opt, e_exact, fidelities)

        for graph in dataset:
            assert graph.x.shape == (4, 2), f"Expected x shape (4, 2), got {graph.x.shape}"

    @given(data=vqe_results_strategy())
    @settings(max_examples=30, deadline=None)
    def test_edge_index_shape(self, data):
        """data.edge_index has shape (2, n_edges*2) for undirected graph."""
        n_points, h_values, theta_opt, e_exact, fidelities = data
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.0)

        dataset = build_graph_dataset(lattice, h_values, theta_opt, e_exact, fidelities)

        for graph in dataset:
            # edge_index must be 2D with first dim = 2
            assert graph.edge_index.shape[0] == 2, (
                f"Expected edge_index first dim 2, got {graph.edge_index.shape[0]}"
            )
            # For undirected graph, n_edges must be even (both directions)
            n_directed_edges = graph.edge_index.shape[1]
            assert n_directed_edges % 2 == 0, (
                f"Expected even number of directed edges, got {n_directed_edges}"
            )
            # chain_1d with 4 qubits has 3 bonds → 6 directed edges
            assert n_directed_edges == 6, (
                f"Expected 6 directed edges for 4-qubit chain, got {n_directed_edges}"
            )

    @given(data=vqe_results_strategy())
    @settings(max_examples=30, deadline=None)
    def test_target_shape(self, data):
        """data.y has shape (output_dim,) where output_dim=2 for p=1."""
        n_points, h_values, theta_opt, e_exact, fidelities = data
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.0)

        dataset = build_graph_dataset(lattice, h_values, theta_opt, e_exact, fidelities)

        for graph in dataset:
            assert graph.y.shape == (2,), f"Expected y shape (2,), got {graph.y.shape}"

    @given(data=vqe_results_strategy())
    @settings(max_examples=30, deadline=None)
    def test_dataset_length_matches_filtered_points(self, data):
        """Number of Data objects equals number of points passing fidelity filter."""
        n_points, h_values, theta_opt, e_exact, fidelities = data
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.0)

        dataset = build_graph_dataset(lattice, h_values, theta_opt, e_exact, fidelities)

        # All fidelities are >= 0.93 in our strategy, so all should pass
        expected_count = int(np.sum(fidelities >= 0.93))
        assert len(dataset) == expected_count, (
            f"Expected {expected_count} graphs, got {len(dataset)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Property 11: Fidelity filter enforcement
# **Validates: Requirements 7.2, 7.4, 7.5, 10.5, 19.6**
# ─────────────────────────────────────────────────────────────────────────────


class TestProperty11FidelityFilterEnforcement:
    """Property 11: When fewer than 3 fidelity values pass the 0.93 threshold,
    build_graph_dataset raises ValueError.
    """

    @given(data=low_fidelity_strategy())
    @settings(max_examples=30, deadline=None)
    def test_raises_when_fewer_than_3_pass(self, data):
        """build_graph_dataset raises ValueError when < 3 points pass fidelity filter."""
        n_points, fidelities = data
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.0)

        # Generate matching synthetic data
        h_values = np.linspace(0.5, 2.5, n_points)
        theta_opt = np.random.default_rng(42).uniform(-np.pi, np.pi, (n_points, 2))
        e_exact = np.linspace(-5.0, -2.0, n_points)

        with pytest.raises(ValueError, match="Fewer than 3"):
            build_graph_dataset(lattice, h_values, theta_opt, e_exact, fidelities)


# ─────────────────────────────────────────────────────────────────────────────
# Property 12: MPNN checkpoint round-trip
# **Validates: Requirements 7.2, 7.4, 7.5, 10.5, 19.6**
# ─────────────────────────────────────────────────────────────────────────────


class TestProperty12MPNNCheckpointRoundTrip:
    """Property 12: For any MPNNPredictor instance, save_mpnn_checkpoint
    followed by load_mpnn_checkpoint produces a model with identical state_dict.
    """

    @given(config=mpnn_config_strategy())
    @settings(max_examples=30, deadline=None)
    def test_checkpoint_preserves_state_dict(self, config):
        """All tensor values in state_dict are identical after save/load."""
        model = MPNNPredictor(**config)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as f:
            path = f.name

        save_mpnn_checkpoint(model, path, training_metadata={"test": True})
        loaded = load_mpnn_checkpoint(path)

        # Compare all state_dict entries
        original_state = model.state_dict()
        loaded_state = loaded.state_dict()

        assert set(original_state.keys()) == set(loaded_state.keys()), (
            f"Key mismatch: {set(original_state.keys()) ^ set(loaded_state.keys())}"
        )

        for key in original_state:
            assert torch.equal(original_state[key], loaded_state[key]), (
                f"Tensor mismatch in '{key}': "
                f"max diff = {(original_state[key] - loaded_state[key]).abs().max().item()}"
            )

    @given(config=mpnn_config_strategy())
    @settings(max_examples=30, deadline=None)
    def test_checkpoint_preserves_architecture(self, config):
        """Loaded model has same architecture attributes as original."""
        model = MPNNPredictor(**config)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as f:
            path = f.name

        save_mpnn_checkpoint(model, path)
        loaded = load_mpnn_checkpoint(path)

        assert loaded.node_features == config["node_features"]
        assert loaded.hidden_dim == config["hidden_dim"]
        assert loaded.n_layers == config["n_layers"]
        assert loaded.output_dim == config["output_dim"]
        assert loaded.per_parameter_heads == config["per_parameter_heads"]
        assert loaded.norm_type == config["norm_type"]
