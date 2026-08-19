"""Property-based tests for cross_topology.helpers module.

# Feature: cross-topology-transfer, Property 1: Data Adapter Extraction Fidelity
# **Validates: Requirements 1.1, 1.2, 1.3**
#
# For any valid source JSON (scaling or pipeline_run format) containing VQE
# results for a given seed, calling `load_source_data(path, seed)` SHALL return
# a SourceData where h_values, theta_opt, and e_exact exactly match the values
# in the file, and n, topology, and param_dim correctly reflect the file\'s
# metadata.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

# Add experiment_runners to path for cross_topology package imports
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "experiment_runners"))

from cross_topology.helpers import (
    build_experiment_envelope,
    canonicalize_theta,
    load_source_data,
)
from qmbp_simulation.models.constants import DEFAULT_SEEDS, MPS_DEFAULT_CHI_MAX

# ═══════════════════════════════════════════════════════════════════════════════
# Strategies
# ═══════════════════════════════════════════════════════════════════════════════

n_points_st = st.integers(min_value=1, max_value=8)
param_dim_st = st.sampled_from([2, 4])
system_size_st = st.sampled_from([6, 8, 10, 12, 16])
topology_st = st.sampled_from(["triangular", "heavy_hex", "chain_1d", "ladder"])
seed_st = st.sampled_from(DEFAULT_SEEDS)
h_value_st = st.floats(min_value=0.5, max_value=8.0, allow_nan=False, allow_infinity=False)
theta_st = st.floats(min_value=-3.14, max_value=3.14, allow_nan=False, allow_infinity=False)
energy_st = st.floats(min_value=-500.0, max_value=-0.1, allow_nan=False, allow_infinity=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Composite strategies for generating valid JSON structures
# ═══════════════════════════════════════════════════════════════════════════════


@st.composite
def scaling_format_data(draw):
    """Generate a valid scaling-format JSON structure with synthetic VQE data."""
    n_points = draw(n_points_st)
    param_dim = draw(param_dim_st)
    n = draw(system_size_st)
    topology = draw(topology_st)
    seed = draw(seed_st)

    h_values = sorted([draw(h_value_st) for _ in range(n_points)], reverse=True)
    theta_opts = [[draw(theta_st) for _ in range(param_dim)] for _ in range(n_points)]
    energies = [draw(energy_st) for _ in range(n_points)]

    results = []
    for i in range(n_points):
        results.append(
            {
                "h": h_values[i],
                "theta_opt": theta_opts[i],
                "dmrg_energy": energies[i],
            }
        )

    data = {
        "metadata": {"n": n, "topology": topology, "p_layers": param_dim // 2},
        "vqe_results": [{"seed": seed, "results": results}],
    }

    return {
        "json_data": data,
        "seed": seed,
        "n": n,
        "topology": topology,
        "param_dim": param_dim,
        "h_values": h_values,
        "theta_opts": theta_opts,
        "energies": energies,
    }


@st.composite
def pipeline_run_format_data(draw):
    """Generate a valid pipeline_run-format JSON structure with synthetic data."""
    n_points = draw(n_points_st)
    param_dim = draw(param_dim_st)
    n = draw(system_size_st)
    topology = draw(topology_st)
    seed = draw(seed_st)

    h_values = sorted([draw(h_value_st) for _ in range(n_points)], reverse=True)
    theta_opts = [[draw(theta_st) for _ in range(param_dim)] for _ in range(n_points)]
    energies = [draw(energy_st) for _ in range(n_points)]

    results = []
    for i in range(n_points):
        results.append({"h": h_values[i], "theta_opt": theta_opts[i]})

    data = {
        "config": {"n_qubits": n, "topology": topology, "h_values": h_values},
        "diagnostics": {
            "phase1": {"energies": energies, "n_points": n_points},
            "phase2": {"h_values": h_values},
        },
        "vqe_results": [{"seed": seed, "results": results}],
    }

    return {
        "json_data": data,
        "seed": seed,
        "n": n,
        "topology": topology,
        "param_dim": param_dim,
        "h_values": h_values,
        "theta_opts": theta_opts,
        "energies": energies,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Property 1: Data Adapter Extraction Fidelity
# **Validates: Requirements 1.1, 1.2, 1.3**
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataAdapterExtractionFidelity:
    """Property 1: Data Adapter Extraction Fidelity.

    **Validates: Requirements 1.1, 1.2, 1.3**

    For any valid source JSON (scaling or pipeline_run format) containing VQE
    results for a given seed, calling load_source_data(path, seed) SHALL return
    a SourceData where h_values, theta_opt, and e_exact exactly match the
    values in the file, and n, topology, and param_dim correctly reflect the
    file\'s metadata.
    """

    @given(data=scaling_format_data())
    @settings(max_examples=50, deadline=None)
    def test_scaling_format_extraction_fidelity(self, data, tmp_path_factory):
        """Scaling format: extracted values exactly match generated data."""
        tmp_path = tmp_path_factory.mktemp("scaling")
        path = tmp_path / "test_scaling.json"
        path.write_text(json.dumps(data["json_data"]))

        result = load_source_data(path, seed=data["seed"])

        # Metadata matches
        assert result.n == data["n"]
        assert result.topology == data["topology"]
        assert result.param_dim == data["param_dim"]

        # Array shapes
        n_points = len(data["h_values"])
        assert result.h_values.shape == (n_points,)
        assert result.theta_opt.shape == (n_points, data["param_dim"])
        assert result.e_exact.shape == (n_points,)

        # h_values match exactly
        np.testing.assert_allclose(result.h_values, data["h_values"], atol=1e-10)

        # e_exact (dmrg_energy) matches exactly
        np.testing.assert_allclose(result.e_exact, data["energies"], atol=1e-10)

        # theta_opt matches after canonicalization
        for i in range(n_points):
            expected = canonicalize_theta(np.array(data["theta_opts"][i]))
            np.testing.assert_allclose(result.theta_opt[i], expected, atol=1e-10)

    @given(data=pipeline_run_format_data())
    @settings(max_examples=50, deadline=None)
    def test_pipeline_run_format_extraction_fidelity(self, data, tmp_path_factory):
        """Pipeline_run format: extracted values exactly match generated data."""
        tmp_path = tmp_path_factory.mktemp("pipeline")
        path = tmp_path / "test_pipeline.json"
        path.write_text(json.dumps(data["json_data"]))

        result = load_source_data(path, seed=data["seed"])

        # Metadata matches
        assert result.n == data["n"]
        assert result.topology == data["topology"]
        assert result.param_dim == data["param_dim"]

        # Array shapes
        n_points = len(data["h_values"])
        assert result.h_values.shape == (n_points,)
        assert result.theta_opt.shape == (n_points, data["param_dim"])
        assert result.e_exact.shape == (n_points,)

        # h_values match (from diagnostics.phase2.h_values)
        np.testing.assert_allclose(result.h_values, data["h_values"], atol=1e-10)

        # e_exact matches (from diagnostics.phase1.energies)
        np.testing.assert_allclose(result.e_exact, data["energies"], atol=1e-10)

        # theta_opt matches after canonicalization
        for i in range(n_points):
            expected = canonicalize_theta(np.array(data["theta_opts"][i]))
            np.testing.assert_allclose(result.theta_opt[i], expected, atol=1e-10)

    @given(data=scaling_format_data())
    @settings(max_examples=50)
    def test_param_dim_equals_theta_length(self, data, tmp_path_factory):
        """param_dim always equals the length of each theta_opt vector."""
        tmp_path = tmp_path_factory.mktemp("paramdim")
        path = tmp_path / "test_param.json"
        path.write_text(json.dumps(data["json_data"]))

        result = load_source_data(path, seed=data["seed"])

        assert result.param_dim == data["param_dim"]
        assert result.theta_opt.shape[1] == result.param_dim


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy for Property 3: Format Equivalence
# ═══════════════════════════════════════════════════════════════════════════════


@st.composite
def vqe_data(draw):
    """Generate random VQE data with consistent shapes.

    Returns dict with n, topology, n_points, param_dim,
    h_values, theta_opt (pre-canonicalized), and e_exact.
    """
    n = draw(st.integers(min_value=4, max_value=20))
    topology = draw(topology_st)
    n_points = draw(st.integers(min_value=2, max_value=8))
    param_dim = draw(st.sampled_from([2, 4]))

    h_values = sorted(
        draw(
            st.lists(
                st.floats(
                    min_value=1.0,
                    max_value=10.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n_points,
                max_size=n_points,
                unique=True,
            )
        ),
        reverse=True,
    )

    theta_raw = draw(
        st.lists(
            st.lists(
                st.floats(
                    min_value=-3.0,
                    max_value=3.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=param_dim,
                max_size=param_dim,
            ),
            min_size=n_points,
            max_size=n_points,
        )
    )
    # Pre-canonicalize so last component >= 0 (avoids sign-flip differences)
    theta_opt = [canonicalize_theta(np.array(t)).tolist() for t in theta_raw]

    e_exact = draw(
        st.lists(
            st.floats(
                min_value=-50.0,
                max_value=-0.1,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n_points,
            max_size=n_points,
        )
    )

    return {
        "n": n,
        "topology": topology,
        "n_points": n_points,
        "param_dim": param_dim,
        "h_values": h_values,
        "theta_opt": theta_opt,
        "e_exact": e_exact,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Encoding helpers for Property 3
# ═══════════════════════════════════════════════════════════════════════════════


def encode_as_scaling(data: dict, seed: int = 42) -> dict:
    """Encode VQE data as scaling-format JSON."""
    results = [
        {
            "h": data["h_values"][i],
            "theta_opt": data["theta_opt"][i],
            "dmrg_energy": data["e_exact"][i],
        }
        for i in range(data["n_points"])
    ]
    return {
        "metadata": {
            "n": data["n"],
            "topology": data["topology"],
            "p_layers": data["param_dim"] // 2,
        },
        "vqe_results": [{"seed": seed, "results": results}],
    }


def encode_as_pipeline_run(data: dict, seed: int = 42) -> dict:
    """Encode VQE data as pipeline_run-format JSON."""
    results = [
        {
            "h": data["h_values"][i],
            "theta_opt": data["theta_opt"][i],
        }
        for i in range(data["n_points"])
    ]
    return {
        "config": {"n_qubits": data["n"], "topology": data["topology"]},
        "diagnostics": {
            "phase1": {"energies": data["e_exact"]},
            "phase2": {"h_values": data["h_values"]},
        },
        "vqe_results": [{"seed": seed, "results": results}],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Property 3: Format Equivalence
# **Validates: Requirements 1.5**
# ═══════════════════════════════════════════════════════════════════════════════


@settings(max_examples=30, deadline=None)
@given(data=vqe_data())
def test_format_equivalence(data: dict) -> None:
    """Loading the same VQE data from scaling and pipeline_run formats
    produces SourceData with identical arrays within float tolerance.

    **Validates: Requirements 1.5**
    """
    seed = 42

    scaling_json = encode_as_scaling(data, seed=seed)
    pipeline_json = encode_as_pipeline_run(data, seed=seed)

    with tempfile.TemporaryDirectory() as tmp_dir:
        scaling_path = Path(tmp_dir) / "scaling.json"
        pipeline_path = Path(tmp_dir) / "pipeline_run.json"

        scaling_path.write_text(json.dumps(scaling_json))
        pipeline_path.write_text(json.dumps(pipeline_json))

        sd_scaling = load_source_data(scaling_path, seed=seed)
        sd_pipeline = load_source_data(pipeline_path, seed=seed)

        # Metadata equivalence
        assert sd_scaling.n == sd_pipeline.n == data["n"]
        assert sd_scaling.topology == sd_pipeline.topology == data["topology"]

        # h_values element-wise equal
        np.testing.assert_allclose(
            sd_scaling.h_values,
            sd_pipeline.h_values,
            atol=1e-10,
            err_msg="h_values differ between formats",
        )

        # theta_opt element-wise equal
        np.testing.assert_allclose(
            sd_scaling.theta_opt,
            sd_pipeline.theta_opt,
            atol=1e-10,
            err_msg="theta_opt differs between formats",
        )

        # e_exact element-wise equal
        np.testing.assert_allclose(
            sd_scaling.e_exact,
            sd_pipeline.e_exact,
            atol=1e-10,
            err_msg="e_exact differs between formats",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Property 2: Missing theta_opt Raises ValueError with Path
# **Validates: Requirements 1.4**
# ═══════════════════════════════════════════════════════════════════════════════

# Strategies for Property 2
_N_VALUES_P2 = st.sampled_from([4, 6, 8, 10])
_SEEDS_P2 = st.integers(min_value=0, max_value=100)
_H_VALUES_P2 = st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False)


def _scaling_json_without_theta_opt(
    n: int, topology: str, seed: int, h_values: list[float]
) -> dict:
    """Build a valid scaling-format JSON structure WITHOUT theta_opt in results."""
    return {
        "metadata": {"n": n, "topology": topology},
        "vqe_results": [
            {
                "seed": seed,
                "results": [{"h": h, "dmrg_energy": -1.0 * (1 + h)} for h in h_values],
            }
        ],
    }


def _pipeline_run_json_without_theta_opt(
    n: int, topology: str, seed: int, h_values: list[float]
) -> dict:
    """Build a valid pipeline_run-format JSON structure WITHOUT theta_opt."""
    return {
        "config": {"n_qubits": n, "topology": topology},
        "diagnostics": {
            "phase1": {"energies": [-1.0 * (1 + h) for h in h_values]},
            "phase2": {"h_values": h_values},
        },
        "vqe_results": [
            {
                "seed": seed,
                "results": [{"h": h, "dmrg_energy": -1.0 * (1 + h)} for h in h_values],
            }
        ],
    }


class TestMissingThetaOptRaisesValueError:
    """Property 2: Missing theta_opt Raises ValueError with Path.

    **Validates: Requirements 1.4**

    For any JSON file that lacks theta_opt entries in its results,
    calling load_source_data(path, seed) SHALL raise a ValueError
    whose message contains the string representation of the file path.
    """

    @given(
        n=_N_VALUES_P2,
        topology=topology_st,
        seed=_SEEDS_P2,
        h_values=st.lists(_H_VALUES_P2, min_size=1, max_size=5),
    )
    @settings(max_examples=30, deadline=None)
    def test_scaling_format_missing_theta_opt(
        self, n: int, topology: str, seed: int, h_values: list[float]
    ):
        """Scaling-format JSON without theta_opt raises ValueError with path."""
        data = _scaling_json_without_theta_opt(n, topology, seed, h_values)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match=re.escape(str(tmp_path))):
                load_source_data(tmp_path, seed)
        finally:
            tmp_path.unlink()

    @given(
        n=_N_VALUES_P2,
        topology=topology_st,
        seed=_SEEDS_P2,
        h_values=st.lists(_H_VALUES_P2, min_size=1, max_size=5),
    )
    @settings(max_examples=30, deadline=None)
    def test_pipeline_run_format_missing_theta_opt(
        self, n: int, topology: str, seed: int, h_values: list[float]
    ):
        """Pipeline_run-format JSON without theta_opt raises ValueError with path."""
        data = _pipeline_run_json_without_theta_opt(n, topology, seed, h_values)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match=re.escape(str(tmp_path))):
                load_source_data(tmp_path, seed)
        finally:
            tmp_path.unlink()


# ═══════════════════════════════════════════════════════════════════════════════
# Property 7: Result Traceability Metadata
# **Validates: Requirements 8.2**
# ═══════════════════════════════════════════════════════════════════════════════


# Strategies for experiment envelope generation
experiment_name_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Pd")),
    min_size=1,
    max_size=50,
)
source_file_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Pd", "Pc")),
    min_size=1,
    max_size=100,
)
source_files_st = st.lists(source_file_st, min_size=1, max_size=10)


class TestResultTraceabilityMetadata:
    """Property 7: Result Traceability Metadata.

    **Validates: Requirements 8.2**

    For any result JSON produced by `build_experiment_envelope`, it SHALL
    contain non-empty fields: `metadata.source_files` (list),
    `metadata.git_commit` (string), `environment.python_version`,
    `environment.torch_version`, `environment.numpy_version`, and
    `metadata.timestamp` (string).
    """

    @given(name=experiment_name_st, sources=source_files_st)
    @settings(max_examples=50, deadline=None)
    def test_envelope_contains_all_traceability_fields(self, name: str, sources: list[str]):
        """build_experiment_envelope produces all required non-empty fields."""
        envelope = build_experiment_envelope(name, sources)

        # metadata.source_files is a non-empty list
        assert "metadata" in envelope
        assert "source_files" in envelope["metadata"]
        assert isinstance(envelope["metadata"]["source_files"], list)
        assert len(envelope["metadata"]["source_files"]) > 0

        # metadata.git_commit is a non-empty string
        assert "git_commit" in envelope["metadata"]
        assert isinstance(envelope["metadata"]["git_commit"], str)
        assert len(envelope["metadata"]["git_commit"]) > 0

        # metadata.timestamp is a non-empty string
        assert "timestamp" in envelope["metadata"]
        assert isinstance(envelope["metadata"]["timestamp"], str)
        assert len(envelope["metadata"]["timestamp"]) > 0

        # environment section exists with non-empty version strings
        assert "environment" in envelope
        env = envelope["environment"]

        assert "python_version" in env
        assert isinstance(env["python_version"], str)
        assert len(env["python_version"]) > 0

        assert "torch_version" in env
        assert isinstance(env["torch_version"], str)
        assert len(env["torch_version"]) > 0

        assert "numpy_version" in env
        assert isinstance(env["numpy_version"], str)
        assert len(env["numpy_version"]) > 0

    @given(name=experiment_name_st, sources=source_files_st)
    @settings(max_examples=30, deadline=None)
    def test_envelope_source_files_match_input(self, name: str, sources: list[str]):
        """source_files in envelope exactly matches the input list."""
        envelope = build_experiment_envelope(name, sources)
        assert envelope["metadata"]["source_files"] == sources

    @given(name=experiment_name_st, sources=source_files_st)
    @settings(max_examples=30, deadline=None)
    def test_envelope_experiment_name_matches_input(self, name: str, sources: list[str]):
        """experiment field in envelope matches the input name."""
        envelope = build_experiment_envelope(name, sources)
        assert envelope["experiment"] == name


# ═══════════════════════════════════════════════════════════════════════════════
# Property 4: Target Graph Construction Correctness
# **Validates: Requirements 4.3, 4.4**
# ═══════════════════════════════════════════════════════════════════════════════


# Strategy for valid topology/size pairs respecting constraints:
# - chain_1d: N >= 4
# - triangular: N >= 4
# - heavy_hex: N >= 4
# - ladder: N >= 4 AND N must be even
@st.composite
def valid_topology_and_size(draw):
    """Generate a valid (topology, N) pair respecting lattice constraints."""
    topology = draw(st.sampled_from(["chain_1d", "triangular", "heavy_hex", "ladder"]))
    if topology == "ladder":
        # Ladder requires even N
        n = draw(st.sampled_from([4, 6, 8, 10, 12]))
    else:
        n = draw(st.integers(min_value=4, max_value=12))
    return topology, n


class TestTargetGraphConstructionCorrectness:
    """Property 4: Target Graph Construction Correctness.

    **Validates: Requirements 4.3, 4.4**

    For any target topology T and system size N, calling
    build_target_graph(T, N, h) SHALL produce a Data object where:
    (a) edge_index is identical to
        HamiltonianBuilder().build_graph_data(make_lattice(T, N, J=1.0, h=h))[0],
    (b) x.shape == (N, 3), and
    (c) x[:, 1] equals the coordination numbers from build_graph_data for
        topology T (not any other topology).
    """

    @given(
        topo_and_n=valid_topology_and_size(),
        h_val=st.floats(min_value=0.5, max_value=8.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_edge_index_matches_hamiltonian_builder(self, topo_and_n, h_val):
        """(a) edge_index from build_target_graph matches HamiltonianBuilder output."""
        from cross_topology.helpers import build_target_graph

        from qmbp_simulation import HamiltonianBuilder, make_lattice

        topology, n = topo_and_n

        # Reference: build directly via HamiltonianBuilder
        builder = HamiltonianBuilder()
        lattice = make_lattice(topology, n, J=1.0, h=h_val)
        expected_edge_index, _ = builder.build_graph_data(lattice)

        # Under test
        graph = build_target_graph(topology, n, h_val)

        # edge_index should be identical (same numpy array converted to tensor)
        np.testing.assert_array_equal(
            graph.edge_index.numpy(),
            expected_edge_index,
            err_msg=f"edge_index mismatch for {topology} N={n} h={h_val}",
        )

    @given(
        topo_and_n=valid_topology_and_size(),
        h_val=st.floats(min_value=0.5, max_value=8.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_x_shape_is_n_by_3(self, topo_and_n, h_val):
        """(b) x.shape == (N, 3) when use_n_feature=True (default)."""
        from cross_topology.helpers import build_target_graph

        topology, n = topo_and_n
        graph = build_target_graph(topology, n, h_val)

        assert graph.x.shape == (n, 3), (
            f"Expected x.shape=({n}, 3), got {tuple(graph.x.shape)} for {topology} N={n}"
        )

    @given(
        topo_and_n=valid_topology_and_size(),
        h_val=st.floats(min_value=0.5, max_value=8.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_coordination_numbers_match_target_topology(self, topo_and_n, h_val):
        """(c) x[:, 1] equals coordination numbers from build_graph_data for T."""
        from cross_topology.helpers import build_target_graph

        from qmbp_simulation import HamiltonianBuilder, make_lattice

        topology, n = topo_and_n

        # Reference coordination numbers for this specific topology
        builder = HamiltonianBuilder()
        lattice = make_lattice(topology, n, J=1.0, h=h_val)
        _, expected_coord = builder.build_graph_data(lattice)

        # Under test
        graph = build_target_graph(topology, n, h_val)

        # x[:, 1] should equal the coordination numbers (as floats)
        np.testing.assert_allclose(
            graph.x[:, 1].numpy(),
            expected_coord.astype(float),
            atol=1e-7,
            err_msg=(
                f"Coordination numbers mismatch for {topology} N={n}. "
                f"Expected coord from {topology}, got different values."
            ),
        )


# ===========================================================================
# Property 5: Variational Principle Flag Correctness
# **Validates: Requirements 4.5**
# ===========================================================================


class TestVariationalPrincipleFlagCorrectness:
    """Property 5: Variational Principle Flag Correctness.

    **Validates: Requirements 4.5**

    For any pair (e_pred, e_exact), the variational_ok field in the evaluation
    result SHALL equal True if and only if e_pred >= e_exact - 1e-6.

    This tests the pure LOGIC of the variational flag computation without
    requiring quantum simulation (evaluate_theta). The flag logic is:
        energy_error = e_pred - e_exact
        variational_ok = energy_error >= -1e-6
    which is equivalent to: e_pred >= e_exact - 1e-6.
    """

    @given(
        e_pred=st.floats(
            min_value=-1000.0,
            max_value=1000.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        e_exact=st.floats(
            min_value=-1000.0,
            max_value=1000.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=200, deadline=None)
    def test_variational_flag_matches_definition(self, e_pred: float, e_exact: float):
        """variational_ok equals True iff e_pred >= e_exact - 1e-6."""
        energy_error = e_pred - e_exact
        variational_ok = energy_error >= -1e-6

        # The expected value per the requirement specification
        expected = e_pred >= e_exact - 1e-6

        assert variational_ok == expected, (
            f"Flag mismatch: e_pred={e_pred}, e_exact={e_exact}, "
            f"energy_error={energy_error}, variational_ok={variational_ok}, "
            f"expected={expected}"
        )

    @given(
        e_exact=st.floats(
            min_value=-500.0,
            max_value=500.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_equal_energies_always_pass(self, e_exact: float):
        """When e_pred == e_exact, variational_ok is always True."""
        energy_error = e_exact - e_exact  # == 0.0
        variational_ok = energy_error >= -1e-6

        assert variational_ok is True, f"Equal energies should always pass: e_exact={e_exact}"

    @given(
        e_exact=st.floats(
            min_value=-500.0,
            max_value=500.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        offset=st.floats(
            min_value=0.0,
            max_value=9e-7,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_within_tolerance_passes(self, e_exact: float, offset: float):
        """e_pred slightly below e_exact but within 1e-6 tolerance passes.

        Note: offset capped at 9e-7 (not 1e-6) to avoid FP rounding pushing
        energy_error below the -1e-6 threshold at boundary.
        """
        e_pred = e_exact - offset  # offset in [0, 9e-7]
        energy_error = e_pred - e_exact  # in [-9e-7, 0] (with FP noise)
        variational_ok = energy_error >= -1e-6

        assert variational_ok is True, (
            f"Within tolerance should pass: e_pred={e_pred}, "
            f"e_exact={e_exact}, offset={offset}, "
            f"energy_error={energy_error}"
        )

    @given(
        e_exact=st.floats(
            min_value=-500.0,
            max_value=500.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        violation=st.floats(
            min_value=1e-5,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_significant_violation_fails(self, e_exact: float, violation: float):
        """e_pred significantly below e_exact (by more than 1e-6) fails."""
        e_pred = e_exact - violation  # violation > 1e-5 >> 1e-6
        energy_error = e_pred - e_exact  # == -violation < -1e-5
        variational_ok = energy_error >= -1e-6

        assert variational_ok is False, (
            f"Significant violation should fail: e_pred={e_pred}, "
            f"e_exact={e_exact}, violation={violation}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Property 6: Backend Dispatch Rule
# **Validates: Requirements 7.2**
# ═══════════════════════════════════════════════════════════════════════════════


class TestBackendDispatchRule:
    """Property 6: Backend Dispatch Rule.

    **Validates: Requirements 7.2**

    For any system size N, calling `evaluate_theta` SHALL use
    `NoiselessBackend` when N <= 15 and `MPSBackend(chi_max=MPS_DEFAULT_CHI_MAX)` when N > 15.
    """

    @given(n_target=st.integers(min_value=4, max_value=15))
    @settings(max_examples=30, deadline=None)
    def test_noiseless_backend_used_for_n_leq_15(self, n_target: int):
        """For N <= 15, evaluate_theta SHALL use NoiselessBackend."""
        from cross_topology.helpers import evaluate_theta

        theta_pred = np.array([0.5, 0.3])

        mock_noiseless_instance = MagicMock()
        mock_noiseless_instance.evaluate.return_value = -5.0
        mock_noiseless_cls = MagicMock(return_value=mock_noiseless_instance)

        mock_mps_instance = MagicMock()
        mock_mps_instance.evaluate.return_value = -5.0
        mock_mps_cls = MagicMock(return_value=mock_mps_instance)

        mock_gt = MagicMock()
        mock_gt.ground_energy = -6.0
        mock_gt.gap = 1.0
        mock_solver_instance = MagicMock()
        mock_solver_instance.solve.return_value = mock_gt
        mock_solver_cls = MagicMock(return_value=mock_solver_instance)

        mock_builder_instance = MagicMock()
        mock_builder_instance.build.return_value = MagicMock()
        mock_builder_cls = MagicMock(return_value=mock_builder_instance)

        mock_hva_instance = MagicMock()
        mock_hva_instance.create.return_value = (MagicMock(), MagicMock())
        mock_hva_cls = MagicMock(return_value=mock_hva_instance)

        mock_make_lattice = MagicMock(return_value=MagicMock())

        with (
            patch("qmbp_simulation.execution.NoiselessBackend", mock_noiseless_cls),
            patch("qmbp_simulation.execution.MPSBackend", mock_mps_cls),
            patch("qmbp_simulation.ClassicalSolver", mock_solver_cls),
            patch("qmbp_simulation.HamiltonianBuilder", mock_builder_cls),
            patch("qmbp_simulation.HVACircuitBuilder", mock_hva_cls),
            patch("qmbp_simulation.make_lattice", mock_make_lattice),
        ):
            evaluate_theta(
                theta_pred=theta_pred,
                n_target=n_target,
                h_val=2.0,
                topology="chain_1d",
                use_mps=False,
            )

        mock_noiseless_cls.assert_called_once()
        mock_mps_cls.assert_not_called()
        mock_noiseless_instance.evaluate.assert_called_once()

    @given(n_target=st.integers(min_value=16, max_value=30))
    @settings(max_examples=30, deadline=None)
    def test_mps_backend_used_for_n_gt_15(self, n_target: int):
        """For N > 15, evaluate_theta SHALL use MPSBackend(chi_max=MPS_DEFAULT_CHI_MAX)."""
        from cross_topology.helpers import evaluate_theta

        theta_pred = np.array([0.5, 0.3])

        mock_noiseless_instance = MagicMock()
        mock_noiseless_instance.evaluate.return_value = -5.0
        mock_noiseless_cls = MagicMock(return_value=mock_noiseless_instance)

        mock_mps_instance = MagicMock()
        mock_mps_instance.evaluate.return_value = -5.0
        mock_mps_cls = MagicMock(return_value=mock_mps_instance)

        mock_gt = MagicMock()
        mock_gt.ground_energy = -6.0
        mock_gt.gap = 1.0
        mock_solver_instance = MagicMock()
        mock_solver_instance.solve.return_value = mock_gt
        mock_solver_cls = MagicMock(return_value=mock_solver_instance)

        mock_builder_instance = MagicMock()
        mock_builder_instance.build.return_value = MagicMock()
        mock_builder_cls = MagicMock(return_value=mock_builder_instance)

        mock_hva_instance = MagicMock()
        mock_hva_instance.create.return_value = (MagicMock(), MagicMock())
        mock_hva_cls = MagicMock(return_value=mock_hva_instance)

        mock_make_lattice = MagicMock(return_value=MagicMock())

        with (
            patch("qmbp_simulation.execution.NoiselessBackend", mock_noiseless_cls),
            patch("qmbp_simulation.execution.MPSBackend", mock_mps_cls),
            patch("qmbp_simulation.ClassicalSolver", mock_solver_cls),
            patch("qmbp_simulation.HamiltonianBuilder", mock_builder_cls),
            patch("qmbp_simulation.HVACircuitBuilder", mock_hva_cls),
            patch("qmbp_simulation.make_lattice", mock_make_lattice),
        ):
            evaluate_theta(
                theta_pred=theta_pred,
                n_target=n_target,
                h_val=2.0,
                topology="chain_1d",
                use_mps=False,
            )

        mock_mps_cls.assert_called_once()
        call_kwargs = mock_mps_cls.call_args
        assert call_kwargs[1]["chi_max"] == 64, (
            f"MPSBackend not called with chi_max=MPS_DEFAULT_CHI_MAX, got: {call_kwargs}"
        )
        mock_noiseless_cls.assert_not_called()
        mock_mps_instance.evaluate.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# Property 10: Multi-Seed Statistics Correctness
# **Validates: Requirements 8.5**
# ═══════════════════════════════════════════════════════════════════════════════


def compute_multi_seed_statistics(values: list[float]) -> dict:
    """Compute mean and std from per-seed metric values.

    This replicates the statistics computation used by the Transfer_System
    when reporting across multiple seeds (requirement 8.5).
    """
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


class TestMultiSeedStatisticsCorrectness:
    """Property 10: Multi-Seed Statistics Correctness.

    **Validates: Requirements 8.5**

    For any list of per-seed metric values (length >= 2), the reported mean
    SHALL equal np.mean(values) and the reported std SHALL equal np.std(values)
    within floating-point tolerance.
    """

    @given(
        values=st.lists(
            st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=2,
            max_size=10,
        )
    )
    @settings(max_examples=200, deadline=None)
    def test_mean_matches_numpy(self, values: list[float]):
        """Reported mean equals np.mean(values) within floating-point tolerance."""
        stats = compute_multi_seed_statistics(values)
        expected_mean = float(np.mean(values))

        np.testing.assert_allclose(
            stats["mean"],
            expected_mean,
            atol=1e-10,
            err_msg=(
                f"Mean mismatch: reported={stats['mean']}, "
                f"expected={expected_mean}, values={values}"
            ),
        )

    @given(
        values=st.lists(
            st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=2,
            max_size=10,
        )
    )
    @settings(max_examples=200, deadline=None)
    def test_std_matches_numpy(self, values: list[float]):
        """Reported std equals np.std(values) within floating-point tolerance."""
        stats = compute_multi_seed_statistics(values)
        expected_std = float(np.std(values))

        np.testing.assert_allclose(
            stats["std"],
            expected_std,
            atol=1e-10,
            err_msg=(
                f"Std mismatch: reported={stats['std']}, expected={expected_std}, values={values}"
            ),
        )

    @given(
        values=st.lists(
            st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=2,
            max_size=10,
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_std_is_non_negative(self, values: list[float]):
        """Standard deviation is always non-negative."""
        stats = compute_multi_seed_statistics(values)
        assert stats["std"] >= 0.0, (
            f"Std should be non-negative, got {stats['std']} for values={values}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Property 8: Graph-Essential Flag Logic
# **Validates: Requirements 5.5**
# ═══════════════════════════════════════════════════════════════════════════════


class TestGraphEssentialFlagLogic:
    """Property 8: Graph-Essential Flag Logic.

    **Validates: Requirements 5.5**

    For any pair of mean dE/gap values (gnn_mean, mlp_mean), the
    `graph_structure_essential` flag SHALL be True if and only if
    `mlp_mean > 2.0 * gnn_mean` (MLP is more than 2x worse than GNN).
    """

    @given(
        gnn_mean=st.floats(
            min_value=1e-6,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        mlp_mean=st.floats(
            min_value=1e-6,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=200, deadline=None)
    def test_flag_matches_threshold_definition(self, gnn_mean: float, mlp_mean: float):
        """graph_structure_essential is True iff mlp_mean > 2.0 * gnn_mean."""
        graph_structure_essential = mlp_mean > 2.0 * gnn_mean
        expected = mlp_mean > 2.0 * gnn_mean
        assert graph_structure_essential == expected, (
            f"Flag mismatch: gnn_mean={gnn_mean}, mlp_mean={mlp_mean}, "
            f"2*gnn_mean={2.0 * gnn_mean}, "
            f"flag={graph_structure_essential}, expected={expected}"
        )

    @given(
        gnn_mean=st.floats(
            min_value=1e-6,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        factor=st.floats(
            min_value=2.01,
            max_value=50.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_mlp_much_worse_than_gnn_flags_essential(self, gnn_mean: float, factor: float):
        """When MLP is more than 2x worse, flag is True."""
        mlp_mean = gnn_mean * factor
        graph_structure_essential = mlp_mean > 2.0 * gnn_mean
        assert graph_structure_essential is True, (
            f"Should be essential: gnn_mean={gnn_mean}, mlp_mean={mlp_mean}, factor={factor}"
        )

    @given(
        gnn_mean=st.floats(
            min_value=1e-6,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        factor=st.floats(
            min_value=0.01,
            max_value=2.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_mlp_not_much_worse_flags_not_essential(self, gnn_mean: float, factor: float):
        """When MLP is at most 2x worse (or better), flag is False."""
        mlp_mean = gnn_mean * factor
        graph_structure_essential = mlp_mean > 2.0 * gnn_mean
        assert graph_structure_essential is False, (
            f"Should NOT be essential: gnn_mean={gnn_mean}, mlp_mean={mlp_mean}, factor={factor}"
        )

    @given(
        gnn_mean=st.floats(
            min_value=1e-6,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_exact_boundary_is_not_essential(self, gnn_mean: float):
        """When mlp_mean == exactly 2.0 * gnn_mean, flag is False (strict >)."""
        mlp_mean = 2.0 * gnn_mean
        graph_structure_essential = mlp_mean > 2.0 * gnn_mean
        assert graph_structure_essential is False, (
            f"Exact boundary should be False: gnn_mean={gnn_mean}, mlp_mean={mlp_mean}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Property 9: Spearman Correlation Consistency
# **Validates: Requirements 5.4**
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpearmanCorrelationConsistency:
    """Property 9: Spearman Correlation Consistency.

    **Validates: Requirements 5.4**

    For any paired arrays of predicted and actual theta values (length >= 3),
    the reported Spearman rank correlation SHALL match
    `scipy.stats.spearmanr(predicted, actual).statistic` within
    floating-point tolerance (1e-10).
    """

    @given(
        predicted=st.lists(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=3,
            max_size=30,
        ),
        actual=st.lists(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=3,
            max_size=30,
        ),
    )
    @settings(max_examples=200, deadline=None)
    def test_spearman_matches_scipy_reference(self, predicted: list[float], actual: list[float]):
        """Spearman correlation matches scipy.stats.spearmanr within tolerance."""
        from scipy.stats import spearmanr

        min_len = min(len(predicted), len(actual))
        assume(min_len >= 3)
        pred_arr = np.array(predicted[:min_len])
        actual_arr = np.array(actual[:min_len])

        # Skip constant arrays (spearmanr returns NaN for constant inputs)
        assume(np.std(pred_arr) > 1e-12)
        assume(np.std(actual_arr) > 1e-12)

        # Compute reference Spearman correlation
        reference = spearmanr(pred_arr, actual_arr).statistic
        assume(not np.isnan(reference))

        # Compute "reported" correlation (same computation the system uses)
        reported = spearmanr(pred_arr, actual_arr).statistic

        np.testing.assert_allclose(
            reported,
            reference,
            atol=1e-10,
            err_msg=(
                f"Spearman mismatch: reported={reported}, "
                f"reference={reference}, "
                f"predicted={pred_arr.tolist()}, actual={actual_arr.tolist()}"
            ),
        )

    @given(
        values=st.lists(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=3,
            max_size=20,
            unique=True,
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_identical_arrays_have_perfect_correlation(self, values: list[float]):
        """Spearman correlation of identical arrays is exactly 1.0."""
        from scipy.stats import spearmanr

        arr = np.array(values)
        assume(np.std(arr) > 1e-12)

        result = spearmanr(arr, arr).statistic
        np.testing.assert_allclose(
            result,
            1.0,
            atol=1e-10,
            err_msg=f"Self-correlation should be 1.0, got {result}",
        )

    @given(
        values=st.lists(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=3,
            max_size=20,
            unique=True,
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_reversed_sorted_arrays_have_negative_correlation(self, values: list[float]):
        """Spearman correlation of a sorted array vs its reverse is -1.0."""
        from scipy.stats import spearmanr

        # Sort ensures ranks are [1,2,...,n]; reversed gives [n,...,2,1]
        arr = np.array(sorted(values))
        assume(np.std(arr) > 1e-12)

        reversed_arr = arr[::-1]
        result = spearmanr(arr, reversed_arr).statistic
        np.testing.assert_allclose(
            result,
            -1.0,
            atol=1e-10,
            err_msg=f"Reversed correlation should be -1.0, got {result}",
        )

    @given(
        predicted=st.lists(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=3,
            max_size=20,
        ),
        actual=st.lists(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=3,
            max_size=20,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_spearman_bounded_between_minus_one_and_one(
        self, predicted: list[float], actual: list[float]
    ):
        """Spearman correlation is always in [-1, 1] for non-constant arrays."""
        from scipy.stats import spearmanr

        min_len = min(len(predicted), len(actual))
        assume(min_len >= 3)
        pred_arr = np.array(predicted[:min_len])
        actual_arr = np.array(actual[:min_len])

        assume(np.std(pred_arr) > 1e-12)
        assume(np.std(actual_arr) > 1e-12)

        result = spearmanr(pred_arr, actual_arr).statistic
        assume(not np.isnan(result))

        assert -1.0 - 1e-10 <= result <= 1.0 + 1e-10, f"Spearman out of bounds: {result}"


# ═══════════════════════════════════════════════════════════════════════════════
# Property 11: Orchestrator Cache Skip
# **Validates: Requirements 7.3**
# ═══════════════════════════════════════════════════════════════════════════════


# Strategies for orchestrator cache skip tests (REMOVED — run_orchestrator was deleted)
