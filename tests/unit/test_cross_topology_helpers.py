"""Unit tests for cross_topology.helpers module.

Tests the data adapter (load_source_data), format detection,
and theta canonicalization.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Add experiment_runners to path for cross_topology package imports
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts" / "experiment_runners"))

from cross_topology.helpers import (
    canonicalize_theta,
    detect_format,
    load_source_data,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def scaling_json(tmp_path: Path) -> Path:
    """Create a minimal scaling-format JSON file."""
    data = {
        "metadata": {"n": 6, "topology": "triangular", "p_layers": 1},
        "vqe_results": [
            {
                "seed": 42,
                "results": [
                    {
                        "h": 4.0,
                        "theta_opt": [0.1, 0.3],
                        "dmrg_energy": -10.5,
                    },
                    {
                        "h": 3.5,
                        "theta_opt": [0.15, 0.35],
                        "dmrg_energy": -9.2,
                    },
                ],
            }
        ],
    }
    path = tmp_path / "scaling_test.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def pipeline_run_json(tmp_path: Path) -> Path:
    """Create a minimal pipeline_run-format JSON file."""
    data = {
        "config": {"n_qubits": 10, "topology": "heavy_hex", "h_values": [5.0, 4.5]},
        "diagnostics": {
            "phase1": {"energies": [-20.1, -18.3], "n_points": 2},
            "phase2": {"h_values": [5.0, 4.5]},
        },
        "vqe_results": [
            {
                "seed": 42,
                "results": [
                    {"h": 5.0, "theta_opt": [0.05, 0.4]},
                    {"h": 4.5, "theta_opt": [0.08, 0.38]},
                ],
            }
        ],
    }
    path = tmp_path / "pipeline_run_test.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def pipeline_run_phase12_json(tmp_path: Path) -> Path:
    """Create a pipeline_run with phase12_data (no vqe_results)."""
    data = {
        "config": {"n_qubits": None, "topology": None},
        "system": {"n_qubits": 10, "topology": "ladder"},
        "diagnostics": {"phase1": {}, "phase2": {}},
        "phase12_data": [
            {"h": 4.0, "e_exact": -40.8, "theta_opt": [0.3, 0.01, 0.38, 0.35]},
            {"h": 3.5, "e_exact": -35.9, "theta_opt": [0.33, 0.02, 0.4, 0.34]},
        ],
    }
    path = tmp_path / "pipeline_phase12_test.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def scaling_no_theta_json(tmp_path: Path) -> Path:
    """Scaling file without theta_opt (like older format)."""
    data = {
        "metadata": {"n": 40, "topology": "chain_1d"},
        "vqe_results": [
            {
                "seed": 42,
                "results": [
                    {"h": 5.0, "vqe_energy": -200.0, "dmrg_energy": -199.9},
                ],
            }
        ],
    }
    path = tmp_path / "no_theta.json"
    path.write_text(json.dumps(data))
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: detect_format
# ═══════════════════════════════════════════════════════════════════════════════


class TestDetectFormat:
    def test_scaling_format(self):
        assert detect_format({"vqe_results": [], "metadata": {}}) == "scaling"

    def test_pipeline_run_format(self):
        assert detect_format({"config": {}, "diagnostics": {}}) == "pipeline_run"

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown JSON format"):
            detect_format({"random_key": 42})

    def test_error_includes_keys(self):
        with pytest.raises(ValueError, match="random_key"):
            detect_format({"random_key": 42, "other": 1})


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: canonicalize_theta
# ═══════════════════════════════════════════════════════════════════════════════


class TestCanonicalizeTheta:
    def test_positive_last_unchanged(self):
        theta = np.array([0.1, 0.3])
        result = canonicalize_theta(theta)
        np.testing.assert_allclose(result, theta, atol=1e-14)

    def test_negative_last_flipped(self):
        theta = np.array([0.1, -0.3])
        result = canonicalize_theta(theta)
        np.testing.assert_allclose(result, [-0.1, 0.3], atol=1e-14)

    def test_empty_array(self):
        result = canonicalize_theta(np.array([]))
        assert len(result) == 0

    def test_zero_last_unchanged(self):
        theta = np.array([0.1, 0.0])
        result = canonicalize_theta(theta)
        np.testing.assert_allclose(result, theta, atol=1e-14)

    def test_four_params(self):
        theta = np.array([0.1, 0.2, 0.3, -0.4])
        result = canonicalize_theta(theta)
        np.testing.assert_allclose(result, [-0.1, -0.2, -0.3, 0.4], atol=1e-14)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: load_source_data
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoadSourceData:
    def test_scaling_format_loads(self, scaling_json: Path):
        sd = load_source_data(scaling_json, seed=42)
        assert sd.n == 6
        assert sd.topology == "triangular"
        assert sd.param_dim == 2
        assert sd.h_values.shape == (2,)
        assert sd.theta_opt.shape == (2, 2)
        assert sd.e_exact.shape == (2,)
        np.testing.assert_array_almost_equal(sd.h_values, [4.0, 3.5])
        np.testing.assert_array_almost_equal(sd.e_exact, [-10.5, -9.2])

    def test_pipeline_run_format_loads(self, pipeline_run_json: Path):
        sd = load_source_data(pipeline_run_json, seed=42)
        assert sd.n == 10
        assert sd.topology == "heavy_hex"
        assert sd.param_dim == 2
        assert sd.h_values.shape == (2,)
        assert sd.theta_opt.shape == (2, 2)
        assert sd.e_exact.shape == (2,)
        # h_values from diagnostics.phase2.h_values
        np.testing.assert_array_almost_equal(sd.h_values, [5.0, 4.5])
        # e_exact from diagnostics.phase1.energies
        np.testing.assert_array_almost_equal(sd.e_exact, [-20.1, -18.3])

    def test_pipeline_run_phase12_format_loads(self, pipeline_run_phase12_json: Path):
        sd = load_source_data(pipeline_run_phase12_json, seed=42)
        assert sd.n == 10
        assert sd.topology == "ladder"
        assert sd.param_dim == 4
        assert sd.h_values.shape == (2,)
        assert sd.theta_opt.shape == (2, 4)
        np.testing.assert_array_almost_equal(sd.h_values, [4.0, 3.5])
        np.testing.assert_array_almost_equal(sd.e_exact, [-40.8, -35.9])

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Source file not found"):
            load_source_data(Path("/nonexistent/file.json"))

    def test_missing_theta_opt_raises_value_error(self, scaling_no_theta_json: Path):
        with pytest.raises(ValueError, match="theta_opt"):
            load_source_data(scaling_no_theta_json, seed=42)

    def test_missing_theta_opt_error_includes_path(self, scaling_no_theta_json: Path):
        with pytest.raises(ValueError, match=str(scaling_no_theta_json)):
            load_source_data(scaling_no_theta_json, seed=42)

    def test_theta_canonicalized_on_load(self, tmp_path: Path):
        """Verify that loaded theta values are canonicalized (last component >= 0)."""
        data = {
            "metadata": {"n": 6, "topology": "chain_1d"},
            "vqe_results": [
                {
                    "seed": 42,
                    "results": [
                        {"h": 4.0, "theta_opt": [0.1, -0.3], "dmrg_energy": -10.0},
                    ],
                }
            ],
        }
        path = tmp_path / "neg_theta.json"
        path.write_text(json.dumps(data))
        sd = load_source_data(path, seed=42)
        # Last component should be positive after canonicalization
        assert sd.theta_opt[0, -1] >= 0

    def test_seed_filtering(self, tmp_path: Path):
        """Verify that multi-seed files filter to the requested seed."""
        data = {
            "metadata": {"n": 6, "topology": "chain_1d"},
            "vqe_results": [
                {
                    "seed": 42,
                    "results": [
                        {"h": 4.0, "theta_opt": [0.1, 0.3], "dmrg_energy": -10.0},
                    ],
                },
                {
                    "seed": 43,
                    "results": [
                        {"h": 4.0, "theta_opt": [0.2, 0.4], "dmrg_energy": -10.1},
                    ],
                },
            ],
        }
        path = tmp_path / "multi_seed.json"
        path.write_text(json.dumps(data))
        sd = load_source_data(path, seed=43)
        np.testing.assert_array_almost_equal(sd.theta_opt[0], [0.2, 0.4])
        np.testing.assert_array_almost_equal(sd.e_exact, [-10.1])
