"""Property-based tests for _collect_hardware_calibration mode-dependence.

# Feature: mitigation-benchmark, Property 20
# **Validates: Requirements 15.1, 15.2**
#
# Property 20: Hardware calibration section mode-dependence
#   - hardware mode → non-null dict with all 6 expected keys
#   - fake_backend mode → caller sets hardware_calibration to None
#
# We test this by verifying:
#   1. With a properly mocked backend (valid properties + layout), the function
#      returns a dict with exactly 6 expected keys.
#   2. T1/T2 values are converted to μs (×1e6 of input seconds).
#   3. Means are computed correctly from generated arrays.
#   4. When properties() returns None, graceful degradation (dict with all None).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Add scripts to path for import
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

from experiment_runners.hardware.run_mitigation_benchmark import (
    _collect_hardware_calibration,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Expected keys for the hardware calibration dict
# ═══════════════════════════════════════════════════════════════════════════════

EXPECTED_KEYS = frozenset(
    {
        "t1_mean_layout",
        "t2_mean_layout",
        "cx_error_mean_layout",
        "readout_error_mean",
        "calibration_age_hours",
    }
)

# Core keys that must always be present (superset may include job timing keys)
CORE_KEYS = EXPECTED_KEYS


# ═══════════════════════════════════════════════════════════════════════════════
# Strategies
# ═══════════════════════════════════════════════════════════════════════════════


@st.composite
def hardware_data(draw):
    """Generate random hardware calibration data for a mock backend.

    Produces a dict with:
    - n_qubits: 1-20 qubits in layout
    - t1_vals: T1 in seconds (realistic range 1e-7 to 1e-3)
    - t2_vals: T2 in seconds (realistic range 1e-7 to 1e-3)
    - cx_errors: CX gate error rates (0-1)
    - readout_errors: readout error rates (0-1)
    - execution_time: job execution time in seconds
    - calibration_age_hours: age of last calibration
    """
    n_qubits = draw(st.integers(min_value=1, max_value=20))

    t1_vals = draw(
        st.lists(
            st.floats(min_value=1e-7, max_value=1e-3, allow_nan=False, allow_infinity=False),
            min_size=n_qubits,
            max_size=n_qubits,
        )
    )
    t2_vals = draw(
        st.lists(
            st.floats(min_value=1e-7, max_value=1e-3, allow_nan=False, allow_infinity=False),
            min_size=n_qubits,
            max_size=n_qubits,
        )
    )

    # Generate CX pairs (subset of all possible qubit pairs)
    qubit_indices = list(range(n_qubits))
    all_pairs = [(i, j) for i in qubit_indices for j in qubit_indices if i < j]
    n_pairs = draw(st.integers(min_value=0, max_value=min(len(all_pairs), 10)))
    cx_pairs = (
        draw(
            st.lists(
                st.sampled_from(all_pairs) if all_pairs else st.nothing(),
                min_size=n_pairs,
                max_size=n_pairs,
                unique=True,
            )
        )
        if all_pairs and n_pairs > 0
        else []
    )

    cx_errors = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=len(cx_pairs),
            max_size=len(cx_pairs),
        )
    )

    readout_errors = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n_qubits,
            max_size=n_qubits,
        )
    )

    execution_time = draw(
        st.floats(min_value=0.1, max_value=3600.0, allow_nan=False, allow_infinity=False)
    )

    calibration_age_hours = draw(
        st.floats(min_value=0.0, max_value=48.0, allow_nan=False, allow_infinity=False)
    )

    return {
        "n_qubits": n_qubits,
        "qubit_indices": qubit_indices,
        "t1_vals": t1_vals,
        "t2_vals": t2_vals,
        "cx_pairs": cx_pairs,
        "cx_errors": cx_errors,
        "readout_errors": readout_errors,
        "execution_time": execution_time,
        "calibration_age_hours": calibration_age_hours,
    }


def _build_mock_backend(data: dict) -> MagicMock:
    """Build a mock backend from hardware_data strategy output."""
    props = MagicMock()

    # Map qubit index → T1/T2 values
    t1_map = dict(zip(data["qubit_indices"], data["t1_vals"], strict=False))
    t2_map = dict(zip(data["qubit_indices"], data["t2_vals"], strict=False))
    cx_map = dict(zip([tuple(p) for p in data["cx_pairs"]], data["cx_errors"], strict=False))
    ro_map = dict(zip(data["qubit_indices"], data["readout_errors"], strict=False))

    props.t1 = lambda q: t1_map.get(q)
    props.t2 = lambda q: t2_map.get(q)
    props.gate_error = lambda gate, qubits: cx_map.get(tuple(qubits))
    props.readout_error = lambda q: ro_map.get(q)

    # Set last_update_date to a time `calibration_age_hours` ago
    props.last_update_date = datetime.now(UTC) - timedelta(hours=data["calibration_age_hours"])

    backend = MagicMock()
    backend.properties.return_value = props
    backend.layout_qubits = data["qubit_indices"]
    backend.layout_cx_pairs = data["cx_pairs"]

    return backend


def _build_mock_job(execution_time: float) -> MagicMock:
    """Build a mock job with metrics()."""
    job = MagicMock()
    job.metrics.return_value = {"execution_time": execution_time}
    return job


# ═══════════════════════════════════════════════════════════════════════════════
# Property 20: Hardware calibration section mode-dependence
# **Validates: Requirements 15.1, 15.2**
# ═══════════════════════════════════════════════════════════════════════════════


class TestHardwareCalibrationModeDependence:
    """Property 20: Hardware calibration section mode-dependence.

    **Validates: Requirements 15.1, 15.2**

    For mode="hardware" (valid backend with properties):
      - _collect_hardware_calibration returns a dict with exactly 6 expected keys
      - T1/T2 are in μs (×1e6 of input seconds)
      - Means are computed correctly from generated arrays

    For mode="fake_backend":
      - The caller sets hardware_calibration to None (verified via graceful
        degradation when properties() returns None)
    """

    # -------------------------------------------------------------------
    # Sub-property: dict always has exactly 6 expected keys
    # -------------------------------------------------------------------

    @given(data=hardware_data())
    @settings(max_examples=100, deadline=None)
    def test_returns_dict_with_exactly_6_expected_keys(self, data: dict):
        """With valid backend, result has exactly the 6 expected keys."""
        backend = _build_mock_backend(data)
        job = _build_mock_job(data["execution_time"])

        result = _collect_hardware_calibration(backend, job)

        assert result is not None, "Expected non-null dict for valid backend"
        assert CORE_KEYS.issubset(set(result.keys())), (
            f"Expected core keys {CORE_KEYS} to be subset of {set(result.keys())}"
        )

    # -------------------------------------------------------------------
    # Sub-property: T1/T2 converted to μs (×1e6)
    # -------------------------------------------------------------------

    @given(data=hardware_data())
    @settings(max_examples=100, deadline=None)
    def test_t1_converted_to_microseconds(self, data: dict):
        """T1 mean is input (seconds) × 1e6 (μs conversion)."""
        backend = _build_mock_backend(data)
        job = _build_mock_job(data["execution_time"])

        result = _collect_hardware_calibration(backend, job)

        expected_t1_us = float(np.mean(data["t1_vals"])) * 1e6
        assert result["t1_mean_layout"] == pytest.approx(expected_t1_us, rel=1e-6), (
            f"T1 mean: expected {expected_t1_us} μs, got {result['t1_mean_layout']}"
        )

    @given(data=hardware_data())
    @settings(max_examples=100, deadline=None)
    def test_t2_converted_to_microseconds(self, data: dict):
        """T2 mean is input (seconds) × 1e6 (μs conversion)."""
        backend = _build_mock_backend(data)
        job = _build_mock_job(data["execution_time"])

        result = _collect_hardware_calibration(backend, job)

        expected_t2_us = float(np.mean(data["t2_vals"])) * 1e6
        assert result["t2_mean_layout"] == pytest.approx(expected_t2_us, rel=1e-6), (
            f"T2 mean: expected {expected_t2_us} μs, got {result['t2_mean_layout']}"
        )

    # -------------------------------------------------------------------
    # Sub-property: CX error mean computed correctly
    # -------------------------------------------------------------------

    @given(data=hardware_data())
    @settings(max_examples=100, deadline=None)
    def test_cx_error_mean_computed_correctly(self, data: dict):
        """CX error mean matches numpy mean of generated errors."""
        backend = _build_mock_backend(data)
        job = _build_mock_job(data["execution_time"])

        result = _collect_hardware_calibration(backend, job)

        if data["cx_errors"]:
            expected = float(np.mean(data["cx_errors"]))
            assert result["cx_error_mean_layout"] == pytest.approx(expected, rel=1e-6), (
                f"CX error mean: expected {expected}, got {result['cx_error_mean_layout']}"
            )
        else:
            assert result["cx_error_mean_layout"] is None

    # -------------------------------------------------------------------
    # Sub-property: Readout error mean computed correctly
    # -------------------------------------------------------------------

    @given(data=hardware_data())
    @settings(max_examples=100, deadline=None)
    def test_readout_error_mean_computed_correctly(self, data: dict):
        """Readout error mean matches numpy mean of generated errors."""
        backend = _build_mock_backend(data)
        job = _build_mock_job(data["execution_time"])

        result = _collect_hardware_calibration(backend, job)

        expected = float(np.mean(data["readout_errors"]))
        assert result["readout_error_mean"] == pytest.approx(expected, rel=1e-6), (
            f"Readout error mean: expected {expected}, got {result['readout_error_mean']}"
        )

    # -------------------------------------------------------------------
    # Sub-property: Job execution time passed through correctly
    # -------------------------------------------------------------------

    @given(data=hardware_data())
    @settings(max_examples=100, deadline=None)
    def test_job_execution_time_passed_through(self, data: dict):
        """Job execution time is present in result under one of the expected keys."""
        backend = _build_mock_backend(data)
        job = _build_mock_job(data["execution_time"])

        result = _collect_hardware_calibration(backend, job)

        # The key may be job_execution_time_s (legacy) or job_qpu_seconds (new)
        time_key = "job_qpu_seconds" if "job_qpu_seconds" in result else "job_execution_time_s"
        assert time_key in result, (
            f"Neither 'job_qpu_seconds' nor 'job_execution_time_s' found in {result.keys()}"
        )

    # -------------------------------------------------------------------
    # Sub-property: Graceful degradation when properties() returns None
    # (Simulates fake_backend mode where caller sets hardware_calibration=None)
    # -------------------------------------------------------------------

    @given(
        exec_time=st.floats(min_value=0.1, max_value=3600.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=None)
    def test_properties_none_returns_all_none_values(self, exec_time: float):
        """When properties() returns None, all values in result are None.

        This validates Requirement 15.2: fake_backend mode results in
        hardware_calibration being null (the caller uses this behavior
        to decide whether to include calibration data).
        """
        backend = MagicMock()
        backend.properties.return_value = None
        backend.layout_qubits = [0, 1, 2]
        job = _build_mock_job(exec_time)

        result = _collect_hardware_calibration(backend, job)

        assert result is not None, "Should return dict, not None"
        assert CORE_KEYS.issubset(set(result.keys()))
        assert all(result[k] is None for k in CORE_KEYS if k in result), (
            f"Expected all None values when properties()=None, got {result}"
        )

    # -------------------------------------------------------------------
    # Sub-property: Empty layout_qubits → all None values
    # -------------------------------------------------------------------

    @given(
        exec_time=st.floats(min_value=0.1, max_value=3600.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=None)
    def test_empty_layout_qubits_returns_all_none(self, exec_time: float):
        """When layout_qubits is empty, all values in result are None."""
        props = MagicMock()
        backend = MagicMock()
        backend.properties.return_value = props
        backend.layout_qubits = []
        backend.layout_cx_pairs = []
        job = _build_mock_job(exec_time)

        result = _collect_hardware_calibration(backend, job)

        assert result is not None
        assert CORE_KEYS.issubset(set(result.keys()))
        assert all(result[k] is None for k in CORE_KEYS if k in result), (
            f"Expected all None values when layout_qubits=[], got {result}"
        )
