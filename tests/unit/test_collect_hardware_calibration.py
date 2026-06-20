"""Unit tests for _collect_hardware_calibration()."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from scripts.experiment_runners.hardware.run_mitigation_benchmark import (
    _collect_hardware_calibration,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_props(t1_vals, t2_vals, cx_errors, readout_errors, last_update=None):
    """Create a mock backend properties object."""
    props = MagicMock()
    props.t1 = lambda q: t1_vals.get(q)
    props.t2 = lambda q: t2_vals.get(q)
    props.gate_error = lambda gate, qubits: cx_errors.get(tuple(qubits))
    props.readout_error = lambda q: readout_errors.get(q)
    props.last_update_date = last_update
    return props


def _make_mock_backend(props, layout_qubits=None, layout_cx_pairs=None):
    """Create a mock backend with properties() and layout attributes."""
    backend = MagicMock()
    backend.properties.return_value = props
    backend.layout_qubits = layout_qubits or []
    backend.layout_cx_pairs = layout_cx_pairs or []
    return backend


def _make_mock_job(execution_time=42.5):
    """Create a mock job with metrics() matching IBM Runtime API.

    IBM Runtime job.metrics() returns:
    {
        "usage": {"quantum_seconds": <float>, "seconds": <float>},
        "timestamps": {"created": <iso>, "running": <iso>, "finished": <iso>}
    }
    """
    job = MagicMock()
    job.metrics.return_value = {
        "usage": {
            "quantum_seconds": execution_time,
            "seconds": execution_time * 1.1,  # billed slightly more
        },
        "timestamps": {
            "created": "2026-06-18T10:00:00Z",
            "running": "2026-06-18T10:00:05Z",
            "finished": "2026-06-18T10:01:00Z",
        },
    }
    return job


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCollectHardwareCalibrationHappyPath:
    """Test normal operation with complete data."""

    def test_returns_dict_with_expected_keys(self):
        """All six expected keys are present."""
        cal_time = datetime.now(UTC) - timedelta(hours=2)
        props = _make_mock_props(
            t1_vals={0: 100e-6, 1: 120e-6, 2: 90e-6},
            t2_vals={0: 80e-6, 1: 100e-6, 2: 70e-6},
            cx_errors={(0, 1): 0.005, (1, 2): 0.008},
            readout_errors={0: 0.01, 1: 0.02, 2: 0.015},
            last_update=cal_time,
        )
        backend = _make_mock_backend(
            props, layout_qubits=[0, 1, 2], layout_cx_pairs=[(0, 1), (1, 2)]
        )
        job = _make_mock_job(execution_time=55.0)

        result = _collect_hardware_calibration(backend, job)

        expected_keys = {
            "t1_mean_layout",
            "t2_mean_layout",
            "cx_error_mean_layout",
            "readout_error_mean",
            "calibration_age_hours",
            "job_qpu_seconds",
            "job_usage_details",
            "job_timestamps",
            "queue_wait_s",
        }
        assert result is not None
        assert set(result.keys()) == expected_keys

    def test_t1_t2_converted_to_microseconds(self):
        """T1/T2 values (in seconds from props) are converted to μs."""
        # props.t1(q) returns seconds, multiply by 1e6 for μs
        cal_time = datetime.now(UTC) - timedelta(hours=1)
        props = _make_mock_props(
            t1_vals={0: 100e-6, 1: 200e-6},  # 100 μs, 200 μs in seconds
            t2_vals={0: 50e-6, 1: 150e-6},
            cx_errors={(0, 1): 0.01},
            readout_errors={0: 0.02, 1: 0.03},
            last_update=cal_time,
        )
        backend = _make_mock_backend(props, layout_qubits=[0, 1], layout_cx_pairs=[(0, 1)])
        job = _make_mock_job()

        result = _collect_hardware_calibration(backend, job)

        # mean of [100e-6, 200e-6] = 150e-6, times 1e6 = 150.0 μs
        assert result["t1_mean_layout"] == pytest.approx(150.0)
        # mean of [50e-6, 150e-6] = 100e-6, times 1e6 = 100.0 μs
        assert result["t2_mean_layout"] == pytest.approx(100.0)

    def test_cx_error_mean(self):
        """CX error is the mean of gate errors for layout CX pairs."""
        cal_time = datetime.now(UTC)
        props = _make_mock_props(
            t1_vals={0: 100e-6, 1: 100e-6, 2: 100e-6},
            t2_vals={0: 80e-6, 1: 80e-6, 2: 80e-6},
            cx_errors={(0, 1): 0.004, (1, 2): 0.006},
            readout_errors={0: 0.01, 1: 0.01, 2: 0.01},
            last_update=cal_time,
        )
        backend = _make_mock_backend(
            props, layout_qubits=[0, 1, 2], layout_cx_pairs=[(0, 1), (1, 2)]
        )
        job = _make_mock_job()

        result = _collect_hardware_calibration(backend, job)

        assert result["cx_error_mean_layout"] == pytest.approx(0.005)

    def test_readout_error_mean(self):
        """Readout error is averaged over layout qubits."""
        cal_time = datetime.now(UTC)
        props = _make_mock_props(
            t1_vals={0: 100e-6, 1: 100e-6},
            t2_vals={0: 80e-6, 1: 80e-6},
            cx_errors={(0, 1): 0.01},
            readout_errors={0: 0.02, 1: 0.04},
            last_update=cal_time,
        )
        backend = _make_mock_backend(props, layout_qubits=[0, 1], layout_cx_pairs=[(0, 1)])
        job = _make_mock_job()

        result = _collect_hardware_calibration(backend, job)

        assert result["readout_error_mean"] == pytest.approx(0.03)

    def test_calibration_age_hours(self):
        """Calibration age computed from last_update_date."""
        cal_time = datetime.now(UTC) - timedelta(hours=3, minutes=30)
        props = _make_mock_props(
            t1_vals={0: 100e-6},
            t2_vals={0: 80e-6},
            cx_errors={},
            readout_errors={0: 0.01},
            last_update=cal_time,
        )
        backend = _make_mock_backend(props, layout_qubits=[0], layout_cx_pairs=[])
        job = _make_mock_job()

        result = _collect_hardware_calibration(backend, job)

        assert result["calibration_age_hours"] == pytest.approx(3.5, abs=0.01)

    def test_job_execution_time(self):
        """Job QPU seconds extracted from job.metrics().usage.quantum_seconds."""
        cal_time = datetime.now(UTC)
        props = _make_mock_props(
            t1_vals={0: 100e-6},
            t2_vals={0: 80e-6},
            cx_errors={},
            readout_errors={0: 0.01},
            last_update=cal_time,
        )
        backend = _make_mock_backend(props, layout_qubits=[0], layout_cx_pairs=[])
        job = _make_mock_job(execution_time=123.4)

        result = _collect_hardware_calibration(backend, job)

        assert result["job_qpu_seconds"] == 123.4
        assert result["job_usage_details"]["quantum_seconds"] == 123.4
        assert result["job_usage_details"]["seconds"] == pytest.approx(123.4 * 1.1)
        # Queue wait: 5 seconds between created and running timestamps
        assert result["queue_wait_s"] == pytest.approx(5.0)
        assert result["job_timestamps"]["created"] == "2026-06-18T10:00:00Z"


class TestCollectHardwareCalibrationEdgeCases:
    """Test edge cases and graceful degradation."""

    def test_properties_returns_none(self):
        """Returns dict with None values when properties() returns None."""
        backend = MagicMock()
        backend.properties.return_value = None
        backend.layout_qubits = [0, 1, 2]
        job = _make_mock_job()

        result = _collect_hardware_calibration(backend, job)

        assert result is not None
        assert all(v is None for v in result.values())

    def test_layout_qubits_empty(self):
        """Returns dict with None values when layout_qubits is empty."""
        props = _make_mock_props({}, {}, {}, {})
        backend = _make_mock_backend(props, layout_qubits=[], layout_cx_pairs=[])
        job = _make_mock_job()

        result = _collect_hardware_calibration(backend, job)

        assert result is not None
        assert all(v is None for v in result.values())

    def test_no_layout_qubits_attribute(self):
        """Returns dict with None values when backend has no layout_qubits attr."""
        backend = MagicMock(spec=[])
        backend.properties = MagicMock(return_value=MagicMock())
        # No layout_qubits attribute
        job = _make_mock_job()

        result = _collect_hardware_calibration(backend, job)

        assert result is not None
        assert all(v is None for v in result.values())

    def test_some_t1_none(self):
        """Handles qubits where T1 returns None (skipped from mean)."""
        cal_time = datetime.now(UTC)
        props = _make_mock_props(
            t1_vals={0: 100e-6, 1: None, 2: 200e-6},
            t2_vals={0: 80e-6, 1: 80e-6, 2: 80e-6},
            cx_errors={},
            readout_errors={0: 0.01, 1: 0.01, 2: 0.01},
            last_update=cal_time,
        )
        backend = _make_mock_backend(props, layout_qubits=[0, 1, 2], layout_cx_pairs=[])
        job = _make_mock_job()

        result = _collect_hardware_calibration(backend, job)

        # Only qubits 0 and 2 have T1 values
        assert result["t1_mean_layout"] == pytest.approx(150.0)

    def test_no_cx_pairs(self):
        """cx_error_mean_layout is None when no CX pairs defined."""
        cal_time = datetime.now(UTC)
        props = _make_mock_props(
            t1_vals={0: 100e-6},
            t2_vals={0: 80e-6},
            cx_errors={},
            readout_errors={0: 0.01},
            last_update=cal_time,
        )
        backend = _make_mock_backend(props, layout_qubits=[0], layout_cx_pairs=[])
        job = _make_mock_job()

        result = _collect_hardware_calibration(backend, job)

        assert result["cx_error_mean_layout"] is None

    def test_job_metrics_raises(self):
        """Handles job.metrics() raising an exception gracefully."""
        cal_time = datetime.now(UTC)
        props = _make_mock_props(
            t1_vals={0: 100e-6},
            t2_vals={0: 80e-6},
            cx_errors={},
            readout_errors={0: 0.01},
            last_update=cal_time,
        )
        backend = _make_mock_backend(props, layout_qubits=[0], layout_cx_pairs=[])
        job = MagicMock()
        job.metrics.side_effect = RuntimeError("Job not complete")

        result = _collect_hardware_calibration(backend, job)

        assert result["job_qpu_seconds"] is None
        assert result["job_usage_details"] is None
        assert result["job_timestamps"] is None
        assert result["queue_wait_s"] is None

    def test_last_update_date_none(self):
        """calibration_age_hours is None when last_update_date is None."""
        props = _make_mock_props(
            t1_vals={0: 100e-6},
            t2_vals={0: 80e-6},
            cx_errors={},
            readout_errors={0: 0.01},
            last_update=None,
        )
        backend = _make_mock_backend(props, layout_qubits=[0], layout_cx_pairs=[])
        job = _make_mock_job()

        result = _collect_hardware_calibration(backend, job)

        assert result["calibration_age_hours"] is None
