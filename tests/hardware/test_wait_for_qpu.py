"""Quick smoke test for wait_for_qpu_execution logic."""

from unittest.mock import MagicMock

from qmbp_simulation.execution.hardware.submission import wait_for_qpu_execution


class FakeStatus:
    def __init__(self, name):
        self.name = name


def test_local_job_no_status():
    """Local PrimitiveJob without status() should not raise."""
    job = MagicMock(spec=[])
    wait_for_qpu_execution(job, qpu_timeout_s=10)


def test_local_job_with_wait():
    """Local PrimitiveJob with wait_for_final_state but no status."""
    job = MagicMock(spec=["wait_for_final_state"])
    wait_for_qpu_execution(job, qpu_timeout_s=10)
    job.wait_for_final_state.assert_called_once()


def test_job_already_done():
    """Job that is already DONE when we first poll."""
    job = MagicMock()
    job.status.return_value = FakeStatus("DONE")
    wait_for_qpu_execution(job, qpu_timeout_s=10)


def test_job_queued_then_running_then_done():
    """Job transitions QUEUED -> RUNNING -> DONE."""
    statuses = iter(
        [
            FakeStatus("QUEUED"),
            FakeStatus("QUEUED"),
            FakeStatus("RUNNING"),
            FakeStatus("RUNNING"),
            FakeStatus("DONE"),
        ]
    )
    job = MagicMock()
    job.status.side_effect = lambda: next(statuses)
    wait_for_qpu_execution(job, qpu_timeout_s=60, poll_interval_s=0.01)


def test_qpu_timeout_raises():
    """Job stuck in RUNNING should raise TimeoutError."""
    import pytest

    job = MagicMock()
    job.status.return_value = FakeStatus("RUNNING")
    with pytest.raises(TimeoutError, match="QPU execution exceeded"):
        wait_for_qpu_execution(job, qpu_timeout_s=0.05, poll_interval_s=0.01)


def test_no_timeout_waits_indefinitely():
    """With qpu_timeout_s=None after leaving queue, calls wait_for_final_state."""
    statuses = iter(
        [
            FakeStatus("QUEUED"),
            FakeStatus("RUNNING"),
        ]
    )
    job = MagicMock()
    job.status.side_effect = lambda: next(statuses)
    job.wait_for_final_state = MagicMock()
    wait_for_qpu_execution(job, qpu_timeout_s=None, poll_interval_s=0.01)
    job.wait_for_final_state.assert_called_once()


def test_cancelled_in_queue_returns():
    """Job cancelled while queued should return without error."""
    job = MagicMock()
    job.status.return_value = FakeStatus("CANCELLED")
    wait_for_qpu_execution(job, qpu_timeout_s=10)
