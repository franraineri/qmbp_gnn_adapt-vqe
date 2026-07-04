"""Test that ValidationRunner saves partial results on KeyboardInterrupt.

Verifies:
1. Results from completed sections are preserved in the JSON.
2. The JSON has 'interrupted' and 'completed_sections' fields.
3. The structured log is also saved.
4. Normal completion still works unchanged.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from qmbp_simulation.framework.runner_base import Section, ValidationRunner


class _InterruptingRunner(ValidationRunner):
    """Test runner that interrupts during section 2."""

    runner_id = "test_interrupt"
    experiment_id = "test_interrupt"
    description = "Test interrupt handling"
    hypothesis = "Partial results are saved on Ctrl+C"

    def define_sections(self):
        return [
            Section(id=1, name="Fast Section", fn=self.section_fast),
            Section(id=2, name="Slow Section (interrupted)", fn=self.section_slow),
            Section(id=3, name="Never Reached", fn=self.section_never),
        ]

    def section_fast(self):
        return {"value": 42, "pass": True}

    def section_slow(self):
        raise KeyboardInterrupt("simulated Ctrl+C")

    def section_never(self):
        return {"pass": True}

    def build_config(self):
        return {
            "system": {"n_qubits": 4, "p_layers": 1},
            "seeds": [42],
        }

    def run_preflight(self):
        return True

    def setup(self):
        pass


class _NormalRunner(ValidationRunner):
    """Test runner that completes normally."""

    runner_id = "test_normal"
    experiment_id = "test_normal"
    description = "Test normal completion"
    hypothesis = "Results are saved normally"

    def define_sections(self):
        return [
            Section(id=1, name="Section A", fn=self.section_a),
            Section(id=2, name="Section B", fn=self.section_b),
        ]

    def section_a(self):
        return {"data": "hello", "pass": True}

    def section_b(self):
        return {"data": "world", "pass": True}

    def build_config(self):
        return {
            "system": {"n_qubits": 4, "p_layers": 1},
            "seeds": [42],
        }

    def run_preflight(self):
        return True

    def setup(self):
        pass


@pytest.fixture
def output_dir(tmp_path):
    """Redirect experiment output to a temp directory."""
    return tmp_path


def _make_runner(cls, output_dir, monkeypatch):
    """Create a runner instance with mocked CLI args and save function."""
    saved_files = []

    def _mock_save(envelope, experiment_id):
        path = output_dir / f"run_test_{experiment_id}.json"
        with open(path, "w") as f:
            json.dump(envelope, f, indent=2, default=str)
        saved_files.append(path)
        return path

    monkeypatch.setattr(
        "qmbp_simulation.framework.runner_base.save_experiment_result",
        _mock_save,
    )

    # Bypass CLI arg parsing by patching sys.argv
    monkeypatch.setattr("sys.argv", ["test_runner", "--skip-preflight"])

    runner = cls()
    return runner, saved_files


def test_interrupt_saves_partial_results(output_dir, monkeypatch):
    """KeyboardInterrupt during section 2 saves section 1 results."""
    runner, saved_files = _make_runner(_InterruptingRunner, output_dir, monkeypatch)

    exit_code = runner.run()

    # Should save a file
    assert len(saved_files) == 1
    result_path = saved_files[0]
    assert result_path.exists()

    # Load and verify
    with open(result_path) as f:
        data = json.load(f)

    # Verify interrupted flag
    assert data["interrupted"] is True
    assert data["completed_sections"] == 1  # Only section 1 completed

    # Verify section 1 data is preserved
    assert "section_1" in data["results"]
    s1 = data["results"]["section_1"]
    assert s1["success"] is True
    assert s1["data"]["value"] == 42

    # Verify section 2 and 3 are NOT in results (never completed)
    assert "section_2" not in data["results"]
    assert "section_3" not in data["results"]

    # Exit code should be 1 (interrupted)
    assert exit_code == 1


def test_normal_completion_unchanged(output_dir, monkeypatch):
    """Normal run still saves all sections without interrupted flag."""
    runner, saved_files = _make_runner(_NormalRunner, output_dir, monkeypatch)

    exit_code = runner.run()

    assert len(saved_files) == 1
    with open(saved_files[0]) as f:
        data = json.load(f)

    # No interrupted flag
    assert "interrupted" not in data

    # Both sections present
    assert "section_1" in data["results"]
    assert "section_2" in data["results"]
    assert data["results"]["section_1"]["success"] is True
    assert data["results"]["section_2"]["success"] is True

    # Exit code 0
    assert exit_code == 0



def test_sigterm_triggers_graceful_save(output_dir, monkeypatch):
    """SIGTERM during execution triggers the same partial save as Ctrl+C."""
    import os
    import signal
    import threading

    class _SigtermRunner(ValidationRunner):
        runner_id = "test_sigterm"
        experiment_id = "test_sigterm"
        description = "Test SIGTERM handling"
        hypothesis = "SIGTERM saves partial results"

        def define_sections(self):
            return [
                Section(id=1, name="Fast", fn=self.section_fast),
                Section(id=2, name="Gets killed", fn=self.section_killed),
            ]

        def section_fast(self):
            return {"data": "saved", "pass": True}

        def section_killed(self):
            # Send SIGTERM to self after a tiny delay
            os.kill(os.getpid(), signal.SIGTERM)
            # This line should not execute (SIGTERM raises KeyboardInterrupt)
            return {"pass": True}  # pragma: no cover

        def build_config(self):
            return {"system": {"n_qubits": 4}, "seeds": [42]}

        def run_preflight(self):
            return True

        def setup(self):
            pass

    runner, saved_files = _make_runner(_SigtermRunner, output_dir, monkeypatch)
    exit_code = runner.run()

    assert len(saved_files) == 1
    with open(saved_files[0]) as f:
        data = json.load(f)

    assert data["interrupted"] is True
    assert data["completed_sections"] == 1
    assert "section_1" in data["results"]
    assert data["results"]["section_1"]["data"]["data"] == "saved"
    assert exit_code == 1


def test_save_failure_still_saves_log(output_dir, monkeypatch):
    """If JSON save fails, the structured log is still saved."""
    log_saved = []

    def _failing_save(envelope, experiment_id):
        raise RuntimeError("Simulated disk full")

    monkeypatch.setattr(
        "qmbp_simulation.framework.runner_base.save_experiment_result",
        _failing_save,
    )
    monkeypatch.setattr("sys.argv", ["test_runner", "--skip-preflight"])

    runner = _NormalRunner()

    # Patch slog.save to track if it's called
    original_save = runner.slog.save

    def _track_log_save(path):
        log_saved.append(path)
        # Don't actually write (path might not exist)

    runner.slog.save = _track_log_save

    exit_code = runner.run()

    # Log save was attempted (even though result save failed)
    assert len(log_saved) == 1
    # Exit code is 0 (sections passed, save failure is logged but not fatal)
    assert exit_code == 0


def test_interrupted_section_id_recorded(output_dir, monkeypatch):
    """The interrupted_section field identifies which section was running."""
    runner, saved_files = _make_runner(_InterruptingRunner, output_dir, monkeypatch)
    runner.run()

    with open(saved_files[0]) as f:
        data = json.load(f)

    # Section 2 was the one being executed when interrupt hit
    assert data["interrupted_section"] == 2



class _FailingRunner(ValidationRunner):
    """Test runner where section 2 returns pass=False."""

    runner_id = "test_failing"
    experiment_id = "test_failing"
    description = "Test failure detail saving"
    hypothesis = "Failed sections are detailed in summary"

    def define_sections(self):
        return [
            Section(id=1, name="Passes", fn=self.section_pass),
            Section(id=2, name="Fails", fn=self.section_fail),
        ]

    def section_pass(self):
        return {"value": 1, "pass": True}

    def section_fail(self):
        return {"pass": False, "reason": "VQE did not converge"}

    def build_config(self):
        return {"system": {"n_qubits": 6}, "seeds": [42]}

    def run_preflight(self):
        return True

    def setup(self):
        pass


def test_failed_sections_detailed_in_summary(output_dir, monkeypatch):
    """When a section returns pass=False, summary includes failure details."""
    runner, saved_files = _make_runner(_FailingRunner, output_dir, monkeypatch)
    exit_code = runner.run()

    assert exit_code == 1
    with open(saved_files[0]) as f:
        data = json.load(f)

    summary = data["summary"]
    assert summary["n_failed"] == 1
    assert summary["all_passed"] is False

    # Verify failed_sections detail
    assert "failed_sections" in summary
    fs = summary["failed_sections"]
    assert len(fs) == 1
    assert fs[0]["section_id"] == 2
    assert fs[0]["name"] == "Fails"
    assert fs[0]["reason"] == "section returned pass=False"



def test_resume_skips_completed_sections(output_dir, monkeypatch):
    """--resume loads completed sections and skips them on re-run."""
    # First: create a "partial" result file that has section_1 completed
    partial_result = {
        "timestamp": "2026-07-03T12:00:00",
        "config": {"system": {"n_qubits": 4}, "seeds": [42]},
        "results": {
            "section_1": {
                "name": "Passes",
                "success": True,
                "elapsed_s": 1.0,
                "data": {"value": 99, "pass": True},
                "error": None,
            },
            "section_2": {
                "name": "Was interrupted",
                "success": False,
                "elapsed_s": 0.5,
                "data": {},
                "error": "KeyboardInterrupt",
            },
        },
        "summary": {"n_sections": 1, "n_passed": 1, "n_failed": 1},
    }
    partial_path = output_dir / "partial_run.json"
    with open(partial_path, "w") as f:
        json.dump(partial_result, f)

    class _ResumableRunner(ValidationRunner):
        runner_id = "test_resume"
        experiment_id = "test_resume"
        description = "Test resume"
        hypothesis = "Resume skips completed"
        sections_executed: list = []

        def define_sections(self):
            return [
                Section(id=1, name="Passes", fn=self.section_1),
                Section(id=2, name="Now works", fn=self.section_2),
            ]

        def section_1(self):
            self.sections_executed.append(1)
            return {"value": 99, "pass": True}

        def section_2(self):
            self.sections_executed.append(2)
            return {"result": "fixed", "pass": True}

        def build_config(self):
            return {"system": {"n_qubits": 4}, "seeds": [42]}

        def run_preflight(self):
            return True

        def setup(self):
            pass

    saved_files = []

    def _mock_save(envelope, experiment_id):
        path = output_dir / f"run_resumed_{experiment_id}.json"
        with open(path, "w") as f:
            json.dump(envelope, f, indent=2, default=str)
        saved_files.append(path)
        return path

    monkeypatch.setattr(
        "qmbp_simulation.framework.runner_base.save_experiment_result",
        _mock_save,
    )
    monkeypatch.setattr("sys.argv", [
        "test_runner", "--skip-preflight", "--resume", str(partial_path)
    ])

    runner = _ResumableRunner()
    exit_code = runner.run()

    # Section 1 should NOT have been executed (it was resumed)
    assert 1 not in runner.sections_executed
    # Section 2 SHOULD have been executed
    assert 2 in runner.sections_executed

    # Result should include both sections
    with open(saved_files[0]) as f:
        data = json.load(f)
    assert "section_1" in data["results"]
    assert "section_2" in data["results"]
    assert data["results"]["section_1"]["data"]["value"] == 99  # From resume
    assert data["results"]["section_2"]["data"]["result"] == "fixed"  # From new run
    assert exit_code == 0
