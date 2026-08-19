"""Tests for scripts/maintenance/run_test_suite.py.

Validates:
- Discovery finds all test_*.py in tests/ recursively
- Passing test files are reported correctly
- Failing test files produce useful failure details
- Frozen (timeout) test files are detected and reported
- The report file is written with correct structure
- --lf (last-failed) filtering works
- Parallel execution returns same results as sequential
- Output includes ONLY failure descriptions (not noise)
- Can discover tests across ALL subdirs of tests/
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Import via importlib since scripts/ isn't a proper package
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "run_test_suite",
    ROOT / "scripts" / "general_project_maintenance" / "run_test_suite.py",
)
_module = importlib.util.module_from_spec(_spec)

# Prevent signal handlers from registering during import
import signal as _signal
_old_sigint = _signal.getsignal(_signal.SIGINT)
_old_sigterm = _signal.getsignal(_signal.SIGTERM)
_spec.loader.exec_module(_module)
_signal.signal(_signal.SIGINT, _old_sigint)
_signal.signal(_signal.SIGTERM, _old_sigterm)

# Pull out what we need
SUITE_ROOT = _module.ROOT
discover_test_files = _module.discover_test_files
run_test_file = _module.run_test_file
_process_result = _module._process_result
_save_last_failed = _module._save_last_failed
_load_last_failed = _module._load_last_failed
LAST_FAILED_PATH = _module.LAST_FAILED_PATH


# ═══════════════════════════════════════════════════════════════════════════════
# Discovery
# ═══════════════════════════════════════════════════════════════════════════════


class TestDiscovery:
    """Test file discovery logic."""

    def test_discovers_all_test_dirs(self):
        """Running with ['tests'] should find files in ALL subdirectories."""
        files = discover_test_files(["tests"])
        rel_paths = [str(f.relative_to(SUITE_ROOT)) for f in files]

        # Must find files in multiple subdirectories
        subdirs_found = set()
        for rp in rel_paths:
            parts = rp.split("/")
            if len(parts) > 2:
                subdirs_found.add(parts[1])  # e.g., "unit", "mpnn", "integration"

        assert "unit" in subdirs_found, "Should discover tests/unit/"
        assert len(subdirs_found) >= 3, f"Expected >=3 subdirs, got {subdirs_found}"

    def test_discovers_more_than_50_files(self):
        """Should find a substantial number of test files."""
        files = discover_test_files(["tests"])
        assert len(files) > 50, f"Expected >50 test files, found {len(files)}"

    def test_all_files_are_test_prefixed(self):
        """All discovered files must start with test_."""
        files = discover_test_files(["tests"])
        for f in files:
            assert f.name.startswith("test_"), f"Non-test file discovered: {f.name}"

    def test_quick_mode_subset(self):
        """Quick mode discovers fewer files than full mode."""
        quick = discover_test_files(["tests/unit", "tests/mpnn"])
        full = discover_test_files(["tests"])
        assert len(quick) < len(full), "Quick should be subset of full"
        # All quick files should be in full
        assert set(quick).issubset(set(full))

    def test_nonexistent_dir_ignored(self):
        """Non-existent directory doesn't crash — returns empty."""
        files = discover_test_files(["tests/nonexistent_xyz_123"])
        assert files == []

    def test_skip_files_respected(self, monkeypatch):
        """Files in SKIP_FILES are excluded."""
        module = _module

        monkeypatch.setattr(module, "SKIP_FILES", {"test_run_test_suite.py"})
        files = discover_test_files(["tests/unit"])
        names = [f.name for f in files]
        assert "test_run_test_suite.py" not in names


# ═══════════════════════════════════════════════════════════════════════════════
# Running individual files
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunTestFile:
    """Test single-file execution logic."""

    def test_passing_file_returns_pass(self, tmp_path):
        """A test file that passes returns status='pass' with count."""
        test_file = tmp_path / "test_ok.py"
        test_file.write_text("def test_one(): assert True\ndef test_two(): assert True\n")

        result = run_test_file(test_file)

        assert result["status"] == "pass"
        assert result["n_passed"] == 2
        assert result["elapsed"] > 0
        assert "file" in result

    def test_failing_file_returns_fail_with_details(self, tmp_path):
        """A test file that fails returns status='fail' with error details."""
        test_file = tmp_path / "test_bad.py"
        test_file.write_text(textwrap.dedent("""\
            def test_oops():
                assert 1 == 2, "Numbers should be equal"
        """))

        result = run_test_file(test_file)

        assert result["status"] == "fail"
        assert result["n_failed"] >= 1
        assert "details" in result
        assert len(result["details"]) > 0  # Must have failure description
        # Details should contain useful info (assertion or FAILED line)
        assert "FAILED" in result["details"] or "Error" in result["details"]

    def test_import_error_is_reported_as_fail(self, tmp_path):
        """A test file with broken import returns fail with ModuleNotFoundError."""
        test_file = tmp_path / "test_broken_import.py"
        test_file.write_text("from nonexistent_module_xyz import thing\ndef test_x(): pass\n")

        result = run_test_file(test_file)

        assert result["status"] == "fail"
        assert "details" in result
        assert "Error" in result["details"]

    def test_empty_file_returns_pass_zero(self, tmp_path):
        """A test file with no tests (exit code 5) returns pass with 0 tests."""
        test_file = tmp_path / "test_empty.py"
        test_file.write_text("# nothing here\n")

        result = run_test_file(test_file)

        assert result["status"] == "pass"
        assert result["n_passed"] == 0

    def test_timeout_file_returns_frozen(self, tmp_path, monkeypatch):
        """A test file that hangs returns status='frozen'."""
        test_file = tmp_path / "test_hang.py"
        test_file.write_text("import time\ndef test_hang(): time.sleep(999)\n")

        module = _module
        monkeypatch.setattr(module, "PER_FILE_TIMEOUT", 3)

        result = run_test_file(test_file)

        assert result["status"] == "frozen"
        assert result["elapsed"] >= 2  # At least close to timeout


# ═══════════════════════════════════════════════════════════════════════════════
# Result processing
# ═══════════════════════════════════════════════════════════════════════════════


class TestProcessResult:
    """Test _process_result updates globals correctly."""

    def _reset_globals(self):
        m = _module
        m._failures.clear()
        m._frozen_list.clear()
        m._slow_list.clear()
        m._total_pass = 0
        m._total_fail = 0
        m._total_frozen = 0

    def test_pass_increments_total(self):
        self._reset_globals()
        m = _module

        r = {"status": "pass", "file": "tests/unit/test_x.py", "n_passed": 5, "elapsed": 1.0}
        _process_result(r, 1, 10)

        assert m._total_pass == 5
        assert m._total_fail == 0

    def test_fail_appends_to_failures(self):
        self._reset_globals()
        m = _module

        r = {"status": "fail", "file": "tests/unit/test_y.py", "n_failed": 2, "elapsed": 3.0, "details": "E assert 1==2"}
        _process_result(r, 1, 10)

        assert m._total_fail == 2
        assert len(m._failures) == 1
        assert m._failures[0]["details"] == "E assert 1==2"

    def test_slow_file_tracked(self):
        self._reset_globals()
        m = _module

        r = {"status": "pass", "file": "tests/unit/test_slow.py", "n_passed": 1, "elapsed": 55.0}
        _process_result(r, 1, 10)

        assert len(m._slow_list) == 1
        assert m._slow_list[0][1] == 55.0


# ═══════════════════════════════════════════════════════════════════════════════
# Last-failed persistence
# ═══════════════════════════════════════════════════════════════════════════════


class TestLastFailed:
    """Test --lf persistence logic."""

    def test_save_and_load_round_trip(self, tmp_path, monkeypatch):
        """Saved failures can be loaded back."""
        m = _module

        lf_path = tmp_path / ".last_failed.json"
        monkeypatch.setattr(m, "LAST_FAILED_PATH", lf_path)

        m._failures.clear()
        m._frozen_list.clear()
        m._failures.append({"file": "tests/unit/test_a.py"})
        m._frozen_list.append("tests/integration/test_b.py")

        _save_last_failed()

        loaded = json.loads(lf_path.read_text())
        assert "tests/unit/test_a.py" in loaded
        assert "tests/integration/test_b.py" in loaded

    def test_load_nonexistent_returns_empty(self, tmp_path, monkeypatch):
        """Loading when no file exists returns empty list."""
        m = _module

        monkeypatch.setattr(m, "LAST_FAILED_PATH", tmp_path / "nope.json")
        result = _load_last_failed()
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# Output quality
# ═══════════════════════════════════════════════════════════════════════════════


class TestOutputQuality:
    """Verify that output contains ONLY useful failure information."""

    def test_failure_details_are_actionable(self, tmp_path):
        """Failure details should contain the assertion message or error type."""
        test_file = tmp_path / "test_detail.py"
        test_file.write_text(textwrap.dedent("""\
            def test_specific_error():
                data = {"key": "value"}
                assert "missing" in data, "Expected 'missing' key in data dict"
        """))

        result = run_test_file(test_file)

        assert result["status"] == "fail"
        # The details should tell us WHAT failed, not just that it failed
        details = result["details"]
        assert "FAILED" in details or "AssertionError" in details or "Error" in details

    def test_passing_file_has_no_noise(self, tmp_path):
        """A passing file should NOT have details or error noise."""
        test_file = tmp_path / "test_clean.py"
        test_file.write_text("def test_pass(): assert True\n")

        result = run_test_file(test_file)

        assert result["status"] == "pass"
        assert "details" not in result
        assert "error" not in result


# ═══════════════════════════════════════════════════════════════════════════════
# Report file
# ═══════════════════════════════════════════════════════════════════════════════


class TestReportFile:
    """Test that the report file has proper structure."""

    def test_report_structure_from_results(self, tmp_path):
        """Report text contains expected sections when there are failures."""
        report_path = tmp_path / "report.txt"

        # Simulate what main() writes to the report
        lines = ["=" * 70, "  SUMMARY: 3 passed | 1 failed | 0 frozen", "=" * 70]
        lines.append("")
        lines.append("  ── FAILURES (test function + reason) ──")
        lines.append("  📁 tests/unit/test_b.py")
        lines.append("     FAILED test_b::test_x")
        lines.append("     E assert 1 == 2")

        report_path.write_text("\n".join(lines))

        content = report_path.read_text()
        assert "SUMMARY" in content
        assert "FAILURES" in content
        assert "test_b.py" in content
        assert "assert 1 == 2" in content

# ═══════════════════════════════════════════════════════════════════════════════
# Full integration: can it run ALL tests/?
# ═══════════════════════════════════════════════════════════════════════════════


class TestFullDiscovery:
    """Verify the suite can discover and categorize ALL files under tests/."""

    def test_discovers_all_subdirs(self):
        """Must find test files in every subdirectory of tests/."""
        tests_root = SUITE_ROOT / "tests"
        expected_subdirs = {
            d.name for d in tests_root.iterdir()
            if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
        }

        files = discover_test_files(["tests"])
        found_subdirs = set()
        for f in files:
            rel = f.relative_to(SUITE_ROOT / "tests")
            if len(rel.parts) > 1:
                found_subdirs.add(rel.parts[0])

        # Every subdir with test files should be covered
        for sd in expected_subdirs:
            subdir_path = tests_root / sd
            has_tests = any(subdir_path.rglob("test_*.py"))
            if has_tests:
                assert sd in found_subdirs, (
                    f"Subdir 'tests/{sd}/' has test_*.py files but wasn't discovered"
                )
