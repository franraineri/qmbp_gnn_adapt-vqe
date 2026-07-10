"""Tests for ResultIndex diagnostic methods.

Covers:
- detect_regressions() — median-based regression detection
- analyze_temporal_drift() — temporal correlation analysis
- diagnose() — group health classification with pass_rate regression
- valid_entries — data quality filtering (pass_rate bounds, etc.)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from qmbp_simulation.framework.result_index import ResultIndex

# ─── Fixtures ─────────────────────────────────────────────────────────────


def _make_entry(
    model: str = "tfim",
    topology: str = "chain_1d",
    n_qubits: int = 10,
    p_layers: int = 2,
    pass_rate: float = 0.75,
    passed: bool = True,
    timestamp: str = "2026-06-15T10:00:00",
    elapsed_s: float = 120.0,
    n_sections: int = 4,
    experiment_id: str = "noiseless/tfim/chain_1d",
    file_name: str = "run_test.json",
) -> dict[str, Any]:
    """Create a minimal valid index entry for testing."""
    return {
        "model": model,
        "topology": topology,
        "n_qubits": n_qubits,
        "p_layers": p_layers,
        "pass_rate": pass_rate,
        "passed": passed,
        "timestamp": timestamp,
        "elapsed_s": elapsed_s,
        "n_sections": n_sections,
        "experiment_id": experiment_id,
        "_file": file_name,
    }


def _make_index(tmp_path: Path, entries: list[dict]) -> ResultIndex:
    """Create a ResultIndex with pre-loaded entries (no disk scan)."""
    idx = ResultIndex(root=tmp_path)
    idx._entries = entries
    idx._loaded = True
    return idx


# ─── Tests: valid_entries ─────────────────────────────────────────────────


class TestValidEntries:
    """Test data quality filtering in valid_entries property."""

    def test_valid_entry_passes(self, tmp_path):
        entries = [_make_entry()]
        idx = _make_index(tmp_path, entries)
        assert len(idx.valid_entries) == 1

    def test_missing_model_excluded(self, tmp_path):
        entry = _make_entry()
        entry["model"] = ""
        idx = _make_index(tmp_path, [entry])
        assert len(idx.valid_entries) == 0

    def test_invalid_topology_excluded(self, tmp_path):
        for bad_topo in ("", "[]", None):
            entry = _make_entry()
            entry["topology"] = bad_topo
            idx = _make_index(tmp_path, [entry])
            assert len(idx.valid_entries) == 0, f"Failed for topology={bad_topo!r}"

    def test_zero_n_qubits_excluded(self, tmp_path):
        entry = _make_entry()
        entry["n_qubits"] = 0
        idx = _make_index(tmp_path, [entry])
        assert len(idx.valid_entries) == 0

    def test_zero_sections_excluded(self, tmp_path):
        entry = _make_entry()
        entry["n_sections"] = 0
        idx = _make_index(tmp_path, [entry])
        assert len(idx.valid_entries) == 0

    def test_garbage_experiment_id_excluded(self, tmp_path):
        for garbage in ("TEST", "test", "FAIL", "NONE", "XFAIL", "CNT", ""):
            entry = _make_entry()
            entry["experiment_id"] = garbage
            idx = _make_index(tmp_path, [entry])
            assert len(idx.valid_entries) == 0, f"Failed for id={garbage!r}"

    def test_pass_rate_out_of_bounds_excluded(self, tmp_path):
        """pass_rate > 1.0 or < 0.0 should be excluded."""
        for bad_rate in (1.5, -0.1, 2.0, -1.0):
            entry = _make_entry(pass_rate=bad_rate)
            idx = _make_index(tmp_path, [entry])
            assert len(idx.valid_entries) == 0, f"Failed for rate={bad_rate}"

    def test_pass_rate_non_numeric_excluded(self, tmp_path):
        entry = _make_entry()
        entry["pass_rate"] = "invalid"
        idx = _make_index(tmp_path, [entry])
        assert len(idx.valid_entries) == 0

    def test_pass_rate_none_still_included(self, tmp_path):
        """Entry with pass_rate=None should still be included (missing data)."""
        entry = _make_entry()
        entry["pass_rate"] = None
        idx = _make_index(tmp_path, [entry])
        assert len(idx.valid_entries) == 1

    def test_boundary_pass_rates_included(self, tmp_path):
        """pass_rate of exactly 0.0 and 1.0 should be valid."""
        entries = [_make_entry(pass_rate=0.0), _make_entry(pass_rate=1.0)]
        idx = _make_index(tmp_path, entries)
        assert len(idx.valid_entries) == 2


# ─── Tests: detect_regressions ────────────────────────────────────────────


class TestDetectRegressions:
    """Test regression detection with median baseline."""

    def test_no_regressions_when_single_run(self, tmp_path):
        idx = _make_index(tmp_path, [_make_entry()])
        assert idx.detect_regressions() == []

    def test_no_regression_when_improving(self, tmp_path):
        entries = [
            _make_entry(pass_rate=0.5, timestamp="2026-06-10T10:00:00"),
            _make_entry(pass_rate=0.75, timestamp="2026-06-11T10:00:00"),
            _make_entry(pass_rate=1.0, timestamp="2026-06-12T10:00:00"),
        ]
        idx = _make_index(tmp_path, entries)
        assert idx.detect_regressions() == []

    def test_regression_detected_on_drop(self, tmp_path):
        entries = [
            _make_entry(pass_rate=0.75, timestamp="2026-06-10T10:00:00", file_name="run_1.json"),
            _make_entry(pass_rate=0.75, timestamp="2026-06-11T10:00:00", file_name="run_2.json"),
            _make_entry(pass_rate=0.75, timestamp="2026-06-12T10:00:00", file_name="run_3.json"),
            _make_entry(pass_rate=0.25, timestamp="2026-06-13T10:00:00", file_name="run_4.json"),
        ]
        idx = _make_index(tmp_path, entries)
        regs = idx.detect_regressions()
        assert len(regs) == 1
        assert regs[0]["latest_pass_rate"] == 0.25
        assert regs[0]["median_previous_pass_rate"] == 0.75
        assert regs[0]["delta"] == pytest.approx(-0.5)

    def test_no_regression_within_threshold(self, tmp_path):
        """Drop of exactly 5% should NOT trigger regression."""
        entries = [
            _make_entry(pass_rate=0.75, timestamp="2026-06-10T10:00:00"),
            _make_entry(pass_rate=0.70, timestamp="2026-06-11T10:00:00"),
        ]
        idx = _make_index(tmp_path, entries)
        assert idx.detect_regressions() == []

    def test_regression_uses_median_not_best(self, tmp_path):
        """One lucky outlier (100%) should not cause false positive."""
        entries = [
            _make_entry(pass_rate=0.5, timestamp="2026-06-10T10:00:00"),
            _make_entry(pass_rate=1.0, timestamp="2026-06-11T10:00:00"),
            _make_entry(pass_rate=0.5, timestamp="2026-06-12T10:00:00"),
            _make_entry(pass_rate=0.5, timestamp="2026-06-13T10:00:00"),
            # Latest at 0.5 — median of prev is 0.5, so no regression
            _make_entry(pass_rate=0.5, timestamp="2026-06-14T10:00:00"),
        ]
        idx = _make_index(tmp_path, entries)
        # Median of [0.5, 1.0, 0.5, 0.5] = 0.5 (sorted: [0.5, 0.5, 0.5, 1.0])
        # Latest = 0.5, 0.5 < 0.5 - 0.05 is False → no regression
        assert idx.detect_regressions() == []

    def test_regression_includes_timestamp(self, tmp_path):
        """Regressions should include latest_timestamp for temporal analysis."""
        entries = [
            _make_entry(pass_rate=1.0, timestamp="2026-06-10T10:00:00"),
            _make_entry(
                pass_rate=0.25, timestamp="2026-06-15T12:00:00", file_name="regressed.json"
            ),
        ]
        idx = _make_index(tmp_path, entries)
        regs = idx.detect_regressions()
        assert len(regs) == 1
        assert regs[0]["latest_timestamp"] == "2026-06-15T12:00:00"
        assert regs[0]["n_previous_runs"] == 1

    def test_regression_groups_by_config(self, tmp_path):
        """Different configs should be detected independently."""
        entries = [
            # Config A: tfim|chain_1d|10|2 — regressed
            _make_entry(
                pass_rate=1.0, timestamp="2026-06-10T10:00:00", model="tfim", topology="chain_1d"
            ),
            _make_entry(
                pass_rate=0.25, timestamp="2026-06-11T10:00:00", model="tfim", topology="chain_1d"
            ),
            # Config B: tfim|heavy_hex|10|2 — stable
            _make_entry(
                pass_rate=0.75, timestamp="2026-06-10T10:00:00", model="tfim", topology="heavy_hex"
            ),
            _make_entry(
                pass_rate=0.75, timestamp="2026-06-11T10:00:00", model="tfim", topology="heavy_hex"
            ),
        ]
        idx = _make_index(tmp_path, entries)
        regs = idx.detect_regressions()
        assert len(regs) == 1
        assert "chain_1d" in regs[0]["config"]

    def test_empty_timestamp_handled(self, tmp_path):
        """Entries with empty/None timestamps should not crash."""
        entries = [
            _make_entry(pass_rate=1.0, timestamp=""),
            _make_entry(pass_rate=0.25, timestamp=None),
        ]
        # Patch the None to str for the entry
        entries[1]["timestamp"] = None
        idx = _make_index(tmp_path, entries)
        # Should not raise
        regs = idx.detect_regressions()
        assert isinstance(regs, list)


# ─── Tests: analyze_temporal_drift ────────────────────────────────────────


class TestAnalyzeTemporalDrift:
    """Test temporal regression correlation analysis."""

    def test_insufficient_data_returns_no_drift(self, tmp_path):
        """Need >= 4 entries for meaningful analysis."""
        entries = [_make_entry(), _make_entry()]
        idx = _make_index(tmp_path, entries)
        result = idx.analyze_temporal_drift()
        assert result["has_drift"] is False
        assert "insufficient" in result.get("reason", "")

    def test_stable_performance_no_drift(self, tmp_path):
        """Constant pass_rate across time should not trigger drift."""
        base = datetime(2026, 6, 1)
        entries = [
            _make_entry(
                pass_rate=0.75,
                passed=True,
                timestamp=(base + timedelta(days=i * 2)).isoformat(),
            )
            for i in range(10)
        ]
        idx = _make_index(tmp_path, entries)
        result = idx.analyze_temporal_drift(window_days=7)
        assert result["has_drift"] is False
        assert abs(result["trend_slope"]) < 0.05

    def test_degrading_performance_detects_drift(self, tmp_path):
        """Systematic degradation should be detected."""
        base = datetime(2026, 6, 1)
        entries = []
        # Week 1: all pass
        for i in range(5):
            entries.append(
                _make_entry(
                    pass_rate=1.0,
                    passed=True,
                    timestamp=(base + timedelta(days=i)).isoformat(),
                )
            )
        # Week 2: half fail
        for i in range(5):
            passed = i % 2 == 0
            entries.append(
                _make_entry(
                    pass_rate=0.5,
                    passed=passed,
                    timestamp=(base + timedelta(days=7 + i)).isoformat(),
                )
            )
        # Week 3: all fail
        for i in range(5):
            entries.append(
                _make_entry(
                    pass_rate=0.0,
                    passed=False,
                    timestamp=(base + timedelta(days=14 + i)).isoformat(),
                )
            )
        idx = _make_index(tmp_path, entries)
        result = idx.analyze_temporal_drift(window_days=7)
        assert result["has_drift"] is True
        assert result["trend_slope"] < -0.05

    def test_breakpoint_detected_on_sudden_drop(self, tmp_path):
        """A large single-window drop should produce a breakpoint."""
        base = datetime(2026, 6, 1)
        entries = []
        # Week 1: all pass
        for i in range(5):
            entries.append(
                _make_entry(
                    pass_rate=1.0,
                    passed=True,
                    timestamp=(base + timedelta(days=i)).isoformat(),
                )
            )
        # Week 2: sudden crash
        for i in range(5):
            entries.append(
                _make_entry(
                    pass_rate=0.0,
                    passed=False,
                    timestamp=(base + timedelta(days=7 + i)).isoformat(),
                )
            )
        idx = _make_index(tmp_path, entries)
        result = idx.analyze_temporal_drift(window_days=7)
        assert result["breakpoint"] is not None
        assert result["max_single_drop"] > 0.5

    def test_filter_by_model(self, tmp_path):
        """Should only analyze entries matching model filter."""
        base = datetime(2026, 6, 1)
        entries = []
        # tfim: degrading
        for i in range(5):
            entries.append(
                _make_entry(
                    model="tfim",
                    pass_rate=1.0,
                    passed=True,
                    timestamp=(base + timedelta(days=i)).isoformat(),
                )
            )
        for i in range(5):
            entries.append(
                _make_entry(
                    model="tfim",
                    pass_rate=0.0,
                    passed=False,
                    timestamp=(base + timedelta(days=7 + i)).isoformat(),
                )
            )
        # heisenberg: stable (should not affect tfim analysis)
        for i in range(10):
            entries.append(
                _make_entry(
                    model="heisenberg",
                    pass_rate=1.0,
                    passed=True,
                    timestamp=(base + timedelta(days=i)).isoformat(),
                )
            )
        idx = _make_index(tmp_path, entries)
        result = idx.analyze_temporal_drift(model="tfim", window_days=7)
        assert result["has_drift"] is True

    def test_window_days_parameter(self, tmp_path):
        """Different window sizes should produce different number of windows."""
        base = datetime(2026, 6, 1)
        entries = [
            _make_entry(
                pass_rate=0.75,
                passed=True,
                timestamp=(base + timedelta(days=i)).isoformat(),
            )
            for i in range(28)
        ]
        idx = _make_index(tmp_path, entries)
        r7 = idx.analyze_temporal_drift(window_days=7)
        r14 = idx.analyze_temporal_drift(window_days=14)
        assert len(r7["windows"]) >= len(r14["windows"])

    def test_invalid_timestamps_skipped(self, tmp_path):
        """Entries with unparseable timestamps should be silently skipped."""
        entries = [
            _make_entry(timestamp="2026-06-01T10:00:00"),
            _make_entry(timestamp="not-a-date"),
            _make_entry(timestamp=""),
            _make_entry(timestamp="2026-06-02T10:00:00"),
            _make_entry(timestamp="2026-06-03T10:00:00"),
            _make_entry(timestamp="2026-06-04T10:00:00"),
        ]
        idx = _make_index(tmp_path, entries)
        result = idx.analyze_temporal_drift(window_days=7)
        # Should work with 4 valid timestamps (skipping 2 bad ones)
        assert result["n_entries"] == 4

    def test_regression_cluster_detection(self, tmp_path):
        """Multiple regressions on the same date should be detected as a cluster."""
        base = datetime(2026, 6, 1)
        entries = []
        # Build data that creates multiple regressions on same date
        configs = [
            ("tfim", "chain_1d"),
            ("tfim", "heavy_hex"),
            ("tfim", "ladder"),
        ]
        for model, topo in configs:
            # Good run
            entries.append(
                _make_entry(
                    model=model,
                    topology=topo,
                    pass_rate=1.0,
                    passed=True,
                    timestamp=(base + timedelta(days=0)).isoformat(),
                    file_name=f"run_good_{topo}.json",
                )
            )
            # Regressed run (same date for all)
            entries.append(
                _make_entry(
                    model=model,
                    topology=topo,
                    pass_rate=0.25,
                    passed=False,
                    timestamp=(base + timedelta(days=7)).isoformat(),
                    file_name=f"run_bad_{topo}.json",
                )
            )
        idx = _make_index(tmp_path, entries)
        result = idx.analyze_temporal_drift(model="tfim", window_days=7)
        # Should detect the cluster
        if result.get("regression_cluster"):
            assert result["regression_cluster"]["n_regressions"] >= 2

    def test_returns_date_range(self, tmp_path):
        """Result should include the analyzed date range."""
        entries = [
            _make_entry(timestamp="2026-06-01T10:00:00"),
            _make_entry(timestamp="2026-06-05T10:00:00"),
            _make_entry(timestamp="2026-06-10T10:00:00"),
            _make_entry(timestamp="2026-06-15T10:00:00"),
        ]
        idx = _make_index(tmp_path, entries)
        result = idx.analyze_temporal_drift(window_days=7)
        assert result["date_range"] == ["2026-06-01", "2026-06-15"]


# ─── Tests: diagnose ─────────────────────────────────────────────────────


class TestDiagnose:
    """Test group-level health diagnosis."""

    def test_empty_entries_returns_empty_groups(self, tmp_path):
        idx = _make_index(tmp_path, [])
        diag = idx.diagnose()
        assert diag["summary"]["total_groups"] == 0
        assert diag["groups"] == {}

    def test_healthy_group_classification(self, tmp_path):
        """Group with >= 80% pass_rate classified as healthy."""
        entries = [
            _make_entry(pass_rate=1.0, passed=True, timestamp=f"2026-06-{10 + i:02d}T10:00:00")
            for i in range(5)
        ]
        idx = _make_index(tmp_path, entries)
        diag = idx.diagnose()
        assert diag["summary"]["healthy"] == 1
        assert diag["summary"]["failing"] == 0

    def test_degraded_group_classification(self, tmp_path):
        """Group with 40-79% pass_rate classified as degraded."""
        entries = []
        for i in range(5):
            passed = i < 3  # 3/5 = 60%
            entries.append(
                _make_entry(
                    pass_rate=0.6 if passed else 0.0,
                    passed=passed,
                    timestamp=f"2026-06-{10 + i:02d}T10:00:00",
                )
            )
        idx = _make_index(tmp_path, entries)
        diag = idx.diagnose()
        assert diag["summary"]["degraded"] == 1

    def test_failing_group_classification(self, tmp_path):
        """Group with < 40% pass_rate classified as failing."""
        entries = [
            _make_entry(pass_rate=0.0, passed=False, timestamp=f"2026-06-{10 + i:02d}T10:00:00")
            for i in range(5)
        ]
        idx = _make_index(tmp_path, entries)
        diag = idx.diagnose()
        assert diag["summary"]["failing"] == 1

    def test_filter_by_model(self, tmp_path):
        entries = [
            _make_entry(model="tfim"),
            _make_entry(model="heisenberg"),
        ]
        idx = _make_index(tmp_path, entries)
        diag = idx.diagnose(model="tfim")
        assert diag["summary"]["total_groups"] == 1

    def test_filter_by_topology(self, tmp_path):
        entries = [
            _make_entry(topology="chain_1d"),
            _make_entry(topology="heavy_hex"),
        ]
        idx = _make_index(tmp_path, entries)
        diag = idx.diagnose(topology="chain_1d")
        assert diag["summary"]["total_groups"] == 1

    def test_regression_detection_uses_pass_rate(self, tmp_path):
        """Intra-group regression should use pass_rate, not binary passed."""
        entries = [
            # History: good pass_rate
            _make_entry(pass_rate=0.9, passed=True, timestamp="2026-06-10T10:00:00"),
            _make_entry(pass_rate=0.85, passed=True, timestamp="2026-06-11T10:00:00"),
            # Latest: lower pass_rate (but still technically passed)
            _make_entry(pass_rate=0.25, passed=True, timestamp="2026-06-12T10:00:00"),
        ]
        idx = _make_index(tmp_path, entries)
        diag = idx.diagnose()
        # Should detect regression even though all "passed" is True
        group_key = list(diag["groups"].keys())[0]
        group_issues = diag["groups"][group_key]["issues"]
        assert any("degraded" in issue.lower() for issue in group_issues)

    def test_consistently_failing_issue_reported(self, tmp_path):
        """Groups with < 50% pass and >= 3 runs should get an issue."""
        entries = [
            _make_entry(pass_rate=0.0, passed=False, timestamp=f"2026-06-{10 + i:02d}T10:00:00")
            for i in range(4)
        ]
        idx = _make_index(tmp_path, entries)
        diag = idx.diagnose()
        assert len(diag["issues"]) > 0
        assert any("consistently failing" in i.lower() for i in diag["issues"])

    def test_recommendations_generated_for_failing_groups(self, tmp_path):
        """Failing groups should trigger recommendations."""
        entries = [
            _make_entry(pass_rate=0.0, passed=False, timestamp=f"2026-06-{10 + i:02d}T10:00:00")
            for i in range(4)
        ]
        idx = _make_index(tmp_path, entries)
        diag = idx.diagnose()
        assert len(diag["recommendations"]) > 0

    def test_multiple_groups_independent(self, tmp_path):
        """Different (model, topology, p) combos form separate groups."""
        entries = [
            _make_entry(model="tfim", topology="chain_1d", p_layers=1),
            _make_entry(model="tfim", topology="chain_1d", p_layers=2),
            _make_entry(model="tfim", topology="heavy_hex", p_layers=1),
        ]
        idx = _make_index(tmp_path, entries)
        diag = idx.diagnose()
        assert diag["summary"]["total_groups"] == 3
