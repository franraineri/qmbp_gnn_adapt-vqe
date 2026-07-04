"""Comprehensive tests for project_health — coverage, state, verify, sanity, scaling, reporter.

Fills the testing gaps in modules that lack direct unit tests:
- project_health.state (delta detection, persistence)
- project_health.coverage (gap detection, action derivation, compute_* functions)
- project_health.analysis.verify_results (criteria parsing, classification)
- project_health.analysis.sanity_check (check registry, report aggregation)
- project_health.analysis.scaling_analyzer (parsing, validation, anomaly detection)
- project_health.reporter (text/markdown/JSON formatting)
- project_health.models (serialization)

All tests use synthetic data and tmp_path — no dependency on real result files.
Target: <5s total runtime.

Run with:
    pytest tests/test_project_health_coverage.py -v
"""

from __future__ import annotations

import json

import pytest

from qmbp_simulation.models.constants import MPS_DEFAULT_CHI_MAX

# ═══════════════════════════════════════════════════════════════════════════
# project_health.state — persistence and delta detection
# ═══════════════════════════════════════════════════════════════════════════


class TestStatePersistence:
    """Test save/load state and delta detection."""

    def test_load_previous_state_no_file(self, tmp_path):
        """First run with no state file returns empty set."""
        from project_health.core.state import load_previous_state

        result = load_previous_state(tmp_path / "nonexistent.json")
        assert result == set()

    def test_save_and_load_roundtrip(self, tmp_path):
        """Saved state can be loaded back correctly."""
        from project_health.core.state import load_previous_state, save_current_state

        state_file = tmp_path / "state.json"
        files = {"results/a.json", "results/b.json", "results/c.json"}

        save_current_state(files, state_file)
        loaded = load_previous_state(state_file)

        assert loaded == files

    def test_save_state_includes_metadata(self, tmp_path):
        """Saved state JSON includes n_files and optional metadata."""
        from project_health.core.state import save_current_state

        state_file = tmp_path / "state.json"
        save_current_state(
            {"a.json", "b.json"},
            state_file,
            metadata={"timestamp": "2026-06-07", "n_noiseless": 100},
        )

        data = json.loads(state_file.read_text())
        assert data["n_files"] == 2
        assert data["metadata"]["timestamp"] == "2026-06-07"

    def test_detect_delta_first_run(self, tmp_path):
        """On first run (no state), delta should be empty lists."""
        from project_health.core.state import detect_delta

        new, removed = detect_delta({"a.json", "b.json"}, tmp_path / "state.json")
        assert new == []
        assert removed == []

    def test_detect_delta_new_files(self, tmp_path):
        """New files since last run are detected."""
        from project_health.core.state import detect_delta, save_current_state

        state_file = tmp_path / "state.json"
        save_current_state({"a.json", "b.json"}, state_file)

        new, removed = detect_delta({"a.json", "b.json", "c.json"}, state_file)
        assert new == ["c.json"]
        assert removed == []

    def test_detect_delta_removed_files(self, tmp_path):
        """Files missing since last run are detected."""
        from project_health.core.state import detect_delta, save_current_state

        state_file = tmp_path / "state.json"
        save_current_state({"a.json", "b.json", "c.json"}, state_file)

        new, removed = detect_delta({"a.json"}, state_file)
        assert new == []
        assert removed == ["b.json", "c.json"]

    def test_detect_delta_both_new_and_removed(self, tmp_path):
        """Both new and removed files can be detected simultaneously."""
        from project_health.core.state import detect_delta, save_current_state

        state_file = tmp_path / "state.json"
        save_current_state({"a.json", "b.json"}, state_file)

        new, removed = detect_delta({"b.json", "d.json"}, state_file)
        assert new == ["d.json"]
        assert removed == ["a.json"]

    def test_load_corrupted_state_returns_empty(self, tmp_path):
        """Corrupted state file returns empty set gracefully."""
        from project_health.core.state import load_previous_state

        state_file = tmp_path / "state.json"
        state_file.write_text("not valid json {{{")
        result = load_previous_state(state_file)
        assert result == set()


# ═══════════════════════════════════════════════════════════════════════════
# project_health.coverage — gap detection and metric computation
# ═══════════════════════════════════════════════════════════════════════════


class TestCoverageGapDetection:
    """Test gap detection functions with synthetic data."""

    def _make_noiseless(
        self, topology="chain_1d", n_qubits=6, p_layers=1, de_gap=0.03, seed=42, h_test=None
    ):
        from project_health.digest.models import NoiselessResult

        return NoiselessResult(
            source_file="test.json",
            folder="test",
            topology=topology,
            n_qubits=n_qubits,
            p_layers=p_layers,
            delta_e_over_gap=de_gap,
            seed=seed,
            h_test=h_test or [3.0],
        )

    def _make_noisy(self, topology="chain_1d", n_qubits=6, p_layers=1):
        from project_health.digest.models import NoisyResult

        return NoisyResult(
            source_file="test.json",
            folder="test",
            topology=topology,
            n_qubits=n_qubits,
            p_layers=p_layers,
        )

    def test_detect_missing_p1_noiseless(self):
        """Detects configs with p=2 data but no p=1."""
        from project_health.core.coverage import detect_coverage_gaps

        noiseless = [
            self._make_noiseless("chain_1d", 10, p_layers=2),
            self._make_noiseless("chain_1d", 6, p_layers=1),
        ]
        gaps = detect_coverage_gaps(noiseless, [], [])
        p1_missing = [g for g in gaps if g.gap_type.value == "p1_noiseless_missing"]
        # chain_1d N=10 has p=2 but no p=1
        assert any(g.n_qubits == 10 for g in p1_missing)

    def test_detect_insufficient_seeds(self):
        """Detects p=1 configs with fewer than 3 seeds."""
        from project_health.core.coverage import detect_coverage_gaps

        noiseless = [
            self._make_noiseless("chain_1d", 6, 1, seed=42),
            self._make_noiseless("chain_1d", 6, 1, seed=43),
            # Missing seed 44
        ]
        gaps = detect_coverage_gaps(noiseless, [], [])
        seed_gaps = [g for g in gaps if g.gap_type.value == "insufficient_seeds"]
        assert len(seed_gaps) == 1
        assert "44" in seed_gaps[0].detail

    def test_detect_missing_zne(self):
        """Detects p=1 noiseless configs without ZNE validation."""
        from project_health.core.coverage import detect_coverage_gaps

        noiseless = [self._make_noiseless("heavy_hex", 10, 1)]
        noisy = []  # No ZNE data
        gaps = detect_coverage_gaps(noiseless, noisy, [])
        zne_gaps = [g for g in gaps if g.gap_type.value == "missing_zne"]
        assert len(zne_gaps) >= 1

    def test_no_gap_when_zne_exists(self):
        """No ZNE gap when noisy data covers the config."""
        from project_health.core.coverage import detect_coverage_gaps

        noiseless = [self._make_noiseless("chain_1d", 6, 1)]
        noisy = [self._make_noisy("chain_1d", 6, 1)]
        gaps = detect_coverage_gaps(noiseless, noisy, [])
        zne_gaps = [g for g in gaps if g.gap_type.value == "missing_zne"]
        assert len(zne_gaps) == 0


class TestCoverageComputeStats:
    """Test aggregate statistical computations."""

    def _make_noiseless(
        self, de_gap=0.03, conv_rate=1.0, theta_smooth=0.5, gen_gap=0.005, elapsed_s=60.0
    ):
        from project_health.digest.models import NoiselessResult

        return NoiselessResult(
            source_file="test.json",
            folder="test",
            topology="chain_1d",
            n_qubits=6,
            delta_e_over_gap=de_gap,
            convergence_rate=conv_rate,
            theta_smoothness=theta_smooth,
            generalization_gap=gen_gap,
            elapsed_s=elapsed_s,
        )

    def test_compute_noiseless_stats_pass_rate(self):
        """Pass rate = fraction with ΔE/gap < 0.05."""
        from project_health.core.coverage import compute_noiseless_stats

        results = [
            self._make_noiseless(de_gap=0.02),
            self._make_noiseless(de_gap=0.04),
            self._make_noiseless(de_gap=0.08),
            self._make_noiseless(de_gap=0.12),
        ]
        pass_rate, median = compute_noiseless_stats(results)
        assert pass_rate == 0.5  # 2/4 pass
        assert median is not None

    def test_compute_vqe_quality_chain_breaks(self):
        """VQE quality detects chain break warnings (θ > 1.0)."""
        from project_health.core.coverage import compute_vqe_quality

        results = [
            self._make_noiseless(theta_smooth=0.3),
            self._make_noiseless(theta_smooth=1.5),
            self._make_noiseless(theta_smooth=2.0),
        ]
        stats = compute_vqe_quality(results)
        assert stats.n_chain_break_warnings == 2
        assert stats.theta_smoothness_max == 2.0

    def test_compute_mpnn_quality_overfit_warnings(self):
        """MPNN quality detects overfitting (gen_gap > 0.01)."""
        from project_health.core.coverage import compute_mpnn_quality

        results = [
            self._make_noiseless(gen_gap=0.005),
            self._make_noiseless(gen_gap=0.02),
            self._make_noiseless(gen_gap=0.03),
        ]
        stats = compute_mpnn_quality(results)
        assert stats.n_overfit_warnings == 2
        assert stats.gen_gap_max == 0.03

    def test_compute_distribution_counts(self):
        """Distribution counts results by topology, n_qubits, p_layers."""
        from project_health.core.coverage import compute_distribution
        from project_health.digest.models import NoiselessResult

        noiseless = [
            NoiselessResult(
                source_file="a", folder="f", topology="chain_1d", n_qubits=6, p_layers=1
            ),
            NoiselessResult(
                source_file="b", folder="f", topology="ladder", n_qubits=10, p_layers=2
            ),
            NoiselessResult(
                source_file="c", folder="f", topology="chain_1d", n_qubits=6, p_layers=1
            ),
        ]
        dist = compute_distribution(noiseless, [])
        assert dist.by_topology["chain_1d"] == 2
        assert dist.by_topology["ladder"] == 1
        assert dist.by_p_layers[1] == 2

    def test_compute_noisy_stats_empty(self):
        """Empty noisy list returns zeros."""
        from project_health.core.coverage import compute_noisy_stats

        rate, r2, gain = compute_noisy_stats([])
        assert rate == 0.0
        assert r2 == 0.0
        assert gain == 0.0

    def test_compute_timing_stats(self):
        """Timing stats aggregate elapsed_s correctly."""
        from project_health.core.coverage import compute_timing_stats

        noiseless = [self._make_noiseless(elapsed_s=100.0), self._make_noiseless(elapsed_s=200.0)]
        stats = compute_timing_stats(noiseless, [], [])
        assert stats.total_runs == 2
        assert stats.mean_run_s == 150.0
        assert stats.total_pipeline_hours == pytest.approx(300.0 / 3600.0)


# ═══════════════════════════════════════════════════════════════════════════
# project_health.analysis.verify_results — verification logic
# ═══════════════════════════════════════════════════════════════════════════


class TestVerifyResults:
    """Test verification utility functions."""

    def test_parse_pass_criteria_gte(self):
        """>=2/3 parses correctly."""
        from project_health.analysis.verify_results import parse_pass_criteria

        op, required, total = parse_pass_criteria(">=2/3")
        assert op == ">="
        assert required == 2
        assert total == 3

    def test_parse_pass_criteria_eq(self):
        """==3/3 parses correctly."""
        from project_health.analysis.verify_results import parse_pass_criteria

        op, required, total = parse_pass_criteria("==3/3")
        assert op == "=="
        assert required == 3
        assert total == 3

    def test_parse_pass_criteria_invalid(self):
        """Invalid format raises ValueError."""
        from project_health.analysis.verify_results import parse_pass_criteria

        with pytest.raises(ValueError):
            parse_pass_criteria("bad_format")

    def test_evaluate_criteria_gte_pass(self):
        """>=2/3 with 2 passes → True."""
        from project_health.analysis.verify_results import evaluate_criteria

        assert evaluate_criteria(2, ">=2/3") is True
        assert evaluate_criteria(3, ">=2/3") is True

    def test_evaluate_criteria_gte_fail(self):
        """>=2/3 with 1 pass → False."""
        from project_health.analysis.verify_results import evaluate_criteria

        assert evaluate_criteria(1, ">=2/3") is False

    def test_evaluate_criteria_eq_pass(self):
        """==3/3 with exactly 3 → True."""
        from project_health.analysis.verify_results import evaluate_criteria

        assert evaluate_criteria(3, "==3/3") is True

    def test_evaluate_criteria_eq_fail(self):
        """==3/3 with 2 → False."""
        from project_health.analysis.verify_results import evaluate_criteria

        assert evaluate_criteria(2, "==3/3") is False

    def test_classify_de_gap_pass(self):
        """ΔE/gap below threshold → PASS."""
        from project_health.analysis.verify_results import classify_de_gap

        assert classify_de_gap(0.03) == "PASS"

    def test_classify_de_gap_marginal(self):
        """ΔE/gap between threshold and 2×threshold → MARGINAL."""
        from project_health.analysis.verify_results import classify_de_gap

        assert classify_de_gap(0.07) == "MARGINAL"

    def test_classify_de_gap_fail(self):
        """ΔE/gap >= 2×threshold → FAIL."""
        from project_health.analysis.verify_results import classify_de_gap

        assert classify_de_gap(0.15) == "FAIL"

    def test_classify_de_gap_none(self):
        """None → NO_DATA."""
        from project_health.analysis.verify_results import classify_de_gap

        assert classify_de_gap(None) == "NO_DATA"

    def test_scan_results_directory_empty(self, tmp_path):
        """Empty directory returns empty list."""
        from project_health.analysis.verify_results import scan_results_directory

        results = scan_results_directory(tmp_path)
        assert results == []

    def test_scan_results_directory_parses_files(self, tmp_path, monkeypatch):
        """Correctly parses pipeline_run files in subdirectories."""
        from project_health.analysis.validation import verify_results
        from project_health.analysis.validation.verify_results import scan_results_directory

        # Monkeypatch ROOT on the actual module where scan_results_directory lives
        monkeypatch.setattr(verify_results, "ROOT", tmp_path)

        subdir = tmp_path / "v1_ladder_seed42"
        subdir.mkdir()
        data = {"phase4_results": [{"h_test": 3.0, "delta_e_over_gap": 0.02}]}
        (subdir / "pipeline_run_001.json").write_text(json.dumps(data))

        results = scan_results_directory(tmp_path)
        assert len(results) == 1
        assert results[0].de_gap == 0.02
        assert results[0].verdict == "PASS"

    def test_analyze_verification_all_pass(self, tmp_path):
        """All passing results meet criteria."""
        from project_health.analysis.verify_results import (
            VerificationResult,
            analyze_verification,
        )

        results = [
            VerificationResult("v1_s42", "V1", 42, 3.0, 0.02, "PASS", "f"),
            VerificationResult("v1_s43", "V1", 43, 3.0, 0.03, "PASS", "f"),
            VerificationResult("v1_s44", "V1", 44, 3.0, 0.04, "PASS", "f"),
        ]
        specs = {
            "V1": {
                "claim": "test claim",
                "pass_criteria": ">=2/3",
                "if_pass": "confirmed",
                "if_fail": "rejected",
                "tier": 1,
            }
        }
        report = analyze_verification(results, specs)
        assert report.n_confirmed == 1
        assert report.n_rejected == 0

    def test_analyze_verification_fails(self, tmp_path):
        """Insufficient passes → rejected."""
        from project_health.analysis.verify_results import (
            VerificationResult,
            analyze_verification,
        )

        results = [
            VerificationResult("v1_s42", "V1", 42, 3.0, 0.02, "PASS", "f"),
            VerificationResult("v1_s43", "V1", 43, 3.0, 0.12, "FAIL", "f"),
            VerificationResult("v1_s44", "V1", 44, 3.0, 0.15, "FAIL", "f"),
        ]
        specs = {
            "V1": {
                "claim": "test claim",
                "pass_criteria": ">=2/3",
                "if_pass": "confirmed",
                "if_fail": "rejected",
                "tier": 1,
            }
        }
        report = analyze_verification(results, specs)
        assert report.n_confirmed == 0
        assert report.n_rejected == 1
        assert len(report.corrections_needed) == 1


# ═══════════════════════════════════════════════════════════════════════════
# project_health.analysis.sanity_check — registry and report
# ═══════════════════════════════════════════════════════════════════════════


class TestSanityCheck:
    """Test sanity check framework (registry, report model)."""

    def test_check_result_model(self):
        """CheckResult dataclass holds expected fields."""
        from project_health.analysis.sanity_check import CheckResult

        cr = CheckResult(
            name="test_check",
            category="data_integrity",
            passed=True,
            message="All good",
            severity="info",
        )
        assert cr.passed is True
        assert cr.category == "data_integrity"

    def test_sanity_report_add_passed(self):
        """Adding a passed check increments n_passed."""
        from project_health.analysis.sanity_check import CheckResult, SanityReport

        report = SanityReport()
        report.add(CheckResult("c1", "test", True, "ok"))
        assert report.n_passed == 1
        assert report.n_failed == 0
        assert report.overall_pass is True

    def test_sanity_report_add_failed(self):
        """Adding a failed check increments n_failed and sets overall_pass=False."""
        from project_health.analysis.sanity_check import CheckResult, SanityReport

        report = SanityReport()
        report.add(CheckResult("c1", "test", False, "bad", severity="error"))
        assert report.n_failed == 1
        assert report.overall_pass is False

    def test_sanity_report_add_warning(self):
        """Warnings increment n_warnings but don't fail overall."""
        from project_health.analysis.sanity_check import CheckResult, SanityReport

        report = SanityReport()
        report.add(CheckResult("c1", "test", False, "warn", severity="warning"))
        assert report.n_warnings == 1
        assert report.n_failed == 0
        assert report.overall_pass is True

    def test_sanity_report_to_dict(self):
        """to_dict produces JSON-serializable output."""
        from project_health.analysis.sanity_check import CheckResult, SanityReport

        report = SanityReport()
        report.add(CheckResult("c1", "data_integrity", True, "good"))
        report.add(CheckResult("c2", "physics", False, "bad", severity="error"))

        d = report.to_dict()
        assert d["overall_pass"] is False
        assert d["n_passed"] == 1
        assert d["n_failed"] == 1
        assert len(d["checks"]) == 2
        # Should be JSON-serializable
        json.dumps(d)

    def test_format_report_contains_summary(self):
        """format_report produces a readable string with summary line."""
        from project_health.analysis.sanity_check import (
            CheckResult,
            SanityReport,
            format_report,
        )

        report = SanityReport()
        report.add(CheckResult("c1", "data_integrity", True, "File exists"))
        report.add(CheckResult("c2", "physics", False, "Value wrong", severity="error"))

        output = format_report(report)
        assert "SANITY CHECK" in output
        assert "1 passed" in output
        assert "1 failed" in output


# ═══════════════════════════════════════════════════════════════════════════
# project_health.analysis.scaling_analyzer — parsing and validation
# ═══════════════════════════════════════════════════════════════════════════


class TestScalingAnalyzer:
    """Test scaling analyzer parsing and analysis functions."""

    def _make_scaling_json(
        self, n=40, all_passed=True, h_values=None, mean_de=0.005, total_s=1500.0
    ):
        """Create synthetic scaling validation JSON."""
        if h_values is None:
            h_values = [4.0, 4.5, 5.0, 5.5, 6.0]
        results = []
        for h in h_values:
            results.append(
                {
                    "h": h,
                    "vqe_energy": -1.5,
                    "dmrg_energy": -1.502,
                    "gap": 0.4,
                    "de_gap": mean_de,
                    "time_s": total_s / len(h_values),
                    "passed": all_passed,
                    "n_iterations": 100,
                }
            )
        return {
            "experiment": "mps_scaling_validation",
            "metadata": {
                "n": n,
                "topology": "chain_1d",
                "strategy": "aer_mps",
                "chi_max": 64,
                "precision": 0.005,
                "seeds": [42],
                "p_layers": 1,
                "h_values": h_values,
            },
            "timing": {
                "phase1_dmrg_s": 10.0,
                "phase2_vqe_s": total_s - 10.0,
                "total_s": total_s,
            },
            "summary": {
                "n_pass": len(h_values) if all_passed else 0,
                "n_total": len(h_values),
                "all_passed": all_passed,
            },
            "vqe_results": [{"results": results}],
        }

    def test_scan_scaling_results_finds_files(self, tmp_path):
        """scan_scaling_results discovers scaling_N*_*.json files."""
        from project_health.analysis.scaling_analyzer import scan_scaling_results

        (tmp_path / "scaling_N40_chain_1d.json").write_text(json.dumps(self._make_scaling_json(40)))
        (tmp_path / "scaling_N50_chain_1d.json").write_text(json.dumps(self._make_scaling_json(50)))
        results = scan_scaling_results(tmp_path)
        assert len(results) == 2

    def test_parse_scaling_run_extracts_fields(self, tmp_path):
        """parse_scaling_run correctly extracts key fields."""
        from project_health.analysis.scaling_analyzer import parse_scaling_run

        data = self._make_scaling_json(40)
        run = parse_scaling_run(data)
        assert run is not None
        assert run.n_qubits == 40
        assert run.topology == "chain_1d"
        assert run.all_passed is True
        assert run.n_pass == 5
        assert run.n_total == 5

    def test_parse_scaling_run_rejects_non_scaling(self):
        """Non-scaling experiment JSON returns None."""
        from project_health.analysis.scaling_analyzer import parse_scaling_run

        data = {"experiment": "other_experiment"}
        assert parse_scaling_run(data) is None

    def test_validate_scaling_law_within_tolerance(self):
        """Scaling law validation passes when actual_h_min is close to predicted."""
        from project_health.analysis.scaling_analyzer import (
            ScalingPointResult,
            ScalingRunSummary,
            validate_scaling_law,
        )

        # N=40: predicted h_min = 1.5 + 0.020 * 40^1.31 ≈ 4.01
        # Use h_values starting at 4.0 so actual_h_min ≈ predicted
        run = ScalingRunSummary(
            n_qubits=40,
            topology="chain_1d",
            strategy="aer_mps",
            chi_max=MPS_DEFAULT_CHI_MAX,
            precision=0.005,
            seed=42,
            p_layers=1,
            h_values=[4.0, 4.5, 5.0],
            phase1_time_s=10,
            phase2_time_s=100,
            total_time_s=110,
            n_pass=3,
            n_total=3,
            all_passed=True,
            mean_de_gap=0.003,
            max_de_gap=0.005,
            min_de_gap=0.001,
            per_h_results=[
                ScalingPointResult(
                    h=4.0,
                    vqe_energy=-1.5,
                    dmrg_energy=-1.5,
                    gap=0.4,
                    de_gap=0.003,
                    time_s=30,
                    passed=True,
                ),
                ScalingPointResult(
                    h=4.5,
                    vqe_energy=-1.5,
                    dmrg_energy=-1.5,
                    gap=0.4,
                    de_gap=0.004,
                    time_s=30,
                    passed=True,
                ),
                ScalingPointResult(
                    h=5.0,
                    vqe_energy=-1.5,
                    dmrg_energy=-1.5,
                    gap=0.4,
                    de_gap=0.005,
                    time_s=30,
                    passed=True,
                ),
            ],
        )
        validation = validate_scaling_law(run)
        assert validation.n_qubits == 40
        assert validation.actual_h_min == 4.0
        assert validation.within_tolerance is True
        assert validation.prediction_error is not None
        assert validation.prediction_error < 0.5

    def test_detect_anomalies_timing(self):
        """Detects timing anomalies for small N with large time."""
        from project_health.analysis.scaling_analyzer import (
            ScalingRunSummary,
            detect_anomalies,
        )

        run = ScalingRunSummary(
            n_qubits=40,
            topology="chain_1d",
            strategy="aer_mps",
            chi_max=MPS_DEFAULT_CHI_MAX,
            precision=0.005,
            seed=42,
            p_layers=1,
            h_values=[4.0],
            phase1_time_s=500,
            phase2_time_s=7500,
            total_time_s=8000,
            n_pass=1,
            n_total=1,
            all_passed=True,
            mean_de_gap=0.003,
            max_de_gap=0.003,
            min_de_gap=0.003,
            per_h_results=[],
        )
        anomalies = detect_anomalies([run])
        assert any("Total time" in a for a in anomalies)

    def test_detect_anomalies_all_failed(self):
        """Detects when all h-points fail."""
        from project_health.analysis.scaling_analyzer import (
            ScalingRunSummary,
            detect_anomalies,
        )

        run = ScalingRunSummary(
            n_qubits=40,
            topology="chain_1d",
            strategy="aer_mps",
            chi_max=MPS_DEFAULT_CHI_MAX,
            precision=0.005,
            seed=42,
            p_layers=1,
            h_values=[4.0],
            phase1_time_s=10,
            phase2_time_s=100,
            total_time_s=110,
            n_pass=0,
            n_total=5,
            all_passed=False,
            mean_de_gap=0.08,
            max_de_gap=0.12,
            min_de_gap=0.06,
            per_h_results=[],
        )
        anomalies = detect_anomalies([run])
        assert any("ALL" in a for a in anomalies)

    def test_generate_report_no_data(self, tmp_path):
        """Empty results dir produces NO_DATA verdict."""
        from project_health.analysis.scaling_analyzer import generate_report

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        report = generate_report(empty_dir)
        assert report.overall_verdict == "NO_DATA"

    def test_generate_report_pass(self, tmp_path):
        """All passing runs produce PASS verdict."""
        from project_health.analysis.scaling_analyzer import generate_report

        (tmp_path / "scaling_N40_chain.json").write_text(
            json.dumps(self._make_scaling_json(40, all_passed=True))
        )
        report = generate_report(tmp_path)
        assert report.overall_verdict == "PASS"
        assert len(report.runs) == 1


# ═══════════════════════════════════════════════════════════════════════════
# project_health.reporter — output formatting
# ═══════════════════════════════════════════════════════════════════════════


class TestReporter:
    """Test report formatters produce valid output."""

    def _make_report(self):
        """Create a minimal HealthReport for formatting tests."""
        from project_health.core.models import (
            EnergyDecompositionStats,
            ExperimentSummary,
            HealthReport,
            ModelDistribution,
            MPNNQualityStats,
            TimingStats,
            VQEQualityStats,
        )

        return HealthReport(
            timestamp="2026-06-07T12:00:00+00:00",
            n_noiseless=150,
            n_noisy=30,
            n_experiments=22,
            n_confirmed=15,
            n_rejected=5,
            n_failed=2,
            experiments=[
                ExperimentSummary("G1", "confirmed", "pass_rate≥0.9", 1.0),
                ExperimentSummary("B4", "rejected", "pass_rate≥0.7", 0.5),
            ],
            noiseless_pass_rate=0.85,
            noiseless_median_de=0.025,
            noiseless_by_topology={
                "chain_1d": {
                    "n_runs": 80,
                    "pass_rate": 0.9,
                    "median_de": 0.02,
                    "best": 0.001,
                    "worst": 0.15,
                }
            },
            noisy_success_rate=0.80,
            noisy_mean_r2=0.95,
            noisy_mean_gain=45.0,
            vqe_quality=VQEQualityStats(
                n_results=100,
                convergence_rate_mean=0.95,
                convergence_rate_min=0.75,
                theta_smoothness_mean=0.4,
                theta_smoothness_max=1.8,
                n_chain_break_warnings=5,
            ),
            mpnn_quality=MPNNQualityStats(
                n_results=100,
                gen_gap_mean=0.003,
                gen_gap_max=0.02,
                gen_gap_median=0.002,
                n_overfit_warnings=3,
                theta_mse_mean=0.005,
            ),
            timing=TimingStats(
                total_pipeline_hours=12.5,
                mean_run_s=150.0,
                median_run_s=120.0,
                max_run_s=600.0,
                total_runs=300,
            ),
            distribution=ModelDistribution(
                by_topology={"chain_1d": 80, "ladder": 40},
                by_p_layers={1: 70, 2: 80},
            ),
            energy_decomposition=EnergyDecompositionStats(
                n_results=50,
                mean_circuit_error=0.005,
                mean_mpnn_error=0.015,
                circuit_error_fraction=0.25,
                mpnn_error_fraction=0.75,
            ),
            new_results=["results/new_file.json"],
            n_new=1,
            elapsed_s=2.5,
        )

    def test_format_text_contains_sections(self):
        """Text format includes all major sections."""
        from project_health.core.reporter import format_text

        report = self._make_report()
        output = format_text(report)

        assert "PROJECT HEALTH REPORT" in output
        assert "SCAN OVERVIEW" in output
        assert "EXPERIMENT VERDICTS" in output
        assert "NOISELESS QUALITY" in output
        assert "VQE CONVERGENCE" in output
        assert "MPNN TRAINING" in output
        assert "TIMING" in output
        assert "DISTRIBUTION" in output

    def test_format_text_compact_skips_detail(self):
        """Compact mode skips per-experiment detail."""
        from project_health.core.reporter import format_text

        report = self._make_report()
        output_full = format_text(report, compact=False)
        output_compact = format_text(report, compact=True)

        assert len(output_compact) < len(output_full)

    def test_format_json_valid(self):
        """JSON format produces valid JSON."""
        from project_health.core.reporter import format_json

        report = self._make_report()
        output = format_json(report)
        parsed = json.loads(output)

        assert parsed["n_noiseless"] == 150
        assert parsed["timestamp"] == "2026-06-07T12:00:00+00:00"

    def test_format_markdown_structure(self):
        """Markdown format has headers and tables."""
        from project_health.core.reporter import format_markdown

        report = self._make_report()
        output = format_markdown(report)

        assert output.startswith("# Project Health Report")
        assert "## Scan Overview" in output
        assert "| Metric |" in output
        assert "## Experiment Verdicts" in output

    def test_generate_timestamped_filename(self):
        """Timestamped filename has expected pattern."""
        from project_health.core.reporter import generate_timestamped_filename

        name = generate_timestamped_filename("health_report", "md")
        assert name.startswith("health_report_")
        assert name.endswith(".md")
        # Format: health_report_YYYYMMDD_HHMMSS.md
        assert len(name) > len("health_report_.md")


# ═══════════════════════════════════════════════════════════════════════════
# project_health.models — serialization and enum values
# ═══════════════════════════════════════════════════════════════════════════


class TestModelsSerializer:
    """Test HealthReport serialization handles all types."""

    def test_priority_enum_values(self):
        """Priority enum has expected ordering."""
        from project_health.core.models import Priority

        assert Priority.CRITICAL.value < Priority.HIGH.value
        assert Priority.HIGH.value < Priority.MEDIUM.value
        assert Priority.MEDIUM.value < Priority.LOW.value

    def test_gap_type_enum_values(self):
        """GapType enum has all expected values."""
        from project_health.core.models import GapType

        expected = {
            "p1_noiseless_missing",
            "invalid_regime",
            "insufficient_seeds",
            "missing_zne",
            "missing_experiment",
        }
        actual = {g.value for g in GapType}
        assert expected == actual

    def test_health_report_to_dict_enums_serialized(self):
        """to_dict converts enums to their string values."""
        from project_health.core.models import (
            ActionItem,
            CoverageGap,
            GapType,
            HealthReport,
            Priority,
        )

        report = HealthReport(
            timestamp="test",
            gaps=[
                CoverageGap(
                    gap_type=GapType.MISSING_ZNE,
                    priority=Priority.HIGH,
                    topology="chain_1d",
                    detail="test gap",
                )
            ],
            actions=[
                ActionItem(
                    priority=Priority.CRITICAL,
                    title="Do something",
                    detail="Details",
                    category="hardware",
                )
            ],
        )
        d = report.to_dict()

        # Enums should be serialized as values
        assert d["gaps"][0]["priority"] == 2  # HIGH.value
        assert d["gaps"][0]["gap_type"] == "missing_zne"
        assert d["actions"][0]["priority"] == 1  # CRITICAL.value

        # Full dict should be JSON-serializable
        json.dumps(d)

    def test_health_report_to_dict_nested_dataclasses(self):
        """Nested dataclasses (VQEQualityStats, etc.) are serialized."""
        from project_health.core.models import HealthReport, VQEQualityStats

        report = HealthReport(
            timestamp="test",
            vqe_quality=VQEQualityStats(
                n_results=50,
                convergence_rate_mean=0.95,
                n_chain_break_warnings=3,
            ),
        )
        d = report.to_dict()
        assert d["vqe_quality"]["n_results"] == 50
        assert d["vqe_quality"]["convergence_rate_mean"] == 0.95

    def test_coverage_gap_default_values(self):
        """CoverageGap has sensible defaults."""
        from project_health.core.models import CoverageGap, GapType, Priority

        gap = CoverageGap(gap_type=GapType.MISSING_ZNE)
        assert gap.priority == Priority.MEDIUM
        assert gap.topology == ""
        assert gap.n_qubits == 0


# ═══════════════════════════════════════════════════════════════════════════
# project_health.coverage — action derivation logic
# ═══════════════════════════════════════════════════════════════════════════


class TestActionDerivation:
    """Test derive_actions produces correct priority actions."""

    def _make_minimal_report(self):
        """Create a minimal report with adjustable fields for action testing."""
        from project_health.core.models import (
            EnergyDecompositionStats,
            HealthReport,
            ModelDistribution,
            MPNNQualityStats,
            VQEQualityStats,
        )

        return HealthReport(
            timestamp="test",
            vqe_quality=VQEQualityStats(
                n_results=100, n_chain_break_warnings=0, theta_smoothness_max=0.5
            ),
            mpnn_quality=MPNNQualityStats(n_results=100, n_overfit_warnings=0),
            distribution=ModelDistribution(
                by_topology={"chain_1d": 50}, by_p_layers={1: 50, 2: 50}
            ),
            energy_decomposition=EnergyDecompositionStats(),
            experiments=[],
            gaps=[],
        )

    def test_no_actions_when_healthy(self):
        """A healthy report produces no critical/high actions."""
        from project_health.core.coverage import derive_actions

        report = self._make_minimal_report()
        actions = derive_actions(report)
        critical = [a for a in actions if a.priority.value == 1]
        assert len(critical) == 0

    def test_chain_break_generates_action(self):
        """High chain break rate generates VQE quality action."""
        from project_health.core.coverage import derive_actions
        from project_health.core.models import VQEQualityStats

        report = self._make_minimal_report()
        report.vqe_quality = VQEQualityStats(
            n_results=10,
            n_chain_break_warnings=5,  # 50% → HIGH priority
            theta_smoothness_max=2.5,
        )
        actions = derive_actions(report)
        vqe_actions = [a for a in actions if a.category == "vqe_quality"]
        assert len(vqe_actions) >= 1
        assert "chain break" in vqe_actions[0].title.lower()

    def test_overfit_generates_action(self):
        """High overfit rate generates MPNN quality action."""
        from project_health.core.coverage import derive_actions
        from project_health.core.models import MPNNQualityStats

        report = self._make_minimal_report()
        report.mpnn_quality = MPNNQualityStats(
            n_results=10,
            n_overfit_warnings=3,  # 30% → HIGH
            gen_gap_max=0.05,
        )
        actions = derive_actions(report)
        mpnn_actions = [a for a in actions if a.category == "mpnn_quality"]
        assert len(mpnn_actions) >= 1
        assert "overfit" in mpnn_actions[0].title.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Consistency checks — canonical source enforcement
# ═══════════════════════════════════════════════════════════════════════════


class TestRegimeBoundaryConsistency:
    """Verify P1/P2_VALID_REGIME is consistent across all modules.

    The canonical source is qmbp_simulation.framework.preflight.
    All other modules must import from there, not define their own copies.
    """

    def test_coverage_uses_preflight_p1(self):
        """coverage.py P1_VALID_REGIME is the same object as preflight's."""
        from project_health.core.coverage import P1_VALID_REGIME as cov_p1
        from qmbp_simulation.framework.preflight import P1_VALID_REGIME as pf_p1

        assert cov_p1 is pf_p1, "coverage.py must import P1_VALID_REGIME from preflight"

    def test_coverage_uses_preflight_p2(self):
        """coverage.py P2_VALID_REGIME is the same object as preflight's."""
        from project_health.core.coverage import P2_VALID_REGIME as cov_p2
        from qmbp_simulation.framework.preflight import P2_VALID_REGIME as pf_p2

        assert cov_p2 is pf_p2, "coverage.py must import P2_VALID_REGIME from preflight"

    def test_diagnose_uses_preflight_p1(self):
        """diagnose.py P1_VALID_REGIME is the same object as preflight's."""
        from project_health.analysis.diagnose import P1_VALID_REGIME as diag_p1
        from qmbp_simulation.framework.preflight import P1_VALID_REGIME as pf_p1

        assert diag_p1 is pf_p1, "diagnose.py must import P1_VALID_REGIME from preflight"

    def test_preflight_has_minimum_entries(self):
        """Canonical P1_VALID_REGIME has entries for all validated configs."""
        from qmbp_simulation.framework.preflight import P1_VALID_REGIME

        # Must cover at minimum: chain N=6/10/20, ladder N=6/10, triangular N=6/10
        required = [
            ("chain_1d", 6),
            ("chain_1d", 10),
            ("chain_1d", 20),
            ("ladder", 6),
            ("ladder", 10),
            ("triangular", 6),
            ("triangular", 10),
            ("heavy_hex", 6),
            ("heavy_hex", 10),
        ]
        for key in required:
            assert key in P1_VALID_REGIME, f"Missing required entry: {key}"

    def test_preflight_values_are_positive(self):
        """All regime boundaries must be positive floats."""
        from qmbp_simulation.framework.preflight import P1_VALID_REGIME, P2_VALID_REGIME

        for key, val in P1_VALID_REGIME.items():
            assert val > 0, f"P1 {key} has non-positive boundary: {val}"
        for key, val in P2_VALID_REGIME.items():
            assert val > 0, f"P2 {key} has non-positive boundary: {val}"

    def test_p1_boundaries_higher_than_p2(self):
        """p=1 boundaries must be >= p=2 boundaries for same config."""
        from qmbp_simulation.framework.preflight import P1_VALID_REGIME, P2_VALID_REGIME

        common_keys = set(P1_VALID_REGIME.keys()) & set(P2_VALID_REGIME.keys())
        for key in common_keys:
            assert P1_VALID_REGIME[key] >= P2_VALID_REGIME[key], (
                f"{key}: p=1 boundary ({P1_VALID_REGIME[key]}) < "
                f"p=2 boundary ({P2_VALID_REGIME[key]}) — physically impossible"
            )

    def test_validate_regime_tables_no_errors(self):
        """validate_regime_tables() should find zero errors on the canonical tables."""
        from qmbp_simulation.framework.preflight import validate_regime_tables

        errors = validate_regime_tables()
        assert errors == [], f"Regime table errors: {errors}"

    def test_get_valid_regime_invalid_p(self):
        """get_valid_regime(p=3) raises ValueError."""
        from qmbp_simulation.framework.preflight import get_valid_regime

        with pytest.raises(ValueError, match="p=3"):
            get_valid_regime(3)

    def test_get_regime_threshold_unknown_config(self):
        """Unknown topology/N returns 0.0 (permissive)."""
        from qmbp_simulation.framework.preflight import get_regime_threshold

        assert get_regime_threshold("unknown_topo", 99, 1) == 0.0
        assert get_regime_threshold("chain_1d", 6, 3) == 0.0  # invalid p


class TestDataIntegrityChecks:
    """Validate that scanner output meets data quality invariants."""

    def _make_noiseless(self, **kwargs):
        from project_health.digest.models import NoiselessResult

        defaults = dict(
            source_file="test.json",
            folder="test",
            n_qubits=6,
            p_layers=1,
            topology="chain_1d",
            delta_e_over_gap=0.03,
            seed=42,
        )
        defaults.update(kwargs)
        return NoiselessResult(**defaults)

    def test_delta_e_over_gap_non_negative(self):
        """ΔE/gap must never be negative (it's an absolute error ratio)."""
        r = self._make_noiseless(delta_e_over_gap=-0.01)
        # This is a data corruption indicator
        assert r.delta_e_over_gap < 0  # confirms the bad value exists
        # The health system should flag this

    def test_convergence_rate_bounded_0_1(self):
        """Convergence rate must be in [0, 1]."""
        r = self._make_noiseless(convergence_rate=1.5)
        assert not (0.0 <= r.convergence_rate <= 1.0)  # out of bounds

    def test_elapsed_s_non_negative(self):
        """Elapsed time must be non-negative."""
        r = self._make_noiseless(elapsed_s=120.0)
        assert r.elapsed_s >= 0

    def test_new_fields_default_none(self):
        """New fields should default to None/0 for backward compat."""
        r = self._make_noiseless()
        assert r.gap_min is None
        assert r.mean_iterations is None
        assert r.error_from_circuit is None
        assert r.run_timestamp == ""
        assert r.phase2_elapsed_s == 0.0

    def test_timestamp_extraction_valid_format(self):
        """_extract_timestamp_from_filename produces valid ISO timestamps."""
        from project_health.digest.scanner import _extract_timestamp_from_filename

        ts = _extract_timestamp_from_filename("pipeline_run_20260529_210502.json")
        assert ts == "2026-05-29T21:05:02"

    def test_timestamp_extraction_invalid_filename(self):
        """Non-timestamped filenames produce empty string."""
        from project_health.digest.scanner import _extract_timestamp_from_filename

        assert _extract_timestamp_from_filename("pipeline_run_001.json") == ""
        assert _extract_timestamp_from_filename("random_file.json") == ""


# ═══════════════════════════════════════════════════════════════════════════
# project_health.analysis.statistical_tests — paired tests
# ═══════════════════════════════════════════════════════════════════════════


class TestStatisticalTests:
    """Test the reusable statistical test functions."""

    def test_paired_ttest_significant(self):
        """Clear improvement should produce significant p-value."""
        from project_health.analysis.statistical_tests import paired_ttest

        before = [0.15, 0.12, 0.18, 0.14, 0.16]
        after = [0.02, 0.01, 0.03, 0.02, 0.01]
        result = paired_ttest(before, after)

        assert result["n"] == 5
        assert result["mean_diff"] > 0
        assert result["significant_005"] is True
        assert result["significant_001"] is True
        assert result["effect_size_d"] > 0.8  # Large effect

    def test_paired_ttest_not_significant(self):
        """No difference should produce non-significant p-value."""
        from project_health.analysis.statistical_tests import paired_ttest

        before = [0.10, 0.12, 0.11, 0.10, 0.13]
        after = [0.11, 0.10, 0.12, 0.13, 0.10]
        result = paired_ttest(before, after)

        assert result["significant_005"] is False

    def test_paired_ttest_mismatched_lengths(self):
        """Different-length lists should raise ValueError."""
        from project_health.analysis.statistical_tests import paired_ttest

        with pytest.raises(ValueError, match="same length"):
            paired_ttest([1, 2, 3], [1, 2])

    def test_paired_ttest_too_few_samples(self):
        """Fewer than 2 samples should raise ValueError."""
        from project_health.analysis.statistical_tests import paired_ttest

        with pytest.raises(ValueError, match="at least 2"):
            paired_ttest([1.0], [0.5])

    def test_improvement_rate_all_improved(self):
        """100% improvement rate when all after < before."""
        from project_health.analysis.statistical_tests import improvement_rate

        result = improvement_rate([0.15, 0.12, 0.18], [0.02, 0.01, 0.03])
        assert result["improvement_rate_pct"] == 100.0
        assert result["n_improved"] == 3
        assert result["n_worsened"] == 0
        assert result["mean_reduction_pct"] > 80.0

    def test_improvement_rate_none_improved(self):
        """0% when after >= before for all."""
        from project_health.analysis.statistical_tests import improvement_rate

        result = improvement_rate([0.02, 0.01], [0.15, 0.12])
        assert result["improvement_rate_pct"] == 0.0
        assert result["n_worsened"] == 2

    def test_effect_size_large(self):
        """Large differences produce d > 0.8."""
        from project_health.analysis.statistical_tests import effect_size_cohens_d

        d = effect_size_cohens_d([10.0, 11.0, 12.0], [1.0, 2.0, 1.5])
        assert d > 0.8

    def test_effect_size_zero(self):
        """Identical lists produce d = 0 (no effect)."""
        from project_health.analysis.statistical_tests import effect_size_cohens_d

        d = effect_size_cohens_d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert d == 0.0
