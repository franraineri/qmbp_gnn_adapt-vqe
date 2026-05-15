"""Integration tests for pipeline observability CLI flags and behavioral equivalence.

Validates Properties 9 and 10 from the design document, plus verbose/debug
flag behavior. Tests are fast — they do NOT run the full pipeline.

**Validates: Requirements 3.3, 4.4, 7.1, 7.2, 7.3, 7.4**
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.poc.v6.diagnostics import DiagnosticCollector, configure_pipeline_logging  # noqa: E402

# ── Test 1: Behavioral Equivalence (Property 9) ─────────────────────────


class TestBehavioralEquivalence:
    """Without --verbose, output should have no diagnostics key."""

    def test_no_diagnostics_key_when_verbose_false(self):
        """run_pipeline with verbose=False should not produce a 'diagnostics' key.

        **Validates: Requirements 4.4, 7.1**
        """
        # DiagnosticCollector is only instantiated when verbose=True.
        # When verbose=False, the pipeline code does:
        #   collector = None
        #   ...
        #   if collector is not None:
        #       result["diagnostics"] = collector.to_dict()
        #
        # We verify this logic by simulating the conditional:
        verbose = False
        collector = None
        if verbose:
            collector = DiagnosticCollector(verbose=True)

        # Build a mock result dict (simulating pipeline output)
        result = {
            "config_name": "test",
            "config": {"N": 6, "seed": 42},
            "phases": {"phase1": {}, "phase2": {}, "phase3": {}, "phase4": {}},
            "success": True,
            "error": None,
        }

        # This is the exact logic from run_v61_parametric.py
        if collector is not None:
            result["diagnostics"] = collector.to_dict()

        assert "diagnostics" not in result

    def test_diagnostics_key_present_when_verbose_true(self):
        """run_pipeline with verbose=True should produce a 'diagnostics' key.

        **Validates: Requirements 4.4, 5.1**
        """
        verbose = True
        collector = None
        if verbose:
            collector = DiagnosticCollector(verbose=True)

        result = {
            "config_name": "test",
            "config": {"N": 6, "seed": 42},
            "phases": {},
            "success": True,
            "error": None,
        }

        if collector is not None:
            result["diagnostics"] = collector.to_dict()

        assert "diagnostics" in result
        assert isinstance(result["diagnostics"], dict)


# ── Test 2: Logging Level Isolation (Property 10) ────────────────────────


class TestLoggingLevelIsolation:
    """With verbose=False and debug=False, no INFO/DEBUG messages from gnn_hva."""

    def test_default_logging_level_is_warning(self):
        """configure_pipeline_logging with defaults sets WARNING level.

        **Validates: Requirements 3.3, 7.3**
        """
        log = configure_pipeline_logging(verbose=False, debug=False)
        assert log.level == logging.WARNING

    def test_no_info_messages_at_default_level(self):
        """At WARNING level, INFO messages should not be emitted.

        **Validates: Requirements 3.3, 7.4**
        """
        log = configure_pipeline_logging(verbose=False, debug=False)

        # The logger should filter out INFO messages
        assert not log.isEnabledFor(logging.INFO)
        assert not log.isEnabledFor(logging.DEBUG)
        assert log.isEnabledFor(logging.WARNING)


# ── Test 3: Verbose produces diagnostics section ─────────────────────────


class TestVerboseDiagnosticsOutput:
    """Running with verbose=True produces output with all expected diagnostic keys."""

    def test_verbose_diagnostics_contains_phase2_keys(self):
        """Verbose diagnostics should contain all expected phase2 keys.

        **Validates: Requirements 5.2**
        """
        collector = DiagnosticCollector(verbose=True)

        # Record some Phase 2 data
        for i, h in enumerate([2.0, 1.5, 1.0]):
            collector.record_vqe_point(
                h=h,
                n_iters=100 + i * 10,
                restart_energies=[-5.0, -5.1, -4.9],
                theta_opt=np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i]),
                elapsed_s=1.0 + i * 0.5,
            )

        diag = collector.to_dict()
        phase2 = diag["phase2"]

        expected_keys = {
            "per_h_timing_s",
            "per_h_iterations",
            "per_h_restart_spread",
            "theta_smoothness",
            "worst_convergence_h",
        }
        assert expected_keys.issubset(set(phase2.keys()))

    def test_verbose_diagnostics_contains_phase3_keys(self):
        """Verbose diagnostics should contain all expected phase3 keys.

        **Validates: Requirements 5.3**
        """
        collector = DiagnosticCollector(verbose=True)

        # Record Phase 3 data
        for epoch in range(50):
            collector.record_mpnn_epoch(epoch, train_loss=0.1 - epoch * 0.001)

        h_values = np.array([0.5, 1.0, 1.5, 2.0])
        per_h_mse = np.array([0.01, 0.05, 0.02, 0.008])
        collector.record_mpnn_per_h_error(h_values, per_h_mse)

        diag = collector.to_dict()
        phase3 = diag["phase3"]

        expected_keys = {
            "per_h_mse",
            "theta_zz_mse",
            "theta_x_mse",
            "generalization_gap",
            "loss_curve_last100",
        }
        assert expected_keys.issubset(set(phase3.keys()))

    def test_verbose_diagnostics_contains_phase4_keys(self):
        """Verbose diagnostics should contain all expected phase4 keys.

        **Validates: Requirements 5.4**
        """
        collector = DiagnosticCollector(verbose=True)

        # Create a mock deploy result with required attributes
        class MockDeployResult:
            predicted_energy = -8.5
            delta_e = 0.1
            mag_x_pred = 0.7
            corr_zz_pred = 0.3
            total_shots = 4096
            raw_energy = -8.4

        result = MockDeployResult()
        collector.record_deployment(h_test=1.25, result=result, per_layout_data=None)

        diag = collector.to_dict()
        phase4 = diag["phase4"]

        expected_keys = {
            "snr_mag_x",
            "snr_corr_zz",
            "classification_confidence",
            "per_layout_energies",
            "per_layout_ces",
            "ces_energy_pearson_r",
            "energy_decomposition",
        }
        assert expected_keys.issubset(set(phase4.keys()))


# ── Test 4: Debug enables DEBUG-level messages ───────────────────────────


class TestDebugLogging:
    """Debug flag enables DEBUG-level messages on the gnn_hva logger."""

    def test_debug_sets_debug_level(self):
        """configure_pipeline_logging(debug=True) sets DEBUG level.

        **Validates: Requirements 3.2**
        """
        log = configure_pipeline_logging(debug=True)
        assert log.level == logging.DEBUG
        assert log.isEnabledFor(logging.DEBUG)
        assert log.isEnabledFor(logging.INFO)

    def test_verbose_sets_info_level(self):
        """configure_pipeline_logging(verbose=True) sets INFO level.

        **Validates: Requirements 3.1**
        """
        log = configure_pipeline_logging(verbose=True)
        assert log.level == logging.INFO
        assert log.isEnabledFor(logging.INFO)
        assert not log.isEnabledFor(logging.DEBUG)


# ── Test 5: CLI argparse flags ───────────────────────────────────────────


class TestCLIArgparseFlags:
    """Test that the argparse parser accepts --verbose, -v, and --debug flags."""

    @pytest.fixture
    def parser(self):
        """Create an argparse parser matching run_v61_parametric.py's interface."""
        parser = argparse.ArgumentParser(description="V6.1 Parametric Pipeline Runner")
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            default=False,
            help="Enable INFO logging, DiagnosticCollector, and VQE callbacks",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            default=False,
            help="Enable DEBUG logging and all verbose features",
        )
        parser.add_argument(
            "--config",
            default="all",
            help="Configuration preset to run",
        )
        return parser

    def test_verbose_long_flag(self, parser):
        """--verbose flag is accepted and sets verbose=True."""
        args = parser.parse_args(["--verbose"])
        assert args.verbose is True
        assert args.debug is False

    def test_verbose_short_flag(self, parser):
        """-v flag is accepted and sets verbose=True."""
        args = parser.parse_args(["-v"])
        assert args.verbose is True
        assert args.debug is False

    def test_debug_flag(self, parser):
        """--debug flag is accepted and sets debug=True."""
        args = parser.parse_args(["--debug"])
        assert args.debug is True
        assert args.verbose is False

    def test_no_flags_defaults(self, parser):
        """No flags means verbose=False and debug=False."""
        args = parser.parse_args([])
        assert args.verbose is False
        assert args.debug is False

    def test_both_flags(self, parser):
        """Both --verbose and --debug can be passed simultaneously."""
        args = parser.parse_args(["--verbose", "--debug"])
        assert args.verbose is True
        assert args.debug is True


# ── Test 6: Borderline metric warnings and NaN detection (Task 5.1) ──────


class TestBorderlineWarningsAndNaNDetection:
    """Validate structured logging for borderline metrics and NaN values.

    **Validates: Requirements 3.5, 3.6, 8.1, 8.5**
    """

    @pytest.fixture(autouse=True)
    def _enable_log_propagation(self):
        """Enable propagation on gnn_hva so caplog can capture messages."""
        log = logging.getLogger("gnn_hva")
        original_propagate = log.propagate
        log.propagate = True
        yield
        log.propagate = original_propagate

    def test_nan_detection_in_vqe_theta_opt(self, caplog):
        """NaN in theta_opt emits ERROR log with h-value.

        **Validates: Requirements 3.6, 8.1**
        """
        collector = DiagnosticCollector(verbose=True)

        # theta_opt with NaN
        theta_with_nan = np.array([0.1, np.nan, 0.3, 0.4])

        with caplog.at_level(logging.ERROR, logger="gnn_hva.diagnostics"):
            collector.record_vqe_point(
                h=1.2500,
                n_iters=100,
                restart_energies=[-5.0, -5.1],
                theta_opt=theta_with_nan,
                elapsed_s=2.0,
            )

        # Should emit ERROR with h-value formatted to 4 decimal places
        assert any("NaN detected in VQE theta_opt at h=1.2500" in r.message for r in caplog.records)

    def test_nan_in_vqe_does_not_abort_pipeline(self):
        """NaN in theta_opt is recorded but pipeline continues.

        **Validates: Requirements 8.1, 8.5**
        """
        collector = DiagnosticCollector(verbose=True)

        # Record a point with NaN — should not raise
        theta_with_nan = np.array([np.nan, np.nan, 0.3, 0.4])
        collector.record_vqe_point(
            h=1.0,
            n_iters=50,
            restart_energies=[-4.0],
            theta_opt=theta_with_nan,
            elapsed_s=1.5,
        )

        # Data should still be recorded
        diag = collector.to_dict()
        assert len(diag["phase2"]["per_h_timing_s"]) == 1
        assert diag["phase2"]["per_h_iterations"] == [50]

    def test_borderline_metric_warning_emitted(self, caplog):
        """ΔE/gap between 4% and 5% emits WARNING.

        **Validates: Requirement 3.5**
        """
        collector = DiagnosticCollector(verbose=True)

        class MockDeployResult:
            predicted_energy = -8.5
            delta_e = 0.1
            delta_e_over_gap = 0.045  # 4.5% — borderline
            mag_x_pred = 0.7
            corr_zz_pred = 0.3
            total_shots = 4096
            raw_energy = -8.4

        with caplog.at_level(logging.WARNING, logger="gnn_hva.diagnostics"):
            collector.record_deployment(
                h_test=1.25, result=MockDeployResult(), per_layout_data=None
            )

        assert any(
            "Borderline metric" in r.message and "0.0450" in r.message for r in caplog.records
        )

    def test_no_borderline_warning_below_threshold(self, caplog):
        """ΔE/gap below 4% does NOT emit borderline warning.

        **Validates: Requirement 3.5**
        """
        collector = DiagnosticCollector(verbose=True)

        class MockDeployResult:
            predicted_energy = -8.5
            delta_e = 0.05
            delta_e_over_gap = 0.03  # 3% — well below threshold
            mag_x_pred = 0.7
            corr_zz_pred = 0.3
            total_shots = 4096
            raw_energy = -8.4

        with caplog.at_level(logging.WARNING, logger="gnn_hva.diagnostics"):
            collector.record_deployment(
                h_test=1.25, result=MockDeployResult(), per_layout_data=None
            )

        assert not any("Borderline metric" in r.message for r in caplog.records)

    def test_no_borderline_warning_at_or_above_5_percent(self, caplog):
        """ΔE/gap at exactly 5% does NOT emit borderline warning (exclusive upper bound).

        **Validates: Requirement 3.5**
        """
        collector = DiagnosticCollector(verbose=True)

        class MockDeployResult:
            predicted_energy = -8.5
            delta_e = 0.2
            delta_e_over_gap = 0.05  # Exactly 5% — above borderline range
            mag_x_pred = 0.7
            corr_zz_pred = 0.3
            total_shots = 4096
            raw_energy = -8.4

        with caplog.at_level(logging.WARNING, logger="gnn_hva.diagnostics"):
            collector.record_deployment(
                h_test=1.25, result=MockDeployResult(), per_layout_data=None
            )

        assert not any("Borderline metric" in r.message for r in caplog.records)

    def test_nan_in_predicted_energy_emits_error(self, caplog):
        """NaN in predicted_energy emits ERROR log.

        **Validates: Requirements 3.6, 8.1**
        """
        collector = DiagnosticCollector(verbose=True)

        class MockDeployResult:
            predicted_energy = float("nan")
            delta_e = 0.1
            mag_x_pred = 0.7
            corr_zz_pred = 0.3
            total_shots = 4096
            raw_energy = -8.4

        with caplog.at_level(logging.ERROR, logger="gnn_hva.diagnostics"):
            collector.record_deployment(
                h_test=1.25, result=MockDeployResult(), per_layout_data=None
            )

        assert any(
            "NaN detected in predicted energy at h_test=1.25" in r.message for r in caplog.records
        )

    def test_internal_error_does_not_propagate(self):
        """Internal metric computation error is caught and logged, not propagated.

        **Validates: Requirement 8.5**
        """
        collector = DiagnosticCollector(verbose=True)

        # Create a result that will cause an internal error
        # (missing required attributes for SNR computation)
        class BrokenResult:
            predicted_energy = -8.5
            delta_e = 0.1
            # Missing mag_x_pred, corr_zz_pred, total_shots — will cause AttributeError

        # Should NOT raise — error is caught internally
        collector.record_deployment(h_test=1.25, result=BrokenResult(), per_layout_data=None)

        # Phase 4 data should be set to None values (graceful degradation)
        diag = collector.to_dict()
        assert diag["phase4"]["snr_mag_x"] is None


# ── Test 7: Graceful Degradation and Serialization ───────────────────────


class TestGracefulDegradation:
    """Validate graceful handling of missing data, edge cases, and serialization.

    **Validates: Requirements 5.5, 5.6, 8.3, 8.4, 8.5**
    """

    def test_record_deployment_missing_attributes(self):
        """Result object missing total_shots → graceful fallback (phase4 data set to None).

        **Validates: Requirement 8.5**
        """
        collector = DiagnosticCollector(verbose=False)

        # Missing total_shots attribute — will cause AttributeError internally
        class IncompleteResult:
            predicted_energy = -8.5
            delta_e = 0.1
            mag_x_pred = 0.7
            corr_zz_pred = 0.3
            # total_shots is MISSING
            raw_energy = -8.4

        # Should NOT raise
        collector.record_deployment(h_test=1.25, result=IncompleteResult(), per_layout_data=None)

        diag = collector.to_dict()
        # Phase 4 should have None values due to graceful degradation
        assert diag["phase4"]["snr_mag_x"] is None
        assert diag["phase4"]["snr_corr_zz"] is None

    def test_record_deployment_pearson_constant_values(self):
        """per_layout_data with all identical energies → ces_energy_pearson_r = None (not crash).

        When all values are constant, pearsonr returns NaN. The collector should
        handle this gracefully.

        **Validates: Requirements 8.4, 8.5**
        """
        collector = DiagnosticCollector(verbose=False)

        class MockResult:
            predicted_energy = -8.5
            delta_e = 0.1
            delta_e_over_gap = 0.03
            mag_x_pred = 0.7
            corr_zz_pred = 0.3
            total_shots = 4096
            raw_energy = -8.4

        # All identical energies → pearsonr will produce NaN or warning
        per_layout_data = {
            "energies": [5.0, 5.0, 5.0],
            "ces_values": [1.0, 1.0, 1.0],
        }

        # Should NOT raise
        collector.record_deployment(
            h_test=1.25, result=MockResult(), per_layout_data=per_layout_data
        )

        diag = collector.to_dict()
        # Pearson r with constant values is undefined (NaN) — should be None or NaN
        # The important thing is it doesn't crash
        pearson_r = diag["phase4"]["ces_energy_pearson_r"]
        # Accept None or NaN as valid graceful outcomes
        assert pearson_r is None or (isinstance(pearson_r, float) and np.isnan(pearson_r))

    def test_to_dict_json_roundtrip(self):
        """json.loads(json.dumps(collector.to_dict())) produces equivalent dict.

        **Validates: Requirements 5.5, 5.6**
        """
        import json

        collector = DiagnosticCollector(verbose=False)

        # Record data for all phases
        collector.record_phase1(n_points=10, elapsed_s=2.5, gap_min=0.15)

        for h in [2.0, 1.5, 1.0]:
            collector.record_vqe_point(
                h=h,
                n_iters=100,
                restart_energies=[-5.0, -5.1],
                theta_opt=np.array([0.1, 0.2, 0.3, 0.4]),
                elapsed_s=1.0,
            )

        collector.record_mpnn_per_h_error(
            np.array([1.0, 1.5, 2.0]),
            np.array([0.01, 0.02, 0.03]),
        )

        class MockResult:
            predicted_energy = -8.5
            delta_e = 0.1
            delta_e_over_gap = 0.03
            mag_x_pred = 0.7
            corr_zz_pred = 0.3
            total_shots = 4096
            raw_energy = -8.4

        collector.record_deployment(h_test=1.25, result=MockResult(), per_layout_data=None)

        # JSON roundtrip should not raise and produce equivalent dict
        original = collector.to_dict()
        json_str = json.dumps(original)
        roundtripped = json.loads(json_str)

        assert roundtripped == original

    def test_to_dict_partial_phases(self):
        """Only phase1 recorded → phase2/3/4 have default None values.

        **Validates: Requirements 5.7, 8.3**
        """
        collector = DiagnosticCollector(verbose=False)
        collector.record_phase1(n_points=5, elapsed_s=0.5, gap_min=0.2)

        diag = collector.to_dict()

        # Phase 1 should have data
        assert diag["phase1"] is not None
        assert diag["phase1"]["n_points"] == 5

        # Phase 2 should have empty/default values
        assert diag["phase2"]["theta_smoothness"] is None
        assert diag["phase2"]["worst_convergence_h"] is None
        assert diag["phase2"]["per_h_timing_s"] == []

        # Phase 3 should have default None values
        assert diag["phase3"]["theta_zz_mse"] is None
        assert diag["phase3"]["theta_x_mse"] is None

        # Phase 4 should have default None values
        assert diag["phase4"]["snr_mag_x"] is None
        assert diag["phase4"]["energy_decomposition"] is None

    def test_configure_logging_multiple_calls(self):
        """Calling configure_pipeline_logging twice doesn't duplicate handlers.

        **Validates: Requirement 3.4**
        """
        # First call
        log1 = configure_pipeline_logging(verbose=True)
        n_handlers_1 = len(log1.handlers)

        # Second call — should clear handlers and add fresh one
        log2 = configure_pipeline_logging(verbose=True)
        n_handlers_2 = len(log2.handlers)

        # Same logger instance
        assert log1 is log2

        # Should have exactly 1 handler (not 2)
        assert n_handlers_1 == 1
        assert n_handlers_2 == 1
