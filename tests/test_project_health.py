"""Basic test coverage for the project_health package.

Verifies core functionality:
- Health check engine produces a valid HealthReport
- Report serialization works
- Figure registry is populated
- Figures can be generated from analysis data
- Scanner discovers experiments
- Verdict computation returns valid results

Usage:
    pytest tests/test_project_health.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


class TestHealthCheckEngine:
    """Tests for the core health check orchestration engine."""

    def test_runs_without_error(self):
        """Engine produces a HealthReport without crashing."""
        from project_health.engine import run_health_check

        report = run_health_check(save_state=False)
        assert report.n_noiseless > 0
        assert report.timestamp != ""

    def test_report_has_experiments(self):
        """Health report includes experiment summaries."""
        from project_health.engine import run_health_check

        report = run_health_check(save_state=False)
        assert report.n_experiments > 0
        assert len(report.experiments) > 0

    def test_report_serializes_to_json(self):
        """HealthReport.to_dict() produces valid JSON-serializable output."""
        from project_health.engine import run_health_check

        report = run_health_check(save_state=False)
        serialized = json.dumps(report.to_dict())
        assert len(serialized) > 100
        parsed = json.loads(serialized)
        assert "n_noiseless" in parsed
        assert "timestamp" in parsed

    def test_report_has_quality_diagnostics(self):
        """Health report includes VQE and MPNN quality stats."""
        from project_health.engine import run_health_check

        report = run_health_check(save_state=False)
        assert report.vqe_quality is not None
        assert report.mpnn_quality is not None

    def test_report_has_actions(self):
        """Health report derives actionable items."""
        from project_health.engine import run_health_check

        report = run_health_check(save_state=False)
        assert report.actions is not None
        assert isinstance(report.actions, list)


class TestFigureRegistry:
    """Tests for the figure registry and generation system."""

    def test_registry_has_all_figures(self):
        """Figure registry contains expected minimum count."""
        from project_health.figures import registry

        names = registry.names()
        assert len(names) >= 15

    def test_registry_has_key_figures(self):
        """Registry includes critical figure types."""
        from project_health.figures import registry

        names = registry.names()
        assert "gen_gap_vs_de_gap" in names
        assert "experiment_verdicts" in names
        assert "smoothness_histogram" in names

    def test_figures_generate_from_analysis(self):
        """At least one diagnostics figure generates without error when data is available."""
        from project_health.figures import ROOT, FigureConfig, generate_figures

        diag_path = ROOT / "analysis" / "raw_data" / "all_diagnostics.json"
        if not diag_path.exists():
            pytest.skip("analysis/raw_data/all_diagnostics.json not available")

        with tempfile.TemporaryDirectory() as td:
            cfg = FigureConfig(output_dir=Path(td), format="png")
            results = generate_figures(source="analysis", only=["gen_gap_vs_de_gap"], config=cfg)
            assert results.get("gen_gap_vs_de_gap") is True

    def test_figure_config_defaults(self):
        """FigureConfig has sensible defaults."""
        from project_health.figures import FigureConfig

        cfg = FigureConfig()
        assert cfg.format == "png"
        assert cfg.dpi == 150
        assert cfg.theme == "default"
        assert cfg.show_title is True


class TestResultScanner:
    """Tests for the digest scanner functionality."""

    def test_scanner_finds_experiments(self):
        """Scanner discovers at least 20 experiments."""
        from project_health.digest.scanner import ResultScanner

        scanner = ResultScanner()
        _, _, experiments = scanner.scan_all(exclude_tests=True)
        assert len(experiments) >= 20

    def test_scanner_finds_noiseless(self):
        """Scanner discovers noiseless pipeline results."""
        from project_health.digest.scanner import ResultScanner

        scanner = ResultScanner()
        noiseless, _, _ = scanner.scan_all(exclude_tests=True)
        assert len(noiseless) >= 100

    def test_scanner_finds_noisy(self):
        """Scanner discovers noisy/ZNE results."""
        from project_health.digest.scanner import ResultScanner

        scanner = ResultScanner()
        _, noisy, _ = scanner.scan_all(exclude_tests=True)
        assert len(noisy) >= 20


class TestVerdictComputation:
    """Tests for the experiment verdict logic."""

    def test_confirmed_verdict(self):
        """High pass rate → confirmed verdict."""
        from qmbp_simulation.framework.criteria import compute_verdict

        verdict, _desc = compute_verdict("G5", {"pass_rate": 0.92})
        assert verdict == "confirmed"

    def test_failed_verdict(self):
        """Low pass rate → failed verdict."""
        from qmbp_simulation.framework.criteria import compute_verdict

        verdict, _desc = compute_verdict("G5", {"pass_rate": 0.50})
        assert verdict == "failed"

    def test_rejected_verdict_is_finding(self):
        """Some experiments have 'rejected' as a valid finding."""
        from project_health.digest.models import REJECTION_IS_FINDING

        # At least some experiments treat rejection as valid
        assert len(REJECTION_IS_FINDING) > 0


class TestModelsAPI:
    """Tests for project_health public model API."""

    def test_health_report_importable(self):
        """HealthReport is importable from public API."""
        from project_health import HealthReport  # noqa: F401

    def test_action_item_importable(self):
        """ActionItem is importable from public API."""
        from project_health import ActionItem  # noqa: F401

    def test_coverage_gap_importable(self):
        """CoverageGap is importable from public API."""
        from project_health import CoverageGap  # noqa: F401
