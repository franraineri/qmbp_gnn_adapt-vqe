"""Tests for MitigationBenchmarkAnalyzer transfer ratio and Spearman correlation (task 8.4)."""

import json
import logging
import tempfile
from pathlib import Path

from project_health.analysis.mitigation_benchmark_analyzer import (
    MitigationBenchmarkAnalyzer,
)


def _entry(config_id: str, mode: str, delta_e_gap: float, h_value: float = 3.5):
    """Create a valid result envelope entry."""
    return {
        "benchmark_metadata": {
            "config_id": config_id,
            "h_value": h_value,
            "execution_mode": mode,
            "timestamp": "2026-06-18T10:30:00",
            "seed": 42,
        },
        "circuit_stats": {"depth_logical": 10, "depth_transpiled": 25, "n_2q_gates": 18},
        "results": {"e_raw": -12.5, "e_exact": -12.8, "delta_e_gap": delta_e_gap},
    }


def _setup_analyzer(entries: list[dict]) -> MitigationBenchmarkAnalyzer:
    """Create analyzer with entries written to a temp dir and scanned."""
    tmpdir = tempfile.mkdtemp()
    for i, entry in enumerate(entries):
        mode = entry["benchmark_metadata"]["execution_mode"]
        config_id = entry["benchmark_metadata"]["config_id"]
        d = Path(tmpdir) / mode / config_id
        d.mkdir(parents=True, exist_ok=True)
        (d / f"h350_run_{i:04d}.json").write_text(json.dumps(entry))
    analyzer = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
    analyzer.scan()
    return analyzer


# ═══════════════════════════════════════════════════════════════════════════════
# compute_transfer_ratios tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeTransferRatios:
    """Tests for compute_transfer_ratios()."""

    def test_basic_dual_mode_ratio(self):
        """Transfer ratio = mean_hw / mean_sim for dual-mode config."""
        entries = [
            _entry("C0_raw", "fake_backend", 0.02),
            _entry("C0_raw", "hardware", 0.06),
        ]
        analyzer = _setup_analyzer(entries)
        ratios = analyzer.compute_transfer_ratios()
        assert "C0_raw" in ratios
        assert abs(ratios["C0_raw"] - 3.0) < 1e-10

    def test_multiple_entries_averaged(self):
        """Mean is computed across multiple h-values per mode."""
        entries = [
            _entry("C5_full", "fake_backend", 0.02, h_value=3.25),
            _entry("C5_full", "fake_backend", 0.04, h_value=3.5),
            _entry("C5_full", "hardware", 0.06, h_value=3.25),
            _entry("C5_full", "hardware", 0.10, h_value=3.5),
        ]
        analyzer = _setup_analyzer(entries)
        ratios = analyzer.compute_transfer_ratios()
        # mean_sim = (0.02+0.04)/2 = 0.03, mean_hw = (0.06+0.10)/2 = 0.08
        expected = 0.08 / 0.03
        assert abs(ratios["C5_full"] - expected) < 1e-10

    def test_empty_when_no_dual_mode(self, caplog):
        """Returns empty dict and warns when no dual-mode configs exist."""
        entries = [
            _entry("C0_raw", "fake_backend", 0.02),
            _entry("C1_dd", "hardware", 0.05),
        ]
        analyzer = _setup_analyzer(entries)
        with caplog.at_level(logging.WARNING):
            ratios = analyzer.compute_transfer_ratios()
        assert ratios == {}
        assert "No dual-mode configs found" in caplog.text

    def test_multiple_configs(self):
        """Computes ratios for multiple dual-mode configs independently."""
        entries = [
            _entry("C0_raw", "fake_backend", 0.02),
            _entry("C0_raw", "hardware", 0.04),
            _entry("C3_full", "fake_backend", 0.01),
            _entry("C3_full", "hardware", 0.025),
        ]
        analyzer = _setup_analyzer(entries)
        ratios = analyzer.compute_transfer_ratios()
        assert len(ratios) == 2
        assert abs(ratios["C0_raw"] - 2.0) < 1e-10
        assert abs(ratios["C3_full"] - 2.5) < 1e-10

    def test_ignores_single_mode_configs(self):
        """Configs with only one mode are excluded from ratios."""
        entries = [
            _entry("C0_raw", "fake_backend", 0.02),
            _entry("C0_raw", "hardware", 0.04),
            _entry("C1_dd", "fake_backend", 0.03),  # only sim
        ]
        analyzer = _setup_analyzer(entries)
        ratios = analyzer.compute_transfer_ratios()
        assert "C0_raw" in ratios
        assert "C1_dd" not in ratios

    def test_skips_zero_sim_delta(self, caplog):
        """Warns and skips config when mean_sim is zero (avoids division)."""
        entries = [
            _entry("C0_raw", "fake_backend", 0.0),
            _entry("C0_raw", "hardware", 0.05),
        ]
        analyzer = _setup_analyzer(entries)
        with caplog.at_level(logging.WARNING):
            ratios = analyzer.compute_transfer_ratios()
        assert "C0_raw" not in ratios
        assert "non-positive" in caplog.text


# ═══════════════════════════════════════════════════════════════════════════════
# compute_spearman_correlation tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeSpearmanCorrelation:
    """Tests for compute_spearman_correlation()."""

    def test_returns_none_below_5_configs(self, caplog):
        """Returns None with warning when < 5 dual-mode configs."""
        entries = [_entry(f"C{i}", "fake_backend", 0.01 * (i + 1)) for i in range(4)] + [
            _entry(f"C{i}", "hardware", 0.02 * (i + 1)) for i in range(4)
        ]
        analyzer = _setup_analyzer(entries)
        with caplog.at_level(logging.WARNING):
            result = analyzer.compute_spearman_correlation()
        assert result is None
        assert "minimum 5 required" in caplog.text

    def test_perfect_correlation(self):
        """Perfect rank agreement gives ρ ≈ 1.0."""
        # Same ranking in both modes (monotonically increasing delta)
        entries = [_entry(f"C{i}", "fake_backend", 0.01 * (i + 1)) for i in range(6)] + [
            _entry(f"C{i}", "hardware", 0.02 * (i + 1)) for i in range(6)
        ]
        analyzer = _setup_analyzer(entries)
        rho = analyzer.compute_spearman_correlation()
        assert rho is not None
        assert abs(rho - 1.0) < 1e-10

    def test_inverted_correlation(self):
        """Inverted ranking gives ρ ≈ -1.0."""
        n = 6
        entries = [_entry(f"C{i}", "fake_backend", 0.01 * (i + 1)) for i in range(n)] + [
            _entry(f"C{i}", "hardware", 0.01 * (n - i)) for i in range(n)
        ]
        analyzer = _setup_analyzer(entries)
        rho = analyzer.compute_spearman_correlation()
        assert rho is not None
        assert abs(rho - (-1.0)) < 1e-10

    def test_returns_none_with_zero_dual_configs(self, caplog):
        """Returns None when no configs have both modes."""
        entries = [
            _entry("C0_raw", "fake_backend", 0.02),
            _entry("C1_dd", "hardware", 0.05),
        ]
        analyzer = _setup_analyzer(entries)
        with caplog.at_level(logging.WARNING):
            result = analyzer.compute_spearman_correlation()
        assert result is None

    def test_exactly_5_configs_computes(self):
        """Exactly 5 dual-mode configs meets the threshold."""
        entries = [_entry(f"C{i}", "fake_backend", 0.01 * (i + 1)) for i in range(5)] + [
            _entry(f"C{i}", "hardware", 0.015 * (i + 1)) for i in range(5)
        ]
        analyzer = _setup_analyzer(entries)
        rho = analyzer.compute_spearman_correlation()
        assert rho is not None
        assert -1.0 <= rho <= 1.0
