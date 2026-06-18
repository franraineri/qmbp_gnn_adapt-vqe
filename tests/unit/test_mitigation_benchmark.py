"""Unit tests for BenchmarkConfig, CLI, and benchmark runner functionality.

Covers:
  - 19 configs instantiation with correct priority and n_layouts
  - CLI parsing: --configs, --h-values, --mode, --shots, --seed, --priority, --export-configs
  - Prefix resolution: C5 → C5_full_pea_balanced
  - Priority filter: --priority P0,P1 filters correctly
  - Export-configs serialization to individual JSON files
  - ResultEnvelope schema completeness (including hw_calibration, derived stats)
  - Hypothesis verdict logic (CONFIRMED/REFUTED/INCONCLUSIVE boundaries)
  - Transfer ratio formula computation
  - Affine on raw correction boundary cases
  - ClassicalSolver cache hit verification

Validates: Requirements 1.1, 1.2, 1.7, 1.8, 2.1, 2.3, 2.8, 2.9, 4.2, 15.1, 18.2, 19.3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.experiment_runners.hardware.benchmark_configs import (
    BENCHMARK_CONFIGS,
    BenchmarkConfig,
)
from scripts.experiment_runners.hardware.run_mitigation_benchmark import (
    _build_envelope,
    _classical_cache,
    _get_exact_energy,
    apply_affine_on_raw,
    export_configs,
    parse_args,
    resolve_configs,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. All 19 configs instantiate correctly with priority and n_layouts
# ═══════════════════════════════════════════════════════════════════════════════


class TestBenchmarkConfigInstantiation:
    """Validates Requirement 1.1, 1.2, 1.7, 1.8."""

    def test_exactly_19_configs_defined(self):
        """BENCHMARK_CONFIGS registry has at least 19 entries (grows as new configs added)."""
        assert len(BENCHMARK_CONFIGS) >= 19

    @pytest.mark.parametrize("config_id", list(BENCHMARK_CONFIGS.keys()))
    def test_each_config_has_valid_priority(self, config_id: str):
        """Each config's priority is in [0, 4]."""
        config = BENCHMARK_CONFIGS[config_id]
        assert 0 <= config.priority <= 4

    @pytest.mark.parametrize("config_id", list(BENCHMARK_CONFIGS.keys()))
    def test_each_config_has_valid_n_layouts(self, config_id: str):
        """Each config's n_layouts >= 1."""
        config = BENCHMARK_CONFIGS[config_id]
        assert config.n_layouts >= 1

    def test_c0_raw_uses_n_layouts_1(self):
        """C0_raw uses n_layouts=1 (no layout averaging needed)."""
        assert BENCHMARK_CONFIGS["C0_raw"].n_layouts == 1

    def test_c5_uses_default_n_layouts_3(self):
        """C5_full_pea_balanced uses default n_layouts=3."""
        assert BENCHMARK_CONFIGS["C5_full_pea_balanced"].n_layouts == 3

    def test_priority_0_configs_are_critical(self):
        """P0 configs: C0_raw, C3_full_gf, C5_full_pea_balanced."""
        p0 = [cid for cid, c in BENCHMARK_CONFIGS.items() if c.priority == 0]
        assert set(p0) == {"C0_raw", "C3_full_gf", "C5_full_pea_balanced"}

    def test_invalid_config_id_raises(self):
        """Invalid config_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid config_id"):
            BenchmarkConfig(config_id="INVALID_CONFIG")

    def test_invalid_priority_raises(self):
        """Priority outside [0, 4] raises ValueError."""
        with pytest.raises(ValueError, match="priority must be 0-4"):
            BenchmarkConfig(config_id="C0_raw", priority=5)

    def test_invalid_n_layouts_raises(self):
        """n_layouts < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_layouts must be >= 1"):
            BenchmarkConfig(config_id="C0_raw", n_layouts=0)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CLI parsing tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCLIParsing:
    """Validates Requirement 2.1, 2.8, 2.9."""

    def test_default_args(self):
        """Default arguments parse correctly."""
        with patch.object(sys, "argv", ["prog"]):
            args = parse_args()
        assert args.mode == "fake_backend"
        assert args.configs is None
        assert args.h_values == "3.25,3.5,3.75,4.0"
        assert args.shots == 16384
        assert args.seed == 42
        assert args.priority is None
        assert args.export_configs is False

    def test_configs_flag(self):
        """--configs parses CSV string."""
        with patch.object(sys, "argv", ["prog", "--configs", "C0,C5,C12"]):
            args = parse_args()
        assert args.configs == "C0,C5,C12"

    def test_h_values_flag(self):
        """--h-values parses custom values."""
        with patch.object(sys, "argv", ["prog", "--h-values", "3.0,3.5,4.0"]):
            args = parse_args()
        assert args.h_values == "3.0,3.5,4.0"

    def test_mode_hardware(self):
        """--mode hardware is accepted."""
        with patch.object(sys, "argv", ["prog", "--mode", "hardware"]):
            args = parse_args()
        assert args.mode == "hardware"

    def test_shots_flag(self):
        """--shots parses custom value."""
        with patch.object(sys, "argv", ["prog", "--shots", "8192"]):
            args = parse_args()
        assert args.shots == 8192

    def test_seed_flag(self):
        """--seed parses custom value."""
        with patch.object(sys, "argv", ["prog", "--seed", "123"]):
            args = parse_args()
        assert args.seed == 123

    def test_priority_flag(self):
        """--priority parses CSV of priority levels."""
        with patch.object(sys, "argv", ["prog", "--priority", "P0,P1"]):
            args = parse_args()
        assert args.priority == "P0,P1"

    def test_export_configs_flag(self):
        """--export-configs is a boolean store_true flag."""
        with patch.object(sys, "argv", ["prog", "--export-configs"]):
            args = parse_args()
        assert args.export_configs is True


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Prefix resolution: C5 → C5_full_pea_balanced
# ═══════════════════════════════════════════════════════════════════════════════


class TestPrefixResolution:
    """Validates Requirement 2.3 — config shortname prefix matching."""

    def test_c5_resolves_to_full_name(self):
        """C5 prefix matches C5_full_pea_balanced."""
        args = argparse.Namespace(configs="C5", priority=None)
        resolved = resolve_configs(args)
        assert resolved == ["C5_full_pea_balanced"]

    def test_c0_resolves_to_c0_raw(self):
        """C0 prefix matches C0_raw."""
        args = argparse.Namespace(configs="C0", priority=None)
        resolved = resolve_configs(args)
        assert resolved == ["C0_raw"]

    def test_c1_resolves_multiple(self):
        """C1 prefix matches C1_dd_only, C10_kitchen_sink, etc."""
        args = argparse.Namespace(configs="C1", priority=None)
        resolved = resolve_configs(args)
        # C1 prefix matches: C1_dd_only, C10_kitchen_sink, C11_mitiq_zne,
        # C12_mitiq_cdr, C13_mitiq_ddd_zne, C14_dd_mitiq_cdr, C15_pea_no_affine,
        # C16_aqc_pea, C17_aqc_mitiq_cdr, C18_aqc_raw
        assert "C1_dd_only" in resolved
        assert len(resolved) > 1

    def test_full_config_id_resolves_exactly(self):
        """Full config_id resolves to itself."""
        args = argparse.Namespace(configs="C5_full_pea_balanced", priority=None)
        resolved = resolve_configs(args)
        assert resolved == ["C5_full_pea_balanced"]

    def test_multiple_prefixes_csv(self):
        """Multiple CSV prefixes resolve correctly."""
        args = argparse.Namespace(configs="C0,C5,C3", priority=None)
        resolved = resolve_configs(args)
        assert "C0_raw" in resolved
        assert "C5_full_pea_balanced" in resolved
        assert "C3_full_gf" in resolved

    def test_invalid_prefix_raises(self):
        """Non-matching prefix raises ValueError."""
        args = argparse.Namespace(configs="INVALID", priority=None)
        with pytest.raises(ValueError, match="No config matches shortname"):
            resolve_configs(args)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Priority filter: --priority P0,P1 executes only matching configs
# ═══════════════════════════════════════════════════════════════════════════════


class TestPriorityFilter:
    """Validates Requirement 2.8 — priority-based config filtering."""

    def test_p0_returns_3_configs(self):
        """--priority P0 returns exactly the 3 P0 configs."""
        args = argparse.Namespace(configs=None, priority="P0")
        resolved = resolve_configs(args)
        assert len(resolved) == 3
        for config_id in resolved:
            assert BENCHMARK_CONFIGS[config_id].priority == 0

    def test_p0_p1_combined(self):
        """--priority P0,P1 returns P0 and P1 configs."""
        args = argparse.Namespace(configs=None, priority="P0,P1")
        resolved = resolve_configs(args)
        for config_id in resolved:
            assert BENCHMARK_CONFIGS[config_id].priority in {0, 1}
        # At least the 3 P0 + P1 configs
        assert len(resolved) >= 3

    def test_priority_and_configs_combined(self):
        """--priority P0 --configs C0 intersects correctly."""
        args = argparse.Namespace(configs="C0", priority="P0")
        resolved = resolve_configs(args)
        assert resolved == ["C0_raw"]

    def test_priority_filter_excludes_higher(self):
        """--priority P0 excludes P1+ configs."""
        args = argparse.Namespace(configs=None, priority="P0")
        resolved = resolve_configs(args)
        for config_id in resolved:
            assert BENCHMARK_CONFIGS[config_id].priority == 0

    def test_no_filters_returns_all_19(self):
        """No --priority, no --configs returns all 19 configs."""
        args = argparse.Namespace(configs=None, priority=None)
        resolved = resolve_configs(args)
        assert len(resolved) == len(BENCHMARK_CONFIGS)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Export-configs serialization to individual JSON files
# ═══════════════════════════════════════════════════════════════════════════════


class TestExportConfigs:
    """Validates Requirement 2.9 — config export to JSON."""

    def test_exports_19_json_files(self, tmp_path: Path):
        """export_configs creates one JSON per config (19 total)."""
        export_configs(tmp_path)
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == len(BENCHMARK_CONFIGS)

    def test_each_file_named_by_config_id(self, tmp_path: Path):
        """Each JSON file is named {config_id}.json."""
        export_configs(tmp_path)
        for config_id in BENCHMARK_CONFIGS:
            assert (tmp_path / f"{config_id}.json").exists()

    def test_json_content_has_config_id(self, tmp_path: Path):
        """Each exported JSON contains the correct config_id field."""
        export_configs(tmp_path)
        for config_id in BENCHMARK_CONFIGS:
            data = json.loads((tmp_path / f"{config_id}.json").read_text())
            assert data["config_id"] == config_id

    def test_json_content_has_priority(self, tmp_path: Path):
        """Each exported JSON contains the priority field."""
        export_configs(tmp_path)
        data = json.loads((tmp_path / "C5_full_pea_balanced.json").read_text())
        assert data["priority"] == 0

    def test_creates_output_dir(self, tmp_path: Path):
        """export_configs creates non-existent output directory."""
        nested = tmp_path / "sub" / "dir"
        export_configs(nested)
        assert nested.exists()
        assert len(list(nested.glob("*.json"))) == len(BENCHMARK_CONFIGS)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ResultEnvelope schema completeness
# ═══════════════════════════════════════════════════════════════════════════════


class TestResultEnvelope:
    """Validates Requirement 4.2, 15.1 — ResultEnvelope structure."""

    def _make_envelope(
        self, config_id: str = "C5_full_pea_balanced", hw_cal: dict | None = None
    ) -> dict:
        """Build a ResultEnvelope with synthetic data."""
        config = BENCHMARK_CONFIGS[config_id]
        circuit_stats = {
            "depth_logical": 20,
            "depth_transpiled": 45,
            "n_2q_gates": 34,
            "n_1q_gates": 60,
            "depth_2q": 12,
            "max_idle_stretch": 5,
            "idle_cycles_per_qubit": 3.2,
            "parallelism_ratio": 2.8,
            "gate_density_2q": 0.28,
            "circuit_depth_with_dd_estimate": 50,
            "routing_overhead_pct": 12.5,
            "transpiled_vs_logical_ratio": 2.25,
        }
        error_budget = {"fidelity_estimate": 0.85}
        execution_result = {
            "e_mitigated": -12.5,
            "e_raw": -10.0,
            "e_exact": -13.0,
            "delta_e_gap": 0.025,
            "improvement_vs_raw": 0.83,
            "zne_r2": 0.998,
            "phase_label": "ordered",
            "correct_label": True,
            "per_site_magnetization_std": 0.02,
            "energy_within_physical_bounds": True,
            "shots": 16384,
        }
        return _build_envelope(
            config=config,
            h_value=3.5,
            mode="fake_backend",
            seed=42,
            circuit_stats=circuit_stats,
            error_budget=error_budget,
            execution_result=execution_result,
            wall_time_s=12.5,
            hardware_calibration=hw_cal,
        )

    def test_has_required_top_level_sections(self):
        """Envelope has benchmark_metadata, circuit_stats, timing, results, shots."""
        env = self._make_envelope()
        assert "benchmark_metadata" in env
        assert "circuit_stats" in env
        assert "timing" in env
        assert "results" in env
        assert "shots" in env
        assert "mitigation_config" in env

    def test_benchmark_metadata_fields(self):
        """benchmark_metadata contains all required fields."""
        env = self._make_envelope()
        meta = env["benchmark_metadata"]
        assert meta["config_id"] == "C5_full_pea_balanced"
        assert meta["execution_mode"] == "fake_backend"
        assert meta["h_value"] == 3.5
        assert meta["benchmark_version"] == "1.0"
        assert meta["seed"] == 42
        assert "timestamp" in meta

    def test_circuit_stats_derived_fields(self):
        """circuit_stats contains derived metrics."""
        env = self._make_envelope()
        cs = env["circuit_stats"]
        assert "circuit_depth_with_dd_estimate" in cs
        assert "routing_overhead_pct" in cs
        assert "transpiled_vs_logical_ratio" in cs
        assert cs["optimization_level"] == 2

    def test_timing_fields(self):
        """timing section has wall_time_s."""
        env = self._make_envelope()
        assert env["timing"]["wall_time_s"] == 12.5

    def test_results_fields_complete(self):
        """results section has all energy and label fields."""
        env = self._make_envelope()
        r = env["results"]
        assert r["e_mitigated"] == -12.5
        assert r["e_raw"] == -10.0
        assert r["e_exact"] == -13.0
        assert r["delta_e_gap"] == 0.025
        assert r["phase_label"] == "ordered"
        assert r["correct_label"] is True
        assert "per_site_magnetization_std" in r
        assert "energy_within_physical_bounds" in r

    def test_hardware_calibration_null_for_fake(self):
        """hardware_calibration is None when mode=fake_backend."""
        env = self._make_envelope()
        assert env["hardware_calibration"] is None

    def test_hardware_calibration_present_when_provided(self):
        """hardware_calibration section included when populated."""
        hw_cal = {
            "t1_mean_layout": 250.0,
            "t2_mean_layout": 150.0,
            "cx_error_mean_layout": 0.008,
            "readout_error_mean": 0.015,
            "calibration_age_hours": 2.5,
            "job_execution_time_s": 45.0,
        }
        env = self._make_envelope(hw_cal=hw_cal)
        assert env["hardware_calibration"] == hw_cal

    def test_aqc_metrics_for_aqc_config(self):
        """AQC configs include aqc_metrics in envelope."""
        config = BENCHMARK_CONFIGS["C16_aqc_pea"]
        circuit_stats = {
            "depth_logical": 20,
            "depth_transpiled": 30,
            "n_2q_gates": 20,
            "depth_2q": 8,
            "max_idle_stretch": 3,
        }
        aqc_metrics = {
            "aqc_fidelity": 0.999,
            "aqc_n_2q_compressed": 18,
            "aqc_2q_reduction_pct": 47.0,
            "aqc_compression_time_s": 1.2,
            "aqc_fallback_triggered": False,
        }
        env = _build_envelope(
            config=config,
            h_value=3.0,
            mode="fake_backend",
            seed=42,
            circuit_stats=circuit_stats,
            error_budget={"fidelity_estimate": 0.9},
            execution_result={
                "e_mitigated": -12.0,
                "e_raw": -10.0,
                "e_exact": -13.0,
                "delta_e_gap": 0.04,
                "shots": 16384,
            },
            wall_time_s=5.0,
            aqc_metrics=aqc_metrics,
        )
        assert "aqc_metrics" in env
        assert env["aqc_metrics"]["aqc_fidelity"] == 0.999


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Hypothesis verdict logic (CONFIRMED/REFUTED/INCONCLUSIVE)
# ═══════════════════════════════════════════════════════════════════════════════


class TestHypothesisVerdicts:
    """Validates Requirement 18.2 — hypothesis evaluation logic."""

    def _make_analyzer_with_entries(self, entries: list[dict]):
        """Create an analyzer pre-loaded with synthetic entries."""
        from project_health.analysis.mitigation_benchmark_analyzer import (
            MitigationBenchmarkAnalyzer,
        )

        analyzer = MitigationBenchmarkAnalyzer.__new__(MitigationBenchmarkAnalyzer)
        analyzer.results_dir = Path("/tmp/fake")
        analyzer.entries = entries
        analyzer.errors = []
        analyzer.derived_metrics = {}
        return analyzer

    def _make_entry(self, config_id: str, delta_e_gap: float, mode: str = "fake_backend") -> dict:
        """Build a minimal valid result envelope entry."""
        return {
            "benchmark_metadata": {
                "config_id": config_id,
                "h_value": 3.5,
                "execution_mode": mode,
            },
            "circuit_stats": {"n_2q_gates": 34},
            "results": {
                "delta_e_gap": delta_e_gap,
                "e_mitigated": -12.0,
                "e_raw": -10.0,
                "correct_label": True,
            },
            "shots": 16384,
        }

    def test_confirmed_when_b_clearly_lower(self):
        """CONFIRMED when config_b delta_e_gap is clearly lower than config_a."""
        # H1: C0_raw vs C1_dd_only, direction="lower" (C1 should be lower)
        entries = [
            self._make_entry("C0_raw", 0.20),
            self._make_entry("C0_raw", 0.22),
            self._make_entry("C0_raw", 0.21),
            self._make_entry("C1_dd_only", 0.05),
            self._make_entry("C1_dd_only", 0.04),
            self._make_entry("C1_dd_only", 0.06),
        ]
        analyzer = self._make_analyzer_with_entries(entries)
        verdicts = analyzer.compute_hypothesis_verdicts()

        h1 = next(v for v in verdicts if v["hypothesis_id"] == "H1")
        assert h1["verdict"] == "CONFIRMED"

    def test_refuted_when_opposite_direction(self):
        """REFUTED when observed direction is opposite to expected."""
        # H1: C0_raw vs C1_dd_only, direction="lower"
        # But C1 is HIGHER than C0 → refuted
        entries = [
            self._make_entry("C0_raw", 0.05),
            self._make_entry("C0_raw", 0.04),
            self._make_entry("C0_raw", 0.06),
            self._make_entry("C1_dd_only", 0.20),
            self._make_entry("C1_dd_only", 0.22),
            self._make_entry("C1_dd_only", 0.21),
        ]
        analyzer = self._make_analyzer_with_entries(entries)
        verdicts = analyzer.compute_hypothesis_verdicts()

        h1 = next(v for v in verdicts if v["hypothesis_id"] == "H1")
        assert h1["verdict"] == "REFUTED"

    def test_inconclusive_when_within_noise(self):
        """INCONCLUSIVE when difference is within shot noise estimate."""
        # Same values for both configs → noise dominates
        entries = [
            self._make_entry("C0_raw", 0.10),
            self._make_entry("C0_raw", 0.10),
            self._make_entry("C0_raw", 0.10),
            self._make_entry("C1_dd_only", 0.10),
            self._make_entry("C1_dd_only", 0.10),
            self._make_entry("C1_dd_only", 0.10),
        ]
        analyzer = self._make_analyzer_with_entries(entries)
        verdicts = analyzer.compute_hypothesis_verdicts()

        h1 = next(v for v in verdicts if v["hypothesis_id"] == "H1")
        assert h1["verdict"] == "INCONCLUSIVE"

    def test_inconclusive_when_no_data(self):
        """INCONCLUSIVE when no entries exist for a config."""
        analyzer = self._make_analyzer_with_entries([])
        verdicts = analyzer.compute_hypothesis_verdicts()

        # All hypotheses should be INCONCLUSIVE with no data
        for v in verdicts:
            assert v["verdict"] == "INCONCLUSIVE"

    def test_threshold_hypothesis_confirmed(self):
        """H18 CONFIRMED when delta_e_gap < 3%."""
        entries = [
            self._make_entry("C5_full_pea_balanced", 0.02),
            self._make_entry("C5_full_pea_balanced", 0.015),
            self._make_entry("C5_full_pea_balanced", 0.025),
        ]
        analyzer = self._make_analyzer_with_entries(entries)
        verdicts = analyzer.compute_hypothesis_verdicts()

        h18 = next(v for v in verdicts if v["hypothesis_id"] == "H18")
        assert h18["verdict"] == "CONFIRMED"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Transfer ratio formula computation
# ═══════════════════════════════════════════════════════════════════════════════


class TestTransferRatio:
    """Validates Requirement 16.1 — transfer_ratio = ΔE/gap(hw) / ΔE/gap(sim)."""

    def _make_analyzer_with_entries(self, entries: list[dict]):
        """Create an analyzer pre-loaded with synthetic entries."""
        from project_health.analysis.mitigation_benchmark_analyzer import (
            MitigationBenchmarkAnalyzer,
        )

        analyzer = MitigationBenchmarkAnalyzer.__new__(MitigationBenchmarkAnalyzer)
        analyzer.results_dir = Path("/tmp/fake")
        analyzer.entries = entries
        analyzer.errors = []
        analyzer.derived_metrics = {}
        return analyzer

    def _make_entry(self, config_id: str, delta_e_gap: float, mode: str) -> dict:
        return {
            "benchmark_metadata": {
                "config_id": config_id,
                "h_value": 3.5,
                "execution_mode": mode,
            },
            "circuit_stats": {"n_2q_gates": 34},
            "results": {
                "delta_e_gap": delta_e_gap,
                "e_mitigated": -12.0,
                "e_raw": -10.0,
            },
            "shots": 16384,
        }

    def test_transfer_ratio_known_values(self):
        """Transfer ratio = hw_mean / sim_mean with known values."""
        entries = [
            self._make_entry("C5_full_pea_balanced", 0.02, "fake_backend"),
            self._make_entry("C5_full_pea_balanced", 0.04, "fake_backend"),
            # mean sim = 0.03
            self._make_entry("C5_full_pea_balanced", 0.06, "hardware"),
            self._make_entry("C5_full_pea_balanced", 0.09, "hardware"),
            # mean hw = 0.075
        ]
        analyzer = self._make_analyzer_with_entries(entries)
        ratios = analyzer.compute_transfer_ratios()

        assert "C5_full_pea_balanced" in ratios
        expected = 0.075 / 0.03  # = 2.5
        assert abs(ratios["C5_full_pea_balanced"] - expected) < 1e-10

    def test_empty_when_no_dual_mode(self):
        """Returns empty dict when no config has results in both modes."""
        entries = [
            self._make_entry("C0_raw", 0.10, "fake_backend"),
            self._make_entry("C5_full_pea_balanced", 0.05, "hardware"),
        ]
        analyzer = self._make_analyzer_with_entries(entries)
        ratios = analyzer.compute_transfer_ratios()
        assert ratios == {}

    def test_multiple_configs_transfer_ratio(self):
        """Transfer ratio computed for each dual-mode config independently."""
        entries = [
            self._make_entry("C0_raw", 0.20, "fake_backend"),
            self._make_entry("C0_raw", 0.40, "hardware"),
            self._make_entry("C5_full_pea_balanced", 0.02, "fake_backend"),
            self._make_entry("C5_full_pea_balanced", 0.06, "hardware"),
        ]
        analyzer = self._make_analyzer_with_entries(entries)
        ratios = analyzer.compute_transfer_ratios()

        assert len(ratios) == 2
        assert abs(ratios["C0_raw"] - 2.0) < 1e-10
        assert abs(ratios["C5_full_pea_balanced"] - 3.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Affine on raw correction — boundary cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestAffineOnRawBoundary:
    """Additional boundary tests for apply_affine_on_raw (extends test_apply_affine_on_raw.py)."""

    def test_e_raw_at_exact_energy(self):
        """When e_raw == e_exact, correction should keep it at e_exact."""
        config = BenchmarkConfig(
            config_id="C1_dd_only",
            dd_enabled=True,
            dd_sequence="XpXm",
            affine_enabled=True,
            priority=1,
        )
        e_exact = -12.0
        e_upper = 5.0
        execution_result = {"e_raw": -12.0, "e_mitigated": None}

        result = apply_affine_on_raw(config, execution_result, e_exact, e_upper)
        assert result["e_mitigated"] is not None
        assert abs(result["e_mitigated"] - e_exact) < 1e-8

    def test_e_raw_at_upper_bound(self):
        """When e_raw == e_upper, correction keeps it within bounds."""
        config = BenchmarkConfig(
            config_id="C1_dd_only",
            dd_enabled=True,
            dd_sequence="XpXm",
            affine_enabled=True,
            priority=1,
        )
        e_exact = -12.0
        e_upper = 5.0
        execution_result = {"e_raw": 5.0, "e_mitigated": None}

        result = apply_affine_on_raw(config, execution_result, e_exact, e_upper)
        assert result["e_mitigated"] is not None
        assert e_exact <= result["e_mitigated"] <= e_upper

    def test_e_raw_far_below_ground(self):
        """When e_raw << e_exact (overshoot below), clips to bounds."""
        config = BenchmarkConfig(
            config_id="C2_dd_tw",
            dd_enabled=True,
            dd_sequence="XpXm",
            twirling_num_randomizations=32,
            trex_enabled=True,
            affine_enabled=True,
            priority=1,
        )
        e_exact = -12.0
        e_upper = 5.0
        execution_result = {"e_raw": -50.0, "e_mitigated": None}

        result = apply_affine_on_raw(config, execution_result, e_exact, e_upper)
        # Should be clipped to within physical bounds
        assert result["e_mitigated"] is not None
        assert result["e_mitigated"] >= e_exact

    def test_e_raw_far_above_upper(self):
        """When e_raw >> e_upper (far positive), clips to bounds."""
        config = BenchmarkConfig(
            config_id="C1_dd_only",
            dd_enabled=True,
            dd_sequence="XpXm",
            affine_enabled=True,
            priority=1,
        )
        e_exact = -12.0
        e_upper = 5.0
        execution_result = {"e_raw": 100.0, "e_mitigated": None}

        result = apply_affine_on_raw(config, execution_result, e_exact, e_upper)
        assert result["e_mitigated"] is not None
        assert result["e_mitigated"] <= e_upper


# ═══════════════════════════════════════════════════════════════════════════════
# 10. ClassicalSolver cache hit verification
# ═══════════════════════════════════════════════════════════════════════════════


class TestClassicalSolverCache:
    """Validates Requirement 19.3 — cache ensures bit-exact reuse across configs."""

    def test_cache_returns_same_tuple(self):
        """Second call to _get_exact_energy returns same (e_exact, gap) tuple."""
        # Clear cache to start fresh
        _classical_cache.clear()

        result1 = _get_exact_energy(3.5)
        result2 = _get_exact_energy(3.5)

        assert result1 == result2
        assert result1[0] == result2[0]  # e_exact
        assert result1[1] == result2[1]  # gap

    def test_cache_hit_after_first_call(self):
        """After first call, the h_value is in _classical_cache."""
        _classical_cache.clear()

        _get_exact_energy(4.0)
        assert 4.0 in _classical_cache

    def test_different_h_values_cached_independently(self):
        """Different h-values produce different cached entries."""
        _classical_cache.clear()

        result_a = _get_exact_energy(3.25)
        result_b = _get_exact_energy(3.75)

        assert 3.25 in _classical_cache
        assert 3.75 in _classical_cache
        # Different h → different energies
        assert result_a[0] != result_b[0]

    def test_returns_negative_ground_energy(self):
        """Ground energy for TFIM is negative (ferromagnetic coupling)."""
        _classical_cache.clear()
        e_exact, gap = _get_exact_energy(3.5)
        assert e_exact < 0

    def test_gap_is_positive(self):
        """Energy gap (E1 - E0) is always positive."""
        _classical_cache.clear()
        e_exact, gap = _get_exact_energy(3.5)
        assert gap > 0
