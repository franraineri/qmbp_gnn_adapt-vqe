"""Integration tests for the mitigation benchmark runner and analyzer.

End-to-end tests covering CLI resolution, execution with mocked backends,
idempotency, manifest persistence, and post-analysis with synthetic results.

Validates: Requirements 2.2, 2.8, 2.9, 4.8, 5.2, 16.1, 17.1, 18.2
"""

from __future__ import annotations

import argparse
import json
from unittest.mock import MagicMock

import pytest

from scripts.experiment_runners.hardware.benchmark_configs import (
    BENCHMARK_CONFIGS,
)
from scripts.experiment_runners.hardware.run_mitigation_benchmark import (
    append_to_manifest,
    export_configs,
    resolve_configs,
    run_single_config,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_namespace(**kwargs) -> argparse.Namespace:
    """Create a Namespace with defaults matching the benchmark CLI."""
    defaults = {
        "mode": "fake_backend",
        "configs": None,
        "h_values": "3.25,3.5,3.75,4.0",
        "shots": 16384,
        "seed": 42,
        "priority": None,
        "export_configs": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _synthetic_result_envelope(
    config_id: str,
    h_value: float,
    mode: str = "fake_backend",
    delta_e_gap: float = 0.05,
    e_mitigated: float = -12.5,
    e_raw: float = -11.0,
    e_exact: float = -13.0,
    correct_label: bool = True,
) -> dict:
    """Build a minimal but valid synthetic ResultEnvelope for testing."""
    return {
        "benchmark_metadata": {
            "config_id": config_id,
            "execution_mode": mode,
            "h_value": h_value,
            "timestamp": "2026-06-18T10:30:00Z",
            "benchmark_version": "1.0",
            "seed": 42,
        },
        "circuit_stats": {
            "depth_logical": 20,
            "depth_transpiled": 45,
            "n_2q_gates": 34,
            "n_1q_gates": 50,
            "depth_2q": 10,
            "optimization_level": 2,
            "fidelity_estimate": 0.85,
            "idle_cycles_per_qubit": 3.5,
            "max_idle_stretch": 8,
            "parallelism_ratio": 3.4,
            "gate_density_2q": 0.34,
        },
        "timing": {
            "wall_time_s": 12.5,
            "qpu_seconds": None,
            "noise_learning_time_s": None,
        },
        "results": {
            "e_mitigated": e_mitigated,
            "e_raw": e_raw,
            "e_exact": e_exact,
            "delta_e_gap": delta_e_gap,
            "improvement_vs_raw": 0.75,
            "zne_r2": 0.998,
            "phase_label": "paramagnetic",
            "correct_label": correct_label,
            "per_site_magnetization_std": 0.02,
            "energy_within_physical_bounds": True,
        },
        "shots": 16384,
        "mitigation_config": {"config_id": config_id},
        "hardware_calibration": None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: Smoke — single config (C0_raw) with mocked execution
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_smoke_single_config_mocked(tmp_path, monkeypatch):
    """Smoke test: run_single_config with C0_raw, h=3.5, mocked execution.

    Verifies that a JSON result file is created and the manifest is updated.
    Uses mocked route_execution to avoid actual FakeTorino overhead.
    """
    # Patch RESULTS_BASE and MANIFEST_PATH to use tmp_path
    monkeypatch.setattr(
        "scripts.experiment_runners.hardware.run_mitigation_benchmark.RESULTS_BASE",
        tmp_path,
    )
    monkeypatch.setattr(
        "scripts.experiment_runners.hardware.run_mitigation_benchmark.MANIFEST_PATH",
        tmp_path / "manifest.json",
    )

    # Mock route_execution to return synthetic result
    mock_result = {
        "e_raw": -11.0,
        "e_mitigated": None,
        "shots": 16384,
        "zne_r2": None,
        "_job": None,
    }
    monkeypatch.setattr(
        "scripts.experiment_runners.hardware.run_mitigation_benchmark.route_execution",
        lambda *args, **kwargs: mock_result,
    )

    # Mock _get_backend to avoid FakeTorino import
    mock_backend = MagicMock()
    mock_backend.num_qubits = 133
    monkeypatch.setattr(
        "scripts.experiment_runners.hardware.run_mitigation_benchmark._get_backend",
        lambda mode: mock_backend,
    )

    # Mock transpilation pipeline
    from unittest.mock import MagicMock as MM

    mock_circuit = MM()
    mock_circuit.data = []
    mock_circuit.layout = None
    mock_circuit.depth.return_value = 45

    monkeypatch.setattr(
        "scripts.experiment_runners.hardware.run_mitigation_benchmark._build_hva_circuit",
        lambda h: mock_circuit,
    )

    # Mock generate_preset_pass_manager (imported at module level in benchmark runner)
    mock_pm = MM()
    mock_transpiled = MM()
    mock_transpiled.data = []
    mock_transpiled.layout = MM()
    mock_transpiled.depth.return_value = 45
    mock_pm.run.return_value = mock_transpiled
    monkeypatch.setattr(
        "scripts.experiment_runners.hardware.run_mitigation_benchmark.generate_preset_pass_manager",
        lambda **kwargs: mock_pm,
    )

    # Mock transpiled_circuit_stats and compute_error_budget
    monkeypatch.setattr(
        "scripts.experiment_runners.hardware.run_mitigation_benchmark.transpiled_circuit_stats",
        lambda circuit: {
            "depth": 45,
            "depth_transpiled": 45,
            "depth_logical": 20,
            "n_2q_gates": 34,
            "n_1q_gates": 50,
            "depth_2q": 10,
            "max_idle_stretch": 5,
        },
    )
    monkeypatch.setattr(
        "scripts.experiment_runners.hardware.run_mitigation_benchmark.compute_error_budget",
        lambda circuit, backend: {"fidelity_estimate": 0.85},
    )

    # Mock HamiltonianBuilder + observable mapping
    mock_H = MM()
    mock_H.apply_layout.return_value = mock_H
    monkeypatch.setattr(
        "scripts.experiment_runners.hardware.run_mitigation_benchmark.HamiltonianBuilder",
        lambda: MM(build=lambda lattice: mock_H),
    )
    monkeypatch.setattr(
        "scripts.experiment_runners.hardware.run_mitigation_benchmark.make_lattice",
        lambda *args, **kwargs: MM(),
    )

    # Mock ClassicalSolver cache
    monkeypatch.setattr(
        "scripts.experiment_runners.hardware.run_mitigation_benchmark._get_exact_energy",
        lambda h: (-13.0, 1.5),
    )

    config = BENCHMARK_CONFIGS["C0_raw"]
    envelope = run_single_config(config, 3.5, "fake_backend", 16384, 42)

    # Verify result file created
    result_files = list(tmp_path.rglob("*.json"))
    # Filter out manifest.json
    result_files = [f for f in result_files if f.name != "manifest.json"]
    assert len(result_files) >= 1, "Expected at least one result JSON file"

    # Verify manifest created
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists(), "Manifest should exist after execution"
    manifest = json.loads(manifest_path.read_text())
    assert isinstance(manifest, list)
    assert len(manifest) == 1
    assert manifest[0]["config_id"] == "C0_raw"

    # Verify envelope structure
    assert envelope != {}, "Envelope should not be empty (not skipped)"
    assert "benchmark_metadata" in envelope
    assert envelope["benchmark_metadata"]["config_id"] == "C0_raw"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: Multi-config with --configs filter
# ═══════════════════════════════════════════════════════════════════════════════


def test_multi_config_filter():
    """--configs C0,C5 resolves to exactly 2 configs: C0_raw and C5_full_pea_balanced."""
    args = _make_namespace(configs="C0,C5")
    resolved = resolve_configs(args)
    assert len(resolved) == 2
    assert "C0_raw" in resolved
    assert "C5_full_pea_balanced" in resolved


def test_multi_config_filter_three():
    """--configs C0,C5,C12 resolves to exactly 3 configs."""
    args = _make_namespace(configs="C0,C5,C12")
    resolved = resolve_configs(args)
    assert len(resolved) == 3
    assert "C0_raw" in resolved
    assert "C5_full_pea_balanced" in resolved
    assert "C12_mitiq_cdr" in resolved


def test_config_filter_invalid_raises():
    """--configs with invalid shortname raises ValueError."""
    args = _make_namespace(configs="C99")
    with pytest.raises(ValueError, match="No config matches"):
        resolve_configs(args)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: Priority filter with --priority P0
# ═══════════════════════════════════════════════════════════════════════════════


def test_priority_filter_p0():
    """--priority P0 resolves to exactly 3 configs (priority=0)."""
    args = _make_namespace(priority="P0")
    resolved = resolve_configs(args)
    # P0 configs: C0_raw, C3_full_gf, C5_full_pea_balanced
    assert len(resolved) == 3
    assert "C0_raw" in resolved
    assert "C3_full_gf" in resolved
    assert "C5_full_pea_balanced" in resolved
    # All resolved configs should have priority 0
    for config_id in resolved:
        assert BENCHMARK_CONFIGS[config_id].priority == 0


def test_priority_filter_p0_p1():
    """--priority P0,P1 resolves to P0 + P1 configs."""
    args = _make_namespace(priority="P0,P1")
    resolved = resolve_configs(args)
    for config_id in resolved:
        assert BENCHMARK_CONFIGS[config_id].priority in {0, 1}
    # P0 has 3, P1 has 4 configs
    assert len(resolved) == 7


def test_priority_and_config_filter_intersection():
    """--priority P0 --configs C0 returns intersection (C0_raw only)."""
    args = _make_namespace(priority="P0", configs="C0")
    resolved = resolve_configs(args)
    assert resolved == ["C0_raw"]


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: --export-configs output validation
# ═══════════════════════════════════════════════════════════════════════════════


def test_export_configs(tmp_path):
    """export_configs writes one JSON per config, each parseable with config_id."""
    output_dir = tmp_path / "configs"
    export_configs(output_dir)

    json_files = list(output_dir.glob("*.json"))
    assert len(json_files) == len(BENCHMARK_CONFIGS), (
        f"Expected {len(BENCHMARK_CONFIGS)} config JSONs, got {len(json_files)}"
    )

    # Each file should be valid JSON with a config_id field
    for json_file in json_files:
        data = json.loads(json_file.read_text())
        assert isinstance(data, dict)
        assert "config_id" in data
        # Filename should match config_id
        expected_name = f"{data['config_id']}.json"
        assert json_file.name == expected_name


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: Idempotency — double-run no overwrite
# ═══════════════════════════════════════════════════════════════════════════════


def test_idempotency_skip_existing(tmp_path, monkeypatch):
    """If result file already exists, run_single_config returns {} without executing.

    Validates Requirement 4.8: idempotency via skip.
    """
    # Patch RESULTS_BASE to use tmp_path
    monkeypatch.setattr(
        "scripts.experiment_runners.hardware.run_mitigation_benchmark.RESULTS_BASE",
        tmp_path,
    )

    # Create a fake result file at the expected path pattern
    config = BENCHMARK_CONFIGS["C0_raw"]
    # We need to ensure _build_result_path returns a path that exists
    fake_result_dir = tmp_path / "fake_backend" / "C0_raw"
    fake_result_dir.mkdir(parents=True)

    # Patch _build_result_path to return our fake existing file
    fake_result_file = fake_result_dir / "h3p5_run_20260618_103000.json"
    fake_result_file.write_text(json.dumps({"existing": True, "results": {"e_raw": -5.0}}))

    monkeypatch.setattr(
        "scripts.experiment_runners.hardware.run_mitigation_benchmark._build_result_path",
        lambda *args, **kwargs: fake_result_file,
    )

    # run_single_config should detect file exists and return {} (skip)
    result = run_single_config(config, 3.5, "fake_backend", 16384, 42)
    assert result == {}, "Should return empty dict when result already exists"

    # Verify original file content unchanged
    data = json.loads(fake_result_file.read_text())
    assert data == {"existing": True, "results": {"e_raw": -5.0}}, (
        "File content should not be modified"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6: Manifest accumulation across runs
# ═══════════════════════════════════════════════════════════════════════════════


def test_manifest_accumulation(tmp_path):
    """append_to_manifest accumulates entries correctly across multiple calls.

    Validates Requirement 5.2: manifest grows with each execution.
    """
    manifest_path = tmp_path / "manifest.json"

    entry_1 = {
        "config_id": "C0_raw",
        "execution_mode": "fake_backend",
        "h_value": 3.5,
        "timestamp": "2026-06-18T10:30:00Z",
        "result_path": "fake_backend/C0_raw/h3p5_run_20260618_103000.json",
        "delta_e_gap": 0.12,
        "correct_label": True,
    }
    entry_2 = {
        "config_id": "C5_full_pea_balanced",
        "execution_mode": "fake_backend",
        "h_value": 3.5,
        "timestamp": "2026-06-18T10:35:00Z",
        "result_path": "fake_backend/C5_full_pea_balanced/h3p5_run_20260618_103500.json",
        "delta_e_gap": 0.024,
        "correct_label": True,
    }

    # First append — creates manifest
    append_to_manifest(entry_1, manifest_path)
    data = json.loads(manifest_path.read_text())
    assert len(data) == 1
    assert data[0]["config_id"] == "C0_raw"

    # Second append — accumulates
    append_to_manifest(entry_2, manifest_path)
    data = json.loads(manifest_path.read_text())
    assert len(data) == 2
    assert data[0]["config_id"] == "C0_raw"
    assert data[1]["config_id"] == "C5_full_pea_balanced"


def test_manifest_creates_parent_dirs(tmp_path):
    """append_to_manifest creates parent directories if they don't exist."""
    manifest_path = tmp_path / "deep" / "nested" / "manifest.json"
    entry = {"config_id": "C0_raw", "h_value": 3.5}
    append_to_manifest(entry, manifest_path)
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert len(data) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7: Analyzer with synthetic results
# ═══════════════════════════════════════════════════════════════════════════════


class TestAnalyzerWithSyntheticResults:
    """Analyzer integration tests using synthetic result JSONs.

    Validates: Requirements 16.1, 17.1, 18.2
    """

    @pytest.fixture
    def results_dir(self, tmp_path):
        """Create a synthetic results directory with multiple configs and modes."""
        base = tmp_path / "results" / "mitigation_benchmark"

        # C0_raw — worse baseline (higher delta_e_gap)
        configs_data = {
            "C0_raw": {"delta_e_gap": 0.15, "e_mitigated": -11.5, "e_raw": -11.0},
            "C1_dd_only": {"delta_e_gap": 0.10, "e_mitigated": -12.0, "e_raw": -11.5},
            "C2_dd_tw": {"delta_e_gap": 0.08, "e_mitigated": -12.2, "e_raw": -11.5},
            "C3_full_gf": {"delta_e_gap": 0.05, "e_mitigated": -12.5, "e_raw": -11.5},
            "C4_full_pea_light": {"delta_e_gap": 0.04, "e_mitigated": -12.6, "e_raw": -11.5},
            "C5_full_pea_balanced": {"delta_e_gap": 0.025, "e_mitigated": -12.8, "e_raw": -11.5},
            "C6_full_pea_heavy": {"delta_e_gap": 0.022, "e_mitigated": -12.82, "e_raw": -11.5},
        }

        h_values = [3.25, 3.5, 3.75, 4.0]

        # Write fake_backend results
        for config_id, metrics in configs_data.items():
            config_dir = base / "fake_backend" / config_id
            config_dir.mkdir(parents=True, exist_ok=True)
            for h in h_values:
                envelope = _synthetic_result_envelope(
                    config_id=config_id,
                    h_value=h,
                    mode="fake_backend",
                    delta_e_gap=metrics["delta_e_gap"],
                    e_mitigated=metrics["e_mitigated"],
                    e_raw=metrics["e_raw"],
                )
                h_str = str(h).replace(".", "p")
                filepath = config_dir / f"h{h_str}_run_20260618_103000.json"
                filepath.write_text(json.dumps(envelope, indent=2))

        # Write hardware results for a subset (for transfer ratio testing)
        hw_configs = {
            "C0_raw": {"delta_e_gap": 0.20},
            "C3_full_gf": {"delta_e_gap": 0.07},
            "C5_full_pea_balanced": {"delta_e_gap": 0.035},
            "C1_dd_only": {"delta_e_gap": 0.14},
            "C2_dd_tw": {"delta_e_gap": 0.11},
            "C4_full_pea_light": {"delta_e_gap": 0.055},
        }
        for config_id, metrics in hw_configs.items():
            config_dir = base / "hardware" / config_id
            config_dir.mkdir(parents=True, exist_ok=True)
            for h in h_values:
                envelope = _synthetic_result_envelope(
                    config_id=config_id,
                    h_value=h,
                    mode="hardware",
                    delta_e_gap=metrics["delta_e_gap"],
                    e_mitigated=-12.0,
                    e_raw=-11.0,
                )
                h_str = str(h).replace(".", "p")
                filepath = config_dir / f"h{h_str}_run_20260618_120000.json"
                filepath.write_text(json.dumps(envelope, indent=2))

        return base

    def test_analyzer_scan_loads_entries(self, results_dir):
        """Analyzer scan loads all valid result JSONs from results directory."""
        from project_health.analysis.mitigation_benchmark_analyzer import (
            MitigationBenchmarkAnalyzer,
        )

        analyzer = MitigationBenchmarkAnalyzer(results_dir=results_dir)
        analyzer.scan()

        # 7 configs × 4 h-values (fake_backend) + 6 configs × 4 h-values (hardware)
        expected = 7 * 4 + 6 * 4
        assert len(analyzer.entries) == expected
        assert len(analyzer.errors) == 0

    def test_analyzer_transfer_ratios(self, results_dir):
        """compute_transfer_ratios returns ratio for dual-mode configs.

        Validates Requirement 16.1.
        """
        from project_health.analysis.mitigation_benchmark_analyzer import (
            MitigationBenchmarkAnalyzer,
        )

        analyzer = MitigationBenchmarkAnalyzer(results_dir=results_dir)
        analyzer.scan()
        ratios = analyzer.compute_transfer_ratios()

        # 6 configs have both fake_backend and hardware results
        assert len(ratios) == 6

        # C0_raw: hw=0.20, sim=0.15 → ratio = 0.20/0.15 ≈ 1.33
        assert pytest.approx(ratios["C0_raw"], rel=0.01) == 0.20 / 0.15

        # C5: hw=0.035, sim=0.025 → ratio = 0.035/0.025 = 1.4
        assert pytest.approx(ratios["C5_full_pea_balanced"], rel=0.01) == 0.035 / 0.025

        # All ratios should be > 1 (hardware generally worse than sim)
        for r in ratios.values():
            assert r > 1.0

    def test_analyzer_spearman_correlation(self, results_dir):
        """compute_spearman_correlation returns valid ρ with ≥5 dual-mode configs.

        Validates Requirement 16.1 (Spearman gating at ≥5 configs).
        """
        from project_health.analysis.mitigation_benchmark_analyzer import (
            MitigationBenchmarkAnalyzer,
        )

        analyzer = MitigationBenchmarkAnalyzer(results_dir=results_dir)
        analyzer.scan()
        rho = analyzer.compute_spearman_correlation()

        # 6 dual-mode configs → should compute (threshold is 5)
        assert rho is not None
        # Rankings are monotonically ordered → Spearman should be very high
        assert rho > 0.8

    def test_analyzer_sensitivity_curves(self, results_dir):
        """compute_sensitivity_curves returns PEA budget and twirling curves.

        Validates Requirement 17.1.
        """
        from project_health.analysis.mitigation_benchmark_analyzer import (
            MitigationBenchmarkAnalyzer,
        )

        analyzer = MitigationBenchmarkAnalyzer(results_dir=results_dir)
        analyzer.scan()
        curves = analyzer.compute_sensitivity_curves()

        # PEA budget curve: C4, C5, C6 all present → 3 data points
        pea = curves["pea_budget"]
        assert pea["sufficient_data"] is True
        assert len(pea["data_points"]) == 3
        # Points sorted by budget (ascending)
        budgets = [pt[0] for pt in pea["data_points"]]
        assert budgets == sorted(budgets)
        # Delta_e_gap should decrease as budget increases
        deltas = [pt[1] for pt in pea["data_points"]]
        assert deltas[0] > deltas[-1]

        # Twirling curve: C2(32), C5(48), C6(64) present → 3 points
        twirl = curves["twirling"]
        assert twirl["sufficient_data"] is True
        assert len(twirl["data_points"]) == 3

    def test_analyzer_hypothesis_verdicts(self, results_dir):
        """compute_hypothesis_verdicts returns valid verdicts for available data.

        Validates Requirement 18.2: verdicts are CONFIRMED/REFUTED/INCONCLUSIVE.
        """
        from project_health.analysis.mitigation_benchmark_analyzer import (
            MitigationBenchmarkAnalyzer,
        )

        analyzer = MitigationBenchmarkAnalyzer(results_dir=results_dir)
        analyzer.scan()
        verdicts = analyzer.compute_hypothesis_verdicts()

        # Should return one entry per hypothesis in HYPOTHESIS_MAP
        assert len(verdicts) == 19

        valid_verdicts = {"CONFIRMED", "REFUTED", "INCONCLUSIVE"}
        for v in verdicts:
            assert "hypothesis_id" in v
            assert "description" in v
            assert "configs_tested" in v
            assert "verdict" in v
            assert v["verdict"] in valid_verdicts

        # H1: DD reduces raw error (C0=0.15 vs C1=0.10) → CONFIRMED
        h1 = next(v for v in verdicts if v["hypothesis_id"] == "H1")
        assert h1["verdict"] == "CONFIRMED"

        # H4: PEA superior to GF (C3=0.05 vs C5=0.025) → CONFIRMED
        h4 = next(v for v in verdicts if v["hypothesis_id"] == "H4")
        assert h4["verdict"] == "CONFIRMED"

    def test_analyzer_derived_metrics(self, results_dir):
        """compute_derived_metrics computes improvement, overhead, etc."""
        from project_health.analysis.mitigation_benchmark_analyzer import (
            MitigationBenchmarkAnalyzer,
        )

        analyzer = MitigationBenchmarkAnalyzer(results_dir=results_dir)
        analyzer.scan()
        metrics = analyzer.compute_derived_metrics()

        # Should have metrics for configs with data
        assert len(metrics) > 0
        # Check structure of a known config
        if "C5_full_pea_balanced" in metrics:
            m = metrics["C5_full_pea_balanced"]
            assert "mean_delta_e_gap" in m
            # C5 has entries in both modes: 0.025 (sim, 4 entries) + 0.035 (hw, 4 entries)
            # Mean = (0.025*4 + 0.035*4) / 8 = 0.030
            assert m["mean_delta_e_gap"] == pytest.approx(0.030, abs=0.001)
            assert "improvement_vs_raw" in m
            assert "overhead_factor" in m
            assert "precision_per_shot" in m
            assert "net_benefit" in m
