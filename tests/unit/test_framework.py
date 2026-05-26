"""Unit tests for qmbp_simulation.framework module."""

from __future__ import annotations

import numpy as np
import pytest

from qmbp_simulation.framework import ExperimentConfig, ExperimentMetrics


class TestExperimentConfigJsonRoundTrip:
    """Test ExperimentConfig JSON round-trip."""

    def test_to_json_from_json_preserves_fields(self, tmp_path):
        config = ExperimentConfig(
            experiment_id="test_A1",
            category="A",
            description="Test experiment",
            hypothesis="Energy scales linearly",
            seeds=[42, 43],
            verbose=True,
        )
        config.system.n_qubits = 10
        config.system.p_layers = 2
        config.vqe.n_restarts = 3

        path = tmp_path / "config.json"
        config.to_json(path)
        loaded = ExperimentConfig.from_json(path)

        assert loaded.experiment_id == "test_A1"
        assert loaded.category == "A"
        assert loaded.description == "Test experiment"
        assert loaded.hypothesis == "Energy scales linearly"
        assert loaded.seeds == [42, 43]
        assert loaded.verbose is True
        assert loaded.system.n_qubits == 10
        assert loaded.system.p_layers == 2
        assert loaded.vqe.n_restarts == 3


class TestExperimentMetricsValidate:
    """Test ExperimentMetrics.validate() catches invalid values."""

    def test_valid_metrics_no_issues(self):
        m = ExperimentMetrics(
            h_value=1.5,
            energy=-4.5,
            exact_energy=-4.6,
            energy_error=0.1,
            gap=0.5,
            relative_error=0.02,
            fidelity=0.98,
        )
        issues = m.validate()
        assert issues == []

    def test_negative_relative_error_detected(self):
        m = ExperimentMetrics(
            h_value=1.5,
            energy=-4.5,
            exact_energy=-4.6,
            energy_error=0.1,
            gap=0.5,
            relative_error=-0.1,
        )
        issues = m.validate()
        assert any("Negative" in i and "ΔE/gap" in i for i in issues)

    def test_invalid_fidelity_detected(self):
        m = ExperimentMetrics(
            h_value=1.5,
            energy=-4.5,
            exact_energy=-4.6,
            energy_error=0.1,
            gap=0.5,
            relative_error=0.02,
            fidelity=1.5,
        )
        issues = m.validate()
        assert any("fidelity" in i.lower() for i in issues)

    def test_nonpositive_gap_detected(self):
        m = ExperimentMetrics(
            h_value=1.5,
            energy=-4.5,
            exact_energy=-4.6,
            energy_error=0.1,
            gap=-0.1,
            relative_error=0.02,
        )
        issues = m.validate()
        assert any("gap" in i.lower() for i in issues)


class TestExperimentConfigValidate:
    """Test ExperimentConfig.validate() rejects p_layers > 2."""

    def test_p_layers_3_raises(self):
        config = ExperimentConfig()
        config.system.p_layers = 3
        with pytest.raises(ValueError, match="p_layers > 2"):
            config.validate()

    def test_valid_config_returns_empty_warnings(self):
        config = ExperimentConfig()
        config.system.p_layers = 2
        warnings = config.validate()
        assert isinstance(warnings, list)

    def test_config_default_values(self):
        config = ExperimentConfig()
        assert config.experiment_id == "unnamed"
        assert config.seeds == [42, 43, 44]
        assert config.verbose is False
        assert config.system.p_layers == 2
        assert config.system.n_qubits == 6


class TestStructuredLogger:
    """Test StructuredLogger basic functionality."""

    def test_logger_creates_and_logs(self, tmp_path):
        from qmbp_simulation.framework import StructuredLogger

        logger = StructuredLogger(experiment_id="test")
        logger.log("test_event", data={"key": "value"})
        path = tmp_path / "log.json"
        logger.save(path)
        assert path.exists()

    def test_logger_timer_records_elapsed(self, tmp_path):
        import time

        from qmbp_simulation.framework import StructuredLogger

        logger = StructuredLogger(experiment_id="test")
        logger.start_timer("phase1")
        time.sleep(0.01)
        elapsed = logger.stop_timer("phase1")
        assert elapsed >= 0.01


class TestWarmColdComparison:
    """Test WarmColdComparison dataclass."""

    def test_warm_cold_comparison_compute(self):
        from qmbp_simulation.framework import WarmColdComparison

        comp = WarmColdComparison.compute(
            h_value=1.5,
            seed=42,
            warm_init=np.array([0.1, 0.2]),
            warm_energy=-4.5,
            warm_de_gap=0.02,
            warm_nit=50,
            cold_init=np.array([0.0, 0.0]),
            cold_energy=-4.0,
            cold_de_gap=0.10,
            cold_nit=100,
        )
        assert comp.h_value == 1.5
        assert comp.warm_final_energy == -4.5
        assert comp.cold_final_energy == -4.0
        assert comp.gain_pct > 0  # warm is better
        assert comp.iteration_savings_pct == 50.0
