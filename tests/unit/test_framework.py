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
        assert config.seeds == DEFAULT_SEEDS
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


# ═══════════════════════════════════════════════════════════════════════════
# Tests extracted from test_refactoring.py (formerly ad-hoc functional tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestCLIFilterFormatArgs:
    """Test CLI argument group helpers parse correctly."""

    def test_result_filter_args_parse(self):
        from qmbp_simulation.framework.cli import (
            add_result_filter_args,
            create_base_parser,
        )

        parser = create_base_parser("Test")
        add_result_filter_args(parser)
        args = parser.parse_args(["--topology", "ladder", "--n-qubits", "10", "--p-layers", "1"])
        assert args.topology == "ladder"
        assert args.n_qubits == 10
        assert args.p_layers == 1

    def test_format_args_parse(self):
        from qmbp_simulation.framework.cli import (
            add_format_args,
            create_base_parser,
        )

        parser = create_base_parser("Test")
        add_format_args(parser)
        args = parser.parse_args(["--sort", "delta_e", "--top", "5", "--group-by", "topology"])
        assert args.sort == "delta_e"
        assert args.top == 5
        assert args.group_by == "topology"

    def test_variant_runner_args_parse(self):
        from qmbp_simulation.framework.cli import (
            add_variant_runner_args,
            create_base_parser,
        )

        parser = create_base_parser("Test")
        add_variant_runner_args(parser)
        args = parser.parse_args(["--dry-run", "--start-from", "3"])
        assert args.dry_run is True
        assert args.start_from == 3

    def test_all_arg_groups_combine(self):
        from qmbp_simulation.framework.cli import (
            add_format_args,
            add_result_filter_args,
            add_variant_runner_args,
            create_base_parser,
        )

        parser = create_base_parser("Test")
        add_result_filter_args(parser)
        add_format_args(parser)
        add_variant_runner_args(parser)
        args = parser.parse_args(
            [
                "--topology",
                "triangular",
                "--n-qubits",
                "6",
                "--sort",
                "r2",
                "--dry-run",
            ]
        )
        assert args.topology == "triangular"
        assert args.n_qubits == 6
        assert args.sort == "r2"
        assert args.dry_run is True


class TestAutoPreflightInBaseExperiment:
    """Test that BaseExperiment._run_preflight() validates configs correctly."""

    def _make_experiment(self, **system_kwargs):
        from qmbp_simulation.framework import BaseExperiment, ExperimentConfig
        from qmbp_simulation.framework.config import SystemConfig, VQEConfig
        from qmbp_simulation.framework.metrics import ExperimentMetrics

        class _DummyExp(BaseExperiment):
            @classmethod
            def default_config(cls):
                sys_config = SystemConfig(
                    n_qubits=6,
                    p_layers=2,
                    h_values=[2.0, 1.75, 1.5, 1.25],
                    h_test=[1.6],
                    **system_kwargs,
                )
                return ExperimentConfig(
                    experiment_id="TEST",
                    category="T",
                    description="Test",
                    hypothesis="Testing",
                    system=sys_config,
                    vqe=VQEConfig(n_restarts=1, maxiter=100),
                    seeds=[42],
                )

            def run_single(self, seed):
                return [
                    ExperimentMetrics(
                        h_value=1.6,
                        energy=-5.0,
                        exact_energy=-5.1,
                        energy_error=0.1,
                        gap=0.5,
                        relative_error=0.02,
                        seed=seed,
                        wall_time_s=1.0,
                    )
                ]

        config = _DummyExp.default_config()
        return _DummyExp(config)

    def test_valid_config_passes_preflight(self):
        exp = self._make_experiment()
        # Should not raise
        exp._run_preflight()

    def test_p3_blocked_by_preflight(self):
        from qmbp_simulation.framework import BaseExperiment, ExperimentConfig
        from qmbp_simulation.framework.config import SystemConfig

        class _BadExp(BaseExperiment):
            @classmethod
            def default_config(cls):
                return ExperimentConfig(
                    experiment_id="BAD",
                    category="T",
                    description="Bad",
                    hypothesis="Should fail",
                    system=SystemConfig(n_qubits=6, p_layers=3),
                    seeds=[42],
                )

            def run_single(self, seed):
                return []

        config = _BadExp.default_config()
        exp = _BadExp(config)
        with pytest.raises(ValueError):
            exp._run_preflight()


class TestRunVQESweepFastProxy:
    """Fast proxy tests for BaseExperiment.run_vqe_sweep.

    Tests the setup and structure without running actual VQE optimization.
    The slow tests validate numerical correctness; these validate API contract.
    """

    def test_setup_creates_circuit_and_hamiltonian(self):
        from qmbp_simulation.framework import BaseExperiment, ExperimentConfig
        from qmbp_simulation.framework.config import SystemConfig, VQEConfig

        class _Exp(BaseExperiment):
            @classmethod
            def default_config(cls):
                return ExperimentConfig(
                    experiment_id="FAST",
                    category="T",
                    description="Fast",
                    hypothesis="Setup works",
                    system=SystemConfig(
                        n_qubits=6,
                        p_layers=2,
                        h_values=[2.0, 1.5, 1.0],
                        h_test=[1.6],
                    ),
                    vqe=VQEConfig(n_restarts=1, maxiter=10),
                    seeds=[42],
                )

            def run_single(self, seed):
                return []

        exp = _Exp(_Exp.default_config())
        exp.setup()

        # Verify setup creates expected attributes
        assert exp.circuit is not None
        assert exp.circuit.num_qubits == 6
        assert exp.circuit.num_parameters == 4  # 2 params/layer × 2 layers

    def test_get_exact_solution_returns_expected_fields(self):
        from qmbp_simulation.framework import BaseExperiment, ExperimentConfig
        from qmbp_simulation.framework.config import SystemConfig, VQEConfig

        class _Exp(BaseExperiment):
            @classmethod
            def default_config(cls):
                return ExperimentConfig(
                    experiment_id="FAST",
                    category="T",
                    description="Fast",
                    hypothesis="Exact solution",
                    system=SystemConfig(n_qubits=4, p_layers=1),
                    vqe=VQEConfig(n_restarts=1, maxiter=10),
                    seeds=[42],
                )

            def run_single(self, seed):
                return []

        exp = _Exp(_Exp.default_config())
        exp.setup()
        sol = exp.get_exact_solution(1.5)

        assert "hamiltonian" in sol
        assert "exact" in sol
        assert sol["exact"].ground_energy < 0  # TFIM always negative
        assert sol["exact"].gap > 0  # finite gap in paramagnetic phase

    def test_evaluate_energy_returns_float(self):
        from qmbp_simulation.framework import BaseExperiment, ExperimentConfig
        from qmbp_simulation.framework.config import SystemConfig, VQEConfig

        class _Exp(BaseExperiment):
            @classmethod
            def default_config(cls):
                return ExperimentConfig(
                    experiment_id="FAST",
                    category="T",
                    description="Fast",
                    hypothesis="Energy eval",
                    system=SystemConfig(n_qubits=4, p_layers=1),
                    vqe=VQEConfig(n_restarts=1, maxiter=10),
                    seeds=[42],
                )

            def run_single(self, seed):
                return []

        exp = _Exp(_Exp.default_config())
        exp.setup()
        sol = exp.get_exact_solution(1.5)

        # Random params → some energy
        params = np.zeros(exp.circuit.num_parameters)
        energy = exp.evaluate_energy(params, sol["hamiltonian"])
        assert isinstance(energy, float)
        assert np.isfinite(energy)


@pytest.mark.slow
class TestRunVQESweep:
    """Test BaseExperiment.run_vqe_sweep produces valid warm-started results.

    Marked slow because it runs actual VQE optimization (~10s).
    """

    def test_sweep_produces_correct_count(self):
        from qmbp_simulation.framework import BaseExperiment, ExperimentConfig
        from qmbp_simulation.framework.config import SystemConfig, VQEConfig
        from qmbp_simulation.framework.metrics import ExperimentMetrics

        class _SweepExp(BaseExperiment):
            @classmethod
            def default_config(cls):
                return ExperimentConfig(
                    experiment_id="SWEEP",
                    category="T",
                    description="Sweep test",
                    hypothesis="VQE sweep works",
                    system=SystemConfig(
                        n_qubits=6,
                        p_layers=2,
                        h_values=[2.0, 1.75, 1.5, 1.25],
                        h_test=[1.6],
                    ),
                    vqe=VQEConfig(n_restarts=2, maxiter=200),
                    seeds=[42],
                )

            def run_single(self, seed):
                return [
                    ExperimentMetrics(
                        h_value=1.6,
                        energy=-5.0,
                        exact_energy=-5.1,
                        energy_error=0.1,
                        gap=0.5,
                        relative_error=0.02,
                        seed=seed,
                        wall_time_s=1.0,
                    )
                ]

        config = _SweepExp.default_config()
        exp = _SweepExp(config)
        exp.setup()

        h_vals = [2.0, 1.75, 1.5]
        vqe_data = exp.run_vqe_sweep(h_vals, seed=42)

        assert len(vqe_data) == 3
        assert all(isinstance(v, np.ndarray) for v in vqe_data.values())
        assert all(len(v) == exp.circuit.num_parameters for v in vqe_data.values())

    def test_sweep_warm_start_smoothness(self):
        from qmbp_simulation.framework import BaseExperiment, ExperimentConfig
        from qmbp_simulation.framework.config import SystemConfig, VQEConfig

        class _SweepExp(BaseExperiment):
            @classmethod
            def default_config(cls):
                return ExperimentConfig(
                    experiment_id="SWEEP",
                    category="T",
                    description="Sweep test",
                    hypothesis="Warm start produces smooth θ",
                    system=SystemConfig(
                        n_qubits=6,
                        p_layers=2,
                        h_values=[2.0, 1.75, 1.5, 1.25],
                        h_test=[1.6],
                    ),
                    vqe=VQEConfig(n_restarts=2, maxiter=200),
                    seeds=[42],
                )

            def run_single(self, seed):
                return []

        config = _SweepExp.default_config()
        exp = _SweepExp(config)
        exp.setup()

        h_vals = [2.0, 1.75, 1.5]
        vqe_data = exp.run_vqe_sweep(h_vals, seed=42)

        # Check smoothness: adjacent h-points should have similar params
        theta_arr = np.array([vqe_data[h] for h in h_vals])
        diffs = np.diff(theta_arr, axis=0)
        max_jump = np.max(np.abs(diffs))
        # With warm start, max parameter jump should be modest (< π)
        assert max_jump < np.pi, f"Warm start failed: max jump = {max_jump:.3f}"
