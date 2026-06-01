"""Comprehensive tests for src/qmbp_simulation/execution/hardware/ module.

Tests config, phase classification, observables, SPSA, preflight,
submission, persistence, and full integration via HardwareBackend.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qmbp_simulation.execution.hardware.config import (
    HardwareConfig,
    HardwareRunResult,
    SPSAConfig,
)
from qmbp_simulation.execution.hardware.observables import (
    build_per_site_observables,
    extract_array_result,
)
from qmbp_simulation.execution.hardware.phase import classify_phase
from qmbp_simulation.execution.hardware.spsa import spsa_refinement
from qmbp_simulation.framework.logging import StructuredLogger

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def fake_torino():
    """Module-scoped FakeTorino (slow init ~3s, reuse across tests)."""
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    return FakeTorino()


@pytest.fixture
def tmp_output(tmp_path):
    """Temporary output directory for persistence tests."""
    return tmp_path / "hw_output"


@pytest.fixture
def logger():
    """Fresh StructuredLogger for each test."""
    return StructuredLogger("test_hardware")


@pytest.fixture
def hw_config(tmp_output):
    """HardwareConfig in fake_backend mode with small params."""
    return HardwareConfig(
        mode="fake_backend",
        n_qubits=6,
        shots=256,
        n_layouts=3,
        n_candidates=20,
        max_ces=1.0,
        optimization_level=1,
        layout_seed=42,
        output_dir=str(tmp_output),
        spsa_enabled=True,
        spsa_threshold=0.05,
        max_total_shots=100_000,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. Config Module
# ═══════════════════════════════════════════════════════════════════════════


class TestConfig:
    def test_defaults(self):
        cfg = HardwareConfig()
        assert cfg.backend_name == "ibm_torino"
        assert cfg.mode == "hardware"
        assert cfg.n_qubits == 10
        assert cfg.shots == 16384
        assert cfg.n_layouts == 3

    def test_mitigation_integration(self):
        cfg = HardwareConfig()
        assert cfg.mitigation.dd_enabled is True
        assert cfg.mitigation.trex_enabled is True
        assert cfg.mitigation.twirling_enabled is True
        assert cfg.mitigation.zne_enabled is True

    def test_spsa_config_defaults(self):
        spsa = SPSAConfig()
        assert spsa.a == 0.1
        assert spsa.c == 0.05
        assert spsa.n_iterations == 200
        assert spsa.alpha == 0.602
        assert spsa.gamma == 0.101

    def test_run_result_fields(self):
        r = HardwareRunResult(
            h_value=3.0,
            e_exact=-5.0,
            e_zne=-4.9,
            delta_e_gap=0.02,
            gap=5.0,
            phase_label="paramagnetic",
            expected_label="paramagnetic",
            zne_r2=0.99,
            zne_gain=0.5,
            mag_x_mean=0.8,
            corr_zz_mean=0.2,
            sigma=0.01,
            total_shots=50000,
        )
        assert r.verdict == ""
        assert r.is_partial is False
        assert r.spsa_applied is False


# ═══════════════════════════════════════════════════════════════════════════
# 2. Phase Classification
# ═══════════════════════════════════════════════════════════════════════════


class TestPhaseClassification:
    def test_paramagnetic(self):
        x_vals = [0.9, 0.85, 0.88, 0.92]
        zz_vals = [0.1, 0.12, 0.08]
        label, mag_x, corr_zz, sigma = classify_phase(x_vals, zz_vals, 10000)
        assert label == "paramagnetic"
        assert mag_x > corr_zz

    def test_ordered(self):
        x_vals = [0.1, 0.12, 0.08, 0.09]
        zz_vals = [0.9, 0.85, 0.88]
        label, mag_x, corr_zz, sigma = classify_phase(x_vals, zz_vals, 10000)
        assert label == "ordered"
        assert corr_zz > mag_x

    def test_indeterminate(self):
        x_vals = [0.5, 0.5, 0.5, 0.5]
        zz_vals = [0.5, 0.5, 0.5]
        label, _, _, sigma = classify_phase(x_vals, zz_vals, 10000)
        assert label == "indeterminate"

    def test_single_element(self):
        label, mag_x, corr_zz, sigma = classify_phase([0.9], [0.1], 1024)
        assert label == "paramagnetic"
        assert mag_x == pytest.approx(0.9)
        assert corr_zz == pytest.approx(0.1)

    def test_empty_lists(self):
        """Edge case: empty lists should return indeterminate with zeros."""
        label, mag_x, corr_zz, sigma = classify_phase([], [], 1024)
        assert label == "indeterminate"
        assert mag_x == 0.0
        assert corr_zz == 0.0

    def test_sigma_decreases_with_shots(self):
        _, _, _, sigma_low = classify_phase([0.5], [0.3], 100)
        _, _, _, sigma_high = classify_phase([0.5], [0.3], 100000)
        assert sigma_high < sigma_low


# ═══════════════════════════════════════════════════════════════════════════
# 3. Observables
# ═══════════════════════════════════════════════════════════════════════════


class TestObservables:
    def test_build_per_site_count(self):
        n = 6
        edges = [(i, i + 1) for i in range(n - 1)]
        x_ops, zz_ops = build_per_site_observables(n, edges)
        assert len(x_ops) == n
        assert len(zz_ops) == n - 1

    def test_build_per_site_num_qubits(self):
        n = 4
        edges = [(0, 1), (1, 2), (2, 3)]
        x_ops, zz_ops = build_per_site_observables(n, edges)
        assert x_ops[0].num_qubits == n
        assert zz_ops[0].num_qubits == n

    def test_extract_array_result(self):
        """Test extraction from mock EstimatorV2 result."""
        mock_data = MagicMock()
        mock_data.evs = np.array([0.1, 0.2, 0.3, 0.4, -0.5, -0.6, -0.7])
        mock_pub = MagicMock()
        mock_pub.data = mock_data
        mock_result = [mock_pub]

        x_vals, zz_vals = extract_array_result(mock_result, n_x=4, n_zz=3)
        assert len(x_vals) == 4
        assert len(zz_vals) == 3
        assert x_vals == pytest.approx([0.1, 0.2, 0.3, 0.4])
        assert zz_vals == pytest.approx([-0.5, -0.6, -0.7])

    def test_extract_array_result_scalar_evs(self):
        """Edge case: what if evs is a scalar (single observable)?"""
        mock_data = MagicMock()
        mock_data.evs = np.array([0.5])
        mock_pub = MagicMock()
        mock_pub.data = mock_data
        mock_result = [mock_pub]

        x_vals, zz_vals = extract_array_result(mock_result, n_x=1, n_zz=0)
        assert x_vals == [pytest.approx(0.5)]
        assert zz_vals == []


# ═══════════════════════════════════════════════════════════════════════════
# 4. SPSA Refinement
# ═══════════════════════════════════════════════════════════════════════════


class TestSPSA:
    def test_skip_below_threshold(self, hw_config, logger):
        """SPSA should skip when delta_e_gap <= threshold."""
        rng = np.random.default_rng(42)
        params = np.array([0.1, 0.2, 0.3])
        e_exact = -5.0
        initial_energy = -4.998  # delta_e_gap = 0.002/1.0 = 0.002 < 0.05
        gap = 1.0
        spsa_cfg = SPSAConfig(n_iterations=10)

        best_p, best_e, applied = spsa_refinement(
            lambda p: -4.9,
            params,
            initial_energy,
            e_exact,
            gap,
            hw_config,
            spsa_cfg,
            logger,
            rng,
            current_total_shots=0,
        )
        assert applied is False
        assert np.array_equal(best_p, params)
        assert best_e == initial_energy

    def test_activates_above_threshold(self, hw_config, logger):
        """SPSA should activate when delta_e_gap > threshold."""
        rng = np.random.default_rng(42)
        params = np.array([0.1, 0.2, 0.3])
        e_exact = -5.0
        initial_energy = -4.0  # delta_e_gap = 1.0/1.0 = 1.0 > 0.05
        gap = 1.0
        spsa_cfg = SPSAConfig(n_iterations=5)

        call_count = [0]

        def eval_fn(p):
            call_count[0] += 1
            return -4.5 + 0.01 * call_count[0]

        best_p, best_e, applied = spsa_refinement(
            eval_fn,
            params,
            initial_energy,
            e_exact,
            gap,
            hw_config,
            spsa_cfg,
            logger,
            rng,
            current_total_shots=0,
        )
        assert applied is True
        assert call_count[0] > 0

    def test_cost_ceiling_abort(self, hw_config, logger):
        """SPSA should abort when cost ceiling is reached."""
        hw_config.max_total_shots = 1000  # Very low ceiling
        hw_config.shots = 256
        hw_config.n_layouts = 3
        rng = np.random.default_rng(42)
        params = np.array([0.1, 0.2])
        spsa_cfg = SPSAConfig(n_iterations=100)

        call_count = [0]

        def eval_fn(p):
            call_count[0] += 1
            return -4.5

        best_p, best_e, applied = spsa_refinement(
            eval_fn,
            params,
            -4.0,
            -5.0,
            1.0,
            hw_config,
            spsa_cfg,
            logger,
            rng,
            current_total_shots=500,
        )
        # Should abort early due to cost ceiling
        assert applied is True
        assert call_count[0] < 200  # Much less than 2*100 iterations

    def test_never_worsens(self, hw_config, logger):
        """SPSA should never return worse result than initial."""
        rng = np.random.default_rng(42)
        params = np.array([0.5, 0.5])
        e_exact = -5.0
        initial_energy = -4.5  # delta_e_gap = 0.5/1.0 = 0.5 > 0.05
        gap = 1.0
        spsa_cfg = SPSAConfig(n_iterations=5)

        # eval_fn always returns worse energy
        def eval_fn(p):
            return -3.0  # worse than initial

        best_p, best_e, applied = spsa_refinement(
            eval_fn,
            params,
            initial_energy,
            e_exact,
            gap,
            hw_config,
            spsa_cfg,
            logger,
            rng,
            current_total_shots=0,
        )
        # Should return initial since refined is worse
        assert abs(best_e - e_exact) <= abs(initial_energy - e_exact)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Preflight
# ═══════════════════════════════════════════════════════════════════════════


class TestPreflight:
    def test_topology_pass_n6(self, fake_torino, logger):
        from qmbp_simulation.execution.hardware.preflight import run_preflight_checks

        cfg = HardwareConfig(mode="fake_backend", n_qubits=6)
        result = run_preflight_checks(fake_torino, cfg, logger)
        assert result["abort"] is False
        assert result["topology_sufficient"] is True

    def test_topology_pass_n10(self, fake_torino, logger):
        from qmbp_simulation.execution.hardware.preflight import run_preflight_checks

        cfg = HardwareConfig(mode="fake_backend", n_qubits=10)
        result = run_preflight_checks(fake_torino, cfg, logger)
        assert result["abort"] is False
        assert result["topology_sufficient"] is True

    def test_topology_fail_huge(self, fake_torino, logger):
        from qmbp_simulation.execution.hardware.preflight import run_preflight_checks

        cfg = HardwareConfig(mode="fake_backend", n_qubits=500)
        result = run_preflight_checks(fake_torino, cfg, logger)
        # FakeTorino has 133 qubits, so 500 should fail
        assert result["abort"] is True

    def test_mean_2q_error(self, fake_torino):
        from qmbp_simulation.execution.hardware.preflight import compute_mean_2q_error

        err = compute_mean_2q_error(fake_torino)
        assert err is not None
        assert 0.0 < err < 0.1  # Reasonable range for fake backend


# ═══════════════════════════════════════════════════════════════════════════
# 6. Submission — select_layouts_for_hardware
# ═══════════════════════════════════════════════════════════════════════════


class TestSubmission:
    def test_select_layouts(self, fake_torino, hw_config, logger):
        from qmbp_simulation.execution.hardware.submission import select_layouts_for_hardware

        qc = QuantumCircuit(6)
        qc.h(range(6))
        for i in range(5):
            qc.cx(i, i + 1)

        hw_config.n_qubits = 6
        selection = select_layouts_for_hardware(qc, fake_torino, hw_config, logger)
        assert len(selection.layouts) >= 1
        assert len(selection.ces_values) == len(selection.layouts)
        assert len(selection.transpiled_circuits) == len(selection.layouts)
        # CES values should be positive
        assert all(c > 0 for c in selection.ces_values)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Persistence
# ═══════════════════════════════════════════════════════════════════════════


class TestPersistence:
    def test_save_run_creates_files(self, hw_config, logger):
        from qmbp_simulation.execution.hardware.persistence import save_run

        result = HardwareRunResult(
            h_value=3.0,
            e_exact=-5.0,
            e_zne=-4.95,
            delta_e_gap=0.01,
            gap=5.0,
            phase_label="paramagnetic",
            expected_label="paramagnetic",
            zne_r2=0.99,
            zne_gain=0.48,
            mag_x_mean=0.8,
            corr_zz_mean=0.2,
            sigma=0.01,
            total_shots=50000,
            verdict="PASS",
        )
        run_dir = save_run(
            result,
            hw_config,
            logger,
            calibration_info={"mean_2q_error": 0.005},
            options_dict={"default_shots": 16384},
            execution_mode_name="Batch",
            raw_per_layout=[{"layout_idx": 0, "energy": -4.9}],
            zne_data={"extrapolated_energy": -4.95, "r_squared": 0.99},
        )
        assert run_dir.exists()
        assert (run_dir / "config.json").exists()
        assert (run_dir / "provenance.json").exists()
        assert (run_dir / "raw_results.json").exists()
        assert (run_dir / "zne_analysis.json").exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "execution_log.json").exists()

        # Verify JSON is valid
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)
        assert summary["h_test"] == 3.0
        assert summary["verdict"] == "PASS"

    def test_save_partial_before_error(self, hw_config, logger):
        from qmbp_simulation.execution.hardware.persistence import save_partial_before_error

        partial = [{"step": "energy", "e_zne": -4.5}]
        run_dir = save_partial_before_error(partial, logger, hw_config, "Test error")
        assert run_dir.exists()
        assert "PARTIAL" in run_dir.name
        assert (run_dir / "partial_results.json").exists()
        assert (run_dir / "config.json").exists()
        assert (run_dir / "execution_log.json").exists()

        with open(run_dir / "partial_results.json") as f:
            data = json.load(f)
        assert data["error"] == "Test error"


# ═══════════════════════════════════════════════════════════════════════════
# 8. Full Integration — HardwareBackend in fake_backend mode
# ═══════════════════════════════════════════════════════════════════════════


class TestHardwareBackendIntegration:
    def test_evaluate_returns_finite(self, tmp_path):
        """HardwareBackend.evaluate() should return a finite float."""
        from qmbp_simulation.execution.hardware import HardwareBackend

        config = HardwareConfig(
            mode="fake_backend",
            n_qubits=4,
            shots=256,
            n_layouts=2,
            n_candidates=10,
            optimization_level=1,
            layout_seed=42,
            output_dir=str(tmp_path),
            spsa_enabled=False,
        )
        backend = HardwareBackend(config=config)

        # Simple 4-qubit parameterized circuit
        qc = QuantumCircuit(4)
        qc.h(range(4))
        from qiskit.circuit import Parameter

        theta = [Parameter(f"t{i}") for i in range(4)]
        for i in range(3):
            qc.rzz(theta[i], i, i + 1)
        qc.rx(theta[3], 0)

        # Simple Hamiltonian
        H = SparsePauliOp.from_list([("ZZZZ", 1.0), ("XXXX", 0.5)])
        params = np.array([0.1, 0.2, 0.3, 0.4])

        energy = backend.evaluate(qc, H, params)
        assert np.isfinite(energy)
        assert isinstance(energy, float)
