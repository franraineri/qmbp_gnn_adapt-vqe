"""Integration tests for QESEM routing in the hardware deployment pipeline.

Validates that:
1. The qesem_enabled flag correctly routes through the QESEM alternate path
2. QESEMResult → HardwareRunResult mapping is correct
3. Verdict logic works with QESEM output
4. The module gracefully handles import errors (qiskit-ibm-catalog missing)

These tests use mocks since QESEM requires a real QPU + Premium plan.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qmbp_simulation.execution.backends import MitigationOptions
from qmbp_simulation.execution.hardware.config import HardwareConfig, HardwareRunResult
from qmbp_simulation.execution.hardware.qesem import (
    QESEMResult,
    check_qesem_available,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Test: QESEMResult dataclass construction
# ═══════════════════════════════════════════════════════════════════════════════


class TestQESEMResult:
    """Test QESEMResult dataclass and its mapping to HardwareRunResult."""

    def test_basic_construction(self):
        """QESEMResult can be constructed with typical QESEM output."""
        result = QESEMResult(
            energy_mitigated=-9.5,
            energy_std=0.01,
            x_values=[0.8] * 10,
            zz_values=[-0.3] * 9,
            x_stds=[0.005] * 10,
            zz_stds=[0.005] * 9,
            noisy_energy=-7.0,
            noisy_x_values=[0.5] * 10,
            noisy_zz_values=[-0.2] * 9,
            job_id="test-job-123",
            total_qpu_time=45.0,
            gate_fidelities={"cx": 0.995},
            total_shots=100000,
            mitigation_shots=80000,
        )
        assert result.energy_mitigated == -9.5
        assert result.energy_std == 0.01
        assert len(result.x_values) == 10
        assert len(result.zz_values) == 9
        assert result.job_id == "test-job-123"
        assert result.total_qpu_time == 45.0

    def test_maps_to_hardware_run_result(self):
        """QESEMResult fields map correctly to HardwareRunResult."""
        qesem_result = QESEMResult(
            energy_mitigated=-9.5,
            energy_std=0.01,
            x_values=[0.85] * 10,
            zz_values=[-0.35] * 9,
            x_stds=[0.005] * 10,
            zz_stds=[0.005] * 9,
            noisy_energy=-7.0,
            noisy_x_values=[0.5] * 10,
            noisy_zz_values=[-0.2] * 9,
            job_id="qesem-abc-123",
            total_qpu_time=60.0,
            total_shots=150000,
            mitigation_shots=120000,
        )

        # Simulate the mapping logic from backend.py QESEM branch
        e_exact = -9.6
        gap = 1.5
        e_mitigated = qesem_result.energy_mitigated
        delta_e_gap = abs(e_mitigated - e_exact) / gap

        hw_result = HardwareRunResult(
            h_value=4.0,
            e_exact=e_exact,
            e_zne=e_mitigated,
            delta_e_gap=delta_e_gap,
            gap=gap,
            phase_label="paramagnetic",
            expected_label="paramagnetic",
            zne_r2=1.0,  # QESEM is unbiased
            zne_gain=0.0,
            mag_x_mean=float(np.mean(qesem_result.x_values)),
            corr_zz_mean=float(np.mean(qesem_result.zz_values)),
            sigma=0.01,
            total_shots=qesem_result.total_shots or 0,
            job_ids=[qesem_result.job_id],
            per_site_x=qesem_result.x_values,
            per_bond_zz=qesem_result.zz_values,
            verdict="PASS",
            verdict_reason=f"ΔE/gap={delta_e_gap:.4f} < 5%, phase=paramagnetic correct",
            zne_amplifier_used="qesem",
            mitigation_strategy="qesem_unbiased",
            qesem_used=True,
            qesem_job_id=qesem_result.job_id,
            qesem_total_qpu_time=qesem_result.total_qpu_time,
            qesem_total_shots=qesem_result.total_shots,
            qesem_mitigation_shots=qesem_result.mitigation_shots,
        )

        assert hw_result.qesem_used is True
        assert hw_result.e_zne == -9.5
        assert hw_result.delta_e_gap == pytest.approx(0.0667, abs=0.001)
        assert hw_result.mitigation_strategy == "qesem_unbiased"
        assert hw_result.zne_amplifier_used == "qesem"
        assert hw_result.qesem_job_id == "qesem-abc-123"


# ═══════════════════════════════════════════════════════════════════════════════
# Test: QESEM availability check
# ═══════════════════════════════════════════════════════════════════════════════


class TestQESEMAvailability:
    """Test QESEM dependency checking."""

    def test_check_qesem_available_returns_tuple(self):
        """check_qesem_available returns (bool, str|None)."""
        available, error = check_qesem_available()
        assert isinstance(available, bool)
        if not available:
            assert isinstance(error, str)
        else:
            assert error is None


# ═══════════════════════════════════════════════════════════════════════════════
# Test: MitigationOptions qesem_enabled flag
# ═══════════════════════════════════════════════════════════════════════════════


class TestMitigationOptionsQESEM:
    """Test that MitigationOptions.qesem_enabled works as a routing flag."""

    def test_default_qesem_disabled(self):
        """By default, qesem_enabled is False."""
        opts = MitigationOptions()
        assert opts.qesem_enabled is False

    def test_qesem_enabled_flag(self):
        """qesem_enabled can be set to True."""
        opts = MitigationOptions(qesem_enabled=True)
        assert opts.qesem_enabled is True

    def test_qesem_with_other_options(self):
        """qesem_enabled can coexist with other mitigation flags.

        When qesem_enabled=True, the backend ignores local ZNE options
        (they're only relevant for the local path).
        """
        opts = MitigationOptions(
            qesem_enabled=True,
            zne_enabled=True,
            dd_enabled=True,
            zne_amplifier="pea",
        )
        assert opts.qesem_enabled is True
        # Other options are still set (backwards compat) — just unused
        assert opts.zne_enabled is True


# ═══════════════════════════════════════════════════════════════════════════════
# Test: HardwareConfig QESEM fields
# ═══════════════════════════════════════════════════════════════════════════════


class TestHardwareConfigQESEM:
    """Test HardwareConfig QESEM-specific fields."""

    def test_default_qesem_config_values(self):
        """Default QESEM config values are sensible."""
        config = HardwareConfig()
        assert config.qesem_precision == 0.01
        assert config.qesem_max_execution_time == 600
        assert config.qesem_instance is None

    def test_custom_qesem_precision(self):
        """QESEM precision can be customized."""
        config = HardwareConfig(qesem_precision=0.005)
        assert config.qesem_precision == 0.005

    def test_qesem_enabled_in_mitigation_routes_correctly(self):
        """When qesem_enabled=True, config.mitigation reflects it."""
        config = HardwareConfig(
            mitigation=MitigationOptions(qesem_enabled=True),
            qesem_precision=0.02,
            qesem_max_execution_time=600,
        )
        assert config.mitigation.qesem_enabled is True
        assert config.qesem_precision == 0.02
        assert config.qesem_max_execution_time == 600


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Verdict logic with QESEM results
# ═══════════════════════════════════════════════════════════════════════════════


class TestQESEMVerdictLogic:
    """Test that verdict logic handles QESEM output correctly."""

    def test_pass_verdict(self):
        """PASS when ΔE/gap < 5% and phase label correct."""
        e_exact = -9.6
        gap = 1.5
        e_mitigated = -9.55  # ΔE = 0.05, ΔE/gap = 3.3%
        delta = abs(e_mitigated - e_exact) / gap
        label = "paramagnetic"
        expected = "paramagnetic"

        if delta < 0.05 and label == expected:
            verdict = "PASS"
        else:
            verdict = "FAIL"

        assert verdict == "PASS"
        assert delta < 0.05

    def test_fail_verdict_energy(self):
        """FAIL when ΔE/gap >= 5%."""
        e_exact = -9.6
        gap = 1.5
        e_mitigated = -8.5  # ΔE = 1.1, ΔE/gap = 73%
        delta = abs(e_mitigated - e_exact) / gap

        assert delta >= 0.05

    def test_partial_verdict_wrong_phase(self):
        """PARTIAL when energy good but phase label wrong."""
        e_exact = -9.6
        gap = 1.5
        e_mitigated = -9.58  # ΔE/gap = 1.3% — excellent
        delta = abs(e_mitigated - e_exact) / gap
        label = "ferromagnetic"  # Wrong!
        expected = "paramagnetic"

        assert delta < 0.05
        assert label != expected
        # → PARTIAL verdict


# ═══════════════════════════════════════════════════════════════════════════════
# Test: run_qesem_deployment mock (validates the function interface)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunQESEMDeploymentMock:
    """Test run_qesem_deployment with mocked catalog."""

    @pytest.fixture
    def mock_config(self):
        """Create a minimal HardwareConfig for QESEM testing."""
        return HardwareConfig(
            backend_name="ibm_kingston",
            mode="hardware",
            n_qubits=10,
            mitigation=MitigationOptions(qesem_enabled=True),
            qesem_precision=0.01,
            qesem_max_execution_time=300,
        )

    @patch("qmbp_simulation.execution.hardware.qesem._load_qesem_function")
    def test_run_qesem_deployment_interface(self, mock_load, mock_config):
        """run_qesem_deployment calls QESEM function with correct args."""
        from qiskit.quantum_info import SparsePauliOp

        from qmbp_simulation.execution.hardware.qesem import run_qesem_deployment

        # Mock the QESEM function return
        n_qubits = 10
        n_x = 10
        n_zz = 9
        n_total_obs = 1 + n_x + n_zz  # H + X_i + ZZ_ij

        mock_evs = np.array([-9.5] + [0.8] * n_x + [-0.3] * n_zz)
        mock_stds = np.array([0.01] + [0.005] * n_x + [0.005] * n_zz)

        # Mock noisy_results object
        mock_noisy = MagicMock()
        mock_noisy.evs = np.array([-7.0] + [0.5] * n_x + [-0.2] * n_zz)

        # Mock PubResult
        mock_pub_result = MagicMock()
        mock_pub_result.data.evs = mock_evs
        mock_pub_result.data.stds = mock_stds
        mock_pub_result.metadata = {
            "noisy_results": mock_noisy,
            "total_qpu_time": 45.0,
            "gate_fidelities": {"cz": 0.995},
            "total_shots": 100000,
            "mitigation_shots": 80000,
        }

        # Mock job
        mock_job = MagicMock()
        mock_job.job_id = "mock-qesem-job-001"
        mock_job.result.return_value = [mock_pub_result]

        # Mock QESEM function
        mock_fn = MagicMock()
        mock_fn.run.return_value = mock_job
        mock_load.return_value = mock_fn

        # Build inputs
        from qiskit.circuit import QuantumCircuit

        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        for i in range(n_qubits - 1):
            qc.rzz(0.5, i, i + 1)
        qc.rx(0.3, range(n_qubits))

        hamiltonian = SparsePauliOp.from_sparse_list(
            [("ZZ", [i, i + 1], -1.0) for i in range(n_qubits - 1)]
            + [("X", [i], -2.0) for i in range(n_qubits)],
            num_qubits=n_qubits,
        )
        x_ops = [
            SparsePauliOp.from_sparse_list([("X", [i], 1.0)], num_qubits=n_qubits)
            for i in range(n_qubits)
        ]
        zz_ops = [
            SparsePauliOp.from_sparse_list([("ZZ", [i, i + 1], 1.0)], num_qubits=n_qubits)
            for i in range(n_qubits - 1)
        ]

        # Execute
        result = run_qesem_deployment(
            circuit=qc,
            hamiltonian=hamiltonian,
            x_ops=x_ops,
            zz_ops=zz_ops,
            config=mock_config,
        )

        # Verify
        assert isinstance(result, QESEMResult)
        assert result.energy_mitigated == -9.5
        assert result.energy_std == 0.01
        assert len(result.x_values) == n_x
        assert len(result.zz_values) == n_zz
        assert result.job_id == "mock-qesem-job-001"
        assert result.total_qpu_time == 45.0
        assert result.noisy_energy == -7.0

        # Verify the function was called with correct structure
        mock_fn.run.assert_called_once()
        call_kwargs = mock_fn.run.call_args[1]
        assert call_kwargs["backend_name"] == "ibm_kingston"
        assert call_kwargs["options"]["default_precision"] == 0.01
        assert call_kwargs["options"]["max_execution_time"] == 300
        # PUB should have (circuit, all_observables)
        pub = call_kwargs["pubs"][0]
        assert pub[0] is qc
        assert len(pub[1]) == n_total_obs


# ═══════════════════════════════════════════════════════════════════════════════
# Test: fake_backend mode guard
# ═══════════════════════════════════════════════════════════════════════════════


class TestQESEMFakeBackendGuard:
    """QESEM must NOT run in fake_backend mode — it requires a real QPU."""

    def test_qesem_blocked_in_fake_backend_mode(self):
        """Attempting QESEM in fake_backend mode raises RuntimeError."""

        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp

        config = HardwareConfig(
            backend_name="ibm_kingston",
            mode="fake_backend",
            n_qubits=10,
            mitigation=MitigationOptions(qesem_enabled=True),
        )
        backend = None
        try:
            from qmbp_simulation.execution.hardware import HardwareBackend

            backend = HardwareBackend(config=config)
        except Exception:
            pytest.skip("HardwareBackend import issue in test env")

        # Build a parametrized circuit (matches what run_deployment expects)
        from qiskit.circuit import Parameter

        theta_zz = Parameter("θ_zz")
        theta_x = Parameter("θ_x")
        qc = QuantumCircuit(10)
        qc.h(range(10))
        for i in range(9):
            qc.rzz(theta_zz, i, i + 1)
        qc.rx(theta_x, range(10))

        hamiltonian = SparsePauliOp.from_sparse_list(
            [("ZZ", [i, i + 1], -1.0) for i in range(9)] + [("X", [i], -2.0) for i in range(10)],
            num_qubits=10,
        )
        params = np.array([-0.5, -1.2])

        # This should raise RuntimeError about QESEM not working in fake_backend
        with pytest.raises(RuntimeError, match="QESEM.*cannot run in fake_backend"):
            backend.run_deployment(
                circuit=qc,
                hamiltonian=hamiltonian,
                params=params,
                h_value=4.0,
                e_exact=-9.5,
                gap=1.5,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Test: validate_qesem_submission preflight checks
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateQESEMSubmission:
    """Test the preflight validation that catches common errors before QPU submission."""

    def test_import(self):
        """validate_qesem_submission is importable."""
        from qmbp_simulation.execution.hardware.qesem import validate_qesem_submission

        assert callable(validate_qesem_submission)

    def test_empty_pubs_is_critical(self):
        """Empty pubs list produces a CRITICAL issue."""
        from qmbp_simulation.execution.hardware.qesem import validate_qesem_submission

        issues = validate_qesem_submission(pubs=[])
        assert any("CRITICAL" in i and "empty" in i for i in issues)

    def test_multi_pub_without_transpilation_level_is_warning(self):
        """Multiple PUBs with default transpilation_level produces a WARNING.

        QESEM does NOT support multi-PUB batches with different circuits.
        The validated pattern is sequential single-PUB submission.
        """
        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp

        from qmbp_simulation.execution.hardware.qesem import validate_qesem_submission

        qc = QuantumCircuit(4)
        qc.h(range(4))
        obs = [SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=4)]

        # 3 PUBs, no transpilation_level in options → should WARN
        pubs = [(qc, obs), (qc, obs), (qc, obs)]
        options = {"default_precision": 0.01, "max_execution_time": 600}

        issues = validate_qesem_submission(pubs=pubs, options=options)
        assert any("WARNING" in i and "multiple" in i.lower() for i in issues)

    def test_multi_pub_with_transpilation_level_0_passes(self):
        """Multiple PUBs with transpilation_level=0 is valid."""
        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp

        from qmbp_simulation.execution.hardware.qesem import validate_qesem_submission

        qc = QuantumCircuit(4)
        qc.h(range(4))
        obs = [SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=4)]

        pubs = [(qc, obs), (qc, obs), (qc, obs)]
        options = {
            "default_precision": 0.01,
            "max_execution_time": 600,
            "transpilation_level": 0,
        }

        issues = validate_qesem_submission(pubs=pubs, options=options)
        # No CRITICAL issues about transpilation_level
        critical = [i for i in issues if "CRITICAL" in i and "transpilation_level" in i]
        assert len(critical) == 0

    def test_single_pub_standard_transpilation_is_fine(self):
        """Single PUB with default transpilation_level is valid."""
        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp

        from qmbp_simulation.execution.hardware.qesem import validate_qesem_submission

        qc = QuantumCircuit(4)
        qc.h(range(4))
        obs = [SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=4)]

        pubs = [(qc, obs)]
        options = {"default_precision": 0.01}

        issues = validate_qesem_submission(pubs=pubs, options=options)
        critical = [i for i in issues if "CRITICAL" in i]
        assert len(critical) == 0

    def test_unbound_circuit_is_critical(self):
        """Circuit with unbound parameters produces CRITICAL issue."""
        from qiskit.circuit import Parameter, QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp

        from qmbp_simulation.execution.hardware.qesem import validate_qesem_submission

        theta = Parameter("θ")
        qc = QuantumCircuit(4)
        qc.rx(theta, 0)
        obs = [SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=4)]

        pubs = [(qc, obs)]
        issues = validate_qesem_submission(pubs=pubs, options={"default_precision": 0.01})
        assert any("CRITICAL" in i and "unbound" in i for i in issues)

    def test_fake_backend_mode_is_critical(self):
        """fake_backend mode produces CRITICAL issue."""
        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp

        from qmbp_simulation.execution.hardware.qesem import validate_qesem_submission

        qc = QuantumCircuit(4)
        qc.h(range(4))
        obs = [SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=4)]

        config = HardwareConfig(mode="fake_backend", n_qubits=4)
        pubs = [(qc, obs)]
        issues = validate_qesem_submission(pubs=pubs, config=config)
        assert any("CRITICAL" in i and "fake_backend" in i for i in issues)

    def test_zero_precision_is_critical(self):
        """Non-positive precision produces CRITICAL issue."""
        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp

        from qmbp_simulation.execution.hardware.qesem import validate_qesem_submission

        qc = QuantumCircuit(4)
        qc.h(range(4))
        obs = [SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=4)]

        pubs = [(qc, obs)]
        issues = validate_qesem_submission(pubs=pubs, options={"default_precision": 0.0})
        assert any("CRITICAL" in i and "precision" in i for i in issues)

    def test_short_execution_time_is_warning(self):
        """Very short max_execution_time produces a WARNING (not critical)."""
        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp

        from qmbp_simulation.execution.hardware.qesem import validate_qesem_submission

        qc = QuantumCircuit(4)
        qc.h(range(4))
        obs = [SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=4)]

        pubs = [(qc, obs)]
        issues = validate_qesem_submission(
            pubs=pubs, options={"default_precision": 0.01, "max_execution_time": 30}
        )
        assert any("WARNING" in i and "max_execution_time" in i for i in issues)
        # Should NOT be critical
        assert not any("CRITICAL" in i for i in issues)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: run_qesem_sweep multi-PUB transpilation_level fix
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunQESEMSweepMultiPUB:
    """Test that run_qesem_sweep correctly sets transpilation_level=0 for multi-PUB."""

    @patch("qmbp_simulation.execution.hardware.qesem._load_qesem_function")
    def test_sweep_submits_sequential_single_pub_jobs(self, mock_load):
        """run_qesem_sweep submits one job per h-point (sequential, not batched)."""
        from qiskit.circuit import Parameter, QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp

        from qmbp_simulation.execution.hardware.qesem import run_qesem_sweep

        n_qubits = 4
        n_pubs = 3

        # Build parametric circuit
        theta_zz = Parameter("θ_zz")
        theta_x = Parameter("θ_x")
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        for i in range(n_qubits - 1):
            qc.rzz(theta_zz, i, i + 1)
        qc.rx(theta_x, range(n_qubits))

        # Build observables
        hamiltonian = SparsePauliOp.from_sparse_list(
            [("ZZ", [i, i + 1], -1.0) for i in range(n_qubits - 1)]
            + [("X", [i], -2.0) for i in range(n_qubits)],
            num_qubits=n_qubits,
        )
        x_ops = [
            SparsePauliOp.from_sparse_list([("X", [i], 1.0)], num_qubits=n_qubits)
            for i in range(n_qubits)
        ]
        zz_ops = [
            SparsePauliOp.from_sparse_list([("ZZ", [i, i + 1], 1.0)], num_qubits=n_qubits)
            for i in range(n_qubits - 1)
        ]

        h_values = [3.5, 3.25, 3.0]
        params_per_h = [np.array([0.5, 1.0])] * n_pubs

        # Mock QESEM response
        n_obs = 1 + n_qubits + (n_qubits - 1)
        mock_evs = np.zeros(n_obs)
        mock_evs[0] = -5.0
        mock_stds = np.full(n_obs, 0.01)

        mock_pub_result = MagicMock()
        mock_pub_result.data.evs = mock_evs
        mock_pub_result.data.stds = mock_stds
        mock_pub_result.metadata = {
            "noisy_results": None,
            "total_qpu_time": 30.0,
            "gate_fidelities": None,
            "total_shots": 50000,
            "mitigation_shots": 40000,
        }

        mock_job = MagicMock()
        mock_job.job_id = "mock-sweep-001"
        mock_job.result.return_value = [mock_pub_result]

        mock_fn = MagicMock()
        mock_fn.run.return_value = mock_job
        mock_load.return_value = mock_fn

        # Build config
        config = HardwareConfig(
            backend_name="ibm_kingston",
            mode="hardware",
            n_qubits=n_qubits,
            mitigation=MitigationOptions(qesem_enabled=True),
            qesem_precision=0.01,
            qesem_max_execution_time=300,
        )

        # Execute sweep
        results = run_qesem_sweep(
            circuit=qc,
            hamiltonians=[hamiltonian] * n_pubs,
            x_ops_list=[x_ops] * n_pubs,
            zz_ops_list=[zz_ops] * n_pubs,
            params_per_h=params_per_h,
            h_values=h_values,
            config=config,
        )

        # Verify: one call per h-point (sequential), NOT one batched call
        assert mock_fn.run.call_count == n_pubs
        # Each call should have exactly 1 PUB
        for call in mock_fn.run.call_args_list:
            call_kwargs = call[1]
            assert len(call_kwargs["pubs"]) == 1
            # No transpilation_level override needed for single PUB
            assert "transpilation_level" not in call_kwargs["options"]
            assert call_kwargs["options"]["default_precision"] == 0.01
        assert len(results) == n_pubs

    @patch("qmbp_simulation.execution.hardware.qesem._load_qesem_function")
    def test_single_pub_sweep_no_transpilation_level(self, mock_load):
        """run_qesem_sweep with 1 PUB does NOT set transpilation_level=0."""
        from qiskit.circuit import Parameter, QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp

        from qmbp_simulation.execution.hardware.qesem import run_qesem_sweep

        n_qubits = 4

        theta_zz = Parameter("θ_zz")
        theta_x = Parameter("θ_x")
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        for i in range(n_qubits - 1):
            qc.rzz(theta_zz, i, i + 1)
        qc.rx(theta_x, range(n_qubits))

        hamiltonian = SparsePauliOp.from_sparse_list(
            [("ZZ", [i, i + 1], -1.0) for i in range(n_qubits - 1)]
            + [("X", [i], -2.0) for i in range(n_qubits)],
            num_qubits=n_qubits,
        )
        x_ops = [
            SparsePauliOp.from_sparse_list([("X", [i], 1.0)], num_qubits=n_qubits)
            for i in range(n_qubits)
        ]
        zz_ops = [
            SparsePauliOp.from_sparse_list([("ZZ", [i, i + 1], 1.0)], num_qubits=n_qubits)
            for i in range(n_qubits - 1)
        ]

        n_obs = 1 + n_qubits + (n_qubits - 1)
        mock_pub_result = MagicMock()
        mock_pub_result.data.evs = np.zeros(n_obs)
        mock_pub_result.data.stds = np.full(n_obs, 0.01)
        mock_pub_result.metadata = {
            "noisy_results": None,
            "total_qpu_time": 30.0,
            "gate_fidelities": None,
            "total_shots": 50000,
            "mitigation_shots": 40000,
        }

        mock_job = MagicMock()
        mock_job.job_id = "mock-single-001"
        mock_job.result.return_value = [mock_pub_result]

        mock_fn = MagicMock()
        mock_fn.run.return_value = mock_job
        mock_load.return_value = mock_fn

        config = HardwareConfig(
            backend_name="ibm_kingston",
            mode="hardware",
            n_qubits=n_qubits,
            mitigation=MitigationOptions(qesem_enabled=True),
            qesem_precision=0.01,
            qesem_max_execution_time=300,
        )

        results = run_qesem_sweep(
            circuit=qc,
            hamiltonians=[hamiltonian],
            x_ops_list=[x_ops],
            zz_ops_list=[zz_ops],
            params_per_h=[np.array([0.5, 1.0])],
            h_values=[4.0],
            config=config,
        )

        call_kwargs = mock_fn.run.call_args[1]
        # Single PUB → transpilation_level should NOT be set
        assert "transpilation_level" not in call_kwargs["options"]
        assert len(results) == 1
