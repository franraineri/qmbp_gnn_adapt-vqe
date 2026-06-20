"""Integration tests for pre-submission validation and benchmark CLI fixes.

Validates:
- Pre-submission manifest generation (QASM, PNGs, quality checks, ZNE check)
- HardwareBackend.run_deployment end-to-end with fake_backend mode
- resolve_configs prefix matching (C1 ≠ C10, C2 ≠ C20, etc.)
- Benchmark runner imports and function availability

These tests ensure that every hardware submission produces a verifiable
audit trail and that the CLI config resolution is correct.

Marked @pytest.mark.slow — excluded from `make test`, included in `make test-full`.

Expected runtime:
- TestBenchmarkRunnerImports: <1s (fast, no FakeTorino)
- TestResolveConfigsPrefixMatching: <1s (fast, no FakeTorino)
- TestPreSubmissionManifest: ~3-5min (full HardwareBackend pipeline with FakeTorino)
- TestRunDeploymentResult: ~3-5min (same)

Reference: test_pre_submission.py, test_fixes.py (original ad-hoc scripts).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC
from pathlib import Path

import numpy as np
import pytest
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qmbp_simulation.execution.backends import MitigationOptions
from qmbp_simulation.execution.hardware.backend import HardwareBackend
from qmbp_simulation.execution.hardware.config import HardwareConfig
from scripts.experiment_runners.hardware.run_mitigation_benchmark import (
    resolve_configs,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Benchmark Runner Imports (regression guard)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBenchmarkRunnerImports:
    """Verify all critical functions import without errors."""

    def test_resolve_configs_imports(self):
        """resolve_configs is importable."""
        from scripts.experiment_runners.hardware.run_mitigation_benchmark import (
            resolve_configs,
        )

        assert callable(resolve_configs)

    def test_execute_hardware_batched_imports(self):
        """_execute_hardware_batched is importable."""
        from scripts.experiment_runners.hardware.run_mitigation_benchmark import (
            _execute_hardware_batched,
        )

        assert callable(_execute_hardware_batched)

    def test_execute_hardware_runtime_imports(self):
        """_execute_hardware_runtime is importable."""
        from scripts.experiment_runners.hardware.run_mitigation_benchmark import (
            _execute_hardware_runtime,
        )

        assert callable(_execute_hardware_runtime)

    def test_route_execution_imports(self):
        """route_execution is importable."""
        from scripts.experiment_runners.hardware.run_mitigation_benchmark import (
            route_execution,
        )

        assert callable(route_execution)

    def test_run_benchmark_imports(self):
        """run_benchmark is importable."""
        from scripts.experiment_runners.hardware.run_mitigation_benchmark import (
            run_benchmark,
        )

        assert callable(run_benchmark)

    def test_main_imports(self):
        """main entrypoint is importable."""
        from scripts.experiment_runners.hardware.run_mitigation_benchmark import (
            main,
        )

        assert callable(main)

    def test_timezone_available(self):
        """timezone.utc is available (regression: was missing import)."""

        assert UTC is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. resolve_configs Prefix Matching
# ═══════════════════════════════════════════════════════════════════════════════


class TestResolveConfigsPrefixMatching:
    """Verify shortname→config_id resolution handles prefix collisions.

    Critical regression: C1 must NOT match C10-C19, C2 must NOT match C20.
    """

    def _ns(self, configs: str) -> argparse.Namespace:
        """Helper to build Namespace with configs filter."""
        return argparse.Namespace(priority=None, configs=configs)

    def test_c0_resolves_to_c0_raw(self):
        """C0 matches only C0_raw."""
        result = resolve_configs(self._ns("C0"))
        assert result == ["C0_raw"]

    def test_c1_resolves_to_c1_dd_only(self):
        """C1 matches only C1_dd_only, NOT C10-C19."""
        result = resolve_configs(self._ns("C1"))
        assert result == ["C1_dd_only"]

    def test_c2_resolves_to_c2_dd_tw(self):
        """C2 matches only C2_dd_tw, NOT C20_aqc_dd_tw."""
        result = resolve_configs(self._ns("C2"))
        assert result == ["C2_dd_tw"]

    def test_c10_resolves_to_c10_kitchen_sink(self):
        """C10 matches C10_kitchen_sink (full prefix)."""
        result = resolve_configs(self._ns("C10"))
        assert result == ["C10_kitchen_sink"]

    def test_multi_config_comma_separated(self):
        """Comma-separated shortnames resolve correctly."""
        result = resolve_configs(self._ns("C0,C1,C3,C5"))
        assert result == ["C0_raw", "C1_dd_only", "C3_full_gf", "C5_full_pea_balanced"]

    def test_full_config_id_accepted(self):
        """Full config_id string works as input."""
        result = resolve_configs(self._ns("C5_full_pea_balanced"))
        assert result == ["C5_full_pea_balanced"]

    def test_invalid_config_raises_valueerror(self):
        """Non-existent config shortname raises ValueError."""
        with pytest.raises(ValueError, match="No config matches"):
            resolve_configs(self._ns("C99"))

    @pytest.mark.parametrize(
        "shortname,expected",
        [
            ("C3", ["C3_full_gf"]),
            ("C4", ["C4_full_pea_light"]),
            ("C5", ["C5_full_pea_balanced"]),
            ("C6", ["C6_full_pea_heavy"]),
            ("C16", ["C16_aqc_pea"]),
            ("C18", ["C18_aqc_raw"]),
            ("C19", ["C19_aqc_gf"]),
            ("C20", ["C20_aqc_dd_tw"]),
        ],
    )
    def test_shortname_resolution_table(self, shortname, expected):
        """Each Cx shortname resolves to exactly one config."""
        result = resolve_configs(self._ns(shortname))
        assert result == expected


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Pre-Submission Manifest Generation
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestPreSubmissionManifest:
    """Validate HardwareBackend.run_deployment produces audit artifacts.

    Every hardware submission must generate:
    - Pre-submission manifest JSON (transpiled quality, ZNE check, calibration)
    - Optional: QASM files, circuit diagram PNGs
    """

    @pytest.fixture
    def simple_circuit(self):
        """4-qubit TFIM-like circuit for fast testing."""
        theta = Parameter("t")
        qc = QuantumCircuit(4)
        qc.h(range(4))
        qc.rzz(theta, 0, 1)
        qc.rzz(theta, 1, 2)
        qc.rzz(theta, 2, 3)
        qc.rx(theta, range(4))
        return qc

    @pytest.fixture
    def simple_hamiltonian(self):
        """4-qubit TFIM Hamiltonian."""
        return SparsePauliOp.from_list(
            [
                ("ZZII", -1.0),
                ("IZZI", -1.0),
                ("IIZZ", -1.0),
                ("XIII", -0.5),
                ("IXII", -0.5),
                ("IIXI", -0.5),
                ("IIIX", -0.5),
            ]
        )

    @pytest.fixture
    def hw_backend(self, tmp_path):
        """HardwareBackend configured for fake_backend mode."""
        config = HardwareConfig(
            mode="fake_backend",
            n_qubits=4,
            shots=1024,
            n_layouts=1,
            n_candidates=5,
            output_dir=str(tmp_path),
            mitigation=MitigationOptions(
                dd_enabled=True,
                twirling_enabled=True,
                trex_enabled=True,
                zne_enabled=True,
                zne_amplifier="gate_folding",
            ),
        )
        return HardwareBackend(config=config)

    def test_manifest_created(self, hw_backend, simple_circuit, simple_hamiltonian, tmp_path):
        """run_deployment creates a pre-submission manifest."""
        params = np.array([0.3])
        hw_backend.run_deployment(
            circuit=simple_circuit,
            hamiltonian=simple_hamiltonian,
            params=params,
            h_value=4.0,
            e_exact=-3.5,
            gap=2.0,
            expected_label="paramagnetic",
        )

        output_path = Path(tmp_path)
        manifests = list(output_path.rglob("*pre_submission_manifest*"))
        assert len(manifests) >= 1, "No pre-submission manifest found"

    def test_manifest_contains_validation_section(
        self, hw_backend, simple_circuit, simple_hamiltonian, tmp_path
    ):
        """Manifest must contain validation with transpiled_quality_per_layout."""
        params = np.array([0.3])
        hw_backend.run_deployment(
            circuit=simple_circuit,
            hamiltonian=simple_hamiltonian,
            params=params,
            h_value=4.0,
            e_exact=-3.5,
            gap=2.0,
            expected_label="paramagnetic",
        )

        output_path = Path(tmp_path)
        manifests = list(output_path.rglob("*pre_submission_manifest*"))
        manifest = json.loads(manifests[0].read_text())

        assert "validation" in manifest
        assert "transpiled_quality_per_layout" in manifest["validation"]
        quality = manifest["validation"]["transpiled_quality_per_layout"]
        assert len(quality) >= 1

    def test_manifest_contains_calibration_snapshot(
        self, hw_backend, simple_circuit, simple_hamiltonian, tmp_path
    ):
        """Manifest must contain calibration_snapshot for TLS drift tracking."""
        params = np.array([0.3])
        hw_backend.run_deployment(
            circuit=simple_circuit,
            hamiltonian=simple_hamiltonian,
            params=params,
            h_value=4.0,
            e_exact=-3.5,
            gap=2.0,
            expected_label="paramagnetic",
        )

        output_path = Path(tmp_path)
        manifests = list(output_path.rglob("*pre_submission_manifest*"))
        manifest = json.loads(manifests[0].read_text())

        assert "calibration_snapshot" in manifest

    def test_manifest_contains_execution_target(
        self, hw_backend, simple_circuit, simple_hamiltonian, tmp_path
    ):
        """Manifest must record execution target h-value."""
        params = np.array([0.3])
        hw_backend.run_deployment(
            circuit=simple_circuit,
            hamiltonian=simple_hamiltonian,
            params=params,
            h_value=4.0,
            e_exact=-3.5,
            gap=2.0,
            expected_label="paramagnetic",
        )

        output_path = Path(tmp_path)
        manifests = list(output_path.rglob("*pre_submission_manifest*"))
        manifest = json.loads(manifests[0].read_text())

        assert "execution_target" in manifest
        assert manifest["execution_target"]["h_value"] == 4.0

    def test_manifest_contains_zne_check(
        self, hw_backend, simple_circuit, simple_hamiltonian, tmp_path
    ):
        """Manifest must include circuit_zne_check with 2Q gate count."""
        params = np.array([0.3])
        hw_backend.run_deployment(
            circuit=simple_circuit,
            hamiltonian=simple_hamiltonian,
            params=params,
            h_value=4.0,
            e_exact=-3.5,
            gap=2.0,
            expected_label="paramagnetic",
        )

        output_path = Path(tmp_path)
        manifests = list(output_path.rglob("*pre_submission_manifest*"))
        manifest = json.loads(manifests[0].read_text())

        zne_check = manifest["validation"]["circuit_zne_check"]
        assert "two_qubit_gate_count" in zne_check
        assert isinstance(zne_check["two_qubit_gate_count"], int)

    def test_manifest_contains_circuit_fingerprints(
        self, hw_backend, simple_circuit, simple_hamiltonian, tmp_path
    ):
        """Manifest must include circuit fingerprints in layouts section."""
        params = np.array([0.3])
        hw_backend.run_deployment(
            circuit=simple_circuit,
            hamiltonian=simple_hamiltonian,
            params=params,
            h_value=4.0,
            e_exact=-3.5,
            gap=2.0,
            expected_label="paramagnetic",
        )

        output_path = Path(tmp_path)
        manifests = list(output_path.rglob("*pre_submission_manifest*"))
        manifest = json.loads(manifests[0].read_text())

        assert "layouts" in manifest
        assert "circuit_fingerprints" in manifest["layouts"]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. run_deployment Result Validation
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestRunDeploymentResult:
    """Validate that run_deployment returns a proper DeployResult."""

    @pytest.fixture
    def deploy_result(self, tmp_path):
        """Execute run_deployment and return the result."""
        config = HardwareConfig(
            mode="fake_backend",
            n_qubits=4,
            shots=1024,
            n_layouts=1,
            n_candidates=5,
            output_dir=str(tmp_path),
            mitigation=MitigationOptions(
                dd_enabled=True,
                twirling_enabled=True,
                trex_enabled=True,
                zne_enabled=True,
                zne_amplifier="gate_folding",
            ),
        )
        backend = HardwareBackend(config=config)

        theta = Parameter("t")
        qc = QuantumCircuit(4)
        qc.h(range(4))
        qc.rzz(theta, 0, 1)
        qc.rzz(theta, 1, 2)
        qc.rzz(theta, 2, 3)
        qc.rx(theta, range(4))

        H = SparsePauliOp.from_list(
            [
                ("ZZII", -1.0),
                ("IZZI", -1.0),
                ("IIZZ", -1.0),
                ("XIII", -0.5),
                ("IXII", -0.5),
                ("IIXI", -0.5),
                ("IIIX", -0.5),
            ]
        )

        return backend.run_deployment(
            circuit=qc,
            hamiltonian=H,
            params=np.array([0.3]),
            h_value=4.0,
            e_exact=-3.5,
            gap=2.0,
            expected_label="paramagnetic",
        )

    def test_result_has_verdict(self, deploy_result):
        """DeployResult must have a verdict field."""
        assert hasattr(deploy_result, "verdict")
        assert deploy_result.verdict in ("PASS", "FAIL", "MARGINAL")

    def test_result_has_delta_e_gap(self, deploy_result):
        """DeployResult must have delta_e_gap (primary success metric)."""
        assert hasattr(deploy_result, "delta_e_gap")
        assert np.isfinite(deploy_result.delta_e_gap)
        assert deploy_result.delta_e_gap >= 0
