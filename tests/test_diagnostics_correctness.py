"""Correctness tests for DiagnosticCollector critical flows.

Validates:
1. Energy decomposition: e_exact reconstruction, ordering clamp, additivity invariant
2. Phase tracking: all phases appear in _completed_phases and checkpoint data
3. Config propagation: _config_dict appears in checkpoint files
4. Worst convergence h: correctly identifies the h-point with most iterations
5. Theta smoothness: computed correctly from accumulated vectors

These tests target specific bugs found during code review.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.poc.v6.diagnostics import DiagnosticCollector  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Energy Decomposition Correctness
# ─────────────────────────────────────────────────────────────────────────────


class TestEnergyDecomposition:
    """Verify energy decomposition computes correct values from DeployResultV61."""

    def _make_result(
        self,
        predicted_energy: float,
        delta_e: float,
        raw_energy: float | None = None,
        delta_e_over_gap: float = 0.03,
    ):
        """Create a mock DeployResultV61 with the given energy values."""

        class MockResult:
            pass

        r = MockResult()
        r.predicted_energy = predicted_energy
        r.delta_e = delta_e
        r.delta_e_over_gap = delta_e_over_gap
        r.raw_energy = raw_energy
        r.mag_x_pred = 0.7
        r.corr_zz_pred = 0.3
        r.total_shots = 4096
        return r

    def test_energy_decomposition_additivity(self):
        """error_from_circuit + error_from_mpnn == |e_predicted - e_exact| within 1e-12.

        This is the core invariant from Requirement 6.5.
        """
        # Typical case: e_exact=-9.25, e_vqe_ceiling=-9.22, e_predicted=-9.15
        # delta_e = |predicted - exact| = |-9.15 - (-9.25)| = 0.10
        result = self._make_result(
            predicted_energy=-9.15,
            delta_e=0.10,
            raw_energy=-9.22,  # VQE ceiling (pre-ZNE)
        )

        collector = DiagnosticCollector(verbose=False)
        collector.record_deployment(h_test=1.25, result=result, per_layout_data=None)

        diag = collector.to_dict()
        decomp = diag["phase4"]["energy_decomposition"]

        # Verify the decomposition
        assert decomp["e_exact"] == pytest.approx(-9.25, abs=1e-12)
        assert decomp["e_vqe_ceiling"] == pytest.approx(-9.22, abs=1e-12)
        assert decomp["e_mpnn_predicted"] == pytest.approx(-9.15, abs=1e-12)
        assert decomp["error_from_circuit"] == pytest.approx(0.03, abs=1e-12)
        assert decomp["error_from_mpnn"] == pytest.approx(0.07, abs=1e-12)

        # Core invariant
        total_error = abs(decomp["e_mpnn_predicted"] - decomp["e_exact"])
        sum_errors = decomp["error_from_circuit"] + decomp["error_from_mpnn"]
        assert sum_errors == pytest.approx(total_error, abs=1e-12)

    def test_energy_decomposition_no_raw_energy(self):
        """When raw_energy is None, e_vqe_ceiling defaults to e_predicted.

        This means error_from_mpnn = 0 and all error is attributed to circuit.
        """
        result = self._make_result(
            predicted_energy=-9.15,
            delta_e=0.10,
            raw_energy=None,  # No raw energy available
        )

        collector = DiagnosticCollector(verbose=False)
        collector.record_deployment(h_test=1.25, result=result, per_layout_data=None)

        diag = collector.to_dict()
        decomp = diag["phase4"]["energy_decomposition"]

        # When raw_energy is None, e_vqe_ceiling = e_predicted
        # So error_from_mpnn = |e_predicted - e_vqe_ceiling| = 0
        assert decomp["error_from_mpnn"] == pytest.approx(0.0, abs=1e-12)
        # All error attributed to circuit
        assert decomp["error_from_circuit"] == pytest.approx(0.10, abs=1e-12)

    def test_energy_decomposition_ordering_clamp(self):
        """When raw_energy violates ordering (noise), it's clamped to valid range.

        If raw_energy < e_exact or raw_energy > e_predicted, the clamp
        ensures e_exact <= e_vqe_ceiling <= e_predicted.
        """
        # Case: raw_energy below exact (impossible physically, but noise can cause it)
        result = self._make_result(
            predicted_energy=-9.15,
            delta_e=0.10,
            raw_energy=-9.30,  # Below exact (-9.25) — should be clamped to -9.25
        )

        collector = DiagnosticCollector(verbose=False)
        collector.record_deployment(h_test=1.25, result=result, per_layout_data=None)

        diag = collector.to_dict()
        decomp = diag["phase4"]["energy_decomposition"]

        # e_vqe_ceiling should be clamped to e_exact (-9.25)
        assert decomp["e_vqe_ceiling"] == pytest.approx(-9.25, abs=1e-12)
        assert decomp["error_from_circuit"] == pytest.approx(0.0, abs=1e-12)
        assert decomp["error_from_mpnn"] == pytest.approx(0.10, abs=1e-12)

        # Invariant still holds
        total_error = abs(decomp["e_mpnn_predicted"] - decomp["e_exact"])
        sum_errors = decomp["error_from_circuit"] + decomp["error_from_mpnn"]
        assert sum_errors == pytest.approx(total_error, abs=1e-12)

    def test_energy_decomposition_raw_above_predicted(self):
        """When raw_energy > e_predicted (noise), it's clamped to e_predicted.

        This means error_from_mpnn = 0.
        """
        result = self._make_result(
            predicted_energy=-9.15,
            delta_e=0.10,
            raw_energy=-9.10,  # Above predicted — should be clamped to -9.15
        )

        collector = DiagnosticCollector(verbose=False)
        collector.record_deployment(h_test=1.25, result=result, per_layout_data=None)

        diag = collector.to_dict()
        decomp = diag["phase4"]["energy_decomposition"]

        # e_vqe_ceiling clamped to e_predicted
        assert decomp["e_vqe_ceiling"] == pytest.approx(-9.15, abs=1e-12)
        assert decomp["error_from_mpnn"] == pytest.approx(0.0, abs=1e-12)
        assert decomp["error_from_circuit"] == pytest.approx(0.10, abs=1e-12)

    def test_energy_decomposition_exact_reconstruction(self):
        """e_exact is correctly reconstructed as predicted_energy - delta_e."""
        # delta_e = |predicted - exact| = 0.05
        # predicted = -8.0, so exact = -8.0 - 0.05 = -8.05
        result = self._make_result(
            predicted_energy=-8.0,
            delta_e=0.05,
            raw_energy=-8.02,
        )

        collector = DiagnosticCollector(verbose=False)
        collector.record_deployment(h_test=1.5, result=result, per_layout_data=None)

        diag = collector.to_dict()
        decomp = diag["phase4"]["energy_decomposition"]

        assert decomp["e_exact"] == pytest.approx(-8.05, abs=1e-12)
        assert decomp["e_vqe_ceiling"] == pytest.approx(-8.02, abs=1e-12)
        assert decomp["e_mpnn_predicted"] == pytest.approx(-8.0, abs=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Phase Tracking in _completed_phases
# ─────────────────────────────────────────────────────────────────────────────


class TestPhaseTracking:
    """Verify all phases are correctly tracked in _completed_phases."""

    def test_phase1_tracked(self):
        """record_phase1 adds 'phase1' to _completed_phases."""
        collector = DiagnosticCollector(verbose=False)
        collector.record_phase1(n_points=27, elapsed_s=1.0, gap_min=0.1)
        assert "phase1" in collector._completed_phases

    def test_phase2_tracked_after_first_vqe_point(self):
        """record_vqe_point adds 'phase2' to _completed_phases on first call."""
        collector = DiagnosticCollector(verbose=False)
        collector.record_vqe_point(
            h=2.0,
            n_iters=50,
            restart_energies=[-5.0, -5.1],
            theta_opt=np.array([0.1, 0.2, 0.3, 0.4]),
            elapsed_s=0.5,
        )
        assert "phase2" in collector._completed_phases

    def test_phase2_not_duplicated(self):
        """Multiple record_vqe_point calls don't duplicate 'phase2'."""
        collector = DiagnosticCollector(verbose=False)
        for h in [2.0, 1.5, 1.0]:
            collector.record_vqe_point(
                h=h,
                n_iters=50,
                restart_energies=[-5.0],
                theta_opt=np.array([0.1, 0.2, 0.3, 0.4]),
                elapsed_s=0.5,
            )
        assert collector._completed_phases.count("phase2") == 1

    def test_phase3_tracked(self):
        """record_mpnn_per_h_error adds 'phase3' to _completed_phases."""
        collector = DiagnosticCollector(verbose=False)
        collector.record_mpnn_per_h_error(np.array([1.0, 1.5]), np.array([0.01, 0.02]))
        assert "phase3" in collector._completed_phases

    def test_phase4_tracked(self):
        """record_deployment adds 'phase4' to _completed_phases."""
        collector = DiagnosticCollector(verbose=False)

        class MockResult:
            predicted_energy = -8.5
            delta_e = 0.1
            delta_e_over_gap = 0.03
            mag_x_pred = 0.7
            corr_zz_pred = 0.3
            total_shots = 4096
            raw_energy = -8.4

        collector.record_deployment(h_test=1.25, result=MockResult(), per_layout_data=None)
        assert "phase4" in collector._completed_phases

    def test_all_phases_tracked_in_full_flow(self):
        """Full recording flow tracks all 4 phases."""
        collector = DiagnosticCollector(verbose=False)

        collector.record_phase1(n_points=5, elapsed_s=0.5, gap_min=0.2)
        collector.record_vqe_point(
            h=2.0,
            n_iters=100,
            restart_energies=[-5.0, -5.1],
            theta_opt=np.array([0.1, 0.2, 0.3, 0.4]),
            elapsed_s=1.0,
        )
        collector.record_mpnn_per_h_error(np.array([1.0, 1.5]), np.array([0.01, 0.02]))

        class MockResult:
            predicted_energy = -8.5
            delta_e = 0.1
            delta_e_over_gap = 0.03
            mag_x_pred = 0.7
            corr_zz_pred = 0.3
            total_shots = 4096
            raw_energy = -8.4

        collector.record_deployment(h_test=1.25, result=MockResult(), per_layout_data=None)

        assert collector._completed_phases == ["phase1", "phase2", "phase3", "phase4"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Config Propagation in Checkpoints
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckpointConfig:
    """Verify _config_dict is included in checkpoint files."""

    def test_checkpoint_contains_config(self, tmp_path):
        """Checkpoint file includes the config dict."""
        collector = DiagnosticCollector(verbose=False, save_dir=tmp_path, run_id="test1234")
        collector._config_dict = {"N": 6, "seed": 42, "h_test": 1.25}

        collector.record_phase1(n_points=27, elapsed_s=1.0, gap_min=0.1)
        path = collector.save_checkpoint("phase1")

        assert path is not None
        with open(path) as f:
            data = json.load(f)

        assert data["config"] == {"N": 6, "seed": 42, "h_test": 1.25}
        assert data["run_id"] == "test1234"
        assert data["phase"] == "phase1"
        assert "phase1" in data["completed_phases"]

    def test_checkpoint_empty_config_when_not_set(self, tmp_path):
        """Checkpoint has empty config dict when _config_dict not set."""
        collector = DiagnosticCollector(verbose=False, save_dir=tmp_path, run_id="abcd1234")
        collector.record_phase1(n_points=5, elapsed_s=0.3, gap_min=0.5)
        path = collector.save_checkpoint("phase1")

        with open(path) as f:
            data = json.load(f)

        assert data["config"] == {}

    def test_checkpoint_tracks_completed_phases(self, tmp_path):
        """Checkpoint completed_phases grows as phases complete."""
        collector = DiagnosticCollector(verbose=False, save_dir=tmp_path, run_id="abcd5678")

        collector.record_phase1(n_points=5, elapsed_s=0.3, gap_min=0.5)
        p1 = collector.save_checkpoint("phase1")

        collector.record_vqe_point(
            h=2.0,
            n_iters=50,
            restart_energies=[-5.0],
            theta_opt=np.array([0.1, 0.2, 0.3, 0.4]),
            elapsed_s=0.5,
        )
        p2 = collector.save_checkpoint("phase2")

        with open(p1) as f:
            d1 = json.load(f)
        with open(p2) as f:
            d2 = json.load(f)

        assert d1["completed_phases"] == ["phase1"]
        assert "phase1" in d2["completed_phases"]
        assert "phase2" in d2["completed_phases"]

    def test_cleanup_removes_all_checkpoints(self, tmp_path):
        """cleanup_checkpoints removes all checkpoint files for the run_id."""
        collector = DiagnosticCollector(verbose=False, save_dir=tmp_path, run_id="cleanup1")

        collector.record_phase1(n_points=5, elapsed_s=0.3, gap_min=0.5)
        collector.save_checkpoint("phase1")
        collector.record_vqe_point(
            h=2.0,
            n_iters=50,
            restart_energies=[-5.0],
            theta_opt=np.array([0.1, 0.2]),
            elapsed_s=0.5,
        )
        collector.save_checkpoint("phase2")

        # Verify files exist
        assert (tmp_path / "checkpoint_cleanup1_phase1.json").exists()
        assert (tmp_path / "checkpoint_cleanup1_phase2.json").exists()

        collector.cleanup_checkpoints()

        # Verify files removed
        assert not (tmp_path / "checkpoint_cleanup1_phase1.json").exists()
        assert not (tmp_path / "checkpoint_cleanup1_phase2.json").exists()


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Worst Convergence H and Theta Smoothness
# ─────────────────────────────────────────────────────────────────────────────


class TestDerivedMetrics:
    """Verify worst_convergence_h and theta_smoothness are computed correctly."""

    def test_worst_convergence_h_identifies_max_iterations(self):
        """worst_convergence_h is the h-point with the most iterations."""
        collector = DiagnosticCollector(verbose=False)

        # h=2.0: 50 iters, h=1.5: 200 iters (worst), h=1.0: 80 iters
        for h, n_iters in [(2.0, 50), (1.5, 200), (1.0, 80)]:
            collector.record_vqe_point(
                h=h,
                n_iters=n_iters,
                restart_energies=[-5.0],
                theta_opt=np.array([0.1, 0.2, 0.3, 0.4]),
                elapsed_s=1.0,
            )

        diag = collector.to_dict()
        assert diag["phase2"]["worst_convergence_h"] == 1.5

    def test_theta_smoothness_computed_from_accumulated_vectors(self):
        """theta_smoothness = max_i ||θ_i - θ_{i-1}||_∞ from recorded vectors."""
        collector = DiagnosticCollector(verbose=False)

        # Three theta vectors: differences are [0.05, 0.05, 0.05, 0.05] and [1.5, 0.05, 0.05, 0.05]
        thetas = [
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.15, 0.25, 0.35, 0.45]),  # max diff = 0.05
            np.array([1.65, 0.30, 0.40, 0.50]),  # max diff = 1.5
        ]
        for i, theta in enumerate(thetas):
            collector.record_vqe_point(
                h=2.0 - i * 0.5,
                n_iters=50,
                restart_energies=[-5.0],
                theta_opt=theta,
                elapsed_s=0.5,
            )

        diag = collector.to_dict()
        assert diag["phase2"]["theta_smoothness"] == pytest.approx(1.5, abs=1e-10)

    def test_theta_smoothness_none_with_single_point(self):
        """theta_smoothness is None when fewer than 2 h-points recorded."""
        collector = DiagnosticCollector(verbose=False)
        collector.record_vqe_point(
            h=2.0,
            n_iters=50,
            restart_energies=[-5.0],
            theta_opt=np.array([0.1, 0.2, 0.3, 0.4]),
            elapsed_s=0.5,
        )

        diag = collector.to_dict()
        assert diag["phase2"]["theta_smoothness"] is None

    def test_n_iterations_used_directly(self):
        """per_h_iterations records the n_iters value passed to record_vqe_point."""
        collector = DiagnosticCollector(verbose=False)

        collector.record_vqe_point(
            h=2.0,
            n_iters=142,
            restart_energies=[-5.0],
            theta_opt=np.array([0.1, 0.2, 0.3, 0.4]),
            elapsed_s=0.5,
        )
        collector.record_vqe_point(
            h=1.5,
            n_iters=87,
            restart_energies=[-5.0],
            theta_opt=np.array([0.15, 0.25, 0.35, 0.45]),
            elapsed_s=0.6,
        )

        diag = collector.to_dict()
        assert diag["phase2"]["per_h_iterations"] == [142, 87]
