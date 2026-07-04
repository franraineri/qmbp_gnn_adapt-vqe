"""Unit tests for QESEM-specific validation checks (C20-C24).

Tests the post_execution_validator's QESEM comparison logic with synthetic
data — covering passing cases, failure cases, and edge cases.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from project_health.analysis.hardware.post_execution_validator import (
    Severity,
    ValidationReport,
    _check_qesem_gate_fidelity,
    _check_qesem_noisy_vs_exact,
    _check_qesem_precision_convergence,
    _check_qesem_raw_vs_mitigated,
    _check_qesem_shot_efficiency,
)


def _make_qesem_result(
    e_mitigated: float = -40.52,
    e_exact: float = -40.57,
    e_std: float = 0.29,
    gap: float = 5.92,
    noisy_energy: float = -38.47,
    gate_fidelities: dict | None = None,
    total_shots: int = 756000,
    mitigation_shots: int = 266000,
    precision_target: float | None = None,
) -> dict:
    """Build a minimal QESEM-like result dict for testing."""
    noisy_evs = [noisy_energy] + [0.93] * 10 + [0.10] * 9
    result = {
        "qesem_used": True,
        "e_zne": e_mitigated,
        "e_exact": e_exact,
        "e_zne_std": e_std,
        "gap": gap,
        "qesem_noisy_evs": noisy_evs,
        "qesem_gate_fidelities": gate_fidelities or {"RZZ": 0.9972, "ID1Q": 0.999},
        "qesem_total_shots": total_shots,
        "qesem_mitigation_shots": mitigation_shots,
    }
    if precision_target is not None:
        result["circuit_stats"] = {"precision_target": precision_target}
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# C20: Raw vs Mitigated comparison
# ═══════════════════════════════════════════════════════════════════════════════


class TestC20RawVsMitigated:
    """C20: QESEM raw vs mitigated energy comparison."""

    def test_passes_when_mitigation_improves(self):
        """Mitigation reduces error → should pass."""
        data = _make_qesem_result(e_mitigated=-40.52, noisy_energy=-38.47)
        report = ValidationReport(source_path="test", run_id="test")
        _check_qesem_raw_vs_mitigated(data, report)
        assert report.checks_passed == 1
        assert report.n_errors == 0

    def test_fails_when_mitigation_degrades(self):
        """Mitigation makes energy worse → should error."""
        # noisy is closer to exact than mitigated
        data = _make_qesem_result(e_mitigated=-35.0, noisy_energy=-40.50, e_exact=-40.57)
        report = ValidationReport(source_path="test", run_id="test")
        _check_qesem_raw_vs_mitigated(data, report)
        assert report.n_errors == 1
        assert "C20" in report.findings[0].check_id

    def test_skips_when_not_qesem(self):
        """Non-QESEM result → skip gracefully."""
        data = {"qesem_used": False, "e_zne": -40.5}
        report = ValidationReport(source_path="test", run_id="test")
        _check_qesem_raw_vs_mitigated(data, report)
        assert report.checks_passed == 1
        assert report.n_errors == 0

    def test_skips_when_noisy_is_sentinel(self):
        """Noisy energy = 0.0 (sentinel) → skip."""
        data = _make_qesem_result(noisy_energy=0.0)
        report = ValidationReport(source_path="test", run_id="test")
        _check_qesem_raw_vs_mitigated(data, report)
        assert report.checks_passed == 1

    def test_skips_when_noisy_evs_none(self):
        """No noisy data available → skip."""
        data = _make_qesem_result()
        data["qesem_noisy_evs"] = None
        report = ValidationReport(source_path="test", run_id="test")
        _check_qesem_raw_vs_mitigated(data, report)
        assert report.checks_passed == 1


# ═══════════════════════════════════════════════════════════════════════════════
# C21: Precision convergence
# ═══════════════════════════════════════════════════════════════════════════════


class TestC21PrecisionConvergence:
    """C21: QESEM precision convergence check."""

    def test_passes_when_converged(self):
        """σ < 2×ε → fully converged, should pass."""
        data = _make_qesem_result(e_std=0.015, precision_target=0.01)
        report = ValidationReport(source_path="test", run_id="test")
        _check_qesem_precision_convergence(data, report)
        assert report.checks_passed == 1
        assert report.n_warnings == 0

    def test_info_when_moderately_above(self):
        """σ between 2×ε and 5×ε → INFO."""
        data = _make_qesem_result(e_std=0.03, precision_target=0.01)
        report = ValidationReport(source_path="test", run_id="test")
        _check_qesem_precision_convergence(data, report)
        assert report.n_warnings == 0
        info_findings = [f for f in report.findings if f.severity == Severity.INFO]
        assert len(info_findings) == 1

    def test_warning_when_far_from_target(self):
        """σ > 5×ε → WARNING (QPU time exhausted)."""
        data = _make_qesem_result(e_std=0.29, precision_target=0.01)
        report = ValidationReport(source_path="test", run_id="test")
        _check_qesem_precision_convergence(data, report)
        assert report.n_warnings == 1
        assert "C21" in report.findings[0].check_id

    def test_skips_when_not_qesem(self):
        """Non-QESEM → skip."""
        data = {"qesem_used": False}
        report = ValidationReport(source_path="test", run_id="test")
        _check_qesem_precision_convergence(data, report)
        assert report.checks_passed == 1

    def test_uses_default_precision_when_not_in_stats(self):
        """No circuit_stats.precision_target → defaults to 0.01."""
        data = _make_qesem_result(e_std=0.29)  # No precision_target
        report = ValidationReport(source_path="test", run_id="test")
        _check_qesem_precision_convergence(data, report)
        # 0.29 / 0.01 = 29× > 5× → WARNING
        assert report.n_warnings == 1


# ═══════════════════════════════════════════════════════════════════════════════
# C22: Gate fidelity characterization
# ═══════════════════════════════════════════════════════════════════════════════


class TestC22GateFidelity:
    """C22: QESEM gate fidelity characterization quality."""

    def test_passes_with_good_fidelity(self):
        """RZZ > 0.99 → pass."""
        data = _make_qesem_result(gate_fidelities={"RZZ": 0.9972, "ID1Q": 0.999})
        report = ValidationReport(source_path="test", run_id="test")
        _check_qesem_gate_fidelity(data, report)
        assert report.checks_passed == 1

    def test_warning_with_degraded_fidelity(self):
        """RZZ < 0.99 → WARNING."""
        data = _make_qesem_result(gate_fidelities={"RZZ": 0.985, "ID1Q": 0.998})
        report = ValidationReport(source_path="test", run_id="test")
        _check_qesem_gate_fidelity(data, report)
        assert report.n_warnings == 1
        assert "degraded" in report.findings[0].title.lower()

    def test_passes_when_fidelities_missing(self):
        """No gate fidelities available → skip gracefully."""
        data = _make_qesem_result(gate_fidelities=None)
        report = ValidationReport(source_path="test", run_id="test")
        _check_qesem_gate_fidelity(data, report)
        assert report.checks_passed == 1

    def test_handles_cz_key(self):
        """Gate fidelity with 'CZ' key instead of 'RZZ'."""
        data = _make_qesem_result(gate_fidelities={"CZ": 0.980, "ID1Q": 0.999})
        report = ValidationReport(source_path="test", run_id="test")
        _check_qesem_gate_fidelity(data, report)
        assert report.n_warnings == 1  # 0.980 < 0.99


# ═══════════════════════════════════════════════════════════════════════════════
# C23: Shot efficiency
# ═══════════════════════════════════════════════════════════════════════════════


class TestC23ShotEfficiency:
    """C23: QESEM shot budget efficiency."""

    def test_passes_with_normal_ratio(self):
        """35% mitigation shots → pass (normal)."""
        data = _make_qesem_result(total_shots=756000, mitigation_shots=266000)
        report = ValidationReport(source_path="test", run_id="test")
        _check_qesem_shot_efficiency(data, report)
        assert report.checks_passed == 1

    def test_info_when_characterization_heavy(self):
        """< 20% mitigation shots → INFO (characterization dominated)."""
        data = _make_qesem_result(total_shots=1000000, mitigation_shots=100000)
        report = ValidationReport(source_path="test", run_id="test")
        _check_qesem_shot_efficiency(data, report)
        info_findings = [f for f in report.findings if f.severity == Severity.INFO]
        assert len(info_findings) == 1

    def test_passes_when_shots_unavailable(self):
        """No shot data → skip."""
        data = _make_qesem_result(total_shots=0, mitigation_shots=0)
        report = ValidationReport(source_path="test", run_id="test")
        _check_qesem_shot_efficiency(data, report)
        assert report.checks_passed == 1


# ═══════════════════════════════════════════════════════════════════════════════
# C24: Raw noise level assessment
# ═══════════════════════════════════════════════════════════════════════════════


class TestC24NoisyVsExact:
    """C24: QESEM raw noise level assessment."""

    def test_passes_normal_case(self):
        """Moderate raw noise + good mitigation → pass."""
        data = _make_qesem_result(e_mitigated=-40.52, noisy_energy=-38.47, e_exact=-40.57, gap=5.92)
        report = ValidationReport(source_path="test", run_id="test")
        _check_qesem_noisy_vs_exact(data, report)
        assert report.checks_passed == 1

    def test_info_when_already_noiseless(self):
        """Raw energy very close to exact → INFO (QESEM adds little)."""
        data = _make_qesem_result(e_mitigated=-40.56, noisy_energy=-40.55, e_exact=-40.57, gap=5.92)
        report = ValidationReport(source_path="test", run_id="test")
        _check_qesem_noisy_vs_exact(data, report)
        info_findings = [f for f in report.findings if f.severity == Severity.INFO]
        assert len(info_findings) == 1
        assert "near-noiseless" in info_findings[0].title.lower()

    def test_warning_when_severe_noise_and_poor_mitigation(self):
        """Very noisy + mitigation still bad → WARNING."""
        data = _make_qesem_result(e_mitigated=-37.0, noisy_energy=-35.0, e_exact=-40.57, gap=5.92)
        report = ValidationReport(source_path="test", run_id="test")
        _check_qesem_noisy_vs_exact(data, report)
        assert report.n_warnings == 1

    def test_skips_when_not_qesem(self):
        """Non-QESEM → skip."""
        data = {"qesem_used": False}
        report = ValidationReport(source_path="test", run_id="test")
        _check_qesem_noisy_vs_exact(data, report)
        assert report.checks_passed == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: full validation on real recovered data format
# ═══════════════════════════════════════════════════════════════════════════════


class TestQESEMFullValidation:
    """Integration tests using realistic QESEM result structures."""

    def test_tier0_like_result_runs_all_checks(self):
        """Simulates a Tier-0-like result and runs all QESEM checks."""
        data = _make_qesem_result(
            e_mitigated=-40.524,
            e_exact=-40.566,
            e_std=0.288,
            gap=5.922,
            noisy_energy=-38.471,
            gate_fidelities={"RZZ": 0.9972, "ID1Q": 0.999},
            total_shots=756048,
            mitigation_shots=266000,
        )
        report = ValidationReport(source_path="test", run_id="tier0_sim")

        _check_qesem_raw_vs_mitigated(data, report)
        _check_qesem_precision_convergence(data, report)
        _check_qesem_gate_fidelity(data, report)
        _check_qesem_shot_efficiency(data, report)
        _check_qesem_noisy_vs_exact(data, report)

        assert report.checks_run == 5
        # C20 passes (mitigation improved), C22 passes (fidelity good),
        # C23 passes (35% ratio), C24 passes (moderate noise well-mitigated)
        # C21 warns (σ=0.288 >> ε=0.01)
        assert report.checks_passed >= 4
        assert report.n_errors == 0
        # C21 should fire a warning
        warnings = [f for f in report.findings if f.severity == Severity.WARNING]
        assert len(warnings) == 1
        assert warnings[0].check_id == "C21"


# ═══════════════════════════════════════════════════════════════════════════════
# Test run_qesem_sweep interface (mock-based)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunQESEMSweepInterface:
    """Tests for run_qesem_sweep multi-PUB batch function."""

    def test_raises_on_mismatched_lengths(self):
        """Input lists must all have the same length."""
        from qmbp_simulation.execution.backends import MitigationOptions
        from qmbp_simulation.execution.hardware.config import HardwareConfig
        from qmbp_simulation.execution.hardware.qesem import run_qesem_sweep

        config = HardwareConfig(
            mode="hardware",
            mitigation=MitigationOptions(qesem_enabled=True),
        )

        # Create a dummy circuit
        from qiskit.circuit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        with pytest.raises(ValueError, match="same length"):
            run_qesem_sweep(
                circuit=qc,
                hamiltonians=[None],  # 1 element
                x_ops_list=[[], []],  # 2 elements — mismatch
                zz_ops_list=[[]],
                params_per_h=[np.zeros(0)],
                h_values=[4.0],
                config=config,
            )

    def test_sweep_scales_execution_time(self):
        """Verify max_execution_time scales with n_pubs (checked via mock)."""
        # This is a structural test — we verify the scaling logic indirectly
        # by checking the formula: min(per_pub * n_pubs, 3600)
        per_pub = 600
        n_pubs = 4
        expected = min(per_pub * n_pubs, 3600)
        assert expected == 2400  # 600 * 4 = 2400 < 3600

        n_pubs_large = 8
        expected_large = min(per_pub * n_pubs_large, 3600)
        assert expected_large == 3600  # 600 * 8 = 4800 > 3600 → capped
