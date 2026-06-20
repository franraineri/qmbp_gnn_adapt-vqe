"""Tests for project_health/analysis/diagnose.py — automated failure diagnosis tool.

Covers:
- Data models (RootCause, DeploymentPoint, Diagnosis)
- Root cause classification logic
- Pipeline run parsing
- Severity determination
- Report filtering

Run with:
    pytest tests/test_diagnose.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

from project_health.analysis.diagnose import (
    MARGINAL_THRESHOLD,
    PASS_THRESHOLD,
    DeploymentPoint,
    Diagnosis,
    RootCause,
    classify_root_causes,
    parse_pipeline_run,
    report_diagnoses,
    scan_folder,
)

# ═══════════════════════════════════════════════════════════════════════════
# DeploymentPoint tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDeploymentPoint:
    """Test DeploymentPoint verdict computation."""

    def test_verdict_pass(self):
        dp = DeploymentPoint(h_test=2.0, de_gap=0.03)
        assert dp.verdict == "PASS"

    def test_verdict_marginal(self):
        dp = DeploymentPoint(h_test=2.0, de_gap=0.07)
        assert dp.verdict == "MARGINAL"

    def test_verdict_fail(self):
        dp = DeploymentPoint(h_test=2.0, de_gap=0.15)
        assert dp.verdict == "FAIL"

    def test_verdict_no_data(self):
        dp = DeploymentPoint(h_test=2.0, de_gap=None)
        assert dp.verdict == "NO_DATA"

    def test_verdict_at_exact_threshold_is_marginal(self):
        """ΔE/gap exactly at PASS_THRESHOLD is not a pass (uses <, not <=)."""
        dp = DeploymentPoint(h_test=2.0, de_gap=PASS_THRESHOLD)
        assert dp.verdict == "MARGINAL"

    def test_verdict_at_exact_marginal_threshold_is_fail(self):
        dp = DeploymentPoint(h_test=2.0, de_gap=MARGINAL_THRESHOLD)
        assert dp.verdict == "FAIL"


# ═══════════════════════════════════════════════════════════════════════════
# Diagnosis data model tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDiagnosis:
    """Test Diagnosis dataclass properties."""

    def _make_diag(self, **kwargs) -> Diagnosis:
        defaults = dict(
            folder="test_folder",
            variant="test_variant",
            file="test.json",
            topology="chain_1d",
            n_qubits=10,
            p_layers=1,
            seed=42,
            n_restarts=5,
        )
        defaults.update(kwargs)
        return Diagnosis(**defaults)

    def test_worst_de_gap_returns_max(self):
        diag = self._make_diag(
            deployment_points=[
                DeploymentPoint(h_test=2.0, de_gap=0.02),
                DeploymentPoint(h_test=2.5, de_gap=0.08),
                DeploymentPoint(h_test=3.0, de_gap=0.04),
            ]
        )
        assert diag.worst_de_gap == 0.08

    def test_worst_de_gap_none_when_no_points(self):
        diag = self._make_diag(deployment_points=[])
        assert diag.worst_de_gap is None

    def test_worst_de_gap_ignores_none_gaps(self):
        diag = self._make_diag(
            deployment_points=[
                DeploymentPoint(h_test=2.0, de_gap=None),
                DeploymentPoint(h_test=2.5, de_gap=0.03),
            ]
        )
        assert diag.worst_de_gap == 0.03

    def test_valid_regime_boundary_p1_chain(self):
        diag = self._make_diag(topology="chain_1d", n_qubits=10, p_layers=1)
        assert diag.valid_regime_boundary == 1.9

    def test_valid_regime_boundary_p2_chain(self):
        diag = self._make_diag(topology="chain_1d", n_qubits=10, p_layers=2)
        assert diag.valid_regime_boundary == 1.5

    def test_valid_regime_boundary_unknown_config(self):
        diag = self._make_diag(topology="unknown_topo", n_qubits=99, p_layers=1)
        assert diag.valid_regime_boundary == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# classify_root_causes tests
# ═══════════════════════════════════════════════════════════════════════════


class TestClassifyRootCauses:
    """Test the root cause classification engine."""

    def _make_diag(self, **kwargs) -> Diagnosis:
        defaults = dict(
            folder="test_folder",
            variant="test_variant",
            file="test.json",
            topology="chain_1d",
            n_qubits=10,
            p_layers=1,
            seed=42,
            n_restarts=5,
        )
        defaults.update(kwargs)
        return Diagnosis(**defaults)

    def test_passing_run_no_causes(self):
        """A run with ΔE/gap < 0.05 should get PASS and no root causes."""
        diag = self._make_diag(deployment_points=[DeploymentPoint(h_test=3.0, de_gap=0.02)])
        classify_root_causes(diag)
        assert diag.severity == "PASS"
        assert diag.root_causes == []

    def test_chain_break_detected(self):
        """θ_smoothness > 1.0 should trigger CHAIN_BREAK."""
        diag = self._make_diag(
            theta_smoothness=1.5,
            deployment_points=[DeploymentPoint(h_test=3.0, de_gap=0.12)],
        )
        classify_root_causes(diag)
        assert RootCause.CHAIN_BREAK in diag.root_causes
        assert diag.severity == "FAIL"

    def test_mpnn_overfit_detected(self):
        """generalization_gap > 0.01 should trigger MPNN_OVERFIT."""
        diag = self._make_diag(
            generalization_gap=0.025,
            deployment_points=[DeploymentPoint(h_test=3.0, de_gap=0.15)],
        )
        classify_root_causes(diag)
        assert RootCause.MPNN_OVERFIT in diag.root_causes

    def test_outside_regime_detected(self):
        """h_test below valid regime boundary should trigger OUTSIDE_REGIME."""
        # chain_1d N=10 p=1 boundary is 1.9
        diag = self._make_diag(
            topology="chain_1d",
            n_qubits=10,
            p_layers=1,
            deployment_points=[DeploymentPoint(h_test=1.5, de_gap=0.20)],
        )
        classify_root_causes(diag)
        assert RootCause.OUTSIDE_REGIME in diag.root_causes

    def test_boundary_effect_detected(self):
        """h_test within 0.5 of boundary (but above) should trigger BOUNDARY_EFFECT."""
        # chain_1d N=10 p=1 boundary is 1.9; h_test=2.1 is 0.2 above
        diag = self._make_diag(
            topology="chain_1d",
            n_qubits=10,
            p_layers=1,
            deployment_points=[DeploymentPoint(h_test=2.1, de_gap=0.08)],
        )
        classify_root_causes(diag)
        assert RootCause.BOUNDARY_EFFECT in diag.root_causes

    def test_vqe_divergence_detected(self):
        """convergence_rate < 1.0 should trigger VQE_DIVERGENCE."""
        diag = self._make_diag(
            convergence_rate=0.8,
            per_h_converged=[True, True, False, True, False],
            deployment_points=[DeploymentPoint(h_test=3.0, de_gap=0.12)],
        )
        classify_root_causes(diag)
        assert RootCause.VQE_DIVERGENCE in diag.root_causes

    def test_hva_limit_detected(self):
        """error_from_circuit > 0.01 should trigger HVA_LIMIT."""
        diag = self._make_diag(
            error_from_circuit=0.05,
            deployment_points=[DeploymentPoint(h_test=3.0, de_gap=0.12)],
        )
        classify_root_causes(diag)
        assert RootCause.HVA_LIMIT in diag.root_causes

    def test_unknown_when_no_pattern_matches(self):
        """If no known pattern matches, should get UNKNOWN."""
        diag = self._make_diag(
            deployment_points=[DeploymentPoint(h_test=5.0, de_gap=0.15)],
        )
        classify_root_causes(diag)
        assert RootCause.UNKNOWN in diag.root_causes

    def test_multiple_causes_detected(self):
        """Multiple failure patterns should all be reported."""
        diag = self._make_diag(
            theta_smoothness=1.5,
            generalization_gap=0.03,
            deployment_points=[DeploymentPoint(h_test=3.0, de_gap=0.20)],
        )
        classify_root_causes(diag)
        assert RootCause.CHAIN_BREAK in diag.root_causes
        assert RootCause.MPNN_OVERFIT in diag.root_causes
        assert len(diag.root_causes) >= 2

    def test_marginal_severity(self):
        """ΔE/gap between 0.05 and 0.10 should get MARGINAL severity."""
        diag = self._make_diag(
            deployment_points=[DeploymentPoint(h_test=5.0, de_gap=0.07)],
        )
        classify_root_causes(diag)
        assert diag.severity == "MARGINAL"

    def test_no_phase4_severity(self):
        """No deployment points → NO_PHASE4 severity."""
        diag = self._make_diag(deployment_points=[])
        classify_root_causes(diag)
        assert diag.severity == "NO_PHASE4"

    def test_outside_regime_takes_priority_over_boundary(self):
        """OUTSIDE_REGIME and BOUNDARY_EFFECT should not coexist."""
        # h_test=1.5 is below boundary 1.9 — only OUTSIDE_REGIME should fire
        diag = self._make_diag(
            topology="chain_1d",
            n_qubits=10,
            p_layers=1,
            deployment_points=[DeploymentPoint(h_test=1.5, de_gap=0.20)],
        )
        classify_root_causes(diag)
        assert RootCause.OUTSIDE_REGIME in diag.root_causes
        assert RootCause.BOUNDARY_EFFECT not in diag.root_causes

    def test_recommendations_populated(self):
        """Recommendations should be provided for each cause."""
        diag = self._make_diag(
            theta_smoothness=2.0,
            deployment_points=[DeploymentPoint(h_test=3.0, de_gap=0.15)],
        )
        classify_root_causes(diag)
        assert len(diag.recommendations) > 0


# ═══════════════════════════════════════════════════════════════════════════
# parse_pipeline_run tests
# ═══════════════════════════════════════════════════════════════════════════


class TestParsePipelineRun:
    """Test JSON parsing into Diagnosis objects."""

    def _write_pipeline_json(self, tmp_path: Path, data: dict) -> Path:
        path = tmp_path / "pipeline_run_001.json"
        path.write_text(json.dumps(data))
        return path

    def test_parses_minimal_json(self, tmp_path):
        data = {
            "config": {
                "topology": "chain_1d",
                "n_qubits": 6,
                "p_layers": 2,
                "seed": 42,
                "n_restarts": 5,
                "h_values": [2.0, 1.5, 1.0],
            },
            "diagnostics": {
                "phase2": {"theta_smoothness": 0.5, "convergence_rate": 1.0},
                "phase3": {"generalization_gap": 0.005},
            },
            "phase4_results": [
                {"h_test": 1.5, "delta_e_over_gap": 0.02, "phase_label": "ordered"},
            ],
        }
        path = self._write_pipeline_json(tmp_path, data)
        diag = parse_pipeline_run(path, "folder1", "variant1")

        assert diag is not None
        assert diag.topology == "chain_1d"
        assert diag.n_qubits == 6
        assert diag.p_layers == 2
        assert diag.seed == 42
        assert diag.theta_smoothness == 0.5
        assert diag.generalization_gap == 0.005
        assert len(diag.deployment_points) == 1
        assert diag.deployment_points[0].de_gap == 0.02

    def test_returns_none_for_invalid_json(self, tmp_path):
        path = tmp_path / "pipeline_run_bad.json"
        path.write_text("not valid json {{{")
        diag = parse_pipeline_run(path, "folder", "variant")
        assert diag is None

    def test_returns_none_for_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        diag = parse_pipeline_run(path, "folder", "variant")
        assert diag is None

    def test_handles_empty_diagnostics(self, tmp_path):
        data = {
            "config": {"topology": "ladder", "n_qubits": 10, "p_layers": 1},
            "diagnostics": {},
            "phase4_results": [],
        }
        path = self._write_pipeline_json(tmp_path, data)
        diag = parse_pipeline_run(path, "folder1", "variant1")
        assert diag is not None
        assert diag.theta_smoothness is None
        assert diag.generalization_gap is None

    def test_multiple_deployment_points(self, tmp_path):
        data = {
            "config": {"topology": "chain_1d", "n_qubits": 10, "p_layers": 2},
            "diagnostics": {},
            "phase4_results": [
                {"h_test": 2.0, "delta_e_over_gap": 0.01},
                {"h_test": 2.5, "delta_e_over_gap": 0.03},
                {"h_test": 3.0, "delta_e_over_gap": 0.09},
            ],
        }
        path = self._write_pipeline_json(tmp_path, data)
        diag = parse_pipeline_run(path, "folder1", "variant1")
        assert len(diag.deployment_points) == 3
        assert diag.worst_de_gap == 0.09

    def test_energy_decomposition_parsed(self, tmp_path):
        data = {
            "config": {"topology": "chain_1d", "n_qubits": 6, "p_layers": 1},
            "diagnostics": {
                "phase4": {
                    "energy_decomposition": {
                        "error_from_circuit": 0.02,
                        "error_from_mpnn": 0.01,
                    }
                }
            },
            "phase4_results": [{"h_test": 2.0, "delta_e_over_gap": 0.05}],
        }
        path = self._write_pipeline_json(tmp_path, data)
        diag = parse_pipeline_run(path, "folder1", "variant1")
        assert diag.error_from_circuit == 0.02
        assert diag.error_from_mpnn == 0.01


# ═══════════════════════════════════════════════════════════════════════════
# scan_folder tests
# ═══════════════════════════════════════════════════════════════════════════


class TestScanFolder:
    """Test folder scanning discovers pipeline_run files."""

    def test_discovers_direct_pipeline_files(self, tmp_path):
        """Finds pipeline_run_*.json directly in folder."""
        data = {
            "config": {"topology": "chain_1d", "n_qubits": 6, "p_layers": 1},
            "diagnostics": {},
            "phase4_results": [{"h_test": 2.0, "delta_e_over_gap": 0.03}],
        }
        (tmp_path / "pipeline_run_001.json").write_text(json.dumps(data))
        (tmp_path / "pipeline_run_002.json").write_text(json.dumps(data))

        diagnoses = scan_folder(tmp_path)
        assert len(diagnoses) == 2

    def test_discovers_nested_subfolders(self, tmp_path):
        """Finds pipeline_run files in variant subdirectories."""
        data = {
            "config": {"topology": "ladder", "n_qubits": 10, "p_layers": 2},
            "diagnostics": {},
            "phase4_results": [{"h_test": 3.0, "delta_e_over_gap": 0.04}],
        }
        variant_dir = tmp_path / "comp1_ladder_p2_seed42"
        variant_dir.mkdir()
        (variant_dir / "pipeline_run_001.json").write_text(json.dumps(data))

        diagnoses = scan_folder(tmp_path)
        assert len(diagnoses) == 1
        assert diagnoses[0].topology == "ladder"

    def test_takes_latest_from_subfolder(self, tmp_path):
        """When multiple pipeline files exist in subfolder, takes the latest."""
        data_old = {
            "config": {"topology": "chain_1d", "n_qubits": 6, "p_layers": 1},
            "diagnostics": {},
            "phase4_results": [{"h_test": 2.0, "delta_e_over_gap": 0.10}],
        }
        data_new = {
            "config": {"topology": "chain_1d", "n_qubits": 6, "p_layers": 1},
            "diagnostics": {},
            "phase4_results": [{"h_test": 2.0, "delta_e_over_gap": 0.02}],
        }
        variant_dir = tmp_path / "comp1_chain"
        variant_dir.mkdir()
        (variant_dir / "pipeline_run_001.json").write_text(json.dumps(data_old))
        (variant_dir / "pipeline_run_002.json").write_text(json.dumps(data_new))

        diagnoses = scan_folder(tmp_path)
        assert len(diagnoses) == 1
        # Sorted reverse → 002 is latest
        assert diagnoses[0].worst_de_gap == 0.02

    def test_skips_hidden_and_checkpoints(self, tmp_path):
        """Hidden dirs and 'checkpoints' are skipped."""
        data = {
            "config": {"topology": "chain_1d", "n_qubits": 6, "p_layers": 1},
            "diagnostics": {},
            "phase4_results": [{"h_test": 2.0, "delta_e_over_gap": 0.03}],
        }
        hidden = tmp_path / ".hidden_dir"
        hidden.mkdir()
        (hidden / "pipeline_run_001.json").write_text(json.dumps(data))

        ckpt = tmp_path / "checkpoints"
        ckpt.mkdir()
        (ckpt / "pipeline_run_001.json").write_text(json.dumps(data))

        diagnoses = scan_folder(tmp_path)
        assert len(diagnoses) == 0

    def test_empty_folder_returns_empty(self, tmp_path):
        diagnoses = scan_folder(tmp_path)
        assert diagnoses == []

    def test_classifies_after_parsing(self, tmp_path):
        """Scanned diagnoses should have root_causes populated."""
        data = {
            "config": {"topology": "chain_1d", "n_qubits": 10, "p_layers": 1},
            "diagnostics": {"phase2": {"theta_smoothness": 2.0}},
            "phase4_results": [{"h_test": 3.0, "delta_e_over_gap": 0.15}],
        }
        (tmp_path / "pipeline_run_001.json").write_text(json.dumps(data))
        diagnoses = scan_folder(tmp_path)
        assert len(diagnoses) == 1
        assert len(diagnoses[0].root_causes) > 0


# ═══════════════════════════════════════════════════════════════════════════
# report_diagnoses tests
# ═══════════════════════════════════════════════════════════════════════════


class TestReportDiagnoses:
    """Test report filtering logic."""

    def _make_diagnosed(self, severity: str, cause: RootCause) -> Diagnosis:
        diag = Diagnosis(
            folder="f",
            variant="v",
            file="f.json",
            topology="chain_1d",
            n_qubits=10,
            p_layers=1,
            seed=42,
            n_restarts=5,
        )
        diag.severity = severity
        diag.root_causes = [cause]
        diag.explanation = "test"
        return diag

    def test_passing_runs_excluded_by_default(self, capsys):
        diagnoses = [
            self._make_diagnosed("PASS", RootCause.UNKNOWN),
            self._make_diagnosed("FAIL", RootCause.CHAIN_BREAK),
        ]
        report_diagnoses(diagnoses)
        captured = capsys.readouterr()
        assert "CHAIN_BREAK" in captured.out
        # PASS runs should not appear
        assert captured.out.count("[PASS]") == 0

    def test_severity_filter(self, capsys):
        diagnoses = [
            self._make_diagnosed("FAIL", RootCause.CHAIN_BREAK),
            self._make_diagnosed("MARGINAL", RootCause.BOUNDARY_EFFECT),
        ]
        report_diagnoses(diagnoses, severity_filter="FAIL")
        captured = capsys.readouterr()
        assert "CHAIN_BREAK" in captured.out
        assert "BOUNDARY_EFFECT" not in captured.out

    def test_show_passing_includes_pass(self, capsys):
        diagnoses = [self._make_diagnosed("PASS", RootCause.UNKNOWN)]
        # When no failures exist and show_passing is True
        report_diagnoses(diagnoses, show_passing=True)
        captured = capsys.readouterr()
        # Should show the passing run
        assert "[PASS]" in captured.out

    def test_empty_diagnoses_shows_success(self, capsys):
        report_diagnoses([])
        captured = capsys.readouterr()
        assert "No failures" in captured.out
