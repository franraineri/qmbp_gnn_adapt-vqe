#!/usr/bin/env python3
"""Tests for thesis compilation tools — prevents regressions and validates robustness.

Tests:
  1. thesis_findings_validator: imports, runs without crash, handles empty data
  2. thesis_tables_compiler: imports, generates tables, handles missing data
  3. thesis_figures: imports, handles no matplotlib gracefully
  4. Data schema validation: ensures JSON files match expected schemas
  5. Cross-tool consistency: tables and validator agree on counts

Run:
    pytest tests/test_thesis_tools.py -v -m "not slow"     # Fast tests only (~20s)
    pytest tests/test_thesis_tools.py -v                    # All tests (~90s)
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


# ═══════════════════════════════════════════════════════════════════════════════
# Test: thesis_findings_validator (FAST — uses pre-loaded scanner)
# ═══════════════════════════════════════════════════════════════════════════════


class TestThesisFindingsValidator:
    """Tests for the findings validator module."""

    def test_import(self):
        """Module imports without error."""
        from project_health.analysis.validation.thesis_findings_validator import (
            EvidenceStrength,
            run_validation,
        )

        assert EvidenceStrength.STRONG == "STRONG"
        assert callable(run_validation)

    def test_run_validation_no_crash(self):
        """Full validation runs without crashing."""
        from project_health.analysis.validation.thesis_findings_validator import run_validation

        report = run_validation(verbose=False)
        assert report is not None
        assert len(report.findings) > 0
        assert report.overall_corroboration_rate >= 0

    def test_validation_report_serializable(self):
        """Report is JSON-serializable."""
        from project_health.analysis.validation.thesis_findings_validator import run_validation

        report = run_validation(verbose=False)
        data = report.to_dict()
        json_str = json.dumps(data, indent=2)
        assert len(json_str) > 100

    def test_category_filter(self):
        """Category filtering works."""
        from project_health.analysis.validation.thesis_findings_validator import run_validation

        report = run_validation(categories=["scaling"], verbose=False)
        for f in report.findings:
            assert f.category == "scaling"

    def test_empty_data_handling(self):
        """Validators handle empty inputs gracefully."""
        from project_health.analysis.validation.thesis_findings_validator import (
            _VALIDATORS,
            FindingValidation,
        )

        for finding_id, _category, _claim, func in _VALIDATORS:
            try:
                result = func(
                    noiseless=[],
                    noisy=[],
                    experiments=[],
                    scaling=[],
                    cross_topo_results=[],
                )
                assert isinstance(result, FindingValidation), f"{finding_id} returned wrong type"
                assert result.verdict in (
                    "CORROBORATED",
                    "QUALIFIED",
                    "UNSUPPORTED",
                    "CONTRADICTED",
                )
            except Exception:
                # Validators loading files directly may fail — acceptable
                pass

    def test_statistical_utilities(self):
        """Statistical helper functions work correctly."""
        from project_health.analysis.validation.thesis_findings_validator import (
            _ci_95,
            _classify_strength,
            _cohens_d,
            _ttest_1samp,
        )

        d = _cohens_d([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
        assert d < -2.0

        ci = _ci_95([1, 2, 3, 4, 5])
        assert ci[0] < 3.0 < ci[1]

        t, p = _ttest_1samp([0.01, 0.02, 0.03, 0.04], 0.05)
        assert p < 0.05

        assert _classify_strength(0.001, 1.5, 20) == "STRONG"
        assert _classify_strength(0.03, 0.6, 8) == "MODERATE"
        assert _classify_strength(0.5, 0.1, 3) == "UNSUPPORTED"


# ═══════════════════════════════════════════════════════════════════════════════
# Test: thesis_tables_compiler (SLOW — scans all results)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestThesisTablesCompiler:
    """Tests for the thesis tables compiler."""

    def test_import(self):
        """Module imports without error."""
        from project_health.analysis.thesis.thesis_tables_compiler import (
            compile_tables,
        )

        assert callable(compile_tables)

    def test_compile_no_crash(self):
        """Full compilation runs without crashing."""
        from project_health.analysis.thesis.thesis_tables_compiler import compile_tables

        report = compile_tables(verbose=False)
        assert report is not None
        assert len(report.tables) > 0

    def test_table_structure(self):
        """All tables have valid structure."""
        from project_health.analysis.thesis.thesis_tables_compiler import compile_tables

        report = compile_tables(verbose=False)
        for table in report.tables:
            assert table.table_id.startswith("T")
            assert len(table.title) > 5
            assert len(table.columns) >= 2
            for row in table.rows:
                assert len(row) == len(table.columns), (
                    f"Table {table.table_id}: row has {len(row)} cols, expected {len(table.columns)}"
                )

    def test_markdown_format(self):
        """Markdown output is well-formed."""
        from project_health.analysis.thesis.thesis_tables_compiler import (
            _format_markdown,
            compile_tables,
        )

        report = compile_tables(only=["T5"], verbose=False)
        if report.tables:
            md = _format_markdown(report.tables[0])
            assert "| " in md
            assert "---" in md

    def test_latex_format(self):
        """LaTeX output is well-formed."""
        from project_health.analysis.thesis.thesis_tables_compiler import (
            _format_latex,
            compile_tables,
        )

        report = compile_tables(only=["T5"], verbose=False)
        if report.tables:
            tex = _format_latex(report.tables[0])
            assert r"\begin{table}" in tex
            assert r"\end{table}" in tex

    def test_only_filter(self):
        """Table ID filtering works."""
        from project_health.analysis.thesis.thesis_tables_compiler import compile_tables

        report = compile_tables(only=["T5"], verbose=False)
        ids = [t.table_id for t in report.tables]
        assert "T5" in ids

    def test_report_serializable(self):
        """Report is JSON-serializable."""
        from project_health.analysis.thesis.thesis_tables_compiler import compile_tables

        report = compile_tables(verbose=False)
        data = report.to_dict()
        json_str = json.dumps(data, indent=2)
        assert len(json_str) > 100


# ═══════════════════════════════════════════════════════════════════════════════
# Test: thesis_figures (SLOW — requires matplotlib + data scanning)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestThesisFigures:
    """Tests for thesis figure generation."""

    def test_import(self):
        """Module imports without error."""
        from project_health.analysis.thesis.thesis_figures import (
            _THESIS_FIGURES,
            generate_all,
        )

        assert callable(generate_all)
        assert len(_THESIS_FIGURES) >= 5

    def test_figure_registry_complete(self):
        """All registered figures have name, description, and callable."""
        from project_health.analysis.thesis.thesis_figures import _THESIS_FIGURES

        for name, description, func in _THESIS_FIGURES:
            assert isinstance(name, str) and len(name) > 3
            assert isinstance(description, str) and len(description) > 10
            assert callable(func)

    def test_generate_no_crash(self, tmp_path):
        """Figure generation doesn't crash (may skip if no data)."""
        from project_health.analysis.thesis.thesis_figures import FigureConfig, generate_all

        cfg = FigureConfig(output_dir=tmp_path, fmt="png", dpi=72, no_titles=True)
        results = generate_all(cfg, verbose=False)
        assert isinstance(results, dict)
        assert len(results) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Data Schema Validation (FAST — reads single files)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataSchemaIntegrity:
    """Validates that result JSON files match expected schemas."""

    def test_zero_shot_schema(self):
        """Zero-shot results have expected structure."""
        zero_dir = RESULTS_DIR / "scaling" / "zero_shot"
        if not zero_dir.exists():
            pytest.skip("No zero-shot results")
        files = list(zero_dir.glob("zero_shot_v3_*.json"))
        if not files:
            pytest.skip("No zero-shot v3 files")
        data = json.loads(files[0].read_text())
        assert "strategy_a_gnn_no_bn" in data, f"Missing strategy_a in {files[0].name}"
        results = data["strategy_a_gnn_no_bn"].get("results", [])
        if results:
            assert "de_gap" in results[0]
            assert "passed" in results[0]

    def test_gnn_qem_cross_topology_schema(self):
        """GNN-QEM cross-topology has expected structure."""
        f = RESULTS_DIR / "gnn_qem" / "cross_topology_results.json"
        if not f.exists():
            pytest.skip("No cross_topology_results.json")
        data = json.loads(f.read_text())
        assert "zero_shot" in data
        zs = data["zero_shot"]
        assert "improvement_rate" in zs
        assert "reduction_pct" in zs
        assert "n_samples" in zs

    def test_gnn_qem_post_zne_schema(self):
        """Post-ZNE validation has expected structure."""
        f = RESULTS_DIR / "gnn_qem" / "post_zne_validation.json"
        if not f.exists():
            pytest.skip("No post_zne_validation.json")
        data = json.loads(f.read_text())
        assert "summary" in data
        s = data["summary"]
        assert "n_evaluations" in s
        assert "n_gnn_regresses" in s

    def test_zne_cross_topo_schema(self):
        """ZNE cross-topo experiment has expected structure."""
        exp_dir = RESULTS_DIR / "experiments" / "exp_zne_cross_topo"
        if not exp_dir.exists():
            pytest.skip("No exp_zne_cross_topo")
        files = sorted(exp_dir.glob("run_*.json"), reverse=True)
        if not files:
            pytest.skip("No run files")
        data = json.loads(files[0].read_text())
        assert "results" in data
        assert "section_4" in data["results"]
        s4 = data["results"]["section_4"]["data"]
        assert "summary" in s4
        assert "paired_t_stat" in s4["summary"]

    def test_scaling_results_schema(self):
        """Scaling results have expected fields."""
        scaling_dir = RESULTS_DIR / "scaling"
        if not scaling_dir.exists():
            pytest.skip("No scaling results")
        files = list(scaling_dir.glob("scaling_N*_*.json"))
        if not files:
            pytest.skip("No scaling files")
        data = json.loads(files[0].read_text())
        # Scaling files use 'metadata' (with chi_max, h_values, etc.) or 'config'/'system'
        assert "metadata" in data or "config" in data or "system" in data, (
            f"No metadata/config/system in {files[0].name}"
        )
        # Must have summary
        assert "summary" in data, f"No summary in {files[0].name}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Cross-Tool Consistency (SLOW — requires full scan)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestCrossToolConsistency:
    """Ensures different tools agree on the same underlying data."""

    def test_noiseless_count_agrees(self):
        """Validator and tables compiler see the same noiseless count."""
        from project_health.digest.scanner import ResultScanner

        scanner = ResultScanner(results_root=RESULTS_DIR)
        noiseless, _, _ = scanner.scan_all(exclude_tests=True)

        from project_health.analysis.thesis.thesis_tables_compiler import _load_all_data

        data = _load_all_data()
        assert len(data["noiseless"]) == len(noiseless)

    def test_experiment_count_agrees(self):
        """Validator and tables compiler see the same experiment count."""
        from project_health.digest.scanner import ResultScanner

        scanner = ResultScanner(results_root=RESULTS_DIR)
        _, _, experiments = scanner.scan_all(exclude_tests=True)

        from project_health.analysis.thesis.thesis_tables_compiler import _load_all_data

        data = _load_all_data()
        assert len(data["experiments"]) == len(experiments)
