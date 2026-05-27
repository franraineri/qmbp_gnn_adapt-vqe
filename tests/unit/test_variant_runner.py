"""Unit tests for qmbp_simulation.framework.variant_runner module.

Tests the shared variant runner infrastructure used by all topology-specific
variant scripts (chain_1d, ladder, triangular, kagome).
"""

from __future__ import annotations

import json

import pytest

from qmbp_simulation.framework.variant_runner import (
    PipelineVariant,
    RunResult,
    VariantRunner,
    extract_metrics_from_output,
    run_variant,
)

# ═══════════════════════════════════════════════════════════════════════════
# PipelineVariant dataclass tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineVariant:
    """Test PipelineVariant dataclass construction and fields."""

    def test_construction_with_all_fields(self):
        v = PipelineVariant(
            id="NL-A1",
            description="VQE restarts=1",
            category="noiseless",
            command=["python", "run.py", "--n-qubits", "6"],
            hypothesis="1 restart converges",
            expected_outcome="PASS",
            output_dir="results/test",
        )
        assert v.id == "NL-A1"
        assert v.category == "noiseless"
        assert v.command == ["python", "run.py", "--n-qubits", "6"]
        assert v.output_dir == "results/test"

    def test_command_is_list_of_strings(self):
        v = PipelineVariant(
            id="X",
            description="",
            category="noiseless",
            command=["a", "b", "c"],
            hypothesis="",
            expected_outcome="",
            output_dir="/tmp",
        )
        assert all(isinstance(c, str) for c in v.command)


# ═══════════════════════════════════════════════════════════════════════════
# RunResult verdict logic tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRunResultVerdict:
    """Test RunResult.verdict property for all possible states."""

    def test_verdict_pass(self):
        r = RunResult(
            variant_id="X",
            success=True,
            elapsed_s=1.0,
            return_code=0,
            delta_e_over_gap=0.03,
        )
        assert r.verdict == "PASS"

    def test_verdict_marginal(self):
        r = RunResult(
            variant_id="X",
            success=True,
            elapsed_s=1.0,
            return_code=0,
            delta_e_over_gap=0.07,
        )
        assert r.verdict == "MARGINAL"

    def test_verdict_fail(self):
        r = RunResult(
            variant_id="X",
            success=True,
            elapsed_s=1.0,
            return_code=0,
            delta_e_over_gap=0.15,
        )
        assert r.verdict == "FAIL"

    def test_verdict_ok_when_no_metric(self):
        r = RunResult(
            variant_id="X",
            success=True,
            elapsed_s=1.0,
            return_code=0,
        )
        assert r.verdict == "OK"

    def test_verdict_error_on_failure(self):
        r = RunResult(
            variant_id="X",
            success=False,
            elapsed_s=1.0,
            return_code=1,
            error_msg="crash",
        )
        assert r.verdict == "ERROR"

    def test_verdict_skip_p3(self):
        r = RunResult(
            variant_id="X",
            success=False,
            elapsed_s=1.0,
            return_code=1,
            phase3_failed=True,
        )
        assert r.verdict == "SKIP-P3"

    def test_verdict_zne_pass(self):
        r = RunResult(
            variant_id="X",
            success=True,
            elapsed_s=1.0,
            return_code=0,
            noisy_summary={"success": True, "mean_r2": 0.99},
        )
        assert r.verdict == "ZNE-PASS"

    def test_verdict_zne_fail(self):
        r = RunResult(
            variant_id="X",
            success=True,
            elapsed_s=1.0,
            return_code=0,
            noisy_summary={"success": False, "mean_r2": 0.3},
        )
        assert r.verdict == "ZNE-FAIL"

    @pytest.mark.parametrize(
        "de_gap,expected",
        [
            (0.0, "PASS"),
            (0.049, "PASS"),
            (0.05, "MARGINAL"),
            (0.099, "MARGINAL"),
            (0.10, "FAIL"),
            (1.0, "FAIL"),
        ],
    )
    def test_verdict_boundary_values(self, de_gap, expected):
        r = RunResult(
            variant_id="X",
            success=True,
            elapsed_s=1.0,
            return_code=0,
            delta_e_over_gap=de_gap,
        )
        assert r.verdict == expected

    def test_noisy_summary_takes_precedence_over_delta_e(self):
        """If both noisy_summary and delta_e_over_gap are set, noisy wins."""
        r = RunResult(
            variant_id="X",
            success=True,
            elapsed_s=1.0,
            return_code=0,
            delta_e_over_gap=0.01,
            noisy_summary={"success": False, "mean_r2": 0.5},
        )
        assert r.verdict == "ZNE-FAIL"

    def test_failure_takes_precedence_over_metrics(self):
        """If success=False, verdict is ERROR regardless of metrics."""
        r = RunResult(
            variant_id="X",
            success=False,
            elapsed_s=1.0,
            return_code=1,
            delta_e_over_gap=0.01,
        )
        assert r.verdict == "ERROR"


# ═══════════════════════════════════════════════════════════════════════════
# extract_metrics_from_output tests
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractMetrics:
    """Test extract_metrics_from_output with various output formats."""

    def test_empty_dir_returns_empty(self, tmp_path):
        result = extract_metrics_from_output(str(tmp_path))
        assert result == {}

    def test_nonexistent_dir_returns_empty(self):
        result = extract_metrics_from_output("/nonexistent/path/xyz")
        assert result == {}

    def test_noiseless_single_test_point(self, tmp_path):
        data = {
            "elapsed_s": 25.0,
            "phase4_results": [{"h_test": 1.5, "delta_e_over_gap": 0.023}],
        }
        (tmp_path / "pipeline_run_20260527_120000.json").write_text(json.dumps(data))
        result = extract_metrics_from_output(str(tmp_path))
        assert result["delta_e_over_gap"] == pytest.approx(0.023)
        assert result["n_test_points"] == 1

    def test_noiseless_multiple_test_points_returns_worst(self, tmp_path):
        data = {
            "elapsed_s": 30.0,
            "phase4_results": [
                {"h_test": 1.5, "delta_e_over_gap": 0.01},
                {"h_test": 1.75, "delta_e_over_gap": 0.08},
                {"h_test": 2.0, "delta_e_over_gap": 0.003},
            ],
        }
        (tmp_path / "pipeline_run_20260527_120000.json").write_text(json.dumps(data))
        result = extract_metrics_from_output(str(tmp_path))
        assert result["delta_e_over_gap"] == pytest.approx(0.08)
        assert result["n_test_points"] == 3

    def test_noiseless_phase3_failed(self, tmp_path):
        data = {"elapsed_s": 20.0, "phase4_results": []}
        (tmp_path / "pipeline_run_20260527_120000.json").write_text(json.dumps(data))
        result = extract_metrics_from_output(str(tmp_path))
        assert result.get("phase3_failed") is True

    def test_noiseless_null_delta_e_skipped(self, tmp_path):
        data = {
            "elapsed_s": 25.0,
            "phase4_results": [
                {"h_test": 1.5, "delta_e_over_gap": None},
                {"h_test": 1.75, "delta_e_over_gap": 0.04},
            ],
        }
        (tmp_path / "pipeline_run_20260527_120000.json").write_text(json.dumps(data))
        result = extract_metrics_from_output(str(tmp_path))
        assert result["delta_e_over_gap"] == pytest.approx(0.04)
        assert result["n_test_points"] == 1

    def test_noisy_output_extracted(self, tmp_path):
        data = {
            "summary": {
                "success_criteria_met": True,
                "mean_r2": 0.995,
                "mean_gain_pct": 15.3,
                "n_mitigated_wins": 5,
                "n_total": 6,
            }
        }
        (tmp_path / "noisy_3mode_20260527_120000.json").write_text(json.dumps(data))
        result = extract_metrics_from_output(str(tmp_path))
        ns = result["noisy_summary"]
        assert ns["success"] is True
        assert ns["mean_r2"] == pytest.approx(0.995)
        assert ns["mean_gain_pct"] == pytest.approx(15.3)
        assert ns["n_mitigated_wins"] == 5
        assert ns["n_total"] == 6

    def test_noisy_failure_extracted(self, tmp_path):
        data = {
            "summary": {
                "success_criteria_met": False,
                "mean_r2": 0.05,
                "mean_gain_pct": -2.0,
                "n_mitigated_wins": 1,
                "n_total": 5,
            }
        }
        (tmp_path / "noisy_3mode_20260527_120000.json").write_text(json.dumps(data))
        result = extract_metrics_from_output(str(tmp_path))
        assert result["noisy_summary"]["success"] is False

    def test_noiseless_preferred_over_noisy(self, tmp_path):
        """If both noiseless and noisy outputs exist, noiseless is returned."""
        noiseless = {
            "elapsed_s": 25.0,
            "phase4_results": [{"h_test": 1.5, "delta_e_over_gap": 0.02}],
        }
        noisy = {"summary": {"success_criteria_met": True, "mean_r2": 0.99}}
        (tmp_path / "pipeline_run_20260527_120000.json").write_text(json.dumps(noiseless))
        (tmp_path / "noisy_3mode_20260527_120000.json").write_text(json.dumps(noisy))
        result = extract_metrics_from_output(str(tmp_path))
        # Noiseless takes precedence
        assert "delta_e_over_gap" in result
        assert "noisy_summary" not in result

    def test_corrupt_json_returns_empty(self, tmp_path):
        (tmp_path / "pipeline_run_20260527_120000.json").write_text("not valid json{{{")
        result = extract_metrics_from_output(str(tmp_path))
        assert result == {}

    def test_most_recent_file_used(self, tmp_path):
        """When multiple pipeline_run files exist, the most recent is used."""
        old = {"elapsed_s": 10.0, "phase4_results": [{"h_test": 1.5, "delta_e_over_gap": 0.5}]}
        new = {"elapsed_s": 20.0, "phase4_results": [{"h_test": 1.5, "delta_e_over_gap": 0.01}]}
        (tmp_path / "pipeline_run_20260527_100000.json").write_text(json.dumps(old))
        (tmp_path / "pipeline_run_20260527_120000.json").write_text(json.dumps(new))
        result = extract_metrics_from_output(str(tmp_path))
        assert result["delta_e_over_gap"] == pytest.approx(0.01)


# ═══════════════════════════════════════════════════════════════════════════
# run_variant tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRunVariant:
    """Test run_variant execution logic."""

    def _make_variant(self, command: list[str], output_dir: str) -> PipelineVariant:
        return PipelineVariant(
            id="TEST-1",
            description="Test variant",
            category="noiseless",
            command=command,
            hypothesis="Test",
            expected_outcome="PASS",
            output_dir=output_dir,
        )

    def test_dry_run_returns_success_without_executing(self, tmp_path):
        v = self._make_variant(["false"], str(tmp_path / "out"))
        result = run_variant(v, dry_run=True)
        assert result.success is True
        assert result.elapsed_s == 0.0
        assert result.return_code == 0

    def test_successful_command(self, tmp_path):
        import sys

        out_dir = tmp_path / "out"
        # Create a pipeline output file so metrics can be extracted
        out_dir.mkdir()
        data = {"elapsed_s": 1.0, "phase4_results": [{"h_test": 1.5, "delta_e_over_gap": 0.02}]}
        (out_dir / "pipeline_run_20260527_120000.json").write_text(json.dumps(data))

        v = self._make_variant([sys.executable, "-c", "pass"], str(out_dir))
        result = run_variant(v, timeout=30)
        assert result.success is True
        assert result.return_code == 0
        assert result.elapsed_s > 0
        assert result.delta_e_over_gap == pytest.approx(0.02)

    def test_failed_command(self, tmp_path):
        import sys

        v = self._make_variant(
            [sys.executable, "-c", "import sys; sys.exit(1)"],
            str(tmp_path / "out"),
        )
        result = run_variant(v, timeout=30)
        assert result.success is False
        assert result.return_code == 1

    def test_timeout_returns_failure(self, tmp_path):
        import sys

        v = self._make_variant(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            str(tmp_path / "out"),
        )
        result = run_variant(v, timeout=1)
        assert result.success is False
        assert result.return_code == -1
        assert "TIMEOUT" in result.error_msg

    def test_phase3_failure_detected(self, tmp_path):
        import sys

        # Command that prints "Phase 3 FAILED" to stderr and exits non-zero
        cmd = "import sys; sys.stderr.write('Phase 3 FAILED: fidelity too low\\n'); sys.exit(1)"
        v = self._make_variant(
            [sys.executable, "-c", cmd],
            str(tmp_path / "out"),
        )
        result = run_variant(v, timeout=30)
        assert result.success is False
        assert result.phase3_failed is True
        assert result.verdict == "SKIP-P3"

    def test_output_dir_created(self, tmp_path):
        import sys

        out_dir = tmp_path / "nested" / "deep" / "out"
        v = self._make_variant([sys.executable, "-c", "pass"], str(out_dir))
        run_variant(v, timeout=30)
        assert out_dir.exists()


# ═══════════════════════════════════════════════════════════════════════════
# VariantRunner tests
# ═══════════════════════════════════════════════════════════════════════════


class TestVariantRunner:
    """Test VariantRunner orchestration logic."""

    def _make_variants(self, n: int, category: str = "noiseless") -> list[PipelineVariant]:
        return [
            PipelineVariant(
                id=f"{category.upper()[:2]}-{i}",
                description=f"Variant {i}",
                category=category,
                command=["echo", "test"],
                hypothesis=f"Hypothesis {i}",
                expected_outcome="PASS",
                output_dir=f"/tmp/test_{i}",
            )
            for i in range(n)
        ]

    def test_get_variants_all(self):
        runner = VariantRunner(
            topology="chain_1d",
            n_qubits=6,
            noiseless=self._make_variants(3, "noiseless"),
            noisy=self._make_variants(2, "noisy"),
            extended=self._make_variants(1, "extended"),
        )
        all_v = runner.get_variants()
        assert len(all_v) == 6

    def test_get_variants_noiseless_only(self):
        runner = VariantRunner(
            topology="chain_1d",
            n_qubits=6,
            noiseless=self._make_variants(3, "noiseless"),
            noisy=self._make_variants(2, "noisy"),
            extended=self._make_variants(1, "extended"),
        )
        v = runner.get_variants(noiseless_only=True)
        assert len(v) == 3
        assert all(x.category == "noiseless" for x in v)

    def test_get_variants_noisy_only(self):
        runner = VariantRunner(
            topology="ladder",
            n_qubits=10,
            noiseless=self._make_variants(3, "noiseless"),
            noisy=self._make_variants(2, "noisy"),
            extended=self._make_variants(1, "extended"),
        )
        v = runner.get_variants(noisy_only=True)
        assert len(v) == 2

    def test_get_variants_extended_only(self):
        runner = VariantRunner(
            topology="triangular",
            n_qubits=10,
            noiseless=self._make_variants(3, "noiseless"),
            noisy=self._make_variants(2, "noisy"),
            extended=self._make_variants(4, "extended"),
        )
        v = runner.get_variants(extended_only=True)
        assert len(v) == 4

    def test_output_base_property(self):
        runner = VariantRunner(
            topology="triangular",
            n_qubits=6,
            noiseless=[],
            noisy=[],
            extended=[],
        )
        assert runner.output_base == "results/thesis/variants_N6_triangular"

    def test_run_single_out_of_range(self, capsys):
        runner = VariantRunner(
            topology="chain_1d",
            n_qubits=6,
            noiseless=self._make_variants(2),
            noisy=[],
            extended=[],
        )
        exit_code = runner.run_single(self._make_variants(2), 5, dry_run=True)
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "out of range" in captured.out

    def test_run_single_dry_run(self, capsys):
        variants = self._make_variants(3)
        runner = VariantRunner(
            topology="chain_1d",
            n_qubits=6,
            noiseless=variants,
            noisy=[],
            extended=[],
        )
        exit_code = runner.run_single(variants, 0, dry_run=True)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out

    def test_run_all_dry_run(self, capsys, tmp_path, monkeypatch):
        variants = self._make_variants(2)
        runner = VariantRunner(
            topology="test",
            n_qubits=4,
            noiseless=variants,
            noisy=[],
            extended=[],
        )
        # Prevent sys.exit in dry_run mode (run_all returns exit code)
        exit_code = runner.run_all(variants, dry_run=True)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "FINAL SUMMARY" in captured.out

    def test_format_metric_pass(self):
        runner = VariantRunner(
            topology="x",
            n_qubits=6,
            noiseless=[],
            noisy=[],
            extended=[],
        )
        r = RunResult(
            variant_id="X",
            success=True,
            elapsed_s=1.0,
            return_code=0,
            delta_e_over_gap=0.023,
        )
        metric = runner._format_metric(r)
        assert "0.0230" in metric

    def test_format_metric_noisy(self):
        runner = VariantRunner(
            topology="x",
            n_qubits=6,
            noiseless=[],
            noisy=[],
            extended=[],
        )
        r = RunResult(
            variant_id="X",
            success=True,
            elapsed_s=1.0,
            return_code=0,
            noisy_summary={"mean_r2": 0.995, "mean_gain_pct": 12.5},
        )
        metric = runner._format_metric(r)
        assert "R²=0.995" in metric
        assert "12.5%" in metric

    def test_format_metric_empty(self):
        runner = VariantRunner(
            topology="x",
            n_qubits=6,
            noiseless=[],
            noisy=[],
            extended=[],
        )
        r = RunResult(variant_id="X", success=True, elapsed_s=1.0, return_code=0)
        metric = runner._format_metric(r)
        assert metric == ""

    def test_save_log_creates_file(self, tmp_path, monkeypatch):
        runner = VariantRunner(
            topology="test_topo",
            n_qubits=4,
            noiseless=[],
            noisy=[],
            extended=[],
        )
        # Override output_base to use tmp_path
        monkeypatch.setattr(
            VariantRunner,
            "output_base",
            property(lambda self: str(tmp_path)),
        )
        results = [
            RunResult(
                variant_id="A", success=True, elapsed_s=1.0, return_code=0, delta_e_over_gap=0.01
            ),
            RunResult(
                variant_id="B", success=False, elapsed_s=2.0, return_code=1, error_msg="fail"
            ),
        ]
        variants = self._make_variants(2)
        runner._save_log(results, variants, start_from=0, total_elapsed=3.0)

        log_files = list(tmp_path.glob("execution_log_*.json"))
        assert len(log_files) == 1

        with open(log_files[0]) as f:
            log = json.load(f)
        assert log["topology"] == "test_topo"
        assert log["n_qubits"] == 4
        assert log["passed"] == 1
        assert log["failed"] == 1
        assert log["verdicts"]["PASS"] == 1
        assert log["verdicts"]["ERROR"] == 1
        assert len(log["results"]) == 2


# ═══════════════════════════════════════════════════════════════════════════
# CLI --seed argument tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCLISeedArgument:
    """Test that --seed argument is properly wired through the CLI framework."""

    def test_seed_argument_exists_in_vqe_args(self):
        import argparse

        from qmbp_simulation.framework.cli import add_vqe_args

        parser = argparse.ArgumentParser()
        add_vqe_args(parser)
        args = parser.parse_args(["--seed", "43"])
        assert args.seed == 43

    def test_seed_default_is_none(self):
        import argparse

        from qmbp_simulation.framework.cli import add_vqe_args

        parser = argparse.ArgumentParser()
        add_vqe_args(parser)
        args = parser.parse_args([])
        assert args.seed is None

    def test_seed_accepts_various_values(self):
        import argparse

        from qmbp_simulation.framework.cli import add_vqe_args

        parser = argparse.ArgumentParser()
        add_vqe_args(parser)
        for seed_val in [0, 1, 42, 43, 44, 12345]:
            args = parser.parse_args(["--seed", str(seed_val)])
            assert args.seed == seed_val


# ═══════════════════════════════════════════════════════════════════════════
# Integration: variant scripts import correctly
# ═══════════════════════════════════════════════════════════════════════════


class TestVariantScriptImports:
    """Test that all variant scripts can be imported without errors."""

    def test_framework_exports_variant_runner_symbols(self):
        from qmbp_simulation.framework import (
            PipelineVariant,
            RunResult,
            VariantRunner,
            run_variant_script,
        )

        assert PipelineVariant is not None
        assert RunResult is not None
        assert VariantRunner is not None
        assert callable(run_variant_script)

    def test_variant_runner_direct_import(self):
        from qmbp_simulation.framework.variant_runner import (
            PipelineVariant,
            RunResult,
            VariantRunner,
            create_variant_cli,
            extract_metrics_from_output,
            run_variant,
            run_variant_script,
        )

        assert all(
            callable(f)
            for f in [
                extract_metrics_from_output,
                run_variant,
                create_variant_cli,
                run_variant_script,
            ]
        )
        assert PipelineVariant is not None
        assert RunResult is not None
        assert VariantRunner is not None
