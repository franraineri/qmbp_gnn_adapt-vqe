"""Tests for the Best Results Scoreboard generator.

Validates the core functionality of
``scripts/analysis/generate_best_results_scoreboard.py``:
- parse_eval_report: markdown → EvalEntry (both old |ΔE|/N and new Fidelity layouts)
- scan_all_reports: directory walk + p_filter
- compute_best_per_topology_n: grouping by (topology, p, N) + best selection
- h-target filtering with tolerance
- separation by p_layers (p=1 vs p=2 never mixed)
- end-to-end generate_scoreboard → one file per p

The module lives under scripts/ (not src/), but its parse/rank logic is the
single source of truth for the scoreboard, so we import and test it directly.
"""

from __future__ import annotations

import importlib

import pytest

sb = importlib.import_module("scripts.analysis.generate_best_results_scoreboard")


# ═══════════════════════════════════════════════════════════════════════════
# Report fixtures
# ═══════════════════════════════════════════════════════════════════════════


def _report_new_format(
    *,
    p_layers: int,
    checkpoint: str,
    is_mt: bool,
    rows_by_n: dict[int, list[tuple]],
) -> str:
    """Build a markdown report in the CURRENT layout (with Fidelity column).

    rows_by_n: {n: [(h, e_pred, e_exact, gap, fidelity_or_None), ...]}
    """
    lines = [
        "# Model Evaluation: dummy",
        "",
        "**Date**: 2026-08-28 12:00 UTC",
        f"**Model**: {checkpoint}",
        f"**p_layers**: {p_layers}",
        f"**Multi-topology**: {'YES' if is_mt else 'no'}",
        "**h-range**: [2.0, 3.0] (5 pts)",
        f"**Target N**: {sorted(rows_by_n)}",
        "",
        "---",
        "",
    ]
    for n, rows in rows_by_n.items():
        lines.append(f"## N = {n} (38 params)")
        lines.append("")
        lines.append(
            "| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | "
            "Factor | Category | Action | Note |"
        )
        lines.append(
            "|---|--------|---------|------|--------|-----|----------|--------|"
            "--------|----------|--------|------|"
        )
        for h, e_pred, e_exact, gap, fid in rows:
            abs_err = abs(e_pred - e_exact)
            de_gap = abs_err / gap
            fid_cell = "N/A" if fid is None else f"{fid:.4f}"
            lines.append(
                f"| {h:.3f} | {e_pred:.4f} | {e_exact:.4f} | {abs_err:.4f} | "
                f"{gap:.4f} | {de_gap:.4f} | {fid_cell} | N/A | — | "
                f"pass(0.5) | none |  |"
            )
        lines.append("")
    return "\n".join(lines)


def _report_old_format(*, p_layers: int, checkpoint: str, rows_by_n: dict[int, list[tuple]]) -> str:
    """Build a markdown report in the OLD layout (|ΔE|/N column, no Fidelity)."""
    lines = [
        "# Model Evaluation: dummy",
        "",
        "**Date**: 2026-08-01 12:00 UTC",
        f"**Model**: {checkpoint}",
        f"**p_layers**: {p_layers}",
        "**Multi-topology**: no",
        "",
        "---",
        "",
    ]
    for n, rows in rows_by_n.items():
        lines.append(f"## N = {n} (38 params)")
        lines.append("")
        lines.append(
            "| h | E_pred | E_exact | |ΔE| | |ΔE|/N | gap | ΔE/gap | Category | Action | Note |"
        )
        lines.append(
            "|---|--------|---------|------|-------|-----|--------|----------|--------|------|"
        )
        for h, e_pred, e_exact, gap in rows:
            abs_err = abs(e_pred - e_exact)
            de_gap = abs_err / gap
            lines.append(
                f"| {h:.3f} | {e_pred:.4f} | {e_exact:.4f} | {abs_err:.4f} | "
                f"{abs_err / n:.2e} | {gap:.4f} | {de_gap:.4f} | pass(0.5) | none |  |"
            )
        lines.append("")
    return "\n".join(lines)


@pytest.fixture
def scoreboard_env(tmp_path, monkeypatch):
    """Redirect the scoreboard module's directory constants into tmp_path."""
    root = tmp_path
    eval_dir = root / "results" / "extrapolation_evals"
    comp_dir = root / "results" / "model_comparison"
    results_dir = root / "results"
    eval_dir.mkdir(parents=True)
    comp_dir.mkdir(parents=True)

    monkeypatch.setattr(sb, "ROOT", root)
    monkeypatch.setattr(sb, "EVAL_DIR", eval_dir)
    monkeypatch.setattr(sb, "COMPARISON_DIR", comp_dir)
    monkeypatch.setattr(sb, "RESULTS_DIR", results_dir)
    # Reset the global h-tolerance to the module default for each test.
    monkeypatch.setattr(sb, "H_TOLERANCE", 0.15)
    return {"root": root, "eval_dir": eval_dir, "comp_dir": comp_dir}


def _write(subdir, name: str, content: str):
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / name).write_text(content, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# parse_eval_report
# ═══════════════════════════════════════════════════════════════════════════


class TestParseEvalReport:
    def test_new_format_extracts_entry_at_target_h(self, scoreboard_env):
        report = _report_new_format(
            p_layers=1,
            checkpoint="ckpt_p1.pt",
            is_mt=False,
            rows_by_n={10: [(2.5, -10.0, -10.05, 0.5, 0.99)]},
        )
        d = scoreboard_env["eval_dir"] / "chain_1d_p1"
        _write(d, "eval_chain_1d_20260828_120000.md", report)

        entries = sb.parse_eval_report(d / "eval_chain_1d_20260828_120000.md", target_h=2.5)
        assert len(entries) == 1
        e = entries[0]
        assert e.topology == "chain_1d"
        assert e.p_layers == 1
        assert e.n_qubits == 10
        assert e.result.abs_error == pytest.approx(0.05, abs=1e-6)
        assert e.result.fidelity == pytest.approx(0.99, abs=1e-6)
        assert e.is_mt is False

    def test_old_format_without_fidelity(self, scoreboard_env):
        report = _report_old_format(
            p_layers=1,
            checkpoint="ckpt.pt",
            rows_by_n={16: [(2.5, -20.0, -20.10, 1.0)]},
        )
        d = scoreboard_env["eval_dir"] / "heavy_hex_p1"
        _write(d, "eval_heavy_hex_20260801_120000.md", report)

        entries = sb.parse_eval_report(d / "eval_heavy_hex_20260801_120000.md", target_h=2.5)
        assert len(entries) == 1
        assert entries[0].result.abs_error == pytest.approx(0.10, abs=1e-6)
        assert entries[0].result.fidelity is None

    def test_p_layers_from_directory_name(self, scoreboard_env):
        # No **p_layers** header line → fall back to dir name _p2
        report = _report_old_format(
            p_layers=2, checkpoint="c.pt", rows_by_n={10: [(2.5, -10.0, -10.02, 0.5)]}
        )
        # Strip the header p_layers line to force dir-name inference
        report = "\n".join(l for l in report.splitlines() if not l.startswith("**p_layers**"))
        d = scoreboard_env["eval_dir"] / "chain_1d_p2"
        _write(d, "eval_chain_1d_20260801_120000.md", report)

        entries = sb.parse_eval_report(d / "eval_chain_1d_20260801_120000.md", target_h=2.5)
        assert len(entries) == 1
        assert entries[0].p_layers == 2

    def test_header_p_layers_overrides_directory(self, scoreboard_env):
        # Dir says _p1 but header says p_layers: 2 → header wins
        report = _report_new_format(
            p_layers=2,
            checkpoint="c.pt",
            is_mt=False,
            rows_by_n={10: [(2.5, -10.0, -10.02, 0.5, None)]},
        )
        d = scoreboard_env["eval_dir"] / "chain_1d_p1"
        _write(d, "eval_chain_1d_20260828_130000.md", report)

        entries = sb.parse_eval_report(d / "eval_chain_1d_20260828_130000.md", target_h=2.5)
        assert entries[0].p_layers == 2

    def test_rows_far_from_target_h_are_skipped(self, scoreboard_env):
        report = _report_new_format(
            p_layers=1,
            checkpoint="c.pt",
            is_mt=False,
            rows_by_n={10: [(4.0, -10.0, -10.02, 0.5, None)]},  # h=4.0, target 2.5
        )
        d = scoreboard_env["eval_dir"] / "chain_1d_p1"
        _write(d, "eval_chain_1d_20260828_140000.md", report)
        entries = sb.parse_eval_report(d / "eval_chain_1d_20260828_140000.md", target_h=2.5)
        assert entries == []

    def test_row_within_tolerance_is_accepted(self, scoreboard_env):
        # h=2.4 with default tolerance 0.15 → within [2.35, 2.65]
        report = _report_new_format(
            p_layers=1,
            checkpoint="c.pt",
            is_mt=False,
            rows_by_n={10: [(2.4, -10.0, -10.02, 0.5, None)]},
        )
        d = scoreboard_env["eval_dir"] / "chain_1d_p1"
        _write(d, "eval_chain_1d_20260828_150000.md", report)
        entries = sb.parse_eval_report(d / "eval_chain_1d_20260828_150000.md", target_h=2.5)
        assert len(entries) == 1
        assert entries[0].h_used == pytest.approx(2.4)

    def test_stale_e_exact_recomputes_abs_error(self, scoreboard_env):
        # Build a row where the printed |ΔE| disagrees with |e_pred - e_exact|.
        lines = [
            "# Model Evaluation: dummy",
            "",
            "**Model**: c.pt",
            "**p_layers**: 1",
            "**Multi-topology**: no",
            "",
            "## N = 10 (38 params)",
            "",
            "| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
            # e_pred=-10.0 e_exact=-10.50 → real |ΔE|=0.50, but printed 0.05 (stale)
            "| 2.500 | -10.0000 | -10.5000 | 0.0500 | 1.0000 | 0.0500 | N/A | N/A | — | pass | none |  |",
            "",
        ]
        d = scoreboard_env["eval_dir"] / "chain_1d_p1"
        _write(d, "eval_chain_1d_20260828_160000.md", "\n".join(lines))
        entries = sb.parse_eval_report(d / "eval_chain_1d_20260828_160000.md", target_h=2.5)
        assert len(entries) == 1
        # Parser should recompute |ΔE| from authoritative energies
        assert entries[0].result.abs_error == pytest.approx(0.50, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# scan_all_reports + p_filter
# ═══════════════════════════════════════════════════════════════════════════


class TestScanAllReports:
    def test_scans_both_dirs_and_filters_by_p(self, scoreboard_env):
        _write(
            scoreboard_env["eval_dir"] / "chain_1d_p1",
            "eval_chain_1d_20260828_120000.md",
            _report_new_format(
                p_layers=1,
                checkpoint="a_p1.pt",
                is_mt=False,
                rows_by_n={10: [(2.5, -10.0, -10.02, 0.5, None)]},
            ),
        )
        _write(
            scoreboard_env["eval_dir"] / "chain_1d_p2",
            "eval_chain_1d_20260828_130000.md",
            _report_new_format(
                p_layers=2,
                checkpoint="a_p2.pt",
                is_mt=False,
                rows_by_n={10: [(2.5, -10.0, -10.03, 0.5, None)]},
            ),
        )
        _write(
            scoreboard_env["comp_dir"] / "heavy_hex_p1",
            "eval_heavy_hex_20260828_140000.md",
            _report_new_format(
                p_layers=1,
                checkpoint="b_p1.pt",
                is_mt=True,
                rows_by_n={16: [(2.5, -20.0, -20.05, 1.0, None)]},
            ),
        )

        all_e = sb.scan_all_reports(target_h=2.5)
        assert len(all_e) == 3

        p1 = sb.scan_all_reports(target_h=2.5, p_filter=1)
        assert {e.p_layers for e in p1} == {1}
        assert len(p1) == 2

        p2 = sb.scan_all_reports(target_h=2.5, p_filter=2)
        assert {e.p_layers for e in p2} == {2}
        assert len(p2) == 1

    def test_skips_underscore_dirs(self, scoreboard_env):
        _write(
            scoreboard_env["eval_dir"] / "_old",
            "eval_chain_1d_20260101_120000.md",
            _report_new_format(
                p_layers=1,
                checkpoint="old.pt",
                is_mt=False,
                rows_by_n={10: [(2.5, -10.0, -10.02, 0.5, None)]},
            ),
        )
        assert sb.scan_all_reports(target_h=2.5) == []


# ═══════════════════════════════════════════════════════════════════════════
# compute_best_per_topology_n — the core ranking
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeBest:
    def test_selects_lowest_abs_error_per_group(self, scoreboard_env):
        # Two reports for chain_1d p1 N10: one worse, one better.
        _write(
            scoreboard_env["eval_dir"] / "chain_1d_p1",
            "eval_chain_1d_20260828_120000.md",
            _report_new_format(
                p_layers=1,
                checkpoint="worse.pt",
                is_mt=False,
                rows_by_n={10: [(2.5, -10.0, -10.30, 0.5, None)]},
            ),  # |ΔE|=0.30
        )
        _write(
            scoreboard_env["eval_dir"] / "chain_1d_p1",
            "eval_chain_1d_20260828_130000.md",
            _report_new_format(
                p_layers=1,
                checkpoint="better.pt",
                is_mt=False,
                rows_by_n={10: [(2.5, -10.0, -10.02, 0.5, None)]},
            ),  # |ΔE|=0.02
        )
        entries = sb.scan_all_reports(target_h=2.5)
        best = sb.compute_best_per_topology_n(entries)
        r = best["chain_1d"][1][10]
        assert r.best_abs_error == pytest.approx(0.02, abs=1e-6)
        assert r.checkpoint == "better.pt"
        assert r.grade == "A"  # |ΔE| < 0.05

    def test_p1_and_p2_never_mix(self, scoreboard_env):
        # Same topology+N, different p: must produce two independent bests.
        _write(
            scoreboard_env["eval_dir"] / "chain_1d_p1",
            "eval_chain_1d_20260828_120000.md",
            _report_new_format(
                p_layers=1,
                checkpoint="p1.pt",
                is_mt=False,
                rows_by_n={10: [(2.5, -10.0, -10.40, 0.5, None)]},
            ),  # |ΔE|=0.40 (D)
        )
        _write(
            scoreboard_env["eval_dir"] / "chain_1d_p2",
            "eval_chain_1d_20260828_130000.md",
            _report_new_format(
                p_layers=2,
                checkpoint="p2.pt",
                is_mt=False,
                rows_by_n={10: [(2.5, -10.0, -10.01, 0.5, None)]},
            ),  # |ΔE|=0.01 (A)
        )
        entries = sb.scan_all_reports(target_h=2.5)
        best = sb.compute_best_per_topology_n(entries)
        assert set(best["chain_1d"].keys()) == {1, 2}
        assert best["chain_1d"][1][10].best_abs_error == pytest.approx(0.40, abs=1e-6)
        assert best["chain_1d"][2][10].best_abs_error == pytest.approx(0.01, abs=1e-6)
        # The bad p1 result must NOT leak into p2's ranking.
        assert best["chain_1d"][2][10].checkpoint == "p2.pt"

    def test_grade_thresholds(self, scoreboard_env):
        cases = [
            (-10.02, "A"),  # 0.02
            (-10.08, "B"),  # 0.08
            (-10.20, "C"),  # 0.20
            (-10.50, "D"),  # 0.50
            (-12.00, "F"),  # 2.00
        ]
        for i, (e_pred_target, _grade) in enumerate(cases):
            _write(
                scoreboard_env["eval_dir"] / f"topo{i}_p1",
                f"eval_topo{i}_2026082{i}_120000.md",
                _report_new_format(
                    p_layers=1,
                    checkpoint=f"c{i}.pt",
                    is_mt=False,
                    rows_by_n={10: [(2.5, e_pred_target, -10.0, 0.5, None)]},
                ),
            )
        entries = sb.scan_all_reports(target_h=2.5)
        best = sb.compute_best_per_topology_n(entries)
        for i, (_e, grade) in enumerate(cases):
            assert best[f"topo{i}"][1][10].grade == grade

    def test_dedup_same_checkpoint_across_dirs(self, scoreboard_env):
        # Identical result in extrapolation_evals AND model_comparison → dedup.
        rep = _report_new_format(
            p_layers=1,
            checkpoint="dup.pt",
            is_mt=False,
            rows_by_n={10: [(2.5, -10.0, -10.02, 0.5, None)]},
        )
        _write(scoreboard_env["eval_dir"] / "chain_1d_p1", "eval_chain_1d_20260828_120000.md", rep)
        _write(scoreboard_env["comp_dir"] / "chain_1d_p1", "eval_chain_1d_20260828_120000.md", rep)
        entries = sb.scan_all_reports(target_h=2.5)
        assert len(entries) == 2  # scanned from both dirs
        best = sb.compute_best_per_topology_n(entries)
        # After dedup, the group collapses to a single best.
        assert best["chain_1d"][1][10].n_reports_scanned == 1


# ═══════════════════════════════════════════════════════════════════════════
# generate_scoreboard — end-to-end, one file per p
# ═══════════════════════════════════════════════════════════════════════════


class TestGenerateScoreboard:
    def test_one_file_per_p(self, scoreboard_env):
        _write(
            scoreboard_env["eval_dir"] / "chain_1d_p1",
            "eval_chain_1d_20260828_120000.md",
            _report_new_format(
                p_layers=1,
                checkpoint="p1.pt",
                is_mt=False,
                rows_by_n={10: [(2.5, -10.0, -10.02, 0.5, None)]},
            ),
        )
        _write(
            scoreboard_env["eval_dir"] / "chain_1d_p2",
            "eval_chain_1d_20260828_130000.md",
            _report_new_format(
                p_layers=2,
                checkpoint="p2.pt",
                is_mt=False,
                rows_by_n={10: [(2.5, -10.0, -10.03, 0.5, None)]},
            ),
        )
        summary = sb.generate_scoreboard(target_h=2.5, output_json=True)
        assert set(summary["p_values"]) == {1, 2}

        p1_md = sb.scoreboard_md_path(1)
        p2_md = sb.scoreboard_md_path(2)
        assert p1_md.exists() and p2_md.exists()
        assert "p=1" in p1_md.read_text()
        assert "p=2" in p2_md.read_text()
        # p1 file must not contain the p2 checkpoint and vice versa
        assert "p2.pt" not in p1_md.read_text()
        assert "p1.pt" not in p2_md.read_text()

        assert sb.scoreboard_json_path(1).exists()
        assert sb.scoreboard_json_path(2).exists()

    def test_p_filter_only_writes_that_p(self, scoreboard_env):
        _write(
            scoreboard_env["eval_dir"] / "chain_1d_p1",
            "eval_chain_1d_20260828_120000.md",
            _report_new_format(
                p_layers=1,
                checkpoint="p1.pt",
                is_mt=False,
                rows_by_n={10: [(2.5, -10.0, -10.02, 0.5, None)]},
            ),
        )
        summary = sb.generate_scoreboard(target_h=2.5, output_json=True, p_filter=1)
        assert summary["p_values"] == [1]
        assert sb.scoreboard_md_path(1).exists()

    def test_empty_when_no_data_for_p(self, scoreboard_env):
        summary = sb.generate_scoreboard(target_h=2.5, output_json=True, p_filter=2)
        assert summary["n_entries_parsed"] == 0
        assert not sb.scoreboard_md_path(2).exists()

    def test_legacy_best_by_topology_is_lowest_p(self, scoreboard_env):
        _write(
            scoreboard_env["eval_dir"] / "chain_1d_p1",
            "eval_chain_1d_20260828_120000.md",
            _report_new_format(
                p_layers=1,
                checkpoint="p1.pt",
                is_mt=False,
                rows_by_n={10: [(2.5, -10.0, -10.02, 0.5, None)]},
            ),
        )
        _write(
            scoreboard_env["eval_dir"] / "chain_1d_p2",
            "eval_chain_1d_20260828_130000.md",
            _report_new_format(
                p_layers=2,
                checkpoint="p2.pt",
                is_mt=False,
                rows_by_n={10: [(2.5, -10.0, -10.03, 0.5, None)]},
            ),
        )
        summary = sb.generate_scoreboard(target_h=2.5, output_json=False)
        # Legacy alias resolves to the lowest p present (p=1).
        assert "chain_1d" in summary["best_by_topology"]
        assert summary["best_by_topology"]["chain_1d"]["10"]["p_layers"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# cross_validate_with_registry — p-aware zoo cross-check
# ═══════════════════════════════════════════════════════════════════════════


class _FakeZooEntry:
    def __init__(self, checkpoint_file, p_layers, pass_rate_by_n):
        self.checkpoint_file = checkpoint_file
        self.p_layers = p_layers
        self.pass_rate_by_n = pass_rate_by_n


def _best(topo, p, n, abs_error, grade, checkpoint):
    return sb.BestResult(
        topology=topo,
        n_qubits=n,
        p_layers=p,
        best_de_gap=abs_error,
        best_abs_error=abs_error,
        gap_at_best=1.0,
        h_used=2.5,
        grade=grade,
        score=0.5,
        checkpoint=checkpoint,
        model_type="ST",
        date="2026-08-28",
        report_file="results/extrapolation_evals/chain_1d_p1/eval_x.md",
    )


class TestCrossValidatePAware:
    def test_skips_zoo_entry_of_different_p(self, monkeypatch):
        # Scoreboard row is p=1 grade A. The only zoo entry with a matching
        # checkpoint name is p=2 with 0% pass. Because p differs, the cross-check
        # must SKIP it (no false "stale zoo data" warning).
        ckpt = "unified_tfim_br_chain_1d_multiN_6+8+10_p2.pt"
        fake_manifest = [_FakeZooEntry(ckpt, p_layers=2, pass_rate_by_n={"10": 0.0})]

        monkeypatch.setattr(
            "qmbp_simulation.predictors.model_zoo._load_manifest",
            lambda: fake_manifest,
        )

        # Registry DB returns no records → Check 2 is a no-op.
        class _FakeDB:
            def get_model(self, name):
                return None

        monkeypatch.setattr(
            "qmbp_simulation.predictors.model_registry_db.ModelRegistryDB",
            lambda: _FakeDB(),
        )

        best_by_topo = {"chain_1d": {1: {10: _best("chain_1d", 1, 10, 0.02, "A", ckpt)}}}
        lines = sb.cross_validate_with_registry(best_by_topo)
        # No warning should be produced (p mismatch skipped).
        assert not any("stale zoo data" in ln for ln in lines)

    def test_flags_stale_zoo_when_p_matches(self, monkeypatch):
        # Same as above but zoo entry is p=1 → cross-check applies and flags.
        ckpt = "unified_tfim_br_chain_1d_multiN_6+8+10_p1.pt"
        fake_manifest = [_FakeZooEntry(ckpt, p_layers=1, pass_rate_by_n={"10": 0.0})]

        monkeypatch.setattr(
            "qmbp_simulation.predictors.model_zoo._load_manifest",
            lambda: fake_manifest,
        )

        class _FakeDB:
            def get_model(self, name):
                return None

        monkeypatch.setattr(
            "qmbp_simulation.predictors.model_registry_db.ModelRegistryDB",
            lambda: _FakeDB(),
        )

        best_by_topo = {"chain_1d": {1: {10: _best("chain_1d", 1, 10, 0.02, "A", ckpt)}}}
        lines = sb.cross_validate_with_registry(best_by_topo)
        assert any("stale zoo data" in ln and "p=1" in ln for ln in lines)
