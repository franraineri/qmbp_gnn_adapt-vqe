"""Scan and analyse noiseless experiment results.

Supports any experiment folder via --dirs flag. Defaults to v3 folders.
Provides deep per-run analysis with per-point breakdown.

For quick group-level health check without full file parsing, use:
    python -m project_health --diagnose --model tfim

Usage:
    python scripts/scan_new_runs.py
    python scripts/scan_new_runs.py --verbose
    python scripts/scan_new_runs.py --dirs exp_noiseless_tfim_4 exp_noiseless_tfim_longitudinal_4
    python scripts/scan_new_runs.py --filter-model tfim --filter-topo chain_1d
    python scripts/scan_new_runs.py --json /tmp/analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DEFAULT_EXPERIMENT_DIRS = [
    "exp_noiseless_heisenberg_transverse_v3",
    "exp_noiseless_tfim_v3",
    "exp_noiseless_tfim_longitudinal_v3",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RunMetrics:
    """Parsed metrics from one noiseless pipeline run."""

    source_file: str
    experiment: str
    timestamp: str
    model: str
    n_qubits: int
    p_layers: int
    topology: str
    elapsed_s: float
    h_points: int
    # Section verdicts
    s1_pass: bool
    s2_pass: bool
    s3_pass: bool | None
    s4_pass: bool | None
    # Section 1: ExactDiag
    gap_min: float | None = None
    gap_max: float | None = None
    # Section 2: VQE
    vqe_n_pass: int = 0
    vqe_n_total: int = 0
    mean_fidelity: float | None = None
    min_fidelity: float | None = None
    mean_de_gap_vqe: float | None = None
    max_de_gap_vqe: float | None = None
    n_converged: int | None = None
    vqe_time_s: float = 0.0
    theta_smoothness_max: float | None = None
    n_variational_violations: int = 0
    # Section 3: MPNN
    mpnn_final_mse: float | None = None
    mpnn_best_mse: float | None = None
    mpnn_n_epochs: int | None = None
    mpnn_stopped_early: bool | None = None
    mpnn_n_params: int | None = None
    # Section 4: Deploy
    deploy_n_pass: int | None = None
    deploy_n_total: int | None = None
    deploy_n_correct_label: int | None = None
    deploy_mean_de_gap: float | None = None
    deploy_max_de_gap: float | None = None
    deploy_mean_fidelity: float | None = None
    deploy_speedup: float | None = None
    deploy_mag_x_error: float | None = None
    deploy_corr_zz_error: float | None = None


@dataclass
class GroupStats:
    """Aggregate stats for a (model, topology, p_layers) group."""

    model: str
    topology: str
    p_layers: int
    n_runs: int = 0
    n_full_pass: int = 0
    n_vqe_pass: int = 0
    n_deploy_pass: int = 0
    mean_elapsed_s: float = 0.0
    # VQE aggregates
    mean_fidelity_avg: float | None = None
    mean_de_gap_vqe_avg: float | None = None
    vqe_converge_rate: float | None = None
    # Deploy aggregates
    deploy_pass_rate: float | None = None
    deploy_label_rate: float | None = None
    deploy_mean_de_gap_avg: float | None = None
    deploy_mean_fidelity_avg: float | None = None
    # Issues
    issues: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def _safe_mean(values: list[float | None]) -> float | None:
    """Compute mean of finite non-None values (filters inf/NaN)."""
    clean = [v for v in values if v is not None and math.isfinite(v)]
    return sum(clean) / len(clean) if clean else None


def _fmt(value: float | None, fmt_str: str = ".4f", fallback: str = "—") -> str:
    """Format a float safely, returning fallback for None/inf/NaN."""
    if value is None or not math.isfinite(value):
        return fallback
    return f"{value:{fmt_str}}"


# ═══════════════════════════════════════════════════════════════════════════════
# Parsing
# ═══════════════════════════════════════════════════════════════════════════════


def resolve_dirs(dir_names: list[str]) -> list[Path]:
    """Resolve directory names to full paths under results/experiments/."""
    base = ROOT / "results" / "experiments"
    resolved = []
    for name in dir_names:
        # Accept full path or just folder name
        p = Path(name) if Path(name).is_absolute() else base / name
        resolved.append(p)
    return resolved


def load_runs(dirs: list[Path]) -> list[dict[str, Any]]:
    """Load all run_*.json from the given directories with dedup.

    Uses recursive glob (rglob) to support nested folder structures
    like exp_noiseless/tfim/heavy_hex/run_*.json.
    """
    from qmbp_simulation.framework.result_io import load_result

    results = []
    seen_files: set[str] = set()
    for d in dirs:
        if not d.exists():
            print(f"  [WARN] Directory not found: {d}")
            continue
        if not d.is_dir():
            print(f"  [WARN] Not a directory: {d}")
            continue
        for f in sorted(d.rglob("run_*.json")):
            abs_key = str(f.resolve())
            if abs_key in seen_files:
                continue
            seen_files.add(abs_key)
            try:
                data = load_result(f)
                data["_source_file"] = str(f.name)
                data["_source_dir"] = d.name
                data["_source_path"] = str(f)
                results.append(data)
            except (ValueError, OSError) as e:
                print(f"  [ERROR] Cannot load {f.name}: {e}")
            except json.JSONDecodeError as e:
                print(f"  [ERROR] Invalid JSON in {f.name}: {e}")
    return results


def parse_run(data: dict[str, Any]) -> RunMetrics | None:
    """Parse a raw JSON dict into RunMetrics. Returns None on critical failure."""
    try:
        cfg = data.get("config", {})
        sys_cfg = cfg.get("system", {})
        results = data.get("results", {})

        topos = sys_cfg.get("topologies", ["unknown"])
        topo = topos[0] if isinstance(topos, list) and topos else "unknown"

        s1 = results.get("section_1", {}) or {}
        s2 = results.get("section_2", {}) or {}
        s3 = results.get("section_3", {}) or {}
        s4 = results.get("section_4", {}) or {}

        s1_data = s1.get("data", {}) or {}
        s2_data = s2.get("data", {}) or {}
        s3_data = s3.get("data", {}) or {}
        s4_data = s4.get("data", {}) or {}

        # Section 1 topology data
        topo_s1 = s1_data.get("topologies", {}).get(topo, {})
        topo_s2 = s2_data.get("topologies", {}).get(topo, {})

        # Section 3 MSE summary
        mse_sum = s3_data.get("mse_summary", {}) or {}

        # h_grid info
        h_grid = cfg.get("h_grid", {})
        h_points = h_grid.get("h_points", topo_s2.get("n_points", 0))

        m = RunMetrics(
            source_file=data.get("_source_file", "?"),
            experiment=data.get("_source_dir", cfg.get("experiment_id", "?")),
            timestamp=data.get("timestamp", ""),
            model=sys_cfg.get("model", "unknown"),
            n_qubits=sys_cfg.get("n_qubits", 0),
            p_layers=sys_cfg.get("p_layers", 0),
            topology=topo,
            elapsed_s=data.get("elapsed_s", 0.0),
            h_points=h_points,
            s1_pass=s1.get("success", False),
            s2_pass=s2.get("success", False),
            s3_pass=s3.get("success") if s3 else None,
            s4_pass=s4.get("success") if s4 else None,
            # S1
            gap_min=topo_s1.get("gap_min"),
            gap_max=topo_s1.get("gap_max"),
            # S2
            vqe_n_pass=topo_s2.get("n_pass_5pct", 0),
            vqe_n_total=topo_s2.get("n_points", 0),
            mean_fidelity=topo_s2.get("mean_fidelity"),
            min_fidelity=topo_s2.get("min_fidelity"),
            mean_de_gap_vqe=topo_s2.get("mean_de_gap"),
            max_de_gap_vqe=topo_s2.get("max_de_gap"),
            n_converged=topo_s2.get("n_converged"),
            vqe_time_s=topo_s2.get("total_time_s", s2.get("elapsed_s", 0.0)),
            theta_smoothness_max=topo_s2.get("theta_smoothness_max"),
            n_variational_violations=topo_s2.get("n_variational_violations", 0),
            # S3
            mpnn_final_mse=mse_sum.get("final", s3_data.get("final_mse")),
            mpnn_best_mse=mse_sum.get("best"),
            mpnn_n_epochs=mse_sum.get("n_epochs_total"),
            mpnn_stopped_early=s3_data.get("stopped_early"),
            mpnn_n_params=s3_data.get("n_output_params"),
            # S4
            deploy_n_pass=s4_data.get("n_pass_energy"),
            deploy_n_total=s4_data.get("n_test_points"),
            deploy_n_correct_label=s4_data.get("n_correct_label"),
            deploy_mean_de_gap=s4_data.get("mean_de_gap"),
            deploy_max_de_gap=s4_data.get("max_de_gap"),
            deploy_mean_fidelity=s4_data.get("mean_fidelity"),
            deploy_speedup=s4_data.get("speedup_factor"),
            deploy_mag_x_error=s4_data.get("mean_mag_x_error"),
            deploy_corr_zz_error=s4_data.get("mean_corr_zz_error"),
        )
        return m
    except (KeyError, TypeError, ValueError, IndexError) as e:
        fname = data.get("_source_file", "unknown")
        print(f"  [ERROR] Failed to parse {fname}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation & Analysis
# ═══════════════════════════════════════════════════════════════════════════════


def group_runs(runs: list[RunMetrics]) -> dict[tuple[str, str, int], list[RunMetrics]]:
    """Group runs by (model, topology, p_layers)."""
    groups: dict[tuple[str, str, int], list[RunMetrics]] = defaultdict(list)
    for r in runs:
        groups[(r.model, r.topology, r.p_layers)].append(r)
    return groups


def compute_group_stats(key: tuple[str, str, int], runs: list[RunMetrics]) -> GroupStats:
    """Compute aggregate stats for a group."""
    model, topo, p = key
    gs = GroupStats(model=model, topology=topo, p_layers=p, n_runs=len(runs))

    gs.n_full_pass = sum(1 for r in runs if r.s4_pass is True)
    gs.n_vqe_pass = sum(1 for r in runs if r.s2_pass is True)
    gs.n_deploy_pass = sum(1 for r in runs if r.s4_pass is True)
    gs.mean_elapsed_s = _safe_mean([r.elapsed_s for r in runs]) or 0.0

    # VQE aggregates
    gs.mean_fidelity_avg = _safe_mean([r.mean_fidelity for r in runs])
    gs.mean_de_gap_vqe_avg = _safe_mean([r.mean_de_gap_vqe for r in runs])
    conv_rates = []
    for r in runs:
        if r.vqe_n_total > 0:
            conv_rates.append(r.vqe_n_pass / r.vqe_n_total)
    gs.vqe_converge_rate = _safe_mean(conv_rates)

    # Deploy aggregates
    deploy_rates: list[float] = []
    label_rates: list[float] = []
    for r in runs:
        if r.deploy_n_total and r.deploy_n_total > 0:
            if r.deploy_n_pass is not None:
                deploy_rates.append(r.deploy_n_pass / r.deploy_n_total)
            if r.deploy_n_correct_label is not None:
                label_rates.append(r.deploy_n_correct_label / r.deploy_n_total)
    gs.deploy_pass_rate = _safe_mean(deploy_rates)
    gs.deploy_label_rate = _safe_mean(label_rates)
    gs.deploy_mean_de_gap_avg = _safe_mean([r.deploy_mean_de_gap for r in runs])
    gs.deploy_mean_fidelity_avg = _safe_mean([r.deploy_mean_fidelity for r in runs])

    # Identify issues
    if gs.vqe_converge_rate is not None and gs.vqe_converge_rate < 0.5:
        gs.issues.append(f"Low VQE convergence: {gs.vqe_converge_rate:.1%}")
    if gs.deploy_pass_rate is not None and gs.deploy_pass_rate < 0.9:
        gs.issues.append(f"Deploy pass rate below 90%: {gs.deploy_pass_rate:.1%}")
    if gs.mean_fidelity_avg is not None and gs.mean_fidelity_avg < 0.90:
        gs.issues.append(f"Low mean fidelity: {gs.mean_fidelity_avg:.4f}")

    return gs


# ═══════════════════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════════════════


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


def print_run_table(runs: list[RunMetrics], verbose: bool = False) -> None:
    """Print per-run summary table."""
    print(f"\n{'─' * 70}")
    print(
        f"  {'File':<30} {'Model':<22} {'Topo':<10} "
        f"{'p':>2} {'N':>3} {'VQE':>5} {'Dpl':>7} {'t(s)':>7}"
    )
    print(f"{'─' * 70}")
    for r in runs:
        vqe_str = f"{r.vqe_n_pass}/{r.vqe_n_total}" if r.vqe_n_total else "—"
        if r.deploy_n_pass is not None and r.deploy_n_total:
            dpl_str = f"{r.deploy_n_pass}/{r.deploy_n_total}"
        else:
            dpl_str = "FAIL" if r.s4_pass is False else "—"
        status = "✓" if r.s4_pass else ("✗" if r.s4_pass is False else "?")
        print(
            f"  {status} {r.source_file:<28} {r.model:<22} {r.topology:<10} "
            f"{r.p_layers:>2} {r.n_qubits:>3} {vqe_str:>5} {dpl_str:>7} "
            f"{r.elapsed_s:>7.0f}"
        )
        if verbose:
            _print_run_detail(r)


def _print_run_detail(r: RunMetrics) -> None:
    """Print verbose detail for a single run (safe against None values)."""
    if r.deploy_mean_de_gap is not None:
        fid = _fmt(r.mean_fidelity)
        deg = _fmt(r.mean_de_gap_vqe)
        viol_str = f"  ⚠️viol={r.n_variational_violations}" if r.n_variational_violations > 0 else ""
        print(f"    ├─ VQE: F̄={fid}  ΔE/gap={deg}{viol_str}")
        mse = _fmt(r.mpnn_final_mse, ".2e", "N/A")
        epochs = r.mpnn_n_epochs if r.mpnn_n_epochs is not None else "?"
        params = r.mpnn_n_params if r.mpnn_n_params is not None else "?"
        print(f"    ├─ MPNN: MSE={mse}  epochs={epochs}  params={params}")
        d_deg = _fmt(r.deploy_mean_de_gap)
        d_max = _fmt(r.deploy_max_de_gap)
        d_fid = _fmt(r.deploy_mean_fidelity)
        labels = (
            f"{r.deploy_n_correct_label}/{r.deploy_n_total}"
            if (r.deploy_n_correct_label is not None and r.deploy_n_total)
            else "?"
        )
        print(f"    └─ Deploy: ΔE/gap={d_deg} (max={d_max})  F̄={d_fid}  labels={labels}")
    elif r.s2_pass is False:
        fid = _fmt(r.mean_fidelity, ".4f", "0.0000")
        conv = r.n_converged if r.n_converged is not None else "?"
        total = r.vqe_n_total if r.vqe_n_total else "?"
        print(f"    └─ VQE FAILED: F̄={fid}  converged={conv}/{total}")
    elif r.s3_pass is False:
        mse = _fmt(r.mpnn_final_mse, ".2e", "N/A")
        print(f"    └─ MPNN FAILED: MSE={mse}")


def print_group_summary(groups: dict[tuple[str, str, int], GroupStats]) -> None:
    """Print per-group aggregate stats."""
    print_header("GROUP SUMMARY (model / topology / p)")
    print(
        f"\n  {'Model':<22} {'Topo':<10} {'p':>2} {'Runs':>4} "
        f"{'VQE✓':>5} {'Dpl✓':>5} {'F̄_vqe':>7} {'ΔE/gap_d':>9} "
        f"{'DplRate':>7} {'LblRate':>7}"
    )
    print(f"  {'─' * 95}")
    for key in sorted(groups.keys()):
        gs = groups[key]
        fid = _fmt(gs.mean_fidelity_avg)
        deg = _fmt(gs.deploy_mean_de_gap_avg)
        dpr = f"{gs.deploy_pass_rate:.1%}" if gs.deploy_pass_rate is not None else "—"
        lbl = f"{gs.deploy_label_rate:.1%}" if gs.deploy_label_rate is not None else "—"
        print(
            f"  {gs.model:<22} {gs.topology:<10} {gs.p_layers:>2} {gs.n_runs:>4} "
            f"{gs.n_vqe_pass:>5} {gs.n_deploy_pass:>5} {fid:>7} {deg:>9} "
            f"{dpr:>7} {lbl:>7}"
        )
        if gs.issues:
            for issue in gs.issues:
                print(f"    ⚠ {issue}")


def print_model_deep_dive(model: str, runs: list[RunMetrics]) -> None:
    """Detailed breakdown for a single model."""
    passing = [r for r in runs if r.s4_pass is True]
    failing = [r for r in runs if r.s4_pass is not True]

    print(f"\n  Total runs: {len(runs)}  |  Pass: {len(passing)}  |  Fail: {len(failing)}")

    if passing:
        print("\n  Passing runs — Deploy metrics:")
        de_gaps = [
            r.deploy_mean_de_gap
            for r in passing
            if r.deploy_mean_de_gap is not None and math.isfinite(r.deploy_mean_de_gap)
        ]
        fids = [
            r.deploy_mean_fidelity
            for r in passing
            if r.deploy_mean_fidelity is not None and math.isfinite(r.deploy_mean_fidelity)
        ]
        speedups = [
            r.deploy_speedup
            for r in passing
            if r.deploy_speedup is not None and math.isfinite(r.deploy_speedup)
        ]
        if de_gaps:
            print(
                f"    ΔE/gap mean: {sum(de_gaps) / len(de_gaps):.5f}  "
                f"min: {min(de_gaps):.5f}  max: {max(de_gaps):.5f}"
            )
        if fids:
            print(
                f"    Fidelity:    {sum(fids) / len(fids):.5f}  "
                f"min: {min(fids):.5f}  max: {max(fids):.5f}"
            )
        if speedups:
            print(
                f"    Speedup:     {sum(speedups) / len(speedups):.1f}×  "
                f"min: {min(speedups):.1f}×  max: {max(speedups):.1f}×"
            )

        # Best/worst deploy by ΔE/gap
        by_de = sorted(passing, key=lambda r: r.deploy_mean_de_gap or float("inf"))
        best = by_de[0]
        worst = by_de[-1]
        print(
            f"\n    Best:  {best.source_file} (p={best.p_layers}, {best.topology}) "
            f"ΔE/gap={_fmt(best.deploy_mean_de_gap, '.5f')}"
        )
        if worst is not best:
            print(
                f"    Worst: {worst.source_file} (p={worst.p_layers}, {worst.topology}) "
                f"ΔE/gap={_fmt(worst.deploy_mean_de_gap, '.5f')}"
            )

    if failing:
        print("\n  Failing runs — failure points:")
        for r in failing:
            reason = _classify_failure(r)
            print(f"    ✗ {r.source_file} p={r.p_layers} {r.topology}: {reason}")


def _classify_failure(r: RunMetrics) -> str:
    """Classify the failure reason for a non-passing run."""
    if not r.s1_pass:
        return "ExactDiag failed"
    if not r.s2_pass:
        fid = _fmt(r.mean_fidelity, ".4f", "0.0000")
        conv = r.n_converged if r.n_converged is not None else "?"
        total = r.vqe_n_total if r.vqe_n_total else "?"
        return f"VQE fail (converged={conv}/{total}, F̄={fid})"
    if r.s3_pass is False:
        mse = _fmt(r.mpnn_final_mse, ".2e", "N/A")
        return f"MPNN fail (MSE={mse})"
    if r.s3_pass is None:
        return "S3 not executed"
    if r.s4_pass is False:
        dg = _fmt(r.deploy_mean_de_gap, ".4f", "N/A")
        n_pass = r.deploy_n_pass if r.deploy_n_pass is not None else "?"
        n_total = r.deploy_n_total if r.deploy_n_total else "?"
        return f"Deploy fail ({n_pass}/{n_total} pass, ΔE/gap={dg})"
    if r.s4_pass is None:
        return "S4 not executed"
    return "Unknown"


def print_cross_model_comparison(all_groups: dict[tuple[str, str, int], GroupStats]) -> None:
    """Compare models at same topology/p."""
    print_header("CROSS-MODEL COMPARISON (same topology & p)")

    by_topo_p: dict[tuple[str, int], list[GroupStats]] = defaultdict(list)
    for gs in all_groups.values():
        by_topo_p[(gs.topology, gs.p_layers)].append(gs)

    for (topo, p), group_list in sorted(by_topo_p.items()):
        if len(group_list) < 2:
            continue
        print(f"\n  {topo} p={p}:")
        for gs in sorted(group_list, key=lambda g: g.deploy_mean_de_gap_avg or float("inf")):
            dg = _fmt(gs.deploy_mean_de_gap_avg, ".5f", "N/A")
            dr = f"{gs.deploy_pass_rate:.0%}" if gs.deploy_pass_rate is not None else "N/A"
            print(
                f"    {gs.model:<22} ΔE/gap={dg}  pass_rate={dr}  "
                f"({gs.n_deploy_pass}/{gs.n_runs} full pass)"
            )


def print_p_layer_analysis(runs: list[RunMetrics]) -> None:
    """Analyse impact of p_layers on performance within each model."""
    print_header("P-LAYER SCALING ANALYSIS")

    by_model: dict[str, list[RunMetrics]] = defaultdict(list)
    for r in runs:
        by_model[r.model].append(r)

    for model, model_runs in sorted(by_model.items()):
        print(f"\n  {model}:")
        by_p: dict[int, list[RunMetrics]] = defaultdict(list)
        for r in model_runs:
            by_p[r.p_layers].append(r)

        for p in sorted(by_p.keys()):
            p_runs = by_p[p]
            passing = [r for r in p_runs if r.s4_pass is True]
            fids = [
                r.mean_fidelity
                for r in p_runs
                if r.mean_fidelity is not None and math.isfinite(r.mean_fidelity)
            ]
            de_gaps_d = [
                r.deploy_mean_de_gap
                for r in passing
                if r.deploy_mean_de_gap is not None and math.isfinite(r.deploy_mean_de_gap)
            ]
            avg_fid = sum(fids) / len(fids) if fids else 0.0
            avg_dg = sum(de_gaps_d) / len(de_gaps_d) if de_gaps_d else None
            dg_str = f"{avg_dg:.5f}" if avg_dg is not None else "N/A"
            avg_time = sum(r.elapsed_s for r in p_runs) / len(p_runs)
            print(
                f"    p={p}: {len(passing)}/{len(p_runs)} pass  "
                f"F̄_vqe={avg_fid:.4f}  ΔE/gap_deploy={dg_str}  "
                f"avg_time={avg_time:.0f}s"
            )


def build_json_report(
    runs: list[RunMetrics],
    all_groups: dict[tuple[str, str, int], GroupStats],
) -> dict[str, Any]:
    """Build a JSON-serializable report."""
    return {
        "summary": {
            "n_runs_total": len(runs),
            "n_full_pass": sum(1 for r in runs if r.s4_pass is True),
            "n_vqe_fail": sum(1 for r in runs if not r.s2_pass),
            "n_s3_fail": sum(1 for r in runs if r.s3_pass is False),
            "n_s4_fail": sum(1 for r in runs if r.s4_pass is False),
            "models": sorted(set(r.model for r in runs)),
            "topologies": sorted(set(r.topology for r in runs)),
            "p_layers": sorted(set(r.p_layers for r in runs)),
            "h_points": sorted(set(r.h_points for r in runs)),
        },
        "groups": {f"{k[0]}_{k[1]}_p{k[2]}": asdict(v) for k, v in sorted(all_groups.items())},
        "runs": [asdict(r) for r in runs],
    }


def filter_runs(
    runs: list[RunMetrics],
    model: str | None = None,
    topology: str | None = None,
    p_layers: int | None = None,
) -> list[RunMetrics]:
    """Filter runs by model, topology, or p_layers."""
    filtered = runs
    if model:
        filtered = [r for r in filtered if model.lower() in r.model.lower()]
    if topology:
        filtered = [r for r in filtered if topology.lower() in r.topology.lower()]
    if p_layers is not None:
        filtered = [r for r in filtered if r.p_layers == p_layers]
    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyse noiseless experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  python scripts/scan_new_runs.py --dirs exp_noiseless_tfim_4\n"
        "  python scripts/scan_new_runs.py --filter-model tfim --filter-p 4\n"
        "  python scripts/scan_new_runs.py --json report.json --verbose\n",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-run detail")
    parser.add_argument("--json", type=str, default=None, help="Write JSON report to this path")
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=None,
        help="Experiment folder names (under results/experiments/)",
    )
    parser.add_argument(
        "--filter-model", type=str, default=None, help="Filter by model name (substring match)"
    )
    parser.add_argument(
        "--filter-topo", type=str, default=None, help="Filter by topology (substring match)"
    )
    parser.add_argument(
        "--filter-p", type=int, default=None, help="Filter by p_layers (exact match)"
    )
    args = parser.parse_args()

    # Resolve directories
    dir_names = args.dirs if args.dirs else DEFAULT_EXPERIMENT_DIRS
    experiment_dirs = resolve_dirs(dir_names)

    # Load data
    print_header(f"LOADING NOISELESS EXPERIMENTS ({len(experiment_dirs)} dirs)")
    raw_data = load_runs(experiment_dirs)
    print(f"\n  Loaded {len(raw_data)} run files from {len(experiment_dirs)} directories")

    if not raw_data:
        print("\n  No data found. Exiting.")
        return 1

    # Parse
    runs: list[RunMetrics] = []
    n_parse_errors = 0
    for d in raw_data:
        m = parse_run(d)
        if m:
            runs.append(m)
        else:
            n_parse_errors += 1

    print(f"  Parsed {len(runs)} runs successfully", end="")
    if n_parse_errors:
        print(f" ({n_parse_errors} parse errors)")
    else:
        print()

    # Apply filters
    if args.filter_model or args.filter_topo or args.filter_p is not None:
        pre_filter = len(runs)
        runs = filter_runs(runs, args.filter_model, args.filter_topo, args.filter_p)
        print(f"  Filtered: {pre_filter} → {len(runs)} runs")
        if not runs:
            print("  No runs match the filter. Exiting.")
            return 1

    # Per-experiment breakdown
    by_experiment: dict[str, list[RunMetrics]] = defaultdict(list)
    for r in runs:
        by_experiment[r.experiment].append(r)

    for exp_name, exp_runs in sorted(by_experiment.items()):
        print_header(f"{exp_name} ({len(exp_runs)} runs)")
        print_run_table(exp_runs, verbose=args.verbose)

        # Model deep dives
        models_in_exp = sorted(set(r.model for r in exp_runs))
        for model in models_in_exp:
            model_runs = [r for r in exp_runs if r.model == model]
            print_model_deep_dive(model, model_runs)

    # Group aggregation
    grouped = group_runs(runs)
    all_groups = {k: compute_group_stats(k, v) for k, v in grouped.items()}
    print_group_summary(all_groups)

    # Cross-model comparison
    print_cross_model_comparison(all_groups)

    # P-layer analysis
    print_p_layer_analysis(runs)

    # Final verdict
    print_header("OVERALL VERDICT")
    n_pass = sum(1 for r in runs if r.s4_pass is True)
    n_fail = sum(1 for r in runs if r.s4_pass is False)
    n_partial = len(runs) - n_pass - n_fail
    print(f"\n  Total: {len(runs)} runs")
    if runs:
        print(f"  ✓ Full pass (S1→S4): {n_pass} ({n_pass / len(runs):.0%})")
        print(f"  ✗ Failed:            {n_fail} ({n_fail / len(runs):.0%})")
        if n_partial:
            print(f"  ? Partial/unknown:   {n_partial}")
    else:
        print("  No runs to analyse.")

    # JSON output
    if args.json:
        report = build_json_report(runs, all_groups)
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fp:
            json.dump(report, fp, indent=2, default=str)
        print(f"\n  JSON report written to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
