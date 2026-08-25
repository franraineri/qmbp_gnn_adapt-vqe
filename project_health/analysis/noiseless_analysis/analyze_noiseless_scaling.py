#!/usr/bin/env python3
"""Noiseless Scaling Analysis — Multi-axis per-h study.

Analyzes all successful noiseless pipeline runs across 4 axes:
  1. h-dependence: ΔE/gap vs h for each config
  2. p-dependence: metrics vs p for fixed (N, topo)
  3. N-dependence: metrics vs N for fixed (p, topo)
  4. Topology comparison: metrics across topologies for fixed (N, p)

Uses per-h-point data from Section 4 (Deploy) of each run.

Usage:
    # Full analysis (all successful runs with deploy data)
    .venv/bin/python scripts/analyze_noiseless_scaling.py

    # Filter by model
    .venv/bin/python scripts/analyze_noiseless_scaling.py --model tfim

    # Only scaling axis (N-dependence)
    .venv/bin/python scripts/analyze_noiseless_scaling.py --axis n-scaling

    # JSON output for further processing
    .venv/bin/python scripts/analyze_noiseless_scaling.py --json results/scaling_analysis.json

    # Verbose (per-h tables)
    .venv/bin/python scripts/analyze_noiseless_scaling.py -v
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qmbp_simulation.framework.result_io import load_result

# ═══════════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PerHPoint:
    """Single h-point result from deploy."""

    h: float
    de_gap: float
    fidelity: float
    label_correct: bool
    de_gap_random: float | None = None
    mpnn_wins: bool = False


@dataclass
class RunSummary:
    """Summary of one pipeline run with per-h data."""

    file: str
    model: str
    topology: str
    n_qubits: int
    p_layers: int
    h_min: float
    h_max: float
    h_points: int
    elapsed_s: float
    # VQE metrics
    vqe_pass_rate: float | None = None
    vqe_mean_fidelity: float | None = None
    theta_smoothness: float | None = None
    # MPNN metrics
    mpnn_mse: float | None = None
    # Deploy aggregate
    deploy_pass_rate: float | None = None
    deploy_mean_de_gap: float | None = None
    deploy_max_de_gap: float | None = None
    deploy_mean_fidelity: float | None = None
    deploy_speedup: float | None = None
    mpnn_wins_total: int = 0
    mpnn_wins_rate: float | None = None
    # Per-h data
    per_h: list[PerHPoint] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Loading & Parsing
# ═══════════════════════════════════════════════════════════════════════════════

NOISELESS_DIRS = [
    "exp_noiseless",
    "exp_noiseless_tfim_v2",
    "exp_noiseless_tfim_v3",
    "exp_noiseless_tfim_4",
    "exp_noiseless_tfim_longitudinal",
    "exp_noiseless_tfim_longitudinal_v2",
    "exp_noiseless_tfim_longitudinal_v3",
    "exp_noiseless_tfim_longitudinal_4",
    "exp_noiseless_heisenberg_v2",
    "exp_noiseless_heisenberg_transverse_v2",
    "exp_noiseless_heisenberg_transverse_v3",
    "exp_noiseless_heisenberg_transverse_4",
]


def find_all_runs(model_filter: str | None = None) -> list[Path]:
    """Find run files using ResultIndex for fast pre-filtering.

    Uses the index to identify runs with ≥4 sections (have deploy data)
    and optionally filter by model before touching disk. Falls back to
    glob scan if index is unavailable.
    """
    base = ROOT / "results" / "experiments"

    # Try index-based fast path
    try:
        from qmbp_simulation.framework.result_index import ResultIndex

        index = ResultIndex(base)
        # Query: noiseless runs with deploy data (n_sections >= 4)
        entries = index.query(model=model_filter) if model_filter else index.query()
        # Filter to noiseless experiment_ids and runs with section_4
        noiseless_files = []
        for entry in entries:
            eid = entry.get("experiment_id", "")
            n_sec = entry.get("n_sections", 0)
            if n_sec < 4:
                continue
            # Only include noiseless runs
            if "noiseless" not in eid.lower() and not any(
                d in entry.get("_file", "") for d in NOISELESS_DIRS
            ):
                continue
            fpath = base / entry["_file"]
            if fpath.exists():
                noiseless_files.append(fpath)

        if noiseless_files:
            return sorted(set(noiseless_files))
    except Exception:
        pass  # Fall back to glob scan

    # Fallback: disk glob (slower but always works)
    seen: set[str] = set()
    files: list[Path] = []
    for d in NOISELESS_DIRS:
        exp_dir = base / d
        if not exp_dir.exists():
            continue
        for f in sorted(exp_dir.rglob("run_*.json")):
            key = str(f.resolve())
            if key not in seen:
                seen.add(key)
                files.append(f)
    return files


def parse_run(path: Path) -> RunSummary | None:
    """Parse a run file into RunSummary with per-h data."""
    try:
        data = load_result(path)
    except Exception:
        return None

    cfg = data.get("config", {})
    system = cfg.get("system", {})
    h_grid = cfg.get("h_grid", {})
    results = data.get("results", {})
    summary = data.get("summary", {})

    # Must have section 4 (deploy) with per-point data
    s4 = results.get("section_4", {})
    if not s4 or not s4.get("data"):
        return None

    s4_data = s4["data"]
    per_point = s4_data.get("per_point", [])
    if not per_point:
        return None

    model = system.get("model", "")
    topos = system.get("topologies", [])
    topology = topos[0] if topos else ""
    n_qubits = system.get("n_qubits", 0)
    p_layers = system.get("p_layers", 0)

    if not model or not topology or not n_qubits:
        return None

    # Parse per-h points
    per_h: list[PerHPoint] = []
    for pp in per_point:
        h = pp.get("h_test", pp.get("h", 0))
        de_gap = pp.get("de_gap", pp.get("delta_e_gap", None))
        fid = pp.get("fidelity", pp.get("f", None))
        label = pp.get("label_correct", pp.get("correct_label", None))
        de_gap_rand = pp.get("de_gap_random", pp.get("de_gap_random_init", None))

        if de_gap is None or fid is None:
            continue

        if label is None:
            # Infer from phase
            label = pp.get("pass_energy", False) and pp.get("pass_label", True)

        per_h.append(
            PerHPoint(
                h=h,
                de_gap=de_gap,
                fidelity=fid,
                label_correct=bool(label),
                de_gap_random=de_gap_rand,
                mpnn_wins=bool(de_gap_rand and de_gap < de_gap_rand),
            )
        )

    if not per_h:
        return None

    # VQE metrics from section 2
    s2_data = results.get("section_2", {}).get("data", {})
    vqe_topo = s2_data.get("topologies", {}).get(topology, {})
    vqe_pass_rate = None
    vqe_mean_fid = None
    theta_smooth = None
    if vqe_topo:
        n_pass = vqe_topo.get("n_pass", 0)
        n_total = vqe_topo.get("n_total", 0)
        vqe_pass_rate = n_pass / n_total if n_total > 0 else None
        vqe_mean_fid = vqe_topo.get("mean_fidelity")
        theta_smooth = vqe_topo.get("theta_smoothness_max")

    # MPNN from section 3
    s3_data = results.get("section_3", {}).get("data", {})
    mpnn_mse = s3_data.get("final_mse") or s3_data.get("best_val_mse")

    # Deploy aggregates
    n_pass_e = s4_data.get("n_pass_energy", 0)
    n_test = s4_data.get("n_test_points", len(per_h))
    n_labels = s4_data.get("n_correct_label", 0)
    mpnn_wins_total = s4_data.get("mpnn_wins_vs_random", 0)

    return RunSummary(
        file=str(path.relative_to(ROOT)),
        model=model,
        topology=topology,
        n_qubits=n_qubits,
        p_layers=p_layers,
        h_min=h_grid.get("h_min", min(p.h for p in per_h)),
        h_max=h_grid.get("h_max", max(p.h for p in per_h)),
        h_points=h_grid.get("h_points", len(per_h)),
        elapsed_s=data.get("elapsed_s", 0),
        vqe_pass_rate=vqe_pass_rate,
        vqe_mean_fidelity=vqe_mean_fid,
        theta_smoothness=theta_smooth,
        mpnn_mse=mpnn_mse,
        deploy_pass_rate=n_pass_e / n_test if n_test > 0 else None,
        deploy_mean_de_gap=s4_data.get("mean_de_gap"),
        deploy_max_de_gap=s4_data.get("max_de_gap"),
        deploy_mean_fidelity=s4_data.get("mean_fidelity"),
        deploy_speedup=s4_data.get("speedup_factor"),
        mpnn_wins_total=mpnn_wins_total or 0,
        mpnn_wins_rate=mpnn_wins_total / n_test if n_test > 0 and mpnn_wins_total else None,
        per_h=per_h,
    )


def load_all_runs(model_filter: str | None = None) -> list[RunSummary]:
    """Load and parse all noiseless runs with deploy data."""
    files = find_all_runs(model_filter=model_filter)
    runs: list[RunSummary] = []
    for f in files:
        r = parse_run(f)
        if r is None:
            continue
        if model_filter and model_filter.lower() not in r.model.lower():
            continue
        runs.append(r)
    return runs


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis Axes
# ═══════════════════════════════════════════════════════════════════════════════


def axis_h_dependence(runs: list[RunSummary], verbose: bool = False) -> dict:
    """Axis 1: How does ΔE/gap behave as h decreases, per config?"""
    print("\n" + "═" * 70)
    print("  AXIS 1: h-DEPENDENCE (ΔE/gap vs h)")
    print("═" * 70)

    # Group by (model, topology, N, p) — use best run per group
    groups: dict[tuple, RunSummary] = {}
    for r in runs:
        key = (r.model, r.topology, r.n_qubits, r.p_layers)
        existing = groups.get(key)
        if existing is None or (r.deploy_pass_rate or 0) > (existing.deploy_pass_rate or 0):
            groups[key] = r

    # For each group, find the h where ΔE/gap crosses 5%
    results_data: list[dict] = []
    print(f"\n  {'Config':<40s} | h_boundary | mean_dE(h>hb) | mean_dE(h<hb) | n_pts")
    print("  " + "-" * 90)

    for key in sorted(groups.keys()):
        r = groups[key]
        model, topo, n, p = key
        points_sorted = sorted(r.per_h, key=lambda x: x.h, reverse=True)

        # Find boundary: lowest h where ΔE/gap < 0.05
        passing = [pt for pt in points_sorted if pt.de_gap < 0.05]
        failing = [pt for pt in points_sorted if pt.de_gap >= 0.05]

        h_boundary = min(pt.h for pt in passing) if passing else None
        mean_above = sum(pt.de_gap for pt in passing) / len(passing) if passing else None
        mean_below = sum(pt.de_gap for pt in failing) / len(failing) if failing else None

        label = f"{model} {topo} N={n} p={p}"
        hb_str = f"{h_boundary:.3f}" if h_boundary else "—"
        ma_str = f"{mean_above:.4f}" if mean_above is not None else "—"
        mb_str = f"{mean_below:.4f}" if mean_below is not None else "—"
        print(
            f"  {label:<40s} | {hb_str:>10s} | {ma_str:>13s} | {mb_str:>13s} | {len(points_sorted):>5d}"
        )

        results_data.append(
            {
                "model": model,
                "topology": topo,
                "n_qubits": n,
                "p_layers": p,
                "h_boundary": h_boundary,
                "n_pass": len(passing),
                "n_fail": len(failing),
                "mean_de_gap_passing": mean_above,
                "mean_de_gap_failing": mean_below,
            }
        )

        if verbose and h_boundary:
            # Show per-h near boundary
            near = [pt for pt in points_sorted if abs(pt.h - h_boundary) < 0.3]
            for pt in sorted(near, key=lambda x: x.h):
                status = "✅" if pt.de_gap < 0.05 else "❌"
                print(f"      {status} h={pt.h:.3f} ΔE/gap={pt.de_gap:.4f} F={pt.fidelity:.4f}")

    return {"axis": "h_dependence", "configs": results_data}


def axis_p_dependence(runs: list[RunSummary], verbose: bool = False) -> dict:
    """Axis 2: How do metrics scale with p for fixed (N, topology)?"""
    print("\n" + "═" * 70)
    print("  AXIS 2: p-DEPENDENCE (metrics vs circuit depth)")
    print("═" * 70)

    # Group by (model, topology, N)
    by_config: dict[tuple, list[RunSummary]] = defaultdict(list)
    for r in runs:
        key = (r.model, r.topology, r.n_qubits)
        by_config[key].append(r)

    results_data: list[dict] = []

    for key in sorted(by_config.keys()):
        model, topo, n = key
        config_runs = by_config[key]

        # Best run per p
        best_per_p: dict[int, RunSummary] = {}
        for r in config_runs:
            existing = best_per_p.get(r.p_layers)
            if existing is None or (r.deploy_pass_rate or 0) > (existing.deploy_pass_rate or 0):
                best_per_p[r.p_layers] = r

        if len(best_per_p) < 2:
            continue  # Need multiple p values to compare

        print(f"\n  {model} | {topo} | N={n}")
        print(
            f"  {'p':>3s} | {'Deploy%':>8s} | {'ΔE/gap':>8s} | {'F̄':>6s} | {'θ_smooth':>9s} | {'MSE':>9s} | {'Speedup':>8s}"
        )
        print("  " + "-" * 70)

        p_data: list[dict] = []
        for p in sorted(best_per_p.keys()):
            r = best_per_p[p]
            dpr = f"{r.deploy_pass_rate * 100:.0f}%" if r.deploy_pass_rate is not None else "—"
            deg = f"{r.deploy_mean_de_gap:.4f}" if r.deploy_mean_de_gap is not None else "—"
            fid = f"{r.deploy_mean_fidelity:.4f}" if r.deploy_mean_fidelity is not None else "—"
            ts = f"{r.theta_smoothness:.2f}" if r.theta_smoothness is not None else "—"
            mse = f"{r.mpnn_mse:.1e}" if r.mpnn_mse is not None else "—"
            spd = f"{r.deploy_speedup:.0f}×" if r.deploy_speedup is not None else "—"
            print(
                f"  {p:>3d} | {dpr:>8s} | {deg:>8s} | {fid:>6s} | {ts:>9s} | {mse:>9s} | {spd:>8s}"
            )

            p_data.append(
                {
                    "p": p,
                    "deploy_rate": r.deploy_pass_rate,
                    "mean_de_gap": r.deploy_mean_de_gap,
                    "mean_fidelity": r.deploy_mean_fidelity,
                    "theta_smoothness": r.theta_smoothness,
                    "mpnn_mse": r.mpnn_mse,
                    "speedup": r.deploy_speedup,
                }
            )

        results_data.append(
            {
                "model": model,
                "topology": topo,
                "n_qubits": n,
                "p_scaling": p_data,
            }
        )

    return {"axis": "p_dependence", "configs": results_data}


def axis_n_scaling(runs: list[RunSummary], verbose: bool = False) -> dict:
    """Axis 3: How do metrics scale with N for fixed (p, topology)?"""
    print("\n" + "═" * 70)
    print("  AXIS 3: N-SCALING (metrics vs system size)")
    print("═" * 70)

    # Group by (model, topology, p)
    by_config: dict[tuple, list[RunSummary]] = defaultdict(list)
    for r in runs:
        key = (r.model, r.topology, r.p_layers)
        by_config[key].append(r)

    results_data: list[dict] = []

    for key in sorted(by_config.keys()):
        model, topo, p = key
        config_runs = by_config[key]

        # Best run per N
        best_per_n: dict[int, RunSummary] = {}
        for r in config_runs:
            existing = best_per_n.get(r.n_qubits)
            if existing is None or (r.deploy_pass_rate or 0) > (existing.deploy_pass_rate or 0):
                best_per_n[r.n_qubits] = r

        if len(best_per_n) < 2:
            continue

        print(f"\n  {model} | {topo} | p={p}")
        print(
            f"  {'N':>4s} | {'Deploy%':>8s} | {'ΔE/gap':>8s} | {'max_dE':>8s} | {'F̄':>6s} | {'Speedup':>8s} | {'Time':>8s}"
        )
        print("  " + "-" * 72)

        n_data: list[dict] = []
        for n in sorted(best_per_n.keys()):
            r = best_per_n[n]
            dpr = f"{r.deploy_pass_rate * 100:.0f}%" if r.deploy_pass_rate is not None else "—"
            deg = f"{r.deploy_mean_de_gap:.4f}" if r.deploy_mean_de_gap is not None else "—"
            maxd = f"{r.deploy_max_de_gap:.4f}" if r.deploy_max_de_gap is not None else "—"
            fid = f"{r.deploy_mean_fidelity:.4f}" if r.deploy_mean_fidelity is not None else "—"
            spd = f"{r.deploy_speedup:.0f}×" if r.deploy_speedup is not None else "—"
            time_str = f"{r.elapsed_s / 3600:.1f}h" if r.elapsed_s > 3600 else f"{r.elapsed_s:.0f}s"
            print(
                f"  {n:>4d} | {dpr:>8s} | {deg:>8s} | {maxd:>8s} | {fid:>6s} | {spd:>8s} | {time_str:>8s}"
            )

            n_data.append(
                {
                    "n_qubits": n,
                    "deploy_rate": r.deploy_pass_rate,
                    "mean_de_gap": r.deploy_mean_de_gap,
                    "max_de_gap": r.deploy_max_de_gap,
                    "mean_fidelity": r.deploy_mean_fidelity,
                    "speedup": r.deploy_speedup,
                    "elapsed_s": r.elapsed_s,
                }
            )

        results_data.append(
            {
                "model": model,
                "topology": topo,
                "p_layers": p,
                "n_scaling": n_data,
            }
        )

    return {"axis": "n_scaling", "configs": results_data}


def axis_topology_comparison(runs: list[RunSummary], verbose: bool = False) -> dict:
    """Axis 4: Compare topologies for fixed (model, N, p)."""
    print("\n" + "═" * 70)
    print("  AXIS 4: TOPOLOGY COMPARISON")
    print("═" * 70)

    # Group by (model, N, p)
    by_config: dict[tuple, list[RunSummary]] = defaultdict(list)
    for r in runs:
        key = (r.model, r.n_qubits, r.p_layers)
        by_config[key].append(r)

    results_data: list[dict] = []

    for key in sorted(by_config.keys()):
        model, n, p = key
        config_runs = by_config[key]

        # Best run per topology
        best_per_topo: dict[str, RunSummary] = {}
        for r in config_runs:
            existing = best_per_topo.get(r.topology)
            if existing is None or (r.deploy_pass_rate or 0) > (existing.deploy_pass_rate or 0):
                best_per_topo[r.topology] = r

        if len(best_per_topo) < 2:
            continue

        print(f"\n  {model} | N={n} | p={p}")
        print(
            f"  {'Topology':<12s} | {'Deploy%':>8s} | {'ΔE/gap':>8s} | {'F̄':>6s} | {'MPNN wins':>10s} | {'Speedup':>8s}"
        )
        print("  " + "-" * 70)

        topo_data: list[dict] = []
        for topo in ["chain_1d", "heavy_hex", "ladder", "square", "triangular"]:
            if topo not in best_per_topo:
                continue
            r = best_per_topo[topo]
            dpr = f"{r.deploy_pass_rate * 100:.0f}%" if r.deploy_pass_rate is not None else "—"
            deg = f"{r.deploy_mean_de_gap:.4f}" if r.deploy_mean_de_gap is not None else "—"
            fid = f"{r.deploy_mean_fidelity:.4f}" if r.deploy_mean_fidelity is not None else "—"
            wins = f"{r.mpnn_wins_rate * 100:.0f}%" if r.mpnn_wins_rate is not None else "—"
            spd = f"{r.deploy_speedup:.0f}×" if r.deploy_speedup is not None else "—"
            print(f"  {topo:<12s} | {dpr:>8s} | {deg:>8s} | {fid:>6s} | {wins:>10s} | {spd:>8s}")

            topo_data.append(
                {
                    "topology": topo,
                    "deploy_rate": r.deploy_pass_rate,
                    "mean_de_gap": r.deploy_mean_de_gap,
                    "mean_fidelity": r.deploy_mean_fidelity,
                    "mpnn_wins_rate": r.mpnn_wins_rate,
                    "speedup": r.deploy_speedup,
                }
            )

        results_data.append(
            {
                "model": model,
                "n_qubits": n,
                "p_layers": p,
                "topologies": topo_data,
            }
        )

    return {"axis": "topology_comparison", "configs": results_data}


def print_global_summary(runs: list[RunSummary]) -> dict:
    """Print overall statistics."""
    print("\n" + "═" * 70)
    print("  GLOBAL SUMMARY")
    print("═" * 70)

    models = set(r.model for r in runs)
    topos = set(r.topology for r in runs)
    ns = sorted(set(r.n_qubits for r in runs))
    ps = sorted(set(r.p_layers for r in runs))

    full_pass = [r for r in runs if r.deploy_pass_rate is not None and r.deploy_pass_rate >= 0.80]
    perfect = [r for r in runs if r.deploy_pass_rate is not None and r.deploy_pass_rate >= 0.95]

    print(f"\n  Runs with deploy data: {len(runs)}")
    print(f"  Models: {', '.join(sorted(models))}")
    print(f"  Topologies: {', '.join(sorted(topos))}")
    print(f"  N values: {ns}")
    print(f"  p values: {ps}")
    print(f"  Deploy ≥80%: {len(full_pass)} runs")
    print(f"  Deploy ≥95%: {len(perfect)} runs")

    # Best overall
    if runs:
        best = max(runs, key=lambda r: (r.deploy_pass_rate or 0, -(r.deploy_mean_de_gap or 999)))
        print(f"\n  BEST: {best.model} {best.topology} N={best.n_qubits} p={best.p_layers}")
        print(
            f"        Deploy={best.deploy_pass_rate * 100:.0f}% ΔE/gap={best.deploy_mean_de_gap:.4f} "
            f"F̄={best.deploy_mean_fidelity:.4f} Speedup={best.deploy_speedup:.0f}×"
        )

    return {
        "n_runs": len(runs),
        "n_pass_80": len(full_pass),
        "n_pass_95": len(perfect),
        "models": sorted(models),
        "topologies": sorted(topos),
        "n_values": ns,
        "p_values": ps,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Speedup Scaling Fit (H5)
# ═══════════════════════════════════════════════════════════════════════════════


def fit_speedup_models(n_scaling_data: list[dict], verbose: bool = False) -> list[dict]:
    """Fit speedup vs N for each config using model selection.

    Tests three models per configuration:
      1. Power law: S = a * N^b
      2. Saturating: S = S_max * (1 - exp(-k*N))
      3. Constant: S = c

    Selects best model by AICc (corrected Akaike Information Criterion).

    Parameters
    ----------
    n_scaling_data : list[dict]
        Output from axis_n_scaling()["configs"]. Each entry has:
        - model, topology, p_layers
        - n_scaling: list of dicts with {n_qubits, speedup, ...}
    verbose : bool
        Print detailed fit info per config.

    Returns
    -------
    list[dict]
        Per-config fit results with best model, params, R², AICc.
    """
    import warnings as _warnings

    import numpy as np
    from scipy.optimize import curve_fit

    def _power_law(n, a, b):
        return a * np.power(n, b)

    def _saturating(n, s_max, k):
        return s_max * (1.0 - np.exp(-k * n))

    def _constant(n, c):
        return np.full_like(n, c, dtype=float)

    def _aicc(n_points: int, n_params: int, rss: float) -> float:
        """Corrected AIC for small samples."""
        if n_points <= n_params + 1:
            return float("inf")
        ll = -n_points / 2 * np.log(rss / n_points + 1e-30)
        aic = 2 * n_params - 2 * ll
        correction = (2 * n_params * (n_params + 1)) / (n_points - n_params - 1)
        return aic + correction

    def _r_squared(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot < 1e-30:
            return 0.0
        return 1.0 - ss_res / ss_tot

    print("\n" + "═" * 70)
    print("  SPEEDUP SCALING FIT (model selection per config)")
    print("═" * 70)

    results: list[dict] = []

    for cfg in n_scaling_data:
        model_name = cfg["model"]
        topo = cfg["topology"]
        p = cfg["p_layers"]
        points = cfg["n_scaling"]

        # Extract valid speedup data
        ns = []
        speedups = []
        for pt in points:
            s = pt.get("speedup")
            if s is not None and s > 0:
                ns.append(pt["n_qubits"])
                speedups.append(s)

        if len(ns) < 2:
            continue

        n_arr = np.array(ns, dtype=float)
        s_arr = np.array(speedups, dtype=float)
        n_pts = len(n_arr)

        # Fit each model
        fits: list[dict] = []

        # 1. Power law: S = a * N^b
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                popt, _ = curve_fit(_power_law, n_arr, s_arr, p0=[1.0, 1.5], maxfev=5000)
            pred = _power_law(n_arr, *popt)
            rss = float(np.sum((s_arr - pred) ** 2))
            fits.append(
                {
                    "name": "power_law",
                    "formula": f"S = {popt[0]:.2f} * N^{popt[1]:.2f}",
                    "params": {"a": float(popt[0]), "b": float(popt[1])},
                    "R2": _r_squared(s_arr, pred),
                    "AICc": _aicc(n_pts, 2, rss),
                    "rss": rss,
                }
            )
        except Exception:
            pass

        # 2. Saturating: S = S_max * (1 - exp(-k*N))
        try:
            s_max_guess = float(np.max(s_arr)) * 1.2
            k_guess = 0.1
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    _saturating,
                    n_arr,
                    s_arr,
                    p0=[s_max_guess, k_guess],
                    bounds=([0, 0], [np.inf, 10]),
                    maxfev=5000,
                )
            pred = _saturating(n_arr, *popt)
            rss = float(np.sum((s_arr - pred) ** 2))
            fits.append(
                {
                    "name": "saturating",
                    "formula": f"S = {popt[0]:.1f} * (1 - exp(-{popt[1]:.4f}*N))",
                    "params": {"S_max": float(popt[0]), "k": float(popt[1])},
                    "R2": _r_squared(s_arr, pred),
                    "AICc": _aicc(n_pts, 2, rss),
                    "rss": rss,
                }
            )
        except Exception:
            pass

        # 3. Constant: S = c
        c_mean = float(np.mean(s_arr))
        pred_const = np.full_like(s_arr, c_mean)
        rss_const = float(np.sum((s_arr - pred_const) ** 2))
        fits.append(
            {
                "name": "constant",
                "formula": f"S = {c_mean:.1f}",
                "params": {"c": c_mean},
                "R2": _r_squared(s_arr, pred_const),
                "AICc": _aicc(n_pts, 1, rss_const),
                "rss": rss_const,
            }
        )

        if not fits:
            continue

        # Select best model.
        # AICc requires n > k+1 (n=data points, k=params). With n≤3 and k=2,
        # AICc is inf for 2-param models. In that case, use R² with a minimum
        # threshold: prefer a 2-param model only if R²>0.90 and significantly
        # better than constant.
        finite_aicc = [f for f in fits if np.isfinite(f["AICc"])]
        if finite_aicc:
            best = min(finite_aicc, key=lambda f: f["AICc"])
            # But check: if best is "constant" with R²=0, and a non-constant
            # model has R²>0.90, prefer the better-fitting model.
            if best["name"] == "constant":
                better = [f for f in fits if f["name"] != "constant" and f["R2"] > 0.90]
                if better:
                    best = max(better, key=lambda f: f["R2"])
        else:
            # All AICc are inf (n=2 case) — use R² directly
            best = max(fits, key=lambda f: f["R2"])

        config_label = f"{model_name}|{topo}|p={p}"
        # Confidence assessment: based on n_points and R²
        if n_pts >= 4 and best["R2"] > 0.90:
            confidence = "high"
        elif n_pts >= 3 and best["R2"] > 0.80:
            confidence = "medium"
        elif n_pts == 2:
            confidence = "underdetermined"
        else:
            confidence = "low"

        entry = {
            "config": config_label,
            "model": model_name,
            "topology": topo,
            "p_layers": p,
            "n_values": ns,
            "speedups": speedups,
            "n_points": n_pts,
            "confidence": confidence,
            "best_model": best["name"],
            "best_formula": best["formula"],
            "best_params": best["params"],
            "R2": best["R2"],
            "AICc": best["AICc"] if np.isfinite(best["AICc"]) else None,
            "all_fits": [
                {
                    k: (v if k != "AICc" or np.isfinite(v) else None)
                    for k, v in f.items()
                    if k != "rss"
                }
                for f in fits
            ],
        }
        results.append(entry)

        # Print
        print(f"\n  {config_label}  [{confidence}]")
        print(f"    Data: N={ns}, S={[f'{s:.0f}x' for s in speedups]}")
        print(f"    Best: {best['name']} → {best['formula']}  (R²={best['R2']:.4f})")
        if verbose:
            for f in fits:
                marker = " ◀" if f["name"] == best["name"] else ""
                print(
                    f"      {f['name']:12s}: {f['formula']:40s} "
                    f"R²={f['R2']:.4f}  AICc={f['AICc']:.1f}{marker}"
                )

    # Summary
    if results:
        print(f"\n  {'─' * 60}")
        print(f"  Summary: {len(results)} configs analyzed")
        by_model = defaultdict(int)
        for r in results:
            by_model[r["best_model"]] += 1
        for m, count in sorted(by_model.items()):
            print(f"    {m}: {count} config(s)")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-axis noiseless scaling analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, default=None, help="Filter by model")
    parser.add_argument(
        "--axis",
        type=str,
        default=None,
        choices=["h-dep", "p-dep", "n-scaling", "topology", "speedup-fit", "all"],
        help="Run specific axis only (default: all)",
    )
    parser.add_argument("--json", type=str, default=None, help="Write JSON report")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Load
    print("Loading noiseless runs with deploy data...")
    runs = load_all_runs(model_filter=args.model)
    print(f"  Loaded {len(runs)} runs with per-h deploy data")

    if not runs:
        print("  No data found. Exiting.")
        return 1

    axis = args.axis or "all"
    report: dict[str, Any] = {}

    # Global summary
    report["summary"] = print_global_summary(runs)

    # Run axes
    if axis in ("all", "h-dep"):
        report["h_dependence"] = axis_h_dependence(runs, verbose=args.verbose)
    if axis in ("all", "p-dep"):
        report["p_dependence"] = axis_p_dependence(runs, verbose=args.verbose)
    if axis in ("all", "n-scaling"):
        report["n_scaling"] = axis_n_scaling(runs, verbose=args.verbose)
    if axis in ("all", "topology"):
        report["topology"] = axis_topology_comparison(runs, verbose=args.verbose)

    # Speedup scaling fit — runs after n-scaling (uses its output)
    if axis in ("all", "n-scaling", "speedup-fit"):
        n_data = report.get("n_scaling")
        if n_data is None:
            # Need to compute n-scaling first for speedup fit
            n_data = axis_n_scaling(runs, verbose=args.verbose)
            report["n_scaling"] = n_data
        configs = n_data.get("configs", [])
        if configs:
            report["speedup_fit"] = fit_speedup_models(configs, verbose=args.verbose)
        else:
            print("\n  No N-scaling data available for speedup fit.")

    # JSON output
    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        from qmbp_simulation.utils.helpers import json_dump

        json_dump(report, out_path)
        print(f"\n  JSON report saved to: {out_path}")

    print("\n" + "═" * 70)
    print("  DONE")
    print("═" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
