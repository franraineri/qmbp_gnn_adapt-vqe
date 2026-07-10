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
    python scripts/analyze_noiseless_scaling.py

    # Filter by model
    python scripts/analyze_noiseless_scaling.py --model tfim

    # Only scaling axis (N-dependence)
    python scripts/analyze_noiseless_scaling.py --axis n-scaling

    # JSON output for further processing
    python scripts/analyze_noiseless_scaling.py --json results/scaling_analysis.json

    # Verbose (per-h tables)
    python scripts/analyze_noiseless_scaling.py -v
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


def find_all_runs() -> list[Path]:
    """Find all run_*.json files in noiseless experiment directories."""
    base = ROOT / "results" / "experiments"
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
    files = find_all_runs()
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
        choices=["h-dep", "p-dep", "n-scaling", "topology", "all"],
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
