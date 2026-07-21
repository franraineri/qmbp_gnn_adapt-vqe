#!/usr/bin/env python3
"""Analyze Phase3 MPNN scaling results with full diagnostics.

Features:
- Summary table with ΔE/gap AND ΔE absolute
- Scaling comparison across N values
- Validates against scaling law (h_min_safe = 1.5 + 0.020·N^1.31)
- Detects anomalies: convergence failures, timing outliers, ΔE masking
- Flags cases where ΔE/gap < 5% but ΔE absolute is large

Usage:
    python scripts/analysis/analyze_all_phase3.py                    # all results
    python scripts/analysis/analyze_all_phase3.py --date 20260717    # filter by date
    python scripts/analysis/analyze_all_phase3.py --pass-only -v     # detailed pass
    python scripts/analysis/analyze_all_phase3.py --json             # machine output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "results" / "experiments" / "exp_scaling" / "phase3"

# Frontier fits (canonical, from H_EXPR_MATRIX — replaces deprecated power law)
# p=1: h_min = 2.36 + 0.0073*N (linear, R²=0.91)
# p=2: h_min = 1.57 + 0.005*N (linear, R²=0.95)
# p≥3: h_min ≈ 1.6 (constant)
# The old formula overestimates by 1.9× at N=60, 2.7× at N=100.
# Kept here as CONSERVATIVE estimator (safe: always overestimates valid regime).
SCALING_LAW_OFFSET = 1.5
SCALING_LAW_COEFF = 0.020
SCALING_LAW_EXPONENT = 1.31


def h_min_scaling_law(n: int) -> float:
    """Conservative h_min estimate (overestimates — safe for experiment design)."""
    return SCALING_LAW_OFFSET + SCALING_LAW_COEFF * n**SCALING_LAW_EXPONENT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase3 MPNN scaling results")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--date", type=str, default=None, help="Filter by date (YYYYMMDD)")
    parser.add_argument("--n-qubits", type=int, default=None, help="Filter by N")
    parser.add_argument("--pass-only", action="store_true")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def load_phase3_result(path: Path) -> dict | None:
    """Load and validate a Phase3 result JSON."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    results = data.get("results", {})
    if "section_1" not in results:
        return None

    config = data.get("config", {})
    sys_cfg = config.get("system", {})
    src_cfg = config.get("source", {})
    s1 = results["section_1"].get("data", {})
    s2 = results.get("section_2", {}).get("data", {})

    # Compute absolute ΔE from per-point data
    per_point = s2.get("per_point", [])
    delta_e_abs = []
    for pt in per_point:
        if "e_pred" in pt and "e_dmrg" in pt:
            delta_e_abs.append(abs(pt["e_pred"] - pt["e_dmrg"]))
        elif "de_gap" in pt and "gap" in pt:
            delta_e_abs.append(pt["de_gap"] * pt["gap"])

    return {
        "path": str(path),
        "filename": path.name,
        "timestamp": path.stem.replace("run_", ""),
        "n_qubits": sys_cfg.get("n_qubits"),
        "p_layers": sys_cfg.get("p_layers", 1),
        "model": sys_cfg.get("model", "tfim"),
        "topology": sys_cfg.get("topology", "chain_1d"),
        "source_file": src_cfg.get("result_file", ""),
        "use_all_seeds": src_cfg.get("use_all_seeds", False),
        # Training
        "final_mse": s1.get("final_mse"),
        "n_train_points": s1.get("n_train_points"),
        "n_model_params": s1.get("n_model_params"),
        "train_time_s": s1.get("train_time_s"),
        "stopped_early": s1.get("stopped_early"),
        # Deployment
        "deploy_pass": s2.get("pass"),
        "n_pass": s2.get("n_pass"),
        "n_total": s2.get("n_total"),
        "mean_de_gap": s2.get("mean_de_gap"),
        "max_de_gap": s2.get("max_de_gap"),
        # Absolute ΔE metrics
        "mean_delta_e": float(np.mean(delta_e_abs)) if delta_e_abs else None,
        "max_delta_e": float(np.max(delta_e_abs)) if delta_e_abs else None,
        "mean_delta_e_per_site": float(np.mean(delta_e_abs)) / sys_cfg.get("n_qubits", 1)
        if delta_e_abs
        else None,
        "per_point": per_point,
    }


def find_results(base_dir: Path, date_filter: str | None = None) -> list[Path]:
    """Find all Phase3 result JSONs under base_dir."""
    pattern = f"run_{date_filter}_*.json" if date_filter else "run_*.json"
    return sorted(base_dir.rglob(pattern))


def detect_anomalies(results: list[dict]) -> list[str]:
    """Detect anomalies across results."""
    anomalies = []

    for r in results:
        if not r["deploy_pass"]:
            continue

        n = r["n_qubits"]
        if n is None:
            continue

        # Anomaly 1: ΔE/gap < 5% but ΔE > 1.0 (gap masking large error)
        if r["max_delta_e"] is not None and r["max_delta_e"] > 1.0:
            anomalies.append(
                f"⚠ N={n}: ΔE/gap={r['max_de_gap'] * 100:.2f}% PASS but "
                f"|ΔE|={r['max_delta_e']:.3f} (absolute error > 1.0)"
            )

        # Anomaly 2: Training MSE not converged but deploy passes
        if r["final_mse"] is not None and r["final_mse"] > 1e-3 and r["deploy_pass"]:
            anomalies.append(
                f"⚠ N={n}: MSE={r['final_mse']:.2e} > 1e-3 but deploy passes. "
                f"Model may be overfitting or test points too easy."
            )

        # Anomaly 3: Training time outlier (>3x median)
        pass  # computed across all results below

    # Timing outliers
    times = [r["train_time_s"] for r in results if r["train_time_s"] is not None]
    if times:
        median_time = np.median(times)
        for r in results:
            if r["train_time_s"] and r["train_time_s"] > 3 * median_time:
                anomalies.append(
                    f"⚠ N={r['n_qubits']}: Train time {r['train_time_s']:.0f}s "
                    f">> median {median_time:.0f}s (3× outlier)"
                )

    return anomalies


def validate_scaling_law(results: list[dict]) -> list[str]:
    """Validate results against the scaling law predictions."""
    findings = []

    passing = [r for r in results if r["deploy_pass"] and r["n_qubits"]]

    for r in passing:
        n = r["n_qubits"]
        h_min_predicted = h_min_scaling_law(n)

        # Check if deploy h-values go below scaling law prediction
        for pt in r["per_point"]:
            if pt["h"] < h_min_predicted and pt["de_gap"] < 0.05:
                findings.append(
                    f"  N={n} h={pt['h']:.2f}: PASSES below scaling law "
                    f"(h_min_safe={h_min_predicted:.2f}, ΔE/gap={pt['de_gap'] * 100:.2f}%)"
                )
                break  # One example per N is enough

    return findings


def print_summary_table(results: list[dict]) -> None:
    """Print formatted summary table with both ΔE/gap and |ΔE|."""
    print(f"\n{'=' * 105}")
    print(f"  PHASE3 MPNN RESULTS — {len(results)} runs")
    print(f"{'=' * 105}")
    hdr = (
        f"  {'Time':<8} {'N':>3} {'p':>1} {'Train':>5} {'MSE':>9} "
        f"{'Deploy':>7} {'ΔE/gap%':>8} {'|ΔE|':>7} {'|ΔE|/N':>8} {'Status':>6}"
    )
    print(hdr)
    print(f"  {'-' * 100}")

    for r in results:
        mse = r["final_mse"]
        mse_s = f"{mse:.1e}" if mse is not None and np.isfinite(mse) else "N/A"
        n_pass = r["n_pass"] if r["n_pass"] is not None else "?"
        n_total = r["n_total"] if r["n_total"] is not None else "?"
        de_gap_s = f"{r['mean_de_gap'] * 100:.2f}" if r["mean_de_gap"] is not None else "?"
        de_abs_s = f"{r['mean_delta_e']:.4f}" if r["mean_delta_e"] is not None else "?"
        de_site_s = (
            f"{r['mean_delta_e_per_site']:.5f}" if r["mean_delta_e_per_site"] is not None else "?"
        )
        status = "PASS" if r["deploy_pass"] else "FAIL"
        n = r["n_qubits"] if r["n_qubits"] else "?"
        p = r["p_layers"]
        n_train = r["n_train_points"] if r["n_train_points"] else "?"
        ts = r["timestamp"][-6:]

        print(
            f"  {ts:<8} {n:>3} {p:>1} {n_train:>5} {mse_s:>9} "
            f"{n_pass}/{n_total:>3} {de_gap_s:>7}% {de_abs_s:>7} {de_site_s:>8} {status:>6}"
        )


def print_scaling_comparison(results: list[dict]) -> None:
    """Best result per N with scaling law validation."""
    passing = [r for r in results if r["deploy_pass"]]
    if not passing:
        return

    by_n: dict[int, dict] = {}
    for r in passing:
        n = r["n_qubits"]
        if n is None:
            continue
        if n not in by_n or (r["final_mse"] or float("inf")) < (
            by_n[n]["final_mse"] or float("inf")
        ):
            by_n[n] = r

    print(f"\n{'=' * 105}")
    print("  SCALING COMPARISON — Best per N (with |ΔE| absolute)")
    print(f"{'=' * 105}")
    print(
        f"  {'N':>4} {'p':>1} {'Train':>5} {'MSE':>9} {'Deploy':>7} {'ΔE/gap%':>8} {'|ΔE|':>8} {'|ΔE|/N':>9} {'h_min_law':>9} {'Status':>6}"
    )
    print(f"  {'-' * 95}")

    for n in sorted(by_n.keys()):
        r = by_n[n]
        h_law = h_min_scaling_law(n)
        de_abs = f"{r['mean_delta_e']:.4f}" if r["mean_delta_e"] is not None else "?"
        de_site = (
            f"{r['mean_delta_e_per_site']:.6f}" if r["mean_delta_e_per_site"] is not None else "?"
        )
        print(
            f"  {n:>4} {r['p_layers']:>1} {r['n_train_points']:>5} {r['final_mse']:.1e} "
            f"{r['n_pass']}/{r['n_total']:>3} "
            f"{r['mean_de_gap'] * 100:>7.3f}% {de_abs:>8} {de_site:>9} "
            f"{h_law:>9.2f} {'PASS':>6}"
        )


def print_detailed_deployment(results: list[dict]) -> None:
    """Per-point deployment with both metrics."""
    passing = [r for r in results if r["deploy_pass"] and r["per_point"]]
    if not passing:
        return

    print(f"\n{'=' * 105}")
    print("  DETAILED DEPLOYMENT (|ΔE| + ΔE/gap)")
    print(f"{'=' * 105}")

    for r in passing:
        print(f"\n  {'─' * 80}")
        print(
            f"  N={r['n_qubits']} p={r['p_layers']} | MSE={r['final_mse']:.2e} | Train={r['n_train_points']}"
        )
        print(f"  {'─' * 80}")
        print(f"  {'h':>6} {'ΔE/gap%':>9} {'|ΔE|':>9} {'gap':>8} {'status':>6}")
        print(f"  {'-' * 45}")
        for pt in r["per_point"]:
            gap = pt.get("gap", 0)
            de_abs = pt["de_gap"] * gap if gap else 0
            st = "+" if pt["de_gap"] < 0.05 else "X"
            flag = " ⚠" if de_abs > 1.0 and pt["de_gap"] < 0.05 else ""
            print(
                f"  {pt['h']:6.3f} {pt['de_gap'] * 100:8.3f}% {de_abs:9.4f} {gap:8.3f} {st:>4}{flag}"
            )


def output_json(results: list[dict]) -> None:
    """JSON output with all metrics."""
    compact = []
    for r in results:
        entry = {k: v for k, v in r.items() if k != "per_point"}
        entry["n_deploy_points"] = len(r["per_point"])
        compact.append(entry)
    print(json.dumps(compact, indent=2, default=str))


def main():
    args = parse_args()
    result_files = find_results(args.dir, date_filter=args.date)

    if not result_files:
        print(f"No Phase3 results in {args.dir}" + (f" (date={args.date})" if args.date else ""))
        sys.exit(1)

    results = []
    for f in result_files:
        r = load_phase3_result(f)
        if r is None:
            continue
        if args.n_qubits and r["n_qubits"] != args.n_qubits:
            continue
        if args.pass_only and not r["deploy_pass"]:
            continue
        results.append(r)

    if not results:
        print("No results match filters.")
        sys.exit(0)

    if args.json:
        output_json(results)
        return

    print_summary_table(results)
    print_scaling_comparison(results)

    # Anomaly detection
    anomalies = detect_anomalies(results)
    if anomalies:
        print(f"\n{'=' * 105}")
        print(f"  ANOMALIES DETECTED ({len(anomalies)})")
        print(f"{'=' * 105}")
        for a in anomalies:
            print(f"  {a}")

    # Scaling law validation
    findings = validate_scaling_law(results)
    if findings:
        print(f"\n{'=' * 105}")
        print("  SCALING LAW VALIDATION")
        print(f"{'=' * 105}")
        for f_str in findings:
            print(f_str)

    if args.verbose:
        print_detailed_deployment(results)

    n_pass = sum(1 for r in results if r["deploy_pass"])
    print(f"\n  Total: {n_pass}/{len(results)} passing ({n_pass / len(results) * 100:.0f}%)")


if __name__ == "__main__":
    main()
