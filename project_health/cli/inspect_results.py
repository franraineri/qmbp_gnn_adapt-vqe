#!/usr/bin/env python3
"""Inspect benchmark results — transpiled circuit properties & pre-submission audit.

Configurable via CLI: select configs, h-values, seeds, mode, and output file.

Usage:
    # Default: 5 target configs, h=4.0, seed=100
    python inspect_results.py

    # Multiple h-values with averages
    python inspect_results.py --h-values 3.25,3.5,3.75,4.0

    # Specific configs and seed
    python inspect_results.py --configs C0_raw,C5_full_pea_balanced --seed 42

    # Save output to file (also prints to stdout)
    python inspect_results.py --h-values 3.25,3.5,3.75,4.0 --output report.txt

    # Hardware results
    python inspect_results.py --mode hardware --h-values 4.0

    # JSON export (machine-readable)
    python inspect_results.py --h-values 3.25,3.5,3.75,4.0 --json --output results_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────

DEFAULT_CONFIGS = [
    "C0_raw",
    "C1_dd_only",
    "C4_full_pea_light",
    "C5_full_pea_balanced",
    "C16_aqc_pea",
]
DEFAULT_H_VALUES = [4.0]
DEFAULT_SEED = 100
DEFAULT_MODE = "fake_backend"
RESULTS_BASE = Path("results/mitigation_benchmark")


# ─── Data Loading ─────────────────────────────────────────────────────────


def find_result_file(config_id: str, h_value: float, mode: str, seed: int) -> Path | None:
    """Find the result JSON for a specific (config, h, mode, seed)."""
    h_str = f"h{str(h_value).replace('.', 'p')}"
    config_dir = RESULTS_BASE / mode / config_id

    if not config_dir.exists():
        return None

    if seed == 42:
        # Default seed: files without _seed suffix
        candidates = sorted(config_dir.glob(f"{h_str}_run_*.json"))
        candidates = [f for f in candidates if "_seed" not in f.stem]
    else:
        candidates = sorted(config_dir.glob(f"{h_str}_run_*_seed{seed}.json"))

    if not candidates:
        return None
    return candidates[-1]  # Most recent


def load_envelope(path: Path) -> dict[str, Any] | None:
    """Load and validate a result envelope."""
    try:
        data = json.loads(path.read_text())
        results = data.get("results", {})
        if results.get("e_raw") is None and results.get("e_mitigated") is None:
            return None  # Error envelope
        return data
    except (json.JSONDecodeError, OSError):
        return None


# ─── Formatting ───────────────────────────────────────────────────────────


def format_value(val, fmt: str = ".4f") -> str:
    """Format a value for display, handling None/NaN."""
    if val is None:
        return "N/A"
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return "N/A"
    if fmt == "d":
        # Integer format — cast to int (JSON may deliver as float)
        try:
            return str(int(val))
        except (ValueError, TypeError):
            return str(val)
    if isinstance(val, (int, float)):
        return f"{val:{fmt}}"
    return str(val)


def build_config_report(
    config_id: str,
    h_value: float,
    envelope: dict[str, Any],
) -> dict[str, Any]:
    """Extract all relevant metrics from an envelope into a flat report dict."""
    cs = envelope.get("circuit_stats", {})
    psa = envelope.get("pre_submission_audit", {})
    results = envelope.get("results", {})
    timing = envelope.get("timing", {})
    bt = envelope.get("budget_tracking", {})

    return {
        "config_id": config_id,
        "h_value": h_value,
        # Circuit structure
        "n_2q_gates": cs.get("n_2q_gates"),
        "depth_2q": cs.get("depth_2q"),
        "depth_total": cs.get("depth"),
        "total_gates": cs.get("total_gates"),
        "n_1q_gates": cs.get("n_1q_gates"),
        "active_qubits": cs.get("active_qubits"),
        "optimization_level": cs.get("optimization_level"),
        # Idle/decoherence
        "idle_cycles_per_qubit": cs.get("idle_cycles_per_qubit"),
        "max_idle_stretch": cs.get("max_idle_stretch"),
        # Parallelism
        "parallelism_ratio": cs.get("parallelism_ratio"),
        "gate_density_2q": cs.get("gate_density_2q"),
        # Error prediction
        "error_budget": cs.get("error_budget"),
        "fidelity_estimate": cs.get("fidelity_estimate"),
        "error_budget_source": cs.get("error_budget_source"),
        "error_budget_per_gate": cs.get("error_budget_per_gate"),
        # Derived
        "routing_overhead_pct": cs.get("routing_overhead_pct"),
        "circuit_depth_with_dd_estimate": cs.get("circuit_depth_with_dd_estimate"),
        "transpiled_vs_logical_ratio": cs.get("transpiled_vs_logical_ratio"),
        # Gate breakdown
        "count_ops": cs.get("count_ops"),
        # Provenance
        "circuit_fingerprint": psa.get("circuit_fingerprint"),
        # Results
        "e_raw": results.get("e_raw"),
        "e_mitigated": results.get("e_mitigated"),
        "e_exact": results.get("e_exact"),
        "delta_e_gap": results.get("delta_e_gap"),
        "zne_r2": results.get("zne_r2"),
        "phase_label": results.get("phase_label"),
        "improvement_vs_raw": results.get("improvement_vs_raw"),
        # Timing
        "wall_time_s": timing.get("wall_time_s"),
        "qpu_seconds": timing.get("qpu_seconds"),
    }


def compute_averages(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute averages across multiple h-values for one config."""
    if not reports:
        return {}

    numeric_keys = [
        "n_2q_gates",
        "depth_2q",
        "depth_total",
        "total_gates",
        "idle_cycles_per_qubit",
        "max_idle_stretch",
        "parallelism_ratio",
        "gate_density_2q",
        "error_budget",
        "fidelity_estimate",
        "routing_overhead_pct",
        "delta_e_gap",
        "wall_time_s",
        "qpu_seconds",
    ]

    averages: dict[str, Any] = {"config_id": reports[0]["config_id"], "n_h_points": len(reports)}
    for key in numeric_keys:
        vals = [r[key] for r in reports if r.get(key) is not None]
        if vals:
            averages[f"mean_{key}"] = float(np.mean(vals))
            if len(vals) > 1:
                averages[f"std_{key}"] = float(np.std(vals, ddof=1))
            averages[f"min_{key}"] = float(np.min(vals))
            averages[f"max_{key}"] = float(np.max(vals))
    return averages


# ─── Text Output ──────────────────────────────────────────────────────────


def print_report(
    reports_by_config: dict[str, list[dict[str, Any]]],
    h_values: list[float],
    output_lines: list[str],
) -> None:
    """Generate human-readable text report."""
    multi_h = len(h_values) > 1

    output_lines.append("=" * 70)
    output_lines.append("  TRANSPILED CIRCUIT PROPERTIES — PRE-QPU AUDIT")
    output_lines.append(f"  h-values: {h_values}")
    output_lines.append("=" * 70)
    output_lines.append("")

    for config_id, reports in reports_by_config.items():
        if not reports:
            output_lines.append(f"  --- {config_id}: NO DATA ---")
            output_lines.append("")
            continue

        for r in reports:
            h = r["h_value"]
            output_lines.append(f"  --- {config_id} | h={h:.2f} ---")
            output_lines.append(f"    n_2q_gates:          {format_value(r['n_2q_gates'], 'd')}")
            output_lines.append(f"    depth_2q:            {format_value(r['depth_2q'], 'd')}")
            output_lines.append(f"    depth_total:         {format_value(r['depth_total'], 'd')}")
            output_lines.append(f"    total_gates:         {format_value(r['total_gates'], 'd')}")
            output_lines.append(
                f"    idle_cycles/qubit:   {format_value(r['idle_cycles_per_qubit'], '.2f')}"
            )
            output_lines.append(
                f"    max_idle_stretch:    {format_value(r['max_idle_stretch'], 'd')}"
            )
            output_lines.append(
                f"    parallelism_ratio:   {format_value(r['parallelism_ratio'], '.3f')}"
            )
            output_lines.append(
                f"    gate_density_2q:     {format_value(r['gate_density_2q'], '.4f')}"
            )
            output_lines.append(
                f"    error_budget:        {format_value(r['error_budget'], '.4f')}"
            )
            output_lines.append(
                f"    fidelity_estimate:   {format_value(r['fidelity_estimate'], '.4f')}"
            )
            output_lines.append(f"    error_budget_source: {r.get('error_budget_source', 'N/A')}")
            output_lines.append(f"    opt_level:           {r.get('optimization_level', 'N/A')}")
            output_lines.append(
                f"    routing_overhead:    {format_value(r['routing_overhead_pct'], '.1f')}%"
            )
            output_lines.append(
                f"    dd_depth_estimate:   {format_value(r['circuit_depth_with_dd_estimate'], 'd')}"
            )
            output_lines.append(f"    fingerprint:         {r.get('circuit_fingerprint', 'N/A')}")
            output_lines.append(f"    delta_e_gap:         {format_value(r['delta_e_gap'], '.4f')}")
            output_lines.append(f"    e_raw:               {format_value(r['e_raw'], '.4f')}")
            output_lines.append(f"    e_mitigated:         {format_value(r['e_mitigated'], '.4f')}")
            output_lines.append(f"    wall_time_s:         {format_value(r['wall_time_s'], '.2f')}")

            # Gate breakdown
            ops = r.get("count_ops", {})
            if ops:
                top = sorted(ops.items(), key=lambda x: -x[1])[:5]
                output_lines.append(f"    gate_breakdown:      {dict(top)}")

            # Error contributors
            ebpg = r.get("error_budget_per_gate", {})
            if ebpg:
                top_c = sorted(ebpg.items(), key=lambda x: -x[1])[:3]
                output_lines.append(f"    error_contributors:  {dict(top_c)}")

            output_lines.append("")

        # Averages for multi-h
        if multi_h and len(reports) > 1:
            avgs = compute_averages(reports)
            output_lines.append(f"  --- {config_id} | AVERAGE ({avgs['n_h_points']} h-points) ---")
            output_lines.append(
                f"    mean_delta_e_gap:    {format_value(avgs.get('mean_delta_e_gap'), '.4f')}"
            )
            if "std_delta_e_gap" in avgs:
                output_lines.append(
                    f"    std_delta_e_gap:     {format_value(avgs.get('std_delta_e_gap'), '.4f')}"
                )
            output_lines.append(
                f"    mean_error_budget:   {format_value(avgs.get('mean_error_budget'), '.4f')}"
            )
            output_lines.append(
                f"    mean_fidelity:       {format_value(avgs.get('mean_fidelity_estimate'), '.4f')}"
            )
            output_lines.append(
                f"    mean_wall_time_s:    {format_value(avgs.get('mean_wall_time_s'), '.2f')}"
            )
            output_lines.append(
                f"    range_delta_e_gap:   [{format_value(avgs.get('min_delta_e_gap'), '.4f')}, {format_value(avgs.get('max_delta_e_gap'), '.4f')}]"
            )
            output_lines.append("")

    # Global summary table
    if multi_h:
        output_lines.append("=" * 70)
        output_lines.append("  SUMMARY TABLE (mean ± std across h-values)")
        output_lines.append("=" * 70)
        output_lines.append(
            f"  {'Config':<25s} {'ΔE/gap':<14s} {'Error Budget':<14s} {'Fidelity':<12s} {'n_2q':<6s} {'d_2q':<6s}"
        )
        output_lines.append(f"  {'-' * 25} {'-' * 14} {'-' * 14} {'-' * 12} {'-' * 6} {'-' * 6}")

        for config_id, reports in reports_by_config.items():
            if not reports:
                continue
            avgs = compute_averages(reports)
            de = avgs.get("mean_delta_e_gap")
            de_s = avgs.get("std_delta_e_gap")
            eb = avgs.get("mean_error_budget")
            fid = avgs.get("mean_fidelity_estimate")
            n2q = avgs.get("mean_n_2q_gates")
            d2q = avgs.get("mean_depth_2q")

            de_str = f"{de:.4f}" if de is not None else "N/A"
            if de_s is not None and de is not None:
                de_str = f"{de:.4f}±{de_s:.4f}"

            output_lines.append(
                f"  {config_id:<25s} "
                f"{de_str:<14s} "
                f"{format_value(eb, '.4f'):<14s} "
                f"{format_value(fid, '.4f'):<12s} "
                f"{format_value(n2q, '.0f'):<6s} "
                f"{format_value(d2q, '.0f'):<6s}"
            )
        output_lines.append("")


# ─── Main ─────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect benchmark results — circuit properties & audit data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=",".join(DEFAULT_CONFIGS),
        help=f"CSV of config IDs (default: {','.join(DEFAULT_CONFIGS)})",
    )
    parser.add_argument(
        "--h-values",
        type=str,
        default=",".join(str(h) for h in DEFAULT_H_VALUES),
        help="CSV of h-values to inspect (default: 4.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed to look for (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        choices=["fake_backend", "hardware"],
        help=f"Execution mode results to inspect (default: {DEFAULT_MODE})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (writes text or JSON depending on --json flag)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Export as JSON instead of text (machine-readable)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    configs = [c.strip() for c in args.configs.split(",")]
    h_values = [float(h.strip()) for h in args.h_values.split(",")]
    seed = args.seed
    mode = args.mode

    # Load all data
    reports_by_config: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []

    for config_id in configs:
        reports_by_config[config_id] = []
        for h in h_values:
            path = find_result_file(config_id, h, mode, seed)
            if path is None:
                missing.append(f"{config_id} h={h}")
                continue
            envelope = load_envelope(path)
            if envelope is None:
                missing.append(f"{config_id} h={h} (error envelope)")
                continue
            report = build_config_report(config_id, h, envelope)
            reports_by_config[config_id].append(report)

    if missing:
        print(
            f"  Warning: {len(missing)} missing results: {missing[:5]}{'...' if len(missing) > 5 else ''}",
            file=sys.stderr,
        )

    # Generate output
    if args.json:
        # JSON export
        export = {
            "metadata": {
                "configs": configs,
                "h_values": h_values,
                "seed": seed,
                "mode": mode,
            },
            "per_config": reports_by_config,
            "averages": {
                config_id: compute_averages(reports)
                for config_id, reports in reports_by_config.items()
                if reports
            },
        }
        output_text = json.dumps(export, indent=2, default=str)
    else:
        # Text report
        output_lines: list[str] = []
        print_report(reports_by_config, h_values, output_lines)
        output_text = "\n".join(output_lines)

    # Output to stdout
    print(output_text)

    # Write to file if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text)
        print(f"\n  → Saved to {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
