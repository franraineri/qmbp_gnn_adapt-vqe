#!/usr/bin/env python3
"""Inspect a noiseless pipeline result JSON — per-h-point breakdown.

Shows per-point VQE and Deploy metrics with pass/fail classification,
distance to threshold, and aggregate diagnostics.

Usage:
    # Inspect a specific run
    python project_health/cli/inspect_noiseless_run.py results/experiments/exp_noiseless_tfim_longitudinal_4/run_20260702_155149.json

    # With custom threshold
    python project_health/cli/inspect_noiseless_run.py run.json --threshold 0.03

    # JSON output for downstream tools
    python project_health/cli/inspect_noiseless_run.py run.json --json

    # Inspect latest run in a directory
    python project_health/cli/inspect_noiseless_run.py --latest exp_noiseless_tfim_4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results" / "experiments"


def find_latest_run(exp_dir: str) -> Path | None:
    """Find the most recent run_*.json in an experiment directory (recursive)."""
    d = RESULTS_DIR / exp_dir
    if not d.exists():
        return None
    # Search recursively to support nested folder structure
    runs = sorted(d.rglob("run_*.json"))
    return runs[-1] if runs else None


def load_run(path: Path) -> dict:
    """Load and validate a noiseless pipeline result JSON."""
    from qmbp_simulation.framework.result_io import load_result

    data = load_result(path)
    if "results" not in data or "config" not in data:
        raise ValueError(f"{path} does not look like a noiseless pipeline result.")
    return data


def extract_config(data: dict) -> dict:
    """Extract key config parameters."""
    c = data["config"]
    sys_cfg = c.get("system", {})
    h_grid = c.get("h_grid", {})
    return {
        "model": sys_cfg.get("model", c.get("model", "?")),
        "n_qubits": sys_cfg.get("n_qubits", c.get("n_qubits", "?")),
        "p_layers": sys_cfg.get("p_layers", c.get("p_layers", "?")),
        "topologies": sys_cfg.get("topologies", c.get("topologies", ["?"])),
        "h_min": h_grid.get("h_min", "?"),
        "h_max": h_grid.get("h_max", "?"),
        "h_points": h_grid.get("h_points", "?"),
        "seeds": c.get("seeds", []),
    }


def extract_section_summary(data: dict) -> list[dict]:
    """Extract per-section pass/fail/elapsed."""
    results = data["results"]
    sections = []
    for key in sorted(results.keys()):
        s = results[key]
        sections.append({
            "id": key,
            "name": s.get("name", key),
            "success": s.get("success", False),
            "elapsed_s": s.get("elapsed_s", 0),
            "pass_field": s.get("data", {}).get("pass", None),
        })
    return sections


def extract_deploy_perpoint(data: dict) -> list[dict] | None:
    """Extract per-h deploy results from section_4."""
    s4 = data["results"].get("section_4", {}).get("data", {})
    return s4.get("per_point", None)


def extract_vqe_summary(data: dict) -> dict | None:
    """Extract VQE aggregate metrics from section_2."""
    s2 = data["results"].get("section_2", {}).get("data", {})
    topos = s2.get("topologies", {})
    if not topos:
        return None
    # Return first topology (usually only one in these runs)
    topo_name = next(iter(topos))
    t = topos[topo_name]
    return {
        "topology": topo_name,
        "n_points": t.get("n_points"),
        "n_pass_5pct": t.get("n_pass_5pct"),
        "mean_fidelity": t.get("mean_fidelity"),
        "min_fidelity": t.get("min_fidelity"),
        "mean_de_gap": t.get("mean_de_gap"),
        "max_de_gap": t.get("max_de_gap"),
        "theta_smoothness_max": t.get("theta_smoothness_max"),
        "n_converged": t.get("n_converged"),
        "total_time_s": t.get("total_time_s"),
    }


def extract_mpnn_summary(data: dict) -> dict | None:
    """Extract MPNN training metrics from section_3."""
    s3 = data["results"].get("section_3", {}).get("data", {})
    if not s3:
        return None
    return {
        "n_training_points": s3.get("n_training_points"),
        "n_output_params": s3.get("n_output_params"),
        "final_mse": s3.get("final_mse"),
        "per_h_mse_mean": s3.get("per_h_mse_mean"),
        "stopped_early": s3.get("stopped_early"),
        "stop_reason": s3.get("stop_reason"),
        "mse_summary": s3.get("mse_summary"),
    }


def extract_ground_truth(data: dict) -> dict | None:
    """Extract Phase 1 ground truth summary."""
    s1 = data["results"].get("section_1", {}).get("data", {})
    topos = s1.get("topologies", {})
    if not topos:
        return None
    topo_name = next(iter(topos))
    t = topos[topo_name]
    return {
        "topology": topo_name,
        "n_points": t.get("n_points"),
        "e_range": [t.get("e_min"), t.get("e_max")],
        "gap_range": [t.get("gap_min"), t.get("gap_max")],
        "validation_passed": t.get("validation_passed"),
        "validation_warnings": t.get("validation_warnings", []),
        "h_values": s1.get("h_values", []),
    }


def classify_point(de_gap: float, threshold: float) -> str:
    """Classify a point: PASS, MARGINAL, FAIL."""
    if de_gap < threshold:
        return "PASS"
    elif de_gap < threshold * 2:
        return "MARGINAL"
    else:
        return "FAIL"


def print_report(data: dict, threshold: float = 0.05) -> dict:
    """Print full inspection report. Returns structured summary for JSON output."""
    cfg = extract_config(data)
    sections = extract_section_summary(data)
    gt = extract_ground_truth(data)
    vqe = extract_vqe_summary(data)
    mpnn = extract_mpnn_summary(data)
    per_point = extract_deploy_perpoint(data)

    summary = data.get("summary", {})
    elapsed = summary.get("total_time_s", data.get("elapsed_s", 0))

    # Header
    print("=" * 80)
    print(f"  NOISELESS PIPELINE INSPECTION")
    print(f"  File: {data.get('_source_file', '?')}")
    print("=" * 80)
    print()

    # Config
    print(f"  Model: {cfg['model']}  |  N={cfg['n_qubits']}  |  p={cfg['p_layers']}")
    print(f"  Topology: {cfg['topologies']}")
    print(f"  h-grid: [{cfg['h_min']}, {cfg['h_max']}] × {cfg['h_points']} pts")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print()

    # Section summary
    print("  SECTIONS:")
    for s in sections:
        icon = "✅" if s["success"] else "❌"
        print(f"    {icon} {s['name']:<45} {s['elapsed_s']:.0f}s")
    print()

    # Ground truth
    if gt:
        print("  GROUND TRUTH (Phase 1):")
        print(f"    E ∈ [{gt['e_range'][0]:.4f}, {gt['e_range'][1]:.4f}]")
        print(f"    gap ∈ [{gt['gap_range'][0]:.6f}, {gt['gap_range'][1]:.6f}]")
        if gt["validation_warnings"]:
            for w in gt["validation_warnings"]:
                print(f"    ⚠️  {w}")
        gap_is_floor = (
            gt["gap_range"][0] is not None
            and gt["gap_range"][1] is not None
            and abs(gt["gap_range"][1] - gt["gap_range"][0]) < 1e-6
        )
        if gap_is_floor:
            print(f"    ⚠️  DMRG gap floor detected: all gaps = {gt['gap_range'][0]:.4f}")
            print(f"       ΔE/gap metrics are inflated. True gaps likely 2-5× larger.")
        print()

    # VQE
    if vqe:
        print(f"  VQE (Phase 2) — {vqe['topology']}:")
        n_pts = vqe["n_points"] or 0
        n_pass = vqe["n_pass_5pct"] or 0
        print(f"    Pass rate: {n_pass}/{n_pts} ({100*n_pass/max(n_pts,1):.0f}%)")
        print(f"    F̄={vqe['mean_fidelity']:.4f}  F_min={vqe['min_fidelity']:.4f}")
        print(f"    ΔE/gap: mean={vqe['mean_de_gap']:.4f}  max={vqe['max_de_gap']:.4f}")
        print(f"    θ smoothness: {vqe['theta_smoothness_max']:.4f}")
        print(f"    Converged: {vqe['n_converged']}/{n_pts}")
        print()

    # MPNN
    if mpnn:
        print(f"  MPNN (Phase 3):")
        print(f"    Training points: {mpnn['n_training_points']}  output params: {mpnn['n_output_params']}")
        ratio = (mpnn["n_training_points"] or 1) / max(mpnn["n_output_params"] or 1, 1)
        print(f"    Data:output ratio: {ratio:.2f}:1", end="")
        if ratio < 2.0:
            print("  ⚠️  LOW (need ≥2.5:1)")
        else:
            print("  ✓")
        mse_sum = mpnn.get("mse_summary", {})
        if mse_sum:
            print(f"    MSE: final={mpnn['final_mse']:.2e}  best={mse_sum.get('best', '?'):.2e}")
            epoch_500 = mse_sum.get("at_epoch_500")
            epoch_1000 = mse_sum.get("at_epoch_1000")
            if epoch_500 and epoch_1000 and epoch_1000 > epoch_500 * 2:
                print(f"    ⚠️  Training UNSTABLE: MSE@500={epoch_500:.2e} → MSE@1000={epoch_1000:.2e}")
        print(f"    Stopped: {mpnn['stop_reason']}")
        print()

    # Per-point deploy
    report_points = []
    if per_point:
        print(f"  DEPLOY (Phase 4) — per h-point (threshold={threshold*100:.0f}%):")
        print(f"  {'h':>7} | {'ΔE/gap':>8} | {'Status':>8} | {'F':>6} | {'ΔE/gap_rand':>11} | {'win':>3} | Distance")
        print("  " + "-" * 76)

        n_pass = n_marginal = n_fail = 0
        for pt in per_point:
            h = pt["h_test"]
            de = pt["de_gap"]
            de_rand = pt.get("de_gap_random_init", None)
            fid = pt.get("fidelity", pt.get("fidelity_pred", None))
            label_ok = pt.get("label_correct", None)

            status = classify_point(de, threshold)
            margin = de - threshold
            mpnn_win = de < de_rand if de_rand else None

            if status == "PASS":
                n_pass += 1
                dist_str = f"+{abs(margin):.4f} margin"
            elif status == "MARGINAL":
                n_marginal += 1
                dist_str = f"-{margin:.4f} needed"
            else:
                n_fail += 1
                dist_str = f"-{margin:.4f} needed"

            fid_str = f"{fid:.4f}" if fid is not None else "  --  "
            de_rand_str = f"{de_rand:.4f}" if de_rand is not None else "   --   "
            win_str = "Y" if mpnn_win else ("N" if mpnn_win is False else "-")

            icon = {"PASS": "✅", "MARGINAL": "⚠️ ", "FAIL": "❌"}[status]
            print(f"  {h:7.4f} | {de:8.4f} | {icon}{status:>5} | {fid_str} | {de_rand_str:>11} | {win_str:>3} | {dist_str}")

            report_points.append({
                "h": h, "de_gap": de, "status": status, "fidelity": fid,
                "de_gap_random": de_rand, "mpnn_wins": mpnn_win,
                "margin": -margin if status != "PASS" else margin,
                "label_correct": label_ok,
            })

        print()
        total = len(per_point)
        print(f"  Summary: PASS={n_pass}/{total}  MARGINAL={n_marginal}/{total}  FAIL={n_fail}/{total}")
        mpnn_wins = sum(1 for p in report_points if p["mpnn_wins"])
        print(f"  MPNN wins vs random: {mpnn_wins}/{total} ({100*mpnn_wins/max(total,1):.0f}%)")
        if per_point:
            s4_data = data["results"]["section_4"]["data"]
            speedup = s4_data.get("speedup_factor")
            if speedup:
                print(f"  Speedup: {speedup:.0f}×")
    else:
        print("  ⚠️  No per-point deploy data found in this result.")
    print()

    # Build output summary
    output = {
        "config": cfg,
        "elapsed_s": elapsed,
        "sections": sections,
        "ground_truth": gt,
        "vqe": vqe,
        "mpnn": mpnn,
        "deploy_per_point": report_points,
        "deploy_summary": {
            "n_pass": sum(1 for p in report_points if p["status"] == "PASS"),
            "n_marginal": sum(1 for p in report_points if p["status"] == "MARGINAL"),
            "n_fail": sum(1 for p in report_points if p["status"] == "FAIL"),
            "threshold": threshold,
        },
    }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a noiseless pipeline result JSON with per-h breakdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "file", nargs="?", default=None,
        help="Path to run_*.json result file.",
    )
    parser.add_argument(
        "--latest", type=str, default=None,
        help="Inspect latest run in the given experiment directory name.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05,
        help="ΔE/gap pass threshold (default: 0.05 = 5%%).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output structured JSON instead of human-readable report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve file path
    if args.latest:
        path = find_latest_run(args.latest)
        if path is None:
            print(f"ERROR: No run_*.json found in {RESULTS_DIR / args.latest}")
            return 1
    elif args.file:
        path = Path(args.file)
        if not path.is_absolute():
            path = ROOT / path
    else:
        print("ERROR: Provide a file path or --latest <exp_dir>")
        return 1

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return 1

    # Load
    data = load_run(path)
    data["_source_file"] = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)

    # Report
    output = print_report(data, threshold=args.threshold)

    if args.json:
        from qmbp_simulation.utils.helpers import json_serialize
        print(json.dumps(output, indent=2, default=json_serialize))

    return 0


if __name__ == "__main__":
    sys.exit(main())
