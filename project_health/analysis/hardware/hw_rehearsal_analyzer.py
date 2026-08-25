#!/usr/bin/env python3
"""Analyze HW_REHEARSAL_V2 results — comprehensive post-run diagnostics.

Parses all run files from results/experiments/exp_hw_rehearsal_v2/ and
provides detailed analysis including:
- Per-section pass/fail breakdown with metrics
- Cross-run comparison (if multiple runs exist)
- ZNE quality assessment (R², gain, amplifier selection)
- Energy error decomposition (VQE vs noise overhead)
- Hardware readiness verdict (GO/NO-GO for IBM Torino)

Usage:
    .venv/bin/python scripts/analyze_hw_rehearsal_v2.py
    .venv/bin/python scripts/analyze_hw_rehearsal_v2.py --all          # All runs (comparison)
    .venv/bin/python scripts/analyze_hw_rehearsal_v2.py --json         # Machine-readable output
    .venv/bin/python scripts/analyze_hw_rehearsal_v2.py --exp-dir results/experiments/exp_hw_rehearsal_v2
    .venv/bin/python scripts/analyze_hw_rehearsal_v2.py --threshold 0.03
    .venv/bin/python scripts/analyze_hw_rehearsal_v2.py --section-filter 1 2 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze HW_REHEARSAL_V2 experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="results/experiments/exp_hw_rehearsal_v2",
        help="Experiment directory (default: %(default)s)",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Analyze only the latest run (use --all to override)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze ALL runs and show cross-run comparison",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON (machine-readable)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="ΔE/gap pass threshold (default: 0.05)",
    )
    parser.add_argument(
        "--section-filter",
        type=int,
        nargs="+",
        default=None,
        help="Only show these section numbers in the report",
    )
    return parser.parse_args()


def load_runs(exp_dir: Path, latest_only: bool = True) -> list[dict]:
    """Load result JSON files from the experiment directory."""
    if not exp_dir.exists():
        print(f"ERROR: Directory not found: {exp_dir}")
        sys.exit(1)

    files = sorted(exp_dir.glob("run_*.json"), reverse=True)
    if not files:
        print(f"ERROR: No run_*.json files in {exp_dir}")
        sys.exit(1)

    if latest_only:
        files = files[:1]

    runs = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            data["_source_file"] = str(f)
            runs.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: Failed to load {f}: {e}")
    return runs


def analyze_single_run(data: dict) -> dict:
    """Analyze a single run and return structured metrics."""
    config = data.get("config", {})
    results = data.get("results", {})
    summary = data.get("summary", {})
    source = data.get("_source_file", "?")

    system = config.get("system", {})
    zne_cfg = config.get("zne", {})

    analysis = {
        "source_file": source,
        "timestamp": data.get("timestamp", "?"),
        "topology": system.get("topology", "?"),
        "n_qubits": system.get("n_qubits", "?"),
        "p_layers": system.get("p_layers", "?"),
        "amplifier": zne_cfg.get("amplifier", "?"),
        "shots": zne_cfg.get("shots", "?"),
        "n_layouts": zne_cfg.get("n_layouts", "?"),
        "total_time_s": summary.get("total_elapsed_s", 0),
        "pass_rate": summary.get("pass_rate", 0),
        "n_passed": summary.get("n_passed", 0),
        "n_sections": summary.get("n_sections", 0),
        "sections": {},
    }

    for sec_key in sorted(results.keys()):
        sec = results[sec_key]
        sec_data = sec.get("data", {})
        sec_analysis = {
            "name": sec.get("name", sec_key),
            "success": sec.get("success", False),
            "elapsed_s": sec.get("elapsed_s", 0),
        }

        # Section-specific metric extraction
        if "mean_de_gap" in sec_data:
            sec_analysis["mean_de_gap"] = sec_data["mean_de_gap"]
            sec_analysis["max_de_gap"] = sec_data.get("max_de_gap")
            sec_analysis["std_de_gap"] = sec_data.get("std_de_gap")
            sec_analysis["train_mse"] = sec_data.get("train_mse")

        if "mean_de_gap_zne" in sec_data:
            sec_analysis["mean_de_gap_zne"] = sec_data["mean_de_gap_zne"]
            sec_analysis["max_de_gap_zne"] = sec_data.get("max_de_gap_zne")
            sec_analysis["mean_noise_error"] = sec_data.get("mean_noise_error")

        if "verdict" in sec_data:
            sec_analysis["verdict"] = sec_data["verdict"]
            sec_analysis["mitigation_strategy"] = sec_data.get("mitigation_strategy")
            sec_analysis["zne_r2"] = sec_data.get("zne_r2")
            sec_analysis["phase_label"] = sec_data.get("phase_label")

        if "amplifier_used" in sec_data:
            sec_analysis["amplifier_used"] = sec_data["amplifier_used"]
            sec_analysis["r_squared"] = sec_data.get("r_squared")
            sec_analysis["fallback_triggered"] = sec_data.get("fallback_triggered")

        if "gf_de_gap" in sec_data:
            sec_analysis["gf_de_gap"] = sec_data["gf_de_gap"]
            sec_analysis["gf_r2"] = sec_data.get("gf_r2")
            sec_analysis["pea_de_gap"] = sec_data.get("pea_de_gap")
            sec_analysis["pea_r2"] = sec_data.get("pea_r2")
            sec_analysis["pea_better"] = sec_data.get("pea_better_energy")
            sec_analysis["pea_overhead"] = sec_data.get("pea_time_overhead")

        # Section 6: Shot noise — only section 6 has n_reps
        if "n_reps" in sec_data:
            sec_analysis["std_de_gap"] = sec_data.get("std_de_gap")
            sec_analysis["n_reps"] = sec_data["n_reps"]
            sec_analysis["reproducible"] = sec_data.get("reproducible")

        # Section 7: Phase classification
        if "all_correct" in sec_data and "mean_confidence" in sec_data:
            sec_analysis["all_correct"] = sec_data["all_correct"]
            sec_analysis["n_correct"] = sec_data.get("n_correct")
            sec_analysis["n_total"] = sec_data.get("n_total")
            sec_analysis["mean_confidence"] = sec_data.get("mean_confidence")
            sec_analysis["all_snr_ok"] = sec_data.get("all_snr_ok")

        # Section 8: Cost estimation
        if "est_grand_total_s" in sec_data:
            sec_analysis["est_total_s"] = sec_data.get("est_grand_total_s")
            sec_analysis["est_total_min"] = sec_data.get("est_grand_total_min")
            sec_analysis["total_shots"] = sec_data.get("total_shots")
            sec_analysis["fits_single_job"] = sec_data.get("fits_single_job")

        # Section 9: Circuit audit
        if "all_zne_viable" in sec_data:
            sec_analysis["all_zne_viable"] = sec_data["all_zne_viable"]
            sec_analysis["mean_2q_gates"] = sec_data.get("mean_2q_gates")
            sec_analysis["max_2q_gates"] = sec_data.get("max_2q_gates")
            sec_analysis["mean_ces"] = sec_data.get("mean_ces")

        # Per-h breakdown
        per_h = sec_data.get("results", [])
        if isinstance(per_h, list) and per_h:
            h_metrics = []
            for r in per_h:
                if isinstance(r, dict) and "h" in r:
                    h_metrics.append(
                        {
                            "h": r["h"],
                            "de_gap": r.get("de_gap", r.get("de_gap_zne")),
                            "pass": r.get("pass", False),
                        }
                    )
            sec_analysis["per_h"] = h_metrics

        analysis["sections"][sec_key] = sec_analysis

    # ZNE regression warning: check if ZNE made energy error worse than noiseless
    s2_data = results.get("section_2", {}).get("data", {})
    s1_data = results.get("section_1", {}).get("data", {})
    de_zne = s2_data.get("mean_de_gap_zne")
    de_noiseless = s1_data.get("mean_de_gap")
    if de_zne is not None and de_noiseless is not None:
        s2_sec = analysis["sections"].get("section_2")
        if s2_sec is not None:
            s2_sec["zne_regression_warning"] = de_zne > de_noiseless

    return analysis


def compute_hardware_readiness(analysis: dict, threshold: float = 0.05) -> dict:
    """Compute hardware readiness verdict from analysis.

    Parameters
    ----------
    analysis : dict
        Single run analysis from analyze_single_run.
    threshold : float
        ΔE/gap pass threshold (default: 0.05).
    """
    sections = analysis.get("sections", {})

    # Check critical sections
    s1 = sections.get("section_1", {})
    s2 = sections.get("section_2", {})
    s3 = sections.get("section_3", {})
    s7 = sections.get("section_7", {})
    s9 = sections.get("section_9", {})

    mean_de_gap_zne = s2.get("mean_de_gap_zne")

    readiness = {
        "mpnn_ready": s1.get("success", False),
        "zne_pipeline_ready": s2.get("success", False),
        "full_deployment_ready": s3.get("success", False),
        "phase_classification_ready": s7.get("success", False),
        "circuit_audit_ready": s9.get("success", False),
        "mean_de_gap_noiseless": s1.get("mean_de_gap"),
        "mean_de_gap_zne": mean_de_gap_zne,
        "deployment_verdict": s3.get("verdict"),
        "deployment_r2": s3.get("zne_r2"),
        "threshold_used": threshold,
    }

    # Additional threshold check on ZNE error
    if mean_de_gap_zne is not None:
        readiness["zne_below_threshold"] = mean_de_gap_zne < threshold

    # GO/NO-GO decision
    all_critical_pass = all(
        [
            readiness["mpnn_ready"],
            readiness["zne_pipeline_ready"],
            readiness["full_deployment_ready"],
            readiness["phase_classification_ready"],
            readiness["circuit_audit_ready"],
        ]
    )
    readiness["hardware_go"] = all_critical_pass
    readiness["recommendation"] = (
        "GO — Pipeline validated end-to-end on FakeTorino. Ready for IBM Torino deployment."
        if all_critical_pass
        else "NO-GO — One or more critical sections failed. Fix issues before committing QPU time."
    )

    return readiness


def print_text_report(
    analyses: list[dict],
    section_filter: list[int] | None = None,
) -> None:
    """Print human-readable analysis report."""
    print("=" * 65)
    print("  HW_REHEARSAL_V2 — Post-Run Analysis")
    print("=" * 65)
    print()

    for i, analysis in enumerate(analyses):
        if len(analyses) > 1:
            print(f"─── Run {i + 1}: {analysis['timestamp']} ───")

        print(
            f"  Config: {analysis['topology']} N={analysis['n_qubits']} "
            f"p={analysis['p_layers']} amplifier={analysis['amplifier']}"
        )
        print(
            f"  Time: {analysis['total_time_s']:.1f}s | "
            f"Pass: {analysis['n_passed']}/{analysis['n_sections']} sections"
        )
        print()

        for sec_key in sorted(analysis["sections"].keys()):
            # Apply section filter if provided
            if section_filter is not None:
                try:
                    sec_num = int(sec_key.replace("section_", ""))
                    if sec_num not in section_filter:
                        continue
                except (ValueError, AttributeError):
                    pass

            sec = analysis["sections"][sec_key]
            status = "✅" if sec["success"] else "❌"
            print(f"  {status} {sec['name']} ({sec['elapsed_s']:.1f}s)")

            if "mean_de_gap" in sec:
                max_de = sec.get("max_de_gap")
                max_s = f"{max_de:.4f}" if isinstance(max_de, (int, float)) else "?"
                mse = sec.get("train_mse")
                mse_s = f"{mse:.2e}" if isinstance(mse, (int, float)) else "?"
                print(
                    f"     MPNN: mean_ΔE/gap={sec['mean_de_gap']:.4f}, "
                    f"max={max_s}, train_mse={mse_s}"
                )
            if "mean_de_gap_zne" in sec:
                noise_err = sec.get("mean_noise_error")
                noise_s = f"{noise_err:+.4f}" if isinstance(noise_err, (int, float)) else "?"
                print(
                    f"     ZNE:  mean_ΔE/gap={sec['mean_de_gap_zne']:.4f}, noise_overhead={noise_s}"
                )
                if sec.get("zne_regression_warning"):
                    print(
                        "     ⚠️  WARNING: ZNE increased error vs noiseless baseline "
                        "(mitigation regressed)"
                    )
            if "verdict" in sec:
                r2_val = sec.get("zne_r2")
                r2_s = f"{r2_val:.4f}" if isinstance(r2_val, (int, float)) else "?"
                print(
                    f"     Deploy: verdict={sec['verdict']}, R²={r2_s}, "
                    f"phase={sec.get('phase_label', '?')}, "
                    f"strategy={sec.get('mitigation_strategy', '?')}"
                )
            if "amplifier_used" in sec:
                r2_val = sec.get("r_squared")
                r2_str = f"{r2_val:.4f}" if isinstance(r2_val, (int, float)) else "?"
                print(
                    f"     Adaptive: used={sec['amplifier_used']}, "
                    f"R²={r2_str}, "
                    f"fallback={sec.get('fallback_triggered', '?')}"
                )
            if "gf_de_gap" in sec:
                gf_r2 = sec.get("gf_r2")
                pea_r2 = sec.get("pea_r2")
                pea_de = sec.get("pea_de_gap")
                pea_oh = sec.get("pea_overhead")
                gf_r2_s = f"{gf_r2:.4f}" if isinstance(gf_r2, (int, float)) else "N/A"
                pea_r2_s = f"{pea_r2:.4f}" if isinstance(pea_r2, (int, float)) else "N/A"
                pea_de_s = f"{pea_de:.4f}" if isinstance(pea_de, (int, float)) else "N/A"
                pea_oh_s = f"{pea_oh:.1f}x" if isinstance(pea_oh, (int, float)) else "N/A"
                print(f"     GF:  ΔE/gap={sec['gf_de_gap']:.4f}, R²={gf_r2_s}")
                print(f"     PEA: ΔE/gap={pea_de_s}, R²={pea_r2_s}, overhead={pea_oh_s}")
            if "n_reps" in sec:
                std_de = sec.get("std_de_gap")
                std_s = f"{std_de:.4f}" if isinstance(std_de, (int, float)) else "?"
                print(
                    f"     Shot noise: std(ΔE/gap)={std_s} "
                    f"({sec['n_reps']} reps), "
                    f"reproducible={sec.get('reproducible', '?')}"
                )
            if "all_correct" in sec and "mean_confidence" in sec:
                print(
                    f"     Phase: {sec.get('n_correct', '?')}/{sec.get('n_total', '?')} correct, "
                    f"mean_confidence={sec['mean_confidence']:.1f}σ, "
                    f"all_snr_ok={sec.get('all_snr_ok', '?')}"
                )
            if "est_total_min" in sec:
                print(
                    f"     Cost: {sec.get('est_total_min', '?'):.1f} min estimated, "
                    f"{sec.get('total_shots', '?'):,} shots, "
                    f"fits_job={sec.get('fits_single_job', '?')}"
                )
            if "all_zne_viable" in sec:
                mean_ces = sec.get("mean_ces")
                ces_s = f"{mean_ces:.4f}" if isinstance(mean_ces, (int, float)) else "?"
                print(
                    f"     Audit: mean_2Q={sec.get('mean_2q_gates', '?'):.0f}, "
                    f"max_2Q={sec.get('max_2q_gates', '?')}, "
                    f"all_viable={sec.get('all_zne_viable', '?')}, "
                    f"CES={ces_s}"
                )

            if "per_h" in sec:
                for h_pt in sec["per_h"]:
                    p = "✓" if h_pt["pass"] else "✗"
                    de = h_pt.get("de_gap")
                    if de is not None:
                        print(f"       h={h_pt['h']:.2f}: ΔE/gap={de:.4f} [{p}]")
            print()

        # Hardware readiness
        readiness = compute_hardware_readiness(analysis)
        go = "🟢 GO" if readiness["hardware_go"] else "🔴 NO-GO"
        print(f"  ─── HARDWARE READINESS: {go} ───")
        print(f"  {readiness['recommendation']}")
        print()

    # Cross-run comparison (if multiple)
    if len(analyses) > 1:
        print("─── Cross-Run Comparison ───")
        print(
            f"  {'Run':<6} {'Topology':<12} {'Amplifier':<12} "
            f"{'Pass%':<8} {'Time':<8} {'ZNE ΔE/gap':<12} {'ZNE Reg':<8}"
        )
        print(f"  {'-' * 6} {'-' * 12} {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 12} {'-' * 8}")
        for i, a in enumerate(analyses):
            s2 = a["sections"].get("section_2", {})
            de_zne = s2.get("mean_de_gap_zne", "?")
            de_str = f"{de_zne:.4f}" if isinstance(de_zne, float) else str(de_zne)
            zne_reg = s2.get("zne_regression_warning", False)
            reg_str = "⚠️" if zne_reg else "ok"
            print(
                f"  {i + 1:<6} {a['topology']:<12} {a['amplifier']:<12} "
                f"{a['pass_rate'] * 100:<8.0f} {a['total_time_s']:<8.1f} "
                f"{de_str:<12} {reg_str:<8}"
            )
        print()


def main():
    """Entry point."""
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    latest_only = not args.all

    runs = load_runs(exp_dir, latest_only=latest_only)
    analyses = [analyze_single_run(r) for r in runs]

    if args.json:
        output = {
            "n_runs": len(analyses),
            "analyses": analyses,
            "hardware_readiness": [
                compute_hardware_readiness(a, threshold=args.threshold) for a in analyses
            ],
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_text_report(analyses, section_filter=args.section_filter)


if __name__ == "__main__":
    main()
