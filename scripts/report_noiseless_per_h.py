#!/usr/bin/env python3
"""Generate a markdown report for noiseless pipeline results at specific h-values.

Reuses the loading and per-point extraction logic from analyze_noiseless_per_h.py
but outputs a structured markdown file suitable for documentation/thesis.

Usage:
    python scripts/report_noiseless_per_h.py results/experiments/exp_noiseless_tfim_v2/ \
        --h 1.0 1.5 2.0 3.0 4.0 5.0 -o noiseless_v2_report.md

    python scripts/report_noiseless_per_h.py results/experiments/exp_noiseless_tfim_v2/ --all
"""

import argparse
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Import from sibling script without requiring scripts/ to be a package
_sibling = Path(__file__).parent / "analyze_noiseless_per_h.py"
_spec = importlib.util.spec_from_file_location("analyze_noiseless_per_h", _sibling)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
find_closest = _mod.find_closest
load_run = _mod.load_run


def _header(data: dict) -> str:
    """Build markdown header from run config."""
    cfg = data["config"]
    sys_cfg = cfg["system"]
    return (
        f"N={sys_cfg['n_qubits']} p={sys_cfg['p_layers']} "
        f"topo={sys_cfg['topologies'][0]} model={sys_cfg['model']} "
        f"h=[{cfg['h_grid']['h_min']}, {cfg['h_grid']['h_max']}] "
        f"pts={cfg['h_grid']['h_points']}"
    )


def _section_summary(data: dict) -> list[str]:
    """Build per-section summary lines."""
    results = data["results"]
    cfg = data["config"]
    sys_cfg = cfg["system"]
    topo = sys_cfg["topologies"][0]
    lines = []

    # Section 2: VQE
    s2 = results.get("section_2", {})
    s2_data = s2.get("data", {})
    topo_s2 = s2_data.get("topologies", {}).get(topo, {})
    status = "✅ PASS" if s2.get("success") else "❌ FAIL"
    lines.append(f"### VQE (Section 2): {status}")
    if topo_s2:
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| pass_rate | {topo_s2.get('n_pass_5pct')}/{topo_s2.get('n_points')} |")
        lines.append(f"| mean_F | {topo_s2.get('mean_fidelity', 0):.5f} |")
        lines.append(f"| min_F | {topo_s2.get('min_fidelity', 0):.4f} |")
        lines.append(f"| mean_ΔE/gap | {topo_s2.get('mean_de_gap', 0):.4e} |")
        lines.append(f"| max_ΔE/gap | {topo_s2.get('max_de_gap', 0):.4e} |")
        lines.append(f"| θ_smooth_max | {topo_s2.get('theta_smoothness_max', 0):.4f} |")
        lines.append(f"| θ_smooth_mean | {topo_s2.get('theta_smoothness_mean', 0):.4f} |")
        lines.append(f"| mean_entropy | {topo_s2.get('mean_entanglement_entropy', 0):.4f} |")

    # Section 3: MPNN
    s3 = results.get("section_3", {})
    s3_data = s3.get("data", {})
    status = "✅ PASS" if s3.get("success") else "❌ FAIL"
    lines.append(f"\n### MPNN (Section 3): {status}")
    if s3_data:
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| final_mse | {s3_data.get('final_mse', 0):.6e} |")
        lines.append(f"| final_de_gap | {s3_data.get('final_de_gap', 0):.4e} |")
        lines.append(f"| n_params | {s3_data.get('n_output_params')} |")
        lines.append(f"| n_training | {s3_data.get('n_training_points')} |")

    # Section 4: Deploy
    s4 = results.get("section_4", {})
    s4_data = s4.get("data", {})
    status = "✅ PASS" if s4.get("success") else "❌ FAIL"
    lines.append(f"\n### Deploy (Section 4): {status}")
    if s4_data:
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| n_test | {s4_data.get('n_test_points')} |")
        lines.append(f"| pass_energy | {s4_data.get('n_pass_energy')} |")
        lines.append(f"| correct_labels | {s4_data.get('n_correct_label')} |")
        lines.append(f"| mean_ΔE/gap | {s4_data.get('mean_de_gap', 0):.4e} |")
        lines.append(f"| max_ΔE/gap | {s4_data.get('max_de_gap', 0):.4e} |")
        lines.append(f"| mean_F | {s4_data.get('mean_fidelity', 0):.5f} |")
        sf = s4_data.get("speedup_factor")
        if sf:
            lines.append(f"| speedup_factor | {sf:.1f}x |")

    return lines


def _per_h_table(data: dict, h_targets: list[float] | None) -> list[str]:
    """Build per-h markdown table from section_4 per_point data."""
    results = data["results"]
    s4 = results.get("section_4", {})
    pp = s4.get("data", {}).get("per_point", [])

    if not pp:
        return ["*(No per-point deploy data available)*"]

    # Select points
    if h_targets is None:
        points = sorted(pp, key=lambda x: x["h_test"])
    else:
        points = []
        for ht in sorted(h_targets):
            best = min(pp, key=lambda p: abs(p["h_test"] - ht))
            if abs(best["h_test"] - ht) <= 0.15:
                points.append(best)

    lines = []
    lines.append("| h | ΔE/gap | F | S_ent | E_pred | E_exact | ⟨X⟩_err | ⟨ZZ⟩_err | Label |")
    lines.append("|----:|-------:|-----:|------:|-------:|--------:|--------:|---------:|:-----:|")

    for p in points:
        h = p["h_test"]
        de = p["de_gap"]
        f = p.get("fidelity", 0)
        s = p.get("entanglement_entropy", 0)
        ep = p["e_pred"]
        ee = p["e_exact"]
        mx = p.get("mag_x_error", 0)
        zz = p.get("corr_zz_error", 0)
        lbl = "✓" if p.get("correct_label") else "✗"
        warn = " ⚠️" if de > 0.05 else ""
        lines.append(
            f"| {h:.2f} | {de:.4e} | {f:.4f} | {s:.4f} "
            f"| {ep:.4f} | {ee:.4f} | {mx:.4e} | {zz:.4e} | {lbl}{warn} |"
        )

    return lines


def _statistics(data: dict) -> list[str]:
    """Summary statistics from all deploy points."""
    pp = data["results"].get("section_4", {}).get("data", {}).get("per_point", [])
    if not pp:
        return []

    de_gaps = [p["de_gap"] for p in pp]
    fids = [p.get("fidelity", 0) for p in pp]
    n_pass = sum(1 for d in de_gaps if d < 0.05)
    n_labels = sum(1 for p in pp if p.get("correct_label"))

    lines = [
        "\n### Statistics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total deploy points | {len(pp)} |",
        f"| Pass ΔE/gap < 5% | {n_pass}/{len(pp)} ({100 * n_pass / len(pp):.0f}%) |",
        f"| ΔE/gap mean | {np.mean(de_gaps):.4e} |",
        f"| ΔE/gap median | {np.median(de_gaps):.4e} |",
        f"| ΔE/gap max | {np.max(de_gaps):.4e} |",
        f"| Fidelity mean | {np.mean(fids):.5f} |",
        f"| Fidelity min | {np.min(fids):.5f} |",
        f"| Correct labels | {n_labels}/{len(pp)} |",
    ]

    # Failures
    failures = [(p["h_test"], p["de_gap"], p.get("fidelity", 0)) for p in pp if p["de_gap"] > 0.05]
    if failures:
        lines.append(f"\n**Failures (ΔE/gap > 5%): {len(failures)} points**")
        lines.append("")
        lines.append("| h | ΔE/gap | F |")
        lines.append("|----:|-------:|-----:|")
        for h, de, f in sorted(failures):
            lines.append(f"| {h:.3f} | {de:.4e} | {f:.4f} |")

    return lines


def generate_report(files: list[Path], h_targets: list[float] | None, folder: str) -> str:
    """Generate the full markdown report."""
    lines = []
    lines.append(f"# Noiseless Pipeline Report — {folder}")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if h_targets:
        lines.append(f"  \nFiltered h-values: {', '.join(f'{h:.1f}' for h in sorted(h_targets))}")
    lines.append("")
    lines.append("---")

    for f in files:
        try:
            data = load_run(f)
        except Exception as e:
            lines.append(f"\n## ❌ Error: {f.name}\n\n`{e}`\n")
            continue

        header = _header(data)
        elapsed = data.get("elapsed_s", 0)

        lines.append(f"\n## {f.name}")
        lines.append("")
        lines.append(f"**Config:** {header}  ")
        lines.append(f"**Elapsed:** {elapsed:.1f}s")
        lines.append("")

        # Section summaries
        lines.extend(_section_summary(data))
        lines.append("")

        # Per-h table
        lines.append("\n### Per-h Deploy Points")
        lines.append("")
        lines.extend(_per_h_table(data, h_targets))

        # Statistics
        lines.extend(_statistics(data))
        lines.append("\n---")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate markdown report for noiseless per-h results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a run_*.json file or directory containing them",
    )
    parser.add_argument(
        "--h",
        "-H",
        type=float,
        nargs="+",
        default=None,
        help="Specific h-values to include (default: all)",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Include all deploy points",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output markdown file (default: stdout)",
    )
    args = parser.parse_args()

    target = Path(args.path)
    h_targets = args.h if not args.all else None

    if target.is_file():
        files = [target]
        folder = target.parent.name
    elif target.is_dir():
        files = sorted(target.glob("run_*.json"))
        folder = target.name
        if not files:
            print(f"No run_*.json files found in {target}", file=sys.stderr)
            return 1
    else:
        print(f"Path not found: {target}", file=sys.stderr)
        return 1

    report = generate_report(files, h_targets, folder)

    if args.output:
        Path(args.output).write_text(report)
        print(f"Report saved to {args.output}", file=sys.stderr)
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
