#!/usr/bin/env python3
"""Noise-Aware & Unified MPNN Experiment Extractor.

Reads JSON results from experiment runners and outputs structured metrics.
Supports both the noise-aware comparison (UNIFIED_NOISE_COMBINED) and the
unified MPNN architecture benchmark (UNIFIED_MPNN_ARCHITECTURE).

Usage:
    # Auto-find latest noise-aware result:
    .venv/bin/python scripts/analysis/noise_aware_extractor.py

    # Unified MPNN architecture benchmark:
    .venv/bin/python scripts/analysis/noise_aware_extractor.py --experiment architecture

    # Specific result file:
    .venv/bin/python scripts/analysis/noise_aware_extractor.py --file results/experiments/exp_unified_mpnn_architecture/run_*.json

    # Output as JSON (for programmatic use):
    .venv/bin/python scripts/analysis/noise_aware_extractor.py --format json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "experiments" / "exp_unified_noise_combined"
RESULTS_ARCH = ROOT / "results" / "experiments" / "exp_unified_mpnn_architecture"
PLAN_DOC = ROOT / "internal" / "documentation" / "next-steps" / "EXPERIMENT_PLAN_04_06.md"

sys.path.insert(0, str(ROOT))


def find_latest_result(result_path: str | None = None, experiment: str = "noise") -> Path | None:
    """Find the most recent run_*.json for the specified experiment type."""
    if result_path:
        p = Path(result_path)
        return p if p.exists() else None
    base = RESULTS_ARCH if experiment == "architecture" else RESULTS
    if not base.exists():
        return None
    runs = sorted(base.glob("run_*.json"), reverse=True)
    return runs[0] if runs else None


def extract_section1(data: dict) -> dict | None:
    """Extract data collection metrics (Step A.1 + A.2)."""
    results = data.get("results", {})
    s1 = results.get("section_1", {}).get("data", {})
    if not s1:
        return None
    return {
        "n_points": s1.get("n_points"),
        "n_params": s1.get("n_params"),
        "noiseless_time_s": s1.get("noiseless_time_s"),
        "noisy_time_s": s1.get("noisy_time_s"),
        "noisy_convergence_rate": s1.get("noisy_convergence_rate"),
        "noisy_mean_de_gap": s1.get("noisy_mean_de_gap"),
        "noisy_max_de_gap": s1.get("noisy_max_de_gap"),
    }


def extract_section2(data: dict) -> dict | None:
    """Extract training metrics (Step A.3)."""
    results = data.get("results", {})
    s2 = results.get("section_2", {}).get("data", {})
    if not s2:
        return None

    variants = {}
    for vid in ["ham_noiseless", "unified_noiseless", "ham_noisy", "unified_noisy"]:
        vdata = s2.get(vid, {})
        if vdata and "final_mse" in vdata:
            variants[vid] = {
                "final_mse": vdata["final_mse"],
                "training_time_s": vdata.get("training_time_s"),
                "include_circuit": vdata.get("include_circuit_nodes"),
                "theta_source": vdata.get("theta_source"),
            }

    return {
        "variants": variants,
        "unified_mse_improvement_pct": s2.get("unified_mse_improvement_pct"),
        "graph_metrics": s2.get("graph_metrics"),
    }


def extract_section3(data: dict) -> dict | None:
    """Extract deployment metrics (Step A.4)."""
    results = data.get("results", {})
    s3 = results.get("section_3", {}).get("data", {})
    if not s3:
        return None

    deploy = {}
    for vid in ["ham_noiseless", "unified_noiseless", "ham_noisy", "unified_noisy"]:
        vdata = s3.get(vid, {})
        if vdata and "mean_de_gap" in vdata:
            deploy[vid] = {
                "mean_de_gap": vdata["mean_de_gap"],
                "max_de_gap": vdata["max_de_gap"],
                "pass_rate": vdata["pass_rate"],
                "n_pass": vdata["n_pass"],
                "n_total": vdata["n_total"],
                "noiseless_mean_de_gap": vdata.get("noiseless_mean_de_gap"),
                "noiseless_pass_rate": vdata.get("noiseless_pass_rate"),
            }

    return {"variants": deploy, "n_test_points": s3.get("n_test_points")}


def extract_section4(data: dict) -> dict | None:
    """Extract statistical analysis (Step A.5)."""
    results = data.get("results", {})
    s4 = results.get("section_4", {}).get("data", {})
    if not s4:
        return None

    comparisons = {}
    for key in ["noise_aware_vs_baseline", "unified_graph_vs_baseline", "combined_vs_baseline"]:
        cdata = s4.get(key, {})
        if cdata:
            tt = cdata.get("paired_ttest", {})
            ir = cdata.get("improvement_rate", {})
            comparisons[key] = {
                "mean_diff": tt.get("mean_diff"),
                "t_stat": tt.get("t_stat"),
                "p_value": tt.get("p_value"),
                "cohens_d": cdata.get("cohens_d"),
                "wins_pct": ir.get("improvement_rate_pct"),
                "significant_005": tt.get("significant_005"),
            }

    return {
        "comparisons": comparisons,
        "verdicts": s4.get("verdicts", {}),
        "ranking": s4.get("ranking", []),
    }


def format_md_report(s1: dict, s2: dict, s3: dict, s4: dict) -> str:
    """Format all extracted data as markdown tables for the plan document."""
    lines = []

    lines.append("## Extracted Results (auto-generated)\n")
    lines.append(f"**Source:** `exp_unified_noise_combined`\n")

    # Section 1: Data Collection
    if s1:
        lines.append("### Paso A.1 + A.2 — Data Collection\n")
        lines.append("| Métrica | Valor |")
        lines.append("|---------|-------|")
        lines.append(f"| N h-points | {s1['n_points']} |")
        lines.append(f"| N params | {s1['n_params']} |")
        lines.append(f"| Noiseless VQE time (s) | {s1['noiseless_time_s']:.1f} |")
        lines.append(f"| Noisy VQE time (s) | {s1['noisy_time_s']:.1f} |")
        lines.append(f"| Noisy convergence rate | {s1['noisy_convergence_rate']:.0%} |")
        lines.append(f"| Noisy mean ΔE/gap | {s1['noisy_mean_de_gap']:.4f} |")
        lines.append(f"| Noisy max ΔE/gap | {s1['noisy_max_de_gap']:.4f} |")
        lines.append("")

    # Section 2: Training
    if s2 and s2.get("variants"):
        lines.append("### Paso A.3 — Training Metrics\n")
        vids = ["ham_noiseless", "unified_noiseless", "ham_noisy", "unified_noisy"]
        labels = ["A", "B", "C", "D"]
        lines.append("| Métrica | A | B | C | D |")
        lines.append("|---------|---|---|---|---|")

        row_mse = "| Final MSE |"
        row_time = "| Training time (s) |"
        for vid in vids:
            v = s2["variants"].get(vid, {})
            mse = v.get("final_mse")
            t = v.get("training_time_s")
            row_mse += f" {mse:.2e} |" if mse else " — |"
            row_time += f" {t:.1f} |" if t else " — |"
        lines.append(row_mse)
        lines.append(row_time)

        if s2.get("unified_mse_improvement_pct") is not None:
            lines.append(
                f"\n**MSE improvement B vs A:** {s2['unified_mse_improvement_pct']:.1f}%"
            )

        gm = s2.get("graph_metrics", {})
        if gm:
            lines.append(f"\n**Graph metrics:** {gm.get('total_nodes')} nodes, "
                         f"{gm.get('node_expansion_ratio', 0):.1f}× expansion, "
                         f"density={gm.get('graph_density', 0):.4f}")
        lines.append("")

    # Section 3: Deployment
    if s3 and s3.get("variants"):
        lines.append("### Paso A.4 — Noisy Deployment\n")
        vids = ["ham_noiseless", "unified_noiseless", "ham_noisy", "unified_noisy"]
        lines.append("| Métrica | A | B | C | D |")
        lines.append("|---------|---|---|---|---|")

        for metric, label in [
            ("mean_de_gap", "Mean ΔE/gap (noisy)"),
            ("max_de_gap", "Max ΔE/gap (noisy)"),
            ("pass_rate", "Pass rate @ 5%"),
            ("noiseless_mean_de_gap", "Mean ΔE/gap (noiseless)"),
            ("noiseless_pass_rate", "Pass rate (noiseless)"),
        ]:
            row = f"| {label} |"
            for vid in vids:
                v = s3["variants"].get(vid, {})
                val = v.get(metric)
                if val is None:
                    row += " — |"
                elif "rate" in metric:
                    row += f" {val:.0%} |"
                else:
                    row += f" {val:.4f} |"
            lines.append(row)
        lines.append("")

    # Section 4: Statistics
    if s4 and s4.get("comparisons"):
        lines.append("### Paso A.5 — Paired Comparisons\n")
        lines.append("| Comparison | Mean diff | t-stat | p-value | Cohen's d | % wins |")
        lines.append("|------------|-----------|--------|---------|-----------|--------|")

        comp_labels = {
            "noise_aware_vs_baseline": "C vs A (#06)",
            "unified_graph_vs_baseline": "B vs A (#04)",
            "combined_vs_baseline": "D vs A (combined)",
        }
        for key, label in comp_labels.items():
            c = s4["comparisons"].get(key, {})
            if c:
                sig = "✓" if c.get("significant_005") else "✗"
                lines.append(
                    f"| {label} | {c.get('mean_diff', 0):.4f} | "
                    f"{c.get('t_stat', 0):.2f} | {c.get('p_value', 1):.4f} | "
                    f"{c.get('cohens_d', 0):.2f} | {c.get('wins_pct', 0):.0f}% {sig} |"
                )
        lines.append("")

        if s4.get("ranking"):
            lines.append("**Ranking (best → worst):** " + " > ".join(
                f"{r['variant']}({r['mean_de_gap']:.4f})" for r in s4["ranking"]
            ))
            lines.append("")

        if s4.get("verdicts"):
            lines.append("**Verdicts:**")
            for crit, v in s4["verdicts"].items():
                lines.append(f"  - {crit}: {v.get('verdict')} — {v.get('desc')}")
            lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Architecture Benchmark Extraction (UNIFIED_MPNN_ARCHITECTURE)
# ═══════════════════════════════════════════════════════════════════════════════


def extract_arch_vqe(data: dict) -> dict | None:
    """Extract VQE data collection from architecture benchmark."""
    results = data.get("results", {})
    s1 = results.get("section_1", {}).get("data", {})
    if not s1:
        return None
    # Per-topology VQE quality
    topos = {k: v for k, v in s1.items() if isinstance(v, dict) and "n_params" in v}
    return {"topologies": topos, "n_topologies": s1.get("n_topologies", len(topos))}


def extract_arch_compare(data: dict) -> dict | None:
    """Extract 3-way architecture comparison results."""
    results = data.get("results", {})
    s2 = results.get("section_2", {}).get("data", {})
    if not s2:
        return None
    topos = {k: v for k, v in s2.items() if isinstance(v, dict) and "graph_metrics" in v}
    return {"topologies": topos}


def extract_arch_stats(data: dict) -> dict | None:
    """Extract statistical analysis from architecture benchmark."""
    results = data.get("results", {})
    s3 = results.get("section_3", {}).get("data", {})
    if not s3:
        return None
    return s3


def format_arch_report(vqe: dict, compare: dict, stats: dict) -> str:
    """Format architecture benchmark results as markdown."""
    lines = []
    lines.append("## Architecture Benchmark Results (auto-generated)\n")
    lines.append(f"**Experiment:** `UNIFIED_MPNN_ARCHITECTURE`\n")

    # VQE quality per topology
    if vqe and vqe.get("topologies"):
        lines.append("### VQE Data Quality\n")
        lines.append("| Topology | N params | Mean ΔE/gap | Pass@5% | VQE time |")
        lines.append("|----------|:--------:|:-----------:|:-------:|:--------:|")
        for topo, td in vqe["topologies"].items():
            lines.append(
                f"| {topo} | {td.get('n_params', '?')} | "
                f"{td.get('mean_de_gap', 0):.4f} | "
                f"{td.get('pass_rate_5pct', 0):.0%} | "
                f"{td.get('vqe_time_s', 0):.1f}s |"
            )
        lines.append("")

    # 3-way comparison per topology
    if compare and compare.get("topologies"):
        lines.append("### 3-Way Architecture Comparison\n")
        for topo, td in compare["topologies"].items():
            gm = td.get("graph_metrics", {})
            lines.append(f"#### {topo} (neighborhood_cv={gm.get('gate_neighborhood_cv', 0):.3f})\n")
            lines.append("| Variant | Architecture | Graph | MSE | Mean ΔE/gap | Pass@5% | Time |")
            lines.append("|---------|:------------:|:-----:|:---:|:-----------:|:-------:|:----:|")

            for vid in ["ham_only", "unified_brm", "unified_type"]:
                vdata = td.get(vid, {})
                if vdata:
                    lines.append(
                        f"| {vid} | {vdata.get('architecture', '?')} | "
                        f"{vdata.get('graph_type', '?')} | "
                        f"{vdata.get('final_mse', 0):.2e} | "
                        f"{vdata.get('mean_de_gap', 0):.4f} | "
                        f"{vdata.get('pass_rate', 0):.0%} | "
                        f"{vdata.get('training_time_s', 0):.1f}s |"
                    )

            comp = td.get("comparisons", {})
            if comp:
                lines.append(f"\n  Best variant: **{comp.get('best_variant', '?')}**")
                if "f_vs_a_deploy_pct" in comp:
                    lines.append(f"  F vs A deploy improvement: {comp['f_vs_a_deploy_pct']:.1f}%")
                if "f_vs_e_deploy_pct" in comp:
                    lines.append(f"  F vs E deploy improvement: {comp['f_vs_e_deploy_pct']:.1f}%")
            lines.append("")

    # Statistical analysis
    if stats:
        lines.append("### Statistical Analysis\n")
        for topo, tdata in stats.items():
            if not isinstance(tdata, dict) or "F_vs_A" not in tdata:
                continue
            lines.append(f"**{topo}** (gate_neighborhood_cv={tdata.get('gate_neighborhood_cv', 0):.3f}):")
            for label in ["F_vs_A", "E_vs_A", "F_vs_E"]:
                c = tdata.get(label, {})
                if c:
                    lines.append(
                        f"  - {label} ({c.get('description', '')}): "
                        f"d={c.get('cohens_d', 0):.2f} ({c.get('interpretation', '?')}), "
                        f"wins={c.get('wins_pct', 0):.0f}%, p={c.get('paired_ttest', {}).get('p_value', 1):.4f}"
                    )
            lines.append("")

        # Cross-topology correlation
        corr = stats.get("heterogeneity_correlation", {})
        if corr:
            lines.append(f"**Heterogeneity correlation:** r={corr.get('pearson_r', 0):.3f}")
            lines.append(f"  {corr.get('interpretation', '')}")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Noise-Aware & Unified MPNN Extractor")
    parser.add_argument("--file", type=str, default=None,
                        help="Specific result JSON file path")
    parser.add_argument("--experiment", type=str, default="noise",
                        choices=["noise", "architecture"],
                        help="Which experiment to extract: noise (2x2 ablation) or "
                        "architecture (3-way UnifiedMPNN benchmark)")
    parser.add_argument("--format", type=str, default="md",
                        choices=["md", "json", "summary"],
                        help="Output format")
    parser.add_argument("--update-plan", action="store_true",
                        help="Append results to EXPERIMENT_PLAN_04_06.md")
    args = parser.parse_args()

    path = find_latest_result(args.file, experiment=args.experiment)
    if not path:
        exp_name = "architecture benchmark" if args.experiment == "architecture" else "noise-aware comparison"
        print(f"❌ No results found for {exp_name}. Run the experiment first:")
        if args.experiment == "architecture":
            print("   .venv/bin/python scripts/experiment_runners/noise_aware/run_unified_mpnn_benchmark.py --topology ladder")
        else:
            print("   .venv/bin/python scripts/experiment_runners/noise_aware/run_noise_aware_comparison.py")
        sys.exit(1)

    print(f"📁 Reading: {path.name}")
    with open(path) as f:
        data = json.load(f)

    if args.experiment == "architecture":
        _run_architecture_extraction(data, args)
    else:
        _run_noise_extraction(data, args)


def _run_noise_extraction(data: dict, args):
    """Handle noise-aware comparison extraction (original behavior)."""
    s1 = extract_section1(data)
    s2 = extract_section2(data)
    s3 = extract_section3(data)
    s4 = extract_section4(data)

    if args.format == "json":
        output = {"section_1": s1, "section_2": s2, "section_3": s3, "section_4": s4}
        print(json.dumps(output, indent=2, default=str))
        return

    if args.format == "summary":
        print("\n=== NOISE-AWARE COMPARISON SUMMARY ===\n")
        if s1:
            print(f"  Data: {s1['n_points']} pts, {s1['n_params']} params")
            print(f"  Noisy convergence: {s1['noisy_convergence_rate']:.0%}")
        if s3:
            for vid, v in s3["variants"].items():
                print(f"  {vid}: mean_ΔE/gap={v['mean_de_gap']:.4f}, "
                      f"pass={v['pass_rate']:.0%}")
        if s4 and s4.get("ranking"):
            print(f"\n  Ranking: {' > '.join(r['variant'] for r in s4['ranking'])}")
        return

    report = format_md_report(s1, s2, s3, s4)
    print(report)

    if args.update_plan and PLAN_DOC.exists():
        with open(PLAN_DOC, "a") as f:
            f.write("\n\n---\n\n")
            f.write(report)
        print(f"\n✅ Appended results to {PLAN_DOC.relative_to(ROOT)}")


def _run_architecture_extraction(data: dict, args):
    """Handle unified MPNN architecture benchmark extraction."""
    vqe = extract_arch_vqe(data)
    compare = extract_arch_compare(data)
    stats = extract_arch_stats(data)

    if args.format == "json":
        output = {"vqe": vqe, "comparison": compare, "statistics": stats}
        print(json.dumps(output, indent=2, default=str))
        return

    if args.format == "summary":
        print("\n=== UNIFIED MPNN ARCHITECTURE BENCHMARK SUMMARY ===\n")
        if compare and compare.get("topologies"):
            for topo, td in compare["topologies"].items():
                comp = td.get("comparisons", {})
                best = comp.get("best_variant", "?")
                print(f"  {topo}: best={best}")
                for vid in ["ham_only", "unified_brm", "unified_type"]:
                    vdata = td.get(vid, {})
                    if vdata:
                        print(f"    {vid}: ΔE/gap={vdata.get('mean_de_gap', 0):.4f}, "
                              f"pass={vdata.get('pass_rate', 0):.0%}")
        if stats and stats.get("heterogeneity_correlation"):
            corr = stats["heterogeneity_correlation"]
            print(f"\n  Heterogeneity correlation: r={corr.get('pearson_r', 0):.3f}")
        return

    report = format_arch_report(vqe, compare, stats)
    print(report)

    if args.update_plan and PLAN_DOC.exists():
        with open(PLAN_DOC, "a") as f:
            f.write("\n\n---\n\n")
            f.write(report)
        print(f"\n✅ Appended results to {PLAN_DOC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
