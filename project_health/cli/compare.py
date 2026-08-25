#!/usr/bin/env python3
"""Cross-experiment and pipeline result comparison CLI.

Each experiment is evaluated against its own success criteria — not a
blanket ΔE/gap baseline. Verdicts: confirmed (hypothesis holds),
rejected (hypothesis disproved = valid finding), failed (unexpected).

Usage:
    .venv/bin/python scripts/compare.py --all
    .venv/bin/python scripts/compare.py --exp G1 G5
    .venv/bin/python scripts/compare.py --category G
    .venv/bin/python scripts/compare.py --noisy
    .venv/bin/python scripts/compare.py --noisy --group-by seed_layout
    .venv/bin/python scripts/compare.py --all --json output.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --all                          Compare all experiments
    %(prog)s --exp G1 G5 B4                 Compare specific experiments
    %(prog)s --category G                   Compare by category letter
    %(prog)s --noisy                        Analyze ZNE robustness results
    %(prog)s --noisy --group-by n_layouts   Group noisy results by key
    %(prog)s --all --json results.json      Save JSON output
        """,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="Compare all experiments")
    mode.add_argument("--exp", nargs="+", dest="experiments", help="Experiment IDs")
    mode.add_argument("--category", type=str, help="Category letter or name")
    mode.add_argument("--noisy", action="store_true", help="Analyze noisy/ZNE results (legacy)")
    mode.add_argument("--zne", action="store_true", help="Analyze GF/PEA ZNE results (new)")

    parser.add_argument("--group-by", type=str, help="Group noisy results by key")
    parser.add_argument("--noisy-file", type=str, help="Specific noisy result file")
    parser.add_argument(
        "--format", choices=["table", "json"], default="table", help="Output format"
    )
    parser.add_argument("--json", type=str, metavar="FILE", dest="json_file", help="Save to file")
    parser.add_argument("--results-dir", type=str, default=None, help="Results directory")

    return parser.parse_args()


def _run_experiment_comparison(store, exp_ids: list[str], args) -> None:
    """Run experiment comparison mode."""
    comparisons = store.compare_experiments(exp_ids)
    if not comparisons:
        print("No comparable results found.")
        return

    if args.json_file:
        _write_json(comparisons, args.json_file)
        return
    if args.format == "json":
        print(json.dumps(comparisons, indent=2, default=str))
        return

    print("\nExperiment Results Summary")
    print("=" * 80)
    print(store.format_experiment_table(comparisons))
    print()

    confirmed = [c for c in comparisons if c["verdict"] == "confirmed"]
    rejected = [c for c in comparisons if c["verdict"] == "rejected"]
    failed = [c for c in comparisons if c["verdict"] == "failed"]
    print(f"  {len(confirmed)} confirmed ✅  {len(rejected)} rejected ⚠️  {len(failed)} failed ❌")

    if rejected:
        print("\n  Rejected hypotheses (valid findings):")
        for c in rejected:
            print(f"    {c['experiment_id']}: {c['hypothesis'][:65]}")
    if failed:
        print("\n  Failed experiments:")
        for c in failed:
            print(f"    {c['experiment_id']}: {c['hypothesis'][:65]}")


def _run_noisy_analysis(store, args) -> None:
    """Run noisy/ZNE analysis mode."""
    results = store.load_noisy_results(filename=args.noisy_file)
    if not results:
        print("No noisy experiment results found in exp_noisy_variants/")
        return

    if args.json_file:
        output = {
            "correlations": store.analyze_noisy_correlations(results),
            "by_group": (
                {args.group_by: store.analyze_noisy_by_group(results, args.group_by)}
                if args.group_by
                else {}
            ),
        }
        _write_json(output, args.json_file)
        return

    if args.format == "json":
        correlations = store.analyze_noisy_correlations(results)
        print(json.dumps(correlations, indent=2))
        return

    # Table output
    correlations = store.analyze_noisy_correlations(results)
    n = int(correlations.get("n_evaluations", len(results)))
    print(f"\nNoisy/ZNE Analysis ({n} evaluations)")
    print("=" * 60)
    print(f"  Mean R²:          {correlations.get('mean_r2', 0):.4f}")
    print(f"  R² > 0.8:         {correlations.get('pct_r2_gt_08', 0):.1f}%")
    print(f"  ZNE helps:        {correlations.get('pct_helps', 0):.1f}%")
    print(f"  Mean gain:        {correlations.get('mean_gain_pct', 0):+.1f}%")
    if "corr_r2_gain" in correlations:
        print(f"  Corr(R², gain):   {correlations['corr_r2_gain']:.4f}")
    if "corr_ces_ratio_r2" in correlations:
        print(f"  Corr(CES ratio, R²): {correlations['corr_ces_ratio_r2']:.4f}")

    # Group-by analysis
    keys = [args.group_by] if args.group_by else ["seed_layout", "n_layouts", "h_test"]
    for key in keys:
        if not any(key in r for r in results):
            continue
        grouped = store.analyze_noisy_by_group(results, key)
        if grouped:
            print(f"\n  By {key}:")
            print(store.format_noisy_table(grouped, key))


def _write_json(data, filepath: str) -> None:
    """Write data to JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    n = len(data) if isinstance(data, list) else "structured"
    print(f"Saved to {path} ({n} entries)")


def _run_zne_analysis(args) -> None:
    """Analyze GF-ZNE and PEA-ZNE results across all new experiments.

    Scans exp_gf_zne_cmp/, exp_zne_3way/, exp_pea_zne_val/,
    exp_pea_hw_ready/, exp_pea_pipeline/ and produces a consolidated
    cross-method comparison.
    """
    results_root = Path(args.results_dir) if args.results_dir else Path("results/experiments")

    zne_dirs = [
        "exp_gf_zne_cmp",
        "exp_zne_3way",
        "exp_pea_zne_val",
        "exp_pea_hw_ready",
        "exp_pea_pipeline",
        "exp_zne_cross_topo",
        "exp_pea_triangular",
    ]

    all_points: list[dict] = []

    for dirname in zne_dirs:
        exp_dir = results_root / dirname
        if not exp_dir.exists():
            continue

        from qmbp_simulation.framework.result_io import load_results_from_dir

        for f, data in load_results_from_dir(exp_dir, recursive=True):

            config = data.get("config", {})
            system = config.get("system", {})
            topology = system.get("topology", "?")
            n_qubits = system.get("n_qubits", 0)
            exp_id = config.get("experiment_id", dirname)

            # Extract per-h comparison data from section_4 or section_5
            results_sections = data.get("results", {})
            comparison_section = None
            for sec_key in ["section_5", "section_4"]:
                sec = results_sections.get(sec_key, {}).get("data", {})
                if sec.get("comparison") or sec.get("summary", {}).get("comparison"):
                    comparison_section = sec
                    break
            if comparison_section is None:
                comparison_section = {}

            comparison = comparison_section.get("comparison", [])
            if not comparison:
                comparison = comparison_section.get("summary", {}).get("comparison", [])

            if comparison:
                summary = comparison_section.get("summary", {})
                for row in comparison:
                    point = {
                        "experiment": exp_id,
                        "file": f.name,
                        "topology": topology or summary.get("topology", "?"),
                        "n_qubits": n_qubits or summary.get("n_qubits", 0),
                        "h": row.get("h", 0),
                    }
                    # GF data
                    if "de_gf_zne" in row or "de_gf" in row:
                        point["gf_de"] = row.get("de_gf_zne") or row.get("de_gf", None)
                        point["gf_r2"] = (
                            row.get("gf_zne_r2") or row.get("gf_r2") or row.get("r2_gf_zne")
                        )
                        point["gf_gain"] = (
                            row.get("gf_zne_gain") or row.get("gf_gain") or row.get("gain_gf_zne")
                        )
                    # PEA data
                    if "de_pea_zne" in row or "de_pea" in row:
                        point["pea_de"] = row.get("de_pea_zne") or row.get("de_pea", None)
                        point["pea_r2"] = (
                            row.get("pea_zne_r2") or row.get("pea_r2") or row.get("r2_pea_zne")
                        )
                        point["pea_gain"] = (
                            row.get("pea_zne_gain")
                            or row.get("pea_gain")
                            or row.get("gain_pea_zne")
                        )
                    # CES data
                    if "de_ces_zne" in row:
                        point["ces_de"] = row.get("de_ces_zne")
                        point["ces_r2"] = (
                            row.get("ces_zne_r2") or row.get("ces_r2") or row.get("r2_ces_zne")
                        )
                        point["ces_gain"] = (
                            row.get("ces_zne_gain")
                            or row.get("ces_gain")
                            or row.get("gain_ces_zne")
                        )

                    all_points.append(point)

            # Also check PEA-specific results in section_3
            s3 = results_sections.get("section_3", {}).get("data", {})
            if "mean_gain" in s3 and not comparison:
                # PEA_ZNE_VAL or PEA_HW_READY without section_4 comparison
                s3_results = s3.get("results", [])
                if isinstance(s3_results, list):
                    for r in s3_results:
                        all_points.append(
                            {
                                "experiment": exp_id,
                                "file": f.name,
                                "topology": topology,
                                "n_qubits": n_qubits,
                                "h": r.get("h", 0),
                                "pea_de": r.get("de_gap_pea") or r.get("de_pea"),
                                "pea_r2": r.get("pea_r2"),
                                "pea_gain": r.get("pea_gain"),
                            }
                        )

            # Fallback: check section_1.results for PEA/GF per-point data
            # (handles experiments like PEA_TRIANGULAR that use section_1 directly)
            if not comparison:
                s1 = results_sections.get("section_1", {}).get("data", {})
                s1_results = s1.get("results", [])
                if isinstance(s1_results, list) and s1_results:
                    first = s1_results[0]
                    # Only process if it has ZNE fields (de_pea, de_gf, pea_gain, etc.)
                    if any(k in first for k in ("de_pea", "pea_gain", "de_gf", "gf_gain")):
                        for row in s1_results:
                            row_topo = row.get("topology", topology)
                            row_n = row.get("n_qubits", n_qubits)
                            point = {
                                "experiment": exp_id,
                                "file": f.name,
                                "topology": row_topo,
                                "n_qubits": row_n,
                                "h": row.get("h", 0),
                            }
                            if "de_gf" in row or "gf_gain" in row:
                                point["gf_de"] = row.get("de_gf")
                                point["gf_r2"] = row.get("gf_r2")
                                point["gf_gain"] = row.get("gf_gain")
                            if "de_pea" in row or "pea_gain" in row:
                                point["pea_de"] = row.get("de_pea")
                                point["pea_r2"] = row.get("pea_r2")
                                point["pea_gain"] = row.get("pea_gain")
                            all_points.append(point)

    if not all_points:
        print("No ZNE experiment data found.")
        return

    if args.json_file:
        _write_json(all_points, args.json_file)
        return

    # Aggregate statistics
    import numpy as np

    gf_gains = [p["gf_gain"] for p in all_points if p.get("gf_gain") is not None]
    pea_gains = [p["pea_gain"] for p in all_points if p.get("pea_gain") is not None]
    ces_gains = [p["ces_gain"] for p in all_points if p.get("ces_gain") is not None]
    gf_r2s = [p["gf_r2"] for p in all_points if p.get("gf_r2") is not None]
    pea_r2s = [p["pea_r2"] for p in all_points if p.get("pea_r2") is not None]

    print(f"\nZNE Technique Analysis ({len(all_points)} h-point evaluations)")
    print("=" * 70)

    if ces_gains:
        print("\n  CES-ZNE (inhomogeneous layout):")
        print(f"    N evaluations: {len(ces_gains)}")
        print(f"    Mean gain:     {np.mean(ces_gains):+.1%}")
        print(f"    Always helps:  {sum(1 for g in ces_gains if g > 0)}/{len(ces_gains)}")

    if gf_gains:
        print("\n  Gate-Folding ZNE:")
        print(f"    N evaluations: {len(gf_gains)}")
        print(f"    Mean gain:     {np.mean(gf_gains):+.1%}")
        print(f"    Mean R²:       {np.mean(gf_r2s):.4f}")
        print(f"    Always helps:  {sum(1 for g in gf_gains if g > 0)}/{len(gf_gains)}")

    if pea_gains:
        print("\n  PEA-ZNE (probabilistic error amplification):")
        print(f"    N evaluations: {len(pea_gains)}")
        print(f"    Mean gain:     {np.mean(pea_gains):+.1%}")
        print(f"    Mean R²:       {np.mean(pea_r2s):.4f}")
        print(f"    Always helps:  {sum(1 for g in pea_gains if g > 0)}/{len(pea_gains)}")

    # By topology
    topologies = sorted(set(p["topology"] for p in all_points if p["topology"] != "?"))
    if len(topologies) > 1:
        print("\n  By Topology:")
        header = f"    {'Topo':<12} {'N':>3} {'GF gain':>9} {'PEA gain':>10}"
        header += f" {'GF R²':>6} {'PEA R²':>7}"
        print(header)
        print(f"    {'-' * 12} {'-' * 3} {'-' * 9} {'-' * 10} {'-' * 6} {'-' * 7}")
        for topo in topologies:
            topo_pts = [p for p in all_points if p["topology"] == topo]
            n = topo_pts[0]["n_qubits"] if topo_pts else "?"
            topo_gf = [p["gf_gain"] for p in topo_pts if p.get("gf_gain") is not None]
            topo_pea = [p["pea_gain"] for p in topo_pts if p.get("pea_gain") is not None]
            topo_gf_r2 = [p["gf_r2"] for p in topo_pts if p.get("gf_r2") is not None]
            topo_pea_r2 = [p["pea_r2"] for p in topo_pts if p.get("pea_r2") is not None]
            gf_str = f"{np.mean(topo_gf):+.1%}" if topo_gf else "—"
            pea_str = f"{np.mean(topo_pea):+.1%}" if topo_pea else "—"
            gf_r2_str = f"{np.mean(topo_gf_r2):.3f}" if topo_gf_r2 else "—"
            pea_r2_str = f"{np.mean(topo_pea_r2):.3f}" if topo_pea_r2 else "—"
            print(f"    {topo:<12} {n:>3} {gf_str:>9} {pea_str:>10} {gf_r2_str:>6} {pea_r2_str:>7}")

    # Coverage matrix
    print("\n  Coverage Matrix:")
    print(f"    {'Config':<20} {'CES':>5} {'GF':>5} {'PEA':>5}")
    print(f"    {'-' * 20} {'-' * 5} {'-' * 5} {'-' * 5}")
    configs = sorted(set((p["topology"], p["n_qubits"]) for p in all_points))
    for topo, n in configs:
        cfg_pts = [p for p in all_points if p["topology"] == topo and p["n_qubits"] == n]
        has_ces = any(p.get("ces_gain") is not None for p in cfg_pts)
        has_gf = any(p.get("gf_gain") is not None for p in cfg_pts)
        has_pea = any(p.get("pea_gain") is not None for p in cfg_pts)
        ces_icon = "  Y  " if has_ces else "  -  "
        gf_icon = "  Y  " if has_gf else "  -  "
        pea_icon = "  Y  " if has_pea else "  -  "
        print(f"    {topo} N={n:<10} {ces_icon}{gf_icon}{pea_icon}")

    # Gaps
    print("\n  Gaps (missing method × config):")
    gaps = []
    for topo, n in configs:
        cfg_pts = [p for p in all_points if p["topology"] == topo and p["n_qubits"] == n]
        if not any(p.get("pea_gain") is not None for p in cfg_pts):
            gaps.append(f"    PEA missing on {topo} N={n}")
        if not any(p.get("gf_gain") is not None for p in cfg_pts):
            gaps.append(f"    GF missing on {topo} N={n}")
    if gaps:
        for g in gaps:
            print(g)
    else:
        print("    None — all methods tested on all configs")

    print()


def main() -> None:
    args = parse_args()

    from qmbp_simulation.framework import ResultStore

    results_dir = Path(args.results_dir) if args.results_dir else None
    store = ResultStore(results_root=results_dir)

    if args.noisy:
        _run_noisy_analysis(store, args)
        return

    if args.zne:
        _run_zne_analysis(args)
        return

    # Determine experiment IDs
    available = store.list_experiments()

    if args.all:
        exp_ids = available
    elif args.category:
        exp_ids = store.resolve_category(args.category, available)
    elif args.experiments:
        exp_ids = [e.upper() for e in args.experiments]
    else:
        print("Specify --all, --exp, --category, or --noisy")
        sys.exit(1)

    exp_ids = [e for e in exp_ids if e in available]
    if not exp_ids:
        print("No results found.")
        if available:
            print(f"Available: {', '.join(available)}")
        return

    _run_experiment_comparison(store, exp_ids, args)


if __name__ == "__main__":
    main()
