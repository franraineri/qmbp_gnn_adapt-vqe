#!/usr/bin/env python3
"""CLI entry point for the digest tool.

Run with: python -m project_health.digest [options]
Or:       python project_health/digest/__main__.py [options]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from project_health.digest.formatters import (
    format_compare_two,
    format_cross_topology_text,
    format_experiment_text,
    format_markdown,
    format_noiseless_grouped,
    format_noiseless_outliers,
    format_noiseless_stats,
    format_noiseless_text,
    format_noisy_grouped,
    format_noisy_stats,
    format_noisy_text,
    format_scaling_text,
)
from project_health.digest.models import NoiselessResult, NoisyResult
from project_health.digest.scanner import ResultScanner

# ── Sort keys ────────────────────────────────────────────────────────────

SORT_KEYS_NOISELESS = {
    "delta_e": lambda r: r.delta_e_over_gap if r.delta_e_over_gap is not None else 999,
    "time": lambda r: r.elapsed_s,
    "smoothness": lambda r: r.theta_smoothness if r.theta_smoothness is not None else 999,
    "gap": lambda r: r.generalization_gap if r.generalization_gap is not None else 999,
    "folder": lambda r: r.folder,
}

SORT_KEYS_NOISY = {
    "r2": lambda r: -r.mean_r2,
    "gain": lambda r: -r.mean_gain_pct,
    "time": lambda r: r.elapsed_s,
    "folder": lambda r: r.folder,
}

SORT_KEYS_EXPERIMENT = {
    "id": lambda r: r.experiment_id,
    "verdict": lambda r: {"confirmed": 0, "rejected": 1, "failed": 2}.get(r.verdict, 3),
    "de_gap": lambda r: r.mean_de_gap if r.mean_de_gap is not None else 999,
    "pass_rate": lambda r: -(r.pass_rate or 0),
    "folder": lambda r: r.folder,
}


# ── Filters ──────────────────────────────────────────────────────────────


def apply_filters(
    noiseless: list[NoiselessResult],
    noisy: list[NoisyResult],
    experiments: list,
    *,
    topology: str | None = None,
    n_qubits: int | None = None,
    p_layers: int | None = None,
    model: str | None = None,
) -> tuple[list[NoiselessResult], list[NoisyResult], list]:
    """Filter results by system parameters."""
    if topology:
        t = topology.lower()
        noiseless = [r for r in noiseless if t in r.topology.lower()]
        noisy = [r for r in noisy if t in r.topology.lower()]
        experiments = [r for r in experiments if t in r.topology.lower()]

    if n_qubits is not None:
        noiseless = [r for r in noiseless if r.n_qubits == n_qubits]
        noisy = [r for r in noisy if r.n_qubits == n_qubits]
        experiments = [r for r in experiments if r.n_qubits == n_qubits]

    if p_layers is not None:
        noiseless = [r for r in noiseless if r.p_layers == p_layers]
        noisy = [r for r in noisy if r.p_layers == p_layers]
        experiments = [r for r in experiments if r.p_layers == p_layers]

    if model:
        m = model.lower()
        noiseless = [r for r in noiseless if m in r.model.lower()]
        experiments = [r for r in experiments if m in r.model.lower()]

    return noiseless, noisy, experiments


# ── CLI ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Digest experiment results by kind — extract key knowledge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m project_health.digest                         All results by kind
    python -m project_health.digest --kind noiseless        Only noiseless runs
    python -m project_health.digest --kind noisy            Only noisy/ZNE
    python -m project_health.digest --kind experiment       Only hypothesis tests
    python -m project_health.digest --topology ladder       Filter by topology
    python -m project_health.digest --n-qubits 10           Filter by system size
    python -m project_health.digest --folder variants_N10_ladder
    python -m project_health.digest --sort delta_e          Sort by ΔE/gap
    python -m project_health.digest --markdown -o digest.md
    python -m project_health.digest --json digest.json
    python -m project_health.digest --verbose
        """,
    )

    parser.add_argument(
        "--kind",
        choices=["noiseless", "noisy", "experiment", "cross_topology", "scaling", "all"],
        default="all",
        help="Which result kind to digest (default: all)",
    )

    filt = parser.add_argument_group("filters")
    filt.add_argument("--topology", type=str, help="Filter by topology")
    filt.add_argument("--n-qubits", type=int, help="Filter by system size")
    filt.add_argument("--p-layers", type=int, help="Filter by ansatz depth")
    filt.add_argument(
        "--model", type=str, help="Filter by model type (tfim, tfim_longitudinal, heisenberg)"
    )
    filt.add_argument("--folder", type=str, help="Specific folder to scan")

    parser.add_argument(
        "--sort",
        type=str,
        default=None,
        help="Sort key (noiseless: delta_e|time|smoothness|gap; "
        "noisy: r2|gain|time; experiment: id|verdict|de_gap|pass_rate)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        metavar="N",
        help="Show only the top N results per kind (after sorting)",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default=None,
        dest="group_by",
        help="Group results by a dimension for comparison. "
        "Noiseless: topology|n_qubits|hidden_dim|n_restarts|p_layers. "
        "Noisy: topology|n_qubits|n_layouts|shots|p_layers.",
    )

    analysis = parser.add_argument_group("analysis")
    analysis.add_argument(
        "--stats",
        action="store_true",
        help="Show detailed statistical summary (percentiles, distribution)",
    )
    analysis.add_argument(
        "--outliers",
        action="store_true",
        help="Detect and explain outlier results (IQR method)",
    )
    analysis.add_argument(
        "--compare",
        nargs=2,
        metavar=("A", "B"),
        help="Side-by-side comparison of two folders/variants",
    )
    analysis.add_argument(
        "--exclude-tests",
        action="store_true",
        help="Exclude test artifacts (T1A, TEST, FAIL, XFAIL, etc.) from experiment list",
    )

    out = parser.add_argument_group("output")
    out.add_argument("--markdown", action="store_true", help="Markdown output")
    out.add_argument("--verbose", action="store_true", help="Include details")
    out.add_argument("--json", type=str, metavar="FILE", help="Save as JSON")
    out.add_argument("-o", "--output", type=str, help="Save text/md to file")

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results root directory (default: results)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configure logging so scanner progress goes to stderr
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )

    scanner = ResultScanner(results_root=Path(args.results_dir))

    _log(f"[digest] Results dir: {args.results_dir}")

    # Scan
    if args.folder:
        _log(f"[digest] Scanning folder: {args.folder}")
        noiseless, noisy, experiments = scanner.scan_folder(args.folder)
    elif args.kind == "scaling":
        # Skip expensive scan_all() when only scaling is requested
        noiseless, noisy, experiments = [], [], []
        _log("[digest] Kind=scaling: skipping experiment/thesis scan")
    else:
        _log("[digest] Scanning all result areas...")
        noiseless, noisy, experiments = scanner.scan_all(
            exclude_tests=args.exclude_tests,
        )

    # Also scan cross-topology results (separate path: results/scaling/cross_topology/)
    cross_topology = scanner.scan_cross_topology() if args.kind in ("cross_topology", "all") else []

    # Also scan scaling results (MPS validation runs + mode comparison + N=120)
    if args.kind in ("scaling", "all"):
        scaling_results = scanner.scan_scaling()
        mode_comparison = scanner.scan_mode_comparison()
        n120_sweep = scanner.scan_n120_sweep()
    else:
        scaling_results = []
        mode_comparison = None
        n120_sweep = None

    _log(
        f"[digest] Scanned: {len(noiseless)} noiseless, "
        f"{len(noisy)} noisy, {len(experiments)} experiments, "
        f"{len(cross_topology)} cross-topology, "
        f"{len(scaling_results)} scaling"
    )

    # Filter
    active_filters = []
    if args.topology:
        active_filters.append(f"topology={args.topology}")
    if args.n_qubits is not None:
        active_filters.append(f"n_qubits={args.n_qubits}")
    if args.p_layers is not None:
        active_filters.append(f"p_layers={args.p_layers}")
    if args.model:
        active_filters.append(f"model={args.model}")

    if active_filters:
        _log(f"[digest] Applying filters: {', '.join(active_filters)}")

    noiseless, noisy, experiments = apply_filters(
        noiseless,
        noisy,
        experiments,
        topology=args.topology,
        n_qubits=args.n_qubits,
        p_layers=args.p_layers,
        model=args.model,
    )

    if active_filters:
        _log(
            f"[digest] After filters: {len(noiseless)} noiseless, "
            f"{len(noisy)} noisy, {len(experiments)} experiments"
        )

    # Filter by kind (clear what's not requested)
    if args.kind == "noiseless":
        noisy, experiments, cross_topology, scaling_results = [], [], [], []
        mode_comparison, n120_sweep = None, None
        _log("[digest] Kind filter: noiseless only")
    elif args.kind == "noisy":
        noiseless, experiments, cross_topology, scaling_results = [], [], [], []
        mode_comparison, n120_sweep = None, None
        _log("[digest] Kind filter: noisy only")
    elif args.kind == "experiment":
        noiseless, noisy, cross_topology, scaling_results = [], [], [], []
        mode_comparison, n120_sweep = None, None
        _log("[digest] Kind filter: experiment only")
    elif args.kind == "cross_topology":
        noiseless, noisy, experiments, scaling_results = [], [], [], []
        mode_comparison, n120_sweep = None, None
        _log("[digest] Kind filter: cross-topology only")
    elif args.kind == "scaling":
        noiseless, noisy, experiments, cross_topology = [], [], [], []
        _log("[digest] Kind filter: scaling only")

    # Sort
    if args.sort:
        key = args.sort.lower()
        _log(f"[digest] Sorting by: {key}")
        if noiseless and key in SORT_KEYS_NOISELESS:
            noiseless.sort(key=SORT_KEYS_NOISELESS[key])
        if noisy and key in SORT_KEYS_NOISY:
            noisy.sort(key=SORT_KEYS_NOISY[key])
        if experiments and key in SORT_KEYS_EXPERIMENT:
            experiments.sort(key=SORT_KEYS_EXPERIMENT[key])

    # Top-N truncation
    if args.top is not None:
        _log(f"[digest] Truncating to top {args.top}")
        noiseless = noiseless[: args.top]
        noisy = noisy[: args.top]
        experiments = experiments[: args.top]

    # Check results
    n_scaling_total = len(scaling_results) + (1 if mode_comparison else 0) + (1 if n120_sweep else 0)
    total = len(noiseless) + len(noisy) + len(experiments) + len(cross_topology) + n_scaling_total
    if total == 0:
        print("No results found matching the specified filters.")
        sys.exit(0)

    # ── Compare mode (special: scans two folders independently)
    if args.compare:
        label_a, label_b = args.compare
        _log(f"[digest] Compare mode: '{label_a}' vs '{label_b}'")
        _log(f"[digest] Scanning folder A: {label_a}")
        nl_a, _, _ = scanner.scan_folder(label_a)
        _log(f"[digest]   → {len(nl_a)} noiseless results")
        _log(f"[digest] Scanning folder B: {label_b}")
        nl_b, _, _ = scanner.scan_folder(label_b)
        _log(f"[digest]   → {len(nl_b)} noiseless results")
        if not nl_a:
            print(f"No noiseless results found for '{label_a}'")
            sys.exit(1)
        if not nl_b:
            print(f"No noiseless results found for '{label_b}'")
            sys.exit(1)
        _log("[digest] Computing comparison metrics...")
        output = format_compare_two(nl_a, nl_b, label_a, label_b)
        _write_output(output, args.output)
        return

    # ── Stats mode
    if args.stats:
        _log("[digest] Computing statistical summary...")
        sections: list[str] = []
        if noiseless:
            _log(f"[digest]   Noiseless stats ({len(noiseless)} results)")
            sections.append(format_noiseless_stats(noiseless))
        if noisy:
            _log(f"[digest]   Noisy stats ({len(noisy)} results)")
            sections.append(format_noisy_stats(noisy))
        output = "\n".join(sections)
        _write_output(output, args.output)
        return

    # ── Outliers mode
    if args.outliers:
        _log(f"[digest] Running outlier detection ({len(noiseless)} noiseless results)...")
        output = format_noiseless_outliers(noiseless)
        _write_output(output, args.output)
        return

    # JSON output
    if args.json:
        _log(f"[digest] Generating JSON output ({total} results)...")
        data = {
            "noiseless": [asdict(r) for r in noiseless],
            "noisy": [asdict(r) for r in noisy],
            "experiments": [asdict(r) for r in experiments],
            "cross_topology": [asdict(r) for r in cross_topology],
            "scaling": [asdict(r) for r in scaling_results],
            "mode_comparison": asdict(mode_comparison) if mode_comparison else None,
            "n120_sweep": asdict(n120_sweep) if n120_sweep else None,
            "summary": {
                "n_noiseless": len(noiseless),
                "n_noisy": len(noisy),
                "n_experiments": len(experiments),
                "n_cross_topology": len(cross_topology),
                "n_scaling": len(scaling_results),
                "has_mode_comparison": mode_comparison is not None,
                "has_n120_sweep": n120_sweep is not None,
            },
        }
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Saved digest to {path} ({total} results)")
        return

    # Text/Markdown output
    _log(f"[digest] Formatting output ({'markdown' if args.markdown else 'text'})...")
    if args.markdown:
        output = format_markdown(noiseless, noisy, experiments, verbose=args.verbose)
    else:
        sections: list[str] = []
        if experiments:
            _log(f"[digest]   Formatting {len(experiments)} experiments")
            sections.append("═" * 80)
            sections.append(" HYPOTHESIS TESTS (BaseExperiment)")
            sections.append("═" * 80)
            sections.append(format_experiment_text(experiments, verbose=args.verbose))
            sections.append("")
        if noiseless:
            mode = f"grouped by {args.group_by}" if args.group_by else "table"
            _log(f"[digest]   Formatting {len(noiseless)} noiseless ({mode})")
            sections.append("═" * 80)
            sections.append(" NOISELESS PIPELINE RUNS")
            sections.append("═" * 80)
            if args.group_by:
                sections.append(format_noiseless_grouped(noiseless, args.group_by))
            else:
                sections.append(format_noiseless_text(noiseless, verbose=args.verbose))
            sections.append("")
        if noisy:
            mode = f"grouped by {args.group_by}" if args.group_by else "table"
            _log(f"[digest]   Formatting {len(noisy)} noisy ({mode})")
            sections.append("═" * 80)
            sections.append(" NOISY / ZNE RUNS")
            sections.append("═" * 80)
            if args.group_by:
                sections.append(format_noisy_grouped(noisy, args.group_by))
            else:
                sections.append(format_noisy_text(noisy, verbose=args.verbose))
            sections.append("")
        if cross_topology:
            _log(f"[digest]   Formatting {len(cross_topology)} cross-topology")
            sections.append("═" * 80)
            sections.append(" CROSS-TOPOLOGY TRANSFER")
            sections.append("═" * 80)
            sections.append(format_cross_topology_text(cross_topology, verbose=args.verbose))
            sections.append("")
        if scaling_results or mode_comparison or n120_sweep:
            n_items = len(scaling_results) + (1 if mode_comparison else 0) + (1 if n120_sweep else 0)
            _log(f"[digest]   Formatting {n_items} scaling results")
            sections.append("═" * 80)
            sections.append(" MPS SCALING VALIDATION")
            sections.append("═" * 80)
            sections.append(
                format_scaling_text(
                    scaling_results,
                    mode_comparison=mode_comparison,
                    n120_sweep=n120_sweep,
                    verbose=args.verbose,
                )
            )
            sections.append("")
        output = "\n".join(sections)

    _write_output(output, args.output)


# ── Helpers ──────────────────────────────────────────────────────────────


def _log(msg: str) -> None:
    """Print progress message to stderr (doesn't pollute stdout output)."""
    print(msg, file=sys.stderr)


def _write_output(output: str, filepath: str | None) -> None:
    """Write output to file or stdout."""
    if filepath:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output)
        _log(f"[digest] Done — saved to {path}")
    else:
        print(output)
        _log("[digest] Done.")


if __name__ == "__main__":
    main()
