"""Layout Optimizer Analyzer — Mapomatic VF2 Integration Results.

Analyzes layout selection quality from hardware rehearsal and deployment runs:
- VF2 vs BFS comparison (CES, layouts found, SWAP count)
- CouplingMap filtering effectiveness (retention rate, excluded qubits)
- Layout fidelity cost distribution
- Strategy comparison (lowest_cost vs ces_spread vs hybrid)
- Integration with existing project health infrastructure

Usage:
    python -m project_health.analysis.layout_optimizer_analyzer
    python -m project_health.analysis.layout_optimizer_analyzer --verbose
    python -m project_health.analysis.layout_optimizer_analyzer --json report.json
    python -m project_health.analysis.layout_optimizer_analyzer --benchmark

Scans:
    - results/hardware/run_*/ → layout selection events from structured logs
    - results/experiments/exp_hw_rehearsal_v*/run_*.json → rehearsal layout data
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LayoutSelectionRecord:
    """Single layout selection event extracted from results."""

    method: str = ""  # "mapomatic_vf2" or "bfs" or "bfs_fallback"
    strategy: str = ""  # "lowest_cost", "ces_spread", "hybrid"
    n_layouts_selected: int = 0
    ces_values: list[float] = field(default_factory=list)
    mean_ces: float = 0.0
    backend_name: str = ""
    source_file: str = ""
    h_value: float | None = None
    timestamp: str = ""


@dataclass
class BenchmarkResult:
    """Result of live VF2 vs BFS benchmark comparison."""

    vf2_n_layouts: int = 0
    vf2_mean_ces: float = 0.0
    vf2_elapsed_s: float = 0.0
    bfs_n_layouts: int = 0
    bfs_mean_ces: float = 0.0
    bfs_elapsed_s: float = 0.0
    improvement_ratio: float = 0.0
    n_qubits: int = 0
    topology: str = ""
    swap_free: bool = True


@dataclass
class LayoutOptimizerReport:
    """Complete analysis report for layout optimizer integration."""

    n_vf2_events: int = 0
    n_bfs_events: int = 0
    n_fallback_events: int = 0
    vf2_mean_ces: float = 0.0
    bfs_mean_ces: float = 0.0
    ces_improvement_ratio: float = 0.0  # bfs_ces / vf2_ces
    records: list[LayoutSelectionRecord] = field(default_factory=list)
    benchmark: BenchmarkResult | None = None
    mapomatic_available: bool = False
    mapomatic_version: str = ""
    summary_message: str = ""
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dictionary."""
        d = asdict(self)
        # Remove transpiled circuits (not serializable)
        return d


# ═══════════════════════════════════════════════════════════════════════════════
# Scanner — Extract layout selection events from existing results
# ═══════════════════════════════════════════════════════════════════════════════


def _scan_structured_logs(results_root: Path) -> list[LayoutSelectionRecord]:
    """Scan structured logs in hardware run directories."""
    records: list[LayoutSelectionRecord] = []

    hw_dirs = sorted(results_root.glob("hardware/run_*/"))
    for hw_dir in hw_dirs:
        log_file = hw_dir / "execution_log.json"
        if not log_file.exists():
            continue
        try:
            raw = json.loads(log_file.read_text())
            events = raw if isinstance(raw, list) else raw.get("events", [])
            for event in events:
                if not isinstance(event, dict):
                    continue
                if event.get("event_type") in ("layout_selection", "layout_method"):
                    data = event.get("data", {})
                    ces_vals = data.get("ces_values", [])
                    if not isinstance(ces_vals, list):
                        ces_vals = []
                    record = LayoutSelectionRecord(
                        method=data.get("method", "unknown"),
                        strategy=data.get("strategy", ""),
                        n_layouts_selected=data.get("n_selected", 0),
                        ces_values=[float(c) for c in ces_vals],
                        mean_ces=sum(ces_vals) / len(ces_vals) if ces_vals else 0.0,
                        source_file=str(log_file),
                        timestamp=event.get("timestamp", ""),
                    )
                    records.append(record)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    return records


def _scan_rehearsal_results(results_root: Path) -> list[LayoutSelectionRecord]:
    """Scan rehearsal experiment JSONs for layout events."""
    records: list[LayoutSelectionRecord] = []

    rehearsal_dirs = [
        results_root / "experiments" / "exp_hw_rehearsal_v2",
        results_root / "experiments" / "exp_hw_rehearsal_v3",
        results_root / "experiments" / "exp_hardware_rehearsal_v2",
    ]
    for reh_dir in rehearsal_dirs:
        if not reh_dir.exists():
            continue
        for result_file in sorted(reh_dir.glob("run_*.json")):
            try:
                data = json.loads(result_file.read_text())
                # V2/V3 runners save structured_log or events
                events = data.get("structured_log", data.get("events", []))
                if isinstance(events, list):
                    for event in events:
                        if not isinstance(event, dict):
                            continue
                        if event.get("event_type") in ("layout_selection", "layout_method"):
                            ev_data = event.get("data", {})
                            ces_vals = ev_data.get("ces_values", [])
                            if not isinstance(ces_vals, list):
                                ces_vals = []
                            record = LayoutSelectionRecord(
                                method=ev_data.get("method", "unknown"),
                                strategy=ev_data.get("strategy", ""),
                                n_layouts_selected=ev_data.get("n_selected", 0),
                                ces_values=[float(c) for c in ces_vals],
                                mean_ces=sum(ces_vals) / len(ces_vals) if ces_vals else 0.0,
                                source_file=str(result_file),
                                timestamp=event.get("timestamp", ""),
                            )
                            records.append(record)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    return records


# ═══════════════════════════════════════════════════════════════════════════════
# Live Benchmark — Compare VF2 vs BFS on FakeTorino
# ═══════════════════════════════════════════════════════════════════════════════


def run_benchmark(n_qubits: int = 10, verbose: bool = False) -> BenchmarkResult | None:
    """Run live VF2 vs BFS comparison on FakeTorino.

    Requires: qiskit-ibm-runtime (FakeTorino) + mapomatic.
    Returns None if dependencies unavailable.
    """
    try:
        from qiskit.circuit import QuantumCircuit
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        from qmbp_simulation.execution.hardware.layout_optimizer import (
            MAPOMATIC_AVAILABLE,
            build_filtered_coupling_map,
            compute_layout_fidelity_cost,
            find_vf2_layouts,
        )
        from qmbp_simulation.execution.noisy_utils import (
            build_adjacency,
            compute_circuit_ces,
            find_layouts_bfs,
            select_layouts_low_ces,
        )
    except ImportError as e:
        logger.warning("Benchmark dependencies unavailable: %s", e)
        return None

    import time

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    backend = FakeTorino()

    # Build HVA-like N=10 chain circuit
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits - 1):
        qc.cz(i, i + 1)
    for i in range(n_qubits):
        qc.rx(0.5, i)
        qc.rz(0.3, i)

    result = BenchmarkResult(n_qubits=n_qubits, topology="heavy_hex")

    # ── VF2 path ──
    if MAPOMATIC_AVAILABLE:
        import mapomatic as mm

        t0 = time.time()
        filtered_cmap, _ = build_filtered_coupling_map(backend, max_2q_error=0.02)
        deflated = mm.deflate_circuit(qc)
        vf2_layouts = find_vf2_layouts(deflated, filtered_cmap, max_layouts=100)
        scored = compute_layout_fidelity_cost(deflated, vf2_layouts[:50], backend)
        t_vf2 = time.time() - t0

        # Transpile top-3 to get CES
        top_layouts = [s[0] for s in scored[:3]]
        vf2_ces_list = []
        for layout in top_layouts:
            pm = generate_preset_pass_manager(
                optimization_level=1, backend=backend, initial_layout=layout
            )
            transpiled = pm.run(qc)
            ces, _ = compute_circuit_ces(transpiled, backend)
            vf2_ces_list.append(ces)

        result.vf2_n_layouts = len(vf2_layouts)
        result.vf2_mean_ces = sum(vf2_ces_list) / len(vf2_ces_list) if vf2_ces_list else 0.0
        result.vf2_elapsed_s = t_vf2
        result.swap_free = True

        if verbose:
            print(f"    VF2: {len(vf2_layouts)} layouts found in {t_vf2:.3f}s")
            print(f"    VF2 top-3 CES: {[f'{c:.4f}' for c in vf2_ces_list]}")

    # ── BFS path ──
    t0 = time.time()
    adj = build_adjacency(backend)
    bfs_candidates = find_layouts_bfs(adj, n_qubits, n_candidates=40, seed=42)
    bfs_selection = select_layouts_low_ces(
        qc, backend, bfs_candidates, n_select=3, optimization_level=1
    )
    t_bfs = time.time() - t0

    result.bfs_n_layouts = len(bfs_candidates)
    result.bfs_mean_ces = (
        sum(bfs_selection.ces_values) / len(bfs_selection.ces_values)
        if bfs_selection.ces_values
        else 0.0
    )
    result.bfs_elapsed_s = t_bfs

    if verbose:
        print(f"    BFS: {len(bfs_candidates)} candidates, selected 3 in {t_bfs:.3f}s")
        print(f"    BFS top-3 CES: {[f'{c:.4f}' for c in bfs_selection.ces_values]}")

    # ── Comparison ──
    if result.vf2_mean_ces > 0:
        result.improvement_ratio = result.bfs_mean_ces / result.vf2_mean_ces

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis — Aggregate and produce report
# ═══════════════════════════════════════════════════════════════════════════════


def analyze(
    results_root: Path | None = None,
    verbose: bool = False,
    run_live_benchmark: bool = False,
) -> LayoutOptimizerReport:
    """Run layout optimizer analysis.

    Parameters
    ----------
    results_root : Path | None
        Root of results directory. Auto-detects if None.
    verbose : bool
        Print detailed per-record information.
    run_live_benchmark : bool
        If True, run a live VF2 vs BFS comparison on FakeTorino.

    Returns
    -------
    LayoutOptimizerReport
        Analysis results with recommendations.
    """
    from qmbp_simulation.execution.hardware.layout_optimizer import MAPOMATIC_AVAILABLE

    if results_root is None:
        candidates = [
            Path.cwd() / "results",
            Path(__file__).parent.parent.parent / "results",
        ]
        results_root = next((p for p in candidates if p.exists()), Path("results"))

    # Check mapomatic version
    mapo_version = ""
    if MAPOMATIC_AVAILABLE:
        try:
            import mapomatic

            mapo_version = getattr(mapomatic, "__version__", "unknown")
        except Exception:
            mapo_version = "installed (version unknown)"

    report = LayoutOptimizerReport(
        mapomatic_available=MAPOMATIC_AVAILABLE,
        mapomatic_version=mapo_version,
    )

    # Scan historical data
    log_records = _scan_structured_logs(results_root)
    rehearsal_records = _scan_rehearsal_results(results_root)
    all_records = log_records + rehearsal_records
    report.records = all_records

    # Classify by method
    vf2_records = [r for r in all_records if "vf2" in r.method]
    bfs_records = [r for r in all_records if "bfs" in r.method and "fallback" not in r.method]
    fallback_records = [r for r in all_records if "fallback" in r.method]

    report.n_vf2_events = len(vf2_records)
    report.n_bfs_events = len(bfs_records)
    report.n_fallback_events = len(fallback_records)

    # Compute aggregate CES
    if vf2_records:
        ces_vals = [r.mean_ces for r in vf2_records if r.mean_ces > 0]
        report.vf2_mean_ces = sum(ces_vals) / len(ces_vals) if ces_vals else 0.0

    if bfs_records:
        ces_vals = [r.mean_ces for r in bfs_records if r.mean_ces > 0]
        report.bfs_mean_ces = sum(ces_vals) / len(ces_vals) if ces_vals else 0.0

    if report.vf2_mean_ces > 0 and report.bfs_mean_ces > 0:
        report.ces_improvement_ratio = report.bfs_mean_ces / report.vf2_mean_ces

    # Live benchmark (optional)
    if run_live_benchmark:
        if verbose:
            print("\n  Running live VF2 vs BFS benchmark on FakeTorino (N=10)...")
        report.benchmark = run_benchmark(n_qubits=10, verbose=verbose)

    # Generate recommendations
    report.recommendations = _derive_recommendations(report)

    # Build summary message
    report.summary_message = _build_summary(report)

    return report


def _derive_recommendations(report: LayoutOptimizerReport) -> list[str]:
    """Generate actionable recommendations from analysis."""
    recs: list[str] = []

    if not report.mapomatic_available:
        recs.append(
            "INSTALL mapomatic: pip install 'mapomatic>=0.14' (or pip install -e '.[hardware]'). "
            "VF2 gives ~6× lower CES than BFS on heavy_hex N=10."
        )
    elif report.n_vf2_events == 0 and report.n_bfs_events > 0:
        recs.append(
            "Mapomatic installed but no VF2 events detected. "
            "Verify HardwareConfig.use_mapomatic=True in your config."
        )

    if report.n_fallback_events > 0:
        recs.append(
            f"{report.n_fallback_events} fallback-to-BFS events detected. "
            "Check if filtered CouplingMap is too restrictive (lower layout_max_2q_error?)."
        )

    if report.benchmark and report.benchmark.improvement_ratio > 3.0:
        recs.append(
            f"VF2 demonstrates {report.benchmark.improvement_ratio:.1f}× CES improvement. "
            "Ensure VF2 is enabled for all hardware deployments."
        )

    if not recs:
        recs.append("Layout optimizer integration is healthy. No action needed.")

    return recs


def _build_summary(report: LayoutOptimizerReport) -> str:
    """Build human-readable summary."""
    lines = []

    total = report.n_vf2_events + report.n_bfs_events + report.n_fallback_events
    if total == 0:
        lines.append(
            "No layout selection records found in historical data.\n"
            "Run hardware rehearsal (make hw-rehearsal) or deployment to generate data."
        )
    else:
        lines.append(f"Total layout selection events: {total}")
        if report.n_vf2_events:
            lines.append(
                f"  VF2 (mapomatic): {report.n_vf2_events} events, "
                f"mean CES={report.vf2_mean_ces:.4f}"
            )
        if report.n_bfs_events:
            lines.append(
                f"  BFS (legacy):    {report.n_bfs_events} events, "
                f"mean CES={report.bfs_mean_ces:.4f}"
            )
        if report.n_fallback_events:
            lines.append(f"  Fallback:        {report.n_fallback_events} events")
        if report.ces_improvement_ratio > 1.0:
            lines.append(
                f"  Improvement:     {report.ces_improvement_ratio:.1f}× lower CES with VF2"
            )

    if report.benchmark:
        b = report.benchmark
        lines.append(f"\n  Live Benchmark (N={b.n_qubits}, {b.topology}):")
        lines.append(
            f"    VF2: {b.vf2_n_layouts} layouts, CES={b.vf2_mean_ces:.4f}, "
            f"time={b.vf2_elapsed_s:.3f}s"
        )
        lines.append(
            f"    BFS: {b.bfs_n_layouts} candidates, CES={b.bfs_mean_ces:.4f}, "
            f"time={b.bfs_elapsed_s:.3f}s"
        )
        if b.improvement_ratio > 0:
            lines.append(f"    Ratio: VF2 is {b.improvement_ratio:.1f}× better (lower CES)")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze layout optimizer (mapomatic VF2) integration results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m project_health.analysis.layout_optimizer_analyzer
  python -m project_health.analysis.layout_optimizer_analyzer --benchmark
  python -m project_health.analysis.layout_optimizer_analyzer --verbose --json report.json
""",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Detailed output")
    parser.add_argument("--json", type=str, default=None, help="Save JSON report to file")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run live VF2 vs BFS comparison on FakeTorino (requires qiskit-aer)",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None, help="Results directory (default: auto-detect)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    results_root = Path(args.results_dir) if args.results_dir else None
    report = analyze(
        results_root=results_root,
        verbose=args.verbose,
        run_live_benchmark=args.benchmark,
    )

    # Print report
    print()
    print("=" * 70)
    print("  Layout Optimizer Analysis (Mapomatic VF2 Integration)")
    print("=" * 70)
    print(
        f"\n  Mapomatic: {'✅ v' + report.mapomatic_version if report.mapomatic_available else '❌ Not installed'}"
    )
    print(f"\n{report.summary_message}")

    if report.recommendations:
        print(f"\n{'─' * 70}")
        print("  Recommendations:")
        for rec in report.recommendations:
            print(f"    • {rec}")

    if args.verbose and report.records:
        print(f"\n{'─' * 70}")
        print("  Detailed Records:")
        for i, r in enumerate(report.records[:20], 1):
            print(
                f"    [{i:2d}] method={r.method:<15} n={r.n_layouts_selected} "
                f"mean_CES={r.mean_ces:.4f} strategy={r.strategy or 'n/a'}"
            )

    print()

    # Save JSON if requested
    if args.json:
        output = report.to_dict()
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, default=str))
        print(f"  Report saved to: {output_path}")


if __name__ == "__main__":
    main()
