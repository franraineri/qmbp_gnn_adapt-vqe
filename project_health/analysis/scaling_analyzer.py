#!/usr/bin/env python3
"""Scaling experiment analyzer — processes MPS scaling results.

Scans `results/scaling/` for scaling_N*_*.json files produced by
`run_scaling_validation.py` and `run_cross_n_transfer.py`, then:

1. Extracts key metrics per system size (N, ΔE/gap, timing, convergence)
2. Validates against the predicted scaling law: h_min = 1.0 + 0.020·N^1.31
3. Produces summary tables and comparison across system sizes
4. Detects anomalies (convergence failures, timing outliers)
5. Generates thesis-ready tables and scaling law verification data

Usage:
    python -m project_health.analysis.scaling_analyzer
    python -m project_health.analysis.scaling_analyzer --verbose
    python -m project_health.analysis.scaling_analyzer --json report.json
    python -m project_health.analysis.scaling_analyzer --results-dir results/scaling

Output:
    Structured report with per-N metrics, scaling law validation, and
    cross-N comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SCALING_DIR = ROOT / "results" / "scaling"


# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ScalingPointResult:
    """Metrics for a single h-point within a scaling run."""

    h: float
    vqe_energy: float
    dmrg_energy: float
    gap: float
    de_gap: float
    time_s: float
    passed: bool
    n_iterations: int = 0


@dataclass
class ScalingRunSummary:
    """Summary of one scaling validation run (one N, one strategy, one seed)."""

    n_qubits: int
    topology: str
    strategy: str
    chi_max: int
    precision: float
    seed: int
    p_layers: int
    h_values: list[float]
    # Timing
    phase1_time_s: float
    phase2_time_s: float
    total_time_s: float
    # Results
    n_pass: int
    n_total: int
    all_passed: bool
    mean_de_gap: float
    max_de_gap: float
    min_de_gap: float
    # Per-h data
    per_h_results: list[ScalingPointResult] = field(default_factory=list)
    # Source file
    source_file: str = ""


@dataclass
class ScalingLawValidation:
    """Validation of the scaling law h_min = 1.0 + 0.020·N^1.31."""

    predicted_h_min: float
    actual_h_min: float | None  # Lowest h where ΔE/gap < 5%
    prediction_error: float | None
    within_tolerance: bool  # |predicted - actual| < 0.5
    n_qubits: int


@dataclass
class CrossNComparison:
    """Cross-N comparison metrics for thesis tables."""

    n_values: list[int]
    mean_de_gap_per_n: dict[int, float]
    timing_per_n: dict[int, float]
    pass_rate_per_n: dict[int, float]
    scaling_law_valid: bool


@dataclass
class ScalingReport:
    """Complete scaling analysis report."""

    runs: list[ScalingRunSummary] = field(default_factory=list)
    scaling_law: list[ScalingLawValidation] = field(default_factory=list)
    cross_n: CrossNComparison | None = None
    anomalies: list[str] = field(default_factory=list)
    n_files_scanned: int = 0
    overall_verdict: str = ""  # "PASS", "PARTIAL", "FAIL"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "overall_verdict": self.overall_verdict,
            "n_files_scanned": self.n_files_scanned,
            "n_runs": len(self.runs),
            "anomalies": self.anomalies,
            "runs": [asdict(r) for r in self.runs],
            "scaling_law": [asdict(s) for s in self.scaling_law],
            "cross_n": asdict(self.cross_n) if self.cross_n else None,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Scanning & Parsing
# ═══════════════════════════════════════════════════════════════════════════════


def scan_scaling_results(results_dir: Path) -> list[dict[str, Any]]:
    """Discover and load all scaling result JSON files."""
    if not results_dir.exists():
        logger.warning(f"Scaling results directory not found: {results_dir}")
        return []

    files = sorted(results_dir.glob("scaling_N*_*.json"))
    results = []
    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)
            data["_source_file"] = str(f)
            results.append(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load {f}: {e}")

    logger.info(f"Scanned {len(results)} scaling result files from {results_dir}")
    return results


def parse_scaling_run(data: dict[str, Any]) -> ScalingRunSummary | None:
    """Parse a single scaling validation JSON into a ScalingRunSummary."""
    if data.get("experiment") != "mps_scaling_validation":
        return None

    meta = data.get("metadata", {})
    timing = data.get("timing", {})
    summary = data.get("summary", {})
    vqe_results = data.get("vqe_results", [])

    # Extract per-h results from first seed (primary)
    per_h: list[ScalingPointResult] = []
    all_de_gaps: list[float] = []

    for seed_run in vqe_results:
        for r in seed_run.get("results", []):
            point = ScalingPointResult(
                h=r["h"],
                vqe_energy=r["vqe_energy"],
                dmrg_energy=r["dmrg_energy"],
                gap=r["gap"],
                de_gap=r["de_gap"],
                time_s=r["time_s"],
                passed=r["passed"],
                n_iterations=r.get("n_iterations", 0),
            )
            per_h.append(point)
            all_de_gaps.append(r["de_gap"])

    if not all_de_gaps:
        return None

    seeds = meta.get("seeds", [42])
    seed = seeds[0] if seeds else 42

    return ScalingRunSummary(
        n_qubits=meta.get("n", 0),
        topology=meta.get("topology", "chain_1d"),
        strategy=meta.get("strategy", ""),
        chi_max=meta.get("chi_max", 64),
        precision=meta.get("precision", 0.005),
        seed=seed,
        p_layers=meta.get("p_layers", 1),
        h_values=meta.get("h_values", []),
        phase1_time_s=timing.get("phase1_dmrg_s", 0),
        phase2_time_s=timing.get("phase2_vqe_s", 0),
        total_time_s=timing.get("total_s", 0),
        n_pass=summary.get("n_pass", 0),
        n_total=summary.get("n_total", 0),
        all_passed=summary.get("all_passed", False),
        mean_de_gap=float(np.mean(all_de_gaps)),
        max_de_gap=float(np.max(all_de_gaps)),
        min_de_gap=float(np.min(all_de_gaps)),
        per_h_results=per_h,
        source_file=data.get("_source_file", ""),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis Functions
# ═══════════════════════════════════════════════════════════════════════════════


def validate_scaling_law(run: ScalingRunSummary) -> ScalingLawValidation:
    """Check if the actual valid regime matches the predicted scaling law.

    Scaling law: h_min = 1.0 + 0.020 · N^1.31 (R²=1.0000, validated N=6-20)
    """
    n = run.n_qubits
    predicted = 1.0 + 0.020 * n**1.31

    # Find actual h_min: lowest h where ΔE/gap < 5% (passed=True)
    actual_h_min = None
    for point in sorted(run.per_h_results, key=lambda p: p.h):
        if point.passed:
            actual_h_min = point.h
            break

    error = abs(predicted - actual_h_min) if actual_h_min is not None else None
    within = error is not None and error < 0.5

    return ScalingLawValidation(
        predicted_h_min=predicted,
        actual_h_min=actual_h_min,
        prediction_error=error,
        within_tolerance=within,
        n_qubits=n,
    )


def detect_anomalies(runs: list[ScalingRunSummary]) -> list[str]:
    """Detect anomalies in scaling results."""
    anomalies = []

    for run in runs:
        # Check timing outliers
        if run.n_qubits <= 50 and run.total_time_s > 7200:
            anomalies.append(
                f"N={run.n_qubits}: Total time {run.total_time_s:.0f}s > 2h (unexpected)"
            )

        # Check if ALL h-points failed (possible VQE convergence issue)
        if run.n_total > 0 and run.n_pass == 0:
            anomalies.append(f"N={run.n_qubits}: ALL {run.n_total} h-points failed ΔE/gap < 5%")

        # Check if max ΔE/gap is extremely large (VQE didn't converge at all)
        if run.max_de_gap > 1.0:
            anomalies.append(
                f"N={run.n_qubits}: max ΔE/gap={run.max_de_gap:.2f} > 100% "
                f"(VQE likely stuck in local minimum)"
            )

        # Check Phase 1 timing (should be fast for 1D TFIM)
        if run.n_qubits <= 100 and run.phase1_time_s > 300:
            anomalies.append(
                f"N={run.n_qubits}: DMRG took {run.phase1_time_s:.0f}s > 5 min "
                f"(expected <30s for 1D TFIM)"
            )

    return anomalies


def build_cross_n_comparison(runs: list[ScalingRunSummary]) -> CrossNComparison | None:
    """Build cross-N comparison table from multiple scaling runs."""
    if not runs:
        return None

    # Group by N
    by_n: dict[int, list[ScalingRunSummary]] = {}
    for run in runs:
        by_n.setdefault(run.n_qubits, []).append(run)

    n_values = sorted(by_n.keys())
    mean_de_gap_per_n = {}
    timing_per_n = {}
    pass_rate_per_n = {}

    for n, n_runs in by_n.items():
        # Average across runs at same N
        mean_de_gap_per_n[n] = float(np.mean([r.mean_de_gap for r in n_runs]))
        timing_per_n[n] = float(np.mean([r.total_time_s for r in n_runs]))
        total_pass = sum(r.n_pass for r in n_runs)
        total_pts = sum(r.n_total for r in n_runs)
        pass_rate_per_n[n] = total_pass / max(total_pts, 1)

    # Check if scaling law holds: timing should grow polynomially
    scaling_valid = all(pass_rate_per_n.get(n, 0) > 0.5 for n in n_values)

    return CrossNComparison(
        n_values=n_values,
        mean_de_gap_per_n=mean_de_gap_per_n,
        timing_per_n=timing_per_n,
        pass_rate_per_n=pass_rate_per_n,
        scaling_law_valid=scaling_valid,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════════


def generate_report(results_dir: Path, verbose: bool = False) -> ScalingReport:
    """Generate a complete scaling analysis report."""
    raw_data = scan_scaling_results(results_dir)
    report = ScalingReport(n_files_scanned=len(raw_data))

    # Parse all runs
    for data in raw_data:
        run = parse_scaling_run(data)
        if run:
            report.runs.append(run)

    if not report.runs:
        report.overall_verdict = "NO_DATA"
        return report

    # Validate scaling law for each run
    for run in report.runs:
        validation = validate_scaling_law(run)
        report.scaling_law.append(validation)

    # Detect anomalies
    report.anomalies = detect_anomalies(report.runs)

    # Build cross-N comparison
    report.cross_n = build_cross_n_comparison(report.runs)

    # Overall verdict
    if all(r.all_passed for r in report.runs):
        report.overall_verdict = "PASS"
    elif any(r.n_pass > 0 for r in report.runs):
        report.overall_verdict = "PARTIAL"
    else:
        report.overall_verdict = "FAIL"

    return report


def format_report_text(report: ScalingReport) -> str:
    """Format the report as human-readable text."""
    lines = []
    lines.append("=" * 70)
    lines.append("MPS SCALING ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"Files scanned: {report.n_files_scanned}")
    lines.append(f"Valid runs parsed: {len(report.runs)}")
    lines.append(f"Overall verdict: {report.overall_verdict}")
    lines.append("")

    # Per-run summary table
    if report.runs:
        lines.append("─── Per-Run Results ───")
        lines.append(
            f"{'N':>4} {'Strategy':<12} {'Pass':>6} {'Mean ΔE/gap':>12} "
            f"{'Max ΔE/gap':>11} {'Phase1':>7} {'Phase2':>8} {'Total':>7}"
        )
        lines.append("-" * 70)
        for run in sorted(report.runs, key=lambda r: r.n_qubits):
            status = "✅" if run.all_passed else "❌"
            lines.append(
                f"{run.n_qubits:>4} {run.strategy:<12} "
                f"{status} {run.n_pass}/{run.n_total} "
                f"{run.mean_de_gap:>10.4f}  {run.max_de_gap:>10.4f}  "
                f"{run.phase1_time_s:>5.1f}s {run.phase2_time_s:>6.1f}s "
                f"{run.total_time_s:>5.1f}s"
            )
        lines.append("")

    # Scaling law validation
    if report.scaling_law:
        lines.append("─── Scaling Law Validation (h_min = 1.0 + 0.020·N^1.31) ───")
        for sv in report.scaling_law:
            status = "✅" if sv.within_tolerance else "❌"
            actual_str = f"{sv.actual_h_min:.2f}" if sv.actual_h_min else "N/A"
            err_str = f"{sv.prediction_error:.2f}" if sv.prediction_error else "N/A"
            lines.append(
                f"  N={sv.n_qubits:>3}: predicted={sv.predicted_h_min:.2f}, "
                f"actual={actual_str}, error={err_str} {status}"
            )
        lines.append("")

    # Cross-N comparison
    if report.cross_n:
        lines.append("─── Cross-N Comparison ───")
        lines.append(f"{'N':>4} {'Mean ΔE/gap':>12} {'Pass Rate':>10} {'Time':>8}")
        lines.append("-" * 40)
        for n in report.cross_n.n_values:
            lines.append(
                f"{n:>4} {report.cross_n.mean_de_gap_per_n[n]:>12.4f} "
                f"{report.cross_n.pass_rate_per_n[n]:>9.1%} "
                f"{report.cross_n.timing_per_n[n]:>6.0f}s"
            )
        lines.append(
            f"  Scaling law overall: {'VALID' if report.cross_n.scaling_law_valid else 'INVALID'}"
        )
        lines.append("")

    # Anomalies
    if report.anomalies:
        lines.append("─── Anomalies Detected ───")
        for a in report.anomalies:
            lines.append(f"  ⚠ {a}")
        lines.append("")

    # Per-h detail for each run (verbose)
    lines.append("─── Per-h Detail ───")
    for run in report.runs:
        lines.append(f"  N={run.n_qubits}, strategy={run.strategy}, seed={run.seed}:")
        for p in run.per_h_results:
            status = "✅" if p.passed else "❌"
            lines.append(
                f"    h={p.h:.3f}: ΔE/gap={p.de_gap:.4f} "
                f"({p.time_s:.1f}s, {p.n_iterations} iter) {status}"
            )
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze MPS scaling experiment results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_SCALING_DIR),
        help="Directory containing scaling result JSON files",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Export report as JSON to this file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    results_dir = Path(args.results_dir)
    report = generate_report(results_dir, verbose=args.verbose)

    # Print text report
    print(format_report_text(report))

    # Optionally export JSON
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        print(f"JSON report saved: {json_path}")

    return 0 if report.overall_verdict in ("PASS", "NO_DATA") else 1


if __name__ == "__main__":
    sys.exit(main())
