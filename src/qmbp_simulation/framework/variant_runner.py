"""Shared variant runner infrastructure for thesis pipeline validation.

Provides the common execution engine used by all topology-specific variant
scripts (chain_1d, ladder, triangular, kagome). Each script defines its
variants and delegates execution to this module.

Usage:
    from qmbp_simulation.framework.variant_runner import (
        PipelineVariant, RunResult, VariantRunner,
    )

    variants = build_my_variants(n_qubits)
    runner = VariantRunner(
        topology="triangular",
        n_qubits=6,
        variants=variants,
    )
    runner.run(dry_run=False, start_from=0)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class PipelineVariant:
    """A single pipeline configuration to test."""

    id: str
    description: str
    category: str  # "noiseless", "noisy", or "extended"
    command: list[str]
    hypothesis: str
    expected_outcome: str
    output_dir: str


@dataclass
class RunResult:
    """Result of a single variant execution."""

    variant_id: str
    success: bool
    elapsed_s: float
    return_code: int
    error_msg: str = ""
    delta_e_over_gap: float | None = None
    phase3_failed: bool = False
    noisy_summary: dict | None = field(default=None)
    max_fidelity: float | None = None
    scientific_conclusion: str | None = None

    @property
    def verdict(self) -> str:
        """Human-readable verdict."""
        if not self.success:
            if self.phase3_failed:
                return "SKIP-P3"
            return "ERROR"
        if self.noisy_summary is not None:
            if self.noisy_summary.get("success"):
                return "ZNE-PASS"
            return "ZNE-FAIL"
        if self.delta_e_over_gap is None:
            # No Phase 4 data — check if we have scientific classification
            if self.scientific_conclusion == "negative_fundamental":
                return "NEG-FUND"
            if self.scientific_conclusion == "negative_expressibility":
                return "NEG-EXPR"
            return "OK"
        from qmbp_simulation.analysis.constants import DE_GAP_THRESHOLD, MAX_ABS_ERROR

        if self.delta_e_over_gap < DE_GAP_THRESHOLD:
            return "PASS"
        if self.delta_e_over_gap < MAX_ABS_ERROR:
            return "MARGINAL"
        return "FAIL"


def extract_metrics_from_output(output_dir: str) -> dict:
    """Extract key metrics from pipeline output.

    Handles both noiseless (pipeline_run_*.json) and noisy (noisy_3mode_*.json)
    output formats. For model-agnostic pipelines, also extracts phase2_summary
    and scientific_conclusion when Phase 4 is not available.

    Parameters
    ----------
    output_dir : str
        Path to the variant's output directory.

    Returns
    -------
    dict
        Extracted metrics. May contain:
        - "delta_e_over_gap": float (worst across test points)
        - "n_test_points": int
        - "phase3_failed": bool
        - "noisy_summary": dict with ZNE metrics
        - "max_fidelity": float (from phase2_summary, when Phase 4 absent)
        - "scientific_conclusion": str (classification from model-agnostic pipeline)
    """
    out_path = Path(output_dir)
    if not out_path.exists():
        return {}

    # Try noiseless pipeline output first
    json_files = sorted(out_path.glob("pipeline_run_*.json"), reverse=True)
    if json_files:
        try:
            with open(json_files[0]) as f:
                data = json.load(f)
            results = data.get("phase4_results", [])
            if results:
                de_gaps = [
                    r["delta_e_over_gap"] for r in results if r.get("delta_e_over_gap") is not None
                ]
                if de_gaps:
                    return {"delta_e_over_gap": max(de_gaps), "n_test_points": len(de_gaps)}
            # Phase 3 failed (no phase4 results but pipeline completed)
            if not results and data.get("elapsed_s", 0) > 0:
                result = {"phase3_failed": True}
                # Extract phase2_summary for model-agnostic pipelines (Heisenberg)
                p2 = data.get("phase2_summary")
                if p2:
                    result["max_fidelity"] = p2.get("max_fidelity")
                sci = data.get("scientific_conclusion")
                if sci:
                    result["scientific_conclusion"] = sci.get("classification")
                return result
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    # Try noisy pipeline output (noisy_3mode_*.json)
    noisy_files = sorted(out_path.glob("noisy_3mode_*.json"), reverse=True)
    if noisy_files:
        try:
            with open(noisy_files[0]) as f:
                data = json.load(f)
            summary = data.get("summary", {})
            if summary:
                return {
                    "noisy_summary": {
                        "success": summary.get("success_criteria_met", False),
                        "mean_r2": summary.get("mean_r2"),
                        "mean_gain_pct": summary.get("mean_gain_pct"),
                        "n_mitigated_wins": summary.get("n_mitigated_wins"),
                        "n_total": summary.get("n_total"),
                    }
                }
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    return {}


def run_variant(
    variant: PipelineVariant,
    *,
    dry_run: bool = False,
    timeout: int = 1200,
) -> RunResult:
    """Execute a single pipeline variant.

    Parameters
    ----------
    variant : PipelineVariant
        The variant configuration to run.
    dry_run : bool
        If True, print command without executing.
    timeout : int
        Maximum execution time in seconds (default: 1200 = 30 min).

    Returns
    -------
    RunResult
        Execution result with metrics extracted from output.
    """
    if dry_run:
        print(f"  [DRY RUN] {' '.join(variant.command)}")
        return RunResult(variant_id=variant.id, success=True, elapsed_s=0.0, return_code=0)

    Path(variant.output_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        result = subprocess.run(
            variant.command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            stderr_lines = result.stderr.strip().split("\n")[-10:]
            error_msg = "\n".join(stderr_lines)
            phase3_failed = "Phase 3 FAILED" in error_msg or "fidelity" in error_msg.lower()
            return RunResult(
                variant_id=variant.id,
                success=False,
                elapsed_s=elapsed,
                return_code=result.returncode,
                error_msg=error_msg,
                phase3_failed=phase3_failed,
            )

        # Extract metrics from saved JSON output
        metrics = extract_metrics_from_output(variant.output_dir)
        return RunResult(
            variant_id=variant.id,
            success=True,
            elapsed_s=elapsed,
            return_code=0,
            delta_e_over_gap=metrics.get("delta_e_over_gap"),
            phase3_failed=metrics.get("phase3_failed", False),
            noisy_summary=metrics.get("noisy_summary"),
            max_fidelity=metrics.get("max_fidelity"),
            scientific_conclusion=metrics.get("scientific_conclusion"),
        )

    except subprocess.TimeoutExpired:
        return RunResult(
            variant_id=variant.id,
            success=False,
            elapsed_s=time.time() - t0,
            return_code=-1,
            error_msg=f"TIMEOUT (>{timeout}s)",
        )
    except Exception as e:
        return RunResult(
            variant_id=variant.id,
            success=False,
            elapsed_s=time.time() - t0,
            return_code=-2,
            error_msg=str(e),
        )


class VariantRunner:
    """Orchestrates execution of multiple pipeline variants.

    Handles listing, single-variant execution, full batch runs,
    progress reporting, and execution log saving.

    Parameters
    ----------
    topology : str
        Topology name (for display and log metadata).
    n_qubits : int
        System size (for display and log metadata).
    noiseless : list[PipelineVariant]
        Noiseless variant definitions.
    noisy : list[PipelineVariant]
        Noisy variant definitions.
    extended : list[PipelineVariant]
        Extended variant definitions.
    timeout : int
        Per-variant timeout in seconds (default: 1200).
    """

    def __init__(
        self,
        *,
        topology: str,
        n_qubits: int,
        noiseless: list[PipelineVariant],
        noisy: list[PipelineVariant],
        extended: list[PipelineVariant],
        timeout: int = 1200,
    ) -> None:
        self.topology = topology
        self.n_qubits = n_qubits
        self.noiseless = noiseless
        self.noisy = noisy
        self.extended = extended
        self.timeout = timeout

    @property
    def output_base(self) -> str:
        return f"results/thesis/variants_N{self.n_qubits}_{self.topology}"

    def get_variants(
        self,
        *,
        noiseless_only: bool = False,
        noisy_only: bool = False,
        extended_only: bool = False,
    ) -> list[PipelineVariant]:
        """Get filtered variant list based on category flags."""
        if noiseless_only:
            return self.noiseless
        if noisy_only:
            return self.noisy
        if extended_only:
            return self.extended
        return self.noiseless + self.noisy + self.extended

    def print_variant_table(self, variants: list[PipelineVariant]) -> None:
        """Print a formatted table of all variants."""
        print(f"\n{'#':<4} {'ID':<22} {'Category':<10} {'Description'}")
        print("-" * 90)
        for i, v in enumerate(variants):
            print(f"{i:<4} {v.id:<22} {v.category:<10} {v.description}")
        print()

    def list_variants(self, variants: list[PipelineVariant]) -> None:
        """Print summary and table of all variants."""
        print("=" * 90)
        print(f"  PIPELINE VARIANTS — N={self.n_qubits}, topology={self.topology}")
        print("=" * 90)
        print(f"\n  Noiseless: {len(self.noiseless)} variants")
        print(f"  Noisy:     {len(self.noisy)} variants")
        print(f"  Extended:  {len(self.extended)} variants")
        print(f"  Total:     {len(variants)} variants")
        self.print_variant_table(variants)

    def run_single(
        self,
        variants: list[PipelineVariant],
        index: int,
        *,
        dry_run: bool = False,
    ) -> int:
        """Run a single variant by index. Returns exit code."""
        if index >= len(variants):
            print(f"ERROR: Index {index} out of range (0-{len(variants) - 1})")
            return 1
        v = variants[index]
        print(f"\nRunning: {v.id}")
        print(f"  {v.description}")
        print(f"  Hypothesis: {v.hypothesis}")
        print(f"  Expected: {v.expected_outcome}")
        print()
        result = run_variant(v, dry_run=dry_run, timeout=self.timeout)
        status = "✅ PASS" if result.success else "❌ FAIL"
        metric_str = self._format_metric(result)
        print(f"\n  {status} ({result.elapsed_s:.1f}s){metric_str}  [{result.verdict}]")
        if not result.success and not result.phase3_failed:
            print(f"  Error: {result.error_msg}")
        elif result.phase3_failed:
            print("  Phase 3 failed: fidelity too low for this h-range/topology")
        return 0 if result.success else 1

    def run_all(
        self,
        variants: list[PipelineVariant],
        *,
        dry_run: bool = False,
        start_from: int = 0,
        prioritize: bool = False,
    ) -> int:
        """Run all variants. Returns exit code (0 if all pass).

        Parameters
        ----------
        variants : list[PipelineVariant]
            Variant definitions to execute.
        dry_run : bool
            If True, print commands without executing.
        start_from : int
            Skip variants before this index.
        prioritize : bool
            If True, reorder variants by predicted pass probability
            (highest first) using QualityPredictor. Enables fail-fast
            scheduling — likely-to-pass configs run first.
        """
        # ── Optional: reorder by predicted pass probability ──────────────
        if prioritize and not dry_run:
            variants = self._prioritize_variants(variants)

        print("=" * 90)
        print(f"  EXHAUSTIVE VARIANT RUNNER — N={self.n_qubits}, topology={self.topology}")
        print("=" * 90)
        print(f"\n  Total variants: {len(variants)}")
        print(f"  Starting from:  #{start_from}")
        print(f"  Mode:           {'DRY RUN' if dry_run else 'EXECUTE'}")
        if prioritize:
            print("  Priority:       BY PREDICTED PASS RATE (highest first)")
        print(f"  Output base:    {self.output_base}/")
        print()

        if not dry_run:
            print(f"  ⚠️  Timeout per variant: {self.timeout}s")
            print()

        t_total = time.time()
        results: list[RunResult] = []

        for i, variant in enumerate(variants):
            if i < start_from:
                continue

            print(f"\n{'─' * 65}")
            print(f"  [{i + 1}/{len(variants)}] {variant.id}: {variant.description}")
            print(f"  Hypothesis: {variant.hypothesis}")
            print(f"{'─' * 65}")

            result = run_variant(variant, dry_run=dry_run, timeout=self.timeout)
            results.append(result)

            status = "✅" if result.success else "❌"
            metric_str = self._format_metric(result)
            print(f"  {status} {variant.id}: {result.elapsed_s:.1f}s{metric_str}")
            if not result.success and not result.phase3_failed:
                print(f"     Error: {result.error_msg[:200]}")

        # Final Summary
        total_elapsed = time.time() - t_total
        self._print_summary(results, variants, start_from, total_elapsed)

        # Save execution log
        if not dry_run:
            self._save_log(results, variants, start_from, total_elapsed)

        n_fail = sum(1 for r in results if not r.success)
        return 0 if n_fail == 0 else 1

    def _format_metric(self, result: RunResult) -> str:
        """Format metric string for display (without verdict — caller adds it)."""
        if result.delta_e_over_gap is not None:
            return f"  ΔE/gap={result.delta_e_over_gap:.4f}"
        if result.noisy_summary is not None:
            ns = result.noisy_summary
            r2 = ns.get("mean_r2")
            gain = ns.get("mean_gain_pct")
            r2_str = f"R²={r2:.3f}" if r2 is not None else "R²=—"
            gain_str = f" gain={gain:.1f}%" if gain is not None else ""
            return f"  {r2_str}{gain_str}"
        if result.max_fidelity is not None:
            return f"  max_fid={result.max_fidelity:.4f}"
        if result.phase3_failed:
            return "  [fidelity too low]"
        return ""

    def _print_summary(
        self,
        results: list[RunResult],
        variants: list[PipelineVariant],
        start_from: int,
        total_elapsed: float,
    ) -> None:
        """Print final summary table."""
        n_pass = sum(1 for r in results if r.success)
        n_fail = sum(1 for r in results if not r.success)

        print("\n" + "=" * 90)
        print("  FINAL SUMMARY")
        print("=" * 90)
        print(f"\n  Topology:       {self.topology}")
        print(f"  N:              {self.n_qubits}")
        print(f"  Total variants: {len(results)}")
        print(f"  Passed:         {n_pass} ✅")
        print(f"  Failed:         {n_fail} ❌")
        print(f"  Total time:     {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
        print()

        # Per-variant summary table
        print(f"  {'#':<4} {'ID':<22} {'Verdict':<10} {'Time':<8} {'Metric':<16} {'Description'}")
        print(f"  {'-' * 88}")
        for i, r in enumerate(results):
            variant_idx = start_from + i
            desc = variants[variant_idx].description[:28] if variant_idx < len(variants) else ""
            time_str = f"{r.elapsed_s:.1f}s"
            if r.delta_e_over_gap is not None:
                metric_str = f"{r.delta_e_over_gap:.4f}"
            elif r.noisy_summary is not None:
                r2 = r.noisy_summary.get("mean_r2")
                metric_str = f"R²={r2:.3f}" if r2 is not None else "—"
            elif r.max_fidelity is not None:
                metric_str = f"fid={r.max_fidelity:.4f}"
            else:
                metric_str = "—"
            print(
                f"  {variant_idx:<4} {r.variant_id:<22} {r.verdict:<10} "
                f"{time_str:<8} {metric_str:<16} {desc}"
            )

        if n_fail > 0:
            print("\n  FAILURES:")
            for r in results:
                if not r.success:
                    first_line = r.error_msg.split("\n")[-1][:80]
                    print(f"    ❌ {r.variant_id}: {first_line}")
            print()

    def _prioritize_variants(self, variants: list[PipelineVariant]) -> list[PipelineVariant]:
        """Reorder variants by predicted pass probability (highest first).

        Uses QualityPredictor to estimate which configs will pass, then
        sorts descending. This enables fail-fast scheduling — the most
        likely-to-succeed variants run first, giving earlier feedback.

        Falls back to original order if QualityPredictor is unavailable.
        """
        try:
            from qmbp_simulation.analysis.quality_predictor import QualityPredictor

            predictor = QualityPredictor()
            scored: list[tuple[float, PipelineVariant]] = []
            for v in variants:
                report = predictor.predict(
                    model=getattr(v, "model", "tfim") or "tfim",
                    topology=self.topology,
                    n_qubits=self.n_qubits,
                    p_layers=getattr(v, "p_layers", 2) or 2,
                )
                scored.append((report.pass_probability, v))
            scored.sort(key=lambda x: x[0], reverse=True)
            reordered = [v for _, v in scored]
            print(
                f"  📊 Prioritized by quality prediction "
                f"(top: {scored[0][0]:.0%}, bottom: {scored[-1][0]:.0%})"
            )
            return reordered
        except (ImportError, Exception) as e:
            print(f"  ⚠️ Could not prioritize variants: {e}")
            return variants

    def _save_log(
        self,
        results: list[RunResult],
        variants: list[PipelineVariant],
        start_from: int,
        total_elapsed: float,
    ) -> None:
        """Save execution log as JSON."""
        log_dir = Path(self.output_base)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"execution_log_{timestamp}.json"

        n_pass = sum(1 for r in results if r.success)
        n_fail = sum(1 for r in results if not r.success)

        log_data = {
            "timestamp": timestamp,
            "topology": self.topology,
            "n_qubits": self.n_qubits,
            "total_variants": len(results),
            "passed": n_pass,
            "failed": n_fail,
            "total_elapsed_s": total_elapsed,
            "verdicts": {
                "PASS": sum(1 for r in results if r.verdict == "PASS"),
                "MARGINAL": sum(1 for r in results if r.verdict == "MARGINAL"),
                "FAIL": sum(1 for r in results if r.verdict == "FAIL"),
                "ZNE-PASS": sum(1 for r in results if r.verdict == "ZNE-PASS"),
                "ZNE-FAIL": sum(1 for r in results if r.verdict == "ZNE-FAIL"),
                "SKIP-P3": sum(1 for r in results if r.verdict == "SKIP-P3"),
                "NEG-FUND": sum(1 for r in results if r.verdict == "NEG-FUND"),
                "NEG-EXPR": sum(1 for r in results if r.verdict == "NEG-EXPR"),
                "OK": sum(1 for r in results if r.verdict == "OK"),
                "ERROR": sum(1 for r in results if r.verdict == "ERROR"),
            },
            "results": [
                {
                    "variant_id": r.variant_id,
                    "success": r.success,
                    "verdict": r.verdict,
                    "elapsed_s": round(r.elapsed_s, 2),
                    "delta_e_over_gap": r.delta_e_over_gap,
                    "max_fidelity": r.max_fidelity,
                    "scientific_conclusion": r.scientific_conclusion,
                    "noisy_summary": r.noisy_summary,
                    "return_code": r.return_code,
                    "error_msg": r.error_msg if not r.success else "",
                }
                for r in results
            ],
            "variants": [
                {
                    "id": v.id,
                    "description": v.description,
                    "category": v.category,
                    "hypothesis": v.hypothesis,
                    "expected_outcome": v.expected_outcome,
                }
                for v in variants[start_from:]
            ],
        }

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2, default=str)
        print(f"  Execution log: {log_path}")
        print("=" * 90)


def create_variant_cli() -> argparse.Namespace:
    """Create the standard CLI for variant runner scripts.

    Returns parsed arguments with standard flags:
    --n-qubits, --dry-run, --noiseless-only, --noisy-only,
    --extended-only, --variant, --start-from, --list

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline variant runner for thesis validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-qubits", type=int, default=None, help="Number of qubits (overrides script default)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print without executing")
    parser.add_argument("--noiseless-only", action="store_true", help="Noiseless variants only")
    parser.add_argument("--noisy-only", action="store_true", help="Noisy variants only")
    parser.add_argument("--extended-only", action="store_true", help="Extended variants only")
    parser.add_argument("--variant", type=int, default=None, help="Run specific variant (0-based)")
    parser.add_argument("--start-from", type=int, default=0, help="Start from variant index")
    parser.add_argument("--list", action="store_true", help="List all variants and exit")
    return parser.parse_args()


def run_variant_script(
    *,
    topology: str,
    default_n_qubits: int,
    build_noiseless: Callable[[int], list[PipelineVariant]],
    build_noisy: Callable[[int], list[PipelineVariant]],
    build_extended: Callable[[int], list[PipelineVariant]],
    timeout: int = 1200,
) -> None:
    """Main entry point for variant runner scripts.

    Handles CLI parsing, variant building, and execution delegation.
    Each topology script calls this with its variant builders.

    Parameters
    ----------
    topology : str
        Topology name (chain_1d, ladder, triangular, kagome).
    default_n_qubits : int
        Default system size if --n-qubits not specified.
    build_noiseless : callable
        Function(n_qubits) -> list[PipelineVariant].
    build_noisy : callable
        Function(n_qubits) -> list[PipelineVariant].
    build_extended : callable
        Function(n_qubits) -> list[PipelineVariant].
    timeout : int
        Per-variant timeout in seconds.
    """
    args = create_variant_cli()
    n_qubits = args.n_qubits if args.n_qubits is not None else default_n_qubits

    runner = VariantRunner(
        topology=topology,
        n_qubits=n_qubits,
        noiseless=build_noiseless(n_qubits),
        noisy=build_noisy(n_qubits),
        extended=build_extended(n_qubits),
        timeout=timeout,
    )

    all_variants = runner.get_variants(
        noiseless_only=args.noiseless_only,
        noisy_only=args.noisy_only,
        extended_only=args.extended_only,
    )

    if args.list:
        runner.list_variants(all_variants)
        return

    if args.variant is not None:
        exit_code = runner.run_single(all_variants, args.variant, dry_run=args.dry_run)
        sys.exit(exit_code)

    exit_code = runner.run_all(all_variants, dry_run=args.dry_run, start_from=args.start_from)
    sys.exit(exit_code)
