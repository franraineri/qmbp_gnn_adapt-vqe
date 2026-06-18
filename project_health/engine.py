"""Core health check engine — orchestrates scan, analysis, and reporting.

This is the main logic module. It:
1. Runs ResultScanner to collect all parsed results.
2. Aggregates experiment verdicts.
3. Computes coverage gaps.
4. Computes VQE/MPNN quality diagnostics.
5. Computes timing and distribution analytics.
6. Detects new results since last run.
7. Derives actionable items.
8. Produces a HealthReport.

No I/O (prints, file writes) happens here — that's the reporter's job.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from project_health.coverage import (
    compute_distribution,
    compute_energy_decomposition,
    compute_mpnn_quality,
    compute_noiseless_by_topology,
    compute_noiseless_stats,
    compute_noisy_stats,
    compute_timing_stats,
    compute_vqe_quality,
    derive_actions,
    detect_coverage_gaps,
)
from project_health.digest.models import ExperimentResult, NoiselessResult, NoisyResult
from project_health.digest.scanner import ResultScanner
from project_health.models import ExperimentSummary, HealthReport
from project_health.state import (
    DEFAULT_STATE_FILE,
    detect_delta,
    save_current_state,
)

logger = logging.getLogger(__name__)


def run_health_check(
    results_dir: Path = Path("results"),
    state_file: Path = DEFAULT_STATE_FILE,
    *,
    save_state: bool = True,
) -> HealthReport:
    """Execute the full health check pipeline.

    Parameters
    ----------
    results_dir : Path
        Root results directory (contains experiments/ and thesis/).
    state_file : Path
        Where to persist/load the run state for delta detection.
    save_state : bool
        If True, persist the current file set for next-run comparison.

    Returns
    -------
    HealthReport
        Complete health report ready for formatting.
    """
    report = HealthReport(
        timestamp=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        results_dir=str(results_dir),
        state_file=str(state_file),
    )

    t_start = time.time()

    # ─── Step 1: Scan ────────────────────────────────────────────────────
    logger.info("Scanning results in %s...", results_dir)
    scanner = ResultScanner(results_root=results_dir)
    noiseless, noisy, experiments = scanner.scan_all(exclude_tests=True)

    report.n_noiseless = len(noiseless)
    report.n_noisy = len(noisy)
    report.n_experiments = len(experiments)
    logger.info(
        "  Scanned: %d noiseless, %d noisy, %d experiments",
        len(noiseless),
        len(noisy),
        len(experiments),
    )

    # ─── Step 2: Experiment verdicts ─────────────────────────────────────
    report.experiments = _aggregate_experiments(experiments)
    report.n_confirmed = sum(1 for e in report.experiments if e.verdict == "confirmed")
    report.n_rejected = sum(1 for e in report.experiments if e.verdict == "rejected")
    report.n_failed = sum(1 for e in report.experiments if e.verdict == "failed")

    # ─── Step 3: Noiseless/Noisy quality stats ───────────────────────────
    report.noiseless_pass_rate, report.noiseless_median_de = compute_noiseless_stats(noiseless)
    report.noiseless_by_topology = compute_noiseless_by_topology(noiseless)
    report.noisy_success_rate, report.noisy_mean_r2, report.noisy_mean_gain = compute_noisy_stats(
        noisy
    )

    # ─── Step 4: VQE & MPNN quality diagnostics ──────────────────────────
    report.vqe_quality = compute_vqe_quality(noiseless)
    report.mpnn_quality = compute_mpnn_quality(noiseless)

    # ─── Step 5: Timing & Distribution ───────────────────────────────────
    report.timing = compute_timing_stats(noiseless, noisy, experiments)
    report.distribution = compute_distribution(noiseless, noisy)
    report.energy_decomposition = compute_energy_decomposition(noiseless)

    # ─── Step 6: Coverage gaps ───────────────────────────────────────────
    report.gaps = detect_coverage_gaps(noiseless, noisy, experiments)
    logger.info("  Coverage gaps detected: %d", len(report.gaps))

    # ─── Step 6b: AQC-Tensor compression status ─────────────────────────
    try:
        from project_health.analysis.aqc_tensor_analyzer import get_aqc_health_summary

        report.aqc_status = get_aqc_health_summary()
        logger.info("  AQC-Tensor status: %s", report.aqc_status.get("status", "unknown"))
    except Exception as e:
        logger.debug("  AQC-Tensor status unavailable: %s", e)
        report.aqc_status = {"status": "unavailable", "error": str(e)}

    # ─── Step 6c: Mitiq integration status ───────────────────────────────
    try:
        from project_health.analysis.mitiq_analyzer import get_mitiq_health_summary

        report.mitiq_status = get_mitiq_health_summary()
        logger.info("  Mitiq status: %s", report.mitiq_status.get("status", "unknown"))
    except Exception as e:
        logger.debug("  Mitiq status unavailable: %s", e)
        report.mitiq_status = {"status": "unavailable", "error": str(e)}

    # ─── Step 7: Delta since last run ────────────────────────────────────
    current_files = _collect_source_files(noiseless, noisy, experiments)
    new_results, removed_results = detect_delta(current_files, state_file)
    report.new_results = new_results
    report.n_new = len(new_results)
    report.removed_results = removed_results
    report.n_removed = len(removed_results)

    if save_state:
        save_current_state(
            current_files,
            state_file,
            metadata={
                "timestamp": report.timestamp,
                "n_noiseless": report.n_noiseless,
                "n_noisy": report.n_noisy,
                "n_experiments": report.n_experiments,
            },
        )

    # ─── Step 8: Derive actions ──────────────────────────────────────────
    report.actions = derive_actions(report)

    # ─── Finalize ────────────────────────────────────────────────────────
    report.elapsed_s = round(time.time() - t_start, 2)

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _aggregate_experiments(experiments: list[ExperimentResult]) -> list[ExperimentSummary]:
    """Convert ExperimentResult list to ExperimentSummary list.

    The scanner already computes verdicts — we just extract the relevant
    fields into a leaner structure.
    """
    summaries: list[ExperimentSummary] = []
    for exp in experiments:
        hypotheses = exp.extras.get("hypotheses", {})
        summaries.append(
            ExperimentSummary(
                experiment_id=exp.experiment_id,
                verdict=exp.verdict,
                criteria=exp.criteria,
                pass_rate=exp.pass_rate,
                hypotheses=hypotheses,
                n_hypotheses=len(hypotheses),
                n_confirmed=sum(1 for v in hypotheses.values() if v),
            )
        )
    return summaries


def _collect_source_files(
    noiseless: list[NoiselessResult],
    noisy: list[NoisyResult],
    experiments: list[ExperimentResult],
) -> set[str]:
    """Collect all source_file paths from scan results."""
    files: set[str] = set()
    for r in noiseless:
        files.add(r.source_file)
    for r in noisy:
        files.add(r.source_file)
    for r in experiments:
        files.add(r.source_file)
    return files
