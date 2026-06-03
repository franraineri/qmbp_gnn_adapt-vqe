"""Output formatters for the health report.

Supports text (console) and JSON output modes.
"""

from __future__ import annotations

import json

from project_health.models import (
    HealthReport,
    Priority,
)


def format_text(report: HealthReport, *, compact: bool = False) -> str:
    """Format the health report as human-readable text.

    Parameters
    ----------
    report : HealthReport
        The computed health report.
    compact : bool
        If True, only show the summary section (no per-experiment detail).
    """
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("╔══════════════════════════════════════════════════════════════════╗")
    lines.append("║              PROJECT HEALTH REPORT                              ║")
    lines.append("╚══════════════════════════════════════════════════════════════════╝")
    lines.append(f"  Generated: {report.timestamp}")
    lines.append(f"  Results dir: {report.results_dir}")
    lines.append("")

    # ── Overview ──────────────────────────────────────────────────────────
    lines.append("─── SCAN OVERVIEW ─────────────────────────────────────────────────")
    lines.append(f"  Noiseless pipeline runs: {report.n_noiseless}")
    lines.append(f"  Noisy/ZNE runs:          {report.n_noisy}")
    lines.append(f"  Experiments:             {report.n_experiments}")
    lines.append("")

    # ── Experiment Verdicts ────────────────────────────────────────────────
    lines.append("─── EXPERIMENT VERDICTS ────────────────────────────────────────────")
    lines.append(
        f"  Confirmed: {report.n_confirmed} ✅  "
        f"Rejected: {report.n_rejected} ⚠️   "
        f"Failed: {report.n_failed} ❌"
    )
    lines.append("")

    if not compact and report.experiments:
        lines.append(f"  {'ID':<12} {'Verdict':<12} {'Pass%':<8} {'Criteria'}")
        lines.append(f"  {'─' * 65}")
        for exp in report.experiments:
            emoji = _verdict_emoji(exp.verdict)
            pr = f"{exp.pass_rate * 100:.0f}%" if exp.pass_rate is not None else "—"
            lines.append(
                f"  {exp.experiment_id:<12} {emoji} {exp.verdict:<9} {pr:<8} {exp.criteria}"
            )
            # Show hypotheses if available
            if exp.hypotheses:
                confirmed = sum(1 for v in exp.hypotheses.values() if v)
                lines.append(
                    f"  {'':12} └─ Hypotheses: {confirmed}/{len(exp.hypotheses)} confirmed"
                )
        lines.append("")

    # ── Noiseless Quality ─────────────────────────────────────────────────
    lines.append("─── NOISELESS QUALITY ─────────────────────────────────────────────")
    lines.append(f"  Pass rate (ΔE/gap < 5%): {report.noiseless_pass_rate:.0%}")
    if report.noiseless_median_de is not None:
        lines.append(f"  Median ΔE/gap:           {report.noiseless_median_de:.4f}")

    # Per-topology breakdown
    if report.noiseless_by_topology:
        lines.append("")
        lines.append(
            f"  {'Topology':<12} {'Runs':<6} {'Pass%':<7} {'Median':<9} {'Best':<9} {'Worst'}"
        )
        lines.append(f"  {'─' * 55}")
        for topo, stats in report.noiseless_by_topology.items():
            lines.append(
                f"  {topo:<12} {stats['n_runs']:<6} "
                f"{stats['pass_rate']:.0%}{'':3} "
                f"{stats['median_de']:.4f}{'':3} "
                f"{stats['best']:.4f}{'':3} "
                f"{stats['worst']:.4f}"
            )
    lines.append("")

    # ── Noisy/ZNE Quality ─────────────────────────────────────────────────
    if report.n_noisy > 0:
        lines.append("─── NOISY/ZNE QUALITY ─────────────────────────────────────────────")
        lines.append(f"  Success rate:  {report.noisy_success_rate:.0%}")
        lines.append(f"  Mean R²:       {report.noisy_mean_r2:.4f}")
        lines.append(f"  Mean gain:     {report.noisy_mean_gain:+.1f}%")
        lines.append("")

    # ── Coverage Gaps ─────────────────────────────────────────────────────
    if report.gaps:
        lines.append("─── COVERAGE GAPS ─────────────────────────────────────────────────")
        lines.append(f"  {len(report.gaps)} gap(s) detected:")
        lines.append("")

        # Group by priority
        for priority in Priority:
            prio_gaps = [g for g in report.gaps if g.priority == priority]
            if not prio_gaps:
                continue
            lines.append(f"  [{priority.name}]")
            for g in prio_gaps:
                icon = _priority_icon(g.priority)
                lines.append(f"    {icon} {g.detail}")
                if g.recommendation and not compact:
                    lines.append(f"       → {g.recommendation}")
            lines.append("")

    # ── New Since Last Run ────────────────────────────────────────────────
    if report.new_results or report.removed_results:
        lines.append("─── DELTA SINCE LAST RUN ──────────────────────────────────────────")
        if report.new_results:
            lines.append(f"  {report.n_new} new result file(s):")
            for f in report.new_results[:10]:
                lines.append(f"    + {f}")
            if report.n_new > 10:
                lines.append(f"    ... and {report.n_new - 10} more")
        if report.removed_results:
            lines.append(f"  {report.n_removed} removed result file(s):")
            for f in report.removed_results[:5]:
                lines.append(f"    − {f}")
            if report.n_removed > 5:
                lines.append(f"    ... and {report.n_removed - 5} more")
        lines.append("")

    # ── Actions ───────────────────────────────────────────────────────────
    if report.actions:
        lines.append("─── ACTIONABLE ITEMS ──────────────────────────────────────────────")
        for i, action in enumerate(report.actions, 1):
            icon = _priority_icon(action.priority)
            lines.append(f"  {i}. {icon} [{action.priority.name}] {action.title}")
            if not compact:
                lines.append(f"     {action.detail}")
        lines.append("")

    # ── Footer ────────────────────────────────────────────────────────────
    lines.append("═" * 68)
    total_useful = report.n_confirmed + report.n_rejected
    total_exp = report.n_confirmed + report.n_rejected + report.n_failed
    rate = total_useful / max(total_exp, 1)
    lines.append(f"  Useful-outcome rate: {rate:.0%} ({total_useful}/{total_exp} experiments)")
    lines.append(
        f"  Total result files tracked: {report.n_noiseless + report.n_noisy + report.n_experiments}"
    )
    if report.elapsed_s > 0:
        lines.append(f"  Health check completed in {report.elapsed_s:.1f}s")
    lines.append("")

    return "\n".join(lines)


def format_json(report: HealthReport, *, indent: int = 2) -> str:
    """Format the health report as JSON string."""
    return json.dumps(report.to_dict(), indent=indent, default=str)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _verdict_emoji(verdict: str) -> str:
    """Return emoji for experiment verdict."""
    return {"confirmed": "✅", "rejected": "⚠️", "failed": "❌"}.get(verdict, "?")


def _priority_icon(priority: Priority) -> str:
    """Return icon for priority level."""
    return {
        Priority.CRITICAL: "🔴",
        Priority.HIGH: "🟠",
        Priority.MEDIUM: "🟡",
        Priority.LOW: "⚪",
    }.get(priority, "·")
