"""Output formatters for the health report.

Supports text (console), JSON, and Markdown output modes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from project_health.models import (
    HealthReport,
    Priority,
)


def generate_timestamped_filename(base: str, ext: str) -> str:
    """Generate a filename with ISO timestamp for unique report outputs.

    Parameters
    ----------
    base : str
        Base name (e.g., "health_report").
    ext : str
        File extension without dot (e.g., "txt", "json", "md").

    Returns
    -------
    str
        Filename like "health_report_20260603_143022.txt".
    """
    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}.{ext}"


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
    total_files = report.n_noiseless + report.n_noisy + report.n_experiments
    lines.append(f"  Total result files:      {total_files}")
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

    # ── VQE Quality (Phase 2 Diagnostics) ───────────────────────────────
    vqe = report.vqe_quality
    if vqe.n_results > 0:
        lines.append("─── VQE CONVERGENCE QUALITY ───────────────────────────────────────")
        lines.append(f"  Results with VQE data:    {vqe.n_results}")
        if vqe.convergence_rate_mean is not None:
            lines.append(f"  Mean convergence rate:   {vqe.convergence_rate_mean:.2%}")
            lines.append(f"  Min convergence rate:    {vqe.convergence_rate_min:.2%}")
        if vqe.theta_smoothness_mean is not None:
            lines.append(f"  Mean θ-smoothness:       {vqe.theta_smoothness_mean:.4f}")
            lines.append(f"  Max θ-smoothness:        {vqe.theta_smoothness_max:.4f}")
        if vqe.n_chain_break_warnings > 0:
            lines.append(f"  ⚠️  Chain break warnings: {vqe.n_chain_break_warnings} (θ > 1.0)")
        lines.append("")

    # ── MPNN Quality (Phase 3 Diagnostics) ────────────────────────────────
    mpnn = report.mpnn_quality
    if mpnn.n_results > 0:
        lines.append("─── MPNN TRAINING QUALITY ─────────────────────────────────────────")
        lines.append(f"  Results with MPNN data:   {mpnn.n_results}")
        if mpnn.gen_gap_mean is not None:
            lines.append(f"  Mean gen. gap:           {mpnn.gen_gap_mean:.6f}")
            lines.append(f"  Median gen. gap:         {mpnn.gen_gap_median:.6f}")
            lines.append(f"  Max gen. gap:            {mpnn.gen_gap_max:.6f}")
        if mpnn.n_overfit_warnings > 0:
            lines.append(f"  ⚠️  Overfit warnings:     {mpnn.n_overfit_warnings} (gen_gap > 0.01)")
        if mpnn.theta_mse_mean is not None:
            lines.append(f"  Mean θ-MSE:              {mpnn.theta_mse_mean:.6f}")
        lines.append("")

    # ── Timing Breakdown ──────────────────────────────────────────────────
    timing = report.timing
    if timing.total_runs > 0:
        lines.append("─── TIMING BREAKDOWN ──────────────────────────────────────────────")
        lines.append(f"  Total compute time:      {timing.total_pipeline_hours:.1f} hours")
        lines.append(f"  Total runs tracked:      {timing.total_runs}")
        lines.append(f"  Mean run time:           {timing.mean_run_s:.1f}s")
        lines.append(f"  Median run time:         {timing.median_run_s:.1f}s")
        lines.append(f"  Max run time:            {timing.max_run_s:.1f}s")
        lines.append("")

    # ── Distribution Analysis ─────────────────────────────────────────────
    dist = report.distribution
    if dist.by_topology or dist.by_model:
        lines.append("─── RESULT DISTRIBUTION ───────────────────────────────────────────")
        if dist.by_model:
            model_parts = [f"{m}: {c}" for m, c in sorted(dist.by_model.items())]
            lines.append(f"  By model:     {', '.join(model_parts)}")
        if dist.by_topology:
            topo_parts = [f"{t}: {c}" for t, c in sorted(dist.by_topology.items())]
            lines.append(f"  By topology:  {', '.join(topo_parts)}")
        if dist.by_n_qubits:
            nq_parts = [f"N={n}: {c}" for n, c in sorted(dist.by_n_qubits.items())]
            lines.append(f"  By N-qubits:  {', '.join(nq_parts)}")
        if dist.by_p_layers:
            pl_parts = [f"p={p}: {c}" for p, c in sorted(dist.by_p_layers.items())]
            lines.append(f"  By p-layers:  {', '.join(pl_parts)}")
        lines.append("")

    # ── Energy Decomposition ──────────────────────────────────────────────
    ed = report.energy_decomposition
    if ed.n_results > 0:
        lines.append("─── ENERGY ERROR DECOMPOSITION ────────────────────────────────────")
        lines.append(f"  Results analyzed:        {ed.n_results}")
        lines.append(f"  Mean circuit error:      {ed.mean_circuit_error:.6f}")
        lines.append(f"  Mean MPNN error:         {ed.mean_mpnn_error:.6f}")
        lines.append(
            f"  Error attribution:       "
            f"circuit {ed.circuit_error_fraction:.0%} / "
            f"MPNN {ed.mpnn_error_fraction:.0%}"
        )
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
        f"  Total result files tracked: "
        f"{report.n_noiseless + report.n_noisy + report.n_experiments}"
    )
    if report.elapsed_s > 0:
        lines.append(f"  Health check completed in {report.elapsed_s:.1f}s")
    lines.append("")

    return "\n".join(lines)


def format_json(report: HealthReport, *, indent: int = 2) -> str:
    """Format the health report as JSON string."""
    return json.dumps(report.to_dict(), indent=indent, default=str)


def format_markdown(report: HealthReport, *, compact: bool = False) -> str:
    """Format the health report as Markdown for persistent documentation.

    Parameters
    ----------
    report : HealthReport
        The computed health report.
    compact : bool
        If True, skip per-experiment detail tables.

    Returns
    -------
    str
        Markdown-formatted report suitable for saving to .md files.
    """
    lines: list[str] = []

    lines.append("# Project Health Report")
    lines.append("")
    lines.append(f"**Generated:** {report.timestamp}")
    lines.append(f"**Results dir:** `{report.results_dir}`")
    lines.append("")

    # Overview table
    lines.append("## Scan Overview")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|--------|------:|")
    lines.append(f"| Noiseless pipeline runs | {report.n_noiseless} |")
    lines.append(f"| Noisy/ZNE runs | {report.n_noisy} |")
    lines.append(f"| Experiments | {report.n_experiments} |")
    total_files = report.n_noiseless + report.n_noisy + report.n_experiments
    lines.append(f"| **Total result files** | **{total_files}** |")
    lines.append("")

    # Verdicts
    lines.append("## Experiment Verdicts")
    lines.append("")
    total_exp = report.n_confirmed + report.n_rejected + report.n_failed
    rate = (report.n_confirmed + report.n_rejected) / max(total_exp, 1)
    lines.append(
        f"- ✅ Confirmed: **{report.n_confirmed}** | "
        f"⚠️ Rejected: **{report.n_rejected}** | "
        f"❌ Failed: **{report.n_failed}**"
    )
    lines.append(f"- Useful-outcome rate: **{rate:.0%}**")
    lines.append("")

    if not compact and report.experiments:
        lines.append("| ID | Verdict | Pass% | Criteria |")
        lines.append("|:---|:--------|------:|:---------|")
        for exp in report.experiments:
            emoji = _verdict_emoji(exp.verdict)
            pr = f"{exp.pass_rate * 100:.0f}%" if exp.pass_rate is not None else "—"
            lines.append(f"| {exp.experiment_id} | {emoji} {exp.verdict} | {pr} | {exp.criteria} |")
        lines.append("")

    # Noiseless quality
    lines.append("## Noiseless Quality")
    lines.append("")
    lines.append(f"- Pass rate (ΔE/gap < 5%): **{report.noiseless_pass_rate:.0%}**")
    if report.noiseless_median_de is not None:
        lines.append(f"- Median ΔE/gap: **{report.noiseless_median_de:.4f}**")
    lines.append("")

    if report.noiseless_by_topology:
        lines.append("| Topology | Runs | Pass% | Median | Best | Worst |")
        lines.append("|:---------|-----:|------:|-------:|-----:|------:|")
        for topo, stats in report.noiseless_by_topology.items():
            lines.append(
                f"| {topo} | {stats['n_runs']} | "
                f"{stats['pass_rate']:.0%} | "
                f"{stats['median_de']:.4f} | "
                f"{stats['best']:.4f} | "
                f"{stats['worst']:.4f} |"
            )
        lines.append("")

    # VQE quality
    vqe = report.vqe_quality
    if vqe.n_results > 0:
        lines.append("## VQE Convergence Quality")
        lines.append("")
        if vqe.convergence_rate_mean is not None:
            lines.append(f"- Mean convergence rate: **{vqe.convergence_rate_mean:.2%}**")
        if vqe.theta_smoothness_mean is not None:
            lines.append(
                f"- Mean θ-smoothness: **{vqe.theta_smoothness_mean:.4f}** "
                f"(max: {vqe.theta_smoothness_max:.4f})"
            )
        if vqe.n_chain_break_warnings > 0:
            lines.append(f"- ⚠️ Chain break warnings: **{vqe.n_chain_break_warnings}** (θ > 1.0)")
        lines.append("")

    # MPNN quality
    mpnn = report.mpnn_quality
    if mpnn.n_results > 0:
        lines.append("## MPNN Training Quality")
        lines.append("")
        if mpnn.gen_gap_mean is not None:
            lines.append(f"- Mean generalization gap: **{mpnn.gen_gap_mean:.6f}**")
            lines.append(f"- Max generalization gap: **{mpnn.gen_gap_max:.6f}**")
        if mpnn.n_overfit_warnings > 0:
            lines.append(f"- ⚠️ Overfit warnings: **{mpnn.n_overfit_warnings}** (gen_gap > 0.01)")
        lines.append("")

    # Distribution
    dist = report.distribution
    if dist.by_topology:
        lines.append("## Result Distribution")
        lines.append("")
        lines.append("| Dimension | Breakdown |")
        lines.append("|:----------|:----------|")
        if dist.by_model:
            parts = [f"{m}: {c}" for m, c in sorted(dist.by_model.items())]
            lines.append(f"| Model | {', '.join(parts)} |")
        if dist.by_topology:
            parts = [f"{t}: {c}" for t, c in sorted(dist.by_topology.items())]
            lines.append(f"| Topology | {', '.join(parts)} |")
        if dist.by_n_qubits:
            parts = [f"N={n}: {c}" for n, c in sorted(dist.by_n_qubits.items())]
            lines.append(f"| N-qubits | {', '.join(parts)} |")
        if dist.by_p_layers:
            parts = [f"p={p}: {c}" for p, c in sorted(dist.by_p_layers.items())]
            lines.append(f"| p-layers | {', '.join(parts)} |")
        lines.append("")

    # Timing
    timing = report.timing
    if timing.total_runs > 0:
        lines.append("## Timing")
        lines.append("")
        lines.append(
            f"- Total compute: **{timing.total_pipeline_hours:.1f} hours** "
            f"across {timing.total_runs} runs"
        )
        lines.append(
            f"- Per-run: mean={timing.mean_run_s:.1f}s, "
            f"median={timing.median_run_s:.1f}s, "
            f"max={timing.max_run_s:.1f}s"
        )
        lines.append("")

    # Actions
    if report.actions:
        lines.append("## Actionable Items")
        lines.append("")
        for i, action in enumerate(report.actions, 1):
            icon = _priority_icon(action.priority)
            lines.append(f"{i}. {icon} **[{action.priority.name}]** {action.title}")
            lines.append(f"   - {action.detail}")
        lines.append("")

    return "\n".join(lines)


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
