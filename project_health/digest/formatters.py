"""Formatters for the result digest system.

Provides text table and markdown output for each result kind,
surfacing the metrics that matter for each.

No external dependencies — stdlib only.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Any

from project_health.digest.models import (
    CrossTopologyResult,
    ExperimentResult,
    ModeComparisonResult,
    N120SweepResult,
    NoiselessResult,
    NoisyResult,
    ScalingResult,
)

# ═══════════════════════════════════════════════════════════════════════════
# Text formatters
# ═══════════════════════════════════════════════════════════════════════════


def format_noiseless_text(results: list[NoiselessResult], verbose: bool = False) -> str:
    """Format noiseless pipeline results as a text table."""
    if not results:
        return "  No noiseless pipeline results found.\n"

    lines: list[str] = []

    # Summary statistics
    de_gaps = [r.delta_e_over_gap for r in results if r.delta_e_over_gap is not None]
    n_pass = sum(1 for d in de_gaps if d < 0.05)
    n_marginal = sum(1 for d in de_gaps if 0.05 <= d < 0.10)
    n_fail = sum(1 for d in de_gaps if d >= 0.10)

    lines.append(f"  {len(results)} runs scanned")
    if de_gaps:
        lines.append(
            f"  Pass(<5%): {n_pass} | Marginal(5-10%): {n_marginal} | Fail(>10%): {n_fail}"
        )
        lines.append(
            f"  ΔE/gap — median: {statistics.median(de_gaps):.4f}, "
            f"mean: {statistics.mean(de_gaps):.4f}, "
            f"best: {min(de_gaps):.4f}, worst: {max(de_gaps):.4f}"
        )
    lines.append("")

    # Table
    hdr = (
        f"  {'Variant':<28} {'N':<4} {'Topo':<10} {'p':<3} "
        f"{'Regime':<12} "
        f"{'ΔE/gap':<9} {'Conv%':<7} {'θ-smooth':<10} {'Gen.gap':<10} "
        f"{'Time':<7}"
    )
    lines.append(hdr)
    lines.append(f"  {'─' * 108}")

    for r in results:
        de_str = f"{r.delta_e_over_gap:.4f}" if r.delta_e_over_gap is not None else "—"
        conv_str = f"{r.convergence_rate * 100:.0f}%" if r.convergence_rate is not None else "—"
        smooth_str = f"{r.theta_smoothness:.5f}" if r.theta_smoothness is not None else "—"
        gap_str = f"{r.generalization_gap:.2e}" if r.generalization_gap is not None else "—"
        time_str = _format_time(r.elapsed_s)
        icon = _noiseless_icon(r.delta_e_over_gap)
        regime_str = r.regime or "—"

        lines.append(
            f"  {icon} {r.variant_id:<25} {r.n_qubits:<4} {r.topology:<10} "
            f"{r.p_layers:<3} {regime_str:<12} {de_str:<9} {conv_str:<7} "
            f"{smooth_str:<10} {gap_str:<10} {time_str:<7}"
        )

        if verbose:
            extras = []
            if r.h_test:
                extras.append(f"h_test={r.h_test}")
            if r.hidden_dim != 128:
                extras.append(f"hidden={r.hidden_dim}")
            if r.n_restarts != 5:
                extras.append(f"restarts={r.n_restarts}")
            if r.mag_x_error is not None:
                extras.append(f"mag_x_err={r.mag_x_error:.4f}")
            if r.corr_zz_error is not None:
                extras.append(f"corr_zz_err={r.corr_zz_error:.4f}")
            if extras:
                lines.append(f"       {', '.join(extras)}")

    return "\n".join(lines)


def format_noisy_text(results: list[NoisyResult], verbose: bool = False) -> str:
    """Format noisy/ZNE results as a text table."""
    if not results:
        return "  No noisy/ZNE results found.\n"

    lines: list[str] = []

    # Summary
    r2_values = [r.mean_r2 for r in results]
    gains = [r.mean_gain_pct for r in results]
    n_success = sum(1 for r in results if r.success_criteria_met)

    lines.append(f"  {len(results)} runs scanned")
    lines.append(f"  Success criteria met: {n_success}/{len(results)}")
    lines.append(
        "  (Success = ZNE beats raw noisy AND R²>0.8. "
        "High R² + negative gain = good fit, wrong direction.)"
    )
    if r2_values:
        lines.append(
            f"  R² — mean: {statistics.mean(r2_values):.4f}, "
            f"best: {max(r2_values):.4f}, worst: {min(r2_values):.4f}"
        )
    if gains:
        lines.append(
            f"  Gain% — mean: {statistics.mean(gains):+.1f}%, "
            f"best: {max(gains):+.1f}%, worst: {min(gains):+.1f}%"
        )
    lines.append("")

    # Table
    hdr = (
        f"  {'Variant':<25} {'N':<4} {'Topo':<10} {'Lay':<5} "
        f"{'Shots':<7} {'R²':<7} {'Gain%':<8} {'Wins':<6} "
        f"{'ΔE/g_raw':<9} {'ΔE/g_zne':<9}"
    )
    lines.append(hdr)
    lines.append(f"  {'─' * 94}")

    for r in results:
        icon = "✅" if r.success_criteria_met else "❌"
        wins_str = f"{r.n_mitigated_wins}/{r.n_total}"

        lines.append(
            f"  {icon} {r.variant_id:<22} {r.n_qubits:<4} {r.topology:<10} "
            f"{r.n_layouts:<5} {r.shots:<7} {r.mean_r2:<7.4f} "
            f"{r.mean_gain_pct:+7.1f}% {wins_str:<6} "
            f"{r.mean_de_noisy_raw:<9.4f} {r.mean_de_zne:<9.4f}"
        )

        if verbose and r.per_h_r2:
            h_detail = ", ".join(
                f"h={h:.1f}:R²={r2:.3f}" for h, r2 in zip(r.h_values, r.per_h_r2, strict=False)
            )
            lines.append(f"       {h_detail}")

    return "\n".join(lines)


def format_experiment_text(results: list[ExperimentResult], verbose: bool = False) -> str:
    """Format BaseExperiment results as a text table."""
    if not results:
        return "  No BaseExperiment results found.\n"

    lines: list[str] = []

    # Summary
    n_confirmed = sum(1 for r in results if r.verdict == "confirmed")
    n_rejected = sum(1 for r in results if r.verdict == "rejected")
    n_failed = sum(1 for r in results if r.verdict == "failed")

    lines.append(f"  {len(results)} experiments scanned")
    lines.append(
        f"  Confirmed: {n_confirmed} ✅ | Rejected: {n_rejected} ⚠️  | Failed: {n_failed} ❌"
    )
    lines.append(
        "  Note: 'rejected' = hypothesis disproved (valid finding). "
        "'failed' = did not meet strict threshold."
    )
    lines.append("")

    # Table
    hdr = (
        f"  {'ID':<7} {'Cat':<5} {'Verdict':<12} "
        f"{'ΔE/gap':<12} {'Pass%':<7} {'Seeds':<6} {'Criteria'}"
    )
    lines.append(hdr)
    lines.append(f"  {'─' * 80}")

    for r in results:
        emoji = _verdict_emoji(r.verdict)
        de_str = f"{r.mean_de_gap:.4f}" if r.mean_de_gap is not None else "—"
        pr_str = f"{r.pass_rate * 100:.0f}%" if r.pass_rate is not None else "—"

        lines.append(
            f"  {r.experiment_id:<7} {r.category:<5} "
            f"{emoji} {r.verdict:<9} {de_str:<12} {pr_str:<7} "
            f"{r.n_seeds:<6} {r.criteria}"
        )

        if verbose:
            if r.hypothesis:
                lines.append(f"       H: {r.hypothesis}")
            for key, val in r.extras.items():
                if val:
                    lines.append(f"       {key}: {val}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Grouped comparison formatters (for --group-by studies)
# ═══════════════════════════════════════════════════════════════════════════

_NOISELESS_GROUP_EXTRACTORS: dict[str, Any] = {
    "topology": lambda r: r.topology or "unknown",
    "n_qubits": lambda r: r.n_qubits,
    "hidden_dim": lambda r: r.hidden_dim,
    "n_restarts": lambda r: r.n_restarts,
    "p_layers": lambda r: r.p_layers,
    "model": lambda r: r.model or "tfim",
}

_NOISY_GROUP_EXTRACTORS: dict[str, Any] = {
    "topology": lambda r: r.topology or "unknown",
    "n_qubits": lambda r: r.n_qubits,
    "n_layouts": lambda r: r.n_layouts,
    "shots": lambda r: r.shots,
    "p_layers": lambda r: r.p_layers,
}


def format_noiseless_grouped(results: list[NoiselessResult], group_key: str) -> str:
    """Group noiseless results by a dimension and show aggregate comparison.

    Supported group keys: topology, n_qubits, hidden_dim, n_restarts, p_layers.
    """
    if not results:
        return "  No results to group.\n"

    key_fn = _NOISELESS_GROUP_EXTRACTORS.get(group_key)
    if key_fn is None:
        valid = ", ".join(_NOISELESS_GROUP_EXTRACTORS.keys())
        return f"  Unknown group key '{group_key}'. Valid: {valid}\n"

    groups: dict[str, list[NoiselessResult]] = {}
    for r in results:
        val = str(key_fn(r))
        groups.setdefault(val, []).append(r)

    lines: list[str] = []
    lines.append(f"  Grouped by: {group_key} ({len(groups)} groups)\n")
    lines.append(
        f"  {'Group':<15} {'Count':<7} {'Pass':<6} {'Marg':<6} {'Fail':<6} "
        f"{'Med ΔE/gap':<12} {'Mean ΔE/gap':<12} {'Best':<10} {'Worst':<10}"
    )
    lines.append(f"  {'─' * 92}")

    for group_val in sorted(groups.keys(), key=_natural_sort_key):
        group = groups[group_val]
        de_gaps = [r.delta_e_over_gap for r in group if r.delta_e_over_gap is not None]
        n_pass = sum(1 for d in de_gaps if d < 0.05)
        n_marg = sum(1 for d in de_gaps if 0.05 <= d < 0.10)
        n_fail = sum(1 for d in de_gaps if d >= 0.10)
        n_total = len(group)

        if de_gaps:
            med = statistics.median(de_gaps)
            mean = statistics.mean(de_gaps)
            best = min(de_gaps)
            worst = max(de_gaps)
            lines.append(
                f"  {group_val:<15} {n_total:<7} {n_pass:<6} {n_marg:<6} "
                f"{n_fail:<6} {med:<12.4f} {mean:<12.4f} {best:<10.4f} {worst:<10.4f}"
            )
        else:
            lines.append(
                f"  {group_val:<15} {n_total:<7} {'—':<6} {'—':<6} "
                f"{'—':<6} {'—':<12} {'—':<12} {'—':<10} {'—':<10}"
            )

    return "\n".join(lines)


def format_noisy_grouped(results: list[NoisyResult], group_key: str) -> str:
    """Group noisy results by a dimension and show aggregate comparison.

    Supported group keys: topology, n_qubits, n_layouts, shots, p_layers.
    """
    if not results:
        return "  No results to group.\n"

    key_fn = _NOISY_GROUP_EXTRACTORS.get(group_key)
    if key_fn is None:
        valid = ", ".join(_NOISY_GROUP_EXTRACTORS.keys())
        return f"  Unknown group key '{group_key}'. Valid: {valid}\n"

    groups: dict[str, list[NoisyResult]] = {}
    for r in results:
        val = str(key_fn(r))
        groups.setdefault(val, []).append(r)

    lines: list[str] = []
    lines.append(f"  Grouped by: {group_key} ({len(groups)} groups)\n")
    lines.append(
        f"  {'Group':<15} {'Count':<7} {'Success':<9} "
        f"{'Mean R²':<9} {'Mean Gain%':<12} {'Best Gain':<11} {'Worst Gain':<11}"
    )
    lines.append(f"  {'─' * 80}")

    for group_val in sorted(groups.keys(), key=_natural_sort_key):
        group = groups[group_val]
        n_total = len(group)
        n_success = sum(1 for r in group if r.success_criteria_met)
        r2_vals = [r.mean_r2 for r in group]
        gain_vals = [r.mean_gain_pct for r in group]

        mean_r2 = statistics.mean(r2_vals) if r2_vals else 0
        mean_gain = statistics.mean(gain_vals) if gain_vals else 0
        best_gain = max(gain_vals) if gain_vals else 0
        worst_gain = min(gain_vals) if gain_vals else 0

        lines.append(
            f"  {group_val:<15} {n_total:<7} {n_success}/{n_total:<6} "
            f"{mean_r2:<9.4f} {mean_gain:+10.1f}%  "
            f"{best_gain:+9.1f}%  {worst_gain:+9.1f}%"
        )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Markdown formatter
# ═══════════════════════════════════════════════════════════════════════════


def format_markdown(
    noiseless: list[NoiselessResult],
    noisy: list[NoisyResult],
    experiments: list[ExperimentResult],
    verbose: bool = False,
) -> str:
    """Format all results as a markdown document."""
    lines: list[str] = []
    lines.append("# Results Digest\n")

    # ── Experiments
    if experiments:
        lines.append("## Hypothesis Tests (BaseExperiment)\n")
        n_conf = sum(1 for r in experiments if r.verdict == "confirmed")
        n_rej = sum(1 for r in experiments if r.verdict == "rejected")
        n_fail = sum(1 for r in experiments if r.verdict == "failed")
        lines.append(
            f"**Summary**: {len(experiments)} experiments — "
            f"{n_conf} confirmed ✅, {n_rej} rejected ⚠️, {n_fail} failed ❌\n"
        )
        lines.append("| ID | Category | Verdict | ΔE/gap | Pass% | Criteria |")
        lines.append("|-----|----------|---------|--------|-------|----------|")
        for r in experiments:
            emoji = _verdict_emoji(r.verdict)
            de_str = f"{r.mean_de_gap:.4f}" if r.mean_de_gap is not None else "—"
            pr_str = f"{r.pass_rate * 100:.0f}%" if r.pass_rate is not None else "—"
            lines.append(
                f"| {r.experiment_id} | {r.category} | {emoji} {r.verdict} | "
                f"{de_str} | {pr_str} | {r.criteria} |"
            )
        lines.append("")

        if verbose:
            lines.append("### Details\n")
            for r in experiments:
                lines.append(f"- **{r.experiment_id}**: {r.hypothesis}")
            lines.append("")

    # ── Noiseless
    if noiseless:
        lines.append("## Noiseless Pipeline Runs\n")
        de_gaps = [r.delta_e_over_gap for r in noiseless if r.delta_e_over_gap is not None]
        n_pass = sum(1 for d in de_gaps if d < 0.05)
        if de_gaps:
            lines.append(
                f"**Summary**: {len(noiseless)} runs — "
                f"{n_pass}/{len(de_gaps)} pass (ΔE/gap < 5%), "
                f"median = {statistics.median(de_gaps):.4f}\n"
            )

        lines.append("| Variant | N | Topology | p | ΔE/gap | Conv% | θ-smooth | Gen.gap |")
        lines.append("|---------|---|----------|---|--------|-------|----------|---------|")
        for r in noiseless:
            de_str = f"{r.delta_e_over_gap:.4f}" if r.delta_e_over_gap is not None else "—"
            conv_str = f"{r.convergence_rate * 100:.0f}%" if r.convergence_rate is not None else "—"
            smooth_str = f"{r.theta_smoothness:.5f}" if r.theta_smoothness is not None else "—"
            gap_str = f"{r.generalization_gap:.2e}" if r.generalization_gap is not None else "—"
            lines.append(
                f"| {r.variant_id} | {r.n_qubits} | {r.topology} | "
                f"{r.p_layers} | {de_str} | {conv_str} | {smooth_str} | {gap_str} |"
            )
        lines.append("")

    # ── Noisy
    if noisy:
        lines.append("## Noisy/ZNE Runs\n")
        n_success = sum(1 for r in noisy if r.success_criteria_met)
        r2_vals = [r.mean_r2 for r in noisy]
        if r2_vals:
            lines.append(
                f"**Summary**: {len(noisy)} runs — "
                f"{n_success}/{len(noisy)} meet criteria, "
                f"mean R² = {statistics.mean(r2_vals):.4f}\n"
            )

        lines.append("| Variant | N | Topology | Layouts | Shots | R² | Gain% | Wins |")
        lines.append("|---------|---|----------|---------|-------|-----|-------|------|")
        for r in noisy:
            wins_str = f"{r.n_mitigated_wins}/{r.n_total}"
            lines.append(
                f"| {r.variant_id} | {r.n_qubits} | {r.topology} | "
                f"{r.n_layouts} | {r.shots} | {r.mean_r2:.4f} | "
                f"{r.mean_gain_pct:+.1f}% | {wins_str} |"
            )
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _noiseless_icon(delta_e: float | None) -> str:
    """Return emoji for noiseless result based on ΔE/gap."""
    if delta_e is None:
        return "? "
    if delta_e < 0.05:
        return "✅"
    if delta_e < 0.10:
        return "⚠️"
    return "❌"


def _verdict_emoji(verdict: str) -> str:
    """Return emoji for experiment verdict."""
    return {"confirmed": "✅", "rejected": "⚠️", "failed": "❌"}.get(verdict, "?")


def _format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds <= 0:
        return "—"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _natural_sort_key(s: str) -> tuple:
    """Sort strings with embedded numbers naturally (e.g., '6' before '10')."""
    parts = re.split(r"(\d+)", s)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


# ═══════════════════════════════════════════════════════════════════════════
# Statistical analysis formatters
# ═══════════════════════════════════════════════════════════════════════════


def format_noiseless_stats(results: list[NoiselessResult]) -> str:
    """Detailed statistical summary of noiseless results."""
    if not results:
        return "  No noiseless results for statistical analysis.\n"

    lines: list[str] = []
    lines.append("  STATISTICAL SUMMARY — Noiseless Pipeline\n")

    de_gaps = sorted(r.delta_e_over_gap for r in results if r.delta_e_over_gap is not None)
    if not de_gaps:
        return "  No ΔE/gap values available.\n"

    n = len(de_gaps)
    mean = statistics.mean(de_gaps)
    med = statistics.median(de_gaps)
    stdev = statistics.stdev(de_gaps) if n > 1 else 0.0
    p25 = de_gaps[n // 4] if n >= 4 else de_gaps[0]
    p75 = de_gaps[3 * n // 4] if n >= 4 else de_gaps[-1]
    p90 = de_gaps[int(n * 0.9)] if n >= 10 else de_gaps[-1]

    lines.append(f"  ΔE/gap Distribution (n={n}):")
    lines.append(f"    Mean:   {mean:.5f}")
    lines.append(f"    Median: {med:.5f}")
    lines.append(f"    Stdev:  {stdev:.5f}")
    lines.append(f"    P25:    {p25:.5f}")
    lines.append(f"    P75:    {p75:.5f}")
    lines.append(f"    P90:    {p90:.5f}")
    lines.append(f"    Min:    {min(de_gaps):.5f}")
    lines.append(f"    Max:    {max(de_gaps):.5f}")
    lines.append(f"    IQR:    {p75 - p25:.5f}")
    lines.append("")

    # Distribution histogram (text-based)
    lines.append("  Distribution:")
    bins = [0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.50, 1.0, float("inf")]
    labels = ["<1%", "1-2%", "2-3%", "3-5%", "5-10%", "10-20%", "20-50%", "50-100%", ">100%"]
    prev = 0.0
    for i, upper in enumerate(bins):
        count = sum(1 for d in de_gaps if prev <= d < upper)
        bar = "█" * count
        pct = count / n * 100
        lines.append(f"    {labels[i]:<8} {bar:<30} {count:>3} ({pct:4.1f}%)")
        prev = upper

    lines.append("")

    # Convergence and MPNN quality
    conv_rates = [r.convergence_rate for r in results if r.convergence_rate is not None]
    gen_gaps = [r.generalization_gap for r in results if r.generalization_gap is not None]

    if conv_rates:
        lines.append(
            f"  Convergence rate: mean={statistics.mean(conv_rates):.3f}, min={min(conv_rates):.3f}"
        )
    if gen_gaps:
        lines.append(
            f"  Generalization gap: mean={statistics.mean(gen_gaps):.2e}, "
            f"median={statistics.median(gen_gaps):.2e}, "
            f"max={max(gen_gaps):.2e}"
        )

    return "\n".join(lines)


def format_noisy_stats(results: list[NoisyResult]) -> str:
    """Detailed statistical summary of noisy/ZNE results."""
    if not results:
        return "  No noisy results for statistical analysis.\n"

    lines: list[str] = []
    lines.append("  STATISTICAL SUMMARY — Noisy/ZNE\n")

    r2_vals = sorted(r.mean_r2 for r in results)
    gains = sorted(r.mean_gain_pct for r in results)
    n = len(results)

    lines.append(f"  R² Distribution (n={n}):")
    lines.append(f"    Mean:   {statistics.mean(r2_vals):.4f}")
    lines.append(f"    Median: {statistics.median(r2_vals):.4f}")
    lines.append(f"    Min:    {min(r2_vals):.4f}")
    lines.append(f"    Max:    {max(r2_vals):.4f}")
    lines.append("")

    lines.append(f"  Gain% Distribution (n={n}):")
    lines.append(f"    Mean:   {statistics.mean(gains):+.1f}%")
    lines.append(f"    Median: {statistics.median(gains):+.1f}%")
    lines.append(f"    Min:    {min(gains):+.1f}%")
    lines.append(f"    Max:    {max(gains):+.1f}%")
    n_positive = sum(1 for g in gains if g > 0)
    lines.append(f"    Positive gain: {n_positive}/{n} ({n_positive / n * 100:.0f}%)")
    lines.append("")

    # Correlation: R² vs gain
    if n >= 3:
        mean_r2 = statistics.mean(r2_vals)
        mean_gain = statistics.mean(gains)
        cov = sum((r.mean_r2 - mean_r2) * (r.mean_gain_pct - mean_gain) for r in results) / n
        std_r2 = statistics.stdev(r2_vals) if n > 1 else 1
        std_gain = statistics.stdev(gains) if n > 1 else 1
        corr = cov / (std_r2 * std_gain) if std_r2 > 0 and std_gain > 0 else 0
        lines.append(f"  Correlation(R², Gain%): {corr:.3f}")
        if abs(corr) < 0.3:
            lines.append("    → Weak: high R² does NOT guarantee positive gain")
        elif corr > 0.7:
            lines.append("    → Strong positive: R² is a good predictor of ZNE success")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Outlier detection
# ═══════════════════════════════════════════════════════════════════════════


def format_noiseless_outliers(results: list[NoiselessResult]) -> str:
    """Identify outliers — results that deviate significantly from the group."""
    if len(results) < 5:
        return "  Need at least 5 results for outlier detection.\n"

    lines: list[str] = []
    lines.append("  OUTLIER ANALYSIS — Noiseless Pipeline\n")

    de_gaps = [r.delta_e_over_gap for r in results if r.delta_e_over_gap is not None]
    if len(de_gaps) < 5:
        return "  Not enough ΔE/gap values for outlier detection.\n"

    # IQR method
    sorted_gaps = sorted(de_gaps)
    n = len(sorted_gaps)
    q1 = sorted_gaps[n // 4]
    q3 = sorted_gaps[3 * n // 4]
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr

    # Find outliers
    outliers = [
        r for r in results if r.delta_e_over_gap is not None and r.delta_e_over_gap > upper_fence
    ]

    if not outliers:
        lines.append(f"  No outliers detected (upper fence = {upper_fence:.4f})")
        lines.append(f"  IQR = {iqr:.4f}, Q1 = {q1:.4f}, Q3 = {q3:.4f}")
    else:
        lines.append(
            f"  {len(outliers)} outliers detected (above {upper_fence:.4f} = Q3 + 1.5×IQR)"
        )
        lines.append(f"  IQR = {iqr:.4f}, Q1 = {q1:.4f}, Q3 = {q3:.4f}\n")
        lines.append(f"  {'Variant':<28} {'ΔE/gap':<10} {'Topo':<10} {'N':<4} {'Why?'}")
        lines.append(f"  {'─' * 70}")
        for r in sorted(outliers, key=lambda x: x.delta_e_over_gap or 0, reverse=True):
            # Try to explain why
            reason = _diagnose_outlier(r)
            lines.append(
                f"  {r.variant_id:<28} {r.delta_e_over_gap:.4f}    "
                f"{r.topology:<10} {r.n_qubits:<4} {reason}"
            )

    # Also flag suspiciously good results (potential overfitting)
    lower_fence = max(0, q1 - 1.5 * iqr)
    too_good = [
        r
        for r in results
        if r.delta_e_over_gap is not None
        and r.delta_e_over_gap < lower_fence
        and r.generalization_gap is not None
        and r.generalization_gap > 0.01
    ]
    if too_good:
        lines.append(f"\n  ⚠️  {len(too_good)} suspiciously good (low ΔE/gap + high gen.gap):")
        for r in too_good:
            lines.append(
                f"    {r.variant_id}: ΔE/gap={r.delta_e_over_gap:.4f}, "
                f"gen_gap={r.generalization_gap:.2e}"
            )

    return "\n".join(lines)


def _diagnose_outlier(r: NoiselessResult) -> str:
    """Try to explain why a result is an outlier."""
    reasons = []
    if r.generalization_gap is not None and r.generalization_gap > 0.01:
        reasons.append("high gen.gap")
    if r.theta_smoothness is not None and r.theta_smoothness > 1.0:
        reasons.append("rough θ-sweep")
    if r.n_restarts < 3:
        reasons.append(f"only {r.n_restarts} restart(s)")
    if r.hidden_dim < 64:
        reasons.append(f"small hidden={r.hidden_dim}")
    if not reasons:
        reasons.append("investigate manually")
    return "; ".join(reasons)


# ═══════════════════════════════════════════════════════════════════════════
# Side-by-side comparison of two variants/folders
# ═══════════════════════════════════════════════════════════════════════════


def format_compare_two(
    results_a: list[NoiselessResult],
    results_b: list[NoiselessResult],
    label_a: str,
    label_b: str,
) -> str:
    """Side-by-side comparison of two result sets."""
    lines: list[str] = []
    lines.append(f"  COMPARISON: {label_a} vs {label_b}\n")

    de_a = [r.delta_e_over_gap for r in results_a if r.delta_e_over_gap is not None]
    de_b = [r.delta_e_over_gap for r in results_b if r.delta_e_over_gap is not None]

    if not de_a or not de_b:
        return f"  Insufficient data for comparison ({len(de_a)} vs {len(de_b)} results).\n"

    # Metrics comparison table
    lines.append(f"  {'Metric':<25} {label_a:<20} {label_b:<20} {'Winner'}")
    lines.append(f"  {'─' * 75}")

    metrics = [
        ("Count", len(results_a), len(results_b), None),
        ("Mean ΔE/gap", statistics.mean(de_a), statistics.mean(de_b), "lower"),
        ("Median ΔE/gap", statistics.median(de_a), statistics.median(de_b), "lower"),
        ("Best ΔE/gap", min(de_a), min(de_b), "lower"),
        ("Worst ΔE/gap", max(de_a), max(de_b), "lower"),
        (
            "Pass rate (<5%)",
            sum(1 for d in de_a if d < 0.05) / len(de_a),
            sum(1 for d in de_b if d < 0.05) / len(de_b),
            "higher",
        ),
    ]

    # Add convergence if available
    conv_a = [r.convergence_rate for r in results_a if r.convergence_rate is not None]
    conv_b = [r.convergence_rate for r in results_b if r.convergence_rate is not None]
    if conv_a and conv_b:
        metrics.append(
            ("Mean conv. rate", statistics.mean(conv_a), statistics.mean(conv_b), "higher")
        )

    # Add gen gap if available
    gap_a = [r.generalization_gap for r in results_a if r.generalization_gap is not None]
    gap_b = [r.generalization_gap for r in results_b if r.generalization_gap is not None]
    if gap_a and gap_b:
        metrics.append(("Mean gen. gap", statistics.mean(gap_a), statistics.mean(gap_b), "lower"))

    # Add timing
    time_a = [r.elapsed_s for r in results_a if r.elapsed_s > 0]
    time_b = [r.elapsed_s for r in results_b if r.elapsed_s > 0]
    if time_a and time_b:
        metrics.append(("Mean time (s)", statistics.mean(time_a), statistics.mean(time_b), "lower"))

    for name, val_a, val_b, direction in metrics:
        if direction is None:
            winner = ""
        elif direction == "lower":
            winner = f"← {label_a}" if val_a < val_b else f"→ {label_b}"
        else:
            winner = f"← {label_a}" if val_a > val_b else f"→ {label_b}"

        if isinstance(val_a, float) and val_a < 1:
            a_str = f"{val_a:.5f}"
            b_str = f"{val_b:.5f}"
        elif isinstance(val_a, float):
            a_str = f"{val_a:.2f}"
            b_str = f"{val_b:.2f}"
        else:
            a_str = str(val_a)
            b_str = str(val_b)

        lines.append(f"  {name:<25} {a_str:<20} {b_str:<20} {winner}")

    # Improvement percentage
    mean_a = statistics.mean(de_a)
    mean_b = statistics.mean(de_b)
    if mean_a > 0 and mean_b > 0:
        improvement = (mean_a - mean_b) / mean_a * 100
        if improvement > 0:
            lines.append(f"\n  → {label_b} is {improvement:.1f}% better (lower mean ΔE/gap)")
        else:
            lines.append(f"\n  → {label_a} is {-improvement:.1f}% better (lower mean ΔE/gap)")

    return "\n".join(lines)


def format_cross_topology_text(results: list[CrossTopologyResult], verbose: bool = False) -> str:
    """Format cross-topology transfer results as a text table.

    Groups by experiment type and shows key metrics for each.
    """
    if not results:
        return "  No cross-topology transfer results found.\n"

    lines: list[str] = []
    lines.append(f"  {len(results)} cross-topology result files scanned")
    lines.append("")

    # Group by experiment type
    by_type: dict[str, list[CrossTopologyResult]] = {}
    for r in results:
        by_type.setdefault(r.experiment_type, []).append(r)

    # Cross-N validation results
    if "cross_n_validation" in by_type:
        lines.append("  ── Cross-N Validation (within-topology) ──")
        lines.append(f"  {'File':<45} {'Verdict':<8} {'Mean ΔE/gap':<12} {'Time':<8}")
        lines.append(f"  {'─' * 73}")
        for r in by_type["cross_n_validation"]:
            fname = Path(r.source_file).name[:42]
            v = "✅" if r.all_pass else "❌"
            lines.append(
                f"  {fname:<45} {v} {r.verdict:<5} "
                f"{r.mean_de_gap * 100:>7.3f}%  {r.total_time_s:>5.0f}s"
            )
        lines.append("")

    # Cross-topology transfer results
    if "cross_topology_transfer" in by_type:
        lines.append("  ── Cross-Topology Transfer (bidirectional) ──")
        lines.append(f"  {'File':<45} {'Verdict':<8} {'Mean ΔE/gap':<12} {'Time':<8}")
        lines.append(f"  {'─' * 73}")
        for r in by_type["cross_topology_transfer"]:
            fname = Path(r.source_file).name[:42]
            v = "✅" if r.all_pass else "❌"
            lines.append(
                f"  {fname:<45} {v} {r.verdict:<5} "
                f"{r.mean_de_gap * 100:>7.3f}%  {r.total_time_s:>5.0f}s"
            )
            if verbose and r.directions:
                for dir_key, dir_data in r.directions.items():
                    if isinstance(dir_data, dict):
                        de = dir_data.get("mean_de_gap", {})
                        de_val = de.get("mean", 0.0) if isinstance(de, dict) else de
                        lines.append(f"    └─ {dir_key}: {de_val * 100:.3f}%")
        lines.append("")

    # Ablation results
    if "ablation_study" in by_type:
        lines.append("  ── Ablation (GNN vs MLP vs Scipy) ──")
        lines.append(f"  {'File':<40} {'Graph Essential':<16} {'MLP/GNN':<10} {'Best Norm':<10}")
        lines.append(f"  {'─' * 76}")
        for r in by_type["ablation_study"]:
            fname = Path(r.source_file).name[:37]
            ess = "YES ✅" if r.graph_structure_essential else "NO"
            lines.append(
                f"  {fname:<40} {ess:<16} {r.mlp_gnn_ratio:>6.1f}×    {r.best_norm_type:<10}"
            )
        lines.append("")

    # Orchestrator summaries
    if "orchestrator_summary" in by_type:
        lines.append("  ── Orchestrator Runs ──")
        for r in by_type["orchestrator_summary"]:
            fname = Path(r.source_file).name[:45]
            v = "✅" if r.all_pass else "⚠️"
            lines.append(f"  {fname}: {v} {r.verdict} ({r.total_time_s / 60:.1f}m)")
        lines.append("")

    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# Scaling results formatter
# ═══════════════════════════════════════════════════════════════════════════


def format_scaling_text(
    scaling: list[ScalingResult],
    mode_comparison: ModeComparisonResult | None = None,
    n120_sweep: N120SweepResult | None = None,
    verbose: bool = False,
) -> str:
    """Format MPS scaling validation results as a text table.

    Includes:
    - Standard scaling validation runs (N=40/50/80 with seeds)
    - Mode comparison (deterministic vs stochastic)
    - N=120 rigorous sweep
    """
    lines: list[str] = []

    # ── Standard scaling results ──────────────────────────────────────
    if scaling:
        lines.append("  ── MPS Scaling Validation Runs ──")
        lines.append(f"  {len(scaling)} result files scanned")
        lines.append("")

        # Aggregate by N
        by_n: dict[int, list[ScalingResult]] = {}
        for r in scaling:
            by_n.setdefault(r.n_qubits, []).append(r)

        # Summary table
        lines.append(
            f"  {'N':<5} {'Seeds':<8} {'Pass':<8} "
            f"{'Mean ΔE/gap':<13} {'Max ΔE/gap':<12} {'Total Time':<12} {'Status'}"
        )
        lines.append(f"  {'─' * 70}")

        for n in sorted(by_n.keys()):
            runs = by_n[n]
            # Pick the best representative run per N:
            # Prefer multi-seed > latest timestamp (encoded in filename)
            best = max(
                runs,
                key=lambda r: (
                    r.n_total,  # More data points = better
                    Path(r.source_file).stem,  # Later timestamp sorts higher
                ),
            )
            n_seeds = best.n_total // max(len(best.h_values), 1)
            status = "✅ PASS" if best.all_passed else "❌ FAIL"
            lines.append(
                f"  {n:<5} {n_seeds:<8} "
                f"{best.n_pass}/{best.n_total:<5} "
                f"{best.mean_de_gap * 100:>8.4f}%   "
                f"{best.max_de_gap * 100:>8.4f}%   "
                f"{_format_time(best.total_time_s):<12} {status}"
            )

            if verbose:
                lines.append(
                    f"       h-values: {[f'{h:.2f}' for h in best.h_values]}"
                )
                lines.append(
                    f"       strategy={best.strategy}, χ={best.chi_max}, "
                    f"precision={best.precision}"
                )
                lines.append(f"       file: {Path(best.source_file).name}")

        lines.append("")

        # Scaling law validation
        lines.append("  ── Scaling Law: h_min = 1.5 + 0.020·N^1.31 ──")
        for n in sorted(by_n.keys()):
            best = max(
                by_n[n],
                key=lambda r: (r.n_total, Path(r.source_file).stem),
            )
            h_min_pred = 1.5 + 0.020 * n**1.31
            lowest_h = min(best.h_values) if best.h_values else 0
            margin = lowest_h - h_min_pred
            icon = "✅" if best.all_passed and margin >= 0 else "⚠️"
            lines.append(
                f"  {icon} N={n:>3}: h_min_pred={h_min_pred:.2f}, "
                f"lowest_h_tested={lowest_h:.2f} (margin=+{margin:.2f})"
            )
        lines.append("")

    # ── N=120 sweep ───────────────────────────────────────────────────
    if n120_sweep:
        lines.append("  ── N=120 Rigorous Sweep ──")
        status = "✅ PASS" if n120_sweep.scaling_law_validated else "❌ FAIL"
        lines.append(f"  Status: {status} ({n120_sweep.n_pass}/{n120_sweep.n_total})")
        lines.append(
            f"  h_min_safe = {n120_sweep.h_min_safe:.4f} "
            f"(formula: 1.5 + 0.020·120^1.31)"
        )
        lines.append(f"  h-values: {n120_sweep.h_values}")
        lines.append(f"  Seeds: {n120_sweep.seeds}")
        lines.append(
            f"  Mean ΔE/gap: {n120_sweep.mean_de_gap * 100:.4f}% ± "
            f"{n120_sweep.std_de_gap * 100:.4f}%"
        )
        lines.append(f"  Max ΔE/gap: {n120_sweep.max_de_gap * 100:.4f}%")
        if n120_sweep.bootstrap_ci_95:
            lo, hi = n120_sweep.bootstrap_ci_95
            lines.append(f"  Bootstrap 95% CI: [{lo * 100:.4f}%, {hi * 100:.4f}%]")
        lines.append(f"  Total time: {_format_time(n120_sweep.total_time_s)}")
        lines.append("")

    # ── Mode comparison ───────────────────────────────────────────────
    if mode_comparison:
        lines.append("  ── MPS Mode Comparison (Deterministic vs Stochastic) ──")
        consist = "✅ CONSISTENT" if mode_comparison.modes_consistent else "⚠️ DIVERGENT"
        lines.append(f"  Modes: {consist}")
        lines.append(
            f"  Mean speedup (det over sto): {mode_comparison.mean_speedup:.1f}×"
        )
        lines.append(
            f"  Mean energy difference: {mode_comparison.mean_energy_diff:.6f}"
        )
        lines.append(
            f"  Deterministic all pass: {mode_comparison.all_det_pass}, "
            f"Stochastic all pass: {mode_comparison.all_sto_pass}"
        )
        lines.append("")

        if verbose and mode_comparison.results:
            lines.append(
                f"  {'N':<5} {'h':<6} {'Det ΔE/gap':<12} {'Sto ΔE/gap':<12} "
                f"{'Speedup':<10} {'Energy Δ'}"
            )
            lines.append(f"  {'─' * 60}")
            for r in mode_comparison.results:
                det = r.get("deterministic", {})
                sto = r.get("stochastic", {})
                comp = r.get("comparison", {})
                lines.append(
                    f"  {r['N']:<5} {r['h']:<6.1f} "
                    f"{det.get('de_gap', 0) * 100:>8.4f}%   "
                    f"{sto.get('de_gap', 0) * 100:>8.4f}%   "
                    f"{comp.get('speedup_factor', 0):>6.1f}×   "
                    f"{comp.get('energy_diff', 0):.6f}"
                )
            lines.append("")

    # ── Overall thesis summary ────────────────────────────────────────
    if scaling or n120_sweep:
        lines.append("  ── Thesis Summary (Scaling) ──")
        all_n = set()
        if scaling:
            for r in scaling:
                if r.all_passed:
                    all_n.add(r.n_qubits)
        if n120_sweep and n120_sweep.scaling_law_validated:
            all_n.add(120)
        all_n_sorted = sorted(all_n)
        lines.append(
            f"  Validated system sizes: N ∈ {{{', '.join(str(n) for n in all_n_sorted)}}}"
        )
        if all_n_sorted:
            lines.append(
                f"  Range: N={min(all_n_sorted)} to N={max(all_n_sorted)}"
            )
        lines.append(
            "  Scaling law: h_min = 1.5 + 0.020·N^1.31 "
            "(corrected formula, +0.50 offset from original)"
        )
        if mode_comparison and mode_comparison.modes_consistent:
            lines.append(
                f"  MPS evaluation: deterministic mode validated "
                f"({mode_comparison.mean_speedup:.0f}× faster, consistent results)"
            )
        lines.append("")

    if not scaling and not n120_sweep and not mode_comparison:
        lines.append("  No MPS scaling results found.\n")

    return "\n".join(lines)
