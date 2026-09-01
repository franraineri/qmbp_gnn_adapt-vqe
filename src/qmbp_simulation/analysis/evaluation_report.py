"""Evaluation Report Generator — Markdown per-point comparison reports.

Generates structured evaluation reports comparing MPNN predictions vs
ground truth and optional VQE baselines. Reports include:
- Quality profile (continuous score, grade, distribution)
- Per-h error breakdown with classification
- Metric reliability warnings
- MPNN vs VQE comparison table (if baseline available)
- MT vs ST head-to-head comparison with topology/N filtering

Designed to be called by any runner that produces per_h_results.
Decoupled from runner internals — operates on standardized dicts.

Usage:
    from qmbp_simulation.analysis.evaluation_report import (
        generate_evaluation_report,
        generate_comparison_table,
        generate_mt_vs_st_table,
        validate_metrics,
    )

    # Generate full markdown report
    path = generate_evaluation_report(
        mpnn_results_by_n={16: {...}, 20: {...}},
        topology="square",
        model_name="tfim_bond_resolved",
        checkpoint="unified_tfim_br_square_multiN_4+6+8+10+12+14_p1.pt",
        h_range=(2.5, 4.5),
        comparison=comparison_dict,  # Optional: from section_summary
    )

    # Just the comparison table (for embedding)
    table_lines = generate_comparison_table(comparison_dict)

    # Validate metric consistency
    warnings = validate_metrics(per_h_results, n_qubits=16)

    # ── MT vs ST Comparison (with filtering) ──────────────────────────────

    # Full MT vs ST across all topologies and N
    lines, summary = generate_mt_vs_st_table()

    # Filter: only chain_1d, N between 10 and 20
    lines, summary = generate_mt_vs_st_table(
        topology_filter="chain_1d", n_min=10, n_max=20
    )

    # Filter: heavy_hex + ladder at large N (extrapolation regime)
    lines, summary = generate_mt_vs_st_table(
        topology_filter=["heavy_hex", "ladder"], n_min=16
    )

    # All topologies but only small N (in-distribution for MT)
    lines, summary = generate_mt_vs_st_table(n_max=12)

    # Save to file + use all historical runs (not just latest)
    lines, summary = generate_mt_vs_st_table(
        latest_only=False,
        output_path="results/mt_vs_st_full_history.md"
    )
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Metric Validation
# ═══════════════════════════════════════════════════════════════════════════════


def validate_metrics(
    per_h_results: list[dict],
    *,
    n_qubits: int | None = None,
) -> list[str]:
    """Validate metric consistency and reliability for a set of per-h results.

    Checks:
    1. Variational principle violations (E_pred < E_exact)
    2. Gap validity (gap > 0 for ΔE/gap to be meaningful)
    3. Outlier detection (max >> mean indicates single catastrophe)
    4. Sample size confidence
    5. ΔE vs ΔE/gap cross-consistency

    Parameters
    ----------
    per_h_results : list[dict]
        Per-h evaluation results with keys: h, e_pred, e_exact, gap, de_gap, abs_error.
    n_qubits : int | None
        System size (for context in messages).

    Returns
    -------
    list[str]
        Human-readable warning messages. Empty list = all metrics reliable.
    """
    if not per_h_results:
        return ["⚠️ No results to validate"]

    warnings: list[str] = []
    n = len(per_h_results)
    n_label = f"N={n_qubits}" if n_qubits else ""

    # Check 1: Variational principle violations
    n_violations = sum(1 for p in per_h_results if p.get("e_pred", 0) < p.get("e_exact", 0) - 1e-8)
    if n_violations > 0:
        warnings.append(
            f"⚠️ {n_violations}/{n} points violate variational principle "
            f"(E_pred < E_exact) {n_label}"
        )

    # Check 2: Gap validity
    n_zero_gap = sum(1 for p in per_h_results if p.get("gap", 0) <= 1e-10)
    if n_zero_gap > 0:
        warnings.append(
            f"⚠️ {n_zero_gap}/{n} points have gap≈0 (ΔE/gap unreliable at criticality) {n_label}"
        )

    # Check 3: Outlier detection
    de_gaps = [p.get("de_gap", 0) for p in per_h_results]
    if de_gaps:
        mean_dg = float(np.mean(de_gaps))
        max_dg = float(np.max(de_gaps))
        if mean_dg > 0 and max_dg > 5 * mean_dg and n > 3:
            warnings.append(
                f"⚠️ Outlier: max ΔE/gap={max_dg:.3f} is {max_dg / mean_dg:.0f}× the mean "
                f"— median may be more representative {n_label}"
            )

    # Check 4: Sample size
    if n < 4:
        warnings.append(f"⚠️ Only {n} points — means have low statistical confidence {n_label}")

    # Check 5: ΔE vs ΔE/gap consistency
    abs_errors = [p.get("abs_error") for p in per_h_results if p.get("abs_error") is not None]
    gaps = [p.get("gap") for p in per_h_results if p.get("gap") is not None]
    if abs_errors and gaps and len(abs_errors) == len(gaps) == n:
        reconstructed_dg = [ae / max(g, 1e-10) for ae, g in zip(abs_errors, gaps, strict=False)]
        stored_dg = [p["de_gap"] for p in per_h_results]
        max_discrepancy = max(abs(r - s) for r, s in zip(reconstructed_dg, stored_dg, strict=False))
        if max_discrepancy > 0.01:
            warnings.append(
                f"⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = "
                f"{max_discrepancy:.4f} (possible stale e_exact or gap) {n_label}"
            )

    return warnings


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison Table
# ═══════════════════════════════════════════════════════════════════════════════


def generate_comparison_table(
    comparison: dict[int, dict],
    *,
    mpnn_results_by_n: dict[int, dict] | None = None,
) -> list[str]:
    """Generate MPNN vs VQE comparison markdown table lines.

    Parameters
    ----------
    comparison : dict[int, dict]
        Comparison dict keyed by N with entries:
        - "mpnn": {"mean_de_gap": ...}
        - "random_vqe": {"mean_de_gap": ..., "mean_abs_error": ..., "total_evals": ...}
        - "speedup": float
        - "mpnn_win_rate": float
    mpnn_results_by_n : dict | None
        Optional full MPNN results dict with "mean_abs_error" key per N.

    Returns
    -------
    list[str]
        Markdown lines for the comparison table (no trailing newline).
    """
    lines = []
    # Filter to int keys only (comparison may have "model_diagnostics" etc.)
    n_values = sorted(k for k in comparison if isinstance(k, int))

    has_random = any("random_vqe" in comparison.get(n, {}) for n in n_values)
    if not has_random:
        return lines

    lines.append("### MPNN vs Random VQE Comparison")
    lines.append("")
    lines.append(
        "| N | MPNN |ΔE| | VQE |ΔE| | MPNN ΔE/gap | VQE ΔE/gap | Speedup (evals) | MPNN win rate |"
    )
    lines.append(
        "|---|---------|---------|-------------|------------|-----------------|---------------|"
    )

    speedups = []
    for n_target in n_values:
        entry = comparison[n_target]
        mpnn_info = entry.get("mpnn", {})
        mpnn_dg = mpnn_info.get("mean_de_gap", 0)
        # Get mean_abs_error from mpnn_results_by_n if available
        mpnn_ae = None
        if mpnn_results_by_n and n_target in mpnn_results_by_n:
            mpnn_ae = mpnn_results_by_n[n_target].get("mean_abs_error")

        rand_info = entry.get("random_vqe", {})
        if rand_info:
            rand_dg = rand_info.get("mean_de_gap", 0)
            rand_ae = rand_info.get("mean_abs_error")
            speedup = entry.get("speedup", 0)
            win_rate = entry.get("mpnn_win_rate", 0)
            speedups.append(speedup)

            mpnn_ae_str = f"{mpnn_ae:.4f}" if mpnn_ae is not None else "—"
            rand_ae_str = f"{rand_ae:.4f}" if rand_ae is not None else "—"
            lines.append(
                f"| {n_target} | {mpnn_ae_str} | {rand_ae_str} | "
                f"{mpnn_dg:.4f} | {rand_dg:.4f} | "
                f"{speedup:.0f}× | {win_rate:.0%} |"
            )

    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        lines.append(f"| **avg** | | | | | **{avg_speedup:.0f}×** | |")

    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# Full Report Generation
# ═══════════════════════════════════════════════════════════════════════════════


def generate_evaluation_report(
    mpnn_results_by_n: dict[int, dict],
    *,
    topology: str,
    model_name: str = "tfim_bond_resolved",
    checkpoint: str = "unknown",
    h_range: tuple[float, float] = (2.5, 5.0),
    n_h_points: int = 6,
    p_layers: int = 1,
    target_n: list[int] | None = None,
    comparison: dict | None = None,
    output_dir: Path | str = "results/extrapolation_evals",
    is_multi_topology: bool | None = None,
) -> Path:
    """Generate a comprehensive markdown evaluation report.

    Parameters
    ----------
    mpnn_results_by_n : dict[int, dict]
        MPNN results keyed by N. Each value must have "per_point" and "n_params".
    topology : str
        Lattice topology.
    model_name : str
        Physics model name.
    checkpoint : str
        Checkpoint file used for predictions.
    h_range : tuple
        (h_min, h_max) of the evaluation sweep.
    n_h_points : int
        Number of h-grid points.
    p_layers : int
        HVA depth (used for subdirectory naming).
    target_n : list[int] | None
        Explicit N ordering. Defaults to sorted keys of mpnn_results_by_n.
    comparison : dict | None
        Full comparison dict (includes "random_vqe", "speedup", "metric_warnings").
    output_dir : Path | str
        Base directory for reports. File saved to {output_dir}/{topology}_p{p}/eval_...md
    is_multi_topology : bool | None
        If True, marks report as multi-topology (MT flag in filename and content).
        If None, auto-detects from checkpoint name (contains "multitopo" or "multi_topology").

    Returns
    -------
    Path
        Path to the generated markdown file.
    """
    from qmbp_simulation.analysis.metrics import classify_point_failure
    from qmbp_simulation.framework.quality_profile import (
        compute_quality_profile,
        format_quality_summary,
    )

    if target_n is None:
        target_n = sorted(mpnn_results_by_n.keys())

    output_dir = Path(output_dir)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    # Auto-detect multi-topology from checkpoint name
    if is_multi_topology is None:
        is_multi_topology = (
            "multitopo" in (checkpoint or "").lower()
            or "multi_topology" in (checkpoint or "").lower()
            or "_MT_" in (checkpoint or "")
            or checkpoint == "unified_tfim_br_MT"
        )

    subdir = output_dir / f"{topology}_p{p_layers}"
    subdir.mkdir(parents=True, exist_ok=True)
    mt_tag = "_MT" if is_multi_topology else ""
    report_path = subdir / f"evaluation_{topology}{mt_tag}_p{p_layers}_{ts}.md"

    # Resolve checkpoint display
    checkpoint_display = checkpoint
    if checkpoint_display in ("auto (zoo)", "", None):
        checkpoint_display = "unknown (auto-selected from zoo)"

    lines = [
        f"# Model Evaluation: {topology}",
        "",
    ]

    # Multi-topology banner
    if is_multi_topology:
        lines.extend(
            [
                "> **🌐 MULTI-TOPOLOGY MODEL** — This evaluation uses a model trained on "
                "multiple topologies simultaneously. Results reflect cross-topology "
                "transfer capability.",
                "",
            ]
        )

    lines.extend(
        [
            f"**Date**: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Model**: {checkpoint_display}",
            f"**p_layers**: {p_layers}",
            f"**Multi-topology**: {'YES' if is_multi_topology else 'no'}",
            f"**h-range**: [{h_range[0]}, {h_range[1]}] ({n_h_points} pts)",
            f"**Target N**: {target_n}",
            "",
        ]
    )

    # ── Comparison table (MPNN vs VQE) ────────────────────────────────────
    if comparison:
        table_lines = generate_comparison_table(comparison, mpnn_results_by_n=mpnn_results_by_n)
        if table_lines:
            lines.extend(table_lines)
            lines.append("")

    lines.extend(["---", ""])

    # ── Per-N detailed breakdown ──────────────────────────────────────────
    # Model-appropriate h_critical for regional analysis
    h_critical_map = {
        "tfim": 1.0,
        "tfim_bond_resolved": 1.0,
        "tfim_longitudinal": 1.0,
        "tfim_frustrated": 1.0,
    }
    h_c = h_critical_map.get(model_name)

    for n_target in target_n:
        if n_target not in mpnn_results_by_n:
            continue

        mpnn = mpnn_results_by_n[n_target]
        per_point = mpnn.get("per_point", [])
        n_params = mpnn.get("n_params", 0)

        if not per_point:
            lines.append(f"## N = {n_target} — no data")
            lines.append("")
            continue

        # Quality profile
        profile = compute_quality_profile(per_point, h_critical=h_c, n_qubits=n_target)

        lines.append(f"## N = {n_target} ({n_params} params)")
        lines.append("")
        lines.append(f"**{format_quality_summary(profile)}**")
        lines.append("")

        # Metric reliability warnings
        metric_warnings = []
        if comparison and n_target in comparison:
            metric_warnings = comparison[n_target].get("metric_warnings", [])
        if not metric_warnings:
            # Compute on the fly if not pre-computed
            metric_warnings = validate_metrics(per_point, n_qubits=n_target)

        if metric_warnings:
            lines.append("> **Metric Reliability Warnings:**")
            for w in metric_warnings:
                lines.append(f"> - {w}")
            lines.append("")

        # Fidelity summary line (exact for N≤16, variance lower bound above).
        summary_stats = mpnn if isinstance(mpnn, dict) else {}
        mean_fid = summary_stats.get("mean_fidelity")
        if mean_fid is None:
            # Recompute from per_point if the section summary lacked it.
            _fids = [pt["fidelity"] for pt in per_point if pt.get("fidelity") is not None]
            mean_fid = float(sum(_fids) / len(_fids)) if _fids else None
        if mean_fid is not None:
            min_fid = summary_stats.get("min_fidelity")
            is_bound = summary_stats.get("fidelity_is_lower_bound") or any(
                pt.get("fidelity_is_bound") for pt in per_point
            )
            n_bound = summary_stats.get("n_fidelity_bound")
            if n_bound is None:
                n_bound = sum(1 for pt in per_point if pt.get("fidelity_is_bound"))
            rel = "≥" if is_bound else "="
            fid_note = (
                " (variance lower bound — N>16, exact statevector infeasible)"
                if is_bound
                else " (exact statevector overlap)"
            )
            min_str = f", min F{rel}{min_fid:.4f}" if min_fid is not None else ""
            lines.append(f"**Fidelity: mean F{rel}{mean_fid:.4f}{min_str}**{fid_note}")
            if n_bound:
                mev = summary_stats.get("mean_energy_variance")
                mev_str = f", mean Var(H)={mev:.4f}" if mev is not None else ""
                lines.append(
                    f"> {n_bound}/{len(per_point)} points use the Eckart bound "
                    f"F ≥ 1 − Var(H)/gap²{mev_str}."
                )
            lines.append("")

        # ── Infidelity decomposition (Var(H) vs gap) ──────────────────────
        # Attribute infidelity to dominant factor: dirty_state (attackable)
        # vs small_gap (physics ceiling near criticality). Diagnostic only.
        n_dirty = summary_stats.get("n_dirty_state")
        n_small_gap = summary_stats.get("n_small_gap")
        if n_dirty is None:
            n_dirty = sum(
                1 for pt in per_point if pt.get("infidelity_dominant_factor") == "dirty_state"
            )
        if n_small_gap is None:
            n_small_gap = sum(
                1 for pt in per_point if pt.get("infidelity_dominant_factor") == "small_gap"
            )
        if n_dirty or n_small_gap:
            mvog = summary_stats.get("mean_variance_over_gap2")
            mvog_str = f", mean Var(H)/gap²={mvog:.4f}" if mvog is not None else ""
            lines.append(
                f"**Infidelity decomposition:** {n_dirty} dirty-state "
                f"(attackable via optimization), {n_small_gap} small-gap "
                f"(physics ceiling near h_c){mvog_str}."
            )
            lines.append("")

        # Per-h table
        lines.append(
            "| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | "
            "Factor | Category | Action | Note |"
        )
        lines.append(
            "|---|--------|---------|------|--------|-----|----------|--------|"
            "--------|----------|--------|------|"
        )

        for p in per_point:
            h = p.get("h", 0)
            e_pred = p.get("e_pred", 0)
            e_exact = p.get("e_exact", 0)
            abs_err = p.get("abs_error", abs(e_pred - e_exact))
            gap = p.get("gap", 0)
            de_gap = p.get("de_gap", 0)
            method = p.get("method", "mpnn")

            # Fidelity cell: annotate lower bounds with ≥.
            fid = p.get("fidelity")
            if fid is None:
                fid_cell = "N/A"
            elif p.get("fidelity_is_bound"):
                fid_cell = f"≥{fid:.4f}"
            else:
                fid_cell = f"{fid:.4f}"

            # Var(H) and dominant infidelity factor (diagnostic).
            ev = p.get("energy_variance")
            var_cell = f"{ev:.4f}" if ev is not None else "N/A"
            factor = p.get("infidelity_dominant_factor")
            factor_cell = factor if factor else "—"

            # Per-point classification
            cls = classify_point_failure(
                de_gap=de_gap,
                abs_error=abs_err,
                gap=gap,
                h=h,
                h_critical=h_c,
                n_params=n_params,
            )

            cat_display = f"{cls.category}({cls.severity:.2f})"
            note = ""
            if method == "vqe_refined":
                note = "refined"
            elif method == "cached":
                note = "cached"

            lines.append(
                f"| {h:.3f} | {e_pred:.4f} | {e_exact:.4f} | "
                f"{abs_err:.4f} | {gap:.4f} | "
                f"{de_gap:.4f} | {fid_cell} | {var_cell} | {factor_cell} | "
                f"{cat_display} | {cls.action} | {note} |"
            )

        lines.append("")

    # Write
    report_path.write_text("\n".join(lines))

    try:
        display_path = report_path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        display_path = report_path
    logger.info(f"  📝 Evaluation report: {display_path}")

    return report_path


# ═══════════════════════════════════════════════════════════════════════════════
# θ Prediction Quality Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_theta_prediction(
    model,
    npz_path: Path | str,
    topology: str,
    *,
    p_layers: int = 1,
    model_name: str = "tfim_bond_resolved",
    max_points: int = 8,
    include_energy: bool = False,
) -> dict:
    """Evaluate MPNN θ prediction quality against NPZ ground truth.

    Computes per-h θ MSE (parameter-space error) and optionally energy error
    (function-space error). Useful for model quality assessment without
    running a full pipeline.

    Parameters
    ----------
    model : UnifiedMPNN | MPNNPredictor
        Loaded model in eval mode.
    npz_path : Path | str
        Path to NPZ with h_values, theta_opt, e_exact, e_vqe, gaps.
    topology : str
        Lattice topology for graph construction.
    p_layers : int
        HVA depth (for graph construction).
    model_name : str
        Physics model (for Hamiltonian construction when include_energy=True).
    max_points : int
        Maximum h-points to evaluate (uniformly sampled). Default 8.
    include_energy : bool
        If True, also evaluate E(θ_pred) via backend (slower but more accurate).

    Returns
    -------
    dict
        {
            "n_qubits": int,
            "n_points_evaluated": int,
            "theta_mse_mean": float,
            "theta_mse_max": float,
            "theta_mse_per_h": list[float],
            "de_gap_mean": float | None,
            "abs_error_mean": float | None,
            "metric_warnings": list[str],
        }
    """
    import torch

    from qmbp_simulation import make_lattice
    from qmbp_simulation.predictors.unified_graph import (
        build_unified_bond_resolved_graph,
    )

    npz_path = Path(npz_path)
    if not npz_path.exists():
        return {"error": f"NPZ not found: {npz_path}", "n_points_evaluated": 0}

    data = np.load(npz_path, allow_pickle=True)
    h_values = data["h_values"]
    theta_opt = data["theta_opt"]
    e_exact = data["e_exact"]

    # Parse N and p from filename: {topo}_N{n}_p{p}.npz
    import re as _re

    n_match = _re.search(r"_N(\d+)", npz_path.stem)
    if n_match is None:
        return {"error": f"Cannot parse N from {npz_path.name}", "n_points_evaluated": 0}
    n_qubits = int(n_match.group(1))

    # Guard against p-mismatch: the NPZ p (from filename) must match the p used
    # to build the graph. Mixing p=1 data with a p=2 graph is silently wrong.
    p_match = _re.search(r"_p(\d+)", npz_path.stem)
    if p_match is not None:
        npz_p = int(p_match.group(1))
        if npz_p != p_layers:
            return {
                "error": (
                    f"p mismatch: NPZ '{npz_path.name}' is p={npz_p} but "
                    f"evaluate_theta_prediction was called with p_layers={p_layers}. "
                    f"Never mix p across data and graph construction."
                ),
                "n_points_evaluated": 0,
            }

    # Uniformly sample h-points
    n_available = len(h_values)
    indices = np.linspace(0, n_available - 1, min(max_points, n_available), dtype=int)

    # lat_ref defines the graph STRUCTURE (edges/topology) with a reference h.
    # The actual h-value for node features is passed to build_unified_bond_resolved_graph.
    lat_ref = make_lattice(topology, n_qubits, J=1.0, h=2.0)
    mse_list = []
    energy_errors = []
    de_gap_list = []

    for idx in indices:
        h = float(h_values[idx])
        theta_true = theta_opt[idx]
        if not hasattr(theta_true, "__len__"):
            continue
        theta_true = np.asarray(theta_true, dtype=np.float64)
        n_params = len(theta_true)

        # MPNN forward pass
        g = build_unified_bond_resolved_graph(
            lat_ref,
            h_value=h,
            p_layers=p_layers,
            include_circuit_nodes=True,
        )
        with torch.no_grad():
            theta_pred = model(g).numpy().flatten()

        # Dimension matching
        if len(theta_pred) < n_params:
            theta_pred = np.pad(theta_pred, (0, n_params - len(theta_pred)))
        elif len(theta_pred) > n_params:
            theta_pred = theta_pred[:n_params]

        mse = float(np.mean((theta_pred - theta_true) ** 2))
        mse_list.append(mse)

        # Energy metrics from NPZ (no extra compute)
        e_key = "e_vqe" if "e_vqe" in data else ("e_pred" if "e_pred" in data else None)
        if e_key:
            e_vqe_val = float(data[e_key][idx])
            e_exact_val = float(e_exact[idx])
            abs_err = abs(e_vqe_val - e_exact_val)
            energy_errors.append(abs_err / n_qubits)
            if "gaps" in data:
                gap_val = float(data["gaps"][idx])
                if gap_val > 1e-6:
                    de_gap_list.append(abs_err / gap_val)

        # Optional: evaluate E(θ_pred) directly (expensive)
        if include_energy:
            from qmbp_simulation.circuits import HVACircuitBuilder
            from qmbp_simulation.execution import select_backend
            from qmbp_simulation.models.model_registry import get_model_spec

            spec = get_model_spec(model_name)
            hva = HVACircuitBuilder()
            lat_h = make_lattice(topology, n_qubits, J=1.0, h=h)
            H = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)
            circuit, _ = hva.create_bond_resolved(n_qubits, p_layers, lat_ref)
            backend = select_backend(n_qubits, for_vqe_loop=True)
            e_pred_eval = backend.evaluate(circuit, H, theta_pred)
            abs_err_pred = abs(e_pred_eval - float(e_exact[idx]))
            if energy_errors:
                energy_errors[-1] = abs_err_pred / n_qubits

    if not mse_list:
        return {"n_qubits": n_qubits, "n_points_evaluated": 0, "error": "no valid points"}

    result = {
        "n_qubits": n_qubits,
        "n_points_evaluated": len(mse_list),
        "theta_mse_mean": float(np.mean(mse_list)),
        "theta_mse_max": float(np.max(mse_list)),
        "theta_mse_per_h": mse_list,
        "de_gap_mean": float(np.mean(de_gap_list)) if de_gap_list else None,
        "abs_error_mean": (float(np.mean(energy_errors)) * n_qubits if energy_errors else None),
    }

    # Validate with shared validator
    if de_gap_list:
        _per_h = [
            {"de_gap": dg, "abs_error": ae * n_qubits if ae else None, "gap": 1.0}
            for dg, ae in zip(de_gap_list, energy_errors or [None] * len(de_gap_list), strict=False)
        ]
        result["metric_warnings"] = validate_metrics(_per_h, n_qubits=n_qubits)
    else:
        result["metric_warnings"] = []

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MT vs ST Head-to-Head Table
# ═══════════════════════════════════════════════════════════════════════════════


def generate_mt_vs_st_table(
    comparison_dir: Path | str | None = None,
    *,
    output_path: Path | str | None = None,
    topology_filter: str | list[str] | None = None,
    n_min: int | None = None,
    n_max: int | None = None,
    latest_only: bool = True,
) -> tuple[list[str], dict]:
    """Generate MT vs ST head-to-head comparison table from comparison JSONs.

    Reads all model_comparison JSON files, identifies MT and ST models,
    and computes which wins per (topology, N). Produces markdown lines +
    structured summary dict with per-N granularity.

    Parameters
    ----------
    comparison_dir : Path | str | None
        Directory with compare_*.json files. Default: results/model_comparison/
    output_path : Path | str | None
        If provided, writes markdown to this file. If None, only returns lines.
    topology_filter : str | list[str] | None
        If set, only include these topologies. Can be a single string or list.
    n_min : int | None
        If set, only include N values >= n_min.
    n_max : int | None
        If set, only include N values <= n_max.
    latest_only : bool
        If True (default), only use the latest comparison file per topology.
        If False, aggregate across all runs.

    Returns
    -------
    tuple[list[str], dict]
        (markdown_lines, summary) where summary has:
        {
            "mt_wins": int,
            "st_wins": int,
            "ties": int,
            "total": int,
            "mt_win_rate": float,
            "mt_avg_pass_rate": float,
            "st_avg_pass_rate": float,
            "per_topology": dict[str, dict],  # per-topology breakdown
            "per_scenario": list[dict],  # per-row details
            "generated_at": str,
        }

    Usage
    -----
    >>> from qmbp_simulation.analysis.evaluation_report import generate_mt_vs_st_table
    >>> lines, summary = generate_mt_vs_st_table()
    >>> # Filter by topology:
    >>> lines, summary = generate_mt_vs_st_table(topology_filter="chain_1d")
    >>> # Filter by N range:
    >>> lines, summary = generate_mt_vs_st_table(n_min=10, n_max=20)
    >>> # Combine filters:
    >>> lines, summary = generate_mt_vs_st_table(
    ...     topology_filter=["chain_1d", "heavy_hex"], n_min=16
    ... )
    """
    import json as _json

    if comparison_dir is None:
        comparison_dir = Path(__file__).resolve().parents[3] / "results" / "model_comparison"
    else:
        comparison_dir = Path(comparison_dir)

    if not comparison_dir.exists():
        return ["*No comparison data found.*"], {"mt_wins": 0, "st_wins": 0, "total": 0}

    # Normalize topology filter
    if isinstance(topology_filter, str):
        topology_filter = [topology_filter]

    # ── Load comparison JSONs (latest per topology if latest_only) ─────────
    all_jsons: list[tuple[Path, str]] = []  # (path, topology)
    for f in sorted(comparison_dir.glob("compare_*.json")):
        try:
            d = _json.loads(f.read_text())
            topo = d.get("topology", "?")
            all_jsons.append((f, topo))
        except Exception:
            continue

    # Filter by topology and select latest
    files_by_topo: dict[str, list[Path]] = {}
    for fpath, topo in all_jsons:
        if topology_filter and topo not in topology_filter:
            continue
        files_by_topo.setdefault(topo, []).append(fpath)

    files_to_analyze: list[Path] = []
    for topo, files in files_by_topo.items():
        if latest_only:
            files_to_analyze.append(files[-1])  # Already sorted by name (timestamp)
        else:
            files_to_analyze.extend(files)

    # ── Parse results with per-N granularity ──────────────────────────────
    # Structure: per_topo_n[topo][n] = {"mt": [...], "st": [...]}
    per_topo_n: dict[str, dict[int, dict[str, list]]] = {}

    for f in files_to_analyze:
        try:
            d = _json.loads(f.read_text())
            topo = d.get("topology", "?")
            if topology_filter and topo not in topology_filter:
                continue

            for r in d.get("results", []):
                if r.get("error") or "results_by_n" not in r:
                    continue
                label = r.get("label", "?")[:42]
                source = r.get("source", "")
                is_mt = (
                    "multi" in source.lower()
                    or "orphan" in source.lower()
                    or "multitopo" in label.lower()
                )
                arch = r.get("arch", "baseline")

                for n_str, metrics in r.get("results_by_n", {}).items():
                    if not isinstance(metrics, dict):
                        continue
                    try:
                        n_val = int(n_str)
                    except (ValueError, TypeError):
                        continue

                    # Apply N filters
                    if n_min is not None and n_val < n_min:
                        continue
                    if n_max is not None and n_val > n_max:
                        continue

                    entry = {
                        "label": label,
                        "arch": arch,
                        "mean_de_gap": metrics.get("mean_de_gap", 1.0),
                        "pass_rate_dual": metrics.get("pass_rate_dual", 0.0),
                        "pass_rate_5pct": metrics.get("pass_rate_5pct", 0.0),
                        "quality_score": metrics.get("quality_score", 0.0),
                        "grade": metrics.get("grade", "?"),
                    }

                    per_topo_n.setdefault(topo, {}).setdefault(n_val, {"mt": [], "st": []})
                    if is_mt:
                        per_topo_n[topo][n_val]["mt"].append(entry)
                    else:
                        per_topo_n[topo][n_val]["st"].append(entry)
        except Exception:
            continue

    if not per_topo_n:
        return ["*No valid comparison results found.*"], {
            "mt_wins": 0,
            "st_wins": 0,
            "ties": 0,
            "total": 0,
        }

    # ── Compute per-topology and per-N winners ────────────────────────────
    per_scenario: list[dict] = []
    per_topology: dict[str, dict] = {}
    mt_wins = 0
    st_wins = 0
    ties = 0
    all_mt_scores = []
    all_st_scores = []

    for topo in sorted(per_topo_n.keys()):
        topo_mt_scores = []
        topo_st_scores = []
        topo_mt_wins = 0
        topo_st_wins = 0
        topo_ties = 0

        for n_val in sorted(per_topo_n[topo].keys()):
            mt_entries = per_topo_n[topo][n_val]["mt"]
            st_entries = per_topo_n[topo][n_val]["st"]

            if not mt_entries and not st_entries:
                continue

            # Best MT and ST by quality_score (higher is better, continuous)
            best_mt = (
                max(mt_entries, key=lambda x: x.get("quality_score", 0.0)) if mt_entries else None
            )
            best_st = (
                max(st_entries, key=lambda x: x.get("quality_score", 0.0)) if st_entries else None
            )

            mt_qs = best_mt.get("quality_score", 0.0) if best_mt else 0.0
            st_qs = best_st.get("quality_score", 0.0) if best_st else 0.0
            mt_dg = best_mt["mean_de_gap"] if best_mt else float("inf")
            st_dg = best_st["mean_de_gap"] if best_st else float("inf")
            mt_pr = best_mt.get("pass_rate_dual", 0.0) if best_mt else 0.0
            st_pr = best_st.get("pass_rate_dual", 0.0) if best_st else 0.0

            topo_mt_scores.append(mt_qs)
            topo_st_scores.append(st_qs)
            all_mt_scores.append(mt_qs)
            all_st_scores.append(st_qs)

            # Winner: use quality_score (continuous), small tolerance for ties
            if mt_qs > st_qs + 0.03:
                winner = "MT"
                topo_mt_wins += 1
                mt_wins += 1
            elif st_qs > mt_qs + 0.03:
                winner = "ST"
                topo_st_wins += 1
                st_wins += 1
            else:
                winner = "tie"
                topo_ties += 1
                ties += 1

            per_scenario.append(
                {
                    "topology": topo,
                    "n_qubits": n_val,
                    "mt_quality_score": round(mt_qs, 3),
                    "st_quality_score": round(st_qs, 3),
                    "mt_pass_rate": mt_pr,
                    "st_pass_rate": st_pr,
                    "mt_mean_de_gap": round(mt_dg, 4),
                    "st_mean_de_gap": round(st_dg, 4),
                    "mt_grade": best_mt["grade"] if best_mt else "—",
                    "st_grade": best_st["grade"] if best_st else "—",
                    "mt_model": best_mt["label"] if best_mt else "—",
                    "st_model": best_st["label"] if best_st else "—",
                    "winner": winner,
                }
            )

        # Per-topology summary
        mt_avg = float(np.mean(topo_mt_scores)) if topo_mt_scores else 0.0
        st_avg = float(np.mean(topo_st_scores)) if topo_st_scores else 0.0
        per_topology[topo] = {
            "mt_avg_quality_score": mt_avg,
            "st_avg_quality_score": st_avg,
            "mt_avg_pass_rate": float(
                np.mean([s["mt_pass_rate"] for s in per_scenario if s["topology"] == topo])
            ),
            "st_avg_pass_rate": float(
                np.mean([s["st_pass_rate"] for s in per_scenario if s["topology"] == topo])
            ),
            "mt_wins": topo_mt_wins,
            "st_wins": topo_st_wins,
            "ties": topo_ties,
            "winner": "MT"
            if mt_avg > st_avg + 0.03
            else ("ST" if st_avg > mt_avg + 0.03 else "tie"),
            "delta": mt_avg - st_avg,
        }

    total = mt_wins + st_wins + ties
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    mt_avg_global = float(np.mean(all_mt_scores)) if all_mt_scores else 0.0
    st_avg_global = float(np.mean(all_st_scores)) if all_st_scores else 0.0

    # ── Build markdown ────────────────────────────────────────────────────
    filter_desc = []
    if topology_filter:
        filter_desc.append(f"topologies={topology_filter}")
    if n_min is not None:
        filter_desc.append(f"N≥{n_min}")
    if n_max is not None:
        filter_desc.append(f"N≤{n_max}")
    filter_str = f" (filter: {', '.join(filter_desc)})" if filter_desc else ""

    lines = [
        "# MT vs ST Head-to-Head Comparison",
        "",
        f"**Generated**: {ts}{filter_str}",
        f"**Score**: MT **{mt_wins}** — ST **{st_wins}** — Ties **{ties}**",
        f"**MT avg quality_score**: {mt_avg_global:.3f} | **ST avg quality_score**: {st_avg_global:.3f}",
        "",
    ]

    # Per-topology summary table
    lines.extend(
        [
            "## Per-Topology Summary",
            "",
            "| Topology | MT score | ST score | Winner | Δ | MT wins | ST wins |",
            "|----------|:--------:|:--------:|:------:|:-:|:-------:|:-------:|",
        ]
    )
    for topo, info in sorted(per_topology.items()):
        icon = "🟢" if info["winner"] == "MT" else ("🔴" if info["winner"] == "ST" else "⚪")
        lines.append(
            f"| {topo} | {info['mt_avg_quality_score']:.3f} | {info['st_avg_quality_score']:.3f} | "
            f"{icon} {info['winner']} | {info['delta']:+.3f} | "
            f"{info['mt_wins']} | {info['st_wins']} |"
        )

    # Detailed per-N table
    lines.extend(
        [
            "",
            "## Per-N Breakdown",
            "",
            "| Topology | N | MT score | MT ΔE/gap | MT grade | ST score | ST ΔE/gap | ST grade | Winner |",
            "|----------|:-:|:--------:|:---------:|:--------:|:--------:|:---------:|:--------:|:------:|",
        ]
    )
    for s in per_scenario:
        icon = "✅" if s["winner"] == "MT" else ("❌" if s["winner"] == "ST" else "—")
        mt_dg_str = f"{s['mt_mean_de_gap']:.1%}" if s["mt_mean_de_gap"] < 10 else "—"
        st_dg_str = f"{s['st_mean_de_gap']:.1%}" if s["st_mean_de_gap"] < 10 else "—"
        lines.append(
            f"| {s['topology']} | {s['n_qubits']} | "
            f"{s['mt_quality_score']:.3f} | {mt_dg_str} | {s['mt_grade']} | "
            f"{s['st_quality_score']:.3f} | {st_dg_str} | {s['st_grade']} | "
            f"{icon} {s['winner']} |"
        )

    lines.extend(
        [
            "",
            "---",
            f"*Auto-generated from {comparison_dir.name}/ ({len(per_scenario)} comparisons)*",
            "*Decision metric: quality_score (continuous 0-1, sigmoid-based on mean ΔE/gap + P90 )*",
        ]
    )

    # Write to file if requested
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines))
        logger.info("  MT vs ST table saved: %s", output_path)

    summary = {
        "mt_wins": mt_wins,
        "st_wins": st_wins,
        "ties": ties,
        "total": total,
        "mt_win_rate": mt_wins / max(total, 1),
        "mt_avg_quality_score": mt_avg_global,
        "st_avg_quality_score": st_avg_global,
        "mt_avg_pass_rate": float(np.mean([s["mt_pass_rate"] for s in per_scenario]))
        if per_scenario
        else 0.0,
        "st_avg_pass_rate": float(np.mean([s["st_pass_rate"] for s in per_scenario]))
        if per_scenario
        else 0.0,
        "per_topology": per_topology,
        "per_scenario": per_scenario,
        "generated_at": ts,
        "decision_metric": "quality_score",
    }

    return lines, summary
