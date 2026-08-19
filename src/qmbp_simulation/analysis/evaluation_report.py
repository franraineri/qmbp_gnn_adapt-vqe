"""Evaluation Report Generator — Markdown per-point comparison reports.

Generates structured evaluation reports comparing MPNN predictions vs
ground truth and optional VQE baselines. Reports include:
- Quality profile (continuous score, grade, distribution)
- Per-h error breakdown with classification
- Metric reliability warnings
- MPNN vs VQE comparison table (if baseline available)

Designed to be called by any runner that produces per_h_results.
Decoupled from runner internals — operates on standardized dicts.

Usage:
    from qmbp_simulation.analysis.evaluation_report import (
        generate_evaluation_report,
        generate_comparison_table,
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
        - "mpnn": {"mean_de_gap": ..., "mean_abs_error_per_site": ...}
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

    # Save to subdirectory: {output_dir}/{topology}_p{p}/eval_{topology}[_MT]_{ts}.md
    subdir = output_dir / f"{topology}_p{p_layers}"
    subdir.mkdir(parents=True, exist_ok=True)
    mt_tag = "_MT" if is_multi_topology else ""
    report_path = subdir / f"eval_{topology}{mt_tag}_{ts}.md"

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
        lines.extend([
            "> **🌐 MULTI-TOPOLOGY MODEL** — This evaluation uses a model trained on "
            "multiple topologies simultaneously. Results reflect cross-topology "
            "transfer capability.",
            "",
        ])

    lines.extend([
        f"**Date**: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Model**: {checkpoint_display}",
        f"**Multi-topology**: {'YES' if is_multi_topology else 'no'}",
        f"**h-range**: [{h_range[0]}, {h_range[1]}] ({n_h_points} pts)",
        f"**Target N**: {target_n}",
        "",
    ])

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

        # Per-h table
        lines.append(
            "| h | E_pred | E_exact | |ΔE| | |ΔE|/N | gap | ΔE/gap | Category | Action | Note |"
        )
        lines.append(
            "|---|--------|---------|------|--------|-----|--------|----------|--------|------|"
        )

        for p in per_point:
            h = p.get("h", 0)
            e_pred = p.get("e_pred", 0)
            e_exact = p.get("e_exact", 0)
            abs_err = p.get("abs_error", abs(e_pred - e_exact))
            gap = p.get("gap", 0)
            de_gap = p.get("de_gap", 0)
            per_site = abs_err / max(n_target, 1)
            method = p.get("method", "mpnn")

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
                f"{abs_err:.4f} | {per_site:.2e} | {gap:.4f} | "
                f"{de_gap:.4f} | {cat_display} | {cls.action} | {note} |"
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
            "energy_per_site_mean": float | None,
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

    # Parse N from filename: {topo}_N{n}_p{p}.npz
    try:
        n_str = npz_path.stem.split("_N")[1].split("_")[0]
        n_qubits = int(n_str)
    except (IndexError, ValueError):
        return {"error": f"Cannot parse N from {npz_path.name}", "n_points_evaluated": 0}

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
        "energy_per_site_mean": float(np.mean(energy_errors)) if energy_errors else None,
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
