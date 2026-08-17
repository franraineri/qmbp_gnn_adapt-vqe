#!/usr/bin/env python3
"""Evaluate all zoo multi-N models: θ_pred quality + energy accuracy.

Loads each registered multi-N model from the zoo, predicts θ on training NPZ
h-points, and computes MSE vs θ_opt + energy error via MPS evaluation.

Outputs:
- Console summary table
- Markdown report auto-saved to results/model_evaluation_report.md

Usage:
    .venv/bin/python scripts/analysis/evaluate_zoo_models.py
    .venv/bin/python scripts/analysis/evaluate_zoo_models.py --max-pts 5
    .venv/bin/python scripts/analysis/evaluate_zoo_models.py --include-archived
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from qmbp_simulation.analysis.constants import compute_quality_score, grade_from_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

REPORT_PATH = ROOT / "results" / "model_evaluation_report.md"
NPZ_DIR = ROOT / "data" / "multi_n_training"
EXTRAP_DIR = ROOT / "data" / "large_n_extrapolation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate zoo models on training data")
    parser.add_argument(
        "--max-pts", type=int, default=8, help="Max h-points to evaluate per N (default: 8)"
    )
    parser.add_argument(
        "--include-archived", action="store_true", help="Also evaluate archived checkpoints"
    )
    parser.add_argument(
        "--energy-eval",
        action="store_true",
        help="Evaluate energy via MPS (slower but more informative)",
    )
    parser.add_argument("--topology", type=str, default=None, help="Evaluate only this topology")
    parser.add_argument(
        "--update-zoo",
        action="store_true",
        help="Auto-update zoo pass_rate after evaluation (fixes zoo_never_evaluated)",
    )
    return parser.parse_args()


def evaluate_model(ckpt_path: Path, topology: str, max_pts: int, energy_eval: bool = False) -> dict:
    """Evaluate a single model checkpoint on available NPZ data.

    Delegates θ MSE computation to the reusable evaluate_theta_prediction().
    Returns dict with per-N results including θ MSE and optionally energy error.
    """
    from qmbp_simulation.analysis.evaluation_report import evaluate_theta_prediction
    from qmbp_simulation.predictors.unified_mpnn import load_unified_checkpoint

    model = load_unified_checkpoint(str(ckpt_path))
    model.eval()

    results_per_n = []

    # Evaluate on training NPZ (in-distribution)
    for npz_file in sorted(NPZ_DIR.glob(f"{topology}_N*_p1.npz")):
        result = evaluate_theta_prediction(
            model,
            npz_file,
            topology,
            p_layers=1,
            model_name="tfim_bond_resolved",
            max_points=max_pts,
            include_energy=energy_eval,
        )
        if result.get("n_points_evaluated", 0) == 0:
            continue

        entry = {
            "n": result["n_qubits"],
            "n_pts_evaluated": result["n_points_evaluated"],
            "theta_mse_mean": result["theta_mse_mean"],
            "theta_mse_max": result["theta_mse_max"],
            "source": "training_npz",
        }
        if result.get("energy_per_site_mean") is not None:
            entry["energy_per_site_mean"] = result["energy_per_site_mean"]
            entry["energy_per_site_max"] = result["theta_mse_max"]  # Approx
            entry["abs_error_mean"] = result["abs_error_mean"]
        if result.get("de_gap_mean") is not None:
            entry["de_gap_mean"] = result["de_gap_mean"]
        if result.get("metric_warnings"):
            entry["metric_warnings"] = result["metric_warnings"]

        results_per_n.append(entry)

    # Also check extrapolation NPZ (out-of-distribution, summary only)
    for npz_file in sorted(EXTRAP_DIR.glob(f"{topology}_N*_p1.npz")):
        n_str = npz_file.stem.split("_N")[1].split("_")[0]
        n_test = int(n_str)

        # Skip if already evaluated from training
        if any(r["n"] == n_test and r["source"] == "training_npz" for r in results_per_n):
            continue

        data = np.load(npz_file, allow_pickle=True)
        h_values = data["h_values"]
        e_key = "e_pred" if "e_pred" in data else ("e_vqe" if "e_vqe" in data else None)
        if e_key is None:
            continue
        e_exact = data["e_exact"]
        gaps = data["gaps"] if "gaps" in data else None

        # Compute per-site error from existing predictions (no re-evaluation needed)
        abs_errs = np.abs(data[e_key] - e_exact)
        per_site = float(abs_errs.mean()) / n_test

        de_gap = None
        if gaps is not None:
            de_gaps = abs_errs / np.maximum(gaps, 1e-10)
            de_gap = float(de_gaps.mean())

        results_per_n.append(
            {
                "n": n_test,
                "n_pts_evaluated": len(h_values),
                "energy_per_site_mean": per_site,
                "de_gap_mean": de_gap,
                "source": "extrapolation_npz",
            }
        )

    return {
        "checkpoint": ckpt_path.name,
        "topology": topology,
        "results_per_n": sorted(results_per_n, key=lambda x: x["n"]),
    }


def format_markdown_report(all_results: list[dict], elapsed_s: float) -> str:
    """Format evaluation results as a markdown report."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Zoo Model Evaluation Report",
        "",
        f"**Generated**: {now}",
        f"**Elapsed**: {elapsed_s:.1f}s",
        f"**Models evaluated**: {len(all_results)}",
        "",
        "---",
        "",
    ]

    for result in all_results:
        topo = result["topology"]
        ckpt = result["checkpoint"]
        per_n = result["results_per_n"]

        lines.append(f"## {topo} — `{ckpt}`")
        lines.append("")

        if not per_n:
            lines.append("*No evaluation data available.*")
            lines.append("")
            continue

        # Always show all metrics columns
        lines.append("| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |")
        lines.append("|---|--------|-----|-------|------|--------|--------|-------|")

        for r in per_n:
            src = "IN" if r["source"] == "training_npz" else "EXT"
            mse = r.get("theta_mse_mean")
            mse_str = f"{mse:.4e}" if mse is not None else "—"
            abs_err = r.get("abs_error_mean")
            abs_str = f"{abs_err:.4f}" if abs_err is not None else "—"
            eps = r.get("energy_per_site_mean")
            eps_str = f"{eps:.2e}" if eps is not None else "—"
            dg = r.get("de_gap_mean")
            dg_str = f"{dg:.4f}" if dg is not None else "—"

            # Grade from continuous quality score
            if dg is not None and eps is not None:
                score = compute_quality_score(dg, dg * 1.5, eps, r["n_pts_evaluated"])
                grade = grade_from_score(score)
            elif dg is not None:
                score = compute_quality_score(dg, dg * 1.5, None, r["n_pts_evaluated"])
                grade = grade_from_score(score)
            else:
                grade = "?"

            lines.append(
                f"| {r['n']} | {src} | {r['n_pts_evaluated']} | "
                f"{mse_str} | {abs_str} | {eps_str} | {dg_str} | {grade} |"
            )

        # Show metric warnings if any
        all_warnings = [(r["n"], w) for r in per_n for w in r.get("metric_warnings", [])]
        if all_warnings:
            lines.append("")
            lines.append("> **Metric Warnings:**")
            for n_val, w in all_warnings:
                lines.append(f"> - {w}")

        lines.append("")

    # Summary ranking
    lines.append("---")
    lines.append("")
    lines.append("## Summary Ranking")
    lines.append("")
    lines.append("| Topology | Checkpoint | In-dist θ MSE | Out-dist |ΔE|/N | Grade |")
    lines.append("|----------|-----------|:---:|:---:|:---:|")

    for result in sorted(all_results, key=lambda r: _grade_score(r)):
        topo = result["topology"]
        ckpt = result["checkpoint"][:50]
        per_n = result["results_per_n"]

        in_dist = [r for r in per_n if r["source"] == "training_npz" and "theta_mse_mean" in r]
        out_dist = [
            r for r in per_n if r["source"] == "extrapolation_npz" and "energy_per_site_mean" in r
        ]

        mse_str = f"{np.mean([r['theta_mse_mean'] for r in in_dist]):.4e}" if in_dist else "—"
        eps_str = (
            f"{np.mean([r['energy_per_site_mean'] for r in out_dist]):.2e}" if out_dist else "—"
        )
        grade = _grade_label(result)

        lines.append(f"| {topo} | {ckpt} | {mse_str} | {eps_str} | {grade} |")

    lines.append("")
    lines.append("---")
    lines.append("*Generated by `scripts/analysis/evaluate_zoo_models.py`*")
    return "\n".join(lines)


def _grade_score(result: dict) -> float:
    """Compute a sortable quality score. Lower is better.

    Uses compute_quality_score from analysis.constants (single source of truth)
    rather than ad-hoc thresholds.
    """
    per_n = result["results_per_n"]
    # Energy-based (out-of-distribution) is the primary quality indicator
    out_entries = [r for r in per_n if r.get("energy_per_site_mean") is not None]
    in_entries = [r for r in per_n if r.get("theta_mse_mean") is not None]

    if out_entries:
        # Use compute_quality_score for consistent grading
        mean_de_gaps = [r.get("de_gap_mean", 0.5) for r in out_entries]
        mean_eps = [r.get("energy_per_site_mean", 0.1) for r in out_entries]
        avg_dg = float(np.mean(mean_de_gaps))
        avg_eps = float(np.mean(mean_eps))
        # Invert: compute_quality_score returns higher=better, we want lower=better for sort
        score = compute_quality_score(
            avg_dg, avg_dg * 1.5, avg_eps, sum(r["n_pts_evaluated"] for r in out_entries)
        )
        return 1.0 - score
    elif in_entries:
        # Fallback: θ MSE (less reliable — variational degeneracy inflates it)
        avg_mse = float(np.mean([r["theta_mse_mean"] for r in in_entries]))
        return min(avg_mse * 10, 1.0)  # Scale to [0,1]
    else:
        return 1.0


def _grade_label(result: dict) -> str:
    """Get letter grade for a model using the shared grade system."""
    score = 1.0 - _grade_score(result)  # Convert back to higher=better
    grade = grade_from_score(score)
    descriptors = {"A": "excellent", "B": "good", "C": "acceptable", "D": "poor", "F": "failing"}
    return f"{grade} ({descriptors.get(grade, 'unknown')})"


def main() -> int:
    args = parse_args()
    t_start = time.perf_counter()

    from qmbp_simulation.predictors.model_zoo import _CHECKPOINTS_DIR, _load_manifest

    print("=" * 80)
    print("ZOO MODEL EVALUATION")
    print("=" * 80)
    print(flush=True)

    entries = _load_manifest()
    multi_n = [e for e in entries if e.n_qubits == 0]

    if args.topology:
        multi_n = [e for e in multi_n if e.topology == args.topology]

    print(f"  Models to evaluate: {len(multi_n)}")
    for e in multi_n:
        print(f"    {e.topology:12} → {e.checkpoint_file}")
    print(flush=True)

    # Evaluate each model
    all_results = []
    for entry in multi_n:
        ckpt_path = _CHECKPOINTS_DIR / entry.checkpoint_file
        if not ckpt_path.exists():
            print(f"\n  ❌ SKIP {entry.topology}: {ckpt_path.name} not found")
            continue

        print(f"\n{'─' * 80}")
        print(f"  Evaluating: {entry.topology} — {entry.checkpoint_file}")
        print(f"  Training pts: {entry.n_training_points}, h-range: {entry.h_range}")
        print(f"{'─' * 80}", flush=True)

        try:
            result = evaluate_model(
                ckpt_path, entry.topology, args.max_pts, energy_eval=args.energy_eval
            )
            all_results.append(result)

            # Print per-N summary
            for r in result["results_per_n"]:
                src = "IN " if r["source"] == "training_npz" else "EXT"
                mse_str = f"θMSE={r['theta_mse_mean']:.4e}" if "theta_mse_mean" in r else ""
                eps_str = (
                    f"|ΔE|/N={r.get('energy_per_site_mean', 0):.2e}"
                    if "energy_per_site_mean" in r
                    else ""
                )
                print(
                    f"    N={r['n']:>3} [{src}] {r['n_pts_evaluated']:>2}pts {mse_str} {eps_str}",
                    flush=True,
                )

        except Exception as ex:
            print(f"    ❌ ERROR: {ex}", flush=True)
            all_results.append(
                {
                    "checkpoint": entry.checkpoint_file,
                    "topology": entry.topology,
                    "results_per_n": [],
                    "error": str(ex),
                }
            )

    # Include archived models if requested
    if args.include_archived:
        archived_dir = _CHECKPOINTS_DIR / "_archived"
        if archived_dir.exists():
            print(f"\n{'=' * 80}")
            print("ARCHIVED MODELS")
            print(f"{'=' * 80}", flush=True)
            for ckpt in sorted(archived_dir.glob("unified_tfim_br_*_multiN_*.pt")):
                # Parse topology from filename
                parts = ckpt.name.split("_multiN_")[0]
                topo = parts.replace("unified_tfim_br_", "")
                if args.topology and topo != args.topology:
                    continue
                print(f"\n  Evaluating archived: {ckpt.name}", flush=True)
                try:
                    result = evaluate_model(ckpt, topo, args.max_pts, energy_eval=args.energy_eval)
                    result["checkpoint"] = f"[ARCHIVED] {ckpt.name}"
                    all_results.append(result)
                    for r in result["results_per_n"]:
                        src = "IN " if r["source"] == "training_npz" else "EXT"
                        mse_str = f"θMSE={r['theta_mse_mean']:.4e}" if "theta_mse_mean" in r else ""
                        print(f"    N={r['n']:>3} [{src}] {mse_str}", flush=True)
                except Exception as ex:
                    print(f"    ❌ ERROR: {ex}", flush=True)

    elapsed = time.perf_counter() - t_start

    # Print final summary
    print(f"\n{'=' * 80}")
    print("FINAL RANKING")
    print(f"{'=' * 80}")
    for result in sorted(all_results, key=lambda r: _grade_score(r)):
        grade = _grade_label(result)
        print(f"  {grade:20} | {result['topology']:12} | {result['checkpoint'][:55]}")
    print(f"\n  Total time: {elapsed:.1f}s")
    print(flush=True)

    # Save markdown report
    report_content = format_markdown_report(all_results, elapsed)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_content)
    print(f"\n  ✅ Report saved: {REPORT_PATH.relative_to(ROOT)}")

    # ── Auto-update zoo pass_rate if --update-zoo ────────────────────────
    if args.update_zoo:
        from qmbp_simulation.analysis.metrics import DE_GAP_THRESHOLD
        from qmbp_simulation.predictors.model_zoo import update_zoo_pass_rate

        print(f"\n{'─' * 80}")
        print("  UPDATING ZOO PASS RATES")
        print(f"{'─' * 80}")

        n_updated = 0
        for result in all_results:
            if result.get("error"):
                continue

            ckpt = result["checkpoint"]
            per_n = result["results_per_n"]

            # Compute pass_rate from evaluation results:
            # Use de_gap_mean where available, fall back to energy_per_site_mean
            n_pass = 0
            n_total = 0
            for r in per_n:
                de_gap = r.get("de_gap_mean")
                if de_gap is not None:
                    n_total += 1
                    if de_gap < DE_GAP_THRESHOLD:
                        n_pass += 1

            if n_total == 0:
                continue

            observed_rate = n_pass / n_total
            updated = update_zoo_pass_rate(
                ckpt,
                observed_rate,
                only_if_better=False,  # Always update — this IS the evaluation
                add_notes=f"eval@{datetime.now(UTC).strftime('%Y%m%d')} "
                f"N={[r['n'] for r in per_n]} {n_pass}/{n_total}",
            )
            if updated:
                n_updated += 1
                print(
                    f"    ✅ {result['topology']:12} → {observed_rate:.0%} "
                    f"({n_pass}/{n_total} N-values pass)"
                )

        if n_updated == 0:
            print("    No updates needed")
        else:
            print(f"\n  Updated {n_updated} zoo entries")

    return 0


if __name__ == "__main__":
    sys.exit(main())
