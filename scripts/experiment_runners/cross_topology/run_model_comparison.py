#!/usr/bin/env python3
"""Compare multiple MPNN checkpoints on identical evaluation conditions.

Loads N checkpoints (from zoo, disk, or auto-detect), evaluates each on the
same topology/N/h-grid, and produces a structured comparison report with:
- Per-model quality profile (score, grade, distribution)
- Per-h point classification (pass/moderate/severe/ansatz-limited)
- Markdown evaluation report (same format as run_large_n_extrapolation)
- Metric validation warnings
- Zoo pass_rate auto-update for the winner

Integrations (same stack as run_large_n_extrapolation):
- GroundTruthCache: 2-level caching (in-memory + disk-persistent)
- CachedBackend: transparent eval cache (shared across models)
- compute_deploy_summary: standardized metrics (dual criterion)
- classify_point_failure: per-point error classification
- generate_evaluation_report: markdown reports
- validate_metrics: metric reliability checks
- update_zoo_pass_rate: auto-update winner in zoo manifest

Usage:
    # Auto-detect all models for this topology and compare
    .venv/bin/python scripts/experiment_runners/cross_topology/run_model_comparison.py \
        --topology chain_1d --target-n 20

    # Compare specific checkpoints
    .venv/bin/python scripts/experiment_runners/cross_topology/run_model_comparison.py \
        --topology ladder --target-n 20 26 \
        --checkpoints path/to/model_a.pt path/to/model_b.pt

    # Promote the winner to zoo
    .venv/bin/python scripts/experiment_runners/cross_topology/run_model_comparison.py \
        --topology chain_1d --target-n 20 --auto-detect --promote-best
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from qmbp_simulation.analysis.evaluation_report import (
    generate_evaluation_report,
    validate_metrics,
)
from qmbp_simulation.analysis.metrics import (
    classify_point_failure,
    compute_deploy_summary,
)
from qmbp_simulation.circuits import HVACircuitBuilder
from qmbp_simulation.execution.backends import select_backend
from qmbp_simulation.models.hamiltonian import make_lattice
from qmbp_simulation.models.model_registry import get_model_spec
from qmbp_simulation.predictors.unified_graph import build_unified_bond_resolved_graph
from qmbp_simulation.predictors.unified_mpnn import load_unified_checkpoint
from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

logger = logging.getLogger(__name__)

ZOO_CHECKPOINTS = ROOT / "data" / "model_zoo" / "checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MPNN checkpoints on identical conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--topology", required=True, help="Evaluation topology")
    parser.add_argument("--target-n", type=int, nargs="+", required=True,
                        help="Target N values for evaluation")
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="Explicit checkpoint paths to compare")
    parser.add_argument("--auto-detect", action="store_true",
                        help="Auto-detect: per-topo zoo model + all multi-topo models")
    parser.add_argument("--p-layers", type=int, default=1)
    parser.add_argument("--h-min", type=float, default=2.5)
    parser.add_argument("--h-max", type=float, default=5.0)
    parser.add_argument("--h-points", type=int, default=6)
    parser.add_argument("--model-name", type=str, default="tfim_bond_resolved")
    parser.add_argument("--promote-best", action="store_true",
                        help="Update zoo pass_rate for the best model")
    parser.add_argument("--include-versions", action="store_true",
                        help="Include _best/ and _versions/ checkpoints for historical comparison")
    parser.add_argument("--save-report", action="store_true", default=True,
                        help="Generate per-model markdown evaluation reports")
    parser.add_argument("--no-report", dest="save_report", action="store_false")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def discover_checkpoints(
    topology: str, p_layers: int, explicit: list[str] | None,
    include_versions: bool = False,
) -> list[dict]:
    """Discover checkpoints to compare.

    Parameters
    ----------
    topology : str
        Target topology for evaluation.
    p_layers : int
        HVA depth.
    explicit : list[str] | None
        Explicit checkpoint paths (bypasses auto-detect if provided).
    include_versions : bool
        If True, also include checkpoints from _best/ and _versions/
        directories for historical comparison.
    """
    candidates = []

    if explicit:
        for path in explicit:
            p = Path(path)
            if not p.exists():
                p = ZOO_CHECKPOINTS / path
            if p.exists():
                candidates.append({"path": p, "label": p.stem[:45], "source": "explicit"})
        return candidates

    # Auto-detect from zoo manifest
    from qmbp_simulation.predictors.model_zoo import _load_manifest
    entries = _load_manifest()

    # Per-topology multi-N model
    per_topo = [
        e for e in entries
        if e.topology == topology and e.n_qubits == 0 and e.p_layers == p_layers
    ]
    for e in per_topo:
        cp = ZOO_CHECKPOINTS / e.checkpoint_file
        if cp.exists():
            candidates.append({
                "path": cp,
                "label": f"per-topo ({e.checkpoint_file[:40]})",
                "source": "zoo_per_topology",
                "zoo_entry": e,
            })

    # Multi-topology models
    multi_topo = [
        e for e in entries
        if e.topology == "multi_topology" and e.p_layers == p_layers
    ]
    for e in multi_topo:
        cp = ZOO_CHECKPOINTS / e.checkpoint_file
        if cp.exists():
            candidates.append({
                "path": cp,
                "label": f"multi-topo ({e.checkpoint_file[:40]})",
                "source": "zoo_multi_topology",
                "zoo_entry": e,
            })

    # Orphan multi-topo on disk (not in manifest)
    manifest_files = {e.checkpoint_file for e in entries}
    for f in sorted(ZOO_CHECKPOINTS.glob("*multitopo*")):
        if f.name not in manifest_files:
            candidates.append({
                "path": f,
                "label": f"orphan ({f.stem[:40]})",
                "source": "disk_orphan",
            })

    # ── Historical versions from _best/ and _versions/ ──────────────────
    if include_versions:
        best_dir = ZOO_CHECKPOINTS / "_best"
        versions_dir = ZOO_CHECKPOINTS / "_versions"

        # _best/: checkpoints tagged with pass_rate at their peak
        if best_dir.exists():
            pattern = f"*{topology}*p{p_layers}*.pt"
            for f in sorted(best_dir.glob(pattern))[-3:]:  # Last 3 best
                candidates.append({
                    "path": f,
                    "label": f"_best/{f.stem[-40:]}",
                    "source": "historical_best",
                })

        # _versions/: numbered versions (v1, v2, v3...)
        if versions_dir.exists():
            pattern = f"*{topology}*p{p_layers}*_v*.pt"
            for f in sorted(versions_dir.glob(pattern))[-3:]:  # Last 3 versions
                candidates.append({
                    "path": f,
                    "label": f"_v/{f.stem[-40:]}",
                    "source": "historical_version",
                })

    return candidates


def evaluate_checkpoint(
    checkpoint_path: Path,
    topology: str,
    target_ns: list[int],
    h_values: list[float],
    p_layers: int,
    model_name: str,
    gt_cache: GroundTruthCache,
    gt_memory: dict,
) -> dict:
    """Evaluate a single checkpoint on all (N, h) with full diagnostics."""
    model = load_unified_checkpoint(str(checkpoint_path), eval_mode=True)
    spec = get_model_spec(model_name)
    hva = HVACircuitBuilder()
    n_params_total = sum(p.numel() for p in model.parameters())

    results_by_n = {}

    for n_target in target_ns:
        lat_ref = make_lattice(topology, n_target, J=1.0, h=2.0)
        circuit, _ = hva.create_bond_resolved(n_target, p_layers, lat_ref)
        n_params = circuit.num_parameters
        backend = select_backend(n_target)

        per_h = []
        for h in h_values:
            # Predict θ
            g = build_unified_bond_resolved_graph(
                lat_ref, h_value=float(h), p_layers=p_layers,
                include_circuit_nodes=True,
            )
            with torch.no_grad():
                theta_pred = model(g).numpy().flatten()
            theta_pred = np.clip(theta_pred, -np.pi, np.pi)

            # MC-Dropout uncertainty estimation
            theta_std = 0.0
            if hasattr(model, "predict_with_uncertainty"):
                _, theta_std = model.predict_with_uncertainty(g)
            if len(theta_pred) != n_params:
                if len(theta_pred) < n_params:
                    theta_pred = np.pad(theta_pred, (0, n_params - len(theta_pred)))
                else:
                    theta_pred = theta_pred[:n_params]

            # Evaluate energy
            lat_h = make_lattice(topology, n_target, J=1.0, h=float(h))
            H = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)
            e_pred = backend.evaluate(circuit, H, theta_pred)

            # Ground truth (2-level: memory dict + disk GroundTruthCache)
            cache_key = (topology, n_target, round(float(h), 6))
            if cache_key in gt_memory:
                e_exact, gap = gt_memory[cache_key]
            else:
                cached = gt_cache.get(topology, n_target, model_name, float(h))
                if cached:
                    e_exact, gap = cached["energy"], cached["gap"]
                else:
                    from qmbp_simulation.solvers.classical import ClassicalSolver
                    solver = ClassicalSolver()
                    gt_obj = solver.solve(H, lat_h)
                    e_exact, gap = gt_obj.ground_energy, gt_obj.gap
                    gt_cache.put(topology, n_target, model_name, float(h), energy=e_exact, gap=gap)
                gt_memory[cache_key] = (e_exact, gap)

            de_gap = abs(e_pred - e_exact) / max(gap, 1e-10)
            abs_error = abs(e_pred - e_exact)

            # Per-point classification (same as run_large_n_extrapolation)
            cls = classify_point_failure(
                de_gap=de_gap, abs_error=abs_error, gap=gap,
                h=float(h), h_critical=1.0, n_params=n_params,
            )

            per_h.append({
                "h": float(h),
                "e_pred": float(e_pred),
                "e_exact": float(e_exact),
                "gap": float(gap),
                "de_gap": float(de_gap),
                "abs_error": float(abs_error),
                "n_qubits": n_target,
                "theta_std": theta_std,
                "category": cls.category,
                "action": cls.action,
                "theta": theta_pred.tolist(),
            })

        # Compute summary (same as run_large_n_extrapolation)
        summary = compute_deploy_summary(per_h)
        # Validate metrics
        warnings = validate_metrics(per_h, n_qubits=n_target)

        # Uncertainty calibration (requires θ_std > 0)
        from qmbp_simulation.analysis.metrics import compute_uncertainty_correlation
        uc_report = compute_uncertainty_correlation(per_h)

        results_by_n[n_target] = {
            **summary,
            "n_params": n_params,
            "metric_warnings": warnings,
            "uncertainty_calibration": uc_report if uc_report["n_points_with_uncertainty"] >= 3 else None,
            "per_point": per_h,
        }

    return {
        "n_model_params": n_params_total,
        "use_residual": getattr(model, "use_residual", False),
        "readout_mode": getattr(model, "readout_mode", "last"),
        "film_conditioning": getattr(model, "film_conditioning", False),
        "results_by_n": results_by_n,
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-5s %(message)s",
    )

    h_values = [round(h, 2) for h in np.linspace(args.h_min, args.h_max, args.h_points)]

    print("=" * 75)
    print("  MODEL COMPARISON")
    print(f"  Topology: {args.topology} | N: {args.target_n} | h: {h_values}")
    print("=" * 75)

    # Discover checkpoints
    candidates = discover_checkpoints(
        args.topology, args.p_layers,
        args.checkpoints if not args.auto_detect else None,
        include_versions=args.include_versions,
    )
    if not candidates:
        print("ERROR: No checkpoints found to compare.")
        return 1

    print(f"\n  Checkpoints to compare: {len(candidates)}")
    for c in candidates:
        print(f"    [{c['source']:20}] {c['label']}")

    # Show selection scores for context (explain_model_selection)
    try:
        from qmbp_simulation.predictors.model_zoo import explain_model_selection

        selection_scores = explain_model_selection(
            args.topology,
            model=args.model_name,
            p_layers=args.p_layers,
            n_target=args.target_n[0],
        )
        if selection_scores:
            print("\n  Zoo selection scores (for reference):")
            for s in selection_scores[:5]:
                marker = "→" if s["selected"] else " "
                print(
                    f"   {marker} {s['checkpoint'][:45]:45} "
                    f"score={s['final_score']:.3f} "
                    f"(pass={s['pass_rate']:.0%}, src={s['source']}, pts={s['n_training_points']})"
                )
    except Exception as e:
        logger.debug(f"  explain_model_selection unavailable: {e}")

    # Shared ground truth (computed once, reused for all models)
    gt_cache = GroundTruthCache()
    gt_memory: dict = {}

    # Evaluate each
    print(f"\n{'─' * 75}")
    print(f"  {'Model':<42} | {'N':>3} | {'ΔE/gap':>8} | {'|ΔE|/N':>10} | {'Pass%':>6} | {'Grade':>5}")
    print(f"{'─' * 75}")

    all_results = []
    for candidate in candidates:
        t0 = time.perf_counter()
        try:
            result = evaluate_checkpoint(
                candidate["path"], args.topology, args.target_n,
                h_values, args.p_layers, args.model_name, gt_cache, gt_memory,
            )
            elapsed = time.perf_counter() - t0

            # Print per-N results
            for n_target, metrics in result["results_by_n"].items():
                per_site = metrics.get("mean_abs_error_per_site", 0) or 0
                print(
                    f"  {candidate['label']:<42} | {n_target:>3} | "
                    f"{metrics['mean_de_gap']:>8.4f} | {per_site:>10.2e} | "
                    f"{metrics['pass_rate_dual']:>5.0%} | {metrics.get('grade', '?'):>5}"
                )
                # Log warnings
                for w in metrics.get("metric_warnings", []):
                    logger.warning(f"    {w}")

            # Architecture info
            arch_parts = []
            if result.get("use_residual"):
                arch_parts.append("residual")
            if result.get("readout_mode", "last") != "last":
                arch_parts.append(result["readout_mode"])
            if result.get("film_conditioning"):
                arch_parts.append("film")
            arch_label = "+".join(arch_parts) if arch_parts else "baseline"

            all_results.append({
                "label": candidate["label"],
                "source": candidate["source"],
                "checkpoint": str(candidate["path"].name),
                "arch": arch_label,
                "elapsed_s": round(elapsed, 1),
                **result,
            })
        except Exception as e:
            print(f"  {candidate['label']:<42} | ERROR: {e}")
            all_results.append({
                "label": candidate["label"],
                "source": candidate["source"],
                "checkpoint": str(candidate["path"].name),
                "error": str(e),
            })

    # Flush GT cache (persist any new computations)
    gt_cache.flush()

    # Determine winner
    print(f"\n{'─' * 75}")
    scoreable = [r for r in all_results if "results_by_n" in r and not r.get("error")]

    best = None
    if scoreable:
        def avg_pass_rate(r):
            rates = [m["pass_rate_dual"] for m in r["results_by_n"].values()]
            return np.mean(rates) if rates else 0

        best = max(scoreable, key=avg_pass_rate)
        print(f"\n  🏆 BEST: {best['label']} (arch={best['arch']}, avg_pass={avg_pass_rate(best):.0%})")

        # Promote if requested
        if args.promote_best:
            from qmbp_simulation.predictors.model_zoo import update_zoo_pass_rate
            ckpt_file = best["checkpoint"]
            rate = float(avg_pass_rate(best))
            updated = update_zoo_pass_rate(
                ckpt_file, rate, only_if_better=True,
                _skip_db_sync=True,  # We write a richer record below
                add_notes=f"comparison@{args.topology} N={args.target_n}",
            )
            if updated:
                print(f"  📊 Zoo updated: {ckpt_file} → {rate:.0%}")

            # Persist rich evaluation to ModelRegistryDB
            try:
                from qmbp_simulation.predictors.model_registry_db import (
                    EvaluationRecord,
                    ModelRegistryDB,
                )
                # Aggregate per-N metrics from the best model
                best_results_by_n = best.get("results_by_n", {})
                all_de_gaps = [m["mean_de_gap"] for m in best_results_by_n.values()]
                all_abs_per_site = [
                    m.get("mean_abs_error_per_site", 0) for m in best_results_by_n.values()
                ]
                eval_record = EvaluationRecord(
                    evaluated_at=datetime.now(timezone.utc).isoformat(),
                    target_n_values=args.target_n,
                    pass_rate_5pct=float(np.mean([
                        m.get("pass_rate_5pct", 0) for m in best_results_by_n.values()
                    ])) if best_results_by_n else 0.0,
                    pass_rate_dual=rate,
                    mean_de_gap=float(np.mean(all_de_gaps)) if all_de_gaps else 0.0,
                    mean_abs_error_per_site=float(np.mean(all_abs_per_site)) if all_abs_per_site else 0.0,
                    notes=f"model_comparison winner @{args.topology} N={args.target_n}",
                )
                db = ModelRegistryDB()
                db.add_evaluation(ckpt_file, eval_record)
            except Exception:
                pass  # Non-critical: registry_db may not have this model

    # Generate per-model evaluation reports (markdown)
    if args.save_report:
        for r in scoreable:
            try:
                # Build mpnn_results_by_n in the format expected by generate_evaluation_report
                mpnn_results = {}
                for n_target, metrics in r["results_by_n"].items():
                    mpnn_results[n_target] = {
                        "per_point": metrics["per_point"],
                        "n_params": metrics["n_params"],
                        **{k: v for k, v in metrics.items() if k not in ("per_point", "metric_warnings")},
                    }

                generate_evaluation_report(
                    mpnn_results_by_n=mpnn_results,
                    topology=args.topology,
                    model_name=args.model_name,
                    checkpoint=r["checkpoint"],
                    h_range=(args.h_min, args.h_max),
                    n_h_points=args.h_points,
                    p_layers=args.p_layers,
                    target_n=args.target_n,
                    output_dir="results/model_comparison",
                )
            except Exception as e:
                logger.debug(f"  Report generation failed for {r['label']}: {e}")

    # Save JSON report
    output_dir = ROOT / "results" / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"compare_{args.topology}_{timestamp}.json"

    # Strip per_point from JSON (too verbose)
    results_for_json = []
    for r in all_results:
        r_copy = {k: v for k, v in r.items()}
        if "results_by_n" in r_copy:
            r_copy["results_by_n"] = {
                str(n): {k: v for k, v in data.items() if k != "per_point"}
                for n, data in r_copy["results_by_n"].items()
            }
        results_for_json.append(r_copy)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "topology": args.topology,
        "target_n": args.target_n,
        "h_values": h_values,
        "p_layers": args.p_layers,
        "model_name": args.model_name,
        "n_models": len(all_results),
        "results": results_for_json,
        "best_model": best["label"] if best else None,
        "best_arch": best["arch"] if best else None,
    }
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Results: {output_file.relative_to(ROOT)}")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
