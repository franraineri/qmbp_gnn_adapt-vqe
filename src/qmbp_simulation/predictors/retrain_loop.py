"""Automated retrain loop — closes the train→eval→compare→update_zoo cycle.

This module implements Tier 2 automation:
- Item 4: Auto-retrain when compute_retrain_queue() returns P3+ items
- Item 5: Auto-compare after retrain to measure impact
- Item 6: Regression guardrail (anti-regression gate) for register_checkpoint

The retrain loop is designed to be called from post_experiment_sync() as a
fire-and-forget subprocess, or manually via CLI.

Usage:
    # Programmatic (from post_experiment_sync)
    from qmbp_simulation.predictors.retrain_loop import run_retrain_loop
    result = run_retrain_loop(max_retrains=2, dry_run=False)

    # CLI
    python -m qmbp_simulation.predictors.retrain_loop --max-retrains 2 --verbose
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]


# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RetrainResult:
    """Result of a single model retrain attempt."""

    topology: str
    checkpoint_file: str
    priority: int
    reason: str
    action: str  # "retrained" | "compared" | "skipped" | "failed" | "blocked"
    old_pass_rate: float = 0.0
    new_pass_rate: float = 0.0
    comparison_winner: str = ""  # "new" | "existing" | "tie" | ""
    training_mse: float = 0.0
    n_training_points: int = 0
    elapsed_s: float = 0.0
    error: str = ""


@dataclass
class RetrainLoopResult:
    """Aggregate result of the full retrain loop."""

    n_candidates: int = 0
    n_retrained: int = 0
    n_improved: int = 0
    n_blocked: int = 0
    n_failed: int = 0
    total_elapsed_s: float = 0.0
    results: list[RetrainResult] = field(default_factory=list)
    dry_run: bool = False

    @property
    def summary(self) -> str:
        if self.dry_run:
            return f"[DRY-RUN] Retrain loop: {self.n_candidates} candidates found"
        return (
            f"Retrain loop: {self.n_retrained} retrained, "
            f"{self.n_improved} improved, {self.n_blocked} blocked, "
            f"{self.n_failed} failed ({self.total_elapsed_s:.0f}s)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Regression Guardrail
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_model_quick(
    model,
    topology: str,
    n_qubits: int = 10,
    p_layers: int = 1,
    n_h_points: int = 6,
    h_range: tuple[float, float] = (1.5, 4.0),
) -> float:
    """Quick evaluation of a model to get pass_rate without a full runner.

    Uses the same metrics stack (compute_deploy_summary) as the runners,
    but in a lightweight inline evaluation. Returns pass_rate_dual.

    Handles both MPNNPredictor (simple graphs) and UnifiedMPNN (unified
    bond-resolved graphs) automatically.

    For multi_topology models, evaluates on representative topologies and
    returns the average pass_rate.

    Parameters
    ----------
    model : UnifiedMPNN | MPNNPredictor
        Model to evaluate.
    topology : str
        Lattice topology. If "multi_topology", evaluates on chain_1d,
        heavy_hex, and ladder.
    n_qubits : int
        System size for evaluation.
    p_layers : int
        HVA depth.
    n_h_points : int
        Number of h-values to test.
    h_range : tuple
        (h_min, h_max) range for evaluation grid.

    Returns
    -------
    float
        pass_rate_dual in [0, 1].
    """
    import numpy as np

    # Multi-topology: evaluate on representative topologies and average
    if topology == "multi_topology":
        _MT_EVAL_TOPOLOGIES = ("chain_1d", "heavy_hex", "ladder")
        rates = []
        for topo in _MT_EVAL_TOPOLOGIES:
            rate = evaluate_model_quick(
                model,
                topology=topo,
                n_qubits=n_qubits,
                p_layers=p_layers,
                n_h_points=n_h_points,
                h_range=h_range,
            )
            rates.append(rate)
        return float(np.mean(rates)) if rates else 0.0

    try:
        import torch
        from torch_geometric.data import Batch

        from qmbp_simulation.analysis.metrics import compute_deploy_summary
        from qmbp_simulation.circuits.hva import HVACircuitBuilder
        from qmbp_simulation.execution.backends import NoiselessBackend
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder, make_lattice
        from qmbp_simulation.predictors.unified_graph import (
            build_unified_bond_resolved_graph,
        )
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache
    except ImportError as e:
        logger.warning("evaluate_model_quick: import failed: %s", e)
        return 0.0

    h_min, h_max = h_range
    h_values = np.linspace(h_min, h_max, n_h_points)

    gt_cache = GroundTruthCache()
    backend = NoiselessBackend()
    ham_builder = HamiltonianBuilder()
    builder = HVACircuitBuilder()

    # Build circuit once (parameters are symbolic)
    lattice_ref = make_lattice(topology, n_qubits, h=float(h_values[0]))
    circuit, _theta_vec = builder.create_bond_resolved(n_qubits, p_layers, lattice_ref)

    # Detect model type
    is_unified = hasattr(model, "type_emb") or "UnifiedMPNN" in type(model).__name__

    model.eval()
    per_h_results = []
    for h in h_values:
        try:
            # Ground truth from cache
            cached = gt_cache.get(
                topology=topology,
                n_qubits=n_qubits,
                model="tfim_bond_resolved",
                h=float(h),
            )
            if cached is None:
                continue
            e_exact = cached["energy"]
            gap = cached["gap"]
            if e_exact is None or gap is None:
                continue

            # Create lattice with this h
            lattice_h = make_lattice(topology, n_qubits, h=float(h))

            # Predict theta based on model type
            with torch.no_grad():
                if is_unified:
                    g = build_unified_bond_resolved_graph(
                        lattice_h,
                        h_value=float(h),
                        p_layers=p_layers,
                        include_circuit_nodes=True,
                    )
                    batch = Batch.from_data_list([g])
                    theta = model(batch).numpy().flatten()
                else:
                    # MPNNPredictor path
                    from qmbp_simulation.predictors.mpnn import predict_theta

                    predictions = predict_theta(model, lattice_h, [float(h)])
                    if not predictions:
                        continue
                    theta = predictions[float(h)]

            if not np.all(np.isfinite(theta)):
                continue

            # Evaluate energy
            H = ham_builder.build(lattice_h)
            e_pred = backend.evaluate(circuit, H, theta)

            # Build result dict
            abs_error = abs(e_pred - e_exact)
            de_gap = abs_error / max(abs(gap), 1e-10) if gap != 0 else abs_error
            per_h_results.append(
                {
                    "h": float(h),
                    "e_pred": float(e_pred),
                    "e_exact": float(e_exact),
                    "gap": float(gap),
                    "abs_error": float(abs_error),
                    "de_gap": float(de_gap),
                }
            )
        except Exception as exc:
            logger.debug("evaluate_model_quick: h=%.2f failed: %s", h, exc)
            continue

    gt_cache.flush()

    if len(per_h_results) < 2:
        return 0.0

    summary = compute_deploy_summary(per_h_results)
    return summary.get("pass_rate_5pct", 0.0)


def regression_guardrail(
    new_model,
    entry,
    *,
    eval_n_qubits: int = 10,
    tolerance: float = 0.10,
) -> tuple[bool, str]:
    """Anti-regression gate: evaluates new model and compares to existing.

    This is a stronger guardrail than the existing require_improvement flag.
    It actually RUNS the model and checks pass_rate, not just training points.

    Parameters
    ----------
    new_model : UnifiedMPNN | MPNNPredictor
        Candidate model to evaluate.
    entry : ZooEntry
        Entry metadata for the candidate.
    eval_n_qubits : int
        System size to use for quick evaluation.
    tolerance : float
        Maximum allowed regression (0.10 = 10% drop is acceptable).

    Returns
    -------
    tuple[bool, str]
        (allowed, reason): True if registration should proceed.
    """
    from qmbp_simulation.predictors.model_zoo import _load_manifest

    # Find existing model for same config
    existing_entries = [
        e
        for e in _load_manifest()
        if e.topology == entry.topology
        and e.model == entry.model
        and e.p_layers == entry.p_layers
        and e.n_qubits == entry.n_qubits
    ]

    if not existing_entries:
        return True, "no_existing_model"

    existing = max(existing_entries, key=lambda e: e.pass_rate)
    old_pass_rate = existing.pass_rate

    # If existing model was never evaluated, allow registration
    if old_pass_rate == 0.0:
        return True, "existing_unevaluated"

    # Quick-evaluate the new model
    new_pass_rate = evaluate_model_quick(
        new_model,
        topology=entry.topology,
        n_qubits=eval_n_qubits,
        p_layers=entry.p_layers,
    )

    # Decision logic
    if new_pass_rate >= old_pass_rate - tolerance:
        reason = (
            f"allowed: new={new_pass_rate:.0%} vs old={old_pass_rate:.0%} "
            f"(within tolerance={tolerance:.0%})"
        )
        return True, reason
    else:
        reason = (
            f"BLOCKED: new={new_pass_rate:.0%} < old={old_pass_rate:.0%} - "
            f"tolerance={tolerance:.0%}. Regression too large."
        )
        return False, reason


# ═══════════════════════════════════════════════════════════════════════════════
# Retrain Loop (Item 4 + 5)
# ═══════════════════════════════════════════════════════════════════════════════


def _retrain_single(
    queue_item: dict,
    *,
    n_epochs: int = 3500,
    hidden_dim: int = 256,
    n_layers: int = 3,
    use_residual: bool = True,
    film: bool = True,
) -> RetrainResult:
    """Retrain a single model from the queue.

    Implements the full cycle:
    1. Aggregate training data (MultiTopologyAggregator or MultiNAggregator)
    2. Train UnifiedMPNN
    3. Evaluate with regression_guardrail
    4. Register if improved (or reject if regression)
    5. Run comparison against existing models

    Returns
    -------
    RetrainResult
        Full result of the retrain attempt.
    """
    t_start = time.time()
    topo = queue_item["topology"]
    checkpoint_file = queue_item["checkpoint_file"]
    priority = queue_item["priority"]
    reason = queue_item["reason"]
    n_values = queue_item.get("n_values_available", [])

    logger.info(
        "  Retraining %s (P%d: %s) with %d N-values...",
        topo,
        priority,
        reason,
        len(n_values),
    )

    try:
        from qmbp_simulation.predictors.model_zoo import (
            ZooEntry,
            _load_manifest,
            register_checkpoint_with_training_metrics,
        )
        from qmbp_simulation.predictors.multi_n_aggregator import (
            MultiNAggregator,
            MultiTopologyAggregator,
        )
        from qmbp_simulation.predictors.unified_mpnn import (
            UnifiedMPNN,
            train_unified_mpnn,
        )
    except ImportError as e:
        return RetrainResult(
            topology=topo,
            checkpoint_file=checkpoint_file,
            priority=priority,
            reason=reason,
            action="failed",
            error=f"Import error: {e}",
            elapsed_s=time.time() - t_start,
        )

    # ── Step 1: Aggregate training data ──────────────────────────────────
    try:
        # Check if it's a multi-topology model
        if "multitopo" in checkpoint_file.lower():
            agg = MultiTopologyAggregator(model="tfim_bond_resolved")
            agg.scan()
            dataset = agg.build_combined_dataset(max_de_gap=0.10, max_n=20)
        else:
            agg = MultiNAggregator(topology=topo, model="tfim_bond_resolved")
            agg.scan()
            dataset = agg.build_combined_dataset(max_de_gap=0.10)

        if len(dataset) < 10:
            return RetrainResult(
                topology=topo,
                checkpoint_file=checkpoint_file,
                priority=priority,
                reason=reason,
                action="skipped",
                error=f"Insufficient data: only {len(dataset)} points after filter",
                elapsed_s=time.time() - t_start,
            )
    except Exception as e:
        return RetrainResult(
            topology=topo,
            checkpoint_file=checkpoint_file,
            priority=priority,
            reason=reason,
            action="failed",
            error=f"Data aggregation failed: {e}",
            elapsed_s=time.time() - t_start,
        )

    # ── Step 2: Train UnifiedMPNN ────────────────────────────────────────
    try:
        sample_g = dataset[0]
        n_node_features = sample_g.x.shape[1] if hasattr(sample_g, "x") else 4

        model = UnifiedMPNN(
            node_features=n_node_features,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            norm_type="none",
            dropout=0.1,
            use_residual=use_residual,
            film_conditioning=film,
        )
        train_result = train_unified_mpnn(
            model,
            dataset,
            n_epochs=n_epochs,
            lr=1e-3,
            patience=300,
            seed=42,
        )
        final_mse = train_result.get("final_mse", 0.0) if isinstance(train_result, dict) else 0.0
    except Exception as e:
        return RetrainResult(
            topology=topo,
            checkpoint_file=checkpoint_file,
            priority=priority,
            reason=reason,
            action="failed",
            error=f"Training failed: {e}",
            elapsed_s=time.time() - t_start,
        )

    # ── Step 3: Regression guardrail ─────────────────────────────────────
    # Get existing pass_rate for comparison
    existing_entries = [e for e in _load_manifest() if e.checkpoint_file == checkpoint_file]
    old_pass_rate = existing_entries[0].pass_rate if existing_entries else 0.0

    allowed, guardrail_reason = regression_guardrail(
        model,
        ZooEntry(
            model="tfim_bond_resolved",
            topology=topo,
            n_qubits=0,
            p_layers=1,
            checkpoint_file=checkpoint_file,
        ),
        eval_n_qubits=10,
        tolerance=0.10,
    )

    if not allowed:
        logger.warning("  🚫 Retrain BLOCKED for %s: %s", topo, guardrail_reason)
        return RetrainResult(
            topology=topo,
            checkpoint_file=checkpoint_file,
            priority=priority,
            reason=reason,
            action="blocked",
            old_pass_rate=old_pass_rate,
            training_mse=float(final_mse),
            n_training_points=len(dataset),
            error=guardrail_reason,
            elapsed_s=time.time() - t_start,
        )

    # ── Step 4: Register with training metrics ───────────────────────────
    try:
        from datetime import datetime

        new_entry = ZooEntry(
            model="tfim_bond_resolved",
            topology=topo,
            n_qubits=0,
            p_layers=1,
            checkpoint_file=checkpoint_file,
            h_range=(1.0, 5.0),
            pass_rate=0.0,  # Will be updated by comparison
            n_training_points=len(dataset),
            seeds=[42],
            created=datetime.now(UTC).isoformat(),
            notes=f"Auto-retrain (P{priority}): {reason}",
            runner_tag="AR",
            date_tag=datetime.now(UTC).strftime("%d%m%y"),
        )

        register_checkpoint_with_training_metrics(
            model,
            new_entry,
            training_result=train_result if isinstance(train_result, dict) else None,
            overwrite=True,
            auto_diagnose=False,  # Skip to save time in batch
            auto_sync_dashboard=False,  # Will sync at end of loop
        )
    except Exception as e:
        return RetrainResult(
            topology=topo,
            checkpoint_file=checkpoint_file,
            priority=priority,
            reason=reason,
            action="failed",
            old_pass_rate=old_pass_rate,
            training_mse=float(final_mse),
            n_training_points=len(dataset),
            error=f"Registration failed: {e}",
            elapsed_s=time.time() - t_start,
        )

    # ── Step 5: Run comparison (Item 5 — comparison-on-retrain) ──────────
    new_pass_rate = _run_comparison_after_retrain(topo)
    comparison_winner = "new" if new_pass_rate > old_pass_rate else "existing"

    return RetrainResult(
        topology=topo,
        checkpoint_file=checkpoint_file,
        priority=priority,
        reason=reason,
        action="retrained",
        old_pass_rate=old_pass_rate,
        new_pass_rate=new_pass_rate,
        comparison_winner=comparison_winner,
        training_mse=float(final_mse),
        n_training_points=len(dataset),
        elapsed_s=time.time() - t_start,
    )


def _run_comparison_after_retrain(topology: str) -> float:
    """Run model_comparison.py after retrain to measure impact (Item 5).

    Executes as subprocess with --auto-detect --promote-best to:
    1. Compare all available models for this topology
    2. Promote the winner to production
    3. Update pass_rate_by_n in zoo manifest

    Returns the winning model's pass_rate (or 0.0 on failure).
    """
    comparison_script = (
        _ROOT / "scripts" / "experiment_runners" / "cross_topology" / "run_model_comparison.py"
    )
    if not comparison_script.exists():
        logger.warning("  Comparison script not found: %s", comparison_script)
        return 0.0

    cmd = [
        sys.executable,
        str(comparison_script),
        "--topology",
        topology,
        "--target-n",
        "10",
        "16",
        "20",
        "--auto-detect",
        "--promote-best",
        "--h-points",
        "8",
    ]

    logger.info("  Running post-retrain comparison on %s...", topology)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(_ROOT),
        )
        if proc.returncode != 0:
            logger.warning(
                "  Comparison failed (exit=%d): %s",
                proc.returncode,
                proc.stderr[-200:] if proc.stderr else "no stderr",
            )
            return 0.0

        # Extract pass_rate from output
        for line in proc.stdout.split("\n"):
            if "pass_rate" in line.lower() and "winner" in line.lower():
                # Try to parse pass_rate from "Winner: ... pass_rate=XX%"
                import re

                match = re.search(r"pass_rate[=:]\s*(\d+(?:\.\d+)?)%?", line)
                if match:
                    val = float(match.group(1))
                    return val / 100.0 if val > 1.0 else val

        # Fallback: read from zoo manifest
        from qmbp_simulation.predictors.model_zoo import _load_manifest

        entries = [e for e in _load_manifest() if e.topology == topology and e.n_qubits == 0]
        if entries:
            return max(e.pass_rate for e in entries)
        return 0.0

    except subprocess.TimeoutExpired:
        logger.warning("  Comparison timed out (600s) for %s", topology)
        return 0.0
    except Exception as e:
        logger.warning("  Comparison error for %s: %s", topology, e)
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


def run_retrain_loop(
    *,
    max_retrains: int = 3,
    min_priority: int = 4,
    n_epochs: int = 3500,
    dry_run: bool = False,
    verbose: bool = False,
) -> RetrainLoopResult:
    """Execute the automated retrain loop.

    Queries compute_retrain_queue(), filters by priority, and retrains
    up to max_retrains models. After each retrain:
    1. Runs regression_guardrail to prevent quality loss
    2. Runs model_comparison to measure improvement
    3. Updates zoo pass_rate based on comparison results

    Parameters
    ----------
    max_retrains : int
        Maximum number of models to retrain per invocation (default: 3).
    min_priority : int
        Only retrain models with priority <= this (1=highest, 4=lowest).
        Default: 4 (retrain all priorities).
    n_epochs : int
        Training epochs (default: 3500).
    dry_run : bool
        If True, report what would be retrained without executing.
    verbose : bool
        If True, print progress to stdout.

    Returns
    -------
    RetrainLoopResult
        Aggregate statistics and per-model results.
    """
    from qmbp_simulation.predictors.model_zoo import compute_retrain_queue

    t_start = time.time()

    # ── Step 1: Get retrain candidates ───────────────────────────────────
    queue = compute_retrain_queue()
    filtered = [item for item in queue if item["priority"] <= min_priority]
    filtered = filtered[:max_retrains]

    result = RetrainLoopResult(
        n_candidates=len(filtered),
        dry_run=dry_run,
    )

    if not filtered:
        logger.info("  Retrain loop: no candidates (queue empty or all filtered)")
        result.total_elapsed_s = time.time() - t_start
        return result

    if verbose:
        print(
            f"\n  {'[DRY-RUN] ' if dry_run else ''}Retrain loop: "
            f"{len(filtered)} candidates (from {len(queue)} total)"
        )
        for item in filtered:
            print(f"    P{item['priority']}: {item['topology']} — {item['reason']}")

    if dry_run:
        for item in filtered:
            result.results.append(
                RetrainResult(
                    topology=item["topology"],
                    checkpoint_file=item["checkpoint_file"],
                    priority=item["priority"],
                    reason=item["reason"],
                    action="would_retrain",
                    old_pass_rate=item.get("current_pass_rate", 0.0),
                )
            )
        result.total_elapsed_s = time.time() - t_start
        return result

    # ── Step 2: Retrain each candidate ───────────────────────────────────
    for item in filtered:
        if verbose:
            print(f"\n  Retraining: {item['topology']} (P{item['priority']})...")

        retrain_result = _retrain_single(item, n_epochs=n_epochs)
        result.results.append(retrain_result)

        if retrain_result.action == "retrained":
            result.n_retrained += 1
            if retrain_result.new_pass_rate > retrain_result.old_pass_rate:
                result.n_improved += 1
        elif retrain_result.action == "blocked":
            result.n_blocked += 1
        elif retrain_result.action == "failed":
            result.n_failed += 1

        if verbose:
            _icon = {
                "retrained": "✅",
                "blocked": "🚫",
                "failed": "❌",
                "skipped": "⏭️",
            }.get(retrain_result.action, "?")
            print(f"    {_icon} {retrain_result.action}: {retrain_result.error or ''}")
            if retrain_result.action == "retrained":
                print(
                    f"       pass_rate: {retrain_result.old_pass_rate:.0%} → "
                    f"{retrain_result.new_pass_rate:.0%} "
                    f"(winner: {retrain_result.comparison_winner})"
                )

    # ── Step 3: Post-loop sync ───────────────────────────────────────────
    if result.n_retrained > 0:
        try:
            from qmbp_simulation.analysis.metrics import (
                generate_model_quality_dashboard,
            )

            generate_model_quality_dashboard()
            logger.info("  Post-retrain: dashboard regenerated")
        except Exception as e:
            logger.debug("  Post-retrain dashboard failed (non-critical): %s", e)

    result.total_elapsed_s = time.time() - t_start
    if verbose:
        print(f"\n  {result.summary}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    """CLI entry point for the retrain loop."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Automated retrain loop: train→eval→compare→update_zoo"
    )
    parser.add_argument(
        "--max-retrains",
        type=int,
        default=3,
        help="Maximum models to retrain (default: 3)",
    )
    parser.add_argument(
        "--min-priority",
        type=int,
        default=4,
        help="Max priority level to include (1=highest, 4=lowest, default: 4)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=3500,
        help="Training epochs (default: 3500)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report without executing")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    result = run_retrain_loop(
        max_retrains=args.max_retrains,
        min_priority=args.min_priority,
        n_epochs=args.n_epochs,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    print(f"\n{result.summary}")
    return 0 if result.n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
