#!/usr/bin/env python3
"""Train a universal multi-topology MPNN from high-quality data across all topologies.

Aggregates verified/approximate data from chain_1d, heavy_hex, ladder, square,
and triangular into a single UnifiedMPNN that can predict θ for any topology.

The architecture is already topology-agnostic (GINConv operates on graph structure
which implicitly encodes topology via edge connectivity). This script just combines
the training data.

Usage:
    # Default: all topologies, max_n=20, 4000 epochs
    .venv/bin/python scripts/experiment_runners/cross_topology/run_multi_topology_training.py

    # Specific topologies
    .venv/bin/python scripts/experiment_runners/cross_topology/run_multi_topology_training.py \
        --topologies chain_1d ladder square

    # Quick test (fewer epochs)
    .venv/bin/python scripts/experiment_runners/cross_topology/run_multi_topology_training.py \
        --epochs 500 --dry-run

    # With larger model
    .venv/bin/python scripts/experiment_runners/cross_topology/run_multi_topology_training.py \
        --hidden-dim 512 --n-layers 4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from qmbp_simulation.predictors.multi_n_aggregator import MultiTopologyAggregator
from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN, train_unified_mpnn

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a universal multi-topology UnifiedMPNN")
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=None,
        help="Topologies to include (default: auto-detect all available)",
    )
    parser.add_argument(
        "--p-layers",
        type=int,
        default=1,
        help="HVA depth — trains on data from this p only (default: 1)",
    )
    parser.add_argument("--max-n", type=int, default=20, help="Max N per topology")
    parser.add_argument("--max-de-gap", type=float, default=0.10, help="Quality filter")
    parser.add_argument("--min-verified", type=int, default=5, help="Min verified pts/topology")
    parser.add_argument("--hidden-dim", type=int, default=256, help="GNN hidden dimension")
    parser.add_argument("--n-layers", type=int, default=3, help="GNN layers")
    parser.add_argument("--epochs", type=int, default=2000, help="Max training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=200, help="LR scheduler patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--mse-floor", type=float, default=0.0, help="Stop early when MSE < this (0=disabled)"
    )
    parser.add_argument(
        "--use-residual", action="store_true", help="Enable residual connections (P1)"
    )
    parser.add_argument(
        "--readout-mode",
        choices=["last", "jk_cat", "jk_max"],
        default="last",
        help="Readout aggregation mode (P2)",
    )
    parser.add_argument("--film", action="store_true", help="Enable FiLM conditioning by h (P4)")
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Curriculum training: first high-quality topos, then fine-tune all",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only aggregate, don't train")
    parser.add_argument(
        "--register-zoo",
        action="store_true",
        default=True,
        help="Register trained model in zoo (default: True)",
    )
    parser.add_argument("--no-register", dest="register_zoo", action="store_false")
    parser.add_argument(
        "--auto-compare",
        action="store_true",
        help="After registration, run model_comparison on all topologies to validate",
    )
    parser.add_argument(
        "--regression-guard",
        action="store_true",
        default=True,
        help="Run active evaluation before registration to block regressions (default: True)",
    )
    parser.add_argument(
        "--no-regression-guard",
        dest="regression_guard",
        action="store_false",
        help="Skip regression guard (allow any model to register)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Override pre-training readiness check (proceed despite issues)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-5s %(message)s",
    )

    print("=" * 70)
    print("Multi-Topology MPNN Training")
    print(f"  p_layers = {args.p_layers}")
    print("=" * 70)

    # ── Pre-training gate: validate readiness ────────────────────────────
    from qmbp_simulation.predictors.training_intelligence import (
        prepare_training_config,
        validate_training_readiness,
    )

    # Resolve topologies (None = auto-detect from available data)
    check_topologies = args.topologies
    if check_topologies is None:
        # Auto-detect: will be resolved later by MultiTopologyAggregator
        # For validation, use all known topologies
        check_topologies = ["chain_1d", "heavy_hex", "ladder", "square", "triangular"]

    is_ready, readiness_issues = validate_training_readiness(
        check_topologies,
        min_useful_points=30,
        min_h_coverage=0.50,
    )
    if readiness_issues:
        print("\n  Pre-training validation:")
        for issue in readiness_issues:
            print(f"    ⚠️ {issue}")
        if not is_ready and not getattr(args, "force", False):
            print("\n  ❌ Training NOT ready. Fix issues above or use --force to override.")
            return 1
        elif not is_ready:
            print("\n  ⚠️ Proceeding with --force despite issues.")
    else:
        print("  ✅ Pre-training validation passed")

    # Prepare optimized config (informational)
    config = prepare_training_config(
        topologies=args.topologies,
        max_n=args.max_n,
        include_extrapolation=True,
    )
    if config.warnings:
        print("\n  Training config warnings:")
        for w in config.warnings[:3]:
            print(f"    ⚠️ {w}")
    print(
        f"  Data: {config.n_useful_points} useful pts, "
        f"extrap={config.use_extrapolation_data} (weight={config.extrapolation_weight:.2f}), "
        f"confidence={config.confidence}"
    )
    print()

    # ── Phase 1: Aggregate data ──────────────────────────────────────────
    t0 = time.perf_counter()
    agg = MultiTopologyAggregator(
        topologies=args.topologies,
        model="tfim_bond_resolved",
        max_n=args.max_n,
        min_verified_points=args.min_verified,
        p_layers=args.p_layers,
    )
    summary = agg.scan()

    if not summary:
        print("ERROR: No data found across any topology.")
        return 1

    # ── Exclusion-policy N-level filter (per topology) ────────────────────
    # Remove N-values flagged with hard failure modes from each topology's data.
    try:
        from qmbp_simulation.analysis.metrics import load_training_exclusions

        _excl_registry = load_training_exclusions()
        _hard_modes = {"contaminated_training", "gap_masking", "intrinsic_vqe_error"}
        for topo, inner_agg in list(agg._aggregators.items()):
            excluded_ns = set()
            for entry in _excl_registry.get("excluded", []):
                if entry.get("topology") == topo and entry.get("failure_mode") in _hard_modes:
                    n_val = entry.get("n_qubits", 0)
                    if n_val > 0:
                        excluded_ns.add(n_val)
            if excluded_ns:
                for n_val in excluded_ns:
                    if n_val in inner_agg._data_by_n:
                        n_pts = len(inner_agg._data_by_n.pop(n_val))
                        print(f"  Exclusion policy: {topo} N={n_val} removed ({n_pts} pts)")
                # Update summary
                summary[topo] = {n: len(pts) for n, pts in inner_agg._data_by_n.items()}
                if not summary[topo]:
                    del summary[topo]
                    del agg._aggregators[topo]
    except Exception:
        pass  # Non-critical

    print(f"\nData summary (max_n={args.max_n}):")
    total_pts = 0
    for topo, ns in sorted(summary.items()):
        pts = sum(ns.values())
        total_pts += pts
        n_list = sorted(ns.keys())
        print(f"  {topo:12s}: N={n_list}, {pts} points")
    print(f"  {'TOTAL':12s}: {total_pts} points across {len(summary)} topologies")

    if args.dry_run:
        print("\n[DRY RUN] Building dataset without training...")

    # ── Phase 2: Build combined dataset ──────────────────────────────────
    print(f"\nBuilding dataset (max_de_gap={args.max_de_gap})...")

    # Pre-training validation per topology
    from qmbp_simulation.analysis.metrics import validate_training_dataset

    for topo, inner_agg in list(agg._aggregators.items()):
        viable, report = validate_training_dataset(
            inner_agg._data_by_n,
            max_de_gap=args.max_de_gap,
            min_total_points=5,
            min_n_values=1,
        )
        if not viable:
            print(f"  ⚠️  {topo}: data not viable — {report.get('recommendation', 'skip')}")
            del agg._aggregators[topo]

    if not agg._aggregators:
        print("ERROR: No topology passed validation. Check data quality.")
        return 1

    dataset = agg.build_combined_dataset(max_de_gap=args.max_de_gap)

    if not dataset:
        print("ERROR: No data after filtering. Check quality tiers.")
        return 1

    # ── Safeguard: warn if strict filter drops too many points ────────────
    if args.max_de_gap < 0.10:
        dataset_relaxed = agg.build_combined_dataset(max_de_gap=0.10)
        retention_ratio = len(dataset) / max(len(dataset_relaxed), 1)
        if retention_ratio < 0.50:
            print(
                f"  ⚠️  Strict filter (max_de_gap={args.max_de_gap}) retains only "
                f"{len(dataset)}/{len(dataset_relaxed)} graphs ({retention_ratio:.0%})."
            )
            print(
                "      Consider using --max-de-gap 0.07 as a compromise, or verify "
                "that remaining data has sufficient topology/N diversity."
            )
        del dataset_relaxed  # Free memory

    # Show topology distribution
    from collections import Counter

    topo_dist = Counter(g.topology for g in dataset)
    print(f"\nDataset: {len(dataset)} training graphs")
    for topo, count in sorted(topo_dist.items()):
        print(f"  {topo:12s}: {count} graphs")

    # Verify feature consistency
    feat_dim = dataset[0].x.shape[1]
    print(f"\nNode features: {feat_dim}")

    # Dataset fingerprint for reproducibility tracking
    from qmbp_simulation.utils.helpers import compute_dataset_fingerprint

    _dataset_fp = compute_dataset_fingerprint(dataset)
    print(f"Dataset fingerprint: {_dataset_fp}")

    if args.dry_run:
        print("\n[DRY RUN] Would train with these parameters:")
        print(f"  hidden_dim={args.hidden_dim}, n_layers={args.n_layers}")
        print(f"  epochs={args.epochs}, lr={args.lr}, patience={args.patience}")
        return 0

    # ── Phase 3: Train model ─────────────────────────────────────────────
    arch_label = "baseline"
    arch_parts = []
    if args.use_residual:
        arch_parts.append("residual")
    if args.readout_mode != "last":
        arch_parts.append(args.readout_mode)
    if args.film:
        arch_parts.append("film")
    if arch_parts:
        arch_label = "+".join(arch_parts)

    print(
        f"\nTraining UnifiedMPNN (hidden={args.hidden_dim}, layers={args.n_layers}, arch={arch_label})..."
    )
    model = UnifiedMPNN(
        node_features=feat_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        norm_type="none",
        dropout=0.1,
        type_embedding_dim=16,
        gate_readout=True,
        use_residual=args.use_residual,
        readout_mode=args.readout_mode,
        film_conditioning=args.film,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # ── Training with interrupt-safe recovery ────────────────────────────
    _training_interrupted = False
    if args.curriculum:
        # ── Curriculum training: high-quality first, then all ─────────
        # Phase A: train on high-quality topologies only (de_gap < 0.03)
        high_quality_topos = set()
        for topo, inner_agg in agg._aggregators.items():
            all_dg = [p.get("de_gap", 1) for pts in inner_agg._data_by_n.values() for p in pts]
            if np.mean(all_dg) < 0.035:
                high_quality_topos.add(topo)

        hq_dataset = [g for g in dataset if g.topology in high_quality_topos]
        if len(hq_dataset) >= 10 and len(hq_dataset) < len(dataset):
            phase_a_epochs = int(args.epochs * 0.6)
            phase_b_epochs = args.epochs - phase_a_epochs

            print(
                f"\n  [CURRICULUM] Phase A: {len(hq_dataset)} graphs from {sorted(high_quality_topos)} "
                f"({phase_a_epochs} epochs)"
            )

            result_a = train_unified_mpnn(
                model,
                hq_dataset,
                n_epochs=phase_a_epochs,
                lr=args.lr,
                patience=args.patience,
                seed=args.seed,
                weight_decay=1e-4,
                val_fraction=0.15,
                mse_floor=args.mse_floor,
            )
            print(
                f"    Phase A done: MSE={result_a['final_mse']:.2e}, val={result_a['val_mse']:.2e}"
            )

            # Phase B: fine-tune on ALL data with reduced LR
            print(
                f"\n  [CURRICULUM] Phase B: {len(dataset)} graphs (all) "
                f"({phase_b_epochs} epochs, lr={args.lr * 0.3:.1e})"
            )

            from qmbp_simulation.predictors.unified_mpnn import fine_tune_unified_mpnn

            result = fine_tune_unified_mpnn(
                model,
                dataset,
                n_epochs=phase_b_epochs,
                lr=args.lr * 0.3,
                patience=args.patience // 2,
            )
            # Merge metrics
            result["n_epochs_run"] = result_a["n_epochs_run"] + result.get("n_epochs_run", 0)
            result["stop_reason"] = (
                f"curriculum: A={result_a['stop_reason']}, B={result.get('stop_reason', '?')}"
            )
            if "final_zz_mse" not in result:
                result["final_zz_mse"] = result.get("final_mse", 0)
                result["final_x_mse"] = result.get("final_mse", 0)
        else:
            print("\n  [CURRICULUM] Skipped: all topos are high-quality or insufficient split")
            result = train_unified_mpnn(
                model,
                dataset,
                n_epochs=args.epochs,
                lr=args.lr,
                patience=args.patience,
                seed=args.seed,
                weight_decay=1e-4,
                val_fraction=0.15,
                mse_floor=args.mse_floor,
            )
    else:
        try:
            result = train_unified_mpnn(
                model,
                dataset,
                n_epochs=args.epochs,
                lr=args.lr,
                patience=args.patience,
                seed=args.seed,
                weight_decay=1e-4,
                val_fraction=0.15,
                mse_floor=args.mse_floor,
            )
        except KeyboardInterrupt:
            _training_interrupted = True
            print("\n  ⚠️  Training interrupted! Saving partial model to _recovery/...")
            try:
                from qmbp_simulation.predictors.unified_mpnn import save_unified_checkpoint

                recovery_dir = ROOT / "data" / "model_zoo" / "checkpoints" / "_recovery"
                recovery_dir.mkdir(parents=True, exist_ok=True)
                recovery_path = recovery_dir / "interrupted_mt_model.pt"
                save_unified_checkpoint(model, str(recovery_path))
                print(f"  Recovery checkpoint: {recovery_path.relative_to(ROOT)}")
            except Exception as _save_err:
                print(f"  Recovery save failed: {_save_err}")
            result = {
                "final_mse": float("inf"),
                "final_zz_mse": float("inf"),
                "final_x_mse": float("inf"),
                "val_mse": None,
                "n_epochs_run": 0,
                "stop_reason": "interrupted",
                "n_train": len(dataset),
                "n_val": 0,
                "mse_history": [],
                "val_mse_history": [],
                "zz_loss_history": [],
                "x_loss_history": [],
            }

    elapsed = time.perf_counter() - t0
    print(f"\nTraining complete ({elapsed:.1f}s):")
    print(
        f"  Final MSE: {result['final_mse']:.2e} (ZZ={result['final_zz_mse']:.2e}, X={result['final_x_mse']:.2e})"
    )
    print(f"  Val MSE: {result['val_mse']:.2e}" if result["val_mse"] else "  Val MSE: N/A")
    print(f"  Epochs: {result['n_epochs_run']}, stop_reason: {result['stop_reason']}")
    print(f"  Train/Val: {result['n_train']}/{result['n_val']}")

    # ── Persist training curve for post-hoc analysis ─────────────────────
    from qmbp_simulation.utils.helpers import persist_training_curve

    curve_path = persist_training_curve(
        result,
        output_dir=ROOT / "results" / "training_curves",
        prefix="mt_training",
    )
    if curve_path:
        print(f"  Training curve: {curve_path.relative_to(ROOT)}")

    # ── Convergence diagnostic ───────────────────────────────────────────
    if result["stop_reason"] == "completed":
        print(
            "\n  ⚠️  Training hit max epochs without early-stop trigger."
            "\n      This may indicate the model has NOT converged."
            "\n      Consider: --epochs (more), --patience (lower LR schedule threshold),"
            "\n      or more training data. Check MSE curve for plateau."
        )
    elif result["stop_reason"] == "overfitting_detected":
        print(
            "\n  ⚠️  Training stopped due to overfitting detection."
            "\n      Val MSE is rising while train MSE drops. Model may still be usable"
            "\n      but generalization is limited. Consider: more data or regularization."
        )

    # ── Phase 4: Save experiment envelope + Register in zoo ─────────────
    # Save JSON envelope FIRST (even if registration fails, we have the record)
    _run_json_path = ""
    try:
        from qmbp_simulation.framework.result_io import build_result_envelope, save_experiment_result

        envelope = build_result_envelope(
            config={
                "runner": "run_multi_topology_training",
                "topologies": sorted(topo_dist.keys()),
                "max_n": args.max_n,
                "max_de_gap": args.max_de_gap,
                "hidden_dim": args.hidden_dim,
                "n_layers": args.n_layers,
                "epochs": args.epochs,
                "lr": args.lr,
                "patience": args.patience,
                "use_residual": args.use_residual,
                "readout_mode": args.readout_mode,
                "film": args.film,
                "curriculum": args.curriculum,
                "seed": args.seed,
                "dataset_fingerprint": _dataset_fp,
                "n_graphs": len(dataset),
                "graphs_per_topology": dict(sorted(topo_dist.items())),
            },
            results={
                "final_mse": result.get("final_mse"),
                "final_zz_mse": result.get("final_zz_mse"),
                "final_x_mse": result.get("final_x_mse"),
                "val_mse": result.get("val_mse"),
                "n_epochs_run": result.get("n_epochs_run"),
                "stop_reason": result.get("stop_reason"),
                "n_train": result.get("n_train"),
                "n_val": result.get("n_val"),
            },
            summary={
                "arch_label": arch_label,
                "n_topologies": len(topo_dist),
                "total_graphs": len(dataset),
                "training_time_s": elapsed,
            },
            elapsed_s=elapsed,
        )
        output_path = save_experiment_result(
            envelope,
            experiment_id="multi_topology_training/tfim_bond_resolved",
            results_dir=ROOT / "results" / "experiments",
        )
        _run_json_path = str(output_path.relative_to(ROOT))
        print(f"\n  📄 Experiment JSON: {_run_json_path}")
    except Exception as _e_env:
        print(f"\n  ⚠️ Experiment envelope save failed (non-blocking): {_e_env}")

    if _training_interrupted:
        print("\n  Skipping zoo registration (training was interrupted).")
        print("  Resume from: data/model_zoo/checkpoints/_recovery/interrupted_mt_model.pt")
        print("=" * 70)
        return 1
    if args.register_zoo:
        print("\nRegistering in model zoo...")
        from qmbp_simulation.predictors.model_zoo import (
            _CHECKPOINTS_DIR,
            ZooEntry,
            _load_manifest,
            register_checkpoint_with_training_metrics,
        )

        topo_str = "+".join(sorted(topo_dist.keys()))
        arch_suffix = f"_{arch_label}" if arch_label != "baseline" else ""
        checkpoint_file = f"unified_tfim_br_MT{arch_suffix}_p{args.p_layers}.pt"

        # ── Safety: save to _recovery/ BEFORE registration attempt ───────
        # If registration crashes (guardrail, validation, etc.), model is preserved.
        from qmbp_simulation.predictors.unified_mpnn import save_unified_checkpoint

        recovery_dir = _CHECKPOINTS_DIR / "_recovery"
        recovery_dir.mkdir(parents=True, exist_ok=True)
        recovery_path = recovery_dir / f"pre_register_{checkpoint_file}"
        save_unified_checkpoint(model, str(recovery_path))
        print(f"  Safety backup: {recovery_path.relative_to(ROOT)}")

        # ── Pre-flight: check if existing model is better ────────────────
        existing_entries = [e for e in _load_manifest() if e.checkpoint_file == checkpoint_file]
        if existing_entries:
            existing = existing_entries[0]
            existing_pass_rate = existing.pass_rate
            existing_pts = existing.n_training_points
            print(f"  Pre-flight: existing model found ({checkpoint_file})")
            print(f"    Existing: pass_rate={existing_pass_rate:.2f}, pts={existing_pts}")
            print(f"    New:      MSE={result['final_mse']:.2e}, pts={len(dataset)}")
            print("    Anti-regression: old model auto-backed up to _versions/ + _best/")
        else:
            print("  Pre-flight: no existing model with this name. Fresh registration.")

        entry = ZooEntry(
            model="tfim_bond_resolved",
            topology="multi_topology",
            n_qubits=0,
            p_layers=args.p_layers,
            checkpoint_file=checkpoint_file,
            h_range=(0.5, 5.0),
            pass_rate=0.0,
            n_training_points=len(dataset),
            seeds=[args.seed],
            created=datetime.now(UTC).isoformat(),
            notes=(
                f"Multi-topology model ({arch_label}): {topo_str}. "
                f"p={args.p_layers}. "
                f"MSE={result['final_mse']:.2e}, val={result['val_mse']:.2e}. "
                f"{len(dataset)} graphs from {len(topo_dist)} topologies."
            ),
        )

        register_checkpoint_with_training_metrics(
            model,
            entry,
            training_result=result,
            overwrite=True,
            architecture_config={
                "hidden_dim": args.hidden_dim,
                "n_conv_layers": args.n_layers,
                "type_embedding_dim": 16,
                "gate_readout": True,
                "dropout": 0.1,
                "use_residual": args.use_residual,
                "readout_mode": args.readout_mode,
                "film_conditioning": args.film,
                "arch_label": arch_label,
                # MT-specific provenance (searchable in ModelRegistryDB)
                "topologies_used": sorted(topo_dist.keys()),
                "max_n": args.max_n,
                "graphs_per_topology": dict(sorted(topo_dist.items())),
                "curriculum": args.curriculum,
                "dataset_fingerprint": _dataset_fp,
            },
            optimizer_config={
                "learning_rate": args.lr,
                "weight_decay": 1e-4,
                "scheduler_patience": args.patience,
                "mse_floor": args.mse_floor,
            },
            auto_diagnose=True,
            auto_sync_dashboard=False,  # Multi-topo not in per-topology dashboard
            regression_guard=args.regression_guard,
            run_json_path=_run_json_path,
        )
        print(f"  Registered: {entry.checkpoint_file}")

        # ── Clean up safety backup (registration succeeded) ──────────────
        if recovery_path.exists():
            recovery_path.unlink()
            print("  Cleaned up safety backup")

        # ── Quick evaluation: compute val-set pass_rate ──────────────────
        print("  Running quick evaluation...", end="", flush=True)
        try:
            import torch

            model.eval()
            n_pass = 0
            n_total = 0
            with torch.no_grad():
                for g in dataset[:50]:  # Evaluate on first 50 graphs (fast)
                    pred = model(g).squeeze(0)
                    target = g.y
                    if len(pred) != len(target):
                        continue
                    mse = torch.nn.functional.mse_loss(pred, target).item()
                    n_total += 1
                    if mse < 0.01:
                        n_pass += 1

            if n_total > 0:
                quick_pass_rate = n_pass / n_total
                print(f" pass_rate={quick_pass_rate:.0%} ({n_pass}/{n_total})")

                from qmbp_simulation.predictors.model_zoo import update_zoo_pass_rate

                update_zoo_pass_rate(checkpoint_file, quick_pass_rate)
            else:
                print(" skipped (no valid graphs)")
        except Exception as eval_err:
            print(f" failed: {eval_err}")

    # ── Phase 5 (optional): Auto-compare against existing models ─────────
    if args.auto_compare and args.register_zoo:
        print("\n  Running auto-comparison against existing models...")
        import subprocess

        topologies_to_check = sorted(topo_dist.keys())[:3]  # Top 3 topologies
        for topo in topologies_to_check:
            cmd = [
                sys.executable,
                str(ROOT / "scripts/experiment_runners/cross_topology/run_model_comparison.py"),
                "--topology",
                topo,
                "--target-n",
                "20",
                "--auto-detect",
                "--promote-best",
                "--h-points",
                "6",
            ]
            print(f"    Comparing on {topo}...", end="", flush=True)
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=str(ROOT),
                )
                # Extract the winner line from output
                for line in proc.stdout.split("\n"):
                    if "Best:" in line or "Winner:" in line or "📊" in line:
                        print(f" {line.strip()}")
                        break
                else:
                    print(" done")
            except (subprocess.TimeoutExpired, Exception) as e:
                print(f" skipped ({e})")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
