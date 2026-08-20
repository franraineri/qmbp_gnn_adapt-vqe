#!/usr/bin/env python3
"""Fine-tune a multi-topology model for a specific topology.

Takes the universal MT model and specializes it with topology-specific data.
Combines: universal knowledge (good interpolation) + local geometry (better extrap).

Uses existing infrastructure:
- load_unified_checkpoint (loads MT model with arch detection)
- MultiNAggregator (filters data to one topology)
- fine_tune_unified_mpnn (layer-wise LR decay, early stopping)
- register_checkpoint_with_training_metrics (full provenance)

Usage:
    # Fine-tune MT model for ladder topology
    .venv/bin/python scripts/experiment_runners/cross_topology/run_finetune_from_mt.py \
        --topology ladder

    # Custom source checkpoint
    .venv/bin/python scripts/experiment_runners/cross_topology/run_finetune_from_mt.py \
        --topology square --source-checkpoint path/to/mt_model.pt

    # More epochs, stricter quality filter
    .venv/bin/python scripts/experiment_runners/cross_topology/run_finetune_from_mt.py \
        --topology triangular --epochs 1000 --max-de-gap 0.05
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from qmbp_simulation.predictors.model_zoo import (
    ZooEntry,
    _CHECKPOINTS_DIR,
    _load_manifest,
    register_checkpoint_with_training_metrics,
)
from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator
from qmbp_simulation.predictors.unified_mpnn import (
    fine_tune_unified_mpnn,
    load_unified_checkpoint,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MT model for a specific topology")
    parser.add_argument("--topology", required=True, help="Target topology to specialize for")
    parser.add_argument("--source-checkpoint", default=None,
                        help="MT checkpoint path (auto-detect from zoo if not provided)")
    parser.add_argument("--max-n", type=int, default=20, help="Max N for training data")
    parser.add_argument("--max-de-gap", type=float, default=0.10, help="Quality filter")
    parser.add_argument("--epochs", type=int, default=500, help="Fine-tune epochs (default: 500)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--patience", type=int, default=150, help="Scheduler patience")
    parser.add_argument("--p-layers", type=int, default=1)
    parser.add_argument("--no-register", action="store_true", help="Don't register in zoo")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def find_mt_checkpoint() -> Path | None:
    """Find the best multi-topology checkpoint from zoo."""
    entries = _load_manifest()
    mt_entries = [
        e for e in entries
        if e.topology == "multi_topology" and e.n_qubits == 0
        and (_CHECKPOINTS_DIR / e.checkpoint_file).exists()
    ]
    if not mt_entries:
        return None
    best = max(mt_entries, key=lambda e: e.n_training_points)
    return _CHECKPOINTS_DIR / best.checkpoint_file


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-5s %(message)s",
    )

    print("=" * 65)
    print(f"  Fine-tune MT → {args.topology}")
    print("=" * 65)

    # Load source MT model
    if args.source_checkpoint:
        source_path = Path(args.source_checkpoint)
    else:
        source_path = find_mt_checkpoint()

    if source_path is None or not source_path.exists():
        print("ERROR: No multi-topology checkpoint found. Train one first.")
        return 1

    print(f"  Source: {source_path.name}")
    model = load_unified_checkpoint(str(source_path), eval_mode=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: hidden={model.hidden_dim}, layers={model.n_layers}, "
          f"residual={model.use_residual}, film={model.film_conditioning}")
    print(f"  Parameters: {n_params:,}")

    # Build topology-specific dataset
    print(f"\n  Aggregating {args.topology} data (max_n={args.max_n})...")
    agg = MultiNAggregator(topology=args.topology, model="tfim_bond_resolved", max_n=args.max_n)
    summary = agg.scan()
    if not summary:
        print(f"ERROR: No data for {args.topology}")
        return 1

    dataset = agg.build_combined_dataset(max_de_gap=args.max_de_gap)
    if len(dataset) < 5:
        print(f"ERROR: Only {len(dataset)} graphs after filtering (need ≥5)")
        return 1

    print(f"  Dataset: {len(dataset)} graphs, N={sorted(summary.keys())}")

    # Dataset fingerprint for provenance
    from qmbp_simulation.utils.helpers import compute_dataset_fingerprint
    _fp = compute_dataset_fingerprint(dataset)
    print(f"  Dataset fingerprint: {_fp}")

    # Fine-tune
    print(f"\n  Fine-tuning ({args.epochs} epochs, lr={args.lr})...")
    t0 = time.perf_counter()
    _training_interrupted = False
    try:
        result = fine_tune_unified_mpnn(
            model, dataset,
            n_epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
        )
    except KeyboardInterrupt:
        _training_interrupted = True
        print("\n  ⚠️  Fine-tune interrupted! Saving partial model to _recovery/...")
        try:
            from qmbp_simulation.predictors.unified_mpnn import save_unified_checkpoint

            _ROOT = Path(__file__).resolve().parents[3]
            recovery_dir = _ROOT / "data" / "model_zoo" / "checkpoints" / "_recovery"
            recovery_dir.mkdir(parents=True, exist_ok=True)
            recovery_path = recovery_dir / f"interrupted_finetune_{args.topology}.pt"
            save_unified_checkpoint(model, str(recovery_path))
            print(f"  Recovery checkpoint: {recovery_path}")
        except Exception as _save_err:
            print(f"  Recovery save failed: {_save_err}")
        result = {
            "final_mse": float("inf"),
            "val_mse": None,
            "n_epochs_run": 0,
            "stop_reason": "interrupted",
        }
    elapsed = time.perf_counter() - t0

    if _training_interrupted:
        print(f"\n  Training interrupted after {elapsed:.1f}s.")
        print(f"  Resume from: data/model_zoo/checkpoints/_recovery/interrupted_finetune_{args.topology}.pt")
        print("=" * 65)
        return 1

    final_mse = result.get("final_mse", 0)
    print(f"  Done: MSE={final_mse:.2e}, epochs={result.get('n_epochs_run', 0)}, "
          f"time={elapsed:.1f}s")

    # Persist training curve
    try:
        from qmbp_simulation.utils.helpers import persist_training_curve
        curve = persist_training_curve(
            result,
            output_dir=Path(__file__).resolve().parents[3] / "results" / "training_curves",
            prefix=f"finetune_{args.topology}",
        )
        if curve:
            print(f"  Training curve: {curve.name}")
    except Exception:
        pass

    # Register
    if not args.no_register:
        n_str = "+".join(str(n) for n in sorted(summary.keys()))
        ckpt_name = f"unified_tfim_br_{args.topology}_fromMT_{n_str}_p{args.p_layers}.pt"

        # ── Save experiment JSON envelope for traceability ────────────────
        _run_json_path = ""
        try:
            from qmbp_simulation.framework.result_io import (
                build_result_envelope,
                save_experiment_result,
            )

            _ROOT = Path(__file__).resolve().parents[3]
            envelope = build_result_envelope(
                config={
                    "runner": "run_finetune_from_mt",
                    "topology": args.topology,
                    "source_checkpoint": source_path.name,
                    "max_n": args.max_n,
                    "max_de_gap": args.max_de_gap,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "patience": args.patience,
                    "dataset_fingerprint": _fp,
                    "n_graphs": len(dataset),
                    "n_values": sorted(summary.keys()),
                },
                results={
                    "final_mse": final_mse,
                    "n_epochs_run": result.get("n_epochs_run"),
                    "stop_reason": result.get("stop_reason"),
                    "val_mse": result.get("val_mse"),
                },
                summary={
                    "topology": args.topology,
                    "source_model": source_path.name,
                    "n_graphs": len(dataset),
                    "training_time_s": elapsed,
                },
                elapsed_s=elapsed,
            )
            output_path = save_experiment_result(
                envelope,
                experiment_id=f"finetune_from_mt/{args.topology}",
                results_dir=_ROOT / "results" / "experiments",
            )
            _run_json_path = str(output_path.relative_to(_ROOT))
            print(f"  📄 Experiment JSON: {_run_json_path}")
        except Exception as _e:
            print(f"  ⚠️ Envelope save failed (non-blocking): {_e}")

        entry = ZooEntry(
            model="tfim_bond_resolved",
            topology=args.topology,
            n_qubits=0,
            p_layers=args.p_layers,
            checkpoint_file=ckpt_name,
            h_range=(0.5, 5.0),
            pass_rate=0.0,
            n_training_points=len(dataset),
            seeds=[42],
            created=datetime.now(timezone.utc).isoformat(),
            notes=(f"Fine-tuned from MT ({source_path.name[:30]}). "
                   f"MSE={final_mse:.2e}, {len(dataset)} graphs."),
        )

        register_checkpoint_with_training_metrics(
            model, entry,
            training_result=result,
            overwrite=True,
            architecture_config={
                "hidden_dim": model.hidden_dim,
                "n_conv_layers": model.n_layers,
                "use_residual": model.use_residual,
                "readout_mode": model.readout_mode,
                "film_conditioning": model.film_conditioning,
                "fine_tuned_from": source_path.name,
                "dataset_fingerprint": _fp,
            },
            optimizer_config={"learning_rate": args.lr, "patience": args.patience},
            auto_diagnose=True,
            run_json_path=_run_json_path,
        )
        print(f"\n  Registered: {entry.checkpoint_file}")

        # ── Quick post-evaluation: compute pass_rate on training data ────
        print("  Running quick evaluation...")
        try:
            model.eval()
            import torch

            n_pass = 0
            n_total = 0
            with torch.no_grad():
                for g in dataset:
                    pred = model(g).squeeze(0)
                    target = g.y
                    if len(pred) != len(target):
                        continue
                    mse = torch.nn.functional.mse_loss(pred, target).item()
                    n_total += 1
                    if mse < 0.01:  # Per-graph MSE < 0.01 threshold
                        n_pass += 1

            if n_total > 0:
                quick_pass_rate = n_pass / n_total
                print(f"  Quick pass_rate: {quick_pass_rate:.0%} ({n_pass}/{n_total} graphs)")

                # Auto-update zoo pass_rate
                from qmbp_simulation.predictors.model_zoo import update_zoo_pass_rate
                update_zoo_pass_rate(ckpt_name, quick_pass_rate)
                print(f"  Zoo pass_rate updated: {quick_pass_rate:.2f}")
        except Exception as eval_err:
            print(f"  ⚠️  Quick evaluation failed (non-blocking): {eval_err}")

    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
