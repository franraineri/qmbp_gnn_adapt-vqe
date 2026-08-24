#!/usr/bin/env python3
"""Architecture ablation: compare UnifiedMPNN variants on shared data.

Trains N variants (same data, same seed) and reports MSE + val_MSE.
Also loads the current zoo model as a reference ("zoo_current") so results
are directly comparable to production.

Integrations:
- Uses MultiNAggregator with exclusion policy (same as runner_base)
- Includes zoo model as reference (no retraining needed for it)
- Respects training_exclusions.json (hard failure mode N-level filtering)
- Optionally registers the best variant in the zoo (--register-best)
- Saves structured JSON for downstream analysis

Variants:
  1. baseline       — current architecture (no enhancements)
  2. residual       — skip connections between GINConv layers
  3. jk_cat         — Jumping Knowledge (concatenate all layer embeddings)
  4. film           — FiLM conditioning (modulate layers by h value)
  5. res+jk+film    — all three combined
  +  zoo_current    — current zoo model (loaded, not retrained — reference)

Usage:
    .venv/bin/python scripts/experiment_runners/cross_topology/run_arch_ablation.py
    .venv/bin/python scripts/experiment_runners/cross_topology/run_arch_ablation.py --topology ladder
    .venv/bin/python scripts/experiment_runners/cross_topology/run_arch_ablation.py --epochs 2000 --register-best
    .venv/bin/python scripts/experiment_runners/cross_topology/run_arch_ablation.py --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator, MultiTopologyAggregator
from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN, train_unified_mpnn

logger = logging.getLogger(__name__)


def _compute_dataset_fingerprint(dataset: list) -> str:
    """Compute a stable fingerprint for a PyG dataset.

    Delegates to the shared utility in qmbp_simulation.utils.helpers.
    """
    from qmbp_simulation.utils.helpers import compute_dataset_fingerprint

    return compute_dataset_fingerprint(dataset)


# Architecture variants to compare
VARIANTS = [
    {"name": "baseline", "use_residual": False, "readout_mode": "last", "film_conditioning": False},
    {"name": "residual", "use_residual": True, "readout_mode": "last", "film_conditioning": False},
    {"name": "jk_cat", "use_residual": False, "readout_mode": "jk_cat", "film_conditioning": False},
    {"name": "film", "use_residual": False, "readout_mode": "last", "film_conditioning": True},
    {
        "name": "res+jk+film",
        "use_residual": True,
        "readout_mode": "jk_cat",
        "film_conditioning": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Architecture ablation study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--topology", default="chain_1d", help="Topology to use (default: chain_1d)"
    )
    parser.add_argument("--max-n", type=int, default=20, help="Max N for training data")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dim (same for all)")
    parser.add_argument("--n-layers", type=int, default=3, help="GNN layers (same for all)")
    parser.add_argument("--epochs", type=int, default=2000, help="Max epochs per variant")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (same for all)")
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Number of seeds for statistical significance (default: 1, use 3+ for thesis claims)",
    )
    parser.add_argument("--quick", action="store_true", help="Quick mode: 500 epochs, hidden=64")
    parser.add_argument(
        "--register-best", action="store_true", help="Register the best variant in the zoo"
    )
    parser.add_argument(
        "--skip-zoo-ref", action="store_true", help="Skip loading zoo model as reference"
    )
    parser.add_argument(
        "--multi-topology",
        action="store_true",
        help="Use combined data from ALL topologies (MultiTopologyAggregator)",
    )
    parser.add_argument(
        "--max-de-gap", type=float, default=0.10, help="Quality filter threshold (default: 0.10)"
    )
    parser.add_argument(
        "--auto-compare",
        action="store_true",
        help="After registration, run model_comparison to validate with ΔE/gap",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def apply_exclusion_policy(agg: MultiNAggregator, topology: str) -> None:
    """Apply N-level exclusion policy (same logic as runner_base)."""
    try:
        from qmbp_simulation.analysis.metrics import load_training_exclusions

        registry = load_training_exclusions()
        hard_modes = {"contaminated_training", "gap_masking", "intrinsic_vqe_error"}
        for entry in registry.get("excluded", []):
            if entry.get("topology") == topology and entry.get("failure_mode") in hard_modes:
                n_val = entry.get("n_qubits", 0)
                if n_val > 0 and n_val in agg._data_by_n:
                    n_pts = len(agg._data_by_n.pop(n_val))
                    logger.info("  Exclusion policy: N=%d removed (%d pts)", n_val, n_pts)
    except Exception:
        pass


def evaluate_zoo_model(topology: str, dataset: list, val_dataset: list) -> dict | None:
    """Load the current zoo model and evaluate on the same val set."""
    try:
        from qmbp_simulation.predictors.model_zoo import load_best_model_for

        mpnn, entry, _source = load_best_model_for(
            topology,
            model="tfim_bond_resolved",
            n_target=20,
            p_layers=1,
        )
        mpnn.eval()

        # Evaluate on val set (same as training variants)
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for data in val_dataset:
                pred = mpnn(data).squeeze(0)
                target = data.y
                if len(pred) != len(target):
                    continue
                loss = torch.nn.functional.mse_loss(pred, target)
                val_loss += loss.item()
                n_val += 1

        if n_val == 0:
            return None

        n_params = sum(p.numel() for p in mpnn.parameters())
        return {
            "name": f"zoo_current ({entry.checkpoint_file[:30]}...)",
            "n_params": n_params,
            "final_mse": None,
            "final_zz_mse": None,
            "final_x_mse": None,
            "val_mse": val_loss / n_val,
            "gen_gap": None,
            "n_epochs": 0,
            "stop_reason": "loaded_from_zoo",
            "elapsed_s": 0,
            "checkpoint": entry.checkpoint_file,
            "pass_rate": entry.pass_rate,
        }
    except (FileNotFoundError, Exception) as e:
        logger.info("  Zoo model not available: %s", e)
        return None


def train_variant(
    variant: dict, dataset: list, args: argparse.Namespace, seed: int | None = None
) -> dict:
    """Train a single architecture variant and return metrics."""
    _seed = seed if seed is not None else args.seed
    torch.manual_seed(_seed)
    np.random.seed(_seed)

    hidden = 64 if args.quick else args.hidden_dim
    epochs = 500 if args.quick else args.epochs
    feat_dim = dataset[0].x.shape[1]

    model = UnifiedMPNN(
        node_features=feat_dim,
        hidden_dim=hidden,
        n_layers=args.n_layers,
        norm_type="none",
        dropout=0.1,
        type_embedding_dim=16,
        gate_readout=True,
        use_residual=variant["use_residual"],
        readout_mode=variant["readout_mode"],
        film_conditioning=variant["film_conditioning"],
    )

    n_params = sum(p.numel() for p in model.parameters())
    t0 = time.perf_counter()

    result = train_unified_mpnn(
        model,
        dataset,
        n_epochs=epochs,
        lr=1e-3,
        patience=200,
        seed=_seed,
        weight_decay=1e-4,
        val_fraction=0.15,
    )

    elapsed = time.perf_counter() - t0

    return {
        "name": variant["name"],
        "n_params": n_params,
        "final_mse": result["final_mse"],
        "final_zz_mse": result["final_zz_mse"],
        "final_x_mse": result["final_x_mse"],
        "val_mse": result["val_mse"],
        "gen_gap": result["generalization_gap"],
        "n_epochs": result["n_epochs_run"],
        "stop_reason": result["stop_reason"],
        "elapsed_s": round(elapsed, 1),
        "model": model,  # Keep for optional registration
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-5s %(message)s",
    )

    hidden = 64 if args.quick else args.hidden_dim
    epochs = 500 if args.quick else args.epochs

    print("=" * 70)
    print("  ARCHITECTURE ABLATION STUDY")
    topo_label = "ALL (multi-topology)" if args.multi_topology else args.topology
    print(f"  Topology: {topo_label} | max_N: {args.max_n}")
    print(f"  Hidden: {hidden} | Layers: {args.n_layers} | Epochs: {epochs}")
    print("=" * 70)

    # ── Build dataset (shared across all variants) ────────────────────────
    print("\nAggregating training data...")

    if args.multi_topology:
        mt_agg = MultiTopologyAggregator(model="tfim_bond_resolved", max_n=args.max_n)
        mt_agg.scan()
        dataset = mt_agg.build_combined_dataset(max_de_gap=args.max_de_gap)
        if not dataset:
            print("ERROR: no multi-topology data available")
            return 1
        summary = {f"MT({len(mt_agg._aggregators)} topos)": len(dataset)}
    else:
        agg = MultiNAggregator(topology=args.topology, model="tfim_bond_resolved", max_n=args.max_n)
        summary = agg.scan()
        if not summary:
            print("ERROR: no data available")
            return 1

        # Apply exclusion policy (same as runner_base)
        apply_exclusion_policy(agg, args.topology)
        summary = {n: len(pts) for n, pts in agg._data_by_n.items()}
        if not summary:
            print("ERROR: all data excluded by policy")
            return 1

        dataset = agg.build_combined_dataset(max_de_gap=args.max_de_gap)
    if len(dataset) < 10:
        print(f"ERROR: only {len(dataset)} graphs — need at least 10")
        return 1

    total_pts = sum(summary.values())
    print(f"  N values: {sorted(summary.keys())}")
    print(f"  Total raw: {total_pts} points -> {len(dataset)} training graphs")

    # ── Create a fixed val split for zoo comparison ───────────────────────
    rng = np.random.default_rng(args.seed)
    n_val = int(len(dataset) * 0.15)
    indices = rng.permutation(len(dataset))
    val_indices = set(indices[:n_val].tolist())
    val_dataset = [dataset[i] for i in range(len(dataset)) if i in val_indices]
    print(f"  Val split: {len(val_dataset)} graphs (for zoo comparison)")

    # ── Zoo reference model ───────────────────────────────────────────────
    results = []
    if not args.skip_zoo_ref and not args.multi_topology:
        print("\n  Loading zoo reference model...", end="", flush=True)
        zoo_result = evaluate_zoo_model(args.topology, dataset, val_dataset)
        if zoo_result:
            results.append(zoo_result)
            print(
                f" val_MSE={zoo_result['val_mse']:.2e} (pass={zoo_result.get('pass_rate', 0):.0%})"
            )
        else:
            print(" not available")
    elif args.multi_topology:
        print("\n  [multi-topology mode: zoo reference skipped]")

    # ── Train each variant ────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(
        f"  {'Variant':<28} {'Params':>8} {'MSE':>10} {'Val MSE':>10} "
        f"{'GenGap':>8} {'Epochs':>6} {'Time':>6}"
    )
    print(f"{'─' * 70}")

    # Print zoo reference first if available
    for r in results:
        val_str = f"{r['val_mse']:.2e}" if r["val_mse"] is not None else "N/A"
        print(
            f"  {r['name']:<28} {r['n_params']:>8,} {'(ref)':>10} "
            f"{val_str:>10} {'—':>8} {'—':>6} {'0':>5}s"
        )

    trained_results = []
    seeds = [args.seed + i * 100 for i in range(args.n_seeds)]

    for variant in VARIANTS:
        if args.n_seeds == 1:
            # Single seed — original behavior
            print(f"  Training {variant['name']}...", end="", flush=True)
            r = train_variant(variant, dataset, args, seed=seeds[0])
            trained_results.append(r)

            gen_gap_str = f"{r['gen_gap']:.2e}" if r["gen_gap"] is not None else "N/A"
            val_str = f"{r['val_mse']:.2e}" if r["val_mse"] is not None else "N/A"
            print(
                f"\r  {r['name']:<28} {r['n_params']:>8,} {r['final_mse']:>10.2e} "
                f"{val_str:>10} {gen_gap_str:>8} {r['n_epochs']:>6} {r['elapsed_s']:>5.0f}s"
            )
        else:
            # Multi-seed: train K times, report mean ± std
            print(f"  Training {variant['name']} ({args.n_seeds} seeds)...", end="", flush=True)
            seed_results = []
            for s in seeds:
                sr = train_variant(variant, dataset, args, seed=s)
                seed_results.append(sr)

            # Aggregate: mean ± std of val_mse
            val_mses = [sr["val_mse"] for sr in seed_results if sr["val_mse"] is not None]
            final_mses = [sr["final_mse"] for sr in seed_results]
            mean_val = float(np.mean(val_mses)) if val_mses else None
            std_val = float(np.std(val_mses)) if len(val_mses) > 1 else 0.0
            mean_mse = float(np.mean(final_mses))

            # Use best seed as the representative model
            best_seed_result = min(
                seed_results, key=lambda x: x["val_mse"] if x["val_mse"] else float("inf")
            )
            aggregated = {
                **best_seed_result,
                "val_mse": mean_val,
                "val_mse_std": std_val,
                "final_mse": mean_mse,
                "final_mse_std": float(np.std(final_mses)),
                "n_seeds": args.n_seeds,
                "seeds_used": seeds,
                "per_seed_val_mse": val_mses,
            }
            trained_results.append(aggregated)

            val_str = f"{mean_val:.2e}±{std_val:.1e}" if mean_val else "N/A"
            print(
                f"\r  {variant['name']:<28} {best_seed_result['n_params']:>8,} "
                f"{mean_mse:>10.2e} {val_str:>14} "
                f"{'—':>8} {best_seed_result['n_epochs']:>6} {best_seed_result['elapsed_s']:>5.0f}s"
            )

    # ── Convergence warnings ──────────────────────────────────────────────
    completed_variants = [r for r in trained_results if r["stop_reason"] == "completed"]
    if completed_variants:
        names = [r["name"] for r in completed_variants]
        print(
            f"\n  ⚠️  {len(completed_variants)} variant(s) hit max epochs without convergence: "
            f"{', '.join(names)}"
        )
        print("      Consider increasing --epochs or adding more training data.")

    # Combine results (zoo + trained)
    all_results_for_json = results + [
        {k: v for k, v in r.items() if k != "model"} for r in trained_results
    ]

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    scoreable = [r for r in all_results_for_json if r.get("val_mse") is not None]
    if scoreable:
        best = min(scoreable, key=lambda r: r["val_mse"])
        baseline_result = next((r for r in trained_results if r["name"] == "baseline"), None)
        baseline_mse = (
            baseline_result["val_mse"] if baseline_result and baseline_result["val_mse"] else None
        )

        print(f"\n  Best variant: {best['name']}")
        print(f"  Best val_MSE: {best['val_mse']:.2e}")
        if baseline_mse and best["val_mse"]:
            improvement = (baseline_mse - best["val_mse"]) / baseline_mse * 100
            print(f"  Improvement over baseline: {improvement:+.1f}%")

        # Compare to zoo
        zoo_ref = next((r for r in results if "zoo_current" in r.get("name", "")), None)
        if zoo_ref and zoo_ref["val_mse"]:
            vs_zoo = (zoo_ref["val_mse"] - best["val_mse"]) / zoo_ref["val_mse"] * 100
            print(f"  Improvement over zoo model: {vs_zoo:+.1f}%")

    # ── Register best if requested ────────────────────────────────────────
    if args.register_best and trained_results:
        best_trained = min(
            trained_results, key=lambda r: r["val_mse"] if r["val_mse"] else float("inf")
        )
        if best_trained.get("model") is not None:
            from qmbp_simulation.predictors.model_zoo import (
                ZooEntry,
                _load_manifest,
                register_checkpoint_with_training_metrics,
            )

            variant_cfg = next(v for v in VARIANTS if v["name"] == best_trained["name"])
            arch_parts = []
            if variant_cfg["use_residual"]:
                arch_parts.append("residual")
            if variant_cfg["readout_mode"] != "last":
                arch_parts.append(variant_cfg["readout_mode"])
            if variant_cfg["film_conditioning"]:
                arch_parts.append("film")
            arch_label = "+".join(arch_parts) if arch_parts else "baseline"

            ckpt_name = (
                f"unified_tfim_br_{args.topology if not args.multi_topology else 'multi_topology'}"
                f"_multiN_ablation_{arch_label}_p1.pt"
            )

            # ── Pre-flight: check if existing model would be overwritten ──
            existing_entries = [e for e in _load_manifest() if e.checkpoint_file == ckpt_name]
            if existing_entries:
                existing = existing_entries[0]
                print(f"\n  Pre-flight: existing model found ({ckpt_name[:50]}...)")
                print(
                    f"    Existing: pass_rate={existing.pass_rate:.2f}, "
                    f"pts={existing.n_training_points}"
                )
                print(f"    New:      val_MSE={best_trained['val_mse']:.2e}, pts={len(dataset)}")
                print("    Anti-regression: old model auto-backed up to _versions/ + _best/")

            entry = ZooEntry(
                model="tfim_bond_resolved",
                topology=args.topology if not args.multi_topology else "multi_topology",
                n_qubits=0,
                p_layers=1,
                checkpoint_file=ckpt_name,
                h_range=(0.5, 5.0),
                pass_rate=0.0,
                n_training_points=len(dataset),
                seeds=[args.seed],
                created=datetime.now(UTC).isoformat(),
                notes=(
                    f"Ablation best ({arch_label}): MSE={best_trained['final_mse']:.2e}, "
                    f"val={best_trained['val_mse']:.2e}"
                ),
            )
            register_checkpoint_with_training_metrics(
                best_trained["model"],
                entry,
                training_result=best_trained,
                overwrite=True,
                architecture_config={
                    "hidden_dim": args.hidden_dim,
                    "n_conv_layers": args.n_layers,
                    "use_residual": variant_cfg["use_residual"],
                    "readout_mode": variant_cfg["readout_mode"],
                    "film_conditioning": variant_cfg["film_conditioning"],
                    "arch_label": arch_label,
                },
            )
            print(f"\n  Registered best variant: {ckpt_name}")

    # ── Save results ──────────────────────────────────────────────────────
    output_dir = ROOT / "results" / "arch_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"ablation_{args.topology}_{timestamp}.json"

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "topology": "multi_topology" if args.multi_topology else args.topology,
        "multi_topology_mode": args.multi_topology,
        "max_n": args.max_n,
        "max_de_gap": args.max_de_gap,
        "n_training_graphs": len(dataset),
        "dataset_fingerprint": _compute_dataset_fingerprint(dataset),
        "hidden_dim": hidden,
        "n_layers": args.n_layers,
        "max_epochs": epochs,
        "seed": args.seed,
        "n_seeds": args.n_seeds,
        "exclusion_policy_applied": not args.multi_topology,
        "results": all_results_for_json,
        "best_variant": best["name"] if scoreable else None,
    }
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Results saved: {output_file.relative_to(ROOT)}")

    # ── Auto-compare: validate best variant with real ΔE/gap evaluation ──
    if args.auto_compare and args.register_best:
        topo_for_compare = "chain_1d" if args.multi_topology else args.topology
        print(f"\n  Running model_comparison on {topo_for_compare} N=20...")
        import subprocess

        cmd = [
            sys.executable,
            str(ROOT / "scripts/experiment_runners/cross_topology/run_model_comparison.py"),
            "--topology",
            topo_for_compare,
            "--target-n",
            "20",
            "--auto-detect",
            "--promote-best",
            "--h-points",
            "6",
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(ROOT),
            )
            # Show key results
            for line in proc.stdout.split("\n"):
                if any(k in line for k in ["Best:", "Winner:", "📊", "Pass%", "Grade"]):
                    print(f"    {line.strip()}")
            if proc.returncode != 0:
                print(f"    ⚠️  Comparison returned code {proc.returncode}")
        except (subprocess.TimeoutExpired, Exception) as e:
            print(f"    ⚠️  Auto-comparison skipped: {e}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
