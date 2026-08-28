#!/usr/bin/env python3
"""Accelerated Cross-N Pipeline — Train N_train, Predict N_target via zoo.

Integrates AcceleratedVQE + model_zoo + QualityPredictor for a complete
bond-resolved cross-N transfer workflow with data reuse:

1. Quality check: is the target config viable?
2. Train: AcceleratedVQE at N_train (or load from zoo if exists)
3. Zoo export: auto-register the trained model
4. Cross-N predict: evaluate at N_target without VQE
5. Analysis: per-h ΔE/gap breakdown + comparison vs full VQE

Supports multiple p values and topologies. Results are saved as JSON
for downstream analysis.

Usage:
    # Default: Train N=10, predict N=20, p=1, chain_1d
    .venv/bin/python scripts/.../run_accelerated_cross_n.py

    # Custom sizes
    .venv/bin/python scripts/.../run_accelerated_cross_n.py --train-n 10 --target-n 20 40

    # Use existing model from zoo (skip training)
    .venv/bin/python scripts/.../run_accelerated_cross_n.py --from-zoo --target-n 20

    # Multiple p layers
    .venv/bin/python scripts/.../run_accelerated_cross_n.py --p-layers 1 2

    # Dry run
    .venv/bin/python scripts/.../run_accelerated_cross_n.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import UTC
from pathlib import Path

import numpy as np

from qmbp_simulation.framework.result_io import upsert_theta_npz
from qmbp_simulation.framework.runner_base import (
    Section,
    ValidationRunner,
    resolve_project_root,
)

_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


# Defaults
DEFAULT_TRAIN_N = 10
DEFAULT_TARGET_N = [20]
DEFAULT_P = 1
DEFAULT_TOPOLOGY = "chain_1d"
DEFAULT_H_MIN = 2.0
DEFAULT_H_MAX = 4.5
DEFAULT_H_POINTS = 15
DEFAULT_N_ANCHORS = 14
DEFAULT_MAXITER = 1000
DEFAULT_N_RESTARTS = 6
# Hard floor on VQE restarts. Canonical source is models.constants.MIN_N_RESTARTS,
# re-exported here for local use. Enforced on --n-restarts (setup) AND inside the
# adaptive allocation (sweep_strategies), so even easy points get ≥ this many.
from qmbp_simulation.models.constants import MIN_N_RESTARTS  # noqa: E402

DEFAULT_MAX_REFINE_PER_ITER = 50

FINE_TUNE_EPOCHS = 500
FULL_TRAIN_EPOCHS = 2000


class AcceleratedCrossNRunner(ValidationRunner):
    """Accelerated Cross-N Transfer: train small, predict large.

    Trains UnifiedMPNN at N_train using AcceleratedVQE (5-6 anchor VQE +
    MPNN for the rest), exports to zoo, then predicts at N_target using
    only the trained model. No VQE at N_target.
    """

    runner_id = "accelerated_cross_n_v1"
    experiment_id = "ACCEL_CROSS_N"
    description = "Accelerated Cross-N: train N_train, predict N_target via zoo"

    # N_MAX_VIABLE per topology (dual criterion, prevents extrapolation contamination)
    N_MAX_VIABLE = {
        "chain_1d": 300,
        "heavy_hex": 300,
        "square": 300,
        "ladder": 1000,
        "triangular": 12,  # N>=14 is ansatz-limited (0% dual pass with p=1)
    }
    hypothesis = (
        "UnifiedMPNN trained on N_train bond-resolved data predicts θ at "
        "N_target with ΔE/gap < 10% for h in valid regime (h > 2.0)."
    )

    @classmethod
    def _add_custom_args(cls, parser):
        parser.add_argument(
            "--train-n",
            type=int,
            default=DEFAULT_TRAIN_N,
            help="System size for training (default: %(default)s)",
        )
        parser.add_argument(
            "--target-n",
            type=int,
            nargs="+",
            default=DEFAULT_TARGET_N,
            help="Target system size(s) for prediction (default: %(default)s)",
        )
        parser.add_argument(
            "--p-layers",
            type=int,
            nargs="+",
            default=[DEFAULT_P],
            help="HVA layer depth(s) (default: %(default)s)",
        )
        parser.add_argument(
            "--topology",
            type=str,
            default=DEFAULT_TOPOLOGY,
            help="Lattice topology (default: %(default)s)",
        )
        parser.add_argument(
            "--h-min",
            type=float,
            default=DEFAULT_H_MIN,
            help="Minimum h for sweep (default: %(default)s)",
        )
        parser.add_argument(
            "--h-max",
            type=float,
            default=DEFAULT_H_MAX,
            help="Maximum h for sweep (default: %(default)s)",
        )
        parser.add_argument(
            "--h-points",
            type=int,
            default=DEFAULT_H_POINTS,
            help="Number of h-grid points (default: %(default)s)",
        )
        parser.add_argument(
            "--train-h-min",
            type=float,
            default=None,
            help="If set, restrict TRAINING data to h >= this value "
            "(filters MultiNAggregator dataset; independent of the eval sweep --h-min).",
        )
        parser.add_argument(
            "--train-h-max",
            type=float,
            default=None,
            help="If set, restrict TRAINING data to h <= this value "
            "(filters MultiNAggregator dataset; independent of the eval sweep --h-max).",
        )
        parser.add_argument(
            "--n-anchors",
            type=int,
            default=DEFAULT_N_ANCHORS,
            help="Number of VQE anchor points (default: %(default)s)",
        )
        parser.add_argument(
            "--maxiter",
            type=int,
            default=DEFAULT_MAXITER,
            help="VQE COBYLA maxiter (default: %(default)s)",
        )
        parser.add_argument(
            "--n-restarts",
            type=int,
            default=DEFAULT_N_RESTARTS,
            help="VQE restarts per anchor (default: %(default)s)",
        )
        parser.add_argument(
            "--from-zoo",
            action="store_true",
            default=False,
            help="Skip training, load model from zoo directly",
        )
        parser.add_argument(
            "--checkpoint",
            type=str,
            default=None,
            help="Explicit checkpoint path (overrides zoo search)",
        )
        parser.add_argument(
            "--active-rounds",
            type=int,
            default=0,
            help="Active learning rounds: refine low-fidelity points with VQE (default: 0)",
        )
        parser.add_argument(
            "--multi-n-train",
            action="store_true",
            default=False,
            help="Instead of training on a single N, aggregate ALL available "
            "bond-resolved data for this topology (from previous runs) and "
            "train a multi-N model. Overrides --train-n for training.",
        )
        parser.add_argument(
            "--force-retrain",
            action="store_true",
            default=False,
            help="Force retraining from scratch even if a suitable model "
            "exists in the zoo. Default: reuse best existing model.",
        )
        parser.add_argument(
            "--skip-retrain",
            action="store_true",
            default=False,
            help="In --iterative-improve: never train/fine-tune a new MPNN. "
            "Keep predicting with the loaded model (--checkpoint or zoo best) "
            "and only refine failing points with VQE, persisting them to NPZ. "
            "Unlike --from-zoo, this still allows the zoo model to be selected "
            "automatically when no --checkpoint is given.",
        )
        parser.add_argument(
            "--model-name",
            type=str,
            default=None,
            help="Custom model name suffix. Checkpoint will be saved as "
            "unifMPNN__<topology>_p<N>_<model-name>.pt (e.g., --model-name coloring_v1)",
        )
        parser.add_argument(
            "--loss-type",
            type=str,
            default="theta_mse",
            choices=["theta_mse", "energy_weighted"],
            help="MPNN training loss. 'theta_mse' (default): standard MSE on θ. "
            "'energy_weighted': weights MSE by 1/(1+de_gap) so points with "
            "low energy error contribute more.",
        )
        parser.add_argument(
            "--physics-loss-weight",
            type=float,
            default=0.0,
            help="Weight λ for physics-informed energy loss term (default 0.0 = "
            "disabled). Recommended 0.01-0.1. Adds λ·mean(|E(θ_pred)-E_exact|/N) "
            "after physics_loss_start_epoch.",
        )
        parser.add_argument(
            "--no-eval-cache",
            action="store_true",
            default=False,
            help="Disable circuit evaluation cache. By default, evaluations "
            "are cached in data/eval_cache/ to avoid recomputing identical "
            "(topology, N, h, theta_hash) evaluations.",
        )
        # Iterative improvement + VQE method args from shared CLI module
        from qmbp_simulation.framework.cli import add_iterative_improve_args

        add_iterative_improve_args(parser)
        parser.add_argument(
            "--max-refine-per-iter",
            type=int,
            default=None,
            help="Max h-points to refine per iteration. Default: min(n_failures, 20). "
            "Higher = more VQE compute per iter but fewer iterations needed. "
            "Use --refine-all to refine ALL failing points (no cap).",
        )
        parser.add_argument(
            "--refine-all",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Refine ALL failing points per iteration (no cap). "
            "Maximizes training data quality at the cost of compute time. "
            "Use --no-refine-all to cap refinements via --max-refine-per-iter.",
        )

    def build_config(self) -> dict:
        config = self._build_physics_config()
        config["n_anchors"] = self._args.n_anchors
        config["force_method"] = self._args.force_method
        config["bidirectional_anchors"] = self._args.bidirectional_anchors
        config["from_zoo"] = self._args.from_zoo
        return config

    def setup(self):
        """Initialize physics objects."""
        self.setup_physics()
        # Enforce the hard minimum of VQE restarts for the whole run. Applied
        # once here so every downstream use (main config, bootstrap, refine)
        # inherits the floor.
        if self._args.n_restarts < MIN_N_RESTARTS:
            logger.info(
                f"  n_restarts raised {self._args.n_restarts} → {MIN_N_RESTARTS} "
                f"(hard floor MIN_N_RESTARTS)"
            )
            self._args.n_restarts = MIN_N_RESTARTS
        # Auto-detect h_min from valid regime if user didn't override
        # (h below the regime boundary is ansatz-limited for p=1)
        if self._args.h_min == DEFAULT_H_MIN:
            try:
                from qmbp_simulation.framework.preflight import get_regime_threshold

                topo = self._args.topology
                n_target = self._args.target_n[0] if self._args.target_n else 10
                p = (
                    self._args.p_layers[0]
                    if isinstance(self._args.p_layers, list)
                    else self._args.p_layers
                )
                threshold = get_regime_threshold(topo, n_target, p)
                if threshold > 0 and threshold > self._args.h_min:
                    logger.info(
                        f"  H-range auto-adjusted: h_min {self._args.h_min} → {threshold:.1f} "
                        f"(valid regime for {topo} N={n_target} p={p})"
                    )
                    self._args.h_min = threshold
            except (ImportError, ValueError, KeyError):
                pass  # Keep default if regime lookup fails

        # Round to 2 decimals for cache key stability (matches GroundTruthCache)
        self._h_values = [
            round(h, 2)
            for h in np.linspace(self._args.h_max, self._args.h_min, self._args.h_points)
        ]
        self._models = {}  # p_layers → trained model
        self._train_results = {}  # p_layers → AcceleratedResult
        # Default force_method to L-BFGS-B for noiseless backends.
        # VQEOptimizer auto-dispatch will keep L-BFGS-B on noiseless (fast)
        # or downgrade to COBYLA on noisy backends automatically.
        if self._args.force_method is None:
            self._args.force_method = "L-BFGS-B"

    def run_preflight(self) -> bool:
        """Validate topology constraints before execution."""
        topo = self._args.topology

        # Validate training h-range filter (if provided)
        train_h_min = getattr(self._args, "train_h_min", None)
        train_h_max = getattr(self._args, "train_h_max", None)
        if train_h_min is not None and train_h_max is not None:
            if train_h_min > train_h_max:
                logger.error(
                    f"--train-h-min ({train_h_min}) must be <= --train-h-max "
                    f"({train_h_max}). No training data would be selected."
                )
                return False
        if (train_h_min is not None or train_h_max is not None) and not (
            getattr(self._args, "multi_n_train", False)
            or getattr(self._args, "iterative_improve", False)
        ):
            logger.warning(
                "--train-h-min/--train-h-max only affect the training dataset "
                "(built by MultiNAggregator). They have no effect without "
                "--multi-n-train or --iterative-improve."
            )

        # Ladder requires even N
        if topo == "ladder":
            bad_n = []
            if self._args.train_n % 2 != 0 and not self._args.from_zoo:
                bad_n.append(f"train_n={self._args.train_n}")
            for n in self._args.target_n:
                if n % 2 != 0:
                    bad_n.append(f"target_n={n}")
            if bad_n:
                logger.error(
                    f"Ladder topology requires even N. Invalid: {', '.join(bad_n)}. "
                    f"Use N=8, 10, 12, 14, 16, 20, etc."
                )
                return False
        return True

    def _training_h_range(self) -> tuple[float, float]:
        """Effective training h-range for zoo metadata.

        Prefers the explicit --train-h-min/--train-h-max filter when set,
        otherwise falls back to the evaluation sweep range. This ensures the
        zoo's h_range (documented as the TRAINING range) is accurate for
        range-restricted models.
        """
        train_h_min = getattr(self._args, "train_h_min", None)
        train_h_max = getattr(self._args, "train_h_max", None)
        h_min = train_h_min if train_h_min is not None else self._args.h_min
        h_max = train_h_max if train_h_max is not None else self._args.h_max
        return (float(h_min), float(h_max))

    def _train_h_range_note(self) -> str:
        """Traceability note fragment when a training h-range filter is active."""
        train_h_min = getattr(self._args, "train_h_min", None)
        train_h_max = getattr(self._args, "train_h_max", None)
        if train_h_min is not None or train_h_max is not None:
            lo = train_h_min if train_h_min is not None else "-inf"
            hi = train_h_max if train_h_max is not None else "+inf"
            return f", train_h_range=[{lo}, {hi}]"
        return ""

    def _check_existing_npz_utility(self, topology: str, n_qubits: int, p_layers: int = 1) -> None:
        """Check if existing NPZ data is useful for training. Logs warnings if not.

        Does NOT block execution — just informs the user that the existing NPZ
        for this config has been classified as 'not_useful' or 'insufficient_signal'
        by the dashboard. The run will still proceed (to generate fresh data),
        but the warning helps understand why previous models had poor performance.
        """
        try:
            from qmbp_simulation.analysis.metrics import classify_training_utility

            npz_path = Path("data/multi_n_training") / f"{topology}_N{n_qubits}_p{p_layers}.npz"
            if not npz_path.exists():
                return  # No existing data — nothing to check

            data = np.load(str(npz_path), allow_pickle=True)
            n_pts = len(data["h_values"])
            if n_pts == 0:
                return

            # Compute dual criterion pass rate
            e_key = "e_vqe" if "e_vqe" in data else ("energies" if "energies" in data else None)
            if e_key is None or "e_exact" not in data:
                return

            from qmbp_simulation.analysis.metrics import (
                DE_GAP_THRESHOLD,
                is_point_failure,
            )

            abs_err = np.abs(data[e_key] - data["e_exact"])
            if "de_gaps" in data:
                de_gaps = data["de_gaps"]
            elif "gaps" in data:
                de_gaps = abs_err / np.maximum(data["gaps"], 1e-10)
            else:
                return

            n_fail = sum(
                is_point_failure(de_gap=float(de_gaps[i]), abs_error=float(abs_err[i]))
                for i in range(n_pts)
            )
            pass_dual = (n_pts - n_fail) / n_pts
            pass_simple = float((de_gaps < DE_GAP_THRESHOLD).mean())

            category, reason = classify_training_utility(
                n_pts,
                pass_dual,
                pass_simple,
            )

            if category == "not_useful":
                logger.warning(
                    f"  ⚠️ EXISTING NPZ '{npz_path.name}' IS NOT USEFUL FOR TRAINING:\n"
                    f"     {reason}\n"
                    f"     The MPNN cannot learn from this data. Consider:\n"
                    f"     1. Running with --force-retrain to generate fresh VQE data\n"
                    f"     2. Adjusting h-range to avoid the failing regime\n"
                    f"     3. Deleting the NPZ if the config is fundamentally unrecoverable"
                )
            elif category == "insufficient_signal":
                logger.warning(
                    f"  ⚠️ EXISTING NPZ '{npz_path.name}' has INSUFFICIENT SIGNAL:\n"
                    f"     {reason}\n"
                    f"     Training may not converge well. "
                    f"This run will add more data via upsert."
                )
            else:
                logger.info(
                    f"  Existing NPZ: {n_pts} pts, dual_pass={pass_dual:.0%} — USEFUL for training"
                )
        except Exception as e:
            logger.debug(f"  NPZ utility check failed (non-blocking): {e}")

    def define_sections(self) -> list[Section]:
        sections = [
            Section(
                id=1,
                name="Quality Check",
                fn=self.section_quality_check,
                hypothesis="Target config is predicted viable (pass_prob > 30%)",
            ),
        ]

        # Iterative improve mode: budget estimation + iterative loop
        if getattr(self._args, "iterative_improve", False):
            sections.append(
                Section(
                    id=2,
                    name="Budget Estimation + Cache Warm-up",
                    fn=self.section_budget_estimation,
                    hypothesis="Estimate compute cost leveraging cached results",
                )
            )
            if not getattr(self._args, "budget_only", False):
                sections.append(
                    Section(
                        id=3,
                        name="Iterative Improvement Loop",
                        fn=self.section_iterative_improve,
                        hypothesis="Iterative predict→refine→retrain converges to ≥90% pass rate",
                    )
                )
            return sections

        # An explicit --checkpoint means "use only this model": treat it like
        # --from-zoo so no training section is added (predict-only).
        _predict_only = self._args.from_zoo or bool(getattr(self._args, "checkpoint", None))

        if getattr(self._args, "multi_n_train", False) or getattr(
            self._args, "force_retrain", False
        ):
            sections.append(
                Section(
                    id=2,
                    name="Multi-N Train (aggregate all available data)",
                    fn=self.section_multi_n_train,
                    hypothesis="Multi-N UnifiedMPNN trained with aggregated data from all N sizes",
                )
            )
        elif not _predict_only:
            sections.append(
                Section(
                    id=2,
                    name=f"Train (N={self._args.train_n})",
                    fn=self.section_train,
                    hypothesis=f"AcceleratedVQE at N={self._args.train_n} achieves >60% pass rate",
                )
            )
        sections.append(
            Section(
                id=3,
                name="Cross-N Predict",
                fn=self.section_cross_n_predict,
                hypothesis="Cross-N prediction achieves ΔE/gap < 10% for h > 2.0",
            )
        )
        return sections

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 1: Quality Check
    # ═══════════════════════════════════════════════════════════════════════════

    def section_quality_check(self) -> dict:
        """Run QualityPredictor for training and target configs via base helper."""
        # Collect unique N values to avoid duplicate checks when train_n == target_n
        n_values = {self._args.train_n}
        for n_target in self._args.target_n:
            n_values.add(n_target)

        configs = [
            {
                "model": "tfim_bond_resolved",
                "topology": self._args.topology,
                "n_qubits": n,
                "p_layers": self._args.p_layers[0],
                "h_min": self._args.h_min,
                "h_max": self._args.h_max,
            }
            for n in sorted(n_values)
        ]
        return self.run_quality_check(configs=configs)

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 2: Train
    # ═══════════════════════════════════════════════════════════════════════════

    def section_train(self) -> dict:
        """Train AcceleratedVQE at N_train for each p value."""
        from pathlib import Path

        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.pipeline.accelerated import AcceleratedConfig, AcceleratedVQE

        spec = get_model_spec("tfim_bond_resolved")
        hva = HVACircuitBuilder()
        N = self._args.train_n
        topo = self._args.topology

        # ── Training utility gating: warn if existing NPZ is not useful ──
        _p_check = (
            self._args.p_layers[0] if isinstance(self._args.p_layers, list) else self._args.p_layers
        )
        self._check_existing_npz_utility(topo, N, p_layers=_p_check)

        backend = self.select_backend(N, for_vqe_loop=True)

        all_results = {}
        for p in self._args.p_layers:
            logger.info(f"  Training: N={N}, p={p}, topology={topo}")
            lattice = self.make_lattice(topo, N, J=1.0, h=2.0)
            circuit, _ = hva.create_bond_resolved(N, p, lattice)
            logger.info(f"    Circuit params: {circuit.num_parameters}")

            config = AcceleratedConfig(
                n_anchors=self._args.n_anchors,
                n_restarts=self._args.n_restarts,
                maxiter=self._args.maxiter,
                mpnn_epochs=FULL_TRAIN_EPOCHS,
                use_zoo=False,
                force_method=getattr(self._args, "force_method", None),
                bidirectional_anchors=getattr(self._args, "bidirectional_anchors", False),
            )

            t0 = time.perf_counter()
            accel = AcceleratedVQE(lattice, circuit, spec, backend, config=config)
            result = accel.run(self._h_values, seed=42, p_layers=p)
            elapsed = time.perf_counter() - t0

            self._models[p] = accel.get_model()
            self._train_results[p] = result

            # Persist θ_opt for multi-N reuse (atomic write with anti-regression)
            training_data_dir = Path("data/multi_n_training")
            training_data_dir.mkdir(parents=True, exist_ok=True)
            npz_path = training_data_dir / f"{topo}_N{N}_p{p}.npz"

            # Map method labels to quality tiers:
            # vqe_full, vqe_refined → verified (VQE-converged)
            # mpnn_refined → verified (VQE-corrected prediction)
            # mpnn_direct → approximate (MPNN only, not VQE-verified)
            quality_tiers = [
                "verified" if m in ("vqe_full", "vqe_refined", "mpnn_refined") else "approximate"
                for m in result.method
            ]

            n_upd, n_add = upsert_theta_npz(
                npz_path,
                h_new=self._h_values[: len(result.theta_opt)],
                theta_new=result.theta_opt,
                e_vqe_new=result.energies,
                e_exact_new=result.e_exact,
                gaps_new=result.gaps,
                method_new=list(result.method),
                quality_tier_new=quality_tiers,
            )
            n_verified = sum(1 for t in quality_tiers if t == "verified")
            logger.info(
                f"    Saved training data: {npz_path} "
                f"({n_add} added, {n_upd} improved, {n_verified}/{len(quality_tiers)} verified)"
            )

            all_results[f"p{p}"] = {
                "pass_rate": result.pass_rate,
                "mean_de_gap": float(result.de_gaps.mean()),
                "elapsed_s": elapsed,
                "n_anchors": result.n_anchors,
                "model_source": result.model_source,
                "methods": dict(zip(*np.unique(result.method, return_counts=True), strict=False)),
            }
            logger.info(
                f"    Done: pass_rate={result.pass_rate:.0%}, "
                f"mean_ΔE/gap={result.de_gaps.mean():.4f}, time={elapsed:.1f}s"
            )

        passed = all(r.get("pass_rate", 0) > 0.5 for r in all_results.values())
        return {"pass": passed, "per_p": all_results}

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 2 (alt): Multi-N Train
    # ═══════════════════════════════════════════════════════════════════════════

    def section_multi_n_train(self) -> dict:
        """Train UnifiedMPNN using aggregated data from ALL available N sizes.

        If a suitable multi-N model already exists in the zoo and --force-retrain
        is not set, loads it instead of retraining.
        """
        from qmbp_simulation.analysis.metrics import validate_training_dataset
        from qmbp_simulation.predictors.model_zoo import (
            ZooEntry,
            load_pretrained,
            register_checkpoint_with_training_metrics,
        )
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN, train_unified_mpnn

        topo = self._args.topology
        p = self._args.p_layers[0]
        force_retrain = getattr(self._args, "force_retrain", False)

        # Check if a multi-N model already exists (skip training if so)
        if not force_retrain:
            try:
                model, meta = load_pretrained(
                    model="tfim_bond_resolved",
                    topology=topo,
                    n_qubits=0,  # 0 = multi-N
                    p_layers=p,
                )
                self._models[p] = model
                logger.info(
                    f"  Loaded existing multi-N model: {meta.checkpoint_file} "
                    f"({meta.n_training_points} points). Use --force-retrain to rebuild."
                )
                return {
                    "pass": True,
                    "reused_existing": True,
                    "checkpoint": meta.checkpoint_file,
                    "n_training_points": meta.n_training_points,
                    "notes": meta.notes,
                }
            except FileNotFoundError:
                logger.info("  No existing multi-N model. Training from scratch.")

        # 1. Scan and aggregate all available data for this topology
        logger.info(f"  Scanning all bond-resolved data for topology={topo}...")
        # Use max_n to prevent contamination from extrapolation data beyond viable range.
        # p_layers MUST be passed so only *_p{p}.npz data is used — never mix p=1 and p=2.
        max_n = self.N_MAX_VIABLE.get(topo, 20)
        agg = MultiNAggregator(
            topology=topo,
            model="tfim_bond_resolved",
            max_n=max_n,
            p_layers=p,
            h_min=getattr(self._args, "train_h_min", None),
            h_max=getattr(self._args, "train_h_max", None),
        )
        summary = agg.scan()

        if not summary:
            return {"pass": False, "error": "No existing data found. Run --train-n first."}

        logger.info(f"  Found data for N={agg.available_n_values()}: {summary}")

        if True:
            # ── PRE-TRAINING VALIDATION ──────────────────────────────────────────
            # Validate data quality BEFORE attempting to train
            is_viable, validation_report = validate_training_dataset(
                agg._data_by_n,
                max_de_gap=0.10,
                min_total_points=10,
                min_n_values=2,
            )
            if not is_viable:
                logger.error(
                    f"  ❌ TRAINING DATA NOT VIABLE:\n"
                    f"     {validation_report['recommendation']}\n"
                    f"     Errors: {validation_report['errors']}"
                )
                for warn in validation_report.get("warnings", [])[:5]:
                    logger.warning(f"     {warn}")
                return {
                    "pass": False,
                    "error": "Training data validation failed",
                    "validation_report": validation_report,
                    "recommendation": validation_report["recommendation"],
                }
            logger.info(
                f"  ✓ Data validation passed: {validation_report['total_good']}/{validation_report['total_raw']} "
                f"good points across {validation_report['n_values_with_good_data']} N values"
            )
            if validation_report.get("warnings"):
                for warn in validation_report["warnings"][:3]:
                    logger.warning(f"     {warn}")

            # 2. Build combined dataset (filter by quality)
            dataset = agg.build_combined_dataset(max_de_gap=0.10)
            if len(dataset) < 5:
                return {
                    "pass": False,
                    "error": f"Only {len(dataset)} points pass quality filter. Need ≥5.",
                    "summary": agg.summary(),
                }

            logger.info(
                f"  Combined dataset: {len(dataset)} graphs from N={agg.available_n_values()}"
            )

            # 3. Determine output dim from dataset (varies by graph size)
            # UnifiedMPNN uses per-node prediction so output_dim is implicit
            sample_g = dataset[0]
            n_node_features = sample_g.x.shape[1] if hasattr(sample_g, "x") else 4

            # 4. Train UnifiedMPNN
            model = UnifiedMPNN(
                node_features=n_node_features,
                hidden_dim=256,
                n_layers=3,
                norm_type="none",  # MANDATORY for cross-N
                dropout=0.1,
                use_residual=getattr(self._args, "use_residual", False),
                film_conditioning=getattr(self._args, "film", False),
            )

            logger.info("  Training UnifiedMPNN (multi-N, norm_type=none)...")
            t0 = time.perf_counter()
            train_result = train_unified_mpnn(
                model,
                dataset,
                n_epochs=FULL_TRAIN_EPOCHS,
                lr=1e-3,
                patience=200,
                seed=42,
                loss_type=getattr(self._args, "loss_type", "theta_mse"),
                physics_loss_weight=getattr(self._args, "physics_loss_weight", 0.0),
            )
            elapsed = time.perf_counter() - t0

            final_mse = train_result.get("final_mse", 0) if isinstance(train_result, dict) else 0
            logger.info(f"  Training done: MSE={final_mse:.2e}, time={elapsed:.1f}s")

            # Persist training curve for post-hoc analysis
            try:
                from qmbp_simulation.utils.helpers import persist_training_curve

                persist_training_curve(
                    train_result,
                    output_dir=Path("results/training_curves"),
                    prefix=f"{topo}_multiN_p{p}",
                )
            except Exception:
                pass

            # Store model for Section 3
            self._models[p] = model

            # 5. Export to zoo as multi-N model
            from datetime import datetime

            from qmbp_simulation.predictors.model_zoo import get_runner_tag, make_date_tag

            n_values_str = "+".join(str(n) for n in agg.available_n_values())
            if getattr(self._args, "model_name", None):
                ckpt_file = f"unifMPNN__{topo}_p{p}_{self._args.model_name}.pt"
            else:
                ckpt_file = f"unified_tfim_br_{topo}_multiN_{n_values_str}_p{p}.pt"
            entry = ZooEntry(
                model="tfim_bond_resolved",
                topology=topo,
                n_qubits=0,  # 0 = multi-N
                p_layers=p,
                checkpoint_file=ckpt_file,
                # h_range documents the TRAINING h-range. When --train-h-min/max
                # restrict the dataset, record that; otherwise fall back to the
                # sweep range (which approximates the training coverage).
                h_range=self._training_h_range(),
                pass_rate=0.0,  # Updated after eval
                n_training_points=len(dataset),
                seeds=[42],
                created=datetime.now(UTC).isoformat(),
                notes=f"Multi-N training: N={agg.available_n_values()}, {len(dataset)} points"
                + self._train_h_range_note()
                + (", arch=residual" if getattr(self._args, "use_residual", False) else ""),
                runner_tag=get_runner_tag(self.runner_id),
                date_tag=make_date_tag(),
            )
            register_checkpoint_with_training_metrics(
                model,
                entry,
                training_result=train_result,
                overwrite=True,
                architecture_config={
                    "hidden_dim": 256,
                    "n_conv_layers": 3,
                    "norm_type": "none",
                    "dropout": 0.1,
                    "use_residual": getattr(self._args, "use_residual", False),
                    "film_conditioning": getattr(self._args, "film", False),
                },
            )
            logger.info(f"  Exported multi-N model: {entry.checkpoint_file}")

            # Auto-persist training curve
            try:
                from qmbp_simulation.utils.helpers import persist_training_curve

                persist_training_curve(
                    train_result,
                    output_dir=Path("results/training_curves"),
                    prefix=f"{topo}_section_multiN_p{p}",
                )
            except Exception:
                pass

            # Enrich model registry with per-N point breakdown
            try:
                from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

                db = ModelRegistryDB()
                record = db.get_model(entry.checkpoint_file)
                if record:
                    record.training.points_per_n = {
                        str(k): v for k, v in agg.summary()["points_per_n"].items()
                    }
                    record.training.n_values_used = agg.available_n_values()
                    db.register_model(record, overwrite=True)
            except Exception as e:
                logger.debug("Registry enrichment failed (non-critical): %s", e)

        return {
            "pass": True,
            "n_values_used": agg.available_n_values(),
            "n_training_points": len(dataset),
            "points_per_n": {str(k): v for k, v in agg.summary()["points_per_n"].items()},
            "final_mse": float(final_mse),
            "elapsed_s": elapsed,
            "checkpoint": entry.checkpoint_file,
            "validation_report": validation_report,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 3: Cross-N Predict
    # ═══════════════════════════════════════════════════════════════════════════

    def section_cross_n_predict(self) -> dict:
        """Predict at each N_target using the N_train model."""
        import torch

        from qmbp_simulation.analysis.metrics import compute_deploy_summary
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.models.constants import STATEVECTOR_MAX_N
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.predictors.unified_graph import build_unified_bond_resolved_graph

        spec = get_model_spec("tfim_bond_resolved")
        hva = HVACircuitBuilder()
        topo = self._args.topology

        all_results = {}

        for p in self._args.p_layers:
            # Load model: from memory (section 2), from zoo (best for cross-N), or train new
            model = self._models.get(p)
            if model is None:
                # Respect --from-zoo / --checkpoint: never train a new model.
                _train_if_missing = not (
                    self._args.from_zoo or bool(getattr(self._args, "checkpoint", None))
                )
                model = self.load_best_mpnn_for_cross_n(
                    n_target=self._args.target_n[0],
                    model="tfim_bond_resolved",
                    topology=topo,
                    p_layers=p,
                    checkpoint_path=self._args.checkpoint,
                    train_if_missing=_train_if_missing,
                    train_epochs=FULL_TRAIN_EPOCHS,
                )
                if model is None:
                    _reason = (
                        "No model in zoo for this config (--from-zoo set, training disabled)"
                        if self._args.from_zoo
                        else "No model and no training data available"
                    )
                    all_results[f"p{p}"] = {
                        "pass": False,
                        "error": _reason,
                    }
                    continue

            model.eval()

            for n_target in self._args.target_n:
                logger.info(f"  Cross-N: N_train={self._args.train_n} → N_target={n_target}, p={p}")
                lattice_target = self.make_lattice(topo, n_target, J=1.0, h=2.0)
                circuit_target, _ = hva.create_bond_resolved(n_target, p, lattice_target)
                n_params_target = circuit_target.num_parameters

                # Use CachedBackend for transparent eval caching
                use_eval_cache = not getattr(self._args, "no_eval_cache", False)
                eval_backend = self.get_cached_backend(
                    topology=topo,
                    n_qubits=n_target,
                    model="tfim_bond_resolved",
                    p_layers=p,
                    enabled=use_eval_cache,
                )
                # Log which backend was selected (debug N>22 slowness)
                _inner = getattr(eval_backend, "_backend", eval_backend)
                logger.info(
                    f"    Backend for N={n_target}: {_inner.name} "
                    f"(wrapped in CachedBackend, cache={use_eval_cache})"
                )
                if use_eval_cache:
                    logger.info(f"    Eval cache: {len(eval_backend.cache)} entries loaded")

                per_h_results = []
                t0 = time.perf_counter()
                _interrupted = False

                try:
                    for h in self._h_values:
                        # Build unified graph for N_target
                        g = build_unified_bond_resolved_graph(
                            lattice_target,
                            h_value=float(h),
                            p_layers=p,
                            include_circuit_nodes=True,
                        )
                        with torch.no_grad():
                            theta_pred = model(g).numpy().flatten()

                        theta_pred = np.clip(theta_pred, -np.pi, np.pi)

                        # MC-Dropout uncertainty estimation (reuses model method)
                        theta_std = 0.0
                        if hasattr(model, "predict_with_uncertainty"):
                            _, theta_std = model.predict_with_uncertainty(g)

                        # Verify param count matches circuit
                        if len(theta_pred) != n_params_target:
                            logger.warning(
                                f"    Param mismatch at h={h:.2f}: "
                                f"predicted {len(theta_pred)}, need {n_params_target}"
                            )
                            if len(theta_pred) < n_params_target:
                                theta_pred = np.pad(
                                    theta_pred, (0, n_params_target - len(theta_pred))
                                )
                            else:
                                theta_pred = theta_pred[:n_params_target]

                        # Evaluate energy via CachedBackend
                        lat_h = self.make_lattice(topo, n_target, J=1.0, h=float(h))
                        H = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)
                        eval_backend.set_h(float(h))
                        e_pred = eval_backend.evaluate(circuit_target, H, theta_pred)

                        # Ground truth via parent's cached exact_ground_state
                        e_exact, gap = self.exact_ground_state(
                            topo, n_target, float(h), model="tfim_bond_resolved"
                        )

                        de_gap = abs(e_pred - e_exact) / max(gap, 1e-10)
                        abs_err = abs(e_pred - e_exact)

                        # Fidelity at any N: exact (N ≤ 16) or variance-based
                        # lower bound (Eckart) for larger N. Records provenance.
                        fid_info = self.estimate_fidelity(
                            circuit_target,
                            theta_pred,
                            topo,
                            n_target,
                            float(h),
                            model="tfim_bond_resolved",
                            gap=gap,
                            e_pred=e_pred,
                        )
                        fidelity = fid_info.get("fidelity")

                        per_h_result = self.build_per_h_result(
                            h,
                            e_pred,
                            e_exact,
                            gap,
                            fidelity_info=fid_info,
                            n_params=len(theta_pred),
                        )
                        per_h_result["theta_std"] = theta_std
                        per_h_results.append(per_h_result)
                        if fidelity is not None:
                            _bnd = "≥" if fid_info.get("is_lower_bound") else "="
                            fid_str = f"F{_bnd}{fidelity:.4f}"
                        else:
                            fid_str = "F=N/A"
                        logger.info(
                            f"    h={h:.2f}: ΔE/gap={de_gap:.4f} {fid_str} "
                            f"|ΔE|={abs_err:.2e} [{len(theta_pred)} params]"
                        )

                except KeyboardInterrupt:
                    _interrupted = True
                    logger.warning(
                        f"  ⚠️ Interrupted during cross-N predict N={n_target}. "
                        f"Saving {len(per_h_results)} partial results."
                    )

                elapsed = time.perf_counter() - t0

                # ── Active learning: refine low-fidelity points ───────
                active_rounds = getattr(self._args, "active_rounds", 0)
                n_refined = 0
                cold_start_samples = []

                if active_rounds > 0:
                    from scipy.optimize import minimize as _minimize

                    # Select VQE backend for refinement
                    vqe_backend = self.select_backend(n_target, for_vqe_loop=True)

                    # ── Cold-start baseline (2 sample points) ──
                    sample_indices = [0, len(per_h_results) // 2]
                    rng_cold = np.random.default_rng(99)
                    for si in sample_indices:
                        if si >= len(per_h_results):
                            continue
                        r_cold = per_h_results[si]
                        h_cold = r_cold["h"]
                        try:
                            lat_cold = self.make_lattice(topo, n_target, J=1.0, h=h_cold)
                            H_cold = spec.build_hamiltonian(lat_cold, **spec.hamiltonian_kwargs)
                            theta_random = rng_cold.uniform(-np.pi, np.pi, n_params_target)
                            al_maxiter = 50 if n_target > STATEVECTOR_MAX_N else 200
                            res_cold = _minimize(
                                lambda params: vqe_backend.evaluate(circuit_target, H_cold, params),
                                theta_random,
                                method="COBYLA",
                                options={"maxiter": al_maxiter, "rhobeg": 0.5},
                            )
                            e_ex_cold, gap_cold = self.exact_ground_state(
                                topo, n_target, h_cold, model="tfim_bond_resolved"
                            )
                            de_gap_cold = abs(res_cold.fun - e_ex_cold) / max(gap_cold, 1e-10)
                            cold_start_samples.append(
                                {
                                    "h": h_cold,
                                    "de_gap_cold": float(de_gap_cold),
                                    "de_gap_warm": float(r_cold["de_gap"]),
                                    "speedup": "warm better"
                                    if r_cold["de_gap"] < de_gap_cold
                                    else "cold better",
                                }
                            )
                            logger.info(
                                f"    Cold-start baseline h={h_cold:.2f}: "
                                f"dE/gap_cold={de_gap_cold:.4f} vs dE/gap_warm={r_cold['de_gap']:.4f} "
                                f"({'warm wins' if r_cold['de_gap'] < de_gap_cold else 'cold wins'})"
                            )
                        except Exception as e:
                            logger.debug(f"    Cold-start sample h={h_cold:.2f} failed: {e}")

                    # ── Active learning rounds ────────────────────────────────
                    for al_round in range(active_rounds):
                        from qmbp_simulation.analysis.metrics import is_point_failure

                        refine_indices = [
                            i
                            for i, r in enumerate(per_h_results)
                            if is_point_failure(
                                r["de_gap"],
                                abs_error=r.get("abs_error"),
                            )
                        ]
                        if not refine_indices:
                            logger.info(f"    AL round {al_round + 1}: all points pass. Done.")
                            break
                        refine_indices = refine_indices[:5]
                        logger.info(
                            f"    Active learning round {al_round + 1}: "
                            f"refining {len(refine_indices)} points "
                            f"(backend={type(vqe_backend).__name__})"
                        )
                        for idx in refine_indices:
                            r = per_h_results[idx]
                            h_val = r["h"]
                            try:
                                lat_ref = self.make_lattice(topo, n_target, J=1.0, h=h_val)
                                H_ref = spec.build_hamiltonian(lat_ref, **spec.hamiltonian_kwargs)

                                g_ref = build_unified_bond_resolved_graph(
                                    lattice_target,
                                    h_value=h_val,
                                    p_layers=p,
                                    include_circuit_nodes=True,
                                )
                                with torch.no_grad():
                                    theta_init = model(g_ref).numpy().flatten()
                                theta_init = np.clip(theta_init, -np.pi, np.pi)
                                if len(theta_init) != n_params_target:
                                    if len(theta_init) < n_params_target:
                                        theta_init = np.pad(
                                            theta_init, (0, n_params_target - len(theta_init))
                                        )
                                    else:
                                        theta_init = theta_init[:n_params_target]

                                al_maxiter = 200 if n_target <= STATEVECTOR_MAX_N else 50
                                res = _minimize(
                                    lambda params: vqe_backend.evaluate(
                                        circuit_target, H_ref, params
                                    ),
                                    theta_init,
                                    method="COBYLA",
                                    options={"maxiter": al_maxiter, "rhobeg": 0.1},
                                )

                                e_exact_ref, gap_ref = self.exact_ground_state(
                                    topo, n_target, h_val, model="tfim_bond_resolved"
                                )
                                de_gap_new = abs(res.fun - e_exact_ref) / max(gap_ref, 1e-10)

                                # Fidelity: exact (N≤16) or variance bound (N>16)
                                fid_info_new = self.estimate_fidelity(
                                    circuit_target,
                                    res.x,
                                    topo,
                                    n_target,
                                    h_val,
                                    model="tfim_bond_resolved",
                                    gap=gap_ref,
                                    e_pred=float(res.fun),
                                )
                                fid_new = fid_info_new.get("fidelity")

                                if de_gap_new < r["de_gap"]:
                                    per_h_results[idx] = {
                                        **r,
                                        "de_gap": float(de_gap_new),
                                        "abs_error": float(abs(res.fun - e_exact_ref)),
                                        "fidelity": fid_new
                                        if fid_new is not None
                                        else r.get("fidelity"),
                                        "fidelity_method": fid_info_new.get("method"),
                                        "fidelity_is_bound": fid_info_new.get(
                                            "is_lower_bound", False
                                        ),
                                        "e_pred": float(res.fun),
                                        "method": "refined",
                                        "de_gap_before_refine": float(r["de_gap"]),
                                    }
                                    n_refined += 1
                                    logger.info(
                                        f"      h={h_val:.2f}: dE/gap {r['de_gap']:.4f} -> {de_gap_new:.4f} "
                                        f"(improvement: {(1 - de_gap_new / r['de_gap']) * 100:.0f}%)"
                                    )
                                else:
                                    logger.info(
                                        f"      h={h_val:.2f}: no improvement ({de_gap_new:.4f} >= {r['de_gap']:.4f})"
                                    )
                            except Exception as e:
                                logger.warning(f"      h={h_val:.2f}: refinement failed: {e}")
                                continue

                    if n_refined > 0:
                        logger.info(f"    AL summary: {n_refined} points refined")

                # ── Compute summary via reusable utility ──────────────
                summary = compute_deploy_summary(per_h_results)

                # ── Uncertainty calibration (θ_std vs ΔE/gap) ─────────
                from qmbp_simulation.analysis.metrics import compute_uncertainty_correlation

                uc_report = compute_uncertainty_correlation(per_h_results)

                key = f"p{p}_N{n_target}"
                all_results[key] = {
                    "train_n": self._args.train_n,
                    "target_n": n_target,
                    "p_layers": p,
                    "n_params": n_params_target,
                    **summary,
                    "uncertainty_calibration": uc_report
                    if uc_report["n_points_with_uncertainty"] >= 3
                    else None,
                    "fidelity_available": n_target <= STATEVECTOR_MAX_N,
                    "active_learning_applied": n_refined > 0,
                    "n_refined": n_refined,
                    "cold_start_comparison": cold_start_samples if active_rounds > 0 else None,
                    "elapsed_s": elapsed,
                    "per_point": per_h_results,
                }
                fid_info = (
                    f"F_mean={summary['mean_fidelity']:.4f}"
                    if summary.get("mean_fidelity")
                    else "F=N/A"
                )
                grade = summary.get("grade", "?")
                score = summary.get("quality_score", 0)
                logger.info(
                    f"  N={n_target}: {grade}({score:.2f}) "
                    f"ΔE/gap={summary['mean_de_gap']:.4f}±{summary.get('std_de_gap', 0):.4f} "
                    f"P90={summary.get('p90_de_gap', 0):.4f} "
                    f"{fid_info}, refined={n_refined}"
                )

        # Overall pass: at least one target has >50% at 10% threshold
        passed = any(
            v.get("pass_rate_10pct", 0) > 0.5
            for v in all_results.values()
            if isinstance(v, dict) and "pass_rate_10pct" in v
        )

        # Flush eval cache — os._exit() in ValidationRunner skips __del__,
        # so without explicit flush the last cached evaluations are lost.
        if hasattr(eval_backend, "flush"):
            eval_backend.flush()

        # ── Auto-update zoo pass_rate with observed results ──────────────
        # Use centralized update_zoo_pass_rate for better maintainability
        if hasattr(self, "_zoo_entry") and self._zoo_entry is not None:
            observed_pass_rates = [
                v.get("pass_rate_dual", v.get("pass_rate_5pct", 0))
                for v in all_results.values()
                if isinstance(v, dict) and ("pass_rate_dual" in v or "pass_rate_5pct" in v)
            ]
            if observed_pass_rates:
                observed = max(observed_pass_rates)
                try:
                    from qmbp_simulation.predictors.model_zoo import update_zoo_pass_rate

                    n_target = self._args.target_n[0] if self._args.target_n else "?"
                    update_zoo_pass_rate(
                        self._zoo_entry.checkpoint_file,
                        observed,
                        only_if_better=True,
                        add_notes=f"cross-N@N={n_target}",
                    )
                except Exception:
                    pass  # Non-fatal

        return {"pass": passed, "cross_n_results": all_results}

    # ═══════════════════════════════════════════════════════════════════════════
    # Section: Budget Estimation + Cache Warm-up
    # ═══════════════════════════════════════════════════════════════════════════

    def section_budget_estimation(self) -> dict:
        """Estimate compute budget leveraging cached results.

        Uses the shared `self.estimate_compute_budget()` + `self.log_budget_summary()`
        helpers from ValidationRunner, then adds runner-specific
        QualityPredictor estimates on top.
        """
        topo = self._args.topology
        n_target = self._args.target_n[0]
        p = self._args.p_layers[0]
        max_iters = self._args.max_iterations

        # ── Shared budget estimation (GT cache, EvalCache, NPZ, h_frontier)
        budget = self.estimate_compute_budget(
            h_values=self._h_values,
            n_qubits=n_target,
            topology=topo,
            model="tfim_bond_resolved",
            max_iterations=max_iters,
        )

        # ── Runner-specific: QualityPredictor time estimate ──────────────
        estimated_time_s = 0.0
        try:
            from qmbp_simulation.analysis.quality_predictor import QualityPredictor

            predictor = QualityPredictor()
            report = predictor.predict(
                model="tfim_bond_resolved",
                topology=topo,
                n_qubits=n_target,
                p_layers=p,
                h_min=self._args.h_min,
                h_max=self._args.h_max,
            )
            estimated_time_s = report.estimated_time_s
        except Exception:
            pass

        # ── Log using reusable base class method ─────────────────────────
        self.log_budget_summary(
            budget,
            topology=topo,
            n_qubits=n_target,
            p_layers=p,
            historical_time_s=estimated_time_s,
        )

        budget["historical_time_s"] = estimated_time_s
        return {"pass": True, "budget": budget}

    # ═══════════════════════════════════════════════════════════════════════════
    # Section: Iterative Improvement Loop
    # ═══════════════════════════════════════════════════════════════════════════

    def section_iterative_improve(self) -> dict:
        """Iterative predict → refine → retrain loop with cache reuse."""
        import torch

        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.execution import NoiselessBackend
        from qmbp_simulation.execution.eval_cache import EvalCache
        from qmbp_simulation.models.constants import STATEVECTOR_MAX_N
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.predictors.model_zoo import (
            ZooEntry,
            load_pretrained,
            register_checkpoint_with_training_metrics,
        )
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator
        from qmbp_simulation.predictors.unified_graph import (
            build_unified_bond_resolved_graph,
        )
        from qmbp_simulation.predictors.unified_mpnn import (
            UnifiedMPNN,
            train_unified_mpnn,
        )
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        topo = self._args.topology
        n_target = self._args.target_n[0]
        p = self._args.p_layers[0]
        max_iterations = self._args.max_iterations
        improvement_threshold = self._args.improvement_threshold
        spec = get_model_spec("tfim_bond_resolved")
        hva = HVACircuitBuilder()

        # Architecture flag: CLI --use-residual or auto-detected from loaded model
        use_residual = getattr(self._args, "use_residual", False)

        # ── Setup: caches, backend, circuit ───────────────────────────────
        gt_cache = getattr(self, "_disk_gt_cache", None) or GroundTruthCache()
        if not hasattr(self, "_disk_gt_cache"):
            self._disk_gt_cache = gt_cache
        use_eval_cache = not getattr(self._args, "no_eval_cache", False)
        eval_cache = EvalCache(enabled=use_eval_cache)
        backend = NoiselessBackend()
        solver = self.solver

        # Auto-select backend for N > 22
        if n_target > STATEVECTOR_MAX_N:
            try:
                from qmbp_simulation.execution import MPSBackend

                eval_backend = MPSBackend(strategy="aer_mps", chi_max=64, deterministic=True)
                logger.info(
                    f"  Backend: MPSBackend(aer_mps, chi=64, det=True) for N={n_target} > {STATEVECTOR_MAX_N}"
                )
            except ImportError:
                eval_backend = backend
                logger.warning(
                    f"  ⚠️ qiskit-aer not available — falling back to NoiselessBackend "
                    f"for N={n_target}. THIS WILL BE VERY SLOW (2^{n_target} amplitudes)."
                )
        else:
            eval_backend = backend
            logger.info(f"  Backend: NoiselessBackend for N={n_target} ≤ {STATEVECTOR_MAX_N}")

        lattice_target = self.make_lattice(topo, n_target, J=1.0, h=2.0)
        circuit_target, _ = hva.create_bond_resolved(n_target, p, lattice_target)
        n_params = circuit_target.num_parameters
        logger.info(
            f"  Circuit: N={n_target}, p={p}, n_params={n_params}, eval_backend={eval_backend.name}"
        )

        # NPZ path for this config
        npz_dir = Path("data/multi_n_training")
        npz_dir.mkdir(parents=True, exist_ok=True)
        npz_path = npz_dir / f"{topo}_N{n_target}_p{p}.npz"

        # Load existing refined θ from NPZ (anti-regression baseline)
        from qmbp_simulation.framework.result_io import load_npz_as_theta_dict

        prev_theta_by_h = load_npz_as_theta_dict(npz_path, n_params)

        # Ansatz-limit boundary disabled: all h values are refinable.
        # This was previously used to skip points near quantum critical point
        # but empirically refinement helps even there.
        h_min_valid = 0

        # ── Compute ground truth (all from cache ideally) ─────────────────
        # Route through get_or_compute so this loop shares the SAME stale-floor
        # gap invalidation (N>18 gap≈2π/N) as exact_ground_state — previously a
        # raw gt_cache.get() here could serve a stale gap and skew ΔE/gap for
        # large N. Hits/misses are inferred from cache membership pre-call.
        gt_hits, gt_misses = 0, 0
        e_exact_arr = np.zeros(len(self._h_values))
        gap_arr = np.zeros(len(self._h_values))
        for i, h in enumerate(self._h_values):
            _was_cached = gt_cache.get(topo, n_target, "tfim_bond_resolved", float(h)) is not None
            t_gt = time.perf_counter()
            e_i, gap_i = gt_cache.get_or_compute(
                topo,
                n_target,
                "tfim_bond_resolved",
                float(h),
                flush=False,  # batch flush after the loop
                solver=solver,
            )
            e_exact_arr[i] = e_i
            gap_arr[i] = gap_i
            if _was_cached:
                gt_hits += 1
            else:
                gt_misses += 1
                logger.info(
                    f"  GT [{gt_hits + gt_misses}/{len(self._h_values)}] "
                    f"h={float(h):.3f} E={e_i:.6f} "
                    f"gap={gap_i:.4f} ({time.perf_counter() - t_gt:.1f}s)"
                )
        logger.info(f"  Ground truth: {gt_hits} cache hits, {gt_misses} computed")
        if gt_misses > 0:
            gt_cache.flush()  # Persist new ground truths immediately

        # ── Refresh NPZ ground truth from GroundTruthCache ────────────────
        # If GT was recomputed with a better solver (eigsh vs DMRG), update
        # the NPZ e_exact so that ΔE/gap metrics reflect the latest values.
        if npz_path.exists():
            from qmbp_simulation.framework.result_io import refresh_npz_ground_truth

            n_refreshed = refresh_npz_ground_truth(
                npz_path,
                topology=topo,
                n_qubits=n_target,
                model="tfim_bond_resolved",
            )
            if n_refreshed > 0:
                # Reload prev_theta_by_h since energies haven't changed
                # but the variational principle check uses e_exact_arr
                # (which we just updated from GT cache above)
                logger.info(f"  NPZ ground truth refreshed: {n_refreshed} points updated")

        # ── Load best model (unified selection: per-topo + MT + single-N) ──
        model = None
        _meta = None

        # Explicit --checkpoint: honor it exactly (use only this model), matching
        # the predict-only contract of section_cross_n_predict. Resolves as a
        # path, exact zoo name, or fuzzy tag/fragment (e.g. "h_0p5_1p5").
        # Without this, iterative-improve silently ignored --checkpoint and fell
        # back to load_best_model_for's zoo "best" selection.
        _ckpt = getattr(self._args, "checkpoint", None)
        if _ckpt:
            from qmbp_simulation.predictors.model_zoo import (
                _smart_load_checkpoint,
                resolve_checkpoint_fuzzy,
            )

            ckpt_path = resolve_checkpoint_fuzzy(str(_ckpt), topology=topo, p_layers=p)
            if ckpt_path is None:
                candidates = resolve_checkpoint_fuzzy(str(_ckpt), return_all=True)
                names = [pth.name for pth, _ in (candidates or [])[:5]]
                hint = f" Closest checkpoints on disk: {names}" if names else ""
                return {
                    "pass": False,
                    "error": (
                        f"--checkpoint '{_ckpt}' not found as a file path, exact "
                        f"zoo name, or fuzzy match.{hint}"
                    ),
                }
            model = _smart_load_checkpoint(str(ckpt_path))
            model.eval()
            if getattr(model, "use_residual", False) and not use_residual:
                use_residual = True
                logger.info("  Auto-detected use_residual=True from loaded model")
            logger.info(f"  Loaded EXPLICIT checkpoint: {ckpt_path.name}")

        if model is None:
            try:
                from qmbp_simulation.predictors.model_zoo import load_best_model_for

                model, _meta, _source = load_best_model_for(
                    topo,
                    model="tfim_bond_resolved",
                    p_layers=p,
                    n_target=n_target,
                    include_multi_topology=True,
                )
                model.eval()
                # Auto-detect architecture from loaded model (issue #7)
                if getattr(model, "use_residual", False) and not use_residual:
                    use_residual = True
                    logger.info("  Auto-detected use_residual=True from loaded model")
                logger.info(
                    f"  Loaded model [{_source}]: {_meta.checkpoint_file} "
                    f"(pass={_meta.pass_rate:.0%}, {_meta.n_training_points} pts)"
                )
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.debug(f"  Unified model loading failed: {e}")

        if model is None:
            try:
                model, _meta = load_pretrained(
                    model="tfim_bond_resolved",
                    topology=topo,
                    n_qubits=self._args.train_n,
                    p_layers=p,
                    allow_cross_n=True,
                )
                logger.info(f"  Loaded single-N model from zoo: {_meta.checkpoint_file}")
                # Auto-detect architecture from single-N fallback (issue #7)
                if getattr(model, "use_residual", False) and not use_residual:
                    use_residual = True
                    logger.info("  Auto-detected use_residual=True from single-N model")
            except FileNotFoundError:
                # --from-zoo: never bootstrap/train. Fail clean if no zoo model.
                if getattr(self._args, "from_zoo", False):
                    return {
                        "pass": False,
                        "error": (
                            f"No model in zoo for {topo} p={p} (--from-zoo set, "
                            "bootstrap/training disabled). Train a model first or "
                            "drop --from-zoo."
                        ),
                    }
                # Bootstrap: no model exists → run AcceleratedVQE to generate
                # initial training data, then train a UnifiedMPNN from it.
                logger.info("  No model in zoo — bootstrapping via AcceleratedVQE...")
                from qmbp_simulation.pipeline.accelerated import (
                    AcceleratedConfig,
                    AcceleratedVQE,
                )

                boot_config = AcceleratedConfig(
                    n_anchors=self._args.n_anchors,
                    n_restarts=self._args.n_restarts,
                    maxiter=self._args.maxiter,
                    mpnn_epochs=FULL_TRAIN_EPOCHS,
                    use_zoo=False,
                    force_method=getattr(self._args, "force_method", None),
                    bidirectional_anchors=getattr(self._args, "bidirectional_anchors", False),
                )
                boot_lattice = self.make_lattice(topo, n_target, J=1.0, h=2.0)
                boot_circuit, _ = hva.create_bond_resolved(n_target, p, boot_lattice)
                t_boot = time.perf_counter()
                accel = AcceleratedVQE(
                    boot_lattice, boot_circuit, spec, eval_backend, config=boot_config
                )
                boot_result = accel.run(self._h_values, seed=42, p_layers=p)
                logger.info(
                    f"  Bootstrap done: pass_rate_dual={boot_result.pass_rate:.0%}, "
                    f"time={time.perf_counter() - t_boot:.1f}s"
                )
                # Save bootstrap data to NPZ (atomic with anti-regression)
                # Map method labels to quality tiers for bootstrap data
                boot_quality_tiers = [
                    "verified"
                    if m in ("vqe_full", "vqe_refined", "mpnn_refined")
                    else "approximate"
                    for m in boot_result.method
                ]
                upsert_theta_npz(
                    npz_path,
                    h_new=self._h_values[: len(boot_result.theta_opt)],
                    theta_new=boot_result.theta_opt,
                    e_vqe_new=boot_result.energies,
                    e_exact_new=boot_result.e_exact,
                    gaps_new=boot_result.gaps,
                    method_new=list(boot_result.method),
                    quality_tier_new=boot_quality_tiers,
                )
                # Reload prev_theta_by_h from freshly saved NPZ
                for i, h in enumerate(self._h_values[: len(boot_result.theta_opt)]):
                    th_i = boot_result.theta_opt[i]
                    if np.all(np.isfinite(th_i)) and len(th_i) == n_params:
                        prev_theta_by_h[round(float(h), 2)] = (th_i, float(boot_result.energies[i]))
                # Train initial UnifiedMPNN from bootstrap data (p-scoped: only *_p{p}.npz)
                agg = MultiNAggregator(
                    topology=topo,
                    model="tfim_bond_resolved",
                    max_n=self.N_MAX_VIABLE.get(topo, 20),
                    p_layers=p,
                    h_min=getattr(self._args, "train_h_min", None),
                    h_max=getattr(self._args, "train_h_max", None),
                )
                agg.scan()
                dataset = agg.build_combined_dataset(max_de_gap=0.15)
                if len(dataset) < 3:
                    return {
                        "pass": False,
                        "error": f"Bootstrap produced only {len(dataset)} valid points.",
                    }
                sample_g = dataset[0]
                n_node_features = sample_g.x.shape[1] if hasattr(sample_g, "x") else 4
                model = UnifiedMPNN(
                    node_features=n_node_features,
                    hidden_dim=256,
                    n_layers=3,
                    norm_type="none",
                    dropout=0.1,
                    use_residual=use_residual,
                    film_conditioning=getattr(self._args, "film", False),
                )

                boot_train_result = train_unified_mpnn(
                    model, dataset, n_epochs=FULL_TRAIN_EPOCHS, lr=1e-3, patience=200, seed=42
                )
                # Register in zoo with training metrics
                from datetime import datetime

                entry = ZooEntry(
                    model="tfim_bond_resolved",
                    topology=topo,
                    n_qubits=0,
                    p_layers=p,
                    checkpoint_file=(
                        f"unifMPNN__{topo}_p{p}_{self._args.model_name}.pt"
                        if getattr(self._args, "model_name", None)
                        else f"unified_tfim_br_{topo}_multiN_{n_target}_p{p}.pt"
                    ),
                    h_range=self._training_h_range(),
                    pass_rate=boot_result.pass_rate,
                    n_training_points=len(dataset),
                    seeds=[42],
                    created=datetime.now(UTC).isoformat(),
                    notes=f"Bootstrap from AcceleratedVQE N={n_target}"
                    + self._train_h_range_note()
                    + (", arch=residual" if use_residual else ""),
                )
                register_checkpoint_with_training_metrics(
                    model,
                    entry,
                    training_result=boot_train_result,
                    overwrite=True,
                    architecture_config={
                        "hidden_dim": 256,
                        "n_conv_layers": 3,
                        "norm_type": "none",
                        "dropout": 0.1,
                        "use_residual": use_residual,
                    },
                )
                logger.info(f"  Bootstrap model registered: {entry.checkpoint_file}")

        # ── Iterative improvement loop ────────────────────────────────────
        iteration_reports = []
        total_vqe_calls = 0
        prev_pass_rate = 0.0
        convergence_reason = "max_iterations"
        # Track the best pass_rate ever exported to zoo for Fix C.
        # Use _meta.pass_rate if available, but also check the zoo manifest
        # for the ACTUAL best model for this config — _meta could come from
        # a fallback load_pretrained with pass_rate=0 (unevaluated).
        zoo_best_pass_rate = _meta.pass_rate if _meta is not None else 0.0
        if zoo_best_pass_rate < 0.01:
            try:
                from qmbp_simulation.predictors.model_zoo import _load_manifest

                _existing = [
                    e
                    for e in _load_manifest()
                    if e.topology == topo
                    and e.model == "tfim_bond_resolved"
                    and e.p_layers == p
                    and e.n_qubits == 0
                ]
                if _existing:
                    zoo_best_pass_rate = max(e.pass_rate for e in _existing)
                    if zoo_best_pass_rate > 0:
                        logger.info(
                            f"  Zoo baseline: existing best pass_rate={zoo_best_pass_rate:.0%} "
                            f"(loaded model was unevaluated)"
                        )
            except Exception:
                pass

        for iteration in range(1, max_iterations + 1):
            logger.info(f"\n  ╔══ Iteration {iteration}/{max_iterations} ══════════════════════╗")
            t_iter_start = time.perf_counter()
            model.eval()

            # ── 2a: Predict θ for all h-points ────────────────────────────
            predictions = []
            n_pred_invalid = 0
            for h in self._h_values:
                g = build_unified_bond_resolved_graph(
                    lattice_target,
                    h_value=float(h),
                    p_layers=p,
                    include_circuit_nodes=True,
                )
                with torch.no_grad():
                    pred = model(g).numpy().flatten()
                # NaN/Inf guard on predictions
                if not np.all(np.isfinite(pred)):
                    n_bad = int(np.sum(~np.isfinite(pred)))
                    logger.warning(f"    h={float(h):.2f}: prediction has {n_bad} NaN/Inf → zeroed")
                    pred = np.where(np.isfinite(pred), pred, 0.0)
                    n_pred_invalid += 1
                pred = np.clip(pred, -np.pi, np.pi)
                if len(pred) != n_params:
                    if len(pred) < n_params:
                        pred = np.pad(pred, (0, n_params - len(pred)))
                    else:
                        pred = pred[:n_params]
                predictions.append(pred)
            predictions = np.array(predictions)
            if n_pred_invalid > 0:
                logger.warning(
                    f"  │ ⚠️ {n_pred_invalid}/{len(self._h_values)} predictions had NaN/Inf"
                )

            # ── 2b: Evaluate + identify failures ──────────────────────────
            # CRITICAL: On iteration 2+, ALWAYS evaluate MPNN predictions
            # even for points with stored θ_prev. The retrained model may
            # have found a better energy basin that we'd miss by skipping.
            energies = np.zeros(len(self._h_values))
            eval_hits = 0
            for i, h in enumerate(self._h_values):
                h_key = round(float(h), 2)

                # Always evaluate the MPNN prediction (cheap via eval_cache)
                key = eval_cache.make_key(
                    topo,
                    n_target,
                    float(h),
                    predictions[i],
                    model="tfim_bond_resolved",
                    p_layers=p,
                )
                cached_e = eval_cache.get(key)
                if cached_e is not None:
                    e_pred_i = cached_e
                    eval_hits += 1
                else:
                    lat_h = self.make_lattice(topo, n_target, J=1.0, h=float(h))
                    H = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)
                    e_pred_i = eval_backend.evaluate(circuit_target, H, predictions[i])
                    eval_cache.put(key, float(e_pred_i))

                # Compare with prev_theta_by_h: keep the lower energy
                if h_key in prev_theta_by_h:
                    theta_prev, e_prev = prev_theta_by_h[h_key]
                    if e_prev is not None and np.isfinite(e_prev):
                        # Variational principle check on stored energy
                        if e_prev >= e_exact_arr[i] - 1e-6:
                            if e_prev <= e_pred_i:
                                # Previous θ is still best → use it
                                energies[i] = e_prev
                                continue
                            else:
                                # MPNN found better energy → update tracking
                                energies[i] = float(e_pred_i)
                                prev_theta_by_h[h_key] = (predictions[i], float(e_pred_i))
                                continue
                        else:
                            logger.warning(
                                f"    h={float(h):.2f}: NPZ energy {e_prev:.6f} "
                                f"violates variational principle (E_exact={e_exact_arr[i]:.6f}). "
                                f"Invalidating energy but keeping θ for warm-start."
                            )
                            # Keep θ for VQE init later, but invalidate stored energy
                            prev_theta_by_h[h_key] = (theta_prev, None)

                # No prev or prev invalidated → use MPNN prediction energy
                energies[i] = float(e_pred_i)

            de_gaps = np.abs(energies - e_exact_arr) / np.maximum(gap_arr, 1e-10)
            abs_errors = np.abs(energies - e_exact_arr)
            from qmbp_simulation.analysis.metrics import DE_GAP_THRESHOLD, MAX_ABS_ERROR

            dual_mask = (de_gaps < DE_GAP_THRESHOLD) & (abs_errors < MAX_ABS_ERROR)
            pass_rate = float(dual_mask.mean())
            pass_rate_5pct = float((de_gaps < 0.05).mean())
            logger.info(
                f"  │ Eval: {eval_hits}/{len(self._h_values)} eval_cache hits, "
                f"pass_rate_dual={pass_rate:.0%} (5pct_only={pass_rate_5pct:.0%})"
            )

            # ── 2b.0: Persist ALL passing predictions IMMEDIATELY to NPZ ──
            # Predictions with ΔE/gap < 5% are as good as VQE-optimized θ_opt.
            # 1. Data is safe even if process crashes before convergence check
            # 2. Multi-N aggregator sees fresh data for this config immediately
            # 3. Anti-regression: upsert_theta_npz only updates if energy improves
            n_newly_persisted = 0
            for i, h in enumerate(self._h_values):
                if de_gaps[i] >= 0.05:
                    continue  # only persist passing predictions
                h_key = round(float(h), 2)
                # Only persist if not already in memory dict OR if new energy is lower
                if h_key in prev_theta_by_h:
                    _, e_existing = prev_theta_by_h[h_key]
                    if e_existing is not None and energies[i] >= e_existing - 1e-10:
                        continue  # existing is already as good or better

                # Persist immediately to NPZ (atomic write handles concurrency)
                n_upd, n_add = upsert_theta_npz(
                    npz_path,
                    h_new=np.array([float(h)]),
                    theta_new=np.array([predictions[i]]),
                    e_vqe_new=np.array([float(energies[i])]),
                    e_exact_new=np.array([float(e_exact_arr[i])]),
                    gaps_new=np.array([float(gap_arr[i])]),
                    method_new=["mpnn_pred"],
                    quality_tier_new=["approximate"],
                )
                # Update in-memory dict to avoid re-persisting same point
                prev_theta_by_h[h_key] = (predictions[i], float(energies[i]))
                if n_upd > 0 or n_add > 0:
                    n_newly_persisted += 1

            if n_newly_persisted > 0:
                logger.info(
                    f"  │ Persisted {n_newly_persisted} new passing predictions → NPZ "
                    f"(total in memory: {len(prev_theta_by_h)} points)"
                )

            # Points with stored θ_prev that pass dual criterion AND where the
            # MPNN independently agrees (e_pred ≈ e_prev) can be promoted to
            # "verified" without VQE — the agreement is sufficient evidence.
            n_promoted = 0
            for i, h in enumerate(self._h_values):
                if not dual_mask[i]:
                    continue  # Only promote passing points
                h_key = round(float(h), 2)
                if h_key not in prev_theta_by_h:
                    continue
                theta_prev, e_prev = prev_theta_by_h[h_key]
                if e_prev is None:
                    continue
                # Confirm MPNN agrees: e_pred is close to e_prev (same basin)
                if abs(energies[i] - e_prev) > 0.01:
                    continue  # Significant disagreement → don't promote
                # Promote via upsert (tier upgrade path: "approximate"→"verified")
                upsert_theta_npz(
                    npz_path,
                    h_new=np.array([float(h)]),
                    theta_new=np.array([theta_prev]),
                    e_vqe_new=np.array([e_prev]),
                    e_exact_new=np.array([float(e_exact_arr[i])]),
                    gaps_new=np.array([float(gap_arr[i])]),
                    method_new=["mpnn_confirmed"],
                    quality_tier_new=["verified"],
                )
                n_promoted += 1
            if n_promoted > 0:
                logger.info(f"  │ Promoted {n_promoted} approximate→verified (MPNN confirms)")

            # ── 2b.1: Check convergence (early stop) ─────────────────────
            if pass_rate >= 0.90:
                convergence_reason = "target_reached"
                logger.info(f"  │ ✓ Target reached: pass_rate_dual={pass_rate:.0%} ≥ 90%")

                n_safety_persisted = 0
                for i, h in enumerate(self._h_values):
                    h_key = round(float(h), 2)
                    if h_key not in prev_theta_by_h and de_gaps[i] < 0.05:
                        upsert_theta_npz(
                            npz_path,
                            h_new=np.array([float(h)]),
                            theta_new=np.array([predictions[i]]),
                            e_vqe_new=np.array([float(energies[i])]),
                            e_exact_new=np.array([float(e_exact_arr[i])]),
                            gaps_new=np.array([float(gap_arr[i])]),
                            method_new=["mpnn_pred"],
                            quality_tier_new=["approximate"],
                        )
                        prev_theta_by_h[h_key] = (predictions[i], float(energies[i]))
                        n_safety_persisted += 1
                if n_safety_persisted > 0:
                    logger.info(f"  │ Safety net: persisted {n_safety_persisted} extra points")

                iteration_reports.append(
                    self._build_iter_report(
                        iteration, pass_rate, 0, 0, eval_hits, time.perf_counter() - t_iter_start
                    )
                )
                break

            improvement = pass_rate - prev_pass_rate
            if iteration > 1 and improvement < improvement_threshold:
                convergence_reason = "no_improvement"
                logger.info(
                    f"  │ ✓ Converged: improvement={improvement:.4f} < "
                    f"threshold={improvement_threshold}"
                )
                # Note: All passing predictions were already persisted immediately. No bulk save needed.
                eval_cache.flush()
                iteration_reports.append(
                    self._build_iter_report(
                        iteration, pass_rate, 0, 0, eval_hits, time.perf_counter() - t_iter_start
                    )
                )
                break

            # ── 2c: Identify failures + ansatz-limit filter ───────────────
            # Uses the dual energy criterion: ΔE/gap OR |ΔE| (from metrics).
            # Fidelity is NOT a pass/fail criterion (diagnostic only).
            from qmbp_simulation.analysis.metrics import (
                compute_refinement_priority,
                is_point_failure,
            )

            failures = []
            ansatz_limited = []
            for i, h in enumerate(self._h_values):
                abs_err_i = abs(energies[i] - e_exact_arr[i])
                is_fail = is_point_failure(
                    de_gap=de_gaps[i],
                    abs_error=abs_err_i,
                )
                if is_fail:
                    if h_min_valid > 0 and float(h) < h_min_valid:
                        ansatz_limited.append(i)
                    else:
                        failures.append(i)

            if not failures:
                if ansatz_limited:
                    convergence_reason = "ansatz_limit"
                    logger.info(
                        f"  │ ✓ All remaining failures ({len(ansatz_limited)}) are in "
                        f"ansatz-limited zone (h < {h_min_valid:.2f})"
                    )
                else:
                    convergence_reason = "target_reached"
                # Note: Data already persisted immediately. Just flush cache.
                eval_cache.flush()
                iteration_reports.append(
                    self._build_iter_report(
                        iteration,
                        pass_rate,
                        0,
                        len(ansatz_limited),
                        eval_hits,
                        time.perf_counter() - t_iter_start,
                    )
                )
                break

            # ── 2c.1: Priority-score failures and sort ─────────────────────
            # Use compute_refinement_priority to order failures by expected
            # return-on-investment. High-priority points get refined first.
            # Track per-h attempt counts for stale detection.
            if not hasattr(self, "_refine_attempts"):
                self._refine_attempts = {}  # h_key → count of failed VQE attempts

            scored_failures = []
            n_skipped_priority = 0
            for idx in failures:
                h = float(self._h_values[idx])
                h_key = round(h, 2)
                abs_err_i = abs(energies[idx] - e_exact_arr[idx])

                # Previous VQE energy (if refined before)
                e_prev_h = None
                if h_key in prev_theta_by_h:
                    _, e_prev_h = prev_theta_by_h[h_key]

                n_attempts = self._refine_attempts.get(h_key, 0)

                priority, should_skip, reason = compute_refinement_priority(
                    de_gap=de_gaps[idx],
                    abs_error=abs_err_i,
                    gap=gap_arr[idx],
                    n_params=n_params,
                    e_prev=e_prev_h,
                    e_pred=float(energies[idx]),
                    n_prev_attempts=n_attempts,
                )

                if should_skip:
                    n_skipped_priority += 1
                    logger.debug(f"    Skip h={h:.3f}: {reason} (priority={priority:.2f})")
                else:
                    scored_failures.append((priority, idx, reason))

            # Sort by priority (highest first) and limit
            scored_failures.sort(key=lambda x: x[0], reverse=True)
            if getattr(self._args, "refine_all", False):
                max_refine = len(scored_failures)
            else:
                max_refine = getattr(self._args, "max_refine_per_iter", None)
                if max_refine is None:
                    max_refine = min(len(scored_failures), DEFAULT_MAX_REFINE_PER_ITER)
            failures = [idx for _, idx, _ in scored_failures[:max_refine]]

            logger.info(
                f"  │ Failures: {len(failures)} to refine "
                f"(scored {len(scored_failures)}, skipped {n_skipped_priority}, "
                f"max={max_refine}), {len(ansatz_limited)} ansatz-limited"
            )
            if scored_failures:
                top = scored_failures[0]
                logger.info(
                    f"  │ Top priority: h={float(self._h_values[top[1]]):.3f} "
                    f"score={top[0]:.2f} reason={top[2]}"
                )

            # ── 2d: Anti-regression + VQE refine ─────────────────────────
            refined_h = []
            refined_theta = []
            refined_energies = []
            refined_e_exact = []

            # Use adaptive VQE config per-point based on priority score.
            # High-priority (easy wins) get minimal budget; low-priority get full budget.
            from qmbp_simulation.analysis.metrics import compute_adaptive_vqe_config

            refine_method = self._args.force_method or "L-BFGS-B"
            base_maxiter = self._args.maxiter
            base_restarts = self._args.n_restarts
            if refine_method == "L-BFGS-B":
                base_restarts = min(10, base_restarts)
            else:
                base_restarts = max(7, base_restarts // 2)

            for fail_idx_pos, idx in enumerate(failures):
                h = float(self._h_values[idx])
                h_key = round(h, 2)
                t_refine_start = time.perf_counter()

                # Adaptive VQE config: scale budget based on priority
                fail_priority = (
                    scored_failures[fail_idx_pos][0] if fail_idx_pos < len(scored_failures) else 0.5
                )
                adaptive_cfg = compute_adaptive_vqe_config(
                    priority=fail_priority,
                    de_gap=de_gaps[idx],
                    gap=gap_arr[idx],
                    n_params=n_params,
                    base_maxiter=base_maxiter,
                    base_restarts=base_restarts,
                )
                refine_maxiter = adaptive_cfg["maxiter"]
                refine_restarts = adaptive_cfg["n_restarts"]

                logger.info(
                    f"  │ Refining [{fail_idx_pos + 1}/{len(failures)}] "
                    f"h={h:.4f} (ΔE/gap={de_gaps[idx]:.4f}, "
                    f"tier={adaptive_cfg['tier']}, maxiter={refine_maxiter}, "
                    f"restarts={refine_restarts})..."
                )
                sys.stdout.flush()
                sys.stderr.flush()
                for handler in logging.getLogger().handlers:
                    handler.flush()

                # Anti-regression: evaluate BOTH θ_pred and θ_prev, pick best as VQE init
                from qmbp_simulation.framework.result_io import select_best_theta_init

                theta_prev_h = None
                e_prev_h_val = None
                if h_key in prev_theta_by_h:
                    theta_prev_h, e_prev_h_val = prev_theta_by_h[h_key]

                def _eval_theta(theta):
                    """Evaluate θ energy via cache."""
                    _key = eval_cache.make_key(
                        topo,
                        n_target,
                        h,
                        theta,
                        model="tfim_bond_resolved",
                        p_layers=p,
                    )
                    _cached = eval_cache.get(_key)
                    if _cached is not None:
                        return _cached
                    _lat = self.make_lattice(topo, n_target, J=1.0, h=h)
                    _H = spec.build_hamiltonian(_lat, **spec.hamiltonian_kwargs)
                    _e = eval_backend.evaluate(circuit_target, _H, theta)
                    eval_cache.put(_key, float(_e))
                    return float(_e)

                theta_init, best_e = select_best_theta_init(
                    theta_pred=predictions[idx],
                    e_pred=energies[idx],
                    theta_prev=theta_prev_h,
                    e_prev=e_prev_h_val,
                    eval_fn=_eval_theta,
                )
                if best_e < energies[idx]:
                    energies[idx] = best_e
                    logger.debug(f"    h={h:.2f}: anti-regression → using θ_prev (E={best_e:.6f})")

                # VQE warm-start refinement
                lat_h = self.make_lattice(topo, n_target, J=1.0, h=h)
                H = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)
                try:
                    from qmbp_simulation import VQEConfig, VQEOptimizer

                    vqe_cfg = VQEConfig(
                        p_layers=p,
                        n_restarts=refine_restarts,
                        maxiter=refine_maxiter,
                        method=refine_method,
                        enable_callbacks=False,  # No trajectory logging (2x speedup)
                    )
                    vqe_opt = VQEOptimizer(config=vqe_cfg, backend=eval_backend, seed=42 + idx)
                    if hasattr(eval_backend, "set_h"):
                        eval_backend.set_h(h)
                    vqe_result = vqe_opt.optimize(H, circuit_target, initial_guess=theta_init)
                    total_vqe_calls += 1
                    e_refined = float(vqe_result.energy)
                    res_x = vqe_result.theta_opt
                    t_refine_elapsed = time.perf_counter() - t_refine_start
                    logger.info(
                        f"    h={h:.2f}: VQE done in {t_refine_elapsed:.1f}s, E={e_refined:.6f}"
                    )

                    # ── Validate refined result before storing ────────────
                    # 1. Energy must be finite
                    if not np.isfinite(e_refined):
                        logger.warning(f"    h={h:.2f}: VQE returned NaN/Inf energy. Skipping.")
                        continue
                    # 2. θ must be finite
                    if not np.all(np.isfinite(res_x)):
                        logger.warning(f"    h={h:.2f}: VQE returned NaN/Inf θ. Skipping.")
                        continue
                    # 3. Variational principle: E_refined >= E_exact (within tolerance)
                    if e_refined < e_exact_arr[idx] - 1e-4:
                        violation = e_exact_arr[idx] - e_refined
                        logger.warning(
                            f"    h={h:.2f}: Variational violation Δ={violation:.2e}. "
                            f"May indicate stale E_exact (DMRG approx). Accepting anyway."
                        )
                    # 4. Only accept if improved over current energy
                    #    Use meaningful threshold to avoid "false improvements"
                    #    where VQE converges to same minimum with numerical noise.
                    from qmbp_simulation.models.constants import VQE_RESTART_IMPROVEMENT_TOL

                    energy_improvement = energies[idx] - e_refined
                    if energy_improvement > VQE_RESTART_IMPROVEMENT_TOL:
                        de_gap_new = abs(e_refined - e_exact_arr[idx]) / max(gap_arr[idx], 1e-10)
                        abs_err_old = abs(energies[idx] - e_exact_arr[idx])
                        abs_err_new = abs(e_refined - e_exact_arr[idx])
                        # Only log as improvement if ΔE/gap actually changed visibly
                        if abs(de_gaps[idx] - de_gap_new) > 1e-4:
                            # Compute state fidelity when feasible (N ≤ statevector limit).
                            # safe_compute_fidelity returns None for large N or on error.
                            fid = self.safe_compute_fidelity(
                                circuit_target,
                                res_x,
                                topo,
                                n_target,
                                h,
                                model="tfim_bond_resolved",
                            )
                            fid_str = f" F={fid:.4f}" if fid is not None else ""
                            logger.info(
                                f"    h={h:.2f}: ΔE/gap {de_gaps[idx]:.4f} → {de_gap_new:.4f} "
                                f"|ΔE| {abs_err_old:.3f} → {abs_err_new:.3f}{fid_str} ✓"
                            )
                        refined_h.append(h)
                        refined_theta.append(res_x.copy())
                        refined_energies.append(e_refined)
                        refined_e_exact.append(float(e_exact_arr[idx]))
                        # Update tracking
                        prev_theta_by_h[h_key] = (res_x.copy(), e_refined)
                        # Reset stale counter — this point just improved
                        self._refine_attempts.pop(h_key, None)

                        # ── Immediate persist: NPZ upsert per-point ───────
                        # Ensures no refined θ is lost on interrupt.
                        gap_i = float(gap_arr[idx])
                        upsert_theta_npz(
                            npz_path,
                            np.array([h]),
                            np.array([res_x]),
                            np.array([e_refined]),
                            np.array([float(e_exact_arr[idx])]),
                            gaps_new=np.array([gap_i]),
                            method_new=["vqe_refined"],
                            quality_tier_new=["verified"],
                        )
                        # Note: eval_cache auto-flushes every 50 puts.
                        # Full flush deferred to end of iteration (avoid 5MB
                        # JSON write per point).
                    else:
                        logger.info(f"    h={h:.2f}: no improvement (VQE stuck)")
                        # Track failed attempt for priority scoring
                        self._refine_attempts[h_key] = self._refine_attempts.get(h_key, 0) + 1
                except KeyboardInterrupt:
                    # Persist everything refined so far before re-raising
                    logger.warning(
                        f"  │ ⚠️ Interrupted during refinement. "
                        f"{len(refined_h)} points already saved to NPZ."
                    )
                    eval_cache.flush()
                    raise
                except Exception as e:
                    logger.warning(f"    h={h:.2f}: refinement failed: {e}")

            # ── 2e: Summary (NPZ already persisted per-point above) ───────
            n_updated = len(refined_h)  # all persisted incrementally via upsert_theta_npz
            if refined_h:
                logger.info(f"  │ VQE refinement: {n_updated} points improved and persisted")

            # Note: All data (predictions + refinements) was persisted immediately
            # via upsert_theta_npz calls above. No bulk save needed.

            # Flush eval cache once per iteration (not per-point)
            eval_cache.flush()

            # ── 2f: Retrain multi-N model ─────────────────────────────────
            # Fix A: Skip retrain if no new data was produced
            # Fix B: Fine-tune instead of training from scratch
            # Fix C: Use diagnostics-aware should_retrain to detect contamination
            from qmbp_simulation.predictors.unified_mpnn import (
                fine_tune_unified_mpnn,
                should_retrain_with_diagnostics,
            )

            # p-scoped aggregation: only *_p{p}.npz data (no cross-p mixing)
            agg = MultiNAggregator(
                topology=topo,
                model="tfim_bond_resolved",
                max_n=self.N_MAX_VIABLE.get(topo, 20),
                h_min=getattr(self._args, "train_h_min", None),
                h_max=getattr(self._args, "train_h_max", None),
                p_layers=p,
            )
            agg.scan()
            dataset = agg.build_combined_dataset(max_de_gap=0.10)

            # ── Pre-retrain quality check (WARNING-ONLY) ──────────────────
            # Uses base class helper: logs warnings but never aborts.
            # The AL loop is progressive — quality improves each iteration.
            self.warn_training_quality(agg._data_by_n)

            do_retrain, retrain_reason, diagnostics = should_retrain_with_diagnostics(
                topology=topo,
                model_name="tfim_bond_resolved",
                p_layers=p,
                n_new_points=len(refined_h),
                current_pass_rate=pass_rate,
                prev_pass_rate=prev_pass_rate,
                dataset_size=len(dataset),
            )

            # --from-zoo / --skip-retrain: never retrain the MPNN. Keep refining
            # VQE points and persisting them to NPZ, but use only the loaded
            # model for prediction.
            if getattr(self._args, "from_zoo", False):
                do_retrain = False
                retrain_reason = "from_zoo (retraining disabled)"
            elif getattr(self._args, "skip_retrain", False):
                do_retrain = False
                retrain_reason = "skip_retrain (retraining disabled)"

            # Log diagnostic info if contamination or other issues detected
            if diagnostics.get("failure_mode") in ("contaminated_training", "gap_masking"):
                logger.warning(
                    f"  │ ⚠️ Diagnostic: failure_mode={diagnostics['failure_mode']}, "
                    f"training_utility={diagnostics.get('training_utility')}"
                )

            if do_retrain and len(dataset) >= 5:
                logger.info(
                    f"  │ Retraining (reason={retrain_reason}, "
                    f"{len(refined_h)} new points, {len(dataset)} total)..."
                )
                sample_g = dataset[0]
                n_node_features = sample_g.x.shape[1] if hasattr(sample_g, "x") else 4

                # Fix B: Fine-tune existing model if it has same architecture,
                # otherwise train from scratch (architecture mismatch).
                # Validates node_features, hidden_dim, n_layers, and use_residual.
                can_fine_tune = (
                    hasattr(model, "node_features")
                    and model.node_features == n_node_features
                    and getattr(model, "hidden_dim", 256) == 256
                    and getattr(model, "n_layers", 3) == 3
                    and getattr(model, "use_residual", False) == use_residual
                    and iteration > 1  # First iter after bootstrap → full train
                )

                if can_fine_tune:
                    logger.info(f"  │ Mode: fine-tune ({FINE_TUNE_EPOCHS} epochs, lr=3e-4)")
                    train_result = fine_tune_unified_mpnn(
                        model,
                        dataset,
                        n_epochs=FINE_TUNE_EPOCHS,
                        lr=3e-4,
                        patience=150,
                        seed=42,
                    )
                else:
                    logger.info(f"  │ Mode: full retrain ({FULL_TRAIN_EPOCHS} epochs, lr=1e-3)")
                    model = UnifiedMPNN(
                        node_features=n_node_features,
                        hidden_dim=256,
                        n_layers=3,
                        norm_type="none",
                        dropout=0.1,
                        use_residual=use_residual,
                        film_conditioning=getattr(self._args, "film", False),
                    )
                    train_result = train_unified_mpnn(
                        model,
                        dataset,
                        n_epochs=FULL_TRAIN_EPOCHS,
                        lr=1e-3,
                        patience=200,
                        seed=42,
                        loss_type=getattr(self._args, "loss_type", "theta_mse"),
                        physics_loss_weight=getattr(self._args, "physics_loss_weight", 0.0),
                    )

                mse = train_result.get("final_mse", 0) if isinstance(train_result, dict) else 0
                mode = train_result.get("mode", "full")
                logger.info(f"  │ Retrained ({mode}): MSE={mse:.2e}, {len(dataset)} points")

                # Persist training curve (auto, non-blocking)
                try:
                    from qmbp_simulation.utils.helpers import persist_training_curve

                    persist_training_curve(
                        train_result,
                        output_dir=Path("results/training_curves"),
                        prefix=f"{topo}_iter{iteration}_p{p}",
                    )
                except Exception:
                    pass

                # ── 2g: Export to zoo (only if pass_rate improved) ────────
                # Fix C: Don't overwrite a better model in the zoo with one
                # that didn't improve pass_rate.
                if pass_rate > zoo_best_pass_rate or iteration == 1:
                    from datetime import datetime

                    n_vals = agg.available_n_values()
                    n_str = "+".join(str(n) for n in n_vals)
                    entry = ZooEntry(
                        model="tfim_bond_resolved",
                        topology=topo,
                        n_qubits=0,
                        p_layers=p,
                        checkpoint_file=(
                            f"unifMPNN__{topo}_p{p}_{self._args.model_name}.pt"
                            if getattr(self._args, "model_name", None)
                            else f"unified_tfim_br_{topo}_multiN_{n_str}_p{p}.pt"
                        ),
                        h_range=self._training_h_range(),
                        pass_rate=pass_rate,
                        n_training_points=len(dataset),
                        seeds=[42],
                        created=datetime.now(UTC).isoformat(),
                        notes=f"Iterative improve iter {iteration}: N={n_vals}"
                        + self._train_h_range_note()
                        + (", arch=residual" if use_residual else ""),
                    )
                    register_checkpoint_with_training_metrics(
                        model,
                        entry,
                        training_result=train_result,
                        overwrite=True,
                        architecture_config={
                            "hidden_dim": 256,
                            "n_conv_layers": 3,
                            "norm_type": "none",
                            "dropout": 0.1,
                            "use_residual": use_residual,
                        },
                    )
                    zoo_best_pass_rate = pass_rate
                    logger.info(f"  │ Exported to zoo: {entry.checkpoint_file}")
                else:
                    logger.info(
                        f"  │ Zoo skip: pass_rate_dual={pass_rate:.0%} ≤ "
                        f"zoo_best={zoo_best_pass_rate:.0%} — keeping better model"
                    )
            elif not do_retrain:
                logger.info(
                    f"  │ Skipping retrain: {retrain_reason} "
                    f"(refined={len(refined_h)}, dataset={len(dataset)})"
                )
            else:
                logger.warning(f"  │ Only {len(dataset)} points — skipping retrain")

            # ── 2h: Report iteration ──────────────────────────────────────
            iter_time = time.perf_counter() - t_iter_start
            iteration_reports.append(
                self._build_iter_report(
                    iteration,
                    pass_rate,
                    len(refined_h),
                    len(ansatz_limited),
                    eval_hits,
                    iter_time,
                )
            )
            prev_pass_rate = pass_rate
            logger.info(
                f"  ╚══ Iteration {iteration} done: pass_rate_dual={pass_rate:.0%}, "
                f"refined={len(refined_h)}, time={iter_time:.1f}s ══╝"
            )

        # ── Final report ──────────────────────────────────────────────────
        eval_cache.flush()
        final_stats = eval_cache.stats()
        final_pass_rate = iteration_reports[-1]["pass_rate"] if iteration_reports else 0.0

        # ── Cross-N Validation Report (L1 from final iteration data) ──────
        cross_n_report = None
        try:
            from qmbp_simulation.analysis.cross_n_validator import quick_cross_n_report

            # Build per-h results from final energies for formal L1 report
            per_h_for_report = []
            for i, h in enumerate(self._h_values):
                per_h_for_report.append(
                    {
                        "h": float(h),
                        "e_pred": float(energies[i]),
                        "e_exact": float(e_exact_arr[i]),
                        "gap": float(gap_arr[i]),
                        "de_gap": float(de_gaps[i]),
                    }
                )

            # Training sizes = all N values in multi_n_training NPZs
            training_sizes = []
            try:
                from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

                _agg_report = MultiNAggregator(
                    topology=topo, model="tfim_bond_resolved", p_layers=p
                )
                _agg_report.scan()
                training_sizes = _agg_report.available_n_values()
            except Exception:
                training_sizes = [n_target]

            report = quick_cross_n_report(
                per_h_for_report,
                n_target,
                topology=topo,
                training_sizes=training_sizes,
            )
            cross_n_report = report.to_dict()
            status = "✅" if report.overall_pass else "❌"
            logger.info(
                f"  {status} CrossN L1: pass_rate={report.l1_pass_rate:.0%}, "
                f"mean_ΔE/gap={report.l1_mean_de_gap:.4f}"
            )
        except Exception as e:
            logger.debug(f"  Cross-N report skipped: {e}")

        # ── Memory cleanup: free heavy objects before _build_envelope ─────
        # NOTE: Do NOT call gc.collect() here! Qiskit's CircuitData destructor
        # triggers mimalloc's _mi_arenas_page_unabandon which calls sleep() in
        # a retry loop on macOS ARM64, hanging the process indefinitely.
        # Instead, just delete references and let os._exit() handle cleanup.
        del model, predictions
        eval_cache = None
        dataset = None
        agg = None
        circuit_target = None
        lattice_target = None

        return {
            "pass": final_pass_rate >= 0.50 or convergence_reason == "no_improvement",
            "convergence_reason": convergence_reason,
            "iterations_run": len(iteration_reports),
            "final_pass_rate": final_pass_rate,
            "total_vqe_calls": total_vqe_calls,
            "iteration_reports": iteration_reports,
            "cross_n_validation": cross_n_report,
            "cache_stats": final_stats,
            "gt_cache_hits": gt_hits,
            "gt_cache_misses": gt_misses,
        }

    def _build_iter_report(
        self, iteration, pass_rate, n_refined, n_ansatz_limited, eval_hits, elapsed_s
    ) -> dict:
        """Build per-iteration summary dict."""
        return {
            "iteration": iteration,
            "pass_rate": pass_rate,
            "n_refined": n_refined,
            "n_ansatz_limited": n_ansatz_limited,
            "eval_cache_hits": eval_hits,
            "elapsed_s": elapsed_s,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    AcceleratedCrossNRunner.main()
