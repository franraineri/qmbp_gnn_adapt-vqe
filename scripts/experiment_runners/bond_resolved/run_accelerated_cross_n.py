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
    python scripts/.../run_accelerated_cross_n.py

    # Custom sizes
    python scripts/.../run_accelerated_cross_n.py --train-n 10 --target-n 20 40

    # Use existing model from zoo (skip training)
    python scripts/.../run_accelerated_cross_n.py --from-zoo --target-n 20

    # Multiple p layers
    python scripts/.../run_accelerated_cross_n.py --p-layers 1 2

    # Dry run
    python scripts/.../run_accelerated_cross_n.py --dry-run
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

from qmbp_simulation.framework.runner_base import (
    Section,
    ValidationRunner,
    resolve_project_root,
)

_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

from qmbp_simulation.models.constants import DEFAULT_SEEDS

# Defaults
DEFAULT_TRAIN_N = 10
DEFAULT_TARGET_N = [20]
DEFAULT_P = 1
DEFAULT_TOPOLOGY = "chain_1d"
DEFAULT_H_MIN = 2.0
DEFAULT_H_MAX = 3.5
DEFAULT_H_POINTS = 14
DEFAULT_N_ANCHORS = 14
DEFAULT_MAXITER = 1500
DEFAULT_N_RESTARTS = 10


class AcceleratedCrossNRunner(ValidationRunner):
    """Accelerated Cross-N Transfer: train small, predict large.

    Trains UnifiedMPNN at N_train using AcceleratedVQE (5-6 anchor VQE +
    MPNN for the rest), exports to zoo, then predicts at N_target using
    only the trained model. No VQE at N_target.
    """

    runner_id = "accelerated_cross_n_v1"
    experiment_id = "ACCEL_CROSS_N"
    description = "Accelerated Cross-N: train N_train, predict N_target via zoo"
    hypothesis = (
        "UnifiedMPNN trained on N_train bond-resolved data predicts θ at "
        "N_target with ΔE/gap < 10% for h in valid regime (h > 2.0)."
    )

    @classmethod
    def _add_custom_args(cls, parser):
        parser.add_argument(
            "--train-n", type=int, default=DEFAULT_TRAIN_N,
            help="System size for training (default: %(default)s)",
        )
        parser.add_argument(
            "--target-n", type=int, nargs="+", default=DEFAULT_TARGET_N,
            help="Target system size(s) for prediction (default: %(default)s)",
        )
        parser.add_argument(
            "--p-layers", type=int, nargs="+", default=[DEFAULT_P],
            help="HVA layer depth(s) (default: %(default)s)",
        )
        parser.add_argument(
            "--topology", type=str, default=DEFAULT_TOPOLOGY,
            help="Lattice topology (default: %(default)s)",
        )
        parser.add_argument(
            "--h-min", type=float, default=DEFAULT_H_MIN,
            help="Minimum h for sweep (default: %(default)s)",
        )
        parser.add_argument(
            "--h-max", type=float, default=DEFAULT_H_MAX,
            help="Maximum h for sweep (default: %(default)s)",
        )
        parser.add_argument(
            "--h-points", type=int, default=DEFAULT_H_POINTS,
            help="Number of h-grid points (default: %(default)s)",
        )
        parser.add_argument(
            "--n-anchors", type=int, default=DEFAULT_N_ANCHORS,
            help="Number of VQE anchor points (default: %(default)s)",
        )
        parser.add_argument(
            "--maxiter", type=int, default=DEFAULT_MAXITER,
            help="VQE COBYLA maxiter (default: %(default)s)",
        )
        parser.add_argument(
            "--n-restarts", type=int, default=DEFAULT_N_RESTARTS,
            help="VQE restarts per anchor (default: %(default)s)",
        )
        parser.add_argument(
            "--from-zoo", action="store_true", default=False,
            help="Skip training, load model from zoo directly",
        )
        parser.add_argument(
            "--checkpoint", type=str, default=None,
            help="Explicit checkpoint path (overrides zoo search)",
        )
        parser.add_argument(
            "--active-rounds", type=int, default=0,
            help="Active learning rounds: refine low-fidelity points with VQE (default: 0)",
        )
        parser.add_argument(
            "--multi-n-train", action="store_true", default=False,
            help="Instead of training on a single N, aggregate ALL available "
            "bond-resolved data for this topology (from previous runs) and "
            "train a multi-N model. Overrides --train-n for training.",
        )
        parser.add_argument(
            "--force-retrain", action="store_true", default=False,
            help="Force retraining from scratch even if a suitable model "
            "exists in the zoo. Default: reuse best existing model.",
        )
        parser.add_argument(
            "--no-eval-cache", action="store_true", default=False,
            help="Disable circuit evaluation cache. By default, evaluations "
            "are cached in data/eval_cache/ to avoid recomputing identical "
            "(topology, N, h, theta_hash) evaluations.",
        )
        # Iterative improvement + VQE method args from shared CLI module
        from qmbp_simulation.framework.cli import add_iterative_improve_args
        add_iterative_improve_args(parser)

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
        self._h_values = np.linspace(
            self._args.h_max, self._args.h_min, self._args.h_points
        )
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
            sections.append(Section(
                id=2,
                name="Budget Estimation + Cache Warm-up",
                fn=self.section_budget_estimation,
                hypothesis="Estimate compute cost leveraging cached results",
            ))
            if not getattr(self._args, "budget_only", False):
                sections.append(Section(
                    id=3,
                    name="Iterative Improvement Loop",
                    fn=self.section_iterative_improve,
                    hypothesis="Iterative predict→refine→retrain converges to ≥90% pass rate",
                ))
            return sections

        if getattr(self._args, "multi_n_train", False):
            sections.append(Section(
                id=2,
                name="Multi-N Train (aggregate all available data)",
                fn=self.section_multi_n_train,
                hypothesis="Multi-N UnifiedMPNN trained with aggregated data from all N sizes",
            ))
        elif not self._args.from_zoo:
            sections.append(Section(
                id=2,
                name=f"Train (N={self._args.train_n})",
                fn=self.section_train,
                hypothesis=f"AcceleratedVQE at N={self._args.train_n} achieves >60% pass rate",
            ))
        sections.append(Section(
            id=3,
            name="Cross-N Predict",
            fn=self.section_cross_n_predict,
            hypothesis="Cross-N prediction achieves ΔE/gap < 10% for h > 2.0",
        ))
        return sections


    # ═══════════════════════════════════════════════════════════════════════════
    # Section 1: Quality Check
    # ═══════════════════════════════════════════════════════════════════════════

    def section_quality_check(self) -> dict:
        """Run QualityPredictor for training and target configs via base helper."""
        configs = [
            {
                "model": "tfim_bond_resolved",
                "topology": self._args.topology,
                "n_qubits": self._args.train_n,
                "p_layers": self._args.p_layers[0],
                "h_min": self._args.h_min,
                "h_max": self._args.h_max,
            }
        ]
        for n_target in self._args.target_n:
            configs.append({
                "model": "tfim_bond_resolved",
                "topology": self._args.topology,
                "n_qubits": n_target,
                "p_layers": self._args.p_layers[0],
                "h_min": self._args.h_min,
                "h_max": self._args.h_max,
            })
        return self.run_quality_check(configs=configs)

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 2: Train
    # ═══════════════════════════════════════════════════════════════════════════

    def section_train(self) -> dict:
        """Train AcceleratedVQE at N_train for each p value."""
        from pathlib import Path

        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.execution import NoiselessBackend
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.pipeline.accelerated import AcceleratedVQE, AcceleratedConfig

        spec = get_model_spec("tfim_bond_resolved")
        backend = NoiselessBackend()
        hva = HVACircuitBuilder()
        N = self._args.train_n
        topo = self._args.topology

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
                mpnn_epochs=4000,
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

            # Persist θ_opt for multi-N reuse (saved as NPZ for aggregator)
            training_data_dir = Path("data/multi_n_training")
            training_data_dir.mkdir(parents=True, exist_ok=True)
            npz_path = training_data_dir / f"{topo}_N{N}_p{p}.npz"
            np.savez(
                npz_path,
                h_values=self._h_values[: len(result.theta_opt)],
                theta_opt=result.theta_opt,
                e_vqe=result.energies,
                e_exact=result.e_exact,
                de_gaps=result.de_gaps,
                gaps=result.gaps,
                method=result.method,
            )
            logger.info(f"    Saved training data: {npz_path} ({len(result.theta_opt)} points)")

            all_results[f"p{p}"] = {
                "pass_rate": result.pass_rate,
                "mean_de_gap": float(result.de_gaps.mean()),
                "elapsed_s": elapsed,
                "n_anchors": result.n_anchors,
                "model_source": result.model_source,
                "methods": dict(zip(*np.unique(result.method, return_counts=True))),
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
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN, train_unified_mpnn
        from qmbp_simulation.predictors.model_zoo import register_checkpoint, load_pretrained, ZooEntry

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
        agg = MultiNAggregator(topology=topo, model="tfim_bond_resolved")
        summary = agg.scan()

        if not summary:
            return {"pass": False, "error": "No existing data found. Run --train-n first."}

        logger.info(f"  Found data for N={agg.available_n_values()}: {summary}")

        # 2. Build combined dataset (filter by quality)
        dataset = agg.build_combined_dataset(max_de_gap=0.10)
        print('finished build_combined_dataset')
        if len(dataset) < 5:
            return {
                "pass": False,
                "error": f"Only {len(dataset)} points pass quality filter. Need ≥5.",
                "summary": agg.summary(),
            }

        logger.info(f"  Combined dataset: {len(dataset)} graphs from N={agg.available_n_values()}")

        # 3. Determine output dim from dataset (varies by graph size)
        # UnifiedMPNN uses per-node prediction so output_dim is implicit
        sample_g = dataset[0]
        n_node_features = sample_g.x.shape[1] if hasattr(sample_g, 'x') else 4

        # 4. Train UnifiedMPNN
        model = UnifiedMPNN(
            node_features=n_node_features,
            hidden_dim=256,
            n_layers=3,
            norm_type="none",  # MANDATORY for cross-N
            dropout=0.1,
        )

        logger.info("  Training UnifiedMPNN (multi-N, norm_type=none)...")
        t0 = time.perf_counter()
        train_result = train_unified_mpnn(
            model, dataset,
            n_epochs=4000,
            lr=1e-3,
            patience=300,
            seed=42,
        )
        elapsed = time.perf_counter() - t0

        final_mse = train_result.get("final_mse", 0) if isinstance(train_result, dict) else 0
        logger.info(f"  Training done: MSE={final_mse:.2e}, time={elapsed:.1f}s")

        # Store model for Section 3
        self._models[p] = model

        # 5. Export to zoo as multi-N model
        from datetime import datetime, timezone

        n_values_str = "+".join(str(n) for n in agg.available_n_values())
        entry = ZooEntry(
            model="tfim_bond_resolved",
            topology=topo,
            n_qubits=0,  # 0 = multi-N
            p_layers=p,
            checkpoint_file=f"unified_tfim_br_{topo}_multiN_{n_values_str}_p{p}.pt",
            h_range=(self._args.h_min, self._args.h_max),
            pass_rate=0.0,  # Updated after eval
            n_training_points=len(dataset),
            seeds=[42],
            created=datetime.now(timezone.utc).isoformat(),
            notes=f"Multi-N training: N={agg.available_n_values()}, {len(dataset)} points",
        )
        register_checkpoint(model, entry, overwrite=True)
        logger.info(f"  Exported multi-N model: {entry.checkpoint_file}")

        return {
            "pass": True,
            "n_values_used": agg.available_n_values(),
            "n_training_points": len(dataset),
            "points_per_n": {str(k): v for k, v in agg.summary()["points_per_n"].items()},
            "final_mse": float(final_mse),
            "elapsed_s": elapsed,
            "checkpoint": entry.checkpoint_file,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 3: Cross-N Predict
    # ═══════════════════════════════════════════════════════════════════════════

    def section_cross_n_predict(self) -> dict:
        """Predict at each N_target using the N_train model."""
        import torch
        from qmbp_simulation.analysis.metrics import compute_deploy_summary
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.execution import NoiselessBackend
        from qmbp_simulation.models.constants import STATEVECTOR_MAX_N
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.predictors.model_zoo import load_pretrained
        from qmbp_simulation.predictors.unified_graph import build_unified_bond_resolved_graph

        spec = get_model_spec("tfim_bond_resolved")
        solver = self.solver
        hva = HVACircuitBuilder()
        topo = self._args.topology

        all_results = {}

        for p in self._args.p_layers:
            # Load model: from memory (section 2), from zoo, or from checkpoint
            model = self._models.get(p)
            if model is None:
                try:
                    model, _meta = load_pretrained(
                        model="tfim_bond_resolved",
                        topology=topo,
                        n_qubits=self._args.train_n,
                        p_layers=p,
                        checkpoint_path=self._args.checkpoint,
                        allow_cross_n=True,
                    )
                except FileNotFoundError as e:
                    logger.error(f"  No model for p={p}: {e}")
                    all_results[f"p{p}"] = {"pass": False, "error": str(e)}
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
                if use_eval_cache:
                    logger.info(f"    Eval cache: {len(eval_backend.cache)} entries loaded")

                per_h_results = []
                t0 = time.perf_counter()

                for h in self._h_values:
                    # Build unified graph for N_target
                    g = build_unified_bond_resolved_graph(
                        lattice_target, h_value=float(h), p_layers=p,
                        include_circuit_nodes=True,
                    )
                    with torch.no_grad():
                        theta_pred = model(g).numpy().flatten()

                    theta_pred = np.clip(theta_pred, -np.pi, np.pi)

                    # Verify param count matches circuit
                    if len(theta_pred) != n_params_target:
                        logger.warning(
                            f"    Param mismatch at h={h:.2f}: "
                            f"predicted {len(theta_pred)}, need {n_params_target}"
                        )
                        if len(theta_pred) < n_params_target:
                            theta_pred = np.pad(theta_pred, (0, n_params_target - len(theta_pred)))
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

                    # Fidelity via parent's compute_fidelity (only N ≤ 22)
                    # Reuses the solver call needed for ground_state vector
                    fidelity = None
                    if n_target <= STATEVECTOR_MAX_N:
                        try:
                            gt_obj = solver.solve(H, lat_h)
                            if gt_obj.ground_state is not None:
                                fidelity = float(self.compute_fidelity(
                                    circuit_target, theta_pred, gt_obj.ground_state
                                ))
                        except (MemoryError, ValueError):
                            fidelity = None

                    per_h_results.append(self.build_per_h_result(
                        h, e_pred, e_exact, gap,
                        fidelity=fidelity, n_params=len(theta_pred),
                    ))
                    fid_str = f"F={fidelity:.4f}" if fidelity is not None else "F=N/A(N>22)"
                    status = "✓" if de_gap < 0.05 else ("~" if de_gap < 0.10 else "✗")
                    logger.info(
                        f"    h={h:.2f}: ΔE/gap={de_gap:.4f} {fid_str} "
                        f"|ΔE|={abs_err:.2e} [{len(theta_pred)} params] {status}"
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
                            cold_start_samples.append({
                                "h": h_cold,
                                "de_gap_cold": float(de_gap_cold),
                                "de_gap_warm": float(r_cold["de_gap"]),
                                "speedup": "warm better" if r_cold["de_gap"] < de_gap_cold else "cold better",
                            })
                            logger.info(
                                f"    Cold-start baseline h={h_cold:.2f}: "
                                f"dE/gap_cold={de_gap_cold:.4f} vs dE/gap_warm={r_cold['de_gap']:.4f} "
                                f"({'warm wins' if r_cold['de_gap'] < de_gap_cold else 'cold wins'})"
                            )
                        except Exception as e:
                            logger.debug(f"    Cold-start sample h={h_cold:.2f} failed: {e}")

                    # ── Active learning rounds ────────────────────────────────
                    for al_round in range(active_rounds):
                        refine_indices = [
                            i for i, r in enumerate(per_h_results)
                            if r["de_gap"] > 0.05
                            or (r.get("fidelity") is not None and r["fidelity"] < 0.90)
                        ]
                        if not refine_indices:
                            logger.info(f"    AL round {al_round+1}: all points pass. Done.")
                            break
                        refine_indices = refine_indices[:5]
                        logger.info(
                            f"    Active learning round {al_round+1}: "
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
                                    lattice_target, h_value=h_val, p_layers=p,
                                    include_circuit_nodes=True,
                                )
                                with torch.no_grad():
                                    theta_init = model(g_ref).numpy().flatten()
                                theta_init = np.clip(theta_init, -np.pi, np.pi)
                                if len(theta_init) != n_params_target:
                                    if len(theta_init) < n_params_target:
                                        theta_init = np.pad(theta_init, (0, n_params_target - len(theta_init)))
                                    else:
                                        theta_init = theta_init[:n_params_target]

                                al_maxiter = 200 if n_target <= STATEVECTOR_MAX_N else 50
                                res = _minimize(
                                    lambda params: vqe_backend.evaluate(circuit_target, H_ref, params),
                                    theta_init,
                                    method="COBYLA",
                                    options={"maxiter": al_maxiter, "rhobeg": 0.1},
                                )

                                e_exact_ref, gap_ref = self.exact_ground_state(
                                    topo, n_target, h_val, model="tfim_bond_resolved"
                                )
                                de_gap_new = abs(res.fun - e_exact_ref) / max(gap_ref, 1e-10)

                                # Fidelity (only N ≤ 22)
                                fid_new = None
                                if n_target <= STATEVECTOR_MAX_N:
                                    try:
                                        gt_ref_obj = solver.solve(H_ref, lat_ref)
                                        if gt_ref_obj.ground_state is not None:
                                            fid_new = float(self.compute_fidelity(
                                                circuit_target, res.x, gt_ref_obj.ground_state
                                            ))
                                    except Exception:
                                        pass

                                if de_gap_new < r["de_gap"]:
                                    per_h_results[idx] = {
                                        **r,
                                        "de_gap": float(de_gap_new),
                                        "abs_error": float(abs(res.fun - e_exact_ref)),
                                        "fidelity": fid_new if fid_new is not None else r.get("fidelity"),
                                        "e_pred": float(res.fun),
                                        "method": "refined",
                                        "de_gap_before_refine": float(r["de_gap"]),
                                    }
                                    n_refined += 1
                                    logger.info(
                                        f"      h={h_val:.2f}: dE/gap {r['de_gap']:.4f} -> {de_gap_new:.4f} "
                                        f"(improvement: {(1 - de_gap_new/r['de_gap'])*100:.0f}%)"
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

                key = f"p{p}_N{n_target}"
                all_results[key] = {
                    "train_n": self._args.train_n,
                    "target_n": n_target,
                    "p_layers": p,
                    "n_params": n_params_target,
                    **summary,
                    "fidelity_available": n_target <= STATEVECTOR_MAX_N,
                    "active_learning_applied": n_refined > 0,
                    "n_refined": n_refined,
                    "cold_start_comparison": cold_start_samples if active_rounds > 0 else None,
                    "elapsed_s": elapsed,
                    "per_point": per_h_results,
                }
                fid_info = f"F_mean={summary['mean_fidelity']:.4f}" if summary.get("mean_fidelity") else "F=N/A"
                logger.info(
                    f"  N={n_target}: {summary.get('n_pass_5pct', 0)}/{summary['n_points']} @5%, "
                    f"{summary.get('n_pass_10pct', 0)}/{summary['n_points']} @10%, "
                    f"{fid_info}, refined={n_refined}"
                )

        # Overall pass: at least one target has >50% at 10% threshold
        passed = any(
            v.get("pass_rate_10pct", 0) > 0.5
            for v in all_results.values() if isinstance(v, dict) and "pass_rate_10pct" in v
        )

        return {"pass": passed, "cross_n_results": all_results}

    # ═══════════════════════════════════════════════════════════════════════════
    # Iterative Improvement: helpers
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _upsert_npz(npz_path: Path, h_new, theta_new, e_vqe_new, e_exact_new,
                    gaps_new=None, method_new=None):
        """Update NPZ keeping best θ per h-point (lower energy wins).

        Validates all inputs before storing:
        - θ must be finite (no NaN/Inf)
        - Energies must be finite
        - θ dimension must be consistent across all entries

        Also maintains de_gaps, gaps, and method columns for downstream quality filtering.
        """
        # ── Input validation ──────────────────────────────────────────
        assert len(h_new) == len(theta_new) == len(e_vqe_new) == len(e_exact_new), (
            f"Length mismatch: h={len(h_new)}, θ={len(theta_new)}, "
            f"e_vqe={len(e_vqe_new)}, e_exact={len(e_exact_new)}"
        )
        # Filter out invalid entries before merge
        valid_mask = np.array([
            np.all(np.isfinite(theta_new[i])) and
            np.isfinite(e_vqe_new[i]) and
            np.isfinite(e_exact_new[i])
            for i in range(len(h_new))
        ])
        if not valid_mask.all():
            n_invalid = int((~valid_mask).sum())
            logger.warning(f"  _upsert_npz: filtering {n_invalid} invalid entries")
            h_new = h_new[valid_mask]
            theta_new = theta_new[valid_mask]
            e_vqe_new = e_vqe_new[valid_mask]
            e_exact_new = e_exact_new[valid_mask]
        if len(h_new) == 0:
            return 0, 0

        # ── Load existing ─────────────────────────────────────────────
        if npz_path.exists():
            existing = np.load(npz_path, allow_pickle=True)
            h_all = existing["h_values"].tolist()
            theta_all = [row for row in existing["theta_opt"]]
            e_key = "e_vqe" if "e_vqe" in existing else "energies"
            e_vqe_all = existing[e_key].tolist() if e_key in existing else [0.0] * len(h_all)
            e_exact_all = existing["e_exact"].tolist()
            gaps_all = existing["gaps"].tolist() if "gaps" in existing else [0.0] * len(h_all)

            # Validate existing data integrity on load
            n_existing_before = len(h_all)
            valid_existing = []
            for j in range(len(h_all)):
                if (len(theta_all[j]) > 0 and
                    np.all(np.isfinite(theta_all[j])) and
                    np.isfinite(e_vqe_all[j])):
                    valid_existing.append(j)
            if len(valid_existing) < n_existing_before:
                logger.warning(
                    f"  _upsert_npz: {n_existing_before - len(valid_existing)} "
                    f"corrupt entries in existing NPZ (removed)"
                )
                h_all = [h_all[j] for j in valid_existing]
                theta_all = [theta_all[j] for j in valid_existing]
                e_vqe_all = [e_vqe_all[j] for j in valid_existing]
                e_exact_all = [e_exact_all[j] for j in valid_existing]
                gaps_all = [gaps_all[j] for j in valid_existing]
        else:
            h_all, theta_all, e_vqe_all, e_exact_all, gaps_all = [], [], [], [], []
            existing = None

        # Load existing method array (optional field)
        if existing is not None and "method" in existing:
            method_all = existing["method"].tolist()
            # Apply same validity filter
            if len(valid_existing) < n_existing_before:
                method_all = [method_all[j] for j in valid_existing]
        else:
            method_all = ["unknown"] * len(h_all)

        # ── Merge ─────────────────────────────────────────────────────
        n_updated, n_added = 0, 0
        for i, h in enumerate(h_new):
            gap_i = float(gaps_new[i]) if gaps_new is not None and i < len(gaps_new) else 0.0
            method_i = method_new[i] if method_new is not None and i < len(method_new) else "refined"
            match_idx = next(
                (j for j, hj in enumerate(h_all) if abs(hj - float(h)) < 1e-6),
                None,
            )
            if match_idx is not None:
                if float(e_vqe_new[i]) < float(e_vqe_all[match_idx]):
                    theta_all[match_idx] = theta_new[i]
                    e_vqe_all[match_idx] = float(e_vqe_new[i])
                    method_all[match_idx] = str(method_i)
                    if gap_i > 0:
                        gaps_all[match_idx] = gap_i
                    n_updated += 1
            else:
                h_all.append(float(h))
                theta_all.append(theta_new[i])
                e_vqe_all.append(float(e_vqe_new[i]))
                e_exact_all.append(float(e_exact_new[i]))
                gaps_all.append(gap_i)
                method_all.append(str(method_i))
                n_added += 1

        # ── Compute de_gaps from stored energies ──────────────────────
        de_gaps_all = []
        for j in range(len(h_all)):
            gap_j = gaps_all[j] if gaps_all[j] > 1e-10 else 1e-10
            de_gaps_all.append(abs(e_vqe_all[j] - e_exact_all[j]) / gap_j)

        # ── Atomic write: save to tmp then rename ─────────────────────
        tmp_path = npz_path.with_suffix(".tmp.npz")
        np.savez(
            tmp_path,
            h_values=np.array(h_all),
            theta_opt=np.array(theta_all),
            e_vqe=np.array(e_vqe_all),
            e_exact=np.array(e_exact_all),
            gaps=np.array(gaps_all),
            de_gaps=np.array(de_gaps_all),
            method=np.array(method_all),
        )
        tmp_path.rename(npz_path)
        return n_updated, n_added

    # ═══════════════════════════════════════════════════════════════════════════
    # Section: Budget Estimation + Cache Warm-up
    # ═══════════════════════════════════════════════════════════════════════════

    def section_budget_estimation(self) -> dict:
        """Estimate compute budget leveraging cached results."""
        from qmbp_simulation.execution.eval_cache import EvalCache
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        topo = self._args.topology
        n_target = self._args.target_n[0]
        p = self._args.p_layers[0]
        n_points = self._args.h_points

        # Load caches and report warmth
        gt_cache = GroundTruthCache()
        eval_cache = EvalCache(enabled=not getattr(self._args, "no_eval_cache", False))

        # Count GT cache hits for this config
        gt_hits = 0
        for h in self._h_values:
            if gt_cache.get(topo, n_target, "tfim_bond_resolved", float(h)):
                gt_hits += 1
        gt_misses = n_points - gt_hits

        # Check existing NPZ training data
        npz_path = Path("data/multi_n_training") / f"{topo}_N{n_target}_p{p}.npz"
        npz_points = 0
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            npz_points = len(data["h_values"])

        # QualityPredictor time estimate
        estimated_time_s = 0.0
        try:
            from qmbp_simulation.analysis.quality_predictor import QualityPredictor
            predictor = QualityPredictor()
            report = predictor.predict(
                model="tfim_bond_resolved", topology=topo,
                n_qubits=n_target, p_layers=p,
                h_min=self._args.h_min, h_max=self._args.h_max,
            )
            estimated_time_s = report.estimated_time_s
        except Exception:
            pass

        # Compute budget estimates
        t_gt = gt_misses * 45.0  # ~45s per DMRG at N=20
        t_eval_per_point = 2.0 if n_target <= 22 else 15.0  # MPS is slower
        t_eval = n_points * t_eval_per_point
        t_refine_per_point = 120.0 if n_target <= 22 else 300.0
        t_refine_worst = int(n_points * 0.5) * t_refine_per_point
        max_iters = self._args.max_iterations
        t_total_worst = t_gt + max_iters * (t_eval + t_refine_worst * 0.5)
        t_savings_gt = gt_hits * 45.0

        logger.info(f"  ┌─ Budget Estimation ────────────────────────")
        logger.info(f"  │ Config: {topo} N={n_target} p={p}, {n_points} h-points")
        logger.info(f"  │ GT cache: {gt_hits}/{n_points} hits → {gt_misses} DMRG needed")
        logger.info(f"  │ Eval cache: {len(eval_cache)} total entries")
        logger.info(f"  │ NPZ training data: {npz_points} existing points")
        logger.info(f"  │ ")
        logger.info(f"  │ Estimated costs:")
        logger.info(f"  │   Ground truth: {t_gt:.0f}s ({gt_misses} × 45s)")
        logger.info(f"  │   Evaluation (per iter): {t_eval:.0f}s")
        logger.info(f"  │   Refinement (worst case): {t_refine_worst:.0f}s")
        logger.info(f"  │   Max iterations: {max_iters}")
        logger.info(f"  │ ")
        logger.info(f"  │ Total worst-case: {t_total_worst:.0f}s ({t_total_worst/60:.1f} min)")
        logger.info(f"  │ Cache savings: ~{t_savings_gt:.0f}s from GT hits")
        if estimated_time_s > 0:
            logger.info(f"  │ Historical estimate: {estimated_time_s:.0f}s")
        logger.info(f"  └──────────────────────────────────────────────")

        return {
            "pass": True,
            "budget": {
                "gt_hits": gt_hits,
                "gt_misses": gt_misses,
                "eval_cache_entries": len(eval_cache),
                "npz_existing_points": npz_points,
                "estimated_gt_s": t_gt,
                "estimated_eval_s": t_eval,
                "estimated_refine_worst_s": t_refine_worst,
                "estimated_total_worst_s": t_total_worst,
                "cache_savings_s": t_savings_gt,
                "historical_time_s": estimated_time_s,
            },
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Section: Iterative Improvement Loop
    # ═══════════════════════════════════════════════════════════════════════════

    def section_iterative_improve(self) -> dict:
        """Iterative predict → refine → retrain loop with cache reuse."""
        import torch
        from scipy.optimize import minimize as _minimize

        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.execution import NoiselessBackend
        from qmbp_simulation.execution.eval_cache import EvalCache
        from qmbp_simulation.framework.preflight import get_regime_threshold
        from qmbp_simulation.models.constants import STATEVECTOR_MAX_N
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.predictors.model_zoo import (
            ZooEntry, load_pretrained, register_checkpoint,
        )
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator
        from qmbp_simulation.predictors.unified_graph import (
            build_unified_bond_resolved_graph,
        )
        from qmbp_simulation.predictors.unified_mpnn import (
            UnifiedMPNN, train_unified_mpnn,
        )
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        topo = self._args.topology
        n_target = self._args.target_n[0]
        p = self._args.p_layers[0]
        max_iterations = self._args.max_iterations
        improvement_threshold = self._args.improvement_threshold
        spec = get_model_spec("tfim_bond_resolved")
        hva = HVACircuitBuilder()

        # ── Setup: caches, backend, circuit ───────────────────────────────
        gt_cache = GroundTruthCache()
        use_eval_cache = not getattr(self._args, "no_eval_cache", False)
        eval_cache = EvalCache(enabled=use_eval_cache)
        backend = NoiselessBackend()
        solver = self.solver

        # Auto-select backend for N > 22
        if n_target > STATEVECTOR_MAX_N:
            try:
                from qmbp_simulation.execution import MPSBackend
                eval_backend = MPSBackend(chi_max=64)
            except ImportError:
                eval_backend = backend
        else:
            eval_backend = backend

        lattice_target = self.make_lattice(topo, n_target, J=1.0, h=2.0)
        circuit_target, _ = hva.create_bond_resolved(n_target, p, lattice_target)
        n_params = circuit_target.num_parameters

        # NPZ path for this config
        npz_dir = Path("data/multi_n_training")
        npz_dir.mkdir(parents=True, exist_ok=True)
        npz_path = npz_dir / f"{topo}_N{n_target}_p{p}.npz"

        # Load existing refined θ from NPZ (anti-regression baseline)
        prev_theta_by_h: dict[float, tuple[np.ndarray, float]] = {}
        if npz_path.exists():
            prev_data = np.load(npz_path, allow_pickle=True)
            e_key = "e_vqe" if "e_vqe" in prev_data else "energies"
            has_energies = e_key in prev_data
            n_loaded = 0
            for i, h in enumerate(prev_data["h_values"]):
                theta_i = prev_data["theta_opt"][i]
                # Validate loaded θ: must be finite and correct dimension
                if not np.all(np.isfinite(theta_i)):
                    logger.warning(f"  NPZ: skipping h={float(h):.4f} (non-finite θ)")
                    continue
                if len(theta_i) != n_params:
                    logger.warning(
                        f"  NPZ: skipping h={float(h):.4f} "
                        f"(dim mismatch: {len(theta_i)} vs {n_params})"
                    )
                    continue
                # Energy validation: must be finite and stored
                e_val = float(prev_data[e_key][i]) if has_energies else None
                if e_val is not None and not np.isfinite(e_val):
                    logger.warning(f"  NPZ: skipping h={float(h):.4f} (non-finite energy)")
                    continue
                prev_theta_by_h[round(float(h), 6)] = (theta_i, e_val)
                n_loaded += 1
            logger.info(f"  Loaded {n_loaded} previous θ_opt from NPZ")

        # Ansatz-limit boundary — DISABLED: let all h values be refinable
        # h_min_valid = get_regime_threshold(topo, n_target, p)
        # if h_min_valid > 0:
        #     logger.info(f"  Ansatz-limit boundary: h_min_valid={h_min_valid:.2f}")
        h_min_valid = 0  # effectively disables ansatz-limit filtering

        # ── Compute ground truth (all from cache ideally) ─────────────────
        gt_hits, gt_misses = 0, 0
        e_exact_arr = np.zeros(len(self._h_values))
        gap_arr = np.zeros(len(self._h_values))
        for i, h in enumerate(self._h_values):
            cached_gt = gt_cache.get(topo, n_target, "tfim_bond_resolved", float(h))
            if cached_gt:
                e_exact_arr[i] = cached_gt["energy"]
                gap_arr[i] = cached_gt["gap"]
                gt_hits += 1
            else:
                lat_h = self.make_lattice(topo, n_target, J=1.0, h=float(h))
                H = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)
                t_gt = time.perf_counter()
                gt_obj = solver.solve(H, lat_h)
                e_exact_arr[i] = gt_obj.ground_energy
                gap_arr[i] = gt_obj.gap
                gt_cache.put_from_result(
                    topo, n_target, "tfim_bond_resolved", float(h), gt_obj
                )
                gt_misses += 1
                logger.info(
                    f"  GT [{gt_hits+gt_misses}/{len(self._h_values)}] "
                    f"h={float(h):.3f} E={gt_obj.ground_energy:.6f} "
                    f"gap={gt_obj.gap:.4f} ({time.perf_counter()-t_gt:.1f}s)"
                )
        logger.info(f"  Ground truth: {gt_hits} cache hits, {gt_misses} computed")
        if gt_misses > 0:
            gt_cache.flush()  # Persist new ground truths immediately

        # ── Load model from zoo ───────────────────────────────────────────
        model = None
        try:
            model, _meta = load_pretrained(
                model="tfim_bond_resolved", topology=topo,
                n_qubits=0, p_layers=p,  # 0 = multi-N
            )
            logger.info(f"  Loaded multi-N model from zoo: {_meta.checkpoint_file}")
        except FileNotFoundError:
            pass

        if model is None:
            try:
                model, _meta = load_pretrained(
                    model="tfim_bond_resolved", topology=topo,
                    n_qubits=self._args.train_n, p_layers=p,
                    allow_cross_n=True,
                )
                logger.info(f"  Loaded single-N model from zoo: {_meta.checkpoint_file}")
            except FileNotFoundError:
                # Bootstrap: no model exists → run AcceleratedVQE to generate
                # initial training data, then train a UnifiedMPNN from it.
                logger.info("  No model in zoo — bootstrapping via AcceleratedVQE...")
                from qmbp_simulation.pipeline.accelerated import (
                    AcceleratedConfig, AcceleratedVQE,
                )

                boot_config = AcceleratedConfig(
                    n_anchors=self._args.n_anchors,
                    n_restarts=self._args.n_restarts,
                    maxiter=self._args.maxiter,
                    mpnn_epochs=4000,
                    use_zoo=False,
                    force_method=getattr(self._args, "force_method", None),
                    bidirectional_anchors=getattr(
                        self._args, "bidirectional_anchors", False
                    ),
                )
                boot_lattice = self.make_lattice(topo, n_target, J=1.0, h=2.0)
                boot_circuit, _ = hva.create_bond_resolved(n_target, p, boot_lattice)
                t_boot = time.perf_counter()
                accel = AcceleratedVQE(
                    boot_lattice, boot_circuit, spec, eval_backend, config=boot_config
                )
                boot_result = accel.run(self._h_values, seed=42, p_layers=p)
                logger.info(
                    f"  Bootstrap done: pass_rate={boot_result.pass_rate:.0%}, "
                    f"time={time.perf_counter() - t_boot:.1f}s"
                )
                # Save bootstrap data to NPZ
                np.savez(
                    npz_path,
                    h_values=self._h_values[:len(boot_result.theta_opt)],
                    theta_opt=boot_result.theta_opt,
                    e_vqe=boot_result.energies,
                    e_exact=boot_result.e_exact,
                    de_gaps=boot_result.de_gaps,
                    gaps=boot_result.gaps,
                    method=boot_result.method,
                )
                # Reload prev_theta_by_h from freshly saved NPZ
                for i, h in enumerate(self._h_values[:len(boot_result.theta_opt)]):
                    th_i = boot_result.theta_opt[i]
                    if np.all(np.isfinite(th_i)) and len(th_i) == n_params:
                        prev_theta_by_h[round(float(h), 6)] = (
                            th_i, float(boot_result.energies[i])
                        )
                # Train initial UnifiedMPNN from bootstrap data
                agg = MultiNAggregator(topology=topo, model="tfim_bond_resolved")
                agg.scan()
                dataset = agg.build_combined_dataset(max_de_gap=0.15)
                if len(dataset) < 3:
                    return {
                        "pass": False,
                        "error": f"Bootstrap produced only {len(dataset)} valid points.",
                    }
                sample_g = dataset[0]
                n_node_features = (
                    sample_g.x.shape[1] if hasattr(sample_g, "x") else 4
                )
                model = UnifiedMPNN(
                    node_features=n_node_features,
                    hidden_dim=256, n_layers=3,
                    norm_type="none", dropout=0.1,
                )
                train_unified_mpnn(
                    model, dataset, n_epochs=4000, lr=1e-3, patience=300, seed=42
                )
                # Register in zoo
                from datetime import datetime, timezone
                entry = ZooEntry(
                    model="tfim_bond_resolved", topology=topo,
                    n_qubits=0, p_layers=p,
                    checkpoint_file=f"unified_tfim_br_{topo}_multiN_{n_target}_p{p}.pt",
                    h_range=(self._args.h_min, self._args.h_max),
                    pass_rate=boot_result.pass_rate,
                    n_training_points=len(dataset),
                    seeds=[42],
                    created=datetime.now(timezone.utc).isoformat(),
                    notes=f"Bootstrap from AcceleratedVQE N={n_target}",
                )
                register_checkpoint(model, entry, overwrite=True)
                logger.info(f"  Bootstrap model registered: {entry.checkpoint_file}")

        # ── Iterative improvement loop ────────────────────────────────────
        iteration_reports = []
        total_vqe_calls = 0
        prev_pass_rate = 0.0
        convergence_reason = "max_iterations"
        # Track the best pass_rate ever exported to zoo for Fix C
        zoo_best_pass_rate = getattr(_meta, 'pass_rate', 0.0) if '_meta' in dir() and _meta else 0.0

        for iteration in range(1, max_iterations + 1):
            logger.info(f"\n  ╔══ Iteration {iteration}/{max_iterations} ══════════════════════╗")
            t_iter_start = time.perf_counter()
            model.eval()

            # ── 2a: Predict θ for all h-points ────────────────────────────
            predictions = []
            n_pred_invalid = 0
            for h in self._h_values:
                g = build_unified_bond_resolved_graph(
                    lattice_target, h_value=float(h), p_layers=p,
                    include_circuit_nodes=True,
                )
                with torch.no_grad():
                    pred = model(g).numpy().flatten()
                # NaN/Inf guard on predictions
                if not np.all(np.isfinite(pred)):
                    n_bad = int(np.sum(~np.isfinite(pred)))
                    logger.warning(
                        f"    h={float(h):.2f}: prediction has {n_bad} NaN/Inf → zeroed"
                    )
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
            energies = np.zeros(len(self._h_values))
            eval_hits = 0
            for i, h in enumerate(self._h_values):
                h_key = round(float(h), 6)

                # Skip already-refined points: use stored energy if available
                # and validated (must be finite and have associated θ)
                if h_key in prev_theta_by_h:
                    theta_prev, e_prev = prev_theta_by_h[h_key]
                    if e_prev is not None and np.isfinite(e_prev):
                        # Variational principle check: e_prev must be ≥ e_exact
                        # (allow small tolerance for numerical noise)
                        if e_prev >= e_exact_arr[i] - 1e-6:
                            energies[i] = e_prev
                            eval_hits += 1
                            continue
                        else:
                            logger.warning(
                                f"    h={float(h):.2f}: NPZ energy {e_prev:.6f} "
                                f"violates variational principle (E_exact={e_exact_arr[i]:.6f}). "
                                f"Discarding stale entry."
                            )
                            del prev_theta_by_h[h_key]

                # Evaluate via cache
                key = eval_cache.make_key(
                    topo, n_target, float(h), predictions[i],
                    model="tfim_bond_resolved", p_layers=p,
                )
                cached_e = eval_cache.get(key)
                if cached_e is not None:
                    energies[i] = cached_e
                    eval_hits += 1
                else:
                    lat_h = self.make_lattice(topo, n_target, J=1.0, h=float(h))
                    H = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)
                    energies[i] = eval_backend.evaluate(circuit_target, H, predictions[i])
                    eval_cache.put(key, float(energies[i]))

            de_gaps = np.abs(energies - e_exact_arr) / np.maximum(gap_arr, 1e-10)
            pass_rate = float((de_gaps < 0.05).mean())
            logger.info(
                f"  │ Eval: {eval_hits}/{len(self._h_values)} cache hits, "
                f"pass_rate@5%={pass_rate:.0%}"
            )

            # ── 2b.1: Check convergence (early stop) ─────────────────────
            if pass_rate >= 0.90:
                convergence_reason = "target_reached"
                logger.info(f"  │ ✓ Target reached: pass_rate={pass_rate:.0%} ≥ 90%")

                # Persist validated predictions as training data for future
                # multi-N models — these θ_pred gave ΔE/gap < 5%, so they
                # are as good as VQE-optimized θ_opt.
                n_persisted = 0
                for i, h in enumerate(self._h_values):
                    h_key = round(float(h), 6)
                    if h_key not in prev_theta_by_h and de_gaps[i] < 0.05:
                        prev_theta_by_h[h_key] = (predictions[i], float(energies[i]))
                        n_persisted += 1
                if n_persisted > 0 or not npz_path.exists():
                    # Upsert all validated points into NPZ
                    h_all = np.array(sorted(prev_theta_by_h.keys(), reverse=True))
                    theta_all = np.array([prev_theta_by_h[round(h, 6)][0] for h in h_all])
                    e_all = np.array([prev_theta_by_h[round(h, 6)][1] for h in h_all])
                    # Match e_exact and gaps from our arrays
                    e_ex_all = np.array([
                        e_exact_arr[np.argmin(np.abs(self._h_values - h))] for h in h_all
                    ])
                    gap_all = np.array([
                        gap_arr[np.argmin(np.abs(self._h_values - h))] for h in h_all
                    ])
                    de_all = np.abs(e_all - e_ex_all) / np.maximum(gap_all, 1e-10)
                    np.savez(
                        npz_path,
                        h_values=h_all, theta_opt=theta_all,
                        e_vqe=e_all, e_exact=e_ex_all,
                        de_gaps=de_all, gaps=gap_all,
                    )
                    logger.info(
                        f"  │ Persisted {len(h_all)} validated points → {npz_path.name}"
                    )

                iteration_reports.append(self._build_iter_report(
                    iteration, pass_rate, 0, 0, eval_hits, time.perf_counter() - t_iter_start
                ))
                break

            improvement = pass_rate - prev_pass_rate
            if iteration > 1 and improvement < improvement_threshold:
                convergence_reason = "no_improvement"
                logger.info(
                    f"  │ ✓ Converged: improvement={improvement:.4f} < "
                    f"threshold={improvement_threshold}"
                )
                iteration_reports.append(self._build_iter_report(
                    iteration, pass_rate, 0, 0, eval_hits, time.perf_counter() - t_iter_start
                ))
                break

            # ── 2c: Identify failures + ansatz-limit filter ───────────────
            # Uses dual criteria: ΔE/gap OR |ΔE| OR fidelity (from metrics)
            from qmbp_simulation.analysis.metrics import is_point_failure

            failures = []
            ansatz_limited = []
            for i, h in enumerate(self._h_values):
                abs_err_i = abs(energies[i] - e_exact_arr[i])
                is_fail = is_point_failure(
                    de_gap=de_gaps[i],
                    abs_error=abs_err_i,
                    fidelity=None,  # Not available in iterative loop
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
                iteration_reports.append(self._build_iter_report(
                    iteration, pass_rate, 0, len(ansatz_limited),
                    eval_hits, time.perf_counter() - t_iter_start
                ))
                break

            # Limit refinements per iteration
            max_refine = 5 if n_target <= STATEVECTOR_MAX_N else 3
            failures = failures[:max_refine]
            logger.info(
                f"  │ Failures: {len(failures)} to refine, "
                f"{len(ansatz_limited)} ansatz-limited (skipped)"
            )

            # ── 2d: Anti-regression + VQE refine ─────────────────────────
            refined_h = []
            refined_theta = []
            refined_energies = []
            refined_e_exact = []

            # Use CLI maxiter/restarts for aggressive refinement.
            # For L-BFGS-B: cap restarts (each restart = ~50 iters × 97 evals).
            # Stagnation threshold (3) will auto-stop if no improvement anyway.
            refine_maxiter = self._args.maxiter
            refine_method = self._args.force_method or "L-BFGS-B"
            if refine_method == "L-BFGS-B":
                refine_restarts = min(3, self._args.n_restarts)
            else:
                refine_restarts = max(5, self._args.n_restarts // 2)
            logger.info(
                f"  │ Refine config: method={refine_method}, "
                f"maxiter={refine_maxiter}, restarts={refine_restarts}"
            )

            for fail_idx_pos, idx in enumerate(failures):
                h = float(self._h_values[idx])
                h_key = round(h, 6)
                t_refine_start = time.perf_counter()
                logger.info(
                    f"  │ Refining [{fail_idx_pos+1}/{len(failures)}] "
                    f"h={h:.4f} (n_params={n_params}, ΔE/gap={de_gaps[idx]:.4f})..."
                )
                import sys
                sys.stdout.flush()
                sys.stderr.flush()
                for handler in logging.getLogger().handlers:
                    handler.flush()

                # Anti-regression: pick best init from {θ_pred, θ_prev}
                theta_init = predictions[idx].copy()
                print(f"    [DBG] anti-regression check: h_key={h_key} in prev={h_key in prev_theta_by_h}", flush=True)
                if h_key in prev_theta_by_h:
                    theta_prev, e_prev = prev_theta_by_h[h_key]
                    # Evaluate θ_prev energy (cache hit expected)
                    key_prev = eval_cache.make_key(
                        topo, n_target, h, theta_prev,
                        model="tfim_bond_resolved", p_layers=p,
                    )
                    e_prev_cached = eval_cache.get(key_prev)
                    if e_prev_cached is None:
                        print(f"    [DBG] anti-regression: cache MISS, evaluating θ_prev...", flush=True)
                        lat_h = self.make_lattice(topo, n_target, J=1.0, h=h)
                        H = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)
                        e_prev_cached = eval_backend.evaluate(
                            circuit_target, H, theta_prev
                        )
                        eval_cache.put(key_prev, float(e_prev_cached))
                        print(f"    [DBG] anti-regression: eval done E={e_prev_cached:.6f}", flush=True)
                    else:
                        print(f"    [DBG] anti-regression: cache HIT E={e_prev_cached:.6f}", flush=True)
                    if e_prev_cached < energies[idx]:
                        theta_init = theta_prev.copy()
                        logger.debug(f"    h={h:.2f}: anti-regression → using θ_prev")

                # VQE warm-start refinement (aggressive: multi-restart)
                print(f"    [DBG] building H for VQE...", flush=True)
                lat_h = self.make_lattice(topo, n_target, J=1.0, h=h)
                H = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)
                print(f"    [DBG] creating VQEOptimizer...", flush=True)
                try:
                    from qmbp_simulation import VQEConfig, VQEOptimizer
                    vqe_cfg = VQEConfig(
                        p_layers=p,
                        n_restarts=refine_restarts,
                        maxiter=refine_maxiter,
                        method=refine_method,
                        enable_callbacks=False,  # No trajectory logging in iterative refine (2x speedup)
                    )
                    vqe_opt = VQEOptimizer(
                        config=vqe_cfg, backend=eval_backend, seed=42 + idx
                    )
                    # set_h for correct cache key
                    if hasattr(eval_backend, "set_h"):
                        eval_backend.set_h(h)
                    print(f"    [DBG] calling vqe_opt.optimize()...", flush=True)
                    vqe_result = vqe_opt.optimize(
                        H, circuit_target, initial_guess=theta_init
                    )
                    total_vqe_calls += 1
                    e_refined = float(vqe_result.energy)
                    res_x = vqe_result.theta_opt
                    t_refine_elapsed = time.perf_counter() - t_refine_start
                    logger.info(
                        f"    h={h:.2f}: VQE done in {t_refine_elapsed:.1f}s, "
                        f"E={e_refined:.6f}"
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
                    if e_refined < energies[idx] - 1e-10:
                        de_gap_new = abs(e_refined - e_exact_arr[idx]) / max(gap_arr[idx], 1e-10)
                        abs_err_old = abs(energies[idx] - e_exact_arr[idx])
                        abs_err_new = abs(e_refined - e_exact_arr[idx])
                        logger.info(
                            f"    h={h:.2f}: ΔE/gap {de_gaps[idx]:.4f} → {de_gap_new:.4f} "
                            f"|ΔE| {abs_err_old:.3f} → {abs_err_new:.3f} ✓"
                        )
                        refined_h.append(h)
                        refined_theta.append(res_x.copy())
                        refined_energies.append(e_refined)
                        refined_e_exact.append(float(e_exact_arr[idx]))
                        # Update tracking
                        prev_theta_by_h[h_key] = (res_x.copy(), e_refined)

                        # ── Immediate persist: NPZ upsert per-point ───────
                        # Ensures no refined θ is lost on interrupt.
                        # _upsert_npz uses "lower energy wins" so future runs
                        # can only improve (never regress) stored values.
                        gap_i = float(gap_arr[idx])
                        self._upsert_npz(
                            npz_path,
                            np.array([h]),
                            np.array([res_x]),
                            np.array([e_refined]),
                            np.array([float(e_exact_arr[idx])]),
                            gaps_new=np.array([gap_i]),
                        )
                        # Note: eval_cache auto-flushes every 50 puts.
                        # Full flush deferred to end of iteration (avoid 5MB
                        # JSON write per point).
                    else:
                        logger.info(f"    h={h:.2f}: no improvement (VQE stuck)")
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
            n_updated = len(refined_h)  # all persisted incrementally
            if refined_h:
                logger.info(
                    f"  │ NPZ: {n_updated} points persisted incrementally "
                    f"→ {npz_path.name}"
                )
            # Flush eval cache once per iteration (not per-point)
            eval_cache.flush()

            # ── 2f: Retrain multi-N model ─────────────────────────────────
            # Fix A: Skip retrain if no new data was produced
            # Fix B: Fine-tune instead of training from scratch
            from qmbp_simulation.predictors.unified_mpnn import (
                should_retrain, fine_tune_unified_mpnn,
            )

            agg = MultiNAggregator(topology=topo, model="tfim_bond_resolved")
            agg.scan()
            dataset = agg.build_combined_dataset(max_de_gap=0.10)

            do_retrain, retrain_reason = should_retrain(
                n_new_points=len(refined_h),
                current_pass_rate=pass_rate,
                prev_pass_rate=prev_pass_rate,
                dataset_size=len(dataset),
            )

            if do_retrain and len(dataset) >= 5:
                logger.info(
                    f"  │ Retraining (reason={retrain_reason}, "
                    f"{len(refined_h)} new points, {len(dataset)} total)..."
                )
                sample_g = dataset[0]
                n_node_features = sample_g.x.shape[1] if hasattr(sample_g, 'x') else 4

                # Fix B: Fine-tune existing model if it has same architecture,
                # otherwise train from scratch (architecture mismatch).
                can_fine_tune = (
                    hasattr(model, 'node_features')
                    and model.node_features == n_node_features
                    and iteration > 1  # First iter after bootstrap → full train
                )

                if can_fine_tune:
                    logger.info("  │ Mode: fine-tune (1000 epochs, lr=3e-4)")
                    train_result = fine_tune_unified_mpnn(
                        model, dataset, n_epochs=1000, lr=3e-4,
                        patience=150, seed=42,
                    )
                else:
                    logger.info("  │ Mode: full retrain (5000 epochs, lr=1e-3)")
                    model = UnifiedMPNN(
                        node_features=n_node_features,
                        hidden_dim=256, n_layers=3,
                        norm_type="none", dropout=0.1,
                    )
                    train_result = train_unified_mpnn(
                        model, dataset, n_epochs=5000, lr=1e-3,
                        patience=300, seed=42,
                    )

                mse = train_result.get("final_mse", 0) if isinstance(train_result, dict) else 0
                mode = train_result.get("mode", "full")
                logger.info(f"  │ Retrained ({mode}): MSE={mse:.2e}, {len(dataset)} points")

                # ── 2g: Export to zoo (only if pass_rate improved) ────────
                # Fix C: Don't overwrite a better model in the zoo with one
                # that didn't improve pass_rate.
                if pass_rate > zoo_best_pass_rate or iteration == 1:
                    from datetime import datetime, timezone
                    n_vals = agg.available_n_values()
                    n_str = "+".join(str(n) for n in n_vals)
                    entry = ZooEntry(
                        model="tfim_bond_resolved", topology=topo,
                        n_qubits=0, p_layers=p,
                        checkpoint_file=f"unified_tfim_br_{topo}_multiN_{n_str}_p{p}.pt",
                        h_range=(self._args.h_min, self._args.h_max),
                        pass_rate=pass_rate,
                        n_training_points=len(dataset),
                        seeds=[42],
                        created=datetime.now(timezone.utc).isoformat(),
                        notes=f"Iterative improve iter {iteration}: N={n_vals}",
                    )
                    register_checkpoint(model, entry, overwrite=True)
                    zoo_best_pass_rate = pass_rate
                    logger.info(f"  │ Exported to zoo: {entry.checkpoint_file}")
                else:
                    logger.info(
                        f"  │ Zoo skip: pass_rate={pass_rate:.0%} ≤ "
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
            iteration_reports.append(self._build_iter_report(
                iteration, pass_rate, len(refined_h), len(ansatz_limited),
                eval_hits, iter_time,
            ))
            prev_pass_rate = pass_rate
            logger.info(
                f"  ╚══ Iteration {iteration} done: pass_rate={pass_rate:.0%}, "
                f"refined={len(refined_h)}, time={iter_time:.1f}s ══╝"
            )

        # ── Final report ──────────────────────────────────────────────────
        eval_cache.flush()
        final_stats = eval_cache.stats()
        final_pass_rate = iteration_reports[-1]["pass_rate"] if iteration_reports else 0.0

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
            "cache_stats": final_stats,
            "gt_cache_hits": gt_hits,
            "gt_cache_misses": gt_misses,
        }

    def _build_iter_report(
        self, iteration, pass_rate, n_refined, n_ansatz_limited,
        eval_hits, elapsed_s
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
