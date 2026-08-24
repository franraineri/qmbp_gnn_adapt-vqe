#!/usr/bin/env python3
"""Large-N Extrapolation Test — MPNN vs Random VQE at N=30-100.

Tests how well a pre-trained UnifiedMPNN (from smaller N data) predicts
optimal HVA parameters for system sizes far beyond its training distribution.
Compares two arms:

1. MPNN warm-start: Single forward pass → θ_pred → evaluate E(θ)
2. VQE random-init: Random θ₀ → L-BFGS-B optimization → E_vqe (optional)
3. Ground truth: DMRG → E_exact, gap

All data persisted to:
- Ground truth: data/ground_truth_cache.json (cross-session reuse)
- θ predictions: data/large_n_extrapolation/{topo}_N{N}_p{p}.npz
- Model tracking: data/model_zoo/manifest.json

Reports ΔE/gap, |ΔE|, |ΔE|, speedup (n_evals ratio), and wall time.
All metrics follow dual criterion (ΔE/gap < 5% AND |ΔE| < 0.10).

Usage:
    # Default: chain_1d, N=[30, 40, 60, 100], 6 h-points
    .venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py

    # Custom topology and sizes
    .venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py \\
        --topology chain_1d --target-n 30 50 80 --h-min 2.5 --h-max 5.0

    # Quick smoke test (fewer points)
    .venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py \\
        --target-n 30 --h-points 3

    # Skip VQE baseline (faster, MPNN-only)
    .venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py \\
        --skip-random-baseline

    # With VQE refinement for failing points
    .venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py \\
        --refine-failing --vqe-maxiter 100

    # Dry run
    .venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --dry-run
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from qmbp_simulation.framework.result_io import persist_predictions_to_training_npz
from qmbp_simulation.framework.runner_base import (
    Section,
    ValidationRunner,
    resolve_project_root,
)

if TYPE_CHECKING:
    pass

_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants (configurable defaults)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_TARGET_N = [20, 30, 40, 60, 80]
DEFAULT_TOPOLOGY = "chain_1d"
DEFAULT_MODEL = "tfim_bond_resolved"
DEFAULT_P = 1
DEFAULT_H_MIN = 2.5
DEFAULT_H_MAX = 5.0
DEFAULT_H_POINTS = 6
DEFAULT_VQE_MAXITER = 50
DEFAULT_VQE_RESTARTS = 2

# NPZ storage for large-N extrapolation (separate from training data)
EXTRAPOLATION_DATA_DIR = Path("data/large_n_extrapolation")


# ═══════════════════════════════════════════════════════════════════════════════
# Reusable Helpers (importable by future topology scripts)
# ═══════════════════════════════════════════════════════════════════════════════


def load_extrapolation_npz(topology: str, n_qubits: int, p_layers: int) -> dict[float, dict] | None:
    """Load existing extrapolation data from NPZ cache.

    Returns dict mapping h → {theta, e_pred, e_exact, gap, de_gap, method}.
    Returns None if no cache exists.
    """
    npz_path = EXTRAPOLATION_DATA_DIR / f"{topology}_N{n_qubits}_p{p_layers}.npz"
    if not npz_path.exists():
        return None

    try:
        data = np.load(npz_path, allow_pickle=True)
        result = {}
        e_key = "e_pred" if "e_pred" in data else ("e_vqe" if "e_vqe" in data else None)
        theta_arr = data["theta_opt"]

        for i, h in enumerate(data["h_values"]):
            h_key = round(float(h), 6)

            # Handle object array (variable-length theta) vs 2D numeric array
            theta_raw = theta_arr[i]
            if theta_raw is None:
                continue
            try:
                # Try direct conversion (works for nested arrays and scalars)
                if isinstance(theta_raw, np.ndarray):
                    theta_i = theta_raw.astype(np.float64)
                else:
                    theta_i = np.array(theta_raw, dtype=np.float64)
                # Ensure 1D
                theta_i = theta_i.flatten()
                if theta_i.size == 0 or not np.all(np.isfinite(theta_i)):
                    continue
            except (ValueError, TypeError):
                continue

            result[h_key] = {
                "theta": theta_i,
                "e_pred": float(data[e_key][i]) if e_key else None,
                "e_exact": float(data["e_exact"][i]),
                "gap": float(data["gaps"][i]) if "gaps" in data else None,
                "de_gap": float(data["de_gaps"][i]) if "de_gaps" in data else None,
                "method": str(data["method"][i]) if "method" in data else "unknown",
            }
        return result
    except Exception as e:
        logger.warning(f"Failed to load NPZ cache: {e}")
        return None


def compute_extrapolation_summary(per_h_results: list[dict]) -> dict:
    """Compute summary statistics for extrapolation results.

    Thin wrapper over compute_deploy_summary() that ensures n_qubits is
    propagated for per-site error computation.
    """
    from qmbp_simulation.analysis.metrics import compute_deploy_summary

    if not per_h_results:
        return {
            "n_points": 0,
            "pass_rate_5pct": 0.0,
            "pass_rate_dual": 0.0,
            "mean_de_gap": 0.0,
            "std_de_gap": 0.0,
            "p90_de_gap": 0.0,
            "max_de_gap": 0.0,
            "quality_score": 0.0,
            "grade": "F",
            "mean_abs_error": None,
        }

    return compute_deploy_summary(per_h_results)


# ═══════════════════════════════════════════════════════════════════════════════
# Runner Class
# ═══════════════════════════════════════════════════════════════════════════════


class LargeNExtrapolationRunner(ValidationRunner):
    """Test MPNN prediction at N >> training data (zero-shot extrapolation).

    Loads the best available multi-N model from the zoo, predicts θ for
    large N, evaluates via MPS, and compares against DMRG ground truth
    and optional random-init VQE baseline.

    All results are persisted:
    - GT cache: data/ground_truth_cache.json
    - θ + metrics: data/large_n_extrapolation/{topo}_N{N}_p{p}.npz
    - JSON results: results/experiments/exp_large_n_extrap/
    """

    runner_id = "large_n_extrapolation_v2"
    experiment_id = "LARGE_N_EXTRAP"
    description = "MPNN extrapolation test at N=30-100 vs random VQE baseline"
    hypothesis = (
        "UnifiedMPNN trained on N≤20 can predict θ at N=30-100 with "
        "per-site error |ΔE| ≈ constant, demonstrating extensive scaling."
    )

    @classmethod
    def _add_custom_args(cls, parser):
        # Target system sizes
        parser.add_argument(
            "--target-n",
            type=int,
            nargs="+",
            default=DEFAULT_TARGET_N,
            help="Target system size(s) to test (default: %(default)s)",
        )
        # Physics config (reuse naming conventions from accelerated_cross_n)
        parser.add_argument(
            "--topology",
            type=str,
            default=DEFAULT_TOPOLOGY,
            help="Lattice topology (default: %(default)s)",
        )
        parser.add_argument(
            "--model-name",
            type=str,
            default=DEFAULT_MODEL,
            help="Physics model name (default: %(default)s)",
        )
        parser.add_argument(
            "--p-layers",
            type=int,
            default=DEFAULT_P,
            help="HVA layer depth (default: %(default)s)",
        )
        # H-grid
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
        # VQE baseline config
        parser.add_argument(
            "--skip-random-baseline",
            action="store_true",
            default=False,
            help="Skip random-init VQE baseline (faster, MPNN-only)",
        )
        parser.add_argument(
            "--vqe-maxiter",
            type=int,
            default=DEFAULT_VQE_MAXITER,
            help="VQE optimizer max iterations (default: %(default)s)",
        )
        parser.add_argument(
            "--vqe-restarts",
            type=int,
            default=DEFAULT_VQE_RESTARTS,
            help="VQE random restarts (default: %(default)s)",
        )
        # Refinement option
        parser.add_argument(
            "--refine-failing",
            action="store_true",
            default=False,
            help="Run VQE refinement on MPNN predictions that fail dual criterion",
        )
        parser.add_argument(
            "--max-refine",
            type=int,
            default=3,
            help="Max points to refine per N (default: 3)",
        )
        # Model selection
        parser.add_argument(
            "--checkpoint",
            type=str,
            default=None,
            help="Explicit model checkpoint path (overrides zoo search)",
        )
        # Cache control
        parser.add_argument(
            "--no-eval-cache",
            action="store_true",
            default=False,
            help="Disable evaluation caching",
        )
        parser.add_argument(
            "--force-recompute",
            action="store_true",
            default=False,
            help="Ignore existing NPZ cache and recompute all predictions",
        )
        # Active learning: targeted VQE for high-uncertainty points
        parser.add_argument(
            "--active-learning-rounds",
            type=int,
            default=0,
            help="Number of AL rounds after evaluation (0=disabled, each round refines top-3 uncertain points)",
        )

    def build_config(self) -> dict:
        """Build config dict reusing base class helper."""
        config = self._build_physics_config()
        config.update(
            {
                "model": self._args.model_name,
                "skip_random_baseline": self._args.skip_random_baseline,
                "refine_failing": self._args.refine_failing,
                "vqe_maxiter": self._args.vqe_maxiter,
                "vqe_restarts": self._args.vqe_restarts,
            }
        )
        return config

    def setup(self):
        """Initialize physics objects and h-grid."""
        self.setup_physics()
        # Round to 2 decimals for cache key stability (matches GroundTruthCache)
        self._h_values = [
            round(h, 2)
            for h in np.linspace(self._args.h_min, self._args.h_max, self._args.h_points)
        ]
        # Ensure extrapolation data directory exists
        EXTRAPOLATION_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Summary line only (minimal console output)
        logger.info(
            f"  Config: {self._args.topology} N={self._args.target_n} p={self._args.p_layers} "
            f"h=[{self._args.h_min}, {self._args.h_max}] ({self._args.h_points} pts)"
        )

    def run_preflight(self) -> bool:
        """Verify targets are within DMRG limits and check extrapolation viability."""
        from qmbp_simulation.models.constants import DMRG_QUBIT_LIMIT

        for n in self._args.target_n:
            if n > DMRG_QUBIT_LIMIT:
                logger.error(f"N={n} exceeds DMRG limit ({DMRG_QUBIT_LIMIT})")
                return False
            if self._args.topology == "ladder" and n % 2 != 0:
                logger.error(f"Ladder requires even N, got {n}")
                return False

        # Optional: check extrapolation viability from cross-N data
        try:
            from qmbp_simulation.analysis.metrics import (
                compute_extrapolation_viability,
                generate_model_quality_dashboard,
            )

            dashboard = generate_model_quality_dashboard()
            topo_summary = dashboard.get("topology_summary", {})
            topo_info = topo_summary.get(self._args.topology, {})
            n_max_viable = topo_info.get("n_max_viable")

            if n_max_viable is not None:
                for target_n in self._args.target_n:
                    viable, reason, _ = compute_extrapolation_viability(
                        self._args.topology, n_max_viable, None, target_n=target_n
                    )
                    if not viable:
                        logger.warning(
                            f"  ⚠️ N={target_n} may not be viable for {self._args.topology}: {reason}"
                        )
        except Exception as e:
            logger.debug(f"Extrapolation viability check skipped: {e}")

        return True

    def define_sections(self) -> list[Section]:
        sections = [
            Section(
                id=1,
                name="Ground Truth (DMRG)",
                fn=self.section_ground_truth,
                hypothesis="DMRG converges for all (N, h) within χ budget",
            ),
            Section(
                id=2,
                name="MPNN Prediction",
                fn=self.section_mpnn_prediction,
                hypothesis="MPNN achieves |ΔE| ≈ constant (extensive scaling)",
            ),
        ]
        if not self._args.skip_random_baseline:
            sections.append(
                Section(
                    id=3,
                    name="Random VQE Baseline",
                    fn=self.section_random_baseline,
                    hypothesis="Random-init VQE with limited budget is worse than MPNN",
                )
            )
        sections.append(
            Section(
                id=4 if not self._args.skip_random_baseline else 3,
                name="Summary",
                fn=self.section_summary,
                hypothesis="MPNN demonstrates extensive scaling with N",
            )
        )
        return sections

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 1: Ground Truth (DMRG)
    # ═══════════════════════════════════════════════════════════════════════════

    def section_ground_truth(self) -> dict:
        """Compute DMRG ground truth E₀ and gap for each (N, h).

        Uses base class exact_ground_state() with 2-level caching:
        - Level 1: in-memory dict (per-run)
        - Level 2: disk-persistent GroundTruthCache (cross-session)
        """
        gt_data: dict[int, list[dict]] = {}
        t_section = time.perf_counter()

        # Process N values sequentially (efficient memory use for large N)
        for n_target in self._args.target_n:
            logger.info(f"  GT N={n_target}: computing {len(self._h_values)} points...")
            gt_data[n_target] = []
            n_cached, n_computed = 0, 0

            for h in self._h_values:
                t0 = time.perf_counter()
                # Uses 2-level cache automatically
                e_exact, gap = self.exact_ground_state(
                    self._args.topology,
                    n_target,
                    float(h),
                    model=self._args.model_name,
                )
                elapsed = time.perf_counter() - t0

                gt_data[n_target].append(
                    {
                        "h": float(h),
                        "e_exact": float(e_exact),
                        "gap": float(gap),
                        "time_s": elapsed,
                    }
                )

                # Track cache vs compute (< 0.1s typically means cache hit)
                if elapsed < 0.1:
                    n_cached += 1
                else:
                    n_computed += 1

            logger.info(f"    N={n_target}: {n_cached} cached, {n_computed} computed")

        self._gt_data = gt_data
        total_time = time.perf_counter() - t_section

        # Summary table
        logger.info(f"  GT complete: {total_time:.1f}s total")

        return {
            "n_values": self._args.target_n,
            "n_h_points": self._args.h_points,
            "total_gt_time_s": total_time,
            "gt_data": {str(k): v for k, v in gt_data.items()},  # JSON-safe keys
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 2: MPNN Prediction
    # ═══════════════════════════════════════════════════════════════════════════

    def section_mpnn_prediction(self) -> dict:
        """Load zoo model and predict θ for each (N, h). Evaluate energy.

        Persists all predictions to NPZ for anti-regression and reuse.
        """
        import torch

        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.predictors.unified_graph import (
            build_unified_bond_resolved_graph,
        )

        topo = self._args.topology
        p = self._args.p_layers
        spec = get_model_spec(self._args.model_name)
        hva = HVACircuitBuilder()
        use_cache = not self._args.no_eval_cache
        force_recompute = self._args.force_recompute

        # Load best model from zoo
        model = self.load_best_mpnn_for_cross_n(
            n_target=max(self._args.target_n),
            model=self._args.model_name,
            topology=topo,
            p_layers=p,
            checkpoint_path=self._args.checkpoint,
            train_if_missing=False,
        )
        if model is None:
            logger.error("No trained model found. Run training first.")
            return {"pass": False, "error": "no_model_available"}

        model.eval()
        self._model = model

        # Capture which checkpoint was actually used (for traceability)
        actual_checkpoint = self._args.checkpoint or "auto (zoo)"
        if not self._args.checkpoint:
            zoo_entry = getattr(self, "_zoo_entry", None)
            if zoo_entry:
                actual_checkpoint = zoo_entry.checkpoint_file
        self._actual_checkpoint = actual_checkpoint
        logger.info(f"  Model: {actual_checkpoint}")

        mpnn_results: dict[int, list[dict]] = {}
        t_section = time.perf_counter()

        # Process N values sequentially (memory efficient)
        for n_target in self._args.target_n:
            logger.info(f"  MPNN N={n_target}: predicting {len(self._h_values)} points...")

            # Load existing NPZ cache (unless force_recompute)
            existing_data = None
            if not force_recompute:
                existing_data = load_extrapolation_npz(topo, n_target, p)
                if existing_data:
                    logger.info(f"    Loaded {len(existing_data)} cached predictions")

            # Build circuit once per N
            lat_ref = self.make_lattice(topo, n_target, J=1.0, h=2.0)
            circuit, _ = hva.create_bond_resolved(n_target, p, lat_ref)
            n_params = circuit.num_parameters

            # ── Integrity check: model output vs circuit params ───────────
            # UnifiedMPNN uses per-node prediction (output_dim = p_layers * n_terms),
            # so the raw output shape depends on graph size. We do a quick probe
            # to detect obvious mismatches (wrong model loaded for different topology/p).
            if n_target == self._args.target_n[0]:  # Only on first N (lightweight)
                try:
                    g_probe = build_unified_bond_resolved_graph(
                        lat_ref, h_value=2.0, p_layers=p, include_circuit_nodes=True,
                    )
                    with torch.no_grad():
                        probe_out = model(g_probe).numpy().flatten()
                    if len(probe_out) < n_params * 0.5:
                        logger.warning(
                            f"  ⚠️ Model output dim ({len(probe_out)}) << circuit params "
                            f"({n_params}). Possible model/topology mismatch."
                        )
                    elif len(probe_out) > n_params * 2.0:
                        logger.warning(
                            f"  ⚠️ Model output dim ({len(probe_out)}) >> circuit params "
                            f"({n_params}). Predictions will be truncated."
                        )
                except Exception as e:
                    logger.debug(f"  Model probe skipped: {e}")

            # CachedBackend for transparent eval caching
            with self.get_cached_backend(
                topology=topo,
                n_qubits=n_target,
                model=self._args.model_name,
                p_layers=p,
                enabled=use_cache,
            ) as eval_backend:
                per_h_results = []
                n_cached, n_predicted = 0, 0

                try:
                  for h in self._h_values:
                    h_key = round(float(h), 6)

                    # Check NPZ cache first
                    if existing_data and h_key in existing_data and not force_recompute:
                        cached = existing_data[h_key]
                        # Only use cache if we have energy (theta might not be stored)
                        if cached.get("e_pred") is not None:
                            gt_entry = next(
                                pt
                                for pt in self._gt_data[n_target]
                                if abs(pt["h"] - float(h)) < 1e-8
                            )
                            result = self.build_per_h_result(
                                h,
                                cached["e_pred"],
                                gt_entry["e_exact"],
                                gt_entry["gap"],
                                n_params=n_params,
                                n_qubits=n_target,
                                method="cached",
                                theta=cached.get("theta", np.zeros(0)).tolist()
                                if cached.get("theta") is not None
                                else None,
                            )
                            per_h_results.append(result)
                            n_cached += 1
                            continue

                    # Fresh prediction
                    t0 = time.perf_counter()
                    g = build_unified_bond_resolved_graph(
                        lat_ref,
                        h_value=float(h),
                        p_layers=p,
                        include_circuit_nodes=True,
                    )
                    with torch.no_grad():
                        theta_pred = model(g).numpy().flatten()
                    theta_pred = np.clip(theta_pred, -np.pi, np.pi)

                    # ── MC-Dropout confidence estimation ─────────────────
                    # Delegates to model.predict_with_uncertainty() which
                    # handles train/eval mode toggling and seed management.
                    theta_std = 0.0
                    if hasattr(model, 'predict_with_uncertainty'):
                        _, theta_std = model.predict_with_uncertainty(g)
                    elif hasattr(model, 'dropout') and getattr(model, 'dropout', 0) > 0:
                        # Fallback for older model versions without the method.
                        # Uses was_training pattern to guarantee mode restoration.
                        mc_preds = []
                        was_training = model.training
                        model.train()
                        try:
                            with torch.no_grad():
                                for _mc_seed in (42, 137, 256, 511, 769):
                                    torch.manual_seed(_mc_seed)
                                    mc_pred = model(g).numpy().flatten()
                                    mc_preds.append(mc_pred)
                        finally:
                            if not was_training:
                                model.eval()
                        mc_preds_arr = np.array(mc_preds)
                        theta_std = float(np.mean(np.std(mc_preds_arr, axis=0)))

                    # Pad/trim if dimension mismatch
                    if len(theta_pred) != n_params:
                        if len(theta_pred) < n_params:
                            theta_pred = np.pad(theta_pred, (0, n_params - len(theta_pred)))
                        else:
                            theta_pred = theta_pred[:n_params]

                    # Evaluate energy
                    lat_h = self.make_lattice(topo, n_target, J=1.0, h=float(h))
                    H = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)
                    eval_backend.set_h(float(h))
                    e_pred = eval_backend.evaluate(circuit, H, theta_pred)
                    elapsed = time.perf_counter() - t0

                    # Get GT
                    gt_entry = next(
                        pt for pt in self._gt_data[n_target] if abs(pt["h"] - float(h)) < 1e-8
                    )
                    result = self.build_per_h_result(
                        h,
                        e_pred,
                        gt_entry["e_exact"],
                        gt_entry["gap"],
                        n_params=n_params,
                        n_qubits=n_target,
                        method="mpnn",
                        time_s=elapsed,
                        theta=theta_pred.tolist(),
                    )
                    result["theta_std"] = theta_std  # MC-Dropout uncertainty
                    per_h_results.append(result)
                    n_predicted += 1

                except KeyboardInterrupt:
                    logger.warning(
                        f"  ⚠️ Interrupted during MPNN prediction N={n_target}. "
                        f"Saving {len(per_h_results)} partial results."
                    )

                # Persist to NPZ (atomic, anti-regression) — runs even on interrupt
                self._persist_extrapolation_npz(topo, n_target, p, per_h_results)

            # Compute summary
            summary = compute_extrapolation_summary(per_h_results)
            mpnn_results[n_target] = {
                "n_qubits": n_target,
                "n_params": n_params,
                **summary,
                "n_cached": n_cached,
                "n_predicted": n_predicted,
                "per_point": per_h_results,
            }

            # Concise progress log
            logger.info(
                f"    N={n_target}: ΔE/gap={summary['mean_de_gap']:.4f} "
                f"|ΔE|={summary['mean_abs_error']:.2e} "
                f"pass={summary['n_pass_dual']}/{summary['n_points']} "
                f"({n_cached} cached, {n_predicted} new)"
            )

        self._mpnn_results = mpnn_results
        total_time = time.perf_counter() - t_section
        logger.info(f"  MPNN complete: {total_time:.1f}s")

        # ── Auto-update zoo pass_rate with real extrapolation results ─────
        # The zoo model was used at large N — update its pass_rate so that
        # load_best_model_for_topology reflects actual deployment quality.
        try:
            zoo_entry = getattr(self, "_zoo_entry", None)
            if zoo_entry is not None:
                # Compute average pass_rate_dual across all target N
                avg_pass = float(np.mean([
                    mpnn_results[n]["pass_rate_dual"]
                    for n in self._args.target_n
                    if "pass_rate_dual" in mpnn_results.get(n, {})
                ]))
                if avg_pass > 0:
                    from qmbp_simulation.predictors.model_zoo import update_zoo_pass_rate
                    update_zoo_pass_rate(
                        zoo_entry.checkpoint_file,
                        avg_pass,
                        only_if_better=True,
                        add_notes=f"extrap@N={list(self._args.target_n)}",
                    )
        except Exception as e:
            logger.debug(f"  Zoo pass_rate auto-update skipped: {e}")

        # ── Auto-refine high-uncertainty points (θ_std > threshold) ───────
        # Even without --refine-failing, refine points where the model
        # self-reports high uncertainty (MC-Dropout std). This is a lighter
        # trigger than full AL — uses the already-computed theta_std.
        self._auto_refine_high_uncertainty(mpnn_results)

        # ── Optional: Refine failing points via VQE from MPNN warm-start ──
        if self._args.refine_failing:
            self._refine_failing_points(mpnn_results)

        # ── Optional: Active learning — ensemble uncertainty-based refinement ──
        if self._args.active_learning_rounds > 0:
            self._run_active_learning(mpnn_results)

        return {
            "mpnn_results": {str(k): v for k, v in mpnn_results.items()},
            "checkpoint_used": self._actual_checkpoint,
        }

    def _auto_refine_high_uncertainty(self, mpnn_results: dict[int, dict]) -> None:
        """Automatically refine points where MC-Dropout θ_std exceeds threshold.

        This is a lightweight uncertainty-based refinement that triggers
        without any explicit flag. Only refines points that:
        1. Have theta_std > 2× median theta_std (outliers in uncertainty)
        2. Also fail the dual criterion (high ΔE/gap)

        This avoids wasting VQE budget on points that are uncertain but
        happen to be correct (uncertainty ≠ error always).
        """
        from scipy.optimize import minimize as _minimize

        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.models.model_registry import get_model_spec

        topo = self._args.topology
        p = self._args.p_layers
        spec = get_model_spec(self._args.model_name)
        hva = HVACircuitBuilder()
        maxiter = self._args.vqe_maxiter
        n_auto_refined = 0

        for n_target, data in mpnn_results.items():
            per_point = data.get("per_point", [])
            # Collect theta_std values for this N
            stds = [pt.get("theta_std", 0.0) for pt in per_point]
            nonzero_stds = [s for s in stds if s > 0]
            if len(nonzero_stds) < 3:
                continue  # Not enough MC-Dropout data

            median_std = float(np.median(nonzero_stds))
            threshold = median_std * 2.0  # Outlier = 2× median

            # Find high-uncertainty AND failing points
            candidates = [
                (i, pt) for i, pt in enumerate(per_point)
                if pt.get("theta_std", 0) > threshold
                and pt.get("de_gap", 0) > 0.05
                and pt.get("method") == "mpnn"
            ]
            if not candidates:
                continue

            # Refine top-2 most uncertain (keep it cheap)
            candidates.sort(key=lambda x: x[1].get("theta_std", 0), reverse=True)
            to_refine = candidates[:2]

            lat_ref = self.make_lattice(topo, n_target, J=1.0, h=2.0)
            circuit, _ = hva.create_bond_resolved(n_target, p, lat_ref)
            vqe_backend = self.select_backend(n_target, for_vqe_loop=True)

            for idx, pt in to_refine:
                theta_init = pt.get("theta")
                if theta_init is None:
                    continue
                theta_init = np.array(theta_init, dtype=np.float64)
                h = pt["h"]

                lat_h = self.make_lattice(topo, n_target, J=1.0, h=float(h))
                H = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)

                try:
                    res = _minimize(
                        lambda params: vqe_backend.evaluate(circuit, H, params),
                        theta_init,
                        method="L-BFGS-B",
                        options={"maxiter": maxiter},
                    )
                    if res.fun < pt["e_pred"]:
                        gt_entry = next(
                            g for g in self._gt_data[n_target]
                            if abs(g["h"] - float(h)) < 1e-8
                        )
                        new_result = self.build_per_h_result(
                            h, res.fun, gt_entry["e_exact"], gt_entry["gap"],
                            n_params=circuit.num_parameters,
                            n_qubits=n_target,
                            method="auto_refined",
                            n_evals=res.nfev,
                            theta=res.x.tolist(),
                        )
                        per_point[idx] = new_result
                        n_auto_refined += 1
                except Exception:
                    continue

            if n_auto_refined > 0:
                mpnn_results[n_target].update(compute_extrapolation_summary(per_point))
                mpnn_results[n_target]["per_point"] = per_point
                self._persist_extrapolation_npz(topo, n_target, p, per_point)

        if n_auto_refined > 0:
            logger.info(f"  Auto-refine (θ_std outliers): {n_auto_refined} points improved")

    def _refine_failing_points(self, mpnn_results: dict[int, dict]) -> None:
        """Run VQE refinement on MPNN predictions that fail dual criterion.

        Uses compute_refinement_priority() (same as AcceleratedCrossNRunner)
        to decide which points are worth refining and in what order.
        Skips ansatz-limited / hopeless points automatically.
        """
        from scipy.optimize import minimize as _minimize

        from qmbp_simulation.analysis.metrics import (
            compute_refinement_priority,
            is_point_failure,
        )
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.models.model_registry import get_model_spec

        topo = self._args.topology
        p = self._args.p_layers
        spec = get_model_spec(self._args.model_name)
        hva = HVACircuitBuilder()
        maxiter = self._args.vqe_maxiter
        max_refine = self._args.max_refine

        logger.info(f"  Refining failing points (maxiter={maxiter}, max={max_refine})...")

        for n_target in self._args.target_n:
            per_point = mpnn_results[n_target]["per_point"]
            n_params = mpnn_results[n_target]["n_params"]

            # Identify failures using canonical is_point_failure
            failing_indices = [
                i
                for i, r in enumerate(per_point)
                if is_point_failure(r["de_gap"], abs_error=r.get("abs_error"))
            ]
            if not failing_indices:
                continue

            # Score and filter using compute_refinement_priority
            scored = []
            n_skipped = 0
            for idx in failing_indices:
                r = per_point[idx]
                priority, should_skip, reason = compute_refinement_priority(
                    de_gap=r["de_gap"],
                    abs_error=r.get("abs_error", 1.0),
                    gap=r.get("gap", 1.0),
                    n_params=n_params,
                )
                if should_skip:
                    n_skipped += 1
                else:
                    scored.append((priority, idx, reason))

            # Sort by priority (highest first) and limit
            scored.sort(key=lambda x: x[0], reverse=True)
            to_refine = [(idx, per_point[idx]) for _, idx, _ in scored[:max_refine]]

            if n_skipped > 0:
                logger.info(
                    f"    N={n_target}: skipping {n_skipped}/{len(failing_indices)} "
                    f"points (ansatz-limited or hopeless)"
                )
            if not to_refine:
                continue

            logger.info(
                f"    N={n_target}: refining {len(to_refine)}/{len(failing_indices)} "
                f"failing points (top priority={scored[0][0]:.2f}, reason={scored[0][2]})"
            )

            # Build circuit and backend
            lat_ref = self.make_lattice(topo, n_target, J=1.0, h=2.0)
            circuit, _ = hva.create_bond_resolved(n_target, p, lat_ref)
            vqe_backend = self.select_backend(n_target, for_vqe_loop=True)
            n_refined = 0

            for idx, result in to_refine:
                h = result["h"]
                theta_init = result.get("theta")
                if theta_init is None:
                    continue
                theta_init = np.array(theta_init, dtype=np.float64)

                lat_h = self.make_lattice(topo, n_target, J=1.0, h=float(h))
                H = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)

                try:
                    res = _minimize(
                        lambda params: vqe_backend.evaluate(circuit, H, params),
                        theta_init,
                        method="L-BFGS-B",
                        options={"maxiter": maxiter},
                    )
                    e_refined = res.fun
                except Exception:
                    continue

                # Only update if improved
                if e_refined < result["e_pred"]:
                    gt_entry = next(
                        pt for pt in self._gt_data[n_target] if abs(pt["h"] - float(h)) < 1e-8
                    )
                    new_result = self.build_per_h_result(
                        h,
                        e_refined,
                        gt_entry["e_exact"],
                        gt_entry["gap"],
                        n_params=circuit.num_parameters,
                        n_qubits=n_target,
                        method="vqe_refined",
                        n_evals=res.nfev,
                        theta=res.x.tolist(),
                    )
                    per_point[idx] = new_result
                    n_refined += 1

            if n_refined > 0:
                # Recompute summary and persist
                mpnn_results[n_target].update(compute_extrapolation_summary(per_point))
                mpnn_results[n_target]["per_point"] = per_point
                self._persist_extrapolation_npz(topo, n_target, p, per_point)
                logger.info(
                    f"    N={n_target}: refined {n_refined}/{len(to_refine)} → "
                    f"ΔE/gap={mpnn_results[n_target]['mean_de_gap']:.4f}"
                )

    def _persist_extrapolation_npz(
        self, topology: str, n_qubits: int, p_layers: int, per_h_results: list[dict]
    ) -> None:
        """Persist extrapolation predictions to NPZ with anti-regression and quality tiers.

        Delegates to the shared persist_predictions_to_training_npz() from result_io,
        targeting EXTRAPOLATION_DATA_DIR instead of training data.
        """
        persist_predictions_to_training_npz(
            per_h_results_by_n={n_qubits: per_h_results},
            topology=topology,
            p_layers=p_layers,
            training_data_dir=EXTRAPOLATION_DATA_DIR,
            # Use no threshold filter — persist everything, let upsert_theta_npz
            # anti-regression handle quality (extrapolation data is valuable even
            # at higher error for tracking scaling behavior)
            de_gap_threshold=float("inf"),
            max_abs_error=float("inf"),
            persist_theta_std=True,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Active Learning: Ensemble-based uncertainty targeting
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_active_learning(self, mpnn_results: dict[int, dict]) -> None:
        """Run AL rounds to identify and refine high-uncertainty predictions.

        Uses MC-Dropout on the loaded MPNN to estimate prediction uncertainty,
        then selects points where the model is most uncertain for VQE refinement.
        Reuses: active_learning helpers, build_per_h_result, _persist_extrapolation_npz.
        """
        from experiments.helpers.active_learning import (
            compute_ensemble_uncertainty,
            select_next_point,
        )
        from scipy.optimize import minimize as _minimize

        import torch

        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.predictors.unified_graph import build_unified_bond_resolved_graph

        n_rounds = self._args.active_learning_rounds
        topo = self._args.topology
        p = self._args.p_layers
        model_name = self._args.model_name
        spec = get_model_spec(model_name)
        hva = HVACircuitBuilder()
        maxiter = self._args.vqe_maxiter

        logger.info(f"\n  Active Learning: {n_rounds} round(s), ensemble-based uncertainty")

        mpnn_model = getattr(self, "_model", None)
        if mpnn_model is None:
            logger.warning("  AL: No loaded MPNN model — skipping.")
            return

        for n_target, data in mpnn_results.items():
            per_point = data.get("per_point", [])
            if not per_point:
                continue

            # Identify candidate h-values (those not already VQE-refined)
            candidates = [
                (i, pt) for i, pt in enumerate(per_point)
                if pt.get("method", "mpnn") == "mpnn" and pt.get("de_gap", 0) > 0.01
            ]
            if not candidates:
                continue

            h_candidates = np.array([pt["h"] for _, pt in candidates])

            for al_round in range(n_rounds):
                if len(h_candidates) == 0:
                    break

                # Per-h uncertainty via predict_with_uncertainty (or fallback)
                uncertainties = []
                for h in h_candidates:
                    lat = self.make_lattice(topo, n_target, J=1.0, h=float(h))
                    g = build_unified_bond_resolved_graph(lat, float(h), p)
                    if hasattr(mpnn_model, "predict_with_uncertainty"):
                        _, theta_std = mpnn_model.predict_with_uncertainty(g)
                        uncertainties.append(theta_std)
                    else:
                        with torch.no_grad():
                            pred = mpnn_model(g).squeeze().cpu().numpy()
                        uncertainties.append(float(np.std(pred)))

                # Select top-3 most uncertain
                n_select = min(3, len(h_candidates))
                selected = []
                avail = list(range(len(h_candidates)))
                for _ in range(n_select):
                    if not avail:
                        break
                    sub_h = np.array([h_candidates[i] for i in avail])
                    sub_u = [uncertainties[i] for i in avail]
                    if max(sub_u) < 0.005:
                        break  # All below noise floor
                    # Adaptive acquisition: round 1 = explore, round 2+ = exploit
                    if al_round == 0:
                        acq_fn = "max_variance"
                    else:
                        acq_fn = "expected_improvement"
                    # Current best error for expected_improvement
                    current_best_dg = min(
                        (pt.get("de_gap", 1.0) for _, pt in candidates), default=0.05
                    )
                    best_sub = select_next_point(
                        sub_h, sub_u,
                        acquisition=acq_fn,
                        current_best_error=current_best_dg,
                        predictions_mean=[sub_u[i] for i in range(len(sub_h))],
                    )[0]
                    actual_idx = avail[best_sub]
                    selected.append(actual_idx)
                    avail.remove(actual_idx)

                if not selected:
                    logger.info(f"    N={n_target} AL round {al_round + 1}: "
                                f"uncertainty below threshold — stopping.")
                    break

                # VQE refinement at selected points
                lat_ref = self.make_lattice(topo, n_target, J=1.0, h=2.0)
                circuit, _ = hva.create_bond_resolved(n_target, p, lat_ref)
                vqe_backend = self.select_backend(n_target, for_vqe_loop=True)
                n_improved = 0

                for sel_idx in selected:
                    orig_idx = candidates[sel_idx][0]
                    pt = per_point[orig_idx]
                    h = pt["h"]
                    theta_init = np.array(pt.get("theta", []), dtype=np.float64)
                    if len(theta_init) == 0:
                        continue

                    lat_h = self.make_lattice(topo, n_target, J=1.0, h=float(h))
                    H = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)

                    try:
                        res = _minimize(
                            lambda params: vqe_backend.evaluate(circuit, H, params),
                            theta_init,
                            method="L-BFGS-B",
                            options={"maxiter": maxiter},
                        )
                        if res.fun < pt["e_pred"]:
                            gt = next(
                                g for g in self._gt_data[n_target]
                                if abs(g["h"] - float(h)) < 1e-8
                            )
                            new_result = self.build_per_h_result(
                                h, res.fun, gt["e_exact"], gt["gap"],
                                n_params=circuit.num_parameters,
                                n_qubits=n_target,
                                method="al_refined",
                                n_evals=res.nfev,
                                theta=res.x.tolist(),
                            )
                            per_point[orig_idx] = new_result
                            n_improved += 1
                    except Exception:
                        continue

                if n_improved > 0:
                    from qmbp_simulation.analysis.metrics import compute_deploy_summary
                    mpnn_results[n_target].update(compute_deploy_summary(per_point))
                    mpnn_results[n_target]["per_point"] = per_point
                    self._persist_extrapolation_npz(topo, n_target, p, per_point)

                logger.info(
                    f"    N={n_target} AL round {al_round + 1}: "
                    f"refined {n_improved}/{len(selected)} selected "
                    f"(max_unc={max(uncertainties):.4f})"
                )

                # Remove successfully refined from candidates for next round
                candidates = [
                    (i, pt) for i, pt in enumerate(per_point)
                    if pt.get("method", "mpnn") == "mpnn" and pt.get("de_gap", 0) > 0.01
                ]
                h_candidates = np.array([pt["h"] for _, pt in candidates]) if candidates else np.array([])

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 3: Random VQE Baseline (Optional)
    # ═══════════════════════════════════════════════════════════════════════════

    def section_random_baseline(self) -> dict:
        """Run VQE with random initialization as baseline comparison.

        Uses minimal restarts and iterations (this is expensive at large N).
        Results are persisted to NPZ per-N (crash-safe) and cached to avoid
        re-running on subsequent executions.
        """
        from scipy.optimize import minimize as _minimize

        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.models.model_registry import get_model_spec

        topo = self._args.topology
        p = self._args.p_layers
        spec = get_model_spec(self._args.model_name)
        hva = HVACircuitBuilder()
        maxiter = self._args.vqe_maxiter
        n_restarts = self._args.vqe_restarts
        force_recompute = self._args.force_recompute

        # NPZ directory for baseline results (separate from MPNN extrapolation)
        baseline_dir = EXTRAPOLATION_DATA_DIR / "_baselines"
        baseline_dir.mkdir(parents=True, exist_ok=True)

        random_results: dict[int, list[dict]] = {}
        t_section = time.perf_counter()

        for n_target in self._args.target_n:
            logger.info(
                f"  VQE N={n_target}: {len(self._h_values)} pts × {n_restarts} restarts "
                f"(maxiter={maxiter})..."
            )

            # ── Check for cached baseline results ─────────────────────────
            baseline_npz = baseline_dir / f"{topo}_N{n_target}_p{p}_random_vqe.npz"
            cached_baseline: dict[float, dict] | None = None
            if not force_recompute and baseline_npz.exists():
                try:
                    _data = np.load(baseline_npz, allow_pickle=True)
                    cached_baseline = {}
                    for i, h in enumerate(_data["h_values"]):
                        cached_baseline[round(float(h), 6)] = {
                            "e_vqe": float(_data["e_vqe"][i]),
                            "e_exact": float(_data["e_exact"][i]),
                            "gap": float(_data["gaps"][i]) if "gaps" in _data else 0.0,
                            "n_evals": int(_data["n_evals"][i]) if "n_evals" in _data else 0,
                            "time_s": float(_data["time_s"][i]) if "time_s" in _data else 0.0,
                        }
                    logger.info(f"    Loaded {len(cached_baseline)} cached baseline results")
                except Exception as e:
                    logger.debug(f"    Failed to load baseline cache: {e}")
                    cached_baseline = None

            lat_ref = self.make_lattice(topo, n_target, J=1.0, h=2.0)
            circuit, _ = hva.create_bond_resolved(n_target, p, lat_ref)
            n_params = circuit.num_parameters

            # Use MPS backend for large N (memory efficient)
            vqe_backend = self.select_backend(n_target, for_vqe_loop=True)

            per_h_results = []
            total_evals = 0
            n_cached_pts, n_computed_pts = 0, 0

            for h in self._h_values:
                h_key = round(float(h), 6)

                # Check cache first
                if cached_baseline and h_key in cached_baseline and not force_recompute:
                    cached = cached_baseline[h_key]
                    gt_entry = next(
                        pt for pt in self._gt_data[n_target] if abs(pt["h"] - float(h)) < 1e-8
                    )
                    result = self.build_per_h_result(
                        h,
                        cached["e_vqe"],
                        gt_entry["e_exact"],
                        gt_entry["gap"],
                        n_params=n_params,
                        n_qubits=n_target,
                        method="random_vqe",
                        n_evals=cached["n_evals"],
                        time_s=cached["time_s"],
                    )
                    per_h_results.append(result)
                    total_evals += cached["n_evals"]
                    n_cached_pts += 1
                    continue

                # Fresh VQE computation
                t0 = time.perf_counter()
                lat_h = self.make_lattice(topo, n_target, J=1.0, h=float(h))
                H = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)

                # Multi-restart random VQE
                best_energy = float("inf")
                best_nfev = 0
                rng = np.random.default_rng(42)

                for restart in range(n_restarts):
                    theta0 = rng.uniform(-np.pi, np.pi, size=n_params)
                    try:
                        res = _minimize(
                            lambda params: vqe_backend.evaluate(circuit, H, params),
                            theta0,
                            method="L-BFGS-B",
                            options={"maxiter": maxiter},
                        )
                        if res.fun < best_energy:
                            best_energy = res.fun
                            best_nfev = res.nfev
                    except Exception:
                        continue

                elapsed = time.perf_counter() - t0
                total_evals += best_nfev * n_restarts

                # Get GT
                gt_entry = next(
                    pt for pt in self._gt_data[n_target] if abs(pt["h"] - float(h)) < 1e-8
                )

                result = self.build_per_h_result(
                    h,
                    best_energy,
                    gt_entry["e_exact"],
                    gt_entry["gap"],
                    n_params=n_params,
                    n_qubits=n_target,
                    method="random_vqe",
                    n_evals=best_nfev * n_restarts,
                    time_s=elapsed,
                )
                per_h_results.append(result)
                n_computed_pts += 1

            # ── Persist baseline results to NPZ (crash-safe, per-N) ───────
            if n_computed_pts > 0:
                h_arr = np.array([r["h"] for r in per_h_results])
                e_vqe_arr = np.array([r["e_pred"] for r in per_h_results])
                e_exact_arr = np.array([r["e_exact"] for r in per_h_results])
                gap_arr = np.array([r["gap"] for r in per_h_results])
                n_evals_arr = np.array([r.get("n_evals", 0) for r in per_h_results])
                time_arr = np.array([r.get("time_s", 0.0) for r in per_h_results])

                # Atomic write: tmp → rename (crash-safe)
                from qmbp_simulation.utils.helpers import atomic_savez

                atomic_savez(
                    baseline_npz,
                    h_values=h_arr,
                    e_vqe=e_vqe_arr,
                    e_exact=e_exact_arr,
                    gaps=gap_arr,
                    n_evals=n_evals_arr,
                    time_s=time_arr,
                    n_restarts=np.array(n_restarts),
                    maxiter=np.array(maxiter),
                )
                logger.info(
                    f"    💾 Persisted: {baseline_npz.name} "
                    f"({n_computed_pts} computed, {n_cached_pts} cached)"
                )

            summary = compute_extrapolation_summary(per_h_results)
            random_results[n_target] = {
                "n_qubits": n_target,
                "n_params": n_params,
                **summary,
                "total_evals": total_evals,
                "n_cached": n_cached_pts,
                "n_computed": n_computed_pts,
                "per_point": per_h_results,
            }

            logger.info(
                f"    N={n_target}: ΔE/gap={summary['mean_de_gap']:.4f} "
                f"pass={summary['n_pass_dual']}/{summary['n_points']} "
                f"evals={total_evals} ({n_cached_pts} cached, {n_computed_pts} new)"
            )

        self._random_results = random_results
        total_time = time.perf_counter() - t_section
        logger.info(f"  VQE baseline complete: {total_time:.1f}s")

        return {"random_results": {str(k): v for k, v in random_results.items()}}

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 4: Summary
    # ═══════════════════════════════════════════════════════════════════════════

    def section_summary(self) -> dict:
        """Generate summary table and comparison metrics."""

        comparison = {}
        has_random = hasattr(self, "_random_results")

        # Build comparison table
        for n_target in self._args.target_n:
            mpnn = self._mpnn_results[n_target]
            entry = {
                "N": n_target,
                "n_params": mpnn["n_params"],
                "mpnn": {
                    "mean_de_gap": mpnn["mean_de_gap"],
                    "mean_abs_error": mpnn["mean_abs_error"],
                    "pass_rate_dual": mpnn["pass_rate_dual"],
                    "n_evals": mpnn["n_points"],  # 1 eval per point
                },
            }

            if has_random:
                rand = self._random_results[n_target]
                entry["random_vqe"] = {
                    "mean_de_gap": rand["mean_de_gap"],
                    "mean_abs_error": rand.get("mean_abs_error"),
                    "pass_rate_dual": rand["pass_rate_dual"],
                    "total_evals": rand["total_evals"],
                }
                # Speedup: random evals / MPNN evals
                entry["speedup"] = rand["total_evals"] / max(mpnn["n_points"], 1)

                # Per-h comparison: MPNN wins where it has lower ΔE/gap
                mpnn_pts = mpnn["per_point"]
                rand_pts = rand["per_point"]
                mpnn_wins = sum(
                    1 for m, r in zip(mpnn_pts, rand_pts, strict=False) if m["de_gap"] < r["de_gap"]
                )
                entry["mpnn_win_rate"] = mpnn_wins / len(mpnn_pts)

            # ── Metric reliability checks (delegated to reusable module) ──
            from qmbp_simulation.analysis.evaluation_report import validate_metrics

            warnings_n = validate_metrics(mpnn["per_point"], n_qubits=n_target)
            if warnings_n:
                entry["metric_warnings"] = warnings_n
                for w in warnings_n:
                    logger.warning(f"  N={n_target}: {w}")

            comparison[n_target] = entry

        # Print summary table
        self._print_summary_table(comparison)

        # Check for gap masking
        for n_target, entry in comparison.items():
            if not isinstance(n_target, int):
                continue
            mpnn = entry["mpnn"]
            pass_5pct = self._mpnn_results[n_target]["pass_rate_5pct"]
            pass_dual = mpnn["pass_rate_dual"]
            if pass_5pct > pass_dual + 0.1:
                logger.warning(
                    f"  ⚠️ N={n_target}: Gap masking detected — "
                    f"single={pass_5pct:.0%} vs dual={pass_dual:.0%}"
                )

        # Extensive scaling check: |ΔE| should be approximately constant
        per_site_errors = [
            self._mpnn_results[n].get("mean_abs_error") for n in self._args.target_n
        ]
        # Filter None values (n_qubits missing in result dicts)
        per_site_errors = [e for e in per_site_errors if e is not None and e > 0]
        if len(per_site_errors) >= 2:
            err_ratio = max(per_site_errors) / max(min(per_site_errors), 1e-10)
            if err_ratio < 3.0:
                logger.info(
                    f"  ✅ Extensive scaling confirmed: |ΔE| varies by {err_ratio:.1f}× "
                    f"across N={self._args.target_n}"
                )
            else:
                logger.warning(
                    f"  ⚠️ Per-site error varies by {err_ratio:.1f}× — "
                    f"extensive scaling may not hold"
                )

        # ── Uncertainty calibration: θ_std vs ΔE/gap correlation ──────────
        from qmbp_simulation.analysis.metrics import compute_uncertainty_correlation

        all_points_with_std = []
        for n_target_uc in self._args.target_n:
            all_points_with_std.extend(self._mpnn_results[n_target_uc].get("per_point", []))

        uc_report = compute_uncertainty_correlation(all_points_with_std)
        if uc_report["n_points_with_uncertainty"] >= 3:
            comparison["uncertainty_calibration"] = uc_report
            cal_status = "✅ calibrated" if uc_report["calibrated"] else "⚠️ uncalibrated"
            logger.info(
                f"  Uncertainty calibration ({uc_report['n_points_with_uncertainty']} pts): "
                f"Pearson r={uc_report['pearson_r']:.3f}, "
                f"Spearman ρ={uc_report['spearman_r']:.3f} — {cal_status}"
            )
            logger.info(
                f"    High-unc ΔE/gap={uc_report['high_uncertainty_mean_de_gap']:.4f} vs "
                f"Low-unc ΔE/gap={uc_report['low_uncertainty_mean_de_gap']:.4f}"
            )

        # ── Model Quality Diagnostics ─────────────────────────────────────
        try:
            model_diagnostics = self._compute_model_diagnostics()
            if model_diagnostics:
                comparison["model_diagnostics"] = model_diagnostics
        except Exception as e:
            logger.debug(f"  Model diagnostics skipped: {e}")

        # ── Cross-N Validation Report (L1 from precomputed data) ──────────
        try:
            from qmbp_simulation.analysis.cross_n_validator import quick_cross_n_report

            # Determine training sizes from MultiNAggregator data
            from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

            agg = MultiNAggregator(topology=self._args.topology, model=self._args.model_name)
            agg.scan()
            training_sizes = agg.available_n_values()

            cross_n_reports = {}
            for n_target in self._args.target_n:
                per_point = self._mpnn_results[n_target]["per_point"]
                report = quick_cross_n_report(
                    per_point,
                    n_target,
                    topology=self._args.topology,
                    training_sizes=training_sizes,
                )
                cross_n_reports[n_target] = report.to_dict()
                logger.info(
                    f"  CrossN L1 N={n_target}: "
                    f"mean_ΔE/gap={report.l1_mean_de_gap:.4f}, "
                    f"pass_rate={report.l1_pass_rate:.0%}"
                )
            comparison["cross_n_validation"] = {str(k): v for k, v in cross_n_reports.items()}
        except Exception as e:
            logger.debug(f"  Cross-N validation skipped: {e}")

        # ── Auto-save per-point evaluation report (markdown) ──────────────
        self._save_evaluation_report(comparison)

        # NOTE: EvaluationRecord persistence to ModelRegistryDB is now handled
        # automatically by runner_base._persist_evaluation_to_registry() in
        # the post-run _log_data_quality_feedback() hook. No inline code needed.

        return {"comparison": {str(k): v for k, v in comparison.items()}}

    def _compute_model_diagnostics(self) -> dict:
        """Compute additional model quality metrics from the prediction data.

        Delegates to centralized `compute_mpnn_diagnostics` from analysis.metrics.
        """
        from qmbp_simulation.analysis.metrics import compute_mpnn_diagnostics

        return compute_mpnn_diagnostics(
            mpnn_results_by_n=self._mpnn_results,
            topology=self._args.topology,
            model_name=self._args.model_name,
            p_layers=self._args.p_layers,
            checkpoint_path=self._args.checkpoint,
            include_training_quality=True,
            logger=logger,
        )

    def _save_evaluation_report(self, comparison: dict) -> None:
        """Save a per-point evaluation breakdown as markdown for analysis.

        Delegates to the reusable generate_evaluation_report() from
        qmbp_simulation.analysis.evaluation_report.
        """
        from qmbp_simulation.analysis.evaluation_report import (
            generate_evaluation_report,
        )

        checkpoint_display = (
            getattr(self, "_actual_checkpoint", None) or self._args.checkpoint or "unknown"
        )

        generate_evaluation_report(
            mpnn_results_by_n=self._mpnn_results,
            topology=self._args.topology,
            model_name=self._args.model_name,
            checkpoint=checkpoint_display,
            h_range=(self._args.h_min, self._args.h_max),
            n_h_points=self._args.h_points,
            p_layers=self._args.p_layers,
            target_n=self._args.target_n,
            comparison=comparison,
            output_dir="results/extrapolation_evals",
        )

    def _print_summary_table(self, comparison: dict) -> None:
        """Print formatted summary table to console with continuous metrics."""
        has_random = any("random_vqe" in v for v in comparison.values())

        logger.info("\n" + "=" * 80)
        logger.info("LARGE-N EXTRAPOLATION SUMMARY")
        logger.info("=" * 80)

        if has_random:
            header = (
                f"{'N':>5} | {'params':>6} | {'ΔE/gap(mean±std)':>18} | "
                f"{'P90':>6} | {'|ΔE|':>10} | {'Grade':>5} | "
                f"{'VQE ΔE/gap':>10} | {'speedup':>8}"
            )
        else:
            header = (
                f"{'N':>5} | {'params':>6} | {'ΔE/gap(mean±std)':>18} | "
                f"{'P90':>6} | {'|ΔE|':>10} | {'Grade':>5}"
            )

        logger.info(header)
        logger.info("-" * 80)

        for n_target in sorted(comparison.keys()):
            if not isinstance(n_target, int):
                continue
            entry = comparison[n_target]
            mpnn = entry["mpnn"]
            mpnn_dg = mpnn["mean_de_gap"]
            mpnn_std = self._mpnn_results[n_target].get("std_de_gap", 0.0) or 0.0
            mpnn_p90 = self._mpnn_results[n_target].get("p90_de_gap", mpnn_dg) or mpnn_dg
            mpnn_abs_error = mpnn.get("mean_abs_error", 0.0) or 0.0
            grade = self._mpnn_results[n_target].get("grade", "?")

            if has_random and "random_vqe" in entry:
                rand_dg = entry["random_vqe"]["mean_de_gap"]
                spd = entry.get("speedup", 0)
                logger.info(
                    f"{n_target:>5} | {entry['n_params']:>6} | "
                    f"{mpnn_dg:.4f}±{mpnn_std:.4f}   | "
                    f"{mpnn_p90:>5.3f} | {mpnn_abs_error:>10.2e} | "
                    f"{grade:>5} | {rand_dg:>10.4f} | {spd:>7.0f}×"
                )
            else:
                logger.info(
                    f"{n_target:>5} | {entry['n_params']:>6} | "
                    f"{mpnn_dg:.4f}±{mpnn_std:.4f}   | "
                    f"{mpnn_p90:>5.3f} | {mpnn_abs_error:>10.2e} | "
                    f"{grade:>5}"
                )

        logger.info("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    LargeNExtrapolationRunner.main()
