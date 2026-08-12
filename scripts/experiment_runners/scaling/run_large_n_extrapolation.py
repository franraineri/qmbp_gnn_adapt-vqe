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

Reports ΔE/gap, |ΔE|, |ΔE|/N, speedup (n_evals ratio), and wall time.
All metrics follow dual criterion (ΔE/gap < 5% AND |ΔE| < 0.10).

Usage:
    # Default: chain_1d, N=[30, 40, 60, 100], 6 h-points
    python scripts/experiment_runners/scaling/run_large_n_extrapolation.py

    # Custom topology and sizes
    python scripts/experiment_runners/scaling/run_large_n_extrapolation.py \\
        --topology chain_1d --target-n 30 50 80 --h-min 2.5 --h-max 5.0

    # Quick smoke test (fewer points)
    python scripts/experiment_runners/scaling/run_large_n_extrapolation.py \\
        --target-n 30 --h-points 3

    # Skip VQE baseline (faster, MPNN-only)
    python scripts/experiment_runners/scaling/run_large_n_extrapolation.py \\
        --skip-random-baseline

    # With VQE refinement for failing points
    python scripts/experiment_runners/scaling/run_large_n_extrapolation.py \\
        --refine-failing --vqe-maxiter 100

    # Dry run
    python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --dry-run
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from qmbp_simulation.framework.runner_base import (
    Section,
    ValidationRunner,
    resolve_project_root,
)
from qmbp_simulation.framework.result_io import upsert_theta_npz

if TYPE_CHECKING:
    from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN

_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants (configurable defaults)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_TARGET_N = [30, 40, 60, 100]
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


def load_extrapolation_npz(
    topology: str, n_qubits: int, p_layers: int
) -> dict[float, dict] | None:
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
        return {"n_points": 0, "pass_rate_5pct": 0, "pass_rate_dual": 0}

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
        "per-site error |ΔE|/N ≈ constant, demonstrating extensive scaling."
    )

    @classmethod
    def _add_custom_args(cls, parser):
        # Target system sizes
        parser.add_argument(
            "--target-n", type=int, nargs="+", default=DEFAULT_TARGET_N,
            help="Target system size(s) to test (default: %(default)s)",
        )
        # Physics config (reuse naming conventions from accelerated_cross_n)
        parser.add_argument(
            "--topology", type=str, default=DEFAULT_TOPOLOGY,
            help="Lattice topology (default: %(default)s)",
        )
        parser.add_argument(
            "--model-name", type=str, default=DEFAULT_MODEL,
            help="Physics model name (default: %(default)s)",
        )
        parser.add_argument(
            "--p-layers", type=int, default=DEFAULT_P,
            help="HVA layer depth (default: %(default)s)",
        )
        # H-grid
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
        # VQE baseline config
        parser.add_argument(
            "--skip-random-baseline", action="store_true", default=False,
            help="Skip random-init VQE baseline (faster, MPNN-only)",
        )
        parser.add_argument(
            "--vqe-maxiter", type=int, default=DEFAULT_VQE_MAXITER,
            help="VQE optimizer max iterations (default: %(default)s)",
        )
        parser.add_argument(
            "--vqe-restarts", type=int, default=DEFAULT_VQE_RESTARTS,
            help="VQE random restarts (default: %(default)s)",
        )
        # Refinement option
        parser.add_argument(
            "--refine-failing", action="store_true", default=False,
            help="Run VQE refinement on MPNN predictions that fail dual criterion",
        )
        parser.add_argument(
            "--max-refine", type=int, default=3,
            help="Max points to refine per N (default: 3)",
        )
        # Model selection
        parser.add_argument(
            "--checkpoint", type=str, default=None,
            help="Explicit model checkpoint path (overrides zoo search)",
        )
        # Cache control
        parser.add_argument(
            "--no-eval-cache", action="store_true", default=False,
            help="Disable evaluation caching",
        )
        parser.add_argument(
            "--force-recompute", action="store_true", default=False,
            help="Ignore existing NPZ cache and recompute all predictions",
        )

    def build_config(self) -> dict:
        """Build config dict reusing base class helper."""
        config = self._build_physics_config()
        config.update({
            "model": self._args.model_name,
            "skip_random_baseline": self._args.skip_random_baseline,
            "refine_failing": self._args.refine_failing,
            "vqe_maxiter": self._args.vqe_maxiter,
            "vqe_restarts": self._args.vqe_restarts,
        })
        return config

    def setup(self):
        """Initialize physics objects and h-grid."""
        self.setup_physics()
        self._h_values = np.linspace(
            self._args.h_min, self._args.h_max, self._args.h_points
        )
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
                hypothesis="MPNN achieves |ΔE|/N ≈ constant (extensive scaling)",
            ),
        ]
        if not self._args.skip_random_baseline:
            sections.append(Section(
                id=3,
                name="Random VQE Baseline",
                fn=self.section_random_baseline,
                hypothesis="Random-init VQE with limited budget is worse than MPNN",
            ))
        sections.append(Section(
            id=4 if not self._args.skip_random_baseline else 3,
            name="Summary",
            fn=self.section_summary,
            hypothesis="MPNN demonstrates extensive scaling with N",
        ))
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
                    self._args.topology, n_target, float(h),
                    model=self._args.model_name,
                )
                elapsed = time.perf_counter() - t0

                gt_data[n_target].append({
                    "h": float(h),
                    "e_exact": float(e_exact),
                    "gap": float(gap),
                    "time_s": elapsed,
                })

                # Track cache vs compute (< 0.1s typically means cache hit)
                if elapsed < 0.1:
                    n_cached += 1
                else:
                    n_computed += 1

            logger.info(
                f"    N={n_target}: {n_cached} cached, {n_computed} computed"
            )

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

            # CachedBackend for transparent eval caching
            with self.get_cached_backend(
                topology=topo, n_qubits=n_target,
                model=self._args.model_name, p_layers=p,
                enabled=use_cache,
            ) as eval_backend:
                per_h_results = []
                n_cached, n_predicted = 0, 0

                for h in self._h_values:
                    h_key = round(float(h), 6)

                    # Check NPZ cache first
                    if existing_data and h_key in existing_data and not force_recompute:
                        cached = existing_data[h_key]
                        # Only use cache if we have energy (theta might not be stored)
                        if cached.get("e_pred") is not None:
                            gt_entry = next(
                                pt for pt in self._gt_data[n_target]
                                if abs(pt["h"] - float(h)) < 1e-8
                            )
                            result = self.build_per_h_result(
                                h, cached["e_pred"], gt_entry["e_exact"], gt_entry["gap"],
                                n_params=n_params, n_qubits=n_target, method="cached",
                                theta=cached.get("theta", np.zeros(0)).tolist() if cached.get("theta") is not None else None,
                            )
                            per_h_results.append(result)
                            n_cached += 1
                            continue

                    # Fresh prediction
                    t0 = time.perf_counter()
                    g = build_unified_bond_resolved_graph(
                        lat_ref, h_value=float(h), p_layers=p,
                        include_circuit_nodes=True,
                    )
                    with torch.no_grad():
                        theta_pred = model(g).numpy().flatten()
                    theta_pred = np.clip(theta_pred, -np.pi, np.pi)

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
                        pt for pt in self._gt_data[n_target]
                        if abs(pt["h"] - float(h)) < 1e-8
                    )
                    result = self.build_per_h_result(
                        h, e_pred, gt_entry["e_exact"], gt_entry["gap"],
                        n_params=n_params, n_qubits=n_target, method="mpnn",
                        time_s=elapsed, theta=theta_pred.tolist(),
                    )
                    per_h_results.append(result)
                    n_predicted += 1

                # Persist to NPZ (atomic, anti-regression)
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
                f"|ΔE|/N={summary['mean_abs_error_per_site']:.2e} "
                f"pass={summary['n_pass_dual']}/{summary['n_points']} "
                f"({n_cached} cached, {n_predicted} new)"
            )

        self._mpnn_results = mpnn_results
        total_time = time.perf_counter() - t_section
        logger.info(f"  MPNN complete: {total_time:.1f}s")

        # ── Optional: Refine failing points via VQE from MPNN warm-start ──
        if self._args.refine_failing:
            self._refine_failing_points(mpnn_results)

        return {"mpnn_results": {str(k): v for k, v in mpnn_results.items()}}

    def _refine_failing_points(self, mpnn_results: dict[int, dict]) -> None:
        """Run VQE refinement on MPNN predictions that fail dual criterion.

        Uses theta_pred from MPNN as initial point for L-BFGS-B (warm-start).
        Only refines up to max_refine points per N, prioritized by highest ΔE/gap.
        Updates per_point results and NPZ in-place.
        """
        from scipy.optimize import minimize as _minimize

        from qmbp_simulation.analysis.metrics import DE_GAP_THRESHOLD, MAX_ABS_ERROR
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.models.model_registry import get_model_spec

        topo = self._args.topology
        p = self._args.p_layers
        spec = get_model_spec(self._args.model_name)
        hva = HVACircuitBuilder()
        maxiter = self._args.vqe_maxiter
        max_refine = self._args.max_refine

        logger.info(f"  Refining failing points (maxiter={maxiter}, max={max_refine}/N)...")

        for n_target in self._args.target_n:
            per_point = mpnn_results[n_target]["per_point"]

            # Find failing points (sorted by worst ΔE/gap first)
            failing = [
                (i, r) for i, r in enumerate(per_point)
                if r["de_gap"] >= DE_GAP_THRESHOLD or r.get("abs_error", 1.0) >= MAX_ABS_ERROR
            ]
            failing.sort(key=lambda x: x[1]["de_gap"], reverse=True)
            to_refine = failing[:max_refine]

            if not to_refine:
                continue

            logger.info(f"    N={n_target}: refining {len(to_refine)}/{len(failing)} failing points")

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
                        pt for pt in self._gt_data[n_target]
                        if abs(pt["h"] - float(h)) < 1e-8
                    )
                    new_result = self.build_per_h_result(
                        h, e_refined, gt_entry["e_exact"], gt_entry["gap"],
                        n_params=circuit.num_parameters, n_qubits=n_target,
                        method="vqe_refined", n_evals=res.nfev,
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
        """Persist extrapolation predictions to NPZ with anti-regression and quality tiers."""
        from qmbp_simulation.analysis.metrics import DE_GAP_THRESHOLD, MAX_ABS_ERROR

        npz_path = EXTRAPOLATION_DATA_DIR / f"{topology}_N{n_qubits}_p{p_layers}.npz"

        # Filter out results without theta (cached entries that didn't have it)
        valid_results = [
            r for r in per_h_results
            if r.get("theta") is not None and len(r.get("theta", [])) > 0
        ]
        if not valid_results:
            return

        h_arr = np.array([r["h"] for r in valid_results], dtype=np.float64)
        
        # Convert theta lists to 2D array (n_points, n_params)
        theta_list = []
        for r in valid_results:
            theta = r["theta"]
            if isinstance(theta, np.ndarray):
                theta_list.append(theta.astype(np.float64))
            else:
                theta_list.append(np.array(theta, dtype=np.float64))
        
        # Stack into 2D array if all have same length, otherwise use object array
        try:
            if len(theta_list) > 0 and all(len(t) == len(theta_list[0]) for t in theta_list):
                theta_arr = np.stack(theta_list).astype(np.float64)
            else:
                theta_arr = np.array(theta_list, dtype=object)
        except Exception:
            theta_arr = np.array(theta_list, dtype=object)
        
        e_pred_arr = np.array([r["e_pred"] for r in valid_results], dtype=np.float64)
        e_exact_arr = np.array([r["e_exact"] for r in valid_results], dtype=np.float64)
        gap_arr = np.array([r["gap"] for r in valid_results], dtype=np.float64)
        method_arr = [r.get("method", "mpnn") for r in valid_results]

        # Assign quality tier based on dual criterion
        quality_tiers = []
        for r in valid_results:
            de_gap = r.get("de_gap", 1.0)
            abs_err = r.get("abs_error", float("inf"))
            method = r.get("method", "mpnn")

            # VQE-refined points that pass dual criterion → verified
            if method in ("vqe_refined", "random_vqe") and de_gap < DE_GAP_THRESHOLD and abs_err < MAX_ABS_ERROR:
                quality_tiers.append("verified")
            # MPNN predictions that pass dual criterion → approximate
            elif de_gap < DE_GAP_THRESHOLD and abs_err < MAX_ABS_ERROR:
                quality_tiers.append("approximate")
            else:
                quality_tiers.append("unverified")

        n_upd, n_add = upsert_theta_npz(
            npz_path,
            h_new=h_arr,
            theta_new=theta_arr,
            e_vqe_new=e_pred_arr,
            e_exact_new=e_exact_arr,
            gaps_new=gap_arr,
            method_new=method_arr,
            quality_tier_new=quality_tiers,
        )
        if n_upd + n_add > 0:
            n_verified = quality_tiers.count("verified")
            n_approx = quality_tiers.count("approximate")
            logger.info(
                f"    NPZ: {n_add} added, {n_upd} improved → {npz_path.name} "
                f"(✅{n_verified} ⚠️{n_approx})"
            )


    # ═══════════════════════════════════════════════════════════════════════════
    # Section 3: Random VQE Baseline (Optional)
    # ═══════════════════════════════════════════════════════════════════════════

    def section_random_baseline(self) -> dict:
        """Run VQE with random initialization as baseline comparison.

        Uses minimal restarts and iterations (this is expensive at large N).
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

        random_results: dict[int, list[dict]] = {}
        t_section = time.perf_counter()

        for n_target in self._args.target_n:
            logger.info(
                f"  VQE N={n_target}: {len(self._h_values)} pts × {n_restarts} restarts "
                f"(maxiter={maxiter})..."
            )

            lat_ref = self.make_lattice(topo, n_target, J=1.0, h=2.0)
            circuit, _ = hva.create_bond_resolved(n_target, p, lat_ref)
            n_params = circuit.num_parameters

            # Use MPS backend for large N (memory efficient)
            vqe_backend = self.select_backend(n_target, for_vqe_loop=True)

            per_h_results = []
            total_evals = 0

            for h in self._h_values:
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
                    pt for pt in self._gt_data[n_target]
                    if abs(pt["h"] - float(h)) < 1e-8
                )

                result = self.build_per_h_result(
                    h, best_energy, gt_entry["e_exact"], gt_entry["gap"],
                    n_params=n_params, n_qubits=n_target, method="random_vqe",
                    n_evals=best_nfev * n_restarts, time_s=elapsed,
                )
                per_h_results.append(result)

            summary = compute_extrapolation_summary(per_h_results)
            random_results[n_target] = {
                "n_qubits": n_target,
                "n_params": n_params,
                **summary,
                "total_evals": total_evals,
                "per_point": per_h_results,
            }

            logger.info(
                f"    N={n_target}: ΔE/gap={summary['mean_de_gap']:.4f} "
                f"pass={summary['n_pass_dual']}/{summary['n_points']} "
                f"evals={total_evals}"
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
        from qmbp_simulation.analysis.metrics import DE_GAP_THRESHOLD, MAX_ABS_ERROR

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
                    "mean_abs_error_per_site": mpnn["mean_abs_error_per_site"],
                    "pass_rate_dual": mpnn["pass_rate_dual"],
                    "n_evals": mpnn["n_points"],  # 1 eval per point
                },
            }

            if has_random:
                rand = self._random_results[n_target]
                entry["random_vqe"] = {
                    "mean_de_gap": rand["mean_de_gap"],
                    "pass_rate_dual": rand["pass_rate_dual"],
                    "total_evals": rand["total_evals"],
                }
                # Speedup: random evals / MPNN evals
                entry["speedup"] = rand["total_evals"] / max(mpnn["n_points"], 1)

                # Per-h comparison: MPNN wins where it has lower ΔE/gap
                mpnn_pts = mpnn["per_point"]
                rand_pts = rand["per_point"]
                mpnn_wins = sum(
                    1 for m, r in zip(mpnn_pts, rand_pts)
                    if m["de_gap"] < r["de_gap"]
                )
                entry["mpnn_win_rate"] = mpnn_wins / len(mpnn_pts)

            comparison[n_target] = entry

        # Print summary table
        self._print_summary_table(comparison)

        # Check for gap masking
        for n_target, entry in comparison.items():
            mpnn = entry["mpnn"]
            pass_5pct = self._mpnn_results[n_target]["pass_rate_5pct"]
            pass_dual = mpnn["pass_rate_dual"]
            if pass_5pct > pass_dual + 0.1:
                logger.warning(
                    f"  ⚠️ N={n_target}: Gap masking detected — "
                    f"single={pass_5pct:.0%} vs dual={pass_dual:.0%}"
                )

        # Extensive scaling check: |ΔE|/N should be approximately constant
        per_site_errors = [
            self._mpnn_results[n]["mean_abs_error_per_site"]
            for n in self._args.target_n
        ]
        if len(per_site_errors) >= 2:
            err_ratio = max(per_site_errors) / max(min(per_site_errors), 1e-10)
            if err_ratio < 3.0:
                logger.info(
                    f"  ✅ Extensive scaling confirmed: |ΔE|/N varies by {err_ratio:.1f}× "
                    f"across N={self._args.target_n}"
                )
            else:
                logger.warning(
                    f"  ⚠️ Per-site error varies by {err_ratio:.1f}× — "
                    f"extensive scaling may not hold"
                )

        return {"comparison": {str(k): v for k, v in comparison.items()}}

    def _print_summary_table(self, comparison: dict) -> None:
        """Print formatted summary table to console."""
        has_random = any("random_vqe" in v for v in comparison.values())

        logger.info("\n" + "=" * 80)
        logger.info("LARGE-N EXTRAPOLATION SUMMARY")
        logger.info("=" * 80)

        if has_random:
            header = (
                f"{'N':>5} | {'params':>6} | {'MPNN ΔE/gap':>11} | {'|ΔE|/N':>10} | "
                f"{'pass%':>6} | {'VQE ΔE/gap':>10} | {'speedup':>8}"
            )
        else:
            header = (
                f"{'N':>5} | {'params':>6} | {'MPNN ΔE/gap':>11} | {'|ΔE|/N':>10} | "
                f"{'pass%':>6}"
            )

        logger.info(header)
        logger.info("-" * 80)

        for n_target in sorted(comparison.keys()):
            entry = comparison[n_target]
            mpnn = entry["mpnn"]
            mpnn_dg = mpnn["mean_de_gap"]
            mpnn_per_site = mpnn["mean_abs_error_per_site"]
            mpnn_pass = mpnn["pass_rate_dual"] * 100

            if has_random and "random_vqe" in entry:
                rand_dg = entry["random_vqe"]["mean_de_gap"]
                spd = entry.get("speedup", 0)
                logger.info(
                    f"{n_target:>5} | {entry['n_params']:>6} | {mpnn_dg:>10.4f} | "
                    f"{mpnn_per_site:>10.2e} | {mpnn_pass:>5.1f}% | "
                    f"{rand_dg:>10.4f} | {spd:>7.0f}×"
                )
            else:
                logger.info(
                    f"{n_target:>5} | {entry['n_params']:>6} | {mpnn_dg:>10.4f} | "
                    f"{mpnn_per_site:>10.2e} | {mpnn_pass:>5.1f}%"
                )

        logger.info("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    LargeNExtrapolationRunner.main()
