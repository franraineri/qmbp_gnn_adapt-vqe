"""Accelerated VQE Pipeline — MPNN-driven warm-start with automatic optimization.

Integrates findings F1-F7 into a single reusable component:
- P1: Pipeline acelerado (5 anchor VQE + MPNN predict rest → 3× speedup)
- P2: Pre-trained model zoo (load checkpoint if available, skip training)
- P3: VQE quality predictor (estimate convergence before running)
- P4: Adaptive h-grid (identify uncertain points, run VQE only there)

Usage (minimal — zero config):
    from qmbp_simulation.pipeline.accelerated import AcceleratedVQE

    accel = AcceleratedVQE(lattice, circuit, spec, backend)
    results = accel.run(h_values)
    # results.theta_opt: np.ndarray [n_points, n_params]
    # results.energies: np.ndarray [n_points]
    # results.de_gaps: np.ndarray [n_points]
    # results.method: list[str] — "vqe_full", "mpnn_direct", "mpnn_refined"

Usage (advanced — control every step):
    accel = AcceleratedVQE(lattice, circuit, spec, backend,
        n_anchors=5,
        mpnn_epochs=3000,
        refine_threshold=0.10,  # refine predictions with ΔE/gap > 10%
        zoo_dir="models/zoo",   # pre-trained model directory
    )
    results = accel.run(h_values, seed=42)

Integration with runners:
    # In any ValidationRunner subclass:
    def section_vqe(self):
        accel = AcceleratedVQE(self._lattice, self._circuit, self._spec, self.noiseless)
        return accel.run(self._h_values, seed=self._seed)

References:
    - Finding F1: MPNN replaces VQE (ratio 1.02×)
    - Finding F2: 3× speedup with 5 anchors
    - Finding F5: Convergence is topology-dependent
    - Finding F6: Extrapolation works UP only
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AcceleratedResult:
    """Result of an accelerated VQE sweep."""

    h_values: np.ndarray
    theta_opt: np.ndarray  # [n_points, n_params]
    energies: np.ndarray  # [n_points]
    de_gaps: np.ndarray  # [n_points]
    gaps: np.ndarray  # [n_points]
    e_exact: np.ndarray  # [n_points]
    method: list[str]  # per-point: "vqe_full", "mpnn_direct", "mpnn_refined"

    # Timing breakdown
    time_total_s: float = 0.0
    time_anchor_vqe_s: float = 0.0
    time_mpnn_train_s: float = 0.0
    time_predict_s: float = 0.0
    time_refine_s: float = 0.0

    # Metadata
    n_anchors: int = 0
    n_predicted: int = 0
    n_refined: int = 0
    mpnn_mse: float = 0.0
    model_source: str = ""  # "trained", "zoo", "none"
    convergence_warnings: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Fraction of points with ΔE/gap < 5%."""
        return float((self.de_gaps < 0.05).mean()) if len(self.de_gaps) > 0 else 0.0

    @property
    def speedup_estimate(self) -> float:
        """Estimated speedup vs full VQE (based on anchor fraction)."""
        if self.n_anchors == 0:
            return 1.0
        full_time = self.time_anchor_vqe_s * len(self.h_values) / max(self.n_anchors, 1)
        return full_time / max(self.time_total_s, 0.1)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        from qmbp_simulation.utils.helpers import json_serialize
        return {
            "h_values": self.h_values.tolist(),
            "theta_opt": self.theta_opt.tolist(),
            "energies": self.energies.tolist(),
            "de_gaps": self.de_gaps.tolist(),
            "method": self.method,
            "timing": {
                "total_s": self.time_total_s,
                "anchor_vqe_s": self.time_anchor_vqe_s,
                "mpnn_train_s": self.time_mpnn_train_s,
                "predict_s": self.time_predict_s,
                "refine_s": self.time_refine_s,
            },
            "summary": {
                "n_anchors": self.n_anchors,
                "n_predicted": self.n_predicted,
                "n_refined": self.n_refined,
                "mpnn_mse": self.mpnn_mse,
                "model_source": self.model_source,
                "pass_rate": self.pass_rate,
                "mean_de_gap": float(self.de_gaps.mean()),
                "speedup_estimate": self.speedup_estimate,
            },
            "convergence_warnings": self.convergence_warnings,
        }


@dataclass
class AcceleratedConfig:
    """Configuration for AcceleratedVQE."""

    # Anchor selection
    n_anchors: int = 5
    anchor_strategy: str = "uniform"  # "uniform", "endpoints_plus_center"

    # VQE for anchors
    n_restarts: int = 10
    maxiter: int = 500
    force_method: str | None = None  # If set, bypasses COBYLA_AUTO_SWITCH. "COBYLA" or "L-BFGS-B"
    bidirectional_anchors: bool = False  # If True, run ascending merge on failing anchors

    # MPNN training
    mpnn_epochs: int = 3000
    mpnn_hidden_dim: int = 256
    mpnn_type_embedding_dim: int = 16

    # Refinement
    refine_threshold: float = 0.10  # Refine predictions with ΔE/gap > this
    refine_restarts: int = 1
    refine_maxiter: int = 200

    # Model zoo (P2) — uses qmbp_simulation.predictors.model_zoo automatically
    use_zoo: bool = True  # Try loading pre-trained model before training

    # Quality prediction (P3)
    skip_below_h_min: bool = False  # DISABLED: let all h-points be computed

    # Active learning (P4)
    active_learning_rounds: int = 0  # 0 = disabled. Each round adds uncertain points.


# ═══════════════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════════════


class AcceleratedVQE:
    """MPNN-accelerated VQE sweep for bond-resolved HVA.

    Orchestrates the full accelerated pipeline:
    1. Preflight: estimate valid regime, warn if h-points are outside (P3)
    2. Zoo lookup: check for pre-trained MPNN matching this config (P2)
    3. Anchor VQE: run full VQE at K strategically chosen points (P1)
    4. Train/load MPNN: fit predictor on anchor data
    5. Predict: generate θ_init for all remaining points
    6. Refine: short VQE from prediction for uncertain points (P4)

    All steps are optional and configurable. The default behavior gives
    the 3× speedup demonstrated in Finding F2.

    Parameters
    ----------
    lattice : LatticeConfig
        Lattice specification (topology, N, edges).
    circuit : QuantumCircuit
        Bond-resolved HVA circuit (from HVACircuitBuilder.create_bond_resolved).
    spec : ModelSpec
        Model spec for Hamiltonian construction.
    backend : ExecutionBackend
        Backend for energy evaluation (NoiselessBackend or MPSBackend).
    solver : ClassicalSolver | None
        For exact energies. If None, created internally.
    config : AcceleratedConfig | None
        Configuration. If None, uses defaults.
    """

    def __init__(
        self,
        lattice,
        circuit,
        spec,
        backend,
        solver=None,
        config: AcceleratedConfig | None = None,
        *,
        eval_cache: bool = True,
    ):
        self.lattice = lattice
        self.circuit = circuit
        self.spec = spec
        self.config = config or AcceleratedConfig()

        # Wrap backend with eval cache for automatic reuse of computations
        if eval_cache:
            from qmbp_simulation.execution.eval_cache import CachedBackend

            self.backend = CachedBackend(
                backend,
                topology=lattice.topology,
                n_qubits=lattice.n_qubits,
                model=spec.name if hasattr(spec, "name") else "tfim",
                p_layers=circuit.num_parameters // 2,  # Approximate
                J=float(lattice.J) if np.isscalar(lattice.J) else 1.0,
            )
        else:
            self.backend = backend

        if solver is None:
            from qmbp_simulation import ClassicalSolver
            solver = ClassicalSolver()
        self.solver = solver

        self._n_params = circuit.num_parameters
        self._N = lattice.n_qubits
        self._topology = lattice.topology
        self._model = None  # MPNN model (trained or loaded from zoo)

    def run(
        self,
        h_values: np.ndarray | list[float],
        seed: int = 42,
        p_layers: int = 1,
    ) -> AcceleratedResult:
        """Execute the full accelerated sweep.

        Parameters
        ----------
        h_values : array-like
            Transverse field values (will be sorted descending internally).
        seed : int
            Random seed for VQE and MPNN training.
        p_layers : int
            HVA layers (must match circuit).

        Returns
        -------
        AcceleratedResult
            Complete results with per-point θ, energies, and timing.
        """
        from qmbp_simulation import VQEConfig, VQEOptimizer, make_lattice
        from qmbp_simulation.utils.helpers import canonicalize_theta

        h_values = np.asarray(h_values, dtype=float)
        t_total_start = time.perf_counter()
        cfg = self.config

        # ── Validation (Table 3) ─────────────────────────────────────
        if len(h_values) < 3:
            raise ValueError(
                f"Need at least 3 h-points, got {len(h_values)}. "
                f"AcceleratedVQE requires enough points for meaningful anchor+predict split."
            )
        if cfg.n_anchors >= len(h_values):
            logger.info("  n_anchors=%d >= n_points=%d: running full VQE (no acceleration)",
                        cfg.n_anchors, len(h_values))

        # Validate p_layers matches circuit
        expected_params_per_layer = self._n_params // p_layers if p_layers > 0 else self._n_params
        if self._n_params % p_layers != 0 and p_layers > 1:
            logger.warning(
                "  ⚠️ circuit.num_parameters=%d is not divisible by p_layers=%d. "
                "This may indicate a p_layers mismatch.",
                self._n_params, p_layers,
            )

        # Ensure h_values are sorted descending for warm-start
        if len(h_values) > 1 and h_values[0] < h_values[-1]:
            logger.info("  Sorting h_values descending for warm-start optimization")
            sort_idx = np.argsort(h_values)[::-1]
            h_values = h_values[sort_idx]
            self._h_reorder = sort_idx  # track for final reassembly
        else:
            self._h_reorder = None

        # ── Step 0: Exact diag (ground truth for ΔE/gap) ─────────────
        e_exact, gaps = self._compute_ground_truth(h_values)

        # ── Step 1: Preflight — estimate valid regime (P3) ────────────
        h_min_valid = self._estimate_regime_boundary(p_layers)
        warnings = []
        if cfg.skip_below_h_min and h_min_valid > 0:
            n_below = (h_values < h_min_valid).sum()
            if n_below > 0:
                warnings.append(
                    f"{n_below}/{len(h_values)} h-points below estimated regime "
                    f"boundary h_min={h_min_valid:.2f} for {self._topology} p={p_layers}. "
                    f"VQE may not converge at these points."
                )
                logger.warning("  P3 preflight: %s", warnings[-1])

        # ── Step 2: Check model zoo (P2) ──────────────────────────────
        model_source = "none"
        if cfg.use_zoo:
            zoo_model = self._try_load_from_zoo(p_layers)
            if zoo_model is not None:
                # Validate output dimension matches circuit (Table 3, issue #3)
                if hasattr(zoo_model, "output_dim") and zoo_model.output_dim != self._n_params:
                    logger.warning(
                        "  ⚠️ Zoo model output_dim=%d != circuit.num_parameters=%d. "
                        "Discarding zoo model (architecture mismatch).",
                        zoo_model.output_dim, self._n_params,
                    )
                else:
                    self._model = zoo_model
                    model_source = "zoo"
                    logger.info("  P2: Loaded pre-trained model from zoo")

        # ── Step 3: Anchor selection + full VQE (P1) ──────────────────
        anchor_idx = self._select_anchors(h_values, gaps)
        target_idx = np.array([i for i in range(len(h_values)) if i not in anchor_idx])

        logger.info("  P1: %d anchor points, %d target points",
                    len(anchor_idx), len(target_idx))

        # Run full VQE at anchors (descending warm-start)
        t_anchor_start = time.perf_counter()
        anchor_theta, anchor_energies = self._run_anchor_vqe(
            h_values, anchor_idx, e_exact, gaps, seed, p_layers
        )
        t_anchor = time.perf_counter() - t_anchor_start

        # Store anchor data for ThetaValidator in _predict_theta
        self._anchor_theta = anchor_theta
        self._anchor_h = h_values[anchor_idx]

        # Validate anchor quality (Table 3): abort if ALL anchors are bad
        anchor_de_gaps = np.abs(anchor_energies - e_exact[anchor_idx]) / np.maximum(gaps[anchor_idx], 1e-10)
        if np.all(anchor_de_gaps > 0.50):
            warnings.append(
                f"ALL {len(anchor_idx)} anchor VQE points have ΔE/gap > 50%. "
                f"MPNN training data is likely too poor for meaningful predictions. "
                f"Consider increasing --n-restarts or restricting h-range."
            )
            logger.error("  ❌ %s", warnings[-1])
        elif np.mean(anchor_de_gaps > 0.20) > 0.5:
            warnings.append(
                f">{50}% of anchor points have ΔE/gap > 20%. Predictions may be degraded."
            )
            logger.warning("  ⚠️ %s", warnings[-1])

        # ── Step 4: Train MPNN (if no zoo model) ──────────────────────
        t_mpnn_start = time.perf_counter()
        if self._model is None:
            self._model = self._train_mpnn(
                h_values[anchor_idx], anchor_theta, p_layers, seed
            )
            model_source = "trained"
        t_mpnn = time.perf_counter() - t_mpnn_start

        # ── Step 5: Predict θ for target points ───────────────────────
        t_predict_start = time.perf_counter()
        if len(target_idx) > 0:
            target_theta = self._predict_theta(h_values[target_idx], p_layers)
        else:
            target_theta = np.empty((0, self._n_params))
        t_predict = time.perf_counter() - t_predict_start

        # ── Step 6: Evaluate and refine (P4) + active learning ────────
        t_refine_start = time.perf_counter()
        if len(target_idx) > 0:
            target_theta, target_energies, n_refined, methods_target = self._evaluate_and_refine(
                h_values[target_idx], target_theta, e_exact[target_idx],
                gaps[target_idx], p_layers, seed
            )
        else:
            target_energies = np.empty(0)
            n_refined = 0
            methods_target = []
        t_refine = time.perf_counter() - t_refine_start

        # ── Assemble final results (all points in original order) ─────
        theta_all = np.zeros((len(h_values), self._n_params))
        energies_all = np.zeros(len(h_values))
        methods_all = [""] * len(h_values)

        for i, idx in enumerate(anchor_idx):
            theta_all[idx] = anchor_theta[i]
            energies_all[idx] = anchor_energies[i]
            methods_all[idx] = "vqe_full"

        for i, idx in enumerate(target_idx):
            theta_all[idx] = target_theta[i]
            energies_all[idx] = target_energies[i]
            methods_all[idx] = methods_target[i]

        de_gaps = np.abs(energies_all - e_exact) / np.maximum(gaps, 1e-10)
        t_total = time.perf_counter() - t_total_start

        mpnn_mse = 0.0
        if hasattr(self, "_train_metrics") and self._train_metrics:
            mpnn_mse = self._train_metrics.get("final_mse", 0.0)

        # ── Auto-export to zoo if quality is good (Table 2) ───────────
        if model_source == "trained" and float(np.mean(de_gaps < 0.05)) > 0.80:
            self._auto_export_to_zoo(p_layers, float(np.mean(de_gaps < 0.05)),
                                     h_values, anchor_theta, seed)

        return AcceleratedResult(
            h_values=h_values,
            theta_opt=theta_all,
            energies=energies_all,
            de_gaps=de_gaps,
            gaps=gaps,
            e_exact=e_exact,
            method=methods_all,
            time_total_s=t_total,
            time_anchor_vqe_s=t_anchor,
            time_mpnn_train_s=t_mpnn,
            time_predict_s=t_predict,
            time_refine_s=t_refine,
            n_anchors=len(anchor_idx),
            n_predicted=len(target_idx),
            n_refined=n_refined,
            mpnn_mse=mpnn_mse,
            model_source=model_source,
            convergence_warnings=warnings,
        )


    # ── Private methods ──────────────────────────────────────────────────

    def _compute_ground_truth(self, h_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute exact energies and gaps for all h-points.

        Uses GroundTruthCache for disk-persistent caching across sessions.
        Avoids redundant DMRG/ED computations when the same (topology, N, h)
        was already solved in a previous run.
        """
        from qmbp_simulation import make_lattice
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        gt_cache = GroundTruthCache()
        model_name = self.spec.name if hasattr(self.spec, "name") else "tfim"

        e_exact, gaps = [], []
        n_hits, n_misses = 0, 0
        for h in h_values:
            # Check disk cache first
            cached = gt_cache.get(self._topology, self._N, model_name, float(h))
            if cached is not None:
                e_exact.append(cached["energy"])
                gaps.append(cached["gap"])
                n_hits += 1
            else:
                lat = make_lattice(self._topology, self._N, J=1.0, h=float(h))
                H = self.spec.build_hamiltonian(lat, **self.spec.hamiltonian_kwargs)
                gt = self.solver.solve(H, lat)
                e_exact.append(gt.ground_energy)
                gaps.append(gt.gap)
                # Persist for cross-session reuse
                gt_cache.put_from_result(
                    self._topology, self._N, model_name, float(h), gt
                )
                n_misses += 1

        if n_hits > 0:
            logger.info(
                "  GT cache: %d/%d hits (saved ~%.0fs)",
                n_hits, len(h_values), n_hits * 0.5,  # ~0.5s per ED for small N
            )
        return np.array(e_exact), np.array(gaps)

    def _estimate_regime_boundary(self, p_layers: int) -> float:
        """Estimate h_min where VQE converges for this topology+p (P3).

        Tries the QualityPredictor first (uses historical data). Falls back
        to canonical preflight regime boundaries.
        """
        # Try QualityPredictor (P3) — uses historical ResultIndex data
        try:
            from qmbp_simulation.analysis.quality_predictor import QualityPredictor
            predictor = QualityPredictor()
            report = predictor.predict(
                model=self.spec.name if hasattr(self.spec, "name") else "tfim",
                topology=self._topology,
                n_qubits=self._N,
                p_layers=p_layers,
                h_min=0.5,
                h_max=3.5,
            )
            if report.estimated_h_min > 0:
                return report.estimated_h_min
        except (ImportError, Exception):
            pass

        # Fallback: canonical regime boundaries from preflight.py
        try:
            from qmbp_simulation.framework.preflight import get_regime_threshold
            h_safe = get_regime_threshold(self._topology, self._N, p_layers)
            if h_safe > 0:
                return h_safe
        except (ImportError, ValueError):
            pass

        # Generic fallback: coordination-based estimate (no data available)
        z_max = max(
            len([e for e in self.lattice.edges if q in e])
            for q in range(self._N)
        ) if self.lattice.edges else 2
        base = 1.3 + 0.4 * max(0, z_max - 2)
        return max(0.8, base - 0.3 * (p_layers - 1))

    def _try_load_from_zoo(self, p_layers: int):
        """Try loading a pre-trained model from the model zoo (P2).

        Delegates to the canonical model_zoo module which uses a manifest.json
        registry for structured checkpoint management.
        """
        try:
            from qmbp_simulation.predictors.model_zoo import load_pretrained
            model, entry = load_pretrained(
                model=self.spec.name if hasattr(self.spec, "name") else "tfim",
                topology=self._topology,
                n_qubits=self._N,
                p_layers=p_layers,
            )
            logger.info("  P2 Zoo: loaded %s (pass_rate=%.0f%%)",
                        entry.checkpoint_file, entry.pass_rate * 100)
            return model
        except (FileNotFoundError, ImportError):
            # No matching model in zoo — will train from scratch
            return None

    def _select_anchors(self, h_values: np.ndarray, gaps: np.ndarray) -> np.ndarray:
        """Select K anchor points for full VQE.

        Strategy: non-uniform spacing weighted toward h_critical (where the
        landscape is hardest), ensuring endpoints are always included.
        Uses generate_nonuniform_h_grid logic for anchor placement.
        """
        K = min(self.config.n_anchors, len(h_values))
        if K >= len(h_values):
            return np.arange(len(h_values))

        if self.config.anchor_strategy == "endpoints_plus_center":
            idx = [0, len(h_values) - 1]
            remaining = K - 2
            if remaining > 0:
                interior = np.linspace(1, len(h_values) - 2, remaining, dtype=int)
                idx.extend(interior.tolist())
            return np.unique(idx)

        # Default: non-uniform anchoring — denser near h_critical
        # Use gap information: smaller gap = harder point = needs anchor
        h_min, h_max = float(h_values[-1]), float(h_values[0])  # Descending
        h_critical = 1.0
        if hasattr(self.spec, "name"):
            _H_CRIT = {"tfim": 1.0, "tfim_longitudinal": 1.0, "tfim_frustrated": 0.8}
            h_critical = _H_CRIT.get(self.spec.name, 1.0)

        # Generate K anchor h-values using non-uniform grid logic
        from qmbp_simulation.pipeline.dataset_io import generate_nonuniform_h_grid

        anchor_h = generate_nonuniform_h_grid(
            h_min=h_min, h_max=h_max, n_points=K,
            h_critical=h_critical, dense_fraction=0.5, dense_radius=0.4,
        )

        # Map each anchor h-value to the nearest index in h_values
        indices = set()
        for ah in anchor_h:
            idx = int(np.argmin(np.abs(h_values - ah)))
            indices.add(idx)

        # Always include endpoints
        indices.add(0)
        indices.add(len(h_values) - 1)

        # If we have too few (due to deduplication), pad with uniform
        if len(indices) < K:
            uniform = np.linspace(0, len(h_values) - 1, K, dtype=int)
            for u in uniform:
                if len(indices) >= K:
                    break
                indices.add(int(u))

        return np.array(sorted(indices)[:K])

    def _run_anchor_vqe(
        self, h_values, anchor_idx, e_exact, gaps, seed, p_layers
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run full VQE at anchor points with descending warm-start.

        Uses AdaptiveRestartConfig (Finding F2) to allocate more restarts
        near h_critical and fewer at easy points. Includes variational
        principle validation (Table 3) and NaN guard.

        When ``config.force_method`` is set, overrides the VQEConfig method
        to bypass the COBYLA_AUTO_SWITCH_THRESHOLD (useful for high-dim
        bond-resolved circuits where COBYLA underperforms).

        When ``config.bidirectional_anchors`` is True, runs a selective
        ascending merge on failing points (ΔE/gap > 5%) using
        ``select_suspicious_points`` from sweep_strategies.
        """
        from qmbp_simulation import VQEConfig, VQEOptimizer, make_lattice
        from qmbp_simulation.optimizers.sweep_strategies import (
            AdaptiveRestartConfig,
            SelectiveAscendingConfig,
            compute_adaptive_restarts,
            select_suspicious_points,
        )
        from qmbp_simulation.utils.helpers import canonicalize_theta

        cfg = self.config

        # Determine h_critical for adaptive restarts
        h_critical = 1.0  # Default TFIM
        if hasattr(self.spec, "name"):
            _H_CRIT_MAP = {"tfim": 1.0, "tfim_longitudinal": 1.0, "tfim_frustrated": 0.8,
                           "heisenberg_transverse": 2.5, "heisenberg": 0.0}
            h_critical = _H_CRIT_MAP.get(self.spec.name, 1.0)

        adaptive_cfg = AdaptiveRestartConfig(
            base_restarts=max(2, cfg.n_restarts // 3),
            max_restarts=cfg.n_restarts,
            critical_restarts=cfg.n_restarts,
            h_critical=h_critical,
            critical_radius=0.3,
        )

        rng = np.random.default_rng(seed)
        prev_theta = rng.uniform(-0.01, 0.01, self._n_params)

        # Sort anchors descending for warm-start
        h_anchors_sorted = sorted(
            [(h_values[i], i) for i in anchor_idx], reverse=True
        )

        results_map: dict[int, tuple[np.ndarray, float]] = {}
        n_violations = 0
        prev_de_gap: float | None = None

        for h, orig_idx in h_anchors_sorted:
            # Adaptive restarts: allocate based on neighbor difficulty + h_critical
            n_restarts = compute_adaptive_restarts(
                float(h), prev_de_gap=prev_de_gap, config=adaptive_cfg
            )

            vqe_config = VQEConfig(
                p_layers=p_layers, n_restarts=n_restarts, maxiter=cfg.maxiter,
                method=cfg.force_method if cfg.force_method else "L-BFGS-B",
            )
            optimizer = VQEOptimizer(config=vqe_config, backend=self.backend, seed=seed)

            lat = make_lattice(self._topology, self._N, J=1.0, h=float(h))
            H = self.spec.build_hamiltonian(lat, **self.spec.hamiltonian_kwargs)
            # Update CachedBackend h for correct cache key generation
            if hasattr(self.backend, "set_h"):
                self.backend.set_h(float(h))
            result = optimizer.optimize(H, self.circuit, initial_guess=prev_theta)

            # NaN guard (Table 3): never propagate NaN through warm-start
            if np.all(np.isfinite(result.theta_opt)):
                prev_theta = result.theta_opt.copy()
            else:
                logger.warning("  ⚠️ NaN in VQE result at h=%.4f, keeping previous θ", h)

            # Variational principle check (Table 3)
            if result.energy < e_exact[orig_idx] - 1e-6:
                n_violations += 1
                violation = e_exact[orig_idx] - result.energy
                if violation > 1e-2:
                    logger.warning(
                        "  ⚠️ Variational violation at h=%.4f: E_vqe=%.6f < E_exact=%.6f (Δ=%.2e)",
                        h, result.energy, e_exact[orig_idx], violation,
                    )

            # Track ΔE/gap for adaptive restarts of next point (use actual gap)
            point_gap = gaps[orig_idx] if gaps[orig_idx] > 1e-10 else 1e-10
            prev_de_gap = abs(result.energy - e_exact[orig_idx]) / point_gap

            theta_canon = canonicalize_theta(prev_theta.copy())
            results_map[orig_idx] = (theta_canon, result.energy)

        if n_violations > len(anchor_idx) // 2:
            logger.warning(
                "  ⚠️ %d/%d anchor points violate variational principle. "
                "E_exact reference may be approximate (e.g., DMRG on 2D topology).",
                n_violations, len(anchor_idx),
            )

        # ── Bidirectional ascending merge (selective) ─────────────────
        # Re-optimize failing anchors in ascending direction using existing
        # select_suspicious_points() from sweep_strategies.
        if cfg.bidirectional_anchors:
            # Build per-anchor result dicts for select_suspicious_points
            desc_results = []
            for orig_idx in anchor_idx:
                theta_desc, e_desc = results_map[orig_idx]
                point_gap = gaps[orig_idx] if gaps[orig_idx] > 1e-10 else 1e-10
                de_gap = abs(e_desc - e_exact[orig_idx]) / point_gap
                desc_results.append({"h": float(h_values[orig_idx]), "de_gap": float(de_gap)})

            asc_config = SelectiveAscendingConfig(
                de_gap_threshold=0.05,
                include_neighbors=True,
                max_fraction=0.6,
            )
            suspicious_indices, asc_report = select_suspicious_points(
                desc_results, config=asc_config
            )

            if suspicious_indices and not asc_report.fell_back_to_full:
                logger.info(
                    "  🔄 Bidirectional: %d/%d anchor points targeted for ascending merge",
                    len(suspicious_indices), len(anchor_idx),
                )

                # Run ascending pass only for suspicious anchors
                # Sort them ascending (h_min → h_max) for warm-start propagation
                h_anchors_ascending = sorted(
                    [(h_values[anchor_idx[i]], anchor_idx[i], i) for i in suspicious_indices]
                )

                # Seed ascending warm-start from the best neighbor above
                prev_theta_asc = rng.uniform(-0.01, 0.01, self._n_params)
                # Find the lowest-h anchor that passed — use its θ as ascending seed
                for r in desc_results:
                    if r["de_gap"] <= 0.05:
                        # Find its index and θ
                        for orig_idx in anchor_idx:
                            if abs(h_values[orig_idx] - r["h"]) < 1e-6:
                                prev_theta_asc = results_map[orig_idx][0].copy()
                                break
                        break

                n_improved_asc = 0
                for h, orig_idx, _ in h_anchors_ascending:
                    n_restarts = compute_adaptive_restarts(
                        float(h), prev_de_gap=None, config=adaptive_cfg
                    )
                    vqe_config = VQEConfig(
                        p_layers=p_layers, n_restarts=n_restarts, maxiter=cfg.maxiter,
                        method=cfg.force_method if cfg.force_method else "L-BFGS-B",
                    )
                    optimizer = VQEOptimizer(
                        config=vqe_config, backend=self.backend, seed=seed + 999
                    )

                    lat = make_lattice(self._topology, self._N, J=1.0, h=float(h))
                    H = self.spec.build_hamiltonian(lat, **self.spec.hamiltonian_kwargs)
                    if hasattr(self.backend, "set_h"):
                        self.backend.set_h(float(h))
                    result_asc = optimizer.optimize(
                        H, self.circuit, initial_guess=prev_theta_asc
                    )

                    # Keep ascending θ if energy improved
                    _, e_desc = results_map[orig_idx]
                    if result_asc.energy < e_desc - 1e-10:
                        theta_asc_canon = canonicalize_theta(result_asc.theta_opt.copy())
                        results_map[orig_idx] = (theta_asc_canon, result_asc.energy)
                        n_improved_asc += 1

                    # Propagate warm-start ascending
                    if np.all(np.isfinite(result_asc.theta_opt)):
                        prev_theta_asc = result_asc.theta_opt.copy()

                logger.info(
                    "  🔄 Bidirectional merge: %d/%d targeted points improved",
                    n_improved_asc, len(suspicious_indices),
                )
            elif asc_report.fell_back_to_full:
                logger.info(
                    "  🔄 Bidirectional: >60%% suspicious — running full ascending pass"
                )
                # Full ascending: iterate all anchors in ascending h order
                h_anchors_ascending = sorted(
                    [(h_values[i], i) for i in anchor_idx]
                )
                prev_theta_asc = rng.uniform(-0.01, 0.01, self._n_params)
                n_improved_asc = 0
                for h, orig_idx in h_anchors_ascending:
                    n_restarts = compute_adaptive_restarts(
                        float(h), prev_de_gap=None, config=adaptive_cfg
                    )
                    vqe_config = VQEConfig(
                        p_layers=p_layers, n_restarts=n_restarts, maxiter=cfg.maxiter,
                        method=cfg.force_method if cfg.force_method else "L-BFGS-B",
                    )
                    optimizer = VQEOptimizer(
                        config=vqe_config, backend=self.backend, seed=seed + 999
                    )
                    lat = make_lattice(self._topology, self._N, J=1.0, h=float(h))
                    H = self.spec.build_hamiltonian(lat, **self.spec.hamiltonian_kwargs)
                    if hasattr(self.backend, "set_h"):
                        self.backend.set_h(float(h))
                    result_asc = optimizer.optimize(
                        H, self.circuit, initial_guess=prev_theta_asc
                    )

                    _, e_desc = results_map[orig_idx]
                    if result_asc.energy < e_desc - 1e-10:
                        theta_asc_canon = canonicalize_theta(result_asc.theta_opt.copy())
                        results_map[orig_idx] = (theta_asc_canon, result_asc.energy)
                        n_improved_asc += 1

                    if np.all(np.isfinite(result_asc.theta_opt)):
                        prev_theta_asc = result_asc.theta_opt.copy()

                logger.info(
                    "  🔄 Full ascending merge: %d/%d points improved",
                    n_improved_asc, len(anchor_idx),
                )

        # Return in anchor_idx order
        theta_out = np.array([results_map[i][0] for i in anchor_idx])
        energy_out = np.array([results_map[i][1] for i in anchor_idx])
        return theta_out, energy_out

    def _train_mpnn(self, h_anchor, theta_anchor, p_layers, seed):
        """Train UnifiedMPNN on anchor data.

        Applies canonicalization + basin filtering (Table 1) before training
        to ensure the MPNN only sees consistent, high-quality targets.
        """
        from qmbp_simulation.predictors.unified_graph import build_unified_bond_resolved_graph
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN, train_unified_mpnn
        from qmbp_simulation.utils.helpers import filter_consistent_theta

        cfg = self.config

        # Basin filter (Table 1): remove outlier θ that are in different local minima
        h_orig, theta_orig = h_anchor.copy(), theta_anchor.copy()
        _, basin_mask = filter_consistent_theta(theta_anchor)
        n_filtered = int((~basin_mask).sum())
        if n_filtered > 0:
            logger.info("  Basin filter: removed %d/%d inconsistent anchor points",
                        n_filtered, len(theta_anchor))
            h_anchor = h_anchor[basin_mask]
            theta_anchor = theta_anchor[basin_mask]

        # Revert if too few points remain after filtering
        if len(h_anchor) < 3:
            logger.warning(
                "  ⚠️ Only %d points after basin filter (need ≥3). "
                "Reverting to unfiltered data.", len(h_anchor)
            )
            h_anchor = h_orig
            theta_anchor = theta_orig

        dataset = []
        for i, h in enumerate(h_anchor):
            g = build_unified_bond_resolved_graph(
                self.lattice, h_value=float(h), p_layers=p_layers,
                theta_opt=theta_anchor[i], include_circuit_nodes=True,
            )
            dataset.append(g)

        model = UnifiedMPNN(
            node_features=4,
            hidden_dim=cfg.mpnn_hidden_dim,
            n_layers=3,
            type_embedding_dim=cfg.mpnn_type_embedding_dim,
        )
        self._train_metrics = train_unified_mpnn(
            model, dataset,
            n_epochs=cfg.mpnn_epochs,
            val_fraction=0.0,
            seed=seed,
            mse_floor=1e-5,  # Stop early if already excellent — saves up to 1000 epochs
        )
        logger.info("  MPNN trained: MSE=%.2e (%d epochs, %d points)",
                    self._train_metrics["final_mse"], self._train_metrics["n_epochs_run"],
                    len(dataset))
        return model

    def _predict_theta(self, h_target, p_layers) -> np.ndarray:
        """Predict θ for target points using trained/loaded MPNN.

        Includes ThetaValidator bounds check (L1-L3), NaN guard (Table 3),
        and logs per-point confidence scores for downstream decisions.
        """
        import torch
        from qmbp_simulation.predictors.unified_graph import build_unified_bond_resolved_graph

        if self._model is None:
            raise RuntimeError("No MPNN model available. Train or load from zoo first.")

        self._model.eval()
        predictions = []

        # Initialize ThetaValidator from anchor data if available
        theta_validator = None
        if hasattr(self, "_anchor_theta") and self._anchor_theta is not None:
            try:
                from qmbp_simulation.analysis.theta_validator import ThetaValidator

                theta_validator = ThetaValidator.from_training_data(
                    theta_opt=self._anchor_theta,
                    h_values=self._anchor_h,
                )
            except Exception:
                pass  # Non-fatal: skip validation if data is insufficient

        n_low_confidence = 0
        for h in h_target:
            g = build_unified_bond_resolved_graph(
                self.lattice, h_value=float(h), p_layers=p_layers,
                include_circuit_nodes=True,
            )
            with torch.no_grad():
                pred = self._model(g).numpy().flatten()

            # NaN/Inf guard (Table 3)
            if not np.all(np.isfinite(pred)):
                n_bad = int(np.sum(~np.isfinite(pred)))
                logger.warning(
                    "  ⚠️ MPNN prediction has %d NaN/Inf values at h=%.4f. "
                    "Replacing with zeros.", n_bad, h,
                )
                pred = np.where(np.isfinite(pred), pred, 0.0)

            # Bounds clipping to valid HVA range [-π, π]
            pred = np.clip(pred, -np.pi, np.pi)

            # ThetaValidator L1-L3 check (bounds, NaN, interpolation)
            if theta_validator is not None:
                try:
                    report = theta_validator.validate(pred, level=3, h_test=float(h))
                    if not report.passes():
                        n_low_confidence += 1
                        logger.debug(
                            "  θ_pred low confidence at h=%.4f (score=%.2f)",
                            h, report.confidence_score,
                        )
                except Exception:
                    pass  # Non-fatal

            predictions.append(pred)

        if n_low_confidence > 0:
            logger.info(
                "  ThetaValidator: %d/%d predictions with low confidence",
                n_low_confidence, len(h_target),
            )

        return np.array(predictions)

    def _evaluate_and_refine(
        self, h_target, theta_pred, e_exact_target, gaps_target, p_layers, seed
    ) -> tuple[np.ndarray, np.ndarray, int, list[str]]:
        """Evaluate predictions and refine uncertain ones (P4).

        Also runs active learning rounds if configured.

        Returns (theta_final, energies, n_refined, methods).
        """
        from qmbp_simulation import VQEConfig, VQEOptimizer, make_lattice

        cfg = self.config
        n_points = len(h_target)
        theta_final = theta_pred.copy()
        energies = np.zeros(n_points)
        methods = ["mpnn_direct"] * n_points
        n_refined = 0

        # Evaluate all predictions
        for i, h in enumerate(h_target):
            lat = make_lattice(self._topology, self._N, J=1.0, h=float(h))
            H = self.spec.build_hamiltonian(lat, **self.spec.hamiltonian_kwargs)
            if hasattr(self.backend, "set_h"):
                self.backend.set_h(float(h))
            energies[i] = self.backend.evaluate(self.circuit, H, theta_pred[i])

        # ── Refinement pass: VQE warm-start for uncertain points ──────
        de_gaps = np.abs(energies - e_exact_target) / np.maximum(gaps_target, 1e-10)
        needs_refine = de_gaps > cfg.refine_threshold

        if needs_refine.any() and cfg.refine_restarts > 0:
            n_to_refine = int(needs_refine.sum())
            logger.info("  P4 Refine: %d/%d points with ΔE/gap > %.0f%%",
                        n_to_refine, n_points, cfg.refine_threshold * 100)

            ws_config = VQEConfig(
                p_layers=p_layers,
                n_restarts=cfg.refine_restarts,
                maxiter=cfg.refine_maxiter,
            )
            ws_optimizer = VQEOptimizer(
                config=ws_config, backend=self.backend, seed=seed + 500
            )

            for i in np.where(needs_refine)[0]:
                h = h_target[i]
                lat = make_lattice(self._topology, self._N, J=1.0, h=float(h))
                H = self.spec.build_hamiltonian(lat, **self.spec.hamiltonian_kwargs)
                if hasattr(self.backend, "set_h"):
                    self.backend.set_h(float(h))
                result = ws_optimizer.optimize(
                    H, self.circuit, initial_guess=theta_pred[i]
                )
                # Only count as refined if energy actually improved (fix issue #2)
                if result.energy < energies[i] - 1e-10:
                    theta_final[i] = result.theta_opt
                    energies[i] = result.energy
                    methods[i] = "mpnn_refined"
                    n_refined += 1

        # ── Active learning rounds (P4 extension) ─────────────────────
        # Each round: identify highest-uncertainty points, run VQE there
        for al_round in range(cfg.active_learning_rounds):
            # Re-evaluate de_gaps with current energies
            de_gaps = np.abs(energies - e_exact_target) / np.maximum(gaps_target, 1e-10)

            # Identify top-3 worst points that haven't been VQE'd yet
            mpnn_only_mask = np.array([m == "mpnn_direct" for m in methods])
            if not mpnn_only_mask.any():
                break  # All points already refined

            mpnn_de_gaps = np.where(mpnn_only_mask, de_gaps, 0)
            worst_indices = np.argsort(mpnn_de_gaps)[::-1][:3]
            worst_indices = [i for i in worst_indices if mpnn_only_mask[i] and de_gaps[i] > 0.05]

            if not worst_indices:
                break  # No uncertain points remaining

            logger.info("  P4 Active round %d: refining %d highest-uncertainty points",
                        al_round + 1, len(worst_indices))

            ws_config_al = VQEConfig(
                p_layers=p_layers, n_restarts=cfg.refine_restarts * 2, maxiter=cfg.refine_maxiter,
            )
            al_optimizer = VQEOptimizer(config=ws_config_al, backend=self.backend, seed=seed + 600 + al_round)

            for i in worst_indices:
                h = h_target[i]
                lat = make_lattice(self._topology, self._N, J=1.0, h=float(h))
                H = self.spec.build_hamiltonian(lat, **self.spec.hamiltonian_kwargs)
                if hasattr(self.backend, "set_h"):
                    self.backend.set_h(float(h))
                result = al_optimizer.optimize(H, self.circuit, initial_guess=theta_final[i])
                if result.energy < energies[i] - 1e-10:
                    theta_final[i] = result.theta_opt
                    energies[i] = result.energy
                    methods[i] = "vqe_refined"
                    n_refined += 1

        return theta_final, energies, n_refined, methods

    # ── Public utilities ─────────────────────────────────────────────────

    def save_model(self, path: str) -> None:
        """Save the trained MPNN to a file (for zoo or reuse)."""
        if self._model is None:
            raise RuntimeError("No model to save. Run the pipeline first.")
        from qmbp_simulation.predictors.unified_mpnn import save_unified_checkpoint
        save_unified_checkpoint(self._model, path, training_metadata={
            "topology": self._topology,
            "n_qubits": self._N,
            "n_params": self._n_params,
            "mpnn_mse": self._train_metrics.get("final_mse", 0) if hasattr(self, "_train_metrics") else 0,
        })

    def get_model(self):
        """Access the trained/loaded MPNN model."""
        return self._model

    def _auto_export_to_zoo(
        self, p_layers: int, pass_rate: float,
        h_values: np.ndarray, anchor_theta: np.ndarray, seed: int,
    ) -> None:
        """Auto-register the trained model to the zoo if quality is good (Table 2).

        Only exports if pass_rate > 80%. This ensures the zoo only contains
        high-quality models that can be trusted for zero-shot inference.
        Also auto-generates a YAML preset for immediate --preset usage.
        """
        try:
            from datetime import datetime, timezone
            from qmbp_simulation.predictors.model_zoo import ZooEntry, register_checkpoint

            model_name = self.spec.name if hasattr(self.spec, "name") else "tfim"
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            filename = f"unified_{model_name}_{self._topology}_n{self._N}_p{p_layers}_{timestamp}.pt"

            entry = ZooEntry(
                model=model_name,
                topology=self._topology,
                n_qubits=self._N,
                p_layers=p_layers,
                checkpoint_file=filename,
                h_range=(float(h_values.min()), float(h_values.max())),
                pass_rate=pass_rate,
                n_training_points=len(anchor_theta),
                seeds=[seed],
                created=timestamp,
                notes=f"Auto-exported by AcceleratedVQE (pass_rate={pass_rate:.0%})",
            )
            register_checkpoint(self._model, entry, overwrite=True)
            logger.info("  Zoo: auto-exported model (pass_rate=%.0f%%)", pass_rate * 100)

            # Auto-generate YAML preset for this validated config
            self._auto_generate_preset(model_name, p_layers, h_values, pass_rate)

        except Exception as e:
            logger.debug("  Zoo auto-export failed (non-blocking): %s", e)

    def _auto_generate_preset(
        self, model_name: str, p_layers: int,
        h_values: np.ndarray, pass_rate: float,
    ) -> None:
        """Generate a YAML preset matching this successful config.

        Integrates with the presets system so `--preset noiseless/<name>` works
        immediately after a successful accelerated run.
        """
        try:
            from pathlib import Path

            presets_dir = Path(__file__).resolve().parents[3] / "configs" / "presets" / "noiseless"
            presets_dir.mkdir(parents=True, exist_ok=True)

            preset_name = f"{model_name}_{self._topology}_n{self._N}_p{p_layers}"
            preset_path = presets_dir / f"{preset_name}.yaml"

            if preset_path.exists():
                return  # Don't overwrite existing presets

            h_min, h_max = float(h_values.min()), float(h_values.max())
            h_points = len(h_values)

            yaml_content = (
                f"# {model_name} {self._topology} N={self._N} p={p_layers}\n"
                f"# Pass rate: {pass_rate:.0%} (auto-generated by AcceleratedVQE)\n"
                f"runner: noiseless_pipeline\n"
                f"model: {model_name}\n"
                f"topology: {self._topology}\n"
                f"n_qubits: {self._N}\n"
                f"p_layers: {p_layers}\n"
                f"h_min: {h_min}\n"
                f"h_max: {h_max}\n"
                f"h_points: {h_points}\n"
                f"maxiter: {self.config.maxiter}\n"
                f"n_restarts: {self.config.n_restarts}\n"
                f"seeds: [42, 43, 44]\n"
                f'description: "{model_name} {self._topology} N={self._N} p={p_layers} '
                f'— {pass_rate:.0%} validated (AcceleratedVQE)"\n'
            )
            preset_path.write_text(yaml_content)
            logger.info("  Preset: auto-generated %s", preset_name)
        except Exception as e:
            logger.debug("  Preset auto-generation failed: %s", e)
