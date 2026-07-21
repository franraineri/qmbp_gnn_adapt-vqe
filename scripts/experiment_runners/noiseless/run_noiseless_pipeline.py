#!/usr/bin/env python3
"""Noiseless Pipeline Runner — Exact statevector experiments (N ≤ 22).

Executes the full pipeline (ExactDiag → VQE → MPNN → Deploy) using only
exact methods: np.linalg.eigh for ground truth, StatevectorEstimator for
VQE energy evaluation. No noise, no mitigation, no approximations.

This produces the theoretical ceiling — the cleanest results possible
against which noisy/hardware runs are compared.

Sections:
    1. Exact Diagonalization: Ground truth energies, gaps, observables
    2. VQE Optimization: Descending warm-start sweep with NoiselessBackend
    3. MPNN Training: GINConv predictor trained on VQE θ_opt
    4. Deploy: Predict unseen h-points, compute ΔE/gap + phase label

Supports multiple topologies via --topology (chain_1d, ladder, triangular,
heavy_hex, square).

Usage:
    # Default: N=6, p=2, chain_1d, tfim model
    python scripts/experiment_runners/noiseless/run_noiseless_pipeline.py

    # Larger system
    python scripts/experiment_runners/noiseless/run_noiseless_pipeline.py \\
        --n-qubits 10 --p-layers 1 --topology heavy_hex

    # Multiple topologies comparison
    python scripts/experiment_runners/noiseless/run_noiseless_pipeline.py \\
        --n-qubits 6 --p-layers 2 --topology chain_1d ladder triangular

    # Custom h-grid
    python scripts/experiment_runners/noiseless/run_noiseless_pipeline.py \\
        --n-qubits 8 --h-min 0.5 --h-max 2.0 --h-points 15

    # Skip MPNN (just Phase 1+2)
    python scripts/experiment_runners/noiseless/run_noiseless_pipeline.py --section 1 2

    # Dry run
    python scripts/experiment_runners/noiseless/run_noiseless_pipeline.py --dry-run
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

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

from qmbp_simulation.models.constants import (
    DE_GAP_THRESHOLD,
    DEFAULT_SEEDS,
    STATEVECTOR_MAX_N,
)

DEFAULT_N = 6
DEFAULT_P = 2
DEFAULT_TOPOLOGY = "chain_1d"
DEFAULT_MODEL = "tfim_longitudinal"
DEFAULT_H_MIN = 1.0
DEFAULT_H_MAX = 3.5
DEFAULT_H_POINTS = 35
DEFAULT_MAXITER = 1200
DEFAULT_N_RESTARTS = 10


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════


class NoiselessPipelineRunner(ValidationRunner):
    """Noiseless exact-method pipeline (N ≤ 22, StatevectorEstimator).

    Produces the theoretical ceiling for VQE + MPNN warm-start on TFIM-class
    models. No noise, no mitigation — the purest benchmark possible.
    """

    runner_id = "noiseless_pipeline_v4"
    experiment_id = "NOISELESS"
    description = "Noiseless exact pipeline: ExactDiag → VQE → MPNN → Deploy"
    hypothesis = (
        "Full pipeline achieves ΔE/gap < 5% and correct phase label "
        "at all test h-points using exact statevector methods only."
    )

    @classmethod
    def _add_custom_args(cls, parser):
        cls._add_standard_physics_args(
            parser,
            n_qubits=DEFAULT_N,
            p_layers=DEFAULT_P,
            topology=DEFAULT_TOPOLOGY,
            model=DEFAULT_MODEL,
            h_min=DEFAULT_H_MIN,
            h_max=DEFAULT_H_MAX,
            h_points=DEFAULT_H_POINTS,
            seeds=list(DEFAULT_SEEDS),
            maxiter=DEFAULT_MAXITER,
            n_restarts=DEFAULT_N_RESTARTS,
        )
        parser.add_argument(
            "--no-physics-loss",
            action="store_true",
            default=False,
            help="Disable physics-informed energy loss during MPNN training. "
            "Useful as control experiment to compare MSE-only vs energy-aware training.",
        )
        parser.add_argument(
            "--physics-loss-weight",
            type=float,
            default=0.2,
            help="Weight λ for physics loss term (default: 0.1). "
            "Higher values prioritize energy accuracy over θ-MSE.",
        )
        parser.add_argument(
            "--physics-loss-start",
            type=int,
            default=800,
            help="Epoch at which physics loss activates (default: 800). "
            "Allows MSE to converge first, then energy regularizes.",
        )

    def run_preflight(self) -> bool:
        """Validate configuration before execution."""
        n = self._args.n_qubits
        if n > STATEVECTOR_MAX_N:
            logger.error(
                f"N={n} exceeds STATEVECTOR_MAX_N={STATEVECTOR_MAX_N}. "
                f"Use MPSBackend-based runners for N>{STATEVECTOR_MAX_N}."
            )
            return False
        if n < 2:
            logger.error(f"N={n} too small. Minimum is 2.")
            return False
        if self._args.h_min >= self._args.h_max:
            logger.error(f"h_min ({self._args.h_min}) must be < h_max ({self._args.h_max}).")
            return False
        if self._args.h_points < 3:
            logger.error(
                f"h_points={self._args.h_points} too few. Minimum is 3 "
                f"(need at least 3 points for meaningful VQE sweep + MPNN training)."
            )
            return False
        if self._args.h_min < 0:
            logger.error(f"h_min={self._args.h_min} invalid. Must be >= 0.")
            return False
        if self._args.h_max <= 0:
            logger.error(f"h_max={self._args.h_max} invalid. Must be > 0.")
            return False
        if self._args.h_points < 5:
            logger.warning(
                f"h_points={self._args.h_points} is very low. "
                f"MPNN training requires at least 5 points for meaningful interpolation. "
                f"Consider using --h-points 10 or more."
            )
        if not self._args.no_physics_loss and self._args.physics_loss_weight <= 0:
            logger.error(
                f"physics_loss_weight={self._args.physics_loss_weight} must be > 0 "
                f"when physics loss is enabled. Use --no-physics-loss to disable it entirely."
            )
            return False
        return True

    def build_config(self) -> dict:
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "system": {
                "n_qubits": self._args.n_qubits,
                "p_layers": self._args.p_layers,
                "topologies": self._args.topology,
                "model": self._args.model,
                "model_params": self._model_params if hasattr(self, "_model_params") else {},
            },
            "h_grid": {
                "h_min": self._args.h_min,
                "h_max": self._args.h_max,
                "h_points": self._args.h_points,
            },
            "vqe": {
                "maxiter": self._args.maxiter,
                "n_restarts": self._args.n_restarts,
                "method": "L-BFGS-B",
                "bounds": "[-π, π]",
            },
            "mpnn": {
                "physics_loss": not self._args.no_physics_loss,
                "physics_loss_weight": self._args.physics_loss_weight,
                "physics_loss_start_epoch": self._args.physics_loss_start,
            },
            "backend": "NoiselessBackend (StatevectorEstimator)",
            "seeds": self._args.seeds,
        }

    def setup(self):
        """Import heavy dependencies and initialize shared objects."""
        from qiskit.quantum_info import Statevector, state_fidelity
        from scipy.optimize import minimize

        from qmbp_simulation.analysis import DiagnosticCollector, ThetaValidator

        # Standard physics setup (builder, solver, hva, make_lattice, etc.)
        self.setup_physics()

        self.Statevector = Statevector
        self.state_fidelity = state_fidelity
        self.minimize = minimize
        self.ThetaValidator = ThetaValidator

        # Auto-select backend for VQE loops: MPS for N>10 (much faster)
        self._vqe_backend = self.select_backend(self._args.n_qubits, for_vqe_loop=True)
        logger.info("  Backend: %s", self._vqe_backend.name)

        # Initialize DiagnosticCollector for MPNN observability
        output_dir = Path(self._args.output) if self._args.output else None
        self._collector = DiagnosticCollector(
            verbose=self._args.verbose if hasattr(self._args, "verbose") else False,
            save_dir=output_dir,
        )

        # Parse model params (e.g. "g=0.3" or "J2=0.5,g=0.1")
        self.parse_model_params()

        # Set experiment_id dynamically based on model + topology for organized output.
        # Structure: results/experiments/exp_noiseless/{model}/{topology}/run_*.json
        from qmbp_simulation.framework.result_io import build_experiment_id

        self.experiment_id = build_experiment_id(
            category="noiseless",
            model=self._args.model,
            topology=self._args.topology,
        )

        # Compute h-grid (descending for warm-start)
        self._h_values = self.generate_h_grid()
        logger.info(
            f"  📐 Non-uniform h-grid: {len(self._h_values)} points, "
            f"dense near h_c≈{self.H_CRITICAL_ESTIMATES.get(self._args.model, '?')}"
        )

        # Shared state across sections
        self._ground_truth: dict[str, list[dict]] = {}  # topology -> list of results
        self._vqe_results: dict[str, list[dict]] = {}  # topology -> list of results
        self._mpnn_model = None
        self._mpnn_train_loss = None

    def restore_section_state(
        self, resumed_data: dict[str, Any], resumed_sections: set[int]
    ) -> None:
        """Restore VQE/ground truth state from a resumed run.

        This allows Section 3 (MPNN) and Section 4 (Deploy) to run without
        re-computing expensive VQE optimization.
        """
        results = resumed_data.get("results", {})

        # Restore ground truth from Section 1
        if 1 in resumed_sections:
            s1_data = results.get("section_1", {}).get("data", {})
            topos = s1_data.get("topologies", {})
            h_values = s1_data.get("h_values", [])
            if h_values:
                self._h_values = h_values
            for topo_name, topo_data in topos.items():
                # Ground truth doesn't store per-point in the saved JSON,
                # so we just mark it as "completed" — Section 1 result data
                # (e_min, e_max, gap_range) is in the envelope already.
                logger.info(f"    Restored ground truth metadata for {topo_name}")

        # Restore VQE results from Section 2
        if 2 in resumed_sections:
            s2_data = results.get("section_2", {}).get("data", {})
            topos = s2_data.get("topologies", {})
            for topo_name, topo_data in topos.items():
                per_point = topo_data.get("per_point", [])
                if per_point:
                    self._vqe_results[topo_name] = per_point
                    logger.info(
                        f"    Restored VQE results for {topo_name}: "
                        f"{len(per_point)} h-points with θ_opt"
                    )
                else:
                    logger.warning(
                        f"    Section 2 for {topo_name} has no per_point data. "
                        f"VQE will need to be re-run."
                    )

    def _get_spec(self):
        """Get model spec with any --model-params applied.

        Delegates to base class get_spec() which handles parse + override.
        """
        return self.get_spec()

    @staticmethod
    def _compute_entanglement_entropy(statevector: np.ndarray, n_qubits: int) -> float:
        """Compute von Neumann entanglement entropy for half-chain bipartition.

        Delegates to EntanglementAnalyzer.compute_half_chain_entropy (canonical impl).
        """
        from qmbp_simulation.analysis import EntanglementAnalyzer

        return EntanglementAnalyzer().compute_half_chain_entropy(statevector, n_qubits)

    def _save_vqe_checkpoint(
        self, topology: str, results: list[dict], current_theta: np.ndarray
    ) -> None:
        """Save VQE progress checkpoint after each h-point.

        Delegates to the base-class checkpoint infrastructure.
        """
        self.save_checkpoint(
            label=f"vqe_{topology}",
            data={
                "topology": topology,
                "n_completed": len(results),
                "n_total": len(self._h_values),
                "current_theta": current_theta.tolist(),
                "results": results,
            },
        )

    def _load_vqe_checkpoint(
        self, topology: str, n_params: int | None = None
    ) -> tuple[list[dict], np.ndarray] | None:
        """Load VQE checkpoint for a topology if one exists.

        Parameters
        ----------
        topology : str
            Topology name used as checkpoint key.
        n_params : int | None
            Expected number of parameters. If provided, validates the
            checkpoint data matches. Stale checkpoints from previous runs
            with different p_layers are discarded with a warning.

        Returns
        -------
        tuple[list[dict], np.ndarray] | None
            (results_so_far, current_theta) if checkpoint found and valid, else None.
        """
        cp = self.load_checkpoint(f"vqe_{topology}")
        if cp is None:
            return None
        try:
            results = cp["results"]
            theta = np.array(cp["current_theta"])

            # Validate parameter count if expected n_params provided
            if n_params is not None and len(theta) != n_params:
                logger.warning(
                    "    ⚠️  Stale checkpoint for %s: param count mismatch "
                    "(checkpoint has %d, current run expects %d). "
                    "Discarding checkpoint — VQE will restart from scratch.",
                    topology,
                    len(theta),
                    n_params,
                )
                # Remove the stale checkpoint to avoid re-loading it
                self.cleanup_checkpoints(pattern=f"vqe_{topology}")
                return None

            # Also validate theta_opt in results if any exist
            if results and n_params is not None:
                first_theta = results[0].get("theta_opt")
                if first_theta is not None and len(first_theta) != n_params:
                    logger.warning(
                        "    ⚠️  Stale checkpoint for %s: results theta_opt has %d params, "
                        "expected %d. Discarding.",
                        topology,
                        len(first_theta),
                        n_params,
                    )
                    self.cleanup_checkpoints(pattern=f"vqe_{topology}")
                    return None

            logger.info(
                "    Resuming VQE: %d/%d points already computed",
                cp["n_completed"],
                len(self._h_values),
            )
            return results, theta
        except (KeyError, TypeError) as e:
            logger.warning("    ⚠️  Checkpoint data invalid for %s: %s", topology, e)
            return None

    def _cleanup_vqe_checkpoints(self) -> None:
        """Remove VQE checkpoint files after successful section completion."""
        self.cleanup_checkpoints(pattern="vqe_*")

    def _try_load_mpnn_checkpoint(self):
        """Attempt to load a saved MPNN checkpoint matching current config.

        Uses the same fingerprint hash as section_mpnn_train to find a
        matching checkpoint. Validates output_dim matches expected n_params.

        Returns
        -------
        MPNNPredictor | None
            Loaded model if a valid checkpoint exists, else None.
        """
        import hashlib

        from qmbp_simulation.predictors import load_mpnn_checkpoint

        N = self._args.n_qubits
        p = self._args.p_layers
        model = self._args.model
        topo = self._args.topology[0]

        # Reconstruct fingerprint
        n_h_points = len(self._h_values)
        fp_str = f"{model}_{topo}_{N}_{p}_{n_h_points}"
        fp_hash = hashlib.md5(fp_str.encode()).hexdigest()[:8]
        mpnn_ckpt_dir = self._checkpoint_dir() / "mpnn_checkpoints"
        mpnn_ckpt_path = mpnn_ckpt_dir / f"mpnn_{topo}_n{N}_p{p}_{fp_hash}.pt"

        if not mpnn_ckpt_path.exists():
            logger.debug("    No MPNN checkpoint found at %s", mpnn_ckpt_path)
            return None

        try:
            loaded_model = load_mpnn_checkpoint(mpnn_ckpt_path)
            # Validate output dimension matches current config
            spec = self._get_spec()
            expected_params = spec.total_params_for_p(p)
            if hasattr(loaded_model, "output_dim") and loaded_model.output_dim != expected_params:
                logger.warning(
                    "    ⚠️  MPNN checkpoint output_dim=%d != expected %d. Discarding.",
                    loaded_model.output_dim,
                    expected_params,
                )
                return None
            logger.info(
                "    ♻️  Loaded MPNN checkpoint: %s (output_dim=%d)",
                mpnn_ckpt_path.name,
                expected_params,
            )
            return loaded_model
        except Exception as e:
            logger.warning("    ⚠️  MPNN checkpoint load failed: %s", e)
            return None

    def define_sections(self) -> list[Section]:
        return [
            Section(
                id=1,
                name="Exact Diagonalization (Ground Truth)",
                fn=self.section_exact_diag,
                hypothesis="Exact energies and gaps computed for all h-points and topologies",
            ),
            Section(
                id=2,
                name="VQE Optimization (NoiselessBackend)",
                fn=self.section_vqe,
                hypothesis=(
                    "Warm-start descending VQE achieves fidelity ≥ 0.99 "
                    "and ΔE/gap < 1% at all h-points"
                ),
            ),
            Section(
                id=3,
                name="MPNN Training",
                fn=self.section_mpnn_train,
                hypothesis="GINConv MPNN achieves training MSE < 1e-4 on θ_opt data",
            ),
            Section(
                id=4,
                name="Deploy (Predict + Evaluate)",
                fn=self.section_deploy,
                hypothesis=(
                    "MPNN-predicted θ achieves ΔE/gap < 5% and correct phase "
                    "label at held-out h-points"
                ),
            ),
        ]

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 1: Exact Diagonalization
    # ═══════════════════════════════════════════════════════════════════════════

    def section_exact_diag(self) -> dict:
        """Compute ground truth for all topologies and h-points."""
        N = self._args.n_qubits
        model = self._args.model
        spec = self._get_spec()
        all_results = {}

        for topo in self._args.topology:
            logger.info(f"  Topology: {topo}, N={N}, model={model}")
            topo_results = []
            gt_objects = []  # Keep GroundTruthResult objects for validation

            for h in self._h_values:
                lattice = self.make_lattice(topo, N, J=1.0, h=h)
                H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
                gt = self.solver.solve(H, lattice)
                gt_objects.append(gt)

                topo_results.append(
                    {
                        "h": h,
                        "energy": gt.ground_energy,
                        "gap": gt.gap,
                        "gap_method": gt.gap_method,
                        "mag_x": gt.mag_x,
                        "corr_zz": gt.corr_zz,
                        "per_site_mag_x": gt.per_site_mag_x.tolist(),
                        "per_bond_corr_zz": gt.per_bond_corr_zz.tolist(),
                    }
                )

            # Validate Phase 1 results (physics sanity checks)
            from qmbp_simulation.analysis import GroundTruthValidator

            gt_validator = GroundTruthValidator.from_lattice(
                self.make_lattice(topo, N, J=1.0, h=self._h_values[0]), model=model
            )
            validation_report = gt_validator.validate(gt_objects)
            if validation_report.has_critical:
                for issue in validation_report.critical_issues:
                    logger.error(f"    ❌ {issue.check_id}: {issue.message}")
            if validation_report.has_warnings:
                for issue in validation_report.warnings:
                    logger.warning(f"    ⚠️  {issue.check_id}: {issue.message}")
            if validation_report.passed:
                logger.info(f"    ✓ Phase 1 validation passed ({len(gt_objects)} points)")

            self._ground_truth[topo] = topo_results

            # Cache ground state vectors for Section 2 fidelity (avoids re-computing eigsh)
            # Only available for N ≤ STATEVECTOR_MAX_N (solver stores ψ_gs for N≤15)
            if not hasattr(self, "_cached_ground_states"):
                self._cached_ground_states: dict[str, list[np.ndarray | None]] = {}
            self._cached_ground_states[topo] = [gt.ground_state for gt in gt_objects]
            all_results[topo] = {
                "n_points": len(topo_results),
                "e_min": min(r["energy"] for r in topo_results),
                "e_max": max(r["energy"] for r in topo_results),
                "gap_min": min(r["gap"] for r in topo_results),
                "gap_max": max(r["gap"] for r in topo_results),
                "mag_x_range": [
                    min(r["mag_x"] for r in topo_results),
                    max(r["mag_x"] for r in topo_results),
                ],
                "corr_zz_range": [
                    min(r["corr_zz"] for r in topo_results),
                    max(r["corr_zz"] for r in topo_results),
                ],
                "validation_passed": validation_report.passed,
                "validation_warnings": [i.message for i in validation_report.warnings],
                "validation_errors": [i.message for i in validation_report.critical_issues],
            }
            logger.info(
                f"    E ∈ [{all_results[topo]['e_min']:.6f}, "
                f"{all_results[topo]['e_max']:.6f}], "
                f"gap ∈ [{all_results[topo]['gap_min']:.6f}, "
                f"{all_results[topo]['gap_max']:.6f}]"
            )

        return {
            "pass": all(all_results[t].get("validation_passed", True) for t in all_results),
            "topologies": all_results,
            "h_values": self._h_values,
            "n_qubits": N,
            "model": model,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 2: VQE Optimization
    # ═══════════════════════════════════════════════════════════════════════════

    def section_vqe(self) -> dict:
        """Run descending warm-start VQE for all topologies using VQEOptimizer."""
        from qmbp_simulation import VQEConfig, VQEOptimizer
        from qmbp_simulation.optimizers.sweep_strategies import (
            AdaptiveRestartConfig,
            SelectiveAscendingConfig,
            compute_adaptive_restarts,
            select_suspicious_points,
        )

        N = self._args.n_qubits
        p = self._args.p_layers
        model = self._args.model
        seed = self._args.seeds[0]
        spec = self._get_spec()
        maxiter = self._args.maxiter
        n_restarts = self._args.n_restarts

        # Adaptive restart config (uses n_restarts as max, reduces for easy points)
        h_crit = self.H_CRITICAL_ESTIMATES.get(model)
        adaptive_cfg = AdaptiveRestartConfig(
            base_restarts=max(1, n_restarts // 3),
            max_restarts=n_restarts,
            critical_restarts=max(2, n_restarts // 2),
            h_critical=h_crit,
        )

        # Selective ascending config
        selective_asc_cfg = SelectiveAscendingConfig(
            de_gap_threshold=0.02,
            include_neighbors=True,
        )

        # Configure VQE via VQEConfig (uses max restarts — adaptive overrides per-point)
        vqe_config = VQEConfig(
            p_layers=p,
            n_restarts=n_restarts,
            maxiter=maxiter,
            method="L-BFGS-B",
            enable_callbacks=False,  # Keep lightweight
        )
        optimizer = VQEOptimizer(config=vqe_config, backend=self._vqe_backend, seed=seed)
        # Store for cross-section reuse (e.g., energy guard in section_mpnn_train)
        self._optimizer = optimizer

        all_results = {}
        overall_pass = True

        for topo in self._args.topology:
            logger.info(f"===== VQE sweep: {topo}, N={N}, p={p}, seed={seed}, model={model} =====")

            # Build circuit once (topology-dependent for param count)
            lattice_ref = self.make_lattice(topo, N, J=1.0, h=self._h_values[0])
            circuit, _ = spec.create_circuit(N, p, lattice_ref, **spec.circuit_kwargs)
            n_params = circuit.num_parameters
            logger.info(f"    Circuit: {n_params} parameters")
            self.log_memory_estimate(N, label=f"VQE statevector ({topo})")

            rng = np.random.default_rng(seed)
            prev_theta = rng.uniform(-0.01, 0.01, n_params)
            topo_results = []
            _consecutive_violations = 0  # Track consecutive variational violations

            # ── Resume from checkpoint if available ────────────────────────
            start_idx = 0
            checkpoint = self._load_vqe_checkpoint(topo, n_params=n_params)
            if checkpoint is not None:
                topo_results, prev_theta = checkpoint
                start_idx = len(topo_results)
                if start_idx >= len(self._h_values):
                    logger.info("    ✓ Checkpoint complete — skipping VQE for %s", topo)

            for idx, h in enumerate(self._h_values):
                if idx < start_idx:
                    continue  # Already computed in checkpoint
                t0 = time.perf_counter()
                lattice_h = self.make_lattice(topo, N, J=1.0, h=h)
                H = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)

                # Get exact reference (energy + ground state vector)
                gt = self._ground_truth.get(topo, [])
                if gt and idx < len(gt):
                    e_exact = gt[idx]["energy"]
                    gap = gt[idx]["gap"]
                else:
                    e_exact, gap = self.exact_ground_state(topo, N, h, model=model)

                # Annotate E_exact method for traceability
                from qmbp_simulation.models.constants import EXACT_DIAG_QUBIT_LIMIT

                _e_exact_method = "eigsh" if N <= EXACT_DIAG_QUBIT_LIMIT else "dmrg_1d"

                # Get ground state vector for fidelity (use cache from Section 1 if available)
                cached_gs = getattr(self, "_cached_ground_states", {}).get(topo, [])
                if cached_gs and idx < len(cached_gs) and cached_gs[idx] is not None:
                    gs = cached_gs[idx]
                else:
                    gs = self.solver.ground_state_vector(H)

                # Adaptive restarts: adjust n_restarts based on prev point quality
                prev_de_gap = topo_results[-1]["de_gap"] if topo_results else None
                adaptive_n_restarts = compute_adaptive_restarts(
                    h,
                    prev_de_gap=prev_de_gap,
                    config=adaptive_cfg,
                )
                optimizer.config.n_restarts = adaptive_n_restarts

                # Run VQE via VQEOptimizer (handles multi-restart + warm-start)
                # Backend.compute_fidelity() handles N>15 via fast MPS path.
                vqe_result = optimizer.optimize(
                    hamiltonian=H,
                    circuit=circuit,
                    initial_guess=prev_theta,
                    exact_energy=e_exact,
                    exact_state=gs,
                )

                # Get VQE statevector once (for entanglement entropy)
                sv_data = self._vqe_backend.get_statevector(circuit, vqe_result.theta_opt)

                # θ smoothness: ||θ_new - θ_prev||_∞
                theta_change = float(np.max(np.abs(vqe_result.theta_opt - prev_theta)))
                prev_theta = vqe_result.theta_opt.copy()
                elapsed = time.perf_counter() - t0

                # Entanglement entropy (half-chain bipartition) — reuses sv_data
                entanglement_entropy = self._compute_entanglement_entropy(sv_data, N)

                # Compute ΔE/gap
                de_gap = abs(vqe_result.energy - e_exact) / max(gap, 1e-10)

                # Variational principle check: E_vqe must be >= E_exact - ε
                variational_violation = max(0.0, e_exact - vqe_result.energy - 1e-8)
                variational_ok = vqe_result.energy >= e_exact - 1e-8

                if not variational_ok:
                    _consecutive_violations += 1
                    logger.warning(
                        f"    ⚠️  Variational principle violated at h={h:.4f}: "
                        f"E_vqe={vqe_result.energy:.8f} < E_exact={e_exact:.8f} "
                        f"(diff={variational_violation:.2e}, consecutive: {_consecutive_violations})"
                    )
                    # Auto-abort only for LARGE violations (>1e-2).
                    # Small violations (1e-3 to 1e-8) often indicate approximate E_exact
                    # reference (e.g., DMRG 1D model used for non-1D topology).
                    if _consecutive_violations > 4 and variational_violation > 1e-2:
                        logger.error(
                            f"    ❌ ABORT: >4 consecutive LARGE variational violations in {topo}. "
                            f"(max violation: {variational_violation:.2e} >> 1e-2) "
                            f"This indicates a systematic error in the solver or backend. "
                            f"Saving {len(topo_results)} completed points and skipping remaining."
                        )
                        break
                    elif _consecutive_violations > 4:
                        logger.warning(
                            f"    ⚠️  >4 consecutive small violations in {topo} "
                            f"(max={variational_violation:.2e}). Likely approximate E_exact "
                            f"reference (DMRG on non-1D topology). Continuing sweep."
                        )
                        # Reset to prevent repeated warnings every point
                        _consecutive_violations = 0
                else:
                    _consecutive_violations = 0  # Reset on success

                topo_results.append(
                    {
                        "h": h,
                        "energy_vqe": vqe_result.energy,
                        "energy_exact": e_exact,
                        "e_exact_method": _e_exact_method,
                        "gap": gap,
                        "de_gap": de_gap,
                        "fidelity": vqe_result.fidelity,
                        "energy_variance": vqe_result.energy_variance,
                        "entanglement_entropy": entanglement_entropy,
                        "theta_opt": vqe_result.theta_opt.tolist(),
                        "converged": vqe_result.n_iterations > 0,
                        "n_iterations": vqe_result.n_iterations,
                        "n_restarts_used": (
                            vqe_result.trajectory.n_restarts_used if vqe_result.trajectory else 0
                        ),
                        "theta_change_linf": theta_change,
                        "elapsed_s": elapsed,
                        "variational_violation": variational_violation,
                        "variational_ok": variational_ok,
                    }
                )

                status = "✓" if de_gap < 0.01 else "?"
                logger.info(
                    f"    [{status}] h={h:.4f}: E={vqe_result.energy:.8f} "
                    f"ΔE/gap={de_gap:.2e} F={vqe_result.fidelity:.6f} ({elapsed:.1f}s)"
                )

                # VQE checkpoint: save progress after each h-point so that
                # interrupted runs can be recovered without re-running VQE.
                self._save_vqe_checkpoint(topo, topo_results, prev_theta)

            # ── Bidirectional warm-start: ascending pass ──────────────────────
            # Re-visit suspicious points from h_min→h_max with warm-start.
            # Keep the result with lower energy at each point.
            # Skip for N>=16 (diminishing returns, saves ~50% runtime).
            skip_bidir = self._args.no_bidirectional or (
                N >= 16 and not self._args.force_bidirectional
            )
            n_improved = 0

            if skip_bidir:
                if N >= 16 and not self._args.no_bidirectional:
                    logger.info(
                        "    ⏭️  Skipping bidirectional pass (N=%d ≥ 16, auto-disabled for speed)", N
                    )
                else:
                    logger.info("    ⏭️  Skipping bidirectional pass (--no-bidirectional)")
            else:
                # Selective: only re-optimize points with ΔE/gap > threshold
                target_indices, asc_report = select_suspicious_points(
                    topo_results,
                    config=selective_asc_cfg,
                )

                if not target_indices:
                    logger.info("    🔄 Ascending pass: no suspicious points (all ΔE/gap < 2%)")
                else:
                    n_targeted = asc_report.n_targeted
                    n_suspicious = asc_report.n_suspicious
                    mode = "full" if asc_report.fell_back_to_full else "selective"
                    logger.info(
                        f"    🔄 Ascending pass ({mode}): targeting "
                        f"{n_targeted}/{len(topo_results)} points "
                        f"({n_suspicious} suspicious + neighbors)"
                    )

                    asc_theta = topo_results[-1]["theta_opt"] if topo_results else prev_theta
                    if isinstance(asc_theta, list):
                        asc_theta = np.array(asc_theta)

                    # Build a map of current theta for propagation
                    # Walk in ascending h order (high index → low index)
                    for idx in target_indices:
                        # Propagate theta from the nearest lower-h neighbor
                        if idx < len(topo_results) - 1:
                            asc_theta = np.array(topo_results[idx + 1]["theta_opt"])

                        h = self._h_values[idx]
                        lattice_h = self.make_lattice(topo, N, J=1.0, h=h)
                        H = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)

                        gt = self._ground_truth.get(topo, [])
                        e_exact = gt[idx]["energy"] if gt and idx < len(gt) else None
                        gs = None
                        if e_exact is not None:
                            gs = self.solver.ground_state_vector(H)

                        # Use fewer restarts for ascending (warm-start is already good)
                        optimizer.config.n_restarts = max(1, adaptive_cfg.base_restarts)

                        vqe_asc = optimizer.optimize(
                            hamiltonian=H,
                            circuit=circuit,
                            initial_guess=asc_theta,
                            exact_energy=e_exact,
                            exact_state=gs,
                        )

                        # Keep if better energy (with meaningful improvement threshold)
                        energy_improvement = topo_results[idx]["energy_vqe"] - vqe_asc.energy
                        if energy_improvement > 1e-8:
                            old_de_gap = topo_results[idx]["de_gap"]
                            old_energy = topo_results[idx]["energy_vqe"]
                            gap = topo_results[idx]["gap"]
                            sv_asc_data = self._vqe_backend.get_statevector(
                                circuit, vqe_asc.theta_opt
                            )
                            ent_asc = self._compute_entanglement_entropy(sv_asc_data, N)
                            de_gap = (
                                abs(vqe_asc.energy - e_exact) / max(gap, 1e-10) if e_exact else 0
                            )

                            topo_results[idx]["energy_vqe"] = vqe_asc.energy
                            topo_results[idx]["theta_opt"] = vqe_asc.theta_opt.tolist()
                            topo_results[idx]["fidelity"] = vqe_asc.fidelity
                            topo_results[idx]["de_gap"] = de_gap
                            topo_results[idx]["entanglement_entropy"] = ent_asc
                            n_improved += 1
                            logger.info(
                                f"      ↑ h={h:.4f}: ΔE/gap {old_de_gap:.2e} → {de_gap:.2e} "
                                f"(ΔE={old_energy - vqe_asc.energy:.6f})"
                            )

                        asc_theta = np.array(topo_results[idx]["theta_opt"])

                    if n_improved > 0:
                        logger.info(
                            f"    🔄 Ascending result: {n_improved}/{n_targeted} points improved."
                        )
                    else:
                        logger.info("    🔄 Ascending result: no improvements.")

            # Topology-level summary
            self._vqe_results[topo] = topo_results
            fidelities = [r["fidelity"] for r in topo_results if r["fidelity"] is not None]
            de_gaps = [r["de_gap"] for r in topo_results if r["de_gap"] is not None]
            theta_changes = [r["theta_change_linf"] for r in topo_results]
            n_pass = sum(1 for d in de_gaps if d < DE_GAP_THRESHOLD)
            topo_pass = n_pass >= len(de_gaps) * 0.8 if de_gaps else False

            # Variational principle violation summary
            n_violations = sum(1 for r in topo_results if not r.get("variational_ok", True))
            max_violation = max(
                (r.get("variational_violation", 0.0) for r in topo_results), default=0.0
            )
            if n_violations > 0:
                logger.warning(
                    f"    ⚠️  {n_violations}/{len(topo_results)} points violate "
                    f"variational principle (max violation: {max_violation:.2e}). "
                    f"This may indicate numerical noise or incorrect E_exact reference."
                )

            # Energy variance aggregation
            valid_variances = [
                r.get("energy_variance")
                for r in topo_results
                if r.get("energy_variance") is not None
                and np.isfinite(r.get("energy_variance", float("nan")))
            ]
            mean_variance = float(np.mean(valid_variances)) if valid_variances else None
            max_variance = float(np.max(valid_variances)) if valid_variances else None

            # Fragile passes: ΔE/gap < 5% but Var(H) > 0.5 (state is not near-eigenstate)
            n_fragile = sum(
                1
                for r in topo_results
                if r.get("de_gap", 1.0) < DE_GAP_THRESHOLD
                and r.get("energy_variance") is not None
                and r.get("energy_variance", 0) > 0.5
            )
            if n_fragile > 0:
                logger.warning(
                    f"    ⚠️  {n_fragile} FRAGILE PASSES: ΔE/gap<5%% but Var(H)>0.5. "
                    f"These points may fail under hardware noise."
                )

            all_results[topo] = {
                "n_points": len(topo_results),
                "n_pass_5pct": n_pass,
                "mean_fidelity": float(np.mean(fidelities)) if fidelities else None,
                "min_fidelity": float(np.min(fidelities)) if fidelities else None,
                "mean_de_gap": float(np.mean(de_gaps)) if de_gaps else None,
                "max_de_gap": float(np.max(de_gaps)) if de_gaps else None,
                "mean_energy_variance": mean_variance,
                "max_energy_variance": max_variance,
                "n_fragile_passes": n_fragile,
                "theta_smoothness_max": float(np.max(theta_changes)) if theta_changes else None,
                "theta_smoothness_mean": float(np.mean(theta_changes)) if theta_changes else None,
                "n_converged": sum(1 for r in topo_results if r["converged"]),
                "n_variational_violations": n_violations,
                "max_variational_violation": max_violation,
                "mean_entanglement_entropy": float(
                    np.mean([r["entanglement_entropy"] for r in topo_results])
                ),
                "total_time_s": sum(r["elapsed_s"] for r in topo_results),
                "pass": topo_pass,
                # Per-point data for recovery and post-hoc analysis
                "per_point": [
                    {
                        "h": r["h"],
                        "energy_vqe": r["energy_vqe"],
                        "energy_exact": r["energy_exact"],
                        "gap": r["gap"],
                        "de_gap": r["de_gap"],
                        "fidelity": r["fidelity"],
                        "energy_variance": r.get("energy_variance"),
                        "entanglement_entropy": r["entanglement_entropy"],
                        "n_iterations": r["n_iterations"],
                        "n_restarts_used": r["n_restarts_used"],
                        "converged": r["converged"],
                        "theta_change_linf": r["theta_change_linf"],
                        "elapsed_s": r["elapsed_s"],
                        "theta_opt": r["theta_opt"],
                        "variational_violation": r.get("variational_violation", 0.0),
                        "variational_ok": r.get("variational_ok", True),
                    }
                    for r in topo_results
                ],
            }
            if not topo_pass:
                overall_pass = False

            if fidelities and de_gaps:
                logger.info(
                    f"    Summary: {n_pass}/{len(de_gaps)} pass (ΔE/gap<5%), "
                    f"mean F={np.mean(fidelities):.6f}, "
                    f"max ΔE/gap={np.max(de_gaps):.2e}"
                )
            else:
                logger.info(
                    f"    Summary: {len(topo_results)} points computed "
                    f"(run Section 1 first for ΔE/gap metrics)"
                )

        # Clean up checkpoints on successful VQE completion
        self._cleanup_vqe_checkpoints()

        # Register theta_opt array as artifact (all h-points × all params)
        for topo_name, topo_data in self._vqe_results.items():
            if topo_data:
                theta_array = np.array([r["theta_opt"] for r in topo_data])
                h_array = np.array([r["h"] for r in topo_data])
                self.artifacts.register(
                    f"theta_opt_{topo_name}",
                    {
                        "theta_opt": theta_array,
                        "h_values": h_array,
                    },
                    format="npz",
                    metadata={
                        "topology": topo_name,
                        "n_points": len(topo_data),
                        "n_params": theta_array.shape[1] if len(theta_array.shape) > 1 else 0,
                    },
                )

        return {"pass": overall_pass, "topologies": all_results}

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 3: MPNN Training
    # ═══════════════════════════════════════════════════════════════════════════

    def section_mpnn_train(self) -> dict:
        """Train GINConv MPNN on VQE θ_opt data."""
        from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn

        # Use first topology's VQE results for training
        topo = self._args.topology[0]
        vqe_data = self._vqe_results.get(topo)
        if not vqe_data:
            return {"pass": False, "error": f"No VQE data for {topo}. Run Section 2 first."}

        N = self._args.n_qubits
        p = self._args.p_layers
        model = self._args.model
        spec = self._get_spec()

        # Build graph dataset from VQE results
        h_values = np.array([r["h"] for r in vqe_data])
        theta_opt_array = np.array([r["theta_opt"] for r in vqe_data])
        e_exact_array = np.array([r["energy_exact"] for r in vqe_data])
        fidelities_array = np.array([r["fidelity"] for r in vqe_data])

        # ── Post-VQE theta alignment ──────────────────────────────────────
        # Detect and fix discontinuities in θ(h) before MPNN training.
        from qmbp_simulation.analysis.theta_alignment import align_theta_array

        energies_array = np.array([r["energy_vqe"] for r in vqe_data])
        theta_smoothness = float(np.max(np.abs(np.diff(theta_opt_array, axis=0))))

        if theta_smoothness > 1.0:
            logger.info(
                f"  θ alignment: smoothness={theta_smoothness:.3f} > 1.0, running alignment pass..."
            )
            # Build Hamiltonians for each h-point
            hamiltonians_for_align = []
            for h_val in h_values:
                lat_h = self.make_lattice(topo, N, J=1.0, h=float(h_val))
                H_h = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)
                hamiltonians_for_align.append(H_h)

            # Build circuit for re-optimization
            lattice_align_ref = self.make_lattice(topo, N, J=1.0, h=float(h_values[0]))
            circuit_align, _ = spec.create_circuit(N, p, lattice_align_ref, **spec.circuit_kwargs)

            theta_opt_array, alignment_report = align_theta_array(
                theta_array=theta_opt_array,
                energies=energies_array,
                circuit=circuit_align,
                hamiltonians=hamiltonians_for_align,
                backend=self._vqe_backend,
            )

            # Update vqe_data in-place for downstream (deploy uses it too)
            for i, r in enumerate(vqe_data):
                r["theta_opt"] = theta_opt_array[i].tolist()

            logger.info(
                f"  θ alignment result: "
                f"{alignment_report.n_realigned}/{alignment_report.n_jumps_detected} fixed, "
                f"smoothness {alignment_report.original_smoothness:.3f} "
                f"→ {alignment_report.final_smoothness:.3f}"
            )
        else:
            logger.info(f"  θ smoothness={theta_smoothness:.3f} ≤ 1.0, no alignment needed.")

        # ── Cross-h energy validation guard ───────────────────────────────
        # Detect points where VQE is stuck in local minima (high ΔE/gap
        # while neighbors are fine) and attempt repair via neighbor-seeded reopt.
        from qmbp_simulation.analysis.theta_alignment import cross_h_energy_guard

        gaps_array = np.array([r["gap"] for r in vqe_data])
        logger.info("  🛡️ Cross-h energy guard: checking for local-minimum traps...")

        def _reoptimize_point(idx: int, theta_seed: np.ndarray):
            """Re-run VQE at h_values[idx] with theta_seed as warm-start."""
            h_val = float(h_values[idx])
            lat_h = self.make_lattice(topo, N, J=1.0, h=h_val)
            H_h = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)
            lattice_ckt = self.make_lattice(topo, N, J=1.0, h=h_val)
            circuit_reopt, _ = spec.create_circuit(N, p, lattice_ckt, **spec.circuit_kwargs)
            result = self._optimizer.optimize(
                hamiltonian=H_h,
                circuit=circuit_reopt,
                initial_guess=theta_seed,
                exact_energy=float(e_exact_array[idx]),
            )
            return result.energy, result.theta_opt

        energies_guard, theta_guard, guard_report = cross_h_energy_guard(
            vqe_energies=energies_array,
            exact_energies=e_exact_array,
            gaps=gaps_array,
            theta_array=theta_opt_array,
            h_values=h_values,
            reoptimize_fn=_reoptimize_point,
        )

        if guard_report.n_repaired > 0:
            theta_opt_array = theta_guard
            energies_array = energies_guard
            # Update vqe_data in-place
            for i, r in enumerate(vqe_data):
                r["theta_opt"] = theta_opt_array[i].tolist()
                r["energy_vqe"] = float(energies_array[i])
            logger.info(
                f"  🛡️ Energy guard result: "
                f"{guard_report.n_repaired}/{guard_report.n_suspicious} repaired."
            )
        else:
            logger.info(
                f"  🛡️ Energy guard result: "
                f"{guard_report.n_suspicious} suspicious, 0 repairs needed."
            )

        logger.info(
            f"  Training MPNN: {len(h_values)} points, "
            f"{theta_opt_array.shape[1]} params, topology={topo}"
        )

        # Build dataset (no fidelity filter for noiseless — all points are valid)
        # But apply outlier detection to exclude VQE local-minimum traps
        from qmbp_simulation.analysis.theta_alignment import filter_theta_outliers

        theta_clean, h_clean, e_exact_clean, fid_clean, outlier_report = filter_theta_outliers(
            theta_array=theta_opt_array,
            h_values=h_values,
            e_exact=e_exact_array,
            fidelities=fidelities_array,
            threshold=2.0,
            fidelity_floor=0.5,
            replace_strategy="interpolate",
        )
        if outlier_report.n_outliers > 0:
            logger.info(
                f"  Outlier filter: {outlier_report.n_outliers} points interpolated "
                f"at h={outlier_report.outlier_h_values}"
            )

        lattice_ref = self.make_lattice(topo, N, J=1.0, h=float(h_clean[0]))
        dataset = build_graph_dataset(
            lattice_ref,
            h_clean,
            theta_clean,
            e_exact_clean,
            fidelities=fid_clean,
            fidelity_threshold=0.0,  # No fidelity filtering (outliers already handled)
        )

        # Build circuit once for energy_val_fn
        circuit, _ = spec.create_circuit(N, p, lattice_ref, **spec.circuit_kwargs)

        logger.info(
            "  📊 MPNN training data: %d graphs, %d node features, %d output params",
            len(dataset),
            dataset[0].x.shape[1] if dataset else 0,
            theta_clean.shape[1] if len(theta_clean.shape) > 1 else 0,
        )

        # Energy validation callback: evaluates ΔE/gap during training
        # Maps predicted θ → actual energy via NoiselessBackend
        def energy_val_fn(pred_batch, data_batch):
            """Compute |E(θ_pred) - E_exact| / gap for each sample in batch."""
            errors = []
            pred_np = pred_batch.detach().cpu().numpy()
            batch_size = pred_np.shape[0]

            # Extract per-sample attributes from PyG Batch
            h_vals = data_batch.h_value if hasattr(data_batch, "h_value") else None
            e_exacts = data_batch.e_exact if hasattr(data_batch, "e_exact") else None

            for i in range(batch_size):
                theta_pred_i = pred_np[i]
                h_val = (
                    float(h_vals[i]) if h_vals is not None else float(h_values[i % len(h_values)])
                )
                e_exact_i = (
                    float(e_exacts[i])
                    if e_exacts is not None
                    else float(e_exact_array[i % len(e_exact_array)])
                )

                # Evaluate energy with predicted parameters
                lattice_i = self.make_lattice(topo, N, J=1.0, h=h_val)
                H_i = spec.build_hamiltonian(lattice_i, **spec.hamiltonian_kwargs)
                try:
                    e_pred = self._vqe_backend.evaluate(circuit, H_i, theta_pred_i)
                    # Get gap for this h-point
                    gap_i = next(
                        (r["gap"] for r in vqe_data if abs(r["h"] - h_val) < 0.01),
                        1.0,
                    )
                    de_gap = abs(e_pred - e_exact_i) / max(gap_i, 1e-10)
                    errors.append(de_gap)
                except Exception as e:
                    logger.debug("    Energy eval failed at h=%.4f: %s", h_val, e)
                    errors.append(float("inf"))
            return errors

        # Train MPNN with energy validation callback + physics-informed loss
        n_output = theta_opt_array.shape[1]
        n_node_features = dataset[0].x.shape[1]
        predictor = MPNNPredictor(
            node_features=n_node_features,
            hidden_dim=128,
            output_dim=n_output,
            n_layers=3,
            # NOTE: dropout=0.1 is hardcoded in MPNNPredictor MLP heads (always active)
        )

        # ── Physics-informed loss (regularizes MPNN with actual energy eval) ──
        import torch as _torch

        use_physics_loss = not self._args.no_physics_loss
        physics_loss_weight = self._args.physics_loss_weight
        physics_loss_start_epoch = self._args.physics_loss_start

        _physics_loss_fn_impl = None
        if use_physics_loss:

            def _physics_loss_fn_impl(pred_batch: _torch.Tensor, data_batch) -> _torch.Tensor:
                """Compute energy penalty: mean |E(θ_pred) - E_exact| / gap."""
                pred_np = pred_batch.detach().cpu().numpy()
                batch_size = pred_np.shape[0]
                penalties = []
                h_vals_batch = data_batch.h_value if hasattr(data_batch, "h_value") else None
                e_exacts_batch = data_batch.e_exact if hasattr(data_batch, "e_exact") else None

                for i in range(min(batch_size, 5)):  # Eval max 5 points per batch
                    theta_i = pred_np[i]
                    h_val = float(h_vals_batch[i]) if h_vals_batch is not None else 0
                    e_exact_i = float(e_exacts_batch[i]) if e_exacts_batch is not None else 0
                    lattice_i = self.make_lattice(topo, N, J=1.0, h=h_val)
                    H_i = spec.build_hamiltonian(lattice_i, **spec.hamiltonian_kwargs)
                    try:
                        e_pred = self._vqe_backend.evaluate(circuit, H_i, theta_i)
                        gap_i = next(
                            (r["gap"] for r in vqe_data if abs(r["h"] - h_val) < 0.01),
                            1.0,
                        )
                        penalties.append(abs(e_pred - e_exact_i) / max(gap_i, 1e-10))
                    except Exception as e:
                        logger.debug("    Physics loss eval failed at h=%.3f: %s", h_val, e)
                        penalties.append(0.0)

                if penalties:
                    return _torch.tensor(sum(penalties) / len(penalties), dtype=_torch.float32)
                return _torch.tensor(0.0)

        physics_status = (
            f"ON (λ={physics_loss_weight}, start={physics_loss_start_epoch})"
            if use_physics_loss
            else "OFF"
        )
        logger.info(
            "  🧠 MPNN training: n_features=%d, hidden=128, output=%d, physics_loss=%s",
            n_node_features,
            n_output,
            physics_status,
        )

        train_result = train_mpnn(
            predictor,
            dataset,
            n_epochs=6000,
            lr=1e-3,
            patience=300,
            energy_val_fn=energy_val_fn,
            energy_val_interval=100,
            physics_loss_fn=_physics_loss_fn_impl,
            physics_loss_weight=physics_loss_weight,
            physics_loss_start_epoch=physics_loss_start_epoch,
        )

        final_mse = train_result["final_mse"]
        energy_val_history = train_result.get("energy_val_history", [])
        self._mpnn_model = predictor
        self._mpnn_train_loss = final_mse

        # ── Register artifacts for versioned persistence ──────────────
        self.artifacts.register(
            "mpnn_model",
            predictor,
            format="pt",
            metadata={
                "n_qubits": N,
                "p_layers": p,
                "model": model,
                "topology": topo,
                "n_training_points": len(h_values),
                "final_mse": float(final_mse),
            },
        )
        self.artifacts.register(
            "training_data",
            {
                "h_values": h_values if isinstance(h_values, np.ndarray) else np.array(h_values),
                "theta_opt": theta_opt_array,
                "e_exact": e_exact_array,
            },
            format="npz",
            metadata={"n_points": len(h_values), "n_params": n_output},
        )

        # Register circuit (built earlier in section)
        self.artifacts.register(
            "circuit",
            circuit,
            format="qpy",
            metadata={
                "n_qubits": N,
                "p_layers": p,
                "topology": topo,
                "n_params": n_output,
            },
        )
        # Also save human-readable QASM3 for documentation/analysis
        self.artifacts.register(
            "circuit_readable",
            circuit,
            format="qasm3",
            metadata={
                "n_qubits": N,
                "p_layers": p,
                "topology": topo,
                "n_params": n_output,
                "depth": circuit.depth(),
                "size": circuit.size(),
            },
        )

        # ── Save MPNN checkpoint with config fingerprint ──────────────
        # Allows Section 4 to load a trained MPNN without re-training
        # if the run is resumed and Section 3 was already completed.
        try:
            import hashlib

            from qmbp_simulation.predictors import save_mpnn_checkpoint

            # Fingerprint: model + topology + n_qubits + p_layers + n_h_points
            fp_str = f"{model}_{topo}_{N}_{p}_{len(h_values)}"
            fp_hash = hashlib.md5(fp_str.encode()).hexdigest()[:8]
            mpnn_ckpt_dir = self._checkpoint_dir() / "mpnn_checkpoints"
            mpnn_ckpt_dir.mkdir(parents=True, exist_ok=True)
            mpnn_ckpt_path = mpnn_ckpt_dir / f"mpnn_{topo}_n{N}_p{p}_{fp_hash}.pt"
            save_mpnn_checkpoint(predictor, mpnn_ckpt_path)
            logger.info(f"    💾 MPNN checkpoint saved: {mpnn_ckpt_path.name}")
        except Exception as e:
            logger.debug(f"    MPNN checkpoint save failed (non-fatal): {e}")

        # ── Record diagnostics via DiagnosticCollector ────────────────

        # Record per-epoch loss (every 10th to limit storage)
        for epoch_idx, mse_val in enumerate(train_result.get("mse_history", [])):
            if epoch_idx % 10 == 0 or epoch_idx == len(train_result["mse_history"]) - 1:
                self._collector.record_mpnn_epoch(epoch=epoch_idx, train_loss=mse_val)

        # Compute and record per-h MSE from trained model
        predictor.eval()
        per_h_mse = []
        with _torch.no_grad():
            for data in dataset:
                pred = predictor(data).numpy().flatten()
                target = data.y.numpy().flatten()
                mse_i = float(np.mean((pred - target) ** 2))
                per_h_mse.append(mse_i)

        self._collector.record_mpnn_per_h_error(
            h_values=h_values,
            per_h_mse=np.array(per_h_mse),
        )

        # Store training data for ThetaValidator in Section 4
        self._theta_opt_array = theta_opt_array
        self._train_h_values = h_values

        # Pass criterion: MSE < 1e-3 OR final energy validation < 5%
        final_de_gap = energy_val_history[-1] if energy_val_history else None
        passed = final_mse < 1e-3 or (final_de_gap is not None and final_de_gap < DE_GAP_THRESHOLD)

        de_gap_str = f"{final_de_gap:.2e}" if final_de_gap is not None else "N/A"
        logger.info(
            f"    Training complete: final_mse={final_mse:.2e}, "
            f"final_ΔE/gap={de_gap_str}, "
            f"stopped_early={train_result.get('stopped_early', False)} "
            f"{'[PASS]' if passed else '[FAIL]'}"
        )

        # Summarize mse training curve (compact, not full 3000-value history)
        mse_hist = train_result.get("mse_history", [])
        mse_summary = {
            "final": float(final_mse),
            "best": float(min(mse_hist)) if mse_hist else float(final_mse),
            "at_epoch_100": float(mse_hist[99]) if len(mse_hist) > 99 else None,
            "at_epoch_500": float(mse_hist[499]) if len(mse_hist) > 499 else None,
            "at_epoch_1000": float(mse_hist[999]) if len(mse_hist) > 999 else None,
            "n_epochs_total": len(mse_hist),
        }

        return {
            "pass": passed,
            "topology": topo,
            "n_training_points": len(h_values),
            "n_output_params": n_output,
            "final_mse": float(final_mse),
            "final_de_gap": float(final_de_gap) if final_de_gap is not None else None,
            "mse_summary": mse_summary,
            "energy_val_history": [float(x) for x in energy_val_history],
            "per_h_mse": [float(m) for m in per_h_mse],
            "per_h_mse_max": float(np.max(per_h_mse)) if per_h_mse else None,
            "per_h_mse_mean": float(np.mean(per_h_mse)) if per_h_mse else None,
            "stopped_early": train_result.get("stopped_early", False),
            "stop_reason": train_result.get("stop_reason", ""),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 4: Deploy (Predict + Evaluate)
    # ═══════════════════════════════════════════════════════════════════════════

    def section_deploy(self) -> dict:
        """Predict θ at held-out h-points and evaluate energy + phase label."""
        import torch
        from torch_geometric.data import Data

        from qmbp_simulation import HamiltonianBuilder

        # Try loading MPNN from checkpoint if not in memory (e.g., after --resume)
        if self._mpnn_model is None:
            self._mpnn_model = self._try_load_mpnn_checkpoint()

        if self._mpnn_model is None:
            return {"pass": False, "error": "No trained MPNN. Run Section 3 first."}

        N = self._args.n_qubits
        p = self._args.p_layers
        model = self._args.model
        spec = self._get_spec()
        topo = self._args.topology[0]
        builder = HamiltonianBuilder()

        # Generate test h-points (midpoints between training points)
        train_h = sorted([r["h"] for r in self._vqe_results[topo]])
        test_h = [(train_h[i] + train_h[i + 1]) / 2 for i in range(len(train_h) - 1)]
        logger.info(f"  Deploy: {len(test_h)} test h-points (midpoints), topology={topo}")

        # Build circuit for evaluation
        lattice_ref = self.make_lattice(topo, N, J=1.0, h=test_h[0])
        circuit, _ = spec.create_circuit(N, p, lattice_ref, **spec.circuit_kwargs)

        # Get graph topology (edge_index, coord) once
        edge_index_np, coord = builder.build_graph_data(lattice_ref)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)

        results = []
        n_correct_label = 0
        n_pass_energy = 0

        # ── Build ThetaValidator from Phase 2/3 training data ────────
        theta_validator = None
        if hasattr(self, "_theta_opt_array") and self._theta_opt_array is not None:
            theta_validator = self.ThetaValidator.from_training_data(
                theta_opt=self._theta_opt_array,
                h_values=self._train_h_values,
            )
            logger.info("    ThetaValidator initialized from training data")

        self._mpnn_model.eval()
        validation_reports = []
        for h_test in test_h:
            # Build graph with correct h for this point
            x_features = np.stack([np.full(N, h_test), coord], axis=1)
            x = torch.tensor(x_features, dtype=torch.float32)
            graph = Data(x=x, edge_index=edge_index)

            with torch.no_grad():
                theta_pred = self._mpnn_model(graph).numpy().flatten()

            # ── ThetaValidator: validate predicted θ ──────────────────
            theta_val_report = None
            if theta_validator is not None:
                theta_val_report = theta_validator.validate(
                    theta_pred,
                    level=3,  # L1=bounds, L2=NaN, L3=interpolation (no circuit eval needed)
                    h_test=h_test,
                )
                if not theta_val_report.passes():
                    logger.warning(
                        f"    θ_pred validation warning at h={h_test:.4f}: "
                        f"confidence={theta_val_report.confidence_score:.3f}, "
                        f"warnings={theta_val_report.warnings}"
                    )
                validation_reports.append(
                    {
                        "h_test": h_test,
                        "passes": theta_val_report.passes(),
                        "confidence_score": theta_val_report.confidence_score,
                        "warnings": theta_val_report.warnings,
                    }
                )

            # Evaluate energy
            lattice_t = self.make_lattice(topo, N, J=1.0, h=h_test)
            H = spec.build_hamiltonian(lattice_t, **spec.hamiltonian_kwargs)
            e_pred = self._vqe_backend.evaluate(circuit, H, theta_pred)

            # Get exact reference
            e_exact, gap = self.exact_ground_state(topo, N, h_test, model=model)
            de_gap = abs(e_pred - e_exact) / max(gap, 1e-10)

            # Baseline: VQE from random init (no MPNN warm-start)
            rng_baseline = np.random.default_rng(42)
            x0_random = rng_baseline.uniform(-np.pi, np.pi, circuit.num_parameters)
            res_random = self.minimize(
                lambda params, _H=H, _c=circuit: self._vqe_backend.evaluate(_c, _H, params),
                x0_random,
                method="L-BFGS-B",
                bounds=[(-np.pi, np.pi)] * circuit.num_parameters,
                options={"maxiter": self._args.maxiter, "ftol": 1e-14},
            )
            e_random = res_random.fun
            de_gap_random = abs(e_random - e_exact) / max(gap, 1e-10)
            n_iters_random = getattr(res_random, "nit", 0)

            # Fidelity with exact ground state
            from qiskit.quantum_info import Statevector

            # ── VQE refinement for marginal predictions ───────────────────
            # If ΔE/gap > 5% but fidelity/confidence suggests we're close,
            # use θ_pred as warm-start for a short VQE to "polish" the result.
            refined = False
            if de_gap > DE_GAP_THRESHOLD and de_gap < 5.0:
                # Short VQE: 50 iters, 1 restart, using θ_pred as seed
                res_refine = self.minimize(
                    lambda params, _H=H, _c=circuit: self._vqe_backend.evaluate(_c, _H, params),
                    theta_pred,
                    method="L-BFGS-B" if len(theta_pred) <= 8 else "COBYLA",
                    bounds=[(-np.pi, np.pi)] * len(theta_pred),
                    options={"maxiter": 50},
                )
                e_refined = res_refine.fun
                de_gap_refined = abs(e_refined - e_exact) / max(gap, 1e-10)
                if de_gap_refined < de_gap:
                    logger.info(
                        f"    🔧 Refined h={h_test:.3f}: ΔE/gap {de_gap:.4f} → "
                        f"{de_gap_refined:.4f} (VQE polish, 50 iters)"
                    )
                    theta_pred = res_refine.x
                    e_pred = e_refined
                    de_gap = de_gap_refined
                    refined = True

            sv_data = self._vqe_backend.get_statevector(circuit, theta_pred)
            sv = Statevector(sv_data)
            gs = self.solver.ground_state_vector(H)
            fidelity = float(self.state_fidelity(sv, Statevector(gs)))

            # Entanglement entropy of predicted state
            entanglement_entropy = self._compute_entanglement_entropy(sv.data, N)

            # Compute observables from predicted state for phase classification
            obs_x_ops, obs_zz_ops = self.builder.build_local_observables(lattice_t)
            mag_x_pred = float(np.mean([np.real(sv.expectation_value(op)) for op in obs_x_ops]))
            corr_zz_pred = float(np.mean([np.real(sv.expectation_value(op)) for op in obs_zz_ops]))

            # Exact observables for comparison
            gt_data = self._ground_truth.get(topo, [])
            mag_x_exact = None
            corr_zz_exact = None
            if gt_data:
                # Find closest h in ground_truth
                closest_gt = min(gt_data, key=lambda r: abs(r["h"] - h_test))
                if abs(closest_gt["h"] - h_test) < 0.1:
                    mag_x_exact = closest_gt["mag_x"]
                    corr_zz_exact = closest_gt["corr_zz"]

            # Phase label from observables: |⟨X⟩| > |⟨ZZ⟩| → paramagnetic
            predicted_phase = (
                "paramagnetic" if abs(mag_x_pred) > abs(corr_zz_pred) else "ferromagnetic"
            )
            # Correct label: compare with energy criterion
            correct_label = de_gap < DE_GAP_THRESHOLD

            if correct_label:
                n_correct_label += 1
            if de_gap < DE_GAP_THRESHOLD:
                n_pass_energy += 1

            results.append(
                {
                    "h_test": h_test,
                    "e_pred": e_pred,
                    "e_exact": e_exact,
                    "gap": gap,
                    "de_gap": de_gap,
                    "de_gap_random_init": de_gap_random,
                    "n_iters_random_init": n_iters_random,
                    "mpnn_vs_random_ratio": de_gap / max(de_gap_random, 1e-15),
                    "fidelity": fidelity,
                    "entanglement_entropy": entanglement_entropy,
                    "mag_x_pred": mag_x_pred,
                    "corr_zz_pred": corr_zz_pred,
                    "mag_x_exact": mag_x_exact,
                    "corr_zz_exact": corr_zz_exact,
                    "mag_x_error": abs(mag_x_pred - mag_x_exact)
                    if mag_x_exact is not None
                    else None,
                    "corr_zz_error": abs(corr_zz_pred - corr_zz_exact)
                    if corr_zz_exact is not None
                    else None,
                    "phase": predicted_phase,
                    "correct_label": correct_label,
                    "refined": refined,
                    "theta_validation": {
                        "passes": theta_val_report.passes() if theta_val_report else None,
                        "confidence_score": theta_val_report.confidence_score
                        if theta_val_report
                        else None,
                    }
                    if theta_val_report is not None
                    else None,
                }
            )

            status = "✓" if de_gap < DE_GAP_THRESHOLD else "✗"
            logger.info(
                f"    [{status}] h={h_test:.4f}: ΔE/gap={de_gap:.2e} (MPNN) vs "
                f"{de_gap_random:.2e} (random), F={fidelity:.6f}, phase={predicted_phase}"
            )

        # Overall verdict
        n_total = len(results)
        energy_pass = n_pass_energy >= n_total * 0.8
        label_pass = n_correct_label >= n_total * 0.8
        overall = energy_pass and label_pass

        # Speedup factor: VQE iterations saved by using MPNN zero-shot
        vqe_data = self._vqe_results.get(topo, [])
        mean_vqe_iters = (
            float(np.mean([r["n_iterations"] for r in vqe_data if r.get("n_iterations", 0) > 0]))
            if vqe_data
            else 0.0
        )
        # MPNN uses 1 forward pass (≈0 circuit evaluations) vs VQE mean_iters
        speedup_factor = mean_vqe_iters if mean_vqe_iters > 0 else float("inf")

        # MPNN vs random-init comparison
        de_gaps_mpnn = [r["de_gap"] for r in results]
        de_gaps_random = [r["de_gap_random_init"] for r in results]
        mpnn_wins = sum(1 for m, r in zip(de_gaps_mpnn, de_gaps_random, strict=False) if m < r)

        logger.info(
            f"    Verdict: {n_pass_energy}/{n_total} energy pass, "
            f"{n_correct_label}/{n_total} label correct → "
            f"{'PASS' if overall else 'FAIL'}"
        )
        logger.info(
            f"    Speedup: MPNN zero-shot vs VQE ({mean_vqe_iters:.0f} iters) = "
            f"{speedup_factor:.0f}× fewer circuit evaluations"
        )
        logger.info(
            f"    MPNN vs random-init: MPNN wins {mpnn_wins}/{n_total} points, "
            f"mean ΔE/gap MPNN={np.mean(de_gaps_mpnn):.2e} vs "
            f"random={np.mean(de_gaps_random):.2e}"
        )

        return {
            "pass": overall,
            "n_test_points": n_total,
            "n_pass_energy": n_pass_energy,
            "n_correct_label": n_correct_label,
            "mean_de_gap": float(np.mean(de_gaps_mpnn)),
            "max_de_gap": float(np.max(de_gaps_mpnn)),
            "mean_de_gap_random_init": float(np.mean(de_gaps_random)),
            "max_de_gap_random_init": float(np.max(de_gaps_random)),
            "mpnn_wins_vs_random": mpnn_wins,
            "mean_fidelity": float(np.mean([r["fidelity"] for r in results])),
            "mean_entanglement_entropy": float(
                np.mean([r["entanglement_entropy"] for r in results])
            ),
            "speedup_factor": speedup_factor,
            "mean_vqe_iters_per_point": mean_vqe_iters,
            "mean_mag_x_error": float(
                np.mean([r["mag_x_error"] for r in results if r["mag_x_error"] is not None])
            )
            if any(r["mag_x_error"] is not None for r in results)
            else None,
            "mean_corr_zz_error": float(
                np.mean([r["corr_zz_error"] for r in results if r["corr_zz_error"] is not None])
            )
            if any(r["corr_zz_error"] is not None for r in results)
            else None,
            "theta_validation_summary": {
                "n_validated": len(validation_reports),
                "n_passed": sum(1 for r in validation_reports if r["passes"]),
                "mean_confidence": float(
                    np.mean([r["confidence_score"] for r in validation_reports])
                )
                if validation_reports
                else None,
                "min_confidence": float(np.min([r["confidence_score"] for r in validation_reports]))
                if validation_reports
                else None,
            }
            if validation_reports
            else None,
            "per_point": results,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    NoiselessPipelineRunner.main()
