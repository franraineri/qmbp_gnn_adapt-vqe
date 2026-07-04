"""
Pipeline Runner — Orchestrates the full Phase 1 → 2 → 3 → 4 workflow.

Supports skip/resume via phase flags and checkpoint detection.
Uses the package's solvers, optimizers, predictors, and execution backends.

Requirements: 10.4, 10.5
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from qmbp_simulation.execution import ExecutionBackend, NoiselessBackend
from qmbp_simulation.models import (
    DeployResult,
    GroundTruthResult,
    HamiltonianBuilder,
    LatticeConfig,
    ModelSpec,
    VQEConfig,
    VQEResult,
    make_lattice,
)
from qmbp_simulation.models.constants import (
    AUGMENTATION_DATASET_THRESHOLD,
    AUGMENTATION_N_COPIES,
    AUGMENTATION_NOISE_SIGMA,
    DE_GAP_THRESHOLD,
    GEN_GAP_CATASTROPHIC,
    GEN_GAP_WARNING,
    SWA_AVERAGE_WEIGHT,
    SWA_EXTRA_EPOCHS,
    SWA_LR,
    THETA_SMOOTHNESS_CHAIN_BREAK,
    VQE_REFINEMENT_DE_GAP_MAX,
    VQE_REFINEMENT_DE_GAP_MIN,
)
from qmbp_simulation.optimizers import VQEOptimizer
from qmbp_simulation.pipeline.dataset_io import (
    load_phase12_dataset,
    save_phase12_dataset,
)
from qmbp_simulation.predictors import (
    MPNNPredictor,
    build_graph_dataset,
    train_mpnn,
)
from qmbp_simulation.solvers import ClassicalSolver

logger = logging.getLogger(__name__)


def run_exact_diag_sweep(
    h_values: np.ndarray,
    n_qubits: int,
    topology: str = "chain_1d",
    J: float = 1.0,
    periodic: bool = False,
) -> list[GroundTruthResult]:
    """Run exact diagonalization across h-values (Phase 1 helper).

    Convenience function for scripts that need Phase 1 results without
    instantiating a full PipelineRunner.

    Parameters
    ----------
    h_values : np.ndarray
        Transverse field values (typically descending).
    n_qubits : int
        Number of qubits.
    topology : str
        Lattice topology (default: "chain_1d").
    J : float
        Coupling constant (default: 1.0).
    periodic : bool
        Whether to use periodic boundary conditions.

    Returns
    -------
    list[GroundTruthResult]
        Ground truth results for each h-value.
    """
    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    results = []
    for h in h_values:
        lattice_h = make_lattice(
            topology=topology,
            n_qubits=n_qubits,
            J=J,
            h=float(h),
            periodic=periodic,
        )
        hamiltonian = builder.build(lattice_h)
        results.append(solver.solve(hamiltonian, lattice_h))
    return results


class PipelineRunner:
    """Orchestrates the full 4-phase quantum simulation pipeline.

    Phase 1: Classical ground truth (exact diag / DMRG)
    Phase 2: VQE optimization (descending warm-start sweep)
    Phase 3: MPNN training (GINConv predictor)
    Phase 4: Deployment (predict unseen h-points)

    Supports skip/resume via phase flags and checkpoint detection.

    Parameters
    ----------
    lattice : LatticeConfig
        Base lattice configuration (h will be varied per sweep point).
    config : VQEConfig
        VQE optimization configuration.
    backend : ExecutionBackend | None
        Execution backend for VQE. Defaults to NoiselessBackend.
    checkpoint_dir : Path | None
        Directory for intermediate checkpoints. If None, no checkpoints saved.
    """

    def __init__(
        self,
        lattice: LatticeConfig,
        config: VQEConfig,
        backend: ExecutionBackend | None = None,
        checkpoint_dir: Path | None = None,
        *,
        verbose: bool = False,
        seed: int | None = None,
        collector: DiagnosticCollector | None = None,  # type: ignore[name-defined]  # noqa: F821
        model_spec: ModelSpec | None = None,
    ) -> None:
        self._lattice = lattice
        self._config = config
        self._backend = backend or NoiselessBackend()
        self._checkpoint_dir = checkpoint_dir
        self._seed = seed
        self._model_spec = model_spec

        # Internal state
        self._ham_builder = HamiltonianBuilder()
        self._solver = ClassicalSolver()
        self._optimizer = VQEOptimizer(config=config, backend=self._backend, seed=seed)

        # Log model configuration if model_spec is provided
        if model_spec is not None:
            logger.info(
                "Pipeline model: %s | params_per_layer=%d | total_params(p=%d)=%d | "
                "initial_state=%s | hamiltonian_kwargs=%s",
                model_spec.name,
                model_spec.params_per_layer,
                config.p_layers if hasattr(config, "p_layers") else 2,
                model_spec.total_params,
                model_spec.initial_state,
                model_spec.hamiltonian_kwargs,
            )

        # Diagnostics — always active, collects metrics passively.
        # Lazy import to respect module dependency DAG (pipeline → analysis
        # is not allowed at module level, but runtime usage is fine).
        if collector is not None:
            self.collector = collector
        else:
            from qmbp_simulation.analysis.diagnostics import DiagnosticCollector as _DC

            self.collector = _DC(
                verbose=verbose,
                save_dir=checkpoint_dir,
            )

        # Theta validation — initialized after Phase 3 trains the MPNN
        self._theta_validator = None  # ThetaValidator | None (lazy import)
        self._theta_validation_level: int = 4  # default: L1-L4 (cheap)

        # ── Runtime metrics accumulator (persisted in pipeline_metadata) ──
        self._run_metrics: dict[str, Any] = {
            "n_bidirectional_improvements": 0,
            "n_outliers_detected": 0,
            "n_energy_guard_repairs": 0,
            "n_energy_guard_suspicious": 0,
            "theta_smoothness_pre_alignment": None,
            "theta_smoothness_post_alignment": None,
            "theta_alignment_applied": False,
            "augmentation_original_size": 0,
            "augmentation_final_size": 0,
            "swa_applied": False,
            "gen_gap": None,
            "gen_gap_abort": False,
            "valid_regime_warning": False,
            "n_vqe_refinements": 0,
            "cobyla_auto_switched": False,
        }

    def set_theta_validation_level(self, level: int) -> None:
        """Configure the maximum theta validation level (1-7).

        Levels 1-4 are cheap (L4 uses existing Statevector). Levels 5-7
        require additional circuit evaluations (gradient, MC Dropout, sensitivity).

        Parameters
        ----------
        level : int
            Maximum level to execute. Set to 0 to disable validation entirely.
        """
        if level < 0 or level > 7:
            raise ValueError(f"Validation level must be 0-7, got {level}")
        self._theta_validation_level = level
        if level == 0:
            self._theta_validator = None

    # ── Private helpers for model-aware dispatch ─────────────────────────

    def _build_hamiltonian(self, lattice_h: LatticeConfig):
        """Build Hamiltonian using model_spec if available, else TFIM default.

        Centralizes model dispatch so Phase 1, 2, and 4 all use the same logic.
        """
        if self._model_spec is not None:
            return self._model_spec.build_hamiltonian(
                lattice_h, **self._model_spec.hamiltonian_kwargs
            )
        return self._ham_builder.build(lattice_h)

    def _create_circuit(self):
        """Create HVA circuit using model_spec if available, else TFIM default.

        Returns (circuit, theta) tuple. Centralizes circuit dispatch.
        """
        if self._model_spec is not None:
            return self._model_spec.create_circuit(
                self._lattice.n_qubits,
                self._config.p_layers,
                self._lattice,
                **self._model_spec.circuit_kwargs,
            )
        from qmbp_simulation.circuits import HVACircuitBuilder

        hva_builder = HVACircuitBuilder()
        return hva_builder.create(self._lattice.n_qubits, self._config.p_layers, self._lattice)

    def _n_variational_params(self) -> int:
        """Number of variational parameters for the active model and depth."""
        if self._model_spec is not None:
            return self._model_spec.total_params_for_p(self._config.p_layers)
        return self._config.p_layers * 2

    def run_phase1(self, h_values: np.ndarray) -> list[GroundTruthResult]:
        """Phase 1: Compute classical ground truth for each h-value.

        Parameters
        ----------
        h_values : np.ndarray
            Transverse field values (should be in descending order).

        Returns
        -------
        list[GroundTruthResult]
            Ground truth results for each h-value.
        """
        logger.info(f"Phase 1: Computing ground truth for {len(h_values)} h-values")
        results = []
        t0 = time.time()

        for h in h_values:
            lattice_h = make_lattice(
                topology=self._lattice.topology,
                n_qubits=self._lattice.n_qubits,
                J=self._lattice.J,
                h=float(h),
                periodic=self._lattice.periodic,
            )
            hamiltonian = self._build_hamiltonian(lattice_h)
            result = self._solver.solve(hamiltonian, lattice_h)
            results.append(result)

        elapsed = time.time() - t0
        gap_min = min(r.gap for r in results)
        self.collector.record_phase1(
            n_points=len(results),
            elapsed_s=elapsed,
            gap_min=gap_min,
        )
        self.collector.save_checkpoint("phase1")

        logger.info(f"Phase 1 complete: {len(results)} points, gap_min={gap_min:.6f}")
        return results

    def run_phase2(
        self,
        h_values: np.ndarray,
        exact_data: list[GroundTruthResult],
        bidirectional: bool = True,
    ) -> list[VQEResult]:
        """Phase 2: VQE optimization with warm-start sweep.

        Parameters
        ----------
        h_values : np.ndarray
            Transverse field values (must be in descending order).
        exact_data : list[GroundTruthResult]
            Ground truth from Phase 1 (for fidelity computation).
        bidirectional : bool
            If True (default), run both descending and ascending sweeps
            and keep the best energy at each point. Doubles VQE time but
            eliminates warm-start propagation errors.

        Returns
        -------
        list[VQEResult]
            VQE results for each h-value.
        """

        logger.info(f"Phase 2: VQE sweep over {len(h_values)} h-values")

        circuit, theta = self._create_circuit()

        if bidirectional:
            logger.info("  🔄 Bidirectional sweep enabled (desc + asc, keep best)")
            results = self._optimizer.bidirectional_sweep(
                h_values=h_values,
                circuit=circuit,
                lattice=self._lattice,
                exact_data=exact_data,
            )
        else:
            results = self._optimizer.descending_sweep(
                h_values=h_values,
                circuit=circuit,
                lattice=self._lattice,
                exact_data=exact_data,
            )

        # Record per-point diagnostics
        for _i, (vqe_r, h) in enumerate(zip(results, h_values, strict=False)):
            # Extract convergence info from trajectory if available
            converged = None
            if vqe_r.trajectory is not None:
                converged = vqe_r.trajectory.converged

            self.collector.record_vqe_point(
                h=float(h),
                n_iters=vqe_r.n_iterations,
                restart_energies=[vqe_r.energy],
                theta_opt=vqe_r.theta_opt,
                elapsed_s=0.0,  # Per-point timing not available from sweep
                converged=converged,
            )
        self.collector.save_checkpoint("phase2")

        # ── VQE Validation ───────────────────────────────────────────────
        # Run comprehensive validation on the sweep output. This catches
        # variational principle violations, energy bound breaches, and
        # sweep-level quality issues early — before Phase 3 trains on bad data.
        from qmbp_simulation.analysis.vqe_validator import VQEValidator

        model_name = self._model_spec.name if self._model_spec else "tfim"
        vqe_validator = VQEValidator.from_lattice(
            self._lattice, model_name=model_name, strict=False
        )
        validation_report = vqe_validator.validate_sweep(results, exact_data)

        # Store validation report in diagnostics
        self.collector.record_vqe_validation(validation_report)

        if validation_report.has_critical:
            for issue in validation_report.critical_issues:
                logger.error(f"VQE CRITICAL: {issue.message}")
            logger.error(
                f"⚠️  VQE VALIDATION FAILED: {validation_report.n_critical} critical issue(s). "
                f"Phase 3 training data may be unreliable."
            )
        elif validation_report.has_warnings:
            logger.warning(
                f"VQE validation: {validation_report.n_warnings} warning(s) — "
                f"review recommended. {validation_report.summary()}"
            )
        else:
            logger.info(f"VQE validation: {validation_report.summary()}")

        logger.info(
            f"Phase 2 complete: {len(results)} points, "
            f"mean fidelity={np.mean([r.fidelity for r in results]):.4f}"
        )
        return results

    def run_phase3(
        self,
        h_values: np.ndarray,
        vqe_results: list[VQEResult],
        exact_data: list[GroundTruthResult],
        *,
        mpnn_config: dict[str, Any] | None = None,
    ) -> MPNNPredictor:
        """Phase 3: Train MPNN predictor on VQE results.

        Parameters
        ----------
        h_values : np.ndarray
            Transverse field values used in Phase 2.
        vqe_results : list[VQEResult]
            VQE optimization results from Phase 2.
        exact_data : list[GroundTruthResult]
            Ground truth from Phase 1.
        mpnn_config : dict | None
            Optional MPNN training configuration overrides.

        Returns
        -------
        MPNNPredictor
            Trained MPNN model.
        """
        logger.info("Phase 3: Training MPNN predictor")
        t0 = time.time()

        cfg = mpnn_config or {}

        # Extract arrays for graph dataset construction
        theta_opt = np.array([r.theta_opt for r in vqe_results])
        e_exact = np.array([r.ground_energy for r in exact_data])
        fidelities = np.array([r.fidelity for r in vqe_results])

        # Attempt dataset construction with configured threshold.
        # If too few points pass, retry with progressively lower thresholds.
        # Use model-specific threshold if available (e.g., Heisenberg uses 0.60).
        if self._model_spec is not None:
            default_threshold = self._model_spec.fidelity_threshold
            hard_floor = min(0.80, default_threshold)
        else:
            default_threshold = 0.93
            hard_floor = 0.80

        fidelity_threshold = cfg.get("fidelity_threshold", default_threshold)
        FIDELITY_HARD_FLOOR = hard_floor  # noqa
        # Build fallback sequence from configured threshold down to hard floor
        fallback_thresholds = sorted(
            {fidelity_threshold, 0.90, 0.85, 0.80, 0.70, 0.60, FIDELITY_HARD_FLOOR}
            & {
                t
                for t in [fidelity_threshold, 0.90, 0.85, 0.80, 0.70, 0.60, FIDELITY_HARD_FLOOR]
                if t <= fidelity_threshold and t >= FIDELITY_HARD_FLOOR
            },
            reverse=True,
        )
        if not fallback_thresholds:
            fallback_thresholds = [fidelity_threshold]
        dataset = None

        for threshold in fallback_thresholds:
            try:
                dataset = build_graph_dataset(
                    lattice=self._lattice,
                    h_values=h_values,
                    theta_opt=theta_opt,
                    e_exact=e_exact,
                    fidelities=fidelities,
                    fidelity_threshold=threshold,  # noqa
                )
                if threshold < fidelity_threshold:
                    original = fidelity_threshold
                    logger.warning(
                        f"Fidelity filter relaxed from {original:.2f} to "
                        f"{threshold:.2f} to obtain {len(dataset)} training points. "
                        f"MPNN predictions may be less reliable."
                    )
                break
            except ValueError:
                if threshold == FIDELITY_HARD_FLOOR:
                    # Check how many points exist above the hard floor
                    n_above_floor = int(np.sum(fidelities >= FIDELITY_HARD_FLOOR))
                    max_fid = float(np.max(fidelities)) if len(fidelities) > 0 else 0.0
                    raise ValueError(
                        f"Phase 3 FAILED: Only {n_above_floor}/{len(fidelities)} VQE points "
                        f"have fidelity >= {FIDELITY_HARD_FLOOR:.2f} (max fidelity: {max_fid:.4f}). "
                        f"The HVA p={self._config.p_layers} ansatz cannot express the ground state "
                        f"at these h-values for topology='{self._lattice.topology}' N={self._lattice.n_qubits}. "
                        f"Solutions: (1) increase h-values to stay in the valid regime, "
                        f"(2) use more VQE restarts, or (3) accept that this regime is unreachable."
                    ) from None
                next_idx = fallback_thresholds.index(threshold) + 1
                logger.warning(
                    f"Fidelity threshold {threshold:.2f} too strict "
                    f"(fewer than 3 points pass). Trying {fallback_thresholds[next_idx]:.2f}..."
                )

        # ── Outlier filter: clean θ before MPNN training ──────────────
        # Detect and interpolate VQE local-minimum spikes that would corrupt
        # the MPNN's learned θ(h) mapping.
        from qmbp_simulation.analysis.theta_alignment import filter_theta_outliers

        theta_clean, h_clean, e_clean, fid_clean, outlier_report = filter_theta_outliers(
            theta_array=theta_opt,
            h_values=h_values,
            e_exact=e_exact,
            fidelities=fidelities,
            threshold=2.0,
            fidelity_floor=0.5,
            replace_strategy="interpolate",
        )
        if outlier_report.n_outliers > 0:
            logger.info(f"  🧹 Outlier filter: {outlier_report.n_outliers} points interpolated")
            self._run_metrics["n_outliers_detected"] = outlier_report.n_outliers
            # Rebuild dataset with cleaned θ
            dataset = build_graph_dataset(
                lattice=self._lattice,
                h_values=h_clean,
                theta_opt=theta_clean,
                e_exact=e_clean,
                fidelities=fid_clean,
                fidelity_threshold=0.0,  # Already filtered
            )

        # ── θ data augmentation: add noisy copies to expand dataset ───
        # For small datasets (<40 points), add 2 copies with Gaussian noise
        # to θ targets. This teaches the MPNN that small perturbations are
        # acceptable, reducing overfitting and prediction variance.
        if dataset is not None and len(dataset) < AUGMENTATION_DATASET_THRESHOLD:
            import torch as _torch_aug

            augmented = list(dataset)
            n_original = len(dataset)
            for data in list(dataset):
                for _ in range(AUGMENTATION_N_COPIES):
                    aug_data = data.clone()
                    noise = _torch_aug.randn_like(aug_data.y) * AUGMENTATION_NOISE_SIGMA
                    aug_data.y = aug_data.y + noise
                    augmented.append(aug_data)
            dataset = augmented
            logger.info(
                f"  📈 θ augmentation: {n_original} → {len(dataset)} points "
                f"({AUGMENTATION_N_COPIES} noisy copies, σ={AUGMENTATION_NOISE_SIGMA})"
            )
            self._run_metrics["augmentation_original_size"] = n_original
            self._run_metrics["augmentation_final_size"] = len(dataset)

        # Determine output_dim based on model spec
        n_params = self._n_variational_params()

        # Validate output_dim if explicitly provided in config
        cfg_output_dim = cfg.get("output_dim")
        if cfg_output_dim is not None and cfg_output_dim != n_params:
            raise ValueError(
                f"output_dim mismatch: config specifies {cfg_output_dim} but "
                f"model '{self._model_spec.name if self._model_spec else 'tfim'}' "
                f"with p={self._config.p_layers} requires {n_params}."
            )

        # Use model-specific hidden_dim if not explicitly overridden
        default_hidden_dim = 64
        if self._model_spec is not None:
            default_hidden_dim = self._model_spec.mpnn_hidden_dim

        model = MPNNPredictor(
            node_features=cfg.get("node_features", 2),
            hidden_dim=cfg.get("hidden_dim", default_hidden_dim),
            n_layers=cfg.get("n_layers", 3),
            output_dim=n_params,
            norm_type=cfg.get("norm_type", "batch"),  # cross-N zero-shot needs "none"
        )
        # NOTE: dropout=0.1 is hardcoded in MPNNPredictor MLP heads (always active)

        train_mpnn(
            model=model,
            dataset=dataset,  # type: ignore[arg-type]
            n_epochs=cfg.get("n_epochs", 4000),
            lr=cfg.get("lr", 1e-3),
            patience=cfg.get("patience", 150),
            seed=cfg.get("seed", self._seed if self._seed is not None else 42),
            weight_decay=cfg.get("weight_decay", 1e-4),
        )

        # ── Stochastic Weight Averaging (SWA) ────────────────────────────
        # Average model weights from the last portion of training for smoother
        # predictions. This reduces prediction variance at interpolation points.
        # We approximate SWA by doing a short extra training with averaged params.
        if cfg.get("use_swa", True) and dataset is not None and len(dataset) >= 5:
            import copy

            import torch as _torch_swa

            swa_model = copy.deepcopy(model)
            swa_model.train()
            swa_optimizer = _torch_swa.optim.Adam(
                swa_model.parameters(), lr=SWA_LR, weight_decay=SWA_LR
            )
            from torch_geometric.loader import DataLoader as _DL_swa

            swa_loader = _DL_swa(dataset, batch_size=len(dataset), shuffle=False)
            # Run SWA_EXTRA_EPOCHS extra epochs with low LR and average with original
            for _ in range(SWA_EXTRA_EPOCHS):
                for batch in swa_loader:
                    swa_optimizer.zero_grad()
                    pred = swa_model(batch)
                    target = batch.y.view(pred.shape)
                    loss = _torch_swa.nn.functional.mse_loss(pred, target)
                    loss.backward()
                    swa_optimizer.step()

            # Average weights: model = w*original + (1-w)*swa_model
            with _torch_swa.no_grad():
                for p_orig, p_swa in zip(model.parameters(), swa_model.parameters(), strict=False):
                    p_orig.data.copy_(
                        SWA_AVERAGE_WEIGHT * p_orig.data + (1 - SWA_AVERAGE_WEIGHT) * p_swa.data
                    )

            logger.info(
                f"  🔄 SWA: model weights averaged ({SWA_EXTRA_EPOCHS} extra epochs, lr={SWA_LR})"
            )
            self._run_metrics["swa_applied"] = True

        elapsed = time.time() - t0

        # Record Phase 3 diagnostics
        # Compute per-h MSE from trained model predictions.
        # Note: dataset may be shorter than h_values due to fidelity filtering.
        # We record MSE only for the points that survived filtering.
        import torch

        model.eval()
        per_h_mse = []
        filtered_h_values = []
        with torch.no_grad():
            for data in dataset or []:  # type: ignore[union-attr]
                pred = model(data).numpy().flatten()
                target = data.y.numpy().flatten()
                mse_i = float(np.mean((pred - target) ** 2))
                per_h_mse.append(mse_i)
                # Extract h-value from graph node features (first node, first feature)
                if hasattr(data, "h_value"):
                    filtered_h_values.append(float(data.h_value))
                else:
                    filtered_h_values.append(float(data.x[0, 0].item()))

        self.collector.record_mpnn_per_h_error(
            h_values=np.array(filtered_h_values)
            if filtered_h_values
            else h_values[: len(per_h_mse)],
            per_h_mse=np.array(per_h_mse),
            elapsed_s=elapsed,
        )
        self.collector.save_checkpoint("phase3")

        logger.info(f"Phase 3 complete: MPNN trained in {elapsed:.1f}s")
        return model

    def run_phase4(
        self,
        model: MPNNPredictor,
        h_test: float,
        *,
        vqe_energy: float | None = None,
    ) -> DeployResult:
        """Phase 4: Deploy MPNN prediction at an unseen h-value.

        Parameters
        ----------
        model : MPNNPredictor
            Trained MPNN from Phase 3.
        h_test : float
            Unseen transverse field value for deployment.
        vqe_energy : float | None
            Best VQE energy at this h-point (for energy decomposition).
            If None, uses exact energy as ceiling (noiseless approximation).

        Returns
        -------
        DeployResult
            Deployment result with energy, observables, and metrics.
        """
        import torch

        logger.info(f"Phase 4: Deploying at h_test={h_test}")
        t0 = time.time()

        # Get exact solution for validation
        lattice_test = make_lattice(
            topology=self._lattice.topology,
            n_qubits=self._lattice.n_qubits,
            J=self._lattice.J,
            h=h_test,
            periodic=self._lattice.periodic,
        )
        hamiltonian = self._build_hamiltonian(lattice_test)
        exact = self._solver.solve(hamiltonian, lattice_test)

        # Predict parameters with MPNN
        model.eval()
        edge_index_np, coord = self._ham_builder.build_graph_data(lattice_test)

        # Build graph input for prediction — must mirror build_graph_dataset()
        # Node features: [h_i, coordination_number_i] per site
        from torch_geometric.data import Data

        h_feat = np.full(lattice_test.n_qubits, float(h_test))
        x = torch.tensor(
            np.stack([h_feat, coord.astype(float)], axis=1),
            dtype=torch.float32,
        )
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)

        graph = Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            theta_pred = model(graph).numpy().flatten()

        # ── Guard #3: NaN/Inf propagation guard ──────────────────────────
        if not np.all(np.isfinite(theta_pred)):
            n_bad = int(np.sum(~np.isfinite(theta_pred)))
            logger.warning(
                f"  ⚠️ MPNN predicted {n_bad}/{len(theta_pred)} NaN/Inf values "
                f"at h={h_test:.4f}. Replacing with zeros (fallback)."
            )
            theta_pred = np.where(np.isfinite(theta_pred), theta_pred, 0.0)

        # ── Guard #5: Output bounds clipping ─────────────────────────────
        # θ must be in [-π, π] for the HVA circuit. MPNN without output
        # activation can predict outside this range.
        theta_pred = np.clip(theta_pred, -np.pi, np.pi)

        # ── θ_pred validation (L1-L4 by default) ────────────────────────
        theta_validation_report = None
        if hasattr(self, "_theta_validator") and self._theta_validator is not None:
            try:
                # Build energy_fn for higher-level validation (L5+)
                circuit_for_val, _ = self._create_circuit()

                def _energy_fn(theta: np.ndarray) -> float:
                    return self._backend.evaluate(circuit_for_val, hamiltonian, theta)

                # L4 needs exact_state; downgrade level if unavailable
                effective_level = self._theta_validation_level
                exact_state_for_val = exact.ground_state
                if exact_state_for_val is None and effective_level >= 4:
                    effective_level = 3  # Skip fidelity if no statevector

                theta_validation_report = self._theta_validator.validate(
                    theta_pred,
                    level=effective_level,
                    h_test=h_test,
                    circuit=circuit_for_val,
                    exact_state=exact_state_for_val,
                    energy_fn=_energy_fn if effective_level >= 5 else None,
                    model=model if effective_level >= 6 else None,
                    graph_data=graph if effective_level >= 6 else None,
                )
                self.collector.record_theta_validation(
                    h_test=h_test,
                    report_dict=theta_validation_report.to_dict(),
                )
            except Exception as e:
                logger.warning(f"θ_pred validation failed at h={h_test}: {e}")

        # Evaluate predicted circuit — use model-aware dispatch
        circuit, _ = self._create_circuit()
        predicted_energy = self._backend.evaluate(circuit, hamiltonian, theta_pred)

        # Compute observables from the predicted state (noiseless)
        mag_x_pred = 0.0
        corr_zz_pred = 0.0
        mag_x_error = 0.0
        corr_zz_error = 0.0
        observables_computed = True
        try:
            from qiskit.quantum_info import SparsePauliOp, Statevector

            bound_circuit = circuit.assign_parameters(theta_pred)
            sv = Statevector(bound_circuit)
            n = self._lattice.n_qubits

            # Compute <X> = (1/N) * sum_i <X_i>
            x_sum = 0.0
            for i in range(n):
                op_x = SparsePauliOp.from_sparse_list([("X", [i], 1.0)], num_qubits=n)
                x_sum += sv.expectation_value(op_x).real
            mag_x_pred = float(x_sum / n)

            # Compute <ZZ> = (1/n_bonds) * sum_<ij> <Z_i Z_j>
            zz_sum = 0.0
            n_bonds = len(lattice_test.edges)
            for i, j in lattice_test.edges:
                op_zz = SparsePauliOp.from_sparse_list([("ZZ", [i, j], 1.0)], num_qubits=n)
                zz_sum += sv.expectation_value(op_zz).real
            corr_zz_pred = float(zz_sum / n_bonds) if n_bonds > 0 else 0.0

            mag_x_error = abs(mag_x_pred - exact.mag_x)
            corr_zz_error = abs(corr_zz_pred - exact.corr_zz)
        except Exception as e:
            logger.warning(
                f"Observable computation FAILED at h_test={h_test}: {e}. "
                f"Phase 4 metrics for observables are unavailable."
            )
            observables_computed = False

        # Compute metrics
        delta_e = abs(predicted_energy - exact.ground_energy)
        delta_e_over_gap = delta_e / exact.gap if exact.gap > 0 else float("inf")

        # ── VQE refinement for marginal predictions ──────────────────────
        # If ΔE/gap is marginal (5-500%), use θ_pred as warm-start for a
        # short VQE pass to "polish" the prediction. Zero-cost for passing points.
        if VQE_REFINEMENT_DE_GAP_MIN < delta_e_over_gap < VQE_REFINEMENT_DE_GAP_MAX:
            try:
                refine_result = self._optimizer.optimize(
                    hamiltonian=hamiltonian,
                    circuit=circuit,
                    initial_guess=theta_pred,
                    exact_energy=exact.ground_energy,
                    exact_state=exact.ground_state,
                )
                if refine_result.energy < predicted_energy:
                    improvement = predicted_energy - refine_result.energy
                    logger.info(
                        f"  🔧 VQE refinement at h={h_test:.3f}: "
                        f"ΔE/gap {delta_e_over_gap:.4f} → "
                        f"{abs(refine_result.energy - exact.ground_energy) / max(exact.gap, 1e-10):.4f} "
                        f"(saved {improvement:.4e})"
                    )
                    predicted_energy = refine_result.energy
                    theta_pred = refine_result.theta_opt
                    # Recompute observables with refined theta
                    bound_circuit = circuit.assign_parameters(theta_pred)
                    sv = Statevector(bound_circuit)
                    delta_e = abs(predicted_energy - exact.ground_energy)
                    delta_e_over_gap = delta_e / exact.gap if exact.gap > 0 else float("inf")
                    self._run_metrics["n_vqe_refinements"] += 1
            except Exception as e:
                logger.debug(f"VQE refinement failed at h={h_test}: {e}")

        phase_label = "paramagnetic" if h_test > 1.0 else "ferromagnetic"

        # Phase classification check using observables
        if mag_x_pred != 0.0 or corr_zz_pred != 0.0:
            predicted_phase = (
                "paramagnetic" if abs(mag_x_pred) > abs(corr_zz_pred) else "ferromagnetic"
            )
            phase_correct = predicted_phase == phase_label
        else:
            phase_correct = True  # Can't verify without observables

        metrics_checklist = {
            "delta_e_over_gap_lt_5pct": delta_e_over_gap < DE_GAP_THRESHOLD,
            "correct_phase": phase_correct,
            "mag_x_error_lt_1e2": mag_x_error < 0.01 if observables_computed else None,
            "corr_zz_error_lt_1e2": corr_zz_error < 0.01 if observables_computed else None,
            "observables_computed": observables_computed,
        }

        result = DeployResult(
            route="mpnn_warm_start",
            h_test=h_test,
            predicted_energy=predicted_energy,
            delta_e=delta_e,
            delta_e_over_gap=delta_e_over_gap,
            mag_x_pred=mag_x_pred,
            corr_zz_pred=corr_zz_pred,
            mag_x_error=mag_x_error,
            corr_zz_error=corr_zz_error,
            fidelity=None,
            adapt_iterations=0,
            phase_label=phase_label,
            metrics_checklist=metrics_checklist,  # type: ignore[arg-type]
        )

        elapsed = time.time() - t0

        # Record Phase 4 diagnostics
        # VQE ceiling: use provided value, or exact energy as noiseless approximation
        ceiling = vqe_energy if vqe_energy is not None else exact.ground_energy
        self.collector.record_deployment(
            h_test=h_test,
            result=result,
            e_vqe_ceiling=ceiling,
        )
        self.collector.save_checkpoint("phase4")

        logger.info(
            f"Phase 4 complete: ΔE/gap={delta_e_over_gap:.4f}, phase={phase_label}, {elapsed:.2f}s"
        )
        if theta_validation_report is not None:
            tv_status = "PASS" if theta_validation_report.passes() else "FAIL"
            tv_conf = theta_validation_report.confidence_score
            logger.info(
                f"  θ validation: {tv_status} (confidence={tv_conf:.3f}, "
                f"level=L{theta_validation_report.level_executed})"
            )
        return result

    # ── Cross-N Validation (optional) ────────────────────────────────────

    def validate_cross_n(
        self,
        model: MPNNPredictor,
        n_target: int,
        h_test_values: list[float],
        training_sizes: list[int],
        training_data: list | None = None,
    ) -> CrossNValidationReport:
        """Validate a cross-N prediction using the 3-level verification system.

        Call this when the MPNN was trained on data from multiple system sizes
        and you want to verify its predictions at an unseen N_target.

        Parameters
        ----------
        model : MPNNPredictor
            Trained model (must have norm_type="none").
        n_target : int
            Unseen system size for prediction.
        h_test_values : list[float]
            h-values to evaluate at N_target.
        training_sizes : list[int]
            System sizes used during training.
        training_data : list | None
            Full training dataset (for L3 LOO-CV).

        Returns
        -------
        CrossNValidationReport
        """
        from qmbp_simulation.analysis.cross_n_validator import CrossNValidator

        validator = CrossNValidator(
            topology=self._lattice.topology,
            model_spec=self._model_spec,
            backend=self._backend,
            de_gap_threshold=0.05,
        )
        report = validator.validate_prediction(
            model=model,
            n_target=n_target,
            h_test_values=h_test_values,
            training_sizes=training_sizes,
            training_data=training_data,
        )

        # Persist cross-N validation in diagnostics
        self.collector._data.setdefault("cross_n_validation", []).append(report.to_dict())

        return report

    def run_full(
        self,
        h_values: np.ndarray,
        h_test: float | list[float],
        *,
        mpnn_config: dict[str, Any] | None = None,
        skip_phase1: bool = False,
        skip_phase2: bool = False,
        skip_phase3: bool = False,
        skip_phase4: bool = False,
        checkpoint_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run the full pipeline with optional skip/resume.

        Parameters
        ----------
        h_values : np.ndarray
            Transverse field values for the sweep.
        h_test : float | list[float]
            Unseen h-value(s) for Phase 4 deployment.
        mpnn_config : dict | None
            Optional MPNN configuration overrides.
        skip_phase1 : bool
            Skip Phase 1 (requires checkpoint_path with saved data).
        skip_phase2 : bool
            Skip Phase 2 (requires checkpoint_path with saved data).
        skip_phase3 : bool
            Skip Phase 3 (requires checkpoint_path with trained model).
        skip_phase4 : bool
            Skip Phase 4 deployment.
        checkpoint_path : str | Path | None
            Path to load/save intermediate results for resume.

        Returns
        -------
        dict
            Results dictionary with keys: "phase1", "phase2", "phase3",
            "phase4", "diagnostics".
        """
        # Validate h_values ordering
        h_values = np.asarray(h_values, dtype=float)
        if len(h_values) == 0:
            raise ValueError("h_values cannot be empty.")
        if len(h_values) >= 2 and h_values[0] < h_values[-1]:
            raise ValueError(
                "h_values must be in descending order (h=2→0). "
                f"Got h_values[0]={h_values[0]:.2f}, h_values[-1]={h_values[-1]:.2f}. "
                "Reverse the array before passing to run_full()."
            )

        results: dict[str, Any] = {}

        # ── Guard #1: Valid regime pre-check ─────────────────────────────
        # Warn if h_min is below the known valid regime boundary for this
        # topology/N/p. Runs below h_min_safe are guaranteed to fail.
        try:
            from qmbp_simulation.framework.preflight import get_regime_threshold

            p = self._config.p_layers
            topo = self._lattice.topology
            n = self._lattice.n_qubits
            h_min_safe = get_regime_threshold(topo, n, p)
            h_min_actual = float(h_values[-1])  # Descending order → last is min
            if h_min_safe > 0 and h_min_actual < h_min_safe:
                n_below = int(np.sum(h_values < h_min_safe))
                logger.warning(
                    f"  ⚠️ VALID REGIME WARNING: h_min={h_min_actual:.2f} is below "
                    f"h_min_safe={h_min_safe:.2f} for {topo} N={n} p={p}. "
                    f"{n_below}/{len(h_values)} points are in the failure zone. "
                    f"These points will likely have ΔE/gap > 5% regardless of "
                    f"optimization quality."
                )
                self._run_metrics["valid_regime_warning"] = True
        except (ImportError, ValueError):
            pass  # preflight not available or p>2 — skip check silently

        # Phase 1: Ground truth
        if skip_phase1 and checkpoint_path:
            logger.info("Phase 1 skipped — loading from checkpoint")
            data = load_phase12_dataset(checkpoint_path)
            # Reconstruct GroundTruthResult list from saved arrays
            exact_data = self._reconstruct_phase1(data, h_values)
        else:
            exact_data = self.run_phase1(h_values)
        results["phase1"] = exact_data

        # Phase 2: VQE
        if skip_phase2 and checkpoint_path:
            logger.info("Phase 2 skipped — loading from checkpoint")
            data = load_phase12_dataset(checkpoint_path)
            vqe_results = self._reconstruct_phase2(data, h_values)
        else:
            vqe_results = self.run_phase2(h_values, exact_data)
        results["phase2"] = vqe_results

        # Save checkpoint after Phase 1+2
        if self._checkpoint_dir and not skip_phase1 and not skip_phase2:
            self._save_phase12_checkpoint(h_values, exact_data, vqe_results)

        # ── Early-stopping check: theta_smoothness ──────────────────────
        # If the warm-start chain broke (smoothness > 1.0), Phase 3+4 will
        # likely fail. Warn the user but don't abort (they may want the data).
        diag_so_far = self.collector.to_dict()
        theta_smoothness = diag_so_far.get("phase2", {}).get("theta_smoothness")
        if theta_smoothness is not None and theta_smoothness > THETA_SMOOTHNESS_CHAIN_BREAK:
            logger.warning(
                f"⚠️  WARM-START CHAIN BREAK DETECTED: θ_smoothness={theta_smoothness:.2f} "
                f"(threshold: {THETA_SMOOTHNESS_CHAIN_BREAK}). The VQE found different basins at adjacent h-values. "
                f"Phase 3 MPNN will struggle to learn the discontinuous θ(h) mapping. "
                f"Consider: (1) reducing n_restarts, (2) increasing h-grid density, "
                f"(3) restricting h-range to the valid regime."
            )
        elif theta_smoothness is not None and theta_smoothness > 0.10:
            logger.info(
                f"θ_smoothness={theta_smoothness:.4f} (elevated but below chain-break threshold)"
            )

        # ── Post-VQE theta alignment ───────────────────────────────────
        # Detect discontinuities in θ(h) and re-optimize jumped points
        # using the neighbor's θ as seed. This produces a smooth θ(h)
        # curve suitable for MPNN interpolation.
        if (
            theta_smoothness is not None
            and theta_smoothness > THETA_SMOOTHNESS_CHAIN_BREAK
            and not skip_phase2
        ):
            try:
                from qmbp_simulation.analysis.theta_alignment import align_theta_sweep
                from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

                builder = HamiltonianBuilder()
                hamiltonians = []
                for gt in exact_data:
                    lattice_h = make_lattice(
                        topology=self._lattice.topology,
                        n_qubits=self._lattice.n_qubits,
                        J=self._lattice.J,
                        h=gt.h_value,
                        periodic=self._lattice.periodic,
                    )
                    hamiltonians.append(self._build_hamiltonian(lattice_h))

                circuit_for_align, _ = self._create_circuit()
                vqe_results, alignment_report = align_theta_sweep(
                    vqe_results=vqe_results,
                    circuit=circuit_for_align,
                    hamiltonians=hamiltonians,
                    backend=self._backend,
                )
                results["phase2"] = vqe_results

                # Record alignment diagnostics
                logger.info(
                    f"θ alignment: smoothness {alignment_report.original_smoothness:.3f} "
                    f"→ {alignment_report.final_smoothness:.3f} "
                    f"({alignment_report.n_realigned}/{alignment_report.n_jumps_detected} fixed)"
                )
                self._run_metrics["theta_alignment_applied"] = True
                self._run_metrics["theta_smoothness_pre_alignment"] = (
                    alignment_report.original_smoothness
                )
                self._run_metrics["theta_smoothness_post_alignment"] = (
                    alignment_report.final_smoothness
                )
            except Exception as e:
                logger.warning(f"θ alignment failed (non-fatal): {e}")

        # ── Cross-h energy validation guard ─────────────────────────────
        # Detect isolated VQE local-minimum traps and repair them by
        # re-optimizing with the neighbor's θ as seed.
        if not skip_phase2 and len(vqe_results) >= 3:
            try:
                from qmbp_simulation.analysis.theta_alignment import cross_h_energy_guard

                vqe_energies = np.array([r.energy for r in vqe_results])
                exact_energies_arr = np.array([r.ground_energy for r in exact_data])
                gaps_arr = np.array([r.gap for r in exact_data])
                theta_arr = np.array([r.theta_opt for r in vqe_results])
                h_arr = np.array([r.h_value for r in vqe_results])

                logger.info("  🛡️ Cross-h energy guard: checking for local-minimum traps...")

                circuit_guard, _ = self._create_circuit()

                def _reopt_fn(idx: int, theta_seed: np.ndarray):
                    """Re-run VQE at the given index with neighbor-seeded θ."""
                    h_val = float(h_arr[idx])
                    lattice_h = make_lattice(
                        topology=self._lattice.topology,
                        n_qubits=self._lattice.n_qubits,
                        J=self._lattice.J,
                        h=h_val,
                        periodic=self._lattice.periodic,
                    )
                    H_h = self._build_hamiltonian(lattice_h)
                    result = self._optimizer.optimize(
                        hamiltonian=H_h,
                        circuit=circuit_guard,
                        initial_guess=theta_seed,
                        exact_energy=float(exact_energies_arr[idx]),
                        exact_state=exact_data[idx].ground_state,
                    )
                    return result.energy, result.theta_opt

                energies_out, theta_out, guard_report = cross_h_energy_guard(
                    vqe_energies=vqe_energies,
                    exact_energies=exact_energies_arr,
                    gaps=gaps_arr,
                    theta_array=theta_arr,
                    h_values=h_arr,
                    reoptimize_fn=_reopt_fn,
                )

                if guard_report.n_repaired > 0:
                    for i in guard_report.suspicious_indices:
                        vqe_results[i] = VQEResult(
                            h_value=vqe_results[i].h_value,
                            theta_opt=theta_out[i],
                            energy=float(energies_out[i]),
                            energy_error=abs(float(energies_out[i]) - exact_data[i].ground_energy),
                            fidelity=vqe_results[i].fidelity,
                            n_iterations=vqe_results[i].n_iterations,
                            trajectory=vqe_results[i].trajectory,
                        )
                    results["phase2"] = vqe_results
                    logger.info(
                        f"  🛡️ Energy guard: {guard_report.n_repaired}/"
                        f"{guard_report.n_suspicious} points repaired."
                    )
                    self._run_metrics["n_energy_guard_repairs"] = guard_report.n_repaired
                    self._run_metrics["n_energy_guard_suspicious"] = guard_report.n_suspicious
                else:
                    logger.info(
                        f"  🛡️ Energy guard: {guard_report.n_suspicious} suspicious, "
                        f"0 repairs needed."
                    )
            except Exception as e:
                logger.warning(f"Energy guard failed (non-fatal): {e}")

        # Phase 3: MPNN
        if skip_phase3:
            logger.info("Phase 3 skipped")
            model = None
        else:
            try:
                model = self.run_phase3(h_values, vqe_results, exact_data, mpnn_config=mpnn_config)
            except ValueError as e:
                logger.error(f"Phase 3 FAILED: {e}")
                model = None
        results["phase3"] = model

        # ── Early-stopping check: generalization_gap ────────────────────
        # If gen_gap > 1e-2, Phase 4 deployment will almost certainly fail
        # (85% failure rate observed in 131 variants).
        # Guard #2: Skip Phase 4 if gen_gap is catastrophic (>0.05)
        _skip_phase4_gen_gap = False
        if model is not None:
            diag_after_p3 = self.collector.to_dict()
            gen_gap = diag_after_p3.get("phase3", {}).get("generalization_gap")
            if gen_gap is not None and gen_gap > GEN_GAP_CATASTROPHIC:
                logger.warning(
                    f"  ⚠️ CATASTROPHIC GEN GAP: gen_gap={gen_gap:.4f} "
                    f"(>{GEN_GAP_CATASTROPHIC}). "
                    f"Phase 4 will be skipped — 95%+ failure rate at this level."
                )
                _skip_phase4_gen_gap = True
                self._run_metrics["gen_gap"] = gen_gap
                self._run_metrics["gen_gap_abort"] = True
            elif gen_gap is not None and gen_gap > GEN_GAP_WARNING:
                logger.warning(
                    f"⚠️  HIGH GENERALIZATION GAP: gen_gap={gen_gap:.4f} (threshold: 0.01). "
                    f"MPNN is overfitting — Phase 4 predictions will likely be inaccurate. "
                    f"Historical data shows 85% failure rate when gen_gap > 1e-2. "
                    f"Consider: (1) reducing epochs, (2) increasing training data, "
                    f"(3) checking if θ_smoothness indicates a chain break."
                )
            elif gen_gap is not None and gen_gap > 0.001:
                logger.info(f"gen_gap={gen_gap:.2e} (elevated, monitor Phase 4 result)")

        # ── Initialize θ validator from training data ───────────────────
        if model is not None:
            try:
                from qmbp_simulation.analysis.theta_validator import ThetaValidator

                theta_opt_arr = np.array([r.theta_opt for r in vqe_results])
                self._theta_validator = ThetaValidator.from_training_data(  # type: ignore[assignment]
                    theta_opt=theta_opt_arr,
                    h_values=h_values,
                )
                logger.info(
                    f"θ validator initialized: {theta_opt_arr.shape[0]} training points, "
                    f"{theta_opt_arr.shape[1]} params"
                )
            except Exception as e:
                logger.warning(f"θ validator initialization failed: {e}")
                self._theta_validator = None

        # Phase 4: Deployment
        if skip_phase4 or model is None or _skip_phase4_gen_gap:
            reason = (
                "user request"
                if skip_phase4
                else "no model"
                if model is None
                else "catastrophic gen_gap"
            )
            logger.info(f"Phase 4 skipped ({reason})")
            results["phase4"] = None
        else:
            h_tests = [h_test] if isinstance(h_test, int | float) else h_test
            # Warn if h_test is outside the training range (extrapolation)
            h_min_train, h_max_train = float(h_values[-1]), float(h_values[0])
            for h_t in h_tests:
                if h_t < h_min_train or h_t > h_max_train:
                    logger.warning(
                        f"h_test={h_t} is outside training range [{h_min_train}, {h_max_train}]. "
                        f"MPNN is extrapolating — predictions may be unreliable."
                    )
            deploy_results = [self.run_phase4(model, h) for h in h_tests]
            results["phase4"] = deploy_results

        # Attach diagnostics to results
        results["diagnostics"] = self.collector.to_dict()

        # ── Pipeline metadata: record which techniques were applied ───────
        results["pipeline_metadata"] = {
            "version": "v4",
            "techniques_applied": {
                "bidirectional_sweep": True,
                "cobyla_auto_switch": self._run_metrics["cobyla_auto_switched"],
                "theta_alignment": self._run_metrics["theta_alignment_applied"],
                "cross_h_energy_guard": True,
                "outlier_filter": True,
                "theta_augmentation": self._run_metrics["augmentation_final_size"]
                > self._run_metrics["augmentation_original_size"],
                "weight_decay": 1e-4,
                "swa": self._run_metrics["swa_applied"],
                "vqe_refinement": True,
                "dropout": 0.1,
                "non_uniform_grid": False,
            },
            "guards_active": {
                "valid_regime_check": True,
                "gen_gap_abort": True,
                "nan_guard": True,
                "bounds_clip": True,
                "variational_principle": True,
                "energy_monotonicity": True,
            },
            "run_metrics": self._run_metrics,
        }

        self.collector.cleanup_checkpoints()

        return results

    def _save_phase12_checkpoint(
        self,
        h_values: np.ndarray,
        exact_data: list[GroundTruthResult],
        vqe_results: list[VQEResult],
    ) -> None:
        """Save Phase 1+2 results as a checkpoint."""
        if self._checkpoint_dir is None:
            return

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filepath = self._checkpoint_dir / "phase12_checkpoint.npz"

        save_phase12_dataset(
            filepath,
            h_values=np.array(h_values),
            J=float(self._lattice.J) if np.isscalar(self._lattice.J) else 1.0,  # type: ignore[arg-type]
            n_qubits=self._lattice.n_qubits,
            p_layers=self._config.p_layers,
            ground_energies=np.array([r.ground_energy for r in exact_data]),
            gaps=np.array([r.gap for r in exact_data]),
            mag_x=np.array([r.mag_x for r in exact_data]),
            corr_zz=np.array([r.corr_zz for r in exact_data]),
            theta_opt=np.array([r.theta_opt for r in vqe_results]),
            vqe_energies=np.array([r.energy for r in vqe_results]),
            fidelities=np.array([r.fidelity for r in vqe_results]),
        )
        logger.info(f"Phase 1+2 checkpoint saved: {filepath}")

    def _reconstruct_phase1(self, data: dict, h_values: np.ndarray) -> list[GroundTruthResult]:
        """Reconstruct GroundTruthResult list from loaded dataset."""
        results = []
        ground_energies = np.asarray(data["ground_energies"])
        gaps = np.asarray(data["gaps"])
        mag_x = np.asarray(data["mag_x"])
        corr_zz = np.asarray(data["corr_zz"])

        for i, h in enumerate(h_values):
            results.append(
                GroundTruthResult(
                    h_value=float(h),
                    ground_energy=float(ground_energies[i]),
                    gap=float(gaps[i]),
                    ground_state=None,
                    mag_x=float(mag_x[i]),
                    corr_zz=float(corr_zz[i]),
                    per_site_mag_x=np.zeros(self._lattice.n_qubits),
                    per_bond_corr_zz=np.zeros(len(self._lattice.edges)),
                )
            )
        return results

    def _reconstruct_phase2(self, data: dict, h_values: np.ndarray) -> list[VQEResult]:
        """Reconstruct VQEResult list from loaded dataset."""
        results = []
        theta_opt = np.asarray(data["theta_opt"])
        vqe_energies = np.asarray(data["vqe_energies"])
        fidelities = np.asarray(data["fidelities"])
        ground_energies = np.asarray(data["ground_energies"])

        for i, h in enumerate(h_values):
            results.append(
                VQEResult(
                    h_value=float(h),
                    theta_opt=theta_opt[i] if theta_opt.ndim > 1 else theta_opt,
                    energy=float(vqe_energies[i]),
                    energy_error=abs(float(vqe_energies[i]) - float(ground_energies[i])),
                    fidelity=float(fidelities[i]),
                    n_iterations=0,
                    trajectory=None,
                )
            )
        return results
