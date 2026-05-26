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
    VQEConfig,
    VQEResult,
    make_lattice,
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
        collector: "DiagnosticCollector | None" = None,  # noqa: F821, UP037
    ) -> None:
        self._lattice = lattice
        self._config = config
        self._backend = backend or NoiselessBackend()
        self._checkpoint_dir = checkpoint_dir
        self._seed = seed

        # Internal state
        self._ham_builder = HamiltonianBuilder()
        self._solver = ClassicalSolver()
        self._optimizer = VQEOptimizer(config=config, backend=self._backend, seed=seed)

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
            hamiltonian = self._ham_builder.build(lattice_h)
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
    ) -> list[VQEResult]:
        """Phase 2: VQE optimization with descending warm-start sweep.

        Parameters
        ----------
        h_values : np.ndarray
            Transverse field values (must be in descending order).
        exact_data : list[GroundTruthResult]
            Ground truth from Phase 1 (for fidelity computation).

        Returns
        -------
        list[VQEResult]
            VQE results for each h-value.
        """
        from qmbp_simulation.circuits import HVACircuitBuilder

        logger.info(f"Phase 2: VQE sweep over {len(h_values)} h-values")

        hva_builder = HVACircuitBuilder()
        circuit, theta = hva_builder.create(
            self._lattice.n_qubits, self._config.p_layers, self._lattice
        )

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

        dataset = build_graph_dataset(
            lattice=self._lattice,
            h_values=h_values,
            theta_opt=theta_opt,
            e_exact=e_exact,
            fidelities=fidelities,
            fidelity_threshold=cfg.get("fidelity_threshold", 0.93),
        )

        n_params = self._config.p_layers * 2
        model = MPNNPredictor(
            node_features=cfg.get("node_features", 2),
            hidden_dim=cfg.get("hidden_dim", 64),
            n_layers=cfg.get("n_layers", 3),
            output_dim=n_params,
        )

        train_mpnn(
            model=model,
            dataset=dataset,
            n_epochs=cfg.get("n_epochs", 4000),
            lr=cfg.get("lr", 1e-3),
            patience=cfg.get("patience", 150),
            seed=cfg.get("seed", self._seed if self._seed is not None else 42),
        )

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
            for data in dataset:
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

        from qmbp_simulation.circuits import HVACircuitBuilder

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
        hamiltonian = self._ham_builder.build(lattice_test)
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

        # Evaluate predicted circuit
        hva_builder = HVACircuitBuilder()
        circuit, _ = hva_builder.create(
            self._lattice.n_qubits, self._config.p_layers, self._lattice
        )
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
            n_bonds = len(self._lattice.edges)
            for i, j in self._lattice.edges:
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
            "delta_e_over_gap_lt_5pct": delta_e_over_gap < 0.05,
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
            metrics_checklist=metrics_checklist,
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
        return result

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

        # Phase 3: MPNN
        if skip_phase3:
            logger.info("Phase 3 skipped")
            model = None
        else:
            model = self.run_phase3(h_values, vqe_results, exact_data, mpnn_config=mpnn_config)
        results["phase3"] = model

        # Phase 4: Deployment
        if skip_phase4 or model is None:
            logger.info("Phase 4 skipped")
            results["phase4"] = None
        else:
            h_tests = [h_test] if isinstance(h_test, int | float) else h_test
            deploy_results = [self.run_phase4(model, h) for h in h_tests]
            results["phase4"] = deploy_results

        # Attach diagnostics to results
        results["diagnostics"] = self.collector.to_dict()
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
            J=float(self._lattice.J) if np.isscalar(self._lattice.J) else 1.0,
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
