#!/usr/bin/env python3
"""Parametric QPU Deployment — N-agnostic hardware execution with full observability.

A clean, parametrized deployment orchestrator that supports any N, topology,
and model. Unlike run_ibm_deployment.py (hardcoded N=10 heavy_hex), this script
accepts all physics parameters via CLI or programmatic API (for notebook use).

Design principles:
    1. ALL parameters are CLI arguments or constructor kwargs — zero hardcoded physics.
    2. Every metric and intermediate result is logged to a structured JSON envelope.
    3. The DeploymentConfig dataclass is the single source of truth for all settings.
    4. Notebook-friendly: `from run_parametric_deployment import ParametricDeployment`
       gives you a clean API with sensible defaults and dict output.

Phases:
    0. Preflight: Cost estimation, circuit feasibility, calibration check
    1. Ground Truth: DMRG/exact diag for reference energies
    2. MPNN Training: VQE data generation + MPNN training (or checkpoint load)
    3. QPU Execution: PEA-ZNE mitigated energy measurement
    4. Analysis: Compare QPU vs MPS(χ=64) vs exact, compute all metrics

Usage (CLI):
    # N=20 heavy_hex (requires IBM credentials)
    python scripts/experiment_runners/hardware/run_parametric_deployment.py \
        --n-qubits 20 --topology heavy_hex --p-layers 1 \
        --h-test 4.0 3.5 3.0 --h-train 4.5 4.0 3.5 3.0 2.5

    # Dry run (cost estimation only, no QPU)
    python scripts/experiment_runners/hardware/run_parametric_deployment.py \
        --n-qubits 20 --dry-run

    # FakeTorino rehearsal mode (no QPU credits, local noise simulation)
    python scripts/experiment_runners/hardware/run_parametric_deployment.py \
        --n-qubits 20 --mode fake_backend

Usage (Notebook):
    from scripts.experiment_runners.hardware.run_parametric_deployment import (
        DeploymentConfig, ParametricDeployment
    )

    config = DeploymentConfig(
        n_qubits=20,
        topology="heavy_hex",
        h_test=[4.0, 3.5, 3.0],
        h_train=[4.5, 4.0, 3.5, 3.0, 2.5],
        mode="fake_backend",  # or "hardware" for real QPU
    )
    deployment = ParametricDeployment(config)
    results = deployment.run()  # Returns dict with all metrics
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from qmbp_simulation.framework.runner_base import resolve_project_root

_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration Dataclass — single source of truth
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DeploymentConfig:
    """Complete deployment configuration — no hardcoded values anywhere.

    All physics, execution, and MPNN parameters are specified here.
    For notebook use, construct directly. For CLI, use from_cli().
    """

    # ── Physics parameters ─────────────────────────────────────────────────
    n_qubits: int = 20
    topology: str = "heavy_hex"
    model: str = "tfim"
    p_layers: int = 1
    h_test: list[float] = field(default_factory=lambda: [4.0, 3.5, 3.0])
    h_train: list[float] = field(default_factory=lambda: [4.5, 4.25, 3.75, 3.25, 2.75, 2.5])

    # ── Execution parameters ───────────────────────────────────────────────
    mode: str = "fake_backend"  # "hardware", "fake_backend", "dry_run"
    backend_name: str = "ibm_kingston"
    shots: int = 16384
    n_layouts: int = 3
    amplifier: str = "pea"
    pea_preset: str = "balanced"
    spsa_enabled: bool = False  # Off by default at N>10 (budget safety)
    job_timeout_s: int = 900
    seed: int = 42

    # ── MPS comparison parameters ──────────────────────────────────────────
    mps_chi_values: list[int] = field(default_factory=lambda: [64, 128, 256])
    mps_chi_for_training: int = 256  # Use high chi for MPNN training data

    # ── MPNN parameters ────────────────────────────────────────────────────
    mpnn_hidden_dim: int = 128
    mpnn_n_layers: int = 3
    mpnn_epochs: int = 6000
    mpnn_lr: float = 1e-3
    mpnn_patience: int = 500
    vqe_maxiter: int = 500
    vqe_n_restarts: int = 3

    # ── Output ─────────────────────────────────────────────────────────────
    output_dir: str = "results/hardware/parametric"
    verbose: bool = False

    # ── Success thresholds ─────────────────────────────────────────────────
    de_gap_threshold: float = 0.05
    abort_de_gap_threshold: float = 0.10

    def validate(self) -> list[str]:
        """Validate configuration, return list of issues (empty = OK)."""
        issues = []
        if self.p_layers > 2:
            issues.append(f"p_layers={self.p_layers} > 2: violates depth constraint")
        if self.n_qubits < 4:
            issues.append(f"n_qubits={self.n_qubits} < 4: too small")
        if not self.h_test:
            issues.append("h_test is empty")
        if not self.h_train:
            issues.append("h_train is empty")
        # Check for data leakage: h_test should not be in h_train
        overlap = set(self.h_test) & set(self.h_train)
        if overlap:
            issues.append(
                f"Data leakage: h_test ∩ h_train = {overlap}. "
                f"Remove overlapping values from h_train."
            )
        if self.mode not in ("hardware", "fake_backend", "dry_run"):
            issues.append(f"Invalid mode: {self.mode}")
        return issues

    @classmethod
    def from_cli(cls, args) -> DeploymentConfig:
        """Construct from argparse Namespace."""
        return cls(
            n_qubits=args.n_qubits,
            topology=args.topology,
            model=args.model,
            p_layers=args.p_layers,
            h_test=args.h_test,
            h_train=args.h_train,
            mode=args.mode,
            backend_name=args.backend,
            shots=args.shots,
            n_layouts=args.n_layouts,
            amplifier=args.amplifier,
            pea_preset=args.pea_preset,
            spsa_enabled=not args.no_spsa,
            seed=args.seed,
            mps_chi_for_training=args.mps_chi_training,
            mpnn_epochs=args.mpnn_epochs,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Deployment Orchestrator — notebook-friendly API
# ═══════════════════════════════════════════════════════════════════════════════


class ParametricDeployment:
    """N-agnostic QPU deployment orchestrator.

    Notebook usage:
        config = DeploymentConfig(n_qubits=20, mode="fake_backend")
        dep = ParametricDeployment(config)
        results = dep.run()
        # results is a dict with all metrics, ready for analysis

    CLI usage:
        Handled by main() at bottom of file.
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self._results: dict[str, Any] = {}
        self._setup_logging()

    def _setup_logging(self):
        level = logging.DEBUG if self.config.verbose else logging.INFO
        logging.basicConfig(level=level, format="%(levelname)s: %(message)s", force=True)

    def run(self) -> dict[str, Any]:
        """Execute the full deployment pipeline. Returns consolidated results.

        Phases executed:
            0. Preflight (always)
            1. Ground Truth (DMRG/exact diag)
            2. MPNN Training (VQE at chi_max + train predictor)
            3. MPS Comparison (evaluate θ_pred at multiple chi)
            4. QPU/FakeTorino Execution (skip if dry_run)
            5. Analysis (QPU vs MPS vs exact comparison)

        Returns
        -------
        dict with keys: config, preflight, ground_truth, mpnn, mps_comparison,
                        qpu_results (if not dry_run), analysis, summary
        """
        t_total = time.time()
        logger.info("=" * 70)
        logger.info(f"  PARAMETRIC DEPLOYMENT: {self.config.topology} "
                    f"N={self.config.n_qubits} p={self.config.p_layers}")
        logger.info(f"  Mode: {self.config.mode}")
        logger.info(f"  h_test: {self.config.h_test}")
        logger.info(f"  h_train: {self.config.h_train}")
        logger.info("=" * 70)

        # Validate config
        issues = self.config.validate()
        if issues:
            for issue in issues:
                logger.error(f"  Config error: {issue}")
            return {"error": issues, "config": self._config_dict()}

        self._results["config"] = self._config_dict()

        # Phase 0: Preflight
        self._results["preflight"] = self._phase_preflight()

        if self.config.mode == "dry_run":
            self._results["summary"] = {"mode": "dry_run", "phases_executed": ["preflight"]}
            return self._results

        # Phase 1: Ground Truth
        self._results["ground_truth"] = self._phase_ground_truth()

        # Phase 2: MPNN Training
        self._results["mpnn"] = self._phase_mpnn_training()

        # Phase 3: MPS Comparison
        self._results["mps_comparison"] = self._phase_mps_comparison()

        # Phase 4: QPU/FakeBackend Execution
        if self.config.mode in ("hardware", "fake_backend"):
            self._results["qpu_results"] = self._phase_qpu_execution()

        # Phase 5: Analysis
        self._results["analysis"] = self._phase_analysis()

        # Summary
        elapsed = time.time() - t_total
        self._results["summary"] = self._build_summary(elapsed)

        # Save results
        self._save_results()

        return self._results

    def _config_dict(self) -> dict:
        """Serialize config to dict (JSON-safe)."""
        from dataclasses import asdict
        return asdict(self.config)


    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 0: Preflight — cost estimation + circuit feasibility
    # ═══════════════════════════════════════════════════════════════════════════

    def _phase_preflight(self) -> dict[str, Any]:
        """Estimate QPU cost and check circuit feasibility."""
        from experiments.helpers.scaling_utils import compute_transpilation_metrics

        from qmbp_simulation import make_lattice
        from qmbp_simulation.execution.hardware.config import HardwareConfig
        from qmbp_simulation.execution.hardware.preflight import (
            QPUThroughputProfile,
            SPSACostModel,
            estimate_qpu_cost_extended,
        )
        from qmbp_simulation.models.model_registry import get_model_spec

        cfg = self.config
        spec = get_model_spec(cfg.model)

        logger.info("\n  ── Phase 0: Preflight ──")

        # Build a reference circuit for metrics
        lattice_ref = make_lattice(cfg.topology, cfg.n_qubits, J=1.0, h=cfg.h_test[0])
        circuit, _ = spec.create_circuit(
            cfg.n_qubits, cfg.p_layers, lattice_ref, **spec.circuit_kwargs
        )

        # Transpilation metrics (uses FakeTorino if available, else skip)
        transpile_data = None
        try:
            from qiskit_ibm_runtime.fake_provider import FakeTorino
            fake_backend = FakeTorino()
            transpile_data = compute_transpilation_metrics(circuit, fake_backend)
            logger.info(
                f"    Circuit: {transpile_data['cx_count_pre_transpile']} CX → "
                f"{transpile_data['cx_count_post_transpile']} post-transpile, "
                f"depth_2q={transpile_data['depth_2q']}"
            )
        except ImportError:
            logger.warning("    FakeTorino not available — skipping transpile check")

        # Cost estimation
        hw_config = HardwareConfig(n_qubits=cfg.n_qubits, shots=cfg.shots, n_layouts=cfg.n_layouts)
        profile = QPUThroughputProfile.ibm_kingston()
        spsa_model = SPSACostModel.disabled() if not cfg.spsa_enabled else SPSACostModel()

        cx_count = transpile_data["cx_count_post_transpile"] if transpile_data else None
        depth = transpile_data["depth_total"] if transpile_data else None

        est = estimate_qpu_cost_extended(
            config=hw_config,
            n_h_points=len(cfg.h_test),
            include_spsa=cfg.spsa_enabled,
            circuit_depth=depth,
            cx_count=cx_count,
            profile=profile,
            spsa_model=spsa_model,
        )

        logger.info(f"    Cost estimate: {est.est_total_s:.0f}s expected "
                    f"({est.est_total_s / 60:.1f} min)")
        logger.info(f"    T1 budget ratio: {est.t1_budget_ratio:.3f}")
        logger.info(f"    SNR at critical: {est.snr_at_critical:.2f}")

        # Feasibility verdict
        feasible = est.t1_budget_ratio < 5.0 and est.snr_at_critical > 1.0
        if not feasible:
            logger.warning(
                f"    ⚠️ FEASIBILITY CONCERN: T1_ratio={est.t1_budget_ratio:.2f}, "
                f"SNR={est.snr_at_critical:.2f}. Results may be noise-dominated."
            )

        return {
            "feasible": feasible,
            "estimated_qpu_s": est.est_total_s,
            "estimated_qpu_min": est.est_total_s / 60,
            "t1_budget_ratio": est.t1_budget_ratio,
            "snr_at_critical": est.snr_at_critical,
            "effective_clops": est.effective_clops,
            "decoherence_fraction": est.decoherence_fraction,
            "transpile_metrics": (
                {k: v for k, v in transpile_data.items() if k != "transpiled_circuit"}
                if transpile_data else None
            ),
        }


    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 1: Ground Truth — exact reference energies
    # ═══════════════════════════════════════════════════════════════════════════

    def _phase_ground_truth(self) -> dict[str, Any]:
        """Compute exact ground state energies for all h-values."""
        from qmbp_simulation import ClassicalSolver, make_lattice
        from qmbp_simulation.models.model_registry import get_model_spec

        cfg = self.config
        spec = get_model_spec(cfg.model)
        solver = ClassicalSolver()

        logger.info("\n  ── Phase 1: Ground Truth ──")

        all_h = sorted(set(cfg.h_test + cfg.h_train), reverse=True)
        gt_data: dict[float, dict] = {}

        for h in all_h:
            t0 = time.perf_counter()
            lattice = make_lattice(cfg.topology, cfg.n_qubits, J=1.0, h=h)
            H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)

            method = "exact" if cfg.n_qubits <= 22 else "dmrg"
            gt = solver.solve(H, lattice, method=method)
            elapsed = time.perf_counter() - t0

            gt_data[h] = {
                "energy": gt.ground_energy,
                "gap": gt.gap,
                "method": method,
                "time_s": round(elapsed, 2),
            }
            logger.info(
                f"    h={h:.3f}: E₀={gt.ground_energy:.8f}, gap={gt.gap:.4f} ({elapsed:.1f}s)"
            )

        self._gt_data = gt_data
        return {"per_h": {str(h): v for h, v in gt_data.items()}, "method": method}

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 2: MPNN Training — generate VQE data + train predictor
    # ═══════════════════════════════════════════════════════════════════════════

    def _phase_mpnn_training(self) -> dict[str, Any]:
        """Generate training data with high-chi MPS and train MPNN."""
        from qmbp_simulation import make_lattice
        from qmbp_simulation.execution.mps_backend import MPSBackend
        from qmbp_simulation.models.data_models import VQEConfig
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.optimizers.vqe import VQEOptimizer
        from qmbp_simulation.predictors import (
            MPNNPredictor,
            build_graph_dataset,
            train_mpnn,
        )

        cfg = self.config
        spec = get_model_spec(cfg.model)
        chi_train = cfg.mps_chi_for_training

        logger.info(f"\n  ── Phase 2: MPNN Training (VQE at χ={chi_train}) ──")

        # VQE at high chi for training data
        backend = MPSBackend(strategy="aer_mps", chi_max=chi_train, seed=cfg.seed)
        lattice_ref = make_lattice(cfg.topology, cfg.n_qubits, J=1.0, h=cfg.h_train[0])
        circuit, _ = spec.create_circuit(
            cfg.n_qubits, cfg.p_layers, lattice_ref, **spec.circuit_kwargs
        )
        n_params = circuit.num_parameters

        vqe_config = VQEConfig(
            p_layers=cfg.p_layers,
            n_restarts=cfg.vqe_n_restarts,
            maxiter=cfg.vqe_maxiter,
            method="L-BFGS-B",
            enable_callbacks=False,
        )
        optimizer = VQEOptimizer(config=vqe_config, backend=backend, seed=cfg.seed)

        rng = np.random.default_rng(cfg.seed)
        prev_theta = rng.uniform(-0.01, 0.01, n_params)
        vqe_results = []

        # Descending sweep (warm-start)
        for h in sorted(cfg.h_train, reverse=True):
            t0 = time.perf_counter()
            lattice_h = make_lattice(cfg.topology, cfg.n_qubits, J=1.0, h=h)
            H = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)
            e_exact = self._gt_data[h]["energy"]

            result = optimizer.optimize(
                hamiltonian=H, circuit=circuit,
                initial_guess=prev_theta, exact_energy=e_exact,
            )
            prev_theta = result.theta_opt.copy()
            elapsed = time.perf_counter() - t0

            de_gap = abs(result.energy - e_exact) / max(self._gt_data[h]["gap"], 1e-10)
            vqe_results.append({
                "h": h, "theta_opt": result.theta_opt, "energy": result.energy,
                "de_gap": de_gap, "time_s": round(elapsed, 2),
            })
            logger.info(f"    VQE h={h:.3f}: E={result.energy:.8f} ΔE/gap={de_gap:.4f} ({elapsed:.1f}s)")

        # Train MPNN
        logger.info(f"    Training MPNN ({cfg.mpnn_epochs} epochs, patience={cfg.mpnn_patience})...")

        # Add ΔE/gap as quality gate for training data
        h_arr = np.array([r["h"] for r in vqe_results])
        theta_arr = np.array([r["theta_opt"] for r in vqe_results])
        e_arr = np.array([self._gt_data[r["h"]]["energy"] for r in vqe_results])
        de_gap_arr = np.array([r["de_gap"] for r in vqe_results])

        # Warn if any training point has high error
        n_bad_training = int(np.sum(de_gap_arr > cfg.abort_de_gap_threshold))
        if n_bad_training > 0:
            logger.warning(
                f"    ⚠️ {n_bad_training}/{len(vqe_results)} training points have "
                f"ΔE/gap > {cfg.abort_de_gap_threshold:.0%}. MPNN quality may suffer."
            )

        dataset = build_graph_dataset(
            lattice=lattice_ref,
            h_values=h_arr,
            theta_opt=theta_arr,
            e_exact=e_arr,
            de_gaps=de_gap_arr,
            de_gap_threshold=0.20,  # Exclude very bad VQE points
            fidelities=None,
            fidelity_threshold=0.0,
        )
        model = MPNNPredictor(
            node_features=2,
            hidden_dim=cfg.mpnn_hidden_dim,
            n_layers=cfg.mpnn_n_layers,
            output_dim=n_params,
        )
        train_mpnn(
            model=model, dataset=dataset,
            n_epochs=cfg.mpnn_epochs, lr=cfg.mpnn_lr,
            patience=cfg.mpnn_patience, seed=cfg.seed,
        )

        # Generate predictions for test h-values
        self._mpnn_model = model
        self._theta_pred = self._predict_theta(model, cfg.h_test)
        self._circuit = circuit

        logger.info(f"    MPNN trained. Predictions for {len(cfg.h_test)} test h-values ready.")

        return {
            "n_training_points": len(vqe_results),
            "chi_for_training": chi_train,
            "n_params": n_params,
            "vqe_per_h": [{k: v for k, v in r.items() if k != "theta_opt"} for r in vqe_results],
            "mpnn_config": {
                "hidden_dim": cfg.mpnn_hidden_dim,
                "n_layers": cfg.mpnn_n_layers,
                "epochs": cfg.mpnn_epochs,
            },
        }

    def _predict_theta(self, model, h_values: list[float]) -> dict[float, np.ndarray]:
        """Generate MPNN predictions using the canonical predict_theta utility."""
        from qmbp_simulation import make_lattice
        from qmbp_simulation.predictors import predict_theta

        cfg = self.config
        lattice_ref = make_lattice(cfg.topology, cfg.n_qubits, J=1.0, h=h_values[0])
        return predict_theta(model, lattice_ref, h_values)


    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 3: MPS Comparison — evaluate θ_pred at multiple chi
    # ═══════════════════════════════════════════════════════════════════════════

    def _phase_mps_comparison(self) -> dict[str, Any]:
        """Evaluate MPNN θ_pred at multiple chi values for comparison."""
        from experiments.helpers.scaling_utils import (
            analyze_chi_convergence,
            evaluate_at_multiple_chi,
        )

        from qmbp_simulation import make_lattice
        from qmbp_simulation.models.model_registry import get_model_spec

        cfg = self.config
        spec = get_model_spec(cfg.model)

        logger.info(f"\n  ── Phase 3: MPS Comparison (χ={cfg.mps_chi_values}) ──")

        per_h_results = {}
        for h in cfg.h_test:
            lattice_h = make_lattice(cfg.topology, cfg.n_qubits, J=1.0, h=h)
            H = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)
            theta = self._theta_pred[h]
            e_exact = self._gt_data[h]["energy"]
            gap = self._gt_data[h]["gap"]

            # Evaluate at each chi
            chi_results = evaluate_at_multiple_chi(
                self._circuit, H, theta, cfg.mps_chi_values, seed=cfg.seed
            )

            # Analyze convergence
            analysis = analyze_chi_convergence(chi_results, e_exact=e_exact, gap=gap)

            per_h_results[h] = {
                "chi_results": {str(k): v for k, v in chi_results.items()},
                "analysis": analysis,
            }

            # Log key result: chi=64 vs exact
            chi64_dg = analysis.get("chi64_de_gap")
            chi64_err = analysis.get("chi64_abs_error")
            logger.info(
                f"    h={h:.3f}: χ=64 |ΔE|={chi64_err:.2e} ΔE/gap={chi64_dg:.6f} "
                f"({'OK' if analysis['chi64_is_sufficient'] else 'INSUFFICIENT'})"
            )

        # Summary: is chi=64 sufficient for all h-test points?
        n_chi64_ok = sum(
            1 for r in per_h_results.values()
            if r["analysis"].get("chi64_is_sufficient")
        )
        chi64_sufficient = n_chi64_ok == len(cfg.h_test)

        self._mps_data = per_h_results
        return {
            "chi_values_tested": cfg.mps_chi_values,
            "per_h": {str(h): v for h, v in per_h_results.items()},
            "chi64_sufficient_all_h": chi64_sufficient,
            "chi64_pass_rate": n_chi64_ok / max(len(cfg.h_test), 1),
            "qpu_advantage_justified": not chi64_sufficient,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 4: QPU/FakeBackend Execution
    # ═══════════════════════════════════════════════════════════════════════════

    def _phase_qpu_execution(self) -> dict[str, Any]:
        """Execute θ_pred on QPU or FakeTorino with PEA-ZNE mitigation."""
        from qmbp_simulation import make_lattice
        from qmbp_simulation.execution.hardware import HardwareBackend, HardwareConfig
        from qmbp_simulation.execution.backends import MitigationOptions
        from qmbp_simulation.models.model_registry import get_model_spec

        cfg = self.config
        spec = get_model_spec(cfg.model)

        logger.info(f"\n  ── Phase 4: {cfg.mode.upper()} Execution ──")

        # Build HardwareConfig
        hw_config = HardwareConfig(
            backend_name=cfg.backend_name,
            mode=cfg.mode,
            n_qubits=cfg.n_qubits,
            shots=cfg.shots,
            n_layouts=cfg.n_layouts,
            optimization_level=2,
            layout_seed=cfg.seed,
            job_timeout_s=cfg.job_timeout_s,
            spsa_enabled=cfg.spsa_enabled,
            spsa_threshold=cfg.de_gap_threshold,
            output_dir=cfg.output_dir,
            mitigation=MitigationOptions(
                dd_enabled=True,
                trex_enabled=True,
                twirling_enabled=True,
                zne_enabled=True,
                zne_amplifier=cfg.amplifier,
            ),
        )

        backend = HardwareBackend(config=hw_config)
        per_h_qpu = {}

        for h in cfg.h_test:
            lattice_h = make_lattice(cfg.topology, cfg.n_qubits, J=1.0, h=h)
            H = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)
            theta = self._theta_pred[h]
            e_exact = self._gt_data[h]["energy"]
            gap = self._gt_data[h]["gap"]

            logger.info(f"    h={h:.3f}: executing...")
            t0 = time.perf_counter()

            try:
                result = backend.run_deployment(
                    self._circuit, H, theta,
                    h_value=h, e_exact=e_exact, gap=gap,
                    expected_label="paramagnetic" if h > 1.5 else "ferromagnetic",
                )
                elapsed = time.perf_counter() - t0

                per_h_qpu[h] = {
                    "e_zne": result.e_zne,
                    "e_exact": result.e_exact,
                    "delta_e_gap": result.delta_e_gap,
                    "gap": result.gap,
                    "zne_r2": result.zne_r2,
                    "zne_gain": result.zne_gain,
                    "phase_label": result.phase_label,
                    "verdict": result.verdict,
                    "wall_clock_s": round(elapsed, 2),
                    "total_shots": result.total_shots,
                    "per_site_x": result.per_site_x,
                    "per_bond_zz": result.per_bond_zz,
                    "spsa_applied": result.spsa_applied,
                    "mitigation_strategy": result.mitigation_strategy,
                }
                status = "✅" if result.delta_e_gap < cfg.de_gap_threshold else "⚠️"
                logger.info(
                    f"      {status} E_zne={result.e_zne:.6f} "
                    f"ΔE/gap={result.delta_e_gap:.4f} ({elapsed:.1f}s)"
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                per_h_qpu[h] = {"error": str(exc), "wall_clock_s": round(elapsed, 2)}
                logger.error(f"      ❌ Failed: {exc}")

        self._qpu_data = per_h_qpu
        return {"per_h": {str(h): v for h, v in per_h_qpu.items()}}


    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 5: Analysis — QPU vs MPS vs Exact comparison
    # ═══════════════════════════════════════════════════════════════════════════

    def _phase_analysis(self) -> dict[str, Any]:
        """Compare QPU, MPS(χ=64), DMRG(1D), and exact energies at each h-point."""
        from qmbp_simulation import ClassicalSolver, make_lattice
        from qmbp_simulation.models.model_registry import get_model_spec

        cfg = self.config
        spec = get_model_spec(cfg.model)
        solver = ClassicalSolver()

        logger.info("\n  ── Phase 5: Analysis ──")

        comparison_table = []
        for h in cfg.h_test:
            e_exact = self._gt_data[h]["energy"]
            gap = self._gt_data[h]["gap"]

            # MPS at chi=64 (from Phase 3)
            mps_data = self._mps_data.get(h, {})
            chi_results = mps_data.get("chi_results", {})
            # Keys may be int or str depending on JSON serialization
            e_mps_64 = (
                chi_results.get("64", chi_results.get(64, {})).get("energy")
            )

            # DMRG(1D model) baseline — the classical limitation we're comparing against
            lattice_h = make_lattice(cfg.topology, cfg.n_qubits, J=1.0, h=h)
            H = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)
            gt_dmrg = solver.solve(H, lattice_h, method="dmrg")
            e_dmrg = gt_dmrg.ground_energy

            # QPU result
            qpu_data = getattr(self, "_qpu_data", {}).get(h, {})
            e_qpu = qpu_data.get("e_zne")

            # Compute comparison metrics
            row = {
                "h": h,
                "e_exact": e_exact,
                "gap": gap,
                "e_mps_chi64": e_mps_64,
                "de_gap_mps_chi64": (
                    abs(e_mps_64 - e_exact) / max(gap, 1e-10) if e_mps_64 else None
                ),
                "e_dmrg_1d": e_dmrg,
                "de_abs_dmrg": abs(e_dmrg - e_exact),
                "de_gap_dmrg": abs(e_dmrg - e_exact) / max(gap, 1e-10),
                "e_qpu": e_qpu,
                "de_gap_qpu": (
                    abs(e_qpu - e_exact) / max(gap, 1e-10) if e_qpu else None
                ),
                "qpu_beats_dmrg": (
                    abs(e_qpu - e_exact) < abs(e_dmrg - e_exact)
                    if e_qpu is not None else None
                ),
                "qpu_beats_mps": (
                    abs(e_qpu - e_exact) < abs(e_mps_64 - e_exact)
                    if e_qpu is not None and e_mps_64 is not None else None
                ),
            }
            comparison_table.append(row)

        # Print summary table
        logger.info(f"\n    {'h':>6} {'E_exact':>12} {'E_DMRG':>12} {'E_QPU':>12} "
                    f"{'ΔE/gap_DMRG':>12} {'ΔE/gap_QPU':>12} {'QPU>DMRG':>9}")
        logger.info(f"    {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*9}")
        for row in comparison_table:
            e_ex = f"{row['e_exact']:.6f}"
            e_d = f"{row['e_dmrg_1d']:.6f}"
            e_q = f"{row['e_qpu']:.6f}" if row["e_qpu"] else "—"
            dg_d = f"{row['de_gap_dmrg']:.5f}"
            dg_q = f"{row['de_gap_qpu']:.5f}" if row["de_gap_qpu"] is not None else "—"
            beats = "✅" if row["qpu_beats_dmrg"] else ("❌" if row["qpu_beats_dmrg"] is False else "—")
            logger.info(f"    {row['h']:>6.3f} {e_ex:>12} {e_d:>12} {e_q:>12} "
                        f"{dg_d:>12} {dg_q:>12} {beats:>9}")

        # Aggregate
        n_qpu_beats_dmrg = sum(1 for r in comparison_table if r["qpu_beats_dmrg"] is True)
        n_qpu_beats_mps = sum(1 for r in comparison_table if r["qpu_beats_mps"] is True)
        n_compared = sum(1 for r in comparison_table if r["qpu_beats_dmrg"] is not None)

        return {
            "comparison_table": comparison_table,
            "n_qpu_beats_dmrg": n_qpu_beats_dmrg,
            "n_qpu_beats_mps": n_qpu_beats_mps,
            "n_compared": n_compared,
            "qpu_advantage_rate_vs_dmrg": n_qpu_beats_dmrg / max(n_compared, 1),
            "qpu_advantage_rate_vs_mps": n_qpu_beats_mps / max(n_compared, 1),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Summary + Persistence
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_summary(self, elapsed_s: float) -> dict:
        """Build final summary dict."""
        cfg = self.config
        analysis = self._results.get("analysis", {})
        preflight = self._results.get("preflight", {})
        mps = self._results.get("mps_comparison", {})

        return {
            "total_time_s": round(elapsed_s, 1),
            "total_time_min": round(elapsed_s / 60, 1),
            "mode": cfg.mode,
            "n_qubits": cfg.n_qubits,
            "topology": cfg.topology,
            "n_h_test": len(cfg.h_test),
            "feasible": preflight.get("feasible"),
            "t1_budget_ratio": preflight.get("t1_budget_ratio"),
            "chi64_sufficient": mps.get("chi64_sufficient_all_h"),
            "qpu_advantage_justified": mps.get("qpu_advantage_justified"),
            "qpu_advantage_rate_vs_dmrg": analysis.get("qpu_advantage_rate_vs_dmrg"),
            "qpu_advantage_rate_vs_mps": analysis.get("qpu_advantage_rate_vs_mps"),
            "thesis_verdict": (
                f"At N={cfg.n_qubits} {cfg.topology}: "
                f"DMRG(1D) error significant → QPU beats DMRG at "
                f"{analysis.get('qpu_advantage_rate_vs_dmrg', 0)*100:.0f}% of h-points."
                if analysis.get("qpu_advantage_rate_vs_dmrg") is not None
                else "QPU execution not completed"
            ),
        }

    def _save_results(self):
        """Save results to JSON file."""
        from qmbp_simulation.utils.helpers import json_dump

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        from qmbp_simulation.framework.result_io import generate_timestamp
        filename = f"deployment_N{self.config.n_qubits}_{self.config.topology}_{generate_timestamp()}.json"
        path = output_dir / filename

        json_dump(self._results, path)
        logger.info(f"\n  Results saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Interface
# ═══════════════════════════════════════════════════════════════════════════════


def build_parser() -> "argparse.ArgumentParser":
    """Build CLI argument parser."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parametric QPU Deployment — N-agnostic hardware execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Physics
    parser.add_argument("--n-qubits", type=int, default=20)
    parser.add_argument("--topology", type=str, default="heavy_hex")
    parser.add_argument("--model", type=str, default="tfim")
    parser.add_argument("--p-layers", type=int, default=1, choices=[1, 2])
    parser.add_argument("--h-test", type=float, nargs="+", default=[4.0, 3.5, 3.0])
    parser.add_argument("--h-train", type=float, nargs="+",
                        default=[4.5, 4.25, 3.75, 3.25, 2.75, 2.5])
    # Execution
    parser.add_argument("--mode", type=str, default="fake_backend",
                        choices=["hardware", "fake_backend", "dry_run"])
    parser.add_argument("--backend", type=str, default="ibm_kingston")
    parser.add_argument("--shots", type=int, default=16384)
    parser.add_argument("--n-layouts", type=int, default=3)
    parser.add_argument("--amplifier", type=str, default="pea",
                        choices=["pea", "gate_folding", "adaptive"])
    parser.add_argument("--pea-preset", type=str, default="balanced")
    parser.add_argument("--no-spsa", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    # MPS
    parser.add_argument("--mps-chi-training", type=int, default=256,
                        help="Chi for VQE training data (higher=more precise, slower)")
    # MPNN
    parser.add_argument("--mpnn-epochs", type=int, default=6000)
    # Output
    parser.add_argument("--output-dir", type=str, default="results/hardware/parametric")
    parser.add_argument("-v", "--verbose", action="store_true")
    # Shortcut
    parser.add_argument("--dry-run", action="store_true",
                        help="Cost estimation only (overrides --mode)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # --dry-run overrides --mode
    if args.dry_run:
        args.mode = "dry_run"

    config = DeploymentConfig.from_cli(args)
    deployment = ParametricDeployment(config)
    results = deployment.run()

    # Print final verdict
    summary = results.get("summary", {})
    if summary:
        print(f"\n{'='*60}")
        print(f"  VERDICT: {summary.get('thesis_verdict', 'N/A')}")
        print(f"  Total time: {summary.get('total_time_min', 0):.1f} min")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
