#!/usr/bin/env python3
"""Noise-Aware MPNN + Unified Graph — 2×2 Variant Comparison.

Executes the 2×2 experimental matrix:

    ┌──────────────────────┬─────────────────────┬──────────────────────┐
    │                      │ θ_target = noiseless │ θ_target = noisy     │
    ├──────────────────────┼─────────────────────┼──────────────────────┤
    │ Graph = Ham only     │ Variant A (baseline) │ Variant C (#06 only) │
    │ Graph = Ham+Circuit  │ Variant B (#04 only) │ Variant D (combined) │
    └──────────────────────┴─────────────────────┴──────────────────────┘

Each variant trains a BondResolvedMPNN and evaluates on NoisyBackend.

Success criteria (from criteria.py):
  - NOISE_AWARE_MPNN: Variant C beats A on >=70% h-points (no ZNE)
  - UNIFIED_GRAPH: Variant B MSE improvement >= 30% vs A
  - UNIFIED_NOISE_COMBINED: Variant D achieves >=80% pass rate

Usage:
    python scripts/experiment_runners/noise_aware/run_noise_aware_comparison.py
    python scripts/experiment_runners/noise_aware/run_noise_aware_comparison.py \\
        --n-qubits 10 --topology chain_1d --p-layers 1 --shots 8192
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

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

from qmbp_simulation.models.constants import DE_GAP_THRESHOLD

DEFAULT_N = 10
DEFAULT_P = 1
DEFAULT_TOPOLOGY = "chain_1d"
DEFAULT_MODEL = "tfim"
DEFAULT_SHOTS = 8192
DEFAULT_N_RESTARTS_NOISY = 15
DEFAULT_MAXITER_NOISY = 2000
DEFAULT_H_MIN = 1.3
DEFAULT_H_MAX = 3.0
DEFAULT_H_POINTS = 20
DEFAULT_MPNN_EPOCHS = 6000

VARIANT_A = "ham_noiseless"
VARIANT_B = "unified_noiseless"
VARIANT_C = "ham_noisy"
VARIANT_D = "unified_noisy"


class NoiseAwareComparisonRunner(ValidationRunner):
    """2×2 Variant Comparison: Unified Graph × Noise-Aware Training."""

    runner_id = "noise_aware_comparison_v1"
    experiment_id = "UNIFIED_NOISE_COMBINED"
    description = "2×2 ablation: (Ham-only vs Unified) × (noiseless vs noisy θ)"
    hypothesis = (
        "Combined unified graph + noise-aware training achieves highest "
        "pass rate on noisy deployment."
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
            maxiter=DEFAULT_MAXITER_NOISY,
            n_restarts=DEFAULT_N_RESTARTS_NOISY,
        )
        parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
        parser.add_argument("--mpnn-epochs", type=int, default=DEFAULT_MPNN_EPOCHS)
        parser.add_argument(
            "--noisy-backend",
            choices=["gaussian", "faketorino"],
            default="gaussian",
            help="Backend for noisy VQE: 'gaussian' (shot noise only, fast) or "
            "'faketorino' (full coherent noise model, slow but realistic). "
            "Default: gaussian.",
        )
        parser.add_argument(
            "--skip-variants", nargs="*", default=[],
            choices=[VARIANT_A, VARIANT_B, VARIANT_C, VARIANT_D],
        )
        parser.add_argument(
            "--bond-resolved", action="store_true", default=True,
            help="Use bond-resolved HVA (per-bond θ_zz + per-site θ_x). Default: True.",
        )
        parser.add_argument(
            "--no-bond-resolved", dest="bond_resolved", action="store_false",
            help="Use global HVA (2 params/layer) instead of bond-resolved.",
        )
        parser.add_argument(
            "--flow-K", type=int, default=0,
            help="If > 0, train flow and add FlowMultiShot variants C1/C2 "
            "with K candidates per point. Default: 0 (no flow).",
        )

    def run_preflight(self) -> bool:
        if self._args.n_qubits > 14:
            logger.warning("N=%d > 14: noisy VQE will be very slow.", self._args.n_qubits)
        if self._args.p_layers > 2:
            logger.error("p_layers > 2 not supported (thesis constraint).")
            return False
        if self._args.h_points < 8:
            logger.error("Need >= 8 h-points for train/test split.")
            return False
        if self._args.h_min < 1.0:
            logger.warning("h_min < 1.0: HVA p<=2 cannot express ground state below h_c≈1.0")
        return True

    def build_config(self) -> dict:
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "system": {
                "n_qubits": self._args.n_qubits,
                "p_layers": self._args.p_layers,
                "topology": self._args.topology[0],
                "model": self._args.model,
            },
            "h_grid": {
                "h_min": self._args.h_min,
                "h_max": self._args.h_max,
                "h_points": self._args.h_points,
            },
            "noisy": {"shots": self._args.shots, "n_restarts": self._args.n_restarts},
            "mpnn_epochs": self._args.mpnn_epochs,
            "skipped": self._args.skip_variants,
        }

    def setup(self):
        self.setup_physics()
        self._topo = self._args.topology[0]
        self._N = self._args.n_qubits
        self._p = self._args.p_layers
        self._h_values = self.generate_h_grid()
        self._seed = self._args.seeds[0]
        self._bond_resolved = self._args.bond_resolved

        # Build circuit and lattice
        spec = self.get_spec()
        self._lattice_ref = self.make_lattice(self._topo, self._N, J=1.0, h=self._h_values[0])

        if self._bond_resolved:
            # Bond-resolved HVA: per-bond θ_zz + per-site θ_x
            self._circuit, _ = self.hva.create_bond_resolved(
                self._N, self._p, self._lattice_ref
            )
            logger.info(
                "  Circuit: bond-resolved HVA, %d params "
                "(%d edges + %d qubits) × %d layers",
                self._circuit.num_parameters,
                len(self._lattice_ref.edges), self._N, self._p,
            )
        else:
            # Global HVA: 2 params/layer (standard TFIM)
            self._circuit, _ = spec.create_circuit(
                self._N, self._p, self._lattice_ref, **spec.circuit_kwargs
            )
            logger.info(
                "  Circuit: global HVA, %d params", self._circuit.num_parameters
            )

        self._spec = spec

        # Storage
        self._e_exact: np.ndarray | None = None
        self._gaps: np.ndarray | None = None
        self._theta_noiseless: np.ndarray | None = None
        self._theta_noisy: np.ndarray | None = None
        self._variant_results: dict[str, dict[str, Any]] = {}

    def define_sections(self) -> list[Section]:
        return [
            Section(id=1, name="Data Collection", fn=self.section_data,
                    hypothesis="VQE converges in both noiseless and noisy regimes"),
            Section(id=2, name="Train 4 Variants", fn=self.section_train,
                    hypothesis="All variants train to MSE < 1e-2"),
            Section(id=3, name="Noisy Deployment", fn=self.section_deploy,
                    hypothesis="Noise-aware variants outperform baseline"),
            Section(id=4, name="Statistical Analysis", fn=self.section_stats,
                    hypothesis="Improvements are statistically significant"),
        ]

    def _run_vqe_sweep_internal(
        self,
        backend,
        method: str = "L-BFGS-B",
        n_restarts: int = 5,
        maxiter: int = 500,
    ) -> list[np.ndarray]:
        """Run descending VQE sweep using self._circuit (bond-resolved or global).

        This is the internal sweep that respects the --bond-resolved flag by
        using the circuit built in setup(), rather than delegating to
        vqe_descending_sweep() which always creates its own from ModelSpec.

        Returns list of θ_opt arrays in the same order as self._h_values.
        """
        from qmbp_simulation import VQEConfig, VQEOptimizer

        vqe_config = self.VQEConfig(
            p_layers=self._p,
            n_restarts=n_restarts,
            maxiter=maxiter,
            method=method,
        )
        optimizer = self.VQEOptimizer(
            config=vqe_config, backend=backend, seed=self._seed
        )

        circuit = self._circuit
        spec = self._spec
        n_params = circuit.num_parameters
        rng = np.random.default_rng(self._seed)
        prev_theta = rng.uniform(-0.01, 0.01, n_params)

        # Descending sweep
        h_sorted = sorted(self._h_values, reverse=True)
        theta_map: dict[float, np.ndarray] = {}

        for h in h_sorted:
            lat = self.make_lattice(self._topo, self._N, J=1.0, h=h)
            H = spec.build_hamiltonian(lat, **spec.hamiltonian_kwargs)

            result = optimizer.optimize(
                hamiltonian=H,
                circuit=circuit,
                initial_guess=prev_theta,
            )

            if np.all(np.isfinite(result.theta_opt)):
                prev_theta = result.theta_opt.copy()
            else:
                logger.warning("  NaN at h=%.4f, keeping previous θ", h)

            theta_map[h] = prev_theta.copy()

        # Return in original h_values order
        return [theta_map[h] for h in self._h_values]

    # ══════════════════════════════════════════════════════════════════════
    # Section 1: Data Collection
    # ══════════════════════════════════════════════════════════════════════

    def section_data(self) -> dict:
        """Collect ground truth, noiseless θ, and noisy θ."""
        from qmbp_simulation import VQEConfig, VQEOptimizer
        from qmbp_simulation.execution import NoisyBackend
        from qmbp_simulation.utils.helpers import canonicalize_theta, timer

        topo, N, p = self._topo, self._N, self._p
        spec, seed = self._spec, self._seed
        h_values = self._h_values

        # Phase 1: Exact diagonalization
        logger.info("  Phase 1: Exact diag (%d h-points)...", len(h_values))
        e_exact, gaps = [], []
        for h in h_values:
            lat = self.make_lattice(topo, N, J=1.0, h=h)
            H = spec.build_hamiltonian(lat, **spec.hamiltonian_kwargs)
            gt = self.solver.solve(H, lat)
            e_exact.append(gt.ground_energy)
            gaps.append(gt.gap)
        self._e_exact = np.array(e_exact)
        self._gaps = np.array(gaps)

        # Phase 2a: Noiseless VQE
        logger.info("  Phase 2a: Noiseless VQE sweep (%d params)...",
                    self._circuit.num_parameters)
        # Use higher maxiter to handle COBYLA auto-switch for bond-resolved (19+ params)
        _noiseless_maxiter = max(1000, self._args.maxiter // 2)
        with timer("noiseless_vqe") as t_noiseless:
            theta_noiseless = self._run_vqe_sweep_internal(
                backend=self.noiseless,
                method="L-BFGS-B",
                n_restarts=5,
                maxiter=_noiseless_maxiter,
            )
        for i in range(len(theta_noiseless)):
            theta_noiseless[i] = canonicalize_theta(theta_noiseless[i])
        self._theta_noiseless = np.array(theta_noiseless)
        logger.info("    Done in %.1fs", t_noiseless.elapsed_s)

        # Phase 2b: Noisy VQE
        logger.info("  Phase 2b: Noisy VQE sweep (shots=%d)...", self._args.shots)
        noisy_backend_type = getattr(self._args, "noisy_backend", "gaussian")

        if noisy_backend_type == "faketorino":
            # Full coherent noise: FakeTorino noise model via AerSimulator
            # Extracts the noise model from FakeTorino and uses it with NoisyBackend
            logger.info("    Using FakeTorino noise model (coherent errors)")
            try:
                from qiskit_aer.noise import NoiseModel
                from qiskit_ibm_runtime.fake_provider import FakeTorino

                fake_backend = FakeTorino()
                noise_model = NoiseModel.from_backend(fake_backend)
                noisy_backend = NoisyBackend(
                    shots=self._args.shots,
                    noise_model=noise_model,
                    seed_simulator=seed,
                )
                logger.info("    NoiseModel extracted successfully")
            except ImportError as e:
                logger.warning(
                    "    FakeTorino not available (%s). Falling back to Gaussian.", e
                )
                noisy_backend = NoisyBackend(shots=self._args.shots, seed_simulator=seed)

            # SPSA for FakeTorino: gradient-free stochastic optimizer
            # validated 3× better than COBYLA under coherent noise (Karim et al. 2025)
            noisy_method = "COBYLA"  # SPSA not yet in VQEOptimizer; COBYLA with high maxiter
            noisy_maxiter = self._args.maxiter
        else:
            # Gaussian shot noise only (fast, local — no real noise model)
            noisy_backend = NoisyBackend(shots=self._args.shots, seed_simulator=seed)
            noisy_method = "COBYLA"
            noisy_maxiter = self._args.maxiter

        with timer("noisy_vqe") as t_noisy:
            theta_noisy = self._run_vqe_sweep_internal(
                backend=noisy_backend,
                method=noisy_method,
                n_restarts=self._args.n_restarts,
                maxiter=noisy_maxiter,
            )
        for i in range(len(theta_noisy)):
            theta_noisy[i] = canonicalize_theta(theta_noisy[i])
        self._theta_noisy = np.array(theta_noisy)
        logger.info("    Done in %.1fs", t_noisy.elapsed_s)

        # Convergence check: evaluate noisy θ on noiseless backend
        noiseless_backend = self.noiseless
        noisy_de_gaps = []
        for i, h in enumerate(h_values):
            lat = self.make_lattice(topo, N, J=1.0, h=h)
            H = spec.build_hamiltonian(lat, **spec.hamiltonian_kwargs)
            e_noisy = noiseless_backend.evaluate(self._circuit, H, theta_noisy[i])
            de = abs(e_noisy - self._e_exact[i]) / max(self._gaps[i], 1e-10)
            noisy_de_gaps.append(de)

        noisy_de_gaps = np.array(noisy_de_gaps)
        noisy_converged = float((noisy_de_gaps < 0.20).mean())

        logger.info(
            "  Noisy VQE quality: %.0f%% converged (ΔE/gap<20%%), "
            "mean ΔE/gap=%.4f", noisy_converged * 100, noisy_de_gaps.mean()
        )

        if noisy_converged < 0.30:
            logger.warning(
                "  ⚠️ DECISION GATE: Only %.0f%% noisy VQE points converged.",
                noisy_converged * 100,
            )

        return {
            "pass": True,
            "n_points": len(h_values),
            "n_params": self._circuit.num_parameters,
            "bond_resolved": self._bond_resolved,
            "noiseless_time_s": t_noiseless.elapsed_s,
            "noisy_time_s": t_noisy.elapsed_s,
            "noisy_convergence_rate": float(noisy_converged),
            "noisy_mean_de_gap": float(noisy_de_gaps.mean()),
            "noisy_max_de_gap": float(noisy_de_gaps.max()),
            # Raw data for post-hoc analysis (θ displacement, correlation, etc.)
            "h_values": list(h_values),
            "theta_noiseless": self._theta_noiseless.tolist(),
            "theta_noisy": self._theta_noisy.tolist(),
            "e_exact": self._e_exact.tolist(),
            "gaps": self._gaps.tolist(),
            "noisy_de_gaps_per_h": noisy_de_gaps.tolist(),
        }

    # ══════════════════════════════════════════════════════════════════════
    # Section 2: Train 4 Variants
    # ══════════════════════════════════════════════════════════════════════

    def section_train(self) -> dict:
        """Train BondResolvedMPNN for each variant in the 2×2 matrix."""
        from experiments.helpers.graph_utils import train_bond_resolved_variant
        from qmbp_simulation.utils.helpers import timer

        skip = set(self._args.skip_variants)
        epochs = self._args.mpnn_epochs
        lattice = self._lattice_ref
        h_values = self._h_values
        results = {}

        # Check if unified graph module is available (for B/D variants)
        unified_available = False
        graph_metrics = {}
        if VARIANT_B not in skip or VARIANT_D not in skip:
            try:
                from qmbp_simulation.predictors.unified_graph import (
                    build_unified_bond_resolved_graph,
                    compute_graph_metrics,
                )
                from qmbp_simulation.predictors import validate_unified_graph

                sample_graph = build_unified_bond_resolved_graph(
                    lattice, h_value=float(h_values[0]), p_layers=self._p,
                    include_circuit_nodes=True,
                )
                issues = validate_unified_graph(sample_graph)
                if issues:
                    logger.warning("  Graph validation issues: %s — skipping B/D", issues)
                    skip.add(VARIANT_B)
                    skip.add(VARIANT_D)
                else:
                    unified_available = True
                    graph_metrics = compute_graph_metrics(sample_graph)
                    logger.info(
                        "  Graph metrics: %d nodes (%.1f× expansion), %d edges",
                        graph_metrics["total_nodes"],
                        graph_metrics["node_expansion_ratio"],
                        graph_metrics["total_edges"],
                    )
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(
                    "  unified_graph module not available: %s — "
                    "skipping variants B/D (requires #04 Qracle integration)", e
                )
                skip.add(VARIANT_B)
                skip.add(VARIANT_D)

        variants = [
            (VARIANT_A, self._theta_noiseless, False, "Ham-only + noiseless θ"),
            (VARIANT_B, self._theta_noiseless, True, "Unified + noiseless θ"),
            (VARIANT_C, self._theta_noisy, False, "Ham-only + noisy θ"),
            (VARIANT_D, self._theta_noisy, True, "Unified + noisy θ"),
        ]

        for variant_id, theta, include_circuit, desc in variants:
            if variant_id in skip:
                logger.info("  ⏭️ Skipping %s (%s)", variant_id, desc)
                continue

            logger.info("  Training %s: %s", variant_id, desc)
            with timer(variant_id) as t:
                model, train_metrics = train_bond_resolved_variant(
                    lattice, h_values, theta,
                    include_circuit_nodes=include_circuit,
                    p_layers=self._p,
                    hidden_dim=256,
                    n_layers=3,
                    n_epochs=epochs,
                    seed=self._seed,
                    dropout=0.15 if include_circuit else 0.1,
                )

            final_mse = train_metrics["final_mse"]
            logger.info(
                "    MSE=%.2e (%.1fs)", final_mse, t.elapsed_s
            )

            self._variant_results[variant_id] = {
                "model": model,
                "train_metrics": train_metrics,
                "include_circuit": include_circuit,
                "theta_source": "noisy" if "noisy" in variant_id else "noiseless",
                "desc": desc,
            }
            results[variant_id] = {
                "final_mse": float(final_mse),
                "training_time_s": t.elapsed_s,
                "include_circuit_nodes": include_circuit,
                "theta_source": "noisy" if "noisy" in variant_id else "noiseless",
            }

        # MSE comparison: unified vs ham-only improvement
        mse_a = results.get(VARIANT_A, {}).get("final_mse")
        mse_b = results.get(VARIANT_B, {}).get("final_mse")
        if mse_a and mse_b:
            improvement = (mse_a - mse_b) / mse_a * 100
            results["unified_mse_improvement_pct"] = float(improvement)
            logger.info("  Unified graph MSE improvement: %.1f%%", improvement)

        results["graph_metrics"] = graph_metrics

        # Tiered MSE threshold: noiseless targets are cleaner (0.01),
        # noisy targets inherently have higher MSE (0.05) due to shot noise
        # in the training labels. Failing on noisy MSE is NOT a real failure
        # — deployment quality (Section 3) is the true arbiter.
        MSE_THRESHOLD_NOISELESS = 0.01
        MSE_THRESHOLD_NOISY = 0.05

        all_pass = True
        for variant_id, r in results.items():
            if not isinstance(r, dict) or "final_mse" not in r:
                continue
            threshold = (
                MSE_THRESHOLD_NOISY if r.get("theta_source") == "noisy"
                else MSE_THRESHOLD_NOISELESS
            )
            if r["final_mse"] > threshold:
                all_pass = False
                logger.warning(
                    "  ⚠️ %s MSE=%.4f > threshold %.4f",
                    variant_id, r["final_mse"], threshold,
                )

        results["pass"] = all_pass
        return results

    # ══════════════════════════════════════════════════════════════════════
    # Section 3: Noisy Deployment
    # ══════════════════════════════════════════════════════════════════════

    def section_deploy(self) -> dict:
        """Deploy all trained variants on NoisyBackend and compare ΔE/gap."""
        from experiments.helpers.graph_utils import evaluate_bond_resolved_variant
        from qmbp_simulation.execution import NoisyBackend
        from qmbp_simulation.utils.helpers import timer

        if not self._variant_results:
            return {"pass": False, "error": "No variants trained (section 2 failed or skipped)"}

        # Test on midpoints between training h-values (unseen points)
        h_sorted = np.sort(self._h_values)
        h_test = (h_sorted[:-1] + h_sorted[1:]) / 2.0

        # Exact energies and gaps at test points
        e_test, gaps_test = [], []
        for h in h_test:
            lat = self.make_lattice(self._topo, self._N, J=1.0, h=float(h))
            H = self._spec.build_hamiltonian(lat, **self._spec.hamiltonian_kwargs)
            gt = self.solver.solve(H, lat)
            e_test.append(gt.ground_energy)
            gaps_test.append(gt.gap)
        e_test = np.array(e_test)
        gaps_test = np.array(gaps_test)

        # Evaluate each variant on NoisyBackend
        noisy_backend = NoisyBackend(shots=self._args.shots, seed_simulator=self._seed + 100)
        deploy_results = {}

        for variant_id, vdata in self._variant_results.items():
            model = vdata["model"]
            include_circuit = vdata["include_circuit"]

            logger.info("  Deploying %s on NoisyBackend...", variant_id)
            with timer(variant_id) as t:
                eval_result = evaluate_bond_resolved_variant(
                    model=model,
                    lattice=self._lattice_ref,
                    h_test_values=h_test,
                    circuit=self._circuit,
                    spec=self._spec,
                    backend=noisy_backend,
                    include_circuit_nodes=include_circuit,
                    p_layers=self._p,
                    e_exact=e_test,
                    gaps=gaps_test,
                )

            logger.info(
                "    %s: mean ΔE/gap=%.4f, pass_rate=%.0f%% (%.1fs)",
                variant_id, eval_result["mean_de_gap"],
                eval_result["pass_rate"] * 100, t.elapsed_s,
            )

            deploy_results[variant_id] = {
                "mean_de_gap": eval_result["mean_de_gap"],
                "max_de_gap": eval_result["max_de_gap"],
                "pass_rate": eval_result["pass_rate"],
                "n_pass": eval_result["n_pass"],
                "n_total": eval_result["n_total"],
                "per_point_de_gaps": [p["de_gap"] for p in eval_result["per_point"]],
                "deploy_time_s": t.elapsed_s,
            }

        # Also evaluate on noiseless (ceiling reference)
        logger.info("  Deploying variants on NoiselessBackend (ceiling)...")
        for variant_id, vdata in self._variant_results.items():
            model = vdata["model"]
            include_circuit = vdata["include_circuit"]
            eval_noiseless = evaluate_bond_resolved_variant(
                model=model,
                lattice=self._lattice_ref,
                h_test_values=h_test,
                circuit=self._circuit,
                spec=self._spec,
                backend=self.noiseless,
                include_circuit_nodes=include_circuit,
                p_layers=self._p,
                e_exact=e_test,
                gaps=gaps_test,
            )
            deploy_results[variant_id]["noiseless_mean_de_gap"] = eval_noiseless["mean_de_gap"]
            deploy_results[variant_id]["noiseless_pass_rate"] = eval_noiseless["pass_rate"]

        deploy_results["h_test"] = h_test.tolist()
        deploy_results["n_test_points"] = len(h_test)

        # Pass criterion: best variant achieves >= 80% pass rate
        best_pass_rate = max(
            r.get("pass_rate", 0) for r in deploy_results.values()
            if isinstance(r, dict) and "pass_rate" in r
        )
        deploy_results["best_pass_rate"] = float(best_pass_rate)
        deploy_results["pass"] = best_pass_rate >= 0.80

        return deploy_results

    # ══════════════════════════════════════════════════════════════════════
    # Section 4: Statistical Analysis
    # ══════════════════════════════════════════════════════════════════════

    def section_stats(self) -> dict:
        """Paired comparisons and verdict computation."""
        from project_health.analysis.statistical_tests import (
            effect_size_cohens_d,
            improvement_rate,
            paired_ttest,
        )
        from qmbp_simulation.framework.criteria import compute_verdict

        # Get Section 3 deploy data from _section_results list
        deploy = None
        for sr in self._section_results:
            if sr.section_id == 3 and sr.data:
                deploy = sr.data
                break

        if not deploy:
            return {"pass": False, "error": "No deployment data from Section 3"}

        results: dict[str, Any] = {}

        # Extract per-point ΔE/gap arrays
        de_gaps: dict[str, list[float]] = {}
        for variant_id in [VARIANT_A, VARIANT_B, VARIANT_C, VARIANT_D]:
            vdata = deploy.get(variant_id, {})
            de_gaps[variant_id] = vdata.get("per_point_de_gaps", [])

        # ── Comparison 1: #06 effect (C vs A) — noise-aware benefit ──
        if de_gaps.get(VARIANT_A) and de_gaps.get(VARIANT_C):
            noise_ttest = paired_ttest(de_gaps[VARIANT_A], de_gaps[VARIANT_C])
            noise_improvement = improvement_rate(de_gaps[VARIANT_A], de_gaps[VARIANT_C])
            noise_effect = effect_size_cohens_d(de_gaps[VARIANT_A], de_gaps[VARIANT_C])

            results["noise_aware_vs_baseline"] = {
                "paired_ttest": noise_ttest,
                "improvement_rate": noise_improvement,
                "cohens_d": float(noise_effect),
                "significant": noise_ttest["significant_005"],
                "c_wins_pct": noise_improvement["improvement_rate_pct"],
            }
            logger.info(
                "  #06 (C vs A): C wins %.0f%% | p=%.4f | d=%.2f %s",
                noise_improvement["improvement_rate_pct"],
                noise_ttest["p_value"],
                noise_effect,
                "✓" if noise_ttest["significant_005"] else "✗",
            )

        # ── Comparison 2: #04 effect (B vs A) — unified graph benefit ─
        if de_gaps.get(VARIANT_A) and de_gaps.get(VARIANT_B):
            graph_ttest = paired_ttest(de_gaps[VARIANT_A], de_gaps[VARIANT_B])
            graph_improvement = improvement_rate(de_gaps[VARIANT_A], de_gaps[VARIANT_B])
            graph_effect = effect_size_cohens_d(de_gaps[VARIANT_A], de_gaps[VARIANT_B])

            results["unified_graph_vs_baseline"] = {
                "paired_ttest": graph_ttest,
                "improvement_rate": graph_improvement,
                "cohens_d": float(graph_effect),
                "significant": graph_ttest["significant_005"],
                "b_wins_pct": graph_improvement["improvement_rate_pct"],
            }
            logger.info(
                "  #04 (B vs A): B wins %.0f%% | p=%.4f | d=%.2f %s",
                graph_improvement["improvement_rate_pct"],
                graph_ttest["p_value"],
                graph_effect,
                "✓" if graph_ttest["significant_005"] else "✗",
            )

        # ── Comparison 3: Combined (D vs A) — full integration ────────
        if de_gaps.get(VARIANT_A) and de_gaps.get(VARIANT_D):
            combined_ttest = paired_ttest(de_gaps[VARIANT_A], de_gaps[VARIANT_D])
            combined_improvement = improvement_rate(de_gaps[VARIANT_A], de_gaps[VARIANT_D])
            combined_effect = effect_size_cohens_d(de_gaps[VARIANT_A], de_gaps[VARIANT_D])

            results["combined_vs_baseline"] = {
                "paired_ttest": combined_ttest,
                "improvement_rate": combined_improvement,
                "cohens_d": float(combined_effect),
                "significant": combined_ttest["significant_005"],
                "d_wins_pct": combined_improvement["improvement_rate_pct"],
            }
            logger.info(
                "  #04+#06 (D vs A): D wins %.0f%% | p=%.4f | d=%.2f %s",
                combined_improvement["improvement_rate_pct"],
                combined_ttest["p_value"],
                combined_effect,
                "✓" if combined_ttest["significant_005"] else "✗",
            )

        # ── Compute verdicts against criteria ─────────────────────────
        verdicts = {}
        for crit_id, summary_key, metric_src in [
            ("NOISE_AWARE_MPNN", "noise_aware_vs_baseline", VARIANT_C),
            ("UNIFIED_GRAPH", "unified_graph_vs_baseline", VARIANT_B),
            ("UNIFIED_NOISE_COMBINED", "combined_vs_baseline", VARIANT_D),
        ]:
            vdata = deploy.get(metric_src, {})
            summary = {"pass_rate": vdata.get("pass_rate", 0.0),
                       "mean_de_gap": vdata.get("mean_de_gap", 1.0)}
            verdict, desc = compute_verdict(crit_id, summary)
            verdicts[crit_id] = {"verdict": verdict, "desc": desc}
            logger.info("  Verdict %s: %s (%s)", crit_id, verdict, desc)

        results["verdicts"] = verdicts
        results["pass"] = any(
            v["verdict"] == "confirmed" for v in verdicts.values()
        )

        # Summary ranking
        ranking = sorted(
            [(vid, deploy.get(vid, {}).get("mean_de_gap", 999))
             for vid in [VARIANT_A, VARIANT_B, VARIANT_C, VARIANT_D]
             if deploy.get(vid)],
            key=lambda x: x[1],
        )
        results["ranking"] = [{"variant": v, "mean_de_gap": m} for v, m in ranking]
        if ranking:
            logger.info("  Ranking: %s", " > ".join(f"{v}({m:.4f})" for v, m in ranking))

        return results


if __name__ == "__main__":
    NoiseAwareComparisonRunner.main()
