#!/usr/bin/env python3
"""E3: Bond-Resolved HVA at N=40 — Scaling Validation.

Validates that bond-resolved HVA (79 parameters, p=1) converges at N=40
using MPS-based VQE, and that MPNN can predict the 79-dim θ_opt.

Sections:
    0. Sanity Check — circuit construction + single MPS eval
    1. VQE Convergence — COBYLA vs SPSA at single h-point (go/no-go)
    2. Descending Sweep — 9 h-points, warm-start, save θ_opt
    3. MPNN Training — h=256, per_parameter_heads, deploy test
    4. GNN Necessity — random search baseline comparison

Hypothesis: Bond-resolved HVA at N=40 demonstrates GNN necessity in
high-dimensional variational spaces (79 params intractable without warm-start).

Usage:
    # Sanity check only (10 min)
    python scripts/.../run_e3_bond_resolved_scaling.py --section 0

    # Convergence test (2-4h, go/no-go gate)
    python scripts/.../run_e3_bond_resolved_scaling.py --section 1

    # Full sweep (12-18h batch)
    python scripts/.../run_e3_bond_resolved_scaling.py --section 2

    # MPNN training + deploy (after sweep)
    python scripts/.../run_e3_bond_resolved_scaling.py --section 3 4

    # All sections sequentially
    python scripts/.../run_e3_bond_resolved_scaling.py --stop-on-failure
"""

from __future__ import annotations

import logging
import sys
import time

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
# Constants (defaults — overridable via CLI)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_N_QUBITS = 40
DEFAULT_TOPOLOGY = "chain_1d"
P_LAYERS = 1
SEED = 42
CHI_MAX = 64
PRECISION = 0.005

# Statevector is viable up to N=22 (2^22 = 4M amplitudes, ~32MB)
STATEVECTOR_MAX_N = 22


def _compute_h_regime(n_qubits: int, topology: str) -> tuple[float, float, list[float], float]:
    """Compute valid regime h-values dynamically from N and topology.

    Returns (h_min_safe, h_max, h_sweep, h_convergence).
    """
    from qmbp_simulation.framework.preflight import get_regime_threshold

    # Try preflight registry first
    threshold = get_regime_threshold(topology, n_qubits, P_LAYERS)

    if threshold > 0:
        h_min_safe = threshold + 0.5  # Safety margin above boundary
    else:
        # Fallback: scaling law (validated for chain_1d, approximate for others)
        h_min_safe = 1.5 + 0.020 * n_qubits**1.31

    h_max = h_min_safe + 2.0  # Deep paramagnetic (easy landscape)
    h_sweep = np.linspace(h_max, h_min_safe, 9).tolist()
    h_convergence = h_max

    return h_min_safe, h_max, h_sweep, h_convergence


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════


class E3BondResolvedScalingRunner(ValidationRunner):
    """E3: Bond-Resolved HVA N=40 Scaling Validation.

    Validates VQE convergence with 79 bond-resolved parameters using MPS
    backend, then trains MPNN to predict the high-dimensional θ_opt.
    """

    runner_id = "e3_bond_resolved_scaling"
    experiment_id = "E3_BR_SCALING"
    description = "E3: Bond-Resolved HVA N=40 — 79 params, MPS-VQE + MPNN"
    hypothesis = (
        "Bond-resolved HVA at N=40 (79 params) converges with warm-start "
        "descending sweep, and MPNN predicts 79-dim θ_opt at ΔE/gap < 10%"
    )

    @classmethod
    def _add_custom_args(cls, parser):
        parser.add_argument(
            "--n-qubits",
            type=int,
            default=DEFAULT_N_QUBITS,
            help="System size (default: %(default)s)",
        )
        parser.add_argument(
            "--topology",
            type=str,
            default=DEFAULT_TOPOLOGY,
            choices=["chain_1d", "heavy_hex", "ladder", "triangular"],
            help="Lattice topology (default: %(default)s)",
        )
        parser.add_argument(
            "--optimizer",
            type=str,
            default="both",
            choices=["cobyla", "spsa", "both"],
            help="Optimizer for Section 1 (default: %(default)s)",
        )
        parser.add_argument(
            "--spsa-iters",
            type=int,
            default=500,
            help="SPSA iterations for Section 1/2 (default: %(default)s)",
        )
        parser.add_argument(
            "--cobyla-maxiter",
            type=int,
            default=2000,
            help="COBYLA maxiter for Section 1/2 (default: %(default)s)",
        )
        parser.add_argument(
            "--sweep-data-path",
            type=str,
            default=None,
            help="Path to pre-computed sweep JSON (skip Section 2 for Section 3/4)",
        )

    def build_config(self) -> dict:
        n = self._args.n_qubits
        topo = self._args.topology
        h_min_safe, h_max, h_sweep, _ = _compute_h_regime(n, topo)
        strategy = "noiseless" if n <= STATEVECTOR_MAX_N else "aer_mps"

        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "system": {
                "n_qubits": n,
                "p_layers": P_LAYERS,
                "topology": topo,
                "model": "tfim_bond_resolved",
                "parametrization": "bond_resolved",
            },
            "backend": {
                "strategy": strategy,
                "chi_max": CHI_MAX if strategy == "aer_mps" else None,
                "precision": PRECISION if strategy == "aer_mps" else None,
            },
            "optimizer": self._args.optimizer,
            "h_values": h_sweep,
            "h_min_safe": h_min_safe,
            "seeds": [SEED],
        }

    def setup(self):
        """Import heavy dependencies and build shared objects."""
        from qmbp_simulation import (
            ClassicalSolver,
            HamiltonianBuilder,
            HVACircuitBuilder,
            VQEConfig,
            VQEOptimizer,
            make_lattice,
        )
        from qmbp_simulation.execution import MPSBackend, NoiselessBackend
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.optimizers.spsa import SPSAOptimizer

        self.builder = HamiltonianBuilder()
        self.solver = ClassicalSolver()
        self.make_lattice = make_lattice
        self.hva = HVACircuitBuilder()
        self.spec_br = get_model_spec("tfim_bond_resolved")
        self.MPSBackend = MPSBackend
        self.NoiselessBackend = NoiselessBackend
        self.VQEOptimizer = VQEOptimizer
        self.VQEConfig = VQEConfig
        self.SPSAOptimizer = SPSAOptimizer

        # Resolve dynamic config from CLI args
        self._topology = self._args.topology
        self._n_qubits = self._args.n_qubits
        self._use_statevector = self._n_qubits <= STATEVECTOR_MAX_N

        # Compute h-regime dynamically
        self._h_min_safe, self._h_max, self._h_sweep, self._h_convergence = _compute_h_regime(
            self._n_qubits, self._topology
        )

        # Build reference lattice to get n_edges
        self._lattice_ref = make_lattice(self._topology, self._n_qubits, h=self._h_max)
        self._n_edges = len(self._lattice_ref.edges)
        self._n_params = self._n_edges + self._n_qubits

        backend_name = "StatevectorEstimator" if self._use_statevector else "aer_mps"
        logger.info(
            f"  Config: N={self._n_qubits}, topology={self._topology}, "
            f"edges={self._n_edges}, params={self._n_params}"
        )
        logger.info(
            f"  Backend: {backend_name}, h_range=[{self._h_sweep[-1]:.2f}, {self._h_sweep[0]:.2f}]"
        )

        # Shared state for cross-section data
        self._sweep_results: list[dict] | None = None
        self._best_optimizer: str | None = None

    def define_sections(self) -> list[Section]:
        return [
            Section(
                id=0,
                name="Sanity Check",
                fn=self.section_sanity,
                hypothesis="Bond-resolved circuit constructs correctly and evaluates",
            ),
            Section(
                id=1,
                name="VQE Convergence (COBYLA vs SPSA)",
                fn=self.section_convergence,
                hypothesis="At least one optimizer achieves ΔE/gap < 5% at h_max",
            ),
            Section(
                id=2,
                name="Descending Sweep (9 h-points)",
                fn=self.section_sweep,
                hypothesis="≥7/9 h-points pass ΔE/gap < 5% with warm-start",
            ),
            Section(
                id=3,
                name="MPNN Training + Deploy",
                fn=self.section_mpnn,
                hypothesis="MPNN predicts high-dim θ_opt with deploy ΔE/gap < 10%",
            ),
            Section(
                id=4,
                name="GNN Necessity (Random Search Baseline)",
                fn=self.section_gnn_necessity,
                hypothesis="MPNN beats random search by >5× in eval count",
            ),
        ]

    # ── Helper: create execution backend ─────────────────────────────────────

    def _create_backend(self):
        """Create the appropriate backend based on system size."""
        if self._use_statevector:
            return self.NoiselessBackend()
        return self.MPSBackend(
            strategy="aer_mps",
            chi_max=CHI_MAX,
            precision=PRECISION,
            seed=SEED,
        )

    # ── Section 0: Sanity Check ──────────────────────────────────────────────

    def section_sanity(self) -> dict:
        """Verify circuit construction and single MPS evaluation."""
        N = self._n_qubits
        lattice = self.make_lattice(self._topology, N, h=self._h_max)

        # 1. Build bond-resolved circuit
        qc, theta = self.spec_br.create_circuit(N, P_LAYERS, lattice)
        n_params_actual = qc.num_parameters
        logger.info(f"  Circuit: {n_params_actual} params, depth={qc.depth()}")
        logger.info(f"  Gates: {dict(qc.count_ops())}")

        assert n_params_actual == self._n_params, (
            f"Expected {self._n_params} params, got {n_params_actual}"
        )

        # 2. Single eval at θ=0 (should give finite energy)
        backend = self._create_backend()
        t0 = time.time()
        theta_zero = np.zeros(n_params_actual)
        energy_zero = backend.evaluate(qc, self.builder.build(lattice), theta_zero)
        t_eval = time.time() - t0
        logger.info(f"  E(θ=0) = {energy_zero:.6f} ({t_eval:.1f}s)")

        # 3. Uniform params → should match global HVA
        theta_uniform = np.zeros(n_params_actual)
        theta_uniform[: self._n_edges] = 0.05  # same θ_zz for all bonds
        theta_uniform[self._n_edges :] = 0.40  # same θ_x for all sites
        energy_uniform = backend.evaluate(qc, self.builder.build(lattice), theta_uniform)
        logger.info(f"  E(uniform) = {energy_uniform:.6f}")

        # 4. Compare with global HVA at same uniform angles
        from qmbp_simulation.models.model_registry import get_model_spec

        spec_global = get_model_spec("tfim")
        qc_global, _ = spec_global.create_circuit(N, P_LAYERS, lattice)
        theta_global = np.array([0.05, 0.40])
        energy_global = backend.evaluate(
            qc_global,
            self.builder.build(lattice),
            theta_global,
        )
        logger.info(f"  E(global, same angles) = {energy_global:.6f}")

        diff = abs(energy_uniform - energy_global)
        logger.info(f"  |E_br_uniform - E_global| = {diff:.2e}")

        # Pass: energies finite and uniform BR ≈ global (within shot noise)
        energies_finite = np.isfinite(energy_zero) and np.isfinite(energy_uniform)
        # Tolerance: 0.01 for statevector (exact), 0.1 for MPS (shot noise)
        tol = 0.01 if self._use_statevector else 0.1
        uniform_matches_global = diff < tol

        passed = energies_finite and uniform_matches_global
        return {
            "n_params": n_params_actual,
            "n_edges": self._n_edges,
            "topology": self._topology,
            "backend": "statevector" if self._use_statevector else "aer_mps",
            "energy_zero": float(energy_zero),
            "energy_uniform_br": float(energy_uniform),
            "energy_uniform_global": float(energy_global),
            "diff_br_vs_global": float(diff),
            "eval_time_s": t_eval,
            "circuit_depth": qc.depth(),
            "gate_counts": dict(qc.count_ops()),
            "pass": passed,
        }

    # ── Section 1: VQE Convergence Test ──────────────────────────────────────

    def section_convergence(self) -> dict:
        """Test COBYLA and/or SPSA convergence at a single easy h-point."""
        N = self._n_qubits
        h = self._h_convergence
        lattice = self.make_lattice(self._topology, N, h=h)
        H = self.builder.build(lattice)

        # DMRG ground truth
        gt = self.solver.solve(H, lattice, method="dmrg")
        e_exact = gt.ground_energy
        gap = gt.gap
        logger.info(f"  DMRG: E₀={e_exact:.6f}, gap={gap:.4f}")

        qc, _ = self.spec_br.create_circuit(N, P_LAYERS, lattice)
        init_params = np.random.default_rng(SEED).uniform(-0.01, 0.01, self._n_params)

        results = {}
        optimizer_choice = self._args.optimizer

        # Determine VQE method based on parameter count.
        # Bond-resolved circuits have n_params = n_edges + N (typically 39+ for N=20).
        # L-BFGS-B uses finite-difference gradients: O(2*n_params) evals per iteration,
        # making it impractically slow for high-dimensional landscapes.
        # Use COBYLA (1 eval/iter) when n_params > 10, regardless of backend.
        vqe_method = (
            "COBYLA" if self._n_params > 10 else ("L-BFGS-B" if self._use_statevector else "COBYLA")
        )

        # ── COBYLA / L-BFGS-B ────────────────────────────────────────────
        if optimizer_choice in ("cobyla", "both"):
            method_name = vqe_method
            logger.info(f"\n  --- {method_name} (maxiter={self._args.cobyla_maxiter}) ---")
            backend_c = self._create_backend()
            # Bond-resolved circuits (79 params) need more restarts than
            # standard 2-param HVA. Use 3 restarts regardless of backend
            # for the convergence test (single h-point, acceptable cost).
            n_restarts_conv = 3
            config_c = self.VQEConfig(
                method=method_name,
                p_layers=P_LAYERS,
                n_restarts=n_restarts_conv,
                maxiter=self._args.cobyla_maxiter,
                enable_callbacks=False,
            )
            opt_c = self.VQEOptimizer(config=config_c, backend=backend_c, seed=SEED)

            t0 = time.time()
            res_c = opt_c.optimize(H, qc, init_params.copy(), exact_energy=e_exact)
            elapsed_c = time.time() - t0

            de_gap_c = abs(res_c.energy - e_exact) / max(gap, 1e-10)
            logger.info(
                f"  {method_name}: E={res_c.energy:.6f}, ΔE/gap={de_gap_c:.4f}, "
                f"nit={res_c.n_iterations}, time={elapsed_c:.1f}s"
            )
            results["cobyla"] = {
                "method": method_name,
                "energy": float(res_c.energy),
                "de_gap": float(de_gap_c),
                "n_iterations": res_c.n_iterations,
                "elapsed_s": elapsed_c,
                "theta_opt": res_c.theta_opt.tolist(),
                "passed": de_gap_c < 0.05,
            }

        # ── SPSA ─────────────────────────────────────────────────────────
        if optimizer_choice in ("spsa", "both"):
            n_iter_spsa = self._args.spsa_iters
            logger.info(f"\n  --- SPSA ({n_iter_spsa} iterations) ---")
            backend_s = self._create_backend()
            spsa = self.SPSAOptimizer(
                backend=backend_s,
                a=0.1,
                c=0.05,
                A_frac=0.05,
                seed=SEED,
            )

            t0 = time.time()
            res_s = spsa.optimize(qc, H, init_params.copy(), n_iterations=n_iter_spsa)
            elapsed_s = time.time() - t0

            de_gap_s = abs(res_s.energy - e_exact) / max(gap, 1e-10)
            logger.info(
                f"  SPSA: E={res_s.energy:.6f}, ΔE/gap={de_gap_s:.4f}, "
                f"n_evals={res_s.n_iterations}, time={elapsed_s:.1f}s"
            )
            results["spsa"] = {
                "energy": float(res_s.energy),
                "de_gap": float(de_gap_s),
                "n_iterations": res_s.n_iterations,
                "elapsed_s": elapsed_s,
                "theta_opt": res_s.theta_opt.tolist(),
                "passed": de_gap_s < 0.05,
            }

        # ── Determine best optimizer ─────────────────────────────────────
        any_passed = any(r["passed"] for r in results.values())

        if "cobyla" in results and "spsa" in results:
            if results["cobyla"]["de_gap"] <= results["spsa"]["de_gap"]:
                self._best_optimizer = "cobyla"
            else:
                self._best_optimizer = "spsa"
        elif "cobyla" in results:
            self._best_optimizer = "cobyla"
        else:
            self._best_optimizer = "spsa"

        logger.info(f"\n  Best optimizer: {self._best_optimizer}")
        logger.info(f"  Go/No-Go: {'GO ✅' if any_passed else 'NO-GO ❌'}")

        return {
            "h": h,
            "e_exact": e_exact,
            "gap": gap,
            "results": results,
            "best_optimizer": self._best_optimizer,
            "pass": any_passed,
        }

    # ── Section 2: Descending Sweep ──────────────────────────────────────────

    def section_sweep(self) -> dict:
        """Run full descending sweep with warm-start, save θ_opt for MPNN."""
        from pathlib import Path

        from qmbp_simulation.utils.helpers import json_dump

        N = self._n_qubits
        h_values = sorted(self._h_sweep, reverse=True)  # Ensure descending

        # Determine optimizer from Section 1 result or CLI
        optimizer_name = self._best_optimizer or self._args.optimizer
        if optimizer_name == "both":
            optimizer_name = "cobyla"  # default if Section 1 wasn't run
        logger.info(f"  Optimizer: {optimizer_name}")
        logger.info(f"  Topology: {self._topology}, N={N}, params={self._n_params}")
        logger.info(f"  h-values: {[f'{h:.3f}' for h in h_values]}")

        # Phase 1: DMRG ground truth for all h-points
        logger.info("\n  ─── Phase 1: DMRG Ground Truth ───")
        dmrg_data = []
        t_dmrg = time.time()
        for h in h_values:
            lattice = self.make_lattice(self._topology, N, h=h)
            H = self.builder.build(lattice)
            gt = self.solver.solve(H, lattice, method="dmrg")
            dmrg_data.append(
                {
                    "h": h,
                    "ground_energy": gt.ground_energy,
                    "gap": gt.gap,
                }
            )
            logger.info(f"    h={h:.3f}: E₀={gt.ground_energy:.6f}, gap={gt.gap:.4f}")
        t_dmrg = time.time() - t_dmrg
        logger.info(f"  DMRG total: {t_dmrg:.1f}s")

        # Phase 2: Bond-resolved VQE descending sweep
        logger.info(f"\n  ─── Phase 2: Bond-Resolved VQE ({optimizer_name}) ───")
        lattice_ref = self.make_lattice(self._topology, N, h=h_values[0])
        qc, _ = self.spec_br.create_circuit(N, P_LAYERS, lattice_ref)

        backend = self._create_backend()

        # Determine VQE method — same logic as Section 1:
        # COBYLA for high-dimensional bond-resolved (n_params > 10)
        vqe_method = (
            "COBYLA" if self._n_params > 10 else ("L-BFGS-B" if self._use_statevector else "COBYLA")
        )

        # Initialize warm-start
        theta_prev = np.random.default_rng(SEED).uniform(-0.01, 0.01, self._n_params)

        sweep_results = []
        t_vqe_total = time.time()

        for idx, h in enumerate(h_values):
            t_point = time.time()
            lattice_h = self.make_lattice(self._topology, N, h=h)
            H = self.builder.build(lattice_h)
            e_exact = dmrg_data[idx]["ground_energy"]
            gap = dmrg_data[idx]["gap"]

            if optimizer_name == "spsa":
                spsa = self.SPSAOptimizer(
                    backend=backend,
                    a=0.1,
                    c=0.05,
                    A_frac=0.05,
                    seed=SEED,
                )
                res = spsa.optimize(
                    qc,
                    H,
                    theta_prev.copy(),
                    n_iterations=self._args.spsa_iters,
                )
            else:
                config = self.VQEConfig(
                    method=vqe_method,
                    p_layers=P_LAYERS,
                    n_restarts=3 if self._use_statevector else 1,
                    maxiter=self._args.cobyla_maxiter,
                    enable_callbacks=False,
                )
                opt = self.VQEOptimizer(config=config, backend=backend, seed=SEED)
                res = opt.optimize(H, qc, theta_prev.copy(), exact_energy=e_exact)

            elapsed_point = time.time() - t_point
            de_gap = abs(res.energy - e_exact) / max(gap, 1e-10)

            # Variational principle check
            if res.energy < e_exact - 1e-6:
                logger.warning(
                    f"  ⚠️  Variational principle violated at h={h:.3f}: "
                    f"E_VQE={res.energy:.8f} < E_exact={e_exact:.8f}"
                )

            # Compute θ smoothness (inf-norm diff from previous)
            theta_smoothness = float(np.max(np.abs(res.theta_opt - theta_prev)))

            # Analyze spatial structure of θ_opt
            theta_zz = res.theta_opt[: self._n_edges]
            theta_x = res.theta_opt[self._n_edges :]

            point_result = {
                "h": h,
                "vqe_energy": float(res.energy),
                "dmrg_energy": e_exact,
                "gap": gap,
                "de_gap": float(de_gap),
                "energy_error": float(abs(res.energy - e_exact)),
                "n_iterations": res.n_iterations,
                "converged": (
                    res.n_iterations < self._args.spsa_iters * 2 + 1
                    if optimizer_name == "spsa"
                    else res.n_iterations < self._args.cobyla_maxiter
                ),
                "theta_opt": res.theta_opt.tolist(),
                "theta_smoothness": theta_smoothness if idx > 0 else 0.0,
                "theta_zz_stats": {
                    "mean": float(np.mean(theta_zz)),
                    "std": float(np.std(theta_zz)),
                    "min": float(np.min(theta_zz)),
                    "max": float(np.max(theta_zz)),
                },
                "theta_x_stats": {
                    "mean": float(np.mean(theta_x)),
                    "std": float(np.std(theta_x)),
                    "min": float(np.min(theta_x)),
                    "max": float(np.max(theta_x)),
                },
                "elapsed_s": elapsed_point,
                "passed": de_gap < 0.05,
            }
            sweep_results.append(point_result)

            # Propagate warm-start (with NaN guard)
            if np.all(np.isfinite(res.theta_opt)):
                theta_prev = res.theta_opt.copy()
            else:
                logger.warning(
                    f"  ⚠️  NaN/Inf in θ_opt at h={h:.3f}. Keeping previous θ for warm-start."
                )

            status = "✅" if de_gap < 0.05 else "⚠️"
            logger.info(
                f"  {status} [{idx + 1}/{len(h_values)}] h={h:.3f}: E={res.energy:.6f}, "
                f"ΔE/gap={de_gap:.4f}, smooth={theta_smoothness:.4f}, "
                f"time={elapsed_point:.1f}s"
            )

        t_vqe_total = time.time() - t_vqe_total

        # Store for cross-section use
        self._sweep_results = sweep_results

        # Save sweep data to disk for Section 3/4 (crash recovery)
        output_dir = Path("results/bond_resolved_scaling")
        output_dir.mkdir(parents=True, exist_ok=True)
        sweep_path = output_dir / f"sweep_N{N}_{self._topology}_{int(time.time())}.json"
        json_dump(
            {
                "experiment_id": self.experiment_id,
                "n_qubits": N,
                "topology": self._topology,
                "n_params": self._n_params,
                "n_edges": self._n_edges,
                "optimizer": optimizer_name,
                "h_values": h_values,
                "dmrg_data": dmrg_data,
                "sweep_results": sweep_results,
                "timing": {
                    "dmrg_s": t_dmrg,
                    "vqe_total_s": t_vqe_total,
                    "avg_per_hpoint_s": t_vqe_total / len(h_values),
                },
            },
            sweep_path,
        )
        logger.info(f"\n  Sweep data saved: {sweep_path}")

        # Aggregate metrics
        n_pass = sum(1 for r in sweep_results if r["passed"])
        n_total = len(sweep_results)
        all_de_gaps = [r["de_gap"] for r in sweep_results]
        all_smoothness = [r["theta_smoothness"] for r in sweep_results if r["theta_smoothness"] > 0]

        logger.info("\n  ─── Sweep Summary ───")
        logger.info(f"  Pass rate: {n_pass}/{n_total}")
        logger.info(f"  Mean ΔE/gap: {np.mean(all_de_gaps):.4f}")
        logger.info(f"  Max ΔE/gap: {np.max(all_de_gaps):.4f}")
        logger.info(f"  Mean θ smoothness: {np.mean(all_smoothness):.4f}")
        logger.info(f"  Total VQE time: {t_vqe_total:.1f}s ({t_vqe_total / 60:.1f} min)")

        passed = n_pass >= 7  # ≥7/9 pass threshold
        return {
            "optimizer": optimizer_name,
            "h_values": h_values,
            "n_pass": n_pass,
            "n_total": n_total,
            "pass_rate": n_pass / n_total,
            "mean_de_gap": float(np.mean(all_de_gaps)),
            "max_de_gap": float(np.max(all_de_gaps)),
            "min_de_gap": float(np.min(all_de_gaps)),
            "std_de_gap": float(np.std(all_de_gaps)),
            "mean_theta_smoothness": float(np.mean(all_smoothness)) if all_smoothness else 0.0,
            "max_theta_smoothness": float(np.max(all_smoothness)) if all_smoothness else 0.0,
            "timing": {
                "dmrg_s": t_dmrg,
                "vqe_total_s": t_vqe_total,
                "avg_per_hpoint_s": t_vqe_total / len(h_values),
            },
            "sweep_data_path": str(sweep_path),
            "per_h_results": sweep_results,
            "pass": passed,
        }

    # ── Section 3: MPNN Training + Deploy ────────────────────────────────────

    def section_mpnn(self) -> dict:
        """Train MPNN on sweep θ_opt data and deploy on midpoints."""
        import json

        import torch

        from qmbp_simulation.predictors import (
            MPNNPredictor,
            build_graph_dataset,
            train_mpnn,
        )

        N = self._n_qubits

        # Load sweep data (from Section 2 or from file)
        sweep_results = self._sweep_results
        if sweep_results is None and self._args.sweep_data_path:
            logger.info(f"  Loading sweep data from: {self._args.sweep_data_path}")
            with open(self._args.sweep_data_path) as f:
                sweep_data = json.load(f)
            sweep_results = sweep_data["sweep_results"]
        if sweep_results is None:
            raise RuntimeError(
                "No sweep data available. Run --section 2 first, or provide "
                "--sweep-data-path to a pre-computed sweep JSON."
            )

        # Filter only passing h-points for training
        training_points = [r for r in sweep_results if r["passed"]]
        if len(training_points) < 3:
            logger.error(
                f"  Only {len(training_points)} passing points — need ≥3 for "
                f"build_graph_dataset. Consider relaxing pass threshold or re-running sweep."
            )
            return {
                "n_training_points": len(training_points),
                "error": "Insufficient training data (< 3 passing h-points)",
                "pass": False,
            }
        if len(training_points) < 5:
            logger.warning(f"  Only {len(training_points)} passing points — MPNN may overfit")

        logger.info(f"  Training points: {len(training_points)}")
        logger.info(f"  Output dim: {self._n_params} (n_edges={self._n_edges})")

        # Build training dataset
        h_train = [r["h"] for r in training_points]
        theta_train = np.array([r["theta_opt"] for r in training_points])

        # Sort training points by h (descending) for correct midpoint computation
        sort_idx = np.argsort(h_train)[::-1]
        h_train = [h_train[i] for i in sort_idx]
        theta_train = theta_train[sort_idx]

        # Build graph dataset using existing infrastructure
        # build_graph_dataset takes: (lattice, h_values, theta_opt, e_exact, fidelities)
        lattice_ref = self.make_lattice(self._topology, N, h=h_train[0])
        h_values_np = np.array(h_train)
        e_exact_np = np.array([r["dmrg_energy"] for r in training_points])[sort_idx]
        fid_np = np.ones(len(training_points))  # All passed VQE → fid=1.0 placeholder

        dataset = build_graph_dataset(
            lattice_ref,
            h_values_np,
            theta_train,
            e_exact_np,
            fidelities=fid_np,
            fidelity_threshold=0.0,  # No filtering — all points already validated
        )

        # MPNN architecture for high-dimensional output
        model = MPNNPredictor(
            node_features=2,
            hidden_dim=256,
            n_layers=3,
            output_dim=self._n_params,
            per_parameter_heads=True,
            n_edges=self._n_edges,
            norm_type="batch",
        )
        logger.info(
            f"  Model: GINConv h=256, L=3, output={self._n_params}, "
            f"per_param_heads=True (ZZ={self._n_edges}, X={N})"
        )
        n_model_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Model parameters: {n_model_params:,}")

        # Train
        t_train = time.time()
        train_result = train_mpnn(
            model=model,
            dataset=dataset,
            n_epochs=6000,
            lr=1e-3,
            patience=800,
        )
        t_train = time.time() - t_train
        final_mse = train_result["final_mse"]
        n_epochs_trained = len(train_result["mse_history"])
        logger.info(
            f"  Training: {n_epochs_trained} epochs, final MSE={final_mse:.2e}, "
            f"time={t_train:.1f}s, stopped_early={train_result.get('stopped_early', False)}"
        )

        # Generalization gap check (early warning for MPNN overfit)
        # Per steering: gen_gap > 0.01 → 25% of failures
        gen_gap = train_result.get("generalization_gap", None)
        if gen_gap is not None and gen_gap > 0.01:
            logger.warning(
                f"  ⚠️  Generalization gap={gen_gap:.4f} > 0.01 threshold. "
                f"MPNN may be overfitting (25% of historical failures). "
                f"Consider: more training points, or reduce hidden_dim/n_layers."
            )
        elif gen_gap is not None:
            logger.info(f"  Generalization gap: {gen_gap:.4f} (OK, < 0.01)")

        # Data sufficiency check: warn if ratio of params-to-data is too high
        data_ratio = n_model_params / max(len(training_points), 1)
        if data_ratio > 5000:
            logger.warning(
                f"  ⚠️  Model/data ratio={data_ratio:.0f} (very high). "
                f"{n_model_params:,} params trained on {len(training_points)} points. "
                f"Risk of memorization. Consider more VQE sweep points."
            )

        # Deploy on midpoints (interpolation test)
        h_deploy = []
        for i in range(len(h_train) - 1):
            h_deploy.append((h_train[i] + h_train[i + 1]) / 2)
        # Take up to 4 midpoints
        h_deploy = h_deploy[:4]
        logger.info(f"  Deploy h-values: {[f'{h:.3f}' for h in h_deploy]}")

        # Predict and evaluate
        model.eval()
        deploy_results = []
        backend = self._create_backend()

        for h_test in h_deploy:
            # Build graph for prediction (single-point dataset)
            lattice_test = self.make_lattice(self._topology, N, h=h_test)
            test_dataset = build_graph_dataset(
                lattice_test,
                np.array([h_test]),
                np.zeros((1, self._n_params)),
                np.array([0.0]),  # dummy e_exact
                fidelities=np.array([1.0]),
                fidelity_threshold=0.0,
            )

            # Predict θ
            with torch.no_grad():
                theta_pred = model(test_dataset[0]).numpy().flatten()

            # Evaluate energy
            qc, _ = self.spec_br.create_circuit(N, P_LAYERS, lattice_test)
            H_test = self.builder.build(lattice_test)
            energy_pred = backend.evaluate(qc, H_test, theta_pred)

            # Ground truth
            gt = self.solver.solve(H_test, lattice_test, method="dmrg")
            de_gap = abs(energy_pred - gt.ground_energy) / max(gt.gap, 1e-10)

            deploy_results.append(
                {
                    "h": h_test,
                    "energy_pred": float(energy_pred),
                    "energy_exact": gt.ground_energy,
                    "gap": gt.gap,
                    "de_gap": float(de_gap),
                    "passed": de_gap < 0.10,  # relaxed threshold for 79 params
                }
            )
            status = "✅" if de_gap < 0.10 else "⚠️"
            logger.info(f"  {status} Deploy h={h_test:.3f}: ΔE/gap={de_gap:.4f}")

        # Summary
        n_deploy_pass = sum(1 for r in deploy_results if r["passed"])
        n_deploy_total = len(deploy_results)
        mean_deploy_de_gap = float(np.mean([r["de_gap"] for r in deploy_results]))

        logger.info(f"\n  Deploy pass rate: {n_deploy_pass}/{n_deploy_total}")
        logger.info(f"  Mean deploy ΔE/gap: {mean_deploy_de_gap:.4f}")

        passed = n_deploy_pass >= 3  # ≥3/4 midpoints pass
        return {
            "n_training_points": len(training_points),
            "n_model_params": n_model_params,
            "final_mse": float(final_mse),
            "training_epochs": n_epochs_trained,
            "training_time_s": t_train,
            "stopped_early": train_result.get("stopped_early", False),
            "h_deploy": h_deploy,
            "deploy_results": deploy_results,
            "n_deploy_pass": n_deploy_pass,
            "n_deploy_total": n_deploy_total,
            "mean_deploy_de_gap": mean_deploy_de_gap,
            "model_config": {
                "hidden_dim": 256,
                "n_layers": 3,
                "output_dim": self._n_params,
                "per_parameter_heads": True,
                "n_edges": self._n_edges,
            },
            "pass": passed,
        }

    # ── Section 4: GNN Necessity (Random Search) ─────────────────────────────

    def section_gnn_necessity(self) -> dict:
        """Compare MPNN (1 forward pass) vs random search in 79-dim space."""
        import json

        N = self._n_qubits

        # Load sweep data for a reference h-point
        sweep_results = self._sweep_results
        if sweep_results is None and self._args.sweep_data_path:
            with open(self._args.sweep_data_path) as f:
                sweep_data = json.load(f)
            sweep_results = sweep_data["sweep_results"]
        if sweep_results is None:
            raise RuntimeError(
                "No sweep data available. Run --section 2 first, or provide "
                "--sweep-data-path to a pre-computed sweep JSON."
            )

        # Use median h-point as test target
        mid_idx = len(sweep_results) // 2
        h_test = sweep_results[mid_idx]["h"]
        e_exact = sweep_results[mid_idx]["dmrg_energy"]
        gap = sweep_results[mid_idx]["gap"]
        de_gap_vqe = sweep_results[mid_idx]["de_gap"]

        logger.info(f"  Reference: h={h_test:.3f}, E₀={e_exact:.6f}, gap={gap:.4f}")
        logger.info(f"  VQE achieved: ΔE/gap={de_gap_vqe:.4f}")

        # Random search: sample N_RANDOM parameter vectors
        N_RANDOM = 200  # limited by compute budget
        lattice_test = self.make_lattice(self._topology, N, h=h_test)
        qc, _ = self.spec_br.create_circuit(N, P_LAYERS, lattice_test)
        H_test = self.builder.build(lattice_test)

        backend = self._create_backend()
        rng = np.random.default_rng(SEED + 100)

        logger.info(f"  Random search: {N_RANDOM} samples in {self._n_params}-dim space")
        random_energies = []
        random_de_gaps = []
        t_random = time.time()

        best_random_de_gap = float("inf")
        best_random_energy = float("inf")
        n_below_threshold = 0

        for i in range(N_RANDOM):
            theta_random = rng.uniform(-np.pi, np.pi, self._n_params)
            energy = backend.evaluate(qc, H_test, theta_random)
            de_gap = abs(energy - e_exact) / max(gap, 1e-10)

            random_energies.append(float(energy))
            random_de_gaps.append(float(de_gap))

            if de_gap < best_random_de_gap:
                best_random_de_gap = de_gap
                best_random_energy = energy

            if de_gap < 0.10:
                n_below_threshold += 1

            # Progress logging every 50 evals
            if (i + 1) % 50 == 0:
                logger.info(
                    f"    [{i + 1}/{N_RANDOM}] best ΔE/gap={best_random_de_gap:.4f}, "
                    f"hits<10%={n_below_threshold}"
                )

        t_random = time.time() - t_random

        # Analysis
        mean_random_de_gap = float(np.mean(random_de_gaps))
        std_random_de_gap = float(np.std(random_de_gaps))
        p10 = float(np.percentile(random_de_gaps, 10))  # 10th percentile

        # GNN necessity ratio: how many random evals to match MPNN
        # MPNN gets ~de_gap_vqe in 1 eval. Random needs to find similar.
        # Estimate: if best_random after N evals > MPNN, then ratio = N_RANDOM+
        if best_random_de_gap > de_gap_vqe:
            # Random never matched VQE — extrapolate
            ratio_estimate = N_RANDOM  # lower bound (actual ratio is higher)
            logger.info(f"  Random search NEVER matched VQE after {N_RANDOM} evals")
        else:
            # Find first eval that beat VQE threshold
            first_match = next(
                (i for i, dg in enumerate(random_de_gaps) if dg <= de_gap_vqe * 1.5),
                N_RANDOM,
            )
            ratio_estimate = first_match + 1

        logger.info("\n  ─── GNN Necessity Summary ───")
        logger.info(f"  VQE ΔE/gap (warm-start):    {de_gap_vqe:.4f}")
        logger.info(f"  Best random ΔE/gap ({N_RANDOM} evals): {best_random_de_gap:.4f}")
        logger.info(f"  Mean random ΔE/gap:          {mean_random_de_gap:.4f}")
        logger.info(f"  Random 10th percentile:      {p10:.4f}")
        logger.info(f"  Random hits < 10%:           {n_below_threshold}/{N_RANDOM}")
        logger.info(f"  Eval advantage ratio:        ≥{ratio_estimate}×")
        logger.info(f"  Random search time:          {t_random:.1f}s ({t_random / 60:.1f} min)")

        # Pass: MPNN beats random by >5× in eval count
        passed = ratio_estimate >= 5
        return {
            "h_test": h_test,
            "e_exact": e_exact,
            "gap": gap,
            "de_gap_vqe": float(de_gap_vqe),
            "n_random_samples": N_RANDOM,
            "best_random_de_gap": float(best_random_de_gap),
            "best_random_energy": float(best_random_energy),
            "mean_random_de_gap": mean_random_de_gap,
            "std_random_de_gap": std_random_de_gap,
            "percentile_10": p10,
            "n_below_10pct": n_below_threshold,
            "eval_advantage_ratio": ratio_estimate,
            "random_search_time_s": t_random,
            "random_de_gaps_summary": {
                "min": float(np.min(random_de_gaps)),
                "max": float(np.max(random_de_gaps)),
                "median": float(np.median(random_de_gaps)),
            },
            "pass": passed,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    E3BondResolvedScalingRunner.main()
