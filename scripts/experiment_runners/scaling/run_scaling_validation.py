#!/usr/bin/env python3
"""MPS Scaling Validation — DMRG ground truth + MPS-VQE at N>30.

Validates that MPS-based VQE converges to DMRG ground truth across
a range of N values and h-points for 1D TFIM.

Sections:
    1. DMRG Ground Truth: Compute E₀ and gap for each h-point
    2. MPS-VQE Sweep: Descending warm-start VQE with MPSBackend

Success criterion: ΔE/gap < 5% for all h-points in valid regime.

Usage:
    python scripts/experiment_runners/scaling/run_scaling_validation.py \\
        --n-qubits 40 --topology chain_1d

    python scripts/experiment_runners/scaling/run_scaling_validation.py \\
        --n-qubits 100 --h-min 3.0 --h-max 5.0 --h-points 10

    python scripts/experiment_runners/scaling/run_scaling_validation.py --dry-run
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

from qmbp_simulation.models.constants import DE_GAP_THRESHOLD

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_N = 40
DEFAULT_TOPOLOGY = "chain_1d"
DEFAULT_MODEL = "tfim"
DEFAULT_P = 1
DEFAULT_MAXITER = 500
DEFAULT_N_RESTARTS = 3


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════


class MPSScalingValidationRunner(ValidationRunner):
    """Validate MPS-VQE convergence to DMRG ground truth at large N.

    Uses setup_physics() for all standard objects. The key difference from
    NoiselessPipelineRunner is that this ALWAYS uses MPSBackend (N>22) and
    only runs Phase 1+2 (no MPNN training or deployment).
    """

    runner_id = "mps_scaling_validation_v2"
    experiment_id = "scaling/validation"
    description = "MPS-VQE convergence validation at N>30 vs DMRG ground truth"
    hypothesis = (
        "MPS-VQE with chi=64 achieves ΔE/gap < 5% at all h-points "
        "within the valid regime predicted by the scaling law."
    )

    @classmethod
    def _add_custom_args(cls, parser):
        parser.add_argument("--n-qubits", type=int, default=DEFAULT_N, help="System size")
        parser.add_argument("--topology", type=str, default=DEFAULT_TOPOLOGY)
        parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
        parser.add_argument("--p-layers", type=int, default=DEFAULT_P, choices=[1, 2, 3, 4])
        parser.add_argument(
            "--h-min", type=float, default=None, help="Min h (auto from scaling law if not given)"
        )
        parser.add_argument("--h-max", type=float, default=None, help="Max h (auto: h_min + 1.5)")
        parser.add_argument("--h-points", type=int, default=8, help="Number of h-values")
        parser.add_argument(
            "--chi-max", type=int, default=None, help="MPS bond dimension (default: from constants)"
        )
        parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
        parser.add_argument("--n-restarts", type=int, default=DEFAULT_N_RESTARTS)
        parser.add_argument("--seeds", type=int, nargs="+", default=[42])
        parser.add_argument(
            "--verify-chi",
            action="store_true",
            help="Run chi-convergence test: re-evaluate best theta with 2×chi and compare energies.",
        )

    def run_preflight(self) -> bool:
        """Validate scaling configuration."""
        N = self._args.n_qubits
        if N < 10:
            logger.error("Scaling validation is for N>=10. Use noiseless pipeline for small N.")
            return False
        # Auto-warn for 2D topologies without --verify-chi
        _2D_TOPOS = ("square", "triangular", "heavy_hex", "kagome")
        chi = self._args.chi_max or 64
        if self._args.topology in _2D_TOPOS and N > 16 and not self._args.verify_chi:
            logger.warning(
                "⚠️  Running 2D topology '%s' at N=%d without --verify-chi. "
                "Chi=%d may be insufficient for 2D. Consider adding --verify-chi.",
                self._args.topology,
                N,
                chi,
            )
        return True

    def build_config(self) -> dict:
        return {
            "runner_id": self.runner_id,
            "system": {
                "n_qubits": self._args.n_qubits,
                "topology": self._args.topology,
                "model": self._args.model,
                "p_layers": self._args.p_layers,
            },
            "mps": {"chi_max": getattr(self, "_chi_max", 64)},
            "vqe": {"maxiter": self._args.maxiter, "n_restarts": self._args.n_restarts},
            "seeds": self._args.seeds,
            "h_grid": {
                "h_min": self._h_min,
                "h_max": self._h_max,
                "h_points": self._args.h_points,
            },
            "scaling_law": {
                "formula": "h_min_safe = 1.5 + 0.020 * N^1.31",
                "predicted_h_min": self._h_min_predicted,
            },
        }

    def setup(self):
        """Initialize physics objects and compute h-grid."""
        self.setup_physics()

        N = self._args.n_qubits

        # Compute h-grid from scaling law if not explicitly provided
        # NOTE: This scaling law was calibrated for TFIM chain_1d p=1 only.
        self._h_min_predicted = 1.5 + 0.020 * N**1.31
        if self._args.h_min is None and (
            self._args.model != "tfim"
            or self._args.topology != "chain_1d"
            or self._args.p_layers != 1
        ):
            logger.warning(
                "  ⚠️  Scaling law h_min = 1.5 + 0.020·N^1.31 was calibrated for "
                "TFIM chain_1d p=1 only. Current config: model=%s, topology=%s, p=%d. "
                "Consider providing explicit --h-min/--h-max.",
                self._args.model,
                self._args.topology,
                self._args.p_layers,
            )
        self._h_min = (
            self._args.h_min if self._args.h_min is not None else self._h_min_predicted + 0.5
        )
        self._h_max = self._args.h_max if self._args.h_max is not None else self._h_min + 1.5
        # Round to 2 decimals for cache key stability (matches GroundTruthCache)
        self._h_values = [
            round(h, 2) for h in np.linspace(self._h_max, self._h_min, self._args.h_points)
        ]

        logger.info(f"  Scaling law h_min_safe = {self._h_min_predicted:.3f} for N={N}")
        logger.info(
            f"  h-grid: {self._args.h_points} points in [{self._h_min:.3f}, {self._h_max:.3f}]"
        )

        # Select MPS backend (forced — this is a scaling runner)
        from qmbp_simulation.models.constants import MPS_DEFAULT_CHI_MAX

        chi_max = self._args.chi_max if self._args.chi_max is not None else MPS_DEFAULT_CHI_MAX
        self._chi_max = chi_max
        self._backend = self.MPSBackend(
            strategy="aer_mps",
            chi_max=chi_max,
            seed=self._args.seeds[0],
        )
        logger.info(f"  Backend: MPSBackend(chi={chi_max}, strategy=aer_mps)")

        # Set experiment_id dynamically
        from qmbp_simulation.framework.result_io import build_experiment_id

        self.experiment_id = build_experiment_id(
            category="scaling/validation",
            model=self._args.model,
            topology=self._args.topology,
        )

        # Shared state
        self._dmrg_data: list[dict] = []

    def define_sections(self) -> list[Section]:
        sections = [
            Section(
                id=1,
                name="DMRG Ground Truth",
                fn=self.section_dmrg,
                hypothesis="DMRG computes converged E₀ and gap for all h-points",
            ),
            Section(
                id=2,
                name="MPS-VQE Descending Sweep",
                fn=self.section_vqe,
                hypothesis=(
                    f"MPS-VQE with chi={getattr(self, '_chi_max', 64)} achieves "
                    f"ΔE/gap < {DE_GAP_THRESHOLD * 100:.0f}% at all h in valid regime"
                ),
            ),
        ]
        if self._args.verify_chi:
            sections.append(
                Section(
                    id=3,
                    name="Chi-Convergence Verification",
                    fn=self.section_chi_convergence,
                    hypothesis=(
                        f"|E(chi={getattr(self, '_chi_max', 64)}) - E(chi={getattr(self, '_chi_max', 64) * 2})| < 1e-10 "
                        f"at all h-points (chi is sufficient)"
                    ),
                )
            )
        return sections

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 1: DMRG Ground Truth
    # ═══════════════════════════════════════════════════════════════════════════

    def section_dmrg(self) -> dict:
        """Compute DMRG ground truth for each h-point."""
        N = self._args.n_qubits
        topology = self._args.topology
        model = self._args.model
        spec = self.get_model_spec(model)

        self._dmrg_data = []

        for h in self._h_values:
            t0 = time.perf_counter()
            lattice = self.make_lattice(topology, N, J=1.0, h=h)
            H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
            gt = self.solver.solve(H, lattice, method="dmrg")
            elapsed = time.perf_counter() - t0

            self._dmrg_data.append(
                {
                    "h": h,
                    "ground_energy": gt.ground_energy,
                    "gap": gt.gap,
                    "mag_x": gt.mag_x,
                    "corr_zz": gt.corr_zz,
                    "time_s": elapsed,
                }
            )
            logger.info(
                f"    DMRG h={h:.3f}: E₀={gt.ground_energy:.8f}, "
                f"gap={gt.gap:.4f}, time={elapsed:.1f}s"
            )

        return {
            "pass": True,
            "n_points": len(self._dmrg_data),
            "e_range": [self._dmrg_data[-1]["ground_energy"], self._dmrg_data[0]["ground_energy"]],
            "gap_range": [
                min(d["gap"] for d in self._dmrg_data),
                max(d["gap"] for d in self._dmrg_data),
            ],
            "per_point": self._dmrg_data,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 2: MPS-VQE Descending Sweep
    # ═══════════════════════════════════════════════════════════════════════════

    def section_vqe(self) -> dict:
        """Run MPS-VQE descending sweep, compare to DMRG."""
        from qmbp_simulation.optimizers.sweep_strategies import (
            AdaptiveRestartConfig,
            compute_adaptive_restarts,
        )

        N = self._args.n_qubits
        p = self._args.p_layers
        topology = self._args.topology
        model = self._args.model
        spec = self.get_model_spec(model)
        seeds = self._args.seeds

        # Build circuit once
        lattice_ref = self.make_lattice(topology, N, J=1.0, h=self._h_values[0])
        circuit, _ = spec.create_circuit(N, p, lattice_ref, **spec.circuit_kwargs)
        n_params = circuit.num_parameters
        logger.info(f"    Circuit: {n_params} params, depth={circuit.depth()}")

        # Adaptive restart config: allocates more restarts near the boundary
        adaptive_cfg = AdaptiveRestartConfig(
            base_restarts=max(1, self._args.n_restarts // 3),
            max_restarts=self._args.n_restarts,
            critical_restarts=max(3, self._args.n_restarts - 1),
            h_critical=2.6 if p == 1 else 1.6,  # expressibility boundary
            critical_radius=0.5,
            de_gap_threshold=0.02,
        )

        # Chi warning for p>2 (not validated exact)
        if p > 2:
            logger.warning(
                f"    ⚠️  p={p} > 2: χ=64 exactness NOT validated for p>{2}. "
                f"Results may have undetected MPS truncation error. "
                f"Consider running with --verify-chi."
            )

        all_seed_results = []

        for seed in seeds:
            logger.info(f"    ── VQE sweep seed={seed} ──")
            rng = np.random.default_rng(seed)
            prev_theta = rng.uniform(-0.01, 0.01, n_params)
            prev_de_gap = None
            seed_results = []

            for idx, h in enumerate(self._h_values):
                t0 = time.perf_counter()
                lattice_h = self.make_lattice(topology, N, J=1.0, h=h)
                H = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)

                e_exact = self._dmrg_data[idx]["ground_energy"]
                gap = self._dmrg_data[idx]["gap"]

                # Adaptive restarts: more near boundary, fewer in trivial regime
                n_restarts_h = compute_adaptive_restarts(
                    h, prev_de_gap=prev_de_gap, config=adaptive_cfg
                )
                vqe_config_h = self.VQEConfig(
                    p_layers=p,
                    n_restarts=n_restarts_h,
                    maxiter=self._args.maxiter,
                    method="L-BFGS-B",
                    enable_callbacks=False,
                )
                optimizer = self.VQEOptimizer(config=vqe_config_h, backend=self._backend, seed=seed)

                vqe_result = optimizer.optimize(
                    hamiltonian=H,
                    circuit=circuit,
                    initial_guess=prev_theta,
                    exact_energy=e_exact,
                )
                elapsed = time.perf_counter() - t0

                de_gap = abs(vqe_result.energy - e_exact) / max(gap, 1e-10)
                delta_e_abs = abs(vqe_result.energy - e_exact)
                theta_change = float(np.max(np.abs(vqe_result.theta_opt - prev_theta)))
                prev_theta = vqe_result.theta_opt.copy()
                prev_de_gap = de_gap

                seed_results.append(
                    {
                        "h": h,
                        "vqe_energy": vqe_result.energy,
                        "dmrg_energy": e_exact,
                        "gap": gap,
                        "de_gap": de_gap,
                        "delta_e_abs": delta_e_abs,
                        "delta_e_per_site": delta_e_abs / N,
                        "energy_variance": vqe_result.energy_variance,
                        "n_iterations": vqe_result.n_iterations,
                        "n_restarts_used": n_restarts_h,
                        "theta_opt": vqe_result.theta_opt.tolist(),
                        "theta_change_linf": theta_change,
                        "elapsed_s": elapsed,
                        "passed": de_gap < DE_GAP_THRESHOLD,
                    }
                )

                status = "✓" if de_gap < DE_GAP_THRESHOLD else "✗"
                logger.info(
                    f"      [{status}] h={h:.3f}: E={vqe_result.energy:.8f} "
                    f"ΔE/gap={de_gap:.4f} |ΔE|={delta_e_abs:.4f} "
                    f"(r={n_restarts_h}, {elapsed:.1f}s)"
                )

                # Checkpoint after each h-point
                self.save_checkpoint(
                    f"vqe_s{seed}",
                    {
                        "seed": seed,
                        "n_done": idx + 1,
                        "results": seed_results,
                        "prev_theta": prev_theta.tolist(),
                    },
                )

            all_seed_results.append({"seed": seed, "results": seed_results})
            self.cleanup_checkpoints(f"vqe_s{seed}")

        # Aggregate
        all_de_gaps = [r["de_gap"] for sr in all_seed_results for r in sr["results"]]
        n_total = len(all_de_gaps)
        n_pass = sum(1 for d in all_de_gaps if d < DE_GAP_THRESHOLD)
        all_passed = n_pass == n_total

        return {
            "pass": all_passed,
            "n_pass": n_pass,
            "n_total": n_total,
            "pass_rate": n_pass / max(n_total, 1),
            "mean_de_gap": float(np.mean(all_de_gaps)),
            "max_de_gap": float(np.max(all_de_gaps)),
            "per_seed": all_seed_results,
            "circuit_info": {
                "n_params": n_params,
                "depth": circuit.depth(),
                "gate_counts": dict(circuit.count_ops()),
            },
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 3: Chi-Convergence Verification (optional, --verify-chi)
    # ═══════════════════════════════════════════════════════════════════════════

    def section_chi_convergence(self) -> dict:
        """Verify chi sufficiency by comparing E(chi) vs E(2*chi).

        For each h-point, re-evaluates the best theta_opt found in Section 2
        using a backend with doubled chi. If |E(chi) - E(2*chi)| < 1e-10 at
        all points, chi is sufficient (MPS truncation error is negligible).
        """
        N = self._args.n_qubits
        p = self._args.p_layers
        topology = self._args.topology
        model = self._args.model
        spec = self.get_model_spec(model)
        chi_1x = self._chi_max
        chi_2x = chi_1x * 2

        logger.info(f"    Chi-convergence: comparing chi={chi_1x} vs chi={chi_2x}")

        # Build 2x backend
        backend_2x = self.MPSBackend(
            strategy="aer_mps",
            chi_max=chi_2x,
            seed=self._args.seeds[0],
        )

        # Build circuit once
        lattice_ref = self.make_lattice(topology, N, J=1.0, h=self._h_values[0])
        circuit, _ = spec.create_circuit(N, p, lattice_ref, **spec.circuit_kwargs)

        # Get theta_opt from Section 2 results (first seed)
        sec2_result = None
        for sr in self._section_results:
            if sr.section_id == 2 and sr.data:
                sec2_result = sr.data
                break

        if sec2_result is None or "per_seed" not in sec2_result:
            logger.error("    Section 2 results not available — cannot verify chi.")
            return {"pass": False, "error": "Section 2 results unavailable"}

        first_seed_results = sec2_result["per_seed"][0]["results"]
        convergence_results = []
        all_converged = True
        CHI_CONVERGENCE_TOL = 1e-10

        for idx, h in enumerate(self._h_values):
            theta_opt = np.array(first_seed_results[idx]["theta_opt"])
            lattice_h = self.make_lattice(topology, N, J=1.0, h=h)
            H = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)

            # Evaluate with 1x chi (cached from Section 2)
            e_1x = first_seed_results[idx]["vqe_energy"]

            # Evaluate with 2x chi
            t0 = time.perf_counter()
            e_2x = backend_2x.evaluate(circuit, H, theta_opt)
            elapsed = time.perf_counter() - t0

            chi_error = abs(e_1x - e_2x)
            converged = chi_error < CHI_CONVERGENCE_TOL

            if not converged:
                all_converged = False
                logger.warning(
                    f"      ⚠️  h={h:.3f}: |E(χ={chi_1x}) - E(χ={chi_2x})| = {chi_error:.2e} "
                    f"> {CHI_CONVERGENCE_TOL:.0e}"
                )
            else:
                logger.info(f"      ✓ h={h:.3f}: |ΔE| = {chi_error:.2e} ({elapsed:.1f}s)")

            convergence_results.append(
                {
                    "h": h,
                    "energy_chi_1x": e_1x,
                    "energy_chi_2x": e_2x,
                    "chi_error": chi_error,
                    "converged": converged,
                    "elapsed_s": elapsed,
                }
            )

        max_chi_error = max(r["chi_error"] for r in convergence_results)
        n_converged = sum(1 for r in convergence_results if r["converged"])

        return {
            "pass": all_converged,
            "chi_1x": chi_1x,
            "chi_2x": chi_2x,
            "tolerance": CHI_CONVERGENCE_TOL,
            "max_chi_error": max_chi_error,
            "n_converged": n_converged,
            "n_total": len(convergence_results),
            "per_point": convergence_results,
        }


if __name__ == "__main__":
    MPSScalingValidationRunner.main()
