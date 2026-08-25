#!/usr/bin/env python3
"""MPS Precision Study — χ=64 precision loss on 2D topologies at N=20-30.

Demonstrates that χ=64 (our default MPS bond dimension) loses precision
on heavy_hex and triangular topologies at N≥20, while remaining exact for
chain_1d. Establishes the quantum advantage boundary where QPU execution
becomes necessary.

Methodology:
    For each (N, topology) pair:
    1. Compute exact ground truth (exact diag for N≤22, high-χ MPS for N>22).
    2. Run VQE with MPSBackend at multiple χ values [16, 32, 64, 128, 256].
    3. For N≤22: compare with exact diag (absolute reference).
    4. For N>22: compare with χ=256 (convergence reference).
    5. Report ΔE(χ) curves showing where χ=64 breaks down.

Sections:
    1. Ground Truth Computation (exact diag N≤22, DMRG+high-χ for N>22)
    2. Chi-Sweep VQE: Run VQE at each χ, record energy convergence
    3. Precision Analysis: Quantify truncation error vs χ
    4. Topology Comparison: chain_1d vs heavy_hex vs triangular

Key metrics:
    - |E(χ=64) - E_exact| / gap — precision loss as fraction of spectral gap
    - |E(χ=64) - E(χ=256)| — absolute truncation error
    - Chi at which |E(χ) - E(χ_max)| < 1e-6 (convergence threshold)
    - Entanglement entropy proxy (from MPS bond truncation)

Usage:
    # Default: N=20 on heavy_hex + triangular + chain_1d
    .venv/bin/python scripts/experiment_runners/scaling/run_mps_precision_study.py

    # Specific topology and N range
    .venv/bin/python scripts/experiment_runners/scaling/run_mps_precision_study.py \
        --n-values 20 22 24 26 30 --topologies heavy_hex triangular

    # Quick check (N=20 only, fewer chi values)
    .venv/bin/python scripts/experiment_runners/scaling/run_mps_precision_study.py \
        --n-values 20 --chi-values 32 64 128

    # Dry run
    .venv/bin/python scripts/experiment_runners/scaling/run_mps_precision_study.py --dry-run
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
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_N_VALUES = [20, 22, 26, 30]
DEFAULT_TOPOLOGIES = ["heavy_hex", "triangular", "chain_1d"]
DEFAULT_CHI_VALUES = [16, 32, 64, 128, 256]
DEFAULT_MODEL = "tfim"
DEFAULT_P = 1
DEFAULT_H_VALUES = [4.0, 3.0, 2.5, 2.0]  # Span valid→boundary→challenging
DEFAULT_MAXITER = 500
DEFAULT_N_RESTARTS = 3

# Maximum N for exact diagonalization (statevector)
EXACT_DIAG_MAX_N = 22

# Convergence threshold: |E(χ) - E_ref| below this means "converged"
# For VQE-based measurements, 1e-4 accounts for optimization noise.
# For pure evaluation (re-evaluate theta at different chi), use 1e-10.
CHI_CONVERGENCE_THRESHOLD = 1e-4


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════


class MPSPrecisionStudyRunner(ValidationRunner):
    """Study MPS precision loss as a function of χ and topology.

    Establishes that χ=64 is insufficient for 2D topologies at N≥20,
    providing the justification for QPU execution.
    """

    runner_id = "mps_precision_study_v1"
    experiment_id = "scaling/mps_precision"
    description = "MPS χ-Precision Study — 2D topology breakdown at N=20-30"
    hypothesis = (
        "χ=64 produces negligible truncation error on chain_1d but significant "
        "error (ΔE/gap > 1%) on heavy_hex and triangular at N≥20, establishing "
        "the quantum advantage boundary."
    )

    @classmethod
    def _add_custom_args(cls, parser):
        parser.add_argument(
            "--n-values", type=int, nargs="+", default=DEFAULT_N_VALUES,
            help="System sizes to test",
        )
        parser.add_argument(
            "--topologies", type=str, nargs="+", default=DEFAULT_TOPOLOGIES,
            help="Topologies to compare",
        )
        parser.add_argument(
            "--chi-values", type=int, nargs="+", default=DEFAULT_CHI_VALUES,
            help="Bond dimensions to sweep",
        )
        parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
        parser.add_argument("--p-layers", type=int, default=DEFAULT_P, choices=[1, 2])
        parser.add_argument(
            "--h-values", type=float, nargs="+", default=DEFAULT_H_VALUES,
            help="h-values to evaluate",
        )
        parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
        parser.add_argument("--n-restarts", type=int, default=DEFAULT_N_RESTARTS)
        parser.add_argument("--seed", type=int, default=42)

    def run_preflight(self) -> bool:
        """Validate configuration."""
        max_n = max(self._args.n_values)
        if max_n > 30 and max(self._args.chi_values) < 128:
            logger.warning(
                f"N={max_n} with max chi={max(self._args.chi_values)} may not "
                f"converge. Consider adding chi=256 or higher."
            )
        # Ensure chi values are sorted
        self._args.chi_values = sorted(self._args.chi_values)
        return True

    def build_config(self) -> dict:
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "system": {
                "n_values": self._args.n_values,
                "topologies": self._args.topologies,
                "model": self._args.model,
                "p_layers": self._args.p_layers,
            },
            "mps": {
                "chi_values": self._args.chi_values,
                "convergence_threshold": CHI_CONVERGENCE_THRESHOLD,
            },
            "vqe": {
                "h_values": self._args.h_values,
                "maxiter": self._args.maxiter,
                "n_restarts": self._args.n_restarts,
            },
            "seeds": [self._args.seed],
        }


    def setup(self):
        """Initialize shared physics objects."""
        self.setup_physics()

        logger.info(f"  Model: {self._args.model}")
        logger.info(f"  N values: {self._args.n_values}")
        logger.info(f"  Topologies: {self._args.topologies}")
        logger.info(f"  Chi values: {self._args.chi_values}")
        logger.info(f"  h values: {self._args.h_values}")

        self._spec = self.get_model_spec(self._args.model)

        # Store results
        self._ground_truth: dict[tuple, dict] = {}  # (N, topology, h) → {energy, gap, ...}
        self._chi_sweep_results: list[dict] = []

    def define_sections(self) -> list[Section]:
        return [
            Section(
                id=1,
                name="Ground Truth Computation",
                fn=self.section_ground_truth,
                hypothesis=(
                    "Exact diag (N≤22) or DMRG (N>22) provides reliable reference "
                    "for all (N, topology, h) configurations"
                ),
            ),
            Section(
                id=2,
                name="Chi-Sweep VQE",
                fn=self.section_chi_sweep,
                hypothesis=(
                    "VQE energy converges monotonically as χ increases, with "
                    "chain_1d converging at χ=64 and 2D topologies requiring χ>64"
                ),
            ),
            Section(
                id=3,
                name="Precision Analysis & Topology Comparison",
                fn=self.section_precision_analysis,
                hypothesis=(
                    "χ=64 truncation error is <1e-10 for chain_1d but >1% of gap "
                    "for heavy_hex/triangular at N≥20"
                ),
            ),
        ]


    # ═══════════════════════════════════════════════════════════════════════════
    # Section 1: Ground Truth Computation
    # ═══════════════════════════════════════════════════════════════════════════

    def section_ground_truth(self) -> dict:
        """Compute ground truth for all (N, topology, h) configurations."""
        results = []

        for topology in self._args.topologies:
            for N in self._args.n_values:
                for h in self._args.h_values:
                    t0 = time.perf_counter()

                    lattice = self.make_lattice(topology, N, J=1.0, h=h)
                    H = self._spec.build_hamiltonian(lattice, **self._spec.hamiltonian_kwargs)

                    # Choose solver method based on N
                    if N <= EXACT_DIAG_MAX_N:
                        method = "exact"
                        gt = self.solver.solve(H, lattice, method="exact")
                    else:
                        method = "dmrg"
                        gt = self.solver.solve(H, lattice, method="dmrg")

                    elapsed = time.perf_counter() - t0

                    key = (N, topology, h)
                    self._ground_truth[key] = {
                        "energy": gt.ground_energy,
                        "gap": gt.gap,
                        "method": method,
                    }

                    record = {
                        "n_qubits": N,
                        "topology": topology,
                        "h": h,
                        "ground_energy": gt.ground_energy,
                        "gap": gt.gap,
                        "method": method,
                        "time_s": round(elapsed, 2),
                    }
                    results.append(record)

                    logger.info(
                        f"    [{method}] {topology} N={N} h={h:.2f}: "
                        f"E₀={gt.ground_energy:.8f}, gap={gt.gap:.4f} ({elapsed:.1f}s)"
                    )

        return {
            "pass": True,
            "n_configs": len(results),
            "per_config": results,
        }


    # ═══════════════════════════════════════════════════════════════════════════
    # Section 2: Chi-Sweep VQE
    # ═══════════════════════════════════════════════════════════════════════════

    def section_chi_sweep(self) -> dict:
        """Run VQE at highest χ, then re-evaluate θ_opt at all lower χ values.

        Strategy: isolate MPS truncation error from VQE optimization noise by:
        1. Run VQE once at chi_max to find the best θ_opt.
        2. Evaluate E(θ_opt) at each lower chi — any deviation is pure MPS truncation.
        """
        from experiments.helpers.scaling_utils import evaluate_at_multiple_chi

        p = self._args.p_layers
        seed = self._args.seed
        chi_values = self._args.chi_values
        chi_max = max(chi_values)
        maxiter = self._args.maxiter
        n_restarts = self._args.n_restarts

        self._chi_sweep_results = []

        for topology in self._args.topologies:
            for N in self._args.n_values:
                logger.info(f"\n    ── {topology} N={N} ──")

                # Build circuit once
                lattice_ref = self.make_lattice(topology, N, J=1.0, h=self._args.h_values[0])
                circuit, _ = self._spec.create_circuit(
                    N, p, lattice_ref, **self._spec.circuit_kwargs
                )
                n_params = circuit.num_parameters

                for h in self._args.h_values:
                    lattice = self.make_lattice(topology, N, J=1.0, h=h)
                    H = self._spec.build_hamiltonian(lattice, **self._spec.hamiltonian_kwargs)

                    gt_key = (N, topology, h)
                    gt_data = self._ground_truth.get(gt_key)
                    e_exact = gt_data["energy"] if gt_data else None
                    gap = gt_data["gap"] if gt_data else 1.0

                    # Step 1: VQE at chi_max to find best theta_opt
                    backend_max = self.MPSBackend(
                        strategy="aer_mps", chi_max=chi_max, seed=seed,
                    )
                    rng = np.random.default_rng(seed)
                    init_guess = rng.uniform(-0.01, 0.01, n_params)

                    vqe_config = self.VQEConfig(
                        p_layers=p,
                        n_restarts=n_restarts,
                        maxiter=maxiter,
                        method="L-BFGS-B",
                        enable_callbacks=False,
                    )
                    optimizer = self.VQEOptimizer(
                        config=vqe_config, backend=backend_max, seed=seed
                    )

                    t0 = time.perf_counter()
                    vqe_result = optimizer.optimize(
                        hamiltonian=H,
                        circuit=circuit,
                        initial_guess=init_guess,
                        exact_energy=e_exact,
                    )
                    vqe_time = time.perf_counter() - t0

                    theta_opt = vqe_result.theta_opt
                    e_at_chi_max = vqe_result.energy

                    logger.info(
                        f"      VQE@χ={chi_max}: E={e_at_chi_max:.8f} ({vqe_time:.1f}s)"
                    )

                    # Step 2: Evaluate theta_opt at each chi (shared utility)
                    chi_energies = evaluate_at_multiple_chi(
                        circuit, H, theta_opt, chi_values, seed=seed
                    )

                    # Log results
                    for chi in chi_values:
                        e_chi = chi_energies[chi]["energy"]
                        trunc_err = abs(e_chi - e_at_chi_max)
                        de_gap_str = ""
                        if e_exact is not None:
                            de = abs(e_chi - e_exact)
                            de_gap = de / max(gap, 1e-10)
                            de_gap_str = f"ΔE/gap={de_gap:.6f}"
                        logger.info(
                            f"      χ={chi:>3d}: E={e_chi:.8f} "
                            f"|ΔE_trunc|={trunc_err:.2e} {de_gap_str} "
                            f"({chi_energies[chi]['time_s']:.2f}s)"
                        )

                    # Store complete record
                    record = {
                        "n_qubits": N,
                        "topology": topology,
                        "h": h,
                        "e_exact": e_exact,
                        "gap": gap,
                        "chi_max_used": chi_max,
                        "e_at_chi_max": e_at_chi_max,
                        "vqe_time_s": round(vqe_time, 2),
                        "theta_opt": theta_opt.tolist(),
                        "chi_results": chi_energies,
                    }
                    self._chi_sweep_results.append(record)

                    # Checkpoint per topology×N block
                    self.save_checkpoint(
                        f"chi_sweep_{topology}_{N}",
                        {"topology": topology, "n_qubits": N, "results": self._chi_sweep_results},
                    )

        return {
            "pass": True,
            "n_configs": len(self._chi_sweep_results),
            "chi_values": chi_values,
            "chi_max_for_vqe": chi_max,
            "methodology": (
                "VQE at chi_max to find theta_opt, then evaluate at each chi. "
                "Energy differences are pure MPS truncation error."
            ),
            "per_config": self._chi_sweep_results,
        }


    # ═══════════════════════════════════════════════════════════════════════════
    # Section 3: Precision Analysis & Topology Comparison
    # ═══════════════════════════════════════════════════════════════════════════

    def section_precision_analysis(self) -> dict:
        """Analyze chi-convergence and quantify precision loss by topology."""
        from experiments.helpers.scaling_utils import analyze_chi_convergence

        chi_values = self._args.chi_values

        per_topology_summary = {}

        for topology in self._args.topologies:
            topo_records = [
                r for r in self._chi_sweep_results if r["topology"] == topology
            ]
            if not topo_records:
                continue

            topo_analysis = []
            chi64_errors = []
            chi64_de_gaps = []

            for rec in topo_records:
                N = rec["n_qubits"]
                h = rec["h"]
                e_exact = rec["e_exact"]
                gap = rec["gap"]
                chi_results = rec["chi_results"]

                # Use shared utility for convergence analysis
                analysis = analyze_chi_convergence(
                    chi_results, e_exact=e_exact, gap=gap,
                    convergence_threshold=CHI_CONVERGENCE_THRESHOLD,
                )

                analysis_record = {
                    "n_qubits": N,
                    "topology": topology,
                    "h": h,
                    **analysis,
                }
                topo_analysis.append(analysis_record)

                if analysis["chi64_abs_error"] is not None:
                    chi64_errors.append(analysis["chi64_abs_error"])
                if analysis["chi64_de_gap"] is not None:
                    chi64_de_gaps.append(analysis["chi64_de_gap"])

            # Topology-level summary
            n_chi64_sufficient = sum(
                1 for r in topo_analysis
                if r["chi64_is_sufficient"] is True
            )
            n_total = len(topo_analysis)
            mean_chi64_error = (
                float(np.mean(chi64_errors)) if chi64_errors else None
            )
            max_chi64_error = (
                float(np.max(chi64_errors)) if chi64_errors else None
            )
            mean_chi64_de_gap = (
                float(np.mean(chi64_de_gaps)) if chi64_de_gaps else None
            )
            max_chi64_de_gap = (
                float(np.max(chi64_de_gaps)) if chi64_de_gaps else None
            )

            per_topology_summary[topology] = {
                "n_configs": n_total,
                "chi64_sufficient_rate": n_chi64_sufficient / max(n_total, 1),
                "mean_chi64_abs_error": mean_chi64_error,
                "max_chi64_abs_error": max_chi64_error,
                "mean_chi64_de_gap": mean_chi64_de_gap,
                "max_chi64_de_gap": max_chi64_de_gap,
                "precision_verdict": (
                    "SUFFICIENT" if n_chi64_sufficient == n_total
                    else "INSUFFICIENT" if n_chi64_sufficient == 0
                    else "PARTIAL"
                ),
                "per_config": topo_analysis,
            }

            # Print topology summary
            verdict = per_topology_summary[topology]["precision_verdict"]
            logger.info(
                f"\n    {topology}: χ=64 {verdict} "
                f"({n_chi64_sufficient}/{n_total} converged)"
            )
            if mean_chi64_de_gap is not None:
                logger.info(
                    f"      mean ΔE/gap(χ=64) = {mean_chi64_de_gap:.6f}, "
                    f"max = {max_chi64_de_gap:.6f}"
                )
            if mean_chi64_error is not None:
                logger.info(
                    f"      mean |ΔE|(χ=64) = {mean_chi64_error:.2e}, "
                    f"max = {max_chi64_error:.2e}"
                )

        # Global verdict: χ=64 is INSUFFICIENT if any 2D topology fails
        chain_ok = per_topology_summary.get("chain_1d", {}).get(
            "precision_verdict", ""
        ) == "SUFFICIENT"
        heavy_hex_breaks = per_topology_summary.get("heavy_hex", {}).get(
            "precision_verdict", ""
        ) != "SUFFICIENT"
        triangular_breaks = per_topology_summary.get("triangular", {}).get(
            "precision_verdict", ""
        ) != "SUFFICIENT"

        # The hypothesis is confirmed if chain_1d works AND at least one 2D breaks
        hypothesis_confirmed = chain_ok and (heavy_hex_breaks or triangular_breaks)

        # Print final summary table
        logger.info("\n    ═══ MPS PRECISION SUMMARY ═══")
        logger.info(f"    {'Topology':<15} {'χ=64 Verdict':<15} {'mean ΔE/gap':>12} "
                    f"{'max ΔE/gap':>12}")
        logger.info(f"    {'-'*15} {'-'*15} {'-'*12} {'-'*12}")
        for topo, summary in per_topology_summary.items():
            v = summary["precision_verdict"]
            mean_dg = summary.get("mean_chi64_de_gap")
            max_dg = summary.get("max_chi64_de_gap")
            mean_s = f"{mean_dg:.6f}" if mean_dg is not None else "—"
            max_s = f"{max_dg:.6f}" if max_dg is not None else "—"
            logger.info(f"    {topo:<15} {v:<15} {mean_s:>12} {max_s:>12}")

        logger.info(f"\n    Hypothesis confirmed: {'YES ✓' if hypothesis_confirmed else 'NO ✗'}")
        if hypothesis_confirmed:
            logger.info(
                "    → QPU execution at N≥20 on 2D topologies is NECESSARY "
                "because MPS(χ=64) cannot provide sufficient precision."
            )

        return {
            "pass": hypothesis_confirmed,
            "per_topology": per_topology_summary,
            "hypothesis_confirmed": hypothesis_confirmed,
            "chain_1d_sufficient": chain_ok,
            "heavy_hex_insufficient": heavy_hex_breaks,
            "triangular_insufficient": triangular_breaks,
            "thesis_claim": (
                "χ=64 MPS produces negligible error on chain_1d but significant "
                f"precision loss on 2D topologies (heavy_hex: "
                f"mean ΔE/gap={per_topology_summary.get('heavy_hex', {}).get('mean_chi64_de_gap', '?')}, "
                f"triangular: "
                f"mean ΔE/gap={per_topology_summary.get('triangular', {}).get('mean_chi64_de_gap', '?')}). "
                "QPU execution is required for N≥20 on 2D geometries."
            ),
        }


if __name__ == "__main__":
    MPSPrecisionStudyRunner.main()
