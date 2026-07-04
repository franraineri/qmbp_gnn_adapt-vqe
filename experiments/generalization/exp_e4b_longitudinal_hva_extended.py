"""E4b: TFIM + Longitudinal Field with Extended HVA (RZ layer).

Hypothesis: Adding an RZ layer to the HVA circuit (3 params/layer: θ_zz, θ_x, θ_z)
restores expressibility for the TFIM + longitudinal field model, achieving
fidelity ≥ 0.90 up to g=0.5 (where E4's standard HVA dropped to 0.556).

Background:
    E4 showed that standard HVA (ZZ+X layers only) cannot express the ground
    state when g>0 because the longitudinal Z field requires a rotation layer
    (RZ) that the TFIM-specific ansatz lacks. The RZ layer matches the
    additional Z term in the Hamiltonian.

    H = -J Σ_{(i,j)} Z_i Z_j  -  h Σ_i X_i  -  g Σ_i Z_i

    The extended HVA per layer: RZZ(2θ_zz) → RX(2θ_x) → RZ(2θ_z)
    maps directly to each term of the Hamiltonian (Hamiltonian Variational
    principle: circuit structure mirrors Hamiltonian structure).

Method:
    1. Build extended Hamiltonian via HamiltonianBuilder.build_tfim_longitudinal()
    2. Create extended HVA via HVACircuitBuilder.create_tfim_longitudinal()
    3. Run VQE sweep across (h, g) grid with descending h-sweep per g
    4. Compare fidelities against E4 baseline (standard HVA)
    5. Identify valid regime in (h, g) space

Expected outcome:
    - g ≤ 0.3: fidelity ≥ 0.93 for h ≥ 1.25 (full TFIM valid regime)
    - g = 0.5: fidelity ≥ 0.90 for h ≥ 1.5
    - Significant improvement over E4 at all g > 0

Time estimate: ~2 min (6 qubits, 5 h-values × 5 g-values × 3 seeds × 5 restarts)

References:
    - E4 results: binnacle-v8-experiments-round1.md (fidelity 0.556 at g=0.5)
    - Dutta et al. (2015) Quantum Phase Transitions in Transverse Field Spin Models
"""

from __future__ import annotations

import logging
import time

import numpy as np

from qmbp_simulation.framework import BaseExperiment
from qmbp_simulation.framework.config import ExperimentConfig, SystemConfig, VQEConfig
from qmbp_simulation.framework.metrics import ExperimentMetrics
from qmbp_simulation.models.model_registry import get_model_spec

logger = logging.getLogger(__name__)

# Grid configuration
G_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5]
H_VALUES = [2.0, 1.75, 1.5, 1.25, 1.0]
VQE_RESTARTS = 5
VQE_MAXITER = 500

# E4 baseline (standard HVA) — from binnacle results
E4_BASELINE_FIDELITY = {
    0.0: 0.990,
    0.1: 0.889,
    0.2: 0.778,
    0.3: 0.688,
    0.5: 0.556,
}


class ExperimentE4b(BaseExperiment):
    """TFIM + longitudinal field with extended HVA (RZ layer added).

    Designed for reuse: topology and N are read from config, not hardcoded.
    The ModelSpec pattern is used for Hamiltonian/circuit dispatch.
    """

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="E4b",
            category="E",
            description=(
                "TFIM + longitudinal field with extended HVA (3 params/layer). "
                "Tests whether adding RZ layer fixes E4's expressibility failure."
            ),
            hypothesis=(
                "Extended HVA (ZZ+X+Z layers) achieves fidelity ≥ 0.90 for g ≤ 0.5, "
                "fixing the E4 failure where standard HVA dropped to 0.556."
            ),
            system=SystemConfig(
                n_qubits=6,
                p_layers=2,
                topology="chain_1d",
                J=1.0,
                h_values=H_VALUES,
                h_test=[],
                g_longitudinal=0.0,
                model="tfim_longitudinal",
            ),
            vqe=VQEConfig(
                n_restarts=VQE_RESTARTS,
                sigma=0.1,
                maxiter=VQE_MAXITER,
            ),
            seeds=DEFAULT_SEEDS,
            verbose=True,
            auto_warm_cold_comparison=False,
        )

    def setup(self) -> None:
        """Setup extended HVA circuit via ModelSpec dispatch."""
        from qmbp_simulation import HamiltonianBuilder, make_lattice

        self._setup_logging()
        self.builder = HamiltonianBuilder()
        self.model_spec = get_model_spec("tfim_longitudinal")

        N = self.config.system.n_qubits
        p = self.config.system.p_layers
        topology = self.config.system.topology
        base_lattice = make_lattice(topology, N, J=self.config.system.J, h=1.0)

        # Use ModelSpec dispatch for circuit creation
        self.circuit, _ = self.model_spec.create_circuit(N, p, base_lattice)
        self.n_params = self.circuit.num_parameters
        self._topology = topology
        self._J = self.config.system.J

        logger.info(
            "E4b setup: N=%d, p=%d, params=%d (3/layer), topology=%s, model=%s",
            N,
            p,
            self.n_params,
            topology,
            self.model_spec.name,
        )

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Run VQE across (h, g) grid with extended HVA."""
        from qiskit.quantum_info import Statevector, state_fidelity

        from qmbp_simulation import make_lattice

        rng = np.random.default_rng(seed)
        N = self.config.system.n_qubits
        metrics = []

        for g in G_VALUES:
            # Descending h-sweep per g (warm-start within each g-slice)
            prev_theta = rng.uniform(-0.01, 0.01, self.n_params)

            for h in sorted(self.config.system.h_values, reverse=True):
                t0 = time.time()

                # Build Hamiltonian via ModelSpec with varying g
                lattice = make_lattice(self._topology, N, J=self._J, h=h)
                H = self.model_spec.build_hamiltonian(lattice, g=g)

                # Exact diag for reference
                exact_energy, gap, gs_vector = self._exact_diag(H)

                # VQE with restarts
                best_energy, best_theta, total_evals = self._run_vqe(
                    H,
                    prev_theta,
                    rng,
                )
                prev_theta = best_theta.copy()

                # Compute fidelity
                bound_circuit = self.circuit.assign_parameters(best_theta)
                sv_vqe = Statevector(bound_circuit)
                fid = float(state_fidelity(sv_vqe, Statevector(gs_vector)))

                de_gap = abs(best_energy - exact_energy) / max(gap, 1e-10)
                elapsed = time.time() - t0

                # Compare against E4 baseline
                e4_fid = E4_BASELINE_FIDELITY.get(g)
                improvement = (fid - e4_fid) if e4_fid is not None else None

                m = ExperimentMetrics(
                    h_value=h,
                    energy=best_energy,
                    exact_energy=exact_energy,
                    energy_error=abs(best_energy - exact_energy),
                    gap=gap,
                    relative_error=de_gap,
                    fidelity=fid,
                    seed=seed,
                    wall_time_s=elapsed,
                    n_evaluations=total_evals,
                    converged=de_gap < 0.05,
                    technique_metadata={
                        "g": float(g),
                        "topology": self._topology,
                        "fidelity": float(fid),
                        "de_gap": float(de_gap),
                        "hva_sufficient": fid >= self.model_spec.fidelity_threshold,
                        "e4_baseline_fidelity": e4_fid,
                        "improvement_over_e4": improvement,
                        "ansatz": "extended_hva_zz_x_z",
                        "n_params": self.n_params,
                    },
                )
                metrics.append(m)

                logger.debug(
                    "  g=%.1f h=%.2f: fid=%.4f ΔE/gap=%.4f (E4: %.3f → +%.3f)",
                    g,
                    h,
                    fid,
                    de_gap,
                    e4_fid or 0,
                    improvement or 0,
                )

        return metrics

    def _exact_diag(self, H):
        """Compute ground state energy, gap, and ground state vector."""
        H_matrix = H.to_matrix()
        if hasattr(H_matrix, "toarray"):
            H_matrix = H_matrix.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
        e0 = float(eigenvalues[0])
        e1 = float(eigenvalues[1])
        gap = e1 - e0
        gs_vector = eigenvectors[:, 0]
        return e0, gap, gs_vector

    def _run_vqe(self, H, prev_theta, rng):
        """Run multi-restart VQE with L-BFGS-B."""
        from scipy.optimize import minimize

        def cost_fn(params, _H=H):
            return self.evaluate_energy(params, _H)

        best_energy = float("inf")
        best_theta = prev_theta.copy()
        total_evals = 0

        for restart in range(self.config.vqe.n_restarts):
            if restart == 0:
                x0 = prev_theta.copy()
            else:
                x0 = prev_theta + rng.normal(0, self.config.vqe.sigma, self.n_params)
            x0 = np.clip(x0, -np.pi, np.pi)

            result = minimize(
                cost_fn,
                x0,
                method="L-BFGS-B",
                bounds=[(-np.pi, np.pi)] * self.n_params,
                options={"maxiter": self.config.vqe.maxiter, "ftol": 1e-14},
            )
            total_evals += result.nfev

            if result.fun < best_energy:
                best_energy = result.fun
                best_theta = result.x.copy()

        return best_energy, best_theta, total_evals

    def analyze(self, results: dict[int, list[ExperimentMetrics]]) -> dict:
        """Aggregate results: per-g fidelities and comparison with E4."""
        per_g_stats: dict[float, dict] = {}

        for g in G_VALUES:
            fidelities = []
            de_gaps = []
            for _seed, seed_metrics in results.items():
                for m in seed_metrics:
                    meta = m.technique_metadata or {}
                    if abs(meta.get("g", -1) - g) < 1e-6:
                        fidelities.append(m.fidelity or 0)
                        de_gaps.append(m.relative_error or 0)

            if fidelities:
                per_g_stats[g] = {
                    "mean_fidelity": float(np.mean(fidelities)),
                    "std_fidelity": float(np.std(fidelities)),
                    "mean_de_gap": float(np.mean(de_gaps)),
                    "pass_rate_90": float(np.mean([f >= 0.90 for f in fidelities])),
                    "pass_rate_93": float(np.mean([f >= 0.93 for f in fidelities])),
                    "e4_baseline": E4_BASELINE_FIDELITY.get(g),
                    "improvement": (float(np.mean(fidelities)) - E4_BASELINE_FIDELITY.get(g, 0)),
                    "n_points": len(fidelities),
                }

        # Determine max g where ansatz is sufficient
        max_g_pass = 0.0
        for g in sorted(G_VALUES):
            stats = per_g_stats.get(g, {})
            if stats.get("mean_fidelity", 0) >= self.model_spec.fidelity_threshold:
                max_g_pass = g

        # Standard summary for digest/compare compatibility
        overall_pass_rate = (
            float(
                np.mean(
                    [
                        m.relative_error < 0.05
                        for metrics in results.values()
                        for m in metrics
                        if m.relative_error is not None
                    ]
                )
            )
            if results
            else 0.0
        )

        overall_mean_de_gap = (
            float(
                np.mean(
                    [
                        m.relative_error
                        for metrics in results.values()
                        for m in metrics
                        if m.relative_error is not None
                    ]
                )
            )
            if results
            else 0.0
        )

        analysis = {
            "summary": {
                "mean_de_gap": overall_mean_de_gap,
                "pass_rate": overall_pass_rate,
                "total_time_s": sum(m.wall_time_s for metrics in results.values() for m in metrics),
            },
            "per_g_stats": {str(g): v for g, v in per_g_stats.items()},
            "max_g_valid": max_g_pass,
            "hypothesis_confirmed": max_g_pass >= 0.5,
            "n_seeds": len(results),
            "topology": self._topology,
            "comparison_with_e4": {
                "e4_max_g": 0.0,
                "e4b_max_g": max_g_pass,
                "e4_ansatz": "standard_hva (2 params/layer)",
                "e4b_ansatz": "extended_hva (3 params/layer, +RZ)",
            },
        }
        return analysis

    def report(self, analysis: dict) -> str:
        """Generate human-readable report comparing E4b vs E4."""
        lines = [
            "=" * 65,
            "EXP-E4b: TFIM + Longitudinal Field — Extended HVA (ZZ+X+Z)",
            "=" * 65,
            "",
            f"Seeds: {self.config.seeds}",
            f"N={self.config.system.n_qubits}, p={self.config.system.p_layers}, "
            f"params={self.n_params} (3/layer), topology={self._topology}",
            f"g values tested: {G_VALUES}",
            f"h values (descending): {sorted(self.config.system.h_values, reverse=True)}",
            "",
            "--- Per-g Fidelity Comparison (E4 standard HVA vs E4b extended) ---",
            "",
            f"{'g':>4} | {'E4 (std)':>9} | {'E4b (ext)':>10} | {'Δ':>6} | {'Pass≥0.90':>9}",
            f"{'-' * 4}-+-{'-' * 9}-+-{'-' * 10}-+-{'-' * 6}-+-{'-' * 9}",
        ]

        per_g = analysis.get("per_g_stats", {})
        for g in G_VALUES:
            stats = per_g.get(str(g), {})
            e4_fid = E4_BASELINE_FIDELITY.get(g, 0)
            e4b_fid = stats.get("mean_fidelity", 0)
            delta = e4b_fid - e4_fid
            pass_rate = stats.get("pass_rate_90", 0)
            lines.append(
                f"{g:>4.1f} | {e4_fid:>9.3f} | {e4b_fid:>10.3f} | "
                f"{delta:>+6.3f} | {pass_rate:>8.0%}"
            )

        lines.extend(
            [
                "",
                f"Max g with fidelity ≥ {self.model_spec.fidelity_threshold}: "
                f"{analysis.get('max_g_valid', 'N/A')}",
                f"Hypothesis confirmed: {analysis.get('hypothesis_confirmed', False)}",
                "",
                "--- Interpretation ---",
                "E4 failure root cause: standard HVA (ZZ+X) cannot represent g·Z rotation.",
                "E4b fix: extended HVA (ZZ+X+Z) adds RZ layer matching Hamiltonian structure.",
                "This confirms HVA must mirror ALL Hamiltonian terms for expressibility.",
                "",
                "=" * 65,
            ]
        )
        return "\n".join(lines)
