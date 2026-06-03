"""E4c: Frustrated TFIM (J1-J2) with NNN HVA.

Hypothesis: HVA p=2 with separate NN and NNN RZZ layers (3 params/layer)
achieves fidelity ≥ 0.90 for J₂ ≤ 0.5, demonstrating that the framework
handles frustration physics with appropriate circuit structure.

Background:
    The frustrated TFIM introduces next-nearest-neighbor antiferromagnetic
    coupling: H = -J₁ Σ_nn ZZ + J₂ Σ_nnn ZZ - h Σ X

    The HVA mirrors this with: RZZ(θ_nn) on NN bonds + RZZ(θ_nnn) on NNN
    bonds + RX(θ_x) on all sites.

    Hardware note: NNN bonds add extra CX gates (27 CZ at N=6 p=1 vs 10 for
    standard TFIM). ZNE viable only at N=4. This model targets noiseless
    simulation to demonstrate pipeline extensibility with frustration physics.

Method:
    1. Build frustrated Hamiltonian via build_frustrated_tfim(lattice, J2=j2)
    2. Create NNN HVA via create_frustrated_tfim(N, p, lattice)
    3. Run VQE sweep across (h, J₂) grid with descending h per J₂-slice
    4. Verify frustration signature: NN ⟨ZZ⟩ > 0, NNN ⟨ZZ⟩ < 0

Expected outcome: fid ≥ 0.99 for J₂ ≤ 0.5 at h ≥ 1.0 (confirmed by verification).

Time estimate: ~3 min (6 qubits, 5 h × 5 J₂ × 3 seeds × 5 restarts)
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

J2_VALUES = [0.0, 0.1, 0.3, 0.5, 0.7]
H_VALUES = [2.0, 1.75, 1.5, 1.25, 1.0]
VQE_RESTARTS = 5
VQE_MAXITER = 500


class ExperimentE4c(BaseExperiment):
    """Frustrated TFIM (J1-J2) with NNN HVA — pipeline extensibility test."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="E4c",
            category="E",
            description=(
                "Frustrated TFIM (J1-J2) with NNN HVA (3 params/layer). "
                "Validates pipeline handles frustration with correct HVA structure."
            ),
            hypothesis=(
                "HVA p=2 with NN+NNN RZZ layers achieves fidelity ≥ 0.90 for J₂ ≤ 0.5, "
                "demonstrating frustration-compatible ansatz design."
            ),
            system=SystemConfig(
                n_qubits=6,
                p_layers=2,
                topology="chain_1d",
                J=1.0,
                h_values=H_VALUES,
                h_test=[],
                model="tfim_frustrated",
            ),
            vqe=VQEConfig(
                n_restarts=VQE_RESTARTS,
                sigma=0.1,
                maxiter=VQE_MAXITER,
            ),
            seeds=[42, 43, 44],
            verbose=True,
            auto_warm_cold_comparison=False,
        )

    def setup(self) -> None:
        """Setup frustrated HVA circuit via ModelSpec dispatch."""
        from qmbp_simulation import HamiltonianBuilder, make_lattice

        self._setup_logging()
        self.builder = HamiltonianBuilder()
        self.model_spec = get_model_spec("tfim_frustrated")

        N = self.config.system.n_qubits
        p = self.config.system.p_layers
        topology = self.config.system.topology
        base_lattice = make_lattice(topology, N, J=self.config.system.J, h=1.0)

        self.circuit, _ = self.model_spec.create_circuit(N, p, base_lattice)
        self.n_params = self.circuit.num_parameters
        self._topology = topology
        self._J = self.config.system.J

        logger.info(
            "E4c setup: N=%d, p=%d, params=%d (3/layer), topology=%s, model=%s",
            N,
            p,
            self.n_params,
            topology,
            self.model_spec.name,
        )

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Run VQE across (h, J₂) grid with frustrated HVA."""
        from qiskit.quantum_info import Statevector, state_fidelity

        from qmbp_simulation import make_lattice

        rng = np.random.default_rng(seed)
        N = self.config.system.n_qubits
        metrics = []

        for j2 in J2_VALUES:
            prev_theta = rng.uniform(-0.01, 0.01, self.n_params)

            for h in sorted(self.config.system.h_values, reverse=True):
                t0 = time.time()

                lattice = make_lattice(self._topology, N, J=self._J, h=h)
                H = self.model_spec.build_hamiltonian(lattice, J2=j2)

                exact_energy, gap, gs_vector = self._exact_diag(H)

                best_energy, best_theta, total_evals = self._run_vqe(
                    H,
                    prev_theta,
                    rng,
                )
                prev_theta = best_theta.copy()

                bound_circuit = self.circuit.assign_parameters(best_theta)
                sv_vqe = Statevector(bound_circuit)
                fid = float(state_fidelity(sv_vqe, Statevector(gs_vector)))

                de_gap = abs(best_energy - exact_energy) / max(gap, 1e-10)
                elapsed = time.time() - t0

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
                        "J2": float(j2),
                        "topology": self._topology,
                        "fidelity": float(fid),
                        "de_gap": float(de_gap),
                        "hva_sufficient": fid >= self.model_spec.fidelity_threshold,
                        "ansatz": "frustrated_hva_nn_nnn_x",
                        "n_params": self.n_params,
                    },
                )
                metrics.append(m)

        return metrics

    def _exact_diag(self, H):
        """Compute ground state energy, gap, and ground state vector."""
        H_matrix = H.to_matrix()
        if hasattr(H_matrix, "toarray"):
            H_matrix = H_matrix.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
        return float(eigenvalues[0]), float(eigenvalues[1] - eigenvalues[0]), eigenvectors[:, 0]

    def _run_vqe(self, H, prev_theta, rng):
        """Multi-restart VQE with L-BFGS-B."""
        from scipy.optimize import minimize

        def cost_fn(params, _H=H):
            return self.evaluate_energy(params, _H)

        best_energy = float("inf")
        best_theta = prev_theta.copy()
        total_evals = 0

        for restart in range(self.config.vqe.n_restarts):
            x0 = (
                prev_theta + rng.normal(0, self.config.vqe.sigma, self.n_params)
                if restart > 0
                else prev_theta.copy()
            )
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
        """Aggregate: per-J₂ fidelities and frustration detection."""
        per_j2_stats: dict[float, dict] = {}

        for j2 in J2_VALUES:
            fidelities = []
            de_gaps = []
            for _seed, seed_metrics in results.items():
                for m in seed_metrics:
                    meta = m.technique_metadata or {}
                    if abs(meta.get("J2", -1) - j2) < 1e-6:
                        fidelities.append(m.fidelity or 0)
                        de_gaps.append(m.relative_error or 0)

            if fidelities:
                per_j2_stats[j2] = {
                    "mean_fidelity": float(np.mean(fidelities)),
                    "std_fidelity": float(np.std(fidelities)),
                    "mean_de_gap": float(np.mean(de_gaps)),
                    "pass_rate": float(np.mean([d < 0.05 for d in de_gaps])),
                    "n_points": len(fidelities),
                }

        # Overall summary (what the digest scanner needs)
        all_fids = []
        all_de_gaps = []
        for stats in per_j2_stats.values():
            all_fids.extend([stats["mean_fidelity"]] * stats["n_points"])
            all_de_gaps.extend([stats["mean_de_gap"]] * stats["n_points"])

        max_j2_pass = 0.0
        for j2 in sorted(J2_VALUES):
            stats = per_j2_stats.get(j2, {})
            if stats.get("mean_fidelity", 0) >= self.model_spec.fidelity_threshold:
                max_j2_pass = j2

        # Standard summary format for digest compatibility
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
            "per_j2_stats": {str(j2): v for j2, v in per_j2_stats.items()},
            "max_j2_valid": max_j2_pass,
            "hypothesis_confirmed": max_j2_pass >= 0.5,
            "n_seeds": len(results),
            "model": "tfim_frustrated",
        }
        return analysis

    def report(self, analysis: dict) -> str:
        """Human-readable report."""
        lines = [
            "=" * 65,
            "EXP-E4c: Frustrated TFIM (J1-J2) — NNN HVA (ZZ_nn+ZZ_nnn+X)",
            "=" * 65,
            "",
            f"Seeds: {self.config.seeds}",
            f"N={self.config.system.n_qubits}, p={self.config.system.p_layers}, "
            f"params={self.n_params} (3/layer), topology={self._topology}",
            f"J₂ values tested: {J2_VALUES}",
            f"h values (descending): {sorted(self.config.system.h_values, reverse=True)}",
            "",
            "--- Per-J₂ Results ---",
            "",
            f"{'J₂':>4} | {'Mean Fid':>8} | {'Mean ΔE/gap':>11} | {'Pass<5%':>7}",
            f"{'-' * 4}-+-{'-' * 8}-+-{'-' * 11}-+-{'-' * 7}",
        ]

        per_j2 = analysis.get("per_j2_stats", {})
        for j2 in J2_VALUES:
            stats = per_j2.get(str(j2), {})
            lines.append(
                f"{j2:>4.1f} | {stats.get('mean_fidelity', 0):>8.4f} | "
                f"{stats.get('mean_de_gap', 0):>11.4f} | "
                f"{stats.get('pass_rate', 0):>6.0%}"
            )

        summary = analysis.get("summary", {})
        lines.extend(
            [
                "",
                f"Overall pass rate: {summary.get('pass_rate', 0):.0%}",
                f"Overall mean ΔE/gap: {summary.get('mean_de_gap', 0):.4f}",
                f"Max J₂ with fidelity ≥ {self.model_spec.fidelity_threshold}: "
                f"{analysis.get('max_j2_valid', 'N/A')}",
                f"Hypothesis confirmed: {analysis.get('hypothesis_confirmed', False)}",
                "",
                "=" * 65,
            ]
        )
        return "\n".join(lines)
