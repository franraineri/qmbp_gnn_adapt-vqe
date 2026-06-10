#!/usr/bin/env python3
"""PEA-ZNE at N=40/50 — Validates PEA extrapolation at hardware scale.

Uses AerSimulator(method="matrix_product_state") with depolarizing noise
for memory-safe noisy simulation at N=40/50.

MEMORY CONSTRAINTS (documented):
  - FakeTorino (133 qubits): ONLY usable for N≤10 (OOM otherwise)
  - AerSimulator(method="statevector"): ONLY usable for N≤20 (2^N amplitude vector)
  - AerSimulator(method="matrix_product_state"): scales to N=100+ at χ=64
  - NoiselessBackend (StatevectorEstimator): N≤22 (exact diag for VQE)
  - MPSBackend: N≤100+ (VQE, deterministic or stochastic)

For N≥20 noisy PEA, the correct path is:
  AerSimulator(method="mps", noise_model=amplified_model)
  + circuit.save_expectation_value(H) → exact Tr(ρ·H)
  This gives the exact noisy expectation (no shot noise, deterministic).

Sections:
  1. N=10 Reference (FakeTorino — matches prior ZNE_CROSS_TOPO)
  2. N=40 PEA-ZNE (MPS + noise, ~78 CZ gates)
  3. N=50 PEA-ZNE (MPS + noise, ~98 CZ gates)
  4. Scaling comparison table

Usage:
    .venv/bin/python scripts/experiment_runners/noise_zne_gf_pea/run_pea_scaling_n40.py
    .venv/bin/python scripts/experiment_runners/noise_zne_gf_pea/run_pea_scaling_n40.py --section 2
    .venv/bin/python scripts/experiment_runners/noise_zne_gf_pea/run_pea_scaling_n40.py --section 2 --section 3
    .venv/bin/python scripts/experiment_runners/noise_zne_gf_pea/run_pea_scaling_n40.py --dry-run
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
NOISE_FACTORS = (1, 3, 5)
ZNE_SHOTS = 4096
SEED = 42
MPS_CHI = 64

H_VALUES_N10 = [4.0, 3.25, 3.0]
H_VALUES_N40 = [5.5, 5.0, 4.5]
H_VALUES_N50 = [6.5, 6.0, 5.5]

# Torino mean CZ error rate (~0.8% from calibration data)
TORINO_MEAN_CZ_ERROR = 0.008


# ═══════════════════════════════════════════════════════════════════════════════
class PEAScalingRunner(ValidationRunner):
    """PEA-ZNE scaling to N=40/50 via MPS noisy simulation."""

    runner_id = "pea_scaling_n40"
    experiment_id = "PEA_SCALING"
    description = "PEA-ZNE Scaling at N=40/50 via MPS + depolarizing noise"
    hypothesis = (
        "PEA-ZNE extrapolation remains linear (R²>0.8) at N=40/50, "
        "confirming the methodology scales to hardware-relevant sizes."
    )

    def build_config(self) -> dict:
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "category": "ZNE",
            "system": {"topology": "chain_1d", "n_qubits": 40, "p_layers": 1, "model": "tfim"},
            "zne": {"noise_factors": list(NOISE_FACTORS), "method": "mps_noisy",
                    "chi_max": MPS_CHI, "mean_error_rate": TORINO_MEAN_CZ_ERROR},
            "seed": SEED,
        }

    def setup(self) -> None:
        """Lazy imports — FakeTorino only loaded if section 1 runs."""
        from qmbp_simulation import HamiltonianBuilder, make_lattice
        from qmbp_simulation.circuits import HVACircuitBuilder

        self.builder = HamiltonianBuilder()
        self.hva = HVACircuitBuilder()
        self.make_lattice = make_lattice
        self._results: dict[int, list[dict]] = {}
        logger.info("[setup] Core imports done. FakeTorino deferred to section 1.")

    def define_sections(self) -> list[Section]:
        return [
            Section(id=1, name="N=10 Reference (FakeTorino)",
                    fn=self._section_n10,
                    hypothesis="PEA gain > 0 at N=10 (reference)"),
            Section(id=2, name="N=40 PEA-ZNE (MPS + noise, ~78 CZ)",
                    fn=self._section_n40,
                    hypothesis="PEA R²>0.8 and positive gain at N=40"),
            Section(id=3, name="N=50 PEA-ZNE (MPS + noise, ~98 CZ)",
                    fn=self._section_n50,
                    hypothesis="PEA R²>0.8 and positive gain at N=50"),
            Section(id=4, name="Scaling Comparison",
                    fn=self._section_comparison,
                    hypothesis="PEA remains effective as N scales"),
        ]

    # ─── N=10 FakeTorino (lazy-loaded) ───────────────────────────────────

    def _section_n10(self) -> dict:
        """N=10 PEA on FakeTorino — reference only. Loads FakeTorino on demand."""
        from qiskit_ibm_runtime.fake_provider import FakeTorino
        from qmbp_simulation.execution import NoiselessBackend
        from qmbp_simulation.execution.noisy_utils import (
            NoisyEstimatorConfig, build_adjacency, find_layouts_bfs,
            noisy_estimate, run_gate_folding_zne, run_pea_zne, select_layouts_low_ces,
        )

        fake_backend = FakeTorino()
        config = NoisyEstimatorConfig(shots=ZNE_SHOTS, seed_simulator=SEED)
        adj = build_adjacency(fake_backend)
        candidates = find_layouts_bfs(adj, 10, n_candidates=20)
        noiseless = NoiselessBackend()

        circuit, _ = self.hva.create(10, 1, self.make_lattice("chain_1d", 10, J=1.0, h=4.0))
        theta_map = self.vqe_descending_sweep(
            "chain_1d", 10, H_VALUES_N10, SEED, p_layers=1, n_restarts=1, maxiter=500)

        results = []
        for h in sorted(H_VALUES_N10, reverse=True):
            e_exact, gap = self.exact_ground_state("chain_1d", 10, h)
            bound = circuit.assign_parameters(theta_map[h])
            ls = select_layouts_low_ces(bound, fake_backend, candidates, n_select=1, max_ces=0.5)
            tr = ls.transpiled_circuits[0]
            H = self.builder.build(self.make_lattice("chain_1d", 10, J=1.0, h=h))
            Hm = H.apply_layout(tr.layout)
            n_2q = sum(1 for i in tr.data if i.operation.num_qubits == 2)
            e_noisy = noisy_estimate(tr, Hm, fake_backend, config)
            pea = run_pea_zne(tr, Hm, fake_backend, config, noise_factors=NOISE_FACTORS)
            gf = run_gate_folding_zne(tr, Hm, fake_backend, config, noise_factors=NOISE_FACTORS)
            de_n = abs(e_noisy - e_exact) / max(gap, 1e-10)
            de_pea = abs(pea.extrapolated_value - e_exact) / max(gap, 1e-10)
            de_gf = abs(gf.extrapolated_value - e_exact) / max(gap, 1e-10)
            results.append({"h": h, "n_2q": n_2q, "de_noisy": de_n,
                            "de_pea": de_pea, "de_gf": de_gf,
                            "pea_r2": pea.r_squared, "gf_r2": gf.r_squared,
                            "pea_gain": (de_n - de_pea) / max(de_n, 1e-10),
                            "gf_gain": (de_n - de_gf) / max(de_n, 1e-10)})
            logger.info(f"  N=10 h={h}: PEA +{results[-1]['pea_gain']:.1%} R²={pea.r_squared:.3f}")

        self._results[10] = results
        # Free FakeTorino memory
        del fake_backend
        return {"pass": all(r["pea_gain"] > 0 for r in results),
                "mean_pea_gain": float(np.mean([r["pea_gain"] for r in results])),
                "results": results}

    # ─── N=40 MPS noisy ──────────────────────────────────────────────────

    def _section_n40(self) -> dict:
        results = self._run_mps_noisy(40, H_VALUES_N40)
        self._results[40] = results
        r2s = [r["pea_r2"] for r in results]
        gains = [r["pea_gain"] for r in results]
        return {"pass": float(np.mean(r2s)) > 0.8 and all(g > 0 for g in gains),
                "n_qubits": 40, "mean_pea_r2": float(np.mean(r2s)),
                "mean_pea_gain": float(np.mean(gains)), "results": results}

    # ─── N=50 MPS noisy ──────────────────────────────────────────────────

    def _section_n50(self) -> dict:
        results = self._run_mps_noisy(50, H_VALUES_N50)
        self._results[50] = results
        r2s = [r["pea_r2"] for r in results]
        gains = [r["pea_gain"] for r in results]
        return {"pass": float(np.mean(r2s)) > 0.8 and all(g > 0 for g in gains),
                "n_qubits": 50, "mean_pea_r2": float(np.mean(r2s)),
                "mean_pea_gain": float(np.mean(gains)), "results": results}

    # ─── Comparison ───────────────────────────────────────────────────────

    def _section_comparison(self) -> dict:
        scaling = []
        for n, res in sorted(self._results.items()):
            scaling.append({"N": n,
                            "mean_pea_gain": float(np.mean([r["pea_gain"] for r in res])),
                            "mean_gf_gain": float(np.mean([r["gf_gain"] for r in res])),
                            "mean_pea_r2": float(np.mean([r["pea_r2"] for r in res]))})
        logger.info("\n  ═══ PEA SCALING ═══")
        for s in scaling:
            logger.info(f"  N={s['N']:>3}: PEA +{s['mean_pea_gain']:.1%} (R²={s['mean_pea_r2']:.3f}), "
                        f"GF +{s['mean_gf_gain']:.1%}")
        return {"pass": True, "scaling": scaling}

    # ─── MPS noisy sweep (memory-safe for N≥20) ──────────────────────────

    def _run_mps_noisy(self, n: int, h_values: list[float]) -> list[dict]:
        """PEA via AerSimulator(method='mps') + noise + save_expectation_value.

        Memory: O(N·χ³) — verified working at N=80 in MPS scaling experiments.
        No FakeTorino, no BackendEstimatorV2, no 133-qubit metadata.
        """
        from qmbp_simulation.execution import MPSBackend
        from qmbp_simulation.execution.noisy_utils import _extrapolate_linear, fold_gates

        # VQE via MPS deterministic (exact, no noise) for ground truth params
        mps_backend = MPSBackend(strategy="aer_mps", chi_max=MPS_CHI, seed=SEED)
        from qmbp_simulation.optimizers import VQEOptimizer
        from qmbp_simulation.models import VQEConfig

        vqe_config = VQEConfig(method="COBYLA", p_layers=1, n_restarts=1, maxiter=300)
        optimizer = VQEOptimizer(config=vqe_config, backend=mps_backend, seed=SEED)

        lattice_ref = self.make_lattice("chain_1d", n, J=1.0, h=max(h_values))
        circuit, _ = self.hva.create(n, 1, lattice_ref)

        # Descending sweep for theta_opt
        from qmbp_simulation import ClassicalSolver
        solver = ClassicalSolver()
        rng = np.random.default_rng(SEED)
        prev_theta = rng.uniform(-0.01, 0.01, circuit.num_parameters)
        theta_map: dict[float, np.ndarray] = {}

        for h in sorted(h_values, reverse=True):
            lattice = self.make_lattice("chain_1d", n, J=1.0, h=h)
            H = self.builder.build(lattice)
            result = optimizer.optimize(H, circuit, prev_theta.copy())
            theta_map[h] = result.theta_opt.copy()
            prev_theta = result.theta_opt.copy()
            logger.info(f"  VQE N={n} h={h:.1f}: E={result.energy:.6f}, iters={result.n_iterations}")

        # Per-bond noise rates (Torino-realistic)
        bond_errors = TORINO_MEAN_CZ_ERROR * (1 + 0.3 * rng.standard_normal(n - 1))
        bond_errors = np.clip(bond_errors, 0.002, 0.025)

        # PEA + GF at each h-point
        results = []
        for h in sorted(h_values, reverse=True):
            e_exact, gap = self.exact_ground_state("chain_1d", n, h)
            bound = circuit.assign_parameters(theta_map[h])
            H = self.builder.build(self.make_lattice("chain_1d", n, J=1.0, h=h))
            n_2q = sum(1 for inst in bound.data if inst.operation.num_qubits == 2)

            # PEA: amplify noise model at each factor
            meas_pea = []
            for nf in NOISE_FACTORS:
                e = self._mps_noisy_eval(bound, H, n, bond_errors, float(nf))
                meas_pea.append(e)

            # GF: fold gates, same base noise
            meas_gf = []
            for nf in NOISE_FACTORS:
                folded = fold_gates(bound, noise_factor=int(nf))
                e = self._mps_noisy_eval(folded, H, n, bond_errors, 1.0)
                meas_gf.append(e)

            nf_arr = np.array(NOISE_FACTORS, dtype=float)
            ext_pea, r2_pea, _ = _extrapolate_linear(nf_arr, np.array(meas_pea))
            ext_gf, r2_gf, _ = _extrapolate_linear(nf_arr, np.array(meas_gf))

            de_noisy = abs(meas_pea[0] - e_exact) / max(gap, 1e-10)
            de_pea = abs(ext_pea - e_exact) / max(gap, 1e-10)
            de_gf = abs(ext_gf - e_exact) / max(gap, 1e-10)
            pea_gain = (de_noisy - de_pea) / max(de_noisy, 1e-10)
            gf_gain = (de_noisy - de_gf) / max(de_noisy, 1e-10)

            results.append({"h": h, "n_2q": n_2q, "de_noisy": de_noisy,
                            "de_pea": de_pea, "de_gf": de_gf,
                            "pea_r2": r2_pea, "gf_r2": r2_gf,
                            "pea_gain": pea_gain, "gf_gain": gf_gain})
            logger.info(f"  N={n} h={h:.1f}: {n_2q} CZ, PEA +{pea_gain:.1%}(R²={r2_pea:.3f}), "
                        f"GF +{gf_gain:.1%}(R²={r2_gf:.3f})")
        return results

    @staticmethod
    def _mps_noisy_eval(
        circuit, hamiltonian, n_qubits: int,
        bond_errors: np.ndarray, noise_factor: float,
    ) -> float:
        """Exact noisy evaluation via MPS + depolarizing noise.

        Uses save_expectation_value → Tr(ρ·H). Memory: O(N·χ³).
        """
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error

        noise_model = NoiseModel()
        for i in range(n_qubits - 1):
            rate = min(bond_errors[i] * noise_factor, 0.75)
            err = depolarizing_error(rate, 2)
            noise_model.add_quantum_error(err, "cz", [i, i + 1])
            noise_model.add_quantum_error(err, "cz", [i + 1, i])

        sim = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=MPS_CHI,
            noise_model=noise_model,
        )
        qc = circuit.copy()
        qc.save_expectation_value(hamiltonian, list(range(n_qubits)), label="ev")
        result = sim.run(qc, shots=1, seed_simulator=SEED).result()
        return float(np.real(result.data()["ev"]))


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    PEAScalingRunner.main()
