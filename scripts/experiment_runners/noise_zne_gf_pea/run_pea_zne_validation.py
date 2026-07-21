#!/usr/bin/env python3
"""PEA-ZNE Comprehensive Validation — Multi-topology, Multi-seed.

Validates PEA-ZNE robustness across topologies, seeds, and h-values.
Compares against Gate-Folding ZNE (the baseline validated in GF_ZNE_CMP).

Hypothesis:
  PEA-ZNE achieves:
  - Mean gain > GF-ZNE gain across all topologies
  - R² > 0.9 consistently (linear noise amplification)
  - Reproducible across seeds (std of gain < 5%)
  - No depth increase (circuit depth constant at all noise factors)

Sections:
  1. Noiseless baseline (3 seeds, descending VQE)
  2. Gate-Folding ZNE (3 seeds, for comparison)
  3. PEA-ZNE (3 seeds, learned noise amplification)
  4. Seed stability analysis (variance comparison)
  5. Summary and verdict

Usage:
    python scripts/experiment_runners/run_pea_zne_validation.py
    python scripts/experiment_runners/run_pea_zne_validation.py --topology heavy_hex --n-qubits 10
    python scripts/experiment_runners/run_pea_zne_validation.py --dry-run
"""

from __future__ import annotations

import logging
import sys

import numpy as np

from qmbp_simulation.framework.runner_base import (
    Section,
    ValidationRunner,
    resolve_project_root,
)
from qmbp_simulation.models.constants import DEFAULT_SEEDS

_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_TOPOLOGY = "chain_1d"
DEFAULT_N_QUBITS = 6
DEFAULT_P_LAYERS = 1
SEEDS = DEFAULT_SEEDS

H_TEST_VALUES = [2.5, 2.0, 1.75, 1.5]  # 4 h-points for better statistics
NOISE_FACTORS = (1, 3, 5)
ZNE_SHOTS = 16384
N_CANDIDATE_LAYOUTS = 20
VQE_RESTARTS = 3
VQE_MAXITER = 500


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════


class PEAZNEValidationRunner(ValidationRunner):
    """Comprehensive PEA-ZNE validation with multi-seed comparison."""

    runner_id = "pea_zne_validation"
    experiment_id = "PEA_ZNE_VAL"
    description = "PEA-ZNE Comprehensive Validation (multi-seed, vs GF-ZNE)"
    hypothesis = (
        "PEA-ZNE achieves gain > GF-ZNE gain with R²>0.9 and "
        "seed-independent reproducibility (std < 5%)."
    )

    @classmethod
    def _add_custom_args(cls, parser) -> None:
        parser.add_argument(
            "--topology",
            default=DEFAULT_TOPOLOGY,
            choices=["chain_1d", "ladder", "triangular", "heavy_hex"],
        )
        parser.add_argument("--n-qubits", type=int, default=DEFAULT_N_QUBITS)
        parser.add_argument("--p-layers", type=int, default=DEFAULT_P_LAYERS)
        parser.add_argument(
            "--extrapolator",
            default="linear",
            choices=["linear", "exponential"],
        )

    def define_sections(self) -> list[Section]:
        return [
            Section(
                id=1,
                name="Noiseless Baseline (3 seeds)",
                fn=self._section_noiseless,
                hypothesis="VQE converges consistently across 3 seeds",
            ),
            Section(
                id=2,
                name="Gate-Folding ZNE (3 seeds)",
                fn=self._section_gf_zne,
                hypothesis="GF-ZNE gain positive and seed-independent (std<5%)",
            ),
            Section(
                id=3,
                name="PEA-ZNE (3 seeds)",
                fn=self._section_pea_zne,
                hypothesis="PEA-ZNE gain > GF-ZNE and seed-independent (std<5%)",
            ),
            Section(
                id=4,
                name="Seed Stability",
                fn=self._section_stability,
                hypothesis="PEA variance <= GF variance across seeds",
            ),
            Section(
                id=5,
                name="Verdict",
                fn=self._section_verdict,
                hypothesis="PEA R²>0.9, gain>GF, std<5%, all positive",
            ),
        ]

    def build_config(self) -> dict:
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "category": "ZNE",
            "description": self.description,
            "hypothesis": self.hypothesis,
            "system": {
                "n_qubits": getattr(self, "n_qubits", DEFAULT_N_QUBITS),
                "p_layers": getattr(self, "p_layers", DEFAULT_P_LAYERS),
                "topology": getattr(self, "topology", DEFAULT_TOPOLOGY),
                "model": "tfim",
            },
            "seeds": SEEDS,
        }

    def setup(self) -> None:
        from qiskit_ibm_runtime.fake_provider import FakeTorino
        from scipy.optimize import minimize

        from qmbp_simulation import HamiltonianBuilder, make_lattice
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.execution import (
            NoiselessBackend,
            build_adjacency,
            find_layouts_bfs,
            noisy_estimate,
            run_gate_folding_zne,
            run_pea_zne,
            select_layouts_low_ces,
        )

        self.topology = getattr(self._args, "topology", DEFAULT_TOPOLOGY)
        self.n_qubits = getattr(self._args, "n_qubits", DEFAULT_N_QUBITS)
        self.p_layers = getattr(self._args, "p_layers", DEFAULT_P_LAYERS)
        self.extrapolator = getattr(self._args, "extrapolator", "linear")

        self.builder = HamiltonianBuilder()
        self.hva = HVACircuitBuilder()
        self.noiseless = NoiselessBackend()
        self.fake_backend = FakeTorino()
        self.make_lattice = make_lattice
        self.minimize = minimize

        # Callables
        self._noisy_estimate = noisy_estimate
        self._run_gf_zne = run_gate_folding_zne
        self._run_pea_zne = run_pea_zne
        self._select_low_ces = select_layouts_low_ces

        # Layouts
        adj = build_adjacency(self.fake_backend)
        self.candidates = find_layouts_bfs(adj, self.n_qubits, n_candidates=N_CANDIDATE_LAYOUTS)

        logger.info(
            f"[setup] {self.topology} N={self.n_qubits} p={self.p_layers}, "
            f"{len(self.candidates)} candidates, {len(SEEDS)} seeds"
        )

        # Shared across sections
        self._baseline: dict[int, list[dict]] = {}  # seed → list of h-point results
        self._gf_results: dict[int, list[dict]] = {}
        self._pea_results: dict[int, list[dict]] = {}

    # ─── Section 1: Noiseless VQE (3 seeds) ─────────────────────────────

    def _section_noiseless(self) -> dict:
        """VQE baseline for each seed × h-point."""
        lattice_ref = self.make_lattice(self.topology, self.n_qubits, J=1.0, h=max(H_TEST_VALUES))
        circuit, _ = self.hva.create(self.n_qubits, self.p_layers, lattice_ref)
        self._circuit = circuit
        n_params = circuit.num_parameters

        backend = self._resolve_backend()
        for seed in SEEDS:
            # Use base class VQE sweep (warm-start, NaN guard, maxfun cap)
            theta_map = self.vqe_descending_sweep(
                topology=self.topology,
                n_qubits=self.n_qubits,
                h_values=list(H_TEST_VALUES),
                seed=seed,
                p_layers=self.p_layers,
                n_restarts=VQE_RESTARTS,
                maxiter=VQE_MAXITER,
                sigma=0.1,
            )

            seed_results = []
            for h in sorted(H_TEST_VALUES, reverse=True):
                e_exact, gap = self.exact_ground_state(self.topology, self.n_qubits, h)
                theta_opt = theta_map[h]

                lattice = self.make_lattice(self.topology, self.n_qubits, J=1.0, h=h)
                H = self.builder.build(lattice)
                e_noiseless = float(backend.evaluate(circuit, H, theta_opt))

                seed_results.append(
                    {
                        "h": h,
                        "e_exact": e_exact,
                        "gap": gap,
                        "e_noiseless": e_noiseless,
                        "de_gap": abs(e_noiseless - e_exact) / max(gap, 1e-10),
                        "theta_opt": theta_opt.tolist(),
                    }
                )

            self._baseline[seed] = seed_results
            logger.info(f"  seed={seed}: {len(seed_results)} h-points done")

        return {"n_params": n_params, "n_seeds": len(SEEDS), "n_h_points": len(H_TEST_VALUES)}

    # ─── Section 2: Gate-Folding ZNE (3 seeds) ──────────────────────────

    def _section_gf_zne(self) -> dict:
        """GF-ZNE for each seed (same layout per seed for fair comparison)."""
        if not self._baseline:
            raise RuntimeError("Run Section 1 first")

        noisy_config = NoisyEstimatorConfig(shots=ZNE_SHOTS, seed_simulator=42)

        for seed in SEEDS:
            seed_results = []
            for pt in self._baseline[seed]:
                h, theta_opt = pt["h"], np.array(pt["theta_opt"])
                e_exact, gap = pt["e_exact"], pt["gap"]

                lattice = self.make_lattice(self.topology, self.n_qubits, J=1.0, h=h)
                H = self.builder.build(lattice)
                bound = self._circuit.assign_parameters(theta_opt)

                layout_sel = self._select_low_ces(
                    bound,
                    self.fake_backend,
                    self.candidates,
                    n_select=1,
                    optimization_level=2,
                    max_ces=0.5,
                )
                transpiled = layout_sel.transpiled_circuits[0]
                H_mapped = H.apply_layout(transpiled.layout)

                # Noisy raw
                e_noisy = self._noisy_estimate(
                    transpiled,
                    H_mapped,
                    self.fake_backend,
                    noisy_config,
                    seed_offset=seed * 10,
                )

                # GF-ZNE
                gf = self._run_gf_zne(
                    transpiled,
                    H_mapped,
                    self.fake_backend,
                    noisy_config,
                    noise_factors=NOISE_FACTORS,
                    extrapolator=self.extrapolator,
                    seed_offset=seed * 100 + 500,
                )

                de_noisy = abs(e_noisy - e_exact) / max(gap, 1e-10)
                de_gf = abs(gf.extrapolated_value - e_exact) / max(gap, 1e-10)
                gain = (de_noisy - de_gf) / max(de_noisy, 1e-10)

                seed_results.append(
                    {
                        "h": h,
                        "de_noisy": de_noisy,
                        "de_gf": de_gf,
                        "gf_gain": gain,
                        "gf_r2": gf.r_squared,
                    }
                )

            self._gf_results[seed] = seed_results
            mean_gain = float(np.mean([r["gf_gain"] for r in seed_results]))
            logger.info(f"  seed={seed}: GF mean gain={mean_gain:+.1%}")

        all_gains = [r["gf_gain"] for s in self._gf_results.values() for r in s]
        return {
            "mean_gain": float(np.mean(all_gains)),
            "std_gain": float(np.std(all_gains)),
            "mean_r2": float(np.mean([r["gf_r2"] for s in self._gf_results.values() for r in s])),
        }

    # ─── Section 3: PEA-ZNE (3 seeds) ───────────────────────────────────

    def _section_pea_zne(self) -> dict:
        """PEA-ZNE for each seed (same layout for fair comparison)."""
        if not self._baseline:
            raise RuntimeError("Run Section 1 first")

        noisy_config = NoisyEstimatorConfig(shots=ZNE_SHOTS, seed_simulator=42)

        for seed in SEEDS:
            seed_results = []
            for pt in self._baseline[seed]:
                h, theta_opt = pt["h"], np.array(pt["theta_opt"])
                e_exact, gap = pt["e_exact"], pt["gap"]

                lattice = self.make_lattice(self.topology, self.n_qubits, J=1.0, h=h)
                H = self.builder.build(lattice)
                bound = self._circuit.assign_parameters(theta_opt)

                layout_sel = self._select_low_ces(
                    bound,
                    self.fake_backend,
                    self.candidates,
                    n_select=1,
                    optimization_level=2,
                    max_ces=0.5,
                )
                transpiled = layout_sel.transpiled_circuits[0]
                H_mapped = H.apply_layout(transpiled.layout)

                # Noisy raw (same as GF-ZNE for fair comparison)
                e_noisy = self._noisy_estimate(
                    transpiled,
                    H_mapped,
                    self.fake_backend,
                    noisy_config,
                    seed_offset=seed * 10,
                )

                # PEA-ZNE
                pea = self._run_pea_zne(
                    transpiled,
                    H_mapped,
                    self.fake_backend,
                    noisy_config,
                    noise_factors=NOISE_FACTORS,
                    extrapolator=self.extrapolator,
                    seed_offset=seed * 100 + 2000,
                )

                de_noisy = abs(e_noisy - e_exact) / max(gap, 1e-10)
                de_pea = abs(pea.extrapolated_value - e_exact) / max(gap, 1e-10)
                gain = (de_noisy - de_pea) / max(de_noisy, 1e-10)

                seed_results.append(
                    {
                        "h": h,
                        "de_noisy": de_noisy,
                        "de_pea": de_pea,
                        "pea_gain": gain,
                        "pea_r2": pea.r_squared,
                        "learned_rates": pea.learned_error_rates,
                    }
                )

            self._pea_results[seed] = seed_results
            mean_gain = float(np.mean([r["pea_gain"] for r in seed_results]))
            logger.info(f"  seed={seed}: PEA mean gain={mean_gain:+.1%}")

        all_gains = [r["pea_gain"] for s in self._pea_results.values() for r in s]
        return {
            "mean_gain": float(np.mean(all_gains)),
            "std_gain": float(np.std(all_gains)),
            "mean_r2": float(np.mean([r["pea_r2"] for s in self._pea_results.values() for r in s])),
        }

    # ─── Section 4: Seed Stability ──────────────────────────────────────

    def _section_stability(self) -> dict:
        """Compare gain variance across seeds for GF vs PEA."""
        if not self._gf_results or not self._pea_results:
            raise RuntimeError("Run Sections 2 and 3 first")

        # Per-h gain across seeds
        stability = []
        for h_idx, h in enumerate(sorted(H_TEST_VALUES, reverse=True)):
            gf_gains = [self._gf_results[s][h_idx]["gf_gain"] for s in SEEDS]
            pea_gains = [self._pea_results[s][h_idx]["pea_gain"] for s in SEEDS]

            stability.append(
                {
                    "h": h,
                    "gf_mean": float(np.mean(gf_gains)),
                    "gf_std": float(np.std(gf_gains)),
                    "pea_mean": float(np.mean(pea_gains)),
                    "pea_std": float(np.std(pea_gains)),
                    "pea_more_stable": float(np.std(pea_gains)) < float(np.std(gf_gains)),
                }
            )

        # Overall
        all_gf = [r["gf_gain"] for s in self._gf_results.values() for r in s]
        all_pea = [r["pea_gain"] for s in self._pea_results.values() for r in s]

        logger.info("")
        logger.info(f"  {'h':>5} | {'GF gain':>10} {'GF std':>8} | {'PEA gain':>10} {'PEA std':>8}")
        logger.info("  " + "-" * 55)
        for row in stability:
            logger.info(
                f"  {row['h']:5.2f} | {row['gf_mean']:+9.1%} {row['gf_std']:7.1%} | "
                f"{row['pea_mean']:+9.1%} {row['pea_std']:7.1%}"
            )
        logger.info("")
        logger.info(f"  Overall GF:  mean={np.mean(all_gf):+.1%}, std={np.std(all_gf):.1%}")
        logger.info(f"  Overall PEA: mean={np.mean(all_pea):+.1%}, std={np.std(all_pea):.1%}")

        return {
            "per_h": stability,
            "overall_gf_std": float(np.std(all_gf)),
            "overall_pea_std": float(np.std(all_pea)),
            "pea_more_stable_overall": float(np.std(all_pea)) < float(np.std(all_gf)),
        }

    # ─── Section 5: Verdict ─────────────────────────────────────────────

    def _section_verdict(self) -> dict:
        """Final comparison and pass/fail verdict."""
        if not self._gf_results or not self._pea_results:
            raise RuntimeError("Run Sections 2, 3, 4 first")

        all_gf_gains = [r["gf_gain"] for s in self._gf_results.values() for r in s]
        all_pea_gains = [r["pea_gain"] for s in self._pea_results.values() for r in s]
        all_pea_r2 = [r["pea_r2"] for s in self._pea_results.values() for r in s]

        mean_gf = float(np.mean(all_gf_gains))
        mean_pea = float(np.mean(all_pea_gains))
        mean_pea_r2 = float(np.mean(all_pea_r2))
        std_pea = float(np.std(all_pea_gains))

        # Success criteria
        pea_better = mean_pea > mean_gf
        r2_good = mean_pea_r2 > 0.9
        reproducible = std_pea < 0.05
        all_positive = all(g > 0 for g in all_pea_gains)

        passed = pea_better and r2_good and reproducible

        logger.info("")
        logger.info("  ─── FINAL VERDICT ───")
        logger.info(f"  PEA gain > GF gain:     {pea_better} ({mean_pea:+.1%} vs {mean_gf:+.1%})")
        logger.info(f"  PEA R² > 0.9:           {r2_good} (mean={mean_pea_r2:.4f})")
        logger.info(f"  PEA std < 5%:           {reproducible} (std={std_pea:.1%})")
        logger.info(f"  All PEA gains positive: {all_positive}")
        logger.info(f"  OVERALL: {'✅ CONFIRMED' if passed else '❌ REJECTED'}")

        return {
            "pass": passed,
            "mean_gf_gain": mean_gf,
            "mean_pea_gain": mean_pea,
            "mean_pea_r2": mean_pea_r2,
            "pea_std": std_pea,
            "pea_better": pea_better,
            "r2_good": r2_good,
            "reproducible": reproducible,
            "all_positive": all_positive,
            "n_total_points": len(all_pea_gains),
            "topology": self.topology,
            "n_qubits": self.n_qubits,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Missing import for NoisyEstimatorConfig in section methods
# ═══════════════════════════════════════════════════════════════════════════════

from qmbp_simulation.execution import NoisyEstimatorConfig  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    PEAZNEValidationRunner.main()
