#!/usr/bin/env python3
"""3-Way ZNE Comparison: CES-ZNE vs Gate-Folding vs PEA.

Compares three noise mitigation strategies on the same circuit:
  1. CES-ZNE (inhomogeneous): 3 layouts → linear extrap to CES=0
  2. Gate-Folding ZNE: single layout, noise factors [1,3,5] → extrap to nf=0
  3. PEA-ZNE: single layout, learned noise amplification → extrap to nf=0

PEA (Probabilistic Error Amplification) is more physically accurate than
gate-folding because it amplifies ONLY the learned noise (not coherent errors).
However, it requires qiskit-aer and is slower due to noise model reconstruction.

Hypothesis:
  PEA-ZNE achieves:
  - R² ≥ GF-ZNE R² (noise model is more physical)
  - Gain ≥ GF-ZNE gain (amplification is more targeted)
  - Same circuit depth at all noise factors (no depth penalty)

Sections:
  1. Noiseless baseline — VQE to get θ_opt
  2. CES-ZNE — standard layout-based extrapolation
  3. Gate-Folding ZNE — digital gate repetition
  4. PEA-ZNE — probabilistic noise amplification
  5. 3-Way Comparison — side-by-side metrics

Usage:
    python scripts/experiment_runners/run_zne_3way_comparison.py
    python scripts/experiment_runners/run_zne_3way_comparison.py --topology heavy_hex --n-qubits 10
    python scripts/experiment_runners/run_zne_3way_comparison.py --section 1 4 5
    python scripts/experiment_runners/run_zne_3way_comparison.py --dry-run
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

H_TEST_VALUES = [2.5, 2.0, 1.75]

ZNE_N_LAYOUTS = 3
ZNE_SHOTS = 16384
N_CANDIDATE_LAYOUTS = 20
GF_NOISE_FACTORS = (1, 3, 5)
PEA_NOISE_FACTORS = (1, 3, 5)

VQE_RESTARTS = 3
VQE_MAXITER = 500


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════


class ZNE3WayComparisonRunner(ValidationRunner):
    """3-way comparison: CES-ZNE vs Gate-Folding vs PEA."""

    runner_id = "zne_3way_comparison"
    experiment_id = "ZNE_3WAY"
    description = "3-Way ZNE Comparison: CES vs Gate-Folding vs PEA"
    hypothesis = (
        "PEA-ZNE achieves gain ≥ GF-ZNE gain due to more physically "
        "accurate noise amplification (targets learned noise only)."
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
                name="Noiseless Baseline",
                fn=self._section_noiseless,
                hypothesis="VQE converges at all h-values (descending warm-start)",
            ),
            Section(
                id=2,
                name="CES-ZNE",
                fn=self._section_ces_zne,
                hypothesis="CES-ZNE R²>0.9 where CES spread exists",
            ),
            Section(
                id=3,
                name="Gate-Folding ZNE",
                fn=self._section_gf_zne,
                hypothesis="GF-ZNE achieves positive gain with R²>0.9",
            ),
            Section(
                id=4,
                name="PEA-ZNE",
                fn=self._section_pea_zne,
                hypothesis="PEA-ZNE gain > GF-ZNE gain with R²>0.9",
            ),
            Section(
                id=5,
                name="3-Way Comparison",
                fn=self._section_comparison,
                hypothesis="PEA > GF > CES in gain (targeted amplification wins)",
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
            "zne": {
                "gf_noise_factors": list(GF_NOISE_FACTORS),
                "pea_noise_factors": list(PEA_NOISE_FACTORS),
                "n_layouts_ces": ZNE_N_LAYOUTS,
                "shots": ZNE_SHOTS,
            },
            "seeds": [],
        }

    def setup(self) -> None:
        from qiskit_ibm_runtime.fake_provider import FakeTorino
        from scipy.optimize import minimize

        from qmbp_simulation import HamiltonianBuilder, make_lattice
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.execution import NoiselessBackend
        from qmbp_simulation.execution.noisy_utils import (
            NoisyEstimatorConfig,
            build_adjacency,
            find_layouts_bfs,
            noisy_estimate,
            run_gate_folding_zne,
            run_pea_zne,
            run_zne_deployment,
            select_layouts_by_circuit_ces,
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
        self.noisy_config = NoisyEstimatorConfig(shots=ZNE_SHOTS, seed_simulator=42)
        self.make_lattice = make_lattice
        self.minimize = minimize

        # Store callables
        self._noisy_estimate = noisy_estimate
        self._run_zne_deployment = run_zne_deployment
        self._run_gf_zne = run_gate_folding_zne
        self._run_pea_zne = run_pea_zne
        self._select_ces = select_layouts_by_circuit_ces
        self._select_low_ces = select_layouts_low_ces

        # Candidate layouts
        adj = build_adjacency(self.fake_backend)
        self.candidates = find_layouts_bfs(adj, self.n_qubits, n_candidates=N_CANDIDATE_LAYOUTS)
        logger.info(
            f"[setup] {self.topology} N={self.n_qubits} p={self.p_layers}, "
            f"{len(self.candidates)} layout candidates"
        )

        # Shared data across sections
        self._baseline: list[dict] = []
        self._ces_data: list[dict] = []
        self._gf_data: list[dict] = []
        self._pea_data: list[dict] = []

    # ─── Section 1: Noiseless VQE baseline ──────────────────────────────

    def _section_noiseless(self) -> dict:
        """Get θ_opt via noiseless VQE for each h-point."""
        lattice_ref = self.make_lattice(self.topology, self.n_qubits, J=1.0, h=max(H_TEST_VALUES))
        circuit, _ = self.hva.create(self.n_qubits, self.p_layers, lattice_ref)
        self._circuit = circuit
        n_params = circuit.num_parameters

        rng = np.random.default_rng(42)
        prev_theta = rng.uniform(-0.01, 0.01, n_params)
        results = []

        for h in sorted(H_TEST_VALUES, reverse=True):
            lattice = self.make_lattice(self.topology, self.n_qubits, J=1.0, h=h)
            H = self.builder.build(lattice)

            H_mat = H.to_matrix()
            if hasattr(H_mat, "toarray"):
                H_mat = H_mat.toarray()
            evals = np.sort(np.linalg.eigvalsh(H_mat))
            e_exact = float(evals[0])
            gap = float(evals[1] - evals[0])

            best_energy = float("inf")
            best_theta = prev_theta.copy()
            for restart in range(VQE_RESTARTS):
                x0 = prev_theta + rng.normal(0, 0.1, n_params) if restart > 0 else prev_theta.copy()
                x0 = np.clip(x0, -np.pi, np.pi)
                res = self.minimize(
                    lambda params, _H=H, _c=circuit: self.noiseless.evaluate(_c, _H, params),
                    x0,
                    method="L-BFGS-B",
                    bounds=[(-np.pi, np.pi)] * n_params,
                    options={"maxiter": VQE_MAXITER, "ftol": 1e-14},
                )
                if res.fun < best_energy:
                    best_energy = res.fun
                    best_theta = res.x.copy()
            prev_theta = best_theta.copy()

            de_gap = abs(best_energy - e_exact) / max(gap, 1e-10)
            results.append(
                {
                    "h": h,
                    "e_exact": e_exact,
                    "gap": gap,
                    "e_noiseless": best_energy,
                    "de_gap_noiseless": de_gap,
                    "theta_opt": best_theta.tolist(),
                }
            )
            logger.info(f"  h={h:.2f}: ΔE/gap={de_gap:.4f}")

        self._baseline = results
        return {"n_params": n_params, "results": results}

    # ─── Section 2: CES-ZNE ─────────────────────────────────────────────

    def _section_ces_zne(self) -> dict:
        """CES-ZNE with 3-layout spread-maximizing."""
        if not self._baseline:
            raise RuntimeError("Run Section 1 first")

        results = []
        for pt in self._baseline:
            h, theta_opt = pt["h"], np.array(pt["theta_opt"])
            e_exact, gap = pt["e_exact"], pt["gap"]

            lattice = self.make_lattice(self.topology, self.n_qubits, J=1.0, h=h)
            H = self.builder.build(lattice)
            bound = self._circuit.assign_parameters(theta_opt)

            layout_sel = self._select_ces(
                bound, self.fake_backend, self.candidates, n_select=ZNE_N_LAYOUTS
            )
            ces_vals = [float(c) for c in layout_sel.ces_values]

            # Noisy raw (lowest-CES layout)
            best_idx = int(np.argmin(ces_vals))
            transpiled_raw = layout_sel.transpiled_circuits[best_idx]
            H_mapped = H.apply_layout(transpiled_raw.layout)
            e_noisy = self._noisy_estimate(
                transpiled_raw, H_mapped, self.fake_backend, self.noisy_config
            )

            # CES-ZNE extrapolation
            zne = self._run_zne_deployment(
                bound, H, self.fake_backend, layout_sel, self.noisy_config, self.n_qubits
            )
            e_zne = zne.energy_zne.extrapolated_value
            r2 = zne.energy_zne.r_squared

            de_noisy = abs(e_noisy - e_exact) / max(gap, 1e-10)
            de_zne = abs(e_zne - e_exact) / max(gap, 1e-10)
            gain = (de_noisy - de_zne) / max(de_noisy, 1e-10)

            results.append(
                {
                    "h": h,
                    "e_noisy_raw": e_noisy,
                    "e_ces_zne": e_zne,
                    "de_noisy_raw": de_noisy,
                    "de_ces_zne": de_zne,
                    "ces_gain": gain,
                    "ces_r2": r2,
                    "ces_values": ces_vals,
                }
            )
            logger.info(f"  h={h:.2f}: gain={gain:+.1%}, R²={r2:.4f}")

        self._ces_data = results
        return {"results": results, "mean_gain": float(np.mean([r["ces_gain"] for r in results]))}

    # ─── Section 3: Gate-Folding ZNE ────────────────────────────────────

    def _section_gf_zne(self) -> dict:
        """Gate-folding ZNE on single best layout."""
        if not self._baseline:
            raise RuntimeError("Run Section 1 first")

        results = []
        for pt in self._baseline:
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
            ces = float(layout_sel.ces_values[0])
            H_mapped = H.apply_layout(transpiled.layout)

            e_noisy = self._noisy_estimate(
                transpiled, H_mapped, self.fake_backend, self.noisy_config
            )

            gf = self._run_gf_zne(
                transpiled,
                H_mapped,
                self.fake_backend,
                self.noisy_config,
                noise_factors=GF_NOISE_FACTORS,
                extrapolator=self.extrapolator,
                seed_offset=500,
            )

            de_noisy = abs(e_noisy - e_exact) / max(gap, 1e-10)
            de_gf = abs(gf.extrapolated_value - e_exact) / max(gap, 1e-10)
            gain = (de_noisy - de_gf) / max(de_noisy, 1e-10)

            results.append(
                {
                    "h": h,
                    "e_noisy_raw": e_noisy,
                    "e_gf_zne": gf.extrapolated_value,
                    "de_noisy_raw": de_noisy,
                    "de_gf_zne": de_gf,
                    "gf_gain": gain,
                    "gf_r2": gf.r_squared,
                    "gf_slope": gf.slope,
                    "layout_ces": ces,
                    "measured": gf.measured_values,
                }
            )
            logger.info(f"  h={h:.2f}: gain={gain:+.1%}, R²={gf.r_squared:.4f}")

        self._gf_data = results
        return {"results": results, "mean_gain": float(np.mean([r["gf_gain"] for r in results]))}

    # ─── Section 4: PEA-ZNE ─────────────────────────────────────────────

    def _section_pea_zne(self) -> dict:
        """PEA-ZNE: probabilistic noise amplification via learned model."""
        if not self._baseline:
            raise RuntimeError("Run Section 1 first")

        results = []
        for pt in self._baseline:
            h, theta_opt = pt["h"], np.array(pt["theta_opt"])
            e_exact, gap = pt["e_exact"], pt["gap"]

            lattice = self.make_lattice(self.topology, self.n_qubits, J=1.0, h=h)
            H = self.builder.build(lattice)
            bound = self._circuit.assign_parameters(theta_opt)

            # Use same lowest-CES layout as GF-ZNE for fair comparison
            layout_sel = self._select_low_ces(
                bound,
                self.fake_backend,
                self.candidates,
                n_select=1,
                optimization_level=2,
                max_ces=0.5,
            )
            transpiled = layout_sel.transpiled_circuits[0]
            ces = float(layout_sel.ces_values[0])
            H_mapped = H.apply_layout(transpiled.layout)

            e_noisy = self._noisy_estimate(
                transpiled, H_mapped, self.fake_backend, self.noisy_config
            )

            # PEA-ZNE (noise amplification via learned error rates)
            pea = self._run_pea_zne(
                transpiled,
                H_mapped,
                self.fake_backend,
                self.noisy_config,
                noise_factors=PEA_NOISE_FACTORS,
                extrapolator=self.extrapolator,
                seed_offset=2000,
            )

            de_noisy = abs(e_noisy - e_exact) / max(gap, 1e-10)
            de_pea = abs(pea.extrapolated_value - e_exact) / max(gap, 1e-10)
            gain = (de_noisy - de_pea) / max(de_noisy, 1e-10)

            results.append(
                {
                    "h": h,
                    "e_noisy_raw": e_noisy,
                    "e_pea_zne": pea.extrapolated_value,
                    "de_noisy_raw": de_noisy,
                    "de_pea_zne": de_pea,
                    "pea_gain": gain,
                    "pea_r2": pea.r_squared,
                    "pea_slope": pea.slope,
                    "layout_ces": ces,
                    "measured": pea.measured_values,
                    "learned_rates": pea.learned_error_rates,
                }
            )
            logger.info(
                f"  h={h:.2f}: gain={gain:+.1%}, R²={pea.r_squared:.4f}, "
                f"learned_rates={pea.learned_error_rates}"
            )

        self._pea_data = results
        return {"results": results, "mean_gain": float(np.mean([r["pea_gain"] for r in results]))}

    # ─── Section 5: 3-Way Comparison ────────────────────────────────────

    def _section_comparison(self) -> dict:
        """Side-by-side comparison of all three ZNE methods."""
        if not self._ces_data or not self._gf_data or not self._pea_data:
            raise RuntimeError("Sections 2, 3, 4 must run first")

        comparison = []
        for ces, gf, pea, bl in zip(
            self._ces_data, self._gf_data, self._pea_data, self._baseline, strict=True
        ):
            h = ces["h"]
            methods = {
                "CES-ZNE": {"de": ces["de_ces_zne"], "r2": ces["ces_r2"], "gain": ces["ces_gain"]},
                "GF-ZNE": {"de": gf["de_gf_zne"], "r2": gf["gf_r2"], "gain": gf["gf_gain"]},
                "PEA-ZNE": {"de": pea["de_pea_zne"], "r2": pea["pea_r2"], "gain": pea["pea_gain"]},
            }
            # Winner = lowest ΔE/gap
            winner = min(methods, key=lambda k: methods[k]["de"])

            comparison.append(
                {
                    "h": h,
                    "e_exact": bl["e_exact"],
                    "gap": bl["gap"],
                    "de_noiseless": bl["de_gap_noiseless"],
                    "de_noisy_raw": ces["de_noisy_raw"],
                    **{f"de_{k.lower().replace('-', '_')}": v["de"] for k, v in methods.items()},
                    **{f"r2_{k.lower().replace('-', '_')}": v["r2"] for k, v in methods.items()},
                    **{
                        f"gain_{k.lower().replace('-', '_')}": v["gain"] for k, v in methods.items()
                    },
                    "winner": winner,
                    "pea_vs_gf_improvement": (
                        (gf["de_gf_zne"] - pea["de_pea_zne"]) / max(gf["de_gf_zne"], 1e-10)
                    ),
                }
            )

        # Print table
        logger.info("")
        logger.info(
            f"  {'h':>5} | {'Noisy':>7} | "
            f"{'CES':>7} {'R²':>5} {'gain':>6} | "
            f"{'GF':>7} {'R²':>5} {'gain':>6} | "
            f"{'PEA':>7} {'R²':>5} {'gain':>6} | Winner"
        )
        logger.info("  " + "-" * 90)
        for row in comparison:
            logger.info(
                f"  {row['h']:5.2f} | {row['de_noisy_raw']:7.4f} | "
                f"{row['de_ces_zne']:7.4f} {row['r2_ces_zne']:5.3f} {row['gain_ces_zne']:+5.1%} | "
                f"{row['de_gf_zne']:7.4f} {row['r2_gf_zne']:5.3f} {row['gain_gf_zne']:+5.1%} | "
                f"{row['de_pea_zne']:7.4f} {row['r2_pea_zne']:5.3f} {row['gain_pea_zne']:+5.1%} | "
                f"{row['winner']}"
            )

        # Aggregate
        mean_ces = float(np.mean([r["gain_ces_zne"] for r in comparison]))
        mean_gf = float(np.mean([r["gain_gf_zne"] for r in comparison]))
        mean_pea = float(np.mean([r["gain_pea_zne"] for r in comparison]))
        mean_pea_r2 = float(np.mean([r["r2_pea_zne"] for r in comparison]))
        mean_gf_r2 = float(np.mean([r["r2_gf_zne"] for r in comparison]))

        wins = {"CES-ZNE": 0, "GF-ZNE": 0, "PEA-ZNE": 0}
        for row in comparison:
            wins[row["winner"]] += 1

        summary = {
            "topology": self.topology,
            "n_qubits": self.n_qubits,
            "p_layers": self.p_layers,
            "n_points": len(comparison),
            "wins": wins,
            "mean_ces_gain": mean_ces,
            "mean_gf_gain": mean_gf,
            "mean_pea_gain": mean_pea,
            "mean_gf_r2": mean_gf_r2,
            "mean_pea_r2": mean_pea_r2,
            "pea_better_than_gf": mean_pea > mean_gf,
            "best_method": max(
                [("CES-ZNE", mean_ces), ("GF-ZNE", mean_gf), ("PEA-ZNE", mean_pea)],
                key=lambda x: x[1],
            )[0],
        }

        logger.info("")
        logger.info("  ─── VERDICT ───")
        logger.info(f"  Wins: CES={wins['CES-ZNE']}, GF={wins['GF-ZNE']}, PEA={wins['PEA-ZNE']}")
        logger.info(f"  Mean gain: CES={mean_ces:+.1%}, GF={mean_gf:+.1%}, PEA={mean_pea:+.1%}")
        logger.info(f"  Mean R²:   GF={mean_gf_r2:.4f}, PEA={mean_pea_r2:.4f}")
        logger.info(f"  PEA > GF: {summary['pea_better_than_gf']}")
        logger.info(f"  Best method overall: {summary['best_method']}")

        return {"comparison": comparison, "summary": summary}


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ZNE3WayComparisonRunner.main()
