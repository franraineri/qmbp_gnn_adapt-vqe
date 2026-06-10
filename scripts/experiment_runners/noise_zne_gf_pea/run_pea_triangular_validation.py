#!/usr/bin/env python3
"""PEA-ZNE Triangular Validation — Close coverage gap G6.

Validates PEA-ZNE on triangular N=6 p=1 with 3 seeds × 3 h-points.
This topology has committed thesis data (results/thesis/analysis_p1_zne/triangular_*)
but NO dedicated PEA comparison experiment.

Hypothesis: PEA-ZNE outperforms GF-ZNE on triangular N=6 with R²>0.9
and positive gain, matching the pattern seen on chain_1d/ladder/heavy_hex.

Usage:
    python scripts/experiment_runners/noise_zne_gf_pea/run_pea_triangular_validation.py
    python scripts/experiment_runners/noise_zne_gf_pea/run_pea_triangular_validation.py --dry-run
    python scripts/experiment_runners/noise_zne_gf_pea/run_pea_triangular_validation.py --section 1
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

NOISE_FACTORS = (1, 3, 5)
ZNE_SHOTS = 16384
N_CANDIDATE_LAYOUTS = 20

# Defaults (overridable via CLI)
DEFAULT_TOPOLOGY = "triangular"
DEFAULT_N_QUBITS = 6
DEFAULT_P_LAYERS = 1

# Valid-regime minimum h per topology (p=1 N=6)
_H_TEST_MAP = {
    "triangular": [4.5, 4.0, 3.5],
    "chain_1d": [2.5, 2.0, 1.75],
    "ladder": [3.5, 3.25, 3.0],
    "heavy_hex": [3.5, 3.25, 3.0],
}
SEEDS = [42, 43, 44]


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════


class PEATriangularRunner(ValidationRunner):
    """PEA vs GF-ZNE validation — configurable topology/N/p."""

    runner_id = "pea_triangular"
    experiment_id = "PEA_TRIANGULAR"
    description = "PEA-ZNE validation on triangular N=6 p=1 (gap G6)"
    hypothesis = (
        "PEA-ZNE outperforms GF-ZNE on triangular N=6 with R²>0.9 "
        "and positive gain, consistent with all other topologies."
    )

    @classmethod
    def _add_custom_args(cls, parser):
        parser.add_argument(
            "--topology",
            type=str,
            default=DEFAULT_TOPOLOGY,
            help=f"Topology to validate (default: {DEFAULT_TOPOLOGY})",
        )
        parser.add_argument(
            "--n-qubits",
            type=int,
            default=DEFAULT_N_QUBITS,
            help=f"Number of qubits (default: {DEFAULT_N_QUBITS})",
        )
        parser.add_argument(
            "--p-layers",
            type=int,
            default=DEFAULT_P_LAYERS,
            help=f"HVA layers (default: {DEFAULT_P_LAYERS})",
        )

    def define_sections(self) -> list[Section]:
        topo = self._args.topology
        n = self._args.n_qubits
        p = self._args.p_layers
        return [
            Section(
                id=1,
                name=f"{topo} N={n} p={p} (3 seeds × 3 h-points)",
                fn=self._section_sweep,
                hypothesis="PEA gain > GF gain, R²>0.9, all positive",
            ),
            Section(
                id=2,
                name="Statistical Verdict",
                fn=self._section_verdict,
                hypothesis="PEA wins majority with p<0.05",
            ),
        ]

    def build_config(self) -> dict:
        topo = self._args.topology
        n = self._args.n_qubits
        p = self._args.p_layers
        h_test = _H_TEST_MAP.get(topo, [3.5, 3.0, 2.5])

        # Update instance metadata for dynamic config
        self.description = f"PEA-ZNE validation on {topo} N={n} p={p}"
        self.hypothesis = (
            f"PEA-ZNE outperforms GF-ZNE on {topo} N={n} with R²>0.9 "
            "and positive gain, consistent with all other topologies."
        )

        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "category": "ZNE",
            "description": self.description,
            "hypothesis": self.hypothesis,
            "system": {
                "topology": topo,
                "n_qubits": n,
                "p_layers": p,
                "model": "tfim",
            },
            "seeds": SEEDS,
            "h_test": h_test,
            "zne": {
                "noise_factors": list(NOISE_FACTORS),
                "shots": ZNE_SHOTS,
                "n_candidate_layouts": N_CANDIDATE_LAYOUTS,
                "extrapolator": "linear",
            },
        }

    def setup(self) -> None:
        """Import dependencies and prepare backend."""
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        from qmbp_simulation.execution.noisy_utils import (
            NoisyEstimatorConfig,
            build_adjacency,
            find_layouts_bfs,
            noisy_estimate,
            run_gate_folding_zne,
            run_pea_zne,
            select_layouts_low_ces,
        )

        topo = self._args.topology
        n = self._args.n_qubits
        p = self._args.p_layers
        self._topology = topo
        self._n_qubits = n
        self._p_layers = p
        self._h_test = _H_TEST_MAP.get(topo, [3.5, 3.0, 2.5])

        self._fake_backend = FakeTorino()
        self._noisy_config = NoisyEstimatorConfig(shots=ZNE_SHOTS, seed_simulator=42)
        self._noisy_estimate = noisy_estimate
        self._run_gf_zne = run_gate_folding_zne
        self._run_pea_zne = run_pea_zne
        self._select_low_ces = select_layouts_low_ces

        adj = build_adjacency(self._fake_backend)
        self._candidates = find_layouts_bfs(adj, n, n_candidates=N_CANDIDATE_LAYOUTS)
        self._all_results: list[dict] = []

        logger.info(f"[setup] FakeTorino loaded, {len(self._candidates)} candidates for {topo} N={n} p={p}")

    def _section_sweep(self) -> dict:
        """Run PEA vs GF on configured topology across all seeds and h-points."""
        from qmbp_simulation import HamiltonianBuilder, make_lattice
        from qmbp_simulation.models.model_registry import get_model_spec

        topo = self._topology
        n = self._n_qubits
        p = self._p_layers
        h_test = self._h_test

        spec = get_model_spec("tfim")
        builder = HamiltonianBuilder()
        lattice_ref = make_lattice(topo, n, J=1.0, h=max(h_test))
        circuit, _ = spec.create_circuit(n, p, lattice_ref)

        results = []
        for seed in SEEDS:
            theta_map = self.vqe_descending_sweep(
                topology=topo,
                n_qubits=n,
                h_values=h_test,
                seed=seed,
                p_layers=p,
                n_restarts=1,
                maxiter=500,
            )
            for h in sorted(h_test, reverse=True):
                e_exact, gap = self.exact_ground_state(topo, n, h)
                theta_opt = theta_map[h]

                lattice = make_lattice(topo, n, J=1.0, h=h)
                H = builder.build(lattice)
                bound = circuit.assign_parameters(theta_opt)
                layout_sel = self._select_low_ces(
                    bound,
                    self._fake_backend,
                    self._candidates,
                    n_select=1,
                    optimization_level=2,
                    max_ces=0.5,
                )
                transpiled = layout_sel.transpiled_circuits[0]
                H_mapped = H.apply_layout(transpiled.layout)

                # Noisy baseline
                e_noisy = self._noisy_estimate(
                    transpiled,
                    H_mapped,
                    self._fake_backend,
                    self._noisy_config,
                    seed_offset=seed,
                )
                de_noisy = abs(e_noisy - e_exact) / max(gap, 1e-10)

                # GF-ZNE
                t0 = time.time()
                gf = self._run_gf_zne(
                    transpiled,
                    H_mapped,
                    self._fake_backend,
                    self._noisy_config,
                    noise_factors=NOISE_FACTORS,
                    extrapolator="linear",
                    seed_offset=seed * 100,
                )
                gf_time = time.time() - t0
                de_gf = abs(gf.extrapolated_value - e_exact) / max(gap, 1e-10)

                # PEA-ZNE
                t0 = time.time()
                pea = self._run_pea_zne(
                    transpiled,
                    H_mapped,
                    self._fake_backend,
                    self._noisy_config,
                    noise_factors=NOISE_FACTORS,
                    extrapolator="linear",
                    seed_offset=seed * 200,
                )
                pea_time = time.time() - t0
                de_pea = abs(pea.extrapolated_value - e_exact) / max(gap, 1e-10)

                gf_gain = (de_noisy - de_gf) / max(de_noisy, 1e-10)
                pea_gain = (de_noisy - de_pea) / max(de_noisy, 1e-10)

                row = {
                    "topology": topo,
                    "n_qubits": n,
                    "seed": seed,
                    "h": h,
                    "e_exact": e_exact,
                    "gap": gap,
                    "e_noisy": e_noisy,
                    "e_gf": gf.extrapolated_value,
                    "e_pea": pea.extrapolated_value,
                    "de_noisy": de_noisy,
                    "de_gf": de_gf,
                    "de_pea": de_pea,
                    "gf_r2": gf.r_squared,
                    "pea_r2": pea.r_squared,
                    "gf_gain": gf_gain,
                    "pea_gain": pea_gain,
                    "gf_slope": gf.slope if hasattr(gf, "slope") else None,
                    "pea_slope": pea.slope if hasattr(pea, "slope") else None,
                    "gf_time_s": round(gf_time, 2),
                    "pea_time_s": round(pea_time, 2),
                }
                results.append(row)
                logger.info(
                    f"  seed={seed} h={h:.2f}: "
                    f"GF={de_gf:.4f}(R²={gf.r_squared:.3f},+{gf_gain:.1%}) "
                    f"PEA={de_pea:.4f}(R²={pea.r_squared:.3f},+{pea_gain:.1%})"
                )

        self._all_results = results
        gains_pea = [r["pea_gain"] for r in results]
        gains_gf = [r["gf_gain"] for r in results]
        r2_pea = [r["pea_r2"] for r in results]
        r2_gf = [r["gf_r2"] for r in results]
        passed = np.mean(r2_pea) > 0.9 and all(g > 0 for g in gains_pea)
        return {
            "pass": passed,
            "results": results,
            "summary": {
                "n_evaluations": len(results),
                "mean_pea_gain": float(np.mean(gains_pea)),
                "std_pea_gain": float(np.std(gains_pea)),
                "mean_gf_gain": float(np.mean(gains_gf)),
                "std_gf_gain": float(np.std(gains_gf)),
                "mean_pea_r2": float(np.mean(r2_pea)),
                "mean_gf_r2": float(np.mean(r2_gf)),
                "min_pea_r2": float(np.min(r2_pea)),
                "pea_always_positive": all(g > 0 for g in gains_pea),
                "gf_always_positive": all(g > 0 for g in gains_gf),
                "pea_wins": int(np.sum(np.array(gains_pea) > np.array(gains_gf))),
            },
        }

    def _section_verdict(self) -> dict:
        """Paired t-test and final verdict."""
        from scipy import stats

        if not self._all_results:
            return {"pass": False, "error": "No results from section 1"}

        gf_gains = np.array([r["gf_gain"] for r in self._all_results])
        pea_gains = np.array([r["pea_gain"] for r in self._all_results])
        t_stat, p_value = stats.ttest_rel(pea_gains, gf_gains)

        summary = {
            "n_evaluations": len(self._all_results),
            "mean_pea_gain": float(np.mean(pea_gains)),
            "mean_gf_gain": float(np.mean(gf_gains)),
            "pea_advantage": float(np.mean(pea_gains - gf_gains)),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "pea_wins": int(np.sum(pea_gains > gf_gains)),
        }
        logger.info(
            f"\n  Triangular verdict: PEA={summary['mean_pea_gain']:+.1%} vs "
            f"GF={summary['mean_gf_gain']:+.1%}, t={t_stat:.2f}, p={p_value:.4f}"
        )
        passed = p_value < 0.05 and summary["pea_wins"] >= len(self._all_results) // 2
        return {"pass": passed, "summary": summary}


if __name__ == "__main__":
    PEATriangularRunner.main()
