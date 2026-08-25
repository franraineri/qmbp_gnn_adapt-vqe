#!/usr/bin/env python3
"""PEA Cross-Topology Dense Validation — 5 seeds × 6 h-points × 4 topologies.

Densifies the ZNE_CROSS_TOPO experiment with more seeds and h-points to
strengthen the statistical evidence for PEA superiority. Takes advantage
of the 10× PEA optimization (noise pair filtering) to run 120 PEA calls
in reasonable time (~3-5 min).

Hypothesis: PEA-ZNE outperforms GF-ZNE with p < 0.001 across 4 topologies
when tested with 5 seeds and 6 h-points per topology (120 total evaluations).

Sections:
  1. Chain_1d N=6 (5 seeds × 6 h-points = 30 evaluations)
  2. Ladder N=6 (5 seeds × 4 h-points = 20 evaluations)
  3. Heavy_hex N=10 (5 seeds × 4 h-points = 20 evaluations)
  4. Triangular N=6 (5 seeds × 4 h-points = 20 evaluations)
  5. Statistical verdict (paired t-test, Cohen's d, bootstrap CI)

Usage:
    .venv/bin/python scripts/experiment_runners/noise_zne_gf_pea/run_pea_cross_topology_dense.py
    .venv/bin/python scripts/experiment_runners/noise_zne_gf_pea/run_pea_cross_topology_dense.py --section 1
    .venv/bin/python scripts/experiment_runners/noise_zne_gf_pea/run_pea_cross_topology_dense.py --dry-run
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

NOISE_FACTORS = (1, 3, 5)
ZNE_SHOTS = 16384
SEEDS = [42, 43, 44, 45, 46]
N_CANDIDATE_LAYOUTS = 20

# Per-topology configs with dense h-grids (all within valid regime)
TOPOLOGY_CONFIGS = {
    "chain_1d": {
        "n_qubits": 6,
        "h_values": [2.5, 2.25, 2.0, 1.85, 1.75, 1.6],
    },
    "ladder": {
        "n_qubits": 6,
        "h_values": [3.5, 3.25, 3.0, 2.75],
    },
    "heavy_hex": {
        "n_qubits": 10,
        "h_values": [4.5, 4.0, 3.5, 3.25],
    },
    "triangular": {
        "n_qubits": 6,
        "h_values": [5.5, 5.0, 4.5, 4.0],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════


class PEACrossTopologyDenseRunner(ValidationRunner):
    """Dense PEA cross-topology: 5 seeds, 4 topologies, 6 h-points."""

    runner_id = "pea_cross_topology_dense"
    experiment_id = "PEA_CROSS_DENSE"
    description = "PEA Cross-Topology Dense (5 seeds × 4 topologies × 4-6 h-points)"
    hypothesis = (
        "PEA-ZNE is universally superior to GF-ZNE with p<0.001 across "
        "4 topologies and 5 seeds (90+ evaluations)."
    )

    def build_config(self) -> dict:
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "category": "ZNE",
            "description": self.description,
            "hypothesis": self.hypothesis,
            "system": {
                "topology": "chain_1d",
                "n_qubits": 6,
                "p_layers": 1,
                "model": "tfim",
            },
            "topologies": list(TOPOLOGY_CONFIGS.keys()),
            "seeds": SEEDS,
            "zne": {
                "noise_factors": list(NOISE_FACTORS),
                "shots": ZNE_SHOTS,
            },
        }

    def setup(self) -> None:
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        from qmbp_simulation import HamiltonianBuilder, make_lattice
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.execution import (
            NoiselessBackend,
            NoisyEstimatorConfig,
            build_adjacency,
            find_layouts_bfs,
            noisy_estimate,
            run_gate_folding_zne,
            run_pea_zne,
            select_layouts_low_ces,
        )

        self.builder = HamiltonianBuilder()
        self.hva = HVACircuitBuilder()
        self.noiseless = NoiselessBackend()
        self.make_lattice = make_lattice
        self.fake_backend = FakeTorino()
        self._noisy_config = NoisyEstimatorConfig(shots=ZNE_SHOTS, seed_simulator=42)
        self._noisy_estimate = noisy_estimate
        self._run_gf_zne = run_gate_folding_zne
        self._run_pea_zne = run_pea_zne
        self._select_low_ces = select_layouts_low_ces

        # Layout candidates per topology
        adj = build_adjacency(self.fake_backend)
        self._candidates = {}
        for topo, cfg in TOPOLOGY_CONFIGS.items():
            self._candidates[topo] = find_layouts_bfs(
                adj, cfg["n_qubits"], n_candidates=N_CANDIDATE_LAYOUTS
            )

        self._all_results: dict[str, list[dict]] = {}
        logger.info(f"[setup] {len(TOPOLOGY_CONFIGS)} topologies, {len(SEEDS)} seeds")

    def define_sections(self) -> list[Section]:
        sections = []
        for i, topo in enumerate(TOPOLOGY_CONFIGS.keys(), start=1):
            cfg = TOPOLOGY_CONFIGS[topo]
            n_evals = len(SEEDS) * len(cfg["h_values"])
            sections.append(
                Section(
                    id=i,
                    name=f"{topo} N={cfg['n_qubits']} ({n_evals} evals)",
                    fn=lambda t=topo: self._section_topology(t),
                    hypothesis=f"PEA gain > GF gain on {topo} (5 seeds)",
                )
            )
        sections.append(
            Section(
                id=len(TOPOLOGY_CONFIGS) + 1,
                name="Statistical Verdict",
                fn=self._section_verdict,
                hypothesis="PEA is universally superior (p < 0.001)",
            )
        )
        return sections

    # ─── Core sweep (reused per topology) ─────────────────────────────────

    def _section_topology(self, topology: str) -> dict:
        """Run PEA vs GF for one topology across all seeds and h-values."""
        cfg = TOPOLOGY_CONFIGS[topology]
        n_qubits = cfg["n_qubits"]
        h_values = cfg["h_values"]
        candidates = self._candidates[topology]

        from qmbp_simulation.models.model_registry import get_model_spec

        spec = get_model_spec("tfim")
        lattice_ref = self.make_lattice(topology, n_qubits, J=1.0, h=max(h_values))
        circuit, _ = spec.create_circuit(n_qubits, 1, lattice_ref)

        results = []
        for seed in SEEDS:
            theta_map = self.vqe_descending_sweep(
                topology=topology,
                n_qubits=n_qubits,
                h_values=h_values,
                seed=seed,
                p_layers=1,
                n_restarts=1,
                maxiter=500,
            )
            for h in sorted(h_values, reverse=True):
                e_exact, gap = self.exact_ground_state(topology, n_qubits, h)
                theta_opt = theta_map[h]
                bound = circuit.assign_parameters(theta_opt)

                layout_sel = self._select_low_ces(
                    bound,
                    self.fake_backend,
                    candidates,
                    n_select=1,
                    optimization_level=2,
                    max_ces=0.5,
                )
                transpiled = layout_sel.transpiled_circuits[0]
                H = self.builder.build(self.make_lattice(topology, n_qubits, J=1.0, h=h))
                H_mapped = H.apply_layout(transpiled.layout)

                e_noisy = self._noisy_estimate(
                    transpiled,
                    H_mapped,
                    self.fake_backend,
                    self._noisy_config,
                    seed_offset=seed,
                )
                de_noisy = abs(e_noisy - e_exact) / max(gap, 1e-10)

                gf = self._run_gf_zne(
                    transpiled,
                    H_mapped,
                    self.fake_backend,
                    self._noisy_config,
                    noise_factors=NOISE_FACTORS,
                    seed_offset=seed * 100,
                )
                de_gf = abs(gf.extrapolated_value - e_exact) / max(gap, 1e-10)

                pea = self._run_pea_zne(
                    transpiled,
                    H_mapped,
                    self.fake_backend,
                    self._noisy_config,
                    noise_factors=NOISE_FACTORS,
                    seed_offset=seed * 200,
                )
                de_pea = abs(pea.extrapolated_value - e_exact) / max(gap, 1e-10)

                gf_gain = (de_noisy - de_gf) / max(de_noisy, 1e-10)
                pea_gain = (de_noisy - de_pea) / max(de_noisy, 1e-10)

                results.append(
                    {
                        "topology": topology,
                        "n_qubits": n_qubits,
                        "seed": seed,
                        "h": h,
                        "de_noisy": de_noisy,
                        "de_gf": de_gf,
                        "de_pea": de_pea,
                        "gf_r2": gf.r_squared,
                        "pea_r2": pea.r_squared,
                        "gf_gain": gf_gain,
                        "pea_gain": pea_gain,
                    }
                )

            logger.info(
                f"  {topology} seed={seed}: "
                f"PEA mean={np.mean([r['pea_gain'] for r in results if r['seed'] == seed]):+.1%}"
            )

        self._all_results[topology] = results
        pea_gains = [r["pea_gain"] for r in results]
        gf_gains = [r["gf_gain"] for r in results]

        summary = {
            "topology": topology,
            "n_evaluations": len(results),
            "mean_pea_gain": float(np.mean(pea_gains)),
            "mean_gf_gain": float(np.mean(gf_gains)),
            "mean_pea_r2": float(np.mean([r["pea_r2"] for r in results])),
            "pea_wins": sum(1 for p, g in zip(pea_gains, gf_gains, strict=False) if p > g),
            "all_pea_positive": all(g > 0 for g in pea_gains),
        }
        logger.info(
            f"  {topology}: PEA={summary['mean_pea_gain']:+.1%}, "
            f"GF={summary['mean_gf_gain']:+.1%}, "
            f"wins={summary['pea_wins']}/{len(results)}"
        )

        passed = summary["all_pea_positive"] and summary["mean_pea_r2"] > 0.9
        return {"pass": passed, "results": results, "summary": summary}

    # ─── Verdict ──────────────────────────────────────────────────────────

    def _section_verdict(self) -> dict:
        """Cross-topology statistical verdict with dense data."""
        from scipy import stats

        all_results = []
        for topo_results in self._all_results.values():
            all_results.extend(topo_results)

        if not all_results:
            return {"pass": False, "error": "No results from topology sections"}

        gf_gains = np.array([r["gf_gain"] for r in all_results])
        pea_gains = np.array([r["pea_gain"] for r in all_results])

        t_stat, p_value = stats.ttest_rel(pea_gains, gf_gains)
        cohens_d = float(np.mean(pea_gains - gf_gains) / np.std(pea_gains - gf_gains))

        # Bootstrap 95% CI on mean PEA advantage
        rng = np.random.default_rng(99)
        diffs = pea_gains - gf_gains
        boot_means = [
            float(np.mean(rng.choice(diffs, size=len(diffs), replace=True))) for _ in range(1000)
        ]
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

        logger.info(f"\n  ═══ DENSE VERDICT ({len(all_results)} evaluations) ═══")
        logger.info(f"  Paired t-test: t={t_stat:.2f}, p={p_value:.2e}")
        logger.info(
            f"  Cohen's d: {cohens_d:.2f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'})"
        )
        logger.info(f"  Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
        logger.info(f"  PEA wins: {int(np.sum(pea_gains > gf_gains))}/{len(all_results)}")
        logger.info(f"  Mean PEA gain: {np.mean(pea_gains):+.1%}")
        logger.info(f"  Mean GF gain:  {np.mean(gf_gains):+.1%}")

        passed = p_value < 0.001 and np.all(pea_gains > 0)
        return {
            "pass": passed,
            "n_total": len(all_results),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": cohens_d,
            "ci_95": [ci_lo, ci_hi],
            "mean_pea_gain": float(np.mean(pea_gains)),
            "mean_gf_gain": float(np.mean(gf_gains)),
            "pea_wins": int(np.sum(pea_gains > gf_gains)),
        }


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    PEACrossTopologyDenseRunner.main()
