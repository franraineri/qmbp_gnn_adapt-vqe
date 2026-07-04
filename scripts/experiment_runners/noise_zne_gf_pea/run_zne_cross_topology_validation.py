#!/usr/bin/env python3
"""ZNE Cross-Topology Validation — Definitive PEA vs GF-ZNE Comparison.

Fills remaining coverage gaps in ZNE validation:
  1. PEA-ZNE on ladder N=6 (never tested as dedicated experiment)
  2. PEA-ZNE multi-seed on heavy_hex N=10 (only seed=42 exists)
  3. Consolidated comparison table across all topologies

Hypothesis: PEA-ZNE outperforms GF-ZNE across ALL topologies with
R²>0.9 and positive gain, providing a universal mitigation strategy
for IBM Torino hardware deployment.

Sections:
  1. Ladder N=6 p=1: PEA vs GF (3 seeds × 3 h-points = 9 evaluations)
  2. Heavy_hex N=10 p=1: PEA multi-seed (seeds 43,44 × 3 h-points = 6 new)
  3. Chain_1d N=6 p=1: Control (seed=42, 3 h-points — verify reproducibility)
  4. Cross-topology verdict: statistical summary + paired t-test

Best practices (from template_validation_runner.py):
- Uses self.vqe_descending_sweep() for noiseless parameter optimization.
- Uses self.exact_ground_state() for (e_exact, gap) lookup.
- Caches layout selection for fair cross-method comparison.
- Includes build_config() with system params for digest/compare.py compatibility.

Usage:
    python scripts/experiment_runners/run_zne_cross_topology_validation.py
    python scripts/experiment_runners/run_zne_cross_topology_validation.py --dry-run
    python scripts/experiment_runners/run_zne_cross_topology_validation.py --section 1
    python scripts/experiment_runners/run_zne_cross_topology_validation.py --stop-on-failure
    python scripts/experiment_runners/run_zne_cross_topology_validation.py --verbose
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

# ─── Project root setup (works from any script depth) ────────────────────────
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

# Per-topology configs (topology, n_qubits, h_test_points)
CONFIGS = {
    "ladder": {"n_qubits": 6, "p_layers": 1, "h_test": [2.50, 2.25, 2.00]},
    "heavy_hex": {"n_qubits": 10, "p_layers": 1, "h_test": [4.00, 3.25, 3.00]},
    "chain_1d": {"n_qubits": 6, "p_layers": 1, "h_test": [2.50, 2.00, 1.75]},
}

# Seeds per section
SEEDS_LADDER = DEFAULT_SEEDS
SEEDS_HEAVY_HEX_NEW = [43, 44]  # seed=42 already validated in PEA_HW_READY
SEEDS_CHAIN_CONTROL = [42]  # Control: verify consistency


# ═══════════════════════════════════════════════════════════════════════════════
# Runner Implementation
# ═══════════════════════════════════════════════════════════════════════════════


class ZNECrossTopologyRunner(ValidationRunner):
    """Cross-topology PEA vs GF-ZNE validation.

    Sections:
        1. Ladder N=6 p=1 — first dedicated PEA validation on ladder
        2. Heavy_hex N=10 p=1 — multi-seed reproducibility (seeds 43, 44)
        3. Chain_1d N=6 p=1 — control run for consistency with PEA_ZNE_VAL
        4. Cross-topology verdict — paired t-test and coverage summary
    """

    # ── Required class attributes ────────────────────────────────────────────
    runner_id = "zne_cross_topology"
    experiment_id = "ZNE_CROSS_TOPO"
    description = "ZNE Cross-Topology Validation (PEA vs GF, 3 topologies)"
    hypothesis = (
        "PEA-ZNE outperforms GF-ZNE across all topologies (ladder, heavy_hex, "
        "chain_1d) with R²>0.9 and consistent positive gain."
    )

    # ── Section definitions ──────────────────────────────────────────────────

    def define_sections(self) -> list[Section]:
        return [
            Section(
                id=1,
                name="Ladder N=6 p=1 (3 seeds)",
                fn=self._section_ladder,
                hypothesis="PEA-ZNE gain > GF-ZNE gain on ladder with R²>0.9",
            ),
            Section(
                id=2,
                name="Heavy_hex N=10 p=1 (seeds 43,44)",
                fn=self._section_heavy_hex,
                hypothesis="PEA reproducible across seeds on heavy_hex (std<10%)",
            ),
            Section(
                id=3,
                name="Chain_1d N=6 p=1 (control)",
                fn=self._section_chain_control,
                hypothesis="PEA results consistent with prior PEA_ZNE_VAL run",
            ),
            Section(
                id=4,
                name="Cross-Topology Verdict",
                fn=self._section_verdict,
                hypothesis="PEA is universally superior (paired t-test p<0.05)",
            ),
        ]

    # ── Config for result envelope (digest/compare.py compatible) ────────────

    def build_config(self) -> dict:
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "category": "ZNE",
            "description": self.description,
            "hypothesis": self.hypothesis,
            "system": {
                "topology": "chain_1d",  # Primary for preflight checks
                "n_qubits": 6,  # Min tested
                "p_layers": 1,
                "model": "tfim",
            },
            "topologies_tested": list(CONFIGS.keys()),
            "seeds": SEEDS_LADDER + SEEDS_HEAVY_HEX_NEW + SEEDS_CHAIN_CONTROL,
            "zne": {
                "noise_factors": list(NOISE_FACTORS),
                "shots": ZNE_SHOTS,
                "n_candidate_layouts": N_CANDIDATE_LAYOUTS,
                "extrapolator": "linear",
            },
        }

    # ── Setup: lazy imports + shared state initialization ────────────────────

    def setup(self) -> None:
        """Import heavy dependencies and pre-compute layout candidates.

        Pattern: store expensive modules as self._ attributes.
        Cross-section data sharing via self._all_results dict.
        """
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

        self._fake_backend = FakeTorino()
        self._noisy_config = NoisyEstimatorConfig(shots=ZNE_SHOTS, seed_simulator=42)
        self._noisy_estimate = noisy_estimate
        self._run_gf_zne = run_gate_folding_zne
        self._run_pea_zne = run_pea_zne
        self._select_low_ces = select_layouts_low_ces

        # Pre-compute layout candidates for each topology
        adj = build_adjacency(self._fake_backend)
        self._candidates = {}
        for topo, cfg in CONFIGS.items():
            self._candidates[topo] = find_layouts_bfs(
                adj, cfg["n_qubits"], n_candidates=N_CANDIDATE_LAYOUTS
            )

        # Cross-section data store
        self._all_results: dict[str, list[dict]] = {}

        logger.info(
            f"[setup] FakeTorino loaded, {len(CONFIGS)} topology candidates "
            f"pre-computed ({sum(len(v) for v in self._candidates.values())} total)"
        )

    # ── Core sweep method (reused by all topology sections) ──────────────────

    def _run_topology_sweep(self, topology: str, seeds: list[int]) -> list[dict]:
        """Run PEA vs GF-ZNE comparison for a given topology across seeds.

        Uses framework helpers:
        - self.vqe_descending_sweep() for warm-start VQE parameter optimization
        - self.exact_ground_state() for (e_exact, gap) lookup

        Caches layout selection per h-value for fair comparison between
        GF-ZNE and PEA-ZNE (same physical layout → isolates method difference).

        Returns a list of per-(seed, h) result dicts with all metrics.
        """
        from qmbp_simulation import HamiltonianBuilder, make_lattice
        from qmbp_simulation.models.model_registry import get_model_spec

        cfg = CONFIGS[topology]
        n_qubits = cfg["n_qubits"]
        p_layers = cfg["p_layers"]
        h_test_points = cfg["h_test"]
        candidates = self._candidates[topology]

        spec = get_model_spec("tfim")
        builder = HamiltonianBuilder()

        # Build circuit once per topology (same structure at all h for TFIM)
        lattice_ref = make_lattice(topology, n_qubits, J=1.0, h=max(h_test_points))
        circuit, _ = spec.create_circuit(n_qubits, p_layers, lattice_ref)

        results = []

        for seed in seeds:
            # Use framework helper for VQE sweep (warm-start descending)
            theta_map = self.vqe_descending_sweep(
                topology=topology,
                n_qubits=n_qubits,
                h_values=h_test_points,
                seed=seed,
                p_layers=p_layers,
                n_restarts=1,  # p=1 only needs 1 restart
                maxiter=500,
            )

            for h in sorted(h_test_points, reverse=True):
                # Framework helper for exact ground state
                e_exact, gap = self.exact_ground_state(topology, n_qubits, h)
                theta_opt = theta_map[h]

                # Build Hamiltonian and transpile with low-CES layout
                lattice = make_lattice(topology, n_qubits, J=1.0, h=h)
                H = builder.build(lattice)
                bound = circuit.assign_parameters(theta_opt)
                layout_sel = self._select_low_ces(
                    bound,
                    self._fake_backend,
                    candidates,
                    n_select=1,
                    optimization_level=2,
                    max_ces=0.5,
                )
                transpiled = layout_sel.transpiled_circuits[0]
                H_mapped = H.apply_layout(transpiled.layout)

                # Noisy baseline (no mitigation)
                e_noisy = self._noisy_estimate(
                    transpiled,
                    H_mapped,
                    self._fake_backend,
                    self._noisy_config,
                    seed_offset=seed,
                )
                de_noisy = abs(e_noisy - e_exact) / max(gap, 1e-10)

                # GF-ZNE (same layout)
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

                # PEA-ZNE (same layout — fair comparison)
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

                # Gains (relative to noisy baseline)
                gf_gain = (de_noisy - de_gf) / max(de_noisy, 1e-10)
                pea_gain = (de_noisy - de_pea) / max(de_noisy, 1e-10)

                row = {
                    "topology": topology,
                    "n_qubits": n_qubits,
                    "seed": seed,
                    "h": h,
                    "e_exact": e_exact,
                    "gap": gap,
                    "de_noisy": de_noisy,
                    "de_gf": de_gf,
                    "de_pea": de_pea,
                    "gf_r2": gf.r_squared,
                    "pea_r2": pea.r_squared,
                    "gf_gain": gf_gain,
                    "pea_gain": pea_gain,
                    "gf_time_s": round(gf_time, 2),
                    "pea_time_s": round(pea_time, 2),
                }
                results.append(row)
                logger.info(
                    f"  {topology} seed={seed} h={h:.2f}: "
                    f"GF={de_gf:.4f}(R²={gf.r_squared:.3f},+{gf_gain:.1%}) "
                    f"PEA={de_pea:.4f}(R²={pea.r_squared:.3f},+{pea_gain:.1%})"
                )

        return results

    # ── Section implementations ──────────────────────────────────────────────

    def _section_ladder(self) -> dict:
        """Section 1: Ladder N=6 p=1 — first dedicated PEA validation on ladder."""
        results = self._run_topology_sweep("ladder", SEEDS_LADDER)
        self._all_results["ladder"] = results

        gains_pea = [r["pea_gain"] for r in results]
        gains_gf = [r["gf_gain"] for r in results]
        r2_pea = [r["pea_r2"] for r in results]

        summary = {
            "n_evaluations": len(results),
            "mean_pea_gain": float(np.mean(gains_pea)),
            "mean_gf_gain": float(np.mean(gains_gf)),
            "mean_pea_r2": float(np.mean(r2_pea)),
            "std_pea_gain": float(np.std(gains_pea)),
            "pea_always_positive": all(g > 0 for g in gains_pea),
            "pea_wins": sum(1 for p, g in zip(gains_pea, gains_gf, strict=True) if p > g),
        }
        logger.info(
            f"\n  Ladder summary: PEA gain={summary['mean_pea_gain']:+.1%} "
            f"(std={summary['std_pea_gain']:.1%}), "
            f"GF gain={summary['mean_gf_gain']:+.1%}, "
            f"PEA R²={summary['mean_pea_r2']:.3f}"
        )

        passed = summary["mean_pea_r2"] > 0.9 and summary["pea_always_positive"]
        return {"pass": passed, "results": results, "summary": summary}

    def _section_heavy_hex(self) -> dict:
        """Section 2: Heavy_hex N=10 p=1 — multi-seed PEA reproducibility."""
        results = self._run_topology_sweep("heavy_hex", SEEDS_HEAVY_HEX_NEW)
        self._all_results["heavy_hex"] = results

        gains_pea = [r["pea_gain"] for r in results]
        r2_pea = [r["pea_r2"] for r in results]

        # Per-h std across seeds (reproducibility metric)
        h_points = CONFIGS["heavy_hex"]["h_test"]
        per_h_std = []
        for h in h_points:
            h_gains = [r["pea_gain"] for r in results if abs(r["h"] - h) < 0.01]
            if len(h_gains) > 1:
                per_h_std.append(float(np.std(h_gains)))

        summary = {
            "n_evaluations": len(results),
            "mean_pea_gain": float(np.mean(gains_pea)),
            "mean_pea_r2": float(np.mean(r2_pea)),
            "std_pea_gain_across_seeds": float(np.std(gains_pea)),
            "per_h_std": per_h_std,
            "pea_always_positive": all(g > 0 for g in gains_pea),
            "seeds_tested": SEEDS_HEAVY_HEX_NEW,
        }
        logger.info(
            f"\n  Heavy_hex multi-seed: PEA gain={summary['mean_pea_gain']:+.1%} "
            f"(std={summary['std_pea_gain_across_seeds']:.1%}), "
            f"R²={summary['mean_pea_r2']:.3f}"
        )

        # Pass: reproducible (std < 10%) and R² > 0.9
        passed = (
            summary["std_pea_gain_across_seeds"] < 0.10
            and summary["mean_pea_r2"] > 0.9
            and summary["pea_always_positive"]
        )
        return {"pass": passed, "results": results, "summary": summary}

    def _section_chain_control(self) -> dict:
        """Section 3: Chain_1d N=6 p=1 — control for consistency check."""
        results = self._run_topology_sweep("chain_1d", SEEDS_CHAIN_CONTROL)
        self._all_results["chain_1d"] = results

        gains_pea = [r["pea_gain"] for r in results]
        r2_pea = [r["pea_r2"] for r in results]

        summary = {
            "n_evaluations": len(results),
            "mean_pea_gain": float(np.mean(gains_pea)),
            "mean_pea_r2": float(np.mean(r2_pea)),
            "pea_always_positive": all(g > 0 for g in gains_pea),
        }
        logger.info(
            f"\n  Chain control: PEA gain={summary['mean_pea_gain']:+.1%}, "
            f"R²={summary['mean_pea_r2']:.3f}"
        )

        # Pass: positive gain and R² > 0.9 (consistency with PEA_ZNE_VAL)
        passed = summary["mean_pea_r2"] > 0.9 and summary["pea_always_positive"]
        return {"pass": passed, "results": results, "summary": summary}

    def _section_verdict(self) -> dict:
        """Section 4: Cross-topology statistical verdict.

        Aggregates all results from sections 1-3 and performs:
        - Per-topology summary table
        - Paired t-test (PEA gain vs GF gain across all evaluations)
        - Coverage matrix for compare.py compatibility
        """
        from scipy import stats

        all_results = []
        for section_results in self._all_results.values():
            all_results.extend(section_results)

        if not all_results:
            return {"pass": False, "error": "No results from previous sections"}

        # Aggregate by topology
        topology_summary = {}
        for topo in CONFIGS:
            topo_data = [r for r in all_results if r["topology"] == topo]
            if not topo_data:
                continue
            topology_summary[topo] = {
                "n_evaluations": len(topo_data),
                "n_qubits": CONFIGS[topo]["n_qubits"],
                "mean_gf_gain": float(np.mean([r["gf_gain"] for r in topo_data])),
                "mean_pea_gain": float(np.mean([r["pea_gain"] for r in topo_data])),
                "mean_gf_r2": float(np.mean([r["gf_r2"] for r in topo_data])),
                "mean_pea_r2": float(np.mean([r["pea_r2"] for r in topo_data])),
                "pea_wins": sum(1 for r in topo_data if r["pea_gain"] > r["gf_gain"]),
            }

        # Paired t-test: PEA gain vs GF gain across ALL points
        gf_gains = np.array([r["gf_gain"] for r in all_results])
        pea_gains = np.array([r["pea_gain"] for r in all_results])
        diff = pea_gains - gf_gains

        t_stat, p_value = stats.ttest_rel(pea_gains, gf_gains)

        # Overall statistics
        overall = {
            "n_total_evaluations": len(all_results),
            "n_topologies": len(topology_summary),
            "mean_gf_gain": float(np.mean(gf_gains)),
            "mean_pea_gain": float(np.mean(pea_gains)),
            "mean_pea_advantage": float(np.mean(diff)),
            "std_pea_advantage": float(np.std(diff)),
            "paired_t_stat": float(t_stat),
            "paired_p_value": float(p_value),
            "pea_wins_total": int(np.sum(pea_gains > gf_gains)),
            "gf_wins_total": int(np.sum(gf_gains > pea_gains)),
            "mean_pea_r2": float(np.mean([r["pea_r2"] for r in all_results])),
            "mean_gf_r2": float(np.mean([r["gf_r2"] for r in all_results])),
            "pea_all_positive": bool(np.all(pea_gains > 0)),
            "gf_all_positive": bool(np.all(gf_gains > 0)),
        }

        # Print verdict table
        logger.info("\n  ═══ CROSS-TOPOLOGY VERDICT ═══\n")
        logger.info(
            f"  {'Topology':<12} {'N':>3} {'GF gain':>9} {'PEA gain':>10} "
            f"{'GF R²':>6} {'PEA R²':>7} {'PEA wins':>9}"
        )
        logger.info(f"  {'-' * 12} {'-' * 3} {'-' * 9} {'-' * 10} {'-' * 6} {'-' * 7} {'-' * 9}")
        for topo, ts in topology_summary.items():
            logger.info(
                f"  {topo:<12} {ts['n_qubits']:>3} "
                f"{ts['mean_gf_gain']:>+8.1%} {ts['mean_pea_gain']:>+9.1%} "
                f"{ts['mean_gf_r2']:>6.3f} {ts['mean_pea_r2']:>7.3f} "
                f"{ts['pea_wins']:>4}/{ts['n_evaluations']}"
            )

        logger.info(f"\n  Overall ({overall['n_total_evaluations']} evaluations):")
        logger.info(f"    Mean GF gain:  {overall['mean_gf_gain']:+.1%}")
        logger.info(f"    Mean PEA gain: {overall['mean_pea_gain']:+.1%}")
        logger.info(
            f"    PEA advantage: {overall['mean_pea_advantage']:+.1%} "
            f"± {overall['std_pea_advantage']:.1%}"
        )
        logger.info(
            f"    Paired t-test: t={overall['paired_t_stat']:.2f}, "
            f"p={overall['paired_p_value']:.4f}"
        )
        logger.info(
            f"    PEA wins:      {overall['pea_wins_total']}/{overall['n_total_evaluations']}"
        )
        logger.info(f"    All PEA > 0:   {overall['pea_all_positive']}")

        # Pass criteria:
        # 1. PEA gain significantly > GF gain (p < 0.05)
        # 2. PEA R² > 0.9 across all topologies
        # 3. PEA always positive
        passed = (
            overall["paired_p_value"] < 0.05
            and overall["mean_pea_r2"] > 0.9
            and overall["pea_all_positive"]
        )

        # Build comparison list (compare.py compatible format)
        comparison = [
            {
                "topology": r["topology"],
                "n_qubits": r["n_qubits"],
                "seed": r["seed"],
                "h": r["h"],
                "de_noisy": r["de_noisy"],
                "de_gf": r["de_gf"],
                "de_pea": r["de_pea"],
                "gf_r2": r["gf_r2"],
                "pea_r2": r["pea_r2"],
                "gf_gain": r["gf_gain"],
                "pea_gain": r["pea_gain"],
            }
            for r in all_results
        ]

        return {
            "pass": passed,
            "comparison": comparison,
            "summary": {
                "topology": "cross_topology",
                "n_qubits": 0,
                **overall,
                "per_topology": topology_summary,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ZNECrossTopologyRunner.main()
