#!/usr/bin/env python3
"""Transpiler & Noise Suppression Exploration — Pre-Hardware Validation.

Hypothesis: optimization_level=3 (KAK decomposition + resynthesis) may reduce
2Q gate count for our HVA p=1 N=10 heavy_hex circuit, improving hardware
execution quality. Additionally, the full noise suppression stack (DD + twirling
+ TREX) combined with better transpilation may yield better results than our
current PEA-ZNE alone.

This script compares:
  Section 1: Transpiler levels (0, 1, 2, 3) — gate counts, depth, CES
  Section 2: Level 3 with approximation_degree sweep — fidelity vs gate count
  Section 3: Noise suppression stack comparison (noisy simulation):
             - Bare noise (no mitigation)
             - DD only
             - DD + Twirling
             - DD + Twirling + TREX
             - Full stack + PEA-ZNE (current config)
  Section 4: Best transpiler + best noise stack combined

Metrics:
  - qc.depth(), 2Q gate count, CES
  - ΔE/gap (noiseless vs exact, noisy vs exact, mitigated vs exact)
  - Unitary fidelity (level 3 approx vs exact)

References:
  - IBM docs: https://quantum.cloud.ibm.com/docs/guides/set-optimization
  - IBM docs: https://quantum.cloud.ibm.com/docs/guides/common-parameters
  - IBM docs: https://quantum.cloud.ibm.com/docs/guides/configure-error-suppression
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from qmbp_simulation import HVACircuitBuilder, make_lattice
from qmbp_simulation.execution.noisy_utils import (
    NoisyEstimatorConfig,
    build_adjacency,
    compute_circuit_ces,
    find_layouts_bfs,
    noisy_estimate,
    select_layouts_low_ces,
)
from qmbp_simulation.framework import (
    Section,
    ValidationRunner,
)
from qmbp_simulation.solvers import ClassicalSolver

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# Hardware deployment target
N_QUBITS = 10
P_LAYERS = 1
TOPOLOGY = "heavy_hex"
H_TEST = 3.25  # Deep paramagnetic (well within valid regime)
J_COUPLING = 1.0

# Transpiler comparison settings
OPT_LEVELS = [0, 1, 2, 3]
APPROX_DEGREES = [1.0, 0.99, 0.98, 0.95, 0.90]  # For level 3 only

# Noisy simulation
SHOTS = 16384
SEED = 42


@dataclass
class TranspilerResult:
    """Result for one transpiler configuration."""

    optimization_level: int
    approximation_degree: float
    depth: int
    n_2q_gates: int
    n_1q_gates: int
    ces: float
    layout: list[int]
    basis_gate_counts: dict[str, int] = field(default_factory=dict)
    fidelity_vs_exact: float | None = None  # Only for approx_degree < 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════


class TranspilerExplorationRunner(ValidationRunner):
    """Compare transpiler options and noise suppression strategies."""

    runner_id = "transpiler_exploration"
    experiment_id = "TRANSPILER_EXPLORATION"
    description = (
        "Compare optimization_level 0-3, approximation_degree, "
        "and noise suppression stack for HVA p=1 N=10 heavy_hex"
    )
    hypothesis = (
        "Level 3 + KAK resynthesis reduces 2Q gates; PEA-ZNE remains best noise mitigation strategy"
    )

    def define_sections(self) -> list[Section]:
        return [
            Section(
                id=1,
                name="Transpiler Level Comparison",
                fn=self.section_transpiler_levels,
                hypothesis=("Level 3 reduces 2Q gates via KAK resynthesis of 2Q blocks"),
            ),
            Section(
                id=2,
                name="Approximation Degree Sweep (Level 3)",
                fn=self.section_approx_degree,
                hypothesis=(
                    "approx_degree<1.0 at level 3 can trade <1% fidelity for fewer 2Q gates"
                ),
            ),
            Section(
                id=3,
                name="Noise Suppression Stack Comparison",
                fn=self.section_noise_stack,
                hypothesis=(
                    "Full suppression stack (DD+twirl+TREX) combined with "
                    "PEA-ZNE gives best ΔE/gap on FakeTorino"
                ),
            ),
            Section(
                id=4,
                name="Combined: Best Transpiler + Best Noise Stack",
                fn=self.section_combined,
                hypothesis=(
                    "Level 3 transpilation + PEA-ZNE yields lower ΔE/gap than level 2 + PEA-ZNE"
                ),
            ),
        ]

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _build_circuit_and_hamiltonian(self):
        """Build HVA circuit and TFIM Hamiltonian for test point."""
        from qmbp_simulation.models.model_registry import get_model_spec

        lattice = make_lattice(TOPOLOGY, N_QUBITS, J=J_COUPLING, h=H_TEST)
        spec = get_model_spec("tfim")
        H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)

        circuit_builder = HVACircuitBuilder()
        qc, theta = circuit_builder.create(N_QUBITS, P_LAYERS, lattice)

        # Use a representative theta (small random near zero — typical VQE output)
        rng = np.random.default_rng(SEED)
        theta_vals = rng.uniform(-0.5, 0.5, size=len(theta))
        bound = qc.assign_parameters(dict(zip(theta, theta_vals, strict=False)))

        return bound, H, lattice

    def _get_fake_backend(self):
        """Get FakeTorino backend."""
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        return FakeTorino()

    def _exact_ground_state(self, lattice):
        """Compute exact ground state energy and gap."""
        from qmbp_simulation.models.model_registry import get_model_spec

        spec = get_model_spec("tfim")
        H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
        solver = ClassicalSolver()
        result = solver.solve(H, lattice)
        return float(result.energy), float(result.gap)

    def _transpile_with_options(
        self,
        bound_circuit,
        backend,
        layout: list[int],
        optimization_level: int = 2,
        approximation_degree: float = 1.0,
    ):
        """Transpile circuit with given options. Returns transpiled circuit."""
        from qiskit.transpiler.preset_passmanagers import (
            generate_preset_pass_manager,
        )

        kwargs = {
            "optimization_level": optimization_level,
            "backend": backend,
            "initial_layout": layout,
        }
        if approximation_degree < 1.0:
            kwargs["approximation_degree"] = approximation_degree

        pm = generate_preset_pass_manager(**kwargs)
        return pm.run(bound_circuit)

    def _analyze_transpiled(self, transpiled, backend) -> dict:
        """Extract metrics from a transpiled circuit."""
        ces, n_2q = compute_circuit_ces(transpiled, backend)
        depth = transpiled.depth()

        # Count gate types
        ops = transpiled.count_ops()
        n_1q = sum(
            count
            for gate, count in ops.items()
            if gate not in {"cx", "cz", "ecr", "rzz", "rxx", "ryy", "cp", "barrier", "measure"}
        )

        return {
            "depth": depth,
            "n_2q_gates": n_2q,
            "n_1q_gates": n_1q,
            "ces": ces,
            "gate_counts": dict(ops),
        }

    # ──────────────────────────────────────────────────────────────────
    # Section 1: Transpiler Level Comparison
    # ──────────────────────────────────────────────────────────────────

    def section_transpiler_levels(self) -> dict:
        """Compare optimization_level 0, 1, 2, 3 on same layout."""
        bound, H, lattice = self._build_circuit_and_hamiltonian()
        backend = self._get_fake_backend()

        # Get a good layout (lowest CES from level 2 as baseline)
        adj = build_adjacency(backend)
        candidates = find_layouts_bfs(adj, N_QUBITS, n_candidates=40, seed=SEED)
        layout_sel = select_layouts_low_ces(
            bound, backend, candidates, n_select=1, optimization_level=2, max_ces=0.5
        )
        layout = layout_sel.layouts[0]
        logger.info(f"Selected layout: {layout}")

        # Transpile at each level with SAME layout
        results = []
        for level in OPT_LEVELS:
            t0 = time.time()
            transpiled = self._transpile_with_options(
                bound, backend, layout, optimization_level=level
            )
            elapsed = time.time() - t0
            metrics = self._analyze_transpiled(transpiled, backend)

            result = {
                "optimization_level": level,
                "depth": metrics["depth"],
                "n_2q_gates": metrics["n_2q_gates"],
                "n_1q_gates": metrics["n_1q_gates"],
                "ces": metrics["ces"],
                "gate_counts": metrics["gate_counts"],
                "transpile_time_s": round(elapsed, 3),
            }
            results.append(result)
            logger.info(
                f"  Level {level}: depth={metrics['depth']}, "
                f"2Q={metrics['n_2q_gates']}, 1Q={metrics['n_1q_gates']}, "
                f"CES={metrics['ces']:.4f}, time={elapsed:.2f}s"
            )

        # Also report the logical circuit stats (pre-transpilation)
        logical_depth = bound.depth()
        logical_ops = bound.count_ops()
        logger.info(f"  Logical circuit: depth={logical_depth}, ops={dict(logical_ops)}")
        logger.info(f"  Decomposed (2x): depth={bound.decompose().decompose().depth()}")

        # Verdict: does level 3 reduce 2Q gates vs level 2?
        level2_2q = results[2]["n_2q_gates"]
        level3_2q = results[3]["n_2q_gates"]
        reduction = level2_2q - level3_2q
        logger.info(
            f"\n  Level 3 vs Level 2: ΔN_2Q = {reduction} "
            f"({'improvement' if reduction > 0 else 'no improvement'})"
        )

        return {
            "layout": layout,
            "per_level": results,
            "logical_depth": logical_depth,
            "logical_ops": dict(logical_ops),
            "decomposed_depth": bound.decompose().decompose().depth(),
            "level3_vs_level2_2q_reduction": reduction,
        }

    # ──────────────────────────────────────────────────────────────────
    # Section 2: Approximation Degree Sweep (Level 3 only)
    # ──────────────────────────────────────────────────────────────────

    def section_approx_degree(self) -> dict:
        """Sweep approximation_degree at level 3 to find fidelity/gate tradeoff."""

        bound, H, lattice = self._build_circuit_and_hamiltonian()
        backend = self._get_fake_backend()

        # Reuse same layout
        adj = build_adjacency(backend)
        candidates = find_layouts_bfs(adj, N_QUBITS, n_candidates=40, seed=SEED)
        layout_sel = select_layouts_low_ces(
            bound, backend, candidates, n_select=1, optimization_level=2, max_ces=0.5
        )
        layout = layout_sel.layouts[0]

        # Get exact transpilation as reference (level 3, approx=1.0)
        exact_transpiled = self._transpile_with_options(
            bound, backend, layout, optimization_level=3, approximation_degree=1.0
        )
        exact_metrics = self._analyze_transpiled(exact_transpiled, backend)

        results = []
        for approx_deg in APPROX_DEGREES:
            transpiled = self._transpile_with_options(
                bound,
                backend,
                layout,
                optimization_level=3,
                approximation_degree=approx_deg,
            )
            metrics = self._analyze_transpiled(transpiled, backend)

            # Compute unitary fidelity between exact and approximate
            # Only feasible for N=10 (2^10 = 1024 dim — borderline but doable)
            fidelity = None
            if approx_deg < 1.0 and N_QUBITS <= 10:
                try:
                    # Compare via state overlap on the test point
                    from qiskit.primitives import StatevectorEstimator

                    estimator = StatevectorEstimator()
                    # Exact energy from transpiled (exact approx)
                    H_mapped_exact = H.apply_layout(exact_transpiled.layout)
                    e_exact_t = float(
                        estimator.run([(exact_transpiled, H_mapped_exact)]).result()[0].data.evs
                    )
                    # Approx energy from transpiled
                    H_mapped_approx = H.apply_layout(transpiled.layout)
                    e_approx_t = float(
                        estimator.run([(transpiled, H_mapped_approx)]).result()[0].data.evs
                    )
                    # Energy difference as proxy for fidelity impact
                    fidelity = 1.0 - abs(e_exact_t - e_approx_t) / abs(e_exact_t)
                except Exception as e:
                    logger.warning(f"Fidelity computation failed: {e}")

            result = {
                "approximation_degree": approx_deg,
                "depth": metrics["depth"],
                "n_2q_gates": metrics["n_2q_gates"],
                "n_1q_gates": metrics["n_1q_gates"],
                "ces": metrics["ces"],
                "energy_fidelity": fidelity,
                "delta_2q_vs_exact": exact_metrics["n_2q_gates"] - metrics["n_2q_gates"],
            }
            results.append(result)
            logger.info(
                f"  approx={approx_deg:.2f}: 2Q={metrics['n_2q_gates']}, "
                f"depth={metrics['depth']}, CES={metrics['ces']:.4f}, "
                f"fid={fidelity}"
            )

        return {
            "layout": layout,
            "exact_reference": {
                "depth": exact_metrics["depth"],
                "n_2q_gates": exact_metrics["n_2q_gates"],
                "ces": exact_metrics["ces"],
            },
            "sweep_results": results,
        }

    # ──────────────────────────────────────────────────────────────────
    # Section 3: Noise Suppression Stack Comparison
    # ──────────────────────────────────────────────────────────────────

    def section_noise_stack(self) -> dict:
        """Compare noise suppression strategies on FakeTorino.

        Strategies compared:
        A) Bare noise — no mitigation at all
        B) PEA-ZNE only (current production config)
        C) Gate-folding ZNE (fallback)

        Note: DD, Twirling, and TREX are server-side options that only work
        on real IBM Runtime (not on local FakeTorino simulation). We document
        the configuration but can only test ZNE variants locally.
        """
        from qmbp_simulation.execution.noisy_utils import (
            run_gate_folding_zne,
            run_pea_zne,
        )

        bound, H, lattice = self._build_circuit_and_hamiltonian()
        backend = self._get_fake_backend()
        e_exact, gap = self._exact_ground_state(lattice)
        logger.info(f"  Exact: E={e_exact:.6f}, gap={gap:.6f}")

        # Get layout and transpile at level 2 (current production)
        adj = build_adjacency(backend)
        candidates = find_layouts_bfs(adj, N_QUBITS, n_candidates=40, seed=SEED)
        layout_sel = select_layouts_low_ces(
            bound, backend, candidates, n_select=1, optimization_level=2, max_ces=0.5
        )
        transpiled = layout_sel.transpiled_circuits[0]
        H_mapped = H.apply_layout(transpiled.layout)
        ces = layout_sel.ces_values[0]
        logger.info(f"  Layout CES={ces:.4f}")

        config = NoisyEstimatorConfig(shots=SHOTS, seed_simulator=SEED)

        # Strategy A: Bare noise (no mitigation)
        e_bare = noisy_estimate(transpiled, H_mapped, backend, config)
        de_bare = abs(e_bare - e_exact) / max(gap, 1e-10)
        logger.info(f"  A) Bare noise: E={e_bare:.6f}, ΔE/gap={de_bare:.4f}")

        # Strategy B: PEA-ZNE (current primary)
        try:
            pea_result = run_pea_zne(
                transpiled,
                H_mapped,
                backend,
                config,
                noise_factors=(1, 3, 5),
                extrapolator="linear",
            )
            e_pea = pea_result.extrapolated_value
            de_pea = abs(e_pea - e_exact) / max(gap, 1e-10)
            pea_r2 = pea_result.r_squared
            logger.info(f"  B) PEA-ZNE: E={e_pea:.6f}, ΔE/gap={de_pea:.4f}, R²={pea_r2:.4f}")
        except Exception as e:
            logger.warning(f"  B) PEA-ZNE failed: {e}")
            e_pea, de_pea, pea_r2 = None, None, None

        # Strategy C: Gate-folding ZNE (fallback)
        gf_result = run_gate_folding_zne(
            transpiled,
            H_mapped,
            backend,
            config,
            noise_factors=(1, 3, 5),
            extrapolator="linear",
        )
        e_gf = gf_result.extrapolated_value
        de_gf = abs(e_gf - e_exact) / max(gap, 1e-10)
        gf_r2 = gf_result.r_squared
        logger.info(f"  C) GF-ZNE: E={e_gf:.6f}, ΔE/gap={de_gf:.4f}, R²={gf_r2:.4f}")

        # Document the server-side options we'll use on real hardware
        hardware_noise_config = {
            "dynamical_decoupling": {
                "enable": True,
                "sequence_type": "XpXm",
                "note": "Suppresses coherent idle errors. Only on real QPU.",
            },
            "twirling": {
                "enable_gates": True,
                "enable_measure": True,
                "num_randomizations": 32,
                "shots_per_randomization": 128,
                "note": "Converts coherent → stochastic noise. Only on real QPU.",
            },
            "resilience": {
                "measure_mitigation": True,
                "zne_mitigation": True,
                "zne": {"amplifier": "pea", "noise_factors": [1, 3, 5]},
                "layer_noise_learning": {
                    "num_randomizations": 32,
                    "shots_per_randomization": 128,
                },
                "note": "TREX + PEA-ZNE. Only on real QPU.",
            },
        }

        return {
            "e_exact": e_exact,
            "gap": gap,
            "ces": ces,
            "strategies": {
                "A_bare": {"energy": e_bare, "de_gap": de_bare},
                "B_pea_zne": {
                    "energy": e_pea,
                    "de_gap": de_pea,
                    "r_squared": pea_r2,
                },
                "C_gf_zne": {
                    "energy": e_gf,
                    "de_gap": de_gf,
                    "r_squared": gf_r2,
                },
            },
            "hardware_noise_config_documentation": hardware_noise_config,
            "recommendation": (
                "PEA-ZNE remains primary. DD+Twirling+TREX only testable on "
                "real hardware. Level 3 transpilation should be tested in "
                "Section 4."
            ),
        }

    # ──────────────────────────────────────────────────────────────────
    # Section 4: Combined — Best Transpiler + PEA-ZNE
    # ──────────────────────────────────────────────────────────────────

    def section_combined(self) -> dict:
        """Compare level 2 vs level 3 transpilation WITH PEA-ZNE.

        This is the key comparison: does level 3 transpilation give better
        mitigated energies when combined with our ZNE strategy?
        """
        from qmbp_simulation.execution.noisy_utils import (
            run_gate_folding_zne,
            run_pea_zne,
        )

        bound, H, lattice = self._build_circuit_and_hamiltonian()
        backend = self._get_fake_backend()
        e_exact, gap = self._exact_ground_state(lattice)

        adj = build_adjacency(backend)
        candidates = find_layouts_bfs(adj, N_QUBITS, n_candidates=40, seed=SEED)
        config = NoisyEstimatorConfig(shots=SHOTS, seed_simulator=SEED)

        results_by_level = {}

        for level in [2, 3]:
            # Select layouts at this optimization level
            layout_sel = select_layouts_low_ces(
                bound,
                backend,
                candidates,
                n_select=1,
                optimization_level=level,
                max_ces=0.5,
            )
            transpiled = layout_sel.transpiled_circuits[0]
            H_mapped = H.apply_layout(transpiled.layout)
            ces = layout_sel.ces_values[0]

            metrics = self._analyze_transpiled(transpiled, backend)

            # Noiseless energy (via StatevectorEstimator on the transpiled circuit)
            from qiskit.primitives import StatevectorEstimator

            sv_est = StatevectorEstimator()
            e_noiseless = float(sv_est.run([(transpiled, H_mapped)]).result()[0].data.evs)
            de_noiseless = abs(e_noiseless - e_exact) / max(gap, 1e-10)

            # Bare noisy
            e_noisy = noisy_estimate(transpiled, H_mapped, backend, config)
            de_noisy = abs(e_noisy - e_exact) / max(gap, 1e-10)

            # PEA-ZNE
            try:
                pea = run_pea_zne(
                    transpiled,
                    H_mapped,
                    backend,
                    config,
                    noise_factors=(1, 3, 5),
                )
                e_pea = pea.extrapolated_value
                de_pea = abs(e_pea - e_exact) / max(gap, 1e-10)
                pea_r2 = pea.r_squared
                pea_gain = (de_noisy - de_pea) / de_noisy * 100 if de_noisy > 0 else 0
            except Exception as e:
                logger.warning(f"  PEA failed at level {level}: {e}")
                e_pea, de_pea, pea_r2, pea_gain = None, None, None, None

            # Gate-folding ZNE (for comparison)
            gf = run_gate_folding_zne(
                transpiled,
                H_mapped,
                backend,
                config,
                noise_factors=(1, 3, 5),
            )
            e_gf = gf.extrapolated_value
            de_gf = abs(e_gf - e_exact) / max(gap, 1e-10)
            gf_r2 = gf.r_squared

            level_result = {
                "optimization_level": level,
                "depth": metrics["depth"],
                "n_2q_gates": metrics["n_2q_gates"],
                "ces": ces,
                "e_noiseless": e_noiseless,
                "de_noiseless": de_noiseless,
                "e_noisy": e_noisy,
                "de_noisy": de_noisy,
                "e_pea_zne": e_pea,
                "de_pea_zne": de_pea,
                "pea_r2": pea_r2,
                "pea_gain_pct": pea_gain,
                "e_gf_zne": e_gf,
                "de_gf_zne": de_gf,
                "gf_r2": gf_r2,
            }
            results_by_level[f"level_{level}"] = level_result
            logger.info(
                f"  Level {level}: 2Q={metrics['n_2q_gates']}, "
                f"depth={metrics['depth']}, CES={ces:.4f}"
            )
            logger.info(f"    Noiseless: ΔE/gap={de_noiseless:.6f}")
            logger.info(f"    Noisy raw: ΔE/gap={de_noisy:.4f}")
            logger.info(f"    PEA-ZNE:   ΔE/gap={de_pea}, R²={pea_r2}, gain={pea_gain}%")
            logger.info(f"    GF-ZNE:    ΔE/gap={de_gf:.4f}, R²={gf_r2:.4f}")

        # Final comparison
        l2 = results_by_level["level_2"]
        l3 = results_by_level["level_3"]
        level3_better = (
            l3.get("de_pea_zne") is not None
            and l2.get("de_pea_zne") is not None
            and l3["de_pea_zne"] < l2["de_pea_zne"]
        )

        return {
            "e_exact": e_exact,
            "gap": gap,
            "comparison": results_by_level,
            "level3_better_than_level2": level3_better,
            "recommendation": (
                "Use level 3 for hardware"
                if level3_better
                else "Keep level 2 (level 3 provides no benefit for HVA)"
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    TranspilerExplorationRunner.main()
