#!/usr/bin/env python3
"""Gate-Folding ZNE vs CES-ZNE — Direct Comparison Experiment.

Compares two ZNE strategies on the same circuit/layout:
  1. CES-ZNE (inhomogeneous): 3 layouts with CES spread → linear extrap to CES=0
  2. Gate-Folding ZNE (GF-ZNE): single layout, noise factors [1,3,5] → extrap to nf=0

Hypothesis:
  On heavy_hex N=10 p=1, GF-ZNE achieves:
  - R² > 0.9 (vs CES-ZNE R²≈0.04 due to uniform CES)
  - ZNE gain > 40% (vs CES-ZNE gain ≈ 0% on heavy_hex)
  - ΔE/gap reduction > 30% relative to noisy-raw

  On chain_1d N=6 p=1, both methods should work (CES spread exists):
  - Both R² > 0.9
  - GF-ZNE gain ≥ CES-ZNE gain (or within 10%)

Sections:
  1. Noiseless baseline — VQE optimization to get θ_opt
  2. CES-ZNE — standard 3-layout inhomogeneous extrapolation
  3. Gate-Folding ZNE — single-layout with noise amplification [1,3,5]
  4. Comparison — direct metrics side-by-side

Usage:
    python scripts/experiment_runners/run_gf_zne_comparison.py
    python scripts/experiment_runners/run_gf_zne_comparison.py --topology heavy_hex --n-qubits 10
    python scripts/experiment_runners/run_gf_zne_comparison.py --topology chain_1d --n-qubits 6
    python scripts/experiment_runners/run_gf_zne_comparison.py --section 1 2 3
    python scripts/experiment_runners/run_gf_zne_comparison.py --dry-run
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
SEEDS = [42, 43, 44]

# h-values inside valid regime (descending)
H_TEST_VALUES = [2.5, 2.0, 1.75]

# ZNE configuration
ZNE_N_LAYOUTS = 3
ZNE_SHOTS = 16384
N_CANDIDATE_LAYOUTS = 20

# Gate-folding noise factors
GF_NOISE_FACTORS = (1, 3, 5)

# VQE
VQE_RESTARTS = 3
VQE_MAXITER = 500


# ═══════════════════════════════════════════════════════════════════════════════
# Runner Implementation
# ═══════════════════════════════════════════════════════════════════════════════


class GFZNEComparisonRunner(ValidationRunner):
    """Direct comparison of CES-ZNE vs Gate-Folding ZNE."""

    runner_id = "gf_zne_comparison"
    experiment_id = "GF_ZNE_CMP"
    description = "Gate-Folding ZNE vs CES-ZNE Direct Comparison"
    hypothesis = (
        "Gate-folding ZNE achieves R²>0.9 and gain>40% on topologies where "
        "CES-ZNE fails due to uniform layout noise (heavy_hex), and performs "
        "comparably on chain_1d where CES-ZNE already works."
    )

    @classmethod
    def _add_custom_args(cls, parser) -> None:
        """Add topology/size args."""
        parser.add_argument(
            "--topology",
            default=DEFAULT_TOPOLOGY,
            choices=["chain_1d", "ladder", "triangular", "heavy_hex"],
            help="Lattice topology (default: chain_1d)",
        )
        parser.add_argument(
            "--n-qubits",
            type=int,
            default=DEFAULT_N_QUBITS,
            help="Number of qubits (default: 6)",
        )
        parser.add_argument(
            "--p-layers",
            type=int,
            default=DEFAULT_P_LAYERS,
            help="HVA layers (default: 1)",
        )
        parser.add_argument(
            "--extrapolator",
            default="linear",
            choices=["linear", "exponential"],
            help="GF-ZNE extrapolation method (default: linear)",
        )

    def define_sections(self) -> list[Section]:
        """Define experiment sections."""
        return [
            Section(
                id=1,
                name="Noiseless Baseline",
                fn=self._run_noiseless,
                hypothesis="VQE converges with ΔE/gap < 50% at h≥1.75 (p=1 limit)",
            ),
            Section(
                id=2,
                name="CES-ZNE (Inhomogeneous)",
                fn=self._run_ces_zne,
                hypothesis="CES-ZNE provides R²>0.9 (topology-dependent)",
            ),
            Section(
                id=3,
                name="Gate-Folding ZNE",
                fn=self._run_gf_zne,
                hypothesis="GF-ZNE achieves consistent positive gain with R²>0.9",
            ),
            Section(
                id=4,
                name="Comparison Summary",
                fn=self._run_comparison,
                hypothesis="GF-ZNE gain ≥ CES-ZNE gain on average",
            ),
        ]

    def build_config(self) -> dict:
        """Build config for result envelope (digest-compatible)."""
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "category": "GF",
            "description": self.description,
            "hypothesis": self.hypothesis,
            "system": {
                "n_qubits": getattr(self, "n_qubits", DEFAULT_N_QUBITS),
                "p_layers": getattr(self, "p_layers", DEFAULT_P_LAYERS),
                "topology": getattr(self, "topology", DEFAULT_TOPOLOGY),
                "model": "tfim",
            },
            "zne": {
                "noise_factors": list(GF_NOISE_FACTORS),
                "n_layouts_ces": ZNE_N_LAYOUTS,
                "shots": ZNE_SHOTS,
                "extrapolator": getattr(self, "extrapolator", "linear"),
            },
            "seeds": SEEDS,
        }

    def setup(self) -> None:
        """Initialize backends and shared state."""
        from qiskit_ibm_runtime.fake_provider import FakeTorino
        from scipy.optimize import minimize

        from qmbp_simulation import HamiltonianBuilder, make_lattice  # noqa: F401
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.execution import NoiselessBackend
        from qmbp_simulation.execution.noisy_utils import (
            NoisyEstimatorConfig,
            build_adjacency,
            find_layouts_bfs,
            noisy_estimate,
            run_gate_folding_zne,
            run_zne_deployment,
            select_layouts_by_circuit_ces,
            select_layouts_low_ces,
        )

        args = self._args
        self.topology = getattr(args, "topology", DEFAULT_TOPOLOGY)
        self.n_qubits = getattr(args, "n_qubits", DEFAULT_N_QUBITS)
        self.p_layers = getattr(args, "p_layers", DEFAULT_P_LAYERS)
        self.extrapolator = getattr(args, "extrapolator", "linear")

        self.builder = HamiltonianBuilder()
        self.hva = HVACircuitBuilder()
        self.noiseless = NoiselessBackend()
        self.fake_backend = FakeTorino()
        self.noisy_config = NoisyEstimatorConfig(shots=ZNE_SHOTS, seed_simulator=42)
        self.make_lattice = make_lattice

        # Store callables
        self.minimize = minimize
        self.noisy_estimate = noisy_estimate
        self.run_zne_deployment = run_zne_deployment
        self.run_gate_folding_zne = run_gate_folding_zne
        self.select_layouts_by_circuit_ces = select_layouts_by_circuit_ces
        self.select_layouts_low_ces = select_layouts_low_ces

        # Find candidate layouts
        adjacency = build_adjacency(self.fake_backend)
        self.candidate_layouts = find_layouts_bfs(
            adjacency, self.n_qubits, n_candidates=N_CANDIDATE_LAYOUTS
        )
        logger.info(
            f"[setup] topology={self.topology}, N={self.n_qubits}, p={self.p_layers}, "
            f"candidates={len(self.candidate_layouts)}"
        )

        # Shared storage across sections
        self._noiseless_data: list[dict] = []
        self._ces_zne_data: list[dict] = []
        self._gf_zne_data: list[dict] = []

    def _run_noiseless(self) -> dict:
        """Section 1: Noiseless VQE baseline to get θ_opt at each h."""
        logger.info("=" * 65)
        logger.info(
            f"SECTION 1: Noiseless Baseline ({self.topology}, N={self.n_qubits}, p={self.p_layers})"
        )
        logger.info("=" * 65)

        lattice_ref = self.make_lattice(self.topology, self.n_qubits, J=1.0, h=max(H_TEST_VALUES))
        circuit, _ = self.hva.create(self.n_qubits, self.p_layers, lattice_ref)
        n_params = circuit.num_parameters

        rng = np.random.default_rng(42)
        prev_theta = rng.uniform(-0.01, 0.01, n_params)
        results = []

        for h in sorted(H_TEST_VALUES, reverse=True):
            lattice = self.make_lattice(self.topology, self.n_qubits, J=1.0, h=h)
            H = self.builder.build(lattice)

            # Exact energy + gap
            H_mat = H.to_matrix()
            if hasattr(H_mat, "toarray"):
                H_mat = H_mat.toarray()
            evals = np.sort(np.linalg.eigvalsh(H_mat))
            e_exact = float(evals[0])
            gap = float(evals[1] - evals[0])

            # VQE optimization (descending warm-start)
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
            point = {
                "h": h,
                "e_exact": e_exact,
                "gap": gap,
                "e_noiseless": best_energy,
                "de_gap_noiseless": de_gap,
                "theta_opt": best_theta.tolist(),
            }
            results.append(point)
            logger.info(
                f"  h={h:.2f}: E_exact={e_exact:.6f}, E_vqe={best_energy:.6f}, ΔE/gap={de_gap:.4f}"
            )

        self._noiseless_data = results
        self._circuit = circuit
        return {
            "topology": self.topology,
            "n_qubits": self.n_qubits,
            "p_layers": self.p_layers,
            "n_params": n_params,
            "results": results,
        }

    def _run_ces_zne(self) -> dict:
        """Section 2: CES-ZNE (inhomogeneous layout extrapolation)."""
        logger.info("")
        logger.info("=" * 65)
        logger.info("SECTION 2: CES-ZNE (Inhomogeneous Layout Extrapolation)")
        logger.info("=" * 65)

        if not self._noiseless_data:
            raise RuntimeError("Section 1 must run before Section 2")

        results = []
        for point in self._noiseless_data:
            h = point["h"]
            theta_opt = np.array(point["theta_opt"])
            e_exact = point["e_exact"]
            gap = point["gap"]

            lattice = self.make_lattice(self.topology, self.n_qubits, J=1.0, h=h)
            H = self.builder.build(lattice)
            bound_circuit = self._circuit.assign_parameters(theta_opt)

            # Layout selection: spread-maximizing for CES-ZNE leverage
            layout_sel = self.select_layouts_by_circuit_ces(
                bound_circuit,
                self.fake_backend,
                self.candidate_layouts,
                n_select=ZNE_N_LAYOUTS,
            )
            ces_values = [float(c) for c in layout_sel.ces_values]
            logger.info(f"  h={h:.2f}: CES values = {[f'{c:.4f}' for c in ces_values]}")

            # Noisy-raw (best layout = lowest CES)
            best_layout_idx = int(np.argmin(ces_values))
            transpiled_raw = layout_sel.transpiled_circuits[best_layout_idx]
            H_mapped = H.apply_layout(transpiled_raw.layout)
            e_noisy_raw = self.noisy_estimate(
                transpiled_raw,
                H_mapped,
                self.fake_backend,
                self.noisy_config,
                seed_offset=0,
            )

            # CES-ZNE (3 layouts)
            zne_result = self.run_zne_deployment(
                bound_circuit,
                H,
                self.fake_backend,
                layout_sel,
                self.noisy_config,
                self.n_qubits,
            )
            e_zne = zne_result.energy_zne.extrapolated_value
            r2 = zne_result.energy_zne.r_squared

            de_noisy = abs(e_noisy_raw - e_exact) / max(gap, 1e-10)
            de_zne = abs(e_zne - e_exact) / max(gap, 1e-10)
            gain = (de_noisy - de_zne) / max(de_noisy, 1e-10)

            data = {
                "h": h,
                "e_noisy_raw": e_noisy_raw,
                "e_ces_zne": e_zne,
                "de_noisy_raw": de_noisy,
                "de_ces_zne": de_zne,
                "ces_zne_gain": gain,
                "ces_zne_r2": r2,
                "ces_values": ces_values,
                "ces_spread": float(np.std(ces_values)),
            }
            results.append(data)
            logger.info(
                f"  h={h:.2f}: noisy={de_noisy:.4f}, CES-ZNE={de_zne:.4f}, "
                f"gain={gain:+.1%}, R²={r2:.4f}, CES_spread={np.std(ces_values):.4f}"
            )

        self._ces_zne_data = results
        mean_r2 = float(np.mean([r["ces_zne_r2"] for r in results]))
        mean_gain = float(np.mean([r["ces_zne_gain"] for r in results]))

        logger.info(f"\n  CES-ZNE Summary: mean_R²={mean_r2:.4f}, mean_gain={mean_gain:+.1%}")
        return {"results": results, "mean_r2": mean_r2, "mean_gain": mean_gain}

    def _run_gf_zne(self) -> dict:
        """Section 3: Gate-Folding ZNE (noise amplification via gate repetition)."""
        logger.info("")
        logger.info("=" * 65)
        logger.info("SECTION 3: Gate-Folding ZNE (Digital Gate Folding)")
        logger.info("=" * 65)
        logger.info(f"  noise_factors={GF_NOISE_FACTORS}, extrapolator={self.extrapolator}")

        if not self._noiseless_data:
            raise RuntimeError("Section 1 must run before Section 3")

        results = []
        for point in self._noiseless_data:
            h = point["h"]
            theta_opt = np.array(point["theta_opt"])
            e_exact = point["e_exact"]
            gap = point["gap"]

            lattice = self.make_lattice(self.topology, self.n_qubits, J=1.0, h=h)
            H = self.builder.build(lattice)
            bound_circuit = self._circuit.assign_parameters(theta_opt)

            # Use lowest-CES layout (single best layout for GF-ZNE)
            layout_sel = self.select_layouts_low_ces(
                bound_circuit,
                self.fake_backend,
                self.candidate_layouts,
                n_select=1,
                optimization_level=self.noisy_config.optimization_level,
                max_ces=0.5,
            )
            transpiled = layout_sel.transpiled_circuits[0]
            ces = float(layout_sel.ces_values[0])
            H_mapped = H.apply_layout(transpiled.layout)

            logger.info(f"  h={h:.2f}: layout CES={ces:.4f}, depth={transpiled.depth()}")

            # Noisy-raw (same layout, no folding = noise_factor=1)
            e_noisy_raw = self.noisy_estimate(
                transpiled,
                H_mapped,
                self.fake_backend,
                self.noisy_config,
                seed_offset=0,
            )

            # Gate-folding ZNE
            gf_result = self.run_gate_folding_zne(
                transpiled,
                H_mapped,
                self.fake_backend,
                self.noisy_config,
                noise_factors=GF_NOISE_FACTORS,
                extrapolator=self.extrapolator,
                seed_offset=500,
            )
            e_gf_zne = gf_result.extrapolated_value
            r2 = gf_result.r_squared

            de_noisy = abs(e_noisy_raw - e_exact) / max(gap, 1e-10)
            de_gf_zne = abs(e_gf_zne - e_exact) / max(gap, 1e-10)
            gain = (de_noisy - de_gf_zne) / max(de_noisy, 1e-10)

            data = {
                "h": h,
                "e_noisy_raw": e_noisy_raw,
                "e_gf_zne": e_gf_zne,
                "de_noisy_raw": de_noisy,
                "de_gf_zne": de_gf_zne,
                "gf_zne_gain": gain,
                "gf_zne_r2": r2,
                "gf_zne_slope": gf_result.slope,
                "noise_factors": gf_result.noise_factors,
                "measured_values": gf_result.measured_values,
                "layout_ces": ces,
                "extrapolator": gf_result.extrapolator,
            }
            results.append(data)
            logger.info(
                f"  h={h:.2f}: noisy={de_noisy:.4f}, GF-ZNE={de_gf_zne:.4f}, "
                f"gain={gain:+.1%}, R²={r2:.4f}, slope={gf_result.slope:.6f}"
            )
            logger.info(
                f"           measurements: {[f'{v:.6f}' for v in gf_result.measured_values]}"
            )

        self._gf_zne_data = results
        mean_r2 = float(np.mean([r["gf_zne_r2"] for r in results]))
        mean_gain = float(np.mean([r["gf_zne_gain"] for r in results]))

        logger.info(f"\n  GF-ZNE Summary: mean_R²={mean_r2:.4f}, mean_gain={mean_gain:+.1%}")
        return {"results": results, "mean_r2": mean_r2, "mean_gain": mean_gain}

    def _run_comparison(self) -> dict:
        """Section 4: Side-by-side comparison and verdict."""
        logger.info("")
        logger.info("=" * 65)
        logger.info("SECTION 4: CES-ZNE vs GF-ZNE — Direct Comparison")
        logger.info("=" * 65)

        if not self._ces_zne_data or not self._gf_zne_data:
            raise RuntimeError("Sections 2 and 3 must run before Section 4")

        comparison = []
        for ces_d, gf_d, nl_d in zip(
            self._ces_zne_data, self._gf_zne_data, self._noiseless_data, strict=True
        ):
            h = ces_d["h"]
            row = {
                "h": h,
                "e_exact": nl_d["e_exact"],
                "gap": nl_d["gap"],
                "de_noiseless": nl_d["de_gap_noiseless"],
                "de_noisy_raw": ces_d["de_noisy_raw"],
                # CES-ZNE
                "de_ces_zne": ces_d["de_ces_zne"],
                "ces_zne_r2": ces_d["ces_zne_r2"],
                "ces_zne_gain": ces_d["ces_zne_gain"],
                "ces_spread": ces_d["ces_spread"],
                # GF-ZNE
                "de_gf_zne": gf_d["de_gf_zne"],
                "gf_zne_r2": gf_d["gf_zne_r2"],
                "gf_zne_gain": gf_d["gf_zne_gain"],
                # Winner
                "gf_wins": gf_d["de_gf_zne"] < ces_d["de_ces_zne"],
                "best_method": "GF-ZNE" if gf_d["de_gf_zne"] < ces_d["de_ces_zne"] else "CES-ZNE",
                "improvement_over_ces": (
                    (ces_d["de_ces_zne"] - gf_d["de_gf_zne"]) / max(ces_d["de_ces_zne"], 1e-10)
                ),
            }
            comparison.append(row)

        # Print comparison table
        logger.info("")
        logger.info(
            f"  {'h':>5} | {'Noisy':>8} | {'CES-ZNE':>8} {'R²':>5} {'Gain':>7} | "
            f"{'GF-ZNE':>8} {'R²':>5} {'Gain':>7} | Winner"
        )
        logger.info("  " + "-" * 80)
        for row in comparison:
            logger.info(
                f"  {row['h']:5.2f} | {row['de_noisy_raw']:8.4f} | "
                f"{row['de_ces_zne']:8.4f} {row['ces_zne_r2']:5.3f} {row['ces_zne_gain']:+6.1%} | "
                f"{row['de_gf_zne']:8.4f} {row['gf_zne_r2']:5.3f} {row['gf_zne_gain']:+6.1%} | "
                f"{row['best_method']}"
            )

        # Aggregate
        n_gf_wins = sum(1 for r in comparison if r["gf_wins"])
        mean_ces_gain = float(np.mean([r["ces_zne_gain"] for r in comparison]))
        mean_gf_gain = float(np.mean([r["gf_zne_gain"] for r in comparison]))
        mean_ces_r2 = float(np.mean([r["ces_zne_r2"] for r in comparison]))
        mean_gf_r2 = float(np.mean([r["gf_zne_r2"] for r in comparison]))

        summary = {
            "topology": self.topology,
            "n_qubits": self.n_qubits,
            "p_layers": self.p_layers,
            "n_points": len(comparison),
            "gf_wins": n_gf_wins,
            "ces_wins": len(comparison) - n_gf_wins,
            "mean_ces_zne_gain": mean_ces_gain,
            "mean_gf_zne_gain": mean_gf_gain,
            "mean_ces_zne_r2": mean_ces_r2,
            "mean_gf_zne_r2": mean_gf_r2,
            "gf_r2_above_09": mean_gf_r2 > 0.9,
            "gf_gain_above_40pct": mean_gf_gain > 0.4,
            "gf_better_overall": mean_gf_gain > mean_ces_gain,
        }

        logger.info("")
        logger.info("  ─── VERDICT ───")
        logger.info(f"  GF-ZNE wins: {n_gf_wins}/{len(comparison)} h-points")
        logger.info(f"  Mean CES-ZNE gain: {mean_ces_gain:+.1%} (R²={mean_ces_r2:.4f})")
        logger.info(f"  Mean GF-ZNE gain:  {mean_gf_gain:+.1%} (R²={mean_gf_r2:.4f})")
        logger.info(
            f"  GF-ZNE superior: {summary['gf_better_overall']} "
            f"(Δgain = {mean_gf_gain - mean_ces_gain:+.1%})"
        )

        return {"comparison": comparison, "summary": summary}


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    GFZNEComparisonRunner.main()
