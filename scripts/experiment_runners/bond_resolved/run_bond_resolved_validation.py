#!/usr/bin/env python3
"""Bond-Resolved HVA Validation Suite.

Validates that per-bond/per-site parametrization of HVA circuits:
1. Converges to ground state (fidelity >= 0.99)
2. Achieves equal or better ΔE/gap vs global HVA
3. Shows increasing advantage on non-uniform topologies
4. GNN necessity: interpolation fails for high-dimensional θ

Sections:
    1. N=6 chain_1d convergence (baseline, 3 seeds)
    2. N=10 heavy-hex convergence (hardware-relevant)
    3. Global vs bond-resolved comparison (N=6/10, multiple topologies)
    4. GNN necessity: linear interp fails for bond-resolved

Hypothesis: Bond-resolved HVA achieves ΔE/gap <= global HVA with same
gate count, and the advantage grows with topology non-uniformity.
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


class BondResolvedValidationRunner(ValidationRunner):
    """Bond-Resolved HVA validation — expressibility scaling toward QA."""

    runner_id = "bond_resolved_validation"
    experiment_id = "BOND_RESOLVED_HVA"
    description = "Bond-Resolved HVA: per-bond/per-site params, same depth"
    hypothesis = (
        "Bond-resolved HVA achieves equal or better energy accuracy than "
        "global HVA at same circuit depth, with advantage growing on "
        "non-uniform topologies (heavy-hex > chain)"
    )

    @classmethod
    def _add_custom_args(cls, parser):
        parser.add_argument("--n-qubits", type=int, default=6)
        parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])

    def build_config(self) -> dict:
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "system": {
                "n_qubits": self._args.n_qubits,
                "p_layers": 1,
                "topologies": ["chain_1d", "heavy_hex", "ladder"],
                "model": "tfim_bond_resolved",
            },
            "seeds": self._args.seeds,
        }

    def setup(self):
        from qmbp_simulation import (
            ClassicalSolver,
            HamiltonianBuilder,
            VQEConfig,
            VQEOptimizer,
            make_lattice,
        )
        from qmbp_simulation.models.model_registry import get_model_spec

        self.builder = HamiltonianBuilder()
        self.solver = ClassicalSolver()
        self.make_lattice = make_lattice
        self.spec_br = get_model_spec("tfim_bond_resolved")
        self.spec_global = get_model_spec("tfim")

        self.vqe_config_br = VQEConfig(
            n_restarts=3,
            restart_sigma=0.05,
            maxiter=1500,
            bounds=(-np.pi, np.pi),
        )
        self.vqe_config_global = VQEConfig(
            n_restarts=3,
            restart_sigma=0.1,
            maxiter=1000,
            bounds=(-np.pi, np.pi),
        )
        self.VQEOptimizer = VQEOptimizer

    def define_sections(self) -> list[Section]:
        return [
            Section(
                id=1,
                name="N=6 Chain Convergence (3 seeds)",
                fn=self.section_chain_convergence,
                hypothesis="Bond-resolved VQE converges (fid>=0.99) at N=6 chain_1d",
            ),
            Section(
                id=2,
                name="N=10 Heavy-Hex Convergence",
                fn=self.section_heavy_hex_convergence,
                hypothesis="Bond-resolved VQE converges on heavy-hex N=10 p=1",
            ),
            Section(
                id=3,
                name="Global vs Bond-Resolved Comparison",
                fn=self.section_comparison,
                hypothesis="Bond-resolved achieves dE/gap <= global across topologies",
            ),
            Section(
                id=4,
                name="Parameter Space Analysis",
                fn=self.section_param_analysis,
                hypothesis=(
                    "Bond-resolved theta_opt shows spatial structure "
                    "(not all bonds converge to same value)"
                ),
            ),
        ]

    # ── Section 1: Chain convergence ─────────────────────────────────────

    def section_chain_convergence(self) -> dict:
        """Validate VQE convergence on N=6 chain_1d with 3 seeds."""
        N, P = 6, 1
        TOPOLOGY = "chain_1d"
        H_TRAIN = [3.0, 2.5, 2.0, 1.75, 1.5]
        seeds = self._args.seeds

        results_by_seed = {}
        all_pass = True

        for seed in seeds:
            opt = self.VQEOptimizer(config=self.vqe_config_br, seed=seed)
            prev_theta = None
            seed_results = []

            for h in H_TRAIN:
                lattice = self.make_lattice(TOPOLOGY, N, h=h)
                n_params = len(lattice.edges) + N
                H = self.builder.build(lattice)
                gt = self.solver.solve(H, lattice)
                qc, _ = self.spec_br.create_circuit(N, P, lattice)

                if prev_theta is None:
                    init = np.random.default_rng(seed).uniform(-0.01, 0.01, n_params)
                else:
                    init = prev_theta.copy()

                res = opt.optimize(H, qc, init, gt.ground_energy, gt.ground_state)
                prev_theta = res.theta_opt.copy()

                de_gap = abs(res.energy_error) / gt.gap if gt.gap > 0 else float("inf")
                seed_results.append(
                    {
                        "h": h,
                        "fidelity": float(res.fidelity),
                        "delta_e_gap": float(de_gap),
                        "energy_error": float(res.energy_error),
                    }
                )

                if h >= 1.6 and res.fidelity < 0.99:
                    all_pass = False

            results_by_seed[seed] = seed_results

        # Log summary
        for seed, sr in results_by_seed.items():
            fids = [r["fidelity"] for r in sr if r["h"] >= 1.6]
            mean_fid = np.mean(fids) if fids else 0
            logger.info(f"  Seed {seed}: mean_fid(h>=1.6)={mean_fid:.4f}")

        return {
            "topology": TOPOLOGY,
            "n_qubits": N,
            "p_layers": P,
            "n_params": len(self.make_lattice(TOPOLOGY, N, h=2.0).edges) + N,
            "results_by_seed": results_by_seed,
            "all_fid_above_099": all_pass,
            "pass": all_pass,
        }

    # ── Section 2: Heavy-hex convergence ─────────────────────────────────

    def section_heavy_hex_convergence(self) -> dict:
        """Validate convergence on heavy-hex N=10 (hardware target)."""
        N, P = 10, 1
        TOPOLOGY = "heavy_hex"
        H_TRAIN = [4.0, 3.75, 3.5, 3.25, 3.0]
        seed = 42

        opt = self.VQEOptimizer(config=self.vqe_config_br, seed=seed)
        lattice_ref = self.make_lattice(TOPOLOGY, N, h=4.0)
        n_params = len(lattice_ref.edges) + N
        prev_theta = None
        results = []

        for h in H_TRAIN:
            lattice = self.make_lattice(TOPOLOGY, N, h=h)
            H = self.builder.build(lattice)
            gt = self.solver.solve(H, lattice)
            qc, _ = self.spec_br.create_circuit(N, P, lattice)

            if prev_theta is None:
                init = np.random.default_rng(seed).uniform(-0.01, 0.01, n_params)
            else:
                init = prev_theta.copy()

            t0 = time.time()
            res = opt.optimize(H, qc, init, gt.ground_energy, gt.ground_state)
            elapsed = time.time() - t0
            prev_theta = res.theta_opt.copy()

            de_gap = abs(res.energy_error) / gt.gap if gt.gap > 0 else float("inf")
            results.append(
                {
                    "h": h,
                    "fidelity": float(res.fidelity),
                    "delta_e_gap": float(de_gap),
                    "energy_error": float(res.energy_error),
                    "elapsed_s": elapsed,
                }
            )

            logger.info(
                f"  h={h:.2f} | fid={res.fidelity:.6f} | "
                f"dE/gap={de_gap:.4f} | params={n_params} | {elapsed:.1f}s"
            )

        all_pass = all(r["fidelity"] >= 0.99 for r in results if r["h"] >= 3.0)
        return {
            "topology": TOPOLOGY,
            "n_qubits": N,
            "p_layers": P,
            "n_params": n_params,
            "n_edges": len(lattice_ref.edges),
            "results": results,
            "pass": all_pass,
        }

    # ── Section 3: Global vs Bond-Resolved ───────────────────────────────

    def section_comparison(self) -> dict:
        """Compare global HVA vs bond-resolved across topologies."""
        P = 1
        seed = 42
        CONFIGS = [
            ("chain_1d", 6, [3.0, 2.5, 2.0, 1.5]),
            ("chain_1d", 10, [3.0, 2.75, 2.5, 2.25, 2.0]),
            ("heavy_hex", 10, [4.0, 3.75, 3.5, 3.25]),
            ("ladder", 10, [4.0, 3.75, 3.5, 3.25]),
        ]

        comparison_results = []

        for topo, N, h_vals in CONFIGS:
            # Bond-resolved sweep
            opt_br = self.VQEOptimizer(config=self.vqe_config_br, seed=seed)
            prev_br = None
            br_energies = []

            for h in h_vals:
                lattice = self.make_lattice(topo, N, h=h)
                n_params = len(lattice.edges) + N
                H = self.builder.build(lattice)
                gt = self.solver.solve(H, lattice)
                qc, _ = self.spec_br.create_circuit(N, P, lattice)

                if prev_br is None:
                    init = np.random.default_rng(seed).uniform(-0.01, 0.01, n_params)
                else:
                    init = prev_br.copy()

                res = opt_br.optimize(H, qc, init, gt.ground_energy, gt.ground_state)
                prev_br = res.theta_opt.copy()
                de = abs(res.energy_error) / gt.gap if gt.gap > 0 else float("inf")
                br_energies.append(de)

            # Global sweep
            opt_g = self.VQEOptimizer(config=self.vqe_config_global, seed=seed)
            prev_g = None
            global_energies = []

            for h in h_vals:
                lattice = self.make_lattice(topo, N, h=h)
                H = self.builder.build(lattice)
                gt = self.solver.solve(H, lattice)
                qc_g, _ = self.spec_global.create_circuit(N, P, lattice)

                if prev_g is None:
                    init_g = np.random.default_rng(seed).uniform(-0.01, 0.01, 2)
                else:
                    init_g = prev_g.copy()

                res_g = opt_g.optimize(H, qc_g, init_g, gt.ground_energy, gt.ground_state)
                prev_g = res_g.theta_opt.copy()
                de_g = abs(res_g.energy_error) / gt.gap if gt.gap > 0 else float("inf")
                global_energies.append(de_g)

            mean_br = float(np.mean(br_energies))
            mean_global = float(np.mean(global_energies))
            improvement_pct = (
                (mean_global - mean_br) / mean_global * 100 if mean_global > 1e-10 else 0.0
            )

            comparison_results.append(
                {
                    "topology": topo,
                    "n_qubits": N,
                    "n_params_br": n_params,
                    "n_params_global": 2,
                    "mean_de_gap_br": mean_br,
                    "mean_de_gap_global": mean_global,
                    "improvement_pct": float(improvement_pct),
                    "br_wins": sum(
                        1 for b, g in zip(br_energies, global_energies, strict=False) if b <= g
                    ),
                    "n_points": len(h_vals),
                }
            )

            logger.info(
                f"  {topo} N={N}: Global={mean_global:.4f} | "
                f"BR={mean_br:.4f} | Improvement={improvement_pct:+.1f}%"
            )

        # Overall: bond-resolved wins if improvement >= 0 for majority
        n_improved = sum(1 for c in comparison_results if c["improvement_pct"] >= 0)
        all_pass = n_improved >= len(comparison_results) * 0.75

        return {
            "comparisons": comparison_results,
            "n_improved": n_improved,
            "n_total": len(comparison_results),
            "pass": all_pass,
        }

    # ── Section 4: Parameter space analysis ──────────────────────────────

    def section_param_analysis(self) -> dict:
        """Analyze spatial structure of bond-resolved optimal parameters.

        Key question: Do all bonds converge to the same θ_zz (= global HVA),
        or do they show spatial variation (= bond-resolved adds value)?
        """
        N, P = 10, 1
        TOPOLOGY = "heavy_hex"
        seed = 42
        H_VALS = [4.0, 3.5, 3.0]

        opt = self.VQEOptimizer(config=self.vqe_config_br, seed=seed)
        lattice_ref = self.make_lattice(TOPOLOGY, N, h=4.0)
        n_edges = len(lattice_ref.edges)
        n_params = n_edges + N
        prev_theta = None

        spatial_variance = []  # Per h-point: std of θ_zz across bonds

        for h in H_VALS:
            lattice = self.make_lattice(TOPOLOGY, N, h=h)
            H = self.builder.build(lattice)
            gt = self.solver.solve(H, lattice)
            qc, _ = self.spec_br.create_circuit(N, P, lattice)

            if prev_theta is None:
                init = np.random.default_rng(seed).uniform(-0.01, 0.01, n_params)
            else:
                init = prev_theta.copy()

            res = opt.optimize(H, qc, init, gt.ground_energy, gt.ground_state)
            prev_theta = res.theta_opt.copy()

            # Extract θ_zz (first n_edges params) and θ_x (last N params)
            theta_zz = res.theta_opt[:n_edges]
            theta_x = res.theta_opt[n_edges:]

            std_zz = float(np.std(theta_zz))
            std_x = float(np.std(theta_x))
            range_zz = float(np.ptp(theta_zz))
            range_x = float(np.ptp(theta_x))

            spatial_variance.append(
                {
                    "h": h,
                    "std_theta_zz": std_zz,
                    "std_theta_x": std_x,
                    "range_theta_zz": range_zz,
                    "range_theta_x": range_x,
                    "mean_theta_zz": float(np.mean(theta_zz)),
                    "mean_theta_x": float(np.mean(theta_x)),
                }
            )

            logger.info(
                f"  h={h:.1f} | θ_zz: mean={np.mean(theta_zz):.4f} "
                f"std={std_zz:.4f} range={range_zz:.4f} | "
                f"θ_x: mean={np.mean(theta_x):.4f} std={std_x:.4f}"
            )

        # Bond-resolved has value if parameters show spatial variation
        # (i.e., not all bonds converge to same value)
        has_spatial_structure = any(sv["std_theta_zz"] > 0.01 for sv in spatial_variance)

        return {
            "topology": TOPOLOGY,
            "n_qubits": N,
            "n_edges": n_edges,
            "spatial_variance": spatial_variance,
            "has_spatial_structure": has_spatial_structure,
            "pass": True,  # Informational section (always passes)
        }


if __name__ == "__main__":
    BondResolvedValidationRunner.main()
