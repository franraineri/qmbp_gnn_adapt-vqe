"""Unified MPNN Architecture Benchmark — 3-way comparison on unified graphs.

Tests whether the type-aware UnifiedMPNN architecture (with learned type
embeddings and gate-node readout) outperforms both:
  - Baseline A: BondResolvedMPNN + Hamiltonian-only graph
  - Variant E: BondResolvedMPNN + unified graph (same arch, richer graph)
  - Variant F: UnifiedMPNN + unified graph (type-aware arch, richer graph)

Key question: Does type-aware message passing help when the circuit has
heterogeneous gate structure (non-symmetric topologies like square, ladder)?

The Camino A results on chain_1d showed B vs A was neutral (Cohen's d = -0.30).
Hypothesis: on non-symmetric topologies (ladder, square) where gate nodes have
different connectivity, the type-aware architecture will provide real benefit.

Usage:
    # Quick test on ladder (recommended first):
    python scripts/experiment_runners/noise_aware/run_unified_mpnn_benchmark.py \\
        --topology ladder --n-qubits 10

    # Multi-topology comparison:
    python scripts/experiment_runners/noise_aware/run_unified_mpnn_benchmark.py \\
        --topology chain_1d ladder square --n-qubits 10

    # Square 4x4 (where heterogeneity is highest):
    python scripts/experiment_runners/noise_aware/run_unified_mpnn_benchmark.py \\
        --topology square --n-qubits 16

References:
    - Integration plan: internal/documentation/next-steps/04_qracle_unified_graph.md
    - Experiment plan: internal/documentation/next-steps/EXPERIMENT_PLAN_04_06.md (B.1)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np

from qmbp_simulation.framework.runner_base import Section, ValidationRunner, resolve_project_root

_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# Defaults optimized for the B.1 path (non-symmetric topologies)
DEFAULT_N = 10
DEFAULT_P = 1
DEFAULT_TOPOLOGY = "ladder"  # Non-symmetric: corner vs bulk qubits differ
DEFAULT_MODEL = "tfim_bond_resolved"
DEFAULT_H_MIN = 1.3
DEFAULT_H_MAX = 3.0
DEFAULT_H_POINTS = 20
DEFAULT_MPNN_EPOCHS = 4000
DEFAULT_HIDDEN_DIM = 256

# Variant labels
VARIANT_A = "ham_only"       # BondResolvedMPNN + Hamiltonian-only (baseline)
VARIANT_E = "unified_brm"   # BondResolvedMPNN + unified graph
VARIANT_F = "unified_type"  # UnifiedMPNN + unified graph (type-aware)


class UnifiedMPNNBenchmark(ValidationRunner):
    """3-way architecture comparison: baseline vs unified vs type-aware."""

    runner_id = "unified_mpnn_benchmark_v2"
    experiment_id = "UNIFIED_MPNN_ARCHITECTURE"
    description = (
        "3-way benchmark: Ham-only baseline vs BondResolvedMPNN+unified "
        "vs UnifiedMPNN+unified, across topologies with varying gate heterogeneity."
    )
    hypothesis = (
        "Type-aware architecture improves prediction on non-symmetric "
        "topologies where gate nodes have heterogeneous connectivity "
        "(gate_degree_cv > 0)."
    )

    @classmethod
    def _add_custom_args(cls, parser):
        cls._add_standard_physics_args(
            parser,
            n_qubits=DEFAULT_N,
            p_layers=DEFAULT_P,
            topology=DEFAULT_TOPOLOGY,
            model=DEFAULT_MODEL,
            h_min=DEFAULT_H_MIN,
            h_max=DEFAULT_H_MAX,
            h_points=DEFAULT_H_POINTS,
        )
        parser.add_argument("--mpnn-epochs", type=int, default=DEFAULT_MPNN_EPOCHS)
        parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
        parser.add_argument(
            "--type-embedding-dim", type=int, default=16,
            help="Learned type embedding dimension for UnifiedMPNN (0 to disable).",
        )
        parser.add_argument(
            "--no-gate-readout", action="store_true", default=False,
            help="Disable gate-node readout (use edge concatenation fallback).",
        )
        parser.add_argument(
            "--test-fraction", type=float, default=0.4,
            help="Fraction of h-points reserved for testing (default 0.4).",
        )
        parser.add_argument(
            "--skip-variants", nargs="*", default=[],
            choices=[VARIANT_A, VARIANT_E, VARIANT_F],
            help="Skip specific variants.",
        )

    def run_preflight(self) -> bool:
        if self._args.p_layers > 8:
            logger.error("p_layers > 8 not supported (circuit too deep).")
            return False
        if self._args.h_points < 10:
            logger.error("Need >= 10 h-points for meaningful train/test split.")
            return False
        # Warn about square N=10 (not a clean grid)
        for topo in self._args.topology:
            if topo == "square" and self._args.n_qubits < 16:
                logger.warning(
                    "  ⚠️ square with N=%d is not a clean grid. "
                    "Recommend N=16 (4×4) for square topology.",
                    self._args.n_qubits,
                )
        return True

    def build_config(self) -> dict:
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "system": {
                "n_qubits": self._args.n_qubits,
                "p_layers": self._args.p_layers,
                "topologies": self._args.topology,
                "model": self._args.model,
            },
            "h_grid": {
                "h_min": self._args.h_min,
                "h_max": self._args.h_max,
                "h_points": self._args.h_points,
            },
            "mpnn": {
                "epochs": self._args.mpnn_epochs,
                "hidden_dim": self._args.hidden_dim,
                "type_embedding_dim": self._args.type_embedding_dim,
                "gate_readout": not self._args.no_gate_readout,
            },
            "test_fraction": self._args.test_fraction,
            "skipped_variants": self._args.skip_variants,
        }

    def setup(self):
        self.setup_physics()
        self._topologies = self._args.topology
        self._N = self._args.n_qubits
        self._p = self._args.p_layers
        self._h_values = self.generate_h_grid()
        self._seed = self._args.seeds[0]
        self._spec = self.get_spec()
        self._test_fraction = self._args.test_fraction
        self._skip = set(self._args.skip_variants)

        # Storage
        self._vqe_data: dict[str, dict] = {}
        self._results_per_topo: dict[str, dict] = {}

    def define_sections(self) -> list[Section]:
        return [
            Section(id=1, name="VQE Data Collection", fn=self.section_vqe,
                    hypothesis="Bond-resolved VQE converges for all topologies"),
            Section(id=2, name="3-Way Architecture Comparison", fn=self.section_compare,
                    hypothesis="UnifiedMPNN outperforms on non-symmetric topologies"),
            Section(id=3, name="Statistical Analysis", fn=self.section_analysis,
                    hypothesis="Improvement correlates with gate_degree_cv"),
        ]

    # ══════════════════════════════════════════════════════════════════════
    # Section 1: VQE Data Collection per topology
    # ══════════════════════════════════════════════════════════════════════

    def section_vqe(self) -> dict:
        """Run bond-resolved VQE sweep for each topology."""
        from qmbp_simulation import VQEConfig, VQEOptimizer, make_lattice
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.utils.helpers import canonicalize_theta, timer

        N, p, seed = self._N, self._p, self._seed
        spec = self._spec
        h_values = self._h_values
        hva = HVACircuitBuilder()
        results = {}

        for topo in self._topologies:
            logger.info("  VQE sweep: %s N=%d p=%d (%d h-points)...",
                        topo, N, p, len(h_values))

            lattice_ref = make_lattice(topo, N, J=1.0, h=float(h_values[0]))
            circuit, _ = hva.create_bond_resolved(N, p, lattice_ref)
            n_params = circuit.num_parameters
            n_edges = len(lattice_ref.edges)

            logger.info("    Circuit: %d params (%d edges + %d qubits)",
                        n_params, n_edges, N)

            # Exact diag
            e_exact, gaps = [], []
            for h in h_values:
                lat = make_lattice(topo, N, J=1.0, h=float(h))
                H = spec.build_hamiltonian(lat, **spec.hamiltonian_kwargs)
                gt = self.solver.solve(H, lat)
                e_exact.append(gt.ground_energy)
                gaps.append(gt.gap)

            # VQE descending sweep
            vqe_config = self.VQEConfig(
                p_layers=p, n_restarts=self._args.n_restarts, maxiter=self._args.maxiter
            )
            optimizer = VQEOptimizer(config=vqe_config, backend=self.noiseless, seed=seed)
            rng = np.random.default_rng(seed)
            prev_theta = rng.uniform(-0.01, 0.01, n_params)

            with timer(f"vqe_{topo}") as t:
                theta_map: dict[float, np.ndarray] = {}
                for h in sorted(h_values, reverse=True):
                    lat = make_lattice(topo, N, J=1.0, h=float(h))
                    H = spec.build_hamiltonian(lat, **spec.hamiltonian_kwargs)
                    result = optimizer.optimize(H, circuit, initial_guess=prev_theta)
                    if np.all(np.isfinite(result.theta_opt)):
                        prev_theta = result.theta_opt.copy()
                    theta_map[float(h)] = canonicalize_theta(prev_theta.copy())

            theta_array = np.array([theta_map[float(h)] for h in h_values])

            self._vqe_data[topo] = {
                "lattice": lattice_ref,
                "circuit": circuit,
                "theta": theta_array,
                "e_exact": np.array(e_exact),
                "gaps": np.array(gaps),
                "n_edges": n_edges,
            }

            # Quality check
            de_gaps = []
            for i, h in enumerate(h_values):
                lat = make_lattice(topo, N, J=1.0, h=float(h))
                H = spec.build_hamiltonian(lat, **spec.hamiltonian_kwargs)
                e = self.noiseless.evaluate(circuit, H, theta_array[i])
                de_gaps.append(abs(e - e_exact[i]) / max(gaps[i], 1e-10))

            results[topo] = {
                "n_params": n_params,
                "n_edges": n_edges,
                "vqe_time_s": t.elapsed_s,
                "mean_de_gap": float(np.mean(de_gaps)),
                "max_de_gap": float(np.max(de_gaps)),
                "pass_rate_5pct": float(np.mean(np.array(de_gaps) < 0.05)),
            }
            logger.info("    Done: mean ΔE/gap=%.4f, pass_rate=%.0f%%, time=%.1fs",
                        results[topo]["mean_de_gap"],
                        results[topo]["pass_rate_5pct"] * 100, t.elapsed_s)

        results["pass"] = True
        results["n_topologies"] = len(self._topologies)
        return results

    # ══════════════════════════════════════════════════════════════════════
    # Section 2: 3-Way Architecture Comparison (A vs E vs F)
    # ══════════════════════════════════════════════════════════════════════

    def _split_train_test(self, h_values, theta, e_exact, gaps):
        """Split data into train/test using gap-aware strategy.

        Test points are the ones CLOSEST to h_c (smallest gap) — these
        are the hardest to predict and most diagnostic of generalization.
        This is more informative than random/interleaved splits.
        """
        h_values = np.asarray(h_values)
        gaps = np.asarray(gaps)
        n_test = max(3, int(len(h_values) * self._test_fraction))
        n_train = len(h_values) - n_test

        # Sort by gap (ascending) — smallest gaps are hardest
        gap_order = np.argsort(gaps)
        test_idx = np.sort(gap_order[:n_test])
        train_idx = np.sort(gap_order[n_test:])

        return {
            "h_train": h_values[train_idx],
            "theta_train": theta[train_idx],
            "h_test": h_values[test_idx],
            "e_test": e_exact[test_idx],
            "gaps_test": gaps[test_idx],
            "train_idx": train_idx,
            "test_idx": test_idx,
        }

    def section_compare(self) -> dict:
        """Train all 3 variants on each topology and compare."""
        from experiments.helpers.graph_utils import (
            evaluate_bond_resolved_variant,
            train_bond_resolved_variant,
            train_unified_mpnn_variant,
        )
        from qmbp_simulation.predictors.unified_graph import (
            build_unified_bond_resolved_graph,
            compute_graph_metrics,
        )
        from qmbp_simulation.utils.helpers import timer

        epochs = self._args.mpnn_epochs
        hidden = self._args.hidden_dim
        type_emb = self._args.type_embedding_dim
        gate_readout = not self._args.no_gate_readout
        results: dict[str, Any] = {}

        for topo in self._topologies:
            vdata = self._vqe_data[topo]
            lattice = vdata["lattice"]
            theta = vdata["theta"]
            circuit = vdata["circuit"]
            e_exact = vdata["e_exact"]
            gaps = vdata["gaps"]

            # Gap-aware train/test split
            split = self._split_train_test(self._h_values, theta, e_exact, gaps)
            h_train = split["h_train"]
            theta_train = split["theta_train"]
            h_test = split["h_test"]
            e_test = split["e_test"]
            gaps_test = split["gaps_test"]

            logger.info("  %s: %d train, %d test (test = smallest-gap points)",
                        topo, len(h_train), len(h_test))

            # Graph metrics (heterogeneity diagnostic)
            sample_g = build_unified_bond_resolved_graph(
                lattice, h_value=float(h_train[0]), p_layers=self._p,
                include_circuit_nodes=True,
            )
            gmetrics = compute_graph_metrics(sample_g)
            logger.info("    Graph: %d nodes (%.1f×), gate_neighborhood_cv=%.3f, "
                        "qubit_degree_cv=%.3f",
                        gmetrics["total_nodes"], gmetrics["node_expansion_ratio"],
                        gmetrics["gate_neighborhood_cv"], gmetrics["qubit_degree_cv"])

            topo_results: dict[str, Any] = {"graph_metrics": gmetrics}
            per_point_data: dict[str, list[float]] = {}

            # ── Variant A: BondResolvedMPNN + Ham-only (BASELINE) ────
            # Note: BondResolvedMPNN is p=1 only (predicts per-node/per-edge,
            # not per-layer). For p>1, skip A and E — only F supports multi-layer.
            if VARIANT_A not in self._skip:
                if self._p > 1:
                    logger.info("    Skipping A (BondResolvedMPNN + Ham-only): "
                                "not compatible with p=%d (per-node output is p=1 only)", self._p)
                else:
                    logger.info("    Training A (BondResolvedMPNN + Ham-only)...")
                    with timer("A") as t_a:
                        model_a, metrics_a = train_bond_resolved_variant(
                            lattice, h_train, theta_train,
                            include_circuit_nodes=False,
                            p_layers=self._p,
                            hidden_dim=hidden,
                            n_layers=3,
                            n_epochs=epochs,
                            seed=self._seed,
                            dropout=0.1,
                            weight_decay=0.0,
                        )
                    eval_a = evaluate_bond_resolved_variant(
                        model=model_a, lattice=lattice,
                        h_test_values=h_test, circuit=circuit,
                        spec=self._spec, backend=self.noiseless,
                        include_circuit_nodes=False,
                        p_layers=self._p,
                        e_exact=e_test, gaps=gaps_test,
                    )
                    topo_results[VARIANT_A] = self._format_variant_result(
                        metrics_a, eval_a, t_a.elapsed_s, "BondResolvedMPNN", "ham_only"
                    )
                    per_point_data[VARIANT_A] = [p["de_gap"] for p in eval_a["per_point"]]
                    logger.info("      A: MSE=%.2e, ΔE/gap=%.4f, pass=%.0f%%",
                                metrics_a["final_mse"], eval_a["mean_de_gap"],
                                eval_a["pass_rate"] * 100)

            # ── Variant E: BondResolvedMPNN + unified graph ──────────
            if VARIANT_E not in self._skip:
                if self._p > 1:
                    logger.info("    Skipping E (BondResolvedMPNN + unified): "
                                "not compatible with p=%d (per-node output is p=1 only)", self._p)
                else:
                    logger.info("    Training E (BondResolvedMPNN + unified)...")
                    with timer("E") as t_e:
                        model_e, metrics_e = train_bond_resolved_variant(
                            lattice, h_train, theta_train,
                            include_circuit_nodes=True,
                            p_layers=self._p,
                            hidden_dim=hidden,
                            n_layers=3,
                            n_epochs=epochs,
                            seed=self._seed,
                            dropout=0.15,
                            weight_decay=1e-4,
                        )
                    eval_e = evaluate_bond_resolved_variant(
                        model=model_e, lattice=lattice,
                        h_test_values=h_test, circuit=circuit,
                        spec=self._spec, backend=self.noiseless,
                        include_circuit_nodes=True,
                        p_layers=self._p,
                        e_exact=e_test, gaps=gaps_test,
                    )
                    topo_results[VARIANT_E] = self._format_variant_result(
                        metrics_e, eval_e, t_e.elapsed_s, "BondResolvedMPNN", "unified"
                    )
                    per_point_data[VARIANT_E] = [p["de_gap"] for p in eval_e["per_point"]]
                    logger.info("      E: MSE=%.2e, ΔE/gap=%.4f, pass=%.0f%%",
                                metrics_e["final_mse"], eval_e["mean_de_gap"],
                                eval_e["pass_rate"] * 100)

            # ── Variant F: UnifiedMPNN + unified graph ───────────────
            if VARIANT_F not in self._skip:
                logger.info("    Training F (UnifiedMPNN + unified, type-aware)...")
                with timer("F") as t_f:
                    model_f, metrics_f = train_unified_mpnn_variant(
                        lattice, h_train, theta_train,
                        p_layers=self._p,
                        hidden_dim=hidden,
                        n_layers=3,
                        n_epochs=epochs,
                        seed=self._seed,
                        dropout=0.1,
                        weight_decay=1e-4,
                        type_embedding_dim=type_emb,
                        gate_readout=gate_readout,
                    )
                eval_f = evaluate_bond_resolved_variant(
                    model=model_f, lattice=lattice,
                    h_test_values=h_test, circuit=circuit,
                    spec=self._spec, backend=self.noiseless,
                    include_circuit_nodes=True,
                    p_layers=self._p,
                    e_exact=e_test, gaps=gaps_test,
                )
                topo_results[VARIANT_F] = self._format_variant_result(
                    metrics_f, eval_f, t_f.elapsed_s, "UnifiedMPNN", "unified",
                    extra={"type_embedding_dim": type_emb, "gate_readout": gate_readout},
                )
                per_point_data[VARIANT_F] = [p["de_gap"] for p in eval_f["per_point"]]
                logger.info("      F: MSE=%.2e, ΔE/gap=%.4f, pass=%.0f%%",
                            metrics_f["final_mse"], eval_f["mean_de_gap"],
                            eval_f["pass_rate"] * 100)

            # ── Cross-variant comparisons ────────────────────────────
            comparisons = {}
            active_variants = [v for v in [VARIANT_A, VARIANT_E, VARIANT_F]
                               if v in topo_results]
            if len(active_variants) >= 2:
                best_variant = min(
                    active_variants,
                    key=lambda v: topo_results[v]["mean_de_gap"],
                )
                comparisons["best_variant"] = best_variant
                comparisons["ranking"] = sorted(
                    [(v, topo_results[v]["mean_de_gap"]) for v in active_variants],
                    key=lambda x: x[1],
                )

                # Pairwise MSE and deploy improvements
                if VARIANT_A in topo_results and VARIANT_F in topo_results:
                    comparisons["f_vs_a_mse_pct"] = self._pct_improvement(
                        topo_results[VARIANT_A]["final_mse"],
                        topo_results[VARIANT_F]["final_mse"],
                    )
                    comparisons["f_vs_a_deploy_pct"] = self._pct_improvement(
                        topo_results[VARIANT_A]["mean_de_gap"],
                        topo_results[VARIANT_F]["mean_de_gap"],
                    )
                if VARIANT_E in topo_results and VARIANT_F in topo_results:
                    comparisons["f_vs_e_mse_pct"] = self._pct_improvement(
                        topo_results[VARIANT_E]["final_mse"],
                        topo_results[VARIANT_F]["final_mse"],
                    )
                    comparisons["f_vs_e_deploy_pct"] = self._pct_improvement(
                        topo_results[VARIANT_E]["mean_de_gap"],
                        topo_results[VARIANT_F]["mean_de_gap"],
                    )

            topo_results["comparisons"] = comparisons
            topo_results["per_point"] = per_point_data
            topo_results["h_test"] = h_test.tolist()

            results[topo] = topo_results
            self._results_per_topo[topo] = topo_results

        results["pass"] = True
        return results

    @staticmethod
    def _format_variant_result(
        train_metrics: dict, eval_result: dict, time_s: float,
        architecture: str, graph_type: str, extra: dict | None = None,
    ) -> dict:
        """Standardized variant result format."""
        r = {
            "architecture": architecture,
            "graph_type": graph_type,
            "final_mse": train_metrics["final_mse"],
            "val_mse": train_metrics.get("val_mse"),
            "generalization_gap": train_metrics.get("generalization_gap"),
            "mean_de_gap": eval_result["mean_de_gap"],
            "median_de_gap": eval_result["median_de_gap"],
            "max_de_gap": eval_result["max_de_gap"],
            "std_de_gap": eval_result["std_de_gap"],
            "pass_rate": eval_result["pass_rate"],
            "n_pass": eval_result["n_pass"],
            "n_total": eval_result["n_total"],
            "training_time_s": time_s,
        }
        if extra:
            r.update(extra)
        return r

    @staticmethod
    def _pct_improvement(baseline_val: float, new_val: float) -> float:
        """Compute % improvement (positive = new is better)."""
        if baseline_val <= 0:
            return 0.0
        return (baseline_val - new_val) / baseline_val * 100

    # ══════════════════════════════════════════════════════════════════════
    # Section 3: Statistical Analysis
    # ══════════════════════════════════════════════════════════════════════

    def section_analysis(self) -> dict:
        """Paired statistical tests and heterogeneity correlation."""
        from project_health.analysis.statistical_tests import (
            effect_size_cohens_d,
            improvement_rate,
            paired_ttest,
        )

        results: dict[str, Any] = {}

        for topo, tdata in self._results_per_topo.items():
            per_point = tdata.get("per_point", {})
            topo_stats: dict[str, Any] = {}

            # Paired comparisons for all available pairs
            pairs = [
                ("F_vs_A", VARIANT_F, VARIANT_A, "#04 type-aware vs baseline"),
                ("E_vs_A", VARIANT_E, VARIANT_A, "#04 unified graph vs baseline"),
                ("F_vs_E", VARIANT_F, VARIANT_E, "type-aware vs same graph"),
            ]

            for label, better, worse, desc in pairs:
                de_better = per_point.get(better, [])
                de_worse = per_point.get(worse, [])

                if not de_better or not de_worse or len(de_better) != len(de_worse):
                    continue

                ttest = paired_ttest(de_worse, de_better)
                imp = improvement_rate(de_worse, de_better)
                cohens_d = effect_size_cohens_d(de_worse, de_better)

                topo_stats[label] = {
                    "description": desc,
                    "paired_ttest": ttest,
                    "improvement_rate": imp,
                    "cohens_d": float(cohens_d),
                    "wins_pct": imp["improvement_rate_pct"],
                    "significant": ttest["significant_005"],
                    "interpretation": self._interpret_effect(cohens_d),
                }

                direction = "better" if cohens_d > 0 else "WORSE"
                logger.info(
                    "  %s %s: wins %.0f%% | d=%.2f (%s) | p=%.4f [%s]",
                    topo, label, imp["improvement_rate_pct"],
                    cohens_d, self._interpret_effect(cohens_d),
                    ttest["p_value"], direction,
                )

            # Attach gate heterogeneity metric for cross-topology correlation
            gmetrics = tdata.get("graph_metrics", {})
            topo_stats["gate_degree_cv"] = gmetrics.get("gate_degree_cv", 0.0)
            topo_stats["qubit_degree_cv"] = gmetrics.get("qubit_degree_cv", 0.0)
            topo_stats["gate_neighborhood_cv"] = gmetrics.get("gate_neighborhood_cv", 0.0)

            results[topo] = topo_stats

        # Cross-topology correlation: gate_neighborhood_cv vs F improvement
        if len(results) > 1:
            gate_cvs = []
            f_improvements = []
            for topo, stats in results.items():
                if isinstance(stats, dict) and "gate_neighborhood_cv" in stats:
                    f_vs_a = stats.get("F_vs_A", {})
                    if f_vs_a:
                        gate_cvs.append(stats["gate_neighborhood_cv"])
                        f_improvements.append(f_vs_a.get("cohens_d", 0.0))

            if len(gate_cvs) >= 2:
                correlation = float(np.corrcoef(gate_cvs, f_improvements)[0, 1])
                results["heterogeneity_correlation"] = {
                    "gate_cvs": gate_cvs,
                    "cohens_d_values": f_improvements,
                    "pearson_r": correlation if np.isfinite(correlation) else 0.0,
                    "interpretation": (
                        "Confirmed: higher heterogeneity → more benefit from type-aware"
                        if correlation > 0.5 else
                        "Not confirmed: heterogeneity does not predict benefit"
                    ),
                }
                logger.info(
                    "  Cross-topo correlation(gate_cv, improvement): r=%.3f",
                    correlation if np.isfinite(correlation) else 0.0,
                )

        # Overall verdict
        results["pass"] = True
        return results

    @staticmethod
    def _interpret_effect(d: float) -> str:
        """Interpret Cohen's d effect size."""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        return "large"


if __name__ == "__main__":
    UnifiedMPNNBenchmark.main()
