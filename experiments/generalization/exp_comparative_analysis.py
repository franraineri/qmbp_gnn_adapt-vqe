"""CA1: Comparative Analysis — TFIM vs Heisenberg Pipeline.

Hypothesis: The GNN-HVA pipeline architecture is model-agnostic, but HVA p=2
expressibility limits restrict the Heisenberg model to a narrower valid regime
(or no valid regime at all). The comparison quantifies this gap.

Method:
    1. Run TFIM pipeline at N=6 (baseline, known to work)
    2. Run Heisenberg pipeline at N=6 (same config except model)
    3. Compute comparative metrics side-by-side
    4. Quantify CX budget difference (3× for Heisenberg)
    5. Compare entanglement entropy profiles
    6. Classify outcome and document result

Expected outcome: TFIM passes (ΔE/gap < 5%), Heisenberg likely fails or has
very narrow valid regime. The comparison provides thesis-ready quantification.

References:
    - E4 experiment — HVA is model-specific
    - B4 experiment — HVA landscape is saddle-free (for TFIM)
"""

from __future__ import annotations

import logging
import time

import numpy as np

from qmbp_simulation.analysis.comparative import (
    ComparativeMetrics,
    classify_outcome,
    compute_cx_budget,
    filter_by_threshold,
    generate_comparison_table,
)
from qmbp_simulation.analysis.entanglement import EntanglementAnalyzer
from qmbp_simulation.framework import BaseExperiment
from qmbp_simulation.framework.config import ExperimentConfig, SystemConfig, VQEConfig
from qmbp_simulation.framework.metrics import ExperimentMetrics
from qmbp_simulation.models import get_model_spec, make_lattice
from qmbp_simulation.pipeline import PipelineRunner

logger = logging.getLogger(__name__)


class ComparativeAnalysisExperiment(BaseExperiment):
    """Side-by-side comparison of TFIM vs Heisenberg pipeline at N=6."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="CA1",
            category="G",
            description="Comparative analysis: TFIM vs Heisenberg at N=6",
            hypothesis=(
                "The pipeline architecture is model-agnostic but HVA p=2 "
                "expressibility limits restrict Heisenberg to a narrower "
                "valid regime than TFIM."
            ),
            system=SystemConfig(
                n_qubits=6,
                p_layers=2,
                topology="chain_1d",
                J=1.0,
                model="tfim",  # Will run both models
                h_values=list(np.arange(4.0, 0.49, -0.25)),
            ),
            vqe=VQEConfig(
                n_restarts=5,
                sigma=0.1,
                maxiter=1000,
            ),
            seeds=DEFAULT_SEEDS,
        )

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Run both TFIM and Heisenberg pipelines and compare."""
        cfg = self.config
        h_values = np.array(cfg.system.h_values)
        analyzer = EntanglementAnalyzer()
        metrics_list = []

        # --- TFIM Baseline ---
        self.slog.log("comparative_tfim_start", seed=seed)
        tfim_metrics = self._run_model_pipeline(
            model_type="tfim",
            h_values=h_values,
            seed=seed,
            analyzer=analyzer,
            n_restarts=5,
            sigma=0.1,
            maxiter=1000,
        )

        # --- Heisenberg ---
        self.slog.log("comparative_heisenberg_start", seed=seed)
        heis_metrics = self._run_model_pipeline(
            model_type="heisenberg",
            h_values=h_values,
            seed=seed,
            analyzer=analyzer,
            n_restarts=10,
            sigma=0.5,
            maxiter=1500,
        )

        # Generate comparison
        table = generate_comparison_table(tfim_metrics, heis_metrics)
        tfim_outcome = classify_outcome(
            tfim_metrics.delta_e_over_gap,
            tfim_metrics.valid_regime_width > 0,
        )
        heis_outcome = classify_outcome(
            heis_metrics.delta_e_over_gap,
            heis_metrics.valid_regime_width > 0,
        )

        self.slog.log(
            "comparative_complete",
            seed=seed,
            data={
                "tfim_outcome": tfim_outcome,
                "heisenberg_outcome": heis_outcome,
                "tfim_valid_width": tfim_metrics.valid_regime_width,
                "heisenberg_valid_width": heis_metrics.valid_regime_width,
                "comparison_table": table,
            },
        )

        metrics_list.append(
            ExperimentMetrics(
                experiment_id="CA1",
                seed=seed,
                h_value=0.0,
                delta_e_over_gap=0.0,
                fidelity=0.0,
                extra={
                    "tfim": {
                        "outcome": tfim_outcome,
                        "avg_fidelity": tfim_metrics.avg_fidelity,
                        "valid_regime_width": tfim_metrics.valid_regime_width,
                        "cx_budget": tfim_metrics.cx_budget_per_layer,
                        "max_entanglement": tfim_metrics.max_entanglement,
                    },
                    "heisenberg": {
                        "outcome": heis_outcome,
                        "avg_fidelity": heis_metrics.avg_fidelity,
                        "valid_regime_width": heis_metrics.valid_regime_width,
                        "cx_budget": heis_metrics.cx_budget_per_layer,
                        "max_entanglement": heis_metrics.max_entanglement,
                    },
                    "comparison_table": table,
                },
            )
        )

        return metrics_list

    def _run_model_pipeline(
        self,
        model_type: str,
        h_values: np.ndarray,
        seed: int,
        analyzer: EntanglementAnalyzer,
        n_restarts: int,
        sigma: float,
        maxiter: int,
    ) -> ComparativeMetrics:
        """Run pipeline for a single model and return metrics."""
        from qmbp_simulation.models import VQEConfig as VQECfg

        cfg = self.config
        spec = get_model_spec(model_type)

        lattice = make_lattice(
            topology=cfg.system.topology,
            n_qubits=cfg.system.n_qubits,
            J=cfg.system.J,
            h=float(h_values[0]),
            periodic=False,
        )

        vqe_config = VQECfg(
            p_layers=cfg.system.p_layers,
            n_restarts=n_restarts,
            maxiter=maxiter,
            restart_sigma=sigma,
        )

        runner = PipelineRunner(
            lattice=lattice,
            config=vqe_config,
            seed=seed,
            model_spec=spec,
        )

        t0 = time.time()

        # Phase 1
        exact_data = runner.run_phase1(h_values)

        # Entanglement
        ground_states = [r.ground_state for r in exact_data]
        ent_results = analyzer.analyze_sweep(h_values, ground_states, cfg.system.n_qubits)

        # Phase 2
        vqe_results = runner.run_phase2(h_values, exact_data)
        fidelities = [r.fidelity for r in vqe_results]

        elapsed = time.time() - t0

        # Compute metrics
        valid_indices = filter_by_threshold(fidelities, 0.93)
        n_edges = len(lattice.edges)
        cx_budget = compute_cx_budget(n_edges, model_type)

        # HVA capacity
        capacity = analyzer.find_hva_capacity_threshold(ent_results, fidelities, 0.93)

        return ComparativeMetrics(
            model_type=model_type,
            delta_e_over_gap=None,  # Would need Phase 4
            avg_fidelity=float(np.mean(fidelities)),
            mpnn_mse=None,
            valid_regime_width=len(valid_indices),
            vqe_time_s=elapsed,
            cx_budget_per_layer=cx_budget,
            max_entanglement=max(r.entropy for r in ent_results),
            hva_capacity=capacity,
        )
