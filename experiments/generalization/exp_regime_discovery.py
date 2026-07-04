"""RD1: Heisenberg XXZ Regime Discovery.

Hypothesis: The Heisenberg HVA p=2 has a valid regime at high h (paramagnetic
limit) where ground state entanglement is low enough for the shallow ansatz.
Prior experiments show max 22% fidelity at N=6 p=2 with h∈[0,2], but the
paramagnetic regime (h>>J) has not been systematically explored.

Method:
    1. Sweep h from 4.0 to 0.0 (step 0.25) for Δ ∈ {0.0, 0.5, 1.0}
    2. Run VQE with 10 restarts at each point
    3. Compute entanglement entropy at each h-point
    4. Identify h_min where fidelity ≥ 0.93 (or report negative result)
    5. Evaluate relaxed thresholds (0.80, 0.70, 0.60)
    6. Compare against TFIM baseline

Expected outcome: Likely negative result (max fidelity < 0.60) confirming
that HVA p=2 is TFIM-specific. The entanglement analysis will quantify WHY.

References:
    - Wiersema et al. (2020) — HVA for Heisenberg
    - Mele et al. (2026) — Depth bounds for variational circuits
    - E4 experiment — HVA is model-specific (fidelity drops at g>0)
"""

from __future__ import annotations

import logging
import time

import numpy as np

from qmbp_simulation.analysis.comparative import (
    RegimeDiscoveryResult,
    classify_result,
    filter_by_threshold,
    find_minimum_viable_threshold,
)
from qmbp_simulation.analysis.entanglement import EntanglementAnalyzer
from qmbp_simulation.framework import BaseExperiment
from qmbp_simulation.framework.config import ExperimentConfig, SystemConfig, VQEConfig
from qmbp_simulation.framework.metrics import ExperimentMetrics
from qmbp_simulation.models import get_model_spec, make_lattice
from qmbp_simulation.pipeline import PipelineRunner

logger = logging.getLogger(__name__)


class RegimeDiscoveryExperiment(BaseExperiment):
    """Systematic regime discovery for Heisenberg XXZ at N=6 p=2."""

    DELTA_VALUES = [0.0, 0.5, 1.0]  # XY, intermediate, isotropic

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="RD1",
            category="G",
            description="Heisenberg XXZ regime discovery (N=6, p=2)",
            hypothesis=(
                "HVA p=2 has a valid regime at high h for Heisenberg XXZ "
                "where ground state entanglement is below the ansatz capacity."
            ),
            system=SystemConfig(
                n_qubits=6,
                p_layers=2,
                topology="chain_1d",
                J=1.0,
                model="heisenberg",
                h_values=list(np.arange(4.0, -0.01, -0.25)),
                delta=1.0,
            ),
            vqe=VQEConfig(
                n_restarts=10,
                restart_sigma=0.5,
                maxiter=1500,
            ),
            seeds=DEFAULT_SEEDS,
        )

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Run regime discovery for one seed across all delta values."""
        cfg = self.config
        h_values = np.array(cfg.system.h_values)
        metrics_list = []
        analyzer = EntanglementAnalyzer()

        for delta in self.DELTA_VALUES:
            self.slog.log(
                "regime_discovery_start",
                seed=seed,
                data={"delta": delta, "n_h_points": len(h_values)},
            )
            t0 = time.time()

            # Get model spec with delta override
            spec = get_model_spec("heisenberg")
            spec_with_delta = spec.with_delta(delta)

            # Build lattice and pipeline
            lattice = make_lattice(
                topology=cfg.system.topology,
                n_qubits=cfg.system.n_qubits,
                J=cfg.system.J,
                h=float(h_values[0]),
                periodic=False,
            )

            vqe_config = self._build_vqe_config()
            runner = PipelineRunner(
                lattice=lattice,
                config=vqe_config,
                seed=seed,
                model_spec=spec_with_delta,
            )

            # Phase 1: Exact diag
            exact_data = runner.run_phase1(h_values)

            # Compute entanglement entropy
            ground_states = [r.ground_state for r in exact_data]
            ent_results = analyzer.analyze_sweep(h_values, ground_states, cfg.system.n_qubits)

            # Phase 2: VQE sweep
            vqe_results = runner.run_phase2(h_values, exact_data)
            fidelities = [r.fidelity for r in vqe_results]

            # Analyze results
            max_fid = max(fidelities)
            classification = classify_result(max_fid)
            valid_indices = filter_by_threshold(fidelities, 0.93)
            viable_threshold = find_minimum_viable_threshold(fidelities)

            # Find HVA capacity
            capacity = analyzer.find_hva_capacity_threshold(
                ent_results, fidelities, fidelity_threshold=0.93
            )

            elapsed = time.time() - t0

            # Build result
            result = RegimeDiscoveryResult(
                model_type="heisenberg",
                delta=delta,
                n_qubits=cfg.system.n_qubits,
                h_values=h_values.tolist(),
                fidelities=fidelities,
                entropies=[r.entropy for r in ent_results],
                h_min=float(h_values[valid_indices[0]]) if valid_indices else None,
                max_fidelity=max_fid,
                valid_regime_width=len(valid_indices),
                threshold_used=0.93,
                is_negative_result=(classification == "fundamental_expressibility_limitation"),
            )

            self.slog.log(
                "regime_discovery_complete",
                seed=seed,
                data={
                    "delta": delta,
                    "max_fidelity": max_fid,
                    "classification": classification,
                    "valid_regime_width": len(valid_indices),
                    "viable_threshold": viable_threshold,
                    "hva_capacity": capacity,
                    "elapsed_s": elapsed,
                },
            )

            # Create metrics
            metrics_list.append(
                ExperimentMetrics(
                    experiment_id=f"RD1_delta{delta:.1f}",
                    seed=seed,
                    h_value=0.0,  # Summary metric
                    delta_e_over_gap=0.0,
                    fidelity=max_fid,
                    extra={
                        "delta": delta,
                        "classification": classification,
                        "valid_regime_width": len(valid_indices),
                        "h_min": result.h_min,
                        "viable_threshold": viable_threshold,
                        "hva_capacity": capacity,
                        "max_entanglement": max(r.entropy for r in ent_results),
                        "elapsed_s": elapsed,
                    },
                )
            )

        return metrics_list

    def _build_vqe_config(self):
        """Build VQEConfig from experiment config."""
        from qmbp_simulation.models import VQEConfig as VQECfg

        return VQECfg(
            p_layers=self.config.system.p_layers,
            n_restarts=self.config.vqe.n_restarts,
            maxiter=self.config.vqe.maxiter,
            restart_sigma=self.config.vqe.sigma,
        )
