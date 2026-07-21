#!/usr/bin/env python3
"""MC-Dropout Uncertainty Quantification — Robust Version.

Inherits the full noiseless pipeline (ExactDiag → VQE → MPNN → Deploy)
and adds an MC-Dropout evaluation section. Reuses all validated logic
from NoiselessPipelineRunner.

Key improvements over exp_s6:
  - N=10 (statistically meaningful: 34 h-test points vs 5)
  - p=1,2,3,4 (tests across circuit depths)
  - Multiple MC-Dropout configurations (20, 50, 100 passes)
  - Spearman + Pearson correlations (rank + linear)
  - Proper statistical significance with n≥30 points

Usage:
    # Single config
    python scripts/experiment_runners/noiseless/run_mc_dropout_uq.py \\
        --n-qubits 10 --p-layers 3 --topology chain_1d

    # Sweep all p values
    for p in 1 2 3 4; do
        python scripts/experiment_runners/noiseless/run_mc_dropout_uq.py \\
            --n-qubits 10 --p-layers $p --topology chain_1d
    done
"""

from __future__ import annotations

import logging
import sys

import numpy as np

from qmbp_simulation.framework.runner_base import Section, resolve_project_root

_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import the production noiseless runner and extend it
from scripts.experiment_runners.noiseless.run_noiseless_pipeline import (
    NoiselessPipelineRunner,
)

logger = logging.getLogger(__name__)

# MC-Dropout configurations to test
MC_CONFIGS = [20, 50, 100]  # Number of forward passes


class MCDropoutRunner(NoiselessPipelineRunner):
    """Extends NoiselessPipelineRunner with MC-Dropout UQ evaluation.

    Inherits Sections 1-4 (ExactDiag, VQE, MPNN, Deploy) unchanged.
    Adds Section 5: MC-Dropout uncertainty quantification.
    """

    runner_id = "mc_dropout_uq_v2"
    experiment_id = "noiseless/mc_dropout"
    description = (
        "MC-Dropout UQ: Evaluate correlation between predicted uncertainty "
        "and actual ΔE/gap error across MC-Dropout configurations."
    )
    hypothesis = (
        "MC-Dropout variance (model.train() with K forward passes) achieves "
        "Pearson r > 0.5 and Spearman ρ > 0.5 between predicted variance and "
        "actual ΔE/gap, with statistical significance (p < 0.05) on N≥30 points."
    )

    def define_sections(self) -> list[Section]:
        """Inherit sections 1-4, add section 5 for MC-Dropout."""
        base_sections = super().define_sections()
        mc_section = Section(
            id=5,
            name="MC-Dropout Uncertainty Quantification",
            fn=self.section_mc_dropout,
            hypothesis=(
                "MC-Dropout variance correlates with actual prediction error "
                "(Pearson r > 0.5, p < 0.05) across multiple dropout configurations."
            ),
        )
        return base_sections + [mc_section]

    def build_config(self) -> dict:
        """Extend config with MC-Dropout parameters."""
        config = super().build_config()
        config["mc_dropout"] = {
            "n_passes_configs": MC_CONFIGS,
            "dropout_rate": 0.1,
            "metric": "mean_variance_across_params",
        }
        return config

    def section_mc_dropout(self) -> dict:
        """Section 5: MC-Dropout uncertainty quantification.

        Uses the trained MPNN from Section 3 and the deploy test points
        from Section 4. For each MC configuration (K forward passes):
          1. Set model to train mode (dropout active)
          2. Run K forward passes per test h-point
          3. Compute mean prediction and variance per point
          4. Correlate variance with actual ΔE/gap from Section 4
          5. Report Pearson r, Spearman ρ, and p-values
        """
        import torch
        from scipy.stats import pearsonr, spearmanr
        from torch_geometric.data import Batch, Data

        from qmbp_simulation import HamiltonianBuilder

        if self._mpnn_model is None:
            self._mpnn_model = self._try_load_mpnn_checkpoint()
        if self._mpnn_model is None:
            return {"pass": False, "error": "No trained MPNN. Run Sections 1-3 first."}

        # Check that model actually has dropout layers
        has_dropout = any(isinstance(m, torch.nn.Dropout) for m in self._mpnn_model.modules())
        if not has_dropout:
            logger.warning(
                "  ⚠️  MPNN has no Dropout layers — MC-Dropout will produce zero variance. "
                "Ensure MPNN was trained with dropout > 0."
            )
            return {
                "pass": False,
                "error": "MPNN has no Dropout layers. MC-Dropout requires dropout > 0.",
            }

        N = self._args.n_qubits
        p = self._args.p_layers
        topo = self._args.topology[0]
        spec = self._get_spec()
        builder = HamiltonianBuilder()

        # Get test h-points (same as Section 4: midpoints between training points)
        train_h = sorted([r["h"] for r in self._vqe_results[topo]])
        test_h = [(train_h[i] + train_h[i + 1]) / 2 for i in range(len(train_h) - 1)]
        logger.info(
            f"  MC-Dropout UQ: {len(test_h)} test points, "
            f"configs={MC_CONFIGS} passes, topology={topo}"
        )

        # Build graph structure once
        lattice_ref = self.make_lattice(topo, N, J=1.0, h=test_h[0])
        edge_index_np, coord = builder.build_graph_data(lattice_ref)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)

        # Build circuit for energy evaluation
        circuit, _ = spec.create_circuit(N, p, lattice_ref, **spec.circuit_kwargs)

        # Compute actual ΔE/gap at each test point using deterministic MPNN prediction
        actual_de_gaps = []
        actual_delta_es = []
        h_test_values = []
        model_name = self._args.model

        self._mpnn_model.eval()
        for h_t in test_h:
            e_exact, gap = self.exact_ground_state(topo, N, h_t, model=model_name)

            # Deterministic prediction (eval mode, no dropout)
            x_feat = np.stack([np.full(N, h_t), coord], axis=1)
            x = torch.tensor(x_feat, dtype=torch.float32)
            graph = Data(x=x, edge_index=edge_index)

            with torch.no_grad():
                theta_det = self._mpnn_model(Batch.from_data_list([graph])).numpy().flatten()

            H = spec.build_hamiltonian(
                self.make_lattice(topo, N, J=1.0, h=h_t), **spec.hamiltonian_kwargs
            )
            e_pred = self._vqe_backend.evaluate(circuit, H, theta_det)
            de_gap = abs(e_pred - e_exact) / max(gap, 1e-10)
            delta_e = abs(e_pred - e_exact)

            actual_de_gaps.append(de_gap)
            actual_delta_es.append(delta_e)
            h_test_values.append(h_t)

        # Run MC-Dropout for each configuration
        mc_results = {}
        overall_pass = False

        for n_passes in MC_CONFIGS:
            logger.info(f"    MC-Dropout K={n_passes}:")
            predicted_variances = []
            predicted_cvs = []  # Coefficient of variation per point

            # Fix seed for reproducibility of dropout stochasticity
            torch.manual_seed(42)
            self._mpnn_model.train()  # Activate dropout

            for h_t in test_h:
                x_feat = np.stack([np.full(N, h_t), coord], axis=1)
                x = torch.tensor(x_feat, dtype=torch.float32)
                graph = Data(x=x, edge_index=edge_index)
                batch = Batch.from_data_list([graph])

                # K forward passes with dropout active
                predictions = []
                with torch.no_grad():
                    for _ in range(n_passes):
                        pred = self._mpnn_model(batch).numpy().flatten()
                        predictions.append(pred)

                predictions = np.array(predictions)  # (K, n_params)
                # Mean variance across parameters = uncertainty estimate
                theta_var = float(predictions.var(axis=0).mean())
                predicted_variances.append(theta_var)
                # Coefficient of variation (normalized uncertainty)
                theta_mean_abs = np.abs(predictions.mean(axis=0))
                theta_std = predictions.std(axis=0)
                cv = float(np.mean(theta_std / np.maximum(theta_mean_abs, 1e-10)))
                predicted_cvs.append(cv)

            # Guard: check if variances are all zero (no dropout effect)
            if np.std(predicted_variances) < 1e-15:
                logger.warning(
                    f"      ⚠️  K={n_passes}: All variances are identical "
                    f"(={predicted_variances[0]:.2e}). Dropout may be inactive."
                )
                mc_results[n_passes] = {
                    "n_passes": n_passes,
                    "n_test_points": len(test_h),
                    "pearson_r": 0.0,
                    "pearson_p_value": 1.0,
                    "spearman_rho": 0.0,
                    "spearman_p_value": 1.0,
                    "pass": False,
                    "error": "Zero variance — dropout inactive or model too confident",
                }
                continue

            # Compute correlations
            n_pts = len(actual_de_gaps)
            pearson_r, pearson_p = pearsonr(predicted_variances, actual_de_gaps)
            spearman_rho, spearman_p = spearmanr(predicted_variances, actual_de_gaps)

            # Also correlate with |ΔE| (not normalized by gap)
            pearson_r_de, pearson_p_de = pearsonr(predicted_variances, actual_delta_es)

            # CV-based correlations (may be more stable)
            pearson_r_cv, pearson_p_cv = pearsonr(predicted_cvs, actual_de_gaps)

            config_pass = pearson_p < 0.05 and pearson_r > 0.5
            if config_pass:
                overall_pass = True

            mc_results[n_passes] = {
                "n_passes": n_passes,
                "n_test_points": n_pts,
                "pearson_r": float(pearson_r),
                "pearson_p_value": float(pearson_p),
                "spearman_rho": float(spearman_rho),
                "spearman_p_value": float(spearman_p),
                "pearson_r_delta_e": float(pearson_r_de),
                "pearson_p_delta_e": float(pearson_p_de),
                "pearson_r_cv": float(pearson_r_cv),
                "pearson_p_cv": float(pearson_p_cv),
                "mean_variance": float(np.mean(predicted_variances)),
                "std_variance": float(np.std(predicted_variances)),
                "mean_cv": float(np.mean(predicted_cvs)),
                "pass": config_pass,
                "per_point_variance": predicted_variances,
                "per_point_cv": predicted_cvs,
            }

            status = "✅" if config_pass else "❌"
            logger.info(
                f"      {status} K={n_passes}: Pearson r={pearson_r:.3f} (p={pearson_p:.4f}), "
                f"Spearman ρ={spearman_rho:.3f} (p={spearman_p:.4f}), "
                f"n={n_pts} points"
            )

        # Summary
        best_config = max(mc_results.values(), key=lambda x: x["pearson_r"])
        logger.info(
            f"    Best config: K={best_config['n_passes']} → "
            f"r={best_config['pearson_r']:.3f}, ρ={best_config['spearman_rho']:.3f}"
        )

        return {
            "pass": overall_pass,
            "n_test_points": len(test_h),
            "h_test_values": h_test_values,
            "n_qubits": N,
            "p_layers": p,
            "topology": topo,
            "actual_de_gaps": actual_de_gaps,
            "actual_delta_es": actual_delta_es,
            "mc_configs": {str(k): v for k, v in mc_results.items()},
            "best_config": {
                "n_passes": best_config["n_passes"],
                "pearson_r": best_config["pearson_r"],
                "spearman_rho": best_config["spearman_rho"],
            },
            "hypothesis_result": ("CONFIRMED" if overall_pass else "REJECTED"),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    runner = MCDropoutRunner()
    sys.exit(runner.run())
