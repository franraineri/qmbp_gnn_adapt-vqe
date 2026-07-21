#!/usr/bin/env python3
"""Tier 1C: Weight-Space Phase Detection (D1) for Frustrated TFIM.

Extends the D1 weight gradient analysis to the frustrated TFIM (J₁-J₂ model).
Frustration shifts/removes the critical point — the question is whether
||dW/dh|| still peaks at the crossover, and how J₂ affects the peak location.

D1 for standard TFIM found peaks near h_c=1.0 (with dropout=0.1 regularization).
Here we test whether the same mechanism generalizes to frustrated systems where
the phase diagram is richer (competing nearest-neighbor and next-nearest-neighbor).

Hypothesis:
  The MPNN weight gradient norm ||dW/dh|| exhibits a peak that shifts with J₂,
  tracking the frustrated TFIM crossover. Peak detection works without QPU.

Sections:
  1. VQE data generation — descending sweep at multiple J₂ values
  2. MPNN training — with dropout=0.1 (D1-regularized variant)
  3. Gradient analysis — WeightGradientAnalyzer on trained model
  4. J₂-dependence — how does peak location vary with frustration?
  5. Comparison with exact crossover — validate against exact diag gap minimum

Usage:
    python scripts/run_t1c_d1_frustrated.py
    python scripts/run_t1c_d1_frustrated.py --section 1 2 3
    python scripts/run_t1c_d1_frustrated.py --dry-run
    python scripts/run_t1c_d1_frustrated.py --j2-values 0.0 0.2 0.4
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
from qmbp_simulation.models.constants import DEFAULT_SEEDS

_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

N_QUBITS = 6
P_LAYERS = 2
TOPOLOGY = "chain_1d"
SEEDS = DEFAULT_SEEDS

# J₂ values to compare (0 = standard TFIM for reference)
J2_VALUES_DEFAULT = [0.0, 0.2, 0.3, 0.5]

# Dense h-grid for phase detection (need fine resolution near crossover)
# Standard TFIM h_c ≈ 1.0 (N=6 OBC: ~0.85-0.95). Frustrated TFIM shifts lower.
# Extended range [0.25, 2.5] to capture the transition even with frustration.
H_GRID = np.linspace(0.25, 2.5, 30).tolist()

# VQE — D1 doesn't need perfect convergence, just reasonable θ(h) landscape
VQE_RESTARTS = 3
VQE_MAXITER = 300
VQE_SIGMA = 0.1

# MPNN (D1-regularized: dropout=0.1 per validated decision)
MPNN_HIDDEN_DIM = 64
MPNN_N_LAYERS = 3
MPNN_EPOCHS = 3000
MPNN_LR = 1e-3
MPNN_PATIENCE = 300
DROPOUT = 0.1  # Critical for D1 robustness (7× lower peak variance)


# ═══════════════════════════════════════════════════════════════════════════════
# Runner Implementation
# ═══════════════════════════════════════════════════════════════════════════════


class D1FrustratedRunner(ValidationRunner):
    """Tier 1C: Weight-space phase detection for frustrated TFIM.

    Sections:
        1. VQE data generation across h-sweep at multiple J₂
        2. MPNN training with dropout=0.1 (per D1-regularized variant)
        3. Weight gradient analysis (WeightGradientAnalyzer)
        4. J₂-dependence of peak location
        5. Comparison with exact gap minimum (crossover validation)
    """

    runner_id = "t1c_d1_frustrated"
    experiment_id = "T1c"
    description = "D1 Weight-Space Phase Detection — Frustrated TFIM (J₁-J₂)"
    hypothesis = (
        "||dW/dh|| peaks track the frustrated TFIM crossover, "
        "generalizing the zero-QPU phase detection to frustrated systems"
    )

    @classmethod
    def _add_custom_args(cls, parser):
        parser.add_argument(
            "--j2-values",
            type=float,
            nargs="+",
            default=None,
            help=f"J₂ values to test (default: {J2_VALUES_DEFAULT})",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Primary seed for VQE + MPNN (default: 42)",
        )

    def build_config(self) -> dict:
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "category": "T",
            "model": "tfim_frustrated",
            "system": {
                "n_qubits": N_QUBITS,
                "p_layers": P_LAYERS,
                "topology": TOPOLOGY,
            },
            "d1_config": {
                "j2_values": self._j2_values,
                "h_grid": H_GRID,
                "dropout": DROPOUT,
                "mpnn_epochs": MPNN_EPOCHS,
            },
            "seeds": SEEDS,
        }

    def setup(self):
        """Lazy imports and shared objects."""
        from scipy.optimize import minimize

        from qmbp_simulation import HamiltonianBuilder, make_lattice
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.execution import NoiselessBackend

        self.builder = HamiltonianBuilder()
        self.hva = HVACircuitBuilder()
        self.backend = NoiselessBackend()
        self._minimize = minimize
        self._make_lattice = make_lattice

        # CLI overrides
        self._j2_values = self._args.j2_values if self._args.j2_values else J2_VALUES_DEFAULT
        self._seed = self._args.seed

        # Build reference circuit
        lattice_ref = make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=2.0)
        self._circuit, _ = self.hva.create_frustrated_tfim(N_QUBITS, P_LAYERS, lattice_ref)
        self._n_params = self._circuit.num_parameters
        self._lattice_ref = lattice_ref

        # Shared state across sections
        self._vqe_data: dict[float, dict] = {}  # j2 → {h_values, theta_opt, ...}
        self._models: dict[float, object] = {}  # j2 → trained MPNN
        self._datasets: dict[float, list] = {}  # j2 → graph dataset
        self._gradient_results: dict[float, object] = {}  # j2 → GradientAnalysisResult
        self._exact_crossovers: dict[float, float] = {}  # j2 → h at min gap

    def define_sections(self) -> list[Section]:
        return [
            Section(
                id=1,
                name="VQE Data Generation (per J₂)",
                fn=self.section_vqe,
                hypothesis=(
                    "VQE converges on dense h-grid across multiple J₂ "
                    "values with fid≥0.80 (extended regime for D1)"
                ),
            ),
            Section(
                id=2,
                name="MPNN Training (dropout=0.1)",
                fn=self.section_mpnn_training,
                hypothesis="MPNN converges with dropout regularization at each J₂",
            ),
            Section(
                id=3,
                name="Weight Gradient Analysis",
                fn=self.section_gradient_analysis,
                hypothesis="||dW/dh|| shows peaks that can be located automatically",
            ),
            Section(
                id=4,
                name="J₂-Dependence of Peak Location",
                fn=self.section_j2_dependence,
                hypothesis=(
                    "Peak location shifts systematically with J₂ "
                    "(moves toward smaller h as frustration increases)"
                ),
            ),
            Section(
                id=5,
                name="Comparison with Exact Crossover",
                fn=self.section_exact_comparison,
                hypothesis=("Gradient peak is within Δh≤0.3 of the exact gap minimum"),
            ),
        ]

    # ── Section 1: VQE Data Generation ───────────────────────────────────────

    def section_vqe(self) -> dict:
        """Generate VQE data across the full h-grid for each J₂ value.

        Uses the full h-range [0.5, 2.5] (including below valid regime) because
        D1 requires seeing the phase transition region to detect it.
        """
        from qiskit.quantum_info import Statevector, state_fidelity

        h_arr = np.array(H_GRID)
        results_summary = {}

        for j2 in self._j2_values:
            logger.info(f"\n  --- J₂ = {j2:.2f} ---")
            rng = np.random.default_rng(self._seed)
            prev_theta = rng.uniform(-0.01, 0.01, self._n_params)

            theta_opt = np.zeros((len(h_arr), self._n_params))
            e_exact_arr = np.zeros(len(h_arr))
            fid_arr = np.zeros(len(h_arr))
            gap_arr = np.zeros(len(h_arr))

            # Map h → index for filling during descending sweep
            h_to_idx = {float(h): i for i, h in enumerate(h_arr)}

            for h in sorted(h_arr, reverse=True):
                idx = h_to_idx[float(h)]
                lattice = self._make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=h)

                if j2 == 0.0:
                    H = self.builder.build(lattice)
                else:
                    H = self.builder.build_frustrated_tfim(lattice, J2=j2)

                # Exact ground state
                H_mat = H.to_matrix()
                if hasattr(H_mat, "toarray"):
                    H_mat = H_mat.toarray()
                evals, evecs = np.linalg.eigh(H_mat)
                e_exact_arr[idx] = float(evals[0])
                gap_arr[idx] = float(evals[1] - evals[0])
                gs = evecs[:, 0]

                # Multi-restart VQE
                best_energy = float("inf")
                best_theta = prev_theta.copy()
                for restart in range(VQE_RESTARTS):
                    x0 = (
                        prev_theta + rng.normal(0, VQE_SIGMA, self._n_params)
                        if restart > 0
                        else prev_theta.copy()
                    )
                    x0 = np.clip(x0, -np.pi, np.pi)
                    res = self._minimize(
                        lambda params, _H=H: self.backend.evaluate(self._circuit, _H, params),
                        x0,
                        method="L-BFGS-B",
                        bounds=[(-np.pi, np.pi)] * self._n_params,
                        options={"maxiter": VQE_MAXITER, "ftol": 1e-14},
                    )
                    if res.fun < best_energy:
                        best_energy = res.fun
                        best_theta = res.x.copy()

                prev_theta = best_theta.copy()
                theta_opt[idx] = best_theta.copy()

                sv = Statevector(self._circuit.assign_parameters(best_theta))
                fid_arr[idx] = float(state_fidelity(sv, Statevector(gs)))

            # Store for later sections
            self._vqe_data[j2] = {
                "h_values": h_arr,
                "theta_opt": theta_opt,
                "e_exact": e_exact_arr,
                "fidelities": fid_arr,
                "gaps": gap_arr,
            }

            # Find exact crossover (gap minimum, excluding grid boundaries)
            # The gap minimum at the grid edge is an artifact — the transition
            # is outside the measured range.
            interior_mask = (h_arr > h_arr[0]) & (h_arr < h_arr[-1])
            if np.any(interior_mask):
                interior_gaps = gap_arr.copy()
                interior_gaps[~interior_mask] = np.inf
                min_gap_idx = int(np.argmin(interior_gaps))
            else:
                min_gap_idx = int(np.argmin(gap_arr))
            h_crossover = float(h_arr[min_gap_idx])
            # Flag if crossover is at grid edge (unreliable)
            at_edge = min_gap_idx == 0 or min_gap_idx == len(h_arr) - 1
            self._exact_crossovers[j2] = h_crossover

            mean_fid = float(np.mean(fid_arr))
            n_high_fid = int(np.sum(fid_arr >= 0.80))
            results_summary[str(j2)] = {
                "mean_fidelity": mean_fid,
                "n_high_fid": n_high_fid,
                "n_total": len(h_arr),
                "min_gap": float(np.min(gap_arr)),
                "h_crossover_exact": h_crossover,
                "crossover_at_edge": at_edge,
            }

            logger.info(
                f"    Mean fid: {mean_fid:.4f}, "
                f"high-fid: {n_high_fid}/{len(h_arr)}, "
                f"gap min at h={h_crossover:.2f}"
                f"{' (EDGE!)' if at_edge else ''}"
            )

        return {
            "per_j2": results_summary,
            "n_j2_values": len(self._j2_values),
            "h_grid_size": len(H_GRID),
            "pass": all(v["mean_fidelity"] >= 0.70 for v in results_summary.values()),
        }

    # ── Section 2: MPNN Training ─────────────────────────────────────────────

    def section_mpnn_training(self) -> dict:
        """Train MPNN with dropout=0.1 at each J₂ value.

        Uses the D1-regularized variant (dropout=0.1) which was validated
        to produce 7× lower peak variance than unregularized.
        """
        from qmbp_simulation.predictors import (
            MPNNPredictor,
            build_graph_dataset,
            train_mpnn,
        )

        if not self._vqe_data:
            raise RuntimeError("Section 1 must run first")

        results_summary = {}

        for j2 in self._j2_values:
            logger.info(f"\n  --- Training MPNN for J₂={j2:.2f} ---")
            data = self._vqe_data[j2]

            # Build graph dataset with low fidelity threshold
            # (D1 needs full h-range including low-fidelity points near h_c)
            dataset = build_graph_dataset(
                lattice=self._lattice_ref,
                h_values=data["h_values"],
                theta_opt=data["theta_opt"],
                e_exact=data["e_exact"],
                fidelities=data["fidelities"],
                fidelity_threshold=0.0,  # noqa — D1 gradient needs ALL points including near h_c
            )

            logger.info(f"    Dataset: {len(dataset)} graphs")

            # Create MPNN with dropout (D1-regularized)
            model = MPNNPredictor(
                node_features=2,
                hidden_dim=MPNN_HIDDEN_DIM,
                n_layers=MPNN_N_LAYERS,
                output_dim=self._n_params,
            )

            # Apply dropout to head layers (D1-regularized variant)
            # The MPNNPredictor already has Dropout(0.1) in its head by default.
            # We verify and potentially override:
            self._apply_dropout(model, DROPOUT)

            train_result = train_mpnn(
                model=model,
                dataset=dataset,
                n_epochs=MPNN_EPOCHS,
                lr=MPNN_LR,
                patience=MPNN_PATIENCE,
                seed=self._seed,
            )

            self._models[j2] = model
            self._datasets[j2] = dataset

            final_mse = train_result["final_mse"]
            n_epochs = len(train_result.get("mse_history", []))

            results_summary[str(j2)] = {
                "final_mse": float(final_mse),
                "n_epochs": n_epochs,
                "stopped_early": train_result.get("stopped_early", False),
                "n_train_graphs": len(dataset),
            }

            logger.info(
                f"    MSE: {final_mse:.2e}, epochs: {n_epochs}, "
                f"early_stop: {train_result.get('stopped_early', False)}"
            )

        return {
            "per_j2": results_summary,
            "dropout": DROPOUT,
            "pass": all(v["final_mse"] < 0.1 for v in results_summary.values()),
        }

    @staticmethod
    def _apply_dropout(model, dropout_rate: float):
        """Ensure dropout layers have the correct rate.

        MPNNPredictor already has Dropout(0.1) in its head. This method
        verifies and overwrites if needed.
        """
        import torch.nn as nn

        for _name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate

    # ── Section 3: Weight Gradient Analysis ──────────────────────────────────

    def section_gradient_analysis(self) -> dict:
        """Run WeightGradientAnalyzer on each trained MPNN.

        Computes ||dW/dh|| across the h-grid and identifies peaks.
        """
        from qmbp_simulation.analysis import WeightGradientAnalyzer

        if not self._models:
            raise RuntimeError("Section 2 must run first")

        results_summary = {}

        for j2 in self._j2_values:
            logger.info(f"\n  --- Gradient analysis for J₂={j2:.2f} ---")
            model = self._models[j2]
            dataset = self._datasets[j2]
            h_values = np.array([d.h_value for d in dataset])

            analyzer = WeightGradientAnalyzer(model)
            result = analyzer.analyze(dataset, h_values=h_values)

            self._gradient_results[j2] = result

            # Log findings
            peak_info = []
            for h_peak, mag in zip(result.peak_h_values, result.peak_magnitudes, strict=True):
                peak_info.append({"h": h_peak, "magnitude": mag})

            # Gradient norm statistics
            grad_max = float(np.max(result.total_gradient_norms))
            grad_mean = float(np.mean(result.total_gradient_norms))
            h_at_max = float(result.h_values[int(np.argmax(result.total_gradient_norms))])

            results_summary[str(j2)] = {
                "peaks_detected": len(result.peak_h_values),
                "peak_h_values": result.peak_h_values,
                "peak_magnitudes": result.peak_magnitudes,
                "h_at_max_gradient": h_at_max,
                "max_gradient_norm": grad_max,
                "mean_gradient_norm": grad_mean,
                "critical_region_detected": result.critical_region_detected,
                "per_layer_names": list(result.per_layer_gradient_norms.keys()),
            }

            if result.peak_h_values:
                primary_peak = result.peak_h_values[0]
                logger.info(
                    f"    Primary peak: h={primary_peak:.2f} (mag={result.peak_magnitudes[0]:.4f})"
                )
            else:
                logger.info(f"    No peaks detected (max gradient at h={h_at_max:.2f})")

            logger.info(f"    Grad norm: max={grad_max:.4f}, mean={grad_mean:.4f}")

        return {
            "per_j2": results_summary,
            "any_peaks_detected": any(v["peaks_detected"] > 0 for v in results_summary.values()),
            "pass": any(v["peaks_detected"] > 0 for v in results_summary.values()),
        }

    # ── Section 4: J₂-Dependence ─────────────────────────────────────────────

    def section_j2_dependence(self) -> dict:
        """Analyze how the gradient peak location varies with J₂.

        Key question: does the peak shift systematically with frustration?
        """
        if not self._gradient_results:
            raise RuntimeError("Section 3 must run first")

        peak_vs_j2 = []

        logger.info(f"  {'J₂':>4} | {'Peak h':>7} | {'Exact h_c':>9} | {'Δh':>5}")
        logger.info(f"  {'-' * 4}-+-{'-' * 7}-+-{'-' * 9}-+-{'-' * 5}")

        for j2 in self._j2_values:
            result = self._gradient_results[j2]
            exact_hc = self._exact_crossovers.get(j2, None)

            # Use primary peak if available, else h at max gradient
            if result.peak_h_values:
                peak_h = result.peak_h_values[0]
            else:
                peak_h = float(result.h_values[int(np.argmax(result.total_gradient_norms))])

            delta_h = abs(peak_h - exact_hc) if exact_hc is not None else None

            peak_vs_j2.append(
                {
                    "j2": j2,
                    "peak_h": peak_h,
                    "exact_crossover_h": exact_hc,
                    "delta_h": delta_h,
                }
            )

            delta_str = f"{delta_h:.2f}" if delta_h is not None else "N/A"
            exact_str = f"{exact_hc:.2f}" if exact_hc is not None else "N/A"
            logger.info(f"  {j2:>4.2f} | {peak_h:>7.2f} | {exact_str:>9} | {delta_str:>5}")

        # Check if peaks shift systematically
        peak_h_values = [p["peak_h"] for p in peak_vs_j2]
        j2_arr = np.array([p["j2"] for p in peak_vs_j2])

        # Monotonicity test: does peak_h decrease as J₂ increases?
        # (Frustration destabilizes the ordered phase → crossover shifts left)
        if len(peak_h_values) >= 3:
            # Compute Spearman correlation between J₂ and peak_h
            from scipy.stats import spearmanr

            corr, p_value = spearmanr(j2_arr, peak_h_values)
            systematic = p_value < 0.1  # Weak significance (small sample)
        else:
            corr = 0.0
            p_value = 1.0
            systematic = False

        logger.info(f"\n  Spearman correlation (J₂ vs peak_h): ρ={corr:.3f}, p={p_value:.3f}")
        logger.info(f"  Systematic shift detected: {systematic}")

        return {
            "peak_vs_j2": peak_vs_j2,
            "spearman_corr": float(corr),
            "spearman_p": float(p_value),
            "systematic_shift": systematic,
            "pass": True,  # This is informational — any result is valid
        }

    # ── Section 5: Exact Crossover Comparison ────────────────────────────────

    def section_exact_comparison(self) -> dict:
        """Compare gradient peaks against exact gap minimum (crossover point).

        The exact crossover is defined as the h-value where the gap is minimum.
        For standard TFIM (J₂=0), this is h_c=1.0. For frustrated TFIM, it
        shifts with J₂.

        Success criterion: gradient peak within Δh≤0.3 of exact crossover.
        This is the same tolerance used in D1-regularized validation.
        """
        if not self._gradient_results:
            raise RuntimeError("Section 3 must run first")

        TOLERANCE = 0.3  # Maximum |h_peak - h_crossover| for "agreement"

        comparisons = []

        for j2 in self._j2_values:
            result = self._gradient_results[j2]
            exact_hc = self._exact_crossovers.get(j2, None)
            vqe_data = self._vqe_data[j2]

            # Peak location
            if result.peak_h_values:
                peak_h = result.peak_h_values[0]
            else:
                peak_h = float(result.h_values[int(np.argmax(result.total_gradient_norms))])

            # Gap profile around crossover
            h_vals = vqe_data["h_values"]
            gaps = vqe_data["gaps"]
            gap_at_peak = float(np.interp(peak_h, h_vals, gaps))

            delta_h = abs(peak_h - exact_hc) if exact_hc is not None else None
            agrees = delta_h <= TOLERANCE if delta_h is not None else False

            comparisons.append(
                {
                    "j2": j2,
                    "peak_h_gradient": peak_h,
                    "exact_crossover_h": exact_hc,
                    "delta_h": delta_h,
                    "gap_at_peak": gap_at_peak,
                    "gap_at_crossover": float(np.min(gaps)),
                    "agrees_within_tolerance": agrees,
                }
            )

        # Summary
        n_agree = sum(1 for c in comparisons if c["agrees_within_tolerance"])
        agreement_rate = n_agree / len(comparisons)

        logger.info(f"\n  {'J₂':>4} | {'Peak h':>7} | {'Exact':>5} | {'Δh':>5} | {'Agree':>5}")
        logger.info(f"  {'-' * 4}-+-{'-' * 7}-+-{'-' * 5}-+-{'-' * 5}-+-{'-' * 5}")
        for c in comparisons:
            delta_str = f"{c['delta_h']:.2f}" if c["delta_h"] is not None else "N/A"
            exact_str = f"{c['exact_crossover_h']:.2f}" if c["exact_crossover_h"] else "N/A"
            agree_str = "✓" if c["agrees_within_tolerance"] else "✗"
            logger.info(
                f"  {c['j2']:>4.2f} | {c['peak_h_gradient']:>7.2f} | "
                f"{exact_str:>5} | {delta_str:>5} | {agree_str:>5}"
            )

        logger.info(
            f"\n  Agreement rate (Δh≤{TOLERANCE}): "
            f"{n_agree}/{len(comparisons)} = {agreement_rate:.0%}"
        )

        # The key scientific finding:
        # - If agreement_rate > 0.5: D1 generalizes to frustrated TFIM
        # - If agreement_rate == 0: D1 does NOT generalize (negative result)
        # Both are valid thesis contributions
        if agreement_rate >= 0.5:
            finding = (
                "D1 GENERALIZES: Weight gradient peaks track the frustrated "
                "TFIM crossover. Zero-QPU phase detection works for J₁-J₂ model."
            )
        else:
            finding = (
                "D1 DOES NOT GENERALIZE: Weight gradient peaks do not reliably "
                "track the frustrated TFIM crossover. The phase transition "
                "mechanism differs from standard TFIM."
            )
        logger.info(f"\n  FINDING: {finding}")

        return {
            "comparisons": comparisons,
            "agreement_rate": agreement_rate,
            "tolerance": TOLERANCE,
            "generalizes": agreement_rate >= 0.5,
            "finding": finding,
            "pass": True,  # Both outcomes are valid findings
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    D1FrustratedRunner.main()
