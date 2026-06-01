"""S8b: Finite-Size Scaling of h_c via MPNN Weight-Space Gradients.

Hypothesis: The peak of ||dW/dh|| from the trained MPNN (GINConv) converges
to h_c=1.0 as N→∞, following h_peak(N) = h_c + a·N^(-1/ν) with ν≈1.

Motivation (from S8 failure):
    S8 used an MLP proxy (h→θ) which produced h_peak≈0.704 INDEPENDENT of N.
    The MLP doesn't encode system size — it sees the same scalar input h
    regardless of N. The peak was an architecture artifact.

    The real MPNN (GINConv) receives a GRAPH with N nodes as input.
    Each node has features [h_i, coordination_number_i]. The graph structure
    explicitly encodes system size, so the weight gradients should reflect
    finite-size effects in the physics.

Method:
    1. For each N in {4, 6, 8, 10}, generate VQE θ_opt data (descending sweep)
    2. Build graph dataset via build_graph_dataset (N nodes per graph)
    3. Train MPNNPredictor (GINConv, h=128, L=3, dropout=0.1)
    4. Use WeightGradientAnalyzer to compute ||dW/dh|| across h-sweep
    5. Extract h_peak = argmax ||dW/dh|| for each (N, seed)
    6. Fit h_peak(N) = 1.0 + a·N^(-1/ν) to extract ν

Expected outcome: ν ≈ 1.0 ± 0.3 (TFIM 1D Ising class).

Time estimate: ~20 min (4 sizes × 5 seeds × 6000 epochs MPNN training).
"""

from __future__ import annotations

import logging
import time

import numpy as np
from scipy.optimize import curve_fit, minimize

from qmbp_simulation.framework import BaseExperiment
from qmbp_simulation.framework.config import (
    AnalysisConfig,
    ExperimentConfig,
    MPNNConfig,
    SystemConfig,
    VQEConfig,
)
from qmbp_simulation.framework.metrics import ExperimentMetrics

logger = logging.getLogger(__name__)


# Configuration constants
N_VALUES = [4, 6, 8, 10]
H_TRAIN = np.arange(0.5, 2.55, 0.1)  # 21 points, step 0.1
H_PROBE = np.linspace(0.5, 2.5, 50)  # 50 probe points for gradient analysis
MPNN_EPOCHS = 6000
MPNN_HIDDEN = 128
MPNN_LAYERS = 3
MPNN_LR = 1e-3
MPNN_PATIENCE = 500
VQE_RESTARTS = 5
VQE_MAXITER = 300


def fss_model(N: np.ndarray, a: float, nu: float) -> np.ndarray:
    """Finite-size scaling model: h_peak(N) = 1.0 + a * N^(-1/nu)."""
    return 1.0 + a * N ** (-1.0 / nu)


class ExperimentS8b(BaseExperiment):
    """Finite-size scaling of h_c from real MPNN (GINConv) weight gradients."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="S8b",
            category="S",
            description=(
                "Finite-size scaling of h_peak from MPNN ||dW/dh|| — "
                "GINConv encodes N via graph structure (fixes S8 MLP failure)"
            ),
            hypothesis=(
                "h_peak(N) = h_c + a·N^(-1/ν) with ν ≈ 1.0 ± 0.3, "
                "consistent with TFIM 1D Ising universality class. "
                "MPNN graph input encodes N explicitly (unlike S8 MLP)."
            ),
            system=SystemConfig(
                n_qubits=6,  # Overridden per-N in run_single
                p_layers=2,
                topology="chain_1d",
                J=1.0,
                h_values=sorted(H_TRAIN.tolist(), reverse=True),
                h_test=[],  # No deployment — analytical experiment
            ),
            vqe=VQEConfig(
                n_restarts=VQE_RESTARTS,
                maxiter=VQE_MAXITER,
                ftol=1e-12,
            ),
            mpnn=MPNNConfig(
                hidden_dim=MPNN_HIDDEN,
                n_layers=MPNN_LAYERS,
                n_epochs=MPNN_EPOCHS,
                lr=MPNN_LR,
                dropout=0.1,
                patience=MPNN_PATIENCE,
            ),
            analysis=AnalysisConfig(
                scaling_n_values=N_VALUES,
                weight_gradient_n_points=50,
            ),
            seeds=[42, 43, 44, 45, 46],
            verbose=True,
            auto_warm_cold_comparison=False,
        )

    def setup(self) -> None:
        """Setup solver and circuit builder."""
        from qmbp_simulation import (
            ClassicalSolver,
            HamiltonianBuilder,
            HVACircuitBuilder,
        )

        self._setup_logging()
        self.builder = HamiltonianBuilder()
        self.solver = ClassicalSolver()
        self.hva = HVACircuitBuilder()
        logger.info(
            "S8b setup: N_values=%s, seeds=%s, MPNN(h=%d, L=%d, epochs=%d)",
            N_VALUES,
            self.config.seeds,
            MPNN_HIDDEN,
            MPNN_LAYERS,
            MPNN_EPOCHS,
        )

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Run MPNN-based weight gradient analysis for all N values.

        For each N:
            1. Generate VQE θ_opt data via descending sweep
            2. Build graph dataset (N nodes per graph — encodes system size)
            3. Train MPNNPredictor (GINConv)
            4. Use WeightGradientAnalyzer to find h_peak
        Then fit finite-size scaling law across all N.
        """
        from qmbp_simulation import make_lattice
        from qmbp_simulation.analysis import WeightGradientAnalyzer
        from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn

        metrics = []
        h_peaks: dict[int, float] = {}

        for N in N_VALUES:
            t0 = time.time()
            logger.info("  Seed %d, N=%d: generating VQE data...", seed, N)

            # Build circuit for this N
            lattice = make_lattice("chain_1d", N, J=1.0, h=1.0)
            circuit, _ = self.hva.create(N, self.config.system.p_layers, lattice)
            n_params = circuit.num_parameters

            # Phase 1: Generate VQE training data (descending sweep)
            theta_data, e_exact = self._generate_vqe_data(N, n_params, seed)

            # Phase 2: Build graph dataset (key difference from S8!)
            # Each graph has N nodes → MPNN "knows" system size
            logger.info("  Seed %d, N=%d: building graph dataset...", seed, N)
            graph_dataset = build_graph_dataset(
                lattice=lattice,
                h_values=H_TRAIN,
                theta_opt=theta_data,
                e_exact=e_exact,
                fidelities=None,  # Skip fidelity filter — full range needed
                fidelity_threshold=0.0,  # Accept all points
            )

            # Phase 3: Train real MPNN (GINConv)
            logger.info(
                "  Seed %d, N=%d: training MPNN (h=%d, L=%d, %d epochs)...",
                seed,
                N,
                MPNN_HIDDEN,
                MPNN_LAYERS,
                MPNN_EPOCHS,
            )
            model = MPNNPredictor(
                node_features=2,  # [h_i, coordination_number]
                hidden_dim=MPNN_HIDDEN,
                n_layers=MPNN_LAYERS,
                output_dim=n_params,
            )
            train_result = train_mpnn(
                model=model,
                dataset=graph_dataset,
                n_epochs=MPNN_EPOCHS,
                lr=MPNN_LR,
                patience=MPNN_PATIENCE,
                seed=seed,
            )
            final_mse = train_result["final_mse"]

            # Phase 4: Weight gradient analysis via WeightGradientAnalyzer
            logger.info(
                "  Seed %d, N=%d: analyzing weight gradients (MSE=%.2e)...",
                seed,
                N,
                final_mse,
            )
            analyzer = WeightGradientAnalyzer(model)

            # Build probe dataset at H_PROBE points for gradient analysis
            probe_dataset = self._build_probe_dataset(lattice, N)
            grad_result = analyzer.analyze(probe_dataset, h_values=H_PROBE)

            # Extract peak
            peak_idx = int(np.argmax(grad_result.total_gradient_norms))
            h_peak = float(H_PROBE[peak_idx])
            h_peaks[N] = h_peak

            elapsed = time.time() - t0
            logger.info(
                "  Seed %d, N=%d: h_peak=%.3f, max||dW/dh||=%.4f, critical_region=%s (%.1fs)",
                seed,
                N,
                h_peak,
                float(grad_result.total_gradient_norms[peak_idx]),
                grad_result.critical_region_detected,
                elapsed,
            )

            # Store per-N metrics
            m = ExperimentMetrics(
                h_value=h_peak,
                energy=0.0,
                exact_energy=0.0,
                energy_error=abs(h_peak - 1.0),
                gap=1.0,
                relative_error=abs(h_peak - 1.0),
                seed=seed,
                wall_time_s=elapsed,
                converged=True,
                technique_metadata={
                    "N": N,
                    "h_peak": h_peak,
                    "grad_norms": grad_result.total_gradient_norms.tolist(),
                    "h_probe": H_PROBE.tolist(),
                    "max_grad_norm": float(grad_result.total_gradient_norms[peak_idx]),
                    "critical_region_detected": grad_result.critical_region_detected,
                    "peak_h_values_detected": grad_result.peak_h_values,
                    "final_mse": final_mse,
                    "known_h_c": 1.0,
                    "deviation_from_hc": h_peak - 1.0,
                },
            )
            metrics.append(m)

        # Fit finite-size scaling law
        fit_result = self._fit_scaling_law(h_peaks)

        # Store scaling fit as summary metric
        summary = ExperimentMetrics(
            h_value=1.0,
            energy=0.0,
            exact_energy=0.0,
            energy_error=0.0,
            gap=1.0,
            relative_error=abs(fit_result["nu"] - 1.0) if fit_result["nu"] else 1.0,
            seed=seed,
            converged=fit_result["fit_success"],
            technique_metadata={
                "type": "scaling_fit",
                "h_peaks": {str(k): v for k, v in h_peaks.items()},
                "nu": fit_result["nu"],
                "a": fit_result["a"],
                "nu_std": fit_result["nu_std"],
                "a_std": fit_result["a_std"],
                "residuals": fit_result["residuals"],
                "fit_success": fit_result["fit_success"],
                "hypothesis_confirmed": fit_result["hypothesis_confirmed"],
            },
        )
        metrics.append(summary)

        logger.info(
            "  Seed %d RESULT: ν=%.3f±%.3f, a=%.3f±%.3f | %s",
            seed,
            fit_result["nu"] or 0,
            fit_result["nu_std"] or 0,
            fit_result["a"] or 0,
            fit_result["a_std"] or 0,
            "CONFIRMED" if fit_result["hypothesis_confirmed"] else "REJECTED",
        )

        return metrics

    def _build_probe_dataset(self, lattice, N: int):
        """Build a probe dataset at H_PROBE points for gradient analysis.

        Creates Data objects with the correct graph structure but dummy y targets.
        The WeightGradientAnalyzer needs forward+backward passes, so y is required.
        """
        import torch
        from torch_geometric.data import Data

        from qmbp_simulation import HamiltonianBuilder

        builder = HamiltonianBuilder()
        # Get graph structure from lattice
        edge_index_np, coord = builder.build_graph_data(lattice)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)

        n_params = self.config.system.p_layers * 2  # HVA p=2 → 4 params

        probe_data = []
        for h in H_PROBE:
            # Node features: [h_i, coordination_number_i]
            h_feat = np.full(N, float(h))
            x = torch.tensor(
                np.stack([h_feat, coord.astype(float)], axis=1),
                dtype=torch.float32,
            )
            # Dummy target (needed for loss.backward in analyzer)
            y = torch.zeros(n_params, dtype=torch.float32)

            data = Data(x=x, edge_index=edge_index, y=y)
            data.h_value = float(h)
            probe_data.append(data)

        return probe_data

    def _generate_vqe_data(self, N: int, n_params: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Run VQE descending sweep to generate θ_opt training data.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (theta_data [n_h, n_params], e_exact [n_h])
        """
        from qmbp_simulation import HamiltonianBuilder, make_lattice
        from qmbp_simulation.execution import NoiselessBackend

        rng = np.random.default_rng(seed)
        builder = HamiltonianBuilder()
        backend = NoiselessBackend()

        # Build circuit for this N
        lattice_ref = make_lattice("chain_1d", N, J=1.0, h=1.0)
        circuit, _ = self.hva.create(N, self.config.system.p_layers, lattice_ref)

        theta_data = np.zeros((len(H_TRAIN), n_params))
        e_exact = np.zeros(len(H_TRAIN))
        h_sorted_desc = np.sort(H_TRAIN)[::-1]
        h_to_idx = {round(float(h), 4): i for i, h in enumerate(H_TRAIN)}

        # Initialize with small random values (never zeros)
        prev_theta = rng.uniform(-0.01, 0.01, n_params)

        for h in h_sorted_desc:
            lattice = make_lattice("chain_1d", N, J=1.0, h=float(h))
            H = builder.build(lattice)

            # Get exact energy for dataset
            sol = self.solver.solve(H, lattice)
            idx = h_to_idx.get(round(float(h), 4))
            if idx is not None:
                e_exact[idx] = sol.ground_energy

            def cost_fn(params, _H=H):
                return backend.evaluate(circuit, _H, params)

            best_energy = float("inf")
            best_theta = prev_theta.copy()

            for restart in range(VQE_RESTARTS):
                if restart == 0:
                    x0 = prev_theta.copy()
                else:
                    x0 = prev_theta + rng.normal(0, 0.1, n_params)
                x0 = np.clip(x0, -np.pi, np.pi)

                result = minimize(
                    cost_fn,
                    x0,
                    method="L-BFGS-B",
                    bounds=[(-np.pi, np.pi)] * n_params,
                    options={"maxiter": VQE_MAXITER, "ftol": 1e-12},
                )
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_theta = result.x.copy()

            if idx is not None:
                theta_data[idx] = best_theta.copy()
            prev_theta = best_theta.copy()

        return theta_data, e_exact

    def _fit_scaling_law(self, h_peaks: dict[int, float]) -> dict:
        """Fit h_peak(N) = 1.0 + a * N^(-1/ν) to extract ν."""
        N_arr = np.array(sorted(h_peaks.keys()), dtype=float)
        h_arr = np.array([h_peaks[int(n)] for n in N_arr])

        result = {
            "nu": None,
            "a": None,
            "nu_std": None,
            "a_std": None,
            "residuals": None,
            "fit_success": False,
            "hypothesis_confirmed": False,
        }

        try:
            popt, pcov = curve_fit(
                fss_model,
                N_arr,
                h_arr,
                p0=[-0.5, 1.0],
                bounds=([-10.0, 0.1], [10.0, 5.0]),
                maxfev=10000,
            )
            perr = np.sqrt(np.diag(pcov))

            a_fit, nu_fit = popt
            a_std, nu_std = perr

            h_predicted = fss_model(N_arr, a_fit, nu_fit)
            residuals = float(np.sqrt(np.mean((h_arr - h_predicted) ** 2)))

            result["a"] = float(a_fit)
            result["nu"] = float(nu_fit)
            result["a_std"] = float(a_std)
            result["nu_std"] = float(nu_std)
            result["residuals"] = residuals
            result["fit_success"] = True
            # Wider acceptance: ν ∈ [0.7, 1.3]
            result["hypothesis_confirmed"] = 0.7 <= nu_fit <= 1.3

        except (RuntimeError, ValueError) as e:
            logger.warning("  Scaling fit failed: %s", e)

        return result

    def analyze(self, results: dict[int, list[ExperimentMetrics]]) -> dict:
        """Aggregate scaling fits across all seeds."""
        nu_values = []
        a_values = []
        h_peaks_all: dict[int, list[float]] = {N: [] for N in N_VALUES}

        for _seed, seed_metrics in results.items():
            for m in seed_metrics:
                meta = m.technique_metadata or {}
                if meta.get("type") == "scaling_fit" and meta.get("fit_success"):
                    nu_values.append(meta["nu"])
                    a_values.append(meta["a"])
                elif "N" in meta and "h_peak" in meta:
                    N = meta["N"]
                    h_peaks_all[N].append(meta["h_peak"])

        analysis = {
            "n_seeds": len(self.config.seeds),
            "n_successful_fits": len(nu_values),
            "h_peaks_per_N": {},
        }

        for N in N_VALUES:
            peaks = h_peaks_all[N]
            if peaks:
                analysis["h_peaks_per_N"][N] = {
                    "mean": float(np.mean(peaks)),
                    "std": float(np.std(peaks)),
                    "median": float(np.median(peaks)),
                    "values": peaks,
                }

        if nu_values:
            analysis["nu_median"] = float(np.median(nu_values))
            analysis["nu_mean"] = float(np.mean(nu_values))
            analysis["nu_std"] = float(np.std(nu_values))
            analysis["a_median"] = float(np.median(a_values))
            analysis["a_mean"] = float(np.mean(a_values))
            analysis["a_std"] = float(np.std(a_values))
            analysis["hypothesis_confirmed"] = 0.7 <= np.median(nu_values) <= 1.3
        else:
            analysis["nu_median"] = None
            analysis["hypothesis_confirmed"] = False

        # Compare with S8 (MLP) — document improvement
        analysis["comparison_with_s8"] = {
            "s8_median_h_peak_all_N": 0.704,
            "s8_n_dependence": False,
            "s8b_shows_n_dependence": self._check_n_dependence(h_peaks_all),
        }

        return analysis

    def _check_n_dependence(self, h_peaks_all: dict[int, list[float]]) -> bool:
        """Check if h_peak shows statistically significant N-dependence."""
        medians = []
        for N in N_VALUES:
            peaks = h_peaks_all.get(N, [])
            if peaks:
                medians.append(np.median(peaks))
        if len(medians) < 3:
            return False
        # N-dependence if range of medians > 0.05 (beyond noise)
        return (max(medians) - min(medians)) > 0.05

    def report(self, analysis: dict) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 60,
            "EXP-S8b: Finite-Size Scaling via MPNN (GINConv)",
            "=" * 60,
            "",
            f"Seeds: {self.config.seeds}",
            f"N values: {N_VALUES}",
            f"Successful fits: {analysis['n_successful_fits']}/{analysis['n_seeds']}",
            "",
            "--- Per-N h_peak statistics ---",
        ]

        for N in N_VALUES:
            stats = analysis["h_peaks_per_N"].get(N, {})
            if stats:
                lines.append(
                    f"  N={N:2d}: h_peak = {stats['mean']:.3f} ± {stats['std']:.3f} "
                    f"(median={stats['median']:.3f})"
                )

        lines.append("")
        lines.append("--- Scaling fit: h_peak(N) = 1.0 + a·N^(-1/ν) ---")

        if analysis.get("nu_median") is not None:
            lines.extend(
                [
                    f"  ν = {analysis['nu_median']:.3f} (median across seeds)",
                    f"  ν = {analysis['nu_mean']:.3f} ± {analysis['nu_std']:.3f} (mean±std)",
                    f"  a = {analysis['a_median']:.3f} (median)",
                    "",
                    "  Known TFIM 1D: ν = 1.0",
                    f"  Hypothesis (ν ∈ [0.7, 1.3]): "
                    f"{'CONFIRMED ✓' if analysis['hypothesis_confirmed'] else 'REJECTED ✗'}",
                ]
            )
        else:
            lines.append("  All fits failed — insufficient data or convergence.")

        # Comparison with S8
        comp = analysis.get("comparison_with_s8", {})
        lines.extend(
            [
                "",
                "--- Comparison with S8 (MLP proxy) ---",
                "  S8 (MLP): h_peak ≈ 0.704 for ALL N (no N-dependence)",
                f"  S8b (MPNN): N-dependence detected = {comp.get('s8b_shows_n_dependence')}",
            ]
        )

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
