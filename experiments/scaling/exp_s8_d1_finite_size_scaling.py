"""S8: Finite-Size Scaling of h_c via Weight-Space Phase Detection (D1).

Hypothesis: The peak of ||dW/dh|| from the trained MPNN converges to h_c=1.0
as N→∞, following the finite-size scaling law:
    h_peak(N) = h_c + a·N^(-1/ν)
with ν=1 (TFIM 1D, Ising universality class).

Method:
    1. For each N in {4, 6, 8, 10}, train an MLP (D1-style) on full h-range
       VQE data with dropout=0.1 (regularized, as validated in D1-reg).
    2. Compute ||dθ/dh|| via finite differences at 50 probe points.
    3. Extract h_peak = argmax ||dθ/dh|| for each (N, seed).
    4. Fit h_peak(N) = 1.0 + a·N^(-1/ν) to extract ν.
    5. Compare ν with known TFIM value ν=1.

Expected outcome: ν ≈ 1.0 ± 0.2 (consistent with TFIM 1D Ising class).

Prior art:
    - D1 (ours): peaks at h≈0.7 (N=6), h≈0.6 (N=10) — never fitted as scaling.
    - Hernandes et al. (2025, arXiv:2503.17140): weight-space phase detection,
      but no finite-size scaling or ν extraction.
    - Differentiation: First extraction of ν from weight-space gradients of an
      MPNN trained for VQE.

Time estimate: ~15 min (4 sizes × 5 seeds × 6000 epochs MLP training).
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
H_TRAIN = np.arange(0.5, 2.55, 0.1)  # 25 points, step 0.1
H_PROBE = np.linspace(0.5, 2.5, 50)  # 50 probe points for gradient
EPSILON = 0.02  # Finite difference step for ||dθ/dh||
MLP_EPOCHS = 6000
MLP_HIDDEN = 128
MLP_DROPOUT = 0.1
MLP_LR = 1e-3
VQE_RESTARTS = 5
VQE_MAXITER = 300


def fss_model(N: np.ndarray, a: float, nu: float) -> np.ndarray:
    """Finite-size scaling model: h_peak(N) = 1.0 + a * N^(-1/nu)."""
    return 1.0 + a * N ** (-1.0 / nu)


class ExperimentS8(BaseExperiment):
    """Finite-size scaling of h_c from MPNN weight-space gradients."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="S8",
            category="S",
            description=(
                "Finite-size scaling of h_peak from ||dW/dh|| — "
                "extract critical exponent ν from weight-space gradients"
            ),
            hypothesis=(
                "h_peak(N) = h_c + a·N^(-1/ν) with ν ≈ 1.0 ± 0.2, "
                "consistent with TFIM 1D Ising universality class."
            ),
            system=SystemConfig(
                n_qubits=6,  # Overridden per-N in run_single
                p_layers=2,
                topology="chain_1d",
                J=1.0,
                h_values=sorted(H_TRAIN.tolist(), reverse=True),
                h_test=[],  # No deployment test — purely analytical experiment
            ),
            vqe=VQEConfig(
                n_restarts=VQE_RESTARTS,
                maxiter=VQE_MAXITER,
                ftol=1e-12,
            ),
            mpnn=MPNNConfig(
                hidden_dim=MLP_HIDDEN,
                n_layers=3,
                n_epochs=MLP_EPOCHS,
                lr=MLP_LR,
                dropout=MLP_DROPOUT,
                patience=500,
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
            "S8 setup: N_values=%s, seeds=%s, h_train=[%.1f, %.1f] (%d pts)",
            N_VALUES,
            self.config.seeds,
            H_TRAIN[0],
            H_TRAIN[-1],
            len(H_TRAIN),
        )

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Run D1-style analysis for all N values with given seed.

        For each N:
            1. Generate VQE θ_opt data via descending sweep
            2. Train regularized MLP (dropout=0.1)
            3. Compute ||dθ/dh|| and find peak
        Then fit finite-size scaling law across all N.
        """
        from qmbp_simulation import make_lattice

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
            theta_data = self._generate_vqe_data(N, n_params, seed)

            # Phase 2: Train regularized MLP
            logger.info(
                "  Seed %d, N=%d: training MLP (h=%d, %d epochs)...",
                seed,
                N,
                MLP_HIDDEN,
                MLP_EPOCHS,
            )
            model = self._train_mlp(theta_data, n_params, seed)

            # Phase 3: Compute ||dθ/dh|| and find peak
            grad_norms = self._compute_weight_gradients(model)
            peak_idx = int(np.argmax(grad_norms))
            h_peak = float(H_PROBE[peak_idx])
            h_peaks[N] = h_peak

            elapsed = time.time() - t0
            logger.info(
                "  Seed %d, N=%d: h_peak=%.3f, max||dθ/dh||=%.4f (%.1fs)",
                seed,
                N,
                h_peak,
                grad_norms[peak_idx],
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
                    "grad_norms": grad_norms.tolist(),
                    "h_probe": H_PROBE.tolist(),
                    "max_grad_norm": float(grad_norms[peak_idx]),
                    "known_h_c": 1.0,
                    "deviation_from_hc": h_peak - 1.0,
                },
            )
            metrics.append(m)

        # Fit finite-size scaling law: h_peak(N) = 1.0 + a * N^(-1/ν)
        fit_result = self._fit_scaling_law(h_peaks)

        # Store scaling fit as a summary metric
        summary = ExperimentMetrics(
            h_value=1.0,  # Known h_c
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

    def _generate_vqe_data(self, N: int, n_params: int, seed: int) -> np.ndarray:
        """Run VQE descending sweep to generate θ_opt training data.

        Parameters
        ----------
        N : int
            Number of qubits.
        n_params : int
            Number of variational parameters.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Shape (len(H_TRAIN), n_params) — optimized parameters per h.
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
        h_sorted_desc = np.sort(H_TRAIN)[::-1]
        h_to_idx = {round(float(h), 4): i for i, h in enumerate(H_TRAIN)}

        # Initialize with small random values (never zeros)
        prev_theta = rng.uniform(-0.01, 0.01, n_params)

        for h in h_sorted_desc:
            lattice = make_lattice("chain_1d", N, J=1.0, h=float(h))
            H = builder.build(lattice)

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

            idx = h_to_idx.get(round(float(h), 4))
            if idx is not None:
                theta_data[idx] = best_theta.copy()
            prev_theta = best_theta.copy()

        return theta_data

    def _train_mlp(self, theta_data: np.ndarray, n_params: int, seed: int):
        """Train a regularized MLP (h → θ) with dropout=0.1.

        Uses the same architecture as D1 but with dropout for robustness
        (validated in D1-reg: std=0.13 vs 0.90 without dropout).

        Parameters
        ----------
        theta_data : np.ndarray
            Shape (n_h_points, n_params) — VQE-optimized parameters.
        n_params : int
            Output dimension.
        seed : int
            Random seed for torch.

        Returns
        -------
        nn.Module
            Trained MLP model in eval mode.
        """
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)

        model = nn.Sequential(
            nn.Linear(1, MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(MLP_HIDDEN, MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(MLP_HIDDEN, n_params),
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR)
        loss_fn = nn.MSELoss()

        X = torch.tensor(H_TRAIN.reshape(-1, 1), dtype=torch.float32)
        Y = torch.tensor(theta_data, dtype=torch.float32)

        model.train()
        for _epoch in range(MLP_EPOCHS):
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            optimizer.step()

        model.eval()
        return model

    def _compute_weight_gradients(self, model) -> np.ndarray:
        """Compute ||dθ/dh|| at each probe point via finite differences.

        Parameters
        ----------
        model : nn.Module
            Trained MLP in eval mode.

        Returns
        -------
        np.ndarray
            Gradient norms at each H_PROBE point.
        """
        import torch

        grad_norms = np.zeros(len(H_PROBE))

        for i, h in enumerate(H_PROBE):
            h_plus = torch.tensor([[h + EPSILON]], dtype=torch.float32)
            h_minus = torch.tensor([[h - EPSILON]], dtype=torch.float32)

            with torch.no_grad():
                pred_plus = model(h_plus).numpy().flatten()
                pred_minus = model(h_minus).numpy().flatten()

            grad = (pred_plus - pred_minus) / (2 * EPSILON)
            grad_norms[i] = float(np.linalg.norm(grad))

        return grad_norms

    def _fit_scaling_law(self, h_peaks: dict[int, float]) -> dict:
        """Fit h_peak(N) = 1.0 + a * N^(-1/ν) to extract ν.

        Parameters
        ----------
        h_peaks : dict[int, float]
            Mapping N → h_peak for this seed.

        Returns
        -------
        dict
            Fit results with keys: nu, a, nu_std, a_std, residuals,
            fit_success, hypothesis_confirmed.
        """
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
            # Fit: h_peak(N) = 1.0 + a * N^(-1/nu)
            # Note: h_peak < h_c=1.0 from D1 data, so a is negative
            popt, pcov = curve_fit(
                fss_model,
                N_arr,
                h_arr,
                p0=[-0.5, 1.0],  # Initial guess: a<0, nu=1
                bounds=([-10.0, 0.1], [10.0, 5.0]),
                maxfev=10000,
            )
            perr = np.sqrt(np.diag(pcov))

            a_fit, nu_fit = popt
            a_std, nu_std = perr

            # Compute residuals
            h_predicted = fss_model(N_arr, a_fit, nu_fit)
            residuals = float(np.sqrt(np.mean((h_arr - h_predicted) ** 2)))

            result["a"] = float(a_fit)
            result["nu"] = float(nu_fit)
            result["a_std"] = float(a_std)
            result["nu_std"] = float(nu_std)
            result["residuals"] = residuals
            result["fit_success"] = True

            # Hypothesis confirmed if ν ∈ [0.8, 1.2] (i.e., ≈1.0 ± 0.2)
            result["hypothesis_confirmed"] = 0.8 <= nu_fit <= 1.2

        except (RuntimeError, ValueError) as e:
            logger.warning("  Scaling fit failed: %s", e)

        return result

    def analyze(self, results: dict[int, list[ExperimentMetrics]]) -> dict:
        """Aggregate scaling fits across all seeds.

        Computes median ν, inter-seed consistency, and overall verdict.
        """
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

        # Per-N statistics
        for N in N_VALUES:
            peaks = h_peaks_all[N]
            if peaks:
                analysis["h_peaks_per_N"][N] = {
                    "mean": float(np.mean(peaks)),
                    "std": float(np.std(peaks)),
                    "median": float(np.median(peaks)),
                    "values": peaks,
                }

        # ν statistics across seeds
        if nu_values:
            analysis["nu_median"] = float(np.median(nu_values))
            analysis["nu_mean"] = float(np.mean(nu_values))
            analysis["nu_std"] = float(np.std(nu_values))
            analysis["a_median"] = float(np.median(a_values))
            analysis["a_mean"] = float(np.mean(a_values))
            analysis["a_std"] = float(np.std(a_values))
            analysis["hypothesis_confirmed"] = 0.8 <= np.median(nu_values) <= 1.2
        else:
            analysis["nu_median"] = None
            analysis["hypothesis_confirmed"] = False

        return analysis

    def report(self, analysis: dict) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 60,
            "EXP-S8: Finite-Size Scaling of h_c via D1",
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
                    f"  Hypothesis (ν ∈ [0.8, 1.2]): "
                    f"{'CONFIRMED ✓' if analysis['hypothesis_confirmed'] else 'REJECTED ✗'}",
                ]
            )
        else:
            lines.append("  All fits failed — insufficient data or convergence issues.")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
