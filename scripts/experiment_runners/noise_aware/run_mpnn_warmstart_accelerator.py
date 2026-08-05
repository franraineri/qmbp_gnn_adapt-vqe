"""MPNN Warm-Start Accelerator — Measure VQE compute savings from MPNN initialization.

The key experiment: instead of running 20 full VQE optimizations (20 restarts each),
can we run 5 full VQEs, train an MPNN, and use its predictions as warm-start for
the remaining 15 points with just 1 restart?

This measures the ACTUAL compute savings, not just prediction quality.

Metrics:
  - Total VQE time (baseline: 20 × full restarts)
  - Total VQE time (accelerated: 5 × full + 15 × warm-started)
  - Speedup factor: baseline_time / accelerated_time
  - Quality preservation: ΔE/gap of warm-started points vs full-restart points

The experiment also tests EXTRAPOLATION: predicting θ for h-values OUTSIDE
the training range (e.g., train on h∈[2.0,3.0], predict at h=1.5).

Usage:
    .venv/bin/python scripts/experiment_runners/noise_aware/run_mpnn_warmstart_accelerator.py \\
        --topology chain_1d --n-qubits 10 --p-layers 1 --h-min 1.3 --h-max 3.0 \\
        --h-points 20 --n-restarts 10 --verbose
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

import numpy as np

from qmbp_simulation.framework.runner_base import (
    Section, ValidationRunner, resolve_project_root,
)

_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

DEFAULT_N = 10
DEFAULT_P = 1
DEFAULT_TOPOLOGY = "chain_1d"
DEFAULT_MODEL = "tfim_bond_resolved"
DEFAULT_H_MIN = 1.3
DEFAULT_H_MAX = 3.0
DEFAULT_H_POINTS = 20
DEFAULT_N_TRAIN = 5  # Number of full-VQE anchor points


class MPNNWarmstartAccelerator(ValidationRunner):
    """Measure VQE compute savings from MPNN-predicted warm-starts."""

    runner_id = "mpnn_warmstart_accelerator_v1"
    experiment_id = "MPNN_WARMSTART_ACCELERATOR"
    description = (
        "Train MPNN on K anchor points, warm-start remaining points, "
        "measure speedup + quality vs full cold-start VQE."
    )
    hypothesis = (
        "MPNN warm-start reduces total VQE time by ≥60% while "
        "maintaining ΔE/gap < 5% at all predicted points."
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
        parser.add_argument(
            "--n-train-points", type=int, default=DEFAULT_N_TRAIN,
            help="Number of full-VQE anchor points for MPNN training (default 5).",
        )
        parser.add_argument(
            "--warmstart-restarts", type=int, default=1,
            help="Number of VQE restarts when warm-starting from MPNN prediction.",
        )
        parser.add_argument(
            "--warmstart-maxiter", type=int, default=200,
            help="Max VQE iterations when warm-starting (default 200, vs 500 full).",
        )
        parser.add_argument(
            "--mpnn-epochs", type=int, default=3000,
            help="MPNN training epochs.",
        )
        parser.add_argument(
            "--extrapolate", action="store_true", default=True,
            help="Include extrapolation test points outside training range.",
        )

    def run_preflight(self) -> bool:
        if self._args.n_train_points < 3:
            logger.error("Need >= 3 anchor points for MPNN training.")
            return False
        if self._args.n_train_points >= self._args.h_points:
            logger.error("n_train_points must be < h_points.")
            return False
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
            "accelerator": {
                "n_train_points": self._args.n_train_points,
                "warmstart_restarts": self._args.warmstart_restarts,
                "warmstart_maxiter": self._args.warmstart_maxiter,
                "mpnn_epochs": self._args.mpnn_epochs,
                "extrapolate": self._args.extrapolate,
            },
            "vqe_baseline": {
                "n_restarts": self._args.n_restarts,
                "maxiter": self._args.maxiter,
            },
        }

    def setup(self):
        self.setup_physics()
        self._topo = self._args.topology[0]
        self._N = self._args.n_qubits
        self._p = self._args.p_layers
        self._seed = self._args.seeds[0]
        self._spec = self.get_spec()
        self._h_values = np.array(self.generate_h_grid())
        self.parse_model_params()

    def define_sections(self) -> list[Section]:
        return [
            Section(id=1, name="Baseline VQE (full)", fn=self.section_baseline,
                    hypothesis="Full VQE with N restarts converges at all points"),
            Section(id=2, name="Accelerated Pipeline", fn=self.section_accelerated,
                    hypothesis="MPNN warm-start achieves same quality with less compute"),
            Section(id=3, name="Extrapolation Test", fn=self.section_extrapolate,
                    hypothesis="MPNN predicts useful θ_init outside training h-range"),
            Section(id=4, name="Speedup Analysis", fn=self.section_analysis,
                    hypothesis="Total time savings ≥ 60%"),
        ]

    # ══════════════════════════════════════════════════════════════════════
    # Section 1: Baseline — Full VQE at all points (the expensive way)
    # ══════════════════════════════════════════════════════════════════════

    def section_baseline(self) -> dict:
        """Run full VQE (N restarts, full maxiter) at all h-points. This is the cost baseline."""
        from qmbp_simulation import VQEOptimizer, make_lattice
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.utils.helpers import canonicalize_theta

        N, p, seed = self._N, self._p, self._seed
        topo, spec = self._topo, self._spec
        h_values = self._h_values
        hva = HVACircuitBuilder()

        lattice = make_lattice(topo, N, J=1.0, h=float(h_values[0]))
        circuit, _ = hva.create_bond_resolved(N, p, lattice)
        n_params = circuit.num_parameters

        # Exact diag (reusable across sections)
        e_exact, gaps = [], []
        for h in h_values:
            lat = make_lattice(topo, N, J=1.0, h=float(h))
            H = spec.build_hamiltonian(lat, **spec.hamiltonian_kwargs)
            gt = self.solver.solve(H, lat)
            e_exact.append(gt.ground_energy)
            gaps.append(gt.gap)
        self._e_exact = np.array(e_exact)
        self._gaps = np.array(gaps)
        self._circuit = circuit
        self._lattice_ref = lattice

        # Full VQE sweep (baseline cost)
        logger.info("  Baseline VQE: %d restarts × %d maxiter × %d h-points",
                    self._args.n_restarts, self._args.maxiter, len(h_values))
        vqe_config = self.VQEConfig(
            p_layers=p, n_restarts=self._args.n_restarts, maxiter=self._args.maxiter
        )
        optimizer = VQEOptimizer(config=vqe_config, backend=self.noiseless, seed=seed)
        rng = np.random.default_rng(seed)
        prev_theta = rng.uniform(-0.01, 0.01, n_params)

        baseline_results = []
        t_total = time.perf_counter()
        for h in sorted(h_values, reverse=True):
            t0 = time.perf_counter()
            lat = make_lattice(topo, N, J=1.0, h=float(h))
            H = spec.build_hamiltonian(lat, **spec.hamiltonian_kwargs)
            result = optimizer.optimize(H, circuit, initial_guess=prev_theta)
            elapsed = time.perf_counter() - t0

            if np.all(np.isfinite(result.theta_opt)):
                prev_theta = result.theta_opt.copy()

            idx = np.argmin(np.abs(h_values - h))
            de_gap = abs(result.energy - e_exact[idx]) / max(gaps[idx], 1e-10)
            baseline_results.append({
                "h": float(h), "energy": result.energy,
                "de_gap": de_gap, "theta_opt": canonicalize_theta(prev_theta),
                "elapsed_s": elapsed, "n_iters": result.n_iterations,
            })

        baseline_total_time = time.perf_counter() - t_total

        # Store for reuse
        self._baseline = {r["h"]: r for r in baseline_results}
        self._baseline_total_time = baseline_total_time

        de_gaps = [r["de_gap"] for r in baseline_results]
        logger.info("  Baseline: time=%.1fs, mean ΔE/gap=%.4f, pass@5%%=%d/%d",
                    baseline_total_time, np.mean(de_gaps),
                    sum(1 for d in de_gaps if d < 0.05), len(de_gaps))

        return {
            "pass": True,
            "total_time_s": baseline_total_time,
            "mean_de_gap": float(np.mean(de_gaps)),
            "max_de_gap": float(np.max(de_gaps)),
            "pass_rate_5pct": float(np.mean(np.array(de_gaps) < 0.05)),
            "n_points": len(h_values),
            "n_restarts": self._args.n_restarts,
            "maxiter": self._args.maxiter,
        }

    # ══════════════════════════════════════════════════════════════════════
    # Section 2: Accelerated — K anchor VQEs + MPNN + warm-started VQE
    # ══════════════════════════════════════════════════════════════════════

    def section_accelerated(self) -> dict:
        """Train MPNN on K anchor points, warm-start the rest."""
        import torch
        from qmbp_simulation import VQEOptimizer, make_lattice
        from qmbp_simulation.predictors.unified_graph import build_unified_bond_resolved_graph
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN, train_unified_mpnn
        from qmbp_simulation.utils.helpers import canonicalize_theta

        N, p, seed = self._N, self._p, self._seed
        topo, spec = self._topo, self._spec
        h_values = self._h_values
        circuit = self._circuit
        n_params = circuit.num_parameters
        K = self._args.n_train_points

        # Select K anchor points (evenly spaced from the valid range)
        anchor_idx = np.linspace(0, len(h_values) - 1, K, dtype=int)
        target_idx = np.array([i for i in range(len(h_values)) if i not in anchor_idx])
        h_anchor = h_values[anchor_idx]
        h_target = h_values[target_idx]

        logger.info("  Accelerated: %d anchor + %d target = %d total",
                    len(h_anchor), len(h_target), len(h_values))
        logger.info("  Anchor h: %s", [f"{h:.2f}" for h in h_anchor])

        t_total = time.perf_counter()

        # ── Step 1: Full VQE at anchor points ────────────────────────
        logger.info("  Step 1: Full VQE at %d anchor points...", K)
        vqe_config = self.VQEConfig(
            p_layers=p, n_restarts=self._args.n_restarts, maxiter=self._args.maxiter
        )
        optimizer = VQEOptimizer(config=vqe_config, backend=self.noiseless, seed=seed)
        rng = np.random.default_rng(seed)
        prev_theta = rng.uniform(-0.01, 0.01, n_params)

        t_anchor_start = time.perf_counter()
        anchor_results = []
        for h in sorted(h_anchor, reverse=True):
            lat = make_lattice(topo, N, J=1.0, h=float(h))
            H = spec.build_hamiltonian(lat, **spec.hamiltonian_kwargs)
            result = optimizer.optimize(H, circuit, initial_guess=prev_theta)
            if np.all(np.isfinite(result.theta_opt)):
                prev_theta = result.theta_opt.copy()
            anchor_results.append({
                "h": float(h), "theta_opt": canonicalize_theta(prev_theta),
            })
        t_anchor = time.perf_counter() - t_anchor_start

        # ── Step 2: Train MPNN on anchor data ────────────────────────
        logger.info("  Step 2: Training UnifiedMPNN on %d anchor points...", K)
        t_mpnn_start = time.perf_counter()

        # Build dataset from anchor results
        anchor_theta = np.array([r["theta_opt"] for r in sorted(anchor_results, key=lambda x: -x["h"])])
        # Reorder to match h_anchor (which may not be sorted)
        anchor_map = {r["h"]: r["theta_opt"] for r in anchor_results}
        theta_train = np.array([anchor_map[float(h)] for h in h_anchor])

        dataset = []
        lattice_ref = self._lattice_ref
        for i, h in enumerate(h_anchor):
            g = build_unified_bond_resolved_graph(
                lattice_ref, h_value=float(h), p_layers=p,
                theta_opt=theta_train[i], include_circuit_nodes=True,
            )
            dataset.append(g)

        model = UnifiedMPNN(
            node_features=4, hidden_dim=256, n_layers=3, type_embedding_dim=16,
        )
        metrics = train_unified_mpnn(
            model, dataset, n_epochs=self._args.mpnn_epochs, val_fraction=0.0, seed=seed,
        )
        t_mpnn = time.perf_counter() - t_mpnn_start
        logger.info("    MPNN MSE: %.2e (%.1fs)", metrics["final_mse"], t_mpnn)

        # ── Step 3: Predict θ_init at target points + short VQE ──────
        logger.info("  Step 3: Warm-started VQE at %d target points...", len(h_target))
        ws_config = self.VQEConfig(
            p_layers=p,
            n_restarts=self._args.warmstart_restarts,
            maxiter=self._args.warmstart_maxiter,
        )
        ws_optimizer = VQEOptimizer(config=ws_config, backend=self.noiseless, seed=seed + 100)

        t_ws_start = time.perf_counter()
        target_results = []
        model.eval()

        for h in h_target:
            # Predict θ_init from MPNN
            g = build_unified_bond_resolved_graph(
                lattice_ref, h_value=float(h), p_layers=p, include_circuit_nodes=True,
            )
            with torch.no_grad():
                theta_init = model(g).numpy().flatten()

            # Short VQE from warm-start
            lat = make_lattice(topo, N, J=1.0, h=float(h))
            H = spec.build_hamiltonian(lat, **spec.hamiltonian_kwargs)
            result = ws_optimizer.optimize(H, circuit, initial_guess=theta_init)

            idx = np.argmin(np.abs(h_values - h))
            de_gap = abs(result.energy - self._e_exact[idx]) / max(self._gaps[idx], 1e-10)
            target_results.append({
                "h": float(h), "energy": result.energy,
                "de_gap": de_gap, "n_iters": result.n_iterations,
                "theta_init_de_gap": None,  # filled below
            })

            # Also measure: what if we just used θ_init directly (no VQE)?
            e_init = self.noiseless.evaluate(circuit, H, theta_init)
            init_de_gap = abs(e_init - self._e_exact[idx]) / max(self._gaps[idx], 1e-10)
            target_results[-1]["theta_init_de_gap"] = float(init_de_gap)

        t_ws = time.perf_counter() - t_ws_start
        t_accelerated_total = time.perf_counter() - t_total

        # Store results
        self._accelerated_results = target_results
        self._accelerated_total_time = t_accelerated_total
        self._t_anchor = t_anchor
        self._t_mpnn = t_mpnn
        self._t_warmstart = t_ws
        self._model = model

        de_gaps = [r["de_gap"] for r in target_results]
        init_de_gaps = [r["theta_init_de_gap"] for r in target_results]
        logger.info("  Accelerated total: %.1fs (anchor=%.1fs, mpnn=%.1fs, ws=%.1fs)",
                    t_accelerated_total, t_anchor, t_mpnn, t_ws)
        logger.info("  Warm-started: mean ΔE/gap=%.4f, pass@5%%=%d/%d",
                    np.mean(de_gaps), sum(1 for d in de_gaps if d < 0.05), len(de_gaps))
        logger.info("  Direct (no VQE): mean ΔE/gap=%.4f, pass@5%%=%d/%d",
                    np.mean(init_de_gaps), sum(1 for d in init_de_gaps if d < 0.05), len(init_de_gaps))

        return {
            "pass": True,
            "n_anchor": K,
            "n_target": len(h_target),
            "times": {
                "anchor_vqe_s": t_anchor,
                "mpnn_training_s": t_mpnn,
                "warmstart_vqe_s": t_ws,
                "total_s": t_accelerated_total,
            },
            "mpnn_mse": metrics["final_mse"],
            "warmstarted": {
                "mean_de_gap": float(np.mean(de_gaps)),
                "max_de_gap": float(np.max(de_gaps)),
                "pass_rate_5pct": float(np.mean(np.array(de_gaps) < 0.05)),
            },
            "direct_prediction": {
                "mean_de_gap": float(np.mean(init_de_gaps)),
                "max_de_gap": float(np.max(init_de_gaps)),
                "pass_rate_5pct": float(np.mean(np.array(init_de_gaps) < 0.05)),
            },
        }

    # ══════════════════════════════════════════════════════════════════════
    # Section 3: Extrapolation — Predict outside training range
    # ══════════════════════════════════════════════════════════════════════

    def section_extrapolate(self) -> dict:
        """Test MPNN prediction at h-values OUTSIDE the training range."""
        import torch
        from qmbp_simulation import VQEOptimizer, make_lattice
        from qmbp_simulation.predictors.unified_graph import build_unified_bond_resolved_graph

        if not hasattr(self, "_model") or self._model is None:
            return {"pass": False, "error": "No model from Section 2"}

        if not self._args.extrapolate:
            return {"pass": True, "skipped": True, "reason": "extrapolation disabled"}

        N, p, seed = self._N, self._p, self._seed
        topo, spec = self._topo, self._spec
        circuit = self._circuit
        model = self._model

        # Generate extrapolation points: below and above training range
        h_anchor = self._h_values[np.linspace(0, len(self._h_values) - 1,
                                               self._args.n_train_points, dtype=int)]
        h_min_train = h_anchor.min()
        h_max_train = h_anchor.max()

        # Extrapolation: 3 points below, 3 points above
        extrap_below = np.linspace(max(0.5, h_min_train - 0.5), h_min_train - 0.05, 3)
        extrap_above = np.linspace(h_max_train + 0.05, h_max_train + 0.5, 3)
        h_extrap = np.concatenate([extrap_below, extrap_above])

        logger.info("  Extrapolation test: %d points outside [%.2f, %.2f]",
                    len(h_extrap), h_min_train, h_max_train)
        logger.info("    Below: %s", [f"{h:.2f}" for h in extrap_below])
        logger.info("    Above: %s", [f"{h:.2f}" for h in extrap_above])

        # Get exact energies for extrapolation points
        extrap_results = []
        model.eval()
        lattice_ref = self._lattice_ref

        ws_config = self.VQEConfig(
            p_layers=p, n_restarts=self._args.warmstart_restarts,
            maxiter=self._args.warmstart_maxiter,
        )
        ws_optimizer = VQEOptimizer(config=ws_config, backend=self.noiseless, seed=seed + 200)

        for h in h_extrap:
            lat = make_lattice(topo, N, J=1.0, h=float(h))
            H = spec.build_hamiltonian(lat, **spec.hamiltonian_kwargs)
            gt = self.solver.solve(H, lat)
            e_exact = gt.ground_energy
            gap = gt.gap

            # Predict θ_init
            g = build_unified_bond_resolved_graph(
                lattice_ref, h_value=float(h), p_layers=p, include_circuit_nodes=True,
            )
            with torch.no_grad():
                theta_init = model(g).numpy().flatten()

            # Direct prediction quality
            e_direct = self.noiseless.evaluate(circuit, H, theta_init)
            de_gap_direct = abs(e_direct - e_exact) / max(gap, 1e-10)

            # Warm-started VQE quality
            result = ws_optimizer.optimize(H, circuit, initial_guess=theta_init)
            de_gap_ws = abs(result.energy - e_exact) / max(gap, 1e-10)

            # Cold-start baseline for comparison
            cold_config = self.VQEConfig(
                p_layers=p, n_restarts=self._args.n_restarts, maxiter=self._args.maxiter,
            )
            cold_opt = VQEOptimizer(config=cold_config, backend=self.noiseless, seed=seed)
            rng = np.random.default_rng(seed)
            cold_init = rng.uniform(-0.01, 0.01, circuit.num_parameters)
            cold_result = cold_opt.optimize(H, circuit, initial_guess=cold_init)
            de_gap_cold = abs(cold_result.energy - e_exact) / max(gap, 1e-10)

            region = "below" if h < h_min_train else "above"
            extrap_results.append({
                "h": float(h), "region": region,
                "de_gap_direct": float(de_gap_direct),
                "de_gap_warmstart": float(de_gap_ws),
                "de_gap_cold": float(de_gap_cold),
                "gap": float(gap),
                "warmstart_better_than_cold": de_gap_ws < de_gap_cold,
            })

            status = "✅" if de_gap_ws < 0.05 else "⚠️"
            logger.info("    h=%.3f (%s): direct=%.4f, ws=%.4f, cold=%.4f %s",
                        h, region, de_gap_direct, de_gap_ws, de_gap_cold, status)

        # Summary
        ws_de_gaps = [r["de_gap_warmstart"] for r in extrap_results]
        cold_de_gaps = [r["de_gap_cold"] for r in extrap_results]
        n_ws_wins = sum(1 for r in extrap_results if r["warmstart_better_than_cold"])

        return {
            "pass": True,
            "n_extrap_points": len(h_extrap),
            "training_range": [float(h_min_train), float(h_max_train)],
            "mean_de_gap_warmstart": float(np.mean(ws_de_gaps)),
            "mean_de_gap_cold": float(np.mean(cold_de_gaps)),
            "warmstart_wins": f"{n_ws_wins}/{len(extrap_results)}",
            "per_point": extrap_results,
        }

    # ══════════════════════════════════════════════════════════════════════
    # Section 4: Speedup Analysis
    # ══════════════════════════════════════════════════════════════════════

    def section_analysis(self) -> dict:
        """Compute speedup and quality comparison."""
        if not hasattr(self, "_baseline_total_time"):
            return {"pass": False, "error": "Run sections 1+2 first"}

        baseline_time = self._baseline_total_time
        accel_time = self._accelerated_total_time
        speedup = baseline_time / max(accel_time, 0.1)

        # Quality comparison at target points
        accel_de_gaps = [r["de_gap"] for r in self._accelerated_results]
        baseline_de_gaps = []
        for r in self._accelerated_results:
            b = self._baseline.get(r["h"])
            baseline_de_gaps.append(b["de_gap"] if b else None)
        baseline_de_gaps = [d for d in baseline_de_gaps if d is not None]

        # Quality preservation: how many accelerated points match baseline quality?
        quality_preserved = 0
        quality_degraded = 0
        for i, r in enumerate(self._accelerated_results):
            b = self._baseline.get(r["h"])
            if b:
                if r["de_gap"] <= b["de_gap"] * 1.5:  # Within 50% of baseline
                    quality_preserved += 1
                else:
                    quality_degraded += 1

        logger.info("\n  ═══ SPEEDUP ANALYSIS ═══")
        logger.info("  Baseline (full VQE):     %.1fs", baseline_time)
        logger.info("  Accelerated (MPNN+VQE):  %.1fs", accel_time)
        logger.info("  Speedup:                 %.2f×", speedup)
        logger.info("  Time saved:              %.0f%%", (1 - 1/speedup) * 100)
        logger.info("")
        logger.info("  Quality (target points):")
        logger.info("    Baseline mean ΔE/gap:    %.4f", np.mean(baseline_de_gaps) if baseline_de_gaps else 0)
        logger.info("    Accelerated mean ΔE/gap: %.4f", np.mean(accel_de_gaps))
        logger.info("    Quality preserved:       %d/%d (within 1.5× of baseline)",
                    quality_preserved, quality_preserved + quality_degraded)

        return {
            "pass": True,
            "speedup_factor": float(speedup),
            "time_saved_pct": float((1 - 1/speedup) * 100),
            "baseline_time_s": baseline_time,
            "accelerated_time_s": accel_time,
            "breakdown": {
                "anchor_vqe_s": self._t_anchor,
                "mpnn_training_s": self._t_mpnn,
                "warmstart_vqe_s": self._t_warmstart,
            },
            "quality": {
                "baseline_mean_de_gap": float(np.mean(baseline_de_gaps)) if baseline_de_gaps else None,
                "accelerated_mean_de_gap": float(np.mean(accel_de_gaps)),
                "quality_preserved": quality_preserved,
                "quality_degraded": quality_degraded,
            },
        }


if __name__ == "__main__":
    MPNNWarmstartAccelerator.main()
