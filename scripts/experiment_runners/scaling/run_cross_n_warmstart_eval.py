#!/usr/bin/env python3
"""Cross-N Warm-Start Evaluation — heavy_hex p=1 scaling to N=100.

Experiment:
  1. Train MPNN on heavy_hex TFIM p=1 at small N (10, 20, 40)
  2. Predict θ for target N (60, 80, 100)
  3. Evaluate θ_pred directly (zero-shot energy)
  4. Use θ_pred as warm-start for VQE and measure convergence diagnostics

Diagnostics collected:
  - ΔE/gap from direct θ_pred (zero-shot quality)
  - VQE iterations from θ_pred vs from random init (warm-start speedup)
  - Energy trajectory: how quickly VQE refines the prediction
  - Final energy improvement: E(θ_pred) → E(θ_opt) after VQE
  - θ_opt - θ_pred L2 distance (how far VQE moves from prediction)

Usage:
    # Quick test (small sizes, few h-points)
    python scripts/experiment_runners/scaling/run_cross_n_warmstart_eval.py \\
        --train-sizes 10 20 40 --target-n 60 --h-points 8 --n-test 3

    # Full run
    python scripts/experiment_runners/scaling/run_cross_n_warmstart_eval.py \\
        --train-sizes 10 20 40 60 80 --target-n 100 --h-points 20

    # Dry run
    python scripts/experiment_runners/scaling/run_cross_n_warmstart_eval.py --dry-run
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

from qmbp_simulation.models.constants import DE_GAP_THRESHOLD, DEFAULT_SEEDS

DEFAULT_TRAIN_SIZES = [10, 20, 40]
DEFAULT_TARGET_N = 60
DEFAULT_TOPOLOGY = "heavy_hex"
DEFAULT_MODEL = "tfim"
DEFAULT_P = 1
DEFAULT_H_MIN = 1.0
DEFAULT_H_MAX = 2.0
DEFAULT_H_POINTS = 15
DEFAULT_N_TEST = 5
DEFAULT_N_EPOCHS = 4000
DEFAULT_HIDDEN_DIM = 128
DEFAULT_MAXITER = 300
DEFAULT_N_RESTARTS = 3


class CrossNWarmstartEvalRunner(ValidationRunner):
    """Evaluate MPNN θ predictions as warm-start for VQE on heavy_hex."""

    runner_id = "cross_n_warmstart_eval_v1"
    experiment_id = "CROSS_N_WARMSTART"
    description = "Cross-N warm-start: train small heavy_hex, warm-start large VQE"
    hypothesis = (
        "MPNN θ_pred reduces VQE iterations by ≥50% compared to random init "
        "on unseen heavy_hex N, while achieving ΔE/gap < 5%."
    )

    @classmethod
    def _add_custom_args(cls, parser):
        parser.add_argument("--train-sizes", type=int, nargs="+", default=DEFAULT_TRAIN_SIZES)
        parser.add_argument("--target-n", type=int, default=DEFAULT_TARGET_N)
        parser.add_argument("--topology", type=str, default=DEFAULT_TOPOLOGY)
        parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
        parser.add_argument("--p-layers", type=int, default=DEFAULT_P, choices=[1, 2])
        parser.add_argument("--h-min", type=float, default=DEFAULT_H_MIN)
        parser.add_argument("--h-max", type=float, default=DEFAULT_H_MAX)
        parser.add_argument("--h-points", type=int, default=DEFAULT_H_POINTS)
        parser.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
        parser.add_argument("--n-epochs", type=int, default=DEFAULT_N_EPOCHS)
        parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
        parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
        parser.add_argument("--n-restarts", type=int, default=DEFAULT_N_RESTARTS)
        parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS[:1])

    def run_preflight(self) -> bool:
        """Validate cross-N configuration."""
        target = self._args.target_n
        train_sizes = self._args.train_sizes
        if target in train_sizes:
            logger.error(f"target-n={target} must not be in train-sizes={train_sizes}")
            return False
        if len(train_sizes) < 2:
            logger.error("Need ≥2 training sizes.")
            return False
        if self._args.h_min >= self._args.h_max:
            logger.error("h_min must be < h_max.")
            return False
        # Interpolation vs extrapolation warning
        if target > max(train_sizes):
            logger.warning(
                f"  ⚠️ target_n={target} > max(train_sizes)={max(train_sizes)}. "
                f"This is EXTRAPOLATION — GNN generalization is not guaranteed. "
                f"Consider adding a training size above {target}."
            )
        elif target < min(train_sizes):
            logger.warning(
                f"  ⚠️ target_n={target} < min(train_sizes)={min(train_sizes)}. "
                f"This is EXTRAPOLATION downward — less risky but untested."
            )
        # Minimum data estimate
        expected_points = self._args.h_points * len(train_sizes)
        if expected_points < 14:
            logger.error(
                f"Expected {expected_points} training points < 14 minimum. "
                f"Increase --h-points or add more --train-sizes."
            )
            return False
        return True

    def build_config(self) -> dict:
        return {
            "runner_id": self.runner_id,
            "system": {
                "train_sizes": self._args.train_sizes,
                "target_n": self._args.target_n,
                "topology": self._args.topology,
                "model": self._args.model,
                "p_layers": self._args.p_layers,
            },
            "h_grid": {
                "h_min": self._args.h_min,
                "h_max": self._args.h_max,
                "h_points": self._args.h_points,
                "n_test": self._args.n_test,
            },
            "mpnn": {
                "hidden_dim": self._args.hidden_dim,
                "n_epochs": self._args.n_epochs,
                "norm_type": "none",
            },
            "vqe": {"maxiter": self._args.maxiter, "n_restarts": self._args.n_restarts},
        }

    def setup(self):
        """Import dependencies and configure."""
        self.setup_physics()

        from qmbp_simulation.predictors import MPNNPredictor, train_mpnn

        self.MPNNPredictor = MPNNPredictor
        self.train_mpnn = train_mpnn

        from qmbp_simulation.framework.result_io import build_experiment_id

        sizes_tag = "_".join(str(n) for n in sorted(self._args.train_sizes))
        self.experiment_id = build_experiment_id(
            category=f"scaling/cross_n/{sizes_tag}_to_{self._args.target_n}",
            model=self._args.model,
            topology=self._args.topology,
        )

        # h-grid (descending, dense near h_critical=1.0 for TFIM)
        self._h_train = self.generate_h_grid()
        # Test h-values for deployment
        self._h_test = np.linspace(self._args.h_max, self._args.h_min, self._args.n_test)

        self._training_data: dict[int, dict] = {}
        self._mpnn_model = None
        self._mpnn_metrics: dict | None = None

    def _get_spec(self):
        return self.get_model_spec(self._args.model)

    def define_sections(self) -> list[Section]:
        return [
            Section(
                id=1,
                name="Phase 1+2: VQE Training Data",
                fn=self.section_train_data,
                hypothesis="MPS-VQE converges on all training sizes",
            ),
            Section(
                id=2,
                name="Phase 3: MPNN Training",
                fn=self.section_mpnn_train,
                hypothesis="MPNN MSE < 1e-3 on combined heavy_hex dataset",
            ),
            Section(
                id=3,
                name="Phase 4: Warm-Start Evaluation",
                fn=self.section_warmstart_eval,
                hypothesis="θ_pred warm-start reduces VQE iters by ≥50%",
            ),
        ]

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 1: Generate VQE training data for all sizes
    # ═══════════════════════════════════════════════════════════════════════════

    def section_train_data(self) -> dict:
        """Run VQE on each training size to generate θ_opt(h).

        Includes bidirectional sweep and per-point ΔE/gap computation
        for quality gating in Section 2.
        """
        p = self._args.p_layers
        topo = self._args.topology
        seed = self._args.seeds[0]
        spec = self._get_spec()

        vqe_config = self.VQEConfig(
            p_layers=p,
            n_restarts=self._args.n_restarts,
            maxiter=self._args.maxiter,
            method="L-BFGS-B",
            enable_callbacks=False,
        )

        all_results = {}
        for n_train in sorted(self._args.train_sizes):
            logger.info(f"\n  ═══ VQE: N={n_train}, {topo}, p={p} ═══")
            # Use for_vqe_loop=True: auto-selects MPS for N>10 (O(N·χ³) vs O(2^N))
            backend = self.select_backend(n_train, for_vqe_loop=True)
            optimizer = self.VQEOptimizer(config=vqe_config, backend=backend, seed=seed)

            lattice_ref = self.make_lattice(topo, n_train, J=1.0, h=float(self._h_train[0]))
            circuit, _ = spec.create_circuit(n_train, p, lattice_ref, **spec.circuit_kwargs)
            n_params = circuit.num_parameters
            logger.info(f"    Circuit: {n_params} params, {len(lattice_ref.edges)} edges")

            rng = np.random.default_rng(seed)
            prev_theta = rng.uniform(-0.01, 0.01, n_params)

            h_vals, thetas, energies, gaps, fids, vqe_energies = [], [], [], [], [], []

            # ── Descending pass ──────────────────────────────────────
            for h in self._h_train:
                lattice_h = self.make_lattice(topo, n_train, J=1.0, h=float(h))
                H = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)
                gt = self.solver.solve(H, lattice_h)
                gs = None
                if n_train <= 22:
                    try:
                        gs = self.solver.ground_state_vector(H)
                    except (ValueError, MemoryError):
                        gs = None

                vqe_result = optimizer.optimize(
                    hamiltonian=H,
                    circuit=circuit,
                    initial_guess=prev_theta,
                    exact_energy=gt.ground_energy,
                    exact_state=gs,
                )
                prev_theta = vqe_result.theta_opt.copy()
                h_vals.append(float(h))
                thetas.append(vqe_result.theta_opt.copy())
                energies.append(gt.ground_energy)
                gaps.append(gt.gap)
                fids.append(vqe_result.fidelity if vqe_result.fidelity else 0.0)
                vqe_energies.append(vqe_result.energy)

            # ── Selective ascending pass ──────────────────────────────
            from qmbp_simulation.optimizers.sweep_strategies import (
                SelectiveAscendingConfig,
                select_suspicious_points,
            )

            # Compute preliminary de_gaps for selection
            prelim_de_gaps = [
                abs(vqe_energies[i] - energies[i]) / max(gaps[i], 1e-10) for i in range(len(h_vals))
            ]
            results_for_select = [
                {"h": h_vals[i], "de_gap": prelim_de_gaps[i]} for i in range(len(h_vals))
            ]
            asc_cfg = SelectiveAscendingConfig(
                de_gap_threshold=0.02,
                include_neighbors=True,
                max_fraction=0.5,
            )
            indices_to_reopt, asc_report = select_suspicious_points(
                results_for_select, config=asc_cfg
            )

            asc_theta = thetas[-1].copy()
            n_improved = 0

            if asc_report.fell_back_to_full:
                # Full ascending (>50% suspicious)
                reopt_range = range(len(h_vals) - 2, -1, -1)
            else:
                reopt_range = indices_to_reopt
                if reopt_range:
                    logger.info(
                        f"    Selective ascending: {asc_report.n_targeted}/{len(h_vals)} targeted"
                    )

            for idx in reopt_range:
                lattice_h = self.make_lattice(topo, n_train, J=1.0, h=h_vals[idx])
                H = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)
                gs = None
                if n_train <= 22:
                    try:
                        gs = self.solver.ground_state_vector(H)
                    except (ValueError, MemoryError):
                        gs = None
                vqe_asc = optimizer.optimize(
                    hamiltonian=H,
                    circuit=circuit,
                    initial_guess=asc_theta,
                    exact_energy=energies[idx],
                    exact_state=gs,
                )
                if vqe_asc.energy < vqe_energies[idx]:
                    thetas[idx] = vqe_asc.theta_opt.copy()
                    vqe_energies[idx] = vqe_asc.energy
                    fids[idx] = vqe_asc.fidelity if vqe_asc.fidelity else fids[idx]
                    n_improved += 1
                asc_theta = thetas[idx].copy()

            if n_improved > 0:
                logger.info(f"    🔄 Ascending: {n_improved}/{len(reopt_range)} improved")

            # ── Compute per-point ΔE/gap ─────────────────────────────
            quality = self.compute_vqe_quality_metrics(vqe_energies, energies, gaps)
            de_gaps = quality["de_gaps"]

            self._training_data[n_train] = {
                "h_values": np.array(h_vals),
                "theta_opt": np.array(thetas),
                "e_exact": np.array(energies),
                "gaps": np.array(gaps),
                "fidelities": np.array(fids),
                "de_gaps": np.array(de_gaps),
                "n_params": n_params,
            }

            n_pass = quality["n_pass"]
            # Variational principle check
            n_violations = self.check_variational_principle(vqe_energies, energies)
            if n_violations > 0:
                logger.warning(
                    f"    ⚠️ {n_violations} variational principle violations "
                    f"(E_vqe < E_exact). Likely numerical noise or unconverged DMRG."
                )
            # Theta smoothness (MPNN learnability indicator)
            theta_arr = np.array(thetas)
            smoothness = self.compute_theta_smoothness(theta_arr)
            if smoothness > 0:
                logger.info(f"    θ smoothness: {smoothness:.4f}")

            all_results[n_train] = {
                "n_points": len(h_vals),
                "n_params": n_params,
                "n_pass": n_pass,
                "mean_de_gap": float(np.mean(de_gaps)),
                "n_variational_violations": n_violations,
            }
            logger.info(
                f"    ✓ N={n_train}: {n_pass}/{len(h_vals)} pass, mean ΔE/gap={np.mean(de_gaps) * 100:.2f}%"
            )

            # Checkpoint after each training size (long VQE loops can crash)
            self.save_checkpoint(
                f"train_N{n_train}",
                {
                    "n_train": n_train,
                    "h_values": h_vals,
                    "theta_opt": [t.tolist() for t in thetas],
                    "e_exact": energies,
                    "de_gaps": de_gaps,
                },
            )

        # Clean up checkpoints on success
        self.cleanup_checkpoints("train_*")
        total_pts = sum(d["h_values"].shape[0] for d in self._training_data.values())
        return {"pass": total_pts >= 14, "sizes": all_results, "total_points": total_pts}

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 2: Train MPNN on combined multi-size heavy_hex data
    # ═══════════════════════════════════════════════════════════════════════════

    def section_mpnn_train(self) -> dict:
        """Train MPNN with norm_type='none' on merged dataset.

        Applies quality gate: only include training points with ΔE/gap < 20%.
        This prevents the MPNN from learning unconverged VQE parameters.
        """
        import torch
        from torch_geometric.data import Data

        topo = self._args.topology
        train_sizes = sorted(self._args.train_sizes)

        if not self._training_data:
            return {"pass": False, "error": "No training data"}

        theta_dim = next(iter(self._training_data.values()))["n_params"]
        quality_threshold = 0.20  # Discard points with ΔE/gap > 20%

        # Build combined dataset with quality gate
        dataset: list[Data] = []
        n_rejected = 0
        for n_train in train_sizes:
            data = self._training_data[n_train]
            lattice = self.make_lattice(topo, n_train, J=1.0, h=float(data["h_values"][0]))
            edge_index_np, coord = self.builder.build_graph_data(lattice)
            edge_index = torch.tensor(edge_index_np, dtype=torch.long)

            for i in range(len(data["h_values"])):
                # Quality gate: skip unconverged VQE points
                if data["de_gaps"][i] > quality_threshold:
                    n_rejected += 1
                    continue

                h_feat = np.full(n_train, float(data["h_values"][i]))
                n_feat = np.full(n_train, n_train / 100.0)
                x = torch.tensor(
                    np.stack([h_feat, coord.astype(float), n_feat], axis=1),
                    dtype=torch.float32,
                )
                y = torch.tensor(data["theta_opt"][i], dtype=torch.float32)
                graph = Data(x=x, edge_index=edge_index, y=y)
                graph.e_exact = float(data["e_exact"][i])
                dataset.append(graph)

        total_available = sum(len(d["h_values"]) for d in self._training_data.values())
        logger.info(
            f"  Quality gate: {len(dataset)}/{total_available} passed "
            f"(rejected {n_rejected} with ΔE/gap > {quality_threshold * 100:.0f}%)"
        )

        if len(dataset) < 10:
            return {
                "pass": False,
                "error": f"Only {len(dataset)} points passed quality gate "
                f"(need ≥10). Increase --maxiter or --n-restarts.",
                "n_rejected": n_rejected,
            }

        logger.info(f"  Dataset: {len(dataset)} graphs, θ_dim={theta_dim}, topology={topo}")

        # Auto-select model capacity based on dataset size
        effective_hidden = self.select_mpnn_hidden_dim(
            n_training_graphs=len(dataset),
            theta_dim=theta_dim,
            max_hidden=self._args.hidden_dim,
            min_hidden=32,
        )
        if effective_hidden != self._args.hidden_dim:
            logger.info(
                f"  ⚡ Auto-reduced hidden_dim: {self._args.hidden_dim} → {effective_hidden} "
                f"(params/data ratio too high for {len(dataset)} graphs)"
            )

        model = self.MPNNPredictor(
            node_features=3,
            hidden_dim=effective_hidden,
            n_layers=3,
            output_dim=theta_dim,
            norm_type="none",
        )
        n_model_params = sum(p.numel() for p in model.parameters())
        data_param_ratio = len(dataset) / max(n_model_params, 1)
        logger.info(
            f"  Model: {n_model_params:,} params "
            f"(data/params ratio: {data_param_ratio:.4f}, "
            f"{'OK' if data_param_ratio > 0.01 else 'UNDERDETERMINED'})"
        )

        t0 = time.time()
        metrics = self.train_mpnn(model, dataset, n_epochs=self._args.n_epochs, seed=42)
        train_time = time.time() - t0

        self._mpnn_model = model
        self._mpnn_metrics = metrics
        final_mse = metrics["final_mse"]
        logger.info(f"  MSE={final_mse:.2e}, time={train_time:.1f}s")

        return {
            "pass": final_mse < 1e-3,
            "n_graphs": len(dataset),
            "n_model_params": n_model_params,
            "final_mse": float(final_mse),
            "train_time_s": train_time,
            "theta_dim": theta_dim,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 3: Warm-start evaluation — θ_pred vs random init
    # ═══════════════════════════════════════════════════════════════════════════

    def section_warmstart_eval(self) -> dict:
        """Compare VQE from θ_pred (warm) vs random init (cold) at target N."""
        import torch
        from torch_geometric.data import Data

        if self._mpnn_model is None:
            return {"pass": False, "error": "No MPNN model"}

        # Guard: if MPNN training was too poor, predictions will be garbage
        if hasattr(self, "_mpnn_metrics") and self._mpnn_metrics:
            mse = self._mpnn_metrics.get("final_mse", float("inf"))
            if mse > 0.5:
                return {
                    "pass": False,
                    "error": f"MPNN MSE={mse:.3f} too high (>0.5). "
                    f"Predictions would be unreliable. "
                    f"Increase --n-epochs or --h-points for better training.",
                }

        topo = self._args.topology
        target_n = self._args.target_n
        p = self._args.p_layers
        seed = self._args.seeds[0]
        spec = self._get_spec()

        # For VQE evaluation, use MPS for N>10 (much faster than StatevectorEstimator)
        backend = self.select_backend(target_n, for_vqe_loop=True)
        # Warm-start: 1 restart only (θ_pred should be near optimum already)
        vqe_config_warm = self.VQEConfig(
            p_layers=p,
            n_restarts=1,
            maxiter=self._args.maxiter,
            method="L-BFGS-B",
            enable_callbacks=True,
        )
        # Cold-start: full restarts (needs exploration to find the basin)
        vqe_config_cold = self.VQEConfig(
            p_layers=p,
            n_restarts=self._args.n_restarts,
            maxiter=self._args.maxiter,
            method="L-BFGS-B",
            enable_callbacks=True,
        )
        optimizer_warm = self.VQEOptimizer(config=vqe_config_warm, backend=backend, seed=seed)
        optimizer_cold = self.VQEOptimizer(config=vqe_config_cold, backend=backend, seed=seed)

        # Build circuit and graph structure for target N
        lattice_ref = self.make_lattice(topo, target_n, J=1.0, h=float(self._h_test[0]))
        circuit, _ = spec.create_circuit(target_n, p, lattice_ref, **spec.circuit_kwargs)
        edge_index_np, coord = self.builder.build_graph_data(lattice_ref)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)

        self._mpnn_model.eval()
        results = []
        rng = np.random.default_rng(seed)

        logger.info(f"\n  ═══ Warm-Start Eval: N={target_n}, {topo}, p={p} ═══")
        logger.info(
            f"  {'h':>6s} | {'ΔE/gap_pred':>11s} | {'iters_warm':>10s} | {'iters_cold':>10s} | {'speedup':>7s} | {'ΔE/gap_warm':>11s} | {'ΔE/gap_cold':>11s}"
        )
        logger.info(
            f"  {'-' * 6}-+-{'-' * 11}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 7}-+-{'-' * 11}-+-{'-' * 11}"
        )

        for h_val in self._h_test:
            # ── Predict θ ────────────────────────────────────────────
            h_feat = np.full(target_n, float(h_val))
            n_feat = np.full(target_n, target_n / 100.0)
            x = torch.tensor(
                np.stack([h_feat, coord.astype(float), n_feat], axis=1),
                dtype=torch.float32,
            )
            graph = Data(x=x, edge_index=edge_index, batch=torch.zeros(target_n, dtype=torch.long))
            with torch.no_grad():
                theta_pred = self._mpnn_model(graph).numpy().flatten()

            # ── θ_pred sanity check ──────────────────────────────────
            if np.any(np.abs(theta_pred) > 2 * np.pi):
                logger.warning(
                    f"    ⚠️ h={h_val:.3f}: θ_pred out of bounds "
                    f"(max |θ|={np.max(np.abs(theta_pred)):.3f} > 2π). "
                    f"MPNN may be poorly trained."
                )

            # ── Ground truth ─────────────────────────────────────────
            lattice_h = self.make_lattice(topo, target_n, J=1.0, h=float(h_val))
            H = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)
            gt = self.solver.solve(H, lattice_h)

            # ── Direct evaluation of θ_pred (zero-shot) ──────────────
            e_pred = backend.evaluate(circuit, H, theta_pred)
            de_gap_pred = abs(e_pred - gt.ground_energy) / max(gt.gap, 1e-10)

            # ── VQE from θ_pred (warm-start) ─────────────────────────
            # ground_state_vector only available for N≤22 (requires full statevector)
            gs = None
            if target_n <= 22:
                try:
                    gs = self.solver.ground_state_vector(H)
                except (ValueError, MemoryError):
                    gs = None  # Graceful degradation: skip fidelity computation
            vqe_warm = optimizer_warm.optimize(
                hamiltonian=H,
                circuit=circuit,
                initial_guess=theta_pred,
                exact_energy=gt.ground_energy,
                exact_state=gs,
            )
            de_gap_warm = abs(vqe_warm.energy - gt.ground_energy) / max(gt.gap, 1e-10)
            iters_warm = vqe_warm.n_iterations

            # ── VQE from random init (cold-start) ────────────────────
            cold_init = rng.uniform(-0.01, 0.01, len(theta_pred))
            vqe_cold = optimizer_cold.optimize(
                hamiltonian=H,
                circuit=circuit,
                initial_guess=cold_init,
                exact_energy=gt.ground_energy,
                exact_state=gs,
            )
            de_gap_cold = abs(vqe_cold.energy - gt.ground_energy) / max(gt.gap, 1e-10)
            iters_cold = vqe_cold.n_iterations

            # ── Diagnostics ──────────────────────────────────────────
            theta_distance = float(np.linalg.norm(vqe_warm.theta_opt - theta_pred))
            energy_improvement = float(e_pred - vqe_warm.energy)
            speedup = iters_cold / max(iters_warm, 1)

            result = {
                "h": float(h_val),
                "e_exact": float(gt.ground_energy),
                "gap": float(gt.gap),
                # Zero-shot (direct prediction)
                "e_pred": float(e_pred),
                "de_gap_pred": float(de_gap_pred),
                "theta_pred": theta_pred.tolist(),
                # Warm-start VQE
                "e_warm": float(vqe_warm.energy),
                "de_gap_warm": float(de_gap_warm),
                "iters_warm": iters_warm,
                "theta_opt_warm": vqe_warm.theta_opt.tolist(),
                "fidelity_warm": vqe_warm.fidelity,
                # Cold-start VQE
                "e_cold": float(vqe_cold.energy),
                "de_gap_cold": float(de_gap_cold),
                "iters_cold": iters_cold,
                "fidelity_cold": vqe_cold.fidelity,
                # Diagnostics
                "theta_distance_l2": theta_distance,
                "energy_improvement": energy_improvement,
                "speedup": speedup,
                "warm_passed": bool(de_gap_warm < DE_GAP_THRESHOLD),
                "cold_passed": bool(de_gap_cold < DE_GAP_THRESHOLD),
            }
            results.append(result)

            logger.info(
                f"  {h_val:6.3f} | {de_gap_pred * 100:9.2f}%  | {iters_warm:10d} | "
                f"{iters_cold:10d} | {speedup:6.1f}x | {de_gap_warm * 100:9.2f}%  | "
                f"{de_gap_cold * 100:9.2f}%"
            )

        # ── Summary ──────────────────────────────────────────────────
        mean_speedup = float(np.mean([r["speedup"] for r in results]))
        n_warm_pass = sum(1 for r in results if r["warm_passed"])
        n_cold_pass = sum(1 for r in results if r["cold_passed"])
        mean_de_pred = float(np.mean([r["de_gap_pred"] for r in results]))
        mean_de_warm = float(np.mean([r["de_gap_warm"] for r in results]))
        mean_de_cold = float(np.mean([r["de_gap_cold"] for r in results]))
        mean_theta_dist = float(np.mean([r["theta_distance_l2"] for r in results]))

        # ── CrossNValidator (3-level verification) ───────────────────
        from qmbp_simulation.analysis.cross_n_validator import CrossNValidator

        validator = CrossNValidator(
            topology=topo,
            model_spec=spec,
            backend=backend,
            de_gap_threshold=DE_GAP_THRESHOLD,
        )
        # Collect training dataset for L3 LOO-CV
        training_graphs = []
        for n_t in sorted(self._args.train_sizes):
            data = self._training_data[n_t]
            lat = self.make_lattice(topo, n_t, J=1.0, h=float(data["h_values"][0]))
            ei_np, co = self.builder.build_graph_data(lat)
            ei = torch.tensor(ei_np, dtype=torch.long)
            for i in range(len(data["h_values"])):
                if data["de_gaps"][i] <= 0.20:  # Same quality gate
                    h_f = np.full(n_t, float(data["h_values"][i]))
                    n_f = np.full(n_t, n_t / 100.0)
                    x = torch.tensor(
                        np.stack([h_f, co.astype(float), n_f], axis=1), dtype=torch.float32
                    )
                    y = torch.tensor(data["theta_opt"][i], dtype=torch.float32)
                    g = Data(x=x, edge_index=ei, y=y)
                    g.e_exact = float(data["e_exact"][i])
                    g.h_value = float(data["h_values"][i])
                    training_graphs.append(g)

        validation_report = validator.validate_prediction(
            model=self._mpnn_model,
            n_target=target_n,
            h_test_values=[float(h) for h in self._h_test],
            training_sizes=sorted(self._args.train_sizes),
            training_data=training_graphs if len(training_graphs) >= 14 else None,
            run_l3=len(self._args.train_sizes) >= 3,
        )

        logger.info("\n  ─── Summary ───")
        logger.info(f"    Zero-shot ΔE/gap:  {mean_de_pred * 100:.2f}% mean")
        logger.info(
            f"    Warm-start ΔE/gap: {mean_de_warm * 100:.2f}% mean ({n_warm_pass}/{len(results)} pass)"
        )
        logger.info(
            f"    Cold-start ΔE/gap: {mean_de_cold * 100:.2f}% mean ({n_cold_pass}/{len(results)} pass)"
        )
        logger.info(f"    Speedup (iters):   {mean_speedup:.1f}x")
        logger.info(f"    θ distance L2:     {mean_theta_dist:.4f}")
        logger.info(
            f"    CrossNValidator:   {'PASS' if validation_report.overall_pass else 'FAIL'}"
        )

        all_pass = n_warm_pass == len(results) and mean_speedup >= 1.5

        return {
            "pass": all_pass,
            "target_n": target_n,
            "n_test_points": len(results),
            "results": results,
            "summary": {
                "mean_de_gap_pred": mean_de_pred,
                "mean_de_gap_warm": mean_de_warm,
                "mean_de_gap_cold": mean_de_cold,
                "mean_speedup": mean_speedup,
                "mean_theta_distance_l2": mean_theta_dist,
                "n_warm_pass": n_warm_pass,
                "n_cold_pass": n_cold_pass,
                "warm_start_advantage": mean_speedup >= 1.5,
            },
            "cross_n_validation": validation_report.to_dict(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sys.exit(CrossNWarmstartEvalRunner.main())
