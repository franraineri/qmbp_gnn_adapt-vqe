#!/usr/bin/env python3
"""Noiseless Cross-N Generalization — MPNN predicts θ for unseen system sizes.

Pipeline:
  1. Phase 1+2 on N=6 (chain_1d p=1) → θ_opt(h) for 30 h-points
  2. Phase 1+2 on N=10 (chain_1d p=1) → θ_opt(h) for 30 h-points
  3. Merge datasets (60 graphs: 30 from N=6 + 30 from N=10)
  4. Train MPNN with norm_type="none" on the 60 combined graphs
  5. Create graph for N=8 at test h-points → MPNN predicts θ directly
  6. Evaluate E(θ_pred) with StatevectorEstimator on N=8 Hamiltonian

Constraints:
  - Same topology only (train chain_1d → predict chain_1d). No cross-topology.
  - norm_type="none" mandatory — BatchNorm destroys cross-N generalization.
  - Only p=1 validated for cross-N (for p≥2 with 2 params, scipy interp works).
  - Cross-N is valuable for bond-resolved (79 params) where interpolation fails.

Usage:
    # Default: train on N=6,10 → predict N=8, chain_1d, p=1, 30 h-points
    python scripts/experiment_runners/noiseless/run_noiseless_cross_n.py

    # Custom sizes
    python scripts/experiment_runners/noiseless/run_noiseless_cross_n.py \\
        --train-sizes 6 10 --target-n 8

    # Custom h-grid
    python scripts/experiment_runners/noiseless/run_noiseless_cross_n.py \\
        --h-min 0.5 --h-max 2.0 --h-points 30

    # With interpolation baseline comparison
    python scripts/experiment_runners/noiseless/run_noiseless_cross_n.py --with-interp
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

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

from qmbp_simulation.models.constants import DEFAULT_SEEDS

DEFAULT_TRAIN_SIZES = [6, 10]
DEFAULT_TARGET_N = 8
DEFAULT_P = 1
DEFAULT_TOPOLOGY = "chain_1d"
DEFAULT_MODEL = "tfim"
DEFAULT_H_MIN = 0.5
DEFAULT_H_MAX = 2.0
DEFAULT_H_POINTS = 30
DEFAULT_MAXITER = 500
DEFAULT_N_RESTARTS = 5
DEFAULT_HIDDEN_DIM = 128
DEFAULT_N_EPOCHS = 6000
DEFAULT_N_TEST_POINTS = 10


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════


class NoiselessCrossNRunner(ValidationRunner):
    """Noiseless Cross-N: train MPNN on multiple sizes, predict unseen size.

    Validates that GINConv with norm_type="none" can interpolate optimal
    VQE parameters across system sizes within the same topology.
    """

    runner_id = "noiseless_cross_n_v1"
    experiment_id = "NOISELESS_CROSS_N"
    description = "Cross-N generalization: train on N∈{6,10}, predict N=8"
    hypothesis = (
        "MPNN with norm_type='none' trained on N=6,10 chain_1d p=1 achieves "
        "ΔE/gap < 5% at all test h-points when predicting θ for N=8."
    )

    @classmethod
    def _add_custom_args(cls, parser):
        parser.add_argument(
            "--train-sizes",
            type=int,
            nargs="+",
            default=DEFAULT_TRAIN_SIZES,
            help="System sizes for training data (default: %(default)s)",
        )
        parser.add_argument(
            "--target-n",
            type=int,
            default=DEFAULT_TARGET_N,
            help="Target system size to predict (default: %(default)s)",
        )
        parser.add_argument(
            "--p-layers",
            type=int,
            default=DEFAULT_P,
            choices=[1, 2],
            help="HVA circuit depth (default: %(default)s)",
        )
        parser.add_argument(
            "--topology",
            type=str,
            default=DEFAULT_TOPOLOGY,
            help="Lattice topology (default: %(default)s)",
        )
        parser.add_argument(
            "--model",
            type=str,
            default=DEFAULT_MODEL,
            help="Model from registry (default: %(default)s)",
        )
        parser.add_argument(
            "--h-min",
            type=float,
            default=DEFAULT_H_MIN,
            help="Minimum h value (default: %(default)s)",
        )
        parser.add_argument(
            "--h-max",
            type=float,
            default=DEFAULT_H_MAX,
            help="Maximum h value (default: %(default)s)",
        )
        parser.add_argument(
            "--h-points",
            type=int,
            default=DEFAULT_H_POINTS,
            help="Number of h-points per training size (default: %(default)s)",
        )
        parser.add_argument(
            "--n-test-points",
            type=int,
            default=DEFAULT_N_TEST_POINTS,
            help="Number of h-test points for target evaluation (default: %(default)s)",
        )
        parser.add_argument(
            "--maxiter",
            type=int,
            default=DEFAULT_MAXITER,
            help="VQE optimizer maxiter per restart (default: %(default)s)",
        )
        parser.add_argument(
            "--n-restarts",
            type=int,
            default=DEFAULT_N_RESTARTS,
            help="VQE restarts per h-point (default: %(default)s)",
        )
        parser.add_argument(
            "--hidden-dim",
            type=int,
            default=DEFAULT_HIDDEN_DIM,
            help="MPNN hidden dimension (default: %(default)s)",
        )
        parser.add_argument(
            "--n-epochs",
            type=int,
            default=DEFAULT_N_EPOCHS,
            help="MPNN training epochs (default: %(default)s)",
        )
        parser.add_argument(
            "--with-interp",
            action="store_true",
            help="Include scipy interpolation baseline comparison",
        )
        parser.add_argument(
            "--seeds",
            type=int,
            nargs="+",
            default=DEFAULT_SEEDS[:1],
            help="VQE seed (default: %(default)s)",
        )

    def run_preflight(self) -> bool:
        """Validate cross-N configuration."""
        target = self._args.target_n
        train_sizes = self._args.train_sizes

        if target in train_sizes:
            logger.error(
                f"target-n={target} is in train-sizes={train_sizes}. Target must be an unseen size."
            )
            return False
        if len(train_sizes) < 2:
            logger.error("Need at least 2 training sizes for cross-N interpolation.")
            return False
        if self._args.h_min >= self._args.h_max:
            logger.error("h_min must be < h_max.")
            return False
        if self._args.topology != "chain_1d":
            logger.warning(
                f"Cross-N only validated for chain_1d. "
                f"Topology '{self._args.topology}' is experimental."
            )
        if self._args.p_layers > 1:
            logger.warning(
                "Cross-N only validated for p=1. For p≥2 with 2 params, "
                "scipy interpolation works equally well."
            )
        return True

    def build_config(self) -> dict:
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "system": {
                "train_sizes": self._args.train_sizes,
                "target_n": self._args.target_n,
                "p_layers": self._args.p_layers,
                "topology": self._args.topology,
                "model": self._args.model,
            },
            "h_grid": {
                "h_min": self._args.h_min,
                "h_max": self._args.h_max,
                "h_points": self._args.h_points,
                "n_test_points": self._args.n_test_points,
            },
            "vqe": {
                "maxiter": self._args.maxiter,
                "n_restarts": self._args.n_restarts,
                "method": "L-BFGS-B",
            },
            "mpnn": {
                "hidden_dim": self._args.hidden_dim,
                "n_epochs": self._args.n_epochs,
                "norm_type": "none",
                "n_layers": 3,
            },
            "backend": "NoiselessBackend (StatevectorEstimator)",
            "seeds": self._args.seeds,
            "with_interpolation_baseline": self._args.with_interp,
        }

    def setup(self):
        """Import heavy dependencies and initialize shared objects."""
        from qmbp_simulation import (
            ClassicalSolver,
            HamiltonianBuilder,
            HVACircuitBuilder,
            VQEConfig,
            VQEOptimizer,
            make_lattice,
        )
        from qmbp_simulation.execution import NoiselessBackend, select_backend
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.pipeline.dataset_io import generate_nonuniform_h_grid
        from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn

        self.builder = HamiltonianBuilder()
        self.solver = ClassicalSolver()
        self.hva = HVACircuitBuilder()
        self.make_lattice = make_lattice
        self.get_model_spec = get_model_spec
        self.generate_nonuniform_h_grid = generate_nonuniform_h_grid
        self.MPNNPredictor = MPNNPredictor
        self.build_graph_dataset = build_graph_dataset
        self.train_mpnn = train_mpnn
        self.NoiselessBackend = NoiselessBackend
        self.select_backend = select_backend
        self.VQEConfig = VQEConfig
        self.VQEOptimizer = VQEOptimizer

        # Minimum training points guard: preflight_cross_n requires ≥14 points.
        # With default h_points=30 × 2 sizes = 60, well above threshold.
        # Log explicit warning if user reduces parameters too aggressively.
        expected_total = self._args.h_points * len(self._args.train_sizes)
        if expected_total < 14:
            logger.warning(
                f"  ⚠️ Expected {expected_total} training points "
                f"(h_points={self._args.h_points} × {len(self._args.train_sizes)} sizes) "
                f"is below the minimum 14 required by preflight_cross_n."
            )

        # Set experiment_id dynamically
        sizes_tag = "_".join(str(n) for n in sorted(self._args.train_sizes))
        from qmbp_simulation.framework.result_io import build_experiment_id
        self.experiment_id = build_experiment_id(
            category="noiseless",
            model=f"cross_n_{sizes_tag}_to_{self._args.target_n}",
            topology=self._args.topology,
        )

        # Generate h-grid (descending for warm-start)
        self._h_train = self.generate_nonuniform_h_grid(
            h_min=self._args.h_min,
            h_max=self._args.h_max,
            n_points=self._args.h_points,
            h_critical=1.0,  # TFIM critical point
            dense_fraction=0.4,
            dense_radius=0.5,
        )

        # Test h-values: uniform grid within training range (for deployment)
        self._h_test = np.linspace(
            self._args.h_max,
            self._args.h_min,
            self._args.n_test_points,
        )

        # Storage for cross-section data flow
        self._training_data: dict[int, dict] = {}  # N -> {h, theta_opt, e_exact, fid}
        self._mpnn_model = None
        self._mpnn_metrics = None

    def _get_spec(self):
        """Get model spec."""
        return self.get_model_spec(self._args.model)

    def define_sections(self) -> list[Section]:
        return [
            Section(
                id=1,
                name="Phase 1+2: VQE on Training Sizes",
                fn=self.section_vqe_training_sizes,
                hypothesis=("Warm-start VQE achieves fidelity ≥ 0.99 on all training sizes"),
            ),
            Section(
                id=2,
                name="Phase 3: MPNN Training (Cross-N)",
                fn=self.section_mpnn_cross_n_train,
                hypothesis=(
                    "MPNN with norm_type='none' achieves training MSE < 1e-4 "
                    "on combined multi-size dataset"
                ),
            ),
            Section(
                id=3,
                name="Phase 4: Deploy on Target N",
                fn=self.section_deploy_target,
                hypothesis=(
                    "MPNN-predicted θ achieves ΔE/gap < 5% at all test h-points "
                    "on the unseen target size"
                ),
            ),
        ]

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 1: VQE on all training sizes (Phase 1+2 combined)
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _canonicalize_theta(theta: np.ndarray) -> np.ndarray:
        """Enforce theta_x > 0 sign convention (breaks MPNN sign ambiguity)."""
        if len(theta) == 0:
            return theta
        # For TFIM p=1: theta = [theta_zz, theta_x]. Canonicalize by theta_x > 0.
        if theta[-1] < 0:
            return -theta
        return theta

    def section_vqe_training_sizes(self) -> dict:
        """Run Phase 1+2 (ExactDiag + VQE) for each training size.

        Includes bidirectional sweep: descending then ascending pass,
        keeping the better result at each h-point. This reduces local
        minima trapping near the critical region.
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

        all_size_results = {}
        overall_pass = True

        for n_train in sorted(self._args.train_sizes):
            logger.info(f"\n  ═══ Phase 1+2: N={n_train}, {topo}, p={p} ═══")

            # Select appropriate backend for this size (MPS for N>10 in VQE loops)
            backend = self.select_backend(n_train, for_vqe_loop=True)
            optimizer = self.VQEOptimizer(config=vqe_config, backend=backend, seed=seed)

            # Build circuit once
            lattice_ref = self.make_lattice(topo, n_train, J=1.0, h=float(self._h_train[0]))
            circuit, _ = spec.create_circuit(n_train, p, lattice_ref, **spec.circuit_kwargs)
            n_params = circuit.num_parameters
            logger.info(f"    Circuit: {n_params} parameters")

            rng = np.random.default_rng(seed)
            prev_theta = rng.uniform(-0.01, 0.01, n_params)

            h_values = []
            theta_opts = []
            e_exacts = []
            fidelities = []
            de_gaps = []
            energies_vqe = []

            # ── Descending pass (h_max → h_min) ─────────────────────────
            for h in self._h_train:
                lattice_h = self.make_lattice(topo, n_train, J=1.0, h=float(h))
                H = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)
                gt = self.solver.solve(H, lattice_h)
                gs = self.solver.ground_state_vector(H)

                vqe_result = optimizer.optimize(
                    hamiltonian=H,
                    circuit=circuit,
                    initial_guess=prev_theta,
                    exact_energy=gt.ground_energy,
                    exact_state=gs,
                )

                de_gap = abs(vqe_result.energy - gt.ground_energy) / max(gt.gap, 1e-10)
                prev_theta = vqe_result.theta_opt.copy()

                h_values.append(float(h))
                theta_opts.append(vqe_result.theta_opt.copy())
                e_exacts.append(gt.ground_energy)
                fidelities.append(vqe_result.fidelity)
                de_gaps.append(de_gap)
                energies_vqe.append(vqe_result.energy)

            # ── Bidirectional ascending pass (h_min → h_max) ─────────────
            # Re-visit from the last point upward, keep better energy.
            asc_theta = theta_opts[-1].copy()
            n_improved = 0

            for idx in range(len(h_values) - 2, -1, -1):
                h = h_values[idx]
                lattice_h = self.make_lattice(topo, n_train, J=1.0, h=h)
                H = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)
                gs = self.solver.ground_state_vector(H)

                vqe_asc = optimizer.optimize(
                    hamiltonian=H,
                    circuit=circuit,
                    initial_guess=asc_theta,
                    exact_energy=e_exacts[idx],
                    exact_state=gs,
                )

                if vqe_asc.energy < energies_vqe[idx]:
                    theta_opts[idx] = vqe_asc.theta_opt.copy()
                    fidelities[idx] = vqe_asc.fidelity
                    energies_vqe[idx] = vqe_asc.energy
                    de_gaps[idx] = abs(vqe_asc.energy - e_exacts[idx]) / max(
                        self.solver.solve(H, lattice_h).gap, 1e-10
                    )
                    n_improved += 1

                asc_theta = theta_opts[idx].copy()

            if n_improved > 0:
                logger.info(f"    🔄 Bidirectional: {n_improved}/{len(h_values)} improved")

            # ── Canonicalize theta (sign convention for MPNN) ─────────────
            theta_opts = [self._canonicalize_theta(t) for t in theta_opts]

            # Store for cross-section use
            self._training_data[n_train] = {
                "h_values": np.array(h_values),
                "theta_opt": np.array(theta_opts),
                "e_exact": np.array(e_exacts),
                "fidelities": np.array(fidelities),
                "n_params": n_params,
            }

            # Summary
            mean_fid = float(np.mean(fidelities))
            min_fid = float(np.min(fidelities))
            n_pass = sum(1 for d in de_gaps if d < 0.05)
            size_pass = n_pass >= len(de_gaps) * 0.8

            all_size_results[n_train] = {
                "n_points": len(h_values),
                "n_params": n_params,
                "mean_fidelity": mean_fid,
                "min_fidelity": min_fid,
                "mean_de_gap": float(np.mean(de_gaps)),
                "max_de_gap": float(np.max(de_gaps)),
                "n_pass_5pct": n_pass,
                "pass": size_pass,
            }

            if not size_pass:
                overall_pass = False

            logger.info(
                f"    ✓ N={n_train}: {n_pass}/{len(de_gaps)} pass (ΔE/gap<5%), "
                f"mean F={mean_fid:.6f}, min F={min_fid:.6f}"
            )

        return {
            "pass": overall_pass,
            "sizes": all_size_results,
            "h_grid": self._h_train.tolist(),
            "total_training_points": sum(
                d["h_values"].shape[0] for d in self._training_data.values()
            ),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 2: MPNN Training on merged multi-size dataset
    # ═══════════════════════════════════════════════════════════════════════════

    def section_mpnn_cross_n_train(self) -> dict:
        """Train MPNN with norm_type='none' on combined multi-size dataset."""
        import torch
        from torch_geometric.data import Data

        topo = self._args.topology
        spec = self._get_spec()
        target_n = self._args.target_n
        train_sizes = sorted(self._args.train_sizes)

        if not self._training_data:
            return {"pass": False, "error": "No training data. Run Section 1 first."}

        # ── Preflight cross-N viability ──────────────────────────────────
        from qmbp_simulation.analysis.cross_n_validator import preflight_cross_n

        n_total_points = sum(d["h_values"].shape[0] for d in self._training_data.values())
        theta_dim = next(iter(self._training_data.values()))["n_params"]

        preflight_issues = preflight_cross_n(
            model=None,  # Not created yet
            topology_train=topo,
            topology_predict=topo,
            n_target=target_n,
            training_sizes=train_sizes,
            n_training_points=n_total_points,
            output_dim=theta_dim,
        )
        critical = [i for i in preflight_issues if "CRITICAL" in i]
        if critical:
            logger.error(f"Cross-N preflight FAILED: {critical}")
            return {"pass": False, "error": str(critical)}

        for issue in preflight_issues:
            logger.warning(f"  ⚠️ {issue}")

        # ── Build combined dataset with N/100 feature ────────────────────
        # Node features: [h_i, coord_i, N/100] — 3 features per node.
        # N/100 encodes system size (size-aware prediction).

        # First: check per-size theta smoothness and align if needed.
        from qmbp_simulation.analysis.theta_alignment import align_theta_array

        for n_train in train_sizes:
            data = self._training_data[n_train]
            theta_smoothness = float(np.max(np.abs(np.diff(data["theta_opt"], axis=0))))
            logger.info(f"  N={n_train}: θ smoothness = {theta_smoothness:.4f}")
            if theta_smoothness > 1.0:
                logger.info(f"    θ smoothness > 1.0 — running alignment pass for N={n_train}...")
                lattice_ref = self.make_lattice(topo, n_train, J=1.0, h=float(data["h_values"][0]))
                circuit_align, _ = spec.create_circuit(
                    n_train, self._args.p_layers, lattice_ref, **spec.circuit_kwargs
                )
                hamiltonians = []
                for h_val in data["h_values"]:
                    lat_h = self.make_lattice(topo, n_train, J=1.0, h=float(h_val))
                    hamiltonians.append(spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs))
                backend_align = self.select_backend(n_train, for_vqe_loop=True)
                aligned, report = align_theta_array(
                    theta_array=data["theta_opt"],
                    energies=np.array(
                        [
                            backend_align.evaluate(circuit_align, H, t)
                            for H, t in zip(hamiltonians, data["theta_opt"], strict=False)
                        ]
                    ),
                    circuit=circuit_align,
                    hamiltonians=hamiltonians,
                    backend=backend_align,
                )
                data["theta_opt"] = aligned
                logger.info(
                    f"    Aligned: smoothness {report.original_smoothness:.3f} "
                    f"→ {report.final_smoothness:.3f}"
                )

        dataset: list[Data] = []

        for n_train in train_sizes:
            data = self._training_data[n_train]
            lattice = self.make_lattice(topo, n_train, J=1.0, h=float(data["h_values"][0]))
            edge_index_np, coord = self.builder.build_graph_data(lattice)
            edge_index = torch.tensor(edge_index_np, dtype=torch.long)

            for i in range(len(data["h_values"])):
                h_feat = np.full(n_train, float(data["h_values"][i]))
                n_feat = np.full(n_train, n_train / 100.0)
                x = torch.tensor(
                    np.stack([h_feat, coord.astype(float), n_feat], axis=1),
                    dtype=torch.float32,
                )
                y = torch.tensor(data["theta_opt"][i], dtype=torch.float32)
                graph = Data(x=x, edge_index=edge_index, y=y)
                graph.e_exact = float(data["e_exact"][i])
                graph.h_value = float(data["h_values"][i])
                dataset.append(graph)

        logger.info(
            f"  Combined dataset: {len(dataset)} graphs "
            f"(from N={train_sizes}), θ_dim={theta_dim}, "
            f"node_features=3 [h, coord, N/100]"
        )

        # ── Create and train MPNN ────────────────────────────────────────
        model = self.MPNNPredictor(
            node_features=3,
            hidden_dim=self._args.hidden_dim,
            n_layers=3,
            output_dim=theta_dim,
            norm_type="none",
        )
        n_model_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Model: {n_model_params:,} parameters, norm_type='none'")

        t0 = time.time()
        metrics = self.train_mpnn(model, dataset, n_epochs=self._args.n_epochs, seed=42)
        train_time = time.time() - t0

        self._mpnn_model = model
        self._mpnn_metrics = metrics

        final_mse = metrics["final_mse"]
        converged = not metrics.get("stopped_early", False)
        mse_ok = final_mse < 1e-4

        logger.info(
            f"  Training complete: MSE={final_mse:.2e}, time={train_time:.1f}s, "
            f"converged={converged}"
        )

        # ── Re-run preflight with trained model ──────────────────────────
        post_issues = preflight_cross_n(
            model=model,
            topology_train=topo,
            topology_predict=topo,
            n_target=target_n,
            training_sizes=train_sizes,
            n_training_points=n_total_points,
            output_dim=theta_dim,
        )
        post_critical = [i for i in post_issues if "CRITICAL" in i]
        if post_critical:
            logger.error(f"Post-training preflight FAILED: {post_critical}")

        return {
            "pass": mse_ok and not post_critical,
            "n_training_graphs": len(dataset),
            "n_model_params": n_model_params,
            "final_mse": float(final_mse),
            "training_converged": converged,
            "train_time_s": train_time,
            "theta_dim": theta_dim,
            "preflight_issues": preflight_issues,
            "post_training_issues": post_issues,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 3: Deploy on target N (predict + evaluate)
    # ═══════════════════════════════════════════════════════════════════════════

    def section_deploy_target(self) -> dict:
        """Predict θ for unseen target_n and evaluate with StatevectorEstimator."""
        import torch
        from torch_geometric.data import Data

        if self._mpnn_model is None:
            return {"pass": False, "error": "No trained MPNN. Run Section 2 first."}

        topo = self._args.topology
        target_n = self._args.target_n
        p = self._args.p_layers
        spec = self._get_spec()

        # Backend for energy evaluation at target size
        noiseless = self.NoiselessBackend()

        # Build circuit for target N
        lattice_ref = self.make_lattice(topo, target_n, J=1.0, h=float(self._h_test[0]))
        circuit, _ = spec.create_circuit(target_n, p, lattice_ref, **spec.circuit_kwargs)

        # Prepare graph structure for target N
        edge_index_np, coord = self.builder.build_graph_data(lattice_ref)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)

        self._mpnn_model.eval()
        results_mpnn = []

        logger.info(f"\n  ═══ Deploy: MPNN → N={target_n}, {topo}, p={p} ═══")

        for h_val in self._h_test:
            # Build graph for prediction
            h_feat = np.full(target_n, float(h_val))
            n_feat = np.full(target_n, target_n / 100.0)
            x = torch.tensor(
                np.stack([h_feat, coord.astype(float), n_feat], axis=1),
                dtype=torch.float32,
            )
            graph = Data(
                x=x,
                edge_index=edge_index,
                batch=torch.zeros(target_n, dtype=torch.long),
            )

            # Predict θ
            with torch.no_grad():
                theta_pred = self._mpnn_model(graph).numpy().flatten()

            # Evaluate energy
            lattice_h = self.make_lattice(topo, target_n, J=1.0, h=float(h_val))
            H = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)
            e_pred = noiseless.evaluate(circuit, H, theta_pred)

            # Ground truth
            gt = self.solver.solve(H, lattice_h)
            de_gap = abs(e_pred - gt.ground_energy) / max(gt.gap, 1e-10)
            energy_error = float(e_pred - gt.ground_energy)

            result = {
                "h": float(h_val),
                "e_pred": float(e_pred),
                "e_exact": float(gt.ground_energy),
                "gap": float(gt.gap),
                "de_gap": float(de_gap),
                "energy_error": energy_error,
                "variational_ok": energy_error >= -1e-6,
                "theta_pred": theta_pred.tolist(),
                "passed": bool(de_gap < 0.05),
            }
            results_mpnn.append(result)

            status = "✅" if result["passed"] else "❌"
            var_flag = "" if result["variational_ok"] else " ⚠VAR"
            logger.info(
                f"    {status} h={h_val:.4f}: ΔE/gap={de_gap * 100:.2f}%"
                f"  E_pred={e_pred:.6f}  E_exact={gt.ground_energy:.6f}{var_flag}"
            )

        # ── Optional: interpolation baseline ─────────────────────────────
        results_interp = None
        if self._args.with_interp:
            results_interp = self._run_interpolation_baseline(
                circuit, spec, topo, target_n, noiseless
            )

        # ── Summary ──────────────────────────────────────────────────────
        de_gaps_mpnn = [r["de_gap"] for r in results_mpnn]
        n_pass_mpnn = sum(1 for r in results_mpnn if r["passed"])
        all_pass = n_pass_mpnn == len(results_mpnn)

        summary = {
            "pass": all_pass,
            "target_n": target_n,
            "n_test_points": len(results_mpnn),
            "mpnn": {
                "results": results_mpnn,
                "n_pass": n_pass_mpnn,
                "mean_de_gap": float(np.mean(de_gaps_mpnn)),
                "max_de_gap": float(np.max(de_gaps_mpnn)),
                "std_de_gap": float(np.std(de_gaps_mpnn)),
                "all_variational_ok": all(r["variational_ok"] for r in results_mpnn),
            },
        }

        if results_interp is not None:
            de_gaps_interp = [r["de_gap"] for r in results_interp]
            n_pass_interp = sum(1 for r in results_interp if r["passed"])
            summary["interpolation"] = {
                "results": results_interp,
                "n_pass": n_pass_interp,
                "mean_de_gap": float(np.mean(de_gaps_interp)),
                "max_de_gap": float(np.max(de_gaps_interp)),
                "std_de_gap": float(np.std(de_gaps_interp)),
            }
            winner = "mpnn" if np.mean(de_gaps_mpnn) < np.mean(de_gaps_interp) else "interpolation"
            summary["comparison_winner"] = winner
            logger.info(
                f"\n  ─── Comparison ───\n"
                f"    MPNN:          mean ΔE/gap={np.mean(de_gaps_mpnn) * 100:.3f}%, "
                f"pass={n_pass_mpnn}/{len(results_mpnn)}\n"
                f"    Interpolation: mean ΔE/gap={np.mean(de_gaps_interp) * 100:.3f}%, "
                f"pass={n_pass_interp}/{len(results_interp)}\n"
                f"    Winner: {winner}"
            )
        else:
            logger.info(
                f"\n  Summary: MPNN pass={n_pass_mpnn}/{len(results_mpnn)}, "
                f"mean ΔE/gap={np.mean(de_gaps_mpnn) * 100:.3f}%"
            )

        return summary

    # ═══════════════════════════════════════════════════════════════════════════
    # Interpolation Baseline (optional)
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_interpolation_baseline(self, circuit, spec, topo, target_n, noiseless):
        """Direct scipy interpolation baseline for comparison."""
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        logger.info(f"\n  ─── Interpolation Baseline: N={target_n} ───")

        # Collect all (h, N) -> theta pairs from training data
        points = []
        values = []
        for n_train, data in self._training_data.items():
            for i in range(len(data["h_values"])):
                points.append([float(data["h_values"][i]), n_train])
                values.append(data["theta_opt"][i])

        points = np.array(points)
        values = np.array(values)

        results = []
        for h_val in self._h_test:
            query = np.array([[float(h_val), target_n]])

            # Per-parameter interpolation
            theta_pred = np.zeros(values.shape[1])
            for col in range(values.shape[1]):
                interp = LinearNDInterpolator(points, values[:, col])
                val = interp(query).flatten()[0]
                if np.isnan(val):
                    nn = NearestNDInterpolator(points, values[:, col])
                    val = nn(query).flatten()[0]
                theta_pred[col] = val

            # Evaluate
            lattice_h = self.make_lattice(topo, target_n, J=1.0, h=float(h_val))
            H = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)
            e_pred = noiseless.evaluate(circuit, H, theta_pred)
            gt = self.solver.solve(H, lattice_h)
            de_gap = abs(e_pred - gt.ground_energy) / max(gt.gap, 1e-10)

            result = {
                "h": float(h_val),
                "e_pred": float(e_pred),
                "e_exact": float(gt.ground_energy),
                "gap": float(gt.gap),
                "de_gap": float(de_gap),
                "theta_pred": theta_pred.tolist(),
                "passed": bool(de_gap < 0.05),
            }
            results.append(result)

            status = "✅" if result["passed"] else "❌"
            logger.info(f"    {status} h={h_val:.4f}: ΔE/gap={de_gap * 100:.2f}% (interp)")

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    sys.exit(NoiselessCrossNRunner.main())
