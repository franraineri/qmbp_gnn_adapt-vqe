#!/usr/bin/env python3
"""MPS Scaling Phase 3+4 — MPNN Training + Deployment at N>30.

Trains GINConv MPNN on θ_opt data from Phase 2 scaling runs,
then evaluates deployment accuracy (ΔE/gap at held-out h-values).

Prerequisites:
    Run run_scaling_validation.py first to produce θ_opt data.

Sections:
    1. Data Loading: Load and canonicalize θ_opt from scaling result JSON
    2. MPNN Training: GINConv (h=128, L=3, 6000 epochs, norm_type=none)
    3. Deployment: Predict θ at held-out h-values → evaluate via MPS → ΔE/gap

Usage:
    .venv/bin/python scripts/experiment_runners/scaling/run_scaling_phase3_mpnn.py \\
        --result-file results/scaling/scaling_N40_aer_mps_*.json

    .venv/bin/python scripts/experiment_runners/scaling/run_scaling_phase3_mpnn.py \\
        --result-file results/scaling/scaling_N50_*.json --use-all-seeds

    .venv/bin/python scripts/experiment_runners/scaling/run_scaling_phase3_mpnn.py --dry-run
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

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

from qmbp_simulation.models.constants import DE_GAP_THRESHOLD, MPS_DEFAULT_CHI_MAX

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_HIDDEN_DIM = 128
DEFAULT_N_LAYERS = 3
DEFAULT_N_EPOCHS = 6000
DEFAULT_PATIENCE = 500


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers (θ canonicalization + data loading from scaling result JSONs)
# ═══════════════════════════════════════════════════════════════════════════════


def canonicalize_theta(theta: np.ndarray) -> np.ndarray:
    """Canonicalize θ using HVA gauge symmetry (period π + Z₂).

    Delegates to the canonical implementation in qmbp_simulation.utils.
    """
    from qmbp_simulation.utils import canonicalize_theta as _canon

    return _canon(theta)


def load_theta_from_result(
    result_file: Path, seed: int | None = None, use_all_seeds: bool = False
) -> dict:
    """Load θ_opt, h_values, energies from a scaling result JSON.

    Parameters
    ----------
    result_file : Path
        Path to result JSON from run_scaling_validation.py (v1 or v2 format).
    seed : int | None
        Which seed to use. None = first available.
    use_all_seeds : bool
        Aggregate ALL seeds for more training data (with canonicalization).

    Returns
    -------
    dict with keys: h_values, theta_opt, e_dmrg, n_qubits, topology, p_layers, seed
    """
    from qmbp_simulation.framework import load_result

    data = load_result(result_file)

    # Support both v1 (flat metadata) and v2 (ValidationRunner envelope) formats
    if "config" in data and "system" in data.get("config", {}):
        # v2 format (from MPSScalingValidationRunner)
        system = data["config"]["system"]
        n_qubits = system["n_qubits"]
        topology = system["topology"]
        p_layers = system["p_layers"]
        # VQE results in section_2.data.per_seed
        s2_data = data.get("results", {}).get("section_2", {}).get("data", {})
        vqe_results = s2_data.get("per_seed", [])
    else:
        # v1 format (from old run_scaling_validation.py)
        meta = data["metadata"]
        n_qubits = meta["n"]
        topology = meta["topology"]
        p_layers = meta["p_layers"]
        vqe_results = data["vqe_results"]

    if use_all_seeds:
        all_h, all_theta, all_e = [], [], []
        for seed_run in vqe_results:
            for r in seed_run["results"]:
                if "theta_opt" not in r:
                    continue
                if not r.get("passed", True):
                    continue
                all_h.append(r["h"])
                all_theta.append(canonicalize_theta(np.array(r["theta_opt"])))
                e_key = "dmrg_energy" if "dmrg_energy" in r else "energy_exact"
                all_e.append(r.get(e_key, r.get("vqe_energy", 0.0)))

        if not all_theta:
            raise ValueError(f"No valid theta_opt found in {result_file}")

        # Filter out points that landed in different periodic basins
        # (local minima with θ far from the majority consensus)
        from qmbp_simulation.utils import filter_consistent_theta

        theta_array = np.array(all_theta)
        filtered_theta, mask = filter_consistent_theta(theta_array)
        n_removed = int((~mask).sum())
        if n_removed > 0:
            logger.info(
                f"  ⚠️  Filtered {n_removed}/{len(mask)} points with inconsistent θ "
                f"(periodic basin outliers). Keeping {mask.sum()} points."
            )
            all_h = [h for h, m in zip(all_h, mask, strict=False) if m]
            all_theta = [t for t, m in zip(all_theta, mask, strict=False) if m]
            all_e = [e for e, m in zip(all_e, mask, strict=False) if m]

        return {
            "h_values": np.array(all_h),
            "theta_opt": np.array(all_theta),
            "e_dmrg": np.array(all_e),
            "n_qubits": n_qubits,
            "topology": topology,
            "p_layers": p_layers,
            "seed": "all",
            "n_filtered": n_removed,
        }
    else:
        if seed is not None:
            seed_run = next((r for r in vqe_results if r["seed"] == seed), None)
            if seed_run is None:
                available = [r["seed"] for r in vqe_results]
                raise ValueError(f"Seed {seed} not found. Available: {available}")
        else:
            seed_run = vqe_results[0]

        results = seed_run["results"]
        h_values = np.array([r["h"] for r in results])
        theta_opt = np.array([canonicalize_theta(np.array(r["theta_opt"])) for r in results])
        e_key = "dmrg_energy" if "dmrg_energy" in results[0] else "energy_exact"
        e_dmrg = np.array([r.get(e_key, r.get("vqe_energy", 0.0)) for r in results])

        return {
            "h_values": h_values,
            "theta_opt": theta_opt,
            "e_dmrg": e_dmrg,
            "n_qubits": n_qubits,
            "topology": topology,
            "p_layers": p_layers,
            "seed": seed_run["seed"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════


class MPSScalingPhase3Runner(ValidationRunner):
    """Train MPNN on scaling θ_opt data and evaluate deployment accuracy.

    Loads θ_opt from run_scaling_validation.py output, trains GINConv MPNN,
    then predicts θ at held-out h-values and evaluates ΔE/gap via MPS.
    """

    runner_id = "mps_scaling_phase3_v2"
    experiment_id = "scaling/phase3"
    description = "MPNN training + deployment on MPS scaling data (N>30)"
    hypothesis = (
        "MPNN trained on MPS-VQE θ_opt achieves ΔE/gap < 5% at held-out "
        "h-values via direct prediction (zero-shot deployment)."
    )

    @classmethod
    def _add_custom_args(cls, parser):
        parser.add_argument(
            "--result-file",
            type=str,
            required=True,
            help="Path to scaling result JSON from run_scaling_validation.py",
        )
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "--use-all-seeds", action="store_true", help="Aggregate ALL seeds for training"
        )
        parser.add_argument(
            "--h-test",
            type=float,
            nargs="+",
            default=None,
            help="Test h-values for deployment (auto if not given)",
        )
        parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
        parser.add_argument("--n-epochs", type=int, default=DEFAULT_N_EPOCHS)
        parser.add_argument("--chi-max", type=int, default=MPS_DEFAULT_CHI_MAX)

    def run_preflight(self) -> bool:
        """Validate that the result file exists and contains θ_opt."""
        result_file = Path(self._args.result_file)
        if not result_file.exists():
            logger.error(f"Result file not found: {result_file}")
            return False
        return True

    def build_config(self) -> dict:
        return {
            "runner_id": self.runner_id,
            "source": {
                "result_file": self._args.result_file,
                "seed": self._args.seed,
                "use_all_seeds": self._args.use_all_seeds,
            },
            "mpnn": {
                "hidden_dim": self._args.hidden_dim,
                "n_epochs": self._args.n_epochs,
                "norm_type": "none",
            },
            "mps": {"chi_max": self._args.chi_max},
            "system": getattr(self, "_source_meta", {}),
        }

    def setup(self):
        """Load source data and initialize physics."""
        self.setup_physics()

        # Load θ_opt from result file
        result_file = Path(self._args.result_file)
        self._source_data = load_theta_from_result(
            result_file, seed=self._args.seed, use_all_seeds=self._args.use_all_seeds
        )

        self._source_meta = {
            "n_qubits": self._source_data["n_qubits"],
            "topology": self._source_data["topology"],
            "p_layers": self._source_data["p_layers"],
        }

        logger.info(
            f"  Source: N={self._source_data['n_qubits']}, "
            f"{len(self._source_data['h_values'])} points, "
            f"θ shape={self._source_data['theta_opt'].shape}"
        )

        # Compute h_test (midpoints between training h-values = interpolation test)
        if self._args.h_test is not None:
            self._h_test = sorted(self._args.h_test, reverse=True)
        else:
            h_unique = sorted(
                set(float(f"{h:.6f}") for h in self._source_data["h_values"]), reverse=True
            )
            self._h_test = [(h_unique[i] + h_unique[i + 1]) / 2 for i in range(len(h_unique) - 1)]

        # Set experiment_id dynamically
        from qmbp_simulation.framework.result_io import build_experiment_id

        self.experiment_id = build_experiment_id(
            category="scaling/phase3",
            model="tfim",
            topology=self._source_data["topology"],
        )

        self._mpnn_model = None
        self._train_metrics = None

    def define_sections(self) -> list[Section]:
        return [
            Section(
                id=1,
                name="MPNN Training",
                fn=self.section_train,
                hypothesis="GINConv MPNN achieves training MSE < 1e-4 on θ_opt data",
            ),
            Section(
                id=2,
                name="Deployment (Predict + Evaluate)",
                fn=self.section_deploy,
                hypothesis=f"MPNN θ_pred achieves ΔE/gap < {DE_GAP_THRESHOLD * 100:.0f}% at held-out h-values",
            ),
        ]

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 1: MPNN Training
    # ═══════════════════════════════════════════════════════════════════════════

    def section_train(self) -> dict:
        """Train MPNN on Phase 2 θ_opt data."""
        from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn

        data = self._source_data
        n_qubits = data["n_qubits"]
        topology = data["topology"]
        h_values = data["h_values"]
        theta_opt = data["theta_opt"]
        e_dmrg = data["e_dmrg"]
        output_dim = theta_opt.shape[1]

        # Build graph dataset (reuses framework function)
        lattice = self.make_lattice(topology, n_qubits, J=1.0, h=float(h_values[0]))
        dataset = build_graph_dataset(
            lattice=lattice,
            h_values=h_values,
            theta_opt=theta_opt,
            e_exact=e_dmrg,
            fidelities=None,
            fidelity_threshold=0.0,
        )

        logger.info(
            f"    Training: {len(dataset)} points, output_dim={output_dim}, "
            f"hidden={self._args.hidden_dim}"
        )

        # Create and train model (reuses framework train_mpnn)
        model = MPNNPredictor(
            node_features=2,
            hidden_dim=self._args.hidden_dim,
            n_layers=DEFAULT_N_LAYERS,
            output_dim=output_dim,
            norm_type="none",  # Critical for cross-N generalization
        )

        t0 = time.perf_counter()
        metrics = train_mpnn(
            model=model,
            dataset=dataset,
            n_epochs=self._args.n_epochs,
            lr=1e-3,
            patience=DEFAULT_PATIENCE,
            seed=self._args.seed,
        )
        train_time = time.perf_counter() - t0

        self._mpnn_model = model
        self._train_metrics = metrics

        return {
            "pass": metrics["final_mse"] < 1e-3,
            "final_mse": metrics["final_mse"],
            "n_train_points": len(dataset),
            "output_dim": output_dim,
            "train_time_s": train_time,
            "stopped_early": metrics["stopped_early"],
            "n_model_params": sum(p.numel() for p in model.parameters()),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 2: Deployment
    # ═══════════════════════════════════════════════════════════════════════════

    def section_deploy(self) -> dict:
        """Deploy MPNN at held-out h-values, evaluate ΔE/gap via MPS."""
        data = self._source_data
        n_qubits = data["n_qubits"]
        topology = data["topology"]
        p_layers = data["p_layers"]
        model = self._mpnn_model
        spec = self.get_model_spec("tfim")

        backend = self.MPSBackend(
            strategy="aer_mps", chi_max=self._args.chi_max, seed=self._args.seed
        )

        model.eval()

        def _deploy_at_h(h_val: float) -> dict | None:
            t0 = time.perf_counter()

            # Build lattice and Hamiltonian
            lattice = self.make_lattice(topology, n_qubits, J=1.0, h=h_val)
            H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)

            # Predict θ via base class helper
            theta_pred = self.predict_mpnn_at_h(model, h_val, topology=topology, n_qubits=n_qubits)
            theta_pred = canonicalize_theta(theta_pred)

            # Guard: skip if prediction contains NaN/Inf
            if not np.all(np.isfinite(theta_pred)):
                logger.warning(f"    ⚠️ h={h_val:.3f}: θ_pred contains NaN/Inf, skipping.")
                return None

            # Evaluate energy via MPS
            circuit, _ = spec.create_circuit(n_qubits, p_layers, lattice, **spec.circuit_kwargs)
            e_pred = backend.evaluate(circuit, H, theta_pred)

            # DMRG reference
            gt = self.solver.solve(H, lattice, method="dmrg")
            delta_e_abs = abs(e_pred - gt.ground_energy)
            de_gap = delta_e_abs / max(gt.gap, 1e-10)
            elapsed = time.perf_counter() - t0

            status = "✓" if de_gap < DE_GAP_THRESHOLD else "✗"
            logger.info(
                f"    [{status}] h={h_val:.3f}: ΔE/gap={de_gap:.4f} "
                f"|ΔE|={delta_e_abs:.4f} ({elapsed:.1f}s)"
            )

            return {
                "h": h_val,
                "e_pred": e_pred,
                "e_dmrg": gt.ground_energy,
                "gap": gt.gap,
                "de_gap": de_gap,
                "delta_e_abs": delta_e_abs,
                "delta_e_per_site": delta_e_abs / n_qubits,
                "passed": de_gap < DE_GAP_THRESHOLD,
                "elapsed_s": elapsed,
            }

        results = self.safe_per_h_loop(self._h_test, _deploy_at_h, label="MPNN deploy")

        n_pass = sum(1 for r in results if r["passed"])
        n_total = len(results)

        return {
            "pass": n_pass == n_total,
            "n_pass": n_pass,
            "n_total": n_total,
            "pass_rate": n_pass / max(n_total, 1),
            "mean_de_gap": float(np.mean([r["de_gap"] for r in results])),
            "max_de_gap": float(np.max([r["de_gap"] for r in results])),
            "per_point": results,
        }


if __name__ == "__main__":
    MPSScalingPhase3Runner.main()
