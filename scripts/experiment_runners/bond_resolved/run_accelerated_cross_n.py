#!/usr/bin/env python3
"""Accelerated Cross-N Pipeline — Train N_train, Predict N_target via zoo.

Integrates AcceleratedVQE + model_zoo + QualityPredictor for a complete
bond-resolved cross-N transfer workflow with data reuse:

1. Quality check: is the target config viable?
2. Train: AcceleratedVQE at N_train (or load from zoo if exists)
3. Zoo export: auto-register the trained model
4. Cross-N predict: evaluate at N_target without VQE
5. Analysis: per-h ΔE/gap breakdown + comparison vs full VQE

Supports multiple p values and topologies. Results are saved as JSON
for downstream analysis.

Usage:
    # Default: Train N=10, predict N=20, p=1, chain_1d
    .venv/bin/python scripts/.../run_accelerated_cross_n.py

    # Custom sizes
    .venv/bin/python scripts/.../run_accelerated_cross_n.py --train-n 10 --target-n 20 40

    # Use existing model from zoo (skip training)
    .venv/bin/python scripts/.../run_accelerated_cross_n.py --from-zoo --target-n 20

    # Multiple p layers
    .venv/bin/python scripts/.../run_accelerated_cross_n.py --p-layers 1 2

    # Dry run
    .venv/bin/python scripts/.../run_accelerated_cross_n.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import UTC
from pathlib import Path

import numpy as np

from qmbp_simulation.framework.result_io import FRUSTRATED_NAMESPACE
from qmbp_simulation.framework.runner_base import (
    Section,
    ValidationRunner,
    resolve_project_root,
)

_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


# Defaults
DEFAULT_TRAIN_N = 10
DEFAULT_TARGET_N = [20]
DEFAULT_P = 1
DEFAULT_TOPOLOGY = "chain_1d"
DEFAULT_H_MIN = 2.0
DEFAULT_H_MAX = 4.5
DEFAULT_H_POINTS = 15
DEFAULT_N_ANCHORS = 14
DEFAULT_MAXITER = 1000
DEFAULT_N_RESTARTS = 6
# Hard floor on VQE restarts. Canonical source is models.constants.MIN_N_RESTARTS,
# re-exported here for local use. Enforced on --n-restarts (setup) AND inside the
# adaptive allocation (sweep_strategies), so even easy points get ≥ this many.
from qmbp_simulation.models.constants import MIN_N_RESTARTS  # noqa: E402

DEFAULT_MAX_REFINE_PER_ITER = 50

FINE_TUNE_EPOCHS = 500
FULL_TRAIN_EPOCHS = 2000


class AcceleratedCrossNRunner(ValidationRunner):
    """Accelerated Cross-N Transfer: train small, predict large.

    Trains UnifiedMPNN at N_train using AcceleratedVQE (5-6 anchor VQE +
    MPNN for the rest), exports to zoo, then predicts at N_target using
    only the trained model. No VQE at N_target.
    """

    runner_id = "accelerated_cross_n_v1"
    experiment_id = "ACCEL_CROSS_N"
    description = "Accelerated Cross-N: train N_train, predict N_target via zoo"

    # N_MAX_VIABLE per topology (dual criterion, prevents extrapolation contamination)
    N_MAX_VIABLE = {
        "chain_1d": 300,
        "heavy_hex": 300,
        "square": 300,
        "ladder": 1000,
        "triangular": 12,  # N>=14 is ansatz-limited (0% dual pass with p=1)
    }
    hypothesis = (
        "UnifiedMPNN trained on N_train bond-resolved data predicts θ at "
        "N_target with ΔE/gap < 10% for h in valid regime (h > 2.0)."
    )

    @classmethod
    def _add_custom_args(cls, parser):
        parser.add_argument(
            "--train-n",
            type=int,
            default=DEFAULT_TRAIN_N,
            help="System size for training (default: %(default)s)",
        )
        parser.add_argument(
            "--target-n",
            type=int,
            nargs="+",
            default=DEFAULT_TARGET_N,
            help="Target system size(s) for prediction (default: %(default)s)",
        )
        parser.add_argument(
            "--p-layers",
            type=int,
            nargs="+",
            default=[DEFAULT_P],
            help="HVA layer depth(s) (default: %(default)s)",
        )
        parser.add_argument(
            "--topology",
            type=str,
            default=DEFAULT_TOPOLOGY,
            help="Lattice topology (default: %(default)s)",
        )
        parser.add_argument(
            "--h-min",
            type=float,
            default=DEFAULT_H_MIN,
            help="Minimum h for sweep (default: %(default)s)",
        )
        parser.add_argument(
            "--h-max",
            type=float,
            default=DEFAULT_H_MAX,
            help="Maximum h for sweep (default: %(default)s)",
        )
        parser.add_argument(
            "--h-points",
            type=int,
            default=DEFAULT_H_POINTS,
            help="Number of h-grid points (default: %(default)s)",
        )
        parser.add_argument(
            "--train-h-min",
            type=float,
            default=None,
            help="If set, restrict TRAINING data to h >= this value "
            "(filters MultiNAggregator dataset; independent of the eval sweep --h-min).",
        )
        parser.add_argument(
            "--train-h-max",
            type=float,
            default=None,
            help="If set, restrict TRAINING data to h <= this value "
            "(filters MultiNAggregator dataset; independent of the eval sweep --h-max).",
        )
        parser.add_argument(
            "--n-anchors",
            type=int,
            default=DEFAULT_N_ANCHORS,
            help="Number of VQE anchor points (default: %(default)s)",
        )
        parser.add_argument(
            "--g",
            type=float,
            default=0.0,
            help="Longitudinal field strength for the TFIM + longitudinal model "
            "(H = -J·ZZ - h·X - g·Z). Only meaningful with "
            "--model tfim_bond_resolved_longitudinal. Default 0.0 (standard TFIM).",
        )
        parser.add_argument(
            "--maxiter",
            type=int,
            default=DEFAULT_MAXITER,
            help="VQE COBYLA maxiter (default: %(default)s)",
        )
        parser.add_argument(
            "--n-restarts",
            type=int,
            default=DEFAULT_N_RESTARTS,
            help="VQE restarts per anchor (default: %(default)s)",
        )
        parser.add_argument(
            "--from-zoo",
            action="store_true",
            default=False,
            help="Skip training, load model from zoo directly",
        )
        parser.add_argument(
            "--checkpoint",
            type=str,
            default=None,
            help="Explicit checkpoint path (overrides zoo search)",
        )
        parser.add_argument(
            "--active-rounds",
            type=int,
            default=0,
            help="Active learning rounds: refine low-fidelity points with VQE (default: 0)",
        )
        parser.add_argument(
            "--multi-n-train",
            action="store_true",
            default=False,
            help="Instead of training on a single N, aggregate ALL available "
            "bond-resolved data for this topology (from previous runs) and "
            "train a multi-N model. Overrides --train-n for training.",
        )
        parser.add_argument(
            "--force-retrain",
            action="store_true",
            default=False,
            help="Force retraining from scratch even if a suitable model "
            "exists in the zoo. Default: reuse best existing model.",
        )
        parser.add_argument(
            "--skip-retrain",
            action="store_true",
            default=False,
            help="In --iterative-improve: never train/fine-tune a new MPNN. "
            "Keep predicting with the loaded model (--checkpoint or zoo best) "
            "and only refine failing points with VQE, persisting them to NPZ. "
            "Unlike --from-zoo, this still allows the zoo model to be selected "
            "automatically when no --checkpoint is given.",
        )
        parser.add_argument(
            "--model-name",
            type=str,
            default=None,
            help="Custom model name suffix. Checkpoint will be saved as "
            "unifMPNN__<topology>_p<N>_<model-name>.pt (e.g., --model-name coloring_v1)",
        )
        parser.add_argument(
            "--loss-type",
            type=str,
            default="sign_invariant",
            choices=["sign_invariant", "theta_mse", "energy_weighted"],
            help="MPNN training loss. 'sign_invariant' (default): Z₂-symmetric "
            "min(MSE(θ,target), MSE(θ,-target)) per ZZ/X block, preserving the "
            "TFIM HVA sign symmetry. 'theta_mse': plain MSE on θ (legacy). "
            "'energy_weighted': sign-invariant MSE weighted by 1/(1+de_gap) so "
            "points with low energy error contribute more.",
        )
        parser.add_argument(
            "--physics-loss-weight",
            type=float,
            default=0.0,
            help="Weight λ for physics-informed energy loss term (default 0.0 = "
            "disabled). Recommended 0.01-0.1. Adds λ·mean(|E(θ_pred)-E_exact|/N) "
            "after physics_loss_start_epoch.",
        )
        parser.add_argument(
            "--fidelity-loss-weight",
            type=float,
            default=0.1,
            help="Weight λ_F for the fidelity loss term (default 0.1 = ON). "
            "Trains the MPNN to produce HVA states faithful to the exact ground "
            "state via 1−F(|ψ(θ_pred)⟩,|ψ_exact⟩). Only acts for N≤16 "
            "(statevector); graceful no-op above. Set 0 to disable.",
        )
        parser.add_argument(
            "--orbit-feature",
            action="store_true",
            default=False,
            help="Add an automorphism-orbit feature per node/bond to the graph. "
            "Sites/bonds equivalent under a lattice symmetry share an orbit id, "
            "giving the MPNN an equivariance hint so it treats the problem as "
            "lower-dimensional. Default off (keeps 5 node features; existing "
            "checkpoints stay loadable). Requires retraining when enabled.",
        )
        parser.add_argument(
            "--no-eval-cache",
            action="store_true",
            default=False,
            help="Disable circuit evaluation cache. By default, evaluations "
            "are cached in data/eval_cache/ to avoid recomputing identical "
            "(topology, N, h, theta_hash) evaluations.",
        )
        # Iterative improvement + VQE method args from shared CLI module
        from qmbp_simulation.framework.cli import add_iterative_improve_args

        add_iterative_improve_args(parser)
        parser.add_argument(
            "--max-refine-per-iter",
            type=int,
            default=None,
            help="Max h-points to refine per iteration. Default: min(n_failures, 20). "
            "Higher = more VQE compute per iter but fewer iterations needed. "
            "Use --refine-all to refine ALL failing points (no cap).",
        )
        parser.add_argument(
            "--refine-all",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Refine ALL failing points per iteration (no cap). "
            "Maximizes training data quality at the cost of compute time. "
            "Use --no-refine-all to cap refinements via --max-refine-per-iter.",
        )
        parser.add_argument(
            "--ansatz-kind",
            type=str,
            default="auto",
            choices=["auto", "bond_resolved", "frustrated", "longitudinal"],
            help="Which bond-resolved HVA ansatz to build. 'auto' (default) "
            "preserves the historical behavior: frustrated (NN+NNN) when J2 != 0, "
            "longitudinal when g != 0, else plain bond-resolved. The explicit "
            "kinds pin the circuit regardless of J2/g, so the runner can be driven "
            "on any ansatz without relying on implicit flag coupling. The resolved "
            "kind also decides include_nnn for the prediction graph.",
        )
        parser.add_argument(
            "--import-bond-ablation",
            type=str,
            default=None,
            metavar="DIR",
            help="Before running, import existing full-circuit θ from a "
            "bond-ablation study JSON directory (e.g. "
            "results/hva_vl_study/bond_ablation) into the training NPZ corpus so "
            "--iterative-improve can seed from it. Only rows whose θ length matches "
            "the full ansatz n_params for each (N, p) are imported (pruned-bond "
            "variants are skipped). No-op when the dir has no matching data.",
        )
        parser.add_argument(
            "--data-gen",
            action="store_true",
            default=False,
            help="Training-data generation mode (use with --iterative-improve). "
            "Inverts the deployment-oriented adaptive budget: EVERY failing point "
            "gets the FULL --maxiter/--n-restarts budget (hard low-h points are no "
            "longer starved as 'minimal' tier), priority-skipping is disabled, and "
            "the h-grid is densified near the transition (--data-gen-hc, default "
            "0.6) so the corpus is dense and clean exactly where the physics is "
            "hard. Implies --skip-retrain unless --no-skip-retrain-override is set: "
            "the goal is to GENERATE verified θ, not train an MPNN.",
        )
        parser.add_argument(
            "--data-gen-hc",
            type=float,
            default=0.6,
            help="Critical field for --data-gen h-grid densification "
            "(default 0.6, the square-frustrated J2=0.5 transition region).",
        )
        parser.add_argument(
            "--data-gen-dense-radius",
            type=float,
            default=0.3,
            help="Half-width of the dense h region around --data-gen-hc "
            "(default 0.3 → dense sampling in [hc-0.3, hc+0.3]). Chosen so the "
            "0.3-1.8 range yields a clean grid with no near-duplicate h-points.",
        )
        parser.add_argument(
            "--data-gen-min-fidelity",
            type=float,
            default=0.95,
            help="Exact-fidelity acceptance floor for the iterative-improve "
            "failure criterion (default 0.95). In the statevector regime the "
            "runner computes each point's exact fidelity and feeds it to "
            "is_point_failure, where it acts as a two-way signal: a point with "
            "fidelity >= this floor PASSES even if ΔE/gap is large (near the "
            "transition the gap is tiny, so ΔE/gap wrongly rejects fidelity-0.99 "
            "states), while fidelity < floor fails a gap-masked bad state. Most "
            "relevant with --data-gen (dense near-critical grid).",
        )
        parser.add_argument(
            "--compare-warmstart",
            action="store_true",
            default=False,
            help="After multi-N training, run an explicit head-to-head warm-start "
            "A/B at each target-N h: MPNN-predicted θ vs the analytic+donor cascade "
            "(_apply_warmstart_seed) vs a cold random seed. Each arm is refined with "
            "the SAME L-BFGS-B optimizer (shared cost/grad) and scored by "
            "init-fidelity, iters-to-fid-threshold, cost evals and wall time. Emits a "
            "per-h table + per-arm winner tally. Requires a trained/loaded model "
            "(pairs with --multi-n-train or --from-zoo/--checkpoint).",
        )
        parser.add_argument(
            "--compare-fid-threshold",
            type=float,
            default=0.90,
            help="Fidelity threshold for the --compare-warmstart "
            "'iters-to-threshold' convergence metric (default 0.90). The optimizer "
            "stops counting once F(|ψ(θ)⟩,|ψ_exact⟩) >= this value.",
        )
        parser.add_argument(
            "--compare-maxiter",
            type=int,
            default=200,
            help="Max L-BFGS-B iterations per arm in --compare-warmstart "
            "(default 200). The threshold-crossing iteration is recorded; an arm "
            "that never crosses reports its final fidelity at this cap.",
        )

    def build_config(self) -> dict:
        config = self._build_physics_config()
        config["n_anchors"] = self._args.n_anchors
        config["force_method"] = self._args.force_method
        config["bidirectional_anchors"] = self._args.bidirectional_anchors
        config["from_zoo"] = self._args.from_zoo
        return config

    def setup(self):
        """Initialize physics objects."""
        self.setup_physics()
        self._physics_model = self.resolve_model_name("tfim_bond_resolved")
        self._model_kwargs = self.model_kwargs()
        self._is_frustrated = self.is_frustrated()
        self._ansatz_kind = self._resolve_ansatz_kind()
        self._training_data_dir = self.training_data_dir()
        if self._is_frustrated:
            logger.info(
                f"  Frustrated mode: model={self._physics_model} "
                f"J2={self._model_kwargs['J2']} | data namespace=/{FRUSTRATED_NAMESPACE}/"
            )
        # Enforce the hard minimum of VQE restarts for the whole run. Applied
        # once here so every downstream use (main config, bootstrap, refine)
        # inherits the floor.
        if self._args.n_restarts < MIN_N_RESTARTS:
            logger.info(f"  n_restarts raised {self._args.n_restarts} → {MIN_N_RESTARTS} (hard floor MIN_N_RESTARTS)")
            self._args.n_restarts = MIN_N_RESTARTS
        # Auto-detect h_min from valid regime if user didn't override
        # (h below the regime boundary is ansatz-limited for p=1)
        if self._args.h_min == DEFAULT_H_MIN:
            try:
                from qmbp_simulation.framework.preflight import get_regime_threshold

                topo = self._args.topology
                n_target = self._args.target_n[0] if self._args.target_n else 10
                p = self._args.p_layers[0] if isinstance(self._args.p_layers, list) else self._args.p_layers
                threshold = get_regime_threshold(topo, n_target, p)
                if threshold > 0 and threshold > self._args.h_min:
                    logger.info(
                        f"  H-range auto-adjusted: h_min {self._args.h_min} → {threshold:.1f} "
                        f"(valid regime for {topo} N={n_target} p={p})"
                    )
                    self._args.h_min = threshold
            except (ImportError, ValueError, KeyError):
                pass  # Keep default if regime lookup fails

        # Round to 2 decimals for cache key stability (matches GroundTruthCache)
        self._h_values = [round(h, 2) for h in np.linspace(self._args.h_max, self._args.h_min, self._args.h_points)]

        # Data-generation mode: densify the h-grid near the transition and force
        # skip-retrain (we want verified θ, not an MPNN). The budget inversion
        # (full budget for every failing point) happens in the refine loop.
        if getattr(self._args, "data_gen", False):
            from qmbp_simulation.utils.h_grid import generate_nonuniform_h_grid

            grid = generate_nonuniform_h_grid(
                self._args.h_min,
                self._args.h_max,
                self._args.h_points,
                h_critical=self._args.data_gen_hc,
                dense_radius=self._args.data_gen_dense_radius,
            )
            self._h_values = [round(float(h), 2) for h in grid]
            self._args.skip_retrain = True
            logger.info(
                f"  DATA-GEN mode: dense h-grid near h_c={self._args.data_gen_hc} "
                f"(radius={self._args.data_gen_dense_radius}), {len(self._h_values)} points: "
                f"{self._h_values}; skip_retrain forced ON; full budget per failing point."
            )

        self._models = {}  # p_layers → trained model
        self._train_results = {}  # p_layers → AcceleratedResult
        # Default force_method to L-BFGS-B for noiseless backends.
        # VQEOptimizer auto-dispatch will keep L-BFGS-B on noiseless (fast)
        # or downgrade to COBYLA on noisy backends automatically.
        if self._args.force_method is None:
            self._args.force_method = "L-BFGS-B"

        # Optional: seed the training corpus from an existing bond-ablation study
        # so --iterative-improve can start from known-good full-circuit θ instead
        # of cold. Runs once here, before any section, so MultiNAggregator sees
        # the imported NPZs. No-op when the dir has no matching full-circuit data.
        import_dir = getattr(self._args, "import_bond_ablation", None)
        if import_dir:
            self._import_bond_ablation_corpus(import_dir)

    def _import_bond_ablation_corpus(self, json_dir: str) -> None:
        """Import full-circuit θ from a bond-ablation study dir into the NPZ corpus."""
        from qmbp_simulation.framework.result_io import import_bond_ablation_to_npz

        p = self._resolve_p_layers() or 1
        j2 = float(self._model_kwargs.get("J2", 0.0) or 0.0)
        summary = import_bond_ablation_to_npz(
            json_dir,
            topology=self._args.topology,
            p_layers=p,
            model=self._physics_model,
            frustrated=self._is_frustrated,
            j2=j2,
            ansatz_kind=self._ansatz_kind,
        )
        logger.info(
            f"  Bond-ablation import: {summary['imported']} full-circuit points "
            f"(per_n={summary['per_n']}, skipped_dim_mismatch={summary['skipped_dim_mismatch']}, "
            f"files_scanned={summary['files_scanned']})"
        )

    def model_kwargs(self) -> dict:
        """Hamiltonian kwargs: J2 (frustrated) and g (longitudinal field)."""
        kwargs = super().model_kwargs()
        g = getattr(self._args, "g", 0.0) or 0.0
        if abs(g) > 1e-15:
            kwargs["g"] = float(g)
        return kwargs

    def _resolve_ansatz_kind(self) -> str:
        """Resolve the concrete ansatz kind from --ansatz-kind + physics flags.

        'auto' reproduces the historical implicit coupling exactly: frustrated
        when J2 != 0, longitudinal when g != 0, else plain bond-resolved. An
        explicit kind pins the circuit regardless of J2/g so the runner is not
        tied to the frustrated/longitudinal family. Returns one of
        ``"frustrated" | "longitudinal" | "bond_resolved"``.
        """
        kind = getattr(self._args, "ansatz_kind", "auto") or "auto"
        if kind != "auto":
            return kind
        if self._is_frustrated:
            return "frustrated"
        if abs(getattr(self._args, "g", 0.0) or 0.0) > 1e-15:
            return "longitudinal"
        return "bond_resolved"

    def _ansatz_includes_nnn(self) -> bool:
        """Whether the resolved ansatz carries NNN bonds (graph include_nnn)."""
        return getattr(self, "_ansatz_kind", "frustrated" if self._is_frustrated else "bond_resolved") == "frustrated"

    def _build_circuit(self, n_qubits: int, p_layers: int, lattice):
        """Build the HVA circuit for the resolved ansatz kind.

        Dispatches on ``self._ansatz_kind`` (see :meth:`_resolve_ansatz_kind`)
        so the circuit choice is explicit and overridable via --ansatz-kind,
        instead of implicit from J2/g. 'auto' keeps the previous behavior.
        """
        kind = getattr(self, "_ansatz_kind", None) or self._resolve_ansatz_kind()
        if kind == "frustrated":
            return self.hva.create_bond_resolved_frustrated(n_qubits, p_layers, lattice)
        if kind == "longitudinal":
            return self.hva.create_bond_resolved_longitudinal(n_qubits, p_layers, lattice)
        return self.hva.create_bond_resolved(n_qubits, p_layers, lattice)

    def _build_graph(self, lattice, h_value: float, p_layers: int, **kwargs):
        """Build the unified graph, including NNN edges when frustrated.

        The orbit feature (--orbit-feature) must be consistent between training
        (via MultiNAggregator) and prediction (this helper): a model trained
        with 6 node features cannot consume a 5-feature graph. We default the
        flag from self._args.orbit_feature so all prediction graphs match the
        trained model; explicit callers may still override via kwargs.
        """
        from qmbp_simulation.predictors.unified_graph import (
            build_unified_bond_resolved_graph,
        )

        # Prefer the value auto-derived from the loaded model (set in
        # section_cross_n_predict); fall back to the CLI flag before a model is
        # loaded. Explicit kwargs still win.
        _orbit_default = getattr(self, "_orbit_feature_effective", getattr(self._args, "orbit_feature", False))
        kwargs.setdefault("include_orbit_feature", _orbit_default)
        return build_unified_bond_resolved_graph(
            lattice,
            h_value=h_value,
            p_layers=p_layers,
            include_nnn=self._ansatz_includes_nnn(),
            **kwargs,
        )

    def _continuation_donor(self, idx: int, p_layers: int, lattice, prev_theta_by_h: dict):
        """Cross-h continuation donor: the adjacent h-step's converged θ (same N).

        In the descending h-sweep, the previous grid entry (idx-1) is the nearest
        higher-h neighbour. If it was already converged this sweep (present in
        ``prev_theta_by_h``) and does NOT sit across a phase boundary from the
        current h (:func:`crosses_transition`), it is the strongest possible donor:
        same N (layout-identical, no cross-N approximation) and same phase. Returns
        a donor dict for :func:`best_combined_warmstart`, or ``None``.
        """
        from qmbp_simulation.analysis.warmstart import crosses_transition
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        if idx <= 0:
            return None
        h_cur = float(self._h_values[idx])
        h_prev = float(self._h_values[idx - 1])
        if crosses_transition(h_prev, h_cur):
            return None  # θ does not transfer across the frustrated transition
        prev = prev_theta_by_h.get(round(h_prev, 2))
        if not prev or prev[0] is None:
            return None
        import numpy as np

        theta_prev = np.asarray(prev[0], dtype=np.float64)
        if not np.all(np.isfinite(theta_prev)):
            return None
        n_nn = len(lattice.edges)
        nnn_edges = HamiltonianBuilder._generate_nnn_edges(lattice) if self._ansatz_includes_nnn() else []
        n_nnn = len(nnn_edges)
        return {
            "theta": theta_prev,
            "n_nn": n_nn,
            "n_nnn": n_nnn,
            "p": p_layers,
            "n_qubits": lattice.n_qubits,
            "nnn_edges": nnn_edges,
            "h": h_prev,
            "label": f"cont<h{h_prev:.2f}>",
        }

    def _apply_warmstart_seed(
        self,
        *,
        circuit,
        H,
        lattice,
        h: float,
        p_layers: int,
        gap: float,
        theta_init,
        e_init: float,
        eval_fn,
        continuation_donor=None,
    ):
        """Improve a VQE init with the project's combined warm-start cascade.

        Statevector-regime only: computes the exact ground eigenvector ``psi``
        from ``H`` (sparse eigsh), then calls the canonical
        :func:`qmbp_simulation.framework.study_core.prepare_warmstart` to build
        the best structure-aware seed (analytic Ising seeds + regime + donors,
        ranked after an adaptive L-BFGS-B micro-descent via the fast adjoint
        gradient). The seed is passed through ``eval_fn`` (the cached energy
        evaluator) and adopted only if it strictly beats the incoming
        ``e_init`` — so this can never regress the refine, only accelerate it by
        handing VQE a better starting point.

        Returns ``(theta_out, e_out, provenance)`` where ``provenance`` is the
        warm-start label when the seed was adopted, else ``None`` (init kept).
        """
        import numpy as np
        from scipy.sparse.linalg import eigsh

        from qmbp_simulation.framework.study_core import prepare_warmstart
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder

        n_params = int(circuit.num_parameters)
        theta_init = np.asarray(theta_init, dtype=np.float64)

        # Exact ground eigenvector (statevector regime → feasible).
        ev, evec = eigsh(H.to_matrix(sparse=True), k=2, which="SA")
        order = np.argsort(ev)
        psi = evec[:, order[0]].astype(complex)

        n_nn = len(lattice.edges)
        n_nnn = len(HamiltonianBuilder._generate_nnn_edges(lattice)) if self._ansatz_includes_nnn() else 0
        j2 = float(self._model_kwargs.get("J2", 0.0) or 0.0) if self._ansatz_includes_nnn() else 0.0

        # Cross-N / cross-h donors from the NPZ training corpus (the data this
        # pipeline itself generates): converged θ at other N in the same phase,
        # transferred onto this layout. This is the strongest warm-start lever
        # NEAR THE TRANSITION, so we only pay the scan+transfer where it helps —
        # gate on phase_proximity (≈0 in the easy ordered/paramagnetic phases
        # where the analytic seed already wins). Self-reinforcing: each refined
        # point becomes a donor for the next (N, h).
        from qmbp_simulation.analysis.warmstart import discover_npz_donors, phase_proximity

        donors = None
        target_nnn_edges = None
        if phase_proximity(float(h)) > 0.5:
            donors = discover_npz_donors(
                self._args.topology,
                lattice.n_qubits,
                float(h),
                p_layers,
                model=self._physics_model,
                frustrated=self._ansatz_includes_nnn(),
            )
        # The cross-h continuation donor (same N, adjacent h) is prepended so the
        # cascade tries it FIRST — within a phase it is the strongest donor. It is
        # added even in the easy phases (where the NPZ cross-N scan is skipped),
        # since reusing the neighbour θ is free and never hurts (micro-descent
        # ranking drops it if it loses).
        if continuation_donor is not None:
            donors = [continuation_donor, *(donors or [])]
        if donors and self._ansatz_includes_nnn():
            target_nnn_edges = HamiltonianBuilder._generate_nnn_edges(lattice)

        res = prepare_warmstart(
            circuit,
            H,
            psi,
            n_nn=n_nn,
            n_nnn=n_nnn,
            n_qubits=lattice.n_qubits,
            p_layers=p_layers,
            h=float(h),
            J=1.0,
            J2=j2,
            gap=gap,
            donors=donors,
            target_nnn_edges=target_nnn_edges,
            topology=self._args.topology,
            model=self._physics_model,
            target_len=n_params,
        )
        seed = np.asarray(res.get("seed"), dtype=np.float64)
        if seed.shape != theta_init.shape:
            return theta_init, e_init, None
        prov = res.get("provenance") or "warmstart"

        # Data-generation mode: decide on POST-DESCENT energy, not init energy.
        # A warm-start seed may start higher than a previously-stuck θ_prev yet
        # relax into a strictly better basin (common at near-critical h where the
        # old θ_prev is trapped). Comparing only init energy wrongly keeps the
        # stuck point; a short L-BFGS probe on BOTH reveals the real winner. The
        # subsequent full VQE then starts from the better basin. In deployment
        # mode we keep the cheap init-energy gate (no extra descent cost).
        if getattr(self._args, "data_gen", False):
            from qmbp_simulation.framework.study_core import make_cost_fid
            from qmbp_simulation.framework.study_runner import _lbfgsb

            cost, _fid, grad, _ = make_cost_fid(circuit, H, psi)
            probe = 40
            xs, es, _ = _lbfgsb(cost, seed, maxiter=probe, grad=grad)
            xi, ei, _ = _lbfgsb(cost, theta_init, maxiter=probe, grad=grad)
            if np.isfinite(es) and es < ei:
                logger.debug(
                    f"    h={h:.2f}: [data-gen] warm-start seed wins post-probe ({prov}): "
                    f"E_seed={es:.6f} < E_prev={ei:.6f} (init_fid={res.get('init_fidelity')})"
                )
                return xs, float(es), str(prov)
            return (xi, float(ei), None) if np.isfinite(ei) and ei < e_init else (theta_init, e_init, None)

        e_seed = eval_fn(seed)
        if np.isfinite(e_seed) and e_seed < e_init:
            logger.debug(
                f"    h={h:.2f}: warm-start seed adopted ({prov}): "
                f"E {e_init:.6f} → {e_seed:.6f} (init_fid={res.get('init_fidelity')})"
            )
            return seed, float(e_seed), str(prov)
        return theta_init, e_init, None

    def run_preflight(self) -> bool:
        """Validate topology constraints before execution."""
        topo = self._args.topology

        # Validate training h-range filter (if provided)
        train_h_min = getattr(self._args, "train_h_min", None)
        train_h_max = getattr(self._args, "train_h_max", None)
        if train_h_min is not None and train_h_max is not None:
            if train_h_min > train_h_max:
                logger.error(
                    f"--train-h-min ({train_h_min}) must be <= --train-h-max "
                    f"({train_h_max}). No training data would be selected."
                )
                return False
        if (train_h_min is not None or train_h_max is not None) and not (
            getattr(self._args, "multi_n_train", False) or getattr(self._args, "iterative_improve", False)
        ):
            logger.warning(
                "--train-h-min/--train-h-max only affect the training dataset "
                "(built by MultiNAggregator). They have no effect without "
                "--multi-n-train or --iterative-improve."
            )

        # Ladder requires even N
        if topo == "ladder":
            bad_n = []
            if self._args.train_n % 2 != 0 and not self._args.from_zoo:
                bad_n.append(f"train_n={self._args.train_n}")
            for n in self._args.target_n:
                if n % 2 != 0:
                    bad_n.append(f"target_n={n}")
            if bad_n:
                logger.error(
                    f"Ladder topology requires even N. Invalid: {', '.join(bad_n)}. Use N=8, 10, 12, 14, 16, 20, etc."
                )
                return False
        return True

    def _training_h_range(self) -> tuple[float, float]:
        """Effective training h-range for zoo metadata.

        Prefers the explicit --train-h-min/--train-h-max filter when set,
        otherwise falls back to the evaluation sweep range. This ensures the
        zoo's h_range (documented as the TRAINING range) is accurate for
        range-restricted models.
        """
        train_h_min = getattr(self._args, "train_h_min", None)
        train_h_max = getattr(self._args, "train_h_max", None)
        h_min = train_h_min if train_h_min is not None else self._args.h_min
        h_max = train_h_max if train_h_max is not None else self._args.h_max
        return (float(h_min), float(h_max))

    def _train_h_range_note(self) -> str:
        """Traceability note fragment when a training h-range filter is active."""
        train_h_min = getattr(self._args, "train_h_min", None)
        train_h_max = getattr(self._args, "train_h_max", None)
        if train_h_min is not None or train_h_max is not None:
            lo = train_h_min if train_h_min is not None else "-inf"
            hi = train_h_max if train_h_max is not None else "+inf"
            return f", train_h_range=[{lo}, {hi}]"
        return ""

    def _check_existing_npz_utility(self, topology: str, n_qubits: int, p_layers: int = 1) -> None:
        """Check if existing NPZ data is useful for training. Logs warnings if not.

        Does NOT block execution — just informs the user that the existing NPZ
        for this config has been classified as 'not_useful' or 'insufficient_signal'
        by the dashboard. The run will still proceed (to generate fresh data),
        but the warning helps understand why previous models had poor performance.
        """
        try:
            from qmbp_simulation.analysis.metrics import classify_training_utility

            npz_path = self._training_data_dir / f"{topology}_N{n_qubits}_p{p_layers}.npz"
            if not npz_path.exists():
                return  # No existing data — nothing to check

            data = np.load(str(npz_path), allow_pickle=True)
            n_pts = len(data["h_values"])
            if n_pts == 0:
                return

            # Compute dual criterion pass rate
            e_key = "e_vqe" if "e_vqe" in data else ("energies" if "energies" in data else None)
            if e_key is None or "e_exact" not in data:
                return

            from qmbp_simulation.analysis.metrics import (
                DE_GAP_THRESHOLD,
                is_point_failure,
            )

            abs_err = np.abs(data[e_key] - data["e_exact"])
            if "de_gaps" in data:
                de_gaps = data["de_gaps"]
            elif "gaps" in data:
                de_gaps = abs_err / np.maximum(data["gaps"], 1e-10)
            else:
                return

            n_fail = sum(is_point_failure(de_gap=float(de_gaps[i]), abs_error=float(abs_err[i])) for i in range(n_pts))
            pass_dual = (n_pts - n_fail) / n_pts
            pass_simple = float((de_gaps < DE_GAP_THRESHOLD).mean())

            category, reason = classify_training_utility(
                n_pts,
                pass_dual,
                pass_simple,
            )

            if category == "not_useful":
                logger.warning(
                    f"  ⚠️ EXISTING NPZ '{npz_path.name}' IS NOT USEFUL FOR TRAINING:\n"
                    f"     {reason}\n"
                    f"     The MPNN cannot learn from this data. Consider:\n"
                    f"     1. Running with --force-retrain to generate fresh VQE data\n"
                    f"     2. Adjusting h-range to avoid the failing regime\n"
                    f"     3. Deleting the NPZ if the config is fundamentally unrecoverable"
                )
            elif category == "insufficient_signal":
                logger.warning(
                    f"  ⚠️ EXISTING NPZ '{npz_path.name}' has INSUFFICIENT SIGNAL:\n"
                    f"     {reason}\n"
                    f"     Training may not converge well. "
                    f"This run will add more data via upsert."
                )
            else:
                logger.info(f"  Existing NPZ: {n_pts} pts, dual_pass={pass_dual:.0%} — USEFUL for training")
        except Exception as e:
            logger.debug(f"  NPZ utility check failed (non-blocking): {e}")

    def define_sections(self) -> list[Section]:
        sections = [
            Section(
                id=1,
                name="Quality Check",
                fn=self.section_quality_check,
                hypothesis="Target config is predicted viable (pass_prob > 30%)",
            ),
        ]

        # Iterative improve mode: budget estimation + iterative loop
        if getattr(self._args, "iterative_improve", False):
            sections.append(
                Section(
                    id=2,
                    name="Budget Estimation + Cache Warm-up",
                    fn=self.section_budget_estimation,
                    hypothesis="Estimate compute cost leveraging cached results",
                )
            )
            if not getattr(self._args, "budget_only", False):
                sections.append(
                    Section(
                        id=3,
                        name="Iterative Improvement Loop",
                        fn=self.section_iterative_improve,
                        hypothesis="Iterative predict→refine→retrain converges to ≥90% pass rate",
                    )
                )
            return sections

        # An explicit --checkpoint means "use only this model": treat it like
        # --from-zoo so no training section is added (predict-only).
        _predict_only = self._args.from_zoo or bool(getattr(self._args, "checkpoint", None))

        if getattr(self._args, "multi_n_train", False) or getattr(self._args, "force_retrain", False):
            sections.append(
                Section(
                    id=2,
                    name="Multi-N Train (aggregate all available data)",
                    fn=self.section_multi_n_train,
                    hypothesis="Multi-N UnifiedMPNN trained with aggregated data from all N sizes",
                )
            )
        elif not _predict_only:
            sections.append(
                Section(
                    id=2,
                    name=f"Train (N={self._args.train_n})",
                    fn=self.section_train,
                    hypothesis=f"AcceleratedVQE at N={self._args.train_n} achieves >60% pass rate",
                )
            )
        # --compare-warmstart replaces the standard cross-N predict with an
        # explicit MPNN-vs-analytic-vs-cold warm-start A/B at each target-N h.
        # The comparison already predicts the MPNN θ and refines it, so running
        # the ordinary predict section first would only duplicate that compute.
        if getattr(self._args, "compare_warmstart", False):
            sections.append(
                Section(
                    id=3,
                    name="Warm-Start A/B (MPNN vs analytic+donor vs cold)",
                    fn=self.section_compare_warmstart,
                    hypothesis="One warm-start method reaches the fidelity threshold "
                    "in fewer L-BFGS-B iterations than the others",
                )
            )
            return sections
        sections.append(
            Section(
                id=3,
                name="Cross-N Predict",
                fn=self.section_cross_n_predict,
                hypothesis="Cross-N prediction achieves ΔE/gap < 10% for h > 2.0",
            )
        )
        return sections

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 1: Quality Check
    # ═══════════════════════════════════════════════════════════════════════════

    def section_quality_check(self) -> dict:
        """Run QualityPredictor for training and target configs via base helper."""
        # Collect unique N values to avoid duplicate checks when train_n == target_n
        n_values = {self._args.train_n}
        for n_target in self._args.target_n:
            n_values.add(n_target)

        configs = [
            {
                "model": self._physics_model,
                "topology": self._args.topology,
                "n_qubits": n,
                "p_layers": self._args.p_layers[0],
                "h_min": self._args.h_min,
                "h_max": self._args.h_max,
            }
            for n in sorted(n_values)
        ]
        return self.run_quality_check(configs=configs)

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 2: Train
    # ═══════════════════════════════════════════════════════════════════════════

    def section_train(self) -> dict:
        """Train AcceleratedVQE at N_train for each p value."""
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.pipeline.accelerated import AcceleratedConfig, AcceleratedVQE

        spec = get_model_spec(self._physics_model).with_params(**self._model_kwargs)
        hva = HVACircuitBuilder()
        N = self._args.train_n
        topo = self._args.topology

        # ── Training utility gating: warn if existing NPZ is not useful ──
        _p_check = self._args.p_layers[0] if isinstance(self._args.p_layers, list) else self._args.p_layers
        self._check_existing_npz_utility(topo, N, p_layers=_p_check)

        backend = self.select_backend(N, for_vqe_loop=True)

        all_results = {}
        for p in self._args.p_layers:
            logger.info(f"  Training: N={N}, p={p}, topology={topo}")
            lattice = self.make_lattice(topo, N, J=1.0, h=2.0)
            circuit, _ = self._build_circuit(N, p, lattice)
            logger.info(f"    Circuit params: {circuit.num_parameters}")

            config = AcceleratedConfig(
                n_anchors=self._args.n_anchors,
                n_restarts=self._args.n_restarts,
                maxiter=self._args.maxiter,
                mpnn_epochs=FULL_TRAIN_EPOCHS,
                use_zoo=False,
                force_method=getattr(self._args, "force_method", None),
                bidirectional_anchors=getattr(self._args, "bidirectional_anchors", False),
            )

            t0 = time.perf_counter()
            accel = AcceleratedVQE(lattice, circuit, spec, backend, config=config, p_layers=p)
            result = accel.run(self._h_values, seed=42, p_layers=p)
            elapsed = time.perf_counter() - t0

            self._models[p] = accel.get_model()
            self._train_results[p] = result

            # Persist θ_opt for multi-N reuse (atomic write with anti-regression)
            training_data_dir = self._training_data_dir
            training_data_dir.mkdir(parents=True, exist_ok=True)
            npz_path = training_data_dir / f"{topo}_N{N}_p{p}.npz"

            # Map method labels to quality tiers:
            # vqe_full, vqe_refined → verified (VQE-converged)
            # mpnn_refined → verified (VQE-corrected prediction)
            # mpnn_direct → approximate (MPNN only, not VQE-verified)
            quality_tiers = [
                "verified" if m in ("vqe_full", "vqe_refined", "mpnn_refined") else "approximate" for m in result.method
            ]

            n_upd, n_add = self.persist_theta_npz(
                npz_path,
                h_new=self._h_values[: len(result.theta_opt)],
                theta_new=result.theta_opt,
                e_vqe_new=result.energies,
                e_exact_new=result.e_exact,
                gaps_new=result.gaps,
                method_new=list(result.method),
                quality_tier_new=quality_tiers,
            )
            n_verified = sum(1 for t in quality_tiers if t == "verified")
            logger.info(
                f"    Saved training data: {npz_path} "
                f"({n_add} added, {n_upd} improved, {n_verified}/{len(quality_tiers)} verified)"
            )

            all_results[f"p{p}"] = {
                "pass_rate": result.pass_rate,
                "mean_de_gap": float(result.de_gaps.mean()),
                "elapsed_s": elapsed,
                "n_anchors": result.n_anchors,
                "model_source": result.model_source,
                "methods": dict(zip(*np.unique(result.method, return_counts=True), strict=False)),
            }
            logger.info(
                f"    Done: pass_rate={result.pass_rate:.0%}, "
                f"mean_ΔE/gap={result.de_gaps.mean():.4f}, time={elapsed:.1f}s"
            )

        passed = all(r.get("pass_rate", 0) > 0.5 for r in all_results.values())
        return {"pass": passed, "per_p": all_results}

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 2 (alt): Multi-N Train
    # ═══════════════════════════════════════════════════════════════════════════

    def section_multi_n_train(self) -> dict:
        """Train UnifiedMPNN using aggregated data from ALL available N sizes.

        If a suitable multi-N model already exists in the zoo and --force-retrain
        is not set, loads it instead of retraining.
        """
        from qmbp_simulation.analysis.metrics import validate_training_dataset
        from qmbp_simulation.predictors.model_zoo import (
            ZooEntry,
            load_pretrained,
            register_checkpoint_with_training_metrics,
        )
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN, train_unified_mpnn

        topo = self._args.topology
        p = self._args.p_layers[0]
        force_retrain = getattr(self._args, "force_retrain", False)

        # Check if a multi-N model already exists (skip training if so)
        if not force_retrain:
            try:
                model, meta = load_pretrained(
                    model=self._physics_model,
                    topology=topo,
                    n_qubits=0,  # 0 = multi-N
                    p_layers=p,
                )
                self._models[p] = model
                logger.info(
                    f"  Loaded existing multi-N model: {meta.checkpoint_file} "
                    f"({meta.n_training_points} points). Use --force-retrain to rebuild."
                )
                return {
                    "pass": True,
                    "reused_existing": True,
                    "checkpoint": meta.checkpoint_file,
                    "n_training_points": meta.n_training_points,
                    "notes": meta.notes,
                }
            except FileNotFoundError:
                logger.info("  No existing multi-N model. Training from scratch.")

        # 1. Scan and aggregate all available data for this topology
        logger.info(f"  Scanning all bond-resolved data for topology={topo}...")
        # Use max_n to prevent contamination from extrapolation data beyond viable range.
        # p_layers MUST be passed so only *_p{p}.npz data is used — never mix p=1 and p=2.
        max_n = self.N_MAX_VIABLE.get(topo, 20)
        agg = MultiNAggregator(
            topology=topo,
            model=self._physics_model,
            frustrated=self._is_frustrated,
            results_dir=self._training_data_dir,
            max_n=max_n,
            p_layers=p,
            h_min=getattr(self._args, "train_h_min", None),
            h_max=getattr(self._args, "train_h_max", None),
            include_orbit_feature=getattr(self._args, "orbit_feature", False),
        )
        summary = agg.scan()

        if not summary:
            return {"pass": False, "error": "No existing data found. Run --train-n first."}

        logger.info(f"  Found data for N={agg.available_n_values()}: {summary}")

        if True:
            # ── PRE-TRAINING VALIDATION ──────────────────────────────────────────
            # Validate data quality BEFORE attempting to train
            is_viable, validation_report = validate_training_dataset(
                agg._data_by_n,
                max_de_gap=0.10,
                min_total_points=10,
                min_n_values=2,
            )
            if not is_viable:
                logger.error(
                    f"  ❌ TRAINING DATA NOT VIABLE:\n"
                    f"     {validation_report['recommendation']}\n"
                    f"     Errors: {validation_report['errors']}"
                )
                for warn in validation_report.get("warnings", [])[:5]:
                    logger.warning(f"     {warn}")
                return {
                    "pass": False,
                    "error": "Training data validation failed",
                    "validation_report": validation_report,
                    "recommendation": validation_report["recommendation"],
                }
            logger.info(
                f"  ✓ Data validation passed: {validation_report['total_good']}/{validation_report['total_raw']} "
                f"good points across {validation_report['n_values_with_good_data']} N values"
            )
            if validation_report.get("warnings"):
                for warn in validation_report["warnings"][:3]:
                    logger.warning(f"     {warn}")

            # 2. Build combined dataset (filter by quality)
            dataset = agg.build_combined_dataset(max_de_gap=0.10)
            if len(dataset) < 5:
                return {
                    "pass": False,
                    "error": f"Only {len(dataset)} points pass quality filter. Need ≥5.",
                    "summary": agg.summary(),
                }

            logger.info(f"  Combined dataset: {len(dataset)} graphs from N={agg.available_n_values()}")

            # 3. Determine output dim from dataset (varies by graph size)
            # UnifiedMPNN uses per-node prediction so output_dim is implicit
            sample_g = dataset[0]
            n_node_features = sample_g.x.shape[1] if hasattr(sample_g, "x") else 4

            # 4. Train UnifiedMPNN
            model = UnifiedMPNN(
                node_features=n_node_features,
                hidden_dim=256,
                n_layers=3,
                norm_type="none",  # MANDATORY for cross-N
                dropout=0.1,
                use_residual=getattr(self._args, "use_residual", False),
                film_conditioning=getattr(self._args, "film", False),
            )

            logger.info("  Training UnifiedMPNN (multi-N, norm_type=none)...")
            t0 = time.perf_counter()
            train_result = train_unified_mpnn(
                model,
                dataset,
                n_epochs=FULL_TRAIN_EPOCHS,
                lr=1e-3,
                patience=200,
                seed=42,
                loss_type=getattr(self._args, "loss_type", "sign_invariant"),
                physics_loss_weight=getattr(self._args, "physics_loss_weight", 0.0),
                fidelity_loss_weight=getattr(self._args, "fidelity_loss_weight", 0.1),
            )
            elapsed = time.perf_counter() - t0

            final_mse = train_result.get("final_mse", 0) if isinstance(train_result, dict) else 0
            logger.info(f"  Training done: MSE={final_mse:.2e}, time={elapsed:.1f}s")

            # Persist training curve for post-hoc analysis
            try:
                from qmbp_simulation.utils.helpers import persist_training_curve

                persist_training_curve(
                    train_result,
                    output_dir=Path("results/training_curves"),
                    prefix=f"{topo}_multiN_p{p}",
                )
            except Exception:
                pass

            # Store model for Section 3
            self._models[p] = model

            # 5. Export to zoo as multi-N model
            from datetime import datetime

            from qmbp_simulation.predictors.model_zoo import get_runner_tag, make_date_tag

            n_values_str = "+".join(str(n) for n in agg.available_n_values())
            if getattr(self._args, "model_name", None):
                ckpt_file = f"unifMPNN__{topo}_p{p}_{self._args.model_name}.pt"
            else:
                ckpt_file = f"unified_tfim_br_{topo}_multiN_{n_values_str}_p{p}.pt"
            entry = ZooEntry(
                model=self._physics_model,
                topology=topo,
                n_qubits=0,  # 0 = multi-N
                p_layers=p,
                checkpoint_file=ckpt_file,
                # h_range documents the TRAINING h-range. When --train-h-min/max
                # restrict the dataset, record that; otherwise fall back to the
                # sweep range (which approximates the training coverage).
                h_range=self._training_h_range(),
                pass_rate=0.0,  # Updated after eval
                n_training_points=len(dataset),
                seeds=[42],
                created=datetime.now(UTC).isoformat(),
                notes=f"Multi-N training: N={agg.available_n_values()}, {len(dataset)} points"
                + self._train_h_range_note()
                + (", arch=residual" if getattr(self._args, "use_residual", False) else ""),
                runner_tag=get_runner_tag(self.runner_id),
                date_tag=make_date_tag(),
            )
            register_checkpoint_with_training_metrics(
                model,
                entry,
                training_result=train_result,
                overwrite=True,
                architecture_config={
                    "hidden_dim": 256,
                    "n_conv_layers": 3,
                    "norm_type": "none",
                    "dropout": 0.1,
                    "use_residual": getattr(self._args, "use_residual", False),
                    "film_conditioning": getattr(self._args, "film", False),
                },
            )
            logger.info(f"  Exported multi-N model: {entry.checkpoint_file}")

            # Auto-persist training curve
            try:
                from qmbp_simulation.utils.helpers import persist_training_curve

                persist_training_curve(
                    train_result,
                    output_dir=Path("results/training_curves"),
                    prefix=f"{topo}_section_multiN_p{p}",
                )
            except Exception:
                pass

            # Enrich model registry with per-N point breakdown
            try:
                from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

                db = ModelRegistryDB()
                record = db.get_model(entry.checkpoint_file)
                if record:
                    record.training.points_per_n = {str(k): v for k, v in agg.summary()["points_per_n"].items()}
                    record.training.n_values_used = agg.available_n_values()
                    db.register_model(record, overwrite=True)
            except Exception as e:
                logger.debug("Registry enrichment failed (non-critical): %s", e)

        return {
            "pass": True,
            "n_values_used": agg.available_n_values(),
            "n_training_points": len(dataset),
            "points_per_n": {str(k): v for k, v in agg.summary()["points_per_n"].items()},
            "final_mse": float(final_mse),
            "elapsed_s": elapsed,
            "checkpoint": entry.checkpoint_file,
            "validation_report": validation_report,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 3: Cross-N Predict
    # ═══════════════════════════════════════════════════════════════════════════

    def section_cross_n_predict(self) -> dict:
        """Predict at each N_target using the N_train model."""
        import torch

        from qmbp_simulation.analysis.metrics import compute_deploy_summary
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.models.constants import STATEVECTOR_MAX_N
        from qmbp_simulation.models.model_registry import get_model_spec

        spec = get_model_spec(self._physics_model).with_params(**self._model_kwargs)
        hva = HVACircuitBuilder()
        topo = self._args.topology

        all_results = {}

        for p in self._args.p_layers:
            # Load model: from memory (section 2), from zoo (best for cross-N), or train new
            model = self._models.get(p)
            if model is None:
                # Respect --from-zoo / --checkpoint: never train a new model.
                _train_if_missing = not (self._args.from_zoo or bool(getattr(self._args, "checkpoint", None)))
                model = self.load_best_mpnn_for_cross_n(
                    n_target=self._args.target_n[0],
                    model=self._physics_model,
                    topology=topo,
                    p_layers=p,
                    checkpoint_path=self._args.checkpoint,
                    train_if_missing=_train_if_missing,
                    train_epochs=FULL_TRAIN_EPOCHS,
                )
                if model is None:
                    _reason = (
                        "No model in zoo for this config (--from-zoo set, training disabled)"
                        if self._args.from_zoo
                        else "No model and no training data available"
                    )
                    all_results[f"p{p}"] = {
                        "pass": False,
                        "error": _reason,
                    }
                    continue

            model.eval()

            # Auto-match graph feature dim to the loaded model. A model trained
            # with the orbit feature has node_features = UNIFIED_NODE_FEATURES+1;
            # prediction graphs MUST carry the same extra column or the forward
            # pass fails with a size mismatch. Deriving this from the model (not
            # the CLI flag) makes predict-only / zoo-loaded runs self-correct.
            from qmbp_simulation.predictors.unified_graph import UNIFIED_NODE_FEATURES

            _model_feat = getattr(model, "node_features", UNIFIED_NODE_FEATURES)
            self._orbit_feature_effective = _model_feat > UNIFIED_NODE_FEATURES
            if self._orbit_feature_effective != getattr(self._args, "orbit_feature", False):
                logger.info(
                    "  Orbit feature auto-set to %s from loaded model (node_features=%d)",
                    self._orbit_feature_effective,
                    _model_feat,
                )

            for n_target in self._args.target_n:
                logger.info(f"  Cross-N: N_train={self._args.train_n} → N_target={n_target}, p={p}")
                lattice_target = self.make_lattice(topo, n_target, J=1.0, h=2.0)
                circuit_target, _ = self._build_circuit(n_target, p, lattice_target)
                n_params_target = circuit_target.num_parameters

                # Use CachedBackend for transparent eval caching
                use_eval_cache = not getattr(self._args, "no_eval_cache", False)
                import os as _os

                _prov = getattr(self, "_model_provenance", None) or {}
                _ckpt_id = _os.path.basename(str(_prov.get("checkpoint") or self._args.checkpoint or ""))
                eval_backend = self.get_cached_backend(
                    topology=topo,
                    n_qubits=n_target,
                    model=self._physics_model,
                    p_layers=p,
                    enabled=use_eval_cache,
                    ckpt_id=_ckpt_id,
                )
                # Log which backend was selected (debug N>22 slowness)
                _inner = getattr(eval_backend, "_backend", eval_backend)
                logger.info(
                    f"    Backend for N={n_target}: {_inner.name} (wrapped in CachedBackend, cache={use_eval_cache})"
                )
                if use_eval_cache:
                    logger.info(f"    Eval cache: {len(eval_backend.cache)} entries loaded")

                per_h_results = []
                t0 = time.perf_counter()
                _interrupted = False

                try:
                    for h in self._h_values:
                        g = self._build_graph(
                            lattice_target,
                            float(h),
                            p,
                            include_circuit_nodes=True,
                        )
                        with torch.no_grad():
                            theta_pred = model(g).numpy().flatten()

                        theta_pred = np.clip(theta_pred, -np.pi, np.pi)

                        # MC-Dropout uncertainty estimation (reuses model method)
                        theta_std = 0.0
                        if hasattr(model, "predict_with_uncertainty"):
                            _, theta_std = model.predict_with_uncertainty(g)

                        # Verify param count matches circuit
                        if len(theta_pred) != n_params_target:
                            logger.warning(
                                f"    Param mismatch at h={h:.2f}: predicted {len(theta_pred)}, need {n_params_target}"
                            )
                            if len(theta_pred) < n_params_target:
                                theta_pred = np.pad(theta_pred, (0, n_params_target - len(theta_pred)))
                            else:
                                theta_pred = theta_pred[:n_params_target]

                        # Evaluate energy via CachedBackend
                        lat_h = self.make_lattice(topo, n_target, J=1.0, h=float(h))
                        H = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)
                        eval_backend.set_h(float(h))
                        e_pred = eval_backend.evaluate(circuit_target, H, theta_pred)

                        # Ground truth via parent's cached exact_ground_state
                        e_exact, gap = self.exact_ground_state(
                            topo,
                            n_target,
                            float(h),
                            model=self._physics_model,
                            model_kwargs=self._model_kwargs or None,
                        )

                        de_gap = abs(e_pred - e_exact) / max(gap, 1e-10)
                        abs_err = abs(e_pred - e_exact)

                        # Fidelity at any N: exact (N ≤ 16) or variance-based
                        # lower bound (Eckart) for larger N. Records provenance.
                        fid_info = self.estimate_fidelity(
                            circuit_target,
                            theta_pred,
                            topo,
                            n_target,
                            float(h),
                            model=self._physics_model,
                            gap=gap,
                            e_pred=e_pred,
                        )
                        fidelity = fid_info.get("fidelity")

                        per_h_result = self.build_per_h_result(
                            h,
                            e_pred,
                            e_exact,
                            gap,
                            fidelity_info=fid_info,
                            n_params=len(theta_pred),
                        )
                        per_h_result["theta_std"] = theta_std
                        per_h_results.append(per_h_result)
                        if fidelity is not None:
                            _bnd = "≥" if fid_info.get("is_lower_bound") else "="
                            fid_str = f"F{_bnd}{fidelity:.4f}"
                        else:
                            fid_str = "F=N/A"
                        logger.info(
                            f"    h={h:.2f}: ΔE/gap={de_gap:.4f} {fid_str} "
                            f"|ΔE|={abs_err:.2e} [{len(theta_pred)} params]"
                        )

                except KeyboardInterrupt:
                    _interrupted = True
                    logger.warning(
                        f"  ⚠️ Interrupted during cross-N predict N={n_target}. "
                        f"Saving {len(per_h_results)} partial results."
                    )

                elapsed = time.perf_counter() - t0

                # ── Active learning: refine low-fidelity points ───────
                active_rounds = getattr(self._args, "active_rounds", 0)
                n_refined = 0
                cold_start_samples = []

                if active_rounds > 0:
                    from scipy.optimize import minimize as _minimize

                    # Select VQE backend for refinement
                    vqe_backend = self.select_backend(n_target, for_vqe_loop=True)

                    # ── Cold-start baseline (2 sample points) ──
                    sample_indices = [0, len(per_h_results) // 2]
                    rng_cold = np.random.default_rng(99)
                    for si in sample_indices:
                        if si >= len(per_h_results):
                            continue
                        r_cold = per_h_results[si]
                        h_cold = r_cold["h"]
                        try:
                            lat_cold = self.make_lattice(topo, n_target, J=1.0, h=h_cold)
                            H_cold = spec.build_hamiltonian(lat_cold, **spec.hamiltonian_kwargs)
                            theta_random = rng_cold.uniform(-np.pi, np.pi, n_params_target)
                            al_maxiter = 50 if n_target > STATEVECTOR_MAX_N else 200
                            res_cold = _minimize(
                                lambda params: vqe_backend.evaluate(circuit_target, H_cold, params),
                                theta_random,
                                method="COBYLA",
                                options={"maxiter": al_maxiter, "rhobeg": 0.5},
                            )
                            e_ex_cold, gap_cold = self.exact_ground_state(
                                topo,
                                n_target,
                                h_cold,
                                model=self._physics_model,
                                model_kwargs=self._model_kwargs or None,
                            )
                            de_gap_cold = abs(res_cold.fun - e_ex_cold) / max(gap_cold, 1e-10)
                            cold_start_samples.append(
                                {
                                    "h": h_cold,
                                    "de_gap_cold": float(de_gap_cold),
                                    "de_gap_warm": float(r_cold["de_gap"]),
                                    "speedup": "warm better" if r_cold["de_gap"] < de_gap_cold else "cold better",
                                }
                            )
                            logger.info(
                                f"    Cold-start baseline h={h_cold:.2f}: "
                                f"dE/gap_cold={de_gap_cold:.4f} vs dE/gap_warm={r_cold['de_gap']:.4f} "
                                f"({'warm wins' if r_cold['de_gap'] < de_gap_cold else 'cold wins'})"
                            )
                        except Exception as e:
                            logger.debug(f"    Cold-start sample h={h_cold:.2f} failed: {e}")

                    # ── Active learning rounds ────────────────────────────────
                    for al_round in range(active_rounds):
                        from qmbp_simulation.analysis.metrics import is_point_failure

                        refine_indices = [
                            i
                            for i, r in enumerate(per_h_results)
                            if is_point_failure(
                                r["de_gap"],
                                abs_error=r.get("abs_error"),
                            )
                        ]
                        if not refine_indices:
                            logger.info(f"    AL round {al_round + 1}: all points pass. Done.")
                            break
                        refine_indices = refine_indices[:5]
                        logger.info(
                            f"    Active learning round {al_round + 1}: "
                            f"refining {len(refine_indices)} points "
                            f"(backend={type(vqe_backend).__name__})"
                        )
                        for idx in refine_indices:
                            r = per_h_results[idx]
                            h_val = r["h"]
                            try:
                                lat_ref = self.make_lattice(topo, n_target, J=1.0, h=h_val)
                                H_ref = spec.build_hamiltonian(lat_ref, **spec.hamiltonian_kwargs)

                                g_ref = self._build_graph(
                                    lattice_target,
                                    h_val,
                                    p,
                                    include_circuit_nodes=True,
                                )
                                with torch.no_grad():
                                    theta_init = model(g_ref).numpy().flatten()
                                theta_init = np.clip(theta_init, -np.pi, np.pi)
                                if len(theta_init) != n_params_target:
                                    if len(theta_init) < n_params_target:
                                        theta_init = np.pad(theta_init, (0, n_params_target - len(theta_init)))
                                    else:
                                        theta_init = theta_init[:n_params_target]

                                al_maxiter = 200 if n_target <= STATEVECTOR_MAX_N else 50
                                res = _minimize(
                                    lambda params: vqe_backend.evaluate(circuit_target, H_ref, params),
                                    theta_init,
                                    method="COBYLA",
                                    options={"maxiter": al_maxiter, "rhobeg": 0.1},
                                )

                                e_exact_ref, gap_ref = self.exact_ground_state(
                                    topo,
                                    n_target,
                                    h_val,
                                    model=self._physics_model,
                                    model_kwargs=self._model_kwargs or None,
                                )
                                de_gap_new = abs(res.fun - e_exact_ref) / max(gap_ref, 1e-10)

                                # Fidelity: exact (N≤16) or variance bound (N>16)
                                fid_info_new = self.estimate_fidelity(
                                    circuit_target,
                                    res.x,
                                    topo,
                                    n_target,
                                    h_val,
                                    model=self._physics_model,
                                    gap=gap_ref,
                                    e_pred=float(res.fun),
                                )
                                fid_new = fid_info_new.get("fidelity")

                                if de_gap_new < r["de_gap"]:
                                    per_h_results[idx] = {
                                        **r,
                                        "de_gap": float(de_gap_new),
                                        "abs_error": float(abs(res.fun - e_exact_ref)),
                                        "fidelity": fid_new if fid_new is not None else r.get("fidelity"),
                                        "fidelity_method": fid_info_new.get("method"),
                                        "fidelity_is_bound": fid_info_new.get("is_lower_bound", False),
                                        "e_pred": float(res.fun),
                                        "method": "refined",
                                        "de_gap_before_refine": float(r["de_gap"]),
                                    }
                                    n_refined += 1
                                    logger.info(
                                        f"      h={h_val:.2f}: dE/gap {r['de_gap']:.4f} -> {de_gap_new:.4f} "
                                        f"(improvement: {(1 - de_gap_new / r['de_gap']) * 100:.0f}%)"
                                    )
                                else:
                                    logger.info(
                                        f"      h={h_val:.2f}: no improvement ({de_gap_new:.4f} >= {r['de_gap']:.4f})"
                                    )
                            except Exception as e:
                                logger.warning(f"      h={h_val:.2f}: refinement failed: {e}")
                                continue

                    if n_refined > 0:
                        logger.info(f"    AL summary: {n_refined} points refined")

                # ── Compute summary via reusable utility ──────────────
                summary = compute_deploy_summary(per_h_results)

                # ── Uncertainty calibration (θ_std vs ΔE/gap) ─────────
                from qmbp_simulation.analysis.metrics import compute_uncertainty_correlation

                uc_report = compute_uncertainty_correlation(per_h_results)

                key = f"p{p}_N{n_target}"
                all_results[key] = {
                    "train_n": self._args.train_n,
                    "target_n": n_target,
                    "p_layers": p,
                    "n_params": n_params_target,
                    **summary,
                    "uncertainty_calibration": uc_report if uc_report["n_points_with_uncertainty"] >= 3 else None,
                    "fidelity_available": n_target <= STATEVECTOR_MAX_N,
                    "active_learning_applied": n_refined > 0,
                    "n_refined": n_refined,
                    "cold_start_comparison": cold_start_samples if active_rounds > 0 else None,
                    "elapsed_s": elapsed,
                    "per_point": per_h_results,
                }
                fid_info = f"F_mean={summary['mean_fidelity']:.4f}" if summary.get("mean_fidelity") else "F=N/A"
                grade = summary.get("grade", "?")
                score = summary.get("quality_score", 0)
                logger.info(
                    f"  N={n_target}: {grade}({score:.2f}) "
                    f"ΔE/gap={summary['mean_de_gap']:.4f}±{summary.get('std_de_gap', 0):.4f} "
                    f"P90={summary.get('p90_de_gap', 0):.4f} "
                    f"{fid_info}, refined={n_refined}"
                )

        # Overall pass: at least one target has >50% at 10% threshold
        passed = any(
            v.get("pass_rate_10pct", 0) > 0.5
            for v in all_results.values()
            if isinstance(v, dict) and "pass_rate_10pct" in v
        )

        # Flush eval cache — os._exit() in ValidationRunner skips __del__,
        # so without explicit flush the last cached evaluations are lost.
        if hasattr(eval_backend, "flush"):
            eval_backend.flush()

        # ── Auto-update zoo pass_rate with observed results ──────────────
        # Use centralized update_zoo_pass_rate for better maintainability
        if hasattr(self, "_zoo_entry") and self._zoo_entry is not None:
            observed_pass_rates = [
                v.get("pass_rate_dual", v.get("pass_rate_5pct", 0))
                for v in all_results.values()
                if isinstance(v, dict) and ("pass_rate_dual" in v or "pass_rate_5pct" in v)
            ]
            if observed_pass_rates:
                observed = max(observed_pass_rates)
                try:
                    from qmbp_simulation.predictors.model_zoo import update_zoo_pass_rate

                    n_target = self._args.target_n[0] if self._args.target_n else "?"
                    update_zoo_pass_rate(
                        self._zoo_entry.checkpoint_file,
                        observed,
                        only_if_better=True,
                        add_notes=f"cross-N@N={n_target}",
                    )
                except Exception:
                    pass  # Non-fatal

        return {"pass": passed, "cross_n_results": all_results}

    # ═══════════════════════════════════════════════════════════════════════════
    # Section: Budget Estimation + Cache Warm-up
    # ═══════════════════════════════════════════════════════════════════════════

    def section_budget_estimation(self) -> dict:
        """Estimate compute budget leveraging cached results.

        Uses the shared `self.estimate_compute_budget()` + `self.log_budget_summary()`
        helpers from ValidationRunner, then adds runner-specific
        QualityPredictor estimates on top.
        """
        topo = self._args.topology
        n_target = self._args.target_n[0]
        p = self._args.p_layers[0]
        max_iters = self._args.max_iterations

        # ── Shared budget estimation (GT cache, EvalCache, NPZ, h_frontier)
        budget = self.estimate_compute_budget(
            h_values=self._h_values,
            n_qubits=n_target,
            topology=topo,
            model=self._physics_model,
            max_iterations=max_iters,
        )

        # ── Runner-specific: QualityPredictor time estimate ──────────────
        estimated_time_s = 0.0
        try:
            from qmbp_simulation.analysis.quality_predictor import QualityPredictor

            predictor = QualityPredictor()
            report = predictor.predict(
                model=self._physics_model,
                topology=topo,
                n_qubits=n_target,
                p_layers=p,
                h_min=self._args.h_min,
                h_max=self._args.h_max,
            )
            estimated_time_s = report.estimated_time_s
        except Exception:
            pass

        # ── Log using reusable base class method ─────────────────────────
        self.log_budget_summary(
            budget,
            topology=topo,
            n_qubits=n_target,
            p_layers=p,
            historical_time_s=estimated_time_s,
        )

        budget["historical_time_s"] = estimated_time_s
        return {"pass": True, "budget": budget}

    # ═══════════════════════════════════════════════════════════════════════════
    # Section: Iterative Improvement Loop
    # ═══════════════════════════════════════════════════════════════════════════

    def section_iterative_improve(self) -> dict:
        """Iterative predict → refine → retrain loop with cache reuse."""
        import torch

        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.execution import NoiselessBackend
        from qmbp_simulation.execution.eval_cache import EvalCache
        from qmbp_simulation.models.constants import STATEVECTOR_MAX_N
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.predictors.model_zoo import (
            ZooEntry,
            load_pretrained,
            register_checkpoint_with_training_metrics,
        )
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator
        from qmbp_simulation.predictors.unified_mpnn import (
            UnifiedMPNN,
            train_unified_mpnn,
        )
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        topo = self._args.topology
        n_target = self._args.target_n[0]
        p = self._args.p_layers[0]
        max_iterations = self._args.max_iterations
        improvement_threshold = self._args.improvement_threshold
        spec = get_model_spec(self._physics_model).with_params(**self._model_kwargs)
        hva = HVACircuitBuilder()

        # Architecture flag: CLI --use-residual or auto-detected from loaded model
        use_residual = getattr(self._args, "use_residual", False)

        # ── Setup: caches, backend, circuit ───────────────────────────────
        gt_cache = getattr(self, "_disk_gt_cache", None) or GroundTruthCache()
        if not hasattr(self, "_disk_gt_cache"):
            self._disk_gt_cache = gt_cache
        use_eval_cache = not getattr(self._args, "no_eval_cache", False)
        eval_cache = EvalCache(enabled=use_eval_cache)
        backend = NoiselessBackend()
        solver = self.solver

        # Auto-select backend for N > 22
        if n_target > STATEVECTOR_MAX_N:
            try:
                from qmbp_simulation.execution import MPSBackend

                eval_backend = MPSBackend(strategy="aer_mps", chi_max=64, deterministic=True)
                logger.info(f"  Backend: MPSBackend(aer_mps, chi=64, det=True) for N={n_target} > {STATEVECTOR_MAX_N}")
            except ImportError:
                eval_backend = backend
                logger.warning(
                    f"  ⚠️ qiskit-aer not available — falling back to NoiselessBackend "
                    f"for N={n_target}. THIS WILL BE VERY SLOW (2^{n_target} amplitudes)."
                )
        else:
            eval_backend = backend
            logger.info(f"  Backend: NoiselessBackend for N={n_target} ≤ {STATEVECTOR_MAX_N}")

        lattice_target = self.make_lattice(topo, n_target, J=1.0, h=2.0)
        circuit_target, _ = self._build_circuit(n_target, p, lattice_target)
        n_params = circuit_target.num_parameters
        logger.info(f"  Circuit: N={n_target}, p={p}, n_params={n_params}, eval_backend={eval_backend.name}")

        # NPZ path for this config
        npz_dir = self._training_data_dir
        npz_dir.mkdir(parents=True, exist_ok=True)
        npz_path = npz_dir / f"{topo}_N{n_target}_p{p}.npz"

        # Load existing refined θ from NPZ (anti-regression baseline)
        from qmbp_simulation.framework.result_io import load_npz_as_theta_dict

        prev_theta_by_h = load_npz_as_theta_dict(npz_path, n_params)

        # Ansatz-limit boundary disabled: all h values are refinable.
        # This was previously used to skip points near quantum critical point
        # but empirically refinement helps even there.
        h_min_valid = 0

        # ── Compute ground truth (all from cache ideally) ─────────────────
        # Route through get_or_compute so this loop shares the SAME stale-floor
        # gap invalidation (N>18 gap≈2π/N) as exact_ground_state — previously a
        # raw gt_cache.get() here could serve a stale gap and skew ΔE/gap for
        # large N. Hits/misses are inferred from cache membership pre-call.
        gt_hits, gt_misses = 0, 0
        e_exact_arr = np.zeros(len(self._h_values))
        gap_arr = np.zeros(len(self._h_values))
        for i, h in enumerate(self._h_values):
            t_gt = time.perf_counter()
            if self._is_frustrated:
                _was_cached = self._gt_in_memory_cached(topo, n_target, float(h), self._model_kwargs)
                e_i, gap_i = self.exact_ground_state(
                    topo,
                    n_target,
                    float(h),
                    model=self._physics_model,
                    model_kwargs=self._model_kwargs,
                )
            else:
                _was_cached = gt_cache.get(topo, n_target, self._physics_model, float(h)) is not None
                e_i, gap_i = gt_cache.get_or_compute(
                    topo,
                    n_target,
                    self._physics_model,
                    float(h),
                    flush=False,
                    solver=solver,
                )
            e_exact_arr[i] = e_i
            gap_arr[i] = gap_i
            if _was_cached:
                gt_hits += 1
            else:
                gt_misses += 1
                logger.info(
                    f"  GT [{gt_hits + gt_misses}/{len(self._h_values)}] "
                    f"h={float(h):.3f} E={e_i:.6f} "
                    f"gap={gap_i:.4f} ({time.perf_counter() - t_gt:.1f}s)"
                )
        logger.info(f"  Ground truth: {gt_hits} cache hits, {gt_misses} computed")
        if gt_misses > 0:
            gt_cache.flush()  # Persist new ground truths immediately

        # ── Refresh NPZ ground truth from GroundTruthCache ────────────────
        # If GT was recomputed with a better solver (eigsh vs DMRG), update
        # the NPZ e_exact so that ΔE/gap metrics reflect the latest values.
        if npz_path.exists():
            from qmbp_simulation.framework.result_io import refresh_npz_ground_truth

            n_refreshed = refresh_npz_ground_truth(
                npz_path,
                topology=topo,
                n_qubits=n_target,
                model=self._physics_model,
            )
            if n_refreshed > 0:
                # Reload prev_theta_by_h since energies haven't changed
                # but the variational principle check uses e_exact_arr
                # (which we just updated from GT cache above)
                logger.info(f"  NPZ ground truth refreshed: {n_refreshed} points updated")

        # ── Load best model (unified selection: per-topo + MT + single-N) ──
        model = None
        _meta = None

        # Explicit --checkpoint: honor it exactly (use only this model), matching
        # the predict-only contract of section_cross_n_predict. Resolves as a
        # path, exact zoo name, or fuzzy tag/fragment (e.g. "h_0p5_1p5").
        # Without this, iterative-improve silently ignored --checkpoint and fell
        # back to load_best_model_for's zoo "best" selection.
        _ckpt = getattr(self._args, "checkpoint", None)
        if _ckpt:
            from qmbp_simulation.predictors.model_zoo import (
                _smart_load_checkpoint,
                resolve_checkpoint_fuzzy,
            )

            ckpt_path = resolve_checkpoint_fuzzy(str(_ckpt), topology=topo, p_layers=p)
            if ckpt_path is None:
                candidates = resolve_checkpoint_fuzzy(str(_ckpt), return_all=True)
                names = [pth.name for pth, _ in (candidates or [])[:5]]
                hint = f" Closest checkpoints on disk: {names}" if names else ""
                return {
                    "pass": False,
                    "error": (
                        f"--checkpoint '{_ckpt}' not found as a file path, exact zoo name, or fuzzy match.{hint}"
                    ),
                }
            model = _smart_load_checkpoint(str(ckpt_path))
            model.eval()
            if getattr(model, "use_residual", False) and not use_residual:
                use_residual = True
                logger.info("  Auto-detected use_residual=True from loaded model")
            logger.info(f"  Loaded EXPLICIT checkpoint: {ckpt_path.name}")

        if model is None:
            try:
                from qmbp_simulation.predictors.model_zoo import load_best_model_for

                model, _meta, _source = load_best_model_for(
                    topo,
                    model=self._physics_model,
                    p_layers=p,
                    n_target=n_target,
                    include_multi_topology=True,
                )
                model.eval()
                # Auto-detect architecture from loaded model (issue #7)
                if getattr(model, "use_residual", False) and not use_residual:
                    use_residual = True
                    logger.info("  Auto-detected use_residual=True from loaded model")
                logger.info(
                    f"  Loaded model [{_source}]: {_meta.checkpoint_file} "
                    f"(pass={_meta.pass_rate:.0%}, {_meta.n_training_points} pts)"
                )
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.debug(f"  Unified model loading failed: {e}")

        if model is None:
            try:
                model, _meta = load_pretrained(
                    model=self._physics_model,
                    topology=topo,
                    n_qubits=self._args.train_n,
                    p_layers=p,
                    allow_cross_n=True,
                )
                logger.info(f"  Loaded single-N model from zoo: {_meta.checkpoint_file}")
                # Auto-detect architecture from single-N fallback (issue #7)
                if getattr(model, "use_residual", False) and not use_residual:
                    use_residual = True
                    logger.info("  Auto-detected use_residual=True from single-N model")
            except FileNotFoundError:
                # --from-zoo: never bootstrap/train. Fail clean if no zoo model.
                if getattr(self._args, "from_zoo", False):
                    return {
                        "pass": False,
                        "error": (
                            f"No model in zoo for {topo} p={p} (--from-zoo set, "
                            "bootstrap/training disabled). Train a model first or "
                            "drop --from-zoo."
                        ),
                    }
                # Bootstrap: no model exists → run AcceleratedVQE to generate
                # initial training data, then train a UnifiedMPNN from it.
                logger.info("  No model in zoo — bootstrapping via AcceleratedVQE...")
                from qmbp_simulation.pipeline.accelerated import (
                    AcceleratedConfig,
                    AcceleratedVQE,
                )

                boot_config = AcceleratedConfig(
                    n_anchors=self._args.n_anchors,
                    n_restarts=self._args.n_restarts,
                    maxiter=self._args.maxiter,
                    mpnn_epochs=FULL_TRAIN_EPOCHS,
                    use_zoo=False,
                    force_method=getattr(self._args, "force_method", None),
                    bidirectional_anchors=getattr(self._args, "bidirectional_anchors", False),
                )
                boot_lattice = self.make_lattice(topo, n_target, J=1.0, h=2.0)
                boot_circuit, _ = self._build_circuit(n_target, p, boot_lattice)
                t_boot = time.perf_counter()
                accel = AcceleratedVQE(
                    boot_lattice,
                    boot_circuit,
                    spec,
                    eval_backend,
                    config=boot_config,
                    p_layers=p,
                )
                boot_result = accel.run(self._h_values, seed=42, p_layers=p)
                logger.info(
                    f"  Bootstrap done: pass_rate_dual={boot_result.pass_rate:.0%}, "
                    f"time={time.perf_counter() - t_boot:.1f}s"
                )
                # Save bootstrap data to NPZ (atomic with anti-regression)
                # Map method labels to quality tiers for bootstrap data
                boot_quality_tiers = [
                    "verified" if m in ("vqe_full", "vqe_refined", "mpnn_refined") else "approximate"
                    for m in boot_result.method
                ]
                self.persist_theta_npz(
                    npz_path,
                    h_new=self._h_values[: len(boot_result.theta_opt)],
                    theta_new=boot_result.theta_opt,
                    e_vqe_new=boot_result.energies,
                    e_exact_new=boot_result.e_exact,
                    gaps_new=boot_result.gaps,
                    method_new=list(boot_result.method),
                    quality_tier_new=boot_quality_tiers,
                )
                # Reload prev_theta_by_h from freshly saved NPZ
                for i, h in enumerate(self._h_values[: len(boot_result.theta_opt)]):
                    th_i = boot_result.theta_opt[i]
                    if np.all(np.isfinite(th_i)) and len(th_i) == n_params:
                        prev_theta_by_h[round(float(h), 2)] = (th_i, float(boot_result.energies[i]))
                # Train initial UnifiedMPNN from bootstrap data (p-scoped: only *_p{p}.npz)
                agg = MultiNAggregator(
                    topology=topo,
                    model=self._physics_model,
                    frustrated=self._is_frustrated,
                    results_dir=self._training_data_dir,
                    max_n=self.N_MAX_VIABLE.get(topo, 20),
                    p_layers=p,
                    h_min=getattr(self._args, "train_h_min", None),
                    h_max=getattr(self._args, "train_h_max", None),
                )
                agg.scan()
                dataset = agg.build_combined_dataset(max_de_gap=0.15)
                if len(dataset) < 3:
                    return {
                        "pass": False,
                        "error": f"Bootstrap produced only {len(dataset)} valid points.",
                    }
                sample_g = dataset[0]
                n_node_features = sample_g.x.shape[1] if hasattr(sample_g, "x") else 4
                model = UnifiedMPNN(
                    node_features=n_node_features,
                    hidden_dim=256,
                    n_layers=3,
                    norm_type="none",
                    dropout=0.1,
                    use_residual=use_residual,
                    film_conditioning=getattr(self._args, "film", False),
                )

                boot_train_result = train_unified_mpnn(
                    model, dataset, n_epochs=FULL_TRAIN_EPOCHS, lr=1e-3, patience=200, seed=42
                )
                # Register in zoo with training metrics
                from datetime import datetime

                entry = ZooEntry(
                    model=self._physics_model,
                    topology=topo,
                    n_qubits=0,
                    p_layers=p,
                    checkpoint_file=(
                        f"unifMPNN__{topo}_p{p}_{self._args.model_name}.pt"
                        if getattr(self._args, "model_name", None)
                        else f"unified_tfim_br_{topo}_multiN_{n_target}_p{p}.pt"
                    ),
                    h_range=self._training_h_range(),
                    pass_rate=boot_result.pass_rate,
                    n_training_points=len(dataset),
                    seeds=[42],
                    created=datetime.now(UTC).isoformat(),
                    notes=f"Bootstrap from AcceleratedVQE N={n_target}"
                    + self._train_h_range_note()
                    + (", arch=residual" if use_residual else ""),
                )
                register_checkpoint_with_training_metrics(
                    model,
                    entry,
                    training_result=boot_train_result,
                    overwrite=True,
                    architecture_config={
                        "hidden_dim": 256,
                        "n_conv_layers": 3,
                        "norm_type": "none",
                        "dropout": 0.1,
                        "use_residual": use_residual,
                    },
                )
                logger.info(f"  Bootstrap model registered: {entry.checkpoint_file}")

        # ── Iterative improvement loop ────────────────────────────────────
        iteration_reports = []
        total_vqe_calls = 0
        prev_pass_rate = 0.0
        convergence_reason = "max_iterations"
        # Track the best pass_rate ever exported to zoo for Fix C.
        # Use _meta.pass_rate if available, but also check the zoo manifest
        # for the ACTUAL best model for this config — _meta could come from
        # a fallback load_pretrained with pass_rate=0 (unevaluated).
        zoo_best_pass_rate = _meta.pass_rate if _meta is not None else 0.0
        if zoo_best_pass_rate < 0.01:
            try:
                from qmbp_simulation.predictors.model_zoo import _load_manifest

                _existing = [
                    e
                    for e in _load_manifest()
                    if e.topology == topo and e.model == self._physics_model and e.p_layers == p and e.n_qubits == 0
                ]
                if _existing:
                    zoo_best_pass_rate = max(e.pass_rate for e in _existing)
                    if zoo_best_pass_rate > 0:
                        logger.info(
                            f"  Zoo baseline: existing best pass_rate={zoo_best_pass_rate:.0%} "
                            f"(loaded model was unevaluated)"
                        )
            except Exception:
                pass

        for iteration in range(1, max_iterations + 1):
            logger.info(f"\n  ╔══ Iteration {iteration}/{max_iterations} ══════════════════════╗")
            t_iter_start = time.perf_counter()
            model.eval()

            # ── 2a: Predict θ for all h-points ────────────────────────────
            predictions = []
            n_pred_invalid = 0
            for h in self._h_values:
                g = self._build_graph(
                    lattice_target,
                    float(h),
                    p,
                    include_circuit_nodes=True,
                )
                with torch.no_grad():
                    pred = model(g).numpy().flatten()
                # NaN/Inf guard on predictions
                if not np.all(np.isfinite(pred)):
                    n_bad = int(np.sum(~np.isfinite(pred)))
                    logger.warning(f"    h={float(h):.2f}: prediction has {n_bad} NaN/Inf → zeroed")
                    pred = np.where(np.isfinite(pred), pred, 0.0)
                    n_pred_invalid += 1
                pred = np.clip(pred, -np.pi, np.pi)
                if len(pred) != n_params:
                    if len(pred) < n_params:
                        pred = np.pad(pred, (0, n_params - len(pred)))
                    else:
                        pred = pred[:n_params]
                predictions.append(pred)
            predictions = np.array(predictions)
            if n_pred_invalid > 0:
                logger.warning(f"  │ ⚠️ {n_pred_invalid}/{len(self._h_values)} predictions had NaN/Inf")

            # ── 2b: Evaluate + identify failures ──────────────────────────
            # CRITICAL: On iteration 2+, ALWAYS evaluate MPNN predictions
            # even for points with stored θ_prev. The retrained model may
            # have found a better energy basin that we'd miss by skipping.
            energies = np.zeros(len(self._h_values))
            eval_hits = 0
            for i, h in enumerate(self._h_values):
                h_key = round(float(h), 2)

                # Always evaluate the MPNN prediction (cheap via eval_cache)
                key = eval_cache.make_key(
                    topo,
                    n_target,
                    float(h),
                    predictions[i],
                    model=self._physics_model,
                    p_layers=p,
                )
                cached_e = eval_cache.get(key)
                if cached_e is not None:
                    e_pred_i = cached_e
                    eval_hits += 1
                else:
                    lat_h = self.make_lattice(topo, n_target, J=1.0, h=float(h))
                    H = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)
                    e_pred_i = eval_backend.evaluate(circuit_target, H, predictions[i])
                    eval_cache.put(key, float(e_pred_i))

                # Compare with prev_theta_by_h: keep the lower energy
                if h_key in prev_theta_by_h:
                    theta_prev, e_prev = prev_theta_by_h[h_key]
                    if e_prev is not None and np.isfinite(e_prev):
                        # Variational principle check on stored energy
                        if e_prev >= e_exact_arr[i] - 1e-6:
                            if e_prev <= e_pred_i:
                                # Previous θ is still best → use it
                                energies[i] = e_prev
                                continue
                            else:
                                # MPNN found better energy → update tracking
                                energies[i] = float(e_pred_i)
                                prev_theta_by_h[h_key] = (predictions[i], float(e_pred_i))
                                continue
                        else:
                            logger.warning(
                                f"    h={float(h):.2f}: NPZ energy {e_prev:.6f} "
                                f"violates variational principle (E_exact={e_exact_arr[i]:.6f}). "
                                f"Invalidating energy but keeping θ for warm-start."
                            )
                            # Keep θ for VQE init later, but invalidate stored energy
                            prev_theta_by_h[h_key] = (theta_prev, None)

                # No prev or prev invalidated → use MPNN prediction energy
                energies[i] = float(e_pred_i)

            de_gaps = np.abs(energies - e_exact_arr) / np.maximum(gap_arr, 1e-10)
            abs_errors = np.abs(energies - e_exact_arr)
            from qmbp_simulation.analysis.metrics import DE_GAP_THRESHOLD, MAX_ABS_ERROR

            dual_mask = (de_gaps < DE_GAP_THRESHOLD) & (abs_errors < MAX_ABS_ERROR)
            pass_rate = float(dual_mask.mean())
            pass_rate_5pct = float((de_gaps < 0.05).mean())
            logger.info(
                f"  │ Eval: {eval_hits}/{len(self._h_values)} eval_cache hits, "
                f"pass_rate_dual={pass_rate:.0%} (5pct_only={pass_rate_5pct:.0%})"
            )

            # ── 2b.0: Persist ALL passing predictions IMMEDIATELY to NPZ ──
            # Predictions with ΔE/gap < 5% are as good as VQE-optimized θ_opt.
            # 1. Data is safe even if process crashes before convergence check
            # 2. Multi-N aggregator sees fresh data for this config immediately
            # 3. Anti-regression: upsert_theta_npz only updates if energy improves
            n_newly_persisted = 0
            for i, h in enumerate(self._h_values):
                if de_gaps[i] >= 0.05:
                    continue  # only persist passing predictions
                h_key = round(float(h), 2)
                # Only persist if not already in memory dict OR if new energy is lower
                if h_key in prev_theta_by_h:
                    _, e_existing = prev_theta_by_h[h_key]
                    if e_existing is not None and energies[i] >= e_existing - 1e-10:
                        continue  # existing is already as good or better

                # Persist immediately to NPZ (atomic write handles concurrency)
                n_upd, n_add = self.persist_theta_npz(
                    npz_path,
                    h_new=np.array([float(h)]),
                    theta_new=np.array([predictions[i]]),
                    e_vqe_new=np.array([float(energies[i])]),
                    e_exact_new=np.array([float(e_exact_arr[i])]),
                    gaps_new=np.array([float(gap_arr[i])]),
                    method_new=["mpnn_pred"],
                    quality_tier_new=["approximate"],
                )
                # Update in-memory dict to avoid re-persisting same point
                prev_theta_by_h[h_key] = (predictions[i], float(energies[i]))
                if n_upd > 0 or n_add > 0:
                    n_newly_persisted += 1

            if n_newly_persisted > 0:
                logger.info(
                    f"  │ Persisted {n_newly_persisted} new passing predictions → NPZ "
                    f"(total in memory: {len(prev_theta_by_h)} points)"
                )

            # Points with stored θ_prev that pass dual criterion AND where the
            # MPNN independently agrees (e_pred ≈ e_prev) can be promoted to
            # "verified" without VQE — the agreement is sufficient evidence.
            n_promoted = 0
            for i, h in enumerate(self._h_values):
                if not dual_mask[i]:
                    continue  # Only promote passing points
                h_key = round(float(h), 2)
                if h_key not in prev_theta_by_h:
                    continue
                theta_prev, e_prev = prev_theta_by_h[h_key]
                if e_prev is None:
                    continue
                # Confirm MPNN agrees: e_pred is close to e_prev (same basin)
                if abs(energies[i] - e_prev) > 0.01:
                    continue  # Significant disagreement → don't promote
                # Promote via upsert (tier upgrade path: "approximate"→"verified")
                self.persist_theta_npz(
                    npz_path,
                    h_new=np.array([float(h)]),
                    theta_new=np.array([theta_prev]),
                    e_vqe_new=np.array([e_prev]),
                    e_exact_new=np.array([float(e_exact_arr[i])]),
                    gaps_new=np.array([float(gap_arr[i])]),
                    method_new=["mpnn_confirmed"],
                    quality_tier_new=["verified"],
                )
                n_promoted += 1
            if n_promoted > 0:
                logger.info(f"  │ Promoted {n_promoted} approximate→verified (MPNN confirms)")

            # ── 2b.1: Check convergence (early stop) ─────────────────────
            if pass_rate >= 0.90:
                convergence_reason = "target_reached"
                logger.info(f"  │ ✓ Target reached: pass_rate_dual={pass_rate:.0%} ≥ 90%")

                n_safety_persisted = 0
                for i, h in enumerate(self._h_values):
                    h_key = round(float(h), 2)
                    if h_key not in prev_theta_by_h and de_gaps[i] < 0.05:
                        self.persist_theta_npz(
                            npz_path,
                            h_new=np.array([float(h)]),
                            theta_new=np.array([predictions[i]]),
                            e_vqe_new=np.array([float(energies[i])]),
                            e_exact_new=np.array([float(e_exact_arr[i])]),
                            gaps_new=np.array([float(gap_arr[i])]),
                            method_new=["mpnn_pred"],
                            quality_tier_new=["approximate"],
                        )
                        prev_theta_by_h[h_key] = (predictions[i], float(energies[i]))
                        n_safety_persisted += 1
                if n_safety_persisted > 0:
                    logger.info(f"  │ Safety net: persisted {n_safety_persisted} extra points")

                iteration_reports.append(
                    self._build_iter_report(iteration, pass_rate, 0, 0, eval_hits, time.perf_counter() - t_iter_start)
                )
                break

            improvement = pass_rate - prev_pass_rate
            if iteration > 1 and improvement < improvement_threshold:
                convergence_reason = "no_improvement"
                logger.info(f"  │ ✓ Converged: improvement={improvement:.4f} < threshold={improvement_threshold}")
                # Note: All passing predictions were already persisted immediately. No bulk save needed.
                eval_cache.flush()
                iteration_reports.append(
                    self._build_iter_report(iteration, pass_rate, 0, 0, eval_hits, time.perf_counter() - t_iter_start)
                )
                break

            # ── 2c: Identify failures + ansatz-limit filter ───────────────
            # Dual energy criterion (ΔE/gap OR |ΔE|) with a fidelity override:
            # a point with high EXACT fidelity is accepted even when ΔE/gap is
            # large (the tiny-gap near-critical case). See is_point_failure.
            from qmbp_simulation.analysis.metrics import (
                compute_refinement_priority,
                is_point_failure,
            )

            # Compute exact fidelity per point when cheap (statevector regime) so
            # the fidelity override can rescue near-critical high-fidelity points.
            _fid_statevector = n_target <= STATEVECTOR_MAX_N
            _min_fid = getattr(self._args, "data_gen_min_fidelity", 0.95)

            failures = []
            ansatz_limited = []
            for i, h in enumerate(self._h_values):
                abs_err_i = abs(energies[i] - e_exact_arr[i])
                fid_i = None
                if _fid_statevector:
                    h_key = round(float(h), 2)
                    theta_eval = predictions[i]
                    if h_key in prev_theta_by_h and prev_theta_by_h[h_key][0] is not None:
                        theta_eval = prev_theta_by_h[h_key][0]
                    fid_i = self.safe_compute_fidelity(
                        circuit_target, theta_eval, topo, n_target, float(h), model=self._physics_model
                    )
                is_fail = is_point_failure(
                    de_gap=de_gaps[i],
                    abs_error=abs_err_i,
                    fidelity=fid_i,
                    min_fidelity=_min_fid,
                )
                if is_fail:
                    if h_min_valid > 0 and float(h) < h_min_valid:
                        ansatz_limited.append(i)
                    else:
                        failures.append(i)

            if not failures:
                if ansatz_limited:
                    convergence_reason = "ansatz_limit"
                    logger.info(
                        f"  │ ✓ All remaining failures ({len(ansatz_limited)}) are in "
                        f"ansatz-limited zone (h < {h_min_valid:.2f})"
                    )
                else:
                    convergence_reason = "target_reached"
                # Note: Data already persisted immediately. Just flush cache.
                eval_cache.flush()
                iteration_reports.append(
                    self._build_iter_report(
                        iteration,
                        pass_rate,
                        0,
                        len(ansatz_limited),
                        eval_hits,
                        time.perf_counter() - t_iter_start,
                    )
                )
                break

            # ── 2c.1: Priority-score failures and sort ─────────────────────
            # Use compute_refinement_priority to order failures by expected
            # return-on-investment. High-priority points get refined first.
            # Track per-h attempt counts for stale detection.
            if not hasattr(self, "_refine_attempts"):
                self._refine_attempts = {}  # h_key → count of failed VQE attempts

            scored_failures = []
            n_skipped_priority = 0
            for idx in failures:
                h = float(self._h_values[idx])
                h_key = round(h, 2)
                abs_err_i = abs(energies[idx] - e_exact_arr[idx])

                # Previous VQE energy (if refined before)
                e_prev_h = None
                if h_key in prev_theta_by_h:
                    _, e_prev_h = prev_theta_by_h[h_key]

                n_attempts = self._refine_attempts.get(h_key, 0)

                priority, should_skip, reason = compute_refinement_priority(
                    de_gap=de_gaps[idx],
                    abs_error=abs_err_i,
                    gap=gap_arr[idx],
                    n_params=n_params,
                    e_prev=e_prev_h,
                    e_pred=float(energies[idx]),
                    n_prev_attempts=n_attempts,
                )

                # Data-generation mode never skips a failing point: every point
                # is a datum we want solved, even "hopeless"-looking ones.
                if should_skip and not getattr(self._args, "data_gen", False):
                    n_skipped_priority += 1
                    logger.debug(f"    Skip h={h:.3f}: {reason} (priority={priority:.2f})")
                else:
                    scored_failures.append((priority, idx, reason))

            # Sort by priority (highest first) and limit
            scored_failures.sort(key=lambda x: x[0], reverse=True)
            if getattr(self._args, "refine_all", False):
                max_refine = len(scored_failures)
            else:
                max_refine = getattr(self._args, "max_refine_per_iter", None)
                if max_refine is None:
                    max_refine = min(len(scored_failures), DEFAULT_MAX_REFINE_PER_ITER)
            failures = [idx for _, idx, _ in scored_failures[:max_refine]]

            logger.info(
                f"  │ Failures: {len(failures)} to refine "
                f"(scored {len(scored_failures)}, skipped {n_skipped_priority}, "
                f"max={max_refine}), {len(ansatz_limited)} ansatz-limited"
            )
            if scored_failures:
                top = scored_failures[0]
                logger.info(
                    f"  │ Top priority: h={float(self._h_values[top[1]]):.3f} score={top[0]:.2f} reason={top[2]}"
                )

            # ── 2d: Anti-regression + VQE refine ─────────────────────────
            refined_h = []
            refined_theta = []
            refined_energies = []
            refined_e_exact = []
            refined_provenance = []  # warm-start seed label per improved point (StudyPersister-style)

            # Use adaptive VQE config per-point based on priority score.
            # High-priority (easy wins) get minimal budget; low-priority get full budget.
            from qmbp_simulation.analysis.metrics import compute_adaptive_vqe_config

            refine_method = self._args.force_method or "L-BFGS-B"
            base_maxiter = self._args.maxiter
            base_restarts = self._args.n_restarts
            if refine_method == "L-BFGS-B":
                base_restarts = min(10, base_restarts)
            else:
                base_restarts = max(7, base_restarts // 2)

            for fail_idx_pos, idx in enumerate(failures):
                h = float(self._h_values[idx])
                h_key = round(h, 2)
                t_refine_start = time.perf_counter()

                # Adaptive VQE config: scale budget based on priority
                fail_priority = scored_failures[fail_idx_pos][0] if fail_idx_pos < len(scored_failures) else 0.5
                adaptive_cfg = compute_adaptive_vqe_config(
                    priority=fail_priority,
                    de_gap=de_gaps[idx],
                    gap=gap_arr[idx],
                    n_params=n_params,
                    base_maxiter=base_maxiter,
                    base_restarts=base_restarts,
                )
                refine_maxiter = adaptive_cfg["maxiter"]
                refine_restarts = adaptive_cfg["n_restarts"]

                # Data-generation mode inverts the deployment-oriented budget:
                # the adaptive tiers starve hard low-h points ("minimal", 1
                # restart) because they look hopeless for *deployment* — but for
                # *data generation* those are exactly the points we must solve
                # well. Give every failing point the full CLI budget instead.
                if getattr(self._args, "data_gen", False):
                    refine_maxiter = base_maxiter
                    refine_restarts = max(base_restarts, refine_restarts)
                    adaptive_cfg = {**adaptive_cfg, "tier": "data_gen_full"}

                # Gentle gap/difficulty-adaptive restarts: spend a FEW extra
                # restarts only on near-critical points (difficulty_index =
                # N·phase_proximity high), where the landscape has competing
                # basins; easy ordered/paramagnetic points keep the base count.
                # budget_for_difficulty is difficulty-aware (the study found raw
                # gap mis-scales the budget) and only adds +1/+2 over base,
                # clipped to a small ceiling — kept light on purpose.
                from qmbp_simulation.analysis.warmstart import budget_for_difficulty

                _adap_restarts, _, _diff = budget_for_difficulty(
                    n_target, h, float(gap_arr[idx]),
                    base_restarts=refine_restarts,
                    J2=float(self._model_kwargs.get("J2", 0.0) or 0.0),
                    max_restarts=refine_restarts + 2,
                )
                if _adap_restarts > refine_restarts:
                    refine_restarts = _adap_restarts

                logger.info(
                    f"  │ Refining [{fail_idx_pos + 1}/{len(failures)}] "
                    f"h={h:.4f} (ΔE/gap={de_gaps[idx]:.4f}, "
                    f"tier={adaptive_cfg['tier']}, maxiter={refine_maxiter}, "
                    f"restarts={refine_restarts}, D={_diff:.1f})..."
                )
                sys.stdout.flush()
                sys.stderr.flush()
                for handler in logging.getLogger().handlers:
                    handler.flush()

                # Anti-regression: evaluate BOTH θ_pred and θ_prev, pick best as VQE init
                from qmbp_simulation.framework.result_io import select_best_theta_init

                theta_prev_h = None
                e_prev_h_val = None
                if h_key in prev_theta_by_h:
                    theta_prev_h, e_prev_h_val = prev_theta_by_h[h_key]

                def _eval_theta(theta):
                    """Evaluate θ energy via cache."""
                    _key = eval_cache.make_key(
                        topo,
                        n_target,
                        h,
                        theta,
                        model=self._physics_model,
                        p_layers=p,
                    )
                    _cached = eval_cache.get(_key)
                    if _cached is not None:
                        return _cached
                    _lat = self.make_lattice(topo, n_target, J=1.0, h=h)
                    _H = spec.build_hamiltonian(_lat, **spec.hamiltonian_kwargs)
                    _e = eval_backend.evaluate(circuit_target, _H, theta)
                    eval_cache.put(_key, float(_e))
                    return float(_e)

                theta_init, best_e = select_best_theta_init(
                    theta_pred=predictions[idx],
                    e_pred=energies[idx],
                    theta_prev=theta_prev_h,
                    e_prev=e_prev_h_val,
                    eval_fn=_eval_theta,
                )
                if best_e < energies[idx]:
                    energies[idx] = best_e
                    logger.debug(f"    h={h:.2f}: anti-regression → using θ_prev (E={best_e:.6f})")

                # VQE warm-start refinement
                lat_h = self.make_lattice(topo, n_target, J=1.0, h=h)
                H = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)

                # ── Structure-aware warm-start seed (statevector regime only) ──
                # Upgrade theta_init with the project's combined warm-start cascade
                # (analytic Ising seeds + regime + donors + adaptive micro-descent,
                # all via the fast adjoint gradient). This is the single
                # integration point `prepare_warmstart`; it needs the exact ground
                # eigenvector ``psi``, so it only runs when N is in the statevector
                # regime (N ≤ STATEVECTOR_MAX_N). Above that it is a no-op and the
                # refine keeps its MPNN/θ_prev init unchanged. The seed is adopted
                # only when it beats the current init by energy (anti-regression).
                ws_provenance = None
                if n_target <= STATEVECTOR_MAX_N:
                    # Continuation donor (cross-h, intra-sweep): the converged θ
                    # from the ADJACENT h-step already refined this sweep. The
                    # h-grid is descending, so the previous grid entry (idx-1) is
                    # the nearest higher-h neighbour. Within a phase this is the
                    # single best donor (same N → layout-identical transfer, same
                    # physics regime); it is dropped across a phase boundary where
                    # θ does not transfer. Same N so no cross-N approximation.
                    cont_donor = self._continuation_donor(idx, p, lat_h, prev_theta_by_h)
                    try:
                        theta_init, best_e, ws_provenance = self._apply_warmstart_seed(
                            circuit=circuit_target,
                            H=H,
                            lattice=lat_h,
                            h=h,
                            p_layers=p,
                            gap=float(gap_arr[idx]),
                            theta_init=theta_init,
                            e_init=best_e,
                            eval_fn=_eval_theta,
                            continuation_donor=cont_donor,
                        )
                        if best_e < energies[idx]:
                            energies[idx] = best_e
                    except Exception as _ws_exc:  # noqa: BLE001
                        logger.debug(f"    h={h:.2f}: warm-start seed skipped ({_ws_exc})")

                try:
                    from qmbp_simulation import VQEConfig, VQEOptimizer

                    vqe_cfg = VQEConfig(
                        p_layers=p,
                        n_restarts=refine_restarts,
                        maxiter=refine_maxiter,
                        method=refine_method,
                        enable_callbacks=False,  # No trajectory logging (2x speedup)
                    )
                    vqe_opt = VQEOptimizer(config=vqe_cfg, backend=eval_backend, seed=42 + idx)
                    if hasattr(eval_backend, "set_h"):
                        eval_backend.set_h(h)
                    vqe_result = vqe_opt.optimize(H, circuit_target, initial_guess=theta_init)
                    total_vqe_calls += 1
                    e_refined = float(vqe_result.energy)
                    res_x = vqe_result.theta_opt
                    t_refine_elapsed = time.perf_counter() - t_refine_start
                    _ws_note = f" [seed={ws_provenance}]" if ws_provenance else ""
                    logger.info(f"    h={h:.2f}: VQE done in {t_refine_elapsed:.1f}s, E={e_refined:.6f}{_ws_note}")

                    # ── Validate refined result before storing ────────────
                    # 1. Energy must be finite
                    if not np.isfinite(e_refined):
                        logger.warning(f"    h={h:.2f}: VQE returned NaN/Inf energy. Skipping.")
                        continue
                    # 2. θ must be finite
                    if not np.all(np.isfinite(res_x)):
                        logger.warning(f"    h={h:.2f}: VQE returned NaN/Inf θ. Skipping.")
                        continue
                    # 3. Variational principle: E_refined >= E_exact (within tolerance)
                    if e_refined < e_exact_arr[idx] - 1e-4:
                        violation = e_exact_arr[idx] - e_refined
                        logger.warning(
                            f"    h={h:.2f}: Variational violation Δ={violation:.2e}. "
                            f"May indicate stale E_exact (DMRG approx). Accepting anyway."
                        )
                    # 4. Only accept if improved over current energy
                    #    Use meaningful threshold to avoid "false improvements"
                    #    where VQE converges to same minimum with numerical noise.
                    from qmbp_simulation.models.constants import VQE_RESTART_IMPROVEMENT_TOL

                    energy_improvement = energies[idx] - e_refined
                    if energy_improvement > VQE_RESTART_IMPROVEMENT_TOL:
                        de_gap_new = abs(e_refined - e_exact_arr[idx]) / max(gap_arr[idx], 1e-10)
                        abs_err_old = abs(energies[idx] - e_exact_arr[idx])
                        abs_err_new = abs(e_refined - e_exact_arr[idx])
                        # Only log as improvement if ΔE/gap actually changed visibly
                        if abs(de_gaps[idx] - de_gap_new) > 1e-4:
                            # Compute state fidelity when feasible (N ≤ statevector limit).
                            # safe_compute_fidelity returns None for large N or on error.
                            fid = self.safe_compute_fidelity(
                                circuit_target,
                                res_x,
                                topo,
                                n_target,
                                h,
                                model=self._physics_model,
                            )
                            fid_str = f" F={fid:.4f}" if fid is not None else ""
                            logger.info(
                                f"    h={h:.2f}: ΔE/gap {de_gaps[idx]:.4f} → {de_gap_new:.4f} "
                                f"|ΔE| {abs_err_old:.3f} → {abs_err_new:.3f}{fid_str} ✓"
                            )
                        refined_h.append(h)
                        refined_theta.append(res_x.copy())
                        refined_energies.append(e_refined)
                        refined_e_exact.append(float(e_exact_arr[idx]))
                        refined_provenance.append(ws_provenance or "no_warmstart")
                        # Update tracking
                        prev_theta_by_h[h_key] = (res_x.copy(), e_refined)
                        # Reset stale counter — this point just improved
                        self._refine_attempts.pop(h_key, None)

                        # ── Immediate persist: NPZ upsert per-point ───────
                        # Ensures no refined θ is lost on interrupt. Record the
                        # warm-start provenance (which seed fed this VQE) in the
                        # source_ckpt field so later analysis knows WHICH
                        # technique produced each training point (analytic
                        # calibrated/structural/second_order vs transferred NPZ
                        # donor vs cross-h continuation). Empty when the warm-start
                        # seed was not adopted (MPNN/prev init kept).
                        gap_i = float(gap_arr[idx])
                        _ws_src = f"warmstart:{ws_provenance}" if ws_provenance else "vqe_refined"
                        self.persist_theta_npz(
                            npz_path,
                            np.array([h]),
                            np.array([res_x]),
                            np.array([e_refined]),
                            np.array([float(e_exact_arr[idx])]),
                            gaps_new=np.array([gap_i]),
                            method_new=["vqe_refined"],
                            quality_tier_new=["verified"],
                            source_ckpt_new=[_ws_src],
                        )
                        # Note: eval_cache auto-flushes every 50 puts.
                        # Full flush deferred to end of iteration (avoid 5MB
                        # JSON write per point).
                    else:
                        logger.info(f"    h={h:.2f}: no improvement (VQE stuck)")
                        # Track failed attempt for priority scoring
                        self._refine_attempts[h_key] = self._refine_attempts.get(h_key, 0) + 1
                except KeyboardInterrupt:
                    # Persist everything refined so far before re-raising
                    logger.warning(
                        f"  │ ⚠️ Interrupted during refinement. {len(refined_h)} points already saved to NPZ."
                    )
                    eval_cache.flush()
                    raise
                except Exception as e:
                    logger.warning(f"    h={h:.2f}: refinement failed: {e}")

            # ── 2e: Summary (NPZ already persisted per-point above) ───────
            n_updated = len(refined_h)  # all persisted incrementally via upsert_theta_npz
            if refined_h:
                from collections import Counter

                _prov_tally = Counter(refined_provenance)
                _prov_str = ", ".join(f"{k}×{v}" for k, v in _prov_tally.most_common())
                logger.info(
                    f"  │ VQE refinement: {n_updated} points improved and persisted "
                    f"(warm-start seeds: {_prov_str})"
                )

            # Note: All data (predictions + refinements) was persisted immediately
            # via upsert_theta_npz calls above. No bulk save needed.

            # Flush eval cache once per iteration (not per-point)
            eval_cache.flush()

            # ── 2f: Retrain multi-N model ─────────────────────────────────
            # Fix A: Skip retrain if no new data was produced
            # Fix B: Fine-tune instead of training from scratch
            # Fix C: Use diagnostics-aware should_retrain to detect contamination
            from qmbp_simulation.predictors.unified_mpnn import (
                fine_tune_unified_mpnn,
                should_retrain_with_diagnostics,
            )

            # p-scoped aggregation: only *_p{p}.npz data (no cross-p mixing)
            agg = MultiNAggregator(
                topology=topo,
                model=self._physics_model,
                frustrated=self._is_frustrated,
                results_dir=self._training_data_dir,
                max_n=self.N_MAX_VIABLE.get(topo, 20),
                h_min=getattr(self._args, "train_h_min", None),
                h_max=getattr(self._args, "train_h_max", None),
                p_layers=p,
            )
            agg.scan()
            dataset = agg.build_combined_dataset(max_de_gap=0.10)

            # ── Pre-retrain quality check (WARNING-ONLY) ──────────────────
            # Uses base class helper: logs warnings but never aborts.
            # The AL loop is progressive — quality improves each iteration.
            self.warn_training_quality(agg._data_by_n)

            do_retrain, retrain_reason, diagnostics = should_retrain_with_diagnostics(
                topology=topo,
                model_name=self._physics_model,
                p_layers=p,
                n_new_points=len(refined_h),
                current_pass_rate=pass_rate,
                prev_pass_rate=prev_pass_rate,
                dataset_size=len(dataset),
            )

            # --from-zoo / --skip-retrain: never retrain the MPNN. Keep refining
            # VQE points and persisting them to NPZ, but use only the loaded
            # model for prediction.
            if getattr(self._args, "from_zoo", False):
                do_retrain = False
                retrain_reason = "from_zoo (retraining disabled)"
            elif getattr(self._args, "skip_retrain", False):
                do_retrain = False
                retrain_reason = "skip_retrain (retraining disabled)"

            # Log diagnostic info if contamination or other issues detected
            if diagnostics.get("failure_mode") in ("contaminated_training", "gap_masking"):
                logger.warning(
                    f"  │ ⚠️ Diagnostic: failure_mode={diagnostics['failure_mode']}, "
                    f"training_utility={diagnostics.get('training_utility')}"
                )

            if do_retrain and len(dataset) >= 5:
                logger.info(
                    f"  │ Retraining (reason={retrain_reason}, {len(refined_h)} new points, {len(dataset)} total)..."
                )
                sample_g = dataset[0]
                n_node_features = sample_g.x.shape[1] if hasattr(sample_g, "x") else 4

                # Fix B: Fine-tune existing model if it has same architecture,
                # otherwise train from scratch (architecture mismatch).
                # Validates node_features, hidden_dim, n_layers, and use_residual.
                can_fine_tune = (
                    hasattr(model, "node_features")
                    and model.node_features == n_node_features
                    and getattr(model, "hidden_dim", 256) == 256
                    and getattr(model, "n_layers", 3) == 3
                    and getattr(model, "use_residual", False) == use_residual
                    and iteration > 1  # First iter after bootstrap → full train
                )

                if can_fine_tune:
                    logger.info(f"  │ Mode: fine-tune ({FINE_TUNE_EPOCHS} epochs, lr=3e-4)")
                    train_result = fine_tune_unified_mpnn(
                        model,
                        dataset,
                        n_epochs=FINE_TUNE_EPOCHS,
                        lr=3e-4,
                        patience=150,
                        seed=42,
                    )
                else:
                    logger.info(f"  │ Mode: full retrain ({FULL_TRAIN_EPOCHS} epochs, lr=1e-3)")
                    model = UnifiedMPNN(
                        node_features=n_node_features,
                        hidden_dim=256,
                        n_layers=3,
                        norm_type="none",
                        dropout=0.1,
                        use_residual=use_residual,
                        film_conditioning=getattr(self._args, "film", False),
                    )
                    train_result = train_unified_mpnn(
                        model,
                        dataset,
                        n_epochs=FULL_TRAIN_EPOCHS,
                        lr=1e-3,
                        patience=200,
                        seed=42,
                        loss_type=getattr(self._args, "loss_type", "sign_invariant"),
                        physics_loss_weight=getattr(self._args, "physics_loss_weight", 0.0),
                        fidelity_loss_weight=getattr(self._args, "fidelity_loss_weight", 0.1),
                    )

                mse = train_result.get("final_mse", 0) if isinstance(train_result, dict) else 0
                mode = train_result.get("mode", "full")
                logger.info(f"  │ Retrained ({mode}): MSE={mse:.2e}, {len(dataset)} points")

                # Persist training curve (auto, non-blocking)
                try:
                    from qmbp_simulation.utils.helpers import persist_training_curve

                    persist_training_curve(
                        train_result,
                        output_dir=Path("results/training_curves"),
                        prefix=f"{topo}_iter{iteration}_p{p}",
                    )
                except Exception:
                    pass

                # ── 2g: Export to zoo (only if pass_rate improved) ────────
                # Fix C: Don't overwrite a better model in the zoo with one
                # that didn't improve pass_rate.
                if pass_rate > zoo_best_pass_rate or iteration == 1:
                    from datetime import datetime

                    n_vals = agg.available_n_values()
                    n_str = "+".join(str(n) for n in n_vals)
                    entry = ZooEntry(
                        model=self._physics_model,
                        topology=topo,
                        n_qubits=0,
                        p_layers=p,
                        checkpoint_file=(
                            f"unifMPNN__{topo}_p{p}_{self._args.model_name}.pt"
                            if getattr(self._args, "model_name", None)
                            else f"unified_tfim_br_{topo}_multiN_{n_str}_p{p}.pt"
                        ),
                        h_range=self._training_h_range(),
                        pass_rate=pass_rate,
                        n_training_points=len(dataset),
                        seeds=[42],
                        created=datetime.now(UTC).isoformat(),
                        notes=f"Iterative improve iter {iteration}: N={n_vals}"
                        + self._train_h_range_note()
                        + (", arch=residual" if use_residual else ""),
                    )
                    register_checkpoint_with_training_metrics(
                        model,
                        entry,
                        training_result=train_result,
                        overwrite=True,
                        architecture_config={
                            "hidden_dim": 256,
                            "n_conv_layers": 3,
                            "norm_type": "none",
                            "dropout": 0.1,
                            "use_residual": use_residual,
                        },
                    )
                    zoo_best_pass_rate = pass_rate
                    logger.info(f"  │ Exported to zoo: {entry.checkpoint_file}")
                else:
                    logger.info(
                        f"  │ Zoo skip: pass_rate_dual={pass_rate:.0%} ≤ "
                        f"zoo_best={zoo_best_pass_rate:.0%} — keeping better model"
                    )
            elif not do_retrain:
                logger.info(
                    f"  │ Skipping retrain: {retrain_reason} (refined={len(refined_h)}, dataset={len(dataset)})"
                )
            else:
                logger.warning(f"  │ Only {len(dataset)} points — skipping retrain")

            # ── 2h: Report iteration ──────────────────────────────────────
            iter_time = time.perf_counter() - t_iter_start
            iteration_reports.append(
                self._build_iter_report(
                    iteration,
                    pass_rate,
                    len(refined_h),
                    len(ansatz_limited),
                    eval_hits,
                    iter_time,
                )
            )
            prev_pass_rate = pass_rate
            logger.info(
                f"  ╚══ Iteration {iteration} done: pass_rate_dual={pass_rate:.0%}, "
                f"refined={len(refined_h)}, time={iter_time:.1f}s ══╝"
            )

        # ── Final report ──────────────────────────────────────────────────
        eval_cache.flush()
        final_stats = eval_cache.stats()
        final_pass_rate = iteration_reports[-1]["pass_rate"] if iteration_reports else 0.0

        # ── Cross-N Validation Report (L1 from final iteration data) ──────
        cross_n_report = None
        try:
            from qmbp_simulation.analysis.cross_n_validator import quick_cross_n_report

            # Build per-h results from final energies for formal L1 report
            per_h_for_report = []
            for i, h in enumerate(self._h_values):
                per_h_for_report.append(
                    {
                        "h": float(h),
                        "e_pred": float(energies[i]),
                        "e_exact": float(e_exact_arr[i]),
                        "gap": float(gap_arr[i]),
                        "de_gap": float(de_gaps[i]),
                    }
                )

            # Training sizes = all N values in multi_n_training NPZs
            training_sizes = []
            try:
                from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

                _agg_report = MultiNAggregator(
                    topology=topo,
                    model=self._physics_model,
                    frustrated=self._is_frustrated,
                    results_dir=self._training_data_dir,
                    p_layers=p,
                )
                _agg_report.scan()
                training_sizes = _agg_report.available_n_values()
            except Exception:
                training_sizes = [n_target]

            report = quick_cross_n_report(
                per_h_for_report,
                n_target,
                topology=topo,
                training_sizes=training_sizes,
            )
            cross_n_report = report.to_dict()
            status = "✅" if report.overall_pass else "❌"
            logger.info(
                f"  {status} CrossN L1: pass_rate={report.l1_pass_rate:.0%}, mean_ΔE/gap={report.l1_mean_de_gap:.4f}"
            )
        except Exception as e:
            logger.debug(f"  Cross-N report skipped: {e}")

        # ── Memory cleanup: free heavy objects before _build_envelope ─────
        # NOTE: Do NOT call gc.collect() here! Qiskit's CircuitData destructor
        # triggers mimalloc's _mi_arenas_page_unabandon which calls sleep() in
        # a retry loop on macOS ARM64, hanging the process indefinitely.
        # Instead, just delete references and let os._exit() handle cleanup.
        del model, predictions
        eval_cache = None
        dataset = None
        agg = None
        circuit_target = None
        lattice_target = None

        return {
            "pass": final_pass_rate >= 0.50 or convergence_reason == "no_improvement",
            "convergence_reason": convergence_reason,
            "iterations_run": len(iteration_reports),
            "final_pass_rate": final_pass_rate,
            "total_vqe_calls": total_vqe_calls,
            "iteration_reports": iteration_reports,
            "cross_n_validation": cross_n_report,
            "cache_stats": final_stats,
            "gt_cache_hits": gt_hits,
            "gt_cache_misses": gt_misses,
        }

    def _build_iter_report(self, iteration, pass_rate, n_refined, n_ansatz_limited, eval_hits, elapsed_s) -> dict:
        """Build per-iteration summary dict."""
        return {
            "iteration": iteration,
            "pass_rate": pass_rate,
            "n_refined": n_refined,
            "n_ansatz_limited": n_ansatz_limited,
            "eval_cache_hits": eval_hits,
            "elapsed_s": elapsed_s,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 3 (alt): Warm-Start A/B comparison
    # ═══════════════════════════════════════════════════════════════════════════

    def _descend_to_threshold(self, cost, fid, grad, x0, *, threshold: float, maxiter: int):
        """L-BFGS-B from ``x0``, recording iters/evals to first reach ``fid>=threshold``.

        Runs ONE L-BFGS-B optimization in the ``[-pi, pi]`` box with the shared
        adjoint gradient, wrapping ``cost`` to count evaluations and attaching a
        per-iteration callback that stamps the first iterate whose exact
        fidelity crosses ``threshold``. This is the honest convergence metric: a
        warm-start that starts closer is only "better" if it reaches the target
        state in fewer iterations/evals, not merely at a higher init-fid.

        Returns a dict with ``init_fid``, ``final_fid``, ``final_energy``,
        ``iters_to_threshold`` (None if never crossed), ``n_iters``, ``n_evals``,
        and ``reached`` (bool).
        """
        import numpy as np
        from scipy.optimize import minimize

        x0 = np.asarray(x0, dtype=np.float64)
        init_fid = float(fid(x0))

        n_evals = {"count": 0}

        def _counted_cost(x):
            n_evals["count"] += 1
            return cost(x)

        crossing = {"iter": None}
        it = {"count": 0}

        def _callback(xk):
            it["count"] += 1
            if crossing["iter"] is None and fid(xk) >= threshold:
                crossing["iter"] = it["count"]

        # Init already above threshold → zero refinement iterations needed.
        if init_fid >= threshold:
            return {
                "init_fid": init_fid,
                "final_fid": init_fid,
                "final_energy": float(cost(x0)),
                "iters_to_threshold": 0,
                "n_iters": 0,
                "n_evals": 0,
                "reached": True,
            }

        r = minimize(
            _counted_cost,
            x0,
            method="L-BFGS-B",
            jac=grad,
            bounds=[(-np.pi, np.pi)] * len(x0),
            options={"maxiter": maxiter, "ftol": 1e-12},
            callback=_callback,
        )
        final_fid = float(fid(r.x))
        # The callback fires once per accepted iteration; if the final point
        # crosses but no intermediate iterate did (rare, e.g. a single big
        # step), attribute the crossing to the last recorded iteration.
        if crossing["iter"] is None and final_fid >= threshold:
            crossing["iter"] = int(r.nit)
        return {
            "init_fid": init_fid,
            "final_fid": final_fid,
            "final_energy": float(r.fun),
            "iters_to_threshold": crossing["iter"],
            "n_iters": int(r.nit),
            "n_evals": n_evals["count"],
            "reached": crossing["iter"] is not None,
        }

    def section_compare_warmstart(self) -> dict:
        """Head-to-head warm-start A/B at each target-N h (held-out N).

        Three arms, each refined by the SAME L-BFGS-B optimizer (shared
        cost/grad built once per h via :func:`make_cost_fid`):

        - **mpnn**: θ predicted by the multi-N UnifiedMPNN (trained in
          section_multi_n_train or loaded from the zoo).
        - **analytic**: the project warm-start cascade via
          :meth:`_apply_warmstart_seed` (analytic Ising seeds + regime + NPZ
          cross-N / cross-h donors, ranked by micro-descent).
        - **cold**: a seeded random θ in ``[-pi, pi]`` (honest baseline).

        For each arm we record init-fidelity, final fidelity, the iteration at
        which exact fidelity first reaches ``--compare-fid-threshold``, cost
        evaluations and wall time. A per-arm winner tally (fewest iters-to-
        threshold, ties broken by higher final fidelity) is reported — honestly,
        whether or not the MPNN wins.

        Statevector-regime only (N ≤ STATEVECTOR_MAX_N), since the exact ground
        eigenvector is needed for the fidelity metric.
        """
        import time

        import numpy as np
        import torch
        from scipy.sparse.linalg import eigsh

        from qmbp_simulation.framework.study_core import make_cost_fid
        from qmbp_simulation.models.constants import STATEVECTOR_MAX_N
        from qmbp_simulation.models.hamiltonian import HamiltonianBuilder
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.predictors.unified_graph import UNIFIED_NODE_FEATURES

        topo = self._args.topology
        threshold = float(getattr(self._args, "compare_fid_threshold", 0.90))
        maxiter = int(getattr(self._args, "compare_maxiter", 200))
        spec = get_model_spec(self._physics_model).with_params(**self._model_kwargs)

        all_results: dict = {}
        for p in self._args.p_layers:
            # Reuse the exact model-resolution path from section_cross_n_predict:
            # memory (just trained) → zoo best → optional train. Never silently
            # fabricate a model.
            model = self._models.get(p)
            if model is None:
                _train_if_missing = not (self._args.from_zoo or bool(getattr(self._args, "checkpoint", None)))
                model = self.load_best_mpnn_for_cross_n(
                    n_target=self._args.target_n[0],
                    model=self._physics_model,
                    topology=topo,
                    p_layers=p,
                    checkpoint_path=self._args.checkpoint,
                    train_if_missing=_train_if_missing,
                    train_epochs=FULL_TRAIN_EPOCHS,
                )
            if model is None:
                all_results[f"p{p}"] = {"pass": False, "error": "No model available for comparison"}
                continue
            model.eval()

            # Match the prediction graph feature dim to the loaded model.
            _model_feat = getattr(model, "node_features", UNIFIED_NODE_FEATURES)
            self._orbit_feature_effective = _model_feat > UNIFIED_NODE_FEATURES

            for n_target in self._args.target_n:
                if n_target > STATEVECTOR_MAX_N:
                    logger.warning(
                        f"  Skipping N={n_target}: compare-warmstart needs the exact "
                        f"ground state (N ≤ {STATEVECTOR_MAX_N})."
                    )
                    all_results[f"p{p}_N{n_target}"] = {
                        "pass": False,
                        "error": f"N={n_target} > STATEVECTOR_MAX_N",
                    }
                    continue

                logger.info(
                    f"  Warm-start A/B: N={n_target}, p={p}, topology={topo}, "
                    f"threshold F≥{threshold:.2f}, maxiter={maxiter}"
                )
                lattice = self.make_lattice(topo, n_target, J=1.0, h=2.0)
                circuit, _ = self._build_circuit(n_target, p, lattice)
                n_params = circuit.num_parameters
                rng_cold = np.random.default_rng(123)

                per_h: list[dict] = []
                for h in self._h_values:
                    lat_h = self.make_lattice(topo, n_target, J=1.0, h=float(h))
                    H = spec.build_hamiltonian(lat_h, **spec.hamiltonian_kwargs)

                    # Exact ground eigenvector + gap (statevector regime).
                    ev, evec = eigsh(H.to_matrix(sparse=True), k=2, which="SA")
                    order = np.argsort(ev)
                    psi = evec[:, order[0]].astype(complex)
                    e0 = float(ev[order[0]])
                    gap = float(ev[order[1]] - ev[order[0]])

                    # Shared cost/fid/grad for ALL arms at this h (one build).
                    cost, fid, grad, _backend = make_cost_fid(circuit, H, psi)

                    def _eval_theta(th, _c=cost):
                        return float(_c(th))

                    arms: dict = {}

                    # ── Arm 1: MPNN prediction ─────────────────────────────
                    try:
                        g = self._build_graph(lat_h, float(h), p, include_circuit_nodes=True)
                        with torch.no_grad():
                            theta_mpnn = model(g).numpy().flatten()
                        theta_mpnn = np.clip(theta_mpnn, -np.pi, np.pi)
                        if len(theta_mpnn) != n_params:
                            theta_mpnn = (
                                np.pad(theta_mpnn, (0, n_params - len(theta_mpnn)))
                                if len(theta_mpnn) < n_params
                                else theta_mpnn[:n_params]
                            )
                        t0 = time.perf_counter()
                        arms["mpnn"] = self._descend_to_threshold(
                            cost, fid, grad, theta_mpnn, threshold=threshold, maxiter=maxiter
                        )
                        arms["mpnn"]["seconds"] = time.perf_counter() - t0
                        arms["mpnn"]["provenance"] = "mpnn"
                    except Exception as e:  # noqa: BLE001
                        logger.debug(f"    h={h:.2f}: MPNN arm failed: {e}")
                        arms["mpnn"] = {"error": str(e)}

                    # ── Arm 2: analytic + donor cascade ────────────────────
                    try:
                        theta_zero = np.zeros(n_params)
                        seed_theta, _e_seed, prov = self._apply_warmstart_seed(
                            circuit=circuit,
                            H=H,
                            lattice=lat_h,
                            h=float(h),
                            p_layers=p,
                            gap=gap,
                            theta_init=theta_zero,
                            e_init=float(cost(theta_zero)),
                            eval_fn=_eval_theta,
                        )
                        t0 = time.perf_counter()
                        arms["analytic"] = self._descend_to_threshold(
                            cost, fid, grad, seed_theta, threshold=threshold, maxiter=maxiter
                        )
                        arms["analytic"]["seconds"] = time.perf_counter() - t0
                        arms["analytic"]["provenance"] = prov or "regime"
                    except Exception as e:  # noqa: BLE001
                        logger.debug(f"    h={h:.2f}: analytic arm failed: {e}")
                        arms["analytic"] = {"error": str(e)}

                    # ── Arm 3: cold random seed ────────────────────────────
                    try:
                        theta_cold = rng_cold.uniform(-np.pi, np.pi, n_params)
                        t0 = time.perf_counter()
                        arms["cold"] = self._descend_to_threshold(
                            cost, fid, grad, theta_cold, threshold=threshold, maxiter=maxiter
                        )
                        arms["cold"]["seconds"] = time.perf_counter() - t0
                        arms["cold"]["provenance"] = "cold_random"
                    except Exception as e:  # noqa: BLE001
                        logger.debug(f"    h={h:.2f}: cold arm failed: {e}")
                        arms["cold"] = {"error": str(e)}

                    winner = self._compare_arms_winner(arms, threshold)
                    per_h.append(
                        {
                            "h": float(h),
                            "e0": e0,
                            "gap": gap,
                            "arms": arms,
                            "winner": winner,
                        }
                    )
                    _fmt = self._fmt_arm_row(arms)
                    logger.info(f"    h={float(h):.2f}: {_fmt} → winner={winner}")

                summary = self._summarize_compare(per_h, threshold)
                self._log_compare_table(per_h, n_target, p, threshold, summary)
                all_results[f"p{p}_N{n_target}"] = {
                    "pass": summary["mpnn_reached"] > 0,
                    "threshold": threshold,
                    "maxiter": maxiter,
                    "per_h": per_h,
                    "summary": summary,
                }

        passed = any(r.get("pass") for r in all_results.values())
        return {"pass": passed, "per_config": all_results}

    @staticmethod
    def _compare_arms_winner(arms: dict, threshold: float) -> str:
        """Pick the arm with fewest iters-to-threshold; tie → higher final fid.

        Arms that never reached the threshold rank below any that did. Among
        non-reaching arms, higher final fidelity wins. Errored arms are excluded.
        """
        valid = {k: v for k, v in arms.items() if isinstance(v, dict) and "error" not in v}
        if not valid:
            return "none"

        def _key(item):
            _name, a = item
            reached = a.get("iters_to_threshold") is not None
            iters = a.get("iters_to_threshold")
            iters = iters if iters is not None else float("inf")
            # Sort: reached-first (0 before 1), then fewer iters, then higher fid.
            return (0 if reached else 1, iters, -float(a.get("final_fid", 0.0)))

        return min(valid.items(), key=_key)[0]

    @staticmethod
    def _fmt_arm_row(arms: dict) -> str:
        """Compact one-line per-h arm summary for the live log."""
        parts = []
        for name in ("mpnn", "analytic", "cold"):
            a = arms.get(name)
            if not isinstance(a, dict) or "error" in a:
                parts.append(f"{name}=ERR")
                continue
            itx = a.get("iters_to_threshold")
            itx_s = str(itx) if itx is not None else "∞"
            parts.append(f"{name}[F0={a.get('init_fid', 0):.3f}→{a.get('final_fid', 0):.3f} it={itx_s}]")
        return " ".join(parts)

    @staticmethod
    def _summarize_compare(per_h: list[dict], threshold: float) -> dict:
        """Aggregate per-arm wins, reach counts, and mean iters/init-fid."""
        import numpy as np

        arms = ("mpnn", "analytic", "cold")
        wins = {a: 0 for a in arms}
        reached = {a: 0 for a in arms}
        iters_when_reached = {a: [] for a in arms}
        init_fids = {a: [] for a in arms}
        for row in per_h:
            w = row.get("winner")
            if w in wins:
                wins[w] += 1
            for a in arms:
                arm = row["arms"].get(a)
                if not isinstance(arm, dict) or "error" in arm:
                    continue
                init_fids[a].append(float(arm.get("init_fid", 0.0)))
                if arm.get("iters_to_threshold") is not None:
                    reached[a] += 1
                    iters_when_reached[a].append(int(arm["iters_to_threshold"]))
        return {
            "n_points": len(per_h),
            "wins": wins,
            "mpnn_reached": reached["mpnn"],
            "reached": reached,
            "mean_iters_when_reached": {
                a: (float(np.mean(iters_when_reached[a])) if iters_when_reached[a] else None) for a in arms
            },
            "mean_init_fid": {a: (float(np.mean(init_fids[a])) if init_fids[a] else None) for a in arms},
        }

    @staticmethod
    def _log_compare_table(per_h: list[dict], n_target: int, p: int, threshold: float, summary: dict) -> None:
        """Emit the per-h comparison table + aggregate verdict to the log."""
        lines = [
            "",
            f"  ┌─ Warm-start A/B — N={n_target} p={p} (F≥{threshold:.2f}) " + "─" * 20,
            "  │  h     | MPNN F0→Ff it  | ANALYTIC F0→Ff it  | COLD F0→Ff it   | winner",
            "  │  " + "-" * 78,
        ]
        for row in per_h:
            h = row["h"]
            cells = []
            for name in ("mpnn", "analytic", "cold"):
                a = row["arms"].get(name)
                if not isinstance(a, dict) or "error" in a:
                    cells.append("   ERR         ")
                    continue
                itx = a.get("iters_to_threshold")
                itx_s = f"{itx:>3}" if itx is not None else "  ∞"
                cells.append(f"{a.get('init_fid', 0):.3f}→{a.get('final_fid', 0):.3f} {itx_s}")
            lines.append(f"  │  {h:<5.2f} | {cells[0]} | {cells[1]}  | {cells[2]}  | {row['winner']}")
        wins = summary["wins"]
        reached = summary["reached"]
        mean_it = summary["mean_iters_when_reached"]
        lines.append("  │  " + "-" * 78)
        lines.append(
            f"  │  WINS: mpnn={wins['mpnn']} analytic={wins['analytic']} cold={wins['cold']}  "
            f"(of {summary['n_points']} points)"
        )
        lines.append(
            f"  │  REACHED F≥{threshold:.2f}: "
            f"mpnn={reached['mpnn']} analytic={reached['analytic']} cold={reached['cold']}"
        )

        def _mi(a):
            return f"{mean_it[a]:.1f}" if mean_it[a] is not None else "n/a"

        lines.append(f"  │  MEAN ITERS→thr: mpnn={_mi('mpnn')} analytic={_mi('analytic')} cold={_mi('cold')}")
        lines.append("  └" + "─" * 70)
        logger.info("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    AcceleratedCrossNRunner.main()
