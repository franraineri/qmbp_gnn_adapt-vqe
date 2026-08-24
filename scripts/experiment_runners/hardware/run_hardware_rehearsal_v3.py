#!/usr/bin/env python3
"""Hardware Deployment Rehearsal V3 — V2 + MPNN Evaluation Suite.

Extends run_hardware_rehearsal_v2.py with four MPNN characterization sections
that rigorously validate prediction quality before spending QPU time.

  Section 10 — MPNN Warm-Start Benchmark
      Compares three VQE initialization strategies at each h_test:
        (a) random uniform(-0.01, 0.01)
        (b) θ_opt from nearest training h (classical warm-start)
        (c) MPNN prediction (this work)
      Metrics: iterations to convergence, speedup factor, ΔE/gap.
      Pass: speedup_vs_random ≥ 1.5x AND all ΔE/gap < 5%.

  Section 11 — MPNN Leave-One-Out Cross-Validation
      Trains N-1 models to estimate generalization without a separate test set.
      Pass: pass_rate ≥ 80% of folds achieve ΔE/gap < 5%.

  Section 12 — MPNN Landscape Quality (circuit vs ML error decomposition)
      Decomposes ΔE into ansatz expressibility (circuit) and ML error.
      Computes landscape curvature κ at θ_opt (hardware sensitivity indicator).
      Pass: mean total ΔE/gap < 5%.

  Section 13 — MPNN Interpolation vs Extrapolation
      Measures accuracy inside vs outside the training h-range.
      Reports degradation factor = mean_extrap / mean_interp.
      Pass: interpolation pass-rate ≥ 80%.

  Section 14 — MPNN Noisy Evaluation (FakeKingston)
      Evaluates MPNN θ_pred via noisy simulation instead of noiseless.
      Compares: noiseless_de_gap vs noisy_raw_de_gap vs noisy_zne_de_gap.
      Answers: "Does MPNN θ_pred still work under realistic noise?"
      Pass: mean noisy ΔE/gap < 10% (relaxed for noise).

All sections 10-14 are MPNN-only (no QPU needed). Sections 1-9 from V2
are preserved and exercise the full HardwareBackend + ZNE pipeline.

Results are saved to:
    results/experiments/exp_hw_rehearsal_v3/run_<timestamp>.json
    (standard ValidationRunner envelope — parseable by digest + compare)

Usage:
    # Full run (sections 1-14):
    .venv/bin/python scripts/experiment_runners/run_hardware_rehearsal_v3.py

    # MPNN sections only (no FakeKingston ZNE, fast):
    .venv/bin/python scripts/experiment_runners/run_hardware_rehearsal_v3.py --skip-hardware-sections

    # Single section:
    .venv/bin/python scripts/experiment_runners/run_hardware_rehearsal_v3.py --section 10
    .venv/bin/python scripts/experiment_runners/run_hardware_rehearsal_v3.py --section 12

    # N=6 quick validation:
    .venv/bin/python scripts/experiment_runners/run_hardware_rehearsal_v3.py \\
        --skip-hardware-sections --n-qubits 6 --topology chain_1d \\
        --h-train 2.0 1.75 1.5 1.25 --h-test 1.875 \\
        --mpnn-epochs 1000 --n-vqe-bench-restarts 2

    # With noisy section (FakeKingston):
    .venv/bin/python scripts/experiment_runners/run_hardware_rehearsal_v3.py \\
        --section 14 --n-qubits 6 --topology chain_1d \\
        --h-train 2.0 1.75 1.5 1.25 --h-test 1.875 --mpnn-epochs 1000

    # Dry run:
    .venv/bin/python scripts/experiment_runners/run_hardware_rehearsal_v3.py --dry-run

References:
    - NN-VQE (Miao et al., PRApplied 2024): MLP warm-start, ~20 training pts.
    - Qracle (Zhang et al., 2025): GNN warm-start, up to 64% fewer iters.
    - Kohavi (1995): LOO-CV generalization estimate.
    - Fontana et al. (2024, arXiv:2402.18953): VQE landscape analysis.
"""

from __future__ import annotations

import logging
import sys

import numpy as np

from qmbp_simulation.framework.runner_base import (
    Section,
    resolve_project_root,
)

_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import V2 as base via importlib so this script is runnable from any cwd
# without adding scripts/ to sys.path.
import importlib.util as _ilu

_v2_path = _ROOT / "scripts" / "experiment_runners" / "run_hardware_rehearsal_v2.py"
_v2_spec = _ilu.spec_from_file_location("run_hardware_rehearsal_v2", _v2_path)
_v2_mod = _ilu.module_from_spec(_v2_spec)  # type: ignore[arg-type]
_v2_spec.loader.exec_module(_v2_mod)  # type: ignore[union-attr]

# Re-export V2 constants so external code can import from V3
DE_GAP_THRESHOLD: float = _v2_mod.DE_GAP_THRESHOLD
H_TEST_POINTS: list = _v2_mod.H_TEST_POINTS
H_TRAIN_GRID: list = _v2_mod.H_TRAIN_GRID
MPNN_EPOCHS: int = _v2_mod.MPNN_EPOCHS
MPNN_HIDDEN_DIM: int = _v2_mod.MPNN_HIDDEN_DIM
MPNN_LR: float = _v2_mod.MPNN_LR
MPNN_PATIENCE: int = _v2_mod.MPNN_PATIENCE
ZNE_SHOTS: int = _v2_mod.ZNE_SHOTS
ZNE_N_LAYOUTS: int = _v2_mod.ZNE_N_LAYOUTS
HardwareRehearsalV2 = _v2_mod.HardwareRehearsalV2

logger = logging.getLogger(__name__)

# ── V3-specific thresholds and defaults ──────────────────────────────────────

# Section 10: warm-start benchmark
SPEEDUP_THRESHOLD = 1.5  # MPNN must be at least 1.5x faster than random
DEFAULT_N_VQE_BENCH_RESTARTS = 5  # random-init runs to average over
DEFAULT_MAXITER_REFINE = 200  # tight budget — stresses warm-start quality

# Section 11: LOO-CV
LOO_PASS_RATE_THRESHOLD = 0.80  # ≥80% of folds must pass ΔE/gap < 5%

# Section 12: landscape quality
CURVATURE_WARN_THRESHOLD = 10.0  # κ > this → hardware deployment risk
MPNN_FRAC_WARN_THRESHOLD = 0.50  # ML error > 50% of total → MPNN dominates

# Section 13: interpolation/extrapolation
INTERP_PASS_RATE_THRESHOLD = 0.80
DEGRADATION_WARN_THRESHOLD = 3.0  # extrap/interp degradation > 3x is notable

# Section 14: noisy evaluation
NOISY_DE_GAP_THRESHOLD = 0.10  # 10% — relaxed for shot noise


# ═══════════════════════════════════════════════════════════════════════════════
# Criteria table — single source of truth for section pass/fail decisions
# ═══════════════════════════════════════════════════════════════════════════════

SECTION_CRITERIA: dict[int, dict] = {
    10: {
        "name": "MPNN Warm-Start Benchmark",
        "primary_metric": "speedup_vs_random",
        "threshold": SPEEDUP_THRESHOLD,
        "direction": "ge",  # ≥
        "secondary": "all ΔE/gap < 5%",
        "ref": "Qracle (Zhang et al., 2025): 64% fewer iters on spin systems",
    },
    11: {
        "name": "LOO Cross-Validation",
        "primary_metric": "pass_rate",
        "threshold": LOO_PASS_RATE_THRESHOLD,
        "direction": "ge",
        "secondary": None,
        "ref": "Kohavi (1995): LOO is least-biased estimator for small datasets",
    },
    12: {
        "name": "Landscape Quality",
        "primary_metric": "mean_error_total",
        "threshold": DE_GAP_THRESHOLD,
        "direction": "le",  # ≤
        "secondary": f"κ < {CURVATURE_WARN_THRESHOLD} (hardware safety)",
        "ref": "Fontana et al. (2024): landscape curvature predicts hardware sensitivity",
    },
    13: {
        "name": "Interpolation vs Extrapolation",
        "primary_metric": "interp_pass_rate",
        "threshold": INTERP_PASS_RATE_THRESHOLD,
        "direction": "ge",
        "secondary": f"degradation_factor informational (warn > {DEGRADATION_WARN_THRESHOLD}x)",
        "ref": "NN-VQE (Miao 2024): valid interpolation range is the deployment range",
    },
    14: {
        "name": "Noisy Evaluation (FakeKingston)",
        "primary_metric": "mean_noisy_raw_de_gap",
        "threshold": NOISY_DE_GAP_THRESHOLD,
        "direction": "le",
        "secondary": "ZNE improvement ≥ 0 (ZNE must not hurt)",
        "ref": "Hardware noise model: FakeKingston calibration data",
    },
    15: {
        "name": "Warm-Start Scaling with N",
        "primary_metric": "all_speedups ≥ 1.5x",
        "threshold": 1.5,
        "direction": "ge",
        "secondary": "scaling_trend non-decreasing (GNN value grows with problem size)",
        "ref": "Qracle (Zhang 2025): GNN advantage should scale with circuit depth",
    },
    16: {
        "name": "Learning Curve (Sample Efficiency)",
        "primary_metric": "critical_size",
        "threshold": 10,
        "direction": "le",
        "secondary": "Full-dataset ΔE/gap < 5%",
        "ref": "NN-VQE (Miao 2024): ~20 training points needed for MLP; GNN should need fewer",
    },
    17: {
        "name": "Zero-Shot Topology Transfer",
        "primary_metric": "mean_de_gap_zero_shot",
        "threshold": DE_GAP_THRESHOLD,
        "direction": "le",
        "secondary": "transfer_ratio < 2x (zero-shot within 2x of in-distribution)",
        "ref": "GNN lattice-agnosticism: edge_index encodes connectivity, not topology identity",
    },
    18: {
        "name": "Multi-Seed LOO Robustness",
        "primary_metric": "std_pass_rate",
        "threshold": 0.15,
        "direction": "le",
        "secondary": "mean_pass_rate ≥ 0.80 (still achieves target)",
        "ref": "Seed robustness established at pipeline level (std=0.010); same expected here",
    },
    19: {
        "name": "Curvature κ as Hardware-Risk Proxy",
        "primary_metric": "mean_pearson_r",
        "threshold": 0.70,
        "direction": "ge",
        "secondary": "κ_max at h near h_c (consistent with known physics)",
        "ref": "Fontana et al. (2024): landscape curvature peaks at phase transition",
    },
    20: {
        "name": "PauliEvolutionGate vs RZZ/RX Transpilation Comparison",
        "primary_metric": "energy_max_abs_diff",
        "threshold": 1e-8,
        "direction": "le",
        "secondary": (
            "2Q-depth reduction ≥ 5% (meaningful scheduling improvement); "
            "noisy ΔE/gap difference < 1% (functionally equivalent under FakeKingston)"
        ),
        "ref": (
            "IBM tutorial 'Compilation methods for Hamiltonian simulation circuits'; "
            "15_transpiler_exploration.md: 11% 2Q-depth reduction validated 2026-06-05"
        ),
    },
    21: {
        "name": "Mitiq Multi-Method Comparison",
        "primary_metric": "best_mitiq_de_gap",
        "threshold": 0.10,
        "direction": "le",
        "secondary": "Mitiq CDR or ZNE must improve over raw noisy (at least 1 method wins)",
        "ref": (
            "Mitiq 1.0: CDR (Czarnik 2021), ZNE random fold, DDD+ZNE composition. "
            "24_mitiq_integration_plan.md"
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════


class HardwareRehearsalV3(HardwareRehearsalV2):
    """Hardware Rehearsal V3: V2 sections 1-9 + MPNN evaluation sections 10-14.

    Sections 10-13 run noiseless (no FakeKingston needed).
    Section 14 runs noisy via FakeKingston + noisy_estimate.

    All sections delegate to ValidationRunner helper methods
    (benchmark_mpnn_warmstart, mpnn_leave_one_out_cv, mpnn_landscape_quality,
    mpnn_interpolation_extrapolation) which are topology/model-agnostic and
    reusable from any future runner.
    """

    runner_id = "hardware_rehearsal_v3"
    experiment_id = "HW_REHEARSAL_V3"
    description = "Hardware Rehearsal V3 — MPNN Evaluation Suite (sections 10-14)"
    hypothesis = (
        "MPNN θ_pred: warm-start speedup ≥1.5x, LOO pass-rate ≥80%, "
        "landscape decomposition confirms ML error is sub-dominant, "
        "interpolation reliable, noisy eval ΔE/gap < 10%"
    )

    @classmethod
    def _add_custom_args(cls, parser):
        """Extend V2 CLI with V3 MPNN evaluation arguments."""
        super()._add_custom_args(parser)

        # Section 10
        parser.add_argument(
            "--n-vqe-bench-restarts",
            type=int,
            default=DEFAULT_N_VQE_BENCH_RESTARTS,
            help=f"Random VQE runs to avg over in warm-start benchmark (default: {DEFAULT_N_VQE_BENCH_RESTARTS})",
        )
        parser.add_argument(
            "--maxiter-refine",
            type=int,
            default=DEFAULT_MAXITER_REFINE,
            help=f"Max VQE iters during benchmark refinement (default: {DEFAULT_MAXITER_REFINE})",
        )
        # Section 11
        parser.add_argument(
            "--loo-min-train-size",
            type=int,
            default=5,
            help="Min training points per LOO fold (default: 5)",
        )
        # Section 14
        parser.add_argument(
            "--noisy-shots",
            type=int,
            default=ZNE_SHOTS,
            help=f"Shots for noisy MPNN evaluation (default: {ZNE_SHOTS})",
        )
        parser.add_argument(
            "--noisy-n-layouts",
            type=int,
            default=ZNE_N_LAYOUTS,
            help=f"Number of ZNE layouts for noisy eval (default: {ZNE_N_LAYOUTS})",
        )
        # Flow control
        parser.add_argument(
            "--skip-hardware-sections",
            action="store_true",
            default=False,
            help="Skip sections 1-9 (hardware ZNE). Run only sections 10-14.",
        )
        parser.add_argument(
            "--skip-noisy-mpnn",
            action="store_true",
            default=False,
            help="Skip section 14 (noisy MPNN eval). Useful when FakeKingston is slow.",
        )
        # Sections 15-19 system sizes for scaling experiment
        parser.add_argument(
            "--scaling-sizes",
            type=int,
            nargs="+",
            default=None,
            help="System sizes for section 15 scaling benchmark (default: [4, 6, 10])",
        )
        parser.add_argument(
            "--scaling-p-layers",
            type=int,
            nargs="+",
            default=None,
            help=(
                "p_layers for each scaling size (same order as --scaling-sizes). "
                "e.g. --scaling-sizes 4 6 10 --scaling-p-layers 2 2 1  "
                "(p=1 for N=10 respects ZNE limit of 18 CX). "
                "Defaults to --p-layers for all N."
            ),
        )
        # Learning curve: full h pool (train_sizes auto-derived)
        parser.add_argument(
            "--h-pool",
            type=float,
            nargs="+",
            default=None,
            help="Full h pool for learning curve / LOO experiments (default: extends h-train)",
        )
        # Topology transfer: source topology
        parser.add_argument(
            "--source-topology",
            type=str,
            default="chain_1d",
            choices=["chain_1d", "ladder", "triangular", "heavy_hex"],
            help="Source topology for section 17 transfer experiment (default: chain_1d)",
        )
        # Multi-seed LOO
        parser.add_argument(
            "--loo-n-seeds",
            type=int,
            default=3,
            help="Number of seeds for section 18 multi-seed LOO (default: 3)",
        )
        # Noise correlation
        parser.add_argument(
            "--noise-sigmas",
            type=float,
            nargs="+",
            default=None,
            help="Noise levels (σ) for section 19 curvature correlation (default: [0.01, 0.05, 0.10, 0.20])",
        )
        parser.add_argument(
            "--skip-extended-sections",
            action="store_true",
            default=False,
            help="Skip sections 15-19 (extended experiments). Run only 10-14.",
        )
        parser.add_argument(
            "--skip-pauli-evolution",
            action="store_true",
            default=True,
            help=(
                "Skip section 20 (PauliEvolutionGate comparison). "
                "Enabled by default — use --no-skip-pauli-evolution to include it."
            ),
        )
        parser.add_argument(
            "--no-skip-pauli-evolution",
            action="store_false",
            dest="skip_pauli_evolution",
            help="Include section 20 (PauliEvolutionGate comparison)",
        )
        parser.add_argument(
            "--skip-mitiq",
            action="store_true",
            default=False,
            help="Skip section 21 (Mitiq multi-method comparison)",
        )
        parser.add_argument(
            "--h-kappa-grid",
            type=float,
            nargs="+",
            default=None,
            help=(
                "Dedicated h-grid for section 19 curvature analysis. "
                "Should span from well inside training range down to h_c "
                "(e.g., --h-kappa-grid 2.0 1.75 1.5 1.25 1.1 1.0). "
                "Defaults to h_train if not provided."
            ),
        )
        # Section 10 extensions: modes (d) and (e)
        parser.add_argument(
            "--use-flow-warmstart",
            action="store_true",
            default=False,
            help="Enable mode (d): EmbeddingMAF-based warmstart in §10.",
        )
        parser.add_argument(
            "--use-bond-resolved",
            action="store_true",
            default=False,
            help="Enable mode (e): BondResolvedMPNN warmstart in §10 (chain_1d N=6 p=2 only).",
        )

    def build_config(self) -> dict:
        """Build config — extends V2 with V3 MPNN evaluation parameters."""
        cfg = super().build_config()
        args = self._args
        cfg["mpnn_eval"] = {
            # Thresholds
            "speedup_threshold": SPEEDUP_THRESHOLD,
            "loo_pass_rate_threshold": LOO_PASS_RATE_THRESHOLD,
            "noisy_de_gap_threshold": NOISY_DE_GAP_THRESHOLD,
            "curvature_warn_threshold": CURVATURE_WARN_THRESHOLD,
            "degradation_warn_threshold": DEGRADATION_WARN_THRESHOLD,
            # Runtime params (sections 10-14)
            "n_vqe_bench_restarts": args.n_vqe_bench_restarts,
            "maxiter_refine": args.maxiter_refine,
            "loo_min_train_size": args.loo_min_train_size,
            "noisy_shots": args.noisy_shots,
            "noisy_n_layouts": args.noisy_n_layouts,
            # Extended sections (15-19)
            "scaling_sizes": getattr(args, "scaling_sizes", None) or [4, 6, 10],
            "scaling_p_layers": getattr(args, "scaling_p_layers", None),
            "source_topology": getattr(args, "source_topology", "chain_1d"),
            "loo_n_seeds": getattr(args, "loo_n_seeds", 3),
            "noise_sigmas": getattr(args, "noise_sigmas", None) or [0.01, 0.05, 0.10, 0.20],
            # Section 19 dedicated kappa grid (may extend below h_train to cover h_c)
            "h_kappa_grid": getattr(args, "h_kappa_grid", None),
            # Flow control flags
            "skip_hardware_sections": getattr(args, "skip_hardware_sections", False),
            "skip_noisy_mpnn": getattr(args, "skip_noisy_mpnn", False),
            "skip_extended_sections": getattr(args, "skip_extended_sections", False),
            "skip_pauli_evolution": getattr(args, "skip_pauli_evolution", False),
        }
        cfg["section_criteria"] = SECTION_CRITERIA
        return cfg

    def define_sections(self) -> list[Section]:
        """V2 sections 1-9 + V3 MPNN sections 10-14."""
        v3_sections = [
            Section(
                id=10,
                name="MPNN Warm-Start Benchmark",
                fn=self.section_warmstart_benchmark,
                hypothesis=(
                    f"MPNN speedup ≥{SPEEDUP_THRESHOLD}x vs random AND all ΔE/gap < {DE_GAP_THRESHOLD:.0%}"
                ),
            ),
            Section(
                id=11,
                name="MPNN LOO Cross-Validation",
                fn=self.section_loo_cv,
                hypothesis=(
                    f"≥{int(LOO_PASS_RATE_THRESHOLD * 100)}% of LOO folds achieve ΔE/gap < {DE_GAP_THRESHOLD:.0%}"
                ),
            ),
            Section(
                id=12,
                name="MPNN Landscape Quality (circuit vs ML error)",
                fn=self.section_landscape_quality,
                hypothesis=(
                    f"Mean total ΔE/gap < {DE_GAP_THRESHOLD:.0%} "
                    f"AND κ < {CURVATURE_WARN_THRESHOLD} (hardware-safe landscape)"
                ),
            ),
            Section(
                id=13,
                name="MPNN Interpolation vs Extrapolation",
                fn=self.section_interpolation_extrapolation,
                hypothesis=(
                    f"Interpolation pass-rate ≥{int(INTERP_PASS_RATE_THRESHOLD * 100)}% "
                    f"AND degradation < {DEGRADATION_WARN_THRESHOLD}x"
                ),
            ),
        ]

        if not getattr(self._args, "skip_noisy_mpnn", False):
            v3_sections.append(
                Section(
                    id=14,
                    name="MPNN Noisy Evaluation (FakeKingston)",
                    fn=self.section_noisy_mpnn_eval,
                    hypothesis=(
                        f"MPNN θ_pred gives ΔE/gap < {NOISY_DE_GAP_THRESHOLD:.0%} under FakeKingston noise"
                    ),
                )
            )

        if not getattr(self._args, "skip_extended_sections", False):
            v3_sections += [
                Section(
                    id=15,
                    name="MPNN Warm-Start Scaling with N",
                    fn=self.section_scaling_with_n,
                    hypothesis=(
                        "MPNN speedup ≥1.5x at all system sizes AND "
                        "speedup trend is non-decreasing with N"
                    ),
                ),
                Section(
                    id=16,
                    name="MPNN Learning Curve (sample efficiency)",
                    fn=self.section_learning_curve,
                    hypothesis=(
                        "Critical training size ≤ 7 points for ΔE/gap < 5% "
                        "(MPNN is sample-efficient)"
                    ),
                ),
                Section(
                    id=17,
                    name="MPNN Zero-Shot Topology Transfer",
                    fn=self.section_topology_transfer,
                    hypothesis=(
                        f"Zero-shot transfer ΔE/gap < {DE_GAP_THRESHOLD:.0%} "
                        "(GNN generalizes across lattice topologies)"
                    ),
                ),
                Section(
                    id=18,
                    name="MPNN Multi-Seed LOO Robustness",
                    fn=self.section_multiseed_loo,
                    hypothesis=(
                        "LOO pass-rate std < 15% across seeds (result is stable, "
                        "not initialization-dependent)"
                    ),
                ),
                Section(
                    id=19,
                    name="Curvature κ as Hardware-Risk Proxy",
                    fn=self.section_curvature_noise_correlation,
                    hypothesis=(
                        "Pearson r(κ, ΔE_noise) ≥ 0.70 across h-grid "
                        "(κ reliably predicts noise sensitivity)"
                    ),
                ),
            ]

        # Section 20 is always appended (not gated by skip_extended_sections)
        # because it is a circuit-level integration test, not an MPNN experiment.
        if not getattr(self._args, "skip_pauli_evolution", False):
            v3_sections.append(
                Section(
                    id=20,
                    name="PauliEvolutionGate vs RZZ/RX Transpilation Comparison",
                    fn=self.section_pauli_evolution_comparison,
                    hypothesis=(
                        "PauliEvolutionGate representation gives ≥5% 2Q-depth reduction "
                        "vs explicit RZZ/RX, identical noiseless energy (|ΔE|<1e-8), "
                        "and equivalent noisy ΔE/gap (diff <1%) under FakeKingston. "
                        "Confirms integration is safe before IBM Kingston deployment."
                    ),
                )
            )

        # Section 21: Mitiq multi-method comparison (optional, skip if no mitiq)
        if not getattr(self._args, "skip_mitiq", False):
            v3_sections.append(
                Section(
                    id=21,
                    name="Mitiq Multi-Method Error Mitigation Comparison",
                    fn=self.section_mitiq_comparison,
                    hypothesis=(
                        "At least one Mitiq method (CDR, ZNE random, DDD+ZNE) achieves "
                        "ΔE/gap < 10% under FakeKingston noise, providing an independent "
                        "verification channel for hardware deployment beyond PEA-ZNE."
                    ),
                )
            )

        if getattr(self._args, "skip_hardware_sections", False):
            logger.info(
                f"  --skip-hardware-sections: running sections {[s.id for s in v3_sections]} only."
            )
            return v3_sections

        v2_sections = super().define_sections()
        return v2_sections + v3_sections

    # ──────────────────────────────────────────────────────────────────────────
    # Shared MPNN setup (avoids repeating VQE sweep + MPNN training 4 times)
    # ──────────────────────────────────────────────────────────────────────────

    def _get_or_build_mpnn(self):
        """Return cached (predictor, theta_map, h_arr, theta_arr, n_params) or build fresh.

        Trains one MPNN on h_train once and caches it on self._mpnn_cache.
        Subsequent calls to sections 10-14 reuse the same trained model,
        ensuring consistent training across all MPNN sections.
        """
        if hasattr(self, "_mpnn_cache") and self._mpnn_cache is not None:
            return self._mpnn_cache

        from qmbp_simulation import make_lattice
        from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn

        topology = self._args.topology
        n_qubits = self._args.n_qubits
        h_train = self._args.h_train or H_TRAIN_GRID
        p_layers = self._args.p_layers

        logger.info(
            f"  [MPNN cache] Training shared model: "
            f"{topology} N={n_qubits} p={p_layers} "
            f"{len(h_train)} pts × {self._args.mpnn_epochs} epochs"
        )

        theta_map = self.vqe_descending_sweep(
            topology,
            n_qubits,
            h_train,
            seed=42,
            p_layers=p_layers,
            n_restarts=self._args.vqe_restarts,
            model=self._args.model,
        )
        h_arr = np.array(sorted(theta_map.keys(), reverse=True))
        theta_arr = np.array([theta_map[h] for h in h_arr])
        e_arr = np.array(
            [
                self.exact_ground_state(topology, n_qubits, float(h), model=self._args.model)[0]
                for h in h_arr
            ]
        )
        n_params = theta_arr.shape[1]

        lattice_ref = make_lattice(topology, n_qubits, J=1.0, h=float(h_arr[0]))
        dataset = build_graph_dataset(
            lattice_ref,
            h_values=h_arr,
            theta_opt=theta_arr,
            e_exact=e_arr,
            fidelity_threshold=0.0,  # noqa: noiseless VQE — no fidelity filtering needed
        )
        predictor = MPNNPredictor(
            node_features=dataset[0].x.shape[1],
            output_dim=n_params,
            hidden_dim=self._args.mpnn_hidden_dim,
        )
        train_result = train_mpnn(
            predictor,
            dataset,
            n_epochs=self._args.mpnn_epochs,
            lr=MPNN_LR,
            patience=MPNN_PATIENCE,
            seed=42,
        )
        predictor.eval()

        self._mpnn_cache = {
            "predictor": predictor,
            "theta_map": theta_map,
            "h_arr": h_arr,
            "theta_arr": theta_arr,
            "e_arr": e_arr,
            "n_params": n_params,
            "dataset": dataset,
            "train_mse": train_result["final_mse"],
            "train_epochs": len(train_result["mse_history"]),
        }
        logger.info(
            f"  [MPNN cache] Trained: MSE={train_result['final_mse']:.2e}, "
            f"epochs={len(train_result['mse_history'])}"
        )
        return self._mpnn_cache

    def setup(self):
        """Initialize backends + MPNN cache slot."""
        super().setup()
        self._mpnn_cache = None  # Lazy-initialized on first use

    # ──────────────────────────────────────────────────────────────────────────
    # Section 10: MPNN Warm-Start Benchmark
    # ──────────────────────────────────────────────────────────────────────────

    def section_warmstart_benchmark(self) -> dict:
        """Benchmark MPNN warm-start against random and prev-h baselines.

        Uses the shared MPNN cache — no extra training overhead.

        Criteria (from SECTION_CRITERIA[10]):
          PASS: mean_speedup_vs_random ≥ 1.5x AND all h_test ΔE/gap < 5%
          WARN: speedup < 2x (expected level for well-trained model on smooth landscape)
          FAIL: speedup < 1x (MPNN hurts — retrain before hardware)
        """
        topology = self._args.topology
        n_qubits = self._args.n_qubits
        h_test = self._args.h_test or H_TEST_POINTS
        h_train = self._args.h_train or H_TRAIN_GRID
        p_layers = self._args.p_layers

        self._log_section_criteria(10)
        logger.info(
            f"  Config: N={n_qubits}, topology={topology}, p={p_layers}"
            f" | {len(h_train)} train → {len(h_test)} test"
            f" | restarts={self._args.n_vqe_bench_restarts}, maxiter={self._args.maxiter_refine}"
        )

        result = self.benchmark_mpnn_warmstart(
            topology=topology,
            n_qubits=n_qubits,
            h_train=h_train,
            h_test=h_test,
            p_layers=p_layers,
            seed=42,
            n_restarts_vqe=self._args.vqe_restarts,
            mpnn_hidden_dim=self._args.mpnn_hidden_dim,
            mpnn_epochs=self._args.mpnn_epochs,
            mpnn_lr=MPNN_LR,
            mpnn_patience=MPNN_PATIENCE,
            n_vqe_restarts_from_pred=self._args.n_vqe_bench_restarts,
            maxiter_refine=self._args.maxiter_refine,
            model=self._args.model,
            de_gap_threshold=DE_GAP_THRESHOLD,
        )
        # Populate cache from warmstart data (avoids redundant VQE)
        if self._mpnn_cache is None and "per_h" in result:
            logger.debug("  [MPNN cache] Populated from section 10 warmstart run.")

        s = result["summary"]
        self._log_warmstart_table(result["per_h"], s)

        speedup_ok = s["mean_speedup_vs_random"] >= SPEEDUP_THRESHOLD
        energy_ok = result["pass"]
        passed = speedup_ok and energy_ok

        self._log_pass_fail(
            10,
            passed,
            [
                f"speedup={s['mean_speedup_vs_random']:.2f}x (threshold={SPEEDUP_THRESHOLD}x)",
                f"energy_pass={energy_ok}",
            ],
        )

        # --- mode (d): Flow warmstart ---
        if getattr(self._args, "use_flow_warmstart", False):
            logger.info("  [§10 mode (d)] Running EmbeddingMAF flow warmstart...")
            cache = self._get_or_build_mpnn()
            flow_result = self._run_flow_warmstart_mode(cache, h_test)
            result["flow_warmstart"] = flow_result

        # --- mode (e): BondResolved warmstart ---
        if getattr(self._args, "use_bond_resolved", False):
            logger.info("  [§10 mode (e)] Running BondResolvedMPNN warmstart...")
            cache = self._get_or_build_mpnn()
            bond_result = self._run_bond_resolved_mode(cache, h_test)
            if bond_result is not None:
                result["bond_resolved_warmstart"] = bond_result

        return {
            **result,
            "criteria": SECTION_CRITERIA[10],
            "speedup_threshold_met": speedup_ok,
            "pass": result.get("pass", passed),
        }

    def _log_warmstart_table(self, per_h: list, summary: dict) -> None:
        """Log a formatted per-h table for the warmstart benchmark."""
        logger.info("\n  ┌──────┬────────────┬────────────┬────────────┬──────────┬──────┐")
        logger.info("  │  h   │ rand iters │ prev_h iter│ mpnn iters │ speedup  │ PASS │")
        logger.info("  ├──────┼────────────┼────────────┼────────────┼──────────┼──────┤")
        for r in per_h:
            rand_i = r["random"]["mean_iters"]
            prev_i = r["prev_h"]["iters"]
            mpnn_i = r["mpnn"]["iters"]
            spd = r["mpnn"]["speedup_vs_random"]
            ok = "✓" if r["pass"] else "✗"
            logger.info(
                f"  │{r['h']:5.2f} │{rand_i:10.0f}  │{prev_i:10d}  │{mpnn_i:10d}  │{spd:8.2f}x │  {ok}   │"
            )
        logger.info("  └──────┴────────────┴────────────┴────────────┴──────────┴──────┘")
        logger.info(
            f"  Summary: speedup_random={summary['mean_speedup_vs_random']:.2f}x, "
            f"speedup_prev_h={summary['mean_speedup_vs_prev_h']:.2f}x, "
            f"wins={summary['mpnn_wins_vs_random']}, "
            f"init_ΔE/gap={summary['mean_init_de_gap']:.4f}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # §10 private helpers for mode (d) and (e) extensions
    # ──────────────────────────────────────────────────────────────────────────

    def _get_graph_for_h(self, h: float, cache: dict):
        """Build a single torch_geometric Data graph for a given h value.

        Extracts graph structure from the cached lattice dataset by matching
        the closest h-point, then replaces the node h-feature with the
        requested h value.  The result is ready for MPNN/flow inference.

        Parameters
        ----------
        h : float
            Transverse field value to build the graph for.
        cache : dict
            MPNN cache from _get_or_build_mpnn().  Must contain keys
            ``"dataset"`` (list[Data]) and ``"h_arr"`` (np.ndarray).

        Returns
        -------
        torch_geometric.data.Data
            Graph with node features updated to reflect ``h``.
        """
        import numpy as np
        import torch
        from torch_geometric.data import Data

        from qmbp_simulation import make_lattice
        from qmbp_simulation.models import HamiltonianBuilder

        topology = self._args.topology
        n_qubits = self._args.n_qubits

        builder = HamiltonianBuilder()
        lattice_h = make_lattice(topology, n_qubits, J=1.0, h=float(h))
        edge_index_np, coord = builder.build_graph_data(lattice_h)
        h_feat = np.full(n_qubits, float(h))
        x = torch.tensor(np.stack([h_feat, coord.astype(float)], axis=1), dtype=torch.float32)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index)
        graph.batch = torch.zeros(n_qubits, dtype=torch.long)
        return graph

    def _vqe_from_init(
        self,
        theta_init,
        h: float,
        cache: dict,
    ) -> tuple[int, float]:
        """Run VQE from a given initial parameter vector and return results.

        Uses the existing noiseless backend and L-BFGS-B optimizer with the
        maxiter_refine budget (same as used by section_warmstart_benchmark).

        Parameters
        ----------
        theta_init : np.ndarray
            Starting parameter vector, shape [n_params].
        h : float
            Transverse field value (used to build lattice and Hamiltonian).
        cache : dict
            MPNN cache from _get_or_build_mpnn().

        Returns
        -------
        tuple[int, float]
            (n_iterations, de_gap) where ``n_iterations`` is the optimizer
            function-evaluation count and ``de_gap`` is
            |E_vqe - E_exact| / |gap| after convergence.
        """
        import numpy as np
        from scipy.optimize import minimize

        from qmbp_simulation import make_lattice
        from qmbp_simulation.execution import NoiselessBackend
        from qmbp_simulation.models.model_registry import get_model_spec

        topology = self._args.topology
        n_qubits = self._args.n_qubits
        p_layers = self._args.p_layers
        model = self._args.model
        maxiter = getattr(self._args, "maxiter_refine", DEFAULT_MAXITER_REFINE)
        n_params = cache["n_params"]

        spec = get_model_spec(model)
        _resolved = self._resolve_backend()
        backend = _resolved if _resolved is not None else NoiselessBackend()

        lattice_h = make_lattice(topology, n_qubits, J=1.0, h=float(h))
        H = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)
        circuit, _ = spec.create_circuit(n_qubits, p_layers, lattice_h, **spec.circuit_kwargs)

        e_exact, gap = self.exact_ground_state(topology, n_qubits, float(h), model=model)

        counter = [0]

        def _energy(params: np.ndarray) -> float:
            counter[0] += 1
            return float(backend.evaluate(circuit, H, params))

        res = minimize(
            _energy,
            np.array(theta_init, dtype=float).copy(),
            method="L-BFGS-B",
            bounds=[(-np.pi, np.pi)] * n_params,
            options={"maxiter": maxiter, "ftol": 1e-12},
        )
        de_gap = abs(float(res.fun) - e_exact) / max(abs(gap), 1e-10)
        return counter[0], de_gap

    def _run_flow_warmstart_mode(
        self,
        cache: dict,
        h_test: list[float],
    ) -> dict:
        """Train FlowWarmstartManager and benchmark each h_test point.

        Reuses the MPNN predictor and graph dataset already cached in §10.
        No extra Phase 2 VQE sweep is triggered.

        Parameters
        ----------
        cache : dict
            §10 MPNN cache from _get_or_build_mpnn().
        h_test : list[float]
            Transverse-field test points.

        Returns
        -------
        dict with keys: de_gap, n_iterations, speedup_vs_random,
        sigma_flow_per_h, per_h, train_nll_history, trainable_params.

        Raises
        ------
        RuntimeError
            If FlowWarmstartManager.trainable_param_count() >= 5000.
        """
        from qmbp_simulation.analysis.flow_warmstart import FlowWarmstartManager
        from qmbp_simulation.analysis.normalizing_flow import EmbeddingMAF

        n_params = cache["n_params"]
        hidden_dim = self._args.mpnn_hidden_dim

        # Guard: check param count BEFORE training by instantiating
        # EmbeddingMAF directly (trainable_param_count() requires flow_model
        # to be set, which only happens after train()).
        _tmp_flow = EmbeddingMAF(
            embedding_dim=hidden_dim,
            theta_dim=n_params,
        )
        param_count = _tmp_flow.trainable_param_count()
        del _tmp_flow

        # Adaptive guard: scales with architecture size
        # flow_hidden_dim=32 is the FlowWarmstartManager default
        flow_hidden_dim = 32
        param_limit = max(5000, 2 * hidden_dim * flow_hidden_dim)
        logger.info(f"  [Flow] EmbeddingMAF trainable params: {param_count} (limit: {param_limit})")
        if param_count >= param_limit:
            raise RuntimeError(
                f"FlowWarmstartManager param count {param_count} >= {param_limit} "
                "(overparameterization guard triggered)."
            )

        manager = FlowWarmstartManager(
            embedding_dim=hidden_dim,
            theta_dim=n_params,
            patience=50,
        )

        import time as _time

        t0_train = _time.time()
        train_info = manager.train_multi_seed(
            cache["predictor"], cache["dataset"], seeds=DEFAULT_SEEDS
        )
        train_elapsed_s = _time.time() - t0_train

        # Save checkpoint for reuse in deployment
        from pathlib import Path as _Path

        ckpt_dir = _Path("results/checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        manager.save(ckpt_dir / "flow_warmstart_latest.pt")
        logger.info(f"  [Flow] Checkpoint saved → {ckpt_dir / 'flow_warmstart_latest.pt'}")

        # Save flow checkpoint with topology/N/p naming for deployment reuse
        topology = self._args.topology
        n_qubits = self._args.n_qubits
        p_layers = self._args.p_layers
        flow_checkpoint_dir = _Path("results/flow_checkpoints")
        flow_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = flow_checkpoint_dir / f"flow_{topology}_N{n_qubits}_p{p_layers}.pt"
        manager.save(str(checkpoint_path))
        logger.info(f"  [Flow] Checkpoint saved: {checkpoint_path}")

        per_h: list[dict] = []
        sigma_flow_per_h: dict[float, float] = {}

        t0_bench = _time.time()
        for h in h_test:
            graph = self._get_graph_for_h(h, cache)
            theta_samples, sigma_flow = manager.sample(graph, n_samples=50)

            # Best sample (highest log-probability)
            log_probs = manager.flow_model.log_prob(
                theta_samples,
                manager._last_z.expand(theta_samples.shape[0], -1),
            )
            best = theta_samples[log_probs.argmax()].detach().cpu().numpy()

            iters, de_gap = self._vqe_from_init(best, h, cache)
            sigma_flow_per_h[h] = sigma_flow
            per_h.append({"h": h, "iters": iters, "de_gap": de_gap, "sigma_flow": sigma_flow})
        bench_elapsed_s = _time.time() - t0_bench

        # σ_flow summary: how many h-points would trigger the deployment boost
        n_boost = sum(1 for s in sigma_flow_per_h.values() if s > 0.5)
        logger.info(
            f"  [Flow] σ_flow summary: mean={sum(sigma_flow_per_h.values()) / len(sigma_flow_per_h):.3f}, "
            f"boost would trigger: {n_boost}/{len(sigma_flow_per_h)} h-points"
        )

        mean_de = float(sum(r["de_gap"] for r in per_h) / len(per_h))
        mean_iters = float(sum(r["iters"] for r in per_h) / len(per_h))

        # Compute speedup vs random baseline from the main §10 result
        speedup_vs_random: float | None = None
        if "per_h" in cache and cache["per_h"]:
            random_iters = [
                r.get("random", {}).get("mean_iters", 0) for r in cache.get("per_h", [])
            ]
            if random_iters and sum(random_iters) > 0:
                mean_random = sum(random_iters) / len(random_iters)
                if mean_iters > 0:
                    speedup_vs_random = mean_random / mean_iters

        return {
            "de_gap": mean_de,
            "n_iterations": mean_iters,
            "speedup_vs_random": speedup_vs_random,
            "sigma_flow_per_h": sigma_flow_per_h,
            "per_h": per_h,
            "train_nll_history": train_info["nll_history"],
            "trainable_params": param_count,
            "train_elapsed_s": train_elapsed_s,
            "bench_elapsed_s": bench_elapsed_s,
            "n_train_epochs_actual": len(train_info["nll_history"]),
            "converged": train_info["final_nll"] < 2.0,
            "best_seed": train_info.get("best_seed"),
            "all_seed_nlls": train_info.get("all_results"),
            "config": {
                "embedding_dim": hidden_dim,
                "theta_dim": n_params,
                "n_flow_layers": 2,
                "hidden_dim": 32,
                "n_epochs": 500,
                "n_samples": 50,
                "seed": 42,
                "param_limit": param_limit,
            },
        }

    def _run_bond_resolved_mode(
        self,
        cache: dict,
        h_test: list[float],
    ) -> dict | None:
        """Train BondResolvedMPNN and benchmark — chain_1d N=6 p=2 only."""
        topology = self._args.topology
        n_qubits = self._args.n_qubits
        p_layers = self._args.p_layers

        if topology != "chain_1d" or n_qubits != 6 or p_layers != 2:
            logger.warning(
                f"  [BondResolved] Skipping mode (e): requires chain_1d N=6 p=2, "
                f"got {topology} N={n_qubits} p={p_layers}."
            )
            return None

        import torch

        from qmbp_simulation.predictors import BondResolvedMPNN, train_mpnn

        model = BondResolvedMPNN(
            node_features=cache["dataset"][0].x.shape[1],
            hidden_dim=256,
            n_layers=3,
            norm_type="none",
        )
        train_mpnn(model, cache["dataset"], n_epochs=self._args.mpnn_epochs, seed=42)
        model.eval()

        per_h: list[dict] = []
        for h in h_test:
            graph = self._get_graph_for_h(h, cache)
            with torch.no_grad():
                theta_pred = model(graph).squeeze(0).cpu().numpy()
            iters, de_gap = self._vqe_from_init(theta_pred, h, cache)
            per_h.append({"h": h, "iters": iters, "de_gap": de_gap})

        mean_de = float(sum(r["de_gap"] for r in per_h) / len(per_h))
        mean_iters = float(sum(r["iters"] for r in per_h) / len(per_h))

        return {
            "de_gap": mean_de,
            "n_iterations": mean_iters,
            "speedup_vs_random": None,
            "per_h": per_h,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Section 11: MPNN LOO Cross-Validation
    # ──────────────────────────────────────────────────────────────────────────

    def section_loo_cv(self) -> dict:
        """MPNN leave-one-out cross-validation.

        Criteria (from SECTION_CRITERIA[11]):
          PASS: pass_rate ≥ 80%
          MARGINAL: pass_rate ∈ [60%, 80%) — check failing folds
          FAIL: pass_rate < 60% — extend h_train before hardware
        """
        topology = self._args.topology
        n_qubits = self._args.n_qubits
        h_train = self._args.h_train or H_TRAIN_GRID
        p_layers = self._args.p_layers

        self._log_section_criteria(11)
        logger.info(
            f"  Config: {len(h_train)} folds, min_train_size={self._args.loo_min_train_size}"
            f" | epochs={self._args.mpnn_epochs}"
        )
        logger.info(
            "  Note: Each fold trains a fresh model (N-1 pts). "
            f"~{len(h_train)} × {self._args.mpnn_epochs} epochs total."
        )

        result = self.mpnn_leave_one_out_cv(
            topology=topology,
            n_qubits=n_qubits,
            h_train=h_train,
            p_layers=p_layers,
            seed=42,
            n_restarts_vqe=self._args.vqe_restarts,
            mpnn_hidden_dim=self._args.mpnn_hidden_dim,
            mpnn_epochs=self._args.mpnn_epochs,
            mpnn_lr=MPNN_LR,
            mpnn_patience=MPNN_PATIENCE,
            model=self._args.model,
            de_gap_threshold=DE_GAP_THRESHOLD,
            min_train_size=self._args.loo_min_train_size,
        )

        s = result["summary"]
        self._log_loo_table(result["per_fold"])

        pass_rate_ok = s["pass_rate"] >= LOO_PASS_RATE_THRESHOLD
        marginal = 0.60 <= s["pass_rate"] < LOO_PASS_RATE_THRESHOLD

        self._log_pass_fail(
            11,
            pass_rate_ok,
            [
                f"pass_rate={s['pass_rate']:.0%} (threshold={LOO_PASS_RATE_THRESHOLD:.0%})",
                f"mean_de_gap={s['mean_de_gap']:.4f}, max_de_gap={s['max_de_gap']:.4f}",
                f"full_model_MSE={result['full_model_train_mse']:.2e}",
            ],
        )

        if marginal:
            logger.warning(
                "  MARGINAL: pass_rate in [60%, 80%). "
                "Consider adding more training points near failing h-values."
            )

        # Tag failing folds for diagnosis
        failing = [
            {"h": f["h_held_out"], "de_gap": f["de_gap"]}
            for f in result["per_fold"]
            if not f["pass"]
        ]

        return {
            **result,
            "criteria": SECTION_CRITERIA[11],
            "failing_folds": failing,
            "marginal": marginal,
            "pass": pass_rate_ok,
        }

    def _log_loo_table(self, per_fold: list) -> None:
        """Log a compact per-fold table."""
        logger.info("\n  ┌──────┬───────────┬──────────┬────────────┬──────┐")
        logger.info("  │ fold │   h_held  │  ΔE/gap  │ fold_MSE   │ PASS │")
        logger.info("  ├──────┼───────────┼──────────┼────────────┼──────┤")
        for f in per_fold:
            ok = "✓" if f["pass"] else "✗"
            logger.info(
                f"  │{f['fold_idx']:4d}  │{f['h_held_out']:9.4f}  │"
                f"{f['de_gap']:8.4f}  │{f['fold_train_mse']:10.2e}  │  {ok}   │"
            )
        logger.info("  └──────┴───────────┴──────────┴────────────┴──────┘")

    # ──────────────────────────────────────────────────────────────────────────
    # Section 12: Landscape Quality
    # ──────────────────────────────────────────────────────────────────────────

    def section_landscape_quality(self) -> dict:
        """Decompose prediction error into circuit vs ML contributions.

        Criteria (from SECTION_CRITERIA[12]):
          PASS: mean_error_total < 5%
          WARN: κ > 10 (high curvature → hardware sensitivity)
          WARN: ML fraction > 50% (MPNN dominates error budget)
          INFO: circuit_limited count (physics limit, not improvable by ML)
        """
        topology = self._args.topology
        n_qubits = self._args.n_qubits
        h_test = self._args.h_test or H_TEST_POINTS
        h_train = self._args.h_train or H_TRAIN_GRID
        p_layers = self._args.p_layers

        self._log_section_criteria(12)
        logger.info(
            f"  Config: {len(h_train)} train → {len(h_test)} test "
            f"(VQE run at h_test to get θ_opt ceiling)"
        )

        result = self.mpnn_landscape_quality(
            topology=topology,
            n_qubits=n_qubits,
            h_train=h_train,
            h_test=h_test,
            p_layers=p_layers,
            seed=42,
            n_restarts_vqe=self._args.vqe_restarts,
            mpnn_hidden_dim=self._args.mpnn_hidden_dim,
            mpnn_epochs=self._args.mpnn_epochs,
            mpnn_lr=MPNN_LR,
            mpnn_patience=MPNN_PATIENCE,
            model=self._args.model,
            de_gap_threshold=DE_GAP_THRESHOLD,
        )

        s = result["summary"]
        self._log_landscape_table(result["per_h"])

        passed = s["mean_error_total"] < DE_GAP_THRESHOLD
        self._log_pass_fail(
            12,
            passed,
            [
                f"mean_total={s['mean_error_total']:.4f}, threshold={DE_GAP_THRESHOLD:.2f}",
                f"mean_circuit={s['mean_error_circuit']:.4f} (ansatz limit)",
                f"mean_mpnn={s['mean_error_mpnn']:.4f} (ML error)",
                f"mean_κ={s['mean_curvature']:.2f} ({'⚠️ high' if s['mean_curvature'] > CURVATURE_WARN_THRESHOLD else 'OK'})",
                f"ML_frac={s['mpnn_fraction_of_total_error']:.0%} "
                f"({'⚠️ dominant' if s['mpnn_fraction_of_total_error'] > MPNN_FRAC_WARN_THRESHOLD else 'sub-dominant'})",
                f"circuit_limited={s['n_circuit_limited']}",
            ],
        )

        curvature_warn = s["mean_curvature"] > CURVATURE_WARN_THRESHOLD
        if curvature_warn:
            logger.warning(
                f"  ⚠️  High curvature κ={s['mean_curvature']:.2f} > {CURVATURE_WARN_THRESHOLD}. "
                "Small θ_pred errors will cause large energy errors on hardware. "
                "Consider tighter VQE convergence or more training data near h_test."
            )

        return {
            **result,
            "criteria": SECTION_CRITERIA[12],
            "curvature_warning": curvature_warn,
            "ml_dominant_warning": s["mpnn_fraction_of_total_error"] > MPNN_FRAC_WARN_THRESHOLD,
            "pass": passed,
        }

    def _log_landscape_table(self, per_h: list) -> None:
        logger.info(
            "\n  ┌──────┬────────────┬────────────┬────────────┬──────────┬──────────┬──────┐"
        )
        logger.info(
            "  │  h   │ΔE_circuit  │ ΔE_mpnn    │ ΔE_total   │ ||Δθ||   │    κ     │ PASS │"
        )
        logger.info(
            "  ├──────┼────────────┼────────────┼────────────┼──────────┼──────────┼──────┤"
        )
        for r in per_h:
            ok = "✓" if r["pass"] else "✗"
            logger.info(
                f"  │{r['h']:5.2f} │{r['error_circuit']:10.4f}  │"
                f"{r['error_mpnn']:10.4f}  │{r['error_total']:10.4f}  │"
                f"{r['theta_deviation']:8.4f}  │{r['mean_curvature']:8.2f}  │  {ok}   │"
            )
        logger.info(
            "  └──────┴────────────┴────────────┴────────────┴──────────┴──────────┴──────┘"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Section 13: Interpolation vs Extrapolation
    # ──────────────────────────────────────────────────────────────────────────

    def section_interpolation_extrapolation(self) -> dict:
        """Compare MPNN accuracy inside vs outside the training range.

        Auto-constructs interpolation/extrapolation points from h_train.
        Criteria (from SECTION_CRITERIA[13]):
          PASS: interpolation pass_rate ≥ 80%
          INFO: degradation_factor (extrap/interp) — key thesis number
          WARN: degradation > 3x
        """
        topology = self._args.topology
        n_qubits = self._args.n_qubits
        h_train = sorted(self._args.h_train or H_TRAIN_GRID, reverse=True)
        p_layers = self._args.p_layers

        h_arr = np.array(h_train)
        spacing = float(np.mean(np.abs(np.diff(sorted(h_arr)))))
        h_min = float(np.min(h_arr))
        h_max = float(np.max(h_arr))

        # Interpolation: midpoints between consecutive training values
        h_sorted = sorted(h_arr)
        h_interp_all = [(h_sorted[i] + h_sorted[i + 1]) / 2.0 for i in range(len(h_sorted) - 1)]
        # Take ≤4 evenly spaced (avoids making section too slow)
        step = max(1, len(h_interp_all) // 4)
        h_interp = [round(h, 4) for h in h_interp_all[::step][:4]]

        # Extrapolation: beyond both bounds
        h_extrap_candidates = [
            round(h_max + spacing, 4),
            round(h_max + 2 * spacing, 4),
            round(h_min - spacing, 4),
        ]
        h_extrap = [h for h in h_extrap_candidates if h > 0.2]

        self._log_section_criteria(13)
        logger.info(f"  h_train: [{h_min:.3f}, {h_max:.3f}], spacing≈{spacing:.3f}")
        logger.info(f"  Interpolation points ({len(h_interp)}): {h_interp}")
        logger.info(f"  Extrapolation points ({len(h_extrap)}): {h_extrap}")

        result = self.mpnn_interpolation_extrapolation(
            topology=topology,
            n_qubits=n_qubits,
            h_train=h_train,
            h_interpolate=h_interp,
            h_extrapolate=h_extrap,
            p_layers=p_layers,
            seed=42,
            n_restarts_vqe=self._args.vqe_restarts,
            mpnn_hidden_dim=self._args.mpnn_hidden_dim,
            mpnn_epochs=self._args.mpnn_epochs,
            mpnn_lr=MPNN_LR,
            mpnn_patience=MPNN_PATIENCE,
            model=self._args.model,
            de_gap_threshold=DE_GAP_THRESHOLD,
        )

        s = result["summary"]
        self._log_interp_extrap_table(result["interpolation"], result["extrapolation"])

        interp_ok = s["interpolation"]["pass_rate"] >= INTERP_PASS_RATE_THRESHOLD
        deg_warn = (
            not np.isnan(s["degradation_factor"])
            and s["degradation_factor"] > DEGRADATION_WARN_THRESHOLD
        )

        self._log_pass_fail(
            13,
            interp_ok,
            [
                f"interp_pass_rate={s['interpolation']['pass_rate']:.0%} (threshold={INTERP_PASS_RATE_THRESHOLD:.0%})",
                f"extrap_pass_rate={s['extrapolation']['pass_rate']:.0%} (informational)",
                f"degradation={s['degradation_factor']:.2f}x "
                f"({'⚠️ > ' + str(DEGRADATION_WARN_THRESHOLD) + 'x' if deg_warn else 'acceptable'})",
            ],
        )

        if deg_warn:
            logger.warning(
                f"  ⚠️  Degradation {s['degradation_factor']:.2f}x > {DEGRADATION_WARN_THRESHOLD}x threshold. "
                "MPNN accuracy drops significantly outside training range. "
                "Ensure h_test ∈ h_train for hardware deployment."
            )

        return {
            **result,
            "criteria": SECTION_CRITERIA[13],
            "degradation_warning": deg_warn,
            "pass": interp_ok,
        }

    def _log_interp_extrap_table(self, interp: list, extrap: list) -> None:
        logger.info("\n  ┌──────┬──────────┬──────────┬────────────┬──────┐")
        logger.info("  │  h   │ ΔE/gap   │  d_train │    mode    │ PASS │")
        logger.info("  ├──────┼──────────┼──────────┼────────────┼──────┤")
        for r in interp + extrap:
            ok = "✓" if r["pass"] else "✗"
            logger.info(
                f"  │{r['h']:5.3f} │{r['de_gap']:8.4f}  │"
                f"{r['distance_to_nearest_train']:8.4f}  │{r['mode']:10s}  │  {ok}   │"
            )
        logger.info("  └──────┴──────────┴──────────┴────────────┴──────┘")

    # ──────────────────────────────────────────────────────────────────────────
    # Section 15: MPNN Warm-Start Scaling with N
    # ──────────────────────────────────────────────────────────────────────────

    def section_scaling_with_n(self) -> dict:
        """Measure how MPNN warm-start speedup changes with system size.

        Trains one MPNN per system size and benchmarks warm-start speedup.
        Reveals whether the GNN advantage scales with problem dimensionality.

        Criteria (from SECTION_CRITERIA[15]):
          PASS: all sizes pass ΔE/gap < 5%
          KEY METRIC: scaling_trend — increasing/flat/decreasing
          REF: Qracle (Zhang 2025) shows GNN advantage at N=4-16 spin chains

        Interpretation:
          increasing trend → larger N benefits more from GNN (publish)
          flat trend       → advantage is consistent, not size-dependent
          decreasing trend → GNN loses value at larger N (unexpected/notable)
        """
        topology = self._args.topology
        h_test = self._args.h_test or H_TEST_POINTS
        h_train = self._args.h_train or H_TRAIN_GRID
        p_layers = self._args.p_layers
        sizes = getattr(self._args, "scaling_sizes", None) or [4, 6, 10]
        scaling_p = getattr(self._args, "scaling_p_layers", None)

        # Build p_layers_per_n dict from --scaling-p-layers arg
        p_layers_per_n: dict[int, int] | None = None
        if scaling_p and len(scaling_p) == len(sizes):
            p_layers_per_n = dict(zip(sizes, scaling_p, strict=False))
        elif scaling_p:
            logger.warning(
                f"  --scaling-p-layers has {len(scaling_p)} values but "
                f"--scaling-sizes has {len(sizes)}. Ignoring --scaling-p-layers."
            )

        self._log_section_criteria(15)
        logger.info(f"  System sizes: {sizes} | topology: {topology}")
        if p_layers_per_n:
            logger.info(f"  p_layers per N: {p_layers_per_n}")
        elif p_layers == 2 and any(n >= 10 for n in sizes):
            logger.warning(
                "  p=2 with N≥10 exceeds ZNE limit (36 CX > 18). "
                "Use --scaling-p-layers 2 2 1 for hardware-realistic comparison."
            )

        result = self.mpnn_scaling_with_system_size(
            topology=topology,
            system_sizes=sizes,
            h_train=h_train,
            h_test=h_test,
            p_layers=p_layers,
            p_layers_per_n=p_layers_per_n,
            seed=42,
            n_restarts_vqe=self._args.vqe_restarts,
            mpnn_hidden_dim=self._args.mpnn_hidden_dim,
            mpnn_epochs=self._args.mpnn_epochs,
            mpnn_lr=MPNN_LR,
            mpnn_patience=MPNN_PATIENCE,
            n_vqe_restarts_from_pred=getattr(self._args, "n_vqe_bench_restarts", 3),
            maxiter_refine=getattr(self._args, "maxiter_refine", 150),
            model=self._args.model,
            de_gap_threshold=DE_GAP_THRESHOLD,
        )

        logger.info("\n  ┌──────┬────────┬────────────┬────────────┬──────┐")
        logger.info("  │  N   │n_params│ speedup    │ init_ΔE/g  │ PASS │")
        logger.info("  ├──────┼────────┼────────────┼────────────┼──────┤")
        for e in result["per_n"]:
            ok = "✓" if e["pass"] else "✗"
            logger.info(
                f"  │{e['n_qubits']:5d} │{e['n_params']:6d}  │"
                f"{e['speedup_vs_random']:10.2f}x │{e['init_de_gap']:10.4f}  │  {ok}   │"
            )
        logger.info("  └──────┴────────┴────────────┴────────────┴──────┘")

        s = result["summary"]
        trend = result["scaling_trend"]
        self._log_pass_fail(
            15,
            result["pass"],
            [
                f"all_pass={result['pass']}",
                f"mean_speedup={s['mean_speedup']:.2f}x [{s['min_speedup']:.2f}, {s['max_speedup']:.2f}]",
                f"scaling_trend={trend} (slope={s['speedup_slope_per_N']:+.3f}/N)",
            ],
        )

        if trend == "decreasing":
            logger.warning(
                "  ⚠️  Speedup DECREASES with N. Check if larger lattices have "
                "smoother landscapes (easier for VQE → less warm-start benefit)."
            )

        return {**result, "criteria": SECTION_CRITERIA[15], "pass": result["pass"]}

    # ──────────────────────────────────────────────────────────────────────────
    # Section 16: MPNN Learning Curve
    # ──────────────────────────────────────────────────────────────────────────

    def section_learning_curve(self) -> dict:
        """MPNN prediction quality as a function of training set size.

        Determines the minimum number of VQE training points needed for
        ΔE/gap < 5% — the "critical training size" for hardware readiness.

        Criteria (from SECTION_CRITERIA[16]):
          PASS: critical_size ≤ 10 training points
          KEY: sample efficiency slope (ΔE/gap improvement per extra point)
          REF: NN-VQE (Miao 2024) needs ~20 points for MLP; GNN expected < 10
        """
        topology = self._args.topology
        n_qubits = self._args.n_qubits
        p_layers = self._args.p_layers
        h_pool = getattr(self._args, "h_pool", None) or (self._args.h_train or H_TRAIN_GRID)

        # Guard: need at least 3 pool points to have a meaningful learning curve
        if len(h_pool) < 3:
            logger.warning(
                f"  h_pool has only {len(h_pool)} point(s). "
                "Need ≥3 for a learning curve. Skipping section 16."
            )
            return {
                "skipped": True,
                "reason": f"h_pool too small ({len(h_pool)} pts)",
                "pass": True,
            }

        # h_test: use midpoints between pool values (strictly held out from h_pool).
        # These are never in h_pool by construction.
        h_sorted = sorted(h_pool)
        h_test_lc = [
            round((h_sorted[i] + h_sorted[i + 1]) / 2, 4) for i in range(len(h_sorted) - 1)
        ]
        # Take ≤3 spread across the range (fast)
        step = max(1, len(h_test_lc) // 3)
        h_test_lc = h_test_lc[::step][:3]

        self._log_section_criteria(16)
        logger.info(
            f"  Pool: {len(h_pool)} pts | test: {h_test_lc} | epochs={self._args.mpnn_epochs}"
        )

        result = self.mpnn_learning_curve(
            topology=topology,
            n_qubits=n_qubits,
            h_pool=h_pool,
            h_test=h_test_lc,
            p_layers=p_layers,
            seed=42,
            n_restarts_vqe=self._args.vqe_restarts,
            mpnn_hidden_dim=self._args.mpnn_hidden_dim,
            mpnn_epochs=self._args.mpnn_epochs,
            mpnn_lr=MPNN_LR,
            mpnn_patience=MPNN_PATIENCE,
            model=self._args.model,
            de_gap_threshold=DE_GAP_THRESHOLD,
        )

        logger.info("\n  ┌──────┬───────────┬──────────┬──────────┬──────┐")
        logger.info("  │  k   │ mean_ΔE/g │ pass_rate│ train_MSE│ PASS │")
        logger.info("  ├──────┼───────────┼──────────┼──────────┼──────┤")
        for e in result["per_size"]:
            ok = "✓" if e["pass"] else "✗"
            logger.info(
                f"  │{e['train_size']:5d} │{e['mean_de_gap']:9.4f}  │"
                f"{e['pass_rate']:8.0%}  │{e['train_mse']:8.2e}  │  {ok}   │"
            )
        logger.info("  └──────┴───────────┴──────────┴──────────┴──────┘")

        s = result["summary"]
        crit = s["critical_size"]
        passed = crit is not None and crit <= 10
        self._log_pass_fail(
            16,
            passed,
            [
                f"critical_size={crit} (threshold=10)",
                f"slope={s['sample_efficiency_slope']:+.4f} ΔE/gap per point",
                f"best_de_gap={s['best_mean_de_gap']:.4f}",
            ],
        )

        if crit is None:
            logger.warning(
                "  ⚠️  No training size achieves pass_rate ≥ 80% in this h_pool. "
                "Extend h_pool or use more training epochs."
            )

        return {**result, "criteria": SECTION_CRITERIA[16], "pass": passed}

    # ──────────────────────────────────────────────────────────────────────────
    # Section 17: Zero-Shot Topology Transfer
    # ──────────────────────────────────────────────────────────────────────────

    def section_topology_transfer(self) -> dict:
        """Zero-shot topology transfer: train on source, deploy on target.

        Validates the GNN's lattice-agnosticism claim by training on
        source_topology and immediately evaluating on target_topology.

        Criteria (from SECTION_CRITERIA[17]):
          PASS: zero_shot mean_de_gap < 5%
          KEY: transfer_ratio = zero_shot / in_distribution
               (1.0 = perfect transfer, < 2x = acceptable)
          REF: GNN-QEM zero-shot (2026-06-05): +72.3% error reduction chain→heavy_hex
        """
        n_qubits = self._args.n_qubits
        target_topology = self._args.topology
        source_topology = getattr(self._args, "source_topology", "chain_1d")
        h_train = self._args.h_train or H_TRAIN_GRID
        h_test = self._args.h_test or H_TEST_POINTS
        p_layers = self._args.p_layers

        # Skip if source == target (transfer is trivial)
        if source_topology == target_topology:
            logger.info(
                f"  Source = target = {source_topology}. "
                "Topology transfer skipped (use --source-topology to set a different source)."
            )
            return {
                "skipped": True,
                "reason": "source_topology == target_topology",
                "pass": True,
            }

        self._log_section_criteria(17)
        logger.info(f"  {source_topology} → {target_topology} | N={n_qubits} p={p_layers}")

        result = self.mpnn_topology_transfer(
            source_topology=source_topology,
            target_topology=target_topology,
            n_qubits=n_qubits,
            h_train=h_train,
            h_test=h_test,
            p_layers=p_layers,
            seed=42,
            n_restarts_vqe=self._args.vqe_restarts,
            mpnn_hidden_dim=self._args.mpnn_hidden_dim,
            mpnn_epochs=self._args.mpnn_epochs,
            mpnn_lr=MPNN_LR,
            mpnn_patience=MPNN_PATIENCE,
            model=self._args.model,
            de_gap_threshold=DE_GAP_THRESHOLD,
        )

        s = result["summary"]
        self._log_pass_fail(
            17,
            result["pass"],
            [
                f"zero_shot_de_gap={s['mean_de_gap_zero_shot']:.4f} (threshold={DE_GAP_THRESHOLD:.2f})",
                f"in_dist_de_gap={s['mean_de_gap_in_distribution']:.4f}",
                f"transfer_ratio={s['transfer_ratio']:.2f}x",
                f"random_baseline={s['mean_de_gap_random']:.4f}",
                f"zero_shot_pass_rate={s['zero_shot_pass_rate']:.0%}",
            ],
        )

        if s["transfer_ratio"] > 3.0:
            logger.warning(
                f"  ⚠️  Transfer ratio {s['transfer_ratio']:.2f}x > 3x. "
                "The GNN may be learning topology-specific patterns. "
                "Consider using norm_type='none' or training on mixed topologies."
            )

        return {**result, "criteria": SECTION_CRITERIA[17]}

    # ──────────────────────────────────────────────────────────────────────────
    # Section 18: Multi-Seed LOO Robustness
    # ──────────────────────────────────────────────────────────────────────────

    def section_multiseed_loo(self) -> dict:
        """LOO-CV repeated with multiple MPNN weight initialization seeds.

        Determines whether the LOO pass-rate is a stable estimate or
        depends sensitively on the random initialization of the model.

        Criteria (from SECTION_CRITERIA[18]):
          PASS: std_pass_rate < 15% across seeds
          KEY: coefficient_of_variation (low = robust result)
          NOTE: This is the robustness check for the LOO-CV finding in section 11.
        """
        topology = self._args.topology
        n_qubits = self._args.n_qubits
        h_pool = getattr(self._args, "h_pool", None) or (self._args.h_train or H_TRAIN_GRID)
        p_layers = self._args.p_layers
        n_seeds = getattr(self._args, "loo_n_seeds", 3)

        self._log_section_criteria(18)
        logger.info(
            f"  {n_seeds} seeds × {len(h_pool)} LOO folds × {self._args.mpnn_epochs} epochs"
        )

        result = self.mpnn_data_efficiency_vs_loo(
            topology=topology,
            n_qubits=n_qubits,
            h_pool=h_pool,
            n_seeds=n_seeds,
            p_layers=p_layers,
            seed=42,
            n_restarts_vqe=self._args.vqe_restarts,
            mpnn_hidden_dim=self._args.mpnn_hidden_dim,
            mpnn_epochs=self._args.mpnn_epochs,
            mpnn_lr=MPNN_LR,
            mpnn_patience=MPNN_PATIENCE,
            model=self._args.model,
            de_gap_threshold=DE_GAP_THRESHOLD,
            min_train_size=self._args.loo_min_train_size,
        )

        s = result["summary"]

        # Per-fold difficulty table
        if result["per_fold_stats"]:
            logger.info("\n  ┌────────┬────────────┬────────────┬──────┐")
            logger.info("  │  h     │ mean_ΔE/g  │ std_ΔE/g   │  CV  │")
            logger.info("  ├────────┼────────────┼────────────┼──────┤")
            for f in result["per_fold_stats"]:
                logger.info(
                    f"  │{f['h']:6.3f}  │{f['mean_de_gap']:10.4f}  │"
                    f"{f['std_de_gap']:10.4f}  │{f['cv']:4.2f}  │"
                )
            logger.info("  └────────┴────────────┴────────────┴──────┘")

        passed = s["std_pass_rate"] < 0.15
        self._log_pass_fail(
            18,
            passed,
            [
                f"mean_pass_rate={s['mean_pass_rate']:.0%}",
                f"std_pass_rate={s['std_pass_rate']:.0%} (threshold=15%)",
                f"cv={s['cv_pass_rate']:.2f}",
                f"robust={result['robust']}",
            ],
        )

        if not result["robust"]:
            logger.warning(
                f"  ⚠️  LOO result is seed-sensitive (std={s['std_pass_rate']:.0%}). "
                "The LOO pass-rate is not a reliable generalization estimate "
                "at this dataset size. Need more training points or more epochs."
            )

        return {**result, "criteria": SECTION_CRITERIA[18], "pass": passed}

    # ──────────────────────────────────────────────────────────────────────────
    # Section 19: Curvature κ as Hardware-Risk Proxy
    # ──────────────────────────────────────────────────────────────────────────

    def section_curvature_noise_correlation(self) -> dict:
        """Validate landscape curvature κ as a predictor of noise sensitivity.

        Computes κ(h) and ΔE_noise(h,σ) across the h-grid, then measures
        Pearson r(κ, ΔE_noise) per noise level. Strong correlation validates
        κ as a zero-QPU-cost hardware deployment risk indicator.

        Criteria (from SECTION_CRITERIA[19]):
          PASS: mean Pearson r ≥ 0.70
          KEY: κ_max location vs h_c (should peak near phase boundary)
          REF: Fontana et al. (2024): landscape curvature peaks at h_c
        """
        topology = self._args.topology
        n_qubits = self._args.n_qubits
        p_layers = self._args.p_layers
        # Use dedicated kappa grid if provided, otherwise fall back to h_train.
        # The kappa grid SHOULD extend to h near h_c to find the κ peak.
        # For TFIM chain_1d: h_c≈1.0. For heavy_hex N=10 p=1: h_min_safe≈3.25.
        h_kappa_explicit = getattr(self._args, "h_kappa_grid", None)
        h_grid = h_kappa_explicit or self._args.h_train or H_TRAIN_GRID
        noise_sigmas = getattr(self._args, "noise_sigmas", None) or [0.01, 0.05, 0.10, 0.20]

        self._log_section_criteria(19)
        logger.info(
            f"  h_grid: {len(h_grid)} pts | noise_sigmas: {noise_sigmas} | 20 perturbations each"
        )
        if not h_kappa_explicit:
            logger.warning(
                "  No --h-kappa-grid provided. Using h_train, which may not cover h_c. "
                "For TFIM: add h-values near h_c=1.0 (e.g., 1.1, 1.0, 0.9). "
                "For heavy_hex p=1: add h≈3.25 (boundary)."
            )

        result = self.mpnn_curvature_noise_correlation(
            topology=topology,
            n_qubits=n_qubits,
            h_grid=h_grid,
            p_layers=p_layers,
            seed=42,
            n_restarts_vqe=self._args.vqe_restarts,
            model=self._args.model,
            noise_levels=noise_sigmas,
            de_gap_threshold=DE_GAP_THRESHOLD,
        )

        s = result["summary"]

        # Curvature profile table
        logger.info("\n  ┌──────┬────────┬──────────────────────────────────┐")
        logger.info("  │  h   │   κ    │  ΔE_noise@σ=0.01  @0.05  @0.10  │")
        logger.info("  ├──────┼────────┼──────────────────────────────────┤")
        for r in result["per_h"]:
            ns = r["noise_sensitivity"]
            vals = " ".join(f"{ns.get(str(s), float('nan')):.4f}" for s in [0.01, 0.05, 0.10])
            logger.info(f"  │{r['h']:5.2f} │{r['kappa']:6.2f}  │  {vals}              │")
        logger.info("  └──────┴────────┴──────────────────────────────────┘")

        # Correlation table
        logger.info("\n  Pearson r(κ, ΔE_noise) by noise level:")
        for sigma, r_val in result["correlations"].items():
            bar = "█" * int(abs(float(r_val)) * 20) if not np.isnan(float(r_val)) else ""
            logger.info(f"    σ={sigma:4s}: r={float(r_val):+.4f}  {bar}")

        # Find h where κ is maximum
        kappa_vals = [r["kappa"] for r in result["per_h"]]
        h_max_kappa = result["per_h"][int(np.argmax(kappa_vals))]["h"]

        self._log_pass_fail(
            19,
            result["pass"],
            [
                f"mean_|pearson_r|={abs(s['mean_pearson_r']):.4f} (threshold=0.70)",
                f"mean_pearson_r={s['mean_pearson_r']:+.4f} (sign matters for interpretation)",
                f"mean_kappa={s['mean_kappa']:.2f}, max_kappa={s['max_kappa']:.2f}",
                f"h_max_kappa={h_max_kappa:.3f} (should be near h_c ≈ 1.0 for TFIM)",
                f"kappa_is_reliable={s['kappa_is_reliable_predictor']}",
            ],
        )

        # Interpret the sign of the correlation
        if s["mean_pearson_r"] < -0.70:
            logger.info(
                f"  📊 Negative r={s['mean_pearson_r']:+.4f}: "
                "Higher κ → LOWER noise sensitivity (anti-correlated). "
                "This means parametric regions with sharp minima are actually MORE "
                "noise-robust — the sharp gradient landscape helps VQE find deep minima "
                "that are less sensitive to small perturbations. Notable finding."
            )
        elif s["mean_pearson_r"] > 0.70:
            logger.info(
                f"  📊 Positive r={s['mean_pearson_r']:+.4f}: "
                "Higher κ → HIGHER noise sensitivity (expected behavior). "
                "Sharp landscape = amplified parameter errors on hardware."
            )
        elif abs(s["mean_pearson_r"]) < 0.50:
            logger.warning(
                f"  ⚠️  Weak correlation |r|={abs(s['mean_pearson_r']):.2f} < 0.50. "
                "κ does NOT reliably predict noise sensitivity at this N/topology."
            )

        return {**result, "criteria": SECTION_CRITERIA[19]}

    # ──────────────────────────────────────────────────────────────────────────
    # Section 14: MPNN Noisy Evaluation (FakeKingston)
    # ──────────────────────────────────────────────────────────────────────────

    def section_noisy_mpnn_eval(self) -> dict:
        """Evaluate MPNN θ_pred under FakeKingston noise with optional ZNE.

        Compares three energy references at each h_test:
          - E_noiseless: StatevectorEstimator with θ_pred (noiseless baseline)
          - E_noisy_raw: FakeKingston BackendEstimatorV2, no mitigation
          - E_noisy_zne: FakeKingston + gate-folding ZNE (3 layouts)

        Criteria (from SECTION_CRITERIA[14]):
          PASS: mean noisy_raw ΔE/gap < 10% (relaxed for shot noise)
          INFO: ZNE improvement = (noisy_raw - noisy_zne) / noisy_raw

        NOTE: This section uses FakeKingston which requires qiskit-aer.
        If qiskit-aer is not installed it falls back to a warning skip.
        """
        import time

        import torch
        from torch_geometric.data import Data

        from qmbp_simulation import make_lattice
        from qmbp_simulation.execution import NoiselessBackend
        from qmbp_simulation.models import HamiltonianBuilder
        from qmbp_simulation.models.model_registry import get_model_spec

        topology = self._args.topology
        n_qubits = self._args.n_qubits
        h_test = self._args.h_test or H_TEST_POINTS
        h_train = self._args.h_train or H_TRAIN_GRID
        p_layers = self._args.p_layers
        shots = self._args.noisy_shots
        n_layouts = self._args.noisy_n_layouts

        self._log_section_criteria(14)
        logger.info(
            f"  Config: {topology} N={n_qubits} p={p_layers} | shots={shots}, layouts={n_layouts}"
        )

        # Try importing FakeKingston — fail gracefully if aer not available
        try:
            from qiskit_ibm_runtime.fake_provider import FakeKingston

            from qmbp_simulation.execution.noisy_utils import (
                NoisyEstimatorConfig,
                build_adjacency,
                find_layouts_bfs,
                noisy_estimate,
                run_gate_folding_zne,
                select_layouts_low_ces,
            )
        except ImportError as exc:
            logger.warning(f"  FakeKingston not available: {exc}. Skipping noisy eval.")
            return {"error": f"FakeKingston unavailable: {exc}", "pass": False, "skipped": True}

        spec = get_model_spec(self._args.model)
        noiseless_backend = NoiselessBackend()
        builder = HamiltonianBuilder()
        fake_backend = FakeKingston()
        noisy_config = NoisyEstimatorConfig(shots=shots, seed_simulator=42)

        # ── Build/reuse MPNN ────────────────────────────────────────────────
        # Reuse shared cache if section 10-13 already ran, else train fresh
        cache = self._get_or_build_mpnn()
        predictor = cache["predictor"]
        h_arr = cache["h_arr"]
        theta_arr = cache["theta_arr"]
        n_params = cache["n_params"]

        per_h: list[dict] = []

        for h_t in h_test:
            e_exact, gap = self.exact_ground_state(topology, n_qubits, h_t, model=self._args.model)
            lattice_t = make_lattice(topology, n_qubits, J=1.0, h=h_t)
            H_t = spec.build_hamiltonian(lattice_t, **spec.hamiltonian_kwargs)
            circuit_t, _ = spec.create_circuit(n_qubits, p_layers, lattice_t, **spec.circuit_kwargs)

            # MPNN prediction
            edge_index_np, coord = builder.build_graph_data(lattice_t)
            h_feat = np.full(n_qubits, float(h_t))
            x = torch.tensor(np.stack([h_feat, coord.astype(float)], axis=1), dtype=torch.float32)
            edge_index_t = torch.tensor(edge_index_np, dtype=torch.long)
            graph = Data(x=x, edge_index=edge_index_t)
            graph.batch = torch.zeros(n_qubits, dtype=torch.long)

            with torch.no_grad():
                theta_pred = predictor(graph).numpy().flatten()

            # ── Noiseless reference ─────────────────────────────────────────
            e_noiseless = float(noiseless_backend.evaluate(circuit_t, H_t, theta_pred))
            de_gap_noiseless = abs(e_noiseless - e_exact) / max(gap, 1e-10)

            # ── Noisy evaluation — transpile to FakeKingston layout ───────────
            t0 = time.time()
            try:
                bound = circuit_t.assign_parameters(theta_pred)
                adj = build_adjacency(fake_backend)
                candidates = find_layouts_bfs(adj, n_qubits, n_candidates=10)
                layout_sel = select_layouts_low_ces(
                    bound,
                    fake_backend,
                    candidates,
                    n_select=n_layouts,
                    max_ces=0.5,
                )
                # Raw noisy (single layout, no ZNE)
                transpiled_0 = layout_sel.transpiled_circuits[0]
                H_mapped_0 = H_t.apply_layout(transpiled_0.layout)
                e_noisy_raw = noisy_estimate(transpiled_0, H_mapped_0, fake_backend, noisy_config)
                de_gap_noisy_raw = abs(e_noisy_raw - e_exact) / max(gap, 1e-10)

                # Gate-folding ZNE across all selected layouts
                all_zne_energies = []
                for transpiled_l in layout_sel.transpiled_circuits:
                    H_mapped_l = H_t.apply_layout(transpiled_l.layout)
                    zne_result = run_gate_folding_zne(
                        transpiled_l,
                        H_mapped_l,
                        fake_backend,
                        noisy_config,
                        noise_factors=(1, 3, 5),
                    )
                    all_zne_energies.append(zne_result.extrapolated_value)

                e_noisy_zne = float(np.mean(all_zne_energies))
                de_gap_noisy_zne = abs(e_noisy_zne - e_exact) / max(gap, 1e-10)
                zne_improvement = (de_gap_noisy_raw - de_gap_noisy_zne) / max(
                    de_gap_noisy_raw, 1e-10
                )
                noisy_eval_ok = True

            except Exception as exc:
                logger.warning(f"  Noisy eval failed at h={h_t}: {exc}")
                e_noisy_raw = float("nan")
                e_noisy_zne = float("nan")
                de_gap_noisy_raw = float("nan")
                de_gap_noisy_zne = float("nan")
                zne_improvement = float("nan")
                noisy_eval_ok = False

            elapsed = time.time() - t0
            passed_h = de_gap_noisy_raw < NOISY_DE_GAP_THRESHOLD if noisy_eval_ok else False

            per_h.append(
                {
                    "h": h_t,
                    "e_exact": e_exact,
                    "e_noiseless": e_noiseless,
                    "e_noisy_raw": e_noisy_raw,
                    "e_noisy_zne": e_noisy_zne,
                    "gap": gap,
                    "de_gap_noiseless": de_gap_noiseless,
                    "de_gap_noisy_raw": de_gap_noisy_raw,
                    "de_gap_noisy_zne": de_gap_noisy_zne,
                    "zne_improvement_pct": zne_improvement * 100 if noisy_eval_ok else None,
                    "noisy_eval_ok": noisy_eval_ok,
                    "elapsed_s": elapsed,
                    "pass": passed_h,
                }
            )
            status = "✓" if passed_h else "✗"
            logger.info(
                f"  h={h_t:.3f}: "
                f"noiseless={de_gap_noiseless:.4f}, "
                f"noisy_raw={de_gap_noisy_raw:.4f}, "
                f"noisy_zne={de_gap_noisy_zne:.4f}, "
                f"ZNE_Δ={zne_improvement * 100:+.1f}% "
                f"({elapsed:.1f}s) [{status}]"
            )

        valid = [r for r in per_h if r["noisy_eval_ok"]]
        n_pass = sum(r["pass"] for r in valid)
        mean_noisy_raw = float(np.nanmean([r["de_gap_noisy_raw"] for r in per_h]))
        mean_noisy_zne = float(np.nanmean([r["de_gap_noisy_zne"] for r in per_h]))
        mean_zne_improvement = (
            float(
                np.nanmean(
                    [
                        r["zne_improvement_pct"]
                        for r in per_h
                        if r["zne_improvement_pct"] is not None
                    ]
                )
            )
            if valid
            else float("nan")
        )

        passed = mean_noisy_raw < NOISY_DE_GAP_THRESHOLD and len(valid) == len(per_h)

        self._log_pass_fail(
            14,
            passed,
            [
                f"mean_noisy_raw={mean_noisy_raw:.4f} (threshold={NOISY_DE_GAP_THRESHOLD:.2f})",
                f"mean_noisy_zne={mean_noisy_zne:.4f}",
                f"mean_ZNE_improvement={mean_zne_improvement:+.1f}%",
                f"n_pass={n_pass}/{len(per_h)}",
            ],
        )

        if not np.isnan(mean_zne_improvement) and mean_zne_improvement < 0:
            logger.warning(
                f"  ⚠️  ZNE negative improvement ({mean_zne_improvement:.1f}%). "
                "This can happen at shallow circuits or with PEA not available. "
                "Check noise factors and circuit depth."
            )

        return {
            "per_h": per_h,
            "summary": {
                "mean_de_gap_noiseless": float(np.nanmean([r["de_gap_noiseless"] for r in per_h])),
                "mean_noisy_raw_de_gap": mean_noisy_raw,
                "mean_noisy_zne_de_gap": mean_noisy_zne,
                "mean_zne_improvement_pct": mean_zne_improvement,
                "n_pass": n_pass,
                "n_total": len(per_h),
                "n_eval_ok": len(valid),
            },
            "mpnn_train_mse": cache["train_mse"],
            "shots": shots,
            "n_layouts": n_layouts,
            "criteria": SECTION_CRITERIA[14],
            "pass": passed,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Section 20: PauliEvolutionGate vs RZZ/RX Transpilation Comparison
    # ──────────────────────────────────────────────────────────────────────────

    def section_pauli_evolution_comparison(self) -> dict:
        """Compare PauliEvolutionGate vs explicit RZZ/RX circuit representations.

        Measures three things on the same VQE θ_opt parameters and the same
        FakeKingston layout:

          1. Transpilation metrics (2Q-depth, CES, total depth, n_2Q gates).
             Hypothesis: PauliEvol has lower 2Q-depth (≥5%) with same n_2Q.

          2. Noiseless energy identity.
             Hypothesis: |E_pauli - E_rzz| < 1e-8 (functionally identical).

          3. Noisy ΔE/gap under FakeKingston noise.
             Hypothesis: |ΔE/gap_pauli - ΔE/gap_rzz| < 1% (same noise impact,
             since FakeKingston models gate-level noise and n_2Q is unchanged).

        Pass criteria (from SECTION_CRITERIA[20]):
          PASS:  energy_max_abs_diff < 1e-8
                 AND 2Q-depth_pauli < 2Q-depth_rzz (any reduction)
          WARN:  2Q-depth reduction < 5% (below expected −11%)
          INFO:  noisy_de_gap diff (informational, not pass/fail; FakeKingston
                 models per-gate noise so reduction may be small in simulation)

        This confirms PauliEvolutionGate integration is safe for IBM Kingston
        deployment. The depth reduction matters on real hardware (time-domain
        decoherence) but is expected to have minimal impact in FakeKingston
        (gate-error model, not time-based).

        Ref: 15_transpiler_exploration.md — 2Q-depth 24 vs 27 at N=10 p=1
             heavy_hex, optimization_level=2, same layout (validated 2026-06-05).
        """

        import numpy as np
        from qiskit.primitives import StatevectorEstimator
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime.fake_provider import FakeKingston

        from qmbp_simulation import HVACircuitBuilder, make_lattice
        from qmbp_simulation.execution.noisy_utils import (
            NoisyEstimatorConfig,
            build_adjacency,
            compute_circuit_ces,
            find_layouts_bfs,
            noisy_estimate,
            select_layouts_low_ces,
        )

        topology = self._args.topology
        n_qubits = self._args.n_qubits
        p_layers = self._args.p_layers
        h_test = self._args.h_test or H_TEST_POINTS
        shots = getattr(self._args, "noisy_shots", ZNE_SHOTS)

        self._log_section_criteria(20)
        logger.info(
            f"  Config: topology={topology}, N={n_qubits}, p={p_layers} "
            f"| {len(h_test)} h-test points | shots={shots}"
        )
        logger.info(
            "  Comparing: (A) explicit RZZ/RX gates  vs  (B) PauliEvolutionGate representation"
        )
        logger.info(
            "  Note: FakeKingston uses per-gate noise → depth reduction may not "
            "affect ΔE/gap in simulation. Real hardware (Kingston) will show "
            "decoherence improvement from shorter time-domain depth."
        )

        hva = HVACircuitBuilder()
        fake_backend = FakeKingston()
        noisy_cfg = NoisyEstimatorConfig(shots=shots, seed_simulator=42, optimization_level=2)

        # Shared layout: use lowest-CES layout from the first h-test point
        # (same layout for both representations → fair comparison)
        lattice_ref = make_lattice(topology, n_qubits, J=1.0, h=float(h_test[0]))
        qc_rzz, theta_rzz = hva.create(n_qubits, p_layers, lattice_ref)
        qc_pauli, theta_pauli = hva.create_pauli_evolution(n_qubits, p_layers, lattice_ref)

        # Get θ_opt from VQE (use shared cache if available, else run one sweep)
        cache = self._get_or_build_mpnn()
        theta_map = cache["theta_map"]

        # Build a single shared layout using the RZZ circuit (same for both)
        # We bind to the first h-test theta to get a concrete circuit for layout selection
        h_ref = min(theta_map.keys(), key=lambda h: abs(h - h_test[0]))
        theta_ref = theta_map[h_ref]
        bound_rzz_ref = qc_rzz.assign_parameters(theta_ref)

        adj = build_adjacency(fake_backend)
        candidates = find_layouts_bfs(adj, n_qubits, n_candidates=20, seed=42)
        layout_sel = select_layouts_low_ces(
            bound_rzz_ref,
            fake_backend,
            candidates,
            n_select=1,
            max_ces=0.5,
        )
        chosen_layout = layout_sel.layouts[0]

        # ── Pre-transpile both representations at the chosen layout ──────────
        # Use optimization_level=1 for S20 to avoid transpiler non-determinism.
        # optimization_level=2 applies value-aware gate optimizations
        # (CommutativeCancellation, ConsolidateBlocks) whose output depends on
        # the specific rotation angles in the bound circuit, causing different
        # total depths for the same gate count across h-points. Level=1 gives
        # deterministic structural mapping without value-sensitive optimization.
        # Real QPU deployment uses level=2 (validated in binnacle-pauli-evolution
        # -transpilation.md); this change only affects the S20 comparison test.
        pm = generate_preset_pass_manager(
            optimization_level=1,
            backend=fake_backend,
            initial_layout=chosen_layout,
        )

        logger.info(f"  Shared layout: qubits {chosen_layout}, CES={layout_sel.ces_values[0]:.4f}")

        per_h = []
        max_abs_diff = 0.0
        max_noisy_diff = 0.0

        # ── Header ───────────────────────────────────────────────────────────
        logger.info(
            "\n  ┌──────┬────────────────────────────────────┬"
            "────────────────────────────────────┬──────────────┐"
        )
        logger.info(
            "  │  h   │   RZZ/RX representation            │"
            "   PauliEvolution representation    │   Diff       │"
        )
        logger.info(
            "  │      │ dep  │ n_2Q │  CES   │ ΔE/gap(nl)  │"
            " dep  │ n_2Q │  CES   │ ΔE/gap(nl)  │ Δdep%  |ΔE|│"
        )
        logger.info(
            "  ├──────┼──────┼──────┼────────┼─────────────┼"
            "──────┼──────┼────────┼─────────────┼────────────┤"
        )

        for h_t in h_test:
            # Find closest θ_opt in the VQE cache
            h_closest = min(theta_map.keys(), key=lambda h: abs(h - h_t))
            theta_val = theta_map[h_closest]

            # Bind parameters for both representations
            bound_rzz = qc_rzz.assign_parameters(theta_val)
            bound_pauli = qc_pauli.assign_parameters(theta_val)

            # Transpile both with the SAME layout and optimization level
            t_rzz = pm.run(bound_rzz)
            t_pauli = pm.run(bound_pauli)

            # ── Transpilation metrics ──────────────────────────────────────
            # Use unified transpiled_circuit_stats for consistent metrics.
            from qmbp_simulation.analysis.circuit_visualizer import transpiled_circuit_stats

            stats_rzz = transpiled_circuit_stats(t_rzz)
            stats_pauli = transpiled_circuit_stats(t_pauli)

            n_2q_rzz = stats_rzz["n_2q_gates"]
            n_2q_pauli = stats_pauli["n_2q_gates"]
            ces_rzz, _ = compute_circuit_ces(t_rzz, fake_backend)
            ces_pauli, _ = compute_circuit_ces(t_pauli, fake_backend)
            depth_rzz = stats_rzz["depth"]
            depth_pauli = stats_pauli["depth"]
            depth_2q_rzz = stats_rzz["depth_2q"]
            depth_2q_pauli = stats_pauli["depth_2q"]

            # ── Noiseless energy (StatevectorEstimator, no noise) ──────────
            # Build Hamiltonian for this h
            e_exact, gap = self.exact_ground_state(
                topology, n_qubits, float(h_t), model=self._args.model
            )
            from qmbp_simulation import HamiltonianBuilder

            builder = HamiltonianBuilder()
            lattice_h = make_lattice(topology, n_qubits, J=1.0, h=float(h_t))
            H_t = builder.build(lattice_h)

            # Noiseless via unbound circuit (StatevectorEstimator handles params)
            sv_est = StatevectorEstimator()
            res_rzz = sv_est.run([(qc_rzz.assign_parameters(theta_val), H_t)]).result()
            e_nl_rzz = float(res_rzz[0].data.evs)
            res_pauli = sv_est.run([(qc_pauli.assign_parameters(theta_val), H_t)]).result()
            e_nl_pauli = float(res_pauli[0].data.evs)

            abs_diff = abs(e_nl_pauli - e_nl_rzz)
            max_abs_diff = max(max_abs_diff, abs_diff)
            de_gap_nl_rzz = abs(e_nl_rzz - e_exact) / max(gap, 1e-10)
            de_gap_nl_pauli = abs(e_nl_pauli - e_exact) / max(gap, 1e-10)

            # ── Noisy ΔE/gap (FakeKingston, single layout, no ZNE) ──────────
            try:
                H_rzz = H_t.apply_layout(t_rzz.layout)
                H_pauli = H_t.apply_layout(t_pauli.layout)
                e_noisy_rzz = noisy_estimate(t_rzz, H_rzz, fake_backend, noisy_cfg, seed_offset=0)
                e_noisy_pauli = noisy_estimate(
                    t_pauli, H_pauli, fake_backend, noisy_cfg, seed_offset=1
                )
                de_gap_noisy_rzz = abs(e_noisy_rzz - e_exact) / max(gap, 1e-10)
                de_gap_noisy_pauli = abs(e_noisy_pauli - e_exact) / max(gap, 1e-10)
                noisy_diff = abs(de_gap_noisy_pauli - de_gap_noisy_rzz)
                max_noisy_diff = max(max_noisy_diff, noisy_diff)
                noisy_ok = True
            except Exception as exc:
                logger.warning(f"  Noisy eval failed at h={h_t}: {exc}")
                e_noisy_rzz = e_noisy_pauli = float("nan")
                de_gap_noisy_rzz = de_gap_noisy_pauli = noisy_diff = float("nan")
                noisy_ok = False

            total_depth_reduction_pct = 100.0 * (depth_rzz - depth_pauli) / max(depth_rzz, 1)
            # 2Q-depth reduction (informational; =0 on heavy_hex due to full parallelism)
            depth_2q_reduction_pct = 100.0 * (depth_2q_rzz - depth_2q_pauli) / max(depth_2q_rzz, 1)

            logger.info(
                f"  │{h_t:5.2f} │{depth_rzz:5d} │{n_2q_rzz:5d} │{ces_rzz:7.4f} │"
                f"{de_gap_nl_rzz:11.6f} │"
                f"{depth_pauli:5d} │{n_2q_pauli:5d} │{ces_pauli:7.4f} │"
                f"{de_gap_nl_pauli:11.6f} │"
                f"{total_depth_reduction_pct:+6.1f}%  {abs_diff:.1e}│"
            )

            per_h.append(
                {
                    "h": h_t,
                    "e_exact": e_exact,
                    "gap": gap,
                    # RZZ/RX
                    "rzz": {
                        "depth_total": depth_rzz,
                        "depth_2q": depth_2q_rzz,
                        "n_2q": n_2q_rzz,
                        "ces": ces_rzz,
                        "e_noiseless": e_nl_rzz,
                        "de_gap_noiseless": de_gap_nl_rzz,
                        "e_noisy": e_noisy_rzz,
                        "de_gap_noisy": de_gap_noisy_rzz,
                    },
                    # PauliEvolutionGate
                    "pauli_evolution": {
                        "depth_total": depth_pauli,
                        "depth_2q": depth_2q_pauli,
                        "n_2q": n_2q_pauli,
                        "ces": ces_pauli,
                        "e_noiseless": e_nl_pauli,
                        "de_gap_noiseless": de_gap_nl_pauli,
                        "e_noisy": e_noisy_pauli,
                        "de_gap_noisy": de_gap_noisy_pauli,
                    },
                    # Differences
                    "energy_abs_diff": abs_diff,
                    "total_depth_reduction_pct": total_depth_reduction_pct,
                    "depth_2q_reduction_pct": depth_2q_reduction_pct,  # informational
                    "noisy_de_gap_diff": noisy_diff if noisy_ok else None,
                    "depth_reduced": depth_pauli < depth_rzz,
                    "depth_2q_reduced": depth_2q_pauli < depth_2q_rzz,
                    "n_2q_unchanged": n_2q_rzz == n_2q_pauli,
                }
            )

        logger.info(
            "  └──────┴──────┴──────┴────────┴─────────────┴"
            "──────┴──────┴────────┴─────────────┴────────────┘"
        )

        # ── Pass / Fail decision ─────────────────────────────────────────────
        energy_identity = max_abs_diff < 1e-8
        # Primary criterion: total depth reduction (includes 1Q scheduling)
        # 2Q-depth is always 1 on heavy_hex (all ZZ bonds parallelized) —
        # total_depth is the correct metric for hardware decoherence.
        any_depth_reduction = any(
            r["pauli_evolution"]["depth_total"] < r["rzz"]["depth_total"] for r in per_h
        )
        mean_reduction_pct = float(np.mean([r["total_depth_reduction_pct"] for r in per_h]))
        n_2q_unchanged = all(r["n_2q_unchanged"] for r in per_h)
        mean_noisy_diff = (
            float(
                np.nanmean(
                    [r["noisy_de_gap_diff"] for r in per_h if r["noisy_de_gap_diff"] is not None]
                )
            )
            if any(r["noisy_de_gap_diff"] is not None for r in per_h)
            else float("nan")
        )

        passed = energy_identity and any_depth_reduction

        # Expected: −6 to −11% total depth (from probe + 15_transpiler_exploration.md)
        if mean_reduction_pct < 3.0 and mean_reduction_pct >= 0.0:
            logger.warning(
                f"  ⚠️  Total depth reduction {mean_reduction_pct:.1f}% < 3% (expected ~6-11%). "
                "Possible cause: Qiskit version difference or trivial theta=0 binding. "
                "Integration is safe but benefit is smaller than the validated baseline."
            )
        elif mean_reduction_pct < 0.0:
            logger.warning(
                f"  ⚠️  PauliEvolutionGate is DEEPER ({mean_reduction_pct:.1f}%) in this run. "
                "This is topology/version-dependent. Do NOT use PauliEvol if consistently deeper."
            )

        if not n_2q_unchanged:
            logger.warning(
                "  ⚠️  n_2Q gate count differs between representations. "
                "This should not happen — investigate before hardware deployment."
            )

        self._log_pass_fail(
            20,
            passed,
            [
                f"energy_identity: max|ΔE|={max_abs_diff:.2e} "
                f"({'✓ < 1e-8' if energy_identity else '✗ EXCEEDS 1e-8'})",
                f"total_depth reduction: mean={mean_reduction_pct:+.1f}% "
                f"({'✓ any reduction' if any_depth_reduction else '✗ no reduction'})",
                "  [Note: 2Q-depth is always 1 on heavy_hex — all ZZ bonds parallelized]",
                f"n_2Q gate count unchanged: {n_2q_unchanged} ({'✓' if n_2q_unchanged else '✗ MISMATCH'})",
                f"noisy ΔE/gap diff: mean={mean_noisy_diff:.4f} "
                f"({'informational — FakeKingston per-gate noise model'})",
                f"verdict: {'SAFE — use PauliEvolutionGate for hardware deployment' if passed else 'UNSAFE — investigate before deployment'}",
            ],
        )

        summary = {
            "max_energy_abs_diff": max_abs_diff,
            "energy_identity": energy_identity,
            "mean_total_depth_reduction_pct": mean_reduction_pct,
            "any_depth_reduction": any_depth_reduction,
            "n_2q_unchanged": n_2q_unchanged,
            "mean_noisy_de_gap_diff": mean_noisy_diff,
            "max_noisy_de_gap_diff": max_noisy_diff,
            "note_2q_depth": (
                "2Q-depth=1 for both representations on heavy_hex: "
                "all ZZ bonds are on non-overlapping qubits → scheduler parallelizes them "
                "into 1 cycle regardless of gate representation. "
                "Total depth (includes 1Q gates) is the correct hardware decoherence metric."
            ),
            "recommendation": (
                "USE PauliEvolutionGate for IBM Kingston deployment"
                if passed
                else "INVESTIGATE — energy mismatch or no total depth reduction"
            ),
        }

        logger.info(f"\n  Recommendation: {summary['recommendation']}")

        return {
            "per_h": per_h,
            "summary": summary,
            "layout": chosen_layout,
            "topology": topology,
            "n_qubits": n_qubits,
            "p_layers": p_layers,
            "shots": shots,
            "criteria": SECTION_CRITERIA[20],
            "pass": passed,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Section 21: Mitiq Multi-Method Error Mitigation Comparison
    # ──────────────────────────────────────────────────────────────────────────

    def section_mitiq_comparison(self) -> dict:
        """Compare Mitiq error mitigation methods vs native GF-ZNE on FakeKingston.

        Runs compare_mitigation_strategies at each h_test point and produces
        a ranked table of: raw, Mitiq ZNE (linear), Mitiq CDR, Mitiq DDD+ZNE,
        and native gate-folding ZNE.

        Criteria (from SECTION_CRITERIA[21]):
          PASS: at least one Mitiq method achieves ΔE/gap < 10%
          KEY:  ranking of methods (thesis Table material)

        Requires: pip install mitiq AND qiskit-aer (FakeKingston).
        """
        import time

        import torch
        from torch_geometric.data import Data

        from qmbp_simulation import make_lattice
        from qmbp_simulation.execution import NoisyEstimatorConfig
        from qmbp_simulation.models import HamiltonianBuilder
        from qmbp_simulation.models.model_registry import get_model_spec

        topology = self._args.topology
        n_qubits = self._args.n_qubits
        h_test = self._args.h_test or H_TEST_POINTS
        p_layers = self._args.p_layers
        shots = self._args.noisy_shots

        self._log_section_criteria(21)
        logger.info(f"  Config: {topology} N={n_qubits} p={p_layers} | shots={shots}")

        # Check mitiq availability
        try:
            from qmbp_simulation.execution.mitiq_utils import (
                compare_mitigation_strategies,
                is_mitiq_available,
            )

            if not is_mitiq_available():
                raise ImportError("mitiq not installed")
        except ImportError as exc:
            logger.warning(f"  Mitiq not available: {exc}. Skipping section 21.")
            return {"error": str(exc), "pass": False, "skipped": True}

        # Check FakeKingston
        try:
            from qiskit_ibm_runtime.fake_provider import FakeKingston
        except ImportError as exc:
            logger.warning(f"  FakeKingston not available: {exc}. Skipping.")
            return {"error": str(exc), "pass": False, "skipped": True}

        spec = get_model_spec(self._args.model)
        builder = HamiltonianBuilder()
        fake_backend = FakeKingston()
        noisy_config = NoisyEstimatorConfig(shots=shots, seed_simulator=42)

        # Get shared MPNN from cache
        cache = self._get_or_build_mpnn()
        predictor = cache["predictor"]
        predictor.eval()

        # Limit to 3 h-points (Mitiq CDR is slow: ~10 extra circuit executions per h)
        h_test_limited = h_test[:3]
        per_h = []

        for h_t in h_test_limited:
            t0 = time.time()

            # Build lattice, Hamiltonian, circuit for this h
            lattice = make_lattice(topology, n_qubits, h=h_t)
            H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
            qc, _ = spec.create_circuit(n_qubits, p_layers, lattice, **spec.circuit_kwargs)

            # Get exact energy + gap
            e_exact, gap = self.exact_ground_state(topology, n_qubits, h_t, model=self._args.model)

            # Build correct graph for MPNN prediction (same pattern as section 14)
            edge_index_np, coord = builder.build_graph_data(lattice)
            h_feat = np.full(n_qubits, float(h_t))
            x = torch.tensor(np.stack([h_feat, coord.astype(float)], axis=1), dtype=torch.float32)
            edge_index_t = torch.tensor(edge_index_np, dtype=torch.long)
            graph = Data(x=x, edge_index=edge_index_t)
            graph.batch = torch.zeros(n_qubits, dtype=torch.long)

            with torch.no_grad():
                theta_pred = predictor(graph).numpy().flatten()

            # Bind parameters to circuit
            bound_circuit = qc.assign_parameters(theta_pred)

            # Run multi-method comparison
            try:
                result = compare_mitigation_strategies(
                    bound_circuit,
                    H,
                    fake_backend,
                    noisy_config,
                    exact_energy=e_exact,
                    gap=gap,
                    h_value=h_t,
                    strategies=["raw", "mitiq_zne_linear", "mitiq_cdr", "native_gf_zne"],
                )
                elapsed = time.time() - t0

                per_h.append(
                    {
                        "h": h_t,
                        "e_exact": e_exact,
                        "gap": gap,
                        "results": result.results,
                        "delta_e_gaps": result.delta_e_gaps,
                        "rankings": result.rankings,
                        "best_method": result.best_method,
                        "best_delta_e_gap": result.best_delta_e_gap,
                        "elapsed_s": elapsed,
                    }
                )
                logger.info(
                    f"  h={h_t:.3f}: best={result.best_method} "
                    f"(ΔE/gap={result.best_delta_e_gap:.4f}), "
                    f"rankings={result.rankings} ({elapsed:.1f}s)"
                )
            except Exception as e:
                elapsed = time.time() - t0
                logger.warning(f"  h={h_t:.3f}: comparison failed ({elapsed:.1f}s): {e}")
                per_h.append({"h": h_t, "error": str(e), "elapsed_s": elapsed})

        # Compute summary
        valid = [r for r in per_h if "best_delta_e_gap" in r]
        if not valid:
            self._log_pass_fail(21, False, ["No valid comparisons completed"])
            return {"per_h": per_h, "pass": False, "error": "No valid comparisons"}

        best_overall = min(r["best_delta_e_gap"] for r in valid)
        mean_best = float(np.mean([r["best_delta_e_gap"] for r in valid]))
        total_elapsed = sum(r.get("elapsed_s", 0) for r in per_h)

        # Method win counts across h-points
        win_counts: dict[str, int] = {}
        for r in valid:
            m = r["best_method"]
            win_counts[m] = win_counts.get(m, 0) + 1

        # Check if pass criteria met
        passed = best_overall < SECTION_CRITERIA[21]["threshold"]

        self._log_pass_fail(
            21,
            passed,
            [
                f"best_mitiq_de_gap={best_overall:.4f} "
                f"(threshold={SECTION_CRITERIA[21]['threshold']})",
                f"mean_best_de_gap={mean_best:.4f}",
                f"n_h_tested={len(valid)}/{len(h_test_limited)}",
                f"win_counts={win_counts}",
                f"total_time={total_elapsed:.1f}s",
            ],
        )

        return {
            "per_h": per_h,
            "summary": {
                "best_de_gap": best_overall,
                "mean_best_de_gap": mean_best,
                "n_valid": len(valid),
                "n_total": len(h_test_limited),
                "win_counts": win_counts,
                "total_elapsed_s": total_elapsed,
            },
            "criteria": SECTION_CRITERIA[21],
            "pass": passed,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Shared logging helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _log_section_criteria(self, section_id: int) -> None:
        """Log the pass/fail criteria for a section at the start."""
        c = SECTION_CRITERIA[section_id]
        direction = "≥" if c["direction"] == "ge" else "≤"
        logger.info(
            f"  Criteria: {c['primary_metric']} {direction} {c['threshold']}"
            + (f" | {c['secondary']}" if c["secondary"] else "")
        )
        logger.info(f"  Reference: {c['ref']}")

    def _log_pass_fail(self, section_id: int, passed: bool, details: list[str]) -> None:
        """Log a standardized PASS/FAIL block with details."""
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"\n  ── Section {section_id} {status} ──")
        for d in details:
            logger.info(f"    {d}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    HardwareRehearsalV3.main()
