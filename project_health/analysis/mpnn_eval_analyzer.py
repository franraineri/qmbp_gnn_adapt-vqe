#!/usr/bin/env python3
"""MPNN Evaluation Analyzer — processes HW_REHEARSAL_V3 section 10-14 results.

Scans results/experiments/exp_hw_rehearsal_v3/ (or any provided path) for
run_*.json files produced by run_hardware_rehearsal_v3.py, then:

1. Extracts per-section metrics for sections 10-14
2. Produces a structured report with pass/fail verdicts and key numbers
3. Generates thesis-ready tables (warm-start speedup, LOO-CV, landscape)
4. Flags regressions or warnings across multiple runs

Usage:
    python -m project_health.analysis.mpnn_eval_analyzer
    python -m project_health.analysis.mpnn_eval_analyzer --verbose
    python -m project_health.analysis.mpnn_eval_analyzer --json report.json
    python -m project_health.analysis.mpnn_eval_analyzer --results-dir results/experiments

Output:
    Structured text report + optional JSON dump.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS_DIR = ROOT / "results" / "experiments"

# Where V3 results land
V3_EXP_ID = "HW_REHEARSAL_V3"
V3_DIR_PATTERN = "exp_hw_rehearsal_v3"


# ═══════════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class WarmstartResult:
    """Section 10 — MPNN warm-start benchmark."""

    mean_speedup_vs_random: float
    mean_speedup_vs_prev_h: float
    mpnn_wins_vs_random: str
    mpnn_wins_vs_prev_h: str
    mean_init_de_gap: float
    mean_final_de_gap: float
    n_train_points: int
    mpnn_train_mse: float
    pass_: bool

    @property
    def speedup_rating(self) -> str:
        if self.mean_speedup_vs_random >= 3.0:
            return "excellent"
        if self.mean_speedup_vs_random >= 1.5:
            return "good"
        if self.mean_speedup_vs_random >= 1.0:
            return "marginal"
        return "negative"


@dataclass
class LOOCVResult:
    """Section 11 — LOO cross-validation."""

    n_folds: int
    n_pass: int
    pass_rate: float
    mean_de_gap: float
    max_de_gap: float
    std_de_gap: float
    full_model_mse: float
    failing_h_values: list[float]
    pass_: bool

    @property
    def reliability_label(self) -> str:
        if self.pass_rate >= 0.90:
            return "reliable"
        if self.pass_rate >= 0.80:
            return "acceptable"
        if self.pass_rate >= 0.60:
            return "marginal"
        return "unreliable"


@dataclass
class LandscapeResult:
    """Section 12 — landscape quality / error decomposition."""

    mean_error_circuit: float
    mean_error_mpnn: float
    mean_error_total: float
    mean_theta_deviation: float
    mean_curvature: float
    n_circuit_limited: str
    mpnn_fraction: float
    pass_: bool

    @property
    def bottleneck(self) -> str:
        """Which component dominates the error budget."""
        if self.mpnn_fraction > 0.7:
            return "ml_dominated"
        if self.mpnn_fraction < 0.3:
            return "circuit_limited"
        return "balanced"

    @property
    def hardware_risk(self) -> str:
        if self.mean_curvature > 20.0:
            return "high"
        if self.mean_curvature > 10.0:
            return "medium"
        return "low"


@dataclass
class InterpExtrapResult:
    """Section 13 — interpolation vs extrapolation."""

    interp_mean_de_gap: float
    interp_pass_rate: float
    extrap_mean_de_gap: float
    extrap_pass_rate: float
    degradation_factor: float
    h_train_range: list[float]
    pass_: bool

    @property
    def deployment_range_note(self) -> str:
        if self.degradation_factor < 1.5:
            return "MPNN generalizes well outside training range"
        if self.degradation_factor < 3.0:
            return "Moderate degradation outside training range — stay within h_train bounds"
        return "Strong degradation — only deploy within h_train range"


@dataclass
class NoisyEvalResult:
    """Section 14 — noisy MPNN evaluation."""

    mean_de_gap_noiseless: float
    mean_noisy_raw_de_gap: float
    mean_noisy_zne_de_gap: float
    mean_zne_improvement_pct: float
    n_pass: int
    n_total: int
    shots: int
    pass_: bool

    @property
    def noise_overhead(self) -> float:
        """How much extra error noise adds (in units of ΔE/gap)."""
        return self.mean_noisy_raw_de_gap - self.mean_de_gap_noiseless

    @property
    def zne_effective(self) -> bool:
        return self.mean_zne_improvement_pct > 5.0  # > 5% improvement


@dataclass
class ScalingWithNResult:
    """Section 15 — warm-start speedup vs system size N."""

    system_sizes: list[int]
    speedups: list[float]
    mean_speedup: float
    min_speedup: float
    max_speedup: float
    speedup_slope_per_n: float
    scaling_trend: str  # "increasing" | "flat" | "decreasing"
    pass_: bool


@dataclass
class LearningCurveResult:
    """Section 16 — MPNN prediction quality vs training set size."""

    train_sizes: list[int]
    mean_de_gaps: list[float]
    pass_rates: list[float]
    critical_size: int | None
    sample_efficiency_slope: float
    best_mean_de_gap: float
    pass_: bool

    @property
    def sample_efficient(self) -> bool:
        return self.critical_size is not None and self.critical_size <= 10


@dataclass
class TopologyTransferResult:
    """Section 17 — zero-shot topology transfer."""

    source_topology: str
    target_topology: str
    mean_de_gap_zero_shot: float
    mean_de_gap_in_distribution: float
    mean_de_gap_random: float
    transfer_ratio: float
    zero_shot_pass_rate: float
    in_dist_pass_rate: float
    pass_: bool
    skipped: bool = False

    @property
    def transfer_quality(self) -> str:
        if self.skipped:
            return "skipped"
        if self.transfer_ratio < 1.5:
            return "excellent"
        if self.transfer_ratio < 2.5:
            return "good"
        if self.transfer_ratio < 5.0:
            return "marginal"
        return "poor"


@dataclass
class MultiSeedLOOResult:
    """Section 18 — LOO-CV robustness across random seeds."""

    n_seeds: int
    mean_pass_rate: float
    std_pass_rate: float
    cv_pass_rate: float
    mean_de_gap: float
    std_de_gap: float
    robust: bool
    pass_: bool

    @property
    def stability_label(self) -> str:
        if self.std_pass_rate < 0.05:
            return "very_stable"
        if self.std_pass_rate < 0.15:
            return "stable"
        if self.std_pass_rate < 0.30:
            return "unstable"
        return "very_unstable"


@dataclass
class CurvatureNoiseResult:
    """Section 19 — curvature κ as hardware-risk proxy."""

    mean_kappa: float
    max_kappa: float
    mean_pearson_r: float
    pearson_r_per_sigma: dict[str, float]
    kappa_is_reliable: bool
    pass_: bool

    @property
    def correlation_interpretation(self) -> str:
        r = self.mean_pearson_r
        if r > 0.70:
            return "positive_strong: high_kappa→high_noise_sensitivity"
        if r < -0.70:
            return "negative_strong: high_kappa→low_noise_sensitivity (anti-correlated)"
        if abs(r) >= 0.50:
            return "moderate"
        return "weak: kappa_not_predictive"


@dataclass
class MPNNEvalReport:
    """Aggregate report for one V3 run."""

    run_file: str
    timestamp: str
    topology: str
    n_qubits: int
    p_layers: int
    model: str
    warmstart: WarmstartResult | None = None
    loo_cv: LOOCVResult | None = None
    landscape: LandscapeResult | None = None
    interp_extrap: InterpExtrapResult | None = None
    noisy_eval: NoisyEvalResult | None = None
    # Extended sections (15-19)
    scaling_with_n: ScalingWithNResult | None = None
    learning_curve: LearningCurveResult | None = None
    topology_transfer: TopologyTransferResult | None = None
    multiseed_loo: MultiSeedLOOResult | None = None
    curvature_noise: CurvatureNoiseResult | None = None
    warnings: list[str] = field(default_factory=list)

    def overall_pass(self) -> bool:
        results = [
            self.warmstart.pass_ if self.warmstart else None,
            self.loo_cv.pass_ if self.loo_cv else None,
            self.landscape.pass_ if self.landscape else None,
            self.interp_extrap.pass_ if self.interp_extrap else None,
            self.scaling_with_n.pass_ if self.scaling_with_n else None,
            self.learning_curve.pass_ if self.learning_curve else None,
            self.topology_transfer.pass_ if self.topology_transfer else None,
            self.multiseed_loo.pass_ if self.multiseed_loo else None,
            self.curvature_noise.pass_ if self.curvature_noise else None,
        ]
        results = [r for r in results if r is not None]
        return all(results)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "run_file": self.run_file,
            "timestamp": self.timestamp,
            "system": {
                "topology": self.topology,
                "n_qubits": self.n_qubits,
                "p_layers": self.p_layers,
                "model": self.model,
            },
            "overall_pass": self.overall_pass(),
            "warnings": self.warnings,
        }
        if self.warmstart:
            d["warmstart"] = asdict(self.warmstart)
            d["warmstart"]["speedup_rating"] = self.warmstart.speedup_rating
        if self.loo_cv:
            d["loo_cv"] = asdict(self.loo_cv)
            d["loo_cv"]["reliability_label"] = self.loo_cv.reliability_label
        if self.landscape:
            d["landscape"] = asdict(self.landscape)
            d["landscape"]["bottleneck"] = self.landscape.bottleneck
            d["landscape"]["hardware_risk"] = self.landscape.hardware_risk
        if self.interp_extrap:
            d["interp_extrap"] = asdict(self.interp_extrap)
            d["interp_extrap"]["deployment_range_note"] = self.interp_extrap.deployment_range_note
        if self.noisy_eval:
            d["noisy_eval"] = asdict(self.noisy_eval)
            d["noisy_eval"]["noise_overhead"] = self.noisy_eval.noise_overhead
            d["noisy_eval"]["zne_effective"] = self.noisy_eval.zne_effective
        if self.scaling_with_n:
            d["scaling_with_n"] = asdict(self.scaling_with_n)
        if self.learning_curve:
            d["learning_curve"] = asdict(self.learning_curve)
            d["learning_curve"]["sample_efficient"] = self.learning_curve.sample_efficient
        if self.topology_transfer:
            d["topology_transfer"] = asdict(self.topology_transfer)
            d["topology_transfer"]["transfer_quality"] = self.topology_transfer.transfer_quality
        if self.multiseed_loo:
            d["multiseed_loo"] = asdict(self.multiseed_loo)
            d["multiseed_loo"]["stability_label"] = self.multiseed_loo.stability_label
        if self.curvature_noise:
            d["curvature_noise"] = asdict(self.curvature_noise)
            d["curvature_noise"]["interpretation"] = self.curvature_noise.correlation_interpretation
        return d


# ═══════════════════════════════════════════════════════════════════════════════
# Parsers — extract structured data from raw JSON
# ═══════════════════════════════════════════════════════════════════════════════


def _safe_float(d: dict, *keys, default: float = float("nan")) -> float:
    """Navigate nested dict with fallback."""
    val = d
    for k in keys:
        if not isinstance(val, dict) or k not in val:
            return default
        val = val[k]
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def parse_warmstart(data: dict) -> WarmstartResult | None:
    """Parse section_10 data dict into WarmstartResult."""
    s = data.get("data", {})
    if not s:
        return None
    summary = s.get("summary", {})
    return WarmstartResult(
        mean_speedup_vs_random=_safe_float(summary, "mean_speedup_vs_random"),
        mean_speedup_vs_prev_h=_safe_float(summary, "mean_speedup_vs_prev_h"),
        mpnn_wins_vs_random=str(summary.get("mpnn_wins_vs_random", "?/?")),
        mpnn_wins_vs_prev_h=str(summary.get("mpnn_wins_vs_prev_h", "?/?")),
        mean_init_de_gap=_safe_float(summary, "mean_init_de_gap"),
        mean_final_de_gap=_safe_float(summary, "mean_final_de_gap_mpnn"),
        n_train_points=int(s.get("n_train_points", 0)),
        mpnn_train_mse=_safe_float(s, "mpnn_train_mse"),
        pass_=bool(s.get("pass", False)),
    )


def parse_loo_cv(data: dict) -> LOOCVResult | None:
    """Parse section_11 data dict into LOOCVResult."""
    s = data.get("data", {})
    if not s:
        return None
    summary = s.get("summary", {})
    failing = [f["h_held_out"] for f in s.get("per_fold", []) if not f.get("pass", True)]
    return LOOCVResult(
        n_folds=int(summary.get("n_folds", 0)),
        n_pass=int(summary.get("n_pass", 0)),
        pass_rate=_safe_float(summary, "pass_rate"),
        mean_de_gap=_safe_float(summary, "mean_de_gap"),
        max_de_gap=_safe_float(summary, "max_de_gap"),
        std_de_gap=_safe_float(summary, "std_de_gap"),
        full_model_mse=_safe_float(s, "full_model_train_mse"),
        failing_h_values=failing,
        pass_=bool(s.get("pass", False)),
    )


def parse_landscape(data: dict) -> LandscapeResult | None:
    """Parse section_12 data dict into LandscapeResult."""
    s = data.get("data", {})
    if not s:
        return None
    summary = s.get("summary", {})
    return LandscapeResult(
        mean_error_circuit=_safe_float(summary, "mean_error_circuit"),
        mean_error_mpnn=_safe_float(summary, "mean_error_mpnn"),
        mean_error_total=_safe_float(summary, "mean_error_total"),
        mean_theta_deviation=_safe_float(summary, "mean_theta_deviation"),
        mean_curvature=_safe_float(summary, "mean_curvature"),
        n_circuit_limited=str(summary.get("n_circuit_limited", "0/0")),
        mpnn_fraction=_safe_float(summary, "mpnn_fraction_of_total_error"),
        pass_=bool(s.get("pass", False)),
    )


def parse_interp_extrap(data: dict) -> InterpExtrapResult | None:
    """Parse section_13 data dict into InterpExtrapResult."""
    s = data.get("data", {})
    if not s:
        return None
    summary = s.get("summary", {})
    h_range = s.get("h_train_range", [float("nan"), float("nan")])
    interp_s = summary.get("interpolation", {})
    extrap_s = summary.get("extrapolation", {})
    return InterpExtrapResult(
        interp_mean_de_gap=_safe_float(interp_s, "mean_de_gap"),
        interp_pass_rate=_safe_float(interp_s, "pass_rate"),
        extrap_mean_de_gap=_safe_float(extrap_s, "mean_de_gap"),
        extrap_pass_rate=_safe_float(extrap_s, "pass_rate"),
        degradation_factor=_safe_float(summary, "degradation_factor"),
        h_train_range=h_range,
        pass_=bool(s.get("pass", False)),
    )


def parse_noisy_eval(data: dict) -> NoisyEvalResult | None:
    """Parse section_14 data dict into NoisyEvalResult."""
    s = data.get("data", {})
    if not s or s.get("skipped"):
        return None
    summary = s.get("summary", {})
    return NoisyEvalResult(
        mean_de_gap_noiseless=_safe_float(summary, "mean_de_gap_noiseless"),
        mean_noisy_raw_de_gap=_safe_float(summary, "mean_noisy_raw_de_gap"),
        mean_noisy_zne_de_gap=_safe_float(summary, "mean_noisy_zne_de_gap"),
        mean_zne_improvement_pct=_safe_float(summary, "mean_zne_improvement_pct"),
        n_pass=int(summary.get("n_pass", 0)),
        n_total=int(summary.get("n_total", 0)),
        shots=int(s.get("shots", 0)),
        pass_=bool(s.get("pass", False)),
    )


def parse_scaling_with_n(data: dict) -> ScalingWithNResult | None:
    """Parse section_15 data dict."""
    s = data.get("data", {})
    if not s:
        return None
    per_n = s.get("per_n", [])
    summary = s.get("summary", {})
    return ScalingWithNResult(
        system_sizes=[e["n_qubits"] for e in per_n],
        speedups=[e["speedup_vs_random"] for e in per_n],
        mean_speedup=_safe_float(summary, "mean_speedup"),
        min_speedup=_safe_float(summary, "min_speedup"),
        max_speedup=_safe_float(summary, "max_speedup"),
        speedup_slope_per_n=_safe_float(summary, "speedup_slope_per_N"),
        scaling_trend=s.get("scaling_trend", "unknown"),
        pass_=bool(s.get("pass", False)),
    )


def parse_learning_curve(data: dict) -> LearningCurveResult | None:
    """Parse section_16 data dict."""
    s = data.get("data", {})
    if not s:
        return None
    per_size = s.get("per_size", [])
    summary = s.get("summary", {})
    return LearningCurveResult(
        train_sizes=[e["train_size"] for e in per_size],
        mean_de_gaps=[e["mean_de_gap"] for e in per_size],
        pass_rates=[e["pass_rate"] for e in per_size],
        critical_size=summary.get("critical_size"),
        sample_efficiency_slope=_safe_float(summary, "sample_efficiency_slope"),
        best_mean_de_gap=_safe_float(summary, "best_mean_de_gap"),
        pass_=bool(s.get("pass", False)),
    )


def parse_topology_transfer(data: dict) -> TopologyTransferResult | None:
    """Parse section_17 data dict."""
    s = data.get("data", {})
    if not s:
        return None
    if s.get("skipped"):
        return TopologyTransferResult(
            source_topology=s.get("reason", ""),
            target_topology="",
            mean_de_gap_zero_shot=float("nan"),
            mean_de_gap_in_distribution=float("nan"),
            mean_de_gap_random=float("nan"),
            transfer_ratio=float("nan"),
            zero_shot_pass_rate=float("nan"),
            in_dist_pass_rate=float("nan"),
            pass_=True,
            skipped=True,
        )
    summary = s.get("summary", {})
    return TopologyTransferResult(
        source_topology=s.get("source_topology", ""),
        target_topology=s.get("target_topology", ""),
        mean_de_gap_zero_shot=_safe_float(summary, "mean_de_gap_zero_shot"),
        mean_de_gap_in_distribution=_safe_float(summary, "mean_de_gap_in_distribution"),
        mean_de_gap_random=_safe_float(summary, "mean_de_gap_random"),
        transfer_ratio=_safe_float(summary, "transfer_ratio"),
        zero_shot_pass_rate=_safe_float(summary, "zero_shot_pass_rate"),
        in_dist_pass_rate=_safe_float(summary, "in_dist_pass_rate"),
        pass_=bool(s.get("pass", False)),
        skipped=False,
    )


def parse_multiseed_loo(data: dict) -> MultiSeedLOOResult | None:
    """Parse section_18 data dict."""
    s = data.get("data", {})
    if not s:
        return None
    summary = s.get("summary", {})
    return MultiSeedLOOResult(
        n_seeds=int(summary.get("n_seeds", 0)),
        mean_pass_rate=_safe_float(summary, "mean_pass_rate"),
        std_pass_rate=_safe_float(summary, "std_pass_rate"),
        cv_pass_rate=_safe_float(summary, "cv_pass_rate"),
        mean_de_gap=_safe_float(summary, "mean_de_gap"),
        std_de_gap=_safe_float(summary, "std_de_gap"),
        robust=bool(s.get("robust", False)),
        pass_=bool(s.get("pass", False)),
    )


def parse_curvature_noise(data: dict) -> CurvatureNoiseResult | None:
    """Parse section_19 data dict."""
    s = data.get("data", {})
    if not s:
        return None
    summary = s.get("summary", {})
    return CurvatureNoiseResult(
        mean_kappa=_safe_float(summary, "mean_kappa"),
        max_kappa=_safe_float(summary, "max_kappa"),
        mean_pearson_r=_safe_float(summary, "mean_pearson_r"),
        pearson_r_per_sigma=s.get("correlations", {}),
        kappa_is_reliable=bool(summary.get("kappa_is_reliable_predictor", False)),
        pass_=bool(s.get("pass", False)),
    )


def parse_run(path: Path) -> MPNNEvalReport | None:
    """Parse a single run_*.json file into an MPNNEvalReport."""
    try:
        with open(path) as f:
            raw = json.load(f)
    except Exception as exc:
        logger.warning(f"Could not read {path}: {exc}")
        return None

    # Validate it's a V3 run
    cfg = raw.get("config", {})
    exp_id = cfg.get("experiment_id", "")
    if exp_id != V3_EXP_ID:
        logger.debug(f"Skipping {path.name}: experiment_id={exp_id!r} (not {V3_EXP_ID})")
        return None

    system = cfg.get("system", {})
    results = raw.get("results", {})
    timestamp = raw.get("timestamp", path.stem.replace("run_", ""))

    report = MPNNEvalReport(
        run_file=str(path),
        timestamp=timestamp,
        topology=system.get("topology", "unknown"),
        n_qubits=int(system.get("n_qubits", 0)),
        p_layers=int(system.get("p_layers", 1)),
        model=system.get("model", "tfim"),
    )

    s10 = results.get("section_10")
    s11 = results.get("section_11")
    s12 = results.get("section_12")
    s13 = results.get("section_13")
    s14 = results.get("section_14")
    s15 = results.get("section_15")
    s16 = results.get("section_16")
    s17 = results.get("section_17")
    s18 = results.get("section_18")
    s19 = results.get("section_19")

    if s10:
        report.warmstart = parse_warmstart(s10)
    if s11:
        report.loo_cv = parse_loo_cv(s11)
    if s12:
        report.landscape = parse_landscape(s12)
    if s13:
        report.interp_extrap = parse_interp_extrap(s13)
    if s14:
        report.noisy_eval = parse_noisy_eval(s14)
    if s15:
        report.scaling_with_n = parse_scaling_with_n(s15)
    if s16:
        report.learning_curve = parse_learning_curve(s16)
    if s17:
        report.topology_transfer = parse_topology_transfer(s17)
    if s18:
        report.multiseed_loo = parse_multiseed_loo(s18)
    if s19:
        report.curvature_noise = parse_curvature_noise(s19)

    # Auto-generate warnings
    if report.warmstart and report.warmstart.mean_speedup_vs_random < 1.0:
        report.warnings.append("MPNN warm-start HURTS: speedup < 1x. Retrain before QPU.")
    if report.loo_cv and report.loo_cv.pass_rate < 0.60:
        report.warnings.append(
            f"LOO-CV unreliable: pass_rate={report.loo_cv.pass_rate:.0%}. "
            "Extend h_train grid before hardware."
        )
    if report.landscape and report.landscape.hardware_risk == "high":
        report.warnings.append(
            f"High landscape curvature κ={report.landscape.mean_curvature:.1f}. "
            "Hardware θ_pred errors will be amplified."
        )
    if report.interp_extrap and report.interp_extrap.degradation_factor > 5.0:
        report.warnings.append(
            f"Strong extrapolation degradation {report.interp_extrap.degradation_factor:.1f}x. "
            "Only deploy within h_train range on QPU."
        )
    # Sections 15-19 auto-warnings
    if report.scaling_with_n and report.scaling_with_n.scaling_trend == "decreasing":
        report.warnings.append(
            f"Speedup trend DECREASING with N "
            f"(slope={report.scaling_with_n.speedup_slope_per_n:+.2f}/N). "
            "GNN advantage shrinks at larger system sizes."
        )
    if report.learning_curve and report.learning_curve.critical_size is None:
        report.warnings.append(
            "No critical training size found — model never achieves 80% pass rate. "
            "Expand h_pool or increase epochs before hardware."
        )
    if (
        report.topology_transfer
        and not report.topology_transfer.skipped
        and report.topology_transfer.transfer_ratio > 3.0
    ):
        report.warnings.append(
            f"Poor topology transfer ratio {report.topology_transfer.transfer_ratio:.1f}x > 3x. "
            "GNN may be memorizing topology-specific patterns."
        )
    if report.multiseed_loo and not report.multiseed_loo.robust:
        report.warnings.append(
            f"LOO result seed-sensitive (std={report.multiseed_loo.std_pass_rate:.0%} > 15%). "
            "Need more training data or epochs for stable LOO estimate."
        )
    if report.curvature_noise and abs(report.curvature_noise.mean_pearson_r) < 0.50:
        report.warnings.append(
            f"Weak κ-noise correlation |r|={abs(report.curvature_noise.mean_pearson_r):.2f}. "
            "κ is NOT a reliable hardware-risk proxy at this N/topology."
        )

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Scanner
# ═══════════════════════════════════════════════════════════════════════════════


def scan_v3_results(results_dir: Path) -> list[MPNNEvalReport]:
    """Find and parse all HW_REHEARSAL_V3 run files under results_dir."""
    reports: list[MPNNEvalReport] = []

    # Look in exp_hw_rehearsal_v3/ and also directly under results_dir
    search_paths = [
        results_dir / V3_DIR_PATTERN,
        results_dir,
    ]
    json_files: list[Path] = []
    for sp in search_paths:
        if sp.is_dir():
            json_files.extend(sorted(sp.glob("run_*.json")))

    if not json_files:
        logger.warning(f"No run_*.json files found in {results_dir}")
        return reports

    for path in json_files:
        report = parse_run(path)
        if report is not None:
            reports.append(report)

    logger.info(f"Parsed {len(reports)} V3 runs from {len(json_files)} files.")
    return reports


# ═══════════════════════════════════════════════════════════════════════════════
# Report formatter
# ═══════════════════════════════════════════════════════════════════════════════


def format_report(reports: list[MPNNEvalReport], verbose: bool = False) -> str:
    """Format parsed reports into a human-readable text report."""
    if not reports:
        return "No HW_REHEARSAL_V3 results found.\n"

    lines: list[str] = []

    def hr(char="─", width=72):
        lines.append(char * width)

    lines.append("MPNN Evaluation Analysis — HW_REHEARSAL_V3")
    lines.append(f"Runs analyzed: {len(reports)}")
    hr("═")

    for i, r in enumerate(reports, 1):
        lines.append(
            f"\n[Run {i}/{len(reports)}] {r.timestamp} | "
            f"{r.topology} N={r.n_qubits} p={r.p_layers} model={r.model}"
        )
        lines.append(f"  File: {Path(r.run_file).name}")
        status = "✅ PASS" if r.overall_pass() else "❌ FAIL"
        lines.append(f"  Overall: {status}")

        if r.warnings:
            lines.append("  ⚠️  Warnings:")
            for w in r.warnings:
                lines.append(f"     • {w}")

        hr()

        # Section 10: Warm-start
        if r.warmstart:
            ws = r.warmstart
            status = "PASS" if ws.pass_ else "FAIL"
            lines.append(f"  [10] Warm-Start Benchmark         [{status}]")
            lines.append(
                f"       speedup_random={ws.mean_speedup_vs_random:.2f}x "
                f"speedup_prev_h={ws.mean_speedup_vs_prev_h:.2f}x "
                f"rating={ws.speedup_rating}"
            )
            lines.append(
                f"       wins_random={ws.mpnn_wins_vs_random} wins_prev_h={ws.mpnn_wins_vs_prev_h}"
            )
            lines.append(
                f"       init_ΔE/gap={ws.mean_init_de_gap:.4f} "
                f"final_ΔE/gap={ws.mean_final_de_gap:.4f} "
                f"train_MSE={ws.mpnn_train_mse:.2e}"
            )

        # Section 11: LOO-CV
        if r.loo_cv:
            loo = r.loo_cv
            status = "PASS" if loo.pass_ else "FAIL"
            lines.append(f"  [11] LOO Cross-Validation         [{status}]")
            lines.append(
                f"       pass_rate={loo.pass_rate:.0%} ({loo.n_pass}/{loo.n_folds}) "
                f"reliability={loo.reliability_label}"
            )
            lines.append(
                f"       mean_ΔE/gap={loo.mean_de_gap:.4f} "
                f"max_ΔE/gap={loo.max_de_gap:.4f} "
                f"std={loo.std_de_gap:.4f}"
            )
            if loo.failing_h_values:
                lines.append(f"       failing_h={[f'{h:.3f}' for h in loo.failing_h_values]}")

        # Section 12: Landscape
        if r.landscape:
            ls = r.landscape
            status = "PASS" if ls.pass_ else "FAIL"
            lines.append(f"  [12] Landscape Quality            [{status}]")
            lines.append(
                f"       ΔE_circuit={ls.mean_error_circuit:.4f} "
                f"ΔE_mpnn={ls.mean_error_mpnn:.4f} "
                f"ΔE_total={ls.mean_error_total:.4f}"
            )
            lines.append(
                f"       ML_frac={ls.mpnn_fraction:.0%} ({ls.bottleneck}) "
                f"κ={ls.mean_curvature:.2f} (risk={ls.hardware_risk})"
            )
            lines.append(
                f"       ||Δθ||={ls.mean_theta_deviation:.4f} "
                f"circuit_limited={ls.n_circuit_limited}"
            )

        # Section 13: Interp/Extrap
        if r.interp_extrap:
            ie = r.interp_extrap
            status = "PASS" if ie.pass_ else "FAIL"
            deg_str = (
                f"{ie.degradation_factor:.2f}x" if not np.isnan(ie.degradation_factor) else "N/A"
            )
            lines.append(f"  [13] Interpolation/Extrapolation  [{status}]")
            lines.append(
                f"       interp: mean={ie.interp_mean_de_gap:.4f} pass={ie.interp_pass_rate:.0%}"
            )
            lines.append(
                f"       extrap: mean={ie.extrap_mean_de_gap:.4f} pass={ie.extrap_pass_rate:.0%}"
            )
            lines.append(f"       degradation={deg_str} | {ie.deployment_range_note}")

        # Section 14: Noisy eval
        if r.noisy_eval:
            ne = r.noisy_eval
            status = "PASS" if ne.pass_ else "FAIL"
            lines.append(f"  [14] Noisy Evaluation             [{status}]")
            lines.append(
                f"       noiseless={ne.mean_de_gap_noiseless:.4f} "
                f"noisy_raw={ne.mean_noisy_raw_de_gap:.4f} "
                f"noisy_zne={ne.mean_noisy_zne_de_gap:.4f}"
            )
            lines.append(
                f"       ZNE_improvement={ne.mean_zne_improvement_pct:+.1f}% "
                f"({'effective' if ne.zne_effective else 'marginal'}) "
                f"noise_overhead={ne.noise_overhead:.4f}"
            )
            lines.append(f"       shots={ne.shots} pass={ne.n_pass}/{ne.n_total}")

        # Section 15: Scaling with N
        if r.scaling_with_n:
            sn = r.scaling_with_n
            status = "PASS" if sn.pass_ else "FAIL"
            lines.append(f"  [15] Scaling with N               [{status}]")
            lines.append(
                f"       sizes={sn.system_sizes} speedups={[f'{s:.2f}x' for s in sn.speedups]}"
            )
            lines.append(
                f"       mean={sn.mean_speedup:.2f}x [{sn.min_speedup:.2f},{sn.max_speedup:.2f}] "
                f"trend={sn.scaling_trend} slope={sn.speedup_slope_per_n:+.3f}/N"
            )

        # Section 16: Learning curve
        if r.learning_curve:
            lc = r.learning_curve
            status = "PASS" if lc.pass_ else "FAIL"
            lines.append(f"  [16] Learning Curve               [{status}]")
            lines.append(
                f"       critical_size={lc.critical_size} sample_efficient={lc.sample_efficient}"
            )
            lines.append(
                f"       slope={lc.sample_efficiency_slope:+.4f} ΔE/gap per pt "
                f"best_de_gap={lc.best_mean_de_gap:.4f}"
            )
            if verbose and lc.train_sizes:
                for k, de, pr in zip(lc.train_sizes, lc.mean_de_gaps, lc.pass_rates, strict=False):
                    lines.append(f"         k={k:3d}: mean_ΔE/gap={de:.4f} pass={pr:.0%}")

        # Section 17: Topology transfer
        if r.topology_transfer:
            tt = r.topology_transfer
            status = "PASS" if tt.pass_ else ("SKIP" if tt.skipped else "FAIL")
            lines.append(f"  [17] Topology Transfer            [{status}]")
            if not tt.skipped:
                lines.append(
                    f"       {tt.source_topology} → {tt.target_topology} "
                    f"quality={tt.transfer_quality}"
                )
                lines.append(
                    f"       zero_shot={tt.mean_de_gap_zero_shot:.4f} "
                    f"in_dist={tt.mean_de_gap_in_distribution:.4f} "
                    f"ratio={tt.transfer_ratio:.2f}x"
                )

        # Section 18: Multi-seed LOO
        if r.multiseed_loo:
            ms = r.multiseed_loo
            status = "PASS" if ms.pass_ else "FAIL"
            lines.append(f"  [18] Multi-Seed LOO               [{status}]")
            lines.append(
                f"       mean_pass_rate={ms.mean_pass_rate:.0%}±{ms.std_pass_rate:.0%} "
                f"stability={ms.stability_label}"
            )
            lines.append(f"       cv={ms.cv_pass_rate:.2f} robust={ms.robust}")

        # Section 19: Curvature-noise correlation
        if r.curvature_noise:
            cn = r.curvature_noise
            status = "PASS" if cn.pass_ else "FAIL"
            lines.append(f"  [19] Curvature κ as Risk Proxy    [{status}]")
            lines.append(
                f"       mean_r={cn.mean_pearson_r:+.4f} "
                f"(|r|={abs(cn.mean_pearson_r):.4f}) "
                f"kappa_reliable={cn.kappa_is_reliable}"
            )
            lines.append(f"       {cn.correlation_interpretation}")
            lines.append(f"       mean_κ={cn.mean_kappa:.2f} max_κ={cn.max_kappa:.2f}")

        hr()

    # ── Cross-run summary (if multiple runs) ─────────────────────────────────
    if len(reports) > 1:
        lines.append("\n── Cross-Run Summary ──")

        speedups = [r.warmstart.mean_speedup_vs_random for r in reports if r.warmstart]
        if speedups:
            lines.append(
                f"  Speedup vs random: "
                f"mean={np.mean(speedups):.2f}x ± {np.std(speedups):.2f} "
                f"[{np.min(speedups):.2f}, {np.max(speedups):.2f}]"
            )

        pass_rates = [r.loo_cv.pass_rate for r in reports if r.loo_cv]
        if pass_rates:
            lines.append(
                f"  LOO pass_rate: mean={np.mean(pass_rates):.0%} ± {np.std(pass_rates):.0%}"
            )

        deg_factors = [
            r.interp_extrap.degradation_factor
            for r in reports
            if r.interp_extrap and not np.isnan(r.interp_extrap.degradation_factor)
        ]
        if deg_factors:
            lines.append(f"  Degradation factor: mean={np.mean(deg_factors):.2f}x")

        hr("═")

    return "\n".join(lines)


def format_thesis_table(reports: list[MPNNEvalReport]) -> str:
    """Generate a thesis-ready LaTeX-style ASCII table for the key metrics."""
    if not reports:
        return ""

    lines = [
        "\n── Thesis Table: MPNN Evaluation Summary ──",
        f"{'Metric':<40} {'Value':<20} {'Reference'}",
        "─" * 72,
    ]

    # Use the most recent run
    r = reports[-1]

    if r.warmstart:
        ws = r.warmstart
        lines.append(
            f"{'Warm-start speedup vs random':<40} {ws.mean_speedup_vs_random:.2f}x {' ' * 10} Qracle: 1.64x (Zhang 2025)"
        )
        lines.append(f"{'Warm-start speedup vs prev-h':<40} {ws.mean_speedup_vs_prev_h:.2f}x")
        lines.append(f"{'MPNN init ΔE/gap (no VQE)':<40} {ws.mean_init_de_gap:.4f}")

    if r.loo_cv:
        loo = r.loo_cv
        lines.append(f"{'LOO-CV pass-rate':<40} {loo.pass_rate:.0%} ({loo.n_pass}/{loo.n_folds})")
        lines.append(f"{'LOO-CV mean ΔE/gap':<40} {loo.mean_de_gap:.4f}")
        lines.append(f"{'LOO-CV max ΔE/gap':<40} {loo.max_de_gap:.4f}")

    if r.landscape:
        ls = r.landscape
        lines.append(f"{'ΔE/gap (circuit expressibility)':<40} {ls.mean_error_circuit:.4f}")
        lines.append(f"{'ΔE/gap (MPNN ML error)':<40} {ls.mean_error_mpnn:.4f}")
        lines.append(f"{'ΔE/gap (total deployment)':<40} {ls.mean_error_total:.4f}")
        lines.append(f"{'ML fraction of total error':<40} {ls.mpnn_fraction:.0%}")
        lines.append(f"{'Mean landscape curvature κ':<40} {ls.mean_curvature:.2f}")

    if r.interp_extrap:
        ie = r.interp_extrap
        lines.append(f"{'Interpolation ΔE/gap (mean)':<40} {ie.interp_mean_de_gap:.4f}")
        if not np.isnan(ie.extrap_mean_de_gap):
            lines.append(f"{'Extrapolation ΔE/gap (mean)':<40} {ie.extrap_mean_de_gap:.4f}")
        if not np.isnan(ie.degradation_factor):
            lines.append(f"{'Extrap degradation factor':<40} {ie.degradation_factor:.2f}x")

    if r.noisy_eval:
        ne = r.noisy_eval
        lines.append(f"{'Noisy ΔE/gap (FakeTorino raw)':<40} {ne.mean_noisy_raw_de_gap:.4f}")
        lines.append(f"{'Noisy ΔE/gap (after ZNE)':<40} {ne.mean_noisy_zne_de_gap:.4f}")
        lines.append(f"{'ZNE improvement':<40} {ne.mean_zne_improvement_pct:+.1f}%")

    if r.scaling_with_n:
        sn = r.scaling_with_n
        lines.append(
            f"{'Speedup scaling trend':<40} {sn.scaling_trend} (slope={sn.speedup_slope_per_n:+.3f}/N)"
        )
        lines.append(
            f"{'Speedup range across N':<40} [{sn.min_speedup:.2f}x, {sn.max_speedup:.2f}x]"
        )

    if r.learning_curve:
        lc = r.learning_curve
        lines.append(
            f"{'Critical training size':<40} {lc.critical_size} pts         NN-VQE: ~20 pts (MLP)"
        )
        lines.append(
            f"{'Learning curve slope':<40} {lc.sample_efficiency_slope:+.4f} ΔE/gap per point"
        )

    if r.topology_transfer and not r.topology_transfer.skipped:
        tt = r.topology_transfer
        lines.append(
            f"{'Zero-shot transfer ΔE/gap':<40} {tt.mean_de_gap_zero_shot:.4f}       GNN-QEM: 72.3% reduction (2026)"
        )
        lines.append(f"{'Transfer ratio (zero/in-dist)':<40} {tt.transfer_ratio:.2f}x")

    if r.multiseed_loo:
        ms = r.multiseed_loo
        lines.append(
            f"{'LOO stability (std pass_rate)':<40} ±{ms.std_pass_rate:.0%} ({ms.stability_label})"
        )

    if r.curvature_noise:
        cn = r.curvature_noise
        lines.append(f"{'Pearson r(κ, ΔE_noise)':<40} {cn.mean_pearson_r:+.4f}")
        lines.append(f"{'κ reliable predictor':<40} {cn.kappa_is_reliable}")
        lines.append(f"{'Interpretation':<40} {cn.correlation_interpretation.split(':')[0]}")

    lines.append("─" * 72)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze HW_REHEARSAL_V3 MPNN evaluation results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Root results directory (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        metavar="PATH",
        help="Also write JSON report to PATH",
    )
    parser.add_argument(
        "--thesis-table",
        action="store_true",
        default=False,
        help="Print thesis-ready table",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Verbose output",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    reports = scan_v3_results(args.results_dir)

    if not reports:
        print(f"No HW_REHEARSAL_V3 results in {args.results_dir}")
        print(
            "Run: python scripts/experiment_runners/run_hardware_rehearsal_v3.py "
            "--skip-hardware-sections"
        )
        return 1

    text = format_report(reports, verbose=args.verbose)
    print(text)

    if args.thesis_table:
        print(format_thesis_table(reports))

    if args.json:
        out = {"runs": [r.to_dict() for r in reports], "n_runs": len(reports)}
        Path(args.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nJSON report written to: {args.json}")

    # Exit 0 only if all runs pass
    all_pass = all(r.overall_pass() for r in reports)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
