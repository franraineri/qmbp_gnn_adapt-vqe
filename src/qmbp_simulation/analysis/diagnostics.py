"""Pipeline Observability — Diagnostic metrics collection and structured logging.

Implements the DiagnosticCollector class that accumulates per-phase diagnostic
metrics during pipeline execution.

This module acts as a passive observer — it never modifies pipeline behavior.

This module has NO heavy imports (no Qiskit, no PyTorch).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from qmbp_simulation.analysis.metrics import (
    compute_classification_confidence,
    compute_energy_decomposition,
    compute_snr,
    compute_theta_smoothness,
)

logger = logging.getLogger(__name__)

# Minimum layouts for meaningful ZNE extrapolation
MIN_LAYOUTS = 3


def configure_pipeline_logging(
    verbose: bool = False,
    debug: bool = False,
) -> logging.Logger:
    """Configure the 'gnn_hva' logger hierarchy.

    Parameters
    ----------
    verbose : bool
        Sets level to INFO. Shows per-h-point progress, phase transitions.
    debug : bool
        Sets level to DEBUG. Shows per-iteration VQE, per-epoch MPNN.

    Returns
    -------
    logging.Logger
        The configured root logger for the pipeline.
    """
    log = logging.getLogger("gnn_hva")

    # Clear existing handlers to prevent duplicate messages
    log.handlers.clear()

    # Determine level
    if debug:
        log.setLevel(logging.DEBUG)
        fmt = "%(asctime)s [%(levelname)s] %(name)s.%(module)s: %(message)s"
    elif verbose:
        log.setLevel(logging.INFO)
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    else:
        log.setLevel(logging.WARNING)
        fmt = "%(levelname)s: %(message)s"

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    log.addHandler(handler)

    # Prevent propagation to root logger (avoids duplicate output)
    log.propagate = False

    return log


class DiagnosticCollector:
    """Accumulates per-phase diagnostic metrics during pipeline execution.

    Acts as a passive observer — never modifies pipeline behavior. Collects
    timing, convergence, and error metrics per phase, computes derived metrics
    (θ smoothness, SNR, energy decomposition), and serializes to JSON.

    Parameters
    ----------
    verbose : bool
        Whether to emit INFO-level progress messages.
    save_dir : Path | None
        Directory for checkpoint files. If None, no checkpoints are written.
    run_id : str | None
        Unique run identifier (8-char hex). Auto-generated if None.
    """

    def __init__(
        self,
        verbose: bool = False,
        save_dir: Path | None = None,
        run_id: str | None = None,
    ) -> None:
        self.verbose = verbose
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.run_id = (
            run_id
            or hashlib.sha1(
                f"diag:{datetime.now().isoformat()}:{os.getpid()}".encode()
            ).hexdigest()[:8]
        )

        self._log = logging.getLogger("gnn_hva.diagnostics")

        # Phase 1 data
        self._phase1_data: dict | None = None

        # Phase 2 data (accumulated per h-point)
        self._phase2_timing: list[float] = []
        self._phase2_iterations: list[int] = []
        self._phase2_restart_spread: list[float] = []
        self._phase2_converged: list[bool | None] = []
        self._theta_vectors: list[np.ndarray] = []
        self._phase2_h_values: list[float] = []

        # Phase 3 data
        self._loss_curve: list[dict[str, float]] = []
        self._phase3_data: dict | None = None
        self._phase3_elapsed_s: float | None = None

        # Phase 4 data
        self._phase4_data: dict | None = None
        self._phase4_elapsed_s: float | None = None

        # Config dict for checkpoints (set externally if needed)
        self._config_dict: dict = {}

        # Track completed phases
        self._completed_phases: list[str] = []

        # Baseline comparison data
        self._baseline_data: dict | None = None

        # VQE validation report
        self._vqe_validation: dict | None = None

    # ── Phase 1 recording ────────────────────────────────────────────────

    def record_phase1(
        self,
        n_points: int,
        elapsed_s: float,
        gap_min: float,
    ) -> None:
        """Record Phase 1 (exact diagonalization) summary.

        Parameters
        ----------
        n_points : int
            Number of h-points in the sweep.
        elapsed_s : float
            Wall-clock time for Phase 1.
        gap_min : float
            Minimum spectral gap across all h-points.
        """
        self._phase1_data = {
            "n_points": int(n_points),
            "elapsed_s": float(elapsed_s),
            "gap_min": float(gap_min),
        }
        self._completed_phases.append("phase1")
        if self.verbose:
            self._log.info(
                f"Phase 1 complete: {n_points} points, gap_min={gap_min:.4f}, {elapsed_s:.1f}s"
            )

    # ── Phase 2 recording ────────────────────────────────────────────────

    def record_vqe_point(
        self,
        h: float,
        n_iters: int,
        restart_energies: list[float],
        theta_opt: np.ndarray,
        elapsed_s: float,
        *,
        converged: bool | None = None,
    ) -> None:
        """Record VQE diagnostics for a single h-point.

        Parameters
        ----------
        h : float
            Transverse field strength.
        n_iters : int
            Total optimizer iterations for this h-point.
        restart_energies : list[float]
            Final energies from each restart.
        theta_opt : np.ndarray
            Optimal parameter vector, shape (2*p,).
        elapsed_s : float
            Wall-clock time for this h-point.
        converged : bool | None
            Whether the optimizer converged (hit tolerance) vs hit maxiter.
            None if convergence status is unknown.
        """
        try:
            # NaN detection
            if np.any(np.isnan(theta_opt)):
                self._log.error(f"NaN detected in VQE theta_opt at h={h:.4f}")

            self._phase2_timing.append(float(elapsed_s))
            self._phase2_iterations.append(int(n_iters))
            self._phase2_h_values.append(float(h))

            # Restart spread: std of restart energies
            if restart_energies and len(restart_energies) > 1:
                spread = float(np.std(restart_energies))
            else:
                spread = 0.0
            self._phase2_restart_spread.append(spread)

            # Convergence tracking
            self._phase2_converged.append(converged)

            # Accumulate theta vectors for smoothness computation
            self._theta_vectors.append(np.asarray(theta_opt).flatten())

            # Mark phase2 as having data (for checkpoint tracking)
            if "phase2" not in self._completed_phases:
                self._completed_phases.append("phase2")

            if self.verbose:
                conv_str = f", converged={converged}" if converged is not None else ""
                self._log.debug(
                    f"VQE h={h:.3f}: {n_iters} iters, spread={spread:.4f}, "
                    f"{elapsed_s:.2f}s{conv_str}"
                )
        except Exception:
            # Log error with traceback, do not propagate
            self._log.exception(f"Error recording VQE point at h={h}")

    # ── Phase 2 convergence events ──────────────────────────────────────

    def record_vqe_convergence_event(
        self,
        h: float,
        event_type: str,
        *,
        n_restarts_used: int = 0,
        n_restarts_skipped: int = 0,
        wall_clock_s: float = 0.0,
        wall_clock_exceeded: bool = False,
        stagnation_triggered: bool = False,
        method: str = "L-BFGS-B",
        n_params: int = 0,
        maxfev_hit: bool = False,
        best_energy: float | None = None,
    ) -> None:
        """Record VQE convergence/termination events for structured diagnostics.

        Captures information about WHY VQE terminated (convergence, maxfev cap,
        stagnation, wall-clock timeout) for post-hoc debugging of slow runs.

        Parameters
        ----------
        h : float
            Transverse field value.
        event_type : str
            One of: "converged", "maxiter_hit", "maxfev_hit", "stagnation_stop",
            "wall_clock_warning", "nan_detected".
        n_restarts_used : int
            Number of restarts that improved the best energy.
        n_restarts_skipped : int
            Number of restarts skipped due to stagnation detection.
        wall_clock_s : float
            Wall-clock time for this h-point.
        wall_clock_exceeded : bool
            Whether the wall-clock limit was exceeded.
        stagnation_triggered : bool
            Whether stagnation early-stop fired.
        method : str
            Optimizer method used ("L-BFGS-B", "COBYLA", "Nelder-Mead").
        n_params : int
            Number of variational parameters.
        maxfev_hit : bool
            Whether the function evaluation cap was hit.
        best_energy : float | None
            Best energy found at this h-point.
        """
        try:
            event = {
                "h": float(h),
                "event_type": event_type,
                "method": method,
                "n_params": n_params,
                "n_restarts_used": n_restarts_used,
                "n_restarts_skipped": n_restarts_skipped,
                "wall_clock_s": round(wall_clock_s, 2),
                "wall_clock_exceeded": wall_clock_exceeded,
                "stagnation_triggered": stagnation_triggered,
                "maxfev_hit": maxfev_hit,
                "best_energy": float(best_energy) if best_energy is not None else None,
            }

            if not hasattr(self, "_vqe_convergence_events"):
                self._vqe_convergence_events: list[dict] = []
            self._vqe_convergence_events.append(event)

            if self.verbose and event_type != "converged":
                self._log.info(
                    f"VQE event at h={h:.3f}: {event_type} "
                    f"(method={method}, wall_clock={wall_clock_s:.1f}s, "
                    f"restarts_used={n_restarts_used})"
                )
        except Exception:
            self._log.exception(f"Error recording VQE convergence event at h={h}")

    # ── Phase 3 recording ────────────────────────────────────────────────

    def record_mpnn_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None = None,
    ) -> None:
        """Record a single MPNN training epoch.

        Parameters
        ----------
        epoch : int
            Epoch number.
        train_loss : float
            Training loss for this epoch.
        val_loss : float | None
            Validation loss (if available).
        """
        entry: dict[str, float] = {"epoch": epoch, "train_loss": float(train_loss)}
        if val_loss is not None:
            entry["val_loss"] = float(val_loss)
        self._loss_curve.append(entry)

    def record_vqe_validation(self, report: Any) -> None:
        """Record VQE validation report from VQEValidator.

        Parameters
        ----------
        report : VQEValidationReport
            Validation report from VQEValidator.validate_sweep().
        """
        try:
            self._vqe_validation = report.to_dict()
            if self.verbose:
                self._log.info(f"VQE validation recorded: {report.summary()}")
        except Exception:
            self._log.exception("Error recording VQE validation report")

    def record_mpnn_per_h_error(
        self,
        h_values: np.ndarray,
        per_h_mse: np.ndarray,
        *,
        elapsed_s: float | None = None,
    ) -> None:
        """Record per-h-point MPNN prediction error after training.

        Parameters
        ----------
        h_values : np.ndarray
            Array of h-values.
        per_h_mse : np.ndarray
            Per-h-point MSE values (same length as h_values).
        elapsed_s : float | None
            Wall-clock time for Phase 3 training.
        """
        h_arr = np.asarray(h_values)
        mse_arr = np.asarray(per_h_mse)

        # Store per-h MSE as dict (h-value string → MSE float)
        per_h_mse_dict: dict[str, float] = {}
        for h_val, mse_val in zip(h_arr, mse_arr, strict=False):
            per_h_mse_dict[f"{float(h_val):.4f}"] = float(mse_val)

        # Compute theta_zz_mse and theta_x_mse
        if mse_arr.ndim == 2:
            p = mse_arr.shape[1] // 2
            theta_zz_mse = float(np.mean(mse_arr[:, :p]))
            theta_x_mse = float(np.mean(mse_arr[:, p:]))
        else:
            theta_zz_mse = float(np.mean(mse_arr))
            theta_x_mse = float(np.mean(mse_arr))

        # Generalization gap: difference between max and min per-h MSE
        generalization_gap = float(np.max(mse_arr) - np.min(mse_arr)) if len(mse_arr) > 1 else None

        # Loss curve last 100 entries
        loss_curve_last100 = [entry["train_loss"] for entry in self._loss_curve[-100:]]

        self._phase3_data = {
            "per_h_mse": per_h_mse_dict,
            "theta_zz_mse": theta_zz_mse,
            "theta_x_mse": theta_x_mse,
            "generalization_gap": generalization_gap,
            "loss_curve_last100": loss_curve_last100,
            "elapsed_s": float(elapsed_s) if elapsed_s is not None else None,
        }
        if elapsed_s is not None:
            self._phase3_elapsed_s = float(elapsed_s)
        if "phase3" not in self._completed_phases:
            self._completed_phases.append("phase3")

        if self.verbose:
            self._log.info(
                f"Phase 3 MSE: θ_zz={theta_zz_mse:.2e}, θ_x={theta_x_mse:.2e}, "
                f"gap={generalization_gap}"
            )

    # ── Phase 4 recording ────────────────────────────────────────────────

    def record_deployment(
        self,
        h_test: float,
        result,
        per_layout_data: dict | None = None,
        *,
        total_shots: int | None = None,
        e_vqe_ceiling: float | None = None,
    ) -> None:
        """Record Phase 4 deployment diagnostics.

        Computes SNR, classification confidence, per-layout metrics,
        CES-energy Pearson correlation, and energy decomposition.

        Parameters
        ----------
        h_test : float
            Test h-value for deployment.
        result : DeployResult
            Deployment result dataclass with predicted_energy, delta_e,
            mag_x_pred, corr_zz_pred, etc.
        per_layout_data : dict | None
            Dict with keys 'energies' and 'ces_values' (lists of floats),
            or None if not available.
        total_shots : int | None
            Total measurement shots used. Required for SNR computation.
            If None, SNR metrics will be None.
        e_vqe_ceiling : float | None
            Best achievable VQE energy E(θ_opt) for energy decomposition.
            If None, energy decomposition uses predicted_energy as ceiling
            (making error_from_mpnn = 0).
        """
        try:
            # Borderline metric warning
            if hasattr(result, "delta_e_over_gap") and result.delta_e_over_gap is not None:
                de_gap = result.delta_e_over_gap
                if 0.04 <= de_gap < 0.05:
                    self._log.warning(f"Borderline metric: ΔE/gap = {de_gap:.4f} (threshold: 5%)")

            # NaN detection in deployment results
            if (
                hasattr(result, "predicted_energy")
                and result.predicted_energy is not None
                and np.isnan(result.predicted_energy)
            ):
                self._log.error(f"NaN detected in predicted energy at h_test={h_test}")

            # SNR computation (requires explicit total_shots)
            shots = total_shots or getattr(result, "total_shots", None)
            snr_mag_x: float | None = None
            snr_corr_zz: float | None = None
            classification_conf: float | None = None

            if shots is not None and shots > 0:
                snr_mag_x = compute_snr(result.mag_x_pred, shots)
                snr_corr_zz = compute_snr(result.corr_zz_pred, shots)
                classification_conf = compute_classification_confidence(
                    result.mag_x_pred, result.corr_zz_pred, shots
                )
            else:
                self._log.debug(
                    f"No shot count available for h_test={h_test}; SNR metrics skipped."
                )

            # Per-layout data
            per_layout_energies: list[float] = []
            per_layout_ces: list[float] = []
            ces_energy_pearson_r: float | None = None

            if per_layout_data is not None:
                energies = per_layout_data.get("energies", [])
                ces_vals = per_layout_data.get("ces_values", [])
                per_layout_energies = [float(e) for e in energies]
                per_layout_ces = [float(c) for c in ces_vals]

                # CES-energy Pearson correlation
                if (
                    len(per_layout_energies) >= MIN_LAYOUTS
                    and len(per_layout_ces) >= MIN_LAYOUTS
                    and len(per_layout_energies) == len(per_layout_ces)
                ):
                    from scipy.stats import pearsonr

                    r_val, _ = pearsonr(per_layout_energies, per_layout_ces)
                    ces_energy_pearson_r = float(r_val)

            # Energy decomposition
            e_predicted = result.predicted_energy
            e_exact = e_predicted - result.delta_e

            # Use explicit ceiling if provided, otherwise fall back
            ceiling = e_vqe_ceiling if e_vqe_ceiling is not None else e_predicted

            energy_decomp = compute_energy_decomposition(e_exact, ceiling, e_predicted)

            self._phase4_data = {
                "h_test": float(h_test),
                "snr_mag_x": snr_mag_x,
                "snr_corr_zz": snr_corr_zz,
                "classification_confidence": classification_conf,
                "per_layout_energies": per_layout_energies,
                "per_layout_ces": per_layout_ces,
                "ces_energy_pearson_r": ces_energy_pearson_r,
                "energy_decomposition": energy_decomp,
                "total_shots": shots,
            }

        except Exception:
            # Log error with traceback, set metrics to None, do not propagate
            self._log.exception(f"Error computing Phase 4 diagnostics at h_test={h_test}")
            self._phase4_data = {
                "h_test": float(h_test),
                "snr_mag_x": None,
                "snr_corr_zz": None,
                "classification_confidence": None,
                "per_layout_energies": [],
                "per_layout_ces": [],
                "ces_energy_pearson_r": None,
                "energy_decomposition": None,
                "total_shots": None,
            }

        if "phase4" not in self._completed_phases:
            self._completed_phases.append("phase4")

        if self.verbose:
            snr_mx = self._phase4_data.get("snr_mag_x")
            snr_cz = self._phase4_data.get("snr_corr_zz")
            snr_mx_str = f"{snr_mx:.1f}" if snr_mx is not None else "N/A"
            snr_cz_str = f"{snr_cz:.1f}" if snr_cz is not None else "N/A"
            self._log.info(
                f"Phase 4 deployment h={h_test}: SNR(mag_x)={snr_mx_str}, SNR(corr_zz)={snr_cz_str}"
            )

    # ── Baseline comparison recording ────────────────────────────────────

    def record_baseline(
        self,
        h_test: float,
        comparison,
    ) -> None:
        """Record baseline comparison diagnostics (warm-start vs cold-start).

        Parameters
        ----------
        h_test : float
            Test h-value.
        comparison : BaselineComparison
            Comparison object with warm vs cold metrics.
        """
        try:
            self._baseline_data = {
                "h_test": float(h_test),
                "n_random_seeds": comparison.n_random_seeds,
                "random_seeds": comparison.random_seeds,
                "gain_energy_pct": float(comparison.gain_energy_pct),
                "gain_fidelity_abs": (
                    float(comparison.gain_fidelity_abs)
                    if comparison.gain_fidelity_abs is not None
                    else None
                ),
                "warm_start_sufficient": comparison.warm_start_sufficient,
                "cold_start_any_success": comparison.cold_start_any_success,
                "warm_delta_e_over_gap": float(comparison.warm_start.delta_e_over_gap),
                "cold_mean_delta_e_over_gap": float(comparison.cold_start_mean["delta_e_over_gap"]),
                "cold_std_delta_e_over_gap": float(comparison.cold_start_std["delta_e_over_gap"]),
                "anomaly_detected": comparison.gain_energy_pct < 0,
            }

            if self.verbose:
                self._log.info(
                    f"Baseline h={h_test}: warm ΔE/gap="
                    f"{comparison.warm_start.delta_e_over_gap:.4f}, "
                    f"cold mean={comparison.cold_start_mean['delta_e_over_gap']:.4f}±"
                    f"{comparison.cold_start_std['delta_e_over_gap']:.4f}, "
                    f"gain={comparison.gain_energy_pct:.1f}%"
                )
        except Exception:
            self._log.exception(f"Error recording baseline comparison at h_test={h_test}")
            self._baseline_data = {"error": "see logs for traceback"}

    # ── Theta Validation recording ───────────────────────────────────────

    def record_theta_validation(
        self,
        h_test: float,
        report_dict: dict,
    ) -> None:
        """Record θ_pred validation diagnostics from ThetaValidator.

        Parameters
        ----------
        h_test : float
            Test h-value where prediction was validated.
        report_dict : dict
            Output of ThetaValidationReport.to_dict().
        """
        if not hasattr(self, "_theta_validation_data"):
            self._theta_validation_data: list[dict] = []

        entry = {"h_test": float(h_test), **report_dict}
        self._theta_validation_data.append(entry)

        if self.verbose:
            status = "PASS" if report_dict.get("overall_pass") else "FAIL"
            conf = report_dict.get("confidence_score", 0.0)
            self._log.info(
                f"θ validation h={h_test}: {status} "
                f"(confidence={conf:.2f}, level=L{report_dict.get('level_executed', '?')})"
            )

    # ── Serialization ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize all collected diagnostics to a JSON-compatible dict.

        All numpy arrays are converted to Python lists and all numpy scalars
        to Python floats.

        Returns
        -------
        dict
            Structure: {"phase1": {...}, "phase2": {...}, "phase3": {...}, "phase4": {...}}
        """
        # Phase 2: compute derived metrics
        theta_smoothness = None
        worst_convergence_h = None

        if len(self._theta_vectors) >= 2:
            theta_array = np.array(self._theta_vectors)
            theta_smoothness = compute_theta_smoothness(theta_array)

        if self._phase2_iterations and self._phase2_h_values:
            worst_idx = int(np.argmax(self._phase2_iterations))
            worst_convergence_h = self._phase2_h_values[worst_idx]

        phase2 = {
            "per_h_timing_s": _to_json_safe(self._phase2_timing),
            "per_h_iterations": _to_json_safe(self._phase2_iterations),
            "per_h_restart_spread": _to_json_safe(self._phase2_restart_spread),
            "per_h_converged": _to_json_safe(self._phase2_converged),
            "theta_smoothness": _to_json_safe(theta_smoothness),
            "worst_convergence_h": _to_json_safe(worst_convergence_h),
            "total_elapsed_s": _to_json_safe(sum(self._phase2_timing))
            if self._phase2_timing
            else None,
            "convergence_rate": _to_json_safe(
                sum(1 for c in self._phase2_converged if c is True) / len(self._phase2_converged)
            )
            if self._phase2_converged and any(c is not None for c in self._phase2_converged)
            else None,
        }

        # Phase 3
        phase3 = (
            _to_json_safe(self._phase3_data)
            if self._phase3_data
            else {
                "per_h_mse": {},
                "theta_zz_mse": None,
                "theta_x_mse": None,
                "generalization_gap": None,
                "loss_curve_last100": [],
            }
        )

        # Phase 4
        phase4 = (
            _to_json_safe(self._phase4_data)
            if self._phase4_data
            else {
                "snr_mag_x": None,
                "snr_corr_zz": None,
                "classification_confidence": None,
                "per_layout_energies": [],
                "per_layout_ces": [],
                "ces_energy_pearson_r": None,
                "energy_decomposition": None,
            }
        )

        return {
            "phase1": _to_json_safe(self._phase1_data),
            "phase2": phase2,
            "phase3": phase3,
            "phase4": phase4,
            "baseline_comparison": _to_json_safe(self._baseline_data),
            "vqe_validation": _to_json_safe(self._vqe_validation),
            "theta_validation": _to_json_safe(getattr(self, "_theta_validation_data", None)),
            "vqe_convergence_events": _to_json_safe(getattr(self, "_vqe_convergence_events", [])),
        }

    # ── Checkpoint persistence ───────────────────────────────────────────

    def save_checkpoint(self, phase: str) -> Path | None:
        """Write incremental checkpoint after a phase completes.

        Parameters
        ----------
        phase : str
            Phase identifier (e.g., "phase1", "phase2", "phase3", "phase4").

        Returns
        -------
        Path | None
            Path to the written checkpoint file, or None if save_dir is None.
        """
        if self.save_dir is None:
            return None

        checkpoint_data = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "config": self._config_dict,
            "completed_phases": self._completed_phases[:],
            "diagnostics_so_far": self.to_dict(),
        }

        path = self.save_dir / f"checkpoint_{self.run_id}_{phase}.json"

        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            self._log.info(f"Checkpoint saved: {path.name}")
            return path
        except OSError as e:
            self._log.warning(f"Failed to write checkpoint {path.name}: {e}")
            return None

    def cleanup_checkpoints(self) -> None:
        """Delete all checkpoint files for this run_id on successful completion."""
        if self.save_dir is None:
            return

        for phase in ("phase1", "phase2", "phase3", "phase4"):
            path = self.save_dir / f"checkpoint_{self.run_id}_{phase}.json"
            try:
                if path.exists():
                    path.unlink()
                    self._log.debug(f"Removed checkpoint: {path.name}")
            except OSError as e:
                self._log.warning(f"Failed to remove checkpoint {path.name}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# JSON serialization helpers
# ─────────────────────────────────────────────────────────────────────────────


def _to_json_safe(obj):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if obj is None:
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_json_safe(item) for item in obj]
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, int | str | bool):
        return obj
    # Fallback: try to convert to float/int
    try:
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    except (TypeError, ValueError):
        return str(obj)
