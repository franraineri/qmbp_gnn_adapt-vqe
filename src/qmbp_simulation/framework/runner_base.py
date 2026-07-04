"""Standardized runner base classes for experiment and validation scripts.

Provides three runner templates that enforce:
1. Mandatory preflight validation before execution.
2. Structured logging via StructuredLogger + ProgressReporter.
3. Standardized result saving via result_io (build_result_envelope + save_experiment_result).
4. Consistent CLI argument handling.
5. Error capture with traceback and graceful degradation.
6. Proper exit codes (non-zero on failure).

Runner Types:
    - ExperimentRunner: Wraps BaseExperiment subclasses (simple lifecycle).
    - ValidationRunner: Multi-section validation suites (complex, table-driven).
    - VariantPipelineRunner: Thin wrapper over existing VariantRunner infrastructure.
    - HardwareValidationRunner: ValidationRunner + HardwareBackend integration.

MPNN Evaluation Helpers (ValidationRunner instance methods — reusable by any subclass):
    Sections 10-13 (basic):
    - benchmark_mpnn_warmstart(): Compare MPNN warm-start vs random vs prev-h init.
    - mpnn_leave_one_out_cv(): LOO cross-validation for generalization estimate.
    - mpnn_landscape_quality(): Decompose ΔE into circuit vs ML error components.
    - mpnn_interpolation_extrapolation(): Measure accuracy inside vs outside h_train range.

    Sections 15-19 (extended):
    - mpnn_scaling_with_system_size(): Warm-start speedup as a function of N (p_layers_per_n supported).
    - mpnn_learning_curve(): ΔE/gap vs training set size (sample efficiency curve).
    - mpnn_topology_transfer(): Zero-shot cross-topology prediction test.
    - mpnn_data_efficiency_vs_loo(): Multi-seed LOO to quantify stability.
    - mpnn_curvature_noise_correlation(): κ(h) vs noise sensitivity — hardware risk proxy.

    Hardware deployment utilities:
    - compute_kappa_per_h(): Landscape curvature κ(h) via finite differences (noiseless).
    - kappa_go_no_go(): Per-h deployment recommendations from κ profile.

    All helpers are topology/model-agnostic and topology-aware (kappa_go_no_go
    auto-calibrates percentile-based thresholds for any topology).

    Runner IDs:
        HW_REHEARSAL_V3 — MPNN Evaluation Suite (sections 10-19)
        Results saved to: results/experiments/exp_hw_rehearsal_v3/

Usage (ValidationRunner — most common for new scripts):

    from qmbp_simulation.framework.runner_base import ValidationRunner, Section

    class MyValidation(ValidationRunner):
        runner_id = "E4b_hw"
        experiment_id = "E4b"
        description = "E4b Hardware Readiness Suite"
        hypothesis = "TFIM+longitudinal behaves like standard TFIM on hardware"

        def define_sections(self) -> list[Section]:
            return [
                Section(id=1, name="ZNE Noisy Simulation", fn=self.section_zne),
                Section(id=2, name="Theta Smoothness", fn=self.section_smoothness),
            ]

        def section_zne(self) -> dict:
            ...
            return {"mean_r2": 0.99, "pass": True}

        def section_smoothness(self) -> dict:
            ...
            return {"max_smoothness": 0.4, "pass": True}

    if __name__ == "__main__":
        MyValidation.main()
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qmbp_simulation.framework.logging import ProgressReporter, StructuredLogger
from qmbp_simulation.framework.result_io import (
    build_result_envelope,
    collect_run_metadata,
    save_experiment_result,
)

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Section:
    """A single section of a validation runner.

    Parameters
    ----------
    id : int
        Section number (for ordering and display).
    name : str
        Human-readable section name.
    fn : Callable[[], dict]
        Function that executes the section and returns a result dict.
        MUST return a dict. Include key "pass": bool to signal pass/fail.
    hypothesis : str
        What this section is testing (optional, displayed in header).
    """

    id: int
    name: str
    fn: Callable[[], dict[str, Any]]
    hypothesis: str = ""


@dataclass
class SectionResult:
    """Result from executing a single section."""

    section_id: int
    name: str
    success: bool
    elapsed_s: float
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Utility: project root resolver
# ═══════════════════════════════════════════════════════════════════════════════


def resolve_project_root(script_path: str | Path) -> Path:
    """Resolve the project root from any script location.

    Walks up from script_path looking for pyproject.toml or Makefile.
    Falls back to two levels up if not found.

    Parameters
    ----------
    script_path : str | Path
        __file__ of the calling script.

    Returns
    -------
    Path
        Absolute path to the project root.
    """
    current = Path(script_path).resolve().parent
    for _ in range(5):  # Max 5 levels up
        if (current / "pyproject.toml").exists() or (current / "Makefile").exists():
            return current
        current = current.parent
    # Fallback: assume scripts are at most 2 levels deep
    return Path(script_path).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════════════
# ValidationRunner — Multi-section validation scripts
# ═══════════════════════════════════════════════════════════════════════════════


class ValidationRunner(ABC):
    """Base class for multi-section validation/testing runners.

    Enforces:
    - Preflight validation before any section runs.
    - Structured event logging for post-hoc analysis (saved independently).
    - ProgressReporter for interactive console feedback.
    - Standardized JSON result saving to results/experiments/exp_{id}/.
    - Per-section error isolation (one failure doesn't abort the rest).
    - CLI with --section filtering, --dry-run, and --stop-on-failure support.
    - Non-zero exit code when sections fail.

    Subclasses must define:
    - runner_id: Unique runner identifier (used in logs/filenames).
    - experiment_id: Experiment ID for result_io (e.g., "E4b").
    - description: One-line description of what this runner validates.
    - hypothesis: Overall hypothesis being tested.
    - define_sections(): Returns the list of Section objects.

    Subclasses may override:
    - build_config(): Returns the config dict for the result envelope.
    - run_preflight(): Custom preflight checks (default validates structure).
    - setup(): One-time setup before sections run (import heavy deps, build objects).
    """

    # ── Subclass must define these ───────────────────────────────────────────
    runner_id: str = ""
    experiment_id: str = ""
    description: str = ""
    hypothesis: str = ""

    def __init__(self, args: argparse.Namespace | None = None):
        self._args = args or self._parse_args()
        self._setup_logging()
        self.slog = StructuredLogger(self.runner_id or "unknown")
        self.reporter = ProgressReporter(self.runner_id or "unknown")
        self._section_results: list[SectionResult] = []
        # Cache sections once (avoid calling define_sections() multiple times)
        self._sections_cache: list[Section] | None = None

    # ── Abstract interface ───────────────────────────────────────────────────

    @abstractmethod
    def define_sections(self) -> list[Section]:
        """Define all sections this runner executes.

        Returns
        -------
        list[Section]
            Ordered list of sections to run.

        Important
        ---------
        Each section's fn MUST return a dict. Include "pass": False to
        explicitly signal failure. If "pass" key is absent, success is assumed
        (unless an exception is raised).
        """
        ...

    # ── Overridable hooks ────────────────────────────────────────────────────

    def build_config(self) -> dict[str, Any]:
        """Build the configuration dict for the result envelope.

        Override to include experiment-specific config (system params, seeds, etc.).
        Default returns basic runner metadata.
        """
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "description": self.description,
            "hypothesis": self.hypothesis,
        }

    def run_preflight(self) -> bool:
        """Run preflight validation before execution.

        Returns True if preflight passes, False otherwise.
        Override for custom validation logic. Always call super() first
        when overriding to retain structural checks.

        Default checks:
        - runner_id, experiment_id, description, hypothesis are set.
        - define_sections() returns non-empty list.
        - No duplicate section IDs.
        - Physics constraints from build_config() (p≤2, expressibility limits).
        """
        logger.info("Preflight: validating runner configuration...")
        self.slog.log("preflight_start")

        errors: list[str] = []
        warnings: list[str] = []

        # Structural checks
        if not self.runner_id:
            errors.append("runner_id is not set")
        if not self.experiment_id:
            errors.append("experiment_id is not set")
        if not self.description:
            errors.append("description is not set")
        if not self.hypothesis:
            errors.append("hypothesis is not set")

        sections = self._get_sections()
        if not sections:
            errors.append("define_sections() returned empty list")
        else:
            # Check for duplicate IDs
            ids = [s.id for s in sections]
            if len(ids) != len(set(ids)):
                duplicates = [i for i in ids if ids.count(i) > 1]
                errors.append(f"Duplicate section IDs: {set(duplicates)}")

            # Warn if any section lacks hypothesis
            for s in sections:
                if not s.hypothesis:
                    warnings.append(f"Section {s.id} ({s.name}) has no hypothesis")

        # Physics-aware check: validate h-values vs topology regime
        warnings.extend(self._check_regime_warnings())

        # Physics-aware check: expressibility and grid constraints
        phys_warnings, phys_errors = self._check_physics_constraints()
        warnings.extend(phys_warnings)
        errors.extend(phys_errors)

        # Report
        for w in warnings:
            logger.warning(f"  Preflight WARNING: {w}")
        if errors:
            for e in errors:
                logger.error(f"  Preflight ERROR: {e}")
            self.slog.log("preflight_failed", data={"errors": errors})
            return False

        self.slog.log("preflight_passed", data={"n_sections": len(sections)})
        logger.info(f"  Preflight PASSED ({len(sections)} sections defined)")
        return True

    def setup(self) -> None:
        """One-time setup before sections execute.

        Override to import heavy dependencies, build shared objects, etc.
        Called after preflight passes but before any section runs.
        """

    def _check_regime_warnings(self) -> list[str]:
        """Check config h-values against known valid regime boundaries.

        Returns a list of warning strings (empty if everything is within regime).
        This is a best-effort check: it inspects build_config() for h-values
        and validates them against the topology-specific boundaries from
        preflight.py. Only produces WARNINGS (never errors) because:
        - Some runners intentionally test outside the valid regime
        - The regime map may not cover all (topology, N) combinations

        Detects h-values from config keys: 'h_train', 'h_test', 'h_values',
        'h_values_sweep', and nested 'system.h_values'.
        """
        warnings: list[str] = []
        try:
            config = self.build_config()
        except Exception:
            return warnings  # Can't check if build_config() fails

        # Extract topology and n_qubits from config
        system = config.get("system", {})
        topology = system.get("topology") or config.get("topology", "")
        n_qubits = system.get("n_qubits") or config.get("n_qubits", 0)
        p_layers = system.get("p_layers") or config.get("p_layers", 1)

        if not topology or not n_qubits:
            return warnings  # Not enough info to check

        # Get regime threshold
        try:
            from qmbp_simulation.framework.preflight import get_regime_threshold

            threshold = get_regime_threshold(topology, n_qubits, p_layers)
        except (ImportError, ValueError):
            return warnings  # Can't load preflight module or invalid p

        if threshold == 0.0:
            return warnings  # No threshold defined for this config

        # Collect all h-values from config
        h_keys = ["h_train", "h_test", "h_values", "h_values_sweep"]
        all_h: list[float] = []
        for key in h_keys:
            val = config.get(key) or system.get(key)
            if isinstance(val, (list, tuple)):
                all_h.extend(float(v) for v in val if isinstance(v, (int, float)))

        # Check for h-values below the valid regime
        below_regime = [h for h in all_h if h < threshold]
        if below_regime:
            warnings.append(
                f"h-values {below_regime} are below valid regime boundary "
                f"({threshold}) for {topology} N={n_qubits} p={p_layers}. "
                f"VQE may not converge well at these points."
            )

        return warnings

    def _check_physics_constraints(self) -> tuple[list[str], list[str]]:
        """Check physics-aware constraints derived from known failure modes.

        Returns (warnings, errors) — errors block execution.

        Checks performed:
        1. p > 2 violation (hard error).
        2. p=1 expressibility: warn if pass criteria assume ΔE/gap < 5%
           at h-values below the valid regime.
        3. Grid density: warn if 2D parameter space has fewer than 8 points
           per dimension (interpolation will fail).
        4. Model constraint: Heisenberg/XY at p≤2 is forbidden.
        5. ZNE budget: p=2 N≥10 exceeds ZNE threshold (36 CX > 18 CX limit).

        Note: build_config() may use attributes set in setup() which runs AFTER
        preflight. We use a try/except and fall back to partial checking.
        """
        warnings: list[str] = []
        errors: list[str] = []

        # Try build_config() — may fail if it depends on setup() state
        config: dict[str, Any] = {}
        try:
            config = self.build_config()
        except Exception:
            pass  # Check what we can without full config

        system = config.get("system", {})
        p_layers = system.get("p_layers") or config.get("p_layers", 0)
        n_qubits = system.get("n_qubits") or config.get("n_qubits", 0)
        model = config.get("model", "")

        # 1. p > 2 hard constraint
        if p_layers > 2:
            errors.append(
                f"p_layers={p_layers} > 2 violates HVA depth constraint "
                f"(Mele et al. 2022). Max allowed: p=2."
            )

        # 2. p=1 expressibility warning
        if p_layers == 1 and n_qubits >= 6:
            warnings.append(
                "p=1 has limited expressibility: ΔE/gap > 5% at low h is "
                "expected (not a VQE failure). Ensure section pass criteria "
                "account for this limit (use ΔE/gap < 50% for convergence, "
                "not < 1%)."
            )

        # 3. Grid density for 2D parameter spaces
        grid_config = config.get("grid", {})
        if grid_config:
            h_train = grid_config.get("h_train", [])
            j2_train = grid_config.get("j2_train", [])
            if h_train and j2_train:
                n_h = len(h_train)
                n_j2 = len(j2_train)
                total = n_h * n_j2
                if n_j2 < 8 and total < 80:
                    warnings.append(
                        f"2D grid has {n_j2} J₂ values × {n_h} h values = "
                        f"{total} points. For reliable J₂ interpolation, "
                        f"consider ≥8 J₂ values or ≥80 total training points."
                    )

        # 4. Forbidden model+ansatz
        if model in ("heisenberg", "xy") and p_layers <= 2:
            errors.append(
                f"Model '{model}' with HVA p≤{p_layers} is known to fail "
                f"(V9: 30 runs, 0% fidelity). Do not attempt."
            )

        # 5. ZNE budget check
        zne_config = config.get("zne", {})
        if zne_config and p_layers == 2 and n_qubits >= 10:
            warnings.append(
                f"ZNE at p=2 N={n_qubits} exceeds the 18 CX gate budget "
                f"(~36 CX). ZNE extrapolation will be unreliable. "
                f"Use p=1 for N≥10 hardware/noisy simulation."
            )

        return warnings, errors

    # ── Execution engine ─────────────────────────────────────────────────────

    def run(self) -> int:
        """Execute the full runner lifecycle.

        Lifecycle:
        1. Preflight validation → abort if fails.
        2. setup() → one-time initialization.
        3. For each section: execute with timing and error capture.
        4. Save results + structured log to JSON.
        5. Print summary.

        Handles KeyboardInterrupt (Ctrl+C) and SIGTERM (kill) gracefully:
        partial results from completed sections are always saved.

        Returns
        -------
        int
            Exit code: 0 if all sections pass, 1 otherwise.
        """
        import signal

        t_total = time.time()

        # Register SIGTERM handler to convert to KeyboardInterrupt
        # (so nohup kills and systemd stops trigger the same save logic)
        _original_sigterm = signal.getsignal(signal.SIGTERM)

        def _sigterm_handler(signum, frame):
            logger.warning("SIGTERM received — triggering graceful shutdown...")
            raise KeyboardInterrupt("SIGTERM received")

        signal.signal(signal.SIGTERM, _sigterm_handler)

        # ─── Step 1: Preflight ───────────────────────────────────────────
        if not self._args.skip_preflight:
            if not self.run_preflight():
                logger.error("Preflight FAILED. Aborting execution.")
                logger.error("Use --skip-preflight to bypass (not recommended).")
                return 1
        else:
            logger.warning("Preflight SKIPPED (--skip-preflight flag)")

        # ─── Step 1b: Dry-run ────────────────────────────────────────────
        sections = self._get_sections()
        selected = self._filter_sections(sections)

        if self._args.dry_run:
            self._print_dry_run(selected)
            return 0

        # ─── Step 2: Setup ───────────────────────────────────────────────
        self.slog.log("setup_start")
        try:
            self.setup()
        except Exception as e:
            logger.error(f"Setup FAILED: {e}")
            self.slog.log("setup_failed", data={"error": str(e)})
            return 1
        self.slog.log("setup_complete")

        # ─── Step 2b: Resume from previous run ───────────────────────────
        _resumed_sections: set[int] = set()
        if getattr(self._args, "resume", None):
            _resumed_sections = self._load_resume(self._args.resume)

        # ─── Step 3: Execute sections ────────────────────────────────────
        self._print_header(selected)
        interrupted = False
        _current_section_id = None

        try:
            for section in selected:
                # Skip sections already completed in the resumed run
                if section.id in _resumed_sections:
                    logger.info(
                        f"  ⏭️  Skipping Section {section.id} ({section.name}) — "
                        f"already completed in resumed run."
                    )
                    continue

                _current_section_id = section.id
                result = self._execute_section(section)
                self._section_results.append(result)
                _current_section_id = None  # Section completed successfully

                # Stop-on-failure: abort remaining sections
                if not result.success and self._args.stop_on_failure:
                    logger.warning(
                        f"  Stopping early (--stop-on-failure). Section {section.id} failed."
                    )
                    break

        except KeyboardInterrupt:
            interrupted = True
            msg = (
                f"\n  ⚠️  INTERRUPTED (Ctrl+C) during section {_current_section_id}. "
                f"Saving partial results..."
            )
            logger.warning(msg)
            self.slog.log(
                "interrupted",
                data={
                    "completed_sections": len(self._section_results),
                    "interrupted_section": _current_section_id,
                },
            )

        # ─── Step 4: Save results ────────────────────────────────────────
        # Always save — even on interrupt, so partial results are preserved.
        # Wrap in try/except so a serialization failure doesn't lose the log.
        total_elapsed = time.time() - t_total
        saved_path = None
        try:
            envelope = self._build_envelope(total_elapsed)
            if interrupted:
                envelope["interrupted"] = True
                envelope["completed_sections"] = len(self._section_results)
                envelope["interrupted_section"] = _current_section_id
            saved_path = save_experiment_result(envelope, experiment_id=self.experiment_id)
        except Exception as save_exc:
            logger.error(
                f"Failed to save results: {save_exc}. "
                f"Completed sections: {len(self._section_results)}."
            )

        # Save structured log independently (even if result save failed)
        try:
            if saved_path is not None:
                log_path = saved_path.parent / f"log_{saved_path.stem.replace('run_', '')}.json"
            else:
                # Fallback: save log to experiment dir root
                from qmbp_simulation.framework.result_io import generate_timestamp
                log_path = (
                    Path("results") / "experiments" / f"exp_{self.experiment_id}"
                    / f"log_emergency_{generate_timestamp()}.json"
                )
                log_path.parent.mkdir(parents=True, exist_ok=True)
            self.slog.save(log_path)
        except Exception as log_exc:
            logger.error(f"Failed to save structured log: {log_exc}")

        # ─── Step 4b: Auto-refresh project status ────────────────────────
        # Keep Kiro's steering context up-to-date after every run.
        if saved_path is not None:
            try:
                from qmbp_simulation.framework.result_index import ResultIndex
                ResultIndex().refresh_status()
            except Exception:
                pass  # Best-effort, never blocks run completion

        # ─── Step 5: Summary ─────────────────────────────────────────────
        if interrupted:
            logger.info(
                f"\n  Partial results saved: {saved_path}"
                f"\n  Completed {len(self._section_results)}/{len(selected)} sections."
            )
        elif saved_path is not None:
            self._print_summary(total_elapsed, saved_path)

        n_fail = sum(1 for r in self._section_results if not r.success)

        # Restore original SIGTERM handler
        signal.signal(signal.SIGTERM, _original_sigterm)

        return 1 if n_fail > 0 or interrupted else 0

    # ── Class-level entry point ──────────────────────────────────────────────

    @classmethod
    def main(cls) -> None:
        """Standard entry point for runner scripts.

        Sets the process exit code based on section results.

        Usage in scripts:
            if __name__ == "__main__":
                MyRunner.main()
        """
        runner = cls()
        exit_code = runner.run()
        sys.exit(exit_code)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _get_sections(self) -> list[Section]:
        """Get sections (cached to avoid multiple define_sections() calls)."""
        if self._sections_cache is None:
            self._sections_cache = self.define_sections()
        return self._sections_cache

    def _execute_section(self, section: Section) -> SectionResult:
        """Execute a single section with timing and error capture."""
        self.slog.log(
            "section_start",
            data={"section_id": section.id, "name": section.name},
        )
        self.slog.start_timer(f"section_{section.id}")

        logger.info("")
        logger.info("=" * 65)
        logger.info(f"SECTION {section.id}: {section.name}")
        if section.hypothesis:
            logger.info(f"  Hypothesis: {section.hypothesis}")
        logger.info("=" * 65)

        t0 = time.time()
        try:
            data = section.fn()
            elapsed = time.time() - t0

            # Validate return type
            if data is None:
                logger.warning(
                    f"  Section {section.id} returned None instead of dict. "
                    f"Treating as empty result."
                )
                data = {}
            elif not isinstance(data, dict):
                logger.warning(
                    f"  Section {section.id} returned {type(data).__name__} "
                    f"instead of dict. Wrapping as {{'value': ...}}."
                )
                data = {"value": data}

            # Determine success: explicit "pass" key takes priority
            if "pass" in data:
                success = bool(data["pass"])
            else:
                success = True

            error = None

            self.slog.stop_timer(
                f"section_{section.id}",
                event_type="section_complete",
                data={
                    "section_id": section.id,
                    "success": success,
                    "elapsed_s": round(elapsed, 3),
                },
            )
            status = "✅ PASS" if success else "⚠️  FAIL"
            logger.info(f"\n  {status} — Section {section.id} ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            success = False
            error = f"{type(e).__name__}: {e}"
            data = {}

            tb_lines = traceback.format_exc().split("\n")
            # Keep last 6 lines for useful context
            short_tb = "\n".join(tb_lines[-6:])

            self.slog.log(
                "section_failed",
                data={
                    "section_id": section.id,
                    "error": error,
                    "traceback": short_tb,
                },
            )
            logger.error(f"\n  ❌ EXCEPTION in Section {section.id} ({elapsed:.1f}s)")
            logger.error(f"     {error}")
            if self._args.verbose:
                logger.error(f"     {short_tb}")

        return SectionResult(
            section_id=section.id,
            name=section.name,
            success=success,
            elapsed_s=elapsed,
            data=data,
            error=error,
        )

    def _load_resume(self, resume_path: str) -> set[int]:
        """Load a previous result file and restore completed section data.

        Reads the result JSON, identifies which sections completed successfully,
        injects their results into self._section_results, and calls the
        overridable hook `restore_section_state()` so subclasses can restore
        internal state (e.g., VQE theta_opt, ground truth data).

        Parameters
        ----------
        resume_path : str
            Path to the previous run_*.json to resume from.

        Returns
        -------
        set[int]
            Section IDs that were successfully loaded and should be skipped.
        """
        from qmbp_simulation.framework.result_io import load_result

        path = Path(resume_path)
        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists():
            logger.error(f"Resume file not found: {path}")
            return set()

        try:
            data = load_result(path)
        except (ValueError, FileNotFoundError) as e:
            logger.error(f"Cannot load resume file: {e}")
            return set()

        results = data.get("results", {})
        resumed_ids: set[int] = set()

        for key, section_data in results.items():
            if not key.startswith("section_"):
                continue
            if not section_data.get("success", False):
                continue  # Only restore sections that passed

            # Extract section_id from key "section_N"
            try:
                section_id = int(key.split("_")[1])
            except (IndexError, ValueError):
                continue

            # Create a SectionResult for the previously completed section
            self._section_results.append(SectionResult(
                section_id=section_id,
                name=section_data.get("name", key),
                success=True,
                elapsed_s=section_data.get("elapsed_s", 0),
                data=section_data.get("data", {}),
                error=None,
            ))
            resumed_ids.add(section_id)

        if resumed_ids:
            logger.info(
                f"  📂 Resumed from {path.name}: "
                f"sections {sorted(resumed_ids)} loaded, will be skipped."
            )
            # Allow subclass to restore internal state from resumed data
            self.restore_section_state(data, resumed_ids)
            self.slog.log(
                "resumed",
                data={
                    "source": str(path),
                    "sections_restored": sorted(resumed_ids),
                },
            )
        else:
            logger.warning(
                f"  Resume file {path.name} has no successfully completed sections. "
                f"Running all sections from scratch."
            )

        return resumed_ids

    def restore_section_state(
        self, resumed_data: dict[str, Any], resumed_sections: set[int]
    ) -> None:
        """Hook for subclasses to restore internal state from a resumed run.

        Override this to reload section-specific data (e.g., VQE theta_opt,
        ground truth arrays) that downstream sections depend on.

        Parameters
        ----------
        resumed_data : dict
            The full result envelope from the resumed run.
        resumed_sections : set[int]
            Set of section IDs that were successfully restored.

        Example (in NoiselessPipelineRunner)::

            def restore_section_state(self, data, sections):
                results = data["results"]
                if 1 in sections:
                    # Restore ground truth from section_1.data
                    s1 = results["section_1"]["data"]
                    ...
                if 2 in sections:
                    # Restore VQE results from section_2.data
                    s2 = results["section_2"]["data"]
                    ...
        """
        # Default: no-op. Subclasses override as needed.

    def _filter_sections(self, sections: list[Section]) -> list[Section]:
        """Filter sections based on CLI --section argument."""
        if self._args.section:
            requested = set(self._args.section)
            selected = [s for s in sections if s.id in requested]
            if not selected:
                available = [s.id for s in sections]
                logger.error(
                    f"No sections match --section {self._args.section}. Available: {available}"
                )
                sys.exit(1)
            return selected
        return sections

    def _build_envelope(self, total_elapsed: float) -> dict[str, Any]:
        """Build the standardized result envelope.

        The output is dual-compatible:
        - Standard result_io format (timestamp, config, results, summary, metadata).
        - Digest/compare format (config.experiment_id, analysis.summary with pass_rate).

        This ensures the digest scanner (_parse_experiment_dir), compare.py,
        and ResultStore can all parse ValidationRunner outputs correctly.
        """
        n_pass = sum(1 for r in self._section_results if r.success)
        n_fail = sum(1 for r in self._section_results if not r.success)
        n_total = len(self._section_results)

        # Summary in digest-compatible format
        summary = {
            "n_sections": n_total,
            "n_passed": n_pass,
            "n_failed": n_fail,
            "pass_rate": n_pass / max(n_total, 1),
            "total_time_s": round(total_elapsed, 2),
            "total_elapsed_s": round(total_elapsed, 2),
            "all_passed": n_fail == 0,
        }

        # Add failure details to summary for quick diagnosis
        if n_fail > 0:
            failed_sections = []
            for r in self._section_results:
                if not r.success:
                    entry = {
                        "section_id": r.section_id,
                        "name": r.name,
                        "elapsed_s": round(r.elapsed_s, 2),
                    }
                    if r.error:
                        entry["error"] = r.error
                    # Include pass=False reason from data if available
                    if r.data and isinstance(r.data, dict):
                        if "pass" in r.data and not r.data["pass"]:
                            entry["reason"] = "section returned pass=False"
                    failed_sections.append(entry)
            summary["failed_sections"] = failed_sections

        # Per-section results
        results = {}
        for r in self._section_results:
            results[f"section_{r.section_id}"] = {
                "name": r.name,
                "success": r.success,
                "elapsed_s": round(r.elapsed_s, 2),
                "data": r.data,
                "error": r.error,
            }

        # Build config with required fields for digest compatibility
        config = self.build_config()
        # Ensure digest-required keys exist
        config.setdefault("experiment_id", self.experiment_id)
        config.setdefault("category", self.experiment_id[0] if self.experiment_id else "")
        config.setdefault("hypothesis", self.hypothesis)
        config.setdefault("description", self.description)
        if "system" not in config:
            config["system"] = {}
        if "seeds" not in config:
            config["seeds"] = []

        # Build envelope using result_io standard
        envelope = build_result_envelope(
            config=config,
            results=results,
            summary=summary,
            elapsed_s=total_elapsed,
            metadata=collect_run_metadata(),
        )

        # Add analysis wrapper for digest/compare.py compatibility
        # The digest scanner reads: data["analysis"]["summary"] and data["analysis"]["n_seeds"]
        envelope["analysis"] = {
            "experiment_id": self.experiment_id,
            "category": config.get("category", ""),
            "hypothesis": self.hypothesis,
            "n_seeds": len(config.get("seeds", [])),
            "summary": summary,
            "per_section": results,
        }

        # Add baseline comparison: find previous best run with same config
        # and record improvement metrics.
        try:
            from qmbp_simulation.framework.result_index import ResultIndex
            system = config.get("system", {})
            model = system.get("model", "")
            topo_list = system.get("topologies", [])
            topology = topo_list[0] if isinstance(topo_list, list) and topo_list else ""
            n_qubits = system.get("n_qubits", 0)
            p_layers = system.get("p_layers", 0)

            if model and topology:
                index = ResultIndex()
                baseline = index.get_best_run(
                    model=model, topology=topology,
                    n_qubits=n_qubits, p_layers=p_layers,
                )
                if baseline:
                    current_pass_rate = summary.get("pass_rate", 0)
                    baseline_pass_rate = baseline.get("pass_rate", 0)
                    envelope["baseline_ref"] = {
                        "file": baseline.get("_file", ""),
                        "pass_rate": baseline_pass_rate,
                        "timestamp": baseline.get("timestamp", ""),
                    }
                    envelope["improvement"] = {
                        "pass_rate_delta": round(current_pass_rate - baseline_pass_rate, 4),
                        "is_improvement": current_pass_rate > baseline_pass_rate,
                    }
        except Exception:
            pass  # Baseline lookup is best-effort

        return envelope

    def _print_header(self, sections: list[Section]) -> None:
        """Print runner header."""
        logger.info("")
        logger.info("╔" + "═" * 63 + "╗")
        logger.info(f"║  {self.description:<61} ║")
        logger.info("╚" + "═" * 63 + "╝")
        logger.info("")
        logger.info(f"  Runner ID:       {self.runner_id}")
        logger.info(f"  Experiment:      {self.experiment_id}")
        logger.info(f"  Hypothesis:      {self.hypothesis}")
        logger.info(f"  Sections:        {len(sections)}")
        if self._args.stop_on_failure:
            logger.info("  Stop on failure: YES")
        logger.info("")

    def _print_dry_run(self, sections: list[Section]) -> None:
        """Print section list for dry-run mode (no execution)."""
        logger.info("")
        logger.info(f"DRY RUN — {self.description}")
        logger.info("")
        logger.info(f"  {'#':<4} {'Section':<40} {'Hypothesis'}")
        logger.info(f"  {'-' * 4} {'-' * 40} {'-' * 40}")
        for s in sections:
            hyp = s.hypothesis[:40] if s.hypothesis else "—"
            logger.info(f"  {s.id:<4} {s.name:<40} {hyp}")
        logger.info("")
        logger.info(f"  Total: {len(sections)} sections (use --section N to select)")

    def _print_summary(self, total_elapsed: float, saved_path: Path) -> None:
        """Print final summary table."""
        n_pass = sum(1 for r in self._section_results if r.success)
        n_fail = sum(1 for r in self._section_results if not r.success)

        logger.info("")
        logger.info("=" * 65)
        logger.info("  FINAL SUMMARY")
        logger.info("=" * 65)
        logger.info(f"  Passed:   {n_pass}/{len(self._section_results)}")
        logger.info(f"  Failed:   {n_fail}/{len(self._section_results)}")
        logger.info(f"  Time:     {total_elapsed:.1f}s")
        logger.info("")

        # Per-section table
        logger.info(f"  {'#':<4} {'Section':<35} {'Status':<10} {'Time':<8}")
        logger.info(f"  {'-' * 4} {'-' * 35} {'-' * 10} {'-' * 8}")
        for r in self._section_results:
            status = "✅ PASS" if r.success else "❌ FAIL"
            logger.info(f"  {r.section_id:<4} {r.name:<35} {status:<10} {r.elapsed_s:.1f}s")

        logger.info("")
        logger.info(f"  Results: {saved_path}")
        logger.info("=" * 65)

        if n_fail > 0:
            logger.info("")
            logger.info("  FAILURES:")
            for r in self._section_results:
                if not r.success:
                    err = r.error or "returned pass=False"
                    logger.info(f"    ❌ Section {r.section_id} ({r.name}): {err[:80]}")
            logger.info("")

    def _setup_logging(self) -> None:
        """Configure logging based on CLI args."""
        level = logging.DEBUG if self._args.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(levelname)s: %(message)s",
            force=True,
        )

    @classmethod
    def _parse_args(cls) -> argparse.Namespace:
        """Parse standard CLI arguments for validation runners."""
        parser = argparse.ArgumentParser(
            description=cls.description or "Validation runner",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "--section",
            type=int,
            nargs="+",
            default=None,
            help="Run only specific sections (by number)",
        )
        parser.add_argument(
            "--skip-preflight",
            action="store_true",
            help="Skip preflight validation (not recommended)",
        )
        parser.add_argument(
            "--stop-on-failure",
            action="store_true",
            help="Stop execution after the first section failure",
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Enable verbose (DEBUG) logging + full tracebacks",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="List sections without executing",
        )
        # Validation flags
        parser.add_argument(
            "--validate-vqe",
            action="store_true",
            default=True,
            help="Run VQE result validation (default: on)",
        )
        parser.add_argument(
            "--no-validate-vqe",
            action="store_false",
            dest="validate_vqe",
            help="Disable VQE result validation",
        )
        parser.add_argument(
            "--validate-theta",
            action="store_true",
            default=True,
            help="Run θ_pred validation after MPNN inference (default: on)",
        )
        parser.add_argument(
            "--no-validate-theta",
            action="store_false",
            dest="validate_theta",
            help="Disable θ_pred validation",
        )
        parser.add_argument(
            "--theta-validation-level",
            type=int,
            default=4,
            choices=range(1, 8),
            metavar="[1-7]",
            help="Max θ validation level (1-4: cheap, 5-7: expensive). Default: 4",
        )
        parser.add_argument(
            "--strict-validation",
            action="store_true",
            default=False,
            help="Abort on CRITICAL validation failures",
        )
        # Resume from interrupted run
        parser.add_argument(
            "--resume",
            type=str,
            default=None,
            help="Resume from a partial/interrupted result JSON. Completed sections "
            "are loaded from the file and skipped. Only remaining sections run.",
        )
        # Allow subclasses to add custom args
        cls._add_custom_args(parser)
        return parser.parse_args()

    @classmethod
    def _add_custom_args(cls, parser: argparse.ArgumentParser) -> None:
        """Override to add custom CLI arguments.

        Called during argument parsing. Subclasses can add experiment-specific
        flags here without modifying the base class.

        Example::

            @classmethod
            def _add_custom_args(cls, parser):
                parser.add_argument("--g-value", type=float, default=0.3)
                parser.add_argument("--n-qubits", type=int, default=6)
        """

    # ── Reusable utility methods (available to all subclasses) ───────────────

    def _resolve_backend(self):
        """Resolve the noiseless backend from subclass attributes or create new.

        Checks self.noiseless → self.backend → creates NoiselessBackend().
        This avoids duplicating backend instances across utility calls.
        """
        from qmbp_simulation.execution import NoiselessBackend

        return (
            getattr(self, "noiseless", None) or getattr(self, "backend", None) or NoiselessBackend()
        )

    def select_backend(self, n_qubits: int, *, for_vqe_loop: bool = False):
        """Select the optimal noiseless backend for a given system size.

        Delegates to the canonical select_backend() from execution module.
        Automatically uses MPS for large N (>10 for VQE loops, >15 otherwise).

        Parameters
        ----------
        n_qubits : int
            Number of qubits in the system.
        for_vqe_loop : bool
            If True, uses MPS threshold at N>10 (VQE is iterative, so
            O(2^N) statevector becomes prohibitive faster). Default False.

        Returns
        -------
        ExecutionBackend
            NoiselessBackend or MPSBackend depending on N.
        """
        from qmbp_simulation.execution import select_backend as _select_backend

        return _select_backend(n_qubits, for_vqe_loop=for_vqe_loop)

    def setup_physics(self) -> None:
        """Initialize standard physics objects used by most runners.

        Sets up: builder, solver, hva, make_lattice, get_model_spec,
        noiseless backend, and VQEOptimizer/VQEConfig imports.

        After calling this method, the following attributes are available:
            self.builder       — HamiltonianBuilder()
            self.solver        — ClassicalSolver()
            self.hva           — HVACircuitBuilder()
            self.make_lattice  — make_lattice function
            self.get_model_spec — get_model_spec function
            self.noiseless     — NoiselessBackend()
            self.NoiselessBackend — NoiselessBackend class
            self.MPSBackend    — MPSBackend class (lazy, for large N)
            self.VQEOptimizer  — VQEOptimizer class
            self.VQEConfig     — VQEConfig class

        Subclasses can call this in setup() to avoid repeating the same
        8-line import block in every runner.
        """
        from qmbp_simulation import (
            ClassicalSolver,
            HamiltonianBuilder,
            HVACircuitBuilder,
            VQEConfig,
            VQEOptimizer,
            make_lattice,
        )
        from qmbp_simulation.execution import NoiselessBackend
        from qmbp_simulation.execution.mps_backend import MPSBackend
        from qmbp_simulation.models.model_registry import get_model_spec

        self.builder = HamiltonianBuilder()
        self.solver = ClassicalSolver()
        self.hva = HVACircuitBuilder()
        self.make_lattice = make_lattice
        self.get_model_spec = get_model_spec
        self.noiseless = NoiselessBackend()
        self.NoiselessBackend = NoiselessBackend
        self.MPSBackend = MPSBackend
        self.VQEOptimizer = VQEOptimizer
        self.VQEConfig = VQEConfig

    # ── Checkpoint infrastructure (reusable by all subclasses) ───────────────

    def _checkpoint_dir(self) -> Path:
        """Return the checkpoint directory for this runner's experiment.

        Creates the directory if it doesn't exist. Checkpoint files are hidden
        (dot-prefixed) and removed on successful run completion.

        Returns
        -------
        Path
            Directory where checkpoint files are stored.
        """
        d = Path("results") / "experiments" / f"exp_{self.experiment_id.lower()}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_checkpoint(
        self,
        label: str,
        data: dict[str, Any],
    ) -> None:
        """Save a named checkpoint for crash recovery.

        Checkpoints are best-effort — a save failure never interrupts the main
        computation. Files are dot-prefixed to be hidden from result scanners.

        Parameters
        ----------
        label : str
            Checkpoint name (e.g., "vqe_chain_1d", "mpnn_epoch_5").
            Used in the filename: .checkpoint_{label}.json
        data : dict
            Checkpoint payload. Must be JSON-serializable (numpy arrays
            should be converted to lists before passing).
        """
        from datetime import datetime

        from qmbp_simulation.utils.helpers import json_dump

        cp_path = self._checkpoint_dir() / f".checkpoint_{label}.json"
        payload = {
            **data,
            "_checkpoint_meta": {
                "runner_id": self.runner_id,
                "experiment_id": self.experiment_id,
                "label": label,
                "saved_at": datetime.now().isoformat(),
            },
        }
        try:
            json_dump(payload, cp_path)
            logger.debug("💾 Checkpoint saved: %s", label)
        except Exception as e:
            logger.debug("Checkpoint save failed for %s: %s", label, e)

    def load_checkpoint(self, label: str) -> dict[str, Any] | None:
        """Load a named checkpoint if it exists.

        Parameters
        ----------
        label : str
            Checkpoint name (same as used in save_checkpoint).

        Returns
        -------
        dict | None
            Checkpoint data (without _checkpoint_meta), or None if not found
            or corrupt.
        """
        import json as _json

        cp_path = self._checkpoint_dir() / f".checkpoint_{label}.json"
        if not cp_path.exists():
            return None

        try:
            with open(cp_path) as f:
                payload = _json.load(f)
            meta = payload.pop("_checkpoint_meta", {})
            saved_at = meta.get("saved_at", "unknown")
            logger.info("♻️  Loaded checkpoint '%s' (saved %s)", label, saved_at)
            return payload
        except (ValueError, KeyError, OSError) as e:
            logger.warning(
                "⚠️  Corrupt checkpoint '%s', ignoring: %s", label, e
            )
            return None

    def cleanup_checkpoints(self, pattern: str = "*") -> None:
        """Remove checkpoint files after successful completion.

        Parameters
        ----------
        pattern : str
            Glob suffix pattern. Default "*" removes all checkpoints.
            Use a specific label to remove one: e.g., "vqe_chain_1d".
        """
        cp_dir = self._checkpoint_dir()
        if not cp_dir.exists():
            return
        glob_pattern = f".checkpoint_{pattern}.json"
        for cp in cp_dir.glob(glob_pattern):
            try:
                cp.unlink()
                logger.debug("🗑️  Removed checkpoint: %s", cp.name)
            except OSError as e:
                logger.debug("Could not remove checkpoint %s: %s", cp.name, e)

    # ── Logging utilities (reusable by all subclasses) ───────────────────────

    @staticmethod
    def log_memory_estimate(n_qubits: int, label: str = "statevector") -> None:
        """Log estimated memory footprint for a statevector computation.

        Useful at section start for large-N runs to anticipate OOM before it
        happens. Only logs at INFO when memory > 100 MB, otherwise DEBUG.

        Parameters
        ----------
        n_qubits : int
            Number of qubits in the system.
        label : str
            Description of what the memory is for (default: "statevector").
        """
        # Complex128 statevector: 2^N * 16 bytes
        mem_bytes = (2 ** n_qubits) * 16
        mem_mb = mem_bytes / 1e6
        if mem_mb > 100:
            logger.info("    📐 Estimated %s memory: %.1f MB (N=%d)", label, mem_mb, n_qubits)
        else:
            logger.debug("    📐 Estimated %s memory: %.1f MB (N=%d)", label, mem_mb, n_qubits)

    def vqe_descending_sweep(
        self,
        topology: str,
        n_qubits: int,
        h_values: list[float],
        seed: int,
        *,
        p_layers: int = 1,
        n_restarts: int = 1,
        maxiter: int = 500,
        sigma: float = 0.1,
        model: str = "tfim",
        model_kwargs: dict | None = None,
    ) -> dict[float, np.ndarray]:
        """Run a descending VQE sweep and return h -> theta_opt mapping.

        Convenience method for validation sections that need VQE data as a
        precondition (smoothness analysis, MPNN training, error decomposition).
        Avoids duplicating the warm-start VQE loop across runners.

        Parameters
        ----------
        topology : str
            Lattice topology name.
        n_qubits : int
            Number of qubits.
        h_values : list[float]
            Transverse field values to sweep (will be sorted descending).
        seed : int
            Random seed for reproducibility.
        p_layers : int
            HVA circuit depth (default: 1).
        n_restarts : int
            Number of VQE restarts per h-point (default: 1 for p=1).
        maxiter : int
            Maximum optimizer iterations per restart.
        sigma : float
            Restart perturbation standard deviation.
        model : str
            Model name from registry ("tfim", "tfim_longitudinal", etc.).
        model_kwargs : dict | None
            Extra kwargs for Hamiltonian/circuit construction (e.g., {"g": 0.3}).

        Returns
        -------
        dict[float, np.ndarray]
            Mapping from h-value to optimized parameter vector.
        """
        import numpy as np
        from scipy.optimize import minimize

        from qmbp_simulation import make_lattice
        from qmbp_simulation.models.model_registry import get_model_spec

        spec = get_model_spec(model)
        backend = self._resolve_backend()
        mkw = model_kwargs or {}

        rng = np.random.default_rng(seed)
        lattice_ref = make_lattice(topology, n_qubits, J=1.0, h=max(h_values))
        circuit, _ = spec.create_circuit(n_qubits, p_layers, lattice_ref, **spec.circuit_kwargs)
        n_params = circuit.num_parameters
        prev_theta = rng.uniform(-0.01, 0.01, n_params)

        results: dict[float, np.ndarray] = {}
        for h in sorted(h_values, reverse=True):
            lattice = make_lattice(topology, n_qubits, J=1.0, h=h)
            H = spec.build_hamiltonian(lattice, **{**spec.hamiltonian_kwargs, **mkw})

            best_energy = float("inf")
            best_theta = prev_theta.copy()
            for restart in range(n_restarts):
                x0 = (
                    prev_theta + rng.normal(0, sigma, n_params)
                    if restart > 0
                    else prev_theta.copy()
                )
                x0 = np.clip(x0, -np.pi, np.pi)
                res = minimize(
                    lambda params, _H=H, _c=circuit: backend.evaluate(_c, _H, params),
                    x0,
                    method="L-BFGS-B",
                    bounds=[(-np.pi, np.pi)] * n_params,
                    options={"maxiter": maxiter, "ftol": 1e-14},
                )
                if res.fun < best_energy:
                    best_energy = res.fun
                    best_theta = res.x.copy()
            prev_theta = best_theta.copy()
            results[h] = best_theta.copy()

        return results

    @staticmethod
    def exact_ground_state(
        topology: str,
        n_qubits: int,
        h: float,
        *,
        model: str = "tfim",
        model_kwargs: dict | None = None,
    ) -> tuple[float, float]:
        """Get exact ground state energy and gap.

        Dispatches to exact diag (N<=15) or DMRG (N>15) automatically.

        Parameters
        ----------
        topology : str
            Lattice topology name.
        n_qubits : int
            Number of qubits.
        h : float
            Transverse field value.
        model : str
            Model name (default: "tfim").
        model_kwargs : dict | None
            Extra kwargs for Hamiltonian construction.

        Returns
        -------
        (e_exact, gap) : tuple[float, float]
            Ground state energy and spectral gap.
        """
        import numpy as np

        from qmbp_simulation import ClassicalSolver, make_lattice
        from qmbp_simulation.models.constants import EXACT_DIAG_QUBIT_LIMIT
        from qmbp_simulation.models.model_registry import get_model_spec

        spec = get_model_spec(model)
        mkw = model_kwargs or {}
        lattice = make_lattice(topology, n_qubits, J=1.0, h=h)
        H = spec.build_hamiltonian(lattice, **{**spec.hamiltonian_kwargs, **mkw})

        if n_qubits <= EXACT_DIAG_QUBIT_LIMIT:
            H_mat = H.to_matrix()
            if hasattr(H_mat, "toarray"):
                H_mat = H_mat.toarray()
            evals = np.sort(np.linalg.eigvalsh(H_mat))
            return float(evals[0]), float(evals[1] - evals[0])
        else:
            solver = ClassicalSolver()
            result = solver.solve(H, lattice)
            return result.ground_energy, result.gap

    @staticmethod
    def compute_fidelity(
        circuit,
        theta: np.ndarray,
        exact_state: np.ndarray,
    ) -> float:
        """Compute state fidelity |⟨ψ_exact|ψ_vqe⟩|².

        Parameters
        ----------
        circuit : QuantumCircuit
            Parametrized quantum circuit.
        theta : np.ndarray
            Optimized parameters.
        exact_state : np.ndarray
            Exact ground state vector (from eigsh or DMRG).

        Returns
        -------
        float
            Fidelity in [0, 1].
        """
        from qiskit.quantum_info import Statevector, state_fidelity

        bound = circuit.assign_parameters(theta)
        sv_vqe = Statevector(bound)
        sv_exact = Statevector(exact_state)
        return float(state_fidelity(sv_vqe, sv_exact))

    @staticmethod
    def validate_theta_prediction(
        theta_pred: np.ndarray,
        theta_training: np.ndarray,
        h_values_training: np.ndarray | None = None,
        h_test: float | None = None,
        circuit=None,
        exact_state: np.ndarray | None = None,
        energy_fn=None,
        model=None,
        graph_data=None,
        level: int = 4,
    ) -> dict:
        """Validate MPNN-predicted θ using ThetaValidator.

        Convenience method for experiment scripts. Returns the report dict
        suitable for embedding in experiment results.

        Parameters
        ----------
        theta_pred : np.ndarray
            Predicted variational parameters.
        theta_training : np.ndarray [n_points, n_params]
            Training θ_opt array for building the validator.
        h_values_training : np.ndarray | None
            Training h-values for interpolation check.
        h_test : float | None
            Test h-value.
        circuit : QuantumCircuit | None
            For L4 fidelity (requires exact_state too).
        exact_state : np.ndarray | None
            For L4 fidelity.
        energy_fn : callable | None
            E(θ) → float for L5/L7.
        model : MPNNPredictor | None
            For L6 MC Dropout.
        graph_data : Data | None
            For L6 MC Dropout.
        level : int
            Maximum validation level (1-7, default 4).

        Returns
        -------
        dict
            JSON-serializable validation report.
        """
        from qmbp_simulation.analysis.theta_validator import ThetaValidator

        validator = ThetaValidator.from_training_data(
            theta_opt=theta_training,
            h_values=h_values_training,
        )
        report = validator.validate(
            theta_pred,
            level=level,
            h_test=h_test,
            circuit=circuit,
            exact_state=exact_state,
            energy_fn=energy_fn,
            model=model,
            graph_data=graph_data,
        )
        return report.to_dict()

    # ── Cross-N / MPNN utility methods (reusable by subclasses) ────────────

    @staticmethod
    def compute_vqe_quality_metrics(
        vqe_energies: list[float],
        exact_energies: list[float],
        gaps: list[float],
    ) -> dict:
        """Compute per-point ΔE/gap and summary quality metrics for VQE data.

        Returns dict with 'de_gaps', 'n_pass', 'mean_de_gap', 'pass_rate'.
        Reusable across all runners that generate VQE training data.
        """
        import numpy as np

        de_gaps = [
            abs(vqe_energies[i] - exact_energies[i]) / max(gaps[i], 1e-10)
            for i in range(len(vqe_energies))
        ]
        n_pass = sum(1 for d in de_gaps if d < 0.05)
        return {
            "de_gaps": de_gaps,
            "n_pass": n_pass,
            "mean_de_gap": float(np.mean(de_gaps)) if de_gaps else 0.0,
            "pass_rate": n_pass / len(de_gaps) if de_gaps else 0.0,
        }

    @staticmethod
    def compute_theta_smoothness(theta_array) -> float:
        """Compute max L-inf change between consecutive θ vectors.

        A high value (>1.0) indicates the MPNN will struggle to learn
        the mapping (discontinuous landscape). Used as learnability predictor.
        """
        import numpy as np

        if len(theta_array) < 2:
            return 0.0
        return float(np.max(np.abs(np.diff(theta_array, axis=0))))

    @staticmethod
    def select_mpnn_hidden_dim(
        n_training_graphs: int,
        theta_dim: int,
        max_hidden: int = 128,
        min_hidden: int = 32,
        max_param_ratio: int = 100,
    ) -> int:
        """Auto-select MPNN hidden_dim based on dataset size.

        Prevents severe overparameterization (99K params for 27 graphs)
        by selecting the largest hidden_dim where n_params < ratio × n_data.

        Parameters
        ----------
        n_training_graphs : int
            Number of graphs in training dataset.
        theta_dim : int
            Output dimension (number of VQE parameters to predict).
        max_hidden : int
            Maximum hidden dimension to try (default 128).
        min_hidden : int
            Minimum hidden dimension floor (default 32).
        max_param_ratio : int
            Maximum acceptable params/data ratio (default 100).

        Returns
        -------
        int : Selected hidden dimension.
        """
        # Lazy import to avoid torch dependency in non-ML runners
        try:
            from qmbp_simulation.predictors import MPNNPredictor
        except ImportError:
            return min_hidden

        for candidate in sorted(set([max_hidden, 64, min_hidden]), reverse=True):
            if candidate < min_hidden:
                continue
            model = MPNNPredictor(
                node_features=3,
                hidden_dim=candidate,
                n_layers=3,
                output_dim=theta_dim,
                norm_type="none",
            )
            n_params = sum(p.numel() for p in model.parameters())
            if n_params < n_training_graphs * max_param_ratio:
                return candidate

        return min_hidden

    @staticmethod
    def check_variational_principle(
        vqe_energies: list[float],
        exact_energies: list[float],
        tolerance: float = 1e-8,
    ) -> int:
        """Count variational principle violations (E_vqe < E_exact).

        Returns the number of points where VQE energy is below exact
        (indicating numerical noise or unconverged reference).
        """
        return sum(
            1 for i in range(len(vqe_energies)) if vqe_energies[i] < exact_energies[i] - tolerance
        )

    @staticmethod
    def truncate_statevector_mps(
        psi: np.ndarray,
        n_qubits: int,
        chi_max: int,
    ) -> np.ndarray:
        """Truncate a statevector to MPS with limited bond dimension.

        Converts |ψ⟩ to MPS via sequential SVD, truncates each bond to
        chi_max singular values, then reconstructs the truncated |ψ⟩.

        Useful for simulating hardware noise effects deterministically:
        low chi ≈ decoherence (both destroy long-range correlations).

        Parameters
        ----------
        psi : np.ndarray
            Full statevector (shape: (2^n_qubits,) or (2^n_qubits, 1)).
        n_qubits : int
            Number of qubits.
        chi_max : int
            Maximum bond dimension (truncation limit).

        Returns
        -------
        np.ndarray
            Normalized truncated statevector (shape: (2^n_qubits,)).
        """
        import numpy as np

        state = np.asarray(psi).reshape(-1)
        dims = [2] * n_qubits
        mps_tensors = []
        remaining = state.reshape(1, -1)

        for site in range(n_qubits - 1):
            chi_left = remaining.shape[0]
            d_site = dims[site]
            mat = remaining.reshape(chi_left * d_site, -1)

            U, S, Vh = np.linalg.svd(mat, full_matrices=False)
            chi_new = min(len(S), chi_max)
            U = U[:, :chi_new]
            S = S[:chi_new]
            Vh = Vh[:chi_new, :]

            norm = np.linalg.norm(S)
            if norm > 1e-15:
                S = S / norm

            mps_tensors.append(U.reshape(chi_left, d_site, chi_new))
            remaining = np.diag(S) @ Vh

        mps_tensors.append(remaining.reshape(remaining.shape[0], dims[-1], 1))

        # Reconstruct truncated statevector
        psi_trunc = mps_tensors[0]
        for t in mps_tensors[1:]:
            psi_trunc = np.einsum("ijk,klm->ijlm", psi_trunc, t)
            psi_trunc = psi_trunc.reshape(
                psi_trunc.shape[0],
                psi_trunc.shape[1] * t.shape[1],
                t.shape[2],
            )  # type: ignore[assignment]

        psi_trunc = psi_trunc.reshape(-1)  # type: ignore[assignment]
        norm = np.linalg.norm(psi_trunc)
        if norm > 1e-15:
            psi_trunc = psi_trunc / norm
        return psi_trunc

    # ── MPNN Evaluation Utilities ────────────────────────────────────────────

    def benchmark_mpnn_warmstart(
        self,
        topology: str,
        n_qubits: int,
        h_train: list[float],
        h_test: list[float],
        *,
        p_layers: int = 1,
        seed: int = 42,
        n_restarts_vqe: int = 1,
        maxiter_vqe: int = 500,
        mpnn_hidden_dim: int = 64,
        mpnn_epochs: int = 4000,
        mpnn_lr: float = 1e-3,
        mpnn_patience: int = 150,
        n_vqe_restarts_from_pred: int = 5,
        maxiter_refine: int = 200,
        model: str = "tfim",
        de_gap_threshold: float = 0.05,
    ) -> dict:
        """Benchmark MPNN warm-start vs random init vs warm-start from previous h.

        Answers: "Does the MPNN produce a better starting point for VQE than
        alternatives, and how many iterations does each strategy need?"

        Three strategies compared at each h_test point:
          - ``random``: init from uniform(-0.01, 0.01), run VQE until convergence.
          - ``prev_h``: init from θ_opt of the nearest training h (classical warm-start).
          - ``mpnn``: init from MPNN prediction, run VQE until convergence.

        For each strategy the reported metrics are:
          - ``iters``: optimizer iterations to convergence (lower = faster).
          - ``final_de_gap``: ΔE/gap after VQE refinement.
          - ``init_de_gap``: ΔE/gap of the initial θ before any VQE (MPNN quality).
          - ``speedup``: iters_random / iters_mpnn (only in mpnn entry).

        Parameters
        ----------
        topology : str
            Lattice topology name.
        n_qubits : int
            System size.
        h_train : list[float]
            Descending grid used for MPNN training data (VQE sweep).
        h_test : list[float]
            Unseen h-values for evaluation (should not overlap h_train).
        p_layers : int
            HVA depth (default: 1).
        seed : int
            Random seed (default: 42).
        n_restarts_vqe : int
            VQE restarts for training data generation (default: 1).
        maxiter_vqe : int
            Max iterations for training VQE sweep (default: 500).
        mpnn_hidden_dim : int
            MPNN hidden dimension (default: 64).
        mpnn_epochs : int
            MPNN training epochs (default: 4000).
        mpnn_lr : float
            MPNN learning rate (default: 1e-3).
        mpnn_patience : int
            MPNN early-stopping patience (default: 150).
        n_vqe_restarts_from_pred : int
            How many independent VQE runs per strategy at h_test (default: 5).
            Higher = more reliable iteration count estimate.
        maxiter_refine : int
            Max iterations allowed for refinement VQE (default: 200).
            Intentionally smaller than maxiter_vqe to stress-test warm-starts.
        model : str
            Hamiltonian model name (default: "tfim").
        de_gap_threshold : float
            Pass criterion for final ΔE/gap (default: 0.05).

        Returns
        -------
        dict with keys:
            ``per_h``: list of per-h-point results (one per h_test value).
            ``summary``: aggregate speedup, win_rate, mean_de_gap per strategy.
            ``mpnn_train_mse``: final MPNN training MSE.
            ``n_train_points``: number of training graphs.
            ``pass``: True if all h_test points pass ΔE/gap threshold with mpnn init.
        """
        import numpy as np
        import torch
        from scipy.optimize import minimize
        from torch_geometric.data import Data

        from qmbp_simulation import make_lattice
        from qmbp_simulation.execution import NoiselessBackend
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn

        spec = get_model_spec(model)
        _resolved = self._resolve_backend()
        backend = _resolved if _resolved is not None else NoiselessBackend()
        rng = np.random.default_rng(seed)

        # ── Step 1: VQE sweep on training grid ──────────────────────────────
        theta_map = self.vqe_descending_sweep(
            topology,
            n_qubits,
            h_train,
            seed=seed,
            p_layers=p_layers,
            n_restarts=n_restarts_vqe,
            maxiter=maxiter_vqe,
            model=model,
        )
        h_arr = np.array(sorted(theta_map.keys(), reverse=True))
        theta_arr = np.array([theta_map[h] for h in h_arr])
        e_arr = np.array(
            [self.exact_ground_state(topology, n_qubits, float(h), model=model)[0] for h in h_arr]
        )
        n_params = theta_arr.shape[1]

        # ── Step 2: Train MPNN ───────────────────────────────────────────────
        lattice_ref = make_lattice(topology, n_qubits, J=1.0, h=float(h_arr[0]))
        from qmbp_simulation.models import HamiltonianBuilder

        builder = HamiltonianBuilder()
        dataset = build_graph_dataset(
            lattice_ref,
            h_values=h_arr,
            theta_opt=theta_arr,
            e_exact=e_arr,
            fidelity_threshold=0.0,  # noqa: noiseless VQE data — no filtering needed
        )
        predictor = MPNNPredictor(
            node_features=dataset[0].x.shape[1],
            output_dim=n_params,
            hidden_dim=mpnn_hidden_dim,
        )
        train_result = train_mpnn(
            predictor,
            dataset,
            n_epochs=mpnn_epochs,
            lr=mpnn_lr,
            patience=mpnn_patience,
            seed=seed,
        )
        predictor.eval()

        # ── Step 3: Benchmark at each h_test ────────────────────────────────
        per_h: list[dict] = []
        for h_t in h_test:
            e_exact, gap = self.exact_ground_state(topology, n_qubits, h_t, model=model)
            lattice_t = make_lattice(topology, n_qubits, J=1.0, h=h_t)
            H_t = spec.build_hamiltonian(lattice_t, **spec.hamiltonian_kwargs)
            circuit_t, _ = spec.create_circuit(n_qubits, p_layers, lattice_t, **spec.circuit_kwargs)

            def _energy(params: np.ndarray) -> float:
                return float(backend.evaluate(circuit_t, H_t, params))

            # ── MPNN prediction (θ_pred) ─────────────────────────────────
            edge_index_np, coord = builder.build_graph_data(lattice_t)
            h_feat = np.full(n_qubits, float(h_t))
            x = torch.tensor(np.stack([h_feat, coord.astype(float)], axis=1), dtype=torch.float32)
            edge_index = torch.tensor(edge_index_np, dtype=torch.long)
            graph = Data(x=x, edge_index=edge_index)
            graph.batch = torch.zeros(n_qubits, dtype=torch.long)
            with torch.no_grad():
                theta_mpnn = predictor(graph).numpy().flatten()

            # ΔE/gap before any VQE (raw MPNN quality)
            e_mpnn_raw = _energy(theta_mpnn)
            init_de_gap_mpnn = abs(e_mpnn_raw - e_exact) / max(gap, 1e-10)

            # ── prev_h warm-start: nearest training θ ───────────────────
            nearest_idx = int(np.argmin(np.abs(h_arr - h_t)))
            theta_prev_h = theta_arr[nearest_idx].copy()
            e_prev_h_raw = _energy(theta_prev_h)
            init_de_gap_prev = abs(e_prev_h_raw - e_exact) / max(gap, 1e-10)

            # ── Run VQE from each starting point, n_vqe_restarts_from_pred times ──
            def _run_vqe_from(theta_init: np.ndarray) -> tuple[float, int]:
                """Return (final_energy, n_iters) from a single VQE run."""
                counter = [0]

                def _counted(params: np.ndarray) -> float:
                    counter[0] += 1
                    return float(backend.evaluate(circuit_t, H_t, params))

                res = minimize(
                    _counted,
                    theta_init.copy(),
                    method="L-BFGS-B",
                    bounds=[(-np.pi, np.pi)] * n_params,
                    options={"maxiter": maxiter_refine, "ftol": 1e-12},
                )
                return float(res.fun), counter[0]

            # Random init: average over n_vqe_restarts_from_pred trials
            rand_energies, rand_iters = [], []
            for _ in range(n_vqe_restarts_from_pred):
                theta_rand = rng.uniform(-0.01, 0.01, n_params)
                e_r, iters_r = _run_vqe_from(theta_rand)
                rand_energies.append(e_r)
                rand_iters.append(iters_r)
            rand_mean_iters = float(np.mean(rand_iters))
            rand_best_e = float(np.min(rand_energies))
            rand_de_gap = abs(rand_best_e - e_exact) / max(gap, 1e-10)

            # prev_h warm-start: single run (deterministic init)
            e_prev_final, iters_prev = _run_vqe_from(theta_prev_h)
            prev_de_gap = abs(e_prev_final - e_exact) / max(gap, 1e-10)

            # MPNN warm-start: single run from prediction
            e_mpnn_final, iters_mpnn = _run_vqe_from(theta_mpnn)
            mpnn_de_gap = abs(e_mpnn_final - e_exact) / max(gap, 1e-10)

            speedup = rand_mean_iters / max(iters_mpnn, 1)
            speedup_vs_prev = iters_prev / max(iters_mpnn, 1)

            per_h.append(
                {
                    "h": h_t,
                    "e_exact": e_exact,
                    "gap": gap,
                    "random": {
                        "mean_iters": rand_mean_iters,
                        "best_de_gap": rand_de_gap,
                        "iters_list": rand_iters,
                    },
                    "prev_h": {
                        "nearest_h_train": float(h_arr[nearest_idx]),
                        "init_de_gap": init_de_gap_prev,
                        "final_de_gap": prev_de_gap,
                        "iters": iters_prev,
                    },
                    "mpnn": {
                        "init_de_gap": init_de_gap_mpnn,
                        "final_de_gap": mpnn_de_gap,
                        "iters": iters_mpnn,
                        "speedup_vs_random": speedup,
                        "speedup_vs_prev_h": speedup_vs_prev,
                    },
                    "pass": mpnn_de_gap < de_gap_threshold,
                }
            )
            logger.info(
                f"  h={h_t:.3f}: "
                f"rand={rand_mean_iters:.0f}it/{rand_de_gap:.4f}, "
                f"prev_h={iters_prev}it/{prev_de_gap:.4f}, "
                f"mpnn_init={init_de_gap_mpnn:.4f} "
                f"→ {iters_mpnn}it/{mpnn_de_gap:.4f} "
                f"(speedup={speedup:.1f}x) "
                f"[{'PASS' if mpnn_de_gap < de_gap_threshold else 'FAIL'}]"
            )

        # ── Summary ─────────────────────────────────────────────────────────
        mean_speedup = float(np.mean([r["mpnn"]["speedup_vs_random"] for r in per_h]))
        mean_speedup_vs_prev = float(np.mean([r["mpnn"]["speedup_vs_prev_h"] for r in per_h]))
        mpnn_wins_vs_random = sum(r["mpnn"]["iters"] < r["random"]["mean_iters"] for r in per_h)
        mpnn_wins_vs_prev = sum(r["mpnn"]["iters"] < r["prev_h"]["iters"] for r in per_h)
        mean_init_de_gap = float(np.mean([r["mpnn"]["init_de_gap"] for r in per_h]))
        all_pass = all(r["pass"] for r in per_h)

        return {
            "per_h": per_h,
            "summary": {
                "mean_speedup_vs_random": mean_speedup,
                "mean_speedup_vs_prev_h": mean_speedup_vs_prev,
                "mpnn_wins_vs_random": f"{mpnn_wins_vs_random}/{len(per_h)}",
                "mpnn_wins_vs_prev_h": f"{mpnn_wins_vs_prev}/{len(per_h)}",
                "mean_init_de_gap": mean_init_de_gap,
                "mean_final_de_gap_mpnn": float(
                    np.mean([r["mpnn"]["final_de_gap"] for r in per_h])
                ),
            },
            "mpnn_train_mse": train_result["final_mse"],
            "n_train_points": len(dataset),
            "n_params": n_params,
            "pass": all_pass,
        }

    def mpnn_leave_one_out_cv(
        self,
        topology: str,
        n_qubits: int,
        h_train: list[float],
        *,
        p_layers: int = 1,
        seed: int = 42,
        n_restarts_vqe: int = 1,
        maxiter_vqe: int = 500,
        mpnn_hidden_dim: int = 64,
        mpnn_epochs: int = 4000,
        mpnn_lr: float = 1e-3,
        mpnn_patience: int = 150,
        model: str = "tfim",
        de_gap_threshold: float = 0.05,
        min_train_size: int = 5,
    ) -> dict:
        """Leave-one-out cross-validation for MPNN generalization estimate.

        For each point h_i in h_train:
          1. Train MPNN on all other N-1 points.
          2. Predict θ at h_i (unseen by the model).
          3. Evaluate ΔE/gap(θ_pred) noiseless.

        This gives an unbiased estimate of generalization without needing
        a separate test set — useful when the h-grid is small (10-20 points).

        The LOO-CV score complements ΔE/gap on a held-out h_test: while
        h_test checks extrapolation/interpolation at a specific point,
        LOO-CV characterizes the model's reliability across the entire
        training distribution.

        Parameters
        ----------
        topology : str
            Lattice topology name.
        n_qubits : int
            System size.
        h_train : list[float]
            Full training grid. Each point is held out once.
        p_layers : int
            HVA depth (default: 1).
        seed : int
            Random seed for MPNN weight initialization (default: 42).
        n_restarts_vqe : int
            VQE restarts for data generation (default: 1).
        maxiter_vqe : int
            Max VQE iterations (default: 500).
        mpnn_hidden_dim : int
            MPNN hidden dimension (default: 64).
        mpnn_epochs : int
            MPNN training epochs per fold (default: 4000).
        mpnn_lr : float
            Learning rate (default: 1e-3).
        mpnn_patience : int
            Early-stopping patience (default: 150).
        model : str
            Hamiltonian model (default: "tfim").
        de_gap_threshold : float
            Pass threshold (default: 0.05).
        min_train_size : int
            Minimum fold size to proceed with training (default: 5).
            Folds with fewer training points are skipped.

        Returns
        -------
        dict with keys:
            ``per_fold``: per-h results (h_held_out, de_gap, pass, train_mse).
            ``summary``: mean/max/std ΔE/gap, pass_rate, n_folds_run.
            ``full_model_train_mse``: MSE of model trained on ALL points (reference).
            ``pass``: True if pass_rate >= 0.80 (80% of folds pass threshold).
        """
        import numpy as np
        import torch
        from torch_geometric.data import Data

        from qmbp_simulation import make_lattice
        from qmbp_simulation.execution import NoiselessBackend
        from qmbp_simulation.models import HamiltonianBuilder
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn

        spec = get_model_spec(model)
        _resolved = self._resolve_backend()
        backend = _resolved if _resolved is not None else NoiselessBackend()

        # ── Step 1: VQE sweep on full grid ──────────────────────────────────
        theta_map = self.vqe_descending_sweep(
            topology,
            n_qubits,
            h_train,
            seed=seed,
            p_layers=p_layers,
            n_restarts=n_restarts_vqe,
            maxiter=maxiter_vqe,
            model=model,
        )
        h_arr = np.array(sorted(theta_map.keys(), reverse=True))
        theta_arr = np.array([theta_map[h] for h in h_arr])
        e_arr = np.array(
            [self.exact_ground_state(topology, n_qubits, float(h), model=model)[0] for h in h_arr]
        )
        n_params = theta_arr.shape[1]
        n_total = len(h_arr)
        builder = HamiltonianBuilder()

        # ── Step 2: Train full model as reference ────────────────────────────
        lattice_ref = make_lattice(topology, n_qubits, J=1.0, h=float(h_arr[0]))
        full_dataset = build_graph_dataset(
            lattice_ref,
            h_values=h_arr,
            theta_opt=theta_arr,
            e_exact=e_arr,
            fidelity_threshold=0.0,  # noqa: noiseless VQE data — no filtering needed
        )
        full_model = MPNNPredictor(
            node_features=full_dataset[0].x.shape[1],
            output_dim=n_params,
            hidden_dim=mpnn_hidden_dim,
        )
        full_train = train_mpnn(
            full_model,
            full_dataset,
            n_epochs=mpnn_epochs,
            lr=mpnn_lr,
            patience=mpnn_patience,
            seed=seed,
        )
        full_model_mse = full_train["final_mse"]
        logger.info(f"  Full model MSE: {full_model_mse:.2e} ({n_total} points)")

        # ── Step 3: LOO folds ────────────────────────────────────────────────
        per_fold: list[dict] = []
        for fold_idx in range(n_total):
            h_held = float(h_arr[fold_idx])
            mask = np.arange(n_total) != fold_idx
            h_fold = h_arr[mask]
            theta_fold = theta_arr[mask]
            e_fold = e_arr[mask]

            if len(h_fold) < min_train_size:
                logger.warning(
                    f"  Fold {fold_idx} (h={h_held:.3f}): only {len(h_fold)} train points "
                    f"< min_train_size={min_train_size}. Skipping."
                )
                continue

            # Build fold dataset
            fold_dataset = build_graph_dataset(
                lattice_ref,
                h_values=h_fold,
                theta_opt=theta_fold,
                e_exact=e_fold,
                fidelity_threshold=0.0,  # noqa: noiseless VQE data — no filtering needed
            )

            # Train fold model — fresh weights each fold
            fold_model = MPNNPredictor(
                node_features=fold_dataset[0].x.shape[1],
                output_dim=n_params,
                hidden_dim=mpnn_hidden_dim,
            )
            fold_train = train_mpnn(
                fold_model,
                fold_dataset,
                n_epochs=mpnn_epochs,
                lr=mpnn_lr,
                patience=mpnn_patience,
                seed=seed + fold_idx,  # distinct seed per fold
            )
            fold_model.eval()

            # Predict at held-out h
            e_exact, gap = self.exact_ground_state(topology, n_qubits, h_held, model=model)
            lattice_t = make_lattice(topology, n_qubits, J=1.0, h=h_held)
            H_t = spec.build_hamiltonian(lattice_t, **spec.hamiltonian_kwargs)
            circuit_t, _ = spec.create_circuit(n_qubits, p_layers, lattice_t, **spec.circuit_kwargs)

            edge_index_np, coord = builder.build_graph_data(lattice_t)
            h_feat = np.full(n_qubits, float(h_held))
            x = torch.tensor(np.stack([h_feat, coord.astype(float)], axis=1), dtype=torch.float32)
            edge_index = torch.tensor(edge_index_np, dtype=torch.long)
            graph = Data(x=x, edge_index=edge_index)
            graph.batch = torch.zeros(n_qubits, dtype=torch.long)

            with torch.no_grad():
                theta_pred = fold_model(graph).numpy().flatten()

            e_pred = float(backend.evaluate(circuit_t, H_t, theta_pred))
            de_gap = abs(e_pred - e_exact) / max(gap, 1e-10)
            passed = de_gap < de_gap_threshold

            per_fold.append(
                {
                    "fold_idx": fold_idx,
                    "h_held_out": h_held,
                    "n_train_points": len(h_fold),
                    "e_exact": e_exact,
                    "gap": gap,
                    "e_pred": e_pred,
                    "de_gap": de_gap,
                    "fold_train_mse": fold_train["final_mse"],
                    "pass": passed,
                }
            )
            logger.info(
                f"  Fold {fold_idx:2d} h={h_held:.3f}: "
                f"ΔE/gap={de_gap:.4f} mse={fold_train['final_mse']:.2e} "
                f"[{'PASS' if passed else 'FAIL'}]"
            )

        # ── Summary ─────────────────────────────────────────────────────────
        n_folds = len(per_fold)
        n_pass = sum(f["pass"] for f in per_fold)
        de_gaps = [f["de_gap"] for f in per_fold]
        pass_rate = n_pass / max(n_folds, 1)

        return {
            "per_fold": per_fold,
            "summary": {
                "mean_de_gap": float(np.mean(de_gaps)) if de_gaps else float("nan"),
                "max_de_gap": float(np.max(de_gaps)) if de_gaps else float("nan"),
                "std_de_gap": float(np.std(de_gaps)) if de_gaps else float("nan"),
                "n_pass": n_pass,
                "n_folds": n_folds,
                "pass_rate": pass_rate,
            },
            "full_model_train_mse": full_model_mse,
            "n_train_points_full": n_total,
            "n_params": n_params,
            # Pass criterion: ≥80% of LOO folds pass ΔE/gap threshold
            "pass": pass_rate >= 0.80,
        }

    def mpnn_landscape_quality(
        self,
        topology: str,
        n_qubits: int,
        h_train: list[float],
        h_test: list[float],
        *,
        p_layers: int = 1,
        seed: int = 42,
        n_restarts_vqe: int = 1,
        maxiter_vqe: int = 500,
        mpnn_hidden_dim: int = 64,
        mpnn_epochs: int = 4000,
        mpnn_lr: float = 1e-3,
        mpnn_patience: int = 150,
        model: str = "tfim",
        de_gap_threshold: float = 0.05,
    ) -> dict:
        """Energy landscape quality check: decompose ΔE into circuit vs MPNN error.

        Three energy references at each h_test point:
          - ``E_exact``: exact ground state (theoretical ceiling).
          - ``E(θ_opt)``: best VQE energy (circuit expressibility limit).
          - ``E(θ_pred)``: MPNN-predicted energy.

        This decomposition answers:
          - ``error_circuit = |E(θ_opt) - E_exact| / gap``
            → physics / ansatz limit (cannot be improved by ML)
          - ``error_mpnn = |E(θ_pred) - E(θ_opt)| / gap``
            → pure ML prediction error (improvable by more data / better arch)
          - ``error_total = |E(θ_pred) - E_exact| / gap``
            → combined error seen in deployment

        Also computes ``θ_pred_deviation = ||θ_pred - θ_opt||₂`` and
        ``landscape_curvature = ∂²E/∂θ²`` around θ_opt, which indicates
        how sensitive the energy is to parameter errors (flat landscape =
        large θ error tolerance, sharp landscape = precise θ_pred needed).

        Parameters
        ----------
        topology, n_qubits, h_train, h_test, p_layers, seed,
        n_restarts_vqe, maxiter_vqe, mpnn_hidden_dim, mpnn_epochs,
        mpnn_lr, mpnn_patience, model, de_gap_threshold :
            Same as benchmark_mpnn_warmstart.

        Returns
        -------
        dict with keys:
            ``per_h``: per-h decomposition with circuit/mpnn/total errors.
            ``summary``: mean errors, mean curvature, fraction circuit-limited.
            ``mpnn_train_mse``: MPNN training MSE.
            ``pass``: True if mean error_total < de_gap_threshold.
        """
        import numpy as np
        import torch
        from torch_geometric.data import Data

        from qmbp_simulation import make_lattice
        from qmbp_simulation.execution import NoiselessBackend
        from qmbp_simulation.models import HamiltonianBuilder
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn

        spec = get_model_spec(model)
        _resolved = self._resolve_backend()
        backend = _resolved if _resolved is not None else NoiselessBackend()
        builder = HamiltonianBuilder()

        # ── VQE sweep on training grid ───────────────────────────────────────
        theta_map = self.vqe_descending_sweep(
            topology,
            n_qubits,
            h_train,
            seed=seed,
            p_layers=p_layers,
            n_restarts=n_restarts_vqe,
            maxiter=maxiter_vqe,
            model=model,
        )
        h_arr = np.array(sorted(theta_map.keys(), reverse=True))
        theta_arr = np.array([theta_map[h] for h in h_arr])
        e_arr = np.array(
            [self.exact_ground_state(topology, n_qubits, float(h), model=model)[0] for h in h_arr]
        )
        n_params = theta_arr.shape[1]

        # ── Train MPNN ───────────────────────────────────────────────────────
        lattice_ref = make_lattice(topology, n_qubits, J=1.0, h=float(h_arr[0]))
        dataset = build_graph_dataset(
            lattice_ref,
            h_values=h_arr,
            theta_opt=theta_arr,
            e_exact=e_arr,
            fidelity_threshold=0.0,  # noqa: noiseless VQE data — no filtering needed
        )
        predictor = MPNNPredictor(
            node_features=dataset[0].x.shape[1],
            output_dim=n_params,
            hidden_dim=mpnn_hidden_dim,
        )
        train_result = train_mpnn(
            predictor,
            dataset,
            n_epochs=mpnn_epochs,
            lr=mpnn_lr,
            patience=mpnn_patience,
            seed=seed,
        )
        predictor.eval()

        # ── Evaluate at each h_test ──────────────────────────────────────────
        # Also run VQE at h_test to get θ_opt (the circuit-expressibility ceiling)
        theta_map_test = self.vqe_descending_sweep(
            topology,
            n_qubits,
            h_test,
            seed=seed,
            p_layers=p_layers,
            n_restarts=n_restarts_vqe,
            maxiter=maxiter_vqe,
            model=model,
        )

        per_h: list[dict] = []
        for h_t in h_test:
            e_exact, gap = self.exact_ground_state(topology, n_qubits, h_t, model=model)
            lattice_t = make_lattice(topology, n_qubits, J=1.0, h=h_t)
            H_t = spec.build_hamiltonian(lattice_t, **spec.hamiltonian_kwargs)
            circuit_t, _ = spec.create_circuit(n_qubits, p_layers, lattice_t, **spec.circuit_kwargs)

            # θ_opt at h_test from VQE (circuit expressibility ceiling)
            theta_opt_test = theta_map_test[h_t]
            e_opt = float(backend.evaluate(circuit_t, H_t, theta_opt_test))

            # MPNN prediction
            edge_index_np, coord = builder.build_graph_data(lattice_t)
            h_feat = np.full(n_qubits, float(h_t))
            x = torch.tensor(np.stack([h_feat, coord.astype(float)], axis=1), dtype=torch.float32)
            edge_index_t = torch.tensor(edge_index_np, dtype=torch.long)
            graph = Data(x=x, edge_index=edge_index_t)
            graph.batch = torch.zeros(n_qubits, dtype=torch.long)
            with torch.no_grad():
                theta_pred = predictor(graph).numpy().flatten()

            e_pred = float(backend.evaluate(circuit_t, H_t, theta_pred))

            # Error decomposition (all normalized by gap)
            gap_safe = max(gap, 1e-10)
            error_circuit = abs(e_opt - e_exact) / gap_safe
            error_mpnn = abs(e_pred - e_opt) / gap_safe
            error_total = abs(e_pred - e_exact) / gap_safe
            # θ deviation: how far MPNN prediction is from optimal parameters
            theta_deviation = float(np.linalg.norm(theta_pred - theta_opt_test))

            # Landscape curvature around θ_opt via finite differences:
            # average |∂²E/∂θᵢ²| ≈ (E(θ+ε) - 2E(θ) + E(θ-ε)) / ε²
            eps = 0.01
            e_center = e_opt
            curvatures = []
            for i in range(n_params):
                th_p = theta_opt_test.copy()
                th_p[i] += eps
                th_m = theta_opt_test.copy()
                th_m[i] -= eps
                e_p = float(backend.evaluate(circuit_t, H_t, th_p))
                e_m = float(backend.evaluate(circuit_t, H_t, th_m))
                curvatures.append(abs(e_p - 2.0 * e_center + e_m) / (eps**2))
            mean_curvature = float(np.mean(curvatures))
            max_curvature = float(np.max(curvatures))

            passed = error_total < de_gap_threshold
            per_h.append(
                {
                    "h": h_t,
                    "e_exact": e_exact,
                    "e_opt": e_opt,
                    "e_pred": e_pred,
                    "gap": gap,
                    "error_circuit": error_circuit,  # ansatz expressibility limit
                    "error_mpnn": error_mpnn,  # pure ML error
                    "error_total": error_total,  # combined deployment error
                    "theta_deviation": theta_deviation,
                    "mean_curvature": mean_curvature,
                    "max_curvature": max_curvature,
                    "circuit_limited": error_circuit > de_gap_threshold,
                    "pass": passed,
                }
            )
            logger.info(
                f"  h={h_t:.3f}: "
                f"ΔE_circuit={error_circuit:.4f}, "
                f"ΔE_mpnn={error_mpnn:.4f}, "
                f"ΔE_total={error_total:.4f}, "
                f"||Δθ||={theta_deviation:.4f}, "
                f"κ={mean_curvature:.2f} "
                f"[{'PASS' if passed else 'FAIL'}]"
            )

        # ── Summary ─────────────────────────────────────────────────────────
        n_circuit_limited = sum(r["circuit_limited"] for r in per_h)
        summary = {
            "mean_error_circuit": float(np.mean([r["error_circuit"] for r in per_h])),
            "mean_error_mpnn": float(np.mean([r["error_mpnn"] for r in per_h])),
            "mean_error_total": float(np.mean([r["error_total"] for r in per_h])),
            "mean_theta_deviation": float(np.mean([r["theta_deviation"] for r in per_h])),
            "mean_curvature": float(np.mean([r["mean_curvature"] for r in per_h])),
            "n_circuit_limited": f"{n_circuit_limited}/{len(per_h)}",
            "mpnn_fraction_of_total_error": float(
                np.mean([r["error_mpnn"] / max(r["error_total"], 1e-10) for r in per_h])
            ),
        }

        mean_total = summary["mean_error_total"]
        logger.info(
            f"  Landscape quality: "
            f"circuit={summary['mean_error_circuit']:.4f}, "
            f"mpnn={summary['mean_error_mpnn']:.4f}, "
            f"total={mean_total:.4f}, "
            f"circuit-limited={n_circuit_limited}/{len(per_h)}"
        )

        return {
            "per_h": per_h,
            "summary": summary,
            "mpnn_train_mse": train_result["final_mse"],
            "n_train_points": len(dataset),
            "n_params": n_params,
            "pass": mean_total < de_gap_threshold,
        }

    def mpnn_interpolation_extrapolation(
        self,
        topology: str,
        n_qubits: int,
        h_train: list[float],
        h_interpolate: list[float],
        h_extrapolate: list[float],
        *,
        p_layers: int = 1,
        seed: int = 42,
        n_restarts_vqe: int = 1,
        maxiter_vqe: int = 500,
        mpnn_hidden_dim: int = 64,
        mpnn_epochs: int = 4000,
        mpnn_lr: float = 1e-3,
        mpnn_patience: int = 150,
        model: str = "tfim",
        de_gap_threshold: float = 0.05,
    ) -> dict:
        """Explicit interpolation vs extrapolation comparison for MPNN.

        Characterizes how MPNN performance degrades when predicting outside
        the training range (extrapolation) versus inside it (interpolation).

        Definitions:
          - ``h_interpolate``: h-values strictly between min(h_train) and
            max(h_train). MPNN is *interpolating* — should be most accurate.
          - ``h_extrapolate``: h-values outside the training range.
            MPNN is *extrapolating* — expect accuracy degradation.

        Metrics per point:
          - ``de_gap``: ΔE/gap (primary quality metric).
          - ``distance_to_train``: distance to nearest training h-value.
          - ``relative_distance``: distance / mean_training_spacing.
          - ``mode``: "interpolation" or "extrapolation".

        The comparison reveals the MPNN's valid deployment range and
        informs how far beyond the training grid one can safely use the
        model — directly relevant to hardware deployment (h_test may not
        always fall exactly on a training grid point).

        Parameters
        ----------
        topology, n_qubits, h_train, p_layers, seed, n_restarts_vqe,
        maxiter_vqe, mpnn_hidden_dim, mpnn_epochs, mpnn_lr, mpnn_patience,
        model, de_gap_threshold :
            Same as benchmark_mpnn_warmstart.
        h_interpolate : list[float]
            H-values inside the training range to test (interpolation mode).
        h_extrapolate : list[float]
            H-values outside the training range to test (extrapolation mode).

        Returns
        -------
        dict with keys:
            ``interpolation``: per-h results for h_interpolate.
            ``extrapolation``: per-h results for h_extrapolate.
            ``summary``: mean ΔE/gap and pass-rate per mode, degradation factor.
            ``mpnn_train_mse``: MPNN training MSE (reference).
            ``pass``: True if interpolation pass-rate ≥ 0.80.
        """
        import numpy as np
        import torch
        from torch_geometric.data import Data

        from qmbp_simulation import make_lattice
        from qmbp_simulation.execution import NoiselessBackend
        from qmbp_simulation.models import HamiltonianBuilder
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn

        spec = get_model_spec(model)
        _resolved = self._resolve_backend()
        backend = _resolved if _resolved is not None else NoiselessBackend()
        builder = HamiltonianBuilder()

        # ── VQE sweep + MPNN training ────────────────────────────────────────
        theta_map = self.vqe_descending_sweep(
            topology,
            n_qubits,
            h_train,
            seed=seed,
            p_layers=p_layers,
            n_restarts=n_restarts_vqe,
            maxiter=maxiter_vqe,
            model=model,
        )
        h_arr = np.array(sorted(theta_map.keys(), reverse=True))
        theta_arr = np.array([theta_map[h] for h in h_arr])
        e_arr = np.array(
            [self.exact_ground_state(topology, n_qubits, float(h), model=model)[0] for h in h_arr]
        )
        n_params = theta_arr.shape[1]
        h_min_train = float(np.min(h_arr))
        h_max_train = float(np.max(h_arr))
        mean_spacing = float(np.mean(np.abs(np.diff(sorted(h_arr)))))

        # Validate that caller's h_interpolate/h_extrapolate are consistent
        for h in h_interpolate:
            if h < h_min_train or h > h_max_train:
                logger.warning(
                    f"  h_interpolate={h:.3f} is outside training range "
                    f"[{h_min_train:.3f}, {h_max_train:.3f}]. "
                    "Treating as extrapolation."
                )
        for h in h_extrapolate:
            if h_min_train <= h <= h_max_train:
                logger.warning(
                    f"  h_extrapolate={h:.3f} is inside training range "
                    f"[{h_min_train:.3f}, {h_max_train:.3f}]. "
                    "Treating as interpolation."
                )

        lattice_ref = make_lattice(topology, n_qubits, J=1.0, h=float(h_arr[0]))
        dataset = build_graph_dataset(
            lattice_ref,
            h_values=h_arr,
            theta_opt=theta_arr,
            e_exact=e_arr,
            fidelity_threshold=0.0,  # noqa: noiseless VQE data — no filtering needed
        )
        predictor = MPNNPredictor(
            node_features=dataset[0].x.shape[1],
            output_dim=n_params,
            hidden_dim=mpnn_hidden_dim,
        )
        train_result = train_mpnn(
            predictor,
            dataset,
            n_epochs=mpnn_epochs,
            lr=mpnn_lr,
            patience=mpnn_patience,
            seed=seed,
        )
        predictor.eval()

        def _evaluate_at_h(h_t: float, mode: str) -> dict:
            """Predict and evaluate at a single h-value, annotated with mode."""
            e_exact, gap = self.exact_ground_state(topology, n_qubits, h_t, model=model)
            lattice_t = make_lattice(topology, n_qubits, J=1.0, h=h_t)
            H_t = spec.build_hamiltonian(lattice_t, **spec.hamiltonian_kwargs)
            circuit_t, _ = spec.create_circuit(n_qubits, p_layers, lattice_t, **spec.circuit_kwargs)
            edge_index_np, coord = builder.build_graph_data(lattice_t)
            h_feat = np.full(n_qubits, float(h_t))
            x = torch.tensor(np.stack([h_feat, coord.astype(float)], axis=1), dtype=torch.float32)
            edge_index_t = torch.tensor(edge_index_np, dtype=torch.long)
            graph = Data(x=x, edge_index=edge_index_t)
            graph.batch = torch.zeros(n_qubits, dtype=torch.long)
            with torch.no_grad():
                theta_pred = predictor(graph).numpy().flatten()

            e_pred = float(backend.evaluate(circuit_t, H_t, theta_pred))
            de_gap = abs(e_pred - e_exact) / max(gap, 1e-10)

            # Distance to nearest training point
            dist_to_train = float(np.min(np.abs(h_arr - h_t)))
            rel_dist = dist_to_train / max(mean_spacing, 1e-10)

            return {
                "h": h_t,
                "e_exact": e_exact,
                "e_pred": e_pred,
                "gap": gap,
                "de_gap": de_gap,
                "distance_to_nearest_train": dist_to_train,
                "relative_distance": rel_dist,
                "mode": mode,
                "pass": de_gap < de_gap_threshold,
            }

        logger.info(
            f"  Interp/extrap: {topology} N={n_qubits} "
            f"| train=[{h_min_train:.2f},{h_max_train:.2f}] "
            f"| {len(h_interpolate)} interp + {len(h_extrapolate)} extrap pts"
        )

        interp_results = [_evaluate_at_h(h, "interpolation") for h in h_interpolate]
        extrap_results = [_evaluate_at_h(h, "extrapolation") for h in h_extrapolate]

        for r in interp_results:
            logger.info(
                f"  [INTERP] h={r['h']:.3f}: ΔE/gap={r['de_gap']:.4f} "
                f"(d={r['distance_to_nearest_train']:.3f}, rel={r['relative_distance']:.1f}x) "
                f"[{'PASS' if r['pass'] else 'FAIL'}]"
            )
        for r in extrap_results:
            logger.info(
                f"  [EXTRAP] h={r['h']:.3f}: ΔE/gap={r['de_gap']:.4f} "
                f"(d={r['distance_to_nearest_train']:.3f}, rel={r['relative_distance']:.1f}x) "
                f"[{'PASS' if r['pass'] else 'FAIL'}]"
            )

        # ── Summary ─────────────────────────────────────────────────────────
        interp_de = [r["de_gap"] for r in interp_results] if interp_results else [float("nan")]
        extrap_de = [r["de_gap"] for r in extrap_results] if extrap_results else []
        interp_pass_rate = (
            sum(r["pass"] for r in interp_results) / len(interp_results)
            if interp_results
            else float("nan")
        )
        extrap_pass_rate = (
            sum(r["pass"] for r in extrap_results) / len(extrap_results)
            if extrap_results
            else float("nan")
        )

        # Degradation: ratio of mean extrap error to mean interp error
        mean_interp = float(np.nanmean(interp_de))
        mean_extrap = float(np.nanmean(extrap_de)) if extrap_de else float("nan")
        degradation = mean_extrap / max(mean_interp, 1e-10) if extrap_de else float("nan")

        summary = {
            "interpolation": {
                "mean_de_gap": mean_interp,
                "max_de_gap": float(np.nanmax(interp_de)),
                "pass_rate": interp_pass_rate,
                "n_points": len(interp_results),
            },
            "extrapolation": {
                "mean_de_gap": mean_extrap,
                "max_de_gap": float(np.nanmax(extrap_de)) if extrap_de else float("nan"),
                "pass_rate": extrap_pass_rate,
                "n_points": len(extrap_results),
            },
            "degradation_factor": degradation,
            "h_train_range": [h_min_train, h_max_train],
            "mean_training_spacing": mean_spacing,
        }

        logger.info(
            f"  Interpolation: mean={mean_interp:.4f}, pass={interp_pass_rate:.0%} | "
            f"Extrapolation: mean={mean_extrap:.4f}, pass={extrap_pass_rate:.0%} | "
            f"Degradation={degradation:.2f}x"
        )

        return {
            "interpolation": interp_results,
            "extrapolation": extrap_results,
            "summary": summary,
            "mpnn_train_mse": train_result["final_mse"],
            "n_train_points": len(dataset),
            "n_params": n_params,
            "h_train_range": [h_min_train, h_max_train],
            # Pass: interpolation must work (≥80% pass); extrapolation degradation is informational
            "pass": interp_pass_rate >= 0.80 if interp_results else True,
        }

    # ── Extended MPNN Evaluation Experiments ─────────────────────────────────

    def mpnn_scaling_with_system_size(
        self,
        topology: str,
        system_sizes: list[int],
        h_train: list[float],
        h_test: list[float],
        *,
        p_layers: int = 1,
        p_layers_per_n: dict[int, int] | None = None,
        seed: int = 42,
        n_restarts_vqe: int = 1,
        maxiter_vqe: int = 500,
        n_vqe_restarts_from_pred: int = 3,
        maxiter_refine: int = 150,
        mpnn_hidden_dim: int = 64,
        mpnn_epochs: int = 3000,
        mpnn_lr: float = 1e-3,
        mpnn_patience: int = 150,
        model: str = "tfim",
        de_gap_threshold: float = 0.05,
    ) -> dict:
        """Measure how MPNN warm-start speedup scales with system size N.

        For each N in system_sizes, trains an MPNN on h_train and benchmarks
        it against random init at h_test. Reports speedup(N) to reveal whether
        the GNN advantage grows, shrinks, or stays constant with system size.

        Scientific question:
            Does the warm-start advantage of the GNN scale with N?
            - If speedup(N) increases: GNN is more valuable at larger N.
            - If speedup(N) ≈ constant: advantage is landscape-driven.
            - If speedup(N) decreases: GNN loses value for large systems.

        IMPORTANT — p_layers per N:
            The default `p_layers` applies to ALL N. For hardware-realistic
            comparisons, p=2 at N≥10 exceeds the ZNE threshold (36 CX ≫ 18).
            Use `p_layers_per_n` to set per-N depth, e.g.:
            ``p_layers_per_n={4: 2, 6: 2, 10: 1}``
            If not provided and p_layers=2 is used with N≥10, a warning is emitted.

        Parameters
        ----------
        topology : str
            Lattice topology. Use "chain_1d" for fair N comparison.
        system_sizes : list[int]
            System sizes to test (e.g. [4, 6, 10]).
        p_layers_per_n : dict[int, int] | None
            Per-N override for HVA depth. Overrides `p_layers` for specified N.
            Keys not in this dict fall back to `p_layers`.
        h_train, h_test, p_layers, seed, n_restarts_vqe, maxiter_vqe,
        n_vqe_restarts_from_pred, maxiter_refine, mpnn_hidden_dim,
        mpnn_epochs, mpnn_lr, mpnn_patience, model, de_gap_threshold :
            Same as benchmark_mpnn_warmstart.

        Returns
        -------
        dict with keys:
            ``per_n``: list of per-N results (each is a benchmark_mpnn_warmstart output).
            ``summary``: speedup trend statistics across N.
            ``scaling_trend``: "increasing" | "decreasing" | "flat" based on linear fit slope.
            ``pass``: True if all N pass ΔE/gap threshold.
        """
        import numpy as np

        per_n: list[dict] = []
        _p_override = p_layers_per_n or {}

        for n in system_sizes:
            # Per-N p_layers: hardware constraint p=1 for N≥10 (ZNE limit: ~18 CX)
            n_p = _p_override.get(n, p_layers)
            if n_p == 2 and n >= 10:
                logger.warning(
                    f"  [scaling N={n}] p_layers=2 with N={n} exceeds ZNE threshold "
                    f"(~36 CX > 18 CX). Use p_layers_per_n={{{n}: 1}} for hardware-realistic "
                    f"comparison. Continuing with p=2 for noiseless benchmark."
                )

            # Also adapt h_train and h_test if N requires different valid regime
            # For N≥10 p=1: valid regime h≥1.9 (chain_1d) — use the provided grid as-is
            logger.info(f"  [scaling N={n} p={n_p}] Benchmarking warm-start...")
            result_n = self.benchmark_mpnn_warmstart(
                topology=topology,
                n_qubits=n,
                h_train=h_train,
                h_test=h_test,
                p_layers=n_p,
                seed=seed,
                n_restarts_vqe=n_restarts_vqe,
                maxiter_vqe=maxiter_vqe,
                mpnn_hidden_dim=mpnn_hidden_dim,
                mpnn_epochs=mpnn_epochs,
                mpnn_lr=mpnn_lr,
                mpnn_patience=mpnn_patience,
                n_vqe_restarts_from_pred=n_vqe_restarts_from_pred,
                maxiter_refine=maxiter_refine,
                model=model,
                de_gap_threshold=de_gap_threshold,
            )
            entry = {
                "n_qubits": n,
                "p_layers": n_p,
                "n_params": result_n["n_params"],
                "speedup_vs_random": result_n["summary"]["mean_speedup_vs_random"],
                "speedup_vs_prev_h": result_n["summary"]["mean_speedup_vs_prev_h"],
                "init_de_gap": result_n["summary"]["mean_init_de_gap"],
                "final_de_gap": result_n["summary"]["mean_final_de_gap_mpnn"],
                "train_mse": result_n["mpnn_train_mse"],
                "pass": result_n["pass"],
            }
            per_n.append(entry)
            logger.info(
                f"  N={n} p={n_p}: speedup={entry['speedup_vs_random']:.2f}x, "
                f"n_params={entry['n_params']}, "
                f"init_ΔE/gap={entry['init_de_gap']:.4f} "
                f"[{'PASS' if entry['pass'] else 'FAIL'}]"
            )

        # Fit linear trend: speedup vs N
        ns = np.array([e["n_qubits"] for e in per_n], dtype=float)
        speedups = np.array([e["speedup_vs_random"] for e in per_n])
        slope = float(np.polyfit(ns, speedups, 1)[0]) if len(per_n) >= 2 else 0.0

        if abs(slope) < 0.05:
            trend = "flat"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"

        return {
            "per_n": per_n,
            "system_sizes": system_sizes,
            "p_layers_per_n": _p_override or {"all": p_layers},
            "summary": {
                "mean_speedup": float(np.mean(speedups)),
                "min_speedup": float(np.min(speedups)),
                "max_speedup": float(np.max(speedups)),
                "speedup_slope_per_N": slope,
            },
            "scaling_trend": trend,
            "pass": all(e["pass"] for e in per_n),
        }
        """Measure how MPNN warm-start speedup scales with system size N.

        For each N in system_sizes, trains an MPNN on h_train and benchmarks
        it against random init at h_test. Reports speedup(N) to reveal whether
        the GNN advantage grows, shrinks, or stays constant with system size.

        Scientific question:
            Does the warm-start advantage of the GNN scale with N?
            - If speedup(N) increases: GNN is more valuable at larger N
              (larger parameter spaces benefit more from a good initialization).
            - If speedup(N) ≈ constant: the advantage is landscape-driven,
              not dimensionality-driven.
            - If speedup(N) decreases: GNN loses value for large systems
              (unexpected — would suggest the model doesn't scale).

        Parameters
        ----------
        topology : str
            Lattice topology. Use "chain_1d" for fair N comparison.
        system_sizes : list[int]
            System sizes to test (e.g. [4, 6, 10]).
            Each N produces one independent MPNN + benchmark.
        h_train, h_test, p_layers, seed, n_restarts_vqe, maxiter_vqe,
        n_vqe_restarts_from_pred, maxiter_refine, mpnn_hidden_dim,
        mpnn_epochs, mpnn_lr, mpnn_patience, model, de_gap_threshold :
            Same as benchmark_mpnn_warmstart.

        Returns
        -------
        dict with keys:
            ``per_n``: list of per-N results (each is a benchmark_mpnn_warmstart output).
            ``summary``: speedup trend statistics across N.
            ``scaling_trend``: "increasing" | "decreasing" | "flat" based on linear fit slope.
            ``pass``: True if all N pass ΔE/gap threshold.
        """
        import numpy as np

        per_n: list[dict] = []
        for n in system_sizes:
            logger.info(f"  [scaling N={n}] Benchmarking warm-start...")
            result_n = self.benchmark_mpnn_warmstart(
                topology=topology,
                n_qubits=n,
                h_train=h_train,
                h_test=h_test,
                p_layers=p_layers,
                seed=seed,
                n_restarts_vqe=n_restarts_vqe,
                maxiter_vqe=maxiter_vqe,
                mpnn_hidden_dim=mpnn_hidden_dim,
                mpnn_epochs=mpnn_epochs,
                mpnn_lr=mpnn_lr,
                mpnn_patience=mpnn_patience,
                n_vqe_restarts_from_pred=n_vqe_restarts_from_pred,
                maxiter_refine=maxiter_refine,
                model=model,
                de_gap_threshold=de_gap_threshold,
            )
            entry = {
                "n_qubits": n,
                "n_params": result_n["n_params"],
                "speedup_vs_random": result_n["summary"]["mean_speedup_vs_random"],
                "speedup_vs_prev_h": result_n["summary"]["mean_speedup_vs_prev_h"],
                "init_de_gap": result_n["summary"]["mean_init_de_gap"],
                "final_de_gap": result_n["summary"]["mean_final_de_gap_mpnn"],
                "train_mse": result_n["mpnn_train_mse"],
                "pass": result_n["pass"],
            }
            per_n.append(entry)
            logger.info(
                f"  N={n}: speedup={entry['speedup_vs_random']:.2f}x, "
                f"n_params={entry['n_params']}, "
                f"init_ΔE/gap={entry['init_de_gap']:.4f} "
                f"[{'PASS' if entry['pass'] else 'FAIL'}]"
            )

        # Fit linear trend: speedup vs N
        ns = np.array([e["n_qubits"] for e in per_n], dtype=float)
        speedups = np.array([e["speedup_vs_random"] for e in per_n])
        slope = float(np.polyfit(ns, speedups, 1)[0]) if len(per_n) >= 2 else 0.0

        if abs(slope) < 0.05:
            trend = "flat"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"

        return {
            "per_n": per_n,
            "system_sizes": system_sizes,
            "summary": {
                "mean_speedup": float(np.mean(speedups)),
                "min_speedup": float(np.min(speedups)),
                "max_speedup": float(np.max(speedups)),
                "speedup_slope_per_N": slope,
            },
            "scaling_trend": trend,
            "pass": all(e["pass"] for e in per_n),
        }

    def mpnn_learning_curve(
        self,
        topology: str,
        n_qubits: int,
        h_pool: list[float],
        h_test: list[float],
        *,
        train_sizes: list[int] | None = None,
        p_layers: int = 1,
        seed: int = 42,
        n_restarts_vqe: int = 1,
        maxiter_vqe: int = 500,
        mpnn_hidden_dim: int = 64,
        mpnn_epochs: int = 3000,
        mpnn_lr: float = 1e-3,
        mpnn_patience: int = 150,
        model: str = "tfim",
        de_gap_threshold: float = 0.05,
    ) -> dict:
        """Measure MPNN prediction quality as a function of training set size.

        Answers: "How many h-training points does the GNN need to achieve
        ΔE/gap < 5%?" This is the sample-efficiency curve of the model.

        Protocol:
          1. Run VQE on the full h_pool to get all θ_opt values.
          2. For each train_size k, sample the first k points from h_pool
             (descending order — warm-start preserves trajectory structure).
          3. Train MPNN on k points, evaluate at h_test.
          4. Report ΔE/gap(k) — the learning curve.

        The minimum k where ΔE/gap < 5% is the "critical training size" for
        this system. Below it, the GNN cannot be relied upon for hardware.

        Parameters
        ----------
        topology : str
            Lattice topology.
        n_qubits : int
            System size.
        h_pool : list[float]
            Full available h-grid (descending). Training subsets taken from here.
        h_test : list[float]
            Held-out test points (must NOT overlap h_pool).
        train_sizes : list[int] | None
            Subset sizes to test. Default: [3, 5, 7, 10, len(h_pool)] or similar.
        p_layers, seed, n_restarts_vqe, maxiter_vqe, mpnn_hidden_dim,
        mpnn_epochs, mpnn_lr, mpnn_patience, model, de_gap_threshold :
            Standard MPNN configuration.

        Returns
        -------
        dict with keys:
            ``per_size``: per-k results with k, mean_de_gap, pass_rate, train_mse.
            ``critical_size``: minimum k achieving pass_rate ≥ 0.80.
            ``sample_efficiency``: ΔE/gap slope per additional training point.
            ``pass``: True if full dataset (last entry) passes threshold.
        """
        import numpy as np
        import torch
        from torch_geometric.data import Data

        from qmbp_simulation import make_lattice
        from qmbp_simulation.execution import NoiselessBackend
        from qmbp_simulation.models import HamiltonianBuilder
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn

        spec = get_model_spec(model)
        _resolved = self._resolve_backend()
        backend = _resolved if _resolved is not None else NoiselessBackend()
        builder = HamiltonianBuilder()

        # ── VQE on full pool ────────────────────────────────────────────────
        theta_map = self.vqe_descending_sweep(
            topology,
            n_qubits,
            h_pool,
            seed=seed,
            p_layers=p_layers,
            n_restarts=n_restarts_vqe,
            maxiter=maxiter_vqe,
            model=model,
        )
        h_arr = np.array(sorted(theta_map.keys(), reverse=True))
        theta_arr = np.array([theta_map[h] for h in h_arr])
        e_arr = np.array(
            [self.exact_ground_state(topology, n_qubits, float(h), model=model)[0] for h in h_arr]
        )
        n_params = theta_arr.shape[1]

        lattice_ref = make_lattice(topology, n_qubits, J=1.0, h=float(h_arr[0]))

        if train_sizes is None:
            n_total = len(h_arr)
            # Logarithmic spacing from 3 to n_total
            raw = np.unique(np.round(np.geomspace(3, n_total, num=min(6, n_total - 2))).astype(int))
            train_sizes = [int(k) for k in raw if 3 <= k <= n_total]
            if n_total not in train_sizes:
                train_sizes.append(n_total)

        # Pre-compute test lattices/Hamiltonians/circuits
        test_setups = []
        for h_t in h_test:
            e_exact, gap = self.exact_ground_state(topology, n_qubits, h_t, model=model)
            lattice_t = make_lattice(topology, n_qubits, J=1.0, h=h_t)
            H_t = spec.build_hamiltonian(lattice_t, **spec.hamiltonian_kwargs)
            circuit_t, _ = spec.create_circuit(n_qubits, p_layers, lattice_t, **spec.circuit_kwargs)
            edge_index_np, coord = builder.build_graph_data(lattice_t)
            test_setups.append((h_t, e_exact, gap, H_t, circuit_t, edge_index_np, coord))

        per_size: list[dict] = []
        for k in sorted(train_sizes):
            if k > len(h_arr):
                logger.warning(f"  train_size={k} > pool size {len(h_arr)}, skipping.")
                continue
            if k < 3:
                logger.warning(f"  train_size={k} < 3 (too small for MPNN), skipping.")
                continue

            # Take first k points (descending — preserves trajectory structure)
            h_k = h_arr[:k]
            theta_k = theta_arr[:k]
            e_k = e_arr[:k]

            try:
                dataset_k = build_graph_dataset(
                    lattice_ref,
                    h_values=h_k,
                    theta_opt=theta_k,
                    e_exact=e_k,
                    fidelity_threshold=0.0,  # noqa: noiseless VQE data — no filtering needed
                )
            except ValueError:
                logger.warning(f"  train_size={k}: dataset build failed, skipping.")
                continue

            model_k = MPNNPredictor(
                node_features=dataset_k[0].x.shape[1],
                output_dim=n_params,
                hidden_dim=mpnn_hidden_dim,
            )
            train_k = train_mpnn(
                model_k,
                dataset_k,
                n_epochs=mpnn_epochs,
                lr=mpnn_lr,
                patience=mpnn_patience,
                seed=seed + k,  # distinct seed per size
            )
            model_k.eval()

            # Evaluate at all h_test
            de_gaps_k: list[float] = []
            for h_t, e_exact, gap, H_t, circuit_t, edge_index_np, coord in test_setups:
                h_feat = np.full(n_qubits, float(h_t))
                x = torch.tensor(
                    np.stack([h_feat, coord.astype(float)], axis=1), dtype=torch.float32
                )
                edge_index_t = torch.tensor(edge_index_np, dtype=torch.long)
                graph = Data(x=x, edge_index=edge_index_t)
                graph.batch = torch.zeros(n_qubits, dtype=torch.long)
                with torch.no_grad():
                    theta_pred = model_k(graph).numpy().flatten()
                e_pred = float(backend.evaluate(circuit_t, H_t, theta_pred))
                de_gaps_k.append(abs(e_pred - e_exact) / max(gap, 1e-10))

            mean_de = float(np.mean(de_gaps_k))
            pass_rate = float(np.mean([d < de_gap_threshold for d in de_gaps_k]))
            per_size.append(
                {
                    "train_size": k,
                    "n_params": n_params,
                    "mean_de_gap": mean_de,
                    "max_de_gap": float(np.max(de_gaps_k)),
                    "pass_rate": pass_rate,
                    "train_mse": train_k["final_mse"],
                    "pass": pass_rate >= 0.80,
                }
            )
            logger.info(
                f"  k={k:2d}: mean_ΔE/gap={mean_de:.4f}, "
                f"pass={pass_rate:.0%}, mse={train_k['final_mse']:.2e} "
                f"[{'PASS' if pass_rate >= 0.80 else 'FAIL'}]"
            )

        # Critical training size: minimum k with pass_rate ≥ 0.80
        critical_size: int | None = None
        for entry in sorted(per_size, key=lambda x: x["train_size"]):
            if entry["pass_rate"] >= 0.80:
                critical_size = entry["train_size"]
                break

        # Sample efficiency: slope of mean_de_gap vs train_size (negative = improving)
        ks = np.array([e["train_size"] for e in per_size], dtype=float)
        des = np.array([e["mean_de_gap"] for e in per_size])
        slope = float(np.polyfit(ks, des, 1)[0]) if len(per_size) >= 2 else 0.0

        return {
            "per_size": per_size,
            "train_sizes_tested": [e["train_size"] for e in per_size],
            "h_pool_size": len(h_arr),
            "summary": {
                "critical_size": critical_size,
                "sample_efficiency_slope": slope,  # ΔE/gap reduction per extra training point
                "best_mean_de_gap": float(np.min(des)) if len(des) > 0 else float("nan"),
                "full_dataset_de_gap": float(des[-1]) if len(des) > 0 else float("nan"),
            },
            "pass": per_size[-1]["pass"] if per_size else False,
        }

    def mpnn_topology_transfer(
        self,
        source_topology: str,
        target_topology: str,
        n_qubits: int,
        h_train: list[float],
        h_test: list[float],
        *,
        p_layers: int = 1,
        seed: int = 42,
        n_restarts_vqe: int = 1,
        maxiter_vqe: int = 500,
        mpnn_hidden_dim: int = 64,
        mpnn_epochs: int = 3000,
        mpnn_lr: float = 1e-3,
        mpnn_patience: int = 150,
        model: str = "tfim",
        de_gap_threshold: float = 0.05,
    ) -> dict:
        """Zero-shot topology transfer: train on source, deploy on target.

        Trains the MPNN on source_topology data (same N, same h-values), then
        evaluates it on target_topology — no retraining. Compares against:
          - in-distribution (trained AND tested on target): the performance ceiling
          - zero-shot transfer (trained on source, tested on target): this experiment
          - random init baseline (no MPNN, random θ): the lower bound

        This directly validates the GNN's lattice-agnosticism claim: the
        message passing + global pooling architecture should generalize across
        topologies because it encodes connectivity structure (edge_index), not
        topology identity. A GNN that fails zero-shot transfer would be
        memorizing topology-specific patterns rather than learning physics.

        Parameters
        ----------
        source_topology : str
            Topology used for training (e.g. "chain_1d").
        target_topology : str
            Topology used for evaluation (e.g. "ladder").
        n_qubits : int
            System size (same for both topologies).
        h_train, h_test : list[float]
            Training and test h-grids (same for both conditions).
        p_layers, seed, n_restarts_vqe, maxiter_vqe, mpnn_hidden_dim,
        mpnn_epochs, mpnn_lr, mpnn_patience, model, de_gap_threshold :
            Standard MPNN config.

        Returns
        -------
        dict with keys:
            ``in_distribution``: result when trained on target (performance ceiling).
            ``zero_shot``: result when trained on source, tested on target.
            ``random_baseline``: random-init ΔE/gap on target (lower bound).
            ``transfer_ratio``: zero_shot_de_gap / in_dist_de_gap (1.0 = perfect transfer).
            ``pass``: True if zero_shot mean_de_gap < de_gap_threshold.
        """
        import numpy as np
        import torch
        from torch_geometric.data import Data

        from qmbp_simulation import make_lattice
        from qmbp_simulation.execution import NoiselessBackend
        from qmbp_simulation.models import HamiltonianBuilder
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn

        spec = get_model_spec(model)
        _resolved = self._resolve_backend()
        backend = _resolved if _resolved is not None else NoiselessBackend()
        builder = HamiltonianBuilder()
        rng = np.random.default_rng(seed)

        def _train_on(topo: str) -> tuple[MPNNPredictor, np.ndarray, np.ndarray, int]:
            """Train MPNN on given topology. Returns (predictor, h_arr, theta_arr, n_params)."""
            tmap = self.vqe_descending_sweep(
                topo,
                n_qubits,
                h_train,
                seed=seed,
                p_layers=p_layers,
                n_restarts=n_restarts_vqe,
                maxiter=maxiter_vqe,
                model=model,
            )
            h_a = np.array(sorted(tmap.keys(), reverse=True))
            th_a = np.array([tmap[h] for h in h_a])
            e_a = np.array(
                [self.exact_ground_state(topo, n_qubits, float(h), model=model)[0] for h in h_a]
            )
            lattice_r = make_lattice(topo, n_qubits, J=1.0, h=float(h_a[0]))
            ds = build_graph_dataset(
                lattice_r,
                h_values=h_a,
                theta_opt=th_a,
                e_exact=e_a,
                fidelity_threshold=0.0,  # noqa: noiseless VQE data — no filtering needed
            )
            n_p = th_a.shape[1]
            pred = MPNNPredictor(
                node_features=ds[0].x.shape[1],
                output_dim=n_p,
                hidden_dim=mpnn_hidden_dim,
            )
            train_mpnn(
                pred, ds, n_epochs=mpnn_epochs, lr=mpnn_lr, patience=mpnn_patience, seed=seed
            )
            pred.eval()
            return pred, h_a, th_a, n_p

        def _eval_on(predictor: MPNNPredictor, topo: str, n_params: int) -> list[dict]:
            """Evaluate predictor on target topology at h_test."""
            results = []
            for h_t in h_test:
                e_exact, gap = self.exact_ground_state(topo, n_qubits, h_t, model=model)
                lattice_t = make_lattice(topo, n_qubits, J=1.0, h=h_t)
                H_t = spec.build_hamiltonian(lattice_t, **spec.hamiltonian_kwargs)
                circuit_t, _ = spec.create_circuit(
                    n_qubits, p_layers, lattice_t, **spec.circuit_kwargs
                )
                edge_index_np, coord = builder.build_graph_data(lattice_t)
                h_feat = np.full(n_qubits, float(h_t))
                x = torch.tensor(
                    np.stack([h_feat, coord.astype(float)], axis=1), dtype=torch.float32
                )
                edge_index_t = torch.tensor(edge_index_np, dtype=torch.long)
                graph = Data(x=x, edge_index=edge_index_t)
                graph.batch = torch.zeros(n_qubits, dtype=torch.long)
                with torch.no_grad():
                    theta_pred = predictor(graph).numpy().flatten()
                e_pred = float(backend.evaluate(circuit_t, H_t, theta_pred))
                de_gap = abs(e_pred - e_exact) / max(gap, 1e-10)
                # Random baseline
                theta_rand = rng.uniform(-0.01, 0.01, n_params)
                e_rand = float(backend.evaluate(circuit_t, H_t, theta_rand))
                de_gap_rand = abs(e_rand - e_exact) / max(gap, 1e-10)
                results.append(
                    {
                        "h": h_t,
                        "e_exact": e_exact,
                        "gap": gap,
                        "e_pred": e_pred,
                        "de_gap": de_gap,
                        "de_gap_random": de_gap_rand,
                        "pass": de_gap < de_gap_threshold,
                    }
                )
            return results

        logger.info(f"  Training on {source_topology} (source)...")
        source_pred, _, _, n_params = _train_on(source_topology)

        logger.info(f"  Training on {target_topology} (in-distribution ceiling)...")
        target_pred, _, _, _ = _train_on(target_topology)

        logger.info(f"  Evaluating zero-shot transfer on {target_topology}...")
        zero_shot_results = _eval_on(source_pred, target_topology, n_params)

        logger.info(f"  Evaluating in-distribution on {target_topology}...")
        in_dist_results = _eval_on(target_pred, target_topology, n_params)

        zero_de_gaps = [r["de_gap"] for r in zero_shot_results]
        in_dist_de_gaps = [r["de_gap"] for r in in_dist_results]
        rand_de_gaps = [r["de_gap_random"] for r in zero_shot_results]

        mean_zero = float(np.mean(zero_de_gaps))
        mean_in_dist = float(np.mean(in_dist_de_gaps))
        mean_rand = float(np.mean(rand_de_gaps))
        transfer_ratio = mean_zero / max(mean_in_dist, 1e-10)

        for r in zero_shot_results:
            in_dist_match = next(
                (x for x in in_dist_results if abs(x["h"] - r["h"]) < 1e-9),
                None,
            )
            in_dist_de = in_dist_match["de_gap"] if in_dist_match else float("nan")
            logger.info(
                f"  h={r['h']:.3f}: zero_shot={r['de_gap']:.4f}, "
                f"in_dist={in_dist_de:.4f} "
                f"[{'PASS' if r['pass'] else 'FAIL'}]"
            )

        logger.info(
            f"  Transfer: mean_zero={mean_zero:.4f}, mean_in_dist={mean_in_dist:.4f}, "
            f"ratio={transfer_ratio:.2f}x, random={mean_rand:.4f}"
        )

        return {
            "source_topology": source_topology,
            "target_topology": target_topology,
            "zero_shot": zero_shot_results,
            "in_distribution": in_dist_results,
            "summary": {
                "mean_de_gap_zero_shot": mean_zero,
                "mean_de_gap_in_distribution": mean_in_dist,
                "mean_de_gap_random": mean_rand,
                "transfer_ratio": transfer_ratio,  # 1.0 = perfect, > 1 = worse
                "zero_shot_pass_rate": float(np.mean([r["pass"] for r in zero_shot_results])),
                "in_dist_pass_rate": float(np.mean([r["pass"] for r in in_dist_results])),
            },
            "pass": mean_zero < de_gap_threshold,
        }

    def mpnn_data_efficiency_vs_loo(
        self,
        topology: str,
        n_qubits: int,
        h_pool: list[float],
        *,
        n_seeds: int = 3,
        p_layers: int = 1,
        seed: int = 42,
        n_restarts_vqe: int = 1,
        maxiter_vqe: int = 500,
        mpnn_hidden_dim: int = 64,
        mpnn_epochs: int = 3000,
        mpnn_lr: float = 1e-3,
        mpnn_patience: int = 150,
        model: str = "tfim",
        de_gap_threshold: float = 0.05,
        min_train_size: int = 3,
    ) -> dict:
        """Multi-seed LOO-CV to quantify data efficiency with confidence intervals.

        Runs LOO-CV n_seeds times with different MPNN weight initializations.
        Unlike standard LOO (single seed), this reveals whether the LOO score is
        stable or seed-dependent — an unstable score means the model is too
        sensitive to initialization for the given dataset size.

        Metrics:
          - mean_pass_rate ± std_pass_rate across seeds
          - per-fold mean_de_gap ± std across seeds (which h-values are hard?)
          - coefficient_of_variation = std/mean (low = robust, high = fragile)

        This answers: "Is the LOO-CV pass rate a reliable estimate of
        generalization, or does it depend on random weight initialization?"

        Parameters
        ----------
        topology, n_qubits, h_pool, p_layers, seed, n_restarts_vqe,
        maxiter_vqe, mpnn_hidden_dim, mpnn_epochs, mpnn_lr, mpnn_patience,
        model, de_gap_threshold :
            Same as mpnn_leave_one_out_cv.
        n_seeds : int
            Number of independent MPNN random seeds (default: 3).

        Returns
        -------
        dict with keys:
            ``per_seed``: LOO-CV result for each seed.
            ``summary``: mean/std pass_rate, per-fold stats, CV score.
            ``robust``: True if std_pass_rate < 0.15 (stable across seeds).
            ``pass``: True if mean_pass_rate ≥ 0.80.
        """
        import numpy as np

        seeds = [seed + i * 7 for i in range(n_seeds)]  # deterministic seed sequence
        per_seed: list[dict] = []

        for i, s in enumerate(seeds):
            logger.info(f"  LOO seed {i + 1}/{n_seeds} (seed={s})...")
            result_s = self.mpnn_leave_one_out_cv(
                topology=topology,
                n_qubits=n_qubits,
                h_train=h_pool,
                p_layers=p_layers,
                seed=s,
                n_restarts_vqe=n_restarts_vqe,
                maxiter_vqe=maxiter_vqe,
                mpnn_hidden_dim=mpnn_hidden_dim,
                mpnn_epochs=mpnn_epochs,
                mpnn_lr=mpnn_lr,
                mpnn_patience=mpnn_patience,
                model=model,
                de_gap_threshold=de_gap_threshold,
                min_train_size=min_train_size,
            )
            per_seed.append(
                {
                    "seed": s,
                    "pass_rate": result_s["summary"]["pass_rate"],
                    "mean_de_gap": result_s["summary"]["mean_de_gap"],
                    "max_de_gap": result_s["summary"]["max_de_gap"],
                    "per_fold": result_s["per_fold"],
                }
            )
            logger.info(
                f"    pass_rate={result_s['summary']['pass_rate']:.0%}, "
                f"mean_de={result_s['summary']['mean_de_gap']:.4f}"
            )

        pass_rates = np.array([s["pass_rate"] for s in per_seed])
        mean_de_gaps = np.array([s["mean_de_gap"] for s in per_seed])

        # Per-fold statistics across seeds
        fold_h = [f["h_held_out"] for f in per_seed[0]["per_fold"]] if per_seed else []
        per_fold_stats: list[dict] = []
        for fi, h in enumerate(fold_h):
            fold_de_gaps = [
                s["per_fold"][fi]["de_gap"] for s in per_seed if fi < len(s["per_fold"])
            ]
            per_fold_stats.append(
                {
                    "h": h,
                    "mean_de_gap": float(np.mean(fold_de_gaps)),
                    "std_de_gap": float(np.std(fold_de_gaps)),
                    "cv": float(np.std(fold_de_gaps) / max(np.mean(fold_de_gaps), 1e-10)),
                }
            )

        mean_pr = float(np.mean(pass_rates))
        std_pr = float(np.std(pass_rates))
        cv_pr = std_pr / max(mean_pr, 1e-10)

        return {
            "per_seed": per_seed,
            "per_fold_stats": per_fold_stats,
            "summary": {
                "mean_pass_rate": mean_pr,
                "std_pass_rate": std_pr,
                "cv_pass_rate": cv_pr,
                "mean_de_gap": float(np.mean(mean_de_gaps)),
                "std_de_gap": float(np.std(mean_de_gaps)),
                "n_seeds": n_seeds,
            },
            "robust": std_pr < 0.15,
            "pass": mean_pr >= 0.80,
        }

    def mpnn_curvature_noise_correlation(
        self,
        topology: str,
        n_qubits: int,
        h_grid: list[float],
        *,
        p_layers: int = 1,
        seed: int = 42,
        n_restarts_vqe: int = 1,
        maxiter_vqe: int = 500,
        mpnn_hidden_dim: int = 64,
        mpnn_epochs: int = 3000,
        mpnn_lr: float = 1e-3,
        mpnn_patience: int = 150,
        model: str = "tfim",
        noise_levels: list[float] | None = None,
        de_gap_threshold: float = 0.05,
    ) -> dict:
        """Correlate landscape curvature κ with sensitivity to parameter perturbations.

        For each h in h_grid, computes:
          1. κ(h) = mean |∂²E/∂θ²| at θ_opt (landscape curvature)
          2. ΔE_noise(h, σ) = E(θ_opt + ε) - E(θ_opt) for ε ~ N(0, σ²)
             averaged over 20 random perturbations at each noise level σ.

        This tests the hypothesis: "High κ predicts high sensitivity to
        parameter errors" — if validated, κ is a cheap hardware-risk proxy
        (compute from VQE data, no QPU needed).

        Scientific value:
          A strong Pearson r between κ(h) and ΔE_noise(h) validates κ as
          a diagnostic for hardware deployment decisions. Before sending
          a job to IBM Torino, compute κ from noiseless VQE data; if
          κ > threshold, use more VQE restarts or tighter convergence.

        Parameters
        ----------
        topology, n_qubits, h_grid, p_layers, seed, n_restarts_vqe,
        maxiter_vqe, mpnn_hidden_dim, mpnn_epochs, mpnn_lr, mpnn_patience,
        model, de_gap_threshold :
            Standard config.
        noise_levels : list[float] | None
            Standard deviations of Gaussian noise applied to θ_opt.
            Default: [0.01, 0.05, 0.10, 0.20].

        Returns
        -------
        dict with keys:
            ``per_h``: per-h curvature + noise sensitivity at each σ.
            ``correlations``: Pearson r(κ, ΔE_noise) per noise level.
            ``summary``: mean κ, mean Pearson r, κ is a reliable predictor (bool).
            ``pass``: True if mean Pearson r ≥ 0.70 (κ predicts noise sensitivity).
        """
        import numpy as np

        if noise_levels is None:
            noise_levels = [0.01, 0.05, 0.10, 0.20]

        spec_obj = None
        try:
            from qmbp_simulation.models.model_registry import get_model_spec

            spec_obj = get_model_spec(model)
        except Exception as exc:
            raise RuntimeError(
                f"mpnn_curvature_noise_correlation: failed to load model spec for "
                f"'{model}'. Ensure the model is registered."
            ) from exc

        from qmbp_simulation import make_lattice
        from qmbp_simulation.execution import NoiselessBackend

        _resolved = self._resolve_backend()
        backend = _resolved if _resolved is not None else NoiselessBackend()
        rng = np.random.default_rng(seed)
        N_PERTURBATIONS = 20

        # ── VQE sweep ──────────────────────────────────────────────────────
        theta_map = self.vqe_descending_sweep(
            topology,
            n_qubits,
            h_grid,
            seed=seed,
            p_layers=p_layers,
            n_restarts=n_restarts_vqe,
            maxiter=maxiter_vqe,
            model=model,
        )

        per_h: list[dict] = []
        for h in sorted(theta_map.keys(), reverse=True):
            theta_opt = theta_map[h]
            n_params = len(theta_opt)
            e_exact, gap = self.exact_ground_state(topology, n_qubits, float(h), model=model)
            lattice_h = make_lattice(topology, n_qubits, J=1.0, h=float(h))
            H_h = spec_obj.build_hamiltonian(lattice_h, **spec_obj.hamiltonian_kwargs)
            circuit_h, _ = spec_obj.create_circuit(
                n_qubits, p_layers, lattice_h, **spec_obj.circuit_kwargs
            )

            def _energy(theta: np.ndarray) -> float:
                return float(backend.evaluate(circuit_h, H_h, theta))

            e_opt = _energy(theta_opt)

            # ── Curvature: mean |∂²E/∂θᵢ²| via finite differences ────────
            # Guarded per-parameter: a single eval failure yields nan for that param.
            eps = 0.01
            curvatures = []
            for i in range(n_params):
                try:
                    th_p = theta_opt.copy()
                    th_p[i] += eps
                    th_m = theta_opt.copy()
                    th_m[i] -= eps
                    curv_i = abs(_energy(th_p) - 2 * e_opt + _energy(th_m)) / (eps**2)
                except Exception as exc_curv:
                    logger.warning(
                        f"  h={h:.3f}, param {i}: curvature eval failed ({exc_curv}), using nan"
                    )
                    curv_i = float("nan")
                curvatures.append(curv_i)
            kappa = float(np.nanmean(curvatures))

            # ── Noise sensitivity: mean ΔE over random perturbations ──────
            noise_sensitivity: dict[float, float] = {}
            for sigma in noise_levels:
                de_list = []
                for _ in range(N_PERTURBATIONS):
                    try:
                        eps_vec = rng.normal(0, sigma, n_params)
                        e_perturbed = _energy(theta_opt + eps_vec)
                        de_list.append(abs(e_perturbed - e_opt) / max(gap, 1e-10))
                    except Exception:
                        de_list.append(float("nan"))
                noise_sensitivity[sigma] = float(np.nanmean(de_list))

            per_h.append(
                {
                    "h": float(h),
                    "kappa": kappa,
                    "e_opt": e_opt,
                    "e_exact": e_exact,
                    "gap": gap,
                    "de_gap_opt": abs(e_opt - e_exact) / max(gap, 1e-10),
                    "noise_sensitivity": {str(s): v for s, v in noise_sensitivity.items()},
                }
            )
            logger.info(
                f"  h={h:.3f}: κ={kappa:.2f}, "
                + ", ".join(f"σ={s}→{noise_sensitivity[s]:.4f}" for s in noise_levels)
            )

        # ── Pearson r(κ, ΔE_noise) per noise level ───────────────────────
        kappas = np.array([r["kappa"] for r in per_h])
        correlations: dict[str, float] = {}
        for sigma in noise_levels:
            sensitivities = np.array([r["noise_sensitivity"][str(sigma)] for r in per_h])
            if len(kappas) >= 3:
                r_val = float(np.corrcoef(kappas, sensitivities)[0, 1])
            else:
                r_val = float("nan")
            correlations[str(sigma)] = r_val
            logger.info(f"  Pearson r(κ, ΔE_noise@σ={sigma}): {r_val:.4f}")

        mean_r = float(np.nanmean(list(correlations.values())))

        return {
            "per_h": per_h,
            "correlations": correlations,
            "noise_levels": noise_levels,
            "summary": {
                "mean_kappa": float(np.mean(kappas)),
                "max_kappa": float(np.max(kappas)),
                "mean_pearson_r": mean_r,
                "kappa_is_reliable_predictor": abs(mean_r) >= 0.70,
            },
            "pass": abs(mean_r) >= 0.70,
        }

    @staticmethod
    def validate_vqe_results(
        vqe_results: list,
        exact_data: list | None = None,
        *,
        lattice=None,
        model_name: str = "tfim",
        strict: bool = False,
    ) -> dict:
        """Validate VQE results using VQEValidator.

        Convenience method for experiment scripts. Runs comprehensive
        validation on a VQE sweep (variational principle, energy bounds,
        convergence, etc.) and returns the report dict.

        Parameters
        ----------
        vqe_results : list[VQEResult]
            VQE optimization results from a descending sweep.
        exact_data : list[GroundTruthResult] | None
            Exact references for variational principle and energy checks.
        lattice : LatticeConfig | None
            Lattice config for energy bound computation. If None,
            bounds are inferred from the first exact result.
        model_name : str
            Hamiltonian model name (default: "tfim").
        strict : bool
            If True, raises ValueError on CRITICAL issues.

        Returns
        -------
        dict
            JSON-serializable validation report with keys:
            "passed", "n_critical", "n_warnings", "n_info",
            "sweep_metrics", "issues".
        """
        from qmbp_simulation.analysis.vqe_validator import VQEValidator

        if lattice is not None:
            validator = VQEValidator.from_lattice(lattice, model_name=model_name, strict=strict)
        else:
            # Infer from first VQE result dimensions
            n_qubits = len(vqe_results[0].theta_opt) if vqe_results else 4
            # Rough edge count estimate from theta dimension
            n_edges = max(n_qubits - 1, 1)
            validator = VQEValidator(
                n_qubits=n_qubits,
                n_edges=n_edges,
                model_name=model_name,
                strict=strict,
            )

        report = validator.validate_sweep(vqe_results, exact_data)
        return report.to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# ExperimentRunner — Simple BaseExperiment lifecycle wrapper
# ═══════════════════════════════════════════════════════════════════════════════


class ExperimentRunner(ABC):
    """Base class for simple runners that wrap a BaseExperiment subclass.

    Enforces:
    - Preflight validation via BaseExperiment.execute()'s built-in preflight.
    - Structured logging (delegated to BaseExperiment).
    - Standardized result saving (delegated to BaseExperiment).
    - CLI for config overrides (seeds, n_qubits, topology, etc.).
    - Non-zero exit code on failure.

    Subclasses must define:
    - runner_id: Unique identifier.
    - get_experiment_class(): Returns the BaseExperiment subclass (lazy import).

    Subclasses may override:
    - build_config(): Customize the ExperimentConfig before execution.
    - post_execute(): Hook for post-analysis (e.g., printing extra tables).

    Usage::

        class RunE4b(ExperimentRunner):
            runner_id = "run_e4b"

            def get_experiment_class(self):
                from experiments.generalization.exp_e4b import ExperimentE4b
                return ExperimentE4b

        if __name__ == "__main__":
            RunE4b.main()
    """

    runner_id: str = ""

    def __init__(self, args: argparse.Namespace | None = None):
        self._args = args or self._parse_args()
        self._setup_logging()

    @abstractmethod
    def get_experiment_class(self) -> type:
        """Return the BaseExperiment subclass to instantiate.

        Use lazy imports here (not at module level) so that:
        - Preflight failures don't require loading heavy dependencies.
        - Import errors are caught with a clear message.
        """
        ...

    def build_config(self):
        """Build or customize the ExperimentConfig.

        Override to apply CLI arg overrides or custom configuration.
        Default calls experiment_class.default_config() and applies
        standard CLI overrides (--n-qubits, --seeds).
        """
        cls = self.get_experiment_class()

        if not hasattr(cls, "default_config"):
            raise AttributeError(
                f"{cls.__name__} does not implement default_config(). "
                f"All BaseExperiment subclasses must define this classmethod."
            )

        config = cls.default_config()

        # Apply CLI overrides if provided
        if hasattr(self._args, "n_qubits") and self._args.n_qubits is not None:
            config.system.n_qubits = self._args.n_qubits
        if hasattr(self._args, "seeds") and self._args.seeds is not None:
            config.seeds = self._args.seeds
        if hasattr(self._args, "topology") and self._args.topology is not None:
            config.system.topology = self._args.topology

        return config

    def post_execute(self, analysis: dict[str, Any]) -> None:
        """Hook called after execute() completes successfully.

        Override for custom post-processing (additional tables, plots, etc.).
        """

    def run(self) -> int:
        """Execute the full lifecycle: config → instantiate → execute.

        Returns
        -------
        int
            Exit code: 0 on success, 1 on failure.
        """
        logger.info(f"Runner: {self.runner_id}")

        # Lazy import with clear error message
        try:
            exp_cls = self.get_experiment_class()
        except (ImportError, ModuleNotFoundError) as e:
            logger.error(f"Failed to import experiment class: {e}")
            logger.error("Check that the experiment module exists and dependencies are installed.")
            return 1

        logger.info(f"  Experiment class: {exp_cls.__name__}")
        logger.info("  Building config...")

        try:
            config = self.build_config()
        except (AttributeError, TypeError) as e:
            logger.error(f"Failed to build config: {e}")
            return 1

        exp = exp_cls(config)

        logger.info(f"  Executing {config.experiment_id}...")
        skip_preflight = getattr(self._args, "skip_preflight", False)

        try:
            analysis = exp.execute(skip_preflight=skip_preflight)
        except ValueError as e:
            # Preflight validation errors raise ValueError
            logger.error(f"Execution failed: {e}")
            return 1
        except Exception as e:
            logger.error(f"Execution failed with {type(e).__name__}: {e}")
            if self._args.verbose:
                traceback.print_exc()
            return 1

        self.post_execute(analysis)

        # Determine exit code from analysis
        summary = analysis.get("summary", {})
        if "error" in summary:
            return 1
        return 0

    @classmethod
    def main(cls) -> None:
        """Standard entry point with proper exit code."""
        runner = cls()
        exit_code = runner.run()
        sys.exit(exit_code)

    def _setup_logging(self) -> None:
        level = logging.DEBUG if getattr(self._args, "verbose", False) else logging.INFO
        logging.basicConfig(level=level, format="%(levelname)s: %(message)s", force=True)

    @classmethod
    def _parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=f"Run {cls.runner_id}",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument("--n-qubits", type=int, default=None, help="Override system size")
        parser.add_argument("--topology", type=str, default=None, help="Override topology")
        parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Override seeds")
        parser.add_argument("--skip-preflight", action="store_true", help="Skip preflight checks")
        parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
        cls._add_custom_args(parser)
        return parser.parse_args()

    @classmethod
    def _add_custom_args(cls, parser: argparse.ArgumentParser) -> None:
        """Override to add experiment-specific CLI arguments."""


# ═══════════════════════════════════════════════════════════════════════════════
# VariantPipelineRunner — Wrapper for variant runner scripts
# ═══════════════════════════════════════════════════════════════════════════════


class VariantPipelineRunner(ABC):
    """Base class for pipeline variant runner scripts.

    Wraps the existing VariantRunner/run_variant_script infrastructure with
    mandatory preflight validation before execution. The n_qubits used for
    preflight is synchronized with what run_variant_script will actually use
    (respects --n-qubits CLI override).

    Enforces:
    - Preflight via PreflightChecker on all variant specs (at actual n_qubits).
    - Consistent CLI handling via run_variant_script().
    - Result logging through VariantRunner's built-in execution log.
    - --skip-preflight flag to bypass validation when needed.

    Subclasses must define:
    - runner_id: Unique identifier.
    - topology: Default topology (or "multi" for cross-topology).
    - default_n_qubits: Default system size.
    - build_noiseless_variants(): Return noiseless PipelineVariant list.

    Subclasses may override:
    - build_noisy_variants(): Return noisy PipelineVariant list.
    - build_extended_variants(): Return extended PipelineVariant list.
    - timeout: Per-variant timeout.

    Usage::

        class RunP1Variants(VariantPipelineRunner):
            runner_id = "p1_pipeline"
            topology = "multi"
            default_n_qubits = 10
            timeout = 1300

            def build_noiseless_variants(self, n_qubits):
                return [PipelineVariant(...), ...]

        if __name__ == "__main__":
            RunP1Variants.main()
    """

    runner_id: str = ""
    topology: str = "chain_1d"
    default_n_qubits: int = 10
    timeout: int = 1200

    @abstractmethod
    def build_noiseless_variants(self, n_qubits: int) -> list:
        """Build noiseless pipeline variants."""
        ...

    def build_noisy_variants(self, n_qubits: int) -> list:
        """Build noisy pipeline variants (default: empty)."""
        return []

    def build_extended_variants(self, n_qubits: int) -> list:
        """Build extended pipeline variants (default: empty)."""
        return []

    def run(self) -> None:
        """Execute with preflight validation + run_variant_script.

        Preflight uses the ACTUAL n_qubits that will be passed to variants
        (from CLI --n-qubits or default_n_qubits).
        """
        from qmbp_simulation.framework.preflight import (
            PreflightChecker,
            specs_from_pipeline_variants,
        )
        from qmbp_simulation.framework.variant_runner import (
            run_variant_script,
        )

        # Parse CLI to determine actual n_qubits (same logic as run_variant_script)
        # We peek at sys.argv to check for --n-qubits without consuming args
        actual_n_qubits = self._resolve_n_qubits()
        skip_preflight = "--skip-preflight" in sys.argv

        # Build all variants at the actual n_qubits for preflight
        if not skip_preflight:
            all_variants = (
                self.build_noiseless_variants(actual_n_qubits)
                + self.build_noisy_variants(actual_n_qubits)
                + self.build_extended_variants(actual_n_qubits)
            )

            if all_variants:
                logger.info(
                    f"Preflight: validating {len(all_variants)} variants (N={actual_n_qubits})..."
                )
                try:
                    specs = specs_from_pipeline_variants(all_variants)
                    checker = PreflightChecker(specs)
                    report = checker.run_all()
                    report.print_summary()

                    if report.has_errors:
                        logger.error("Preflight FAILED. Fix errors before running.")
                        logger.error("Add --skip-preflight to bypass (not recommended).")
                        sys.exit(1)
                except Exception as e:
                    logger.warning(f"Preflight could not complete: {e}")
                    logger.warning("Proceeding with execution (preflight non-blocking).")
        else:
            logger.warning("Preflight SKIPPED (--skip-preflight flag)")

        # Delegate to standard variant runner (handles its own CLI parsing)
        run_variant_script(
            topology=self.topology,
            default_n_qubits=self.default_n_qubits,
            build_noiseless=self.build_noiseless_variants,
            build_noisy=self.build_noisy_variants,
            build_extended=self.build_extended_variants,
            timeout=self.timeout,
        )

    def _resolve_n_qubits(self) -> int:
        """Resolve actual n_qubits from CLI args without consuming them.

        Peeks at sys.argv for --n-qubits flag. Falls back to default_n_qubits.
        """
        try:
            idx = sys.argv.index("--n-qubits")
            if idx + 1 < len(sys.argv):
                return int(sys.argv[idx + 1])
        except (ValueError, IndexError):
            pass
        return self.default_n_qubits

    @classmethod
    def main(cls) -> None:
        """Standard entry point."""
        runner = cls()
        runner.run()


# ═══════════════════════════════════════════════════════════════════════════════
# HardwareValidationRunner — ValidationRunner with hardware integration
# ═══════════════════════════════════════════════════════════════════════════════


class HardwareValidationRunner(ValidationRunner):
    """ValidationRunner with integrated HardwareBackend support.

    Extends ValidationRunner with:
    - Hardware preflight (QPU status, calibration, topology) in addition to
      structural preflight (runner_id, sections, hypothesis).
    - Automatic HardwareBackend initialization with configurable mode.
    - Cross-reference between runner results and hardware output directories.
    - Shared StructuredLogger between runner and hardware backend.
    - CLI args for --mode (hardware/fake_backend) and --shots.

    Use this for runners that execute on real QPU or FakeKingston simulation.

    Example::

        class HardwareDeployment(HardwareValidationRunner):
            runner_id = "hw_deploy_n10"
            experiment_id = "HW_DEPLOY"
            description = "IBM Torino deployment validation"
            hypothesis = "ΔE/gap<5% and correct phase at h=3.25"

            def define_sections(self):
                return [
                    Section(id=1, name="Single-point", fn=self.section_single),
                    Section(id=2, name="Multi-h sweep", fn=self.section_sweep),
                ]

            def section_single(self) -> dict:
                result = self.hw_backend.run_deployment(
                    self.circuit, self.H, self.params,
                    h_value=3.25, e_exact=-12.5, gap=0.8,
                )
                return {"delta_e_gap": result.delta_e_gap, "pass": result.verdict == "PASS"}

        if __name__ == "__main__":
            HardwareDeployment.main()
    """

    # Subclass may override these defaults
    default_mode: str = "fake_backend"
    default_shots: int = 16384
    default_n_layouts: int = 3

    def __init__(self, args: argparse.Namespace | None = None):
        super().__init__(args)
        self.hw_backend = None  # Initialized in setup()

    @classmethod
    def _add_custom_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add hardware-specific CLI args."""
        parser.add_argument(
            "--mode",
            choices=["hardware", "fake_backend"],
            default=cls.default_mode,
            help="Execution mode (default: %(default)s)",
        )
        parser.add_argument(
            "--shots",
            type=int,
            default=cls.default_shots,
            help="Shots per circuit (default: %(default)s)",
        )
        parser.add_argument(
            "--n-layouts",
            type=int,
            default=cls.default_n_layouts,
            help="Number of ZNE layouts (default: %(default)s)",
        )
        parser.add_argument(
            "--n-qubits",
            type=int,
            default=10,
            help="Number of qubits (default: %(default)s)",
        )
        parser.add_argument(
            "--topology",
            type=str,
            default="heavy_hex",
            help="Lattice topology (default: %(default)s)",
        )
        parser.add_argument(
            "--zne-amplifier",
            choices=["gate_folding", "pea", "adaptive"],
            default="pea",
            help="ZNE noise amplification strategy (default: %(default)s). "
            "'pea' uses Probabilistic Error Amplification (learns noise model). "
            "'adaptive' tries gate_folding first, falls back to PEA if R²<threshold.",
        )
        parser.add_argument(
            "--zne-noise-factors",
            type=float,
            nargs="+",
            default=None,
            help="ZNE noise amplification factors (default: [1, 3, 5])",
        )
        parser.add_argument(
            "--zne-r2-threshold",
            type=float,
            default=0.90,
            help="R² threshold for adaptive ZNE fallback (default: %(default)s)",
        )
        # Layout optimizer (mapomatic VF2)
        parser.add_argument(
            "--no-mapomatic",
            action="store_true",
            default=False,
            help="Disable mapomatic VF2 layout optimization (use BFS fallback)",
        )
        parser.add_argument(
            "--layout-strategy",
            choices=["lowest_cost", "ces_spread", "hybrid"],
            default="lowest_cost",
            help="Layout selection strategy (default: %(default)s)",
        )

    def build_hardware_config(self):
        """Build HardwareConfig from CLI args.

        Override to customize hardware configuration beyond CLI defaults.

        Returns
        -------
        HardwareConfig
            Configuration for the hardware backend.
        """
        from qmbp_simulation.execution.backends import MitigationOptions
        from qmbp_simulation.execution.hardware import HardwareConfig

        # Build mitigation options with amplifier selection
        zne_noise_factors = getattr(self._args, "zne_noise_factors", None)
        zne_amplifier = getattr(self._args, "zne_amplifier", "pea")
        zne_r2_threshold = getattr(self._args, "zne_r2_threshold", 0.90)

        mitigation = MitigationOptions(
            dd_enabled=True,
            trex_enabled=True,
            twirling_enabled=True,
            zne_enabled=True,
            zne_amplifier=zne_amplifier,
            zne_noise_factors=zne_noise_factors,
            zne_r2_fallback_threshold=zne_r2_threshold,
            num_randomizations=32,
            shots_per_randomization=128,
        )

        return HardwareConfig(
            mode=self._args.mode,
            n_qubits=self._args.n_qubits,
            shots=self._args.shots,
            n_layouts=self._args.n_layouts,
            output_dir=f"results/hardware/{self.runner_id}",
            use_mapomatic=not getattr(self._args, "no_mapomatic", False),
            layout_strategy=getattr(self._args, "layout_strategy", "lowest_cost"),
            mitigation=mitigation,
        )

    def setup(self) -> None:
        """Initialize HardwareBackend + shared infrastructure.

        Override and call super().setup() first to retain hardware init.
        """
        from qmbp_simulation.execution.hardware import HardwareBackend

        hw_config = self.build_hardware_config()
        self.hw_backend = HardwareBackend(config=hw_config)  # type: ignore[assignment, no-redef]

        # Share the runner's StructuredLogger with the backend
        self.hw_backend._logger = self.slog  # type: ignore[attr-defined]

        logger.info(f"  Hardware backend: {hw_config.mode} ({hw_config.backend_name})")
        logger.info(f"  Shots: {hw_config.shots}, Layouts: {hw_config.n_layouts}")

    def run_preflight(self) -> bool:
        """Run structural preflight + hardware preflight (QPU status).

        Structural checks first (fast). Hardware preflight only if mode=hardware
        (avoids unnecessary FakeTorino initialization for structure-only validation).
        """
        # Structural checks (runner_id, sections, hypothesis)
        if not super().run_preflight():
            return False

        # Hardware preflight only in hardware mode (FakeKingston always passes)
        if self._args.mode == "hardware":
            logger.info("  Hardware preflight: checking QPU status...")
            from qmbp_simulation.execution.hardware import HardwareBackend

            hw_config = self.build_hardware_config()
            # Temporary backend just for preflight (avoid full setup cost)
            temp_backend = HardwareBackend(config=hw_config)
            checks = temp_backend.run_preflight()

            if checks.get("abort"):
                reason = checks.get("abort_reason", "Unknown")
                logger.error(f"  Hardware preflight FAILED: {reason}")
                self.slog.log("hw_preflight_failed", data=checks)
                return False

            logger.info("  Hardware preflight PASSED")
            self.slog.log("hw_preflight_passed", data=checks)

        return True

    def build_config(self) -> dict[str, Any]:
        """Build config with hardware-specific fields."""
        config = {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "category": self.experiment_id[0] if self.experiment_id else "",
            "description": self.description,
            "hypothesis": self.hypothesis,
            "system": {
                "n_qubits": self._args.n_qubits,
                "p_layers": 1,
                "topology": self._args.topology,
                "model": "tfim",
            },
            "hardware": {
                "mode": self._args.mode,
                "shots": self._args.shots,
                "n_layouts": self._args.n_layouts,
                "backend": getattr(self._args, "backend", "ibm_kingston"),
                "zne_amplifier": getattr(self._args, "zne_amplifier", "pea"),
                "zne_noise_factors": getattr(self._args, "zne_noise_factors", None),
                "zne_r2_threshold": getattr(self._args, "zne_r2_threshold", 0.90),
            },
            "seeds": [],
        }
        return config

    def _build_envelope(self, total_elapsed: float) -> dict[str, Any]:
        """Build envelope with hardware output cross-reference."""
        envelope = super()._build_envelope(total_elapsed)

        # Add hardware output directory reference
        if self.hw_backend and hasattr(self.hw_backend, "_config"):
            envelope["hardware_output_dir"] = self.hw_backend._config.output_dir

        return envelope

    def validate_transpiled_quality(
        self,
        transpiled_circuit,
        layout: list[int] | None = None,
        *,
        error_budget_abort: float = 0.50,
        error_budget_warn: float = 0.30,
        depth_2q_warn: int = 30,
    ) -> dict[str, Any]:
        """Post-transpilation quality check (call from sections after layout selection).

        Wraps ``validate_transpiled_circuit_quality`` from the hardware preflight
        module, using this runner's backend and logger.

        Parameters
        ----------
        transpiled_circuit : QuantumCircuit
            The transpiled (ISA) circuit for one layout.
        layout : list[int] | None
            Physical qubit indices (from layout_selection.layouts[i]).
        error_budget_abort : float
            Abort threshold for error budget (default: 0.50).
        error_budget_warn : float
            Warning threshold for error budget (default: 0.30).
        depth_2q_warn : int
            Warning threshold for 2Q critical path depth.

        Returns
        -------
        dict[str, Any]
            Quality checks with "abort" boolean.
        """
        from qmbp_simulation.execution.hardware.preflight import (
            validate_transpiled_circuit_quality,
        )

        backend = self.hw_backend.backend if self.hw_backend else None
        if backend is None:
            return {"abort": False, "skipped": True, "reason": "no backend available"}

        return validate_transpiled_circuit_quality(
            transpiled_circuit,
            backend,
            layout=layout,
            logger=self.slog,
            error_budget_abort_threshold=error_budget_abort,
            error_budget_warn_threshold=error_budget_warn,
            depth_2q_warn_threshold=depth_2q_warn,
        )
