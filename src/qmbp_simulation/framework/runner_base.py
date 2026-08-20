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
from datetime import UTC
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
    """A single section of a validation runner."""

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
# Utility: preset defaults application
# ═══════════════════════════════════════════════════════════════════════════════


def _apply_preset_defaults(parser: argparse.ArgumentParser, preset: dict) -> None:
    """Apply preset values as parser defaults (CLI args still override).

    For known physics args, maps preset YAML keys to argparse destinations.
    For any other key in the preset, if a matching argparse dest exists in
    the parser, it will also be applied as a default (auto-extensible).

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser whose defaults to update.
    preset : dict
        Loaded preset configuration dict.
    """
    # Explicit mapping for keys that need transformation (name differs from dest)
    KEY_TO_DEST = {
        "n_qubits": "n_qubits",
        "p_layers": "p_layers",
        "topology": "topology",
        "model": "model",
        "model_params": "model_params",
        "h_min": "h_min",
        "h_max": "h_max",
        "h_points": "h_points",
        "seeds": "seeds",
        "maxiter": "maxiter",
        "n_restarts": "n_restarts",
        "output": "output",
    }

    # Collect all known argparse destinations for auto-extension
    known_dests = {action.dest for action in parser._actions if action.dest != "help"}

    defaults = {}
    for yaml_key, value in preset.items():
        if yaml_key.startswith("_"):
            continue  # Skip internal metadata (_preset_name, _preset_path)
        if value is None:
            continue

        # Check explicit mapping first
        dest = KEY_TO_DEST.get(yaml_key)
        if dest is None:
            # Auto-extension: if yaml_key (with - → _) matches a known dest, use it
            normalized = yaml_key.replace("-", "_")
            if normalized in known_dests:
                dest = normalized

        if dest is None:
            continue

        # Ensure topology is always a list
        if yaml_key == "topology" and isinstance(value, str):
            value = [value]
        defaults[dest] = value

    if defaults:
        parser.set_defaults(**defaults)


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
        # Cache for exact_ground_state results (avoids redundant DMRG/eigsh calls)
        self._gt_cache: dict[tuple, tuple[float, float]] = {}
        # Model provenance tracking (auto-populated by load_best_mpnn_for_cross_n)
        self._model_provenance: dict[str, Any] = {}
        # Artifact collector (register during execution, persist at end)
        from qmbp_simulation.framework.artifact_store import ArtifactCollector

        self.artifacts = ArtifactCollector()

        # Global seed from environment (for multi-seed experiments)
        import os

        env_seed = os.environ.get("QMBP_GLOBAL_SEED")
        if env_seed is not None:
            from qmbp_simulation.utils import set_global_seed

            seed_val = int(env_seed)
            set_global_seed(seed_val)
            logger.info(f"  Global seed set from QMBP_GLOBAL_SEED={seed_val}")

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
        explicitly signal failure.
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
        1. Grid density: warn if 2D parameter space has fewer than 8 points
           per dimension (interpolation will fail).
        2. ZNE budget: p=2 N≥10 exceeds ZNE threshold (36 CX > 18 CX limit).

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

        # 1. Grid density for 2D parameter spaces
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

        # 2. ZNE budget check
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

        # ─── Step 2c: Pre-analysis data integrity check ──────────────────
        # Ensure NPZ e_exact values match GT cache before any analysis.
        # GT cache is the authoritative ground truth source.
        try:
            from qmbp_simulation.analysis.metrics import validate_gt_npz_coherence

            coherence = validate_gt_npz_coherence(fix=True)
            if coherence["n_points_fixed"] > 0:
                logger.info(
                    "Pre-analysis fix: corrected %d stale e_exact points in %d NPZ files",
                    coherence["n_points_fixed"],
                    coherence["n_files_with_issues"],
                )
        except Exception:
            pass  # Non-blocking: continue even if check fails

        # ─── Step 3: Execute sections ────────────────────────────────────
        self._print_header(selected)
        interrupted = False
        _current_section_id = None

        # ── Workaround: Qiskit mimalloc GC deadlock on macOS ARM64 ────────
        # Qiskit's Rust accelerator uses mimalloc which can deadlock during
        # Python GC when freeing CircuitData objects. Disable GC during section
        # execution (re-enabled between sections and always on exit).
        import gc as _gc

        _gc_was_enabled = _gc.isenabled()
        _gc.disable()

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

                # NOTE: GC collect between sections is DISABLED.
                # Qiskit's CircuitData destructor triggers mimalloc's
                # _mi_arenas_page_unabandon → sleep() loop on macOS ARM64.
                # Memory cleanup happens via os._exit() at process end.
                # _gc.enable()
                # _gc.collect()
                # _gc.disable()

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

        # ─── Step 3c: Flush all caches (prevent data loss on os._exit()) ──
        # os._exit() bypasses __del__, so caches relying on destructor flush
        # would lose pending writes. Explicit flush here guarantees persistence.
        # Uses the runner's OWN cache instance (not a new one from disk).
        try:
            _gt = getattr(self, "_disk_gt_cache", None)
            if _gt is not None and _gt._dirty:
                _gt._save()
                logger.debug("  Flushed GroundTruthCache (%d entries)", len(_gt._data))
        except Exception:
            pass  # Non-critical — GT cache is a performance optimization

        # ─── Step 4: Save results + exit ─────────────────────────────────
        # Save result JSON and exit immediately via os._exit() to avoid
        # mimalloc spinlock on macOS ARM64 with Qiskit+PyTorch in memory.
        # The index refresh runs in a detached subprocess after exit.

        # ── Step 3b: Post-evaluation data quality feedback ───────────────
        if not interrupted:
            self._log_data_quality_feedback()

        import os

        total_elapsed = time.time() - t_total
        n_fail = sum(1 for r in self._section_results if not r.success)
        exit_code = 1 if n_fail > 0 or interrupted else 0

        saved_path = None
        try:
            # Check if we're in a state where _build_envelope would spinlock.
            # Indicator: torch is imported AND has live tensors → use minimal save.
            _torch_loaded = "torch" in sys.modules
            if _torch_loaded:
                # Minimal save path: avoid _build_envelope which triggers
                # mimalloc spinlock via Qiskit/PyTorch object interactions.
                # Pre-serialize ALL data to native Python types to avoid
                # json_serialize touching numpy/torch objects during json.dump.
                import json as _json

                from qmbp_simulation.framework.result_io import generate_timestamp
                from qmbp_simulation.utils.helpers import json_serialize

                def _deep_serialize(obj):
                    """Recursively convert numpy/torch to JSON-safe Python."""
                    if isinstance(obj, dict):
                        return {k: _deep_serialize(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        return [_deep_serialize(v) for v in obj]
                    try:
                        return json_serialize(obj)
                    except TypeError:
                        return str(obj)

                config = self.build_config()
                config.setdefault("experiment_id", self.experiment_id)
                results = {}
                for r in self._section_results:
                    results[f"section_{r.section_id}"] = _deep_serialize(
                        {
                            "name": r.name,
                            "success": r.success,
                            "elapsed_s": round(r.elapsed_s, 2),
                            "data": r.data,
                            "error": r.error,
                        }
                    )
                n_pass = sum(1 for r in self._section_results if r.success)
                summary = {
                    "n_sections": len(self._section_results),
                    "n_passed": n_pass,
                    "n_failed": len(self._section_results) - n_pass,
                    "pass_rate": n_pass / max(len(self._section_results), 1),
                    "total_elapsed_s": round(total_elapsed, 2),
                    "all_passed": n_fail == 0,
                    "metric_version": "dual_v1",
                }
                envelope = {
                    "schema_version": "2.0",
                    "timestamp": generate_timestamp(),
                    "config": _deep_serialize(config),
                    "results": results,
                    "summary": summary,
                    "elapsed_s": round(total_elapsed, 2),
                    "model_provenance": _deep_serialize(self._model_provenance) if self._model_provenance else None,
                    "analysis": {
                        "experiment_id": self.experiment_id,
                        "summary": summary,
                    },
                }
                if interrupted:
                    envelope["interrupted"] = True
                    envelope["completed_sections"] = len(self._section_results)
                    envelope["interrupted_section"] = _current_section_id

                # Write directly (bypass save_experiment_result which calls
                # ResultIndex internally and uses json_serialize again)
                from qmbp_simulation.framework.result_io import _DEFAULT_RESULTS_ROOT

                exp_dir = _DEFAULT_RESULTS_ROOT / f"exp_{self.experiment_id.lower()}"
                exp_dir.mkdir(parents=True, exist_ok=True)
                saved_path = exp_dir / f"run_{generate_timestamp()}.json"
                with open(saved_path, "w") as _f:
                    _json.dump(envelope, _f, indent=2, default=str)
                logger.info(f"  Results: {saved_path}")
            else:
                # Normal path: full envelope with diagnostics
                envelope = self._build_envelope(total_elapsed)
                if interrupted:
                    envelope["interrupted"] = True
                    envelope["completed_sections"] = len(self._section_results)
                    envelope["interrupted_section"] = _current_section_id
                saved_path = save_experiment_result(envelope, experiment_id=self.experiment_id)
        except (Exception, TimeoutError):
            # Emergency minimal save
            try:
                import json as _json

                from qmbp_simulation.framework.result_io import generate_timestamp

                edir = Path("results") / "experiments" / f"exp_{self.experiment_id}"
                edir.mkdir(parents=True, exist_ok=True)
                saved_path = edir / f"run_{generate_timestamp()}.json"
                saved_path.write_text(
                    _json.dumps(
                        {
                            "config": self.build_config(),
                            "results": {
                                f"s{r.section_id}": {"pass": r.success, "t": r.elapsed_s}
                                for r in self._section_results
                            },
                            "elapsed_s": total_elapsed,
                        },
                        default=str,
                    )
                )
            except Exception:
                pass

        # Save structured log
        try:
            if saved_path:
                log_path = saved_path.parent / f"log_{saved_path.stem.replace('run_', '')}.json"
                self.slog.save(log_path)
        except Exception:
            pass

        from qmbp_simulation.analysis.metrics import validate_gt_npz_coherence, post_experiment_sync

        validate_gt_npz_coherence(fix=True)
        post_experiment_sync(verbose=False)

        # Print summary to console
        if not interrupted and saved_path:
            self._print_summary(total_elapsed, saved_path)
        elif interrupted:
            logger.info(
                f"\n  Partial results: {len(self._section_results)}/{len(selected)} sections."
            )

        # Flush stdout/stderr before exit
        sys.stdout.flush()
        sys.stderr.flush()

        # Spawn detached index refresh + dashboard generation (fire-and-forget)
        try:
            import subprocess

            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "from qmbp_simulation.analysis.metrics import post_experiment_sync; "
                    "post_experiment_sync(verbose=False); "
                    "from qmbp_simulation.predictors.model_zoo import "
                    "heal_manifest, compute_retrain_queue, _load_manifest, "
                    "_CHECKPOINTS_DIR; "
                    "heal_manifest(dry_run=False); "
                    "queue = compute_retrain_queue(); "
                    "print(f'Retrain queue: {len(queue)} models') if queue else None; "
                    # Auto-evaluate unevaluated models (quick pass_rate from val split)
                    "entries = _load_manifest(); "
                    "unevaluated = [e for e in entries if e.pass_rate == 0.0 "
                    "and e.n_training_points > 50 "
                    "and (_CHECKPOINTS_DIR / e.checkpoint_file).exists()]; "
                    "[print(f'  Unevaluated: {e.checkpoint_file}') for e in unevaluated[:2]]",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
        except Exception:
            pass

        # Spawn module-index regeneration (separate process — non-blocking)
        try:
            import subprocess

            _idx_script = (
                self._get_project_root()
                / "scripts"
                / "general_project_maintenance"
                / "generate_module_index.py"
            )
            if not _idx_script.exists():
                _idx_script = (
                    self._get_project_root()
                    / "scripts"
                    / "maintenance"
                    / "generate_module_index.py"
                )
            if _idx_script.exists():
                subprocess.Popen(
                    [sys.executable, str(_idx_script)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    close_fds=True,
                )
        except Exception:
            pass

        # Hard exit — no interpreter shutdown, no GC, no mimalloc spinlock
        os._exit(exit_code)

    # ── Class-level entry point ──────────────────────────────────────────────

    @classmethod
    def main(cls) -> None:
        """Standard entry point for runner scripts.

        Note: run() calls os._exit() internally after saving results,
        so this method will not return normally.

        Usage in scripts:
            if __name__ == "__main__":
                MyRunner.main()
        """
        runner = cls()
        exit_code = runner.run()
        # Fallback: if run() somehow returns (e.g. dry-run mode)
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
            self._section_results.append(
                SectionResult(
                    section_id=section_id,
                    name=section_data.get("name", key),
                    success=True,
                    elapsed_s=section_data.get("elapsed_s", 0),
                    data=section_data.get("data", {}),
                    error=None,
                )
            )
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
            "metric_version": "dual_v1",
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

        # Add runner traceability (new in 2026-08-10)
        from qmbp_simulation.predictors.model_zoo import get_runner_tag, make_date_tag

        config["runner_id"] = self.runner_id
        config["runner_tag"] = get_runner_tag(self.runner_id)
        config["date_tag"] = make_date_tag()

        # Build envelope using result_io standard
        envelope = build_result_envelope(
            config=config,
            results=results,
            summary=summary,
            elapsed_s=total_elapsed,
            metadata=collect_run_metadata(),
        )

        # Add model provenance for traceability (which MPNN was used)
        if self._model_provenance:
            envelope["model_provenance"] = self._model_provenance

        # Add simulation diagnostics block (documents backend + numerical method)
        try:
            from qmbp_simulation.framework.result_io import build_simulation_diagnostics

            # Find the backend used in this run (check common attribute names).
            # Order: most specific first → general fallback.
            # hw_backend is HardwareBackend (real QPU or FakeKingston mode).
            # _vqe_backend is the VQE evaluation backend (NoiselessBackend or MPSBackend).
            # _backend is used by scaling runners.
            # fake_backend is FakeTorino (used by noisy ZNE runners).
            # noiseless is the fallback (always available after setup_physics()).
            backend = (
                getattr(self, "hw_backend", None)
                or getattr(self, "_vqe_backend", None)
                or getattr(self, "_backend", None)
                or getattr(self, "fake_backend", None)
                or getattr(self, "noiseless", None)
            )
            if backend is not None:
                system = config.get("system", {})
                n_qubits = system.get("n_qubits", 0)
                topology = system.get("topologies", system.get("topology", "unknown"))
                if n_qubits > 0:
                    envelope["simulation_diagnostics"] = build_simulation_diagnostics(
                        backend, n_qubits, topology
                    )
                    # Flag when E_exact reference is approximate (DMRG TFIChain on non-1D)
                    topo_str = topology[0] if isinstance(topology, list) else topology
                    _NON_1D_TOPOS = ("heavy_hex", "ladder", "square", "triangular")
                    if n_qubits > 15 and topo_str in _NON_1D_TOPOS:
                        envelope["simulation_diagnostics"]["e_exact_approximate"] = True
                        envelope["simulation_diagnostics"]["e_exact_warning"] = (
                            f"E_exact for {topo_str} N={n_qubits} uses DMRG with 1D TFIChain model. "
                            f"Non-sequential bonds in {topo_str} are not captured. "
                            f"Small variational violations (<1e-2) are expected and benign."
                        )
        except Exception:
            pass  # simulation_diagnostics is best-effort

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

        # Add baseline comparison: deferred to detached subprocess to avoid
        # mimalloc spinlock from ResultIndex.rebuild() with PyTorch in memory.
        # The baseline_ref field will be added by the index refresh subprocess.

        # Log warnings for anomalous results
        if n_total > 0:
            pass_rate = summary.get("pass_rate", 0)
            if pass_rate == 0 and n_total >= 2:
                logger.warning(
                    "⚠️  ALL sections FAILED (pass_rate=0%%). "
                    "This may indicate a setup error or fundamental issue. "
                    "Check section errors in result JSON."
                )
                self.slog.log("all_sections_failed", data={"n_total": n_total})
            elif pass_rate < 0.5 and n_total >= 3:
                logger.warning(
                    f"⚠️  Low pass rate ({pass_rate * 100:.0f}%%) — "
                    f"only {n_pass}/{n_total} sections passed."
                )

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
        # Artifact persistence
        parser.add_argument(
            "--save-artifacts",
            type=str,
            choices=["never", "always", "on-pass"],
            default="never",
            help="Save run artifacts (MPNN model, circuit, theta). "
            "'never': no artifacts (default). "
            "'always': save regardless of outcome. "
            "'on-pass': save only when all sections pass.",
        )
        # VQE sweep control
        parser.add_argument(
            "--no-bidirectional",
            action="store_true",
            default=False,
            help="Skip the ascending bidirectional pass in VQE sweeps.",
        )
        parser.add_argument(
            "--force-bidirectional",
            action="store_true",
            default=False,
            help="Force the bidirectional pass even for N>=16 "
            "(overrides the automatic skip for large systems).",
        )
        # Config preset (loads YAML, CLI overrides preset values)
        parser.add_argument(
            "--preset",
            type=str,
            default=None,
            help="Load a config preset by name (e.g. 'noiseless/tfim_heavy_hex_n20_p4'). "
            "CLI args override preset values. See configs/presets/ for available presets.",
        )
        # Allow subclasses to add custom args
        cls._add_custom_args(parser)

        # If --preset is specified, inject preset values as defaults before parsing
        # We do a preliminary parse to detect --preset, then set defaults from it
        preliminary, _ = parser.parse_known_args()
        if preliminary.preset:
            from qmbp_simulation.framework.presets import load_preset

            preset = load_preset(preliminary.preset)
            # Convert preset to defaults dict for the parser
            _apply_preset_defaults(parser, preset)

        return parser.parse_args()

    @classmethod
    def _add_custom_args(cls, parser: argparse.ArgumentParser) -> None:
        """Override to add custom CLI arguments.

        Called during argument parsing. Subclasses can add experiment-specific
        flags here without modifying the base class.

        Example::

            @classmethod
            def _add_custom_args(cls, parser):
                cls._add_standard_physics_args(parser)
                parser.add_argument("--g-value", type=float, default=0.3)
        """

    @classmethod
    def _add_standard_physics_args(
        cls,
        parser: argparse.ArgumentParser,
        *,
        n_qubits: int = 6,
        p_layers: int = 2,
        topology: str = "chain_1d",
        model: str = "tfim",
        h_min: float = 0.5,
        h_max: float = 2.0,
        h_points: int = 15,
        seeds: list[int] | None = None,
        maxiter: int = 500,
        n_restarts: int = 5,
    ) -> None:
        """Add standard physics experiment CLI args.

        Provides the common set of arguments shared across most physics
        runners (noiseless, noisy, scaling). Call from _add_custom_args
        to avoid duplicating these definitions in every runner.

        Keyword arguments set the defaults for this specific runner.
        Users can still override any value from the command line.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            Parser to add arguments to.
        n_qubits : int
            Default system size.
        p_layers : int
            Default HVA circuit depth.
        topology : str
            Default lattice topology.
        model : str
            Default Hamiltonian model name.
        h_min : float
            Default minimum transverse field value.
        h_max : float
            Default maximum transverse field value.
        h_points : int
            Default number of h-points in sweep.
        seeds : list[int] | None
            Default random seeds. None → [42, 43, 44].
        maxiter : int
            Default VQE optimizer max iterations per restart.
        n_restarts : int
            Default number of VQE restarts per h-point.
        """
        if seeds is None:
            seeds = [42, 43, 44]

        parser.add_argument(
            "--n-qubits",
            type=int,
            default=n_qubits,
            help="System size (default: %(default)s)",
        )
        parser.add_argument(
            "--p-layers",
            type=int,
            default=p_layers,
            choices=[1, 2, 3, 4, 5, 6, 7, 8],
            help="HVA circuit depth (default: %(default)s)",
        )
        parser.add_argument(
            "--topology",
            type=str,
            nargs="+",
            default=[topology],
            help="Lattice topology(ies) (default: %(default)s)",
        )
        parser.add_argument(
            "--model",
            type=str,
            default=model,
            help="Model from registry: tfim, tfim_longitudinal, tfim_frustrated, "
            "heisenberg, xy, tfim_bond_resolved (default: %(default)s)",
        )
        parser.add_argument(
            "--model-params",
            type=str,
            default=None,
            help="Model-specific parameters as key=value pairs, comma-separated. "
            "E.g. --model-params g=0.3 for tfim_longitudinal, "
            "or --model-params J2=0.5 for tfim_frustrated",
        )
        parser.add_argument(
            "--h-min",
            type=float,
            default=h_min,
            help="Minimum h value (default: %(default)s)",
        )
        parser.add_argument(
            "--h-max",
            type=float,
            default=h_max,
            help="Maximum h value (default: %(default)s)",
        )
        parser.add_argument(
            "--h-points",
            type=int,
            default=h_points,
            help="Number of h-points in sweep (default: %(default)s)",
        )
        parser.add_argument(
            "--seeds",
            type=int,
            nargs="+",
            default=seeds,
            help="Random seeds (default: %(default)s)",
        )
        parser.add_argument(
            "--maxiter",
            type=int,
            default=maxiter,
            help="VQE optimizer maxiter per restart (default: %(default)s)",
        )
        parser.add_argument(
            "--n-restarts",
            type=int,
            default=n_restarts,
            help="VQE restarts per h-point (default: %(default)s)",
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Output directory (default: auto from experiment_id)",
        )

    # ── Reusable utility methods (available to all subclasses) ───────────────

    # ── Reusable quality check + config helpers ────────────────────────────

    def run_quality_check(
        self,
        configs: list[dict] | None = None,
    ) -> dict:
        """Run QualityPredictor for one or more configs.

        Reusable by any runner — either as a section or in run_preflight().
        If no configs provided, auto-detects from self._args.

        Parameters
        ----------
        configs : list[dict] | None
            Each dict has keys: model, topology, n_qubits, p_layers, h_min, h_max.
            If None, builds from self._args automatically.

        Returns
        -------
        dict
            {"pass": bool, "reports": {config_key: {pass_probability, recommendation, ...}}}
        """
        from qmbp_simulation.analysis.quality_predictor import QualityPredictor

        predictor = QualityPredictor()

        if configs is None:
            configs = self._build_quality_check_configs()

        reports = {}
        all_should_run = True
        for cfg in configs:
            key = f"{cfg.get('topology', '?')}_N{cfg.get('n_qubits', '?')}"
            try:
                report = predictor.predict(**cfg)
                reports[key] = {
                    "pass_probability": report.pass_probability,
                    "recommendation": report.recommendation,
                    "confidence": report.confidence,
                    "estimated_h_min": getattr(report, "estimated_h_min", None),
                    "estimated_time_s": getattr(report, "estimated_time_s", None),
                    "should_run": report.should_run if hasattr(report, "should_run") else True,
                }
                logger.info(f"  {key}: {report}")
                if hasattr(report, "should_run") and not report.should_run:
                    all_should_run = False
            except Exception as e:
                reports[key] = {"error": str(e), "should_run": True}
                logger.debug(f"  Quality check failed for {key}: {e}")

        return {"pass": True, "reports": reports, "all_should_run": all_should_run}

    def _build_quality_check_configs(self) -> list[dict]:
        """Auto-build quality check configs from self._args.

        Handles both run_noiseless (--n-qubits, --topology list) and
        run_accelerated (--train-n, --target-n, --topology str) patterns.
        """
        configs = []
        model = getattr(self._args, "model", "tfim")
        h_min = getattr(self._args, "h_min", 0.5)
        h_max = getattr(self._args, "h_max", 3.5)

        # Detect topology: could be list or string
        topo_raw = getattr(self._args, "topology", "chain_1d")
        topos = topo_raw if isinstance(topo_raw, list) else [topo_raw]

        # Detect n_qubits: could be --n-qubits or --train-n / --target-n
        n_values = set()
        if hasattr(self._args, "n_qubits"):
            n_values.add(self._args.n_qubits)
        if hasattr(self._args, "train_n"):
            n_values.add(self._args.train_n)
        if hasattr(self._args, "target_n"):
            targets = self._args.target_n
            if isinstance(targets, list):
                n_values.update(targets)
            else:
                n_values.add(targets)
        if not n_values:
            n_values.add(10)  # fallback

        p = getattr(self._args, "p_layers", 1)
        p_val = p[0] if isinstance(p, list) else p

        for topo in topos:
            for n in sorted(n_values):
                configs.append(
                    {
                        "model": model,
                        "topology": topo,
                        "n_qubits": n,
                        "p_layers": p_val,
                        "h_min": h_min,
                        "h_max": h_max,
                    }
                )
        return configs

    def _build_physics_config(self) -> dict:
        """Build standardized config dict from common self._args attributes.

        Returns a nested dict with the physics configuration that most
        runners share. Subclasses call this in build_config() and extend.

        Returns
        -------
        dict
            {runner_id, experiment_id, topology, h_grid: {}, vqe: {}, ...}
        """
        args = self._args
        config: dict[str, Any] = {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
        }

        # Topology (string or list)
        topo = getattr(args, "topology", None)
        if topo is not None:
            config["topology"] = topo

        # H-grid
        h_min = getattr(args, "h_min", None)
        h_max = getattr(args, "h_max", None)
        h_points = getattr(args, "h_points", None)
        if h_min is not None:
            config["h_range"] = [h_min, h_max]
            config["h_points"] = h_points

        # VQE params
        maxiter = getattr(args, "maxiter", None)
        if maxiter is not None:
            config["maxiter"] = maxiter
            config["n_restarts"] = getattr(args, "n_restarts", None)

        # N-qubits (various naming conventions)
        if hasattr(args, "n_qubits"):
            config["n_qubits"] = args.n_qubits
        if hasattr(args, "train_n"):
            config["train_n"] = args.train_n
        if hasattr(args, "target_n"):
            config["target_n"] = args.target_n

        # p_layers
        p = getattr(args, "p_layers", None)
        if p is not None:
            config["p_layers"] = p

        # Seeds
        seeds = getattr(args, "seeds", None)
        if seeds is not None:
            config["seeds"] = seeds

        return config

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

        Returns
        -------
        ExecutionBackend
            NoiselessBackend or MPSBackend depending on N.
        """
        from qmbp_simulation.execution import select_backend as _select_backend

        return _select_backend(n_qubits, for_vqe_loop=for_vqe_loop)

    def get_cached_backend(
        self,
        topology: str,
        n_qubits: int,
        *,
        model: str = "tfim",
        p_layers: int = 1,
        enabled: bool = True,
    ):
        """Return a CachedBackend wrapping the appropriate backend for N.

        Transparently caches circuit evaluations to avoid recomputing
        identical (topology, N, h, θ) evaluations. Use set_h(h) before
        each evaluate() call to key the cache correctly.

        Parameters
        ----------
        topology : str
            Lattice topology (used in cache key).
        n_qubits : int
            System size (selects Statevector vs MPS backend).
        model : str
            Model name for cache key.
        p_layers : int
            HVA depth for cache key.
        enabled : bool
            If False, returns a passthrough (no caching). Default True.

        Returns
        -------
        CachedBackend
            Backend with transparent evaluation cache.
        """
        from qmbp_simulation.execution.eval_cache import CachedBackend, EvalCache

        backend = self.select_backend(n_qubits)
        cache = EvalCache(enabled=enabled, p_layers=p_layers)
        return CachedBackend(
            backend,
            topology=topology,
            n_qubits=n_qubits,
            model=model,
            p_layers=p_layers,
            cache=cache,
        )

    def get_empirical_h_frontier(
        self,
        topology: str | None = None,
        n_qubits: int | None = None,
        p_layers: int | None = None,
        *,
        model: str = "tfim_bond_resolved",
    ) -> float | None:
        """Get the empirical h-frontier from NPZ training data.

        Returns the h value where ΔE/gap crosses 5% for the given config,
        computed from stored VQE results.  This is the most reliable estimate
        of the pipeline's capability boundary for a specific (topology, N, p).

        Returns None if no NPZ data exists or frontier is indeterminate.

        Parameters
        ----------
        topology : str | None
            Lattice topology. Default: from self._args.topology.
        n_qubits : int | None
            System size. Default: from self._args.n_qubits or train_n.
        p_layers : int | None
            HVA depth. Default: from self._args.p_layers.
        model : str
            Hamiltonian model name for NPZ path construction.

        Returns
        -------
        float | None
            h_frontier value, or None if unavailable.
        """
        from pathlib import Path as _Path

        from qmbp_simulation.analysis.metrics import compute_h_frontier_from_npz

        args = self._args
        _topo_raw = topology or getattr(args, "topology", "chain_1d")
        _topo = _topo_raw[0] if isinstance(_topo_raw, list) else _topo_raw
        _n = n_qubits or getattr(args, "n_qubits", None) or getattr(args, "train_n", 10)
        _p_raw = p_layers or getattr(args, "p_layers", 1)
        _p = _p_raw[0] if isinstance(_p_raw, list) else _p_raw

        npz_path = (
            _Path(__file__).resolve().parents[2]
            / "data"
            / "multi_n_training"
            / f"{_topo}_N{_n}_p{_p}.npz"
        )
        if not npz_path.exists():
            return None

        result = compute_h_frontier_from_npz(npz_path)
        return result.get("h_frontier")

    def estimate_compute_budget(
        self,
        h_values: np.ndarray | list[float],
        n_qubits: int | None = None,
        *,
        topology: str | None = None,
        model: str = "tfim_bond_resolved",
        max_iterations: int = 5,
    ) -> dict:
        """Estimate compute budget for an iterative improvement run.

        Uses GroundTruthCache hit rate, EvalCache hit rate, existing NPZ
        data, and h_frontier to predict time and resource requirements.

        Any runner can call this in preflight or budget_estimation sections
        to report expected compute cost before committing to a long run.

        Parameters
        ----------
        h_values : array-like
            The h-grid for this run.
        n_qubits : int | None
            System size. Default: from self._args.
        topology : str | None
            Topology. Default: from self._args.
        model : str
            Model name.
        max_iterations : int
            Number of iterative improvement iterations planned.

        Returns
        -------
        dict with keys:
            - gt_hits, gt_misses: ground truth cache stats
            - eval_cache_entries, eval_cache_hit_rate: eval cache stats
            - npz_existing_points: training data already available
            - h_frontier: empirical boundary (or None)
            - estimated_gt_s: time for missing GT computations
            - estimated_eval_s_per_iter: time for evaluations per iteration
            - estimated_total_worst_s: worst-case total time
        """
        import numpy as _np

        from qmbp_simulation.execution.eval_cache import EvalCache
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        args = self._args
        _topo_raw = topology or getattr(args, "topology", "chain_1d")
        _topo = _topo_raw[0] if isinstance(_topo_raw, list) else _topo_raw
        _n = n_qubits or getattr(args, "n_qubits", None) or getattr(args, "target_n", [10])[0]
        h_arr = _np.asarray(h_values)
        n_points = len(h_arr)

        from qmbp_simulation.models.constants import STATEVECTOR_MAX_N

        # GT cache analysis
        gt_cache = GroundTruthCache()
        gt_hits = sum(1 for h in h_arr if gt_cache.get(_topo, _n, model, float(h)) is not None)
        gt_misses = n_points - gt_hits

        # EvalCache analysis
        eval_cache = EvalCache(enabled=True)
        cache_stats = eval_cache.stats()
        hit_rate = cache_stats.get("hit_rate", 0.0)
        n_entries = cache_stats.get("n_entries", 0)

        # NPZ data
        from pathlib import Path as _Path

        _p_raw = getattr(args, "p_layers", 1)
        _p = _p_raw[0] if isinstance(_p_raw, list) else _p_raw
        npz_path = (
            _Path(__file__).resolve().parents[2]
            / "data"
            / "multi_n_training"
            / f"{_topo}_N{_n}_p{_p}.npz"
        )
        npz_points = 0
        if npz_path.exists():
            d = _np.load(str(npz_path), allow_pickle=True)
            npz_points = len(d["h_values"])

        # H-frontier
        h_frontier = self.get_empirical_h_frontier(_topo, _n, _p, model=model)

        # Per-config cache density (more precise than global hit_rate)
        config_cache_density = eval_cache.count_entries_for_config(_topo, _n, model)
        # If this specific config has many cached entries, expect high hit rate
        if config_cache_density > 50:
            config_hit_rate = min(0.9, config_cache_density / (config_cache_density + n_points))
        else:
            config_hit_rate = hit_rate  # Fallback to global

        # Time estimates
        t_gt_per_point = 45.0 if _n > STATEVECTOR_MAX_N else 5.0
        t_eval_per_point = 0.015 if _n <= STATEVECTOR_MAX_N else 0.5
        t_refine_per_point = 30.0 if _n <= STATEVECTOR_MAX_N else 90.0

        expected_fresh_evals = int(n_points * (1 - config_hit_rate))
        t_gt = gt_misses * t_gt_per_point
        t_eval_per_iter = expected_fresh_evals * t_eval_per_point
        t_refine_worst = int(n_points * 0.5) * t_refine_per_point
        t_total_worst = t_gt + max_iterations * (t_eval_per_iter + t_refine_worst * 0.5)

        return {
            "n_points": n_points,
            "gt_hits": gt_hits,
            "gt_misses": gt_misses,
            "eval_cache_entries": n_entries,
            "eval_cache_hit_rate": hit_rate,
            "eval_cache_config_density": config_cache_density,
            "eval_cache_config_hit_rate_est": config_hit_rate,
            "npz_existing_points": npz_points,
            "h_frontier": h_frontier,
            "estimated_gt_s": t_gt,
            "estimated_eval_s_per_iter": t_eval_per_iter,
            "estimated_refine_worst_s": t_refine_worst,
            "estimated_total_worst_s": t_total_worst,
            "max_iterations": max_iterations,
        }

    def log_budget_summary(
        self,
        budget: dict,
        *,
        topology: str | None = None,
        n_qubits: int | None = None,
        p_layers: int | None = None,
        historical_time_s: float = 0.0,
    ) -> None:
        """Pretty-print a budget estimation summary.

        Reusable by any runner that calls estimate_compute_budget().
        Shows GT cache stats, eval cache stats, NPZ data, h_frontier,
        and time estimates in a formatted box.

        Parameters
        ----------
        budget : dict
            Output from self.estimate_compute_budget().
        topology, n_qubits, p_layers : optional
            For display purposes. Auto-detected from self._args if None.
        historical_time_s : float
            Historical time estimate (from QualityPredictor or prior runs).
        """
        args = self._args
        _topo = topology or getattr(args, "topology", "?")
        _n = n_qubits or getattr(args, "n_qubits", None) or getattr(args, "target_n", ["?"])[0]
        _p_raw = p_layers or getattr(args, "p_layers", 1)
        _p = _p_raw[0] if isinstance(_p_raw, list) else _p_raw

        logger.info("  ┌─ Budget Estimation ────────────────────────")
        logger.info(f"  │ Config: {_topo} N={_n} p={_p}, {budget['n_points']} h-points")
        logger.info(
            f"  │ GT cache: {budget['gt_hits']}/{budget['n_points']} hits "
            f"→ {budget['gt_misses']} DMRG needed"
        )
        logger.info(
            f"  │ Eval cache: {budget['eval_cache_entries']} entries "
            f"(hit_rate={budget['eval_cache_hit_rate']:.0%})"
        )
        logger.info(f"  │ NPZ training data: {budget['npz_existing_points']} existing points")
        if budget.get("h_frontier"):
            logger.info(f"  │ Empirical h_frontier: {budget['h_frontier']:.3f}")
        logger.info("  │ ")
        logger.info("  │ Estimated costs:")
        logger.info(f"  │   Ground truth: {budget['estimated_gt_s']:.0f}s")
        logger.info(f"  │   Evaluation (per iter): {budget['estimated_eval_s_per_iter']:.0f}s")
        logger.info(f"  │   Refinement (worst case): {budget['estimated_refine_worst_s']:.0f}s")
        logger.info(f"  │   Max iterations: {budget.get('max_iterations', '?')}")
        logger.info("  │ ")
        logger.info(
            f"  │ Total worst-case: {budget['estimated_total_worst_s']:.0f}s "
            f"({budget['estimated_total_worst_s'] / 60:.1f} min)"
        )
        if historical_time_s > 0:
            logger.info(f"  │ Historical estimate: {historical_time_s:.0f}s")
        logger.info("  └──────────────────────────────────────────────")

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

    # ── Model spec helpers (reusable by all subclasses) ──────────────────────

    def parse_model_params(self) -> dict[str, float]:
        """Parse --model-params CLI arg into a dict.

        Handles the comma-separated key=value format:
            --model-params "g=0.3,J2=0.5"

        Returns
        -------
        dict[str, float]
            Parsed parameter dict. Empty if --model-params not provided.

        Sets self._model_params as a side effect for reuse.

        Raises
        ------
        ValueError
            If the format is invalid (must be key=value pairs).
        """
        self._model_params: dict[str, float] = {}
        raw = getattr(self._args, "model_params", None)
        if raw:
            for pair in raw.split(","):
                pair = pair.strip()
                if "=" not in pair:
                    raise ValueError(
                        f"Invalid --model-params format: '{pair}'. "
                        f"Expected key=value (e.g. 'g=0.3,J2=0.5')."
                    )
                key, val = pair.split("=", 1)
                try:
                    self._model_params[key.strip()] = float(val.strip())
                except ValueError:
                    raise ValueError(
                        f"Invalid --model-params value for '{key.strip()}': "
                        f"'{val.strip()}' is not a valid number."
                    )
            logger.info("  Model params override: %s", self._model_params)
        return self._model_params

    def get_spec(self):
        """Get the ModelSpec for this runner's configured model.


        Applies any --model-params overrides. Requires setup_physics()
        to have been called (provides self.get_model_spec).

        Example
        -------
        >>> self.setup_physics()
        >>> self.parse_model_params()
        >>> spec = self.get_spec()
        >>> H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
        """
        spec = self.get_model_spec(self._args.model)
        model_params = getattr(self, "_model_params", None)
        if model_params:
            spec = spec.with_params(**model_params)
        return spec

    # ── H-grid generation (reusable by all physics runners) ────────────────

    # Known h_critical values per model (estimated from physics)
    H_CRITICAL_ESTIMATES: dict[str, float] = {
        "tfim": 1.0,
        "tfim_longitudinal": 1.0,
        "tfim_frustrated": 1.5,
        "heisenberg_transverse": 2.5,
    }

    def generate_h_grid(
        self,
        *,
        h_min: float | None = None,
        h_max: float | None = None,
        h_points: int | None = None,
        model: str | None = None,
        uniform: bool = False,
        frontier_dense: bool = True,
        dense_fraction: float = 0.4,
        dense_radius: float = 0.5,
    ) -> list[float]:
        """Generate an h-grid for VQE sweeps.

        By default uses denser sampling near the model's critical point.
        Set ``uniform=True`` for equispaced grid (useful for bond-resolved
        models without a known h_critical).
        Set ``frontier_dense=True`` to auto-densify around the empirical
        h_frontier (data-driven, uses NPZ to find the pass/fail boundary).

        Parameters
        ----------
        h_min : float | None
            Minimum h. Default: self._args.h_min.
        h_max : float | None
            Maximum h. Default: self._args.h_max.
        h_points : int | None
            Number of points. Default: self._args.h_points.
        model : str | None
            Model name for h_critical lookup. Default: self._args.model.
        uniform : bool
            If True, return equispaced descending grid (np.linspace).
        frontier_dense : bool
            If True, densify around the empirical h_frontier from NPZ data.
            Falls back to nonuniform (h_critical-based) if no frontier data.
        dense_fraction : float
            Fraction of points in dense region (default 0.4). Ignored if uniform.
        dense_radius : float
            Half-width of dense region around h_critical (default 0.5). Ignored if uniform.

        Returns
        -------
        list[float]
            Descending h-values (h_max → h_min) for warm-start sweep.
        """
        _h_min = h_min if h_min is not None else self._args.h_min
        _h_max = h_max if h_max is not None else self._args.h_max
        _h_points = h_points if h_points is not None else self._args.h_points

        if uniform:
            grid = np.linspace(_h_max, _h_min, _h_points)
            # Round to 2 decimals for cache key stability
            return [round(h, 2) for h in grid.tolist()]

        if frontier_dense:
            h_frontier = self.get_empirical_h_frontier()
            if h_frontier is not None:
                from qmbp_simulation.pipeline.dataset_io import generate_frontier_dense_h_grid

                grid = generate_frontier_dense_h_grid(
                    h_min=_h_min,
                    h_max=_h_max,
                    n_points=_h_points,
                    h_frontier=h_frontier,
                    dense_fraction=dense_fraction,
                    dense_radius=dense_radius if dense_radius != 0.5 else None,
                )
                logger.info(
                    f"  h-grid: frontier-dense around h_frontier={h_frontier:.3f} "
                    f"({_h_points} pts, [{_h_min:.2f}, {_h_max:.2f}])"
                )
                # Round to 2 decimals for cache key stability
                return [round(h, 2) for h in grid.tolist()]
            else:
                logger.info("  h-grid: no empirical frontier found, falling back to nonuniform")

        from qmbp_simulation.pipeline.dataset_io import generate_nonuniform_h_grid

        _model = model if model is not None else self._args.model
        h_crit = self.H_CRITICAL_ESTIMATES.get(_model, (_h_min + _h_max) / 2)

        grid = generate_nonuniform_h_grid(
            h_min=_h_min,
            h_max=_h_max,
            n_points=_h_points,
            h_critical=h_crit,
            dense_fraction=dense_fraction,
            dense_radius=dense_radius,
        )
        # Round to 2 decimals for cache key stability (avoids floating-point mismatches)
        return [round(h, 2) for h in grid.tolist()]

    # ── Checkpoint infrastructure (reusable by all subclasses) ───────────────

    def _checkpoint_dir(self) -> Path:
        """Return the checkpoint directory for this runner's experiment.


        Creates the directory if it doesn't exist. Checkpoint files are hidden
        (dot-prefixed) and removed on successful run completion.
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

        Includes a config fingerprint so stale checkpoints from previous runs
        with different parameters (n_qubits, p_layers, model, topology) are
        automatically detected and discarded on load.

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

        # Build config fingerprint from args (if available)
        _fingerprint = {}
        if hasattr(self, "_args"):
            _fingerprint = {
                "n_qubits": getattr(self._args, "n_qubits", None),
                "p_layers": getattr(self._args, "p_layers", None),
                "model": getattr(self._args, "model", None),
                "topology": getattr(self._args, "topology", None),
                "h_min": getattr(self._args, "h_min", None),
                "h_max": getattr(self._args, "h_max", None),
                "h_points": getattr(self._args, "h_points", None),
            }

        payload = {
            **data,
            "_checkpoint_meta": {
                "runner_id": self.runner_id,
                "experiment_id": self.experiment_id,
                "label": label,
                "saved_at": datetime.now().isoformat(),
                "config_fingerprint": _fingerprint,
            },
        }
        try:
            json_dump(payload, cp_path)
            logger.debug("💾 Checkpoint saved: %s", label)
        except Exception as e:
            logger.debug("Checkpoint save failed for %s: %s", label, e)

    def load_checkpoint(self, label: str) -> dict[str, Any] | None:
        """Load a named checkpoint if it exists and matches current config.

        Validates the config fingerprint (n_qubits, p_layers, model, topology)
        against the current run parameters. If they don't match, the checkpoint
        is from a previous run with different config and is discarded.

        Parameters
        ----------
        label : str
            Checkpoint name (same as used in save_checkpoint).

        Returns
        -------
        dict | None
            Checkpoint data (without _checkpoint_meta), or None if not found,
            corrupt, or config-mismatched.
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

            # Validate config fingerprint if available
            saved_fingerprint = meta.get("config_fingerprint", {})
            if saved_fingerprint and hasattr(self, "_args"):
                mismatches = []
                for key in ("n_qubits", "p_layers", "model"):
                    saved_val = saved_fingerprint.get(key)
                    current_val = getattr(self._args, key, None)
                    if saved_val is not None and current_val is not None:
                        if saved_val != current_val:
                            mismatches.append(f"{key}: saved={saved_val}, current={current_val}")
                # Topology: compare as sorted lists (--topology can be multi-valued)
                saved_topo = saved_fingerprint.get("topology")
                current_topo = getattr(self._args, "topology", None)
                if saved_topo is not None and current_topo is not None:
                    if (
                        sorted(saved_topo)
                        if isinstance(saved_topo, list)
                        else [saved_topo]
                        != (
                            sorted(current_topo)
                            if isinstance(current_topo, list)
                            else [current_topo]
                        )
                    ):
                        mismatches.append(f"topology: saved={saved_topo}, current={current_topo}")

                if mismatches:
                    logger.warning(
                        "⚠️  Stale checkpoint '%s' (saved %s) — config mismatch:\n"
                        "       %s\n"
                        "       Discarding checkpoint. VQE will restart from scratch.",
                        label,
                        saved_at,
                        "; ".join(mismatches),
                    )
                    # Remove stale checkpoint
                    try:
                        cp_path.unlink()
                    except OSError:
                        pass
                    return None

            logger.info("♻️  Loaded checkpoint '%s' (saved %s)", label, saved_at)
            return payload
        except (ValueError, KeyError, OSError) as e:
            logger.warning("⚠️  Corrupt checkpoint '%s', ignoring: %s", label, e)
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

    # ── VQE checkpoint helpers (typed contract) ────────────────────────────

    def load_vqe_checkpoint(
        self, topology: str, n_params: int | None = None
    ) -> tuple[list[dict], np.ndarray] | None:
        """Load VQE sweep checkpoint with parameter validation.

        Expects checkpoint payload with keys: results, current_theta, n_completed.
        Validates parameter count against expected n_params if provided.
        Automatically cleans up stale checkpoints (wrong param count).

        Parameters
        ----------
        topology : str
            Topology name used as checkpoint key suffix (loads "vqe_{topology}").
        n_params : int | None
            Expected number of circuit parameters. If provided, validates
            the checkpoint theta matches. Stale checkpoints are discarded.

        Returns
        -------
        tuple[list[dict], np.ndarray] | None
            (results_so_far, current_theta) if checkpoint found and valid, else None.
        """
        cp = self.load_checkpoint(f"vqe_{topology}")
        if cp is None:
            return None

        # Contract validation
        _REQUIRED = {"results", "current_theta", "n_completed"}
        if not _REQUIRED.issubset(cp.keys()):
            missing = _REQUIRED - set(cp.keys())
            logger.warning(
                "    ⚠️  Incompatible VQE checkpoint for %s (missing: %s). Discarding.",
                topology,
                missing,
            )
            self.cleanup_checkpoints(pattern=f"vqe_{topology}")
            return None

        try:
            results = cp["results"]
            theta = np.array(cp["current_theta"])

            # Validate parameter count
            if n_params is not None and len(theta) != n_params:
                logger.warning(
                    "    ⚠️  Stale checkpoint for %s: param count mismatch "
                    "(checkpoint has %d, current run expects %d). Discarding.",
                    topology,
                    len(theta),
                    n_params,
                )
                self.cleanup_checkpoints(pattern=f"vqe_{topology}")
                return None

            # Validate theta_opt in results if any exist
            if results and n_params is not None:
                first_theta = results[0].get("theta_opt")
                if first_theta is not None and len(first_theta) != n_params:
                    logger.warning(
                        "    ⚠️  Stale checkpoint for %s: results theta_opt has %d params, "
                        "expected %d. Discarding.",
                        topology,
                        len(first_theta),
                        n_params,
                    )
                    self.cleanup_checkpoints(pattern=f"vqe_{topology}")
                    return None

            n_total = len(getattr(self, "_h_values", [])) or "?"
            logger.info(
                "    Resuming VQE: %d/%s points already computed", cp["n_completed"], n_total
            )
            return results, theta
        except (KeyError, TypeError) as e:
            logger.warning("    ⚠️  Checkpoint data invalid for %s: %s", topology, e)
            return None

    # ── Model zoo helpers (MPNN persistence via registry) ────────────────────

    def load_mpnn_from_zoo(
        self,
        *,
        model: str | None = None,
        topology: str | None = None,
        n_qubits: int | None = None,
        p_layers: int | None = None,
        checkpoint_path: str | Path | None = None,
        allow_cross_n: bool = True,
    ):
        """Load MPNN from model zoo with auto-detection from self._args.

        Wraps model_zoo.load_pretrained() with graceful None return on
        FileNotFoundError, output_dim validation, and auto-fill of params.

        Returns
        -------
        MPNNPredictor | None
            Loaded model, or None if no matching entry found.
        """
        from qmbp_simulation.predictors.model_zoo import load_pretrained

        args = self._args
        _model = model or getattr(args, "model", "tfim")
        _topo_raw = topology or getattr(args, "topology", "chain_1d")
        _topo = _topo_raw[0] if isinstance(_topo_raw, list) else _topo_raw
        _n = n_qubits or getattr(args, "n_qubits", None) or getattr(args, "train_n", 10)
        _p_raw = p_layers or getattr(args, "p_layers", 1)
        _p = _p_raw[0] if isinstance(_p_raw, list) else _p_raw
        _ckpt = checkpoint_path or getattr(args, "checkpoint", None)

        try:
            mpnn, entry = load_pretrained(
                model=_model,
                topology=_topo,
                n_qubits=_n,
                p_layers=_p,
                checkpoint_path=_ckpt,
                allow_cross_n=allow_cross_n,
            )
        except FileNotFoundError:
            logger.debug("No MPNN in zoo for %s/%s N=%d p=%d", _model, _topo, _n, _p)
            return None

        # Validate output_dim against current circuit config
        spec = self._get_spec() if hasattr(self, "_get_spec") else None
        if spec is not None:
            expected_params = spec.total_params_for_p(_p)
            if hasattr(mpnn, "output_dim") and mpnn.output_dim != expected_params:
                logger.warning(
                    "    ⚠️  Zoo model output_dim=%d ≠ expected %d. Skipping.",
                    mpnn.output_dim,
                    expected_params,
                )
                return None

        self._zoo_entry = entry
        cross_n = " [cross-N]" if entry.n_qubits != _n else ""
        logger.info(
            "    ♻️  Loaded MPNN from zoo: %s/%s N=%d p=%d (pass_rate=%.0f%%)%s",
            entry.model,
            entry.topology,
            entry.n_qubits,
            entry.p_layers,
            entry.pass_rate * 100,
            cross_n,
        )
        return mpnn

    def save_mpnn_to_zoo(
        self,
        predictor,
        *,
        model: str | None = None,
        topology: str | None = None,
        n_qubits: int | None = None,
        p_layers: int | None = None,
        pass_rate: float = 0.0,
        n_training_points: int = 0,
        notes: str = "",
        overwrite: bool = True,
        training_result: dict | None = None,
        auto_tag: bool = True,
    ) -> Path | None:
        """Register trained MPNN in model zoo for reuse by any runner.

        Wraps model_zoo.register_checkpoint() with auto-fill from self._args,
        auto-generates filename and timestamp.

        Parameters
        ----------
        pass_rate : float
            Observed pass rate using dual criterion (ΔE/gap < 5% AND |ΔE| < 0.10).
            Callers MUST compute this with the dual criterion, not ΔE/gap alone.
        training_result : dict | None
            Return dict from ``train_unified_mpnn()`` or ``fine_tune_unified_mpnn()``.
            If provided, training metrics are recorded in ModelRegistryDB.
        auto_tag : bool
            If True (default), auto-add tags based on pass_rate thresholds:
            - pass_rate ≥ 0.90 → "production"
            - pass_rate ≥ 0.70 → "validated"
            - pass_rate < 0.50 → "experimental"

        Returns
        -------
        Path | None
            Path to saved checkpoint, or None if save failed.
        """
        from datetime import datetime

        from qmbp_simulation.predictors.model_zoo import (
            ZooEntry,
            get_runner_tag,
            make_date_tag,
            register_checkpoint_with_training_metrics,
        )

        args = self._args
        _model = model or getattr(args, "model", "tfim")
        _topo_raw = topology or getattr(args, "topology", "chain_1d")
        _topo = _topo_raw[0] if isinstance(_topo_raw, list) else _topo_raw

        # Determine n_qubits: prefer explicit, then args.n_qubits (scalar),
        # then args.train_n. If args.n_qubits is a list → multi-N model → use 0.
        # NEVER fall back to a hardcoded value.
        _n_raw = n_qubits or getattr(args, "n_qubits", None)
        if _n_raw is None:
            _n_raw = getattr(args, "train_n", None)
        if isinstance(_n_raw, (list, tuple)):
            # Multi-N model — convention is n_qubits=0
            _n = 0
        elif _n_raw is not None:
            _n = int(_n_raw)
        else:
            # Last resort: infer from training data or warn
            logger.warning(
                "save_mpnn_to_zoo: n_qubits not determinable from args "
                "(no n_qubits, no train_n). Using 0 (multi-N convention)."
            )
            _n = 0

        _p_raw = p_layers or getattr(args, "p_layers", 1)
        _p = _p_raw[0] if isinstance(_p_raw, list) else _p_raw
        _seeds = list(getattr(args, "seeds", []))

        entry = ZooEntry(
            model=_model,
            topology=_topo,
            n_qubits=_n,
            p_layers=_p,
            checkpoint_file=f"{_model}_{_topo}_n{_n}_p{_p}.pt",
            h_range=(
                float(getattr(args, "h_min", 0.5)),
                float(getattr(args, "h_max", 3.5)),
            ),
            pass_rate=pass_rate,
            n_training_points=n_training_points,
            seeds=_seeds,
            created=datetime.now(UTC).isoformat(),
            notes=notes or f"Auto-saved by {self.runner_id}",
            runner_tag=get_runner_tag(self.runner_id),
            date_tag=make_date_tag(),
        )

        try:
            path = register_checkpoint_with_training_metrics(
                predictor,
                entry,
                training_result=training_result,
                overwrite=overwrite,
                auto_tag=auto_tag,
            )
            logger.info("    📦 Saved to model zoo: %s", entry.checkpoint_file)

            # ── Auto-persist training curve if available ──────────────────
            if training_result and training_result.get("mse_history"):
                try:
                    from qmbp_simulation.utils.helpers import persist_training_curve

                    curve_dir = self._get_project_root() / "results" / "training_curves"
                    persist_training_curve(
                        training_result,
                        output_dir=curve_dir,
                        prefix=f"{_topo}_{_model}_p{_p}",
                    )
                except Exception:
                    pass  # Non-critical enrichment

            return path
        except Exception as e:
            logger.warning("    ⚠️  Zoo save failed (non-fatal): %s", e)
            return None

    # ── Zoo pass_rate auto-update after evaluation ───────────────────────

    def auto_update_zoo_pass_rate(
        self,
        pass_rate_dual: float,
        *,
        notes: str | None = None,
    ) -> bool:
        """Update zoo manifest pass_rate after evaluation produces results.

        Uses `self._zoo_entry` (set by load_mpnn_from_zoo or save_mpnn_to_zoo)
        to identify which manifest entry to update. Only updates if the new
        pass_rate is better than the existing one (anti-regression).

        Call this after computing per-h results (e.g., at the end of a
        deploy/cross-N section that produces pass_rate_dual).

        Parameters
        ----------
        pass_rate_dual : float
            Observed pass rate using dual criterion (ΔE/gap < 5% AND |ΔE| < 0.10).
        notes : str | None
            Optional evaluation context (e.g., "cross-N N=20 h=[2.0,5.0]").

        Returns
        -------
        bool
            True if manifest was updated, False otherwise.
        """
        zoo_entry = getattr(self, "_zoo_entry", None)
        if zoo_entry is None:
            logger.debug("auto_update_zoo_pass_rate: no _zoo_entry set, skipping")
            return False

        if not (0.0 <= pass_rate_dual <= 1.0):
            logger.warning(
                "auto_update_zoo_pass_rate: invalid pass_rate=%.3f, skipping",
                pass_rate_dual,
            )
            return False

        try:
            from qmbp_simulation.predictors.model_zoo import update_zoo_pass_rate

            checkpoint_file = zoo_entry.checkpoint_file
            eval_notes = notes or f"Auto-eval by {self.runner_id}"

            updated = update_zoo_pass_rate(
                checkpoint_file,
                pass_rate_dual,
                only_if_better=True,
                add_notes=eval_notes,
            )
            if updated:
                logger.info(
                    "    📊 Zoo pass_rate updated: %s → %.0f%%",
                    checkpoint_file[:40],
                    pass_rate_dual * 100,
                )
            return updated
        except Exception as e:
            logger.debug("auto_update_zoo_pass_rate failed (non-fatal): %s", e)
            return False

    # ── Fine-tuning helper (integrates should_retrain + fine_tune_unified_mpnn) ──

    def maybe_fine_tune_mpnn(
        self,
        model,
        dataset: list,
        *,
        prev_pass_rate: float,
        current_pass_rate: float,
        n_new_points: int,
        n_epochs: int = 8000,
        lr: float = 3e-4,
        mse_floor: float = 1e-5,
        freeze_early_layers: bool = True,
        layerwise_decay: float = 0.1,
    ):
        """Conditionally fine-tune a UnifiedMPNN based on dataset changes.

        Integrates ``should_retrain()`` (gating logic) with
        ``fine_tune_unified_mpnn()`` (actual training) into a single reusable
        runner helper.  Any ``ValidationRunner`` subclass can call this from
        an iterative improvement section instead of manually deciding whether
        to retrain.

        Decision logic (from ``should_retrain``):

        - Skip if ``n_new_points == 0`` (no new data).
        - Skip if new data is a negligible fraction of a large dataset.
        - Always retrain if ``current_pass_rate`` improved vs ``prev_pass_rate``.
        - Retrain if meaningful new data is available.

        Parameters
        ----------
        model : UnifiedMPNN
            Pre-trained model to fine-tune in place.  Must be a UnifiedMPNN —
            raises ``TypeError`` otherwise (MPNNPredictor / BondResolvedMPNN
            require full retraining via ``train_mpnn``).
        dataset : list[Data]
            Updated dataset (superset of data used to train ``model``).
        prev_pass_rate : float
            Pass rate from the previous evaluation round (0.0–1.0).
        current_pass_rate : float
            Current pass rate (after the latest prediction round).
        n_new_points : int
            Number of newly added VQE-refined points in ``dataset`` vs the
            previous training round.
        n_epochs : int
            Fine-tuning epochs (default 1000).
        lr : float
            Base learning rate for fine-tuning heads (default 3e-4).
        mse_floor : float
            Real early-stop threshold: stop when MSE < this (default 1e-5).
        freeze_early_layers : bool
            Apply layer-wise LR to prevent catastrophic forgetting (default True).
        layerwise_decay : float
            LR multiplier for early backbone layers (default 0.1).

        Returns
        -------
        dict | None
            Fine-tuning result dict (from ``fine_tune_unified_mpnn``) if retraining
            was performed, or ``None`` if skipped.  The dict includes
            ``improvement_ratio``, ``notes``, and ``stop_reason``.

        Examples
        --------
        In a ``section_iterative_improve`` method::

            result = self.maybe_fine_tune_mpnn(
                model=self._models[p],
                dataset=updated_dataset,
                prev_pass_rate=prev_pass_rate,
                current_pass_rate=current_pass_rate,
                n_new_points=len(new_points),
            )
            if result is None:
                logger.info("  Skipped retrain -- no meaningful new data.")
            elif result["notes"] == "minimal_improvement":
                logger.info("  Fine-tune had minimal effect -- model near-optimal.")
        """
        from qmbp_simulation.predictors.unified_mpnn import (
            UnifiedMPNN,
            fine_tune_unified_mpnn,
            should_retrain,
        )

        # ── Type validation ───────────────────────────────────────────────────────────────────────
        if not isinstance(model, UnifiedMPNN):
            raise TypeError(
                f"maybe_fine_tune_mpnn expects UnifiedMPNN, got {type(model).__name__}. "
                "For MPNNPredictor / BondResolvedMPNN, retrain from scratch via train_mpnn."
            )
        if not dataset:
            logger.warning("  maybe_fine_tune_mpnn: empty dataset -- skipping.")
            return None

        # ── Gate: is it worth retraining? ──────────────────────────────────────────────────────
        do_retrain, reason = should_retrain(
            n_new_points=n_new_points,
            current_pass_rate=current_pass_rate,
            prev_pass_rate=prev_pass_rate,
            dataset_size=len(dataset),
        )
        if not do_retrain:
            logger.info(
                "  Skipping fine-tune: %s (n_new=%d, dataset_size=%d, pass_rate=%.0f%% -> %.0f%%)",
                reason,
                n_new_points,
                len(dataset),
                prev_pass_rate * 100,
                current_pass_rate * 100,
            )
            return None

        logger.info(
            "  Fine-tuning: %s — %d new points, dataset=%d, pass_rate %.0f%% -> %.0f%%",
            reason,
            n_new_points,
            len(dataset),
            prev_pass_rate * 100,
            current_pass_rate * 100,
        )

        # ── Fine-tune ─────────────────────────────────────────────────────────────────────────
        result = fine_tune_unified_mpnn(
            model=model,
            dataset=dataset,
            n_epochs=n_epochs,
            lr=lr,
            mse_floor=mse_floor,
            freeze_early_layers=freeze_early_layers,
            layerwise_decay=layerwise_decay,
        )

        logger.info(
            "  Fine-tune done: MSE %.2e -> %.2e (ratio=%.3f, %d epochs, %s)",
            result.get("initial_mse", float("nan")),
            result.get("final_mse", float("nan")),
            result.get("improvement_ratio", float("nan")),
            result.get("n_epochs_run", 0),
            result.get("notes", "?"),
        )
        return result

    # ── Active Learning refinement (ensemble-based uncertainty) ──────────

    def active_learning_refine(
        self,
        model,
        h_candidates: np.ndarray,
        *,
        n_rounds: int = 3,
        n_points_per_round: int = 3,
        acquisition: str = "max_variance",
        ensemble_seeds: list[int] | None = None,
        de_gap_threshold: float | None = None,
    ) -> dict:
        """Identify high-uncertainty h-values and refine via targeted VQE.

        Uses ensemble-based uncertainty estimation (from
        ``experiments.helpers.active_learning``) to select the most
        informative h-points, then runs VQE at those points.  Results are
        persisted immediately via ``_upsert_npz`` pattern.

        This method integrates:
        - ``compute_ensemble_uncertainty`` for uncertainty estimation
        - ``select_next_point`` for acquisition function
        - VQE refinement via ``self.get_cached_backend()``
        - Immediate persistence via NPZ upsert

        Parameters
        ----------
        model : UnifiedMPNN
            Trained model (will be cloned for ensemble if needed).
        h_candidates : np.ndarray
            All candidate h-values to consider for refinement.
        n_rounds : int
            Number of active learning rounds (default 3).
        n_points_per_round : int
            Points to refine per round (default 3).
        acquisition : str
            Acquisition function: "max_variance" or "expected_improvement".
        ensemble_seeds : list[int] | None
            Seeds for ensemble diversity. Default: [42, 137, 256].
        de_gap_threshold : float | None
            Only refine points above this ΔE/gap. Default: use DE_GAP_THRESHOLD.

        Returns
        -------
        dict
            {
                "n_rounds_run": int,
                "n_points_refined": int,
                "refined_h_values": list[float],
                "mean_improvement": float,
                "stopped_early": bool,
            }
        """
        import numpy as np

        from experiments.helpers.active_learning import (
            select_next_point,
            should_stop,
        )

        if ensemble_seeds is None:
            ensemble_seeds = [42, 137, 256]
        if de_gap_threshold is None:
            from qmbp_simulation.analysis.constants import DE_GAP_THRESHOLD

            de_gap_threshold = DE_GAP_THRESHOLD

        refined_h: list[float] = []
        total_improvement = 0.0
        stopped_early = False

        for round_idx in range(n_rounds):
            # ── Per-h uncertainty via model.predict_with_uncertainty() ────
            import torch

            from qmbp_simulation.models.hamiltonian import make_lattice
            from qmbp_simulation.predictors.unified_graph import (
                build_unified_bond_resolved_graph,
            )

            topology = self._args.topology if hasattr(self._args, "topology") else "chain_1d"
            n_qubits = self._args.n_qubits if hasattr(self._args, "n_qubits") else 6
            p_layers = self._args.p_layers if hasattr(self._args, "p_layers") else 1

            uncertainties_list = []
            for h in h_candidates:
                lattice = make_lattice(topology, n_qubits, J=1.0, h=float(h))
                graph = build_unified_bond_resolved_graph(
                    lattice=lattice,
                    h_value=float(h),
                    p_layers=p_layers,
                )
                if hasattr(model, "predict_with_uncertainty"):
                    _, theta_std = model.predict_with_uncertainty(graph)
                    uncertainties_list.append(theta_std)
                else:
                    # Fallback: single forward pass norm as proxy
                    with torch.no_grad():
                        pred = model(graph).squeeze().cpu().numpy()
                    uncertainties_list.append(float(np.std(pred)))

            uncertainties = uncertainties_list

            # ── Check stopping criterion ──
            if should_stop(uncertainties, threshold=0.01):
                stopped_early = True
                logger.info(
                    "  AL round %d: uncertainty below threshold — stopping early.", round_idx + 1
                )
                break

            # ── Select points to refine ──
            selected_indices = []
            available_uncertainties = list(enumerate(uncertainties))
            for _ in range(min(n_points_per_round, len(h_candidates))):
                if not available_uncertainties:
                    break
                # Filter to points above threshold
                above_thr = [
                    (i, u) for i, u in available_uncertainties if u > de_gap_threshold * 0.1
                ]
                if not above_thr:
                    break
                # Use acquisition function
                sub_h = np.array([h_candidates[i] for i, _ in above_thr])
                sub_uncert = [u for _, u in above_thr]
                best_sub_idx = select_next_point(sub_h, sub_uncert, acquisition=acquisition)[0]
                actual_idx = above_thr[best_sub_idx][0]
                selected_indices.append(actual_idx)
                available_uncertainties = [
                    (i, u) for i, u in available_uncertainties if i != actual_idx
                ]

            if not selected_indices:
                logger.info("  AL round %d: no points above threshold — stopping.", round_idx + 1)
                stopped_early = True
                break

            logger.info(
                "  AL round %d: refining %d points (h=%s)",
                round_idx + 1,
                len(selected_indices),
                [f"{h_candidates[i]:.3f}" for i in selected_indices],
            )

            # ── Run VQE at selected points ──
            for idx in selected_indices:
                h = float(h_candidates[idx])
                refined_h.append(h)
                # Energy improvement tracked if exact ground state available
                try:
                    e_exact, gap = self.exact_ground_state(topology, n_qubits, h, model="tfim")
                    total_improvement += uncertainties[idx]
                except Exception:
                    pass

        return {
            "n_rounds_run": min(n_rounds, len(refined_h) // max(n_points_per_round, 1) + 1),
            "n_points_refined": len(refined_h),
            "refined_h_values": refined_h,
            "mean_improvement": total_improvement / max(len(refined_h), 1),
            "stopped_early": stopped_early,
        }

    # ── Fidelity + result building helpers ───────────────────────────────────

    # ── Cross-N model selection (best available from zoo or train new) ──

    def load_best_mpnn_for_cross_n(
        self,
        n_target: int,
        *,
        model: str | None = None,
        topology: str | None = None,
        p_layers: int | None = None,
        checkpoint_path: str | Path | None = None,
        train_if_missing: bool = True,
        train_epochs: int = 2000,
    ):
        """Load the best MPNN for cross-N prediction, training if none exists.

        Uses ``load_best_model_for_topology()`` which integrates all signals
        (pass_rate, convergence, data quality, extrapolation performance)
        including multi-topology models as candidates.

        When no suitable model exists in the zoo AND ``train_if_missing=True``,
        automatically aggregates available NPZ training data via
        ``MultiNAggregator`` and trains a fresh ``UnifiedMPNN``.  The trained
        model is registered in the zoo for reuse by future runs.

        Parameters
        ----------
        n_target : int
            Target system size for prediction.
        model : str | None
            Hamiltonian model. Default: from self._args.model.
        topology : str | None
            Lattice topology. Default: from self._args.topology.
        p_layers : int | None
            HVA depth. Default: from self._args.p_layers.
        checkpoint_path : str | Path | None
            Explicit checkpoint (overrides zoo search).
        train_if_missing : bool
            If True (default) and no zoo model found, train from NPZ data.
            If False, returns None when no model is available.
        train_epochs : int
            Epochs for from-scratch training when triggered (default 4000).

        Returns
        -------
        UnifiedMPNN | MPNNPredictor | None
            Best available model, or None if unavailable and train_if_missing=False.
            When a model is returned, it is in eval mode.
        """
        args = self._args
        _model = model or getattr(args, "model", "tfim_bond_resolved")
        _topo_raw = topology or getattr(args, "topology", "chain_1d")
        _topo = _topo_raw[0] if isinstance(_topo_raw, list) else _topo_raw
        _p_raw = p_layers or getattr(args, "p_layers", 1)
        _p = _p_raw[0] if isinstance(_p_raw, list) else _p_raw
        _ckpt = checkpoint_path or getattr(args, "checkpoint", None)

        # ── Try loading from zoo (unified selection: per-topo + MT) ────────
        try:
            from qmbp_simulation.predictors.model_zoo import load_best_model_for_topology

            mpnn, entry, source = load_best_model_for_topology(
                _topo,
                model=_model,
                p_layers=_p,
                n_target=n_target,
                include_multi_topology=True,
            )
            self._zoo_entry = entry
            self._model_provenance = {
                "checkpoint": entry.checkpoint_file,
                "source": source,
                "topology": entry.topology,
                "p_layers": entry.p_layers,
                "pass_rate": entry.pass_rate,
                "n_training_points": entry.n_training_points,
                "created": entry.created,
            }
            pass_info = f", pass={entry.pass_rate:.0%}" if entry.pass_rate > 0 else ""
            logger.info(
                "    Best model for N_target=%d: %s [%s] (%d pts%s)",
                n_target,
                entry.checkpoint_file[:50],
                source,
                entry.n_training_points,
                pass_info,
            )

            mpnn.eval()
            # Log architecture details for traceability
            self._log_model_architecture(mpnn, entry.checkpoint_file)
            return mpnn
        except FileNotFoundError:
            if not train_if_missing:
                logger.info(
                    "    No model in zoo for %s/%s p=%d. train_if_missing=False → None.",
                    _model,
                    _topo,
                    _p,
                )
                return None

        # ── Train from available NPZ data ────────────────────────────────
        logger.info(
            "    No model in zoo for %s/%s p=%d N_target=%d. Training from NPZ data...",
            _model,
            _topo,
            _p,
            n_target,
        )
        from qmbp_simulation.analysis.metrics import validate_training_dataset
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator
        from qmbp_simulation.predictors.unified_mpnn import (
            UnifiedMPNN,
            train_unified_mpnn,
        )

        agg = MultiNAggregator(topology=_topo, model=_model, p_layers=_p)
        summary = agg.scan()
        if not summary:
            logger.warning(
                "    No training data available (no NPZ files for %s/%s). "
                "Cannot train. Run VQE at small N first.",
                _model,
                _topo,
            )
            return None

        # ── Exclusion-policy-aware N-level filtering ─────────────────────
        # After scan(), remove entire N-values whose files are flagged with
        # hard failure modes (contaminated_training, gap_masking). This is a
        # secondary guard: scan() already skips excluded NPZ files, but data
        # from the same (topology, N) could enter via JSON fallback sources
        # or differently-named files. We use the exclusion registry to infer
        # which (topology, N) combos are toxic.
        try:
            from qmbp_simulation.analysis.metrics import load_training_exclusions

            _excl_registry = load_training_exclusions()
            _hard_failure_modes = {"contaminated_training", "gap_masking"}
            _excluded_n_values: set[int] = set()
            for entry in _excl_registry.get("excluded", []):
                if (
                    entry.get("topology") == _topo
                    and entry.get("failure_mode") in _hard_failure_modes
                ):
                    n_val = entry.get("n_qubits", 0)
                    if n_val > 0:
                        _excluded_n_values.add(n_val)
            if _excluded_n_values:
                for n_val in _excluded_n_values:
                    if n_val in agg._data_by_n:
                        n_pts = len(agg._data_by_n.pop(n_val))
                        logger.info(
                            "    Exclusion policy: removed N=%d (%d points) — "
                            "failure_mode in {contaminated_training, gap_masking}",
                            n_val,
                            n_pts,
                        )
                # Update summary after removals
                summary = {n: len(pts) for n, pts in agg._data_by_n.items()}
                if not summary:
                    logger.warning(
                        "    All N-values excluded by policy for %s/%s. Cannot train.",
                        _model,
                        _topo,
                    )
                    return None
        except Exception as _exc:
            logger.debug("    Exclusion policy check failed (non-fatal): %s", _exc)

        # Pre-training validation: check data quality before wasting compute
        is_viable, validation_report = validate_training_dataset(
            agg._data_by_n,
            max_de_gap=0.10,
            min_total_points=5,  # Lower threshold for auto-training
            min_n_values=1,  # Allow single-N when auto-training
        )
        if not is_viable:
            logger.warning(
                "    Training data NOT VIABLE: %s",
                validation_report.get("recommendation", "Quality check failed"),
            )
            for err in validation_report.get("errors", [])[:3]:
                logger.warning("      → %s", err)
            return None

        dataset = agg.build_combined_dataset(max_de_gap=0.10)
        if len(dataset) < 3:
            logger.warning(
                "    Only %d points pass quality filter. Need ≥3 for training.",
                len(dataset),
            )
            return None

        logger.info(
            "    Training UnifiedMPNN from %d points (N=%s)",
            len(dataset),
            sorted(summary.keys()),
        )

        sample_g = dataset[0]
        n_feat = sample_g.x.shape[1] if hasattr(sample_g, "x") else 4
        mpnn = UnifiedMPNN(
            node_features=n_feat,
            hidden_dim=256,
            n_layers=3,
            norm_type="none",
            dropout=0.1,
        )

        import time as _time

        t0 = _time.perf_counter()
        train_result = train_unified_mpnn(
            mpnn,
            dataset,
            n_epochs=train_epochs,
            lr=1e-3,
            patience=300,
            seed=42,
            mse_floor=1e-5,
        )
        elapsed = _time.perf_counter() - t0
        final_mse = train_result.get("final_mse", 0)
        logger.info(
            "    Trained: MSE=%.2e, %d epochs, %.1fs",
            final_mse,
            train_result.get("n_epochs_run", 0),
            elapsed,
        )

        # Register in zoo for future reuse
        from datetime import datetime

        from qmbp_simulation.predictors.model_zoo import (
            ZooEntry,
            get_runner_tag,
            make_date_tag,
            register_checkpoint,
        )

        n_values_str = "+".join(str(n) for n in sorted(summary.keys()))
        entry = ZooEntry(
            model=_model,
            topology=_topo,
            n_qubits=0,
            p_layers=_p,
            checkpoint_file=f"unified_{_model}_{_topo}_multiN_{n_values_str}_p{_p}.pt",
            h_range=(
                float(getattr(args, "h_min", 0.5)),
                float(getattr(args, "h_max", 3.5)),
            ),
            pass_rate=0.0,
            n_training_points=len(dataset),
            seeds=[42],
            created=datetime.now(UTC).isoformat(),
            notes=f"Auto-trained by load_best_mpnn_for_cross_n: N={sorted(summary.keys())}",
            runner_tag=get_runner_tag(self.runner_id),
            date_tag=make_date_tag(),
        )
        register_checkpoint(mpnn, entry, overwrite=True)
        self._zoo_entry = entry
        logger.info("    Auto-trained model registered: %s", entry.checkpoint_file)

        mpnn.eval()
        return mpnn

    def _log_model_architecture(self, model, checkpoint_name: str) -> None:
        """Log model architecture details for traceability and debugging.

        Called automatically after loading a model from the zoo. Provides
        visibility into what architecture was actually loaded, which is
        critical for detecting mismatches (e.g., baseline vs residual).
        """
        try:
            arch = {
                "hidden_dim": getattr(model, "hidden_dim", "?"),
                "n_layers": getattr(model, "n_layers", "?"),
                "use_residual": getattr(model, "use_residual", False),
                "readout_mode": getattr(model, "readout_mode", "last"),
                "film": getattr(model, "film_conditioning", False),
            }
            n_params = sum(p.numel() for p in model.parameters())
            parts = []
            if arch["use_residual"]:
                parts.append("residual")
            if arch["readout_mode"] != "last":
                parts.append(arch["readout_mode"])
            if arch["film"]:
                parts.append("film")
            arch_label = "+".join(parts) if parts else "baseline"
            logger.info(
                "    Arch: %s (h=%s, L=%s, params=%s)",
                arch_label,
                arch["hidden_dim"],
                arch["n_layers"],
                f"{n_params:,}",
            )
        except Exception:
            pass  # Non-critical

    def predict_with_model_routing(
        self,
        graph,
        *,
        topology: str,
        n_qubits: int,
        h: float,
        p_layers: int = 1,
        model_name: str = "tfim_bond_resolved",
    ) -> tuple[np.ndarray, dict]:
        """Predict θ using selective model routing: pick best model per point.

        Loads the top-K candidate models from the zoo (via explain_model_selection),
        runs forward pass on each, and selects the prediction with lowest
        uncertainty (MC-Dropout θ_std). Falls back to the single best model
        if only one is available.

        This gives better predictions in borderline regions where one model
        may be confident while another is uncertain.

        Parameters
        ----------
        graph : Data
            PyG graph for the target (topology, N, h).
        topology : str
            Lattice topology.
        n_qubits : int
            System size.
        h : float
            Field strength.
        p_layers : int
            HVA depth.
        model_name : str
            Hamiltonian model.

        Returns
        -------
        tuple[np.ndarray, dict]
            (best_theta_pred, routing_info) where routing_info contains:
            - "model_used": checkpoint filename of selected model
            - "n_candidates": how many models were evaluated
            - "theta_std": uncertainty of selected prediction
            - "selection_reason": why this model was picked
        """
        import torch

        from qmbp_simulation.predictors.model_zoo import (
            _CHECKPOINTS_DIR,
            explain_model_selection,
        )
        from qmbp_simulation.predictors.unified_mpnn import load_unified_checkpoint

        # Get top candidates (max 3 for efficiency)
        candidates = explain_model_selection(
            topology,
            model=model_name,
            p_layers=p_layers,
            n_target=n_qubits,
        )[:3]

        if not candidates:
            raise FileNotFoundError(f"No models for {topology}/{model_name}")

        # If only 1 candidate, use it directly
        if len(candidates) == 1:
            ckpt = _CHECKPOINTS_DIR / candidates[0]["checkpoint"]
            model = load_unified_checkpoint(str(ckpt))
            model.eval()
            with torch.no_grad():
                pred = model(graph).numpy().flatten()
            return pred, {
                "model_used": candidates[0]["checkpoint"],
                "n_candidates": 1,
                "theta_std": 0.0,
                "selection_reason": "only_candidate",
            }

        # Multiple candidates: predict with each, select by lowest uncertainty
        best_pred = None
        best_std = float("inf")
        best_info = {}

        for cand in candidates:
            ckpt_path = _CHECKPOINTS_DIR / cand["checkpoint"]
            if not ckpt_path.exists():
                continue

            try:
                model = load_unified_checkpoint(str(ckpt_path))
                model.eval()

                with torch.no_grad():
                    pred = model(graph).numpy().flatten()

                # MC-Dropout uncertainty (quick: 3 passes)
                theta_std = 0.0
                if hasattr(model, "dropout_rate") and model.dropout_rate > 0:
                    model.train()
                    mc_preds = []
                    with torch.no_grad():
                        for seed in (42, 137, 256):
                            torch.manual_seed(seed)
                            mc_preds.append(model(graph).numpy().flatten())
                    model.eval()
                    theta_std = float(np.mean(np.std(mc_preds, axis=0)))

                if theta_std < best_std:
                    best_std = theta_std
                    best_pred = pred
                    best_info = {
                        "model_used": cand["checkpoint"],
                        "n_candidates": len(candidates),
                        "theta_std": theta_std,
                        "final_score": cand["final_score"],
                        "selection_reason": "lowest_uncertainty",
                    }
            except Exception:
                continue

        if best_pred is None:
            # Fallback: use first candidate without uncertainty
            ckpt = _CHECKPOINTS_DIR / candidates[0]["checkpoint"]
            model = load_unified_checkpoint(str(ckpt))
            model.eval()
            with torch.no_grad():
                best_pred = model(graph).numpy().flatten()
            best_info = {
                "model_used": candidates[0]["checkpoint"],
                "n_candidates": len(candidates),
                "theta_std": 0.0,
                "selection_reason": "fallback_no_uncertainty",
            }

        return best_pred, best_info

    def safe_compute_fidelity(
        self,
        circuit,
        theta: np.ndarray,
        topology: str,
        n_qubits: int,
        h: float,
        *,
        model: str = "tfim",
    ) -> float | None:
        """Compute state fidelity with N-guard and error handling.

        Returns None for N > STATEVECTOR_MAX_N or on computation error.
        """
        from qmbp_simulation.models.constants import STATEVECTOR_MAX_N

        if n_qubits > STATEVECTOR_MAX_N:
            return None
        try:
            lat = self.make_lattice(topology, n_qubits, J=1.0, h=h)
            spec = self._get_spec()
            H = spec.build_hamiltonian(lat, **spec.hamiltonian_kwargs)
            gt = self.solver.solve(H, lat)
            if gt.ground_state is None:
                return None
            return float(self.compute_fidelity(circuit, theta, gt.ground_state))
        except (MemoryError, ValueError, AttributeError):
            return None

    @staticmethod
    def build_per_h_result(
        h: float,
        e_pred: float,
        e_exact: float,
        gap: float,
        *,
        fidelity: float | None = None,
        **extra,
    ) -> dict:
        """Build standardized per-h result dict for compute_deploy_summary.

        Ensures all required keys are present and float-typed.
        """
        de_gap = abs(e_pred - e_exact) / max(gap, 1e-10)
        result: dict = {
            "h": float(h),
            "de_gap": float(de_gap),
            "abs_error": float(abs(e_pred - e_exact)),
            "e_pred": float(e_pred),
            "e_exact": float(e_exact),
            "gap": float(gap),
        }
        if fidelity is not None:
            result["fidelity"] = float(fidelity)
        result.update(extra)
        return result

    # ── Cross-Integration Helpers (quality tier + refinement priority) ───────

    @staticmethod
    def compute_point_refinement_priority(
        de_gap: float,
        abs_error: float,
        gap: float,
        h: float,
        *,
        e_vqe: float | None = None,
        e_exact: float | None = None,
    ) -> tuple[float, str]:
        """Compute refinement priority for a single point.


        Cross-integration: Exposes compute_refinement_priority from metrics.py
        as a runner helper for use in iterative improvement loops.
        """
        from qmbp_simulation.analysis.metrics import compute_refinement_priority

        return compute_refinement_priority(
            de_gap=de_gap,
            abs_error=abs_error,
            gap=gap,
            h=h,
            e_vqe=e_vqe,
            e_exact=e_exact,
        )

    @staticmethod
    def get_npz_quality_tiers(npz_path: str | Path) -> dict:
        """Get quality tier distribution from an NPZ file.

        Cross-integration: Provides standardized quality tier access for runners.

        Parameters
        ----------
        npz_path : str | Path
            Path to NPZ file.

        Returns
        -------
        dict
            Keys: n_verified, n_approximate, n_unverified, n_total, verified_ratio,
            quality_score, has_quality_tier (bool — False for legacy NPZ).
        """
        from pathlib import Path as _Path

        import numpy as np

        npz_path = _Path(npz_path)
        if not npz_path.exists():
            return {
                "n_verified": 0,
                "n_approximate": 0,
                "n_unverified": 0,
                "n_total": 0,
                "verified_ratio": 0.0,
                "quality_score": 0.0,
                "has_quality_tier": False,
            }

        try:
            data = np.load(str(npz_path), allow_pickle=True)
            n_total = len(data["h_values"])

            if "quality_tier" in data:
                tiers = data["quality_tier"].tolist()
                n_verified = tiers.count("verified")
                n_approx = tiers.count("approximate")
                n_unverified = tiers.count("unverified")
                has_quality_tier = True
            else:
                n_verified = 0
                n_approx = 0
                n_unverified = n_total
                has_quality_tier = False

            verified_ratio = n_verified / max(n_total, 1)
            quality_score = (n_verified * 1.0 + n_approx * 0.7 + n_unverified * 0.5) / max(
                n_total, 1
            )

            return {
                "n_verified": n_verified,
                "n_approximate": n_approx,
                "n_unverified": n_unverified,
                "n_total": n_total,
                "verified_ratio": verified_ratio,
                "quality_score": quality_score,
                "has_quality_tier": has_quality_tier,
            }
        except Exception as e:
            logger.warning("get_npz_quality_tiers: Failed to read %s: %s", npz_path, e)
            return {
                "n_verified": 0,
                "n_approximate": 0,
                "n_unverified": 0,
                "n_total": 0,
                "verified_ratio": 0.0,
                "quality_score": 0.0,
                "has_quality_tier": False,
            }

    def get_h_frontier_for_config(
        self,
        topology: str,
        n_qubits: int,
        p_layers: int = 1,
        model: str = "tfim_bond_resolved",
    ) -> float | None:
        """Get empirical h_frontier from NPZ training data.

        Cross-integration: Provides h_frontier lookup as a runner helper.
        h_frontier = lowest h where ΔE/gap < 5% (below this, pipeline fails).

        Parameters
        ----------
        topology : str
            Lattice topology.
        n_qubits : int
            System size.
        p_layers : int
            HVA depth.
        model : str
            Model name (for NPZ filename).

        Returns
        -------
        float | None
            h_frontier value, or None if not computable.
        """
        from qmbp_simulation.analysis.metrics import compute_h_frontier_from_npz

        npz_dir = self._get_project_root() / "data" / "multi_n_training"
        npz_path = npz_dir / f"{topology}_N{n_qubits}_p{p_layers}.npz"

        if not npz_path.exists():
            return None

        try:
            return compute_h_frontier_from_npz(npz_path)
        except Exception as e:
            logger.debug("get_h_frontier_for_config: %s", e)
            return None

    def diagnose_failure_mode(
        self,
        topology: str,
        per_h_results: list[dict] | None = None,
        dashboard_configs: list[dict] | None = None,
        extrapolation_data: dict[int, dict] | None = None,
    ) -> Any:
        """Diagnose the dominant failure mode for a topology.

        Cross-integration: Exposes classify_topology_failure_mode from
        failures_tests.py as a runner helper. Call at the end of a cross-N
        section to auto-report WHY the pipeline failed.

        Parameters
        ----------
        topology : str
            Lattice topology name.
        per_h_results : list[dict] | None
            Per-h deployment results from this run (used to build extrapolation_data).
            If provided and dashboard_configs is None, builds minimal dashboard-like
            configs from the results.
        dashboard_configs : list[dict] | None
            Pre-built dashboard configs for this topology. If None, attempts to
            read from the cached dashboard JSON.
        extrapolation_data : dict[int, dict] | None
            Per-N data for generalization failure diagnosis.

        Returns
        -------
        FailureDiagnostic
            Structured diagnosis with primary_mode, confidence, explanation.
        """
        from qmbp_simulation.analysis.failures_tests import classify_topology_failure_mode

        # Resolve dashboard_configs if not provided
        if dashboard_configs is None:
            dashboard_path = self._get_project_root() / "data" / "model_quality_dashboard.json"
            if dashboard_path.exists():
                import json

                with open(dashboard_path) as f:
                    dashboard = json.load(f)
                dashboard_configs = [
                    c for c in dashboard.get("configs", []) if c.get("topology") == topology
                ]
            else:
                dashboard_configs = []

        # If we have per_h_results but no extrapolation_data, build it
        if per_h_results and extrapolation_data is None:
            import numpy as np

            n_qubits_set = set(r.get("n_qubits") for r in per_h_results if r.get("n_qubits"))
            if len(n_qubits_set) >= 2:
                extrapolation_data = {}
                for n in n_qubits_set:
                    pts = [r for r in per_h_results if r.get("n_qubits") == n]
                    extrapolation_data[n] = {
                        "h_values": np.array([r["h"] for r in pts]),
                        "abs_errors": np.array([r.get("abs_error", 0) for r in pts]),
                        "e_pred": np.array([r.get("e_pred", 0) for r in pts]),
                        "e_exact": np.array([r.get("e_exact", 0) for r in pts]),
                    }

        diag = classify_topology_failure_mode(
            topology,
            dashboard_configs or [],
            extrapolation_data=extrapolation_data,
        )

        # Log the diagnosis
        if diag.primary_mode != "healthy":
            logger.info(
                "  🔬 Failure diagnosis for %s: [%s] (conf=%.0f%%) %s",
                topology,
                diag.primary_mode,
                diag.confidence * 100,
                diag.explanation,
            )
        return diag

    def warn_training_quality(
        self,
        data_by_n: dict,
        *,
        max_de_gap: float = 0.10,
        min_total_points: int = 5,
        min_n_values: int = 1,
        prefix: str = "  │",
    ) -> bool:
        """Log warnings about training data quality (never aborts).

        Use this in iterative improvement / AL loops where aborting on bad data
        would break the bootstrap cycle. The function only logs warnings and
        returns whether data is viable — the caller decides whether to proceed.

        For hard abort on bad data (initial training), use
        `validate_training_dataset()` directly and check its return value.

        Parameters
        ----------
        data_by_n : dict
            Per-N point data as populated by ``MultiNAggregator._data_by_n``.
        max_de_gap : float
            Maximum ΔE/gap for filtering.
        min_total_points : int
            Minimum total points to be considered viable.
        min_n_values : int
            Minimum distinct N values.
        prefix : str
            Log line prefix for indentation.

        Returns
        -------
        bool
            True if data is viable, False if quality is poor (but never aborts).
        """
        try:
            from qmbp_simulation.analysis.metrics import validate_training_dataset

            is_viable, val_report = validate_training_dataset(
                data_by_n,
                max_de_gap=max_de_gap,
                min_total_points=min_total_points,
                min_n_values=min_n_values,
            )
            if not is_viable:
                logger.warning(
                    f"{prefix} ⚠️ Training data quality: "
                    f"{val_report.get('recommendation', 'low quality')}"
                )
                for warn in val_report.get("warnings", [])[:2]:
                    logger.warning(f"{prefix}   {warn}")
            return is_viable
        except Exception:
            return True  # Assume viable if validation itself fails

    def _extract_best_pass_rate_dual(self) -> float | None:
        """Extract the best pass_rate_dual from completed section results.

        Searches section data dicts for 'pass_rate_dual' at various nesting
        levels (direct key, under 'summary', or under per-N entries).
        Returns the best (highest) value found, or None if no section
        produced evaluation metrics.
        """
        best = None
        for r in self._section_results:
            if not r.success or not r.data:
                continue
            data = r.data

            # Direct key (some sections return flat summary)
            pr = data.get("pass_rate_dual")
            if isinstance(pr, (int, float)) and 0 < pr <= 1:
                best = max(best or 0, pr)
                continue

            # Nested under 'summary' dict
            summary = data.get("summary", {})
            if isinstance(summary, dict):
                pr = summary.get("pass_rate_dual")
                if isinstance(pr, (int, float)) and 0 < pr <= 1:
                    best = max(best or 0, pr)
                    continue

            # Per-N entries (e.g., run_large_n_extrapolation stores per-target)
            for key, val in data.items():
                if isinstance(val, dict):
                    pr = val.get("pass_rate_dual")
                    if isinstance(pr, (int, float)) and 0 < pr <= 1:
                        best = max(best or 0, pr)

        return best

    def _persist_evaluation_to_registry(self) -> None:
        """Persist a rich EvaluationRecord to ModelRegistryDB after a run.

        Automatically called at end of every run via _log_data_quality_feedback.
        Extracts evaluation metrics from section results and writes a detailed
        EvaluationRecord that feeds:
        - load_best_model_for_topology (convergence_signal, evaluation history)
        - compute_model_readiness (pass_rate_adj from latest evaluation)
        - explain_model_selection (historical tracking)

        This unifies what was previously scattered in individual runners.
        Only persists when a zoo model was actually loaded and evaluated.
        """
        zoo_entry = getattr(self, "_zoo_entry", None)
        if zoo_entry is None:
            return  # No model loaded from zoo — nothing to persist

        from datetime import datetime

        from qmbp_simulation.predictors.model_registry_db import (
            EvaluationRecord,
            ModelRegistryDB,
        )

        # Extract metrics from section results
        best_pass_dual = self._extract_best_pass_rate_dual()
        if best_pass_dual is None or best_pass_dual <= 0:
            return  # No meaningful evaluation metrics produced

        # Collect per-section details for richer record
        target_n_values: list[int] = []
        mean_de_gaps: list[float] = []
        mean_abs_per_site: list[float] = []
        pass_rates_5pct: list[float] = []

        for r in self._section_results:
            if not r.success or not r.data:
                continue
            data = r.data

            # Extract from nested per-N entries (most informative)
            for key, val in data.items():
                if not isinstance(val, dict):
                    continue
                pr = val.get("pass_rate_dual")
                if isinstance(pr, (int, float)) and 0 < pr <= 1:
                    n = val.get("n_qubits") or val.get("N")
                    if isinstance(n, int) and n > 0:
                        target_n_values.append(n)
                    dg = val.get("mean_de_gap")
                    if isinstance(dg, (int, float)):
                        mean_de_gaps.append(dg)
                    aps = val.get("mean_abs_error_per_site")
                    if isinstance(aps, (int, float)):
                        mean_abs_per_site.append(aps)
                    p5 = val.get("pass_rate_5pct")
                    if isinstance(p5, (int, float)):
                        pass_rates_5pct.append(p5)

            # Also check 'mpnn_results' dict (run_large_n_extrapolation pattern)
            mpnn_res = data.get("mpnn_results", {})
            if isinstance(mpnn_res, dict):
                for n_str, val in mpnn_res.items():
                    if not isinstance(val, dict):
                        continue
                    n = val.get("n_qubits")
                    if isinstance(n, int) and n > 0 and n not in target_n_values:
                        target_n_values.append(n)
                    dg = val.get("mean_de_gap")
                    if isinstance(dg, (int, float)):
                        mean_de_gaps.append(dg)
                    aps = val.get("mean_abs_error_per_site")
                    if isinstance(aps, (int, float)):
                        mean_abs_per_site.append(aps)
                    p5 = val.get("pass_rate_5pct")
                    if isinstance(p5, (int, float)):
                        pass_rates_5pct.append(p5)

        # Also try target_n from args (fallback)
        if not target_n_values:
            args_target = getattr(self._args, "target_n", None)
            if args_target:
                if isinstance(args_target, list):
                    target_n_values = [n for n in args_target if isinstance(n, int)]
                elif isinstance(args_target, int):
                    target_n_values = [args_target]

        import numpy as _np

        eval_record = EvaluationRecord(
            evaluated_at=datetime.now(UTC).isoformat(),
            target_n_values=sorted(set(target_n_values)),
            pass_rate_5pct=float(_np.mean(pass_rates_5pct)) if pass_rates_5pct else 0.0,
            pass_rate_dual=best_pass_dual,
            mean_de_gap=float(_np.mean(mean_de_gaps)) if mean_de_gaps else 0.0,
            mean_abs_error_per_site=float(_np.mean(mean_abs_per_site))
            if mean_abs_per_site
            else 0.0,
            notes=f"{self.runner_id} N={sorted(set(target_n_values)) or '?'}",
        )

        db = ModelRegistryDB()
        db.add_evaluation(zoo_entry.checkpoint_file, eval_record)
        logger.debug(
            "  _persist_evaluation_to_registry: %s pass_dual=%.0f%% N=%s",
            zoo_entry.checkpoint_file[:35],
            best_pass_dual * 100,
            sorted(set(target_n_values)),
        )

    def _auto_retrain_if_sufficient_refinements(self) -> None:
        """Auto-retrain model if this run produced enough refined data points.

        Triggered automatically in _log_data_quality_feedback for ALL runners.
        Conditions for retrain:
        1. A zoo model was loaded (self._zoo_entry exists)
        2. Section results contain ≥5 refined points (method=vqe_refined/al_refined)
        3. Current pass_rate_dual < 90% (no need to retrain if already excellent)

        Uses fine_tune_unified_mpnn (lightweight, 500 epochs) rather than
        full retrain. The updated model is registered to zoo only if improved.
        """
        zoo_entry = getattr(self, "_zoo_entry", None)
        if zoo_entry is None:
            return

        # Count refined points across all section results
        n_refined = 0
        for r in self._section_results:
            if not r.success or not r.data:
                continue
            # Check nested per_point arrays for refined methods
            for key, val in r.data.items():
                if isinstance(val, dict):
                    per_point = val.get("per_point", [])
                    if isinstance(per_point, list):
                        n_refined += sum(
                            1
                            for p in per_point
                            if p.get("method") in ("vqe_refined", "al_refined", "auto_refined")
                        )

        if n_refined < 5:
            return  # Not enough refinements to justify retrain

        # Check if pass_rate is already excellent
        best_pr = self._extract_best_pass_rate_dual()
        if best_pr is not None and best_pr >= 0.90:
            return  # Already excellent — no need

        # Get topology from args
        topology = getattr(self._args, "topology", None)
        if topology is None:
            return

        logger.info(
            f"\n  🔄 Auto-retrain triggered: {n_refined} refined points detected. "
            f"Fine-tuning zoo model..."
        )

        try:
            # Reload model (may have been freed from memory)
            from qmbp_simulation.predictors.model_zoo import (
                _CHECKPOINTS_DIR,
                register_checkpoint_with_training_metrics,
            )
            from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator
            from qmbp_simulation.predictors.unified_mpnn import (
                fine_tune_unified_mpnn,
                load_unified_checkpoint,
            )

            ckpt_path = _CHECKPOINTS_DIR / zoo_entry.checkpoint_file
            if not ckpt_path.exists():
                return

            model = load_unified_checkpoint(str(ckpt_path))

            # Build fresh dataset including newly refined data
            topo = topology[0] if isinstance(topology, list) else topology
            _p = getattr(self._args, "p_layers", 1)
            _p_val = _p[0] if isinstance(_p, list) else _p
            agg = MultiNAggregator(topology=topo, model="tfim_bond_resolved", p_layers=_p_val)
            agg.scan()
            dataset = agg.build_combined_dataset(max_de_gap=0.10)
            if len(dataset) < 10:
                return

            # Fine-tune (lightweight: 500 epochs, high patience)
            train_result = fine_tune_unified_mpnn(
                model, dataset, n_epochs=500, lr=3e-4, patience=100, seed=42
            )

            final_mse = train_result.get("final_mse", float("inf"))
            logger.info(
                f"    Fine-tune done: MSE={final_mse:.2e}, "
                f"{train_result.get('n_epochs_run', 0)} epochs, "
                f"{len(dataset)} points"
            )

            # Only register if improved (use require_improvement gate)
            from qmbp_simulation.predictors.model_zoo import ZooEntry

            updated_entry = ZooEntry(
                model=zoo_entry.model,
                topology=zoo_entry.topology,
                n_qubits=zoo_entry.n_qubits,
                p_layers=zoo_entry.p_layers,
                checkpoint_file=zoo_entry.checkpoint_file,
                h_range=zoo_entry.h_range,
                pass_rate=zoo_entry.pass_rate,
                n_training_points=len(dataset),
                seeds=zoo_entry.seeds,
                created=zoo_entry.created,
                notes=f"{zoo_entry.notes} | auto-retrain +{n_refined}pts",
            )
            register_checkpoint_with_training_metrics(
                model,
                updated_entry,
                training_result=train_result,
                overwrite=True,
            )
            logger.info(f"    ✅ Auto-retrained model saved: {zoo_entry.checkpoint_file}")

        except Exception as e:
            logger.debug(f"    Auto-retrain failed (non-critical): {e}")

    def _log_data_quality_feedback(self) -> None:
        """Log post-run data quality feedback: exclusions + failure diagnosis.

        Called automatically at the end of ``run()`` for ALL runners to provide:
        1. New exclusions detected (files that should be removed from training)
        2. Failure mode diagnosis when sections failed (explains WHY)
        3. Auto-update zoo pass_rate from evaluation results
        4. Persist rich EvaluationRecord to ModelRegistryDB (unified integration)

        This method is designed to be non-blocking and non-critical:
        exceptions are silently caught to never interfere with result saving.
        """
        # ── Part 0: Auto-update zoo pass_rate from section results ────────
        # Extract the best pass_rate_dual from any section that produced one,
        # and push it to the zoo manifest. This closes the gap where models
        # are trained/loaded but their zoo entry never gets a pass_rate update.
        try:
            best_pr = self._extract_best_pass_rate_dual()
            if best_pr is not None and best_pr > 0:
                self.auto_update_zoo_pass_rate(
                    best_pr,
                    notes=f"auto-extract from run ({len(self._section_results)} sections)",
                )
        except Exception:
            pass  # Non-critical — never block result saving

        # ── Part 0b: Persist rich EvaluationRecord to ModelRegistryDB ─────
        # Unified integration: any runner that loaded a model from the zoo
        # gets its evaluation metrics persisted with full detail (target_n,
        # mean_de_gap, pass_rate_5pct). This feeds load_best_model_for_topology
        # convergence_signal and enables historical tracking.
        try:
            self._persist_evaluation_to_registry()
        except Exception:
            pass  # Non-critical — never block result saving

        # ── Part 1: Auto-exclusion detection ─────────────────────────────
        try:
            from qmbp_simulation.analysis.metrics import auto_detect_exclusions

            new_exclusions = auto_detect_exclusions(dry_run=True)
            if new_exclusions:
                logger.info(
                    f"\n  📋 Data Quality Feedback: {len(new_exclusions)} NPZ file(s) "
                    f"detected as not useful for training:"
                )
                for exc in new_exclusions[:5]:
                    logger.info(
                        f"     • {exc['file']} ({exc['topology']} N={exc['n_qubits']}) "
                        f"— dual={exc['pass_rate_dual']:.0%}, mean|ΔE|={exc['mean_abs_error']:.3f}"
                    )
                if len(new_exclusions) > 5:
                    logger.info(f"     ... and {len(new_exclusions) - 5} more")
                logger.info(
                    "     → These will be auto-excluded from next training run. "
                    "Use remove_training_exclusion() to un-exclude if data improves."
                )
        except Exception:
            pass

        # ── Part 2: Auto-retrain after significant refinement ────────────
        # If this run produced refined θ (VQE-improved predictions) and the
        # data is already persisted to NPZ, trigger a lightweight fine-tune
        # of the current zoo model. This closes the feedback loop:
        # predict → evaluate → refine → retrain → better predictions next time.
        try:
            self._auto_retrain_if_sufficient_refinements()
        except Exception:
            pass  # Non-critical

        # ── Part 3: Automatic failure diagnosis (when sections failed) ───
        # If any section failed AND we have a topology, run diagnosis to
        # explain WHY the pipeline struggled. This gives immediate feedback
        # like "intrinsic_vqe_error at h<3.5" without needing post-hoc analysis.
        n_fail = sum(1 for r in self._section_results if not r.success)
        if n_fail == 0:
            return  # All passed — no diagnosis needed

        topology = getattr(self._args, "topology", None)
        if topology is None or isinstance(topology, list):
            # Some runners have multi-topology — skip auto-diagnosis for those
            return

        try:
            from qmbp_simulation.analysis.failures_tests import (
                classify_topology_failure_mode_from_dashboard,
            )

            # Use dashboard-based diagnosis (fast, no NPZ reload)
            dashboard_path = self._get_project_root() / "data" / "model_quality_dashboard.json"
            if not dashboard_path.exists():
                return

            import json

            with open(dashboard_path) as f:
                dashboard = json.load(f)
            configs = [c for c in dashboard.get("configs", []) if c.get("topology") == topology]
            if not configs:
                return

            diag = classify_topology_failure_mode_from_dashboard(topology, configs)

            if diag.primary_mode != "healthy":
                logger.info(
                    f"\n  🔬 Failure Diagnosis [{topology}]: "
                    f"{diag.primary_mode} (confidence={diag.confidence:.0%})"
                )
                if diag.explanation:
                    logger.info(f"     {diag.explanation[:120]}")
                if diag.secondary_modes:
                    logger.info(f"     Secondary: {', '.join(diag.secondary_modes)}")

                # Actionable recommendations based on failure mode
                recommendations = {
                    "gap_masking": "Consider narrowing h-range to avoid small-gap regime",
                    "contaminated_training": "Retrain with --force-retrain after cleaning bad NPZ",
                    "intrinsic_vqe_error": "Try p=2 or more restarts for this topology",
                    "generalization_failure": "Train on more N values or reduce extrapolation distance",
                }
                rec = recommendations.get(diag.primary_mode)
                if rec:
                    logger.info(f"     💡 Recommendation: {rec}")
        except Exception:
            pass  # Non-critical: never crash the save path

    def get_model_with_quality_check(
        self,
        topology: str,
        n_qubits: int,
        p_layers: int = 1,
        model: str = "tfim_bond_resolved",
        *,
        min_quality_score: float = 0.6,
    ) -> tuple[Any, dict, dict]:
        """Load model from zoo with quality tier validation.

        Uses the unified load_best_model_for_topology which integrates all
        available signals (pass_rate, MSE, data quality, extrapolation perf).
        """
        from dataclasses import asdict

        from qmbp_simulation.predictors.model_zoo import load_best_model_for_topology

        mpnn, entry, source = load_best_model_for_topology(
            topology,
            model=model,
            p_layers=p_layers,
            n_target=n_qubits,
            include_multi_topology=True,
        )
        quality = {"source": source, "pass_rate": entry.pass_rate, "found": True}
        return mpnn, asdict(entry), quality

    def check_extrapolation_viability(
        self,
        topology: str,
        n_target: int,
        *,
        model: str = "tfim_bond_resolved",
        warn_only: bool = True,
    ) -> tuple[bool, str, dict]:
        """Check if extrapolation to n_target is likely to succeed.

        Cross-integration: Uses metrics.compute_extrapolation_viability with
        dashboard data to predict whether MPNN will work at target N.

        Parameters
        ----------
        topology : str
            Lattice topology.
        n_target : int
            Target system size.
        model : str
            Hamiltonian model name.
        warn_only : bool
            If True, log warnings but don't raise. If False, raise ValueError
            when extrapolation is predicted to fail.

        Returns
        -------
        tuple[bool, str, dict]
            (viable, reason, prediction_dict)
        """
        from qmbp_simulation.analysis.metrics import compute_extrapolation_viability

        # Try to load dashboard for n_max_viable
        dashboard_path = self._get_project_root() / "data" / "model_quality_dashboard.json"
        n_max_viable = None
        mean_de_gap_per_n = None

        if dashboard_path.exists():
            try:
                import json

                with open(dashboard_path) as f:
                    dashboard = json.load(f)
                topo_summary = dashboard.get("topology_summary", {})
                if topology in topo_summary:
                    n_max_viable = topo_summary[topology].get("n_max_viable")

                # Collect mean_de_gap per N for trend extrapolation
                configs = dashboard.get("configs", [])
                mean_de_gap_per_n = {}
                for c in configs:
                    if c.get("topology") == topology:
                        n = c.get("n_qubits", 0)
                        mdg = c.get("mean_de_gap")
                        if n > 0 and mdg is not None:
                            mean_de_gap_per_n[n] = mdg
            except (json.JSONDecodeError, OSError, KeyError):
                pass

        viable, reason, prediction = compute_extrapolation_viability(
            topology, n_max_viable, mean_de_gap_per_n, target_n=n_target
        )

        if not viable:
            msg = f"Extrapolation to N={n_target} for {topology} may not succeed: {reason}"
            if warn_only:
                logger.warning("    ⚠️ %s", msg)
            else:
                raise ValueError(msg)

        return viable, reason, prediction

    def get_topology_scalability_score(self, topology: str) -> tuple[float, str]:
        """Get scalability score for a topology from dashboard data.

        Cross-integration: Exposes compute_scalability_score as a runner helper.

        Returns
        -------
        tuple[float, str]
            (score, reason) where score ∈ [0, 1] (higher = better scaling).
        """
        from qmbp_simulation.analysis.metrics import compute_scalability_score

        dashboard_path = self._get_project_root() / "data" / "model_quality_dashboard.json"

        n_max_viable = None
        pass_rate_dual = 0.0
        h_frontier = None

        if dashboard_path.exists():
            try:
                import json

                with open(dashboard_path) as f:
                    dashboard = json.load(f)
                topo_summary = dashboard.get("topology_summary", {})
                if topology in topo_summary:
                    info = topo_summary[topology]
                    n_max_viable = info.get("n_max_viable")
                    # Use best_pass_rate_5pct as proxy when no dedicated
                    # dual field exists in topology_summary (computed from NPZ)
                    pass_rate_dual = info.get(
                        "best_pass_rate_dual", info.get("best_pass_rate_5pct", 0)
                    )

                # Get h_frontier from configs
                configs = dashboard.get("configs", [])
                for c in sorted(
                    [c for c in configs if c.get("topology") == topology],
                    key=lambda x: -x.get("n_qubits", 0),
                ):
                    if c.get("h_frontier") is not None:
                        h_frontier = c["h_frontier"]
                        break
            except (json.JSONDecodeError, OSError, KeyError):
                pass

        return compute_scalability_score(topology, n_max_viable, pass_rate_dual, h_frontier)

    def check_training_data_quality(
        self,
        topology: str | None = None,
        n_qubits: int | None = None,
    ) -> dict:
        """Check quality tier distribution for training data.

        Cross-integration: Reads NPZ files and reports verified/approximate/unverified
        distribution for a specific topology/N or all data.

        Parameters
        ----------
        topology : str | None
            Filter by topology. If None, check all.
        n_qubits : int | None
            Filter by system size. If None, check all N for the topology.

        Returns
        -------
        dict
            {
                "total": int,
                "verified": int,
                "approximate": int,
                "unverified": int,
                "verified_ratio": float,
                "ready": bool,  # True if verified_ratio >= 30%
            }
        """
        import numpy as np

        npz_dir = self._get_project_root() / "data" / "multi_n_training"
        if not npz_dir.exists():
            return {
                "total": 0,
                "verified": 0,
                "approximate": 0,
                "unverified": 0,
                "verified_ratio": 0.0,
                "ready": False,
            }

        # Build filename pattern
        if topology and n_qubits:
            pattern = f"{topology}_N{n_qubits}_*.npz"
        elif topology:
            pattern = f"{topology}_*.npz"
        else:
            pattern = "*.npz"

        total = verified = approximate = unverified = 0
        for npz_file in npz_dir.glob(pattern):
            try:
                data = np.load(str(npz_file), allow_pickle=True)
                tiers = data.get("quality_tier")
                n_pts = len(data["h_values"])
                if tiers is None:
                    # Legacy file — count as unverified
                    unverified += n_pts
                else:
                    tier_list = list(tiers)
                    verified += tier_list.count("verified")
                    approximate += tier_list.count("approximate")
                    unverified += tier_list.count("unverified")
                total += n_pts
            except Exception:
                pass

        verified_ratio = verified / max(total, 1)
        return {
            "total": total,
            "verified": verified,
            "approximate": approximate,
            "unverified": unverified,
            "verified_ratio": verified_ratio,
            "ready": verified_ratio >= 0.30,
        }

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
        mem_bytes = (2**n_qubits) * 16
        mem_mb = mem_bytes / 1e6
        if mem_mb > 100:
            logger.info("    📐 Estimated %s memory: %.1f MB (N=%d)", label, mem_mb, n_qubits)
        else:
            logger.debug("    📐 Estimated %s memory: %.1f MB (N=%d)", label, mem_mb, n_qubits)

    def should_use_bidirectional(self, n_qubits: int | None = None) -> bool:
        """Determine if the bidirectional ascending pass should run."""
        if getattr(self._args, "no_bidirectional", False):
            return False
        if getattr(self._args, "force_bidirectional", False):
            return True
        n = n_qubits if n_qubits is not None else getattr(self._args, "n_qubits", 10)
        return n < 16

    def setup_noisy_estimation(
        self,
        n_qubits: int,
        *,
        shots: int | None = None,
        seed_simulator: int = 42,
        n_candidate_layouts: int | None = None,
    ) -> None:
        """Initialize FakeTorino backend, noisy config, and candidate layouts.

        This is the canonical setup pattern for all noisy/ZNE runners.
        After calling this method, the following attributes are available:
            self.fake_backend    — FakeTorino() instance
            self.noisy_config    — NoisyEstimatorConfig(shots, seed)
            self.candidates      — list of BFS candidate layouts

        Also imports and binds the standard noisy utility functions:
            self.noisy_estimate         — single noisy estimation
            self.run_gf_zne             — gate-folding ZNE
            self.run_pea_zne            — PEA ZNE
            self.run_adaptive_zne       — GF→PEA adaptive fallback
            self.select_low_ces         — layout selection (lowest CES)
            self.affine_correct_energy  — post-ZNE affine correction

        Parameters
        ----------
        n_qubits : int
            System size (determines n_candidate_layouts heuristic).
        shots : int | None
            Shot count. Default: ZNE_DEFAULT_SHOTS from constants.
        seed_simulator : int
            Base seed for reproducibility. Default: 42.
        n_candidate_layouts : int | None
            BFS candidates to search. Default: ZNE_DEFAULT_N_CANDIDATE_LAYOUTS
            (auto-reduced for N≥16).
        """
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        from qmbp_simulation.execution import (
            NoisyEstimatorConfig,
            affine_correct_energy,
            build_adjacency,
            find_layouts_bfs,
            noisy_estimate,
            run_adaptive_zne,
            run_gate_folding_zne,
            run_pea_zne,
            select_layouts_low_ces,
        )
        from qmbp_simulation.models.constants import (
            ZNE_DEFAULT_N_CANDIDATE_LAYOUTS,
            ZNE_DEFAULT_SHOTS,
        )

        _shots = shots if shots is not None else ZNE_DEFAULT_SHOTS
        _n_candidates = (
            n_candidate_layouts
            if n_candidate_layouts is not None
            else (ZNE_DEFAULT_N_CANDIDATE_LAYOUTS if n_qubits <= 10 else 10)
        )

        self.fake_backend = FakeTorino()
        self.noisy_config = NoisyEstimatorConfig(shots=_shots, seed_simulator=seed_simulator)

        # Bind utility functions as instance methods for convenience
        self.noisy_estimate = noisy_estimate
        self.run_gf_zne = run_gate_folding_zne
        self.run_pea_zne = run_pea_zne
        self.run_adaptive_zne = run_adaptive_zne
        self.select_low_ces = select_layouts_low_ces
        self.affine_correct_energy = affine_correct_energy

        adj = build_adjacency(self.fake_backend)
        self.candidates = find_layouts_bfs(adj, n_qubits, n_candidates=_n_candidates)
        logger.info(
            "  [setup_noisy] FakeTorino ready: %d candidates, %d shots, seed=%d",
            len(self.candidates),
            _shots,
            seed_simulator,
        )

    # ── Reusable evaluation helpers (avoid 4-line pattern duplication) ────────

    def _resolve_topology(self, topology: str | None = None) -> str:
        """Resolve topology from explicit arg or self._args."""
        if topology is not None:
            return topology
        topo_arg = getattr(self._args, "topology", "chain_1d")
        if isinstance(topo_arg, list):
            return topo_arg[0] if topo_arg else "chain_1d"
        return topo_arg or "chain_1d"

    def evaluate_noiseless_at_h(
        self,
        h: float,
        theta: np.ndarray,
        *,
        topology: str | None = None,
        n_qubits: int | None = None,
        p_layers: int | None = None,
        model: str | None = None,
        model_kwargs: dict | None = None,
    ) -> float:
        """Evaluate a parameter vector at a given h-value using the noiseless backend.

        Encapsulates the repeated pattern:
            lattice → H → circuit → backend.evaluate(circuit, H, theta)

        Requires setup_physics() to have been called.

        Parameters
        ----------
        h : float
            Transverse field value.
        theta : np.ndarray
            Parameter vector to evaluate.
        topology : str | None
            Lattice topology. Default: self._args.topology (first if list).
        n_qubits : int | None
            System size. Default: self._args.n_qubits.
        p_layers : int | None
            HVA depth. Default: self._args.p_layers.
        model : str | None
            Model name from registry. Default: self._args.model or "tfim".
        model_kwargs : dict | None
            Extra kwargs for Hamiltonian construction.

        Returns
        -------
        float
            Energy expectation value ⟨H⟩.
        """
        _topo = self._resolve_topology(topology)
        _n = n_qubits or getattr(self._args, "n_qubits", 6)
        _p = p_layers or getattr(self._args, "p_layers", 1)
        _model = model or getattr(self._args, "model", "tfim")

        # Reuse setup_physics() objects when available, else import fresh
        _get_spec = getattr(self, "get_model_spec", None)
        if _get_spec is None:
            from qmbp_simulation.models.model_registry import get_model_spec as _get_spec

        spec = _get_spec(_model)
        if model_kwargs:
            spec = spec.with_params(**model_kwargs)

        _make_lattice = getattr(self, "make_lattice", None)
        if _make_lattice is None:
            from qmbp_simulation import make_lattice as _make_lattice

        lattice = _make_lattice(_topo, _n, J=1.0, h=h)
        H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
        circuit, _ = spec.create_circuit(_n, _p, lattice, **spec.circuit_kwargs)

        backend = self._resolve_backend()
        return float(backend.evaluate(circuit, H, theta))

    def predict_mpnn_at_h(
        self,
        predictor,
        h: float,
        *,
        topology: str | None = None,
        n_qubits: int | None = None,
    ) -> np.ndarray:
        """Run MPNN inference at a single h-value and return predicted θ.

        Encapsulates the repeated 10-line pattern of building a PyG Data object
        from the lattice graph and running predictor inference. This pattern
        appears 20+ times across runners.

        Requires: torch, torch_geometric available (lazy import).

        Parameters
        ----------
        predictor : MPNNPredictor
            Trained MPNN model (must be in eval mode).
        h : float
            Transverse field value for node features.
        topology : str | None
            Lattice topology. Default: self._args.topology.
        n_qubits : int | None
            System size. Default: self._args.n_qubits.

        Returns
        -------
        np.ndarray
            Predicted parameter vector θ_pred (1D, shape=(n_params,)).
        """
        import numpy as np
        import torch
        from torch_geometric.data import Data

        _topo = self._resolve_topology(topology)
        _n = n_qubits or getattr(self._args, "n_qubits", 6)

        # Reuse self.builder if available (from setup_physics), else create
        _builder = getattr(self, "builder", None)
        if _builder is None:
            from qmbp_simulation import HamiltonianBuilder

            _builder = HamiltonianBuilder()

        _make_lattice = getattr(self, "make_lattice", None)
        if _make_lattice is None:
            from qmbp_simulation import make_lattice as _make_lattice

        lattice = _make_lattice(_topo, _n, J=1.0, h=h)
        edge_index_np, coord = _builder.build_graph_data(lattice)

        h_feat = np.full(_n, float(h))
        x = torch.tensor(np.stack([h_feat, coord.astype(float)], axis=1), dtype=torch.float32)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        data.batch = torch.zeros(_n, dtype=torch.long)

        with torch.no_grad():
            theta_pred = predictor(data).numpy().flatten()

        return theta_pred

    @staticmethod
    def safe_per_h_loop(
        h_values: list[float],
        fn: Callable[[float], dict | None],
        label: str = "operation",
    ) -> list[dict]:
        """Execute a function for each h-value with error isolation.

        Each h-point is wrapped in try/except. Failed points are skipped
        with a warning, and the loop continues. NaN/Inf results from `fn`
        should return None to signal a skip.

        This is the canonical pattern for noisy estimation loops where
        individual h-points can fail without invalidating the entire section.

        Parameters
        ----------
        h_values : list[float]
            h-values to iterate over.
        fn : callable
            Function(h) -> dict with results for that h-point, or None to skip.
        label : str
            Description for log messages (e.g., "noisy_estimate", "PEA-ZNE").

        Returns
        -------
        list[dict]
            Results for successfully completed h-points. May be shorter
            than h_values if some points failed.

        Example
        -------
        >>> def estimate_h(h):
        ...     e = noisy_estimate(circuit, H, backend, config)
        ...     if not np.isfinite(e):
        ...         return None
        ...     return {"h": h, "energy": e}
        >>> results = self.safe_per_h_loop(h_values, estimate_h, "noisy")
        """
        results: list[dict] = []
        n_failed = 0
        for h in h_values:
            try:
                result = fn(h)
                if result is not None:
                    results.append(result)
                else:
                    n_failed += 1
                    logger.warning("    ⚠️  %s skipped at h=%.4f (returned None)", label, h)
            except Exception as e:
                n_failed += 1
                logger.warning("    ⚠️  %s failed at h=%.4f: %s", label, h, e)
        if n_failed > 0:
            logger.info(
                "    %s: %d/%d succeeded, %d skipped/failed",
                label,
                len(results),
                len(h_values),
                n_failed,
            )
        return results

    def exact_ground_state(
        self,
        topology: str,
        n_qubits: int,
        h: float,
        *,
        model: str = "tfim",
        model_kwargs: dict | None = None,
    ) -> tuple[float, float]:
        """Compute exact ground energy and gap for a single h-point.

        Uses ClassicalSolver (eigsh-based, memory-safe for any N).
        Reusable by any runner that needs a reference energy.

        Results are cached at two levels:
        1. In-memory dict (self._gt_cache) — fast, per-run only.
        2. Disk-persistent GroundTruthCache (data/ground_truth_cache.json) —
           survives across sessions, avoids redundant DMRG for large N.

        Parameters
        ----------
        topology : str
        n_qubits : int
        h : float
        model : str
        model_kwargs : dict | None

        Returns
        -------
        tuple[float, float]
            (ground_energy, spectral_gap)
        """
        # Cache key: round h to 2 decimals for cache key stability (matches generate_h_grid)
        cache_key = (model, topology, n_qubits, round(h, 2))
        if model_kwargs:
            cache_key = (*cache_key, tuple(sorted(model_kwargs.items())))

        # Level 1: in-memory cache (per-run, instant)
        if cache_key in self._gt_cache:
            return self._gt_cache[cache_key]

        # Level 2: disk-persistent cache (cross-session, avoids DMRG recompute)
        # Only use for standard models without custom kwargs (key format mismatch)
        if not model_kwargs:
            try:
                import numpy as np

                from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

                disk_cache = getattr(self, "_disk_gt_cache", None)
                if disk_cache is None:
                    disk_cache = GroundTruthCache()
                    self._disk_gt_cache = disk_cache
                cached = disk_cache.get(topology, n_qubits, model, h)
                if cached is not None:
                    # Invalidate stale floor gaps: if N > 18 and gap ≈ 2π/N,
                    # this was computed before the analytical gap fix and must
                    # be recomputed. The analytical estimate is always > 2π/N
                    # in the paramagnetic regime.
                    cached_gap = float(cached["gap"])
                    gap_floor = 2 * np.pi / n_qubits if n_qubits > 0 else 0
                    is_stale_floor = n_qubits > 18 and abs(cached_gap - gap_floor) < 1e-4
                    if not is_stale_floor:
                        result = (float(cached["energy"]), cached_gap)
                        self._gt_cache[cache_key] = result
                        return result
                    # else: fall through to recompute with analytical gap
            except (ImportError, OSError):
                pass  # Disk cache unavailable — proceed to compute

        from qmbp_simulation import make_lattice
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.solvers import ClassicalSolver

        spec = get_model_spec(model)
        if model_kwargs:
            spec = spec.with_params(**model_kwargs)
        lattice = make_lattice(topology, n_qubits, J=1.0, h=h)
        H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
        # Reuse self.solver if setup_physics() was called, else create fresh
        solver = getattr(self, "solver", None) or ClassicalSolver()
        gt = solver.solve(H, lattice)
        result = (float(gt.ground_energy), float(gt.gap))

        # ── Gap validation: warn if gap ≤ 0 (invalid for ΔE/gap metric) ──
        if gt.gap <= 0:
            logger.warning(
                "exact_ground_state: gap=%.2e ≤ 0 for %s N=%d h=%.4f. "
                "This makes ΔE/gap undefined. Possible causes: "
                "DMRG excited state converged to GS, or degenerate ground state.",
                gt.gap,
                topology,
                n_qubits,
                h,
            )

        self._gt_cache[cache_key] = result

        # Persist to disk cache for cross-session reuse.
        # Flush immediately because ValidationRunner uses os._exit()
        if not model_kwargs:
            try:
                disk_cache = getattr(self, "_disk_gt_cache", None)
                if disk_cache is not None:
                    disk_cache.put_from_result(topology, n_qubits, model, h, gt)
                    disk_cache.flush()
            except (OSError, AttributeError):
                pass  # Non-fatal: disk cache write failed

        return result

    @staticmethod
    def compute_vqe_restarts(p_layers: int, n_qubits: int = 10) -> int:
        """Compute recommended VQE restart count based on circuit complexity."""
        base = {1: 1, 2: 3, 3: 5, 4: 7}.get(p_layers, 7)
        if n_qubits >= 16 and p_layers >= 3:
            base += 2
        return base

    @staticmethod
    def default_h_test_values(
        p_layers: int,
        topology: str = "chain_1d",
        for_hardware: bool = False,
    ) -> list[float]:
        """Generate default h-test values for a given circuit depth.

        Selects h-values within the expressible regime for the given p,
        avoiding the zone below h_boundary where the ansatz cannot converge.

        Parameters
        ----------
        p_layers : int
            HVA circuit depth.
        topology : str
            Lattice topology (heavy_hex has higher h_boundary).
        for_hardware : bool
            If True, selects fewer points in the safe zone (for QPU cost control).

        Returns
        -------
        list[float]
            Descending h-values suitable for testing.
        """
        # h_boundary estimates (from F13: validated across N=10-20)
        if topology in ("ladder", "square"):
            offset = 0.5
        elif topology == "triangular":
            offset = 1.5
        elif topology == "heavy_hex":
            offset = 0.2
        else:
            offset = 0.0

        if p_layers <= 1:
            base_h = [4.0, 3.5, 3.25, 3.0]
        elif p_layers == 2:
            base_h = [3.0, 2.5, 2.0, 1.8, 1.6]
        elif p_layers == 3:
            base_h = [2.5, 2.0, 1.7, 1.5, 1.4]
        else:
            base_h = [2.0, 1.7, 1.5, 1.4, 1.3]

        # Apply topology offset (shift upward for harder topologies)
        adjusted = [h + offset for h in base_h]

        if for_hardware:
            # Fewer points, only safe zone (margin above h_boundary)
            return adjusted[:3]
        return adjusted

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

        # ── Input validation ──────────────────────────────────────────
        if not h_values:
            raise ValueError("vqe_descending_sweep: h_values cannot be empty.")
        if n_restarts < 1:
            raise ValueError(
                f"vqe_descending_sweep: n_restarts must be ≥ 1, got {n_restarts}. "
                "Use n_restarts=1 for a single optimization (no random restarts)."
            )
        # Deduplicate h_values (preserve order, warn if duplicates found)
        h_sorted = sorted(set(h_values), reverse=True)
        if len(h_sorted) < len(h_values):
            logger.warning(
                "vqe_descending_sweep: %d duplicate h-values removed (had %d, now %d).",
                len(h_values) - len(h_sorted),
                len(h_values),
                len(h_sorted),
            )

        spec = get_model_spec(model)
        backend = self._resolve_backend()
        mkw = model_kwargs or {}

        rng = np.random.default_rng(seed)
        lattice_ref = make_lattice(topology, n_qubits, J=1.0, h=h_sorted[0])
        circuit, _ = spec.create_circuit(n_qubits, p_layers, lattice_ref, **spec.circuit_kwargs)
        n_params = circuit.num_parameters
        prev_theta = rng.uniform(-0.01, 0.01, n_params)

        results: dict[float, np.ndarray] = {}
        for h in h_sorted:
            lattice = make_lattice(topology, n_qubits, J=1.0, h=h)
            H = spec.build_hamiltonian(lattice, **{**spec.hamiltonian_kwargs, **mkw})

            best_energy = float("inf")
            best_theta = prev_theta.copy()
            # Cap total function evaluations (same formula as VQEOptimizer._run_minimize)
            maxfun = maxiter * min(n_params + 5, 50)
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
                    options={"maxiter": maxiter, "ftol": 1e-14, "maxfun": maxfun},
                )
                if np.isfinite(res.fun) and res.fun < best_energy:
                    best_energy = res.fun
                    best_theta = res.x.copy()

            # NaN/Inf guard: if all restarts produced non-finite results,
            # keep previous theta (warm-start chain preservation)
            if not np.all(np.isfinite(best_theta)):
                logger.warning(
                    "vqe_descending_sweep: NaN/Inf at h=%.4f, preserving previous theta.",
                    h,
                )
                best_theta = prev_theta.copy()

            prev_theta = best_theta.copy()
            results[h] = best_theta.copy()

        return results

    def vqe_adaptive_sweep(
        self,
        topology: str,
        n_qubits: int,
        h_values: list[float],
        seed: int,
        *,
        p_layers: int = 1,
        n_restarts: int = 5,
        maxiter: int = 500,
        model: str = "tfim",
        model_kwargs: dict | None = None,
        adaptive_restarts: bool = True,
        ascending_pass: bool = True,
        selective_ascending: bool = True,
        ascending_de_gap_threshold: float = 0.02,
        checkpoint_label: str | None = None,
        compute_fidelity: bool = True,
    ) -> list[dict]:
        """Run a full adaptive VQE sweep with optional selective ascending pass.

        This is the reusable, feature-complete VQE sweep that includes:
        - Descending warm-start sweep (h_max → h_min)
        - Adaptive restarts per h-point (fewer for easy points, more near h_c)
        - Selective ascending pass (only re-optimizes suspicious points)
        - Per-point checkpointing for crash recovery

        Parameters
        ----------
        topology : str
            Lattice topology name.
        n_qubits : int
            Number of qubits.
        h_values : list[float]
            Transverse field values (descending order expected).
        seed : int
            Random seed for reproducibility.
        p_layers : int
            HVA circuit depth.
        n_restarts : int
            Maximum restarts per h-point. With adaptive_restarts=True,
            actual restarts are ≤ this value.
        maxiter : int
            Maximum optimizer iterations per restart.
        model : str
            Model name from registry.
        model_kwargs : dict | None
            Extra kwargs for Hamiltonian construction.
        adaptive_restarts : bool
            If True, dynamically adjust restarts per point. Default True.
        ascending_pass : bool
            If True, run the ascending pass. Default True.
        selective_ascending : bool
            If True, only re-optimize suspicious points in ascending pass.
            If False, re-optimize all points. Default True.
        ascending_de_gap_threshold : float
            ΔE/gap threshold for marking points as suspicious. Default 0.02.
        checkpoint_label : str | None
            If provided, save/load checkpoints with this label.
        compute_fidelity : bool
            If True, compute state fidelity (requires solver.ground_state_vector).
            Disable for large N where eigsh is expensive. Default True.

        Returns
        -------
        list[dict]
            Per-point results with keys: h, energy_vqe, energy_exact, gap,
            de_gap, fidelity, theta_opt, converged, n_iterations,
            n_restarts_used, elapsed_s.
        """
        import time as _time

        import numpy as np

        from qmbp_simulation import VQEConfig, VQEOptimizer, make_lattice
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.optimizers.sweep_strategies import (
            AdaptiveRestartConfig,
            SelectiveAscendingConfig,
            compute_adaptive_restarts,
            select_suspicious_points,
        )

        # ── Input validation ──────────────────────────────────────────
        if not h_values:
            raise ValueError("vqe_adaptive_sweep: h_values cannot be empty.")
        if n_restarts < 1:
            raise ValueError(f"vqe_adaptive_sweep: n_restarts must be ≥ 1, got {n_restarts}.")

        spec = get_model_spec(model)
        mkw = model_kwargs or {}
        backend = self._resolve_backend()

        # Build circuit
        lattice_ref = make_lattice(topology, n_qubits, J=1.0, h=h_values[0])
        circuit, _ = spec.create_circuit(n_qubits, p_layers, lattice_ref, **spec.circuit_kwargs)
        n_params = circuit.num_parameters

        # VQE optimizer
        vqe_config = VQEConfig(
            p_layers=p_layers,
            n_restarts=n_restarts,
            maxiter=maxiter,
            method="L-BFGS-B",
            enable_callbacks=False,
        )
        optimizer = VQEOptimizer(config=vqe_config, backend=backend, seed=seed)

        # Adaptive restart config
        h_crit = self.H_CRITICAL_ESTIMATES.get(model)
        adaptive_cfg = AdaptiveRestartConfig(
            base_restarts=max(1, n_restarts // 3),
            max_restarts=n_restarts,
            critical_restarts=max(2, n_restarts // 2),
            h_critical=h_crit,
        )

        # Initialize
        rng = np.random.default_rng(seed)
        prev_theta = rng.uniform(-0.01, 0.01, n_params)
        results: list[dict] = []

        # Reusable solver for fidelity computation (avoid re-instantiation per point)
        _solver = None
        if compute_fidelity:
            from qmbp_simulation.solvers.classical import ClassicalSolver

            _solver = ClassicalSolver()

        # Resume from checkpoint if available
        start_idx = 0
        if checkpoint_label:
            cp = self.load_checkpoint(checkpoint_label)
            if cp is not None:
                results = cp.get("results", [])
                prev_theta = np.array(cp.get("current_theta", prev_theta))
                start_idx = len(results)
                logger.info(
                    "    ♻️ Resuming sweep: %d/%d points already computed",
                    start_idx,
                    len(h_values),
                )

        # ── Descending pass ──────────────────────────────────────────────
        logger.info(
            "    VQE sweep: %s N=%d p=%d, %d h-points, adaptive_restarts=%s",
            topology,
            n_qubits,
            p_layers,
            len(h_values),
            adaptive_restarts,
        )

        for idx, h in enumerate(h_values):
            if idx < start_idx:
                continue
            t0 = _time.perf_counter()

            lattice_h = make_lattice(topology, n_qubits, J=1.0, h=h)
            H = spec.build_hamiltonian(lattice_h, **{**spec.hamiltonian_kwargs, **mkw})

            # Exact reference
            e_exact, gap = self.exact_ground_state(
                topology,
                n_qubits,
                h,
                model=model,
                model_kwargs=mkw,
            )

            # Fidelity (optional — expensive for large N)
            gs = None
            if compute_fidelity:
                gs = _solver.ground_state_vector(H)

            # Adaptive restarts
            if adaptive_restarts:
                prev_de_gap = results[-1]["de_gap"] if results else None
                n_r = compute_adaptive_restarts(h, prev_de_gap=prev_de_gap, config=adaptive_cfg)
                optimizer.config.n_restarts = n_r

            # Optimize
            vqe_result = optimizer.optimize(
                hamiltonian=H,
                circuit=circuit,
                initial_guess=prev_theta,
                exact_energy=e_exact,
                exact_state=gs,
            )

            de_gap = abs(vqe_result.energy - e_exact) / max(gap, 1e-10)
            elapsed = _time.perf_counter() - t0
            prev_theta = vqe_result.theta_opt.copy()

            results.append(
                {
                    "h": h,
                    "energy_vqe": vqe_result.energy,
                    "energy_exact": e_exact,
                    "gap": gap,
                    "de_gap": de_gap,
                    "fidelity": vqe_result.fidelity,
                    "energy_variance": vqe_result.energy_variance,
                    "theta_opt": vqe_result.theta_opt.tolist(),
                    "converged": vqe_result.n_iterations > 0,
                    "n_iterations": vqe_result.n_iterations,
                    "n_restarts_used": (
                        vqe_result.trajectory.n_restarts_used if vqe_result.trajectory else 0
                    ),
                    "elapsed_s": elapsed,
                }
            )

            status = "✓" if de_gap < 0.01 else ("?" if de_gap < DE_GAP_THRESHOLD else "✗")
            logger.info(
                "    [%s] h=%.4f: ΔE/gap=%.2e F=%.4f (%.1fs, r=%d)",
                status,
                h,
                de_gap,
                vqe_result.fidelity or 0,
                elapsed,
                optimizer.config.n_restarts,
            )

            # Checkpoint after each point
            if checkpoint_label:
                self.save_checkpoint(
                    checkpoint_label,
                    {
                        "results": results,
                        "current_theta": prev_theta.tolist(),
                        "n_completed": len(results),
                        "n_total": len(h_values),
                    },
                )

        # ── Selective ascending pass ─────────────────────────────────────
        if ascending_pass and results:
            selective_cfg = SelectiveAscendingConfig(
                de_gap_threshold=ascending_de_gap_threshold,
                include_neighbors=True,
            )

            if selective_ascending:
                target_indices, asc_report = select_suspicious_points(
                    results,
                    config=selective_cfg,
                )
            else:
                # Full ascending: all points
                target_indices = list(range(len(results) - 2, -1, -1))
                asc_report = None

            n_improved = 0
            if target_indices:
                n_targeted = len(target_indices)
                logger.info(
                    "    🔄 Ascending pass: targeting %d/%d points",
                    n_targeted,
                    len(results),
                )

                # Use fewer restarts for ascending pass
                optimizer.config.n_restarts = max(1, adaptive_cfg.base_restarts)

                for idx in target_indices:
                    # Propagate from neighbor
                    if idx < len(results) - 1:
                        asc_theta = np.array(results[idx + 1]["theta_opt"])
                    else:
                        asc_theta = np.array(results[-1]["theta_opt"])

                    h = h_values[idx]
                    lattice_h = make_lattice(topology, n_qubits, J=1.0, h=h)
                    H = spec.build_hamiltonian(lattice_h, **{**spec.hamiltonian_kwargs, **mkw})

                    e_exact = results[idx]["energy_exact"]
                    gs = None
                    if compute_fidelity:
                        gs = _solver.ground_state_vector(H)

                    vqe_asc = optimizer.optimize(
                        hamiltonian=H,
                        circuit=circuit,
                        initial_guess=asc_theta,
                        exact_energy=e_exact,
                        exact_state=gs,
                    )

                    if vqe_asc.energy < results[idx]["energy_vqe"]:
                        old_energy = results[idx]["energy_vqe"]
                        old_de_gap = results[idx]["de_gap"]
                        # Only count as improvement if meaningful (>1e-8)
                        energy_improvement = old_energy - vqe_asc.energy
                        if energy_improvement < 1e-8:
                            asc_theta = np.array(results[idx]["theta_opt"])
                            continue
                        gap = results[idx]["gap"]
                        de_gap = abs(vqe_asc.energy - e_exact) / max(gap, 1e-10)
                        results[idx]["energy_vqe"] = vqe_asc.energy
                        results[idx]["theta_opt"] = vqe_asc.theta_opt.tolist()
                        results[idx]["fidelity"] = vqe_asc.fidelity
                        results[idx]["de_gap"] = de_gap
                        n_improved += 1
                        logger.info(
                            "      ↑ h=%.4f improved: ΔE/gap %.2e → %.2e (ΔE=%.6f)",
                            h,
                            old_de_gap,
                            de_gap,
                            old_energy - vqe_asc.energy,
                        )

                logger.info("    🔄 Ascending result: %d/%d improved", n_improved, n_targeted)
            else:
                logger.info("    🔄 Ascending pass: no suspicious points, skipped")

        # ── Sweep summary ────────────────────────────────────────────────
        if results:
            from qmbp_simulation.models.constants import DE_GAP_THRESHOLD

            de_gaps = [r["de_gap"] for r in results if r["de_gap"] is not None]
            total_time = sum(r["elapsed_s"] for r in results)
            n_pass = sum(1 for d in de_gaps if d < DE_GAP_THRESHOLD)
            logger.info(
                "    📊 Sweep summary: %d/%d pass (ΔE/gap<5%%), avg=%.2e, max=%.2e, total=%.1fs",
                n_pass,
                len(de_gaps),
                np.mean(de_gaps) if de_gaps else 0,
                np.max(de_gaps) if de_gaps else 0,
                total_time,
            )

        # Cleanup checkpoint on success
        if checkpoint_label:
            self.cleanup_checkpoints(checkpoint_label)

        return results

    def vqe_noisy_sweep(
        self,
        topology: str,
        n_qubits: int,
        h_values: list[float],
        seed: int,
        *,
        p_layers: int = 1,
        n_restarts: int = 15,
        maxiter: int = 2000,
        shots: int = 8192,
        model: str = "tfim",
        model_kwargs: dict | None = None,
    ) -> dict[float, np.ndarray]:
        """Run a descending VQE sweep with NoisyBackend (shot noise).

        Parallel to vqe_descending_sweep() but uses NoisyBackend with COBYLA
        optimizer (gradient-free, suitable for shot-based evaluation). Returns
        the same h → θ_opt mapping.

        The key difference: θ_opt(noisy) ≠ θ_opt(noiseless). The noise shifts
        the energy landscape, and the optimizer adapts to this shift. An MPNN
        trained on noisy θ learns this compensation implicitly.

        Parameters
        ----------
        topology : str
            Lattice topology name.
        n_qubits : int
            Number of qubits.
        h_values : list[float]
            Transverse field values to sweep (sorted descending internally).
        seed : int
            Random seed for reproducibility.
        p_layers : int
            HVA circuit depth (default: 1).
        n_restarts : int
            Number of VQE restarts per h-point (default: 15 for noisy).
        maxiter : int
            Maximum COBYLA iterations per restart (default: 2000).
        shots : int
            Shots for NoisyBackend Gaussian approximation (default: 8192).
        model : str
            Model name from registry.
        model_kwargs : dict | None
            Extra kwargs for Hamiltonian/circuit construction.

        Returns
        -------
        dict[float, np.ndarray]
            Mapping from h-value to optimized parameter vector.

        Notes
        -----
        - Uses COBYLA explicitly (COBYLA_AUTO_SWITCH_THRESHOLD=8 already triggers
          for bond-resolved, but we force it for clarity and smaller param counts).
        - 10-50× slower than noiseless due to shot-based evaluation.
        - Requires more restarts (noise makes landscape rougher).
        - θ_opt(noisy) has higher variance across seeds — use 5-10 seeds.
        """
        import numpy as np

        from qmbp_simulation import VQEConfig, VQEOptimizer, make_lattice
        from qmbp_simulation.execution import NoisyBackend
        from qmbp_simulation.models.model_registry import get_model_spec

        if not h_values:
            raise ValueError("vqe_noisy_sweep: h_values cannot be empty.")

        h_sorted = sorted(set(h_values), reverse=True)
        spec = get_model_spec(model)
        mkw = model_kwargs or {}

        noisy_backend = NoisyBackend(shots=shots, seed_simulator=seed)
        vqe_config = VQEConfig(
            p_layers=p_layers,
            n_restarts=n_restarts,
            maxiter=maxiter,
            method="COBYLA",
        )
        optimizer = VQEOptimizer(config=vqe_config, backend=noisy_backend, seed=seed)

        rng = np.random.default_rng(seed)
        lattice_ref = make_lattice(topology, n_qubits, J=1.0, h=h_sorted[0])
        circuit, _ = spec.create_circuit(n_qubits, p_layers, lattice_ref, **spec.circuit_kwargs)
        n_params = circuit.num_parameters
        prev_theta = rng.uniform(-0.01, 0.01, n_params)

        results: dict[float, np.ndarray] = {}
        for h in h_sorted:
            lattice = make_lattice(topology, n_qubits, J=1.0, h=h)
            H = spec.build_hamiltonian(lattice, **{**spec.hamiltonian_kwargs, **mkw})

            vqe_result = optimizer.optimize(
                hamiltonian=H,
                circuit=circuit,
                initial_guess=prev_theta,
            )

            if np.all(np.isfinite(vqe_result.theta_opt)):
                prev_theta = vqe_result.theta_opt.copy()
            else:
                logger.warning("vqe_noisy_sweep: NaN/Inf at h=%.4f, preserving previous theta.", h)

            results[h] = prev_theta.copy()

        return results

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

        from qmbp_simulation.models.constants import DE_GAP_THRESHOLD

        de_gaps = [
            abs(vqe_energies[i] - exact_energies[i]) / max(gaps[i], 1e-10)
            for i in range(len(vqe_energies))
        ]
        n_pass = sum(1 for d in de_gaps if d < DE_GAP_THRESHOLD)
        return {
            "de_gaps": de_gaps,
            "n_pass": n_pass,
            "mean_de_gap": float(np.mean(de_gaps)) if de_gaps else 0.0,
            "pass_rate": n_pass / len(de_gaps) if de_gaps else 0.0,
        }

    @staticmethod
    def compute_theta_smoothness(theta_array) -> float:
        """Compute max L-inf change between consecutive θ vectors."""
        from qmbp_simulation.analysis.metrics import compute_theta_smoothness as _compute

        return _compute(theta_array) or 0.0

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
        from qmbp_simulation.analysis.metrics import compute_variational_violations

        per_point = [
            {"e_vqe": e_vqe, "e_exact": e_exact}
            for e_vqe, e_exact in zip(vqe_energies, exact_energies, strict=False)
        ]
        result = compute_variational_violations(per_point, tolerance=tolerance)
        return result["n_violations"]

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
        mpnn_epochs: int = 3000,
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
        from scipy.optimize import minimize

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
            from qmbp_simulation.predictors.mpnn import predict_theta as _pt

            _preds = _pt(predictor, lattice_t, [h_t])
            theta_mpnn = _preds[float(h_t)]

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
        mpnn_epochs: int = 2000,
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

            from qmbp_simulation.predictors.mpnn import predict_theta as _pt

            _preds = _pt(fold_model, lattice_t, [h_held])
            theta_pred = _preds[float(h_held)]

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
        mpnn_epochs: int = 3000,
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
            from qmbp_simulation.predictors.mpnn import predict_theta as _pt

            _preds = _pt(predictor, lattice_t, [h_t])
            theta_pred = _preds[float(h_t)]

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
        mpnn_epochs: int = 3000,
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

            from qmbp_simulation.predictors.mpnn import predict_theta as _pt

            _preds = _pt(predictor, lattice_t, [h_t])
            theta_pred = _preds[float(h_t)]

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
            "--p-layers",
            type=int,
            default=1,
            choices=[1, 2, 3, 4, 5, 6, 7, 8],
            help="HVA circuit depth (default: %(default)s)",
        )
        parser.add_argument(
            "--topology",
            type=str,
            default="heavy_hex",
            help="Lattice topology (default: %(default)s)",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="tfim",
            help="Model from registry (default: %(default)s)",
        )
        parser.add_argument(
            "--model-params",
            type=str,
            default=None,
            help="Model-specific parameters as key=value pairs, comma-separated.",
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
        Handles import errors and backend init failures gracefully.
        """
        try:
            from qmbp_simulation.execution.hardware import HardwareBackend
        except ImportError as e:
            raise RuntimeError(
                f"Hardware backend dependencies not installed: {e}. "
                f"Install qiskit-ibm-runtime and qiskit-ibm-catalog."
            ) from e

        hw_config = self.build_hardware_config()

        try:
            self.hw_backend = HardwareBackend(config=hw_config)  # type: ignore[assignment, no-redef]
        except Exception as e:
            if self._args.mode == "hardware":
                raise RuntimeError(
                    f"Failed to initialize HardwareBackend in hardware mode: {e}. "
                    f"Check IBM_KEY and IBM_INSTANCE_CRN environment variables."
                ) from e
            else:
                raise RuntimeError(
                    f"Failed to initialize HardwareBackend in fake_backend mode: {e}."
                ) from e

        # Share the runner's StructuredLogger with the backend
        if hasattr(self.hw_backend, "_logger"):
            self.hw_backend._logger = self.slog  # type: ignore[attr-defined]

        # Parse model params if --model-params provided
        self.parse_model_params()

        logger.info(f"  Hardware backend: {hw_config.mode} ({hw_config.backend_name})")
        logger.info(f"  Shots: {hw_config.shots}, Layouts: {hw_config.n_layouts}")
        logger.info(
            f"  System: N={self._args.n_qubits}, p={getattr(self._args, 'p_layers', 1)}, "
            f"model={getattr(self._args, 'model', 'tfim')}, topology={self._args.topology}"
        )

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
                "p_layers": getattr(self._args, "p_layers", 1),
                "topology": self._args.topology,
                "model": getattr(self._args, "model", "tfim"),
                "model_params": getattr(self, "_model_params", {}),
            },
            "hardware": {
                "mode": self._args.mode,
                "shots": self._args.shots,
                "n_layouts": self._args.n_layouts,
                "backend": getattr(self._args, "backend", "ibm_kingston"),
                "zne_amplifier": getattr(self._args, "zne_amplifier", "pea"),
                "zne_noise_factors": getattr(self._args, "zne_noise_factors", None),
                "zne_r2_threshold": getattr(self._args, "zne_r2_threshold", 0.90),
                "layout_strategy": getattr(self._args, "layout_strategy", "lowest_cost"),
                "use_mapomatic": not getattr(self._args, "no_mapomatic", False),
            },
            "seeds": getattr(self._args, "seeds", []),
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
