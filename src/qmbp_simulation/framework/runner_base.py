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
        all_h = []
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

        Returns
        -------
        int
            Exit code: 0 if all sections pass, 1 otherwise.
        """
        t_total = time.time()

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

        # ─── Step 3: Execute sections ────────────────────────────────────
        self._print_header(selected)

        for section in selected:
            result = self._execute_section(section)
            self._section_results.append(result)

            # Stop-on-failure: abort remaining sections
            if not result.success and self._args.stop_on_failure:
                logger.warning(
                    f"  Stopping early (--stop-on-failure). Section {section.id} failed."
                )
                break

        # ─── Step 4: Save results ────────────────────────────────────────
        total_elapsed = time.time() - t_total
        envelope = self._build_envelope(total_elapsed)
        saved_path = save_experiment_result(envelope, experiment_id=self.experiment_id)

        # Save structured log independently for post-hoc analysis
        log_path = saved_path.parent / f"log_{saved_path.stem.replace('run_', '')}.json"
        self.slog.save(log_path)

        # ─── Step 5: Summary ─────────────────────────────────────────────
        self._print_summary(total_elapsed, saved_path)

        n_fail = sum(1 for r in self._section_results if not r.success)
        return 1 if n_fail > 0 else 0

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
            )

        psi_trunc = psi_trunc.reshape(-1)
        norm = np.linalg.norm(psi_trunc)
        if norm > 1e-15:
            psi_trunc = psi_trunc / norm
        return psi_trunc


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

    Use this for runners that execute on real QPU or FakeTorino simulation.

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
            choices=["gate_folding", "pea"],
            default="gate_folding",
            help="ZNE noise amplification strategy (default: %(default)s). "
            "'pea' uses Probabilistic Error Amplification (learns noise model).",
        )
        parser.add_argument(
            "--zne-noise-factors",
            type=float,
            nargs="+",
            default=None,
            help="ZNE noise amplification factors (default: [1, 3, 5])",
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
        zne_amplifier = getattr(self._args, "zne_amplifier", "gate_folding")

        mitigation = MitigationOptions(
            dd_enabled=True,
            trex_enabled=True,
            twirling_enabled=True,
            zne_enabled=True,
            zne_amplifier=zne_amplifier,
            zne_noise_factors=zne_noise_factors,
            num_randomizations=32,
            shots_per_randomization=128,
        )

        return HardwareConfig(
            mode=self._args.mode,
            n_qubits=self._args.n_qubits,
            shots=self._args.shots,
            n_layouts=self._args.n_layouts,
            output_dir=f"results/hardware/{self.runner_id}",
            mitigation=mitigation,
        )

    def setup(self) -> None:
        """Initialize HardwareBackend + shared infrastructure.

        Override and call super().setup() first to retain hardware init.
        """
        from qmbp_simulation.execution.hardware import HardwareBackend

        hw_config = self.build_hardware_config()
        self.hw_backend = HardwareBackend(config=hw_config)

        # Share the runner's StructuredLogger with the backend
        self.hw_backend._logger = self.slog

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

        # Hardware preflight only in hardware mode (FakeTorino always passes)
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
                "backend": "ibm_torino",
                "zne_amplifier": getattr(self._args, "zne_amplifier", "gate_folding"),
                "zne_noise_factors": getattr(self._args, "zne_noise_factors", None),
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
