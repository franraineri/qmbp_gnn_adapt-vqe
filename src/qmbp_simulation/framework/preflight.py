"""Reusable pre-flight validation for pipeline variant configurations.

Validates experiment configurations before execution to catch common errors:
- Data leakage (h_test in training set)
- Invalid regime violations (h_test or h_values outside valid regime)
- Extrapolation risk (h_test outside training range)
- Output directory collisions
- Missing script dependencies

Designed to work with any variant runner script or PipelineVariant list.

Usage (CLI):
    # Validate variants from a runner script
    python -m qmbp_simulation.framework.preflight \\
        --from-script scripts/experiment_runners/run_p1_pipeline_variants_r2.py

    # Validate a JSON variant file
    python -m qmbp_simulation.framework.preflight --from-json variants.json

    # Validate with custom valid regime overrides
    python -m qmbp_simulation.framework.preflight \\
        --from-script my_script.py --strict

Usage (programmatic):
    from qmbp_simulation.framework.preflight import (
        PreflightChecker, VariantSpec, PreflightReport,
    )

    specs = [VariantSpec(id="V1", topo="chain_1d", n=10, p=1,
                         h_values=[4.0, 3.5, 3.0], h_test=[2.75],
                         output_dir="results/v1")]
    checker = PreflightChecker(specs)
    report = checker.run_all()
    report.print_summary()
    if report.has_errors:
        sys.exit(1)
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Tolerance for floating-point comparisons (h-values are typically multiples of 0.25)
_FLOAT_TOL = 1e-9

# ═══════════════════════════════════════════════════════════════════════════════
# Valid regime boundaries (canonical source — other modules should import from here)
# ═══════════════════════════════════════════════════════════════════════════════

P1_VALID_REGIME: dict[tuple[str, int], float] = {
    ("chain_1d", 6): 1.6,
    ("chain_1d", 10): 1.9,
    ("chain_1d", 20): 2.25,
    ("heavy_hex", 6): 2.0,
    ("heavy_hex", 10): 2.5,
    ("ladder", 6): 2.0,
    ("ladder", 10): 3.0,
    ("triangular", 6): 4.0,
    ("triangular", 10): 3.5,
}

P2_VALID_REGIME: dict[tuple[str, int], float] = {
    ("chain_1d", 6): 1.25,
    ("chain_1d", 10): 1.5,
    ("chain_1d", 20): 2.0,
    ("heavy_hex", 6): 1.5,
    ("heavy_hex", 10): 1.5,
    ("ladder", 6): 1.5,
    ("ladder", 10): 2.0,
    ("triangular", 6): 2.0,
    ("triangular", 10): 2.5,
}


def get_valid_regime(p: int) -> dict[tuple[str, int], float]:
    """Return the valid regime dict for a given p value."""
    if p == 1:
        return P1_VALID_REGIME
    if p == 2:
        return P2_VALID_REGIME
    msg = f"No valid regime defined for p={p} (only p=1 and p=2 supported)"
    raise ValueError(msg)


def get_regime_threshold(topology: str, n_qubits: int, p: int) -> float:
    """Get the valid regime threshold for a specific configuration.

    Returns 0.0 if no threshold is defined (permissive fallback).
    """
    regime = get_valid_regime(p)
    return regime.get((topology, n_qubits), 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════════


class Severity(Enum):
    """Issue severity level."""

    ERROR = "error"  # Must fix before running
    WARNING = "warning"  # Review but may be acceptable
    INFO = "info"  # Informational only


@dataclass
class Issue:
    """A single validation issue found during preflight checks."""

    severity: Severity
    check_name: str
    variant_id: str
    message: str

    @property
    def icon(self) -> str:
        """Terminal icon for this severity."""
        return {"error": "❌", "warning": "⚠️ ", "info": "ℹ️ "}[self.severity.value]


@dataclass
class VariantSpec:
    """Minimal specification of a variant for preflight validation.

    This is intentionally decoupled from PipelineVariant to allow validation
    of configurations before they are fully built into commands.
    """

    id: str
    topology: str
    n_qubits: int
    p: int
    h_values: list[float]
    h_test: list[float]
    output_dir: str
    script_path: str | None = None  # Pipeline script this variant will invoke
    seeds: list[int] = field(default_factory=list)

    @classmethod
    def from_pipeline_variant(cls, variant: Any) -> VariantSpec:
        """Extract a VariantSpec from a PipelineVariant by parsing its command.

        Parses the command list to extract topology, n_qubits, p, h_values,
        h_test, and output_dir.
        """
        cmd = variant.command
        spec = cls(
            id=variant.id,
            topology=_extract_arg(cmd, "--topology", "unknown"),
            n_qubits=int(_extract_arg(cmd, "--n-qubits", "0")),
            p=int(_extract_arg(cmd, "--p", "2")),
            h_values=_extract_float_list(cmd, "--h-values"),
            h_test=_extract_float_list(cmd, "--h-test"),
            output_dir=variant.output_dir,
            script_path=cmd[1] if len(cmd) > 1 else None,
            seeds=[int(s) for s in _extract_multi_arg(cmd, "--seed")],
        )
        return spec

    @classmethod
    def from_dict(cls, d: dict) -> VariantSpec:
        """Create from a dictionary (JSON-compatible format)."""
        return cls(
            id=d["id"],
            topology=d.get("topo", d.get("topology", "unknown")),
            n_qubits=d.get("n", d.get("n_qubits", 0)),
            p=d.get("p", 2),
            h_values=d.get("h_values", []),
            h_test=d.get("h_test", []),
            output_dir=d.get("output", d.get("output_dir", "")),
            script_path=d.get("script_path"),
            seeds=d.get("seeds", []),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PreflightReport:
    """Aggregated results from all preflight checks."""

    issues: list[Issue] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)
    n_variants: int = 0

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def infos(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.INFO]

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def is_clean(self) -> bool:
        return len(self.issues) == 0

    def print_summary(self) -> None:
        """Print a formatted summary to stdout."""
        print("\n" + "=" * 70)
        print("  PREFLIGHT SUMMARY")
        print("=" * 70)
        print(f"\n  Variants checked: {self.n_variants}")
        print(f"  Checks run:       {len(self.checks_run)}")
        print(f"  Errors:           {len(self.errors)}")
        print(f"  Warnings:         {len(self.warnings)}")

        if self.errors:
            print("\n  ❌ ERRORS (must fix before running):")
            for e in self.errors:
                print(f"    • [{e.variant_id}] {e.message}")

        if self.warnings:
            print("\n  ⚠️  WARNINGS (review but may be acceptable):")
            for w in self.warnings:
                print(f"    • [{w.variant_id}] {w.message}")

        if not self.has_errors and not self.warnings:
            print("\n  ✅ ALL CHECKS PASSED — safe to execute!")
        elif not self.has_errors:
            print("\n  ✅ No blocking errors — safe to execute (review warnings)")
        else:
            print("\n  ❌ FIX ERRORS BEFORE EXECUTING!")

        print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# Checker
# ═══════════════════════════════════════════════════════════════════════════════


class PreflightChecker:
    """Validates variant configurations before execution.

    Each check method returns a list of Issues. The `run_all()` method
    executes all checks and returns a PreflightReport.

    Parameters
    ----------
    specs : list[VariantSpec]
        Variant specifications to validate.
    project_root : Path | None
        Project root for resolving relative paths. Defaults to cwd.
    strict : bool
        If True, treat warnings as errors (useful for CI).
    """

    def __init__(
        self,
        specs: list[VariantSpec],
        *,
        project_root: Path | None = None,
        strict: bool = False,
    ) -> None:
        self.specs = specs
        self.root = project_root or Path.cwd()
        self.strict = strict

    # ─── Individual checks ─────────────────────────────────────────────────

    def check_script_exists(self) -> list[Issue]:
        """Verify that referenced pipeline scripts exist on disk."""
        issues: list[Issue] = []
        seen: set[str] = set()

        for spec in self.specs:
            if spec.script_path is None or spec.script_path in seen:
                continue
            seen.add(spec.script_path)
            path = self.root / spec.script_path
            if not path.exists():
                issues.append(
                    Issue(
                        severity=Severity.ERROR,
                        check_name="script_exists",
                        variant_id=spec.id,
                        message=f"Pipeline script not found: {spec.script_path}",
                    )
                )
        return issues

    def check_h_test_unseen(self) -> list[Issue]:
        """Verify h_test values are NOT in the training set (data leakage)."""
        issues: list[Issue] = []
        for spec in self.specs:
            for ht in spec.h_test:
                if _float_in_list(ht, spec.h_values):
                    issues.append(
                        Issue(
                            severity=Severity.ERROR,
                            check_name="h_test_unseen",
                            variant_id=spec.id,
                            message=(
                                f"h_test={ht} IS in training h_values={spec.h_values} "
                                f"(data leakage!)"
                            ),
                        )
                    )
        return issues

    def check_h_test_valid_regime(self) -> list[Issue]:
        """Verify h_test values are within the valid regime."""
        issues: list[Issue] = []
        for spec in self.specs:
            threshold = get_regime_threshold(spec.topology, spec.n_qubits, spec.p)
            if threshold == 0.0:
                continue  # No regime defined — skip
            for ht in spec.h_test:
                if ht < threshold:
                    issues.append(
                        Issue(
                            severity=Severity.ERROR,
                            check_name="h_test_valid_regime",
                            variant_id=spec.id,
                            message=(
                                f"h_test={ht} < {threshold} "
                                f"(outside valid regime for {spec.topology} "
                                f"N={spec.n_qubits} p={spec.p})"
                            ),
                        )
                    )
        return issues

    def check_h_values_valid_regime(self) -> list[Issue]:
        """Warn if training h_values fall outside the valid regime."""
        issues: list[Issue] = []
        for spec in self.specs:
            threshold = get_regime_threshold(spec.topology, spec.n_qubits, spec.p)
            if threshold == 0.0:
                continue
            below = [h for h in spec.h_values if h < threshold]
            if below:
                issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        check_name="h_values_valid_regime",
                        variant_id=spec.id,
                        message=(
                            f"Training h_values {below} < {threshold} "
                            f"(VQE may not converge at these points)"
                        ),
                    )
                )
        return issues

    def check_interpolation(self) -> list[Issue]:
        """Warn if h_test falls outside the training range (extrapolation)."""
        issues: list[Issue] = []
        for spec in self.specs:
            if not spec.h_values:
                continue
            h_min = min(spec.h_values)
            h_max = max(spec.h_values)
            for ht in spec.h_test:
                if not (h_min <= ht <= h_max):
                    issues.append(
                        Issue(
                            severity=Severity.WARNING,
                            check_name="interpolation",
                            variant_id=spec.id,
                            message=(
                                f"h_test={ht} outside training range "
                                f"[{h_min}, {h_max}] (extrapolation risk)"
                            ),
                        )
                    )
        return issues

    def check_output_fresh(self) -> list[Issue]:
        """Warn if output directories already contain results."""
        issues: list[Issue] = []
        seen_dirs: set[str] = set()

        for spec in self.specs:
            # Resolve seed templates
            out_dir = spec.output_dir
            if "{seed}" in out_dir:
                # Check each seed variant
                for seed in spec.seeds or [42]:
                    resolved = out_dir.format(seed=seed)
                    if resolved in seen_dirs:
                        continue
                    seen_dirs.add(resolved)
                    self._check_single_output(resolved, spec.id, issues)
            else:
                if out_dir in seen_dirs:
                    continue
                seen_dirs.add(out_dir)
                self._check_single_output(out_dir, spec.id, issues)

        return issues

    def _check_single_output(self, out_dir: str, variant_id: str, issues: list[Issue]) -> None:
        """Check a single output directory for existing results."""
        path = self.root / out_dir
        if path.exists():
            json_files = list(path.glob("*.json"))
            if json_files:
                issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        check_name="output_fresh",
                        variant_id=variant_id,
                        message=(
                            f"Output dir already has {len(json_files)} JSON file(s): {out_dir}"
                        ),
                    )
                )

    def check_descending_sweep(self) -> list[Issue]:
        """Verify h_values are in descending order (warm-start requirement)."""
        issues: list[Issue] = []
        for spec in self.specs:
            if len(spec.h_values) < 2:
                continue
            if spec.h_values != sorted(spec.h_values, reverse=True):
                issues.append(
                    Issue(
                        severity=Severity.ERROR,
                        check_name="descending_sweep",
                        variant_id=spec.id,
                        message=(
                            f"h_values not descending: {spec.h_values} "
                            f"(warm-start requires h=high→low)"
                        ),
                    )
                )
        return issues

    def check_duplicate_ids(self) -> list[Issue]:
        """Check for duplicate variant IDs."""
        issues: list[Issue] = []
        seen: dict[str, int] = {}
        for spec in self.specs:
            if spec.id in seen:
                issues.append(
                    Issue(
                        severity=Severity.ERROR,
                        check_name="duplicate_ids",
                        variant_id=spec.id,
                        message=f"Duplicate variant ID (first seen at index {seen[spec.id]})",
                    )
                )
            else:
                seen[spec.id] = len(seen)
        return issues

    def check_minimum_config(self) -> list[Issue]:
        """Check that variants have minimum required configuration.

        Catches specs with empty h_values, empty h_test, or missing topology.
        """
        issues: list[Issue] = []
        for spec in self.specs:
            if not spec.h_values:
                issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        check_name="minimum_config",
                        variant_id=spec.id,
                        message="No h_values defined (training set is empty)",
                    )
                )
            if not spec.h_test:
                issues.append(
                    Issue(
                        severity=Severity.INFO,
                        check_name="minimum_config",
                        variant_id=spec.id,
                        message="No h_test defined (no generalization test point)",
                    )
                )
            if spec.topology == "unknown":
                issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        check_name="minimum_config",
                        variant_id=spec.id,
                        message="Topology not specified or could not be parsed",
                    )
                )
            if spec.n_qubits == 0:
                issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        check_name="minimum_config",
                        variant_id=spec.id,
                        message="n_qubits=0 (not specified or could not be parsed)",
                    )
                )
        return issues

    # ─── Orchestration ─────────────────────────────────────────────────────

    def run_all(self, *, verbose: bool = True) -> PreflightReport:
        """Execute all checks and return a consolidated report.

        Parameters
        ----------
        verbose : bool
            If True, print per-check results as they run.
        """
        report = PreflightReport(n_variants=len(self.specs))

        checks = [
            ("script_exists", self.check_script_exists),
            ("minimum_config", self.check_minimum_config),
            ("h_test_unseen", self.check_h_test_unseen),
            ("h_test_valid_regime", self.check_h_test_valid_regime),
            ("h_values_valid_regime", self.check_h_values_valid_regime),
            ("interpolation", self.check_interpolation),
            ("descending_sweep", self.check_descending_sweep),
            ("duplicate_ids", self.check_duplicate_ids),
            ("output_fresh", self.check_output_fresh),
        ]

        if verbose:
            print("=" * 70)
            print("  PRE-FLIGHT CHECKS")
            print("=" * 70)

        for i, (name, check_fn) in enumerate(checks, 1):
            issues = check_fn()
            report.checks_run.append(name)
            report.issues.extend(issues)

            if verbose:
                n_err = sum(1 for x in issues if x.severity == Severity.ERROR)
                n_warn = sum(1 for x in issues if x.severity == Severity.WARNING)
                n_info = sum(1 for x in issues if x.severity == Severity.INFO)
                if not issues:
                    print(f"\n  [{i}] {name}: ✅ all pass")
                else:
                    parts = []
                    if n_err:
                        parts.append(f"{n_err} errors")
                    if n_warn:
                        parts.append(f"{n_warn} warnings")
                    if n_info:
                        parts.append(f"{n_info} info")
                    print(f"\n  [{i}] {name}: {', '.join(parts)}")
                    for issue in issues:
                        if issue.severity != Severity.INFO:
                            print(f"      {issue.icon} {issue.variant_id}: {issue.message}")

        # In strict mode, promote warnings to errors
        if self.strict:
            for issue in report.issues:
                if issue.severity == Severity.WARNING:
                    issue.severity = Severity.ERROR

        if verbose:
            report.print_summary()

        return report


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_arg(cmd: list[str], flag: str, default: str = "") -> str:
    """Extract a single-value argument from a command list."""
    try:
        idx = cmd.index(flag)
        return cmd[idx + 1] if idx + 1 < len(cmd) else default
    except ValueError:
        return default


def _float_in_list(value: float, lst: list[float]) -> bool:
    """Check if a float value is in a list using tolerance-based comparison."""
    return any(math.isclose(value, x, abs_tol=_FLOAT_TOL) for x in lst)


def _extract_float_list(cmd: list[str], flag: str) -> list[float]:
    """Extract a multi-value float argument from a command list.

    Collects all values after the flag until the next flag (starts with --).
    """
    try:
        idx = cmd.index(flag)
    except ValueError:
        return []

    values: list[float] = []
    for item in cmd[idx + 1 :]:
        if item.startswith("--"):
            break
        try:
            values.append(float(item))
        except ValueError:
            break
    return values


def _extract_multi_arg(cmd: list[str], flag: str) -> list[str]:
    """Extract all occurrences of a single-value flag from a command list."""
    values: list[str] = []
    for i, item in enumerate(cmd):
        if item == flag and i + 1 < len(cmd):
            values.append(cmd[i + 1])
    return values


# ═══════════════════════════════════════════════════════════════════════════════
# Loaders — extract VariantSpecs from different sources
# ═══════════════════════════════════════════════════════════════════════════════


def specs_from_pipeline_variants(variants: list[Any]) -> list[VariantSpec]:
    """Convert a list of PipelineVariant objects to VariantSpecs."""
    return [VariantSpec.from_pipeline_variant(v) for v in variants]


def specs_from_json(path: str | Path) -> list[VariantSpec]:
    """Load VariantSpecs from a JSON file.

    Expected format:
    [
        {"id": "V1", "topo": "chain_1d", "n": 10, "p": 1,
         "h_values": [4.0, 3.5, 3.0], "h_test": [2.75],
         "output": "results/v1"},
        ...
    ]
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "variants" in data:
        data = data["variants"]
    return [VariantSpec.from_dict(d) for d in data]


def specs_from_variant_runner(
    build_noiseless: Any,
    build_noisy: Any,
    build_extended: Any,
    n_qubits: int,
) -> list[VariantSpec]:
    """Build specs by calling variant builder functions directly.

    This is the preferred integration path for variant runner scripts.
    """
    all_variants = build_noiseless(n_qubits) + build_noisy(n_qubits) + build_extended(n_qubits)
    return specs_from_pipeline_variants(all_variants)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════


def _load_specs_from_script(script_path: str) -> list[VariantSpec]:
    """Dynamically import a variant runner script and extract its variants.

    Looks for build_noiseless_variants, build_noisy_variants,
    build_extended_variants functions (the standard pattern).
    Falls back to VARIANTS list if present (legacy pattern).
    """
    import importlib.util

    path = Path(script_path).resolve()
    if not path.exists():
        print(f"  ❌ Script not found: {script_path}")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("_variant_module", path)
    if spec is None or spec.loader is None:
        print(f"  ❌ Cannot load module from: {script_path}")
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)

    # Protect against sys.argv being parsed by the imported module
    # (e.g., if it calls argparse at module level)
    original_argv = sys.argv
    sys.argv = [str(path)]

    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except SystemExit:
        # Some scripts call sys.exit() at module level via argparse errors
        pass
    except Exception as e:
        print(f"  ❌ Error loading script: {script_path}")
        print(f"     {type(e).__name__}: {e}")
        sys.exit(1)
    finally:
        sys.argv = original_argv

    # Try standard variant runner pattern
    build_noiseless = getattr(module, "build_noiseless_variants", None)
    build_noisy = getattr(module, "build_noisy_variants", None)
    build_extended = getattr(module, "build_extended_variants", None)
    default_n = getattr(module, "DEFAULT_N_QUBITS", 10)

    if build_noiseless is not None:
        build_noisy = build_noisy or (lambda n: [])
        build_extended = build_extended or (lambda n: [])
        return specs_from_variant_runner(build_noiseless, build_noisy, build_extended, default_n)

    # Try legacy VARIANTS list pattern
    variants_list = getattr(module, "VARIANTS", None)
    if variants_list is not None:
        return [VariantSpec.from_dict(v) for v in variants_list]

    print(f"  ❌ No variant definitions found in: {script_path}")
    print("     Expected: build_noiseless_variants() or VARIANTS list")
    sys.exit(1)


def main() -> None:
    """CLI entry point for preflight checks."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-flight validation for pipeline variant configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Validate a variant runner script
  python -m qmbp_simulation.framework.preflight \\
      --from-script scripts/experiment_runners/run_p1_pipeline_variants_r2.py

  # Validate a JSON file
  python -m qmbp_simulation.framework.preflight --from-json variants.json

  # Strict mode (warnings become errors)
  python -m qmbp_simulation.framework.preflight --from-script my_script.py --strict
""",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--from-script",
        metavar="PATH",
        help="Path to a variant runner script (imports and extracts variants)",
    )
    source.add_argument(
        "--from-json",
        metavar="PATH",
        help="Path to a JSON file with variant definitions",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (useful for CI)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary (no per-check details)",
    )
    parser.add_argument(
        "--project-root",
        metavar="PATH",
        default=None,
        help="Project root directory (default: auto-detect)",
    )

    args = parser.parse_args()

    # Resolve project root
    if args.project_root:
        root = Path(args.project_root).resolve()
    else:
        # Walk up from cwd looking for Makefile or src/
        root = Path.cwd()
        for parent in [root, *root.parents]:
            if (parent / "Makefile").exists() or (parent / "src").exists():
                root = parent
                break

    # Load specs
    if args.from_script:
        specs = _load_specs_from_script(args.from_script)
    else:
        specs = specs_from_json(args.from_json)

    if not specs:
        print("  ❌ No variant specs loaded. Nothing to check.")
        sys.exit(1)

    print(f"\n  Loaded {len(specs)} variant specs")
    print(f"  Project root: {root}")

    # Run checks
    checker = PreflightChecker(specs, project_root=root, strict=args.strict)
    report = checker.run_all(verbose=not args.quiet)

    if args.quiet:
        report.print_summary()

    sys.exit(1 if report.has_errors else 0)


if __name__ == "__main__":
    main()
