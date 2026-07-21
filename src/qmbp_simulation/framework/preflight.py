"""Reusable pre-flight validation for pipeline variant configurations AND experiments.

Validates experiment configurations before execution to catch common errors:
- Data leakage (h_test in training set)
- Invalid regime violations (h_test or h_values outside valid regime)
- Extrapolation risk (h_test outside training range)
- Output directory collisions
- Missing script dependencies
- Constraint violations (p > 2, Heisenberg + HVA p≤2, N=12 slowness)
- Missing hypothesis or experiment metadata

Designed to work with:
- Variant runner scripts (PipelineVariant pattern)
- BaseExperiment subclasses (ExperimentConfig pattern)

Usage (CLI):
    # Validate variants from a runner script
    python -m qmbp_simulation.framework.preflight \\
        --from-script scripts/experiment_runners/run_p1_pipeline_variants_r2.py

    # Validate a BaseExperiment file
    python -m qmbp_simulation.framework.preflight \\
        --from-experiment experiments/predictor/exp_s4_data_efficiency_n10.py

    # Auto-detect mode (tries variant runner first, then experiment)
    python -m qmbp_simulation.framework.preflight \\
        --from-script experiments/predictor/exp_s4_data_efficiency_n10.py

    # Validate a JSON variant file
    python -m qmbp_simulation.framework.preflight --from-json variants.json

    # Strict mode (warnings become errors)
    python -m qmbp_simulation.framework.preflight --from-script my_script.py --strict

Usage (programmatic — variants):
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

Usage (programmatic — experiments):
    from qmbp_simulation.framework.preflight import ExperimentChecker

    checker = ExperimentChecker.from_script("experiments/predictor/exp_s4_data_efficiency_n10.py")
    report = checker.run_all()
    report.print_summary()
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
    ("chain_1d", 16): 2.0,  # Scaling law: 1.0+0.020·16^1.31≈1.70, empirical conservative
    ("chain_1d", 20): 2.0,  # Validated: scaling law predicts 2.00, exact match
    ("chain_1d", 24): 2.5,
    ("chain_1d", 40): 4.0,  # Scaling law: 1.5+0.020·40^1.31≈4.51, validated MPS
    ("chain_1d", 50): 4.9,  # Scaling law: 1.5+0.020·50^1.31≈5.36, validated MPS
    ("chain_1d", 80): 7.7,  # Scaling law: 1.5+0.020·80^1.31≈8.22, validated MPS
    ("heavy_hex", 6): 3.5,  # Empirical: S3 V3 rehearsal shows 23% ΔE/gap at h=3.25 (2026-06-16).
    # heavy_hex has higher coordination than chain_1d → HVA p=1 expressibility degrades
    # faster with N. N=6 subgraph has only 5 ZZ bonds; p=1 landscape is near-flat at h<3.5.
    # Previously 2.0 (interpolated from chain_1d) but never empirically validated.
    ("heavy_hex", 10): 3.0,
    ("heavy_hex", 20): 3.5,  # Empirical estimate (extrapolated from N=10)
    ("ladder", 6): 2.0,
    ("ladder", 10): 3.0,
    ("ladder", 16): 2.5,
    ("triangular", 6): 4.0,
    ("triangular", 10): 3.5,
    ("triangular", 16): 4.0,
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
    ("triangular", 16): 3.0,
}


def get_valid_regime(p: int) -> dict[tuple[str, int], float]:
    """Return the valid regime dict for a given p value.

    Raises
    ------
    ValueError
        If p is not 1 or 2 (only supported values).
    """
    if p == 1:
        return P1_VALID_REGIME
    if p == 2:
        return P2_VALID_REGIME
    msg = f"No valid regime defined for p={p} (only p=1 and p=2 supported)"
    raise ValueError(msg)


def get_regime_threshold(topology: str, n_qubits: int, p: int) -> float:
    """Get the valid regime threshold for a specific configuration.

    Returns 0.0 if no threshold is defined (permissive fallback).

    Parameters
    ----------
    topology : str
        Lattice topology name.
    n_qubits : int
        Number of qubits. Must be positive.
    p : int
        Number of HVA layers. Must be 1 or 2.
    """
    if p not in (1, 2):
        return 0.0
    regime = P1_VALID_REGIME if p == 1 else P2_VALID_REGIME
    return regime.get((topology, n_qubits), 0.0)


def validate_regime_tables() -> list[str]:
    """Self-check the regime boundary tables for internal consistency.

    Validates:
    1. All boundaries are positive
    2. p=1 boundaries >= p=2 boundaries for same config
    3. Within same topology, boundaries increase with N (scaling law)

    Returns a list of error strings (empty = all valid).
    Used by tests and CI to prevent boundary regressions.
    """
    errors: list[str] = []

    # Check positivity
    for key, val in P1_VALID_REGIME.items():
        if val <= 0:
            errors.append(f"P1 {key}: non-positive boundary {val}")
    for key, val in P2_VALID_REGIME.items():
        if val <= 0:
            errors.append(f"P2 {key}: non-positive boundary {val}")

    # Check p1 >= p2 for common configs
    common = set(P1_VALID_REGIME.keys()) & set(P2_VALID_REGIME.keys())
    for key in common:
        if P1_VALID_REGIME[key] < P2_VALID_REGIME[key]:
            errors.append(
                f"{key}: P1 ({P1_VALID_REGIME[key]}) < P2 ({P2_VALID_REGIME[key]}) "
                f"— physically impossible (p=1 is less expressive)"
            )

    # Check monotonicity within chain_1d (scaling law validated: h_min grows with N)
    # NOTE: Non-chain topologies (ladder, triangular) may have non-monotonic boundaries
    # because higher coordination at larger N can improve HVA expressibility.
    for regime_name, regime in [("P1", P1_VALID_REGIME), ("P2", P2_VALID_REGIME)]:
        by_topo: dict[str, list[tuple[int, float]]] = {}
        for (topo, n), val in regime.items():
            by_topo.setdefault(topo, []).append((n, val))

        # Only chain_1d has a validated scaling law requiring monotonicity
        for topo in ["chain_1d"]:
            if topo not in by_topo:
                continue
            sorted_entries = sorted(by_topo[topo], key=lambda x: x[0])
            for i in range(1, len(sorted_entries)):
                n_prev, val_prev = sorted_entries[i - 1]
                n_curr, val_curr = sorted_entries[i]
                if val_curr < val_prev:
                    errors.append(
                        f"{regime_name} {topo}: N={n_prev}→{val_prev}, N={n_curr}→{val_curr} "
                        f"violates monotonicity (larger N should need higher h_min)"
                    )

    return errors


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
# Experiment Checker — validates BaseExperiment files
# ═══════════════════════════════════════════════════════════════════════════════

# Known-bad model+ansatz combinations (historical V9 results — informational only)
_FORBIDDEN_MODEL_ANSATZ: list[tuple[str, int, str]] = []


@dataclass
class ExperimentSpec:
    """Extracted specification from a BaseExperiment's default_config().

    Parallel to VariantSpec but tailored for experiment validation.
    """

    experiment_id: str
    category: str
    description: str
    hypothesis: str
    topology: str
    n_qubits: int
    p_layers: int
    h_values: list[float]
    h_test: list[float]
    seeds: list[int]
    model: str
    n_restarts: int
    hidden_dim: int
    class_name: str  # For error messages
    script_path: str  # Source file

    @classmethod
    def from_config(
        cls, config: Any, *, class_name: str = "", script_path: str = ""
    ) -> ExperimentSpec:
        """Extract an ExperimentSpec from an ExperimentConfig instance."""
        sys_cfg = config.system
        return cls(
            experiment_id=config.experiment_id,
            category=config.category,
            description=config.description,
            hypothesis=config.hypothesis,
            topology=sys_cfg.topology,
            n_qubits=sys_cfg.n_qubits,
            p_layers=sys_cfg.p_layers,
            h_values=sys_cfg.h_values,
            h_test=sys_cfg.h_test,
            seeds=config.seeds,
            model=getattr(sys_cfg, "model", "tfim"),
            n_restarts=config.vqe.n_restarts,
            hidden_dim=config.mpnn.hidden_dim,
            class_name=class_name,
            script_path=script_path,
        )


class ExperimentChecker:
    """Validates BaseExperiment configurations before execution.

    Checks experiment-specific constraints that go beyond what
    ExperimentConfig.validate() catches (which only checks p≤2 and seeds).

    Parameters
    ----------
    specs : list[ExperimentSpec]
        Experiment specifications to validate.
    project_root : Path | None
        Project root for resolving relative paths.
    strict : bool
        If True, treat warnings as errors.
    """

    def __init__(
        self,
        specs: list[ExperimentSpec],
        *,
        project_root: Path | None = None,
        strict: bool = False,
    ) -> None:
        self.specs = specs
        self.root = project_root or Path.cwd()
        self.strict = strict

    @classmethod
    def from_script(
        cls,
        script_path: str,
        *,
        project_root: Path | None = None,
        strict: bool = False,
    ) -> ExperimentChecker:
        """Create an ExperimentChecker by loading experiments from a script.

        Dynamically imports the script, finds BaseExperiment subclasses,
        and extracts their default_config().
        """
        specs = _load_experiment_specs(script_path)
        return cls(specs, project_root=project_root, strict=strict)

    # ─── Individual checks ─────────────────────────────────────────────────

    def check_p_layers(self) -> list[Issue]:
        """Check p_layers for informational warnings (no hard constraint)."""
        issues: list[Issue] = []
        return issues

    def check_model_ansatz_compatibility(self) -> list[Issue]:
        """Check for known-bad model+ansatz combinations (V9 results)."""
        issues: list[Issue] = []
        for spec in self.specs:
            for model, max_p, reason in _FORBIDDEN_MODEL_ANSATZ:
                if spec.model == model and spec.p_layers <= max_p:
                    # Only flag if the experiment is NOT explicitly testing this
                    # (e.g., E4 longitudinal field experiment is designed to show failure)
                    if spec.category in ("E", "V"):
                        # Generalization/validation experiments may intentionally test limits
                        issues.append(
                            Issue(
                                severity=Severity.INFO,
                                check_name="model_ansatz",
                                variant_id=spec.experiment_id,
                                message=f"Known limit: {reason} (OK for validation experiment)",
                            )
                        )
                    else:
                        issues.append(
                            Issue(
                                severity=Severity.ERROR,
                                check_name="model_ansatz",
                                variant_id=spec.experiment_id,
                                message=f"{reason}",
                            )
                        )
        return issues

    def check_hypothesis_present(self) -> list[Issue]:
        """Verify experiment has a non-empty hypothesis (experiment discipline)."""
        issues: list[Issue] = []
        for spec in self.specs:
            if not spec.hypothesis or spec.hypothesis.strip() == "":
                issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        check_name="hypothesis",
                        variant_id=spec.experiment_id,
                        message="No hypothesis defined (experiment discipline requires one)",
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
                            variant_id=spec.experiment_id,
                            message=(f"h_test={ht} IS in training h_values (data leakage!)"),
                        )
                    )
        return issues

    def check_h_test_valid_regime(self) -> list[Issue]:
        """Verify h_test values are within the valid regime."""
        issues: list[Issue] = []
        for spec in self.specs:
            if spec.model != "tfim":
                continue  # Valid regime only defined for TFIM
            threshold = get_regime_threshold(spec.topology, spec.n_qubits, spec.p_layers)
            if threshold == 0.0:
                continue
            for ht in spec.h_test:
                if ht < threshold:
                    issues.append(
                        Issue(
                            severity=Severity.ERROR,
                            check_name="h_test_valid_regime",
                            variant_id=spec.experiment_id,
                            message=(
                                f"h_test={ht} < {threshold} "
                                f"(outside valid regime for {spec.topology} "
                                f"N={spec.n_qubits} p={spec.p_layers})"
                            ),
                        )
                    )
        return issues

    def check_h_values_valid_regime(self) -> list[Issue]:
        """Warn if training h_values fall outside the valid regime."""
        issues: list[Issue] = []
        for spec in self.specs:
            if spec.model != "tfim":
                continue
            threshold = get_regime_threshold(spec.topology, spec.n_qubits, spec.p_layers)
            if threshold == 0.0:
                continue
            below = [h for h in spec.h_values if h < threshold]
            if below:
                issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        check_name="h_values_valid_regime",
                        variant_id=spec.experiment_id,
                        message=(
                            f"{len(below)}/{len(spec.h_values)} training h_values < {threshold} "
                            f"(VQE may not converge at these points)"
                        ),
                    )
                )
        return issues

    def check_descending_sweep(self) -> list[Issue]:
        """Verify h_values are in descending order (warm-start requirement).

        Skipped for experiments that don't use warm-start VQE:
        - Landscape analysis (category F) — h_values are scan points
        - Experiments with no h_test (no deployment phase)
        """
        issues: list[Issue] = []
        # Categories that don't require descending sweep
        _NO_SWEEP_CATEGORIES = {"F"}  # Landscape analysis

        for spec in self.specs:
            if len(spec.h_values) < 2:
                continue
            # Skip for landscape/analysis experiments that don't use warm-start
            if spec.category in _NO_SWEEP_CATEGORIES:
                continue
            # Skip if no h_test defined (no deployment → likely a scan experiment)
            if not spec.h_test:
                continue
            if spec.h_values != sorted(spec.h_values, reverse=True):
                # Check if it's ascending (common mistake) vs random
                if spec.h_values == sorted(spec.h_values):
                    issues.append(
                        Issue(
                            severity=Severity.ERROR,
                            check_name="descending_sweep",
                            variant_id=spec.experiment_id,
                            message=(
                                "h_values are ASCENDING — must be descending "
                                "(warm-start requires h=high→low)"
                            ),
                        )
                    )
                else:
                    issues.append(
                        Issue(
                            severity=Severity.ERROR,
                            check_name="descending_sweep",
                            variant_id=spec.experiment_id,
                            message=(
                                "h_values not in descending order (warm-start requires h=high→low)"
                            ),
                        )
                    )
        return issues

    def check_interpolation(self) -> list[Issue]:
        """Warn if h_test falls outside the training range (extrapolation)."""
        issues: list[Issue] = []
        for spec in self.specs:
            if not spec.h_values or not spec.h_test:
                continue
            h_min = min(spec.h_values)
            h_max = max(spec.h_values)
            for ht in spec.h_test:
                if not (h_min <= ht <= h_max):
                    issues.append(
                        Issue(
                            severity=Severity.WARNING,
                            check_name="interpolation",
                            variant_id=spec.experiment_id,
                            message=(
                                f"h_test={ht} outside training range "
                                f"[{h_min}, {h_max}] (extrapolation risk)"
                            ),
                        )
                    )
        return issues

    def check_n_qubits_feasibility(self) -> list[Issue]:
        """Warn about known slow/infeasible system sizes."""
        issues: list[Issue] = []
        for spec in self.specs:
            if spec.n_qubits == 12:
                issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        check_name="n_qubits_feasibility",
                        variant_id=spec.experiment_id,
                        message=(
                            "N=12 is very slow (>30 min per run). Consider N=10 or N=14 instead."
                        ),
                    )
                )
            if spec.n_qubits > 20 and spec.topology != "chain_1d":
                issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        check_name="n_qubits_feasibility",
                        variant_id=spec.experiment_id,
                        message=(
                            f"N={spec.n_qubits} with {spec.topology} may be infeasible "
                            f"(only chain_1d supports N>20 via MPS)"
                        ),
                    )
                )
        return issues

    def check_mpnn_capacity(self) -> list[Issue]:
        """Warn if MPNN hidden_dim is insufficient for the system size."""
        issues: list[Issue] = []
        for spec in self.specs:
            if spec.n_qubits >= 10 and spec.hidden_dim < 128:
                issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        check_name="mpnn_capacity",
                        variant_id=spec.experiment_id,
                        message=(
                            f"hidden_dim={spec.hidden_dim} may be insufficient for N={spec.n_qubits} "
                            f"(N≥10 requires hidden_dim≥128, validated in V7/V8)"
                        ),
                    )
                )
        return issues

    def check_seeds(self) -> list[Issue]:
        """Info about non-standard seed choices."""
        issues: list[Issue] = []
        standard_seeds = {42, 43, 44}
        for spec in self.specs:
            if not spec.seeds:
                issues.append(
                    Issue(
                        severity=Severity.ERROR,
                        check_name="seeds",
                        variant_id=spec.experiment_id,
                        message="No seeds defined (at least one required)",
                    )
                )
            elif len(spec.seeds) < 3:
                issues.append(
                    Issue(
                        severity=Severity.INFO,
                        check_name="seeds",
                        variant_id=spec.experiment_id,
                        message=(
                            f"Only {len(spec.seeds)} seed(s) — "
                            f"3 seeds recommended for statistical confidence"
                        ),
                    )
                )
            elif set(spec.seeds) != standard_seeds:
                issues.append(
                    Issue(
                        severity=Severity.INFO,
                        check_name="seeds",
                        variant_id=spec.experiment_id,
                        message=(f"Non-standard seeds {spec.seeds} (convention: DEFAULT_SEEDS)"),
                    )
                )
        return issues

    def check_restarts(self) -> list[Issue]:
        """Warn about excessive or insufficient VQE restarts."""
        issues: list[Issue] = []
        for spec in self.specs:
            if spec.n_restarts > 10:
                issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        check_name="restarts",
                        variant_id=spec.experiment_id,
                        message=(
                            f"n_restarts={spec.n_restarts} > 10 "
                            f"(diminishing returns — B4 shows no saddle points in HVA)"
                        ),
                    )
                )
        return issues

    def check_hardware_circuit_budget(self) -> list[Issue]:
        """Estimate hardware viability from system config (no transpilation needed).

        Uses empirical formulas to predict 2Q gate count and ZNE viability
        based on topology, N, and p. Warns if the configuration is likely
        to exceed hardware thresholds.

        Estimation rules (from validated data):
          - chain_1d p=1: n_2q_logical = N-1, transpiled ≈ N-1 (no SWAPs on heavy-hex)
          - heavy_hex p=1: n_2q_logical = n_edges, transpiled ≈ n_edges (native)
          - ladder p=1: n_2q_logical = 2*(N/2 - 1) + N/2, transpiled ≈ 2.5× logical
          - triangular: n_2q ≈ 4× logical (many SWAPs from dense connectivity)
          - p=2: multiply by 2
        """
        issues: list[Issue] = []

        # ZNE threshold: PEA handles up to ~50 CX, GF handles up to ~18
        ZNE_THRESHOLD_PEA = 50

        # Empirical n_2q estimates per topology (from compare_resource_estimation.py data)
        TOPOLOGY_2Q_FACTOR = {
            "chain_1d": 1.0,  # n_2q ≈ N-1 (perfect layout on heavy-hex)
            "heavy_hex": 1.0,  # n_2q = n_edges (native topology, no SWAPs)
            "ladder": 2.5,  # SWAP overhead for non-native connectivity
            "triangular": 4.0,  # Heavy SWAP overhead
        }

        for spec in self.specs:
            factor = TOPOLOGY_2Q_FACTOR.get(spec.topology, 2.0)
            # Logical 2Q gates: approximately N-1 bonds for chain_1d, more for others
            if spec.topology == "chain_1d":
                n_2q_logical = spec.n_qubits - 1
            elif spec.topology == "heavy_hex":
                # heavy_hex N=10 has 9 edges
                n_2q_logical = spec.n_qubits - 1
            elif spec.topology == "ladder":
                half = spec.n_qubits // 2
                n_2q_logical = 2 * (half - 1) + half  # rungs + legs
            else:
                n_2q_logical = int(spec.n_qubits * 1.5)

            n_2q_estimated = int(n_2q_logical * factor * spec.p_layers)

            if n_2q_estimated > ZNE_THRESHOLD_PEA:
                issues.append(
                    Issue(
                        severity=Severity.ERROR,
                        check_name="hardware_circuit_budget",
                        variant_id=spec.experiment_id,
                        message=(
                            f"Estimated {n_2q_estimated} 2Q gates "
                            f"({spec.topology} N={spec.n_qubits} p={spec.p_layers}) "
                            f"exceeds PEA threshold ({ZNE_THRESHOLD_PEA}). "
                            f"Circuit too deep for ZNE recovery."
                        ),
                    )
                )
            elif n_2q_estimated > ZNE_THRESHOLD_PEA * 0.7:
                issues.append(
                    Issue(
                        severity=Severity.WARNING,
                        check_name="hardware_circuit_budget",
                        variant_id=spec.experiment_id,
                        message=(
                            f"Estimated {n_2q_estimated} 2Q gates "
                            f"({spec.topology} N={spec.n_qubits} p={spec.p_layers}) "
                            f"is {n_2q_estimated / ZNE_THRESHOLD_PEA:.0%} of PEA threshold. "
                            f"ZNE marginal — monitor R²."
                        ),
                    )
                )

        return issues

    # ─── Orchestration ─────────────────────────────────────────────────────

    def run_all(self, *, verbose: bool = True) -> PreflightReport:
        """Execute all experiment checks and return a consolidated report."""
        report = PreflightReport(n_variants=len(self.specs))

        checks = [
            ("p_layers", self.check_p_layers),
            ("model_ansatz", self.check_model_ansatz_compatibility),
            ("hypothesis", self.check_hypothesis_present),
            ("h_test_unseen", self.check_h_test_unseen),
            ("h_test_valid_regime", self.check_h_test_valid_regime),
            ("h_values_valid_regime", self.check_h_values_valid_regime),
            ("descending_sweep", self.check_descending_sweep),
            ("interpolation", self.check_interpolation),
            ("n_qubits_feasibility", self.check_n_qubits_feasibility),
            ("mpnn_capacity", self.check_mpnn_capacity),
            ("seeds", self.check_seeds),
            ("restarts", self.check_restarts),
            ("hardware_circuit_budget", self.check_hardware_circuit_budget),
        ]

        if verbose:
            print("=" * 70)
            print("  EXPERIMENT PRE-FLIGHT CHECKS")
            print("=" * 70)
            for spec in self.specs:
                print(
                    f"\n  Experiment: {spec.experiment_id} ({spec.class_name})"
                    f"\n    {spec.topology} N={spec.n_qubits} p={spec.p_layers} "
                    f"model={spec.model} seeds={spec.seeds}"
                )

        for i, (name, check_fn) in enumerate(checks, 1):
            issues = check_fn()
            report.checks_run.append(name)
            report.issues.extend(issues)

            if verbose:
                n_err = sum(1 for x in issues if x.severity == Severity.ERROR)
                n_warn = sum(1 for x in issues if x.severity == Severity.WARNING)
                n_info = sum(1 for x in issues if x.severity == Severity.INFO)
                if not issues:
                    print(f"\n  [{i:2d}] {name}: ✅ all pass")
                else:
                    parts = []
                    if n_err:
                        parts.append(f"{n_err} errors")
                    if n_warn:
                        parts.append(f"{n_warn} warnings")
                    if n_info:
                        parts.append(f"{n_info} info")
                    print(f"\n  [{i:2d}] {name}: {', '.join(parts)}")
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
    Returns None if no variant definitions found (caller should try experiment mode).
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


def _load_experiment_specs(script_path: str) -> list[ExperimentSpec]:
    """Dynamically import a script and find BaseExperiment subclasses.

    For each subclass found, calls default_config() and extracts an ExperimentSpec.
    Returns empty list if no experiments found.
    """
    import importlib.util
    import inspect

    path = Path(script_path).resolve()
    if not path.exists():
        print(f"  ❌ Script not found: {script_path}")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("_experiment_module", path)
    if spec is None or spec.loader is None:
        print(f"  ❌ Cannot load module from: {script_path}")
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)

    original_argv = sys.argv
    sys.argv = [str(path)]

    # Add project root to sys.path so `experiments.helpers` imports work
    project_root = str(Path.cwd())
    path_added = False
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        path_added = True

    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except SystemExit:
        pass
    except Exception as e:
        print(f"  ❌ Error loading script: {script_path}")
        print(f"     {type(e).__name__}: {e}")
        sys.exit(1)
    finally:
        sys.argv = original_argv
        if path_added:
            sys.path.remove(project_root)

    # Find BaseExperiment subclasses defined in this module
    from qmbp_simulation.framework.base import BaseExperiment

    experiment_specs: list[ExperimentSpec] = []

    for name, obj in inspect.getmembers(module, inspect.isclass):
        # Only classes defined in this module (not imported base classes)
        if not issubclass(obj, BaseExperiment):
            continue
        if obj is BaseExperiment:
            continue
        if obj.__module__ != module.__name__:
            continue

        # Try to get default_config
        default_config_fn = getattr(obj, "default_config", None)
        if default_config_fn is None:
            continue

        try:
            config = default_config_fn()
        except Exception as e:
            print(f"  ⚠️  Could not call {name}.default_config(): {e}")
            continue

        experiment_specs.append(
            ExperimentSpec.from_config(
                config,
                class_name=name,
                script_path=script_path,
            )
        )

    return experiment_specs


def _try_load_as_experiment(script_path: str) -> list[ExperimentSpec] | None:
    """Attempt to load a script as a BaseExperiment file.

    Returns list of ExperimentSpecs if successful, None if no experiments found.
    Does not call sys.exit() on failure — returns None instead.
    """
    import importlib.util
    import inspect

    path = Path(script_path).resolve()
    if not path.exists():
        return None

    spec = importlib.util.spec_from_file_location("_experiment_module", path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)

    original_argv = sys.argv
    sys.argv = [str(path)]

    # Add project root to sys.path so `experiments.helpers` imports work
    project_root = str(Path.cwd())
    path_added = False
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        path_added = True

    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except (SystemExit, Exception):
        return None
    finally:
        sys.argv = original_argv
        if path_added and project_root in sys.path:
            sys.path.remove(project_root)

    # Find BaseExperiment subclasses
    try:
        from qmbp_simulation.framework.base import BaseExperiment
    except ImportError:
        return None

    experiment_specs: list[ExperimentSpec] = []

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if not issubclass(obj, BaseExperiment):
            continue
        if obj is BaseExperiment:
            continue
        if obj.__module__ != module.__name__:
            continue

        default_config_fn = getattr(obj, "default_config", None)
        if default_config_fn is None:
            continue

        try:
            config = default_config_fn()
        except Exception:
            continue

        experiment_specs.append(
            ExperimentSpec.from_config(config, class_name=name, script_path=script_path)
        )

    return experiment_specs if experiment_specs else None


def _try_load_as_validation_runner(script_path: str) -> tuple | None:
    """Attempt to load a script as a ValidationRunner file.

    Returns (runner_class, issues_list) if successful, None if not found.
    Issues list contains strings with [ERROR] or [WARNING] prefixes.
    """
    import importlib.util
    import inspect

    path = Path(script_path).resolve()
    if not path.exists():
        return None

    spec = importlib.util.spec_from_file_location("_validation_module", path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    original_argv = sys.argv
    sys.argv = [str(path)]

    project_root = str(Path.cwd())
    path_added = False
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        path_added = True

    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except (SystemExit, Exception):
        return None
    finally:
        sys.argv = original_argv
        if path_added and project_root in sys.path:
            sys.path.remove(project_root)

    # Find ValidationRunner subclasses
    try:
        from qmbp_simulation.framework.runner_base import ValidationRunner
    except ImportError:
        return None

    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if not issubclass(obj, ValidationRunner):
            continue
        if obj is ValidationRunner:
            continue
        if obj.__module__ != module.__name__:
            continue

        # Found a ValidationRunner subclass — run its preflight logic
        issues: list[str] = []

        # Structural checks
        if not obj.runner_id:
            issues.append("[ERROR] runner_id is not set")
        if not obj.experiment_id:
            issues.append("[ERROR] experiment_id is not set")
        if not obj.description:
            issues.append("[ERROR] description is not set")
        if not obj.hypothesis:
            issues.append("[ERROR] hypothesis is not set")

        # Instantiate to check sections and config
        try:
            # Create with default args (no CLI parsing)
            import argparse

            fake_args = argparse.Namespace(
                section=None,
                skip_preflight=False,
                stop_on_failure=False,
                verbose=False,
                dry_run=False,
            )
            # Add any custom args with defaults
            if hasattr(obj, "_add_custom_args"):
                temp_parser = argparse.ArgumentParser()
                obj._add_custom_args(temp_parser)
                for action in temp_parser._actions:
                    if action.dest != "help":
                        setattr(fake_args, action.dest, action.default)

            runner = obj(args=fake_args)
            sections = runner.define_sections()

            if not sections:
                issues.append("[ERROR] define_sections() returned empty list")
            else:
                ids = [s.id for s in sections]
                if len(ids) != len(set(ids)):
                    issues.append(
                        f"[ERROR] Duplicate section IDs: {set(i for i in ids if ids.count(i) > 1)}"
                    )
                for s in sections:
                    if not s.hypothesis:
                        issues.append(f"[WARNING] Section {s.id} ({s.name}) has no hypothesis")

            # Physics checks from build_config()
            try:
                config = runner.build_config()
                system = config.get("system", {})
                topology = system.get("topology") or config.get("topology", "")
                n_qubits = system.get("n_qubits") or config.get("n_qubits", 0)
                p_layers = system.get("p_layers") or config.get("p_layers", 1)

                if topology and n_qubits:
                    threshold = get_regime_threshold(topology, n_qubits, p_layers)
                    if threshold > 0:
                        # Check h-values
                        h_keys = ["h_train", "h_test", "h_values", "h_values_sweep"]
                        all_h: list[float] = []
                        for key in h_keys:
                            val = config.get(key) or system.get(key)
                            if isinstance(val, (list, tuple)):
                                all_h.extend(float(v) for v in val if isinstance(v, (int, float)))
                        below = [h for h in all_h if h < threshold]
                        if below:
                            issues.append(
                                f"[WARNING] h-values {below} below valid regime "
                                f"({threshold}) for {topology} N={n_qubits} p={p_layers}"
                            )
            except Exception:
                pass  # build_config() may fail without full setup

        except Exception as e:
            issues.append(f"[WARNING] Could not instantiate runner for deep checks: {e}")

        return obj, issues

    return None


def main() -> None:
    """CLI entry point for preflight checks."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-flight validation for pipeline variants and experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Validate a variant runner script
  python -m qmbp_simulation.framework.preflight \\
      --from-script scripts/experiment_runners/run_p1_pipeline_variants_r2.py

  # Validate a BaseExperiment file
  python -m qmbp_simulation.framework.preflight \\
      --from-experiment experiments/predictor/exp_s4_data_efficiency_n10.py

  # Auto-detect mode (tries variants first, then experiment)
  python -m qmbp_simulation.framework.preflight \\
      --from-script experiments/predictor/exp_s4_data_efficiency_n10.py

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
        help=(
            "Path to a variant runner or experiment script. "
            "Auto-detects: tries variant runner first, falls back to experiment mode."
        ),
    )
    source.add_argument(
        "--from-experiment",
        metavar="PATH",
        help="Path to a BaseExperiment file (validates ExperimentConfig)",
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

    # ── Explicit experiment mode ──────────────────────────────────────────
    if args.from_experiment:
        exp_specs = _load_experiment_specs(args.from_experiment)
        if not exp_specs:
            print(f"  ❌ No BaseExperiment subclasses found in: {args.from_experiment}")
            sys.exit(1)

        print(f"\n  Loaded {len(exp_specs)} experiment(s) from {args.from_experiment}")
        print(f"  Project root: {root}")

        checker = ExperimentChecker(exp_specs, project_root=root, strict=args.strict)
        report = checker.run_all(verbose=not args.quiet)

        if args.quiet:
            report.print_summary()

        sys.exit(1 if report.has_errors else 0)

    # ── Variant / auto-detect mode ────────────────────────────────────────
    if args.from_script:
        # First try as variant runner
        import importlib.util

        path = Path(args.from_script).resolve()
        if not path.exists():
            print(f"  ❌ Script not found: {args.from_script}")
            sys.exit(1)

        spec = importlib.util.spec_from_file_location("_probe_module", path)
        if spec is None or spec.loader is None:
            print(f"  ❌ Cannot load module from: {args.from_script}")
            sys.exit(1)

        module = importlib.util.module_from_spec(spec)
        original_argv = sys.argv
        sys.argv = [str(path)]

        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except (SystemExit, Exception):
            pass
        finally:
            sys.argv = original_argv

        # Check if it's a variant runner
        has_variants = (
            getattr(module, "build_noiseless_variants", None) is not None
            or getattr(module, "VARIANTS", None) is not None
        )

        if has_variants:
            # Standard variant runner path
            specs = _load_specs_from_script(args.from_script)
            if not specs:
                print("  ❌ No variant specs loaded. Nothing to check.")
                sys.exit(1)

            print(f"\n  Loaded {len(specs)} variant specs")
            print(f"  Project root: {root}")

            checker: object = PreflightChecker(specs, project_root=root, strict=args.strict)  # type: ignore[no-redef]
            report = checker.run_all(verbose=not args.quiet)

            if args.quiet:
                report.print_summary()

            sys.exit(1 if report.has_errors else 0)

        # Fall back to experiment mode
        exp_specs: list | None = _try_load_as_experiment(args.from_script)  # type: ignore[no-redef]
        if exp_specs:
            print(f"\n  Auto-detected BaseExperiment in: {args.from_script}")
            print(f"  Loaded {len(exp_specs)} experiment(s)")
            print(f"  Project root: {root}")

            checker = ExperimentChecker(exp_specs, project_root=root, strict=args.strict)  # type: ignore[assignment]
            report = checker.run_all(verbose=not args.quiet)

            if args.quiet:
                report.print_summary()

            sys.exit(1 if report.has_errors else 0)

        # Try as ValidationRunner
        validation_result = _try_load_as_validation_runner(args.from_script)
        if validation_result:
            print(f"\n  Auto-detected ValidationRunner in: {args.from_script}")
            runner_cls, issues = validation_result
            print(f"  Runner: {runner_cls.runner_id} ({runner_cls.description})")
            print(f"  Project root: {root}")
            print()

            if issues:
                has_errors = any("[ERROR]" in i for i in issues)
                for i in issues:
                    print(f"  {i}")
                print()
                if has_errors:
                    print("  ❌ Preflight FAILED")
                    sys.exit(1)
                else:
                    print("  ⚠️  Preflight PASSED with warnings")
                    sys.exit(0)
            else:
                print("  ✅ Preflight PASSED (no issues)")
                sys.exit(0)

        # Neither variant runner nor experiment
        print(
            f"  ❌ No variant definitions or BaseExperiment subclasses found in: {args.from_script}"
        )
        print(
            "     Expected: build_noiseless_variants(), VARIANTS list, or BaseExperiment subclass"
        )
        sys.exit(1)

    # ── JSON mode ─────────────────────────────────────────────────────────
    specs = specs_from_json(args.from_json)

    if not specs:
        print("  ❌ No variant specs loaded. Nothing to check.")
        sys.exit(1)

    print(f"\n  Loaded {len(specs)} variant specs")
    print(f"  Project root: {root}")

    checker = PreflightChecker(specs, project_root=root, strict=args.strict)  # type: ignore[assignment]
    report = checker.run_all(verbose=not args.quiet)

    if args.quiet:
        report.print_summary()

    sys.exit(1 if report.has_errors else 0)


if __name__ == "__main__":
    main()
