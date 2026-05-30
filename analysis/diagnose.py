#!/usr/bin/env python3
"""Automated failure diagnosis for GNN-HVA pipeline results.

Reads pipeline_run JSON files, classifies failure root causes, and produces
structured diagnostic reports. Designed to replace ad-hoc investigation scripts.

Root cause categories:
  - CHAIN_BREAK: θ_smoothness > 1.0 (warm-start chain disrupted by restarts)
  - MPNN_OVERFIT: generalization_gap > 0.01 (MPNN memorized training data)
  - HVA_LIMIT: error_from_circuit > 0 (ansatz cannot express ground state)
  - OUTSIDE_REGIME: h_test below valid regime boundary
  - VQE_DIVERGENCE: convergence_rate < 1.0 (VQE did not converge at some h)
  - BOUNDARY_EFFECT: h_test within 0.5 of valid regime boundary
  - UNKNOWN: none of the above detected

Usage:
    # Diagnose all failures in a results folder
    python analysis/diagnose.py results/thesis/p1_variants_N10_r2

    # Diagnose a specific variant
    python analysis/diagnose.py results/thesis/p1_variants_N10_r2/comp4_tri_p2_seed44

    # Diagnose all failures across all thesis results
    python analysis/diagnose.py --all

    # Only show FAIL verdicts (skip MARGINAL)
    python analysis/diagnose.py --all --severity fail

    # Export structured JSON
    python analysis/diagnose.py --all --json analysis/raw_data/diagnoses.json

    # Filter by topology or p
    python analysis/diagnose.py --all --topology triangular --p 1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
THESIS = RESULTS / "thesis"


# ─── Constants ───────────────────────────────────────────────────────────────

# Thresholds for root cause classification
THETA_SMOOTHNESS_CHAIN_BREAK = 1.0
GEN_GAP_OVERFIT = 0.01
CONVERGENCE_RATE_MIN = 1.0
BOUNDARY_PROXIMITY = 0.5  # h_test within this distance of regime boundary

# Valid regime boundaries
P1_VALID_REGIME: dict[tuple[str, int], float] = {
    ("chain_1d", 6): 1.6,
    ("chain_1d", 10): 1.9,
    ("chain_1d", 16): 2.3,
    ("chain_1d", 20): 2.25,
    ("chain_1d", 24): 2.5,
    ("ladder", 6): 2.0,
    ("ladder", 10): 2.0,
    ("ladder", 16): 2.5,
    ("triangular", 6): 3.0,
    ("triangular", 10): 3.5,
    ("triangular", 16): 4.0,
}

P2_VALID_REGIME: dict[tuple[str, int], float] = {
    ("chain_1d", 6): 1.25,
    ("chain_1d", 10): 1.5,
    ("chain_1d", 20): 2.0,
    ("ladder", 6): 1.5,
    ("ladder", 10): 2.0,
    ("triangular", 6): 2.0,
    ("triangular", 10): 2.5,
    ("triangular", 16): 3.0,
}

PASS_THRESHOLD = 0.05
MARGINAL_THRESHOLD = 0.10


# ─── Data Models ─────────────────────────────────────────────────────────────


class RootCause(str, Enum):
    """Enumeration of failure root causes."""

    CHAIN_BREAK = "CHAIN_BREAK"
    MPNN_OVERFIT = "MPNN_OVERFIT"
    HVA_LIMIT = "HVA_LIMIT"
    OUTSIDE_REGIME = "OUTSIDE_REGIME"
    VQE_DIVERGENCE = "VQE_DIVERGENCE"
    BOUNDARY_EFFECT = "BOUNDARY_EFFECT"
    UNKNOWN = "UNKNOWN"


@dataclass
class DeploymentPoint:
    """Diagnosis for a single h_test deployment point."""

    h_test: float | None
    de_gap: float | None
    predicted_energy: float | None = None
    phase_label: str | None = None

    @property
    def verdict(self) -> str:
        if self.de_gap is None:
            return "NO_DATA"
        if self.de_gap < PASS_THRESHOLD:
            return "PASS"
        if self.de_gap < MARGINAL_THRESHOLD:
            return "MARGINAL"
        return "FAIL"


@dataclass
class Diagnosis:
    """Complete diagnosis for a single pipeline run."""

    # Identity
    folder: str
    variant: str
    file: str

    # Config
    topology: str
    n_qubits: int | None
    p_layers: int | None
    seed: int | None
    n_restarts: int | None
    h_values: list[float] = field(default_factory=list)

    # Phase 2 diagnostics
    theta_smoothness: float | None = None
    convergence_rate: float | None = None
    per_h_converged: list[bool] = field(default_factory=list)

    # Phase 3 diagnostics
    generalization_gap: float | None = None
    per_h_mse: dict[str, float] = field(default_factory=dict)

    # Phase 4 results (per deployment point)
    deployment_points: list[DeploymentPoint] = field(default_factory=list)

    # Energy decomposition
    error_from_circuit: float | None = None
    error_from_mpnn: float | None = None

    # Timing
    elapsed_s: float | None = None

    # Computed diagnosis
    root_causes: list[RootCause] = field(default_factory=list)
    severity: str = "UNKNOWN"  # PASS, MARGINAL, FAIL, NO_DATA
    explanation: str = ""
    recommendations: list[str] = field(default_factory=list)

    @property
    def worst_de_gap(self) -> float | None:
        """Worst ΔE/gap across all deployment points."""
        gaps = [dp.de_gap for dp in self.deployment_points if dp.de_gap is not None]
        return max(gaps) if gaps else None

    @property
    def valid_regime_boundary(self) -> float:
        """Get the valid regime boundary for this config."""
        regime = P1_VALID_REGIME if self.p_layers == 1 else P2_VALID_REGIME
        return regime.get((self.topology, self.n_qubits or 0), 0.0)


# ─── Core Logic ──────────────────────────────────────────────────────────────


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    """Safely load a JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def parse_pipeline_run(path: Path, folder: str, variant: str) -> Diagnosis | None:
    """Parse a pipeline_run JSON into a Diagnosis object."""
    data = _safe_load_json(path)
    if data is None:
        return None

    # Resolve path for relative_to computation
    resolved_path = path.resolve()
    try:
        file_rel = str(resolved_path.relative_to(ROOT))
    except ValueError:
        file_rel = str(path)

    config = data.get("config", {})
    system = data.get("system", {})
    diag = data.get("diagnostics", {})
    p4_results = data.get("phase4_results", [])

    # Phase 2
    p2 = diag.get("phase2", {})
    # Phase 3
    p3 = diag.get("phase3", {})
    # Phase 4 diagnostics
    p4_diag = diag.get("phase4", {})
    decomp = p4_diag.get("energy_decomposition") if p4_diag else None

    # Build deployment points
    deploy_points = []
    for entry in p4_results:
        deploy_points.append(
            DeploymentPoint(
                h_test=entry.get("h_test"),
                de_gap=entry.get("delta_e_over_gap"),
                predicted_energy=entry.get("predicted_energy"),
                phase_label=entry.get("phase_label"),
            )
        )

    return Diagnosis(
        folder=folder,
        variant=variant,
        file=file_rel,
        topology=config.get("topology") or system.get("topology", "unknown"),
        n_qubits=config.get("n_qubits") or system.get("n_qubits"),
        p_layers=config.get("p_layers") or system.get("p_layers"),
        seed=config.get("seed"),
        n_restarts=config.get("n_restarts"),
        h_values=config.get("h_values", []),
        theta_smoothness=p2.get("theta_smoothness"),
        convergence_rate=p2.get("convergence_rate"),
        per_h_converged=p2.get("per_h_converged", []),
        generalization_gap=p3.get("generalization_gap"),
        per_h_mse=p3.get("per_h_mse", {}),
        deployment_points=deploy_points,
        error_from_circuit=decomp.get("error_from_circuit") if decomp else None,
        error_from_mpnn=decomp.get("error_from_mpnn") if decomp else None,
        elapsed_s=data.get("elapsed_s"),
    )


def classify_root_causes(diag: Diagnosis) -> None:
    """Classify the root cause(s) of a failure. Mutates diag in place."""
    causes: list[RootCause] = []
    explanations: list[str] = []
    recommendations: list[str] = []

    # Determine severity
    worst = diag.worst_de_gap
    if worst is None:
        if not diag.deployment_points:
            diag.severity = "NO_PHASE4"
        else:
            diag.severity = "NO_DATA"
    elif worst < PASS_THRESHOLD:
        diag.severity = "PASS"
    elif worst < MARGINAL_THRESHOLD:
        diag.severity = "MARGINAL"
    else:
        diag.severity = "FAIL"

    # Skip diagnosis for passing runs
    if diag.severity == "PASS":
        diag.root_causes = []
        diag.explanation = "Run passed — no diagnosis needed."
        return

    # Check: outside valid regime
    boundary = diag.valid_regime_boundary
    for dp in diag.deployment_points:
        if dp.h_test is not None and dp.h_test < boundary and dp.verdict != "PASS":
            causes.append(RootCause.OUTSIDE_REGIME)
            explanations.append(f"h_test={dp.h_test} is below valid regime boundary h≥{boundary}")
            recommendations.append(
                f"Re-run with h_test≥{boundary} (use h_test≥{boundary + 0.5} for safety)"
            )
            break

    # Check: boundary proximity (within 0.5 of boundary)
    if RootCause.OUTSIDE_REGIME not in causes:
        for dp in diag.deployment_points:
            if dp.h_test is not None and dp.verdict != "PASS":
                distance = dp.h_test - boundary
                if 0 <= distance < BOUNDARY_PROXIMITY:
                    causes.append(RootCause.BOUNDARY_EFFECT)
                    explanations.append(
                        f"h_test={dp.h_test} is only {distance:.2f} above boundary "
                        f"h≥{boundary} — MPNN interpolation degrades near boundary"
                    )
                    recommendations.append(f"Use h_test≥{boundary + 0.5} for reliable results")
                    break

    # Check: chain break (theta_smoothness)
    if diag.theta_smoothness is not None and diag.theta_smoothness > THETA_SMOOTHNESS_CHAIN_BREAK:
        causes.append(RootCause.CHAIN_BREAK)
        explanations.append(
            f"θ_smoothness={diag.theta_smoothness:.3f} > {THETA_SMOOTHNESS_CHAIN_BREAK} "
            f"— warm-start chain disrupted (restart paradox)"
        )
        recommendations.append("Reduce n_restarts to 3 or increase h-grid density")

    # Check: MPNN overfitting
    if diag.generalization_gap is not None and diag.generalization_gap > GEN_GAP_OVERFIT:
        causes.append(RootCause.MPNN_OVERFIT)
        explanations.append(
            f"gen_gap={diag.generalization_gap:.4f} > {GEN_GAP_OVERFIT} "
            f"— MPNN overfitting on training data"
        )
        recommendations.append("Increase training points or reduce epochs")

    # Check: VQE divergence
    if diag.convergence_rate is not None and diag.convergence_rate < CONVERGENCE_RATE_MIN:
        n_failed = sum(1 for c in diag.per_h_converged if not c)
        causes.append(RootCause.VQE_DIVERGENCE)
        explanations.append(
            f"convergence_rate={diag.convergence_rate:.2f} ({n_failed} h-points did not converge)"
        )
        recommendations.append("Increase maxiter or n_restarts")

    # Check: HVA expressibility limit
    if diag.error_from_circuit is not None and diag.error_from_circuit > 0.01:
        causes.append(RootCause.HVA_LIMIT)
        explanations.append(
            f"error_from_circuit={diag.error_from_circuit:.4f} "
            f"— HVA cannot express ground state at this h"
        )
        recommendations.append("Increase p_layers or move h_test further from h_c")

    # If no cause identified
    if not causes:
        causes.append(RootCause.UNKNOWN)
        explanations.append("No standard failure pattern detected — manual investigation needed")
        recommendations.append("Check per-h MSE and MPNN training curves")

    diag.root_causes = causes
    diag.explanation = "; ".join(explanations)
    diag.recommendations = recommendations


# ─── Scanning ────────────────────────────────────────────────────────────────


def scan_folder(folder_path: Path) -> list[Diagnosis]:
    """Scan a folder (and subfolders) for pipeline_run files and diagnose each."""
    diagnoses: list[Diagnosis] = []
    folder_name = folder_path.name

    # Direct pipeline files in this folder
    for pf in sorted(folder_path.glob("pipeline_run_*.json")):
        diag = parse_pipeline_run(pf, folder_name, folder_name)
        if diag:
            classify_root_causes(diag)
            diagnoses.append(diag)

    # Nested subfolders
    for subdir in sorted(folder_path.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith(".") or subdir.name == "checkpoints":
            continue
        # Take the latest pipeline_run in each subfolder
        pipeline_files = sorted(subdir.glob("pipeline_run_*.json"), reverse=True)
        if not pipeline_files:
            continue
        diag = parse_pipeline_run(pipeline_files[0], folder_name, subdir.name)
        if diag:
            classify_root_causes(diag)
            diagnoses.append(diag)

    return diagnoses


def scan_all_thesis() -> list[Diagnosis]:
    """Scan all thesis result folders."""
    diagnoses: list[Diagnosis] = []

    if not THESIS.exists():
        return diagnoses

    for entry in sorted(THESIS.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        diagnoses.extend(scan_folder(entry))

    return diagnoses


# ─── Reporting ───────────────────────────────────────────────────────────────


def report_diagnoses(
    diagnoses: list[Diagnosis],
    *,
    severity_filter: str | None = None,
    show_passing: bool = False,
) -> None:
    """Print a formatted diagnostic report."""
    # Filter
    filtered = diagnoses
    if not show_passing:
        filtered = [d for d in filtered if d.severity not in ("PASS",)]
    if severity_filter:
        severity_filter_upper = severity_filter.upper()
        filtered = [d for d in filtered if d.severity == severity_filter_upper]

    if not filtered:
        print("\n  ✅ No failures to diagnose!")
        return

    # Group by severity
    by_severity: dict[str, list[Diagnosis]] = {}
    for d in filtered:
        by_severity.setdefault(d.severity, []).append(d)

    # Summary
    print(f"\n  Total diagnosed: {len(filtered)}")
    for sev in ["FAIL", "MARGINAL", "NO_PHASE4", "NO_DATA"]:
        if sev in by_severity:
            print(f"    {sev}: {len(by_severity[sev])}")

    # Root cause distribution
    cause_counts: dict[str, int] = {}
    for d in filtered:
        for c in d.root_causes:
            cause_counts[c.value] = cause_counts.get(c.value, 0) + 1

    if cause_counts:
        print("\n  Root cause distribution:")
        for cause, count in sorted(cause_counts.items(), key=lambda x: -x[1]):
            print(f"    {cause}: {count}")

    # Detail per failure
    print("\n" + "─" * 78)
    for d in sorted(filtered, key=lambda x: (x.severity, x.topology, x.n_qubits or 0)):
        _print_single_diagnosis(d)


def _print_single_diagnosis(d: Diagnosis) -> None:
    """Print a single diagnosis in detail."""
    causes_str = ", ".join(c.value for c in d.root_causes)
    seed_str = str(d.seed) if d.seed is not None else "—"

    print(
        f"\n  [{d.severity}] {d.topology} N={d.n_qubits} p={d.p_layers} "
        f"seed={seed_str} → {d.variant}"
    )
    print(f"    Causes: {causes_str}")

    # Deployment points
    if d.deployment_points:
        for dp in d.deployment_points:
            de_str = f"{dp.de_gap:.4f}" if dp.de_gap is not None else "N/A"
            print(f"    h_test={dp.h_test}: ΔE/gap={de_str} [{dp.verdict}]")
    else:
        print("    No Phase 4 results (pipeline aborted before deployment)")

    # Key diagnostics
    if d.theta_smoothness is not None:
        flag = " ⚠️" if d.theta_smoothness > THETA_SMOOTHNESS_CHAIN_BREAK else ""
        print(f"    θ_smoothness: {d.theta_smoothness:.4f}{flag}")
    if d.generalization_gap is not None:
        flag = " ⚠️" if d.generalization_gap > GEN_GAP_OVERFIT else ""
        print(f"    gen_gap: {d.generalization_gap:.2e}{flag}")
    if d.convergence_rate is not None and d.convergence_rate < 1.0:
        print(f"    conv_rate: {d.convergence_rate:.2f} ⚠️")
    if d.error_from_circuit is not None and d.error_from_circuit > 0:
        print(f"    error_circuit: {d.error_from_circuit:.4f}")
    if d.error_from_mpnn is not None:
        print(f"    error_mpnn: {d.error_from_mpnn:.4f}")

    # Explanation and recommendations
    print(f"    Explanation: {d.explanation}")
    if d.recommendations:
        for rec in d.recommendations:
            print(f"    → {rec}")

    print("  " + "─" * 76)


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the diagnose CLI."""
    parser = argparse.ArgumentParser(
        description="Diagnose pipeline failures with automated root cause analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Diagnose all failures across thesis results
  python analysis/diagnose.py --all

  # Diagnose a specific folder
  python analysis/diagnose.py results/thesis/p1_variants_N10_r2

  # Only FAIL severity (skip MARGINAL)
  python analysis/diagnose.py --all --severity fail

  # Filter by topology
  python analysis/diagnose.py --all --topology triangular

  # Export to JSON
  python analysis/diagnose.py --all --json analysis/raw_data/diagnoses.json

  # Include passing runs in output
  python analysis/diagnose.py --all --show-passing
""",
    )

    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="Path to a results folder or specific variant subfolder",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan all thesis result folders",
    )
    parser.add_argument(
        "--severity",
        type=str,
        default=None,
        choices=["fail", "marginal", "no_phase4"],
        help="Filter to specific severity level",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default=None,
        help="Filter to specific topology",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=None,
        choices=[1, 2],
        help="Filter to p=1 or p=2",
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=None,
        help="Filter to specific system size",
    )
    parser.add_argument(
        "--show-passing",
        action="store_true",
        help="Include PASS results in output",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Export diagnoses to JSON file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show debug information",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(message)s", stream=sys.stderr)

    # Validate arguments
    if not args.all and args.path is None:
        parser.error("Provide a path or use --all to scan all thesis results")

    # Scan
    print("=" * 78)
    print("  GNN-HVA FAILURE DIAGNOSIS")
    print("=" * 78)

    if args.all:
        print("\n  Scanning all thesis results...", file=sys.stderr)
        diagnoses = scan_all_thesis()
    else:
        target = args.path
        if not target.exists():
            print(f"  ❌ Path not found: {target}", file=sys.stderr)
            sys.exit(1)
        print(f"\n  Scanning: {target}", file=sys.stderr)
        diagnoses = scan_folder(target)

    print(f"  Found {len(diagnoses)} pipeline runs", file=sys.stderr)

    # Apply filters
    if args.topology:
        diagnoses = [d for d in diagnoses if d.topology == args.topology]
    if args.p is not None:
        diagnoses = [d for d in diagnoses if d.p_layers == args.p]
    if args.n_qubits is not None:
        diagnoses = [d for d in diagnoses if d.n_qubits == args.n_qubits]

    if args.topology or args.p or args.n_qubits:
        print(f"  After filters: {len(diagnoses)} runs", file=sys.stderr)

    # Report
    report_diagnoses(
        diagnoses,
        severity_filter=args.severity,
        show_passing=args.show_passing,
    )

    # Export
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        export = []
        for d in diagnoses:
            entry = asdict(d)
            # Convert enums to strings for JSON
            entry["root_causes"] = [c.value for c in d.root_causes]
            entry["deployment_points"] = [
                {"h_test": dp.h_test, "de_gap": dp.de_gap, "verdict": dp.verdict}
                for dp in d.deployment_points
            ]
            export.append(entry)

        with open(args.json, "w") as f:
            json.dump(export, f, indent=2, default=str)
        print(f"\n  📄 Exported {len(export)} diagnoses to {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
