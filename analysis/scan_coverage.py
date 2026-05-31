#!/usr/bin/env python3
"""Comprehensive coverage scanner for all result types in the GNN-HVA project.

Scans ALL result locations (thesis variants, experiments, baselines, noisy, ZNE)
and produces a gap analysis identifying what simulations are needed next.

Covers:
- Noiseless pipeline results (pipeline_run_*.json)
- Noisy/ZNE results (noisy_*.json)
- V8 experiment results (run_*.json in exp_*/)
- Baseline results (results/experiments/baselines/)
- Legacy variants (results/thesis/variants/)

Usage:
    # Basic scan
    python analysis/scan_coverage.py

    # Auto-discover all result folders (finds new folders automatically)
    python analysis/scan_coverage.py --discover

    # Filter by topology, system size, or p-layers
    python analysis/scan_coverage.py --topology chain_1d --n-qubits 10 --p 1

    # Custom PASS/FAIL thresholds
    python analysis/scan_coverage.py --pass-threshold 0.03 --marginal-threshold 0.08

    # Extended analytics (reproducibility, staleness, h-coverage, data quality)
    python analysis/scan_coverage.py --extended

    # Export to multiple formats
    python analysis/scan_coverage.py --json out.json --csv out.csv --markdown out.md

    # Quiet mode (summary only, no section reports)
    python analysis/scan_coverage.py --quiet

    # Combine: discover + filter + extended + export
    python analysis/scan_coverage.py --discover --p 1 --extended --json coverage.json

CLI Options:
    --discover          Auto-discover folders under results/thesis/
    --topology T        Filter: chain_1d | ladder | triangular | kagome
    --n-qubits N        Filter: system size (6, 10, 16, 20, 24)
    --p P               Filter: p=1 or p=2
    --pass-threshold F  ΔE/gap < F → PASS (default: 0.05)
    --marginal-threshold F  ΔE/gap < F → MARGINAL (default: 0.10)
    --extended          Include reproducibility, staleness, h-coverage, quality reports
    --json PATH         Export structured JSON
    --csv PATH          Export flat CSV (for spreadsheets)
    --markdown PATH     Export summary as Markdown table
    --verbose / -v      Show per-file parse warnings
    --quiet / -q        Only show final summary
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
THESIS = RESULTS / "thesis"
EXPERIMENTS = RESULTS / "experiments"

# Known thesis variant folders with their default topology (used when --discover is off)
VARIANT_FOLDERS: list[tuple[str, str]] = [
    ("variants_N6_N10_1D_linnear", "chain_1d"),
    ("variants_N6_ladder", "ladder"),
    ("variants_N6_triangular", "triangular"),
    ("variants_N10_ladder", "ladder"),
    ("variants_N10_triangular", "triangular"),
    ("variants", "chain_1d"),  # Legacy variants folder
    ("n6_noiseless", "chain_1d"),  # Single baseline run
    ("p1_variants_N10", "chain_1d"),  # p=1 multi-topology variants
    ("p1_variants_N10_r2", "chain_1d"),
    ("p1_variants_N16_r2", "chain_1d"),
    ("p1_variants_N24_r2", "chain_1d"),
    ("variants_N10_multi", "chain_1d"),
    ("variants_N16_multi", "chain_1d"),
]

# Topology inference heuristics for auto-discovery mode
_TOPO_HINTS: dict[str, str] = {
    "ladder": "ladder",
    "triangular": "triangular",
    "tri": "triangular",
    "heavy_hex": "heavy_hex",
    "heavy-hex": "heavy_hex",
    "hex": "heavy_hex",
    "1d": "chain_1d",
    "1D": "chain_1d",
    "linnear": "chain_1d",
    "linear": "chain_1d",
    "chain": "chain_1d",
}


def _infer_default_topology(folder_name: str) -> str:
    """Infer default topology from folder name heuristics."""
    for hint, topo in _TOPO_HINTS.items():
        if hint in folder_name:
            return topo
    return "chain_1d"


def discover_variant_folders() -> list[tuple[str, str]]:
    """Auto-discover all subdirectories under results/thesis/ as variant folders.

    Returns (folder_name, inferred_default_topology) pairs.
    Skips hidden directories and files.
    """
    if not THESIS.exists():
        return []
    folders: list[tuple[str, str]] = []
    for entry in sorted(THESIS.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        topo = _infer_default_topology(entry.name)
        folders.append((entry.name, topo))
    return folders


# Valid regime boundaries for p=1 (h_test must be >= this value)
P1_VALID_REGIME: dict[tuple[str, int], float] = {
    ("chain_1d", 6): 1.6,
    ("chain_1d", 10): 1.9,
    ("chain_1d", 20): 2.25,
    ("heavy_hex", 6): 2.0,
    ("heavy_hex", 10): 2.5,
    ("ladder", 6): 2.0,
    ("ladder", 10): 3.0,  # Corrected: was 2.0, verification shows failures at 2.75
    ("triangular", 6): 4.0,  # Corrected: was 3.0, failure at h=4.0 but pass at h=4.5
    ("triangular", 10): 3.5,
}

# Valid regime boundaries for p=2
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

# Default PASS/FAIL thresholds (can be overridden via CLI)
DEFAULT_PASS_THRESHOLD: float = 0.05
DEFAULT_MARGINAL_THRESHOLD: float = 0.10


def _update_thresholds(pass_thr: float, marginal_thr: float) -> None:
    """Update module-level threshold defaults (called from main when CLI overrides)."""
    global DEFAULT_PASS_THRESHOLD, DEFAULT_MARGINAL_THRESHOLD  # noqa: PLW0603
    DEFAULT_PASS_THRESHOLD = pass_thr
    DEFAULT_MARGINAL_THRESHOLD = marginal_thr


@dataclass
class PipelineRecord:
    """A single noiseless pipeline result (one per h_test point)."""

    folder: str
    variant: str
    topology: str
    n_qubits: int | None
    p_layers: int | None
    de_gap: float | None
    h_test: float | None
    h_values: list[float] = field(default_factory=list)
    seed: int | None = None
    n_restarts: int | None = None
    hidden_dim: int | None = None
    theta_smoothness: float | None = None
    generalization_gap: float | None = None
    file: str = ""
    # When a run has multiple h_test points, this is the worst ΔE/gap across all
    worst_de_gap: float | None = None
    n_test_points: int = 1

    def verdict_with_thresholds(self, pass_thr: float, marginal_thr: float) -> str:
        """Compute verdict using custom thresholds."""
        if self.de_gap is None:
            return "NO_DATA"
        if self.de_gap < pass_thr:
            return "PASS"
        if self.de_gap < marginal_thr:
            return "MARGINAL"
        return "FAIL"

    @property
    def verdict(self) -> str:
        return self.verdict_with_thresholds(DEFAULT_PASS_THRESHOLD, DEFAULT_MARGINAL_THRESHOLD)

    @property
    def in_valid_regime(self) -> bool:
        """Check if h_test is within the valid regime for this p/topology/N."""
        if self.h_test is None or self.n_qubits is None or self.p_layers is None:
            return False
        regime = P1_VALID_REGIME if self.p_layers == 1 else P2_VALID_REGIME
        threshold = regime.get((self.topology, self.n_qubits), 0.0)
        return self.h_test >= threshold


@dataclass
class NoisyRecord:
    """A single noisy/ZNE result."""

    folder: str
    variant: str
    topology: str
    n_qubits: int | None
    p_layers: int | None
    seed: int | None = None
    n_layouts: int | None = None
    h_values: list[float] = field(default_factory=list)
    mean_gain_pct: float | None = None
    mean_r2: float | None = None
    n_mitigated_wins: int | None = None
    n_total: int | None = None
    file: str = ""

    @property
    def zne_works(self) -> bool:
        return (self.mean_gain_pct or 0) > 0 and (self.n_mitigated_wins or 0) > 0


@dataclass
class ExperimentRecord:
    """A V8 experiment result."""

    experiment_id: str
    p_layers: int | None
    n_qubits: int | None
    topology: str | None
    description: str = ""
    hypothesis: str = ""
    pass_rate: float | None = None
    mean_de_gap: float | None = None
    seeds: list[int] = field(default_factory=list)
    file: str = ""


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    """Safely load a JSON file, returning None on any error."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def _compute_median(values: list[float]) -> float | None:
    """Compute the correct median of a sorted list of floats."""
    if not values:
        return None
    n = len(values)
    if n % 2 == 1:
        return values[n // 2]
    return (values[n // 2 - 1] + values[n // 2]) / 2.0


def _extract_p_layers(data: dict, filepath: Path | None = None) -> int | None:
    """Extract p_layers from various JSON structures.

    Returns None if p_layers cannot be determined (caller should handle).
    Logs a warning when defaulting is needed.
    """
    config = data.get("config", {})
    system = data.get("system", {})
    # Check config first, then system
    p = config.get("p_layers") or system.get("p_layers")
    if p is not None:
        return int(p)
    # Cannot determine — log warning and return None
    loc = f" in {filepath}" if filepath else ""
    logger.warning(f"p_layers not found{loc}; record will be marked p_layers=None")
    return None


def _extract_topology(data: dict, default: str) -> str:
    """Extract topology from various JSON structures."""
    config = data.get("config", {})
    system = data.get("system", {})
    return config.get("topology") or system.get("topology") or default


def _extract_n_qubits(data: dict) -> int | None:
    """Extract n_qubits from various JSON structures."""
    config = data.get("config", {})
    system = data.get("system", {})
    n = config.get("n_qubits") or system.get("n_qubits")
    return int(n) if n is not None else None


def scan_pipeline_results(*, discover: bool = False) -> list[PipelineRecord]:
    """Scan ALL pipeline_run_*.json files across all thesis folders.

    Args:
        discover: If True, auto-discover all subdirectories under results/thesis/
                  instead of using the hardcoded VARIANT_FOLDERS list.
    """
    records: list[PipelineRecord] = []

    folders = discover_variant_folders() if discover else VARIANT_FOLDERS

    for folder_name, default_topo in folders:
        folder_path = THESIS / folder_name
        if not folder_path.exists():
            continue

        # Handle flat folder — parse ALL pipeline_run files (Fix #4)
        pipeline_files = sorted(folder_path.glob("pipeline_run_*.json"))
        for pf in pipeline_files:
            _parse_pipeline_file(pf, folder_name, folder_name, default_topo, records)

        # Handle nested folders (variants_N10_ladder/nl_restarts_5/pipeline_run_*.json)
        for subdir in sorted(folder_path.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue
            if subdir.name == "checkpoints":
                continue

            # Parse ALL pipeline_run files in each subdir (Fix #4)
            pipeline_files = sorted(subdir.glob("pipeline_run_*.json"))
            for pf in pipeline_files:
                _parse_pipeline_file(pf, folder_name, subdir.name, default_topo, records)

    return records


def _parse_pipeline_file(
    path: Path,
    folder_name: str,
    variant_name: str,
    default_topo: str,
    records: list[PipelineRecord],
) -> None:
    """Parse a single pipeline_run JSON file into PipelineRecord(s).

    Fix #1: Creates one record per h_test point in phase4_results.
    If phase4_results has multiple entries, each gets its own record,
    and all records carry the worst-case ΔE/gap for the run.
    """
    data = _safe_load_json(path)
    if data is None:
        return

    config = data.get("config", {})
    p4 = data.get("phase4_results", [])
    diag = data.get("diagnostics", {})
    mpnn_config = config.get("mpnn", {})

    # Shared fields
    topology = _extract_topology(data, default_topo)
    n_qubits = _extract_n_qubits(data)
    p_layers = _extract_p_layers(data, filepath=path)
    seed = config.get("seed")
    n_restarts = config.get("n_restarts")
    hidden_dim = mpnn_config.get("hidden_dim")
    theta_smoothness = diag.get("phase2", {}).get("theta_smoothness")
    generalization_gap = diag.get("phase3", {}).get("generalization_gap")
    h_values = config.get("h_values", [])
    file_rel = str(path.relative_to(ROOT))

    if not p4:
        # No phase4 results — emit a single record with no data
        records.append(
            PipelineRecord(
                folder=folder_name,
                variant=variant_name,
                topology=topology,
                n_qubits=n_qubits,
                p_layers=p_layers,
                de_gap=None,
                h_test=None,
                h_values=h_values,
                seed=seed,
                n_restarts=n_restarts,
                hidden_dim=hidden_dim,
                theta_smoothness=theta_smoothness,
                generalization_gap=generalization_gap,
                file=file_rel,
                worst_de_gap=None,
                n_test_points=0,
            )
        )
        return

    # Compute worst-case ΔE/gap across all test points
    all_de_gaps = [
        entry.get("delta_e_over_gap") for entry in p4 if entry.get("delta_e_over_gap") is not None
    ]
    worst_de_gap = max(all_de_gaps) if all_de_gaps else None
    n_test_points = len(p4)

    # Emit one record per h_test point (Fix #1)
    for entry in p4:
        de_gap = entry.get("delta_e_over_gap")
        h_test = entry.get("h_test")

        records.append(
            PipelineRecord(
                folder=folder_name,
                variant=variant_name,
                topology=topology,
                n_qubits=n_qubits,
                p_layers=p_layers,
                de_gap=de_gap,
                h_test=h_test,
                h_values=h_values,
                seed=seed,
                n_restarts=n_restarts,
                hidden_dim=hidden_dim,
                theta_smoothness=theta_smoothness,
                generalization_gap=generalization_gap,
                file=file_rel,
                worst_de_gap=worst_de_gap,
                n_test_points=n_test_points,
            )
        )


def scan_noisy_results(*, discover: bool = False) -> list[NoisyRecord]:
    """Scan ALL noisy_*.json files across all thesis folders.

    Args:
        discover: If True, auto-discover all subdirectories under results/thesis/.
    """
    records: list[NoisyRecord] = []

    if discover:
        # Auto-discover: scan all thesis subdirectories for noisy files
        search_folders = discover_variant_folders()
    else:
        # All folders that may contain noisy results
        search_folders = [
            ("variants_N6_N10_1D_linnear", "chain_1d"),
            ("variants_N6_ladder", "ladder"),
            ("variants_N6_triangular", "triangular"),
            ("variants_N10_ladder", "ladder"),
            ("variants_N10_triangular", "triangular"),
            ("variants", "chain_1d"),
            ("n6_noisy", "chain_1d"),
            ("analysis_p1_zne", "unknown"),
            ("p1_variants_N10", "chain_1d"),
            ("p1_variants_N10_r2", "chain_1d"),
            ("p1_variants_N16_r2", "chain_1d"),
            ("p1_variants_N24_r2", "chain_1d"),
        ]

    for folder_name, default_topo in search_folders:
        folder_path = THESIS / folder_name
        if not folder_path.exists():
            continue

        # Flat noisy files
        for nf in sorted(folder_path.glob("noisy_*.json")):
            _parse_noisy_file(nf, folder_name, folder_name, default_topo, records)

        # Nested noisy files
        for subdir in sorted(folder_path.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue
            for nf in sorted(subdir.glob("noisy_*.json")):
                _parse_noisy_file(nf, folder_name, subdir.name, default_topo, records)

    return records


def _parse_noisy_file(
    path: Path,
    folder_name: str,
    variant_name: str,
    default_topo: str,
    records: list[NoisyRecord],
) -> None:
    """Parse a single noisy JSON file into a NoisyRecord."""
    data = _safe_load_json(path)
    if data is None:
        return

    config = data.get("config", {})
    summary = data.get("summary", {})

    records.append(
        NoisyRecord(
            folder=folder_name,
            variant=variant_name,
            topology=_extract_topology(data, default_topo),
            n_qubits=_extract_n_qubits(data),
            p_layers=_extract_p_layers(data, filepath=path),
            seed=config.get("seed"),
            n_layouts=config.get("n_layouts"),
            h_values=config.get("h_values", []),
            mean_gain_pct=summary.get("mean_gain_pct"),
            mean_r2=summary.get("mean_r2"),
            n_mitigated_wins=summary.get("n_mitigated_wins"),
            n_total=summary.get("n_total"),
            file=str(path.relative_to(ROOT)),
        )
    )


def scan_experiment_results() -> list[ExperimentRecord]:
    """Scan ALL V8 experiment results (run_*.json in exp_*/ folders)."""
    records: list[ExperimentRecord] = []

    if not EXPERIMENTS.exists():
        return records

    for exp_dir in sorted(EXPERIMENTS.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("exp_"):
            continue

        run_files = sorted(exp_dir.glob("run_*.json"), reverse=True)
        if not run_files:
            continue

        data = _safe_load_json(run_files[0])
        if data is None:
            continue

        config = data.get("config", {})
        system = config.get("system", {})
        analysis = data.get("analysis", {})
        summary = analysis.get("summary", {})

        p_layers = system.get("p_layers") or config.get("p_layers")
        if p_layers is not None:
            p_layers = int(p_layers)

        records.append(
            ExperimentRecord(
                experiment_id=exp_dir.name,
                p_layers=p_layers,
                n_qubits=system.get("n_qubits"),
                topology=system.get("topology"),
                description=config.get("description", ""),
                hypothesis=config.get("hypothesis", ""),
                pass_rate=summary.get("pass_rate"),
                mean_de_gap=summary.get("mean_de_gap"),
                seeds=config.get("seeds", []),
                file=str(run_files[0].relative_to(ROOT)),
            )
        )

    return records


# ─── Reporting ───────────────────────────────────────────────────────────────


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print("=" * 80)


def report_pipeline_coverage(p2: list[PipelineRecord], p1: list[PipelineRecord]) -> None:
    """Report noiseless pipeline coverage for p=1 and p=2."""
    print_section("NOISELESS PIPELINE COVERAGE")

    # p=2 summary
    print(f"\n  ── p=2 ({len(p2)} test points) ──")
    p2_by = defaultdict(list)
    for r in p2:
        p2_by[(r.topology, r.n_qubits)].append(r)

    print(f"\n  {'Topology':<15} {'N':<5} {'Pts':<6} {'Pass':<6} {'Median':<10} {'Seeds'}")
    print("  " + "-" * 65)
    for key in sorted(p2_by.keys()):
        topo, n = key
        group = p2_by[key]
        n_pass = sum(1 for r in group if r.verdict == "PASS")
        de_vals = sorted(r.de_gap for r in group if r.de_gap is not None)
        median = _compute_median(de_vals)
        seeds = sorted(set(r.seed for r in group if r.seed is not None))
        med_str = f"{median:.4f}" if median is not None else "N/A"
        print(f"  {topo:<15} {n:<5} {len(group):<6} {n_pass:<6} {med_str:<10} {seeds}")

    # p=1 summary
    print(f"\n  ── p=1 ({len(p1)} test points) ──")
    if not p1:
        print("  ⚠️  NO p=1 noiseless pipeline results found!")
        return

    p1_by = defaultdict(list)
    for r in p1:
        p1_by[(r.topology, r.n_qubits)].append(r)

    print(
        f"\n  {'Topology':<15} {'N':<5} {'Pts':<6} {'Pass':<6} {'h_test':<8} "
        f"{'Valid?':<8} {'Seeds'}"
    )
    print("  " + "-" * 70)
    for key in sorted(p1_by.keys()):
        topo, n = key
        group = p1_by[key]
        n_pass = sum(1 for r in group if r.verdict == "PASS")
        seeds = sorted(set(r.seed for r in group if r.seed is not None))
        h_tests = sorted(set(r.h_test for r in group if r.h_test is not None))
        valid = all(r.in_valid_regime for r in group)
        valid_str = "✅" if valid else "❌"
        h_str = str(h_tests) if h_tests else "N/A"
        print(f"  {topo:<15} {n:<5} {len(group):<6} {n_pass:<6} {h_str:<8} {valid_str:<8} {seeds}")

    # Detail
    print("\n  Detail (all p=1 noiseless):")
    for r in sorted(p1, key=lambda x: (x.topology, x.n_qubits or 0, x.h_test or 0)):
        de = f"{r.de_gap:.4f}" if r.de_gap is not None else "N/A"
        seed_str = str(r.seed) if r.seed is not None else "—"
        valid_str = "✓" if r.in_valid_regime else "✗"
        multi_str = (
            f" (worst={r.worst_de_gap:.4f})"
            if (r.n_test_points > 1 and r.worst_de_gap is not None)
            else ""
        )
        print(
            f"    {r.topology:<12} N={r.n_qubits or '?':<3} seed={seed_str:<5} "
            f"h_test={r.h_test} [{valid_str}] ΔE/gap={de}{multi_str} [{r.verdict}] "
            f"→ {r.variant}"
        )


def report_noisy_coverage(records: list[NoisyRecord]) -> None:
    """Report noisy/ZNE coverage."""
    print_section("NOISY / ZNE COVERAGE")

    p1 = [r for r in records if r.p_layers == 1]
    p2 = [r for r in records if r.p_layers == 2]

    print(f"\n  Total: {len(records)} (p=1: {len(p1)}, p=2: {len(p2)})")

    # p=1 detail
    if p1:
        print(f"\n  ── p=1 noisy ({len(p1)} runs) ──")
        print(
            f"  {'Topology':<15} {'N':<5} {'Seed':<6} {'Gain%':<10} {'R²':<8} "
            f"{'Works?':<8} {'Variant'}"
        )
        print("  " + "-" * 75)
        for r in sorted(p1, key=lambda x: (x.topology, x.n_qubits or 0, x.seed or 0)):
            gain = f"{r.mean_gain_pct:+.1f}%" if r.mean_gain_pct is not None else "N/A"
            r2 = f"{r.mean_r2:.3f}" if r.mean_r2 is not None else "N/A"
            works = "✅" if r.zne_works else "❌"
            print(
                f"  {r.topology:<15} {r.n_qubits or '?':<5} {r.seed or '?':<6} "
                f"{gain:<10} {r2:<8} {works:<8} {r.variant}"
            )

    # p=2 summary by config
    if p2:
        print(f"\n  ── p=2 noisy ({len(p2)} runs) ──")
        p2_by = defaultdict(list)
        for r in p2:
            p2_by[(r.topology, r.n_qubits)].append(r)

        print(f"  {'Topology':<15} {'N':<5} {'Runs':<6} {'Mean Gain':<12} {'ZNE Works?'}")
        print("  " + "-" * 55)
        for key in sorted(p2_by.keys()):
            topo, n = key
            group = p2_by[key]
            gains = [r.mean_gain_pct for r in group if r.mean_gain_pct is not None]
            mean_g = sum(gains) / len(gains) if gains else 0
            n_works = sum(1 for r in group if r.zne_works)
            works_str = f"{n_works}/{len(group)}"
            print(f"  {topo:<15} {n:<5} {len(group):<6} {mean_g:+.1f}%{'':5} {works_str}")


def report_experiments(records: list[ExperimentRecord]) -> None:
    """Report V8 experiment coverage."""
    print_section("V8 EXPERIMENTS")

    p1 = [r for r in records if r.p_layers == 1]
    p2 = [r for r in records if r.p_layers == 2]
    p_unknown = [r for r in records if r.p_layers is None]

    print(f"\n  Total: {len(records)} (p=1: {len(p1)}, p=2: {len(p2)}", end="")
    if p_unknown:
        print(f", p=?: {len(p_unknown)}", end="")
    print(")")

    if p1:
        print("\n  p=1 experiments:")
        for r in p1:
            desc = r.description[:55] if r.description else "—"
            print(f"    {r.experiment_id}: N={r.n_qubits} {r.topology} — {desc}")

    if p_unknown:
        print("\n  ⚠️  Experiments with unknown p_layers:")
        for r in p_unknown:
            print(f"    {r.experiment_id}: {r.file}")

    # Show all experiments with pass_rate
    print(f"\n  {'Experiment':<14} {'p':<4} {'N':<5} {'Topology':<12} {'Pass%':<8} {'Description'}")
    print("  " + "-" * 80)
    for r in sorted(records, key=lambda x: x.experiment_id):
        pr = f"{r.pass_rate:.0%}" if r.pass_rate is not None else "—"
        desc = r.description[:40] if r.description else "—"
        p_str = str(r.p_layers) if r.p_layers is not None else "?"
        print(
            f"  {r.experiment_id:<14} p={p_str:<3} {r.n_qubits or '?':<5} "
            f"{r.topology or '—':<12} {pr:<8} {desc}"
        )


def report_gap_analysis(
    p2_pipeline: list[PipelineRecord],
    p1_pipeline: list[PipelineRecord],
    noisy: list[NoisyRecord],
) -> list[dict]:
    """Identify gaps and produce recommendations."""
    print_section("GAP ANALYSIS — What Simulations Are Needed")

    # Build config sets
    p2_configs = set(
        (r.topology, r.n_qubits)
        for r in p2_pipeline
        if r.de_gap is not None and r.n_qubits is not None
    )
    p1_nl_configs = set(
        (r.topology, r.n_qubits)
        for r in p1_pipeline
        if r.de_gap is not None and r.n_qubits is not None
    )
    p1_noisy_configs = set(
        (r.topology, r.n_qubits) for r in noisy if r.p_layers == 1 and r.n_qubits is not None
    )

    print(f"\n  p=2 noiseless configs: {sorted(p2_configs)}")
    print(f"  p=1 noiseless configs: {sorted(p1_nl_configs)}")
    print(f"  p=1 noisy configs:     {sorted(p1_noisy_configs)}")

    recommendations: list[dict] = []

    # GAP 1: p=2 exists but p=1 noiseless missing
    gap1 = p2_configs - p1_nl_configs
    # Exclude kagome (only 1 run, not central to thesis)
    gap1 = {c for c in gap1 if c[0] != "kagome"}
    print("\n  ─── GAP 1: p=1 noiseless pipeline missing ───")
    if gap1:
        for config in sorted(gap1):
            topo, n = config
            p2_count = sum(
                1
                for r in p2_pipeline
                if r.topology == topo and r.n_qubits == n and r.de_gap is not None
            )
            print(f"    → {topo} N={n} (p=2 has {p2_count} results)")
            recommendations.append(
                {
                    "priority": 1,
                    "type": "p=1 noiseless pipeline",
                    "topology": topo,
                    "n_qubits": n,
                    "reason": "p=2 has data but p=1 missing",
                    "action": f"Run pipeline with --p 1 --n {n} --topology {topo} --seed 42,43,44",
                }
            )
    else:
        print("    ✅ All p=2 configs have p=1 noiseless data")

    # GAP 2: p=1 noiseless exists but h_test is outside valid regime
    print("\n  ─── GAP 2: p=1 runs with h_test OUTSIDE valid regime ───")
    invalid_runs = [r for r in p1_pipeline if not r.in_valid_regime and r.de_gap is not None]
    if invalid_runs:
        for r in invalid_runs:
            threshold = P1_VALID_REGIME.get((r.topology, r.n_qubits or 0), "?")
            print(
                f"    → {r.topology} N={r.n_qubits} h_test={r.h_test} "
                f"(need h≥{threshold}) [{r.verdict}] — {r.variant}"
            )
            # Only recommend re-run if it failed AND was outside valid regime
            if r.verdict != "PASS":
                recommendations.append(
                    {
                        "priority": 2,
                        "type": "p=1 re-run with correct h_test",
                        "topology": r.topology,
                        "n_qubits": r.n_qubits,
                        "reason": f"h_test={r.h_test} outside valid regime (need h≥{threshold})",
                        "action": f"Re-run with h_test≥{threshold}",
                    }
                )
    else:
        print("    ✅ All p=1 runs use h_test within valid regime")

    # GAP 3: Seed coverage for p=1
    print("\n  ─── GAP 3: p=1 seed coverage (need ≥3 for reproducibility) ───")
    for config in sorted(p1_nl_configs):
        topo, n = config
        seeds = set(
            r.seed
            for r in p1_pipeline
            if r.topology == topo and r.n_qubits == n and r.seed is not None
        )
        if len(seeds) < 3:
            missing = sorted(set([42, 43, 44]) - seeds)
            print(f"    → {topo} N={n}: {len(seeds)} seeds {sorted(seeds)} (missing: {missing})")
            recommendations.append(
                {
                    "priority": 3,
                    "type": "p=1 additional seeds",
                    "topology": topo,
                    "n_qubits": n,
                    "reason": f"Only {len(seeds)} seeds, need 3",
                    "missing_seeds": missing,
                    "action": f"Run with seeds {missing}",
                }
            )
        else:
            print(f"    ✅ {topo} N={n}: {len(seeds)} seeds")

    # GAP 4: p=1 noisy/ZNE missing
    print("\n  ─── GAP 4: p=1 noisy/ZNE coverage ───")
    gap4 = p1_nl_configs - p1_noisy_configs
    if gap4:
        for config in sorted(gap4):
            topo, n = config
            print(f"    → {topo} N={n}: has noiseless but no ZNE validation")
            # Only recommend if N=10 (ZNE is the hardware strategy for N≥10)
            if n >= 10:
                recommendations.append(
                    {
                        "priority": 4,
                        "type": "p=1 noisy/ZNE",
                        "topology": topo,
                        "n_qubits": n,
                        "reason": "p=1 noiseless exists but ZNE not validated",
                        "action": f"Run noisy simulation with p=1 N={n} {topo}",
                    }
                )
    else:
        print("    ✅ All p=1 noiseless configs have ZNE data")

    # Print recommendations
    print_section("RECOMMENDATIONS (sorted by priority)")

    if not recommendations:
        print("\n  ✅ No critical gaps! All configurations are covered.")
        print("     Next step: hardware deployment on IBM Torino.")
    else:
        for i, rec in enumerate(sorted(recommendations, key=lambda x: x["priority"]), 1):
            p_label = {1: "HIGH", 2: "HIGH", 3: "MEDIUM", 4: "LOW"}
            priority = p_label.get(rec["priority"], "LOW")
            print(f"\n  {i}. [{priority}] {rec['type']}: {rec['topology']} N={rec['n_qubits']}")
            print(f"     Reason: {rec['reason']}")
            print(f"     Action: {rec['action']}")
            if "missing_seeds" in rec:
                print(f"     Missing seeds: {rec['missing_seeds']}")

    return recommendations


def report_summary(
    pipeline: list[PipelineRecord],
    noisy: list[NoisyRecord],
    experiments: list[ExperimentRecord],
    recommendations: list[dict],
) -> None:
    """Print final summary."""
    print_section("SUMMARY")

    p1_nl = [r for r in pipeline if r.p_layers == 1]
    p2_nl = [r for r in pipeline if r.p_layers == 2]
    p_unknown_nl = [r for r in pipeline if r.p_layers is None]
    p1_noisy = [r for r in noisy if r.p_layers == 1]
    p2_noisy = [r for r in noisy if r.p_layers == 2]

    total = len(pipeline) + len(noisy) + len(experiments)
    print(f"\n  Total data points scanned: {total}")
    print(
        f"  ├── Noiseless pipeline: {len(pipeline)} (p=1: {len(p1_nl)}, p=2: {len(p2_nl)}", end=""
    )
    if p_unknown_nl:
        print(f", p=?: {len(p_unknown_nl)}", end="")
    print(")")
    print(f"  ├── Noisy/ZNE:          {len(noisy)} (p=1: {len(p1_noisy)}, p=2: {len(p2_noisy)})")
    print(f"  └── Experiments:        {len(experiments)}")
    print(f"\n  Gaps identified: {len(recommendations)}")
    high = sum(1 for r in recommendations if r["priority"] <= 2)
    med = sum(1 for r in recommendations if r["priority"] == 3)
    low = sum(1 for r in recommendations if r["priority"] >= 4)
    print(f"  ├── HIGH priority: {high}")
    print(f"  ├── MEDIUM priority: {med}")
    print(f"  └── LOW priority: {low}")


# ─── Filtering ───────────────────────────────────────────────────────────────


@dataclass
class FilterConfig:
    """CLI filter configuration."""

    topology: str | None = None
    n_qubits: int | None = None
    p_layers: int | None = None
    pass_threshold: float = DEFAULT_PASS_THRESHOLD
    marginal_threshold: float = DEFAULT_MARGINAL_THRESHOLD


def _apply_pipeline_filters(
    records: list[PipelineRecord], filt: FilterConfig
) -> list[PipelineRecord]:
    """Apply CLI filters to pipeline records."""
    result = records
    if filt.topology:
        result = [r for r in result if r.topology == filt.topology]
    if filt.n_qubits is not None:
        result = [r for r in result if r.n_qubits == filt.n_qubits]
    if filt.p_layers is not None:
        result = [r for r in result if r.p_layers == filt.p_layers]
    return result


def _apply_noisy_filters(records: list[NoisyRecord], filt: FilterConfig) -> list[NoisyRecord]:
    """Apply CLI filters to noisy records."""
    result = records
    if filt.topology:
        result = [r for r in result if r.topology == filt.topology]
    if filt.n_qubits is not None:
        result = [r for r in result if r.n_qubits == filt.n_qubits]
    if filt.p_layers is not None:
        result = [r for r in result if r.p_layers == filt.p_layers]
    return result


def _apply_experiment_filters(
    records: list[ExperimentRecord], filt: FilterConfig
) -> list[ExperimentRecord]:
    """Apply CLI filters to experiment records."""
    result = records
    if filt.topology:
        result = [r for r in result if r.topology == filt.topology]
    if filt.n_qubits is not None:
        result = [r for r in result if r.n_qubits == filt.n_qubits]
    if filt.p_layers is not None:
        result = [r for r in result if r.p_layers == filt.p_layers]
    return result


# ─── Extended Analytics (13-18) ──────────────────────────────────────────────


def report_reproducibility(pipeline: list[PipelineRecord], filt: FilterConfig) -> None:
    """Report per-seed reproducibility statistics (#13).

    For each (topology, N, p) config, compute:
    - std(ΔE/gap) across seeds
    - max - min spread
    - Flag configs with high variance
    """
    print_section("REPRODUCIBILITY ANALYSIS (per-seed variance)")

    # Group by (topology, n_qubits, p_layers, h_test)
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in pipeline:
        if r.de_gap is None or r.seed is None or r.p_layers is None:
            continue
        key = (r.topology, r.n_qubits, r.p_layers, r.h_test)
        groups[key].append(r.de_gap)

    # Only report groups with ≥2 seeds
    multi_seed = {k: v for k, v in groups.items() if len(v) >= 2}

    if not multi_seed:
        print("\n  No multi-seed configurations found.")
        return

    print(
        f"\n  {'Topology':<12} {'N':<4} {'p':<3} {'h_test':<8} "
        f"{'Seeds':<6} {'Mean':<9} {'Std':<9} {'Spread':<9} {'Status'}"
    )
    print("  " + "-" * 75)

    high_variance_count = 0
    for key in sorted(multi_seed.keys()):
        topo, n, p, h_test = key
        vals = multi_seed[key]
        n_seeds = len(vals)
        mean_val = sum(vals) / n_seeds
        variance = sum((v - mean_val) ** 2 for v in vals) / n_seeds
        std_val = variance**0.5
        spread = max(vals) - min(vals)

        # Flag if std > 50% of mean or spread > 2× pass threshold
        is_high_var = std_val > 0.5 * mean_val or spread > 2 * filt.pass_threshold
        status = "⚠️ HIGH" if is_high_var else "✅"
        if is_high_var:
            high_variance_count += 1

        h_str = f"{h_test:.2f}" if h_test is not None else "N/A"
        print(
            f"  {topo:<12} {n or '?':<4} {p:<3} {h_str:<8} "
            f"{n_seeds:<6} {mean_val:.4f}  {std_val:.4f}  {spread:.4f}  {status}"
        )

    print(f"\n  Total multi-seed configs: {len(multi_seed)}")
    print(f"  High-variance configs:    {high_variance_count}")


def report_staleness(pipeline: list[PipelineRecord]) -> None:
    """Report temporal staleness of results (#14).

    Extracts timestamps from pipeline_run_YYYYMMDD_HHMMSS.json filenames
    and flags configs not updated recently.
    """
    import re
    from datetime import datetime

    print_section("RESULT STALENESS (last update per config)")

    # Extract timestamp from filename
    ts_pattern = re.compile(r"pipeline_run_(\d{8})_(\d{6})\.json")

    config_dates: dict[tuple, datetime] = {}
    for r in pipeline:
        match = ts_pattern.search(r.file)
        if not match:
            continue
        date_str = match.group(1)
        time_str = match.group(2)
        try:
            ts = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
        except ValueError:
            continue
        key = (r.topology, r.n_qubits, r.p_layers)
        if key not in config_dates or ts > config_dates[key]:
            config_dates[key] = ts

    if not config_dates:
        print("\n  No timestamped results found.")
        return

    now = datetime.now()
    stale_threshold_days = 7

    print(f"\n  {'Topology':<12} {'N':<4} {'p':<4} {'Last Run':<20} {'Age (days)':<12} {'Status'}")
    print("  " + "-" * 65)

    stale_count = 0
    for key in sorted(config_dates.keys()):
        topo, n, p = key
        last_run = config_dates[key]
        age_days = (now - last_run).days
        is_stale = age_days > stale_threshold_days
        status = "⚠️ STALE" if is_stale else "✅ fresh"
        if is_stale:
            stale_count += 1
        p_str = str(p) if p is not None else "?"
        print(
            f"  {topo:<12} {n or '?':<4} p={p_str:<3} "
            f"{last_run.strftime('%Y-%m-%d %H:%M'):<20} {age_days:<12} {status}"
        )

    print(f"\n  Stale configs (>{stale_threshold_days} days): {stale_count}/{len(config_dates)}")


def report_h_coverage(pipeline: list[PipelineRecord]) -> None:
    """Report h_values coverage per config (#16).

    Shows which h points have been evaluated vs the training range.
    """
    print_section("h-VALUE COVERAGE (training points per config)")

    # Group by (topology, n_qubits, p_layers)
    configs: dict[tuple, dict] = defaultdict(lambda: {"h_train": set(), "h_test": set()})
    for r in pipeline:
        if r.p_layers is None:
            continue
        key = (r.topology, r.n_qubits, r.p_layers)
        for h in r.h_values:
            configs[key]["h_train"].add(h)
        if r.h_test is not None:
            configs[key]["h_test"].add(r.h_test)

    if not configs:
        print("\n  No data.")
        return

    print(
        f"\n  {'Topology':<12} {'N':<4} {'p':<3} "
        f"{'Train pts':<10} {'h range':<16} {'Test pts':<10} {'h_test values'}"
    )
    print("  " + "-" * 80)

    for key in sorted(configs.keys()):
        topo, n, p = key
        h_train = sorted(configs[key]["h_train"])
        h_test = sorted(configs[key]["h_test"])
        n_train = len(h_train)
        n_test = len(h_test)
        h_range = f"[{min(h_train):.2f}, {max(h_train):.2f}]" if h_train else "N/A"
        h_test_str = str([f"{h:.2f}" for h in h_test[:5]])
        if len(h_test) > 5:
            h_test_str += "..."
        print(
            f"  {topo:<12} {n or '?':<4} {p:<3} "
            f"{n_train:<10} {h_range:<16} {n_test:<10} {h_test_str}"
        )


def report_data_quality(pipeline: list[PipelineRecord], filt: FilterConfig) -> None:
    """Report data quality issues (#17).

    Flags:
    - Runs with abnormally high theta_smoothness
    - Runs with generalization_gap > threshold
    - Runs with empty diagnostics (old pipeline)
    """
    print_section("DATA QUALITY REPORT")

    smoothness_threshold = 0.5  # Flag if theta_smoothness > this
    gen_gap_threshold = 0.1  # Flag if generalization_gap > this

    high_smoothness: list[PipelineRecord] = []
    high_gen_gap: list[PipelineRecord] = []
    no_diagnostics: list[PipelineRecord] = []

    for r in pipeline:
        if r.theta_smoothness is not None and r.theta_smoothness > smoothness_threshold:
            high_smoothness.append(r)
        if r.generalization_gap is not None and r.generalization_gap > gen_gap_threshold:
            high_gen_gap.append(r)
        if r.theta_smoothness is None and r.generalization_gap is None and r.de_gap is not None:
            no_diagnostics.append(r)

    print(f"\n  Thresholds: smoothness>{smoothness_threshold}, gen_gap>{gen_gap_threshold}")

    # High smoothness
    print(f"\n  ── High θ-smoothness ({len(high_smoothness)} records) ──")
    if high_smoothness:
        for r in high_smoothness[:10]:
            print(
                f"    {r.topology:<12} N={r.n_qubits} p={r.p_layers} "
                f"smoothness={r.theta_smoothness:.4f} → {r.variant}"
            )
        if len(high_smoothness) > 10:
            print(f"    ... and {len(high_smoothness) - 10} more")
    else:
        print("    ✅ All within normal range")

    # High generalization gap
    print(f"\n  ── High generalization gap ({len(high_gen_gap)} records) ──")
    if high_gen_gap:
        for r in high_gen_gap[:10]:
            print(
                f"    {r.topology:<12} N={r.n_qubits} p={r.p_layers} "
                f"gen_gap={r.generalization_gap:.4f} → {r.variant}"
            )
        if len(high_gen_gap) > 10:
            print(f"    ... and {len(high_gen_gap) - 10} more")
    else:
        print("    ✅ All within normal range")

    # Missing diagnostics
    print(f"\n  ── Missing diagnostics ({len(no_diagnostics)} records) ──")
    if no_diagnostics:
        # Group by folder for conciseness
        by_folder: dict[str, int] = defaultdict(int)
        for r in no_diagnostics:
            by_folder[r.folder] += 1
        for folder, count in sorted(by_folder.items()):
            print(f"    {folder}: {count} records without diagnostics")
    else:
        print("    ✅ All records have diagnostics")

    total_issues = len(high_smoothness) + len(high_gen_gap) + len(no_diagnostics)
    print(f"\n  Total quality issues: {total_issues}")


def export_csv(
    pipeline: list[PipelineRecord],
    noisy: list[NoisyRecord],
    output_path: Path,
) -> None:
    """Export pipeline results to CSV (#18)."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "type",
                "folder",
                "variant",
                "topology",
                "n_qubits",
                "p_layers",
                "seed",
                "h_test",
                "de_gap",
                "verdict",
                "in_valid_regime",
                "worst_de_gap",
                "n_test_points",
                "n_restarts",
                "hidden_dim",
                "theta_smoothness",
                "generalization_gap",
                "file",
            ]
        )
        for r in pipeline:
            writer.writerow(
                [
                    "pipeline",
                    r.folder,
                    r.variant,
                    r.topology,
                    r.n_qubits,
                    r.p_layers,
                    r.seed,
                    r.h_test,
                    r.de_gap,
                    r.verdict,
                    r.in_valid_regime,
                    r.worst_de_gap,
                    r.n_test_points,
                    r.n_restarts,
                    r.hidden_dim,
                    r.theta_smoothness,
                    r.generalization_gap,
                    r.file,
                ]
            )
        for r in noisy:
            writer.writerow(
                [
                    "noisy",
                    r.folder,
                    r.variant,
                    r.topology,
                    r.n_qubits,
                    r.p_layers,
                    r.seed,
                    "",
                    r.mean_gain_pct,
                    r.zne_works,
                    "",
                    r.mean_r2,
                    r.n_layouts,
                    "",
                    "",
                    "",
                    "",
                    r.file,
                ]
            )

    print(f"  📄 Exported CSV to {output_path}", file=sys.stderr)


def export_markdown(
    pipeline: list[PipelineRecord],
    noisy: list[NoisyRecord],
    experiments: list[ExperimentRecord],
    recommendations: list[dict],
    output_path: Path,
) -> None:
    """Export summary as markdown table (#18)."""
    lines: list[str] = []
    lines.append("# Coverage Scanner Results\n")
    lines.append(f"Generated: {__import__('datetime').datetime.now().isoformat()}\n")

    # Pipeline summary table
    lines.append("## Pipeline Results Summary\n")
    lines.append("| Topology | N | p | Points | Pass | Median ΔE/gap |")
    lines.append("|----------|---|---|--------|------|---------------|")

    by_config: dict[tuple, list[PipelineRecord]] = defaultdict(list)
    for r in pipeline:
        if r.p_layers is not None:
            by_config[(r.topology, r.n_qubits, r.p_layers)].append(r)

    for key in sorted(by_config.keys()):
        topo, n, p = key
        group = by_config[key]
        n_pass = sum(1 for r in group if r.verdict == "PASS")
        de_vals = sorted(r.de_gap for r in group if r.de_gap is not None)
        median = _compute_median(de_vals)
        med_str = f"{median:.4f}" if median is not None else "N/A"
        lines.append(f"| {topo} | {n} | {p} | {len(group)} | {n_pass} | {med_str} |")

    # Noisy summary
    lines.append("\n## Noisy/ZNE Results\n")
    lines.append("| Topology | N | p | Runs | Mean Gain | ZNE Works |")
    lines.append("|----------|---|---|------|-----------|-----------|")

    noisy_by: dict[tuple, list[NoisyRecord]] = defaultdict(list)
    for r in noisy:
        if r.p_layers is not None:
            noisy_by[(r.topology, r.n_qubits, r.p_layers)].append(r)

    for key in sorted(noisy_by.keys()):
        topo, n, p = key
        group = noisy_by[key]
        gains = [r.mean_gain_pct for r in group if r.mean_gain_pct is not None]
        mean_g = sum(gains) / len(gains) if gains else 0
        n_works = sum(1 for r in group if r.zne_works)
        lines.append(
            f"| {topo} | {n} | {p} | {len(group)} | {mean_g:+.1f}% | {n_works}/{len(group)} |"
        )

    # Recommendations
    if recommendations:
        lines.append("\n## Recommendations\n")
        for i, rec in enumerate(sorted(recommendations, key=lambda x: x["priority"]), 1):
            p_label = {1: "HIGH", 2: "HIGH", 3: "MEDIUM", 4: "LOW"}
            priority = p_label.get(rec["priority"], "LOW")
            lines.append(
                f"{i}. **[{priority}]** {rec['type']}: {rec['topology']} N={rec['n_qubits']}"
            )
            lines.append(f"   - Reason: {rec['reason']}")
            lines.append(f"   - Action: {rec['action']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"  📄 Exported Markdown to {output_path}", file=sys.stderr)


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run the comprehensive coverage scan."""
    parser = argparse.ArgumentParser(
        description="Scan all results and identify coverage gaps for p=1 vs p=2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Basic scan (hardcoded folder list)
  python analysis/scan_coverage.py

  # Auto-discover all result folders
  python analysis/scan_coverage.py --discover

  # Filter to specific topology and system size
  python analysis/scan_coverage.py --topology chain_1d --n-qubits 10

  # Only p=1 results with stricter thresholds
  python analysis/scan_coverage.py --p 1 --pass-threshold 0.03

  # Full analysis with all extended reports
  python analysis/scan_coverage.py --discover --extended

  # Export to JSON, CSV, and Markdown
  python analysis/scan_coverage.py --json out.json --csv out.csv --markdown out.md

  # Quiet mode (summary only)
  python analysis/scan_coverage.py --quiet
""",
    )

    # ── Output options ──
    out_group = parser.add_argument_group("output options")
    out_group.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Save structured results to JSON file",
    )
    out_group.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Export results to CSV file",
    )
    out_group.add_argument(
        "--markdown",
        "--md",
        type=Path,
        default=None,
        help="Export summary as Markdown file",
    )

    # ── Filter options ──
    filt_group = parser.add_argument_group("filter options")
    filt_group.add_argument(
        "--topology",
        type=str,
        default=None,
        choices=["chain_1d", "heavy_hex", "ladder", "triangular", "kagome"],
        help="Filter results to a specific topology",
    )
    filt_group.add_argument(
        "--n-qubits",
        type=int,
        default=None,
        help="Filter results to a specific system size (e.g., 6, 10, 16, 20)",
    )
    filt_group.add_argument(
        "--p",
        type=int,
        default=None,
        choices=[1, 2],
        help="Filter results to p=1 or p=2 only",
    )
    filt_group.add_argument(
        "--pass-threshold",
        type=float,
        default=DEFAULT_PASS_THRESHOLD,
        help=f"ΔE/gap threshold for PASS verdict (default: {DEFAULT_PASS_THRESHOLD})",
    )
    filt_group.add_argument(
        "--marginal-threshold",
        type=float,
        default=DEFAULT_MARGINAL_THRESHOLD,
        help=f"ΔE/gap threshold for MARGINAL verdict (default: {DEFAULT_MARGINAL_THRESHOLD})",
    )

    # ── Scan options ──
    scan_group = parser.add_argument_group("scan options")
    scan_group.add_argument(
        "--discover",
        action="store_true",
        default=False,
        help="Auto-discover all subdirectories under results/thesis/ "
        "(instead of using the hardcoded folder list)",
    )
    scan_group.add_argument(
        "--extended",
        action="store_true",
        default=False,
        help="Include extended analytics: reproducibility, staleness, "
        "h-coverage, and data quality reports",
    )

    # ── Verbosity ──
    verb_group = parser.add_argument_group("verbosity")
    verb_excl = verb_group.add_mutually_exclusive_group()
    verb_excl.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Show warnings about missing p_layers and per-file parse details",
    )
    verb_excl.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Only show the final summary (suppress section reports)",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="  ⚠️  %(message)s",
        stream=sys.stderr,
    )

    # Apply custom thresholds globally
    # Note: These are module-level defaults used by PipelineRecord.verdict property
    if args.pass_threshold != 0.05 or args.marginal_threshold != 0.10:
        _update_thresholds(args.pass_threshold, args.marginal_threshold)

    # Build filter config
    filt = FilterConfig(
        topology=args.topology,
        n_qubits=args.n_qubits,
        p_layers=args.p,
        pass_threshold=args.pass_threshold,
        marginal_threshold=args.marginal_threshold,
    )

    if not args.quiet:
        print("=" * 80)
        print("  GNN-HVA COVERAGE SCANNER — Identifying Simulation Gaps")
        mode = "auto-discovery" if args.discover else "hardcoded folders"
        print(f"  Scanning: results/thesis/, results/experiments/ ({mode})")
        if any([filt.topology, filt.n_qubits is not None, filt.p_layers is not None]):
            filters_str = ", ".join(
                f
                for f in [
                    f"topology={filt.topology}" if filt.topology else "",
                    f"N={filt.n_qubits}" if filt.n_qubits is not None else "",
                    f"p={filt.p_layers}" if filt.p_layers is not None else "",
                ]
                if f
            )
            print(f"  Filters: {filters_str}")
        if args.pass_threshold != 0.05 or args.marginal_threshold != 0.10:
            print(f"  Thresholds: PASS<{args.pass_threshold}, MARGINAL<{args.marginal_threshold}")
        print("=" * 80)

    # 1. Scan all sources
    if not args.quiet:
        print("\n[1/3] Scanning pipeline results...", file=sys.stderr)
    pipeline = scan_pipeline_results(discover=args.discover)
    if not args.quiet:
        print(f"  Found {len(pipeline)} pipeline test points", file=sys.stderr)

    if not args.quiet:
        print("[2/3] Scanning noisy/ZNE results...", file=sys.stderr)
    noisy = scan_noisy_results(discover=args.discover)
    if not args.quiet:
        print(f"  Found {len(noisy)} noisy results", file=sys.stderr)

    if not args.quiet:
        print("[3/3] Scanning experiments...", file=sys.stderr)
    experiments = scan_experiment_results()
    if not args.quiet:
        print(f"  Found {len(experiments)} experiments", file=sys.stderr)

    # 2. Apply filters
    pipeline = _apply_pipeline_filters(pipeline, filt)
    noisy = _apply_noisy_filters(noisy, filt)
    experiments = _apply_experiment_filters(experiments, filt)

    # 3. Separate by p_layers (None goes to a separate bucket for reporting)
    p1_pipeline = [r for r in pipeline if r.p_layers == 1]
    p2_pipeline = [r for r in pipeline if r.p_layers == 2]
    p_unknown = [r for r in pipeline if r.p_layers is None]

    if p_unknown and not args.quiet:
        print(
            f"\n  ⚠️  {len(p_unknown)} records with unknown p_layers (use --verbose for details)",
            file=sys.stderr,
        )

    # 4. Reports
    if not args.quiet:
        report_pipeline_coverage(p2_pipeline, p1_pipeline)
        report_noisy_coverage(noisy)
        report_experiments(experiments)

    recommendations = report_gap_analysis(p2_pipeline, p1_pipeline, noisy)

    # 5. Extended analytics (opt-in)
    if args.extended and not args.quiet:
        report_reproducibility(pipeline, filt)
        report_staleness(pipeline)
        report_h_coverage(pipeline)
        report_data_quality(pipeline, filt)

    # 6. Summary (always shown)
    report_summary(pipeline, noisy, experiments, recommendations)

    # 7. Exports
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "pipeline": [asdict(r) for r in pipeline],
            "noisy": [asdict(r) for r in noisy],
            "experiments": [asdict(r) for r in experiments],
            "recommendations": recommendations,
            "summary": {
                "total_pipeline": len(pipeline),
                "p1_noiseless": len(p1_pipeline),
                "p2_noiseless": len(p2_pipeline),
                "p_unknown_noiseless": len(p_unknown),
                "total_noisy": len(noisy),
                "total_experiments": len(experiments),
                "n_gaps": len(recommendations),
            },
            "filters": asdict(filt),
        }
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  📄 Saved structured data to {args.json}", file=sys.stderr)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        export_csv(pipeline, noisy, args.csv)

    if args.markdown:
        export_markdown(pipeline, noisy, experiments, recommendations, args.markdown)


if __name__ == "__main__":
    main()
