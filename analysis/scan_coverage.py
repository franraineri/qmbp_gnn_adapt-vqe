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
    python analysis/scan_coverage.py
    python analysis/scan_coverage.py --json output.json   # Save structured data
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
THESIS = RESULTS / "thesis"
EXPERIMENTS = RESULTS / "experiments"

# All known thesis variant folders with their default topology
VARIANT_FOLDERS: list[tuple[str, str]] = [
    ("variants_N6_N10_1D_linnear", "chain_1d"),
    ("variants_N6_ladder", "ladder"),
    ("variants_N6_triangular", "triangular"),
    ("variants_N10_ladder", "ladder"),
    ("variants_N10_triangular", "triangular"),
    ("variants", "chain_1d"),  # Legacy variants folder
    ("n6_noiseless", "chain_1d"),  # Single baseline run
]

# Valid regime boundaries for p=1 (h_test must be >= this value)
P1_VALID_REGIME: dict[tuple[str, int], float] = {
    ("chain_1d", 6): 1.6,
    ("chain_1d", 10): 1.9,
    ("chain_1d", 20): 2.25,
    ("ladder", 6): 2.0,
    ("ladder", 10): 2.0,
    ("triangular", 6): 3.0,
    ("triangular", 10): 3.5,
}

# Valid regime boundaries for p=2
P2_VALID_REGIME: dict[tuple[str, int], float] = {
    ("chain_1d", 6): 1.25,
    ("chain_1d", 10): 1.5,
    ("chain_1d", 20): 2.0,
    ("ladder", 6): 1.5,
    ("ladder", 10): 2.0,
    ("triangular", 6): 2.0,
    ("triangular", 10): 2.5,
}


@dataclass
class PipelineRecord:
    """A single noiseless pipeline result."""

    folder: str
    variant: str
    topology: str
    n_qubits: int | None
    p_layers: int
    de_gap: float | None
    h_test: float | None
    h_values: list[float] = field(default_factory=list)
    seed: int | None = None
    n_restarts: int | None = None
    hidden_dim: int | None = None
    theta_smoothness: float | None = None
    generalization_gap: float | None = None
    file: str = ""

    @property
    def verdict(self) -> str:
        if self.de_gap is None:
            return "NO_DATA"
        if self.de_gap < 0.05:
            return "PASS"
        if self.de_gap < 0.10:
            return "MARGINAL"
        return "FAIL"

    @property
    def in_valid_regime(self) -> bool:
        """Check if h_test is within the valid regime for this p/topology/N."""
        if self.h_test is None or self.n_qubits is None:
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
    p_layers: int
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
    p_layers: int
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


def _extract_p_layers(data: dict) -> int:
    """Extract p_layers from various JSON structures."""
    config = data.get("config", {})
    system = data.get("system", {})
    # Check config first, then system, default to 2
    p = config.get("p_layers") or system.get("p_layers")
    if p is not None:
        return int(p)
    return 2


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


def scan_pipeline_results() -> list[PipelineRecord]:
    """Scan ALL pipeline_run_*.json files across all thesis folders."""
    records: list[PipelineRecord] = []

    for folder_name, default_topo in VARIANT_FOLDERS:
        folder_path = THESIS / folder_name
        if not folder_path.exists():
            continue

        # Handle flat folder (n6_noiseless has pipeline_run directly)
        pipeline_files = sorted(folder_path.glob("pipeline_run_*.json"), reverse=True)
        if pipeline_files:
            _parse_pipeline_file(pipeline_files[0], folder_name, folder_name, default_topo, records)

        # Handle nested folders (variants_N10_ladder/nl_restarts_5/pipeline_run_*.json)
        for subdir in sorted(folder_path.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue
            if subdir.name == "checkpoints":
                continue

            pipeline_files = sorted(subdir.glob("pipeline_run_*.json"), reverse=True)
            if pipeline_files:
                _parse_pipeline_file(
                    pipeline_files[0], folder_name, subdir.name, default_topo, records
                )

    return records


def _parse_pipeline_file(
    path: Path,
    folder_name: str,
    variant_name: str,
    default_topo: str,
    records: list[PipelineRecord],
) -> None:
    """Parse a single pipeline_run JSON file into a PipelineRecord."""
    data = _safe_load_json(path)
    if data is None:
        return

    config = data.get("config", {})
    p4 = data.get("phase4_results", [])
    diag = data.get("diagnostics", {})
    mpnn_config = config.get("mpnn", {})

    de_gap = p4[0].get("delta_e_over_gap") if p4 else None
    h_test = p4[0].get("h_test") if p4 else None

    records.append(
        PipelineRecord(
            folder=folder_name,
            variant=variant_name,
            topology=_extract_topology(data, default_topo),
            n_qubits=_extract_n_qubits(data),
            p_layers=_extract_p_layers(data),
            de_gap=de_gap,
            h_test=h_test,
            h_values=config.get("h_values", []),
            seed=config.get("seed"),
            n_restarts=config.get("n_restarts"),
            hidden_dim=mpnn_config.get("hidden_dim"),
            theta_smoothness=diag.get("phase2", {}).get("theta_smoothness"),
            generalization_gap=diag.get("phase3", {}).get("generalization_gap"),
            file=str(path.relative_to(ROOT)),
        )
    )


def scan_noisy_results() -> list[NoisyRecord]:
    """Scan ALL noisy_*.json files across all thesis folders."""
    records: list[NoisyRecord] = []

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
    ]

    for folder_name, default_topo in search_folders:
        folder_path = THESIS / folder_name
        if not folder_path.exists():
            continue

        # Flat noisy files
        for nf in sorted(folder_path.glob("noisy_*.json"), reverse=True):
            _parse_noisy_file(nf, folder_name, folder_name, default_topo, records)

        # Nested noisy files
        for subdir in sorted(folder_path.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue
            for nf in sorted(subdir.glob("noisy_*.json"), reverse=True):
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
            p_layers=_extract_p_layers(data),
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

        records.append(
            ExperimentRecord(
                experiment_id=exp_dir.name,
                p_layers=system.get("p_layers", config.get("p_layers", 2)),
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
    print(f"\n  ── p=2 ({len(p2)} runs) ──")
    p2_by = defaultdict(list)
    for r in p2:
        p2_by[(r.topology, r.n_qubits)].append(r)

    print(f"\n  {'Topology':<15} {'N':<5} {'Runs':<6} {'Pass':<6} {'Median':<10} {'Seeds'}")
    print("  " + "-" * 65)
    for key in sorted(p2_by.keys()):
        topo, n = key
        group = p2_by[key]
        n_pass = sum(1 for r in group if r.verdict == "PASS")
        de_vals = sorted(r.de_gap for r in group if r.de_gap is not None)
        median = de_vals[len(de_vals) // 2] if de_vals else None
        seeds = sorted(set(r.seed for r in group if r.seed is not None))
        med_str = f"{median:.4f}" if median is not None else "N/A"
        print(f"  {topo:<15} {n:<5} {len(group):<6} {n_pass:<6} {med_str:<10} {seeds}")

    # p=1 summary
    print(f"\n  ── p=1 ({len(p1)} runs) ──")
    if not p1:
        print("  ⚠️  NO p=1 noiseless pipeline results found!")
        return

    p1_by = defaultdict(list)
    for r in p1:
        p1_by[(r.topology, r.n_qubits)].append(r)

    print(
        f"\n  {'Topology':<15} {'N':<5} {'Runs':<6} {'Pass':<6} {'h_test':<8} "
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
    for r in sorted(p1, key=lambda x: (x.topology, x.n_qubits or 0)):
        de = f"{r.de_gap:.4f}" if r.de_gap is not None else "N/A"
        seed_str = str(r.seed) if r.seed is not None else "—"
        valid_str = "✓" if r.in_valid_regime else "✗"
        print(
            f"    {r.topology:<12} N={r.n_qubits or '?':<3} seed={seed_str:<5} "
            f"h_test={r.h_test} [{valid_str}] ΔE/gap={de} [{r.verdict}] "
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

    print(f"\n  Total: {len(records)} (p=1: {len(p1)}, p=2: {len(p2)})")

    if p1:
        print("\n  p=1 experiments:")
        for r in p1:
            desc = r.description[:55] if r.description else "—"
            print(f"    {r.experiment_id}: N={r.n_qubits} {r.topology} — {desc}")

    # Show all experiments with pass_rate
    print(f"\n  {'Experiment':<14} {'p':<4} {'N':<5} {'Topology':<12} {'Pass%':<8} {'Description'}")
    print("  " + "-" * 80)
    for r in sorted(records, key=lambda x: x.experiment_id):
        pr = f"{r.pass_rate:.0%}" if r.pass_rate is not None else "—"
        desc = r.description[:40] if r.description else "—"
        print(
            f"  {r.experiment_id:<14} p={r.p_layers:<3} {r.n_qubits or '?':<5} "
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
    p1_noisy = [r for r in noisy if r.p_layers == 1]
    p2_noisy = [r for r in noisy if r.p_layers == 2]

    total = len(pipeline) + len(noisy) + len(experiments)
    print(f"\n  Total data points scanned: {total}")
    print(f"  ├── Noiseless pipeline: {len(pipeline)} (p=1: {len(p1_nl)}, p=2: {len(p2_nl)})")
    print(f"  ├── Noisy/ZNE:          {len(noisy)} (p=1: {len(p1_noisy)}, p=2: {len(p2_noisy)})")
    print(f"  └── Experiments:        {len(experiments)}")
    print(f"\n  Gaps identified: {len(recommendations)}")
    high = sum(1 for r in recommendations if r["priority"] <= 2)
    med = sum(1 for r in recommendations if r["priority"] == 3)
    low = sum(1 for r in recommendations if r["priority"] >= 4)
    print(f"  ├── HIGH priority: {high}")
    print(f"  ├── MEDIUM priority: {med}")
    print(f"  └── LOW priority: {low}")


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run the comprehensive coverage scan."""
    parser = argparse.ArgumentParser(
        description="Scan all results and identify coverage gaps for p=1 vs p=2.",
        epilog="Example: python analysis/scan_coverage.py --json analysis/raw_data/coverage.json",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Save structured results to JSON file",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("  GNN-HVA COVERAGE SCANNER — Identifying Simulation Gaps")
    print("  Scanning: results/thesis/, results/experiments/")
    print("=" * 80)

    # 1. Scan all sources
    print("\n[1/3] Scanning pipeline results...", file=sys.stderr)
    pipeline = scan_pipeline_results()
    print(f"  Found {len(pipeline)} pipeline results", file=sys.stderr)

    print("[2/3] Scanning noisy/ZNE results...", file=sys.stderr)
    noisy = scan_noisy_results()
    print(f"  Found {len(noisy)} noisy results", file=sys.stderr)

    print("[3/3] Scanning experiments...", file=sys.stderr)
    experiments = scan_experiment_results()
    print(f"  Found {len(experiments)} experiments", file=sys.stderr)

    # 2. Separate by p_layers
    p1_pipeline = [r for r in pipeline if r.p_layers == 1]
    p2_pipeline = [r for r in pipeline if r.p_layers == 2]

    # 3. Reports
    report_pipeline_coverage(p2_pipeline, p1_pipeline)
    report_noisy_coverage(noisy)
    report_experiments(experiments)
    recommendations = report_gap_analysis(p2_pipeline, p1_pipeline, noisy)
    report_summary(pipeline, noisy, experiments, recommendations)

    # 4. Optional JSON export
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
                "total_noisy": len(noisy),
                "total_experiments": len(experiments),
                "n_gaps": len(recommendations),
            },
        }
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  📄 Saved structured data to {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
