#!/usr/bin/env python
"""Extract ΔE (absolute) and Fidelity metrics from noiseless pipeline runs.

Groups results by N, p, topology, and model. Outputs a summary table
showing how ΔE and Fidelity behave across configurations.

Usage:
    .venv/bin/python scripts/analysis/extract_delta_e_fidelity.py [--json output.json]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PointMetrics:
    """Metrics for a single h-point deployment."""

    h: float
    delta_e: float  # |E_pred - E_exact|
    de_gap: float  # ΔE / gap
    gap: float
    fidelity: float | None
    passed: bool  # de_gap < 0.05


@dataclass
class RunSummary:
    """Summary metrics for a single pipeline run."""

    source: str
    model: str
    topology: str
    n_qubits: int
    p_layers: int
    n_points: int
    n_pass: int
    # ΔE stats
    delta_e_mean: float
    delta_e_max: float
    delta_e_min: float
    # Fidelity stats
    fidelity_mean: float | None
    fidelity_min: float | None
    # ΔE/gap stats (for reference)
    de_gap_mean: float
    de_gap_max: float


def extract_from_run(data: dict[str, Any], source: str) -> RunSummary | None:
    """Extract ΔE and fidelity from a single run JSON."""
    results = data.get("results", {})
    config = data.get("config", {})

    # Get section_4 deploy data
    s4 = results.get("section_4", {})
    s4d = s4.get("data", {}) if isinstance(s4, dict) else {}
    per_point = s4d.get("per_point", []) if isinstance(s4d, dict) else []

    if not per_point or not isinstance(per_point, list):
        return None
    if not isinstance(per_point[0], dict):
        return None
    if "e_exact" not in per_point[0] or "e_pred" not in per_point[0]:
        return None

    # Extract config (try multiple locations)
    n = config.get("n_qubits") or config.get("n") or config.get("N")
    if not n:
        s1d = results.get("section_1", {}).get("data", {})
        if isinstance(s1d, dict):
            n = s1d.get("n_qubits")
    if not n:
        n = data.get("metadata", {}).get("n_qubits")
    if not n:
        return None

    p = config.get("p_layers") or config.get("p")
    if not p:
        p = data.get("metadata", {}).get("p_layers")

    topology = config.get("topology", "")
    if not topology:
        # Infer from source path
        for topo in ["chain_1d", "heavy_hex", "ladder", "square", "triangular"]:
            if topo in source:
                topology = topo
                break

    model = config.get("model", "")
    if not model:
        for m in ["tfim_longitudinal", "tfim_frustrated", "heisenberg", "tfim"]:
            if m in source:
                model = m
                break

    # Compute metrics per point
    delta_es = []
    fidelities = []
    de_gaps = []
    n_pass = 0

    for pt in per_point:
        e_pred = pt.get("e_pred")
        e_exact = pt.get("e_exact")
        if e_pred is None or e_exact is None:
            continue
        de = abs(e_pred - e_exact)
        delta_es.append(de)

        deg = pt.get("de_gap", 0)
        de_gaps.append(deg)
        if deg < 0.05:
            n_pass += 1

        fid = pt.get("fidelity")
        if fid is not None and isinstance(fid, (int, float)):
            fidelities.append(fid)

    if not delta_es:
        return None

    return RunSummary(
        source=source,
        model=model or "unknown",
        topology=topology or "unknown",
        n_qubits=int(n),
        p_layers=int(p) if p else 0,
        n_points=len(delta_es),
        n_pass=n_pass,
        delta_e_mean=sum(delta_es) / len(delta_es),
        delta_e_max=max(delta_es),
        delta_e_min=min(delta_es),
        fidelity_mean=sum(fidelities) / len(fidelities) if fidelities else None,
        fidelity_min=min(fidelities) if fidelities else None,
        de_gap_mean=sum(de_gaps) / len(de_gaps) if de_gaps else 0,
        de_gap_max=max(de_gaps) if de_gaps else 0,
    )


def scan_all_runs() -> list[RunSummary]:
    """Scan all noiseless experiment directories for runs with deploy data."""
    base = Path("results/experiments/exp_noiseless")
    summaries = []

    for f in sorted(base.rglob("run_*.json")):
        try:
            data = json.loads(f.read_text())
            if not isinstance(data, dict):
                continue
        except (json.JSONDecodeError, OSError):
            continue

        rel = str(f.relative_to(base))
        summary = extract_from_run(data, rel)
        if summary:
            summaries.append(summary)

    return summaries


def scan_cross_n() -> list[RunSummary]:
    """Scan cross-N zero-shot results."""
    base = Path("results/scaling/zero_shot")
    summaries = []

    for f in sorted(base.glob("zero_shot_v3_*.json")):
        try:
            data = json.loads(f.read_text())
            if not isinstance(data, dict):
                continue
        except (json.JSONDecodeError, OSError):
            continue

        strat = data.get("strategy_a_gnn_no_bn", {})
        results = strat.get("results", [])
        if not results:
            continue

        # Determine target N from filename
        name = f.stem
        n_target = None
        for token in name.split("_"):
            if token.startswith("N") and token[1:].isdigit():
                n_target = int(token[1:])
        if not n_target:
            # Try to infer from last "to_NXX" pattern
            parts = name.split("_to_")
            if len(parts) > 1:
                for t in parts[1].split("_"):
                    if t.startswith("N") and t[1:].isdigit():
                        n_target = int(t[1:])
                        break

        if not n_target:
            continue

        delta_es = []
        de_gaps = []
        n_pass = 0
        for r in results:
            de = r.get("energy_error_abs", abs(r.get("e_pred", 0) - r.get("e_dmrg", 0)))
            delta_es.append(de)
            deg = r.get("de_gap", 0)
            de_gaps.append(deg)
            if deg < 0.05:
                n_pass += 1

        if delta_es:
            summaries.append(
                RunSummary(
                    source=f.name,
                    model="tfim",
                    topology="chain_1d",
                    n_qubits=n_target,
                    p_layers=1,
                    n_points=len(delta_es),
                    n_pass=n_pass,
                    delta_e_mean=sum(delta_es) / len(delta_es),
                    delta_e_max=max(delta_es),
                    delta_e_min=min(delta_es),
                    fidelity_mean=None,  # MPS - no fidelity available
                    fidelity_min=None,
                    de_gap_mean=sum(de_gaps) / len(de_gaps),
                    de_gap_max=max(de_gaps),
                )
            )

    return summaries


def format_table(summaries: list[RunSummary]) -> str:
    """Format results as a readable table grouped by (N, p, topology)."""
    # Group by (model, topology, N, p) — keep best run per group
    groups: dict[tuple, list[RunSummary]] = defaultdict(list)
    for s in summaries:
        key = (s.model, s.topology, s.n_qubits, s.p_layers)
        groups[key].append(s)

    lines = []
    lines.append("=" * 95)
    lines.append(
        f"{'Model':<18} {'Topo':<12} {'N':>3} {'p':>2} "
        f"{'Deploy':>8} {'ΔE_mean':>10} {'ΔE_max':>10} "
        f"{'F_mean':>8} {'F_min':>8} {'ΔE/gap':>8}"
    )
    lines.append("-" * 95)

    # Sort by model, then N, then topology
    for key in sorted(groups.keys(), key=lambda k: (k[0], k[2], k[1], k[3])):
        runs = groups[key]
        # Pick best run (highest pass rate, then lowest ΔE)
        best = max(runs, key=lambda r: (r.n_pass / max(r.n_points, 1), -r.delta_e_mean))
        model, topo, n, p = key
        deploy = f"{best.n_pass}/{best.n_points}"
        de_mean = f"{best.delta_e_mean:.5f}"
        de_max = f"{best.delta_e_max:.5f}"
        f_mean = f"{best.fidelity_mean:.4f}" if best.fidelity_mean else "N/A"
        f_min = f"{best.fidelity_min:.4f}" if best.fidelity_min else "N/A"
        deg = f"{best.de_gap_mean:.4f}"

        lines.append(
            f"{model:<18} {topo:<12} {n:>3} {p:>2} "
            f"{deploy:>8} {de_mean:>10} {de_max:>10} "
            f"{f_mean:>8} {f_min:>8} {deg:>8}"
        )

    lines.append("=" * 95)
    lines.append(f"\nTotal configurations: {len(groups)}")
    lines.append(f"Total runs analyzed: {sum(len(v) for v in groups.values())}")
    return "\n".join(lines)


def main() -> int:
    print("Scanning noiseless pipeline runs...")
    noiseless = scan_all_runs()
    print(f"  Found {len(noiseless)} runs with deploy data (e_exact + e_pred)\n")

    print("Scanning cross-N zero-shot runs...")
    cross_n = scan_cross_n()
    print(f"  Found {len(cross_n)} cross-N results\n")

    all_runs = noiseless + cross_n

    if not all_runs:
        print("No runs found with ΔE data.")
        return 1

    print(format_table(all_runs))

    # Save JSON if requested
    if "--json" in sys.argv:
        idx = sys.argv.index("--json")
        if idx + 1 < len(sys.argv):
            out_path = Path(sys.argv[idx + 1])
            out_data = []
            for s in all_runs:
                out_data.append(
                    {
                        "model": s.model,
                        "topology": s.topology,
                        "n_qubits": s.n_qubits,
                        "p_layers": s.p_layers,
                        "n_points": s.n_points,
                        "n_pass": s.n_pass,
                        "delta_e_mean": s.delta_e_mean,
                        "delta_e_max": s.delta_e_max,
                        "delta_e_min": s.delta_e_min,
                        "fidelity_mean": s.fidelity_mean,
                        "fidelity_min": s.fidelity_min,
                        "de_gap_mean": s.de_gap_mean,
                        "source": s.source,
                    }
                )
            out_path.write_text(json.dumps(out_data, indent=2))
            print(f"\nSaved {len(out_data)} entries to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
