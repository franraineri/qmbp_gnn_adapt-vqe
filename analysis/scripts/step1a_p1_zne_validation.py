#!/usr/bin/env python3
"""Step 1A: Validate p=1 ZNE at N=10 across topologies and seeds.

Tests the CX-budget hypothesis: p=1 at N=10 has ~18 CX gates (same as
p=2 at N=6), so ZNE should work. Currently only n=2 data points exist.

Runs: 3 topologies × 3 seeds = 9 experiments using run_noisy_pipeline.py
Expected time: ~15 min total (~100s each)

Output: documentation/analysis/14_p1_zne_validation.md
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
OUTPUT_BASE = Path("results/thesis/analysis_p1_zne")
TOPOLOGIES = ["chain_1d", "ladder", "triangular"]
SEEDS = [42, 43, 44]
N_QUBITS = 10
P_LAYERS = 1
H_VALUES = ["4.0", "3.5", "3.0"]  # Deep paramagnetic (safe for p=1)
N_LAYOUTS = 3
SHOTS = 16384


def run_single(topology: str, seed: int) -> dict:
    """Run one noisy p=1 experiment."""
    output_dir = OUTPUT_BASE / f"{topology}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip if result already exists
    existing = sorted(output_dir.glob("noisy_*.json"), reverse=True)
    if existing:
        with open(existing[0]) as f:
            data = json.load(f)
        summary = data.get("summary", {})
        return {
            "topology": topology,
            "seed": seed,
            "mean_r2": summary.get("mean_r2", 0),
            "mean_gain_pct": summary.get("mean_gain_pct", 0),
            "n_mitigated_wins": summary.get("n_mitigated_wins", 0),
            "n_total": summary.get("n_total", 0),
            "success": summary.get("success_criteria_met", False),
            "elapsed_s": 0,
            "returncode": 0,
            "error": "(cached)",
        }

    cmd = [
        PYTHON,
        "scripts/run_noisy_pipeline.py",
        "--n-qubits",
        str(N_QUBITS),
        "--p",
        str(P_LAYERS),
        "--topology",
        topology,
        "--h-values",
        *H_VALUES,
        "--n-layouts",
        str(N_LAYOUTS),
        "--shots",
        str(SHOTS),
        "--seed",
        str(seed),
        "--n-restarts",
        "5",
        "--output-dir",
        str(output_dir),
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0

    # Parse the output JSON
    noisy_files = sorted(output_dir.glob("noisy_*.json"), reverse=True)
    if noisy_files:
        with open(noisy_files[0]) as f:
            data = json.load(f)
        summary = data.get("summary", {})
        return {
            "topology": topology,
            "seed": seed,
            "mean_r2": summary.get("mean_r2", 0),
            "mean_gain_pct": summary.get("mean_gain_pct", 0),
            "n_mitigated_wins": summary.get("n_mitigated_wins", 0),
            "n_total": summary.get("n_total", 0),
            "success": summary.get("success_criteria_met", False),
            "elapsed_s": elapsed,
            "returncode": result.returncode,
            "error": "",
        }
    else:
        return {
            "topology": topology,
            "seed": seed,
            "mean_r2": 0,
            "mean_gain_pct": 0,
            "n_mitigated_wins": 0,
            "n_total": 0,
            "success": False,
            "elapsed_s": elapsed,
            "returncode": result.returncode,
            "error": result.stderr[-200:] if result.stderr else "no output file",
        }


def main():
    print("=" * 70)
    print("Step 1A: p=1 ZNE Validation at N=10")
    print("=" * 70)
    print(f"Config: N={N_QUBITS}, p={P_LAYERS}, h={H_VALUES}")
    print(f"Topologies: {TOPOLOGIES}")
    print(f"Seeds: {SEEDS}")
    print(f"Layouts: {N_LAYOUTS}, Shots: {SHOTS}")
    print(f"Total runs: {len(TOPOLOGIES) * len(SEEDS)}")
    print()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    results = []

    for topo in TOPOLOGIES:
        for seed in SEEDS:
            label = f"{topo}/seed={seed}"
            print(f"  [{len(results) + 1}/9] {label}...", end=" ", flush=True)
            r = run_single(topo, seed)
            results.append(r)

            if r["returncode"] == 0:
                icon = (
                    "✅" if r["mean_gain_pct"] > 30 else ("⚠️" if r["mean_gain_pct"] > 0 else "❌")
                )
                print(
                    f"{icon} R²={r['mean_r2']:.3f}, "
                    f"gain={r['mean_gain_pct']:+.1f}%, "
                    f"wins={r['n_mitigated_wins']}/{r['n_total']} "
                    f"({r['elapsed_s']:.0f}s)"
                )
            else:
                print(f"💥 FAILED (rc={r['returncode']}, {r['elapsed_s']:.0f}s)")
                if r["error"]:
                    print(f"       {r['error'][:100]}")

    # ── Generate report
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    lines = []
    lines.append("# Estudio 1A — p=1 ZNE Validation at N=10\n")
    lines.append(
        f"**Config**: N={N_QUBITS}, p={P_LAYERS}, h={H_VALUES}, layouts={N_LAYOUTS}, shots={SHOTS}"
    )
    lines.append(f"**Runs**: {len(TOPOLOGIES)} topologies × {len(SEEDS)} seeds = 9 total\n")
    lines.append("## Resultados\n")
    lines.append("| Topology | Seed | R² | Gain% | Wins | Success? |")
    lines.append("|----------|------|----|-------|------|----------|")

    for r in results:
        icon = "✅" if r["mean_gain_pct"] > 30 else ("⚠️" if r["mean_gain_pct"] > 0 else "❌")
        wins = f"{r['n_mitigated_wins']}/{r['n_total']}"
        lines.append(
            f"| {r['topology']} | {r['seed']} | {r['mean_r2']:.4f} | "
            f"{r['mean_gain_pct']:+.1f}% | {wins} | {icon} |"
        )

    lines.append("")

    # Aggregate by topology
    lines.append("## Agregado por Topología\n")
    lines.append("| Topology | Mean R² | Mean Gain% | Positive Gain | Verdict |")
    lines.append("|----------|---------|------------|---------------|---------|")

    for topo in TOPOLOGIES:
        topo_results = [r for r in results if r["topology"] == topo and r["returncode"] == 0]
        if not topo_results:
            lines.append(f"| {topo} | — | — | — | ERROR |")
            continue
        mean_r2 = sum(r["mean_r2"] for r in topo_results) / len(topo_results)
        mean_gain = sum(r["mean_gain_pct"] for r in topo_results) / len(topo_results)
        n_positive = sum(1 for r in topo_results if r["mean_gain_pct"] > 0)
        verdict = (
            "✅ CONFIRMED" if mean_gain > 30 else ("⚠️ PARTIAL" if mean_gain > 0 else "❌ FAILS")
        )
        lines.append(
            f"| {topo} | {mean_r2:.4f} | {mean_gain:+.1f}% | "
            f"{n_positive}/{len(topo_results)} | {verdict} |"
        )

    lines.append("")

    # Overall verdict
    successful = [r for r in results if r["returncode"] == 0]
    n_positive_gain = sum(1 for r in successful if r["mean_gain_pct"] > 30)
    n_total_valid = len(successful)

    lines.append("## Veredicto Global\n")
    lines.append(f"- Runs exitosos: {n_total_valid}/9")
    lines.append(f"- Gain > +30%: {n_positive_gain}/{n_total_valid}")

    if n_positive_gain >= 7:
        lines.append(
            "\n**✅ CLAIM CONFIRMED**: p=1 at N=10 recovers ZNE effectiveness "
            "across topologies and seeds."
        )
    elif n_positive_gain >= 4:
        lines.append(
            "\n**⚠️ PARTIALLY CONFIRMED**: p=1 ZNE works on some topologies/seeds "
            "but not universally."
        )
    else:
        lines.append("\n**❌ CLAIM REJECTED**: p=1 does NOT reliably recover ZNE at N=10.")

    lines.append("\n## Implicación para la Tesis\n")
    lines.append("> [To be filled based on results]")

    output = "\n".join(lines)
    out_path = Path("documentation/analysis/14_p1_zne_validation.md")
    out_path.write_text(output)
    print(f"\nSaved to {out_path}")

    # Save raw JSON
    json_path = Path("documentation/analysis/raw_p1_zne_validation.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Raw data: {json_path}")

    # Print summary
    print(f"\nOverall: {n_positive_gain}/{n_total_valid} runs with gain > +30%")


if __name__ == "__main__":
    main()
