#!/usr/bin/env python
"""Check how |ΔE| (absolute, NOT divided by gap) behaves for harder topologies
as p changes. Focus: N=10, TFIM, all topologies, p=1-4.

This answers: do ladder/square/triangular improve in absolute energy error
when we increase depth, even if ΔE/gap doesn't pass the 5% threshold?

Usage:
    python scripts/analysis/check_delta_e_by_topo_p.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE / "results" / "experiments"


def extract_deploy_points(data: dict) -> list[dict] | None:
    results = data.get("results", {})
    for sec_key in ["section_4", "section_5", "section_6"]:
        sec = results.get(sec_key, {})
        if not isinstance(sec, dict):
            continue
        if "deploy" not in sec.get("name", "").lower():
            continue
        sec_data = sec.get("data", {})
        if not isinstance(sec_data, dict):
            continue
        points = sec_data.get("per_point", [])
        if points and isinstance(points[0], dict) and "e_exact" in points[0]:
            return points
        for topo_data in sec_data.get("topologies", {}).values():
            pts = topo_data.get("per_point", [])
            if pts and isinstance(pts[0], dict) and "e_exact" in pts[0]:
                return pts
    return None


def get_config(data: dict, filepath: str) -> dict:
    config = data.get("config", {})
    system = config.get("system", {})
    model = system.get("model", config.get("model", ""))
    n = system.get("n_qubits", config.get("n_qubits"))
    p = system.get("p_layers", config.get("p_layers"))
    topos = system.get("topologies", [])
    topology = topos[0] if isinstance(topos, list) and len(topos) == 1 else ""
    if not model:
        if "tfim_longitudinal" in filepath:
            model = "tfim_longitudinal"
        elif "tfim" in filepath:
            model = "tfim"
    if not topology:
        for t in ["heavy_hex", "chain_1d", "ladder", "square", "triangular"]:
            if t in filepath:
                topology = t
                break
    return {"model": model, "topology": topology, "n_qubits": n, "p_layers": p}


def main() -> int:
    print("Scanning N=10, TFIM, all topologies, p=1-4...")
    print("Looking at |ΔE| absolute (not divided by gap)\n")

    # Collect: (topology, p) -> list of per-run mean |ΔE|
    results = defaultdict(list)

    for exp_dir in RESULTS_DIR.glob("exp_noiseless*"):
        for run_file in exp_dir.rglob("run_*.json"):
            fp = str(run_file)
            try:
                data = json.loads(run_file.read_text())
            except:
                continue
            cfg = get_config(data, fp)

            # Filter: N=10, TFIM (not longitudinal), p=1-4
            if cfg["model"] != "tfim":
                continue
            if cfg["n_qubits"] != 10:
                continue
            p = cfg["p_layers"]
            if p is None or p < 1 or p > 4:
                continue
            topo = cfg["topology"]
            if not topo:
                continue

            points = extract_deploy_points(data)
            if not points:
                continue

            # Compute |ΔE| per point
            delta_es = []
            de_gaps = []
            for pt in points:
                e_pred = pt.get("e_pred")
                e_exact = pt.get("e_exact")
                if e_pred is None or e_exact is None:
                    continue
                delta_es.append(abs(e_pred - e_exact))
                deg = pt.get("de_gap")
                if deg is not None:
                    de_gaps.append(deg)

            if not delta_es:
                continue

            results[(topo, p)].append(
                {
                    "delta_e_mean": np.mean(delta_es),
                    "delta_e_median": np.median(delta_es),
                    "delta_e_max": np.max(delta_es),
                    "delta_e_std": np.std(delta_es),
                    "de_gap_mean": np.mean(de_gaps) if de_gaps else None,
                    "n_points": len(delta_es),
                    "pass_rate": sum(1 for d in de_gaps if d < 0.05) / len(de_gaps)
                    if de_gaps
                    else 0,
                }
            )

    # Print table grouped by topology
    topos_order = ["chain_1d", "heavy_hex", "ladder", "square", "triangular"]
    p_values = [1, 2, 3, 4]

    print("=" * 100)
    print(
        f"{'Topology':<12} {'p':>2} {'Runs':>5} "
        f"{'|ΔE| mean':>10} {'|ΔE| med':>10} {'|ΔE| max':>10} "
        f"{'ΔE/gap':>8} {'Pass%':>7}"
    )
    print("-" * 100)

    for topo in topos_order:
        for p in p_values:
            key = (topo, p)
            if key not in results:
                continue
            runs = results[key]
            # Pick best run (highest pass rate)
            best = max(runs, key=lambda r: r["pass_rate"])
            deg_str = f"{best['de_gap_mean']:.4f}" if best["de_gap_mean"] else "N/A"
            print(
                f"{topo:<12} {p:>2} {len(runs):>5} "
                f"{best['delta_e_mean']:>10.5f} {best['delta_e_median']:>10.5f} "
                f"{best['delta_e_max']:>10.5f} {deg_str:>8} "
                f"{best['pass_rate'] * 100:>6.0f}%"
            )
        if topo != topos_order[-1]:
            print()

    print("=" * 100)

    # Summary: focus on "hard" topologies
    print("\n\n=== RESUMEN: Topologías difíciles (ladder, square, triangular) ===\n")
    print("¿Mejora |ΔE| absoluto al aumentar p?")
    print("-" * 70)
    for topo in ["ladder", "square", "triangular"]:
        print(f"\n  {topo}:")
        for p in p_values:
            key = (topo, p)
            if key not in results:
                print(f"    p={p}: sin datos")
                continue
            best = max(results[key], key=lambda r: r["pass_rate"])
            deg_str = f"ΔE/gap={best['de_gap_mean']:.3f}" if best["de_gap_mean"] else ""
            print(
                f"    p={p}: |ΔE|={best['delta_e_mean']:.5f} "
                f"(max={best['delta_e_max']:.4f}) {deg_str} "
                f"pass={best['pass_rate'] * 100:.0f}%"
            )

    # Compute improvement ratios
    print("\n\n=== MEJORA p=1→p=4 en |ΔE| ===\n")
    for topo in topos_order:
        p1_key = (topo, 1)
        p4_key = (topo, 4)
        if p1_key in results and p4_key in results:
            best_p1 = max(results[p1_key], key=lambda r: r["pass_rate"])
            best_p4 = max(results[p4_key], key=lambda r: r["pass_rate"])
            ratio = (
                best_p1["delta_e_mean"] / best_p4["delta_e_mean"]
                if best_p4["delta_e_mean"] > 0
                else 0
            )
            print(
                f"  {topo:<12}: p=1 |ΔE|={best_p1['delta_e_mean']:.5f} → "
                f"p=4 |ΔE|={best_p4['delta_e_mean']:.5f} "
                f"(mejora {ratio:.1f}×)"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
