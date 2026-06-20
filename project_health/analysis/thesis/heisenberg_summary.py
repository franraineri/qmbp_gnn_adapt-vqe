#!/usr/bin/env python3
"""Heisenberg XXZ experiment summary and cross-N comparison.

Reuses the existing ResultScanner (scripts/digest/scanner.py) for discovery,
then enriches with Heisenberg-specific fields (phase2_summary, entanglement,
scientific_conclusion) from the pipeline_run JSON files.

Usage:
    # Summarize all Heisenberg results (auto-discovers N=6, N=10, etc.)
    python analysis/heisenberg_summary.py

    # Specific system size
    python analysis/heisenberg_summary.py --n-qubits 10

    # Export to JSON
    python analysis/heisenberg_summary.py --json results/thesis/heisenberg_summary.json

    # Compare N=6 vs N=10 scaling
    python analysis/heisenberg_summary.py --compare-scaling

    # Verbose (show per-h checkpoint data)
    python analysis/heisenberg_summary.py --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Reuse existing scanner infrastructure
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

RESULTS = ROOT / "results" / "thesis"


def find_heisenberg_folders() -> list[Path]:
    """Discover all Heisenberg variant result folders."""
    return sorted(RESULTS.glob("variants_N*_heisenberg"))


def enrich_with_heisenberg_data(folder: Path) -> list[dict]:
    """Parse Heisenberg-specific data from pipeline_run JSONs in a folder.

    Extracts phase2_summary, entanglement, scientific_conclusion, and
    checkpoint data that the standard NoiselessResult doesn't capture.
    """
    results = []
    for variant_dir in sorted(folder.iterdir()):
        if not variant_dir.is_dir():
            continue
        jsons = sorted(variant_dir.glob("pipeline_run_*.json"))
        if not jsons:
            continue

        try:
            data = json.loads(jsons[-1].read_text())
        except (json.JSONDecodeError, OSError):
            continue

        config = data.get("config", {})
        p2 = data.get("phase2_summary")
        sci = data.get("scientific_conclusion", {})
        ent = data.get("entanglement", [])
        diag = data.get("diagnostics", {})
        phase2_diag = diag.get("phase2", {})

        if not p2:
            continue

        # Extract N from folder name
        folder_n = int(folder.name.split("_")[1].replace("N", ""))

        # Load checkpoint for raw VQE/exact energies
        ckpt = variant_dir / "checkpoints" / "phase12_checkpoint.npz"
        ground_e = vqe_e = h_vals = None
        if ckpt.exists():
            try:
                npz = np.load(ckpt)
                ground_e = npz["ground_energies"].tolist()
                vqe_e = npz["vqe_energies"].tolist()
                h_vals = npz["h_values"].tolist()
            except Exception:
                pass

        results.append(
            {
                "directory": variant_dir.name,
                "n_qubits": folder_n,
                "model": config.get("model", "?"),
                "delta": config.get("delta"),
                "topology": config.get("topology", "?"),
                "p_layers": config.get("p_layers", 2),
                "seed": config.get("seed"),
                "n_restarts": config.get("n_restarts", 10),
                "max_fidelity": p2.get("max_fidelity", 0),
                "mean_fidelity": p2.get("mean_fidelity", 0),
                "n_above_threshold": p2.get("n_above_threshold", 0),
                "fidelity_threshold": p2.get("fidelity_threshold", 0.6),
                "classification": sci.get("classification", "?"),
                "max_entropy": max(e["entropy"] for e in ent) if ent else None,
                "theta_smoothness": phase2_diag.get("theta_smoothness"),
                "convergence_rate": phase2_diag.get("convergence_rate"),
                "elapsed_s": data.get("elapsed_s", 0),
                "ground_energies": ground_e,
                "vqe_energies": vqe_e,
                "h_values": h_vals,
                "per_h_fidelity": p2.get("per_h_fidelity"),
                "delta_e_over_gap": (
                    max(
                        r["delta_e_over_gap"]
                        for r in data.get("phase4_results", [])
                        if r.get("delta_e_over_gap") is not None
                    )
                    if data.get("phase4_results")
                    else None
                ),
            }
        )

    return results


def print_summary(results: list[dict], verbose: bool = False) -> None:
    """Print formatted summary grouped by N."""
    if not results:
        print("  No Heisenberg results found.")
        return

    # Group by N
    by_n: dict[int, list[dict]] = {}
    for r in results:
        by_n.setdefault(r["n_qubits"], []).append(r)

    for n in sorted(by_n.keys()):
        group = by_n[n]
        print(f"\n{'═' * 90}")
        print(f"  N={n} — {len(group)} variants")
        print(f"{'═' * 90}")

        # Classification counts
        from collections import Counter

        cls_counts = Counter(r["classification"] for r in group)
        print(f"  Classifications: {dict(cls_counts)}")
        print()

        # Table
        print(
            f"  {'Directory':<30} {'Model':<10} {'Δ':<5} {'Topo':<10} "
            f"{'MaxFid':<9} {'MaxS':<7} {'θ_sm':<6} {'Class'}"
        )
        print(f"  {'-' * 90}")

        for r in sorted(group, key=lambda x: x["max_fidelity"], reverse=True):
            d_str = f"{r['delta']:.1f}" if r["delta"] is not None else "N/A"
            s_str = f"{r['max_entropy']:.3f}" if r["max_entropy"] is not None else "—"
            th_str = f"{r['theta_smoothness']:.2f}" if r["theta_smoothness"] is not None else "—"
            cls_short = r["classification"].replace("negative_", "neg_").replace("full_", "")
            print(
                f"  {r['directory']:<30} {r['model']:<10} {d_str:<5} {r['topology']:<10} "
                f"{r['max_fidelity']:<9.4f} {s_str:<7} {th_str:<6} {cls_short}"
            )

            if verbose and r.get("ground_energies") and r.get("vqe_energies"):
                ge = r["ground_energies"]
                ve = r["vqe_energies"]
                hv = r["h_values"]
                print(
                    f"    h=[{hv[0]:.1f}→{hv[-1]:.1f}]  "
                    f"E_exact=[{ge[0]:.2f}→{ge[-1]:.2f}]  "
                    f"E_vqe=[{ve[0]:.2f}→{ve[-1]:.2f}]  "
                    f"gap={ve[0] - ge[0]:.1f}"
                )


def print_scaling_comparison(results: list[dict]) -> None:
    """Compare results across system sizes for the same model/delta."""
    print(f"\n{'═' * 90}")
    print("  CROSS-N SCALING COMPARISON")
    print(f"{'═' * 90}")

    # Group by (model, delta, topology)
    configs: dict[tuple, list[dict]] = {}
    for r in results:
        key = (r["model"], r["delta"], r["topology"], r["p_layers"])
        configs.setdefault(key, []).append(r)

    for key, group in sorted(configs.items()):
        n_values = sorted(set(r["n_qubits"] for r in group))
        if len(n_values) < 2:
            continue

        model, delta, topo, p = key
        d_str = f"Δ={delta:.1f}" if delta is not None else ""
        print(f"\n  {model} {d_str} on {topo} (p={p}):")
        print(
            f"  {'N':<5} {'MaxFid':<10} {'MaxS':<8} {'E_exact[0]':<12} "
            f"{'E_vqe[0]':<12} {'Classification'}"
        )
        print(f"  {'-' * 70}")

        for n in n_values:
            r = next(x for x in group if x["n_qubits"] == n)
            s_str = f"{r['max_entropy']:.3f}" if r["max_entropy"] is not None else "—"
            e_exact = f"{r['ground_energies'][0]:.3f}" if r.get("ground_energies") else "—"
            e_vqe = f"{r['vqe_energies'][0]:.3f}" if r.get("vqe_energies") else "—"
            print(
                f"  {n:<5} {r['max_fidelity']:<10.6f} {s_str:<8} {e_exact:<12} "
                f"{e_vqe:<12} {r['classification']}"
            )

    # TFIM comparison
    tfim = [r for r in results if r["model"] == "tfim"]
    if tfim:
        print("\n  TFIM Baseline (same h-range, same pipeline):")
        print(f"  {'N':<5} {'MaxFid':<10} {'ΔE/gap':<10} {'Classification'}")
        print(f"  {'-' * 50}")
        for r in sorted(tfim, key=lambda x: x["n_qubits"]):
            de_str = f"{r['delta_e_over_gap']:.4f}" if r.get("delta_e_over_gap") else "—"
            print(
                f"  {r['n_qubits']:<5} {r['max_fidelity']:<10.6f} {de_str:<10} "
                f"{r['classification']}"
            )


def export_json(results: list[dict], path: Path) -> None:
    """Export results to JSON (without large arrays)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    from collections import Counter

    output = {
        "tool": "analysis/heisenberg_summary.py",
        "n_results": len(results),
        "by_n": {},
        "by_classification": dict(Counter(r["classification"] for r in results)),
        "results": [],
    }

    by_n: dict[int, list] = {}
    for r in results:
        by_n.setdefault(r["n_qubits"], []).append(r)
    for n, group in sorted(by_n.items()):
        output["by_n"][str(n)] = {
            "count": len(group),
            "classifications": dict(Counter(g["classification"] for g in group)),
        }

    for r in results:
        # Exclude large arrays from export
        entry = {
            k: v
            for k, v in r.items()
            if k not in ("ground_energies", "vqe_energies", "h_values", "per_h_fidelity")
        }
        output["results"].append(entry)

    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  📄 Exported {len(results)} results to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Heisenberg XXZ experiment summary (reuses ResultScanner)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--n-qubits", type=int, default=None, help="Filter by system size")
    parser.add_argument("--compare-scaling", action="store_true", help="Cross-N comparison")
    parser.add_argument("--json", type=str, default=None, help="Export to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-h checkpoint data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    # Discover Heisenberg folders
    folders = find_heisenberg_folders()
    if not folders:
        print("No Heisenberg result folders found in results/thesis/")
        sys.exit(1)

    # Filter by N if requested
    if args.n_qubits:
        folders = [f for f in folders if f"N{args.n_qubits}" in f.name]

    print("=" * 90)
    print("  HEISENBERG XXZ EXPERIMENT SUMMARY")
    print("=" * 90)
    print(f"  Folders: {[f.name for f in folders]}")

    # Enrich with Heisenberg-specific data
    all_results = []
    for folder in folders:
        all_results.extend(enrich_with_heisenberg_data(folder))

    print(f"  Total variants: {len(all_results)}")

    # Print summary table
    print_summary(all_results, verbose=args.verbose)

    # Cross-N comparison
    if args.compare_scaling or len(folders) > 1:
        print_scaling_comparison(all_results)

    # Export
    if args.json:
        export_json(all_results, Path(args.json))


if __name__ == "__main__":
    main()
