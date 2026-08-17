#!/usr/bin/env python3
"""Re-evaluate zoo models against current NPZ training data.

Computes pass_rate_dual from NPZ files for each topology and updates
the zoo manifest. This resolves zoo↔dashboard divergence without
retraining — just syncs the pass_rate to reflect current data quality.

Zero VQE compute. Only reads existing NPZ data (~1s total).

Usage:
    python scripts/maintenance/reevaluate_zoo_models.py           # evaluate + update
    python scripts/maintenance/reevaluate_zoo_models.py --dry-run # show what would change
    python scripts/maintenance/reevaluate_zoo_models.py --topology heavy_hex
    python scripts/maintenance/reevaluate_zoo_models.py --force   # update even if worse
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def evaluate_npz_quality(topology: str, p_layers: int = 1) -> dict:
    """Compute pass_rate_dual from NPZ training data for a topology.

    Same calculation as the dashboard — counts dual-criterion failures
    across all N values.
    """
    from qmbp_simulation.analysis.metrics import is_point_failure

    npz_dir = ROOT / "data" / "multi_n_training"
    pattern = f"{topology}_N*_p{p_layers}.npz"
    npz_files = sorted(npz_dir.glob(pattern))

    if not npz_files:
        return {"error": f"No NPZ files for {topology} p={p_layers}"}

    total_points = 0
    total_failures = 0
    per_n = {}

    for npz_file in npz_files:
        stem = npz_file.stem
        parts = stem.split("_")
        n_idx = next((i for i, p in enumerate(parts) if p.startswith("N")), None)
        if n_idx is None:
            continue
        n_qubits = int(parts[n_idx][1:])

        data = np.load(npz_file, allow_pickle=True)
        h_values = data["h_values"]
        n_pts = len(h_values)
        if n_pts == 0:
            continue

        e_key = "e_vqe" if "e_vqe" in data else ("energies" if "energies" in data else None)
        if e_key is None or "e_exact" not in data:
            continue

        e_vqe = data[e_key].astype(np.float64)
        e_exact = data["e_exact"].astype(np.float64)
        gaps = data["gaps"].astype(np.float64) if "gaps" in data else np.ones(n_pts)

        n_fail = 0
        for i in range(n_pts):
            abs_err = abs(float(e_vqe[i]) - float(e_exact[i]))
            gap_i = max(float(gaps[i]), 1e-10)
            de_gap = abs_err / gap_i
            if is_point_failure(de_gap=de_gap, abs_error=abs_err):
                n_fail += 1

        total_points += n_pts
        total_failures += n_fail
        per_n[n_qubits] = {
            "n_points": n_pts,
            "n_failures": n_fail,
            "pass_rate_dual": (n_pts - n_fail) / n_pts,
        }

    if total_points == 0:
        return {"error": "No evaluable points"}

    return {
        "topology": topology,
        "total_points": total_points,
        "total_failures": total_failures,
        "pass_rate_dual": (total_points - total_failures) / total_points,
        "per_n": per_n,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--topology", type=str)
    parser.add_argument("--force", action="store_true", help="Update even if new rate is lower")
    args = parser.parse_args()

    from qmbp_simulation.predictors.model_zoo import list_pretrained, update_zoo_pass_rate

    multi_n = list_pretrained(n_qubits=0)
    if args.topology:
        multi_n = [e for e in multi_n if e.topology == args.topology]

    if not multi_n:
        print("No multi-N models to evaluate.")
        return 0

    print(f"{'─' * 60}")
    print(f"  RE-EVALUATING {len(multi_n)} ZOO MODELS vs NPZ DATA")
    print(f"{'─' * 60}")

    results = []
    for entry in multi_n:
        topo = entry.topology
        print(f"\n  {topo}: {entry.checkpoint_file[:50]}")
        print(f"    Current manifest pass_rate: {entry.pass_rate:.0%}")

        result = evaluate_npz_quality(topo, p_layers=entry.p_layers)
        if "error" in result:
            print(f"    ⚠️ {result['error']}")
            continue

        new_rate = result["pass_rate_dual"]
        old_rate = entry.pass_rate
        delta = new_rate - old_rate

        print(f"    NPZ pass_rate_dual: {new_rate:.0%} ({result['total_points']} pts)")
        print(f"    Delta: {delta:+.0%}")

        for n, nr in sorted(result["per_n"].items()):
            icon = (
                "✅"
                if nr["pass_rate_dual"] >= 0.8
                else ("⚠️" if nr["pass_rate_dual"] >= 0.5 else "❌")
            )
            print(f"      N={n:2d}: {nr['pass_rate_dual']:.0%} ({nr['n_points']} pts) {icon}")

        if not args.dry_run:
            only_if_better = not args.force
            updated = update_zoo_pass_rate(
                entry.checkpoint_file,
                new_rate,
                only_if_better=only_if_better,
                add_notes=f"reevaluate: {result['total_points']} pts NPZ eval",
            )
            if updated:
                print(f"    ✅ Updated: {old_rate:.0%} → {new_rate:.0%}")
            else:
                print("    ℹ️ Not updated (current >= new)")
        else:
            action = "WOULD UPDATE" if new_rate > old_rate else "would NOT update"
            print(f"    [DRY-RUN] {action}")

        results.append({"topology": topo, "old": old_rate, "new": new_rate, "delta": delta})

    print(f"\n{'─' * 60}")
    print("  SUMMARY")
    print(f"{'─' * 60}")
    for r in results:
        sym = "→" if r["delta"] != 0 else "="
        print(f"  {r['topology']:12s}: {r['old']:.0%} {sym} {r['new']:.0%} (Δ={r['delta']:+.0%})")

    if args.dry_run:
        print("\n  Run without --dry-run to apply changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
