#!/usr/bin/env python3
"""Import Colab ground truth results into the local GroundTruthCache.

After running colab_worker.py remotely and fetching results, use this script
to populate the local GT cache so subsequent local runs don't need DMRG.

Usage:
    .venv/bin/python scripts/remote/import_colab_gt.py results/colab/heavy_hex_n20_gt_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <colab_results.json>")
        return 1

    results_path = Path(sys.argv[1])
    if not results_path.exists():
        print(f"ERROR: File not found: {results_path}")
        return 1

    with open(results_path) as f:
        data = json.load(f)

    task_type = data.get("task_type", "")
    if task_type != "dmrg":
        print(f"ERROR: Expected task_type='dmrg', got '{task_type}'")
        return 1

    metadata = data.get("metadata", {})
    results = data.get("results", [])
    n_success = data.get("n_success", 0)
    n_fail = data.get("n_fail", 0)

    print(
        f"Importing {n_success} ground truth results (topology={metadata.get('topology')}, "
        f"N={metadata.get('n_qubits')}, model={metadata.get('model')})"
    )
    if n_fail > 0:
        print(f"  ⚠️ {n_fail} failed tasks (skipped)")

    from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

    cache = GroundTruthCache()
    n_imported = 0

    topology = metadata.get("topology", "")
    n_qubits = metadata.get("n_qubits", 0)
    model = metadata.get("model", "tfim_bond_resolved")

    for r in results:
        if r.get("status") != "ok":
            continue
        h = r["h"]
        energy = r["energy"]
        gap = r["gap"]
        gap_method = r.get("gap_method", "unknown")
        mag_x = r.get("mag_x")
        corr_zz = r.get("corr_zz")

        # Check if already in cache
        existing = cache.get(topology, n_qubits, model, h)
        if existing:
            e_diff = abs(existing["energy"] - energy)
            if e_diff < 1e-6:
                continue  # Already have this exact value
            print(f"  h={h:.4f}: updating (old E={existing['energy']:.8f} → new E={energy:.8f})")

        cache.put(
            topology=topology,
            n_qubits=n_qubits,
            model=model,
            h=h,
            energy=energy,
            gap=gap,
            method=gap_method,
            mag_x=mag_x,
            corr_zz=corr_zz,
        )
        n_imported += 1

    cache.flush()
    print(f"\n  ✅ Imported {n_imported} entries into GroundTruthCache")
    print(f"     Cache now has {len(cache)} total entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
