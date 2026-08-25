#!/usr/bin/env python
"""Backfill time_s into existing GroundTruthCache entries.

One-shot utility: re-computes each (topology, N, model, h) entry and
stores the measured wall-time. This is safe because get_or_compute
uses the same solver pipeline and will produce identical results.

The script is designed for efficiency:
- Batches by (topology, N, model) to reuse the same solver instance.
- Skips entries that already have time_s > 0.
- Flushes every 20 entries (crash-safe).

Usage:
    # Backfill all entries
    python scripts/analysis/backfill_gt_timing.py

    # Dry-run (show what would be computed)
    python scripts/analysis/backfill_gt_timing.py --dry-run

    # Only specific topology
    python scripts/analysis/backfill_gt_timing.py --topology chain_1d
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Backfill time_s in GT cache")
    parser.add_argument("--topology", type=str, default=None, help="Filter by topology")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

    cache = GroundTruthCache()

    # Find entries missing time_s
    missing = []
    for key, entry in cache._data.items():
        if entry.get("time_s") and entry["time_s"] > 0:
            continue
        parts = key.split("|")
        if len(parts) < 4:
            continue
        topo = parts[0]
        if args.topology and topo != args.topology:
            continue
        try:
            n = int(parts[1])
            model = parts[2]
            h = float(parts[3])
        except (ValueError, IndexError):
            continue
        missing.append((topo, n, model, h, key))

    # Group by (topo, n, model) for efficient batching
    batches: dict[tuple, list] = defaultdict(list)
    for topo, n, model, h, key in missing:
        batches[(topo, n, model)].append((h, key))

    total = len(missing)
    already_timed = sum(1 for e in cache._data.values() if e.get("time_s", 0) > 0)

    print(f"\n  GT Cache: {len(cache._data)} total entries")
    print(f"  Already timed: {already_timed}")
    print(f"  Missing time_s: {total}")
    print(f"  Batches: {len(batches)} (topology, N, model) groups")

    if args.dry_run:
        print("\n  DRY RUN — would compute:")
        for (topo, n, model), items in sorted(batches.items()):
            print(f"    {topo} N={n} {model}: {len(items)} h-points")
        return

    if total == 0:
        print("  Nothing to backfill!")
        return

    # Execute backfill
    from qmbp_simulation.models.hamiltonian import make_lattice
    from qmbp_simulation.models.model_registry import get_model_spec
    from qmbp_simulation.solvers.classical import ClassicalSolver

    solver = ClassicalSolver()
    done = 0
    t_start = time.time()

    for (topo, n, model), items in sorted(batches.items()):
        spec = get_model_spec(model)

        for h, key in sorted(items):
            t0 = time.perf_counter()
            try:
                lattice = make_lattice(topo, n, J=1.0, h=h)
                H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
                gt = solver.solve(H, lattice)
                elapsed = time.perf_counter() - t0

                # Update in-place with timing
                entry = cache._data[key]
                entry["time_s"] = round(elapsed, 3)
                # Also update energy/gap in case solver improved
                entry["energy"] = float(gt.ground_energy)
                entry["gap"] = float(gt.gap)
                cache._dirty = True
            except Exception as e:
                logger.warning(f"  FAILED {key}: {e}")
                continue

            done += 1
            if done % 20 == 0:
                cache._save()
                eta = (time.time() - t_start) / done * (total - done)
                print(f"  [{done}/{total}] {topo} N={n} h={h:.2f}: "
                      f"{elapsed:.3f}s (ETA: {eta:.0f}s)")

    # Final flush
    cache.flush()
    elapsed_total = time.time() - t_start
    print(f"\n  Done: {done}/{total} entries backfilled in {elapsed_total:.1f}s")


if __name__ == "__main__":
    main()
