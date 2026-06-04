#!/usr/bin/env python3
"""Task 2.1: Extract θ_opt(h) trajectories from existing pipeline results.

Scans results/thesis/ for pipeline_run_*.json and noisy_3mode_*.json files
containing phase12_data with theta_opt arrays.

Groups by (topology, n_qubits, p_layers, seed) and saves to
analysis/raw_data/theta_trajectories.json.

Usage:
    python scripts/analysis/extract_theta_trajectories.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "results" / "thesis"
OUTPUT_FILE = ROOT / "analysis" / "raw_data" / "theta_trajectories.json"


def extract_trajectory(filepath: Path) -> dict | None:
    """Extract a single θ_opt trajectory from a pipeline result JSON.

    Returns None if file lacks phase12_data or theta_opt.
    """
    try:
        with open(filepath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # Determine data location: phase12_data (standard) or phase2_data (variant)
    phase_data = data.get("phase12_data") or data.get("phase2_data")
    if not phase_data:
        return None

    # Check that theta_opt exists and is non-empty
    valid_points = [p for p in phase_data if p.get("theta_opt") and len(p["theta_opt"]) > 0]
    if len(valid_points) < 3:
        return None

    # Extract system metadata
    system = data.get("system", {})
    config = data.get("config", {})

    topology = system.get("topology") or config.get("topology", "unknown")
    n_qubits = system.get("n_qubits") or config.get("n_qubits", 0)
    p_layers = system.get("p_layers") or config.get("p_layers", 0)

    # Infer seed from config or filename
    seed = config.get("seed", None)
    if seed is None:
        # Try to infer from parent directory name
        dirname = filepath.parent.name
        for s in [42, 43, 44]:
            if f"seed_{s}" in dirname or f"seed{s}" in dirname:
                seed = s
                break
        if seed is None:
            seed = 42  # default

    h_values = [p["h"] for p in valid_points]
    theta_opt = [p["theta_opt"] for p in valid_points]

    # Additional metadata
    fidelities = [p.get("fidelity") for p in valid_points if p.get("fidelity")]

    return {
        "topology": topology,
        "n_qubits": n_qubits,
        "p_layers": p_layers,
        "seed": seed,
        "h_values": h_values,
        "theta_opt": theta_opt,
        "n_points": len(valid_points),
        "n_params": len(theta_opt[0]),
        "mean_fidelity": sum(fidelities) / len(fidelities) if fidelities else None,
        "source_file": str(filepath.relative_to(ROOT)),
    }


def scan_results() -> list[dict]:
    """Scan all thesis result files for theta trajectories."""
    trajectories: list[dict] = []
    seen_keys: set[str] = set()

    # Scan pipeline_run_*.json and noisy_3mode_*.json files
    patterns = ["**/pipeline_run_*.json", "**/noisy_3mode_*.json"]

    for pattern in patterns:
        for filepath in sorted(RESULTS_DIR.rglob(pattern.split("/")[-1])):
            traj = extract_trajectory(filepath)
            if traj is None:
                continue

            # Dedup by (topology, n_qubits, p_layers, seed, n_points, first_h)
            key = (
                f"{traj['topology']}_{traj['n_qubits']}_p{traj['p_layers']}"
                f"_s{traj['seed']}_{traj['n_points']}_{traj['h_values'][0]:.2f}"
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            trajectories.append(traj)

    return trajectories


def filter_best_trajectories(trajectories: list[dict]) -> list[dict]:
    """Keep only the best trajectory per (topology, n_qubits, p_layers, seed).

    Preference: most h-points > higher mean fidelity.
    """
    from collections import defaultdict

    groups: dict[str, list[dict]] = defaultdict(list)
    for t in trajectories:
        key = f"{t['topology']}_{t['n_qubits']}_p{t['p_layers']}_s{t['seed']}"
        groups[key].append(t)

    best: list[dict] = []
    for _key, group in groups.items():
        # Sort: more h-points first, then higher fidelity
        group.sort(
            key=lambda x: (x["n_points"], x.get("mean_fidelity") or 0),
            reverse=True,
        )
        best.append(group[0])

    return best


def main() -> None:
    """Extract, filter, and save theta trajectories."""
    logger.info("=" * 60)
    logger.info("Task 2.1: Extract θ_opt(h) trajectories")
    logger.info("=" * 60)

    logger.info(f"\nScanning: {RESULTS_DIR}")
    all_trajectories = scan_results()
    logger.info(f"  Found {len(all_trajectories)} raw trajectories")

    # Filter to best per config
    best = filter_best_trajectories(all_trajectories)
    logger.info(f"  After dedup: {len(best)} unique trajectories")

    # Summary by topology
    from collections import Counter

    topo_counts = Counter(t["topology"] for t in best)
    logger.info("\n  Per-topology breakdown:")
    for topo, count in sorted(topo_counts.items()):
        logger.info(f"    {topo}: {count} trajectories")

    # Summary by (topology, p_layers)
    config_counts = Counter((t["topology"], t["n_qubits"], t["p_layers"]) for t in best)
    logger.info("\n  Per-config breakdown:")
    for (topo, nq, p), count in sorted(config_counts.items()):
        logger.info(f"    {topo} N={nq} p={p}: {count} seeds")

    # Save output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "n_trajectories": len(best),
            "topologies": sorted(set(t["topology"] for t in best)),
            "source_dir": str(RESULTS_DIR.relative_to(ROOT)),
        },
        "trajectories": best,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n  Saved to: {OUTPUT_FILE.relative_to(ROOT)}")
    logger.info("  Done.")


if __name__ == "__main__":
    main()
