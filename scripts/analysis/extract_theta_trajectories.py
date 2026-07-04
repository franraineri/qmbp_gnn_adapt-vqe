#!/usr/bin/env python3
"""Task 2.1: Extract θ_opt(h) trajectories from ALL pipeline/scaling results.

Scans:
  - results/thesis/ for pipeline_run_*.json and noisy_3mode_*.json files
  - results/scaling/scaling_N*.json for MPS VQE descending sweeps (N=40-200)
  - results/scaling/scaling_N120_full_sweep.json for N=120 rigorous sweep

Groups by (topology, n_qubits, p_layers, seed) and saves to
analysis/raw_data/theta_trajectories.json.

Usage:
    python scripts/analysis/extract_theta_trajectories.py
    python scripts/analysis/extract_theta_trajectories.py --include-scaling
    python scripts/analysis/extract_theta_trajectories.py --only-scaling
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "results" / "thesis"
SCALING_DIR = ROOT / "results" / "scaling"
OUTPUT_FILE = ROOT / "results" / "analysis" / "raw_data" / "theta_trajectories.json"


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
        for s in DEFAULT_SEEDS:
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


def scan_scaling_results() -> list[dict]:
    """Scan results/scaling/ for θ_opt trajectories from MPS-VQE runs.

    Handles two formats:
    1. scaling_N*_*.json (run_scaling_validation.py output):
       vqe_results[seed_idx].results[h_idx].theta_opt
    2. scaling_N120_full_sweep.json (N=120 sweep):
       per_point[i].theta_opt (sorted by seed then h descending)

    Returns list of trajectory dicts compatible with theta_trajectories.json schema.
    """
    if not SCALING_DIR.exists():
        return []

    trajectories: list[dict] = []
    seen_keys: set[str] = set()

    # ── Format 1: scaling_N*_*.json (run_scaling_validation.py) ──────
    for filepath in sorted(SCALING_DIR.glob("scaling_N*_*.json")):
        if "full_sweep" in filepath.name:
            continue  # Handle separately below
        try:
            with open(filepath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if data.get("experiment") != "mps_scaling_validation":
            continue

        meta = data.get("metadata", {})
        n_qubits = meta.get("n", 0)
        topology = meta.get("topology", "chain_1d")
        p_layers = meta.get("p_layers", 1)
        vqe_results = data.get("vqe_results", [])

        for seed_run in vqe_results:
            seed = seed_run.get("seed", 42)
            results = seed_run.get("results", [])

            # Only include if theta_opt is present
            valid_points = [r for r in results if r.get("theta_opt") and len(r["theta_opt"]) > 0]
            if len(valid_points) < 3:
                continue

            h_values = [p["h"] for p in valid_points]
            theta_opt = [p["theta_opt"] for p in valid_points]

            key = f"{topology}_{n_qubits}_p{p_layers}_s{seed}_{len(valid_points)}_{h_values[0]:.2f}"
            if key in seen_keys:
                continue
            seen_keys.add(key)

            trajectories.append(
                {
                    "topology": topology,
                    "n_qubits": n_qubits,
                    "p_layers": p_layers,
                    "seed": seed,
                    "h_values": h_values,
                    "theta_opt": theta_opt,
                    "n_points": len(valid_points),
                    "n_params": len(theta_opt[0]),
                    "mean_fidelity": None,
                    "source_file": str(filepath.relative_to(ROOT)),
                    "source_type": "scaling_validation",
                }
            )

    # ── Format 2: scaling_N120_full_sweep.json ───────────────────────
    sweep_file = SCALING_DIR / "scaling_N120_full_sweep.json"
    if sweep_file.exists():
        try:
            with open(sweep_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = None

        if data and data.get("experiment") == "N120_full_sweep":
            n_qubits = data.get("n_qubits", 120)
            data.get("seeds", DEFAULT_SEEDS)
            per_point = data.get("per_point", [])

            # Group by seed
            from collections import defaultdict

            by_seed: dict[int, list[dict]] = defaultdict(list)
            for pt in per_point:
                if pt.get("theta_opt") and len(pt["theta_opt"]) > 0:
                    by_seed[pt["seed"]].append(pt)

            for seed, points in by_seed.items():
                if len(points) < 3:
                    continue
                # Sort by h descending (matches sweep order)
                points.sort(key=lambda p: p["h"], reverse=True)
                h_values = [p["h"] for p in points]
                theta_opt = [p["theta_opt"] for p in points]

                key = f"chain_1d_{n_qubits}_p1_s{seed}_{len(points)}_{h_values[0]:.2f}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                trajectories.append(
                    {
                        "topology": "chain_1d",
                        "n_qubits": n_qubits,
                        "p_layers": 1,
                        "seed": seed,
                        "h_values": h_values,
                        "theta_opt": theta_opt,
                        "n_points": len(points),
                        "n_params": len(theta_opt[0]),
                        "mean_fidelity": None,
                        "source_file": str(sweep_file.relative_to(ROOT)),
                        "source_type": "n120_full_sweep",
                    }
                )

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
    parser = argparse.ArgumentParser(description="Extract θ_opt(h) trajectories")
    parser.add_argument(
        "--include-scaling",
        action="store_true",
        default=True,
        help="Include MPS scaling results (N=40-200). Default: True.",
    )
    parser.add_argument(
        "--only-scaling",
        action="store_true",
        help="Only scan scaling results (skip thesis/ pipeline data).",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Task 2.1: Extract θ_opt(h) trajectories")
    logger.info("=" * 60)

    all_trajectories: list[dict] = []

    # Scan thesis pipeline results (unless --only-scaling)
    if not args.only_scaling:
        logger.info(f"\nScanning thesis data: {RESULTS_DIR}")
        thesis_trajs = scan_results()
        logger.info(f"  Found {len(thesis_trajs)} thesis trajectories")
        all_trajectories.extend(thesis_trajs)

    # Scan scaling results (always unless disabled)
    if args.include_scaling or args.only_scaling:
        logger.info(f"\nScanning scaling data: {SCALING_DIR}")
        scaling_trajs = scan_scaling_results()
        logger.info(f"  Found {len(scaling_trajs)} scaling trajectories")
        all_trajectories.extend(scaling_trajs)

    logger.info(f"\n  Total raw: {len(all_trajectories)} trajectories")

    # Filter to best per config
    best = filter_best_trajectories(all_trajectories)
    logger.info(f"  After dedup: {len(best)} unique trajectories")

    # Summary by topology
    from collections import Counter

    topo_counts = Counter(t["topology"] for t in best)
    logger.info("\n  Per-topology breakdown:")
    for topo, count in sorted(topo_counts.items()):
        logger.info(f"    {topo}: {count} trajectories")

    # Summary by (topology, n_qubits, p_layers)
    config_counts = Counter((t["topology"], t["n_qubits"], t["p_layers"]) for t in best)
    logger.info("\n  Per-config breakdown:")
    for (topo, nq, p), count in sorted(config_counts.items()):
        logger.info(f"    {topo} N={nq} p={p}: {count} seeds")

    # Summary by source type
    source_types = Counter(t.get("source_type", "thesis_pipeline") for t in best)
    logger.info("\n  Per-source breakdown:")
    for src, count in sorted(source_types.items()):
        logger.info(f"    {src}: {count}")

    # Save output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "n_trajectories": len(best),
            "topologies": sorted(set(t["topology"] for t in best)),
            "n_values": sorted(set(t["n_qubits"] for t in best)),
            "source_dirs": [
                str(RESULTS_DIR.relative_to(ROOT)),
                str(SCALING_DIR.relative_to(ROOT)),
            ],
            "includes_scaling": args.include_scaling or args.only_scaling,
        },
        "trajectories": best,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n  Saved to: {OUTPUT_FILE.relative_to(ROOT)}")
    logger.info("  Done.")


if __name__ == "__main__":
    main()
