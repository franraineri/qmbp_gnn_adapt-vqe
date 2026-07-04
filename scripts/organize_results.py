#!/usr/bin/env python3
"""Organize experiment results into the hierarchical folder structure.

Scans flat legacy directories (exp_noiseless_tfim_4/, exp_noiseless_tfim_longitudinal_4/,
etc.) and creates symlinks or copies into the new nested structure:

    results/experiments/exp_noiseless/{model}/{topology}/run_*.json

Also archives deprecated/conclusively-failed experiments (heisenberg) to:
    results/archive/{original_folder_name}/

Usage:
    # Dry run (show what would happen):
    python scripts/organize_results.py --dry-run

    # Create symlinks (non-destructive, originals untouched):
    python scripts/organize_results.py --mode symlink

    # Move files (destructive — moves originals):
    python scripts/organize_results.py --mode move

    # Archive failed experiments:
    python scripts/organize_results.py --archive-failed
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results" / "experiments"
ARCHIVE_ROOT = ROOT / "results" / "archive"

# Models that are conclusively inviable (0% pass across all configs)
ARCHIVE_MODELS = {"heisenberg", "heisenberg_transverse"}


def extract_config_from_file(path: Path) -> dict | None:
    """Extract model/topology from a result file without full parsing."""
    try:
        with open(path) as f:
            data = json.load(f)
        config = data.get("config", {})
        system = config.get("system", {})
        model = system.get("model", "")
        topos = system.get("topologies", [])
        topology = topos[0] if isinstance(topos, list) and topos else ""
        return {"model": model, "topology": topology}
    except (json.JSONDecodeError, OSError, KeyError):
        return None


def organize_flat_dirs(mode: str = "dry-run") -> list[str]:
    """Organize flat exp_noiseless_* dirs into nested structure."""
    actions: list[str] = []

    # Find flat noiseless dirs
    flat_dirs = [
        d for d in sorted(RESULTS_ROOT.iterdir())
        if d.is_dir() and d.name.startswith("exp_noiseless_") and "/" not in d.name
    ]

    for flat_dir in flat_dirs:
        for run_file in sorted(flat_dir.glob("run_*.json")):
            cfg = extract_config_from_file(run_file)
            if not cfg or not cfg["model"] or not cfg["topology"]:
                continue

            # Target: exp_noiseless/{model}/{topology}/run_*.json
            target_dir = RESULTS_ROOT / "exp_noiseless" / cfg["model"] / cfg["topology"]
            target_path = target_dir / run_file.name

            if target_path.exists():
                continue  # Already organized

            action = f"{run_file.relative_to(RESULTS_ROOT)} → {target_path.relative_to(RESULTS_ROOT)}"
            actions.append(action)

            if mode == "symlink":
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path.symlink_to(run_file.resolve())
            elif mode == "move":
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(run_file), str(target_path))

    return actions


def archive_failed(mode: str = "dry-run") -> list[str]:
    """Move conclusively failed model results to archive."""
    actions: list[str] = []

    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        # Check if this dir contains only failed models
        dir_lower = d.name.lower()
        should_archive = any(m in dir_lower for m in ARCHIVE_MODELS)
        if not should_archive:
            continue

        target = ARCHIVE_ROOT / d.name
        action = f"ARCHIVE: {d.name} → results/archive/{d.name}"
        actions.append(action)

        if mode == "move":
            ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.move(str(d), str(target))

    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description="Organize experiment results.")
    parser.add_argument(
        "--mode", choices=["dry-run", "symlink", "move"], default="dry-run",
        help="Action mode: dry-run (default), symlink, or move",
    )
    parser.add_argument(
        "--archive-failed", action="store_true",
        help="Archive conclusively failed experiments (heisenberg)",
    )
    args = parser.parse_args()

    print(f"Mode: {args.mode}")
    print()

    # Organize flat dirs
    actions = organize_flat_dirs(args.mode)
    if actions:
        print(f"=== Organize ({len(actions)} files) ===")
        for a in actions[:20]:
            print(f"  {a}")
        if len(actions) > 20:
            print(f"  ... and {len(actions) - 20} more")
    else:
        print("  No files to organize (already done or no flat dirs found)")

    # Archive
    if args.archive_failed:
        print()
        archive_actions = archive_failed(args.mode)
        if archive_actions:
            print(f"=== Archive ({len(archive_actions)} dirs) ===")
            for a in archive_actions:
                print(f"  {a}")
        else:
            print("  No dirs to archive")

    if args.mode == "dry-run":
        print()
        print("(dry-run — no changes made. Use --mode symlink or --mode move to apply)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
