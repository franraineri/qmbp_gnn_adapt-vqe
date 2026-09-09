#!/usr/bin/env python3
"""Migrate training NPZs into the per-model subdirectory layout.

Since training data was separated by model (data/multi_n_training/{model}/...),
the default model (tfim_bond_resolved) still lives at the root for backward
compatibility. This tool helps keep the layout clean and safe:

MODES
-----
--audit (default): report what's at the root and whether any file looks like it
    belongs to a non-default model (a data-crossing risk). Read-only.

--to-default-subdir: move the root-level default-model NPZs into an explicit
    tfim_bond_resolved/ subdirectory (only do this if you also flip
    DEFAULT_MODEL_NAMESPACE; otherwise the aggregator would stop finding them).

The audit mode is the safe everyday tool. The move mode is opt-in and guarded.

SAFETY
------
- Refuses to run while a writer process (a runner / post_experiment_sync) is
  active, unless --force.
- --dry-run reports moves without touching disk.
- Every move is logged; a manifest of moves is written for rollback.

USAGE
-----
    .venv/bin/python scripts/general_project_maintenance/migrate_npz_by_model.py --audit
    .venv/bin/python scripts/general_project_maintenance/migrate_npz_by_model.py --to-default-subdir --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TRAIN_DIR = ROOT / "data" / "multi_n_training"
EXTRAP_DIR = ROOT / "data" / "large_n_extrapolation"

# Canonical NPZ filename: {topology}_N{n}_p{p}.npz
_NPZ_RE = re.compile(r"^(?P<topo>.+)_N(?P<n>\d+)_p(?P<p>\d+)\.npz$")


def _default_model() -> str:
    from qmbp_simulation.framework.result_io import DEFAULT_MODEL_NAMESPACE

    return DEFAULT_MODEL_NAMESPACE


def _writer_may_be_active() -> list[str]:
    import subprocess

    try:
        out = subprocess.run(["ps", "axo", "pid,command"], capture_output=True, text=True).stdout
    except Exception:
        return []
    return [
        ln.strip()
        for ln in out.splitlines()
        if "run_accelerated_cross_n" in ln.lower() or "post_experiment_sync" in ln.lower()
    ]


def _npz_declared_model(npz_path: Path) -> str | None:
    """Read the 'model' field a runner stamps into the NPZ payload, if present."""
    import numpy as np

    try:
        data = np.load(str(npz_path), allow_pickle=True)
    except Exception:
        return None
    if "model" in data:
        try:
            val = data["model"]
            return str(val.item() if hasattr(val, "item") else val)
        except Exception:
            return None
    # J2 field present and nonzero → frustrated variant of the default model.
    if "J2" in data:
        try:
            j2 = float(np.asarray(data["J2"]).ravel()[0])
            if abs(j2) > 1e-15:
                return f"{_default_model()} (frustrated, J2={j2})"
        except Exception:
            pass
    return None


def audit() -> int:
    default = _default_model()
    print("=" * 70)
    print(f"NPZ layout audit — default model at root: {default}")
    print("=" * 70)

    for label, base in (("multi_n_training", TRAIN_DIR), ("large_n_extrapolation", EXTRAP_DIR)):
        if not base.exists():
            continue
        root_files = sorted(f for f in base.glob("*.npz"))
        subdirs = sorted(d for d in base.iterdir() if d.is_dir())
        print(f"\n[{label}]")
        print(f"  Root-level NPZs (treated as '{default}'): {len(root_files)}")
        # Flag any root file whose stamped model disagrees with the default.
        risky = []
        for f in root_files:
            declared = _npz_declared_model(f)
            if declared and not declared.startswith(default):
                risky.append((f.name, declared))
        if risky:
            print(f"  ⚠️ {len(risky)} root NPZ(s) declare a NON-default model — data-crossing risk:")
            for name, m in risky[:10]:
                print(f"       {name}  (declares model={m})")
            print("     → These should live under a per-model subdir. Consider moving them.")
        else:
            print("  ✓ No root NPZ declares a non-default model (no crossing).")
        if subdirs:
            print(f"  Per-model / namespace subdirs: {[d.name for d in subdirs]}")
            for d in subdirs:
                n = len(list(d.rglob('*.npz')))
                print(f"       {d.name}/: {n} NPZ(s)")
    print("\n" + "=" * 70)
    return 0


def to_default_subdir(dry_run: bool, force: bool) -> int:
    default = _default_model()
    risky = _writer_may_be_active()
    if risky and not force:
        print("  ❌ A writer process may be active:")
        for r in risky[:5]:
            print(f"     {r[:120]}")
        print("  Stop it first or pass --force.")
        return 2

    moves: list[tuple[str, str]] = []
    for base in (TRAIN_DIR, EXTRAP_DIR):
        if not base.exists():
            continue
        dest = base / default
        for f in sorted(base.glob("*.npz")):
            if _NPZ_RE.match(f.name):
                moves.append((str(f), str(dest / f.name)))

    if not moves:
        print("  Nothing to move — no root-level NPZs found.")
        return 0

    print(f"  {'DRY RUN — ' if dry_run else ''}Moving {len(moves)} NPZ(s) into '{default}/':")
    for src, dst in moves[:10]:
        print(f"    {Path(src).name} -> {default}/{Path(dst).name}")
    if len(moves) > 10:
        print(f"    ... and {len(moves) - 10} more")

    if dry_run:
        print("\n  (--dry-run) No files moved.")
        return 0

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    manifest = ROOT / "data" / f"npz_migration_manifest_{ts}.json"
    for src, dst in moves:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dst)
    manifest.write_text(json.dumps({"moved": moves, "timestamp": ts}, indent=2))
    print(f"\n  ✅ Moved {len(moves)} NPZ(s). Rollback manifest: {manifest.name}")
    print(f"  ⚠️ Remember to flip DEFAULT_MODEL_NAMESPACE only if you intend the")
    print(f"     aggregator to read from '{default}/' instead of the root.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit / migrate training NPZs to per-model layout")
    ap.add_argument("--audit", action="store_true", help="Report layout + crossing risks (default)")
    ap.add_argument(
        "--to-default-subdir",
        action="store_true",
        help="Move root-level default-model NPZs into an explicit {default}/ subdir",
    )
    ap.add_argument("--dry-run", action="store_true", help="Report moves without touching disk")
    ap.add_argument("--force", action="store_true", help="Proceed even if a writer seems active")
    args = ap.parse_args()

    if args.to_default_subdir:
        return to_default_subdir(dry_run=args.dry_run, force=args.force)
    return audit()


if __name__ == "__main__":
    sys.exit(main())
