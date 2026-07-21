#!/usr/bin/env python3
"""Check what (N, p) combinations have h_expr data and identify gaps.

h_expr = lowest h tested that still passes ΔE/gap < 5% (coarse lower bound).

Usage:
    python scripts/analysis/check_matrix_gaps.py
    python scripts/analysis/check_matrix_gaps.py --target-ns 20 30 40 50 60 80 100 120
    python scripts/analysis/check_matrix_gaps.py --target-ps 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "results" / "experiments" / "exp_scaling" / "validation" / "tfim" / "chain_1d"
DEFAULT_TARGET_NS = [20, 30, 40, 50, 60, 80, 100, 120]
DEFAULT_TARGET_PS = [1, 2, 3, 4]
THRESHOLD = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description="Check h_expr matrix gaps")
    parser.add_argument("--dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--target-ns", type=int, nargs="+", default=DEFAULT_TARGET_NS)
    parser.add_argument("--target-ps", type=int, nargs="+", default=DEFAULT_TARGET_PS)
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    return parser.parse_args()


def main():
    args = parse_args()
    base = args.dir

    if not base.exists():
        print(f"ERROR: Directory not found: {base}", file=sys.stderr)
        sys.exit(1)

    files = sorted(base.glob("run_*.json"))
    if not files:
        print(f"ERROR: No result files found in {base}", file=sys.stderr)
        sys.exit(1)

    matrix: dict[tuple[int, int], float] = {}
    n_files_parsed = 0
    n_files_skipped = 0

    for f in files:
        try:
            with open(f) as fp:
                d = json.load(fp)
        except (json.JSONDecodeError, OSError):
            n_files_skipped += 1
            continue

        cfg = d.get("config", {}).get("system", {})
        n = cfg.get("n_qubits")
        p = cfg.get("p_layers")
        if not n or not p:
            n_files_skipped += 1
            continue

        s2 = d.get("results", {}).get("section_2", {}).get("data", {})
        if not s2 or "per_seed" not in s2:
            n_files_skipped += 1
            continue

        n_files_parsed += 1

        # Use first seed (consistent — all seeds have same h-grid)
        pts = s2["per_seed"][0]["results"]
        pass_h = [r["h"] for r in pts if r.get("de_gap", 1.0) < THRESHOLD]
        if not pass_h:
            continue

        frontier = min(pass_h)
        key = (n, p)
        # Keep lowest frontier (most aggressive h-range tested)
        if key not in matrix or frontier < matrix[key]:
            matrix[key] = frontier

    if args.json:
        out = [{"n": k[0], "p": k[1], "h_expr": v} for k, v in sorted(matrix.items())]
        print(
            json.dumps(
                {"data": out, "files_parsed": n_files_parsed, "files_skipped": n_files_skipped},
                indent=2,
            )
        )
        return

    # Print matrix
    ns = sorted(set(k[0] for k in matrix))
    ps = sorted(set(k[1] for k in matrix))

    print(f"h_expr(N, p) — lowest h where ΔE/gap < {THRESHOLD * 100:.0f}%")
    print(f"Source: {base} ({n_files_parsed} files parsed, {n_files_skipped} skipped)\n")

    header = f"{'N':>5} " + "  ".join(f"{'p=' + str(p):>6}" for p in ps)
    print(header)
    print("-" * (6 + 8 * len(ps)))
    for n in ns:
        row = f"{n:>5} "
        for p in ps:
            val = matrix.get((n, p))
            row += f" {val:6.2f}" if val else "      -"
        print(row)

    # Identify gaps
    print(f"\n\nGAPS (target N={args.target_ns}, p={args.target_ps}):")
    total_gaps = 0
    for p in args.target_ps:
        missing = [n for n in args.target_ns if (n, p) not in matrix]
        have = [n for n in args.target_ns if (n, p) in matrix]
        total_gaps += len(missing)
        if missing:
            print(f"  p={p}: HAVE N={have}, MISSING N={missing}")
        else:
            print(f"  p={p}: COMPLETE")
    print(f"\n  Total gaps: {total_gaps} cells")


if __name__ == "__main__":
    main()
