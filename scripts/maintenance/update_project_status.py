#!/usr/bin/env python3
"""Auto-generate .kiro/steering/project-status.md from the result index.

Keeps Kiro's context always up-to-date with: best results per config,
pending experiments, regressions, and compute stats.

NOTE: This is now a thin wrapper around ResultIndex.refresh_status().
Runners auto-call refresh_status() after every save, so manual invocation
is rarely needed. Use `python -m project_health --refresh-status` as
the canonical alternative.

Usage:
    .venv/bin/python scripts/update_project_status.py
    python -m project_health --refresh-status    # equivalent
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qmbp_simulation.framework.result_index import ResultIndex


def main() -> int:
    index = ResultIndex()
    if len(index) == 0:
        index.rebuild()

    path = index.refresh_status()
    if path:
        print(f"Updated: {path}")
        return 0
    else:
        print("ERROR: Could not refresh project status")
        return 1


if __name__ == "__main__":
    sys.exit(main())
