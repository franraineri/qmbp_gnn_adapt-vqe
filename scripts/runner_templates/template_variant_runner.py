#!/usr/bin/env python3
"""Template: Pipeline variant runner with preflight.

Use this when running multiple pipeline configurations (topologies × seeds × params)
that each invoke run_pipeline.py as a subprocess.

Guarantees:
- Preflight validates ALL variants before any execution starts.
- Preflight uses the ACTUAL n_qubits (from --n-qubits or default).
- --skip-preflight bypasses validation if needed.
- Execution, timing, and logging handled by VariantRunner infrastructure.
- Results saved as execution_log_{timestamp}.json in the output base dir.

Usage:
    .venv/bin/python scripts/my_variants.py
    .venv/bin/python scripts/my_variants.py --dry-run
    .venv/bin/python scripts/my_variants.py --list
    .venv/bin/python scripts/my_variants.py --variant 0
    .venv/bin/python scripts/my_variants.py --noiseless-only
    .venv/bin/python scripts/my_variants.py --n-qubits 6
    .venv/bin/python scripts/my_variants.py --skip-preflight
"""

from __future__ import annotations

import sys

from qmbp_simulation.framework.runner_base import resolve_project_root

# ─── Project root setup (works from any script depth) ────────────────────────
_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
