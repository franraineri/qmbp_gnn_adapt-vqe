#!/usr/bin/env python3
"""MOVED: This script now lives at project_health/analysis/verify_claims.py.

This shim exists for backward compatibility with existing invocations.
"""

import runpy
import sys
from pathlib import Path

# Redirect to the new location
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

target = ROOT / "project_health" / "analysis" / "verify_claims.py"
runpy.run_path(str(target), run_name="__main__")
