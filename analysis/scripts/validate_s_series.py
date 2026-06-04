#!/usr/bin/env python3
"""MOVED: This script now lives at project_health/analysis/validate_s_series.py.

This shim exists for backward compatibility with existing invocations.
"""

import sys
from pathlib import Path

# Redirect to the new location
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from project_health.analysis.validate_s_series import *  # noqa: E402, F401, F403
from project_health.analysis.validate_s_series import main

if __name__ == "__main__":
    main()
