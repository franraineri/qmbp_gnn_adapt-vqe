#!/usr/bin/env python3
"""Convenience entry point for the digest tool.

Usage:
    python scripts/run_digest.py [options]

Equivalent to:
    python -m scripts.digest [options]

See `python -m scripts.digest --help` for full usage.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `scripts.digest` resolves
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.digest.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
