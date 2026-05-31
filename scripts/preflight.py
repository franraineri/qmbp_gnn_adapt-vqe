#!/usr/bin/env python3
"""Convenience wrapper for preflight checks.

Usage:
    python scripts/preflight.py --from-script scripts/experiment_runners/run_p1_pipeline_variants_r2.py
    python scripts/preflight.py --from-script scripts/experiment_runners/run_p1_pipeline_variants.py --strict
    python scripts/preflight.py --from-json variants.json
    python scripts/preflight.py --help
"""

from qmbp_simulation.framework.preflight import main

if __name__ == "__main__":
    main()
