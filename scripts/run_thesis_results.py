#!/usr/bin/env python
"""GNN-HVA v6.1 — Thesis Results Consolidation.

⚠️  DEPRECATED (2026-05-25): This script references HardwareDeployerV61 which was
removed during the V7 package restructure. The thesis results it produced are
already committed in results/thesis/ and documented in the binnacles.

To reproduce thesis results with the current package, use:
    python scripts/run_experiment.py --exp B4 F3 A3 --verbose

For the full pipeline:
    python scripts/run_pipeline.py --n-qubits 6 --p 2

Original description:
Runs the definitive experiments for Chapter 4 (Results):
  - Table 4.2: N=6, 3 seeds x 3 h_test values = 9 runs
  - Table 4.3: N=10, 3 seeds x 2 h_test values = 6 runs
"""

import sys
import warnings

warnings.warn(
    "run_thesis_results.py is DEPRECATED. It references HardwareDeployerV61 which "
    "was removed in V7. Thesis results are already committed in results/thesis/. "
    "Use scripts/run_experiment.py or scripts/run_pipeline.py instead.",
    DeprecationWarning,
    stacklevel=1,
)
print("ERROR: This script is deprecated. See docstring for alternatives.")
sys.exit(1)
