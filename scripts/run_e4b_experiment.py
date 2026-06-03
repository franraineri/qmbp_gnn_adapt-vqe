#!/usr/bin/env python3
"""Run E4b experiment: TFIM + Longitudinal Field with Extended HVA.

Executes all 3 seeds and produces full analysis report.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is in path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from experiments.generalization.exp_e4b_longitudinal_hva_extended import ExperimentE4b

exp = ExperimentE4b(ExperimentE4b.default_config())
exp.setup()

# Run all seeds
results = {}
for seed in exp.config.seeds:
    print(f"\n--- Running seed {seed} ---")
    metrics = exp.run_single(seed)
    results[seed] = metrics
    print(f"  -> {len(metrics)} data points collected")

# Analyze
analysis = exp.analyze(results)

# Report
report = exp.report(analysis)
print("\n")
print(report)
