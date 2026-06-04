#!/usr/bin/env python3
"""Run E4b (TFIM+longitudinal) and E4c (TFIM frustrated) using standard framework.

Uses BaseExperiment.execute() for standardized result saving that integrates
with the digest/compare/analysis tools.

Usage:
    python scripts/run_e4b_e4c_standard.py              # Run both
    python scripts/run_e4b_e4c_standard.py --exp E4b    # Run E4b only
    python scripts/run_e4b_e4c_standard.py --exp E4c    # Run E4c only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_e4b():
    """Run E4b: TFIM + Longitudinal Field with Extended HVA."""
    from experiments.generalization.exp_e4b_longitudinal_hva_extended import ExperimentE4b

    exp = ExperimentE4b(ExperimentE4b.default_config())
    analysis = exp.execute()
    return analysis


def run_e4c():
    """Run E4c: Frustrated TFIM (J1-J2) with NNN HVA."""
    from experiments.generalization.exp_e4c_frustrated_tfim import ExperimentE4c

    exp = ExperimentE4c(ExperimentE4c.default_config())
    analysis = exp.execute()
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Run Hamiltonian extension experiments")
    parser.add_argument("--exp", choices=["E4b", "E4c", "both"], default="both")
    args = parser.parse_args()

    if args.exp in ("E4b", "both"):
        print("\n" + "=" * 65)
        print("Running E4b: TFIM + Longitudinal Field")
        print("=" * 65)
        run_e4b()

    if args.exp in ("E4c", "both"):
        print("\n" + "=" * 65)
        print("Running E4c: Frustrated TFIM (J1-J2)")
        print("=" * 65)
        run_e4c()


if __name__ == "__main__":
    main()
