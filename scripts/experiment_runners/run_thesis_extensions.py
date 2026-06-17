#!/usr/bin/env python3
"""CLI entry point for thesis extension analysis pipeline.

Runs the unified ThesisExtensionAnalyzer and serializes the result to
results/thesis_extensions/analysis_result.json.

Does NOT modify results/thesis/ (Req 4.6).

Usage:
    python scripts/experiment_runners/run_thesis_extensions.py
    python scripts/experiment_runners/run_thesis_extensions.py \\
        --project-root /path/to/project \\
        --phase3-results results/experiments/phase3/run_latest.json \\
        --output-dir results/thesis_extensions/ \\
        --parallel
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# ---------------------------------------------------------------------------
# Ensure the project src/ is on PYTHONPATH when running as a script
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_thesis_extensions")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Thesis Extension Analysis Pipeline — Ext1, Ext2, Ext3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--project-root",
        default=_PROJECT_ROOT,
        help="Root directory of the project (default: auto-detected)",
    )
    parser.add_argument(
        "--phase3-results",
        default=None,
        help="Path to Phase 3 JSON results file (for MC-Dropout L6 baseline)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(_PROJECT_ROOT, "results", "thesis_extensions"),
        help="Output directory for analysis_result.json",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Run Ext1/Ext2/Ext3 analyses concurrently (ThreadPoolExecutor)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Instantiate analyzer and print config, but do not run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logger.info("=== Thesis Extension Analysis Pipeline ===")
    logger.info("Project root : %s", args.project_root)
    logger.info("Phase3 path  : %s", args.phase3_results or "(not set)")
    logger.info("Output dir   : %s", args.output_dir)
    logger.info("Parallel     : %s", args.parallel)

    if args.dry_run:
        logger.info("--dry-run: configuration valid, exiting without running.")
        return 0

    from qmbp_simulation.analysis.extension_analyzer import ThesisExtensionAnalyzer

    analyzer = ThesisExtensionAnalyzer(
        project_root=args.project_root,
        phase3_results_path=args.phase3_results,
        output_dir=args.output_dir,
        parallel=args.parallel,
    )

    result = analyzer.run()

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("EXTENSION ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Run timestamp : {result.run_timestamp}")
    print(f"Priority rank : {' > '.join(result.priority_ranking)}")
    print()
    for ext_id, ext_result in [
        ("Ext1 Bond-Resolved 2D", result.ext1_bond_resolved),
        ("Ext2 Kagomé / QSL", result.ext2_kagome),
        ("Ext3 Normalizing Flows", result.ext3_normalizing_flows),
    ]:
        print(f"  {ext_id}: {ext_result.classification.value}")
        print(f"    Key metric : {ext_result.key_metric}")
        print(f"    HW viable  : {ext_result.hardware_viable}")
        print()

    if result.prerequisite_failures:
        print(f"Prerequisite failures: {result.prerequisite_failures}")

    out_path = os.path.join(args.output_dir, "analysis_result.json")
    print(f"Full result written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
