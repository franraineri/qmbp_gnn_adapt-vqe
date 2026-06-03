#!/usr/bin/env python3
"""CLI entry point for the project health checker.

Usage:
    python -m project_health                    # Full text report
    python -m project_health --json             # JSON output to stdout
    python -m project_health --compact          # Summary only (no per-experiment detail)
    python -m project_health --json -o health.json   # Save JSON to file
    python -m project_health --no-state         # Don't persist state for delta tracking
    python -m project_health --results-dir results   # Custom results directory
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from project_health.engine import run_health_check
from project_health.models import HealthReport
from project_health.reporter import format_json, format_text
from project_health.state import DEFAULT_STATE_FILE


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Project health checker — unified Phase 4 analysis report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    python -m project_health                    Full text report
    python -m project_health --compact          Summary only
    python -m project_health --json             JSON to stdout
    python -m project_health --json -o out.json Save JSON to file
    python -m project_health --no-state         Skip delta tracking
""",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of text",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Show summary only (skip per-experiment details)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Save output to file (default: stdout)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results root directory (default: results)",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default=str(DEFAULT_STATE_FILE),
        help="State file for delta tracking (default: .project_health_state.json)",
    )
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Don't save state (disables 'new since last run' on next invocation)",
    )
    parser.add_argument(
        "--diff-only",
        action="store_true",
        help="Only show new/removed results since last run (requires previous state)",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: exit with code 1 if CRITICAL actions exist",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stderr,
    )

    # Run the health check
    report = run_health_check(
        results_dir=Path(args.results_dir),
        state_file=Path(args.state_file),
        save_state=not args.no_state,
    )

    # Diff-only mode: only show delta
    if args.diff_only:
        output = _format_diff_only(report)
    elif args.json:
        output = format_json(report)
    else:
        output = format_text(report, compact=args.compact)

    # Write output
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output)
        logging.info("Report saved to %s", path)
    else:
        print(output)

    # CI exit code: non-zero if CRITICAL actions exist
    if args.ci:
        from project_health.models import Priority

        has_critical = any(a.priority == Priority.CRITICAL for a in report.actions)
        sys.exit(1 if has_critical else 0)


def _format_diff_only(report: HealthReport) -> str:
    """Format only the delta (new/removed) for quick checks."""
    lines: list[str] = []
    lines.append(f"Delta since last run ({report.timestamp}):")
    lines.append("")

    if not report.new_results and not report.removed_results:
        lines.append("  No changes since last run.")
        return "\n".join(lines)

    if report.new_results:
        lines.append(f"  + {report.n_new} new result(s):")
        for f in report.new_results:
            lines.append(f"    + {f}")
        lines.append("")

    if report.removed_results:
        lines.append(f"  − {report.n_removed} removed result(s):")
        for f in report.removed_results:
            lines.append(f"    − {f}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
