#!/usr/bin/env python3
"""CLI entry point for the project health checker.

Usage:
    python -m project_health                    # Full text report
    python -m project_health --json             # JSON output to stdout
    python -m project_health --markdown         # Markdown output to stdout
    python -m project_health --compact          # Summary only (no per-experiment detail)
    python -m project_health --json -o health.json   # Save JSON to file
    python -m project_health --no-state         # Don't persist state for delta tracking
    python -m project_health --results-dir results   # Custom results directory
    python -m project_health -o reports/        # Auto-timestamped file in directory
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from project_health.engine import run_health_check
from project_health.models import HealthReport
from project_health.reporter import (
    format_json,
    format_markdown,
    format_text,
    generate_timestamped_filename,
)
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
    python -m project_health --markdown         Markdown to stdout
    python -m project_health --json -o out.json Save JSON to file
    python -m project_health -o reports/        Auto-timestamped file in dir
    python -m project_health --no-state         Skip delta tracking
""",
    )

    # Output format (mutually exclusive)
    fmt_group = parser.add_mutually_exclusive_group()
    fmt_group.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of text",
    )
    fmt_group.add_argument(
        "--markdown",
        "--md",
        action="store_true",
        help="Output as Markdown (suitable for documentation)",
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
        help=(
            "Save output to file. If path is a directory, generates a "
            "timestamped filename automatically. If path ends with a file "
            "extension, uses that exact filename."
        ),
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


def _resolve_output_path(output_arg: str | None, fmt: str) -> Path | None:
    """Resolve the output path, adding timestamp if target is a directory.

    Parameters
    ----------
    output_arg : str | None
        The --output value from CLI (may be None, a dir, or a filepath).
    fmt : str
        One of "txt", "json", "md" — used for extension when auto-naming.

    Returns
    -------
    Path | None
        Resolved file path, or None for stdout.
    """
    if output_arg is None:
        return None

    path = Path(output_arg)

    # If the path is an existing directory OR ends with '/', generate a
    # timestamped filename inside it.
    if path.is_dir() or output_arg.endswith("/"):
        path.mkdir(parents=True, exist_ok=True)
        filename = generate_timestamped_filename("health_report", fmt)
        return path / filename

    # Otherwise, use the path as-is (but still ensure parent dirs exist).
    return path


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

    # Determine format
    if args.json:
        fmt = "json"
    elif args.markdown:
        fmt = "md"
    else:
        fmt = "txt"

    # Diff-only mode: only show delta
    if args.diff_only:
        output = _format_diff_only(report)
    elif fmt == "json":
        output = format_json(report)
    elif fmt == "md":
        output = format_markdown(report, compact=args.compact)
    else:
        output = format_text(report, compact=args.compact)

    # Resolve output path (with auto-timestamp if directory)
    output_path = _resolve_output_path(args.output, fmt)

    # Write output
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output)
        logging.info("Report saved to %s", output_path)
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
