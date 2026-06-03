"""Persistence layer for tracking "new since last run" delta.

Stores a set of known source_file paths from the previous run.
On the next run, any paths not in that set are flagged as new.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_STATE_FILE = Path(".project_health_state.json")


def load_previous_state(state_file: Path = DEFAULT_STATE_FILE) -> set[str]:
    """Load known source_file paths from the last run.

    Returns an empty set if no state file exists (first run).
    """
    if not state_file.exists():
        logger.debug("No previous state file found (%s) — treating as first run.", state_file)
        return set()

    try:
        with open(state_file) as f:
            data = json.load(f)
        return set(data.get("known_files", []))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load state file %s: %s", state_file, e)
        return set()


def save_current_state(
    known_files: set[str],
    state_file: Path = DEFAULT_STATE_FILE,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save current source_file paths for next-run comparison.

    Parameters
    ----------
    known_files : set[str]
        All source_file paths seen in this run.
    state_file : Path
        Where to persist the state.
    metadata : dict | None
        Optional metadata (timestamp, counts) for debugging.
    """
    data: dict[str, Any] = {
        "known_files": sorted(known_files),
        "n_files": len(known_files),
    }
    if metadata:
        data["metadata"] = metadata

    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(data, f, indent=2)
    logger.debug("Saved state with %d known files to %s", len(known_files), state_file)


def detect_new_results(
    current_files: set[str],
    state_file: Path = DEFAULT_STATE_FILE,
) -> list[str]:
    """Compare current scan results against previous state.

    Returns list of source_file paths that are new since last run.
    """
    previous = load_previous_state(state_file)
    if not previous:
        # First run — everything is "new" but we don't flag it
        return []

    new = sorted(current_files - previous)
    return new


def detect_removed_results(
    current_files: set[str],
    state_file: Path = DEFAULT_STATE_FILE,
) -> list[str]:
    """Detect results that existed in previous state but are now missing.

    Returns list of source_file paths that were removed since last run.
    Useful for detecting deleted/cleaned-up results.
    """
    previous = load_previous_state(state_file)
    if not previous:
        return []

    removed = sorted(previous - current_files)
    return removed
