"""Crash-safe, resumable checkpointing for long study runs.

Experiment-agnostic helper promoted out of the ``vl_vs_hva`` study scripts,
which each reimplemented the same JSON checkpoint + resume dance (per-hop in the
Metropolis point runner, per-seed in the second-order confirmation, per-config
cache in the VL quality sweep). Centralizing it removes that duplication and
fixes the two real failures the study hit: a runner that only saved at the end,
and outputs misrouted by a broken repo-root path.

Design goals:

- **Pure and testable.** Round-trips through a caller-supplied JSON path (or an
  injected writer), so it can be unit-tested with ``tmp_path`` and no
  ``scripts/`` import. It does not know about ``results/hva_vl_study/`` — the
  study service wires that in.
- **Unit-of-work granularity.** :class:`StudyCheckpoint` stores a dict of
  completed units keyed by a stable string (e.g. ``"0.50|3"`` for (h, seed) or
  ``"7"`` for hop 7). ``is_done`` / ``record`` / ``pending`` drive the
  reload-and-skip loop.
- **Atomic writes.** The JSON is written tmp + rename so a crash mid-write never
  corrupts the checkpoint.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any


def atomic_write_json(path: str | Path, payload: dict) -> Path:
    """Write ``payload`` as JSON to ``path`` atomically (tmp + rename).

    Parent directories are created. The temporary file carries the writer PID so
    concurrent writers do not collide. Returns the path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        tmp.rename(path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return path


def read_json(path: str | Path) -> dict | None:
    """Read a JSON file, returning ``None`` if missing or unpar. (best-effort)."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


class StudyCheckpoint:
    """Track completed units of work and persist them crash-safely.

    Parameters
    ----------
    units_key : str
        Top-level key in the payload holding the ``{unit_id: record}`` dict of
        completed work (e.g. ``"per_seed"``, ``"runs"``, ``"cache"``).
    writer : callable | None
        A ``writer(payload: dict) -> Any`` used to persist. When ``None`` and a
        ``path`` is given, an atomic JSON write to ``path`` is used. The study
        service injects its ``save_json`` here so traceability metadata
        (result_path/source_script/generated_utc) is preserved.
    path : str | Path | None
        JSON path used for the default writer and for :meth:`resume`.

    Notes
    -----
    Either ``writer`` or ``path`` (or both) must be provided. ``path`` is
    required to resume from disk.
    """

    def __init__(
        self,
        units_key: str,
        *,
        writer: Callable[[dict], Any] | None = None,
        path: str | Path | None = None,
    ) -> None:
        if writer is None and path is None:
            raise ValueError("StudyCheckpoint requires a writer or a path")
        self.units_key = units_key
        self._path = Path(path) if path is not None else None
        self._writer = writer
        self._units: dict[str, Any] = {}

    # ── resume ────────────────────────────────────────────────────────────
    def resume(self) -> int:
        """Load previously completed units from ``path`` (if any).

        Returns the number of units restored (0 on a fresh start). Safe to call
        when no checkpoint exists.
        """
        if self._path is None:
            return 0
        data = read_json(self._path)
        if not data:
            return 0
        units = data.get(self.units_key)
        if isinstance(units, dict):
            self._units = dict(units)
        return len(self._units)

    # ── unit tracking ──────────────────────────────────────────────────────
    @property
    def units(self) -> dict[str, Any]:
        """The completed-unit records, keyed by unit id."""
        return self._units

    def is_done(self, unit_id: str) -> bool:
        """Whether ``unit_id`` has already been completed."""
        return unit_id in self._units

    def record(self, unit_id: str, value: Any) -> None:
        """Mark ``unit_id`` completed with its result ``value`` (in memory)."""
        self._units[unit_id] = value

    def pending(self, all_units: Iterable[str]) -> list[str]:
        """Return the subset of ``all_units`` not yet completed, in order."""
        return [u for u in all_units if u not in self._units]

    # ── persistence ─────────────────────────────────────────────────────────
    def persist(self, extra: dict | None = None) -> Any:
        """Write the checkpoint now (called after each unit for crash-safety).

        The payload always contains ``{units_key: units}``; ``extra`` merges in
        derived/aggregate fields (summaries, params, schema) without being part
        of the resumable unit map. Uses the injected writer when present, else
        an atomic JSON write to ``path``.
        """
        payload: dict[str, Any] = {self.units_key: self._units}
        if extra:
            payload.update(extra)
        if self._writer is not None:
            return self._writer(payload)
        return atomic_write_json(self._path, payload)


def resume_ordered_list(
    path: str | Path, list_key: str, *, inside: str | None = None
) -> list:
    """Resume an ordered list of records (e.g. per-hop ``runs``) from a payload.

    Some runners store progress as an ordered list rather than a keyed map (the
    Metropolis point runner appends one record per hop). This returns that list
    (empty if absent), optionally nested under ``inside`` (e.g. ``"point"``), so
    the caller can restart from ``len(list)``.
    """
    data = read_json(path)
    if not data:
        return []
    container = data.get(inside, {}) if inside else data
    if not isinstance(container, dict):
        return []
    value = container.get(list_key)
    return list(value) if isinstance(value, Sequence) and not isinstance(value, str) else []
