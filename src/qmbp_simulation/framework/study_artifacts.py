"""Canonical artifact naming, metadata tagging, and status for study runs.

Experiment-agnostic, dependency-light core for the ``vl_vs_hva`` study's
artifact organization. Promoted to ``src`` so it is unit-testable without a
``scripts/`` import (repo testing rule). The study service
(``hva_vl_study_common.py``) re-exports these and wires in its results-tree
layout + traceable ``save_json``.

Owns:

- :func:`build_artifact_name` — the single filename scheme for every artifact
  (``{model}_{topo}_N{n}_p{p}_h{h:.2f}_J2{j2:.2f}_{method}_{kind}.{ext}``),
  preserving the repo's ``h:.2f`` and ``N{n}_p{p}`` conventions.
- :data:`STATUS_*` and :func:`promote_status` — the file/catalog-level
  ``partial | final | deprecated`` distinction (not only an internal ``done``).
- :func:`build_meta` — the structured tag block embedded in JSON and written as
  a ``.meta.json`` sidecar for binary artifacts (NPZ/QPY/PNG).
- :func:`git_commit` — best-effort short commit for provenance.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

META_SCHEMA = "hva_vl_study_meta_v1"

# File- and catalog-level status. A run is written as ``partial`` while it is
# still checkpointing and flips to ``final`` atomically on completion.
# ``deprecated`` marks superseded artifacts that are kept, not deleted.
STATUS_PARTIAL = "partial"
STATUS_FINAL = "final"
STATUS_DEPRECATED = "deprecated"
VALID_STATUS = (STATUS_PARTIAL, STATUS_FINAL, STATUS_DEPRECATED)

# Artifact kinds encoded in the filename.
VALID_KINDS = ("final", "partial", "theta", "circuit", "analysis", "figure", "cache")

_ALLOWED_TRANSITIONS = {
    STATUS_PARTIAL: {STATUS_FINAL, STATUS_DEPRECATED, STATUS_PARTIAL},
    STATUS_FINAL: {STATUS_DEPRECATED, STATUS_FINAL},
    STATUS_DEPRECATED: {STATUS_DEPRECATED},
}


def _fmt_h(h: float) -> str:
    """Format an h-value with the repo's mandatory 2-decimal key precision."""
    return f"{float(h):.2f}"


def build_artifact_name(
    *,
    model: str,
    topology: str,
    n: int,
    p: int,
    h: float | None = None,
    j2: float = 0.0,
    method: str = "vqe",
    kind: str = "final",
    ext: str = "json",
) -> str:
    """Build the one canonical artifact filename.

    Scheme::

        {model}_{topo}_N{n}_p{p}_h{h:.2f}_J2{j2:.2f}_{method}_{kind}.{ext}

    ``h`` may be ``None`` for artifacts that span an h-sweep (the ``h{...}``
    token is then omitted). ``N{n}_p{p}`` and ``h:.2f`` follow the repo
    conventions exactly. Raises on an unknown ``kind``.
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {VALID_KINDS}")
    parts = [model, topology, f"N{int(n)}", f"p{int(p)}"]
    if h is not None:
        parts.append(f"h{_fmt_h(h)}")
    parts.append(f"J2{float(j2):.2f}")
    parts.append(method)
    parts.append(kind)
    stem = "_".join(str(x) for x in parts)
    ext = ext.lstrip(".")
    return f"{stem}.{ext}"


def meta_sidecar_path(artifact_path: str | Path) -> Path:
    """Return the ``.meta.json`` sidecar path for a binary artifact."""
    p = Path(artifact_path)
    return p.with_suffix(p.suffix + ".meta.json")


def git_commit(repo_root: str | Path | None = None) -> str | None:
    """Best-effort short git commit hash for provenance (None on failure)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True, text=True, timeout=5, check=False,
        )
        sha = out.stdout.strip()
        return sha or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_meta(
    *,
    experiment: str,
    run_id: str,
    status: str = STATUS_PARTIAL,
    source_script: str = "unknown",
    physics: dict | None = None,
    method: dict | None = None,
    metrics: dict | None = None,
    provenance: dict | None = None,
    artifacts: dict | None = None,
    created_utc: str | None = None,
    repo_root: str | Path | None = None,
) -> dict:
    """Build the structured metadata tag block.

    Returns a dict with the versioned schema, status, identity, timestamps,
    git commit, and the physics/method/metrics/provenance/artifacts sub-blocks.
    ``created_utc`` is preserved across updates when provided (so re-tagging a
    run keeps its original creation time while refreshing ``updated_utc``).
    Raises on an invalid ``status``.
    """
    if status not in VALID_STATUS:
        raise ValueError(f"invalid status {status!r}; expected one of {VALID_STATUS}")
    now = datetime.now(UTC).isoformat()
    return {
        "schema": META_SCHEMA,
        "status": status,
        "experiment": experiment,
        "run_id": run_id,
        "created_utc": created_utc or now,
        "updated_utc": now,
        "source_script": source_script,
        "git_commit": git_commit(repo_root),
        "physics": dict(physics or {}),
        "method": dict(method or {}),
        "metrics": dict(metrics or {}),
        "provenance": dict(provenance or {}),
        "artifacts": dict(artifacts or {}),
    }


def promote_status(current: str, target: str) -> str:
    """Validate and return a status transition (partial→final→deprecated).

    Raises ``ValueError`` on an illegal transition (e.g. final→partial). Same
    status is always allowed (idempotent re-writes).
    """
    if current not in VALID_STATUS:
        raise ValueError(f"invalid current status {current!r}")
    if target not in VALID_STATUS:
        raise ValueError(f"invalid target status {target!r}")
    if target not in _ALLOWED_TRANSITIONS[current]:
        raise ValueError(f"illegal status transition {current!r} -> {target!r}")
    return target


def infer_status_from_payload(payload: dict) -> str:
    """Infer a status for a legacy JSON payload lacking an explicit status.

    Rules (used by the migration):
    - explicit ``meta.status`` or top-level ``status`` wins if valid;
    - a ``point``/``per_seed``/``runs`` block with ``done: True`` → ``final``;
    - ``done: False`` or a partials-only checkpoint → ``partial``;
    - otherwise ``final`` (a completed aggregate sweep with rows).
    """
    for loc in (payload.get("meta", {}), payload):
        s = loc.get("status") if isinstance(loc, dict) else None
        if s in VALID_STATUS:
            return s
    point = payload.get("point")
    if isinstance(point, dict) and "done" in point:
        return STATUS_FINAL if point.get("done") else STATUS_PARTIAL
    if "done" in payload:
        return STATUS_FINAL if payload.get("done") else STATUS_PARTIAL
    # per_seed / rows aggregates with content are completed results by default.
    return STATUS_FINAL


import json  # noqa: E402
import os  # noqa: E402


def write_json_atomic(path: str | Path, payload: dict) -> Path:
    """Atomic JSON write (tmp + rename) with PID-tagged temp file."""
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


def write_meta_sidecar(artifact_path: str | Path, meta: dict) -> Path:
    """Write a ``.meta.json`` sidecar next to a (binary) artifact. Returns path."""
    return write_json_atomic(meta_sidecar_path(artifact_path), meta)


class StudyArtifactWriter:
    """Concrete artifact writer: canonical names + tag block + status + sidecars.

    Plugs into the study lifecycle: the runner produces a per-point dict and
    calls :meth:`write_partial` after each unit of work and :meth:`write_final`
    on completion (an atomic ``partial -> final`` promotion). Binary artifacts
    (θ NPZ, bound QPY, figures) get a ``.meta.json`` sidecar via
    :meth:`tag_artifact`.

    This class owns the *mechanics* (naming, tagging, status, atomic writes);
    the *results-tree layout* (which directory a run/sweep lives in) is injected
    as ``run_dir`` by the study service so ``src`` stays layout-agnostic.

    Parameters
    ----------
    experiment : str
        Logical experiment name (e.g. ``"hva_nnn_sweep"``).
    run_dir : Path
        Directory for this run's artifacts (created on demand).
    run_id : str
        Stable identifier for the run.
    source_script : str
        Repo-relative path of the producing script (traceability).
    repo_root : Path | None
        For git provenance.
    """

    def __init__(self, *, experiment: str, run_dir: str | Path, run_id: str,
                 source_script: str = "unknown", repo_root: str | Path | None = None):
        self.experiment = experiment
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.source_script = source_script
        self.repo_root = repo_root
        self._created_utc: str | None = None

    def _meta(self, status: str, **blocks) -> dict:
        meta = build_meta(
            experiment=self.experiment, run_id=self.run_id, status=status,
            source_script=self.source_script, repo_root=self.repo_root,
            created_utc=self._created_utc, **blocks,
        )
        # Preserve the original creation time across subsequent writes.
        self._created_utc = meta["created_utc"]
        return meta

    def _write(self, filename: str, status: str, payload: dict, blocks: dict) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        meta = self._meta(status, **blocks)
        enriched = {"meta": meta, **payload}
        path = self.run_dir / filename
        enriched["meta"]["result_path"] = str(path)
        return write_json_atomic(path, enriched)

    def write_partial(self, filename: str, payload: dict, **blocks) -> Path:
        """Write a crash-safe partial checkpoint (``status: partial``)."""
        return self._write(filename, STATUS_PARTIAL, payload, blocks)

    def write_final(self, filename: str, payload: dict, **blocks) -> Path:
        """Write the completed result (``status: final``)."""
        return self._write(filename, STATUS_FINAL, payload, blocks)

    def tag_artifact(self, artifact_path: str | Path, *, status: str = STATUS_FINAL,
                     **blocks) -> Path:
        """Write a ``.meta.json`` sidecar for a binary artifact (NPZ/QPY/PNG)."""
        meta = self._meta(status, **blocks)
        meta["result_path"] = str(artifact_path)
        return write_meta_sidecar(artifact_path, meta)


class PathShim:
    """Resolver mapping legacy result paths to their post-migration locations.

    Built from the migration map (``{old_rel: new_rel}``) recorded in
    ``index.json``. Readers that still reference old paths call
    :meth:`resolve` to get the current path, so nothing breaks after files move.
    """

    def __init__(self, mapping: dict[str, str], *, root: str | Path):
        self.root = Path(root)
        # normalize keys/values to repo-relative posix strings
        self._map = {str(k): str(v) for k, v in mapping.items()}

    def resolve(self, path: str | Path) -> Path:
        """Return the current path for ``path`` (identity if not remapped)."""
        p = Path(path)
        candidates = [str(p)]
        try:
            candidates.append(str(p.relative_to(self.root)))
        except ValueError:
            pass
        candidates.append(p.name)
        for c in candidates:
            if c in self._map:
                new = Path(self._map[c])
                return new if new.is_absolute() else self.root / new
        return p

    def exists(self, path: str | Path) -> bool:
        """Whether the resolved path exists on disk."""
        return self.resolve(path).exists()
