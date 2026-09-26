"""Catalog builder for the vl_vs_hva study results tree.

Scans a results directory for study JSON artifacts, extracts each one's physics
/ method / metric coordinates and its ``status`` (partial | final | deprecated),
and produces a machine-readable ``index.json`` plus a human-readable ``INDEX.md``.
Idempotent: rebuilding from the same tree yields the same catalog.

Pure and dependency-light (stdlib only) so it is unit-testable without importing
from ``scripts/``. The study service points it at ``results/hva_vl_study`` and
adds the CLI; this module owns the scan + extraction + rendering logic and the
coordinate heuristics that cope with the study's several JSON shapes
(``point`` / ``rows`` / ``per_seed`` / ``data`` / flat analysis dicts).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from qmbp_simulation.framework.study_artifacts import (
    STATUS_FINAL,
    infer_status_from_payload,
)

INDEX_SCHEMA = "hva_vl_study_index_v1"

# Deprecated artifacts: superseded / overturned results the study explicitly
# flagged (kept, not deleted). Matched by filename stem substring.
_DEPRECATED_MARKERS = (
    "n9_improve_efg",          # theta_x directed perturbation — false positive (report E)
    "n9_confirm_multiseed",    # superseded by confirm_second_order
)


def _first(d: dict, *keys, default=None):
    """Return the first present, non-None value among ``keys`` in ``d``."""
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return default


def _coerce_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def extract_coordinates(payload: dict) -> dict:
    """Extract physics/method/metrics coordinates from a study JSON payload.

    Copes with the study's shapes by probing, in order: the ``params`` block,
    top-level physics fields, a ``point`` dict, the first element of ``rows``,
    and (for aggregates) leaving per-point fields empty. Returns a dict with
    ``physics`` {model, topo, N, p, h, J2, gap, e0}, ``method`` {optimizer,
    maxiter, sigma, seeds, n_hops, strategy}, and ``metrics`` {fidelity,
    abs_error, de_over_gap, cx, depth_2q}. Missing values are ``None``.
    """
    params = payload.get("params", {}) if isinstance(payload.get("params"), dict) else {}
    point = payload.get("point", {}) if isinstance(payload.get("point"), dict) else {}
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else None
    row0 = rows[0] if rows and isinstance(rows[0], dict) else {}
    # a flat "analysis" dict carries physics at top level
    flat = payload

    def pick(*keys):
        for src in (point, params, flat, row0):
            v = _first(src, *keys)
            if v is not None:
                return v
        return None

    physics = {
        "model": pick("model"),
        "topo": pick("topology", "topo"),
        "N": pick("N", "n_qubits", "n"),
        "p": pick("p_layers", "p"),
        "h": _coerce_float(pick("h", "h_test")),
        "J2": _coerce_float(pick("J2", "j2")),
        "gap": _coerce_float(pick("gap")),
        "e0": _coerce_float(pick("E0", "e0_exact", "e0")),
    }
    method = {
        "optimizer": pick("optimizer"),
        "maxiter": pick("maxiter"),
        "sigma": _coerce_float(pick("sigma")),
        "seeds": pick("n_seeds", "seeds"),
        "n_hops": pick("n_hops"),
        "strategy": pick("strategy", "strategy_resolved", "strategy_requested"),
    }
    metrics = {
        "fidelity": _coerce_float(pick("fidelity")),
        "abs_error": _coerce_float(pick("abs_error")),
        "de_over_gap": _coerce_float(pick("de_over_gap", "de_gap")),
        "cx": pick("cx", "n_2q_transpiled"),
        "depth_2q": pick("depth_2q"),
    }
    return {"physics": physics, "method": method, "metrics": metrics}


def _status_for(path: Path, payload: dict) -> str:
    stem = path.stem.lower()
    if any(m in stem for m in _DEPRECATED_MARKERS):
        return "deprecated"
    return infer_status_from_payload(payload)


def _count_points(payload: dict) -> int:
    """Best-effort count of data points a JSON holds (rows / per_seed / points)."""
    for key in ("rows", "points"):
        v = payload.get(key)
        if isinstance(v, list):
            return len(v)
    for key in ("per_seed", "data"):
        v = payload.get(key)
        if isinstance(v, dict):
            return len(v)
    if isinstance(payload.get("point"), dict):
        return 1
    return 0


def build_entry(path: Path, root: Path) -> dict:
    """Build one index entry for a study JSON file (best-effort, never raises)."""
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        payload = {}
    coords = extract_coordinates(payload)
    meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
    experiment = meta.get("experiment") or path.parent.name
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = path.name
    return {
        "run_id": meta.get("run_id") or path.stem,
        "experiment": experiment,
        "status": meta.get("status") or _status_for(path, payload),
        "path": rel,
        "schema": payload.get("schema") or meta.get("schema"),
        "source_script": payload.get("source_script") or meta.get("source_script"),
        "generated_utc": payload.get("generated_utc") or meta.get("updated_utc"),
        "n_points": _count_points(payload),
        **coords,
    }


def build_index(root: str | Path, *, migration_map: dict | None = None) -> dict:
    """Scan ``root`` for study JSON files and build the catalog dict.

    Skips the index files themselves and any ``.meta.json`` sidecars. Entries
    are sorted by (experiment, path) for stable, idempotent output. Includes the
    optional ``migration_map`` (old→new path) so readers can resolve legacy
    paths via the shim.
    """
    root = Path(root)
    entries = []
    for path in sorted(root.rglob("*.json")):
        name = path.name
        if name in ("index.json",) or name.endswith(".meta.json"):
            continue
        entries.append(build_entry(path, root))
    entries.sort(key=lambda e: (str(e["experiment"]), e["path"]))
    return {
        "schema": INDEX_SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "root": str(root),
        "n_entries": len(entries),
        "migration_map": dict(migration_map or {}),
        "entries": entries,
    }


def query_index(index: dict, *, experiment=None, status=None, n=None, h=None,
                method=None) -> list[dict]:
    """Filter index entries by experiment / status / N / h / optimizer-or-strategy."""
    out = []
    for e in index.get("entries", []):
        if experiment is not None and e.get("experiment") != experiment:
            continue
        if status is not None and e.get("status") != status:
            continue
        if n is not None and e["physics"].get("N") != n:
            continue
        if h is not None:
            eh = e["physics"].get("h")
            if eh is None or abs(eh - float(h)) > 0.005:
                continue
        if method is not None:
            m = (e["method"].get("optimizer") or "") + " " + (e["method"].get("strategy") or "")
            if method.lower() not in m.lower():
                continue
        out.append(e)
    return out


def _fmt(v, spec=""):
    if v is None:
        return "—"
    if spec and isinstance(v, (int, float)):
        return format(v, spec)
    return str(v)


def render_markdown(index: dict) -> str:
    """Render an aligned INDEX.md grouped by experiment from an index dict."""
    lines = [
        "# vl_vs_hva Study — Artifact Index",
        "",
        f"Generated: {index.get('generated_utc', '?')}",
        "",
        f"Total artifacts: {index.get('n_entries', 0)}",
        "",
    ]
    # status summary
    by_status: dict[str, int] = {}
    for e in index.get("entries", []):
        by_status[e["status"]] = by_status.get(e["status"], 0) + 1
    if by_status:
        lines.append("Status: " + ", ".join(
            f"{k}={v}" for k, v in sorted(by_status.items())) + ".")
        lines.append("")

    # group by experiment
    by_exp: dict[str, list] = {}
    for e in index.get("entries", []):
        by_exp.setdefault(str(e["experiment"]), []).append(e)

    header = ("| status | N | p | h | method | fidelity | dE/gap | CX | pts | path |")
    sep = ("|--------|---|---|------|--------|----------|--------|----|-----|------|")
    for exp in sorted(by_exp):
        lines.append(f"## {exp}")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for e in sorted(by_exp[exp], key=lambda x: x["path"]):
            ph, me, mt = e["physics"], e["method"], e["metrics"]
            method = me.get("strategy") or me.get("optimizer") or "—"
            lines.append(
                f"| {e['status']} | {_fmt(ph.get('N'))} | {_fmt(ph.get('p'))} "
                f"| {_fmt(ph.get('h'), '.2f')} | {method} "
                f"| {_fmt(mt.get('fidelity'), '.4f')} | {_fmt(mt.get('de_over_gap'), '.3f')} "
                f"| {_fmt(mt.get('cx'))} | {_fmt(e.get('n_points'))} | `{e['path']}` |"
            )
        lines.append("")
    return "\n".join(lines)


def write_index(root: str | Path, *, migration_map: dict | None = None,
                index_json: str | Path | None = None,
                index_md: str | Path | None = None) -> tuple[Path, Path]:
    """Build and write both ``index.json`` and ``INDEX.md``. Returns their paths.

    Idempotent except for the ``generated_utc`` timestamp: two consecutive
    builds of an unchanged tree differ only in that field.
    """
    root = Path(root)
    index = build_index(root, migration_map=migration_map)
    ij = Path(index_json) if index_json else root / "index.json"
    im = Path(index_md) if index_md else root / "INDEX.md"
    ij.write_text(json.dumps(index, indent=2, default=str))
    im.write_text(render_markdown(index))
    return ij, im


# ── Migration planning (pure; the CLI applies the plan) ───────────────────────
#
# Conservative, reversible migration: relocate the DISORGANIZED loose-root files
# (reports + n9_* heuristic diagnostics) into the taxonomy, and tag every other
# artifact in place. The already-organized subfolders (circuit_comparison/,
# resources/, vl_mps_n18/, ...) are where the scripts actively write via
# study_dir(name), so moving them would break write paths for no gain — they are
# tagged in place instead. Live paths are skipped.

# Root-level JSON diagnostics from the warm-start heuristics study → one sweep.
_HEURISTICS_EXPERIMENT = "warmstart_heuristics"
_REPORT_PREFIXES = ("REPORT_",)
_REPORT_EXTRA_NAMES = ("SUMMARY.md",)


def plan_migration(root: str | Path, *, live_paths: set[str] | None = None) -> dict:
    """Plan a conservative migration of the study tree (no side effects).

    Returns a dict with ``moves`` ({old_rel: new_rel}) for the loose-root files
    that should be relocated (reports → ``reports/``; ``n9_*`` diagnostics →
    ``sweeps/warmstart_heuristics/<stem>/``), and ``tag_in_place`` (list of rel
    paths to tag with a ``.meta.json`` sidecar without moving). Paths in
    ``live_paths`` (repo-relative to ``root``) are skipped entirely.

    The plan is deterministic and reversible (every move is invertible via the
    returned map). It is intentionally minimal: it does not disturb the
    already-organized subfolders that scripts write to.
    """
    root = Path(root)
    live = set(live_paths or set())
    moves: dict[str, str] = {}
    tag_in_place: list[str] = []
    skipped: list[str] = []

    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel = str(path.relative_to(root))
        name = path.name
        # never touch our own catalog files or existing sidecars
        if name in ("index.json", "INDEX.md") or name.endswith(".meta.json"):
            continue
        # skip anything under a live path (prefix match on the rel path)
        if any(rel == lp or rel.startswith(lp.rstrip("/") + "/") for lp in live):
            skipped.append(rel)
            continue

        at_root = path.parent == root
        # Reports (root .md) and SUMMARY.md → reports/
        if name.endswith(".md") and (
            any(name.startswith(pre) for pre in _REPORT_PREFIXES)
            or name in _REPORT_EXTRA_NAMES
        ):
            moves[rel] = f"reports/{name}"
            continue
        # Loose root n9_* diagnostics → sweeps/warmstart_heuristics/<stem>/
        if at_root and name.startswith("n9_") and name.endswith(".json"):
            stem = Path(name).stem
            moves[rel] = f"sweeps/{_HEURISTICS_EXPERIMENT}/{stem}/{name}"
            continue
        # Everything else stays where the scripts write it — tag in place.
        tag_in_place.append(rel)

    return {
        "moves": moves,
        "tag_in_place": tag_in_place,
        "skipped_live": sorted(skipped),
        "n_moves": len(moves),
        "n_tag_in_place": len(tag_in_place),
    }
