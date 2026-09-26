#!/usr/bin/env python
"""Organize / tag / index the vl_vs_hva study results tree.

Subcommands:

- ``reindex``   Scan the study tree and (re)generate ``index.json`` + ``INDEX.md``.
                Idempotent (only ``generated_utc`` changes between runs of an
                unchanged tree).
- ``query``     Filter the catalog by experiment / status / N / h / method and
                print matching runs.
- ``validate-cache``  Check GroundTruthCache keys follow ``topo|N|model|h:.2f``,
                report stale/duplicate/mis-formatted keys and coverage. Read-only
                unless ``--fix`` is passed, and even then only reformats KEY
                strings (never energies) and refuses collisions.

All paths resolve via the service's ``find_repo_root`` / ``STUDY_ROOT`` (never
``parents[N]``). This is the CLI layer; the scan/extraction/rendering logic lives
in ``qmbp_simulation.framework.study_index`` (pure, unit-tested).

Usage:
    .venv/bin/python scripts/analysis/vl_vs_hva/organize_results.py reindex
    .venv/bin/python scripts/analysis/vl_vs_hva/organize_results.py query --status final --n 9
    .venv/bin/python scripts/analysis/vl_vs_hva/organize_results.py validate-cache
    .venv/bin/python scripts/analysis/vl_vs_hva/organize_results.py validate-cache --fix
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import (  # noqa: E402
    INDEX_JSON,
    STUDY_ROOT,
    find_repo_root,
)
from qmbp_simulation.framework.study_artifacts import (  # noqa: E402
    build_meta,
    infer_status_from_payload,
    write_meta_sidecar,
)
from qmbp_simulation.framework.study_index import (  # noqa: E402
    build_entry,
    build_index,
    plan_migration,
    query_index,
    render_markdown,
    write_index,
)

_KEY_RE = re.compile(r"^(?P<topo>[^|]+)\|(?P<n>\d+)\|(?P<model>[^|]+)\|(?P<h>-?\d+\.\d+)$")


# ── reindex ───────────────────────────────────────────────────────────────────
def cmd_reindex(args) -> int:
    migration_map = {}
    if INDEX_JSON.exists():  # preserve an existing migration map across reindexes
        try:
            migration_map = json.loads(INDEX_JSON.read_text()).get("migration_map", {})
        except (json.JSONDecodeError, OSError):
            pass
    ij, im = write_index(STUDY_ROOT, migration_map=migration_map)
    idx = json.loads(ij.read_text())
    print(f"[reindex] {idx['n_entries']} artifacts indexed")
    print(f"  index.json -> {ij}")
    print(f"  INDEX.md   -> {im}")
    by_status: dict[str, int] = {}
    for e in idx["entries"]:
        by_status[e["status"]] = by_status.get(e["status"], 0) + 1
    print("  status:", ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))
    return 0


# ── query ─────────────────────────────────────────────────────────────────────
def cmd_query(args) -> int:
    if not INDEX_JSON.exists():
        print("[query] no index.json — run `reindex` first", flush=True)
        return 1
    index = json.loads(INDEX_JSON.read_text())
    hits = query_index(index, experiment=args.experiment, status=args.status,
                       n=args.n, h=args.h, method=args.method)
    print(f"[query] {len(hits)} match(es)")
    for e in hits:
        ph, mt = e["physics"], e["metrics"]
        fid = f"{mt['fidelity']:.4f}" if mt.get("fidelity") is not None else "—"
        print(f"  [{e['status']:9s}] {e['experiment']}/{e['run_id']} "
              f"N={ph.get('N')} h={ph.get('h')} fid={fid}  {e['path']}")
    return 0


# ── validate-cache ─────────────────────────────────────────────────────────────
def cmd_validate_cache(args) -> int:
    repo = find_repo_root(Path(__file__).resolve().parent)
    gt_path = repo / "data" / "ground_truth_cache.json"
    if not gt_path.exists():
        print(f"[validate-cache] not found: {gt_path}")
        return 1
    raw = json.loads(gt_path.read_text())
    entries = raw.get("entries", raw) if isinstance(raw, dict) else {}

    ok, malformed, non_2dp = [], [], []
    for k in entries:
        m = _KEY_RE.match(k)
        if not m:
            malformed.append(k)
            continue
        h_str = m.group("h")
        # correct format iff exactly 2 decimals
        if len(h_str.split(".")[1]) == 2:
            ok.append(k)
        else:
            non_2dp.append(k)

    # coverage by (topo, model, N)
    coverage: dict[tuple, int] = {}
    for k in ok + non_2dp:
        m = _KEY_RE.match(k)
        if m:
            coverage[(m.group("topo"), m.group("model"), m.group("n"))] = \
                coverage.get((m.group("topo"), m.group("model"), m.group("n")), 0) + 1

    print(f"[validate-cache] {gt_path.relative_to(repo)}")
    print(f"  total keys        : {len(entries)}")
    print(f"  well-formed (h.2f): {len(ok)}")
    print(f"  wrong h precision : {len(non_2dp)}")
    print(f"  malformed keys    : {len(malformed)}")
    print(f"  coverage cells    : {len(coverage)} (topo,model,N)")
    if malformed[:5]:
        print("  malformed sample  :", malformed[:5])
    if non_2dp[:5]:
        print("  wrong-precision   :", non_2dp[:5])

    # Detect collisions the reformat would cause (never silently overwrite).
    would_collide = []
    reformable = []
    ok_set = set(ok)
    for k in non_2dp:
        m = _KEY_RE.match(k)
        new = f"{m.group('topo')}|{m.group('n')}|{m.group('model')}|{float(m.group('h')):.2f}"
        if new in ok_set or new in {r[1] for r in reformable}:
            would_collide.append((k, new))
        else:
            reformable.append((k, new))
    print(f"  reformattable     : {len(reformable)}  |  would-collide: {len(would_collide)}")

    if not args.fix:
        print("  (read-only — pass --fix to reformat KEY strings only; energies never touched)")
        return 0

    if would_collide:
        print(f"  REFUSING --fix: {len(would_collide)} key collisions "
              f"(a 2-dp key already exists). Resolve manually — energies must not merge blindly.")
        print("  first collisions:", would_collide[:5])
        return 2

    # Apply: reformat keys only, keep values byte-identical. Backup first.
    backup = gt_path.with_suffix(".json.bak")
    backup.write_text(gt_path.read_text())
    new_entries = dict(entries)
    for old, new in reformable:
        new_entries[new] = new_entries.pop(old)
    if "entries" in raw:
        raw["entries"] = new_entries
        out = raw
    else:
        out = new_entries
    gt_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"  FIXED {len(reformable)} keys (backup: {backup.name}); energies unchanged.")
    return 0


def _detect_live_paths() -> set[str]:
    """Return repo-relative study paths that a running experiment is writing to.

    Best-effort: scans running processes for vl_vs_hva scripts and maps the
    known writers to their output folders/files so the migration skips them.
    """
    import subprocess

    live: set[str] = set()
    try:
        out = subprocess.run(["ps", "aux"], capture_output=True, text=True,
                             timeout=5, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return live
    for line in out.splitlines():
        if "run_n18_second_order.py" in line:
            live.add("hva_nnn_sweep")          # writes n18_*.json here
            live.add("resources/n18_frustrated")  # theta/qpy/hops artifacts
        if "run_hva_metropolis_point.py" in line:
            live.add("hva_nnn_sweep")
            live.add("resources")
        if "confirm_second_order_warmstart.py" in line:
            live.add("hva_nnn_sweep")
        if "hva_nnn_h_sweep_frustrated.py" in line:
            live.add("hva_nnn_sweep")
    return live


def _write_sidecar_for(abs_path: Path, root: Path) -> None:
    """Write a .meta.json sidecar for a JSON/binary artifact (best-effort)."""
    rel = str(abs_path.relative_to(root))
    experiment = abs_path.parent.name
    payload = {}
    if abs_path.suffix == ".json":
        try:
            payload = json.loads(abs_path.read_text())
        except (json.JSONDecodeError, OSError):
            payload = {}
    status = (payload.get("meta", {}) or {}).get("status") or infer_status_from_payload(payload)
    entry = build_entry(abs_path, root) if abs_path.suffix == ".json" else {}
    meta = build_meta(
        experiment=(payload.get("meta", {}) or {}).get("experiment") or experiment,
        run_id=abs_path.stem,
        status=status,
        source_script=payload.get("source_script", "migrated"),
        physics=entry.get("physics") if entry else None,
        method=entry.get("method") if entry else None,
        metrics=entry.get("metrics") if entry else None,
        repo_root=find_repo_root(Path(__file__).resolve().parent),
    )
    meta["result_path"] = rel
    write_meta_sidecar(abs_path, meta)


def cmd_migrate(args) -> int:
    import shutil

    root = STUDY_ROOT
    repo = find_repo_root(Path(__file__).resolve().parent)
    live = _detect_live_paths()
    plan = plan_migration(root, live_paths=live)

    print(f"[migrate] {'APPLY' if args.apply else 'DRY-RUN'} "
          f"(reversible; move-not-delete)")
    if live:
        print(f"  live paths skipped: {sorted(live)}")
    print(f"  moves: {plan['n_moves']} | tag-in-place: {plan['n_tag_in_place']} "
          f"| skipped(live): {len(plan['skipped_live'])}")
    for old, new in sorted(plan["moves"].items()):
        print(f"    MOVE {old}  ->  {new}")

    if not args.apply:
        print("  (dry-run — pass --apply to execute; a migration_map is recorded in index.json)")
        return 0

    # Apply moves (move-not-delete), updating embedded result_path in JSON.
    applied: dict[str, str] = {}
    for old, new in plan["moves"].items():
        src = root / old
        dst = root / new
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            print(f"    SKIP (dest exists): {new}")
            continue
        # update embedded result_path for JSON before moving
        if src.suffix == ".json":
            try:
                d = json.loads(src.read_text())
                if isinstance(d, dict) and "result_path" in d:
                    d["result_path"] = str((root / new).relative_to(repo))
                    src.write_text(json.dumps(d, indent=2, default=str))
            except (json.JSONDecodeError, OSError):
                pass
        shutil.move(str(src), str(dst))
        applied[str((root / old).relative_to(repo))] = str((root / new).relative_to(repo))
        _write_sidecar_for(dst, root)

    # Update .md cross-references to moved paths (both old rel-to-study and
    # rel-to-repo forms) so report links keep resolving.
    _update_md_references(root, plan["moves"])

    # Tag-in-place artifacts with sidecars (skip live, skip if already tagged).
    n_tagged = 0
    for rel in plan["tag_in_place"]:
        p = root / rel
        if not p.exists():
            continue
        _write_sidecar_for(p, root)
        n_tagged += 1

    # Persist migration map into index.json + regenerate INDEX.md.
    write_index(root, migration_map=applied)
    print(f"  applied {len(applied)} moves, tagged {n_tagged} in place.")
    print(f"  migration_map recorded in {INDEX_JSON.name}; INDEX.md regenerated.")
    return 0


def _update_md_references(root: Path, moves: dict) -> None:
    """Rewrite references to moved files inside .md documents (best-effort)."""
    md_files = list(root.rglob("*.md"))
    # build old->new in study-relative and repo-relative-ish forms
    repl = {}
    for old, new in moves.items():
        repl[old] = new
        repl[f"results/hva_vl_study/{old}"] = f"results/hva_vl_study/{new}"
    for md in md_files:
        try:
            text = md.read_text()
        except OSError:
            continue
        new_text = text
        for a, b in repl.items():
            new_text = new_text.replace(a, b)
        if new_text != text:
            md.write_text(new_text)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Organize/tag/index the vl_vs_hva study results")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("reindex", help="(re)generate index.json + INDEX.md from the tree")

    q = sub.add_parser("query", help="filter the catalog")
    q.add_argument("--experiment")
    q.add_argument("--status", choices=["partial", "final", "deprecated"])
    q.add_argument("--n", type=int)
    q.add_argument("--h", type=float)
    q.add_argument("--method")

    v = sub.add_parser("validate-cache", help="check GroundTruthCache key formatting")
    v.add_argument("--fix", action="store_true",
                   help="reformat wrong-precision KEYS only (never energies); refuses collisions")

    m = sub.add_parser("migrate", help="relocate loose files into the taxonomy + tag (dry-run by default)")
    m.add_argument("--apply", action="store_true",
                   help="execute the migration (default is dry-run); move-not-delete, reversible")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "reindex":
        return cmd_reindex(args)
    if args.command == "query":
        return cmd_query(args)
    if args.command == "validate-cache":
        return cmd_validate_cache(args)
    if args.command == "migrate":
        return cmd_migrate(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
