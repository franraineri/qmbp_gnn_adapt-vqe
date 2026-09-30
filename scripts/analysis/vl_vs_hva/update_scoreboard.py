#!/usr/bin/env python
"""Update the best-results scoreboard from the study's JSON artifacts.

Reads every source via the consolidated report's ``collect()`` (single source of
truth for source parsing), turns each run into a scoreboard entry, and upserts it
into ``best_results_scoreboard_2d_frustrated.json`` — replacing a config's entry
only when a new run's fidelity improves. Also renders the markdown view.

Incremental + deterministic: re-running with no better data leaves the files
byte-stable (only the improved/added entries change). Safe to run after every
experiment.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/update_scoreboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from generate_vl_hva_consolidated_report import collect  # noqa: E402
from hva_vl_study_common import study_dir  # noqa: E402

from qmbp_simulation.analysis.state_prep_scoreboard import (  # noqa: E402
    build_document,
    make_entry,
    sort_entries,
    to_markdown,
    upsert_many,
)
from qmbp_simulation.framework.runner_base import resolve_project_root  # noqa: E402

_JSON_NAME = "best_results_scoreboard_2d_frustrated.json"
_MD_NAME = "best_results_scoreboard_2d_frustrated.md"


def _candidates_from_scenarios(scen: dict) -> list[dict]:
    """Flatten collected scenarios into scoreboard entries (one per Rec)."""
    out: list[dict] = []
    for s in scen.values():
        for r in s.recs:
            if r.fidelity is None:
                continue
            out.append(
                make_entry(
                    method=r.method,
                    n_qubits=s.n,
                    h=s.h,
                    variant=r.variant,
                    fidelity=r.fidelity,
                    loader=r.loader,
                    chi=r.chi,
                    abs_error=r.abs_error,
                    de_gap=r.de_gap,
                    n_2q=r.n_2q,
                    total_gates=r.total_gates,
                    depth=r.depth,
                    converged=r.converged,
                    source=r.source,
                )
            )
    return out


def main(argv=None) -> int:
    root = resolve_project_root(Path(__file__))
    scen = collect(root)
    candidates = _candidates_from_scenarios(scen)

    reports_dir = study_dir("reports")
    json_path = reports_dir / _JSON_NAME
    md_path = reports_dir / _MD_NAME

    # Load existing entries (incremental) or start fresh.
    existing: list[dict] = []
    if json_path.exists():
        existing = json.loads(json_path.read_text()).get("entries", [])

    entries, counts = upsert_many(existing, candidates)
    entries = sort_entries(entries)
    doc = build_document(entries)

    json_path.write_text(json.dumps(doc, indent=2) + "\n")
    md_path.write_text(to_markdown(doc))

    print(
        f"[scoreboard] {counts['added']} added, {counts['improved']} improved, "
        f"{counts['kept']} kept  → {len(entries)} entries",
        flush=True,
    )
    print(f"[scoreboard] wrote {json_path.name} + {md_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
