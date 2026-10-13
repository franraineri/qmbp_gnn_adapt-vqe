#!/usr/bin/env python
"""Study-level post-run sync for the vl_vs_hva study (integration layer, I1).

The study analogue of ``post_experiment_sync`` (which only covers the
training/zoo pipeline and never reads ``results/hva_vl_study/``). One idempotent,
best-effort entry point that refreshes everything the study produces, so a run
updates its own tracking the same way a training run refreshes the zoo:

1. **Coherence** — ``validate_study_coherence`` (C1): re-evaluate a sample of
   specs + cross-check bond_ablation e0 vs the GroundTruthCache. Writes
   ``auto_tuning/study_coherence.md``.
2. **Summaries** — regenerate ``ansatz_summary/<topo>_<model>/solutions_N*_h*.md``
   for every (topology, N, h) that has artifacts (reusing the summary's own
   ``aggregate``/``render_markdown``/``output_path`` — no re-implementation).
3. **Scoreboard** — ``update_scoreboard.main()`` (study scoreboard; idempotent).
4. **Learning** — ``analyze_override_log`` (adopt --auto overrides past the
   evidence gate). Writes ``auto_tuning/override_analysis.md`` + learned rules.

Every step is wrapped so one failing step never sinks the rest (same contract as
``post_experiment_sync`` / ``sync_scoreboard``). Idempotent and byte-stable when
nothing changed, so it is safe to call after every run AND on demand.

Usage
-----
    .venv/bin/python scripts/analysis/vl_vs_hva/study_sync.py            # all
    .venv/bin/python scripts/analysis/vl_vs_hva/study_sync.py --no-coherence
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hva_vl_study_common import STUDY_ROOT  # noqa: E402

# (topology, N, h) discoverable from artifact filenames. The topology is one of
# the known lattice names, always immediately before the _N{n}_h{h} suffix — the
# variant/prune prefix can itself contain substrings like 'nn_dom_rx', so we
# anchor on the KNOWN topology tokens (an unanchored .+? mis-captures those).
_TOPOLOGIES = ("chain_1d", "heavy_hex", "kagome", "square", "triangular", "ladder")
_SUFFIX_RE = re.compile(
    rf"_(?P<topo>{'|'.join(_TOPOLOGIES)})_N(?P<n>\d+)_h(?P<h>[0-9.]+)\.(spec\.json|json)$"
)


def _discover_configs() -> set[tuple[str, int, float]]:
    """Scan ansatz_specs/ + bond_ablation/ for (topology, N, h) triples present."""
    configs: set[tuple[str, int, float]] = set()
    for pat in (STUDY_ROOT / "ansatz_specs" / "*.spec.json",
                STUDY_ROOT / "bond_ablation" / "*.json"):
        for fp in glob.glob(str(pat)):
            m = _SUFFIX_RE.search(Path(fp).name)
            if not m:
                continue
            try:
                configs.add((m.group("topo"), int(m.group("n")), round(float(m.group("h")), 2)))
            except ValueError:
                continue
    return configs


def run_study_sync(*, coherence: bool = True, summaries: bool = True,
                   scoreboard: bool = True, learning: bool = True,
                   verbose: bool = True) -> dict:
    """Run the study sync steps. Returns a dict of per-step outcomes."""
    out: dict = {"steps_completed": [], "steps_failed": []}

    def _log(msg):
        if verbose:
            print(f"[study_sync] {msg}", flush=True)

    # 1) Coherence (C1)
    if coherence:
        try:
            from qmbp_simulation.analysis.auto_tuning import validate_study_coherence

            r = validate_study_coherence(write=True)
            out["coherence"] = {"n_issues": r["n_issues"], "ok": r["ok"]}
            out["steps_completed"].append("coherence")
            _log(f"coherence: {r['n_specs_checked']} specs, {r['n_rows_checked']} rows, "
                 f"{r['n_issues']} issue(s)")
        except Exception as e:  # noqa: BLE001
            out["steps_failed"].append(f"coherence: {e}")
            _log(f"coherence FAILED: {e}")

    # 2) ONE general overview per topology (NOT one file per config — markdown is
    #    a view, not a data layer). Groups discovered configs by topology and
    #    writes a single OVERVIEW_<topo>_<model>.md each.
    if summaries:
        try:
            import generate_ansatz_summary as gas

            topos = sorted({topo for topo, _n, _h in _discover_configs()})
            n_files = 0
            for topo in topos:
                try:
                    gas.main(["--overview", "--topology", topo, "--model", "tfim_frustrated"])
                    n_files += 1
                except SystemExit:
                    n_files += 1
                except Exception as ce:  # noqa: BLE001
                    _log(f"  overview {topo} skipped: {ce}")
            out["summaries"] = {"n_overview_files": n_files}
            out["steps_completed"].append("summaries")
            _log(f"summaries: {n_files} general overview file(s) (one per topology)")
        except Exception as e:  # noqa: BLE001
            out["steps_failed"].append(f"summaries: {e}")
            _log(f"summaries FAILED: {e}")

    # 3) Study scoreboard (idempotent upsert)
    if scoreboard:
        try:
            import update_scoreboard

            update_scoreboard.main([])
            out["steps_completed"].append("scoreboard")
            _log("scoreboard refreshed")
        except SystemExit:
            out["steps_completed"].append("scoreboard")  # argparse-style clean exit
        except Exception as e:  # noqa: BLE001
            out["steps_failed"].append(f"scoreboard: {e}")
            _log(f"scoreboard FAILED: {e}")

    # 4) Auto-tuning learning (adopt --auto overrides past the evidence gate)
    if learning:
        try:
            from qmbp_simulation.analysis.auto_tuning import analyze_override_log, read_override_log

            if read_override_log():
                s = analyze_override_log(write=True)
                n_adopted = sum(1 for reg in s.values() for d in reg.values() if d.get("adopted"))
                out["learning"] = {"n_adopted_rules": n_adopted}
                out["steps_completed"].append("learning")
                _log(f"learning: {n_adopted} rule(s) adopted")
            else:
                out["steps_completed"].append("learning (no evidence)")
                _log("learning: no override evidence yet")
        except Exception as e:  # noqa: BLE001
            out["steps_failed"].append(f"learning: {e}")
            _log(f"learning FAILED: {e}")

    _log(f"done: {len(out['steps_completed'])} ok, {len(out['steps_failed'])} failed")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Study-level post-run sync (vl_vs_hva)")
    p.add_argument("--no-coherence", action="store_true")
    p.add_argument("--no-summaries", action="store_true")
    p.add_argument("--no-scoreboard", action="store_true")
    p.add_argument("--no-learning", action="store_true")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    r = run_study_sync(
        coherence=not a.no_coherence, summaries=not a.no_summaries,
        scoreboard=not a.no_scoreboard, learning=not a.no_learning,
        verbose=not a.quiet)
    return 0 if not r["steps_failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
