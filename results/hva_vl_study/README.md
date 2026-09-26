# vl_vs_hva Study — Results Taxonomy, Naming & Tagging

This tree holds every artifact produced by the HVA-vs-VL study: partial
checkpoints, caches, and final results. It is organized, consistently named,
tagged with structured metadata, and discoverable via a machine- and
human-readable index. **Nothing is deleted** — superseded artifacts are marked
`deprecated` and kept.

## Directory taxonomy

```text
results/hva_vl_study/
  reports/                     # narrative .md reports (REPORT_*, SUMMARY)
  runs/<experiment>/<run_id>/  # single-point runs (one folder per run)
  sweeps/<experiment>/<run_id>/# multi-point aggregates (h-sweeps, multi-seed)
  <experiment>/                # active writer subfolders (scripts write here)
    circuits/                  # bound-circuit QPY for that experiment
  resources/n{N}_frustrated/   # reusable θ NPZ, bound QPY, per-hop checkpoints
  cache/                       # study-local caches (or pointers to data/*)
  index.json                   # machine-readable catalog of all artifacts
  INDEX.md                     # human-readable catalog (generated from index.json)
  README.md                    # this file
```

Rationale for the **hybrid `runs/` + `sweeps/`** split: a single JSON in this
study is often a *multi-point aggregate* (a whole h-sweep or an 8-seed
confirmation), which does not fit a one-run-per-folder model. Single-point
results (`run_hva_metropolis_point`, `run_n18_second_order` at one h) go under
`runs/`; aggregates go under `sweeps/`.

The already-organized writer subfolders (`circuit_comparison/`, `resources/`,
`vl_mps_n18/`, `vl_characterization/`, …) are where the scripts actively write
via `study_dir(name)`. They are **tagged in place** (a `.meta.json` sidecar per
artifact) rather than moved, so no write path breaks. New runs adopt the
`runs/`+`sweeps/` layout via `study_artifact_writer(...)`.

## Filename scheme

One parseable scheme for every artifact, via
`build_artifact_name(...)` (in `qmbp_simulation.framework.study_artifacts`):

```text
{model}_{topo}_N{n}_p{p}_h{h:.2f}_J2{j2:.2f}_{method}_{kind}.{ext}
```

- `h` is always 2 decimals (`h0.50`); `N{n}_p{p}` follow the repo conventions.
- `h` is omitted for artifacts that span an h-sweep.
- `kind` ∈ `{final, partial, theta, circuit, analysis, figure, cache}`.
- Example: `tfim_frustrated_square_N9_p2_h0.50_J20.50_metropolis_final.json`.

Never hand-roll names — call `build_artifact_name` so every artifact is
consistent and machine-parseable.

## Status: partial | final | deprecated

Status is tracked at the **file and catalog level**, not only via an internal
`done` flag:

| status | meaning |
|--------|---------|
| `partial` | a crash-safe checkpoint of an in-progress run (per-hop / per-seed) |
| `final` | a completed result; a run flips `partial → final` atomically on completion |
| `deprecated` | a superseded / overturned artifact, kept for the audit trail |

Transitions are validated (`promote_status`): `partial → final → deprecated`;
`final → partial` is rejected. The index infers status for legacy files from
`done` / explicit `meta.status` / known deprecated markers.

## Metadata tag block

Every JSON carries a `meta` block; every binary artifact (NPZ/QPY/PNG) gets a
`.meta.json` sidecar. Schema `hva_vl_study_meta_v1`:

```json
{
  "schema": "hva_vl_study_meta_v1",
  "status": "final",
  "experiment": "hva_nnn_sweep",
  "run_id": "...",
  "created_utc": "...", "updated_utc": "...",
  "source_script": "scripts/analysis/vl_vs_hva/...py",
  "git_commit": "abc1234",
  "physics":    {"model", "topo", "N", "p", "h", "J2", "gap", "e0"},
  "method":     {"optimizer", "sigma", "maxiter", "seeds", "n_hops", "strategy"},
  "metrics":    {"fidelity", "abs_error", "de_over_gap", "cx", "depth_2q"},
  "provenance": {"input files", "cache_keys", "parent run_id if resumed"},
  "artifacts":  {"theta": "...", "circuit": "...", "figure": "..."}
}
```

## The index

- `index.json` — the single catalog: one entry per artifact with its tag block,
  status, path, and key metrics, plus the `migration_map` (old → new paths).
- `INDEX.md` — generated from `index.json`; aligned tables grouped by
  experiment (status, N, p, h, method, fidelity, dE/gap, CX, points, path).

Rebuild both from disk (idempotent) and query the catalog:

```bash
.venv/bin/python scripts/analysis/vl_vs_hva/organize_results.py reindex
.venv/bin/python scripts/analysis/vl_vs_hva/organize_results.py query --status final --n 9
.venv/bin/python scripts/analysis/vl_vs_hva/organize_results.py query --method metropolis
```

## Legacy path resolution (shim)

Files that moved during the migration are recorded in
`index.json → migration_map`. Readers that still reference old paths resolve
them via `PathShim`:

```python
import json
from qmbp_simulation.framework.study_artifacts import PathShim

idx = json.load(open("results/hva_vl_study/index.json"))
shim = PathShim(idx["migration_map"], root="<repo-root>")
current = shim.resolve("results/hva_vl_study/n9_restart_schemes_frustrated.json")
# -> .../sweeps/warmstart_heuristics/n9_restart_schemes_frustrated/n9_restart_schemes_frustrated.json
```

## Caches

The `GroundTruthCache` (`data/ground_truth_cache.json`, key
`topo|N|model|h:.2f`) and the VL cache (`data/vl_cache/`) are first-class reuse
artifacts. Validate key hygiene (read-only) without ever mutating energies:

```bash
.venv/bin/python scripts/analysis/vl_vs_hva/organize_results.py validate-cache
# --fix reformats WRONG-PRECISION KEYS only (never energies) and refuses collisions
```

## How to produce a correctly-organized artifact from a new run

Use the `StudyRunner` + `StudyCheckpoint` + the study service's
`study_artifact_writer` / `save_circuit_qpy`. Example (see
`run_n18_second_order.py`):

```python
from hva_vl_study_common import save_circuit_qpy, save_json, study_dir
from qmbp_simulation.framework.study_checkpoint import StudyCheckpoint
from qmbp_simulation.framework.study_runner import StudyRunner

runner = StudyRunner(topology="square", n_qubits=18, p_layers=2, j2=0.5,
                     model="tfim_frustrated", strategy="second_order",  # regime-gated
                     maxiter=150, ansatz="nnn")
ckpt = StudyCheckpoint("per_h", writer=my_save_json_writer, path=out_path)
runner.run_sweep(h_values, ckpt, artifact_dir=study_dir("resources") / "n18_frustrated",
                 extra={"schema": "..._v1"}, qpy_saver=save_circuit_qpy)
```

Then refresh the catalog with `organize_results.py reindex`. New runs are
tagged and discoverable automatically; migration is only needed once, for
legacy files.

## Migration

The one-time migration is reversible (move-not-delete; every move is in
`migration_map`). It skips folders a live experiment is writing to. To preview
or run it:

```bash
.venv/bin/python scripts/analysis/vl_vs_hva/organize_results.py migrate           # dry-run
.venv/bin/python scripts/analysis/vl_vs_hva/organize_results.py migrate --apply    # execute
```
