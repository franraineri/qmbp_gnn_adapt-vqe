---
name: study-runner-helpers
description: Shared helpers for vl_vs_hva study runners (converge_circuit, build_variant_row, StudyPersister, sync_scoreboard, bond-selection on arbitrary structures). Use when writing or editing any scripts/analysis/vl_vs_hva/run_*.py so the result-dict, optimize wrapper, per-bond weights, and crash-safe persistence are reused — never re-implemented.
---

# Study Runner Helpers (anti-duplication)

Every `scripts/analysis/vl_vs_hva/run_*.py` runner shares four concerns that
were copy-pasted and silently diverged (`de_gap` was missing in some result
rows). They are now centralized. **When writing or editing a study runner,
REUSE these — never inline the result dict, the optimize wrapper, the per-bond
weight aggregation, or the persist/on_restart boilerplate.**

## The helpers

| Need | Reuse (never inline) |
|------|----------------------|
| Converge one circuit (cost/fid build + multi-restart best-of) | `study_core.converge_circuit(qc, H, psi, *, restarts, maxiter, seed0=0, warm_theta=None, on_restart=None, backend=None)` → `(fid, e, runs)` |
| The per-variant result dict | `study_core.build_variant_row(name, fid, e_best, runs, *, n_2q, n_params, e0, gap, blocks=None, rx_final=None, bond_selection=None, seed_kind=None, seconds=None, **extra)` |
| Crash-safe partial+final persist | `hva_vl_study_common.StudyPersister(subdir, out_file, fingerprint, extra, params, description, rows_ref, total_restarts)` → `.persist(rows, status)`, `.restart_callback(label, n2q, npar)` |
| Post-run scoreboard refresh | `hva_vl_study_common.sync_scoreboard()` at end of `run()`, gated by a `--no-sync` flag (fire-and-forget subprocess, idempotent) |
| Per-bond \|θ\| for standard `[nn,nnn,x]*p` | `bond_mask.bond_weights_from_theta(theta, n_nn, n_nnn, n_qubits, p)` |
| Per-bond \|θ\| for arbitrary structure (half_nn_rx …) | `bond_mask.bond_weights_for_blocks(theta, blocks, n_nn, n_nnn, n_qubits, rx_final=, rz_final=)` |
| θ → BondSelection (T1 prune / T2 top-k) on any structure | `bond_mask.selection_from_variant_theta(theta, blocks, nn_edges, nnn_edges, n_qubits, *, rx_final=, method='top_k'|'prune', keep_frac=, tol=)` |
| Save a portable, COMPLETE ansatz definition (exact edges + θ) | `hva_vl_study_common.save_ansatz_spec(variant, topology=, n_qubits=, h=, j2=, theta=, nn_edges=, nnn_edges=, **metrics)` → `ansatz_specs/<name>.spec.json` |
| Reuse a saved ansatz in any runner | `AnsatzSpec.load(path).build()` → `(qc, theta)` (from `circuits.ansatz_spec`) |

## Canonical runner skeleton

```python
from hva_vl_study_common import StudyPersister, sync_scoreboard
from qmbp_simulation.execution import NoiselessBackend
from qmbp_simulation.framework.study_core import (
    build_variant_row, converge_circuit, cx_and_params, ground_state,
)

def run(args) -> int:
    lat, _qc, H, psi, e0, gap, n_nn, n_nnn = ground_state(
        args.topology, args.n, args.h, args.j2, 1)
    backend = NoiselessBackend()           # ONE per sweep (reuses cached dense H)
    rows: list[dict] = []
    persister = StudyPersister(
        subdir="bond_ablation", out_file=out_file,
        fingerprint={"n_qubits": args.n, "p_layers": None,
                     "model": "tfim_frustrated", "topology": args.topology, "h": args.h},
        extra={"e0": e0, "gap": gap, "schema": "my_v1"},
        params={"experiment": "my_exp", "N": args.n, "h": args.h},
        description="…", rows_ref=rows, total_restarts=args.restarts)

    qc, _ = build_variant(builder, args.n, lat, variant)
    n_2q, npar = cx_and_params(qc)
    fid, e_best, runs = converge_circuit(
        qc, H, psi, restarts=args.restarts, maxiter=args.maxiter, seed0=args.seed0,
        warm_theta=seed, backend=backend,
        on_restart=persister.restart_callback(variant.name, n_2q, npar))
    rows.append(build_variant_row(
        variant.name, fid, e_best, runs, n_2q=n_2q, n_params=npar, e0=e0, gap=gap,
        blocks=variant.blocks, rx_final=variant.rx_final,
        bond_selection=variant.bond_selection, seconds=secs))
    persister.persist(rows, status="final")
    if not args.no_sync:
        sync_scoreboard()
    return 0
```

## Rules

- **Every converged ansatz MUST be saved as an `AnsatzSpec`** via
  `save_ansatz_spec(...)` — the portable complete definition (system + structure
  + EXACT nn/nnn edges + converged θ + metrics) in one self-contained JSON under
  `results/hva_vl_study/ansatz_specs/`. Artifacts that store only bond COUNTS are
  NOT replicable; the spec is. Reuse in any runner with
  `AnsatzSpec.load(path).build()` → `(qc, theta)`.
- **Warm-start is MANDATORY via `prepare_warmstart` for EVERY converged circuit**
  — the reference/full AND every masked/pruned variant. Never converge from a
  cold θ=0 start or a bare analytic seed. `prepare_warmstart` runs the full
  cascade (calibrated + regime + transferred donors + micro-descent) and returns
  the best seed; pass its `["seed"]` as `converge_circuit(..., warm_theta=...)`
  and persist its `["provenance"]` as the row's `seed_kind`. For a non-standard
  structure (half-layer, rx_final), inject the structure-aware seed(s) as
  `extra_candidates` and pass `target_len=qc.num_parameters` so they survive the
  length filter. This keeps every row's warm-start homogeneous and comparable.
- Create ONE `NoiselessBackend()` per sweep and pass it to every
  `converge_circuit(..., backend=backend)` — it reuses the cached dense
  Hamiltonian across variants instead of rebuilding per circuit.
- To prune/subset bonds on a structure variant, use
  `selection_from_variant_theta`, NOT hand-sliced θ assuming the uniform-layer
  offset (asymmetric layouts like `p2_half_nn_rx` have nn appearing more times
  than nnn).
- A runner whose in-progress row needs extra fields may keep its own
  `on_restart` closure, but it MUST persist through `StudyPersister.persist`
  (not a hand-rolled `build_resumable_payload` + `save_json`).
- Detect new duplication by searching for repeated helper names across runners
  (`_bond_weights`, `_optimize`, inline result dicts, inline
  `build_resumable_payload` wrappers). If two runners share it, it belongs in
  `study_core` / `bond_mask` / `hva_vl_study_common`.

## Tests

- `tests/unit/test_study_core.py`: `TestBuildVariantRow`, `TestConvergeCircuit`
- `tests/unit/test_bond_mask.py`: `TestBondWeightsFromTheta`,
  `TestBondWeightsForBlocks`, `TestSelectionFromVariantTheta`
