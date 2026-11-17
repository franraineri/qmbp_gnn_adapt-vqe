# Binnacle — Haiqu Vector Loading on the 2D frustrated square TFIM

> Study of how well Haiqu data loading (`vector_loading`, `mps_loading`) prepares
> the ground state of the frustrated square-lattice TFIM, measured by fidelity,
> energy error `|ΔE|` and circuit cost. Includes the discovery and fix of a
> `ClassicalSolver` DMRG bug (NNN/J2 terms dropped at large N).
>
> **Work dates**: 2026-09-22 → 2026-09-25 (committed 2026-09-26)
> **Closure date**: 2026-11-16
> **Status**: DONE for N = 9, 10, 18 at L8/F50. PENDING (blocked by Haiqu credits):
> F200, small `truncation_cutoff` and Vidal-form runs at N = 18.

---

## 1. Read this first

- Model: `tfim_frustrated`, `J2 = 0.5`, `J = 1.0`, lattice `square`.
- VL = Haiqu **state-preparation synthesizer**, not a solver. It needs the ground
  state (dense vector, or an MPS) as input. Its native error is infidelity `1 − F`.
- `|ΔE| = |⟨ψ_VL|H|ψ_VL⟩ − E0|` is computed **locally** from the loaded circuit.
  Only obtaining the circuit needs the Haiqu cloud.
- Narrative report with all tables:
  `results/hva_vl_study/reports/REPORT_VL_square_frustrated.md`.
- Study-wide artifact conventions: `results/hva_vl_study/README.md` (taxonomy,
  naming, status, sidecars) and `results/hva_vl_study/INDEX.md` (catalog).
- The study tree also holds HVA-vs-VL and warm-start work from other sessions.
  This binnacle covers **only the VL side** plus the solver fix.

## 2. Results in one table

All fidelities are exact overlaps with the exact (N ≤ 10) or DMRG-MPS (N = 18)
reference. `E0` for N = 18 is the DMRG energy, not exact diagonalization.

| N   | h    |    gap | config (layers/ft) | loader         |      F | \|ΔE\| | 2q gates |
|-----|------|-------:|--------------------|----------------|-------:|-------:|---------:|
| 9   | 0.50 | 0.2238 | L2 / F20 (default) | vector_loading | 0.8086 | 0.7165 |       30 |
| 9   | 1.00 | 0.8601 | L2 / F20 (default) | vector_loading | 0.8999 | 0.5399 |       30 |
| 9   | 2.00 | 2.6252 | L2 / F20 (default) | vector_loading | 0.9613 | 0.3887 |       30 |
| 9   | 3.00 | 4.5394 | L2 / F20 (default) | vector_loading | 0.9870 | 0.1721 |       30 |
| 10  | 1.00 | 0.5023 | L2 / F20 (default) | vector_loading | 0.8966 | 0.6180 |       34 |
| 9   | 0.50 | 0.2238 | L8 / F50 (best)    | vector_loading | 0.9431 | 0.2750 |      120 |
| 9   | 1.00 | 0.8601 | L8 / F50 (best)    | vector_loading | 0.9754 | 0.1981 |      120 |
| 18  | 0.50 |    n/a | L8 / F50           | mps_loading    | 0.8625 | 0.5382 |      264 |
| 18  | 1.00 |    n/a | L8 / F50           | mps_loading    | 0.7924 | 1.3939 |      264 |

Conclusions supported by the data:

1. At default settings fidelity is **gap-limited**: lowest at small gap
   (frustrated/critical regime), 0.987 in the paramagnetic limit.
2. The small-gap drop is a **configuration limit**: raising `num_layers` to 8 and
   `fine_tuning_iterations` to 50 gives F = 0.94–0.98 at N = 9. Cost: 4× the 2q
   gates (30 → 120).
3. `fine_tuning_iterations` 20 → 50 gains about 2–3 pp of F at **no 2q-gate cost**;
   `num_layers` is the strong lever and doubles 2q gates per step (30/60/120).
4. At N = 18 with the same L8/F50, F drops to 0.79–0.86 and 2q gates reach 264.
   The input MPS is faithful (χ = 64 vs χ = 256 energy differs ≤ 1.5e-3), so the
   loss is in the MPS → circuit loading.
5. Caveat: the N = 9 vs N = 18 comparison mixes two loaders (`vector_loading` vs
   `mps_loading`), so part of the drop may be loader-specific, not size alone.

Triangular `h = 1.0` (quality sweep only, not in the report):
L2/F20 F = 0.9060 → L8/F50 F = 0.9824, `|ΔE|` 0.5089 → 0.1633.
Data: `vl_quality_sweep/vl_quality_sweep_frustrated.json`.

## 3. Chronology

| Phase | What was done                                                                                                     |
|-------|-------------------------------------------------------------------------------------------------------------------|
| 1     | VL characterization at N = 9/10 (default settings), shared helper `hva_vl_study_common`, results organized        |
| 2     | Quality-knob sweep `num_layers {2,4,8}` × `fine_tuning {20,50}` on small-gap points, 18 jobs                      |
| 3     | `\|ΔE\|` column: `circuit_energy`, `return_hamiltonian`; missing points re-run; consolidated h-sweep JSON rebuilt |
| 4     | Persistence: EvalCache-based result cache, `--force-recompute`, local QPY circuits, `circuit_id`, `vl_meta`       |
| 5     | N = 18 via `mps_loading`: general `mps_extraction` module, `run_vl_mps_job`, `vl_mps_large_n.py`                  |
| 6     | Solver bug found while validating N = 18 (see section 5); fixed with auto-detection and tests                     |
| 7     | F200 / `truncation_cutoff` 1e-8 / Vidal form implemented; runs blocked by Haiqu credits (HTTP 402)                |

## 4. Code map

Scripts live in `scripts/analysis/vl_vs_hva/` and import each other as siblings
(run with the file path, or `PYTHONPATH=scripts/analysis/vl_vs_hva`).

| File                                | Role                                                                                                                                                                                          |
|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `hva_vl_study_common.py`            | Shared helpers: `exact_ground_state_vector(..., return_hamiltonian)`, `circuit_energy`, `run_vl_job`, `run_vl_mps_job`, `save_circuit_qpy` / `load_circuit_qpy`, `haiqu_session`, `save_json` |
| `mps_extraction.py`                 | General (no Haiqu): `dmrg_mps_capped`, `capped_mps_vidal`, `mps_to_statevector`, `mps_input_quality`                                                                                          |
| `vl_h_sweep_frustrated.py`          | Default-settings h-sweep (N = 9/10), calls `compare()`                                                                                                                                        |
| `compare_circuits_square_n10_p2.py` | Per-point VL circuit characterization, saves PNG, QPY, `\|ΔE\|`                                                                                                                               |
| `vl_quality_sweep_frustrated.py`    | Knob grid with result cache (`data/vl_cache/vl_quality_sweep.json`) and `--force-recompute`                                                                                                   |
| `vl_mps_large_n.py`                 | N = 18 route: DMRG (χ ≤ 64) → `mps_loading`; flags `--fine-tuning`, `--truncation-cutoff`, `--mps-form`                                                                                       |
| `vl_circuit_characterization.py`    | Earlier VL characterization (non-frustrated, heavy_hex/square/triangular)                                                                                                                     |
| `compare_hva_vl_mps_fidelity.py`    | VL-ideal vs GNN vs ansatz ceiling (moved here at closure; was broken outside this folder)                                                                                                     |

Solver fix (library code):

- `src/qmbp_simulation/solvers/classical.py`: `ClassicalSolver.solve(...,
  dmrg_from_operator=None)`, `_solve_dmrg_from_operator`, `_pauli_op_to_tenpy_terms`.
- Tests: `tests/unit/test_guards_and_validations.py::TestDMRGCrossValidation`
  (4 regression tests added).

## 5. Bugs found and fixed

| Bug                                                                                      | Impact                                                               | Fix                                                                                             |
|------------------------------------------------------------------------------------------|----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| `ClassicalSolver` DMRG built the MPO from `lattice.edges` only, dropping NNN/J2 (N > 16) | Frustrated energy wrong: N = 18 h = 1.0 gave -30.2355, true -23.0280 | Operator-based MPO from the `SparsePauliOp`; auto-detect by ZZ-terms > edges; warning on legacy |
| Client poll budget equal to Haiqu `max_time` (900 s)                                     | F200 jobs left at status `RUNNING`, `F = None`                       | `max_time` passed explicitly; poll budget defaults to `max_time + 300 s`                        |
| Partial re-run overwrote `vl_h_sweep_frustrated.json`                                    | Consolidated sweep lost rows                                         | Rebuilt from per-point JSONs; `\|ΔE\|` for h = 0.5/1.0 backfilled from the quality sweep        |
| Script left outside the reorganized folder                                               | `compare_hva_vl_mps_fidelity.py` raised `ModuleNotFoundError`        | Moved to `vl_vs_hva/` at closure                                                                |

Data audit after the solver fix: **no** frustrated entries with N > 16 existed in
the ground-truth cache or the training NPZs, and every VL `|ΔE|` used exact
diagonalization (N ≤ 10) or the operator-based DMRG (N = 18). No stored result
was contaminated.

## 6. Verification state at closure (2026-11-16)

- 43 tests pass: `test_guards_and_validations`, `test_solvers`,
  `test_solver_phase1`, `test_prop_solvers`.
- All 7 VL scripts import from `scripts/analysis/vl_vs_hva/`; `--help` verified
  for the three commands in the report.
- `mps_extraction` Pauli → TeNPy conversion verified exact at N = 9
  (E = -11.1802, fidelity 1.0). Vidal form verified at N = 9 (fidelity 1.0000000000
  against the standard form).
- **No unit tests** exist for `mps_extraction`, `run_vl_job` or `run_vl_mps_job`.

## 7. How to resume

Requires `HAIQU_API_KEY` in the environment (never commit it) and **Haiqu credits**
(the account hit `HTTP 402 Payment Required`).

```bash
# Pending: the N = 18 experiments blocked by credits (standard, then Vidal form)
HAIQU_API_KEY=... .venv/bin/python scripts/analysis/vl_vs_hva/vl_mps_large_n.py \
    --topology square --n 18 --j2 0.5 --h 0.5 1.0 --num-layers 8 \
    --fine-tuning 200 --truncation-cutoff 1e-8 --mps-form standard
HAIQU_API_KEY=... .venv/bin/python scripts/analysis/vl_vs_hva/vl_mps_large_n.py \
    --topology square --n 18 --j2 0.5 --h 0.5 1.0 --num-layers 8 \
    --fine-tuning 200 --truncation-cutoff 1e-8 --mps-form vidal
```

Compare against the baseline `vl_mps_n18/vl_mps_square_N18.json` (F50, standard).
After running, refresh the catalog:
`.venv/bin/python scripts/analysis/vl_vs_hva/organize_results.py reindex`.

Re-analysis without Haiqu: `load_circuit_qpy(path)` then `circuit_stats`,
`transpile_hw` or `circuit_energy` (all local).

## 8. Data map

| Path (under `results/hva_vl_study/`)                                      | Content                                                 | Status     |
|---------------------------------------------------------------------------|---------------------------------------------------------|------------|
| `reports/REPORT_VL_square_frustrated.md`                                  | Narrative report with all tables                        | final      |
| `circuit_comparison/tfim_frustrated_square_*_analysis.json` + `circuits/` | Default-setting per-point stats and QPY (N = 9/10)      | final      |
| `vl_h_sweep/vl_h_sweep_frustrated.json`                                   | Consolidated default-settings sweep (5 square rows)     | final      |
| `vl_quality_sweep/vl_quality_sweep_frustrated.json`                       | 18-config grid (square h = 0.5/1.0, triangular h = 1.0) | final      |
| `vl_mps_n18/vl_mps_square_N18.json` + `circuits/`                         | N = 18 baseline, L8/F50, standard form                  | final      |
| `vl_mps_n18/vl_mps_square_N18_L8_F200_tc1e-08_standard.json`              | Failed attempt, HTTP 402 only, no measurements          | deprecated |
| `data/vl_cache/vl_quality_sweep.json` (repo root)                         | Result cache for the quality sweep (54 entries)         | cache      |

## 9. Pitfalls for the next person or agent

1. **Cached runs skip Haiqu.** A normal re-run of the quality sweep prints
   `[cached]` and produces no new circuits. Use `--force-recompute` to refresh.
2. **`vl_meta` is only populated in runs after 2026-09-24** (the N = 18 runs).
   Haiqu returned `job_time` but `circuit_metrics` / `circuit_analytics` were
   `None` for `mps_loading`. Older rows have no `vl_meta`.
3. **VL is stochastic.** The same config can change the 4th decimal of F between
   runs (e.g. h = 3.0: 0.9868 vs 0.9870). Every point is a single run; there are no
   repeats or error bars.
4. **Use the DMRG energy as the N = 18 reference.** Do not trust
   `ClassicalSolver` energies for frustrated models before the fix (`6cf7465`).
5. **`gap` at N > 18 is a TFIM analytical estimate** in the solver
   (`_estimate_tfim_gap_analytical`) and ignores frustration. Not validated for
   frustrated models. N = 18 itself uses exact `eigsh`.
6. **TeNPy `trunc_err` reads 0 when the bond saturates `chi_max`.** Use the
   energy-vs-higher-χ proxy in `mps_input_quality` instead.
7. `h` keys use 2 decimals (`:.2f`); GT cache for frustrated models carries
   `model_kwargs={"J2": ...}` so it never collides with `J2 = 0`.
8. `mps_loading` jobs at N = 18 take about 660–720 s at L8/F50; F200 reaches the
   900 s compute budget.
9. The working tree at closure also contains uncommitted work from **other
   sessions** (ansatz specs, `src/qmbp_simulation/circuits/bond_mask.py` and `hva.py`, warm-start tests, removed `.sh`
   queues). It is unrelated to this study; do not bundle it with VL commits.

## 10. Open items and technical debt

| Item                                                                                                                                                                                | Type        |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| Run F200, `truncation_cutoff` 1e-8 and Vidal form at N = 18 (needs credits)                                                                                                         | Experiment  |
| Triangular: `\|ΔE\|` and QPY only in the quality sweep; default-settings table missing                                                                                              | Experiment  |
| Sweep `truncation_cutoff`; call `vector_loading_estimates` to log cost/time                                                                                                         | Experiment  |
| Repeats or seeds to quantify VL stochasticity                                                                                                                                       | Experiment  |
| N-scaling with one loader (e.g. `mps_loading` at N = 9–16) to separate size from loader effects                                                                                     | Experiment  |
| Unit tests for `mps_extraction`, `run_vl_job`, `run_vl_mps_job` (scripts are not importable from tests by repo rule)                                                                | Tech debt   |
| Pauli → TeNPy conversion duplicated: `src/qmbp_simulation/solvers/classical.py::_pauli_op_to_tenpy_terms` and `scripts/analysis/vl_vs_hva/mps_extraction.py::_pauli_op_to_termlist` | Tech debt   |
| Consider moving the MPS extraction core to `src/qmbp_simulation/` (reuse-first rule) so it can be tested                                                                            | Tech debt   |
| Validate the solver gap for frustrated models at N > 18                                                                                                                             | Risk        |
| Rotate the Haiqu API key (it appeared in a chat transcript) and top up credits                                                                                                      | Operational |
| Pre-existing ruff errors (counts at closure): 1 in `src/qmbp_simulation/solvers/classical.py`, 2 in `tests/unit/test_guards_and_validations.py`                                     | Style       |

## 11. Related documents

- `results/hva_vl_study/reports/REPORT_VL_square_frustrated.md`
- `results/hva_vl_study/README.md`, `results/hva_vl_study/INDEX.md`
- `.kiro/knowledge/validated-decisions.md` (section "Haiqu Data Loading")
- `.kiro/knowledge/error-patterns.md` (DMRG NNN/J2 pattern)
- `.kiro/steering/haiqu-integration.md` (section "Data loading")
- `internal/documentation/next-steps/next_steps.md`
- Agent memory topic keys: `vl-square-frustrated-report`, `vl-quality-sweep-cache`,
  `vl-circuit-persistence-qpy`, `vl-hsweep-consolidation-metrics`,
  `vl-mps-loading-n18`, `vl-mps-n18-f200-vidal`, `classical-solver-nnn-bug`.
