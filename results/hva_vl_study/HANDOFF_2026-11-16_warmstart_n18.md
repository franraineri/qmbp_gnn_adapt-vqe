# Handoff: warm-start cascade and N=18 compression study

Closure snapshot taken on 2026-11-16. It is written so that a person or an AI agent with no memory of the sessions can continue the work. Every statement was checked against the repository at closure, except lines marked "not re-verified".

- Repository: `qmbp_gnn_adapt-vqe`, branch `develop`, HEAD `b5cb8d6` (equal to `origin/develop`).
- Model and regime: frustrated square TFIM (`tfim_frustrated`, `J = 1`, `J2 = 0.5`) at `h = 0.5`, next to the transition (gap about 0.006 at N=18).
- Questions answered: how few 2-qubit gates a bond-resolved HVA needs for a given ground-state fidelity at N=10..18, and how much the warm-start helps the optimizer.
- Reported fidelities are expressivity ceilings: noiseless statevector, multi-restart L-BFGS-B, best of restarts by energy.

## 1. Read this first

- The code from these sessions is committed (HEAD). The two audit scripts and this file were added at closure and are not committed yet. Until this file, the narrative (results, decisions, open work) lived only in a local memory store that other agents cannot read.
- Two features are implemented but not validated on a real landscape, and are off by default: the two-pass selector and `warm_restart_full`. One audit is unfinished: the N=18 donor audit. See section 7.
- One earlier conclusion was wrong and is retracted in section 5.4. Do not reuse the "budget was the confound" claim.
- The working tree holds uncommitted symmetry work by someone else (section 9). Do not mix it with this work when committing.
- Fast test gate (verified, about 10 s): see section 10.

## 2. State at closure

| Area                                            | Status                                  | Evidence                                     |
|-------------------------------------------------|-----------------------------------------|----------------------------------------------|
| Geometric cross-N donor matching                | Done, default for grid topologies       | in HEAD, `TestGeometricCrossNMatching`       |
| Cascade efficiency (dedup, per-degree θ_x fill) | Done                                    | in HEAD, `TestCascadeEfficiencyImprovements` |
| Two-pass selector (short pre-rank, then full)   | Implemented, off by default             | tested on stubs and an N=6 run only          |
| Short-budget helper `short_descent_budget`      | Done                                    | `TestShortDescentBudget`                     |
| Analytic seed caches (M5)                       | Done, always on                         | `TestAnalyticSeedCache`                      |
| `warm_restart_full` option                      | Implemented, off, not validated         | stub A/B only                                |
| nn-dominant variants (`p2/p3_nn_dom*`)          | Experiment finished, hypothesis refuted | artifacts in `bond_ablation/`, section 3     |
| N=18 Pareto frontier                            | Done and documented                     | [report][report-bond-selection]              |
| Warm-start vs random audit                      | N=14 done (n=1); N=18 not finished      | section 5.4                                  |
| Classical simulation limits                     | Analytic and extrapolated               | section 6; nothing measured beyond N=18      |
| Better restart points and N=16 donor            | Plan only, nothing implemented          | section 7                                    |
| ADAPT growth (pool A and B, gates 0 to 2)       | Exists in the repo, not reviewed here   | section 9; 28 artifacts in the catalog       |

Test status at closure, on the current working tree: 345 tests pass in 5 files (`test_warmstart_service` 176, `test_study_core` 85, `test_bond_mask` 56, `test_ansatz_spec` 13, `test_hva_session_infra` 15), no failures. Full run takes about 3.5 minutes because of 24 slow tests that are not part of this work (section 7).

## 3. N=18 compression results (h = 0.5)

Source artifacts: `results/hva_vl_study/bond_ablation/variant_topk_*_N18_h0.50.json` and `ansatz_specs/*N18_h0.50.spec.json`, all tracked in git. Full discussion: [report][report-bond-selection], section "Scaling to N = 18".

Pareto frontier (a point stays if no other point has at least its fidelity with at most its 2q count):

| 2q (CX) | Fidelity | Variant                | Note                                       |
|:-------:|:--------:|------------------------|--------------------------------------------|
|   128   |  0.8202  | `p2_half_nn_rx_topk8`  | cheap, low fidelity                        |
|   148   |  0.8483  | `p2_half_nn_rx_topk13` |                                            |
|   276   |  0.8903  | `p3_half_nn_rx_topk10` |                                            |
|   318   |  0.9084  | `p2_half_nn_rx_full`   | cheapest point at or above 0.90            |
|   336   |  0.9359  | `p3_half_nn_rx_topk20` | best efficiency, 25% fewer gates than full |
|   390   |  0.9515  | `p3_half_nn_rx_topk29` | near full fidelity                         |
|   450   |  0.9516  | `p3_half_nn_rx_full`   | fidelity ceiling                           |

Number of nnn layers (p3, nn backbone fixed at 4 layers):

| Variant         | nnn layers | 2q (CX) | Fidelity | Change vs previous |
|-----------------|:----------:|:-------:|:--------:|:------------------:|
| `p3_nn_dom_rx`  |     1      |   294   |  0.8385  |         -          |
| `p3_nn_dom2_rx` |     2      |   372   |  0.9252  |       +0.087       |
| `p3_half_nn_rx` |     3      |   450   |  0.9516  |       +0.026       |

Conclusions:

- One nnn layer is a structural ceiling (0.8385; both restarts identical). `p2_nn_dom_rx` reaches only 0.7785.
- Two layers give 97% of the full fidelity with 78 fewer gates. The minimum useful count is about two.
- The nn-dominant design is Pareto-dominated: `p3_nn_dom2_rx` (0.9252 at 372) loses to `p3_half_nn_rx_topk20` (0.9359 at 336). Keep the full `half_nn_rx` structure and prune nnn by importance.
- Not meant as a recipe at other N or h: all numbers are for N=18, h=0.5, J2=0.5.

Size of the ansatz at larger N (built with the real circuit engine, transpiled to `rz, sx, x, cx`):

| N   | nn edges | nnn edges | `p2_half_nn_rx` CX / params | `p3_half_nn_rx` CX / params | Pareto-efficient point (extrapolated) |
|-----|:--------:|:---------:|:---------------------------:|:---------------------------:|:-------------------------------------:|
| 18  |    27    |     39    |          318 / 231          |          450 / 315          |             336 (measured)            |
| 24  |    38    |     59    |          464 / 328          |          658 / 449          |               about 486               |
| 30  |    49    |     78    |          606 / 423          |          860 / 580          |               about 636               |

The circuit grows linearly in N (about 25 to 29 CX per qubit). The extrapolated Pareto point assumes the same 51% nnn fraction that worked at N=18; it may need more nnn at larger N.

## 4. Warm-start cascade

Everything is pure numpy in [`warmstart.py`][warmstart] and is reached through `prepare_warmstart` in [`study_core.py`][study-core]. `best_combined_warmstart` builds candidates, then ranks them by the fidelity reached after a short L-BFGS-B descent (the basin they fall into), not by raw seed fidelity.

Candidates: calibrated analytic seed, structural seed, regime seed, caller-supplied extras (variant analytic seed, checkpoints), and transferred cross-N donors.

Changes made in these sessions (library paths are under `src/qmbp_simulation/`, scripts under `scripts/analysis/vl_vs_hva/`):

- `analysis/warmstart.py`
  - `lattice_coords`, `remap_edges_by_coords`, `_grid_degree`: geometric cross-N matching and per-degree θ_x fill.
  - Dedup of identical candidates (`_rep_of`), `descent_fn_short`, `two_pass_top_k`, `warm_restart_full`.
  - `short_descent_budget`, and the seed caches `cached_calibrated_warmstart_theta`, `cached_structural_warmstart_theta`, `cached_select_regime_seed`, `cached_variant_warmstart_theta`, `clear_analytic_seed_caches`.
- `framework/study_core.py`: `prepare_warmstart(two_pass, short_descent, two_pass_top_k, warm_restart_full)`. The result carries `_two_pass` and `_short_descent`.
- `circuits/hva_variants.py`: variants `p2_nn_dom_rx`, `p3_nn_dom_rx`, `p3_nn_dom2_rx`.
- `run_variant_topk.py`: flags `--two-pass`, `--two-pass-top-k`, `--micro-descent -1` (adaptive budget). It uses the cached variant seed.
- `audit_warmstart_vs_random.py`, `audit_warmstart_donors.py`: added at closure (section 10).

Defaults that matter:

- `two_pass=False`: the default descends every candidate fully (exhaustive). When on, a short descent (25% of the full budget, minimum 24 iterations) pre-ranks candidates and only the top `two_pass_top_k` get the full descent. The short pass prunes, it must not decide: see 5.1.
- `warm_restart_full=False`: when on, the full descent starts from the short-refined point. It is not decision-neutral: on a multi-basin landscape a different start can end in a different basin and change the winner. A stub test showed exactly that, which is why it is opt-in.
- Seed caches (M5) key on the full layout plus `h`, `J`, `J2` as exact float bits (`float.hex()`), plus each seed's own knob (`nnn_suppress`, `use_calibrated`, block list, `rx/rz` flags). They store and return copies, because the builders mutate arrays in place. The exact-bit key is deliberate: the `:.2f` rule in the h-precision convention applies to keys that cross with the ground-truth cache, and this cache is internal. Do not "fix" it to `:.2f`: that would alias `h = 0.501` and `0.502`.
- Note on M5: it removes repeated seed construction (2000 calls on one config: 123 ms to 18 ms), but a seed costs milliseconds and a descent costs minutes. Expect no visible end-to-end speed-up. It is safe, not a performance win.

## 5. Evidence about the warm-start

### 5.1 The ranking needs a long descent near h_c

Fidelity of each candidate after K descent iterations, `p2_half_nn_rx`, N=18, h=0.5 (the original log was lost; the protocol is reproducible with `audit_warmstart_donors.py --selector-budgets`, section 10):

| Candidate | K=100  | K=400  | K=800  |
|-----------|:------:|:------:|:------:|
| analytic  | 0.7708 | 0.8947 | 0.9060 |
| donor N8  | 0.8479 | 0.8883 | 0.8899 |
| donor N10 | 0.6529 | 0.9215 | 0.9244 |
| donor N14 | 0.8117 | 0.8989 | 0.9080 |

At K=100 the ranking picks donor N8; the real winner (donor N10) starts slowly and is only visible from K=400. This is the reason for the long adaptive budget (`micro_descent_budget`, about 400 at N=18 near h_c) and for using a short pass only to prune.

### 5.2 Coverage does not predict the final fidelity

Not re-verified at closure (from session notes). For `p2_half_nn_rx` at N=18: donor N8 (33% bond coverage) converged to 0.9094, donor N10 (43%) to 0.9247, donor N14 (72%) to 0.9088. The automatic choice (highest coverage) took N14, the worse one. Consequence: do not spend effort generating a donor at every intermediate N.

### 5.3 Geometric matching

Not re-verified at closure (from session notes). On a square lattice `cols = ceil(sqrt(N))` changes with N, so matching bonds by raw qubit index misaligns physical bonds (about 15% coverage N14 to N18). Matching by (row, col) gives about 74%. Converged `p2_half_nn_rx` at N=18: 0.79 with raw-index donor, about 0.92 with geometric matching, and roughly 30 times faster (24380 s to 720-1050 s).

### 5.4 Warm-start against random, and a retracted claim

Two audits, [`audit_warmstart_vs_random.py`][audit-random] and [`audit_warmstart_donors.py`][audit-donors]. Random starts are uniform in (-π, π), best of R by energy; the warm run is a single descent, so random gets more budget.

Standard p=1 bond-resolved ansatz, analytic cascade only, no donors:

| N   |  h   | maxiter |  warm  | random best-of | single random start |
|-----|:----:|:-------:|:------:|:--------------:|:-------------------:|
| 8   | 0.50 |   400   | 0.7072 |   0.7069 (4)   |        0.1473       |
| 10  | 1.00 |   400   | 0.8863 |   0.8861 (4)   |        0.0519       |
| 10  | 2.00 |   400   | 0.9889 |   0.9889 (4)   |        0.9754       |
| 10  | 0.50 |   2000  | 0.3586 |   0.3562 (8)   |          -          |
| 14  | 0.50 |   2000  | 0.2759 |   0.3101 (8)   |          -          |

`p2_half_nn_rx`, N=14, h=0.5, cascade with and without donors (N6, N8, N10), 3 random starts, same seeds:

| maxiter | analytic only |    cascade with donors    | random best-of-3 |
|:-------:|:-------------:|:-------------------------:|:----------------:|
|   400   |     0.9470    | 0.9632 (winner donor N10) |      0.9308      |
|   800   |     0.9497    | 0.9655 (winner donor N10) |      0.9167      |

What the data supports:

- Against a single cold start the analytic seed wins by +0.56 (N8) and +0.83 (N10, h=1.0), when the ansatz can represent the state.
- The p=1 ansatz cannot represent the state at h=0.5 for N of 10 or more: every method ends between 0.28 and 0.39, while `p2_half_nn_rx` reaches 0.995 at N=10 and the same h. Those rows say nothing about initialization quality.
- With an expressive ansatz, the order is the same at both budgets: cascade with donors, then analytic only, then random. The donor adds +0.016, the cascade beats random by +0.03 to +0.05.
- Caveats: one run per condition and one seed. The gaps are about as large as the spread between random starts (about 0.015). The 400 and 800 rows share seeds, so they are not independent evidence. Both warm runs reach the iteration cap, so they are not fully converged. Treat these as indications, and repeat with 5 or more seeds before quoting them.

Retraction: an earlier version of this analysis (and a local memory entry) said that the p=1 audits were "budget-starved at maxiter 400" and that 800 iterations revealed the warm-start advantage. That is false. The harder p=1 audit (N=10 and N=14 at h=0.5) already used maxiter 2000 and the warm seed had `nit = 0` (already a stationary point), so more budget changes nothing. The difference between the p=1 and the later results is the ansatz (p=1 against `p2_half_nn_rx`). The direct check above moves the budget from 400 to 800 on the same ansatz and the warm results change by 0.003 at most.

`nit = 0` means the seed is already a stationary point of its own basin (gradient about 1e-5). It does not mean the seed is a good basin.

## 6. Classical simulation limits (Apple M2, 8 cores, 24 GB)

Measured on this machine, `p3_half_nn_rx`, one cost plus adjoint-gradient evaluation: N=12 0.056 s, N=14 0.154 s, N=16 1.14 s, N=18 3.12 s. Beyond N=18 the table is extrapolated with a factor of 2.75 per two qubits (the N=14 to N=16 step was 7.4, so the real figure may be worse). Convergence assumes 800 iterations at 1.5 evaluations each. Working memory is taken as 4 times the state vector.

| N   | State memory | Working memory | One evaluation | Convergence (800 it) |   Fits in RAM    |
|-----|:------------:|:--------------:|:--------------:|:--------------------:|:----------------:|
| 18  |     4 MB     |    0.02 GB     |     3.1 s      |      about 1 h       |  yes (measured)  |
| 20  |    17 MB     |    0.07 GB     |     8.6 s      |     about 2.9 h      |       yes        |
| 22  |    67 MB     |    0.27 GB     |      24 s      |     about 7.9 h      |       yes        |
| 24  |    268 MB    |     1.1 GB     |      65 s      |      about 22 h      |    yes, slow     |
| 26  |    1.1 GB    |     4.3 GB     |     179 s      |      about 60 h      | yes, impractical |
| 28  |    4.3 GB    |     17 GB      |     491 s      |     about 164 h      |   at the limit   |
| 30  |    17 GB     |     69 GB      |     1351 s     |     about 450 h      |        no        |

- Statevector stops being practical at N=24 for time and at N=28 to 30 for memory.
- MPS memory is `N * chi^2 * 16` bytes: 1.2 / 1.6 / 2.0 MB for N = 18 / 24 / 30 at chi=64, and 19 / 25 / 32 MB at chi=256. Not benchmarked: MPS time per evaluation was not measured, and whether chi=64 captures the entanglement near h_c at N=24..30 is unknown.
- The reference state is the second limit: exact diagonalization of the ground state is `2^N` and was not timed here. Beyond the exact range, fidelity can only be measured against a DMRG state of capped bond dimension, which is a proxy.

## 7. Open items, in priority order

1. Validate the two-pass selector on real landscapes.
   - Why: it is implemented but only tested on stubs and one N=6 run; the savings are unmeasured.
   - How: run `run_variant_topk.py` with and without `--two-pass --micro-descent -1` at N=14, then N=18. Compare the winner (`seed_kind`), fidelity and wall time.
1. Finish the N=18 donor audit.
   - Why: the run was killed after 9 minutes while a hung `koda graphify` was using a core and 6 GB.
   - How: `audit_warmstart_donors.py --n-list 18 --maxiter 800 --random-starts 2`, in the background.
1. Repeat the N=14 audit with 5 or more seeds.
   - Why: the current gaps are about as large as the spread between random starts.
   - How: vary `--seed`.
1. Close the donor chain with an N=16 donor.
   - Why: the chain N6, N8, N10, N14 stops before N=18, and N14 (0.95) is a weaker donor than N10 (0.996).
   - How: generate a `p2_half_nn_rx` N=16, h=0.5 spec with `run_variant_topk.py`, then rerun the N=18 donor audit.
1. Read the existing ADAPT results before planning anything new.
   - Why: ADAPT is already implemented and was run after the planning in these sessions: `grow_adapt_pool` in `study_core.py` (action pool of "add nnn bonds", "repeat layer" and a zero-CX "thin" rotation layer, ranked by Δfidelity per Δ2q), runners `run_adapt_poolA.py`, `run_adapt_poolB.py` (operator list) and `run_adapt_gate0/1/2.py`. Gate 0 asks whether the gradient ranking is informative near h_c, Gate 1 compares ADAPT with top-k at equal 2q budget, Gate 2 is the N=18 production run. An ADAPT plan written in these sessions overlooked all of this.
   - How: list the 28 ADAPT entries in `index.json` (section 9), read the Gate 0 and Gate 1 outputs for N=10 and N=14, and decide from them. The N=18 pool runs are marked partial.
   - Idea still open after that review: seed `grow_adapt_pool` through its `warm_theta0` argument from the known top-k solution (20 of 39 nnn give 0.9359 at N=18), so the growth starts near the optimum instead of at the backbone.
1. A/B of `warm_restart_full`, with the same protocol as the two-pass validation. It is not decision-neutral and needs evidence before it is enabled.
1. Port the adaptive budget to the other runners. `run_ansatz_variants.py` and `run_bond_topk_regime.py` still default to `--micro-descent 60`. Copy the `-1` handling and the `--two-pass` flags from `run_variant_topk.py`.
1. Measure the classical limits of section 6: time exact diagonalization against DMRG, and one MPS evaluation, at N=20 to 24.
1. Mark the slow tests with `@pytest.mark.slow` (`TestGrowAdaptPool*`, about 18 s each; `TestSymmetryTiedBuilder`, 36 s). The default test run went from about 5 s to 3.5 minutes.
1. Third research axis: the user named three axes and described two. Ask which one was meant. Candidates discussed were a GNN or MPNN θ predictor and an anti-local-minima optimizer.

Rejected on evidence, do not retry without a new argument: ensemble (circular mean) of donors, block-mix of seeds from different sources, block coordinate descent, per-role averaging of θ (`rel_disp`), geometry-based bond importance, a donor at every intermediate N, AQC compression, transpiler optimization levels. Details are in [`README`][readme] and the [report][report-bond-selection].

## 8. Lessons and pitfalls

- `/tmp` is not durable. Every diagnostic script and log from the previous sessions was gone at closure. Keep scripts in the repo and results in a report.
- A hung `koda graphify` process (5 days, 78% CPU, 6 GB RAM) slowed everything. If N=18 evaluations are unusually slow, check for it. The fix documented in the steering file is `pkill -9 -f "koda graphify"`; the index on disk survives.
- Look at what already exists before planning. An ADAPT plan was written with only `run_adapt_bonds.py` in view, while pool, operator-list and gate runners already existed (section 9). A search of the module index and of the catalog takes a minute.
- Do not name a confound without varying it. The retracted claim in 5.4 changed two things at once (ansatz and budget) and blamed one.
- An ansatz that cannot represent the state makes every initialization look equal. Check expressivity before auditing a warm-start.
- N=18 evaluations are slow. Run them in the background with a log and poll. macOS has no `timeout`; use `perl -e 'alarm shift; exec @ARGV' SECONDS cmd`.
- `grep`, `find` and `rg` are banned in this repo. Use graphify. Its index was 85 hours old at the last check, so reindex before trusting callers or impact output.
- Pre-existing lint findings that are not from this work: `B007` in `discover_donors`, `E741` and `F841` in two test files.
- The local memory store used during these sessions changed project scope midway. Older entries (ids 301 and 316-320, legacy project `/`) are not reachable from the current instance, and id 320 contains the retracted claim. Treat this file, not memory, as the source of truth.

## 9. Not covered: unrelated work in the working tree

At closure the tree has uncommitted changes that were not made in these sessions and were not reviewed here. The tests that cover them pass, but this document makes no claim about their quality. They appear to be one line of work on lattice symmetry and N=20:

- symmetry-tied circuit builder `create_bond_resolved_symmetry_tied`, orbit-complete selection `select_orbit_complete`, `AnsatzSpec.to_variant(symmetry_group)`;
- symmetry projection of transferred donors (`project_transfer`), a `y` block in `variant_warmstart_theta` and `compose_extend_theta`;
- degenerate-ground-manifold cache `cached_degenerate_ground_manifold` and `data/manifold_cache/`;
- extra flags in `run_variant_topk.py` (`--symmetry-tied`, `--reference-from-spec`, `--auto`, `--rank-by`, `--target-fid`, `--ftol`);
- N=20 specs and results, and the removal of 14 `run_*_queue.sh` scripts;
- the untracked note [`docs/research/symmetry_frustrated_vqe_findings.md`][symmetry-notes] and `.github/context-mode/`.

Also present and not reviewed here: the ADAPT work described in section 7 (committed in `46ae10f` and later). The refreshed catalog marks 17 of 348 entries as `partial`, which are interrupted runs: eight `adapt_poolA_*` (N=14, 18, 20), `symmetry_tied` and `frontier_symmetry` at N=20, two tilted-lattice runs at N=16, `variant_topk_p3_half_nn_rx_square_N20` and `variant_topk_p1_half_nn_rx_fg_square_N20`, plus three older ones from late September and 1 October (`bond_ablation_square_N10_p2_from2`, `bond_topk_regime_square_N18_p2`, `hva_nnn_square_N18_p2_h0.50_metropolis`). List them with `index.json` entries whose `status` is `partial`.

Commit them separately from the files listed in section 4.

## 10. Reproduction

Commands below were run at closure unless stated.

```bash
# Fast test gate for the touched areas (321 passed, about 10 s)
.venv/bin/python -m pytest tests/unit/test_warmstart_service.py tests/unit/test_study_core.py tests/unit/test_bond_mask.py tests/unit/test_ansatz_spec.py tests/unit/test_hva_session_infra.py -q -p no:cacheprovider -k "not GrowAdaptPool and not SymmetryTiedBuilder"

# One N=18 point of the frontier (production budget; takes tens of minutes, run in background)
.venv/bin/python scripts/analysis/vl_vs_hva/run_variant_topk.py --n 18 --h 0.5 --variant p3_half_nn_rx --keep-fracs 0.5 --restarts 2 --maxiter 800 --micro-descent 60 --no-sync

# Same with the two-pass selector and adaptive budget (flags checked with --help; run end to end only at N=6)
.venv/bin/python scripts/analysis/vl_vs_hva/run_variant_topk.py --n 18 --h 0.5 --variant p3_half_nn_rx --keep-fracs 0.5 --restarts 2 --maxiter 800 --micro-descent -1 --two-pass --no-sync

# Analytic warm-start against random (reproduces N=8: 0.7072 / 0.7069 / 0.1473)
.venv/bin/python scripts/analysis/vl_vs_hva/audit_warmstart_vs_random.py --points 8:0.5 --maxiter 400 --random-starts 4

# Donor audit and selector-budget table (N=14 gives section 5.4; the N=18 run is the unfinished donor audit of section 7)
.venv/bin/python scripts/analysis/vl_vs_hva/audit_warmstart_donors.py --n-list 14 --maxiter 800 --random-starts 3
.venv/bin/python scripts/analysis/vl_vs_hva/audit_warmstart_donors.py --n-list 18 --selector-budgets 100 400 800 --skip-conditions

# Refresh generated catalogs
.venv/bin/python scripts/analysis/vl_vs_hva/organize_results.py reindex
.venv/bin/python scripts/general_project_maintenance/generate_module_index.py
```

## 11. Where things live

- Study entry point and taxonomy: [`README`][readme]. Narrative report with the N=10 and N=18 frontiers: [report][report-bond-selection].
- Warm-start code: [`warmstart.py`][warmstart], [`study_core.py`][study-core]. Variant registry: [`hva_variants.py`][hva-variants]. Runner: [`run_variant_topk.py`][run-variant-topk].
- Artifact catalog: `INDEX.md` and `index.json`, regenerated at closure with `organize_results.py reindex` (348 entries: 328 final, 17 partial, 3 deprecated).
- Data behind the N=18 results: `results/hva_vl_study/bond_ablation/` and `results/hva_vl_study/ansatz_specs/` (tracked).

<!-- Links -->
[audit-donors]: ../../scripts/analysis/vl_vs_hva/audit_warmstart_donors.py
[audit-random]: ../../scripts/analysis/vl_vs_hva/audit_warmstart_vs_random.py
[hva-variants]: ../../src/qmbp_simulation/circuits/hva_variants.py
[readme]: README.md
[report-bond-selection]: reports/REPORT_bond_selection_2q.md
[run-variant-topk]: ../../scripts/analysis/vl_vs_hva/run_variant_topk.py
[study-core]: ../../src/qmbp_simulation/framework/study_core.py
[symmetry-notes]: ../../docs/research/symmetry_frustrated_vqe_findings.md
[warmstart]: ../../src/qmbp_simulation/analysis/warmstart.py
