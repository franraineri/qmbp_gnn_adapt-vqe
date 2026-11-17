# Binnacle: lattice symmetry, short ansatz and tilted PBC clusters

Closed temporarily on **2026-11-16**. Written so a person or an AI agent with no memory of the
sessions can continue. Branch `develop`. Nothing from this workstream is committed.

---

## 1. Start here

1. Read the [handoff][handoff] (status per experiment, artifacts, verified commands, pitfalls,
   repository state). It is the entry point.
2. Read sections 9 to 11 of the [findings note][findings] for the physics, the tables and the open
   work. Sections 1 to 8 hold the earlier OBC symmetry study.
3. Use section 4 of this file before trusting any older statement or memory.

Scope of this workstream: **can lattice symmetry give an ansatz with fewer 2-qubit gates at the
same fidelity for the frustrated square TFIM (J2 = 0.5, h = 0.5) at N >= 16.** Covered: orbit-tied
ansatz, symmetry-breaking layers, N=20 pruning closure, and the new tilted PBC clusters.

Not covered here (separate parallel workstreams, each with its own closure):

- Dynamics and classical limits: [binnacle][binnacle-dynamics].
- Warm-start cascade and N=18 compression: [handoff][handoff-warmstart].
- Haiqu vector loading: [binnacle][binnacle-haiqu].

---

## 2. Question and answer in five lines

- **OBC square, N >= 16: symmetry does not help.** Tying fails (degenerate ground state, spontaneous
  symmetry breaking) and an explicit symmetry-breaking RY layer is redundant with bond-resolved
  parameters (delta within +-0.003 at N=9/12/16/18).
- **N=20 OBC: nothing tried improved the known frontier** (0.8988 @148, 0.9491 @176, 0.9939 @524)
  with the budget used. This is not a proof of a physical limit.
- **Tilted PBC clusters (a different physical model) break the symmetry-versus-degeneracy
  exclusion:** N=16 and N=18 have the full point group and a non-degenerate ground state.
- **The tied ansatz reaches 0.9960 at N=9 with 13 parameters** and, seeded from the N=9 angles and
  optimized on fidelity, **0.7951 at N=16 (512 2q)**. Omitting the layer-0 nnn block gives 0.7881
  at 448 2q.
- **Open and most promising: p=5/6 at N=16.** The fidelity had not saturated at p=4.

---

## 3. What was done

| Step                                             | Result                                                           | State          |
|--------------------------------------------------|------------------------------------------------------------------|----------------|
| Symmetry-breaking RY block, A/B at N=9/12/16/18  | Redundant with bond-resolved                                      | Closed         |
| ADAPT read from the existing scoreboard          | N=14 competitive, N=20 collapses (no new runs)                    | Closed         |
| Tilted generator, gate, symmetry audit           | N=9/16/18 symmetric and non-degenerate; GS invariant at N=16      | Closed         |
| Analytic seeds on the tied tilted ansatz         | Calibrated seed best (N=9: 0.32, N=16: 0.065 before optimizing)   | Closed         |
| N=9 warm-start and 2q Pareto                     | 0.9960 @288 (p=4); nnn blocks are indispensable                   | Closed         |
| N=16 calibrated p-sweep                          | Stalls (max 0.3031 at p=2)                                        | Closed         |
| Transfer N=9 to N=16/18 and fidelity objective   | N=16 p=4 0.7951; N=18 p=4 0.5796; energy objective degrades it    | Closed         |
| Overnight campaign (8x4000 and 6x6000 iter)      | 0.7951 unchanged: saturated at fixed p                             | Closed         |
| Drop nnn blocks at N=16 p=4                      | Layer 0 only: 0.7881 @448 (-64 2q); more drops degrade strongly   | Closed         |
| N=14 prune from stored spec                      | New point 0.9021 @98 (best fidelity per 2q)                       | Closed         |
| N=20 nnn top-k 2/3/4, clusters, refine `stored`  | Nothing enters the frontier                                       | See section 4  |
| Closure: tests, docs, module index               | 44 tests for the tilted generator; 141 pass on the touched areas  | Done           |

---

## 4. Corrections and retractions (read before reusing older notes)

- **"0.9939 @ 370 2q" is wrong.** `p3_half_nn_rx full` at N=20 is 0.9939 @ **524** 2q; 370 belongs to
  `nnn2_half` (0.9647).
- **The N=20 geometric-cluster experiment is inconclusive, not a refutation.** It used the
  `p2_half_nn` structure (198-218 2q) against a `p3_half_nn_rx` baseline and only 20 iterations.
- **"Physical limit at N=20" was not shown.** What was observed: no technique improved the frontier
  at 2 restarts x 20 iterations. `refine stored` was not tried with a large budget.
- **The first `drop_nnn` run was invalid.** Its loader read the failed N=16 Pareto angles (fidelity
  0.10) instead of the transfer angles (0.7951); the baseline ended at 0.5565. Fixed, with a guard
  that aborts if the baseline does not start near 0.795. The corrected run is the one reported.
- **The memories numbered 388 to 416 are not reachable** from the current memory instance (a search
  for their contents returns nothing). They contained some of the overstated claims above. This
  repository (the handoff and the findings note) is the source of truth.

---

## 5. Pitfalls that cost time

All are detailed with fixes in the [handoff][handoff]. The ones most likely to bite again:

- JSON outputs are overwritten, not appended. Pass every tolerance or depth in a single invocation.
- `detect_symmetry_group` caps at 16 permutations by default and reports a spurious order 17 on PBC
  clusters. Use `max_syms=100000`.
- The tilted manifold cache key is `square_tilted`. The key `square` silently loads the OBC manifold.
- `_load_src_tied_theta(N, ...)` reads the Pareto JSON of N, not the transfer JSON.
- The fidelity objective needs the exact ground state. It tests the ansatz expressivity; it is not a
  hardware method.

---

## 6. How to resume

1. Check that `run_tilted_drop_nnn.py` still starts `p4_full` near 0.795 (the guard enforces it).
2. Sweep p=5 and p=6 at N=16 with the fidelity objective. Seeds: N=9 at p=5/6 (cheap) or tiling of
   the N=16 p=4 optimum. The expectation that p=6 passes 0.95 is a hypothesis, not a result.
3. Repeat the layer-0 drop with several random seeds and on other depths before relying on it.
4. Repeat the N=20 clusters on `p3_half_nn_rx` with a comparable budget.
5. Move `optimize_fidelity` into `src/` with tests and add an OBC/PBC field to the scoreboard.

---

## 7. Where things live and repository state

- Code: `src/qmbp_simulation/models/tilted_square.py` (tests in `tests/unit/test_tilted_square.py`);
  runners and probes in `scripts/analysis/vl_vs_hva/` (`probe_tilted_*`, `run_tilted_*`,
  `run_n20_geo_cluster.py`, `run_sb_layers_ab.py`). The runners have no tests.
- Data: `results/hva_vl_study/bond_ablation/tilted_*.json`, new specs in
  `results/hva_vl_study/ansatz_specs/`, manifold cache in `data/manifold_cache/`.
- **Uncommitted, and shared with other workstreams.** The working tree mixes this line with the
  warm-start closure files (`audit_warmstart_*.py` and their handoff) and with generated catalogs
  (`INDEX.md`, `index.json`, `module-index.md`, scoreboard files). Commit the warm-start files
  separately, then this line, and regenerate the catalogs last. Fourteen `run_*_queue.sh` scripts
  show as deleted and are listed by the warm-start handoff as part of this symmetry/N=20 line.
  Confirm the removal is intended before committing.

<!-- Links -->
[binnacle-dynamics]: binnacle-session-20261116-square-dynamics-limits.md
[binnacle-haiqu]: binnacle-vl-haiqu-square-frustrated.md
[findings]: ../../../docs/research/symmetry_frustrated_vqe_findings.md
[handoff]: ../../../docs/research/HANDOFF_symmetry_tilted.md
[handoff-warmstart]: ../../../results/hva_vl_study/HANDOFF_2026-11-16_warmstart_n18.md
