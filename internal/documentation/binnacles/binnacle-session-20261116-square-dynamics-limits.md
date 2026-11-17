# Binnacle: square frustrated dynamics, classical limits and hardware budget

Closed temporarily on **2026-11-16**. Written so a person or an AI agent with no
memory of the sessions can continue. Branch `develop`.

---

## 1. Start here

1. Read `../analysis/classical_simulability_square_frustrated.md` §1 (results) and
   §8 (open caveats). It is the single technical source of truth.
2. Read `../hardware/nighthawk_r2_gate_limits_and_noise.md` for the hardware budget.
3. Use §9 of this file to resume work. Use §6 before trusting any older statement.

Scope of this workstream: **dynamics (quench) of the 2D square frustrated TFIM, how far
a classical MPS can follow it, and what a Nighthawk r2 run would need to exceed that.**
Ansatz construction (HVA variants, ADAPT, pruning, warm-start) is a **separate,
parallel workstream** that is not covered here.

---

## 2. Question and answer in five lines

- Question: where does classical simulation of square frustrated dynamics stop, and
  what must a quantum run provide to go past it?
- Ground state: trivially MPS-representable (entropy 0.7–1.3 nats, N ≤ 18).
- Dynamics across h_c: classical limit tightens with N. At N ≤ 20 exact simulation
  (≤ 17 MB) makes the limit academic. Frontier by extrapolation: **N ≈ 50–53**.
- Quantum budget at N=18: ≈ 13.7k transpiled 2q gates versus ≈ 333 (noise) or
  ≈ 5,000 (architecture). Gap ×41 and ×2.7.
- No hardware advantage is claimed. Details and caveats in the analysis doc.

---

## 3. What was delivered

### 3.1 Library code (`src/qmbp_simulation/`)

| Module                         | Added or changed                                                                                                                                                                                                  |
|--------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `analysis/dynamics.py` (new)   | `truncate_mps`, `evolve_states` (dense propagator or callable), `classical_crossover` and `_sparse`, `correlator_zz`, `count_2q_gates`, `tnoise_curve` and `tnoise_on_backend` (optional `eps_2q`), analytic noise model (`NIGHTHAWK_CZ_ERROR`, `fidelity_from_2q`, `tnoise_from_2q_curve`, `gates_to_floor`), routing tools (`nighthawk_coupling_map`, `square_grid_coupling_map`, `transpiled_2q_count`, `routing_overhead_breakdown`) |
| `analysis/observables.py`      | `bipartition_entropy`, `square_vertical_cut_qubits`                                                                                                                                                               |
| `circuits/trotter.py`          | `build_frustrated_trotter_step`, `build_kitaev_trotter_step`                                                                                                                                                      |
| `circuits/ansatz_spec.py`, `circuits/hva.py` | Kitaev support: `AnsatzSpec.from_kitaev`, `ansatz_kind`, `initial_state`, `create_kitaev_bond_resolved` (earlier in the workstream)                                                                 |
| `circuits/bond_mask.py`        | `bond_weights_for_blocks` error now names the likely cause (θ from another N)                                                                                                                                      |

The three runners previously carried private copies of `_truncate_mps` (×3),
`_classical_crossover` (×2), `_trotter_step`, `_tnoise_curve` (×3), `_correlator_zz`
and `_n2q`. All were deleted in favour of the shared code above.

### 3.2 Runners (`scripts/analysis/vl_vs_hva/`)

| Script                         | State | Notes                                                                                                                                                                      |
|--------------------------------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `run_prune03_dynamics.py`      | done  | `--derive-prep` (any N), `--chi-values`, `--noise-model {fake,analytic}`, `--eps-2q`, `--load-spec`; sparse evolution; per-quench checkpoint (`status` partial → complete); guard for N ≠ 10 without `--derive-prep` |
| `run_kitaev_heavyhex_noise.py` | done  | Shared helpers; checkpoint `partial_tnoise` before the costly sparse crossover; `--noise-model`                                                                            |
| `run_prep_evolution_noise.py`  | done  | Shared helpers; `status: complete`. **Has no `--noise-model` flag** (fake backend only)                                                                                    |
| `run_native_routing_budget.py` (new) | done | Routing overhead on the real Nighthawk square map, nn versus nnn breakdown                                                                                         |

All JSON is written with `utils.helpers.write_json_atomic` (atomic and non-finite-safe;
see §5, decision 3).

### 3.3 Tests

`tests/unit/test_dynamics.py` has **34 tests** (truncation, crossover dense versus
sparse, correlator, evolution, Trotter-step unitarity, analytic noise model, JSON
robustness, bipartition entropy, native routing). `tests/unit/test_ansatz_spec.py`
gained a Kitaev class. Runner scripts are not imported by tests (project rule); they are
covered by smoke runs.

### 3.4 Documents

| File                                                      | Role                                       |
|-----------------------------------------------------------|--------------------------------------------|
| `analysis/classical_simulability_square_frustrated.md`    | Technical source of truth                  |
| `hardware/nighthawk_r2_gate_limits_and_noise.md`          | Gate limits, error rates, routing, API     |
| `binnacles/binnacle-session-20261116-square-dynamics-limits.md` | This file                            |
| `next-steps/next_steps.md`                                | Section A0 points here                     |

### 3.5 Experiments and where the outputs are

`results/experiments/` is **git-ignored**: these files exist only on the machine that
ran them. The numbers are recorded in the analysis doc.

| Experiment                                   | Output directory (under `results/experiments/exp_frustrated/`) | State                |
|----------------------------------------------|----------------------------------------------------------------|----------------------|
| `qd1` quench study, square N=10–18           | `qd1/tfim/square_p1/run_20261002_012532.json`                  | complete             |
| prune0.3 dynamics, h_prep 0.5, N=10 and N=14 | `prune03_dynamics/tfim_frustrated/square_N{10,14}_hprep0.50/`  | complete             |
| Crossing quench h_prep 3.0 → 0.3, N=12,16,18 | `prune03_dynamics/tfim_frustrated/square_N{12,16,18}_hprep3.00/` | complete           |
| Native routing budget                        | `native_routing_budget/`                                       | complete             |
| Kitaev heavy-hex and prep+evolution (early)  | `kitaev_heavyhex_noise/`, `prep_evolution_noise/`              | on disk (6 and 1 JSON); scale not re-audited, see memory `dynamics-*` notes |
| prune0.3 N=18, h_prep 0.5                    | none                                                           | **never completed** (see §7) |
| S_sat at N=22 and N=24                       | none                                                           | **never completed**  |

Portable ansatz files `results/hva_vl_study/ansatz_specs/prune0.3_square_N{12,14,16,18}_*.spec.json`
are in a **tracked** directory. Re-deriving them costs hours (two VQE per N); committing
them lets any runner reuse them with `--load-spec`.

---

## 4. Findings

See the analysis doc, §3 to §6, for numbers and caveats. One-line versions:

- Critical region of our model: h ≈ 0.3–0.6. Default quench from h=0.5 leaves it.
- t\*(χ) shrinks with N; at N=18 χ=128 fails in 13 steps.
- S_sat ≈ 0.26·N + 0.1, validated at N=20 (1.1 % error).
- 2D cuts do not exceed the 1D half-chain at widths 4–5 (negative result).
- Square-native routing ×3.1–3.55; nn bonds exactly ×2.00.
- Ground state is far easier than dynamics.

---

## 5. Design decisions and why

1. **Shared code in `src`, not in scripts** (project rule). The extraction removed ~7
   private duplicates and made every runner use one tested implementation.
2. **Sparse evolution only.** Dense `expm` at N=18 needs a 2¹⁸×2¹⁸ matrix (≈ 1.1 TB)
   and was killed by the OS (exit 137). `expm_multiply` on a sparse H scales.
3. **`write_json_atomic` instead of `json.dump(default=json_serialize)`.** `default=`
   is only called for non-serializable types, so a native `inf`/`nan` inside a list
   (for example the Loschmidt rate function) was written as the invalid token
   `Infinity`. `write_json_atomic` pre-serializes recursively and writes tmp + rename.
4. **`--derive-prep` rather than reusing the N=10 artefact.** The prune0.3 bond
   selection comes from |θ| of a specific N; reusing N=10 θ at another N is wrong,
   and the runner now refuses it with an explanatory error.
5. **Prepare in the paramagnet for crossing quenches.** The ansatz has a fidelity
   ceiling ≈ 0.96 near h_c (p=2); at h=3.0 it reaches ≈ 0.99.
6. **Analytic noise model (`ε₂q = 3×10⁻³`)** instead of the fake-backend calibration,
   whose couplers with error 1.0 inflate the mean. Transpilation still goes to a real
   coupling map so the 2q count includes routing.
7. **No "2D truncation" mode in the runners.** The measured 2D/1D entropy ratio was
   ≈ 1 at N ≤ 20, so it would add complexity for no gain. Helpers remain available.

---

## 6. Corrections to earlier statements

These were said during the sessions and later found wrong or imprecise. Do not rely
on the original wording if you find it in memory entries or chat history.

| Earlier statement                                                     | Correct statement                                                                                                           |
|-----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| "The ×6 routing overhead was a generic transpilation artefact"        | The ×6 was measured on `FakeTorino` (heavy-hex): a square lattice on the wrong topology. Native square gives ×3.5.          |
| "nnn are the diagonal bonds"                                          | `_generate_nnn_edges` returns all graph-distance-2 pairs: ~46 % are straight two-step pairs (N=16: 18 diagonal + 16 straight). |
| "`magnetization_z` has a latent bug (nested `if`)"                    | It is dead code (a redundant inner `if`); the function is correct for the sizes used.                                       |
| "S_sat(N) from two points is enough"                                  | Replaced by five measured points N=10–20; slope validated at N=20 (1.1 %). Beyond N=20 it is still an extrapolation.        |
| "χ ≈ e^S is the bond dimension needed"                                | Only a loose lower heuristic. Measured crossover χ exceeds it.                                                              |
| "The GNN-prepared state gives harder dynamics than a trivial quench"  | Not supported: S_max is below the all-zero product state (4.24 versus 7.98 at N=18) and above the all-plus one (1.46). Confounded by energy density. See analysis §8.1. |
| "The crossing-quench preparation shows the GNN's value"               | The h=3.0 ground state is close to the all-plus product state. It shows cheap preparation, not GNN value. See analysis §8.2. |

---

## 7. Gotchas for the next person

- **`results/experiments/` is git-ignored.** Copy or re-run before relying on a file.
- **`/tmp` logs and queue-status files disappear** (a reboot cleared them). Persisted
  state lives in the JSON files and in this documentation.
- **A tool timeout does not kill the Python process it started.** After a timed-out
  command, check `pgrep -fl python` and `kill -9` orphans, or they compete for CPU.
- `control_bash_process stop` closes the record but may leave the detached process.
- macOS has no `timeout` command. Run long jobs in the background with `nohup`.
- The interactive shell tool occasionally wedges (echoes the command, exit −1).
  Starting a fresh process or writing the script to a file first works around it.
- `--derive-prep` optimizes two variants: N=16–18 takes hours. N=22+ exact evolution
  with a per-step 2¹¹×2¹¹ SVD exceeded 20 minutes without finishing.
- The prune0.3 N=18, h_prep 0.5 run died of OOM before the sparse fix; it was not
  re-run. The crossing-quench series (h_prep 3.0) replaced it.
- The `tnoise_*` functions emit a `UserWarning` that FakeNighthawk properties are not
  typical Nighthawk values. This is expected; only connectivity is used.

---

## 8. State of things

| Item                                                             | Status      |
|------------------------------------------------------------------|-------------|
| Shared dynamics library, Trotter steps, tests, runners           | **done**    |
| Classical limit measured to N=18 (crossover) and N=20 (entropy)  | **done**    |
| Native square routing budget                                     | **done**    |
| Analytic noise model and CLI flags (2 of 3 runners)              | **done**    |
| Analytic noise flag on `run_prep_evolution_noise.py`             | not done    |
| S_sat at N=22, 24                                                | not done    |
| DMRG ground state at N=24–36 (validates ground-state claim)      | not done    |
| t\*(χ) at N=20 with χ up to 256                                  | not done    |
| nnn pruning versus physics-lost study                            | not done    |
| GNN versus trivial quench at matched energy density              | not done    |
| Noise-suppression model (ZNE/QESEM) entering `ε₂q`               | not started (planned) |
| Commit of this session's last edits (see §10)                    | **pending** |

---

## 9. How to resume

Verify the environment:

```bash
.venv/bin/python -m pytest tests/unit/test_dynamics.py tests/unit/test_ansatz_spec.py tests/unit/test_bond_mask.py -q
```

Re-run the cheapest experiment (about a minute):

```bash
.venv/bin/python scripts/analysis/vl_vs_hva/run_native_routing_budget.py --n-values 10 12 14 16 18 --h2 2.5 --j2 0.5
```

Re-run one point of the crossing series (hours at N=18; use `nohup`, background):

```bash
.venv/bin/python scripts/analysis/vl_vs_hva/run_prune03_dynamics.py --n 16 --derive-prep --h-prep 3.0 --h2-list 0.3 0.5 1.0 2.5 --dt 0.05 --n-trotter 120 --restarts 6 --maxiter 1500 --chi-values 8 16 32 64 128
```

Reuse a stored ansatz instead of re-optimizing:

```bash
.venv/bin/python scripts/analysis/vl_vs_hva/run_prune03_dynamics.py --n 16 --load-spec results/hva_vl_study/ansatz_specs/prune0.3_square_N16_h3.00.spec.json --h-prep 3.0 --h2-list 0.3 --chi-values 8 16 32 64 128
```

(The `--load-spec` form was verified to parse; the N=16 spec file exists on disk.)

Highest-value next step: close analysis §8.3 and §8.5 (S_sat at N=22/24 and DMRG
ground state at N=24–36), because every N > 20 statement rests on them.

---

## 10. Verification at closing (2026-11-16)

- `pytest` on `test_dynamics`, `test_ansatz_spec`, `test_bond_mask`, `test_analysis`,
  `test_imports`, `integration/test_smoke`: **211 passed**.
- `ruff` clean on every file this workstream created or edited. One pre-existing
  `SIM102` remains in `observables.py::detect_dqpt_critical_times`, not from this work.
- `--help` of all four runners parsed; flags in this file match them.
- Module index regenerated (262 entries).

**Uncommitted at closing** (the user did not ask for a commit): the 9 new tests in
`tests/unit/test_dynamics.py`, the two rewritten analysis/hardware documents, this
binnacle, the `next_steps.md` edit, `.kiro/steering/module-index.md`, and the new
`prune0.3_square_N{12,14,16,18}` spec files. Suggested message:
`docs+tests: close square dynamics workstream (classical limits, native routing budget)`.

---

## 11. Not part of this work (do not attribute)

- The working tree also holds many uncommitted changes from the ansatz workstream
  (`ansatz_spec.py`, `bond_mask.py`, `hva.py`, `hva_variants.py`, `study_core.py`,
  `warmstart.py`, `fidelity.py`, deleted `*_queue.sh` scripts, new specs and
  `docs/research/`). Review those separately before committing.
- About 59 ruff findings exist in other `vl_vs_hva/*.py` scripts (`analyze_*`,
  `verify_*`, and others). They predate this work.
- `.kiro/steering/project-status.md` is auto-generated by `ResultIndex.refresh_status()`.
  It was not edited by hand and does not list this workstream.

---

## 12. Pointers for AI agents

- Project memory (`mem`/`yax`) holds the working notes under these topic keys:
  `classical-limit-hypothesis-audit`, `crossing-transition-NScan-classical-limit`,
  `nighthawk-analytic-noise-model`, `nighthawk-square-native-routing-overhead`,
  `quantum-advantage-recipe-refined`, `classical-limit-N30-transitions`,
  `2d-cut-vs-halfchain-crossover`, `prune03-sparse-evolution-oom`,
  `prune03-derive-prep-mode`, `dynamics-runners-json-persistence`,
  `dynamics-helpers-src-extraction`. The documents in this repo override memory where
  they differ (memory still contains the wording corrected in §6).
- Project rules to follow: no `grep`/`find` (use graphify), verify CLI flags with
  `--help` before quoting a command, reuse shared helpers instead of copying, add a
  regression test for each new path, regenerate the module index after adding symbols.
