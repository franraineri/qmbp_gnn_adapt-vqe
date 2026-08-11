# Validation Targets & Historical Results

> **Single source of truth** for numerical baselines and thesis-ready tables.
> Use these numbers to evaluate whether a change improved or regressed the pipeline.
> For analysis and interpretation of these numbers, see `poc-results.md`.
> For experiment methodology and protocols, see `steering/experiment-protocol.md`.

## Validation Targets (V6.1, HardwareDeployerV61 4-metric checklist)

> **Note**: `HardwareDeployerV61` was refactored into `HardwareBackend` (2026-06-04).
> The 4-metric checklist (ΔE/gap, ⟨X⟩, ⟨ZZ⟩, ADAPT) remains unchanged.
> Current class: `from qmbp_simulation.execution.hardware import HardwareBackend`

### N=6 (all pass ✅, from 9 definitive runs)

| h_test | ΔE/gap (mean±std) | Checklist | Status |
|--------|-------------------|-----------|--------|
| 1.25 | 3.77% ± 0.53% | 4/4 | ✅ All seeds pass |
| 1.4 | 1.71% ± 0.14% | 4/4 | ✅ Comfortable |
| 1.5 | 1.19% ± 0.23% | 4/4 | ✅ Best |

### N=10 (from 6 definitive runs)

| h_test | ΔE/gap (mean±std) | Checklist | Status |
|--------|-------------------|-----------|--------|
| 1.4 | 4.79% ± 0.50% | 3.7/4 | ✅ Mean passes (seed 42 borderline at 5.49%) |
| 1.5 | 2.96% ± 0.33% | 4/4 | ✅ All seeds pass |

### N=10 Ladder (non-uniform J, NNConv)

| h_test | ΔE/gap | Checklist | Status |
|--------|--------|-----------|--------|
| 2.0 | 11.75% | 1/4 | ❌ Physics limit — HVA p=2 too shallow for ladder connectivity |

## Completed Milestones

- ~~Test MPNN on ladder topology~~ **DONE** — ladder with non-uniform J (J_rung=0.5) tested; HVA p=2 too shallow (physics limit).
- ~~Scale to N=10~~ **DONE** — optimal config: seed=43, patience=500, MPNN h=128. h=1.4 passes (4.44%).
- ~~Prepare thesis results chapter~~ **DONE** — Tables 4.2 and 4.3 consolidated (15 runs, 3 seeds × h_test values).

## Literature-Informed Insights

- h=1.25 ceiling confirmed as physics limit by Tripathi et al. 2026 (independent validation)
- Hardware noise will broaden critical crossover (Sharma 2026) — expected, not a failure
- GNN > CNN by 36% for circuit property prediction (Meng 2025) — validates GINConv
- Shot noise (~1.6e-2 at 4096 shots) will dominate over gate errors on hardware
- MPNN weights encode phase transition info (Hernandes 2025) — **validated at N=6 and N=10**
- NN-enhanced ZNE (Sun 2025) implemented in `NNExtrapolator` — ready for hardware
- NNConv edge features validated for non-uniform J ladder (V6.1 Task 10)

## Scripts Quick Reference

| Script | Purpose | Time |
|--------|---------|------|
| `tests/smoke_test.py` | V6.0 legacy end-to-end (6 h-points) | ~7s |

### Makefile Test Targets
- `make test` — runs fast tests only, excludes `@pytest.mark.slow` (~12s)
- `make test-full` — runs ALL tests including slow FakeTorino tests (~60s)

## Checkpoint Files (verbose mode only)

When `--verbose` is passed to `run_v61_parametric.py`:
- Pattern: `checkpoint_<run_id>_<phase>.json`
- Location: `scripts/notebook_results/`
- Deleted automatically on successful completion; remain on disk after crashes

## Notebook Executor Features

- Wall-clock timeout guard (prevents runaway VQE)
- Auto-extracts structured metrics (fidelity, MSE, ΔE/gap, checklist, phase, gradients, ZNE)
- Peak memory tracking (critical for N=10 scaling)
- Environment capture (git state, package versions, platform)
- Binnacle-ready markdown output (auto-observations, run-to-run comparison)
- Results pruning (`--keep-last N`)
- Structured exit codes: 0=pass, 1=execution fail, 2=validation fail, 3=pre-flight fail
