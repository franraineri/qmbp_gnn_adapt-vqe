---
inclusion: fileMatch
fileMatchPattern: "scripts/run_*,scripts/experiment_runners/*,experiments/**"
---

# Experiment Protocol — Full Pipeline (5 Phases)

## Pipeline Overview

```
DESIGN ──▶ CHECK ──▶ RUN ──▶ ANALYZE ──▶ DOCUMENT
```

Every experiment follows this sequence. Skipping phases leads to
unvalidated results, broken analysis tooling, or undocumented findings.

### Phase 1: DESIGN (create script)

```bash
cp scripts/runner_templates/template_validation_runner.py scripts/run_<name>.py
```

- Inherit from `ValidationRunner` (or `ExperimentRunner`/`VariantPipelineRunner`)
- Define `runner_id`, `experiment_id`, `description`, `hypothesis`
- Implement `define_sections()` with per-section hypotheses
- Use framework utilities: `self.vqe_descending_sweep()`, `self.exact_ground_state()`
- See: `.kiro/steering/runner-standards.md`

### Phase 2: CHECK (preflight validation)

```bash
python scripts/preflight.py --from-script scripts/run_<name>.py
python scripts/run_<name>.py --dry-run
```

- **ERRORS → fix before executing** (regime violations, structural issues)
- WARNINGs → acceptable if intentional (e.g., below-regime test points)
- `--dry-run` lists sections without executing

### Phase 3: RUN (execute)

```bash
python scripts/run_<name>.py
python scripts/run_<name>.py --section 1 2  # selective
python scripts/run_<name>.py --verbose      # debug
```

- Output: `results/experiments/exp_<id>/run_<timestamp>.json`
- Structured log: `results/experiments/exp_<id>/log_<timestamp>.json`

### Phase 4: ANALYZE (verify with existing tools)

```bash
# 1. Does the experiment appear correctly in the digest?
python -m scripts.digest --kind experiment --sort verdict

# 2. Does compare.py work without errors?
python scripts/compare.py --all

# 3. Are thesis claims still valid?
python analysis/verify_claims.py

# 4. Are there new coverage gaps?
python analysis/scan_coverage.py

# 5. (If failures) What's the root cause?
python analysis/diagnose.py --all
```

- See: `.kiro/steering/analysis-tooling.md`

### Phase 5: DOCUMENT (record findings)

Update these files (using REFERENCES, never duplicate data):

| File | What to add |
|------|-------------|
| `documentation/binnacles/binnacle-<name>.md` | Full binnacle entry with tables + cross-references |
| `analysis/10_key_findings_corrected.md` | New hallazgo # (if novel finding) |
| `documentation/analysis/08_summary.md` | Session entry (date, experiments, verdicts) |
| `documentation/analysis/09_thesis_tables.md` | Only if new table for Chapter 5 |

### Quick-Reference Checklist

```
□ Script created from template (ValidationRunner / ExperimentRunner)
□ preflight.py --from-script → PASS (or justified WARNINGs only)
□ --dry-run shows correct sections
□ Execution complete → JSON saved in results/experiments/
□ python -m scripts.digest --kind experiment → experiment visible with correct verdict
□ python scripts/compare.py --all → no crash, experiment listed
□ python analysis/verify_claims.py → no new contradictions
□ python analysis/scan_coverage.py → check if gaps closed
□ Binnacle created/updated with cross-references
□ 10_key_findings updated (if new finding)
□ 08_summary updated (session entry)
```

---

## Runner Standards (ALWAYS ENFORCE)

All `scripts/run_*.py` and `scripts/experiment_runners/run_*.py` scripts MUST use the
standardized runner base classes from `qmbp_simulation.framework.runner_base`:
- `ValidationRunner` — multi-section validation suites.
- `ExperimentRunner` — BaseExperiment lifecycle wrappers.
- `VariantPipelineRunner` — batch pipeline variant runners.

See `.kiro/steering/runner-standards.md` for full reference and templates in
`scripts/runner_templates/`.

## Preflight Validation (ALWAYS ENFORCE)

Preflight is BUILT INTO the runner base classes. It runs automatically before execution.
- `ValidationRunner`: validates runner_id, sections, hypotheses.
- `ExperimentRunner`: delegates to `BaseExperiment.execute()` built-in preflight.
- `VariantPipelineRunner`: runs `PreflightChecker` on all variant specs.

For legacy scripts or manual validation:
1. **Run preflight**: `.venv/bin/python -m qmbp_simulation.framework.preflight --from-script <path>`
2. **If exit code = 1 (ERRORS)**: Do NOT execute. Report errors and suggest fixes.
3. **If exit code = 0**: Safe to execute (note any warnings).

This catches:
- Data leakage (h_test in training set)
- Valid regime violations (h_test below threshold)
- Descending sweep violations (warm-start requires h=high→low)
- Duplicate variant IDs / section IDs
- Output directory collisions
- Missing pipeline scripts

**When to skip**: Scripts that don't define variants (e.g., `scan_coverage.py`, `diagnose.py`, `compare.py`, standalone analysis scripts without PipelineVariant definitions).

## Seed Management

- **Default seeds**: 42, 43, 44 (thesis uses 3 seeds for statistical significance)
- **Seed 43 is optimal for N=10** (10x better MSE than seed 42)
- Seeds control: NumPy RNG, PyTorch RNG, Python `random` module
- Always set all three: `np.random.seed(s)`, `torch.manual_seed(s)`, `random.seed(s)`
- Report seed in all result JSONs and binnacle entries

## Result Logging Structure

All experiment runs produce results in:
```
results/experiments/exp_<id>/run_<timestamp>.json
```

Required fields in result JSON:
- `timestamp`, `run_id` (8-char hex)
- `config` — full `ExperimentConfig` serialized (N, J, p, hidden, epochs, lr, patience, seeds, h_values, etc.)
- `analysis` — summary metrics and per-seed results
- `results` — dict keyed by seed with list of `ExperimentMetrics`
- `environment` — Python version, Qiskit version, etc.
- `elapsed_s` — total wall-clock time

## Binnacle Protocol

When results are thesis-relevant:
1. Run with `--binnacle --label "description"` flag
2. Auto-appends to `documentation/binnacles/binnacle-N6.md` or `binnacle-N10.md`
3. Include: config table, per-seed results, aggregated mean±std, observations
4. Compare against previous run (auto-generated by `scripts/compare.py`)

## Parametric Run Conventions

### Configuration Naming
- `baseline_N6` — standard N=6 config (h=64, 6000ep, seed=42)
- `baseline_N10` — standard N=10 config (h=128, 6000ep, patience=500, seed=43)
- `dense_grid_N6` — 40-point h-grid (for gradient analysis only)
- `ladder_N6` — non-uniform J ladder topology
- `edge_features_N6` — NNConv with edge features enabled

### Validation Checklist (before committing results)
1. ΔE/gap < 5% at h_test? (primary criterion)
2. Correct phase label? (paramagnetic for h > 1)
3. MSE reasonable? (< 0.01 for N=6, < 0.05 for N=10)
4. No NaN/Inf in outputs?
5. Runtime within expected bounds? (N=6: ~50s, N=10: ~2-6min)

## Script-Specific Notes

### `scripts/run_experiment.py`
- Unified CLI for running experiments by ID
- Discovers experiments from `experiments/` category directories
- Supports `--exp A3 B4 F3` for multiple experiments
- Supports `--list` to show all registered experiments
- Supports `--verbose` / `-v` for INFO logging + StructuredLogger output
- Supports `--debug` for DEBUG logging + all verbose features
- Exit codes: 0=all pass, 1=some fail, 2=validation fail

### `scripts/run_pipeline.py`
- Full 4-phase pipeline CLI
- Accepts lattice config, h-values, and output directory as arguments
- Supports skip/resume via phase flags and checkpoint detection
- Uses `PipelineRunner` from `qmbp_simulation.pipeline`

### `scripts/compare.py`
- Cross-experiment result comparison
- Supports `--all` for comparing all results
- Produces structured JSON output + terminal summary

### `scripts/smoke_test.py`
- Imports all public submodules of `qmbp_simulation`
- Runs minimal pipeline (N=4, p=1, 3 h-points)
- Verifies ΔE/gap < 5%
- Completes in under 30 seconds
- Exit non-zero on failure with descriptive error

### `scripts/run_thesis_results.py`
- Consolidates definitive runs into thesis tables
- Computes mean±std across seeds for each h_test
- Produces `results/thesis/thesis_results_<timestamp>.json`

## When to Re-Run

| Change | Re-run |
|--------|--------|
| Modified `src/qmbp_simulation/optimizers/` | `make test` + `scripts/smoke_test.py` |
| Modified `src/qmbp_simulation/predictors/` | `make test` + `scripts/smoke_test.py` |
| Modified `src/qmbp_simulation/execution/` | `make test` + `scripts/smoke_test.py` |
| Modified `src/qmbp_simulation/analysis/` | `make test` |
| Modified `src/qmbp_simulation/framework/` | `make test` + run one experiment to verify |
| New thesis results needed | `scripts/run_thesis_results.py` |

## When to Escalate (Ask User vs. Proceed)

| Situation | Action |
|-----------|--------|
| Test fails after source change | Fix if obvious (typo, import); ask user if architectural |
| ΔE/gap > 10% on a run | Report immediately — likely a bug, not physics |
| ΔE/gap 5-10% on a run | Proceed — this is the borderline regime, expected for some configs |
| New experiment proposed | Ask user — state hypothesis first (experiment discipline rule) |
| Modifying stable module needed | Always ask — explain why the stable module needs changing |
| Hardware credentials required | Ask user — never attempt to read/expose credential values |
| Run would take >5 min | Inform user of expected time before starting |
| Result contradicts binnacle | Report the contradiction — may indicate a regression or new finding |

## ZNE Scaling Rules (Validated 2026-05-14)

- **N=6, 3 layouts**: Works perfectly (R²>0.99, +40% gain). Linear E(CES) holds.
- **N=10, 3 layouts**: Completely fails (R²<0.05, negative gain). Non-perturbative regime.
- **Rule**: n_layouts must scale with system size. Use O(n) for n-qubit circuits.
- **Literature**: Tsubouchi et al. (2023, PRL 131:210602) proves exp(depth×qubits) cost. Rabinovich et al. (2025, arXiv:2511.02901) proposes CLP-ZNE with O(n) cyclic permutations.
- **Before running noisy sweeps at new N**: verify CES range is in perturbative regime (total CES < 0.5) or use sufficient layouts.

## Experiment Value Checklist (Before Running)

Ask these questions before starting any experiment:

1. **What hypothesis does this test?** (If none, don't run it.)
2. **What would I learn if it passes?** (If "nothing new," don't run it.)
3. **What would I learn if it fails?** (If "nothing new," don't run it.)
4. **Has this already been established?** (Check binnacle — if 3 seeds already confirm, don't re-run.)
5. **Is this a physics limit or a tunable parameter?** (If physics limit confirmed, no hyperparameter will help.)

## Known Physics Limits (Do NOT Try to Tune Past These)

| Limit | Evidence | Implication |
|-------|----------|-------------|
| h=1.25 ceiling at N=6 (2-3/6 V6.0 checklist) | 40+ experiments, all configs | HVA p=2 expressibility |
| h=1.4 fails with seed 42 at N=10 | Confirmed 3× | Seed-dependent MPNN convergence |
| h=1.5 ceiling at N=20 (ΔE/gap≈7.7%) | V7 3C with L-BFGS-B + 3 restarts | HVA expressibility degrades with N |
| Valid regime shifts: N=6→h≥1.25, N=10→h≥1.5, N=20→h≥2.0 | V7 3C + binnacles | Physics limit, not tunable |
| N=20 full pipeline: ΔE/gap=1.75% ✅ (h≥1.5 training only) | 3 runs, Run 3 passes | Train ONLY on valid regime |
| Energy-error filter HURTS at N=20 | Run 2 worse than Run 1 (7.4% vs 6.0%) | Coverage > purity for MPNN training |
| Training on invalid regime poisons MPNN | Runs 1-2 included h<1.5 → bad θ | Restrict h-grid to valid regime per N |
| N=6 VQE config (5 rst, σ=0.1) fails at N=20 | Runs 1-2 got avg ΔE=0.09-0.14 | Scale restarts and σ with N |
| "More h-points = better" is FALSE | 19 pts (h∈[0.8,2.0]) worse than 11 pts (h∈[1.5,2.0]) | Quality of regime > quantity of points |
| Fidelity filter unavailable at N≥15 (DMRG) | ground_state=None → fidelity=0 | Must manually restrict h-grid |
| ZNE fails at N=10 with 3 layouts | R²<0.05, 6/6 losses | Exponential mitigation cost |
| Ladder topology fails with HVA p=2 | ΔE/gap=203% | Coordination number 3 needs deeper circuits |
| Heisenberg XXZ fails with HVA p=2 | Max fid=48% (Néel), 22% (|+⟩) | GS too entangled for 2 layers |
| XY model (Δ=0) fails with HVA p=2 | Max fid=23% | Same — shallow circuits insufficient |
| N=12 too slow for local iteration | 14+ min for Phase 1 alone | 2^12 exact diag on single core |
| Predictor is NOT the bottleneck (N≥10) | V7 2B: QRC=MPNN, both ceiling-limited | No ML improvement possible |
| Noise-aware training fails under shot noise | V7 5B: 6× worse than noiseless | Only coherent errors could help |
| SPSA refinement hurts warm-start | V7 4B: -146% to -356% | Don't refine good predictions |
| optimization_level=1 for noisy sim | Tested: 3× SLOWER (more gates = more noise channels) | Always use level 2 for noisy simulation |
| DD on FakeTorino (XY4) | YGate not in basis — pass fails silently | DD only testable on real hardware via EstimatorV2 options |
| More N=10 noisy simulation experiments | A, A', B all exhausted; R² never >0.08 | Go to real hardware — local sim cannot validate ZNE at N=10 |
| MAX_CES_RATIO < 10 with 5+ layouts | Layout search becomes 45+ min (too expensive) | Keep MAX_CES_RATIO=10, accept outlier filtering at 3 layouts |
| Gate folding locally (nf=3,5) | Folded circuits are 3-5× heavier to simulate | Only viable on real hardware via Runtime ZNE options |
| MPS chi is NOT the bottleneck for 1D HVA | V7 3A/3B: chi=64=chi=256 (identical) | Use chi=64 for speed |
| Transfer learning N→N' fails | V7 TL: baseline wins by 7%, different θ landscapes | Don't pre-train across system sizes |
| p=1 valid regime (chain_1d): N=6 h≥1.6, N=10 h≥1.9, N=20 h≥2.25 | Exp 6A/6B/6D + Verification R1 | Boundary shifts +0.25 to +0.40 vs p=2 |
| p=1 valid regime (ladder): N=6 h≥2.0, N=10 h≥3.0 (safe: h≥3.25) | Verification R1 (2026-05-30) | +1.0 shift vs p=2 at N=10 |
| p=1 valid regime (triangular): N=6 h≥4.0, N=10 h≥3.5 (safe: h≥4.25) | Verification R1 (2026-05-30) | Largest shift at N=6 |
| p=1 seed-independent only at N≤10 | Exp 6A: identical across seeds; 6B: seed 44 fails at N=20 | N=20 needs better init (analytical guess) |
| p=1 frustrated topologies: ~33% chain break rate per seed | Verification R1 (2026-05-30) | Seed 43 → ladder breaks, seed 44 → triangular breaks |
| p=1 θ_x constant (±3π/8) at N=20 | Exp 6B: same |θ_x| across all h and seeds | Only θ_zz varies; Z₂ sign symmetry |
| p=1 MPNN needs sign canonicalization | Exp 6B: seeds find ±θ → inconsistent targets | NOT needed — C3 proved warm-start resolves this with 3 restarts |
| p=1 N=20 needs >6 training points | Exp 6B: only h=3.0 passes deployment | Use 15-20 pts in [2.25, 4.0] |
