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
python src/qmbp_simulation/framework/preflight.py --from-script scripts/run_<name>.py
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
python -m project_health.digest --kind experiment --sort verdict

# 2. Does compare.py work without errors?
python project_health/cli/compare.py --all

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
| `documentation/binnacles/binnacle-mpnn-eval-suite.md` | Addendum if MPNN evaluation was run |
| `analysis/10_key_findings_corrected.md` | New hallazgo # (if novel finding) |
| `documentation/analysis/08_summary.md` | Session entry (date, experiments, verdicts) |
| `documentation/analysis/09_thesis_tables.md` | Only if new table for Chapter 5 |

### For Hardware-Adjacent Experiments

After running MPNN evaluation suite (sections 10-19 via V3 runner):

```bash
# Analyze all V3 runs
python -m project_health.analysis.mpnn_eval_analyzer --thesis-table

# The analyzer auto-generates warnings for:
# - speedup < 1.5x (retrain before QPU)
# - LOO pass_rate < 60% (extend h_train)
# - topology_transfer ratio > 3x (GNN is topology-specific)
# - κ |r| < 0.50 (κ not reliable for this topology)
```

### Quick-Reference Checklist

```
□ Script created from template (ValidationRunner / ExperimentRunner)
□ preflight.py --from-script → PASS (or justified WARNINGs only)
□ --dry-run shows correct sections
□ Execution complete → JSON saved in results/experiments/
□ python -m project_health.digest --kind experiment → experiment visible with correct verdict
□ python project_health/cli/compare.py --all → no crash, experiment listed
□ python analysis/verify_claims.py → no new contradictions
□ python analysis/scan_coverage.py → check if gaps closed
□ Binnacle created/updated with cross-references
□ 10_key_findings updated (if new finding)
□ 08_summary updated (session entry)
```

---

## Runner Standards (reference)

See `.kiro/steering/runner-standards.md` for full runner base class documentation,
templates, anti-patterns, and the 4 runner types (ValidationRunner, ExperimentRunner,
VariantPipelineRunner, HardwareValidationRunner).

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
4. Compare against previous run (auto-generated by `project_health/cli/compare.py`)

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

- Unified CLI for running experiments by ID
- Discovers experiments from `experiments/` category directories
- Supports `--exp A3 B4 F3` for multiple experiments
- Supports `--list` to show all registered experiments
- Supports `--verbose` / `-v` for INFO logging + StructuredLogger output
- Supports `--debug` for DEBUG logging + all verbose features
- Exit codes: 0=all pass, 1=some fail, 2=validation fail

- Full 4-phase pipeline CLI
- Accepts lattice config, h-values, and output directory as arguments
- Supports skip/resume via phase flags and checkpoint detection
- Uses `PipelineRunner` from `qmbp_simulation.pipeline`

### `project_health/cli/compare.py`
- Cross-experiment result comparison
- Supports `--all` for comparing all results
- Produces structured JSON output + terminal summary

- Imports all public submodules of `qmbp_simulation`
- Runs minimal pipeline (N=4, p=1, 3 h-points)
- Verifies ΔE/gap < 5%
- Completes in under 30 seconds
- Exit non-zero on failure with descriptive error

- Consolidates definitive runs into thesis tables
- Computes mean±std across seeds for each h_test
- Produces `results/thesis/thesis_results_<timestamp>.json`

## When to Re-Run

| Change | Re-run |
|--------|--------|
| Modified `src/qmbp_simulation/analysis/` | `make test` |
| Modified `src/qmbp_simulation/framework/` | `make test` + run one experiment to verify |

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

## ZNE Scaling Rules

- p=1 N=10 ≈ 18 CX → ZNE works. p=2 N=10 ≈ 36 CX → ZNE fails.
- **p=1 + ZNE is the recommended strategy for N≥10.**
- PEA primary, gate-folding fallback. See `context-zne-mitigation.md` for full details.

## Experiment Value Checklist (Before Running)

Ask these questions before starting any experiment:

1. **What hypothesis does this test?** (If none, don't run it.)
2. **What would I learn if it passes?** (If "nothing new," don't run it.)
3. **What would I learn if it fails?** (If "nothing new," don't run it.)
4. **Has this already been established?** (Check binnacle — if 3 seeds already confirm, don't re-run.)
5. **Is this a physics limit or a tunable parameter?** (If physics limit confirmed, no hyperparameter will help.)

## Known Physics Limits

See `.kiro/steering/project-status.md` → "Key Constraints" section for the complete
list of validated physics limits. The most critical ones for experiment design:

- h_min valid regime shifts with N: N=6→h≥1.25, N=10→h≥1.5, N=20→h≥2.0 (p=2)
- p=1 valid regime: h_min = 2.36 + 0.0073·N (linear, R²=0.91); p≥3: ~1.6 constant
- ZNE CX threshold: ~18 CX (p=2 N=10 = 36 CX → fails)
- HVA is TFIM-specific (Heisenberg/XY fail with p≤2)
- N=12 too slow for iteration (~30+ min)
- Predictor is NOT the bottleneck at N≥10

**Full table with evidence**: `.kiro/steering/project-status.md`
**Full validated decisions**: `.kiro/knowledge/validated-decisions.md`

## Analysis: Reuse Existing Scripts

Do not create new analysis scripts when an existing one covers the need. See `scripts/analysis/` and `project_health/analysis/`.
