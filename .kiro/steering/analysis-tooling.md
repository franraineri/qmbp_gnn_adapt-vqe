---
inclusion: fileMatch
fileMatchPattern: "analysis/**,scripts/**,experiments/**,results/**"
---

# Analysis Tooling — ALWAYS Use Existing Scripts

## Rule (MANDATORY)

When analyzing data, inspecting results, or producing output files:

**ALWAYS use the existing analysis and digest scripts first.**
Do NOT write ad-hoc one-off scripts or inline Python when an existing tool covers the need.
If the existing tool is close but not sufficient, **extend it** rather than creating a new file.

## Available Tools (use these)

| Tool | Purpose | Usage |
|------|---------|-------|
| `scripts/digest/run_digest.py` | Quick overview of all results | `python scripts/digest/run_digest.py [--kind noiseless] [--group-by topology] [--model tfim_longitudinal]` |
| `analysis/scan_coverage.py` | Coverage scan, gap analysis, extended analytics | `python analysis/scan_coverage.py [--discover] [--extended] [--topology X] [--p N]` |
| `analysis/diagnose.py` | Automated failure root cause analysis | `python analysis/diagnose.py [--all] [path] [--severity fail]` |
| `scripts/compare.py` | Cross-experiment result comparison | `python scripts/compare.py [--all] [--category X] [--noisy]` |
| `scripts/experiment_runners/verify_results.py` | Post-verification analysis | `python scripts/experiment_runners/verify_results.py` |
| `scripts/experiment_runners/run_verification_plan.py` | Systematic claim validation | `python scripts/experiment_runners/run_verification_plan.py [--list] [--noiseless-only]` |
| `analysis/run_analysis.py` | Full analysis pipeline | `python analysis/run_analysis.py` |
| `analysis/generate_figures.py` | Generate thesis figures | `python analysis/generate_figures.py` |

## Decision Process

1. **Need to inspect results?** → `scripts/digest/run_digest.py`
2. **Need to check what data exists?** → `analysis/scan_coverage.py --discover`
3. **Need to understand a failure?** → `analysis/diagnose.py`
4. **Need to compare experiments?** → `scripts/compare.py`
5. **Need to validate a claim?** → `scripts/experiment_runners/verify_results.py`
6. **Need to check/validate a variant runner script?** → `python scripts/preflight.py --from-script <path>`
7. **Need something the tools don't cover?** → Extend the closest existing tool

## Preflight Validation (MANDATORY for runner scripts)

When reviewing, checking, or validating any variant runner script (`run_*.py`), **ALWAYS use
`scripts/preflight.py`** instead of manually reading the file and checking constraints by hand.

```bash
# Validate a variant runner script before execution
python scripts/preflight.py --from-script scripts/experiment_runners/run_p1_pipeline_variants_r2.py

# Strict mode (warnings = errors)
python scripts/preflight.py --from-script my_script.py --strict

# Validate from JSON config
python scripts/preflight.py --from-json variants.json
```

Preflight checks 9 validations automatically:
1. Pipeline script exists on disk
2. Minimum config present (h_values, h_test, topology, n_qubits)
3. h_test NOT in training set (data leakage)
4. h_test within valid regime for topology/N/p
5. Training h_values within valid regime
6. h_test is interpolation (not extrapolation)
7. h_values in descending order (warm-start requirement)
8. No duplicate variant IDs
9. Output directories are fresh (no collision)

**DO NOT** manually open a runner script and check these constraints by reading the code.
The preflight tool does it faster, more reliably, and catches edge cases you'd miss.

## Extension Guidelines

When extending an existing script:
- Add new flags/options rather than changing existing behavior
- Keep backward compatibility (existing invocations must still work)
- Add the new capability to the tool's `--help` output
- Update this steering file if a new major capability is added

## Anti-Patterns (DO NOT)

- ❌ Writing `analysis/_tmp_*.py` throwaway scripts for one-off analysis
- ❌ Using `python -c "..."` for multi-line data inspection
- ❌ Creating new scripts in `scripts/` when `digest/` or `analysis/` already covers it
- ❌ Manually parsing JSON result files when `scan_coverage.py` or `diagnose.py` can do it
- ❌ Duplicating formatting/reporting logic that exists in `scripts/digest/formatters.py`

## When a New Script IS Justified

A new standalone script is appropriate ONLY when:
1. The analysis is fundamentally different from all existing tools (new experiment type)
2. It will be reused multiple times (not a one-off)
3. It belongs to a new category not covered by existing tools
4. It's registered in the experiment framework (`experiments/<category>/exp_*.py`)

In that case, place it in the correct location:
- Experiment scripts → `experiments/<category>/`
- Reusable analysis → extend `analysis/scan_coverage.py` or `analysis/diagnose.py`
- Result formatting → extend `scripts/digest/`
