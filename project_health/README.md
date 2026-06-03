# Project Health Checker

Unified orchestration of Phase 4 analysis tools. Produces a single
health report covering experiments, coverage, and actionable items.

## Usage

```bash
# Full text report
python -m project_health

# JSON output
python -m project_health --json

# Summary only (no per-experiment table)
python -m project_health --compact

# Save to file
python -m project_health -o health_report.txt
python -m project_health --json -o health_report.json

# Only show what changed since last run
python -m project_health --diff-only

# CI mode (exit code 1 if CRITICAL gaps exist)
python -m project_health --ci

# Skip state persistence (no delta tracking)
python -m project_health --no-state

# Verbose logging
python -m project_health -v
```

## Architecture

```
project_health/
├── __init__.py       Public API (HealthReport, CoverageGap, ActionItem)
├── __main__.py       CLI entry point (--json, --compact, --diff-only, --ci)
├── engine.py         Core orchestration (scan → analyze → report)
├── coverage.py       Coverage gap detection logic + stats
├── models.py         Data models (all typed, serializable)
├── reporter.py       Output formatters (text, JSON)
├── state.py          Persistence for delta tracking
└── README.md         This file
```

## Design Principles

1. **No re-parsing** — uses `ResultScanner` from `scripts/digest/` as the
   single data source. No duplicate JSON parsing.

2. **No re-evaluation** — experiment verdicts come pre-computed from the
   scanner (which uses `framework/criteria.py`). We read, not recalculate.

3. **Pure logic** — `engine.py` produces a `HealthReport` dataclass with
   zero I/O. Formatting is a separate concern in `reporter.py`.

4. **Delta tracking** — persists seen files in `.project_health_state.json`.
   On subsequent runs, new AND removed results are highlighted.

5. **CI-friendly** — `--ci` flag exits non-zero on CRITICAL gaps.
   `--json` output is machine-parseable.

6. **Doesn't replace tools** — each Phase 4 tool (digest, scan_coverage,
   verify_claims) remains usable independently for deep-dive analysis.

## Dependencies

- `scripts/digest/scanner.py` → `ResultScanner` (scan + parse)
- `scripts/digest/models.py` → `NoiselessResult`, `NoisyResult`, `ExperimentResult`
- `qmbp_simulation.framework.criteria` → `EXPERIMENT_CRITERIA`, `compute_verdict`

## What It Reports

| Section | Source |
|---------|--------|
| Experiment verdicts | `ResultScanner` → `ExperimentResult.verdict` |
| Noiseless quality (global + per-topology) | `ResultScanner` → `NoiselessResult.delta_e_over_gap` |
| Noisy/ZNE quality | `ResultScanner` → `NoisyResult.success_criteria_met` |
| Coverage gaps | Logic in `coverage.py` over scan results |
| Delta since last run (new + removed) | File-set diff via `state.py` |
| Actionable items | Derived from gaps in `coverage.py` |

## Gap Types Detected

| Gap | Priority | Description |
|-----|----------|-------------|
| `MISSING_ZNE` (heavy_hex N≥10) | CRITICAL | Blocks hardware deployment |
| `MISSING_EXPERIMENT` | HIGH | Defined in criteria.py but no results |
| `MISSING_P1_NOISELESS` | MEDIUM | p=2 exists but p=1 missing |
| `INSUFFICIENT_SEEDS` | MEDIUM | < 3 seeds for reproducibility |
| `MISSING_ZNE` (other) | MEDIUM | p=1 noiseless without ZNE validation |
| `INVALID_REGIME` | LOW | h_test below valid boundary (expected failures) |

## Programmatic Usage

```python
from project_health.engine import run_health_check
from project_health.models import Priority

report = run_health_check()

# Check if deployment-ready
critical = [a for a in report.actions if a.priority == Priority.CRITICAL]
if critical:
    print("NOT ready for hardware deployment")

# Access per-topology stats
for topo, stats in report.noiseless_by_topology.items():
    print(f"{topo}: {stats['pass_rate']:.0%} pass rate")
```
