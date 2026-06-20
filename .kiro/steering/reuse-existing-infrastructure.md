---
inclusion: fileMatch
fileMatchPattern: "tests/**,scripts/**,project_health/**,scripts/validation/**, src/qmbp_simulation/**"
---

# Reuse Existing Infrastructure — MANDATORY

## Rule 1: Testing (ALWAYS USE EXISTING TESTS)

When asked to test or verify functionality:

1. **FIRST**: Check if a test already exists in `tests/` (see mapping below).
2. **If test exists**: Run it, extend it, or add a new test case to the existing file.
3. **If no test exists**: Add a new test in the correct location per the module-to-test mapping.
4. **NEVER** create temporary `_tmp_test_*.py` or `scripts/validation/_tmp_*.py` files.
5. **NEVER** create standalone scripts just to "check if something works" — use pytest.

### Module-to-Test Mapping

| Package module | Test file |
|----------------|-----------|
| `qmbp_simulation.utils` | `tests/unit/test_utils.py` |
| `qmbp_simulation.models` | `tests/unit/test_models.py` |
| `qmbp_simulation.solvers` | `tests/unit/test_solvers.py` |
| `qmbp_simulation.circuits` | `tests/unit/test_circuits.py` |
| `qmbp_simulation.execution` | `tests/unit/test_execution.py` |
| `qmbp_simulation.execution.hardware` | `tests/unit/test_layout_optimizer.py`, `tests/integration/test_layout_optimizer_integration.py` |
| `qmbp_simulation.execution.mitiq_utils` | `tests/test_mitiq_integration.py` |
| `qmbp_simulation.optimizers` | `tests/unit/test_optimizers.py` |
| `qmbp_simulation.predictors` | `tests/unit/test_predictors.py` |
| `qmbp_simulation.pipeline` | `tests/unit/test_pipeline.py` |
| `qmbp_simulation.framework` | `tests/unit/test_framework.py` |
| `qmbp_simulation.analysis` | `tests/unit/test_analysis.py` |
| `project_health` | `tests/test_project_health_coverage.py` |
| `project_health.digest` | `tests/test_digest.py` |
| MPNN eval helpers | `tests/test_mpnn_eval_helpers.py`, `tests/mpnn/test_mpnn_eval_extended.py` |

### How to Run Tests

```bash
make test              # Fast tests only (~12s)
make test-full         # All tests including @pytest.mark.slow (~60s)
pytest tests/unit/test_circuits.py -v   # Specific module
pytest tests/ -k "test_specific_name"   # By name
```

### When Extending Tests

- Add to the existing `Test<Feature>` class in the correct file.
- Use `@pytest.mark.parametrize` for multi-config validation.
- Use `@pytest.mark.slow` if the test takes >10s (needs FakeTorino, etc.).
- Use `tmp_path` fixture for file I/O.
- Use `np.testing.assert_allclose(actual, expected, atol=1e-6)` for numerics.

## Rule 2: Analysis & Results Inspection (ALWAYS USE project_health/)

When asked to analyze data, inspect results, or understand outputs:

1. **FIRST**: Check `analysis-tooling.md` decision tree for the right command.
2. **If a tool exists**: Use it (digest, analyzer, sanity_check, etc.).
3. **If tool is close but insufficient**: Extend it with a new flag/option.
4. **NEVER** create new `project_health/analyze_*.py` or `project_health/inspect_*.py` files.
5. **NEVER** write inline Python to parse JSON results — use the scanner/digest.

### Quick Decision Shortcuts

| Need | Use |
|------|-----|
| "Does this work?" | `pytest tests/ -k <relevant>` |
| "What does the result look like?" | `python -m project_health.digest --kind <kind>` |
| "Is something broken?" | `python -m project_health.analysis.sanity_check` |
| "Compare two runs" | `python -m project_health.digest --compare folder_A folder_B` |
| "Validate new feature" | Add test cases to existing test file, run `make test` |

## Anti-Patterns (NEVER DO)

- ❌ `*_tmp_test_*.py` — use pytest in `tests/`
- ❌ `project_health/analyze_<new_thing>.py` — extend closest existing analyzer
- ❌ `project_health/inspect_*.py` — use digest with appropriate `--kind`
- ❌ Writing a new script to "quickly check" something — write a test or extend existing ones instead
- ❌ `python -c "import json; ..."` for result inspection — use digest
- ❌ Creating throwaway scripts for analysis — extend project_health tools
