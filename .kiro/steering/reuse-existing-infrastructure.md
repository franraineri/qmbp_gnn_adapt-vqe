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
| `qmbp_simulation.execution.hardware` | `tests/unit/test_layout_optimizer.py`, `tests/integration/test_layout_optimizer_integration.py` |
| `qmbp_simulation.optimizers` | `tests/unit/test_optimizers.py` |
| `qmbp_simulation.predictors` | `tests/unit/test_predictors.py` |
| `qmbp_simulation.pipeline` | `tests/unit/test_pipeline.py` |
| `qmbp_simulation.framework` | `tests/unit/test_framework.py` |
| `qmbp_simulation.analysis` | `tests/unit/test_analysis.py` |

### How to Run Tests

```bash
make test              # Fast tests only (~12s)
make test-full         # All tests including @pytest.mark.slow (~60s)
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
| "Quick health by model/topology?" | `python -m project_health --diagnose --model <m>` |
| "What does the result look like?" | `python project_health/cli/inspect_noiseless_run.py --latest <exp>` |
| "Is something broken?" | `python -m project_health.analysis.sanity_check` |
| "Compare two runs" | `python -m project_health.digest --compare folder_A folder_B` |
| "Load results programmatically?" | `from qmbp_simulation.framework import load_results_from_dir` |
| "Query index fast?" | `from qmbp_simulation.framework.result_index import ResultIndex` |
| "Validate new feature" | Add test cases to existing test file, run `make test` |

## Rule 3: New Runners (ALWAYS subclass ValidationRunner)

When creating a new experiment runner:

1. **MUST** subclass `ValidationRunner` — never write standalone scripts with `main()`.
2. **MUST** call `self.setup_physics()` in `setup()` — never duplicate imports.
3. **MUST** use `self.select_backend(N)` — never `if N <= 22: NoiselessBackend() else: MPSBackend(...)`.
4. **MUST** use `self.save_checkpoint()` for long loops — never raw `json_dump` to custom paths.
5. **MUST** use `self.log_memory_estimate(N)` before large computations.
6. **MUST** return `{"pass": bool, ...}` from section functions.
7. See `infrastructure.md` for the full template.

## Anti-Patterns (NEVER DO)

- ❌ `*_tmp_test_*.py` — use pytest in `tests/`
- ❌ `project_health/analyze_<new_thing>.py` — extend closest existing analyzer
- ❌ `project_health/inspect_*.py` — use digest with appropriate `--kind`
- ❌ Writing a new script to "quickly check" something — write a test or extend existing ones instead
- ❌ `python -c "import json; ..."` for result inspection — use digest
- ❌ Creating throwaway scripts for analysis — extend project_health tools
- ❌ `with open(f) as fh: json.load(fh)` — use `load_result(path)` or `load_results_from_dir(dir)`
- ❌ Standalone runner scripts without `ValidationRunner` — subclass it, get all features free
- ❌ Manual `NoiselessBackend() if N <= 22 else MPSBackend(...)` — use `self.select_backend(N)`
- ❌ Duplicating builder/solver/hva imports — use `self.setup_physics()`
- ❌ `except Exception: pass` — at minimum `logger.debug("context: %s", e)`
- ❌ Custom `_json_default` functions — use `json_serialize` from `utils.helpers`
- ❌ Manual `update_project_status.py` — runners auto-refresh; use `--refresh-status` if needed
