---
inclusion: fileMatch
fileMatchPattern: "tests/**,scripts/**,project_health/**,scripts/validation/**, src/qmbp_simulation/**"
---

# Reuse Existing Infrastructure

Reuse what already exists before creating anything new. The three rules below are mandatory.

## Rule 1: Testing — use existing tests

When asked to test or verify functionality:

1. Check if a test already exists in `tests/` (see mapping below).
2. If a test exists, run it, extend it, or add a new case to that file.
3. If none exists, add a new test in the correct location per the module-to-test mapping.
4. Do not create temporary `_tmp_test_*.py` or `scripts/validation/_tmp_*.py` files.
5. Do not create standalone scripts to "check if something works" — use pytest.

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

## Rule 2: Analysis and results inspection — use project_health/

When asked to analyze data, inspect results, or understand outputs:

1. Check the `analysis-tooling.md` decision tree for the right command.
2. If a tool exists, use it (digest, analyzer, sanity_check, etc.).
3. If a tool is close but insufficient, extend it with a new flag/option.
4. Do not create new `project_health/analyze_*.py` or `project_health/inspect_*.py` files.
5. Do not write inline Python to parse JSON results — use the scanner/digest.

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

## Rule 3: New runners — subclass ValidationRunner

When creating a new experiment runner, subclass `ValidationRunner` rather than writing a standalone `main()` script. Then:

1. Call `self.setup_physics()` in `setup()` instead of duplicating imports.
2. Use `self.select_backend(N)` instead of `if N <= 22: NoiselessBackend() else: MPSBackend(...)`.
3. Use `self.save_checkpoint()` for long loops instead of raw `json_dump` to custom paths.
4. Call `self.log_memory_estimate(N)` before large computations.
5. Return `{"pass": bool, ...}` from section functions.
6. See `infrastructure.md` for the full template.

## Anti-patterns to avoid

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
