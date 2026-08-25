---
inclusion: fileMatch
fileMatchPattern: "tests/**"
---

# Testing Patterns — qmbp_simulation Test Suite

## Test Structure

Tests live in `tests/` and use pytest. Run with `make test` (~8s) or `python -m pytest tests/ -v`.

### Fixture Hierarchy (conftest.py)

- `_isolate_results_io` → **GLOBAL, autouse=True** — redirects ALL `save_experiment_result()` calls to `tmp_path`. Prevents tests from polluting `results/experiments/` with garbage entries. Every test gets this automatically.
- `seed_rng` → pins numpy/torch random seeds (autouse)
- `small_lattice` → `LatticeConfig` for N=4 chain
- `small_circuit` → N=4 p=1 HVA circuit
- `noiseless_backend` → `NoiselessBackend()` instance
- `mock_backend` → constant-energy backend for optimizer tests

### Test Classes

| Class | Tests | What it validates |
|-------|-------|-------------------|
| `TestHamiltonianBuilder` | Hermiticity, dimensions, known energies | Phase 1 correctness |
| `TestClassicalSolver` | Exact diag matches analytical | Phase 1 correctness |
| `TestVQEOptimizer` | Single optimize, descending sweep | Phase 2 correctness |
| `TestHardwareDeployer` | Phase classification logic | Phase 4 correctness |
| `TestPipelineIntegrity` | Save/load roundtrip, metadata | Cross-phase data flow |
| `TestHardwareDeployerSimulation` | Full V6.1 pipeline in simulation | Integration |
| `TestObservableGroupingIntegration` | Grouped vs individual observables | Phase 4 correctness |
| `TestMPNNEdgeFeatures` | NNConv model, backward compat | Phase 3 architecture |
| `TestWeightGradientAnalyzer` | Gradient output structure | Analysis correctness |
| `TestDiagnosticsIntegration` | CLI flags, logging levels, diagnostics output | Observability correctness |
| `TestSeedDeterminism` | Layout reproducibility with same seed | Noisy simulation correctness |
| `TestZNEEdgeCases` | n_layouts=1 fallback, R²<0.8 warning | ZNE edge case handling |

## Conventions

- Test files: `test_<module>.py`
- Test classes: `Test<Feature>`
- Test methods: `test_<behavior>` — describe what's being verified, not how
- Use `@pytest.mark.parametrize` for multi-config tests
- Use `@pytest.mark.slow` for tests that depend on FakeTorino or take >10s (excluded by `make test`, included by `make test-full`)
- Use `tmp_path` fixture for file I/O tests
- Use `np.testing.assert_allclose(actual, expected, atol=1e-6)` for numerical comparisons

### Test Speed Tiers

| Command | Scope | Time |
|---------|-------|------|
| `make test` | Fast tests only (`-m "not slow"`) | ~12s |
| `make test-full` | All tests including `@pytest.mark.slow` | ~60s |

## What to Assert

| Phase | Key assertions |
|-------|---------------|
| Phase 1 | Hamiltonian is Hermitian, correct dimensions (2^N × 2^N), known energies match |
| Phase 2 | VQE energy ≤ exact energy + tolerance, fidelity > 0, results ordered by h |
| Phase 3 | Model output shape matches 2*p, loss decreases, checkpoint roundtrip |
| Phase 4 | Phase label is valid string, ΔE/gap is finite, checklist count ≤ total |
| Analysis | Gradient norms are non-negative, h_values sorted, peaks within valid range |

## Anti-Patterns (don't do these)

- Don't test implementation details (internal variable names, call order)
- Don't use `time.sleep()` — use deterministic conditions
- Don't hardcode seeds in tests unless testing seed-specific behavior
- Don't test against exact floating-point values — use tolerances
- Don't import from `scripts/` in tests — test the `src/` modules directly
- **Don't add module-level `_isolate_runner_output` fixtures** — the global `_isolate_results_io` in `tests/conftest.py` handles all result I/O isolation. Duplicate fixtures are redundant.
- **Don't call `runner.run()` in tests without verifying the global isolation is active** — if you subclass `ValidationRunner` outside `tests/`, any `.run()` call writes real files.

## Adding New Tests

When adding a new feature to `src/qmbp_simulation/`:
1. Add unit tests in the appropriate `Test<Feature>` class in `tests/unit/`
2. If it's a new module, create `tests/unit/test_<module>.py`
3. Add integration coverage in `tests/integration/test_pipeline_e2e.py` if it affects the pipeline
4. Run `make test` to verify, then `make check-full` for the complete gate

## Module-to-Test Mapping

| Package module | Test file(s) |
|----------------|-------------|
| `qmbp_simulation.utils` | `tests/unit/test_utils.py` |
| `qmbp_simulation.models` | `tests/unit/test_models.py` |
| `qmbp_simulation.solvers` | `tests/unit/test_solvers.py` |
| `qmbp_simulation.optimizers` | `tests/unit/test_optimizers.py` |
| `qmbp_simulation.predictors` | `tests/unit/test_predictors.py` |
| `qmbp_simulation.pipeline` | `tests/unit/test_pipeline.py` |
| `qmbp_simulation.framework` | `tests/unit/test_framework.py` |
| `qmbp_simulation.analysis` | `tests/unit/test_analysis.py` |
| Import dependency order | `tests/unit/test_imports.py` |
| Smoke test (all imports) | `tests/integration/test_smoke.py` |
| End-to-end pipeline | `tests/integration/test_pipeline_e2e.py` |
| Backward compatibility | `tests/integration/test_backward_compat.py` |
| `project_health.analysis.diagnose` | `tests/project_health/test_diagnose.py` |
