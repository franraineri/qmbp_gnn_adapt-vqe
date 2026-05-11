---
inclusion: fileMatch
fileMatchPattern: "tests/**"
---

# Testing Patterns — V6 Test Suite

## Test Structure

Tests live in `tests/` and use pytest. Run with `make test` (~8s) or `python -m pytest tests/ -v`.

### Fixture Hierarchy (conftest.py)

- `builder` → `HamiltonianBuilder()` instance
- `solver` → `ClassicalSolver()` instance
- `hva` → `HVACircuitBuilder()` instance
- `h_values_reduced` → 3-point h-grid for fast tests
- `exact_data_reduced` → pre-computed ground truth for 3 points

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

## Conventions

- Test files: `test_<module>.py`
- Test classes: `Test<Feature>`
- Test methods: `test_<behavior>` — describe what's being verified, not how
- Use `@pytest.mark.parametrize` for multi-config tests
- Use `tmp_path` fixture for file I/O tests
- Use `np.testing.assert_allclose(actual, expected, atol=1e-6)` for numerical comparisons

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

## Adding New Tests

When adding a new feature to `src/poc/v6/`:
1. Add unit tests in the appropriate `Test<Feature>` class
2. If it's a new module, create `tests/test_<module>.py`
3. Add integration coverage in `TestHardwareDeployerSimulation` if it affects the pipeline
4. Run `make test` to verify, then `make check-full` for the complete gate
