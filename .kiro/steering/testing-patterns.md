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
- `chain_6` → `LatticeConfig` for N=6 chain
- `h_values_reduced` → 6-point h-grid for fast tests
- `exact_data_reduced` → pre-computed ground truth for 6 points

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

## Adding New Tests

When adding a new feature to `src/poc/v6/`:
1. Add unit tests in the appropriate `Test<Feature>` class
2. If it's a new module, create `tests/test_<module>.py`
3. Add integration coverage in `TestHardwareDeployerSimulation` if it affects the pipeline
4. Run `make test` to verify, then `make check-full` for the complete gate

## Module-to-Test Mapping

| Module | Test file(s) |
|--------|-------------|
| `config.py`, `hamiltonian_builder.py`, `classical_solver.py` | `test_v6_pipeline.py` |
| `vqe_optimizer.py`, `hva_builder.py` | `test_v6_pipeline.py` |
| `hardware_deployer_v61.py` | `test_v6_pipeline.py`, `test_v61_integration.py`, `test_noisy_simulation.py` |
| `analysis_utils.py` | `test_analysis_utils.py` |
| `diagnostics.py` | `test_diagnostics_correctness.py`, `test_diagnostics_integration.py` |
| `mpnn_predictor.py` | `test_v61_integration.py` (edge features, per-param heads) |
