# Benchmark & Experiment Runner Patterns

> Lessons learned from the Mitigation Benchmark implementation (2026-06-18).
> Apply these patterns to ALL future experiment runners and benchmarks.

---

## Pattern 1: Never Use θ=zeros for Bound Circuits

**Problem**: `RZZ(0) = Identity`. Qiskit's transpiler at `optimization_level≥1`
correctly identifies and cancels identity gates, producing a circuit with 0 CZ
gates — no noise, no decoherence, no meaningful benchmark.

**Rule**: Any circuit that will be transpiled MUST use non-trivial parameters.

**Implementation**:
```python
# ❌ WRONG — will be cancelled by transpiler
params = np.zeros(len(theta))

# ✅ CORRECT — run quick VQE to get physically meaningful θ_opt
from qmbp_simulation import VQEConfig, VQEOptimizer
from qmbp_simulation.execution import NoiselessBackend

vqe_config = VQEConfig(n_restarts=1, maxiter=100)
optimizer = VQEOptimizer(config=vqe_config, seed=42)
H = HamiltonianBuilder().build(lattice)
result = optimizer.optimize(H, circuit, rng.uniform(-0.01, 0.01, n_params))
params = result.theta_opt
```

**Validation**: After transpilation, always check `n_2q_gates > 0`:
```python
transpiled = pm.run(bound_circuit)
n_2q = sum(1 for i in transpiled.data if i.operation.num_qubits == 2)
assert n_2q > 0, "Circuit has 0 2Q gates — θ values likely zero/trivial"
```

---

## Pattern 2: Mitiq Needs Logical (Unmapped) Observables

**Problem**: `make_mitiq_executor(observable, backend, config, transpile=True)`
internally transpiles the circuit and maps the observable via `apply_layout()`.
If you pass an already-mapped observable (133-qubit), it fails with dimension
mismatch when re-mapping.

**Rule**: Mitiq executors receive the **logical** circuit and **unmapped**
observable. IBM-native executors receive the **transpiled** circuit and
**layout-mapped** observable.

**Implementation**:
```python
# Build both versions
H = HamiltonianBuilder().build(lattice)        # 10-qubit logical
H_mapped = H.apply_layout(transpiled.layout)   # 133-qubit physical

# Route correctly
if config.is_mitiq:
    result = _execute_mitiq(logical_circuit, H, backend)  # unmapped
else:
    result = _execute_native(transpiled, H_mapped, backend)  # mapped
```

---

## Pattern 3: Gate-Folding Requires Odd Integer Factors

**Problem**: Gate-folding ZNE folds circuits by repeating gate blocks an odd
number of times (1, 3, 5, ...). Float factors like 1.5 get truncated to int(1)
which duplicates the base factor, producing effectively fewer extrapolation
points.

**Rule**: Use only odd integers for gate-folding factors. Mitiq ZNE supports
fractional factors (it uses random partial folding).

**Implementation**:
```python
# For IBM-native gate-folding:
zne_noise_factors = [1.0, 3.0, 5.0]  # Must be odd integers

# For Mitiq ZNE (supports fractions):
scale_factors = (1.0, 1.5, 2.0, 3.0)  # Fractional OK

# Deduplication guard:
raw_factors = config.zne_noise_factors or [1, 3, 5]
noise_factors_int = tuple(max(1, int(round(f)) | 1) for f in raw_factors)
seen = set()
noise_factors_int = tuple(x for x in noise_factors_int if x not in seen and not seen.add(x))
```

---

## Pattern 4: AQC Compression Needs VQE-Optimized Target

**Problem**: AQC-Tensor compresses a "target" circuit by finding a shallower
equivalent. If the target is bound with trivial parameters (zeros or random),
the compression is faithful to a MEANINGLESS state — producing correct fidelity
metrics but garbage energy relative to the Hamiltonian ground state.

**Rule**: The AQC target circuit MUST be bound with VQE-optimized θ for the
target depth (p=2 if compressing p=2→p=1-equivalent).

**Implementation**:
```python
# ❌ WRONG — compresses a random state
params = rng.uniform(-0.5, 0.5, size=len(theta_p2))
target = circuit_p2.assign_parameters(params)
compressed = compressor.compress(target)  # High fidelity to WRONG state

# ✅ CORRECT — compresses the ground state approximation
vqe = VQEOptimizer(VQEConfig(n_restarts=1, maxiter=200), seed=42)
result = vqe.optimize(H, circuit_p2, init_params)
target = circuit_p2.assign_parameters(result.theta_opt)
compressed = compressor.compress(target)  # Approximates ground state
```

---

## Pattern 5: `transpiled_circuit_stats()` Returns "depth" Not "depth_transpiled"

**Problem**: The canonical function `transpiled_circuit_stats()` returns the key
`"depth"` for total circuit depth. Code referencing `"depth_transpiled"` will
silently get 0 or KeyError.

**Rule**: Always use `stats["depth"]` or `stats.get("depth", 0)` when reading
the output of `transpiled_circuit_stats()`. If you need `depth_logical`, compute
it BEFORE transpilation and inject it into the stats dict.

```python
stats = transpiled_circuit_stats(transpiled)
stats["depth_logical"] = logical_circuit.depth()  # inject before derivation
```

---

## Pattern 6: Cache Expensive Operations in h-Outer Loops

**Problem**: Benchmark loops iterate `configs × h_values`. If the inner loop
rebuilds circuits and transpiles for each config at the same h, this wastes
~70% of compute time (transpilation is ~2-5s per call).

**Rule**: Loop h-values as OUTER, configs as INNER. Cache per h:
- Logical circuit (same for all non-AQC configs)
- Transpiled circuit per optimization_level
- ClassicalSolver results (e_exact, gap)

```python
for h in h_values:
    circuit = _build_circuit(h)  # once per h
    transpiled_by_level = {
        lvl: pm(lvl).run(circuit) for lvl in levels_needed
    }
    for config in configs:
        transpiled = transpiled_by_level[config.optimization_level]
        # ... execute
```

---

## Pattern 7: Validate Circuit After Transpilation (Guard Against Silent Corruption)

**Problem**: Several bugs (θ=zeros, wrong opt_level) produce transpiled circuits
that are syntactically valid but semantically corrupt (no 2Q gates, wrong depth).
These silently produce garbage results without errors.

**Rule**: Add a post-transpilation sanity check in any benchmark runner:

```python
def _validate_transpiled(transpiled, config_id, h_value):
    """Guard against silent circuit corruption."""
    n_2q = sum(1 for i in transpiled.data if i.operation.num_qubits == 2)
    if n_2q == 0 and not config_id.startswith("C0"):  # C0_raw might legitimately have 0
        raise RuntimeError(
            f"{config_id} h={h_value}: transpiled circuit has 0 2Q gates! "
            f"Likely θ=zeros bug. Check parameter binding."
        )
    depth = transpiled.depth()
    if depth < 5:
        logger.warning(f"{config_id} h={h_value}: depth={depth} (suspiciously shallow)")
```

---

## Pattern 8: Regression Tests for Every Bug Fix

Every bug fixed in a runner MUST have a corresponding regression test that:
1. Reproduces the exact failure condition (the bug)
2. Asserts the fixed behavior
3. Is fast enough to run in CI (<10s)

Template:
```python
def test_regression_<bug_description>():
    """Regression: <one-line bug description>.

    Bug: <what happened>
    Fix: <what was changed>
    """
    # Reproduce the bug condition
    ...
    # Assert the fix works
    assert <fixed_behavior>, "<failure message explaining what regressed>"
```

---

## Quick Reference: Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| θ=zeros | n_2q=0 after transpile | Use VQE θ_opt |
| H_mapped to Mitiq | "qargs does not match" | Pass H_logical |
| noise_factors with floats | Duplicate points | Use odd ints + dedup |
| AQC with random params | ΔE/gap >> 100% | VQE θ_opt(p=2) target |
| `stats["depth_transpiled"]` | KeyError or 0 | Use `stats["depth"]` |
| Backend per iteration | Slow (FakeTorino reload) | Hoist to outer loop |
| No post-transpile check | Silent garbage results | Assert n_2q > 0 |
