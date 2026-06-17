# Binnacle — Performance Optimizations

> Registry of all performance optimizations applied to the qmbp_simulation
> pipeline. Tracks profiling data, speedup measurements, validation evidence,
> and rejected approaches with justification.
>
> **Created**: 2026-06-10
> **Scope**: All pipeline components (VQE, ZNE, MPNN, MPS)
> **Principle**: Every optimization must be bit-exact or statistically equivalent.
> No accuracy degradation. Measurable speedup. Minimal code surface change.

---

## Summary Table

| Date | Optimization | Module | Speedup | Validated |
|------|-------------|--------|---------|-----------|
| 2026-06-10 | MPS deterministic mode | `mps_backend.py` | **375×** | ✅ 9/9 tests |
| 2026-06-10 | PEA noise pair filtering + pre-build | `noisy_utils.py` | **5-10×** | ✅ 94 tests, bit-exact |
| 2026-06-10 | Parallel noise factors (N≥14 auto) | `noisy_utils.py` | 1.2× (N=10) | ✅ bit-exact, scalability |
| 2026-06-09 | AerSimulator caching (MPS) | `mps_backend.py` | ~50ms/eval | ✅ In production |
| 2026-06-07 | COBYLA dispatch for MPS | `vqe.py` | ~3× fewer evals | ✅ In production |
| **2026-06-15** | **PauliEvolutionGate circuit repr.** | **`hva.py`** | **−6–10% total_depth** | **✅ Section 20** |

---

## 1. MPS Deterministic Mode (2026-06-10)

### Problem

`MPSBackend(strategy="aer_mps")` created a new `AerSimulator` + `BackendEstimatorV2`
on **every function evaluation**. VQE with COBYLA at N=40 bond-resolved:
79 params × 500 maxiter × 3 restarts × 5 h-points ≈ 7500 evals × ~6s = 4-12 hours.

The transpilation inside `BackendEstimatorV2.run()` was the dominant cost (~500ms/call),
not the MPS simulation (~10ms for N=40 χ=64).

### Solution

Replace `BackendEstimatorV2` with Aer's `save_expectation_value` instruction:
exact ⟨ψ|H|ψ⟩ from the MPS state, no transpilation, deterministic.

```python
# Deterministic: exact, no shot noise, no transpile
qc.save_expectation_value(hamiltonian, list(range(n_qubits)), label="ev")
result = backend.run(qc, shots=1).result()
energy = float(np.real(result.data()["ev"]))
```

### Results

| Mode | Time/eval | Accuracy | Shot Noise |
|------|:---------:|:--------:|:----------:|
| **Deterministic** (new) | **12 ms** | 1.78×10⁻¹⁴ | None |
| Stochastic (legacy) | 6,000 ms | σ ≈ 0.005 | Yes |
| **Speedup** | **375×** | — | — |

### Impact on Experiments

| Experiment | Before | After |
|-----------|:---:|:---:|
| N=40 scaling (5h × 3 seeds) | ~4.5 hours | **~2-3 min** |
| N=50 scaling | ~5 hours | **~3-4 min** |
| E5 Section 2 (N=120 VQE) | ~90s | **~5s** |
| B4 Bond-Resolved (N=40, 79 params) | **4+ hours** | **~10-30 min** |
| N=120 full sweep (5h × 3 seeds) | ~2 hours | **~5-10 min** |

### Design Decisions

1. **Default = deterministic**: All new experiments use exact evaluation.
2. **Stochastic preserved** (`deterministic=False`): For reproducing old results.
3. **Backend caching**: `AerSimulator` created once, reused. Invalidates on N change.
4. **Metadata tagging**: `metadata.mps_evaluation_mode` for traceability.

### Validation

- 9/9 new tests pass (`tests/test_mps_backend_cache.py`)
- Accuracy = machine epsilon (1.78e-14 vs statevector)
- Pre-2026-06-10 results remain valid (noise σ=0.005 ≪ 5% threshold)

### Files

- `src/qmbp_simulation/execution/mps_backend.py` — dual-mode + caching
- `tests/test_mps_backend_cache.py` — 9 validation tests
- `.github/workflows/ci.yml` — added to CI


---

## 2. PEA-ZNE Noise Pair Filtering (2026-06-10)

### Problem

`run_pea_zne()` called `_build_amplified_noise_model()` which constructed
`depolarizing_error()` objects for ALL 300 FakeTorino qubit pairs × 3 noise
factors. The circuit only uses 6-10 qubits (~20-30 relevant pairs).

**Profiling breakdown** (N=6, chain_1d, 3 noise factors, total ~2.85s):

```
Component                          Time     % Total
─────────────────────────────────────────────────────
_build_amplified_noise_model ×3    1.79s    63%  ← BOTTLENECK
  └─ depolarizing_error() ×900     1.79s   (300 pairs × 3 factors)
estimator.run() ×3                 0.28s    10%
AerSimulator.from_backend ×3       0.04s     1%
_learn_noise_rates ×1              0.0003s   0%
BackendEstimatorV2 init ×3         0.002s    0%
Python overhead                    0.74s    26%
─────────────────────────────────────────────────────
TOTAL (run_pea_zne)                2.85s   100%
```

### Root Cause

`depolarizing_error(rate, 2)` is expensive (~2ms per call). Building errors
for 300 pairs × 3 factors = 900 calls × 2ms = 1.8s. But only 20 of those
pairs have qubits used by the circuit — the other 280 are wasted.

**Critical finding**: `BackendEstimatorV2` does NOT transpile on each `run()`.
It only applies `Optimize1qGatesDecomposition` (~0.7ms) for measurement rotations.
The initial hypothesis of "transpilation overhead" was **wrong**.

### Solution

Three optimizations applied together:

1. **Filter noise pairs** (`_filter_rates_to_circuit`): Only build `depolarizing_error`
   for pairs where at least one qubit is in the circuit. Noise on unused qubits
   cannot affect measurement outcomes — proven bit-exact.

2. **Pre-build all noise models** before the measurement loop: Avoids repeated
   Python overhead and allows batch allocation.

3. **Pass pre-built model** to `_pea_estimate()` via `prebuilt_noise_model` parameter:
   Skips redundant model construction inside the per-factor loop.

```python
# New helpers in noisy_utils.py
circuit_qubits = _get_circuit_qubits(transpiled_circuit)      # {8, 17, 27, 28, 29, 36}
relevant_rates = _filter_rates_to_circuit(learned_rates, circuit_qubits)  # 20/300 pairs

# Pre-build all models upfront
noise_models = {nf: _build_amplified_noise_model(backend, circuit, nf, relevant_rates)
                for nf in noise_factors}
```

### Results

| Configuration | Original | Optimized | Speedup | Accuracy |
|---|---|---|---|---|
| N=6, chain_1d, 3 factors | 1.97s | 0.19s | **10.3×** | Bit-exact |
| N=10, chain_1d, 3 factors | 2.96s | 1.56s | **1.9×** | Bit-exact |

**Validation**: `diff = 0.000000e+00` (energy values identical to 15 decimal places).
94 existing noisy/PEA/ZNE tests pass.

### Impact on Experiment Sweeps

| Scenario | PEA Calls | Time Saved |
|---|---|---|
| Single h-sweep (6 pts, 1 seed) | 6 | ~10.7s → 1.1s |
| Multi-seed sweep (6 pts, 3 seeds) | 18 | ~32s → 3.4s |
| Cross-topology (4 topo, 6 pts, 3 seeds) | 72 | ~128s → 13.5s |
| Full PEA deployment (multi-layout) | 216 | ~384s → 40s |

### Why Bit-Exact

Noise on qubit pairs outside the circuit's light cone cannot affect measurement
outcomes. AerSimulator only applies noise channels to gates that actually execute —
unused pairs are never triggered regardless of whether they are in the NoiseModel.

### Files

- `src/qmbp_simulation/execution/noisy_utils.py` — added `_get_circuit_qubits()`,
  `_filter_rates_to_circuit()`, modified `run_pea_zne()` and `_pea_estimate()`
- `scripts/benchmarks/benchmark_pea_performance.py` — permanent regression benchmark

---

## 3. AerSimulator Backend Caching (2026-06-09)

### Problem

`_AerMPSStrategy` created a new `AerSimulator` on every evaluation call,
wasting ~50ms on backend initialization that doesn't change.

### Solution

Cache the backend instance (`_cached_backend`) and invalidate only when
`n_qubits` changes. Also cache the `BackendEstimatorV2` instance for
stochastic mode.

### Result

~50ms saved per evaluation. For VQE loops with 1000+ evals, this saves ~50s.
Combined with deterministic mode (which eliminates the estimator entirely),
this is now only relevant for the stochastic path.

### Files

- `src/qmbp_simulation/execution/mps_backend.py`

---

## 4. COBYLA Dispatch for Shot-Based VQE (2026-06-07)

### Problem

L-BFGS-B uses finite-difference gradients: 2n+1 function evaluations per
iteration. At N=40 with 2 params: 5 evals/iter (manageable). But at bond-resolved
with 79 params: 159 evals/iter × noisy landscape = very slow convergence and
gradient corruption from shot noise.

### Solution

Dispatch to COBYLA (gradient-free) when `config.method == "COBYLA"`. Required
for MPS stochastic mode and all hardware execution. COBYLA needs ~3× fewer
total evaluations on noisy landscapes because it doesn't waste evaluations
on corrupted gradient estimates.

### Files

- `src/qmbp_simulation/optimizers/vqe.py` — method dispatch in `_run_minimize()`

---

## 6. PauliEvolutionGate Circuit Representation (2026-06-15)

### Problem

`HVACircuitBuilder.create()` uses explicit `RZZ`/`RX` gate loops. The transpiler
sees N sequential `RZZ` gates per layer without knowing they represent a commuting
sum — it can't optimize their scheduling relative to the `RX` field layer.

On heavy_hex N=10 p=1: `total_depth` after transpilation = 89–90 cycles.

### Solution

`HVACircuitBuilder.create_pauli_evolution()` wraps each commuting group (the full
ZZ layer and the full X layer) in a `PauliEvolutionGate`. The Qiskit synthesizer
then decomposes with awareness of the full operator structure, scheduling gates more
compactly between 2Q cycles.

**Coefficient convention** (important — bug in original 2026-06-05 version):
```python
# Correct: coefficient=0.5 so that e^{-i·2θ·0.5·ZZ} = e^{-iθ·ZZ} = RZZ(2θ)
H_zz = SparsePauliOp.from_list([("...ZZ...", 0.5) for ...])
H_x  = SparsePauliOp.from_list([("...X...", 0.5) for ...])
```

### Results (Section 20, run_hardware_rehearsal_v3.py)

| h | RZZ total_depth | PauliEvol total_depth | Reduction | n_2Q | \|ΔE\| |
|---|:---:|:---:|:---:|:---:|:---:|
| 4.00 | 89 | 82 | **−7.9%** | 34 (equal) | 3.6e-14 |
| 3.25 | 90 | 81 | **−10.0%** | 34 (equal) | 1.4e-14 |
| 3.00 | 90 | 90 | 0.0% | 34 (equal) | 2.1e-14 |
| **Mean** | **89.7** | **84.3** | **−6.0%** | **34** | |

**2Q-depth = 1 for both** on heavy_hex (non-overlapping ZZ bonds, fully parallelized
by scheduler regardless). The reduction is in total_depth (1Q scheduling between
2Q cycles). On FakeTorino (per-gate noise), energy impact is ~0.67% (noise floor).
On real hardware (time-based decoherence), the shorter circuit reduces T1/T2 errors.

### Applicability

- ✅ **Apply** when transpiling to real hardware or FakeTorino
- ❌ **Don't apply** for noiseless StatevectorEstimator (no transpilation, no benefit)
- ❌ **Not available** for `tfim_longitudinal`, `tfim_frustrated`, Heisenberg (each
  has its own circuit builder without a PauliEvol variant)

### Production Status

Applied to `run_ibm_deployment.py` Tiers 0, 1, 2. VQE training paths
(noiseless) continue to use `create()`.

### Files

- `src/qmbp_simulation/circuits/hva.py` — bug fix + updated docstring
- `scripts/experiment_runners/hardware/run_ibm_deployment.py` — 3 call sites updated
- `scripts/experiment_runners/run_hardware_rehearsal_v3.py` — Section 20 added
- `documentation/binnacles/binnacle-pauli-evolution-transpilation.md` — full details

---

## Rejected Approaches (with justification)

### Pre-transpiling the circuit once for PEA noise factors

**Hypothesis**: The circuit is transpiled inside BackendEstimatorV2 on each call.
**Finding**: **WRONG**. BackendEstimatorV2 does NOT transpile. It only applies
`Optimize1qGatesDecomposition` (~0.7ms) for measurement rotations.
**Conclusion**: No optimization opportunity here.

### Using `save_expectation_value` for PEA (bypass BackendEstimatorV2)

**Hypothesis**: Exact post-noise ⟨H⟩ would be faster and more precise.
**Finding**: `save_expectation_value` computes Tr(ρ·H) deterministically —
removes shot noise entirely. This changes the measurement semantics: PEA
local simulation must include shot noise to predict hardware behavior.
**Conclusion**: Rejected for production PEA. Valid for deterministic MPS.

### Caching `depolarizing_error()` objects across factors

**Hypothesis**: Pre-compute all errors and reuse.
**Finding**: Each pair×factor combination has a unique rate, so there's no
reuse. After filter optimization (20 pairs instead of 300), total error
construction time is ~50ms — not worth additional complexity.
**Conclusion**: Superseded by filtering optimization.

### Parallel noise factor execution (ProcessPoolExecutor)

**Hypothesis**: Run 3 noise factors in parallel for ~3× speedup.
**Finding**: After filter optimization, per-factor time is ~60ms. Parallelism
overhead (process spawn, memory) exceeds the gain at this scale. Would help
only at N≥20 where simulation time dominates.
**Conclusion**: Available but low priority. Implement only if N≥20 PEA needed.

---

## Available (Not Yet Applied)

| Optimization | Module | Expected Speedup | Rationale for not applying |
|---|---|---|---|
| Observable batching (multi-PUB) | `noisy_utils.py` | ~10ms | BackendEstimatorV2 abelian grouping already active; single call/deployment |
| Cached `_learn_noise_rates` per backend | `noisy_utils.py` | 0.3ms | Negligible absolute time |

---

## Benchmark Script

```bash
# Run PEA performance regression check
.venv/bin/python scripts/benchmarks/benchmark_pea_performance.py

# Expected output:
#   N=6:  <1.0s (threshold), typically ~0.3s
#   N=10: <3.0s (threshold), typically ~1.7s
```


---

## Análisis Final: Qué Ayudó y Qué No (2026-06-10)

### Optimizaciones que SÍ ayudan (impacto medido)

| # | Optimización | Speedup Real | Dónde importa | Veredicto |
|---|---|---|---|---|
| 1 | MPS deterministic (`save_expectation_value`) | **375×** | VQE loops N≥22 | 🟢 CRÍTICA — habilitó experimentos imposibles |
| 2 | PEA noise pair filtering | **5-10×** | Toda simulación noisy PEA | 🟢 ALTA — reduce 2s→0.2s por PEA call |
| 3 | Pre-build noise models | **~2×** adicional sobre #2 | Incluido en #2 | 🟢 MEDIA — elimina overhead de loop |
| 4 | COBYLA dispatch | **~3× menos evals** | VQE shot-based (MPS stochastic) | 🟢 MEDIA — evita gradientes corruptos |
| 5 | AerSimulator caching | **~50ms/eval** | MPS stochastic path | 🟡 BAJA — solo relevante en modo legacy |

### Optimizaciones marginales (no justifican complejidad)

| # | Optimización | Speedup Medido | Por qué no ayuda tanto | Veredicto |
|---|---|---|---|---|
| 6 | Parallel noise factors (ThreadPool) | **1.03× (N=6), 1.17× (N=10)** | Thread overhead (~20ms) compite con simulation time (~60-200ms per factor). Solo rinde a N≥14 donde per-factor > 500ms | 🟡 MARGINAL — implementado pero auto-disabled para N<14 |
| 7 | Observable batching (multi-PUB) | **~10ms** teórico | BackendEstimatorV2 ya hace abelian grouping. Solo se llama 1 vez/deployment | ⚪ NO IMPLEMENTADA — beneficio negligible |

### Métricas Consolidadas (benchmarks ejecutados 2026-06-10)

**benchmark_pea_performance.py:**
```
N=6:  0.13-0.18s total (threshold <1.0s) ✅ PASS
N=10: 0.51-0.61s total (threshold <3.0s) ✅ PASS
```

**benchmark_pea_parallel.py (measurement phase only):**
```
N=6:  seq=0.063s, par=0.061s → 1.03× (no gain, correctly off)
N=10: seq=0.661s, par=0.564s → 1.17× (modest gain, correctly off)
N≥14: projected ~1.9× based on per-factor scaling
```

**Test suite:** 94/94 noisy/PEA/ZNE tests pass sin regresión.

### Lección Principal

El **verdadero bottleneck** nunca fue lo que parecía:
- ❌ NO era transpilación (BackendEstimatorV2 no transpila en run())
- ❌ NO era paralelismo (threads no ayudan a <200ms/factor)
- ✅ ERA la construcción de 900 `depolarizing_error()` objects innecesarios

La optimización de **filtrar a pares relevantes** (Opt-2) produjo el 90% del gain total.
El paralelismo es útil como inversión de escalabilidad (N≥14) pero no cambia el juego a nuestros N actuales (6-10).



---

## Appendix: PEA N≥20 Scalability Limit (2026-06-10)

### Experiment PEA_SCALING — Valid Negative Result

Attempted PEA-ZNE at N=40/50 using MPS noisy simulation. Result: **PEA gain = 0%,
R² = 0.0** because native chain_1d circuits have insufficient noise for ZNE to act on.

| N | CZ gates | Routing | Noise impact | PEA gain | Verdict |
|---|---|---|---|---|---|
| 10 | 54 | heavy_hex SWAP | 43% total | +97% | ✅ Valid |
| 40 | 39 | None (native) | 0.3% relative | ~0% | ⚠️ No effect |
| 50 | 49 | None (native) | 0.4% relative | ~0% | ⚠️ No effect |

**Root cause**: SWAP routing (FakeTorino → heavy_hex) adds 5× more CZ gates that
create the noise PEA mitigates. Without routing, native chain circuits are too clean.

**Conclusion**: PEA N≥20 validation requires real QPU (where routing is physical).
Local MPS simulation is only valid for VQE convergence (noiseless) and scaling law
validation. ZNE/PEA simulation is limited to N≤10 with FakeTorino.

See: `documentation/analysis/23_noisy_simulation_scalability_limits.md`
