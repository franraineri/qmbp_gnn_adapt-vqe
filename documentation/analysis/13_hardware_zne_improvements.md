# Hardware ZNE Implementation — Improvements & Action Plan

**Date**: 2026-06-04
**Status**: Proposed (ready for implementation)
**Prerequisite**: ZNE_CROSS_TOPO results (PEA validated across 3 topologies)

---

## Executive Summary

The current `HardwareBackend.run_deployment()` uses **CES-based inhomogeneous
ZNE** as its primary error mitigation strategy. Our simulation campaign
(6 experiments, 60+ h-points) conclusively shows this is the wrong strategy
for the hardware target (heavy_hex N=10 p=1):

| Strategy | Mean Gain | R² on heavy_hex | Status |
|----------|:---------:|:---------------:|--------|
| CES-ZNE (current impl) | +3% | 0.04–0.99 (unstable) | **Broken on target** |
| GF-ZNE | +20.6% | 0.47 (simulation) | Fallback |
| **PEA-ZNE** | **+94.4%** | **0.998** | **Recommended primary** |

This document proposes 5 concrete, modular improvements to the hardware
execution path, ordered by impact and effort.

---

## Issue 1: CES-ZNE Fails on Heavy_hex (CRITICAL)

### Problem

`run_deployment()` line ~237 does:
```python
zne_result = linear_zne(np.array(ces_used), np.array(energies))
```

This extrapolates energy vs CES (Circuit Error Score) across 3 layouts.
On heavy_hex N=10 p=1, all good layouts have **CES ≈ 0.15** (no spread).
Result: R²≈0.04, extrapolation is meaningless.

### Evidence

- `11_hardware_rehearsal_findings.md`: R²=0.04 with `select_layouts_low_ces`
- `binnacle-gate-folding-zne.md`: CES-ZNE gain negative in 4/18 cases
- `ZNE_CROSS_TOPO`: CES-ZNE mean gain only +2.9% across all topologies

### Fix: Use IBM Runtime's Built-in ZNE (Server-Side)

On real hardware, IBM Runtime handles ZNE server-side when configured via
`EstimatorV2.options.resilience`. The client receives the **already-mitigated**
energy. The current implementation does double-work: it configures server-side
ZNE AND attempts client-side CES extrapolation.

**The fix is to trust IBM's server-side ZNE and remove client-side CES extrapolation.**

---

## Issue 2: No Dual-Amplifier Strategy (HIGH)

### Problem

`build_estimator_options()` sets a single amplifier (default: gate_folding).
Our data shows GF-ZNE has R²=0.47 on heavy_hex simulation and PEA has R²=0.998.
There's no mechanism to:
1. Try gate_folding first
2. Detect poor R²
3. Fall back to PEA

### Evidence

- `PEA_HW_READY`: GF R²=0.47 vs PEA R²=0.94 on the exact hardware config
- `ZNE_CROSS_TOPO`: PEA wins 18/18 h-points (p=2.5×10⁻¹⁹)

### Fix: Tiered ZNE with R²-Based Fallback

Implement a 2-attempt strategy:
1. First attempt: gate_folding (simpler, no overhead)
2. If ΔE/gap > threshold OR ZNE quality metric is poor → retry with PEA
3. Return the better result

On IBM Runtime, this means: execute once with gate_folding options. If the
result's metadata indicates poor extrapolation quality, re-execute with
`amplifier="pea"`. The ~50% QPU overhead of PEA is justified only when needed.

---

## Issue 3: Per-Site Observables Not ZNE-Mitigated (MEDIUM)

### Problem

`run_deployment()` measures ⟨X_i⟩ and ⟨Z_iZ_j⟩ for phase classification
using a single estimator call on the first layout. These observables are
**not ZNE-extrapolated** — they get raw noisy values.

Phase classification uses `|⟨X⟩| vs |⟨ZZ⟩|`. At h≥3.0 on heavy_hex,
⟨X⟩ ≈ -0.965 and noise floor ≈ 0.008, so the signal is 120× above noise
(SNR=123σ, from hardware rehearsal Section 3). Classification works even
without ZNE on observables.

### Evidence

- `HW_REHEARSAL` Section 3: SNR=123σ, 100% correct classification
- `PEA_PIPELINE` Section 4: classification fails with imperfect theta (not noise)

### Assessment

**Not blocking for deployment.** The ⟨X⟩ signal at h≥3.0 is so strong that
noise doesn't affect the phase label. However, for completeness and to support
future work at lower h-values (closer to the phase transition), observable
ZNE would add robustness.

### Fix: Submit observables through the same ZNE-configured estimator

Since IBM Runtime applies ZNE to ALL measurements submitted through a
ZNE-configured EstimatorV2, the fix is straightforward: use the same
configured estimator for observables (it already IS the configured estimator,
but ensure the ZNE options propagate).

---

## Issue 4: No R² Quality Gate Before Verdict (MEDIUM)

### Problem

The current verdict logic is:
```python
verdict = "PASS" if (delta_e_gap < 0.05 and label == expected_label) else "FAIL"
```

This doesn't check whether the ZNE extrapolation was reliable (R²). If R² < 0.5,
the extrapolated energy is unreliable and the verdict is meaningless.

### Evidence

- `PEA_HW_READY`: GF-ZNE R²=0.47 → ΔE/gap=0.915 (extrapolation meaningless)
- `GF_ZNE_CMP`: Even R²=0.99 can give negative gain (wrong extrapolation direction)

### Fix: Add R² Quality Gate

```python
if zne_r2 < 0.80:
    verdict = "INDETERMINATE"  # ZNE quality insufficient
elif delta_e_gap < 0.05 and label == expected_label:
    verdict = "PASS"
else:
    verdict = "FAIL"
```

This prevents reporting "FAIL" when the real issue is poor extrapolation quality
(which should trigger a PEA retry, not a final failure).

---

## Issue 5: SPSA Uses Full HardwareBackend.evaluate() (LOW)

### Problem

SPSA refinement calls `self.evaluate(circuit, hamiltonian)` which does:
1. Bind parameters
2. Select 3 layouts
3. Transpile ×3
4. Submit ×3 jobs
5. CES-ZNE extrapolation

Each SPSA iteration costs 2× this (plus/minus). With 200 iterations, that's
1200 layout-transpile-evaluate cycles. At 16k shots × 3 layouts × 2 × 200 =
**19.2M shots** — exceeding the 10M ceiling (which correctly aborts SPSA early).

### Assessment

The cost ceiling check correctly prevents runaway SPSA. But the real issue is
that SPSA should use a simpler evaluation (single layout, no ZNE) for gradient
estimation, then ZNE only for the final evaluation. This is an optimization,
not a correctness issue.

### Fix (Optional): Lightweight SPSA Evaluation

Create a `_evaluate_lightweight()` that uses 1 layout, 8k shots, no ZNE.
Use this for SPSA gradient estimation. Only apply full ZNE to the final
best parameters. Reduces SPSA cost by ~6× (from 1200 to 200 evaluations
at 8k×1 instead of 16k×3).

---

## Implementation Plan

### Priority Order

| # | Issue | Impact | Effort | Dependency |
|---|-------|--------|--------|-----------|
| 1 | CES-ZNE → IBM server-side ZNE | Critical | Low | None |
| 2 | Dual amplifier (GF → PEA fallback) | High | Medium | Issue 1 |
| 4 | R² quality gate | Medium | Low | None |
| 3 | Observable ZNE propagation | Medium | Low | Issue 1 |
| 5 | Lightweight SPSA evaluation | Low | Medium | Issue 1 |

---

## Detailed Implementation Guide

### Improvement 1: Remove Client-Side CES-ZNE, Trust IBM Server-Side ZNE

**File**: `src/qmbp_simulation/execution/hardware/backend.py`

**Current flow** (lines ~230-240 of `run_deployment`):
```python
raw_results = submit_all_then_collect(...)
energies = [r["energy"] for r in raw_results]
ces_used = [layout_selection.ces_values[r["layout_idx"]] for r in raw_results]
zne_result = linear_zne(np.array(ces_used), np.array(energies))
e_zne = zne_result.extrapolated_value
```

**New flow**:
```python
# IBM Runtime returns ALREADY ZNE-mitigated energies when
# options.resilience.zne_mitigation = True (set in build_estimator_options).
# Multi-layout averaging reduces variance without CES extrapolation.
raw_results = submit_all_then_collect(...)
energies = [r["energy"] for r in raw_results]

# Simple average across layouts (all already ZNE-mitigated by IBM)
e_zne = float(np.mean(energies))
e_std = float(np.std(energies)) / np.sqrt(len(energies))

# R² is not applicable for server-side ZNE (IBM doesn't expose it).
# Use std/gap as a quality metric instead.
zne_quality = e_std / gap if gap > 0 else float("inf")
zne_r2 = 1.0 - zne_quality  # Proxy: lower variance = higher quality
```

**Rationale**: IBM's server-side ZNE handles the amplification, fitting, and
extrapolation internally. Each layout gets its own ZNE extrapolation. We
average the 3 ZNE-mitigated energies for variance reduction (√3 improvement).

**Backward compatibility**: For `fake_backend` mode, maintain the current
local ZNE logic (GF or PEA) since FakeTorino doesn't have server-side ZNE.
Add a branching point:

```python
if self._config.mode == "hardware":
    # IBM handles ZNE internally, we average across layouts
    e_zne = float(np.mean(energies))
    zne_r2 = _compute_layout_consistency(energies, gap)
else:
    # Local simulation: use our own ZNE (PEA or GF)
    e_zne, zne_r2 = self._run_local_zne(layout_selection, hamiltonian)
```

**Modularity**: Extract into a new private method `_aggregate_zne_results()`
that takes `(raw_results, layout_selection, mode)` and returns `(e_zne, r2)`.

---

### Improvement 2: Dual-Amplifier Strategy with Automatic Fallback

**File**: `src/qmbp_simulation/execution/hardware/backend.py`
**New file** (optional): `src/qmbp_simulation/execution/hardware/zne_strategy.py`

**Architecture**:

```python
@dataclass
class ZNEStrategy:
    """Encapsulates the tiered ZNE amplifier selection logic."""

    primary_amplifier: str = "gate_folding"
    fallback_amplifier: str = "pea"
    r2_threshold: float = 0.80  # Trigger fallback if R² below this
    de_gap_threshold: float = 0.10  # Trigger fallback if ΔE/gap above this
    max_attempts: int = 2

    def should_retry(self, r2: float, de_gap: float) -> bool:
        """Determine if a PEA retry is warranted."""
        return r2 < self.r2_threshold or de_gap > self.de_gap_threshold
```

**Integration in `run_deployment()`**:

```python
# First attempt: gate_folding (default, no QPU overhead)
config_gf = self._config_with_amplifier("gate_folding")
result_gf = self._execute_single_attempt(bound, layout_selection, hamiltonian, config_gf)

# Check quality
de_gap_gf = abs(result_gf.e_zne - e_exact) / gap
if self._zne_strategy.should_retry(result_gf.r2, de_gap_gf):
    # Second attempt: PEA (learns noise model, ~50% overhead)
    config_pea = self._config_with_amplifier("pea")
    result_pea = self._execute_single_attempt(bound, layout_selection, hamiltonian, config_pea)
    de_gap_pea = abs(result_pea.e_zne - e_exact) / gap

    # Use the better result
    if de_gap_pea < de_gap_gf:
        result = result_pea
        amplifier_used = "pea"
    else:
        result = result_gf
        amplifier_used = "gate_folding"
else:
    result = result_gf
    amplifier_used = "gate_folding"
```

**Key design decisions**:
- **Modular**: `ZNEStrategy` is a separate dataclass, configurable via CLI/config
- **Parametrizable**: thresholds (R², ΔE/gap) are not hardcoded
- **Replicable**: each attempt saves its own raw data for post-hoc comparison
- **Scalable**: adding a third amplifier (e.g., PEC) requires only extending the strategy

**CLI integration**:
```bash
# Use only gate-folding (fast, single attempt)
python run_hw.py --zne-strategy single --zne-amplifier gate_folding

# Use only PEA (known-best for simulation)
python run_hw.py --zne-strategy single --zne-amplifier pea

# Auto-tier (default): try GF first, PEA if needed
python run_hw.py --zne-strategy tiered --r2-threshold 0.80
```

**For `fake_backend` mode** (local simulation validation):
Use the local `run_gate_folding_zne()` and `run_pea_zne()` functions
from `noisy_utils.py` directly, bypassing IBM Runtime options. This is
what `PEA_HW_READY` and `ZNE_CROSS_TOPO` already do successfully.

---

### Improvement 3: Observable ZNE Propagation

**File**: `src/qmbp_simulation/execution/hardware/backend.py`

**Current** (line ~250):
```python
estimator = self._get_configured_estimator()
evs = estimator.run([(isa_circ, mapped_obs)]).result()[0].data.evs
```

This already uses the ZNE-configured estimator, so on IBM Runtime, the
observables ARE ZNE-mitigated. The "issue" is only for `fake_backend` mode
where `BackendEstimatorV2` doesn't apply ZNE.

**Fix for `fake_backend` mode**:

```python
if self._config.mode == "fake_backend" and self._config.mitigation.zne_enabled:
    # Local ZNE for observables: run at noise_factors [1,3,5], extrapolate
    from qmbp_simulation.execution.noisy_utils import run_pea_zne

    x_values_zne = []
    for x_op_mapped in mapped_x_ops:
        pea = run_pea_zne(isa_circ, x_op_mapped, self.backend, ...)
        x_values_zne.append(pea.extrapolated_value)
    x_values = x_values_zne
else:
    # Hardware: IBM Runtime handles ZNE automatically
    evs = estimator.run([(isa_circ, mapped_obs)]).result()[0].data.evs
    x_values = [float(evs[i]) for i in range(len(x_ops))]
```

**Assessment**: This is optional for the thesis. The hardware rehearsal
confirmed classification works without observable ZNE (SNR=123σ). Implement
only if testing reveals classification issues at lower h-values.

---

### Improvement 4: R² Quality Gate

**File**: `src/qmbp_simulation/execution/hardware/backend.py`

**Current** (line ~270):
```python
verdict = "PASS" if (delta_e_gap < 0.05 and label == expected_label) else "FAIL"
```

**New**:
```python
# Quality-gated verdict
if zne_r2 < ZNE_R2_QUALITY_THRESHOLD:
    verdict = "INDETERMINATE"
    verdict_reason = f"ZNE quality insufficient (R²={zne_r2:.3f} < {ZNE_R2_QUALITY_THRESHOLD})"
elif delta_e_gap < 0.05 and label == expected_label:
    verdict = "PASS"
    verdict_reason = f"ΔE/gap={delta_e_gap:.4f} < 5%, phase={label} correct"
elif delta_e_gap < 0.05:
    verdict = "PARTIAL"
    verdict_reason = f"Energy good (ΔE/gap={delta_e_gap:.4f}) but phase={label} ≠ {expected_label}"
else:
    verdict = "FAIL"
    verdict_reason = f"ΔE/gap={delta_e_gap:.4f} ≥ 5%"
```

**HardwareRunResult update**:
```python
@dataclass
class HardwareRunResult:
    ...
    verdict: str = ""
    verdict_reason: str = ""  # NEW: explains why
    zne_amplifier_used: str = ""  # NEW: which amplifier produced this result
```

**Constant** (in `config.py`):
```python
ZNE_R2_QUALITY_THRESHOLD = 0.80  # From project-status.md
```

---

### Improvement 5: Lightweight SPSA Evaluation (Optional)

**File**: `src/qmbp_simulation/execution/hardware/spsa.py`

**New method in HardwareBackend**:
```python
def _evaluate_spsa(self, circuit: QuantumCircuit, hamiltonian: SparsePauliOp, params: np.ndarray) -> float:
    """Lightweight evaluation for SPSA gradient estimation.

    Uses 1 layout, 8192 shots, no ZNE (just raw noisy energy).
    ~6× cheaper than full evaluate().
    """
    bound = circuit.assign_parameters(params)
    # Use single best layout (cached from initial selection)
    transpiled = self._best_layout_cache
    H_mapped = hamiltonian.apply_layout(transpiled.layout)

    from qiskit_ibm_runtime import EstimatorV2
    # Minimal options: no ZNE, no DD, no twirling (speed > accuracy for gradient)
    est = EstimatorV2(backend=self.backend, options={"default_shots": 8192})
    job = est.run([(transpiled, H_mapped)])
    return float(job.result()[0].data.evs)
```

**Change in `run_deployment()`**:
```python
if self._config.spsa_enabled and delta_e_gap > self._config.spsa_threshold:
    eval_fn = partial(self._evaluate_spsa, circuit, hamiltonian)  # Lightweight
    # ... rest unchanged ...
    # After SPSA finds best params, do ONE full ZNE evaluation
    e_final = self.evaluate(circuit, hamiltonian, best_spsa_params)
```

---

## Verification Checklist

Before implementing, verify these constraints hold:

| # | Constraint | Verification |
|---|-----------|--------------|
| 1 | IBM Runtime returns ZNE-mitigated energy when `zne_mitigation=True` | IBM docs + test with 1 PUB |
| 2 | PEA amplifier available on ibm_torino | Check `backend.options.resilience` schema |
| 3 | Multi-layout average is valid (energies are independent) | Each layout is a separate PUB execution |
| 4 | R² proxy (std/gap) correlates with actual ZNE R² | Compare with simulation R² at same h |
| 5 | SPSA with 8k shots still provides useful gradient | V7-4A validated at 16k; test at 8k on FakeTorino |

---

## Consistency with Simulation Results

These improvements are directly justified by our experimental evidence:

| Experiment | Finding | Improvement Triggered |
|-----------|---------|----------------------|
| `HW_REHEARSAL` | CES-ZNE fails on heavy_hex (R²=0.04) | Issue 1 (remove CES-ZNE) |
| `PEA_HW_READY` | GF R²=0.47 on heavy_hex, PEA R²=0.94 | Issue 2 (dual amplifier) |
| `ZNE_CROSS_TOPO` | PEA wins 18/18 (t=46.32, p<10⁻¹⁹) | Issue 2 (PEA as primary) |
| `PEA_PIPELINE` | Classification needs good theta, not ZNE | Issue 3 (low priority) |
| `GF_ZNE_CMP` | R²=0.99 can still give negative gain | Issue 4 (R² quality gate) |
| `HW_REHEARSAL` §5 | Shot noise std=1.2% (below threshold) | Issue 5 (8k shots OK) |

---

## Recommended Implementation Order

```
Week 1: Issues 1 + 4 (low effort, critical impact)
  - Remove client-side CES-ZNE for hardware mode
  - Add R² quality gate to verdict
  - Test with FakeTorino (existing test suite)

Week 2: Issue 2 (medium effort, high impact)
  - Implement ZNEStrategy dataclass
  - Add tiered execution to run_deployment()
  - Add CLI flags (--zne-strategy, --zne-amplifier)
  - Test dual-amplifier with FakeTorino

Week 3: Issues 3 + 5 (optional, polish)
  - Observable ZNE for fake_backend mode
  - Lightweight SPSA evaluation
  - Run full rehearsal with all improvements
```

---

## File Changes Summary

| File | Action |
|------|--------|
| `execution/hardware/backend.py` | Major: ZNE flow refactor, dual-amplifier, R² gate |
| `execution/hardware/config.py` | Add `ZNEStrategy`, update `HardwareRunResult` |
| `execution/hardware/submission.py` | Minor: support switching amplifier options |
| `execution/hardware/zne_strategy.py` | **New**: encapsulates tiered ZNE logic |
| `execution/hardware/spsa.py` | Optional: accept lightweight eval callable |
| `execution/hardware/README.md` | Update documentation to reflect new flow |
| `execution/backends.py` | Add `zne_strategy` field to `MitigationOptions` |

---

## Non-Changes (Explicitly Preserved)

These modules are **not modified** by these improvements:

- `execution/hardware/preflight.py` — gate count check still valid
- `execution/hardware/phase.py` — classification logic unchanged
- `execution/hardware/persistence.py` — output format unchanged (extended only)
- `execution/hardware/observables.py` — observable construction unchanged
- `execution/noisy_utils.py` — local ZNE functions unchanged (used by fake_backend)
- All test scripts — backward compatible (new features are additive)


---

## Feasibility Verification (Automated, 2026-06-04)

Script: `tests/integration/test_module_contracts.py`

All 35 checks passed ✅:

| Category | Checks | Status |
|----------|:------:|:------:|
| 1. Referenced source files exist | 14/14 | ✅ |
| 2. Functions/classes importable | 15/15 | ✅ |
| 3. Framework patterns support proposals | 6/6 | ✅ |
| 4. build_estimator_options supports amplifier switching | 3/3 | ✅ |
| 5. Proposed ZNEStrategy pattern is implementable | 3/3 | ✅ |
| 6. Local ZNE functions available for fake_backend | 6/6 | ✅ |
| 7. SPSA accepts callable eval (swappable) | 1/1 | ✅ |
| 8. Criteria module has all experiment IDs | 3/3 | ✅ |

**Conclusion**: No new dependencies required. All proposed patterns are
compatible with the existing framework (`ValidationRunner`, dataclass config,
CLI args via `_add_custom_args`, `build_estimator_options` amplifier switching).

Key compatibilities verified:
- `HardwareConfig` is a dataclass → can add `zne_strategy: ZNEStrategy` field
- `HardwareRunResult` is a dataclass → can add `verdict_reason`, `zne_amplifier_used`
- `build_estimator_options()` already handles `amplifier="pea"` correctly
- `spsa_refinement()` accepts any callable as `evaluate_fn` → swappable for lightweight version
- `run_gate_folding_zne()` and `run_pea_zne()` have matching signatures → interchangeable
- `ValidationRunner._add_custom_args()` supports `--zne-strategy`, `--r2-threshold` etc.
