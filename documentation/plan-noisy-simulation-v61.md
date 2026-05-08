# Plan: Noisy Simulation & Multi-Point Deployment (V6.1, N=10)

## Objective

Exercise the full V6.1 error mitigation stack in simulation before real QPU deployment. This validates that:
1. Inhomogeneous ZNE actually reduces errors (not just implemented but effective)
2. The multi-point deployment workflow mirrors what we'd do on hardware
3. We can quantify the noise gap: noiseless → noisy → mitigated

---

## Architecture

### New Mode: `mode="noisy_simulation"`

Add a third mode to `HardwareDeployerV61` that uses `FakeTorino` + `BackendEstimatorV2`:
- Uses real IBM Torino calibration data (CZ errors ~0.5%, readout ~1-2%)
- Exercises the FULL hardware code path: layout selection, transpilation, ZNE, observable grouping
- No IBM credentials needed — runs locally with `qiskit-aer`

### Routing Logic in `deploy_adapt_vqe`

Current branching:
```python
if self._mode == "hardware":
    # Full ZNE path
else:
    # StatevectorEstimator (noiseless)
```

New branching:
```python
if self._mode in ("hardware", "noisy_simulation"):
    # Full ZNE path — uses self._backend (real or FakeTorino)
else:
    # StatevectorEstimator (noiseless)
```

The key insight: `noisy_simulation` is treated identically to `hardware` in the deploy method. The only difference is in `__init__` where the backend is constructed locally instead of via IBM Runtime service.

### Estimator Construction

For `noisy_simulation` mode, we **cannot** use `qiskit_ibm_runtime.EstimatorV2` (requires a real session). Instead, use `qiskit.primitives.BackendEstimatorV2`:

```python
from qiskit.primitives import BackendEstimatorV2
estimator = BackendEstimatorV2(backend=self._backend)
```

This means `_run_inhomogeneous_zne` needs a small change: instead of always constructing `qiskit_ibm_runtime.EstimatorV2`, it should use `BackendEstimatorV2` for noisy_simulation mode. The interface is compatible (both accept PUBs and return the same result structure).

### "Noisy Without Mitigation" Mode

The 3-mode comparison requires a "raw noisy" baseline (no ZNE). This is achieved by:
- Using `n_layouts=1` in the deployer constructor
- With 1 layout, `_run_inhomogeneous_zne` cannot extrapolate (needs ≥2 points) and falls back to returning the single-layout raw values
- No additional flag needed — the existing code handles this case

Script-level control:
```python
# Noisy, no mitigation (single layout, no extrapolation)
deployer_raw = HardwareDeployerV61(mode="noisy_simulation", n_layouts=1)

# Noisy, with ZNE mitigation (3 layouts, linear extrapolation)
deployer_mitigated = HardwareDeployerV61(mode="noisy_simulation", n_layouts=3)
```

---

## Code Changes

### File: `src/poc/v6/hardware_deployer_v61.py`

**1. `__init__` — add noisy_simulation branch (~15 lines)**

```python
elif mode == "noisy_simulation":
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    self._backend = FakeTorino()
    self._layout_selector = LayoutSelector(self._backend, seed=seed)
    logger.info("HardwareDeployerV61 initialized in noisy_simulation mode (FakeTorino).")
```

Add `seed` parameter to `__init__` (default=42) for reproducibility across:
- LayoutSelector BFS search
- Aer shot sampling (via backend seed)

**2. `deploy_adapt_vqe` — update mode check (~2 lines)**

```python
if self._mode in ("hardware", "noisy_simulation"):
    # Full ZNE path
```

**3. `_run_inhomogeneous_zne` — estimator construction (~10 lines)**

Replace:
```python
from qiskit_ibm_runtime import EstimatorV2
estimator = EstimatorV2(backend=self._backend)
```

With:
```python
if self._mode == "noisy_simulation":
    from qiskit.primitives import BackendEstimatorV2
    estimator = BackendEstimatorV2(backend=self._backend)
else:
    from qiskit_ibm_runtime import EstimatorV2
    estimator = EstimatorV2(backend=self._backend)
```

**4. Remove EstimatorV2 options for noisy_simulation**

`BackendEstimatorV2` does NOT support the same options API as `qiskit_ibm_runtime.EstimatorV2` (no `dynamical_decoupling`, `twirling`, `resilience` sub-objects). For noisy_simulation:
- DD, twirling, TREX are **NOT applied** (isolates ZNE contribution)
- This is intentional: we want to measure ZNE effectiveness in isolation first
- On real hardware, all 5 mitigation layers would be active simultaneously

The options block should be guarded:
```python
if self._mode == "hardware":
    # Apply full options (DD, twirling, TREX)
    options = build_estimator_options(shots=total_shots)
    for key, value in options.items():
        ...
# noisy_simulation: no options applied (raw noise + ZNE only)
```

### File: `scripts/run_v61_noisy.py` (new, ~250 lines)

Multi-point deployment sweep with three deployment modes.

---

## Experiment Design

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| N | 10 | Target system size |
| Topology | chain_1d | Validated optimal |
| MPNN | h=128, L=3, 6000ep, patience=500 | N=10 optimal |
| Seed (MPNN) | 43 | Best convergence |
| Seed (Aer/Layout) | 42 | Reproducibility |
| VQE | 5 restarts, 1000 iter | Standard |
| Shots | **16384** | N=10 recommended (σ ≈ 7.8e-3) |
| ZNE layouts | 3 (mitigated) / 1 (raw) | Standard for inhomogeneous ZNE |
| DD/Twirling/TREX | **OFF** | Isolate ZNE contribution |
| NN Extrapolator | **OFF** | Needs ≥5 layouts (we use 3) |

### Deployment Points

h ∈ {1.0, 1.25, 1.4, 1.5, 1.7, 2.0} — spans critical region to deep paramagnetic.

### Three Modes Per Point

| Mode | Deployer Config | What It Measures |
|------|----------------|------------------|
| Noiseless | `mode="simulation"` | Pipeline ceiling (no noise) |
| Noisy (raw) | `mode="noisy_simulation", n_layouts=1` | Hardware-like degradation |
| Mitigated (ZNE) | `mode="noisy_simulation", n_layouts=3` | ZNE effectiveness |

Total: 6 h-values × 3 modes = 18 deployments.

### Seed Control

| Component | Seed | Purpose |
|-----------|------|---------|
| MPNN training | 43 | Reproducible model |
| VQE optimizer | 43 | Reproducible θ_opt |
| LayoutSelector | 42 | Reproducible qubit mapping |
| Aer simulator | Not directly seedable per-run; FakeTorino uses internal RNG | Shot noise varies between runs |

Note: Aer shot sampling is not perfectly reproducible across runs (internal threading). For thesis, report single-run values (not mean±std) since the noise is the variable under study.

---

## Expected Results

| h_test | Noiseless ΔE/gap | Noisy (raw) ΔE/gap | Mitigated (ZNE) ΔE/gap | ZNE gain |
|--------|-----------------|--------------------|-----------------------|----------|
| 1.0 | ~10% | ~20-30% | ~15-20% | Moderate |
| 1.25 | ~5% | ~12-18% | ~8-12% | Moderate |
| 1.4 | ~4.4% | ~10-15% | ~6-9% | Significant |
| 1.5 | ~2.7% | ~8-12% | ~4-7% | Significant |
| 1.7 | ~1.5% | ~6-10% | ~3-5% | Significant |
| 2.0 | ~1% | ~5-8% | ~2-4% | Significant |

### Success Criteria

1. **Mitigated ΔE/gap < Noisy ΔE/gap** for at least 4/6 h-values (ZNE helps)
2. **ZNE R² > 0.8** for at least 3/6 h-values (extrapolation is meaningful)
3. **Phase classification correct** for h ≥ 1.5 even with noise (noise resilience)
4. **Full code path executes without errors** (integration validation)
5. **Mitigated results closer to noiseless** than raw noisy (mitigation direction correct)

---

## Runtime Estimate

| Mode | Deployments | Layouts/each | Shots | Est. time/deployment | Total |
|------|-------------|-------------|-------|---------------------|-------|
| Noiseless | 6 | N/A | N/A | ~1s | ~6s |
| Noisy (raw) | 6 | 1 | 16384 | ~10-20s | ~1-2 min |
| Mitigated (ZNE) | 6 | 3 | 16384 | ~30-60s | ~3-6 min |

Plus MPNN training: ~70s (one-time).

**Total estimated: ~6-10 minutes.**

---

## Scope Exclusions (Explicit)

- **DD, Pauli Twirling, TREX**: NOT exercised in this experiment. These are `EstimatorV2` (IBM Runtime) options that `BackendEstimatorV2` doesn't support. They will be tested on real hardware only. This experiment isolates ZNE contribution.
- **NNExtrapolator**: NOT activated (requires ≥5 layouts, we use 3). Could be tested in a follow-up with n_layouts=5.
- **Multi-seed noise averaging**: NOT done. Single Aer run per deployment. Noise variance is the variable under study, not a nuisance to average out.

---

## Implementation Steps

### Step 1: Modify `hardware_deployer_v61.py` (~30 lines total)

1. Add `seed` parameter to `__init__`
2. Add `noisy_simulation` branch in `__init__`
3. Update `deploy_adapt_vqe` mode check: `if self._mode in ("hardware", "noisy_simulation")`
4. Update `_run_inhomogeneous_zne` estimator construction (BackendEstimatorV2 for noisy_sim)
5. Guard options block (skip DD/twirling/TREX for noisy_simulation)

### Step 2: Create `scripts/run_v61_noisy.py` (~250 lines)

1. Train MPNN once (N=10 optimal config)
2. For each h_test in sweep:
   a. Deploy noiseless (existing `mode="simulation"`)
   b. Deploy noisy raw (`mode="noisy_simulation"`, n_layouts=1)
   c. Deploy mitigated (`mode="noisy_simulation"`, n_layouts=3)
3. Report comparison table
4. Compute ZNE gain metrics
5. Save JSON + append binnacle

### Step 3: Run and document (~10 min)

### Step 4: Update binnacle-N10.md with findings

---

## Thesis Value

This experiment produces:
- **Section 4.5 data**: "Simulated Hardware Deployment" — the bridge between noiseless validation and real QPU
- **Figure 4.x**: ΔE/gap vs h curve with three lines (noiseless / noisy / mitigated)
- **Quantitative ZNE validation**: R² values and gain percentages
- **Confidence for hardware**: if ZNE helps in simulation, it will help on real hardware (noise model is calibration-derived)
