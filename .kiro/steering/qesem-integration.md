---
inclusion: fileMatch
fileMatchPattern: "**/qesem*,**/hardware/**,**/config.py,**/backends.py,scripts/recover_qesem*,scripts/convert_qesem*,scripts/estimate_qesem*,scripts/experiment_runners/hardware/**,tests/hardware/**,project_health/analysis/hardware/**"
---

# QESEM Integration — Architecture & Rules

## What is QESEM

QESEM (Quantum Error Suppression and Error Mitigation) is Qedma's Qiskit Function
that provides unbiased, characterization-based error mitigation. It replaces our
entire local mitigation stack (PEA noise learning, ZNE extrapolation, affine
correction, multi-layout averaging) when `mitigation.qesem_enabled=True`.

Reference: arXiv:2508.10997 — "Reliable high-accuracy error mitigation for
utility-scale quantum circuits"

## File Locations

| Purpose | File |
|---------|------|
| Core module | `src/qmbp_simulation/execution/hardware/qesem.py` |
| Routing logic | `src/qmbp_simulation/execution/hardware/backend.py` (QESEM alternate path) |
| Config fields | `src/qmbp_simulation/execution/hardware/config.py` (HardwareConfig + HardwareRunResult) |
| Flag definition | `src/qmbp_simulation/execution/backends.py` (MitigationOptions.qesem_enabled) |
| Budget estimator | `scripts/estimate_qesem_budget.py` |
| Tests (QESEM) | `tests/hardware/test_qesem_integration.py` |
| Tests (QET) | `tests/hardware/test_qet_integration.py` |
| QET validator | `project_health/analysis/hardware/validate_qet.py` |
| Deployment CLI | `scripts/experiment_runners/hardware/run_ibm_deployment.py` (--qesem flag) |
| Recovery script | `scripts/recover_qesem_job.py` |
| Result converter | `scripts/convert_qesem_to_hwresult.py` |

## Architecture

```
run_deployment(circuit, H, params, h, e_exact, gap)
│
├── if config.mitigation.qesem_enabled:
│     ├── preflight checks (same as local path)
│     ├── calibration snapshot (pre-execution record)
│     ├── resolve QET mode (noise_scale2precision or config.qesem_noise_scales)
│     ├── run_qesem_deployment(bound_circuit, H, x_ops, zz_ops, config)
│     │     ├── Validate QET scales (if QET mode: no negative scales, precision > 0)
│     │     ├── Load catalog → catalog.load("qedma/qesem")
│     │     ├── Build PUB:
│     │     │     ├── Standard: (circuit, observables)
│     │     │     └── QET: (circuit, observables, None, noise_scale2precision)
│     │     ├── Submit with mode-specific options
│     │     ├── Wait for result with client-side timeout + retry logic
│     │     ├── Parse results:
│     │     │     ├── Standard: pub_result.data.evs → energy + observables
│     │     │     └── QET (no scale=0.0): noise_scaling → WLS extrapolation
│     │     ├── Parse noise_scale_results for all observables
│     │     ├── Parse qesem_heuristic (exponential ZNE from 1.0+2.0)
│     │     └── Return QESEMResult (mitigated evs + stds + scale data + metadata)
│     ├── Precision quality gate (warn if std > 2×ε)
│     ├── classify_phase(x_values, zz_values)
│     ├── verdict (ΔE/gap < 5% + correct label)
│     ├── save_run() → persist HardwareRunResult
│     ├── QET post-execution validation (automatic, non-blocking)
│     └── return HardwareRunResult
│
└── else: (local PEA/GF-ZNE pipeline — current default)
```

## Key Design Decisions

1. **QESEM replaces the ENTIRE local mitigation stack** — it does NOT complement PEA.
   When enabled, skip: layout selection for ZNE, PEA noise learning, GF/PEA ZNE
   extrapolation, affine correction, multi-layout observable averaging.

2. **QESEM handles its own transpilation** — our VF2 layout selection is skipped.
   `layouts_used=[]` and `ces_values=[]` in the result for QESEM runs.

3. **Default is OFF** — `qesem_enabled=False`. Zero impact on existing PEA-ZNE workflow.

4. **Cannot run in fake_backend mode** — QESEM is a server-side function requiring
   a real QPU + Premium/Flex plan. Guard raises RuntimeError immediately.

5. **Same HardwareRunResult fields** — downstream validators, thesis figures, and
   comparison scripts work unchanged. Distinguish via `mitigation_strategy` field:
   - `"pea_local"` or `"ibm_zne_layout_avg"` → PEA/GF-ZNE path
   - `"qesem_unbiased"` → QESEM path

6. **Statistical uncertainty propagated** — `e_zne_std` field in HardwareRunResult
   captures QESEM's reported std. Essential for publication error bars.

## Rules When Modifying QESEM Code

- **Never skip the fake_backend guard** — QESEM physically cannot run locally.
- **Always use client-side timeout** on `job.result()` — QESEM jobs can hang.
  Formula: `timeout = 2 × max_execution_time + 900s`.
- **Guard noisy_results parsing** — the QESEM SDK metadata format may change.
  Always use `hasattr` / `isinstance` / try-except checks, never bare attribute access.
- **zne_gain requires noisy_data_available=True** — when noisy baseline is
  unavailable (sentinel zeros), zne_gain must be 0.0, not a fake computation.
- **Precision quality gate** — if `energy_std > 2×qesem_precision`, log a warning.
  This indicates QPU time cap was hit before QESEM converged.
- **Observables use logical qubit edges** — `[(i, i+1) for i in range(n-1)]`.
  This matches heavy_hex N=10 chain subgraph. If topology changes, update.
- **Validate QET inputs early** — check noise_scale2precision BEFORE calling
  `_load_qesem_function()`. Invalid scales waste time/credentials.
- **QET complementary pairs are correlated** — when reporting error bars from
  QET data, do NOT treat complementary pairs (s₁+s₂=2.0) as independent points.
- **WLS non-finite fallback** — if WLS extrapolation produces NaN/Inf, fall back
  to weighted mean of input data (never propagate non-finite results).
- **All diagnostic messages must follow dato→significado→causa→acción format** —
  every Finding/issue must explain: what was observed, what it means physically,
  why it might happen, and what action to take.

## Config Fields

```python
# In MitigationOptions (backends.py):
qesem_enabled: bool = False  # Routing flag

# In HardwareConfig (config.py):
qesem_precision: float = 0.01        # Target ε for ⟨O⟩
qesem_max_execution_time: int = 600  # QPU time cap per PUB (seconds)
qesem_instance: str | None = None    # IBM instance override (None = env)
qesem_noise_scales: dict[float, float] | None = None  # QET mode (see below)

# In QESEMResult (qesem.py) — new QET fields:
noise_scale_results: list[dict[float, tuple[float, float]]] | None  # Per-observable scale data
extrapolation_method: str  # "qesem_standard" | "qet_user_wls" | "qesem_heuristic"
qesem_heuristic_energy: float | None  # Exponential ZNE from scales 1.0+2.0
qesem_heuristic_std: float | None

# In HardwareRunResult (config.py):
qesem_used: bool                     # True if QESEM was the mitigation method
qesem_job_id: str                    # Job ID for provenance/recovery
qesem_total_qpu_time: float | None   # Seconds reported by QESEM
qesem_gate_fidelities: dict | None   # Per-gate fidelities from characterization
qesem_total_shots: int | None        # Total shots used
qesem_mitigation_shots: int | None   # Shots allocated to mitigation
qesem_noisy_evs: list[float] | None  # Pre-mitigation raw estimates (None if unavailable)
e_zne_std: float | None              # Statistical uncertainty on energy
```

## QET (Quasi-probabilistic Error Tuning)

QET is QESEM's explicit noise-scaling mode. Instead of requesting only the fully
mitigated result (scale=0.0), QET returns expectation values at user-specified
noise levels so you can perform your own extrapolation.

### QET Noise Scale Semantics

- `scale = 0.0` → fully mitigated QESEM result (quasi-probabilistic)
- `0 < scale < 1` → partially reduced noise (between ideal and physical)
- `scale = 1.0` → physical device noise (with Readout Error Mitigation only)
- `scale > 1.0` → amplified noise (more noise than physical device)

### Complementary Pairs (Free)

When you request a scale, QESEM automatically provides its complement around 1.0:
- Request 0.5 → get 1.5 free
- Request 0.7 → get 1.3 free
- Request 0.0 → get 2.0 free

**IMPORTANT**: Complementary pairs share the same circuit data. They are NOT
statistically independent measurements. For ZNE error bars, only count
explicitly-requested scales as independent data points.

### Usage

```python
# Via config (applies to all QESEM calls)
config = HardwareConfig(
    qesem_noise_scales={0.5: 0.02, 1.0: 0.01, 1.5: 0.03},
    ...
)

# Via function parameter (per-call override)
result = run_qesem_deployment(
    circuit, H, x_ops, zz_ops, config,
    noise_scale2precision={0.3: 0.02, 0.5: 0.02, 0.7: 0.02, 1.3: 0.03},
)
```

### WLS Extrapolation

When QET mode does not include scale=0.0, the pipeline uses Weighted Least
Squares (WLS) linear extrapolation to estimate the zero-noise value:

```python
from qmbp_simulation.execution.hardware.qesem import extrapolate_qet_wls

# noise_scale_data: {scale: (value, std), ...}
e_extrapolated, std_extrapolated = extrapolate_qet_wls(
    noise_scale_data, extrapolation_order=1
)
```

Weights are σ_i^{-2} (inverse variance), consistent with our PEA-ZNE WLS.

### Validation (Automatic)

After every QESEM execution, `validate_qet_result()` runs automatically
(non-blocking) and logs diagnostics to the StructuredLogger. Checks include:
1. Noise-scale data availability
2. Scale monotonicity (energy should degrade with noise)
3. WLS extrapolation quality vs QESEM standard result
4. Precision convergence (σ vs ε)
5. Complementary pair correlation detection
6. Observable consistency with expected phase
7. Gate fidelity assessment
8. Mitigation gain (negative = mitigation degraded the result)

Run manually:
```bash
.venv/bin/python project_health/analysis/hardware/validate_qet.py --all
```

### Reference

- GitHub: `Qedma/QET-tutorial` (qet_scales.ipynb, qesem_heuristic.ipynb)
- arXiv:2508.10997 — QESEM paper (noise tuning in Sec. 3)

## When to Use QESEM vs QET vs PEA-ZNE

| Scenario | Recommendation |
|----------|---------------|
| Thesis primary data (p=1 N=10, 18 CZ) | PEA-ZNE — validated, self-contained |
| Cross-validation of PEA results | QESEM standard — independent unbiased method |
| Deeper circuits (p≥2, >30 CZ) | QESEM or QET — PEA loses linearity |
| Publication benchmark comparison | Both — same h-points with both methods |
| QPU budget constrained | PEA-ZNE — you control shots exactly |
| Custom extrapolation research | QET with 5-7 scales + WLS — full control |
| Verify QESEM accuracy independently | QET scales + your own linear fit |
| Noise characterization study | QET with many scales (0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0) |

## Testing

Run QESEM + QET tests with:
```bash
# All QESEM/QET tests (42 tests, ~12s)
python -m pytest tests/hardware/test_qesem_integration.py tests/hardware/test_qet_integration.py -v

# QET-specific only (17 tests)
python -m pytest tests/hardware/test_qet_integration.py -v

# QESEM validator checks (24 tests)
python -m pytest tests/hardware/test_qesem_validator_checks.py -v

# Run QET post-execution validator on all recovered results
.venv/bin/python project_health/analysis/hardware/validate_qet.py --all
```

All tests use mocks (QESEM/QET require real QPU). Tests validate:
- QESEMResult dataclass construction (with new QET fields)
- QESEMResult → HardwareRunResult mapping
- Verdict logic with QESEM output
- fake_backend mode guard
- run_qesem_deployment interface via mock (standard and QET modes)
- WLS extrapolation accuracy (linear, quadratic, edge cases)
- Noise-scaling metadata parsing from real QESEM results
- QESEM heuristic extraction
- Input validation (negative scales, zero precision, empty dict)
- Post-execution validator checks (C20-C24)

## Dependencies

- `qiskit-ibm-catalog >= 0.8.0` (for `QiskitFunctionsCatalog`)
- IBM Quantum Premium/Flex/On-Prem plan access
- Environment: `IBM_KEY` + `IBM_INSTANCE_CRN`
