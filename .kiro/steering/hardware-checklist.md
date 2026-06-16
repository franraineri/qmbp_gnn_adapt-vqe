---
inclusion: manual
---

# Hardware Run Checklist (invoke with #hardware-checklist)

## Pre-flight

- [ ] `export IBM_KEY=<token>`
- [ ] `export IBM_INSTANCE_CRN=<crn>`
- [ ] Verify connection:
  ```python
  from qiskit_ibm_runtime import QiskitRuntimeService
  service = QiskitRuntimeService(channel="ibm_quantum_platform", token=..., instance=...)
  backend = service.backend("ibm_kingston")
  print(backend.status())
  ```
- [ ] Check Kingston queue depth (< 50 jobs ideal, execute UTC 2-6 AM)
- [ ] Confirm calibration: `compute_mean_2q_error(backend)` < 1%
- [ ] Run smoke test: `python tests/smoke_test.py`
- [ ] Run rehearsal V2 (green light):
  ```bash
  python scripts/experiment_runners/run_hardware_rehearsal_v2.py --section 1 2 3
  ```
  All 3 sections must PASS before proceeding.

## Execution

Use the tiered deployment script:

```bash
# Dry run first (no QPU usage)
python scripts/experiment_runners/hardware/run_ibm_torino_deployment.py --dry-run

# Tier 0: Smoke test (h=4.0, ~5 min QPU)
python scripts/experiment_runners/hardware/run_ibm_torino_deployment.py --tier 0

# Tier 1: Core validation (4 h-points, ~25 min QPU)
python scripts/experiment_runners/hardware/run_ibm_torino_deployment.py --tier 1

# Full auto (Tier 0 → 1 → 2 → 3, auto-advancing on success)
python scripts/experiment_runners/hardware/run_ibm_torino_deployment.py
```

### Tier Details

| Tier | h-values | Seeds | Purpose | Success |
|:----:|----------|:-----:|---------|---------|
| 0 | 4.0 | 42 | Smoke test (infra validation) | ΔE/gap < 10% |
| 1 | 4.0, 3.25, 3.0, 2.5 | 42 | Core thesis data (Table 5.23) | ≥3/4 PASS |
| 2 | Tier 1 × 3 seeds | 42,43,44 | Statistical robustness | ≥75% pass rate |
| 3 | 3.25 (tfim_longitudinal) | 42 | Model extensibility | PASS |

## Configuration (validated, do not change without reason)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Topology | heavy_hex | IBM Torino native |
| N | 10 | Validated scaling point |
| p | 1 | Within ZNE perturbative regime (18 CX) |
| Shots | 16,384 | σ=7.8e-3, below ⟨X⟩ signal |
| Layouts | 3 | Validated: +3% marginal for n=5 |
| ZNE amplifier | PEA (primary) | +94.4% gain, R²=0.998 |
| h_test | ≥ 3.0 | Valid regime for heavy_hex p=1 |
| SPSA | enabled if ΔE/gap > 5% | a=0.1, c=0.05, A=10, 200 iter |

## Shot Budget

| Scenario | Shots | Circuits/h | Total/h | Est. QPU time/h |
|----------|:-----:|:----------:|:-------:|:---------------:|
| PEA (3 layouts × 3 noise factors) | 16,384 | 9 | 147K | ~3 min |
| + observables (2 groups) | 16,384 | 2 | 33K | ~30s |
| + SPSA worst case (30% × 200 iter) | 16,384 | 120 | 1.97M | ~15 min |

Total for Tier 1 (4 h-points, no SPSA): ~15 min QPU time.

## Success Criteria

- [ ] ΔE/gap < 5% at h ≥ 3.0 (primary)
- [ ] Correct phase label ("paramagnetic" for all h_test)
- [ ] ZNE R² > 0.80 (quality gate — INDETERMINATE if below)
- [ ] ZNE gain > 0% (mitigation helps, not hurts)
- [ ] Layout std < 1.0 (layouts are consistent)

## Post-run

- [ ] Verify results in `results/hardware/run_*/summary.json`
- [ ] Check `provenance.json` for `total_qpu_seconds`
- [ ] Update `documentation/binnacles/` with hardware session binnacle
- [ ] Update `.kiro/steering/project-status.md` with hardware results
- [ ] Run `python -m project_health` to integrate into health report
- [ ] Compare HW results vs FakeTorino predictions (expected ±5-10%)

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `EnvironmentError: IBM_KEY not set` | Missing credentials | `export IBM_KEY=...` |
| Preflight abort: mean_2q_error > 1% | Bad calibration cycle | Wait 1-2h, retry |
| Preflight abort: queue > 50 | Peak hours | Run during UTC 2-6 AM |
| ZNE R² < 0.80 (INDETERMINATE) | PEA noise learning failed | Check layout quality, retry |
| ΔE/gap > 10% at h=4.0 (Tier 0 fail) | Fundamental pipeline issue | Debug with fake_backend |
| All energies identical | Cached estimator | Verify unique job_ids in provenance |
| Timeout on job collection | IBM queue congestion | Increase `job_timeout_s` to 1200 |
| Layout std > 2.0 | Diverse noise across qubits | Reduce `max_ces` to 0.3 |
| TLS drift abort mid-sweep | T1 degradation during run | Natural, restart later |

## Key Constraints (NEVER violate)

- p=1 for N≥10 (18 CX gates, within ZNE perturbative regime)
- PEA amplifier primary (gate-folding only as fallback if PEA unavailable)
- CES-ZNE is DEPRECATED on heavy_hex (uniform CES, R²≈0.04)
- Never apply SPSA if ΔE/gap ≤ 5% (refinement hurts good warm-starts)
- Never measure global fidelity on hardware (exponential tomography)
- Cost ceiling: 10M total shots per execution run
