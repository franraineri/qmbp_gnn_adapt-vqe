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
  backend = service.backend("ibm_torino")
  print(backend.status())
  ```
- [ ] Check Torino queue depth (< 50 jobs ideal)
- [ ] Confirm calibration freshness (error rates accessible via `backend.target`)
- [ ] Run noisy simulation first: `python scripts/run_v61_noisy.py` (validates ZNE locally)

## Execution

- [ ] Start with h=2.0 (easiest — validates connection and pipeline)
- [ ] Then h=1.5 (thesis target, expected ΔE/gap ~3%)
- [ ] Then h=1.4 (harder, expected ΔE/gap ~5%)
- [ ] Use: `HardwareDeployerV61(mode="hardware", n_layouts=3, shots=16384)`
- [ ] Save all job IDs for provenance
- [ ] Monitor: expect ~2-5 min per h-value (queue + execution)

## Shot Budget

| N | Shots | σ | Sufficient? |
|---|-------|---|-------------|
| 6 | 8192 | 1.1e-2 | ✅ |
| 10 | 16384 | 7.8e-3 | ✅ |

## Success Criteria

- [ ] ΔE/gap < 5% at h=1.5 (primary)
- [ ] Correct phase label (paramagnetic for h > 1)
- [ ] ZNE R² > 0.8 (linear fit quality)
- [ ] No "indeterminate" phase labels at h=1.5 or h=2.0

## Post-run

- [ ] Log to binnacle: `--binnacle --label "IBM Torino run <date>"`
- [ ] Update `.kiro/steering/project-status.md` with hardware results
- [ ] Update `.kiro/knowledge/validation-targets.md` with hardware table
- [ ] Compare against noisy simulation predictions
- [ ] If ΔE/gap > 5%: check shot budget, try more layouts (n_layouts=5)

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| All energies identical | Estimator returning cached result | Check job ID uniqueness |
| ΔE/gap > 10% | Insufficient shots or bad layout | Increase shots to 32768 |
| "indeterminate" phase | Near critical point + noise | Expected at h≈1.0-1.2 |
| ZNE R² < 0.5 | Layouts too similar in CES | Increase n_layouts to 5 |
| AlgorithmError iter 0 | Warm-start was optimal | This is the IDEAL outcome |
| Connection timeout | IBM queue congestion | Retry after 30 min |
