# Multi-Seed Cross-Topology Evaluation

**Date**: 2026-08-12
**Model**: tfim_bond_resolved, p=1
**Method**: UnifiedMPNN retrained per seed (--multi-n-train --force-retrain)
**Seeds**: 42, 43, 44 (via QMBP_GLOBAL_SEED env var)

---

## chain_1d (N=10, h=[1.5, 5.5])

| Seed | Pass@5% | Pass@dual | Mean ΔE/gap | Mean |ΔE| | Result file |
|------|---------|-----------|-------------|-----------|-------------|
| 42 | no_data | — | — | — | run_20260812_114041.json |
| 43 | no_data | — | — | — | run_20260812_120550.json |
| 44 | no_data | — | — | — | run_20260812_122457.json |

## heavy_hex (N=10, h=[1.4, 4.5])

| Seed | Pass@5% | Pass@dual | Mean ΔE/gap | Mean |ΔE| | Result file |
|------|---------|-----------|-------------|-----------|-------------|
