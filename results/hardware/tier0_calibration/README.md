# Tier 0 PEA Calibration Study

Systematic comparison of PEA noise learning configurations on IBM Kingston (156Q Eagle r3).

## Purpose

Determine the minimum PEA learning budget that achieves ΔE/gap < 5% on real hardware
at h=4.0 (deep paramagnetic, N=10, p=1, heavy_hex).

## Directory Structure

Each subdirectory corresponds to a distinct PEA configuration:

```
tier0_calibration/
├── pea_32x128/          ← IBM default (4K learning shots)
├── pea_64x256/          ← Aggressive (16K learning shots)
├── pea_48x192/          ← Balanced (9K learning shots) [pending]
├── pea_32x128_3layout/  ← IBM default + 3 layouts [pending]
└── README.md
```

## Configurations Tested

| Config ID | num_rand | shots/rand | noise_factors | n_layouts | Total learning | Status |
|-----------|:--------:|:----------:|:-------------:|:---------:|:--------------:|--------|
| pea_32x128 | 32 | 128 | (IBM default) | 1 | 4,096 | ✅ ΔE/gap=32.5% |
| pea_64x256 | 64 | 256 | [1,1.5,2,3] | 3 | 65,536 | ❌ Cancelled (>17min) |
| pea_48x192 | 48 | 192 | [1,1.5,3] | 1 | — | Pending |
| pea_32x128_3layout | 32 | 128 | (IBM default) | 3 | 4,096 | Pending |

## Key Finding (2026-06-14)

- IBM default (32×128) is INSUFFICIENT at 3.36% mean 2Q error → ΔE/gap=32.5%
- Aggressive (64×256, 3 layouts, 4 factors) exceeds 15 min wall-clock → impractical for Tier 1+
- Next: test intermediate configs to find the sweet spot

## Fixed Hardware Parameters

- Backend: ibm_kingston (156Q Eagle r3)
- Topology: heavy_hex, N=10, p=1
- h_test: 4.0 (deep paramagnetic)
- DD: XpXm, Twirling: 32 rand, TREX: enabled
- Circuit: 9 RZZ gates, CES ≈ 0.050
- E_exact: -40.565690435512735, gap: 5.921971082752528
