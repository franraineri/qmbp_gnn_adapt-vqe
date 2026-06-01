# Coverage Scanner Results

Generated: 2026-06-01T01:41:13.470075

## Pipeline Results Summary

| Topology | N | p | Points | Pass | Median ΔE/gap |
|----------|---|---|--------|------|---------------|
| chain_1d | 6 | 1 | 5 | 4 | 0.0314 |
| chain_1d | 6 | 2 | 67 | 30 | 0.0223 |
| kagome | 6 | 2 | 1 | 1 | 0.0316 |
| ladder | 6 | 1 | 6 | 4 | 0.0155 |
| ladder | 6 | 2 | 46 | 24 | 0.0298 |
| triangular | 6 | 1 | 4 | 2 | 0.1011 |
| triangular | 6 | 2 | 42 | 22 | 0.0479 |

## Noisy/ZNE Results

| Topology | N | p | Runs | Mean Gain | ZNE Works |
|----------|---|---|------|-----------|-----------|
| chain_1d | 6 | 2 | 17 | +78.8% | 16/17 |
| ladder | 6 | 2 | 14 | +74.6% | 13/14 |
| triangular | 6 | 1 | 1 | +14.4% | 1/1 |
| triangular | 6 | 2 | 11 | +1.3% | 8/11 |

## Recommendations

1. **[MEDIUM]** p=1 additional seeds: chain_1d N=6
   - Reason: Only 1 seeds, need 3
   - Action: Run with seeds [43, 44]
