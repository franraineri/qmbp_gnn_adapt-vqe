---
inclusion: fileMatch
fileMatchPattern: "**/mps_backend*,**/scaling/**,**/run_mps*,**/binnacle-mps*"
---

# MPS Scaling Context (invoke with #context-mps-scaling)

> Pre-digested context for any work involving MPS backend, N>30 scaling, or large-system VQE.

## What's Done

- N=40, 50, 80 validated (5/5 h-points each, ΔE/gap < 0.60%).
- Frontier fits (MPS deterministic, N=20-250): p=1: h=2.36+0.0073·N (R²=0.91), p=2: h=1.57+0.005·N (R²=0.95), p≥3: quasi-constant ~1.4-1.6.
- Multi-seed (42/43/44) at N=40: 27/27 pass, std=0.074%.
- Phase 3 MPNN at N=40: 0.46% deploy error with 27 training points.
- Zero-shot cross-N GNN validated with `norm_type="none"` (30/30 PASS).

## Key APIs

```python
from qmbp_simulation.execution import MPSBackend

# Default: deterministic mode (exact, 12ms/eval, no shot noise)
backend = MPSBackend(strategy="aer_mps", chi_max=64, seed=42)

# Legacy stochastic mode (backward-compatible, 6s/eval, σ≈precision)
backend = MPSBackend(strategy="aer_mps", chi_max=64, deterministic=False, precision=0.005)

from qmbp_simulation import VQEOptimizer, VQEConfig
# COBYLA works with both modes. L-BFGS-B only with deterministic=True (no shot noise).
config = VQEConfig(method="COBYLA", n_restarts=3, maxiter=500)
```

## MPS Evaluation Modes (since 2026-06-10)

| Mode | `deterministic=` | Per-eval | Accuracy | Shot noise | Use case |
|------|:-:|:-:|:-:|:-:|---------|
| **Exact** | `True` (default) | ~12ms | 10⁻¹⁴ | None | VQE loops, all new experiments |
| Stochastic | `False` | ~6s | σ≈0.005 | Yes | Reproduce pre-2026-06-10 results |

New results include `metadata.mps_evaluation_mode = "deterministic"` for traceability.
Results before 2026-06-10 implicitly used stochastic mode (no field present).

## Configuration Quick Reference

| N | chi | optimizer | restarts | h tested (passed) | timing (boundary) |
|---|-----|-----------|----------|-------------------|-------------------|
| 40 | 64 | COBYLA | 3 | 3.5-5.0 | ~5 min/point |
| 50 | 64 | COBYLA | 3 | 4.9-5.9 | ~6 min/point |
| 80 | 64 | COBYLA | 3 | 7.7-8.7 | ~22s/point |

Note: actual frontier at p=1 is h=2.36+0.0073·N (lower than tested). Experiments
used conservative h-values from the old formula, which is safe (overestimates).
| 100 | 64 | COBYLA | 3 | 9.9 (predicted) | untested |

## Constraints

- χ=64 is validated exact for HVA p≤2 on 1D TFIM at ANY N (actual DMRG χ=9-15).
- Dynamic chi formula: `min(400, max(200, 4*N))` — but 64 suffices for 1D.
- DMRG_QUBIT_LIMIT=100 works (N=40/50/60/80 converge <70s).
- Timing: T(N) ≈ 0.08·N^2.56 at boundary h-values.
- N=80 at h>>h_c is anomalously fast (trivial landscape).
- Hardware viability: N=40 (78 CX ✅), N=50 (98 CX ✅), N=80 (158 CX ⚠️ marginal).

## DO NOT

- Re-run χ convergence tests (proven: diff=1e-14).
- Use L-BFGS-B with shot-based backends (will fail silently).
- Attempt zero-shot cross-N without `norm_type="none"`.
- Extrapolate MPNN below h_min_safe (fails at boundary).
- Use VQE warm-start for cross-N with 2 params (useless — COBYLA always finds global min).

## Source Files

- #[[file:src/qmbp_simulation/execution/mps_backend.py]]
- #[[file:internal/documentation/binnacles/binnacle-mps-scaling.md]]
- #[[file:internal/documentation/analysis/17_scaling_N30_research_plan.md]]
- #[[file:results/scaling/]] (scaling_N40_*.json, scaling_N50_*.json, scaling_N80_*.json)
- #[[file:scripts/experiment_runners/scaling/]] (run_mps_*.py scripts)
