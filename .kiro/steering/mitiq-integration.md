---
inclusion: fileMatch
fileMatchPattern: "**/mitiq*"
---

# Mitiq Integration — Steering Rules

## Module Location

`src/qmbp_simulation/execution/mitiq_utils.py` — optional dependency (`pip install mitiq`).

## Critical: optimization_level=0

**Qiskit 2.x at optimization_level ≥ 1 cancels U·U† gate pairs, destroying Mitiq's gate folding.**
The `make_mitiq_executor()` forces `optimization_level=0` internally — NEVER change this.
Our native GF-ZNE is not affected (it folds AFTER transpilation).

## Available Functions

```python
from qmbp_simulation.execution.mitiq_utils import (
    make_mitiq_executor,       # Backend+observable → executor(circuit) → float
    make_noiseless_executor,   # StatevectorEstimator executor for CDR training
    run_mitiq_zne,             # ZNE: random/global/all folding + linear/richardson/poly/exp
    run_mitiq_cdr,             # CDR: Clifford Data Regression (learning-based)
    run_mitiq_ddd_zne,         # DDD+ZNE: dynamical decoupling + ZNE composition
    run_mitiq_pec,             # PEC: Probabilistic Error Cancellation (benchmark only)
    compare_mitigation_strategies,  # Run all methods, produce ranked comparison
    is_mitiq_available,        # Check if mitiq installed
)
```

## Strategy Hierarchy for Hardware

```
PRIMARY:   PEA-ZNE (our implementation, +94.4% gain, IBM server-side)
VERIFY:    Mitiq CDR (independent cross-check, no noise model needed)
ALWAYS:    Affine correction (zero cost, physics bounds)
FALLBACK:  Mitiq ZNE random (if PEA unavailable)
BENCHMARK: compare_mitigation_strategies() (thesis table material)
```

**CRITICAL FINDING (2026-06-18 benchmark V2)**: Mitiq ZNE with opt_level=0 is
**destructive** for N≥10 with hardware routing (FakeTorino: 81% ΔE/gap vs 41.9% raw).
The optimization_level=0 requirement produces 45 CZ gates (vs 18 at opt_level=2),
completely negating ZNE benefits. Mitiq is ONLY useful for:
- CDR cross-verification (`--mitiq-verify`, if dimension permits)
- Simple depolarizing benchmarks (N=4-6, no routing)
- NOT as primary mitigation for production circuits with routing

**V2 correction**: V1 showed 67.9% — V2 with θ_opt shows 81% (worse, because
real noise on 45 CZ is stronger than on trivial θ=zeros circuits).

## Key Constraints

- Mitiq ZNE `folding_method`: "random", "global", "all" (Mitiq 1.0 API)
- Mitiq ZNE `factory_name`: "linear", "richardson", "poly", "exp"
- DDD rules: "xx", "yy", "xyxy"
- CDR requires circuit in IBM native basis {Rz, √X, CX} for near-Clifford generation
- PEC overhead: exp(2·n_2q·ε) — viable for our short circuits (N=10 p=1: ~11%)
- All closures passed to Mitiq MUST have `__annotations__ = {"circuit": QuantumCircuit, "return": float}`
  (Mitiq 1.0 Executor bug with closure type detection)

## Integration Points

- **Rehearsal V3 Section 21**: `section_mitiq_comparison()` — compares methods at h_test points
- **Hardware deployment**: Use `compare_mitigation_strategies` post-PEA for verification
- **Noisy simulation**: `run_mitiq_zne` / `run_mitiq_cdr` as alternatives to our native GF-ZNE

## Results Pattern

All functions return typed dataclasses: `MitiqZNEResult`, `MitiqCDRResult`,
`MitiqDDDZNEResult`, `MitiqPECResult`, `MitiqComparisonResult`.
All are JSON-serializable via `json_serialize()`.
