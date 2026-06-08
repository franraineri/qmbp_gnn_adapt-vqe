# Bond-Resolved HVA Experiments

Per-bond/per-site parametrization of HVA circuits for scaling toward quantum advantage.

## Overview

Standard HVA uses 2 global parameters (one θ_zz for all bonds, one θ_x for all sites).
Bond-resolved HVA assigns independent parameters to each bond and each site, increasing
the variational space from 2 to (N_edges + N_qubits) without adding gates or depth.

## Key Results

| Metric | Global HVA | Bond-Resolved HVA | Improvement |
|--------|:---:|:---:|:---:|
| Heavy-hex N=10 ΔE/gap | 0.36% | 0.18% | **+49.7%** |
| Chain N=10 ΔE/gap | 1.58% | 1.57% | +1.0% |
| Square N=9 (3×3) ΔE/gap | — | 0.91% | (new topology) |
| Square N=12 (3×4) ΔE/gap | — | 0.54% | (new topology) |
| ZNE gain (heavy-hex) | +62.7% | **+30.8%** | Same CX budget |
| MPNN prediction error | — | 0.16-0.23% | GNN works on 19-dim output |

## Experiments

### `run_bond_resolved_validation.py`
Initial 4-section validation: convergence, comparison, spatial analysis.
**Result**: 4/4 PASS. Experiment ID: `BOND_RESOLVED_HVA`.

### `run_bond_resolved_scaling.py`
Scaling suite: 2D square lattice, MPNN training, ZNE validation.
**Result**: 4/4 PASS. Experiment ID: `BOND_RESOLVED_SCALING`.

### `run_n16_square_dmrg2d.py`
N=16 (4×4) square lattice with DMRG 2D ground truth (40 bond-resolved params).
**Result**: Pending. Experiment ID: `N16_SQUARE_DMRG2D`.

## Usage

```bash
# Full validation suite
python scripts/experiment_runners/bond_resolved/run_bond_resolved_validation.py

# Scaling (2D + MPNN + ZNE)
python scripts/experiment_runners/bond_resolved/run_bond_resolved_scaling.py

# N=16 square (slow: ~15-20 min)
python scripts/experiment_runners/bond_resolved/run_n16_square_dmrg2d.py

# Dry run (list sections)
python scripts/experiment_runners/bond_resolved/run_bond_resolved_scaling.py --dry-run

# Run specific section
python scripts/experiment_runners/bond_resolved/run_bond_resolved_scaling.py --section 3
```

## Architecture

```
HVACircuitBuilder.create_bond_resolved(n_qubits, p_layers, lattice)
    → QuantumCircuit with (n_edges + n_qubits) * p_layers parameters
    → Same RZZ/RX gates as standard HVA (identical CX budget)
    → Parameter ordering: [θ_zz_0, ..., θ_zz_{E-1}, θ_x_0, ..., θ_x_{N-1}] per layer
```

## Why This Matters for Quantum Advantage

1. **Higher-dimensional variational space**: 40 params at N=16 vs 2 — harder for TN
2. **GNN becomes essential**: interpolation fails in 19+ dimensions
3. **Same CX budget**: ZNE still works, no extra noise
4. **Topology-aware**: captures non-uniform bond structure (heavy-hex nodes of degree 2 vs 3)
5. **Scales to N=50+**: at N=50 heavy-hex → 60+ params, genuinely classical-hard
