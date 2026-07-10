---
inclusion: fileMatch
fileMatchPattern: "**/solvers/**,**/execution/**,**/optimizers/**,**/run_noiseless*,**/run_scaling*"
---

# Method Dispatch by System Size

## Phase 1: Ground Truth (ClassicalSolver.solve)

| N range | Method | Module | Time | Output |
|---------|--------|--------|------|--------|
| N ≤ 15 | `np.linalg.eigh` (dense) | `solvers/classical.py:_solve_exact` | <1s | E₀, gap, ψ_gs (2^N), ⟨X_i⟩, ⟨Z_iZ_j⟩ |
| 15 < N ≤ 22 | DMRG (TeNPy TFIChain) | `solvers/classical.py:_solve_dmrg_1d` | 2-10s | E₀, gap (analytical fallback), ⟨X_i⟩, ⟨Z_iZ_j⟩. **No ψ_gs** |
| N > 22, 1D | DMRG (TeNPy TFIChain) | `solvers/classical.py:_solve_dmrg_1d` | 5-70s | Same as above |
| N > 22, 2D | DMRG (TeNPy SpinModel) | `solvers/classical.py:_solve_dmrg_2d` | 10-120s | Same. Gap = finite-size floor |

Constants: `EXACT_DIAG_QUBIT_LIMIT=15`, `DMRG_QUBIT_LIMIT=200`

### Gap estimation

- N ≤ 15: exact (full spectrum from eigh)
- N > 15, chain_1d: analytical `2|J-h|` with floor `2π/N` (valid far from h_c)
- N > 15, other topologies: finite-size floor `2π/N` only (conservative lower bound)
- Near h_c ≈ 1.0: gap estimate is UNRELIABLE → ΔE/gap metrics lose meaning

## Phase 2: VQE Backend (energy evaluation ⟨H⟩)

| N range | Backend | Module | Time/eval | Selection |
|---------|---------|--------|-----------|-----------|
| N ≤ 10 (VQE loop) | `NoiselessBackend` (StatevectorEstimator) | `execution/backends.py` | 0.01-0.05s | `select_backend(N, for_vqe_loop=True)` |
| N > 10 (VQE loop) | `MPSBackend` (Aer MPS, deterministic) | `execution/mps_backend.py` | 0.1-0.3s | `select_backend(N, for_vqe_loop=True)` |
| N ≤ 15 (single eval) | `NoiselessBackend` | same | 0.01s | `select_backend(N, for_vqe_loop=False)` |
| N > 15 (single eval) | `MPSBackend` | same | 0.1-0.3s | `select_backend(N, for_vqe_loop=False)` |

### Why MPS for VQE at N > 10

StatevectorEstimator is O(2^N) per gate application. At N=12 each eval takes ~5s
(vs 0.01s at N=6). VQE does thousands of evals per h-point → unacceptable.
MPSBackend is O(N·χ³) = O(N) at fixed χ=64 → scales linearly.

## Phase 2: Fidelity Computation (ψ_VQE vs ψ_exact)

| N range | Method | Time | Notes |
|---------|--------|------|-------|
| N ≤ 15 | `np.linalg.eigh` → full vector + `state_fidelity()` | <0.1s | Exact |
| 15 < N ≤ 22 | `scipy.eigsh` (sparse Lanczos) + MPS→statevector extraction | 13-16s | **Bottleneck for N=16-22** |
| N > 22 | **Not available** — raise ValueError | — | Cannot create 2^N vector |

### Recommendation for N ≥ 16

Disable fidelity computation (`compute_fidelity=False` in `vqe_adaptive_sweep`).
Use ΔE/gap as primary quality metric instead. The 13s/point eigsh overhead is
the main reason N=20 heavy_hex runs take 68 min/point instead of 6 min/point.

## MPNN Training Data Flow

```
Phase 1 (solver.solve)  →  e_exact, gap per h-point
Phase 2 (VQE)           →  theta_opt, energy_vqe, fidelity per h-point
                        ↓
build_graph_dataset(lattice, h_values, theta_opt, e_exact, fidelities)
                        ↓
train_mpnn(predictor, dataset, energy_val_fn=...)
```

### Input validation checks (already enforced)

- h_points ≥ 3 (preflight error), ≥ 5 recommended (preflight warning)
- h_min < h_max (preflight error)
- h_min ≥ 0 (preflight error)
- fidelity_threshold=0.0 for noiseless (no filtering — outlier detection handles bad points)
- theta_alignment pass if smoothness > 1.0
- cross_h_energy_guard for local-minimum traps

### What MPNN receives as features

- Node features: `[h/100, coordination_number/max_coord]` (2 features per node)
- Edge connectivity: from lattice topology
- Target: theta_opt array (2*p floats for TFIM)

## Key Thresholds (constants.py)

| Constant | Value | Used by |
|----------|-------|---------|
| `EXACT_DIAG_QUBIT_LIMIT` | 15 | solver dispatch, select_backend |
| `STATEVECTOR_MAX_N` | 22 | ground_state_vector, get_statevector |
| `DMRG_QUBIT_LIMIT` | 200 | solver maximum |
| `MPS_DEFAULT_CHI_MAX` | 64 | MPSBackend bond dimension |
| `COBYLA_AUTO_SWITCH_THRESHOLD` | 8 | VQE optimizer method selection |
| `VQE_WALL_CLOCK_LIMIT_PER_POINT` | 600 | COBYLA timeout per optimize() call |
