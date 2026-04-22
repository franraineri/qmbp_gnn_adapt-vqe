# Optimization & Hardware Deployment

## Optimizer Selection

| Optimizer | Use case | Context |
|-----------|----------|---------|
| L-BFGS-B | Noiseless statevector (Phase 2) | Best convergence, needs gradient |
| COBYLA | VQE loops, noise-robust | Gradient-free |
| SPSA | Hardware with shot noise | Stochastic gradient |

## Warm-Start Protocol

- Sweep DESCENDING: h=2→0 (|+⟩^N exact at h→∞)
- **Non-uniform h-grid** (NN-VQE, Miao et al. 2024): θ_opt changes abruptly near the phase transition. Use denser sampling near the finite-size critical point:
  - Δh=0.1 far from criticality (h∈[0,0.7] and h∈[1.5,2.0])
  - Δh=0.05 in the critical region (h∈[0.8,1.4])
  - Current PoC: 27 points total (was 21 uniform)
- Init: `np.random.uniform(-0.01, 0.01, n_params)` — never zeros (symmetry saddle)
- Convergence: |ΔE| < 1e-6 (statevector), |ΔE| < 1e-3 (hardware)

## Known Pitfalls

### Symmetry Saddle at θ=0
|+⟩^N is eigenstate of X operators with symmetric ZZ correlations → gradient ~1e-6 at θ=0. L-BFGS-B default `pgtol=1e-5` declares convergence at iter 0.

### HVA Expressibility Limit
HVA p=2 + |+⟩^N cannot reach ferromagnetic ground state (h→0 = |000...0⟩). Fidelity: ~22% at h=0 for N=6 (verified with 50 random restarts over [-π,π]). Pipeline valid for h ≥ 1.0 (fid > 96%).

## Observable Extraction

- ⟨Xᵢ⟩ per site → paramagnetic phase
- ⟨ZᵢZᵢ₊₁⟩ per bond → ferromagnetic order
- ⟨Hᵢ,ᵢ₊₁⟩ per bond → local energy landscape

## Hardware Checklist

- [ ] `generate_preset_pass_manager(optimization_level=2)`
- [ ] `.apply_layout()` on observables
- [ ] Error mitigation: TREX (p≤2), ZNE, or PEC
- [ ] VQE iterations ≤ 2 on hardware
- [ ] Noise budget: p=1 (~12 2q gates) OK; p=2 (~24 2q gates) marginal
- [ ] If ΔE > 5% of gap → fall back to p=1
- [ ] Log: backend, calibration date, job ID, raw + mitigated results

## Numerical Limits

| Constraint | Limit |
|-----------|-------|
| Exact diagonalization | n ≤ 14 |
| Statevector simulation | n ≤ 20 |
| Minimum shots | ≥ 1024 |

## Data Pipeline

- Phase 1: (h, J) → {ground_energy, ground_state, local_observables}
- Phase 2: (h, J) → {θ_opt, final_energy, energy_error, n_iterations}
- Phase 3: Hamiltonian graph → θ_pred
- All datasets: include n_qubits, model_type, p_layers, optimizer, timestamp
- Storage: .npz or HDF5, version with (n_qubits, p_layers, optimizer) in filename
