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
- [ ] Shot budget ≥ 8192 (shot noise ~1/√shots must be < ⟨X⟩ signal ~8e-3)
- [ ] Learned DD sequences via Qiskit DD pass manager (Pokharel et al. 2025)
- [ ] Inhomogeneous ZNE: multiple qubit mappings for natural noise scaling (Uvarov et al. 2024)

## Error Mitigation Strategy (Literature-Informed)

### Standard ZNE (Gate Folding)
- Insert identity-equivalent gate pairs to amplify noise: 1x, 2x, 3x
- Extrapolate to 0x noise via linear/polynomial fit
- Overhead: 3× circuit executions

### Inhomogeneous ZNE (Uvarov et al. 2024) — PREFERRED
- Exploit non-uniform error rates across IBM chip (different qubits have different T1/T2)
- Transpile same circuit with different qubit mappings → different total Circuit Error Sum (CES)
- Linear energy-CES extrapolation to zero CES
- Advantage: no gate folding needed, uses natural hardware noise variation
- Implementation: multiple calls to `generate_preset_pass_manager()` with different `initial_layout`

### NN-Enhanced ZNE (Sun et al. 2025)
- Replace linear/polynomial extrapolation with 2-layer MLP
- Train on (noise_level, energy) pairs from multiple noise amplification runs
- Constrains errors to O(10⁻²)–O(10⁻¹) vs O(10⁻¹) for standard ZNE
- Implementation: after collecting ZNE data points, fit `MLPRegressor(hidden_layer_sizes=(16,8))` instead of `np.polyfit`

### Dynamical Decoupling (Pokharel et al. 2025)
- Genetic algorithm finds device-specific DD sequences outperforming canonical XY4/CPMG
- Scales to 100 qubits, generalizes from small sub-circuits
- Free improvement (no extra shots, no extra circuit depth)
- Implementation: use Qiskit `PadDynamicalDecoupling` pass with optimized sequences

### U-VQNHE Post-Processing (Kim et al. 2026)
- Learnable diagonal reweighting of measurement outcomes
- Variational safety guaranteed (never sub-variational)
- Tested on TFIM — directly applicable
- Implementation: classical post-processing of raw measurement counts

## Shot Noise Analysis (Sharma 2026)

| Shots | Statistical uncertainty | Sufficient for ⟨X⟩ ~8e-3? |
|-------|------------------------|---------------------------|
| 1024 | ~3.1e-2 | ❌ Noise dominates signal |
| 4096 | ~1.6e-2 | ❌ Noise ≈ signal |
| 8192 | ~1.1e-2 | ⚠️ Borderline |
| 16384 | ~7.8e-3 | ✅ Signal > noise |

Recommendation: use 8192 shots minimum for Phase 4. Group commuting observables to reduce total circuit executions.

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
