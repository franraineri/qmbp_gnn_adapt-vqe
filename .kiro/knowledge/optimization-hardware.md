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

### Qiskit Runtime Resilience Levels (IBM Official)

| Level | Techniques Enabled | Overhead | Use Case |
|-------|-------------------|----------|----------|
| 0 | None | 1× | Debugging, raw noise characterization |
| 1 | TREX (measurement mitigation) | ~1.5× | Baseline — always use at minimum |
| 2 | TREX + ZNE + Pauli twirling | ~3× | **Recommended for our Phase 4** |
| 3 | TREX + PEC | 10-100× | Unbiased but expensive — future work |

### Recommended Configuration for Phase 4

```python
from qiskit_ibm_runtime import EstimatorV2, Options

options = Options()
# Error suppression (free)
options.dynamical_decoupling.enable = True
options.dynamical_decoupling.sequence_type = "XpXm"  # or "XY4"
# Twirling (converts coherent → stochastic noise)
options.twirling.enable_gates = True
options.twirling.num_randomizations = 32
options.twirling.shots_per_randomization = 256  # total = 32×256 = 8192 shots
# Error mitigation
options.resilience.measure_mitigation = True  # TREX
options.resilience.zne_mitigation = True  # ZNE
options.resilience.zne.noise_factors = [1, 2, 3]
options.resilience.zne.extrapolator = "exponential"
# For PEA (more accurate noise amplification):
# options.resilience.pea = True  # requires noise learning phase
```

### IBM Heron r2 vs Eagle r3 (2026 Status)

| Property | Eagle r3 (ibm_torino) | Heron r2 (ibm_kingston) |
|----------|----------------------|------------------------|
| Qubits | 133 | 156 |
| Connectivity | Heavy-hex, fixed-frequency | Heavy-hex, tunable couplers |
| 2Q gate error | ~0.3-0.5% | ~0.1-0.2% |
| 2Q gate type | ECR | CZ (via tunable coupler) |
| Crosstalk | Higher (fixed coupling) | Lower (tunable isolation) |
| Best for | Larger circuits, more qubits | Higher fidelity, fewer qubits |

**Recommendation**: If available, prefer Heron r2 for our N=6-10 circuits (fewer qubits needed, higher fidelity matters more). Use Eagle r3 for N=20+ scaling.

### GADD: Learned Dynamical Decoupling (Qiskit Community Package)

```bash
pip install gadd  # Qiskit community package
```

```python
from gadd import GADD
# Optimize DD sequences for specific backend + circuit
gadd = GADD(backend=backend)
optimized_sequences = gadd.optimize(circuit, num_generations=50)
```

Alternative: use built-in sequences ("XX", "XpXm", "XY4") via `options.dynamical_decoupling.sequence_type`.

### Inhomogeneous ZNE (Uvarov et al. 2024)
- Transpile same circuit with different `initial_layout` values
- Each layout maps to qubits with different error rates → different CES
- Linear extrapolation of energy vs CES to zero
- Implementation: multiple `generate_preset_pass_manager()` calls with `initial_layout=[...]`

### NN-Enhanced ZNE (Sun et al. 2025)
- After collecting ZNE data at noise factors [1, 2, 3], fit MLP instead of polynomial
- `sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1000)`
- Constrains errors to O(10⁻²) vs O(10⁻¹) for standard linear/exponential fit

### QESEM Framework (Aharonov et al. 2026)
- Resolves ZNE vs PEC tradeoff: quasi-probabilistic mitigation with reduced overhead
- Tested on kicked TFIM on IBM Heron — directly applicable
- Higher accuracy than ZNE, lower cost than PEC
- Not yet in Qiskit Runtime — requires custom implementation or Classiq SDK

### Utility-Scale Kagome Reference (Ahsan et al. 2025)
- 103-site Kagome on IBM Heron r1/r2 with single-repetition HEA
- Hybrid local(classical) + global(quantum) VQE split
- Hamiltonian engineering to simplify ansatz
- Per-site energy: -0.417J (matches thermodynamic limit after boundary corrections)
- **Validates**: IBM Heron can handle frustrated 2D systems at utility scale

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
