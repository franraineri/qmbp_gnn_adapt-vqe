---
inclusion: fileMatch
fileMatchPattern: "src/poc/v6/hardware_deployer.py"
---

# Hardware Deployment — Phase 4 Guidelines

## IBM Torino Target (133 qubits, Eagle r3)

### Connection
- Use `QiskitRuntimeService` with `channel="ibm_quantum_platform"`
- Token from `os.environ["IBM_KEY"]`, instance from `os.environ["IBM_INSTANCE_CRN"]`
- Backend: `"ibm_torino"` (or latest Eagle r3 processor)

### Circuit Preparation
- `generate_preset_pass_manager(backend=backend, optimization_level=2)`
- Apply layout to observables: `obs.apply_layout(isa_qc.layout)`
- Add dynamical decoupling: `PadDynamicalDecoupling` pass after transpilation

### Error Mitigation Stack (in order of application)
1. **Dynamical Decoupling** — free, always apply. Use optimized sequences if available.
2. **TREX** — twirled readout error extinction. Enable via EstimatorV2 options.
3. **Inhomogeneous ZNE** — preferred over gate folding:
   - Transpile same circuit with 3-5 different `initial_layout` values
   - Each layout produces different Circuit Error Sum (CES)
   - Linear extrapolation of energy vs CES to zero
4. **NN-enhanced extrapolation** — optional improvement:
   - After collecting ZNE data, fit 2-layer MLP instead of linear regression
   - `MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1000)`

### Shot Budget
- Minimum: **8192 shots** (σ ≈ 1.1e-2, comparable to ⟨X⟩ signal)
- Recommended for N=10: **16384 shots** (σ ≈ 7.8e-3, below ⟨X⟩ signal of 8.4e-3)
- Use `EstimatorV2` `precision` parameter to control shot allocation

### Observable Grouping
- ⟨X_i⟩ observables: all commute (single measurement basis)
- ⟨Z_iZ_{i+1}⟩ observables: all commute (single measurement basis)
- Total: 2 circuit executions per noise level (not N+N-1 separate runs)

### Expected Hardware Behavior (from literature)
- Ground-state energies: reliably captured across full parameter space
- Magnetic order parameters: noise broadening near critical crossover
- Phase classification: correct away from h_c, "smeared" near transition
- Success criterion: ΔE/gap < 5% AND correct phase label — NOT fidelity ≥ 99.5%

### AdaptVQE on Hardware
- max_iterations = 2 (Mele et al. constraint)
- gradient_threshold = 1e-3
- If AlgorithmError at iteration 0 → ideal outcome (warm-start was optimal)
- Pauli pool: Hamiltonian terms only (ZZ bonds + X sites)
- Use COBYLA or SPSA optimizer (gradient-free, noise-robust)

### Do NOT
- Measure global fidelity on hardware (requires exponential tomography)
- Use more than p=2 total HVA layers (including ADAPT additions)
- Use Primitives V1 or `backend.run()`
- Hardcode h_c = 1.0 for phase classification (use data-driven crossover)
