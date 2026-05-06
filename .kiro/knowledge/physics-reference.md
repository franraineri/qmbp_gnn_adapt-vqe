# Condensed Matter Physics Reference

## 1D Transverse Field Ising Model (TFIM)

- H = -J Σ ZᵢZᵢ₊₁ - h Σ Xᵢ
- QPT at h/J = 1.0 (thermodynamic limit)
- h/J < 1: ferromagnetic — ⟨ZᵢZᵢ₊₁⟩ → 1, ⟨Xᵢ⟩ → 0
- h/J > 1: paramagnetic — ⟨ZᵢZᵢ₊₁⟩ → 0, ⟨Xᵢ⟩ → 1
- Finite-size effects shift critical point; gap closes as 1/N
- Exact solution via Jordan-Wigner → free fermions

## Spin Ladders (quasi-1D)

- H = J_leg Σ ZᵢZⱼ (legs) + J_rung Σ ZᵢZⱼ (rungs) + h Σ Xᵢ
- DMRG-friendly, MPS efficient. Fallback from full 2D.
- Bond dimension scaling: χ ∝ e^W (W = ladder width). W=2: χ~50-100 sufficient. W=4: χ~500+.
- TeNPy: use `Ladder` lattice class with `Lx` (length) and `Ly=2` (width)
- Phase diagram richer than 1D chain: rung-singlet phase, Haldane-like phases possible
- MPNN transfer: graph structure changes (degree 3 vs degree 2) — expect fine-tuning needed

## Kagome Lattice (2D Frustrated)

- Triangular arrangement with 3 sites per unit cell, coordination number z=4
- Antiferromagnetic Heisenberg: ground state is candidate quantum spin liquid (QSL)
- No local order parameter — characterized by topological entanglement entropy
- DMRG: practical up to N~24-36 with χ~2000-4000 (Yan et al. 2011, Depenbrock et al. 2012)
- **IBM Heron achievement** (Ahsan et al. 2025): 103-site Kagome VQE on Heron r1/r2
  - Hybrid local(classical) + global(quantum) VQE split
  - Single-repetition HEA with Hamiltonian engineering
  - Per-site energy: -0.417J (matches thermodynamic limit)
- **Contextual Subspace VQE** (Weaving et al. 2025): Kagome on NISQ with DMRG-biased subspaces
  - Qubit reduction via contextual subspace methodology
  - REM + SV + ZNE error mitigation → 0.01% energy error
  - Validates: DMRG + VQE hybrid approach works for frustrated systems
- Our scaling target: N=12 (exact diag limit) → N=36 (QPU only)

## Scaling Roadmap (Updated with Literature)

| System | N | Method | Status | Reference |
|--------|---|--------|--------|-----------|
| 1D chain | 6 | Exact diag + VQE | ✅ Done (5/6 at h=1.5) | Our binnacle |
| 1D chain | 10 | Exact diag + VQE | ✅ Done (3/6 at h=1.5) | Our binnacle |
| Ladder (2-leg) | 6 | Exact diag + VQE | 🔜 Next | — |
| Ladder (2-leg) | 10 | Exact diag + VQE | 🔜 Planned | — |
| 1D chain | 14 | Exact diag limit | Planned | — |
| Ladder (2-leg) | 20 | DMRG + VQE | Future | — |
| Kagome | 12 | Exact diag + VQE | Future | — |
| Kagome | 36 | QPU only | Thesis target | Ahsan 2025 validates feasibility |
| Kagome | 103 | QPU (Heron) | Literature | Ahsan et al. 2025 |

## Order Parameters

| Observable | Detects | Formula |
|-----------|---------|---------|
| Transverse magnetization | Paramagnetic order | M_x = (1/N) Σ ⟨Xᵢ⟩ |
| Staggered magnetization | AF order | M_z^stag = (1/N) Σ (-1)^i ⟨Zᵢ⟩ |
| Correlation function | Phase type (decay rate) | C(r) = ⟨ZᵢZᵢ₊ᵣ⟩ |
| Entanglement entropy | Critical point | S = -Tr(ρ_A log ρ_A) |
| Energy gap | Phase transition | Δ = E₁ - E₀ |

Finite-size phase classification: use ⟨X⟩ = ⟨ZZ⟩ crossover from exact data, not hardcoded h_c = 1.0.

## SPT Phases (Fallback)

- Constant-depth circuits (noise-friendly)
- Detected via string order parameters, not local magnetization
- Use if QSL characterization fails due to hardware noise

## Quantum Spin Liquids (QSL)

- Emerge in frustrated lattices (Triangular, Kagome) where AF interactions can't all be satisfied
- No local order parameters; signature is Topological Entanglement Entropy
- Require quasi-1D spin ladders as proxy (deep circuits prohibited by Mele et al.)

## HVA Mathematical Construction

|ψ(θ)⟩ = ∏_{l=1}^{p} ( e^{-i θ_{l,B} H_B} e^{-i θ_{l,A} H_A} ) |+⟩^{⊗N}

- H_A = ZZ terms, H_B = X terms
- Qiskit: e^{-i θ Z_i Z_{i+1}} → `RZZ(2*theta)`, e^{-i φ X_i} → `RX(2*phi)`

## Mele et al. — Noise & Barren Plateaus

- **Unital noise** (depolarizing): maps maximally mixed state to itself → standard barren plateaus
- **Non-unital noise** (amplitude damping, typical IBM): biases toward |00...0⟩ → breaks isotropic flatness
- Non-unital noise: gradient variance does NOT vanish exponentially for local cost functions → shallow HVA trainable
- Same noise erases early gate influence → effective depth truncation to O(log n)
- **Conclusion**: GNN + Shallow HVA is required by the thermodynamics of current processors

## Three-Way Synergy (Unique to Our Architecture)

No other known approach simultaneously achieves:
1. **Noise resilience** — shallow circuits (p≤2) survive decoherence (Mele et al. 2026)
2. **Trainability** — local costs avoid barren plateaus (Cerezo et al. 2021)
3. **Physical expressibility** — HVA respects Hamiltonian symmetries (Wiersema et al. 2020)
4. **Efficiency** — MPNN warm-start eliminates quantum optimization cost (our contribution)

This is the *only* known regime where VQAs are simultaneously trainable, noise-resilient, and physically meaningful on NISQ hardware. The GNN adds a fourth dimension: near-zero quantum resource usage.

## Warm-Start Gradient Amplification (Puig et al. 2025)

- Warm-starts provide provably larger loss variances (stronger gradient signals) vs random initialization
- At h=2, ground state ≈ |+⟩^N (our initial state), so θ* ≈ 0 → large gradients
- Descending sweep keeps each optimization in a region of strong gradients
- Quantitative: 10-50× speedup vs random init at h=1.25 (from our benchmarks)
- The resulting θ(h) landscape is smooth → ideal for MPNN learning

## Quantum Advantage Boundary

- **N=6-10 (1D chain)**: fully classically simulable. Pipeline demonstrates METHODOLOGY, not quantum advantage.
- **N≈20 (2D systems)**: TN simulations become expensive. QPU starts to offer better scaling (Martin et al. 2026).
- **N=36 (Kagome)**: no classical method works. Only QPU + MPNN warm-start can characterize the phase.
- **Thesis narrative**: PoC validates the pipeline; scaling targets demonstrate quantum utility.
