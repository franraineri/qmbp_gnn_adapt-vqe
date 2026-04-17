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
