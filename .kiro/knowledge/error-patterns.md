# Common Errors & Debugging Recipes

## V5.x Failure Mode (Phase Coupling)

**Symptom**: MSE converges but ΔE stagnates or increases.
**Root cause**: Phase 2 cost function changed (e.g., fidelity-weighted) without updating Phase 3 training target.
**Fix**: Phase 2 MUST use pure energy cost. V6 enforces this via `cost_function="energy"` metadata in the .npz dataset. Phase 3 loading validates this field.

## Symmetry Saddle at θ=0

**Symptom**: VQE converges at iteration 0 with E ≈ E(|+⟩^N), fidelity ~50%.
**Root cause**: |+⟩^N is a saddle point — gradient ≈ 1e-6 at θ=0, and L-BFGS-B's default `pgtol=1e-5` declares convergence.
**Fix**: Initialize with `np.random.uniform(-0.01, 0.01)`, never zeros. V6 enforces this in `VQEOptimizer.get_initial_guess()`.

## HVA Expressibility Ceiling

**Symptom**: Fidelity < 50% for h < 0.8 (ferromagnetic regime).
**Root cause**: HVA p=2 + |+⟩^N cannot express the ferromagnetic ground state |000...0⟩. Verified with 50 random restarts over [-π, π].
**Fix**: This is a known limitation, not a bug. Pipeline is valid for h ≥ 1.0 (fid > 96%). Use fidelity filter ≥ 93% in Phase 3 to exclude bad training data.

## AdaptVQE AlgorithmError at Iteration 0

**Symptom**: `AlgorithmError: "All gradients below threshold at first iteration"`.
**Root cause**: The warm-start θ_pred is already near-optimal — all Pauli pool gradients are below `gradient_threshold`.
**This is the IDEAL outcome** — it means the GNN/MPNN prediction was so good that no circuit depth increase was needed.
**Fix**: Catch `AlgorithmError`, evaluate energy of the initial state directly.

## DMRG Symmetry Breaking

**Symptom**: ⟨X⟩ ≈ 0 even in the paramagnetic phase (h > 1).
**Root cause**: DMRG may find a Z₂ symmetry-broken ground state.
**Fix**: Use |⟨X⟩| (absolute value) or XX correlation estimator. For N < 15, use exact diag instead.

## PoC V6 Baseline Results (N=6, TFIM, p=2, best config: 5 rst, 6000 ep)

| Metric | h=1.25 | h=1.4 | h=1.5 | Threshold |
|--------|--------|-------|-------|-----------|
| ΔE/gap | 3.5% ✅ | 1.9% ✅ | 1.4% ✅ | < 5% |
| ⟨X⟩ error | ~1e-2 ⚠️ | 5e-3 ✅ | 2.6e-3 ✅ | < 1e-2 |
| ⟨ZZ⟩ error | 2.5e-2 ❌ | ~1.5e-2 ❌ | 5e-3 ✅ | < 1e-2 |
| ΔE | 3.5e-2 ❌ | ~1.7e-2 ❌ | ~1.4e-2 ❌ | < 1e-2 |
| Fidelity | 0.991 ❌ | 0.995 ✅ | 0.997 ✅ | ≥ 99.5% |
| ADAPT iters | 2 ✅ | 2 ✅ | 2 ✅ | ≤ 2 |
| **Checklist** | **2–3/6** | **4–5/6** | **5/6** | |

**Key insight**: ΔE threshold is aspirational — bounded by HVA expressibility at each h. The ΔE/gap metric correctly shows the pipeline resolves the physics.
