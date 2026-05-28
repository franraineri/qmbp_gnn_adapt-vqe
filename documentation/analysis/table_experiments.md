# Results Digest

## Hypothesis Tests (BaseExperiment)

**Summary**: 15 experiments — 8 confirmed ✅, 5 rejected ⚠️, 2 failed ❌

| ID | Category | Verdict | ΔE/gap | Pass% | Criteria |
|-----|----------|---------|--------|-------|----------|
| A3 | A | ✅ confirmed | 0.0000 | 100% | Scaling law R²>0.99 |
| A3_N20 | A | ✅ confirmed | 0.0000 | 100% | Scaling at N=20 |
| B2 | B | ✅ confirmed | 0.1681 | 67% | Freeze works at h≥1.5 |
| B4 | B | ✅ confirmed | 0.0501 | 75% | No saddle points (physics-limited pts excluded) |
| D1 | D | ✅ confirmed | 1.1527 | 1% | Gradient peak detected near h_c |
| F3 | F | ✅ confirmed | 5.4552 | 0% | Fluctuation > 1.0 everywhere |
| G1 | G | ✅ confirmed | 0.0222 | 86% | ≤9 pts sufficient |
| G5 | G | ✅ confirmed | 0.0174 | 92% | Seed-independent (std<0.01) |
| E4 | E | ⚠️ rejected | 0.2464 | 24% | HVA fails at g>0 |
| F1 | F | ⚠️ rejected | 0.1372 | 64% | DyPP > 30% |
| G2 | G | ⚠️ rejected | 2.5842 | 52% | Ensemble r > 0.7 |
| G3 | G | ⚠️ rejected | 1.2291 | 11% | N=20 < 5% |
| G4 | G | ⚠️ rejected | 0.0459 | 73% | κ predicts restarts |
| B1 | B | ❌ failed | 0.2502 | 12% | ΔE/gap < 5% |
| C3 | C | ❌ failed | 0.1561 | 67% | N=20 VQE < 5% |

### Details

- **A3**: h_min(N) = h_c + alpha * N^(beta) with beta ~ 0.5-1.0 (TFIM universality class, nu=1 in 1D)
- **A3_N20**: Testing speedup
- **B2**: Second-layer HVA params (θ_zz2, θ_x2) are frozen for h>=1.5, enabling 40% VQE cost reduction with <1% accuracy loss.
- **B4**: Hessian analysis at convergence identifies saddle points, enabling escape with 2-3 restarts instead of 5 blind restarts.
- **D1**: ||dW/dh|| peaks near h_c for MPNN trained on full h-range, but at training boundary for valid-regime-only MPNN.
- **F3**: Landscape fluctuation (Var(E)/E_mean^2) drops sharply at h < h_min, providing a training-free predictor of the valid regime boundary.
- **G1**: MPNN achieves ΔE/gap < 5% with 9-11 training points (vs 17 current) if uniformly spaced in [1.0, 2.0].
- **G5**: MPNN trained on seed-42 VQE data achieves ΔE/gap < 5% when deployed with seeds 43/44, proving it learns physics.
- **E4**: HVA p=2 works for g<=0.3 and MPNN generalizes across g with (h, g) as input features.
- **F1**: DyPP linear/quadratic extrapolation reduces VQE iterations by 30-50% in the smooth regime (h>1.5) compared to standard warm-start.
- **G2**: 5-MPNN ensemble variance correlates with ΔE/gap (r>0.7), enabling unreliable prediction detection without VQE.
- **G3**: V8 optimizations (1 restart + freeze) achieve ΔE/gap < 3% at N=20 p=2 in ≤15 min, vs 50 min with V7 config.
- **G4**: κ < 100 → 1 restart sufficient; κ > 500 → 3+ restarts. Enables adaptive restart allocation.
- **B1**: Analytical theta from perturbation theory is within 5% of VQE-optimal at h>=2.0, eliminating seed sensitivity for the first sweep point.
- **C3**: Enforcing θ_x > 0 resolves Z₂ ambiguity at N=20 p=1, enabling MPNN to deploy at all h >= 2.25 (not just h=3.0).
