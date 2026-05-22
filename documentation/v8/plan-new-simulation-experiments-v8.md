# Plan: New Noiseless Simulation Experiments (V8)

> Expanding the experimental portfolio beyond p=1 scaling.
> Focus: new techniques for ground truth, VQE improvement, MPNN enhancement,
> landscape analysis, and scaling demonstrations — all executable locally.
>
> Date: 2026-05-21
> Prerequisites: V6.1 pipeline stable, V7 complete, p=1 scaling validated.

---

## Guiding Principles

1. Every experiment must have a **falsifiable hypothesis**.
2. No experiment duplicates V7 results (check binnacles first).
3. Respect constraints: HVA p<=2, pure energy cost, descending sweep.
4. Target thesis Chapter 4 (Results) and Chapter 5 (Discussion) gaps.
5. Prioritize experiments that produce **new physics insight** or **methodology contribution**.

---

## Category A: Ground Truth Enhancement

### A1. Improved Excited-State Gap via Orthogonal Projection DMRG

**Hypothesis:** Using explicit orthogonal projection against the ground state MPS
in TeNPy's second DMRG run will yield accurate gaps at N>=15, eliminating the
current `gap=0` failure that forces us to use analytical approximations.

**Motivation:** The current `ClassicalSolver._solve_dmrg()` fails to compute gaps
at N>=15 because the excited-state DMRG converges to the ground state. This forces
us to use `gap_approx = max(2|J-h|, 2*pi/N)` which is only accurate far from h_c.
Accurate gaps would let us validate DE/gap at N=20 near the boundary (h~2.0).

**Method:**
- Use TeNPy's `OrthogonalExcitations` or manual projection: add penalty
  `lambda * |psi_0><psi_0|` to the Hamiltonian for the second DMRG run.
- Validate at N=6,10 where exact diag gives the true gap.
- Then apply at N=20 to get precise gaps for h in [1.5, 2.5].

**Expected outcome:** Gaps accurate to <1% at N=20, enabling rigorous DE/gap validation.

**Effort:** Low (TeNPy supports this natively via `OrthogonalExcitations`).

**Thesis value:** Strengthens all N=20 claims by removing the "approximate gap" caveat.

---

### A2. Tensor Cross Interpolation (TCI) for VQE Landscape Mapping

**Hypothesis:** TCI can efficiently reconstruct the full E(theta) landscape of the
4-parameter HVA circuit using O(poly(p)) evaluations instead of exhaustive grid search,
revealing the global structure (number of minima, saddle points, symmetries).

**Motivation:** We know the p=2 landscape has warm-start-friendly structure (smooth,
few local minima in the valid regime). But we've never PROVEN this — only inferred it
from VQE convergence. TCI (a form of active learning for tensor trains) can map the
full 4D landscape with ~100-500 evaluations instead of 10^4 grid points.

**Method:**
- Implement TCI (via `xfac` or custom) on E(theta_zz1, theta_x1, theta_zz2, theta_x2).
- Map landscape at h=1.5 (easy), h=1.25 (boundary), h=1.0 (hard) for N=6.
- Count local minima, measure basin volumes, identify symmetries.
- Compare with p=1 landscape (2D — can be exhaustively mapped).

**Expected outcome:**
- h=1.5: single basin (confirms warm-start works trivially)
- h=1.25: 2-4 basins (explains seed sensitivity)
- h=1.0: many basins (explains VQE failure)

**Effort:** Medium (need TCI library or custom implementation).

**Memory estimate:** 20^4 tensor = 160,000 float64 entries ≈ 1.2 MB (trivial).
TCI intermediate storage: O(rank × 20 × 4) ≈ negligible.

**Thesis value:** HIGH — provides the first rigorous characterization of the HVA
optimization landscape for TFIM. Novel contribution not in the literature.

**References:** Tensor Cross Interpolation (Oseledets 2010), feedback-based learning
with TCI (OpenReview 2025, t5d71yUHlW).

---

### A3. Finite-Size Scaling of the Valid Regime Boundary

**Hypothesis:** The valid regime boundary h_min(N) follows a power law
h_min = h_c + alpha * N^(-beta) that can be extracted from N=6,10,20 data
and used to predict h_min for arbitrary N.

**Motivation:** We have empirical data for p=2: h_min(6)=1.25, h_min(10)=1.50, h_min(20)=2.00.
(Note: for p=1 the boundaries shift: h_min(6)=1.6, h_min(10)=1.9, h_min(20)=2.25.)
The current linear fit (h_min ~ 0.95 + 0.053*N) is ad-hoc. A proper finite-size
scaling analysis would connect this to known TFIM critical exponents (nu=1 for 1D).

**Method:**
- Collect h_min at N=4,6,8,10,14,20 (some need new VQE runs at N=4,8,14).
- NOTE: N=12 excluded — too slow for iterative experimentation on local hardware.
- Fit h_min(N) = 1.0 + a/N^nu with nu as free parameter.
- Compare extracted nu with exact value (nu=1 for 1D TFIM).
- Predict h_min(30), h_min(50) for thesis scaling claims.
- Repeat analysis for p=1 using existing data (N=6,10,20).

**Expected outcome:** nu ~ 0.8-1.2 (consistent with TFIM universality class).
The boundary shift is a finite-size effect of the HVA expressibility, not just
the gap closing.

**Effort:** Low-Medium (N=4,8 are fast; N=14 is ~5 min with statevector).
N=12 skipped per project constraints (>30 min per run).

**Thesis value:** HIGH — connects pipeline performance to known physics (universality).


---

## Category B: VQE Optimization Enhancement

### B1. Analytical Initial Guess from Perturbation Theory

**Hypothesis:** For h >> h_c (deep paramagnetic), the optimal HVA parameters can be
derived analytically from first-order perturbation theory, providing a deterministic
initialization that eliminates seed sensitivity and VQE failures at N=20.

**Motivation:** At large h, the ground state is |+>^N with small ZZ corrections.
The optimal theta_x should be ~pi/2 (rotate from |0> to |+>) and theta_zz should
scale as ~J/h (perturbative ZZ correction). Seed 44's failure at N=20 (binnacle-p1-scaling)
is caused by bad random initialization — an analytical guess would fix this.

**Method:**
- Derive theta_opt(h) in the h>>1 limit using first-order perturbation theory.
- For p=1: theta_x = pi/2 - epsilon(h), theta_zz = J/(2h) (leading order).
- For p=2: extend to second order (4 parameters).
- Validate: compare analytical guess vs VQE-optimized theta at h=2.0, 3.0, 4.0.
- Use as initialization for the first h-point in descending sweep.

**Expected outcome:** Analytical guess within 5% of optimal at h>=2.0.
Eliminates seed sensitivity at N=20 (all seeds converge to same minimum).

**Effort:** Low (pen-and-paper derivation + validation script).

**Thesis value:** Medium-High — demonstrates physics-informed initialization.
Connects to Mele et al. 2022 (parameter transferability in HVA).

---

### B2. TITAN-Style Parameter Freezing for Multi-Layer HVA

**Hypothesis:** In the p=2 HVA, the second-layer parameters (theta_zz2, theta_x2)
contribute less than the first layer at large h. Freezing them after initial
convergence and only optimizing layer 1 reduces VQE cost by ~50% with <1% accuracy loss.

**Motivation:** TITAN (Peng et al. 2025, NeurIPS) uses trajectory analysis to identify
and freeze inactive parameters, achieving 40-60% fewer evaluations. For our 4-parameter
HVA, if 2 parameters are near-frozen in the valid regime, we can halve VQE cost.
This is especially valuable at N=20 where VQE takes 50+ minutes.

**Method:**
- Analyze theta_opt trajectories across h-sweep: compute |d(theta_i)/dh| for each param.
- Identify parameters with |d(theta)/dh| < threshold (frozen).
- Run VQE with frozen params at N=6, N=10, N=20.
- Compare: full 4-param VQE vs 2-param frozen VQE (time, accuracy).

**Expected outcome:** At h>=1.5, theta_zz2 and theta_x2 are nearly constant.
Freezing them saves ~40% VQE time with <0.5% accuracy loss.

**Effort:** Low (analysis of existing VQE data + modified optimizer).

**Thesis value:** Medium — practical speedup for scaling. Validates TITAN principle
on quantum circuits (their paper tests molecular Hamiltonians).

**Reference:** Peng et al. (2025). TITAN: A trajectory-informed technique for
adaptive parameter freezing in large-scale VQE. NeurIPS 2025. arXiv:2509.15193.

---

### B3. Light Cone Cancellation (LCC) for Efficient VQE at Large N

**Hypothesis:** For 1D HVA with local observables, the effective circuit that
contributes to the energy expectation value has a bounded light cone, enabling
polynomial-time exact simulation even at N=30-50.

**Motivation:** LCC (arXiv:2404.19497) shows that for VQE with local cost functions,
only the gates within the backward light cone of the observable contribute.
For 1D TFIM with nearest-neighbor terms and HVA p=2, the light cone is O(p)=O(1)
sites wide per term. This means we can simulate N=50 VQE in polynomial time
by only contracting the relevant sub-circuit per Hamiltonian term.

**Method:**
- Implement LCC-aware energy evaluation: for each ZZ_i term, only simulate
  the sub-circuit within the light cone of qubits i, i+1.
- Benchmark: compare LCC evaluation time vs full statevector at N=10, 20, 30.
- Run full VQE at N=30, N=50 using LCC evaluation.
- Validate accuracy: LCC energy vs full statevector energy at N=10, 20.

**Expected outcome:** LCC is exact (no approximation) and enables N=30-50 VQE
in minutes instead of hours. Scaling: O(N² * 2^(2p+2)) instead of O(2^N).
(Note: O(N) Hamiltonian terms × O(N) light-cone evaluation per term = O(N²) prefactor.)

**Effort:** Medium-High (custom implementation of light cone extraction, ~8h).

**Memory estimate:** Per-term sub-circuit: 2^(2p+2) = 64 amplitudes (negligible).
Total: N terms × 64 complex128 ≈ 50×1KB = 50KB. Memory is NOT the bottleneck.

**Thesis value:** VERY HIGH — enables scaling demonstration at N=50 without MPS.
Novel application of LCC to the GNN-HVA pipeline. Would be a strong thesis contribution.

**Reference:** arXiv:2404.19497 — Light Cone Cancellation for VQE Ansatz (2024).

---

### B4. Landscape-Aware Multi-Start: Hessian-Guided Restart Selection

**Hypothesis:** Computing the Hessian at the converged VQE minimum identifies whether
it's a true minimum (all eigenvalues positive) or a saddle point (negative eigenvalue),
enabling intelligent restart decisions instead of blind multi-start.

**Motivation:** Current multi-start VQE uses 5 random restarts and takes the best.
This is wasteful — most restarts find the same minimum. If we compute the Hessian
(4x4 matrix for p=2, cheap via finite differences), we can:
1. Verify the minimum is genuine (not a saddle).
2. Estimate the basin curvature (flat = hard to converge, sharp = easy).
3. Use negative Hessian eigenvectors as escape directions for new restarts.

**Method:**
- After L-BFGS-B converges, compute 4x4 Hessian via finite differences (8 extra evals).
- If any eigenvalue < 0: follow the negative eigenvector direction for next restart.
- If all eigenvalues > 0: accept the minimum, skip remaining restarts.
- Compare: standard 5-restart vs Hessian-guided adaptive restart at N=6, N=10.

**Expected outcome:** Hessian-guided VQE achieves same accuracy with 2-3 restarts
instead of 5, saving 40-60% VQE time. Particularly valuable at N=20.

**Effort:** Low (finite-difference Hessian is trivial for 4 parameters).

**Thesis value:** Medium — practical optimization improvement with theoretical backing.

---

## Category C: MPNN Predictor Enhancement

### C1. Physics-Informed Loss: Energy Validation During Training

**Hypothesis:** Adding an energy-validation term to the MPNN loss (evaluate predicted
theta on the actual Hamiltonian every K epochs) prevents the MPNN from learning
parameters that have low MSE but high energy error.

**Motivation:** Comparison 4 (binnacle-comparative-analysis) showed MSE is NOT a good
predictor of deployment quality (8 points: lower MSE but worse DE/gap than 17 points).
The MPNN can achieve low MSE by fitting theta patterns that don't minimize energy.
A physics-informed loss would catch this during training.

**Method:**
- Every 100 epochs, evaluate E(theta_pred) for a subset of training points.
- Add penalty: loss = MSE(theta) + lambda * mean(|E(theta_pred) - E_exact|).
- Lambda schedule: start at 0, ramp to 0.1 after 1000 epochs (let MSE converge first).
- Compare: standard MSE-only vs physics-informed at N=6, N=10, N=20.

**Important:** V5.x showed that changing the VQE cost function breaks things.
Here we're NOT changing VQE — we're adding a validation signal to MPNN training.
The theta targets remain pure-energy VQE optima. The energy term is a regularizer,
not a replacement for MSE.

**Expected outcome:** 10-30% improvement in DE/gap at the valid regime boundary
(h=1.25 for N=6, h=1.5 for N=10) where MSE-DE/gap correlation is weakest.

**Effort:** Low-Medium (energy evaluation already exists in pipeline).

**Thesis value:** Medium-High — addresses a known weakness (MSE != deployment quality).
Connects to Miao et al. 2024 (NN-VQE uses energy-aware training).

---

### C2. Unified Hamiltonian+Ansatz Graph Encoding (Qracle-Style)

**Hypothesis:** Encoding both the Hamiltonian structure AND the HVA circuit structure
in a single graph (as in Qracle, Zhang et al. 2025) improves MPNN generalization
across different system sizes and topologies.

**Motivation:** Our current MPNN encodes only the Hamiltonian graph (nodes=qubits,
edges=interactions, features=[h_i, coord_number]). The ansatz structure is implicit.
Qracle (2025) showed that a unified graph encoding (Hamiltonian nodes + circuit gate
nodes + parameter nodes) achieves 64% fewer optimization steps. This encoding could
help our MPNN generalize from N=6 to N=10 (where transfer learning currently fails).

**Method:**
- Construct unified graph: qubit nodes + gate nodes + parameter nodes.
- Qubit nodes: features [h_i, coord_number].
- Gate nodes: features [gate_type (RZZ=0, RX=1), layer_index].
- Parameter nodes: features [param_index, layer].
- Edges: qubit-gate (which qubits the gate acts on), gate-param (which param controls it).
- Train on N=6 data, test on N=10 (cross-size generalization).

**Expected outcome:** Better cross-size generalization than current MPNN (which fails
at transfer learning). May not beat same-size training but provides a unified model.

**Effort:** Medium (new graph construction + GNN architecture modification).

**Thesis value:** Medium — demonstrates awareness of state-of-the-art (Qracle comparison).
If it works for cross-size, it's a novel contribution.

**Reference:** Zhang et al. (2025). Qracle: A GNN-based parameter initializer for VQE.
arXiv:2505.01236. Lee et al. (2026). arXiv:2602.19752.

---

### C3. Sign-Equivariant MPNN for Z2 Symmetry

**Hypothesis:** A sign-equivariant MPNN architecture that respects the Z2 symmetry
of the HVA landscape (theta, -theta give same energy) eliminates the sign
canonicalization problem at N=20 p=1 and improves training stability.

**Motivation:** The p=1 scaling experiments (binnacle-p1-scaling) revealed that
different seeds find theta with different sign conventions due to Z2 symmetry.
The MPNN sees inconsistent targets unless signs are manually canonicalized.
A sign-equivariant architecture would handle this automatically.

**Method:**
- Option A (simple): Canonicalize theta before training (enforce theta_x > 0).
  This breaks equivariance but is trivial to implement.
- Option B (elegant): Use a sign-invariant loss: loss = min(MSE(pred, target),
  MSE(pred, -target)). The MPNN learns one canonical form automatically.
- Option C (full): Predict |theta| and sign separately. The magnitude head uses
  standard regression; the sign head uses the graph structure to determine convention.
- Compare all three at N=20 p=1 with 3 seeds.

**Expected outcome:** Option B is likely sufficient — eliminates the sign issue
with minimal code change. All 3 seeds should give identical deployment results.

**Effort:** Low (Option A/B are trivial; Option C is medium).

**Thesis value:** Medium — solves a practical problem for p=1 scaling.
Demonstrates awareness of symmetry in ML for quantum systems.


---

## Category D: Landscape & Phase Transition Analysis

### D1. Unsupervised Phase Detection from MPNN Weight Space

**Hypothesis:** The trained MPNN's internal weight structure encodes information about
the quantum phase transition that can be extracted WITHOUT measuring quantum observables,
following Hernandes et al. (2025).

**Motivation:** We already have WeightGradientAnalyzer that detects peaks in
||d(weights)/d(h)||. But we haven't done a systematic study: does the peak location
correlate with h_c? Does it shift with N? Can we extract critical exponents?
This is a zero-QPU-cost phase detection method — purely classical.

**Method:**
- Train MPNN at N=6, 10, 20 with dense h-grid (40+ points in valid regime).
- Compute weight gradient norm ||dW/dh|| at each h (already implemented).
- Also compute: Fisher information of weights, singular value spectrum of weight matrices.
- Locate peaks/transitions in these quantities.
- Compare peak location with known h_c(N) = 1.0 (thermodynamic) and finite-size h_c(N).
- Extract scaling: does peak shift follow h_c(N) = 1 + const/N?

**Expected outcome:** Peaks at h ~ 1.7-1.8 (training boundary, not h_c) based on
Analysis A (binnacle-comparative-analysis). BUT: if we train on a symmetric h-grid
centered on h_c (not just the valid regime), peaks might align with h_c.

**Key insight to test:** Train two MPNNs:
- MPNN-A: trained on h in [0.5, 2.0] (includes invalid regime, low-fidelity data)
- MPNN-B: trained on h in [1.0, 2.0] (valid regime only)
Does MPNN-A detect h_c while MPNN-B detects the training boundary?

**Effort:** Low (uses existing infrastructure).

**Thesis value:** HIGH — novel contribution. Zero-QPU phase detection from ML weights.
Extends Hernandes et al. (2025) to the VQE parameter prediction context.

**Reference:** Hernandes et al. (2025). Adiabatic fine-tuning of neural quantum states
enables detection of phase transitions in weight space. arXiv:2503.17140.

---

### D2. Attention-Based Unsupervised Phase Boundary Detection

**Hypothesis:** An attention mechanism (transformer-style) applied to the VQE parameter
trajectories theta(h) can identify phase boundaries in an unsupervised manner,
without explicit observable measurement.

**Motivation:** Li et al. (2026, arXiv:2506.06678) showed that LLM-style attention + VAE
captures hidden correlations in VQE circuit parameters and detects phase transitions
unsupervised. We have rich theta(h) data from 75+ experiments. Can we extract phase
information from the parameter trajectories alone?

**Method:**
- Collect all theta_opt(h) trajectories from N=6 (27 h-points x 3 seeds x multiple configs).
- Treat as a sequence: [theta(h_1), theta(h_2), ..., theta(h_27)].
- Train a small transformer encoder (2 layers, 4 heads) with reconstruction loss.
- Extract attention maps: where does the model "attend" most?
- Hypothesis: attention peaks at h ~ h_c where theta changes most rapidly.
- Also try VAE: encode theta trajectories, examine latent space clustering.

**Expected outcome:** Attention peaks near h=1.0-1.25 (critical region).
Latent space shows two clusters (paramagnetic vs ferromagnetic theta patterns).

**Effort:** Medium (~5h: new architecture, but small model — trains in seconds.
Main time: data collection from existing results + architecture implementation).

**Thesis value:** Medium-High — demonstrates modern ML technique (attention/VAE)
applied to quantum phase detection. Complementary to supervised MPNN approach.

**Reference:** Li et al. (2026). Learning VQC parameters with classical AI for
quantum phase transition detection. arXiv:2506.06678.

---

### D3. VQE Landscape Reconstruction via Low-Rank Tensor Completion

**Hypothesis:** The 4D energy landscape E(theta) of HVA p=2 has low tensor rank
(due to the circuit's local structure), enabling full reconstruction from sparse
VQE evaluations using tensor completion algorithms.

**Motivation:** arXiv:2405.10941 showed that VQA landscapes can be reconstructed
via low-rank tensor completion. If the HVA landscape has rank r << 2^4 = 16,
we can map the full landscape from O(r * 4 * resolution) evaluations instead of
resolution^4. This would provide definitive answers about:
- Number of local minima at each h
- Basin connectivity (can warm-start always reach the global minimum?)
- Landscape flatness vs h (explains why VQE fails near h_c)

**Method:**
- Discretize each theta_i into 20 values in [-pi, pi] -> 20^4 = 160,000 grid.
- Sample 1000-5000 random points, evaluate E(theta) via statevector.
- Apply tensor completion (e.g., ALS, RIEMANNIAN) to reconstruct full tensor.
- Validate: compare reconstructed landscape with dense grid at N=6.
- Analyze: count minima, measure basin volumes, compute landscape entropy.

**Expected outcome:** Rank 4-8 (low rank due to HVA structure). Full landscape
reconstructed from ~2000 evaluations. Reveals that h=1.5 has 1-2 minima while
h=1.0 has 4-8 minima (explaining VQE difficulty).

**Effort:** Medium (tensor completion libraries exist; evaluation is fast at N=6).

**Memory estimate:** Full 20^4 tensor = 160,000 × 8 bytes ≈ 1.2 MB.
Sparse sampling storage: 5000 × (4 indices + 1 value) ≈ 200 KB. Trivial.

**Thesis value:** HIGH — first complete characterization of HVA-TFIM landscape.
Provides rigorous justification for warm-start strategy.

**Reference:** arXiv:2405.10941 — VQA Landscape Reconstruction by Low-Rank
Tensor Completion (2024).

---

## Category E: Scaling & Generalization Demonstrations

### E1. Full Pipeline at N=30 via MPS (p=1)

**Hypothesis:** The p=1 pipeline scales to N=30 with DE/gap < 5% at h >= 2.5,
demonstrating that the framework works at system sizes beyond classical exact
diagonalization (2^30 = 1 billion states).

**Motivation:** N=20 p=1 is validated (38 CX gates, DE/gap < 5% at h>=2.25).
N=30 p=1 would have 58 CX gates — still within IBM Torino's coherence budget.
MPS VQE should work (chi=64 sufficient for 1D, validated in V7 3A/3B).

**Method:**
- Phase 1: DMRG ground truth at N=30 (TeNPy, chi=128, ~60s/point).
- Phase 2: MPS VQE with L-BFGS-B, 5 restarts, h in [2.5, 4.0] (8 points).
- Phase 3: MPNN training (2 parameters, sign-canonicalized).
- Phase 4: Deploy at h_test = 3.0.
- Use analytical initial guess (B1) to avoid seed sensitivity.

**Expected outcome:** DE/gap < 5% at h >= 2.5. Pipeline time ~30-60 min total.
Demonstrates scaling to a regime where exact diagonalization is infeasible
(2^30 ≈ 10^9 states). Note: MPS still solves this classically — the claim is
"beyond exact diag" not "beyond all classical methods."

**Effort:** Medium (MPS VQE is slow but validated; main risk is convergence).

**Thesis value:** VERY HIGH — strongest scaling claim in the thesis.
N=30 with 2^30 Hilbert space dimension is clearly beyond brute-force classical.

---

### E2. Topology Generalization: Star Graph and Ring with Defect

**Hypothesis:** The MPNN trained on chain_1d topology can predict theta for
topologically different graphs (star, ring with defect) without retraining,
demonstrating true lattice-agnostic generalization.

**Motivation:** The thesis claims "lattice-agnostic architecture." We've tested
chain_1d and ladder (ladder fails due to HVA expressibility, not MPNN).
Testing on topologies where HVA p=2 CAN express the ground state would validate
the MPNN's generalization capability.

**Method:**
- Star graph (N=6): central qubit connected to 5 peripheral qubits.
  H = -J * sum(Z_0 Z_i) - h * sum(X_i). HVA should work (low entanglement).
- Ring with defect (N=6): periodic chain with one weak bond (J'=0.5).
  Tests edge-feature generalization (NNConv).
- Open boundary vs periodic boundary (N=6): same chain, different boundary conditions.
- Train MPNN on chain_1d, test on star/ring/periodic (zero-shot transfer).
- Also: train on star, test on chain (reverse transfer).

**Expected outcome:**
- Star: HVA p=2 should work (star has low entanglement). MPNN transfer uncertain.
- Ring with defect: NNConv should handle (edge features encode J').
- Periodic: should be easy (same local structure as open chain).

**Effort:** Low-Medium (~4h: new lattice definitions + VQE runs at N=6 for 3 topologies).

**Thesis value:** Medium — validates "lattice-agnostic" claim beyond chain/ladder.

---

### E3. Data Efficiency: Active Learning for Optimal h-Grid Selection

**Hypothesis:** An active learning strategy that selects the next h-point based on
MPNN prediction uncertainty reduces the number of VQE runs needed by 30-50%
while maintaining DE/gap < 5%.

**Motivation:** Comparison 4 showed 17 points are needed for good predictions.
But which 17 points? Currently we use a fixed grid. Active learning would:
1. Start with 5 points (endpoints + 3 random).
2. Train MPNN, identify h-values with highest prediction uncertainty.
3. Run VQE at those h-values, add to training set.
4. Repeat until DE/gap < 5% at test point.

**Method:**
- Uncertainty estimation: train ensemble of 5 MPNNs, use prediction variance.
- Acquisition function: select h with max variance (exploration) or max
  expected improvement (exploitation).
- Stopping criterion: ensemble variance < threshold at all test points.
- Compare: active learning (adaptive) vs fixed grid (uniform) vs random.
- Metric: number of VQE runs to achieve DE/gap < 5%.

**Expected outcome:** Active learning achieves DE/gap < 5% with 10-12 points
instead of 17 (30-40% reduction). Points concentrate near the valid regime boundary.

**Effort:** Medium (ensemble training + acquisition loop).

**Thesis value:** HIGH — demonstrates practical data efficiency improvement.
Connects to Miao et al. 2024 (active learning for NN-VQE).

**Reference:** Miao et al. (2024). Neural-network-encoded VQA. PRApplied 21, 014053.

---

### E4. Cross-Model Generalization: TFIM with Longitudinal Field

**Hypothesis:** The MPNN trained on the standard TFIM (H = -J*ZZ - h*X) can
predict parameters for the TFIM with longitudinal field (H = -J*ZZ - h*X - g*Z)
with minimal fine-tuning, demonstrating model-agnostic generalization.

**Motivation:** The Heisenberg extension (binnacle-heisenberg-extension) showed HVA p=2
is insufficient for Heisenberg. But TFIM + longitudinal field is a PERTURBATION of
our base model — HVA p=2 should still work for small g. This tests whether the MPNN
can generalize across a continuous family of Hamiltonians.

**Method:**
- Add longitudinal field: H = -J*sum(ZZ) - h*sum(X) - g*sum(Z), g in [0, 0.5].
- Phase diagram: g breaks Z2 symmetry, crossover (not transition) at h_c(g).
- Run VQE sweep at g=0 (baseline), g=0.1, g=0.2, g=0.5.
- Train MPNN on (h, g) -> theta (add g as node feature).
- Test: predict theta at unseen (h, g) combinations.

**Expected outcome:** HVA p=2 works for g <= 0.3 (small perturbation).
MPNN generalizes across g with 2D input features.
At g=0.5, the crossover is smeared and HVA may struggle.

**Effort:** Low-Medium (small modification to HamiltonianBuilder + VQE sweep).

**Thesis value:** HIGH — demonstrates the framework handles a 2-parameter phase diagram.
Goes beyond single-parameter (h-only) sweeps. Novel for the GNN-VQE literature.

---

## Category F: Novel Methodological Contributions

### F1. Dynamic Parameter Prediction (DyPP) for VQE Acceleration

**Hypothesis:** During the descending h-sweep, the theta trajectory is smooth enough
that extrapolation from the last 2-3 converged points predicts the next theta_opt
with sufficient accuracy to replace the MPNN entirely for adjacent h-values.

**Motivation:** DyPP (arXiv:2307.12449) exploits regular trends in VQE parameters
to predict future values. In our descending sweep, theta(h) is smooth — we could
predict theta(h_{i+1}) from theta(h_i) and theta(h_{i-1}) using linear/quadratic
extrapolation, eliminating the need for MPNN at intermediate points.

**Method:**
- During Phase 2 descending sweep, after converging at h_i and h_{i-1}:
  - Linear prediction: theta_pred(h_{i+1}) = theta(h_i) + (theta(h_i) - theta(h_{i-1}))
  - Quadratic: use 3 previous points for parabolic extrapolation.
- Use prediction as VQE initial guess (replaces random restart).
- Compare: standard warm-start (previous h only) vs DyPP (extrapolation from 2-3 points).
- Metric: VQE iterations to converge, final energy accuracy.

**Expected outcome:** DyPP reduces VQE iterations by 30-50% in the smooth regime (h>1.5).
Near h_c, extrapolation fails (theta changes non-linearly) — DyPP should detect this
and fall back to standard warm-start.

**Effort:** Low (~2h: simple extrapolation logic in VQE sweep).

**Thesis value:** Medium — practical VQE acceleration. Validates DyPP principle
on quantum spin systems (original paper tests molecular systems).

**Reference:** arXiv:2307.12449 — Dynamic Parameter Prediction for VQA (2023).

---

### F2. Generative Flow Warm-Start Comparison (Flow-VQE Benchmark)

**Hypothesis:** A normalizing flow trained on theta_opt(h) data provides comparable
or better warm-start quality than our deterministic MPNN, with the added benefit
of uncertainty quantification via the flow's likelihood.

**Motivation:** Flow-VQE (Zou et al. 2026, npj QI) uses normalizing flows for VQE
warm-start, achieving up to 50x acceleration. We chose deterministic MPNN for
simplicity, but a flow model could:
1. Provide uncertainty estimates (high likelihood = confident prediction).
2. Generate multiple candidate initializations (sample from the flow).
3. Detect out-of-distribution h-values (low likelihood = unreliable prediction).

**Method:**
- Implement a simple RealNVP flow: 4D (theta) conditioned on h.
- Train on same data as MPNN (17-27 h-points, fidelity-filtered).
- Compare: MPNN prediction vs flow mean vs flow samples (best of 5).
- Evaluate: DE/gap, calibration (does likelihood correlate with accuracy?).
- Test OOD detection: does the flow assign low likelihood to h < h_min?

**Expected outcome:** Flow achieves similar DE/gap to MPNN (both are ceiling-limited
by HVA expressibility). The added value is uncertainty quantification — the flow
"knows" when it's extrapolating. This could replace the fidelity filter.

**Effort:** Medium (~5h: normalizing flow implementation ~100 lines of PyTorch,
plus training, evaluation, and OOD analysis).

**Thesis value:** Medium-High — direct comparison with state-of-the-art (Flow-VQE).
Demonstrates awareness of generative alternatives. Uncertainty quantification is novel.

**Reference:** Zou et al. (2026). Generative flow-based warm start of VQE.
npj Quantum Information 12, 5. arXiv:2507.01726.

---

### F3. Scalable QAS: Landscape Fluctuation Analysis for Circuit Selection

**Hypothesis:** The landscape fluctuation metric (variance of energy over random
parameter samples) can predict HVA circuit quality WITHOUT running VQE, enabling
automated selection of optimal circuit depth and structure.

**Motivation:** arXiv:2505.05380 introduces a training-free quantum architecture search
(QAS) that uses landscape fluctuation to predict circuit learnability. We could use
this to:
1. Confirm p=2 is optimal (vs p=1, p=3) without running full VQE.
2. Predict the valid regime boundary from landscape properties alone.
3. Select optimal initial_layout for hardware deployment.

**Method:**
- For each (N, p, h): sample 100 random theta, compute E(theta) for each.
- Landscape fluctuation = Var(E) / |E_mean|^2.
- High fluctuation = trainable (good landscape). Low fluctuation = flat (barren plateau).
- Map fluctuation vs h for p=1, p=2, p=3 at N=6, 10, 20.
- Correlate with actual VQE success (DE/gap < 5%).

**Expected outcome:** Fluctuation drops sharply at h < h_min (landscape becomes flat
in the ferromagnetic phase where HVA can't express the GS). This provides a
training-free predictor of the valid regime boundary.

**Effort:** Low (100 random evaluations per configuration — seconds).

**Thesis value:** Medium — provides theoretical justification for valid regime boundary.
Connects to barren plateau literature (Cerezo et al. 2021).

**Reference:** arXiv:2505.05380 — Scalable QAS via Landscape Analysis (2025).

---

## Abort Criteria (when to stop an experiment)

Each experiment should be abandoned if ANY of the following occur:

1. **VQE non-convergence:** >50% of h-points fail to converge after max restarts.
   → Indicates the regime is invalid for HVA. Document and move on.
2. **Wall time exceeded:** >3× the estimated time without meaningful progress.
   → Likely an implementation bug or wrong parameter regime.
3. **Numerical instability:** NaN/Inf in energies or gradients.
   → Check Hamiltonian construction and circuit parameters.
4. **Negative result confirmed:** If the first 3 seeds all show the technique
   doesn't help (e.g., no improvement over baseline), stop at 3 seeds.
   Document as validated rejection.
5. **Memory exceeded:** If a single evaluation requires >16GB RAM, the experiment
   is not locally executable. Redesign or exclude.

---

## Experiment Dependency Graph

```
B1 (analytical init) ──→ E1 (N=30 pipeline, uses B1 for initialization)
C3 (sign canon.) ──────→ E1 (N=30 pipeline, needs sign-consistent data)
A3 (scaling law) ──────→ (standalone, informs thesis claims)
F3 (fluctuation) ──────→ (standalone, informs valid regime theory)
D1 (weight space) ─────→ (standalone)
B4 (Hessian) ──────────→ B2 (freezing, uses Hessian to identify flat params)
E4 (longitudinal) ─────→ (standalone, needs HamiltonianBuilder extension)
E3 (active learning) ──→ (standalone, benefits from C1's physics loss)
C1 (physics loss) ─────→ (standalone)
F1 (DyPP) ─────────────→ (standalone)
A2 (TCI) ──────────────→ D3 (tensor completion, similar methodology)
B3 (LCC) ──────────────→ E1 (alternative to MPS for N=30-50)
```

---

## Execution Priority & Timeline

### Tier 1: High Impact, Low Effort (do first)

| ID | Experiment | Time | Thesis Section |
|----|-----------|------|----------------|
| B1 | Analytical initial guess | 2h | 3.3 (Methodology) |
| C3-A | Sign canonicalization (simple) | 1h | 4.6 (p=1 results) |
| A3 | Finite-size scaling law | 4h | 4.4 (Physics limits) |
| F3 | Landscape fluctuation analysis | 2h | 4.4 (Physics limits) |
| D1 | Weight space phase detection | 3h | 5.1 (Discussion) |

### Tier 2: High Impact, Medium Effort (do second)

| ID | Experiment | Time | Thesis Section |
|----|-----------|------|----------------|
| A2 | TCI landscape mapping | 6h | 4.4 + novel contribution |
| B3 | Light Cone Cancellation N=30-50 | 8h | 4.3 (Scaling) |
| E1 | Full pipeline N=30 p=1 | 4h | 4.3 (Scaling) |
| E3 | Active learning h-grid | 4h | 3.4 (Methodology) |
| E4 | TFIM + longitudinal field | 4h | 5.5 (Generalization) |

### Tier 3: Medium Impact, Variable Effort (if time permits)

| ID | Experiment | Time | Thesis Section |
|----|-----------|------|----------------|
| A1 | Orthogonal projection DMRG | 3h | 3.2 (Ground truth) |
| B2 | TITAN parameter freezing | 3h | 3.3 (VQE optimization) |
| B4 | Hessian-guided restarts | 3h | 3.3 (VQE optimization) |
| C1 | Physics-informed MPNN loss | 4h | 3.4 (MPNN training) |
| C2 | Qracle-style unified graph | 6h | 5.2 (Literature comparison) |
| D2 | Attention-based phase detection | 5h | 5.1 (Discussion) |
| D3 | Tensor completion landscape | 6h | 4.4 (novel contribution) |
| E2 | Topology generalization | 4h | 5.5 (Generalization) |
| F1 | DyPP extrapolation | 2h | 3.3 (VQE optimization) |
| F2 | Flow-VQE comparison | 5h | 5.2 (Literature comparison) |

---

## Experiments NOT Proposed (and why)

| Technique | Why excluded |
|-----------|-------------|
| p=3 HVA | Violates Mele et al. depth constraint |
| ADAPT-VQE > 2 iterations | Violates ADAPT constraint |
| Quantum Natural Gradient | Overkill for 4 parameters (V7 already settled optimizer) |
| Transfer learning N->N' | Definitively rejected in V7 (7% worse) |
| Noise-aware training (shot noise) | Definitively rejected in V7 5B (6x worse) |
| Nevergrad optimizers | Definitively rejected in V7 1A (31-95% worse) |
| Data augmentation (interpolation) | Rejected at N=10 (hurts, binnacle-N10) |
| GATConv architecture | Rejected at N=6 (adds instability, binnacle-N6) |
| Hybrid cost function | Catastrophically failed in V5.x |
| Angle wrapping | Creates discontinuities (V5.1 lesson) |

---

## New Bibliography Entries (to add if experiments are executed)

| Paper | Relevance | Experiment |
|-------|-----------|-----------|
| Oseledets (2010). Tensor-train decomposition. SIAM J. Sci. Comput. | TCI foundation | A2 |
| arXiv:2404.19497 — Light Cone Cancellation for VQE (2024) | LCC method | B3 |
| arXiv:2405.10941 — VQA Landscape by Tensor Completion (2024) | Landscape reconstruction | D3 |
| arXiv:2505.05380 — Scalable QAS via Landscape Analysis (2025) | Fluctuation metric | F3 |
| arXiv:2307.12449 — DyPP for VQA (2023) | Parameter prediction | F1 |
| OpenReview t5d71yUHlW — TCI active learning (2025) | TCI for quantum | A2 |
| Peng et al. (2025). TITAN. NeurIPS. arXiv:2509.15193 | Parameter freezing | B2 |
| Li et al. (2026). Learning VQC parameters with classical AI. arXiv:2506.06678 | Attention-based phase detection | D2 |
| Zou et al. (2026). Generative flow-based warm start of VQE. npj QI 12, 5 | Flow warm-start comparison | F2 |

---

## Success Criteria

An experiment is considered successful if it produces ONE of:
1. **New physics insight** (e.g., landscape structure, scaling law, symmetry)
2. **Practical improvement** (>10% reduction in time or error)
3. **Validated rejection** (technique X doesn't work because Y — closes a question)
4. **Novel methodology** (new approach not in the VQE/GNN literature)

An experiment is considered a FAILURE only if it produces no learning
(e.g., confirms what we already know without new nuance).

---

## Relationship to Existing Work

| This plan | Extends | Does NOT duplicate |
|-----------|---------|-------------------|
| A2 (TCI landscape) | V7 landscape intuitions | V7 tested optimizers, not landscape structure |
| B3 (LCC) | V7 3C (MPS at N=20) | Different scaling mechanism (exact, not approximate) |
| C1 (physics loss) | V5.x lessons | V5.x changed VQE cost; C1 adds MPNN regularizer |
| D1 (weight space) | Existing WeightGradientAnalyzer | Systematic study vs single-point detection |
| E1 (N=30) | V7 3C (N=20 MPS) | Larger N, p=1, full pipeline |
| E4 (longitudinal field) | Heisenberg extension | Different model (perturbation vs new symmetry) |
| F2 (Flow-VQE) | Current MPNN | Different generative paradigm |
