# Literature Review: NN-Based Noise-Aware VQE Parameter Prediction (2024–2026)

**Date**: 2026-08-03
**Scope**: How neural networks can learn the relationship between hardware noise (T1, T2, gate errors, readout) and optimal VQE circuit parameters. Focus on approaches that use calibration data as NN input features.

---

## 1. The Physics: How Noise Shifts θ_opt

### 1.1 Coherent Errors → Deterministic, Absorbable Shift

**Chen et al. (2025)** — "Exploiting biased noise in variational quantum models" [arXiv:2510.24050](https://arxiv.org/abs/2510.24050)

Key finding: coherent gate over-rotations can be *fully absorbed by reparametrization*. For a parametric gate R(θ) followed by a coherent error R(ε), the combined effect is simply R(θ+ε). The VQE optimizer naturally compensates by finding θ_opt(noisy) = θ_opt(noiseless) − ε. This shift is deterministic given the gate calibration and in principle doesn't need learning — it's a computable correction.

**Implication**: For the coherent component of noise, a direct analytical correction (θ_corrected = θ_noiseless − gate_over_rotation) may outperform any NN. The NN's value lies in handling the *interaction* between coherent errors across multiple gates and layers, where the correction is no longer simply additive.

### 1.2 Incoherent Noise → Phase Transition in θ_opt

**Li & Hernandez (2024, Fermilab)** — "Noise-induced transition in optimal solutions of variational quantum algorithms" [arXiv:2403.02762](https://arxiv.org/abs/2403.02762)

Critical result: for a TFIM spin chain VQE, there exists a **critical noise rate** p_c above which θ_opt jumps discontinuously to a trivial configuration (θ → 0, producing the maximally mixed state). Below p_c, the shift is smooth and small. Above p_c, VQE effectively gives up on entanglement.

This explains our F18 result: near h_c where the ground state is maximally entangled, the noise rate crosses p_c on FakeTorino, producing the scattered θ we observed. The landscape doesn't just shift — it undergoes a phase transition.

**Implication**: Any NN trying to learn θ_opt(noisy) will face a discontinuity at p_c. A two-regime model (below/above transition) would be needed, but detecting p_c requires knowing the noise level relative to the circuit's entanglement structure.

### 1.3 Analytical Bounds on Parameter Perturbation

**Legnini & Berberich (2026)** — "Noise Resilience and Robust Convergence Guarantees for the VQE" [arXiv:2601.16758](https://arxiv.org/abs/2601.16758)

First analytical upper bound on ‖θ_opt(noisy) − θ_opt(noiseless)‖. They show:
- Coherent noise (over-rotation ε): Δθ ∝ ε × (circuit structure factor)
- Incoherent noise (depolarizing p): Δθ ∝ p × (spectral gap)⁻¹
- The bound tightens when the spectral gap is large (paramagnetic regime)

Provides theoretical justification that in the paramagnetic regime (h >> h_c), noise corrections to θ are small and smooth. Near criticality, the (gap)⁻¹ factor diverges → corrections become large and potentially discontinuous.

---

## 2. NN Architectures That Use Noise Features (Most Relevant to Our Project)

### 2.1 ★ Scalable QEM with Physically-Informed GNNs (2026)

[arXiv:2604.16815](https://arxiv.org/abs/2604.16815)

**Architecture** (directly comparable to our GNNQEMCorrector):
- **Node features**: T1, T2, readout errors (calibration parameters per qubit)
- **Edge features**: 2-qubit gate errors (on physical coupling map edges)
- GNN message-passing learns how errors propagate along the device topology
- Predicts correction to mitigate noise in expectation values

**Key results**:
- Outperforms ZNE at larger circuit depths
- Generalizes across different calibration snapshots (handles drift)
- The physically-informed encoding (noise at nodes/edges matching hardware graph) is critical

**Comparison with our GNNQEMCorrector**: Almost identical feature encoding ([T1/100, T2/100, readout_err, gate_err] at nodes). Difference: they use per-edge 2Q gate errors (we average to nodes). Their architecture is more explicitly tied to the hardware coupling map.

### 2.2 ★ GNN Output Prediction of Quantum Circuits (2025)

[arXiv:2504.00464](https://arxiv.org/abs/2504.00464)

Proposes GNNs to predict output expectation values of parameterized circuits under noise. Node feature vectors are specifically designed to include noise information. Compares performance against CNNs on the same dataset.

**Key finding**: GNNs with noise node features outperform CNNs and scale better with qubit count. The graph representation captures the local-to-global error propagation that flat architectures miss.

**Relevance**: Validates that (T1, T2, gate_error) as GNN node features is a sound approach for noise-aware prediction. However, they predict *energy* (scalar), not *θ* (vector). Predicting θ from noise features is harder because θ is high-dimensional and noise affects each parameter differently depending on its position in the circuit.

### 2.3 ★ GNN Forensic Framework for Backend Noise Inference (2024)

[arXiv:2512.14541](https://arxiv.org/abs/2512.14541) — Das, Ghosh & Ghosh (Penn State)

GNN-based framework that predicts per-qubit and per-qubit-link error rates of an *unseen* backend from topology + transpiled circuit statistics. Trained on IBM 27-qubit devices.

**Key results**:
- Predicts error rates with ~22% mismatch for 1Q errors, ~18% for 2Q link errors
- Robust under temporal noise drift
- High Spearman correlation with true calibration ordering

**Relevance**: Demonstrates that GNNs can infer noise structure from circuit behavior. In our context, this suggests a reverse architecture: if we *know* the noise (from FakeBackend calibration) and the circuit structure, we can predict how the noise will modify the optimal parameters.

### 2.4 ★ QAGT-MLP: Attention Graph Transformer for QEM (2025)

[arXiv:2511.03119](https://arxiv.org/abs/2511.03119)

Encodes quantum circuits as graphs (nodes = gate instances, edges = qubit connectivity + causal adjacency). Uses attention mechanism for global dependencies. Concatenates learned graph representations with circuit-level descriptor features and noisy expected values, then passes to MLP to predict noise-mitigated values.

**Key architecture insight**: Separates *circuit structure encoding* (via graph transformer) from *noise conditioning* (via descriptor features). This two-stage design could be adapted: Stage 1 encodes the Hamiltonian, Stage 2 conditions on noise.

### 2.5 GTranQEM: Non-Message-Passing Graph Transformer (2025)

[OpenReview/NeurIPS 2025](https://openreview.net/forum?id=XnVttczoAV)

Uses quantum-specific positional encoding + structure matrix as attention bias. Virtual "quantum-representative" node captures global entanglement. Outperforms all message-passing GNN baselines across noise types and circuit scales.

**Relevance**: Shows that for QEM, attention-based architectures may outperform standard MPNN message-passing (which is what we currently use). If we build a noise-correction module, a transformer layer might capture long-range noise correlations better.

---

## 3. ML Approaches to Noise-Aware VQE Optimization

### 3.1 ★ Karim et al. (2025) — Fast & Noise-Aware ML VQE Optimiser

[arXiv:2503.20210](https://arxiv.org/abs/2503.20210) — CSIRO/Melbourne

**Approach**: Train NN on intermediate (θ_i, E_i) pairs from VQE optimization trajectories. The NN learns to predict θ_final from early measurements. Tested on IBM hardware (1-4 qubits).

**Key finding on noise**: Training the NN on data from devices with *different* gate setting errors allows it to generalize to devices with arbitrary coherent errors. The NN implicitly learns the noise → θ correction without explicit noise features as input.

**Limitation**: Only small molecules (H₂, H₃, HeH⁺) up to 4 qubits. Does NOT use explicit noise features (T1/T2/etc.). Whether implicit learning works at N=10+ with 19+ parameters is untested.

**Comparison with us**: Our approach is fundamentally different. We predict θ from the Hamiltonian graph (not from VQE trajectories). Karim needs a partial VQE run first; we predict before any QPU execution. Complementary rather than competing.

### 3.2 Cantori & Pilati (2024) — Synergy: Noisy Quantum + Classical Deep Learning

[arXiv:2404.07802](https://arxiv.org/abs/2404.07802) — Published in EPJ Quantum Technology

**Approach**: CNNs trained on (circuit descriptors θ, noisy expectation values E_noisy) → predict E_exact. Tested on Ising Trotter circuits.

**Key findings**:
- CNNs with noisy quantum data + classical features **outperform ZNE** 
- Transfer learning enables predictions for larger circuits than training set
- The "crossover regime" exists: at moderate noise, quantum data helps. At very high noise, it becomes useless.
- Classical-only learning (no quantum data) fails for certain circuit classes where the quantum information is essential

**Relevance for us**: Confirms the paradigm: use *noiseless* features for prediction, then apply a noise-conditioned correction layer. The correction layer needs access to noisy measurement data to be useful. In our pipeline: MPNN predicts θ_noiseless → circuit executed on noisy backend → correction NN maps (θ_pred, E_noisy, calibration) → Δθ correction.

### 3.3 Deep-Learned Error Mitigation via Partially Knitted Circuits (2025)

[arXiv:2506.04146](https://arxiv.org/abs/2506.04146)

MLPs trained *on the fly* during VQE to predict ideal expectation values from noisy outputs + circuit descriptors. Uses circuit knitting to generate training labels efficiently.

**Key insight**: The NN doesn't need pre-training on external data. It learns the noise correction during the VQE loop itself. However, this requires running many auxiliary circuits for training data, adding overhead.

### 3.4 NN-Enhanced ZNE (2024)

[arXiv:2403.07025](https://arxiv.org/abs/2403.07025)

Feed-forward NN trained on *error probabilities* and their associated expectation values to predict the zero-noise limit. Replaces polynomial extrapolation in ZNE with a learned function.

**Input features to NN**: Error probability levels (depolarizing rates) + measured expectation values at those rates.

**Relevance**: Shows that using noise parameters directly as NN inputs works for predicting noiseless outcomes. The architecture is simple (feedforward) but the principle applies: noise rates are informative features.

### 3.5 Noise-Mitigated VQE with Pre-training and ZNE (2025, IEEE)

[arXiv:2501.01646](https://arxiv.org/abs/2501.01646)

Combines MPS-inspired circuit pre-training (for parameter initialization) with ZNE+neural network for noise fitting. The NN improves accuracy of the ZNE extrapolation function.

**Key insight**: Pre-training parameters from MPS structure (similar to our MPNN warm-start) + NN noise correction = better than either alone.

### 3.6 Meta-Learning for Quantum Control (2026, ICML)

[arXiv:2601.18973](https://arxiv.org/abs/2601.18973) — Leclerc, Miller & Brawand

Derives scaling laws for when *adaptation* (device-specific fine-tuning) beats a robust fixed controller. Shows adaptation gain scales linearly with task variance (how much noise varies between devices) and saturates exponentially with gradient steps.

**Relevance**: Provides a framework for deciding whether noise-adaptive parameter correction is worth the overhead. If device-to-device variance in θ_opt is small (our finding F2: coherent shift too small), adaptation doesn't justify its cost. But if we consider day-to-day drift on the same device, the variance may be larger.

---

## 4. IBM's PNA: The Industrial Approach (2025–2026)

**Propagated Noise Absorption** ([qiskit-addon-pna](https://qiskit.github.io/qiskit-addon-pna/))

IBM's production approach to noise-aware correction:
- Uses learned Pauli-Lindblad noise model (from noise learning)
- Classically propagates inverse noise channels through the circuit
- Applies the correction to the *observable* rather than the *parameters*
- No NN needed — analytical propagation given a fitted noise model

**Key difference from NN approach**: PNA corrects the measurement/observable, not the circuit parameters. This sidesteps the θ_opt shift problem entirely. The question for us becomes: is correcting θ (our approach) or correcting the observable (PNA) more effective for VQE warm-start?

---

## 5. Approaches That Use FakeBackend / Calibration Data Directly

### 5.1 Data-Efficient Quantum Noise Modeling (2025, Fraunhofer)

[arXiv:2509.12933](https://arxiv.org/abs/2509.12933)

ML framework that constructs parameterized noise models from measurement data. Achieves 65% improvement in prediction fidelity (Hellinger distance) over standard calibration-derived models. Trains on small circuits, predicts larger ones.

**Relevance**: Standard noise models (from T1/T2/gate_error) are insufficient. Learned models capture crosstalk, drift, and non-Markovian effects. If we use T1/T2/gate_error as NN features, we're limited by the same insufficiency. Consider adding *derived* features (CES, effective error per layer, etc.) or training the GNN to learn effective noise from raw data.

### 5.2 ML-QEM at Scale (2023, IBM — Liao, Wang, Minev)

[arXiv:2309.17368](https://arxiv.org/abs/2309.17368) — Up to 100 qubits

Benchmarks ML models (linear regression, random forests, MLP, **GNN**) for QEM. GNNs achieve best performance at scale. Demonstrates ML can *mimic* ZNE with drastically reduced runtime.

**Key finding for us**: GNNs are the architecture of choice for noise-aware prediction at scale. Linear models and random forests fall off after ~20 qubits. GNNs maintain quality because they encode the hardware topology.

---

## 6. Comparative Assessment: What Works vs. What We Tried

| Approach | Predicts | Uses Noise Features | Scale Tested | Result |
|----------|----------|:-------------------:|:------------:|--------|
| **Our MPNN (noiseless)** | θ_opt from H-graph | ❌ | N=4–20, 19 params | ✅ Works (100% pass h>2.0) |
| **Our noise-aware MPNN (F18)** | θ_opt from H-graph (trained on noisy θ) | ❌ (implicit via training data) | N=6–10 | ❌ Failed (2-14× worse) |
| **Our GNN-QEM** | ΔE correction | ✅ T1,T2,readout,gate_err | N=6–10 | ✅ Works for energy correction |
| Karim et al. | θ_final from trajectory | ❌ (implicit) | N=1–4 | ✅ At small scale |
| arXiv:2604.16815 | E_corrected | ✅ T1,T2,readout + edge 2Q | Up to 127 qubits | ✅ Outperforms ZNE |
| arXiv:2504.00464 | E under noise | ✅ noise node features | Scalable | ✅ Outperforms CNNs |
| Cantori & Pilati | E_exact from (θ, E_noisy) | Implicitly (via E_noisy) | N=4–16 (transfer) | ✅ Outperforms ZNE |
| PNA (IBM) | Corrected observable | Uses learned noise model | Production (Eagle/Heron) | ✅ Production-grade |

### Key insight from comparison:

**What works**: Using noise features to correct *energy/observable* (scalar output).
**What doesn't**: Using noise features (or noisy training data) to correct *θ* directly (vector output).

The fundamental asymmetry: E is a smooth function of noise parameters (monotone degradation). θ_opt is NOT — it can jump discontinuously (Li & Hernandez) or be trivially shifted (Chen et al.). There's no "middle ground" where a smooth NN can learn the correction profitably.

---

## 7. Viable Paths Forward (Ranked by Feasibility)

### Path A: Two-Stage Correction (Energy, not θ)

**Architecture**: MPNN predicts θ_noiseless → Execute on FakeBackend → GNN-QEM corrects E_noisy → E_corrected

This is what our existing GNN-QEM already does. The path forward is to improve it, not to replace it with θ-correction. Improvements:
1. Per-edge 2Q gate errors (not averaged to nodes) — per arXiv:2604.16815
2. Add circuit depth features (n_layers, n_2Q_gates per qubit)
3. Train across multiple FakeBackend snapshots for drift robustness

**Feasibility**: HIGH — leverages existing code, validated by 3 papers.

### Path B: Noise-Adaptive VQE Warm-Start (Hybrid)

**Architecture**: MPNN predicts θ_noiseless → Apply small analytical correction based on known gate over-rotations → Use as VQE warm-start on noisy device → Few SPSA iterations to fine-tune

This leverages Chen et al.'s reparametrization insight. The correction is:
```
θ_corrected[i] = θ_noiseless[i] - Σ_gates(ε_gate_i)
```
where ε is the coherent over-rotation of each gate affecting parameter i.

**Feasibility**: MEDIUM — requires extracting per-gate coherent error from FakeBackend calibration (available in the noise model).


### Path C: Noise-Conditioned GNN for θ Correction (Research)

**Architecture**: A second GNN takes as input the hardware graph (with T1/T2/gate_err) AND the predicted θ_noiseless, and outputs Δθ correction. Trained on pairs (θ_noiseless, θ_noisy_converged) from multiple noise configurations.

This is the most ambitious and requires:
- Large dataset of (noise_config, θ_noiseless, θ_noisy) triples
- Multiple FakeBackend configurations with varying noise levels
- Careful handling of the discontinuity (Li & Hernandez) — restrict to h >> h_c where corrections are smooth

**Feasibility**: LOW — our F18/F2 results suggest the learnable signal is too small in the smooth regime and discontinuous near criticality.

### Path D: Meta-Learning Adaptive Fine-Tuning

Per Leclerc et al. (2026): use MAML-style meta-learning where the base model is our MPNN, and a few gradient steps on device-specific data adapts θ_pred to the current calibration. The meta-learner learns *how to adapt* from calibration data.

**Feasibility**: MEDIUM-LOW — requires framework changes and extensive multi-device data.

---

## 8. Noise Feature Utility Ranking

Based on the literature evidence:

| Feature | Usefulness for θ correction | Usefulness for E correction (QEM) | Evidence |
|---------|:---------------------------:|:---------------------------------:|----------|
| **2Q gate error (per edge)** | ⭐⭐ Medium | ⭐⭐⭐ High | 2604.16815, 2504.00464 |
| **T1** | ⭐ Low-Med | ⭐⭐⭐ High | 2604.16815, 2509.12933 |
| **T2** | ⭐ Low-Med | ⭐⭐ Medium | Same |
| **Readout error** | ⭐ Low | ⭐⭐⭐ High | Directly correctable analytically |
| **1Q gate error** | ⭐ Negligible | ⭐ Low | Typical rates 0.01-0.1% |
| **CES (product of 1-err)** | ⭐⭐ Medium | ⭐⭐⭐ High | Our GNN-QEM validated |
| **n_2Q_gates (depth proxy)** | ⭐⭐ Medium | ⭐⭐⭐ High | All papers agree |
| **Circuit structure (graph)** | ⭐⭐⭐ High | ⭐⭐⭐ High | GNN papers unanimously |

### Features to ADD that we're missing (from recent papers):

1. **Per-edge 2Q gate error** (currently averaged to nodes) — arXiv:2604.16815
2. **Gate count per qubit** (local depth) — arXiv:2504.00464, 2511.03119
3. **Light-cone features** (which qubits causally affect observable) — arXiv:2512.23817
4. **Temporal calibration snapshot ID** (for drift) — arXiv:2512.14541

---

## 9. Conclusions and Recommendations

### What the literature tells us:

1. **NN for θ correction is fundamentally harder than NN for E correction**. The community has converged on correcting *outputs* not *inputs*. This aligns with our F18/F2 results.

2. **GNNs with calibration node/edge features are the state of the art** for noise-aware QEM. Our GNNQEMCorrector is well-positioned architecturally. Enhancements: per-edge 2Q errors, attention layers.

3. **FakeBackends provide sufficient noise realism for training**. Multiple papers validate using fake/simulated noise for NN training that transfers to real hardware. The key is diversity of noise configurations in training data.

4. **The "two-stage" paradigm dominates**: predict noiseless first, then correct. No 2025-2026 paper successfully trains a single model end-to-end from Hamiltonian+noise → θ_optimal_on_hardware.

5. **PNA from IBM is the industrial competitor** to our GNN-QEM approach. It's analytical (no NN) but requires a fitted noise model. Our GNN approach could complement PNA by handling non-Markovian effects PNA misses.

### Recommended next steps:

1. **Improve GNN-QEM** (Path A) — per-edge features, multi-calibration training → highest ROI
2. **Analytical coherent correction** (Path B) — simple formula, quick to implement
3. **Do NOT pursue Path C** (θ-correction NN) — literature + our data both say no
4. **Keep watching**: attention-based graph transformers (QAGT-MLP, GTranQEM) may replace MPNN as the architecture of choice by 2027

---

## References

| ID | Paper | Year | Key Contribution |
|----|-------|:----:|-----------------|
| R1 | Li & Hernandez, arXiv:2403.02762 | 2024 | Noise-induced θ phase transition |
| R2 | Chen et al., arXiv:2510.24050 | 2025 | Coherent errors absorbed by reparametrization |
| R3 | Legnini & Berberich, arXiv:2601.16758 | 2026 | Analytical bounds on θ perturbation |
| R4 | Karim et al., arXiv:2503.20210 | 2025 | ML VQE optimiser from trajectories |
| R5 | arXiv:2604.16815 | 2026 | Physically-informed GNN QEM (T1/T2 nodes) |
| R6 | arXiv:2504.00464 | 2025 | GNN circuit output prediction with noise features |
| R7 | Das et al., arXiv:2512.14541 | 2024 | GNN forensic noise inference |
| R8 | arXiv:2511.03119 | 2025 | QAGT-MLP attention graph transformer QEM |
| R9 | Cantori & Pilati, arXiv:2404.07802 | 2024 | CNN synergy noisy quantum + classical |
| R10 | arXiv:2506.04146 | 2025 | Deep-learned mitigation, partially knitted circuits |
| R11 | arXiv:2403.07025 | 2024 | NN-enhanced ZNE |
| R12 | arXiv:2501.01646 | 2025 | MPS pre-training + NN ZNE |
| R13 | Liao et al., arXiv:2309.17368 | 2023 | ML-QEM at 100 qubits (IBM) |
| R14 | arXiv:2509.12933 | 2025 | Data-efficient noise modeling via ML |
| R15 | Leclerc et al., arXiv:2601.18973 | 2026 | Meta-learning scaling laws for quantum control |
| R16 | arXiv:2512.23817 | 2025 | Attention GNN for Burgers eq. QEM |
| R17 | GTranQEM, OpenReview 2025 | 2025 | Non-message-passing graph transformer QEM |
| R18 | IBM PNA, qiskit-addon-pna | 2025-26 | Analytical noise absorption via Pauli propagation |
| R19 | arXiv:2503.22590 | 2025 | Parameter recycling / transfer learning VQE |
| R20 | arXiv:2504.10801 | 2025 | Q-Cluster: unsupervised noise-aware QEM |
