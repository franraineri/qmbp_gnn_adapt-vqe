# GNN Architecture & Training Reference

## Graph Representation

- Nodes (V): Represent qubits
- Node Features: Local external fields (e.g., scalar h_i from -h_i X_i)
- Edges (E): Represent interaction terms between qubits
- Edge Features: Interaction coupling strengths (e.g., scalar J_ij from -J_ij Z_i Z_j)

## Training Data Format

```python
sample = {
    "node_features": torch.tensor([[h]] * n_qubits, dtype=torch.float32),
    "edge_index": torch.tensor([[i, i+1] for i in range(n_qubits-1)], dtype=torch.long).T,
    "edge_features": torch.tensor([[J]] * (n_qubits-1), dtype=torch.float32),
    "theta_opt": torch.tensor(theta_opt, dtype=torch.float32),
}
```

## ML Pipeline (Curriculum Learning)

### Supervised Pre-training (MSE Loss)

L_MSE = (1/M) Σ || θ_pred^(k) - θ_opt^(k) ||²

The GNN learns to regress the exact angles discovered by the classical optimizer.

### Physics-Informed Fine-Tuning (Energy Loss)

L_Physics = ⟨ψ(GNN(G)) | H_G | ψ(GNN(G))⟩

Requires a differentiable quantum simulator backend (like TorchQuantum) or analytical gradients via parameter-shift rules to backpropagate through the quantum circuit into the GNN weights.

## Architecture Details

- Input: Hamiltonian graph (nodes=qubits+field features, edges=couplings+J features)
- Output: θ_pred ∈ ℝ^(2p) for TFIM HVA
- Loss: MSE(θ_pred, θ_opt) + λ·E(θ_pred) physics-informed regularizer
- Architecture: Message-passing GNN (2-3 layers), global pooling → MLP head
- Training: normalize θ_opt to [-π, π]; split by h-value ranges (not random)

## PoC Architecture (V6.0 — Current)

The V6 pipeline uses a full MPNN (GINConv + global_mean_pool) as the predictor. The earlier V4 MLP (h → θ_pred) served as proof-of-concept but is superseded. The MPNN accepts arbitrary graph topologies, enabling scaling to ladders and 2D lattices without architecture changes.

For 1D TFIM with uniform J, the graph structure is fixed and only h varies — the MPNN still works correctly (it just doesn't exploit edge heterogeneity). When extending to non-uniform couplings or 2D lattices, the MPNN's graph-awareness becomes critical.

## Scaling: Qracle Unified Graph Encoding (Zhang et al., 2025)

When upgrading from MLP to GNN, adopt Qracle's unified representation that encodes BOTH the Hamiltonian AND the ansatz circuit into a single graph:

- **Current PoC**: Only Hamiltonian encoded (nodes=qubits, edges=couplings). Sufficient for fixed-topology TFIM.
- **Scaling target**: Unified graph where Hamiltonian structure + ansatz gate topology are jointly encoded. This captures how the circuit structure interacts with the physical problem.
- **Why it matters**: Qracle showed GNN with unified encoding outperforms MLP/diffusion methods by up to 64% fewer optimization steps and 26% lower SMAPE. The advantage is largest on physically structured Hamiltonians (spin systems) — exactly our domain.
- **Architecture reference**: GCNConv (2 layers, dim=256) + GATConv (3 layers, dim=512) + MLP head (dim=1024). PyTorch Geometric.

## GINConv Theoretical Justification (Xu et al., ICLR 2019)

- GIN (Graph Isomorphism Network) is provably as powerful as the Weisfeiler-Lehman graph isomorphism test
- This means GINConv can distinguish any two non-isomorphic graphs that WL can distinguish
- For uniform lattices (all edges equivalent), GINConv is optimal — attention (GAT) adds nothing because there's no heterogeneity to attend to
- For non-uniform lattices (different J values, mixed topologies), GATConv may help — attention can weight edges by coupling strength
- **Our validated choice**: GINConv for 1D uniform TFIM. Consider GATConv only for non-uniform couplings or mixed topologies.

## MPNN Capacity Scaling Rule (from 40+ experiments)

| System size | Optimal hidden_dim | Rationale |
|-------------|-------------------|-----------|
| N=6 (6 nodes, 5 edges) | 64 | 128 overfits on 17 training points |
| N=10 (10 nodes, 9 edges) | 128 | 64 underfits — more graph structure to learn |
| N=20 (projected) | 256 | Scale ~10-13× number of nodes |

Rule of thumb: `hidden_dim ≈ 10-13 × N_nodes`. Always validate with energy-driven callback.

## Literature-Informed Architecture Insights

### GNN > CNN by 36% (Meng et al., 2025)
- GNN naturally captures circuit/lattice topology that CNNs must learn implicitly
- Node features encoding noise information improves noisy predictions
- Direct comparison scheme (predicting relative performance) outperforms indirect (predicting absolute values) by 36.2%

### GNN for Ising Magnetization (Slavin, 2025)
- Lattice geometry encoded as graph → GNN → magnetization prediction
- Captures plateaus, critical transitions, geometric frustration effects
- Trained on Monte Carlo data (analogous to our exact diag training)
- Validates: graph → physical property paradigm works for Ising systems specifically

### Transferability Limits (Bincoletto et al., 2025)
- ML parameter prediction transfers between "similar" systems (same topology, different size)
- Degrades for qualitatively different structures (1D chain → 2D lattice)
- **Implication for us**: MPNN trained on 1D chains needs fine-tuning for ladders/2D. The architecture (GINConv + global_mean_pool) is topology-agnostic, but learned weights are topology-specific.

### MPNN Weight Structure Encodes Phase Information (Hernandes et al., 2025)
- Neural network weights trained across a phase diagram exhibit structural changes at phase boundaries
- Phase transitions manifest as distinct structures in weight space
- **Opportunity**: Analyze our MPNN's weight gradients across the h-sweep to detect phase transitions without additional quantum measurements. Zero-cost analysis.

## Training Best Practices

- **Dropout (NN-VQE, Miao et al. 2024)**: Add `nn.Dropout(0.1)` after hidden layers. Proven to improve generalization on small datasets (≤30 training points). Current PoC uses dropout=0.1 after first hidden layer.
- **Physics validation callback**: Every N epochs, feed θ_pred into StatevectorEstimator to compute E(θ_pred) and compare against exact ground energy. Ensures predicted angles retain physical meaning.
- **LR scheduling**: Use ReduceLROnPlateau or CosineAnnealing to avoid oscillation on small datasets.
- **Interpolation test**: Always validate on at least one h value not in the training set.
- **Fidelity filter (critical)**: Only train on Phase 2 data points where fidelity ≥ 96%. Points below this threshold have θ_opt that don't represent the true ground state — training on them poisons the model.
- **Active learning (future)**: NN-VQE showed that actively selecting training points near phase transitions (where θ_opt changes abruptly) can halve the required dataset size while maintaining accuracy.

## Data Management

- Store as .npz or HDF5
- Schema: {h, J, n_qubits, ground_energy, ground_state, theta_opt, local_obs, metadata}
- Version with (n_qubits, p_layers, optimizer) in filename
