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

## PoC Simplification

For 1D TFIM with uniform J, the graph structure is fixed and only h varies. Use a simple MLP (h → θ_pred) as the PoC predictor; upgrade to full GNN when extending to non-uniform couplings or 2D lattices.

## Training Best Practices

- **Physics validation callback**: Every N epochs, feed θ_pred into StatevectorEstimator to compute E(θ_pred) and compare against exact ground energy. Ensures predicted angles retain physical meaning.
- **LR scheduling**: Use ReduceLROnPlateau or CosineAnnealing to avoid oscillation on small datasets.
- **Interpolation test**: Always validate on at least one h value not in the training set.
- **Fidelity filter (critical)**: Only train on Phase 2 data points where fidelity ≥ 96%. Points below this threshold have θ_opt that don't represent the true ground state — training on them poisons the model.

## Data Management

- Store as .npz or HDF5
- Schema: {h, J, n_qubits, ground_energy, ground_state, theta_opt, local_obs, metadata}
- Version with (n_qubits, p_layers, optimizer) in filename
