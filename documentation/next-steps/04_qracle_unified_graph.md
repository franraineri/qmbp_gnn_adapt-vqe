# Integration Plan 04: Qracle Unified Hamiltonian+Circuit Graph

**Paper:** Zhang et al. (2025) — Qracle: A GNN-based Parameter Initializer for VQEs  
**arXiv:** 2505.01236  
**Code:** ✅ `https://github.com/chizhang24/Qracle`  
**Priority:** MEDIUM (3-4 days, strongest benefit for `tfim_bond_resolved`)

## What It Does

Qracle encodes BOTH the Hamiltonian structure AND the ansatz circuit structure
into a single unified graph. Hamiltonian terms become nodes/edges, AND circuit
gates become additional nodes with connectivity reflecting the circuit topology.
The GNN processes this combined representation to predict optimal parameters.

Our current approach: graph = Hamiltonian only (qubits=nodes, interactions=edges).
Qracle's approach: graph = Hamiltonian + Circuit (adds gate nodes + parameter nodes).

## Viability Assessment

| Criterion | Status |
|-----------|--------|
| Compatible with our pipeline? | ⚠️ Requires new graph construction logic |
| Requires new dependencies? | ❌ PyTorch Geometric (already have) |
| Reuses existing modules? | ✅ `MPNNPredictor` (same GINConv backbone) |
| Addresses a real problem? | ⚠️ For global-param HVA: NO (circuit is uniform) |
| For bond-resolved HVA? | ✅ YES — each bond has different θ, circuit structure matters |
| Publishable? | ✅ If improves bond-resolved cross-N generalization |

## How To Integrate

### What It Proves

That encoding circuit structure in the graph improves MPNN prediction for
high-dimensional parameter spaces (bond-resolved HVA: 19+ params at N=10).

### Conditions Where It Makes Sense

- **Models:** `tfim_bond_resolved` ONLY (global-param models don't benefit)
- **Topologies:** ALL (but strongest benefit on non-chain topologies)
- **N:** 10-20 (where bond-resolved has 19-39 params)
- **p:** 1-2 (more layers = more circuit nodes = richer graph)

### When NOT to Use

- Standard TFIM/TFIM-long with global parameters (2-3 params/layer):
  circuit is uniform → no structural information to encode
- Small N (N=4-6): few enough params that MLP or standard GNN suffices
- Time-critical: larger graph = slower training (2-3× more nodes)

### Integration Architecture

```
src/qmbp_simulation/
└── predictors/
    ├── mpnn.py                  # ✅ EXISTS: MPNNPredictor, BondResolvedMPNN
    ├── unified_graph.py         # NEW: Build Hamiltonian+Circuit unified graph
    └── unified_mpnn.py          # NEW: MPNN variant for unified graphs
```

### Modules to Reuse

| Module | Usage |
|--------|-------|
| `predictors.mpnn.MPNNPredictor` | Base GINConv architecture (extend, don't rewrite) |
| `predictors.mpnn.build_bond_resolved_graph` | Current graph builder (to compare against) |
| `predictors.mpnn.BondResolvedMPNN` | Existing bond-resolved predictor |
| `models.make_lattice` | Lattice edge structure |
| `circuits.hva.HVACircuitBuilder.create_bond_resolved` | Circuit structure info |

### Graph Construction Design

```python
def build_unified_graph(
    lattice: LatticeConfig,
    p_layers: int,
    h: float,
    theta_opt: np.ndarray,
) -> Data:
    """Build unified Hamiltonian+Circuit graph following Qracle pattern.

    Node types:
      - Qubit nodes (N): features = [h_i, coord_i, type=0]
      - ZZ gate nodes (n_edges × p): features = [layer_idx, bond_idx, type=1]
      - RX gate nodes (N × p): features = [layer_idx, qubit_idx, type=2]

    Edge types:
      - Hamiltonian edges: qubit ↔ qubit (from lattice.edges)
      - Gate-qubit edges: gate_node → qubit_node (which qubits the gate acts on)
      - Sequential edges: gate_l → gate_{l+1} (circuit ordering within a layer)

    Target: theta_opt (one param per gate node)
    """
```

### Implementation Steps

1. **Study Qracle's graph format** from their GitHub repo (understand node/edge types)
2. **Create `predictors/unified_graph.py`** (~100 lines):
   - `build_unified_graph()` function
   - Handle heterogeneous node types via type-encoding features
3. **Create `predictors/unified_mpnn.py`** (~60 lines):
   - Subclass or wrap `MPNNPredictor` with modified readout
   - Instead of `global_mean_pool → head → [2p]`, do:
     `pool gate nodes only → predict θ per gate node`
4. **Benchmark script** comparing:
   - Current `BondResolvedMPNN` (Hamiltonian-only graph)
   - New `UnifiedMPNN` (Hamiltonian+Circuit graph)
   - Metrics: MSE, ΔE/gap, cross-N generalization

### Expected Output

```json
{
  "model": "tfim_bond_resolved",
  "topology": "chain_1d",
  "N_train": 10,
  "N_test": 10,
  "current_bond_resolved_de_gap": 0.015,
  "unified_graph_de_gap": 0.008,
  "cross_N_current": "FAIL (324%)",
  "cross_N_unified": "TBD",
  "graph_size_increase": "2.5x (nodes: 10→25 for N=10 p=1)",
  "training_time_increase": "1.8x"
}
```

### Success Criterion

- ΔE/gap improvement ≥ 30% on bond-resolved tasks → worth the complexity
- Cross-N transfer improves (currently fails at 324%) → breakthrough finding
- If no improvement on chain_1d: test on square/triangular where heterogeneity matters

### Risks

- Qracle's benefit comes from encoding DIFFERENT ansätze (HEA, UCCSD, etc.) in
  the same graph. For a SINGLE ansatz (bond-resolved HVA), the circuit structure
  is always the same → may add no information
- Heterogeneous graph processing requires careful pooling (can't just mean-pool
  all nodes — need to distinguish qubit vs gate nodes)
- Training on 17-35 points with 25+ node graphs may overfit
- May need > 64 hidden dim due to larger graph complexity
