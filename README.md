# Hybrid GNN-HVA Framework for Quantum Phase Classification

Accelerates VQE for quantum phase classification via a Graph Neural Network that predicts optimal HVA circuit parameters directly from the Hamiltonian's graph structure, eliminating iterative quantum optimization (29–500× speedup).

Includes a GNN-based Quantum Error Mitigation (GNN-QEM) module with zero-shot cross-topology transfer capability.

## Key Results

| Metric | Value |
|--------|-------|
| Pass rate (optimal configs) | 95–100% (ΔE/gap < 5%) |
| Speedup vs random-init VQE | 29–500× |
| Topologies validated | 5 (chain, ladder, square, heavy-hex, triangular) |
| System sizes | N = 4–20 (exact), N = 40–250 (MPS) |
| Models | TFIM, TFIM+longitudinal, TFIM frustrated, Heisenberg (neg.), Kitaev (neg.) |
| GNN-QEM improvement | 100% rate on unseen topologies (zero-shot) |

## Quick Start

```bash
# Install
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,test]"

# Smoke test (~30s)
python tests/smoke_test.py

# Full pipeline (N=10, chain_1d, p=2)
python scripts/experiment_runners/noiseless/run_noiseless_pipeline.py \
    --model tfim_longitudinal --topology chain_1d --n-qubits 10 --p-layers 2
```

## Demo Notebooks

Interactive demonstrations (no hardware required):

| Notebook | What it does | Runtime |
|----------|-------------|:-------:|
| `notebooks/01_pipeline_quickstart.ipynb` | Full Phase 1→4 pipeline (N=6) | ~30s |
| `notebooks/02_expressibility_check.ipynb` | Pre-flight diagnostics + h_min atlas | ~5s |
| `notebooks/03_gnn_qem_demo.ipynb` | GNN-QEM error correction + cross-topology | ~10s |

To generate pre-trained model checkpoints for notebooks:
```bash
python notebooks/data/generate_samples.py
```

## Pipeline (3 Phases — Noiseless Simulation)

1. **Phase 1**: Ground truth via exact diagonalization (N≤14) or DMRG (N>14, χ=64)
2. **Phase 2**: Warm-start VQE (descending h sweep) → θ_opt(h) training data
3. **Phase 3**: GINConv MPNN learns h→θ mapping, deploys zero-shot predictions

## Expressibility Frontier

The HVA cannot express the ground state below a model/topology-dependent threshold:

```
p=1: h_min = 2.36 + 0.0073·N   (linear)
p=2: h_min = 1.57 + 0.005·N    (linear)
p≥3: h_min ≈ 1.4–1.6           (constant — area law)
```

Topology ranking: chain_1d ≈ heavy_hex < kagome < ladder ≈ square < triangular.

See `results/HVA_EXPRESSIBILITY_ANALYSIS.md` for the full atlas.

## Project Structure

```
src/qmbp_simulation/           # Installable Python package
├── models/                    #   Hamiltonians, lattices, model registry
├── circuits/                  #   HVA circuit builder (7 ansatz variants)
├── solvers/                   #   Exact diag, DMRG
├── execution/                 #   Backends (noiseless, noisy, hardware)
├── optimizers/                #   VQE with warm-start sweep
├── predictors/                #   MPNN (GINConv), GNN-QEM error correction
├── pipeline/                  #   PipelineRunner orchestration
├── framework/                 #   Experiment engine, CLI, result I/O
└── analysis/                  #   Diagnostics, normalizing flows, validators

notebooks/                     # 3 interactive demos
scripts/
├── analysis/                  # Canonical analysis scripts
├── experiment_runners/        # Pipeline & experiment runners
├── hardware/                  # IBM QPU deployment scripts
├── maintenance/               # Repo maintenance utilities
└── validation/                # Validation scripts

tests/                         # pytest suite (unit + integration + property)
configs/presets/               # Experiment presets (hardware, noiseless, noisy)
project_health/                # Automated health monitoring & digest

results/                       # Key results (expressibility atlas, GNN-QEM models)
internal/                      # Development materials (thesis, binnacles, analysis notes)
```

## Analysis Scripts

```bash
python scripts/analysis/compute_h_frontier.py --json           # h_min vs N
python scripts/analysis/compute_h_frontier_all.py --model tfim # All topologies
python scripts/analysis/analyze_all_phase3.py --date 20260717  # Phase3 MPNN
python scripts/analysis/check_matrix_gaps.py --json            # Coverage gaps
```

## Key Findings

1. **HVA depth determines frontier** — for p≥3, h_min is independent of N (area law)
2. **Circuit-Hamiltonian match is critical** — Heisenberg at h≈3.5 regardless of p
3. **heavy_hex ≈ chain_1d** — IBM's native topology has no expressibility penalty
4. **Bond-resolved crosses h_c** — only strategy reaching the ordered phase (h_min=0.83)
5. **GNN-QEM generalizes cross-topology** — 100% improvement rate zero-shot to heavy_hex

## Dependencies

- Python 3.12+
- Qiskit 2.x (Primitives V2, SparsePauliOp)
- PyTorch + PyTorch Geometric (GINConv)
- qiskit-aer (MPS backend for N>22)
- TeNPy (DMRG ground truth)

## References

- Mele et al., Nature Physics 2026 (depth truncation)
- Wiersema et al., PRX Quantum 2020 (HVA)
- Puig et al., PRX Quantum 2025 (warm-start)
- Xu et al., ICLR 2019 (GINConv)
- Miao et al., PRApplied 2024 (NN-VQE)
- Zhang et al., 2025 (Qracle, VQEzy)
- Tripathi et al., 2026 (HVA vs HEA benchmark)

## License

MIT — [github.com/franraineri/qmbp_gnn_adapt-vqe](https://github.com/franraineri/qmbp_gnn_adapt-vqe)
