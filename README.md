# Hybrid GNN-HVA Framework for Topological Phase Characterization

Master's Thesis — Quantum Computing & Condensed Matter Physics (UNIR, 2026).

Accelerates VQE for quantum phase classification via a Graph Neural Network that predicts HVA circuit parameters directly from the Hamiltonian's graph structure, eliminating iterative quantum optimization (29–500× speedup).

## Key Results

| Metric | Value |
|--------|-------|
| Pass rate (optimal configs) | 95–100% (ΔE/gap < 5%) |
| Speedup vs VQE random init | 29–500× |
| Topologies validated | 5 (chain, ladder, square, heavy-hex, triangular) |
| System sizes | N = 4–20 (exact), N = 40–250 (MPS) |
| Models | TFIM, TFIM+longitudinal, TFIM frustrated, Heisenberg (negative), Kitaev (negative) |
| Depth range | p = 1–8 |

## Expressibility Frontier (h_min)

The HVA cannot express the ground state below a topology/model-dependent threshold:

```
p=1: h_min = 2.36 + 0.0073·N  (linear, R²=0.91)
p=2: h_min = 1.57 + 0.005·N   (linear, R²=0.95)
p≥3: h_min ≈ 1.4–1.6          (constant, independent of N — area law)
```

Topology ranking (best→worst): chain_1d (1.09) ≈ heavy_hex (1.12) < kagome (1.49) < ladder (1.67) ≈ square (1.68) < triangular (2.20).

See `results/H_EXPR_MATRIX.md`, `results/H_FRONTIER_MODELS.md`, `results/H_FRONTIER_TOPOLOGIES.md` for full data.

## Quick Start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,test]"
python tests/smoke_test.py          # N=4, p=1, <30s
make test                           # Full test suite
make check-full                     # Lint + tests + smoke
```

## Pipeline (3 Phases — Noiseless Simulation)

1. **Phase 1**: Ground truth via exact diagonalization (N≤14) or DMRG (N>14, χ=64)
2. **Phase 2**: Warm-start VQE (descending h sweep) produces θ_opt(h) training data
3. **Phase 3**: GINConv MPNN learns h→θ mapping, deploys predictions without VQE

```bash
# Run full pipeline
.venv/bin/python scripts/experiment_runners/noiseless/run_noiseless_pipeline.py \
  --model tfim --topology chain_1d --n-qubits 10 --p-layers 3 --seeds 42 43 44
```

## Project Structure

```
src/qmbp_simulation/       # Installable package (models, solvers, circuits, execution,
                           #   optimizers, predictors, pipeline, framework, analysis)
scripts/
  analysis/                # 13 canonical analysis scripts (compute_h_frontier, etc.)
  experiment_runners/      # Pipeline & experiment runners
project_health/            # Automated project health, coverage, diagnostics
tests/                     # pytest suite (unit + integration)
results/                   # Experiment outputs (JSON)
tesis-v3.0.tex             # Thesis document (LaTeX)
```

## Analysis Scripts

```bash
.venv/bin/python scripts/analysis/compute_h_frontier.py --json          # Frontier vs N
.venv/bin/python scripts/analysis/compute_h_frontier_all.py --model tfim # All topologies
.venv/bin/python scripts/analysis/analyze_all_phase3.py --date 20260717  # Phase3 MPNN
.venv/bin/python scripts/analysis/check_matrix_gaps.py --json            # Missing data
.venv/bin/python scripts/analysis/extract_theta_trajectories.py          # θ(h) data
```

## Key Findings

1. **HVA depth determines frontier** — for p≥3, h_min is independent of N (area law)
2. **Circuit-Hamiltonian match is critical** — Heisenberg stays at h≈3.5 regardless of p (Wiersema2020)
3. **heavy_hex ≈ chain_1d** — IBM's native topology has no expressibility penalty
4. **p=3 is optimal for pipeline** — more depth degrades MPNN (θ_smoothness explosion)
5. **Bond-resolved crosses h_c** — only strategy reaching the ordered phase (h_min=0.83)
6. **Cross-N fails with global HVA** — θ(h) is N-specific; fixed with N/100 node feature

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
- Zhang et al., 2025 (Qracle)
- Tripathi et al., 2026 (HVA vs HEA benchmark)

## License

Academic use. Code: [github.com/franraineri/qmbp_gnn_adapt-vqe](https://github.com/franraineri/qmbp_gnn_adapt-vqe)
