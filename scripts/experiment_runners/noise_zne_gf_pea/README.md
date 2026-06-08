# Noise Suppression & ZNE Experiments

Error mitigation validation scripts for the GNN-HVA hardware deployment.

## Scripts

| Script | Experiment ID | Purpose | Time |
|--------|:------------:|---------|:----:|
| `run_gf_zne_comparison.py` | GF_ZNE_CMP | GF vs CES-ZNE per topology | ~3 min |
| `run_gf_zne_batch.py` | GF_ZNE_CMP | Batch all 3 topologies | ~3 min |
| `run_zne_3way_comparison.py` | ZNE_3WAY | CES vs GF vs PEA 3-way | ~2 min |
| `run_pea_zne_validation.py` | PEA_ZNE_VAL | Multi-seed PEA validation | ~5 min |
| `run_pea_hardware_readiness.py` | PEA_HW_READY | PEA on heavy_hex N=10 | ~3 min |
| `run_pea_full_pipeline.py` | PEA_PIPELINE | VQE→MPNN→PEA→Classify | ~5 min |
| `run_zne_cross_topology_validation.py` | ZNE_CROSS_TOPO | Definitive 3-topo comparison | ~2 min |
| `run_pea_triangular_validation.py` | PEA_TRIANGULAR | PEA on triangular (gap G6) | ~1 min |

## Supporting Scripts (project root `scripts/`)

| Script | Purpose | Time |
|--------|---------|:----:|
| `run_gnn_qem_training.py` | Train GNN-QEM model (chain+ladder) | ~1 min |
| `run_gnn_qem_cross_topology.py` | Cross-topology generalization test | ~1 min |
| `audit_affine_overshoot.py` | Audit ZNE overshoot frequency | <1s |

## Execution Commands

```bash
# PEA triangular — 3 seeds × 3 h-points, N=6 p=1
.venv/bin/python scripts/experiment_runners/noise_zne_gf_pea/run_pea_triangular_validation.py

# GNN-QEM cross-topology — train chain+ladder, test heavy_hex N=10
.venv/bin/python scripts/run_gnn_qem_cross_topology.py

# Affine overshoot audit — scan all ZNE results
.venv/bin/python scripts/audit_affine_overshoot.py

# Definitive ZNE cross-topology (PEA vs GF, 18 evaluations)
.venv/bin/python scripts/experiment_runners/noise_zne_gf_pea/run_zne_cross_topology_validation.py

# Full ZNE campaign (all 8 experiments sequentially)
.venv/bin/python scripts/experiment_runners/noise_zne_gf_pea/run_pea_triangular_validation.py && \
.venv/bin/python scripts/run_gnn_qem_cross_topology.py && \
.venv/bin/python scripts/audit_affine_overshoot.py
```

## Key Results (2026-06-05)

| Experiment | Verdict | Key Metric |
|-----------|:-------:|------------|
| PEA_TRIANGULAR | ✅ | PEA +96.8%, t=111.22, p≈0, 9/9 wins |
| GNN-QEM Cross-Topo | ✅ | 100% improvement on unseen heavy_hex (+72.3%) |
| Affine Audit | ✅ | 0% overshoot in 102 records |

## Definitive Mitigation Ranking (60+ evaluations)

| Rank | Method | Mean Gain | R² | Robustness |
|:----:|--------|:---------:|:--:|:----------:|
| 1 | PEA-ZNE | +83% | 0.86–1.00 | 48/48 always positive |
| 2 | GF-ZNE | +12% | 0.88 | 54/60 always positive |
| 3 | CES-ZNE | +3% | 0.99 | 14/18 (78%) |

## Coverage Matrix (all gaps closed except hardware)

| Topology | N | CES-ZNE | GF-ZNE | PEA-ZNE | GNN-QEM |
|----------|---|:-------:|:------:|:-------:|:-------:|
| chain_1d | 6 | ✅ | ✅ | ✅ | ✅ (train) |
| ladder | 6 | ✅ | ✅ | ✅ | ✅ (train) |
| heavy_hex | 10 | ❌ (broken) | ✅ | ✅ | ✅ (zero-shot) |
| triangular | 6 | ✅ | ✅ | ✅ | — |

## Output Locations

| File | Content |
|------|---------|
| `results/experiments/exp_pea_triangular/` | PEA triangular per-run JSON |
| `results/gnn_qem/cross_topology_results.json` | GNN-QEM generalization results |
| `results/gnn_qem/model_cross_topo.pt` | Fine-tuned model checkpoint |
| `results/gnn_qem/affine_overshoot_audit.json` | Overshoot audit |
| `documentation/analysis/16_noise_suppression_analysis.md` | Full gap analysis |
