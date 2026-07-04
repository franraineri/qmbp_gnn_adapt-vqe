## TN1 — Noiseless Model Ranking

*Deploy pass rate (ΔE/gap < 5%) by Hamiltonian model, aggregated across all topologies and p-layers.*

| Model | N Runs | Mean Pass% | Best | Worst | Viable |
|---|---|---|---|---|---|
| tfim | 32 | 64.6% | 94.9% | 25.6% | ✅ |
| tfim_longitudinal | 21 | 59.2% | 89.7% | 12.8% | ✅ |
| heisenberg | 26 | 0.0% | 0.0% | 0.0% | ❌ |

**Notes**: Heisenberg p=4 uniformly fails — HVA expressibility insufficient for frustrated Heisenberg.

## TN2 — Noiseless Topology Ranking

*Deploy pass rate by topology, aggregated across all models and p-layers.*

| Topology | N Runs | Mean Pass% | Best | Status |
|---|---|---|---|---|
| heavy_hex | 16 | 53.5% | 92.3% | ✅ |
| chain_1d | 17 | 46.6% | 94.9% | ⚠️ |
| ladder | 15 | 42.9% | 79.5% | ⚠️ |
| square | 15 | 40.5% | 79.5% | ⚠️ |
| triangular | 16 | 25.6% | 56.4% | ❌ |

**Notes**: heavy_hex leads due to low connectivity (9 edges for N=10) matching HVA expressibility.

## TN3 — Model × Topology Cross-Table

*Mean deploy pass rate for each (model, topology) combination.*

| Model | chain_1d | heavy_hex | ladder | square | triangular |
|---|---|---|---|---|---|
| heisenberg | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| tfim | 76.9% | 82.1% | 65.8% | 64.1% | 35.9% |
| tfim_longitudinal | 66.2% | 70.5% | 62.2% | 55.8% | 39.7% |

**Notes**: TFIM models perform best on heavy_hex; triangular topology penalizes all models.

## TN4 — Best Configurations Per Model

*Top-3 performing (topology, p) configurations for each Hamiltonian.*

| Model | Topology | p | Pass% | Mean ΔE/gap | Mean F | Speedup |
|---|---|---|---|---|---|---|
| heisenberg | chain_1d | 4 | 0% | 3.03e+01 | 0.0131 | 72.8x |
| heisenberg | ladder | 4 | 0% | 5.03e+02 | 0.0483 | 120.0x |
| heisenberg | triangular | 4 | 0% | 6.42e+01 | 0.0008 | 82.1x |
| tfim | chain_1d | 3 | 95% | 8.24e-03 | 0.9981 | 44.1x |
| tfim | heavy_hex | 4 | 92% | 1.36e-02 | 0.9974 | 73.7x |
| tfim | heavy_hex | 3 | 90% | 2.38e-02 | 0.9952 | 50.0x |
| tfim_longitudinal | chain_1d | 2 | 90% | 2.80e-02 | 0.9915 | 19.6x |
| tfim_longitudinal | heavy_hex | 3 | 90% | 2.38e-02 | 0.9952 | 81.7x |
| tfim_longitudinal | heavy_hex | 2 | 85% | 5.78e-02 | 0.9817 | 19.9x |

**Notes**: Speedup = MPNN inference time / VQE solve time (higher = more practical).

## TN5 — VQE Optimization Quality by Model

*VQE convergence quality — fidelity, energy gap error, and entanglement entropy.*

| Model | Runs | VQE Pass | Mean F | Mean ΔE/gap | Mean S | Quality |
|---|---|---|---|---|---|---|
| heisenberg | 26 | 0/26 | 0.0291 | 8.47e+01 | 1.644 | ❌ |
| tfim | 32 | 7/32 | 0.9382 | 1.78e+02 | 0.424 | ⚠️ |
| tfim_longitudinal | 21 | 7/21 | 0.9289 | 5.65e+01 | 0.302 | ⚠️ |

**Notes**: Heisenberg VQE fidelity ≈ 0 confirms HVA ansatz cannot express ground state.

## TN6 — MPNN Training Quality by Model

*MPNN generalization quality — final MSE and per-h prediction error.*

| Model | Runs | MPNN Pass | Mean MSE | Mean per-h MSE | Quality |
|---|---|---|---|---|---|
| heisenberg | 26 | 9/26 | 5.27e-03 | 6.41e-04 | ✅ |
| tfim | 32 | 11/32 | 8.72e-03 | 1.23e-04 | ✅ |
| tfim_longitudinal | 21 | 6/21 | 1.02e-02 | 3.61e-04 | ⚠️ |

**Notes**: MPNN can fit even bad VQE data (low MSE) — but garbage in → garbage out at deploy.

## TN7 — Performance by Circuit Depth (p-layers)

*Deploy pass rate grouped by model and HVA p-layers.*

| Model | p | N Runs | Mean Pass% | Best | Worst |
|---|---|---|---|---|---|
| heisenberg | 1 | 5 | 0.0% | 0.0% | 0.0% |
| heisenberg | 2 | 5 | 0.0% | 0.0% | 0.0% |
| heisenberg | 3 | 5 | 0.0% | 0.0% | 0.0% |
| heisenberg | 4 | 11 | 0.0% | 0.0% | 0.0% |
| tfim | 1 | 13 | 53.5% | 74.4% | 25.6% |
| tfim | 2 | 9 | 67.5% | 84.6% | 33.3% |
| tfim | 3 | 6 | 76.1% | 94.9% | 48.7% |
| tfim | 4 | 4 | 76.9% | 92.3% | 56.4% |
| tfim_longitudinal | 1 | 5 | 44.1% | 74.4% | 15.4% |
| tfim_longitudinal | 2 | 6 | 69.2% | 89.7% | 41.0% |
| tfim_longitudinal | 3 | 5 | 70.8% | 89.7% | 46.2% |
| tfim_longitudinal | 4 | 5 | 50.8% | 79.5% | 12.8% |

**Notes**: TFIM: p=2 optimal for chain_1d/heavy_hex. p=4 helps ladder/square. Heisenberg: p=4 insufficient.
