---
inclusion: fileMatch
fileMatchPattern: "**/aqc_compression*,**/aqc_tensor*,**/aqc*compress*"
---

# AQC-Tensor Compression Context

> Pre-digested context for circuit compression via Approximate Quantum Compilation
> with Tensor Networks (qiskit-addon-aqc-tensor v0.3.0).

## What It Does

Compresses a bound HVA circuit (e.g., p=2 with θ_opt) into a shallower circuit
that prepares (approximately) the same state. The compressed circuit has fewer
2Q gates and is ZNE-compatible even when the original is not.

## Key Results (POC 2026-06-17)

| Topology | Fidelity | ΔE/gap | 2Q reduction | vs p=1 direct |
|----------|:--------:|:------:|:------------:|:-------------:|
| heavy_hex N=10 | 0.9996 | 0.42% | 50% | +15.6% better |
| chain_1d N=10 | 0.9992 | 0.29% | 50% | +14% near h_c |
| ladder N=10 | 0.9979 | 0.96% | 50% | better near h_c |
| triangular N=10 | 0.9986 | 1.63% | 50% | better near h_c |

## Module Location

```
src/qmbp_simulation/circuits/aqc_compression.py
```

## Public API

```python
from qmbp_simulation.circuits import (
    AQCCircuitCompressor,
    AQCCompressionConfig,
    AQCCompressionResult,
    CompressionValidation,
    AQCCompressionCache,
)
```

## Usage Pattern

```python
from qmbp_simulation.circuits.aqc_compression import (
    AQCCircuitCompressor, AQCCompressionConfig,
)

# 1. Build + bind target circuit (p=2 with VQE-optimized params)
target_circuit = circuit_p2.assign_parameters(theta_opt)

# 2. Compress
config = AQCCompressionConfig(max_bond_dim=64, fidelity_threshold=0.998)
compressor = AQCCircuitCompressor(config)
result = compressor.compress_circuit(target_circuit, lattice)

# 3. Validate
if result.fidelity >= 0.998 and result.is_zne_viable(amplifier="pea"):
    hw_circuit = result.compressed_circuit  # Ready for hardware
```

## Integration with Hardware Deployment

```bash
python scripts/experiment_runners/hardware/run_ibm_deployment.py --aqc-compress
```

CLI flags:
- `--aqc-compress`: Enable AQC compression (p=2→shallow)
- `--aqc-bond-dim 64`: MPS bond dimension (default: 64)
- `--aqc-fidelity 0.998`: Minimum fidelity threshold
- `--aqc-p-source 2`: Source p_layers to compress from

## Key Constraints

- **Optional dependency**: requires `pip install 'qiskit-addon-aqc-tensor[quimb-jax]'`
- **Lazy imports**: never breaks if not installed — ImportError with install instructions
- **Only for bound circuits**: target_circuit must have 0 free parameters
- **MPS-simulable states only**: works best for h > h_c (low entanglement)
- **Fidelity threshold**: reject compression if F < 0.998 → fallback to p=1 direct
- **Convergence**: uses `result.success or result.status == 0` (scipy L-BFGS-B)
- **DAG compliance**: does NOT import from `optimizers` module; uses scipy directly

## When AQC Adds Value

- p=2 circuits (36 CX → ZNE fails) compressed to p=1-equivalent (18 CX → ZNE works)
- Near phase boundary (h close to h_c): compressed p=2 is better than direct p=1
- Heavy_hex topology: best results (100% pass rate at threshold=0.998)

## When AQC Does NOT Help

- p=1 circuits: already minimal depth, nothing to compress
- Deep paramagnetic (h >> h_c): p=1 already gives excellent ΔE/gap
- Very near h_c: MPS truncation error grows, fidelity degrades
- **FakeTorino simulation** (depolarizing noise): AQC+PEA is 3.7% worse than PEA alone
  (fidelity loss penalty outweighs depth reduction in isotropic depolarizing model).
  This is a simulation artifact — on hardware, depth reduction reduces T1/T2 decay
  which is NOT modeled by FakeTorino. Ref: `binnacle-mitigation-benchmark.md` H15.

## Caching

```python
from qmbp_simulation.circuits import AQCCompressionCache

cache = AQCCompressionCache()  # Default: results/aqc_cache/
cached_params = cache.get("heavy_hex", 10, 3.5, theta_opt, 64)
if cached_params is None:
    result = compressor.compress_circuit(...)
    cache.put("heavy_hex", 10, 3.5, theta_opt, 64, result.optimal_params, result.fidelity)
```

## ZNE Viability Check

```python
# After compression, verify the circuit fits the ZNE budget
assert result.is_zne_viable(amplifier="pea")    # n_2q ≤ 50
assert result.is_zne_viable(amplifier="gate_folding")  # n_2q ≤ 18
```

## Relationship to Other Components

| Component | Relationship |
|-----------|-------------|
| `HVACircuitBuilder` | Produces the target circuit to compress |
| PEA-ZNE | Complementary: lower depth → PEA more effective |
| MPNN predictor | Provides θ_opt for the target circuit |
| `MPSBackend` | Can generate target MPS (alternative to quimb) |
| `transpiled_circuit_stats()` | Measures depth before/after compression |
| Affine correction | Applied after ZNE on compressed circuit |

## References

- arXiv:2301.08609 — AQC method
- qiskit-addon-aqc-tensor docs: https://qiskit.github.io/qiskit-addon-aqc-tensor/
- Integration plan: `documentation/analysis/24_aqc_tensor_integration_plan.md`
- POC results: `results/aqc_tensor/poc_*.json`
- Cross-topology: `results/aqc_tensor/cross_topology_*.json`
- Analyzer: `python -m project_health.analysis.aqc_tensor_analyzer`
- Health report: AQC status in `python -m project_health` (Step 6b in engine.py)
- Statistical tests: `python -m project_health.analysis.aqc_tensor_analyzer --statistical`
