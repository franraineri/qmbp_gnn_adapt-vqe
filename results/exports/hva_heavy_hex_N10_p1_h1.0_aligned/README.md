# HVA bond-resolved circuit export — heavy_hex N=10 p=1 h=1.0

TFIM bond-resolved Hardware-Efficient Variational Ansatz (HVA), 19 parameters
(9 per-bond RZZ + 10 per-site RX), for the heavy_hex lattice.

## Files

| File | What it is | Needs |
|------|-----------|-------|
| `circuit_bound.qasm3` | OpenQASM 3, GNN angles already assigned | qiskit + `qiskit_qasm3_import` |
| `circuit_bound.qpy` | Qiskit QPY, GNN angles assigned | Qiskit (full fidelity) |
| `circuit_parametric.qpy` | Qiskit QPY, 19 symbolic θ | Qiskit, re-optimizable |
| `theta_gnn.json` | the 19 predicted angles + parameter order | — |
| `metadata.json` | topology, edges, energies, versions | — |
| `recreate.py` | standalone rebuilder (Qiskit-only, no repo needed) | qiskit, numpy |
| `README.md` | this file | — |
| `circuit_native.qasm3` | OpenQASM 3, transpiled to Heron basis | qiskit + `qiskit_qasm3_import` |
| `circuit_native.qpy` | Qiskit QPY, transpiled to Heron basis | Qiskit |

## Requirements

```bash
pip install "qiskit>=1.0" numpy
pip install qiskit_qasm3_import   # ONLY to LOAD the .qasm3 file (see Option A)
```

Note: base Qiskit can *write* QASM3 but cannot *read* it — loading needs the
extra `qiskit_qasm3_import` package. If you'd rather not install it, use the QPY
file (Option B) or `recreate.py` (Option C), which need only qiskit + numpy.

## Quick start

### Option A — load the QASM3 (most portable across frameworks)

```python
from qiskit.qasm3 import load     # requires: pip install qiskit_qasm3_import
qc = load("circuit_bound.qasm3")   # angles already baked in
```

### Option B — load the QPY (Qiskit-native, exact)

```python
from qiskit.qpy import load
with open("circuit_bound.qpy", "rb") as f:
    qc = load(f)[0]
```

### Option C — rebuild from scratch (no QASM/QPY, just code)

```bash
python recreate.py --draw
```

`recreate.py` hardcodes the lattice edges and gate pattern, then reads
`theta_gnn.json` to bind the angles. Use this if QASM/QPY versions mismatch.

### Put your own angles on the parametric circuit

```python
import json, numpy as np
from qiskit.qpy import load
with open("circuit_parametric.qpy", "rb") as f:
    qc = load(f)[0]                          # 19 free parameters
angles = np.array(json.load(open("theta_gnn.json"))["theta"])
bound = qc.assign_parameters(dict(zip(qc.parameters, angles)))
```

## Parameter convention (important)

Order is `[θ_zz_0 … θ_zz_{E-1}, θ_x_0 … θ_x_{N-1}]`. Gates apply `RZZ(2·θ_zz_k)`
on each edge k and `RX(2·θ_x_i)` on each qubit i. Initial state is |+⟩^N (H on all
qubits). Hamiltonian: TFIM H = −J·ΣZZ − h·ΣX with J=1, h=1.0.

`theta_gnn.json` stores the **raw θ**. The QASM3 file shows the **materialized
`2·θ`** as the gate argument (e.g. θ_zz[0]=0.2839 → `rzz(0.5678)`).
Both describe the same circuit — don't double the QASM angles again.

## Sanity check (reproduce these numbers)

Build the TFIM Hamiltonian H = −ΣZZ − h·ΣX and evaluate ⟨H⟩ on the bound circuit:

| Quantity | Energy (h=1.0) |
|----------|----------------|
| Ground state (exact diag.) | -12.47217 |
| GNN raw prediction (this bundle's angles) | -11.83023 |
| θ=0 reference (|+⟩^N = −h·N) | -10.0 |

If your loaded circuit gives ⟨H⟩ = -11.83023 you reconstructed it correctly.

## Compressed circuit (native IBM Heron basis)

`circuit_native.qasm3` / `circuit_native.qpy` are the same bound circuit
transpiled to the Heron native gate set `['cz', 'rz', 'sx', 'x']`
(optimization_level=3, seed_transpiler=42). Each logical `RZZ` becomes 2 `CZ`.

| | depth | 2q-depth | 2q-gates | gate counts |
|---|-------|----------|----------|-------------|
| logical | 9 | 7 | 9 | {'h': 10, 'rx': 10, 'rzz': 9} |
| native (Heron) | 57 | 14 | 18 | {'rz': 59, 'sx': 57, 'cz': 18, 'x': 7} |

The logical and native circuits are equivalent unitaries (up to transpiler
optimization); the native one reflects what actually runs on hardware.

Generated 2026-09-30T20:59:43.733983+00:00 · qiskit 2.2.3
