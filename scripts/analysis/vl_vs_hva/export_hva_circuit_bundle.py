#!/usr/bin/env python
"""Export a portable bundle of the bond-resolved HVA circuit + GNN angles.

Produces a self-contained folder another user can use to recreate the exact
circuit on their machine, with or without this repository:

    results/exports/hva_heavy_hex_N10_p1_h1.0/
      ├── circuit_bound.qasm3      OpenQASM 3 with GNN angles assigned (portable)
      ├── circuit_bound.qpy        Qiskit QPY, GNN angles assigned (full fidelity)
      ├── circuit_parametric.qpy   Qiskit QPY, 19 symbolic θ (re-optimizable)
      ├── theta_gnn.json           the 19 predicted angles + parameter order
      ├── metadata.json            topology, edges, N, p, h, energies, versions
      ├── recreate.py              standalone rebuilder (Qiskit-only, no repo)
      └── README.md                how to use each file

The circuit is the bond-resolved HVA (create_bond_resolved): per-edge RZZ(2·θ_zz)
and per-site RX(2·θ_x). Parameter order is [θ_zz_0..θ_zz_{E-1}, θ_x_0..θ_x_{N-1}].

Usage:
    .venv/bin/python scripts/analysis/export_hva_circuit_bundle.py
    .venv/bin/python scripts/analysis/export_hva_circuit_bundle.py --h 2.5 --no-angles
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

import qiskit
from qmbp_simulation.framework.artifact_serializers import get_serializer
from qmbp_simulation.framework.runner_base import (
    Section,
    ValidationRunner,
    resolve_project_root,
)
from qmbp_simulation.predictors.unified_graph import build_graph_for_model

logging.basicConfig(level=logging.ERROR)

TOPOLOGY = "heavy_hex"
N_QUBITS = 10
P_LAYERS = 1
MODEL = "tfim_bond_resolved"
# signinv_fid_v1 is the physically-correct checkpoint for heavy_hex N=10 p=1
# (the zoo auto-selector picks a numerically broken multi-N checkpoint — see
# memory heavy-hex-n10-p1-model-selection).
CHECKPOINT = "data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_signinv_fid_v1.pt"
# Move-resilient repo-root resolution (never count parents[N] — walks up for
# pyproject.toml / Makefile via the framework helper).
REPO_ROOT = resolve_project_root(__file__)


class _ExportRunner(ValidationRunner):
    runner_id = "export_bundle"
    experiment_id = "EXPORT_BUNDLE"
    description = "Export HVA circuit bundle"
    hypothesis = "n/a"

    def define_sections(self):
        return [Section(id=1, name="noop", fn=lambda: {}, hypothesis="noop")]


def _make_args(h: float) -> argparse.Namespace:
    return argparse.Namespace(
        n_qubits=N_QUBITS, p_layers=P_LAYERS, topology=[TOPOLOGY], model=MODEL,
        h_min=h, h_max=h, h_points=1, seeds=[42], maxiter=100, n_restarts=1,
        verbose=False, section=None, dry_run=False, skip_preflight=False,
        stop_on_failure=False, validate_vqe=False, validate_theta=False,
        theta_validation_level=0, strict_validation=False, resume=None,
        save_artifacts="never", no_bidirectional=False, force_bidirectional=False,
        preset=None, output=None, model_params=None,
    )


def _predict_theta(model, lattice, h: float, n_params: int) -> np.ndarray:
    g = build_graph_for_model(model, lattice, h_value=h, p_layers=P_LAYERS)
    model.eval()
    with torch.no_grad():
        theta = model(g).cpu().numpy().flatten()
    if len(theta) < n_params:
        theta = np.pad(theta, (0, n_params - len(theta)))
    elif len(theta) > n_params:
        theta = theta[:n_params]
    return theta


def _param_names(edges: list[tuple[int, int]], n_qubits: int) -> list[str]:
    names = [f"theta_zz[{k}] (edge {i}-{j})" for k, (i, j) in enumerate(edges)]
    names += [f"theta_x[{i}] (qubit {i})" for i in range(n_qubits)]
    return names


RECREATE_TEMPLATE = '''#!/usr/bin/env python
"""Standalone rebuilder for the bond-resolved HVA circuit (Qiskit-only).

No project dependency: this recreates the SAME circuit as the exporter using
only `qiskit` and `numpy`. Run it to get the Qiskit QuantumCircuit and (if
angles are provided) the bound circuit ready to simulate.

    python recreate.py            # build parametric + bound, print summary
    python recreate.py --draw     # also print an ASCII drawing

Requires: qiskit>=1.0, numpy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# ── Fixed topology (heavy_hex N={N_QUBITS}) ─────────────────────────────────
N_QUBITS = {N_QUBITS}
P_LAYERS = {P_LAYERS}
EDGES = {edges!r}  # {n_edges} bonds
N_EDGES = len(EDGES)
N_PARAMS = (N_EDGES + N_QUBITS) * P_LAYERS  # = {n_params}


def build_parametric() -> tuple[QuantumCircuit, ParameterVector]:
    """Build the bond-resolved HVA circuit with symbolic parameters.

    Layout per layer: [theta_zz_0..theta_zz_{{E-1}}, theta_x_0..theta_x_{{N-1}}].
    Gates: RZZ(2*theta_zz_k) per edge, RX(2*theta_x_i) per qubit.
    """
    theta = ParameterVector("theta", N_PARAMS)
    qc = QuantumCircuit(N_QUBITS)
    qc.h(range(N_QUBITS))  # |+>^N initial state
    params_per_layer = N_EDGES + N_QUBITS
    for layer in range(P_LAYERS):
        off = layer * params_per_layer
        for k, (i, j) in enumerate(EDGES):
            qc.rzz(2 * theta[off + k], i, j)
        for i in range(N_QUBITS):
            qc.rx(2 * theta[off + N_EDGES + i], i)
    return qc, theta


def build_bound(angles: np.ndarray) -> QuantumCircuit:
    """Build the circuit with concrete angles assigned."""
    qc, theta = build_parametric()
    angles = np.asarray(angles, dtype=float).flatten()
    if angles.shape[0] != N_PARAMS:
        raise ValueError(f"expected {{N_PARAMS}} angles, got {{angles.shape[0]}}")
    return qc.assign_parameters({{theta: angles}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", action="store_true", help="print ASCII drawing")
    args = ap.parse_args()

    qc, _ = build_parametric()
    print(f"Parametric circuit: {{qc.num_qubits}} qubits, "
          f"{{qc.num_parameters}} params, depth {{qc.depth()}}")
    print("Gate counts:", dict(qc.count_ops()))

    theta_path = Path(__file__).with_name("theta_gnn.json")
    if theta_path.exists():
        data = json.loads(theta_path.read_text())
        angles = np.array(data["theta"], dtype=float)
        bound = build_bound(angles)
        print(f"\\nBound circuit built with {{len(angles)}} GNN angles.")
        if args.draw:
            print(bound.draw(output="text", fold=100))
        # Optional: evaluate energy if qiskit has the estimator + you build H
    else:
        print("\\n(theta_gnn.json not found — parametric circuit only)")
        if args.draw:
            print(qc.draw(output="text", fold=100))


if __name__ == "__main__":
    main()
'''


README_TEMPLATE = """# HVA bond-resolved circuit export — {topology} N={n} p={p} h={h}

TFIM bond-resolved Hardware-Efficient Variational Ansatz (HVA), {n_params} parameters
({n_edges} per-bond RZZ + {n} per-site RX), for the {topology} lattice.

## Files

| File | What it is | Needs |
|------|-----------|-------|
| `circuit_bound.qasm3` | OpenQASM 3, GNN angles already assigned | qiskit + `qiskit_qasm3_import` |
| `circuit_bound.qpy` | Qiskit QPY, GNN angles assigned | Qiskit (full fidelity) |
| `circuit_parametric.qpy` | Qiskit QPY, {n_params} symbolic θ | Qiskit, re-optimizable |
| `theta_gnn.json` | the {n_params} predicted angles + parameter order | — |
| `metadata.json` | topology, edges, energies, versions | — |
| `recreate.py` | standalone rebuilder (Qiskit-only, no repo needed) | qiskit, numpy |
| `README.md` | this file | — |

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
    qc = load(f)[0]                          # {n_params} free parameters
angles = np.array(json.load(open("theta_gnn.json"))["theta"])
bound = qc.assign_parameters(dict(zip(qc.parameters, angles)))
```

## Parameter convention (important)

Order is `[θ_zz_0 … θ_zz_{{E-1}}, θ_x_0 … θ_x_{{N-1}}]`. Gates apply `RZZ(2·θ_zz_k)`
on each edge k and `RX(2·θ_x_i)` on each qubit i. Initial state is |+⟩^N (H on all
qubits). Hamiltonian: TFIM H = −J·ΣZZ − h·ΣX with J=1, h={h}.

`theta_gnn.json` stores the **raw θ**. The QASM3 file shows the **materialized
`2·θ`** as the gate argument (e.g. θ_zz=0.0527 → `rzz(0.1055)`). Both describe the
same circuit — don't double the QASM angles again.

## Sanity check (reproduce these numbers)

Build the TFIM Hamiltonian H = −ΣZZ − h·ΣX and evaluate ⟨H⟩ on the bound circuit:

| Quantity | Energy (h={h}) |
|----------|----------------|
| Ground state (exact diag.) | {e_exact:.5f} |
| GNN raw prediction (this bundle's angles) | {e_gnn} |
| θ=0 reference (|+⟩^N = −h·N) | {e_zero:.1f} |

If your loaded circuit gives ⟨H⟩ = {e_gnn} you reconstructed it correctly.

Generated {generated} · qiskit {qiskit_version}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--no-angles", action="store_true",
                    help="export parametric circuit only (skip GNN prediction)")
    ap.add_argument("--out", type=str, default=None,
                    help="output dir (default results/exports/hva_<topo>_N<n>_p<p>_h<h>)")
    args = ap.parse_args()
    h = float(args.h)

    runner = _ExportRunner(_make_args(h))
    runner.setup_physics()

    lattice = runner.make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=h)
    edges = [tuple(int(x) for x in e) for e in lattice.edges]
    n_edges = len(edges)
    qc_param, theta_vec = runner.hva.create_bond_resolved(N_QUBITS, P_LAYERS, lattice)
    n_params = qc_param.num_parameters

    out_dir = Path(args.out) if args.out else (
        REPO_ROOT / "results" / "exports" / f"hva_{TOPOLOGY}_N{N_QUBITS}_p{P_LAYERS}_h{h}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    qpy = get_serializer("qpy")
    qasm3 = get_serializer("qasm3")
    js = get_serializer("json")

    # Parametric circuit (symbolic θ)
    qpy.save(qc_param, out_dir / "circuit_parametric.qpy")

    # Metadata scaffold
    e_exact, gap = runner.exact_ground_state(TOPOLOGY, N_QUBITS, h, model=MODEL)
    metadata = {
        "topology": TOPOLOGY,
        "n_qubits": N_QUBITS,
        "p_layers": P_LAYERS,
        "model": MODEL,
        "h": h,
        "J": 1.0,
        "ansatz": "bond_resolved_hva",
        "n_parameters": n_params,
        "n_edges": n_edges,
        "edges": edges,
        "parameter_order": _param_names(edges, N_QUBITS),
        "gate_convention": "RZZ(2*theta_zz_k) per edge; RX(2*theta_x_i) per qubit; init |+>^N",
        "hamiltonian": "H = -J * sum_edges Z_i Z_j - h * sum_i X_i",
        "e_exact": float(e_exact),
        "gap": float(gap),
        "qiskit_version": qiskit.__version__,
        "generated_utc": datetime.now(UTC).isoformat(),
    }

    angles = None
    if not args.no_angles:
        model = runner.load_best_mpnn_for_cross_n(
            n_target=N_QUBITS, model=MODEL, topology=TOPOLOGY, p_layers=P_LAYERS,
            checkpoint_path=CHECKPOINT, train_if_missing=False,
        )
        if model is None:
            raise RuntimeError(f"Failed to load checkpoint {CHECKPOINT}")
        angles = _predict_theta(model, lattice, h, n_params)

        H = runner.builder.build(lattice)
        e_gnn = float(runner.noiseless.evaluate(qc_param, H, angles))
        fid = runner.safe_compute_fidelity(qc_param, angles, TOPOLOGY, N_QUBITS, h, model=MODEL)

        bound = qc_param.assign_parameters(dict(zip(qc_param.parameters, angles)))
        qpy.save(bound, out_dir / "circuit_bound.qpy")
        try:
            qasm3.save(bound, out_dir / "circuit_bound.qasm3")
        except Exception as exc:  # QASM3 export can fail on some gates
            (out_dir / "circuit_bound.qasm3").write_text(
                f"// QASM3 export failed: {exc}\n// Use circuit_bound.qpy instead.\n"
            )

        js.save(
            {
                "theta": [float(x) for x in angles],
                "parameter_order": _param_names(edges, N_QUBITS),
                "checkpoint": CHECKPOINT.split("/")[-1],
                "note": "raw GNN-predicted angles, no VQE refinement",
            },
            out_dir / "theta_gnn.json",
        )
        metadata["e_gnn"] = e_gnn
        metadata["abs_error"] = abs(e_gnn - e_exact)
        metadata["de_gap"] = abs(e_gnn - e_exact) / max(gap, 1e-10)
        metadata["fidelity"] = float(fid) if fid is not None else None
        metadata["checkpoint"] = CHECKPOINT.split("/")[-1]

    js.save(metadata, out_dir / "metadata.json")

    # Standalone recreate.py (no repo dependency)
    recreate = RECREATE_TEMPLATE.format(
        N_QUBITS=N_QUBITS, P_LAYERS=P_LAYERS, edges=edges, n_edges=n_edges, n_params=n_params
    )
    (out_dir / "recreate.py").write_text(recreate, encoding="utf-8")

    e_zero = -h * N_QUBITS
    e_gnn_str = f"{metadata['e_gnn']:.5f}" if "e_gnn" in metadata else "n/a (--no-angles)"
    (out_dir / "README.md").write_text(
        README_TEMPLATE.format(
            topology=TOPOLOGY, n=N_QUBITS, p=P_LAYERS, h=h, n_params=n_params,
            n_edges=n_edges, generated=metadata["generated_utc"],
            qiskit_version=qiskit.__version__,
            e_exact=metadata["e_exact"], e_gnn=e_gnn_str, e_zero=e_zero,
        ),
        encoding="utf-8",
    )

    print(f"Bundle written to: {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name:26s} {f.stat().st_size:>8d} bytes")
    if angles is not None:
        print(
            f"\nGNN angles baked in. E_gnn={metadata['e_gnn']:.5f} "
            f"E_exact={metadata['e_exact']:.5f} |dE|={metadata['abs_error']:.4f} "
            f"dE/gap={metadata['de_gap']:.4f} F={metadata['fidelity']:.4f}"
        )


if __name__ == "__main__":
    main()
