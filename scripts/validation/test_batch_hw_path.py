"""Test the batch hardware path with FakeTorino as mock backend."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import os

os.environ["BENCHMARK_BACKEND"] = "fake"  # won't be used

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeTorino

from qmbp_simulation import HamiltonianBuilder, make_lattice
from scripts.experiment_runners.hardware.benchmark_configs import BENCHMARK_CONFIGS
from scripts.experiment_runners.hardware.run_mitigation_benchmark import (
    _build_hva_circuit,
    _execute_hardware_batched,
)

backend = FakeTorino()
configs = ["C0_raw", "C5_full_pea_balanced", "C3_full_gf"]
h_values = [4.0]
seed = 42
shots = 1024

# Build jobs_spec
jobs_spec = []
for h in h_values:
    circuit_hva = _build_hva_circuit(h)
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend, seed_transpiler=seed)
    transpiled = pm.run(circuit_hva)
    lattice_h = make_lattice("heavy_hex", 10, J=1.0, h=h)
    H = HamiltonianBuilder().build(lattice_h)
    H_mapped = H.apply_layout(transpiled.layout)

    for config_id in configs:
        config = BENCHMARK_CONFIGS[config_id]
        if config.zne_method not in (None, "gf", "pea"):
            continue
        jobs_spec.append((config, transpiled, H_mapped, h))

print(f"Built {len(jobs_spec)} jobs_spec entries")
print("Attempting _execute_hardware_batched with FakeTorino...")

try:
    results = _execute_hardware_batched(jobs_spec, backend, shots)
    print(f"\nResults: {len(results)}")
    for i, r in enumerate(results):
        cfg = jobs_spec[i][0].config_id
        e = r.get("e_raw") or r.get("e_mitigated")
        err = r.get("error")
        if err:
            print(f"  {cfg}: ERROR — {err}")
        else:
            print(f"  {cfg}: E={e:.4f}")
except Exception as e:
    print(f"\nCRASH: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
