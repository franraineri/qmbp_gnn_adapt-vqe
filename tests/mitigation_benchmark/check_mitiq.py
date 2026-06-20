"""Check Mitiq integration status across benchmark results."""

import json
from pathlib import Path

base = Path("results/mitigation_benchmark/fake_backend")
mitiq_configs = [
    "C11_mitiq_zne",
    "C12_mitiq_cdr",
    "C13_mitiq_ddd_zne",
    "C14_dd_mitiq_cdr",
    "C17_aqc_mitiq_cdr",
]

print("MITIQ CONFIG STATUS (from existing fake_backend results)")
print("=" * 70)

for cfg in mitiq_configs:
    cfg_dir = base / cfg
    if not cfg_dir.exists():
        print(f"  {cfg}: NO DIRECTORY")
        continue
    files = sorted(cfg_dir.glob("*.json"))
    valid = 0
    errors = 0
    error_msgs = set()
    sample_de = None

    for f in files:
        try:
            d = json.loads(f.read_text())
            r = d.get("results", {})
            if r.get("e_raw") is not None or r.get("e_mitigated") is not None:
                valid += 1
                if sample_de is None and r.get("delta_e_gap") is not None:
                    h = d.get("benchmark_metadata", {}).get("h_value", "?")
                    sample_de = f"h={h}, dE/gap={r['delta_e_gap']:.4f}"
            else:
                errors += 1
        except Exception:
            errors += 1

    status = "✓ OK" if valid > 0 and errors == 0 else ("⚠ PARTIAL" if valid > 0 else "✗ BROKEN")
    print(f"  {cfg:<25s} valid={valid:>3d} errors={errors:>3d} [{status}]")
    if sample_de:
        print(f"    Sample: {sample_de}")

# Quick functional test of Mitiq ZNE (the one that works)
print("\n" + "=" * 70)
print("MITIQ FUNCTIONAL TEST (ZNE on 4-qubit circuit)")
print("=" * 70)

try:
    from qmbp_simulation.execution.mitiq_utils import (
        is_mitiq_available,
        run_mitiq_zne,
    )

    print(f"  mitiq available: {is_mitiq_available()}")

    if is_mitiq_available():
        from qiskit.circuit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime.fake_provider import FakeKingston

        from qmbp_simulation.execution.noisy_utils import NoisyEstimatorConfig

        # Build and transpile a test circuit
        qc = QuantumCircuit(4)
        qc.h(range(4))
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.rz(0.5, range(4))

        H = SparsePauliOp.from_list(
            [
                ("ZZII", -1.0),
                ("IZZI", -1.0),
                ("IIZZ", -1.0),
                ("XIII", -0.5),
                ("IXII", -0.5),
                ("IIXI", -0.5),
                ("IIIX", -0.5),
            ]
        )

        backend = FakeKingston()
        pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
        transpiled = pm.run(qc)
        H_mapped = H.apply_layout(transpiled.layout)
        config = NoisyEstimatorConfig(shots=4096, seed_simulator=42)

        result = run_mitiq_zne(
            transpiled,
            H_mapped,
            backend,
            config,
            scale_factors=(1.0, 2.0, 3.0),
        )

        print(f"  ZNE result: e_mitigated={result.mitigated_value:.4f}")
        print(f"  ZNE unmitigated: {result.unmitigated_value:.4f}")
        print(f"  Folding method: {result.folding_method}")
        print("  ✓ Mitiq ZNE functional on noisy simulation")
except Exception as e:
    print(f"  ✗ Mitiq test FAILED: {e}")
