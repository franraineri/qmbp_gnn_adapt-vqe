"""Integration test: verify pre-submission improvements work end-to-end."""

import json
import tempfile
from pathlib import Path

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qmbp_simulation.execution.backends import MitigationOptions
from qmbp_simulation.execution.hardware.backend import HardwareBackend
from qmbp_simulation.execution.hardware.config import HardwareConfig


def test_pre_submission_flow():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = HardwareConfig(
            mode="fake_backend",
            n_qubits=4,
            shots=1024,
            n_layouts=1,
            n_candidates=5,
            output_dir=tmpdir,
            mitigation=MitigationOptions(
                dd_enabled=True,
                twirling_enabled=True,
                trex_enabled=True,
                zne_enabled=True,
                zne_amplifier="gate_folding",
            ),
        )
        backend = HardwareBackend(config=config)

        # Build a simple 4-qubit circuit
        theta = Parameter("t")
        qc = QuantumCircuit(4)
        qc.h(range(4))
        qc.rzz(theta, 0, 1)
        qc.rzz(theta, 1, 2)
        qc.rzz(theta, 2, 3)
        qc.rx(theta, range(4))

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

        params = np.array([0.3])
        result = backend.run_deployment(
            circuit=qc,
            hamiltonian=H,
            params=params,
            h_value=4.0,
            e_exact=-3.5,
            gap=2.0,
            expected_label="paramagnetic",
        )

        # Verify output files
        output_path = Path(tmpdir)
        files = list(output_path.rglob("*"))

        # Check for pre-submission manifest
        manifests = [f for f in files if "pre_submission_manifest" in f.name]
        print(f"  Pre-submission manifests: {len(manifests)}")
        assert len(manifests) >= 1, "No pre-submission manifest found!"

        # Check for QASM files
        qasm_files = [f for f in files if f.suffix == ".qasm"]
        print(f"  QASM circuit files: {len(qasm_files)}")

        # Check for timestamped PNG files
        png_files = [f for f in files if f.suffix == ".png"]
        print(f"  Circuit diagram PNGs: {len(png_files)}")

        # Verify manifest content
        manifest = json.loads(manifests[0].read_text())
        assert "validation" in manifest
        assert "transpiled_quality_per_layout" in manifest["validation"]
        assert "calibration_snapshot" in manifest
        assert "execution_target" in manifest
        assert manifest["execution_target"]["h_value"] == 4.0
        assert "circuit_fingerprints" in manifest["layouts"]
        print("  Manifest contains all required sections")

        # Check the transpiled quality checks ran
        quality = manifest["validation"]["transpiled_quality_per_layout"]
        assert len(quality) >= 1
        eb = quality[0].get("error_budget")
        print(f"  Transpiled quality check: error_budget={eb}")

        # Verify circuit_zne_check
        zne_check = manifest["validation"]["circuit_zne_check"]
        assert "two_qubit_gate_count" in zne_check
        print(f"  ZNE check: 2Q gates={zne_check['two_qubit_gate_count']}")

        print(f"\n  Result: verdict={result.verdict}, dE/gap={result.delta_e_gap:.4f}")
        print("\n✅ ALL PRE-SUBMISSION IMPROVEMENTS VERIFIED")


if __name__ == "__main__":
    test_pre_submission_flow()
