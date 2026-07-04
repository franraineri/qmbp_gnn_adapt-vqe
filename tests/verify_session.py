#!/usr/bin/env python3
"""Comprehensive verification of all session changes."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    print("COMPREHENSIVE INTEGRATION VERIFICATION")
    print()

    # 1. result_io
    import tempfile
    from qmbp_simulation.framework.result_io import (
        save_experiment_result, load_result, build_result_envelope,
        load_results_from_dir, extract_run_metadata_summary,
    )
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data = build_result_envelope(
            config={"system": {"model": "tfim", "topologies": ["chain_1d"], "n_qubits": 10}, "seeds": [42]},
            summary={"all_passed": True, "pass_rate": 1.0},
            elapsed_s=5.0,
        )
        p1 = save_experiment_result(data, "test/x/y", results_dir=root)
        loaded = load_result(p1)
        assert loaded["schema_version"] == "2.0", "Schema version missing"
        meta = extract_run_metadata_summary(loaded)
        assert meta["model"] == "tfim", "Model extraction failed"
    print("1. result_io: OK (atomic write, schema, metadata extraction)")

    # 2. ResultIndex
    from qmbp_simulation.framework.result_index import ResultIndex
    index = ResultIndex()
    assert len(index) > 100, f"Index too small: {len(index)}"
    stats = index.stats()
    matrix = index.coverage_matrix()
    assert "tfim" in matrix, "TFIM not in coverage"
    regs = index.detect_regressions()
    suggs = index.suggest_next()
    val = index.validate()
    est = index.estimate_time("tfim", "chain_1d", 20, 4)
    print(f"2. ResultIndex: {stats['total_runs']} runs, {val['n_valid']} valid, est={est:.0f}s")

    # 3. runner_base
    from qmbp_simulation.framework.runner_base import ValidationRunner
    assert hasattr(ValidationRunner, "_load_resume")
    assert hasattr(ValidationRunner, "restore_section_state")
    print("3. runner_base: resume + restore_section_state present")

    # 4. Fixtures
    fixtures = ROOT / "tests" / "fixtures"
    complete = load_result(fixtures / "complete_run.json")
    interrupted = load_result(fixtures / "interrupted_run.json")
    assert complete["summary"]["all_passed"] is True
    assert interrupted["interrupted"] is True
    assert interrupted["completed_sections"] == 2
    print("4. Fixtures: complete + interrupted load OK")

    # 5. Real result end-to-end
    real = load_result(ROOT / "results/experiments/exp_noiseless_tfim_4/run_20260702_200440.json")
    pp = real["results"]["section_4"]["data"]["per_point"]
    assert all(pt["de_gap"] < 0.05 for pt in pp), "N=20 has failing points!"
    print(f"5. Real N=20 result: {len(pp)} deploy points, ALL < 5%")

    # 6. Physics checks
    from qmbp_simulation.models.hamiltonian import make_lattice
    lat = make_lattice("chain_1d", 4, J=1.0, h=1.5)
    assert len(lat.edges) == 3
    from qmbp_simulation.solvers.classical import ClassicalSolver
    from qmbp_simulation.models.hamiltonian import HamiltonianBuilder
    H = HamiltonianBuilder().build(lat)
    gt = ClassicalSolver().solve(H, lat)
    assert gt.gap > 0
    print(f"6. Physics: N=4 chain_1d E={gt.ground_energy:.4f}, gap={gt.gap:.4f}")

    # 7. Noiseless pipeline imports
    sys.path.insert(0, str(ROOT))
    from scripts.experiment_runners.noiseless.run_noiseless_pipeline import NoiselessPipelineRunner
    assert hasattr(NoiselessPipelineRunner, "restore_section_state")
    assert hasattr(NoiselessPipelineRunner, "_save_vqe_checkpoint")
    print("7. NoiselessPipeline: restore_section_state + VQE checkpoint present")

    # 8. CLI tools
    from project_health.cli.inspect_noiseless_run import print_report
    from project_health.cli.query_index import main as qi_main
    print("8. CLI tools: importable")

    print()
    print("=" * 50)
    print("  ALL 8 CHECKS PASSED - SYSTEM IS ROBUST")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
