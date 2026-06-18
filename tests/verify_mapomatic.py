#!/usr/bin/env python3
"""Verify full mapomatic layout optimizer integration end-to-end.

Checks all 8 integration points:
1. Module imports (all public API)
2. HardwareConfig fields (6 mapomatic fields)
3. submission.py integration (select_layouts_for_hardware)
4. Full pipeline (FakeTorino N=10 VF2 vs BFS)
5. Structured log events (method field logged)
6. Analyzer (project_health)
7. Sanity check (hardware_readiness)
8. Benchmark component (BenchmarkSuite)
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np


def main():
    print("=" * 60)
    print("  MAPOMATIC INTEGRATION VERIFICATION")
    print("=" * 60)
    errors = []

    # TEST 1
    print("\n=== TEST 1: Module imports ===")
    try:
        from qmbp_simulation.execution.hardware import (
            MAPOMATIC_AVAILABLE,
            HardwareConfig,
        )

        print(f"  All imports OK. MAPOMATIC_AVAILABLE={MAPOMATIC_AVAILABLE}")
    except Exception as e:
        errors.append(f"TEST 1 FAILED: {e}")
        print(f"  FAIL: {e}")
        return

    # TEST 2
    print("\n=== TEST 2: HardwareConfig fields ===")
    config = HardwareConfig()
    fields = [
        "use_mapomatic",
        "layout_max_2q_error",
        "layout_min_t1_us",
        "layout_call_limit",
        "layout_exclude_qubits",
        "layout_strategy",
    ]
    for f in fields:
        val = getattr(config, f, "MISSING")
        print(f"  {f} = {val}")
        if val == "MISSING":
            errors.append(f"TEST 2: Missing field {f}")

    # TEST 3
    print("\n=== TEST 3: submission.py integration ===")
    try:
        from qmbp_simulation.execution.hardware.submission import select_layouts_for_hardware

        print("  select_layouts_for_hardware imported OK")
    except Exception as e:
        errors.append(f"TEST 3 FAILED: {e}")
        print(f"  FAIL: {e}")

    # TEST 4
    print("\n=== TEST 4: Full pipeline (FakeTorino N=10) ===")
    try:
        from qiskit.circuit import QuantumCircuit
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        from qmbp_simulation.framework.logging import StructuredLogger

        backend = FakeTorino()
        logger = StructuredLogger("test")

        qc = QuantumCircuit(10)
        for i in range(9):
            qc.cz(i, i + 1)
        for i in range(10):
            qc.rx(0.5, i)

        # VF2 path
        config_vf2 = HardwareConfig(
            n_qubits=10,
            n_layouts=3,
            use_mapomatic=True,
            max_ces=1.0,
            layout_max_2q_error=0.02,
        )
        result_vf2 = select_layouts_for_hardware(qc, backend, config_vf2, logger)
        print(
            f"  VF2: {len(result_vf2.layouts)} layouts, "
            f"CES={[round(c, 4) for c in result_vf2.ces_values]}"
        )

        # BFS path
        config_bfs = HardwareConfig(
            n_qubits=10,
            n_layouts=3,
            use_mapomatic=False,
            max_ces=1.0,
        )
        result_bfs = select_layouts_for_hardware(qc, backend, config_bfs, logger)
        print(
            f"  BFS: {len(result_bfs.layouts)} layouts, "
            f"CES={[round(c, 4) for c in result_bfs.ces_values]}"
        )

        vf2_mean = float(np.mean(result_vf2.ces_values))
        bfs_mean = float(np.mean(result_bfs.ces_values))
        print(f"  VF2 mean CES={vf2_mean:.4f}, BFS mean CES={bfs_mean:.4f}")
        print(f"  Improvement: {bfs_mean / vf2_mean:.1f}x")

        assert len(result_vf2.layouts) == 3, "VF2 should return 3 layouts"
        assert len(result_vf2.transpiled_circuits) == 3, "Should have transpiled circuits"
        assert vf2_mean < bfs_mean, "VF2 should have lower CES"
    except AssertionError as e:
        errors.append(f"TEST 4 ASSERTION: {e}")
        print(f"  FAIL: {e}")
    except Exception as e:
        errors.append(f"TEST 4 FAILED: {e}")
        print(f"  FAIL: {e}")

    # TEST 5
    print("\n=== TEST 5: Structured log events ===")
    try:
        events = [e for e in logger.events if e.event_type in ("layout_selection", "layout_method")]
        for e in events:
            print(f"  {e.event_type}: {e.data}")
        has_vf2 = any(e.data.get("method") == "mapomatic_vf2" for e in events)
        if not has_vf2:
            errors.append("TEST 5: No 'mapomatic_vf2' method in log events")
            print("  FAIL: No VF2 method logged")
        else:
            print("  VF2 method correctly logged ✓")
    except Exception as e:
        errors.append(f"TEST 5 FAILED: {e}")
        print(f"  FAIL: {e}")

    # TEST 6
    print("\n=== TEST 6: Analyzer ===")
    try:
        import sys
        from pathlib import Path

        # Ensure project root is on path (same as pyproject.toml pythonpath)
        root = Path(__file__).parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from project_health.analysis.layout_optimizer_analyzer import analyze

        report = analyze(verbose=False)
        print(f"  mapomatic_available={report.mapomatic_available}")
        print(f"  Records found: {len(report.records)}")
    except Exception as e:
        errors.append(f"TEST 6 FAILED: {e}")
        print(f"  FAIL: {e}")

    # TEST 7
    print("\n=== TEST 7: Sanity check ===")
    try:
        from project_health.analysis.sanity_check import check_layout_optimizer_integration

        checks = check_layout_optimizer_integration(verbose=False)
        for c in checks:
            status = "PASS" if c.passed else "FAIL"
            print(f"  [{status}] {c.name}")
        if not all(c.passed for c in checks):
            errors.append("TEST 7: Not all sanity checks pass")
    except Exception as e:
        errors.append(f"TEST 7 FAILED: {e}")
        print(f"  FAIL: {e}")

    # TEST 8
    print("\n=== TEST 8: Benchmark component ===")
    try:
        from qmbp_simulation.framework.benchmarking import _bench_layout

        result = _bench_layout(6, 2)
        print(f"  N=6: {result.elapsed_s * 1000:.1f}ms, method={result.details['method']}")
        assert result.details["mapomatic_available"] is True
        assert result.elapsed_s > 0
        print("  Benchmark OK ✓")
    except Exception as e:
        errors.append(f"TEST 8 FAILED: {e}")
        print(f"  FAIL: {e}")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"  ❌ {len(errors)} ERROR(S) FOUND:")
        for err in errors:
            print(f"     • {err}")
    else:
        print("  ✅ ALL 8 INTEGRATION TESTS PASS")
    print("=" * 60)
    return len(errors)


if __name__ == "__main__":
    import sys

    sys.exit(main() or 0)
