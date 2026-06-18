"""Test MitigationBenchmarkAnalyzer scan and validation (task 8.1)."""

import json
import tempfile
from pathlib import Path

from project_health.analysis.mitigation_benchmark_analyzer import (
    MitigationBenchmarkAnalyzer,
)


def _valid_entry(config_id="C0_raw", h_value=3.5, mode="fake_backend"):
    return {
        "benchmark_metadata": {
            "config_id": config_id,
            "h_value": h_value,
            "execution_mode": mode,
            "timestamp": "2026-06-18T10:30:00",
            "seed": 42,
        },
        "circuit_stats": {"depth_logical": 10, "depth_transpiled": 25, "n_2q_gates": 18},
        "results": {"e_raw": -12.5, "e_exact": -12.8, "delta_e_gap": 0.035},
    }


def test_scan_valid_entries():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir) / "fake_backend" / "C0_raw"
        d.mkdir(parents=True)
        for i in range(3):
            (d / f"h{325 + i * 25}.json").write_text(
                json.dumps(_valid_entry(h_value=3.25 + i * 0.25))
            )
        a = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
        a.scan()
        assert len(a.entries) == 3 and len(a.errors) == 0


def test_scan_skips_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "manifest.json").write_text("[]")
        d = Path(tmpdir) / "fake_backend" / "C0_raw"
        d.mkdir(parents=True)
        (d / "h350.json").write_text(json.dumps(_valid_entry()))
        a = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
        a.scan()
        assert len(a.entries) == 1 and len(a.errors) == 0
