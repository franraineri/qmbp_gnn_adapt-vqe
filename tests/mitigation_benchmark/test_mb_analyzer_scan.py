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
        "circuit_stats": {"depth_logical": 10, "depth_transpiled": 25},
        "results": {"e_raw": -12.5, "e_exact": -12.8, "delta_e_gap": 0.035},
    }


def test_scan_valid_entries():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir) / "fake_backend" / "C0_raw"
        d.mkdir(parents=True)
        for i in range(3):
            p = d / f"h{325 + i * 25}.json"
            p.write_text(json.dumps(_valid_entry(h_value=3.25 + i * 0.25)))
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


def test_scan_skips_configs_and_analysis():
    with tempfile.TemporaryDirectory() as tmpdir:
        for sub in ("configs", "analysis"):
            sd = Path(tmpdir) / sub
            sd.mkdir(parents=True)
            (sd / "f.json").write_text(json.dumps({"data": 1}))
        a = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
        a.scan()
        assert len(a.entries) == 0 and len(a.errors) == 0


def test_validate_missing_sections():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir) / "fake_backend" / "C0_raw"
        d.mkdir(parents=True)
        bad = {
            "benchmark_metadata": {
                "config_id": "C0_raw",
                "h_value": 3.5,
                "execution_mode": "fake_backend",
            },
            "circuit_stats": {"depth_logical": 10},
        }
        (d / "h350.json").write_text(json.dumps(bad))
        a = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
        a.scan()
        assert len(a.entries) == 0 and len(a.errors) == 1
        assert "Missing required sections" in a.errors[0]["reason"]


def test_validate_missing_metadata_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir) / "fake_backend" / "C0_raw"
        d.mkdir(parents=True)
        bad = {
            "benchmark_metadata": {"config_id": "C0_raw", "h_value": 3.5},
            "circuit_stats": {},
            "results": {"e_raw": -12.5, "delta_e_gap": 0.035},
        }
        (d / "h350.json").write_text(json.dumps(bad))
        a = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
        a.scan()
        assert len(a.entries) == 0 and len(a.errors) == 1
        assert "Missing metadata keys" in a.errors[0]["reason"]


def test_validate_missing_energy():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir) / "fake_backend" / "C0_raw"
        d.mkdir(parents=True)
        bad = {
            "benchmark_metadata": {
                "config_id": "C0_raw",
                "h_value": 3.5,
                "execution_mode": "fake_backend",
            },
            "circuit_stats": {},
            "results": {"delta_e_gap": 0.035},
        }
        (d / "h350.json").write_text(json.dumps(bad))
        a = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
        a.scan()
        assert len(a.entries) == 0 and len(a.errors) == 1
        assert "No energy metric" in a.errors[0]["reason"]


def test_validate_corrupt_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir) / "fake_backend" / "C0_raw"
        d.mkdir(parents=True)
        (d / "h350.json").write_text("{broken")
        a = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
        a.scan()
        assert len(a.entries) == 0 and len(a.errors) == 1
        assert "Failed to parse JSON" in a.errors[0]["reason"]


def test_validate_non_dict_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir) / "fake_backend" / "C0_raw"
        d.mkdir(parents=True)
        (d / "h350.json").write_text(json.dumps([1, 2, 3]))
        a = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
        a.scan()
        assert len(a.entries) == 0 and len(a.errors) == 1
        assert "not a dict" in a.errors[0]["reason"]


def test_scan_nonexistent_dir():
    a = MitigationBenchmarkAnalyzer(results_dir=Path("/nonexistent"))
    a.scan()
    assert len(a.entries) == 0 and len(a.errors) == 0


def test_mixed_valid_and_invalid():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir) / "fake_backend" / "C0_raw"
        d.mkdir(parents=True)
        for i in range(2):
            (d / f"v{i}.json").write_text(json.dumps(_valid_entry()))
        bad = {
            "benchmark_metadata": {
                "config_id": "C0",
                "h_value": 4.0,
                "execution_mode": "fake_backend",
            },
            "circuit_stats": {},
        }
        (d / "invalid.json").write_text(json.dumps(bad))
        (d / "corrupt.json").write_text("{bad")
        a = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
        a.scan()
        assert len(a.entries) == 2 and len(a.errors) == 2


def test_e_mitigated_only_is_valid():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir) / "fake_backend" / "C5"
        d.mkdir(parents=True)
        entry = {
            "benchmark_metadata": {
                "config_id": "C5",
                "h_value": 3.5,
                "execution_mode": "fake_backend",
            },
            "circuit_stats": {"depth_transpiled": 30},
            "results": {"e_mitigated": -12.75, "delta_e_gap": 0.012},
        }
        (d / "h350.json").write_text(json.dumps(entry))
        a = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
        a.scan()
        assert len(a.entries) == 1 and len(a.errors) == 0


def test_source_path_attached():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir) / "fake_backend" / "C0_raw"
        d.mkdir(parents=True)
        fp = d / "h350.json"
        fp.write_text(json.dumps(_valid_entry()))
        a = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
        a.scan()
        assert a.entries[0]["_source_path"] == str(fp)
