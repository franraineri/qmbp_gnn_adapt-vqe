"""Unit tests for qmbp_simulation.utils module."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pytest

from qmbp_simulation.utils import json_serialize, set_global_seed, timer


class TestSetGlobalSeed:
    """Test set_global_seed determinism."""

    def test_same_seed_produces_identical_numpy_sequences(self):
        set_global_seed(123)
        a = np.random.rand(10)
        set_global_seed(123)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_produce_different_sequences(self):
        set_global_seed(1)
        a = np.random.rand(10)
        set_global_seed(2)
        b = np.random.rand(10)
        assert not np.array_equal(a, b)

    def test_torch_seed_determinism(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        set_global_seed(99)
        a = torch.randn(5)
        set_global_seed(99)
        b = torch.randn(5)
        assert torch.equal(a, b)


class TestJsonSerialize:
    """Test json_serialize handles numpy arrays, dataclasses, datetime."""

    def test_numpy_array_to_list(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = json_serialize(arr)
        assert result == [1.0, 2.0, 3.0]
        assert isinstance(result, list)

    def test_numpy_integer(self):
        val = np.int64(42)
        result = json_serialize(val)
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_floating(self):
        val = np.float64(3.14)
        result = json_serialize(val)
        assert abs(result - 3.14) < 1e-10
        assert isinstance(result, float)

    def test_nan_becomes_none(self):
        assert json_serialize(float("nan")) is None
        assert json_serialize(np.float64("nan")) is None

    def test_inf_becomes_none(self):
        assert json_serialize(float("inf")) is None
        assert json_serialize(np.float64("inf")) is None

    def test_dataclass_serialization(self):
        @dataclass
        class Point:
            x: float
            y: float

        p = Point(x=1.5, y=2.5)
        result = json_serialize(p)
        assert result == {"x": 1.5, "y": 2.5}

    def test_datetime_to_isoformat(self):
        dt = datetime(2025, 1, 15, 10, 30, 0)
        result = json_serialize(dt)
        assert result == "2025-01-15T10:30:00"


class TestTimer:
    """Test timer context manager."""

    def test_timer_measures_elapsed_time(self):
        with timer("test") as t:
            time.sleep(0.05)
        assert t.elapsed_s >= 0.04
        assert t.label == "test"

    def test_timer_default_label(self):
        with timer() as t:
            pass
        assert t.label == ""
        assert t.elapsed_s >= 0.0

    def test_timer_result_accessible_after_block(self):
        with timer("block") as t:
            _ = sum(range(1000))
        assert isinstance(t.elapsed_s, float)
        assert t.elapsed_s > 0


class TestJsonDump:
    """Test json_dump writes valid JSON files."""

    def test_json_dump_creates_file(self, tmp_path):
        from qmbp_simulation.utils import json_dump

        data = {"key": "value", "number": 42}
        path = tmp_path / "test.json"
        json_dump(data, path)
        assert path.exists()

    def test_json_dump_handles_numpy(self, tmp_path):
        import json

        from qmbp_simulation.utils import json_dump

        data = {"array": np.array([1.0, 2.0, 3.0]), "int": np.int64(5)}
        path = tmp_path / "test.json"
        json_dump(data, path)
        # Should be valid JSON
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["array"] == [1.0, 2.0, 3.0]
        assert loaded["int"] == 5

    def test_json_dump_with_indent(self, tmp_path):
        from qmbp_simulation.utils import json_dump

        data = {"a": 1, "b": 2}
        path = tmp_path / "test.json"
        json_dump(data, path, indent=4)
        content = path.read_text()
        # Indented JSON has newlines
        assert "\n" in content


class TestJsonSerializeUnification:
    """Test that all _json_default implementations delegate to json_serialize.

    Verifies Action B: the three separate _json_default implementations
    (base.py, persistence.py, result_io.py) all produce the same output
    as the canonical json_serialize from utils/helpers.py.
    """

    def test_base_experiment_json_default_delegates(self):
        """BaseExperiment._json_default uses json_serialize under the hood."""
        from qmbp_simulation.framework.base import BaseExperiment

        fn = BaseExperiment._json_default

        assert fn(np.int64(42)) == 42
        assert fn(np.float64(3.14)) == pytest.approx(3.14)
        assert fn(np.array([1, 2, 3])) == [1, 2, 3]
        assert fn(np.bool_(True)) is True
        # Path support (json_serialize handles it)
        from pathlib import Path

        assert fn(Path("/tmp/test")) == "/tmp/test"

    def test_base_experiment_json_default_handles_nan(self):
        """NaN and Inf become None (safe for JSON)."""
        from qmbp_simulation.framework.base import BaseExperiment

        fn = BaseExperiment._json_default
        assert fn(np.float64(float("nan"))) is None
        assert fn(np.float64(float("inf"))) is None
        assert fn(np.float64(float("-inf"))) is None

    def test_persistence_write_json_roundtrip(self, tmp_path):
        """Hardware persistence _write_json produces valid JSON with numpy types."""
        import json

        from qmbp_simulation.execution.hardware.persistence import _write_json

        data = {
            "energy": np.float64(-5.123),
            "n_qubits": np.int64(10),
            "layouts": np.array([[0, 1, 2], [3, 4, 5]]),
            "flag": np.bool_(False),
        }
        path = tmp_path / "test_persistence.json"
        _write_json(data, path)

        with open(path) as f:
            loaded = json.load(f)

        assert loaded["energy"] == pytest.approx(-5.123)
        assert loaded["n_qubits"] == 10
        assert loaded["layouts"] == [[0, 1, 2], [3, 4, 5]]
        assert loaded["flag"] is False

    def test_persistence_write_json_handles_path(self, tmp_path):
        """Hardware persistence handles Path objects in data."""
        import json
        from pathlib import Path

        from qmbp_simulation.execution.hardware.persistence import _write_json

        data = {"output_dir": Path("/results/hardware/run_001")}
        path = tmp_path / "test_path.json"
        _write_json(data, path)

        with open(path) as f:
            loaded = json.load(f)

        assert loaded["output_dir"] == "/results/hardware/run_001"

    def test_result_io_uses_json_serialize(self, tmp_path):
        """result_io save functions use json_serialize for numpy types."""
        from qmbp_simulation.framework.result_io import (
            build_result_envelope,
        )

        config = {"n_qubits": np.int64(6), "h_values": np.array([2.0, 1.5, 1.0])}
        results = {"energies": np.array([-4.0, -3.5, -3.0])}
        summary = {"mean_de_gap": np.float64(0.023)}

        envelope = build_result_envelope(
            config=config,
            results=results,
            summary=summary,
            elapsed_s=10.5,
        )

        # Envelope values should already be serialized
        assert envelope["config"]["n_qubits"] == 6
        assert envelope["config"]["h_values"] == [2.0, 1.5, 1.0]
        assert envelope["results"]["energies"] == [-4.0, -3.5, -3.0]
        assert envelope["summary"]["mean_de_gap"] == pytest.approx(0.023)

    def test_all_implementations_agree_on_types(self):
        """All three json_default callsites produce identical output."""
        from pathlib import Path

        from qmbp_simulation.framework.base import BaseExperiment
        from qmbp_simulation.utils.helpers import json_serialize

        test_cases = [
            (np.int64(7), 7),
            (np.float64(2.718), pytest.approx(2.718)),
            (np.array([10, 20]), [10, 20]),
            (np.bool_(False), False),
            (Path("/a/b"), "/a/b"),
        ]

        for obj, expected in test_cases:
            base_result = BaseExperiment._json_default(obj)
            canonical_result = json_serialize(obj)
            # Both should produce the same value
            if isinstance(expected, type(pytest.approx(0))):
                assert base_result == expected
                assert canonical_result == expected
            else:
                assert base_result == canonical_result == expected, (
                    f"Mismatch for {type(obj).__name__}: "
                    f"base={base_result}, canonical={canonical_result}"
                )
