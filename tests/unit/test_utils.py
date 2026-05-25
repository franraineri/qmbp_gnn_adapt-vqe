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
