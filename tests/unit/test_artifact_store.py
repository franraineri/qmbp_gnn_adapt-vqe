"""Tests for the artifact store module.

Covers: ArtifactCollector, serializers, manifest generation, roundtrip I/O.
"""

from __future__ import annotations

import numpy as np
import pytest

from qmbp_simulation.framework.artifact_serializers import (
    ArtifactSerializer,
    JSONSerializer,
    NumpyCompressedSerializer,
    NumpySerializer,
    get_serializer,
    register_serializer,
)
from qmbp_simulation.framework.artifact_store import (
    ARTIFACTS_SUFFIX,
    ArtifactCollector,
    find_artifacts_for_run,
    load_artifact,
    load_manifest,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Serializer Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetSerializer:
    def test_known_formats(self):
        for fmt in ("qpy", "qasm3", "pt", "npy", "npz", "json"):
            s = get_serializer(fmt)
            assert isinstance(s, ArtifactSerializer)

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown artifact format"):
            get_serializer("parquet")

    def test_register_custom_serializer(self):
        class FakeSerializer(ArtifactSerializer):
            def save(self, obj, path):
                path.write_text(str(obj))

            def load(self, path):
                return path.read_text()

        register_serializer("fake", FakeSerializer)
        s = get_serializer("fake")
        assert isinstance(s, FakeSerializer)


class TestNumpySerializer:
    def test_roundtrip_1d(self, tmp_path):
        s = NumpySerializer()
        arr = np.array([1.0, 2.0, 3.0])
        path = tmp_path / "test.npy"
        s.save(arr, path)
        loaded = s.load(path)
        assert np.allclose(loaded, arr)

    def test_roundtrip_2d(self, tmp_path):
        s = NumpySerializer()
        arr = np.random.randn(10, 4)
        path = tmp_path / "theta.npy"
        s.save(arr, path)
        loaded = s.load(path)
        assert np.allclose(loaded, arr)
        assert loaded.shape == (10, 4)


class TestNumpyCompressedSerializer:
    def test_roundtrip_dict(self, tmp_path):
        s = NumpyCompressedSerializer()
        data = {"h_values": np.linspace(1.0, 3.0, 20), "theta": np.random.randn(20, 6)}
        path = tmp_path / "data.npz"
        s.save(data, path)
        loaded = s.load(path)
        assert "h_values" in loaded
        assert "theta" in loaded
        assert np.allclose(loaded["h_values"], data["h_values"])
        assert np.allclose(loaded["theta"], data["theta"])


class TestJSONSerializer:
    def test_roundtrip(self, tmp_path):
        s = JSONSerializer()
        data = {"model": "tfim", "n_qubits": 16, "values": [1.0, 2.0]}
        path = tmp_path / "config.json"
        s.save(data, path)
        loaded = s.load(path)
        assert loaded == data


class TestQPYSerializer:
    def test_roundtrip(self, tmp_path):
        from qiskit import QuantumCircuit

        from qmbp_simulation.framework.artifact_serializers import QPYSerializer

        s = QPYSerializer()
        qc = QuantumCircuit(4)
        qc.h(range(4))
        qc.cx(0, 1)
        qc.rz(0.5, 2)

        path = tmp_path / "circuit.qpy"
        s.save(qc, path)
        loaded = s.load(path)
        assert loaded.num_qubits == 4
        assert loaded.size() == qc.size()


class TestSHA256:
    def test_compute_sha256(self, tmp_path):
        path = tmp_path / "test.bin"
        path.write_bytes(b"hello world")
        sha = ArtifactSerializer.compute_sha256(path)
        assert len(sha) == 64  # SHA-256 hex digest
        assert sha == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


# ═══════════════════════════════════════════════════════════════════════════════
# ArtifactCollector Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestArtifactCollector:
    def test_register_increments_count(self):
        c = ArtifactCollector()
        assert c.n_registered == 0
        c.register("arr", np.zeros(5), format="npy")
        assert c.n_registered == 1
        c.register("meta", {"k": "v"}, format="json")
        assert c.n_registered == 2

    def test_register_invalid_format_raises(self):
        c = ArtifactCollector()
        with pytest.raises(ValueError):
            c.register("bad", "data", format="unknown_fmt")

    def test_clear(self):
        c = ArtifactCollector()
        c.register("a", np.zeros(3), format="npy")
        c.register("b", np.ones(3), format="npy")
        assert c.n_registered == 2
        c.clear()
        assert c.n_registered == 0

    def test_persist_creates_directory(self, tmp_path):
        c = ArtifactCollector({"test": True})
        c.register("data", np.array([1, 2, 3]), format="npy")
        run_path = tmp_path / "run_test.json"
        run_path.write_text("{}")

        ad = c.persist(run_path)
        assert ad is not None
        assert ad.exists()
        assert ad.name == "run_test.artifacts"
        assert (ad / "manifest.json").exists()
        assert (ad / "data.npy").exists()

    def test_persist_empty_returns_none(self, tmp_path):
        c = ArtifactCollector()
        run_path = tmp_path / "run_empty.json"
        run_path.write_text("{}")
        assert c.persist(run_path) is None

    def test_persist_clears_entries(self, tmp_path):
        c = ArtifactCollector()
        c.register("x", np.zeros(2), format="npy")
        run_path = tmp_path / "run_clear.json"
        run_path.write_text("{}")
        c.persist(run_path)
        assert c.n_registered == 0

    def test_persist_multiple_artifacts(self, tmp_path):
        c = ArtifactCollector({"multi": True})
        c.register("arr1", np.zeros(10), format="npy")
        c.register("arr2", np.ones(5), format="npy")
        c.register("config", {"a": 1}, format="json")

        run_path = tmp_path / "run_multi.json"
        run_path.write_text("{}")
        ad = c.persist(run_path)

        manifest = load_manifest(ad)
        assert manifest["n_artifacts"] == 3
        assert all(e["sha256"] != "" for e in manifest["artifacts"])


# ═══════════════════════════════════════════════════════════════════════════════
# Load & Find Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoadFunctions:
    def test_load_manifest(self, tmp_path):
        c = ArtifactCollector({"model": "tfim"})
        c.register("theta", np.random.randn(5, 3), format="npy")
        run_path = tmp_path / "run_load.json"
        run_path.write_text("{}")
        ad = c.persist(run_path)

        manifest = load_manifest(ad)
        assert manifest["schema_version"] == "1.0"
        assert "config_fingerprint" in manifest
        assert manifest["config_fingerprint"]["model"] == "tfim"

    def test_load_manifest_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nonexistent")

    def test_load_artifact_infers_format(self, tmp_path):
        s = NumpySerializer()
        arr = np.array([10, 20, 30])
        path = tmp_path / "test.npy"
        s.save(arr, path)

        loaded = load_artifact(path)  # No format specified — inferred from .npy
        assert np.allclose(loaded, arr)

    def test_find_artifacts_for_run_exists(self, tmp_path):
        run_path = tmp_path / "run_find.json"
        run_path.write_text("{}")
        art_dir = run_path.with_suffix(ARTIFACTS_SUFFIX)
        art_dir.mkdir()

        found = find_artifacts_for_run(run_path)
        assert found == art_dir

    def test_find_artifacts_for_run_missing(self, tmp_path):
        run_path = tmp_path / "run_nope.json"
        run_path.write_text("{}")
        assert find_artifacts_for_run(run_path) is None
