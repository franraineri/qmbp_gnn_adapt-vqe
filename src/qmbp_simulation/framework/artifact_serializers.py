"""Artifact serializers — lazy-import format handlers.

Each serializer knows how to save and load one artifact type.
Lazy imports avoid pulling torch/qiskit into modules that don't need them.

Usage:
    from qmbp_simulation.framework.artifact_serializers import get_serializer
    serializer = get_serializer("qpy")
    serializer.save(circuit, path)
    circuit = serializer.load(path)
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ArtifactSerializer(ABC):
    """Base class for artifact serializers."""

    @abstractmethod
    def save(self, obj: Any, path: Path) -> None:
        """Serialize object to file."""

    @abstractmethod
    def load(self, path: Path) -> Any:
        """Deserialize object from file."""

    @staticmethod
    def compute_sha256(path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


class QPYSerializer(ArtifactSerializer):
    """Serialize Qiskit QuantumCircuit via QPY binary format."""

    def save(self, obj: Any, path: Path) -> None:
        from qiskit.qpy import dump as qpy_dump

        with open(path, "wb") as f:
            qpy_dump([obj], f)

    def load(self, path: Path) -> Any:
        from qiskit.qpy import load as qpy_load

        with open(path, "rb") as f:
            circuits = qpy_load(f)
        return circuits[0]


class QASM3Serializer(ArtifactSerializer):
    """Serialize QuantumCircuit as OpenQASM 3 string (human-readable)."""

    def save(self, obj: Any, path: Path) -> None:
        from qiskit.qasm3 import dumps

        qasm_str = dumps(obj)
        path.write_text(qasm_str, encoding="utf-8")

    def load(self, path: Path) -> Any:
        from qiskit.qasm3 import loads

        return loads(path.read_text(encoding="utf-8"))


class TorchSerializer(ArtifactSerializer):
    """Serialize PyTorch model state_dict."""

    def save(self, obj: Any, path: Path) -> None:
        import torch

        if hasattr(obj, "state_dict"):
            payload = {
                "state_dict": obj.state_dict(),
                "class_name": type(obj).__name__,
            }
            # Preserve model config if available
            if hasattr(obj, "node_features"):
                payload["config"] = {
                    "node_features": obj.node_features,
                    "hidden_dim": obj.hidden_dim,
                    "output_dim": obj.output_dim,
                    "n_layers": obj.n_layers,
                }
            torch.save(payload, path)
        else:
            torch.save(obj, path)

    def load(self, path: Path) -> Any:
        import torch

        return torch.load(path, map_location="cpu", weights_only=False)


class NumpySerializer(ArtifactSerializer):
    """Serialize numpy array to .npy format."""

    def save(self, obj: Any, path: Path) -> None:
        import numpy as np

        np.save(path, obj)

    def load(self, path: Path) -> Any:
        import numpy as np

        return np.load(path, allow_pickle=False)


class NumpyCompressedSerializer(ArtifactSerializer):
    """Serialize multiple numpy arrays to .npz compressed format."""

    def save(self, obj: Any, path: Path) -> None:
        import numpy as np

        if isinstance(obj, dict):
            np.savez_compressed(path, **obj)
        else:
            np.savez_compressed(path, data=obj)

    def load(self, path: Path) -> Any:
        import numpy as np

        return dict(np.load(path, allow_pickle=False))


class JSONSerializer(ArtifactSerializer):
    """Serialize JSON-compatible dicts."""

    def save(self, obj: Any, path: Path) -> None:
        from qmbp_simulation.utils.helpers import json_serialize

        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=json_serialize)

    def load(self, path: Path) -> Any:
        with open(path) as f:
            return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════

_SERIALIZER_MAP: dict[str, type[ArtifactSerializer]] = {
    "qpy": QPYSerializer,
    "qasm3": QASM3Serializer,
    "pt": TorchSerializer,
    "npy": NumpySerializer,
    "npz": NumpyCompressedSerializer,
    "json": JSONSerializer,
}


def get_serializer(fmt: str) -> ArtifactSerializer:
    """Get a serializer instance by format name."""
    cls = _SERIALIZER_MAP.get(fmt)
    if cls is None:
        available = ", ".join(sorted(_SERIALIZER_MAP.keys()))
        raise ValueError(f"Unknown artifact format '{fmt}'. Available: {available}")
    return cls()


def register_serializer(fmt: str, cls: type[ArtifactSerializer]) -> None:
    """Register a custom serializer for a new format."""
    _SERIALIZER_MAP[fmt] = cls
