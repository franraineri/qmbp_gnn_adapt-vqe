"""Artifact Store — versioned persistence of experiment artifacts.

Provides ArtifactCollector (register during execution) and persist/load
functions for saving trained models, circuits, and data alongside results.

Artifacts are stored adjacent to the result JSON:
    results/experiments/.../run_20260710_095700.json
    results/experiments/.../run_20260710_095700.artifacts/
        ├── manifest.json
        ├── mpnn_model.pt
        ├── circuit.qpy
        └── theta_opt.npy

Usage in runners:
    self.artifacts.register("circuit", qc, format="qpy")
    self.artifacts.register("mpnn_model", model, format="pt")
    # Called automatically at end of run():
    self.artifacts.persist(saved_result_path)

Loading artifacts later:
    from qmbp_simulation.framework.artifact_store import load_artifact, load_manifest
    manifest = load_manifest(artifact_dir)
    model_data = load_artifact(artifact_dir / "mpnn_model.pt", format="pt")
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from qmbp_simulation.framework.artifact_serializers import (
    ArtifactSerializer,
    get_serializer,
)

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = "1.0"
ARTIFACTS_SUFFIX = ".artifacts"


# ═══════════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ArtifactEntry:
    """One registered artifact awaiting persistence."""

    name: str
    obj: Any
    format: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ManifestEntry:
    """One persisted artifact described in manifest.json."""

    name: str
    filename: str
    format: str
    sha256: str
    size_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# ArtifactCollector
# ═══════════════════════════════════════════════════════════════════════════════


class ArtifactCollector:
    """Collects artifacts during runner execution for later persistence.

    Thread-safe for single-runner use (not concurrent). Artifacts are held
    in memory until persist() is called.

    Parameters
    ----------
    config_fingerprint : dict | None
        Experiment configuration for provenance tracking.
    """

    def __init__(self, config_fingerprint: dict[str, Any] | None = None):
        self._entries: list[ArtifactEntry] = []
        self._config_fingerprint = config_fingerprint or {}

    def register(
        self,
        name: str,
        obj: Any,
        *,
        format: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register an artifact for later persistence.

        Parameters
        ----------
        name : str
            Artifact name (used as filename stem). E.g., "mpnn_model", "circuit".
        obj : Any
            The object to serialize. Type must match format expectations.
        format : str
            Serialization format: "qpy", "qasm3", "pt", "npy", "npz", "json".
        metadata : dict | None
            Optional metadata stored in manifest (e.g., {"n_qubits": 16}).
        """
        # Validate format is known
        get_serializer(format)  # raises ValueError if unknown
        self._entries.append(
            ArtifactEntry(
                name=name,
                obj=obj,
                format=format,
                metadata=metadata or {},
            )
        )
        logger.debug("Artifact registered: %s (format=%s)", name, format)

    @property
    def n_registered(self) -> int:
        """Number of registered artifacts."""
        return len(self._entries)

    def clear(self) -> None:
        """Discard all registered artifacts (e.g., on failure)."""
        self._entries.clear()

    def persist(self, run_result_path: Path) -> Path | None:
        """Persist all registered artifacts to disk.

        Creates a directory adjacent to the result JSON:
            run_20260710_095700.json → run_20260710_095700.artifacts/

        Parameters
        ----------
        run_result_path : Path
            Path to the saved run_*.json file.

        Returns
        -------
        Path | None
            Path to the artifacts directory, or None if nothing to persist.
        """
        if not self._entries:
            return None

        # Create artifacts directory adjacent to result file
        artifact_dir = run_result_path.with_suffix(ARTIFACTS_SUFFIX)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        manifest_entries: list[dict] = []

        for entry in self._entries:
            serializer = get_serializer(entry.format)
            # Determine filename
            ext = _format_extension(entry.format)
            filename = f"{entry.name}{ext}"
            filepath = artifact_dir / filename

            try:
                serializer.save(entry.obj, filepath)
                sha256 = ArtifactSerializer.compute_sha256(filepath)
                size = filepath.stat().st_size

                manifest_entries.append(
                    {
                        "name": entry.name,
                        "filename": filename,
                        "format": entry.format,
                        "sha256": sha256,
                        "size_bytes": size,
                        "metadata": entry.metadata,
                    }
                )
                logger.info("  💾 Artifact saved: %s (%d bytes)", filename, size)

            except Exception as e:
                logger.warning("  ⚠️  Failed to save artifact '%s': %s", entry.name, e)
                manifest_entries.append(
                    {
                        "name": entry.name,
                        "filename": filename,
                        "format": entry.format,
                        "sha256": "",
                        "size_bytes": 0,
                        "metadata": {**entry.metadata, "error": str(e)},
                    }
                )

        # Write manifest
        manifest = _build_manifest(
            run_result_path=run_result_path,
            entries=manifest_entries,
            config_fingerprint=self._config_fingerprint,
        )
        manifest_path = artifact_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            "  📦 Artifacts persisted: %d files in %s",
            len(manifest_entries),
            artifact_dir.name,
        )
        self._entries.clear()
        return artifact_dir


# ═══════════════════════════════════════════════════════════════════════════════
# Load functions
# ═══════════════════════════════════════════════════════════════════════════════


def load_manifest(artifact_dir: Path) -> dict:
    """Load the manifest.json from an artifacts directory.

    Parameters
    ----------
    artifact_dir : Path
        Path to the .artifacts/ directory.

    Returns
    -------
    dict
        Manifest contents.
    """
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {artifact_dir}")
    with open(manifest_path) as f:
        return json.load(f)


def load_artifact(path: Path, *, format: str | None = None) -> Any:
    """Load a single artifact file.

    Parameters
    ----------
    path : Path
        Path to the artifact file (e.g., mpnn_model.pt).
    format : str | None
        Format override. If None, inferred from extension.

    Returns
    -------
    Any
        Deserialized artifact object.
    """
    if format is None:
        format = _extension_to_format(path.suffix)
    serializer = get_serializer(format)
    return serializer.load(path)


def find_artifacts_for_run(run_path: Path) -> Path | None:
    """Find the artifacts directory for a given run result file.

    Parameters
    ----------
    run_path : Path
        Path to run_*.json.

    Returns
    -------
    Path | None
        Path to .artifacts/ directory if it exists.
    """
    artifact_dir = run_path.with_suffix(ARTIFACTS_SUFFIX)
    return artifact_dir if artifact_dir.exists() else None


def inspect_circuit_artifact(artifact_dir: Path) -> dict:
    """Load and analyze a saved circuit artifact.

    Delegates to the canonical `circuit_summary()` from
    `qmbp_simulation.analysis.circuit_visualizer` for comprehensive analysis.

    Parameters
    ----------
    artifact_dir : Path
        Path to the .artifacts/ directory containing circuit.qpy.

    Returns
    -------
    dict
        Circuit analysis from `circuit_summary()` plus drawing string.
        Keys: n_qubits, depth, n_parameters, n_gates_total, n_2q_gates,
        n_1q_gates, gate_counts, drawing.

    Raises
    ------
    FileNotFoundError
        If no circuit artifact exists in the directory.
    """
    qpy_path = artifact_dir / "circuit.qpy"
    qasm_path = artifact_dir / "circuit_readable.qasm3"

    if qpy_path.exists():
        circuit = load_artifact(qpy_path, format="qpy")
    elif qasm_path.exists():
        circuit = load_artifact(qasm_path, format="qasm3")
    else:
        raise FileNotFoundError(
            f"No circuit artifact (circuit.qpy or circuit_readable.qasm3) in {artifact_dir}"
        )

    from qmbp_simulation.analysis import circuit_summary

    info = circuit_summary(circuit)
    info["drawing"] = circuit.draw(output="text", fold=120).__str__()
    return info


def print_circuit_summary(artifact_dir: Path) -> None:
    """Print a human-readable circuit analysis to stdout.

    Uses `circuit_summary()` from the analysis module for consistent metrics.

    Parameters
    ----------
    artifact_dir : Path
        Path to the .artifacts/ directory.
    """
    info = inspect_circuit_artifact(artifact_dir)
    print(f"Circuit Analysis ({artifact_dir.name})")
    print("=" * 60)
    print(f"  Qubits:          {info['n_qubits']}")
    print(f"  Depth:           {info['depth']}")
    print(f"  Total gates:     {info['n_gates_total']}")
    print(f"  Parameters:      {info['n_parameters']}")
    print(f"  2Q gates:        {info['n_2q_gates']}")
    print(f"  1Q gates:        {info['n_1q_gates']}")
    print("  Gate breakdown:")
    for gate, count in sorted(info["gate_counts"].items(), key=lambda x: -x[1]):
        print(f"    {gate:12s}: {count}")
    print()
    print("  Circuit drawing:")
    print(info["drawing"])


def save_circuit_png(
    artifact_dir: Path,
    output_path: Path | str | None = None,
    params: np.ndarray | None = None,
    title: str | None = None,
    fold: int = 30,
    dpi: int = 300,
) -> Path:
    """Save a PNG diagram of the circuit stored in an artifact directory.

    Delegates to `save_circuit_diagram()` from the analysis module.

    Parameters
    ----------
    artifact_dir : Path
        Path to the .artifacts/ directory containing circuit.qpy.
    output_path : Path | str | None
        Output file path. If None, saves as circuit_diagram.png inside
        the artifact directory itself.
    params : np.ndarray | None
        Parameter values to bind before drawing. If None, draws the
        parameterized (symbolic) circuit.
    title : str | None
        Title for the figure.
    fold : int
        Gates per line before folding (default 30).
    dpi : int
        Resolution (default 300 for thesis quality).

    Returns
    -------
    Path
        The path where the PNG was saved.
    """
    from qmbp_simulation.analysis.circuit_visualizer import save_circuit_diagram

    qpy_path = artifact_dir / "circuit.qpy"
    qasm_path = artifact_dir / "circuit_readable.qasm3"

    if qpy_path.exists():
        circuit = load_artifact(qpy_path, format="qpy")
    elif qasm_path.exists():
        circuit = load_artifact(qasm_path, format="qasm3")
    else:
        raise FileNotFoundError(f"No circuit artifact in {artifact_dir}")

    if output_path is None:
        output_path = artifact_dir / "circuit_diagram.png"
    else:
        output_path = Path(output_path)

    return save_circuit_diagram(
        circuit,
        output_path=output_path,
        params=params,
        title=title,
        fold=fold,
        dpi=dpi,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _format_extension(fmt: str) -> str:
    """Map format name to file extension."""
    return {
        "qpy": ".qpy",
        "qasm3": ".qasm3",
        "pt": ".pt",
        "npy": ".npy",
        "npz": ".npz",
        "json": ".json",
    }.get(fmt, f".{fmt}")


def _extension_to_format(ext: str) -> str:
    """Map file extension to format name."""
    ext = ext.lstrip(".")
    mapping = {
        "qpy": "qpy",
        "qasm3": "qasm3",
        "pt": "pt",
        "npy": "npy",
        "npz": "npz",
        "json": "json",
    }
    if ext not in mapping:
        raise ValueError(f"Cannot infer format from extension '.{ext}'")
    return mapping[ext]


def _get_git_commit() -> str | None:
    """Get current git commit hash (short)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _build_manifest(
    run_result_path: Path,
    entries: list[dict],
    config_fingerprint: dict[str, Any],
) -> dict:
    """Build the manifest.json content."""
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_file": f"../{run_result_path.name}",
        "created_at": datetime.now().isoformat(),
        "git_commit": _get_git_commit(),
        "config_fingerprint": config_fingerprint,
        "n_artifacts": len(entries),
        "total_size_bytes": sum(e.get("size_bytes", 0) for e in entries),
        "artifacts": entries,
    }
