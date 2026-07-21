"""Standardized result saving and loading for experiments and pipelines.

Eliminates duplicated result-saving boilerplate across scripts by providing
a single entry point with consistent naming, timestamping, and serialization.

Usage:
    from qmbp_simulation.framework.result_io import (
        save_experiment_result,
        save_pipeline_result,
        generate_timestamp,
        build_result_envelope,
    )

    result = build_result_envelope(
        config={"n_qubits": 6, "p": 2},
        results=my_results,
        summary={"mean_de_gap": 0.014},
        elapsed_s=42.5,
    )
    path = save_experiment_result(result, experiment_id="F3")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from qmbp_simulation.utils.helpers import json_serialize

logger = logging.getLogger(__name__)

_DEFAULT_RESULTS_ROOT = Path("results/experiments")
_DEFAULT_PIPELINE_ROOT = Path("results/pipeline")


def build_experiment_id(
    category: str,
    model: str,
    topology: str | list[str] | None = None,
) -> str:
    """Build a hierarchical experiment ID for organized folder output.

    Produces IDs like "noiseless/tfim/heavy_hex" which result in folder
    structures: results/experiments/exp_noiseless/tfim/heavy_hex/

    Parameters
    ----------
    category : str
        Top-level category. Standard values:
        - "noiseless": exact statevector/MPS simulations
        - "noisy": simulated noise (FakeTorino + ZNE/PEA)
        - "hardware": real QPU (IBM Kingston/Torino)
        - "experiment": general research experiments (scaling, cross-N, etc.)
    model : str
        Model name: "tfim", "tfim_longitudinal", "heisenberg_transverse", etc.
    topology : str | list[str] | None
        Topology name(s). If list with >1 element, uses "multi".
        If None, topology level is omitted.

    Returns
    -------
    str
        Hierarchical experiment ID (e.g., "noiseless/tfim/heavy_hex").
    """
    parts = [category, model]
    if topology is not None:
        if isinstance(topology, list):
            topo_str = topology[0] if len(topology) == 1 else "multi"
        else:
            topo_str = topology
        parts.append(topo_str)
    return "/".join(parts)


def generate_timestamp() -> str:
    """Generate a timestamp string for file naming.

    Returns
    -------
    str
        Timestamp in format YYYYMMDD_HHMMSS.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_result_envelope(
    config: dict[str, Any],
    results: Any = None,
    summary: dict[str, Any] | None = None,
    elapsed_s: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standardized result envelope.

    All experiment and pipeline results follow this structure:
    {timestamp, config, results, summary, elapsed_s, metadata}

    Parameters
    ----------
    config : dict
        Configuration used for the run.
    results : Any
        Raw results (will be serialized via json_serialize).
    summary : dict | None
        Summary statistics.
    elapsed_s : float
        Total elapsed time in seconds.
    metadata : dict | None
        Additional metadata (version, git hash, etc.).

    Returns
    -------
    dict
        Standardized result envelope.
    """
    envelope: dict[str, Any] = {
        "schema_version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "config": json_serialize(config),
        "elapsed_s": elapsed_s,
    }
    if results is not None:
        envelope["results"] = json_serialize(results)
    if summary is not None:
        envelope["summary"] = json_serialize(summary)
    if metadata is not None:
        envelope["metadata"] = json_serialize(metadata)
    return envelope


def save_experiment_result(
    data: dict[str, Any],
    experiment_id: str,
    results_dir: Path | None = None,
    timestamp: str | None = None,
) -> Path:
    """Save an experiment result with standard naming convention.

    File is saved as: {results_dir}/exp_{id}/run_{timestamp}.json

    Parameters
    ----------
    data : dict
        Result data (typically from build_result_envelope).
    experiment_id : str
        Experiment identifier (e.g., "F3", "A3").
    results_dir : Path | None
        Root results directory. Defaults to results/experiments/.
    timestamp : str | None
        Timestamp for filename. Auto-generated if None.

    Returns
    -------
    Path
        Path to the saved file.
    """
    root = results_dir or _DEFAULT_RESULTS_ROOT

    # ── Guard: refuse to save empty/invalid envelopes ─────────────────────
    # Prevents garbage from polluting the index. Interrupted runs with partial
    # results are still saved (they have completed_sections > 0 or explicit
    # interrupted=True flag).
    results_dict = data.get("results", {})
    is_interrupted = data.get("interrupted", False)
    if not results_dict and not is_interrupted:
        logger.warning(
            "save_experiment_result: refusing to save envelope with zero sections "
            f"(experiment_id={experiment_id!r}). This prevents index pollution. "
            "If this is intentional, add at least one section result."
        )
        # Still return a path for API compatibility, but don't write the file
        ts = timestamp or generate_timestamp()
        return root / f"exp_{experiment_id.lower()}" / f"run_{ts}_REJECTED.json"

    exp_dir = root / f"exp_{experiment_id.lower()}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    ts = timestamp or generate_timestamp()
    filepath = exp_dir / f"run_{ts}.json"

    # Collision prevention: if file already exists (two runs in same second),
    # append a suffix to avoid overwriting.
    if filepath.exists():
        for suffix in range(1, 100):
            alt = exp_dir / f"run_{ts}_{suffix}.json"
            if not alt.exists():
                filepath = alt
                break

    _write_json(data, filepath)

    # Auto-update the result index with the new entry
    try:
        from qmbp_simulation.framework.result_index import ResultIndex

        index = ResultIndex(root=root)
        index.add_entry(filepath, data)
    except Exception:
        pass  # Index update is best-effort, never blocks saving

    return filepath


def save_pipeline_result(
    data: dict[str, Any],
    output_dir: Path | None = None,
    timestamp: str | None = None,
) -> Path:
    """Save a pipeline run result with standard naming convention.

    File is saved as: {output_dir}/pipeline_run_{timestamp}.json

    Parameters
    ----------
    data : dict
        Result data (typically from build_result_envelope).
    output_dir : Path | None
        Output directory. Defaults to results/pipeline/.
    timestamp : str | None
        Timestamp for filename. Auto-generated if None.

    Returns
    -------
    Path
        Path to the saved file.
    """
    root = output_dir or _DEFAULT_PIPELINE_ROOT
    root.mkdir(parents=True, exist_ok=True)

    ts = timestamp or generate_timestamp()
    filepath = root / f"pipeline_run_{ts}.json"

    _write_json(data, filepath)
    return filepath


def save_benchmark_result(
    data: dict[str, Any],
    output_path: Path | None = None,
) -> Path:
    """Save benchmark results.

    Parameters
    ----------
    data : dict
        Benchmark result data.
    output_path : Path | None
        Explicit output path. If None, uses default location.

    Returns
    -------
    Path
        Path to the saved file.
    """
    if output_path is None:
        output_path = Path("results/benchmarks") / f"bench_{generate_timestamp()}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(data, output_path)
    return output_path


def load_result(path: Path) -> dict[str, Any]:
    """Load a JSON result file.

    Parameters
    ----------
    path : Path
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not valid JSON (wraps JSONDecodeError with context).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError(f"Result file is empty (0 bytes): {path}")

    # Guard against accidentally loading huge files (e.g., raw datasets)
    _MAX_RESULT_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    if file_size > _MAX_RESULT_FILE_SIZE:
        raise ValueError(
            f"Result file suspiciously large ({file_size / 1e6:.1f} MB): {path}. "
            f"Max allowed: {_MAX_RESULT_FILE_SIZE / 1e6:.0f} MB."
        )

    try:
        with open(path) as f:
            result: dict[str, Any] = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Corrupt JSON in {path} (size={file_size} bytes, "
            f"error at line {e.lineno} col {e.colno}): {e.msg}"
        ) from e

    return result


def _write_json(data: dict[str, Any], path: Path) -> None:
    """Write data to JSON with numpy/dataclass serialization support.

    Uses atomic write pattern (write to temp → rename) to prevent
    corruption from interrupted writes. Validates the output is valid JSON.

    Parameters
    ----------
    data : dict
        Data to serialize.
    path : Path
        Output file path.
    """
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to temp file in same directory, then rename.
    # This prevents corrupt JSON from half-written files on crash/kill.
    tmp_fd = None
    tmp_path = None
    try:
        tmp_fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=".run_")
        tmp_path = Path(tmp_path_str)
        with open(tmp_fd, "w", closefd=True) as f:
            tmp_fd = None  # Prevent double-close
            json.dump(data, f, indent=2, default=json_serialize)

        # Validate: re-read to ensure valid JSON was written
        with open(tmp_path) as f:
            json.load(f)  # Raises JSONDecodeError if corrupt

        # Atomic rename (POSIX guarantees this is atomic on same filesystem)
        tmp_path.rename(path)

    except Exception:
        # Clean up temp file on failure
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
        if tmp_fd is not None:
            import os

            os.close(tmp_fd)
        raise


def collect_run_metadata(seed: int | None = None) -> dict[str, Any]:
    """Collect standard run metadata for result files.

    Parameters
    ----------
    seed : int | None
        Random seed used for this run.

    Returns
    -------
    dict
        Metadata including timestamp, python version, package versions, and seed.
    """
    import platform

    metadata: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    if seed is not None:
        metadata["seed"] = seed

    # MPS evaluation mode: deterministic is the default since 2026-06-10.
    # Results generated before this date used stochastic mode (precision=0.005).
    metadata["mps_evaluation_mode"] = "deterministic"

    try:
        import qiskit

        metadata["qiskit_version"] = qiskit.__version__
    except ImportError:
        pass

    try:
        import torch

        metadata["torch_version"] = torch.__version__
    except ImportError:
        pass

    try:
        import qiskit_aer

        metadata["qiskit_aer_version"] = qiskit_aer.__version__
    except ImportError:
        pass

    # Git commit hash for traceability (best-effort)
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            metadata["git_commit"] = result.stdout.strip()
    except Exception:
        pass

    return metadata


def build_simulation_diagnostics(
    backend,
    n_qubits: int,
    topology: str | list[str],
) -> dict[str, Any]:
    """Build simulation method diagnostics for result envelope.

    Documents the properties of the simulation backend used for a run,
    enabling post-hoc assessment of numerical reliability.

    Parameters
    ----------
    backend : ExecutionBackend | BackendV2 | Any
        The backend instance used for evaluation. Supports both
        qmbp_simulation ExecutionBackend instances and Qiskit BackendV2
        instances (e.g. FakeTorino).
    n_qubits : int
        Number of qubits in the system.
    topology : str | list[str]
        Topology name(s) used in this run.

    Returns
    -------
    dict
        Diagnostics dict with backend_type, method_exact, chi info, and warnings.
        Always JSON-serializable. Returns a valid dict even if backend inspection
        raises an exception (defensive against unknown backend types).
    """
    topo_str = topology[0] if isinstance(topology, list) else topology

    # Determine backend name safely
    try:
        if hasattr(backend, "name") and callable(getattr(type(backend), "name", None)):
            # Property-style .name (ExecutionBackend subclasses)
            backend_name = backend.name
        elif hasattr(backend, "name"):
            # Attribute-style .name (Qiskit BackendV2, FakeTorino)
            backend_name = str(backend.name) if backend.name else type(backend).__name__
        else:
            backend_name = type(backend).__name__
    except Exception:
        backend_name = type(backend).__name__

    diag: dict[str, Any] = {
        "backend_type": backend_name,
        "n_qubits": n_qubits,
        "topology": topo_str,
        "method_exact": True,  # Default: statevector/deterministic MPS are exact
    }

    # MPS-specific diagnostics
    chi_max = getattr(backend, "_chi_max", None)
    if chi_max is not None:
        diag["chi_max"] = chi_max
        # Deterministic MPS is exact for 1D with sufficient chi
        diag["method_exact"] = True

        # Flag: warn if topology is 2D and chi might be insufficient
        _2D_TOPOLOGIES = ("square", "triangular", "heavy_hex", "kagome")
        if topo_str in _2D_TOPOLOGIES and n_qubits > 16:
            diag["chi_sufficiency_warning"] = (
                f"2D topology '{topo_str}' with N={n_qubits}: chi={chi_max} "
                f"may be insufficient. Run chi-convergence test (--verify-chi)."
            )

    # Noisy/fake/hardware backend detection
    backend_name_lower = backend_name.lower()
    is_noisy = (
        "noisy" in backend_name_lower
        or "fake" in backend_name_lower
        or "hardware" in backend_name_lower
        or hasattr(backend, "noise_model")
        or "Fake" in type(backend).__name__
    )
    if is_noisy:
        diag["method_exact"] = False
        diag["noise_sources"] = ["gate_error", "readout_error", "decoherence"]
        # Extract shot count if available
        shots = getattr(backend, "_shots", None)
        if shots is None and hasattr(backend, "_config"):
            shots = getattr(backend._config, "shots", None)
        if shots is not None:
            diag["shots"] = shots
        # Hardware-specific: record mode (hardware vs fake_backend)
        if hasattr(backend, "_config"):
            config = backend._config
            if hasattr(config, "mode"):
                diag["hardware_mode"] = config.mode
            if hasattr(config, "backend_name"):
                diag["hardware_backend_name"] = config.backend_name

    return diag


# ═══════════════════════════════════════════════════════════════════════════════
# Batch loading utilities
# ═══════════════════════════════════════════════════════════════════════════════


def load_results_from_dir(
    directory: Path,
    pattern: str = "run_*.json",
    recursive: bool = True,
) -> list[tuple[Path, dict[str, Any]]]:
    """Load all matching JSON results from a directory.

    Skips corrupt/empty files with a warning. Returns (path, data) tuples
    sorted chronologically by filename.

    Parameters
    ----------
    directory : Path
        Directory to scan.
    pattern : str
        Glob pattern for result files. Default: "run_*.json".
    recursive : bool
        If True (default), searches subdirectories recursively.

    Returns
    -------
    list[tuple[Path, dict]]
        List of (file_path, parsed_data) for all successfully loaded files.
    """
    import logging

    _log = logging.getLogger(__name__)

    directory = Path(directory)
    if not directory.exists():
        return []

    glob_fn = directory.rglob if recursive else directory.glob
    results: list[tuple[Path, dict[str, Any]]] = []

    for f in sorted(glob_fn(pattern)):
        try:
            data = load_result(f)
            results.append((f, data))
        except (FileNotFoundError, ValueError) as e:
            _log.warning("Skipping %s: %s", f.name, e)
            continue

    return results


def extract_run_metadata_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Extract lightweight metadata from a result envelope for indexing.

    Returns only the fields needed for filtering/searching without loading
    the full result data (VQE theta arrays, per-point data, etc.).

    Parameters
    ----------
    data : dict
        Full result envelope (as returned by load_result).

    Returns
    -------
    dict
        Lightweight summary with: model, topology, n_qubits, p_layers,
        passed, timestamp, elapsed_s, schema_version, experiment_id.
    """
    config = data.get("config", {})
    system = config.get("system", {})
    summary = data.get("summary", {})

    # Handle both old and new config structures
    model = system.get("model", config.get("model", ""))
    topologies = system.get("topologies", config.get("topologies", []))
    topology = topologies[0] if isinstance(topologies, list) and topologies else str(topologies)
    n_qubits = system.get("n_qubits", config.get("n_qubits", 0))
    p_layers = system.get("p_layers", config.get("p_layers", 0))

    # Detect gap_method from section_1 data (post-fix runs will have this)
    results_data = data.get("results", {})
    if not isinstance(results_data, dict):
        results_data = {}
    s1_raw = results_data.get("section_1", {})
    s1_data = s1_raw.get("data", {}) if isinstance(s1_raw, dict) else {}
    if not isinstance(s1_data, dict):
        s1_data = {}
    gap_methods_found = set()
    for topo_data in s1_data.get("topologies", {}).values():
        if not isinstance(topo_data, dict):
            continue
        for pt in topo_data.get("points", []):
            if isinstance(pt, dict):
                gm = pt.get("gap_method")
                if gm:
                    gap_methods_found.add(gm)

    return {
        "model": model,
        "topology": topology,
        "n_qubits": n_qubits,
        "p_layers": p_layers,
        "passed": summary.get("all_passed", False),
        "pass_rate": summary.get("pass_rate", 0.0),
        "n_sections": summary.get("n_sections", 0),
        "timestamp": data.get("timestamp", ""),
        "elapsed_s": data.get("elapsed_s", 0.0),
        "schema_version": data.get("schema_version", "1.0"),
        "experiment_id": config.get("experiment_id", ""),
        "interrupted": data.get("interrupted", False),
        "gap_methods": sorted(gap_methods_found) if gap_methods_found else None,
        # Simulation diagnostics (populated for runs after 2026-07-13)
        "backend_type": data.get("simulation_diagnostics", {}).get("backend_type"),
        "method_exact": data.get("simulation_diagnostics", {}).get("method_exact"),
    }
