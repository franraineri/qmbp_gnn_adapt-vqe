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
        # Runner traceability (populated for runs after 2026-08-10)
        "runner_tag": config.get("runner_tag", data.get("runner_tag")),
        "date_tag": config.get("date_tag", data.get("date_tag")),
        "runner_id": config.get("runner_id", ""),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NPZ Dataset Utilities — Atomic upsert for θ training data
# ═══════════════════════════════════════════════════════════════════════════════


def upsert_theta_npz(
    npz_path: Path,
    h_new: "np.ndarray",
    theta_new: "np.ndarray",
    e_vqe_new: "np.ndarray",
    e_exact_new: "np.ndarray",
    gaps_new: "np.ndarray | None" = None,
    method_new: "list[str] | None" = None,
    quality_tier_new: "list[str] | None" = None,
) -> tuple[int, int]:
    """Atomically update NPZ keeping best θ per h-point (lower energy wins).

    This is the canonical utility for persisting VQE/MPNN training data.
    All runners should use this instead of manual np.savez() to ensure:
    1. Atomic writes (tmp → rename) — no corrupt files on crash
    2. Anti-regression — only updates if new energy is lower
    3. Input validation — filters NaN/Inf automatically
    4. Consistent schema — h_values, theta_opt, e_vqe, e_exact, gaps, de_gaps, method, quality_tier

    Parameters
    ----------
    npz_path : Path
        Path to the .npz file (created if not exists).
    h_new : np.ndarray
        Array of h-values to upsert. Shape: (n_points,)
    theta_new : np.ndarray
        Array of θ vectors. Shape: (n_points, n_params)
    e_vqe_new : np.ndarray
        Computed/predicted energies. Shape: (n_points,)
    e_exact_new : np.ndarray
        Exact ground state energies. Shape: (n_points,)
    gaps_new : np.ndarray | None
        Spectral gaps (optional). Shape: (n_points,)
    method_new : list[str] | None
        Method labels (e.g., "vqe", "mpnn_pred"). Defaults to "unknown".
    quality_tier_new : list[str] | None
        Quality tier per point. Valid values:
        - "verified": VQE-converged, satisfies variational principle
        - "approximate": MPNN prediction passing dual criterion but not VQE-refined
        - "unverified": unknown quality (legacy data, fallback)
        Defaults to "unverified" if not provided.

    Returns
    -------
    tuple[int, int]
        (n_updated, n_added) — counts of updated and newly added entries.
    """
    import numpy as np

    npz_path = Path(npz_path)

    # ── Input validation ──────────────────────────────────────────────
    if len(h_new) != len(theta_new) or len(h_new) != len(e_vqe_new) or len(h_new) != len(e_exact_new):
        raise ValueError(
            f"Length mismatch: h={len(h_new)}, θ={len(theta_new)}, "
            f"e_vqe={len(e_vqe_new)}, e_exact={len(e_exact_new)}"
        )

    # Filter out invalid entries (NaN/Inf in θ or energies)
    valid_mask = np.array([
        np.all(np.isfinite(theta_new[i])) and
        np.isfinite(e_vqe_new[i]) and
        np.isfinite(e_exact_new[i])
        for i in range(len(h_new))
    ])
    if not valid_mask.all():
        n_invalid = int((~valid_mask).sum())
        logger.warning(f"upsert_theta_npz: filtering {n_invalid} invalid entries (NaN/Inf)")
        h_new = h_new[valid_mask]
        theta_new = theta_new[valid_mask]
        e_vqe_new = e_vqe_new[valid_mask]
        e_exact_new = e_exact_new[valid_mask]
        if gaps_new is not None:
            gaps_new = gaps_new[valid_mask]
        if method_new is not None:
            method_new = [method_new[i] for i in range(len(valid_mask)) if valid_mask[i]]
        if quality_tier_new is not None:
            quality_tier_new = [quality_tier_new[i] for i in range(len(valid_mask)) if valid_mask[i]]

    if len(h_new) == 0:
        return 0, 0

    # ── Dimension consistency check ───────────────────────────────────
    # All new θ vectors must have the same dimension. If they don't,
    # something is fundamentally wrong (mixing p_layers or circuit types).
    if len(h_new) > 1:
        new_dims = set()
        for i in range(len(h_new)):
            t = theta_new[i]
            dim = len(t) if hasattr(t, '__len__') else 0
            new_dims.add(dim)
        if len(new_dims) > 1:
            logger.warning(
                "upsert_theta_npz: mixed θ dimensions in new data: %s. "
                "This may indicate mixing p_layers or circuit types. "
                "Proceeding but results may be invalid.", new_dims,
            )

    # If existing NPZ exists, verify new θ dim matches existing
    if npz_path.exists() and len(h_new) > 0:
        try:
            _peek = np.load(npz_path, allow_pickle=True)
            if "theta_opt" in _peek and len(_peek["theta_opt"]) > 0:
                existing_sample = _peek["theta_opt"][0]
                existing_dim = len(existing_sample) if hasattr(existing_sample, '__len__') else 0
                new_dim = len(theta_new[0]) if hasattr(theta_new[0], '__len__') else 0
                if existing_dim > 0 and new_dim > 0 and existing_dim != new_dim:
                    logger.warning(
                        "upsert_theta_npz: new θ dim=%d ≠ existing dim=%d in %s. "
                        "This will create a mixed-dimension NPZ (dtype=object). "
                        "Check that you're not mixing p_layers.",
                        new_dim, existing_dim, npz_path.name,
                    )
        except Exception:
            pass  # Non-fatal — proceed with upsert

    # ── Load existing data ────────────────────────────────────────────
    if npz_path.exists():
        try:
            existing = np.load(npz_path, allow_pickle=True)
            h_all = existing["h_values"].tolist()
            # Handle both legacy (dtype=object) and new (dtype=float64) theta arrays
            raw_theta = existing["theta_opt"]
            theta_all = []
            for row in raw_theta:
                # Convert to float64 array regardless of original dtype
                try:
                    theta_all.append(np.asarray(row, dtype=np.float64))
                except (ValueError, TypeError):
                    # Corrupt entry — will be filtered in validation below
                    theta_all.append(np.array([np.nan]))

            e_key = "e_vqe" if "e_vqe" in existing else "energies"
            e_vqe_all = existing[e_key].tolist() if e_key in existing else [0.0] * len(h_all)
            e_exact_all = existing["e_exact"].tolist()
            gaps_all = existing["gaps"].tolist() if "gaps" in existing else [0.0] * len(h_all)
            method_all = existing["method"].tolist() if "method" in existing else ["unknown"] * len(h_all)
            tier_all = existing["quality_tier"].tolist() if "quality_tier" in existing else ["unverified"] * len(h_all)

            # Validate existing data integrity
            n_existing_before = len(h_all)
            valid_existing = []
            for j in range(len(h_all)):
                theta_j = theta_all[j]
                # Check: non-empty, finite values, finite energy
                if (len(theta_j) > 0 and
                    np.all(np.isfinite(theta_j)) and
                    np.isfinite(e_vqe_all[j])):
                    valid_existing.append(j)

            if len(valid_existing) < n_existing_before:
                logger.warning(
                    f"upsert_theta_npz: removed {n_existing_before - len(valid_existing)} "
                    f"corrupt entries from existing NPZ"
                )
                h_all = [h_all[j] for j in valid_existing]
                theta_all = [theta_all[j] for j in valid_existing]
                e_vqe_all = [e_vqe_all[j] for j in valid_existing]
                e_exact_all = [e_exact_all[j] for j in valid_existing]
                gaps_all = [gaps_all[j] for j in valid_existing]
                method_all = [method_all[j] for j in valid_existing]
                tier_all = [tier_all[j] for j in valid_existing]
        except Exception as e:
            logger.warning(f"upsert_theta_npz: failed to load existing NPZ ({e}), starting fresh")
            h_all, theta_all, e_vqe_all, e_exact_all, gaps_all, method_all, tier_all = [], [], [], [], [], [], []
    else:
        h_all, theta_all, e_vqe_all, e_exact_all, gaps_all, method_all, tier_all = [], [], [], [], [], [], []

    # ── Merge: anti-regression (lower energy wins) ────────────────────
    n_updated, n_added = 0, 0
    for i, h in enumerate(h_new):
        h_val = float(h)
        gap_i = float(gaps_new[i]) if gaps_new is not None and i < len(gaps_new) else 0.0
        method_i = method_new[i] if method_new is not None and i < len(method_new) else "unknown"
        tier_i = quality_tier_new[i] if quality_tier_new is not None and i < len(quality_tier_new) else "unverified"

        # Find existing entry for this h (within tolerance)
        match_idx = next(
            (j for j, hj in enumerate(h_all) if abs(hj - h_val) < 1e-6),
            None,
        )
        if match_idx is not None:
            # Only update if new energy is strictly lower (anti-regression)
            if float(e_vqe_new[i]) < float(e_vqe_all[match_idx]):
                theta_all[match_idx] = theta_new[i]
                e_vqe_all[match_idx] = float(e_vqe_new[i])
                method_all[match_idx] = str(method_i)
                tier_all[match_idx] = str(tier_i)
                if gap_i > 0:
                    gaps_all[match_idx] = gap_i
                n_updated += 1
            elif tier_i == "verified" and tier_all[match_idx] != "verified":
                # Upgrade tier even if energy is same (VQE-verified is more reliable)
                tier_all[match_idx] = "verified"
                method_all[match_idx] = str(method_i)
        else:
            # New h-point: append
            h_all.append(h_val)
            theta_all.append(theta_new[i])
            e_vqe_all.append(float(e_vqe_new[i]))
            e_exact_all.append(float(e_exact_new[i]))
            gaps_all.append(gap_i)
            method_all.append(str(method_i))
            tier_all.append(str(tier_i))
            n_added += 1

    # ── Compute ΔE/gap from stored energies ───────────────────────────
    de_gaps_all = []
    for j in range(len(h_all)):
        gap_j = gaps_all[j] if gaps_all[j] > 1e-10 else 1e-10
        de_gaps_all.append(abs(e_vqe_all[j] - e_exact_all[j]) / gap_j)

    # ── Atomic write: tmp → rename ────────────────────────────────────
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = npz_path.with_suffix(".tmp.npz")
    try:
        # Convert theta_all to array. If all entries have the same shape,
        # use a regular 2D float array. Otherwise fall back to object array.
        theta_shapes = {np.asarray(t).shape for t in theta_all}
        if len(theta_shapes) == 1:
            # All same shape → regular 2D array (PyTorch compatible)
            theta_arr = np.array([np.asarray(t, dtype=np.float64) for t in theta_all])
        else:
            # Ragged array (different N/p mixed) → object array
            # Note: This case is rare and may cause issues with PyTorch loaders
            logger.warning(
                f"upsert_theta_npz: mixed θ shapes {theta_shapes} → using object array"
            )
            theta_arr = np.array(theta_all, dtype=object)

        np.savez(
            tmp_path,
            h_values=np.array(h_all),
            theta_opt=theta_arr,
            e_vqe=np.array(e_vqe_all),
            e_exact=np.array(e_exact_all),
            gaps=np.array(gaps_all),
            de_gaps=np.array(de_gaps_all),
            method=np.array(method_all),
            quality_tier=np.array(tier_all),
            last_updated=np.array(datetime.now().isoformat()),
        )
        tmp_path.rename(npz_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return n_updated, n_added


def load_theta_npz(npz_path: Path) -> dict[str, "np.ndarray"]:
    """Load θ training data from NPZ file with validation.

    Parameters
    ----------
    npz_path : Path
        Path to the .npz file.

    Returns
    -------
    dict
        Keys: h_values, theta_opt, e_vqe, e_exact, gaps, de_gaps, method
        All arrays are validated (no NaN/Inf in θ or energies).

    Raises
    ------
    FileNotFoundError
        If file doesn't exist.
    ValueError
        If file is corrupt or has invalid schema.
    """
    import numpy as np

    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Theta NPZ file not found: {npz_path}")

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        raise ValueError(f"Failed to load NPZ file {npz_path}: {e}") from e

    required_keys = {"h_values", "theta_opt", "e_exact"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"NPZ file missing required keys: {missing}")

    # Convert theta_opt to consistent float64 format (handles legacy object arrays)
    raw_theta = data["theta_opt"]
    if raw_theta.dtype == object:
        # Legacy format: array of arrays with dtype=object
        theta_list = []
        for row in raw_theta:
            try:
                theta_list.append(np.asarray(row, dtype=np.float64))
            except (ValueError, TypeError):
                theta_list.append(np.array([np.nan]))  # Mark as invalid
        # Check if all have same shape (normal case)
        shapes = {t.shape for t in theta_list}
        if len(shapes) == 1:
            theta_arr = np.array(theta_list, dtype=np.float64)
        else:
            # Ragged array — keep as object but convert inner arrays
            theta_arr = np.array(theta_list, dtype=object)
    else:
        theta_arr = raw_theta

    result = {
        "h_values": data["h_values"],
        "theta_opt": theta_arr,
        "e_vqe": data.get("e_vqe", data.get("energies", np.zeros_like(data["h_values"]))),
        "e_exact": data["e_exact"],
        "gaps": data.get("gaps", np.zeros_like(data["h_values"])),
        "de_gaps": data.get("de_gaps", np.zeros_like(data["h_values"])),
        "method": data.get("method", np.array(["unknown"] * len(data["h_values"]))),
    }

    # Validate: filter corrupt entries
    n_total = len(result["h_values"])
    valid_mask = []
    for i in range(n_total):
        theta_i = result["theta_opt"][i]
        # Convert to array if needed and check finite
        try:
            theta_arr_i = np.asarray(theta_i, dtype=np.float64)
            is_valid = np.all(np.isfinite(theta_arr_i)) and np.isfinite(result["e_vqe"][i])
        except (ValueError, TypeError):
            is_valid = False
        valid_mask.append(is_valid)

    valid_mask = np.array(valid_mask)
    if not valid_mask.all():
        n_invalid = int((~valid_mask).sum())
        logger.warning(f"load_theta_npz: {n_invalid}/{n_total} entries have NaN/Inf (filtered)")
        for key in result:
            if isinstance(result[key], np.ndarray) and len(result[key]) == n_total:
                result[key] = result[key][valid_mask]

    return result
