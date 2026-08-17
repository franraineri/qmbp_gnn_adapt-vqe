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
    """Generate a timestamp string for file naming."""
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

    # ── Auto-sync evaluation metrics to ModelRegistryDB ───────────────────
    # When saving experiment results, automatically update the corresponding
    # model's evaluation records in the registry for regression tracking.
    try:
        from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

        db = ModelRegistryDB()
        db.sync_evaluation_from_result(data, auto_create=False)
    except Exception:
        pass  # Registry sync is best-effort, never blocks saving

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
# Result metadata extraction utilities
# ═══════════════════════════════════════════════════════════════════════════════


def extract_checkpoint_used(result: dict[str, Any]) -> str:
    """Extract which MPNN checkpoint was used for predictions in a result JSON.

    Checks multiple locations (in priority order):
    1. section_2.data.checkpoint_used — set by run_large_n_extrapolation (v2+)
    2. config.checkpoint — explicit --checkpoint flag
    3. Falls back to "unknown"

    Parameters
    ----------
    result : dict
        Loaded result JSON (from load_result or json.load).

    Returns
    -------
    str
        Checkpoint filename or path. Returns "unknown" if not determinable.
    """
    # Priority 1: explicitly stored by the runner after model loading
    results_block = result.get("results", {})
    for sec_key in ("section_2", "section_3", "section_4"):
        sec_data = results_block.get(sec_key, {}).get("data", {})
        if "checkpoint_used" in sec_data:
            return sec_data["checkpoint_used"]

    # Priority 2: config.checkpoint (--checkpoint flag)
    config = result.get("config", {})
    if config.get("checkpoint"):
        return config["checkpoint"]

    # Priority 3: unknown
    return "unknown"


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

    # Continuous quality metrics: use from summary if available,
    # otherwise estimate from pass_rate for legacy results.
    quality_score = summary.get("quality_score")
    grade = summary.get("grade")
    mean_de_gap = summary.get("mean_de_gap")

    if quality_score is None and summary.get("pass_rate") is not None:
        pr = float(summary.get("pass_rate", 0))
        quality_score = pr * 0.90  # Linear approximation
        try:
            from qmbp_simulation.analysis.constants import grade_from_score

            grade = grade_from_score(quality_score)
        except ImportError:
            pass

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
        # Continuous quality metrics (estimated from pass_rate for legacy runs)
        "quality_score": quality_score,
        "grade": grade,
        "mean_de_gap": mean_de_gap,
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
    h_new: np.ndarray,
    theta_new: np.ndarray,
    e_vqe_new: np.ndarray,
    e_exact_new: np.ndarray,
    gaps_new: np.ndarray | None = None,
    method_new: list[str] | None = None,
    quality_tier_new: list[str] | None = None,
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
    if (
        len(h_new) != len(theta_new)
        or len(h_new) != len(e_vqe_new)
        or len(h_new) != len(e_exact_new)
    ):
        raise ValueError(
            f"Length mismatch: h={len(h_new)}, θ={len(theta_new)}, "
            f"e_vqe={len(e_vqe_new)}, e_exact={len(e_exact_new)}"
        )

    # Filter out invalid entries (NaN/Inf in θ or energies)
    valid_mask = np.array(
        [
            np.all(np.isfinite(theta_new[i]))
            and np.isfinite(e_vqe_new[i])
            and np.isfinite(e_exact_new[i])
            for i in range(len(h_new))
        ]
    )
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
            quality_tier_new = [
                quality_tier_new[i] for i in range(len(valid_mask)) if valid_mask[i]
            ]

    if len(h_new) == 0:
        return 0, 0

    # ── Dimension consistency check ───────────────────────────────────
    # All new θ vectors must have the same dimension. If they don't,
    # something is fundamentally wrong (mixing p_layers or circuit types).
    if len(h_new) > 1:
        new_dims = set()
        for i in range(len(h_new)):
            t = theta_new[i]
            dim = len(t) if hasattr(t, "__len__") else 0
            new_dims.add(dim)
        if len(new_dims) > 1:
            logger.warning(
                "upsert_theta_npz: mixed θ dimensions in new data: %s. "
                "This may indicate mixing p_layers or circuit types. "
                "Proceeding but results may be invalid.",
                new_dims,
            )

    # If existing NPZ exists, verify new θ dim matches existing
    if npz_path.exists() and len(h_new) > 0:
        try:
            _peek = np.load(npz_path, allow_pickle=True)
            if "theta_opt" in _peek and len(_peek["theta_opt"]) > 0:
                existing_sample = _peek["theta_opt"][0]
                existing_dim = len(existing_sample) if hasattr(existing_sample, "__len__") else 0
                new_dim = len(theta_new[0]) if hasattr(theta_new[0], "__len__") else 0
                if existing_dim > 0 and new_dim > 0 and existing_dim != new_dim:
                    logger.warning(
                        "upsert_theta_npz: new θ dim=%d ≠ existing dim=%d in %s. "
                        "This will create a mixed-dimension NPZ (dtype=object). "
                        "Check that you're not mixing p_layers.",
                        new_dim,
                        existing_dim,
                        npz_path.name,
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
            method_all = (
                existing["method"].tolist() if "method" in existing else ["unknown"] * len(h_all)
            )
            tier_all = (
                existing["quality_tier"].tolist()
                if "quality_tier" in existing
                else ["unverified"] * len(h_all)
            )

            # Validate existing data integrity
            n_existing_before = len(h_all)
            valid_existing = []
            for j in range(len(h_all)):
                theta_j = theta_all[j]
                # Check: non-empty, finite values, finite energy
                if len(theta_j) > 0 and np.all(np.isfinite(theta_j)) and np.isfinite(e_vqe_all[j]):
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
            h_all, theta_all, e_vqe_all, e_exact_all, gaps_all, method_all, tier_all = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
    else:
        h_all, theta_all, e_vqe_all, e_exact_all, gaps_all, method_all, tier_all = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

    # ── Merge: anti-regression (lower energy wins) ────────────────────
    # Build O(1) lookup index for h-matching (avoids O(n²) linear scan)
    h_index: dict[int, int] = {round(hj * 1_000_000): j for j, hj in enumerate(h_all)}

    n_updated, n_added = 0, 0
    for i, h in enumerate(h_new):
        h_val = float(h)
        gap_i = float(gaps_new[i]) if gaps_new is not None and i < len(gaps_new) else 0.0
        method_i = method_new[i] if method_new is not None and i < len(method_new) else "unknown"
        tier_i = (
            quality_tier_new[i]
            if quality_tier_new is not None and i < len(quality_tier_new)
            else "unverified"
        )

        # Find existing entry for this h via O(1) index lookup
        h_key_int = round(h_val * 1_000_000)
        match_idx = h_index.get(h_key_int)
        if match_idx is not None:
            # Only update if new energy is strictly lower (anti-regression)
            if float(e_vqe_new[i]) < float(e_vqe_all[match_idx]):
                theta_all[match_idx] = theta_new[i]
                e_vqe_all[match_idx] = float(e_vqe_new[i])
                e_exact_all[match_idx] = float(e_exact_new[i])  # Always refresh e_exact
                method_all[match_idx] = str(method_i)
                tier_all[match_idx] = str(tier_i)
                if gap_i > 0:
                    gaps_all[match_idx] = gap_i
                n_updated += 1
            elif tier_i == "verified" and tier_all[match_idx] != "verified":
                # Upgrade tier even if energy is same (VQE-verified is more reliable)
                tier_all[match_idx] = "verified"
                method_all[match_idx] = str(method_i)
                e_exact_all[match_idx] = float(e_exact_new[i])  # Refresh e_exact on tier upgrade
                if gap_i > 0:
                    gaps_all[match_idx] = gap_i
                n_updated += 1
        else:
            # New h-point: append
            h_index[h_key_int] = len(h_all)
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
            logger.warning(f"upsert_theta_npz: mixed θ shapes {theta_shapes} → using object array")
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

    # ── NPZ Quality Tier → ModelRegistryDB Sync ───────────────────────
    # Notify the model registry when quality tier improves significantly.
    # This triggers needs_retrain flag for stale models.
    if n_updated > 0 or n_added > 0:
        try:
            # Compute verified ratio before and after
            old_verified = sum(1 for t in tier_all if t == "verified") - sum(
                1 for t in (quality_tier_new or []) if t == "verified"
            )
            old_total = len(tier_all) - len(h_new)
            new_verified = sum(1 for t in tier_all if t == "verified")
            new_total = len(tier_all)

            if old_total > 0 and new_total > 0:
                old_ratio = old_verified / old_total
                new_ratio = new_verified / new_total

                # Only notify if improvement is significant (>5%)
                if new_ratio - old_ratio > 0.05:
                    # Extract topology/n_qubits/p from filename
                    # Expected format: {topology}_N{n_qubits}_p{p_layers}.npz
                    filename = npz_path.stem  # e.g., "chain_1d_N10_p1"
                    parts = filename.rsplit("_", 2)  # ["chain_1d", "N10", "p1"]
                    if len(parts) >= 3 and parts[1].startswith("N") and parts[2].startswith("p"):
                        topology = parts[0]
                        n_qubits = int(parts[1][1:])
                        p_layers = int(parts[2][1:])

                        from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

                        db = ModelRegistryDB()
                        db.mark_needs_retrain_from_npz_update(
                            topology=topology,
                            n_qubits=n_qubits,
                            model_name="tfim_bond_resolved",  # Default, could be inferred
                            p_layers=p_layers,
                            old_verified_ratio=old_ratio,
                            new_verified_ratio=new_ratio,
                        )

                        # Auto-refresh zoo quality scores for affected topology
                        from qmbp_simulation.predictors.model_zoo import (
                            _load_manifest,
                            _save_manifest,
                            compute_training_quality_score,
                        )

                        entries = _load_manifest()
                        for ent in entries:
                            if ent.topology == topology and ent.p_layers == p_layers:
                                new_score = compute_training_quality_score(
                                    topology=topology,
                                    n_qubits=ent.n_qubits,
                                    p_layers=p_layers,
                                )
                                if abs(new_score - ent.training_quality_score) > 0.01:
                                    ent.training_quality_score = new_score
                        _save_manifest(entries)
        except Exception:
            pass  # Best-effort, never block NPZ writes

    return n_updated, n_added


def load_npz_as_theta_dict(
    npz_path: Path,
    n_params: int,
    *,
    h_precision: int = 6,
) -> dict[float, tuple[np.ndarray, float | None]]:
    """Load NPZ and build a validated h→(θ, energy) lookup dict.

    This is the canonical way to load previous θ for anti-regression baselines.
    Validates each entry: finite θ, correct dimension, finite energy.
    Returns a dict keyed by `round(h, h_precision)` for O(1) lookup.

    Parameters
    ----------
    npz_path : Path
        Path to the .npz file.
    n_params : int
        Expected number of parameters (entries with different dim are skipped).
    h_precision : int
        Decimal places for rounding h keys (default: 6).

    Returns
    -------
    dict[float, tuple[np.ndarray, float | None]]
        Mapping from rounded h-value to (theta_opt, energy).
        Energy is None if the stored value was non-finite.
    """
    import numpy as np

    npz_path = Path(npz_path)
    if not npz_path.exists():
        return {}

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        logger.warning(f"load_npz_as_theta_dict: failed to load {npz_path}: {e}")
        return {}

    e_key = "e_vqe" if "e_vqe" in data else ("energies" if "energies" in data else None)
    has_energies = e_key is not None

    result: dict[float, tuple[np.ndarray, float | None]] = {}
    n_skipped = 0

    for i, h in enumerate(data["h_values"]):
        # Convert θ to float64
        try:
            theta_i = np.asarray(data["theta_opt"][i], dtype=np.float64)
        except (ValueError, TypeError):
            n_skipped += 1
            continue

        # Validate θ: finite and correct dimension
        if not np.all(np.isfinite(theta_i)):
            n_skipped += 1
            continue
        if len(theta_i) != n_params:
            n_skipped += 1
            continue

        # Validate energy
        e_val: float | None = None
        if has_energies:
            e_raw = float(data[e_key][i])
            if np.isfinite(e_raw):
                e_val = e_raw

        result[round(float(h), h_precision)] = (theta_i, e_val)

    if n_skipped > 0:
        logger.info(
            f"load_npz_as_theta_dict: loaded {len(result)}, "
            f"skipped {n_skipped} (dim mismatch, NaN, or corrupt)"
        )
    else:
        logger.info(f"load_npz_as_theta_dict: loaded {len(result)} entries from {npz_path.name}")

    return result


def select_best_theta_init(
    theta_pred: np.ndarray,
    e_pred: float,
    theta_prev: np.ndarray | None,
    e_prev: float | None,
    eval_fn: callable | None = None,
) -> tuple[np.ndarray, float]:
    """Select the best θ for VQE warm-start from {θ_pred, θ_prev}.

    Compares energies from both candidates and returns the one with lower
    energy. If θ_prev has no stored energy but eval_fn is provided,
    evaluates θ_prev to get its energy for comparison.

    This implements the anti-regression init pattern used across runners.

    Parameters
    ----------
    theta_pred : np.ndarray
        MPNN prediction θ vector.
    e_pred : float
        Energy from evaluating theta_pred.
    theta_prev : np.ndarray | None
        Previous best θ from NPZ (may be None if no previous data).
    e_prev : float | None
        Stored energy for theta_prev. If None and eval_fn is provided,
        eval_fn(theta_prev) is called to compute it.
    eval_fn : callable | None
        Function (theta) → float energy. Used to evaluate theta_prev
        when e_prev is None. Should use cached backend for efficiency.

    Returns
    -------
    tuple[np.ndarray, float]
        (best_theta, best_energy) — the candidate with lower energy.
    """
    import numpy as np

    if theta_prev is None:
        return theta_pred.copy(), e_pred

    # If we don't have energy for θ_prev, try to evaluate it
    if e_prev is None and eval_fn is not None:
        try:
            e_prev = float(eval_fn(theta_prev))
        except Exception:
            # Evaluation failed — fall back to θ_pred
            return theta_pred.copy(), e_pred

    # If still no energy for θ_prev, use θ_pred
    if e_prev is None or not np.isfinite(e_prev):
        return theta_pred.copy(), e_pred

    # Pick the one with lower energy
    if e_prev < e_pred:
        return theta_prev.copy(), e_prev
    return theta_pred.copy(), e_pred


def refresh_npz_ground_truth(
    npz_path: Path,
    topology: str,
    n_qubits: int,
    model: str = "tfim_bond_resolved",
) -> int:
    """Refresh e_exact and gaps in an NPZ from the GroundTruthCache.

    Scans the NPZ for each h-value and checks if the GroundTruthCache has
    a more accurate ground truth (lower energy = more converged). Updates
    the NPZ atomically if any values improved.

    This prevents stale e_exact from inflating/deflating ΔE/gap metrics
    when the solver has been upgraded (e.g., DMRG → eigsh, or higher χ).

    Parameters
    ----------
    npz_path : Path
        Path to the .npz file to refresh.
    topology : str
        Lattice topology for GroundTruthCache lookup.
    n_qubits : int
        System size for GroundTruthCache lookup.
    model : str
        Hamiltonian model name (default: "tfim_bond_resolved").

    Returns
    -------
    int
        Number of h-points where e_exact was updated.
    """
    import numpy as np

    npz_path = Path(npz_path)
    if not npz_path.exists():
        return 0

    try:
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        gt_cache = GroundTruthCache()
    except ImportError:
        logger.debug("refresh_npz_ground_truth: GroundTruthCache not available")
        return 0

    try:
        data = np.load(npz_path, allow_pickle=True)
        h_values = data["h_values"]
        e_exact = np.asarray(data["e_exact"], dtype=np.float64)
        gaps = np.asarray(data.get("gaps", np.zeros_like(h_values)), dtype=np.float64)
    except Exception as e:
        logger.warning(f"refresh_npz_ground_truth: failed to load {npz_path.name}: {e}")
        return 0

    n_refreshed = 0
    for i, h in enumerate(h_values):
        cached = gt_cache.get(topology, n_qubits, model, float(h))
        if cached is None:
            continue

        cached_e = cached["energy"]
        cached_gap = cached["gap"]

        # Only update if GT cache has lower energy (more converged)
        # This handles DMRG→eigsh upgrades where eigsh is exact
        if cached_e < e_exact[i] - 1e-10:
            e_exact[i] = cached_e
            if cached_gap > 0:
                gaps[i] = cached_gap
            n_refreshed += 1
        elif cached_gap > 0 and gaps[i] <= 1e-10:
            # Gap was missing, now available from cache
            gaps[i] = cached_gap
            n_refreshed += 1

    if n_refreshed == 0:
        return 0

    # Recompute de_gaps with refreshed e_exact
    e_key = "e_vqe" if "e_vqe" in data else ("energies" if "energies" in data else None)
    if e_key is not None:
        e_vqe = np.asarray(data[e_key], dtype=np.float64)
        de_gaps = np.abs(e_vqe - e_exact) / np.maximum(gaps, 1e-10)
    else:
        de_gaps = np.zeros_like(h_values)

    # Atomic write with all original fields preserved + refreshed values
    # NOTE: Eagerly load theta_opt into memory before writing tmp file,
    # because NpzFile is lazy and the rename could invalidate the mmap.
    theta_opt = data["theta_opt"]
    if hasattr(theta_opt, "copy"):
        theta_opt = theta_opt.copy()  # Force materialization from mmap

    tmp_path = npz_path.with_suffix(".tmp.npz")
    try:
        from datetime import datetime

        save_dict = {
            "h_values": h_values,
            "theta_opt": theta_opt,
            "e_exact": e_exact,
            "gaps": gaps,
            "de_gaps": de_gaps,
            "last_updated": np.array(datetime.now().isoformat()),
        }
        # Preserve energy field under canonical name "e_vqe"
        if e_key is not None:
            save_dict["e_vqe"] = np.asarray(data[e_key])
        # Preserve metadata fields
        if "method" in data:
            save_dict["method"] = (
                data["method"].copy() if hasattr(data["method"], "copy") else data["method"]
            )
        if "quality_tier" in data:
            save_dict["quality_tier"] = (
                data["quality_tier"].copy()
                if hasattr(data["quality_tier"], "copy")
                else data["quality_tier"]
            )

        np.savez(tmp_path, **save_dict)
        tmp_path.rename(npz_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    logger.info(
        f"refresh_npz_ground_truth: {npz_path.name} → "
        f"{n_refreshed}/{len(h_values)} e_exact values refreshed from GroundTruthCache"
    )
    return n_refreshed


def backfill_ground_truth_for_npz(
    npz_path: Path,
    topology: str,
    n_qubits: int,
    model: str = "tfim_bond_resolved",
    *,
    max_points: int | None = None,
) -> int:
    """Compute and cache ground truth for NPZ h-points missing from GT cache.

    Scans the NPZ for each h-value, checks the GroundTruthCache, and for
    any missing entries, computes them via ClassicalSolver and stores in cache.
    This ensures subsequent calls to `refresh_npz_ground_truth` or
    `exact_ground_state` will have cache hits.

    Call this after `upsert_theta_npz` to keep the GT cache synchronized
    with training data. Safe to call repeatedly (idempotent — skips
    already-cached points).

    Parameters
    ----------
    npz_path : Path
        Path to the .npz file to check.
    topology : str
        Lattice topology for computation.
    n_qubits : int
        System size.
    model : str
        Hamiltonian model name (default: "tfim_bond_resolved").
    max_points : int | None
        Limit computation to at most N missing points (useful for
        bounding compute time). None = compute all missing.

    Returns
    -------
    int
        Number of new GT entries computed and cached.
    """
    from pathlib import Path as _Path

    import numpy as np

    npz_path = _Path(npz_path)
    if not npz_path.exists():
        return 0

    try:
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        gt_cache = GroundTruthCache()
    except ImportError:
        logger.debug("backfill_ground_truth_for_npz: GroundTruthCache not available")
        return 0

    # Load h-values from NPZ
    try:
        data = np.load(npz_path, allow_pickle=True)
        h_values = data["h_values"]
    except Exception as e:
        logger.warning("backfill_ground_truth_for_npz: failed to load %s: %s", npz_path.name, e)
        return 0

    # Find missing h-points
    missing_h = []
    for h in h_values:
        if gt_cache.get(topology, n_qubits, model, float(h)) is None:
            missing_h.append(float(h))

    if not missing_h:
        return 0

    if max_points is not None:
        missing_h = missing_h[:max_points]

    # Compute ground truth for missing points
    try:
        from qmbp_simulation import make_lattice
        from qmbp_simulation.models.model_registry import get_model_spec
        from qmbp_simulation.solvers import ClassicalSolver

        spec = get_model_spec(model)
        solver = ClassicalSolver()
        n_computed = 0

        for h in missing_h:
            try:
                lattice = make_lattice(topology, n_qubits, J=1.0, h=h)
                H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
                gt = solver.solve(H, lattice)
                gt_cache.put_from_result(topology, n_qubits, model, h, gt)
                n_computed += 1
            except Exception as e:
                logger.debug("backfill_ground_truth_for_npz: failed for h=%.4f: %s", h, e)
                continue

        gt_cache.flush()

        if n_computed > 0:
            logger.info(
                "backfill_ground_truth_for_npz: %s → computed %d/%d missing GT entries",
                npz_path.name,
                n_computed,
                len(missing_h),
            )
        return n_computed

    except ImportError as e:
        logger.debug("backfill_ground_truth_for_npz: missing dependency: %s", e)
        return 0


def purge_npz_below_frontier(
    npz_path: Path,
    h_frontier: float,
    *,
    archive: bool = True,
) -> tuple[int, int]:
    """Remove NPZ points below h_frontier that harm MPNN training.

    Points with h < h_frontier consistently fail the dual criterion
    (physics limit of HVA p=1). Including them teaches the MPNN incorrect
    θ mappings. This function:
    1. Archives the original NPZ (if archive=True)
    2. Removes all points with h < h_frontier
    3. Writes a cleaned NPZ with only viable points

    Parameters
    ----------
    npz_path : Path
        Path to the .npz file to clean.
    h_frontier : float
        The frontier h-value. All points with h < h_frontier are removed.
    archive : bool
        If True (default), save original to `<name>_pre_purge.npz` before modifying.

    Returns
    -------
    tuple[int, int]
        (n_removed, n_kept) — counts of removed and remaining points.

    Raises
    ------
    ValueError
        If h_frontier would remove ALL points (safety check).
    """
    from pathlib import Path as _Path

    import numpy as np

    npz_path = _Path(npz_path)
    if not npz_path.exists():
        return 0, 0

    data = np.load(npz_path, allow_pickle=True)
    h_values = data["h_values"]
    n_total = len(h_values)

    # Safety: never purge everything
    keep_mask = h_values >= h_frontier
    n_kept = int(keep_mask.sum())
    n_removed = n_total - n_kept

    if n_removed == 0:
        return 0, n_total

    if n_kept == 0:
        raise ValueError(
            f"purge_npz_below_frontier: h_frontier={h_frontier:.4f} would remove ALL "
            f"{n_total} points from {npz_path.name}. Aborting (safety check)."
        )

    if n_kept < 5:
        logger.warning(
            "purge_npz_below_frontier: only %d points would remain in %s after purge. "
            "Consider if this is enough for training.",
            n_kept,
            npz_path.name,
        )

    # Archive original
    if archive:
        import shutil

        archive_path = npz_path.with_stem(npz_path.stem + "_pre_purge")
        if not archive_path.exists():
            shutil.copy2(npz_path, archive_path)
            logger.info("  Archived original: %s", archive_path.name)

    # Build cleaned arrays
    theta_opt = data["theta_opt"]
    if hasattr(theta_opt, "copy"):
        theta_opt = theta_opt.copy()

    save_dict = {
        "h_values": h_values[keep_mask],
        "theta_opt": theta_opt[keep_mask],
    }

    # Preserve all known fields
    for field in ["e_vqe", "energies", "e_exact", "gaps", "de_gaps"]:
        if field in data:
            save_dict[field] = np.asarray(data[field])[keep_mask]

    # Handle string/object arrays (method, quality_tier)
    for field in ["method", "quality_tier"]:
        if field in data:
            arr = data[field]
            filtered = [arr[i] for i in range(n_total) if keep_mask[i]]
            save_dict[field] = np.array(filtered)

    # Recompute de_gaps if both energy fields present
    e_key = "e_vqe" if "e_vqe" in save_dict else ("energies" if "energies" in save_dict else None)
    if e_key and "e_exact" in save_dict and "gaps" in save_dict:
        abs_err = np.abs(save_dict[e_key] - save_dict["e_exact"])
        save_dict["de_gaps"] = abs_err / np.maximum(save_dict["gaps"], 1e-10)

    # Add metadata
    from datetime import datetime

    save_dict["last_updated"] = np.array(datetime.now().isoformat())

    # Atomic write
    tmp_path = npz_path.with_suffix(".tmp.npz")
    try:
        np.savez(tmp_path, **save_dict)
        tmp_path.rename(npz_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    logger.info(
        "purge_npz_below_frontier: %s → removed %d pts (h < %.2f), kept %d",
        npz_path.name,
        n_removed,
        h_frontier,
        n_kept,
    )
    return n_removed, n_kept


def build_clean_training_dataset(
    topology: str,
    p_layers: int = 1,
    *,
    model: str = "tfim_bond_resolved",
    reject_below_frontier: bool = True,
    reject_not_useful: bool = True,
    h_frontier_override: dict[int, float] | None = None,
    min_quality_tier: str = "unverified",
) -> dict:
    """Build a unified, clean training dataset from all NPZ files for a topology.

    Aggregates data from ``data/multi_n_training/{topology}_N*_p{p}.npz``,
    applying exclusion policies to produce a dataset ready for
    ``train_unified_mpnn()``.

    Exclusion policies (all optional):
    - ``reject_below_frontier``: removes points with h < h_frontier(N)
      (uses monotonicity-corrected frontiers from dashboard)
    - ``reject_not_useful``: skips entire NPZ files classified as "not_useful"
      by ``classify_training_utility``
    - ``min_quality_tier``: filters by quality_tier field in NPZ
      ("verified" > "approximate" > "unverified")

    Parameters
    ----------
    topology : str
        Lattice topology (chain_1d, ladder, square, triangular, heavy_hex).
    p_layers : int
        HVA depth (default: 1).
    model : str
        Hamiltonian model name.
    reject_below_frontier : bool
        If True, exclude h-points below the (corrected) h_frontier for each N.
    reject_not_useful : bool
        If True, skip NPZ files where training_utility == "not_useful".
    h_frontier_override : dict[int, float] | None
        Manual h_frontier per N. Overrides dashboard values if provided.
    min_quality_tier : str
        Minimum quality tier to include. "verified" = strictest,
        "unverified" = include everything.

    Returns
    -------
    dict
        {
            "h_values": np.ndarray,         # (N_total,)
            "theta_opt": np.ndarray,        # (N_total, n_params)
            "e_vqe": np.ndarray,            # (N_total,)
            "e_exact": np.ndarray,          # (N_total,)
            "gaps": np.ndarray,             # (N_total,)
            "n_qubits_per_point": np.ndarray,  # (N_total,) — which N each point belongs to
            "n_values_used": list[int],     # sorted list of N values included
            "n_points_per_n": dict[int,int],# count per N
            "n_excluded": int,              # total points excluded
            "exclusion_reasons": dict,      # {reason: count}
        }
    """
    from pathlib import Path as _Path

    import numpy as np

    _ROOT = _Path(__file__).resolve().parents[3]
    npz_dir = _ROOT / "data" / "multi_n_training"

    tier_rank = {"verified": 2, "approximate": 1, "unverified": 0}
    min_tier_val = tier_rank.get(min_quality_tier, 0)

    # ── Load h_frontier from dashboard (with monotonicity correction) ────
    h_frontiers: dict[int, float] = {}
    if reject_below_frontier:
        if h_frontier_override:
            h_frontiers = dict(h_frontier_override)
        else:
            try:
                dashboard_path = _ROOT / "data" / "model_quality_dashboard.json"
                if dashboard_path.exists():
                    import json

                    with open(dashboard_path) as f:
                        dashboard = json.load(f)
                    configs = [
                        c
                        for c in dashboard.get("configs", [])
                        if c["topology"] == topology and c.get("p_layers", 1) == p_layers
                    ]
                    # Apply monotonicity correction
                    from qmbp_simulation.analysis.metrics import enforce_h_frontier_monotonicity

                    enforce_h_frontier_monotonicity(configs)
                    for c in configs:
                        if c.get("h_frontier") is not None:
                            h_frontiers[c["n_qubits"]] = c["h_frontier"]
            except Exception:
                pass  # Proceed without frontier filtering

    # ── Determine which NPZ files to skip (not_useful) ───────────────────
    skip_n_values: set[int] = set()
    if reject_not_useful:
        try:
            dashboard_path = _ROOT / "data" / "model_quality_dashboard.json"
            if dashboard_path.exists():
                import json

                with open(dashboard_path) as f:
                    dashboard = json.load(f)
                for c in dashboard.get("configs", []):
                    if c["topology"] == topology and c.get("p_layers", 1) == p_layers:
                        if c.get("training_utility") == "not_useful":
                            skip_n_values.add(c["n_qubits"])
        except Exception:
            pass

    # ── Aggregate data from NPZ files ────────────────────────────────────
    all_h, all_theta, all_e_vqe, all_e_exact, all_gaps, all_n = [], [], [], [], [], []
    n_excluded = 0
    exclusion_reasons: dict[str, int] = {}
    n_points_per_n: dict[int, int] = {}

    pattern = f"{topology}_N*_p{p_layers}.npz"
    npz_files = sorted(npz_dir.glob(pattern)) if npz_dir.exists() else []

    for npz_file in npz_files:
        # Parse N from filename
        stem = npz_file.stem
        parts = stem.split("_")
        n_idx = next((i for i, p in enumerate(parts) if p.startswith("N")), None)
        if n_idx is None:
            continue
        try:
            n_qubits = int(parts[n_idx][1:])
        except ValueError:
            continue

        # Skip not_useful
        if n_qubits in skip_n_values:
            exclusion_reasons["not_useful_config"] = (
                exclusion_reasons.get("not_useful_config", 0) + 1
            )
            continue

        # Load data
        try:
            data = np.load(npz_file, allow_pickle=True)
        except Exception:
            continue

        h_values = data["h_values"]
        theta_opt = data["theta_opt"]
        e_key = "e_vqe" if "e_vqe" in data else ("energies" if "energies" in data else None)
        if e_key is None or "e_exact" not in data:
            continue

        e_vqe = np.asarray(data[e_key], dtype=np.float64)
        e_exact = np.asarray(data["e_exact"], dtype=np.float64)
        gaps = np.asarray(data.get("gaps", np.ones(len(h_values))), dtype=np.float64)
        tiers = (
            data["quality_tier"].tolist()
            if "quality_tier" in data
            else ["unverified"] * len(h_values)
        )

        # Per-point filtering
        h_frontier = h_frontiers.get(n_qubits, 0.0) if reject_below_frontier else 0.0
        n_kept_this_file = 0

        for i in range(len(h_values)):
            # Filter: below frontier
            if float(h_values[i]) < h_frontier - 1e-6:
                n_excluded += 1
                exclusion_reasons["below_frontier"] = exclusion_reasons.get("below_frontier", 0) + 1
                continue

            # Filter: quality tier
            tier_val = tier_rank.get(str(tiers[i]), 0)
            if tier_val < min_tier_val:
                n_excluded += 1
                exclusion_reasons["below_min_tier"] = exclusion_reasons.get("below_min_tier", 0) + 1
                continue

            # Filter: NaN/Inf
            if not np.all(np.isfinite(theta_opt[i])) or not np.isfinite(e_vqe[i]):
                n_excluded += 1
                exclusion_reasons["nan_inf"] = exclusion_reasons.get("nan_inf", 0) + 1
                continue

            all_h.append(float(h_values[i]))
            all_theta.append(np.asarray(theta_opt[i], dtype=np.float64))
            all_e_vqe.append(float(e_vqe[i]))
            all_e_exact.append(float(e_exact[i]))
            all_gaps.append(float(gaps[i]))
            all_n.append(n_qubits)
            n_kept_this_file += 1

        if n_kept_this_file > 0:
            n_points_per_n[n_qubits] = n_points_per_n.get(n_qubits, 0) + n_kept_this_file

    n_values_used = sorted(n_points_per_n.keys())

    # Handle empty result
    if not all_h:
        return {
            "h_values": np.array([]),
            "theta_opt": np.array([], dtype=object),
            "e_vqe": np.array([]),
            "e_exact": np.array([]),
            "gaps": np.array([]),
            "n_qubits_per_point": np.array([], dtype=int),
            "n_values_used": [],
            "n_points_per_n": {},
            "n_excluded": n_excluded,
            "exclusion_reasons": exclusion_reasons,
        }

    # theta_opt may be ragged (different N → different n_params).
    # If all same shape → regular 2D array; otherwise → object array.
    theta_shapes = {t.shape for t in all_theta}
    if len(theta_shapes) == 1:
        theta_arr = np.array(all_theta, dtype=np.float64)
    else:
        theta_arr = np.array(all_theta, dtype=object)

    return {
        "h_values": np.array(all_h),
        "theta_opt": theta_arr,
        "e_vqe": np.array(all_e_vqe),
        "e_exact": np.array(all_e_exact),
        "gaps": np.array(all_gaps),
        "n_qubits_per_point": np.array(all_n, dtype=int),
        "n_values_used": n_values_used,
        "n_points_per_n": n_points_per_n,
        "n_excluded": n_excluded,
        "exclusion_reasons": exclusion_reasons,
    }


def load_theta_npz(npz_path: Path) -> dict[str, np.ndarray]:
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
