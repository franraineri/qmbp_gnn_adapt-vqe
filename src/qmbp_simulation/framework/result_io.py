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
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from qmbp_simulation.utils.helpers import json_serialize

_DEFAULT_RESULTS_ROOT = Path("results/experiments")
_DEFAULT_PIPELINE_ROOT = Path("results/pipeline")


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
    exp_dir = root / f"exp_{experiment_id.lower()}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    ts = timestamp or generate_timestamp()
    filepath = exp_dir / f"run_{ts}.json"

    _write_json(data, filepath)
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
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    with open(path) as f:
        return json.load(f)


def _write_json(data: dict[str, Any], path: Path) -> None:
    """Write data to JSON with numpy/dataclass serialization support.

    Parameters
    ----------
    data : dict
        Data to serialize.
    path : Path
        Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    """JSON serializer fallback for non-standard types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    # Fallback
    return json_serialize(obj)
