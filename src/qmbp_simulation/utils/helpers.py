"""
Shared utility functions — seeding, JSON serialization, and timing.

This module is the leaf node of the dependency graph: it has NO imports
from other qmbp_simulation submodules.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────


def set_global_seed(seed: int) -> None:
    """Seed NumPy, PyTorch, and Python random for reproducibility.

    Parameters
    ----------
    seed : int
        Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# JSON Serialization
# ─────────────────────────────────────────────────────────────────────────────


def json_serialize(obj: Any) -> Any:
    """Recursively convert Python/numpy objects to JSON-serializable types.

    Handles:
    - numpy bool → bool
    - numpy arrays → list
    - numpy integer/floating scalars → int/float
    - dataclasses → dict (via asdict)
    - datetime → ISO format string
    - Path objects → str
    - NaN/Inf floats → None

    Parameters
    ----------
    obj : Any
        Object to serialize.

    Returns
    -------
    Any
        JSON-serializable equivalent.
    """
    if obj is None:
        return None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if is_dataclass(obj) and not isinstance(obj, type):
        return json_serialize(asdict(obj))
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [json_serialize(item) for item in obj]
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, int | str | bool):
        return obj
    # Fallback: try numeric conversion, else str
    try:
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    except (TypeError, ValueError):
        return str(obj)


def json_dump(obj: Any, path: Path, indent: int = 2) -> None:
    """Serialize obj to JSON and write to path.

    Uses `json_serialize` as the default handler for non-standard types.

    Parameters
    ----------
    obj : Any
        Object to serialize (typically a dict).
    path : Path
        Output file path.
    indent : int
        JSON indentation level (default 2).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=json_serialize)


# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TimerResult:
    """Result from the timer context manager.

    Attributes
    ----------
    elapsed_s : float
        Wall-clock elapsed time in seconds.
    label : str
        Descriptive label for the timed block.
    """

    elapsed_s: float = 0.0
    label: str = ""


@contextmanager
def timer(label: str = "") -> Generator[TimerResult, None, None]:
    """Context manager that measures wall-clock time.

    Usage
    -----
    >>> with timer("phase1") as t:
    ...     do_work()
    >>> print(f"{t.label} took {t.elapsed_s:.2f}s")

    Parameters
    ----------
    label : str
        Descriptive label for the timed block.

    Yields
    ------
    TimerResult
        Mutable result object; `elapsed_s` is set on exit.
    """
    result = TimerResult(label=label)
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_s = time.perf_counter() - start
