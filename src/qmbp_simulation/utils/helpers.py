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


# ─────────────────────────────────────────────────────────────────────────────
# Theta Canonicalization (HVA parameter gauge symmetry)
# ─────────────────────────────────────────────────────────────────────────────


def canonicalize_theta(theta: np.ndarray, *, period: float = np.pi) -> np.ndarray:
    """Canonicalize VQE parameters to the fundamental domain.

    The HVA circuit has gauge symmetries that make multiple θ values produce
    the same quantum state:

    1. **Periodicity**: RZZ(2θ) and RX(2θ) both have period π in θ.
       Verified: |ψ(θ)⟩ = |ψ(θ+π)⟩ (fidelity=1.0, same state).
    2. **Z₂ symmetry**: (-θ_zz, -θ_x) gives the same *energy* but a
       different state (fidelity≈0.9998). We canonicalize by sign to ensure
       consistent MPNN targets.

    This function maps θ to a canonical representative:
    - Wrap each parameter to [-period/2, period/2] using modular arithmetic.
    - Apply Z₂ convention: ensure the last parameter is non-negative.

    Parameters
    ----------
    theta : np.ndarray or array-like
        Parameter vector from VQE optimization. Shape (n_params,).
    period : float
        Periodicity of the gate parameters. Default π for standard HVA
        (RZZ(2θ) and RX(2θ) both have period π in θ). Use 2π for circuits
        with single-angle rotations (RZ(θ), RX(θ)).

    Returns
    -------
    np.ndarray
        Canonicalized θ in the fundamental domain.

    Notes
    -----
    This function handles the most common gauge equivalences in HVA circuits.
    It does NOT detect genuine local minima with different energy — those must
    be filtered by energy comparison or `filter_consistent_theta`.

    For bond-resolved HVA (many parameters per layer), translational invariance
    can create additional equivalences not handled here.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.size == 0:
        return theta

    result = theta.copy()
    half_period = period / 2.0

    # Step 1: Wrap each parameter to [-period/2, period/2]
    result = ((result + half_period) % period) - half_period

    # Step 2: Z₂ convention — ensure last parameter is non-negative
    if result[-1] < 0:
        result = -result

    return result


def filter_consistent_theta(
    theta_array: np.ndarray,
    *,
    outlier_sigma: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter θ_opt array to remove points in different local minima.

    After canonicalization (mod-π + Z₂), most gauge-equivalent θ converge
    to the same canonical value. However, the VQE can still find genuine
    local minima (different state, different energy) that canonicalization
    cannot fix. These appear as outliers far from the cluster of normal points.

    Uses robust MAD-based outlier detection: removes points whose distance
    from the median θ exceeds `outlier_sigma × MAD`.

    Parameters
    ----------
    theta_array : np.ndarray
        Array of canonicalized θ vectors, shape (n_points, n_params).
        MUST be canonicalized first (call canonicalize_theta on each row).
    outlier_sigma : float
        Number of MAD-scaled deviations to consider as outlier. Default 5.0
        (very conservative — only catches gross outliers like basin jumps
        where Δθ ≈ π/2 or larger).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (filtered_theta, mask) where mask[i] = True if point i was kept.

    Notes
    -----
    For typical TFIM HVA p=1 data:
    - Normal cluster: θ_zz ∈ [0.04, 0.12], θ_x ≈ π/8. MAD ≈ 0.01.
    - Periodic basin outliers: θ_x ≈ 3π/8 (distance ~0.8 from cluster).
    - Threshold at 5σ: 0.01 × 5 × 1.48 ≈ 0.07 → catches outliers at 0.8.

    For random/synthetic data (property tests):
    - MAD is large (~0.3+) → threshold is large → nothing gets filtered.
    """
    theta_array = np.asarray(theta_array, dtype=float)

    if len(theta_array) < 3:
        return theta_array, np.ones(len(theta_array), dtype=bool)

    # Compute distance of each point from the median (robust center)
    median_theta = np.median(theta_array, axis=0)
    distances = np.linalg.norm(theta_array - median_theta, axis=1)

    # MAD (median absolute deviation) — robust scale estimator
    med_dist = np.median(distances)
    mad = np.median(np.abs(distances - med_dist))

    if mad < 1e-10:
        # All points are nearly identical — no outliers detectable
        return theta_array, np.ones(len(theta_array), dtype=bool)

    # Adaptive threshold: median_distance + sigma × scaled_MAD
    # The 1.4826 factor converts MAD to Gaussian-equivalent σ
    threshold = med_dist + outlier_sigma * 1.4826 * mad
    mask = distances <= threshold

    return theta_array[mask], mask


def augment_theta_symmetries(
    theta: np.ndarray,
    *,
    period: float = np.pi,
    include_z2: bool = True,
    include_shift: bool = False,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Generate symmetry-equivalent θ variants for data augmentation.

    HVA circuits have gauge symmetries that produce identical or
    near-identical quantum states. This function exploits these symmetries
    to multiply training data without additional VQE cost.

    Symmetries used:
    1. **Z₂ reflection**: (-θ) produces the same energy as (+θ) for TFIM HVA.
    2. **Period shift**: (θ + π) produces the same state as θ (periodicity).
    3. **Optional noise**: small Gaussian perturbation for regularization.

    Parameters
    ----------
    theta : np.ndarray
        Single parameter vector, shape (n_params,).
    period : float
        Periodicity of gate parameters. Default π (standard HVA).
    include_z2 : bool
        Include Z₂ reflection (-θ). Default True.
    include_shift : bool
        Include period-shifted variants. Default False (risk of
        creating out-of-canonical-domain points if not careful).
    noise_std : float
        Standard deviation of Gaussian noise to add. Default 0.0 (no noise).
        Values like 0.01-0.05 provide regularization without degrading quality.
    seed : int | None
        Random seed for noise generation.

    Returns
    -------
    list[np.ndarray]
        List of augmented θ variants (does NOT include the original).
        Each variant is canonicalized to [-π/2, π/2].

    Example
    -------
    >>> theta = np.array([0.3, -0.1, 0.5])
    >>> augmented = augment_theta_symmetries(theta, include_z2=True)
    >>> len(augmented)  # 1 variant from Z₂
    1
    >>> # With noise: generates 1 noisy variant per symmetry
    >>> augmented = augment_theta_symmetries(theta, noise_std=0.02, seed=42)
    >>> len(augmented)  # 1 Z₂ + 1 noisy Z₂ = 2 (if include_z2=True)
    """
    theta = np.asarray(theta, dtype=float)
    if theta.size == 0:
        return []

    variants: list[np.ndarray] = []
    half_period = period / 2.0

    # Z₂ symmetry: -θ gives same energy for TFIM HVA
    if include_z2:
        z2 = -theta.copy()
        # Canonicalize
        z2 = ((z2 + half_period) % period) - half_period
        variants.append(z2)

    # Period shift: θ + π (same state due to 2θ in gates)
    if include_shift:
        shifted = theta + period
        shifted = ((shifted + half_period) % period) - half_period
        # Only add if different from original (after wrapping)
        if not np.allclose(shifted, theta, atol=1e-6):
            variants.append(shifted)

    # Gaussian noise augmentation
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        n_base = len(variants) if variants else 1
        for base in ([theta] + variants)[:n_base]:
            noisy = base + rng.normal(0, noise_std, size=base.shape)
            noisy = ((noisy + half_period) % period) - half_period
            variants.append(noisy)

    return variants
