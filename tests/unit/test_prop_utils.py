"""Property-based tests for qmbp_simulation.utils module.

# Feature: framework-restructure, Property 17: Seed determinism
# Feature: framework-restructure, Property 18: JSON serialization handles numpy types
"""

from __future__ import annotations

import json
import random

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from qmbp_simulation.utils import json_serialize, set_global_seed

# ---------------------------------------------------------------------------
# Property 17: Seed determinism
# For any integer seed, calling set_global_seed(seed) twice and generating
# random numbers after each call SHALL produce identical sequences.
# **Validates: Requirements 11.1**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_seed_determinism_numpy(seed: int) -> None:
    """Same seed produces identical numpy random sequences.

    **Validates: Requirements 11.1**
    """
    set_global_seed(seed)
    seq_a = np.random.rand(10)

    set_global_seed(seed)
    seq_b = np.random.rand(10)

    np.testing.assert_array_equal(seq_a, seq_b)


@settings(max_examples=100)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_seed_determinism_python_random(seed: int) -> None:
    """Same seed produces identical Python random sequences.

    **Validates: Requirements 11.1**
    """
    set_global_seed(seed)
    seq_a = [random.random() for _ in range(10)]

    set_global_seed(seed)
    seq_b = [random.random() for _ in range(10)]

    assert seq_a == seq_b


@settings(max_examples=100)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_seed_determinism_torch(seed: int) -> None:
    """Same seed produces identical torch random sequences.

    **Validates: Requirements 11.1**
    """
    try:
        import torch
    except ImportError:
        return  # Skip if torch not available

    set_global_seed(seed)
    seq_a = torch.randn(10)

    set_global_seed(seed)
    seq_b = torch.randn(10)

    assert torch.equal(seq_a, seq_b)


# ---------------------------------------------------------------------------
# Property 18: JSON serialization handles numpy types
# For any numpy array, numpy scalar, or dataclass instance, json_serialize()
# SHALL produce a JSON-serializable Python object (no numpy types remain).
# **Validates: Requirements 11.2**
# ---------------------------------------------------------------------------


def _contains_numpy_types(obj) -> bool:
    """Recursively check if any numpy types remain in the serialized output."""
    if isinstance(obj, np.generic | np.ndarray):
        return True
    if isinstance(obj, dict):
        return any(_contains_numpy_types(k) or _contains_numpy_types(v) for k, v in obj.items())
    if isinstance(obj, list | tuple):
        return any(_contains_numpy_types(item) for item in obj)
    return False


# Strategy for numpy dtypes
numpy_dtypes = st.sampled_from([np.int32, np.int64, np.float32, np.float64, np.bool_])

# Strategy for numpy arrays with various dtypes and shapes
numpy_arrays = st.one_of(
    # 1D arrays
    st.builds(
        lambda dtype, size: np.random.default_rng(42).standard_normal(size).astype(dtype)
        if np.issubdtype(dtype, np.floating)
        else np.random.default_rng(42).integers(0, 100, size=size).astype(dtype)
        if np.issubdtype(dtype, np.integer)
        else np.random.default_rng(42).integers(0, 2, size=size).astype(dtype),
        dtype=numpy_dtypes,
        size=st.integers(min_value=1, max_value=20),
    ),
    # 2D arrays
    st.builds(
        lambda dtype, rows, cols: np.random.default_rng(42)
        .standard_normal((rows, cols))
        .astype(dtype)
        if np.issubdtype(dtype, np.floating)
        else np.random.default_rng(42).integers(0, 100, size=(rows, cols)).astype(dtype)
        if np.issubdtype(dtype, np.integer)
        else np.random.default_rng(42).integers(0, 2, size=(rows, cols)).astype(dtype),
        dtype=numpy_dtypes,
        rows=st.integers(min_value=1, max_value=5),
        cols=st.integers(min_value=1, max_value=5),
    ),
)

# Strategy for numpy scalars
numpy_scalars = st.one_of(
    st.builds(np.int32, st.integers(min_value=-1000, max_value=1000)),
    st.builds(np.int64, st.integers(min_value=-1000, max_value=1000)),
    st.builds(
        np.float32, st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
    ),
    st.builds(
        np.float64, st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
    ),
    st.builds(np.bool_, st.booleans()),
)


@settings(max_examples=100)
@given(arr=numpy_arrays)
def test_json_serialize_numpy_array_no_numpy_types(arr: np.ndarray) -> None:
    """json_serialize on numpy arrays produces no numpy types in output.

    **Validates: Requirements 11.2**
    """
    result = json_serialize(arr)
    assert not _contains_numpy_types(result), (
        f"numpy types remain after serialization of {arr.dtype} array"
    )


@settings(max_examples=100)
@given(arr=numpy_arrays)
def test_json_serialize_numpy_array_is_json_serializable(arr: np.ndarray) -> None:
    """json_serialize on numpy arrays produces JSON-serializable output.

    **Validates: Requirements 11.2**
    """
    result = json_serialize(arr)
    # json.dumps should not raise
    json_str = json.dumps(result)
    assert isinstance(json_str, str)


@settings(max_examples=100)
@given(scalar=numpy_scalars)
def test_json_serialize_numpy_scalar_no_numpy_types(scalar) -> None:
    """json_serialize on numpy scalars produces no numpy types in output.

    **Validates: Requirements 11.2**
    """
    result = json_serialize(scalar)
    assert not _contains_numpy_types(result), (
        f"numpy types remain after serialization of {type(scalar).__name__}"
    )


@settings(max_examples=100)
@given(scalar=numpy_scalars)
def test_json_serialize_numpy_scalar_is_json_serializable(scalar) -> None:
    """json_serialize on numpy scalars produces JSON-serializable output.

    **Validates: Requirements 11.2**
    """
    result = json_serialize(scalar)
    json_str = json.dumps(result)
    assert isinstance(json_str, str)
