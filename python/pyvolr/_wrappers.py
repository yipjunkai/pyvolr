"""Internal FFI wrapper helpers shared by every Python-facing pricing module.

The Rust core (`pyvolr._core`) only accepts flat 1-D contiguous arrays of equal
length. This module owns the contract between that surface and the public
numpy-broadcasting API. Every `pyvolr.<model>` module must route through
these helpers — do not reimplement broadcasting, flag normalization, or the
scalar-collapse rule anywhere else.

The leading underscore marks this as an internal module; downstream users
should depend on `pyvolr.bs` / `pyvolr.black76`, not on the helpers here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "FlagInput",
    "Result",
    "broadcast_f64",
    "normalize_flag",
    "scalar_or_array",
]

FlagInput = ArrayLike | str
Result = float | NDArray[np.float64]


def normalize_flag(flag: FlagInput, shape: tuple[int, ...]) -> NDArray[np.int8]:
    """Convert a flag input into an int8 array of the given broadcast shape.

    Accepts:
        - a string: 'c'/'C' (call) or 'p'/'P' (put)
        - an ndarray of strings
        - an ndarray of ints (±1, where >=0 is call, <0 is put)
    """
    if isinstance(flag, str):
        s = flag.lower()
        if s not in ("c", "p"):
            raise ValueError(f"flag must be 'c' or 'p', got {flag!r}")
        return np.full(shape, 1 if s == "c" else -1, dtype=np.int8)

    arr = np.asarray(flag)
    if arr.dtype.kind in ("U", "S", "O"):
        lower_flat = np.array([str(x).lower() for x in arr.ravel()], dtype="U1")
        lower = np.reshape(lower_flat, arr.shape)
        if not np.isin(lower, ("c", "p")).all():
            raise ValueError("flag array must contain only 'c' or 'p' (case-insensitive)")
        encoded = np.where(lower == "c", 1, -1).astype(np.int8)
        return np.ascontiguousarray(np.broadcast_to(encoded, shape))
    return np.ascontiguousarray(np.broadcast_to(arr.astype(np.int8), shape))


def broadcast_f64(*arrs: ArrayLike) -> tuple[list[NDArray[np.float64]], tuple[int, ...]]:
    """Broadcast inputs to a common shape; return flat contiguous f64 arrays."""
    cast = [np.asarray(a, dtype=np.float64) for a in arrs]
    bcast = np.broadcast_arrays(*cast)
    shape = bcast[0].shape
    flat = [np.ascontiguousarray(a.ravel()) for a in bcast]
    return flat, shape


def scalar_or_array(arr: NDArray[np.float64], shape: tuple[int, ...]) -> Result:
    """If the broadcast shape is () return a Python scalar; otherwise reshape."""
    if shape == ():
        return float(arr[0])
    return np.reshape(arr, shape)
