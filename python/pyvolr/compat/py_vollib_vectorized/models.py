"""Vectorized pricing — drop-in for ``py_vollib_vectorized.models``.

Thin wrappers over the native vectorized core (``pyvolr.bs`` / ``pyvolr.black76``)
with py_vollib_vectorized's signatures, ``return_as`` conventions, and the
``"Price"`` output column.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyvolr import black76 as _b76
from pyvolr import bs as _bs

from ._format import as_1d, format_single

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike

    from ._format import SingleResult

__all__ = [
    "vectorized_black",
    "vectorized_black_scholes",
    "vectorized_black_scholes_merton",
]


def vectorized_black_scholes(
    flag: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    t: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: str = "dataframe",
    dtype: DTypeLike = np.float64,
) -> SingleResult:
    """Black-Scholes price (no dividend) for arrays of contracts."""
    out = as_1d(_bs.price(flag, S, K, t, r, sigma), dtype)
    return format_single(out, "Price", return_as)


def vectorized_black_scholes_merton(
    flag: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    t: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike,
    *,
    return_as: str = "dataframe",
    dtype: DTypeLike = np.float64,
) -> SingleResult:
    """Black-Scholes-Merton price (continuous dividend ``q``) for arrays of contracts."""
    out = as_1d(_bs.price(flag, S, K, t, r, sigma, q), dtype)
    return format_single(out, "Price", return_as)


def vectorized_black(
    flag: ArrayLike,
    F: ArrayLike,
    K: ArrayLike,
    t: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: str = "dataframe",
    dtype: DTypeLike = np.float64,
) -> SingleResult:
    """Black-76 price on a forward/future ``F`` for arrays of contracts."""
    out = as_1d(_b76.price(flag, F, K, t, r, sigma), dtype)
    return format_single(out, "Price", return_as)
