"""Vectorized implied volatility — drop-in for ``py_vollib_vectorized.implied_volatility``.

Argument orders match py_vollib_vectorized exactly (they are deliberately quirky):
    ``vectorized_implied_volatility(price, S, K, t, r, flag, q=None, ...)``  -- flag AFTER r
    ``vectorized_implied_volatility_black(price, F, K, r, t, flag, ...)``    -- r and t SWAPPED

``on_error`` (``"warn" | "raise" | "ignore"``) is passed straight through to the
native solver, which surfaces failures via ``pyvolr.ImpliedVolError`` /
``ImpliedVolWarning`` (pyvolr's types, not py_vollib's).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyvolr import black76 as _b76
from pyvolr import bs as _bs

from ._format import as_1d, format_single

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import ArrayLike, DTypeLike

    from ._format import SingleResult

__all__ = ["vectorized_implied_volatility", "vectorized_implied_volatility_black"]


def vectorized_implied_volatility(
    price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    t: ArrayLike,
    r: ArrayLike,
    flag: ArrayLike,
    q: ArrayLike | None = None,
    *,
    on_error: Literal["warn", "raise", "ignore"] = "warn",
    model: str = "black_scholes",
    return_as: str = "dataframe",
    dtype: DTypeLike = np.float64,
    **kwargs: object,
) -> SingleResult:
    """Implied volatility for arrays of contracts (column ``"IV"``)."""
    del kwargs  # accepted for py_vollib_vectorized signature parity; unused
    if model == "black_scholes":
        out = _bs.implied_vol(price, flag, S, K, t, r, on_error=on_error)
    elif model == "black_scholes_merton":
        out = _bs.implied_vol(price, flag, S, K, t, r, 0.0 if q is None else q, on_error=on_error)
    elif model == "black":
        out = _b76.implied_vol(price, flag, S, K, t, r, on_error=on_error)
    else:
        raise ValueError("model must be 'black_scholes', 'black_scholes_merton', or 'black'")
    return format_single(as_1d(out, dtype), "IV", return_as)


def vectorized_implied_volatility_black(
    price: ArrayLike,
    F: ArrayLike,
    K: ArrayLike,
    r: ArrayLike,
    t: ArrayLike,
    flag: ArrayLike,
    *,
    on_error: Literal["warn", "raise", "ignore"] = "warn",
    return_as: str = "dataframe",
    dtype: DTypeLike = np.float64,
    **kwargs: object,
) -> SingleResult:
    """Black-76 implied volatility (note: ``r`` precedes ``t`` in this signature)."""
    del kwargs
    out = _b76.implied_vol(price, flag, F, K, t, r, on_error=on_error)
    return format_single(as_1d(out, dtype), "IV", return_as)
