"""Drop-in shim for `py_vollib.black_scholes_merton.implied_volatility`."""

from __future__ import annotations

import math

from pyvolr import bs as _bs

from .._bounds import raise_for_iv_price

__all__ = ["implied_volatility"]


def implied_volatility(
    price: float, S: float, K: float, t: float, r: float, q: float, flag: str
) -> float:
    """Solve for implied volatility from a market price (BSM with dividend yield).

    Mirrors py_vollib's `BelowIntrinsicException` / `AboveMaximumException`
    contract (from `pyvolr.compat.py_vollib.exceptions`) on out-of-bounds
    prices; the modern `pyvolr.bs.implied_vol` returns NaN instead.
    """
    # on_error="ignore": the shim does its own out-of-bounds handling below.
    sigma = float(_bs.implied_vol(price, flag, S, K, t, r, q, on_error="ignore"))
    if math.isnan(sigma):
        raise_for_iv_price(price, S * math.exp(-q * t), K * math.exp(-r * t), flag)
    return sigma
