"""Drop-in shim for `py_vollib.black.implied_volatility`.

py_vollib's signature here puts `r` BEFORE `t` (opposite of
`py_vollib.black_scholes.implied_volatility`, which is `t, r`). This shim
preserves that exact argument order — flag is last either way.
"""

from __future__ import annotations

import math

from pyvolr import black76 as _b76

from .._bounds import raise_for_iv_price

__all__ = ["implied_volatility"]


def implied_volatility(price: float, F: float, K: float, r: float, t: float, flag: str) -> float:
    """Black-76 implied volatility from a market price.

    Argument order matches py_vollib.black exactly: (price, F, K, r, t, flag).

    Mirrors py_vollib's `BelowIntrinsicException` / `AboveMaximumException`
    contract (from `pyvolr.compat.py_vollib.exceptions`) on out-of-bounds
    prices; the modern `pyvolr.black76.implied_vol` returns NaN instead.
    """
    # on_error="ignore": the shim does its own out-of-bounds handling below.
    sigma = float(_b76.implied_vol(price, flag, F, K, t, r, on_error="ignore"))
    if math.isnan(sigma):
        # Black-76: underlying is the forward F, discounted at r (q = r).
        disc_r = math.exp(-r * t)
        raise_for_iv_price(price, F * disc_r, K * disc_r, flag)
    return sigma
