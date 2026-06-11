"""Drop-in shim for `py_vollib.black_scholes.implied_volatility`."""

from __future__ import annotations

import math

from pyvolr import bs as _bs

from .._bounds import raise_for_iv_price

__all__ = ["implied_volatility"]


def implied_volatility(price: float, S: float, K: float, t: float, r: float, flag: str) -> float:
    """Solve for implied volatility from a market price.

    Note: py_vollib's parameter ORDER places `flag` LAST. Preserved here.

    Mirrors py_vollib's contract of raising `BelowIntrinsicException` /
    `AboveMaximumException` (from `pyvolr.compat.py_vollib.exceptions`) when the
    price has no real implied vol. The modern `pyvolr.bs.implied_vol` returns
    NaN instead.
    """
    sigma = float(_bs.implied_vol(price, flag, S, K, t, r))
    if math.isnan(sigma):
        # black_scholes: q = 0, so the discounted underlying is S itself.
        raise_for_iv_price(price, S, K * math.exp(-r * t), flag)
    return sigma
