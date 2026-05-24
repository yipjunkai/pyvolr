"""Drop-in shim for `py_vollib.black_scholes.implied_volatility`."""

from __future__ import annotations

from pyvolr import bs as _bs

__all__ = ["implied_volatility"]


def implied_volatility(price: float, S: float, K: float, t: float, r: float, flag: str) -> float:
    """Solve for implied volatility from a market price.

    Note: py_vollib's parameter ORDER places `flag` LAST. Preserved here.
    """
    return float(_bs.implied_vol(price, flag, S, K, t, r))
