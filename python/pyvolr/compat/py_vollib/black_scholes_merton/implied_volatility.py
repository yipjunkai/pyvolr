"""Drop-in shim for `py_vollib.black_scholes_merton.implied_volatility`."""

from __future__ import annotations

from pyvolr import bs as _bs

__all__ = ["implied_volatility"]


def implied_volatility(
    price: float, S: float, K: float, t: float, r: float, q: float, flag: str
) -> float:
    """Solve for implied volatility from a market price (BSM with dividend yield)."""
    return float(_bs.implied_vol(price, flag, S, K, t, r, q))
