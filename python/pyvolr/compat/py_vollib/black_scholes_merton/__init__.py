"""Drop-in shim for `py_vollib.black_scholes_merton` (BSM with dividend yield)."""

from __future__ import annotations

from pyvolr import bs as _bs

__all__ = ["black_scholes_merton"]


def black_scholes_merton(
    flag: str, S: float, K: float, t: float, r: float, sigma: float, q: float
) -> float:
    """European BSM call/put price with continuous dividend yield `q`."""
    return float(_bs.price(flag, S, K, t, r, sigma, q))
