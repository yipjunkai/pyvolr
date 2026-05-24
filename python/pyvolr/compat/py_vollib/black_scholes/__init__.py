"""Drop-in shim for `py_vollib.black_scholes`."""

from __future__ import annotations

from pyvolr import bs as _bs

__all__ = ["black_scholes"]


def black_scholes(flag: str, S: float, K: float, t: float, r: float, sigma: float) -> float:
    """European Black-Scholes call/put price (no dividend yield).

    Matches the original `py_vollib.black_scholes.black_scholes` signature exactly.

    Args:
        flag: 'c' for call, 'p' for put.
        S: spot price.
        K: strike.
        t: time to expiry in years.
        r: risk-free rate (continuous compounding).
        sigma: annualized volatility.
    """
    return float(_bs.price(flag, S, K, t, r, sigma))
