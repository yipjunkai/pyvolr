"""Drop-in shim for `py_vollib.black_scholes.greeks.analytical`."""

from __future__ import annotations

from pyvolr import bs as _bs

__all__ = ["delta", "gamma", "rho", "theta", "vega"]


def delta(flag: str, S: float, K: float, t: float, r: float, sigma: float) -> float:
    return float(_bs.delta(flag, S, K, t, r, sigma))


def gamma(flag: str, S: float, K: float, t: float, r: float, sigma: float) -> float:
    # py_vollib takes flag for symmetry; gamma is identical for call/put.
    del flag
    return float(_bs.gamma(S, K, t, r, sigma))


def theta(flag: str, S: float, K: float, t: float, r: float, sigma: float) -> float:
    # py_vollib's theta convention is per-day; divide annualized by 365.
    return float(_bs.theta(flag, S, K, t, r, sigma)) / 365.0


def vega(flag: str, S: float, K: float, t: float, r: float, sigma: float) -> float:
    # py_vollib's vega convention is per 1% vol; divide per-unit-vol by 100.
    del flag
    return float(_bs.vega(S, K, t, r, sigma)) / 100.0


def rho(flag: str, S: float, K: float, t: float, r: float, sigma: float) -> float:
    # py_vollib's rho convention is per 1% rate; divide per-unit-r by 100.
    return float(_bs.rho(flag, S, K, t, r, sigma)) / 100.0
