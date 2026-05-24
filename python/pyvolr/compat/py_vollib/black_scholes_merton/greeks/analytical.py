"""Drop-in shim for `py_vollib.black_scholes_merton.greeks.analytical`."""

from __future__ import annotations

from pyvolr import bs as _bs

__all__ = ["delta", "gamma", "rho", "theta", "vega"]


def delta(flag: str, S: float, K: float, t: float, r: float, sigma: float, q: float) -> float:
    return float(_bs.delta(flag, S, K, t, r, sigma, q))


def gamma(flag: str, S: float, K: float, t: float, r: float, sigma: float, q: float) -> float:
    del flag
    return float(_bs.gamma(S, K, t, r, sigma, q))


def theta(flag: str, S: float, K: float, t: float, r: float, sigma: float, q: float) -> float:
    return float(_bs.theta(flag, S, K, t, r, sigma, q)) / 365.0


def vega(flag: str, S: float, K: float, t: float, r: float, sigma: float, q: float) -> float:
    del flag
    return float(_bs.vega(S, K, t, r, sigma, q)) / 100.0


def rho(flag: str, S: float, K: float, t: float, r: float, sigma: float, q: float) -> float:
    return float(_bs.rho(flag, S, K, t, r, sigma, q)) / 100.0
