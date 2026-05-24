"""Drop-in shim for `py_vollib.black.greeks.analytical`.

Unit conventions match py_vollib's exactly:
    - vega:  per 1% vol change (divide modern per-unit by 100)
    - theta: per day            (divide modern per-year by 365)
    - rho:   per 1% rate change (divide modern per-unit by 100)

py_vollib.black.gamma takes a flag argument but ignores it (gamma is
identical for call and put); we mirror that signature exactly.
"""

from __future__ import annotations

from pyvolr import black76 as _b76

__all__ = ["delta", "gamma", "rho", "theta", "vega"]


def delta(flag: str, F: float, K: float, t: float, r: float, sigma: float) -> float:
    return float(_b76.delta(flag, F, K, t, r, sigma))


def gamma(flag: str, F: float, K: float, t: float, r: float, sigma: float) -> float:
    del flag  # call/put gamma are identical; py_vollib accepts the arg anyway
    return float(_b76.gamma(F, K, t, r, sigma))


def theta(flag: str, F: float, K: float, t: float, r: float, sigma: float) -> float:
    return float(_b76.theta(flag, F, K, t, r, sigma)) / 365.0


def vega(flag: str, F: float, K: float, t: float, r: float, sigma: float) -> float:
    del flag  # call/put vega are identical
    return float(_b76.vega(F, K, t, r, sigma)) / 100.0


def rho(flag: str, F: float, K: float, t: float, r: float, sigma: float) -> float:
    return float(_b76.rho(flag, F, K, t, r, sigma)) / 100.0
