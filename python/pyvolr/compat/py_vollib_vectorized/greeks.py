"""Vectorized greeks — drop-in for ``py_vollib_vectorized.greeks``.

Model-dispatched (``black_scholes`` / ``black_scholes_merton`` / ``black``) and in
py_vollib's unit conventions: theta per day (``/365``), vega and rho per 1%
(``/100``); delta and gamma per unit. (py_vollib_vectorized computes these
numerically; pyvolr uses analytical formulas, so values agree only to
finite-difference tolerance.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyvolr import black76 as _b76
from pyvolr import bs as _bs

from ._format import as_1d, format_single

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike

    from ._format import SingleResult

__all__ = ["delta", "gamma", "rho", "theta", "vega"]

MODEL_ERR = "model must be 'black_scholes', 'black_scholes_merton', or 'black'"

THETA_PER_DAY = 1.0 / 365.0
PER_1PCT = 0.01


def _finish(
    raw: ArrayLike, name: str, factor: float, return_as: str, dtype: DTypeLike
) -> SingleResult:
    """Apply the py_vollib unit factor and shape per ``return_as``."""
    return format_single(as_1d(raw, dtype) * factor, name, return_as)


def delta(
    flag: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    t: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike | None = None,
    *,
    model: str = "black_scholes",
    return_as: str = "dataframe",
    dtype: DTypeLike = np.float64,
) -> SingleResult:
    if model == "black_scholes":
        raw = _bs.delta(flag, S, K, t, r, sigma, 0.0)
    elif model == "black_scholes_merton":
        raw = _bs.delta(flag, S, K, t, r, sigma, 0.0 if q is None else q)
    elif model == "black":
        raw = _b76.delta(flag, S, K, t, r, sigma)
    else:
        raise ValueError(MODEL_ERR)
    return _finish(raw, "delta", 1.0, return_as, dtype)


def gamma(
    flag: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    t: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike | None = None,
    *,
    model: str = "black_scholes",
    return_as: str = "dataframe",
    dtype: DTypeLike = np.float64,
) -> SingleResult:
    del flag  # gamma is identical for calls and puts; accepted for parity
    if model == "black_scholes":
        raw = _bs.gamma(S, K, t, r, sigma, 0.0)
    elif model == "black_scholes_merton":
        raw = _bs.gamma(S, K, t, r, sigma, 0.0 if q is None else q)
    elif model == "black":
        raw = _b76.gamma(S, K, t, r, sigma)
    else:
        raise ValueError(MODEL_ERR)
    return _finish(raw, "gamma", 1.0, return_as, dtype)


def theta(
    flag: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    t: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike | None = None,
    *,
    model: str = "black_scholes",
    return_as: str = "dataframe",
    dtype: DTypeLike = np.float64,
) -> SingleResult:
    if model == "black_scholes":
        raw = _bs.theta(flag, S, K, t, r, sigma, 0.0)
    elif model == "black_scholes_merton":
        raw = _bs.theta(flag, S, K, t, r, sigma, 0.0 if q is None else q)
    elif model == "black":
        raw = _b76.theta(flag, S, K, t, r, sigma)
    else:
        raise ValueError(MODEL_ERR)
    return _finish(raw, "theta", THETA_PER_DAY, return_as, dtype)


def vega(
    flag: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    t: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike | None = None,
    *,
    model: str = "black_scholes",
    return_as: str = "dataframe",
    dtype: DTypeLike = np.float64,
) -> SingleResult:
    del flag  # vega is identical for calls and puts; accepted for parity
    if model == "black_scholes":
        raw = _bs.vega(S, K, t, r, sigma, 0.0)
    elif model == "black_scholes_merton":
        raw = _bs.vega(S, K, t, r, sigma, 0.0 if q is None else q)
    elif model == "black":
        raw = _b76.vega(S, K, t, r, sigma)
    else:
        raise ValueError(MODEL_ERR)
    return _finish(raw, "vega", PER_1PCT, return_as, dtype)


def rho(
    flag: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    t: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike | None = None,
    *,
    model: str = "black_scholes",
    return_as: str = "dataframe",
    dtype: DTypeLike = np.float64,
) -> SingleResult:
    if model == "black_scholes":
        raw = _bs.rho(flag, S, K, t, r, sigma, 0.0)
    elif model == "black_scholes_merton":
        raw = _bs.rho(flag, S, K, t, r, sigma, 0.0 if q is None else q)
    elif model == "black":
        raw = _b76.rho(flag, S, K, t, r, sigma)
    else:
        raise ValueError(MODEL_ERR)
    return _finish(raw, "rho", PER_1PCT, return_as, dtype)
