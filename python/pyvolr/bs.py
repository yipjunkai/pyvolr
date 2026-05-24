"""Black-Scholes-Merton pricing, Greeks, and implied volatility (vectorized).

All functions accept scalars or numpy-compatible arrays. Inputs are broadcast
to a common shape, then evaluated in the Rust core. Results are returned
as a numpy array, or as a Python scalar if all inputs are scalar.

The `flag` argument indicates option type:
    - `'c'`, `'C'` -> call
    - `'p'`, `'P'` -> put
    - array of strings or array of ±1 ints also accepted

Conventions:
    - `T` is time to expiry in years (e.g. 0.5 = six months).
    - `r` is the continuously compounded risk-free rate, per year.
    - `sigma` is the annualized volatility (e.g. 0.20 = 20%).
    - `q` is the continuous dividend yield (default 0).
    - `vega` is per unit of vol (not per 1% vol). Multiply by 0.01 for the
      "per 1% vol" convention.
    - `theta` is per year. Divide by 365 (or 252) for daily theta.
    - `rho` is per unit of `r` (not per 1% r).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyvolr import _core

__all__ = [
    "delta",
    "gamma",
    "greeks",
    "implied_vol",
    "price",
    "rho",
    "theta",
    "vega",
]

_FlagInput = ArrayLike | str
_Result = float | NDArray[np.float64]


def _normalize_flag(flag: _FlagInput, shape: tuple[int, ...]) -> NDArray[np.int8]:
    """Convert a flag input into an int8 array of the given broadcast shape.

    Accepts:
        - a string: 'c'/'C' (call) or 'p'/'P' (put)
        - an ndarray of strings
        - an ndarray of ints (±1, where >=0 is call, <0 is put)
    """
    if isinstance(flag, str):
        s = flag.lower()
        if s not in ("c", "p"):
            raise ValueError(f"flag must be 'c' or 'p', got {flag!r}")
        return np.full(shape, 1 if s == "c" else -1, dtype=np.int8)

    arr = np.asarray(flag)
    if arr.dtype.kind in ("U", "S", "O"):
        lower_flat = np.array([str(x).lower() for x in arr.ravel()], dtype="U1")
        lower = np.reshape(lower_flat, arr.shape)
        if not np.isin(lower, ("c", "p")).all():
            raise ValueError("flag array must contain only 'c' or 'p' (case-insensitive)")
        encoded = np.where(lower == "c", 1, -1).astype(np.int8)
        return np.ascontiguousarray(np.broadcast_to(encoded, shape))
    return np.ascontiguousarray(np.broadcast_to(arr.astype(np.int8), shape))


def _broadcast_f64(*arrs: ArrayLike) -> tuple[list[NDArray[np.float64]], tuple[int, ...]]:
    """Broadcast inputs to a common shape; return flat contiguous f64 arrays."""
    cast = [np.asarray(a, dtype=np.float64) for a in arrs]
    bcast = np.broadcast_arrays(*cast)
    shape = bcast[0].shape
    flat = [np.ascontiguousarray(a.ravel()) for a in bcast]
    return flat, shape


def _scalar_or_array(arr: NDArray[np.float64], shape: tuple[int, ...]) -> _Result:
    """If the broadcast shape is () return a Python scalar; otherwise reshape."""
    if shape == ():
        return float(arr[0])
    return np.reshape(arr, shape)


def price(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> _Result:
    """European Black-Scholes-Merton option price."""
    flat, shape = _broadcast_f64(S, K, T, r, q, sigma)
    flag_arr = _normalize_flag(flag, shape).ravel()
    out = _core.bsm_price(flag_arr, *flat)
    return _scalar_or_array(out, shape)


def delta(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> _Result:
    """First derivative of price with respect to spot."""
    flat, shape = _broadcast_f64(S, K, T, r, q, sigma)
    flag_arr = _normalize_flag(flag, shape).ravel()
    out = _core.bsm_delta(flag_arr, *flat)
    return _scalar_or_array(out, shape)


def gamma(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> _Result:
    """Second derivative of price with respect to spot. Independent of call/put."""
    flat, shape = _broadcast_f64(S, K, T, r, q, sigma)
    out = _core.bsm_gamma(*flat)
    return _scalar_or_array(out, shape)


def vega(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> _Result:
    """Derivative of price with respect to volatility (per unit vol)."""
    flat, shape = _broadcast_f64(S, K, T, r, q, sigma)
    out = _core.bsm_vega(*flat)
    return _scalar_or_array(out, shape)


def theta(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> _Result:
    """Derivative of price with respect to time-to-expiry (per year, annualized)."""
    flat, shape = _broadcast_f64(S, K, T, r, q, sigma)
    flag_arr = _normalize_flag(flag, shape).ravel()
    out = _core.bsm_theta(flag_arr, *flat)
    return _scalar_or_array(out, shape)


def rho(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> _Result:
    """Derivative of price with respect to the risk-free rate (per unit r)."""
    flat, shape = _broadcast_f64(S, K, T, r, q, sigma)
    flag_arr = _normalize_flag(flag, shape).ravel()
    out = _core.bsm_rho(flag_arr, *flat)
    return _scalar_or_array(out, shape)


def implied_vol(
    price: ArrayLike,
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
) -> _Result:
    """Solve for implied volatility given a market price.

    Returns NaN where:
      - the target price is outside the no-arbitrage bounds,
      - `T <= 0`,
      - the solver cannot bracket a root within `[1e-9, 5.0]`.
    """
    flat, shape = _broadcast_f64(price, S, K, T, r, q)
    flag_arr = _normalize_flag(flag, shape).ravel()
    p_arr, s_arr, k_arr, t_arr, r_arr, q_arr = flat
    out = _core.bsm_iv(p_arr, flag_arr, s_arr, k_arr, t_arr, r_arr, q_arr)
    return _scalar_or_array(out, shape)


def greeks(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> dict[str, Any]:
    """Compute the standard five Greeks at once. Returns a dict.

    Slightly more efficient than calling each Greek separately because the
    broadcast and flag-normalization happen once.
    """
    return {
        "delta": delta(flag, S, K, T, r, sigma, q),
        "gamma": gamma(S, K, T, r, sigma, q),
        "theta": theta(flag, S, K, T, r, sigma, q),
        "vega": vega(S, K, T, r, sigma, q),
        "rho": rho(flag, S, K, T, r, sigma, q),
    }
