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

from typing import TYPE_CHECKING, Any

from pyvolr import _core
from pyvolr._wrappers import (
    FlagInput as _FlagInput,
)
from pyvolr._wrappers import (
    Result as _Result,
)
from pyvolr._wrappers import (
    broadcast_f64,
    normalize_flag,
    scalar_or_array,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

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
    flat, shape = broadcast_f64(S, K, T, r, q, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    out = _core.bsm_price(flag_arr, *flat)
    return scalar_or_array(out, shape)


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
    flat, shape = broadcast_f64(S, K, T, r, q, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    out = _core.bsm_delta(flag_arr, *flat)
    return scalar_or_array(out, shape)


def gamma(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> _Result:
    """Second derivative of price with respect to spot. Independent of call/put."""
    flat, shape = broadcast_f64(S, K, T, r, q, sigma)
    out = _core.bsm_gamma(*flat)
    return scalar_or_array(out, shape)


def vega(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> _Result:
    """Derivative of price with respect to volatility (per unit vol)."""
    flat, shape = broadcast_f64(S, K, T, r, q, sigma)
    out = _core.bsm_vega(*flat)
    return scalar_or_array(out, shape)


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
    flat, shape = broadcast_f64(S, K, T, r, q, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    out = _core.bsm_theta(flag_arr, *flat)
    return scalar_or_array(out, shape)


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
    flat, shape = broadcast_f64(S, K, T, r, q, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    out = _core.bsm_rho(flag_arr, *flat)
    return scalar_or_array(out, shape)


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

    Uses the Jäckel "Let's Be Rational" algorithm: converges to ~1e-13
    precision in at most two Householder iterations across the full
    no-arbitrage range.

    Returns NaN where:
      - the target price is outside the no-arbitrage bounds,
      - `T <= 0`, `S <= 0`, or `K <= 0`,
      - any input is non-finite.
    """
    flat, shape = broadcast_f64(price, S, K, T, r, q)
    flag_arr = normalize_flag(flag, shape).ravel()
    p_arr, s_arr, k_arr, t_arr, r_arr, q_arr = flat
    out = _core.bsm_iv(p_arr, flag_arr, s_arr, k_arr, t_arr, r_arr, q_arr)
    return scalar_or_array(out, shape)


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

    Single FFI call into a shared Rust kernel that computes `d1`/`d2`, the
    discount factors, `cdf(d1)`/`cdf(d2)`, and `pdf(d1)` once and reuses them
    across all five Greeks — 3-5x faster than calling each Greek separately.
    """
    flat, shape = broadcast_f64(S, K, T, r, q, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    d, g, v, th, rh = _core.bsm_greeks(flag_arr, *flat)
    return {
        "delta": scalar_or_array(d, shape),
        "gamma": scalar_or_array(g, shape),
        "theta": scalar_or_array(th, shape),
        "vega": scalar_or_array(v, shape),
        "rho": scalar_or_array(rh, shape),
    }
