"""Black-76 pricing, Greeks, and implied volatility (vectorized).

The Black model (Black 1976) prices European options on futures or forwards.
Mathematically a specialization of Black-Scholes-Merton with the cost-of-carry
set to zero (the forward is already priced for delivery, so it doesn't drift
at the risk-free rate). See `docs/why.md` and `crates/core/src/black76.rs`
for the relationship to BSM.

API mirrors `pyvolr.bs`: positional flag, keyword `F` (forward), `K`, `T`, `r`,
`sigma`. No `q` parameter — Black-76 has no dividend yield concept.

Conventions (same as `pyvolr.bs`):
    - `T` is time to expiry in years.
    - `r` is the continuously compounded risk-free rate, per year.
    - `sigma` is annualized volatility.
    - `vega` is per unit of vol.
    - `theta` is per year (divide by 365 for daily theta).
    - `rho` is per unit of r.

Put-call parity for Black-76: `C - P = exp(-r*T) * (F - K)`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyvolr import _core
from pyvolr.bs import broadcast_f64, normalize_flag, scalar_or_array

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


def price(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
) -> _Result:
    """European Black-76 option price on a futures/forward `F`."""
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    out = _core.black76_price(flag_arr, *flat)
    return scalar_or_array(out, shape)


def delta(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
) -> _Result:
    """First derivative of price with respect to the forward."""
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    out = _core.black76_delta(flag_arr, *flat)
    return scalar_or_array(out, shape)


def gamma(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
) -> _Result:
    """Second derivative of price with respect to the forward. Independent of call/put."""
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    out = _core.black76_gamma(*flat)
    return scalar_or_array(out, shape)


def vega(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
) -> _Result:
    """Derivative of price with respect to volatility, per unit vol."""
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    out = _core.black76_vega(*flat)
    return scalar_or_array(out, shape)


def theta(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
) -> _Result:
    """Derivative of price with respect to time-to-expiry (per year)."""
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    out = _core.black76_theta(flag_arr, *flat)
    return scalar_or_array(out, shape)


def rho(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
) -> _Result:
    """Derivative of price with respect to the risk-free rate, per unit `r`.

    In Black-76 the forward is exogenous (doesn't depend on `r`), so the only
    `r`-dependent piece is the discount factor `exp(-r*T)`. Therefore
    `rho = -T * price` for both calls and puts — quite different from BSM rho.
    """
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    out = _core.black76_rho(flag_arr, *flat)
    return scalar_or_array(out, shape)


def implied_vol(
    price: ArrayLike,
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
) -> _Result:
    """Solve for implied volatility given a market price.

    Returns NaN where:
      - the target price is outside the no-arbitrage bounds for the forward,
      - `T <= 0`,
      - the solver cannot bracket a root within `[1e-9, 5.0]`.
    """
    flat, shape = broadcast_f64(price, F, K, T, r)
    flag_arr = normalize_flag(flag, shape).ravel()
    p_arr, f_arr, k_arr, t_arr, r_arr = flat
    out = _core.black76_iv(p_arr, flag_arr, f_arr, k_arr, t_arr, r_arr)
    return scalar_or_array(out, shape)


def greeks(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
) -> dict[str, Any]:
    """Compute the standard five Greeks at once. Returns a dict."""
    return {
        "delta": delta(flag, F, K, T, r, sigma),
        "gamma": gamma(F, K, T, r, sigma),
        "theta": theta(flag, F, K, T, r, sigma),
        "vega": vega(F, K, T, r, sigma),
        "rho": rho(flag, F, K, T, r, sigma),
    }
