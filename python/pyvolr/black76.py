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

    Uses the Jäckel "Let's Be Rational" algorithm (routes through
    ``iv::solve`` with ``q = r``). Converges to ~1e-13 precision in at
    most two Householder iterations across the full no-arbitrage range.

    Batches of ~1000 rows or more run on rayon's global thread pool with
    the GIL released. Set ``RAYON_NUM_THREADS=1`` in the environment to
    force serial execution — useful when calling pyvolr from inside a
    caller-managed thread pool that already saturates the cores.

    Returns NaN where:
      - the target price is outside the no-arbitrage bounds for the forward,
      - `T <= 0`, `F <= 0`, or `K <= 0`,
      - any input is non-finite.
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
    """Compute the standard five Greeks at once. Returns a dict.

    Single FFI call into a shared Rust kernel — see `pyvolr.bs.greeks` for
    the rationale. Batches of ~4000 rows or more parallelise on rayon's
    global thread pool (GIL released); set ``RAYON_NUM_THREADS=1`` to
    force serial.
    """
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    d, g, v, th, rh = _core.black76_greeks(flag_arr, *flat)
    return {
        "delta": scalar_or_array(d, shape),
        "gamma": scalar_or_array(g, shape),
        "theta": scalar_or_array(th, shape),
        "vega": scalar_or_array(v, shape),
        "rho": scalar_or_array(rh, shape),
    }
