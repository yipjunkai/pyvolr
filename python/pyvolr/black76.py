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

Output container:
    Every function takes a keyword-only ``return_as``. ``None``/``"numpy"``
    (default) returns a numpy array, or a scalar when all inputs are scalar;
    ``"dict"`` returns ``{name: value}``; ``"dataframe"`` returns a pandas
    DataFrame (pandas is an optional dependency — ``pip install pandas`` to use
    this mode; otherwise it raises ``ModuleNotFoundError``). ``greeks`` returns
    a ``Greeks`` dict for ``None``/``"numpy"``/``"dict"`` and a five-column
    DataFrame for ``"dataframe"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

from pyvolr import _core
from pyvolr._wrappers import (
    FlagInput as _FlagInput,
)
from pyvolr._wrappers import (
    Formatted,
    Greeks,
    GreeksResult,
    OnError,
    ReturnAs,
    apply_on_error,
    broadcast_f64,
    format_result,
    normalize_flag,
)
from pyvolr._wrappers import (
    Result as _Result,
)

if TYPE_CHECKING:
    from typing import Literal

    import pandas as pd
    from numpy.typing import ArrayLike

__all__ = [
    "Greeks",
    "delta",
    "gamma",
    "greeks",
    "implied_vol",
    "price",
    "rho",
    "theta",
    "vega",
]


@overload
def price(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["numpy"] | None = ...,
) -> _Result: ...
@overload
def price(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dict"],
) -> dict[str, _Result]: ...
@overload
def price(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def price(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: ReturnAs = None,
) -> Formatted:
    """European Black-76 option price on a futures/forward `F`.

    ``return_as``: ``"numpy"`` (default), ``"dict"``, or ``"dataframe"`` (needs pandas).
    """
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    out = _core.black76_price(flag_arr, *flat)
    return format_result({"price": out}, shape, return_as)


@overload
def delta(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["numpy"] | None = ...,
) -> _Result: ...
@overload
def delta(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dict"],
) -> dict[str, _Result]: ...
@overload
def delta(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def delta(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: ReturnAs = None,
) -> Formatted:
    """First derivative of price with respect to the forward.

    ``return_as``: ``"numpy"`` (default), ``"dict"``, or ``"dataframe"`` (needs pandas).
    """
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    out = _core.black76_delta(flag_arr, *flat)
    return format_result({"delta": out}, shape, return_as)


@overload
def gamma(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["numpy"] | None = ...,
) -> _Result: ...
@overload
def gamma(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dict"],
) -> dict[str, _Result]: ...
@overload
def gamma(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def gamma(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: ReturnAs = None,
) -> Formatted:
    """Second derivative of price with respect to the forward. Independent of call/put.

    ``return_as``: ``"numpy"`` (default), ``"dict"``, or ``"dataframe"`` (needs pandas).
    """
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    out = _core.black76_gamma(*flat)
    return format_result({"gamma": out}, shape, return_as)


@overload
def vega(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["numpy"] | None = ...,
) -> _Result: ...
@overload
def vega(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dict"],
) -> dict[str, _Result]: ...
@overload
def vega(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def vega(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: ReturnAs = None,
) -> Formatted:
    """Derivative of price with respect to volatility, per unit vol.

    ``return_as``: ``"numpy"`` (default), ``"dict"``, or ``"dataframe"`` (needs pandas).
    """
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    out = _core.black76_vega(*flat)
    return format_result({"vega": out}, shape, return_as)


@overload
def theta(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["numpy"] | None = ...,
) -> _Result: ...
@overload
def theta(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dict"],
) -> dict[str, _Result]: ...
@overload
def theta(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def theta(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: ReturnAs = None,
) -> Formatted:
    """Calendar theta, per year: minus the derivative of price w.r.t. time-to-expiry.

    Typically negative for long calls and puts (value decays as the clock
    advances). Divide by 365 for the per-day convention.

    ``return_as``: ``"numpy"`` (default), ``"dict"``, or ``"dataframe"`` (needs pandas).
    """
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    out = _core.black76_theta(flag_arr, *flat)
    return format_result({"theta": out}, shape, return_as)


@overload
def rho(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["numpy"] | None = ...,
) -> _Result: ...
@overload
def rho(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dict"],
) -> dict[str, _Result]: ...
@overload
def rho(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def rho(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: ReturnAs = None,
) -> Formatted:
    """Derivative of price with respect to the risk-free rate, per unit `r`.

    In Black-76 the forward is exogenous (doesn't depend on `r`), so the only
    `r`-dependent piece is the discount factor `exp(-r*T)`. Therefore
    `rho = -T * price` for both calls and puts — quite different from BSM rho.

    ``return_as``: ``"numpy"`` (default), ``"dict"``, or ``"dataframe"`` (needs pandas).
    """
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    out = _core.black76_rho(flag_arr, *flat)
    return format_result({"rho": out}, shape, return_as)


@overload
def implied_vol(
    price: ArrayLike,
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    *,
    return_as: Literal["numpy"] | None = ...,
    on_error: OnError = ...,
) -> _Result: ...
@overload
def implied_vol(
    price: ArrayLike,
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    *,
    return_as: Literal["dict"],
    on_error: OnError = ...,
) -> dict[str, _Result]: ...
@overload
def implied_vol(
    price: ArrayLike,
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    *,
    return_as: Literal["dataframe"],
    on_error: OnError = ...,
) -> pd.DataFrame: ...
def implied_vol(
    price: ArrayLike,
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    *,
    return_as: ReturnAs = None,
    on_error: OnError = "warn",
) -> Formatted:
    """Solve for implied volatility given a market price.

    ``return_as``: ``"numpy"`` (default), ``"dict"`` (``{"iv": ...}``), or
    ``"dataframe"`` (needs pandas). ``on_error`` controls unsolvable inputs (see
    "Produces NaN" below): ``"warn"`` (default) emits an ``ImpliedVolWarning``,
    ``"raise"`` raises ``ImpliedVolError``, ``"ignore"`` returns NaN silently.

    Uses the Jäckel "Let's Be Rational" algorithm (routes through
    ``iv::solve`` with ``q = r``). Converges to ~1e-13 precision in at
    most two Householder iterations across the full no-arbitrage range,
    on **well-posed inputs** (see caveat below).

    Batches of ~1000 rows or more run on rayon's global thread pool with
    the GIL released. Set ``RAYON_NUM_THREADS=1`` in the environment to
    force serial execution — useful when calling pyvolr from inside a
    caller-managed thread pool that already saturates the cores. While the
    kernel runs (GIL released, and always on free-threaded builds), do not
    mutate the input arrays from other threads — they are read in place,
    zero-copy.

    .. note::

       **Ill-conditioned inverse cases.** When the option price equals its
       intrinsic value to f64 precision — typically deep ITM with very
       short expiry — the price carries no signal about volatility. The
       solver returns the sigma that *matches the price* (correct), but this
       sigma may differ from the sigma that originally produced the price. This
       is a property of the inverse problem, not the algorithm. Affects
       strikes where ``|F/K|`` is far from 1 *and* ``T`` is small. See
       the ``pyvolr.bs.implied_vol`` docstring for more detail.

    Produces NaN (subject to ``on_error``) where:
      - the target price is outside the no-arbitrage bounds for the forward,
      - `T <= 0`, `F <= 0`, or `K <= 0`,
      - any input is non-finite.
    """
    flat, shape = broadcast_f64(price, F, K, T, r)
    flag_arr = normalize_flag(flag, shape).ravel()
    p_arr, f_arr, k_arr, t_arr, r_arr = flat
    out = _core.black76_iv(p_arr, flag_arr, f_arr, k_arr, t_arr, r_arr)
    apply_on_error(out, on_error)
    return format_result({"iv": out}, shape, return_as)


@overload
def greeks(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["numpy", "dict"] | None = ...,
) -> Greeks: ...
@overload
def greeks(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def greeks(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: ReturnAs = None,
) -> GreeksResult:
    """Compute the standard five Greeks at once.

    Returns a ``Greeks`` typed dict for ``return_as`` ``None``/``"numpy"``/
    ``"dict"`` (the default), or a five-column DataFrame for ``"dataframe"``
    (columns ``delta, gamma, theta, vega, rho``; needs pandas).

    Single FFI call into a shared Rust kernel — see `pyvolr.bs.greeks` for
    the rationale. Batches of ~4000 rows or more parallelise on rayon's
    global thread pool (GIL released); set ``RAYON_NUM_THREADS=1`` to
    force serial. While the kernel runs (GIL released, and always on
    free-threaded builds), do not mutate the input arrays from other
    threads — they are read in place, zero-copy.
    """
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    d, g, v, th, rh = _core.black76_greeks(flag_arr, *flat)
    cols = {"delta": d, "gamma": g, "theta": th, "vega": v, "rho": rh}
    return cast("GreeksResult", format_result(cols, shape, return_as))
