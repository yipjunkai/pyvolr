"""Black-76 pricing, Greeks, and implied volatility (vectorized).

The Black model (Black 1976) prices European options on futures or forwards.
Mathematically a specialization of Black-Scholes-Merton with the cost-of-carry
set to zero (the forward is already priced for delivery, so it doesn't drift
at the risk-free rate). See `docs/why.md` and `crates/core/src/black76.rs`
for the relationship to BSM.

API mirrors `pyvolr.bs`: positional flag, keyword `F` (forward), `K`, `T`, `r`,
`sigma`. No `q` parameter — Black-76 has no dividend yield concept. All-scalar
calls take the same transparent scalar fast path as `pyvolr.bs` (dedicated
scalar kernels, bit-identical results).

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
    SCALAR_NUMERIC,
    Formatted,
    Greeks,
    GreeksResult,
    HigherGreeks,
    HigherGreeksResult,
    OnError,
    ReturnAs,
    apply_on_error,
    apply_on_error_scalar,
    broadcast_f64,
    format_result,
    normalize_flag,
    scalar_flag_or_none,
)
from pyvolr._wrappers import (
    FlagInput as _FlagInput,
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
    "HigherGreeks",
    "charm",
    "color",
    "delta",
    "gamma",
    "greeks",
    "higher_greeks",
    "implied_vol",
    "price",
    "rho",
    "speed",
    "theta",
    "ultima",
    "vanna",
    "vega",
    "veta",
    "vomma",
    "zomma",
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
    # Scalar fast path; same dispatch pattern as pyvolr.bs.price().
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(F, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
    ):
        iflag = scalar_flag_or_none(flag)
        if iflag is not None:
            return _core.black76_price_scalar(iflag, F, K, T, r, sigma)  # pyright: ignore[reportArgumentType]
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
    # Scalar fast path; same dispatch pattern as pyvolr.bs.price().
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(F, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
    ):
        iflag = scalar_flag_or_none(flag)
        if iflag is not None:
            return _core.black76_delta_scalar(iflag, F, K, T, r, sigma)  # pyright: ignore[reportArgumentType]
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
    # Scalar fast path; same dispatch pattern as pyvolr.bs.price() (no flag).
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(F, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
    ):
        return _core.black76_gamma_scalar(F, K, T, r, sigma)  # pyright: ignore[reportArgumentType]
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
    # Scalar fast path; same dispatch pattern as pyvolr.bs.price() (no flag).
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(F, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
    ):
        return _core.black76_vega_scalar(F, K, T, r, sigma)  # pyright: ignore[reportArgumentType]
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
    # Scalar fast path; same dispatch pattern as pyvolr.bs.price().
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(F, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
    ):
        iflag = scalar_flag_or_none(flag)
        if iflag is not None:
            return _core.black76_theta_scalar(iflag, F, K, T, r, sigma)  # pyright: ignore[reportArgumentType]
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
    # Scalar fast path; same dispatch pattern as pyvolr.bs.price().
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(F, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
    ):
        iflag = scalar_flag_or_none(flag)
        if iflag is not None:
            return _core.black76_rho_scalar(iflag, F, K, T, r, sigma)  # pyright: ignore[reportArgumentType]
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
    # Scalar fast path; same dispatch pattern as pyvolr.bs.price(), with the
    # scalar on_error twin (identical messages and warning depth).
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(price, SCALAR_NUMERIC)
        and isinstance(F, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
    ):
        iflag = scalar_flag_or_none(flag)
        if iflag is not None:
            out = _core.black76_iv_scalar(price, iflag, F, K, T, r)  # pyright: ignore[reportArgumentType]
            apply_on_error_scalar(out, on_error)
            return out
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
    # Scalar fast path; same dispatch pattern as pyvolr.bs.price(). Produces
    # the same Greeks dict the array path yields for all-scalar inputs.
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(F, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
    ):
        iflag = scalar_flag_or_none(flag)
        if iflag is not None:
            d, g, v, th, rh = _core.black76_greeks_scalar(iflag, F, K, T, r, sigma)  # pyright: ignore[reportArgumentType]
            return {"delta": d, "gamma": g, "theta": th, "vega": v, "rho": rh}
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    d, g, v, th, rh = _core.black76_greeks(flag_arr, *flat)
    cols = {"delta": d, "gamma": g, "theta": th, "vega": v, "rho": rh}
    return cast("GreeksResult", format_result(cols, shape, return_as))


# --- Higher-order Greeks ---


@overload
def vanna(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["numpy"] | None = ...,
) -> _Result: ...
@overload
def vanna(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dict"],
) -> dict[str, _Result]: ...
@overload
def vanna(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def vanna(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: ReturnAs = None,
) -> Formatted:
    """Vanna: rate of change of vega with forward (equivalently, of delta with volatility).

    ``return_as``: ``"numpy"`` (default), ``"dict"``, or ``"dataframe"`` (needs pandas).
    """
    # Scalar fast path; same dispatch pattern as price().
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(F, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
    ):
        return _core.black76_vanna_scalar(F, K, T, r, sigma)  # pyright: ignore[reportArgumentType]
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    out = _core.black76_vanna(*flat)
    return format_result({"vanna": out}, shape, return_as)


@overload
def vomma(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["numpy"] | None = ...,
) -> _Result: ...
@overload
def vomma(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dict"],
) -> dict[str, _Result]: ...
@overload
def vomma(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def vomma(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: ReturnAs = None,
) -> Formatted:
    """Vomma (volga): rate of change of vega with volatility.

    ``return_as``: ``"numpy"`` (default), ``"dict"``, or ``"dataframe"`` (needs pandas).
    """
    # Scalar fast path; same dispatch pattern as price().
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(F, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
    ):
        return _core.black76_vomma_scalar(F, K, T, r, sigma)  # pyright: ignore[reportArgumentType]
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    out = _core.black76_vomma(*flat)
    return format_result({"vomma": out}, shape, return_as)


@overload
def charm(
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
def charm(
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
def charm(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def charm(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: ReturnAs = None,
) -> Formatted:
    """Charm (delta decay): minus the rate of change of delta with time-to-expiry, per year.

    ``return_as``: ``"numpy"`` (default), ``"dict"``, or ``"dataframe"`` (needs pandas).
    """
    # Scalar fast path; same dispatch pattern as price().
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(F, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
    ):
        iflag = scalar_flag_or_none(flag)
        if iflag is not None:
            return _core.black76_charm_scalar(iflag, F, K, T, r, sigma)  # pyright: ignore[reportArgumentType]
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    out = _core.black76_charm(flag_arr, *flat)
    return format_result({"charm": out}, shape, return_as)


@overload
def speed(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["numpy"] | None = ...,
) -> _Result: ...
@overload
def speed(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dict"],
) -> dict[str, _Result]: ...
@overload
def speed(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def speed(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: ReturnAs = None,
) -> Formatted:
    """Speed: rate of change of gamma with forward (third order in forward).

    ``return_as``: ``"numpy"`` (default), ``"dict"``, or ``"dataframe"`` (needs pandas).
    """
    # Scalar fast path; same dispatch pattern as price().
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(F, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
    ):
        return _core.black76_speed_scalar(F, K, T, r, sigma)  # pyright: ignore[reportArgumentType]
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    out = _core.black76_speed(*flat)
    return format_result({"speed": out}, shape, return_as)


@overload
def zomma(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["numpy"] | None = ...,
) -> _Result: ...
@overload
def zomma(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dict"],
) -> dict[str, _Result]: ...
@overload
def zomma(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def zomma(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: ReturnAs = None,
) -> Formatted:
    """Zomma: rate of change of gamma with volatility.

    ``return_as``: ``"numpy"`` (default), ``"dict"``, or ``"dataframe"`` (needs pandas).
    """
    # Scalar fast path; same dispatch pattern as price().
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(F, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
    ):
        return _core.black76_zomma_scalar(F, K, T, r, sigma)  # pyright: ignore[reportArgumentType]
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    out = _core.black76_zomma(*flat)
    return format_result({"zomma": out}, shape, return_as)


@overload
def color(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["numpy"] | None = ...,
) -> _Result: ...
@overload
def color(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dict"],
) -> dict[str, _Result]: ...
@overload
def color(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def color(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: ReturnAs = None,
) -> Formatted:
    """Color (gamma decay): minus the rate of change of gamma with time-to-expiry, per year.

    ``return_as``: ``"numpy"`` (default), ``"dict"``, or ``"dataframe"`` (needs pandas).
    """
    # Scalar fast path; same dispatch pattern as price().
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(F, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
    ):
        return _core.black76_color_scalar(F, K, T, r, sigma)  # pyright: ignore[reportArgumentType]
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    out = _core.black76_color(*flat)
    return format_result({"color": out}, shape, return_as)


@overload
def veta(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["numpy"] | None = ...,
) -> _Result: ...
@overload
def veta(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dict"],
) -> dict[str, _Result]: ...
@overload
def veta(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def veta(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: ReturnAs = None,
) -> Formatted:
    """Veta (vega decay): minus the rate of change of vega with time-to-expiry, per year.

    ``return_as``: ``"numpy"`` (default), ``"dict"``, or ``"dataframe"`` (needs pandas).
    """
    # Scalar fast path; same dispatch pattern as price().
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(F, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
    ):
        return _core.black76_veta_scalar(F, K, T, r, sigma)  # pyright: ignore[reportArgumentType]
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    out = _core.black76_veta(*flat)
    return format_result({"veta": out}, shape, return_as)


@overload
def ultima(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["numpy"] | None = ...,
) -> _Result: ...
@overload
def ultima(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dict"],
) -> dict[str, _Result]: ...
@overload
def ultima(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def ultima(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: ReturnAs = None,
) -> Formatted:
    """Ultima: rate of change of vomma with volatility (third order in volatility).

    ``return_as``: ``"numpy"`` (default), ``"dict"``, or ``"dataframe"`` (needs pandas).
    """
    # Scalar fast path; same dispatch pattern as price().
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(F, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
    ):
        return _core.black76_ultima_scalar(F, K, T, r, sigma)  # pyright: ignore[reportArgumentType]
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    out = _core.black76_ultima(*flat)
    return format_result({"ultima": out}, shape, return_as)


@overload
def higher_greeks(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["numpy", "dict"] | None = ...,
) -> HigherGreeks: ...
@overload
def higher_greeks(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def higher_greeks(
    flag: _FlagInput,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    *,
    return_as: ReturnAs = None,
) -> HigherGreeksResult:
    """Compute all eight higher-order Greeks at once.

    Returns a ``HigherGreeks`` typed dict for ``return_as`` ``None``/``"numpy"``/
    ``"dict"`` (the default), or an eight-column DataFrame for ``"dataframe"``
    (columns "vanna", "vomma", "charm", "speed", "zomma", "color", "veta", "ultima"; needs pandas).

    Single FFI call into a shared Rust kernel that computes ``d1``/``d2``, the
    discount factor, and ``pdf(d1)`` once and reuses them across all eight —
    cheaper than calling each separately. Batches of ~4000 rows or more run on
    rayon's global thread pool with the GIL released; set ``RAYON_NUM_THREADS=1``
    to force serial. While the kernel runs (GIL released, and always on
    free-threaded builds), do not mutate the input arrays from other threads —
    they are read in place, zero-copy.
    """
    # Scalar fast path; same dispatch pattern as greeks().
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(F, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
    ):
        iflag = scalar_flag_or_none(flag)
        if iflag is not None:
            vals = _core.black76_higher_greeks_scalar(iflag, F, K, T, r, sigma)  # pyright: ignore[reportArgumentType]
            va, vo, ch, sp, zo, co, ve, ul = vals
            return {
                "vanna": va,
                "vomma": vo,
                "charm": ch,
                "speed": sp,
                "zomma": zo,
                "color": co,
                "veta": ve,
                "ultima": ul,
            }
    flat, shape = broadcast_f64(F, K, T, r, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    va, vo, ch, sp, zo, co, ve, ul = _core.black76_higher_greeks(flag_arr, *flat)
    cols = {
        "vanna": va,
        "vomma": vo,
        "charm": ch,
        "speed": sp,
        "zomma": zo,
        "color": co,
        "veta": ve,
        "ultima": ul,
    }
    return cast("HigherGreeksResult", format_result(cols, shape, return_as))
