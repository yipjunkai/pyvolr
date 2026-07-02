"""Black-Scholes-Merton pricing, Greeks, and implied volatility (vectorized).

All functions accept scalars or numpy-compatible arrays. Inputs are broadcast
to a common shape, then evaluated in the Rust core. Results are returned
as a numpy array, or as a Python scalar if all inputs are scalar.

All-scalar calls (a string flag plus float/int/numpy numeric scalars, default
``return_as``) skip the broadcast machinery entirely and dispatch to dedicated
scalar kernels in the Rust core — same results bit-for-bit at a fraction of
the per-call latency. This is transparent; no API opt-in exists or is needed.

The `flag` argument indicates option type:
    - `'c'`, `'C'` -> call
    - `'p'`, `'P'` -> put
    - array of strings, or array of ints strictly 1 (call) or -1 (put)

Conventions:
    - `T` is time to expiry in years (e.g. 0.5 = six months).
    - `r` is the continuously compounded risk-free rate, per year.
    - `sigma` is the annualized volatility (e.g. 0.20 = 20%).
    - `q` is the continuous dividend yield (default 0).
    - `vega` is per unit of vol (not per 1% vol). Multiply by 0.01 for the
      "per 1% vol" convention.
    - `theta` is per year. Divide by 365 (or 252) for daily theta.
    - `rho` is per unit of `r` (not per 1% r).

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
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["numpy"] | None = ...,
) -> _Result: ...
@overload
def price(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["dict"],
) -> dict[str, _Result]: ...
@overload
def price(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def price(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    *,
    return_as: ReturnAs = None,
) -> Formatted:
    """European Black-Scholes-Merton option price.

    ``return_as``: ``"numpy"`` (default), ``"dict"``, or ``"dataframe"`` (needs pandas).
    """
    # Scalar fast path: all-scalar inputs skip the numpy broadcast machinery
    # and hit a dedicated scalar FFI entry point (same kernel, bit-identical).
    # Numeric eligibility is checked before the flag so that doubly-invalid
    # inputs raise the same error the array path would.
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(S, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
        and isinstance(q, SCALAR_NUMERIC)
    ):
        iflag = scalar_flag_or_none(flag)
        if iflag is not None:
            # The isinstance chain above proves these are numeric scalars, which
            # multi-argument narrowing can't express to the type checker.
            return _core.bsm_price_scalar(iflag, S, K, T, r, q, sigma)  # pyright: ignore[reportArgumentType]
    flat, shape = broadcast_f64(S, K, T, r, q, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    out = _core.bsm_price(flag_arr, *flat)
    return format_result({"price": out}, shape, return_as)


@overload
def delta(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["numpy"] | None = ...,
) -> _Result: ...
@overload
def delta(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["dict"],
) -> dict[str, _Result]: ...
@overload
def delta(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def delta(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    *,
    return_as: ReturnAs = None,
) -> Formatted:
    """First derivative of price with respect to spot.

    ``return_as``: ``"numpy"`` (default), ``"dict"``, or ``"dataframe"`` (needs pandas).
    """
    # Scalar fast path; same dispatch pattern as price().
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(S, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
        and isinstance(q, SCALAR_NUMERIC)
    ):
        iflag = scalar_flag_or_none(flag)
        if iflag is not None:
            return _core.bsm_delta_scalar(iflag, S, K, T, r, q, sigma)  # pyright: ignore[reportArgumentType]
    flat, shape = broadcast_f64(S, K, T, r, q, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    out = _core.bsm_delta(flag_arr, *flat)
    return format_result({"delta": out}, shape, return_as)


@overload
def gamma(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["numpy"] | None = ...,
) -> _Result: ...
@overload
def gamma(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["dict"],
) -> dict[str, _Result]: ...
@overload
def gamma(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def gamma(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    *,
    return_as: ReturnAs = None,
) -> Formatted:
    """Second derivative of price with respect to spot. Independent of call/put.

    ``return_as``: ``"numpy"`` (default), ``"dict"``, or ``"dataframe"`` (needs pandas).
    """
    # Scalar fast path; same dispatch pattern as price() (no flag here).
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(S, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
        and isinstance(q, SCALAR_NUMERIC)
    ):
        return _core.bsm_gamma_scalar(S, K, T, r, q, sigma)  # pyright: ignore[reportArgumentType]
    flat, shape = broadcast_f64(S, K, T, r, q, sigma)
    out = _core.bsm_gamma(*flat)
    return format_result({"gamma": out}, shape, return_as)


@overload
def vega(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["numpy"] | None = ...,
) -> _Result: ...
@overload
def vega(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["dict"],
) -> dict[str, _Result]: ...
@overload
def vega(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def vega(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    *,
    return_as: ReturnAs = None,
) -> Formatted:
    """Derivative of price with respect to volatility (per unit vol).

    ``return_as``: ``"numpy"`` (default), ``"dict"``, or ``"dataframe"`` (needs pandas).
    """
    # Scalar fast path; same dispatch pattern as price() (no flag here).
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(S, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
        and isinstance(q, SCALAR_NUMERIC)
    ):
        return _core.bsm_vega_scalar(S, K, T, r, q, sigma)  # pyright: ignore[reportArgumentType]
    flat, shape = broadcast_f64(S, K, T, r, q, sigma)
    out = _core.bsm_vega(*flat)
    return format_result({"vega": out}, shape, return_as)


@overload
def theta(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["numpy"] | None = ...,
) -> _Result: ...
@overload
def theta(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["dict"],
) -> dict[str, _Result]: ...
@overload
def theta(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def theta(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    *,
    return_as: ReturnAs = None,
) -> Formatted:
    """Calendar theta, per year: minus the derivative of price w.r.t. time-to-expiry.

    Typically negative for long calls and puts (value decays as the clock
    advances). Divide by 365 for the per-day convention.

    ``return_as``: ``"numpy"`` (default), ``"dict"``, or ``"dataframe"`` (needs pandas).
    """
    # Scalar fast path; same dispatch pattern as price().
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(S, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
        and isinstance(q, SCALAR_NUMERIC)
    ):
        iflag = scalar_flag_or_none(flag)
        if iflag is not None:
            return _core.bsm_theta_scalar(iflag, S, K, T, r, q, sigma)  # pyright: ignore[reportArgumentType]
    flat, shape = broadcast_f64(S, K, T, r, q, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    out = _core.bsm_theta(flag_arr, *flat)
    return format_result({"theta": out}, shape, return_as)


@overload
def rho(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["numpy"] | None = ...,
) -> _Result: ...
@overload
def rho(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["dict"],
) -> dict[str, _Result]: ...
@overload
def rho(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def rho(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    *,
    return_as: ReturnAs = None,
) -> Formatted:
    """Derivative of price with respect to the risk-free rate (per unit r).

    ``return_as``: ``"numpy"`` (default), ``"dict"``, or ``"dataframe"`` (needs pandas).
    """
    # Scalar fast path; same dispatch pattern as price().
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(S, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
        and isinstance(q, SCALAR_NUMERIC)
    ):
        iflag = scalar_flag_or_none(flag)
        if iflag is not None:
            return _core.bsm_rho_scalar(iflag, S, K, T, r, q, sigma)  # pyright: ignore[reportArgumentType]
    flat, shape = broadcast_f64(S, K, T, r, q, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    out = _core.bsm_rho(flag_arr, *flat)
    return format_result({"rho": out}, shape, return_as)


@overload
def implied_vol(
    price: ArrayLike,
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["numpy"] | None = ...,
    on_error: OnError = ...,
) -> _Result: ...
@overload
def implied_vol(
    price: ArrayLike,
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["dict"],
    on_error: OnError = ...,
) -> dict[str, _Result]: ...
@overload
def implied_vol(
    price: ArrayLike,
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["dataframe"],
    on_error: OnError = ...,
) -> pd.DataFrame: ...
def implied_vol(
    price: ArrayLike,
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
    *,
    return_as: ReturnAs = None,
    on_error: OnError = "warn",
) -> Formatted:
    """Solve for implied volatility given a market price.

    ``return_as``: ``"numpy"`` (default), ``"dict"`` (``{"iv": ...}``), or
    ``"dataframe"`` (needs pandas). ``on_error`` controls unsolvable inputs (see
    "Produces NaN" below): ``"warn"`` (default) emits an ``ImpliedVolWarning``,
    ``"raise"`` raises ``ImpliedVolError``, ``"ignore"`` returns NaN silently.

    Uses the Jäckel "Let's Be Rational" algorithm: converges to ~1e-13
    precision in at most two Householder iterations across the full
    no-arbitrage range, on **well-posed inputs** (see caveat below).

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
       solver returns the sigma that *matches the price* to f64 (correct), but
       this sigma may differ substantially from the sigma that produced the price.
       This is a property of the inverse problem, not the algorithm:
       distinguishing sigma=5% from sigma=50% on a 1000-strike call expiring in
       3 days is below the representable precision of the price itself.

       In practice this affects strikes where ``|S/K|`` is far from 1
       *and* ``T`` is small. If your workflow surfaces this, round-trip
       the result through ``bs.price`` to verify; a mismatch in sigma with a
       matching price is the ill-conditioning signature.

    Produces NaN (subject to ``on_error``) where:
      - the target price is outside the no-arbitrage bounds,
      - `T <= 0`, `S <= 0`, or `K <= 0`,
      - any input is non-finite.
    """
    # Scalar fast path; same dispatch pattern as price(), with the scalar
    # on_error twin (identical messages and warning depth).
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(price, SCALAR_NUMERIC)
        and isinstance(S, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(q, SCALAR_NUMERIC)
    ):
        iflag = scalar_flag_or_none(flag)
        if iflag is not None:
            out = _core.bsm_iv_scalar(price, iflag, S, K, T, r, q)  # pyright: ignore[reportArgumentType]
            apply_on_error_scalar(out, on_error)
            return out
    flat, shape = broadcast_f64(price, S, K, T, r, q)
    flag_arr = normalize_flag(flag, shape).ravel()
    p_arr, s_arr, k_arr, t_arr, r_arr, q_arr = flat
    out = _core.bsm_iv(p_arr, flag_arr, s_arr, k_arr, t_arr, r_arr, q_arr)
    apply_on_error(out, on_error)
    return format_result({"iv": out}, shape, return_as)


@overload
def greeks(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["numpy", "dict"] | None = ...,
) -> Greeks: ...
@overload
def greeks(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = ...,
    *,
    return_as: Literal["dataframe"],
) -> pd.DataFrame: ...
def greeks(
    flag: _FlagInput,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    *,
    return_as: ReturnAs = None,
) -> GreeksResult:
    """Compute the standard five Greeks at once.

    Returns a ``Greeks`` typed dict for ``return_as`` ``None``/``"numpy"``/
    ``"dict"`` (the default), or a five-column DataFrame for ``"dataframe"``
    (columns ``delta, gamma, theta, vega, rho``; needs pandas).

    Single FFI call into a shared Rust kernel that computes `d1`/`d2`, the
    discount factors, `cdf(d1)`/`cdf(d2)`, and `pdf(d1)` once and reuses them
    across all five Greeks — ~3x faster than calling each Greek separately.

    Batches of ~4000 rows or more run on rayon's global thread pool with the
    GIL released. Set ``RAYON_NUM_THREADS=1`` in the environment to force
    serial execution. While the kernel runs (GIL released, and always on
    free-threaded builds), do not mutate the input arrays from other
    threads — they are read in place, zero-copy.
    """
    # Scalar fast path; same dispatch pattern as price(). Produces the same
    # Greeks dict the array path yields for all-scalar inputs.
    if (
        (return_as is None or return_as == "numpy")
        and isinstance(S, SCALAR_NUMERIC)
        and isinstance(K, SCALAR_NUMERIC)
        and isinstance(T, SCALAR_NUMERIC)
        and isinstance(r, SCALAR_NUMERIC)
        and isinstance(sigma, SCALAR_NUMERIC)
        and isinstance(q, SCALAR_NUMERIC)
    ):
        iflag = scalar_flag_or_none(flag)
        if iflag is not None:
            d, g, v, th, rh = _core.bsm_greeks_scalar(iflag, S, K, T, r, q, sigma)  # pyright: ignore[reportArgumentType]
            return {"delta": d, "gamma": g, "theta": th, "vega": v, "rho": rh}
    flat, shape = broadcast_f64(S, K, T, r, q, sigma)
    flag_arr = normalize_flag(flag, shape).ravel()
    d, g, v, th, rh = _core.bsm_greeks(flag_arr, *flat)
    cols = {"delta": d, "gamma": g, "theta": th, "vega": v, "rho": rh}
    return cast("GreeksResult", format_result(cols, shape, return_as))
