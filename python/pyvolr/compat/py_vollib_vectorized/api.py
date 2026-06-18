"""Aggregate utilities — drop-in for ``py_vollib_vectorized.api``.

``get_all_greeks`` returns all five greeks at once (reusing pyvolr's bundled
single-pass kernel); ``price_dataframe`` prices a whole DataFrame of contracts by
column name. Both follow py_vollib_vectorized's conventions (greek column order
``delta, gamma, theta, rho, vega``; py_vollib unit factors).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyvolr import black76 as _b76
from pyvolr import bs as _bs

from ._format import as_1d, format_multi
from .greeks import MODEL_ERR, PER_1PCT, THETA_PER_DAY
from .implied_volatility import vectorized_implied_volatility
from .models import (
    vectorized_black,
    vectorized_black_scholes,
    vectorized_black_scholes_merton,
)

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from ._format import MultiResult

__all__ = ["get_all_greeks", "price_dataframe"]


def get_all_greeks(
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
) -> MultiResult:
    """All five greeks, in py_vollib's units, ordered delta, gamma, theta, rho, vega."""
    return format_multi(_greeks_dict(flag, S, K, t, r, sigma, q, model, dtype), return_as)


def _greeks_dict(
    flag: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    t: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike | None,
    model: str,
    dtype: DTypeLike,
) -> dict[str, NDArray[np.float64]]:
    """The five greeks as a typed dict (pvv order, py_vollib units)."""
    if model == "black_scholes":
        g = _bs.greeks(flag, S, K, t, r, sigma, 0.0)
    elif model == "black_scholes_merton":
        if q is None:
            raise ValueError("black_scholes_merton requires q (continuous dividend yield)")
        g = _bs.greeks(flag, S, K, t, r, sigma, q)
    elif model == "black":
        g = _b76.greeks(flag, S, K, t, r, sigma)
    else:
        raise ValueError(MODEL_ERR)
    return {
        "delta": as_1d(g["delta"], dtype),
        "gamma": as_1d(g["gamma"], dtype),
        "theta": as_1d(g["theta"], dtype) * THETA_PER_DAY,
        "rho": as_1d(g["rho"], dtype) * PER_1PCT,
        "vega": as_1d(g["vega"], dtype) * PER_1PCT,
    }


def price_dataframe(
    df: pd.DataFrame,
    *,
    flag_col: str | None = None,
    underlying_price_col: str | None = None,
    strike_col: str | None = None,
    annualized_tte_col: str | None = None,
    riskfree_rate_col: str | None = None,
    sigma_col: str | None = None,
    price_col: str | None = None,
    dividend_col: str | None = None,
    model: str = "black_scholes",
    inplace: bool = False,
    dtype: DTypeLike = np.float64,
) -> pd.DataFrame | None:
    """Price a DataFrame of contracts by column name.

    Specifying ``sigma_col`` computes ``Price`` (+ greeks); ``price_col`` computes
    ``IV`` (+ greeks); both computes greeks only. Returns a new frame, or writes
    columns in place and returns ``None`` when ``inplace=True``. Requires pandas.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ModuleNotFoundError(
            "price_dataframe requires pandas. Install it with `pip install pyvolr[pandas]`."
        ) from exc

    required = {
        "flag_col": flag_col,
        "underlying_price_col": underlying_price_col,
        "strike_col": strike_col,
        "annualized_tte_col": annualized_tte_col,
        "riskfree_rate_col": riskfree_rate_col,
    }
    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise ValueError(f"price_dataframe requires column args: {', '.join(missing)}")

    # No dtype: flag stays string ('c'/'p') or int (±1); native normalize_flag
    # handles both. (pandas-stubs types the Series element as Unknown here.)
    flag = np.asarray(df[flag_col])  # pyright: ignore[reportUnknownArgumentType]
    S = np.asarray(df[underlying_price_col], dtype=dtype)
    K = np.asarray(df[strike_col], dtype=dtype)
    t = np.asarray(df[annualized_tte_col], dtype=dtype)
    r = np.asarray(df[riskfree_rate_col], dtype=dtype)
    q = None if dividend_col is None else np.asarray(df[dividend_col], dtype=dtype)
    if model == "black_scholes_merton" and q is None:
        raise ValueError("black_scholes_merton model requires a `dividend_col`")

    out = df if inplace else pd.DataFrame(index=df.index)

    if price_col is not None and sigma_col is not None:
        # Both given -> greeks only (values may be inconsistent if from different libs).
        sigma: NDArray[np.float64] = np.asarray(df[sigma_col], dtype=dtype)
    elif price_col is not None:
        price = np.asarray(df[price_col], dtype=dtype)
        iv = vectorized_implied_volatility(
            price, S, K, t, r, flag, q, model=model, return_as="numpy"
        )
        out["IV"] = iv
        sigma = np.asarray(iv, dtype=dtype)
    elif sigma_col is not None:
        sigma = np.asarray(df[sigma_col], dtype=dtype)
        out["Price"] = _price_for(model, flag, S, K, t, r, sigma, q)
    else:
        raise ValueError("price_dataframe requires `sigma_col`, `price_col`, or both")

    for name, col in _greeks_dict(flag, S, K, t, r, sigma, q, model, dtype).items():
        out[name] = col

    return None if inplace else out


def _price_for(
    model: str,
    flag: NDArray[np.float64],
    S: NDArray[np.float64],
    K: NDArray[np.float64],
    t: NDArray[np.float64],
    r: NDArray[np.float64],
    sigma: NDArray[np.float64],
    q: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Numpy price array for the given model (used to fill the ``Price`` column)."""
    if model == "black_scholes":
        return np.asarray(vectorized_black_scholes(flag, S, K, t, r, sigma, return_as="numpy"))
    if model == "black_scholes_merton":
        if q is None:
            raise ValueError("black_scholes_merton model requires a `dividend_col`")
        return np.asarray(
            vectorized_black_scholes_merton(flag, S, K, t, r, sigma, q, return_as="numpy")
        )
    if model == "black":
        return np.asarray(vectorized_black(flag, S, K, t, r, sigma, return_as="numpy"))
    raise ValueError(MODEL_ERR)
