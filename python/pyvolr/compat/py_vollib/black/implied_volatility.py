"""Drop-in shim for `py_vollib.black.implied_volatility`.

py_vollib's signature here puts `r` BEFORE `t` (opposite of
`py_vollib.black_scholes.implied_volatility`, which is `t, r`). This shim
preserves that exact argument order — flag is last either way.
"""

from __future__ import annotations

from pyvolr import black76 as _b76

__all__ = ["implied_volatility"]


def implied_volatility(price: float, F: float, K: float, r: float, t: float, flag: str) -> float:
    """Black-76 implied volatility from a market price.

    Argument order matches py_vollib.black exactly: (price, F, K, r, t, flag).
    """
    return float(_b76.implied_vol(price, flag, F, K, t, r))
