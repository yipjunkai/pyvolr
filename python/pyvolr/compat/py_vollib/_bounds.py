"""Shared no-arbitrage bound check for the compat `implied_volatility` shims.

pyvolr's native IV returns NaN when a price has no real implied volatility
(below intrinsic / above the no-arbitrage maximum) — the correct vectorised
semantics. The compat shims instead mirror py_vollib's scalar *raise* contract,
so this helper maps a NaN result onto the matching py_vollib exception.
"""

from __future__ import annotations

import math

from .exceptions import AboveMaximumException, BelowIntrinsicException


def raise_for_iv_price(price: float, disc_underlying: float, disc_strike: float, flag: str) -> None:
    """Raise the py_vollib exception matching an out-of-bounds IV `price`.

    `disc_underlying = S*exp(-q*t)` and `disc_strike = K*exp(-r*t)` (for
    Black-76, `S = F` and `q = r`). The no-arbitrage bounds mirror
    `pyvolr_core::iv::solve`: a call lives in `[max(0, U - Kd), U]`, a put in
    `[max(0, Kd - U), Kd]`.

    No-op for non-finite or non-positive inputs: those are degenerate (t <= 0,
    S <= 0, K <= 0), not arbitrage violations, and the shim returns pyvolr's NaN
    there -- graceful where py_vollib leaks a `ZeroDivisionError` / `ValueError`.
    Only reached after the native solver already returned NaN, i.e. the price is
    genuinely outside the bounds.
    """
    if not (math.isfinite(price) and disc_underlying > 0.0 and disc_strike > 0.0):
        return
    if flag.lower() == "c":
        lower, upper = max(0.0, disc_underlying - disc_strike), disc_underlying
    else:
        lower, upper = max(0.0, disc_strike - disc_underlying), disc_strike
    if price < lower:
        raise BelowIntrinsicException
    if price > upper:
        raise AboveMaximumException
