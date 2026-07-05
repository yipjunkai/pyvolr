"""Algorithmically-independent implied-volatility oracle cross-check.

The release-gating differential (``test_differential.py``) compares pyvolr's
implied vol against py_vollib -- but both descend from the same "Let's Be
Rational" lineage, so a shared-algorithm bug could slip past unnoticed. This
suite closes that blind spot with an oracle that shares *no* code with the LBR
inverse: it inverts ``price(sigma) = target`` by plain bisection, calling only
the (independent) forward pricer. Agreement between the two methods on
well-conditioned inputs is evidence the LBR solver converges to the right root
-- not merely to a root the same algorithm's differential also accepts.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from pyvolr import black76, bs

if TYPE_CHECKING:
    from collections.abc import Callable


def _iv_by_bisection(price_of: Callable[[float], float], target: float) -> float:
    """Invert ``price_of(sigma) = target`` for sigma by bisection.

    ``price_of`` is a one-argument pricing callable, monotonically increasing in
    sigma. Independent of the LBR inverse by construction -- it only ever calls
    the forward pricer. Returns ``NaN`` if ``target`` falls outside the
    achievable ``(price(0+), price(inf))`` range.
    """
    lo, hi = 1e-12, 1.0
    if price_of(lo) > target:
        return math.nan  # below the intrinsic / no-arbitrage floor
    for _ in range(64):  # grow the upper bracket until it straddles target
        if price_of(hi) >= target:
            break
        hi *= 2.0
    else:
        return math.nan  # target above the achievable price ceiling
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if price_of(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * max(mid, 1.0):
            break
    return 0.5 * (lo + hi)


# Well-conditioned strategy: ATM-ish, moderate vol & expiry, so vega is large
# and IV is sharply identified (mirrors test_property's conditioning filter).
spot = st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
k_ratio = st.floats(min_value=0.6, max_value=1.6, allow_nan=False, allow_infinity=False)
ttm = st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False)
rate = st.floats(min_value=-0.02, max_value=0.15, allow_nan=False, allow_infinity=False)
yield_ = st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False)
vol = st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False)
flag = st.sampled_from(["c", "p"])

# Two independent roots of price(sigma) = target agree to the pricing precision
# divided by vega (~1e-13 for well-conditioned inputs); 1e-7 leaves ample margin
# against flakiness while still catching a wrong-root / wrong-branch regression.
_ORACLE_REL = 1e-7
_ORACLE_ABS = 1e-9


@pytest.mark.property
class TestIvOracleBSM:
    @given(spot, k_ratio, ttm, rate, yield_, vol, flag)
    def test_lbr_matches_bisection(
        self, s: float, kr: float, t: float, r: float, q: float, sigma: float, fl: str
    ) -> None:
        k = s * kr
        price = bs.price(fl, S=s, K=k, T=t, r=r, sigma=sigma, q=q)
        assume(price > 1e-6 * max(1.0, s))
        assume(bs.vega(S=s, K=k, T=t, r=r, sigma=sigma, q=q) > 1e-3 * max(1.0, s))

        lbr = bs.implied_vol(price, fl, S=s, K=k, T=t, r=r, q=q, on_error="ignore")
        assume(not np.isnan(lbr))

        oracle = _iv_by_bisection(
            lambda sig: bs.price(fl, S=s, K=k, T=t, r=r, sigma=sig, q=q), price
        )
        assert not math.isnan(oracle)
        assert lbr == pytest.approx(oracle, rel=_ORACLE_REL, abs=_ORACLE_ABS)


@pytest.mark.property
class TestIvOracleBlack76:
    @given(spot, k_ratio, ttm, rate, vol, flag)
    def test_lbr_matches_bisection(
        self, f: float, kr: float, t: float, r: float, sigma: float, fl: str
    ) -> None:
        k = f * kr
        price = black76.price(fl, F=f, K=k, T=t, r=r, sigma=sigma)
        assume(price > 1e-6 * max(1.0, f))
        assume(black76.vega(F=f, K=k, T=t, r=r, sigma=sigma) > 1e-3 * max(1.0, f))

        lbr = black76.implied_vol(price, fl, F=f, K=k, T=t, r=r, on_error="ignore")
        assume(not np.isnan(lbr))

        oracle = _iv_by_bisection(
            lambda sig: black76.price(fl, F=f, K=k, T=t, r=r, sigma=sig), price
        )
        assert not math.isnan(oracle)
        assert lbr == pytest.approx(oracle, rel=_ORACLE_REL, abs=_ORACLE_ABS)
