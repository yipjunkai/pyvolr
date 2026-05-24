"""Hypothesis property tests covering BSM and Black-76 invariants.

These tests describe what _must_ hold for any inputs in a sensible range,
not specific values. They are the cheapest insurance against regressions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from pyvolr import black76, bs

# Range that excludes pathological extremes while still stress-testing the math.
spot = st.floats(min_value=0.5, max_value=10_000.0, allow_nan=False, allow_infinity=False)
strike = st.floats(min_value=0.5, max_value=10_000.0, allow_nan=False, allow_infinity=False)
ttm = st.floats(min_value=1e-4, max_value=10.0, allow_nan=False, allow_infinity=False)
rate = st.floats(min_value=-0.05, max_value=0.20, allow_nan=False, allow_infinity=False)
yield_ = st.floats(min_value=0.0, max_value=0.15, allow_nan=False, allow_infinity=False)
vol = st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False)
flag = st.sampled_from(["c", "p"])


@pytest.mark.property
class TestParity:
    @given(spot, strike, ttm, rate, yield_, vol)
    def test_put_call_parity(
        self, s: float, k: float, t: float, r: float, q: float, sigma: float
    ) -> None:
        c = bs.price("c", S=s, K=k, T=t, r=r, sigma=sigma, q=q)
        p = bs.price("p", S=s, K=k, T=t, r=r, sigma=sigma, q=q)
        lhs = c - p
        rhs = s * math.exp(-q * t) - k * math.exp(-r * t)
        # Absolute tolerance scaled by the magnitude of the larger leg.
        atol = 1e-8 * max(1.0, abs(s) + abs(k))
        assert math.isclose(lhs, rhs, abs_tol=atol)


@pytest.mark.property
class TestMonotonicity:
    @given(spot, strike, ttm, rate, yield_, vol)
    def test_call_increasing_in_spot(
        self, s: float, k: float, t: float, r: float, q: float, sigma: float
    ) -> None:
        p1 = bs.price("c", S=s, K=k, T=t, r=r, sigma=sigma, q=q)
        p2 = bs.price("c", S=s * 1.05, K=k, T=t, r=r, sigma=sigma, q=q)
        assert p2 >= p1 - 1e-12

    @given(spot, strike, ttm, rate, yield_, vol)
    def test_call_decreasing_in_strike(
        self, s: float, k: float, t: float, r: float, q: float, sigma: float
    ) -> None:
        p1 = bs.price("c", S=s, K=k, T=t, r=r, sigma=sigma, q=q)
        p2 = bs.price("c", S=s, K=k * 1.05, T=t, r=r, sigma=sigma, q=q)
        assert p2 <= p1 + 1e-12

    @given(spot, strike, ttm, rate, yield_, vol, flag)
    def test_price_increasing_in_vol(
        self, s: float, k: float, t: float, r: float, q: float, sigma: float, fl: str
    ) -> None:
        assume(sigma < 1.9)
        p1 = bs.price(fl, S=s, K=k, T=t, r=r, sigma=sigma, q=q)
        p2 = bs.price(fl, S=s, K=k, T=t, r=r, sigma=sigma + 0.05, q=q)
        assert p2 >= p1 - 1e-10


@pytest.mark.property
class TestNonNegative:
    @given(spot, strike, ttm, rate, yield_, vol, flag)
    def test_price_nonneg(
        self, s: float, k: float, t: float, r: float, q: float, sigma: float, fl: str
    ) -> None:
        p = bs.price(fl, S=s, K=k, T=t, r=r, sigma=sigma, q=q)
        assert p >= -1e-12

    @given(spot, strike, ttm, rate, yield_, vol)
    def test_gamma_nonneg(
        self, s: float, k: float, t: float, r: float, q: float, sigma: float
    ) -> None:
        g = bs.gamma(S=s, K=k, T=t, r=r, sigma=sigma, q=q)
        assert g >= -1e-12

    @given(spot, strike, ttm, rate, yield_, vol)
    def test_vega_nonneg(
        self, s: float, k: float, t: float, r: float, q: float, sigma: float
    ) -> None:
        v = bs.vega(S=s, K=k, T=t, r=r, sigma=sigma, q=q)
        assert v >= -1e-12


@pytest.mark.property
class TestIvRoundtrip:
    """IV is only well-identified when vega is non-trivial. The correct
    correctness statement is `price(IV(p)) = p`, not `IV(price(sigma)) = sigma`
    (deep ITM/OTM options have ill-conditioned IV-from-price inversion).
    """

    @given(
        spot,
        strike,
        st.floats(min_value=0.01, max_value=2.0, allow_nan=False),
        rate,
        yield_,
        st.floats(min_value=0.05, max_value=1.5, allow_nan=False),
        flag,
    )
    def test_iv_then_price_reproduces_price(
        self, s: float, k: float, t: float, r: float, q: float, sigma: float, fl: str
    ) -> None:
        p = bs.price(fl, S=s, K=k, T=t, r=r, sigma=sigma, q=q)
        assume(p > 1e-6)
        iv = bs.implied_vol(p, fl, S=s, K=k, T=t, r=r, q=q)
        assume(not np.isnan(iv))
        p_recovered = bs.price(fl, S=s, K=k, T=t, r=r, sigma=iv, q=q)
        # Price -> IV -> price must agree to within the solver's PRICE_TOL (1e-10).
        atol = max(1e-9, 1e-9 * max(1.0, p))
        assert p_recovered == pytest.approx(p, abs=atol, rel=1e-8)

    @given(
        spot,
        st.floats(min_value=0.5, max_value=2.0, allow_nan=False),  # K near S
        st.floats(min_value=0.1, max_value=5.0, allow_nan=False),  # not too short
        rate,
        yield_,
        st.floats(min_value=0.1, max_value=1.0, allow_nan=False),  # mid vol
        flag,
    )
    def test_iv_recovers_sigma_when_well_conditioned(
        self, s: float, k_ratio: float, t: float, r: float, q: float, sigma: float, fl: str
    ) -> None:
        # ATM-ish, moderate vol & expiry: vega is large, IV is identified.
        k = s * k_ratio
        p = bs.price(fl, S=s, K=k, T=t, r=r, sigma=sigma, q=q)
        v = bs.vega(S=s, K=k, T=t, r=r, sigma=sigma, q=q)
        assume(v > 1e-4 * max(1.0, s))  # filter ill-conditioned tails
        iv = bs.implied_vol(p, fl, S=s, K=k, T=t, r=r, q=q)
        assume(not np.isnan(iv))
        assert iv == pytest.approx(sigma, abs=1e-6)


# Black-76 uses forward F instead of spot S; same parameter ranges otherwise.
forward = spot


@pytest.mark.property
class TestBlack76Parity:
    @given(forward, strike, ttm, rate, vol)
    def test_put_call_parity(self, f: float, k: float, t: float, r: float, sigma: float) -> None:
        # C - P = exp(-r*T) * (F - K)
        c = black76.price("c", F=f, K=k, T=t, r=r, sigma=sigma)
        p = black76.price("p", F=f, K=k, T=t, r=r, sigma=sigma)
        lhs = c - p
        rhs = math.exp(-r * t) * (f - k)
        atol = 1e-8 * max(1.0, abs(f) + abs(k))
        assert math.isclose(lhs, rhs, abs_tol=atol)


@pytest.mark.property
class TestBlack76Monotonicity:
    @given(forward, strike, ttm, rate, vol)
    def test_call_increasing_in_forward(
        self, f: float, k: float, t: float, r: float, sigma: float
    ) -> None:
        p1 = black76.price("c", F=f, K=k, T=t, r=r, sigma=sigma)
        p2 = black76.price("c", F=f * 1.05, K=k, T=t, r=r, sigma=sigma)
        assert p2 >= p1 - 1e-12

    @given(forward, strike, ttm, rate, vol)
    def test_call_decreasing_in_strike(
        self, f: float, k: float, t: float, r: float, sigma: float
    ) -> None:
        p1 = black76.price("c", F=f, K=k, T=t, r=r, sigma=sigma)
        p2 = black76.price("c", F=f, K=k * 1.05, T=t, r=r, sigma=sigma)
        assert p2 <= p1 + 1e-12

    @given(forward, strike, ttm, rate, vol, flag)
    def test_price_increasing_in_vol(
        self, f: float, k: float, t: float, r: float, sigma: float, fl: str
    ) -> None:
        assume(sigma < 1.9)
        p1 = black76.price(fl, F=f, K=k, T=t, r=r, sigma=sigma)
        p2 = black76.price(fl, F=f, K=k, T=t, r=r, sigma=sigma + 0.05)
        assert p2 >= p1 - 1e-10


@pytest.mark.property
class TestBlack76NonNegative:
    @given(forward, strike, ttm, rate, vol, flag)
    def test_price_nonneg(
        self, f: float, k: float, t: float, r: float, sigma: float, fl: str
    ) -> None:
        p = black76.price(fl, F=f, K=k, T=t, r=r, sigma=sigma)
        assert p >= -1e-12

    @given(forward, strike, ttm, rate, vol)
    def test_gamma_nonneg(self, f: float, k: float, t: float, r: float, sigma: float) -> None:
        g = black76.gamma(F=f, K=k, T=t, r=r, sigma=sigma)
        assert g >= -1e-12

    @given(forward, strike, ttm, rate, vol)
    def test_vega_nonneg(self, f: float, k: float, t: float, r: float, sigma: float) -> None:
        v = black76.vega(F=f, K=k, T=t, r=r, sigma=sigma)
        assert v >= -1e-12


@pytest.mark.property
class TestBlack76IvRoundtrip:
    @given(
        forward,
        st.floats(min_value=0.5, max_value=2.0, allow_nan=False),  # K/F ratio
        st.floats(min_value=0.1, max_value=5.0, allow_nan=False),
        rate,
        st.floats(min_value=0.1, max_value=1.0, allow_nan=False),
        flag,
    )
    def test_iv_recovers_sigma_when_well_conditioned(
        self, f: float, k_ratio: float, t: float, r: float, sigma: float, fl: str
    ) -> None:
        k = f * k_ratio
        p = black76.price(fl, F=f, K=k, T=t, r=r, sigma=sigma)
        v = black76.vega(F=f, K=k, T=t, r=r, sigma=sigma)
        assume(v > 1e-4 * max(1.0, f))
        iv = black76.implied_vol(p, fl, F=f, K=k, T=t, r=r)
        assume(not np.isnan(iv))
        assert iv == pytest.approx(sigma, abs=1e-6)
