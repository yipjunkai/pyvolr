"""Tests for `pyvolr.bs` pricing."""

from __future__ import annotations

import numpy as np
import pytest

from pyvolr import bs


class TestSingleValuePricing:
    """Reference values from Hull (10th ed.) and direct closed-form computation."""

    def test_atm_call_hull_example(self) -> None:
        # Hull Ch. 15: S=K=100, r=5%, sigma=20%, T=1 -> call ~10.4506
        p = bs.price("c", S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert isinstance(p, float)
        assert p == pytest.approx(10.450583572185565, abs=1e-10)

    def test_atm_put_hull_example(self) -> None:
        p = bs.price("p", S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert p == pytest.approx(5.573526022256971, abs=1e-10)

    def test_uppercase_flag(self) -> None:
        a = bs.price("c", S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        b = bs.price("C", S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert a == b

    def test_invalid_flag_raises(self) -> None:
        with pytest.raises(ValueError, match="flag must be"):
            bs.price("x", S=100, K=100, T=1.0, r=0.05, sigma=0.20)

    def test_with_dividend_yield(self) -> None:
        # With q > 0, call should be cheaper, put more expensive.
        c_no_div = bs.price("c", S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        c_with_div = bs.price("c", S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.03)
        p_no_div = bs.price("p", S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        p_with_div = bs.price("p", S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.03)
        assert c_with_div < c_no_div
        assert p_with_div > p_no_div


class TestPutCallParity:
    """C - P = S*exp(-qT) - K*exp(-rT) for any S, K, T, r, q, sigma."""

    @pytest.mark.parametrize(
        ("s", "k", "t", "r", "q", "sigma"),
        [
            (100.0, 100.0, 1.0, 0.05, 0.0, 0.20),
            (100.0, 110.0, 0.5, 0.03, 0.02, 0.30),
            (50.0, 60.0, 2.0, 0.04, 0.01, 0.50),
            (1000.0, 950.0, 0.1, 0.06, 0.0, 0.15),
        ],
    )
    def test_parity(self, s: float, k: float, t: float, r: float, q: float, sigma: float) -> None:
        c = bs.price("c", S=s, K=k, T=t, r=r, sigma=sigma, q=q)
        p = bs.price("p", S=s, K=k, T=t, r=r, sigma=sigma, q=q)
        assert isinstance(c, float)
        assert isinstance(p, float)
        lhs = c - p
        rhs = s * np.exp(-q * t) - k * np.exp(-r * t)
        assert lhs == pytest.approx(rhs, abs=1e-10)


class TestBroadcasting:
    """Inputs of mixed shape should broadcast like numpy ufuncs."""

    def test_vector_strikes(self) -> None:
        strikes = np.linspace(80, 120, 9)
        prices = bs.price("c", S=100, K=strikes, T=0.5, r=0.05, sigma=0.20)
        assert isinstance(prices, np.ndarray)
        assert prices.shape == (9,)
        # Call price is decreasing in strike.
        assert np.all(np.diff(prices) < 0)

    def test_2d_grid(self) -> None:
        strikes = np.linspace(80, 120, 5).reshape(-1, 1)
        sigmas = np.linspace(0.1, 0.4, 3).reshape(1, -1)
        prices = bs.price("c", S=100, K=strikes, T=0.5, r=0.05, sigma=sigmas)
        assert isinstance(prices, np.ndarray)
        assert prices.shape == (5, 3)
        # Price is increasing in vol.
        assert np.all(np.diff(prices, axis=1) > 0)

    def test_flag_array(self) -> None:
        flags = np.array(["c", "p", "c", "p"])
        strikes = np.array([100.0, 100.0, 110.0, 110.0])
        out = bs.price(flags, S=100, K=strikes, T=1.0, r=0.05, sigma=0.20)
        assert isinstance(out, np.ndarray)
        assert out.shape == (4,)

    def test_scalar_in_scalar_out(self) -> None:
        out = bs.price("c", S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert isinstance(out, float)


class TestEdgeCases:
    def test_zero_time_returns_intrinsic(self) -> None:
        assert bs.price("c", S=110, K=100, T=0.0, r=0.05, sigma=0.20) == pytest.approx(10.0)
        assert bs.price("p", S=90, K=100, T=0.0, r=0.05, sigma=0.20) == pytest.approx(10.0)
        assert bs.price("c", S=90, K=100, T=0.0, r=0.05, sigma=0.20) == pytest.approx(0.0)

    def test_zero_vol(self) -> None:
        # Deterministic forward.
        p = bs.price("c", S=100, K=100, T=1.0, r=0.05, sigma=0.0)
        forward = 100 * np.exp(0.05)
        expected = np.exp(-0.05) * max(0.0, forward - 100)
        assert p == pytest.approx(expected, abs=1e-12)

    def test_deep_otm_call_near_zero(self) -> None:
        p = bs.price("c", S=100, K=1000, T=0.1, r=0.05, sigma=0.20)
        assert p < 1e-30

    def test_deep_itm_call_near_intrinsic(self) -> None:
        p = bs.price("c", S=1000, K=100, T=0.01, r=0.05, sigma=0.20)
        # Should be very close to S - K*exp(-rT)
        expected = 1000 - 100 * np.exp(-0.05 * 0.01)
        assert p == pytest.approx(expected, rel=1e-6)
