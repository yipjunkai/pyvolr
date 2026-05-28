"""Tests for `pyvolr.black76` — Black-76 model for futures/forward options."""

from __future__ import annotations

import numpy as np
import pytest

from pyvolr import black76, bs


class TestSingleValuePricing:
    """Reference values from py_vollib.black doctests and the BSM equivalence."""

    def test_atm_call_py_vollib_doctest(self) -> None:
        # py_vollib doctest: F=K=100, r=0.02, t=0.5, sigma=0.2 -> 5.5811067246048118
        p = black76.price("c", F=100.0, K=100.0, T=0.5, r=0.02, sigma=0.20)
        assert isinstance(p, float)
        assert p == pytest.approx(5.5811067246048118, abs=1e-10)

    def test_matches_bsm_with_q_equal_r(self) -> None:
        # Black-76 is BSM with S=F, q=r.
        F, K, T, r, sigma = 100.0, 105.0, 0.5, 0.05, 0.20
        b76_c = black76.price("c", F=F, K=K, T=T, r=r, sigma=sigma)
        bsm_c = bs.price("c", S=F, K=K, T=T, r=r, sigma=sigma, q=r)
        assert b76_c == pytest.approx(bsm_c, abs=1e-12)

        b76_p = black76.price("p", F=F, K=K, T=T, r=r, sigma=sigma)
        bsm_p = bs.price("p", S=F, K=K, T=T, r=r, sigma=sigma, q=r)
        assert b76_p == pytest.approx(bsm_p, abs=1e-12)

    def test_uppercase_flag(self) -> None:
        a = black76.price("c", F=100, K=100, T=1.0, r=0.05, sigma=0.20)
        b = black76.price("C", F=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert a == b

    def test_invalid_flag_raises(self) -> None:
        with pytest.raises(ValueError, match="flag must be"):
            black76.price("x", F=100, K=100, T=1.0, r=0.05, sigma=0.20)


class TestPutCallParity:
    """C - P = exp(-r*T) * (F - K) for Black-76."""

    @pytest.mark.parametrize(
        ("f", "k", "t", "r", "sigma"),
        [
            (100.0, 100.0, 1.0, 0.05, 0.20),
            (100.0, 110.0, 0.5, 0.03, 0.30),
            (50.0, 60.0, 2.0, 0.04, 0.50),
            (1000.0, 950.0, 0.1, 0.06, 0.15),
        ],
    )
    def test_parity(self, f: float, k: float, t: float, r: float, sigma: float) -> None:
        c = black76.price("c", F=f, K=k, T=t, r=r, sigma=sigma)
        p = black76.price("p", F=f, K=k, T=t, r=r, sigma=sigma)
        lhs = c - p
        rhs = np.exp(-r * t) * (f - k)
        assert lhs == pytest.approx(rhs, abs=1e-10)


class TestGreeks:
    """Golden values from py_vollib doctests, converted to per-unit conventions."""

    F: float = 49.0
    K: float = 50.0
    T: float = 0.3846
    R: float = 0.05
    SIGMA: float = 0.20

    def test_delta_call(self) -> None:
        # py_vollib doctest: 0.45107017482201828
        d = black76.delta("c", F=self.F, K=self.K, T=self.T, r=self.R, sigma=self.SIGMA)
        assert d == pytest.approx(0.45107017482201828, abs=1e-10)

    def test_gamma_call(self) -> None:
        # py_vollib doctest: 0.0640646705882 (call/put identical)
        g = black76.gamma(F=self.F, K=self.K, T=self.T, r=self.R, sigma=self.SIGMA)
        assert g == pytest.approx(0.0640646705882, abs=1e-10)

    def test_vega_call_per_unit(self) -> None:
        # py_vollib's 0.118317785624 is per-1%-vol; per-unit is 100x.
        v = black76.vega(F=self.F, K=self.K, T=self.T, r=self.R, sigma=self.SIGMA)
        assert v == pytest.approx(0.118317785624 * 100.0, abs=1e-8)

    def test_theta_call_per_year(self) -> None:
        # py_vollib's -0.00816236877462 is per-day; per-year is 365x.
        th = black76.theta("c", F=self.F, K=self.K, T=self.T, r=self.R, sigma=self.SIGMA)
        assert th == pytest.approx(-0.00816236877462 * 365.0, abs=1e-7)

    def test_rho_call_per_unit(self) -> None:
        # py_vollib's -0.0074705380059582258 is per-1%-rate; per-unit is 100x.
        rh = black76.rho("c", F=self.F, K=self.K, T=self.T, r=self.R, sigma=self.SIGMA)
        assert rh == pytest.approx(-0.0074705380059582258 * 100.0, abs=1e-8)

    def test_rho_equals_minus_t_times_price(self) -> None:
        # The defining property of Black-76 rho.
        for flag in ("c", "p"):
            p = black76.price(flag, F=self.F, K=self.K, T=self.T, r=self.R, sigma=self.SIGMA)
            rh = black76.rho(flag, F=self.F, K=self.K, T=self.T, r=self.R, sigma=self.SIGMA)
            assert rh == pytest.approx(-self.T * p, abs=1e-12)

    def test_greeks_dict(self) -> None:
        g = black76.greeks("c", F=self.F, K=self.K, T=self.T, r=self.R, sigma=self.SIGMA)
        assert set(g.keys()) == {"delta", "gamma", "theta", "vega", "rho"}


class TestBroadcasting:
    def test_vector_strikes(self) -> None:
        strikes = np.linspace(80, 120, 9)
        prices = black76.price("c", F=100, K=strikes, T=0.5, r=0.05, sigma=0.20)
        assert isinstance(prices, np.ndarray)
        assert prices.shape == (9,)
        # Call price is decreasing in strike.
        assert np.all(np.diff(prices) < 0)

    def test_2d_grid(self) -> None:
        strikes = np.linspace(80, 120, 5).reshape(-1, 1)
        sigmas = np.linspace(0.1, 0.4, 3).reshape(1, -1)
        prices = black76.price("c", F=100, K=strikes, T=0.5, r=0.05, sigma=sigmas)
        assert isinstance(prices, np.ndarray)
        assert prices.shape == (5, 3)
        # Price is increasing in vol.
        assert np.all(np.diff(prices, axis=1) > 0)

    def test_scalar_in_scalar_out(self) -> None:
        out = black76.price("c", F=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert isinstance(out, float)


class TestEdgeCases:
    def test_zero_time_returns_intrinsic(self) -> None:
        # At t=0, no discounting either: price = max(F-K, 0) for call, max(K-F, 0) for put.
        assert black76.price("c", F=110, K=100, T=0.0, r=0.05, sigma=0.20) == pytest.approx(10.0)
        assert black76.price("p", F=90, K=100, T=0.0, r=0.05, sigma=0.20) == pytest.approx(10.0)
        assert black76.price("c", F=90, K=100, T=0.0, r=0.05, sigma=0.20) == pytest.approx(0.0)

    def test_zero_vol_returns_discounted_intrinsic(self) -> None:
        # With sigma=0, the forward IS the price at expiry (no randomness).
        # Call payoff at expiry: max(F-K, 0). Discounted: exp(-rT) * max(F-K, 0).
        p = black76.price("c", F=110, K=100, T=1.0, r=0.05, sigma=0.0)
        expected = np.exp(-0.05) * 10.0
        assert p == pytest.approx(expected, abs=1e-12)

    def test_deep_otm_call_near_zero(self) -> None:
        p = black76.price("c", F=100, K=1000, T=0.1, r=0.05, sigma=0.20)
        assert p < 1e-30

    def test_deep_itm_call_near_discounted_intrinsic(self) -> None:
        p = black76.price("c", F=1000, K=100, T=0.01, r=0.05, sigma=0.20)
        expected = np.exp(-0.05 * 0.01) * (1000 - 100)
        assert p == pytest.approx(expected, rel=1e-6)


class TestImpliedVol:
    def test_roundtrip_atm(self) -> None:
        F, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.25
        p = black76.price("c", F=F, K=K, T=T, r=r, sigma=sigma)
        recovered = black76.implied_vol(price=p, flag="c", F=F, K=K, T=T, r=r)
        # Black-76 IV routes through iv::solve (the LBR solver), so the same
        # ~1e-13 precision applies as for BSM IV.
        assert recovered == pytest.approx(sigma, rel=1e-12)

    def test_below_no_arb_returns_nan(self) -> None:
        # No-arb lower bound for call: exp(-rT) * max(F-K, 0).
        iv = black76.implied_vol(price=-1.0, flag="c", F=100, K=100, T=1.0, r=0.05)
        assert np.isnan(iv)

    def test_above_no_arb_returns_nan(self) -> None:
        # Upper bound for call: exp(-rT) * F.
        iv = black76.implied_vol(price=1e6, flag="c", F=100, K=100, T=1.0, r=0.05)
        assert np.isnan(iv)


class TestParallelDispatch:
    """Exercise the rayon-parallel branch of `black76_greeks` (above N=4096)
    and `black76_iv` (above N=1024).

    Mirrors the BSM tests in `test_greeks.py` / `test_iv.py`: the
    Rust-level parity (`black76::tests::all_matches_individual_at_grid`)
    covers `black76::all` itself; these tests cover the PyO3 dispatch +
    tuple-unzip path at the FFI boundary, for both call and put flags.
    """

    @pytest.mark.parametrize("flag", ["c", "p"])
    def test_greeks_above_threshold_matches_individual(self, flag: str) -> None:
        n = 8192  # > GREEKS_PARALLEL_THRESHOLD (4096)
        K = np.linspace(80, 120, n)
        F, T, r, sigma = 100.0, 0.5, 0.05, 0.20
        g = black76.greeks(flag, F=F, K=K, T=T, r=r, sigma=sigma)
        np.testing.assert_allclose(
            g["delta"], black76.delta(flag, F=F, K=K, T=T, r=r, sigma=sigma), rtol=1e-14
        )
        np.testing.assert_allclose(
            g["gamma"], black76.gamma(F=F, K=K, T=T, r=r, sigma=sigma), rtol=1e-14
        )
        np.testing.assert_allclose(
            g["vega"], black76.vega(F=F, K=K, T=T, r=r, sigma=sigma), rtol=1e-14
        )
        np.testing.assert_allclose(
            g["theta"], black76.theta(flag, F=F, K=K, T=T, r=r, sigma=sigma), rtol=1e-14
        )
        np.testing.assert_allclose(
            g["rho"], black76.rho(flag, F=F, K=K, T=T, r=r, sigma=sigma), rtol=1e-14
        )

    @pytest.mark.parametrize("flag", ["c", "p"])
    def test_iv_above_threshold(self, flag: str) -> None:
        n = 2048  # > PARALLEL_THRESHOLD (1024)
        K = np.linspace(80, 120, n)
        F, T, r, sigma = 100.0, 0.5, 0.05, 0.20
        prices = black76.price(flag, F=F, K=K, T=T, r=r, sigma=sigma)
        ivs = black76.implied_vol(prices, flag, F=F, K=K, T=T, r=r)
        assert isinstance(ivs, np.ndarray)
        assert ivs.shape == (n,)
        np.testing.assert_allclose(ivs, sigma, rtol=1e-12)
