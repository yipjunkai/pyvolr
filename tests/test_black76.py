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

    def test_invalid_flag_array_raises(self) -> None:
        with pytest.raises(ValueError, match="flag array"):
            black76.price(["call"], F=[100.0], K=[100.0], T=[1.0], r=[0.05], sigma=[0.20])

    def test_flag_array_mixed_case_ok(self) -> None:
        # The broadcast shape comes from the numeric inputs, so the flag
        # array needs array-shaped numerics to broadcast against.
        f = [100.0, 100.0]
        out = black76.price(np.array(["C", "p"]), F=f, K=100, T=1.0, r=0.05, sigma=0.20)
        call = black76.price("c", F=100, K=100, T=1.0, r=0.05, sigma=0.20)
        put = black76.price("p", F=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert isinstance(out, np.ndarray)
        assert out[0] == call
        assert out[1] == put


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
        iv = black76.implied_vol(
            price=-1.0, flag="c", F=100, K=100, T=1.0, r=0.05, on_error="ignore"
        )
        assert np.isnan(iv)

    def test_above_no_arb_returns_nan(self) -> None:
        # Upper bound for call: exp(-rT) * F.
        iv = black76.implied_vol(
            price=1e6, flag="c", F=100, K=100, T=1.0, r=0.05, on_error="ignore"
        )
        assert np.isnan(iv)


class TestPrecisionCorners:
    """Python-level guards for the deep-OTM put-delta corner.

    Mirror of the BSM tests in `test_greeks.py::TestPrecisionCorners`.
    Black-76's `delta` and `greeks` route through `greeks::delta` /
    `black76::all`, both touched by the put-delta fix (commit 30d5d1f).
    """

    def test_put_delta_deep_otm_retains_precision(self) -> None:
        d = black76.delta("p", F=1000, K=100, T=0.5, r=0.05, sigma=0.20)
        assert isinstance(d, float)
        assert d < 0.0, f"put delta lost sign at deep OTM (got {d:e})"
        assert 0.0 < abs(d) < 1e-50

    def test_bundled_greeks_put_delta_deep_otm_retains_precision(self) -> None:
        g = black76.greeks("p", F=1000, K=100, T=0.5, r=0.05, sigma=0.20)
        d = g["delta"]
        assert isinstance(d, float)
        assert d < 0.0, f"put delta lost sign at deep OTM via greeks() (got {d:e})"
        assert 0.0 < abs(d) < 1e-50


class TestParallelDispatch:
    """Exercise the rayon-parallel branches of the Black-76 array endpoints:
    `black76_greeks` (above N=4096), `black76_iv` (above N=1024), and — since
    0.1.7 — `black76_price` (above N=8192) and the single Greeks (above
    N=16384).

    Mirrors the BSM tests in `test_greeks.py` / `test_iv.py` / `test_bsm.py`:
    the Rust-level parity (`black76::tests::all_matches_individual_at_grid`)
    covers `black76::all` itself; these tests cover the PyO3 dispatch path at
    the FFI boundary, for both call and put flags. `price`/single-Greek rows
    are computed independently, so the parallel path is bit-identical to
    serial (tile trick: diverse pattern serially, tiled past the gate).
    """

    @pytest.mark.parametrize("flag", ["c", "p"])
    def test_price_parallel_matches_serial(self, flag: str) -> None:
        pk = np.linspace(20.0, 500.0, 384)  # 384 < PRICE_PARALLEL_THRESHOLD
        ref = black76.price(flag, F=100.0, K=pk, T=0.5, r=0.05, sigma=0.20)
        reps = 40  # 384 * 40 = 15360 >= PRICE_PARALLEL_THRESHOLD (8192)
        bk = np.tile(pk, reps)
        got = black76.price(flag, F=100.0, K=bk, T=0.5, r=0.05, sigma=0.20)
        np.testing.assert_array_equal(got, np.tile(ref, reps))

    @pytest.mark.parametrize("flag", ["c", "p"])
    def test_single_greeks_parallel_match_serial(self, flag: str) -> None:
        pk = np.linspace(20.0, 500.0, 384)
        reps = 48  # 384 * 48 = 18432 >= SINGLE_GREEK_PARALLEL_THRESHOLD (16384)
        bk = np.tile(pk, reps)
        F, T, r, sigma = 100.0, 0.5, 0.05, 0.20

        def check(name: str, ref: np.ndarray, got: np.ndarray) -> None:
            np.testing.assert_array_equal(
                got, np.tile(ref, reps), err_msg=f"{name} parallel != serial (flag={flag})"
            )

        check(
            "delta",
            black76.delta(flag, F=F, K=pk, T=T, r=r, sigma=sigma),
            black76.delta(flag, F=F, K=bk, T=T, r=r, sigma=sigma),
        )
        check(
            "theta",
            black76.theta(flag, F=F, K=pk, T=T, r=r, sigma=sigma),
            black76.theta(flag, F=F, K=bk, T=T, r=r, sigma=sigma),
        )
        check(
            "rho",
            black76.rho(flag, F=F, K=pk, T=T, r=r, sigma=sigma),
            black76.rho(flag, F=F, K=bk, T=T, r=r, sigma=sigma),
        )
        check(
            "gamma",
            black76.gamma(F=F, K=pk, T=T, r=r, sigma=sigma),
            black76.gamma(F=F, K=bk, T=T, r=r, sigma=sigma),
        )
        check(
            "vega",
            black76.vega(F=F, K=pk, T=T, r=r, sigma=sigma),
            black76.vega(F=F, K=bk, T=T, r=r, sigma=sigma),
        )

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


class TestHigherGreeks:
    """Higher-order Black-76 Greeks: q = r specialization of BSM."""

    _ORDER = ("vanna", "vomma", "charm", "speed", "zomma", "color", "veta", "ultima")
    _GRID = (
        ("c", 100.0, 100.0, 1.0, 0.05, 0.20),
        ("p", 100.0, 120.0, 0.5, 0.03, 0.30),
        ("c", 49.0, 50.0, 0.3846, 0.05, 0.20),
    )

    @pytest.mark.parametrize("case", _GRID)
    def test_matches_bsm_with_q_equal_r(
        self, case: tuple[str, float, float, float, float, float]
    ) -> None:
        # Bit-identical: both resolve to greeks::higher_all(F, K, T, r, r, sigma).
        # This ties Black-76 to the mpmath goldens transitively (see test_greeks).
        flag, f, k, t, r, sigma = case
        b = black76.higher_greeks(flag, F=f, K=k, T=t, r=r, sigma=sigma)
        m = bs.higher_greeks(flag, S=f, K=k, T=t, r=r, sigma=sigma, q=r)
        for name in self._ORDER:
            assert b[name] == m[name], name

    @pytest.mark.parametrize("flag", ["c", "p"])
    def test_individual_equals_bundle(self, flag: str) -> None:
        f, k, t, r, sigma = 100.0, 105.0, 0.5, 0.05, 0.25
        g = black76.higher_greeks(flag, F=f, K=k, T=t, r=r, sigma=sigma)
        kw = {"F": f, "K": k, "T": t, "r": r, "sigma": sigma}
        for name in self._ORDER:
            # charm needs explicit kwargs (a **dict splat can't match the
            # overloads under pyright-strict); the rest go through getattr.
            if name == "charm":
                got = black76.charm(flag, F=f, K=k, T=t, r=r, sigma=sigma)
            else:
                got = getattr(black76, name)(**kw)
            assert got == pytest.approx(g[name], rel=1e-14, abs=1e-300), name

    @pytest.mark.parametrize("flag", ["c", "p"])
    def test_scalar_equals_one_elem_array(self, flag: str) -> None:
        f, k, t, r, sigma = 100.0, 105.0, 0.5, 0.05, 0.25
        sc = black76.higher_greeks(flag, F=f, K=k, T=t, r=r, sigma=sigma)
        ar = black76.higher_greeks(flag, F=np.array([f]), K=k, T=t, r=r, sigma=sigma)
        for name in self._ORDER:
            assert sc[name] == np.asarray(ar[name])[0], name

    def test_returns_ordered_dict_of_floats(self) -> None:
        g = black76.higher_greeks("c", F=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert tuple(g.keys()) == self._ORDER
        for v in g.values():
            assert isinstance(v, float)

    @pytest.mark.parametrize("flag", ["c", "p"])
    def test_parallel_matches_serial(self, flag: str) -> None:
        # Single-Greek gate (16384) via the tile trick, plus the bundle gate (4096).
        pk = np.linspace(20.0, 500.0, 384)
        reps = 48  # 18432 >= SINGLE_GREEK_PARALLEL_THRESHOLD
        bk = np.tile(pk, reps)
        f, t, r, sigma = 100.0, 0.5, 0.05, 0.20
        ref = black76.charm(flag, F=f, K=pk, T=t, r=r, sigma=sigma)
        got = black76.charm(flag, F=f, K=bk, T=t, r=r, sigma=sigma)
        np.testing.assert_array_equal(got, np.tile(ref, reps))
        n = 8192  # > GREEKS_PARALLEL_THRESHOLD
        strikes = np.linspace(80, 120, n)
        g = black76.higher_greeks(flag, F=f, K=strikes, T=t, r=r, sigma=sigma)
        np.testing.assert_allclose(
            g["ultima"], black76.ultima(F=f, K=strikes, T=t, r=r, sigma=sigma), rtol=1e-14
        )
