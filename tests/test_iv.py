"""Tests for `pyvolr.bs.implied_vol`."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from pyvolr import ImpliedVolError, ImpliedVolWarning, black76, bs


class TestRoundtrip:
    """Price -> IV -> price should reproduce the original volatility."""

    @pytest.mark.parametrize("flag", ["c", "p"])
    @pytest.mark.parametrize(
        ("s", "k", "t", "r", "q", "sigma"),
        [
            (100.0, 100.0, 1.0, 0.05, 0.0, 0.20),  # ATM
            (100.0, 110.0, 0.5, 0.03, 0.02, 0.30),  # slightly OTM call
            (100.0, 90.0, 0.5, 0.03, 0.02, 0.30),  # ITM call
            (100.0, 80.0, 0.05, 0.03, 0.01, 0.45),  # deep OTM short expiry
            (100.0, 200.0, 0.5, 0.05, 0.0, 0.30),  # very deep OTM call
            (100.0, 100.0, 2.0, 0.05, 0.0, 0.60),  # long-dated, high vol
        ],
    )
    def test_roundtrip(
        self, flag: str, s: float, k: float, t: float, r: float, q: float, sigma: float
    ) -> None:
        # LBR converges to ~1e-14 relative; the limit at moderate moneyness is
        # bsm price precision, not the solver. Tightened from 1e-6.
        p = bs.price(flag, S=s, K=k, T=t, r=r, sigma=sigma, q=q)
        iv = bs.implied_vol(p, flag, S=s, K=k, T=t, r=r, q=q)
        assert isinstance(iv, float)
        assert iv == pytest.approx(sigma, rel=1e-12)


class TestBoundsHandling:
    def test_price_below_intrinsic_returns_nan(self) -> None:
        iv = bs.implied_vol(-1.0, "c", S=100, K=100, T=1.0, r=0.05, on_error="ignore")
        assert math.isnan(iv)

    def test_zero_time_returns_nan(self) -> None:
        iv = bs.implied_vol(5.0, "c", S=100, K=100, T=0.0, r=0.05, on_error="ignore")
        assert math.isnan(iv)

    def test_price_above_upper_bound_returns_nan(self) -> None:
        # Call price cannot exceed S; pass a clearly absurd value.
        iv = bs.implied_vol(1e9, "c", S=100, K=100, T=1.0, r=0.05, on_error="ignore")
        assert math.isnan(iv)


class TestVectorized:
    def test_vector_of_prices(self) -> None:
        sigmas = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
        prices = bs.price("c", S=100, K=100, T=0.5, r=0.05, sigma=sigmas)
        ivs = bs.implied_vol(prices, "c", S=100, K=100, T=0.5, r=0.05)
        assert isinstance(ivs, np.ndarray)
        assert ivs.shape == sigmas.shape
        np.testing.assert_allclose(ivs, sigmas, rtol=1e-12)

    @pytest.mark.parametrize("flag", ["c", "p"])
    def test_parallel_branch_above_threshold(self, flag: str) -> None:
        # bsm_iv dispatches per-row work to rayon above N = PARALLEL_THRESHOLD
        # (1024 in `crates/core/src/lib.rs`). Below the threshold all calls go
        # through the serial path. This test exercises the parallel branch
        # explicitly so the dispatch / `py.detach` / collect path is covered
        # by CI rather than only by manual benches. Parametrized over flag
        # because the put arm uses different cdf calls (`cdf(-d1)` vs `cdf(d1)`)
        # and would silently regress if only the call path were tested.
        n = 2048  # comfortably above PARALLEL_THRESHOLD
        sigma = 0.20
        K = np.linspace(80, 120, n)
        prices = bs.price(flag, S=100, K=K, T=0.5, r=0.05, sigma=sigma)
        ivs = bs.implied_vol(prices, flag, S=100, K=K, T=0.5, r=0.05)
        assert isinstance(ivs, np.ndarray)
        assert ivs.shape == (n,)
        # LBR converges to ~1e-13; the parallel branch produces bit-identical
        # results to serial (same arithmetic, different scheduling).
        np.testing.assert_allclose(ivs, sigma, rtol=1e-12)


class TestOnError:
    """`on_error` governs the response to an unsolvable (NaN) solve.

    Unsolvable scalar used throughout: price -1.0 is below a call's intrinsic.
    The repo runs pytest under filterwarnings=["error"], so a stray warning on a
    path expected to be silent fails the test on its own.
    """

    def test_warn_is_the_default(self) -> None:
        with pytest.warns(ImpliedVolWarning, match="no-arbitrage"):
            iv = bs.implied_vol(-1.0, "c", S=100, K=100, T=1.0, r=0.05)
        assert math.isnan(iv)

    def test_raise(self) -> None:
        with pytest.raises(ImpliedVolError, match="no-arbitrage"):
            bs.implied_vol(-1.0, "c", S=100, K=100, T=1.0, r=0.05, on_error="raise")

    def test_raise_is_a_valueerror(self) -> None:
        # ImpliedVolError subclasses ValueError, so existing handlers keep working.
        with pytest.raises(ValueError, match="no-arbitrage"):
            bs.implied_vol(-1.0, "c", S=100, K=100, T=1.0, r=0.05, on_error="raise")

    def test_ignore_is_silent(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            iv = bs.implied_vol(-1.0, "c", S=100, K=100, T=1.0, r=0.05, on_error="ignore")
        assert math.isnan(iv)

    def test_valid_solve_does_not_warn(self) -> None:
        # A solvable input must stay silent even under the default.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            iv = bs.implied_vol(5.0, "c", S=100, K=100, T=0.25, r=0.05)
        assert not math.isnan(iv)

    def test_invalid_on_error_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="on_error must be"):
            bs.implied_vol(5.0, "c", S=100, K=100, T=0.25, r=0.05, on_error="boom")

    def test_batch_emits_one_summarizing_warning(self) -> None:
        # 2 valid (in [~2.47, 100]) + 2 invalid (below intrinsic / above max).
        prices = np.array([5.0, -1.0, 6.0, 1e9])
        with pytest.warns(ImpliedVolWarning, match="2 of 4") as record:
            ivs = bs.implied_vol(prices, "c", S=100, K=100, T=0.5, r=0.05)
        assert len(record) == 1
        assert isinstance(ivs, np.ndarray)
        assert np.isnan(ivs[1])
        assert np.isnan(ivs[3])
        assert not np.isnan(ivs[0])
        assert not np.isnan(ivs[2])

    def test_batch_raise_aborts_whole_call(self) -> None:
        prices = np.array([5.0, -1.0])
        with pytest.raises(ImpliedVolError, match="1 of 2"):
            bs.implied_vol(prices, "c", S=100, K=100, T=0.5, r=0.05, on_error="raise")

    def test_composes_with_return_as(self) -> None:
        # warn fires, and the NaN-containing result is still shaped by return_as.
        with pytest.warns(ImpliedVolWarning):
            d = bs.implied_vol(-1.0, "c", S=100, K=100, T=1.0, r=0.05, return_as="dict")
        assert set(d) == {"iv"}
        assert math.isnan(d["iv"])

    def test_black76_parity(self) -> None:
        with pytest.warns(ImpliedVolWarning):
            iv = black76.implied_vol(-1.0, "c", F=100, K=100, T=1.0, r=0.05)
        assert math.isnan(iv)
        with pytest.raises(ImpliedVolError):
            black76.implied_vol(-1.0, "c", F=100, K=100, T=1.0, r=0.05, on_error="raise")
