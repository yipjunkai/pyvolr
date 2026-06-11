"""Tests for the py_vollib compatibility shim."""

from __future__ import annotations

from typing import ClassVar

import pytest

from pyvolr import black76, bs


class TestBlackScholesShim:
    def test_price_matches_modern_api(self) -> None:
        from pyvolr.compat.py_vollib.black_scholes import black_scholes

        p = black_scholes("c", 100.0, 100.0, 1.0, 0.05, 0.20)
        ref = bs.price("c", S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)
        assert p == pytest.approx(ref)
        assert isinstance(p, float)

    def test_iv_signature_flag_last(self) -> None:
        # py_vollib's order: (price, S, K, t, r, flag) -- flag at the end.
        from pyvolr.compat.py_vollib.black_scholes.implied_volatility import implied_volatility

        p = bs.price("c", S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        iv = implied_volatility(p, 100.0, 100.0, 1.0, 0.05, "c")
        # Compat shim routes through iv::solve (LBR), so the same ~1e-13
        # precision applies as for the modern pyvolr.bs.implied_vol API.
        assert iv == pytest.approx(0.20, rel=1e-12)


class TestBlackScholesGreeksShim:
    """py_vollib's Greeks use the per-1%-vol / per-1%-rate / per-day conventions."""

    P: ClassVar[dict[str, float | str]] = {
        "flag": "c",
        "S": 100.0,
        "K": 100.0,
        "t": 1.0,
        "r": 0.05,
        "sigma": 0.20,
    }

    def test_delta_unchanged(self) -> None:
        from pyvolr.compat.py_vollib.black_scholes.greeks.analytical import delta

        shim = delta(**self.P)
        ref = bs.delta(
            self.P["flag"],
            S=self.P["S"],
            K=self.P["K"],
            T=self.P["t"],
            r=self.P["r"],
            sigma=self.P["sigma"],
        )
        assert shim == pytest.approx(ref)

    def test_vega_per_1pct_vol(self) -> None:
        from pyvolr.compat.py_vollib.black_scholes.greeks.analytical import vega

        shim = vega(**self.P)
        modern_per_unit = bs.vega(
            S=self.P["S"], K=self.P["K"], T=self.P["t"], r=self.P["r"], sigma=self.P["sigma"]
        )
        assert shim == pytest.approx(modern_per_unit / 100.0)

    def test_theta_per_day(self) -> None:
        from pyvolr.compat.py_vollib.black_scholes.greeks.analytical import theta

        shim = theta(**self.P)
        modern_per_year = bs.theta(
            self.P["flag"],
            S=self.P["S"],
            K=self.P["K"],
            T=self.P["t"],
            r=self.P["r"],
            sigma=self.P["sigma"],
        )
        assert shim == pytest.approx(modern_per_year / 365.0)

    def test_rho_per_1pct_rate(self) -> None:
        from pyvolr.compat.py_vollib.black_scholes.greeks.analytical import rho

        shim = rho(**self.P)
        modern_per_unit = bs.rho(
            self.P["flag"],
            S=self.P["S"],
            K=self.P["K"],
            T=self.P["t"],
            r=self.P["r"],
            sigma=self.P["sigma"],
        )
        assert shim == pytest.approx(modern_per_unit / 100.0)


class TestMertonShim:
    """The Merton (dividend-yield) surface — price, IV, and all five Greeks.

    Previously only `price` was covered; a wrong /365 or /100 convention factor
    or a swapped argument in the Greeks/IV shims would have shipped a silently
    wrong number on an advertised drop-in surface.
    """

    P: ClassVar[dict[str, float | str]] = {
        "flag": "c",
        "S": 100.0,
        "K": 105.0,
        "t": 0.5,
        "r": 0.05,
        "sigma": 0.25,
        "q": 0.03,
    }

    def test_price_with_dividend(self) -> None:
        from pyvolr.compat.py_vollib.black_scholes_merton import black_scholes_merton

        p = black_scholes_merton("c", 100.0, 100.0, 1.0, 0.05, 0.20, 0.03)
        ref = bs.price("c", S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.03)
        assert p == pytest.approx(ref)

    def test_iv_roundtrip_q_before_flag(self) -> None:
        # py_vollib.black_scholes_merton IV signature: (price, S, K, t, r, q, flag)
        from pyvolr.compat.py_vollib.black_scholes_merton.implied_volatility import (
            implied_volatility,
        )

        price = bs.price("c", S=100.0, K=105.0, T=0.5, r=0.05, sigma=0.25, q=0.03)
        iv = implied_volatility(price, 100.0, 105.0, 0.5, 0.05, 0.03, "c")
        assert iv == pytest.approx(0.25, rel=1e-12)

    def test_delta(self) -> None:
        from pyvolr.compat.py_vollib.black_scholes_merton.greeks.analytical import delta

        shim = delta(**self.P)
        ref = bs.delta(
            self.P["flag"],
            S=self.P["S"],
            K=self.P["K"],
            T=self.P["t"],
            r=self.P["r"],
            sigma=self.P["sigma"],
            q=self.P["q"],
        )
        assert shim == pytest.approx(ref)

    def test_gamma_independent_of_flag(self) -> None:
        from pyvolr.compat.py_vollib.black_scholes_merton.greeks.analytical import gamma

        shim = gamma(**self.P)
        ref = bs.gamma(
            S=self.P["S"],
            K=self.P["K"],
            T=self.P["t"],
            r=self.P["r"],
            sigma=self.P["sigma"],
            q=self.P["q"],
        )
        assert shim == pytest.approx(ref)
        assert gamma(**{**self.P, "flag": "p"}) == pytest.approx(shim)

    def test_vega_per_1pct_vol(self) -> None:
        from pyvolr.compat.py_vollib.black_scholes_merton.greeks.analytical import vega

        shim = vega(**self.P)
        modern_per_unit = bs.vega(
            S=self.P["S"],
            K=self.P["K"],
            T=self.P["t"],
            r=self.P["r"],
            sigma=self.P["sigma"],
            q=self.P["q"],
        )
        assert shim == pytest.approx(modern_per_unit / 100.0)

    def test_theta_per_day(self) -> None:
        from pyvolr.compat.py_vollib.black_scholes_merton.greeks.analytical import theta

        shim = theta(**self.P)
        modern_per_year = bs.theta(
            self.P["flag"],
            S=self.P["S"],
            K=self.P["K"],
            T=self.P["t"],
            r=self.P["r"],
            sigma=self.P["sigma"],
            q=self.P["q"],
        )
        assert shim == pytest.approx(modern_per_year / 365.0)

    def test_rho_per_1pct_rate(self) -> None:
        from pyvolr.compat.py_vollib.black_scholes_merton.greeks.analytical import rho

        shim = rho(**self.P)
        modern_per_unit = bs.rho(
            self.P["flag"],
            S=self.P["S"],
            K=self.P["K"],
            T=self.P["t"],
            r=self.P["r"],
            sigma=self.P["sigma"],
            q=self.P["q"],
        )
        assert shim == pytest.approx(modern_per_unit / 100.0)


class TestCompatErrorContract:
    """The shim mirrors py_vollib's ONE real error contract (IV bound
    exceptions) and deliberately diverges from its leaked artifacts elsewhere.
    """

    def test_exceptions_importable_and_hierarchy(self) -> None:
        import sys

        from pyvolr.compat.py_vollib.exceptions import (
            AboveMaximumException,
            BelowIntrinsicException,
            VolatilityValueException,
        )

        assert issubclass(BelowIntrinsicException, VolatilityValueException)
        assert issubclass(AboveMaximumException, VolatilityValueException)
        # Messages and ±DBL_MAX sentinels match py_lets_be_rational exactly.
        assert str(BelowIntrinsicException()) == "The volatility is below the intrinsic value."
        assert str(AboveMaximumException()) == "The volatility is above the maximum value."
        assert BelowIntrinsicException().value == -sys.float_info.max
        assert AboveMaximumException().value == sys.float_info.max

    def test_bs_iv_bound_violations_raise(self) -> None:
        from pyvolr.compat.py_vollib.black_scholes.implied_volatility import implied_volatility
        from pyvolr.compat.py_vollib.exceptions import (
            AboveMaximumException,
            BelowIntrinsicException,
        )

        # Deep-ITM call priced below intrinsic; ATM call priced above the max.
        with pytest.raises(BelowIntrinsicException):
            implied_volatility(0.01, 100.0, 50.0, 1.0, 0.05, "c")
        with pytest.raises(AboveMaximumException):
            implied_volatility(200.0, 100.0, 100.0, 1.0, 0.05, "c")
        # Put side too.
        with pytest.raises(BelowIntrinsicException):
            implied_volatility(0.01, 50.0, 100.0, 1.0, 0.05, "p")

    def test_merton_iv_bound_violations_raise(self) -> None:
        from pyvolr.compat.py_vollib.black_scholes_merton.implied_volatility import (
            implied_volatility,
        )
        from pyvolr.compat.py_vollib.exceptions import (
            AboveMaximumException,
            BelowIntrinsicException,
        )

        with pytest.raises(BelowIntrinsicException):
            implied_volatility(0.001, 100.0, 50.0, 1.0, 0.05, 0.03, "c")
        with pytest.raises(AboveMaximumException):
            implied_volatility(200.0, 100.0, 100.0, 1.0, 0.05, 0.03, "c")

    def test_black76_iv_bound_violations_raise(self) -> None:
        from pyvolr.compat.py_vollib.black.implied_volatility import implied_volatility
        from pyvolr.compat.py_vollib.exceptions import (
            AboveMaximumException,
            BelowIntrinsicException,
        )

        # Black-76 signature: (price, F, K, r, t, flag).
        with pytest.raises(BelowIntrinsicException):
            implied_volatility(0.001, 100.0, 50.0, 0.05, 1.0, "c")
        with pytest.raises(AboveMaximumException):
            implied_volatility(200.0, 100.0, 100.0, 0.05, 1.0, "c")

    def test_degenerate_inputs_are_graceful_not_artifacts(self) -> None:
        # Deliberate divergence: py_vollib leaks ZeroDivisionError / ValueError on
        # these; pyvolr returns clean values (see the compat package docstring).
        import math

        from pyvolr.compat.py_vollib.black_scholes import black_scholes
        from pyvolr.compat.py_vollib.black_scholes.implied_volatility import implied_volatility

        # K = 0 (py_vollib ZeroDivisionError): zero-strike call = discounted forward.
        assert black_scholes("c", 100.0, 0.0, 1.0, 0.05, 0.20) == pytest.approx(100.0)
        # t = 0 expired (py_vollib leaks NaN): clean intrinsic.
        assert black_scholes("c", 110.0, 100.0, 0.0, 0.05, 0.20) == pytest.approx(10.0)
        # IV at a degenerate-but-in-bounds input returns NaN, not an exception.
        assert math.isnan(implied_volatility(10.0, 110.0, 100.0, 0.0, 0.05, "c"))

    def test_invalid_flag_is_clean_valueerror_not_keyerror(self) -> None:
        from pyvolr.compat.py_vollib.black_scholes import black_scholes

        # py_vollib raises a bare KeyError; pyvolr raises a ValueError with a message.
        with pytest.raises(ValueError, match="flag"):
            black_scholes("x", 100.0, 100.0, 1.0, 0.05, 0.20)


class TestBlack76Shim:
    def test_price_matches_modern_api(self) -> None:
        from pyvolr.compat.py_vollib.black import black

        # py_vollib.black signature: black(flag, F, K, t, r, sigma)
        p = black("c", 100.0, 100.0, 0.5, 0.02, 0.20)
        ref = black76.price("c", F=100.0, K=100.0, T=0.5, r=0.02, sigma=0.20)
        assert p == pytest.approx(ref)
        assert isinstance(p, float)

    def test_iv_signature_flag_last_r_before_t(self) -> None:
        # py_vollib.black IV signature: (price, F, K, r, t, flag) — note
        # r comes BEFORE t here, opposite of black_scholes.implied_volatility.
        from pyvolr.compat.py_vollib.black.implied_volatility import implied_volatility

        p = black76.price("c", F=100, K=100, T=1.0, r=0.05, sigma=0.20)
        iv = implied_volatility(p, 100.0, 100.0, 0.05, 1.0, "c")
        assert iv == pytest.approx(0.20, rel=1e-12)


class TestBlack76GreeksShim:
    """py_vollib.black uses per-1%-vol / per-1%-rate / per-day, like black_scholes."""

    P: ClassVar[dict[str, float | str]] = {
        "flag": "c",
        "F": 100.0,
        "K": 100.0,
        "t": 1.0,
        "r": 0.05,
        "sigma": 0.20,
    }

    def test_delta(self) -> None:
        from pyvolr.compat.py_vollib.black.greeks.analytical import delta

        shim = delta(**self.P)
        ref = black76.delta(
            self.P["flag"],
            F=self.P["F"],
            K=self.P["K"],
            T=self.P["t"],
            r=self.P["r"],
            sigma=self.P["sigma"],
        )
        assert shim == pytest.approx(ref)

    def test_gamma_independent_of_flag(self) -> None:
        from pyvolr.compat.py_vollib.black.greeks.analytical import gamma

        # py_vollib.black.gamma takes a flag argument but it's ignored.
        shim = gamma(**self.P)
        ref = black76.gamma(
            F=self.P["F"],
            K=self.P["K"],
            T=self.P["t"],
            r=self.P["r"],
            sigma=self.P["sigma"],
        )
        assert shim == pytest.approx(ref)

    def test_vega_per_1pct_vol(self) -> None:
        from pyvolr.compat.py_vollib.black.greeks.analytical import vega

        shim = vega(**self.P)
        modern_per_unit = black76.vega(
            F=self.P["F"], K=self.P["K"], T=self.P["t"], r=self.P["r"], sigma=self.P["sigma"]
        )
        assert shim == pytest.approx(modern_per_unit / 100.0)

    def test_theta_per_day(self) -> None:
        from pyvolr.compat.py_vollib.black.greeks.analytical import theta

        shim = theta(**self.P)
        modern_per_year = black76.theta(
            self.P["flag"],
            F=self.P["F"],
            K=self.P["K"],
            T=self.P["t"],
            r=self.P["r"],
            sigma=self.P["sigma"],
        )
        assert shim == pytest.approx(modern_per_year / 365.0)

    def test_rho_per_1pct_rate(self) -> None:
        from pyvolr.compat.py_vollib.black.greeks.analytical import rho

        shim = rho(**self.P)
        modern_per_unit = black76.rho(
            self.P["flag"],
            F=self.P["F"],
            K=self.P["K"],
            T=self.P["t"],
            r=self.P["r"],
            sigma=self.P["sigma"],
        )
        assert shim == pytest.approx(modern_per_unit / 100.0)
