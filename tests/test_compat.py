"""Tests for the py_vollib compatibility shim."""

from __future__ import annotations

from typing import ClassVar

import pytest

from pyvolr import bs


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
        assert iv == pytest.approx(0.20, abs=1e-6)


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
    def test_price_with_dividend(self) -> None:
        from pyvolr.compat.py_vollib.black_scholes_merton import black_scholes_merton

        p = black_scholes_merton("c", 100.0, 100.0, 1.0, 0.05, 0.20, 0.03)
        ref = bs.price("c", S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.03)
        assert p == pytest.approx(ref)
