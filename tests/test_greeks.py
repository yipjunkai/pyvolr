"""Tests for `pyvolr.bs` Greeks."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from pyvolr import bs


def _central_fd(f, x: float, h: float) -> float:
    return (f(x + h) - f(x - h)) / (2.0 * h)


class TestGreeksVsFiniteDifference:
    """Each analytical Greek must match a central-difference numerical estimate."""

    PARAMS: ClassVar[dict[str, float]] = {
        "S": 100.0,
        "K": 105.0,
        "T": 0.5,
        "r": 0.05,
        "sigma": 0.25,
        "q": 0.02,
    }

    @pytest.mark.parametrize("flag", ["c", "p"])
    def test_delta(self, flag: str) -> None:
        analytical = bs.delta(flag, **self.PARAMS)
        fd = _central_fd(
            lambda s: bs.price(
                flag,
                S=s,
                K=self.PARAMS["K"],
                T=self.PARAMS["T"],
                r=self.PARAMS["r"],
                sigma=self.PARAMS["sigma"],
                q=self.PARAMS["q"],
            ),
            self.PARAMS["S"],
            0.01,
        )
        assert analytical == pytest.approx(fd, abs=1e-6)

    def test_gamma(self) -> None:
        analytical = bs.gamma(**{k: v for k, v in self.PARAMS.items()})
        h = 0.01
        f0 = bs.price(
            "c",
            S=self.PARAMS["S"],
            K=self.PARAMS["K"],
            T=self.PARAMS["T"],
            r=self.PARAMS["r"],
            sigma=self.PARAMS["sigma"],
            q=self.PARAMS["q"],
        )
        fp = bs.price(
            "c",
            S=self.PARAMS["S"] + h,
            K=self.PARAMS["K"],
            T=self.PARAMS["T"],
            r=self.PARAMS["r"],
            sigma=self.PARAMS["sigma"],
            q=self.PARAMS["q"],
        )
        fm = bs.price(
            "c",
            S=self.PARAMS["S"] - h,
            K=self.PARAMS["K"],
            T=self.PARAMS["T"],
            r=self.PARAMS["r"],
            sigma=self.PARAMS["sigma"],
            q=self.PARAMS["q"],
        )
        fd = (fp - 2 * f0 + fm) / (h * h)
        assert analytical == pytest.approx(fd, rel=1e-4)

    def test_vega(self) -> None:
        analytical = bs.vega(**self.PARAMS)
        fd = _central_fd(
            lambda v: bs.price(
                "c",
                S=self.PARAMS["S"],
                K=self.PARAMS["K"],
                T=self.PARAMS["T"],
                r=self.PARAMS["r"],
                sigma=v,
                q=self.PARAMS["q"],
            ),
            self.PARAMS["sigma"],
            1e-4,
        )
        assert analytical == pytest.approx(fd, abs=1e-6)

    @pytest.mark.parametrize("flag", ["c", "p"])
    def test_theta_is_negative_dprice_dT(self, flag: str) -> None:
        analytical = bs.theta(flag, **self.PARAMS)
        fd = _central_fd(
            lambda x: bs.price(
                flag,
                S=self.PARAMS["S"],
                K=self.PARAMS["K"],
                T=x,
                r=self.PARAMS["r"],
                sigma=self.PARAMS["sigma"],
                q=self.PARAMS["q"],
            ),
            self.PARAMS["T"],
            1e-5,
        )
        assert analytical == pytest.approx(-fd, abs=1e-5)

    @pytest.mark.parametrize("flag", ["c", "p"])
    def test_rho(self, flag: str) -> None:
        analytical = bs.rho(flag, **self.PARAMS)
        fd = _central_fd(
            lambda x: bs.price(
                flag,
                S=self.PARAMS["S"],
                K=self.PARAMS["K"],
                T=self.PARAMS["T"],
                r=x,
                sigma=self.PARAMS["sigma"],
                q=self.PARAMS["q"],
            ),
            self.PARAMS["r"],
            1e-5,
        )
        assert analytical == pytest.approx(fd, abs=1e-6)


class TestGreeksDict:
    def test_greeks_returns_dict(self) -> None:
        g = bs.greeks("c", S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert set(g.keys()) == {"delta", "gamma", "theta", "vega", "rho"}
        for v in g.values():
            assert isinstance(v, float)

    def test_greeks_dict_vectorized(self) -> None:
        strikes = np.linspace(80, 120, 5)
        g = bs.greeks("c", S=100, K=strikes, T=0.5, r=0.05, sigma=0.20)
        for v in g.values():
            assert isinstance(v, np.ndarray)
            assert v.shape == (5,)


class TestGreekProperties:
    def test_call_delta_in_range_with_dividend(self) -> None:
        # 0 < call_delta < exp(-qT) for any q >= 0
        strikes = np.linspace(50, 200, 30)
        d = bs.delta("c", S=100, K=strikes, T=1.0, r=0.05, sigma=0.20, q=0.02)
        cap = np.exp(-0.02 * 1.0)
        assert np.all(d > 0)
        assert np.all(d < cap + 1e-12)

    def test_gamma_nonnegative(self) -> None:
        strikes = np.linspace(50, 200, 30)
        g = bs.gamma(S=100, K=strikes, T=1.0, r=0.05, sigma=0.20)
        assert np.all(g >= 0)

    def test_vega_nonnegative(self) -> None:
        strikes = np.linspace(50, 200, 30)
        v = bs.vega(S=100, K=strikes, T=1.0, r=0.05, sigma=0.20)
        assert np.all(v >= 0)


class TestParallelDispatch:
    """Exercise the rayon-parallel branch of `bsm_greeks` (above N=4096).

    The parallel branch in `crates/core/src/lib.rs` collects 5-tuples in
    rayon, then unzips serially into the five output Vecs. The Rust-level
    parity test (`greeks::tests::all_matches_individual_at_grid`) covers
    `greeks::all` itself; this test covers the dispatch + unzip path at
    the FFI boundary, at a batch size above `GREEKS_PARALLEL_THRESHOLD`.
    """

    @pytest.mark.parametrize("flag", ["c", "p"])
    def test_greeks_above_threshold_matches_individual(self, flag: str) -> None:
        # Parametrized over flag because the put arm of `greeks::all` uses
        # different cdf calls (`cdf(-d1)` / `cdf(-d2)`) than the call arm,
        # and would silently regress if only the call path were tested.
        n = 8192  # > GREEKS_PARALLEL_THRESHOLD (4096)
        K = np.linspace(80, 120, n)
        S, T, r, sigma = 100.0, 0.5, 0.05, 0.20
        # Bundled, goes through the rayon parallel branch.
        g = bs.greeks(flag, S=S, K=K, T=T, r=r, sigma=sigma)
        # The per-Greek functions stay serial regardless of N. They use the
        # same `greeks::all`-compatible formulas, so should match the
        # bundled output to f64 (`greeks::tests::all_matches_individual_*`
        # proves the Rust-level parity at 1e-15).
        np.testing.assert_allclose(
            g["delta"], bs.delta(flag, S=S, K=K, T=T, r=r, sigma=sigma), rtol=1e-14
        )
        np.testing.assert_allclose(
            g["gamma"], bs.gamma(S=S, K=K, T=T, r=r, sigma=sigma), rtol=1e-14
        )
        np.testing.assert_allclose(g["vega"], bs.vega(S=S, K=K, T=T, r=r, sigma=sigma), rtol=1e-14)
        np.testing.assert_allclose(
            g["theta"], bs.theta(flag, S=S, K=K, T=T, r=r, sigma=sigma), rtol=1e-14
        )
        np.testing.assert_allclose(
            g["rho"], bs.rho(flag, S=S, K=K, T=T, r=r, sigma=sigma), rtol=1e-14
        )
