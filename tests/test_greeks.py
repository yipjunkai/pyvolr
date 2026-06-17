"""Tests for `pyvolr.bs` Greeks."""

from __future__ import annotations

from typing import ClassVar, TypedDict

import numpy as np
import pytest

from pyvolr import bs


class _GreekParams(TypedDict):
    """Precise keys so ``**PARAMS`` splat matches the typed function overloads.

    A plain ``dict[str, float]`` lets the type checker assume an arbitrary key
    (e.g. ``return_as``) might be present, which no longer unifies with the
    keyword-only ``return_as`` parameter.
    """

    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float


def _central_fd(f, x: float, h: float) -> float:
    return (f(x + h) - f(x - h)) / (2.0 * h)


class TestGreeksVsFiniteDifference:
    """Each analytical Greek must match a central-difference numerical estimate."""

    PARAMS: ClassVar[_GreekParams] = {
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
        analytical = bs.gamma(**self.PARAMS)
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


class TestPrecisionCorners:
    """Python-level guards for the precision corners the audit fixed.

    The Rust-side parity tests (`greeks::tests::put_delta_deep_otm_retains_precision`,
    `call_and_vega_matches_separate`) catch regressions in the scalar Rust
    functions. These mirror the same corners through the PyO3 FFI boundary
    so a regression in the macro dispatch / numpy roundtrip also fires.

    The differential test (`tests/test_differential.py`) does NOT cover
    these corners: at the deep-OTM saturation cliff, py_vollib's own
    formula underflows to zero and pyvolr's erfcx-tail formula returns
    ~1e-61. Both are below the `abs=1e-10` differential tolerance, so
    the bug-fix slipping back through the macro layer would be silent.
    """

    def test_put_delta_deep_otm_retains_precision(self) -> None:
        # S=1000, K=100, T=0.5, sigma=20% → d1 ≈ 16.5 → cdf(d1) saturates to
        # 1.0 in f64. The old `cdf(d1) - 1.0` form returned exactly 0;
        # the `-cdf(-d1)` form (commit 30d5d1f) returns ~-1.1e-61 via
        # the erfcx tail. This test guards the fix at the Python API.
        d = bs.delta("p", S=1000, K=100, T=0.5, r=0.05, sigma=0.20)
        assert isinstance(d, float)
        assert d < 0.0, f"put delta lost sign at deep OTM (returned {d:e})"
        assert 0.0 < abs(d) < 1e-50, f"expected ~1e-61, got {d:e}"

    def test_bundled_greeks_put_delta_deep_otm_retains_precision(self) -> None:
        # Same corner via the bundled `bs.greeks()` path, which routes
        # through `greeks::all` (different code path than `bs.delta`'s
        # `greeks::delta`). Catches regressions in the put-arm of the
        # all-in-one kernel.
        g = bs.greeks("p", S=1000, K=100, T=0.5, r=0.05, sigma=0.20)
        d = g["delta"]
        assert isinstance(d, float)
        assert d < 0.0, f"put delta lost sign at deep OTM via greeks() (got {d:e})"
        assert 0.0 < abs(d) < 1e-50


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
