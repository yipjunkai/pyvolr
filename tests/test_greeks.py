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

    @pytest.mark.parametrize("flag", ["c", "p"])
    def test_single_greeks_parallel_match_serial(self, flag: str) -> None:
        # The *individual* Greek endpoints (bs.delta/gamma/vega/theta/rho) gate
        # at SINGLE_GREEK_PARALLEL_THRESHOLD (16384) — higher than the bundled
        # gate because one Greek does a fraction of the five-Greek per-row work,
        # so rayon's fixed overhead pays off at a larger N. Same tile trick as
        # `test_bsm.TestParallelDispatch`: a diverse pattern computed serially
        # (below the gate), tiled past it, must match the parallel result to the
        # bit (independent rows, identical kernel).
        pk = np.linspace(20.0, 500.0, 384)
        reps = 48  # 384 * 48 = 18432 >= SINGLE_GREEK_PARALLEL_THRESHOLD (16384)
        bk = np.tile(pk, reps)
        S, T, r, sigma = 100.0, 0.5, 0.05, 0.20

        def check(name: str, ref: np.ndarray, got: np.ndarray) -> None:
            np.testing.assert_array_equal(
                got, np.tile(ref, reps), err_msg=f"{name} parallel != serial (flag={flag})"
            )

        check(
            "delta",
            bs.delta(flag, S=S, K=pk, T=T, r=r, sigma=sigma),
            bs.delta(flag, S=S, K=bk, T=T, r=r, sigma=sigma),
        )
        check(
            "theta",
            bs.theta(flag, S=S, K=pk, T=T, r=r, sigma=sigma),
            bs.theta(flag, S=S, K=bk, T=T, r=r, sigma=sigma),
        )
        check(
            "rho",
            bs.rho(flag, S=S, K=pk, T=T, r=r, sigma=sigma),
            bs.rho(flag, S=S, K=bk, T=T, r=r, sigma=sigma),
        )
        check(
            "gamma",
            bs.gamma(S=S, K=pk, T=T, r=r, sigma=sigma),
            bs.gamma(S=S, K=bk, T=T, r=r, sigma=sigma),
        )
        check(
            "vega",
            bs.vega(S=S, K=pk, T=T, r=r, sigma=sigma),
            bs.vega(S=S, K=bk, T=T, r=r, sigma=sigma),
        )


# Higher-order Greek order + mpmath 50-digit goldens (tools/gen_goldens.py,
# section `greeks.rs::higher_greeks_match_mpmath_goldens`). Tuple order matches
# the Rust `higher_all` kernel and the `HigherGreeks` dict.
_HG_ORDER = ("vanna", "vomma", "charm", "speed", "zomma", "color", "veta", "ultima")
_HG_GOLDENS: dict[tuple[str, float, float, float, float, float, float], tuple[float, ...]] = {
    # (flag, S, K, T, r, q, sigma): the eight Greeks in _HG_ORDER
    ("c", 100.0, 100.0, 1.0, 0.05, 0.02, 0.20): (
        -0.09475289377504355,
        2.3688223443760887,
        -0.03563942396482651,
        -0.00042638802198769604,
        -0.09356848260285552,
        0.010446506538698554,
        -17.00814443262032,
        -73.28544127913524,
    ),
    ("c", 100.0, 120.0, 0.5, 0.03, 0.01, 0.30): (
        0.9469621924180325,
        47.29178329769534,
        -0.31086449941248695,
        0.00033966826829557055,
        -0.017078787798722746,
        0.004298479873034124,
        -37.298259187916486,
        -381.6043464146372,
    ),
    ("p", 100.0, 90.0, 0.5, 0.05, 0.02, 0.25): (
        -0.6963059072708816,
        37.87500714232858,
        0.11984125058347952,
        -0.0008894562783342539,
        -0.03618230478964687,
        0.011547739256155704,
        -27.116769994498952,
        -395.8402301731359,
    ),
}


class TestHigherGreeksGoldens:
    """Pin the higher Greeks through the FFI against the mpmath goldens."""

    @pytest.mark.parametrize("key", list(_HG_GOLDENS))
    def test_bundle_matches_goldens(
        self, key: tuple[str, float, float, float, float, float, float]
    ) -> None:
        flag, s, k, t, r, q, sigma = key
        g = bs.higher_greeks(flag, S=s, K=k, T=t, r=r, sigma=sigma, q=q)
        for name, expected in zip(_HG_ORDER, _HG_GOLDENS[key], strict=True):
            assert g[name] == pytest.approx(expected, rel=1e-12), name

    def test_put_charm_golden(self) -> None:
        # Only charm depends on the flag; check the put charm at the ATM point.
        g = bs.higher_greeks("p", S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.02)
        assert g["charm"] == pytest.approx(-0.05524339743096161, rel=1e-12)


class TestHigherGreeksConsistency:
    S, K, T, r, q, sigma = 100.0, 105.0, 0.5, 0.05, 0.02, 0.25

    def _individual(self, name: str, flag: str, s: object):
        # charm is the one flag-dependent higher Greek; call it explicitly with
        # keyword args (a ``**dict`` splat can't be matched to the overloads
        # under pyright-strict). The rest go through getattr — untyped, but the
        # tests-only pyright env allows it.
        if name == "charm":
            return bs.charm(flag, S=s, K=self.K, T=self.T, r=self.r, sigma=self.sigma, q=self.q)
        return getattr(bs, name)(S=s, K=self.K, T=self.T, r=self.r, sigma=self.sigma, q=self.q)

    @pytest.mark.parametrize("flag", ["c", "p"])
    def test_individual_equals_bundle(self, flag: str) -> None:
        g = bs.higher_greeks(
            flag, S=self.S, K=self.K, T=self.T, r=self.r, sigma=self.sigma, q=self.q
        )
        for name in _HG_ORDER:
            # Bundle (`higher_all`) and the standalone fns share bit-identical
            # expressions in Rust; allow a last-ULP slack to stay robust.
            got = self._individual(name, flag, self.S)
            assert got == pytest.approx(g[name], rel=1e-14, abs=1e-300), name

    @pytest.mark.parametrize("flag", ["c", "p"])
    def test_scalar_equals_one_elem_array(self, flag: str) -> None:
        # All-scalar inputs hit the dedicated scalar FFI; a 1-elem array hits the
        # array endpoint. Same kernel, must be bit-identical.
        for name in _HG_ORDER:
            sc = self._individual(name, flag, self.S)
            ar = self._individual(name, flag, np.array([self.S]))
            assert sc == np.asarray(ar)[0], name

    @pytest.mark.parametrize("flag", ["c", "p"])
    def test_bundle_scalar_equals_one_elem_array(self, flag: str) -> None:
        sc = bs.higher_greeks(
            flag, S=self.S, K=self.K, T=self.T, r=self.r, sigma=self.sigma, q=self.q
        )
        ar = bs.higher_greeks(
            flag, S=np.array([self.S]), K=self.K, T=self.T, r=self.r, sigma=self.sigma, q=self.q
        )
        for name in _HG_ORDER:
            assert sc[name] == np.asarray(ar[name])[0], name


class TestHigherGreeksDict:
    def test_returns_ordered_dict_of_floats(self) -> None:
        g = bs.higher_greeks("c", S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert tuple(g.keys()) == _HG_ORDER
        for v in g.values():
            assert isinstance(v, float)

    def test_vectorized(self) -> None:
        strikes = np.linspace(80, 120, 5)
        g = bs.higher_greeks("c", S=100, K=strikes, T=0.5, r=0.05, sigma=0.20)
        for v in g.values():
            assert isinstance(v, np.ndarray)
            assert v.shape == (5,)

    def test_dataframe(self) -> None:
        pytest.importorskip("pandas")
        df = bs.higher_greeks(
            "c", S=100, K=np.linspace(80, 120, 4), T=0.5, r=0.05, sigma=0.20, return_as="dataframe"
        )
        assert list(df.columns) == list(_HG_ORDER)
        assert len(df) == 4


class TestHigherGreeksDegenerate:
    """t <= 0 and sigma <= 0 return 0 for all eight (documented policy)."""

    @pytest.mark.parametrize("flag", ["c", "p"])
    @pytest.mark.parametrize(("t", "sigma"), [(0.0, 0.20), (1.0, 0.0)])
    def test_all_zero(self, flag: str, t: float, sigma: float) -> None:
        g = bs.higher_greeks(flag, S=100.0, K=100.0, T=t, r=0.05, sigma=sigma)
        for name in _HG_ORDER:
            assert g[name] == 0.0, name
        assert bs.charm(flag, S=100.0, K=100.0, T=t, r=0.05, sigma=sigma) == 0.0
        assert bs.vanna(S=100.0, K=100.0, T=t, r=0.05, sigma=sigma) == 0.0


class TestHigherParallelDispatch:
    """Higher Greeks cross the same rayon gates as the first-order ones."""

    @pytest.mark.parametrize("flag", ["c", "p"])
    def test_single_higher_greek_parallel_matches_serial(self, flag: str) -> None:
        # Tile trick (see TestParallelDispatch): a diverse serial pattern (below
        # SINGLE_GREEK_PARALLEL_THRESHOLD=16384) tiled past it must match the
        # parallel result bit-for-bit — independent rows, identical kernel.
        pk = np.linspace(20.0, 500.0, 384)
        reps = 48  # 384 * 48 = 18432 >= 16384
        bk = np.tile(pk, reps)
        s, t, r, sigma = 100.0, 0.5, 0.05, 0.20
        for name in ("vanna", "charm", "speed", "veta", "ultima"):
            if name == "charm":
                ref = bs.charm(flag, S=s, K=pk, T=t, r=r, sigma=sigma)
                got = bs.charm(flag, S=s, K=bk, T=t, r=r, sigma=sigma)
            else:
                ref = getattr(bs, name)(S=s, K=pk, T=t, r=r, sigma=sigma)
                got = getattr(bs, name)(S=s, K=bk, T=t, r=r, sigma=sigma)
            np.testing.assert_array_equal(
                got, np.tile(ref, reps), err_msg=f"{name} parallel != serial (flag={flag})"
            )

    @pytest.mark.parametrize("flag", ["c", "p"])
    def test_bundle_parallel_matches_individual(self, flag: str) -> None:
        n = 8192  # > GREEKS_PARALLEL_THRESHOLD (4096)
        strikes = np.linspace(80, 120, n)
        s, t, r, sigma = 100.0, 0.5, 0.05, 0.20
        g = bs.higher_greeks(flag, S=s, K=strikes, T=t, r=r, sigma=sigma)
        np.testing.assert_allclose(
            g["vanna"], bs.vanna(S=s, K=strikes, T=t, r=r, sigma=sigma), rtol=1e-14
        )
        np.testing.assert_allclose(
            g["charm"], bs.charm(flag, S=s, K=strikes, T=t, r=r, sigma=sigma), rtol=1e-14
        )
        np.testing.assert_allclose(
            g["ultima"], bs.ultima(S=s, K=strikes, T=t, r=r, sigma=sigma), rtol=1e-14
        )
