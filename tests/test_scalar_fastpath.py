"""Scalar fast-path tests: dispatch routing, parity, and error behavior.

The fast path must be observably identical to the array path for eligible
scalar inputs — same result bits, same Python types, same exceptions, same
warnings at the same depth — with only the latency differing. Failures here
mean the dispatch guard and the array pipeline have drifted.

Also covers the strict numeric-flag validation that landed alongside the
fast path (values outside {-1, 1} used to wrap silently via ``astype(int8)``).
"""

from __future__ import annotations

import math
import warnings
from decimal import Decimal

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pyvolr import _core, black76, bs
from pyvolr.exceptions import ImpliedVolError, ImpliedVolWarning


def same_bits(a: float, b: float) -> bool:
    """Bit-for-bit f64 equality (distinguishes ±0.0; treats any NaN as NaN)."""
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return np.float64(a).tobytes() == np.float64(b).tobytes()


def elem(result: object, i: int = 0) -> float:
    """Index an array-path result, proving to the type checker it's an ndarray."""
    assert isinstance(result, np.ndarray)
    return float(result[i])


# Strategy bounds mirror the domain the solver documents as well-posed; the
# non-finite/overflow cells get explicit deterministic coverage below.
spots = st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False)
expiries = st.floats(min_value=1e-6, max_value=30.0, allow_nan=False, allow_infinity=False)
rates = st.floats(min_value=-0.2, max_value=0.3, allow_nan=False, allow_infinity=False)
sigmas = st.floats(min_value=1e-4, max_value=5.0, allow_nan=False, allow_infinity=False)
flags = st.sampled_from(["c", "p", "C", "P"])

GREEK_KEYS = ("delta", "gamma", "theta", "vega", "rho")


class TestBsmScalarArrayParity:
    """Fast path == array path, bit for bit (array path forced via one 1-elem array)."""

    @given(flag=flags, s=spots, k=spots, t=expiries, r=rates, q=rates, sigma=sigmas)
    def test_price(
        self, flag: str, s: float, k: float, t: float, r: float, q: float, sigma: float
    ) -> None:
        fast = bs.price(flag, s, k, t, r, sigma, q)
        slow = bs.price(flag, np.array([s]), k, t, r, sigma, q)
        assert type(fast) is float
        assert same_bits(fast, elem(slow))

    @given(flag=flags, s=spots, k=spots, t=expiries, r=rates, q=rates, sigma=sigmas)
    def test_delta(
        self, flag: str, s: float, k: float, t: float, r: float, q: float, sigma: float
    ) -> None:
        fast = bs.delta(flag, s, k, t, r, sigma, q)
        slow = bs.delta(flag, np.array([s]), k, t, r, sigma, q)
        assert same_bits(fast, elem(slow))

    @given(s=spots, k=spots, t=expiries, r=rates, q=rates, sigma=sigmas)
    def test_gamma(self, s: float, k: float, t: float, r: float, q: float, sigma: float) -> None:
        fast = bs.gamma(s, k, t, r, sigma, q)
        slow = bs.gamma(np.array([s]), k, t, r, sigma, q)
        assert same_bits(fast, elem(slow))

    @given(s=spots, k=spots, t=expiries, r=rates, q=rates, sigma=sigmas)
    def test_vega(self, s: float, k: float, t: float, r: float, q: float, sigma: float) -> None:
        fast = bs.vega(s, k, t, r, sigma, q)
        slow = bs.vega(np.array([s]), k, t, r, sigma, q)
        assert same_bits(fast, elem(slow))

    @given(flag=flags, s=spots, k=spots, t=expiries, r=rates, q=rates, sigma=sigmas)
    def test_theta(
        self, flag: str, s: float, k: float, t: float, r: float, q: float, sigma: float
    ) -> None:
        fast = bs.theta(flag, s, k, t, r, sigma, q)
        slow = bs.theta(flag, np.array([s]), k, t, r, sigma, q)
        assert same_bits(fast, elem(slow))

    @given(flag=flags, s=spots, k=spots, t=expiries, r=rates, q=rates, sigma=sigmas)
    def test_rho(
        self, flag: str, s: float, k: float, t: float, r: float, q: float, sigma: float
    ) -> None:
        fast = bs.rho(flag, s, k, t, r, sigma, q)
        slow = bs.rho(flag, np.array([s]), k, t, r, sigma, q)
        assert same_bits(fast, elem(slow))

    @given(flag=flags, s=spots, k=spots, t=expiries, r=rates, q=rates, sigma=sigmas)
    def test_greeks(
        self, flag: str, s: float, k: float, t: float, r: float, q: float, sigma: float
    ) -> None:
        fast = bs.greeks(flag, s, k, t, r, sigma, q)
        slow = bs.greeks(flag, np.array([s]), k, t, r, sigma, q)
        assert tuple(fast.keys()) == tuple(slow.keys()) == GREEK_KEYS
        for key in GREEK_KEYS:
            assert type(fast[key]) is float
            assert same_bits(fast[key], elem(slow[key]))

    @given(flag=flags, s=spots, k=spots, t=expiries, r=rates, q=rates, sigma=sigmas)
    def test_implied_vol(
        self, flag: str, s: float, k: float, t: float, r: float, q: float, sigma: float
    ) -> None:
        # Solve against a self-generated price: always inside the no-arb
        # bounds, so the default on_error="warn" stays silent on both paths.
        target = bs.price(flag, s, k, t, r, sigma, q)
        fast = bs.implied_vol(target, flag, s, k, t, r, q, on_error="ignore")
        slow = bs.implied_vol(target, flag, np.array([s]), k, t, r, q, on_error="ignore")
        assert type(fast) is float
        assert same_bits(fast, elem(slow))


class TestBlack76ScalarArrayParity:
    @given(flag=flags, f=spots, k=spots, t=expiries, r=rates, sigma=sigmas)
    def test_price(self, flag: str, f: float, k: float, t: float, r: float, sigma: float) -> None:
        fast = black76.price(flag, f, k, t, r, sigma)
        slow = black76.price(flag, np.array([f]), k, t, r, sigma)
        assert type(fast) is float
        assert same_bits(fast, elem(slow))

    @given(flag=flags, f=spots, k=spots, t=expiries, r=rates, sigma=sigmas)
    def test_delta(self, flag: str, f: float, k: float, t: float, r: float, sigma: float) -> None:
        fast = black76.delta(flag, f, k, t, r, sigma)
        slow = black76.delta(flag, np.array([f]), k, t, r, sigma)
        assert same_bits(fast, elem(slow))

    @given(f=spots, k=spots, t=expiries, r=rates, sigma=sigmas)
    def test_gamma(self, f: float, k: float, t: float, r: float, sigma: float) -> None:
        fast = black76.gamma(f, k, t, r, sigma)
        slow = black76.gamma(np.array([f]), k, t, r, sigma)
        assert same_bits(fast, elem(slow))

    @given(f=spots, k=spots, t=expiries, r=rates, sigma=sigmas)
    def test_vega(self, f: float, k: float, t: float, r: float, sigma: float) -> None:
        fast = black76.vega(f, k, t, r, sigma)
        slow = black76.vega(np.array([f]), k, t, r, sigma)
        assert same_bits(fast, elem(slow))

    @given(flag=flags, f=spots, k=spots, t=expiries, r=rates, sigma=sigmas)
    def test_theta(self, flag: str, f: float, k: float, t: float, r: float, sigma: float) -> None:
        fast = black76.theta(flag, f, k, t, r, sigma)
        slow = black76.theta(flag, np.array([f]), k, t, r, sigma)
        assert same_bits(fast, elem(slow))

    @given(flag=flags, f=spots, k=spots, t=expiries, r=rates, sigma=sigmas)
    def test_rho(self, flag: str, f: float, k: float, t: float, r: float, sigma: float) -> None:
        fast = black76.rho(flag, f, k, t, r, sigma)
        slow = black76.rho(flag, np.array([f]), k, t, r, sigma)
        assert same_bits(fast, elem(slow))

    @given(flag=flags, f=spots, k=spots, t=expiries, r=rates, sigma=sigmas)
    def test_greeks(self, flag: str, f: float, k: float, t: float, r: float, sigma: float) -> None:
        fast = black76.greeks(flag, f, k, t, r, sigma)
        slow = black76.greeks(flag, np.array([f]), k, t, r, sigma)
        assert tuple(fast.keys()) == tuple(slow.keys()) == GREEK_KEYS
        for key in GREEK_KEYS:
            assert same_bits(fast[key], elem(slow[key]))

    @given(flag=flags, f=spots, k=spots, t=expiries, r=rates, sigma=sigmas)
    def test_implied_vol(
        self, flag: str, f: float, k: float, t: float, r: float, sigma: float
    ) -> None:
        target = black76.price(flag, f, k, t, r, sigma)
        fast = black76.implied_vol(target, flag, f, k, t, r, on_error="ignore")
        slow = black76.implied_vol(target, flag, np.array([f]), k, t, r, on_error="ignore")
        assert same_bits(fast, elem(slow))


class TestNonFiniteParity:
    """Deterministic edge cells Hypothesis' strategies deliberately exclude."""

    EDGE = (0.0, -1.0, 1e-308, 1e308, math.inf, -math.inf, math.nan)

    @pytest.mark.parametrize("s", EDGE)
    @pytest.mark.parametrize("flag", ["c", "p"])
    def test_price_edge_spot(self, flag: str, s: float) -> None:
        fast = bs.price(flag, s, 100.0, 0.25, 0.05, 0.2)
        slow = bs.price(flag, np.array([s]), 100.0, 0.25, 0.05, 0.2)
        assert same_bits(fast, elem(slow))

    @pytest.mark.parametrize("t", [0.0, -1.0, math.inf, math.nan])
    def test_price_edge_expiry(self, t: float) -> None:
        fast = bs.price("c", 100.0, 100.0, t, 0.05, 0.2)
        slow = bs.price("c", np.array([100.0]), 100.0, t, 0.05, 0.2)
        assert same_bits(fast, elem(slow))


class TestDispatchRouting:
    """Which inputs take the fast path, verified by spying on the scalar FFI."""

    def _spy(self, monkeypatch: pytest.MonkeyPatch, name: str) -> list[tuple[object, ...]]:
        calls: list[tuple[object, ...]] = []
        orig = getattr(_core, name)

        def wrapper(*args: object) -> object:
            calls.append(args)
            return orig(*args)

        monkeypatch.setattr(_core, name, wrapper)
        return calls

    @pytest.mark.parametrize(
        "value",
        [100.0, 100, np.float64(100.0), np.float32(100.0), np.int64(100), True],
        ids=["float", "int", "np.float64", "np.float32", "np.int64", "bool"],
    )
    def test_numeric_scalars_take_fast_path(
        self, monkeypatch: pytest.MonkeyPatch, value: object
    ) -> None:
        calls = self._spy(monkeypatch, "bsm_price_scalar")
        result = bs.price("c", value, 100.0, 0.25, 0.05, 0.2)
        assert len(calls) == 1
        assert type(result) is float

    @pytest.mark.parametrize(
        "value",
        [np.array([100.0]), np.array(100.0), [100.0], (100.0,), Decimal(100), "100"],
        ids=["ndarray", "0d-ndarray", "list", "tuple", "Decimal", "str"],
    )
    def test_non_scalars_fall_through(self, monkeypatch: pytest.MonkeyPatch, value: object) -> None:
        calls = self._spy(monkeypatch, "bsm_price_scalar")
        bs.price("c", value, 100.0, 0.25, 0.05, 0.2)
        assert calls == []

    def test_numeric_flag_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._spy(monkeypatch, "bsm_price_scalar")
        result = bs.price(1, 100.0, 100.0, 0.25, 0.05, 0.2)
        assert calls == []
        # The array path still collapses all-scalar inputs to a Python float,
        # and both paths must price the same call.
        assert type(result) is float
        assert same_bits(result, bs.price("c", 100.0, 100.0, 0.25, 0.05, 0.2))

    def test_return_as_dict_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._spy(monkeypatch, "bsm_price_scalar")
        result = bs.price("c", 100.0, 100.0, 0.25, 0.05, 0.2, return_as="dict")
        assert calls == []
        assert isinstance(result, dict)

    def test_black76_fast_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._spy(monkeypatch, "black76_price_scalar")
        result = black76.price("p", 100.0, 100.0, 0.25, 0.05, 0.2)
        assert len(calls) == 1
        assert type(result) is float

    def test_greeks_fast_path_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._spy(monkeypatch, "bsm_greeks_scalar")
        result = bs.greeks("c", 100.0, 100.0, 0.25, 0.05, 0.2)
        assert len(calls) == 1
        assert tuple(result.keys()) == GREEK_KEYS
        assert all(type(v) is float for v in result.values())


class TestErrorParity:
    """Eligible inputs must raise exactly what the array path raises."""

    @pytest.mark.parametrize("bad", ["x", "price", "", "cc"])
    def test_bad_string_flag_message_identical(self, bad: str) -> None:
        with pytest.raises(ValueError, match="flag must be") as fast_exc:
            bs.price(bad, 100.0, 100.0, 0.25, 0.05, 0.2)
        with pytest.raises(ValueError, match="flag must be") as slow_exc:
            bs.price(bad, np.array([100.0]), 100.0, 0.25, 0.05, 0.2)
        assert str(fast_exc.value) == str(slow_exc.value)

    def test_oversized_int_overflows_on_both_paths(self) -> None:
        with pytest.raises(OverflowError, match="int too large"):
            bs.price("c", 10**400, 100.0, 0.25, 0.05, 0.2)
        with pytest.raises(OverflowError, match="int too large"):
            bs.price("c", 10**400, np.array([100.0]), 0.25, 0.05, 0.2)


class TestOnErrorScalarParity:
    """The scalar on_error twin: same classes, messages, and warning depth."""

    # A call price above S violates the upper no-arbitrage bound -> NaN.
    UNSOLVABLE = (1000.0, "c", 100.0, 100.0, 0.25, 0.05)

    def test_warn_default_message_and_depth(self) -> None:
        p, flag, s, k, t, r = self.UNSOLVABLE
        with warnings.catch_warnings(record=True) as fast_w:
            warnings.simplefilter("always")
            fast = bs.implied_vol(p, flag, s, k, t, r)
        with warnings.catch_warnings(record=True) as slow_w:
            warnings.simplefilter("always")
            slow = bs.implied_vol(p, flag, np.array([s]), k, t, r)
        assert math.isnan(fast)
        assert math.isnan(elem(slow))
        assert len(fast_w) == len(slow_w) == 1
        assert fast_w[0].category is ImpliedVolWarning is slow_w[0].category
        assert str(fast_w[0].message) == str(slow_w[0].message)
        # stacklevel parity: both warnings must attribute to THIS file.
        assert fast_w[0].filename == __file__ == slow_w[0].filename

    def test_raise_message_identical(self) -> None:
        p, flag, s, k, t, r = self.UNSOLVABLE
        with pytest.raises(ImpliedVolError) as fast_exc:
            bs.implied_vol(p, flag, s, k, t, r, on_error="raise")
        with pytest.raises(ImpliedVolError) as slow_exc:
            bs.implied_vol(p, flag, np.array([s]), k, t, r, on_error="raise")
        assert str(fast_exc.value) == str(slow_exc.value)

    def test_ignore_is_silent(self) -> None:
        p, flag, s, k, t, r = self.UNSOLVABLE
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = bs.implied_vol(p, flag, s, k, t, r, on_error="ignore")
        assert math.isnan(out)

    def test_invalid_on_error_value_identical(self) -> None:
        with pytest.raises(ValueError, match="on_error must be") as fast_exc:
            bs.implied_vol(2.3, "c", 100.0, 100.0, 0.25, 0.05, on_error="bogus")
        with pytest.raises(ValueError, match="on_error must be") as slow_exc:
            bs.implied_vol(2.3, "c", np.array([100.0]), 100.0, 0.25, 0.05, on_error="bogus")
        assert str(fast_exc.value) == str(slow_exc.value)

    def test_black76_warn_parity(self) -> None:
        with pytest.warns(ImpliedVolWarning) as fast_w:
            black76.implied_vol(1000.0, "c", 100.0, 100.0, 0.25, 0.05)
        with pytest.warns(ImpliedVolWarning) as slow_w:
            black76.implied_vol(1000.0, "c", np.array([100.0]), 100.0, 0.25, 0.05)
        assert str(fast_w[0].message) == str(slow_w[0].message)


class TestStrictNumericFlags:
    """Numeric flags outside {-1, 1} must be rejected, not wrapped by astype."""

    def test_plus_minus_one_arrays_work(self) -> None:
        out = bs.price([1, -1], [100.0, 100.0], 100.0, 0.25, 0.05, 0.2)
        ref_call = bs.price("c", 100.0, 100.0, 0.25, 0.05, 0.2)
        ref_put = bs.price("p", 100.0, 100.0, 0.25, 0.05, 0.2)
        assert same_bits(elem(out, 0), ref_call)
        assert same_bits(elem(out, 1), ref_put)

    def test_float_pm_one_arrays_work(self) -> None:
        out = bs.price(np.array([1.0, -1.0]), [100.0, 100.0], 100.0, 0.25, 0.05, 0.2)
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)

    @pytest.mark.parametrize("bad", [0, -256, 300, 2, -2])
    def test_out_of_range_int_flags_rejected(self, bad: int) -> None:
        with pytest.raises(ValueError, match=r"flag values must be 1 \(call\) or -1 \(put\)"):
            bs.price([bad], [100.0], 100.0, 0.25, 0.05, 0.2)

    def test_nan_flag_rejected(self) -> None:
        # The old astype(int8) path mapped NaN to 0, silently pricing a call.
        with pytest.raises(ValueError, match="flag values must be"):
            bs.price([math.nan], [100.0], 100.0, 0.25, 0.05, 0.2)

    def test_scalar_int_flag_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="flag values must be"):
            bs.price(5, 100.0, 100.0, 0.25, 0.05, 0.2)

    def test_false_bool_flag_rejected(self) -> None:
        # False == 0: neither a call nor a put; must not silently price a call.
        with pytest.raises(ValueError, match="flag values must be"):
            bs.price([True, False], [100.0, 100.0], 100.0, 0.25, 0.05, 0.2)

    def test_black76_strictness(self) -> None:
        with pytest.raises(ValueError, match="flag values must be"):
            black76.price([0], [100.0], 100.0, 0.25, 0.05, 0.2)
