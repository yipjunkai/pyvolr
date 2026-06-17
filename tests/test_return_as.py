"""Tests for the ``return_as`` output-container option on ``bs`` / ``black76``.

Every public function accepts a keyword-only ``return_as``:

    None / "numpy"  -> bare numpy result (scalar when all inputs scalar)
    "dict"          -> {name: result}
    "dataframe"     -> pandas DataFrame (pandas is a soft dependency)

The source of truth for *values* is the default (numpy) mode; these tests assert
the other containers agree with it, plus the container shape/keys/columns. pandas
cases ``importorskip`` so legs without pandas skip cleanly; the missing-pandas
error path is exercised separately via ``sys.modules`` stubbing.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from pyvolr import black76, bs

MODULES = [pytest.param(bs, id="bs"), pytest.param(black76, id="black76")]
GREEK_COLUMNS = ["delta", "gamma", "theta", "vega", "rho"]


def price_kw(mod, *, K=105.0, sigma=0.2):
    """Keyword args for ``mod.price`` / ``mod.greeks`` (S for bs, F for black76)."""
    spot = {"S": 100.0} if mod is bs else {"F": 100.0}
    return {"flag": "c", **spot, "K": K, "T": 0.5, "r": 0.05, "sigma": sigma}


def iv_kw(mod):
    """Keyword args for ``mod.implied_vol`` (different signature: price, flag, ...)."""
    spot = {"S": 100.0} if mod is bs else {"F": 100.0}
    return {"price": 5.0, "flag": "c", **spot, "K": 100.0, "T": 0.5, "r": 0.05}


@pytest.fixture
def pd():
    return pytest.importorskip("pandas")


class TestBackwardCompatDefault:
    """The default (no return_as) return is unchanged from before the feature."""

    @pytest.mark.parametrize(("mod"), MODULES)
    def test_scalar_price_is_float(self, mod) -> None:
        out = mod.price(**price_kw(mod))
        assert isinstance(out, float)

    @pytest.mark.parametrize(("mod"), MODULES)
    def test_array_price_is_ndarray(self, mod) -> None:
        out = mod.price(**price_kw(mod, K=np.array([100.0, 105.0])))
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)

    @pytest.mark.parametrize(("mod"), MODULES)
    def test_greeks_is_dict(self, mod) -> None:
        g = mod.greeks(**price_kw(mod))
        assert isinstance(g, dict)
        assert set(g) == set(GREEK_COLUMNS)
        assert all(isinstance(v, float) for v in g.values())

    @pytest.mark.parametrize(("mod"), MODULES)
    def test_none_equals_numpy(self, mod) -> None:
        kw = price_kw(mod, K=np.linspace(80.0, 120.0, 7))
        np.testing.assert_array_equal(mod.price(**kw), mod.price(**kw, return_as="numpy"))

    @pytest.mark.parametrize(("mod"), MODULES)
    def test_explicit_none_matches_default(self, mod) -> None:
        kw = price_kw(mod)
        assert mod.price(**kw) == mod.price(**kw, return_as=None)


class TestDictMode:
    @pytest.mark.parametrize(("mod"), MODULES)
    def test_scalar_single_output(self, mod) -> None:
        ref = mod.price(**price_kw(mod))
        d = mod.price(**price_kw(mod), return_as="dict")
        assert d == {"price": ref}
        assert isinstance(d["price"], float)

    @pytest.mark.parametrize(("mod"), MODULES)
    def test_array_single_output(self, mod) -> None:
        kw = price_kw(mod, K=np.linspace(80.0, 120.0, 7))
        ref = mod.price(**kw)
        d = mod.price(**kw, return_as="dict")
        assert set(d) == {"price"}
        np.testing.assert_array_equal(d["price"], ref)

    @pytest.mark.parametrize(("mod"), MODULES)
    def test_implied_vol_key_is_iv(self, mod) -> None:
        d = mod.implied_vol(**iv_kw(mod), return_as="dict")
        assert set(d) == {"iv"}
        assert d["iv"] == mod.implied_vol(**iv_kw(mod))

    @pytest.mark.parametrize(("mod"), MODULES)
    def test_greeks_numpy_dict_none_agree(self, mod) -> None:
        kw = price_kw(mod)
        assert (
            mod.greeks(**kw)
            == mod.greeks(**kw, return_as="numpy")
            == mod.greeks(**kw, return_as="dict")
        )


class TestDataFrameMode:
    @pytest.mark.parametrize(("mod"), MODULES)
    def test_scalar_is_one_row(self, mod, pd) -> None:
        ref = mod.price(**price_kw(mod))
        df = mod.price(**price_kw(mod), return_as="dataframe")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["price"]
        assert len(df) == 1
        assert df["price"].iloc[0] == ref

    @pytest.mark.parametrize(("mod"), MODULES)
    def test_array_single_output(self, mod, pd) -> None:
        kw = price_kw(mod, K=np.linspace(80.0, 120.0, 7))
        ref = mod.price(**kw)
        df = mod.price(**kw, return_as="dataframe")
        assert list(df.columns) == ["price"]
        assert len(df) == 7
        np.testing.assert_array_equal(df["price"].to_numpy(), ref)

    @pytest.mark.parametrize(("mod"), MODULES)
    def test_implied_vol_column_is_iv(self, mod, pd) -> None:
        df = mod.implied_vol(**iv_kw(mod), return_as="dataframe")
        assert list(df.columns) == ["iv"]

    @pytest.mark.parametrize(("mod"), MODULES)
    def test_greeks_columns_ordered(self, mod, pd) -> None:
        kw = price_kw(mod, K=np.linspace(80.0, 120.0, 7))
        ref = mod.greeks(**kw)
        df = mod.greeks(**kw, return_as="dataframe")
        assert list(df.columns) == GREEK_COLUMNS
        assert len(df) == 7
        for name in GREEK_COLUMNS:
            np.testing.assert_array_equal(df[name].to_numpy(), ref[name])

    @pytest.mark.parametrize(("mod"), MODULES)
    def test_multidim_broadcast_ravels_c_order(self, mod, pd) -> None:
        K = np.linspace(80.0, 120.0, 5).reshape(-1, 1)
        sigma = np.linspace(0.1, 0.4, 4).reshape(1, -1)
        kw = price_kw(mod, K=K, sigma=sigma)
        ref = mod.price(**kw)
        assert np.asarray(ref).shape == (5, 4)
        df = mod.price(**kw, return_as="dataframe")
        assert len(df) == 20
        np.testing.assert_array_equal(df["price"].to_numpy(), np.asarray(ref).ravel())


class TestErrors:
    @pytest.mark.parametrize(("mod"), MODULES)
    def test_invalid_return_as_price(self, mod) -> None:
        with pytest.raises(ValueError, match="return_as must be"):
            mod.price(**price_kw(mod), return_as="series")

    @pytest.mark.parametrize(("mod"), MODULES)
    def test_invalid_return_as_greeks(self, mod) -> None:
        with pytest.raises(ValueError, match="return_as must be"):
            mod.greeks(**price_kw(mod), return_as="json")

    @pytest.mark.parametrize(("mod"), MODULES)
    def test_dataframe_without_pandas_raises(self, mod, monkeypatch) -> None:
        # Stubbing sys.modules["pandas"] = None makes `import pandas` raise
        # ImportError; the helper re-raises a ModuleNotFoundError with guidance.
        monkeypatch.setitem(sys.modules, "pandas", None)
        with pytest.raises(ModuleNotFoundError, match="requires pandas"):
            mod.price(**price_kw(mod), return_as="dataframe")
