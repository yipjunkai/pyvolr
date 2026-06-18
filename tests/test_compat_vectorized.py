"""Tests for `pyvolr.compat.py_vollib_vectorized` (drop-in shim).

Value checks use ``return_as="numpy"`` (a bare array — no union narrowing);
structure checks (`columns`, Series `name`) use the real pandas types. pandas is
always present in the test environment (the `test` extra), so it is imported at
module scope; the soft-dependency *fallback* is exercised by stubbing
``sys.modules["pandas"] = None`` at call time. The differential vs the real
py_vollib_vectorized lives in `test_differential_vectorized.py`.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from pyvolr import ImpliedVolError, ImpliedVolWarning, black76, bs
from pyvolr.compat import py_vollib_vectorized as pvv

S = np.array([100.0, 100.0, 100.0])
K = np.array([90.0, 100.0, 110.0])
T = np.array([0.5, 1.0, 1.5])
R = np.array([0.05, 0.03, 0.04])
SIG = np.array([0.20, 0.25, 0.30])
Q = np.array([0.0, 0.01, 0.02])
GREEK_ORDER = ["delta", "gamma", "theta", "rho", "vega"]


class TestPricing:
    def test_black_scholes_matches_native(self) -> None:
        out = pvv.vectorized_black_scholes("c", S, K, T, R, SIG, return_as="numpy")
        np.testing.assert_allclose(out, bs.price("c", S=S, K=K, T=T, r=R, sigma=SIG))

    def test_bsm_uses_q(self) -> None:
        out = pvv.vectorized_black_scholes_merton("c", S, K, T, R, SIG, Q, return_as="numpy")
        np.testing.assert_allclose(out, bs.price("c", S=S, K=K, T=T, r=R, sigma=SIG, q=Q))

    def test_black_matches_native(self) -> None:
        out = pvv.vectorized_black("c", S, K, T, R, SIG, return_as="numpy")
        np.testing.assert_allclose(out, black76.price("c", F=S, K=K, T=T, r=R, sigma=SIG))

    def test_price_column_name(self) -> None:
        df = pvv.vectorized_black_scholes("c", S, K, T, R, SIG)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["Price"]


class TestReturnAs:
    def test_default_is_dataframe(self) -> None:
        assert isinstance(pvv.vectorized_black_scholes("c", S, K, T, R, SIG), pd.DataFrame)

    def test_series_has_name(self) -> None:
        s = pvv.vectorized_black_scholes("c", S, K, T, R, SIG, return_as="series")
        assert isinstance(s, pd.Series)
        assert s.name == "Price"

    def test_other_value_returns_numpy(self) -> None:
        out = pvv.vectorized_black_scholes("c", S, K, T, R, SIG, return_as="numpy")
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)

    def test_scalar_broadcasts_to_length_one(self) -> None:
        # pvv never collapses to a scalar — scalar input yields a length-1 array.
        out = pvv.vectorized_black_scholes("c", 100, 100, 1.0, 0.05, 0.2, return_as="numpy")
        assert isinstance(out, np.ndarray)
        assert out.shape == (1,)

    def test_dtype_is_honored(self) -> None:
        out = pvv.vectorized_black_scholes(
            "c", S, K, T, R, SIG, return_as="numpy", dtype=np.float32
        )
        assert out.dtype == np.float32


class TestImpliedVol:
    def test_flag_after_r_arg_order(self) -> None:
        p = bs.price("c", S=S, K=K, T=T, r=R, sigma=SIG)
        iv = pvv.vectorized_implied_volatility(p, S, K, T, R, "c", return_as="numpy")
        np.testing.assert_allclose(iv, SIG, atol=1e-9)

    def test_bsm_model_uses_q(self) -> None:
        p = bs.price("c", S=S, K=K, T=T, r=R, sigma=SIG, q=Q)
        iv = pvv.vectorized_implied_volatility(
            p, S, K, T, R, "c", Q, model="black_scholes_merton", return_as="numpy"
        )
        np.testing.assert_allclose(iv, SIG, atol=1e-9)

    def test_black_variant_r_before_t(self) -> None:
        p = black76.price("c", F=S, K=K, T=T, r=R, sigma=SIG)
        iv = pvv.vectorized_implied_volatility_black(p, S, K, R, T, "c", return_as="numpy")
        np.testing.assert_allclose(iv, SIG, atol=1e-9)

    def test_iv_column_name(self) -> None:
        p = bs.price("c", S=S, K=K, T=T, r=R, sigma=SIG)
        df = pvv.vectorized_implied_volatility(p, S, K, T, R, "c")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["IV"]

    def test_on_error_raise(self) -> None:
        with pytest.raises(ImpliedVolError):
            pvv.vectorized_implied_volatility(
                np.array([-1.0]),
                np.array([100.0]),
                np.array([100.0]),
                np.array([1.0]),
                np.array([0.05]),
                "c",
                on_error="raise",
                return_as="numpy",
            )

    def test_on_error_warn_is_default(self) -> None:
        with pytest.warns(ImpliedVolWarning):
            pvv.vectorized_implied_volatility(
                np.array([-1.0]),
                np.array([100.0]),
                np.array([100.0]),
                np.array([1.0]),
                np.array([0.05]),
                "c",
                return_as="numpy",
            )


class TestGreeks:
    def test_get_all_greeks_column_order(self) -> None:
        g = pvv.get_all_greeks("c", S, K, T, R, SIG)
        assert isinstance(g, pd.DataFrame)
        assert list(g.columns) == GREEK_ORDER

    def test_get_all_greeks_dict(self) -> None:
        d = pvv.get_all_greeks("c", S, K, T, R, SIG, return_as="dict")
        assert isinstance(d, dict)
        assert list(d) == GREEK_ORDER

    def test_get_all_greeks_json(self) -> None:
        out = pvv.get_all_greeks("c", S, K, T, R, SIG, return_as="json")
        assert isinstance(out, str)
        assert set(json.loads(out)) == set(GREEK_ORDER)

    def test_get_all_greeks_bsm_requires_q(self) -> None:
        with pytest.raises(ValueError, match="requires q"):
            pvv.get_all_greeks("c", S, K, T, R, SIG, model="black_scholes_merton")

    def test_single_greek_unit_conventions(self) -> None:
        # py_vollib units: theta per day, vega/rho per 1%; delta/gamma per unit.
        np.testing.assert_allclose(
            pvv.vectorized_theta("c", S, K, T, R, SIG, return_as="numpy"),
            bs.theta("c", S=S, K=K, T=T, r=R, sigma=SIG) / 365.0,
        )
        np.testing.assert_allclose(
            pvv.vectorized_vega("c", S, K, T, R, SIG, return_as="numpy"),
            bs.vega(S=S, K=K, T=T, r=R, sigma=SIG) / 100.0,
        )
        np.testing.assert_allclose(
            pvv.vectorized_rho("c", S, K, T, R, SIG, return_as="numpy"),
            bs.rho("c", S=S, K=K, T=T, r=R, sigma=SIG) / 100.0,
        )

    def test_greek_model_black_uses_forward(self) -> None:
        d = pvv.vectorized_delta("c", S, K, T, R, SIG, model="black", return_as="numpy")
        np.testing.assert_allclose(d, black76.delta("c", F=S, K=K, T=T, r=R, sigma=SIG))


class TestPandasSoftDependency:
    def test_dataframe_falls_back_to_numpy_with_warning(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "pandas", None)
        with pytest.warns(UserWarning, match="pandas is not installed"):
            out = pvv.vectorized_black_scholes("c", S, K, T, R, SIG)  # default "dataframe"
        assert isinstance(out, np.ndarray)

    def test_get_all_greeks_falls_back_to_dict(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "pandas", None)
        with pytest.warns(UserWarning, match="pandas is not installed"):
            out = pvv.get_all_greeks("c", S, K, T, R, SIG)  # default "dataframe"
        assert isinstance(out, dict)

    def test_json_needs_no_pandas(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "pandas", None)
        out = pvv.get_all_greeks("c", S, K, T, R, SIG, return_as="json")
        assert isinstance(out, str)

    def test_price_dataframe_requires_pandas(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "pandas", None)
        with pytest.raises(ModuleNotFoundError, match="requires pandas"):
            pvv.price_dataframe({}, flag_col="f")  # pandas import fails before anything else


class TestPriceDataframe:
    def test_sigma_col_computes_price_and_greeks(self) -> None:
        df = pd.DataFrame(
            {
                "Flag": ["c", "p"],
                "S": [100.0, 100.0],
                "K": [100.0, 90.0],
                "T": [0.5, 0.5],
                "R": [0.05, 0.05],
                "IV": [0.2, 0.2],
            }
        )
        out = pvv.price_dataframe(
            df,
            flag_col="Flag",
            underlying_price_col="S",
            strike_col="K",
            annualized_tte_col="T",
            riskfree_rate_col="R",
            sigma_col="IV",
        )
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["Price", *GREEK_ORDER]

    def test_price_col_computes_iv_and_greeks(self) -> None:
        df = pd.DataFrame(
            {"Flag": ["c"], "S": [100.0], "K": [100.0], "T": [0.5], "R": [0.05], "P": [6.0]}
        )
        out = pvv.price_dataframe(
            df,
            flag_col="Flag",
            underlying_price_col="S",
            strike_col="K",
            annualized_tte_col="T",
            riskfree_rate_col="R",
            price_col="P",
        )
        assert isinstance(out, pd.DataFrame)
        assert "IV" in out.columns
        assert "delta" in out.columns

    def test_inplace_returns_none_and_mutates(self) -> None:
        df = pd.DataFrame(
            {"Flag": ["c"], "S": [100.0], "K": [100.0], "T": [0.5], "R": [0.05], "IV": [0.2]}
        )
        ret = pvv.price_dataframe(
            df,
            flag_col="Flag",
            underlying_price_col="S",
            strike_col="K",
            annualized_tte_col="T",
            riskfree_rate_col="R",
            sigma_col="IV",
            inplace=True,
        )
        assert ret is None
        assert "Price" in df.columns

    def test_missing_required_col_raises(self) -> None:
        df = pd.DataFrame({"Flag": ["c"]})
        with pytest.raises(ValueError, match="requires column args"):
            pvv.price_dataframe(df, flag_col="Flag", sigma_col="IV")
