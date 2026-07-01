"""Differential: the vectorized shim vs the real py_vollib_vectorized.

Skips unless py_vollib_vectorized is importable — it depends on py_vollib, so this
only runs on Python <=3.11 (e.g. the `.venv-bench` environment). Pricing and IV
are compared tightly; greeks loosely, because pyvolr's greeks are analytical
while py_vollib_vectorized's are numerical (finite-difference).
"""

# pyright: reportMissingImports=false, reportUnknownMemberType=false
from __future__ import annotations

import numpy as np
import pytest

from pyvolr import bs
from pyvolr.compat import py_vollib_vectorized as pvv

real = pytest.importorskip("py_vollib_vectorized")

pytestmark = pytest.mark.differential

S = np.array([100.0, 100.0, 100.0])
K = np.array([90.0, 100.0, 110.0])
T = np.array([0.5, 1.0, 1.5])
R = np.array([0.05, 0.03, 0.04])
SIG = np.array([0.20, 0.25, 0.30])
GREEK_ORDER = ["delta", "gamma", "theta", "rho", "vega"]


def test_pricing_matches() -> None:
    ours = pvv.vectorized_black_scholes("c", S, K, T, R, SIG, return_as="numpy")
    theirs = np.asarray(
        real.vectorized_black_scholes("c", S, K, T, R, SIG, return_as="numpy")
    ).ravel()
    np.testing.assert_allclose(ours, theirs, rtol=1e-10)


def test_iv_matches() -> None:
    p = bs.price("c", S=S, K=K, T=T, r=R, sigma=SIG)
    ours = pvv.vectorized_implied_volatility(p, S, K, T, R, "c", return_as="numpy")
    theirs = np.asarray(
        real.vectorized_implied_volatility(p, S, K, T, R, "c", return_as="numpy")
    ).ravel()
    np.testing.assert_allclose(ours, theirs, rtol=1e-8)


def test_greeks_match_to_fd_tolerance() -> None:
    ours = pvv.get_all_greeks("c", S, K, T, R, SIG, return_as="dict")
    theirs = real.get_all_greeks("c", S, K, T, R, SIG, return_as="dict")
    assert isinstance(ours, dict)
    for g in GREEK_ORDER:
        # pyvolr analytical vs py_vollib_vectorized numerical (finite difference).
        np.testing.assert_allclose(
            np.asarray(ours[g]).ravel(), np.asarray(theirs[g]).ravel(), atol=1e-4, rtol=1e-3
        )
