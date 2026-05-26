"""Differential tests against the reference `py_vollib` implementation.

Compares the `pyvolr.compat.py_vollib.*` shim — same module tree, same
signatures, same unit conventions — against `py_vollib` itself. If pyvolr
ever drifts numerically from the reference, these tests fail.

Skipped when `py_vollib` isn't importable (the common case, since it only
imports cleanly on Python 3.10/3.11 and even there often needs the
`_testcapi` patch — see docs/why.md). On a 3.10 sidecar with both libraries
installed, these run as part of the standard test suite.

Marked `@pytest.mark.differential` so the main `pytest` run on Python 3.12+
doesn't try to collect them.
"""

# ruff: noqa: E402
# pyright: reportMissingImports=false, reportUntypedFunctionDecorator=false
# pytest.importorskip must run before py_vollib's imports so this file skips
# cleanly on Pythons where py_vollib doesn't install. That requires module-level
# imports after a statement, which E402 normally flags. Pyright also runs in the
# lint job where py_vollib isn't installed; with no module visible, the
# parametrize decorators become "untyped" from pyright's view.

from __future__ import annotations

from itertools import product

import pytest

py_vollib = pytest.importorskip("py_vollib")

# Reference implementations from py_vollib.
from py_vollib.black import black as _pvol_black
from py_vollib.black.greeks.analytical import (
    delta as _pvol_black_delta,
)
from py_vollib.black.greeks.analytical import (
    gamma as _pvol_black_gamma,
)
from py_vollib.black.greeks.analytical import (
    rho as _pvol_black_rho,
)
from py_vollib.black.greeks.analytical import (
    theta as _pvol_black_theta,
)
from py_vollib.black.greeks.analytical import (
    vega as _pvol_black_vega,
)
from py_vollib.black.implied_volatility import (
    implied_volatility as _pvol_black_iv,
)
from py_vollib.black_scholes import black_scholes as _pvol_bs
from py_vollib.black_scholes.greeks.analytical import (
    delta as _pvol_delta,
)
from py_vollib.black_scholes.greeks.analytical import (
    gamma as _pvol_gamma,
)
from py_vollib.black_scholes.greeks.analytical import (
    rho as _pvol_rho,
)
from py_vollib.black_scholes.greeks.analytical import (
    theta as _pvol_theta,
)
from py_vollib.black_scholes.greeks.analytical import (
    vega as _pvol_vega,
)
from py_vollib.black_scholes.implied_volatility import (
    implied_volatility as _pvol_iv,
)
from py_vollib.black_scholes_merton import black_scholes_merton as _pvol_bsm

from pyvolr.compat.py_vollib.black import (
    black as _pv_black,
)
from pyvolr.compat.py_vollib.black.greeks.analytical import (
    delta as _pv_black_delta,
)
from pyvolr.compat.py_vollib.black.greeks.analytical import (
    gamma as _pv_black_gamma,
)
from pyvolr.compat.py_vollib.black.greeks.analytical import (
    rho as _pv_black_rho,
)
from pyvolr.compat.py_vollib.black.greeks.analytical import (
    theta as _pv_black_theta,
)
from pyvolr.compat.py_vollib.black.greeks.analytical import (
    vega as _pv_black_vega,
)
from pyvolr.compat.py_vollib.black.implied_volatility import (
    implied_volatility as _pv_black_iv,
)

# pyvolr compat shim — same module tree, same signatures.
from pyvolr.compat.py_vollib.black_scholes import (
    black_scholes as _pv_bs,
)
from pyvolr.compat.py_vollib.black_scholes.greeks.analytical import (
    delta as _pv_delta,
)
from pyvolr.compat.py_vollib.black_scholes.greeks.analytical import (
    gamma as _pv_gamma,
)
from pyvolr.compat.py_vollib.black_scholes.greeks.analytical import (
    rho as _pv_rho,
)
from pyvolr.compat.py_vollib.black_scholes.greeks.analytical import (
    theta as _pv_theta,
)
from pyvolr.compat.py_vollib.black_scholes.greeks.analytical import (
    vega as _pv_vega,
)
from pyvolr.compat.py_vollib.black_scholes.implied_volatility import (
    implied_volatility as _pv_iv,
)
from pyvolr.compat.py_vollib.black_scholes_merton import (
    black_scholes_merton as _pv_bsm,
)

pytestmark = pytest.mark.differential


# Grid spanning ATM/ITM/OTM x short/medium/long expiry x low/medium/high vol.
# Realistic financial ranges; not the f64 extremes the fuzz harness explores.
SPOTS = [50.0, 100.0, 200.0]
STRIKES = [50.0, 90.0, 100.0, 110.0, 150.0]
TIMES = [0.05, 0.5, 1.0, 2.0]
RATES = [-0.01, 0.0, 0.05]
SIGMAS = [0.10, 0.25, 0.50]
FLAGS = ["c", "p"]
YIELDS = [0.0, 0.03]  # for Merton variant

# 1e-10 absolute for price/Greeks is well above f64 roundoff at these
# magnitudes (max price ~150) and matches what the bsm/black76 unit tests use.
PRICE_TOL = 1e-10
GREEK_TOL = 1e-10
# With LBR, both libraries now use Jäckel's algorithm under the hood; the
# differential collapses to f64 precision. Tightened from 1e-6.
IV_TOL = 1e-12


def _params(*axes):
    """Cross-product of input grids for pytest.mark.parametrize."""
    return list(product(*axes))


@pytest.mark.parametrize(
    ("s", "k", "t", "r", "sigma", "flag"),
    _params(SPOTS, STRIKES, TIMES, RATES, SIGMAS, FLAGS),
)
def test_price_matches_py_vollib(
    s: float, k: float, t: float, r: float, sigma: float, flag: str
) -> None:
    pv = _pv_bs(flag, s, k, t, r, sigma)
    ref = _pvol_bs(flag, s, k, t, r, sigma)
    assert pv == pytest.approx(ref, abs=PRICE_TOL)


@pytest.mark.parametrize(
    ("s", "k", "t", "r", "sigma", "flag"),
    _params(SPOTS, STRIKES, TIMES, RATES, SIGMAS, FLAGS),
)
def test_delta_matches_py_vollib(
    s: float, k: float, t: float, r: float, sigma: float, flag: str
) -> None:
    pv = _pv_delta(flag, s, k, t, r, sigma)
    ref = _pvol_delta(flag, s, k, t, r, sigma)
    assert pv == pytest.approx(ref, abs=GREEK_TOL)


@pytest.mark.parametrize(
    ("s", "k", "t", "r", "sigma", "flag"),
    _params(SPOTS, STRIKES, TIMES, RATES, SIGMAS, FLAGS),
)
def test_gamma_matches_py_vollib(
    s: float, k: float, t: float, r: float, sigma: float, flag: str
) -> None:
    pv = _pv_gamma(flag, s, k, t, r, sigma)
    ref = _pvol_gamma(flag, s, k, t, r, sigma)
    assert pv == pytest.approx(ref, abs=GREEK_TOL)


@pytest.mark.parametrize(
    ("s", "k", "t", "r", "sigma", "flag"),
    _params(SPOTS, STRIKES, TIMES, RATES, SIGMAS, FLAGS),
)
def test_vega_matches_py_vollib(
    s: float, k: float, t: float, r: float, sigma: float, flag: str
) -> None:
    # py_vollib vega: per 1% vol change. Compat shim preserves this.
    pv = _pv_vega(flag, s, k, t, r, sigma)
    ref = _pvol_vega(flag, s, k, t, r, sigma)
    assert pv == pytest.approx(ref, abs=GREEK_TOL)


@pytest.mark.parametrize(
    ("s", "k", "t", "r", "sigma", "flag"),
    _params(SPOTS, STRIKES, TIMES, RATES, SIGMAS, FLAGS),
)
def test_theta_matches_py_vollib(
    s: float, k: float, t: float, r: float, sigma: float, flag: str
) -> None:
    # py_vollib theta: per day. Compat shim preserves this.
    pv = _pv_theta(flag, s, k, t, r, sigma)
    ref = _pvol_theta(flag, s, k, t, r, sigma)
    assert pv == pytest.approx(ref, abs=GREEK_TOL)


@pytest.mark.parametrize(
    ("s", "k", "t", "r", "sigma", "flag"),
    _params(SPOTS, STRIKES, TIMES, RATES, SIGMAS, FLAGS),
)
def test_rho_matches_py_vollib(
    s: float, k: float, t: float, r: float, sigma: float, flag: str
) -> None:
    # py_vollib rho: per 1% rate change. Compat shim preserves this.
    pv = _pv_rho(flag, s, k, t, r, sigma)
    ref = _pvol_rho(flag, s, k, t, r, sigma)
    assert pv == pytest.approx(ref, abs=GREEK_TOL)


@pytest.mark.parametrize(
    ("s", "k", "t", "r", "sigma", "flag"),
    # Trim IV grid: solving is expensive and we only need to confirm both
    # libraries agree on sigma, which is a single root regardless of inputs.
    _params([100.0], [90.0, 100.0, 110.0], [0.25, 1.0], [0.0, 0.05], SIGMAS, FLAGS),
)
def test_iv_matches_py_vollib(
    s: float, k: float, t: float, r: float, sigma: float, flag: str
) -> None:
    # Forward: price the option, then invert both libraries from that price.
    ref_price = _pvol_bs(flag, s, k, t, r, sigma)
    pv_iv = _pv_iv(ref_price, s, k, t, r, flag)
    ref_iv = _pvol_iv(ref_price, s, k, t, r, flag)
    assert pv_iv == pytest.approx(ref_iv, abs=IV_TOL)


@pytest.mark.parametrize(
    ("s", "k", "t", "r", "sigma", "flag", "q"),
    _params(SPOTS, [90.0, 100.0, 110.0], TIMES, [0.0, 0.05], SIGMAS, FLAGS, YIELDS),
)
def test_merton_price_matches_py_vollib(
    s: float, k: float, t: float, r: float, sigma: float, flag: str, q: float
) -> None:
    pv = _pv_bsm(flag, s, k, t, r, q, sigma)
    ref = _pvol_bsm(flag, s, k, t, r, q, sigma)
    assert pv == pytest.approx(ref, abs=PRICE_TOL)


# Black-76: futures options. Same grid shape as BSM; F replaces S.
FORWARDS = SPOTS


@pytest.mark.parametrize(
    ("f", "k", "t", "r", "sigma", "flag"),
    _params(FORWARDS, STRIKES, TIMES, RATES, SIGMAS, FLAGS),
)
def test_black76_price_matches_py_vollib(
    f: float, k: float, t: float, r: float, sigma: float, flag: str
) -> None:
    pv = _pv_black(flag, f, k, t, r, sigma)
    ref = _pvol_black(flag, f, k, t, r, sigma)
    assert pv == pytest.approx(ref, abs=PRICE_TOL)


@pytest.mark.parametrize(
    ("f", "k", "t", "r", "sigma", "flag"),
    _params(FORWARDS, STRIKES, TIMES, RATES, SIGMAS, FLAGS),
)
def test_black76_delta_matches_py_vollib(
    f: float, k: float, t: float, r: float, sigma: float, flag: str
) -> None:
    pv = _pv_black_delta(flag, f, k, t, r, sigma)
    ref = _pvol_black_delta(flag, f, k, t, r, sigma)
    assert pv == pytest.approx(ref, abs=GREEK_TOL)


@pytest.mark.parametrize(
    ("f", "k", "t", "r", "sigma", "flag"),
    _params(FORWARDS, STRIKES, TIMES, RATES, SIGMAS, FLAGS),
)
def test_black76_gamma_matches_py_vollib(
    f: float, k: float, t: float, r: float, sigma: float, flag: str
) -> None:
    pv = _pv_black_gamma(flag, f, k, t, r, sigma)
    ref = _pvol_black_gamma(flag, f, k, t, r, sigma)
    assert pv == pytest.approx(ref, abs=GREEK_TOL)


@pytest.mark.parametrize(
    ("f", "k", "t", "r", "sigma", "flag"),
    _params(FORWARDS, STRIKES, TIMES, RATES, SIGMAS, FLAGS),
)
def test_black76_vega_matches_py_vollib(
    f: float, k: float, t: float, r: float, sigma: float, flag: str
) -> None:
    # py_vollib.black.vega is per 1% vol change. Compat shim preserves this.
    pv = _pv_black_vega(flag, f, k, t, r, sigma)
    ref = _pvol_black_vega(flag, f, k, t, r, sigma)
    assert pv == pytest.approx(ref, abs=GREEK_TOL)


@pytest.mark.parametrize(
    ("f", "k", "t", "r", "sigma", "flag"),
    _params(FORWARDS, STRIKES, TIMES, RATES, SIGMAS, FLAGS),
)
def test_black76_theta_matches_py_vollib(
    f: float, k: float, t: float, r: float, sigma: float, flag: str
) -> None:
    # py_vollib.black.theta is per day. Compat shim preserves this.
    pv = _pv_black_theta(flag, f, k, t, r, sigma)
    ref = _pvol_black_theta(flag, f, k, t, r, sigma)
    assert pv == pytest.approx(ref, abs=GREEK_TOL)


@pytest.mark.parametrize(
    ("f", "k", "t", "r", "sigma", "flag"),
    _params(FORWARDS, STRIKES, TIMES, RATES, SIGMAS, FLAGS),
)
def test_black76_rho_matches_py_vollib(
    f: float, k: float, t: float, r: float, sigma: float, flag: str
) -> None:
    # py_vollib.black.rho is per 1% rate change. Compat shim preserves this.
    pv = _pv_black_rho(flag, f, k, t, r, sigma)
    ref = _pvol_black_rho(flag, f, k, t, r, sigma)
    assert pv == pytest.approx(ref, abs=GREEK_TOL)


@pytest.mark.parametrize(
    ("f", "k", "t", "r", "sigma", "flag"),
    _params([100.0], [90.0, 100.0, 110.0], [0.25, 1.0], [0.0, 0.05], SIGMAS, FLAGS),
)
def test_black76_iv_matches_py_vollib(
    f: float, k: float, t: float, r: float, sigma: float, flag: str
) -> None:
    # py_vollib.black IV signature is (price, F, K, r, t, flag) — r/t order
    # differs from black_scholes.implied_volatility (t/r). Mirror exactly.
    ref_price = _pvol_black(flag, f, k, t, r, sigma)
    pv_iv = _pv_black_iv(ref_price, f, k, r, t, flag)
    ref_iv = _pvol_black_iv(ref_price, f, k, r, t, flag)
    assert pv_iv == pytest.approx(ref_iv, abs=IV_TOL)
