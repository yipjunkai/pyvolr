"""Adversarial FFI-contract tests driving ``pyvolr._core`` directly.

The public ``pyvolr.bs`` / ``pyvolr.black76`` wrappers normalise every input
(broadcasting, dtype coercion, contiguity, flag validation) before anything
reaches Rust, so tests that go through them never actually exercise the
PyO3/rust-numpy boundary. ``SECURITY.md`` names that boundary as the top risk,
the five ``cargo fuzz`` targets all bypass it (they link the Rust functions
directly), and ``panic = abort`` would turn any regression there into an
interpreter abort rather than a catchable error.

This suite pins the boundary contract from the Python side. For arbitrary
malformed inputs -- mismatched / non-contiguous / wrong-dtype / 0-d / 2-d
arrays, ``i8`` extremes, ``NaN``/``inf`` values, and sizes straddling the
rayon dispatch thresholds -- every ``_core`` entry point must resolve to
either a clean Python exception or a well-formed ``float64`` result. It must
never abort: a Rust panic under ``panic = abort`` kills this test runner, so
the suite simply completing is itself part of the assertion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pyvolr import _core

if TYPE_CHECKING:
    from collections.abc import Callable

I8 = "i8"  # flag column (int8)
F64 = "f64"  # numeric column (float64)

# Every array entry point, described as the ordered kind of each positional
# argument. Mirrors the signatures in ``_core.pyi``.
SPECS: dict[str, tuple[Callable[..., object], tuple[str, ...]]] = {
    "bsm_price": (_core.bsm_price, (I8, F64, F64, F64, F64, F64, F64)),
    "bsm_delta": (_core.bsm_delta, (I8, F64, F64, F64, F64, F64, F64)),
    "bsm_gamma": (_core.bsm_gamma, (F64, F64, F64, F64, F64, F64)),
    "bsm_vega": (_core.bsm_vega, (F64, F64, F64, F64, F64, F64)),
    "bsm_theta": (_core.bsm_theta, (I8, F64, F64, F64, F64, F64, F64)),
    "bsm_rho": (_core.bsm_rho, (I8, F64, F64, F64, F64, F64, F64)),
    "bsm_iv": (_core.bsm_iv, (F64, I8, F64, F64, F64, F64, F64)),
    "bsm_greeks": (_core.bsm_greeks, (I8, F64, F64, F64, F64, F64, F64)),
    "black76_price": (_core.black76_price, (I8, F64, F64, F64, F64, F64)),
    "black76_delta": (_core.black76_delta, (I8, F64, F64, F64, F64, F64)),
    "black76_gamma": (_core.black76_gamma, (F64, F64, F64, F64, F64)),
    "black76_vega": (_core.black76_vega, (F64, F64, F64, F64, F64)),
    "black76_theta": (_core.black76_theta, (I8, F64, F64, F64, F64, F64)),
    "black76_rho": (_core.black76_rho, (I8, F64, F64, F64, F64, F64)),
    "black76_iv": (_core.black76_iv, (F64, I8, F64, F64, F64, F64)),
    "black76_greeks": (_core.black76_greeks, (I8, F64, F64, F64, F64, F64)),
}

NAMES = list(SPECS)


def _col(kind: str, n: int) -> np.ndarray:
    """One valid contiguous column of length ``n`` for the given kind."""
    if kind == I8:
        return np.ones(n, dtype=np.int8)  # 1 -> call
    return np.ones(n, dtype=np.float64)


def _valid_args(kinds: tuple[str, ...], n: int) -> list[np.ndarray]:
    return [_col(k, n) for k in kinds]


def _arrays(result: object) -> tuple[object, ...]:
    """A single-array or greeks-tuple result flattened to a tuple of arrays."""
    return result if isinstance(result, tuple) else (result,)


def _check_result(result: object, n: int) -> None:
    """Well-formed: an ``f64`` ndarray (or a tuple of them) of length ``n``."""
    for a in _arrays(result):
        assert isinstance(a, np.ndarray)
        assert a.dtype == np.float64
        assert a.shape == (n,)


def _assert_no_abort(func: Callable[..., object], args: list[np.ndarray]) -> None:
    """Malformed input: a clean Python exception OR a well-formed result --
    never a crash. Process survival is the real assertion (see module docstring);
    if it does return, the result must still be a well-formed ``f64`` array."""
    try:
        result = func(*args)
    except Exception:  # any Python-level exception satisfies the contract
        return
    for a in _arrays(result):
        assert isinstance(a, np.ndarray)
        assert a.dtype == np.float64


# --------------------------------------------------------------------------
#  Structurally valid inputs: must succeed and return the right shape.
# --------------------------------------------------------------------------

# Sizes straddle the serial/parallel dispatch thresholds (IV 1024, greeks 4096,
# price 8192, single-greek 16384) plus the empty and singleton edges, so both
# code paths and every boundary run.
THRESHOLD_SIZES = [0, 1, 2, 1023, 1024, 1025, 4095, 4096, 4097, 8192, 16384, 16385]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("n", THRESHOLD_SIZES)
def test_valid_sizes_across_thresholds(name: str, n: int) -> None:
    func, kinds = SPECS[name]
    _check_result(func(*_valid_args(kinds, n)), n)


@pytest.mark.parametrize("name", NAMES)
def test_i8_flag_extremes_are_total(name: str) -> None:
    """``Flag::from_i8`` is total (>= 0 call, < 0 put); every i8 is accepted."""
    func, kinds = SPECS[name]
    if I8 not in kinds:
        pytest.skip("no flag column")
    n = 2048  # above IV's 1024 gate
    for extreme in (-128, 127, 0, -1, 42):
        args = _valid_args(kinds, n)
        args[kinds.index(I8)] = np.full(n, extreme, dtype=np.int8)
        _check_result(func(*args), n)


@pytest.mark.parametrize("name", NAMES)
def test_nan_inf_values_do_not_crash(name: str) -> None:
    """``NaN``/``inf`` in a numeric column: a well-formed (possibly non-finite)
    result of the right shape, never a crash."""
    func, kinds = SPECS[name]
    n = 512
    for bad in (np.nan, np.inf, -np.inf):
        for i, kind in enumerate(kinds):
            if kind != F64:
                continue
            args = _valid_args(kinds, n)
            args[i] = np.full(n, bad, dtype=np.float64)
            _check_result(func(*args), n)


# --------------------------------------------------------------------------
#  Malformed inputs: raise-or-well-formed, never a crash.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_mismatched_lengths(name: str) -> None:
    """One column longer than the rest violates the equal-length contract."""
    func, kinds = SPECS[name]
    n = 64
    for i in range(len(kinds)):
        args = _valid_args(kinds, n)
        args[i] = _col(kinds[i], n + 1)
        _assert_no_abort(func, args)


@pytest.mark.parametrize("name", NAMES)
def test_noncontiguous_views(name: str) -> None:
    """Strided (non-contiguous) views violate the flat-contiguous contract."""
    func, kinds = SPECS[name]
    n = 64
    args = [_col(k, 2 * n)[::2] for k in kinds]
    assert all(not a.flags["C_CONTIGUOUS"] for a in args)
    _assert_no_abort(func, args)


@pytest.mark.parametrize("name", NAMES)
def test_wrong_dtypes(name: str) -> None:
    """Mismatched element dtypes (f32/i64 where f64 is expected; f64 where i8)."""
    func, kinds = SPECS[name]
    n = 64
    for i, kind in enumerate(kinds):
        if kind == I8:
            wrongs = [np.ones(n, np.float64), np.ones(n, np.int64), np.ones(n, np.int32)]
        else:
            wrongs = [np.ones(n, np.float32), np.ones(n, np.int8), np.ones(n, np.int64)]
        for wrong in wrongs:
            args = _valid_args(kinds, n)
            args[i] = wrong
            _assert_no_abort(func, args)


@pytest.mark.parametrize("name", NAMES)
def test_zero_and_higher_dim(name: str) -> None:
    """0-d and 2-d arrays violate the 1-D contract."""
    func, kinds = SPECS[name]
    n = 64
    for i, kind in enumerate(kinds):
        dt = np.int8 if kind == I8 else np.float64
        for bad in (np.array(1, dtype=dt), np.ones((n, 1), dtype=dt), np.ones((2, n), dtype=dt)):
            args = _valid_args(kinds, n)
            args[i] = bad
            _assert_no_abort(func, args)
