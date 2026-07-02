"""Type stubs for the pyvolr._core Rust extension.

Array entry points take flat (1-D, contiguous) numpy arrays of equal length.
Most return a single numpy float64 array of the same length; the bundled
``bsm_greeks`` / ``black76_greeks`` entry points return a 5-tuple of such
arrays (one per Greek). The Python wrapper in ``pyvolr.bs`` /
``pyvolr.black76`` is responsible for broadcasting and reshape.

The ``*_scalar`` twins take plain numeric scalars and return bare floats —
same kernels as the array forms, bit-for-bit. The wrapper dispatches to them
when every input is a numeric scalar (``_wrappers.SCALAR_NUMERIC``). Their
``flag`` follows the Rust ``Flag::from_i8`` total mapping (>= 0 is call,
< 0 is put), but the wrapper layer only ever passes a strictly validated ±1.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

# What the scalar twins accept at runtime: anything PyO3 extracts to f64 via
# __float__ (int is float-assignable per the numeric tower).
_Scalar: TypeAlias = float | np.floating[Any] | np.integer[Any]

__version__: str

def bsm_price(
    flag: NDArray[np.int8],
    s: NDArray[np.float64],
    k: NDArray[np.float64],
    t: NDArray[np.float64],
    r: NDArray[np.float64],
    q: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def bsm_delta(
    flag: NDArray[np.int8],
    s: NDArray[np.float64],
    k: NDArray[np.float64],
    t: NDArray[np.float64],
    r: NDArray[np.float64],
    q: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def bsm_gamma(
    s: NDArray[np.float64],
    k: NDArray[np.float64],
    t: NDArray[np.float64],
    r: NDArray[np.float64],
    q: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def bsm_vega(
    s: NDArray[np.float64],
    k: NDArray[np.float64],
    t: NDArray[np.float64],
    r: NDArray[np.float64],
    q: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def bsm_theta(
    flag: NDArray[np.int8],
    s: NDArray[np.float64],
    k: NDArray[np.float64],
    t: NDArray[np.float64],
    r: NDArray[np.float64],
    q: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def bsm_rho(
    flag: NDArray[np.int8],
    s: NDArray[np.float64],
    k: NDArray[np.float64],
    t: NDArray[np.float64],
    r: NDArray[np.float64],
    q: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def bsm_iv(
    target_price: NDArray[np.float64],
    flag: NDArray[np.int8],
    s: NDArray[np.float64],
    k: NDArray[np.float64],
    t: NDArray[np.float64],
    r: NDArray[np.float64],
    q: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def bsm_greeks(
    flag: NDArray[np.int8],
    s: NDArray[np.float64],
    k: NDArray[np.float64],
    t: NDArray[np.float64],
    r: NDArray[np.float64],
    q: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]: ...
def black76_price(
    flag: NDArray[np.int8],
    f: NDArray[np.float64],
    k: NDArray[np.float64],
    t: NDArray[np.float64],
    r: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def black76_delta(
    flag: NDArray[np.int8],
    f: NDArray[np.float64],
    k: NDArray[np.float64],
    t: NDArray[np.float64],
    r: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def black76_gamma(
    f: NDArray[np.float64],
    k: NDArray[np.float64],
    t: NDArray[np.float64],
    r: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def black76_vega(
    f: NDArray[np.float64],
    k: NDArray[np.float64],
    t: NDArray[np.float64],
    r: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def black76_theta(
    flag: NDArray[np.int8],
    f: NDArray[np.float64],
    k: NDArray[np.float64],
    t: NDArray[np.float64],
    r: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def black76_rho(
    flag: NDArray[np.int8],
    f: NDArray[np.float64],
    k: NDArray[np.float64],
    t: NDArray[np.float64],
    r: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def black76_iv(
    target_price: NDArray[np.float64],
    flag: NDArray[np.int8],
    f: NDArray[np.float64],
    k: NDArray[np.float64],
    t: NDArray[np.float64],
    r: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def black76_greeks(
    flag: NDArray[np.int8],
    f: NDArray[np.float64],
    k: NDArray[np.float64],
    t: NDArray[np.float64],
    r: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]: ...

# Scalar fast-path twins (see module docstring).

def bsm_price_scalar(
    flag: int, s: _Scalar, k: _Scalar, t: _Scalar, r: _Scalar, q: _Scalar, sigma: _Scalar
) -> float: ...
def bsm_delta_scalar(
    flag: int, s: _Scalar, k: _Scalar, t: _Scalar, r: _Scalar, q: _Scalar, sigma: _Scalar
) -> float: ...
def bsm_gamma_scalar(
    s: _Scalar, k: _Scalar, t: _Scalar, r: _Scalar, q: _Scalar, sigma: _Scalar
) -> float: ...
def bsm_vega_scalar(
    s: _Scalar, k: _Scalar, t: _Scalar, r: _Scalar, q: _Scalar, sigma: _Scalar
) -> float: ...
def bsm_theta_scalar(
    flag: int, s: _Scalar, k: _Scalar, t: _Scalar, r: _Scalar, q: _Scalar, sigma: _Scalar
) -> float: ...
def bsm_rho_scalar(
    flag: int, s: _Scalar, k: _Scalar, t: _Scalar, r: _Scalar, q: _Scalar, sigma: _Scalar
) -> float: ...
def bsm_iv_scalar(
    target_price: _Scalar, flag: int, s: _Scalar, k: _Scalar, t: _Scalar, r: _Scalar, q: _Scalar
) -> float: ...
def bsm_greeks_scalar(
    flag: int, s: _Scalar, k: _Scalar, t: _Scalar, r: _Scalar, q: _Scalar, sigma: _Scalar
) -> tuple[float, float, float, float, float]: ...
def black76_price_scalar(
    flag: int, f: _Scalar, k: _Scalar, t: _Scalar, r: _Scalar, sigma: _Scalar
) -> float: ...
def black76_delta_scalar(
    flag: int, f: _Scalar, k: _Scalar, t: _Scalar, r: _Scalar, sigma: _Scalar
) -> float: ...
def black76_gamma_scalar(
    f: _Scalar, k: _Scalar, t: _Scalar, r: _Scalar, sigma: _Scalar
) -> float: ...
def black76_vega_scalar(
    f: _Scalar, k: _Scalar, t: _Scalar, r: _Scalar, sigma: _Scalar
) -> float: ...
def black76_theta_scalar(
    flag: int, f: _Scalar, k: _Scalar, t: _Scalar, r: _Scalar, sigma: _Scalar
) -> float: ...
def black76_rho_scalar(
    flag: int, f: _Scalar, k: _Scalar, t: _Scalar, r: _Scalar, sigma: _Scalar
) -> float: ...
def black76_iv_scalar(
    target_price: _Scalar, flag: int, f: _Scalar, k: _Scalar, t: _Scalar, r: _Scalar
) -> float: ...
def black76_greeks_scalar(
    flag: int, f: _Scalar, k: _Scalar, t: _Scalar, r: _Scalar, sigma: _Scalar
) -> tuple[float, float, float, float, float]: ...
