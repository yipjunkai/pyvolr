"""Type stubs for the pyvolr._core Rust extension.

All functions take flat (1-D, contiguous) numpy arrays of equal length and
return a numpy float64 array of the same length. The Python wrapper in
``pyvolr.bs`` is responsible for broadcasting and reshape.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

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
