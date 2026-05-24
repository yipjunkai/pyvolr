"""Drop-in shim for `py_vollib.black` (Black-76 for futures/forward options).

Mirrors py_vollib's `black(flag, F, K, t, r, sigma)` signature exactly,
returning a Python float so the contract is identical to the reference.
"""

from __future__ import annotations

from pyvolr import black76 as _b76

__all__ = ["black"]


def black(flag: str, F: float, K: float, t: float, r: float, sigma: float) -> float:
    """European Black-76 option price on a futures/forward `F`."""
    return float(_b76.price(flag, F, K, t, r, sigma))
