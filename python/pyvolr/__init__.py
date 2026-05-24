"""pyvolr: Black-Scholes-Merton pricing, Greeks, and implied volatility.

Modern Rust-cored replacement for the abandoned py_vollib library.

Quick start:

    >>> from pyvolr import bs
    >>> bs.price("c", S=100, K=105, T=0.5, r=0.05, sigma=0.2)
    3.2240...

For a drop-in py_vollib replacement, use the compatibility shim:

    >>> from pyvolr.compat.py_vollib.black_scholes import black_scholes
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from pyvolr import bs as bs

try:
    __version__ = _pkg_version("pyvolr")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

__all__ = ["__version__", "bs"]
