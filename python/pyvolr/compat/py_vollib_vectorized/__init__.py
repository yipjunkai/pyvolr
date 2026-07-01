"""Drop-in compatibility shim for `py_vollib_vectorized`.

Replace your imports:

    # Before
    from py_vollib_vectorized import vectorized_black_scholes, vectorized_implied_volatility

    # After
    from pyvolr.compat.py_vollib_vectorized import vectorized_black_scholes, vectorized_implied_volatility

Signatures, argument orders (including the quirks — `flag` after `r` in
`vectorized_implied_volatility`; `r`/`t` swapped in
`vectorized_implied_volatility_black`), `return_as` values (`"series"` /
`"dataframe"` / else-numpy for single outputs; `"dataframe"` / `"json"` /
else-dict for `get_all_greeks`), and the `"Price"` / `"IV"` / greek column names
match py_vollib_vectorized. Greeks use py_vollib's unit conventions (per-day
theta, per-1% vega/rho).

Two deliberate divergences:
  - **No import-time monkeypatch.** py_vollib_vectorized patches py_vollib on
    import so its scalar entry points accept arrays. pyvolr owns its compat tree
    and exposes the vectorized functions directly, so importing this module has no
    global side effects — changing your import line is the whole migration.
  - **pandas is a soft dependency** (`pip install pyvolr[pandas]`). A DataFrame /
    Series request with pandas absent falls back to numpy with a one-time warning;
    `price_dataframe` requires pandas (its input is a DataFrame).

Values come from pyvolr's native vectorized core; greeks are analytical (vs
py_vollib_vectorized's numerical), so they agree to finite-difference tolerance.
"""

from __future__ import annotations

from .api import get_all_greeks, price_dataframe
from .greeks import delta as vectorized_delta
from .greeks import gamma as vectorized_gamma
from .greeks import rho as vectorized_rho
from .greeks import theta as vectorized_theta
from .greeks import vega as vectorized_vega
from .implied_volatility import (
    vectorized_implied_volatility,
    vectorized_implied_volatility_black,
)
from .models import (
    vectorized_black,
    vectorized_black_scholes,
    vectorized_black_scholes_merton,
)

__all__ = [
    "get_all_greeks",
    "price_dataframe",
    "vectorized_black",
    "vectorized_black_scholes",
    "vectorized_black_scholes_merton",
    "vectorized_delta",
    "vectorized_gamma",
    "vectorized_implied_volatility",
    "vectorized_implied_volatility_black",
    "vectorized_rho",
    "vectorized_theta",
    "vectorized_vega",
]
