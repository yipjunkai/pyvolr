"""Output formatting + pandas soft-dependency handling for the vectorized shim.

Mirrors py_vollib_vectorized's ``return_as`` conventions exactly:

  - single-output (pricing / IV / one greek):
      ``"series"``    -> ``pd.Series(arr, name=NAME)``
      ``"dataframe"`` -> ``pd.DataFrame(arr, columns=[NAME])``
      anything else   -> the raw numpy array
  - multi-output (``get_all_greeks``):
      ``"dataframe"`` -> ``pd.DataFrame.from_dict(cols)``
      ``"json"``      -> ``json.dumps({col: list})`` (stdlib; never needs pandas)
      anything else   -> a ``dict`` of numpy arrays

pandas is a soft dependency: a ``"series"`` / ``"dataframe"`` request with pandas
absent falls back to the numpy / dict form with a one-time warning. Like pvv, an
unrecognised ``return_as`` is not an error — it falls through to numpy / dict.
"""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import TypeAlias

    import pandas as pd
    from numpy.typing import ArrayLike, DTypeLike, NDArray

__all__ = ["MultiResult", "SingleResult", "as_1d", "format_multi", "format_single"]

SingleResult: TypeAlias = "NDArray[np.float64] | pd.Series | pd.DataFrame"
MultiResult: TypeAlias = "dict[str, NDArray[np.float64]] | pd.DataFrame | str"


def as_1d(x: ArrayLike, dtype: DTypeLike = np.float64) -> NDArray[np.float64]:
    """Coerce a native result to a flat 1-D array of ``dtype``.

    py_vollib_vectorized always returns arrays (scalars broadcast to length 1) and
    never collapses to a scalar, unlike pyvolr's native API — so flatten and keep
    at least one dimension to match.
    """
    return np.ravel(np.asarray(x, dtype=dtype))


def format_single(arr: NDArray[np.float64], name: str, return_as: str) -> SingleResult:
    """Shape a single-output result per py_vollib_vectorized's ``return_as`` rules."""
    if return_as in ("series", "dataframe"):
        try:
            import pandas as pd
        except ImportError:
            # stacklevel=3: warn -> format_single -> the vectorized_* wrapper -> caller.
            warnings.warn(
                f"pandas is not installed; returning a numpy array instead of a "
                f"{return_as}. Install it with `pip install pyvolr[pandas]` for "
                f"{return_as} output.",
                stacklevel=3,
            )
            return arr
        if return_as == "series":
            return pd.Series(arr, name=name)
        return pd.DataFrame(arr, columns=[name])
    return arr


def format_multi(cols: dict[str, NDArray[np.float64]], return_as: str) -> MultiResult:
    """Shape a multi-output result (``get_all_greeks``) per its ``return_as`` rules."""
    if return_as == "dataframe":
        try:
            import pandas as pd
        except ImportError:
            warnings.warn(
                "pandas is not installed; returning a dict instead of a dataframe. "
                "Install it with `pip install pyvolr[pandas]` for dataframe output.",
                stacklevel=3,
            )
            return cols
        return pd.DataFrame(cols)
    if return_as == "json":
        return json.dumps({k: np.asarray(v).tolist() for k, v in cols.items()})
    return cols
