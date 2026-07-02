"""Internal FFI wrapper helpers shared by every Python-facing pricing module.

The Rust core (`pyvolr._core`) has two surfaces: array entry points that
accept flat 1-D contiguous arrays of equal length, and `*_scalar` twins that
take plain numeric scalars (same kernels, bit-for-bit). This module owns the
contract between both surfaces and the public numpy-broadcasting API. Every
`pyvolr.<model>` module must route through these helpers — do not reimplement
broadcasting, flag normalization, the scalar-collapse rule, or the fast-path
eligibility rule (`SCALAR_NUMERIC` / `scalar_flag_or_none`) anywhere else.

The leading underscore marks this as an internal module; downstream users
should depend on `pyvolr.bs` / `pyvolr.black76`, not on the helpers here.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyvolr.exceptions import ImpliedVolError, ImpliedVolWarning

if TYPE_CHECKING:
    from typing import Literal, TypeAlias

    import pandas as pd

__all__ = [
    "SCALAR_NUMERIC",
    "FlagInput",
    "Formatted",
    "Greeks",
    "GreeksResult",
    "OnError",
    "Result",
    "ReturnAs",
    "apply_on_error",
    "apply_on_error_scalar",
    "broadcast_f64",
    "format_result",
    "normalize_flag",
    "scalar_flag_or_none",
    "scalar_or_array",
]

FlagInput = ArrayLike | str
Result = float | NDArray[np.float64]

# `return_as` selects the output container shared by every public pricing
# function. These are string forward-ref aliases so the `pd.DataFrame` member
# stays type-check-only — pandas is never imported at runtime unless a caller
# actually asks for `return_as="dataframe"`.
ReturnAs: TypeAlias = "Literal['numpy', 'dict', 'dataframe'] | None"
Formatted: TypeAlias = "Result | dict[str, Result] | pd.DataFrame"
GreeksResult: TypeAlias = "Greeks | pd.DataFrame"

# `on_error` governs how `implied_vol` reacts to an unsolvable input (NaN result).
OnError: TypeAlias = "Literal['warn', 'raise', 'ignore']"


class Greeks(TypedDict):
    """The five standard Greeks; each value is a float or ndarray per the input shape.

    Re-exported as `pyvolr.bs.Greeks` / `pyvolr.black76.Greeks` for user
    annotations.
    """

    delta: Result
    gamma: Result
    theta: Result
    vega: Result
    rho: Result


# Types eligible for the scalar fast path. np.floating / np.integer scalars
# convert to f64 exactly as the array path's `asarray(..., dtype=float64)`
# does, so routing them through the scalar FFI is value-identical. bool passes
# as int (True -> 1.0 on both paths). Everything else — 0-d arrays, lists,
# strings — falls through to the array path, which owns their error behavior.
# The tuple[type, ...] annotation deliberately disables isinstance narrowing:
# bare np.floating would narrow to floating[Unknown] under strict pyright and
# pollute the array-path branch. The runtime guard proves scalarness; the one
# FFI call site carries a targeted ignore instead.
SCALAR_NUMERIC: tuple[type, ...] = (float, int, np.floating, np.integer)


def scalar_flag_or_none(flag: FlagInput) -> int | None:
    """Return ±1 for a scalar string flag, or None to defer to the array path.

    Raises the same ValueError as `normalize_flag` for an invalid string, so
    the fast path and the array path are indistinguishable on bad flags.
    Callers must check numeric-input eligibility BEFORE calling this: on the
    array path a numeric-conversion error fires before flag validation, and
    checking in the same order keeps doubly-invalid inputs error-identical.
    """
    if type(flag) is str:
        if flag in ("c", "C"):
            return 1
        if flag in ("p", "P"):
            return -1
        raise ValueError(f"flag must be 'c' or 'p', got {flag!r}")
    return None


def normalize_flag(flag: FlagInput, shape: tuple[int, ...]) -> NDArray[np.int8]:
    """Convert a flag input into an int8 array of the given broadcast shape.

    Accepts:
        - a string: 'c'/'C' (call) or 'p'/'P' (put)
        - an ndarray of strings
        - an ndarray of numbers, strictly 1 (call) or -1 (put)
    """
    if isinstance(flag, str):
        s = flag.lower()
        if s not in ("c", "p"):
            raise ValueError(f"flag must be 'c' or 'p', got {flag!r}")
        return np.full(shape, 1 if s == "c" else -1, dtype=np.int8)

    arr = np.asarray(flag)
    if arr.dtype.kind in ("U", "S", "O"):
        # dtype=str keeps the full element width. A fixed "U1" here once
        # truncated every entry to its first character BEFORE the validation
        # below, so 'price' silently priced a put and 'cow' a call.
        lower_flat = np.array([str(x).lower() for x in arr.ravel()], dtype=str)
        lower = np.reshape(lower_flat, arr.shape)
        valid = np.isin(lower, ("c", "p"))
        if not valid.all():
            bad = str(lower[~valid].ravel()[0])
            raise ValueError(
                f"flag array must contain only 'c' or 'p' (case-insensitive), got {bad!r}"
            )
        encoded = np.where(lower == "c", 1, -1).astype(np.int8)
        return np.ascontiguousarray(np.broadcast_to(encoded, shape))
    # Strict numeric validation, mirroring the string path. The old bare
    # `astype(np.int8)` silently wrapped out-of-range values (-256 -> 0 ->
    # priced as a CALL despite the minus sign) and mapped NaN to 0 — the
    # financially dangerous silent-wrong-answer class.
    valid = np.isin(arr, (-1, 1))
    if not valid.all():
        bad = arr[~valid].ravel()[0]
        raise ValueError(f"flag values must be 1 (call) or -1 (put), got {bad!r}")
    return np.ascontiguousarray(np.broadcast_to(arr.astype(np.int8), shape))


def broadcast_f64(*arrs: ArrayLike) -> tuple[list[NDArray[np.float64]], tuple[int, ...]]:
    """Broadcast inputs to a common shape; return flat contiguous f64 arrays."""
    cast = [np.asarray(a, dtype=np.float64) for a in arrs]
    bcast = np.broadcast_arrays(*cast)
    shape = bcast[0].shape
    # ravel() already guarantees a contiguous result (it copies exactly when
    # the broadcast view is non-contiguous), so no ascontiguousarray needed.
    flat = [a.ravel() for a in bcast]
    return flat, shape


def scalar_or_array(arr: NDArray[np.float64], shape: tuple[int, ...]) -> Result:
    """If the broadcast shape is () return a Python scalar; otherwise reshape."""
    if shape == ():
        return arr.item()
    return np.reshape(arr, shape)


def format_result(
    columns: dict[str, NDArray[np.float64]],
    shape: tuple[int, ...],
    return_as: str | None,
) -> Formatted:
    """Shape flat Rust outputs into the container requested by ``return_as``.

    ``columns`` maps each output name to its flat 1-D result array (length N).
    The scalar-collapse rule (``scalar_or_array``) applies to every mode except
    ``"dataframe"``, which is inherently tabular (scalar input -> one row):

      - ``None`` / ``"numpy"``: a bare ``Result`` for a single column, otherwise
        a ``dict`` of arrays (a multi-output result has no bare numpy form).
      - ``"dict"``: always a ``{name: Result}`` dict.
      - ``"dataframe"``: a pandas ``DataFrame`` (pandas is an optional
        dependency; see ``_to_dataframe``). Multi-dimensional broadcasts are
        raveled in C order.

    Raises ``ValueError`` for any other ``return_as`` value. The parameter is
    typed ``str | None`` (not the narrower ``ReturnAs``) so this runtime guard
    stays reachable for callers that bypass the type checker.
    """
    if return_as in (None, "numpy"):
        if len(columns) == 1:
            (out,) = columns.values()
            return scalar_or_array(out, shape)
        return {name: scalar_or_array(col, shape) for name, col in columns.items()}
    if return_as == "dict":
        return {name: scalar_or_array(col, shape) for name, col in columns.items()}
    if return_as == "dataframe":
        return _to_dataframe(columns)
    raise ValueError(f"return_as must be 'numpy', 'dict', or 'dataframe', got {return_as!r}")


def _to_dataframe(columns: dict[str, NDArray[np.float64]]) -> pd.DataFrame:
    """Build a DataFrame from flat result columns. pandas is a soft dependency.

    Caught as ``ImportError`` (not ``ModuleNotFoundError``) so the case where a
    test stubs ``sys.modules["pandas"] = None`` is handled alongside a genuine
    missing install; both surface the same actionable ``ModuleNotFoundError``.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ModuleNotFoundError(
            "return_as='dataframe' requires pandas, which is not installed. "
            "Install it with `pip install pandas`, or use return_as='numpy' or 'dict'."
        ) from exc
    return pd.DataFrame({name: np.reshape(col, -1) for name, col in columns.items()})


# The failure modes named in every implied-vol NaN message; shared by
# `apply_on_error` and `apply_on_error_scalar` so the texts stay in lockstep.
_IV_NAN_REASON = "price outside the no-arbitrage bounds, non-positive T/S/K, or non-finite input"


def apply_on_error(out: NDArray[np.float64], on_error: str) -> None:
    """React to NaN failures in a flat implied-vol result per ``on_error``.

    A failed solve (price outside the no-arbitrage bounds, non-positive
    ``T``/``S``/``K``, or non-finite input) is returned by the Rust core as
    ``NaN``; a solvable implied vol is always finite, so ``NaN`` uniquely marks a
    failure. ``"ignore"`` returns silently (today's behaviour), ``"warn"`` emits a
    single ``ImpliedVolWarning`` summarising the failures, and ``"raise"`` raises
    ``ImpliedVolError`` if any element failed. Any other value raises
    ``ValueError`` (mirrors ``format_result``).

    Call on the raw result before ``format_result``. The ``"ignore"`` fast path
    skips the NaN scan entirely, so opted-out callers pay nothing.
    """
    if on_error == "ignore":
        return
    if on_error not in ("warn", "raise"):
        raise ValueError(f"on_error must be 'warn', 'raise', or 'ignore', got {on_error!r}")
    mask = np.isnan(out)
    n_failed = int(mask.sum())
    if n_failed == 0:
        return
    total = out.size
    if total == 1:
        msg = f"implied-vol solve returned NaN: {_IV_NAN_REASON}."
    else:
        first = int(np.argmax(mask))
        msg = (
            f"{n_failed} of {total} implied-vol solves returned NaN "
            f"(first at index {first}): {_IV_NAN_REASON}."
        )
    if on_error == "raise":
        raise ImpliedVolError(msg)
    # stacklevel=3: warnings.warn -> apply_on_error -> implied_vol -> user call site.
    warnings.warn(msg, ImpliedVolWarning, stacklevel=3)


def apply_on_error_scalar(out: float, on_error: str) -> None:
    """Scalar twin of ``apply_on_error`` for the fast path.

    Identical validation order, messages, exception/warning classes, and
    warning depth (both are called directly from ``implied_vol``, so the
    user's call site sits at the same stacklevel).
    """
    if on_error == "ignore":
        return
    if on_error not in ("warn", "raise"):
        raise ValueError(f"on_error must be 'warn', 'raise', or 'ignore', got {on_error!r}")
    if not math.isnan(out):
        return
    msg = f"implied-vol solve returned NaN: {_IV_NAN_REASON}."
    if on_error == "raise":
        raise ImpliedVolError(msg)
    warnings.warn(msg, ImpliedVolWarning, stacklevel=3)
