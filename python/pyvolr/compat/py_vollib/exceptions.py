"""Mirror of py_vollib's implied-volatility exceptions.

py_vollib raises these from `implied_volatility` when the price is outside the
no-arbitrage bounds — the one *intentional* error contract in py_vollib (the
classes originate in Jäckel's `py_lets_be_rational`). Every other py_vollib
"raise" on bad input (`ZeroDivisionError` at K=0, `ValueError: cannot convert
float NaN to integer` for Greeks at t=0, `KeyError` on a bad flag) is a leaked
implementation artifact, not a contract — pyvolr deliberately does not
reproduce those (see `pyvolr.compat.py_vollib`'s module docstring).

The originals live in `py_lets_be_rational.exceptions`, which is unimportable
on Python 3.12+ (the `_testcapi` bug pyvolr exists to fix), so a migrating
`except` clause must re-point its import here:

    # Before
    from py_lets_be_rational.exceptions import BelowIntrinsicException
    # After
    from pyvolr.compat.py_vollib.exceptions import BelowIntrinsicException

Names, messages, hierarchy, and the `.value` sentinels (±DBL_MAX) match the
originals so existing handlers keep working unchanged.
"""

from __future__ import annotations

import sys

__all__ = [
    "AboveMaximumException",
    "BelowIntrinsicException",
    "VolatilityValueException",
]

# py_lets_be_rational stamps these sentinels onto the exception's `.value`.
_BELOW_INTRINSIC: float = -sys.float_info.max
_ABOVE_MAXIMUM: float = sys.float_info.max


class VolatilityValueException(Exception):
    """Base for an implied volatility that falls outside the valid range."""

    def __init__(self, message: str, value: float) -> None:
        super().__init__(message)
        self.value = value


class BelowIntrinsicException(VolatilityValueException):
    """The option price is below its intrinsic value (no real implied vol)."""

    def __init__(self) -> None:
        super().__init__("The volatility is below the intrinsic value.", _BELOW_INTRINSIC)


class AboveMaximumException(VolatilityValueException):
    """The option price exceeds the no-arbitrage maximum (no real implied vol)."""

    def __init__(self) -> None:
        super().__init__("The volatility is above the maximum value.", _ABOVE_MAXIMUM)
