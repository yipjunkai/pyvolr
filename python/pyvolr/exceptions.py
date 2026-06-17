"""Public exceptions and warnings for pyvolr's native API.

Raised / emitted by ``implied_vol`` (``pyvolr.bs`` and ``pyvolr.black76``)
according to its ``on_error`` argument when a solve has no real answer — the
price is outside the no-arbitrage bounds, ``T`` / ``S`` / ``K`` is non-positive,
or an input is non-finite. Those cases produce ``NaN``; ``on_error`` decides
whether that is raised (``ImpliedVolError``), warned about (``ImpliedVolWarning``),
or returned silently.

Distinct from ``pyvolr.compat.py_vollib.exceptions`` — those mirror py_vollib's
own ``BelowIntrinsicException`` / ``AboveMaximumException`` contract and are
scoped to the drop-in compatibility layer.
"""

from __future__ import annotations

__all__ = ["ImpliedVolError", "ImpliedVolWarning"]


class ImpliedVolError(ValueError):
    """An implied-volatility solve had no real answer (``on_error="raise"``).

    Subclasses ``ValueError`` so existing ``except ValueError`` handlers keep
    working unchanged.
    """


class ImpliedVolWarning(UserWarning):
    """Warns that an implied-volatility solve returned ``NaN`` (``on_error="warn"``)."""
