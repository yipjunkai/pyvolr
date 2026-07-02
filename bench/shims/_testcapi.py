"""Minimal _testcapi shim for the benchmark's legacy venv only.

py_lets_be_rational 1.0.1 does `from _testcapi import DBL_MIN, DBL_MAX`, and
CPython's internal test module isn't shipped in slim distributions (uv-managed
Pythons, official installers) — the exact bug documented in docs/why.md. These
are the correct values (== sys.float_info.{min,max}); this is precisely the
two-line fix upstream never released. The benchmark justfile puts this on
PYTHONPATH; only the legacy environment actually imports it.
"""

import sys as _sys

DBL_MIN = _sys.float_info.min
DBL_MAX = _sys.float_info.max
