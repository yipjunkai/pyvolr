#!/usr/bin/env python3
"""Coarse Python-wrapper perf gate: measure or compare public-API timings.

Usage:
  check_python_perf.py measure <out.json>
  check_python_perf.py compare <base.json> <head.json> <max-ratio>

``measure`` times the public API's scalar calls plus a 10k-row vector call
per endpoint family (timeit, min of repeats — the load-robust statistic) and
writes ``{metric: seconds}`` JSON. ``compare`` fails when ``head/base``
exceeds ``max-ratio`` for any shared metric.

Why this exists (TODO §3): ~86-98% of scalar-call latency lives in the
``python/`` wrapper, where the criterion gate (perf.yml) has no visibility —
a wrapper regression would ship with zero perf signal on exactly the axis
the README is judged. The workflow measures base and head wrappers against
the SAME compiled extension, so the ratio is immune to runner speed. The
ceiling is deliberately coarse: the regression class that matters (a broken
dispatch guard re-entering the numpy broadcast machinery) is 15-20x, and
min-of-repeats on a shared runner is stable well within 2x.

Bootstrap behavior: ``compare`` exits 0 with a notice when the base
measurement file is missing (the first PR carrying this gate, or a base
wrapper that could not run against the head-built extension).
"""

from __future__ import annotations

import json
import sys
import timeit
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable


def _build_cases() -> dict[str, Callable[[], object]]:
    # Imported lazily so `compare` needs nothing beyond the standard library.
    import numpy as np

    from pyvolr import bs

    n = 10_000
    rng = np.random.default_rng(7)
    spots = rng.uniform(50.0, 150.0, n)
    scalar_target = bs.price("c", 100.0, 105.0, 0.25, 0.05, 0.2)
    vector_targets = bs.price("c", spots, 105.0, 0.25, 0.05, 0.2)

    return {
        "scalar_price": lambda: bs.price("c", 100.0, 105.0, 0.25, 0.05, 0.2),
        "scalar_greeks": lambda: bs.greeks("c", 100.0, 105.0, 0.25, 0.05, 0.2),
        "scalar_iv": lambda: bs.implied_vol(scalar_target, "c", 100.0, 105.0, 0.25, 0.05),
        "vector_price_10k": lambda: bs.price("c", spots, 105.0, 0.25, 0.05, 0.2),
        "vector_iv_10k": lambda: bs.implied_vol(vector_targets, "c", spots, 105.0, 0.25, 0.05),
    }


def measure(out_path: str) -> int:
    results: dict[str, float] = {}
    for name, fn in _build_cases().items():
        timer = timeit.Timer(fn)
        number, _ = timer.autorange()
        results[name] = min(timer.repeat(repeat=5, number=number)) / number
        print(f"{name}: {results[name] * 1e9:,.0f} ns")
    Path(out_path).write_text(json.dumps(results, indent=2))
    return 0


def compare(base_path: str, head_path: str, max_ratio: float) -> int:
    base_file = Path(base_path)
    if not base_file.is_file():
        print(f"::notice::No base measurements at {base_path}; bootstrapping this run.")
        return 0
    base = cast("dict[str, float]", json.loads(base_file.read_text()))
    head = cast("dict[str, float]", json.loads(Path(head_path).read_text()))

    failures: list[str] = []
    for name, base_s in sorted(base.items()):
        head_s = head.get(name)
        if head_s is None:
            print(f"::notice::Metric {name} missing from head; skipping.")
            continue
        ratio = head_s / base_s
        print(f"{name}: base {base_s * 1e9:,.0f} ns -> head {head_s * 1e9:,.0f} ns  (x{ratio:.2f})")
        if ratio > max_ratio:
            failures.append(f"{name} regressed x{ratio:.2f} (ceiling x{max_ratio:.2f})")
    if failures:
        for failure in failures:
            print(f"::error::{failure}")
        return 1
    print(f"OK: no wrapper metric regressed beyond x{max_ratio:.2f}.")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[1] == "measure":
        return measure(argv[2])
    if len(argv) >= 5 and argv[1] == "compare":
        return compare(argv[2], argv[3], float(argv[4]))
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
