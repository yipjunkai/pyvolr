"""Single-library hot-loop driver for `perf stat` (and other samplers).

The comparison scripts measure every library through an adaptive timing harness
— great for wall-clock tables, useless for `perf stat`, which counts the WHOLE
process (every library, every size, imports, JIT). This driver runs ONE
library's `price` / `iv` / `greeks` call at one array size in a tight loop, after
a warmup, so a profiler wrapped around the process sees essentially that kernel.

Intended entry point (Linux; wraps this in perf, single-thread + core-pinned).
Recipe args are positional — lib, workload, n:

    just perf-stat pyvolr iv 100000

Or directly, from inside the right uv environment:

    python bench/profile_one.py pyvolr iv 100000 --seconds 5

Libraries: pyvolr, opengreeks, fast-vollib-numba, fast-vollib-numpy, vollib
— the modern env's full price/iv/greeks set (reuses compare_new_entrants'
adapters). Run the SAME workload + n for each and diff the counter blocks.

Caveat: `perf stat` counts the whole process, so the warmup (imports, and
fast-vollib's one-time numba JIT ~0.4 s) is included. Keep the loop long
(--seconds) so it dominates, and read the ratios (IPC, cache-miss %, branch-miss
%) rather than raw totals — they're robust to a little startup noise.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

# Run as `python bench/profile_one.py`, so bench/ is sys.path[0] and the
# adapters (and shared workload constants) import directly.
from compare_new_entrants import (
    K1,
    SIGMA,
    R,
    S,
    T,
    _strikes,
    adapter_fast_vollib_numba,
    adapter_fast_vollib_numpy,
    adapter_opengreeks,
    adapter_pyvolr,
    adapter_vollib,
)

ADAPTERS = {
    "pyvolr": adapter_pyvolr,
    "opengreeks": adapter_opengreeks,
    "fast-vollib-numba": adapter_fast_vollib_numba,
    "fast-vollib-numpy": adapter_fast_vollib_numpy,
    "vollib": adapter_vollib,
}


def build_call(lib: str, workload: str, n: int):
    """Return (display_name, zero-arg callable) for one library's hot path."""
    make = ADAPTERS.get(lib)
    if make is None:
        sys.exit(f"unknown lib {lib!r}; choose from: {', '.join(ADAPTERS)}")
    name, fns, _cap = make()
    key = f"{workload}_{'scalar' if n == 1 else 'vec'}"
    fn = fns.get(key)
    if fn is None:
        sys.exit(f"{name} has no {key!r} path")

    if workload == "iv":
        # Invert prices produced by pyvolr's golden-pinned forward map, so every
        # library inverts the same well-posed targets.
        from pyvolr import bs as pv

        if n == 1:
            price = float(pv.price("c", S=S, K=K1, T=T, r=R, sigma=SIGMA))
            return name, lambda: fn(price, K1)
        ks = _strikes(n)
        prices = np.asarray(pv.price("c", S=S, K=ks, T=T, r=R, sigma=SIGMA))
        return name, lambda: fn(prices, ks)

    if n == 1:
        return name, lambda: fn(K1)
    ks = _strikes(n)
    return name, lambda: fn(ks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "lib", help="pyvolr | opengreeks | fast-vollib-numba | fast-vollib-numpy | vollib"
    )
    ap.add_argument("workload", choices=["price", "iv", "greeks"])
    ap.add_argument("n", type=int, help="array size (1 = scalar)")
    ap.add_argument("--seconds", type=float, default=5.0, help="hot-loop duration (default 5)")
    ap.add_argument("--warmup", type=int, default=3, help="untimed warmup calls (default 3)")
    args = ap.parse_args()

    name, call = build_call(args.lib, args.workload, args.n)

    for _ in range(args.warmup):  # imports + numba JIT + warm caches, before the loop
        call()
    print(
        f"[profile_one] {name} · {args.workload} · N={args.n:,} · warmup done; "
        f"looping ~{args.seconds:.0f}s — wrap this process in `perf stat`",
        flush=True,
    )

    iters = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        call()
        iters += 1
    dt = time.perf_counter() - t0
    print(
        f"[profile_one] {iters:,} iterations in {dt:.2f}s ({iters * args.n / dt:,.0f} rows/s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
