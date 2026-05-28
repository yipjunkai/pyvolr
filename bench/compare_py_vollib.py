"""Benchmark pyvolr against py_vollib.

Usage:
    uv venv --python 3.10 .bench-venv
    source .bench-venv/bin/activate
    uv pip install pyvolr py_vollib numpy
    python bench/compare_py_vollib.py

Python 3.10 / 3.11 are required because py_vollib's transitive dependency
`py_lets_be_rational` imports `DBL_MIN` / `DBL_MAX` from CPython's internal
`_testcapi` module, which isn't shipped on 3.12+. See docs/why.md.

Some 3.10 distributions (notably uv-managed) also don't ship `_testcapi`.
If `from py_lets_be_rational ...` fails with `ModuleNotFoundError: _testcapi`,
patch the installed library with the two-line fix:

    # in <site-packages>/py_lets_be_rational/constants.py replace:
    #   from _testcapi import DBL_MIN, DBL_MAX
    # with:
    from sys import float_info as _fi
    DBL_MIN, DBL_MAX = _fi.min, _fi.max
"""

from __future__ import annotations

import statistics
import time

import numpy as np
from py_vollib.black import black as pvol_black
from py_vollib.black.implied_volatility import (
    implied_volatility as pvol_black_iv,
)
from py_vollib.black_scholes import black_scholes as pvol_price
from py_vollib.black_scholes.greeks.analytical import (
    delta as pvol_delta,
)
from py_vollib.black_scholes.greeks.analytical import (
    gamma as pvol_gamma,
)
from py_vollib.black_scholes.greeks.analytical import (
    rho as pvol_rho,
)
from py_vollib.black_scholes.greeks.analytical import (
    theta as pvol_theta,
)
from py_vollib.black_scholes.greeks.analytical import (
    vega as pvol_vega,
)
from py_vollib.black_scholes.implied_volatility import (
    implied_volatility as pvol_iv,
)

from pyvolr import black76 as pvb
from pyvolr import bs as pv


def bench(fn, reps: int, warmup: int = 3) -> float:
    """Median wall time (seconds) of `reps` calls to `fn()`."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def fmt(secs: float) -> str:
    if secs < 1e-3:
        return f"{secs * 1e6:.1f} us"
    if secs < 1.0:
        return f"{secs * 1e3:.2f} ms"
    return f"{secs:.2f} s"


S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.05, 0.20
strikes_1k = np.linspace(80.0, 120.0, 1_000)
strikes_10k = np.linspace(80.0, 120.0, 10_000)
strikes_100k = np.linspace(80.0, 120.0, 100_000)
strikes_1m = np.linspace(80.0, 120.0, 1_000_000)

ref_price = pv.price("c", S=S, K=K, T=T, r=r, sigma=sigma)
# Per-row reference prices for the vectorised IV scenario below. py_vollib
# has no vectorised IV API, so the comparison loops scalar calls (10k of them
# per repetition — slow, hence the low `reps` count in that scenario).
strikes_10k_prices = pv.price("c", S=S, K=strikes_10k, T=T, r=r, sigma=sigma)


def pvol_array_price(strikes):
    return np.array([pvol_price("c", S, k, T, r, sigma) for k in strikes])


def pvol_array_iv(targets, strikes):
    return [pvol_iv(p, S, k, T, r, "c") for p, k in zip(targets, strikes, strict=True)]


def pvol_array_greeks(strikes):
    return (
        [pvol_delta("c", S, k, T, r, sigma) for k in strikes],
        [pvol_gamma("c", S, k, T, r, sigma) for k in strikes],
        [pvol_theta("c", S, k, T, r, sigma) for k in strikes],
        [pvol_vega("c", S, k, T, r, sigma) for k in strikes],
        [pvol_rho("c", S, k, T, r, sigma) for k in strikes],
    )


# Black-76: same inputs but the underlying is the forward F. Use F == S so the
# magnitudes match the BSM rows above and the speedup numbers are directly
# comparable. ref_b76_price is the Black-76 price used for IV inversion.
ref_b76_price = pvb.price("c", F=S, K=K, T=T, r=r, sigma=sigma)


def pvol_black_array_price(strikes):
    return np.array([pvol_black("c", S, k, T, r, sigma) for k in strikes])


scenarios = [
    (
        "price, scalar",
        lambda: pv.price("c", S=S, K=K, T=T, r=r, sigma=sigma),
        lambda: pvol_price("c", S, K, T, r, sigma),
        20_000,
    ),
    (
        "price, 1k strikes",
        lambda: pv.price("c", S=S, K=strikes_1k, T=T, r=r, sigma=sigma),
        lambda: pvol_array_price(strikes_1k),
        200,
    ),
    (
        "price, 10k strikes",
        lambda: pv.price("c", S=S, K=strikes_10k, T=T, r=r, sigma=sigma),
        lambda: pvol_array_price(strikes_10k),
        50,
    ),
    (
        "price, 100k strikes",
        lambda: pv.price("c", S=S, K=strikes_100k, T=T, r=r, sigma=sigma),
        lambda: pvol_array_price(strikes_100k),
        10,
    ),
    (
        "price, 1M strikes",
        lambda: pv.price("c", S=S, K=strikes_1m, T=T, r=r, sigma=sigma),
        lambda: pvol_array_price(strikes_1m),
        3,
    ),
    (
        "all 5 Greeks, 10k strikes",
        lambda: pv.greeks("c", S=S, K=strikes_10k, T=T, r=r, sigma=sigma),
        lambda: pvol_array_greeks(strikes_10k),
        20,
    ),
    (
        "implied_vol, scalar",
        lambda: pv.implied_vol(price=ref_price, flag="c", S=S, K=K, T=T, r=r),
        lambda: pvol_iv(ref_price, S, K, T, r, "c"),
        5_000,
    ),
    (
        # py_vollib has no vectorised IV — comparison column is 10k scalar
        # calls per rep, so keep `reps` low to avoid blowing wall time.
        "implied_vol, 10k strikes",
        lambda: pv.implied_vol(price=strikes_10k_prices, flag="c", S=S, K=strikes_10k, T=T, r=r),
        lambda: pvol_array_iv(strikes_10k_prices, strikes_10k),
        5,
    ),
    # Black-76 rows: structurally identical machinery (the Rust core delegates
    # to bsm::price with q=r), so expect speedups very close to the BSM rows.
    (
        "black76 price, scalar",
        lambda: pvb.price("c", F=S, K=K, T=T, r=r, sigma=sigma),
        lambda: pvol_black("c", S, K, T, r, sigma),
        20_000,
    ),
    (
        "black76 price, 10k strikes",
        lambda: pvb.price("c", F=S, K=strikes_10k, T=T, r=r, sigma=sigma),
        lambda: pvol_black_array_price(strikes_10k),
        50,
    ),
    (
        "black76 implied_vol, scalar",
        lambda: pvb.implied_vol(price=ref_b76_price, flag="c", F=S, K=K, T=T, r=r),
        # py_vollib.black IV: (price, F, K, r, t, flag) — r and t swapped vs BSM.
        lambda: pvol_black_iv(ref_b76_price, S, K, r, T, "c"),
        5_000,
    ),
]


def main() -> None:
    print(f"{'scenario':<30} {'pyvolr':>14} {'py_vollib':>14} {'speedup':>10}")
    print("-" * 72)
    for name, pv_fn, pvol_fn, reps in scenarios:
        pv_t = bench(pv_fn, reps)
        pvol_t = bench(pvol_fn, reps)
        speedup = pvol_t / pv_t
        print(f"{name:<30} {fmt(pv_t):>14} {fmt(pvol_t):>14} {speedup:>9.1f}x")


if __name__ == "__main__":
    main()
