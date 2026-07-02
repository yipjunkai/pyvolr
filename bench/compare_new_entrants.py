"""Trigger-evaluation benchmark: pyvolr vs the 2026 entrants (price, IV, greeks).

One-command runner: `just table` (env pins + setup live in the repo justfile).
Runs in the modern venv (.venv-bench-entrants — pyvolr, vollib, opengreeks,
fast-vollib[numba]):

    .venv-bench-entrants/bin/python bench/compare_new_entrants.py

Measures median wall time for an option-chain-shaped workload (K varies,
everything else scalar) at N in {1, 1k, 10k, 100k, 1M} across three workloads:

  * price  — European BSM call price
  * iv     — implied vol from target prices (prices precomputed, untimed)
  * greeks — all five first-order Greeks for the chain in one logical call;
             libraries without a bundled kernel pay their real per-Greek cost

This is the TODO §1 competitor-pressure trigger evaluation (opengreeks,
fast-vollib), with vollib 1.0.11 as the resurrected-upstream scalar reference —
a decision artifact, not the README chart (see compare_competitors.py for
that). JIT warmup (fast-vollib's numba backend) is absorbed by the discarded
warmup call in bench(), which is deliberately generous to them.

Writes bench/.new_entrants_results.json:
    {library: {workload: {str(N): seconds}}}
"""

from __future__ import annotations

import json
import statistics
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

warnings.filterwarnings("ignore")  # numba typeof FutureWarning spam on broadcast inputs

RESULTS_PATH = Path(__file__).resolve().parent / ".new_entrants_results.json"

S, K1, T, R, SIGMA = 100.0, 105.0, 0.5, 0.05, 0.20
SIZES = [1, 1_000, 10_000, 100_000, 1_000_000]
WORKLOADS = ("price", "iv", "greeks")


def _strikes(n: int) -> np.ndarray:
    return np.linspace(80.0, 120.0, n)


def bench(fn: Callable[[], object], *, target_seconds: float, max_reps: int = 20_000) -> float:
    """Median wall time of fn(); one discarded warmup sizes the rep count."""
    t0 = time.perf_counter()
    fn()
    warm = max(time.perf_counter() - t0, 1e-6)
    reps = min(max_reps, max(3, int(target_seconds / warm)))
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def fmt(secs: float) -> str:
    if secs < 1e-3:
        return f"{secs * 1e6:8.1f} us"
    if secs < 1.0:
        return f"{secs * 1e3:8.2f} ms"
    return f"{secs:8.2f} s"


# ----------------------------------------------------------------------------
# Adapters. Each returns (name, fns, cap) where fns maps
#   price_scalar(k) / price_vec(ks)
#   iv_scalar(price, k) / iv_vec(prices, ks)
#   greeks_scalar(k) / greeks_vec(ks)
# Missing keys mean "no such path". `cap` limits N for scalar-loop libraries.
# ----------------------------------------------------------------------------


def adapter_pyvolr():
    import pyvolr
    from pyvolr import bs as pv

    fns = {
        "price_scalar": lambda k: pv.price("c", S=S, K=k, T=T, r=R, sigma=SIGMA),
        "price_vec": lambda ks: pv.price("c", S=S, K=ks, T=T, r=R, sigma=SIGMA),
        "iv_scalar": lambda p, k: pv.implied_vol(p, "c", S=S, K=k, T=T, r=R),
        "iv_vec": lambda ps, ks: pv.implied_vol(ps, "c", S=S, K=ks, T=T, r=R),
        "greeks_scalar": lambda k: pv.greeks("c", S=S, K=k, T=T, r=R, sigma=SIGMA),
        "greeks_vec": lambda ks: pv.greeks("c", S=S, K=ks, T=T, r=R, sigma=SIGMA),
    }
    return f"pyvolr {pyvolr.__version__}", fns, None


def adapter_opengreeks():
    from importlib.metadata import version as _pkg_version

    from opengreeks import black_scholes as og

    # *_array takes ndarrays for S/K/t/sigma and a scalar r, with no scalar
    # broadcasting; the np.full expansion is the user-visible cost of a chain
    # call, so it stays inside the timed region. The greeks workload reuses
    # one set of expanded arrays across the five per-Greek calls (there is no
    # bundled all-Greeks kernel) — the cheapest call pattern available.

    def price_vec(ks):
        n = ks.size
        return og.black_scholes_array("c", np.full(n, S), ks, np.full(n, T), R, np.full(n, SIGMA))

    def iv_vec(ps, ks):
        n = ks.size
        return og.implied_volatility_array(ps, np.full(n, S), ks, np.full(n, T), R, "c")

    def greeks_vec(ks):
        n = ks.size
        s_arr, t_arr, sig_arr = np.full(n, S), np.full(n, T), np.full(n, SIGMA)
        return (
            og.delta_array("c", s_arr, ks, t_arr, R, sig_arr),
            og.gamma_array("c", s_arr, ks, t_arr, R, sig_arr),
            og.vega_array("c", s_arr, ks, t_arr, R, sig_arr),
            og.theta_array("c", s_arr, ks, t_arr, R, sig_arr),
            og.rho_array("c", s_arr, ks, t_arr, R, sig_arr),
        )

    fns = {
        "price_scalar": lambda k: og.black_scholes("c", S, k, T, R, SIGMA),
        "price_vec": price_vec,
        "iv_scalar": lambda p, k: og.implied_volatility(p, S, k, T, R, "c"),
        "iv_vec": iv_vec,
        "greeks_scalar": lambda k: (
            og.delta("c", S, k, T, R, SIGMA),
            og.gamma("c", S, k, T, R, SIGMA),
            og.vega("c", S, k, T, R, SIGMA),
            og.theta("c", S, k, T, R, SIGMA),
            og.rho("c", S, k, T, R, SIGMA),
        ),
        "greeks_vec": greeks_vec,
    }
    return f"opengreeks {_pkg_version('opengreeks')}", fns, None


def _adapter_fast_vollib(backend):
    from importlib.metadata import version as _pkg_version

    import fast_vollib as fv

    if backend == "numba":
        import numba  # noqa: F401 — raise (=> clean skip) when the extra isn't installed

    fns = {
        "price_scalar": lambda k: fv.fast_black_scholes(
            "c", S, k, T, R, SIGMA, return_as="numpy", backend=backend
        ),
        "price_vec": lambda ks: fv.fast_black_scholes(
            "c", S, ks, T, R, SIGMA, return_as="numpy", backend=backend
        ),
        "iv_scalar": lambda p, k: fv.fast_implied_volatility(
            p, S, k, T, R, "c", return_as="numpy", backend=backend
        ),
        "iv_vec": lambda ps, ks: fv.fast_implied_volatility(
            ps, S, ks, T, R, "c", return_as="numpy", backend=backend
        ),
        "greeks_scalar": lambda k: fv.get_all_greeks(
            "c", S, k, T, R, SIGMA, return_as="numpy", backend=backend
        ),
        "greeks_vec": lambda ks: fv.get_all_greeks(
            "c", S, ks, T, R, SIGMA, return_as="numpy", backend=backend
        ),
    }
    return f"fast-vollib {_pkg_version('fast-vollib')} ({backend})", fns, None


def adapter_fast_vollib_numba():
    return _adapter_fast_vollib("numba")


def adapter_fast_vollib_numpy():
    return _adapter_fast_vollib("numpy")


def adapter_vollib():
    from importlib.metadata import version as _pkg_version

    from vollib.black_scholes import black_scholes as v_bs
    from vollib.black_scholes.greeks.analytical import delta as v_d
    from vollib.black_scholes.greeks.analytical import gamma as v_g
    from vollib.black_scholes.greeks.analytical import rho as v_r
    from vollib.black_scholes.greeks.analytical import theta as v_t
    from vollib.black_scholes.greeks.analytical import vega as v_v
    from vollib.black_scholes.implied_volatility import implied_volatility as v_iv

    fns = {
        "price_scalar": lambda k: v_bs("c", S, k, T, R, SIGMA),
        "price_vec": lambda ks: [v_bs("c", S, k, T, R, SIGMA) for k in ks],
        "iv_scalar": lambda p, k: v_iv(p, S, k, T, R, "c"),
        "iv_vec": lambda ps, ks: [v_iv(p, S, k, T, R, "c") for p, k in zip(ps, ks, strict=True)],
        "greeks_scalar": lambda k: (
            v_d("c", S, k, T, R, SIGMA),
            v_g("c", S, k, T, R, SIGMA),
            v_v("c", S, k, T, R, SIGMA),
            v_t("c", S, k, T, R, SIGMA),
            v_r("c", S, k, T, R, SIGMA),
        ),
        "greeks_vec": lambda ks: [
            (
                v_d("c", S, k, T, R, SIGMA),
                v_g("c", S, k, T, R, SIGMA),
                v_v("c", S, k, T, R, SIGMA),
                v_t("c", S, k, T, R, SIGMA),
                v_r("c", S, k, T, R, SIGMA),
            )
            for k in ks
        ],
    }
    # Scalar loops: IV is ~15µs/solve and greeks are 5 calls/row — cap so a
    # single rep stays in seconds. The price chart covers vollib to 1M already.
    return f"vollib {_pkg_version('vollib')}", fns, 100_000


ADAPTERS = [
    adapter_pyvolr,
    adapter_opengreeks,
    adapter_fast_vollib_numba,
    adapter_fast_vollib_numpy,
    adapter_vollib,
]


def main() -> None:
    # Precompute target prices for the IV workload with pyvolr (untimed);
    # every library then inverts the same prices.
    from pyvolr import bs as pv

    scalar_price = float(pv.price("c", S=S, K=K1, T=T, r=R, sigma=SIGMA))
    prices_by_n: dict[int, np.ndarray] = {
        n: np.asarray(pv.price("c", S=S, K=_strikes(n), T=T, r=R, sigma=SIGMA))
        for n in SIZES
        if n > 1
    }

    results: dict[str, dict[str, dict[str, float]]] = {}
    print(f"{'library':<28} {'workload':<8} {'size':>10} {'time/call':>12} {'rows/s':>14}")
    print("-" * 78)

    for adapter in ADAPTERS:
        try:
            name, fns, cap = adapter()
        except Exception as exc:
            print(f"# skipped: {adapter.__name__}: {exc}")
            continue
        lib_results: dict[str, dict[str, float]] = {w: {} for w in WORKLOADS}

        for workload in WORKLOADS:
            for n in SIZES:
                if cap is not None and n > cap:
                    print(f"{name:<28} {workload:<8} {n:>10,} {'(skip > cap)':>12}")
                    continue
                if n == 1:
                    key = f"{workload}_scalar"
                    if key not in fns:
                        continue
                    if workload == "iv":
                        fn = lambda f=fns[key]: f(scalar_price, K1)  # noqa: E731
                    else:
                        fn = lambda f=fns[key]: f(K1)  # noqa: E731
                else:
                    key = f"{workload}_vec"
                    if key not in fns:
                        continue
                    ks = _strikes(n)
                    if workload == "iv":
                        ps = prices_by_n[n]
                        fn = lambda f=fns[key], p=ps, k=ks: f(p, k)  # noqa: E731
                    else:
                        fn = lambda f=fns[key], k=ks: f(k)  # noqa: E731

                try:
                    t = bench(fn, target_seconds=1.5 if n <= 100_000 else 4.0)
                except Exception as exc:
                    print(f"{name:<28} {workload:<8} {n:>10,} FAILED — {type(exc).__name__}: {exc}")
                    continue
                lib_results[workload][str(n)] = t
                print(f"{name:<28} {workload:<8} {n:>10,} {fmt(t):>12} {n / t:>14,.0f}")

        results[name] = lib_results
        RESULTS_PATH.write_text(json.dumps(results, indent=2, sort_keys=True))

    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
