"""Cross-validate pyvolr's numerical outputs against the live competitor set.

One-command runner: `just sanity` (env pins + setup live in the repo justfile).
Runs in each venv and aggregates:

    .venv-bench-entrants/bin/python bench/sanity_check_competitors.py
    .venv-bench/bin/python          bench/sanity_check_competitors.py
    .venv-bench312/bin/python       bench/sanity_check_competitors.py

Each phase reports max abs/rel diffs (vs pyvolr) for every library that
imports successfully, across price + 5 Greeks + implied volatility, for
both call and put flags, over a 9-cell grid spanning ATM/ITM/OTM x
short/long expiry x low/high vol.

This is a *correctness* sanity check, not a benchmark. The verdict is
"does pyvolr agree with the ecosystem to f64 precision on well-posed
inputs". Catastrophic disagreement on any cell would be a regression
worth investigating.

Conventions are normalised to pyvolr's (per-unit-vol vega, per-year
theta, per-unit-rate rho). Libraries using the legacy py_vollib
convention (vollib, py_vollib_vectorized) get the conversion factors
applied here so the comparison is apples-to-apples.

Skipped: QuantLib Greeks (needs the full EuropeanOption + process +
engine setup; the bare `ql.blackFormula` path the speed benchmark uses
has price + IV via stddev only). That gap is documented inline; pyvolr
vs QuantLib for price + IV is still checked.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

# pyvolr's convention is the reference. Adapters normalise everything to it.
S0, R = 100.0, 0.05  # constants across the grid

# (label, S, K, T, sigma)
GRID: list[tuple[str, float, float, float, float]] = [
    ("ATM moderate", 100.0, 100.0, 0.5, 0.20),
    ("ATM short expiry", 100.0, 100.0, 0.05, 0.20),
    ("ATM long expiry", 100.0, 100.0, 2.0, 0.20),
    ("ATM low vol", 100.0, 100.0, 0.5, 0.05),
    ("ATM high vol", 100.0, 100.0, 0.5, 0.80),
    ("Deep ITM call", 150.0, 100.0, 0.5, 0.20),
    ("Mod OTM call", 90.0, 100.0, 0.5, 0.20),
    ("Deep OTM call", 50.0, 100.0, 0.5, 0.20),
    ("Very deep OTM short", 50.0, 100.0, 0.05, 0.20),
]

FUNCS = [
    "price_call",
    "price_put",
    "delta_call",
    "delta_put",
    "gamma",
    "vega",
    "theta_call",
    "theta_put",
    "rho_call",
    "rho_put",
    "iv_call",
    "iv_put",
]


# ----------------------------------------------------------------------------
# Library adapters: each returns {function_name -> callable(*args)} normalised
# to pyvolr's per-unit convention. Missing functions are simply absent from
# the dict. `iv_*` callables take (price, S, K, T, r); price callables take
# (S, K, T, r, sigma); Greek callables take (S, K, T, r, sigma).
# ----------------------------------------------------------------------------


def adapter_pyvolr() -> tuple[str, dict[str, Callable[..., float]]]:
    import pyvolr
    from pyvolr import bs as pv

    return f"pyvolr {pyvolr.__version__}", {
        "price_call": lambda S, K, T, r, sigma: float(
            pv.price("c", S=S, K=K, T=T, r=r, sigma=sigma)
        ),
        "price_put": lambda S, K, T, r, sigma: float(
            pv.price("p", S=S, K=K, T=T, r=r, sigma=sigma)
        ),
        "delta_call": lambda S, K, T, r, sigma: float(
            pv.delta("c", S=S, K=K, T=T, r=r, sigma=sigma)
        ),
        "delta_put": lambda S, K, T, r, sigma: float(
            pv.delta("p", S=S, K=K, T=T, r=r, sigma=sigma)
        ),
        "gamma": lambda S, K, T, r, sigma: float(pv.gamma(S=S, K=K, T=T, r=r, sigma=sigma)),
        "vega": lambda S, K, T, r, sigma: float(pv.vega(S=S, K=K, T=T, r=r, sigma=sigma)),
        "theta_call": lambda S, K, T, r, sigma: float(
            pv.theta("c", S=S, K=K, T=T, r=r, sigma=sigma)
        ),
        "theta_put": lambda S, K, T, r, sigma: float(
            pv.theta("p", S=S, K=K, T=T, r=r, sigma=sigma)
        ),
        "rho_call": lambda S, K, T, r, sigma: float(pv.rho("c", S=S, K=K, T=T, r=r, sigma=sigma)),
        "rho_put": lambda S, K, T, r, sigma: float(pv.rho("p", S=S, K=K, T=T, r=r, sigma=sigma)),
        "iv_call": lambda price, S, K, T, r: float(pv.implied_vol(price, "c", S=S, K=K, T=T, r=r)),
        "iv_put": lambda price, S, K, T, r: float(pv.implied_vol(price, "p", S=S, K=K, T=T, r=r)),
    }


def adapter_vollib() -> tuple[str, dict[str, Callable[..., float]]]:
    # vollib (resurrected upstream of py_vollib) uses the legacy convention:
    # vega per-1%-vol, theta per-day, rho per-1%-rate. Apply factors here.
    from vollib.black_scholes import black_scholes as v_bs
    from vollib.black_scholes.greeks.analytical import delta as v_d
    from vollib.black_scholes.greeks.analytical import gamma as v_g
    from vollib.black_scholes.greeks.analytical import rho as v_r
    from vollib.black_scholes.greeks.analytical import theta as v_t
    from vollib.black_scholes.greeks.analytical import vega as v_v
    from vollib.black_scholes.implied_volatility import implied_volatility as v_iv

    def s(x: Any) -> float:
        # vollib may return scalars or 1-element DataFrames if py_vollib_vectorized
        # has monkeypatched it. Extract the scalar.
        try:
            return float(x)
        except (TypeError, ValueError):
            return float(x.iloc[0, 0])  # type: ignore[attr-defined]

    from importlib.metadata import version as _pkg_version

    return f"vollib {_pkg_version('vollib')}", {
        "price_call": lambda S, K, T, r, sigma: s(v_bs("c", S, K, T, r, sigma)),
        "price_put": lambda S, K, T, r, sigma: s(v_bs("p", S, K, T, r, sigma)),
        "delta_call": lambda S, K, T, r, sigma: s(v_d("c", S, K, T, r, sigma)),
        "delta_put": lambda S, K, T, r, sigma: s(v_d("p", S, K, T, r, sigma)),
        "gamma": lambda S, K, T, r, sigma: s(v_g("c", S, K, T, r, sigma)),
        # Legacy convention: multiply per-1% / per-day back to per-unit / per-year.
        "vega": lambda S, K, T, r, sigma: s(v_v("c", S, K, T, r, sigma)) * 100.0,
        "theta_call": lambda S, K, T, r, sigma: s(v_t("c", S, K, T, r, sigma)) * 365.0,
        "theta_put": lambda S, K, T, r, sigma: s(v_t("p", S, K, T, r, sigma)) * 365.0,
        "rho_call": lambda S, K, T, r, sigma: s(v_r("c", S, K, T, r, sigma)) * 100.0,
        "rho_put": lambda S, K, T, r, sigma: s(v_r("p", S, K, T, r, sigma)) * 100.0,
        "iv_call": lambda price, S, K, T, r: s(v_iv(price, S, K, T, r, "c")),
        "iv_put": lambda price, S, K, T, r: s(v_iv(price, S, K, T, r, "p")),
    }


def adapter_blackscholes() -> tuple[str, dict[str, Callable[..., float]]]:
    # Same convention as pyvolr (per-unit vega, per-year theta, per-unit rho).
    # No IV solver exposed at the same API level — skip iv_*.
    from blackscholes import BlackScholesCall, BlackScholesPut

    def c(S: float, K: float, T: float, r: float, sigma: float) -> Any:
        return BlackScholesCall(S, K, T, r, sigma)

    def p(S: float, K: float, T: float, r: float, sigma: float) -> Any:
        return BlackScholesPut(S, K, T, r, sigma)

    return "blackscholes 0.2.0", {
        "price_call": lambda S, K, T, r, sigma: float(c(S, K, T, r, sigma).price()),
        "price_put": lambda S, K, T, r, sigma: float(p(S, K, T, r, sigma).price()),
        "delta_call": lambda S, K, T, r, sigma: float(c(S, K, T, r, sigma).delta()),
        "delta_put": lambda S, K, T, r, sigma: float(p(S, K, T, r, sigma).delta()),
        "gamma": lambda S, K, T, r, sigma: float(c(S, K, T, r, sigma).gamma()),
        "vega": lambda S, K, T, r, sigma: float(c(S, K, T, r, sigma).vega()),
        "theta_call": lambda S, K, T, r, sigma: float(c(S, K, T, r, sigma).theta()),
        "theta_put": lambda S, K, T, r, sigma: float(p(S, K, T, r, sigma).theta()),
        "rho_call": lambda S, K, T, r, sigma: float(c(S, K, T, r, sigma).rho()),
        "rho_put": lambda S, K, T, r, sigma: float(p(S, K, T, r, sigma).rho()),
        # No IV in blackscholes 0.2.0's public API at this level.
    }


def adapter_quantlib() -> tuple[str, dict[str, Callable[..., float]]]:
    # QuantLib's bare `blackFormula` only gives price + IV via stddev. Full
    # Greeks would need EuropeanOption + BlackScholesMertonProcess +
    # AnalyticEuropeanEngine — out of scope for this sanity check.
    import QuantLib as ql

    def price(S: float, K: float, T: float, r: float, sigma: float, *, call: bool) -> float:
        discount = float(np.exp(-r * T))
        stddev = sigma * float(np.sqrt(T))
        forward = S / discount
        option = ql.Option.Call if call else ql.Option.Put
        return ql.blackFormula(option, K, forward, stddev, discount)

    def iv(price_val: float, S: float, K: float, T: float, r: float, *, call: bool) -> float:
        discount = float(np.exp(-r * T))
        forward = S / discount
        option = ql.Option.Call if call else ql.Option.Put
        stddev = ql.blackFormulaImpliedStdDev(option, K, forward, price_val, discount)
        return stddev / float(np.sqrt(T))

    return f"QuantLib {ql.__version__}", {
        "price_call": lambda S, K, T, r, sigma: price(S, K, T, r, sigma, call=True),
        "price_put": lambda S, K, T, r, sigma: price(S, K, T, r, sigma, call=False),
        "iv_call": lambda p, S, K, T, r: iv(p, S, K, T, r, call=True),
        "iv_put": lambda p, S, K, T, r: iv(p, S, K, T, r, call=False),
    }


def adapter_quantforge() -> tuple[str, dict[str, Callable[..., float]]]:
    # Same convention as pyvolr.
    import quantforge as qf

    def greek(name: str, *, call: bool):
        def fn(S: float, K: float, T: float, r: float, sigma: float) -> float:
            return float(qf.black_scholes.greeks(S, K, T, r, sigma, is_call=call)[name])

        return fn

    return f"quantforge {qf.__version__}", {
        "price_call": lambda S, K, T, r, sigma: float(
            qf.black_scholes.call_price(S, K, T, r, sigma)
        ),
        "price_put": lambda S, K, T, r, sigma: float(qf.black_scholes.put_price(S, K, T, r, sigma)),
        "delta_call": greek("delta", call=True),
        "delta_put": greek("delta", call=False),
        "gamma": greek("gamma", call=True),
        "vega": greek("vega", call=True),
        "theta_call": greek("theta", call=True),
        "theta_put": greek("theta", call=False),
        "rho_call": greek("rho", call=True),
        "rho_put": greek("rho", call=False),
        "iv_call": lambda p, S, K, T, r: float(
            qf.black_scholes.implied_volatility(p, S, K, T, r, is_call=True)
        ),
        "iv_put": lambda p, S, K, T, r: float(
            qf.black_scholes.implied_volatility(p, S, K, T, r, is_call=False)
        ),
    }


def adapter_py_vollib_vectorized() -> tuple[str, dict[str, Callable[..., float]]]:
    # Monkeypatches py_vollib + vollib to return DataFrames. The values are
    # identical to vollib's (same engine underneath), so we only check that
    # the monkeypatched path returns the same numbers.
    import py_vollib_vectorized  # noqa: F401 -- side-effect: monkeypatches py_vollib
    from py_vollib.black_scholes import black_scholes as pv_bs
    from py_vollib.black_scholes.greeks.analytical import delta as pv_d
    from py_vollib.black_scholes.greeks.analytical import gamma as pv_g
    from py_vollib.black_scholes.greeks.analytical import rho as pv_r
    from py_vollib.black_scholes.greeks.analytical import theta as pv_t
    from py_vollib.black_scholes.greeks.analytical import vega as pv_v
    from py_vollib.black_scholes.implied_volatility import implied_volatility as pv_iv

    def s(x: Any) -> float:
        try:
            return float(x)
        except (TypeError, ValueError):
            return float(x.iloc[0, 0])  # type: ignore[attr-defined]

    return "py_vollib_vectorized 0.1.1", {
        "price_call": lambda S, K, T, r, sigma: s(pv_bs("c", S, K, T, r, sigma)),
        "price_put": lambda S, K, T, r, sigma: s(pv_bs("p", S, K, T, r, sigma)),
        "delta_call": lambda S, K, T, r, sigma: s(pv_d("c", S, K, T, r, sigma)),
        "delta_put": lambda S, K, T, r, sigma: s(pv_d("p", S, K, T, r, sigma)),
        "gamma": lambda S, K, T, r, sigma: s(pv_g("c", S, K, T, r, sigma)),
        "vega": lambda S, K, T, r, sigma: s(pv_v("c", S, K, T, r, sigma)) * 100.0,
        "theta_call": lambda S, K, T, r, sigma: s(pv_t("c", S, K, T, r, sigma)) * 365.0,
        "theta_put": lambda S, K, T, r, sigma: s(pv_t("p", S, K, T, r, sigma)) * 365.0,
        "rho_call": lambda S, K, T, r, sigma: s(pv_r("c", S, K, T, r, sigma)) * 100.0,
        "rho_put": lambda S, K, T, r, sigma: s(pv_r("p", S, K, T, r, sigma)) * 100.0,
        "iv_call": lambda p, S, K, T, r: s(pv_iv(p, S, K, T, r, "c")),
        "iv_put": lambda p, S, K, T, r: s(pv_iv(p, S, K, T, r, "p")),
    }


def adapter_opengreeks() -> tuple[str, dict[str, Callable[..., float]]]:
    # Rust+PyO3 py_vollib-compatible competitor (2026). Legacy py_vollib
    # conventions: per-1%-vol vega, per-day theta, per-1%-rate rho.
    from importlib.metadata import version as _pkg_version

    from opengreeks import black_scholes as og

    return f"opengreeks {_pkg_version('opengreeks')}", {
        "price_call": lambda S, K, T, r, sigma: float(og.black_scholes("c", S, K, T, r, sigma)),
        "price_put": lambda S, K, T, r, sigma: float(og.black_scholes("p", S, K, T, r, sigma)),
        "delta_call": lambda S, K, T, r, sigma: float(og.delta("c", S, K, T, r, sigma)),
        "delta_put": lambda S, K, T, r, sigma: float(og.delta("p", S, K, T, r, sigma)),
        "gamma": lambda S, K, T, r, sigma: float(og.gamma("c", S, K, T, r, sigma)),
        "vega": lambda S, K, T, r, sigma: float(og.vega("c", S, K, T, r, sigma)) * 100.0,
        "theta_call": lambda S, K, T, r, sigma: float(og.theta("c", S, K, T, r, sigma)) * 365.0,
        "theta_put": lambda S, K, T, r, sigma: float(og.theta("p", S, K, T, r, sigma)) * 365.0,
        "rho_call": lambda S, K, T, r, sigma: float(og.rho("c", S, K, T, r, sigma)) * 100.0,
        "rho_put": lambda S, K, T, r, sigma: float(og.rho("p", S, K, T, r, sigma)) * 100.0,
        "iv_call": lambda p, S, K, T, r: float(og.implied_volatility(p, S, K, T, r, "c")),
        "iv_put": lambda p, S, K, T, r: float(og.implied_volatility(p, S, K, T, r, "p")),
    }


def _adapter_fast_vollib(backend: str) -> tuple[str, dict[str, Callable[..., float]]]:
    # fast-vollib (arXiv:2604.27210). Legacy py_vollib conventions, same
    # factors as vollib above. Array-in/array-out API; extract the scalar.
    from importlib.metadata import version as _pkg_version

    import fast_vollib as fv

    if backend == "numba":
        import numba  # noqa: F401 — raise (=> clean skip) when the extra isn't installed

    def one(x: Any) -> float:
        return float(np.asarray(x).ravel()[0])

    def greek(name: str, flag: str, scale: float) -> Callable[..., float]:
        fn = getattr(fv, f"vectorized_{name}")

        def call(S: float, K: float, T: float, r: float, sigma: float) -> float:
            return one(fn(flag, S, K, T, r, sigma, return_as="numpy", backend=backend)) * scale

        return call

    def price(flag: str) -> Callable[..., float]:
        def call(S: float, K: float, T: float, r: float, sigma: float) -> float:
            return one(
                fv.fast_black_scholes(flag, S, K, T, r, sigma, return_as="numpy", backend=backend)
            )

        return call

    def iv(flag: str) -> Callable[..., float]:
        def call(p: float, S: float, K: float, T: float, r: float) -> float:
            return one(
                fv.fast_implied_volatility(p, S, K, T, r, flag, return_as="numpy", backend=backend)
            )

        return call

    return f"fast-vollib {_pkg_version('fast-vollib')} ({backend})", {
        "price_call": price("c"),
        "price_put": price("p"),
        "delta_call": greek("delta", "c", 1.0),
        "delta_put": greek("delta", "p", 1.0),
        "gamma": greek("gamma", "c", 1.0),
        "vega": greek("vega", "c", 100.0),
        "theta_call": greek("theta", "c", 365.0),
        "theta_put": greek("theta", "p", 365.0),
        "rho_call": greek("rho", "c", 100.0),
        "rho_put": greek("rho", "p", 100.0),
        "iv_call": iv("c"),
        "iv_put": iv("p"),
    }


def adapter_fast_vollib_numpy() -> tuple[str, dict[str, Callable[..., float]]]:
    return _adapter_fast_vollib("numpy")


def adapter_fast_vollib_numba() -> tuple[str, dict[str, Callable[..., float]]]:
    return _adapter_fast_vollib("numba")


# IMPORTANT: vollib must be imported BEFORE py_vollib_vectorized so vollib's
# scalar return type isn't monkeypatched out from under us.
ADAPTERS = [
    adapter_pyvolr,
    adapter_vollib,
    adapter_blackscholes,
    adapter_quantlib,
    adapter_quantforge,
    adapter_opengreeks,
    adapter_fast_vollib_numpy,
    adapter_fast_vollib_numba,
    adapter_py_vollib_vectorized,
]


def main() -> int:
    # Build the reference (pyvolr) and load every adapter that imports.
    name_ref, ref_funcs = adapter_pyvolr()
    competitor_data: list[tuple[str, dict[str, Callable[..., float]]]] = []
    for adapter in ADAPTERS:
        if adapter is adapter_pyvolr:
            continue
        try:
            n, fns = adapter()
        except Exception as exc:
            print(f"# skipped: {adapter.__name__}: {exc}")
            continue
        competitor_data.append((n, fns))

    # max_rel[(library, function)] tracks the worst relative drift over the grid.
    # Cells where pyvolr's value is at the f64 floor (or the competitor underflows)
    # are reported separately as "underflow".
    max_rel: dict[tuple[str, str], float] = {}
    underflow_cases: list[tuple[str, str, str]] = []  # (library, function, cell)
    skipped_lib_fn: set[tuple[str, str]] = set()

    for label, S, K, T, sigma in GRID:
        for func in FUNCS:
            # `price` is only meaningful for the IV cross-check; initialised here
            # to silence pyright (the runtime flow guards usage via `func.startswith`).
            price: float = 0.0
            # Reference value.
            ref_fn = ref_funcs[func]
            if func.startswith("iv_"):
                # IV cross-check: regenerate the price via pyvolr at the known
                # sigma, then ask each library to invert. Compare recovered to
                # original sigma.
                flag = "c" if func == "iv_call" else "p"
                price_func = "price_call" if flag == "c" else "price_put"
                price = ref_funcs[price_func](S, K, T, R, sigma)
                if not np.isfinite(price) or price <= 0:
                    continue
                ref_val = sigma  # the input is the reference for IV roundtrip
                try:
                    pv_val = ref_fn(price, S, K, T, R)
                except Exception:
                    continue
                if not np.isfinite(pv_val):
                    continue
            else:
                ref_val = ref_fn(S, K, T, R, sigma)
                if not np.isfinite(ref_val):
                    continue

            for lib_name, lib_funcs in competitor_data:
                if func not in lib_funcs:
                    skipped_lib_fn.add((lib_name, func))
                    continue
                try:
                    if func.startswith("iv_"):
                        cand = lib_funcs[func](price, S, K, T, R)
                    else:
                        cand = lib_funcs[func](S, K, T, R, sigma)
                except Exception:
                    skipped_lib_fn.add((lib_name, func))
                    continue
                if not np.isfinite(cand):
                    underflow_cases.append((lib_name, func, label))
                    continue
                # Two cases: both ~0 (acceptable agreement) or compute relative.
                if abs(ref_val) < 1e-200:
                    if abs(cand) < 1e-200:
                        continue  # both at the floor
                    rel = float("inf")
                elif cand == 0 and ref_val != 0:
                    underflow_cases.append((lib_name, func, label))
                    continue
                else:
                    rel = abs(cand - ref_val) / abs(ref_val)
                # Use `not in` so a bit-exact match (rel == 0) still records the key —
                # otherwise the report would mis-classify it as "no data".
                key = (lib_name, func)
                if key not in max_rel or rel > max_rel[key]:
                    max_rel[key] = rel

    # ---------- Report ----------
    print(f"\n=== Cross-validation summary (vs {name_ref}) ===\n")

    libs_seen = [n for n, _ in competitor_data]
    print(f"{'function':<14}", end="")
    for lib in libs_seen:
        # Short lib name for column header
        short = lib.split()[0]
        print(f"  {short:>22}", end="")
    print()
    print("-" * (14 + 24 * len(libs_seen)))

    for func in FUNCS:
        print(f"{func:<14}", end="")
        for lib in libs_seen:
            key = (lib, func)
            if key in skipped_lib_fn and key not in max_rel:
                cell = "(no API)"
            elif key not in max_rel:
                cell = "(no data)"
            else:
                rel = max_rel[key]
                if rel < 1e-13:
                    cell = f"OK ({rel:.1e})"
                elif rel < 1e-9:
                    cell = f"~ ULPs ({rel:.1e})"
                elif rel < 1e-3:
                    cell = f"loose ({rel:.1e})"
                else:
                    cell = f"BAD ({rel:.1e})"
            print(f"  {cell:>22}", end="")
        print()

    if underflow_cases:
        print(
            "\nUnderflow / NaN cases (competitor returned 0 or non-finite where pyvolr is finite):"
        )
        for lib, func, cell in underflow_cases:
            print(f"  {lib:<32}  {func:<12}  {cell}")
    else:
        print("\nNo underflow / NaN cases.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
