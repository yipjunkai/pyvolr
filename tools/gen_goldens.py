#!/usr/bin/env python3
"""Regenerate and verify the mpmath golden constants used in the Rust tests.

The reference values asserted in `crates/core/src/{normal,bsm,iv}.rs` test
modules are high-precision (>= 50 digit) computations rounded to f64. They were
historically hand-pasted, which makes them unauditable magic numbers. This
script is their executable provenance: it recomputes every one at 60-digit
precision with mpmath and either prints them (`--emit`) or verifies the in-tree
values still match a fresh computation (default).

    python tools/gen_goldens.py          # verify in-tree goldens (CI-style)
    python tools/gen_goldens.py --emit    # print Rust-pasteable literals

mpmath ships in the `dev`/`test` extras (`uv sync --extra dev`); run under the
project venv. This is a developer tool — it is not imported by the package and
not run in CI (the Rust tests already assert these literals; this validates
that the literals are themselves correct to f64).

`expected` columns below are copied from the Rust sources, so editing a golden
means editing it in both places — that duplication IS the cross-check.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

import mpmath as mp

if TYPE_CHECKING:
    from collections.abc import Callable

mp.mp.dps = 60


# --- mpmath reference implementations (match the closed forms in the Rust docs) ---


def phi(x: mp.mpf) -> mp.mpf:
    """Standard normal CDF via the sign-split erfc form (matches normal::cdf)."""
    return mp.mpf("0.5") * mp.erfc(-x / mp.sqrt(2))


def pdf(x: mp.mpf) -> mp.mpf:
    return mp.e ** (-x * x / 2) / mp.sqrt(2 * mp.pi)


def erfcx(z: mp.mpf) -> mp.mpf:
    """exp(z^2) * erfc(z) — the scaled complementary error function."""
    return mp.e ** (z * z) * mp.erfc(z)


def inverse_cdf(u: mp.mpf) -> mp.mpf:
    """Standard normal quantile (the AS241 target): sqrt(2) * erfinv(2u - 1)."""
    return mp.sqrt(2) * mp.erfinv(2 * u - 1)


def bsm_price(flag: str, s: float, k: float, t: float, r: float, q: float, sigma: float) -> mp.mpf:
    s_, k_, t_, r_, q_, v_ = (mp.mpf(str(a)) for a in (s, k, t, r, q, sigma))
    vol_sqrt_t = v_ * mp.sqrt(t_)
    d1 = (mp.log(s_ / k_) + (r_ - q_ + v_ * v_ / 2) * t_) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    disc_q, disc_r = mp.e ** (-q_ * t_), mp.e ** (-r_ * t_)
    if flag == "c":
        return s_ * disc_q * phi(d1) - k_ * disc_r * phi(d2)
    return k_ * disc_r * phi(-d2) - s_ * disc_q * phi(-d1)


def nb_call(x: float, s: float) -> mp.mpf:
    """Jaeckel normalised Black call: exp(x/2) Phi(x/s + s/2) - exp(-x/2) Phi(x/s - s/2)."""
    x_, s_ = mp.mpf(str(x)), mp.mpf(str(s))
    return mp.e ** (x_ / 2) * phi(x_ / s_ + s_ / 2) - mp.e ** (-x_ / 2) * phi(x_ / s_ - s_ / 2)


# --- golden tables: (rust_location, tol, compute, [(label, args, expected_f64), ...]) ---

GOLDENS: list[tuple[str, float, Callable[..., mp.mpf], list[tuple[str, tuple, float]]]] = [
    (
        "normal.rs::cdf_negative_tail_matches_mpmath",
        1e-14,
        lambda x: phi(mp.mpf(str(x))),
        [
            ("cdf(-3.99)", (-3.99,), 3.303664762940242e-5),
            ("cdf(-3.9)", (-3.9,), 4.8096344017602736e-5),
            ("cdf(-3.5)", (-3.5,), 0.00023262907903552504),
            ("cdf(-3.0)", (-3.0,), 0.0013498980316300946),
            ("cdf(-2.5)", (-2.5,), 0.006209665325776135),
            ("cdf(-2.0)", (-2.0,), 0.02275013194817921),
            ("cdf(-1.5)", (-1.5,), 0.06680720126885807),
            ("cdf(-1.0)", (-1.0,), 0.15865525393145705),
        ],
    ),
    (
        "normal.rs::pdf_known_values",
        1e-12,
        lambda x: pdf(mp.mpf(str(x))),
        [("pdf(1.0)", (1.0,), 0.2419707245191434)],
    ),
    (
        "normal.rs::erfcx_golden_values",
        1e-14,
        lambda z: erfcx(mp.mpf(str(z))),
        [
            ("erfcx(-3.0)", (-3.0,), 1.6205988853999586e4),
            ("erfcx(-1.0)", (-1.0,), 5.008980080762283),
            ("erfcx(-0.5)", (-0.5,), 1.952360489182557),
            ("erfcx(0.0)", (0.0,), 1.0),
            ("erfcx(0.25)", (0.25,), 7.70346547730997e-1),
            ("erfcx(0.5)", (0.5,), 6.156903441929258e-1),
            ("erfcx(1.0)", (1.0,), 4.27583576155807e-1),
            ("erfcx(2.0)", (2.0,), 2.553956763105058e-1),
            ("erfcx(3.0)", (3.0,), 1.7900115118139e-1),
            ("erfcx(4.0)", (4.0,), 1.369994576250614e-1),
            ("erfcx(5.0)", (5.0,), 1.107046377330686e-1),
            ("erfcx(10.0)", (10.0,), 5.614099274382259e-2),
            ("erfcx(25.0)", (25.0,), 2.2549572432641355e-2),
            ("erfcx(100.0)", (100.0,), 5.641613782989433e-3),
            ("erfcx(1000.0)", (1000.0,), 5.641893014533876e-4),
        ],
    ),
    (
        "normal.rs::inverse_cdf_golden_values",
        1e-14,
        lambda u: inverse_cdf(mp.mpf(str(u))),
        [
            ("invcdf(1e-15)", (1e-15,), -7.941345326170998),
            ("invcdf(1e-10)", (1e-10,), -6.361340902404056),
            ("invcdf(1e-6)", (1e-6,), -4.753424308822899),
            ("invcdf(1e-3)", (1e-3,), -3.090232306167813),
            ("invcdf(1e-2)", (1e-2,), -2.3263478740408407),
            ("invcdf(1e-1)", (1e-1,), -1.2815515655446004),
            ("invcdf(0.25)", (0.25,), -6.744897501960817e-1),
            ("invcdf(0.4)", (0.4,), -2.533471031357997e-1),
            ("invcdf(0.6)", (0.6,), 2.533471031357997e-1),
            ("invcdf(0.75)", (0.75,), 6.744897501960817e-1),
            ("invcdf(0.9)", (0.9,), 1.2815515655446004),
            ("invcdf(0.99)", (0.99,), 2.3263478740408407),
            ("invcdf(0.999)", (0.999,), 3.090232306167813),
        ],
    ),
    (
        "bsm.rs::price_matches_mpmath_goldens",
        1e-12,
        bsm_price,
        [
            ("benign_call", ("c", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2), 10.450583572185568),
            ("benign_put", ("p", 100.0, 105.0, 0.5, 0.05, 0.02, 0.25), 8.92305213750915),
            ("deep_otm_put", ("p", 100.0, 60.0, 0.02, 0.05, 0.0, 0.12), 1.743957269143156e-201),
            (
                "very_deep_otm_call",
                ("c", 100.0, 300.0, 0.25, 0.03, 0.0, 0.2),
                7.924761804025656e-28,
            ),
            ("deep_itm_call", ("c", 100.0, 50.0, 0.5, 0.05, 0.0, 0.2), 51.23450474417435),
        ],
    ),
    (
        "iv.rs::call_matches_mpmath_goldens (normalised_black::call)",
        1e-14,
        nb_call,
        [
            ("r4_moderate_otm", (-0.1, 0.2), 3.9458602641807e-2),
            ("r4_atm", (0.0, 0.2), 7.965567455405796e-2),
            ("r4_otm_high", (-0.5, 0.4), 1.996050989765552e-2),
            ("r3_atm_high_vol", (0.0, 2.0), 6.826894921370859e-1),
            ("r3_otm_long_t", (-0.5, 2.0), 4.666462289193513e-1),
            ("r2_small_t_atm", (0.0, 0.02), 7.978712629263208e-3),
            ("r2_small_t_otm", (-0.0953101798043248, 0.02), 3.663393972840609e-9),
            ("r1_deep_asymp", (-1.1, 0.1), 1.7072701604888685e-30),
            ("r1_deeper", (-1.5, 0.1), 2.423020570391163e-53),
            ("itm_call", (0.5, 0.2), 5.056237971192526e-1),
            ("itm_call_high_vol", (1.0, 0.5), 1.0463329697381733),
        ],
    ),
]


def rel_diff(got: float, expected: float) -> float:
    if expected == 0.0:
        return abs(got)
    return abs((got - expected) / expected)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--emit", action="store_true", help="print Rust-pasteable literals instead of verifying"
    )
    args = ap.parse_args()

    failures = 0
    for location, tol, compute, items in GOLDENS:
        print(f"\n# {location}  (tol {tol:.0e})")
        for label, call_args, expected in items:
            got = float(compute(*call_args))
            if args.emit:
                print(f"  {label}: {got!r}")
                continue
            rd = rel_diff(got, expected)
            ok = rd <= tol
            failures += not ok
            # NOTE flags a golden sitting within 2x of its category floor —
            # i.e. f64-conditioning-limited, not exact. Visible, not buried,
            # so the floor above stays honest.
            mark = "FAIL" if not ok else ("note" if rd > 0.5 * tol else "ok  ")
            print(f"  [{mark}] {label}: in-tree {expected!r}  mpmath {got!r}  rel {rd:.2e}")

    if args.emit:
        return 0
    print(
        f"\n{'OK: all goldens match' if not failures else f'FAILED: {failures} golden(s) drifted'}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
