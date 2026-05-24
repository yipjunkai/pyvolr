# Changelog

All notable changes to pyvolr are documented in this file. This file is
maintained by [release-please](https://github.com/googleapis/release-please)
from conventional commits — do not edit version sections by hand.

## [0.1.0] - 2026-05-24

### Features

- Black-Scholes-Merton pricing for European calls and puts with continuous dividend yield.
- Analytical Greeks: delta, gamma, theta, vega, rho.
- Implied volatility via Newton-Raphson seeded with the Manaster-Koehler initial guess, with bisection fallback for poorly-conditioned inputs.
- numpy broadcasting across all pricing and Greek inputs.
- `pyvolr.compat.py_vollib` and `pyvolr.compat.py_vollib_merton` drop-in replacements for the abandoned `py_vollib` library.
- Type stubs for the Rust extension.
- abi3 wheels for Python 3.10–3.14 and free-threaded wheels for Python 3.13t/3.14t across Linux (x86_64, aarch64; manylinux + musllinux), macOS (Intel, Apple Silicon), and Windows (x86_64).
- cargo-fuzz harnesses for `bsm_price` and `iv_solve`.
