# Changelog

All notable changes to pyvolr are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release.
- Black-Scholes-Merton pricing for European calls and puts with continuous dividend yield.
- Analytical Greeks: delta, gamma, theta, vega, rho.
- Implied volatility via Newton-Raphson seeded with the Manaster-Koehler initial guess, with bisection fallback for poorly-conditioned inputs.
- numpy broadcasting across all pricing and Greek inputs.
- `pyvolr.compat.py_vollib` and `pyvolr.compat.py_vollib_merton` drop-in replacements for the abandoned `py_vollib` library.
- Type stubs for the Rust extension.

[Unreleased]: https://github.com/yipjunkai/pyvolr/commits/main
