# Changelog

All notable changes to pyvolr are documented in this file. This file is
maintained by [release-please](https://github.com/googleapis/release-please)
from conventional commits — do not edit version sections by hand.

## [0.1.6](https://github.com/yipjunkai/pyvolr/compare/v0.1.5...v0.1.6) (2026-07-03)


### Features

* scalar fast path for every pricing endpoint ([#66](https://github.com/yipjunkai/pyvolr/issues/66)) ([ba77599](https://github.com/yipjunkai/pyvolr/commit/ba7759915c120abe913494a163e1ce2c50b40136))


### Documentation

* **bench:** add the IV tail-accuracy chart — throughput left, precision right ([#63](https://github.com/yipjunkai/pyvolr/issues/63)) ([ddfb237](https://github.com/yipjunkai/pyvolr/commit/ddfb23790068b27a0325ebf1b3bcd4ac87819855))
* **bench:** refresh the scalar rows for the 0.1.6 fast path ([#67](https://github.com/yipjunkai/pyvolr/issues/67)) ([ead8b8c](https://github.com/yipjunkai/pyvolr/commit/ead8b8c85b362184f51be445e8286fc251f4c117))
* refresh SECURITY + README, and fix the CI changes-gate ([#60](https://github.com/yipjunkai/pyvolr/issues/60)) ([721e387](https://github.com/yipjunkai/pyvolr/commit/721e3873a7f688289327351a7f90ad1aa4b2d015))
* reposition for the vollib revival and refresh competitor benchmarks ([#62](https://github.com/yipjunkai/pyvolr/issues/62)) ([374fa2d](https://github.com/yipjunkai/pyvolr/commit/374fa2d3c1b4f39570267d065f3b55cdbbdf6e3e))
* shrink the stale-doc surface (dirs-only tree; fold bench/README into the justfile) ([#65](https://github.com/yipjunkai/pyvolr/issues/65)) ([c7c3e56](https://github.com/yipjunkai/pyvolr/commit/c7c3e564cf3ec8d97e37fce968b9c8fade3a235d))

## [0.1.5](https://github.com/yipjunkai/pyvolr/compare/v0.1.4...v0.1.5) (2026-07-01)


### Features

* add on_error to native implied_vol ([#46](https://github.com/yipjunkai/pyvolr/issues/46)) ([f2f49a0](https://github.com/yipjunkai/pyvolr/commit/f2f49a00654862fb29e89707e18d0ec3b7194fb4))
* add py_vollib_vectorized compat shim ([#47](https://github.com/yipjunkai/pyvolr/issues/47)) ([7878a1c](https://github.com/yipjunkai/pyvolr/commit/7878a1c3cc07e530852a7e95a83bba081a8a1ecd))
* add return_as to native bs/black76 API ([#45](https://github.com/yipjunkai/pyvolr/issues/45)) ([b0ba347](https://github.com/yipjunkai/pyvolr/commit/b0ba347945b380bb454ff22004b339f998efb261))
* support numpy &gt;= 1.26 (lower the floor from &gt;= 2.0) ([#41](https://github.com/yipjunkai/pyvolr/issues/41)) ([5c16cc0](https://github.com/yipjunkai/pyvolr/commit/5c16cc06a06cb3871e3bd2c286b85195d6f17561))


### Bug Fixes

* **deps:** bump pyo3 and rust-numpy to 0.29 ([#56](https://github.com/yipjunkai/pyvolr/issues/56)) ([c1e69a1](https://github.com/yipjunkai/pyvolr/commit/c1e69a1d1e7e8000883c7f1739e379505f368417))
* truth-and-hygiene pass — claims, typed greeks, template drift ([#37](https://github.com/yipjunkai/pyvolr/issues/37)) ([59ec7ec](https://github.com/yipjunkai/pyvolr/commit/59ec7ecbeddb8eb2f5c4db8b53320daff3a653ae))
* validate whole flag strings in array inputs ([#33](https://github.com/yipjunkai/pyvolr/issues/33)) ([4b6546f](https://github.com/yipjunkai/pyvolr/commit/4b6546fc0ad93fa85c70cf28138f921953bba041))


### Documentation

* theta is negative calendar theta, minus dPrice/dT ([#35](https://github.com/yipjunkai/pyvolr/issues/35)) ([ce9d5b9](https://github.com/yipjunkai/pyvolr/commit/ce9d5b9636e4f27e20a9db140d748caad97ec7ac))

## [0.1.4](https://github.com/yipjunkai/pyvolr/compare/v0.1.3...v0.1.4) (2026-06-12)


### Bug Fixes

* **ci:** drop the removed CIBW_ENABLE group for cibuildwheel v4 ([#29](https://github.com/yipjunkai/pyvolr/issues/29)) ([ac36ef3](https://github.com/yipjunkai/pyvolr/commit/ac36ef326413036cab6da2fc7445b644ea510979))
* **compat:** raise py_vollib's IV bound exceptions instead of returning NaN ([#21](https://github.com/yipjunkai/pyvolr/issues/21)) ([b9eb457](https://github.com/yipjunkai/pyvolr/commit/b9eb45742a044e413c11c1b0452a45686c3f1276))
* **core:** correct CDF tail, zero-vol delta, and deep-OTM pricing accuracy ([#19](https://github.com/yipjunkai/pyvolr/issues/19)) ([240844e](https://github.com/yipjunkai/pyvolr/commit/240844e8e2bad924719002e9a36340c1a82fe477))
* drop 3.13t wheels — cibuildwheel 4 removed Python 3.13 free-threading ([#30](https://github.com/yipjunkai/pyvolr/issues/30)) ([ef2b77a](https://github.com/yipjunkai/pyvolr/commit/ef2b77abbebdc5727325089fa59407d43a36a7ee))


### Documentation

* 3.13t wheels were last published at 0.1.3, not 0.1.4 ([562553d](https://github.com/yipjunkai/pyvolr/commit/562553da68b5d31bdc1c869c97f55980594af998))
* differential is a release gate now, not a nightly cron ([37e0725](https://github.com/yipjunkai/pyvolr/commit/37e0725f168989e5bcf6f617abbfb680afe00f31))


### Benchmarks

* split the experiment harness out of the perf gate ([#26](https://github.com/yipjunkai/pyvolr/issues/26)) ([91f11eb](https://github.com/yipjunkai/pyvolr/commit/91f11eb38825bf093a1d20f81eefe8775e4a0b8f))

## [0.1.3](https://github.com/yipjunkai/pyvolr/compare/v0.1.2...v0.1.3) (2026-05-29)


### Performance

* mechanical-sympathy audit + 2026 competitor positioning ([#11](https://github.com/yipjunkai/pyvolr/issues/11)) ([002c763](https://github.com/yipjunkai/pyvolr/commit/002c76331b983df9be12523d8198c848fc72dbe3))

## [0.1.2](https://github.com/yipjunkai/pyvolr/compare/v0.1.1...v0.1.2) (2026-05-26)


### Features

* **iv:** replace Newton/bisection with Jäckel "Let's Be Rational" ([#10](https://github.com/yipjunkai/pyvolr/issues/10)) ([4fe0ff1](https://github.com/yipjunkai/pyvolr/commit/4fe0ff14d5d83576bb7fea1239a69d085646b0d6))


### Documentation

* bump version reference to 0.1.2 in README + chart footer ([b66f9a7](https://github.com/yipjunkai/pyvolr/commit/b66f9a7afb0b93f43954819f3e61fb7c4614af91))
* **readme:** refresh perf table to post-LBR bench; bump 0.1.0 -&gt; 0.1.1 ([6398f5b](https://github.com/yipjunkai/pyvolr/commit/6398f5ba1c46c6aa8a802b573baa9f729c966bf5))
* **readme:** use absolute URLs for perf chart so PyPI renders it ([c26d1f6](https://github.com/yipjunkai/pyvolr/commit/c26d1f6a25e8bf932cd398756b3ee161b54d556c))

## [0.1.1](https://github.com/yipjunkai/pyvolr/compare/v0.1.0...v0.1.1) (2026-05-25)


### Features

* **black76:** add Black-76 pricing model for futures and forward options ([#5](https://github.com/yipjunkai/pyvolr/issues/5)) ([0b86681](https://github.com/yipjunkai/pyvolr/commit/0b86681dd1d05ca8c85c5e7617ee96ee6326cf8b))


### Bug Fixes

* **differential:** locate py_lets_be_rational/constants.py via find, not import ([93ad431](https://github.com/yipjunkai/pyvolr/commit/93ad431d982b28e92029cd071c241d8297482d6f))
* **fuzz:** align harness assertions with the math contract ([d073584](https://github.com/yipjunkai/pyvolr/commit/d073584db568de43cb497106947a2b5806e9bd4f))
* **fuzz:** allow NaN price for realistic inputs (graceful-failure value) ([32a5db3](https://github.com/yipjunkai/pyvolr/commit/32a5db3e2748563fde37232d87c282a9001570e2))
* **fuzz:** bound r, q, sigma absolutely to keep BSM intermediates finite ([2417bf4](https://github.com/yipjunkai/pyvolr/commit/2417bf4339fc877552694d7b88f27e7a48ad207d))
* **fuzz:** require s * disc_q and k * disc_r to be finite in well-conditioned ([ed4fd8b](https://github.com/yipjunkai/pyvolr/commit/ed4fd8b7a100e5bd851701770a9b2f378314b626))
* **fuzz:** rewrite bsm_price harness around a physically realistic input band ([0cf1512](https://github.com/yipjunkai/pyvolr/commit/0cf1512517cbcc701cadf38ef03c3b657e5fb4ba))
* **fuzz:** scale negative-price tolerance relative to s + k ([3e4701a](https://github.com/yipjunkai/pyvolr/commit/3e4701adc80a9917ccaf54b6045ad1d1e707447d))
* **fuzz:** tighten well-conditioned input bands to physically realistic ranges ([0d46d73](https://github.com/yipjunkai/pyvolr/commit/0d46d73cf9e92d81116145127492b9bc70ebf989))
* **release-please:** use bare v* tag format for the root package ([5d76e1d](https://github.com/yipjunkai/pyvolr/commit/5d76e1d7dd63128c5ebe1ff11eab301f9b1c6c24))


### Documentation

* **governance, readme, why:** clarify release automation and credential management ([4f01aa9](https://github.com/yipjunkai/pyvolr/commit/4f01aa956f54b7cd6d1fdcc2afc952864a7c36dc))
* **readme:** add performance chart (light + dark SVG variants) ([b186a84](https://github.com/yipjunkai/pyvolr/commit/b186a8451188caa7306a085e05371a4992a3258f))
* **readme:** surface free-threaded Python (3.13t / 3.14t) support ([da1f2a0](https://github.com/yipjunkai/pyvolr/commit/da1f2a047d8555308d99cf8678ed8c2dca22f81f))
* **readme:** swap the architecture diagram for performance numbers; expand badge row ([4a4f4df](https://github.com/yipjunkai/pyvolr/commit/4a4f4dfdb16b1d8ccf8f00bc170d79cefa87259e))

## [0.1.0] - 2026-05-24

### Features

- Black-Scholes-Merton pricing for European calls and puts with continuous dividend yield.
- Analytical Greeks: delta, gamma, theta, vega, rho.
- Implied volatility via Newton-Raphson seeded with the Manaster-Koehler initial guess, with bisection fallback for poorly-conditioned inputs.
- numpy broadcasting across all pricing and Greek inputs.
- `pyvolr.compat.py_vollib` drop-in replacement (including `black_scholes_merton`) for the abandoned `py_vollib` library.
- Type stubs for the Rust extension.
- abi3 wheels for Python 3.10–3.14 and free-threaded wheels for Python 3.13t/3.14t across Linux (x86_64, aarch64; manylinux + musllinux), macOS (Intel, Apple Silicon), and Windows (x86_64).
- cargo-fuzz harnesses for `bsm_price` and `iv_solve`.
