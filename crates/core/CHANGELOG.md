# Changelog

## [0.1.5](https://github.com/yipjunkai/pyvolr/compare/v0.1.4...v0.1.5) (2026-07-01)


### Bug Fixes

* **deps:** bump pyo3 and rust-numpy to 0.29 ([#56](https://github.com/yipjunkai/pyvolr/issues/56)) ([c1e69a1](https://github.com/yipjunkai/pyvolr/commit/c1e69a1d1e7e8000883c7f1739e379505f368417))
* truth-and-hygiene pass — claims, typed greeks, template drift ([#37](https://github.com/yipjunkai/pyvolr/issues/37)) ([59ec7ec](https://github.com/yipjunkai/pyvolr/commit/59ec7ecbeddb8eb2f5c4db8b53320daff3a653ae))


### Documentation

* theta is negative calendar theta, minus dPrice/dT ([#35](https://github.com/yipjunkai/pyvolr/issues/35)) ([ce9d5b9](https://github.com/yipjunkai/pyvolr/commit/ce9d5b9636e4f27e20a9db140d748caad97ec7ac))

## [0.1.4](https://github.com/yipjunkai/pyvolr/compare/v0.1.3...v0.1.4) (2026-06-12)


### Bug Fixes

* **core:** correct CDF tail, zero-vol delta, and deep-OTM pricing accuracy ([#19](https://github.com/yipjunkai/pyvolr/issues/19)) ([240844e](https://github.com/yipjunkai/pyvolr/commit/240844e8e2bad924719002e9a36340c1a82fe477))


### Benchmarks

* split the experiment harness out of the perf gate ([#26](https://github.com/yipjunkai/pyvolr/issues/26)) ([91f11eb](https://github.com/yipjunkai/pyvolr/commit/91f11eb38825bf093a1d20f81eefe8775e4a0b8f))

## [0.1.3](https://github.com/yipjunkai/pyvolr/compare/v0.1.2...v0.1.3) (2026-05-29)


### Performance

* mechanical-sympathy audit + 2026 competitor positioning ([#11](https://github.com/yipjunkai/pyvolr/issues/11)) ([002c763](https://github.com/yipjunkai/pyvolr/commit/002c76331b983df9be12523d8198c848fc72dbe3))

## [0.1.2](https://github.com/yipjunkai/pyvolr/compare/v0.1.1...v0.1.2) (2026-05-26)


### Features

* **iv:** replace Newton/bisection with Jäckel "Let's Be Rational" ([#10](https://github.com/yipjunkai/pyvolr/issues/10)) ([4fe0ff1](https://github.com/yipjunkai/pyvolr/commit/4fe0ff14d5d83576bb7fea1239a69d085646b0d6))

## [0.1.1](https://github.com/yipjunkai/pyvolr/compare/v0.1.0...v0.1.1) (2026-05-25)


### Features

* **black76:** add Black-76 pricing model for futures and forward options ([#5](https://github.com/yipjunkai/pyvolr/issues/5)) ([0b86681](https://github.com/yipjunkai/pyvolr/commit/0b86681dd1d05ca8c85c5e7617ee96ee6326cf8b))
