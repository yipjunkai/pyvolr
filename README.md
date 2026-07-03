# pyvolr

[![PyPI](https://img.shields.io/pypi/v/pyvolr.svg)](https://pypi.org/project/pyvolr/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyvolr.svg)](https://pypi.org/project/pyvolr/)
[![Wheel](https://img.shields.io/pypi/wheel/pyvolr.svg)](https://pypi.org/project/pyvolr/#files)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/yipjunkai/pyvolr/badge)](https://scorecard.dev/viewer/?uri=github.com/yipjunkai/pyvolr)
[![CI](https://github.com/yipjunkai/pyvolr/actions/workflows/ci.yml/badge.svg)](https://github.com/yipjunkai/pyvolr/actions/workflows/ci.yml)
[![License](https://img.shields.io/pypi/l/pyvolr.svg)](#-license)

**Modern Black-Scholes-Merton pricing, Greeks, and implied volatility for Python.** Rust core. Vectorized. Correct in the tails. Drop-in compatible with `py_vollib`/`vollib`.

```python
from pyvolr import bs

bs.price("c", S=100, K=105, T=0.5, r=0.05, sigma=0.2) # 4.581680167540007
```

## ⚡ Performance

<table>
<tr>
<td>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/yipjunkai/pyvolr/main/docs/assets/perf-competitors-thru-dark.svg">
  <img alt="Throughput: pyvolr vs the active BSM-pricing ecosystem, log-log by array size" src="https://raw.githubusercontent.com/yipjunkai/pyvolr/main/docs/assets/perf-competitors-thru-light.svg">
</picture>

</td>
<td>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/yipjunkai/pyvolr/main/docs/assets/accuracy-iv-tail-dark.svg">
  <img alt="Implied-vol recovery accuracy (correct digits, higher is better) vs option-price depth: pyvolr stays f64-exact down to prices of 1e-215 while the fast 2026 entrants and QuantLib fall to zero correct digits between 1e-6 and 1e-26" src="https://raw.githubusercontent.com/yipjunkai/pyvolr/main/docs/assets/accuracy-iv-tail-light.svg">
</picture>

</td>
</tr>
</table>

pyvolr **owns implied vol at scale** — 2× faster than fast-vollib's numba backend and 12× faster than opengreeks at 1M solves — and ties fast-vollib on the bundled five-Greeks kernel. It concedes bulk `price` throughput to fast-vollib's multithreaded numba kernels; its scalar-call latency, once 20–30× behind opengreeks' dedicated FFI, closed to under 2.5× with the 0.1.6 scalar fast path (both now sub-microsecond). The right chart is why that trade is worth making: pushed deep out-of-the-money, every fast 2026 entrant starts returning **silently wrong** implied vols (a wrong-constant σ, with onset between prices of ~1e-6 and ~1e-26), while pyvolr — with the other two Let's-Be-Rational descendants, vollib and py_vollib_vectorized — stays f64-exact down to prices of 1e-215.

| Workload                      |       pyvolr | fast-vollib 0.1.6 ¹ | opengreeks 0.2.0 | vollib 1.0.11 ² |
| ----------------------------- | -----------: | ------------------: | ---------------: | --------------: |
| `bs.price`, scalar ⁴          |      0.25 µs |               88 µs |      **0.13 µs** |          1.5 µs |
| `bs.price`, 10k strikes       |       326 µs |          **174 µs** |           223 µs |         14.7 ms |
| `bs.price`, 1M strikes        |      31.9 ms |          **4.4 ms** |          21.9 ms |          1.53 s |
| `bs.implied_vol`, scalar      |      0.50 µs |              156 µs |      **0.21 µs** |         13.2 µs |
| `bs.implied_vol`, 10k         |   **449 µs** |              727 µs |          3.35 ms |         97.7 ms |
| `bs.implied_vol`, 1M          |  **27.1 ms** |             53.2 ms |           331 ms |        ≈9.7 s ³ |
| `bs.greeks` (all 5), 10k      |       251 µs |          **191 µs** |           566 µs |         47.0 ms |
| `bs.greeks` (all 5), 1M       |   **5.0 ms** |              5.2 ms |          56.1 ms |        ≈4.7 s ³ |

¹ The numba backend — fast-vollib's fast CPU path (a plain `pip install fast-vollib` runs its numpy backend, 5–7× slower). One-time ~0.4 s JIT warmup per function excluded.
² vollib (the revived `py_vollib` upstream — see [docs/why.md](docs/why.md)) is scalar-only pure Python: the migration baseline, not a vectorization competitor. pyvolr runs its workloads 45–360× faster in batch.
³ Extrapolated ×10 from the measured 100k-row time (scalar loop).
⁴ pyvolr's `price` uses the normalised-Black engine — ~1-ULP into the deep-OTM tail, ~2.3× slower than the textbook `S·Φ(d1) − K·Φ(d2)` form (that is opengreeks' per-core price edge). A deliberate trade; Greeks and IV are unaffected.

Also on the throughput chart: [`py_vollib_vectorized`](https://pypi.org/project/py_vollib_vectorized/) (numba, unmaintained since 2021), [`blackscholes`](https://pypi.org/project/blackscholes/) (pure Python), [`QuantLib`](https://pypi.org/project/QuantLib/) (C++ core, looped scalar), and [`quantforge`](https://pypi.org/project/quantforge/) (Rust + rayon, reduced-precision `fast_erf`; unmaintained since Sep 2025). pyvolr's `implied_vol` parallelises via rayon above N≈1k (`greeks` ≥4096); `RAYON_NUM_THREADS=1` forces serial.

**Numerical agreement:** pyvolr matches every library above to f64 precision (~1e-13) on all well-posed inputs across price + 5 Greeks + IV (`bench/sanity_check_competitors.py`). The edges differ: `blackscholes` underflows deep-OTM prices to zero and `quantforge` hard-clamps Φ at ±8σ where pyvolr's `erfcx`-based cdf keeps the ~1e-50 price; the IV tail is the right chart, methodology in `bench/compare_tail_accuracy.py` (a known-σ ladder priced through pyvolr's mpmath-golden-pinned forward map).

Reproduce it all with **`just all`**, or per-chart via the recipes in the [`justfile`](justfile) (needs [`just`](https://just.systems) + [`uv`](https://docs.astral.sh/uv); uv builds the pinned environments on demand). Measured on an Apple M4 Pro: scalar rows on pyvolr 0.1.6 (the scalar fast path), vector rows and both charts unchanged from 0.1.5 — the fast path leaves the vectorized and IV code paths untouched. Competitor versions are pinned in the justfile.

## 📦 Install

```bash
pip install pyvolr
```

Or via [`uv`](https://github.com/astral-sh/uv):

```bash
uv pip install pyvolr
```

Pre-built wheels are published for Linux (x86_64, aarch64), macOS (Intel, Apple Silicon), and Windows (x86_64) across Python 3.10–3.14, plus a free-threaded build for 3.14t. (3.13t wheels were last published at pyvolr 0.1.3 — cibuildwheel 4 dropped Python 3.13 free-threading, which never left experimental status.)

### Tested on

|         | 3.10 | 3.11 | 3.12 | 3.13 | 3.14 |
| ------- | :--: | :--: | :--: | :--: | :--: |
| Linux   |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |
| macOS   |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |
| Windows |  —   |  —   |  ✅  |  ✅  |  ✅  |

Every push and PR runs the full `pytest` + `cargo test` suites across the matrix above. Windows × {3.10, 3.11} are skipped intentionally to keep CI minutes reasonable — the wheels themselves still build for those combinations and are published. The free-threaded wheel (3.14t) is built and exercised through `cibuildwheel`'s in-wheel test pass on every release across Linux/macOS/Windows, and on packaging-touching PRs via the wheel-smoke check.

From source (requires Rust):

```bash
git clone https://github.com/yipjunkai/pyvolr
cd pyvolr
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,test]"
maturin develop --release
```

## 🚀 Quick start

```python
import numpy as np
from pyvolr import bs

# Scalar
bs.price("c", S=100, K=105, T=0.5, r=0.05, sigma=0.2)

# Vectorized — broadcast over any combination of inputs
strikes = np.linspace(80, 120, 41)
prices = bs.price("c", S=100, K=strikes, T=0.5, r=0.05, sigma=0.2)

# All five Greeks in one call
greeks = bs.greeks("c", S=100, K=strikes, T=0.5, r=0.05, sigma=0.2)
# {"delta": [...], "gamma": [...], "theta": [...], "vega": [...], "rho": [...]}

# Implied volatility from a market price
bs.implied_vol(price=5.20, flag="c", S=100, K=100, T=0.25, r=0.05)

# Broadcasting works in any dimension
strike_grid = np.linspace(80, 120, 5).reshape(-1, 1)
vol_grid = np.linspace(0.10, 0.40, 4).reshape(1, -1)
surface = bs.price("c", S=100, K=strike_grid, T=0.5, r=0.05, sigma=vol_grid)
# shape (5, 4)

# Black-76 for options on futures / forwards — same API, F replaces S, no q.
from pyvolr import black76
black76.price("c", F=100, K=105, T=0.5, r=0.05, sigma=0.2)
```

## ✨ Features

- **Black-Scholes-Merton pricing** — calls and puts with continuous dividend yield
- **Black-76 pricing** — European options on futures/forwards (`pyvolr.black76`), same vectorized API as `bs`
- **Analytical Greeks** — delta, gamma, theta, vega, rho (with documented sign and unit conventions)
- **Robust implied volatility** — Jäckel "Let's Be Rational" algorithm: rational-cubic initial guess plus Householder order-4 iteration converges to ~1e-13 precision in ≤2 iterations across the full no-arbitrage range
- **Automatic parallelism on large batches** — `implied_vol` (above N≈1,000 rows) and the bundled `greeks` kernel (above N≈4,000) release the GIL and dispatch per-row work to rayon's global thread pool; set `RAYON_NUM_THREADS=1` to opt out
- **Full numpy broadcasting** — any combination of inputs in any shape, scalar-in scalar-out
- **`py_vollib` drop-in shims** — `pyvolr.compat.py_vollib` mirrors the upstream module tree (including `py_vollib.black`) for one-import-line migration; `pyvolr.compat.py_vollib_vectorized` mirrors the vectorized API (`vectorized_*`, `get_all_greeks`, `price_dataframe`)
- **Rust core, no compiler needed** — abi3 wheels for Python 3.10–3.14 × {Linux, macOS, Windows}
- **Free-threaded Python ready** — a dedicated 3.14t wheel: with no GIL, every entry point scales across threads. On standard (GIL) builds, the large-batch `implied_vol` (≥1k rows) and bundled `greeks` (≥4k rows) kernels release the GIL while rayon works; `price` and single-Greek calls hold it
- **Typed end-to-end** — pyright-strict library code, full type stubs for the Rust extension

## 🗺️ Coming soon

- [ ] Bachelier (normal model, for negative rates) — with analytic implied-normal-vol inversion
- [ ] Higher-order Greeks (vanna, vomma, charm, speed, zomma, color)
- [ ] American options (Andersen-Lake-Offengenden spectral collocation)
- [ ] Volatility surface fitting (arbitrage-free eSSVI)

SIMD batch evaluation used to be on this list and was deliberately dropped: no vectorized math library currently meets pyvolr's precision bar on the erfc-dependent tails (~1 ULP against 60-digit references), and the crate forbids `unsafe` code. Batch throughput comes from rayon parallelism instead.

## 🔄 Migrating from py_vollib

Replace your imports — the signatures and `'c'`/`'p'` flag convention are preserved exactly:

```python
# Before
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks.analytical import delta
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_vollib.black import black  # futures options

# After
from pyvolr.compat.py_vollib.black_scholes import black_scholes
from pyvolr.compat.py_vollib.black_scholes.greeks.analytical import delta
from pyvolr.compat.py_vollib.black_scholes.implied_volatility import implied_volatility
from pyvolr.compat.py_vollib.black import black  # futures options
```

The compat shim also preserves py_vollib's _unit conventions_: vega is per-1% vol, theta is per-day, rho is per-1% rate, and `implied_volatility` takes `flag` as its last argument. For new code, prefer the modern `pyvolr.bs` API — it accepts numpy arrays, broadcasts naturally, uses per-unit conventions consistently, and returns all Greeks in a single call.

### Migrating from `py_vollib_vectorized`

`pyvolr.compat.py_vollib_vectorized` mirrors the vectorized API — the `vectorized_*` functions, `get_all_greeks`, and `price_dataframe` — with the same (quirky) argument orders, `return_as` values, and `"Price"`/`"IV"`/greek column names:

```python
# Before
from py_vollib_vectorized import vectorized_black_scholes, vectorized_implied_volatility

# After
from pyvolr.compat.py_vollib_vectorized import vectorized_black_scholes, vectorized_implied_volatility
```

Two deliberate differences: importing the shim does **not** monkeypatch `py_vollib` (call the `vectorized_*` functions directly), and `pandas` is an optional dependency — `return_as="dataframe"`/`"series"` need it (`pip install pyvolr[pandas]`) and otherwise fall back to numpy with a one-time warning.

## 🤔 Why pyvolr exists

`py_vollib` spent six years abandoned (2020–2026) and hard-broken on Python 3.12+ — a transitive dependency imported `DBL_MIN` / `DBL_MAX` from CPython's internal `_testcapi` test module. pyvolr was built so that failure mode cannot recur: the numerical core is Rust and touches no CPython internals, and the release pipeline is engineered to survive its maintainer ([GOVERNANCE.md](GOVERNANCE.md)).

In April 2026 the upstream was revived as [`vollib`](https://pypi.org/project/vollib/), which fixes the import error. It doesn't change the math: vollib remains scalar-only pure Python — 45–360× slower on the batch workloads in the table above — with no Python 3.14 or free-threaded wheels.

Full backstory, including the revival: [docs/why.md](docs/why.md).

## 📁 Project structure

Directory-level map; each module documents itself in its docstring/rustdoc header.

```text
pyvolr/
├── crates/core/     # Rust numerical core (BSM, Black-76, Greeks, Jäckel IV, Φ/erfcx) + criterion benches
├── python/pyvolr/   # numpy-broadcasting public API, type stubs, py_vollib(+vectorized) compat shims
├── tests/           # pytest + hypothesis property tests
├── bench/           # dev-only speed/accuracy benchmarks vs the competitor field (not in CI)
├── fuzz/            # cargo-fuzz targets for the numerical core
├── tools/           # regenerable mpmath golden generator
├── docs/            # backstory (why.md) + chart assets
├── .github/         # CI/release workflows (all SHA-pinned) + helper scripts
└── justfile         # benchmark-reproduction recipes (`just all`, `just perf-stat`, ...) via uv
```

## 📚 API reference

| Function                                       | Returns                    | Vectorized over        |
| ---------------------------------------------- | -------------------------- | ---------------------- |
| `bs.price(flag, S, K, T, r, sigma, q=0)`       | option price               | all numeric inputs     |
| `bs.delta(flag, S, K, T, r, sigma, q=0)`       | ∂Price/∂S                  | all numeric inputs     |
| `bs.gamma(S, K, T, r, sigma, q=0)`             | ∂²Price/∂S²                | all numeric inputs     |
| `bs.vega(S, K, T, r, sigma, q=0)`              | ∂Price/∂σ (per unit vol)   | all numeric inputs     |
| `bs.theta(flag, S, K, T, r, sigma, q=0)`       | −∂Price/∂T (per year)      | all numeric inputs     |
| `bs.rho(flag, S, K, T, r, sigma, q=0)`         | ∂Price/∂r (per unit r)     | all numeric inputs     |
| `bs.greeks(flag, S, K, T, r, sigma, q=0)`      | `dict` of all five Greeks  | all numeric inputs     |
| `bs.implied_vol(price, flag, S, K, T, r, q=0)` | σ (NaN on bound violation) | price + numeric inputs |
| `black76.price(flag, F, K, T, r, sigma)`       | option price on a forward  | all numeric inputs     |
| `black76.{delta,gamma,vega,theta,rho}(...)`    | Greeks for Black-76        | all numeric inputs     |
| `black76.greeks(flag, F, K, T, r, sigma)`      | `dict` of all five Greeks  | all numeric inputs     |
| `black76.implied_vol(price, flag, F, K, T, r)` | σ (NaN on bound violation) | price + numeric inputs |
| `pyvolr.compat.py_vollib.…`                    | py_vollib-shaped scalars   | n/a (scalar API)       |
| `pyvolr.compat.py_vollib_vectorized.…`         | arrays / DataFrame         | all numeric inputs     |

`flag` accepts `'c'`/`'C'` (call), `'p'`/`'P'` (put), or an array thereof.

Every function also accepts a keyword-only `return_as`: `"numpy"` (default — array, or scalar for scalar input), `"dict"` (`{name: value}`), or `"dataframe"` (a pandas DataFrame; `pandas` is an optional dependency, installed only if you use this mode).

`implied_vol` additionally accepts a keyword-only `on_error` for unsolvable inputs (price outside the no-arbitrage bounds, non-positive `T`/`S`/`K`, or non-finite input): `"warn"` (default — emit an `ImpliedVolWarning` and return NaN), `"raise"` (raise `ImpliedVolError`), or `"ignore"` (return NaN silently).

## 🛡️ Sustainability

`py_vollib` went dark for six years because nobody was paid to maintain it — its 2026 revival came only after the ecosystem had moved on. pyvolr is engineered to outlive its maintainer:

- **One-click releases** via release-please + PyPI Trusted Publishing — PyPI publication needs no stored credentials (OIDC), and release-please authenticates as a repo-scoped GitHub App rather than a user PAT, so the credential survives a maintainer handoff
- **Release-gated differential tests** against `py_vollib` (Python 3.10 sidecar) — every release is blocked unless pyvolr still matches the reference
- **Wide CI matrix** (Python 3.10–3.14 × Linux/macOS/Windows) — the specific failure mode that killed the predecessor
- **All GitHub Actions pinned** with weekly Dependabot bumps, hardening against supply-chain attacks
- **Hand-off plan documented** in [GOVERNANCE.md](GOVERNANCE.md)

Commercial sponsorship channels will be added if demand warrants. For now the best support is real-world use, good bug reports, and PRs.

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Particularly welcome: new pricing models (Bachelier, American), higher-order Greeks, and property tests for edge cases.

## 📄 License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE), at your option.

Algorithms are reimplemented from published references (Hull, Merton, Jäckel); no third-party source code is incorporated.
