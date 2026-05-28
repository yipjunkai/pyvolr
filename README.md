# pyvolr

[![PyPI](https://img.shields.io/pypi/v/pyvolr.svg)](https://pypi.org/project/pyvolr/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyvolr.svg)](https://pypi.org/project/pyvolr/)
[![Wheel](https://img.shields.io/pypi/wheel/pyvolr.svg)](https://pypi.org/project/pyvolr/#files)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/yipjunkai/pyvolr/badge)](https://securityscorecards.dev/viewer/?uri=github.com/yipjunkai/pyvolr)
[![CI](https://github.com/yipjunkai/pyvolr/actions/workflows/ci.yml/badge.svg)](https://github.com/yipjunkai/pyvolr/actions/workflows/ci.yml)
[![License](https://img.shields.io/pypi/l/pyvolr.svg)](#-license)

**Modern Black-Scholes-Merton pricing, Greeks, and implied volatility for Python.** Rust core. Vectorized. Drop-in replacement for the abandoned `py_vollib`.

```python
from pyvolr import bs

bs.price("c", S=100, K=105, T=0.5, r=0.05, sigma=0.2) # 4.581680167540007
```

## ⚡ Performance

<table>
<tr>
<td>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/yipjunkai/pyvolr/main/docs/assets/perf-competitors-time-dark.svg">
  <img alt="Time per call: pyvolr vs the active BSM-pricing ecosystem, log-log by array size" src="https://raw.githubusercontent.com/yipjunkai/pyvolr/main/docs/assets/perf-competitors-time-light.svg">
</picture>

</td>
<td>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/yipjunkai/pyvolr/main/docs/assets/perf-competitors-thru-dark.svg">
  <img alt="Throughput: pyvolr vs the active BSM-pricing ecosystem, log-log by array size" src="https://raw.githubusercontent.com/yipjunkai/pyvolr/main/docs/assets/perf-competitors-thru-light.svg">
</picture>

</td>
</tr>
</table>

Six libraries on the chart: **`pyvolr`**, [`vollib`](https://pypi.org/project/vollib/) (resurrected upstream of `py_vollib`, pure Python), [`py_vollib_vectorized`](https://pypi.org/project/py_vollib_vectorized/) (numba), [`blackscholes`](https://pypi.org/project/blackscholes/) (pure Python, object-per-call), [`QuantLib`](https://pypi.org/project/QuantLib/) (C++ core, looped scalar), and [`quantforge`](https://pypi.org/project/quantforge/) (Rust + SIMD).

pyvolr leads at every input size up to ~1M strikes. **quantforge overtakes at very large batches** via explicit SIMD vectorisation — the same axis [fast-vollib](https://arxiv.org/abs/2604.27210) takes with Triton kernels. pyvolr's positioning is explicitly the "Rust-cored CPU option" — no `unsafe` SIMD intrinsics, no GPU dependency, abi3 wheel ships in one file. If you're pricing 10M+ strikes per call and CPU-only, prefer quantforge.

| Scenario                       |   pyvolr |  py_vollib |  speedup |
| ------------------------------ | -------: | ---------: | -------: |
| `bs.price`, scalar             |   4.2 µs |     2.2 µs |     0.5× |
| `bs.price`, 1k strikes         |  24.6 µs |    2.32 ms |      94× |
| `bs.price`, 10k strikes        |   153 µs |   23.32 ms |     152× |
| `bs.price`, 100k strikes       |  1.39 ms |  234.91 ms |     169× |
| `bs.price`, 1M strikes         | 14.54 ms |   2,350 ms |     162× |
| `bs.greeks` (all 5), 10k       |   273 µs |   89.95 ms |     330× |
| `bs.implied_vol`, scalar       |   4.4 µs |    15.0 µs |     3.4× |
| `bs.implied_vol`, 10k strikes  |   465 µs | ≈ 150 ms ¹ |   ≈ 323× |
| `black76.price`, scalar        |   3.7 µs |     2.2 µs |     0.6× |
| `black76.price`, 10k strikes   |   141 µs |   23.19 ms |     164× |
| `black76.implied_vol`, scalar  |   3.9 µs |    14.7 µs |     3.8× |

¹ py_vollib's `implied_volatility` is scalar-only; the 10k figure is `N × scalar`. pyvolr's vectorised path also parallelises automatically above N=1024 via rayon — set `RAYON_NUM_THREADS=1` to force serial.

The table above is the headline-vs-the-abandoned-upstream comparison (py_vollib's last release is broken on Python 3.12+, see [docs/why.md](docs/why.md)). For the workload most people actually run — a smile, an option chain, an IV snapshot — pyvolr is faster than every actively-maintained alternative and installs cleanly on every modern Python.

`bs.greeks` returning all five Greeks at once uses a single-pass Rust kernel that shares `d1`/`d2`, discount factors, `cdf`, and `pdf` across the five outputs — ~3× faster than the equivalent five separate calls. For batches ≥4096 rows, the work also dispatches across CPU cores in parallel.

**Numerical agreement:** pyvolr matches every library above to f64 precision (~1e-13 relative) on every well-posed input across price + 5 Greeks + IV. At deep-OTM short-expiry corners pyvolr is *more* precise than the rest — `blackscholes` and `quantforge` underflow to zero where pyvolr's `erfcx`-based cdf retains the ~1e-50 price; QuantLib and the alternatives lose 1-2 digits. Run `python bench/sanity_check_competitors.py` in each venv to re-validate.

Reproduce the table with `python bench/compare_py_vollib.py`; reproduce the chart with `python bench/compare_competitors.py bench` then `python bench/compare_competitors.py chart` (across the Python 3.11 + 3.12 venvs documented in the script's docstring). Library versions: Apple M4 Pro / Python 3.10.20 / numpy 2.2.6 / pyvolr 0.1.2 / py_vollib 1.0.1 (table) / vollib 1.0.7 / py_vollib_vectorized 0.1.1 / blackscholes 0.2.0 / QuantLib 1.42.1 / quantforge 0.1.1 (chart).

## 📦 Install

```bash
pip install pyvolr
```

Or via [`uv`](https://github.com/astral-sh/uv):

```bash
uv pip install pyvolr
```

Pre-built wheels are published for Linux (x86_64, aarch64), macOS (Intel, Apple Silicon), and Windows (x86_64) across Python 3.10–3.14, plus free-threaded builds for 3.13t and 3.14t.

### Tested on

|         | 3.10 | 3.11 | 3.12 | 3.13 | 3.14 |
| ------- | :--: | :--: | :--: | :--: | :--: |
| Linux   |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |
| macOS   |  ✅  |  ✅  |  ✅  |  ✅  |  ✅  |
| Windows |  —   |  —   |  ✅  |  ✅  |  ✅  |

Every push and PR runs the full `pytest` + `cargo test` suites across the matrix above. Windows × {3.10, 3.11} are skipped intentionally to keep CI minutes reasonable — the wheels themselves still build for those combinations and are published. Free-threaded wheels (3.13t, 3.14t) are built and exercised through `cibuildwheel`'s in-wheel test pass on every release across Linux/macOS/Windows.

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
- **`py_vollib` drop-in shim** — `pyvolr.compat.py_vollib` mirrors the upstream module tree (including `py_vollib.black`) for one-import-line migration
- **Rust core, no compiler needed** — abi3 wheels for Python 3.10–3.14 × {Linux, macOS, Windows}
- **Free-threaded Python ready** — dedicated wheels for 3.13t and 3.14t; the Rust core releases the GIL around the math, so pricing scales across threads without a process pool
- **Typed end-to-end** — pyright-strict library code, full type stubs for the Rust extension

## 🗺️ Coming soon

- [ ] Drop-in compat shim for `py_vollib_vectorized` (`vectorized_*` API + `price_dataframe`/`get_all_greeks`, pandas as soft dep)
- [ ] Bachelier (normal model, for negative rates)
- [ ] Higher-order Greeks (vanna, vomma, charm, speed, zomma, color)
- [ ] SIMD batch evaluation
- [ ] American options (CRR binomial → finite difference)
- [ ] Volatility surface fitting (SVI, SSVI)

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

## 🤔 Why pyvolr exists

`py_vollib` has been broken on Python 3.12+ since the release — a transitive dependency imports `DBL_MIN` / `DBL_MAX` from CPython's internal `_testcapi` test module, which isn't shipped with modern Python distributions. The fix is two lines (`sys.float_info.{min,max}` are the correct sources), but `py_lets_be_rational` hasn't released since 2017, `py_vollib` since 2020, and the maintainers are gone.

Full backstory: [docs/why.md](docs/why.md).

## 📁 Project structure

```text
pyvolr/
├── crates/core/             # Rust numerical core
│   ├── src/
│   │   ├── lib.rs           # PyO3 bindings (flat-array entry points)
│   │   ├── bsm.rs           # BSM pricing, d1/d2, forward price
│   │   ├── black76.rs       # Black-76 (futures options) — delegates to BSM with q=r
│   │   ├── greeks.rs        # Delta, gamma, theta, vega, rho
│   │   ├── iv.rs            # Jäckel "Let's Be Rational" IV solver (Householder-4, ≤2 iters)
│   │   └── normal.rs        # Φ / φ, erfcx (Lentz CF), inverse CDF (Wichura AS241)
│   └── benches/             # criterion benches gating the README's perf claims
├── bench/                   # Python-level speed/precision scripts (dev-only, not in CI)
│   ├── compare_py_vollib.py            # reproduces the perf table
│   ├── compare_competitors.py          # reproduces the perf chart (6 libraries)
│   └── sanity_check_competitors.py     # cross-validates numerical agreement
├── python/pyvolr/
│   ├── bs.py                # BSM public API (numpy-broadcast wrappers)
│   ├── black76.py           # Black-76 public API
│   ├── _wrappers.py         # Shared FFI helpers (broadcast, flag normalize)
│   ├── _core.pyi            # Type stubs for the Rust extension
│   └── compat/py_vollib/    # Drop-in shim mirroring py_vollib's tree
├── tests/                   # pytest + hypothesis property tests
├── .github/workflows/       # ci, release, release-please, differential, fuzz, perf, security, scorecard, stale
├── .github/scripts/         # CI helper scripts (perf-gate comparator)
├── Cargo.toml               # Rust workspace
└── pyproject.toml           # maturin build backend + project config
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

`flag` accepts `'c'`/`'C'` (call), `'p'`/`'P'` (put), or an array thereof.

## 🛡️ Sustainability

`py_vollib` died because nobody was paid to maintain it. pyvolr is engineered to outlive its maintainer:

- **One-click releases** via release-please + PyPI Trusted Publishing — PyPI publication needs no stored credentials (OIDC), and release-please authenticates as a repo-scoped GitHub App rather than a user PAT, so the credential survives a maintainer handoff
- **Nightly differential tests** against `py_vollib` on a Python 3.10 sidecar to catch numerical drift
- **Wide CI matrix** (Python 3.10–3.14 × Linux/macOS/Windows) — the specific failure mode that killed the predecessor
- **All GitHub Actions pinned** with weekly Dependabot bumps, hardening against supply-chain attacks
- **Hand-off plan documented** in [GOVERNANCE.md](GOVERNANCE.md)

Commercial sponsorship channels will be added if demand warrants. For now the best support is real-world use, good bug reports, and PRs.

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Particularly welcome: new pricing models (Bachelier, American), higher-order Greeks, SIMD/vectorization work, and property tests for edge cases.

## 📄 License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE), at your option.

Algorithms are reimplemented from published references (Hull, Merton, Jäckel); no third-party source code is incorporated.
