# Why pyvolr exists

If you arrived here from an `ImportError: No module named '_testcapi'` on `py_vollib` — that's the bug. pyvolr fixes it, and since April 2026 upgrading to the revived [`vollib`](https://pypi.org/project/vollib/) fixes the import too (see the [postscript](#postscript-june-2026-the-upstream-revival) for what that changes and what it doesn't). The longer version of the story:

## The bug

`py_vollib` is the de facto Python library for Black-Scholes-Merton pricing in the open-source quant ecosystem. It depends transitively on `py_lets_be_rational`, which has this at the top of `constants.py`:

```python
from _testcapi import DBL_MIN, DBL_MAX
```

`_testcapi` is a CPython _internal test_ module — never part of the public Python C API, and not included in most pre-built Python distributions (the official python.org installers, `uv`-managed Pythons, slim Docker images, conda-forge in many configurations).

On older CPython builds it sometimes shipped alongside the interpreter as a side effect of how the test suite was packaged. The import "worked" by accident. On Python 3.12 and newer, the test module is reliably absent from production distributions, so:

```python
>>> import py_vollib.black_scholes
Traceback (most recent call last):
  ...
ImportError: No module named '_testcapi'
```

This is not a "py_vollib bug needing a fix on your end" — it is a bug at the bottom of the dependency tree that _cannot_ be worked around in user code, because the failing import runs at module load time inside `py_lets_be_rational`.

## The fix that nobody upstream would make (until 2026)

The replacement is trivially in the standard library:

```python
# What py_lets_be_rational does today (broken):
from _testcapi import DBL_MIN, DBL_MAX

# What it should do (one-line fix):
from sys import float_info
DBL_MIN, DBL_MAX = float_info.min, float_info.max
```

This was proposed in `py_lets_be_rational` GitHub issues multiple times. For nine years, none of it shipped:

| Project                | Last release of the original era | Status today (mid-2026)                                        |
| ---------------------- | -------------------------------- | -------------------------------------------------------------- |
| `py_lets_be_rational`  | 2017                             | fix merged to source in 2024, unreleased; superseded by `lets-be-rational` under the revived org |
| `py_vollib`            | 2020                             | revived April 2026 as a deprecated shim over `vollib`          |
| `py_vollib_vectorized` | 2021                             | still unmaintained                                             |

From 2017 through early 2026 the whole tree was unmaintained, and importing `py_vollib` on Python 3.12+ was a hard error on every platform that ships a clean CPython. That window — six years with no working install and no maintainer to accept a two-line fix — is why pyvolr exists. (The revival is covered in the postscript below.)

## Why this matters beyond one library

`py_vollib` is downstream of a larger pattern. The Python quant ecosystem had a one-time burst of open-source library production from roughly 2015–2020 — much of it funded by Quantopian, which was the gravity well for both contributors and users. When Quantopian was acquihired by Robinhood and shut down in 2020, the _paid_ maintainers stopped, and the volunteer base was never large enough to replace them.

The result is that "industry standard" Python libraries for options pricing (`py_vollib`), performance analytics (`pyfolio`), factor research (`alphalens`), and backtesting (`zipline`) all went dead or survive as tiny community forks — `py_vollib` being the one to find a new maintainer, in 2026, after six dark years. The serious answer in 2026 is QuantLib (C++98-era ergonomics, heavyweight) or write the math yourself. There is no good middle ground.

## What pyvolr fixes

For options pricing specifically:

1. **Native install on every modern Python.** abi3 wheels for 3.10 through 3.14, on Linux (x86_64, aarch64), macOS (Intel, Apple Silicon), and Windows. No `_testcapi` dependency. No compile-from-source dance.
2. **Rust core.** Numerical kernels in Rust via PyO3. The `_testcapi` failure mode cannot reoccur because no part of the math touches CPython internals. Future Python versions are extremely unlikely to break the extension.
3. **`py_vollib` compat shim.** `pyvolr.compat.py_vollib` mirrors the upstream module layout exactly. Migration is one line per import:

   ```python
   # Before
   from py_vollib.black_scholes import black_scholes
   # After
   from pyvolr.compat.py_vollib.black_scholes import black_scholes
   ```

   Function signatures, scalar/return types, `'c'`/`'p'` flag conventions, and unit conventions (per-1%-vol vega, per-day theta, per-1%-rate rho) are all preserved.

4. **Modern API for new code.** `pyvolr.bs` accepts numpy arrays, broadcasts in any shape, returns scalars when given scalars, and uses per-unit conventions consistently. The compat shim handles the legacy.
5. **Engineered against re-abandonment.** Releases are automated via release-please + PyPI Trusted Publishing. PyPI publication needs no stored credentials (OIDC); release-please authenticates as a repo-scoped GitHub App rather than a personal access token, so the credential survives a maintainer handoff. Releases can be cut from a phone. Nightly differential tests run against `py_vollib` on a Python 3.10 sidecar to catch numerical drift. The full CI matrix tests every supported Python on every supported OS. The governance plan is in [GOVERNANCE.md](../GOVERNANCE.md).

## What pyvolr doesn't fix

The wider Python quant ecosystem is full of similar abandoned-library bombs. `pyfolio`, `empyrical`, `alphalens`, `zipline`, `mlfinlab` (free version) are all in the same boat. pyvolr addresses one specific corner — Black-Scholes-Merton pricing and Greeks. Other tools to look at:

- **QuantLib-Python** for production pricing (curves, exotic instruments, calibration)
- **`quantstats`** for a maintained `pyfolio` replacement
- **`nautilus_trader`** as a newer-generation backtesting framework
- **`riskfolio-lib`** for portfolio optimization

## Postscript (June 2026): the upstream revival

The story above froze in early 2026. Two things have changed since it was written:

1. The `_testcapi` fix was merged into `py_lets_be_rational`'s source in mid-2024 — but sat unreleased, so installs stayed broken.
2. In April 2026, GammonCap revived the vollib GitHub org and shipped `vollib` 1.0.7–1.0.11 (April 30 – June 1, 2026). The canonical package name is `vollib` again, with a fixed `lets-be-rational` dependency underneath; `py_vollib` lives on as a deprecated compatibility shim. It installs cleanly on Python 3.9–3.13.

This is genuinely good news for the ecosystem, and it retires the "pyvolr is the only way out of the ImportError" pitch. What it does not retire:

- **The performance gap.** vollib is scalar-only pure Python: 45–360× slower than pyvolr on batch workloads (option chains, IV snapshots), with no vectorized API, no Python 3.14 wheels, and no free-threaded build. See the README's performance table.
- **The failure mode.** The revival came six years after the last release, by luck rather than by plan. pyvolr's Rust core cannot break the way `_testcapi` did — no part of the math touches CPython internals — and its release pipeline is designed for succession ([GOVERNANCE.md](../GOVERNANCE.md)).
- **The accuracy bar.** Both libraries descend from Jäckel's "Let's Be Rational" and both invert implied vol correctly into the deep tail (unlike the 2026 crop of fast entrants — see the README's numerical-agreement section). pyvolr does it two orders of magnitude faster at scale, to a documented ~1-ULP standard against 60-digit references.

pyvolr's positioning after the revival is simpler, not weaker: the Rust-cored, vectorized way to run py_vollib-style pricing at scale, correct at every moneyness.

## References

- Hull, J. C. (2017). _Options, Futures, and Other Derivatives_ (10th ed.). Pearson.
- Merton, R. C. (1973). Theory of rational option pricing. _Bell Journal of Economics and Management Science_, 4(1), 141–183.
- Jäckel, P. (2015). Let's Be Rational. _Wilmott_, 75, 40–53.
