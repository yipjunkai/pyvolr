# Benchmarks

The scripts behind the README's performance section — speed, throughput, and
deep-OTM implied-vol accuracy against the live competitor field. Everything here
is **dev-only** and never runs in CI; the result caches (`bench/.*_results.json`)
are git-ignored and machine-specific.

## Quick start

You need [`just`](https://just.systems) and [`uv`](https://docs.astral.sh/uv).
uv fetches the pinned Python versions and builds the environments on demand
(cached, ephemeral) — there are no venvs to create, activate, or clean.

```bash
just            # list recipes
just all        # build everything: the table + both charts
```

Or one piece at a time:

```bash
just table        # the price / IV / greeks speed table
just throughput   # docs/assets/perf-competitors-*.svg
just accuracy     # docs/assets/accuracy-iv-tail-*.svg
just sanity       # numerical cross-validation (not a benchmark)
just clean        # remove the result caches
```

By default `pyvolr` is installed from PyPI (latest release). To benchmark a
**local build** — e.g. to check a change before pushing — point it at the repo:

```bash
just pyvolr_with='--with-editable .' all    # compiles the Rust core (needs Rust + maturin)
```

> Absolute timings are hardware-specific — the committed numbers are an Apple M4
> Pro. **Ratios between libraries are what reproduce across machines.**

## Why three environments

The competitor set can't coexist in one environment, so each chart is stitched
from sweeps run in three (each script measures whatever imports and skips the
rest). The `justfile` builds them with `uv run --with` from the pins declared there:

| environment            | Python | libraries under test                                    | why isolated                                        |
| ---------------------- | :----: | ------------------------------------------------------- | --------------------------------------------------- |
| entrants               |  3.12  | pyvolr · vollib · opengreeks · fast-vollib · (renders)  | the modern-numpy stack; also holds matplotlib       |
| legacy                 |  3.11  | pyvolr · py_vollib_vectorized · blackscholes · QuantLib | py_vollib_vectorized (2021) pins an old numba/numpy |
| quantforge             |  3.12  | pyvolr · quantforge                                     | quantforge pins numpy and needs Python ≥ 3.12       |

`pyvolr` is installed identically in all three (so there's a single pyvolr line);
the competitor versions are pinned in the `justfile`'s `--with` specs to what the
README's numbers were measured against. Most are frozen upstream — to refresh
one, bump its pin and re-run, then re-measure the affected chart.

**The `py_vollib_vectorized` time capsule.** Its 2021 release only works against
an exact, mutually-pinned legacy stack (old `numba`, old `py_lets_be_rational`),
and that old `py_lets_be_rational` imports `DBL_MIN`/`DBL_MAX` from CPython's
internal `_testcapi` — the very `ModuleNotFoundError` [docs/why.md](../docs/why.md)
is about, which uv-managed Pythons don't ship. The `justfile` puts
`bench/shims/_testcapi.py` on `PYTHONPATH` to supply the two correct constants
(the two-line fix upstream never released). Reproducing a dead library takes a
shim; running pyvolr takes `pip install pyvolr`. If any of this still fails on
your machine, the sweep just skips `py_vollib_vectorized` and every other line
is unaffected.

## What each script produces

| Script                          | Output                                                                   |
| ------------------------------- | ------------------------------------------------------------------------ |
| `compare_new_entrants.py`       | The speed **table** — price / IV / greeks at N = 1 … 1M vs the 2026 field |
| `compare_competitors.py`        | The **throughput chart** (8 libraries): `bench` sweeps, `chart` renders   |
| `compare_tail_accuracy.py`      | The **IV accuracy chart**: `sweep` per env, `chart` renders               |
| `sanity_check_competitors.py`   | Numerical cross-validation over a 9-cell grid (a correctness check)       |
| `compare_py_vollib.py`          | Legacy pyvolr-vs-py_vollib scalar reproducer (predates the field above)   |

## Without `just`

Every recipe is a few `uv run --with …` lines — copy them straight from the
`justfile` (the `entrants` / `legacy` / `quantforge` variables hold each
environment's `--python` + `--with` pins). The pattern is one ephemeral
environment per environment; sweep the chart scripts in **every** one, then
render once from the entrants environment. For example, the accuracy chart's
modern-env sweep by hand:

```bash
export PYTHONPATH="$PWD/bench/shims"   # the _testcapi shim (see above)
uv run --no-project --python 3.12 \
  --with 'vollib==1.0.11' --with 'opengreeks==0.2.0' \
  --with 'fast-vollib[numba]==0.1.6' --with 'matplotlib>=3.8' --with pyvolr \
  -- python bench/compare_tail_accuracy.py sweep
# ...repeat for the legacy and quantforge envs, then `... chart`.
```
