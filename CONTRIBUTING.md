# Contributing to pyvolr

Thanks for your interest. pyvolr is maintained by a small team (initially one person) — clear, focused contributions help enormously.

## Quick start

```bash
git clone https://github.com/yipjunkai/pyvolr
cd pyvolr

# Create a Python 3.12 venv with all dev deps
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev,test]"

# Build the Rust extension into the venv (develop mode)
maturin develop --release

# Run the test suite
pytest
```

For day-to-day work, `maturin develop` (no `--release`) is faster but slower at runtime.

## Pull request checklist

- [ ] Tests added (and run green locally)
- [ ] If algorithmic: property test added (parity, monotonicity, roundtrip, etc.)
- [ ] If numerical: golden reference value documented (textbook, paper, or trusted external)
- [ ] If a public API change: type stubs in `python/pyvolr/_core.pyi` updated
- [ ] `CHANGELOG.md` entry under `[Unreleased]`
- [ ] `cargo fmt && cargo clippy && ruff check && ruff format && pyright` all clean
- [ ] py_vollib compat preserved (if relevant): existing compat shim tests still pass

## Adding a new pricing model

1. Implement the closed-form (or numerical) pricer in a new Rust module under `crates/core/src/`
2. Expose batched f64-array entry points in `crates/core/src/lib.rs`
3. Add a Python wrapper in `python/pyvolr/api.py` with numpy broadcasting
4. Add type stubs in `python/pyvolr/_core.pyi`
5. Add tests:
   - Single-value golden cases from a textbook (Hull, Wilmott, McDonald)
   - Property tests (put-call parity if applicable, monotonicities, asymptotic limits)
   - Vectorized roundtrip tests

## Adding a Greek

1. Add analytical formula in `crates/core/src/greeks.rs`
2. Vectorized entry point in `crates/core/src/lib.rs`
3. Python wrapper in `python/pyvolr/api.py`
4. Test against finite-difference approximation of the relevant price function (tolerance ~1e-5)
5. Update `greeks()` dict-returning function if it's a standard Greek

## Numerical correctness expectations

- All public functions must handle the edge cases: `T=0`, `sigma=0`, deep ITM, deep OTM. Document the chosen behavior in the docstring.
- Implied volatility must converge or return `nan` — never silently return a wrong value.
- Property tests are required for new public API.

## Reporting py_vollib drift

If you find an input where `pyvolr.compat.py_vollib.X(...)` returns a value that disagrees with `py_vollib.X(...)` by more than 1e-8, please open an issue using the "py_vollib drift" template with the exact inputs. These are treated as bugs.

## Style

- Rust: `rustfmt` defaults + the configured clippy lints
- Python: `ruff format` + `ruff check`, type-checked under `pyright` strict mode
- Conventional commits (`feat:`, `fix:`, `docs:`, `chore:`, `perf:`, `refactor:`, `test:`) — release-please uses these to generate the changelog and bump versions

## Discussion before large PRs

For anything beyond a bug fix or small feature, please open a discussion or draft issue first. The maintenance budget is finite; ensuring scope alignment up-front avoids wasted work.
