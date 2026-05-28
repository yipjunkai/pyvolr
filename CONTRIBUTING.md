# Contributing to pyvolr

Thanks for your interest. pyvolr is solo-maintained — clear, focused contributions help enormously. The repo's Discussions, Wiki, Projects, and Sponsorship surfaces are intentionally disabled while it stays this size; **Issues and Pull Requests are the only inbound channels**.

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
- [ ] Conventional commit message (`feat:` / `fix:` / `docs:` / ...) — release-please reads these to generate the changelog automatically; don't hand-edit `CHANGELOG.md`
- [ ] `cargo fmt && cargo clippy && ruff check && ruff format && pyright` all clean
- [ ] py_vollib compat preserved (if relevant): existing compat shim tests still pass

## Adding a new pricing model

`crates/core/src/black76.rs` is the most recent worked example — copy its structure.

1. Implement the pricer in a new Rust module under `crates/core/src/<model>.rs`. If the math is a specialization of BSM (as Black-76 is), delegate to `bsm`/`greeks` rather than duplicating closed-form code; only carry your own implementation for Greeks that genuinely diverge.
2. Expose batched f64-array entry points in `crates/core/src/lib.rs`. Reuse the `define_price_or_greek!` macro (or the `define_black76!` variant) if the arity matches. For high-per-row-cost functions (`iv::solve`-class, ~280ns/row) or bundled multi-output kernels (`greeks::all`-class), don't use the macro — copy the hand-written pattern from `bsm_iv` / `bsm_greeks`: a `work` closure, `py.detach(|| (0..n).into_par_iter().map(work).collect())` above `PARALLEL_THRESHOLD` (1024 for IV) or `GREEKS_PARALLEL_THRESHOLD` (4096 for Greeks), and the serial fallback below. The threshold constants in `lib.rs` document the per-row-cost / break-even rationale.
3. Add a Python wrapper at `python/pyvolr/<model>.py` mirroring the shape of `bs.py` / `black76.py` (numpy broadcasting via the helpers re-exported from `pyvolr.bs`).
4. Add type stubs in `python/pyvolr/_core.pyi`.
5. Export the new module from `python/pyvolr/__init__.py`.
6. If a `py_vollib` equivalent exists, mirror its tree under `python/pyvolr/compat/py_vollib/<model>/` preserving the upstream signatures and unit conventions (per-day theta, per-1%-vol vega, per-1%-rate rho).
7. Add tests:
   - Single-value golden cases from a textbook (Hull, Wilmott, McDonald) or from the reference library's doctests
   - Property tests in `tests/test_property.py` (put-call parity if applicable, monotonicities, asymptotic limits)
   - Differential cases in `tests/test_differential.py` if there's a py_vollib equivalent (gates the "matches reference" claim)
   - A fuzz target under `fuzz/fuzz_targets/<model>_price.rs` and add it to the matrix in `.github/workflows/fuzz.yml`
   - If applicable, an adapter in `bench/sanity_check_competitors.py` so the new model's outputs get cross-checked against the live competitor set on the next manual sweep

## Adding a Greek

1. Add the analytical formula in `crates/core/src/greeks.rs` (or in the model-specific module if it doesn't apply to BSM).
2. Vectorized entry point in `crates/core/src/lib.rs`.
3. Python wrapper in the relevant model module (`python/pyvolr/bs.py`, `python/pyvolr/black76.py`, ...).
4. Test against a finite-difference approximation of the relevant price function (tolerance ~1e-5).
5. If it's a standard Greek that ships in `bs.greeks()` / `black76.greeks()`, also extend the single-pass kernels:
   - `greeks::all` (and `black76::all`) — share `d1_d2` / discount factors / `cdf` / `pdf` with the existing Greeks rather than recomputing
   - `GreeksTuple` arity + the bundled `bsm_greeks` / `black76_greeks` PyO3 entries in `crates/core/src/lib.rs`
   - The matching `_core.pyi` stub return-type tuples
   - Extend the `all_matches_individual_at_grid` parity test in both modules so the bundled kernel stays bit-equal to the per-Greek functions

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

## Before large PRs

For anything beyond a bug fix or small feature, please open a draft issue first. The maintenance budget is finite; ensuring scope alignment up-front avoids wasted work.
