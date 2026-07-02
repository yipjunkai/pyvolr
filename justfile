# Benchmark reproduction for pyvolr — details in bench/README.md.
#
# Requires `just` and `uv`. uv fetches the pinned Python versions and builds the
# (cached, ephemeral) environments on demand from the `--with` specs below, so
# there are no venvs and no requirements files to manage.
#
#   just              # list recipes
#   just all          # the speed table + both charts
#   just accuracy     # just the deep-OTM IV-accuracy chart
#   just pyvolr_with='--with-editable .' all   # benchmark a LOCAL build (needs Rust + maturin)
#
# Absolute timings are hardware-specific (the committed numbers are an Apple M4
# Pro); ratios between libraries are what reproduce across machines.

# pyvolr under test — the released wheel by default; override for a local build.
pyvolr_with := "--with pyvolr"

# The legacy env's py_lets_be_rational 1.0.1 imports DBL_MIN/DBL_MAX from
# CPython's internal `_testcapi` (the bug docs/why.md is about), which uv's
# Pythons omit. Supply the two correct constants; unused by the other envs.
export PYTHONPATH := justfile_directory() / "bench/shims"

run := "uv run --no-project"

# The three environments the charts are stitched from — competitor versions
# pinned to what the README's numbers were measured against. The split is
# inherent: py_vollib_vectorized's old numba stack can't coexist with the modern
# one, and quantforge pins numpy and needs Python >= 3.12.
entrants := "--python 3.12 --with 'vollib==1.0.11' --with 'opengreeks==0.2.0' --with 'fast-vollib[numba]==0.1.6' --with 'matplotlib>=3.8'"
quantforge := "--python 3.12 --with 'quantforge==0.1.1'"
# Exact, mutually-compatible legacy pins (anything newer breaks py_vollib_vectorized 0.1.1):
# numba 0.66 fails its jitted kernels; the revived py_lets_be_rational (>=1.1) dropped the
# numba internals it calls, so both are pinned old (the _testcapi shim above covers the rest).
legacy := "--python 3.11 --with 'py_vollib_vectorized==0.1.1' --with 'py_vollib==1.0.7' --with 'py_lets_be_rational==1.0.1' --with 'numba==0.65.1' --with 'blackscholes==0.2.0' --with 'QuantLib==1.42.1' --with 'matplotlib>=3.8'"

# list recipes
_default:
    @just --list

# the speed table + both charts
all: table throughput accuracy
    @echo "Done — charts written to docs/assets/*.svg"

# speed table: price / IV / greeks vs the 2026 field (modern env only)
table:
    rm -f bench/.new_entrants_results.json
    {{run}} {{entrants}} {{pyvolr_with}} -- python bench/compare_new_entrants.py

# throughput chart (8 libraries): sweep every env, then render from the modern env
throughput:
    rm -f bench/.competitor_results.json
    {{run}} {{entrants}}   {{pyvolr_with}} -- python bench/compare_competitors.py bench
    {{run}} {{legacy}}     {{pyvolr_with}} -- python bench/compare_competitors.py bench
    {{run}} {{quantforge}} {{pyvolr_with}} -- python bench/compare_competitors.py bench
    {{run}} {{entrants}}   {{pyvolr_with}} -- python bench/compare_competitors.py chart

# deep-OTM implied-vol accuracy chart: sweep every env, then render
accuracy:
    rm -f bench/.tail_accuracy_results.json
    {{run}} {{entrants}}   {{pyvolr_with}} -- python bench/compare_tail_accuracy.py sweep
    {{run}} {{legacy}}     {{pyvolr_with}} -- python bench/compare_tail_accuracy.py sweep
    {{run}} {{quantforge}} {{pyvolr_with}} -- python bench/compare_tail_accuracy.py sweep
    {{run}} {{entrants}}   {{pyvolr_with}} -- python bench/compare_tail_accuracy.py chart

# numerical cross-validation over a 9-cell grid (a correctness check, not a benchmark)
sanity:
    {{run}} {{entrants}}   {{pyvolr_with}} -- python bench/sanity_check_competitors.py
    {{run}} {{legacy}}     {{pyvolr_with}} -- python bench/sanity_check_competitors.py
    {{run}} {{quantforge}} {{pyvolr_with}} -- python bench/sanity_check_competitors.py

# remove the local result caches (they merge across runs otherwise)
clean:
    rm -f bench/.competitor_results.json bench/.new_entrants_results.json bench/.tail_accuracy_results.json
