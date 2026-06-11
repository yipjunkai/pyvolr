# Numerical stability policy

How pyvolr versions and discloses changes to its numerical **outputs** — the
prices, Greeks, and implied vols it returns — so that upgrading never silently
moves your numbers.

## The guarantee

pyvolr's contract is **correctness to a documented tolerance, not bit-for-bit
reproducibility across versions.** On every well-posed input pyvolr agrees with
the analytic value (and with the reference libraries) to f64 precision, ~1e-13
relative, and stays ~1 ULP accurate into the deep-OTM / short-expiry tail where
other libraries underflow — see the **Numerical agreement** section of the
[README](../README.md). The `compat.py_vollib` shim agrees with `py_vollib` to
within `1e-8`.

That *tolerance* is what we hold stable — not a specific bit pattern. pyvolr
reserves the right to make an output **more** correct, and a patch release may
do exactly that. If you assert on pyvolr's outputs in a regression test, assert
with a tolerance; do not pin them for exact equality across versions.

## How an output change is versioned

pyvolr is pre-1.0, so release-please bumps `feat`/`fix` as a **patch** and any
breaking change as a **minor** (`bump-minor-pre-major`). After 1.0 a breaking
change bumps **major**, per semver. Within that, an output change is one of:

| Kind of change | Example | Bump | Disclosed as |
| --- | --- | --- | --- |
| **Accuracy refinement** — same inputs, a more-correct number, still inside the tolerance contract | [#19](https://github.com/yipjunkai/pyvolr/pull/19) (deep-OTM pricing, CDF tail) | `fix:` / `perf:` -> **patch** | subject states the change; body quantifies it |
| **Behaviour / contract change** — a different return value, exception, signature, or unit, or a redefinition of the tolerance itself | [#21](https://github.com/yipjunkai/pyvolr/pull/21) (`nan` -> raises `py_vollib`'s bound exceptions) | breaking -> **minor** | `BREAKING CHANGE:` footer in the changelog |
| **No output change** — refactor, or a `perf` change that is bit-identical | — | `refactor:` / `perf:` -> patch | normal |

The line between the first two rows: if correct code keeps working and merely
gets better numbers, it is an accuracy refinement (patch). If code that relied
on the documented behaviour could break — a caught `nan`, a return type, a unit
— it is a contract change (minor pre-1.0, major after).

## How an output change is disclosed

There is **no dedicated "Numerical changes" changelog section** — output
changes ride the normal `Bug Fixes` / `Performance` sections that release-please
generates from commit subjects. To keep them visible without a bespoke section:

1. **The subject states the output change** — e.g. *"fix(core): correct
   deep-OTM put prices (values change for d1 < -5)"*. release-please copies the
   subject verbatim into `CHANGELOG.md`, so the change is greppable there and
   the commit link carries the detail.
2. **The body quantifies it** — which inputs move, by how much (ULP / relative
   / absolute), and how the new values were validated.
3. **The guards move deliberately** — an accuracy fix updates the affected
   mpmath goldens in the same commit (the test suite enforces them on every
   PR), the new values are justified in the body, and the 20,448-case
   differential against `py_vollib` must still agree.

[#19](https://github.com/yipjunkai/pyvolr/pull/19) is the model: an
accuracy-naming subject, a body explaining the corrected tail, and updated
goldens.

## See also

- [CONTRIBUTING.md](../CONTRIBUTING.md) — the contributor checklist that enforces this.
- [README.md](../README.md) — the published **Numerical agreement** figures.
- [SECURITY.md](../SECURITY.md) — correctness disagreements *within* this tolerance are regular issues, not security reports.
