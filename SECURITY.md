# Security Policy

## Supported versions

pyvolr is solo-maintained: only the **latest released minor version** receives
security fixes. Older minors are not patched — upgrade to the most recent
[release][rel] before reporting.

[rel]: https://github.com/yipjunkai/pyvolr/releases/latest

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Use GitHub's private security advisory mechanism:

1. Go to <https://github.com/yipjunkai/pyvolr/security/advisories>
2. Click "Report a vulnerability"
3. Fill in the form with as much detail as you can share

Acknowledgement is targeted within 72 hours. There is no separate email contact at this stage; the GitHub advisory channel is the only supported route.

## What to report

- Memory-safety issues at the Rust ↔ Python boundary — the numerical core sets `unsafe_code = "forbid"`, so the likeliest surface is the `rust-numpy` / PyO3 buffer handling (out-of-bounds reads/writes, buffer aliasing, use-after-free crossing the FFI edge)
- Robustness / denial-of-service — crafted input that triggers a panic-as-crash, an infinite loop, or unbounded allocation (adversarial array shapes, non-finite or extreme values, or inputs that stop the implied-volatility solver from terminating)
- Supply-chain concerns — dependency vulnerabilities not yet flagged by Dependabot or `cargo audit`, or a problem with the integrity or build provenance of the published PyPI wheels

## What is not in scope

- Numerical disagreement or precision differences against a reference implementation — these are correctness bugs, not vulnerabilities (open a regular bug report, or use the "py_vollib drift" template)
- Missing models or instruments not yet supported (open a feature request)
- Performance or throughput issues (open a regular issue)
- Feature requests
