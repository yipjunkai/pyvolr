# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | yes       |

Until pyvolr reaches 1.0, only the latest minor version receives security fixes.

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Use GitHub's private security advisory mechanism:

1. Go to <https://github.com/yipjunkai/pyvolr/security/advisories>
2. Click "Report a vulnerability"
3. Fill in the form with as much detail as you can share

Acknowledgement is targeted within 72 hours. There is no separate email contact at this stage; the GitHub advisory channel is the only supported route.

## What to report

- Numerical correctness bugs that could be exploited (e.g. crafted inputs causing infinite loops, panics that propagate as crashes, or values that wildly diverge from the reference implementation in a financially material way)
- Memory safety issues in the Rust extension surface (FFI boundary, numpy array handling)
- Supply chain concerns (dependency vulnerabilities not yet flagged by `cargo audit` / `pip-audit`)

## What is not in scope

- Correctness disagreements within the [published numerical tolerance](docs/numerical-stability.md) (open a regular issue)
- Performance issues (open a regular issue)
- Feature requests (open a regular issue)
