# Governance

pyvolr is currently maintained by a single person. This document describes how it stays maintained even if that changes.

## Current status

Stage: solo maintainer, active development.

## Bus factor mitigation

The following commitments aim to keep the project bootstrappable even if the current maintainer disappears:

1. **All build and release infrastructure is in-repo**: every workflow and config is in `.github/`; there is no out-of-band CI server. Anyone with repo access can cut a release. A fork can rebuild identical wheels, but PyPI Trusted Publishing is bound to this repository, so publishing to the `pyvolr` name from a fork goes through the succession path below.
2. **Releases are automated**: `release-please` opens release PRs from conventional commits. Merging a release PR triggers wheel builds and PyPI publication via Trusted Publishing. PyPI publication uses no stored credentials (OIDC). The only stored credential is a GitHub App private key scoped to this repo with two narrow permissions (contents + PRs); the App is a machine identity that survives the maintainer leaving, and the tokens it mints at runtime expire in 1 hour.
3. **No proprietary algorithms**: every numerical method is implemented from a public reference (paper, textbook). References are cited in source comments. Anyone can audit, fork, or replace.
4. **Documentation lives with the code**: long-form rationale is in `docs/` (markdown); the README is the primary entry point. No external CMS or hosted docs site to maintain.

## Becoming a co-maintainer

If you've contributed substantively (multiple merged PRs, sustained engagement over months) and want commit access, open an issue. Co-maintainer status comes with:

- Triage and merge rights
- The ability to cut releases
- Listed credit in the README

Expectations:

- Maintain conventional commits and the CHANGELOG discipline
- Respond to security advisories within a reasonable window
- Help review at least one PR per quarter when active

## Succession

If the current maintainer becomes inactive for 6+ months and no co-maintainer exists, any contributor with merged PRs in the project may fork and request the PyPI name from the package index per its abandoned-project policy ([PEP 541](https://peps.python.org/pep-0541/)). The MIT/Apache-2.0 license is intended to make this transition friction-free.

This document will be updated as the project's governance evolves.
