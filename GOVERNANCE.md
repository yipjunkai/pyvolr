# Governance

pyvolr is currently maintained by a single person. This document describes how it stays maintained even if that changes.

## Current status

Stage: solo maintainer, active development.

## Bus factor mitigation

The following commitments aim to keep the project bootstrappable even if the current maintainer disappears:

1. **All build and release infrastructure is in-repo**: every workflow, every config, every secret is in `.github/`. There is no out-of-band CI server. Anyone with repo access (or a fork) can ship a release.
2. **Releases are automated**: `release-please` opens release PRs from conventional commits. Merging a release PR triggers wheel builds and PyPI publication via Trusted Publishing. No human credentials are stored.
3. **No proprietary algorithms**: every numerical method is implemented from a public reference (paper, textbook). References are cited in source comments. Anyone can audit, fork, or replace.
4. **Documentation builds from source**: the docs site lives in `docs/`, builds with `mkdocs`, and is hosted on GitHub Pages. No external CMS.

## Becoming a co-maintainer

If you've contributed substantively (multiple merged PRs, sustained engagement over months) and want commit access, open a discussion. Co-maintainer status comes with:

- Triage and merge rights
- The ability to cut releases
- Listed credit in the README and on the project website

Expectations:

- Maintain conventional commits and the CHANGELOG discipline
- Respond to security advisories within a reasonable window
- Help review at least one PR per quarter when active

## Succession

If the current maintainer becomes inactive for 6+ months and no co-maintainer exists, any contributor with merged PRs in the project may fork and request the PyPI/crates.io name from the package indexes per their respective abandoned-project policies. The MIT/Apache-2.0 license is intended to make this transition friction-free.

This document will be updated as the project's governance evolves.
